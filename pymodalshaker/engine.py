# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Measurement engine: drives the continuous AI/AO stream block-by-block, estimates FRFs and
writes the result files. Qt-free so it can be tested headlessly; the GUI wraps it in a QThread.

Output schedule (one AO block written for every AI block read, `queue` blocks ahead):

    [decay zeros] [fade-in] [r1 × dwell] [r2 × dwell] ... [rR × dwell] [fade-out] [zeros]

AI block k is the response to AO block k (AO starts on the AI start trigger). In each
realization the last dwell block is the "converged" (steady-state periodic) block: it is the
one used for the FRF average and saved to TDMS.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from . import export
from .config import AppConfig, wants_folder
from .excitation import block_info, fade_window, generate_block
from .frf import FRFEstimator, FRFResult


@dataclass
class BlockLabel:
    kind: str                 # 'decay' | 'fadein' | 'excite' | 'fadeout' | 'tail' | 'pad'
    r: int = -1               # realization index
    d: int = -1               # dwell index


@dataclass
class Callbacks:
    """All optional. Called from the engine's thread."""
    log: Callable[[str], None] = lambda s: None
    block: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray], BlockLabel, float], None] = lambda *a: None
    progress: Callable[[int, int, str], None] = lambda *a: None
    realization: Callable[[int, FRFResult, np.ndarray], None] = lambda *a: None
    finished: Callable[[Dict], None] = lambda d: None
    control: Callable[[Dict], None] = lambda d: None      # stepped sine: controller events
    ask: Callable[[Dict], None] = lambda d: None          # stepped sine: control failed, ask the user


@dataclass
class RunSummary:
    completed_realizations: int = 0
    aborted: bool = False
    files: List[str] = field(default_factory=list)
    run_folder: str = ""
    fs: float = 0.0


class MeasurementEngine:
    def __init__(self, cfg: AppConfig, backend, callbacks: Optional[Callbacks] = None):
        self.cfg = cfg
        self.backend = backend
        self.cb = callbacks or Callbacks()
        self.stop_event = threading.Event()
        self.channels = cfg.enabled_channels()
        self.ref_i = cfg.reference_index()
        self.resp_i = [i for i in range(len(self.channels)) if i != self.ref_i]
        self.result: Optional[FRFResult] = None

    def request_stop(self) -> None:
        self.stop_event.set()

    # ------------------------------------------------------------------ schedule
    def _schedule(self, fs: float) -> List[BlockLabel]:
        o = self.cfg.output
        sched = [BlockLabel("decay")] * int(math.ceil(o.decay_time * fs / o.block_size))
        if o.fade:
            sched.append(BlockLabel("fadein", 0))
        for r in range(o.realizations):
            sched += [BlockLabel("excite", r, d) for d in range(o.dwell_blocks)]
        if o.fade:
            sched.append(BlockLabel("fadeout", o.realizations - 1))
        sched.append(BlockLabel("tail"))
        return sched

    def _block_for(self, lab: BlockLabel) -> np.ndarray:
        n = self.cfg.output.block_size
        if lab.kind in ("decay", "tail"):
            return np.zeros(n)
        x = self._realization_block(lab.r)
        if lab.kind == "fadein":
            return x * fade_window(n, True)
        if lab.kind == "fadeout":
            return x * fade_window(n, False)
        return x

    def _realization_block(self, r: int) -> np.ndarray:
        if r not in self._rblocks:
            # one independent, reproducible random stream per realization
            self._rblocks[r] = generate_block(self.cfg.output, np.random.default_rng([self.seed, r]))
            for old in [k for k in self._rblocks if k < r - 1]:
                del self._rblocks[old]
        return self._rblocks[r]

    # ------------------------------------------------------------------ files
    def _prepare_files(self, fs: float) -> None:
        s = self.cfg.save
        self.run_folder = ""
        self.tdms_conv = None
        self.trig_csv = None
        if not wants_folder(self.cfg):
            return
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_folder = os.path.join(s.folder, f"{s.basename}_{stamp}")
        k = 2
        while os.path.exists(self.run_folder):          # never mix two runs in one folder
            self.run_folder = os.path.join(s.folder, f"{s.basename}_{stamp}_{k}")
            k += 1
        os.makedirs(self.run_folder)
        with open(os.path.join(self.run_folder, "setup.json"), "w", encoding="utf-8") as f:
            from . import __version__
            import json as _json
            d = _json.loads(self.cfg.to_json())
            d["app_version"] = __version__
            f.write(_json.dumps(d, indent=2))
        self.summary.files.append(os.path.join(self.run_folder, "setup.json"))
        self.root_props = export.root_properties(self.cfg, fs)
        self.root_props["random_seed"] = getattr(self, "seed", -1)
        self.root_props.update(self._extra_root_props())
        from . import __version__
        self.root_props["app_version"] = __version__
        t = self.cfg.trigger
        if t.enabled and t.frame_sync:
            self.root_props.update({"frame_sync_terminal": t.physical, "trigger_mode": t.mode,
                                    "frame_sync_note": "camera frame f corresponds to AI sample index f"})
        elif t.enabled:
            self.root_props.update({"trigger_channel": t.physical, "trigger_mode": t.mode,
                                    "trigger_pulse_samples": t.pulse_samples, "trigger_polarity": t.polarity})
            p = os.path.join(self.run_folder, f"{s.basename}_trigger_log.csv")
            self.trig_csv = open(p, "w", encoding="utf-8")
            self.trig_csv.write("pulse,sample_index,time_s,block,block_type,realization,dwell_block\n")
            self.summary.files.append(p)
        if s.save_converged_tdms and self._tdms_name():
            p = os.path.join(self.run_folder, self._tdms_name())
            self.tdms_conv = export.TdmsBlockWriter(p, self.root_props)
            self.summary.files.append(p)

    def _tdms_name(self) -> str:
        return f"{self.cfg.save.basename}_converged_blocks.tdms"

    def _extra_root_props(self) -> Dict:
        return {}

    def _write_uff(self) -> None:
        s, o = self.cfg.save, self.cfg.output
        if not (s.save_uff and self.run_folder and self.result and self.result.n_avg > 0):
            return
        res = self.result
        sl = slice(1, None)            # always the full range Δf … fs/2 (DC dropped)
        f = res.freqs[sl]
        ref = self.channels[self.ref_i]
        resp = [self.channels[i] for i in self.resp_i]
        info = (f"{o.excitation}, {res.n_avg} avg, fs={self.summary.fs:g}Hz, N={o.block_size}, "
                f"dwell={o.dwell_blocks}, {o.window}")
        for func, flag, data in (("H1", s.uff_h1, res.H1), ("H2", s.uff_h2, res.H2),
                                 ("Hv", s.uff_hv, res.Hv), ("Coherence", s.uff_coherence, res.coherence)):
            if flag:
                p = os.path.join(self.run_folder, f"{s.basename}_{func}.uff")
                export.write_uff58(p, func, f, data[:, sl], resp, ref, info)
                self.summary.files.append(p)
                self.cb.log(f"Wrote {os.path.basename(p)} ({len(resp)} functions, {f.size} lines)")

    # ------------------------------------------------------------------ main loop
    def run(self) -> RunSummary:
        o = self.cfg.output
        self.summary = RunSummary()
        self.seed = o.seed if o.seed >= 0 else int(np.random.default_rng().integers(0, 2 ** 31 - 1))
        self._rblocks: Dict[int, np.ndarray] = {}
        be = self.backend
        # check every realization against the drive limit BEFORE touching the hardware
        for r in range(o.realizations):
            peak = float(np.max(np.abs(self._realization_block(r))))
            if peak > o.max_voltage:
                raise ValueError(f"Realization {r + 1}: peak drive {peak:.3f} V exceeds the {o.max_voltage:g} V "
                                 "limit. Lower the RMS or raise the limit.")
            self._rblocks.clear()
        try:
            fs, notes = be.configure()
            self.summary.fs = fs
            for n in notes:
                self.cb.log(n)
            info = block_info(o, generate_block(o, np.random.default_rng(0)))
            self.cb.log(f"Excitation: {o.excitation}, {o.rms:g} V RMS, crest factor ≈ {info.crest_factor:.2f}, "
                        f"seed {self.seed}"
                        + (f", {info.n_lines} lines {info.f_lo:g}–{info.f_hi:g} Hz" if info.n_lines else "")
                        + (f" ({info.note})" if info.note else ""))
            if o.window != "Rectangular" and o.excitation == "Burst random":
                self.cb.log(f"Note: {o.window} window on burst random. The burst already decays inside the "
                            "block; a window cuts into the burst and biases the FRF. Rectangular is recommended.")
            elif o.window != "Rectangular":
                self.cb.log(f"Note: {o.window} window on a periodic excitation. Rectangular is leakage-free "
                            "for converged periodic blocks; a window only widens the peaks here.")
            self._prepare_files(fs)
            est = FRFEstimator(len(self.resp_i), o.block_size, fs, o.window)

            sched = self._schedule(fs)
            total = len(sched)
            it = iter(sched)
            queued: deque = deque()          # (label, drive, trigger, fired) written, not yet read
            self._n_written = 0
            q = max(2, o.ao_queue_blocks)

            def write(lab: BlockLabel, x: np.ndarray) -> None:
                trig, fired = self._trigger_for(lab, self._n_written)
                be.write_block(x, trig)
                queued.append((lab, x, trig, fired))
                self._n_written += 1

            for _ in range(q):
                lab = next(it, None) or BlockLabel("tail")
                write(lab, self._block_for(lab))
            be.start()
            self.cb.log(f"Started: fs = {fs:g} Hz, block = {o.block_size} ({o.block_size / fs:.3f} s), "
                        f"df = {fs / o.block_size:.4g} Hz, {total} blocks ≈ {total * o.block_size / fs:.1f} s")

            aborting = False
            remaining_reads = total
            last_written: np.ndarray = queued[-1][1]
            prev_block = None
            real_buf: List[np.ndarray] = []
            real_drive: List[np.ndarray] = []
            real_trig: List[np.ndarray] = []
            conv = np.zeros(len(self.channels))
            k = 0
            while remaining_reads > 0:
                data = be.read_block()
                lab, drive, trig, fired = queued.popleft()
                remaining_reads -= 1
                k += 1

                # keep the AO queue full
                if self.stop_event.is_set() and not aborting:
                    aborting = True
                    self.summary.aborted = True
                    self.cb.log("Stop requested: fading out drive…")
                    fade = last_written * fade_window(o.block_size, False) if np.any(last_written) else None
                    it = iter(([BlockLabel("abort-fade")] if fade is not None else []) + [BlockLabel("tail")])
                    self._abort_fade = fade
                    remaining_reads = len(queued) + (2 if fade is not None else 1)
                if remaining_reads - len(queued) > 0:
                    nl = next(it, None) or BlockLabel("tail")
                    nx = self._abort_fade if nl.kind == "abort-fade" else self._block_for(nl)
                else:
                    nl, nx = BlockLabel("pad"), np.zeros(o.block_size)   # padding beyond the end
                write(nl, nx)
                last_written = nx

                # ---- process the block just read
                if fired:
                    self._log_trigger((k - 1) * o.block_size, k - 1, lab, fs)
                phase = self._phase_text(lab, fs, k, total)
                self.cb.block(data, drive, trig, lab, be.backlog())
                self.cb.progress(k, total, phase)
                if lab.kind == "excite":
                    if lab.d == 0:
                        real_buf, real_drive, real_trig, prev_block = [], [], [], None
                    if prev_block is not None:
                        diff = np.sqrt(np.mean((data - prev_block) ** 2, axis=1))
                        conv = diff / (np.sqrt(np.mean(data ** 2, axis=1)) + 1e-30)
                    prev_block = data.copy()
                    if self.cfg.save.save_each_realization:
                        real_buf.append(data.copy())
                        real_drive.append(drive.copy())
                        if trig is not None:
                            real_trig.append(trig.copy())
                    if lab.d == o.dwell_blocks - 1:
                        self._realization_done(lab.r, data, drive, trig, conv, est, fs,
                                               real_buf, real_drive, real_trig, (k - 1) * o.block_size)

            self.cb.log("Aborted by user." if self.summary.aborted else "Measurement complete.")
            self._write_uff()
        except Exception as e:
            self.summary.aborted = True
            self.cb.log(f"ERROR: {type(e).__name__}: {e}")
            try:
                self._write_uff()   # keep what was measured
            except Exception as e2:
                self.cb.log(f"Could not write UFF: {e2}")
            raise
        finally:
            try:
                be.stop()
            finally:
                be.close()
                if getattr(self, "tdms_conv", None):
                    self.tdms_conv.close()
                if getattr(self, "trig_csv", None):
                    self.trig_csv.close()
                self.summary.run_folder = getattr(self, "run_folder", "")
                self.cb.finished(self.summary.__dict__)
        return self.summary

    def _realization_done(self, r, data, drive, trig, conv, est, fs, real_buf, real_drive, real_trig,
                          start_index):
        o, s = self.cfg.output, self.cfg.save
        est.add(data[self.ref_i], data[self.resp_i])
        self.result = est.result()
        self.summary.completed_realizations = r + 1
        conv_txt = ", ".join(f"{c.name} {v * 100:.2g}%" for c, v in zip(self.channels, conv)) \
            if o.dwell_blocks > 1 else "n/a (dwell = 1)"
        self.cb.log(f"Realization {r + 1}/{o.realizations} done. Block-to-block change: {conv_txt}")
        gprops = {"realization": r + 1, "dwell_blocks": o.dwell_blocks,
                  "block_change_percent": ", ".join(f"{v * 100:.3g}" for v in conv),
                  "first_sample_index": int(start_index),
                  "first_sample_time_s": start_index / fs}
        extras = {"Drive": drive}
        if trig is not None:
            extras["Trigger"] = trig
        if self.tdms_conv is not None:
            self.tdms_conv.write_group(f"Realization_{r + 1:03d}", self.channels, data, extras, fs, gprops)
        if s.save_each_realization and self.run_folder:
            p = os.path.join(self.run_folder, f"{s.basename}_realization_{r + 1:03d}.tdms")
            first = start_index - (o.dwell_blocks - 1) * o.block_size
            rprops = {**gprops, "first_sample_index": int(first), "first_sample_time_s": first / fs}
            rextras = {"Drive": np.concatenate(real_drive)}
            if real_trig:
                rextras["Trigger"] = np.concatenate(real_trig)
            export.write_realization_tdms(p, self.root_props, f"Realization_{r + 1:03d}", self.channels,
                                          np.concatenate(real_buf, axis=1), rextras, fs, rprops)
            self.summary.files.append(p)
        self.cb.realization(r, self.result, conv)

    # ------------------------------------------------------------------ camera trigger
    def _trigger_for(self, lab: BlockLabel, block_no: int):
        """Trigger-line samples for one output block, and whether it contains a pulse.
        Levels are in volts for an AO trigger and 0/1 for a DO line."""
        t = self.cfg.trigger
        if not t.enabled or t.frame_sync:      # frame sync is a hardware clock route, no waveform
            return None, False
        d_last = self.cfg.output.dwell_blocks - 1
        if t.mode.startswith("Once"):
            fire = block_no == 0                                     # sample 0 = AI task start
        elif t.mode.startswith("Start of the converged"):
            fire = lab.kind == "excite" and lab.d == d_last
        else:
            fire = lab.kind == "excite" and lab.d == 0
        return self._trigger_wave(fire), fire

    def _trigger_wave(self, fire: bool) -> np.ndarray:
        """One block of the trigger line: idle level, with a pulse at sample 0 if `fire`."""
        t, n = self.cfg.trigger, self.cfg.output.block_size
        hi = t.level_v if t.kind == "AO" else 1.0
        active, idle = (0.0, hi) if t.polarity.startswith("Active low") else (hi, 0.0)
        y = np.full(n, idle)
        if fire:
            y[:max(1, min(t.pulse_samples, n - 1))] = active
        return y

    def _log_trigger(self, index: int, block: int, lab: BlockLabel, fs: float) -> None:
        self._n_trig = getattr(self, "_n_trig", 0) + 1
        r = lab.r + 1 if lab.kind == "excite" else ""
        d = lab.d + 1 if lab.kind == "excite" else ""
        self.cb.log(f"Trigger pulse #{self._n_trig} at sample {index} (t = {index / fs:.6f} s)"
                    + (f", realization {r} block {d}" if r != "" else ""))
        if getattr(self, "trig_csv", None):
            self.trig_csv.write(f"{self._n_trig},{index},{index / fs:.9f},{block + 1},{lab.kind},{r},{d}\n")
            self.trig_csv.flush()

    def _phase_text(self, lab: BlockLabel, fs: float, k: int, total: int) -> str:
        o = self.cfg.output
        if lab.kind == "decay":
            left = max(0.0, o.decay_time - k * o.block_size / fs)
            return f"IEPE settling (drive off) – {left:.1f} s left"
        if lab.kind == "fadein":
            return "Fading in drive"
        if lab.kind == "excite":
            return f"Realization {lab.r + 1}/{o.realizations} · dwell block {lab.d + 1}/{o.dwell_blocks}"
        if lab.kind in ("fadeout", "abort-fade"):
            return "Fading out drive"
        return "Finishing"
