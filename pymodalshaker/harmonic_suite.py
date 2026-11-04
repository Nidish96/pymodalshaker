# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Harmonic suite: a periodic multi-harmonic drive at one base frequency.

    v(t) = Σ_i a_i · cos(n_i·Ω·t + φ_i),   Ω = 2π·f0

The base frequency is snapped to an FFT line k·fs/N, so every block holds whole periods of every
harmonic: blocks join seamlessly and the single-line DFT at line n_i·k gives the complex
amplitude of harmonic n_i without leakage (cosine convention: x = Re{X·e^{jωt}}).

No control
    (n_i, a_i [V], φ_i [°]) is the drive. Schedule: [decay] [fade-in] [settle] [test × M] [fade-out].

Quasi-Newton control (Broyden)
    (n_i, a_i, φ_i) is the target for harmonic n_i of the control channel (a_i peak, in that
    channel's units). The unknowns are the complex drive harmonics V_i, handled as the real vector
    x = [Re V; Im V], and the residual is r(x) = [Re(Y − T); Im(Y − T)] with Y the measured
    harmonics of the control channel. Iteration = [settle × S] + [1 measure block] at one drive.

      first Jacobian   B₀ = diagonal: each harmonic's own gain Y_i/V_i (no cross-coupling);
                       harmonics whose target is 0 are probed with a small drive so a gain exists
      step             x⁺ = x − B⁻¹·r                                (quasi-Newton step)
      update           B ← B + (Δr − B·Δx)·Δxᵀ / (Δxᵀ·Δx)            (Broyden's "good" update)

    Cross-coupling between harmonics (e.g. a nonlinear shaker/structure turning drive at Ω into
    response at 2Ω) is learnt by the Broyden updates.
    Safeguards: each harmonic's drive may at most double per step (relative to the largest
    harmonic for small ones); the drive waveform's peak stays within the drive limit; if the error
    grows by more than 1.5× the step is halved back towards the previous drive.
    Error of harmonic i: e_i = |Y_i − T_i| / s_i with s_i = |T_i|, or max|T| for targets below
    10 % of the largest (e.g. harmonics to be suppressed). Converged when max e_i ≤ tolerance, and
    never on the first iteration (at least one refinement).

Detected failures (as in the stepped-sine suite): input overload before the target is reached,
drive limit reached, input saturation (the target needs more drive than overloads an input), and
max iterations. Failure actions: ask (drive held at a safe level), accept the best iteration
(flagged) and run the test, or stop.

After convergence the drive is held and `test_blocks` converged blocks are run, then the drive
fades out. Only time data is saved (TDMS): all blocks, only the converged test blocks, or one
test block every S seconds. A small control log (CSV) records every iteration.
"""

from __future__ import annotations

import math
import os
import queue
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from . import export
from .config import (HARM_FAIL_ACTIONS, HARM_SAVE_MODES, AppConfig, harmonic_line, harmonic_waveform)
from .engine import BlockLabel, Callbacks, MeasurementEngine, RunSummary
from .excitation import fade_window

HASK_CHOICES = ["Retry (another set of iterations)", "Accept best and run the test", "Stop the test"]
SMALL_TARGET = 0.1          # targets below this fraction of the largest are judged against the largest


@dataclass
class HLabel:
    kind: str                                  # decay fadein settle measure hold test pause fadeout tail pad
    V: Optional[np.ndarray] = None             # complex drive harmonics [V peak]
    seg: int = -1
    it: int = 0
    test_no: int = 0                           # 1-based number of the test block
    first_of_iter: bool = False
    first_test: bool = False


class HarmonicEngine(MeasurementEngine):
    def __init__(self, cfg: AppConfig, backend, callbacks: Optional[Callbacks] = None):
        super().__init__(cfg, backend, callbacks)
        self.answers: "queue.Queue[str]" = queue.Queue()
        self.h = cfg.harmonic
        self.closed_loop = self.h.closed_loop
        names = [c.name for c in self.channels]
        self.tgt_i = names.index(self.h.target_channel) if self.closed_loop else -1
        self.orders, self.values = self.h.complex_values()      # drive (no control) or target (control)
        self.k = harmonic_line(self.h, cfg.output)
        self.lines = np.array([self.k])                         # for the GUI worker (stepped-like API)
        self.freqs = np.array([self.h.f0])

    def answer(self, choice: str) -> None:
        self.answers.put(choice)

    # ================================================================== files
    def _tdms_name(self) -> str:
        return ""                                   # the base-class converged-block writer is not used

    def _extra_root_props(self) -> Dict:
        h = self.h
        props = {"suite": "Harmonic", "control": h.control, "base_line": int(self.k),
                 "orders": ",".join(str(int(n)) for n in self.orders),
                 "settle_blocks": h.settle_blocks, "test_blocks": h.test_blocks,
                 "save_mode": self.cfg.save.harm_save_mode}
        kind = "target" if self.closed_loop else "drive"
        for n, v in zip(self.orders, self.values):
            props[f"{kind}_h{n}_amplitude"] = float(abs(v))
            props[f"{kind}_h{n}_phase_deg"] = float(np.degrees(np.angle(v)))
        if self.closed_loop:
            props.update({"control_channel": h.target_channel, "tolerance_percent": h.tolerance_pct,
                          "max_iterations": h.max_iterations, "on_failure": h.on_failure,
                          "initial_drive_V": h.initial_drive})
        return props

    def _open_files(self, fs: float):
        self.tdms = None
        self.iter_csv = None
        if not self.run_folder:
            return
        s, b = self.cfg.save, self.cfg.save.basename
        if s.harm_save_tdms:
            p = os.path.join(self.run_folder, f"{b}_harmonic.tdms")
            self.root_props["f0_Hz"] = float(self.f0)
            self.tdms = export.TdmsBlockWriter(p, self.root_props)
            self.summary.files.append(p)
        if self.closed_loop:
            p = os.path.join(self.run_folder, f"{b}_control_iterations.csv")
            self.iter_csv = open(p, "w", encoding="utf-8")
            cols = ["iteration", "max_error_percent", "overload", "note"]
            for n in self.orders:
                cols += [f"drive_h{n}_amp_V", f"drive_h{n}_phase_deg", f"resp_h{n}_amp", f"resp_h{n}_phase_deg",
                         f"err_h{n}_percent"]
            self.iter_csv.write(",".join(cols) + "\n")
            self.summary.files.append(p)

    def _write_iter_csv(self, it, V, Y, e, err, over, note):
        if not self.iter_csv:
            return
        vals = [str(it), f"{err * 100:.4g}", str(int(over)), f'"{note}"' if "," in note else note]
        for i in range(len(self.orders)):
            vals += [f"{abs(V[i]):.6g}", f"{np.degrees(np.angle(V[i])):.3f}", f"{abs(Y[i]):.6g}",
                     f"{np.degrees(np.angle(Y[i])):.3f}", f"{e[i] * 100:.4g}"]
        self.iter_csv.write(",".join(vals) + "\n")
        self.iter_csv.flush()

    def _save_block(self, lab: HLabel, data, drive, trig, start: int, extra: Dict):
        """Write one block to TDMS if the save mode wants it."""
        if self.tdms is None:
            return
        mode = self.cfg.save.harm_save_mode
        if mode == HARM_SAVE_MODES[0]:
            pass
        elif lab.kind != "test":
            return
        elif mode == HARM_SAVE_MODES[2]:
            t = start / self.summary.fs
            if self.last_saved_t is not None and t < self.last_saved_t + self.cfg.save.harm_save_every_s - 1e-9:
                return
            self.last_saved_t = t
        fs = self.summary.fs
        ext = {"Drive": drive}
        if trig is not None:
            ext["Trigger"] = trig
        props = {"block_index": int(start // self.cfg.output.block_size), "kind": lab.kind,
                 "first_sample_index": int(start), "first_sample_time_s": start / fs,
                 "f0_Hz": float(self.f0), "base_line": int(self.k)}
        if lab.it:
            props["iteration"] = int(lab.it)
        if lab.kind == "test":
            props["test_block"] = int(lab.test_no)
            props["status"] = self.status
        if lab.V is not None:
            for n, v in zip(self.orders, lab.V):
                props[f"drive_h{n}_amp_V"] = float(abs(v))
                props[f"drive_h{n}_phase_deg"] = float(np.degrees(np.angle(v)))
        props.update(extra)
        self.n_saved += 1
        self.tdms.write_group(f"Block_{props['block_index'] + 1:06d}", self.channels, data, ext, fs, props)

    # ================================================================== waveform
    def _wave(self, V) -> np.ndarray:
        return harmonic_waveform(self.orders, V, self.k, self.cfg.output.block_size)

    def _peak(self, V) -> float:
        return float(np.max(np.abs(self._wave(V))))

    def _block_for(self, lab: HLabel) -> np.ndarray:
        n = self.cfg.output.block_size
        if lab.kind in ("decay", "tail", "pad") or lab.V is None:
            return np.zeros(n)
        x = self._wave(lab.V)
        if lab.kind == "fadein":
            return x * fade_window(n, True)
        if lab.kind == "fadeout":
            return x * fade_window(n, False)
        return x

    def _trigger(self, lab: HLabel, block_no: int):
        t = self.cfg.trigger
        if not t.enabled or t.frame_sync:
            return None, False
        if t.mode.startswith("Once"):
            fire = block_no == 0
        elif t.mode.startswith("Start of the converged"):
            fire = lab.first_test
        else:
            fire = lab.first_of_iter
        return self._trigger_wave(fire), fire

    def _harmonics(self, x: np.ndarray) -> np.ndarray:
        """Complex amplitudes of x at the listed orders (cosine convention, peak)."""
        n = x.size
        m = np.arange(n)
        return np.array([2.0 / n * x @ np.exp(-2j * np.pi * o * self.k * m / n) for o in self.orders])

    # ================================================================== scheduling
    def _queue(self, kinds: List[str], V, it: int = 0):
        self.seg_id += 1
        for i, kd in enumerate(kinds):
            lab = HLabel(kd, None if V is None else V.copy(), self.seg_id, it, first_of_iter=(i == 0 and it > 0))
            if kd == "test":
                self.n_test_queued += 1
                lab.test_no = self.n_test_queued
                lab.first_test = lab.test_no == 1
            self.to_write.append(lab)
        return self.seg_id

    def _queue_iteration(self, V):
        self.it += 1
        self.V = V.copy()
        self.pending = self._queue(["settle"] * self.h.settle_blocks + ["measure"], V, self.it)

    def _queue_test(self, V, status: str, settle: bool):
        self.V = V.copy()
        self.status = status
        kinds = ["settle"] * (self.h.settle_blocks if settle else 0) + ["test"] * self.h.test_blocks
        self._queue(kinds, V)
        self.after_segment = "end"
        self.cb.control({"type": "htest_start", "status": status, "V": V.copy(), "peak": self._peak(V),
                         "n": self.h.test_blocks})
        self.cb.log(f"Running {self.h.test_blocks} converged test block(s) ({status}), drive peak {self._peak(V):.4g} V")

    def _finish(self):
        if self.finishing:
            return
        self.finishing = True
        self.to_write.clear()
        self.pending = None
        if self.last_written_V is not None:
            self.to_write.append(HLabel("fadeout", self.last_written_V))
        self.to_write.append(HLabel("tail"))
        self.after_segment = "ended"
        self.final_count = self.n_written + len(self.to_write)

    def _next_label(self) -> HLabel:
        if not self.to_write and self.after_segment == "end":
            self._finish()
        if self.to_write:
            return self.to_write.popleft()
        if self.after_segment == "ended":
            return HLabel("pad")
        if self.waiting:
            return HLabel("pause", self.safe_V)
        return HLabel("hold", self.V)

    # ================================================================== controller
    def _errors(self, Y):
        T = self.values
        big = float(np.max(np.abs(T))) if T.size else 1.0
        scale = np.where(np.abs(T) >= SMALL_TARGET * big, np.abs(T), big)
        e = np.abs(Y - T) / np.maximum(scale, 1e-30)
        return e, float(np.max(e))

    def _overloads(self, data: np.ndarray) -> List[str]:
        lim = self.h.overload_pct / 100.0
        return [c.name for c, row in zip(self.channels, data) if np.max(np.abs(row)) >= lim * c.max_range]

    @staticmethod
    def _vec(z):
        return np.concatenate([z.real, z.imag])

    def _cplx(self, x):
        m = len(self.orders)
        return x[:m] + 1j * x[m:]

    def _initial_jacobian(self, V, Y):
        """Diagonal (per-harmonic gain) Jacobian of r(x) = [Re(Y−T); Im(Y−T)] w.r.t. x = [Re V; Im V]."""
        m = len(self.orders)
        g = np.full(m, np.nan + 0j)
        ok = np.abs(V) > 1e-12
        g[ok] = Y[ok] / V[ok]
        good = np.isfinite(g) & (np.abs(g) > 1e-12)
        fill = np.median(np.abs(g[good])) if good.any() else 1.0
        g[~good] = fill
        B = np.zeros((2 * m, 2 * m))
        for i in range(m):
            B[i, i], B[i, m + i] = g[i].real, -g[i].imag
            B[m + i, i], B[m + i, m + i] = g[i].imag, g[i].real
        return B

    def _limit_step(self, V, dV):
        """Scale the step so no harmonic's drive more than doubles and the peak stays within the limit.
        Returns (V_next, note, failure-reason or '')."""
        vmax = self.cfg.output.max_voltage
        big = float(np.max(np.abs(V))) if V.size else 0.0
        cap = 2.0 * np.maximum(np.abs(V), 0.1 * big)
        alpha, note = 1.0, ""
        for _ in range(40):
            if np.all(np.abs(V + alpha * dV) <= cap * (1 + 1e-9)):
                break
            alpha *= 0.8
            note = "step limited (≤ 2× per harmonic)"
        Vn = V + alpha * dV
        if self._peak(Vn) > vmax:
            if self._peak(V) >= 0.98 * vmax:
                return V, note, f"drive limit {vmax:g} V reached (peak {self._peak(V):.3g} V)"
            lo, hi = 0.0, alpha                    # bisection on the step length: peak = vmax
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if self._peak(V + mid * dV) > vmax:
                    hi = mid
                else:
                    lo = mid
            Vn, note = V + lo * dV, "clamped to drive limit"
        return Vn, note, ""

    def _on_measure(self, lab: HLabel, data, drive, trig, start: int):
        h = self.h
        tol = h.tolerance_pct / 100.0
        V = lab.V.copy()
        Y = self._harmonics(data[self.tgt_i])
        e, err = self._errors(Y)
        over = self._overloads(data)
        self.measured[lab.it] = (data.copy(), drive.copy(), None if trig is None else trig.copy(), start)
        note, fail, V_next = "", "", None
        T = self.values
        below = np.linalg.norm(Y) < np.linalg.norm(T) * (1 - tol)
        if over:
            note = "input overload: " + ", ".join(over)
            self.over_peak = min(self.over_peak, self._peak(V))
            self.over_chans.update(over)
            if below:
                fail = f"input overload ({', '.join(over)}) before the target was reached"
            else:
                V_next = 0.5 * V
                note += " — drive halved"
        else:
            if self.best is None or err < self.best[2]:
                self.best = (V.copy(), Y.copy(), err)
                self.best_it = lab.it
            converged = err <= tol and lab.it >= 2
            if converged:
                self._record_iter(lab, V, Y, e, err, over, "converged")
                self.cb.log(f"Control converged in {lab.it} iterations (max error {err * 100:.2f} %)")
                self._queue_test(V, "converged", settle=False)
                self._save_block(lab, data, drive, trig, start, self._meas_props(Y, e, err))
                return
            x, r = self._vec(V), self._vec(Y - T)
            if self.B is None:
                self.B = self._initial_jacobian(V, Y)
                note = "first Jacobian: per-harmonic gains"
            elif self.prev is not None:
                s_, y_ = x - self.prev[0], r - self.prev[1]
                ss = float(s_ @ s_)
                if ss > 0:
                    self.B += np.outer(y_ - self.B @ s_, s_) / ss
            if self.prev is not None and err > 1.5 * self.prev[2]:
                x_next = self.prev[0] + 0.5 * (x - self.prev[0])           # back off half-way
                dV = self._cplx(x_next) - V
                note = "error grew: step halved back"            # keep the better point as reference
            else:
                try:
                    dx = -np.linalg.lstsq(self.B, r, rcond=None)[0]
                except np.linalg.LinAlgError:
                    dx = np.zeros_like(x)
                dV = self._cplx(dx)
                self.prev = (x, r, err)
                if err <= tol:
                    note = "within tolerance: one refinement step"
            V_next, note2, fail = self._limit_step(V, dV)
            note = "; ".join(t for t in (note, note2) if t)
            if not fail and below and self._peak(V_next) >= 0.98 * self.over_peak:
                fail = (f"the target needs ≈{self._peak(V_next):.3g} V peak, but "
                        f"{', '.join(sorted(self.over_chans))} overloads from {self.over_peak:.3g} V (input saturation)")
        if not fail and lab.it >= max(2, h.max_iterations):
            fail = f"not converged in {h.max_iterations} iterations (max error {err * 100:.1f} %)"
        self._record_iter(lab, V, Y, e, err, over, note)
        self._save_block(lab, data, drive, trig, start, self._meas_props(Y, e, err))
        if fail:
            self._failure(fail)
        else:
            self._queue_iteration(V_next)

    def _meas_props(self, Y, e, err) -> Dict:
        if not self.closed_loop:
            return {}
        p = {"control_channel": self.h.target_channel, "max_error_percent": float(err * 100)}
        for n, y, ei in zip(self.orders, Y, e):
            p[f"resp_h{n}_amp"] = float(abs(y))
            p[f"resp_h{n}_phase_deg"] = float(np.degrees(np.angle(y)))
            p[f"err_h{n}_percent"] = float(ei * 100)
        return p

    def _record_iter(self, lab, V, Y, e, err, over, note):
        self.n_iter_total += 1
        self._write_iter_csv(lab.it, V, Y, e, err, bool(over), note)
        self.cb.control({"type": "hiter", "it": lab.it, "V": V.copy(), "Y": Y.copy(), "e": e.copy(), "err": err,
                         "overload": bool(over), "note": note, "peak": self._peak(V)})

    def _failure(self, reason: str):
        action = self.h.on_failure
        best = self.best
        self.cb.log(f"Control failed: {reason}")
        if action == HARM_FAIL_ACTIONS[0]:
            self.waiting = True
            self.fail_reason = reason
            if best is not None and self._peak(best[0]) <= self._peak(self.V):
                self.safe_V = best[0].copy()
            else:
                self.safe_V = 0.5 * self.V
            self.pending = None
            self.cb.control({"type": "hwait", "reason": reason})
            self.cb.ask({"f": float(self.f0), "reason": reason, "choices": HASK_CHOICES,
                         "best": None if best is None else {"amp": self._peak(best[0]), "err_pct": best[2] * 100}})
            return
        self._apply_choice(HASK_CHOICES[1] if action == HARM_FAIL_ACTIONS[1] else HASK_CHOICES[2], reason)

    def _apply_choice(self, choice: str, reason: str):
        self.waiting = False
        best = self.best
        if choice == HASK_CHOICES[0]:
            self.it, self.prev = 0, None
            self.over_peak, self.over_chans = float("inf"), set()
            self.cb.log("  → retrying")
            self._queue_iteration(best[0] if best else self.safe_V)
        elif choice == HASK_CHOICES[1] and best is not None:
            self.cb.log(f"  → accepting the best iteration (max error {best[2] * 100:.1f} %), flagged")
            self.fail_note = reason
            self._queue_test(best[0], "accepted (flagged)", settle=True)
        else:
            self.cb.log("  → stopping the test")
            self.summary.aborted = True
            self.status = "failed"
            self.cb.control({"type": "hdone", "status": "failed", "reason": reason})
            self._finish()

    # ================================================================== main loop
    def run(self) -> RunSummary:
        o, h = self.cfg.output, self.h
        self.summary = RunSummary()
        be = self.backend
        self.seg_id, self.it, self.n_written, self.n_test_queued, self.n_test_done = 0, 0, 0, 0, 0
        self.to_write: deque = deque()
        self.pending, self.after_segment, self.finishing, self.final_count = None, None, False, None
        self.waiting, self.safe_V, self.status, self.fail_note = False, None, "", ""
        self.B, self.prev, self.best, self.best_it = None, None, None, None
        self.over_peak, self.over_chans = float("inf"), set()
        self.measured, self.last_written_V, self.last_saved_t = {}, None, None
        self.n_saved, self.n_iter_total = 0, 0
        self.test_err: List[float] = []
        self.test_over: set = set()
        self.tdms = self.iter_csv = None
        try:
            fs, notes = be.configure()
            self.summary.fs = fs
            self.f0 = self.k * fs / o.block_size
            self.freqs = np.array([self.f0])
            for n_ in notes:
                self.cb.log(n_)
            self._prepare_files(fs)
            self._open_files(fs)
            unit = self.channels[self.tgt_i].units if self.closed_loop else "V"
            spec = ", ".join(f"h{n}: {abs(v):.4g}∠{np.degrees(np.angle(v)):.1f}°" for n, v in zip(self.orders, self.values))
            if self.closed_loop:
                mode = (f"Broyden control of {h.target_channel} to [{spec}] {unit} ±{h.tolerance_pct:g} %, "
                        f"≤ {h.max_iterations} iterations, on failure: {h.on_failure.lower()}")
            else:
                mode = f"no control, drive [{spec}] V (peak {self._peak(self.values):.4g} V)"
            if abs(self.f0 - h.f0) > 1e-9 * max(1.0, h.f0):
                self.cb.log(f"Base frequency {h.f0:g} Hz moved to the FFT line {self.f0:.6g} Hz "
                            f"(k = {self.k}, Δf = {fs / o.block_size:.4g} Hz)")
            self.cb.log(f"Harmonic: f0 = {self.f0:.6g} Hz (line {self.k}, {self.k} periods per block); {mode}; "
                        f"then {h.test_blocks} test block(s)")
            self.cb.control({"type": "hstart", "orders": self.orders.tolist(), "T": self.values.copy(),
                             "closed_loop": self.closed_loop, "unit": unit, "f0": self.f0,
                             "tolerance_pct": h.tolerance_pct, "max_iterations": h.max_iterations,
                             "channel": h.target_channel if self.closed_loop else "", "test_blocks": h.test_blocks})

            n_decay = int(math.ceil(o.decay_time * fs / o.block_size))
            for _ in range(n_decay):
                self.to_write.append(HLabel("decay"))
            if self.closed_loop:
                big = float(np.max(np.abs(self.values)))
                rel = np.abs(self.values) / big
                # probe every harmonic: targets of 0 still get 5 % of the initial drive (target phase)
                V0 = h.initial_drive * np.maximum(rel, 0.05) * np.exp(1j * np.angle(self.values))
                if self._peak(V0) > o.max_voltage:
                    V0 *= o.max_voltage / self._peak(V0)
            else:
                V0 = self.values.copy()
            self.V = V0.copy()
            if o.fade:
                self.to_write.append(HLabel("fadein", V0))
            if self.closed_loop:
                self._queue_iteration(V0)
            else:
                self._queue_test(V0, "open loop", settle=True)

            queued: deque = deque()
            q = max(2, o.ao_queue_blocks)

            def write_one():
                lab = self._next_label()
                x = self._block_for(lab)
                trig, fired = self._trigger(lab, self.n_written)
                be.write_block(x, trig)
                queued.append((lab, x, trig, fired))
                self.n_written += 1
                if lab.kind != "pad":
                    self.last_written_V = lab.V if lab.kind not in ("decay", "tail", "fadeout") else None

            for _ in range(q):
                write_one()
            be.start()
            self.cb.log(f"Started: fs = {fs:g} Hz, block = {o.block_size} ({o.block_size / fs:.3f} s)")

            kblk = 0
            while self.final_count is None or kblk < self.final_count:
                data = be.read_block()
                lab, drive, trig, fired = queued.popleft()
                kblk += 1
                start = (kblk - 1) * o.block_size
                bl = _as_block_label(lab, self._peak(lab.V) if lab.V is not None else 0.0)
                if fired:
                    self._log_trigger(start, kblk - 1, bl, fs)
                if self.stop_event.is_set() and not self.finishing:
                    self.summary.aborted = True
                    self.cb.log("Stop requested: fading out drive…")
                    self._finish()
                if self.waiting and not self.finishing:
                    try:
                        self._apply_choice(self.answers.get_nowait(), self.fail_reason)
                    except queue.Empty:
                        pass
                if lab.kind == "measure" and lab.seg == self.pending and not self.finishing:
                    self.pending = None
                    self._on_measure(lab, data, drive, trig, start)
                elif lab.kind == "test":
                    self._on_test(lab, data, drive, trig, start)
                else:
                    self._save_block(lab, data, drive, trig, start, {})
                self.cb.block(data, drive, trig, bl, be.backlog())
                self.cb.progress(self.n_test_done, h.test_blocks, self._phase(lab))
                write_one()

            self.summary.completed_realizations = self.n_test_done
            if self.test_err:
                self.cb.log(f"Test blocks: {self.n_test_done}; control-channel error max {max(self.test_err) * 100:.2f} %, "
                            f"mean {np.mean(self.test_err) * 100:.2f} %")
            if self.test_over:
                self.cb.log("WARNING: input overload during the test on " + ", ".join(sorted(self.test_over)))
            if self.tdms is not None:
                self.cb.log(f"Wrote {self.n_saved} block(s) to {self.cfg.save.basename}_harmonic.tdms "
                            f"({self.cfg.save.harm_save_mode.replace('S seconds', f'{self.cfg.save.harm_save_every_s:g} s')})")
            if not self.summary.aborted or self.n_test_done:
                self.cb.control({"type": "hdone", "status": self.status or "stopped", "n": self.n_test_done,
                                 "err_max": max(self.test_err) if self.test_err else None})
            self.cb.log("Aborted." if self.summary.aborted else "Harmonic test complete.")
        except Exception as e:
            self.summary.aborted = True
            self.cb.log(f"ERROR: {type(e).__name__}: {e}")
            raise
        finally:
            try:
                be.stop()
            finally:
                be.close()
                for fh in (self.tdms, getattr(self, "trig_csv", None), self.iter_csv):
                    if fh:
                        fh.close()
                self.summary.run_folder = getattr(self, "run_folder", "")
                self.cb.finished(self.summary.__dict__)
        return self.summary

    def _on_test(self, lab: HLabel, data, drive, trig, start: int):
        self.n_test_done += 1
        over = self._overloads(data)
        self.test_over.update(over)
        extra = {}
        ev = {"type": "htest", "n": lab.test_no, "total": self.h.test_blocks, "overload": over}
        if self.closed_loop:
            Y = self._harmonics(data[self.tgt_i])
            e, err = self._errors(Y)
            self.test_err.append(err)
            extra = self._meas_props(Y, e, err)
            ev.update({"Y": Y, "e": e, "err": err})
        self._save_block(lab, data, drive, trig, start, extra)
        self.cb.control(ev)

    def _phase(self, lab: HLabel) -> str:
        h = self.h
        if lab.kind == "decay":
            return "IEPE settling (drive off)"
        if lab.kind in ("fadeout", "tail", "pad"):
            return "Finishing"
        if lab.kind == "fadein":
            return "Fading in drive"
        head = f"Harmonic · f0 = {self.f0:.4g} Hz · "
        if self.waiting:
            return head + "control FAILED — waiting for your decision (drive held at a safe level)"
        if lab.kind in ("measure", "hold") or (lab.kind == "settle" and lab.it):
            return head + f"control iteration {lab.it}/{h.max_iterations}"
        if lab.kind == "settle":
            return head + "settling"
        if lab.kind == "test":
            return head + f"test block {lab.test_no}/{h.test_blocks} ({self.status})"
        return head + lab.kind


def _as_block_label(lab: HLabel, peak: float) -> BlockLabel:
    """Adapter so the shared GUI/trigger code sees something BlockLabel-like."""
    kind = {"test": "excite"}.get(lab.kind, lab.kind)
    bl = BlockLabel(kind, 0, lab.test_no - 1 if lab.kind == "test" else -1)
    bl.step_kind = lab.kind
    bl.amp = peak
    bl.it = lab.it
    bl.test_no = lab.test_no
    return bl
