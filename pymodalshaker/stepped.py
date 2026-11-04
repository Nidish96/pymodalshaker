# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Stepped-sine measurement with secant amplitude control (or constant drive).

Streaming model (same continuous AI/AO tasks as the random suite): one AO block is written for
every AI block read, `queue` blocks ahead. Every frequency is an FFT line k·fs/N, so each block
holds whole cycles: blocks join seamlessly and a single-line DFT of a block is leakage-free.

Per frequency
-------------
  control (secant):  iteration = [settle × block] + [1 measure block], all at one drive amplitude.
                     Because of the AO queue, the measure block is read `queue` blocks after it was
                     written; meanwhile the same amplitude is held ('hold' blocks, not analysed).
                     y = RMS of the target channel in the measure block. Converged when
                     |y − target| ≤ tolerance · target, then the dwell starts at that amplitude.
                     Every frequency gets at least one refinement: the first iteration (drive
                     carried over from the previous frequency) never counts as converged, so the
                     drive is always corrected from a measurement at this frequency.
  no control:        [settle blocks] + [dwell blocks] at the fixed drive amplitude.
  dwell:             ceil(dwell_time·fs/N) blocks → single-line spectra of every channel →
                     H1/H2/Hv/γ² at that line, response amplitudes, THD.

Secant rule on f(A) = y(A) − target, using the last two valid (A, y) points; the first update
uses the origin (0, 0) as the second point (proportional step). Every update is limited to
½ … 2 × the current amplitude and to the drive limit.

Detected conditions
-------------------
  input overload      any channel |peak| ≥ overload % of its range (measurement not used; if it
                      happens below the target it is a failure, above the target the drive is halved)
  drive limit         the drive is at its limit and the target is still not reached
  max iterations      not converged within the allowed iterations
  non-monotonic       response did not grow with drive: proportional step used instead of secant
Failure actions: ask the user (drive held at a safe level meanwhile), accept the best iteration
(flagged), skip the frequency, or stop the test.
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
from .config import CONTROL_MODES, FAIL_ACTIONS, AppConfig, stepped_frequencies
from .engine import Callbacks, MeasurementEngine, RunSummary
from .excitation import fade_window
from .frf import FRFResult, estimates, harmonic_frf


ASK_CHOICES = ["Retry (another set of iterations)", "Accept best and continue", "Skip this frequency",
               "Stop the test"]


@dataclass
class SLabel:
    kind: str            # decay fadein settle measure hold dwell pause fadeout tail pad
    j: int = -1          # frequency index
    amp: float = 0.0     # drive amplitude [V peak]
    seg: int = -1        # segment id
    it: int = 0          # control iteration (1-based) for settle/measure
    first_of_freq: bool = False
    first_dwell: bool = False


class SteppedSineEngine(MeasurementEngine):
    def __init__(self, cfg: AppConfig, backend, callbacks: Optional[Callbacks] = None):
        super().__init__(cfg, backend, callbacks)
        self.answers: "queue.Queue[str]" = queue.Queue()
        self.st = cfg.stepped
        self.closed_loop = self.st.control == CONTROL_MODES[0]
        names = [c.name for c in self.channels]
        self.tgt_i = names.index(self.st.target_channel) if self.closed_loop else -1

    def answer(self, choice: str) -> None:
        """Reply to a `ask` callback (thread-safe). choice: one of ASK_CHOICES."""
        self.answers.put(choice)

    def _tdms_name(self) -> str:
        """Converged block of every frequency (written by the base class's converged-block writer)."""
        return f"{self.cfg.save.basename}_stepped_converged_blocks.tdms"

    def _dwell_name(self) -> str:
        return f"{self.cfg.save.basename}_stepped_sine.tdms"

    def _extra_root_props(self) -> Dict:
        st = self.st
        props = {"suite": "Stepped sine", "control": st.control, "settle_blocks": st.settle_blocks,
                 "dwell_blocks_per_step": self.dwell_blocks, "f_start_Hz": st.f_start, "f_stop_Hz": st.f_stop,
                 "f_step": st.f_step, "spacing": st.spacing}
        if self.closed_loop:
            props.update({"target_channel": st.target_channel, "target_value": st.target_value,
                          "target_kind": st.target_kind,
                          "tolerance_percent": st.tolerance_pct, "max_iterations": st.max_iterations,
                          "on_failure": st.on_failure})
        else:
            props["drive_amplitude_V"] = st.drive_amplitude
        return props

    # ================================================================== block generation
    def _sine(self, j: int, amp: float) -> np.ndarray:
        n = self.cfg.output.block_size
        return amp * np.sin(2 * np.pi * self.lines[j] * np.arange(n) / n)

    def _block_for(self, lab: SLabel) -> np.ndarray:
        n = self.cfg.output.block_size
        if lab.kind in ("decay", "tail", "pad"):
            return np.zeros(n)
        x = self._sine(lab.j, lab.amp)
        if lab.kind == "fadein":
            return x * fade_window(n, True)
        if lab.kind == "fadeout":
            return x * fade_window(n, False)
        return x

    def _trigger(self, lab: SLabel, block_no: int):
        t = self.cfg.trigger
        if not t.enabled or t.frame_sync:
            return None, False
        if t.mode.startswith("Once"):
            fire = block_no == 0
        elif t.mode.startswith("Start of the converged"):
            fire = lab.first_dwell                     # start of the analysed (steady) dwell
        else:
            fire = lab.first_of_freq                   # start of each frequency step
        return self._trigger_wave(fire), fire

    # ================================================================== segment planning
    def _queue_segment(self, kinds: List[str], j: int, amp: float, it: int = 0) -> int:
        self.seg_id += 1
        first = j != self.last_j
        for i, k in enumerate(kinds):
            self.to_write.append(SLabel(k, j, amp, self.seg_id, it,
                                        first_of_freq=first and i == 0,
                                        first_dwell=(k == "dwell" and (i == 0 or kinds[i - 1] != "dwell"))))
        self.last_j = j
        return self.seg_id

    def _queue_iteration(self, amp: float):
        self.it += 1
        kinds = ["settle"] * self.st.settle_blocks + ["measure"]
        self.pending = self._queue_segment(kinds, self.j, amp, self.it)
        self.amp = amp

    def _queue_dwell(self, amp: float, status: str, settle: bool = True):
        """settle=False only when the drive is already at `amp` and steady (converged control)."""
        kinds = ["settle"] * (self.st.settle_blocks if settle else 0) + ["dwell"] * self.dwell_blocks
        self.step_status[self.j] = status
        self.step_iters[self.j] = len(self.iter_rows_j)
        self.amp = amp
        self._queue_segment(kinds, self.j, amp)
        self.after_segment = "next_freq"

    def _start_freq(self, j: int, amp: float):
        self.j, self.it, self.hist, self.best = j, 0, [], None
        self.measured, self.best_it = {}, None
        self.over_amp, self.over_chans = float("inf"), set()
        self.iter_rows_j = []
        amp = min(amp, self.cfg.output.max_voltage)
        self.cb.control({"type": "freq", "j": j, "f": float(self.freqs[j]), "n": len(self.freqs),
                         "amp": amp, "target": self.st.target_value if self.closed_loop else None})
        if self.closed_loop:
            self._queue_iteration(amp)
        else:
            self._queue_dwell(amp, "open loop")

    def _advance(self):
        """After a frequency's dwell (or a skip): next frequency or the end of the test."""
        nxt = self.j + 1
        if nxt >= len(self.freqs) or self.stop_event.is_set():
            self._finish()
        else:
            start = self.amp if self.closed_loop else self.st.drive_amplitude
            self._start_freq(nxt, start)

    def _finish(self):
        if self.finishing:
            return
        self.finishing = True
        self.to_write.clear()
        self.pending = None
        if self.last_written_amp > 0:
            self.to_write.append(SLabel("fadeout", self.last_written_j, self.last_written_amp))
        self.to_write.append(SLabel("tail"))
        self.after_segment = "end"

    def _next_label(self) -> SLabel:
        if not self.to_write:
            if self.after_segment == "next_freq":
                self.after_segment = None
                self._advance()
            elif self.after_segment == "end":
                self.after_segment = "ended"
                self.final_count = self.n_written
        if self.to_write:
            return self.to_write.popleft()
        if self.after_segment == "ended":
            return SLabel("pad")
        if self.waiting:                                   # waiting for the user's decision
            return SLabel("pause", self.j, self.safe_amp)
        return SLabel("hold", self.j, self.amp)            # waiting for a measurement to come back

    # ================================================================== controller
    def _y(self, data: np.ndarray) -> float:
        """Controlled quantity: block RMS of the control channel (× √2 for the amplitude target)."""
        x = data[self.tgt_i]
        return float(np.sqrt(np.mean((x - x.mean()) ** 2))) * self.st.target_scale

    def _overloads(self, data: np.ndarray) -> List[str]:
        lim = self.st.overload_pct / 100.0
        return [c.name for c, row in zip(self.channels, data) if np.max(np.abs(row)) >= lim * c.max_range]

    def _on_measure(self, lab: SLabel, data: np.ndarray, drive=None, trig=None, start: int = 0):
        st, vmax = self.st, self.cfg.output.max_voltage
        yt, tol = st.target_value, st.tolerance_pct / 100.0
        A, y = lab.amp, self._y(data)
        over = self._overloads(data)
        err = (y - yt) / yt
        valid = not over
        self.measured[lab.it] = (data.copy(), None if drive is None else drive.copy(),
                                 None if trig is None else trig.copy(), start, A, y, err)
        if valid:
            self.hist.append((A, y))
            if self.best is None or abs(err) < abs(self.best[2]):
                self.best = (A, y, err)
                self.best_it = lab.it
        note = ""
        decision = None
        if over:
            note = "input overload: " + ", ".join(over)
            self.over_amp = min(self.over_amp, A)
            self.over_chans.update(over)
            if y < yt * (1 - tol):
                decision = ("fail", f"input overload ({', '.join(over)}) before the target was reached")
            else:
                A_next = A * 0.5
        elif abs(err) <= tol and lab.it >= 2:
            decision = ("converged", "")
        elif A >= vmax * (1 - 1e-9) and y < yt * (1 - tol):
            decision = ("fail", f"drive limit {vmax:g} V reached at {y:.4g} (target {yt:.4g})")
        if decision is None and not over:
            A_next, note2 = self._secant(A, y, yt)
            note = note or note2
            if abs(err) <= tol:                            # first try already in tolerance: refine once
                note = 'within tolerance: one refinement step'
            if A_next >= 0.98 * self.over_amp and y < yt * (1 - tol):
                decision = ("fail", f"the target needs ≈{A_next:.3g} V, but {', '.join(sorted(self.over_chans))} "
                                    f"overloads from {self.over_amp:.3g} V (input saturation)")
        if decision is None and lab.it >= max(2, st.max_iterations):
            decision = ("fail", f"not converged in {st.max_iterations} iterations (error {err * 100:+.1f} %)")

        row = {"type": "iteration", "j": lab.j, "f": float(self.freqs[lab.j]), "it": lab.it, "amp": A,
               "y": y, "target": yt, "err_pct": err * 100, "overload": bool(over), "note": note}
        self.iter_rows_j.append(row)
        self._write_iter_csv(row)
        self.cb.control(row)

        if decision and decision[0] == "converged":
            self._write_converged(lab.j, lab.it, "converged")
            self._queue_dwell(A, "converged", settle=False)
        elif decision and decision[0] == "fail":
            self._failure(decision[1])
        else:
            self._queue_iteration(min(max(A_next, 1e-6), vmax))

    def _secant(self, A1: float, y1: float, yt: float):
        """Next amplitude from the secant rule, limited to ½…2× the current drive."""
        A0, y0 = self.hist[-2] if len(self.hist) >= 2 else (0.0, 0.0)
        note = ""
        slope = (y1 - y0) / (A1 - A0) if A1 != A0 else 0.0
        if slope <= 0 or not np.isfinite(slope):
            A_new = A1 * yt / max(y1, 1e-30)               # proportional fallback
            note = "non-monotonic response: proportional step"
        else:
            A_new = A1 - (y1 - yt) / slope
        ratio = A_new / A1 if A1 > 0 else 2.0
        if not np.isfinite(ratio) or ratio > 2.0:
            A_new, note = 2.0 * A1, note or "step limited to 2×"
        elif ratio < 0.5:
            A_new, note = 0.5 * A1, note or "step limited to ½×"
        if A_new > self.cfg.output.max_voltage:
            A_new, note = self.cfg.output.max_voltage, "clamped to drive limit"
        return A_new, note

    def _failure(self, reason: str):
        action = self.st.on_failure
        best = self.best
        self.cb.log(f"Control failed at {self.freqs[self.j]:g} Hz: {reason}")
        if action == FAIL_ACTIONS[0]:          # ask
            self.waiting = True
            self.fail_reason = reason
            self.safe_amp = min(best[0], self.amp) if best else self.amp * 0.5
            self.pending = None
            self.cb.control({"type": "wait", "j": self.j, "f": float(self.freqs[self.j]), "reason": reason})
            self.cb.ask({"f": float(self.freqs[self.j]), "reason": reason, "choices": ASK_CHOICES,
                         "best": None if best is None else {"amp": best[0], "y": best[1], "err_pct": best[2] * 100}})
            return
        self._apply_choice({FAIL_ACTIONS[1]: ASK_CHOICES[1], FAIL_ACTIONS[2]: ASK_CHOICES[2],
                            FAIL_ACTIONS[3]: ASK_CHOICES[3]}[action], reason)

    def _apply_choice(self, choice: str, reason: str):
        self.waiting = False
        best = self.best
        if choice == ASK_CHOICES[0]:                       # retry
            self.it, self.hist = 0, ([best[:2]] if best else [])
            self.over_amp, self.over_chans = float("inf"), set()
            self.cb.log(f"  → retrying at {self.freqs[self.j]:g} Hz")
            self._queue_iteration(best[0] if best else self.safe_amp)
        elif choice == ASK_CHOICES[1] and best is not None:
            self.cb.log(f"  → accepting best iteration ({best[2] * 100:+.1f} %) at {self.freqs[self.j]:g} Hz")
            self._write_converged(self.j, self.best_it, "accepted (flagged)")
            self._queue_dwell(best[0], "accepted (flagged)")
            self.step_reason[self.j] = reason
        elif choice == ASK_CHOICES[3]:
            self.cb.log("  → stopping the test")
            self.summary.aborted = True
            self.step_status[self.j] = "failed"
            self._record_step(self.j, None, reason)
            self._finish()
        else:                                              # skip (or accept with nothing valid)
            self.cb.log(f"  → skipping {self.freqs[self.j]:g} Hz")
            self.step_status[self.j] = "skipped"
            self._record_step(self.j, None, reason)
            if best:
                self.amp = best[0]
            self._advance()

    # ================================================================== dwell analysis
    def _on_dwell(self, lab: SLabel, data: np.ndarray, drive: np.ndarray, trig, start_index: int):
        acc = self.dwell_acc.setdefault(lab.j, {"blocks": [], "drive": [], "trig": [], "start": start_index,
                                                "over": set(), "amp": lab.amp})
        if not self.closed_loop and not acc["blocks"]:
            self.measured = {0: (data.copy(), drive.copy(), None if trig is None else trig.copy(), start_index,
                                 lab.amp, None, None)}
            self._write_converged(lab.j, 0, "open loop")
        acc["blocks"].append(data.copy())
        acc["drive"].append(drive.copy())
        if trig is not None:
            acc["trig"].append(trig.copy())
        acc["over"].update(self._overloads(data))
        if len(acc["blocks"]) == self.dwell_blocks:
            self._finish_freq(lab.j, acc)
            del self.dwell_acc[lab.j]

    def _line_spectra(self, blocks: np.ndarray, k: int) -> np.ndarray:
        """Complex amplitude of line k for every block and channel: (n_blocks, n_ch)."""
        n = blocks.shape[-1]
        e = np.exp(-2j * np.pi * k * np.arange(n) / n)
        return 2.0 / n * blocks @ e

    def _finish_freq(self, j: int, acc: Dict):
        k = int(self.lines[j])
        blocks = np.stack(acc["blocks"])                               # (nb, nch, N)
        X = self._line_spectra(blocks, k)                              # (nb, nch)
        xr, yr = X[:, self.ref_i], X[:, self.resp_i]
        Gxx = np.mean(np.abs(xr) ** 2)
        Gyy = np.mean(np.abs(yr) ** 2, axis=0)
        Gxy = np.mean(np.conj(xr)[:, None] * yr, axis=0)
        H1, H2, Hv, _ = estimates(np.array([Gxx]), Gyy[:, None], Gxy[:, None])
        # Fundamental fraction η (replaces coherence for sine testing): share of each channel's AC
        # power at the excitation line, averaged over the dwell blocks. η_FRF = η_ref · η_resp.
        ac = blocks - blocks.mean(axis=-1, keepdims=True)
        p_tot = np.mean(ac ** 2, axis=-1)                              # (nb, nch)
        p_fund = np.abs(X) ** 2 / 2.0
        eta = np.clip(np.mean(p_fund / (p_tot + 1e-300), axis=0), 0.0, 1.0)   # (nch,)
        eta_frf = eta[self.ref_i] * eta[self.resp_i]
        self.frf_f.append(float(self.freqs[j]))
        for name, v in (("H1", H1[:, 0]), ("H2", H2[:, 0]), ("Hv", Hv[:, 0]), ("coh", eta_frf)):
            self.frf_cols[name].append(v)
        # harmonics 1…n of every channel, complex, averaged over the dwell blocks
        nyq_lines = blocks.shape[-1] // 2
        n_h = max(2, self.st.n_harmonics)
        harm = np.full((blocks.shape[1], n_h), np.nan + 0j)
        for h in range(1, n_h + 1):
            if h * k < nyq_lines:
                harm[:, h - 1] = self._line_spectra(blocks, h * k).mean(axis=0)
        self.harm_rows.append(harm)
        last = blocks[-1]
        spec = np.abs(np.fft.rfft(last, axis=-1))
        harm = [h * k for h in range(2, n_h + 1) if h * k < spec.shape[-1]]
        thd = np.sqrt(np.sum(spec[:, harm] ** 2, axis=-1)) / (spec[:, k] + 1e-30) if harm else np.zeros(len(self.channels))
        y = self._y(last) if self.closed_loop else None
        info = {"amp": acc["amp"], "y": y, "thd": thd, "over": sorted(acc["over"]), "eta": eta,
                "resp_amp": np.abs(X).mean(axis=0) / np.sqrt(2), "start": acc["start"]}
        self._record_step(j, info, self.step_reason.get(j, ""))
        # TDMS: all dwell blocks of this frequency
        if self.tdms_dwell is not None:
            data = np.concatenate(list(blocks), axis=1)
            extras = {"Drive": np.concatenate(acc["drive"])}
            if acc["trig"]:
                extras["Trigger"] = np.concatenate(acc["trig"])
            fs = self.summary.fs
            row = self.step_rows[-1]
            props = {k_: (float(v) if isinstance(v, (float, np.floating)) else v)
                     for k_, v in row.items() if v is not None and k_ not in ("type", "j")}
            props["step"] = j + 1
            props.update({"first_sample_index": int(acc["start"]), "first_sample_time_s": acc["start"] / fs,
                          "line": k, "dwell_blocks": self.dwell_blocks})
            self.tdms_dwell.write_group(f"Step_{j + 1:03d}", self.channels, data, extras, fs, props)
        self.result = self._current_result()
        self.summary.completed_realizations = len(self.frf_f)
        self.cb.realization(j, self.result, np.zeros(len(self.channels)))

    def _write_converged(self, j: int, it, status: str):
        """Time waveform of the block at which control converged (or the best iteration's block for
        'accept best', or the first steady block without control) → <base>_stepped_converged_blocks.tdms."""
        if self.tdms_conv is None or it not in self.measured:
            return
        data, drive, trig, start, amp, y, err = self.measured[it]
        fs = self.summary.fs
        extras = {"Drive": drive if drive is not None else self._sine(j, amp)}
        if trig is not None:
            extras["Trigger"] = trig
        st = self.st
        props = {"step": j + 1, "frequency_Hz": float(self.freqs[j]), "line": int(self.lines[j]),
                 "status": status, "iteration": int(it), "drive_amplitude_V": float(amp),
                 "first_sample_index": int(start), "first_sample_time_s": start / fs,
                 "block": "converged control block" if self.closed_loop else "first block after settling"}
        if self.closed_loop and y is not None:
            props.update({"target_kind": st.target_label, "target": float(st.target_value),
                          "achieved": float(y), "error_percent": float(err * 100)})
        self.tdms_conv.write_group(f"Step_{j + 1:03d}", self.channels, data, extras, fs, props)

    def _current_result(self) -> FRFResult:
        """FRFs so far. For stepped sine the `coherence` field holds the fundamental fraction η."""
        f = np.array(self.frf_f)
        cols = {k: np.array(v).T if v else np.zeros((len(self.resp_i), 0)) for k, v in self.frf_cols.items()}
        harm = np.array(self.harm_rows) if self.harm_rows else None
        return FRFResult(f, cols["H1"], cols["H2"], cols["Hv"], cols["coh"].real, len(f),
                         harm=harm, ref_i=self.ref_i, resp_i=list(self.resp_i))

    def _record_step(self, j: int, info: Optional[Dict], reason: str):
        st = self.st
        status = self.step_status.get(j, "")
        its = self.step_iters.get(j, len(self.iter_rows_j)) if self.closed_loop else 0
        row = {"type": "step", "j": j, "frequency_Hz": float(self.freqs[j]), "status": status,
               "iterations": its, "reason": reason}
        if info is not None:
            tgt = st.target_value if self.closed_loop else None
            row.update({"drive_amplitude_V": info["amp"], "target": tgt, "achieved": info["y"],
                        "target_kind": st.target_label if self.closed_loop else None,
                        "error_percent": (info["y"] - tgt) / tgt * 100 if tgt else None,
                        "thd_target_percent": float(info["thd"][self.tgt_i] * 100) if self.closed_loop else None,
                        "thd_reference_percent": float(info["thd"][self.ref_i] * 100),
                        "overload": ", ".join(info["over"]),
                        "fundamental_fraction_min": float(info["eta"].min()),
                        "reference_amplitude_rms": float(info["resp_amp"][self.ref_i])})
            if info["over"] and "overload" not in status:
                row["status"] = status + " · overload in dwell"
        self.step_rows.append(row)
        self._write_step_csv(row)
        self.cb.control(row)

    # ================================================================== files
    def _open_logs(self):
        self.iter_csv = self.step_csv = None
        if not self.run_folder:
            return
        b = self.cfg.save.basename
        p1 = os.path.join(self.run_folder, f"{b}_control_summary.csv")
        p2 = os.path.join(self.run_folder, f"{b}_control_iterations.csv")
        self.step_csv = open(p1, "w", encoding="utf-8")
        self.step_csv.write("step,frequency_Hz,status,iterations,drive_amplitude_V,target_kind,target,achieved,"
                            "error_percent,thd_target_percent,thd_reference_percent,fundamental_fraction,"
                            "overload,reason\n")
        self.summary.files.append(p1)
        if self.closed_loop:
            self.iter_csv = open(p2, "w", encoding="utf-8")
            self.iter_csv.write(f"frequency_Hz,iteration,drive_amplitude_V,measured_{self.st.target_label},"
                                f"target_{self.st.target_label},error_percent,overload,note\n")
            self.summary.files.append(p2)

    def _write_step_csv(self, r: Dict):
        if self.step_csv:
            def f(v):
                if v is None:
                    return ""
                v = f"{v:.6g}" if isinstance(v, float) else str(v)
                return f'"{v}"' if "," in v else v
            cols = [str(r["j"] + 1)] + [f(r.get(k)) for k in (
                "frequency_Hz", "status", "iterations", "drive_amplitude_V", "target_kind", "target", "achieved",
                "error_percent", "thd_target_percent", "thd_reference_percent", "fundamental_fraction_min",
                "overload", "reason")]
            self.step_csv.write(",".join(cols) + "\n")
            self.step_csv.flush()

    def _write_iter_csv(self, r: Dict):
        if self.iter_csv:
            self.iter_csv.write(f"{r['f']:.6g},{r['it']},{r['amp']:.6g},{r['y']:.6g},{r['target']:.6g},"
                                f"{r['err_pct']:.4g},{int(r['overload'])},{r['note']}\n")
            self.iter_csv.flush()

    def _write_uff(self) -> None:
        s = self.cfg.save
        if not (s.save_uff and self.run_folder and self.frf_f):
            return
        if len(self.frf_f) < 2:
            self.cb.log("UFF not written: only one frequency has an FRF (UFF 58 needs at least two). "
                        "Its values are in the TDMS file and control summary.")
            return
        res = self._current_result()
        ref = self.channels[self.ref_i]
        resp = [self.channels[i] for i in self.resp_i]
        mode = "secant control" if self.closed_loop else "no control"
        info = f"Stepped sine ({mode}), {len(res.freqs)} lines, fs={self.summary.fs:g}Hz, N={self.cfg.output.block_size}"
        for func, flag, data in (("H1", s.uff_h1, res.H1), ("H2", s.uff_h2, res.H2),
                                 ("Hv", s.uff_hv, res.Hv), ("FundamentalFraction", s.uff_coherence, res.coherence)):
            if flag:
                p = os.path.join(self.run_folder, f"{s.basename}_{func}.uff")
                export.write_uff58(p, func, res.freqs, data, resp, ref, info)
                self.summary.files.append(p)
                self.cb.log(f"Wrote {os.path.basename(p)} ({len(resp)} functions, {res.freqs.size} frequencies)")
        if s.save_harmonics and res.harm is not None:
            self._write_harmonics(res, resp, ref, info)

    def _write_harmonics(self, res, resp, ref, info):
        """CSV of every channel's harmonics, and one UFF per order h ≥ 2 with Y_h / X_1 of the
        responses (only frequencies where h·f0 is below Nyquist)."""
        s = self.cfg.save
        H = res.harm.shape[2]
        p = os.path.join(self.run_folder, f"{s.basename}_harmonics.csv")
        with open(p, "w", encoding="utf-8") as fh:
            # complex amplitude X_h = re + j·im (peak amplitude, x(t) = Re{X_h·e^(j·2π·h·f0·t)}),
            # plus magnitude, phase and level relative to the channel's own fundamental
            fh.write("frequency_Hz,channel,units,order,harmonic_frequency_Hz,re,im,amplitude,phase_deg,level_dBc\n")
            for i, f0 in enumerate(res.freqs):
                for c, ch in enumerate(self.channels):
                    a1 = abs(res.harm[i, c, 0])
                    for h in range(1, H + 1):
                        z = res.harm[i, c, h - 1]
                        if not np.isfinite(z):
                            continue
                        dbc = 20 * np.log10(abs(z) / (a1 + 1e-300) + 1e-300)
                        fh.write(f"{f0:.6g},{ch.name},{ch.units},{h},{h * f0:.6g},{z.real:.9g},{z.imag:.9g},"
                                 f"{abs(z):.6g},{np.degrees(np.angle(z)):.4f},{dbc:.3f}\n")
        self.summary.files.append(p)
        written = []
        for h in range(2, H + 1):
            Hh = np.array([harmonic_frf(res, j, h)[0] for j in range(len(resp))])
            ok = np.all(np.isfinite(Hh), axis=0)
            if ok.sum() < 2:
                continue
            p = os.path.join(self.run_folder, f"{s.basename}_Harmonic{h}.uff")
            export.write_uff58(p, f"Harmonic{h}", res.freqs[ok], Hh[:, ok], resp, ref, info + f", Y{h}/X1")
            self.summary.files.append(p)
            written.append(h)
        self.cb.log(f"Wrote {s.basename}_harmonics.csv (orders 1–{H})"
                    + (f" and harmonic FRF UFFs for h = {', '.join(map(str, written))}" if written else ""))

    # ================================================================== main loop
    def run(self) -> RunSummary:
        o, st = self.cfg.output, self.st
        self.summary = RunSummary()
        self.freqs, self.lines, merged = stepped_frequencies(st, o)
        be = self.backend
        # state
        self.seg_id, self.last_j, self.j, self.amp, self.it = 0, -1, 0, 0.0, 0
        self.to_write: deque = deque()
        self.pending, self.after_segment, self.finishing = None, None, False
        self.waiting, self.safe_amp, self.final_count = False, 0.0, None
        self.hist, self.best, self.iter_rows_j = [], None, []
        self.step_status: Dict[int, str] = {}
        self.step_reason: Dict[int, str] = {}
        self.step_iters: Dict[int, int] = {}
        self.step_rows: List[Dict] = []
        self.dwell_acc: Dict[int, Dict] = {}
        self.frf_f: List[float] = []
        self.frf_cols = {"H1": [], "H2": [], "Hv": [], "coh": []}
        self.harm_rows: List[np.ndarray] = []
        self.n_written = 0
        self.measured, self.best_it, self.tdms_dwell = {}, None, None
        self.last_written_amp, self.last_written_j = 0.0, -1
        self.iter_csv = self.step_csv = None
        try:
            fs, notes = be.configure()
            self.summary.fs = fs
            # the hardware may coerce the rate: the sine sits on line k, so its frequency is k·fs/N
            self.freqs = self.lines * fs / o.block_size
            for n_ in notes:
                self.cb.log(n_)
            self.dwell_blocks = max(1, int(math.ceil(st.dwell_time * fs / o.block_size)))
            self._prepare_files(fs)
            self.tdms_dwell = None
            if self.run_folder and self.cfg.save.save_dwell_tdms:
                p = os.path.join(self.run_folder, self._dwell_name())
                self.tdms_dwell = export.TdmsBlockWriter(p, self.root_props)
                self.summary.files.append(p)
            self._open_logs()
            mode = (f"secant control of {st.target_channel} to {st.target_value:g} {st.target_label} ±{st.tolerance_pct:g} %, "
                    f"≤ {st.max_iterations} iterations, on failure: {st.on_failure.lower()}"
                    if self.closed_loop else f"no control, {st.drive_amplitude:g} V peak")
            self.cb.log(f"Stepped sine: {len(self.freqs)} frequencies {self.freqs[0]:g}–{self.freqs[-1]:g} Hz "
                        f"(lines of Δf = {fs / o.block_size:.4g} Hz" + (f", {merged} requested points merged" if merged else "")
                        + f"); settle {st.settle_blocks} + dwell {self.dwell_blocks} blocks; {mode}")
            self.cb.control({"type": "start", "freqs": self.freqs.tolist(), "closed_loop": self.closed_loop,
                             "target": st.target_value if self.closed_loop else None,
                             "target_label": st.target_label,
                             "tolerance_pct": st.tolerance_pct, "max_iterations": st.max_iterations,
                             "target_channel": st.target_channel if self.closed_loop else ""})

            # schedule: decay zeros, fade-in at the first frequency, then the controller takes over
            n_decay = int(math.ceil(o.decay_time * fs / o.block_size))
            for _ in range(n_decay):
                self.to_write.append(SLabel("decay"))
            amp0 = min(st.initial_drive if self.closed_loop else st.drive_amplitude, o.max_voltage)
            if o.fade:
                self.to_write.append(SLabel("fadein", 0, amp0, first_of_freq=True))
                self.last_j = 0
            self._start_freq(0, amp0)

            queued: deque = deque()
            q = max(2, o.ao_queue_blocks)

            def write_one():
                lab = self._next_label()
                x = self._block_for(lab)
                trig, fired = self._trigger(lab, self.n_written)
                be.write_block(x, trig)
                queued.append((lab, x, trig, fired))
                self.n_written += 1
                if lab.kind not in ("pad",):
                    self.last_written_amp = lab.amp if lab.kind not in ("decay", "tail", "fadeout") else 0.0
                    self.last_written_j = lab.j

            for _ in range(q):
                write_one()
            be.start()
            self.cb.log(f"Started: fs = {fs:g} Hz, block = {o.block_size} ({o.block_size / fs:.3f} s)")

            k = 0
            while self.final_count is None or k < self.final_count:
                data = be.read_block()
                lab, drive, trig, fired = queued.popleft()
                k += 1
                start = (k - 1) * o.block_size
                if fired:
                    self._log_trigger(start, k - 1, _as_block_label(lab), fs)
                # user stop / answers
                if self.stop_event.is_set() and not self.finishing:
                    self.summary.aborted = True
                    self.cb.log("Stop requested: fading out drive…")
                    self._finish()
                if self.waiting and not self.finishing:
                    try:
                        self._apply_choice(self.answers.get_nowait(), self.fail_reason)
                    except queue.Empty:
                        pass
                # analyse the block
                if lab.kind == "measure" and lab.seg == self.pending and not self.finishing:
                    self.pending = None
                    self._on_measure(lab, data, drive, trig, start)
                elif lab.kind == "dwell":
                    self._on_dwell(lab, data, drive, trig, start)
                self.cb.block(data, drive, trig, _as_block_label(lab), be.backlog())
                self.cb.progress(min(self.j + (1 if self.final_count else 0), len(self.freqs)), len(self.freqs),
                                 self._phase(lab))
                write_one()

            self.cb.log("Aborted." if self.summary.aborted else "Stepped sine complete.")
            n_ok = sum(1 for r in self.step_rows if r["status"].startswith(("converged", "open loop")))
            n_flag = sum(1 for r in self.step_rows if r["status"].startswith("accepted"))
            n_bad = sum(1 for r in self.step_rows if r["status"] in ("skipped", "failed"))
            self.cb.log(f"Frequencies: {n_ok} OK, {n_flag} accepted with flag, {n_bad} skipped/failed, "
                        f"of {len(self.freqs)} planned.")
            self._write_uff()
        except Exception as e:
            self.summary.aborted = True
            self.cb.log(f"ERROR: {type(e).__name__}: {e}")
            try:
                self._write_uff()
            except Exception as e2:
                self.cb.log(f"Could not write UFF: {e2}")
            raise
        finally:
            try:
                be.stop()
            finally:
                be.close()
                for fh in (getattr(self, "tdms_conv", None), getattr(self, "tdms_dwell", None),
                           getattr(self, "trig_csv", None),
                           self.iter_csv, self.step_csv):
                    if fh:
                        fh.close()
                self.summary.run_folder = getattr(self, "run_folder", "")
                self.cb.finished(self.summary.__dict__)
        return self.summary

    def _phase(self, lab: SLabel) -> str:
        n = len(self.freqs)
        if lab.kind == "decay":
            return "IEPE settling (drive off)"
        if lab.kind in ("fadeout", "tail", "pad"):
            return "Finishing"
        f = self.freqs[lab.j] if 0 <= lab.j < n else 0
        head = f"Step {lab.j + 1}/{n} · {f:g} Hz · "
        if self.waiting:
            return head + "control FAILED — waiting for your decision (drive held at a safe level)"
        if lab.kind in ("settle", "measure", "hold"):
            return head + (f"iteration {lab.it}/{self.st.max_iterations}" if self.closed_loop else "settling")
        if lab.kind == "dwell":
            return head + f"dwell at {lab.amp:.4g} V"
        return head + lab.kind


def _as_block_label(lab: SLabel):
    """Adapter so the shared GUI/trigger code sees something BlockLabel-like."""
    from .engine import BlockLabel
    kind = {"dwell": "excite"}.get(lab.kind, lab.kind)
    bl = BlockLabel(kind, lab.j, 0)
    bl.step_kind = lab.kind
    bl.amp = lab.amp
    bl.it = lab.it
    return bl
