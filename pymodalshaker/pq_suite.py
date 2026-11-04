# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Phase-quadrature suite: amplitude-dependent resonance (phase resonance testing).

For every drive level (fixed sine amplitude, V peak) the excitation frequency is iterated until
  - phase quadrature: the response channel's fundamental is at the target phase relative to the
    reference channel's fundamental (+90° for acceleration/force, 0° velocity, −90° displacement),
    using a secant method on the phase error e(f) = wrap(φ(f) − φ_target); or
  - reference-free: |H1| = |Y1/X1| is locally maximal (|Y1| per drive volt if there is no
    reference), using a bracketing search with parabolic (secant-like) refinement.
The frequency is NOT restricted to FFT lines: the drive is a phase-continuous sine at any
frequency, and every block is analysed by a least-squares fit of DC + harmonics at the exact
frequency against absolute time (unbiased with a non-integer number of periods per block).

Per level:  [settle × S] [measure]  …iterations…  [dwell × D at the converged frequency]
The first iteration of a level (frequency carried over from the previous level) never counts as
converged: there is always at least one refinement. After convergence the frequency is held for
D dwell blocks; they are averaged for the level's result and saved.

Quadrature secant:  f⁺ = f − e·(f − f_prev)/(e − e_prev), used when the phase falls with frequency
(as it does through a resonance); otherwise a fixed step towards quadrature (phase above target ⇒
raise the frequency), scaled with |e|/45°. Steps are limited to ± max step and to the search range.

Detected failures: input overload, search-range limit, max iterations. Failure actions: ask
(drive held at half level), accept the best iteration (flagged), skip the level, stop the test.

Files (time data only where it is relevant):
  <base>_phase_quadrature.csv    one row per level: drive, frequency, phase, reference and response
                                 fundamentals (complex), H1 = Y1/X1 (complex), ratio |Y1|/|X1|, |Y1|/V, THD
  <base>_pq_harmonics.csv        complex harmonics 1…n of every channel at every level
  <base>_control_iterations.csv  every iteration (frequency, phase / objective, note)
  <base>_phase_quadrature_blocks.tdms  the converged (dwell) block(s) of every level  [option]
  <base>_H1.uff                  H1 at the resonance of every level vs frequency (UFF 58, uneven)  [option]
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
from .config import PQ_FAIL_ACTIONS, AppConfig
from .engine import BlockLabel, Callbacks, MeasurementEngine, RunSummary
from .excitation import fade_window
from .harmonics import ls_harmonics

PQ_ASK = ["Retry (another set of iterations)", "Accept best and continue", "Skip this level", "Stop the test"]


def wrap180(d):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


@dataclass
class PLabel:
    kind: str                      # decay fadein settle measure hold dwell pause fadeout tail pad
    j: int = -1                    # level index
    amp: float = 0.0               # V peak
    f: float = 0.0                 # Hz
    seg: int = -1
    it: int = 0
    first_of_level: bool = False
    first_dwell: bool = False


class PQEngine(MeasurementEngine):
    def __init__(self, cfg: AppConfig, backend, callbacks: Optional[Callbacks] = None):
        super().__init__(cfg, backend, callbacks)
        self.answers: "queue.Queue[str]" = queue.Queue()
        self.p = cfg.pq
        names = [c.name for c in self.channels]
        self.y_i = names.index(self.p.response_channel)
        self.has_ref = self.ref_i >= 0
        self.levels = self.p.levels()
        self.n_h = max(2, self.p.n_harmonics)
        # GUI worker compatibility (stepped-like attributes)
        self.freqs = np.array([self.p.f_init])
        self.lines = np.array([0])

    def answer(self, choice: str) -> None:
        self.answers.put(choice)

    # ================================================================== files
    def _tdms_name(self) -> str:
        return ""

    def _extra_root_props(self) -> Dict:
        p = self.p
        return {"suite": "Phase Quadrature", "control": p.control, "response_channel": p.response_channel,
                "reference_channel": self.channels[self.ref_i].name if self.has_ref else "",
                "target_phase_deg": p.target_phase, "phase_tol_deg": p.phase_tol, "freq_tol_Hz": p.freq_tol,
                "levels_V": ",".join(f"{v:.6g}" for v in self.levels), "f_init_Hz": p.f_init,
                "settle_blocks": p.settle_blocks, "dwell_blocks": p.dwell_blocks}

    def _open_files(self):
        self.res_csv = self.harm_csv = self.iter_csv = self.tdms = None
        if not self.run_folder:
            return
        b = self.cfg.save.basename
        ref_u = self.channels[self.ref_i].units if self.has_ref else ""
        y_u = self.channels[self.y_i].units

        def op(name, header):
            path = os.path.join(self.run_folder, f"{b}_{name}")
            fh = open(path, "w", encoding="utf-8")
            fh.write(header + "\n")
            self.summary.files.append(path)
            return fh
        self.res_csv = op("phase_quadrature.csv",
                          "level,drive_V,status,iterations,frequency_Hz,phase_resp_re_ref_deg,phase_error_deg,"
                          "ref_channel,ref_units,ref_amp,ref_re,ref_im,resp_channel,resp_units,resp_amp,resp_re,resp_im,"
                          "H1_re,H1_im,H1_amp,H1_phase_deg,ratio_resp_to_ref,resp_per_volt,thd_resp_percent,"
                          "thd_ref_percent,overload,reason")
        self._units = (ref_u, y_u)
        self.harm_csv = op("pq_harmonics.csv", "level,frequency_Hz,channel,units,order,harmonic_frequency_Hz,"
                                               "re,im,amplitude,phase_deg,level_dBc")
        self.iter_csv = op("control_iterations.csv", "level,drive_V,iteration,frequency_Hz,phase_deg,"
                                                     "phase_error_deg,objective,overload,note")
        if self.cfg.save.pq_save_blocks:
            path = os.path.join(self.run_folder, f"{b}_phase_quadrature_blocks.tdms")
            self.tdms = export.TdmsBlockWriter(path, self.root_props)
            self.summary.files.append(path)

    # ================================================================== drive
    def _block_for(self, lab: PLabel) -> np.ndarray:
        n, fs = self.cfg.output.block_size, self.summary.fs
        if lab.kind in ("decay", "tail", "pad") or lab.amp <= 0:
            return np.zeros(n)
        w = 2 * np.pi * lab.f / fs
        x = lab.amp * np.sin(self.ph + w * np.arange(n))
        self.ph = (self.ph + w * n) % (2 * np.pi)          # phase-continuous across blocks and frequency changes
        if lab.kind == "fadein":
            return x * fade_window(n, True)
        if lab.kind == "fadeout":
            return x * fade_window(n, False)
        return x

    def _trigger(self, lab: PLabel, block_no: int):
        t = self.cfg.trigger
        if not t.enabled or t.frame_sync:
            return None, False
        if t.mode.startswith("Once"):
            fire = block_no == 0
        elif t.mode.startswith("Start of the converged"):
            fire = lab.first_dwell
        else:
            fire = lab.first_of_level
        return self._trigger_wave(fire), fire

    # ================================================================== scheduling
    def _queue(self, kinds, j, amp, f, it=0):
        self.seg_id += 1
        first_level = j != self.last_j
        for i, k in enumerate(kinds):
            self.to_write.append(PLabel(k, j, amp, f, self.seg_id, it, first_of_level=first_level and i == 0,
                                        first_dwell=(k == "dwell" and (i == 0 or kinds[i - 1] != "dwell"))))
        self.last_j = j
        return self.seg_id

    def _queue_iteration(self, f):
        p = self.p
        if not p.f_min - 1e-9 <= f <= p.f_max + 1e-9:
            f = min(max(f, p.f_min), p.f_max)
        self.it += 1
        self.f = f
        self.pending = self._queue(["settle"] * p.settle_blocks + ["measure"], self.j, self.amp, f, self.it)

    def _queue_dwell(self, f, status, settle):
        self.f = f
        self.status = status
        kinds = ["settle"] * (self.p.settle_blocks if settle else 0) + ["dwell"] * self.p.dwell_blocks
        self._queue(kinds, self.j, self.amp, f)
        self.after_segment = "next_level"
        self.f_res_last = f                              # warm start of the next level
        # the level's context travels with its dwell blocks: by the time they come back through the
        # output queue the scheduler may already be on the next level
        self.dwell = {"blocks": [], "drive": [], "trig": [], "start": [], "f": f, "j": self.j, "amp": self.amp,
                      "status": status, "iterations": self.it, "reason": self.reason}

    def _start_level(self, j: int, f: float):
        self.j, self.it, self.amp = j, 0, float(self.levels[j])
        self.hist, self.pts, self.best, self.edge_moves = [], [], None, 0
        self.measured = {}
        self.reason = ""
        self.cb.control({"type": "pq_level", "j": j, "amp": self.amp, "f": f, "n": len(self.levels)})
        self._queue_iteration(f)

    def _advance(self):
        nxt = self.j + 1
        if nxt >= len(self.levels) or self.stop_event.is_set():
            self._finish()
        else:
            self._start_level(nxt, self.f_res_last if self.f_res_last else self.p.f_init)

    def _finish(self):
        if self.finishing:
            return
        self.finishing = True
        self.to_write.clear()
        self.pending = None
        if self.last_written is not None and self.last_written.amp > 0:
            lw = self.last_written
            self.to_write.append(PLabel("fadeout", lw.j, lw.amp, lw.f))
        self.to_write.append(PLabel("tail"))
        self.after_segment = "ended"
        self.final_count = self.n_written + len(self.to_write)

    def _next_label(self) -> PLabel:
        if not self.to_write:
            if self.after_segment == "next_level":
                self.after_segment = None
                self._advance()
        if self.to_write:
            return self.to_write.popleft()
        if self.after_segment == "ended":
            return PLabel("pad")
        if self.waiting:
            return PLabel("pause", self.j, 0.5 * self.amp, self.f)
        return PLabel("hold", self.j, self.amp, self.f)

    # ================================================================== analysis
    def _fund(self, data: np.ndarray, f: float, start: int, n_h: int = 1):
        """dc (nch,), complex harmonics (nch, n_h) of every channel at frequency f, absolute time."""
        return ls_harmonics(data, f, self.summary.fs, n_h, start)

    def _overloads(self, data) -> List[str]:
        lim = self.p.overload_pct / 100.0
        return [c.name for c, row in zip(self.channels, data) if np.max(np.abs(row)) >= lim * c.max_range]

    def _measure_quantities(self, data, f, start):
        _, X = self._fund(data, f, start)
        Y1 = X[self.y_i, 0]
        X1 = X[self.ref_i, 0] if self.has_ref else None
        H1 = Y1 / X1 if X1 is not None and abs(X1) > 0 else None
        phase = float(np.degrees(np.angle(H1))) if H1 is not None else float("nan")
        J = abs(H1) if H1 is not None else abs(Y1) / max(self.amp, 1e-30)
        return Y1, X1, H1, phase, J

    # ================================================================== controller
    def _on_measure(self, lab: PLabel, data, drive, trig, start: int):
        p = self.p
        f = lab.f
        Y1, X1, H1, phase, J = self._measure_quantities(data, f, start)
        over = self._overloads(data)
        self.measured[lab.it] = f
        note, fail, f_next, conv_f = "", "", None, None
        e = float(wrap180(phase - p.target_phase)) if p.quadrature else float("nan")
        if over:
            note = "input overload: " + ", ".join(over)
            fail = f"input overload ({', '.join(over)}) at {self.amp:g} V"
        elif p.quadrature:
            self.hist.append((f, e))
            if self.best is None or abs(e) < abs(self.best[1]):
                self.best = (f, e)
            if abs(e) <= p.phase_tol and lab.it >= 2:
                conv_f, note = f, "converged"
            else:
                f_next, note = self._quad_next(f, e)
                if abs(e) <= p.phase_tol:
                    note = "within tolerance: one refinement step; " + note
        else:
            self.pts.append((f, J))
            if self.best is None or J > self.best[1]:
                self.best = (f, J)
            f_next, conv, note = self._max_next()
            if conv and lab.it >= 2:
                conv_f, note = self.best[0], f"converged (peak bracketed within ±{p.freq_tol:g} Hz)"
        if not fail and f_next is not None and conv_f is None:
            if not p.f_min - 1e-9 <= f_next <= p.f_max + 1e-9:
                at_edge = abs(f - p.f_min) < 1e-9 or abs(f - p.f_max) < 1e-9
                if at_edge:
                    fail = f"the search range {p.f_min:g}–{p.f_max:g} Hz is exhausted (resonance outside it?)"
                else:
                    f_next = min(max(f_next, p.f_min), p.f_max)
                    note += "; clamped to the search range"
        if not fail and conv_f is None and lab.it >= p.max_iterations:
            fail = f"not converged in {p.max_iterations} iterations"
        self._log_iter(lab, f, phase, e, J, over, note)
        if fail:
            self._failure(fail)
        elif conv_f is not None:
            self.cb.log(f"Level {lab.j + 1}: {self.amp:g} V → resonance at {conv_f:.5g} Hz after {lab.it} iterations")
            self._queue_dwell(conv_f, "converged", settle=abs(conv_f - f) > 1e-12)
        else:
            self._queue_iteration(f_next)

    def _quad_next(self, f, e):
        p = self.p
        note = ""
        f_new = None
        if len(self.hist) >= 2:
            (f0, e0), (f1, e1) = self.hist[-2], self.hist[-1]
            if abs(f1 - f0) > 1e-12:
                slope = (e1 - e0) / (f1 - f0)
                if slope < 0 and abs(e1 - e0) < 170:          # phase falling with frequency (through a resonance)
                    f_new, note = f1 - e1 / slope, "secant"
                else:
                    note = "phase not falling with frequency: fixed step"
        if f_new is None:
            step = p.step0 * min(1.0, max(abs(e), 0.5) / 45.0)
            f_new = f + (step if e > 0 else -step)
            note = note or "first step towards quadrature"
        d = f_new - f
        if abs(d) > p.max_step:
            f_new = f + math.copysign(p.max_step, d)
            note += f"; step limited to {p.max_step:g} Hz"
        return f_new, note

    def _max_next(self):
        """Reference-free: next frequency, whether converged, note."""
        p = self.p
        P = sorted(self.pts)
        fs_ = [q[0] for q in P]
        Js = [q[1] for q in P]
        ib = int(np.argmax(Js))
        if len(P) == 1:
            return P[0][0] + p.step0, False, "first probe step"
        if 0 < ib < len(P) - 1:                                   # peak bracketed
            (f1, j1), (f2, j2), (f3, j3) = P[ib - 1], P[ib], P[ib + 1]
            den = (f2 - f1) * (j2 - j3) - (f2 - f3) * (j2 - j1)
            fv = None
            if abs(den) > 1e-300:
                fv = f2 - 0.5 * ((f2 - f1) ** 2 * (j2 - j3) - (f2 - f3) ** 2 * (j2 - j1)) / den
            if fv is None or not f1 < fv < f3:
                fv = 0.5 * (f2 + (f1 if f2 - f1 > f3 - f2 else f3))
                note = "bracketed: bisection"
            else:
                note = "bracketed: parabolic step"
            width = f3 - f1
            if abs(fv - f2) <= p.freq_tol and width <= 6 * p.freq_tol + 2 * abs(fv - f2):
                return f2, True, note
            if abs(fv - f2) <= p.freq_tol:                    # vertex at the best point: shrink the bracket
                side = f1 if f2 - f1 > f3 - f2 else f3
                fv = f2 + 0.5 * (side - f2) * 0.5
                note = "bracketed: tightening"
            if min(abs(fv - x) for x in fs_) < 0.25 * p.freq_tol:
                return f2, True, note + " (no further change)"
            return fv, False, note
        direction = 1.0 if ib == len(P) - 1 else -1.0
        self.edge_moves += 1
        step = min(p.max_step, p.step0 * 2 ** (self.edge_moves - 1))
        return P[ib][0] + direction * step, False, f"climbing {'up' if direction > 0 else 'down'} ({step:g} Hz)"

    def _log_iter(self, lab, f, phase, e, J, over, note):
        if self.iter_csv:
            self.iter_csv.write(f"{lab.j + 1},{self.amp:.6g},{lab.it},{f:.8g},{phase:.4f},{e:.4f},{J:.6g},"
                                f"{int(bool(over))},\"{note}\"\n")
            self.iter_csv.flush()
        self.cb.control({"type": "pq_iter", "j": lab.j, "it": lab.it, "f": f, "phase": phase, "e": e, "J": J,
                         "overload": bool(over), "note": note, "amp": self.amp})

    def _failure(self, reason: str):
        action = self.p.on_failure
        self.reason = reason
        self.cb.log(f"Control failed at level {self.j + 1} ({self.amp:g} V): {reason}")
        best = self.best
        if action == PQ_FAIL_ACTIONS[0]:
            self.waiting = True
            self.pending = None
            self.cb.control({"type": "pq_wait", "j": self.j, "reason": reason})
            bt = None
            if best is not None:
                bt = (f"Best iteration: {best[0]:.5g} Hz, " + (f"phase error {best[1]:+.2f}°" if self.p.quadrature
                                                              else f"|H1| {best[1]:.4g}") + ".")
            self.cb.ask({"f": float(self.f), "reason": f"Level {self.j + 1} ({self.amp:g} V): {reason}",
                         "choices": PQ_ASK, "best": None if best is None else {"amp": self.amp, "err_pct": 0.0},
                         "best_text": bt})
            return
        self._apply_choice({PQ_FAIL_ACTIONS[1]: PQ_ASK[1], PQ_FAIL_ACTIONS[2]: PQ_ASK[2],
                            PQ_FAIL_ACTIONS[3]: PQ_ASK[3]}[action], reason)

    def _apply_choice(self, choice: str, reason: str):
        self.waiting = False
        best = self.best
        if choice == PQ_ASK[0]:
            self.cb.log("  → retrying this level")
            f0 = best[0] if best else self.f
            self.it, self.hist, self.pts, self.best, self.edge_moves = 0, [], [], None, 0
            self._queue_iteration(f0)
        elif choice == PQ_ASK[1] and best is not None:
            self.cb.log(f"  → accepting the best iteration at {best[0]:.5g} Hz (flagged)")
            self._queue_dwell(best[0], "accepted (flagged)", settle=True)
        elif choice == PQ_ASK[3]:
            self.cb.log("  → stopping the test")
            self.summary.aborted = True
            self._record_level(None, "failed")
            self._finish()
        else:
            self.cb.log(f"  → skipping level {self.j + 1}")
            self._record_level(None, "skipped")
            self._advance()

    # ================================================================== level result
    def _on_dwell(self, lab, data, drive, trig, start):
        d = self.dwell
        d["blocks"].append(data.copy())
        d["drive"].append(drive.copy())
        if trig is not None:
            d["trig"].append(trig.copy())
        d["start"].append(start)
        if len(d["blocks"]) == self.p.dwell_blocks:
            self._finish_level()
            self.dwell = None

    def _finish_level(self):
        d, f, fs = self.dwell, self.dwell["f"], self.summary.fs
        H = self.n_h
        Xs, Vs, over = [], [], set()
        for blk, drv, st in zip(d["blocks"], d["drive"], d["start"]):
            _, X = self._fund(blk, f, st, H)
            _, V = self._fund(drv, f, st, 1)
            Xs.append(X)
            Vs.append(V[0, 0])
            over.update(self._overloads(blk))
        Xs = np.array(Xs)                                   # (nb, nch, H)
        Xm = np.nanmean(Xs, axis=0)                         # averaged complex harmonics
        Y = Xs[:, self.y_i, 0]
        res = {"f": f, "harm": Xm, "Y1": Xm[self.y_i, 0], "V1": complex(np.mean(Vs)), "over": sorted(over)}
        if self.has_ref:
            Xr = Xs[:, self.ref_i, 0]
            H1 = np.sum(Y * np.conj(Xr)) / max(np.sum(np.abs(Xr) ** 2), 1e-300)
            res.update({"X1": Xm[self.ref_i, 0], "H1": complex(H1)})
        self._record_level(res, d["status"], ctx=d)
        # TDMS: the converged block(s) of this level
        if self.tdms is not None:
            r = self.rows[-1]
            props = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()
                     if v is not None and not isinstance(v, (complex, np.complexfloating, list, np.ndarray))}
            props.update({"first_sample_index": int(d["start"][0]), "first_sample_time_s": d["start"][0] / fs,
                          "dwell_blocks": len(d["blocks"]), "frequency_Hz": float(f)})
            ext = {"Drive": np.concatenate(d["drive"])}
            if d["trig"]:
                ext["Trigger"] = np.concatenate(d["trig"])
            self.tdms.write_group(f"Level_{d['j'] + 1:03d}", self.channels, np.concatenate(d["blocks"], axis=1),
                                  ext, fs, props)

    def _record_level(self, res: Optional[Dict], status: str, ctx: Optional[Dict] = None):
        p = self.p
        ctx = ctx or {"j": self.j, "amp": self.amp, "iterations": self.it, "reason": self.reason}
        row = {"level": ctx["j"] + 1, "drive_V": float(ctx["amp"]), "status": status,
               "iterations": int(ctx["iterations"]), "reason": ctx["reason"]}
        if res is not None:
            Y1, H1 = res["Y1"], res.get("H1")
            X1 = res.get("X1")
            ph = float(np.degrees(np.angle(H1))) if H1 is not None else float("nan")
            thd = lambda c: float(np.sqrt(np.nansum(np.abs(res["harm"][c, 1:]) ** 2)) / (abs(res["harm"][c, 0]) + 1e-300) * 100)
            row.update({"frequency_Hz": float(res["f"]), "phase_deg": ph,
                        "phase_error_deg": float(wrap180(ph - p.target_phase)) if H1 is not None else float("nan"),
                        "ref_amp": abs(X1) if X1 is not None else float("nan"), "X1": X1,
                        "resp_amp": abs(Y1), "Y1": Y1, "H1": H1,
                        "H1_amp": abs(H1) if H1 is not None else float("nan"),
                        "ratio": abs(Y1) / abs(X1) if X1 is not None and abs(X1) > 0 else float("nan"),
                        "resp_per_volt": abs(Y1) / max(abs(res["V1"]), 1e-30),
                        "thd_resp": thd(self.y_i), "thd_ref": thd(self.ref_i) if self.has_ref else float("nan"),
                        "overload": ", ".join(res["over"]), "harm": res["harm"]})
        self.rows.append(row)
        self._write_level_csv(row)
        ev = {"type": "pq_result", **{k: v for k, v in row.items()}}
        self.cb.control(ev)
        self.summary.completed_realizations = sum(1 for r in self.rows if "frequency_Hz" in r)

    def _write_level_csv(self, r: Dict):
        if self.res_csv is None:
            return
        g = lambda k: r.get(k, float("nan"))
        cz = lambda z: ("", "") if z is None or (isinstance(z, float) and np.isnan(z)) else (f"{z.real:.9g}", f"{z.imag:.9g}")
        num = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.8g}"
        ref_name = self.channels[self.ref_i].name if self.has_ref else ""
        Xr, Xi = cz(r.get("X1"))
        Yr, Yi = cz(r.get("Y1"))
        Hr, Hi = cz(r.get("H1"))
        cols = [str(r["level"]), num(r["drive_V"]), r["status"], str(r["iterations"]), num(g("frequency_Hz")),
                num(g("phase_deg")), num(g("phase_error_deg")), ref_name, self._units[0], num(g("ref_amp")), Xr, Xi,
                self.p.response_channel, self._units[1], num(g("resp_amp")), Yr, Yi, Hr, Hi, num(g("H1_amp")),
                num(g("phase_deg")), num(g("ratio")), num(g("resp_per_volt")), num(g("thd_resp")), num(g("thd_ref")),
                f'"{r.get("overload", "")}"', f'"{r.get("reason", "")}"']
        self.res_csv.write(",".join(cols) + "\n")
        self.res_csv.flush()
        if self.harm_csv is not None and "harm" in r:
            Xh = r["harm"]
            f0 = r["frequency_Hz"]
            for c, ch in enumerate(self.channels):
                a1 = abs(Xh[c, 0])
                for h in range(1, Xh.shape[1] + 1):
                    z = Xh[c, h - 1]
                    if not np.isfinite(z):
                        continue
                    dbc = 20 * np.log10(abs(z) / (a1 + 1e-300) + 1e-300)
                    self.harm_csv.write(f"{r['level']},{f0:.8g},{ch.name},{ch.units},{h},{h * f0:.8g},{z.real:.9g},"
                                        f"{z.imag:.9g},{abs(z):.6g},{np.degrees(np.angle(z)):.4f},{dbc:.3f}\n")
            self.harm_csv.flush()

    def _write_uff(self) -> None:
        if not (self.run_folder and self.cfg.save.pq_save_uff and self.has_ref):
            return
        ok = [r for r in self.rows if r.get("H1") is not None]
        if len(ok) < 2:
            self.cb.log("H1 UFF not written: fewer than two converged levels.")
            return
        ok = sorted(ok, key=lambda r: r["frequency_Hz"])
        f = np.array([r["frequency_Hz"] for r in ok])
        H = np.array([[r["H1"] for r in ok]])
        ref, resp = self.channels[self.ref_i], [self.channels[self.y_i]]
        path = os.path.join(self.run_folder, f"{self.cfg.save.basename}_H1.uff")
        export.write_uff58(path, "H1", f, H, resp, ref,
                           f"Phase quadrature ({self.p.control.split(' (')[0]}), H1 at resonance per level")
        self.summary.files.append(path)
        self.cb.log(f"Wrote {os.path.basename(path)} (H1 at {len(ok)} resonance points)")

    # ================================================================== main loop
    def run(self) -> RunSummary:
        o, p = self.cfg.output, self.p
        self.summary = RunSummary()
        be = self.backend
        self.seg_id, self.last_j, self.j, self.it, self.n_written = 0, -1, 0, 0, 0
        self.to_write: deque = deque()
        self.pending, self.after_segment, self.finishing, self.final_count = None, None, False, None
        self.waiting, self.status, self.reason = False, "", ""
        self.ph, self.f, self.amp, self.f_res_last = 0.0, p.f_init, 0.0, None
        self.hist, self.pts, self.best, self.edge_moves, self.measured = [], [], None, 0, {}
        self.dwell = None
        self.rows: List[Dict] = []
        self.last_written: Optional[PLabel] = None
        self.res_csv = self.harm_csv = self.iter_csv = self.tdms = None
        try:
            fs, notes = be.configure()
            self.summary.fs = fs
            for n_ in notes:
                self.cb.log(n_)
            self._prepare_files(fs)
            self._open_files()
            ref = self.channels[self.ref_i].name if self.has_ref else "(none)"
            mode = (f"phase quadrature of {p.response_channel} re {ref} at {p.target_phase:+g}° ±{p.phase_tol:g}°"
                    if p.quadrature else
                    f"reference-free: maximise |{p.response_channel}/{ref}| (±{p.freq_tol:g} Hz)" if self.has_ref else
                    f"reference-free: maximise |{p.response_channel}| per volt (±{p.freq_tol:g} Hz)")
            self.cb.log(f"Phase Quadrature: {len(self.levels)} drive levels {', '.join(f'{v:.4g}' for v in self.levels)} V; "
                        f"start at {p.f_init:g} Hz, search {p.f_min:g}–{p.f_max:g} Hz; {mode}; settle {p.settle_blocks}, "
                        f"dwell {p.dwell_blocks} block(s)")
            self.cb.control({"type": "pq_start", "levels": self.levels.tolist(), "quadrature": p.quadrature,
                             "target": p.target_phase, "phase_tol": p.phase_tol, "freq_tol": p.freq_tol,
                             "max_iterations": p.max_iterations, "resp": p.response_channel, "ref": ref,
                             "resp_units": self.channels[self.y_i].units,
                             "ref_units": self.channels[self.ref_i].units if self.has_ref else ""})
            n_decay = int(math.ceil(o.decay_time * fs / o.block_size))
            for _ in range(n_decay):
                self.to_write.append(PLabel("decay"))
            if o.fade:
                self.to_write.append(PLabel("fadein", 0, float(self.levels[0]), p.f_init, first_of_level=True))
                self.last_j = 0
            self._start_level(0, p.f_init)

            queued: deque = deque()
            q = max(2, o.ao_queue_blocks)

            def write_one():
                lab = self._next_label()
                x = self._block_for(lab)
                trig, fired = self._trigger(lab, self.n_written)
                be.write_block(x, trig)
                queued.append((lab, x, trig, fired))
                self.n_written += 1
                if lab.kind not in ("pad", "decay", "tail"):
                    self.last_written = lab if lab.kind != "fadeout" else None

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
                bl = _as_block_label(lab)
                if fired:
                    self._log_trigger(start, kblk - 1, bl, fs)
                if self.stop_event.is_set() and not self.finishing:
                    self.summary.aborted = True
                    self.cb.log("Stop requested: fading out drive…")
                    self._finish()
                if self.waiting and not self.finishing:
                    try:
                        self._apply_choice(self.answers.get_nowait(), self.reason)
                    except queue.Empty:
                        pass
                if lab.kind == "measure" and lab.seg == self.pending and not self.finishing:
                    self.pending = None
                    self._on_measure(lab, data, drive, trig, start)
                elif lab.kind == "dwell" and self.dwell is not None and lab.j == self.dwell["j"]:
                    self._on_dwell(lab, data, drive, trig, start)
                self.cb.block(data, drive, trig, bl, be.backlog())
                done = sum(1 for r in self.rows)
                self.cb.progress(min(done, len(self.levels)), len(self.levels), self._phase(lab))
                write_one()
            n_ok = sum(1 for r in self.rows if r["status"] == "converged")
            n_fl = sum(1 for r in self.rows if r["status"].startswith("accepted"))
            n_bad = sum(1 for r in self.rows if r["status"] in ("skipped", "failed"))
            self.cb.log(f"Levels: {n_ok} converged, {n_fl} accepted with flag, {n_bad} skipped/failed, of {len(self.levels)}.")
            self._write_uff()
            self.cb.log("Aborted." if self.summary.aborted else "Phase quadrature complete.")
            self.cb.control({"type": "pq_done"})
        except Exception as e:
            self.summary.aborted = True
            self.cb.log(f"ERROR: {type(e).__name__}: {e}")
            raise
        finally:
            try:
                be.stop()
            finally:
                be.close()
                for fh in (self.res_csv, self.harm_csv, self.iter_csv, self.tdms, getattr(self, "trig_csv", None)):
                    if fh:
                        fh.close()
                self.summary.run_folder = getattr(self, "run_folder", "")
                self.cb.finished(self.summary.__dict__)
        return self.summary

    def _phase(self, lab: PLabel) -> str:
        n = len(self.levels)
        if lab.kind == "decay":
            return "IEPE settling (drive off)"
        if lab.kind in ("fadeout", "tail", "pad"):
            return "Finishing"
        head = f"Level {lab.j + 1}/{n} · {lab.amp:.4g} V · {lab.f:.5g} Hz · "
        if self.waiting:
            return head + "control FAILED — waiting for your decision (drive held at half level)"
        if lab.kind in ("settle", "measure", "hold"):
            return head + f"iteration {lab.it}/{self.p.max_iterations}"
        if lab.kind == "dwell":
            return head + "recording the converged block"
        return head + lab.kind


def _as_block_label(lab: PLabel) -> BlockLabel:
    kind = {"dwell": "excite"}.get(lab.kind, lab.kind)
    bl = BlockLabel(kind, lab.j, 0)
    bl.step_kind = lab.kind
    bl.amp = lab.amp
    bl.it = lab.it
    bl.freq = lab.f if lab.kind in ("settle", "measure", "hold", "dwell", "pause") else None
    return bl
