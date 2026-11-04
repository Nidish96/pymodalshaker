# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Read back saved runs (Qt-free): find run folders, list what was saved, load FRFs from UFF 58
files and time blocks from the TDMS files."""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .config import DIRECTION_CODE

FUNCS = ("H1", "H2", "Hv", "Coherence")
_DIR_NAME = {v: k for k, v in DIRECTION_CODE.items()}
_REAL_RE = re.compile(r"_realization_(\d+)\.tdms$", re.IGNORECASE)
_GROUP_RE = re.compile(r"(?:Realization|Step)_(\d+)$", re.IGNORECASE)


@dataclass
class Run:
    folder: str
    base: str = ""
    setup: Dict = field(default_factory=dict)
    uff: Dict[str, str] = field(default_factory=dict)          # func -> path
    converged: str = ""                    # time-data TDMS: converged blocks (random) or dwell blocks per step
    stepped: bool = False                  # stepped-sine run (groups Step_NNN, one per frequency)
    control_files: List[str] = field(default_factory=list)
    quality_label: str = "Coherence γ²"    # stepped sine: "Fundamental fraction η"
    stepped_converged: str = ""            # stepped sine: converged block per frequency (TDMS)
    harmonics_csv: str = ""                # stepped sine: <base>_harmonics.csv (complex harmonics, no time data needed)
    pq: bool = False                       # phase-quadrature run (resonance per drive level)
    pq_csv: str = ""                       # <base>_phase_quadrature.csv
    pq_harm_csv: str = ""                  # <base>_pq_harmonics.csv
    pq_blocks: str = ""                    # <base>_phase_quadrature_blocks.tdms
    harmonic: bool = False                 # harmonic-suite run (time data only)
    harmonic_file: str = ""                # <base>_harmonic.tdms: one group Block_NNNNNN per saved block
    realization_files: Dict[int, str] = field(default_factory=dict)   # r (1-based) -> path
    trigger_csv: str = ""
    _uff_cache: Dict[str, list] = field(default_factory=dict, repr=False)
    _conv_cache: object = field(default=None, repr=False)
    _spec_cache: object = field(default=None, repr=False)
    _frf_tables: Dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ summary
    @property
    def name(self) -> str:
        return os.path.basename(os.path.normpath(self.folder))

    def output(self) -> Dict:
        return self.setup.get("output", {}) if self.setup else {}

    def converged_realizations(self) -> List[int]:
        if not self.converged:
            return []
        out = []
        for g in self._conv().groups():
            m = _GROUP_RE.match(g.name)
            if m:
                out.append(int(m.group(1)))
        return sorted(out)

    def realizations(self) -> List[int]:
        return sorted(set(self.converged_realizations()) | set(self.realization_files))

    def step_info(self, r: int) -> Dict:
        """Stepped sine: properties of step r (frequency_Hz, status, …)."""
        try:
            return dict(self._conv()[f"Step_{r:03d}"].properties)
        except Exception:
            return {}

    # ------------------------------------------------------------------ phase quadrature
    def pq_rows(self) -> List[Dict]:
        """One dict per level (as written by the engine): level, drive_V, status, iterations, frequency_Hz,
        phase_deg, ref_amp, resp_amp, X1, Y1, H1 (complex or None), H1_amp, ratio, resp_per_volt, …"""
        if getattr(self, "_pq_rows", None) is not None:
            return self._pq_rows
        rows = []
        if self.pq_csv:
            fnum = lambda v: float(v) if v not in ("", None) else float("nan")
            cz = lambda a, b: complex(float(a), float(b)) if a not in ("", None) and b not in ("", None) else None
            with open(self.pq_csv, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    rows.append({"level": int(r["level"]), "drive_V": fnum(r["drive_V"]), "status": r["status"],
                                 "iterations": int(r["iterations"] or 0), "frequency_Hz": fnum(r["frequency_Hz"]),
                                 "phase_deg": fnum(r["phase_resp_re_ref_deg"]), "phase_error_deg": fnum(r["phase_error_deg"]),
                                 "ref_amp": fnum(r["ref_amp"]), "resp_amp": fnum(r["resp_amp"]),
                                 "X1": cz(r["ref_re"], r["ref_im"]), "Y1": cz(r["resp_re"], r["resp_im"]),
                                 "H1": cz(r["H1_re"], r["H1_im"]), "H1_amp": fnum(r["H1_amp"]),
                                 "ratio": fnum(r["ratio_resp_to_ref"]), "resp_per_volt": fnum(r["resp_per_volt"]),
                                 "thd_resp": fnum(r["thd_resp_percent"]), "thd_ref": fnum(r["thd_ref_percent"]),
                                 "overload": r.get("overload", ""), "reason": r.get("reason", ""),
                                 "ref_channel": r["ref_channel"], "ref_units": r["ref_units"],
                                 "resp_channel": r["resp_channel"], "resp_units": r["resp_units"]})
        self._pq_rows = rows
        return rows

    def pq_block(self, level: int) -> Dict:
        """Converged block(s) of one level: channels [(name, unit, array)], fs, props, start index."""
        from nptdms import TdmsFile
        if getattr(self, "_pq_tdms", None) is None:
            self._pq_tdms = TdmsFile.read(self.pq_blocks)
        g = self._pq_tdms[f"Level_{level:03d}"]
        chans, fs = [], None
        for ch in g.channels():
            p = ch.properties
            if fs is None and p.get("wf_increment"):
                fs = 1.0 / float(p["wf_increment"])
            chans.append((ch.name, p.get("unit_string", ""), np.asarray(ch[:], dtype=float)))
        props = dict(g.properties)
        return {"channels": chans, "fs": fs or float(self.output().get("fs", 1.0)), "props": props,
                "start": int(props.get("first_sample_index", 0)), "f": float(props.get("frequency_Hz", 0.0)),
                "n_blocks": int(props.get("dwell_blocks", 1))}

    def pq_levels_with_blocks(self) -> List[int]:
        if not self.pq_blocks:
            return []
        from nptdms import TdmsFile
        if getattr(self, "_pq_tdms", None) is None:
            self._pq_tdms = TdmsFile.read(self.pq_blocks)
        out = []
        for g in self._pq_tdms.groups():
            if g.name.startswith("Level_"):
                out.append(int(g.name.split("_")[1]))
        return sorted(out)

    # ------------------------------------------------------------------ harmonic suite
    def _harm(self):
        if getattr(self, "_harm_cache", None) is None:
            from nptdms import TdmsFile
            self._harm_cache = TdmsFile.read(self.harmonic_file)
        return self._harm_cache

    def harmonic_blocks(self) -> List[Dict]:
        """Saved blocks of a harmonic run, in order: [{'group', 'kind', 'index', 'test', 'iteration', 'props'}]."""
        if not self.harmonic_file:
            return []
        out = []
        for g in self._harm().groups():
            p = dict(g.properties)
            out.append({"group": g.name, "kind": p.get("kind", ""), "index": int(p.get("block_index", 0)),
                        "test": p.get("test_block"), "iteration": p.get("iteration"), "props": p})
        return sorted(out, key=lambda b: b["index"])

    def harmonic_block(self, group: str) -> Dict:
        """channels [(name, unit, array)], fs, props, line (FFT line of f0) of one saved block."""
        g = self._harm()[group]
        chans, fs = [], None
        for ch in g.channels():
            p = ch.properties
            if fs is None and p.get("wf_increment"):
                fs = 1.0 / float(p["wf_increment"])
            chans.append((ch.name, p.get("unit_string", ""), np.asarray(ch[:], dtype=float)))
        props = dict(g.properties)
        root = dict(self._harm().properties)
        fs = fs or float(root.get("sample_rate_Hz", self.output().get("fs", 1.0)))
        line = int(props.get("base_line", root.get("base_line", 0)))
        return {"channels": chans, "fs": fs, "props": props, "line": line,
                "f0": float(props.get("f0_Hz", root.get("f0_Hz", 0.0)))}

    def harmonic_orders(self) -> List[int]:
        root = dict(self._harm().properties) if self.harmonic_file else {}
        try:
            return [int(x) for x in str(root.get("orders", "")).split(",") if x.strip()]
        except ValueError:
            return []

    def summary_html(self) -> str:
        o = self.output()
        ok = lambda b: "<span style='color:#1e8449'>✓</span>" if b else "<span style='color:#c0392b'>✗</span>"
        parts = [f"<b>{self.name}</b>"]
        ok_ = ok
        if self.pq:
            p = self.setup.get("pq", {}) if self.setup else {}
            rows = self.pq_rows()
            ok = [r for r in rows if np.isfinite(r["frequency_Hz"])]
            quad = str(p.get("control", "")).startswith("Phase")
            parts.append(f"<b>Phase quadrature</b>: {len(rows)} levels, "
                         + (f"quadrature of {p.get('response_channel')} at {p.get('target_phase', 90):+g}°" if quad else
                            f"reference-free maximum of {p.get('response_channel')}")
                         + (f"; resonance {min(r['frequency_Hz'] for r in ok):.5g}–{max(r['frequency_Hz'] for r in ok):.5g} Hz"
                            if ok else "") + f", fs = {o.get('fs', 0):g} Hz, N = {o.get('block_size', '?')}")
            parts.append(f"{ok_(bool(self.pq_csv))} resonance table ({len(ok)} converged) · "
                         f"{ok_(bool(self.pq_harm_csv))} harmonics · {ok_(bool(self.pq_blocks))} converged blocks (TDMS) · "
                         f"{ok_('H1' in self.uff)} H1 UFF · {ok_(bool(self.control_files))} control log · "
                         f"{ok_(bool(self.setup))} setup.json")
            return "<br>".join(parts)
        if self.harmonic:
            h = self.setup.get("harmonic", {}) if self.setup else {}
            rows = h.get("harmonics", [])
            spec = ", ".join(f"h{int(r[0])}: {r[1]:g}∠{r[2]:g}°" for r in rows
                             if isinstance(r, (list, tuple)) and len(r) >= 3 and all(isinstance(v, (int, float)) for v in r[:3]))
            ctl = str(h.get("control", ""))
            closed = ctl.startswith("Quasi")
            sv = self.setup.get("save", {}) if self.setup else {}
            nb = len(self.harmonic_blocks())
            try:
                f0 = float(self._harm().properties.get("f0_Hz", h.get("f0", 0)))
            except Exception:
                f0 = h.get("f0", 0)
            parts.append(f"<b>Harmonic</b> f0 = {f0:.6g} Hz, "
                         + (f"Broyden control of {h.get('target_channel')} to [{spec}]" if closed
                            else f"no control, drive [{spec}] V")
                         + f", {h.get('test_blocks', '?')} test blocks, fs = {o.get('fs', 0):g} Hz, "
                           f"N = {o.get('block_size', '?')}")
            parts.append(f"{ok(bool(self.harmonic_file))} time data: {nb} saved block(s) "
                         f"({sv.get('harm_save_mode', '?')}) · {ok(bool(self.control_files))} control log · "
                         f"{ok(bool(self.trigger_csv))} trigger log · {ok(bool(self.setup))} setup.json")
            return "<br>".join(parts)
        if o:
            parts.append(f"{o.get('excitation', '?')}, {o.get('rms', '?')} V RMS, fs = {o.get('fs', '?'):g} Hz, "
                         f"N = {o.get('block_size', '?')}, dwell = {o.get('dwell_blocks', '?')}, "
                         f"realizations = {o.get('realizations', '?')}, band {o.get('fmin', '?'):g}–"
                         f"{o.get('fmax', '?'):g} Hz")
        names = {"Coherence": "Fundamental fraction" if self.stepped else "Coherence"}
        files = " · ".join(f"{ok(f in self.uff)} {names.get(f, f)}" for f in FUNCS)
        conv = self.converged_realizations()
        parts.append(f"FRF files: {files}")
        if self.stepped:
            st = self.setup.get("stepped", {}) if self.setup else {}
            ctl = st.get("control", "")
            parts[1] = (f"<b>Stepped sine</b> {st.get('f_start', '?')}–{st.get('f_stop', '?')} Hz, {ctl}"
                        + (f" of {st.get('target_channel')} to {st.get('target_value', st.get('target_rms'))} "
                           f"{'amplitude' if str(st.get('target_kind', '')).startswith('Amplitude') else 'RMS'}"
                           if ctl.startswith("Secant") else "")
                        + f", fs = {self.output().get('fs', 0):g} Hz, N = {self.output().get('block_size', '?')}")
            dwell = self.converged.endswith("_stepped_sine.tdms")
            parts.append(f"{ok(dwell)} dwell blocks of {len(conv) if dwell else 0} frequencies · "
                         f"{ok(bool(self.stepped_converged))} converged block per frequency · "
                         f"{ok(bool(self.control_files))} control logs · {ok(bool(self.trigger_csv))} trigger log · "
                         f"{ok(bool(self.setup))} setup.json")
            return "<br>".join(parts)
        parts.append(f"{ok(bool(conv))} converged blocks ({len(conv)} realizations) · "
                     f"{ok(bool(self.realization_files))} full realizations ({len(self.realization_files)}) · "
                     f"{ok(bool(self.trigger_csv))} trigger log · {ok(bool(self.setup))} setup.json")
        return "<br>".join(parts)

    # ------------------------------------------------------------------ FRFs
    def uff_sets(self, func: str) -> list:
        if func not in self.uff:
            return []
        if func not in self._uff_cache:
            import pyuff
            d = pyuff.UFF(self.uff[func]).read_sets()
            sets = d if isinstance(d, list) else [d]
            self._uff_cache[func] = [s for s in sets if s.get("type") == 58]
        return self._uff_cache[func]

    def frf_responses(self) -> List[str]:
        """Labels of the response channels, from the first available UFF file."""
        for f in FUNCS:
            sets = self.uff_sets(f)
            if sets:
                return [frf_label(s) for s in sets]
        return []

    def frf(self, index: int) -> Dict:
        """{'freqs': f, 'H1': …, 'H2': …, 'Hv': …, 'Coherence': …, 'units': 'g/N'} for one response."""
        out: Dict = {}
        for f in FUNCS:
            sets = self.uff_sets(f)
            if index < len(sets):
                s = sets[index]
                out[f] = np.asarray(s["data"])
                out.setdefault("freqs", np.asarray(s["x"], dtype=float))
                if f != "Coherence":
                    out["units"] = f"{s.get('ordinate_axis_units_lab', '').strip()}/" \
                                   f"{s.get('orddenom_axis_units_lab', '').strip()}"
        return out

    # ------------------------------------------------------------------ time data
    def _conv(self):
        if self._conv_cache is None:
            from nptdms import TdmsFile
            self._conv_cache = TdmsFile.read(self.converged)
        return self._conv_cache

    def realization(self, r: int, all_blocks: bool) -> Dict:
        """Time data of realization r (1-based). all_blocks: the full dwell history from the
        per-realization file; otherwise the converged block. Returns dict with
        channels [(name, unit, array)], fs, block_size, n_blocks, t0 (s), converged_index."""
        from nptdms import TdmsFile
        if self.stepped:
            group = self._conv()[f"Step_{r:03d}"]
        elif all_blocks:
            group = TdmsFile.read(self.realization_files[r]).groups()[0]
        else:
            group = self._conv()[f"Realization_{r:03d}"]
        chans = []
        fs = None
        for ch in group.channels():
            p = ch.properties
            if fs is None and p.get("wf_increment"):
                fs = 1.0 / float(p["wf_increment"])
            unit = p.get("unit_string", "")
            chans.append((ch.name, unit, np.asarray(ch[:], dtype=float)))
        o = self.output()
        n = int(o.get("block_size") or (chans[0][2].size if chans else 1))
        fs = fs or float(o.get("fs", 1.0))
        n_blocks = max(1, chans[0][2].size // n) if chans else 1
        t0 = float(group.properties.get("first_sample_time_s", 0.0))
        out = {"channels": chans, "fs": fs, "block_size": n, "n_blocks": n_blocks, "t0": t0,
               "converged_index": n_blocks - 1, "dwell": int(o.get("dwell_blocks", n_blocks))}
        if self.stepped:
            f = float(group.properties.get("frequency_Hz", 0))
            out.update({"labels": [f"{f:g} Hz · dwell {i + 1}/{n_blocks}" for i in range(n_blocks)],
                        "converged": [True] * n_blocks, "converged_index": 0, "frequency": f,
                        "line": int(group.properties.get("line", 0)),
                        "status": group.properties.get("status", "")})
        else:
            out.update({"labels": [f"R{r}·{i + 1}/{out['dwell']}" if n_blocks > 1 else f"R{r} converged"
                                   for i in range(n_blocks)],
                        "converged": [i == n_blocks - 1 for i in range(n_blocks)]})
        return out

    def trigger_rows(self) -> List[Dict]:
        if not self.trigger_csv:
            return []
        with open(self.trigger_csv, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


def frf_label(s: Dict) -> str:
    rd = _DIR_NAME.get(int(s.get("rsp_dir", 0)), str(s.get("rsp_dir", "")))
    fd = _DIR_NAME.get(int(s.get("ref_dir", 0)), str(s.get("ref_dir", "")))
    return (f"{s.get('rsp_ent_name', '').strip()} ({s.get('rsp_node')}{rd}) / "
            f"{s.get('ref_ent_name', '').strip()} ({s.get('ref_node')}{fd})")


def detect_suite(folder: str) -> str:
    """'stepped' or 'random' from setup.json (authoritative), otherwise from the file names; '' if unknown."""
    sj = os.path.join(folder, "setup.json")
    if os.path.isfile(sj):
        try:
            with open(sj, encoding="utf-8") as f:
                suite = json.load(f).get("suite", "")
            if suite:
                s = suite.lower()
                return ("stepped" if s.startswith("stepped") else "harmonic" if s.startswith("harmonic")
                        else "quadrature" if s.startswith("phase") else "random")
        except Exception:
            pass
    if glob.glob(os.path.join(folder, "*_harmonic.tdms")):
        return "harmonic"
    if glob.glob(os.path.join(folder, "*_phase_quadrature.csv")):
        return "quadrature"
    if glob.glob(os.path.join(folder, "*_stepped_sine.tdms")) or glob.glob(os.path.join(folder, "*_FundamentalFraction.uff")):
        return "stepped"
    if glob.glob(os.path.join(folder, "*_converged_blocks.tdms")) or glob.glob(os.path.join(folder, "*.uff")):
        return "random"
    return ""


def scan_run(folder: str) -> Run:
    run = Run(folder=folder)
    sj = os.path.join(folder, "setup.json")
    if os.path.isfile(sj):
        try:
            with open(sj, encoding="utf-8") as f:
                run.setup = json.load(f)
        except Exception:
            run.setup = {}
    for f in FUNCS:
        hits = sorted(glob.glob(os.path.join(folder, f"*_{f}.uff")))
        if hits:
            run.uff[f] = hits[0]
            run.base = run.base or os.path.basename(hits[0])[: -len(f"_{f}.uff")]
    stepped_conv = sorted(glob.glob(os.path.join(folder, "*_stepped_converged_blocks.tdms")))
    hits = sorted(p for p in glob.glob(os.path.join(folder, "*_converged_blocks.tdms")) if p not in stepped_conv)
    run.converged = hits[0] if hits else ""
    run.stepped_converged = stepped_conv[0] if stepped_conv else ""
    hits = sorted(glob.glob(os.path.join(folder, "*_stepped_sine.tdms")))
    if hits:                                      # stepped: all dwell blocks (preferred time data)
        run.converged, run.stepped = hits[0], True
    elif stepped_conv:                            # stepped: only the converged block per frequency
        run.converged, run.stepped = stepped_conv[0], True
    ff = sorted(glob.glob(os.path.join(folder, "*_FundamentalFraction.uff")))
    if ff and "Coherence" not in run.uff:
        run.uff["Coherence"] = ff[0]          # stepped sine: η takes the coherence slot
        run.base = run.base or os.path.basename(ff[0])[: -len("_FundamentalFraction.uff")]
    if detect_suite(folder) == "stepped":
        run.stepped = True
    if run.stepped:
        run.quality_label = "Fundamental fraction η"
    run.control_files = sorted(glob.glob(os.path.join(folder, "*_control_*.csv")))
    for p in glob.glob(os.path.join(folder, "*_realization_*.tdms")):
        m = _REAL_RE.search(p)
        if m:
            run.realization_files[int(m.group(1))] = p
    hits = sorted(glob.glob(os.path.join(folder, "*_trigger_log.csv")))
    run.trigger_csv = hits[0] if hits else ""
    hits = sorted(glob.glob(os.path.join(folder, "*_harmonics.csv")))
    run.harmonics_csv = hits[0] if hits else ""
    hits = sorted(glob.glob(os.path.join(folder, "*_phase_quadrature.csv")))
    if hits:
        run.pq, run.stepped, run.converged = True, False, ""
        run.pq_csv = hits[0]
        run.base = os.path.basename(hits[0])[: -len("_phase_quadrature.csv")]
        hh = sorted(glob.glob(os.path.join(folder, "*_pq_harmonics.csv")))
        run.pq_harm_csv = hh[0] if hh else ""
        tb = sorted(glob.glob(os.path.join(folder, "*_phase_quadrature_blocks.tdms")))
        run.pq_blocks = tb[0] if tb else ""
        run.quality_label = "—"
    hits = sorted(glob.glob(os.path.join(folder, "*_harmonic.tdms")))
    if hits or detect_suite(folder) == "harmonic":
        run.harmonic, run.stepped, run.converged = True, False, ""
        run.harmonic_file = hits[0] if hits else ""
        run.base = run.base or (os.path.basename(hits[0])[: -len("_harmonic.tdms")] if hits else "")
    return run


def _is_run(folder: str) -> bool:
    return any(glob.glob(os.path.join(folder, pat)) for pat in
               ("*.uff", "*_converged_blocks.tdms", "*_realization_*.tdms", "*_stepped_sine.tdms",
                "*_stepped_converged_blocks.tdms", "*_harmonic.tdms", "*_phase_quadrature.csv"))


def find_runs(folder: str) -> List[str]:
    """The folder itself if it is a run, otherwise its run sub-folders (newest first)."""
    if not folder or not os.path.isdir(folder):
        return []
    if _is_run(folder):
        return [folder]
    subs = [os.path.join(folder, d) for d in os.listdir(folder)]
    runs = [d for d in subs if os.path.isdir(d) and _is_run(d)]
    return sorted(runs, key=lambda d: os.path.basename(d), reverse=True)


def amplitude_spectrum(x: np.ndarray, fs: float):
    """Single-sided amplitude spectrum (rectangular window), same units as x. DC excluded."""
    n = x.size
    X = np.abs(np.fft.rfft(x)) * 2.0 / n
    f = np.arange(X.size) * fs / n
    return f[1:], X[1:]


# ============================================================================= re-processing
# A "channel row" is a plain dict:
#   key (original TDMS / UFF name), name, units, sensor_type, node, direction, reference (bool),
#   use (bool: include as a response), physical, sensitivity
_SKIP_CHANNELS = ("Trigger",)


def channel_rows(run: Run) -> List[Dict]:
    """Editable channel list of a run. From the converged-block TDMS when present (then the
    reference can be changed and FRFs recomputed), otherwise from the stored UFF files."""
    rows: List[Dict] = []
    if run.converged:
        groups = run._conv().groups()
        if groups:
            for ch in groups[0].channels():
                if ch.name in _SKIP_CHANNELS:
                    continue
                p = ch.properties
                drive = ch.name == "Drive"
                rows.append({
                    "key": ch.name, "name": ch.name, "units": p.get("unit_string", "V" if drive else ""),
                    "sensor_type": "Voltage" if drive else p.get("sensor_type", "Voltage"),
                    "node": int(p.get("node", 0)), "direction": p.get("direction", "+Z"),
                    "reference": bool(p.get("is_reference", False)), "use": not drive,
                    "physical": p.get("physical_channel", ""), "sensitivity": float(p.get("sensitivity", 0) or 0),
                })
            if not any(r["reference"] for r in rows) and rows:
                rows[0]["reference"] = True
            for r in rows:
                if r["reference"]:
                    r["use"] = False
            return rows
    # UFF only: responses + the (fixed) reference
    sets = next((run.uff_sets(f) for f in FUNCS if run.uff_sets(f)), [])
    for s in sets:
        rows.append({"key": s.get("rsp_ent_name", "").strip(), "name": s.get("rsp_ent_name", "").strip(),
                     "units": s.get("ordinate_axis_units_lab", "").strip(),
                     "sensor_type": _type_from_spec(s.get("ordinate_spec_data_type", 1)),
                     "node": int(s.get("rsp_node", 0)), "direction": _DIR_NAME.get(int(s.get("rsp_dir", 3)), "+Z"),
                     "reference": False, "use": True, "physical": "", "sensitivity": 0.0})
    if sets:
        s = sets[0]
        rows.insert(0, {"key": s.get("ref_ent_name", "").strip(), "name": s.get("ref_ent_name", "").strip(),
                        "units": s.get("orddenom_axis_units_lab", "").strip(),
                        "sensor_type": _type_from_spec(s.get("orddenom_spec_data_type", 13)),
                        "node": int(s.get("ref_node", 0)), "direction": _DIR_NAME.get(int(s.get("ref_dir", 3)), "+Z"),
                        "reference": True, "use": False, "physical": "", "sensitivity": 0.0})
    return rows


def _type_from_spec(code) -> str:
    return {12: "Accelerometer (IEPE)", 13: "Force (IEPE)", 3: "Strain gauge (bridge)",
            15: "Microphone (IEPE)"}.get(int(code or 1), "Voltage")


def can_recompute(run: Run) -> bool:
    return bool(run.converged and run.converged_realizations())


def recompute_frf(run: Run, rows: List[Dict]):
    """H1/H2/Hv/coherence from the converged blocks, using the rows' reference and the rows
    marked `use` as responses, over the full 0…fs/2 range. Returns (FRFResult, resp_rows, ref_row)."""
    from .frf import FRFEstimator
    ref = next(r for r in rows if r["reference"])
    resp = [r for r in rows if r["use"] and not r["reference"]]
    if not resp:
        raise ValueError("Tick at least one response channel (other than the reference).")
    conv = run._conv()
    reals = run.converged_realizations()
    if run.stepped:
        return _recompute_stepped(run, ref, resp), resp, ref
    first = conv[f"Realization_{reals[0]:03d}"]
    n = len(first[ref["key"]][:])
    inc = first[ref["key"]].properties.get("wf_increment")
    fs = 1.0 / float(inc) if inc else float(run.output().get("fs", 1.0))
    est = FRFEstimator(len(resp), n, fs, run.output().get("window", "Rectangular"))
    for r in reals:
        g = conv[f"Realization_{r:03d}"]
        est.add(np.asarray(g[ref["key"]][:], float), np.vstack([np.asarray(g[x["key"]][:], float) for x in resp]))
    return est.result(), resp, ref


def _recompute_stepped(run: Run, ref: Dict, resp: List[Dict]):
    """Single-line estimates per step from its dwell blocks (one block = whole cycles)."""
    from .frf import FRFResult, estimates
    conv = run._conv()
    n = int(run.output().get("block_size") or 0)
    freqs, cols = [], {"H1": [], "H2": [], "Hv": [], "coh": []}
    harm_rows = []
    for r in run.converged_realizations():
        g = conv[f"Step_{r:03d}"]
        k = int(g.properties.get("line", 0))
        x = np.asarray(g[ref["key"]][:], float)
        nb_n = n or x.size
        if k <= 0:
            continue
        e = np.exp(-2j * np.pi * k * np.arange(nb_n) / nb_n)
        X = lambda sig: 2.0 / nb_n * np.asarray(sig, float).reshape(-1, nb_n) @ e
        xr = X(x)
        ys = [np.asarray(g[c["key"]][:], float) for c in resp]
        yr = np.stack([X(y) for y in ys], axis=1)                          # (blocks, n_resp)
        Gxx = np.mean(np.abs(xr) ** 2)
        Gyy = np.mean(np.abs(yr) ** 2, axis=0)
        Gxy = np.mean(np.conj(xr)[:, None] * yr, axis=0)
        H1, H2, Hv, _ = estimates(np.array([Gxx]), Gyy[:, None], Gxy[:, None])

        def eta(sig, Xl):                       # fundamental fraction, averaged over blocks
            b = np.asarray(sig, float).reshape(-1, nb_n)
            p_tot = np.mean((b - b.mean(axis=1, keepdims=True)) ** 2, axis=1)
            return float(np.clip(np.mean(np.abs(Xl) ** 2 / 2 / (p_tot + 1e-300)), 0, 1))
        e_ref = eta(x, xr)
        e_frf = np.array([e_ref * eta(y, yr[:, i]) for i, y in enumerate(ys)])
        freqs.append(float(g.properties.get("frequency_Hz", 0)))
        for name, v in (("H1", H1[:, 0]), ("H2", H2[:, 0]), ("Hv", Hv[:, 0]), ("coh", e_frf)):
            cols[name].append(v)
        harm_rows.append(_harmonics_of([x] + ys, k, nb_n, n_harmonics(run)))
    arr = {k: np.array(v).T for k, v in cols.items()}
    return FRFResult(np.array(freqs), arr["H1"], arr["H2"], arr["Hv"], arr["coh"].real, len(freqs),
                     harm=np.array(harm_rows), ref_i=0, resp_i=list(range(1, len(resp) + 1)))


N_HARM = 5          # default for runs whose setup does not say


def n_harmonics(run) -> int:
    """Harmonic orders 1…n for a stepped run: the run's own setting (Output Setup), else the default."""
    st = (getattr(run, "setup", None) or {}).get("stepped", {}) or {}
    n = st.get("n_harmonics") or (run.setup or {}).get("save", {}).get("harmonics_order") or N_HARM
    return max(2, int(n))


def _harmonics_of(signals, k: int, n: int, n_h: int = N_HARM) -> np.ndarray:
    """(n_signals, n_h) complex harmonic amplitudes, averaged over the blocks of each signal."""
    out = np.full((len(signals), n_h), np.nan + 0j)
    idx = np.arange(n)
    for h in range(1, n_h + 1):
        if h * k >= n // 2:
            break
        e = np.exp(-2j * np.pi * h * k * idx / n)
        for i, sig in enumerate(signals):
            out[i, h - 1] = (2.0 / n * np.asarray(sig, float).reshape(-1, n) @ e).mean()
    return out


def harmonics_from_csv(path: str) -> Dict[str, Dict]:
    """Stepped sine: complex harmonics per channel from <base>_harmonics.csv, in the same form as
    amplitude_spectra(): {channel: {'freqs', 'harm' (nf, H) complex, 'amp', 'units', 'source'}}.
    Uses the re/im columns; files written before those existed are rebuilt from amplitude ∠ phase."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                if r.get("re", "") != "" and r.get("im", "") != "":
                    z = complex(float(r["re"]), float(r["im"]))
                else:
                    z = float(r["amplitude"]) * np.exp(1j * np.radians(float(r["phase_deg"])))
                key = (int(r["level"]), float(r["frequency_Hz"])) if r.get("level") else float(r["frequency_Hz"])
                rows.append((key, r["channel"], r.get("units", ""), int(r["order"]), z))
            except (KeyError, ValueError):
                continue
    if not rows:
        return {}
    freqs = sorted({r[0] for r in rows})
    fi = {f: i for i, f in enumerate(freqs)}
    H = max(r[3] for r in rows)
    names, units = [], {}
    for r in rows:
        if r[1] not in units:
            names.append(r[1])
            units[r[1]] = r[2]
    out: Dict[str, Dict] = {}
    for nm in names:
        harm = np.full((len(freqs), H), np.nan + 0j)
        for f, ch, _, h, z in rows:
            if ch == nm:
                harm[fi[f], h - 1] = z
        fr = np.array([k[1] if isinstance(k, tuple) else k for k in freqs], float)
        out[nm] = {"freqs": fr, "harm": harm, "amp": np.abs(harm[:, 0]), "units": units[nm], "source": "csv"}
    return out


def amplitude_spectra(run: Run) -> Dict[str, Dict]:
    """Raw frequency-domain amplitudes of every channel (Trigger excluded), from the TDMS file.
    Random runs: single-sided amplitude spectrum, RMS-averaged over the converged blocks of all
    realizations (rectangular window), Δf … fs/2.
    Stepped runs: amplitude of harmonic h at every measured frequency ('harm', complex (nf, H))."""
    if getattr(run, "_spec_cache", None) is not None:
        return run._spec_cache
    out: Dict[str, Dict] = {}
    if run.pq:                                         # phase quadrature: harmonics at every level's resonance
        out = harmonics_from_csv(run.pq_harm_csv) if run.pq_harm_csv else {}
        run._spec_cache = out
        return out
    if not (run.converged and run.converged_realizations()):
        if run.stepped and run.harmonics_csv:          # no time data, but the harmonics were saved
            out = harmonics_from_csv(run.harmonics_csv)
        run._spec_cache = out
        return out
    conv = run._conv()
    reals = run.converged_realizations()
    prefix = "Step" if run.stepped else "Realization"
    first = conv[f"{prefix}_{reals[0]:03d}"]
    names = [c.name for c in first.channels() if c.name not in _SKIP_CHANNELS]
    units = {c.name: c.properties.get("unit_string", "") for c in first.channels()}
    inc = first.channels()[0].properties.get("wf_increment")
    fs = 1.0 / float(inc) if inc else float(run.output().get("fs", 1.0))
    n = int(run.output().get("block_size") or len(first.channels()[0]))
    if run.stepped:
        f0, rows = [], []
        for r in reals:
            g = conv[f"Step_{r:03d}"]
            k = int(g.properties.get("line", 0))
            if k <= 0:
                continue
            f0.append(float(g.properties.get("frequency_Hz", 0)))
            rows.append(_harmonics_of([g[nm][:] for nm in names], k, n, n_harmonics(run)))
        harm = np.array(rows)                                   # (nf, nch, H)
        for i, nm in enumerate(names):
            out[nm] = {"freqs": np.array(f0), "harm": harm[:, i, :], "amp": np.abs(harm[:, i, 0]),
                       "units": units.get(nm, "")}
    else:
        acc = {nm: np.zeros(n // 2 + 1) for nm in names}
        for r in reals:
            g = conv[f"Realization_{r:03d}"]
            for nm in names:
                acc[nm] += np.abs(np.fft.rfft(np.asarray(g[nm][:], float)[:n])) ** 2
        f = np.arange(n // 2 + 1) * fs / n
        for nm in names:
            out[nm] = {"freqs": f[1:], "amp": np.sqrt(acc[nm][1:] / len(reals)) * 2.0 / n, "units": units.get(nm, "")}
    run._spec_cache = out
    return out


def frf_table(run: Run, prefer_tdms: bool = True) -> Dict[str, Dict]:
    """FRFs of a run keyed by response name: freqs, H1/H2/Hv, q (coherence or η), units,
    and for recomputed stepped runs 'res' + 'j' (for harmonic FRFs)."""
    key = ("_frf_cache", prefer_tdms)
    cache = getattr(run, "_frf_tables", {})
    if key in cache:
        return cache[key]
    out: Dict[str, Dict] = {}
    if run.pq:                     # H1 at the resonance of every level (level order: the backbone path)
        rows = [r for r in run.pq_rows() if r["H1"] is not None and np.isfinite(r["frequency_Hz"])]
        if rows:
            r0 = rows[0]
            out[r0["resp_channel"]] = {"freqs": np.array([r["frequency_Hz"] for r in rows]),
                                       "H1": np.array([r["H1"] for r in rows]), "H2": None, "Hv": None, "q": None,
                                       "units": f"{r0['resp_units']}/{r0['ref_units']}", "res": None, "j": 0,
                                       "ref": r0["ref_channel"], "pq": True}
        cache[key] = out
        run._frf_tables = cache
        return out
    if prefer_tdms and can_recompute(run):
        rows = channel_rows(run)
        try:
            res, resp, ref = recompute_frf(run, rows)
            for j, r in enumerate(resp):
                out[r["name"]] = {"freqs": res.freqs, "H1": res.H1[j], "H2": res.H2[j], "Hv": res.Hv[j],
                                  "q": res.coherence[j], "units": f"{r['units']}/{ref['units']}",
                                  "res": res, "j": j, "ref": ref["name"]}
        except Exception:
            out = {}
    if not out:
        for i, lbl in enumerate(run.frf_responses()):
            d = run.frf(i)
            name = lbl.split(" (")[0]
            out[name] = {"freqs": d.get("freqs"), "H1": d.get("H1"), "H2": d.get("H2"), "Hv": d.get("Hv"),
                         "q": d.get("Coherence"), "units": d.get("units", ""), "res": None, "j": i,
                         "ref": lbl.split("/ ")[-1].split(" (")[0]}
    cache[key] = out
    run._frf_tables = cache
    return out


def band_of(run: Run):
    """Band markers: stepped sine → start…stop frequency; random → the output band."""
    if run.stepped:
        st = (run.setup or {}).get("stepped", {})
        if "f_start" in st:
            return float(st["f_start"]), float(st["f_stop"])
        return None
    o = run.output()
    return (float(o["fmin"]), float(o["fmax"])) if "fmin" in o else None


def stored_frf_arrays(run: Run) -> Dict[str, Dict[str, np.ndarray]]:
    """{func: {response key: data}} plus 'freqs', from the stored UFF files."""
    out: Dict = {}
    for f in FUNCS:
        for s in run.uff_sets(f):
            out.setdefault(f, {})[s.get("rsp_ent_name", "").strip()] = np.asarray(s["data"])
            out.setdefault("freqs", np.asarray(s["x"], float))
    return out


def _cfg(row: Dict):
    from .config import ChannelConfig
    return ChannelConfig(name=row["name"], node=int(row["node"]), direction=row["direction"],
                         sensor_type=row["sensor_type"], units=row["units"], physical=row["physical"],
                         sensitivity=row["sensitivity"] or 0.0, is_reference=row["reference"])


def _rewrite_tdms(src: str, dst: str, rows: List[Dict], extra_root: Dict) -> None:
    """Copy a TDMS file, applying new channel names / node / direction / reference flags."""
    from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter
    by_key = {r["key"]: r for r in rows}
    tf = TdmsFile.read(src)
    with TdmsWriter(dst) as w:
        w.write_segment([RootObject(properties={**tf.properties, **extra_root})])
        for g in tf.groups():
            objs = [GroupObject(g.name, properties=dict(g.properties))]
            for ch in g.channels():
                props = dict(ch.properties)
                name = ch.name
                r = by_key.get(ch.name)
                if r is not None:
                    name = r["name"]
                    props.update({"NI_ChannelName": r["name"], "node": int(r["node"]),
                                  "direction": r["direction"], "is_reference": bool(r["reference"])})
                objs.append(ChannelObject(g.name, name, np.asarray(ch[:]), properties=props))
            w.write_segment(objs)


def save_reprocessed(run: Run, rows: List[Dict], out_folder: str, include_realizations: bool = True,
                     log=lambda s: None) -> List[str]:
    """Write a complete, self-consistent copy of the run with the edited channel table:
    UFF H1/H2/Hv/Coherence (recomputed from TDMS when possible, else relabelled stored FRFs),
    TDMS files with updated metadata, an updated setup.json, and the trigger log."""
    import datetime as _dt
    import shutil
    from . import export
    os.makedirs(out_folder, exist_ok=True)
    base = run.base or "modal_test"
    files: List[str] = []
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note = {"reprocessed_from": run.folder, "reprocessed_at": stamp}

    ref_row = next(r for r in rows if r["reference"])
    resp_rows = [r for r in rows if r["use"] and not r["reference"]]
    o = run.output()
    info = (f"Reprocessed {stamp[:10]}, {o.get('excitation', '')}, fs={o.get('fs', '')}Hz, "
            f"N={o.get('block_size', '')}")

    # ---- FRFs
    if can_recompute(run):
        res, resp_rows, ref_row = recompute_frf(run, rows)
        sl = slice(None) if run.stepped else slice(1, None)        # random: drop DC
        f = res.freqs[sl]
        info += f", {res.n_avg} {'frequencies' if run.stepped else 'avg'}"
        data = {"H1": res.H1[:, sl], "H2": res.H2[:, sl], "Hv": res.Hv[:, sl], "Coherence": res.coherence[:, sl]}
        log(f"Recomputed FRFs from {res.n_avg} {'stepped-sine frequencies' if run.stepped else 'converged blocks'}, "
            f"reference {ref_row['name']}.")
    else:
        stored = stored_frf_arrays(run)
        f = stored["freqs"]
        data = {k: np.vstack([stored[k][r["key"]] for r in resp_rows]) for k in FUNCS if k in stored}
        log("No converged-block TDMS: stored FRFs relabelled (reference unchanged).")
    for func, arr in data.items():
        if func == "Coherence" and run.stepped:
            func = "FundamentalFraction"
        p = os.path.join(out_folder, f"{base}_{func}.uff")
        export.write_uff58(p, func, f, arr, [_cfg(r) for r in resp_rows], _cfg(ref_row), info)
        files.append(p)
        log(f"Wrote {os.path.basename(p)}")

    # ---- TDMS with updated metadata
    for src in dict.fromkeys(x for x in (run.converged, run.stepped_converged) if x):
        p = os.path.join(out_folder, os.path.basename(src))
        _rewrite_tdms(src, p, rows, note)
        files.append(p)
        log(f"Wrote {os.path.basename(p)}")
    if include_realizations:
        for r, src in sorted(run.realization_files.items()):
            p = os.path.join(out_folder, os.path.basename(src))
            _rewrite_tdms(src, p, rows, note)
            files.append(p)
        if run.realization_files:
            log(f"Wrote {len(run.realization_files)} realization TDMS files")

    # ---- setup.json and trigger log
    setup = json.loads(json.dumps(run.setup or {}))
    by_key = {r["key"]: r for r in rows}
    for c in setup.get("channels", []):
        r = by_key.get(c.get("name"))
        if r is not None:
            c.update({"name": r["name"], "node": int(r["node"]), "direction": r["direction"],
                      "is_reference": bool(r["reference"])})
        elif c.get("is_reference"):
            c["is_reference"] = False
    setup["reprocessing"] = {**note, "reference": ref_row["name"],
                             "responses": [r["name"] for r in resp_rows]}
    p = os.path.join(out_folder, "setup.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(setup, fh, indent=2)
    files.append(p)
    for src in ([run.trigger_csv] if run.trigger_csv else []) + list(run.control_files):
        p = os.path.join(out_folder, os.path.basename(src))
        shutil.copy2(src, p)
        files.append(p)
    return files
