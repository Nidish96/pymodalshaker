# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Data Viewer tab: browse a saved run folder, plot FRFs/coherence from the UFF files and the
time blocks (converged or all dwell blocks) of any realization with their spectra; export images."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
import datetime as _dt

from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton,
                               QVBoxLayout, QWidget)

from ..config import DIRECTIONS
from ..results import (Run, amplitude_spectra, amplitude_spectrum, band_of, can_recompute, channel_rows,
                       detect_suite, find_runs, n_harmonics, recompute_frf, save_reprocessed, scan_run)
from .compare import CompareTab
from .harmonic_run_view import HarmonicRunView
from .pq_run_view import PQRunView
from .strobed_view import StrobedView
from .harmonic_sweep import HarmonicSweepView
from .harmonics_view import HarmonicsView
from .plots import COLOR_A, COLOR_B, EST_COLORS, DualAxisPlot, FRFStack, export_dialog

NONE_ITEM = "(none)"
SRC_COMPUTED = "Recomputed from TDMS"
SRC_STORED = "Stored UFF files"
CH_COLS = ["Response", "Reference", "Name", "Node", "Direction", "Units", "Sensor type", "Saved as"]


class ViewerTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.run: Optional[Run] = None
        self._data = None          # currently loaded realization dict

        # ---------------- folder / run selection
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("A run folder, or a folder containing run folders")
        self.folder.returnPressed.connect(self.rescan)
        browse = QToolButton()
        browse.setText("…")
        browse.clicked.connect(self._browse)
        rescan = QPushButton("↻ Scan")
        rescan.clicked.connect(self.rescan)
        self.run_sel = QComboBox()
        self.run_sel.currentIndexChanged.connect(self._load_run)
        self.summary = QLabel("Choose a folder to view saved results.")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.RichText)
        self.summary.setStyleSheet("QLabel { background: palette(alternate-base); padding: 6px; }")
        top = QHBoxLayout()
        top.addWidget(QLabel("<b>Folder</b>"))
        top.addWidget(self.folder, 1)
        top.addWidget(browse)
        top.addWidget(rescan)
        top.addWidget(QLabel("Run"))
        top.addWidget(self.run_sel, 1)

        # ---------------- FRF sub-tab
        self.frf_sel = QComboBox()
        self.frf_sel.currentIndexChanged.connect(self._draw_frf)
        self.est_cb = {k: QCheckBox(k) for k in ("H1", "H2", "Hv")}
        for k, cb in self.est_cb.items():
            cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color: {EST_COLORS[k]}; font-weight: bold }}")
            cb.toggled.connect(self._draw_frf)
        self.frf_info = QLabel()
        self.frf_src = QComboBox()
        self.frf_src.setToolTip("Computed: H1/H2/Hv/γ² recalculated from the converged blocks with the "
                                "reference and labels set on the 'Channels & re-processing' tab.\n"
                                "Stored: exactly what is in the run's UFF files.")
        self.frf_src.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.frf_src.setMinimumContentsLength(30)
        self.frf_src.currentIndexChanged.connect(self._fill_frf_list)
        self.frf_plot = FRFStack()
        self.qty = QComboBox()
        self.qty.addItems(["FRF", "Amplitude spectrum"])
        self.qty.setToolTip("FRF: response / reference.\nAmplitude spectrum: the raw frequency-domain amplitude of "
                            "one channel (random: RMS-averaged over the realizations' converged blocks; stepped sine: "
                            "amplitude of the chosen harmonic at every excitation frequency).")
        self.qty.currentIndexChanged.connect(self._fill_frf_list)
        exp1 = QPushButton("📷 Export image…")
        exp1.clicked.connect(lambda: export_dialog(self, self.frf_plot, self._img_name("FRF")))
        fh = QHBoxLayout()
        fh.addWidget(QLabel("Show"))
        fh.addWidget(self.qty)
        self.src_label = QLabel("  Source")
        fh.addWidget(self.src_label)
        fh.addWidget(self.frf_src)
        self.item_label = QLabel("  Response")
        fh.addWidget(self.item_label)
        fh.addWidget(self.frf_sel, 1)
        for cb in self.est_cb.values():
            fh.addWidget(cb)
        fh.addWidget(self.frf_info)
        fh.addWidget(exp1)
        frf_page = QWidget()
        fl = QVBoxLayout(frf_page)
        fl.addLayout(fh)
        fl.addWidget(self.frf_plot, 1)

        # ---------------- time-block sub-tab
        self.real_sel = QComboBox()
        self.real_sel.setToolTip("Realization to show")
        self.src_sel = QComboBox()
        self.src_sel.addItems(["Converged block", "All dwell blocks"])
        self.src_sel.setToolTip("All dwell blocks needs 'Every realization → its own TDMS file' to have been saved")
        self.ch_a, self.ch_b = QComboBox(), QComboBox()
        self.ch_a.setStyleSheet(f"QComboBox {{ color: {COLOR_A} }}")
        self.ch_b.setStyleSheet(f"QComboBox {{ color: {COLOR_B} }}")
        self.spec_sel = QComboBox()
        self.spec_sel.setToolTip("Which block the spectrum is computed from")
        self.spec_log = QCheckBox("log")
        self.spec_log.setChecked(True)
        for w in (self.real_sel, self.src_sel):
            w.currentIndexChanged.connect(self._load_realization)
        for w in (self.ch_a, self.ch_b, self.spec_sel):
            w.currentIndexChanged.connect(self._draw_time)
        self.spec_log.toggled.connect(self._draw_time)
        self.t_plot = DualAxisPlot(downsample=True)
        self.t_plot.setLabel("bottom", "Time from start of acquisition", "s")
        self.s_plot = DualAxisPlot()
        self.s_plot.setLabel("bottom", "Frequency", "Hz")
        self.s_title = QLabel()
        exp2 = QPushButton("📷 Export image…")
        tplots = QSplitter(Qt.Vertical)
        tw = QWidget()
        twl = QVBoxLayout(tw)
        twl.setContentsMargins(0, 0, 0, 0)
        self.t_title = QLabel()
        twl.addWidget(self.t_title)
        twl.addWidget(self.t_plot, 1)
        sw = QWidget()
        swl = QVBoxLayout(sw)
        swl.setContentsMargins(0, 0, 0, 0)
        swl.addWidget(self.s_title)
        swl.addWidget(self.s_plot, 1)
        self.harm = HarmonicsView()
        self.harm.set_orientation(True)
        self.harm.setVisible(False)
        self.lower = QComboBox()
        self.lower.addItems(["Harmonics", "Spectrum", "Strobed view"])
        self.lower.setToolTip("Stepped sine: harmonic decomposition (Fourier series in the excitation frequency) "
                              "or the full amplitude spectrum of the chosen block")
        self.lower.currentIndexChanged.connect(self._draw_time)
        self.spec_widget = sw
        tplots.addWidget(tw)
        tplots.addWidget(sw)
        tplots.addWidget(self.harm)
        self.v_strobe = StrobedView()
        self.v_strobe.setVisible(False)
        self.v_strobe.n_over.setValue(100)
        self.v_strobe.n_over.valueChanged.connect(self._draw_time)
        tplots.addWidget(self.v_strobe)
        self.time_plots = tplots
        exp2.clicked.connect(lambda: export_dialog(self, self.time_plots, self._img_name("time")))
        th = QHBoxLayout()
        self.real_label = QLabel("Realization")
        th.addWidget(self.real_label)
        th.addWidget(self.real_sel)
        th.addWidget(self.src_sel)
        th.addWidget(QLabel("  A"))
        th.addWidget(self.ch_a, 1)
        th.addWidget(QLabel("B"))
        th.addWidget(self.ch_b, 1)
        self.lower_label = QLabel("  Show")
        th.addWidget(self.lower_label)
        th.addWidget(self.lower)
        th.addWidget(QLabel(" of"))
        th.addWidget(self.spec_sel)
        th.addWidget(self.spec_log)
        th.addWidget(exp2)
        time_page = QWidget()
        tl = QVBoxLayout(time_page)
        tl.addLayout(th)
        tl.addWidget(tplots, 1)

        # ---------------- channels & re-processing sub-tab
        self.ch_table = QTableWidget(0, len(CH_COLS))
        self.ch_table.setHorizontalHeaderLabels(CH_COLS)
        self.ch_table.verticalHeader().setDefaultSectionSize(28)
        self.ch_table.setSelectionMode(QAbstractItemView.NoSelection)
        hh = self.ch_table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setStretchLastSection(True)
        for i, wdt in enumerate([75, 75, 170, 80, 85, 70, 170]):
            self.ch_table.setColumnWidth(i, wdt)
        self.ch_mode = QLabel()
        self.ch_mode.setWordWrap(True)
        self.btn_apply = QPushButton("✓ Apply — recompute FRFs")
        self.btn_apply.setToolTip("Recompute H1/H2/Hv/coherence with this reference and these labels")
        self.btn_revert = QPushButton("↺ Revert to saved")
        self.btn_save = QPushButton("💾 Save re-processed copy…")
        self.btn_save.setToolTip("Write a new run folder: UFF FRFs, TDMS files with the new labels, "
                                 "updated setup.json. The original run is not modified.")
        self.inc_real = QCheckBox("include per-realization TDMS files")
        self.inc_real.setChecked(True)
        self.ch_status = QLabel()
        self.ch_status.setWordWrap(True)
        self.btn_apply.clicked.connect(self._apply_rows)
        self.btn_revert.clicked.connect(self._revert_rows)
        self.btn_save.clicked.connect(self._save_reprocessed)
        bl = QHBoxLayout()
        for b in (self.btn_apply, self.btn_revert):
            bl.addWidget(b)
        bl.addStretch()
        bl.addWidget(self.inc_real)
        bl.addWidget(self.btn_save)
        ch_page = QWidget()
        cl = QVBoxLayout(ch_page)
        cl.addWidget(self.ch_mode)
        cl.addWidget(self.ch_table, 1)
        cl.addLayout(bl)
        cl.addWidget(self.ch_status)
        self.rows: list = []
        self.computed = None

        self.sub = QTabWidget()
        self.compare = CompareTab()
        self.sweep = HarmonicSweepView()
        self.sub.addTab(frf_page, "Frequency domain")
        self.sub.addTab(time_page, "Realizations — time && spectrum (TDMS)")
        self.sub.addTab(ch_page, "Channels && re-processing")
        self.sub.addTab(self.sweep, "Harmonics (stepped sine)")
        self.hrun = HarmonicRunView()
        self.sub.addTab(self.hrun, "Blocks — time && strobed view (harmonic)")
        self.pqrun = PQRunView()
        self.sub.addTab(self.pqrun, "Resonance vs amplitude (phase quadrature)")
        self.sub.addTab(self.compare, "Compare runs")
        self.HARM_TAB = 3
        self.HRUN_TAB = 4
        self.PQ_TAB = 5

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.summary)
        lay.addWidget(self.sub, 1)

    # ------------------------------------------------------------------ folder / run
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Results folder", self.folder.text() or os.getcwd())
        if d:
            self.folder.setText(d)
            self.rescan()

    def open_folder(self, folder: str):
        self.folder.setText(folder)
        self.rescan()

    def rescan(self):
        runs = find_runs(self.folder.text().strip())
        self.compare.set_runs(runs)
        self.run_sel.blockSignals(True)
        self.run_sel.clear()
        for r in runs:
            suite = detect_suite(r)
            self.run_sel.addItem((f"[{suite}] " if suite else "") + os.path.basename(os.path.normpath(r)), r)
        self.run_sel.blockSignals(False)
        if not runs:
            self.run = None
            self.summary.setText("<span style='color:#c0392b'>No saved runs (UFF / TDMS files) found in this folder.</span>")
            self.frf_sel.clear()
            self.real_sel.clear()
            self.frf_plot.clear()
            return
        self._load_run()

    def _load_run(self, *_):
        path = self.run_sel.currentData()
        if not path:
            return
        self.run = run = scan_run(path)
        badge = ("<span style='background:#d35400; color:white; padding:2px 6px'><b>PHASE QUADRATURE</b></span>"
                 if run.pq else
                 "<span style='background:#16a085; color:white; padding:2px 6px'><b>HARMONIC</b></span>"
                 if run.harmonic else
                 "<span style='background:#8e44ad; color:white; padding:2px 6px'><b>STEPPED SINE</b></span>"
                 if run.stepped else
                 "<span style='background:#2b7bb9; color:white; padding:2px 6px'><b>RANDOM EXCITATION</b></span>")
        self.summary.setText(f"Suite (detected): {badge}<br>" + run.summary_html())
        # harmonic runs hold time data only: their own page, no FRF / realization / channel pages
        was_hrun = self.sub.currentIndex() in (self.HRUN_TAB, self.PQ_TAB)
        own = run.harmonic or run.pq
        for i in range(self.HRUN_TAB):
            self.sub.setTabVisible(i, not own and (i != self.HARM_TAB or run.stepped))
        self.sub.setTabVisible(self.HRUN_TAB, run.harmonic)
        self.sub.setTabVisible(self.PQ_TAB, run.pq)
        if run.pq:
            self.sub.setCurrentIndex(self.PQ_TAB)
            try:
                self.pqrun.set_run(run)
            except Exception as e:
                self.pqrun.results.info.setText(f"<span style='color:#c0392b'>Could not read this run: "
                                                f"{type(e).__name__}: {e}</span>")
            return
        if run.harmonic:
            self.sub.setCurrentIndex(self.HRUN_TAB)
            try:
                n_h = int((run.setup or {}).get("harmonic", {}).get("n_harmonics", 5))
                self.hrun.set_run(run, n_h)
            except Exception as e:
                self.hrun.t_title.setText(f"<span style='color:#c0392b'>Could not read the time data: "
                                          f"{type(e).__name__}: {e}</span>")
            return
        if was_hrun or not self.sub.isTabVisible(self.sub.currentIndex()):
            self.sub.setCurrentIndex(0)
        self.sub.setTabVisible(self.HARM_TAB, run.stepped)
        self.frf_plot.set_band(band_of(run))
        self.frf_plot.set_quality_label(run.quality_label)
        # channel table + FRFs (recomputed from TDMS when possible)
        self.rows = channel_rows(run)
        self._fill_table()
        self._fill_sweep()
        self.computed = None
        if can_recompute(run):
            self.ch_mode.setText("<b>Full re-processing available</b> — the "
                                 + ("dwell blocks of every frequency step" if run.stepped else
                                    "converged blocks of every realization") + " "
                                 "are stored, so any channel can be the reference and the FRFs are recomputed "
                                 + ("at every measured frequency" if run.stepped else "over 0 … fs/2")
                                 + ". Edit the table, then <i>Apply</i>; "
                                 "<i>Save re-processed copy</i> writes a new run folder.")
            try:
                self.computed = recompute_frf(run, self.rows)
            except Exception as e:
                self.ch_status.setText(f"<span style='color:#c0392b'>{e}</span>")
        else:
            self.ch_mode.setText("<b>Relabelling only</b> — this run has no converged-block TDMS file, so the "
                                 "reference cannot be changed. Names, nodes and directions of the stored FRFs "
                                 "can be edited and re-saved.")
        self.frf_src.blockSignals(True)
        self.frf_src.clear()
        if self.computed is not None:
            self.frf_src.addItem(SRC_COMPUTED)
        if run.uff:
            self.frf_src.addItem(SRC_STORED)
        self.frf_src.blockSignals(False)
        self._fill_frf_list()
        # realizations
        full = set(run.realization_files)
        self.real_sel.blockSignals(True)
        self.real_sel.clear()
        for r in run.realizations():
            if run.stepped:
                p = run.step_info(r)
                self.real_sel.addItem(f"Step {r} · {p.get('frequency_Hz', 0):g} Hz · {p.get('status', '')}", r)
            else:
                tag = "all blocks" if r in full else "converged only"
                self.real_sel.addItem(f"{r}  ({tag})", r)
        self.real_sel.blockSignals(False)
        self.src_sel.setVisible(not run.stepped)
        self.real_label.setText("Step" if run.stepped else "Realization")
        for w in (self.lower, self.lower_label):
            w.setVisible(run.stepped)
        self.sub.setTabText(1, "Frequency steps — time && spectrum (TDMS)" if run.stepped
                            else "Realizations — time && spectrum (TDMS)")
        self._load_realization()

    # ------------------------------------------------------------------ FRF view
    def _fill_sweep(self):
        """Stepped sine: harmonics of every channel at every frequency, from the TDMS time data
        (dwell blocks, or the converged block per frequency if only that was saved)."""
        run = self.run
        if run is None or not run.stepped:
            self.sweep.clear("Harmonics exist only for stepped-sine runs.")
            return
        self.sweep.source_note = ""
        n_h = n_harmonics(run)                       # the run's single harmonic-count setting
        self.harm.set_max_order(n_h)
        self.sweep.set_max_order(n_h)
        try:
            spec = amplitude_spectra(run)
        except Exception as e:
            self.sweep.clear(f"<span style='color:#c0392b'>Could not read harmonics from "
                             f"{os.path.basename(run.converged)}: {type(e).__name__}: {e}</span>")
            return
        if not spec:
            self.sweep.clear("<span style='color:#c0392b'>This run has neither stepped-sine time data nor a "
                             "harmonics file. Harmonics need 'Dwell blocks' or 'Converged block of every "
                             "frequency → TDMS', or 'Harmonics → CSV (complex) + UFF' to have been saved.</span>")
            return
        names = list(spec)
        rows = channel_rows(run)                     # this run's channels (not the previous run's)
        if getattr(self, "rows", None) and {r["key"] for r in self.rows} == {r["key"] for r in rows}:
            rows = self.rows                         # keep reference / labels edited in the table
        ref_key = next((r["key"] for r in rows if r["reference"]), names[0])
        ref_i = names.index(ref_key) if ref_key in names else 0
        try:
            harm = np.stack([spec[n]["harm"] for n in names], axis=1)        # (nf, nch, H)
            label = {r["key"]: r["name"] for r in rows}
            self.sweep.set_data(spec[names[0]]["freqs"], harm, [label.get(n, n) for n in names],
                                [spec[n]["units"] for n in names], ref_i)
            from_csv = spec[names[0]].get("source") == "csv"
            if from_csv:
                src = f"saved harmonics file {os.path.basename(run.harmonics_csv)}"
            else:
                src = ("dwell blocks" if run.converged.endswith("_stepped_sine.tdms") else "converged block") \
                      + " of every frequency"
            self.sweep.source_note = f"  Source: {src}."
            self.sweep._redraw()
        except Exception as e:
            self.sweep.clear(f"<span style='color:#c0392b'>Could not show harmonics: {type(e).__name__}: {e}</span>")

    def _spectrum_mode(self) -> bool:
        return self.qty.currentIndex() == 1

    def _fill_frf_list(self, *_):
        cur = self.frf_sel.currentText()
        self.frf_sel.blockSignals(True)
        self.frf_sel.clear()
        spec = self._spectrum_mode()
        stepped = self.run is not None and self.run.stepped
        for w in (self.src_label, self.frf_src, *self.est_cb.values()):
            w.setVisible(not spec)
        self.item_label.setText("  Channel" if spec else "  Response")
        self.frf_plot.show_aux(not spec)
        if self.run is not None and spec:
            self.frf_plot.set_points(stepped)
            self.frf_sel.addItems(list(amplitude_spectra(self.run)))
            if cur and self.frf_sel.findText(cur) >= 0:
                self.frf_sel.setCurrentText(cur)
        elif self.run is not None:
            if self.frf_src.currentText() == SRC_COMPUTED and self.computed is not None:
                _, resp, ref = self.computed
                self.frf_plot.set_points(self.run.stepped)
                self.frf_sel.addItems([f"{r['name']} ({r['node']}{r['direction']}) / "
                                       f"{ref['name']} ({ref['node']}{ref['direction']})" for r in resp])
                for cb in self.est_cb.values():
                    cb.setEnabled(True)
            else:
                self.frf_plot.set_points(self.run.stepped)
                self.frf_sel.addItems(self.run.frf_responses())
                for k, cb in self.est_cb.items():
                    cb.setEnabled(k in self.run.uff)
            if cur and self.frf_sel.findText(cur) >= 0:
                self.frf_sel.setCurrentText(cur)
        self.frf_sel.blockSignals(False)
        self._draw_frf()

    def _frf_data(self, j: int) -> dict:
        if self.frf_src.currentText() == SRC_COMPUTED and self.computed is not None:
            res, resp, ref = self.computed
            return {"freqs": res.freqs, "H1": res.H1[j], "H2": res.H2[j], "Hv": res.Hv[j],
                    "Coherence": res.coherence[j], "units": f"{resp[j]['units']}/{ref['units']}"}
        return self.run.frf(j)

    def _draw_frf(self, *_):
        if self.run is None or self.frf_sel.currentIndex() < 0:
            self.frf_plot.clear()
            self.frf_info.setText("" if self.run is None or not self._spectrum_mode() else
                                  "Amplitude spectra need the run's TDMS time data.")
            return
        h = 1
        self.frf_plot.legend.setVisible(not self._spectrum_mode())
        for cb in self.est_cb.values():
            cb.setVisible(not self._spectrum_mode())
        if self._spectrum_mode():
            name = self.frf_sel.currentText()
            d = amplitude_spectra(self.run)[name]
            amp = np.abs(d["harm"][:, h - 1]) if "harm" in d else d["amp"]
            ok = np.isfinite(amp)
            lbl = f"|{name}|  [{d['units']}]"
            self.frf_plot.set_frf(d["freqs"][ok], {"H1": amp[ok]}, None, lbl)
            kind = ("amplitude of the fundamental at each excitation frequency (harmonics: see the "
                    "Harmonics tab)" if self.run.stepped else
                    "single-sided amplitude, RMS-averaged over realizations")
            self.frf_info.setText(f"{ok.sum()} points · {kind}")
            return
        self.frf_plot.set_quality_label(self.run.quality_label)
        self.frf_plot.p_coh.setYRange(0, 1.05)
        d = self._frf_data(self.frf_sel.currentIndex())
        if "freqs" not in d:
            self.frf_plot.clear()
            return
        f = d["freqs"]
        keep = f > 0
        curves = {k: d[k][keep] if (k in d and self.est_cb[k].isChecked()) else None for k in ("H1", "H2", "Hv")}
        coh = d["Coherence"][keep] if "Coherence" in d else None
        self.frf_plot.set_frf(f[keep], curves, coh, f"|H|  [{d.get('units', '')}]")
        self.frf_info.setText(f"{f[keep].size} {'frequencies' if self.run.stepped else 'lines'}, "
                              f"{f[keep][0]:g}–{f[keep][-1]:g} Hz" if keep.any() else "")

    # ------------------------------------------------------------------ time view
    def _load_realization(self, *_):
        r = self.real_sel.currentData()
        self._data = None
        if self.run is None or r is None:
            self.t_plot.set_curves([], [])
            self.s_plot.set_curves([], [])
            return
        if self.run.stepped:
            self._data = self.run.realization(r, True)
            self._after_load()
            return
        want_all = self.src_sel.currentIndex() == 1
        has_all = r in self.run.realization_files
        has_conv = r in self.run.converged_realizations()
        if want_all and not has_all:
            self.src_sel.blockSignals(True)
            self.src_sel.setCurrentIndex(0)
            self.src_sel.blockSignals(False)
            want_all = False
        if not want_all and not has_conv:      # only the full file exists
            want_all = True
            self.src_sel.blockSignals(True)
            self.src_sel.setCurrentIndex(1)
            self.src_sel.blockSignals(False)
        self._data = self.run.realization(r, want_all)
        self._after_load(want_all)

    def _after_load(self, want_all: bool = True):
        d = self._data
        names = [f"{n}  [{u}]" if u else n for n, u, _ in d["channels"]]
        for combo, extra in ((self.ch_a, []), (self.ch_b, [NONE_ITEM])):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(extra + names)
            if cur and combo.findText(cur) >= 0:
                combo.setCurrentText(cur)
            elif combo is self.ch_a and len(names) > 1:
                combo.setCurrentIndex(1)
            combo.blockSignals(False)
        cur = self.spec_sel.currentIndex()
        self.spec_sel.blockSignals(True)
        self.spec_sel.clear()
        nb = d["n_blocks"]
        for i in range(nb):
            self.spec_sel.addItem(f"block {i + 1}/{nb}" + (" (converged)" if (i == d["converged_index"] and not
                                                                               self.run.stepped) else ""))
        self.spec_sel.setCurrentIndex(cur if 0 <= cur < nb and want_all else d["converged_index"])
        self.spec_sel.blockSignals(False)
        self._draw_time()

    def _chan(self, combo: QComboBox):
        if self._data is None or not combo.currentText() or combo.currentText() == NONE_ITEM:
            return None
        i = combo.currentIndex() - (1 if combo is self.ch_b else 0)
        return self._data["channels"][i]

    def _draw_time(self, *_):
        d = self._data
        a, b = self._chan(self.ch_a), self._chan(self.ch_b)
        if d is None or a is None:
            return
        fs, n, nb = d["fs"], d["block_size"], d["n_blocks"]
        r = self.real_sel.currentData()
        t = d["t0"] + np.arange(a[2].size) / fs
        self.t_plot.set_curves(t, a[2], a[0], a[1], b[2] if b else None, *(b[:2] if b else ("", "")))
        sel = self.spec_sel.currentIndex()
        starts = [d["t0"] + i * n / fs for i in range(nb)]
        texts, conv = d["labels"], d["converged"]
        self.t_plot.decorate_blocks(starts, n / fs, texts, conv, float(np.max(a[2])),
                                    selected=None if self.run.stepped else sel)
        self.t_plot.setXRange(t[0], t[-1] + 1 / fs, padding=0.01)
        if self.run.stepped:
            self.t_title.setText(f"<b>Step {r}: {d['frequency']:g} Hz</b> ({d['status']}) — {nb} dwell block(s) "
                                 f"· fs = {fs:g} Hz · block = {n} samples")
        else:
            self.t_title.setText(f"<b>Realization {r}</b> — {'all ' + str(nb) + ' dwell blocks' if nb > 1 else 'converged block'}"
                                 f" · fs = {fs:g} Hz · block = {n} samples; <span style='color:#1e8449'>green</span> = "
                                 "converged block" + (", <span style='color:#2980b9'>blue</span> = spectrum block"
                                                      if sel != d["converged_index"] else ""))
        # lower panel: harmonics (stepped sine) or spectrum of the chosen block
        seg = slice(sel * n, (sel + 1) * n)
        show_harm = self.run.stepped and self.lower.currentIndex() == 0
        show_strobe = self.run.stepped and self.lower.currentIndex() == 2
        self.harm.setVisible(show_harm)
        self.v_strobe.setVisible(show_strobe)
        self.spec_widget.setVisible(not (show_harm or show_strobe))
        for w in (self.spec_sel, self.spec_log):
            w.setEnabled(not show_strobe)
        sz = self.time_plots.sizes()                # give the panel being shown a usable height
        tot = sum(sz)
        want = 3 if show_strobe else (2 if show_harm else 1)
        if sz[want] < 150 and tot > 0:
            new = [max(200, tot * 2 // 5), 0, 0, 0]
            new[want] = tot - new[0]
            self.time_plots.setSizes(new)
        if show_strobe:                         # every dwell block of this step folded onto one period
            def blocks(sig):
                return [sig[i * n:(i + 1) * n] for i in range(nb)]
            self.v_strobe.set_max_order(n_harmonics(self.run))
            self.v_strobe.show_blocks(fs, d.get("line"), (a[0], a[1], blocks(a[2])),
                                      (b[0], b[1], blocks(b[2])) if b else None, [d.get("line")] * nb)
            return
        if show_harm:
            self.harm.show_block(d.get("line"), fs, (a[0], a[1], a[2][seg]), (b[0], b[1], b[2][seg]) if b else None)
            return
        fa, Xa = amplitude_spectrum(a[2][seg], fs)
        Xb = amplitude_spectrum(b[2][seg], fs)[1] if b else None
        log = self.spec_log.isChecked()
        self.s_plot.set_log_y(log)
        floor = 1e-30 if log else 0.0
        self.s_plot.set_curves(fa, Xa + floor, f"|{a[0]}|", a[1], None if Xb is None else Xb + floor,
                               f"|{b[0]}|" if b else "", b[1] if b else "")
        self.s_title.setText(f"<b>Amplitude spectrum</b> of block {sel + 1}/{nb} (rectangular window, "
                             f"Δf = {fs / n:.4g} Hz)")

    # ------------------------------------------------------------------ channel table
    def _fill_table(self):
        t = self.ch_table
        t.setRowCount(0)
        editable_ref = self.run is not None and can_recompute(self.run)
        for i, r in enumerate(self.rows):
            t.insertRow(i)
            use = QCheckBox()
            use.setChecked(r["use"] and not r["reference"])
            ref = QCheckBox()
            ref.setChecked(r["reference"])
            ref.setEnabled(editable_ref)
            ref.toggled.connect(lambda on, row=i: self._ref_toggled(row, on))
            for col, w in ((0, use), (1, ref)):
                box = QWidget()
                lay = QHBoxLayout(box)
                lay.setContentsMargins(0, 0, 0, 0)
                lay.setAlignment(Qt.AlignCenter)
                lay.addWidget(w)
                t.setCellWidget(i, col, box)
            name = QLineEdit(r["name"])
            name.setFrame(False)
            node = QSpinBox()
            node.setRange(0, 999999)
            node.setValue(int(r["node"]))
            node.setFrame(False)
            d = QComboBox()
            d.addItems(DIRECTIONS)
            d.setCurrentText(r["direction"])
            d.setFrame(False)
            t.setCellWidget(i, 2, name)
            t.setCellWidget(i, 3, node)
            t.setCellWidget(i, 4, d)
            for col, txt in ((5, r["units"]), (6, r["sensor_type"]), (7, r["key"])):
                it = QTableWidgetItem(txt)
                it.setFlags(Qt.ItemIsEnabled)
                t.setItem(i, col, it)
            use.setEnabled(not r["reference"] and (editable_ref or r["use"]))
            r["_w"] = (use, ref, name, node, d)
        self.ch_status.setText("")

    def _ref_toggled(self, row: int, on: bool):
        for i, r in enumerate(self.rows):
            use, ref = r["_w"][0], r["_w"][1]
            if i == row:
                if on:
                    use.setChecked(False)
                use.setEnabled(not on)
            elif on and ref.isChecked():
                ref.blockSignals(True)
                ref.setChecked(False)
                ref.blockSignals(False)
                use.setEnabled(True)

    def _rows_from_table(self):
        out = []
        for r in self.rows:
            use, ref, name, node, d = r["_w"]
            out.append({**{k: v for k, v in r.items() if k != "_w"},
                        "use": use.isChecked(), "reference": ref.isChecked(), "name": name.text().strip(),
                        "node": node.value(), "direction": d.currentText()})
        errs = []
        if sum(x["reference"] for x in out) != 1:
            errs.append("tick exactly one Reference")
        if not any(x["use"] and not x["reference"] for x in out):
            errs.append("tick at least one Response")
        names = [x["name"] for x in out]
        if any(not n for n in names):
            errs.append("every channel needs a name")
        if len(set(names)) != len(names):
            errs.append("channel names must be unique")
        if errs:
            raise ValueError("; ".join(errs).capitalize() + ".")
        return out

    def _apply_rows(self):
        if self.run is None:
            return
        try:
            rows = self._rows_from_table()
            if can_recompute(self.run):
                self.computed = recompute_frf(self.run, rows)
                if self.frf_src.findText(SRC_COMPUTED) < 0:
                    self.frf_src.insertItem(0, SRC_COMPUTED)
                self.frf_src.setCurrentText(SRC_COMPUTED)
                res, resp, ref = self.computed
                what = "frequency steps" if self.run.stepped else "converged blocks"
                msg = (f"FRFs recomputed from {res.n_avg} {what}: reference <b>{ref['name']}</b> "
                       f"({ref['node']}{ref['direction']}), {len(resp)} response(s). "
                       "See the Frequency response tab; save to keep.")
            else:
                msg = "Labels updated. They are applied to the FRFs when you save a re-processed copy."
            for r, new in zip(self.rows, rows):
                r.update({k: new[k] for k in ("use", "reference", "name", "node", "direction")})
            self._fill_frf_list()
            self._fill_sweep()
            self.ch_status.setText(f"<span style='color:#1e8449'>{msg}</span>")
        except Exception as e:
            self.ch_status.setText(f"<span style='color:#c0392b'>{e}</span>")

    def _revert_rows(self):
        if self.run is not None:
            self._load_run()
            self.ch_status.setText("Reverted to the saved channel labels and reference.")

    def _save_reprocessed(self):
        if self.run is None:
            return
        try:
            rows = self._rows_from_table()
        except Exception as e:
            self.ch_status.setText(f"<span style='color:#c0392b'>{e}</span>")
            return
        parent = os.path.dirname(os.path.normpath(self.run.folder))
        dest = QFileDialog.getExistingDirectory(self, "Save re-processed run into folder", parent)
        if not dest:
            return
        out = os.path.join(dest, f"{self.run.name}_reprocessed_{_dt.datetime.now():%Y%m%d_%H%M%S}")
        log = []
        try:
            save_reprocessed(self.run, rows, out, self.inc_real.isChecked(), log.append)
        except Exception as e:
            QMessageBox.critical(self, "Save re-processed copy", f"{type(e).__name__}: {e}")
            return
        self.ch_status.setText("<span style='color:#1e8449'>Saved to " + out + "</span><br>" + "<br>".join(log))
        # show the new run
        self.folder.setText(dest)
        self.rescan()
        i = self.run_sel.findData(out)
        if i >= 0:
            self.run_sel.setCurrentIndex(i)

    def _img_name(self, what: str) -> str:
        base = self.run.name if self.run else "results"
        extra = ""
        if what == "FRF" and self.frf_sel.currentText():
            extra = "_" + self.frf_sel.currentText().split(" (")[0]
        elif what == "time" and self.real_sel.currentData() is not None:
            extra = f"_R{self.real_sel.currentData():03d}"
        folder = self.run.folder if self.run else ""
        return os.path.join(folder, f"{base}_{what}{extra}.png")
