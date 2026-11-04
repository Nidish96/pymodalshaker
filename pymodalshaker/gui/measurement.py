# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Measurement tab: save options, start/stop, live time data, FRF + coherence plots, log."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from PySide6.QtCore import Slot, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QProgressBar,
                               QPushButton, QSpinBox, QSplitter, QTabWidget, QToolButton, QVBoxLayout,
                               QWidget)

from PySide6.QtWidgets import QDoubleSpinBox

from ..config import HARM_SAVE_MODES, SUITES, ChannelConfig, SaveConfig
from .controller_panel import ControllerPanel
from .harmonic_control import HarmonicControlPanel
from .pq_views import PQControlPanel, PQResultsView
from .strobed_view import StrobedView
from .harmonic_sweep import HarmonicSweepView
from .harmonics_view import HarmonicsView
from ..frf import FRFResult
from .plots import COLOR_A, COLOR_B, EST_COLORS, DualAxisPlot, FRFStack, export_dialog

NONE_ITEM = "(none)"


class MeasurementTab(QWidget):
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # ---------------- save options
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("Folder for result files")
        browse = QToolButton()
        browse.setText("…")
        browse.clicked.connect(self._browse)
        self.basename = QLineEdit()
        self.save_uff = QCheckBox("FRFs → UFF 58:")
        self.uff = {k: QCheckBox(k) for k in ("H1", "H2", "Hv", "Coherence")}
        self.save_conv = QCheckBox("Converged block of each realization")
        self.save_dwell = QCheckBox("Dwell blocks of every frequency")
        self.save_dwell.setToolTip("Stepped sine: all analysed (steady-state) dwell blocks of every frequency, "
                                   "every channel + drive → <base>_stepped_sine.tdms")
        self.save_each = QCheckBox("Every realization (one file each)")
        self.save_harm = QCheckBox("Harmonics → CSV (complex) + UFF")
        self.save_harm.setToolTip("Stepped sine: every channel's complex harmonics to <base>_harmonics.csv, and "
                                  "Y_h/X_1 of the responses to <base>_Harmonic<h>.uff. The orders are set once on "
                                  "the Output Setup tab (Stepped sine → Harmonics).")
        self.harm_note = QLabel()
        self.harm_note.setStyleSheet("color: gray")
        # harmonic suite: time data only
        self.h_save = QCheckBox("Time data → TDMS:")
        self.h_save.setToolTip("Harmonic suite: every channel + drive (+ trigger) → <base>_harmonic.tdms, one group "
                               "per saved block, with the drive and response harmonics as properties")
        self.h_mode = QComboBox()
        self.h_mode.addItems(HARM_SAVE_MODES)
        self.h_mode.setToolTip("All blocks: everything from the start (settling, every control iteration, test).\n"
                               "Converged (test) blocks only: the blocks run after control converged.\n"
                               "One converged block every S seconds: a sparse record of a long test.")
        self.h_every = QDoubleSpinBox()
        self.h_every.setRange(0.001, 1e6)
        self.h_every.setDecimals(3)
        self.h_every.setSuffix(" s")
        self.h_every_l = QLabel("S =")
        self.h_save.toggled.connect(self._h_save_enable)
        # phase quadrature: resonance table always; converged blocks and H1 UFF optional
        self.pq_blocks = QCheckBox("Converged block of every level → TDMS")
        self.pq_blocks.setToolTip("Time data (every channel + drive) of the block(s) recorded at the converged "
                                  "frequency of every level → <base>_phase_quadrature_blocks.tdms")
        self.pq_uff = QCheckBox("H1 at resonance → UFF 58")
        self.pq_uff.setToolTip("H1 = response / reference at the resonance of every level, against frequency "
                               "→ <base>_H1.uff (for comparison with other runs and other software)")
        self.h_mode.currentIndexChanged.connect(self._h_save_enable)
        self.save_uff.toggled.connect(lambda on: [c.setEnabled(on) for c in self.uff.values()])
        g_save = QGroupBox("Save results")
        gl = QGridLayout(g_save)
        fl = QHBoxLayout()
        fl.addWidget(QLabel("Folder"))
        fl.addWidget(self.folder, 3)
        fl.addWidget(browse)
        fl.addSpacing(12)
        fl.addWidget(QLabel("Base name"))
        fl.addWidget(self.basename, 1)
        gl.addLayout(fl, 0, 0, 1, 6)
        ul = QHBoxLayout()
        ul.setContentsMargins(0, 0, 0, 0)
        ul.addWidget(self.save_uff)
        for c in self.uff.values():
            ul.addWidget(c)
        ul.addStretch()
        self.uff_row_w = QWidget()
        self.uff_row_w.setLayout(ul)
        gl.addWidget(self.uff_row_w, 1, 0, 1, 6)
        tl_ = QHBoxLayout()                          # time-domain data
        tl_.setContentsMargins(0, 0, 0, 0)
        tl_.addWidget(QLabel("Time data → TDMS:"))
        for c in (self.save_conv, self.save_dwell, self.save_each):
            tl_.addWidget(c)
        tl_.addStretch()
        self.tdms_row_w = QWidget()
        self.tdms_row_w.setLayout(tl_)
        gl.addWidget(self.tdms_row_w, 2, 0, 1, 6)
        hs = QHBoxLayout()
        hs.setContentsMargins(0, 0, 0, 0)
        for w in (self.h_save, self.h_mode, self.h_every_l, self.h_every):
            hs.addWidget(w)
        hs.addWidget(QLabel("<span style='color:gray'>(only time data is saved for the harmonic suite)</span>"))
        hs.addStretch()
        self.hsave_row_w = QWidget()
        self.hsave_row_w.setLayout(hs)
        gl.addWidget(self.hsave_row_w, 5, 0, 1, 6)
        ps = QHBoxLayout()
        ps.setContentsMargins(0, 0, 0, 0)
        ps.addWidget(self.pq_blocks)
        ps.addWidget(self.pq_uff)
        ps.addWidget(QLabel("<span style='color:gray'>(the resonance table, harmonics and control log are always "
                            "saved as CSV)</span>"))
        ps.addStretch()
        self.pqsave_row_w = QWidget()
        self.pqsave_row_w.setLayout(ps)
        gl.addWidget(self.pqsave_row_w, 6, 0, 1, 6)
        hrow = QHBoxLayout()
        hrow.setContentsMargins(0, 0, 0, 0)
        hrow.addWidget(self.save_harm)
        hrow.addWidget(self.harm_note)
        hrow.addStretch()
        self.harm_row_w = QWidget()
        self.harm_row_w.setLayout(hrow)
        gl.addWidget(self.harm_row_w, 3, 0, 1, 6)
        self.save_hint = hint = QLabel("A time-stamped sub-folder is created per run (with a copy of the setup). "
                      "FRFs are always saved over the full range Δf … fs/2.")
        hint.setStyleSheet("color: gray")
        gl.addWidget(hint, 4, 0, 1, 6)

        # ---------------- run controls
        self.btn_start = QPushButton("▶  Start measurement")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("QPushButton { background:#1e8449; color:white; font-weight:bold; "
                                     "border-radius:4px; padding:0 18px }"
                                     "QPushButton:disabled { background:#95a5a6 }")
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("QPushButton { background:#c0392b; color:white; font-weight:bold; "
                                    "border-radius:4px; padding:0 18px }"
                                    "QPushButton:disabled { background:#95a5a6 }")
        self.btn_start.clicked.connect(self.start_requested)
        self.btn_stop.clicked.connect(self.stop_requested)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.phase = QLabel("Idle")
        self.phase.setStyleSheet("font-weight: bold")
        self.health = QLabel("")
        # The suite is chosen on the Output Setup tab; this tab only shows which one will run.
        self.suite_badge = QLabel()
        self.suite_badge.setToolTip("Chosen on the Output Setup tab. This tab shows only what applies to it.")
        self.suite_badge.setAlignment(Qt.AlignCenter)
        self.suite_badge.setMinimumWidth(130)
        sv = QVBoxLayout()
        sv.addWidget(self.suite_badge)
        run = QHBoxLayout()
        run.addLayout(sv)
        run.addWidget(self.btn_start)
        run.addWidget(self.btn_stop)
        pv = QVBoxLayout()
        pv.addWidget(self.phase)
        pv.addWidget(self.progress)
        pv.addWidget(self.health)
        run.addLayout(pv, 1)
        self.btn_export = QPushButton("📷  Export plots…")
        self.btn_export.setToolTip("Save the live-data and FRF plots as PNG, SVG or PDF")
        self.btn_export.clicked.connect(lambda: export_dialog(self, self.plots, "measurement_plots.png"))
        run.addWidget(self.btn_export)

        # ---------------- live time data: two channels (B on its own right axis)
        self.sel_a = QComboBox()
        self.sel_b = QComboBox()
        self.sel_a.setStyleSheet(f"QComboBox {{ color: {COLOR_A} }}")
        self.sel_b.setStyleSheet(f"QComboBox {{ color: {COLOR_B} }}")
        self.n_recent = QSpinBox()
        self.n_recent.setRange(1, 100)
        self.n_recent.setSuffix(" blocks")
        self.n_recent.setToolTip("How many of the most recent blocks to show in the strip below")
        for w in (self.sel_a, self.sel_b):
            w.currentIndexChanged.connect(self._redraw_time)
        self.n_recent.valueChanged.connect(self._redraw_time)
        self.time_plot = DualAxisPlot()
        self.time_plot.setLabel("bottom", "Time in block", "s")
        self.hist_plot = DualAxisPlot(downsample=True)
        self.hist_plot.showGrid(x=False, y=True, alpha=0.3)
        self.hist_plot.setLabel("bottom", "Time from start of acquisition", "s")
        self.hist_title = QLabel("<b>Recent blocks</b>")

        tl = QVBoxLayout()
        tl.setContentsMargins(0, 0, 0, 0)
        self.ab_row = QHBoxLayout()                 # shared by Live data, Controller and Harmonics
        self.ab_row.addWidget(QLabel("<b>Channels</b>  A:"))
        self.ab_row.addWidget(self.sel_a, 1)
        self.ab_row.addWidget(QLabel("B:"))
        self.ab_row.addWidget(self.sel_b, 1)
        tsplit = QSplitter(Qt.Vertical)
        tsplit.addWidget(self.time_plot)
        hw = QWidget()
        hl = QVBoxLayout(hw)
        hl.setContentsMargins(0, 4, 0, 0)
        hh = QHBoxLayout()
        hh.addWidget(self.hist_title, 1)
        hh.addWidget(QLabel("show last"))
        hh.addWidget(self.n_recent)
        hl.addLayout(hh)
        hl.addWidget(self.hist_plot, 1)
        tsplit.addWidget(hw)
        tsplit.setSizes([330, 300])
        tl.addWidget(tsplit, 1)
        tw = QWidget()
        tw.setLayout(tl)

        # ---------------- FRF / coherence
        self.frf_sel = QComboBox()
        self.est_cb = {k: QCheckBox(k) for k in ("H1", "H2", "Hv")}
        for k, cb in self.est_cb.items():
            cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color: {EST_COLORS[k]}; font-weight: bold }}")
            cb.toggled.connect(self._redraw_frf)
        self.navg = QLabel("")
        self.frf_sel.currentIndexChanged.connect(self._redraw_frf)
        self.frf = FRFStack()
        fh = QHBoxLayout()
        fh.addWidget(QLabel("<b>FRF</b>  response:"))
        fh.addWidget(self.frf_sel, 1)
        for cb in self.est_cb.values():
            fh.addWidget(cb)
        fh.addWidget(self.navg)
        fl = QVBoxLayout()
        fl.addLayout(fh)
        fl.addWidget(self.frf, 1)
        fw = QWidget()
        fw.setLayout(fl)

        # ---------------- log
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)

        self.controller = ControllerPanel()
        self.left_tabs = QTabWidget()
        self.left_tabs.addTab(tw, "Live data")
        self.left_tabs.addTab(self.controller, "Controller (stepped sine)")
        self.harm = HarmonicsView()
        self.left_tabs.addTab(self.harm, "Block harmonics")
        self.hcontrol = HarmonicControlPanel()
        self.left_tabs.addTab(self.hcontrol, "Controller (harmonic)")
        self.pqcontrol = PQControlPanel()
        self.left_tabs.addTab(self.pqcontrol, "Controller (phase quadrature)")
        self.left_tabs.currentChanged.connect(lambda _: self._redraw_time())
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addLayout(self.ab_row)
        ll.addWidget(self.left_tabs, 1)
        self.plots = QSplitter(Qt.Horizontal)
        self.plots.addWidget(left)
        self.sweep = HarmonicSweepView()
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(fw, "FRF")
        self.right_tabs.addTab(self.sweep, "Harmonics vs frequency (stepped sine)")
        self.strobe = StrobedView()
        self.right_tabs.addTab(self.strobe, "Strobed view")
        self.pqres = PQResultsView()
        self.right_tabs.addTab(self.pqres, "Resonance vs amplitude")
        self.right_tabs.currentChanged.connect(lambda _: self._redraw_time())
        self.plots.addWidget(self.right_tabs)
        self.plots.setSizes([450, 700])
        main_split = QSplitter(Qt.Vertical)
        main_split.addWidget(self.plots)
        main_split.addWidget(self.log)
        main_split.setSizes([600, 120])

        lay = QVBoxLayout(self)
        lay.addWidget(g_save)
        lay.addLayout(run)
        lay.addWidget(main_split, 1)

        self.channels: List[ChannelConfig] = []
        self.resp_idx: List[int] = []
        self._history: list = []
        self._fs = 1.0
        self._result: Optional[FRFResult] = None
        self._dwell = 1

    # ------------------------------------------------------------------ save config
    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Result folder", self.folder.text() or os.getcwd())
        if d:
            self.folder.setText(d)

    def set_save(self, s: SaveConfig):
        self.folder.setText(s.folder)
        self.basename.setText(s.basename)
        self.save_uff.setChecked(s.save_uff)
        for k, attr in (("H1", "uff_h1"), ("H2", "uff_h2"), ("Hv", "uff_hv"), ("Coherence", "uff_coherence")):
            self.uff[k].setChecked(getattr(s, attr))
            self.uff[k].setEnabled(s.save_uff)
        self.save_conv.setChecked(s.save_converged_tdms)
        self.save_each.setChecked(s.save_each_realization)
        self.save_harm.setChecked(s.save_harmonics)
        self.save_dwell.setChecked(s.save_dwell_tdms)
        self.h_save.setChecked(s.harm_save_tdms)
        self.h_mode.setCurrentText(s.harm_save_mode)
        self.h_every.setValue(s.harm_save_every_s)
        self._h_save_enable()
        self.pq_blocks.setChecked(s.pq_save_blocks)
        self.pq_uff.setChecked(s.pq_save_uff)

    def _h_save_enable(self, *_):
        on = self.h_save.isChecked()
        self.h_mode.setEnabled(on)
        every = self.h_mode.currentIndex() == 2
        self.h_every.setVisible(every)
        self.h_every_l.setVisible(every)
        self.h_every.setEnabled(on)

    def get_save(self) -> SaveConfig:
        return SaveConfig(folder=self.folder.text().strip(), basename=self.basename.text().strip() or "modal_test",
                          save_uff=self.save_uff.isChecked(), uff_h1=self.uff["H1"].isChecked(),
                          uff_h2=self.uff["H2"].isChecked(), uff_hv=self.uff["Hv"].isChecked(),
                          uff_coherence=self.uff["Coherence"].isChecked(),
                          save_converged_tdms=self.save_conv.isChecked(),
                          save_each_realization=self.save_each.isChecked(),
                          save_harmonics=self.save_harm.isChecked(),
                          save_dwell_tdms=self.save_dwell.isChecked(),
                          harm_save_tdms=self.h_save.isChecked(), harm_save_mode=self.h_mode.currentText(),
                          harm_save_every_s=self.h_every.value(),
                          pq_save_blocks=self.pq_blocks.isChecked(), pq_save_uff=self.pq_uff.isChecked())

    # ------------------------------------------------------------------ run state
    def set_harmonic_count(self, n: int):
        """The single harmonic-count setting (Output Setup) → every harmonics display and the save note."""
        self.harm_note.setText(f"orders 1 … {n}  (set on Output Setup)")
        self.harm.set_max_order(n)
        self.sweep.set_max_order(n)
        self.strobe.set_max_order(n)

    def set_suite_mode(self, suite: str):
        stepped, harmonic, pq = suite == SUITES[1], suite == SUITES[2], suite == SUITES[3]
        name = ("Stepped sine" if stepped else "Harmonic" if harmonic else "Phase<br>quadrature" if pq else "Random")
        self.suite_badge.setText(f"<b>{name}</b><br><span style='font-size:small'>suite (Output Setup)</span>")
        col = "#8e44ad" if stepped else ("#16a085" if harmonic else "#d35400" if pq else "#2b7bb9")
        self.suite_badge.setStyleSheet(f"QLabel {{ border: 2px solid {col}; border-radius: 4px; padding: 3px 8px; }}")
        self.save_conv.setText("Converged block of every frequency" if stepped
                               else "Converged block of each realization")
        self.save_conv.setToolTip("Stepped sine: the time waveform (every channel + drive) of the block in which "
                                  "control converged at each frequency (best iteration if accepted; first steady "
                                  "block without control) → <base>_stepped_converged_blocks.tdms"
                                  if stepped else "The converged (last) dwell block of every realization → "
                                  "<base>_converged_blocks.tdms")
        self.save_dwell.setVisible(stepped)
        self.save_each.setVisible(not stepped)
        self.harm_row_w.setVisible(stepped)
        self.uff["Coherence"].setText("Fundamental fraction η" if stepped else "Coherence")
        self.uff["Coherence"].setToolTip("Stepped sine: share of each channel's power at the excitation "
                                         "frequency (η_ref·η_resp), saved as <base>_FundamentalFraction.uff"
                                         if stepped else "Ordinary coherence γ²")
        self.save_hint.setText("A time-stamped sub-folder is created per run (with a copy of the setup)."
                               + ("" if (harmonic or pq) else " FRFs are always saved over the full range Δf … fs/2."))
        # save rows: harmonic = time data only; phase quadrature = its own
        self.uff_row_w.setVisible(not (harmonic or pq))
        self.tdms_row_w.setVisible(not (harmonic or pq))
        self.hsave_row_w.setVisible(harmonic)
        self.pqsave_row_w.setVisible(pq)
        if harmonic or pq:
            self.harm_row_w.setVisible(False)
        # left: Live data | Controller (stepped) | Block harmonics (sine suites) | Controller (harmonic) | Controller (PQ)
        for i, on in ((1, stepped), (2, stepped or harmonic or pq), (3, harmonic), (4, pq)):
            self.left_tabs.setTabEnabled(i, on)
            self.left_tabs.setTabVisible(i, on)
        if not self.left_tabs.isTabVisible(self.left_tabs.currentIndex()):
            self.left_tabs.setCurrentIndex(0)
        self.frf.set_points(stepped)
        self.frf.set_quality_label("Fundamental fraction η" if stepped else "Coherence γ²")
        # right: FRF (random, stepped) | Harmonics vs frequency (stepped) | Strobed view (sine suites) |
        #        Resonance vs amplitude (PQ)
        for i, on in ((0, not (harmonic or pq)), (1, stepped), (2, stepped or harmonic or pq), (3, pq)):
            self.right_tabs.setTabEnabled(i, on)
            self.right_tabs.setTabVisible(i, on)
        if not self.right_tabs.isTabVisible(self.right_tabs.currentIndex()):
            self.right_tabs.setCurrentIndex(3 if pq else 2 if harmonic else 0)

    def prepare_run(self, channels: List[ChannelConfig], ref_i: int, band, trigger: bool = False,
                    dwell: int = 1, keep: int = 100, stepped: bool = False, harmonic: bool = False,
                    pq: bool = False):
        """keep: how many recent blocks the acquisition thread retains (upper limit of the strip)."""
        self._stepped = stepped
        self._harmonic = harmonic
        self._pq = pq
        self.controller.reset()
        self.hcontrol.reset()
        self.pqcontrol.reset()
        self.pqres.clear("Results appear when the first level has converged.")
        self.strobe.clear()
        self.left_tabs.setCurrentIndex(1 if stepped else (3 if harmonic else 4 if pq else 0))
        if harmonic:
            self.right_tabs.setCurrentIndex(2)
        if pq:
            self.right_tabs.setCurrentIndex(3)
        ref_i = max(ref_i, 0)
        self.channels = channels
        self.resp_idx = [i for i in range(len(channels)) if i != ref_i]
        self._dwell = dwell
        self._history = []
        names = ([f"{c.name}  [{c.units}]" + ("  (reference)" if i == ref_i else "")
                  for i, c in enumerate(channels)] + ["Drive  [V]"] + (["Trigger output"] if trigger else []))
        for combo, extra in ((self.sel_a, []), (self.sel_b, [NONE_ITEM])):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(extra + names)
            if cur and combo.findText(cur) >= 0:
                combo.setCurrentText(cur)
            elif combo is self.sel_a:
                combo.setCurrentIndex(1 if len(channels) > 1 else 0)    # first response channel
            combo.blockSignals(False)
        self.n_recent.blockSignals(True)
        self.n_recent.setMaximum(max(1, keep))
        self.n_recent.setValue(min(dwell + 2, keep))
        self.n_recent.blockSignals(False)
        self.frf_sel.blockSignals(True)
        cur_f = self.frf_sel.currentText()
        self.frf_sel.clear()
        ref = channels[ref_i]
        self.frf_sel.addItems([f"{channels[i].name} ({channels[i].node}{channels[i].direction}) / "
                               f"{ref.name}  [{channels[i].units}/{ref.units}]" for i in self.resp_idx])
        if cur_f:
            self.frf_sel.setCurrentText(cur_f)
        self.frf_sel.blockSignals(False)
        self.frf.set_band(band)
        self._result = None
        self.frf.clear()
        self.sweep.clear("Harmonics appear after the first frequency has been measured.")
        self.navg.setText("")
        self.progress.setValue(0)

    def set_running(self, running: bool):
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

    @Slot(object)
    def on_pq_event(self, ev):
        """Phase quadrature controller events → controller panel and the resonance view."""
        self.pqcontrol.on_event(ev)
        t = ev.get("type")
        if t == "pq_start":
            self.pqres.set_units(ev["resp"], ev["resp_units"], ev["ref"], ev["ref_units"])
        elif t == "pq_result":
            self.pqres.add_row(ev)

    @Slot(str)
    def append_log(self, s: str):
        self.log.appendPlainText(s)

    @Slot(int, int, str)
    def on_progress(self, k: int, total: int, phase: str):
        self.progress.setMaximum(total)
        self.progress.setValue(k)
        self.phase.setText(phase)

    @Slot(object, float, int)
    def on_blocks(self, history: list, fs: float, backlog: int):
        """history: recent blocks, oldest first; each a dict(data, drive, trig, text, converged, start)."""
        if not history:
            return
        self._history = history
        self._fs = fs
        n = history[-1]["data"].shape[1]
        lag = backlog / n if n else 0
        self.health.setText(f"Input backlog: {lag:.1f} blocks" + ("  ⚠ PC is falling behind" if lag > 3 else ""))
        self._redraw_time()

    # ------------------------------------------------------------------ time plots
    def _channel(self, combo: QComboBox):
        """(index into [channels…, drive, trigger], name, unit) for a selector, or None."""
        txt = combo.currentText()
        if not txt or txt == NONE_ITEM:
            return None
        i = combo.currentIndex() - (1 if combo is self.sel_b else 0)
        n = len(self.channels)
        unit = self.channels[i].units if i < n else ("V" if i == n else "")
        return i, txt.split("[")[0].replace("(reference)", "").strip(), unit

    def _signal_of(self, entry: dict, i: int):
        n = len(self.channels)
        return entry["data"][i] if i < n else (entry["drive"] if i == n else entry["trig"])

    def _redraw_time(self, *_):
        if not self._history:
            return
        a, b = self._channel(self.sel_a), self._channel(self.sel_b)
        if a is None:
            return
        last = self._history[-1]
        ya = self._signal_of(last, a[0])
        yb = self._signal_of(last, b[0]) if b else None
        if ya is None:
            return
        t = np.arange(ya.size) / self._fs
        self.time_plot.set_curves(t, ya, a[1], a[2], yb, *(b[1:] if b else ("", "")))
        sine = getattr(self, "_stepped", False) or getattr(self, "_harmonic", False)
        if getattr(self, "_pq", False):                     # any frequency: least-squares views
            if self.left_tabs.currentWidget() is self.harm:
                self.harm.show_block(None, self._fs, (a[1], a[2], ya), (b[1], b[2], yb) if b else None,
                                     f=last.get("freq"))
            if self.right_tabs.currentWidget() is self.strobe:
                last_i = max((i for i, h in enumerate(self._history) if h.get("freq")), default=None)
                if last_i is not None:
                    hs = self._history[max(0, last_i + 1 - self.strobe.n_over.value()):last_i + 1]
                    key = hs[-1].get("drive_key")
                    ks = [1 if h.get("drive_key") == key else None for h in hs]
                    self.strobe.show_blocks(self._fs, None, (a[1], a[2], [self._signal_of(h, a[0]) for h in hs]),
                                            (b[1], b[2], [self._signal_of(h, b[0]) for h in hs]) if b else None, ks,
                                            f=hs[-1]["freq"], starts=[h["start"] for h in hs])
        if sine and self.left_tabs.currentWidget() is self.harm:
            self.harm.show_block(last.get("line"), self._fs, (a[1], a[2], ya), (b[1], b[2], yb) if b else None)
        if sine and self.right_tabs.currentWidget() is self.strobe:
            # latest blocks up to the last one with a sine in it (so the view survives the fade-out)
            last_i = max((i for i, h in enumerate(self._history) if h.get("line")), default=None)
            if last_i is None:
                return
            hs = self._history[max(0, last_i + 1 - self.strobe.n_over.value()):last_i + 1]
            ref = hs[-1]["drive"]          # overlay only blocks with the same drive (no transients between levels)
            ks = [h.get("line") if h["drive"].shape == ref.shape and np.array_equal(h["drive"], ref) else None
                  for h in hs]
            self.strobe.show_blocks(self._fs, hs[-1].get("line"),
                                    (a[1], a[2], [self._signal_of(h, a[0]) for h in hs]),
                                    (b[1], b[2], [self._signal_of(h, b[0]) for h in hs]) if b else None, ks)

        # recent-blocks strip
        want = self.n_recent.value()
        hist = [h for h in self._history if self._signal_of(h, a[0]) is not None][-want:]
        self.hist_title.setText(
            f"<b>Recent blocks</b> — {len(hist)} of {want} (dwell {self._dwell}); "
            "<span style='color:#1e8449'>green</span> = "
            + ("dwell (analysed) block" if getattr(self, "_stepped", False) else
               "converged (recorded) block" if getattr(self, "_pq", False) else
               "converged test block" if getattr(self, "_harmonic", False) else "converged block used for the FRF"))
        if not hist:
            return
        fs, nblk = self._fs, hist[0]["data"].shape[1]
        ya = np.concatenate([self._signal_of(h, a[0]) for h in hist])
        yb = np.concatenate([self._signal_of(h, b[0]) for h in hist]) if b else None
        t = (hist[0]["start"] + np.arange(ya.size)) / fs
        self.hist_plot.set_curves(t, ya, a[1], a[2], yb, *(b[1:] if b else ("", "")))
        self.hist_plot.decorate_blocks([h["start"] / fs for h in hist], nblk / fs,
                                       [h["text"] for h in hist], [h["converged"] for h in hist],
                                       float(np.max(ya)))
        self.hist_plot.setXRange(t[0], t[-1] + 1 / fs, padding=0.01)

    # ------------------------------------------------------------------ FRF
    @Slot(object)
    def on_result(self, res: FRFResult):
        self._result = res
        if getattr(self, "_stepped", False) and res.harm is not None:
            self.sweep.set_data(res.freqs, res.harm, [c.name for c in self.channels],
                                [c.units for c in self.channels], res.ref_i)
        self.navg.setText(f"  {res.n_avg} frequencies" if getattr(self, "_stepped", False) else f"  {res.n_avg} avg")
        self._redraw_frf()

    def _redraw_frf(self, *_):
        res = self._result
        j = self.frf_sel.currentIndex()
        if res is None or j < 0:
            return
        sel = res.freqs > 0                     # 0 … fs/2 (Nyquist); DC skipped for the log axis
        if getattr(self, "_stepped", False):
            self.frf.p_coh.setYRange(0, 1.05)
        curves = {k: getattr(res, k)[j][sel] if self.est_cb[k].isChecked() else None
                  for k in ("H1", "H2", "Hv")}
        self.frf.set_frf(res.freqs[sel], curves, res.coherence[j][sel])
