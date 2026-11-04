# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Stepped-sine controller panel: live status, iterations at the current frequency, sweep
overview and a colour-coded table of every frequency step."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Slot, Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QGridLayout, QHeaderView, QLabel, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .plots import COLOR_A, COLOR_B, DualAxisPlot

STEP_COLS = ["Step", "f (Hz)", "Status", "Iter.", "Drive (V pk)", "Achieved", "Error %", "THD target %",
             "THD ref %", "η min", "Overload", "Reason"]
# Semi-transparent tints over the theme's own base colour, with the theme's own text colour:
# readable in light and dark themes alike.
STATUS_COLORS = {"converged": (46, 204, 113, 75), "open loop": (52, 152, 219, 70),
                 "accepted": (243, 156, 18, 95), "skipped": (231, 76, 60, 85), "failed": (231, 76, 60, 140)}


class ControllerPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status = {}
        grid = QGridLayout()
        for i, (key, label) in enumerate((("step", "Step"), ("f", "Frequency"), ("phase", "Phase"),
                                          ("amp", "Drive"), ("y", "Control channel"), ("err", "Error"))):
            grid.addWidget(QLabel(f"<span style='color:gray'>{label}</span>"), 0, i)
            v = QLabel("–")
            v.setStyleSheet("font-weight: bold; font-size: 11pt")
            grid.addWidget(v, 1, i)
            self.status[key] = v

        self.it_plot = DualAxisPlot()
        self.it_plot.setTitle("Current frequency: control iterations")
        self.it_plot.setLabel("bottom", "Iteration")
        self.it_plot.curve_a.setSymbol("o")
        self.it_plot.curve_a.setSymbolBrush(COLOR_A)
        self.it_plot.curve_a.setSymbolSize(7)
        self.it_plot.curve_b.setSymbol("s")
        self.it_plot.curve_b.setSymbolBrush(COLOR_B)
        self.it_plot.curve_b.setSymbolSize(6)
        self.target_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#1e8449", width=1.5))
        self.tol_band = pg.LinearRegionItem(orientation="horizontal", movable=False,
                                            brush=pg.mkBrush(46, 204, 113, 40), pen=pg.mkPen(None))
        self.it_plot.addItem(self.tol_band)
        self.it_plot.addItem(self.target_line)
        self.over_pts = self.it_plot.plot(pen=None, symbol="x", symbolSize=12, symbolPen=pg.mkPen("#c0392b", width=2))

        self.sweep = DualAxisPlot()
        self.sweep.setTitle("Sweep: drive (blue), final error % (orange); ○ flagged, ✕ skipped/failed")
        self.sweep.setLabel("bottom", "Frequency", "Hz")
        for c, sym, col in ((self.sweep.curve_a, "o", COLOR_A), (self.sweep.curve_b, "t", COLOR_B)):
            c.setSymbol(sym)
            c.setSymbolBrush(col)
            c.setSymbolSize(7)
        self.bad_pts = self.sweep.plot(pen=None, symbol="x", symbolSize=12, symbolPen=pg.mkPen("#c0392b", width=2))
        self.flag_pts = self.sweep.plot(pen=None, symbol="o", symbolSize=13, symbolBrush=None,
                                        symbolPen=pg.mkPen("#e67e22", width=2))

        self.table = QTableWidget(0, len(STEP_COLS))
        self.table.setHorizontalHeaderLabels(STEP_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(22)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self.it_plot)
        split.addWidget(self.sweep)
        split.addWidget(self.table)
        split.setSizes([230, 200, 200])
        lay = QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addWidget(split, 1)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.closed_loop = True
        self.target_label = "RMS"
        self.unit_channel = ""
        self.target = None
        self.tol = 3.0
        self.its: List[Dict] = []
        self.steps: List[Dict] = []
        self.freqs: List[float] = []
        self.table.setRowCount(0)
        self.it_plot.set_curves([], [])
        self.sweep.set_curves([], [])
        self.over_pts.setData([], [])
        self.bad_pts.setData([], [])
        for v in self.status.values():
            v.setText("–")

    @Slot(object)
    def on_event(self, ev: Dict):
        t = ev.get("type")
        if t == "start":
            self.reset()
            self.closed_loop = ev["closed_loop"]
            self.target, self.tol = ev["target"], ev["tolerance_pct"]
            self.freqs = ev["freqs"]
            self.unit_channel = ev.get("target_channel", "")
            self.target_label = ev.get("target_label", "RMS")
            self.it_plot.setVisible(self.closed_loop)
            if self.target:
                self.target_line.setValue(self.target)
                self.tol_band.setRegion((self.target * (1 - self.tol / 100), self.target * (1 + self.tol / 100)))
        elif t == "freq":
            self.its = []
            self.status["step"].setText(f"{ev['j'] + 1} / {ev['n']}")
            self.status["f"].setText(f"{ev['f']:g} Hz")
            self.status["phase"].setText("controlling" if self.closed_loop else "settle + dwell")
            self.status["amp"].setText(f"{ev['amp']:.4g} V")
            self.status["err"].setText("–")
            self.status["err"].setStyleSheet("font-weight: bold; font-size: 11pt")
            self._draw_iterations()
            self.it_plot.setTitle(f"{ev['f']:g} Hz: waiting for the first measurement…"
                                  if self.closed_loop else f"{ev['f']:g} Hz")
        elif t == "iteration":
            self.its.append(ev)
            ok = abs(ev["err_pct"]) <= self.tol and not ev["overload"]
            self.status["phase"].setText(f"iteration {ev['it']}" + (" · OVERLOAD" if ev["overload"] else ""))
            self.status["amp"].setText(f"{ev['amp']:.4g} V")
            self.status["y"].setText(f"{ev['y']:.4g} / {ev['target']:.4g} {self.target_label}")
            self.status["err"].setText(f"{ev['err_pct']:+.2f} %")
            self.status["err"].setStyleSheet("font-weight: bold; font-size: 11pt; color: "
                                             + ("#1e8449" if ok else "#c0392b"))
            self._draw_iterations()
        elif t == "wait":
            self.status["phase"].setText("FAILED — waiting for decision")
            self.status["phase"].setStyleSheet("font-weight: bold; font-size: 11pt; color: #c0392b")
        elif t == "step":
            self.status["phase"].setStyleSheet("font-weight: bold; font-size: 11pt")
            self.steps.append(ev)
            self._add_row(ev)
            self._draw_sweep()

    # ------------------------------------------------------------------ drawing
    def _draw_iterations(self):
        if not self.its:
            self.it_plot.set_curves([], [])
            self.over_pts.setData([], [])
            return
        x = np.array([r["it"] for r in self.its], float)
        y = np.array([r["y"] for r in self.its])
        a = np.array([r["amp"] for r in self.its])
        self.it_plot.set_curves(x, y, f"{self.unit_channel} {self.target_label}", "", a, "Drive", "V pk")
        ov = [r for r in self.its if r["overload"]]
        self.over_pts.setData([r["it"] for r in ov], [r["y"] for r in ov])
        f = self.its[-1]["f"]
        self.it_plot.setTitle(f"{f:g} Hz: control {self.target_label} (blue; green = target ± tol.) and drive (orange)")
        lo = min(float(y.min()), self.target * (1 - self.tol / 100))
        hi = max(float(y.max()), self.target * (1 + self.tol / 100))
        self.it_plot.setYRange(lo, hi, padding=0.08)
        self.it_plot.setXRange(0.5, max(2.5, x.max() + 0.5), padding=0)

    def _draw_sweep(self):
        good = [r for r in self.steps if r.get("drive_amplitude_V") is not None]
        f = np.array([r["frequency_Hz"] for r in good])
        amp = np.array([r["drive_amplitude_V"] for r in good])
        if self.closed_loop:
            err = np.array([r.get("error_percent") or 0.0 for r in good])
            self.sweep.set_curves(f, amp, "Drive", "V pk", err, "Final error", "%")
        else:
            ref = np.array([r.get("reference_amplitude_rms") or 0.0 for r in good])
            self.sweep.set_curves(f, amp, "Drive", "V pk", ref, "Reference RMS", "")
        bad = [r for r in self.steps if r["status"] in ("skipped", "failed")]
        y0 = float(amp.min()) if amp.size else 0.0
        self.bad_pts.setData([r["frequency_Hz"] for r in bad], [y0] * len(bad))
        flag = [r for r in good if r["status"].startswith("accepted") or "overload" in r["status"]]
        self.flag_pts.setData([r["frequency_Hz"] for r in flag], [r["drive_amplitude_V"] for r in flag])

    def _add_row(self, r: Dict):
        i = self.table.rowCount()
        self.table.insertRow(i)
        fmt = lambda v, p="{:.4g}": "" if v is None else p.format(v)
        vals = [str(r["j"] + 1), f"{r['frequency_Hz']:g}", r["status"], str(r["iterations"] or ""),
                fmt(r.get("drive_amplitude_V")), fmt(r.get("achieved")), fmt(r.get("error_percent"), "{:+.2f}"),
                fmt(r.get("thd_target_percent"), "{:.2f}"), fmt(r.get("thd_reference_percent"), "{:.2f}"),
                fmt(r.get("fundamental_fraction_min"), "{:.4f}"), r.get("overload") or "", r.get("reason") or ""]
        color = next((c for k, c in STATUS_COLORS.items() if r["status"].startswith(k)), None)
        if r.get("overload"):
            color = STATUS_COLORS["accepted"] if color == STATUS_COLORS["converged"] else color
        for col, v in enumerate(vals):
            it = QTableWidgetItem(v)
            if color:
                it.setBackground(QBrush(QColor(*color)))
            if col == len(vals) - 1:
                it.setToolTip(v)
            self.table.setItem(i, col, it)
        self.table.scrollToBottom()

    def summary_text(self) -> str:
        n = len(self.steps)
        ok = sum(1 for r in self.steps if r["status"].startswith(("converged", "open loop")))
        flag = sum(1 for r in self.steps if r["status"].startswith("accepted"))
        return f"{n} frequencies: {ok} OK, {flag} flagged, {n - ok - flag} skipped/failed"
