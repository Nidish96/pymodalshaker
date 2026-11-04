# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Phase Quadrature suite: controller panel (live) and the resonance-vs-amplitude results view
(used live on the Measurement tab and for saved runs in the Data Viewer)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .controller_panel import STATUS_COLORS
from .plots import COLOR_A, COLOR_B, DualAxisPlot, export_dialog

LEVEL_COLS = ["Level", "Drive (V)", "Status", "Iter.", "f res (Hz)", "Phase (°)", "|H1|", "Reference amp",
              "Response amp", "THD resp %", "Overload", "Reason"]
UP_COLOR, DOWN_COLOR = "#2b7bb9", "#c0392b"


def _fmt(v, f="{:.5g}"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f.format(v)


def _tint(status: str):
    return next((c for k, c in STATUS_COLORS.items() if status.startswith(k)), None)


class PQControlPanel(QWidget):
    """Live: status, iterations of the current level (frequency and phase error / objective), level table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.status = {}
        grid = QGridLayout()
        for i, (key, label) in enumerate((("lvl", "Level"), ("amp", "Drive"), ("f", "Frequency"),
                                          ("q", "Phase re ref."), ("err", "Error"), ("phase", "Phase"))):
            grid.addWidget(QLabel(f"<span style='color:gray'>{label}</span>"), 0, i)
            v = QLabel("–")
            v.setStyleSheet("font-weight: bold; font-size: 11pt")
            grid.addWidget(v, 1, i)
            self.status[key] = v
        self.it_plot = DualAxisPlot()
        self.it_plot.setLabel("bottom", "Iteration")
        self.it_plot.curve_a.setSymbol("o")
        self.it_plot.curve_a.setSymbolBrush(COLOR_A)
        self.it_plot.curve_b.setSymbol("s")
        self.it_plot.curve_b.setSymbolBrush(COLOR_B)
        self.tol_band = pg.LinearRegionItem(orientation="horizontal", movable=False,
                                            brush=pg.mkBrush(46, 204, 113, 40), pen=pg.mkPen(None))
        self.it_plot.vb2.addItem(self.tol_band)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.table = QTableWidget(0, len(LEVEL_COLS))
        self.table.setHorizontalHeaderLabels(LEVEL_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(22)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.it_plot)
        split.addWidget(self.table)
        split.setSizes([260, 220])
        lay = QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addWidget(self.note)
        lay.addWidget(split, 1)
        self.reset()

    def reset(self):
        self.its: List[Dict] = []
        self.rows: List[Dict] = []
        self.quad, self.tol, self.ftol, self.target = True, 1.0, 0.05, 90.0
        self.table.setRowCount(0)
        self.it_plot.set_curves([], [])
        self.note.setText("")
        for v in self.status.values():
            v.setText("–")

    @Slot(object)
    def on_event(self, ev: Dict):
        t = ev.get("type")
        if t == "pq_start":
            self.reset()
            self.quad, self.tol, self.ftol, self.target = ev["quadrature"], ev["phase_tol"], ev["freq_tol"], ev["target"]
            self.n = len(ev["levels"])
            self.tol_band.setVisible(self.quad)
            self.tol_band.setRegion((-self.tol, self.tol))
            self.note.setText(
                (f"Phase quadrature of <b>{ev['resp']}</b> re <b>{ev['ref']}</b> at {self.target:+g}° ± {self.tol:g}°"
                 if self.quad else f"Reference-free: maximum of |{ev['resp']}" + (f" / {ev['ref']}|" if ev["ref"] != "(none)"
                                                                               else "| per volt")
                 + f", ± {self.ftol:g} Hz") + f", ≤ {ev['max_iterations']} iterations per level.")
        elif t == "pq_level":
            self.its = []
            self.status["lvl"].setText(f"{ev['j'] + 1} / {ev['n']}")
            self.status["amp"].setText(f"{ev['amp']:.4g} V")
            self.status["f"].setText(f"{ev['f']:.5g} Hz")
            self.status["phase"].setText("searching")
            self._draw()
        elif t == "pq_iter":
            self.its.append(ev)
            self.status["f"].setText(f"{ev['f']:.6g} Hz")
            self.status["q"].setText(f"{ev['phase']:+.2f}°" if np.isfinite(ev["phase"]) else f"|·| {ev['J']:.4g}")
            if self.quad:
                ok = abs(ev["e"]) <= self.tol
                self.status["err"].setText(f"{ev['e']:+.2f}°")
            else:
                ok = "converged" in ev["note"]
                self.status["err"].setText(f"{ev['J']:.4g}")
            self.status["err"].setStyleSheet("font-weight: bold; font-size: 11pt; color: " + ("#1e8449" if ok else "#c0392b"))
            self.status["phase"].setText(f"iteration {ev['it']}" + (" · OVERLOAD" if ev["overload"] else ""))
            self._draw()
        elif t == "pq_wait":
            self.status["phase"].setText("FAILED — waiting for decision")
            self.note.setText(self.note.text().split("<br>")[0] + f"<br><span style='color:#c0392b'>{ev['reason']}</span>")
        elif t == "pq_result":
            self.rows.append(ev)
            self._add_row(ev)
            self.status["phase"].setText(ev["status"])
        elif t == "pq_done":
            self.status["phase"].setText("done")

    def _draw(self):
        if not self.its:
            self.it_plot.set_curves([], [])
            self.it_plot.setTitle("waiting for the first measurement…")
            return
        x = np.array([r["it"] for r in self.its], float)
        f = np.array([r["f"] for r in self.its])
        if self.quad:
            y = np.array([r["e"] for r in self.its])
            self.it_plot.set_curves(x, f, "Frequency", "Hz", y, "Phase error", "°")
            self.it_plot.setTitle("Frequency (blue) and phase error (orange; green band = tolerance)")
        else:
            y = np.array([r["J"] for r in self.its])
            self.it_plot.set_curves(x, f, "Frequency", "Hz", y, "Objective |H1|", "")
            self.it_plot.setTitle("Frequency (blue) and |H1| (orange)")
        self.it_plot.setXRange(0.5, max(2.5, x.max() + 0.5), padding=0)

    def _add_row(self, r: Dict):
        i = self.table.rowCount()
        self.table.insertRow(i)
        vals = [str(r["level"]), _fmt(r["drive_V"], "{:.4g}"), r["status"], str(r["iterations"]),
                _fmt(r.get("frequency_Hz"), "{:.6g}"), _fmt(r.get("phase_deg"), "{:+.2f}"), _fmt(r.get("H1_amp")),
                _fmt(r.get("ref_amp")), _fmt(r.get("resp_amp")), _fmt(r.get("thd_resp"), "{:.2f}"),
                r.get("overload") or "", r.get("reason") or ""]
        color = _tint(r["status"])
        for c, v in enumerate(vals):
            it = QTableWidgetItem(v)
            if color:
                it.setBackground(QBrush(QColor(*color)))
            self.table.setItem(i, c, it)
        self.table.scrollToBottom()

    def summary_text(self) -> str:
        ok = sum(1 for r in self.rows if r["status"] == "converged")
        fl = sum(1 for r in self.rows if r["status"].startswith("accepted"))
        return f"{len(self.rows)} levels: {ok} converged, {fl} flagged, {len(self.rows) - ok - fl} skipped/failed"


Y_CHOICES = ["Response amplitude", "Reference amplitude", "Drive level (V)", "|H1| (response / reference)",
             "Response per drive volt"]
Y_KEYS = ["resp_amp", "ref_amp", "drive_V", "H1_amp", "resp_per_volt"]


class PQResultsView(QWidget):
    """Amplitude-dependent resonance: backbone (amplitude vs resonance frequency) and H1 at resonance."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ysel = QComboBox()
        self.ysel.addItems(Y_CHOICES)
        self.ysel.setToolTip("Quantity plotted against the resonance frequency of each level")
        self.logy = QCheckBox("log")
        self.labels = QCheckBox("level numbers")
        self.labels.setChecked(True)
        exp = QPushButton("📷 Export image…")
        self.info = QLabel()
        self.info.setWordWrap(True)
        top = QHBoxLayout()
        top.addWidget(QLabel("Backbone: plot"))
        top.addWidget(self.ysel)
        top.addWidget(self.logy)
        top.addWidget(self.labels)
        top.addStretch()
        top.addWidget(exp)
        self.p_bb = pg.PlotWidget()
        self.p_h1 = pg.PlotWidget()
        self.p_ph = pg.PlotWidget()
        for p in (self.p_bb, self.p_h1, self.p_ph):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            p.getAxis("left").enableAutoSIPrefix(False)
        self.p_h1.setXLink(self.p_bb)
        self.p_ph.setXLink(self.p_bb)
        self.p_ph.setLabel("bottom", "Resonance frequency", "Hz")
        self.p_ph.setLabel("left", "Phase of H1", "°")
        self.leg = self.p_bb.addLegend(offset=(-10, 10))
        self.plots = QSplitter(Qt.Vertical)
        for p in (self.p_bb, self.p_h1, self.p_ph):
            self.plots.addWidget(p)
        self.plots.setSizes([340, 220, 140])
        exp.clicked.connect(lambda: export_dialog(self, self.plots, "resonance_vs_amplitude.png"))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addWidget(self.info)
        lay.addWidget(self.plots, 1)
        for w in (self.ysel,):
            w.currentIndexChanged.connect(self._redraw)
        for w in (self.logy, self.labels):
            w.toggled.connect(self._redraw)
        self.rows: List[Dict] = []
        self.units = {"resp": "", "ref": "", "resp_name": "response", "ref_name": "reference"}
        self._texts = []

    def set_units(self, resp_name, resp_units, ref_name, ref_units):
        self.units = {"resp": resp_units, "ref": ref_units, "resp_name": resp_name, "ref_name": ref_name}

    def clear(self, msg=""):
        self.rows = []
        self._redraw()
        self.info.setText(msg)

    def add_row(self, r: Dict):
        self.rows.append(r)
        self._redraw()

    def set_rows(self, rows: List[Dict]):
        self.rows = list(rows)
        self._redraw()

    def _redraw(self, *_):
        for p in (self.p_bb, self.p_h1, self.p_ph):
            for it in list(p.listDataItems()):
                p.removeItem(it)
        for t in self._texts:
            self.p_bb.removeItem(t)
        self._texts = []
        self.leg.clear()
        good = [r for r in self.rows if r.get("frequency_Hz") is not None and np.isfinite(r.get("frequency_Hz", np.nan))]
        u = self.units
        key = Y_KEYS[self.ysel.currentIndex()]
        ylab = {"resp_amp": (f"|{u['resp_name']}| fundamental", u["resp"]),
                "ref_amp": (f"|{u['ref_name']}| fundamental", u["ref"]),
                "drive_V": ("Drive level", "V peak"),
                "H1_amp": ("|H1|", f"{u['resp']}/{u['ref']}" if u["ref"] else ""),
                "resp_per_volt": (f"|{u['resp_name']}| per drive volt", f"{u['resp']}/V")}[key]
        self.p_bb.setLabel("left", *ylab)
        self.p_bb.setTitle("Backbone: amplitude vs resonance frequency (blue: increasing level, red: decreasing)")
        self.p_bb.setLogMode(y=self.logy.isChecked())
        self.p_h1.setLogMode(y=True)
        self.p_h1.setLabel("left", "|H1| at resonance", f"{u['resp']}/{u['ref']}" if u["ref"] else "")
        if not good:
            self.info.setText("No converged levels yet." if not self.rows else "No level converged.")
            return
        # split into legs of increasing / decreasing drive so up-down sweeps show hysteresis
        legs, cur = [], [good[0]]
        for a, b in zip(good, good[1:]):
            up_ab = b["drive_V"] >= a["drive_V"]
            up_cur = len(cur) < 2 or cur[-1]["drive_V"] >= cur[-2]["drive_V"]
            if up_ab == up_cur:
                cur.append(b)
            else:
                legs.append(cur)
                cur = [a, b]
        legs.append(cur)
        for leg in legs:
            up = len(leg) < 2 or leg[-1]["drive_V"] >= leg[0]["drive_V"]
            col = UP_COLOR if up else DOWN_COLOR
            f = np.array([r["frequency_Hz"] for r in leg])
            y = np.array([r.get(key, np.nan) for r in leg], float)
            ok = np.isfinite(y)
            kw = dict(pen=pg.mkPen(col, width=1.6), symbol="o" if up else "t", symbolSize=8, symbolBrush=col,
                      symbolPen=None)
            name = "increasing level" if up else "decreasing level"
            if not any(it.name() == name for it in self.p_bb.listDataItems()):
                self.p_bb.plot(f[ok], y[ok], name=name, **kw)
            else:
                self.p_bb.plot(f[ok], y[ok], **kw)
            h1 = np.array([r.get("H1") if r.get("H1") is not None else np.nan for r in leg], complex)
            okh = np.isfinite(h1)
            if okh.any():
                self.p_h1.plot(f[okh], np.abs(h1[okh]), **kw)
                self.p_ph.plot(f[okh], np.degrees(np.angle(h1[okh])), **kw)
        if self.labels.isChecked():
            for r in good:
                v = r.get(key, np.nan)
                if np.isfinite(v):
                    t = pg.TextItem(str(r["level"]), anchor=(0, 1), color="#7f8c8d")
                    yy = np.log10(v) if self.logy.isChecked() and v > 0 else v
                    t.setPos(r["frequency_Hz"], yy)
                    self.p_bb.addItem(t)
                    self._texts.append(t)
        has_h1 = any(r.get("H1") is not None for r in good)
        self.p_h1.setVisible(has_h1)
        self.p_ph.setVisible(has_h1)
        f_all = np.array([r["frequency_Hz"] for r in good])
        v_all = np.array([r["drive_V"] for r in good])
        lo, hi = int(np.argmin(v_all)), int(np.argmax(v_all))
        self.info.setText(f"{len(good)} converged level(s); resonance {f_all.min():.5g}–{f_all.max():.5g} Hz; "
                          f"from the lowest ({v_all[lo]:.4g} V) to the highest drive ({v_all[hi]:.4g} V) it moves "
                          f"{f_all[hi] - f_all[lo]:+.4g} Hz ({100 * (f_all[hi] - f_all[lo]) / f_all[lo]:+.2f} %)."
                          + ("" if has_h1 else "  No reference channel: H1 is not available."))
