# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Harmonic suite: controller panel — status, per-harmonic table (target, response, error, drive)
and the error history over control iterations and test blocks."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QGridLayout, QHeaderView, QLabel, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .plots import COLOR_A, COLOR_B

COLS = ["Order", "Target", "Response", "Error %", "Drive (V)"]
OK_TINT, BAD_TINT, FLAG_TINT = (46, 204, 113, 75), (231, 76, 60, 85), (243, 156, 18, 95)


def _polar(z, unit="", digits=3) -> str:
    if z is None or not np.isfinite(z):
        return "–"
    return f"{abs(z):.{digits}g} {unit} ∠ {np.degrees(np.angle(z)):+.1f}°".replace("  ", " ")


class HarmonicControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status = {}
        grid = QGridLayout()
        for i, (key, label) in enumerate((("f0", "Base frequency"), ("phase", "Phase"), ("it", "Iteration"),
                                          ("err", "Max error"), ("peak", "Drive peak"))):
            grid.addWidget(QLabel(f"<span style='color:gray'>{label}</span>"), 0, i)
            v = QLabel("–")
            v.setStyleSheet("font-weight: bold; font-size: 11pt")
            grid.addWidget(v, 1, i)
            self.status[key] = v
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(22)

        self.plot = pg.PlotWidget(title="Max harmonic error: control iterations (blue), test blocks (orange)")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        # log10 of the error is plotted on a linear axis with decade ticks (clean labels)
        self.plot.setLabel("left", "Max error (%, log scale)")
        self.plot.setLabel("bottom", "Iteration / block")
        self.plot.getAxis("left").setTicks([[(float(e), f"{10.0 ** e:g}") for e in range(-3, 5)],
                                            [(float(np.log10(m * 10.0 ** e)), "") for e in range(-3, 4)
                                             for m in (2, 5)]])
        self.c_it = self.plot.plot(pen=pg.mkPen(COLOR_A, width=1.5), symbol="o", symbolBrush=COLOR_A, symbolSize=7)
        self.c_test = self.plot.plot(pen=pg.mkPen(COLOR_B, width=1), symbol="s", symbolBrush=COLOR_B, symbolSize=4)
        self.c_over = self.plot.plot(pen=None, symbol="x", symbolSize=12, symbolPen=pg.mkPen("#c0392b", width=2))
        self.tol_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#1e8449", width=1.5, style=Qt.DashLine))
        self.plot.addItem(self.tol_line)
        self.note = QLabel()
        self.note.setWordWrap(True)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.table)
        split.addWidget(self.plot)
        split.setSizes([200, 260])
        lay = QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addWidget(self.note)
        lay.addWidget(split, 1)
        self.reset()

    def reset(self):
        self.orders: List[int] = []
        self.T = np.array([])
        self.closed = True
        self.unit = ""
        self.tol = 3.0
        self.its: List[Dict] = []
        self.tests: List[Dict] = []
        self.state = ""
        self.table.setRowCount(0)
        for c in (self.c_it, self.c_test, self.c_over):
            c.setData([], [])
        self.note.setText("")
        self._head = ""
        for v in self.status.values():
            v.setText("–")

    # ------------------------------------------------------------------ events
    @Slot(object)
    def on_event(self, ev: Dict):
        t = ev.get("type")
        if t == "hstart":
            self.reset()
            self.orders, self.T, self.closed = ev["orders"], np.asarray(ev["T"]), ev["closed_loop"]
            self.unit, self.tol = ev["unit"], ev["tolerance_pct"]
            self.status["f0"].setText(f"{ev['f0']:.5g} Hz")
            self.status["phase"].setText("control" if self.closed else "no control")
            self.tol_line.setValue(np.log10(self.tol))
            self._head = (f"Target channel <b>{ev['channel']}</b>, tolerance ±{self.tol:g} % (complex error), "
                          f"≤ {ev['max_iterations']} iterations, then {ev['test_blocks']} test block(s)."
                          if ev["closed_loop"] else
                          f"No control: the drive harmonics are fixed; {ev['test_blocks']} test block(s).")
            self.tol_line.setVisible(self.closed)
            self.table.setColumnHidden(1, not self.closed)
            self.table.setColumnHidden(2, not self.closed)
            self.table.setColumnHidden(3, not self.closed)
            self.table.setRowCount(len(self.orders))
            self.table.clearContents()
            for r, n in enumerate(self.orders):
                self._set(r, 0, f"h{n}")
                self._set(r, 1, _polar(self.T[r], self.unit))
                if not self.closed:
                    self._set(r, 4, _polar(self.T[r], "V"))
            self.note.setText(self._head)
        elif t == "hiter":
            self.its.append(ev)
            self.status["it"].setText(str(ev["it"]))
            self.status["err"].setText(f"{ev['err'] * 100:.3g} %")
            self.status["peak"].setText(f"{ev['peak']:.4g} V")
            self.status["phase"].setText("control" + (" (overload)" if ev["overload"] else ""))
            self.note.setText(self._head + (f"<br><span style='color:gray'>Iteration {ev['it']}: {ev['note']}</span>"
                                            if ev["note"] else ""))
            self._fill(ev["Y"], ev["e"], ev["V"])
            self._plot()
        elif t == "htest_start":
            self.state = ev["status"]
            self.status["phase"].setText(f"test ({ev['status']})")
            self.status["peak"].setText(f"{ev['peak']:.4g} V")
            for r in range(len(self.orders)):
                self._set(r, 4, _polar(ev["V"][r], "V"))
        elif t == "htest":
            self.tests.append(ev)
            self.status["phase"].setText(f"test block {ev['n']}/{ev['total']} ({self.state})")
            if "err" in ev:
                self.status["err"].setText(f"{ev['err'] * 100:.3g} %")
                self._fill(ev["Y"], ev["e"], None)
            self._plot()
        elif t == "hwait":
            self.status["phase"].setText("FAILED — waiting for your decision")
            self.note.setText(f"<span style='color:#c0392b'><b>Control failed:</b> {ev['reason']}</span>")
        elif t == "hdone":
            self.state = ev.get("status", self.state)
            self.status["phase"].setText(f"done ({self.state})")

    def _set(self, r, c, txt, tint=None):
        it = self.table.item(r, c)
        if it is None:
            it = QTableWidgetItem()
            it.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, c, it)
        it.setText(txt)
        it.setBackground(QBrush(QColor(*tint)) if tint else QBrush())

    def _fill(self, Y, e, V):
        for r in range(len(self.orders)):
            self._set(r, 2, _polar(Y[r], self.unit))
            ok = e[r] * 100 <= self.tol
            self._set(r, 3, f"{e[r] * 100:.3g}", OK_TINT if ok else BAD_TINT)
            if V is not None:
                self._set(r, 4, _polar(V[r], "V"))

    def _plot(self):
        if self.its:
            x = np.arange(1, len(self.its) + 1)
            y = np.log10([max(i["err"] * 100, 1e-3) for i in self.its])
            self.c_it.setData(x, y)
            ov = [(xi, yi) for xi, yi, i in zip(x, y, self.its) if i["overload"]]
            self.c_over.setData([p[0] for p in ov], [p[1] for p in ov])
        tests = [t for t in self.tests if "err" in t]
        if tests:
            x0 = len(self.its)
            self.c_test.setData(x0 + np.arange(1, len(tests) + 1), np.log10([max(t["err"] * 100, 1e-3) for t in tests]))

    def summary_text(self) -> str:
        n = len(self.tests)
        if not self.closed:
            return f"{n} test block(s), no control"
        errs = [t["err"] for t in self.tests if "err" in t]
        tail = f", test error ≤ {max(errs) * 100:.2f} %" if errs else ""
        return f"{self.state or 'stopped'} after {len(self.its)} iteration(s); {n} test block(s){tail}"
