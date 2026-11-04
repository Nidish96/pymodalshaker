# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Data Viewer page for phase-quadrature runs: resonance vs amplitude (backbone, H1 at resonance),
the level table, and a block inspector (time, strobed view and harmonics of the converged block)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..harmonics import ls_harmonics
from .plots import COLOR_A, COLOR_B, DualAxisPlot
from .pq_views import LEVEL_COLS, PQResultsView, _fmt, _tint
from .strobed_view import StrobedView

NONE_ITEM = "(none)"


class PQRunView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results = PQResultsView()
        self.table = QTableWidget(0, len(LEVEL_COLS))
        self.table.setHorizontalHeaderLabels(LEVEL_COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setToolTip("Click a level to inspect its converged block on the right")
        self.table.cellClicked.connect(lambda r, _c: self._pick_row(r))
        left = QSplitter(Qt.Vertical)
        left.addWidget(self.results)
        left.addWidget(self.table)
        left.setSizes([560, 220])

        self.level = QComboBox()
        self.ch_a, self.ch_b = QComboBox(), QComboBox()
        self.ch_a.setStyleSheet(f"QComboBox {{ color: {COLOR_A} }}")
        self.ch_b.setStyleSheet(f"QComboBox {{ color: {COLOR_B} }}")
        top = QHBoxLayout()
        top.addWidget(QLabel("Level"))
        top.addWidget(self.level, 1)
        top.addWidget(QLabel(" A"))
        top.addWidget(self.ch_a, 1)
        top.addWidget(QLabel("B"))
        top.addWidget(self.ch_b, 1)
        self.title = QLabel()
        self.title.setWordWrap(True)
        self.t_plot = DualAxisPlot(downsample=True)
        self.t_plot.setLabel("bottom", "Time from start of acquisition", "s")
        self.strobe = StrobedView()
        self.harm = QTableWidget(0, 0)
        self.harm.verticalHeader().setVisible(False)
        self.harm.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.harm.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.harm.horizontalHeader().setStretchLastSection(True)
        self.harm.verticalHeader().setDefaultSectionSize(22)
        self.harm.setToolTip("Harmonics of every channel at this level's resonance: peak amplitude ∠ phase")
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addLayout(top)
        rl.addWidget(self.title)
        rs = QSplitter(Qt.Vertical)
        rs.addWidget(self.t_plot)
        rs.addWidget(self.strobe)
        rs.addWidget(self.harm)
        rs.setSizes([200, 360, 160])
        rl.addWidget(rs, 1)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(rw)
        split.setSizes([700, 600])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        self.run = None
        self.level.currentIndexChanged.connect(self._draw_block)
        self.ch_a.currentIndexChanged.connect(self._draw_block)
        self.ch_b.currentIndexChanged.connect(self._draw_block)
        self.strobe.n_over.setValue(100)

    def set_run(self, run):
        self.run = run
        rows = run.pq_rows()
        p = (run.setup or {}).get("pq", {})
        self.H = int(p.get("n_harmonics", 5))
        self.strobe.set_max_order(self.H)
        r0 = rows[0] if rows else {}
        self.results.set_units(r0.get("resp_channel", "response"), r0.get("resp_units", ""),
                               r0.get("ref_channel") or "(none)", r0.get("ref_units", ""))
        self.results.set_rows(rows)
        self.table.setRowCount(0)
        for r in rows:
            i = self.table.rowCount()
            self.table.insertRow(i)
            vals = [str(r["level"]), _fmt(r["drive_V"], "{:.4g}"), r["status"], str(r["iterations"]),
                    _fmt(r["frequency_Hz"], "{:.6g}"), _fmt(r["phase_deg"], "{:+.2f}"), _fmt(r["H1_amp"]),
                    _fmt(r["ref_amp"]), _fmt(r["resp_amp"]), _fmt(r["thd_resp"], "{:.2f}"), r["overload"], r["reason"]]
            col = _tint(r["status"])
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if col:
                    it.setBackground(QBrush(QColor(*col)))
                self.table.setItem(i, c, it)
        self.level.blockSignals(True)
        self.level.clear()
        have = set(run.pq_levels_with_blocks())
        for r in rows:
            if np.isfinite(r["frequency_Hz"]):
                self.level.addItem(f"{r['level']}: {r['drive_V']:.4g} V · {r['frequency_Hz']:.6g} Hz · {r['status']}"
                                   + ("" if r["level"] in have else "  (no time data)"), r["level"])
        self.level.blockSignals(False)
        names = []
        if have:
            d = run.pq_block(sorted(have)[0])
            names = [f"{n}  [{u}]" if u else n for n, u, _ in d["channels"]]
        for combo, extra in ((self.ch_a, []), (self.ch_b, [NONE_ITEM])):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(extra + names)
            if cur and combo.findText(cur) >= 0:
                combo.setCurrentText(cur)
            elif combo is self.ch_a and names:
                hit = [i for i, n in enumerate(names) if n.split("  [")[0] == r0.get("resp_channel")]
                combo.setCurrentIndex(hit[0] if hit else 0)
            elif combo is self.ch_b and names:
                hit = [i for i, n in enumerate(names) if n.split("  [")[0] == r0.get("ref_channel")]
                combo.setCurrentIndex(hit[0] + 1 if hit else 0)
            combo.blockSignals(False)
        self._draw_block()

    def _pick_row(self, r):
        lvl = int(self.table.item(r, 0).text())
        i = self.level.findData(lvl)
        if i >= 0:
            self.level.setCurrentIndex(i)

    def _draw_block(self, *_):
        lvl = self.level.currentData()
        run = self.run
        if run is None or lvl is None:
            return
        row = next(r for r in run.pq_rows() if r["level"] == lvl)
        if lvl not in run.pq_levels_with_blocks():
            self.title.setText(f"<b>Level {lvl}</b>: the converged block was not saved (option 'Converged block of "
                               "every level → TDMS' off). Harmonics below are from the harmonics file.")
            self.t_plot.set_curves([], [])
            self.strobe.clear("No time data for this level.")
            self._fill_harm_from_csv(lvl, row)
            return
        d = run.pq_block(lvl)
        fs, f, st = d["fs"], d["f"], d["start"]
        ia = self.ch_a.currentIndex()
        ib = self.ch_b.currentIndex() - 1
        if ia < 0:
            return
        a = d["channels"][ia]
        b = d["channels"][ib] if ib >= 0 else None
        t = st / fs + np.arange(a[2].size) / fs
        self.t_plot.set_curves(t, a[2], a[0], a[1], b[2] if b else None, *(b[:2] if b else ("", "")))
        self.t_plot.setXRange(t[0], t[-1] + 1 / fs, padding=0.01)
        self.title.setText(f"<b>Level {lvl}</b>: {row['drive_V']:.4g} V · resonance {f:.6g} Hz · phase "
                           f"{row['phase_deg']:+.2f}° · |H1| {row['H1_amp']:.5g} · {row['status']} · "
                           f"{d['n_blocks']} converged block(s)")
        n = a[2].size // max(1, d["n_blocks"])
        split = lambda x: [x[i * n:(i + 1) * n] for i in range(d["n_blocks"])]
        starts = [st + i * n for i in range(d["n_blocks"])]
        self.strobe.show_blocks(fs, None, (a[0], a[1], split(a[2])), (b[0], b[1], split(b[2])) if b else None,
                                [1] * d["n_blocks"], f=f, starts=starts)
        # harmonics of every channel (least squares at the exact frequency, averaged over the blocks)
        chans = [c for c in d["channels"]]
        H = self.H
        self.harm.setColumnCount(1 + H)
        self.harm.setHorizontalHeaderLabels(["Channel"] + [f"h{h}" for h in range(1, H + 1)])
        self.harm.setRowCount(len(chans))
        for r, (name, unit, x) in enumerate(chans):
            Xs = [ls_harmonics(blk, f, fs, H, s0)[1][0] for blk, s0 in zip(split(x), starts)]
            X = np.mean(np.array(Xs), axis=0)
            cells = [f"{name} [{unit}]" if unit else name] + ["–" if not np.isfinite(z) else
                                                              f"{abs(z):.4g} ∠ {np.degrees(np.angle(z)):+.1f}°" for z in X]
            for c, v in enumerate(cells):
                self.harm.setItem(r, c, QTableWidgetItem(v))

    def _fill_harm_from_csv(self, lvl, row):
        from ..results import amplitude_spectra
        spec = amplitude_spectra(self.run)
        rows_lv = [r["level"] for r in self.run.pq_rows() if np.isfinite(r["frequency_Hz"])]
        if not spec or lvl not in rows_lv:
            self.harm.setRowCount(0)
            return
        i = rows_lv.index(lvl)
        names = list(spec)
        H = next(iter(spec.values()))["harm"].shape[1]
        self.harm.setColumnCount(1 + H)
        self.harm.setHorizontalHeaderLabels(["Channel"] + [f"h{h}" for h in range(1, H + 1)])
        self.harm.setRowCount(len(names))
        for r, nm in enumerate(names):
            X = spec[nm]["harm"][i] if i < spec[nm]["harm"].shape[0] else np.full(H, np.nan)
            cells = [nm] + ["–" if not np.isfinite(z) else f"{abs(z):.4g} ∠ {np.degrees(np.angle(z)):+.1f}°" for z in X]
            for c, v in enumerate(cells):
                self.harm.setItem(r, c, QTableWidgetItem(v))
