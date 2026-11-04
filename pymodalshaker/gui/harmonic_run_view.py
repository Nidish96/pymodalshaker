# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Data Viewer page for harmonic-suite runs: saved blocks in time, strobed view (point cloud folded
onto one period with the Fourier fit) and the harmonics of every channel in the chosen block."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QPushButton,
                               QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .plots import COLOR_A, COLOR_B, DualAxisPlot, export_dialog
from .strobed_view import StrobedView, fourier_fit

NONE_ITEM = "(none)"


class HarmonicRunView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.block_sel = QComboBox()
        self.block_sel.setToolTip("Saved block (the strobed view overlays it with the preceding saved blocks "
                                  "that have the same drive)")
        self.block_sel.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.block_sel.setMinimumContentsLength(34)
        self.ch_a, self.ch_b = QComboBox(), QComboBox()
        self.ch_a.setStyleSheet(f"QComboBox {{ color: {COLOR_A} }}")
        self.ch_b.setStyleSheet(f"QComboBox {{ color: {COLOR_B} }}")
        exp = QPushButton("📷 Export image…")
        top = QHBoxLayout()
        top.addWidget(QLabel("Block"))
        top.addWidget(self.block_sel, 2)
        top.addWidget(QLabel("  A"))
        top.addWidget(self.ch_a, 1)
        top.addWidget(QLabel("B"))
        top.addWidget(self.ch_b, 1)
        top.addWidget(exp)

        self.t_title = QLabel()
        self.t_title.setWordWrap(True)
        self.t_plot = DualAxisPlot(downsample=True)
        self.t_plot.setLabel("bottom", "Time from start of acquisition", "s")
        self.props = QLabel()
        self.props.setWordWrap(True)
        self.props.setTextFormat(Qt.RichText)
        self.table = QTableWidget(0, 0)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.setToolTip("Harmonics of every channel in this block: peak amplitude ∠ phase (cosine convention, "
                              "phase relative to the block start)")
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.t_title)
        tsplit = QSplitter(Qt.Vertical)
        tsplit.addWidget(self.t_plot)
        bw = QWidget()
        bl = QVBoxLayout(bw)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.props)
        bl.addWidget(self.table, 1)
        tsplit.addWidget(bw)
        tsplit.setSizes([300, 260])
        ll.addWidget(tsplit, 1)
        self.strobe = StrobedView()
        self.plots = QSplitter(Qt.Horizontal)
        self.plots.addWidget(lw)
        self.plots.addWidget(self.strobe)
        self.plots.setSizes([600, 600])
        exp.clicked.connect(lambda: export_dialog(self, self.plots, "harmonic_block.png"))
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.plots, 1)

        self.run = None
        self.blocks = []
        self._cache = {}
        self.block_sel.currentIndexChanged.connect(self._draw)
        self.ch_a.currentIndexChanged.connect(self._draw)
        self.ch_b.currentIndexChanged.connect(self._draw)
        self.strobe.n_over.valueChanged.connect(self._draw)

    # ------------------------------------------------------------------ data
    def set_run(self, run, n_harm: int = 5):
        self.run = run
        self._cache = {}
        self.blocks = run.harmonic_blocks() if run is not None else []
        self.orders = run.harmonic_orders() if run is not None else []
        self.H = max([n_harm] + self.orders) if self.orders else n_harm
        self.strobe.set_max_order(self.H)
        self.block_sel.blockSignals(True)
        self.block_sel.clear()
        first_test = None
        for i, b in enumerate(self.blocks):
            p = b["props"]
            kind = b["kind"]
            what = (f"test {b['test']}" if kind == "test" else
                    f"iteration {b['iteration']} ({kind})" if b.get("iteration") else kind)
            err = p.get("max_error_percent")
            self.block_sel.addItem(f"#{b['index'] + 1} · t = {p.get('first_sample_time_s', 0):.3f} s · {what}"
                                   + (f" · err {err:.2f} %" if err is not None else ""), i)
            if first_test is None and kind == "test":
                first_test = i
        self.block_sel.setCurrentIndex(first_test if first_test is not None else 0)
        self.block_sel.blockSignals(False)
        if not self.blocks:
            self.t_title.setText("<span style='color:#c0392b'>This harmonic run has no saved time data "
                                 "(it was run with 'Time data → TDMS' off).</span>")
            self.strobe.clear("No saved blocks.")
            self.table.setRowCount(0)
            return
        d = self._block(0)
        names = [f"{n}  [{u}]" if u else n for n, u, _ in d["channels"]]
        for combo, extra in ((self.ch_a, []), (self.ch_b, [NONE_ITEM])):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(extra + names)
            if cur and combo.findText(cur) >= 0:
                combo.setCurrentText(cur)
            elif combo is self.ch_a:
                tgt = (run.setup or {}).get("harmonic", {}).get("target_channel", "")
                hit = [i for i, n in enumerate(names) if n.split("  [")[0] == tgt]
                combo.setCurrentIndex(hit[0] if hit else min(1, len(names) - 1))
            combo.blockSignals(False)
        self._draw()

    def _block(self, i: int):
        if i not in self._cache:
            self._cache[i] = self.run.harmonic_block(self.blocks[i]["group"])
        return self._cache[i]

    def _chan_index(self, combo) -> Optional[int]:
        if not combo.currentText() or combo.currentText() == NONE_ITEM:
            return None
        return combo.currentIndex() - (1 if combo is self.ch_b else 0)

    # ------------------------------------------------------------------ drawing
    def _draw(self, *_):
        i = self.block_sel.currentData()
        if self.run is None or i is None or not self.blocks:
            return
        d = self._block(i)
        ia, ib = self._chan_index(self.ch_a), self._chan_index(self.ch_b)
        if ia is None:
            return
        fs, k, p = d["fs"], d["line"], d["props"]
        a, b = d["channels"][ia], (d["channels"][ib] if ib is not None else None)
        t = float(p.get("first_sample_time_s", 0.0)) + np.arange(a[2].size) / fs
        self.t_plot.set_curves(t, a[2], a[0], a[1], b[2] if b else None, *(b[:2] if b else ("", "")))
        self.t_plot.setXRange(t[0], t[-1] + 1 / fs, padding=0.01)
        kind = p.get("kind", "")
        self.t_title.setText(f"<b>Block #{int(p.get('block_index', 0)) + 1}</b> ({kind}"
                             + (f" {p.get('test_block')}" if kind == "test" else "")
                             + (f", iteration {p.get('iteration')}" if p.get("iteration") else "")
                             + (f", {p.get('status')}" if p.get("status") else "")
                             + f") · f0 = {d['f0']:.6g} Hz · fs = {fs:g} Hz · {a[2].size} samples")
        # strobed: this block and the preceding saved blocks with the same drive
        drive = next((c[2] for c in d["channels"] if c[0] == "Drive"), None)
        idx = [i]
        j = i - 1
        while j >= 0 and len(idx) < self.strobe.n_over.value():
            dj = self._block(j)
            dr = next((c[2] for c in dj["channels"] if c[0] == "Drive"), None)
            if dj["line"] != k or drive is None or dr is None or dr.shape != drive.shape or not np.allclose(dr, drive):
                break
            idx.insert(0, j)
            j -= 1
        sig_a = (a[0], a[1], [self._block(m)["channels"][ia][2] for m in idx])
        sig_b = (b[0], b[1], [self._block(m)["channels"][ib][2] for m in idx]) if b else None
        if kind in ("decay", "tail", "fadein", "fadeout"):
            self.strobe.clear(f"Block #{int(p.get('block_index', 0)) + 1} is a '{kind}' block: no steady periodic drive.")
        else:
            self.strobe.show_blocks(fs, k, sig_a, sig_b, [k] * len(idx))
        # drive + control props
        drv = [(int(key[len("drive_h"):-len("_amp_V")]), v, p.get(key.replace("_amp_V", "_phase_deg"), 0.0))
               for key, v in p.items() if key.startswith("drive_h") and key.endswith("_amp_V")]
        txt = []
        if drv:
            txt.append("<b>Drive</b>: " + ", ".join(f"h{n}: {v:.4g} V ∠ {ph:+.1f}°" for n, v, ph in sorted(drv)))
        if "max_error_percent" in p:
            txt.append(f"<b>Control channel {p.get('control_channel', '')}</b>: max error {p['max_error_percent']:.3g} %")
        self.props.setText("<br>".join(txt))
        # harmonics of every channel in this block
        H = self.H
        chans = d["channels"]
        self.table.setColumnCount(1 + H + 1)
        self.table.setHorizontalHeaderLabels(["Channel"] + [f"h{h}" for h in range(1, H + 1)] + ["DC"])
        self.table.setRowCount(len(chans))
        for r, (name, unit, x) in enumerate(chans):
            dc, X = fourier_fit(x[None, :], k, H) if k else (float(np.mean(x)), np.full(H, np.nan))
            cells = [f"{name} [{unit}]" if unit else name]
            for z in X:
                cells.append("–" if not np.isfinite(z) else f"{abs(z):.4g} ∠ {np.degrees(np.angle(z)):+.1f}°")
            cells.append(f"{dc:.4g}")
            for c, txt_ in enumerate(cells):
                it = QTableWidgetItem(txt_)
                it.setTextAlignment(Qt.AlignCenter if c else Qt.AlignLeft | Qt.AlignVCenter)
                self.table.setItem(r, c, it)
