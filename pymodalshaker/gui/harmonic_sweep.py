# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Stepped sine: harmonics against excitation frequency for several channels and orders at once.
Used on the Measurement tab (live) and in the Data Viewer (saved runs)."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QSplitter, QVBoxLayout, QWidget)

from .plots import export_dialog

PALETTE = ["#2b7bb9", "#d35400", "#27ae60", "#8e44ad", "#c0392b", "#16a085", "#7f8c8d", "#f39c12",
           "#2c3e50", "#e84393"]
ORDER_STYLES = [Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine, Qt.DashDotDotLine]
ORDER_SYMBOLS = ["o", "s", "t", "d", "+", "x", "star", "p", "h", "t1"]
MODES = ["Amplitude |X_h|", "Level re own fundamental (dBc)", "Harmonic FRF |Y_h / X_1,ref|"]


class HarmonicSweepView(QWidget):
    def __init__(self, parent=None, max_order: int = 5):
        super().__init__(parent)
        self.chans = QListWidget()
        self.chans.setToolTip("Tick the channels to show")
        self.chans.itemChanged.connect(self._redraw)
        self.orders = QListWidget()
        self.orders.setToolTip("Tick the harmonic orders to show (1 = fundamental). The number of orders "
                               "is set once, on the Output Setup tab (stepped sine → Harmonics).")
        self.orders.itemChanged.connect(self._orders_changed)
        # quick group toggles: fundamental, odd harmonics (3, 5, …), even harmonics (2, 4, …)
        self.grp = {}
        for key, label, tip in (("fund", "Fundamental", "Show / hide the fundamental (h = 1)"),
                                ("odd", "Odd", "Tick / untick all odd harmonics h = 3, 5, 7, …"),
                                ("even", "Even", "Tick / untick all even harmonics h = 2, 4, 6, …")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda on, k=key: self._toggle_group(k, on))
            self.grp[key] = b
        self.max_order = 0
        self.set_max_order(max_order)
        self.mode = QComboBox()
        self.mode.addItems(MODES)
        self.mode.setToolTip("Amplitude: |X_h| of each channel (its own units).\n"
                             "Level: 20·log10(|X_h| / |X_1|) of each channel.\n"
                             "Harmonic FRF: |X_h| of each channel over the reference channel's fundamental |X_1,ref|.")
        self.mode.currentIndexChanged.connect(self._redraw)
        self.log = QCheckBox("log |·|")
        self.log.setChecked(True)
        self.log.toggled.connect(self._redraw)
        self.show_phase = QCheckBox("phase")
        self.show_phase.setChecked(True)
        self.show_phase.setToolTip("Phase of each harmonic relative to the excitation: φ_h − h·φ_ref,1")
        self.show_phase.toggled.connect(self._redraw)
        self.info = QLabel()
        self.info.setWordWrap(True)
        exp = QPushButton("📷 Export image…")

        self.p_mag = pg.PlotWidget()
        self.p_ph = pg.PlotWidget()
        for p in (self.p_mag, self.p_ph):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("bottom").enableAutoSIPrefix(False)
        self.p_ph.setXLink(self.p_mag)
        self.p_ph.setLabel("left", "Phase re excitation", "°")
        self.p_ph.setLabel("bottom", "Excitation frequency f₀", "Hz")
        self.p_ph.setYRange(-180, 180)
        self.legend = self.p_mag.addLegend(offset=(-10, 10))
        self.plots = QSplitter(Qt.Vertical)
        self.plots.addWidget(self.p_mag)
        self.plots.addWidget(self.p_ph)
        self.plots.setSizes([380, 170])
        exp.clicked.connect(lambda: export_dialog(self, self.plots, "harmonics.png"))

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(QLabel("<b>Channels</b>"))
        sl.addWidget(self.chans, 3)
        sl.addWidget(QLabel("<b>Harmonic orders</b>"))
        gl = QHBoxLayout()
        gl.setSpacing(2)
        for b in self.grp.values():
            gl.addWidget(b)
        sl.addLayout(gl)
        sl.addWidget(self.orders, 2)
        top = QHBoxLayout()
        top.addWidget(QLabel("Show"))
        top.addWidget(self.mode)
        top.addWidget(self.log)
        top.addWidget(self.show_phase)
        top.addStretch()
        top.addWidget(exp)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addLayout(top)
        rl.addWidget(self.info)
        rl.addWidget(self.plots, 1)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(side)
        split.addWidget(right)
        split.setSizes([190, 800])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        self.freqs = np.array([])
        self.harm: Optional[np.ndarray] = None
        self.names: List[str] = []
        self.units: List[str] = []
        self.ref_i = 0

    # ------------------------------------------------------------------ orders
    @staticmethod
    def _group_of(h: int) -> str:
        return "fund" if h == 1 else ("even" if h % 2 == 0 else "odd")

    def set_max_order(self, n: int):
        """Offer orders 1…n (the single harmonic-count setting), keeping the current ticks."""
        n = max(2, int(n))
        if n == self.max_order:
            return
        ticked = set(self._checked(self.orders)) if self.max_order else {2, 3}
        self.orders.blockSignals(True)
        self.orders.clear()
        for h in range(1, n + 1):
            it = QListWidgetItem(f"h = {h}" + (" (fundamental)" if h == 1 else ""))
            it.setData(Qt.UserRole, h)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if h in ticked else Qt.Unchecked)
            self.orders.addItem(it)
        self.orders.blockSignals(False)
        self.max_order = n
        self._sync_groups()

    def _toggle_group(self, key: str, on: bool):
        self.orders.blockSignals(True)
        for i in range(self.orders.count()):
            it = self.orders.item(i)
            if self._group_of(it.data(Qt.UserRole)) == key:
                it.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.orders.blockSignals(False)
        self._orders_changed()

    def _sync_groups(self):
        """A group button is down when every order of that group is ticked."""
        for key, b in self.grp.items():
            items = [self.orders.item(i) for i in range(self.orders.count())
                     if self._group_of(self.orders.item(i).data(Qt.UserRole)) == key]
            b.setEnabled(bool(items))
            b.setChecked(bool(items) and all(it.checkState() == Qt.Checked for it in items))

    def _orders_changed(self, *_):
        self._sync_groups()
        self._redraw()

    # ------------------------------------------------------------------ data
    def set_data(self, freqs, harm, names: List[str], units: List[str], ref_i: int):
        """freqs (nf,), harm (nf, n_channels, H) complex peak amplitudes, channel names/units,
        index of the reference channel. The order list follows H (the run's harmonic count)."""
        self.freqs = np.asarray(freqs, float)
        self.harm = None if harm is None else np.asarray(harm)
        if self.harm is not None and self.harm.ndim == 3 and self.harm.shape[2] > self.max_order:
            self.set_max_order(self.harm.shape[2])
        if names != self.names or ref_i != self.ref_i or not self._checked(self.chans):
            checked = {self.chans.item(i).text() for i in range(self.chans.count())
                       if self.chans.item(i).checkState() == Qt.Checked}
            self.chans.blockSignals(True)
            self.chans.clear()
            for i, n in enumerate(names):
                it = QListWidgetItem(n + ("  (reference)" if i == ref_i else ""))
                it.setData(Qt.UserRole, i)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                default = (i != ref_i and i == (1 if ref_i == 0 else 0))
                keep = it.text() in checked
                it.setCheckState(Qt.Checked if (keep or (not checked and default)) else Qt.Unchecked)
                it.setForeground(pg.mkColor(PALETTE[i % len(PALETTE)]))
                self.chans.addItem(it)
            self.chans.blockSignals(False)
        self.names, self.units, self.ref_i = list(names), list(units), ref_i
        self._redraw()

    def clear(self, msg: str = ""):
        self.set_data([], None, [], [], 0)
        self.info.setText(msg)

    def _checked(self, lw: QListWidget):
        return [lw.item(i).data(Qt.UserRole) for i in range(lw.count())
                if lw.item(i).checkState() == Qt.Checked]

    # ------------------------------------------------------------------ drawing
    def _redraw(self, *_):
        for p in (self.p_mag, self.p_ph):
            for it in list(p.listDataItems()):
                p.removeItem(it)
        self.legend.clear()
        if self.harm is None or self.harm.size == 0:
            if not self.info.text():
                self.info.setText("No stepped-sine harmonics yet.")
            return
        mode = self.mode.currentIndex()
        chans, orders = self._checked(self.chans), self._checked(self.orders)
        H = self.harm.shape[2]
        # orders that are above Nyquist at every frequency of the data (never measured)
        dead = [h for h in range(1, self.max_order + 1)
                if h > H or not np.isfinite(self.harm[:, :, h - 1]).any()]
        for i in range(self.orders.count()):
            it = self.orders.item(i)
            h = it.data(Qt.UserRole)
            base = f"h = {h}" + (" (fundamental)" if h == 1 else "")
            it.setText(base + ("  — above Nyquist, not measured" if h in dead else ""))
            it.setForeground(pg.mkColor("#888888") if h in dead else self.orders.palette().text().color())
        orders = [h for h in orders if h <= H]
        dbc = mode == 1
        self.p_mag.setLogMode(y=self.log.isChecked() and not dbc)
        self.log.setEnabled(not dbc)
        self.p_ph.setVisible(self.show_phase.isChecked())
        ref1 = self.harm[:, self.ref_i, 0]
        n_curves = 0
        for c in chans:
            for oi, h in enumerate(orders):
                X = self.harm[:, c, h - 1]
                if mode == 0:
                    y = np.abs(X)
                elif mode == 1:
                    y = 20 * np.log10(np.abs(X) / (np.abs(self.harm[:, c, 0]) + 1e-300) + 1e-300)
                else:
                    y = np.abs(X) / (np.abs(ref1) + 1e-300)
                ok = np.isfinite(y) & np.isfinite(X)
                if not ok.any():
                    continue
                color = PALETTE[c % len(PALETTE)]
                pen = pg.mkPen(color, width=1.6, style=ORDER_STYLES[oi % len(ORDER_STYLES)])
                kw = dict(pen=pen, symbol=ORDER_SYMBOLS[(h - 1) % len(ORDER_SYMBOLS)], symbolSize=6,
                          symbolBrush=color, symbolPen=None)
                yy = y[ok] + (1e-30 if (self.log.isChecked() and not dbc) else 0)
                self.p_mag.plot(self.freqs[ok], yy, name=f"{self.names[c]} · h={h}", **kw)
                ph = np.angle(X[ok]) - h * np.angle(ref1[ok])
                self.p_ph.plot(self.freqs[ok], np.degrees(np.angle(np.exp(1j * ph))), **kw)
                n_curves += 1
        units = sorted({self.units[c] for c in chans})
        if mode == 0:
            self.p_mag.setLabel("left", "|X_h|", " / ".join(u for u in units if u))
        elif mode == 1:
            self.p_mag.setLabel("left", "Level re own fundamental", "dBc")
        else:
            ru = self.units[self.ref_i] if self.units else ""
            self.p_mag.setLabel("left", f"|Y_h| / |X_1,ref|  [{'/'.join(units)} per {ru}]")
        self.p_mag.getAxis("left").enableAutoSIPrefix(False)
        msg = f"{n_curves} curve(s): colour = channel, line style / marker = harmonic order."
        if not chans or not orders:
            msg = "Tick at least one channel and one harmonic order."
        elif mode == 0 and len(units) > 1:
            msg += "  ⚠ channels with different units share one axis — consider 'Level (dBc)'."
        if dead:
            rng = f"{dead[0]}…{dead[-1]}" if len(dead) > 1 else f"{dead[0]}"
            msg += (f"  Orders {rng} lie above Nyquist (fs/2) at every frequency of this run and could not be "
                    "measured — use a higher sampling rate.")
        else:
            msg += "  Orders with h·f₀ ≥ fs/2 are omitted where they occur."
        self.info.setText(msg + getattr(self, "source_note", ""))
        if n_curves == 0 and chans and orders:
            self.info.setText("Nothing to draw: the ticked orders are above Nyquist at every frequency, or the "
                              "data contain no finite values for the ticked channels.")
