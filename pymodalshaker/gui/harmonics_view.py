# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Harmonic-decomposition display for stepped-sine blocks: harmonic levels (dB re fundamental)
for two channels, and one cycle of channel A rebuilt from its Fourier series."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget

from ..harmonics import decompose, decompose_ls
from .plots import COLOR_A, COLOR_B

FLOOR = -120.0


class HarmonicsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_harm = 5                       # set once on Output Setup (stepped sine → Harmonics)
        self._args = None
        self.info = QLabel()
        self.info.setTextFormat(Qt.RichText)
        self.info.setWordWrap(True)

        self.bars = pg.PlotWidget(title="Harmonic levels (dB re fundamental)")
        self.bars.showGrid(x=False, y=True, alpha=0.3)
        self.bars.setLabel("bottom", "Harmonic order h (frequency h·f₀)")
        self.bars.setLabel("left", "Level", "dBc")
        self.bars.getAxis("left").enableAutoSIPrefix(False)
        self.bars.addLegend(offset=(-10, 10))
        self.bar_a = pg.BarGraphItem(x=[], height=[], width=0.35, brush=COLOR_A, name="A")
        self.bar_b = pg.BarGraphItem(x=[], height=[], width=0.35, brush=COLOR_B, name="B")
        self.bars.addItem(self.bar_a)
        self.bars.addItem(self.bar_b)

        self.cycle = pg.PlotWidget(title="One cycle of A: measured and Fourier series")
        self.cycle.showGrid(x=True, y=True, alpha=0.3)
        self.cycle.setLabel("bottom", "Phase of the excitation cycle", "°")
        self.cycle.getAxis("bottom").enableAutoSIPrefix(False)
        self.cycle.getAxis("left").enableAutoSIPrefix(False)
        self.cycle.addLegend(offset=(-10, 10))
        self.c_meas = self.cycle.plot(pen=None, symbol="o", symbolSize=5, symbolBrush="#7f8c8d",
                                      symbolPen=None, name="measured samples")
        self.c_sum = self.cycle.plot(pen=pg.mkPen(COLOR_A, width=2), name="DC + harmonics 1…H")
        self.c_fund = self.cycle.plot(pen=pg.mkPen("#27ae60", width=1.5, style=Qt.DashLine), name="fundamental")
        self.c_harm = self.cycle.plot(pen=pg.mkPen("#c0392b", width=1.5), name="harmonics 2…H")

        self.split = QSplitter(Qt.Vertical)
        self.split.addWidget(self.bars)
        self.split.addWidget(self.cycle)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.info)
        lay.addWidget(self.split, 1)

    def set_max_order(self, n: int):
        """Harmonic orders 1…n (the single setting on Output Setup)."""
        self.n_harm = max(2, int(n))
        if getattr(self, "_args", None) is not None:
            self._redraw()

    def set_orientation(self, horizontal: bool):
        """Side by side when there is room (Data Viewer), stacked in narrow panels."""
        self.split.setOrientation(Qt.Horizontal if horizontal else Qt.Vertical)
        self._args = None

    def show_block(self, k: Optional[int], fs: float, a, b=None, f: Optional[float] = None):
        """a, b: (name, unit, samples) or None. k: excitation line of this block (None = no sine).
        f: excitation frequency when it is not on an FFT line (least-squares decomposition)."""
        self._args = (k, fs, a, b, f)
        self._redraw()

    def clear(self, msg: str = ""):
        self._args = None
        self.bar_a.setOpts(x=[], height=[])
        self.bar_b.setOpts(x=[], height=[])
        for c in (self.c_meas, self.c_sum, self.c_fund, self.c_harm):
            c.setData([], [])
        self.info.setText(msg)

    def _redraw(self, *_):
        if self._args is None:
            return
        k, fs, a, b, fq = self._args
        if fq:
            dec = lambda x: decompose_ls(x, fq, fs, self.n_harm)
        else:
            dec = lambda x: decompose(x, k, fs, self.n_harm)
        if not (k or fq) or a is None or a[2] is None:
            self.clear("<span style='color:gray'>No sine in this block (settling, fade or silence).</span>")
            return
        ha = dec(a[2])
        hb = dec(b[2]) if b is not None and b[2] is not None else None
        da = ha.dbc() - FLOOR
        self.bar_a.setOpts(x=ha.orders - (0.19 if hb else 0), height=da, y0=FLOOR, width=0.35 if hb else 0.6)
        if hb is not None:
            self.bar_b.setOpts(x=hb.orders + 0.19, height=hb.dbc() - FLOOR, y0=FLOOR, width=0.35)
        else:
            self.bar_b.setOpts(x=[], height=[])
        self.bars.setXRange(0.4, ha.orders[-1] + 0.6, padding=0)
        self.bars.getAxis("bottom").setTicks([[(int(h), str(int(h))) for h in ha.orders]])
        self.bars.setYRange(max(FLOOR, min(-100, float(ha.dbc().min()) - 10)), 5, padding=0)
        self.bars.setTitle(f"Harmonic levels at f₀ = {ha.f0:g} Hz (dB re fundamental)")

        # one cycle of A
        T = 1.0 / ha.f0
        t = np.linspace(0, T, 361)
        n_cyc = max(1, int(np.floor(a[2].size * ha.f0 / fs)))
        ts = np.arange(a[2].size) / fs
        sel = ts < T * min(n_cyc, 1)
        deg = lambda tt: 360.0 * tt / T
        self.c_meas.setData(deg(ts[sel]), a[2][sel])
        self.c_sum.setData(deg(t), ha.dc + ha.series(t))
        self.c_fund.setData(deg(t), ha.dc + ha.series(t, [1]))
        self.c_harm.setData(deg(t), ha.series(t, ha.orders[1:]) if ha.orders.size > 1 else 0 * t)
        self.cycle.setLabel("left", a[0], a[1])
        self.cycle.setXRange(0, 360, padding=0.01)

        def line(h, nm, unit, col):
            top = ", ".join(f"H{o}: {d:.1f}" for o, d in zip(h.orders[1:], h.dbc()[1:]))
            return (f"<span style='color:{col}'><b>{nm}</b></span>: fundamental {h.amp[0]:.4g} {unit} "
                    f"(phase {np.degrees(h.phase[0]):.1f}°), THD {h.thd * 100:.3g} %, DC {h.dc:.3g}, "
                    f"non-harmonic residual {h.residual_rms:.3g} RMS; {top} dBc")
        txt = line(ha, a[0], a[1], COLOR_A)
        if hb is not None:
            txt += "<br>" + line(hb, b[0], b[1], COLOR_B)
        if ha.orders.size < self.n_harm:
            txt += (f"<br><span style='color:gray'>Orders {ha.orders.size + 1}…{self.n_harm} are above Nyquist "
                    f"({fs / 2:g} Hz) at f₀ = {ha.f0:g} Hz and are not shown.</span>")
        self.info.setText(txt)
