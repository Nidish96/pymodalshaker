# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Strobed view: samples of the latest block(s) folded onto one period of the base frequency,
0 … T = 2π/Ω, as a point cloud, overlaid with the fitted Fourier series (DC + harmonics 1…H).

Used by the stepped-sine suite (base frequency = current step) and the harmonic suite (f0).
Every block holds k whole periods (the frequency sits on FFT line k), so each block starts at
phase 0 and blocks of the same frequency overlay exactly."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSpinBox, QSplitter, QVBoxLayout, QWidget

from ..harmonics import ls_harmonics
from .plots import COLOR_A, COLOR_B

MAX_POINTS = 40000          # per channel and redraw (decimated above this)


def fourier_fit(blocks: np.ndarray, k: int, H: int) -> Tuple[float, np.ndarray]:
    """DC and complex harmonics 1…H (cosine convention, peak) of blocks (nb, N), averaged over blocks."""
    n = blocks.shape[-1]
    m = np.arange(n)
    dc = float(blocks.mean())
    X = np.full(H, np.nan + 0j)
    for h in range(1, H + 1):
        if h * k >= n // 2:
            break
        X[h - 1] = (2.0 / n * blocks @ np.exp(-2j * np.pi * h * k * m / n)).mean()
    return dc, X


def eval_fourier(dc: float, X: np.ndarray, tau: np.ndarray, T: float) -> np.ndarray:
    y = np.full(tau.shape, dc)
    for h, z in enumerate(X, start=1):
        if np.isfinite(z):
            y += np.real(z * np.exp(2j * np.pi * h * tau / T))
    return y


class StrobedView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.n_over = QSpinBox()
        self.n_over.setRange(1, 100)
        self.n_over.setValue(3)
        self.n_over.setToolTip("Overlay the latest blocks with the same drive (they all start at phase 0)")
        self.fit = QCheckBox("Fourier fit")
        self.fit.setChecked(True)
        self.fit.setToolTip("DC + harmonics 1…H fitted to the overlaid blocks (H set on Output Setup)")
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        top = QHBoxLayout()
        top.addWidget(QLabel("Overlay last"))
        top.addWidget(self.n_over)
        top.addWidget(QLabel("block(s)"))
        top.addSpacing(10)
        top.addWidget(self.fit)
        top.addStretch()

        self.plots, self.pts, self.curves = [], [], []
        self.split = QSplitter(Qt.Vertical)
        for col in (COLOR_A, COLOR_B):
            p = pg.PlotWidget()
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            p.addLegend(offset=(-10, 10))
            c = pg.mkColor(col)
            c.setAlpha(110)
            sp = p.plot(pen=None, symbol="o", symbolSize=3, symbolPen=None, symbolBrush=c, name="samples")
            cv = p.plot(pen=pg.mkPen("#c0392b" if col == COLOR_A else "#16a085", width=2.2), name="Fourier fit")
            self.plots.append(p)
            self.pts.append(sp)
            self.curves.append(cv)
            self.split.addWidget(p)
        self.plots[1].setXLink(self.plots[0])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addWidget(self.info)
        lay.addWidget(self.split, 1)
        self.H = 5
        self._args = None
        self.n_over.valueChanged.connect(self._redraw)
        self.fit.toggled.connect(self._redraw)
        self.clear()

    def set_max_order(self, n: int):
        self.H = max(1, int(n))
        self._redraw()

    def clear(self, msg: str = "No sine data yet."):
        self._args = None
        for sp, cv in zip(self.pts, self.curves):
            sp.setData([], [])
            cv.setData([], [])
        self.plots[1].setVisible(False)
        self.info.setText(f"<span style='color:gray'>{msg}</span>")

    def show_blocks(self, fs: float, k: Optional[int], a, b=None, prev: Optional[List] = None,
                    f: Optional[float] = None, starts: Optional[List[int]] = None):
        """a, b: (name, unit, [signal of each recent block, oldest first]) with the line k of each block
        given in `prev` (list of k per block, same order). Only the trailing blocks with the latest k are used.
        Frequency mode (any frequency, not on an FFT line): pass f and the absolute start sample of every
        block; the samples are folded with their absolute time and the fit is a least-squares fit."""
        self._args = (fs, k, a, b, prev, f, starts)
        self._redraw()

    def _redraw(self, *_):
        if self._args is None:
            return
        fs, k, a, b, ks, fq, starts = self._args
        if fq:
            return self._redraw_f(fs, fq, a, b, ks, starts)
        if not k:
            self.clear("No periodic excitation in the latest block (settling, fade or silence).")
            return
        n_want = self.n_over.value()
        N = len(a[2][-1])
        T = N / (k * fs)                                    # period of the base frequency
        # trailing blocks at the same line
        idx = []
        for i in range(len(ks) - 1, -1, -1):
            if ks[i] != k or len(idx) >= n_want:
                break
            idx.append(i)
        idx = idx[::-1]
        tau_s = (np.arange(N) / fs) % T
        txt = []
        for p_i, (sig, sp, cv, plot) in enumerate(zip((a, b), self.pts, self.curves, self.plots)):
            if sig is None or sig[2] is None or any(sig[2][i] is None for i in idx):
                plot.setVisible(p_i == 0)
                sp.setData([], [])
                cv.setData([], [])
                continue
            plot.setVisible(True)
            name, unit, blocks = sig
            Bk = np.array([blocks[i] for i in idx], float)
            x = np.tile(tau_s, len(idx))
            y = Bk.ravel()
            if y.size > MAX_POINTS:
                step = int(np.ceil(y.size / MAX_POINTS))
                x, y = x[::step], y[::step]
            sp.setData(x, y)
            plot.setLabel("left", name, unit)
            plot.getAxis("left").enableAutoSIPrefix(False)
            if self.fit.isChecked():
                dc, X = fourier_fit(Bk, k, self.H)
                tt = np.linspace(0, T, 400)
                cv.setData(tt, eval_fourier(dc, X, tt, T))
                resid = Bk - eval_fourier(dc, X, np.tile(tau_s, (len(idx), 1)), T)
                ac = np.sqrt(np.mean((Bk - Bk.mean()) ** 2)) + 1e-300
                rr = np.sqrt(np.mean(resid ** 2))
                Hs = int(np.sum(np.isfinite(X)))
                txt.append(f"<b>{name}</b>: fit DC + h1…{Hs}, residual {rr:.3g} {unit} RMS "
                           f"({100 * rr / ac:.2f} % of the AC RMS)")
            else:
                cv.setData([], [])
        for plot in self.plots:
            plot.setLabel("bottom", "Time within one period  t mod T", "s")
        self.plots[0].setXRange(0, T, padding=0.01)
        self.plots[0].setTitle(f"Strobed: {len(idx)} block(s) folded onto T = 2π/Ω = {T * 1e3:.4g} ms "
                               f"(f = {1 / T:.5g} Hz)")
        self.info.setText(" · ".join(txt))

    def _redraw_f(self, fs, fq, a, b, ks, starts):
        """Frequency mode: absolute-time folding and least-squares Fourier fit."""
        n_want = self.n_over.value()
        idx = []
        for i in range(len(ks) - 1, -1, -1):
            if ks[i] is None or len(idx) >= n_want:
                break
            idx.append(i)
        idx = idx[::-1]
        if not idx:
            self.clear("No steady sine in the latest block.")
            return
        T = 1.0 / fq
        txt = []
        for p_i, (sig, sp, cv, plot) in enumerate(zip((a, b), self.pts, self.curves, self.plots)):
            if sig is None or sig[2] is None or any(sig[2][i] is None for i in idx):
                plot.setVisible(p_i == 0)
                sp.setData([], [])
                cv.setData([], [])
                continue
            plot.setVisible(True)
            name, unit, blocks = sig
            xs, ys, dcs, Xs = [], [], [], []
            for i in idx:
                blk = np.asarray(blocks[i], float)
                t_abs = (starts[i] + np.arange(blk.size)) / fs
                xs.append(t_abs % T)
                ys.append(blk)
                dc, X = ls_harmonics(blk, fq, fs, self.H, starts[i])
                dcs.append(float(dc[0]))
                Xs.append(X[0])
            x, y = np.concatenate(xs), np.concatenate(ys)
            if y.size > MAX_POINTS:
                step = int(np.ceil(y.size / MAX_POINTS))
                x, y = x[::step], y[::step]
            sp.setData(x, y)
            plot.setLabel("left", name, unit)
            plot.getAxis("left").enableAutoSIPrefix(False)
            if self.fit.isChecked():
                dc, X = float(np.mean(dcs)), np.nanmean(np.array(Xs), axis=0)
                X = np.where(np.isfinite(np.array(Xs)).all(axis=0), X, np.nan)
                tt = np.linspace(0, T, 400)
                cv.setData(tt, eval_fourier(dc, X, tt, T))
                resid = np.concatenate([np.asarray(ys[m]) - eval_fourier(dc, X, xs[m], T) for m in range(len(idx))])
                yy = np.concatenate(ys)
                ac = np.sqrt(np.mean((yy - yy.mean()) ** 2)) + 1e-300
                rr = np.sqrt(np.mean(resid ** 2))
                txt.append(f"<b>{name}</b>: fit DC + h1…{int(np.sum(np.isfinite(X)))}, residual {rr:.3g} {unit} RMS "
                           f"({100 * rr / ac:.2f} % of the AC RMS)")
            else:
                cv.setData([], [])
        for plot in self.plots:
            plot.setLabel("bottom", "Time within one period  t mod T", "s")
        self.plots[0].setXRange(0, T, padding=0.01)
        self.plots[0].setTitle(f"Strobed: {len(idx)} block(s) folded onto T = 2π/Ω = {T * 1e3:.5g} ms (f = {fq:.6g} Hz)")
        self.info.setText(" · ".join(txt))
