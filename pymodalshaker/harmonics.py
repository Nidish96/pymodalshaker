# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Harmonic decomposition of a stepped-sine block.

The excitation sits on FFT line k (k whole cycles per block of N samples), so harmonic h is
exactly line h·k and the block is a Fourier series in the excitation frequency f0 = k·fs/N:

    x(t) ≈ a0 + Σ_h  A_h · cos(2π·h·f0·t + φ_h),   A_h = |X(h·k)|·2/N,  a0 = mean(x)

No window and no leakage, as long as the block holds whole cycles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Harmonics:
    orders: np.ndarray       # 1 … H
    amp: np.ndarray          # A_h (same units as x)
    phase: np.ndarray        # φ_h [rad]
    dc: float
    f0: float
    thd: float               # sqrt(Σ_{h≥2} A_h²) / A_1
    residual_rms: float      # RMS of what is not DC or harmonics 1…H (noise, non-harmonic content)
    rms: float               # AC RMS of the block

    def dbc(self, floor: float = -140.0) -> np.ndarray:
        """Harmonic levels in dB relative to the fundamental."""
        return np.maximum(20 * np.log10(self.amp / (self.amp[0] + 1e-300) + 1e-300), floor)

    def series(self, t: np.ndarray, orders=None) -> np.ndarray:
        """Fourier-series value at times t using the given orders (default: all; DC excluded)."""
        sel = self.orders if orders is None else np.asarray(orders)
        y = np.zeros_like(t, dtype=float)
        for h in sel:
            i = int(h) - 1
            y += self.amp[i] * np.cos(2 * np.pi * h * self.f0 * t + self.phase[i])
        return y


def max_order(k: int, n: int) -> int:
    """Highest harmonic below Nyquist for line k of an n-point block."""
    return max(1, (n // 2 - 1) // max(k, 1))


def decompose(x: np.ndarray, k: int, fs: float, n_harm: int = 10) -> Harmonics:
    x = np.asarray(x, dtype=float)
    n = x.size
    H = min(n_harm, max_order(k, n))
    X = np.fft.rfft(x)
    lines = k * np.arange(1, H + 1)
    amp = np.abs(X[lines]) * 2.0 / n
    phase = np.angle(X[lines])
    dc = float(x.mean())
    ac = x - dc
    rms = float(np.sqrt(np.mean(ac ** 2)))
    harm_power = float(np.sum(amp ** 2) / 2.0)
    residual = float(np.sqrt(max(rms ** 2 - harm_power, 0.0)))
    thd = float(np.sqrt(np.sum(amp[1:] ** 2)) / (amp[0] + 1e-300)) if H > 1 else 0.0
    return Harmonics(np.arange(1, H + 1), amp, phase, dc, k * fs / n, thd, residual, rms)


# ----------------------------------------------------------------------------- any frequency
def ls_harmonics(x: np.ndarray, f: float, fs: float, n_harm: int, n0: int = 0):
    """Least-squares Fourier fit at an arbitrary frequency f (no FFT-line restriction):
        x(t) ≈ dc + Σ_h Re{X_h·e^{j2π·h·f·t}},   t = (n0 + n)/fs   (absolute time if n0 = block start)
    x: (N,) or (n_signals, N). Returns (dc, X) with X complex (…, H); orders with h·f ≥ fs/2 are NaN.
    Unbiased for a steady periodic signal even with a non-integer number of periods per block."""
    x = np.atleast_2d(np.asarray(x, float))
    n = x.shape[-1]
    t = (n0 + np.arange(n)) / fs
    hs = [h for h in range(1, n_harm + 1) if h * f < fs / 2]
    cols = [np.ones(n)]
    for h in hs:
        w = 2 * np.pi * h * f * t
        cols += [np.cos(w), np.sin(w)]
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, x.T, rcond=None)          # (1 + 2H', n_signals)
    X = np.full((x.shape[0], n_harm), np.nan + 0j)
    for i, h in enumerate(hs):
        X[:, h - 1] = coef[1 + 2 * i] - 1j * coef[2 + 2 * i]
    return coef[0], X


def decompose_ls(x: np.ndarray, f: float, fs: float, n_harm: int = 10) -> Harmonics:
    """Like decompose(), for any frequency (least squares, time measured from the block start)."""
    x = np.asarray(x, dtype=float)
    dc, X = ls_harmonics(x, f, fs, n_harm)
    X = X[0]
    ok = np.isfinite(X)
    H = int(np.sum(ok)) if ok.any() else 1
    X = X[:H]
    amp, phase = np.abs(X), np.angle(X)
    ac = x - x.mean()
    rms = float(np.sqrt(np.mean(ac ** 2)))
    harm_power = float(np.sum(amp ** 2) / 2.0)
    residual = float(np.sqrt(max(rms ** 2 - harm_power, 0.0)))
    thd = float(np.sqrt(np.sum(amp[1:] ** 2)) / (amp[0] + 1e-300)) if H > 1 else 0.0
    return Harmonics(np.arange(1, H + 1), amp, phase, float(dc[0]), f, thd, residual, rms)
