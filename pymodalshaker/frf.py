# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""FRF estimation (H1, H2, Hv) and ordinary coherence from averaged auto/cross spectra.

Convention: X = reference (force) spectrum, Y = response spectrum.
    Gxx = <|X|²>,  Gyy = <|Y|²>,  Gxy = <X* · Y>
    H1  = Gxy / Gxx                 (noise on output)
    H2  = Gyy / Gxy*                (noise on input)
    Hv  = Gxy / (Gxx − λmin)        (total least squares; λmin = smallest eigenvalue of
                                     [[Gxx, Gxy*], [Gxy, Gyy]]; lies between H1 and H2)
    γ²  = |Gxy|² / (Gxx · Gyy)
Window scaling cancels in all ratios, so spectra are left unscaled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

_EPS = 1e-300


_WINDOW_SPECS = {
    "Rectangular": "boxcar",
    "Hann": "hann",
    "Hamming": "hamming",
    "Blackman": "blackman",
    "Blackman-Harris": "blackmanharris",
    "Flat top": "flattop",
    "Tukey (10 %)": ("tukey", 0.1),
    "Kaiser-Bessel (β = 6)": ("kaiser", 6.0),
}


def make_window(name: str, n: int) -> np.ndarray:
    """Periodic (DFT-even) window of length n. Scaling is irrelevant: it cancels in H and γ²."""
    from scipy.signal import get_window
    spec = _WINDOW_SPECS.get(name)
    if spec is None:
        raise ValueError(f"Unknown window '{name}'. Choose one of: {', '.join(_WINDOW_SPECS)}")
    return get_window(spec, n, fftbins=True)


@dataclass
class FRFResult:
    freqs: np.ndarray       # (nf,)
    H1: np.ndarray          # (n_resp, nf) complex
    H2: np.ndarray
    Hv: np.ndarray
    coherence: np.ndarray   # (n_resp, nf) real (stepped sine: fundamental fraction η)
    n_avg: int
    harm: Optional[np.ndarray] = None   # stepped sine: (nf, n_channels, H) complex harmonic amplitudes
    ref_i: int = -1                     # channel index of the reference within `harm`
    resp_i: Optional[List[int]] = None  # channel indices of the responses (FRF row order)


def harmonic_frf(res: "FRFResult", j: int, h: int):
    """Stepped sine: response harmonic h over the reference fundamental, Y_h / X_1, and the
    response's harmonic-h level relative to its own fundamental in dBc. NaN where h·f0 ≥ fs/2."""
    ri = res.resp_i[j]
    Y = res.harm[:, ri, :]
    X1 = res.harm[:, res.ref_i, 0]
    Hh = Y[:, h - 1] / (X1 + 1e-300)
    dbc = 20 * np.log10(np.abs(Y[:, h - 1]) / (np.abs(Y[:, 0]) + 1e-300) + 1e-300)
    return Hh, dbc


class FRFEstimator:
    def __init__(self, n_resp: int, block_size: int, fs: float, window: str = "Rectangular"):
        self.n = block_size
        self.fs = fs
        self.win = make_window(window, block_size)
        nf = block_size // 2 + 1
        self.freqs = np.arange(nf) * fs / block_size
        self.Gxx = np.zeros(nf)
        self.Gyy = np.zeros((n_resp, nf))
        self.Gxy = np.zeros((n_resp, nf), dtype=complex)
        self.n_avg = 0

    def add(self, ref: np.ndarray, resp: np.ndarray) -> None:
        """ref: (N,) reference block; resp: (n_resp, N) response blocks."""
        X = np.fft.rfft(ref * self.win)
        Y = np.fft.rfft(resp * self.win, axis=-1)
        self.Gxx += np.abs(X) ** 2
        self.Gyy += np.abs(Y) ** 2
        self.Gxy += np.conj(X)[None, :] * Y
        self.n_avg += 1

    def result(self) -> FRFResult:
        k = max(self.n_avg, 1)
        H1, H2, Hv, coh = estimates(self.Gxx / k, self.Gyy / k, self.Gxy / k)
        return FRFResult(self.freqs, H1, H2, Hv, coh, self.n_avg)


def estimates(Gxx, Gyy, Gxy):
    """H1, H2, Hv, coherence from averaged spectra. Gxx broadcasts against Gyy/Gxy (…, nf)."""
    Gxx = np.asarray(Gxx)[..., None, :] if np.ndim(Gyy) > np.ndim(Gxx) else np.asarray(Gxx)
    absGxy2 = np.abs(Gxy) ** 2
    H1 = Gxy / (Gxx + _EPS)
    H2 = Gyy / (np.conj(Gxy) + _EPS)
    lam_min = 0.5 * (Gxx + Gyy) - np.sqrt((0.5 * (Gxx - Gyy)) ** 2 + absGxy2)
    Hv = Gxy / (Gxx - lam_min + _EPS)
    coh = np.clip(absGxy2 / (Gxx * Gyy + _EPS), 0.0, 1.0)
    return H1, H2, Hv, coh
