# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Excitation signal generators. Every block is exactly `block_size` samples long and, except for
burst random, periodic in the block, so repeating it (dwell blocks) gives a periodic excitation
whose steady-state response has no leakage with a rectangular window.

Types
-----
Uniform random                  white, uniformly distributed samples (0 … fs/2), exact RMS
Multisine random                equal amplitude on every FFT line in [fmin, fmax], phases U(0, 2π)
Gaussian random (band-limited)  Rayleigh-distributed amplitudes + uniform phases on the band lines:
                                Gaussian-distributed, band-limited, periodic random noise
Periodic chirp                  equal amplitude in band, Schroeder phases (a fast sweep that repeats
                                every block): lowest crest factor -> most force per amplifier volt.
                                Deterministic: every realization is the same block.
Burst random                    band-limited Gaussian noise for the first `burst_fraction` of the
                                block (short cosine tapers), then zero; RMS = RMS while on
Custom spectrum                 user table (f, value) interpolated onto the FFT lines, either as
                                amplitude (linear) or PSD (log-log); phases U(0, 2π)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OutputConfig


@dataclass
class BlockInfo:
    rms: float
    peak: float
    crest_factor: float
    n_lines: int          # number of excited spectral lines, 0 = broadband
    f_lo: float           # first / last excited line
    f_hi: float
    note: str = ""


# ----------------------------------------------------------------------------- helpers
def multisine_lines(cfg: OutputConfig) -> np.ndarray:
    """Indices k of the FFT lines (f = k*fs/N) inside [fmin, fmax], excluding DC and Nyquist."""
    n = cfg.block_size
    k_lo = max(1, int(np.ceil(cfg.fmin / cfg.df - 1e-9)))
    k_hi = min(n // 2 - 1, int(np.floor(cfg.fmax / cfg.df + 1e-9)))
    return np.arange(k_lo, k_hi + 1)


def _from_spectrum(cfg: OutputConfig, amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Real block from one-sided line amplitudes/phases (length N/2+1), scaled to cfg.rms."""
    x = np.fft.irfft(amp * np.exp(1j * phase), cfg.block_size)
    r = np.sqrt(np.mean(x ** 2))
    if r <= 0:
        raise ValueError("The excitation spectrum is empty (all amplitudes are zero on the FFT lines).")
    return x * (cfg.rms / r)


def custom_line_amplitudes(cfg: OutputConfig) -> np.ndarray:
    """Amplitude on every FFT line (length N/2+1) from the custom table; zero outside its range,
    at DC and at Nyquist. Relative values only: the overall level is set by the RMS."""
    pts = np.asarray(cfg.custom_points, dtype=float).reshape(-1, 2)
    fp, ap = pts[:, 0], pts[:, 1]
    n = cfg.block_size
    f = np.arange(n // 2 + 1) * cfg.df
    amp = np.zeros_like(f)
    inside = (f >= fp[0]) & (f <= fp[-1])
    if cfg.custom_kind.startswith("PSD"):
        lf = np.log(np.maximum(f[inside], 1e-12))
        lp = np.log(np.maximum(ap, 1e-300))
        psd = np.exp(np.interp(lf, np.log(np.maximum(fp, 1e-12)), lp))
        psd[psd < 1e-250] = 0.0
        amp[inside] = np.sqrt(psd)
    else:
        amp[inside] = np.interp(f[inside], fp, ap)
    amp[0] = 0.0
    amp[-1] = 0.0
    return amp


# ----------------------------------------------------------------------------- generators
def uniform_random(cfg: OutputConfig, rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(-1.0, 1.0, cfg.block_size)
    x -= x.mean()
    return x * (cfg.rms / np.sqrt(np.mean(x ** 2)))


def multisine_random(cfg: OutputConfig, rng: np.random.Generator) -> np.ndarray:
    k = multisine_lines(cfg)
    amp = np.zeros(cfg.block_size // 2 + 1)
    amp[k] = 1.0
    return _from_spectrum(cfg, amp, rng.uniform(0.0, 2.0 * np.pi, amp.size))


def gaussian_random(cfg: OutputConfig, rng: np.random.Generator) -> np.ndarray:
    k = multisine_lines(cfg)
    amp = np.zeros(cfg.block_size // 2 + 1)
    amp[k] = rng.rayleigh(1.0, k.size)
    return _from_spectrum(cfg, amp, rng.uniform(0.0, 2.0 * np.pi, amp.size))


def periodic_chirp(cfg: OutputConfig, rng: np.random.Generator = None) -> np.ndarray:
    k = multisine_lines(cfg)
    amp = np.zeros(cfg.block_size // 2 + 1)
    phase = np.zeros_like(amp)
    amp[k] = 1.0
    i = np.arange(1, k.size + 1)
    phase[k] = -np.pi * i * (i - 1) / k.size          # Schroeder phases
    return _from_spectrum(cfg, amp, phase)


def burst_random(cfg: OutputConfig, rng: np.random.Generator) -> np.ndarray:
    n = cfg.block_size
    on = max(8, int(round(np.clip(cfg.burst_fraction, 0.05, 1.0) * n)))
    noise = gaussian_random(cfg, rng)
    if on >= n:
        return noise
    gate = np.zeros(n)
    gate[:on] = 1.0
    tap = max(2, min(on // 8, n // 50))                  # short cosine tapers at both ends
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(tap) / tap)
    gate[:tap] = ramp
    gate[on - tap:on] = ramp[::-1]
    x = noise * gate
    r_on = np.sqrt(np.mean(x[:on] ** 2))
    return x * (cfg.rms / r_on)


def custom_spectrum(cfg: OutputConfig, rng: np.random.Generator) -> np.ndarray:
    amp = custom_line_amplitudes(cfg)
    return _from_spectrum(cfg, amp, rng.uniform(0.0, 2.0 * np.pi, amp.size))


_GENERATORS = {
    "Uniform random": uniform_random,
    "Multisine random": multisine_random,
    "Gaussian random (band-limited)": gaussian_random,
    "Periodic chirp": periodic_chirp,
    "Burst random": burst_random,
    "Custom spectrum": custom_spectrum,
}


def generate_block(cfg: OutputConfig, rng: np.random.Generator) -> np.ndarray:
    try:
        gen = _GENERATORS[cfg.excitation]
    except KeyError:
        raise ValueError(f"Unknown excitation type: {cfg.excitation}") from None
    return gen(cfg, rng)


def block_info(cfg: OutputConfig, x: np.ndarray) -> BlockInfo:
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    t = cfg.excitation
    if t == "Burst random":
        on = int(round(np.clip(cfg.burst_fraction, 0.05, 1.0) * x.size))
        r_on = float(np.sqrt(np.mean(x[:on] ** 2)))
        k = multisine_lines(cfg)
        return BlockInfo(r_on, peak, peak / r_on, k.size, k[0] * cfg.df, k[-1] * cfg.df,
                         f"burst {cfg.burst_fraction * 100:.0f} % of block, then silence")
    if t in ("Multisine random", "Gaussian random (band-limited)", "Periodic chirp"):
        k = multisine_lines(cfg)
        note = "deterministic: identical block in every realization" if t == "Periodic chirp" else ""
        return BlockInfo(rms, peak, peak / rms, k.size, k[0] * cfg.df, k[-1] * cfg.df, note)
    if t == "Custom spectrum":
        amp = custom_line_amplitudes(cfg)
        nz = np.flatnonzero(amp > 0)
        if nz.size:
            return BlockInfo(rms, peak, peak / rms, nz.size, nz[0] * cfg.df, nz[-1] * cfg.df)
    return BlockInfo(rms, peak, peak / rms, 0, 0.0, cfg.fs / 2)


def fade_window(n: int, rising: bool = True) -> np.ndarray:
    """Half-cosine ramp 0→1 (or 1→0) over n samples."""
    w = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / (n - 1))
    return w if rising else w[::-1]
