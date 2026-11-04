# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Output Setup tab: drive channel, excitation type/level, block timing, IEPE settling."""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame,
                               QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSpinBox, QSplitter,
                               QVBoxLayout, QWidget)

from ..config import PQ_TRIGGER_LABELS, PQConfig, harmonic_nyquist_note
from ..config import (BAND_TYPES, CLOCK_MODES, CONTROL_MODES, EXCITATION_TYPES, FAIL_ACTIONS, HARMONIC_TRIGGER_LABELS,
                      HarmonicConfig, harmonic_line, harmonic_waveform,
                      FRAME_SYNC, SPACINGS, SUITES, TARGET_KINDS, TRIGGER_MODES, TRIGGER_POLARITY, WINDOWS, OutputConfig,
                      SteppedConfig, TriggerConfig, custom_points_problems, output_kind, stepped_frequencies)
from ..excitation import block_info, custom_line_amplitudes, generate_block
from .custom_spectrum import CustomSpectrumEditor
from .harmonic_setup import HarmonicSetup
from .pq_setup import PQSetup

STEPPED_TRIGGER_LABELS = [
    "Once, at the start of the acquisition",
    "At the start of the converged (dwell) block at every frequency",
    "At the start of every frequency step",
    "Frame sync: one pulse per sample (AI sample clock → PFI)",
]
STEPPED_TRIGGER_TIPS = [
    "One pulse at sample 0, when the input task starts",
    "One pulse per frequency, at the first sample of the analysed dwell (after control has converged)",
    "One pulse per frequency, at the first sample after the frequency changes (before control/settling)",
    "Routes the input task's sample clock to a PFI pin: one pulse per sample. Frame f = sample f.",
]
PQ_TRIGGER_TIPS = [
    "One pulse at sample 0, when the input task starts",
    "One pulse per level, at the first sample of its converged (recorded) block",
    "One pulse per level, at the first sample of the level (before the frequency search)",
    "Routes the input task's sample clock to a PFI pin: one pulse per sample. Frame f = sample f.",
]
HARMONIC_TRIGGER_TIPS = [
    "One pulse at sample 0, when the input task starts",
    "One pulse at the first sample of the converged test blocks (after control has converged)",
    "One pulse at the first sample of every control iteration (every drive change)",
    "Routes the input task's sample clock to a PFI pin: one pulse per sample. Frame f = sample f.",
]
RANDOM_TRIGGER_TIPS = [
    "One pulse at sample 0, i.e. when the input task starts",
    "One pulse per realization, at the first sample of its last (converged) dwell block",
    "One pulse per realization, at the first sample of its first dwell block",
    "Routes the input task's sample clock to a PFI pin: one pulse per sample from sample 0 to the end. "
    "Drive the camera's external-sync (frame clock) input with it: frame f = sample f, no drift.",
]

EXC_TIPS = [
    "White random samples (uniform distribution) over 0 … fs/2, scaled to the RMS.",
    "Equal-amplitude sines on every FFT line in the band, random phases. Leakage-free once periodic.",
    "Rayleigh amplitudes + random phases on the band lines: band-limited, Gaussian, periodic random noise.",
    "Fast sweep through the band, repeated every block (Schroeder phases). Lowest crest factor, i.e. "
    "the most force for a given amplifier voltage. Deterministic: every realization is identical, "
    "averaging only reduces noise.",
    "Band-limited random for the first part of each block, then silence so the response decays "
    "inside the block. Leakage-free without a window even for lightly damped structures.",
    "Your own spectrum shape from a frequency / amplitude (or PSD) table, random phases, scaled to the RMS.",
]
WINDOW_TIP = ("Rectangular is leakage-free for converged periodic excitation (all types here once the dwell "
              "blocks have settled, and burst random). Use Hann / Hamming / Blackman(-Harris) / Kaiser if "
              "responses are not periodic in the block, Flat top for accurate amplitudes of sines, Tukey "
              "for a near-rectangular window with soft edges.")

COMMON_RATES = ["1024", "2048", "2560", "5120", "10240", "12800", "25600", "51200"]
BLOCK_SIZES = [str(2 ** k) for k in range(8, 18)]


def _dspin(lo, hi, dec, suffix="", step=None):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    if suffix:
        w.setSuffix(" " + suffix)
    if step:
        w.setSingleStep(step)
    return w


def _ispin(lo, hi, suffix=""):
    w = QSpinBox()
    w.setRange(lo, hi)
    if suffix:
        w.setSuffix(" " + suffix)
    return w


class OutputSetup(QWidget):
    changed = Signal()
    check_rate_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # ---------------- drive channel
        self.ao = QComboBox()
        self.ao.setEditable(True)
        self.vmax = _dspin(0.01, 10, 3, "V", 0.1)
        self.vmax.setToolTip("Safety limit: the run is refused if any drive sample would exceed this")
        self.clock = QComboBox()
        self.clock.addItems(CLOCK_MODES)
        self.clock.setToolTip("Shared start trigger: AO and AI start together, each at the same rate.\n"
                              "AO clocked by AI sample clock: exact sample lockstep (not supported "
                              "by every module combination).")
        g_out = QGroupBox("Shaker drive (analog output)")
        f = QFormLayout(g_out)
        f.addRow("AO channel", self.ao)
        f.addRow("Drive limit  |V| ≤", self.vmax)
        f.addRow("Synchronisation", self.clock)

        # ---------------- excitation
        self.exc = QComboBox()
        self.exc.addItems(EXCITATION_TYPES)
        for i, tip in enumerate(EXC_TIPS):
            self.exc.setItemData(i, tip, 3)
        self.rms = _dspin(0.001, 10, 3, "V RMS", 0.05)
        self.fmin = _dspin(0, 1e5, 2, "Hz")
        self.fmax = _dspin(0.1, 1e5, 2, "Hz")
        for w in (self.fmin, self.fmax):
            w.setToolTip("Multisine: the band that is excited. Uniform random: only marked on the plots.\n"
                         "It never limits what is saved — FRFs are always stored from Δf to fs/2.")
        self.window = QComboBox()
        self.window.addItems(WINDOWS)
        self.window.setToolTip(WINDOW_TIP)
        self.burst = _dspin(5, 100, 0, "% of block", 5)
        self.burst.setToolTip("Burst random: the part of each block that is driven; the rest is silence "
                              "so the response can decay within the block.")
        self.custom = CustomSpectrumEditor()
        self.seed = _ispin(-1, 2 ** 31 - 1)
        self.seed.setSpecialValueText("random each run")
        self.seed.setToolTip("Fix the seed to repeat exactly the same random blocks")
        g_exc = QGroupBox("Excitation")
        f = QFormLayout(g_exc)
        f.addRow("Type", self.exc)
        f.addRow("Level", self.rms)
        fb = QHBoxLayout()
        fb.addWidget(self.fmin)
        fb.addWidget(QLabel("to"))
        fb.addWidget(self.fmax)
        self.band_label = QLabel("Band")
        f.addRow(self.band_label, fb)
        f.addRow("Burst length", self.burst)
        f.addRow(self.custom)
        f.addRow("FFT window", self.window)
        f.addRow("Random seed", self.seed)
        self.exc_form = f
        self.exc_hint = QLabel()
        self.exc_hint.setWordWrap(True)
        self.exc_hint.setStyleSheet("color: gray")
        f.insertRow(1, self.exc_hint)

        # ---------------- suite selector
        self.suite = QComboBox()
        self.suite.addItems(SUITES)
        self.suite.setItemData(0, "Broadband periodic/burst random, multisine, chirp or custom spectra; "
                                  "all lines measured at once, averaged over realizations", 3)
        self.suite.setItemData(1, "One frequency at a time with optional closed-loop amplitude control; "
                                  "best SNR and level control, slower", 3)
        self.suite.setItemData(2, "One base frequency with a list of harmonics (amplitude and phase each), "
                                  "fixed or controlled on a response channel; time data only", 3)
        self.g_harm = HarmonicSetup()
        self.suite.setItemData(3, "Resonance tracking at increasing drive levels: the frequency is iterated to "
                                  "phase quadrature (or maximum |H1|) at every level; amplitude-dependent resonance", 3)
        self.g_pq = PQSetup()
        g_suite = QGroupBox("Excitation suite")
        sl = QFormLayout(g_suite)
        sl.addRow("Suite", self.suite)
        self.g_exc = g_exc

        # ---------------- stepped sine
        self.ss_start = _dspin(0.01, 1e5, 2, "Hz")
        self.ss_stop = _dspin(0.01, 1e5, 2, "Hz")
        self.ss_step = _dspin(0.001, 1e5, 3, "Hz")
        self.ss_spacing = QComboBox()
        self.ss_spacing.addItems(SPACINGS)
        self.ss_plan = QLabel()
        self.ss_plan.setWordWrap(True)
        self.ss_plan.setStyleSheet("color: gray")
        self.ss_settle = _ispin(0, 100, "blocks")
        self.ss_settle.setToolTip("Blocks discarded after every change of frequency or drive level, so that "
                                  "transients die out before measuring")
        self.ss_dwell = _dspin(0.001, 3600, 3, "s", 0.1)
        self.ss_dwell.setToolTip("Steady-state time analysed at each frequency (rounded up to whole blocks)")
        self.ss_dwell_n = QLabel()
        self.ss_control = QComboBox()
        self.ss_control.addItems(CONTROL_MODES)
        self.ss_drive = _dspin(0.001, 10, 4, "V peak", 0.05)
        self.ss_drive.setToolTip("No control: the same sine amplitude at every frequency")
        self.ss_target_ch = QComboBox()
        self.ss_target_ch.setToolTip("Channel whose block RMS is controlled")
        self.ss_target = _dspin(1e-9, 1e9, 5, "", 0.1)
        self.ss_target_unit = QLabel()
        self.ss_target_kind = QComboBox()
        self.ss_target_kind.addItems(TARGET_KINDS)
        self.ss_target_kind.setToolTip("RMS: the control channel's block RMS is controlled.\n"
                                       "Amplitude (RMS·√2): the equivalent single-harmonic amplitude, i.e. the "
                                       "peak of a pure sine with the same RMS.")
        self.ss_init = _dspin(0.0001, 10, 4, "V peak", 0.01)
        self.ss_init.setToolTip("Drive used for the first iteration at the first frequency. Later frequencies "
                                "start from the previous converged drive.")
        self.ss_tol = _dspin(0.1, 50, 1, "%", 0.5)
        self.ss_maxit = _ispin(2, 100)
        self.ss_maxit.setToolTip("Control iterations allowed per frequency. Every frequency uses at least two: "
                                 "the first measurement with the carried-over drive, then at least one refinement.")
        self.ss_nharm = _ispin(2, 20)
        self.ss_nharm.setToolTip("Harmonic orders 1…n used everywhere: the Block harmonics display, the "
                                 "Harmonics vs frequency page, and the saved harmonics CSV / UFF files. "
                                 "Orders above Nyquist are left out automatically.")
        self.ss_fail = QComboBox()
        self.ss_fail.addItems(FAIL_ACTIONS)
        self.ss_fail.setToolTip("What to do when control does not converge (max iterations), the drive limit "
                                "is reached, or an input overloads before the target is reached")
        self.ss_over = _dspin(50, 100, 1, "% of range", 1)
        self.ss_over.setToolTip("A channel is overloaded when its |peak| reaches this fraction of its input range")
        for cb in (self.ss_spacing, self.ss_control, self.ss_fail, self.ss_target_ch, self.suite):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(12)
        self.g_step = QGroupBox("Stepped sine")
        f = QFormLayout(self.g_step)
        fr = QHBoxLayout()
        fr.addWidget(self.ss_start)
        fr.addWidget(QLabel("to"))
        fr.addWidget(self.ss_stop)
        f.addRow("Frequencies", fr)
        self.ss_step_label = QLabel("Step")
        f.addRow(self.ss_step_label, self.ss_step)
        f.addRow("Spacing", self.ss_spacing)
        f.addRow(self.ss_plan)
        f.addRow("Settle after change", self.ss_settle)
        dw = QHBoxLayout()
        dw.addWidget(self.ss_dwell, 1)
        dw.addWidget(self.ss_dwell_n)
        f.addRow("Dwell per frequency", dw)
        f.addRow("Control", self.ss_control)
        f.addRow("Drive amplitude", self.ss_drive)
        f.addRow("Control channel", self.ss_target_ch)
        tg = QHBoxLayout()
        tg.addWidget(self.ss_target, 1)
        tg.addWidget(self.ss_target_unit)
        tg.addWidget(self.ss_target_kind)
        f.addRow("Target", tg)
        f.addRow("Initial drive", self.ss_init)
        f.addRow("Tolerance  ±", self.ss_tol)
        f.addRow("Max iterations", self.ss_maxit)
        f.addRow("If control fails", self.ss_fail)
        f.addRow("Input overload at", self.ss_over)
        f.addRow("Harmonics: orders 1 …", self.ss_nharm)
        self.ss_form = f
        self._ch_units = {}

        # ---------------- timing
        self.fs = QComboBox()
        self.fs.setEditable(True)
        self.fs.addItems(COMMON_RATES)
        self.fs.setToolTip("Samples/s. Delta-sigma modules (e.g. NI 9234) only support discrete rates; "
                           "the hardware-coerced rate is reported when the run starts.")
        self.n = QComboBox()
        self.n.setEditable(True)
        self.n.addItems(BLOCK_SIZES)
        self.n.setToolTip("Samples read and written per block (= FFT length)")
        self.dwell = _ispin(1, 10000, "blocks")
        self.dwell.setToolTip("Each random block is output this many times in a row. The response "
                              "becomes periodic; only the last (converged) block is used.")
        self.real = _ispin(1, 100000)
        self.real.setToolTip("Number of different random blocks = number of FRF averages")
        self.decay = _dspin(0, 600, 1, "s", 1)
        self.decay.setToolTip("Drive held at 0 V while the IEPE signal conditioners settle (DC bias "
                              "decay). Data in this period is shown but discarded.")
        self.fade = QCheckBox("Fade drive in / out (one block each)")
        self.queue = _ispin(2, 50, "blocks")
        self.queue.setToolTip("AO queue depth. Larger = tolerant of PC/USB hiccups, but more latency")
        g_tim = QGroupBox("Blocks && timing")
        f = QFormLayout(g_tim)
        self.tim_form = f
        self.btn_rate = QPushButton("Check with hardware")
        self.btn_rate.setToolTip("Ask NI-DAQmx which sample rate it would really use for the enabled input "
                                 "channels and the drive channel (nothing is started), and list the rates "
                                 "they all support exactly")
        self.btn_rate.clicked.connect(self.check_rate_requested)
        fsr = QHBoxLayout()
        fsr.addWidget(self.fs, 1)
        fsr.addWidget(self.btn_rate)
        f.addRow("Sampling rate", fsr)
        self.rate_info = QLabel()
        self.rate_info.setWordWrap(True)
        self.rate_info.setTextFormat(Qt.RichText)
        self.rate_info.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.rate_info.linkActivated.connect(lambda r: self.fs.setCurrentText(r))
        self.rate_info.setVisible(False)
        f.addRow(self.rate_info)
        f.addRow("Block size", self.n)
        f.addRow("Dwell blocks", self.dwell)
        f.addRow("Realizations", self.real)
        f.addRow("IEPE decay time", self.decay)
        f.addRow("", self.fade)
        f.addRow("AO queue depth", self.queue)

        # ---------------- external (camera) trigger
        self.trg = QGroupBox("External trigger (e.g. high-speed camera)")
        self.trg.setCheckable(True)
        self.trg.setChecked(False)
        self.trg.setToolTip("Output a hardware-timed pulse on a second AO channel or a DO line, on the "
                            "same sample clock as the acquisition.")
        self.trg_chan = QComboBox()
        self.trg_chan.setEditable(True)
        self.trg_chan.setToolTip("Another AO channel (…/aoN) or a digital line (…/portN/lineM). A DO line "
                                 "needs a hardware-timed (correlated) DIO module such as the NI 9401.")
        self.trg_kind = QLabel()
        self.trg_mode = QComboBox()
        self.trg_mode.addItems(TRIGGER_MODES)        # relabelled per suite in _relabel_trigger_modes
        self.trg_mode.setItemData(0, "One pulse at sample 0, i.e. when the input task starts", 3)
        self.trg_mode.setItemData(1, "One pulse per realization, at the first sample of its last "
                                     "(converged) dwell block", 3)
        self.trg_mode.setItemData(2, "One pulse per realization, at the first sample of its first dwell block", 3)
        self.trg_mode.setItemData(3, "Routes the input task's sample clock to a PFI pin: one pulse per sample "
                                     "from sample 0 to the end. Drive the camera's external-sync (frame clock) "
                                     "input with it: frame f = sample f, no drift.", 3)
        self.trg_width = _ispin(1, 1_000_000, "samples")
        self.trg_width_t = QLabel()
        self.trg_level = _dspin(0.1, 10, 2, "V", 0.5)
        self.trg_level.setToolTip("Active level of an analog trigger. TTL inputs read ≥ 2 V as high; 3.3 V fits ±4.24 V modules such as the NI 9260")
        self.trg_pol = QComboBox()
        self.trg_pol.addItems(TRIGGER_POLARITY)
        for cb in (self.trg_chan, self.trg_mode, self.trg_pol, self.exc, self.clock, self.window):
            # don't let long item texts widen the settings column (full text shows in the drop-down)
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(12)
        self.trg_mode.view().setMinimumWidth(430)
        f = QFormLayout(self.trg)
        tc = QHBoxLayout()
        tc.addWidget(self.trg_chan, 1)
        tc.addWidget(self.trg_kind)
        f.addRow("Channel", tc)
        f.addRow("Send pulse", self.trg_mode)
        tw = QHBoxLayout()
        tw.addWidget(self.trg_width, 1)
        tw.addWidget(self.trg_width_t)
        f.addRow("Pulse width", tw)
        f.addRow("Level (AO only)", self.trg_level)
        f.addRow("Polarity", self.trg_pol)

        # ---------------- summary + preview
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("QLabel { background: palette(alternate-base); padding: 8px; }")
        self.btn_preview = QPushButton("↻ New preview block")
        self.p_time = pg.PlotWidget(title="Drive block (one realization)")
        self.p_time.setLabel("bottom", "Time", "s")
        self.p_time.setLabel("left", "Drive", "V")
        self.p_spec = pg.PlotWidget(title="Drive spectrum")
        self.p_spec.setLabel("bottom", "Frequency", "Hz")
        self.p_spec.setLabel("left", "Amplitude", "dB re 1 V")
        self.p_spec.getAxis("bottom").enableAutoSIPrefix(False)
        for p in (self.p_time, self.p_spec):
            p.showGrid(x=True, y=True, alpha=0.3)
        self.c_time = self.p_time.plot(pen=pg.mkPen("#2b7bb9", width=1))
        self.lim_hi = pg.InfiniteLine(angle=0, pen=pg.mkPen("#c0392b", style=pg.QtCore.Qt.DashLine))
        self.lim_lo = pg.InfiniteLine(angle=0, pen=pg.mkPen("#c0392b", style=pg.QtCore.Qt.DashLine))
        self.p_time.addItem(self.lim_hi)
        self.p_time.addItem(self.lim_lo)
        self.c_spec = self.p_spec.plot(pen=pg.mkPen("#2b7bb9", width=1))
        self.p_trig = pg.PlotWidget(title="Trigger pulse (one event)")
        self.p_trig.showGrid(x=True, y=True, alpha=0.3)
        self.p_trig.getAxis("bottom").enableAutoSIPrefix(False)
        self.p_trig.getAxis("left").enableAutoSIPrefix(False)
        self.c_trig = self.p_trig.plot(pen=pg.mkPen("#8e44ad", width=2), stepMode="center")
        self.c_trig_pts = self.p_trig.plot(pen=None, symbol="o", symbolSize=5,
                                           symbolBrush="#8e44ad", symbolPen=None)
        self.trig_zero = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("#7f8c8d", style=pg.QtCore.Qt.DashLine))
        self.p_trig.addItem(self.trig_zero)
        self.p_trig.setVisible(False)

        left = QVBoxLayout()
        for g in (g_suite, g_out, g_exc, self.g_step, self.g_harm, self.g_pq, g_tim, self.trg):
            left.addWidget(g)
        left.addStretch()
        rw = QWidget()
        right = QVBoxLayout(rw)
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(self.summary)
        hb = QHBoxLayout()
        hb.addStretch()
        hb.addWidget(self.btn_preview)
        right.addLayout(hb)
        self.preview_split = QSplitter(Qt.Vertical)       # previews resizable against each other
        for p in (self.p_time, self.p_spec, self.p_trig):
            p.setMinimumHeight(120)
            self.preview_split.addWidget(p)
        right.addWidget(self.preview_split, 1)
        lw = QWidget()
        lw.setLayout(left)
        scroll = QScrollArea()             # settings column scrolls on small screens
        scroll.setWidget(lw)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(380)
        self.main_split = QSplitter(Qt.Horizontal)       # drag the divider to resize the settings
        self.main_split.addWidget(scroll)
        self.main_split.addWidget(rw)
        self.main_split.setStretchFactor(0, 0)
        self.main_split.setStretchFactor(1, 1)
        self.main_split.setSizes([500, 900])
        self.main_split.setChildrenCollapsible(False)
        lay = QVBoxLayout(self)
        lay.addWidget(self.main_split)

        self._preview_seed = 0
        for w in (self.vmax, self.rms, self.fmin, self.fmax, self.decay):
            w.valueChanged.connect(self._on_change)
        for w in (self.dwell, self.real, self.queue, self.seed):
            w.valueChanged.connect(self._on_change)
        for w in (self.exc, self.window, self.clock):
            w.currentTextChanged.connect(self._on_change)
        for w in (self.fs, self.n):
            w.currentTextChanged.connect(self._on_change)
        self.fs.currentTextChanged.connect(lambda _: self._rate_stale())
        self.fade.toggled.connect(self._on_change)
        self.burst.valueChanged.connect(self._on_change)
        for w in (self.ss_start, self.ss_stop, self.ss_step, self.ss_dwell, self.ss_drive, self.ss_target,
                  self.ss_init, self.ss_tol, self.ss_over, self.ss_settle, self.ss_maxit, self.ss_nharm):
            w.valueChanged.connect(self._on_change)
        for w in (self.suite, self.ss_spacing, self.ss_control, self.ss_fail, self.ss_target_ch, self.ss_target_kind):
            w.currentTextChanged.connect(self._on_change)
        self.custom.changed.connect(self._on_change)
        self.g_harm.changed.connect(self._on_change)
        self.g_pq.changed.connect(self._on_change)
        self.btn_preview.clicked.connect(self._new_preview)
        self.trg.toggled.connect(self._on_change)
        self.trg_chan.currentTextChanged.connect(self._on_change)
        self.trg_mode.currentTextChanged.connect(self._mode_changed)
        self.trg_pol.currentTextChanged.connect(self._on_change)
        self.trg_width.valueChanged.connect(self._on_change)
        self.trg_level.valueChanged.connect(self._on_change)

    # ------------------------------------------------------------------ data
    def set_output_names(self, ao: List[str], do: List[str], pfi: List[str] = ()):
        self._names = {"pulse": list(ao) + list(do), "pfi": list(pfi)}
        cur = self.ao.currentText()
        self.ao.clear()
        self.ao.addItems(ao)
        self.ao.setCurrentText(cur)
        self._fill_trigger_names()

    def _fill_trigger_names(self):
        names = getattr(self, "_names", {"pulse": [], "pfi": []})
        frame = self.trg_mode.currentIndex() == TRIGGER_MODES.index(FRAME_SYNC)
        cur = self.trg_chan.currentText().strip()
        memory = self.__dict__.setdefault("_trig_memory", {})
        if output_kind(cur):                       # remember the last choice per mode family
            memory["pfi" if output_kind(cur) == "PFI" else "pulse"] = cur
        want = "pfi" if frame else "pulse"
        options = [n for n in names[want] if n != self.ao.currentText().strip()]   # never the drive
        self.trg_chan.blockSignals(True)
        self.trg_chan.clear()
        self.trg_chan.addItems(options)
        pick = memory.get(want) or (options[0] if options else "")
        self.trg_chan.setCurrentText(pick)
        self.trg_chan.blockSignals(False)

    def set_channel_names(self, chans, ref_name: str = ""):
        """chans: [(name, units)] of the enabled input channels (control-channel choices)."""
        self._ch_units = dict(chans)
        cur = self.ss_target_ch.currentText()
        self.ss_target_ch.blockSignals(True)
        self.ss_target_ch.clear()
        self.ss_target_ch.addItems([n for n, _ in chans])
        if cur:
            self.ss_target_ch.setCurrentText(cur)
        self.ss_target_ch.blockSignals(False)
        self.g_harm.set_channels(chans)
        self.g_pq.set_channels(chans, ref_name)
        self._on_change()

    def get_suite(self) -> str:
        return self.suite.currentText()

    def get_stepped(self) -> SteppedConfig:
        return SteppedConfig(
            f_start=self.ss_start.value(), f_stop=self.ss_stop.value(), f_step=self.ss_step.value(),
            spacing=self.ss_spacing.currentText(), settle_blocks=self.ss_settle.value(),
            dwell_time=self.ss_dwell.value(), control=self.ss_control.currentText(),
            drive_amplitude=self.ss_drive.value(), target_channel=self.ss_target_ch.currentText(),
            target_value=self.ss_target.value(), target_kind=self.ss_target_kind.currentText(),
            initial_drive=self.ss_init.value(),
            tolerance_pct=self.ss_tol.value(), max_iterations=self.ss_maxit.value(),
            on_failure=self.ss_fail.currentText(), overload_pct=self.ss_over.value(),
            n_harmonics=self.ss_nharm.value())

    def set_stepped(self, suite: str, st: SteppedConfig):
        self._loading = True
        self.suite.setCurrentText(suite)
        self.ss_start.setValue(st.f_start)
        self.ss_stop.setValue(st.f_stop)
        self.ss_spacing.setCurrentText(st.spacing)
        self.ss_step.setValue(st.f_step)
        self.ss_settle.setValue(st.settle_blocks)
        self.ss_dwell.setValue(st.dwell_time)
        self.ss_control.setCurrentText(st.control)
        self.ss_drive.setValue(st.drive_amplitude)
        if self.ss_target_ch.findText(st.target_channel) < 0 and st.target_channel:
            self.ss_target_ch.addItem(st.target_channel)
        self.ss_target_ch.setCurrentText(st.target_channel)
        self.ss_target.setValue(st.target_value)
        self.ss_target_kind.setCurrentText(st.target_kind)
        self.ss_init.setValue(st.initial_drive)
        self.ss_tol.setValue(st.tolerance_pct)
        self.ss_maxit.setValue(st.max_iterations)
        self.ss_fail.setCurrentText(st.on_failure)
        self.ss_over.setValue(st.overload_pct)
        self.ss_nharm.setValue(st.n_harmonics)
        self._loading = False
        self._on_change()

    def _relabel_trigger_modes(self, suite: str):
        if suite == SUITES[3]:
            labels, tips = PQ_TRIGGER_LABELS + [STEPPED_TRIGGER_LABELS[3]], PQ_TRIGGER_TIPS
        elif suite == SUITES[2]:
            labels, tips = HARMONIC_TRIGGER_LABELS + [STEPPED_TRIGGER_LABELS[3]], HARMONIC_TRIGGER_TIPS
        elif suite == SUITES[1]:
            labels, tips = STEPPED_TRIGGER_LABELS, STEPPED_TRIGGER_TIPS
        else:
            labels, tips = TRIGGER_MODES, RANDOM_TRIGGER_TIPS
        if [self.trg_mode.itemText(i) for i in range(self.trg_mode.count())] == list(labels):
            return
        self.trg_mode.blockSignals(True)
        for i, (lbl, tip) in enumerate(zip(labels, tips)):
            self.trg_mode.setItemText(i, lbl)
            self.trg_mode.setItemData(i, tip, 3)
        self.trg_mode.blockSignals(False)

    def _mode_changed(self, *_):
        if not getattr(self, "_loading", False):
            self._fill_trigger_names()
        self._on_change()

    def set_trigger(self, t: TriggerConfig):
        self._loading = True
        self.trg.setChecked(t.enabled)
        self.trg_mode.setCurrentIndex(TRIGGER_MODES.index(t.mode) if t.mode in TRIGGER_MODES else 0)
        self._fill_trigger_names()
        self.trg_chan.setCurrentText(t.physical)
        self.trg_width.setValue(t.pulse_samples)
        self.trg_level.setValue(t.level_v)
        self.trg_pol.setCurrentText(t.polarity)
        self._loading = False
        self._on_change()

    def get_trigger(self) -> TriggerConfig:
        return TriggerConfig(enabled=self.trg.isChecked(), physical=self.trg_chan.currentText().strip(),
                             mode=TRIGGER_MODES[max(0, self.trg_mode.currentIndex())],
                             pulse_samples=self.trg_width.value(),
                             level_v=self.trg_level.value(), polarity=self.trg_pol.currentText())

    def set_config(self, o: OutputConfig):
        self._loading = True
        self.ao.setCurrentText(o.physical)
        self.vmax.setValue(o.max_voltage)
        self.clock.setCurrentText(o.clock_mode)
        self.exc.setCurrentText(o.excitation)
        self.rms.setValue(o.rms)
        self.fmin.setValue(o.fmin)
        self.fmax.setValue(o.fmax)
        self.window.setCurrentText(o.window)
        self.seed.setValue(o.seed)
        self.fs.setCurrentText(f"{o.fs:g}")
        self.n.setCurrentText(str(o.block_size))
        self.dwell.setValue(o.dwell_blocks)
        self.real.setValue(o.realizations)
        self.decay.setValue(o.decay_time)
        self.fade.setChecked(o.fade)
        self.queue.setValue(o.ao_queue_blocks)
        self.burst.setValue(o.burst_fraction * 100)
        self.custom.set_kind(o.custom_kind)
        self.custom.set_points(o.custom_points)
        self._loading = False
        self._on_change()

    def get_config(self) -> OutputConfig:
        def num(txt, typ, default):
            try:
                return typ(float(txt))
            except ValueError:
                return default
        return OutputConfig(
            physical=self.ao.currentText().strip(), max_voltage=self.vmax.value(),
            clock_mode=self.clock.currentText(), excitation=self.exc.currentText(),
            rms=self.rms.value(), fmin=self.fmin.value(), fmax=self.fmax.value(),
            fs=num(self.fs.currentText(), float, 0.0), block_size=num(self.n.currentText(), int, 0),
            dwell_blocks=self.dwell.value(), realizations=self.real.value(),
            decay_time=self.decay.value(), fade=self.fade.isChecked(), window=self.window.currentText(),
            ao_queue_blocks=self.queue.value(), seed=self.seed.value(),
            burst_fraction=self.burst.value() / 100.0, custom_kind=self.custom.kind.currentText(),
            custom_points=self.custom.points())

    # ------------------------------------------------------------------ preview / summary
    def _new_preview(self):
        self._preview_seed += 1
        self._on_change()

    def _on_change(self, *_):
        if getattr(self, "_loading", False):
            return
        o = self.get_config()
        suite = self.suite.currentText()
        stepped, harmonic, pq = suite == SUITES[1], suite == SUITES[2], suite == SUITES[3]
        sine = stepped or harmonic or pq
        self._relabel_trigger_modes(suite)
        self.g_exc.setVisible(not sine)
        self.g_step.setVisible(stepped)
        self.g_harm.setVisible(harmonic)
        self.g_pq.setVisible(pq)
        for w in (self.dwell, self.real):
            self.tim_form.setRowVisible(w, not sine)
        self.btn_preview.setVisible(not sine)
        if pq:
            self.changed.emit()
            self._pq_preview(o, self.g_pq.get_config())
            return
        if harmonic:
            self.changed.emit()
            self._harmonic_preview(o, self.g_harm.get_config())
            return
        if stepped:
            self.changed.emit()
            self._stepped_preview(o, self.get_stepped())
            return
        self.p_time.setTitle("Drive block (one realization)")
        self.p_spec.setTitle("Drive spectrum")
        self.p_spec.setLabel("left", "Amplitude", "dB re 1 V")
        self.p_spec.setLabel("bottom", "Frequency", "Hz")
        self.c_spec.setSymbol(None)
        band = o.excitation in BAND_TYPES
        self.band_label.setText("Excited band" if band else "Band (markers)")
        self.exc_form.setRowVisible(self.burst, o.excitation == "Burst random")
        self.exc_form.setRowVisible(self.custom, o.excitation == "Custom spectrum")
        self.exc_hint.setText(EXC_TIPS[EXCITATION_TYPES.index(o.excitation)])
        self.changed.emit()
        if o.fs <= 0 or o.block_size < 16 or o.fmax <= o.fmin:
            self.summary.setText("<span style='color:#c0392b'>Enter a valid rate, block size and band.</span>")
            return
        if o.excitation == "Custom spectrum":
            probs = custom_points_problems(o.custom_points, o.fs) if len(o.custom_points) else \
                ["Enter at least two (frequency, amplitude) points."]
            amp = custom_line_amplitudes(o) if not probs else np.zeros(o.block_size // 2 + 1)
            self.custom.show_preview(np.arange(amp.size) * o.df, amp, probs)
            if probs:
                self.summary.setText("<span style='color:#c0392b'>" + "<br>".join(probs) + "</span>")
                return
        try:
            x = generate_block(o, np.random.default_rng(self._preview_seed))
        except Exception as e:
            self.summary.setText(f"<span style='color:#c0392b'>{e}</span>")
            return
        info = block_info(o, x)
        trig_txt = self._trigger_summary(o)
        self._draw_trigger(o)
        t = np.arange(o.block_size) / o.fs
        self.c_time.setData(t, x)
        self.lim_hi.setValue(o.max_voltage)
        self.lim_lo.setValue(-o.max_voltage)
        spec = np.abs(np.fft.rfft(x)) / o.block_size * 2
        f = np.arange(spec.size) * o.df
        db = 20 * np.log10(spec[1:] + 1e-12)
        self.c_spec.setData(f[1:], np.maximum(db, db.max() - 80), connect="all")   # 80 dB display floor

        n_dec = int(math.ceil(o.decay_time * o.fs / o.block_size))
        n_tot = n_dec + o.realizations * o.dwell_blocks + (2 if o.fade else 0) + 1
        dur = n_tot * o.block_size / o.fs
        peak_ok = info.peak <= o.max_voltage
        lines = (f"{info.n_lines} lines, {info.f_lo:g}–{info.f_hi:g} Hz" if info.n_lines
                 else f"broadband 0–{o.fs / 2:g} Hz") + (f" · <i>{info.note}</i>" if info.note else "")
        nyq = "" if o.fmax <= o.fs / 2 else " <span style='color:#c0392b'>(f_max above Nyquist!)</span>"
        self.summary.setText(
            f"<b>Δf = {o.df:.4g} Hz</b> · block length {o.block_size / o.fs:.4g} s · {lines}{nyq}<br>"
            f"Drive: {info.rms:.3g} V RMS{' (during burst)' if o.excitation == 'Burst random' else ''}, "
            f"peak {info.peak:.3g} V, crest factor {info.crest_factor:.2f} "
            + ("<span style='color:#1e8449'>✓ within limit</span>" if peak_ok else
               f"<span style='color:#c0392b'><b>✗ peak exceeds {o.max_voltage:g} V limit</b></span>")
            + f"<br>Averages: {o.realizations} · blocks per average: {o.dwell_blocks} "
              f"(first {o.dwell_blocks - 1} discarded as transient)<br>"
              f"<b>Test duration ≈ {dur:.0f} s</b> ({n_dec} settling + "
              f"{o.realizations * o.dwell_blocks} excitation blocks)" + trig_txt)

    def _stepped_preview(self, o: OutputConfig, st: SteppedConfig) -> None:
        closed = st.control == CONTROL_MODES[0]
        log = st.spacing.startswith("Log")
        self.ss_step_label.setText("Points / decade" if log else "Step")
        self.ss_step.setSuffix("" if log else " Hz")
        rows_closed = [self.ss_target_ch, self.ss_init, self.ss_tol, self.ss_maxit, self.ss_fail]
        for w in rows_closed:
            self.ss_form.setRowVisible(w, closed)
        self.ss_form.setRowVisible(self.ss_drive, not closed)
        self._set_target_row_visible(closed)
        unit = self._ch_units.get(st.target_channel, "")
        self.ss_target_unit.setText(unit)
        trig_txt = self._trigger_summary(o)
        self._draw_trigger(o)
        if o.fs <= 0 or o.block_size < 16:
            self.summary.setText("<span style='color:#c0392b'>Enter a valid rate and block size.</span>")
            return
        freqs, lines, merged = stepped_frequencies(st, o)
        nd = max(1, int(math.ceil(st.dwell_time * o.fs / o.block_size)))
        tb = o.block_size / o.fs
        self.ss_dwell_n.setText(f"= {nd} block{'s' if nd > 1 else ''}")
        if freqs.size == 0:
            self.ss_plan.setText("<span style='color:#c0392b'>No frequencies: check start < stop and the step.</span>")
            self.summary.setText("<span style='color:#c0392b'>No frequencies planned.</span>")
            return
        self.ss_plan.setText(f"{freqs.size} frequencies, {freqs[0]:g}–{freqs[-1]:g} Hz, on FFT lines of "
                             f"Δf = {o.df:.4g} Hz" + (f" ({merged} requested points merged — use a larger block "
                                                      "for finer steps)" if merged else ""))
        q = max(2, o.ao_queue_blocks)
        n_dec = int(math.ceil(o.decay_time * o.fs / o.block_size))
        if closed:
            per_it = st.settle_blocks + 1 + q
            lo = freqs.size * (2 * per_it + nd + st.settle_blocks) + n_dec
            hi = freqs.size * (5 * per_it + nd + st.settle_blocks) + n_dec
            dur = f"≈ {lo * tb / 60:.1f}–{hi * tb / 60:.1f} min (2–5 iterations per frequency)"
            amp = st.initial_drive
        else:
            dur = f"≈ {(freqs.size * (st.settle_blocks + nd) + n_dec) * tb / 60:.1f} min"
            amp = st.drive_amplitude
        x = amp * np.sin(2 * np.pi * lines[0] * np.arange(o.block_size) / o.block_size)
        self.c_time.setData(np.arange(o.block_size) / o.fs, x)
        self.lim_hi.setValue(o.max_voltage)
        self.lim_lo.setValue(-o.max_voltage)
        self.p_time.setTitle(f"Drive block at the first frequency ({freqs[0]:g} Hz, {amp:g} V peak)")
        self.c_spec.setData(freqs, np.arange(1, freqs.size + 1), connect="all")
        self.c_spec.setSymbol("o")
        self.c_spec.setSymbolSize(5)
        self.p_spec.setTitle("Planned frequencies")
        self.p_spec.setLabel("bottom", "Frequency", "Hz")
        self.p_spec.setLabel("left", "Step #")
        eq = (f" (= {st.target_value / np.sqrt(2):.4g} RMS)" if st.is_amplitude
              else f" (= {st.target_value * np.sqrt(2):.4g} amplitude)")
        ctl = (f"Secant control of <b>{st.target_channel}</b> to <b>{st.target_value:g} {unit} {st.target_label}</b>{eq} "
               f"± {st.tolerance_pct:g} %, ≤ {st.max_iterations} iterations; drive ≤ {o.max_voltage:g} V, "
               f"each update ½…2× the current drive; on failure: {st.on_failure.lower()}"
               if closed else f"No control: constant {st.drive_amplitude:g} V peak")
        amp_ok = (st.initial_drive if closed else st.drive_amplitude) <= o.max_voltage
        self.summary.setText(
            f"<b>{freqs.size} frequencies</b> {freqs[0]:g}–{freqs[-1]:g} Hz · block {tb:.4g} s · "
            f"settle {st.settle_blocks} + dwell {nd} block(s) per frequency<br>{ctl}"
            + ("" if amp_ok else f" <span style='color:#c0392b'><b>✗ drive above the {o.max_voltage:g} V limit</b></span>")
            + f"<br><b>Test duration {dur}</b>" + trig_txt
            + (f"<br><span style='color:#d35400'>⚠ <b>Harmonics:</b> {nyq}</span>" if (nyq := harmonic_nyquist_note(
                st.n_harmonics, float(freqs[0]), float(freqs[-1]), o.fs, [float(r) for r in COMMON_RATES])) else ""))

    def _rate_stale(self):
        if self.rate_info.isVisible() and "stale" not in self.rate_info.text():
            self.rate_info.setText(self.rate_info.text() + "<br><span style='color:gray'>(rate changed since the "
                                   "check — the result above may be stale)</span>")

    def show_rate_check(self, res: dict, simulated: bool = False):
        """Show the result of check_rates(); rates are links that set the sampling rate."""
        self.rate_info.setVisible(True)
        if simulated:
            self.rate_info.setText("<span style='color:gray'>Simulation mode: the simulator uses the rate as "
                                   "entered. Turn simulation off (Hardware menu) to check the real modules.</span>")
            return
        if res.get("error"):
            self.rate_info.setText(f"<span style='color:#c0392b'><b>DAQmx refused this setup</b> at "
                                   f"{res['requested']:g} Hz: {res['error']}</span>"
                                   + self._exact_links(res))
            return
        req, ai, ao = res["requested"], res["ai"], res["ao"]
        same = abs(ai - req) <= 1e-6 * req
        ao_ok = ao is not None and abs(ao - ai) <= 1e-9 * ai
        txt = (f"<b>Hardware:</b> input <b>{ai:.8g} Hz</b>"
               + (" <span style='color:#1e8449'>✓ as requested</span>" if same else
                  f" <span style='color:#d35400'>(requested {req:g} Hz — coerced)</span>")
               + f" · output {ao:.8g} Hz "
               + ("<span style='color:#1e8449'>✓ matches</span>" if ao_ok else
                  "<span style='color:#c0392b'>✗ differs from the input rate (leakage)</span>"))
        if not same:
            txt += f" · <a href='{ai:.10g}'>use {ai:.8g} Hz</a>"
        self.rate_info.setText(txt + self._exact_links(res))

    @staticmethod
    def _exact_links(res) -> str:
        ex = res.get("exact") or []
        if not ex:
            return ""
        return ("<br>Rates every module here supports exactly (click to use): "
                + ", ".join(f"<a href='{r:.10g}'>{r:.6g}</a>" for r in ex) + " Hz")

    def get_pq(self) -> PQConfig:
        return self.g_pq.get_config()

    def set_pq(self, p: PQConfig):
        self.g_pq.set_config(p)

    def _pq_preview(self, o: OutputConfig, p: PQConfig) -> None:
        trig_txt = self._trigger_summary(o)
        self._draw_trigger(o)
        lv = p.levels()
        if o.fs <= 0 or o.block_size < 16 or lv.size == 0:
            self.summary.setText("<span style='color:#c0392b'>Enter a valid rate, block size and drive levels.</span>")
            return
        tb = o.block_size / o.fs
        t = np.arange(o.block_size) / o.fs
        self.c_time.setData(t, lv[0] * np.sin(2 * np.pi * p.f_init * t))
        self.lim_hi.setValue(o.max_voltage)
        self.lim_lo.setValue(-o.max_voltage)
        self.p_time.setTitle(f"Drive block at the first level ({lv[0]:.4g} V peak at {p.f_init:g} Hz)")
        self.c_spec.setData(np.arange(1, lv.size + 1), lv, connect="all")
        self.c_spec.setSymbol("o")
        self.c_spec.setSymbolSize(6)
        self.p_spec.setTitle("Planned drive levels (run order)")
        self.p_spec.setLabel("left", "Drive", "V peak")
        self.p_spec.setLabel("bottom", "Level #")
        q = max(2, o.ao_queue_blocks)
        n_dec = int(np.ceil(o.decay_time * o.fs / o.block_size))
        per_it = p.settle_blocks + 1 + q
        lo = n_dec + lv.size * (3 * per_it + p.dwell_blocks)
        hi = n_dec + lv.size * (8 * per_it + p.dwell_blocks)
        ref = self.g_pq._ref_name or "(none)"
        if p.quadrature:
            ctl = (f"Secant on frequency to phase quadrature: <b>{p.response_channel}</b> re <b>{ref}</b> at "
                   f"<b>{p.target_phase:+g}°</b> ± {p.phase_tol:g}°")
        else:
            ctl = (f"Reference-free: frequency of maximum <b>|{p.response_channel}" + (f" / {ref}|</b>" if self.g_pq._ref_name
                   else "|</b> per drive volt") + f", located to ± {p.freq_tol:g} Hz")
        nyq = harmonic_nyquist_note(p.n_harmonics, p.f_min, p.f_max, o.fs, [float(r) for r in COMMON_RATES])
        over = lv.max() > o.max_voltage
        self.p_spec.getAxis("bottom").enableAutoSIPrefix(False)
        self.summary.setText(
            f"<b>Phase quadrature</b>: {lv.size} drive levels {lv.min():.4g}–{lv.max():.4g} V ({p.order.lower()}), "
            f"start at {p.f_init:g} Hz, search {p.f_min:g}–{p.f_max:g} Hz · block {tb:.4g} s<br>{ctl}; "
            f"≤ {p.max_iterations} iterations per level, steps ≤ {p.max_step:g} Hz; on failure: {p.on_failure.lower()}"
            + (f" <span style='color:#c0392b'><b>✗ highest level above the {o.max_voltage:g} V limit</b></span>" if over else "")
            + f"<br>Per level: settle {p.settle_blocks} + 1 measured block per iteration, then {p.dwell_blocks} "
              f"converged block(s) recorded. Frequency resolution is not limited to FFT lines (least-squares sine fit)."
            + f"<br><b>Test duration ≈ {lo * tb / 60:.1f}–{hi * tb / 60:.1f} min</b> (3–8 iterations per level)"
            + (f"<br><span style='color:#d35400'>⚠ <b>Harmonics:</b> {nyq}</span>" if nyq else "") + trig_txt)

    def get_harmonic(self) -> HarmonicConfig:
        return self.g_harm.get_config()

    def set_harmonic(self, h: HarmonicConfig):
        self.g_harm.set_config(h)

    def _harmonic_preview(self, o: OutputConfig, h: HarmonicConfig) -> None:
        trig_txt = self._trigger_summary(o)
        self._draw_trigger(o)
        k = harmonic_line(h, o)
        tb = o.block_size / o.fs if o.fs > 0 else 0
        f0a = k * o.fs / o.block_size if k else 0
        self.g_harm.update_info(f0a, k, tb)
        cfg_err = []
        rows = h.rows()
        if not k or not rows:
            self.summary.setText("<span style='color:#c0392b'>Enter a valid rate, block size, base frequency "
                                 "and at least one harmonic.</span>")
            return
        orders, vals = h.complex_values()
        closed = h.closed_loop
        if closed:                       # first iteration's drive (as the engine builds it)
            big = float(np.max(np.abs(vals))) or 1.0
            V = h.initial_drive * np.maximum(np.abs(vals) / big, 0.05) * np.exp(1j * np.angle(vals))
        else:
            V = vals
        x = harmonic_waveform(orders, V, k, o.block_size)
        t = np.arange(o.block_size) / o.fs
        self.c_time.setData(t, x)
        self.lim_hi.setValue(o.max_voltage)
        self.lim_lo.setValue(-o.max_voltage)
        self.p_time.setTitle("Drive block" + (" at the first control iteration" if closed else "")
                             + f" ({k} period{'s' if k != 1 else ''} of f0 = {f0a:.4g} Hz)")
        # spectrum: stems at the harmonic frequencies
        fx, fy = [], []
        for n_, v in zip(orders, V):
            fx += [n_ * f0a, n_ * f0a]
            fy += [0.0, abs(v)]
        self.c_spec.setData(fx, fy, connect="pairs")
        self.c_spec.setSymbol("o")
        self.c_spec.setSymbolSize(6)
        self.p_spec.setTitle("Drive harmonics" + (" (first iteration)" if closed else ""))
        self.p_spec.setLabel("bottom", "Frequency", "Hz")
        self.p_spec.setLabel("left", "Amplitude", "V peak")
        nyq_bad = [n_ for n_ in orders if n_ * k >= o.block_size // 2]
        if nyq_bad:
            cfg_err.append(f"order(s) {', '.join(map(str, nyq_bad))} above Nyquist")
        pk = float(np.max(np.abs(x)))
        n_dec = int(np.ceil(o.decay_time * o.fs / o.block_size))
        q = max(2, o.ao_queue_blocks)
        if closed:
            per_it = h.settle_blocks + 1 + q
            lo, hi = n_dec + 2 * per_it + h.test_blocks, n_dec + 8 * per_it + h.test_blocks
            dur = f"≈ {lo * tb:.0f}–{hi * tb:.0f} s (2–8 iterations)"
            unit = self.g_harm._units.get(h.target_channel, "")
            spec = ", ".join(f"h{n_}: {abs(v):.4g}∠{np.degrees(np.angle(v)):.0f}°" for n_, v in zip(orders, vals))
            ctl = (f"Broyden control of <b>{h.target_channel}</b> to [{spec}] {unit} ± {h.tolerance_pct:g} %, "
                   f"≤ {h.max_iterations} iterations; drive ≤ {o.max_voltage:g} V; on failure: {h.on_failure.lower()}")
        else:
            dur = f"≈ {(n_dec + h.settle_blocks + h.test_blocks + 2) * tb:.0f} s"
            ctl = (f"No control: drive peak <b>{pk:.4g} V</b> "
                   + ("<span style='color:#1e8449'>✓ within limit</span>" if pk <= o.max_voltage else
                      f"<span style='color:#c0392b'><b>✗ exceeds the {o.max_voltage:g} V limit</b></span>"))
        self.summary.setText(
            f"<b>Harmonic</b>: f0 = {f0a:.6g} Hz (line {k}), orders {', '.join(map(str, orders))} · block {tb:.4g} s<br>"
            f"{ctl}<br>Then <b>{h.test_blocks} converged test block(s)</b> ({h.test_blocks * tb:.4g} s), drive off."
            + (f"<br><span style='color:#c0392b'>{'; '.join(cfg_err)}</span>" if cfg_err else "")
            + f"<br><b>Test duration {dur}</b>" + trig_txt)

    def _set_target_row_visible(self, on: bool):
        """The target row's field is a layout (spin box + unit), so find its row index."""
        for i in range(self.ss_form.rowCount()):
            item = self.ss_form.itemAt(i, QFormLayout.FieldRole)
            if item is not None and item.layout() is not None and item.layout().indexOf(self.ss_target) >= 0:
                self.ss_form.setRowVisible(i, on)

    def _draw_trigger(self, o: OutputConfig) -> None:
        """Plot one trigger event sample by sample (as written to the output, before any module
        filtering), with the trigger sample at x = 0."""
        t = self.get_trigger()
        self.p_trig.setVisible(t.enabled)
        if not t.enabled or o.fs <= 0:
            return
        dt_us = 1e6 / o.fs
        low_active = t.polarity.startswith("Active low")
        if t.frame_sync:
            # schematic: one short pulse at every sample clock edge (width set by hardware)
            n, w = 8, 0.2
            edges, ys = [-1.0], []
            for k in range(n):
                edges += [k, k + w]
                ys += [0.0, 1.0]
            edges.append(n)
            ys.append(0.0)
            y = np.array(ys)
            y = 1.0 - y if low_active else y
            self.c_trig.setData(np.array(edges), y)
            self.c_trig_pts.setData([], [])
            self.p_trig.setTitle(f"Frame clock: one pulse per sample (every {dt_us:,.1f} µs), "
                                 "pulse width set by hardware (schematic)")
            self.p_trig.setLabel("left", "Logic level")
            self.p_trig.setLabel("bottom", "Sample index from start of acquisition")
            return
        hi = t.level_v if t.kind == "AO" else 1.0
        active, idle = (0.0, hi) if low_active else (hi, 0.0)
        w = max(1, min(t.pulse_samples, max(o.block_size - 1, 1)))
        pre, post = max(4, w // 2), max(8, w)
        k = np.arange(-pre, w + post)                     # sample indices relative to the trigger sample
        y = np.where((k >= 0) & (k < w), active, idle)
        self.c_trig.setData(np.append(k - 0.5, k[-1] + 0.5), y)   # each sample held for one period
        self.c_trig_pts.setData(k, y)
        unit = "V" if t.kind == "AO" else "logic"
        self.p_trig.setLabel("left", "Trigger", unit)
        self.p_trig.setLabel("bottom", f"Samples from trigger sample (1 sample = {dt_us:,.1f} µs)")
        self.p_trig.setTitle(f"Trigger pulse: {w} samples = {w * dt_us:,.1f} µs, "
                             f"{'falling' if low_active else 'rising'} edge at sample 0"
                             + (" (AO module filtering not shown)" if t.kind == "AO" else ""))
        self.p_trig.setYRange(-0.1 * hi, 1.15 * hi)

    def _trigger_summary(self, o: OutputConfig) -> str:
        t = self.get_trigger()
        kind = output_kind(t.physical)
        self.trg_kind.setText({"AO": "AO", "DO": "DO line", "PFI": "PFI"}.get(
            kind, "<span style='color:#c0392b'>unknown type</span>"))
        frame = t.frame_sync
        self.trg_level.setEnabled(kind == "AO" and not frame)
        self.trg_width.setEnabled(not frame)
        if o.fs > 0:
            self.trg_width_t.setText("(fixed by hardware)" if frame else f"= {t.pulse_samples / o.fs * 1e6:,.0f} µs")
        if not t.enabled:
            return ""
        if frame:
            ok = kind == "PFI"
            return (f"<br><b>Frame sync:</b> AI sample clock → {t.physical}: one pulse per sample, "
                    f"{o.fs:g} pulses/s from sample 0 to the end of the run. Camera in external-sync mode "
                    f"⇒ <b>frame f = sample f</b>, no drift."
                    + ("" if ok else " <span style='color:#c0392b'><b>Choose a PFI terminal.</b></span>"))
        if self.suite.currentText() == SUITES[3]:
            what = {0: "once, at sample 0 (start of the acquisition)",
                    1: "at the start of the converged block of every level",
                    2: "at the start of every drive level"}[TRIGGER_MODES.index(t.mode)]
            return f"<br><b>Trigger:</b> pulse on {t.physical} {what}; sample indices go to the trigger log."
        if self.suite.currentText() == SUITES[2]:
            what = {0: "once, at sample 0 (start of the acquisition)",
                    1: "at the start of the converged test blocks",
                    2: "at the start of every control iteration"}[TRIGGER_MODES.index(t.mode)]
            return f"<br><b>Trigger:</b> pulse on {t.physical} {what}; sample indices go to the trigger log."
        if self.suite.currentText() == SUITES[1]:
            what = {0: "once, at sample 0 (start of the acquisition)",
                    1: "at the start of the analysed dwell at every frequency",
                    2: "at the start of every frequency step"}[TRIGGER_MODES.index(t.mode)]
            return f"<br><b>Trigger:</b> pulse on {t.physical} {what}; sample indices go to the trigger log."
        n_dec = int(math.ceil(o.decay_time * o.fs / o.block_size)) + (1 if o.fade else 0)
        if t.mode.startswith("Once"):
            first, n = 0, 1
        elif t.mode.startswith("Start of the converged"):
            first, n = (n_dec + o.dwell_blocks - 1) * o.block_size, o.realizations
        else:
            first, n = n_dec * o.block_size, o.realizations
        every = o.dwell_blocks * o.block_size / o.fs
        return (f"<br><b>Trigger:</b> {n} pulse{'s' if n > 1 else ''} on {t.physical}, first at sample "
                f"{first} (t = {first / o.fs:.3f} s)" + (f", then every {every:.3f} s" if n > 1 else "")
                + f". Camera at <b>{o.fs:g} fps</b> ⇒ 1 frame per sample.")
