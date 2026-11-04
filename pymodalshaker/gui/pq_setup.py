# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Output Setup: settings of the Phase Quadrature suite (drive levels, frequency search, control)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QSpinBox)

from ..config import (PQ_CONTROL_MODES, PQ_FAIL_ACTIONS, PQ_LEVEL_MODES, PQ_ORDERS, PQ_PHASE_PRESETS, PQConfig)


def _d(lo, hi, dec, suffix="", step=None):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    if suffix:
        w.setSuffix(" " + suffix)
    if step:
        w.setSingleStep(step)
    return w


def _i(lo, hi, suffix=""):
    w = QSpinBox()
    w.setRange(lo, hi)
    if suffix:
        w.setSuffix(" " + suffix)
    return w


class PQSetup(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Phase quadrature (amplitude-dependent resonance)", parent)
        self.v_min = _d(0.0001, 10, 4, "V", 0.01)
        self.v_max = _d(0.0001, 10, 4, "V", 0.05)
        self.lv_mode = QComboBox()
        self.lv_mode.addItems(PQ_LEVEL_MODES)
        self.n_lv = _i(1, 1000)
        self.v_step = _d(0.0001, 10, 4, "V", 0.01)
        self.order = QComboBox()
        self.order.addItems(PQ_ORDERS)
        self.order.setToolTip("Up, then down: repeats the levels in reverse to show hysteresis / jumps")
        self.lv_info = QLabel()
        self.lv_info.setWordWrap(True)
        self.lv_info.setStyleSheet("color: gray")
        self.f_init = _d(0.01, 1e5, 3, "Hz", 1)
        self.f_init.setToolTip("Initial guess of the resonance frequency. Each further level starts from the "
                               "previous level's resonance.")
        self.f_min = _d(0.01, 1e5, 2, "Hz", 1)
        self.f_max = _d(0.01, 1e5, 2, "Hz", 1)
        for w in (self.f_min, self.f_max):
            w.setToolTip("Search range: control fails if the resonance would lie outside it")
        self.control = QComboBox()
        self.control.addItems(PQ_CONTROL_MODES)
        self.control.setToolTip("Phase quadrature: secant iteration on the phase of response re reference.\n"
                                "Reference-free: frequency of maximum |H1| = |response / reference| (or |response| "
                                "per drive volt if no reference is marked), by a bracketing / parabolic search.")
        self.resp = QComboBox()
        self.resp.setToolTip("Channel whose phase (or amplitude) is controlled")
        self.ref_label = QLabel()
        self.ref_label.setToolTip("Set on the Channel Setup tab (Reference column)")
        self.preset = QComboBox()
        self.preset.addItems([p[0] for p in PQ_PHASE_PRESETS])
        self.preset.setToolTip("Phase of the response re the reference at phase resonance, by response type")
        self.target = _d(-180, 180, 2, "°", 1)
        self.phase_tol = _d(0.01, 45, 2, "°", 0.1)
        self.freq_tol = _d(0.0001, 100, 4, "Hz", 0.01)
        self.freq_tol.setToolTip("Reference-free: converged when the peak is located to within this")
        self.step0 = _d(0.0001, 1e4, 3, "Hz", 0.1)
        self.step0.setToolTip("First frequency step of every level (scaled down when close to quadrature)")
        self.max_step = _d(0.0001, 1e4, 3, "Hz", 1)
        self.max_step.setToolTip("Largest frequency change per iteration")
        self.maxit = _i(2, 500)
        self.fail = QComboBox()
        self.fail.addItems(PQ_FAIL_ACTIONS)
        self.over = _d(50, 100, 1, "% of range", 1)
        self.settle = _i(0, 1000, "blocks")
        self.settle.setToolTip("Blocks discarded after every change of frequency or level (transient decay)")
        self.dwell = _i(1, 10000, "blocks")
        self.dwell.setToolTip("Blocks recorded at the converged frequency of every level: averaged for the "
                              "result and saved (TDMS)")
        self.nh = _i(2, 50)
        self.nh.setToolTip("Harmonic orders 1…n analysed and saved (orders above Nyquist are left out)")
        for cb in (self.lv_mode, self.order, self.control, self.resp, self.preset, self.fail):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(12)

        f = QFormLayout(self)
        lv = QHBoxLayout()
        lv.addWidget(self.v_min)
        lv.addWidget(QLabel("to"))
        lv.addWidget(self.v_max)
        f.addRow("Drive levels (peak)", lv)
        lm = QHBoxLayout()
        lm.addWidget(self.lv_mode, 1)
        lm.addWidget(self.n_lv)
        lm.addWidget(self.v_step)
        f.addRow("Spacing", lm)
        f.addRow("Order", self.order)
        f.addRow(self.lv_info)
        f.addRow("Initial frequency", self.f_init)
        fr = QHBoxLayout()
        fr.addWidget(self.f_min)
        fr.addWidget(QLabel("to"))
        fr.addWidget(self.f_max)
        f.addRow("Search range", fr)
        f.addRow("Control", self.control)
        f.addRow("Response channel", self.resp)
        f.addRow("Reference channel", self.ref_label)
        tp = QHBoxLayout()
        tp.addWidget(self.preset, 1)
        tp.addWidget(self.target)
        self.target_row = tp
        f.addRow("Quadrature phase", tp)
        f.addRow("Phase tolerance  ±", self.phase_tol)
        f.addRow("Frequency tolerance  ±", self.freq_tol)
        f.addRow("First step", self.step0)
        f.addRow("Max step", self.max_step)
        f.addRow("Max iterations", self.maxit)
        f.addRow("If control fails", self.fail)
        f.addRow("Input overload at", self.over)
        f.addRow("Settle after change", self.settle)
        f.addRow("Converged blocks", self.dwell)
        f.addRow("Harmonics: orders 1 …", self.nh)
        self.form = f
        self._ref_name = ""
        self._loading = False
        for w in (self.v_min, self.v_max, self.v_step, self.f_init, self.f_min, self.f_max, self.target,
                  self.phase_tol, self.freq_tol, self.step0, self.max_step, self.over):
            w.valueChanged.connect(self._emit)
        for w in (self.n_lv, self.maxit, self.settle, self.dwell, self.nh):
            w.valueChanged.connect(self._emit)
        for w in (self.lv_mode, self.order, self.control, self.resp, self.fail):
            w.currentTextChanged.connect(self._emit)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.target.valueChanged.connect(self._target_edited)
        self.set_config(PQConfig())

    # ------------------------------------------------------------------ channels
    def set_channels(self, chans, ref_name: str):
        """chans: [(name, units)] of the enabled channels; ref_name: the reference channel ('' if none)."""
        self._ref_name = ref_name
        cur = self.resp.currentText()
        self.resp.blockSignals(True)
        self.resp.clear()
        self.resp.addItems([n for n, _ in chans if n != ref_name])
        if cur:
            self.resp.setCurrentText(cur)
        self.resp.blockSignals(False)
        self._emit()

    def _preset_changed(self, i):
        v = PQ_PHASE_PRESETS[i][1]
        if v is not None:
            self.target.blockSignals(True)
            self.target.setValue(v)
            self.target.blockSignals(False)
        self._emit()

    def _target_edited(self, v):
        match = next((i for i, p in enumerate(PQ_PHASE_PRESETS) if p[1] is not None and abs(p[1] - v) < 1e-9), None)
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(match if match is not None else len(PQ_PHASE_PRESETS) - 1)
        self.preset.blockSignals(False)
        self._emit()

    # ------------------------------------------------------------------ config
    def set_config(self, p: PQConfig):
        self._loading = True
        self.v_min.setValue(p.v_min)
        self.v_max.setValue(p.v_max)
        self.lv_mode.setCurrentText(p.level_mode)
        self.n_lv.setValue(p.n_levels)
        self.v_step.setValue(p.v_step)
        self.order.setCurrentText(p.order)
        self.f_init.setValue(p.f_init)
        self.f_min.setValue(p.f_min)
        self.f_max.setValue(p.f_max)
        self.control.setCurrentText(p.control)
        if self.resp.findText(p.response_channel) < 0 and p.response_channel:
            self.resp.addItem(p.response_channel)
        self.resp.setCurrentText(p.response_channel)
        self.target.setValue(p.target_phase)
        self.preset.setCurrentText(p.phase_preset)
        self.phase_tol.setValue(p.phase_tol)
        self.freq_tol.setValue(p.freq_tol)
        self.step0.setValue(p.step0)
        self.max_step.setValue(p.max_step)
        self.maxit.setValue(p.max_iterations)
        self.fail.setCurrentText(p.on_failure)
        self.over.setValue(p.overload_pct)
        self.settle.setValue(p.settle_blocks)
        self.dwell.setValue(p.dwell_blocks)
        self.nh.setValue(p.n_harmonics)
        self._loading = False
        self._emit()

    def get_config(self) -> PQConfig:
        return PQConfig(v_min=self.v_min.value(), v_max=self.v_max.value(), level_mode=self.lv_mode.currentText(),
                        n_levels=self.n_lv.value(), v_step=self.v_step.value(), order=self.order.currentText(),
                        f_init=self.f_init.value(), f_min=self.f_min.value(), f_max=self.f_max.value(),
                        control=self.control.currentText(), response_channel=self.resp.currentText(),
                        target_phase=self.target.value(), phase_preset=self.preset.currentText(),
                        phase_tol=self.phase_tol.value(), freq_tol=self.freq_tol.value(), step0=self.step0.value(),
                        max_step=self.max_step.value(), max_iterations=self.maxit.value(),
                        on_failure=self.fail.currentText(), overload_pct=self.over.value(),
                        settle_blocks=self.settle.value(), dwell_blocks=self.dwell.value(), n_harmonics=self.nh.value())

    def _set_row_visible(self, layout_or_widget, on: bool):
        for i in range(self.form.rowCount()):
            it = self.form.itemAt(i, QFormLayout.FieldRole)
            if it is None:
                continue
            if it.widget() is layout_or_widget or it.layout() is layout_or_widget:
                self.form.setRowVisible(i, on)

    def _emit(self, *_):
        if self._loading:
            return
        step_mode = self.lv_mode.currentIndex() == 1
        self.n_lv.setVisible(not step_mode)
        self.v_step.setVisible(step_mode)
        quad = self.control.currentIndex() == 0
        self._set_row_visible(self.target_row, quad)
        self.form.setRowVisible(self.phase_tol, quad)
        self.form.setRowVisible(self.freq_tol, not quad)
        ref = self._ref_name
        self.ref_label.setText(f"<b>{ref}</b>  (Channel Setup)" if ref else
                               ("<span style='color:#c0392b'><b>none</b> — tick Reference on Channel Setup</span>"
                                if quad else "<i>none</i> — |response| per drive volt is maximised"))
        lv = self.get_config().levels()
        self.lv_info.setText(f"{lv.size} level(s): " + ", ".join(f"{v:.4g}" for v in lv[:12]) + (" …" if lv.size > 12 else "")
                             + " V" if lv.size else "<span style='color:#c0392b'>No levels: check min/max and spacing.</span>")
        self.changed.emit()
