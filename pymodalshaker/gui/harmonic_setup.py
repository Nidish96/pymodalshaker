# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Output Setup: settings of the harmonic suite (base frequency, harmonic table, control, test length)."""

from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

from ..config import HARM_CONTROL_MODES, HARM_FAIL_ACTIONS, HarmonicConfig


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


class HarmonicSetup(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__("Harmonic excitation", parent)
        self.f0 = _d(0.01, 1e5, 3, "Hz", 1)
        self.f0.setToolTip("Base frequency f0 (Ω = 2π·f0). It is moved to the nearest FFT line k·fs/N so that "
                           "every block holds whole periods of every harmonic.")
        self.f0_info = QLabel()
        self.f0_info.setStyleSheet("color: gray")
        self.f0_info.setWordWrap(True)
        self.control = QComboBox()
        self.control.addItems(HARM_CONTROL_MODES)
        self.control.setToolTip("No control: the table is the drive voltage.\n"
                                "Control: the table is the target for the control channel's harmonics; the drive "
                                "harmonics are adjusted by Broyden's quasi-Newton method.")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Order n", "Amplitude", "Phase (°)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(130)
        self.table.setToolTip("v(t) = Σ a·cos(n·Ω·t + φ). Amplitudes are peak values: volts without control, "
                              "the control channel's units with control. An amplitude of 0 with control "
                              "means: suppress this harmonic.")
        self.table.itemChanged.connect(lambda *_: self._emit())
        add, rem = QPushButton("+ harmonic"), QPushButton("− remove")
        add.clicked.connect(self._add_row)
        rem.clicked.connect(self._remove_rows)
        self.formula = QLabel()
        self.formula.setWordWrap(True)
        self.formula.setStyleSheet("color: gray")
        self.channel = QComboBox()
        self.channel.setToolTip("Channel whose harmonics are controlled to the table values")
        self.init = _d(0.0001, 10, 4, "V peak", 0.01)
        self.init.setToolTip("Drive of the largest harmonic at the first iteration (other harmonics in proportion "
                             "to their targets, at least 5 % so that every harmonic is probed)")
        self.tol = _d(0.1, 50, 1, "%", 0.5)
        self.tol.setToolTip("Converged when every harmonic's complex error |Y − T| is within this % of its "
                            "target (targets below 10 % of the largest: of the largest target). 3 % ≈ 3 % in "
                            "amplitude and 1.7° in phase.")
        self.maxit = _i(2, 200)
        self.maxit.setToolTip("Iterations allowed; there are always at least two (one refinement)")
        self.fail = QComboBox()
        self.fail.addItems(HARM_FAIL_ACTIONS)
        self.over = _d(50, 100, 1, "% of range", 1)
        self.settle = _i(0, 100, "blocks")
        self.settle.setToolTip("Blocks discarded after every change of the drive")
        self.test = _i(1, 10_000_000, "blocks")
        self.test.setToolTip("Converged blocks run (and analysed) before the excitation is switched off")
        self.test_t = QLabel()
        self.nh = _i(2, 50)
        self.nh.setToolTip("Harmonic orders 1…n used by the displays (Block harmonics, strobed-view Fourier fit). "
                           "At least the highest order in the table is always used.")
        for cb in (self.control, self.fail, self.channel):
            cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(12)

        f = QFormLayout(self)
        fr = QVBoxLayout()
        fr.addWidget(self.f0)
        fr.addWidget(self.f0_info)
        f.addRow("Base frequency f0", fr)
        f.addRow("Control", self.control)
        tl = QVBoxLayout()
        tl.addWidget(self.table)
        bl = QHBoxLayout()
        bl.addWidget(add)
        bl.addWidget(rem)
        bl.addStretch()
        tl.addLayout(bl)
        tl.addWidget(self.formula)
        self.table_label = QLabel("Harmonics")
        f.addRow(self.table_label, tl)
        f.addRow("Control channel", self.channel)
        f.addRow("Initial drive", self.init)
        f.addRow("Tolerance  ±", self.tol)
        f.addRow("Max iterations", self.maxit)
        f.addRow("If control fails", self.fail)
        f.addRow("Input overload at", self.over)
        f.addRow("Settle after change", self.settle)
        tb = QHBoxLayout()
        tb.addWidget(self.test, 1)
        tb.addWidget(self.test_t)
        f.addRow("Converged test", tb)
        f.addRow("Harmonics shown: 1 …", self.nh)
        self.form = f
        self._units: Dict[str, str] = {}
        self._loading = False
        for w in (self.f0, self.init, self.tol, self.over):
            w.valueChanged.connect(self._emit)
        for w in (self.maxit, self.settle, self.test, self.nh):
            w.valueChanged.connect(self._emit)
        for w in (self.control, self.fail, self.channel):
            w.currentTextChanged.connect(self._emit)
        self.set_config(HarmonicConfig())

    # ------------------------------------------------------------------ table
    def _add_row(self, order=None, amp=0.0, phase=0.0):
        rows = self.rows()
        if order is None or order is False:
            order = max([r[0] for r in rows] + [0]) + 1
        i = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(i)
        def fmt(v, as_int=False):
            try:
                return f"{int(round(float(v)))}" if as_int else f"{float(v):g}"
            except (TypeError, ValueError):
                return str(v)
        for c, v in enumerate((fmt(order, True), fmt(amp), fmt(phase))):
            it = QTableWidgetItem(v)
            it.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, c, it)
        self.table.blockSignals(False)
        self._emit()

    def _remove_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True) or [self.table.rowCount() - 1]
        for r in rows:
            if r >= 0:
                self.table.removeRow(r)
        self._emit()

    def raw_rows(self) -> List[List[float]]:
        """Table content; unparsable cells are kept as text so validation can report them."""
        out = []
        for i in range(self.table.rowCount()):
            row = []
            for c in range(3):
                it = self.table.item(i, c)
                txt = it.text().strip() if it else ""
                try:
                    row.append(float(txt))
                except ValueError:
                    row.append(txt)
            out.append(row)
        return out

    def rows(self):
        return HarmonicConfig(harmonics=self.raw_rows()).rows()

    # ------------------------------------------------------------------ config
    def set_channels(self, chans):
        self._units = dict(chans)
        cur = self.channel.currentText()
        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.addItems([n for n, _ in chans])
        if cur:
            self.channel.setCurrentText(cur)
        self.channel.blockSignals(False)
        self._emit()

    def set_config(self, h: HarmonicConfig):
        self._loading = True
        self.f0.setValue(h.f0)
        self.control.setCurrentText(h.control)
        self.table.setRowCount(0)
        for r in h.harmonics:
            self._add_row(*(list(r) + [0, 0, 0])[:3])
        if self.channel.findText(h.target_channel) < 0 and h.target_channel:
            self.channel.addItem(h.target_channel)
        self.channel.setCurrentText(h.target_channel)
        self.init.setValue(h.initial_drive)
        self.tol.setValue(h.tolerance_pct)
        self.maxit.setValue(h.max_iterations)
        self.fail.setCurrentText(h.on_failure)
        self.over.setValue(h.overload_pct)
        self.settle.setValue(h.settle_blocks)
        self.test.setValue(h.test_blocks)
        self.nh.setValue(h.n_harmonics)
        self._loading = False
        self._emit()

    def get_config(self) -> HarmonicConfig:
        return HarmonicConfig(f0=self.f0.value(), harmonics=self.raw_rows(), control=self.control.currentText(),
                              target_channel=self.channel.currentText(), initial_drive=self.init.value(),
                              tolerance_pct=self.tol.value(), max_iterations=self.maxit.value(),
                              on_failure=self.fail.currentText(), overload_pct=self.over.value(),
                              settle_blocks=self.settle.value(), test_blocks=self.test.value(),
                              n_harmonics=self.nh.value())

    # ------------------------------------------------------------------ display
    def update_info(self, f0_actual: float, k: int, block_s: float):
        """Called by Output Setup with the snapped base frequency and the block length."""
        self.f0_info.setText(f"→ {f0_actual:.6g} Hz on FFT line {k} ({k} period{'s' if k != 1 else ''} per block)"
                             if k else "Enter a valid sampling rate and block size.")
        self.test_t.setText(f"= {self.test.value() * block_s:.4g} s" if block_s > 0 else "")

    def _emit(self, *_):
        if self._loading:
            return
        closed = self.control.currentText() == HARM_CONTROL_MODES[0]
        for w in (self.channel, self.init, self.tol, self.maxit, self.fail):
            self.form.setRowVisible(w, closed)
        unit = self._units.get(self.channel.currentText(), "") if closed else "V"
        self.table.horizontalHeaderItem(1).setText(f"Ampl. ({unit} pk)" if unit else "Ampl. (pk)")
        self.table_label.setText("Target harmonics" if closed else "Drive harmonics")
        self.formula.setText(
            (f"Target for {self.channel.currentText()}: y(t) = Σ a·cos(n·Ω·t + φ); the drive harmonics are found "
             "iteratively (Broyden). a = 0 suppresses that harmonic." if closed else
             "Drive: v(t) = Σ a·cos(n·Ω·t + φ), Ω = 2π·f0."))
        self.changed.emit()
