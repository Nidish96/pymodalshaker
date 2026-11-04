# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Editor for the 'Custom spectrum' excitation: a (frequency, amplitude) table with paste/CSV
import-export and a live plot of the breakpoints and the resulting FFT-line amplitudes."""

from __future__ import annotations

import csv
import re
from typing import List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from ..config import CUSTOM_KINDS

PRESETS = {
    "Flat": [[5, 1], [500, 1]],
    "Pink-ish (−3 dB/oct PSD)": [[5, 1], [500, 0.01]],
    "Low-frequency emphasis": [[5, 1], [50, 1], [100, 0.3], [500, 0.3]],
    "Notch around 100 Hz": [[5, 1], [90, 1], [100, 0.05], [110, 1], [500, 1]],
}


class CustomSpectrumEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.kind = QComboBox()
        self.kind.addItems(CUSTOM_KINDS)
        self.kind.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.kind.setMinimumContentsLength(14)
        self.kind.setToolTip("Amplitude: values are line amplitudes, interpolated linearly in frequency.\n"
                             "PSD: values are a power spectral density shape (e.g. g²/Hz breakpoints), "
                             "interpolated log-log (straight lines in dB/octave); amplitude = √PSD.\n"
                             "Only the shape matters — the overall level is set by the RMS.")
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Frequency (Hz)", "Amplitude / PSD"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(150)
        self.table.itemChanged.connect(self._edited)
        QShortcut(QKeySequence.Paste, self.table, activated=self.paste)

        btns = QHBoxLayout()
        for text, slot, tip in (("＋", self.add_row, "Add a row"),
                                ("－", self.remove_rows, "Remove selected rows"),
                                ("Paste", self.paste, "Paste two columns (tab, comma or space separated), e.g. from Excel"),
                                ("Import…", self.import_csv, "Read a two-column CSV / text file"),
                                ("Export…", self.export_csv, "Save the table as CSV")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            if len(text) == 1:
                b.setFixedWidth(32)
            btns.addWidget(b)
        btns.addStretch()
        self.preset = QComboBox()
        self.preset.addItem("Load a preset shape…")
        self.preset.addItems(list(PRESETS))
        self.preset.activated.connect(self._apply_preset)
        self.preset.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.preset.setMinimumContentsLength(14)

        self.plot = pg.PlotWidget()
        self.plot.setFixedHeight(170)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "Frequency", "Hz")
        self.plot.setLabel("left", "Relative amplitude")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.lines = self.plot.plot(pen=pg.mkPen("#2b7bb9", width=1.5), name="FFT lines")
        self.pts_item = self.plot.plot(pen=None, symbol="o", symbolSize=7, symbolBrush="#d35400",
                                     symbolPen=None, name="table points")
        self.msg = QLabel()
        self.msg.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        kl = QHBoxLayout()
        kl.addWidget(QLabel("Values are"))
        kl.addWidget(self.kind, 1)
        lay.addLayout(kl)
        lay.addWidget(self.table)
        lay.addLayout(btns)
        lay.addWidget(self.preset)
        lay.addWidget(self.plot)
        lay.addWidget(self.msg)
        self.kind.currentTextChanged.connect(lambda _: self.changed.emit())
        self._loading = False

    # ------------------------------------------------------------------ data
    def set_points(self, pts):
        self._loading = True
        self.table.setRowCount(0)
        for f, a in pts:
            self._append(f, a)
        self._loading = False

    def _append(self, f, a):
        i = self.table.rowCount()
        self.table.insertRow(i)
        self.table.setItem(i, 0, QTableWidgetItem(f"{float(f):g}"))
        self.table.setItem(i, 1, QTableWidgetItem(f"{float(a):g}"))

    def points(self) -> List[List[float]]:
        """Valid numeric rows, sorted by frequency (blank / bad rows are skipped)."""
        out = []
        for i in range(self.table.rowCount()):
            try:
                f = float(self.table.item(i, 0).text())
                a = float(self.table.item(i, 1).text())
                out.append([f, a])
            except (AttributeError, ValueError):
                continue
        return sorted(out)

    def bad_rows(self) -> int:
        return self.table.rowCount() - len(self.points())

    def set_kind(self, k: str):
        self.kind.setCurrentText(k)

    # ------------------------------------------------------------------ actions
    def _edited(self, *_):
        if not self._loading:
            self.changed.emit()

    def add_row(self):
        pts = self.points()
        f = pts[-1][0] * 1.5 if pts else 10.0
        a = pts[-1][1] if pts else 1.0
        self._loading = True
        self._append(f, a)
        self._loading = False
        self.changed.emit()

    def remove_rows(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows or [self.table.rowCount() - 1]:
            if r >= 0:
                self.table.removeRow(r)
        self.changed.emit()

    @staticmethod
    def parse_text(text: str) -> List[List[float]]:
        pts = []
        for line in text.splitlines():
            parts = [p for p in re.split(r"[\t,; ]+", line.strip()) if p]
            if len(parts) < 2:
                continue
            try:
                pts.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue          # header lines etc.
        return pts

    def paste(self):
        pts = self.parse_text(QApplication.clipboard().text())
        if not pts:
            QMessageBox.information(self, "Paste", "The clipboard has no two-column numeric data.")
            return
        self.set_points(pts)
        self.changed.emit()

    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import spectrum", "", "CSV / text (*.csv *.txt *.tsv);;All (*)")
        if not path:
            return
        with open(path, encoding="utf-8-sig") as f:
            pts = self.parse_text(f.read())
        if not pts:
            QMessageBox.warning(self, "Import", "No two-column numeric data found in the file.")
            return
        self.set_points(pts)
        self.changed.emit()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export spectrum", "custom_spectrum.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frequency_Hz", "PSD" if self.kind.currentText().startswith("PSD") else "amplitude"])
            w.writerows(self.points())

    def _apply_preset(self, i: int):
        if i <= 0:
            return
        name = self.preset.itemText(i)
        if "PSD" in name:
            self.kind.setCurrentText(CUSTOM_KINDS[1])
        self.set_points(PRESETS[name])
        self.preset.setCurrentIndex(0)
        self.changed.emit()

    # ------------------------------------------------------------------ preview
    def show_preview(self, f_lines: np.ndarray, amp_lines: np.ndarray, problems: List[str]):
        """f_lines/amp_lines: the interpolated FFT-line amplitudes for the current fs and block size."""
        pts = np.asarray(self.points(), float).reshape(-1, 2)
        peak = amp_lines.max() if amp_lines.size and amp_lines.max() > 0 else 1.0
        self.lines.setData(f_lines, amp_lines / peak, stepMode=None)
        if pts.size:
            vals = np.sqrt(np.maximum(pts[:, 1], 0)) if self.kind.currentText().startswith("PSD") else pts[:, 1]
            self.pts_item.setData(pts[:, 0], vals / peak)
        else:
            self.pts_item.setData([], [])
        bad = self.bad_rows()
        msgs = list(problems) + ([f"{bad} row(s) are not numbers and are ignored."] if bad else [])
        nz = int(np.count_nonzero(amp_lines))
        self.msg.setText("<span style='color:#c0392b'>" + "<br>".join(msgs) + "</span>" if msgs else
                         f"<span style='color:#1e8449'>✓ {nz} FFT lines excited</span> "
                         "<span style='color:gray'>(blue: line amplitudes, orange: table points; "
                         "scaled to peak 1 — the RMS sets the level)</span>")
