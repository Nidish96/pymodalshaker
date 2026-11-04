# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Compare runs: overlay FRFs (or raw amplitude spectra) of the same channel from several saved
measurements. Channels are matched by name; each run uses its own reference."""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPushButton, QSplitter, QVBoxLayout, QWidget)

from ..frf import harmonic_frf
from ..results import amplitude_spectra, detect_suite, find_runs, frf_table, scan_run
from .plots import export_dialog

PALETTE = ["#2b7bb9", "#d35400", "#27ae60", "#8e44ad", "#c0392b", "#16a085", "#7f8c8d", "#f39c12",
           "#2c3e50", "#e84393"]


class CompareTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.runs: Dict[str, object] = {}          # folder -> Run (loaded on demand)

        # ---------------- run list
        self.list = QListWidget()
        self.list.setToolTip("Tick the runs to overlay")
        self.list.itemChanged.connect(self._refresh_items)
        b_add = QPushButton("＋ Add runs from folder…")
        b_add.clicked.connect(self._add_folder)
        b_all = QPushButton("All")
        b_none = QPushButton("None")
        b_all.clicked.connect(lambda: self._check_all(True))
        b_none.clicked.connect(lambda: self._check_all(False))
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("<b>Runs</b> (tick to overlay)"))
        ll.addWidget(self.list, 1)
        bl = QHBoxLayout()
        bl.addWidget(b_all)
        bl.addWidget(b_none)
        ll.addLayout(bl)
        ll.addWidget(b_add)

        # ---------------- controls
        self.qty = QComboBox()
        self.qty.addItems(["FRF", "Amplitude spectrum"])
        self.item = QComboBox()
        self.item.setToolTip("Channel name; matched across runs")
        self.est = QComboBox()
        self.est.addItems(["H1", "H2", "Hv"])
        self.tdms = QCheckBox("Recompute from TDMS when available")
        self.tdms.setChecked(True)
        self.tdms.setToolTip("Use the runs' time data (full range, current reference); otherwise the stored UFF files")
        self.info = QLabel()
        exp = QPushButton("📷 Export image…")
        for w in (self.qty, self.tdms):
            sig = w.currentIndexChanged if isinstance(w, QComboBox) else w.toggled
            sig.connect(self._refresh_items)
        for w in (self.item, self.est):
            w.currentIndexChanged.connect(self._draw)

        # ---------------- plots
        self.p_q, self.p_mag, self.p_ph = pg.PlotWidget(), pg.PlotWidget(), pg.PlotWidget()
        self.p_mag.setLogMode(y=True)
        self.p_ph.setLabel("left", "Phase", "°")
        self.p_ph.setYRange(-180, 180)
        self.p_ph.setLabel("bottom", "Frequency", "Hz")
        for p in (self.p_q, self.p_mag, self.p_ph):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis("bottom").enableAutoSIPrefix(False)
        self.p_ph.setXLink(self.p_mag)
        self.p_q.setXLink(self.p_mag)
        self.legend = self.p_mag.addLegend(offset=(-10, 10))
        self.plots = QSplitter(Qt.Vertical)
        for p in (self.p_q, self.p_mag, self.p_ph):
            self.plots.addWidget(p)
        self.plots.setSizes([140, 320, 160])
        exp.clicked.connect(lambda: export_dialog(self, self.plots, "compare_runs.png"))

        ctl = QHBoxLayout()
        for lbl, w in (("Show", self.qty), ("  Channel", self.item), ("  Estimator", self.est)):
            ql = QLabel(lbl)
            ctl.addWidget(ql)
            ctl.addWidget(w, 1 if w is self.item else 0)
            if w is self.est:
                self.est_label = ql
        ctl.addWidget(self.tdms)
        ctl.addWidget(exp)
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addLayout(ctl)
        rl.addWidget(self.info)
        rl.addWidget(self.plots, 1)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(lw)
        split.addWidget(rw)
        split.setSizes([260, 1000])
        lay = QVBoxLayout(self)
        lay.addWidget(split)

    # ------------------------------------------------------------------ runs
    def set_runs(self, folders: List[str]):
        """Called by the viewer after a scan: offers these runs (unticked), keeps added ones."""
        have = {self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())}
        self.list.blockSignals(True)
        for f in folders:
            if f not in have and detect_suite(f) != "harmonic":     # harmonic runs have no FRFs
                self._add_item(f)
        self.list.blockSignals(False)

    def _add_item(self, folder: str, checked: bool = False):
        suite = detect_suite(folder)
        it = QListWidgetItem((f"[{suite}] " if suite else "") + os.path.basename(os.path.normpath(folder)))
        it.setData(Qt.UserRole, folder)
        it.setToolTip(folder)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
        it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.list.addItem(it)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with runs")
        if d:
            self.set_runs(find_runs(d))

    def _check_all(self, on: bool):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.list.blockSignals(False)
        self._refresh_items()

    def check(self, folders: List[str]):
        """Programmatic selection (also used by tests)."""
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in folders else Qt.Unchecked)
        self.list.blockSignals(False)
        self._refresh_items()

    def _selected(self):
        out = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                f = it.data(Qt.UserRole)
                if f not in self.runs:
                    self.runs[f] = scan_run(f)
                out.append((it.text(), self.runs[f]))
        return out

    # ------------------------------------------------------------------ data
    def _table(self, run) -> Dict[str, Dict]:
        if self.qty.currentIndex() == 1:
            return amplitude_spectra(run)
        return frf_table(run, self.tdms.isChecked())

    def _refresh_items(self, *_):
        spec = self.qty.currentIndex() == 1
        for w in (self.est, self.est_label, self.tdms):
            w.setVisible(not spec)
        names: List[str] = []
        for _, run in self._selected():
            for n in self._table(run):
                if n not in names:
                    names.append(n)
        cur = self.item.currentText()
        self.item.blockSignals(True)
        self.item.clear()
        self.item.addItems(names)
        if cur in names:
            self.item.setCurrentText(cur)
        self.item.blockSignals(False)
        self._draw()

    def _draw(self, *_):
        for p in (self.p_q, self.p_mag, self.p_ph):
            for it in list(p.listDataItems()):
                p.removeItem(it)
        self.legend.clear()
        spec = self.qty.currentIndex() == 1
        self.p_q.setVisible(not spec)
        self.p_ph.setVisible(not spec)
        self.p_mag.setLabel("bottom", "Frequency" if spec else "", "Hz" if spec else "")
        name, h, est = self.item.currentText(), 1, self.est.currentText()
        sel = self._selected()
        if not name or not sel:
            self.info.setText("Tick two or more runs on the left and choose a channel.")
            return
        shown, skipped, units, qlabel = 0, [], set(), set()
        for i, (label, run) in enumerate(sel):
            d = self._table(run).get(name)
            if d is None:
                skipped.append(f"{label} (no '{name}')")
                continue
            color = PALETTE[i % len(PALETTE)]
            sym = "o" if run.stepped else ("s" if getattr(run, "pq", False) else None)
            kw = dict(pen=pg.mkPen(color, width=1.5, style=Qt.DashLine if getattr(run, "pq", False) else Qt.SolidLine),
                      symbol=sym, symbolSize=8 if getattr(run, "pq", False) else 5, symbolBrush=color, symbolPen=None)
            if getattr(run, "pq", False):
                label += "  (resonance per level)"
            if spec:
                if "harm" in d:
                    y = np.abs(d["harm"][:, h - 1])
                elif h == 1:
                    y = d["amp"]
                else:
                    skipped.append(f"{label} (random run: no harmonics)")
                    continue
                f = d["freqs"]
                ok = np.isfinite(y) & (f > 0)
                self.p_mag.plot(f[ok], y[ok] + 1e-30, name=label, **kw)
                units.add(d["units"])
            else:
                if h > 1:
                    if d.get("res") is None or d["res"].harm is None:
                        skipped.append(f"{label} (no harmonic data)")
                        continue
                    Hh, q = harmonic_frf(d["res"], d["j"], h)
                    f = d["res"].freqs
                    qlabel.add(f"Harmonic {h} level (dBc)")
                else:
                    Hh, q, f = d[est], d["q"], d["freqs"]
                    qlabel.add(run.quality_label)
                if Hh is None:
                    skipped.append(f"{label} (no {est})")
                    continue
                ok = np.isfinite(Hh) & (f > 0)
                self.p_mag.plot(f[ok], np.abs(Hh[ok]) + 1e-30, name=label, **kw)
                self.p_ph.plot(f[ok], np.degrees(np.angle(Hh[ok])), **kw)
                if q is not None:
                    self.p_q.plot(f[ok], np.real(q[ok]), **kw)
                units.add(d["units"])
            shown += 1
        u = " / ".join(sorted(u for u in units if u))
        if spec:
            self.p_mag.setLabel("left", f"|{name}|" + (f" harmonic {h}" if h > 1 else ""), u)
        else:
            self.p_mag.setLabel("left", (f"|Y{h}/X1|" if h > 1 else f"|{est}|") + (f"  [{u}]" if u else ""))
            self.p_q.setLabel("left", " / ".join(sorted(qlabel)) or "Coherence / η")
            if h > 1:
                self.p_q.enableAutoRange(axis="y")
            else:
                self.p_q.setYRange(0, 1.05)
        msg = f"{shown} run(s) overlaid."
        if skipped:
            msg += " Not shown: " + "; ".join(skipped)
        if len(units) > 1:
            msg += "  ⚠ different units across runs"
        self.info.setText(msg)
