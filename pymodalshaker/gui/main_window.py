# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Main window: tabs, menus, hardware/simulation mode and the measurement thread."""

from __future__ import annotations

import os
import time
import traceback
from collections import deque

from PySide6.QtCore import QObject, QSettings, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QLabel, QMainWindow, QMessageBox, QTabWidget)

from ..backends import discover_devices, make_backend
from ..config import SUITES, AppConfig, default_config
from ..engine import Callbacks, MeasurementEngine
from .channel_setup import ChannelTable
from .measurement import MeasurementTab
from .output_setup import OutputSetup
from .viewer import ViewerTab


def history_capacity(cfg: AppConfig) -> int:
    """How many recent blocks the GUI may show: up to 100, limited to ~300 MB of block data."""
    n_sig = len(cfg.enabled_channels()) + 2
    per_block = n_sig * cfg.output.block_size * 8
    return max(cfg.output.dwell_blocks + 2, min(100, int(3e8 // max(per_block, 1))))


class MeasurementWorker(QObject):
    """Runs MeasurementEngine in a QThread and turns its callbacks into Qt signals."""
    log = Signal(str)
    blocks = Signal(object, float, int)       # (recent-block history list, fs, backlog)
    progress = Signal(int, int, str)
    result = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    control = Signal(object)                    # stepped sine controller events
    ask = Signal(object)                        # stepped sine: control failed, user decides

    def __init__(self, cfg: AppConfig, simulate: bool):
        super().__init__()
        self.cfg = cfg
        self.simulate = simulate
        self.engine = None
        self._last_emit = 0.0
        # every block goes into this rolling window (dwell + 2 blocks); the GUI gets snapshots of
        # it at a limited rate, so the history never misses a block even with tiny blocks
        self.history = deque(maxlen=history_capacity(cfg))
        self._n_blocks = 0

    def _on_block(self, data, drive, trig, label, backlog):
        D = self.cfg.output.dwell_blocks
        text = {"decay": "settling", "fadein": "fade in", "fadeout": "fade out",
                "abort-fade": "fade out", "tail": "end"}.get(label.kind, "")
        step_kind = getattr(label, "step_kind", None)
        line = None
        freq, drive_key = None, None
        if step_kind is not None and self.cfg.suite == SUITES[3]:      # phase quadrature
            freq = getattr(label, "freq", None)
            if freq:
                drive_key = (label.r, round(float(freq), 9), round(float(label.amp), 12))
            text = {"settle": f"{freq or 0:.5g} Hz settle", "measure": f"it{label.it} {freq or 0:.5g} Hz",
                    "hold": "hold", "pause": "PAUSED", "dwell": f"L{label.r + 1} converged"}.get(step_kind, text)
            converged = step_kind == "dwell"
        elif step_kind is not None and self.cfg.suite == SUITES[2]:      # harmonic
            if step_kind in ("settle", "measure", "hold", "test", "pause"):
                line = int(self.engine.k)
            text = {"settle": "settle", "measure": f"it{label.it}", "hold": "hold", "pause": "PAUSED",
                    "test": f"test {getattr(label, 'test_no', 0)}"}.get(step_kind, text)
            converged = step_kind == "test"
        elif step_kind is not None:                  # stepped sine
            if step_kind in ("settle", "measure", "hold", "dwell", "pause") and 0 <= label.r < len(self.engine.lines):
                line = int(self.engine.lines[label.r])
            f = self.engine.freqs[label.r] if 0 <= label.r < len(self.engine.freqs) else 0
            text = {"settle": f"{f:g} Hz settle", "measure": f"{f:g} Hz it{label.it}", "hold": f"{f:g} Hz hold",
                    "dwell": f"{f:g} Hz dwell", "pause": f"{f:g} Hz PAUSED"}.get(step_kind, text)
            converged = step_kind == "dwell"
        else:
            if label.kind == "excite":
                text = f"R{label.r + 1}·{label.d + 1}/{D}"
            converged = label.kind == "excite" and label.d == D - 1
        self.history.append({"data": data.copy(), "drive": drive.copy(),
                             "trig": None if trig is None else trig.copy(), "text": text,
                             "converged": converged, "line": line, "freq": freq, "drive_key": drive_key,
                             "start": self._n_blocks * data.shape[1]})
        self._n_blocks += 1
        now = time.perf_counter()
        if now - self._last_emit >= 0.08:          # ≤ ~12 GUI updates per second
            self._last_emit = now
            self.blocks.emit(list(self.history), float(self.engine.summary.fs), int(backlog))

    @Slot()
    def run(self):
        summary = {}
        try:
            backend = make_backend(self.cfg, self.simulate)
            cb = Callbacks(log=self.log.emit, block=self._on_block, progress=self.progress.emit,
                           realization=lambda r, res, conv: self.result.emit(res),
                           finished=lambda d: summary.update(d),
                           control=self.control.emit, ask=self.ask.emit)
            if self.cfg.suite == SUITES[1]:
                from ..stepped import SteppedSineEngine
                self.engine = SteppedSineEngine(self.cfg, backend, cb)
            elif self.cfg.suite == SUITES[2]:
                from ..harmonic_suite import HarmonicEngine
                self.engine = HarmonicEngine(self.cfg, backend, cb)
            elif self.cfg.suite == SUITES[3]:
                from ..pq_suite import PQEngine
                self.engine = PQEngine(self.cfg, backend, cb)
            else:
                self.engine = MeasurementEngine(self.cfg, backend, cb)
            self.engine.run()
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=3)}")
        self.finished.emit(summary)

    def stop(self):
        if self.engine is not None:
            self.engine.request_stop()

    def answer(self, choice: str):
        if self.engine is not None and hasattr(self.engine, "answer"):
            self.engine.answer(choice)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from .. import __version__
        self.setWindowTitle(f"pymodalshaker {__version__} — shaker modal testing with NI cDAQ")
        self.resize(1400, 860)
        # settings key kept from the pre-release name, so a saved last setup survives the rename
        self.settings = QSettings("ModalShaker", "ModalShakerApp")
        self.current_file = ""

        self.chan = ChannelTable()
        self.out = OutputSetup()
        self.meas = MeasurementTab()
        self.viewer = ViewerTab()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.chan, "1 · Channel Setup")
        self.tabs.addTab(self.out, "2 · Output Setup")
        self.tabs.addTab(self.meas, "3 · Measurement")
        self.tabs.addTab(self.viewer, "4 · Data Viewer")
        self.tabs.setStyleSheet("QTabBar::tab { padding: 8px 18px; font-size: 10.5pt }")
        self.setCentralWidget(self.tabs)

        self.mode_label = QLabel()
        self.statusBar().addPermanentWidget(self.mode_label)

        # ---------------- menus
        m = self.menuBar().addMenu("&File")
        self._act(m, "&New setup", self.new_setup, QKeySequence.New)
        self._act(m, "&Open setup…", self.open_setup, QKeySequence.Open)
        self._act(m, "&Save setup", self.save_setup, QKeySequence.Save)
        self._act(m, "Save setup &as…", self.save_setup_as, QKeySequence.SaveAs)
        m.addSeparator()
        self._act(m, "&Quit", self.close, QKeySequence.Quit)
        h = self.menuBar().addMenu("&Hardware")
        self._act(h, "&Refresh NI devices", self.refresh_devices, "F5")
        self.act_sim = QAction("&Simulation mode (no hardware)", self, checkable=True)
        self.act_sim.toggled.connect(self._update_mode_label)
        h.addAction(self.act_sim)
        hm = self.menuBar().addMenu("&Help")
        self._act(hm, "&About pymodalshaker", self._about)

        self.meas.start_requested.connect(self.start_measurement)
        # the suite and the harmonic count are chosen on Output Setup; the Measurement tab follows
        self.out.suite.currentTextChanged.connect(self._suite_from_setup)
        self.out.changed.connect(self._sync_harmonic_count)
        self._sync_harmonic_count()
        self.chan.changed.connect(self._update_channel_names)
        self.out.check_rate_requested.connect(self._check_rate)
        self.meas.stop_requested.connect(self.stop_measurement)
        self.thread = None
        self.worker = None

        # ---------------- initial state
        self.refresh_devices(first=True)
        saved = self.settings.value("last_setup", "")
        try:
            cfg = AppConfig.from_json(saved) if saved else default_config()
        except Exception:
            cfg = default_config()
        if not cfg.save.folder:
            cfg.save.folder = os.path.join(os.path.expanduser("~"), "ModalTests")
        self.apply_config(cfg)

    def _about(self):
        from .. import __url__, __version__
        QMessageBox.about(
            self, "About pymodalshaker",
            f"<h3>pymodalshaker {__version__} (beta)</h3>"
            "<p>Shaker modal testing with NI CompactDAQ: random, stepped-sine, harmonic and "
            "phase-quadrature excitation, with FRF, harmonic and resonance analysis.</p>"
            "<p>Copyright © 2026 The pymodalshaker authors.</p>"
            "<p>This program is free software: you can redistribute it and/or modify it under the terms of "
            "the GNU General Public License as published by the Free Software Foundation, either version 3 "
            "of the License, or (at your option) any later version.</p>"
            "<p>This program is distributed in the hope that it will be useful, but <b>WITHOUT ANY "
            "WARRANTY</b>; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR "
            "PURPOSE. See the GNU General Public License "
            "(<a href='https://www.gnu.org/licenses/gpl-3.0.html'>gnu.org/licenses/gpl-3.0</a>) for details.</p>"
            "<p>Developed with the help of an AI model (Claude Opus 5.5); results come with no guarantee of "
            f"correctness.</p><p><a href='{__url__}'>{__url__}</a></p>")

    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(shortcut)
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    # ------------------------------------------------------------------ devices / mode
    def refresh_devices(self, first=False):
        info = discover_devices()
        if first:
            self.act_sim.setChecked(not (info["ok"] and info["devices"]))
        if not info["ai"]:   # no hardware: offer the channel names of a virtual cDAQ-9174
            info["ai"] = [f"cDAQ1Mod{m}/ai{i}" for m in (1, 3) for i in range(4)]
            info["ao"] = [f"cDAQ1Mod2/ao{i}" for i in range(4)]
            info["do"] = [f"cDAQ1Mod4/port0/line{i}" for i in range(8)]
            info["pfi"] = [f"/cDAQ1Mod4/PFI{i}" for i in range(8)]
        self.chan.set_ai_names(info["ai"])
        self.out.set_output_names(info["ao"], info["do"], info["pfi"])
        self.hw_message = info["message"]
        self._update_mode_label()
        self.statusBar().showMessage(info["message"], 8000)

    def _update_mode_label(self, *_):
        if self.act_sim.isChecked():
            self.mode_label.setText("<b style='color:#d35400'>SIMULATION MODE</b> — no hardware is used")
        else:
            self.mode_label.setText(f"<b style='color:#1e8449'>HARDWARE</b> — {self.hw_message}")

    # ------------------------------------------------------------------ config
    def _suite_from_setup(self, suite: str):
        self.meas.set_suite_mode(suite)
        self._sync_harmonic_count()

    def _sync_harmonic_count(self):
        """The suite's single harmonic-count setting → every harmonics display on the Measurement tab."""
        if self.out.get_suite() == SUITES[3]:
            n = self.out.get_pq().n_harmonics
        elif self.out.get_suite() == SUITES[2]:
            n = self.out.get_harmonic().analysis_orders()
        else:
            n = self.out.ss_nharm.value()
        if n != getattr(self, "_harm_n", None):
            self._harm_n = n
            self.meas.set_harmonic_count(n)

    def _check_rate(self):
        """Output Setup → 'Check with hardware': the sample rate DAQmx would really use."""
        if self.act_sim.isChecked():
            self.out.show_rate_check({}, simulated=True)
            return
        cfg = self.gather_config()
        if not cfg.enabled_channels() or not cfg.output.physical.strip():
            QMessageBox.warning(self, "Check rate", "Enable at least one input channel and set the AO channel first.")
            return
        from PySide6.QtGui import QGuiApplication
        from ..backends import check_rates
        QGuiApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Asking NI-DAQmx for the actual sample rate…")
        try:
            res = check_rates(cfg)
        except Exception as e:                     # e.g. NI-DAQmx not installed
            res = {"requested": cfg.output.fs, "error": f"{type(e).__name__}: {e}", "exact": []}
        finally:
            QGuiApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()
        self.out.show_rate_check(res)

    def _update_channel_names(self):
        en = [c for c in self.chan.get_channels() if c.enabled]
        ref = next((c.name for c in en if c.is_reference), "")
        self.out.set_channel_names([(c.name, c.units) for c in en], ref)

    def gather_config(self) -> AppConfig:
        return AppConfig(channels=self.chan.get_channels(), output=self.out.get_config(),
                         save=self.meas.get_save(), trigger=self.out.get_trigger(),
                         suite=self.out.get_suite(), stepped=self.out.get_stepped(),
                         harmonic=self.out.get_harmonic(), pq=self.out.get_pq(),
                         null_bridges=self.chan.null_bridges.isChecked())

    def apply_config(self, cfg: AppConfig):
        self.chan.set_channels(cfg.channels)
        self.chan.null_bridges.setChecked(cfg.null_bridges)
        self.out.set_config(cfg.output)
        self.out.set_trigger(cfg.trigger)
        self._update_channel_names()
        self.out.set_stepped(cfg.suite, cfg.stepped)
        self._suite_from_setup(cfg.suite)
        self.out.set_harmonic(cfg.harmonic)
        self.out.set_pq(cfg.pq)
        self._sync_harmonic_count()
        self.meas.set_save(cfg.save)

    def new_setup(self):
        cfg = default_config()
        cfg.save.folder = self.meas.folder.text()
        self.apply_config(cfg)
        self.current_file = ""

    def open_setup(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open setup", "", "Setup (*.json)")
        if p:
            try:
                with open(p, encoding="utf-8") as f:
                    self.apply_config(AppConfig.from_json(f.read()))
                self.current_file = p
                self.statusBar().showMessage(f"Loaded {p}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Open setup", f"Could not read setup:\n{e}")

    def save_setup(self):
        if not self.current_file:
            return self.save_setup_as()
        with open(self.current_file, "w", encoding="utf-8") as f:
            f.write(self.gather_config().to_json())
        self.statusBar().showMessage(f"Saved {self.current_file}", 5000)

    def save_setup_as(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save setup", "modal_setup.json", "Setup (*.json)")
        if p:
            self.current_file = p
            self.save_setup()

    # ------------------------------------------------------------------ measurement
    def start_measurement(self):
        cfg = self.gather_config()
        errs = cfg.validate()
        if errs:
            QMessageBox.warning(self, "Check the setup", "Please fix the following:\n\n• " + "\n• ".join(errs))
            return
        if not self.act_sim.isChecked() and self.out.ao.count() == 0:
            QMessageBox.warning(self, "No hardware", "No NI devices were found. Connect the cDAQ and use "
                                "Hardware → Refresh, or enable simulation mode.")
            return
        sim = self.act_sim.isChecked()
        stepped = cfg.suite == SUITES[1]
        harmonic = cfg.suite == SUITES[2]
        pq = cfg.suite == SUITES[3]
        if not sim:
            st = cfg.stepped
            if pq:
                lv = cfg.pq.levels()
                what = (f"sine levels {lv.min():.4g}–{lv.max():.4g} V peak ({lv.size} levels), frequency tracked from "
                        f"{cfg.pq.f_init:g} Hz within {cfg.pq.f_min:g}–{cfg.pq.f_max:g} Hz "
                        f"(limit ±{cfg.output.max_voltage:g} V)")
            elif harmonic:
                h = cfg.harmonic
                spec = ", ".join(f"h{n}: {a:g}∠{p:g}°" for n, a, p in h.rows())
                what = (f"a harmonic waveform at f0 = {h.f0:g} Hz, "
                        + (f"controlled on {h.target_channel} to [{spec}] (initial drive {h.initial_drive:g} V peak)"
                           if h.closed_loop else f"drive [{spec}] V") + f" (limit ±{cfg.output.max_voltage:g} V)")
            elif stepped:
                what = (f"a stepped sine {st.f_start:g}–{st.f_stop:g} Hz, "
                        + (f"controlled to {st.target_value:g} {st.target_label} on {st.target_channel} (initial drive "
                           f"{st.initial_drive:g} V peak)" if st.control.startswith("Secant")
                           else f"{st.drive_amplitude:g} V peak") + f" (limit ±{cfg.output.max_voltage:g} V)")
            else:
                what = (f"{cfg.output.excitation.lower()} at {cfg.output.rms:g} V RMS "
                        f"(limit ±{cfg.output.max_voltage:g} V)")
            ok = QMessageBox.question(
                self, "Start shaker excitation",
                f"The shaker will be driven with {what}.\n"
                + (f"Camera trigger: {cfg.trigger.physical}, {cfg.trigger.mode[0].lower() + cfg.trigger.mode[1:]}.\n"
                   if cfg.trigger.enabled else "") + "\n"
                "Check the amplifier gain and that the test article is secure. Start?")
            if ok != QMessageBox.Yes:
                return
        self.settings.setValue("last_setup", cfg.to_json())

        en = cfg.enabled_channels()
        o = cfg.output
        band = (cfg.stepped.f_start, cfg.stepped.f_stop) if stepped else (o.fmin, o.fmax)
        self.meas.prepare_run(en, cfg.reference_index(), band,
                              cfg.trigger.enabled and not cfg.trigger.frame_sync, o.dwell_blocks,
                              history_capacity(cfg), stepped, harmonic, pq)
        from .. import __version__
        self.meas.append_log(f"---- {time.strftime('%H:%M:%S')} start ({'simulation' if sim else 'hardware'}), "
                             f"app version {__version__} ----")
        if stepped:
            from ..config import harmonic_nyquist_note, stepped_frequencies
            fr = stepped_frequencies(cfg.stepped, cfg.output)[0]
            note = harmonic_nyquist_note(cfg.stepped.n_harmonics, float(fr[0]), float(fr[-1]), cfg.output.fs,
                                         [1024, 2048, 2560, 5120, 10240, 12800, 25600, 51200]) if fr.size else ""
            self.meas.append_log(f"Harmonics: orders 1…{cfg.stepped.n_harmonics} (Output Setup)"
                                 + (f". WARNING: {note}" if note else ", all below Nyquist."))
        self._set_editing_enabled(False)
        self.meas.set_running(True)
        self.tabs.setCurrentWidget(self.meas)

        self.thread = QThread(self)
        self.worker = MeasurementWorker(cfg, sim)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.meas.append_log, Qt.ConnectionType.QueuedConnection)
        self.worker.blocks.connect(self.meas.on_blocks, Qt.ConnectionType.QueuedConnection)
        self.worker.progress.connect(self.meas.on_progress, Qt.ConnectionType.QueuedConnection)
        self.worker.result.connect(self.meas.on_result, Qt.ConnectionType.QueuedConnection)
        self.worker.failed.connect(self._on_failed, Qt.ConnectionType.QueuedConnection)
        if pq:
            self.worker.control.connect(self.meas.on_pq_event, Qt.ConnectionType.QueuedConnection)
        else:
            ctl = self.meas.hcontrol if harmonic else self.meas.controller
            self.worker.control.connect(ctl.on_event, Qt.ConnectionType.QueuedConnection)
        self.worker.ask.connect(self._ask_control_failure, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self._on_finished, Qt.ConnectionType.QueuedConnection)
        self.worker.finished.connect(self.thread.quit, Qt.ConnectionType.QueuedConnection)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.start()

    @Slot(object)
    def _ask_control_failure(self, info: dict):
        """Stepped sine: control failed and the setup says 'ask me'. The drive is held at a safe
        level by the engine while this dialog is open."""
        best = info.get("best")
        best_txt = info.get("best_text") or (f"Best iteration: drive {best['amp']:.4g} V peak, error "
                                             f"{best['err_pct']:+.2f} %." if best else "")
        txt = (f"<b>Control did not succeed at {info['f']:g} Hz.</b><br>{info['reason']}<br><br>"
               + (f"{best_txt}<br>" if best else "There is no valid (non-overloaded) iteration.<br>")
               + "The drive is held at a safe level until you decide.")
        box = QMessageBox(QMessageBox.Warning, "Stepped sine: control failed", txt, parent=self)
        choices = info["choices"]
        buttons = {}
        for c, role in zip(choices, (QMessageBox.ActionRole, QMessageBox.AcceptRole,
                                     QMessageBox.ActionRole, QMessageBox.RejectRole)):
            b = box.addButton(c, role)
            buttons[b] = c
            if c.startswith("Accept") and not best:
                b.setEnabled(False)
        skip = (next((b for b, c in buttons.items() if c.startswith("Skip")), None)
                or next(b for b, c in buttons.items() if c.startswith("Stop")))
        box.setDefaultButton(skip)
        box.setEscapeButton(skip)
        box.exec()
        choice = buttons.get(box.clickedButton(), buttons[skip])
        self.meas.append_log(f"Your decision at {info['f']:g} Hz: {choice}")
        if self.worker is not None:
            self.worker.answer(choice)

    def stop_measurement(self):
        if self.worker:
            self.meas.btn_stop.setEnabled(False)
            self.worker.stop()

    @Slot(str)
    def _on_failed(self, msg: str):
        self.meas.append_log("ERROR: " + msg.splitlines()[0])
        QMessageBox.critical(self, "Measurement error", msg)

    @Slot(object)
    def _on_finished(self, summary: dict):
        self._set_editing_enabled(True)
        self.meas.set_running(False)
        done = summary.get("completed_realizations", 0)
        folder = summary.get("run_folder", "")
        if self.out.get_suite() == SUITES[1]:
            self.meas.phase.setText(("Stopped" if summary.get("aborted") else "Finished")
                                    + " — " + self.meas.controller.summary_text())
        elif self.out.get_suite() == SUITES[2]:
            self.meas.phase.setText(("Stopped" if summary.get("aborted") else "Finished")
                                    + " — " + self.meas.hcontrol.summary_text())
        elif self.out.get_suite() == SUITES[3]:
            self.meas.phase.setText(("Stopped" if summary.get("aborted") else "Finished")
                                    + " — " + self.meas.pqcontrol.summary_text())
        else:
            self.meas.phase.setText(("Stopped" if summary.get("aborted") else "Finished")
                                    + f" — {done} realization(s) averaged")
        if folder:
            self.meas.append_log(f"Files saved in: {folder}  (opened in the Data Viewer tab)")
            self.viewer.open_folder(folder)
        self.worker = None

    def _set_editing_enabled(self, on: bool):
        self.chan.setEnabled(on)
        self.out.setEnabled(on)
        for w in (self.meas.folder, self.meas.basename, self.meas.save_uff, self.meas.save_conv,
                  self.meas.save_each, self.meas.save_harm, self.meas.save_dwell,
                  self.meas.h_save, self.meas.h_mode, self.meas.h_every, self.meas.pq_blocks, self.meas.pq_uff,
                  *self.meas.uff.values()):
            w.setEnabled(on)
        if on:
            self.meas._h_save_enable()
        for a in self.menuBar().actions():
            a.setEnabled(on)

    def closeEvent(self, ev):
        if self.worker is not None:
            if QMessageBox.question(self, "Quit", "A measurement is running. Stop it and quit?") != QMessageBox.Yes:
                ev.ignore()
                return
            self.worker.stop()
            self.thread.quit()
            self.thread.wait(15000)
        self.settings.setValue("last_setup", self.gather_config().to_json())
        ev.accept()
