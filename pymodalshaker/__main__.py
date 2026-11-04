# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The pymodalshaker authors
# This file is part of pymodalshaker; see the LICENSE file for the full GPL v3 terms.

"""Start the GUI:  pymodalshaker   (or: python -m pymodalshaker)"""

import os
import sys

import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

from .gui.main_window import MainWindow


def _quiet_qt_noise() -> None:
    """Silence Qt's harmless Wayland text-input focus warnings ('Trying to disable … but 0x0 is
    focused', printed on GNOME/Wayland when focus moves between widgets and dialogs). Only that
    logging category is muted; rules you set yourself in QT_LOGGING_RULES are kept."""
    rules = os.environ.get("QT_LOGGING_RULES", "")
    if "qt.qpa.wayland.textinput" not in rules:
        os.environ["QT_LOGGING_RULES"] = (rules + ";" if rules else "") + "qt.qpa.wayland.textinput=false"


def main() -> int:
    _quiet_qt_noise()                      # must run before the QApplication is created
    app = QApplication(sys.argv)
    app.setApplicationName("pymodalshaker")
    app.setStyle("Fusion")
    pg.setConfigOptions(antialias=True, background="w", foreground="k")
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
