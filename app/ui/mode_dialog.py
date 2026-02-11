from __future__ import annotations

from typing import List, Dict, Any

from PyQt5 import QtCore, QtWidgets


class ModeDialog(QtWidgets.QDialog):
    """
    Простой диалог для режима: выбрать один ряд из списка и запустить анализ.
    Никаких дополнительных шагов/страниц — только список + кнопки ОК/Отмена.
    """

    def __init__(self, items: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор ряда для режима")
        self.resize(420, 160)

        self._items = items

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Выберите ряд, который отвечает за режим:"))

        self._combo = QtWidgets.QComboBox()
        for item in self._items:
            label = item.get("label", "")
            self._combo.addItem(label, item)
        layout.addWidget(self._combo)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_accept(self):
        item = self._combo.currentData()
        main_win = self.parent()
        if not isinstance(item, dict) or main_win is None:
            self.reject()
            return
        if not hasattr(main_win, "_run_mode_analysis_for_item"):
            self.reject()
            return
        main_win._run_mode_analysis_for_item(item)
        self.accept()

