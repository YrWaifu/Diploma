# tools/dump_ui.py (fixed)
# Создаёт валидный для Qt Designer .ui каркас из текущего MainWindow.
# Ограничения: кастомные виджеты заменяются на плейсхолдеры QWidget,
# действия тулбара выгружаются как <action> и привязываются через <addaction>.

from __future__ import annotations
from pathlib import Path
import sys
from xml.etree.ElementTree import Element, SubElement, ElementTree
from PyQt5 import QtWidgets

# --- импорт приложения ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui.main_window import MainWindow  # noqa: E402


# -- маленькие хелперы для .ui-XML --

def prop(parent: Element, name: str, tag: str, value: str) -> Element:
    p = SubElement(parent, "property", name=name)
    v = SubElement(p, tag)
    v.text = value
    return p


def attrib(parent: Element, name: str, tag: str, value: str) -> Element:
    a = SubElement(parent, "attribute", name=name)
    v = SubElement(a, tag)
    v.text = value
    return a


def number(parent: Element, name: str, value: int) -> Element:
    p = SubElement(parent, "property", name=name)
    v = SubElement(p, "number")
    v.text = str(int(value))
    return p


def rect(parent: Element, name: str, x: int, y: int, w: int, h: int) -> Element:
    p = SubElement(parent, "property", name=name)
    r = SubElement(p, "rect")
    for tag, val in (("x", x), ("y", y), ("width", w), ("height", h)):
        el = SubElement(r, tag)
        el.text = str(int(val))
    return p


def add_action_def(widget: Element, qaction: QtWidgets.QAction) -> str:
    name = qaction.objectName() or f"action_{(qaction.text() or 'Action').strip().replace(' ', '_')}"
    act = SubElement(widget, "action", name=name)
    # text
    prop(act, "text", "string", qaction.text() or "")
    # tooltip
    if qaction.toolTip():
        prop(act, "toolTip", "string", qaction.toolTip())
    # checkable/checked
    if qaction.isCheckable():
        prop(act, "checkable", "bool", "true")
        if qaction.isChecked():
            prop(act, "checked", "bool", "true")
    return name


def build_ui_skeleton(win: MainWindow) -> Element:
    ui = Element("ui", version="4.0")
    cls = SubElement(ui, "class"); cls.text = win.__class__.__name__

    mw = SubElement(ui, "widget", **{"class": "QMainWindow", "name": "MainWindow"})

    # geometry + title
    rect(mw, "geometry", 0, 0, max(800, win.width()), max(600, win.height()))
    prop(mw, "windowTitle", "string", win.windowTitle() or "ODiploma")

    # central widget with HBox and three placeholders
    central = SubElement(mw, "widget", **{"class": "QWidget", "name": "centralwidget"})
    hbox = SubElement(central, "layout", **{"class": "QHBoxLayout", "name": "horizontalLayout"})

    def add_placeholder(parent_layout: Element, name: str):
        item = SubElement(parent_layout, "item")
        w = SubElement(item, "widget", **{"class": "QWidget", "name": name})
        SubElement(w, "layout", **{"class": "QVBoxLayout", "name": name + "Layout"})
        return w

    add_placeholder(hbox, "leftPlaceholder")
    add_placeholder(hbox, "plotPlaceholder")
    add_placeholder(hbox, "coordsPlaceholder")

    # ВАЖНО: stretch задаётся на сам QHBoxLayout одной строкой
    prop(hbox, "stretch", "string", "1,4,1")

    # ToolBar
    toolbar = SubElement(mw, "widget", **{"class": "QToolBar", "name": "toolBar"})
    prop(toolbar, "windowTitle", "string", "toolBar")
    attrib(toolbar, "toolBarArea", "enum", "TopToolBarArea")
    attrib(toolbar, "toolBarBreak", "bool", "false")

    # StatusBar (пустой, но многим темам нужен)
    SubElement(mw, "widget", **{"class": "QStatusBar", "name": "statusbar"})

    # Actions из первого тулбара окна
    for tb in win.findChildren(QtWidgets.QToolBar):
        for qact in tb.actions():
            name = add_action_def(mw, qact)
            SubElement(toolbar, "addaction", name=name)
        break  # берём первый тулбар

    # хвосты
    SubElement(ui, "resources")
    SubElement(ui, "connections")
    return ui


def write_ui_file(root: Element, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show(); app.processEvents()
    ui_root = build_ui_skeleton(w)
    out = ROOT / "app/ui/forms/main_window.ui"
    write_ui_file(ui_root, out)
    print(f"OK: записал {out}")
