# entrypoint: python -m app.app или python app/app.py
import sys
from PyQt5 import QtWidgets
import pyqtgraph as pg

from app.ui.main_window import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pg.setConfigOptions(background=None, foreground='k', antialias=True)

    w = MainWindow()
    w.showMaximized()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
