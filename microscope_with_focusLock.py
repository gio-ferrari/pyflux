# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:18:19 2018

@author: Florencia D. Choque
Mini script with three Threads: scan, TCSPC and focus lock
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:18:19 2018
@author: Florencia D. Choque
Mini script with two Threads for scan and focus lock
"""

import numpy as np
import time
import os
import sys
from datetime import date, datetime

from threading import Thread

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import qdarkstyle

from instrumental.drivers.cameras import uc480

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QDockWidget
from tkinter import Tk, filedialog

import drivers.ADwin as ADwin

#import focus_2 as focus
#import xy_tracking_2 as xy_tracking_flor

import scan as scan
import focus_flor_pruebaIDS_7_11 as focus

import tools.tools as tools

from drivers.minilasevo import MiniLasEvo
import drivers.ids_cam as ids_cam
π = np.pi



class Frontend(QtGui.QMainWindow):

    closeSignal = pyqtSignal()

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.setWindowTitle('PyFLUX')

        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        # Actions in menubar

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('Mini microscope for scan')


        # GUI layout
        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)

        ## scan dock


        ## tcspc dock


        ## xy tracking dock
        self.xyWidget = scan.Frontend()

        xyDock = QDockWidget('Scan', self)
        xyDock.setWidget(self.xyWidget)
        xyDock.setFeatures(QDockWidget.DockWidgetVerticalTitleBar | 
                                 QDockWidget.DockWidgetFloatable |
                                 QDockWidget.DockWidgetClosable)
        xyDock.setAllowedAreas(Qt.LeftDockWidgetArea)

        self.addDockWidget(Qt.LeftDockWidgetArea, xyDock)

        ## focus lock dock
        self.focusWidget = focus.Frontend()

        focusDock = QDockWidget('Focus Lock', self)
        focusDock.setWidget(self.focusWidget)
        focusDock.setFeatures(QDockWidget.DockWidgetVerticalTitleBar | 
                                 QDockWidget.DockWidgetFloatable |
                                 QDockWidget.DockWidgetClosable)
        focusDock.setAllowedAreas(Qt.RightDockWidgetArea)

        self.addDockWidget(Qt.RightDockWidgetArea, focusDock)

        # sizes to fit my screen properly
        self.xyWidget.setMinimumSize(700, 598)
        self.focusWidget.setMinimumSize(400, 598)
        self.move(1, 1)

    def make_connection(self, backend):

        backend.zWorker.make_connection(self.focusWidget)
        backend.xyWorker.make_connection(self.xyWidget)


    def closeEvent(self, *args, **kwargs):

        self.closeSignal.emit()
        time.sleep(1)

        focusThread.exit()
        xyThread.exit()
        super().closeEvent(*args, **kwargs)

        app.quit()        

class Backend(QtCore.QObject):

    askROIcenterSignal = pyqtSignal()
    moveToSignal = pyqtSignal(np.ndarray)
    xyzStartSignal = pyqtSignal()
    xyzEndSignal = pyqtSignal(str)
    xyMoveAndLockSignal = pyqtSignal(np.ndarray)

    def __init__(self, adw, camera, diodelaser, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.zWorker = focus.Backend(camera, adw)
        self.xyWorker = scan.Backend(adw, diodelaser)

    def setup_make_connection(self):
        pass

        #self.xyWorker.changedImage_tofocus.connect(self.zWorker.get_image)

    def make_connection(self, frontend):
        #2
        frontend.focusWidget.make_connection(self.zWorker)
        frontend.xyWidget.make_connection(self.xyWorker)

        frontend.closeSignal.connect(self.stop)

        self.setup_make_connection()

    def stop(self):

        self.xyWorker.stop()
        self.zWorker.stop()

if __name__ == '__main__':

    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()

    app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())



    port = tools.get_MiniLasEvoPort()
#    port = 'COM5'
    print('[scan] MiniLasEvo diode laser port:', port)
    diodelaser = MiniLasEvo(port)

    #if camera wasnt closed properly just keep using it without opening new one
    try:
        cam = ids_cam.IDS_U3()
    except:
        pass


    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
#    scan.setupDevice(adw)

    gui = Frontend()
    worker = Backend(adw, cam, diodelaser)

    gui.make_connection(worker)
    worker.make_connection(gui)


    # focus thread

    # xyzThread = QtCore.QThread()
    # worker.zWorker.moveToThread(xyzThread)
    # worker.zWorker.focusTimer.moveToThread(xyzThread)
    # worker.zWorker.focusTimer.timeout.connect(worker.zWorker.update)

    # worker.xyWorker.moveToThread(xyzThread)
    # worker.xyWorker.viewtimer.moveToThread(xyzThread)
    # worker.xyWorker.viewtimer.timeout.connect(worker.xyWorker.update_view)

    # xyzThread.start()

    focusThread = QtCore.QThread()
    worker.zWorker.moveToThread(focusThread)
    worker.zWorker.focusTimer.moveToThread(focusThread)
    worker.zWorker.focusTimer.timeout.connect(worker.zWorker.update)
    print("Focus Timer:", worker.zWorker.focusTimer)

    focusThread.start()

    # focus GUI thread

    # focusGUIThread = QtCore.QThread()
    # gui.focusWidget.moveToThread(focusGUIThread)

    # focusGUIThread.start()

    # xy worker thread

    xyThread = QtCore.QThread()
    worker.xyWorker.moveToThread(xyThread)
    worker.xyWorker.viewtimer.moveToThread(xyThread)
    worker.xyWorker.viewtimer.timeout.connect(worker.xyWorker.update_view)
    print("View Timer xy:", worker.xyWorker.viewtimer)

    xyThread.start()

    # #xy GUI thread

    # xyGUIThread = QtCore.QThread()
    # gui.xyWidget.moveToThread(xyGUIThread)

    # xyGUIThread.start()


    gui.showMaximized()
    app.exec_()