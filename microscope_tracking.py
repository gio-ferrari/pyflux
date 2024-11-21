# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:18:19 2018

@author: Florencia D. Choque based on microscope from PyFlux made by Luciano Masullo
In the widget we have 3 main parts: the module TCSPC, the focus control (xyz_tracking) and the scan module.
Also 2 modules run in the backend: minflux and psf, these were not modified, they belong to the older minflux control made by Luciano Masullo
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

import drivers.picoharp as picoharp
from drivers.minilasevo import MiniLasEvo

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QDockWidget
from tkinter import Tk, filedialog

import drivers.ADwin as ADwin
import drivers.ids_cam as ids_cam

import xyz_focus_lock as focus
import scan #scan works with minilasEvo 632, new_scan to work with another wavelength
import tcspc
import measurements.minflux as minflux
import measurements.psf as psf

import tools.tools as tools

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
        fileMenu = menubar.addMenu('Measurement')
        
        self.psfWidget = psf.Frontend()
        self.minfluxWidget = minflux.Frontend()

        self.psfMeasAction = QtGui.QAction('PSF measurement', self)
        self.psfMeasAction.setStatusTip('Routine to measure one MINFLUX PSF')
        fileMenu.addAction(self.psfMeasAction)
        
        self.psfMeasAction.triggered.connect(self.psf_measurement)
    
        self.minfluxMeasAction = QtGui.QAction('MINFLUX measurement', self)
        self.minfluxMeasAction.setStatusTip('Routine to perform a tcspc-MINFLUX measurement')
        fileMenu.addAction(self.minfluxMeasAction)
        
        self.minfluxMeasAction.triggered.connect(self.minflux_measurement)
        
        # GUI layout
        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)

        ## scan dock
        self.scanWidget = scan.Frontend()

        scanDock = QDockWidget('Confocal scan', self)
        scanDock.setWidget(self.scanWidget)
        scanDock.setFeatures(QDockWidget.DockWidgetVerticalTitleBar | 
                                 QDockWidget.DockWidgetFloatable |
                                 QDockWidget.DockWidgetClosable)
        scanDock.setAllowedAreas(Qt.LeftDockWidgetArea)

        self.addDockWidget(Qt.LeftDockWidgetArea, scanDock)

        ## tcspc dock
        self.tcspcWidget = tcspc.Frontend()
        
        tcspcDock = QDockWidget('Time-correlated single-photon counting', self)
        tcspcDock.setWidget(self.tcspcWidget)
        tcspcDock.setFeatures(QDockWidget.DockWidgetVerticalTitleBar | 
                                 QDockWidget.DockWidgetFloatable |
                                 QDockWidget.DockWidgetClosable)
        tcspcDock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, tcspcDock)
        
        ## focus lock dock
        self.focusWidget = focus.Frontend()

        focusDock = QDockWidget('Focus Lock', self)
        focusDock.setWidget(self.focusWidget)
        focusDock.setFeatures(QDockWidget.DockWidgetVerticalTitleBar | 
                                 QDockWidget.DockWidgetFloatable |
                                 QDockWidget.DockWidgetClosable)
        focusDock.setAllowedAreas(Qt.BottomDockWidgetArea)
        
        self.addDockWidget(Qt.BottomDockWidgetArea, focusDock)

        # sizes to fit my screen properly
        self.focusWidget.setMinimumSize(1100, 370)
        self.tcspcWidget.setMinimumSize(598, 370)
        self.scanWidget.setMinimumSize(598, 370)
        self.move(1, 1)
        
    def make_connection(self, backend):
        
        backend.xyzWorker.make_connection(self.focusWidget)
        backend.scanWorker.make_connection(self.scanWidget)
        backend.tcspcWorker.make_connection(self.tcspcWidget)
        
        backend.minfluxWorker.make_connection(self.minfluxWidget)
        backend.psfWorker.make_connection(self.psfWidget)

    def psf_measurement(self):

        self.psfWidget.show()
        
    def minflux_measurement(self):
        
        self.minfluxWidget.show()
        self.minfluxWidget.emit_filename()

    def closeEvent(self, *args, **kwargs):
        
        self.closeSignal.emit()
        time.sleep(1)
        
        focusThread.exit()
        tcspcWorkerThread.exit()
        scanThread.exit()
        minfluxThread.exit()
        super().closeEvent(*args, **kwargs)
        
        app.quit()        
        
class Backend(QtCore.QObject):
    
    askROIcenterSignal = pyqtSignal()
    moveToSignal = pyqtSignal(np.ndarray)
    tcspcStartSignal = pyqtSignal(str, int, int)
    xyzStartSignal = pyqtSignal()
    xyzEndSignal = pyqtSignal(str)
    xyMoveAndLockSignal = pyqtSignal(np.ndarray)
    
    def __init__(self, adw, ph, scmos, diodelaser, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.scanWorker = scan.Backend(adw, diodelaser)
        self.xyzWorker = focus.Backend(scmos, adw)
        self.tcspcWorker = tcspc.Backend(ph)
        
        self.minfluxWorker = minflux.Backend(adw)
        self.psfWorker = psf.Backend()
            
    def setup_minflux_connections(self):
        
        self.scanWorker.ROIcenterSignal.connect(self.minfluxWorker.get_ROI_center)
        
        self.minfluxWorker.tcspcPrepareSignal.connect(self.tcspcWorker.prepare_minflux)
        self.minfluxWorker.tcspcStartSignal.connect(self.xyzWorker.start_tracking_pattern) #Esto no ocurre! Falta decorador
        self.minfluxWorker.tcspcStartSignal.connect(self.tcspcWorker.measure_minflux)
        
        self.minfluxWorker.xyzStartSignal.connect(self.xyzWorker.get_lock_signal)
        
        self.minfluxWorker.moveToSignal.connect(self.xyzWorker.get_move_signal) #esta señal sólo se emite en modo: 'Predefined positions', que arranca el loop
        
        self.minfluxWorker.shutterSignal.connect(self.scanWorker.shutter_handler)
        self.minfluxWorker.shutterSignal.connect(self.xyzWorker.shutter_handler)
        
        self.tcspcWorker.tcspcDoneSignal.connect(self.minfluxWorker.get_tcspc_done_signal)
       
        self.minfluxWorker.saveConfigSignal.connect(self.scanWorker.saveConfigfile)
        self.minfluxWorker.xyzEndSignal.connect(self.xyzWorker.get_end_measurement_signal)
        # TODO: check before use
        self.minfluxWorker.xyStopSignal.connect(self.xyzWorker.get_stop_signal) #Esta señal jamás se emite desde minflux.py

    def setup_psf_connections(self):
        self.psfWorker.scanSignal.connect(self.scanWorker.get_scan_signal) #Esta es la conexion que permite cambiar el punto de inicio del escaneo
        self.psfWorker.xySignal.connect(self.xyzWorker.single_xy_correction)
        # self.psfWorker.zSignal.connect(self.xyzWorker.single_z_correction)
        self.psfWorker.xyStopSignal.connect(self.xyzWorker.get_stop_signal)
        # self.psfWorker.zStopSignal.connect(self.xyzWorker.get_stop_signal)
        self.psfWorker.moveToInitialSignal.connect(self.scanWorker.get_moveTo_initial_signal)
       
        self.psfWorker.shutterSignal.connect(self.scanWorker.shutter_handler)
        self.psfWorker.shutterSignal.connect(self.xyzWorker.shutter_handler)
                
        self.psfWorker.endSignal.connect(self.xyzWorker.get_end_measurement_signal)
        self.psfWorker.saveConfigSignal.connect(self.scanWorker.saveConfigfile)
        
        self.scanWorker.frameIsDone.connect(self.psfWorker.get_scan_is_done)
        self.xyzWorker.xyIsDone.connect(self.psfWorker.get_xy_is_done)
        #self.xyzWorker.zIsDone.connect(self.psfWorker.get_z_is_done)
         
    def make_connection(self, frontend):
        
        frontend.focusWidget.make_connection(self.xyzWorker)
        frontend.scanWidget.make_connection(self.scanWorker)
        frontend.tcspcWidget.make_connection(self.tcspcWorker)
    
        frontend.minfluxWidget.make_connection(self.minfluxWorker)
        frontend.psfWidget.make_connection(self.psfWorker)
    
        self.setup_minflux_connections()
        self.setup_psf_connections()
        
        frontend.scanWidget.paramSignal.connect(self.psfWorker.get_scan_parameters)
        # TO DO: write this in a cleaner way, i. e. not in this section, not using frontend
        
        self.scanWorker.focuslockpositionSignal.connect(self.xyzWorker.get_focuslockposition) #Signal & Slot connection Checked FC
        self.xyzWorker.focuslockpositionSignal.connect(self.scanWorker.get_focuslockposition) #Signal & Slot connection Checked FC
        #FC NOTE: Both scan & focus emit the same signal focuslockpositionSignal and both have the same function get_focuslockposition (but though they have the same name they do not do the same)
        frontend.closeSignal.connect(self.stop)

    def stop(self):
        
        self.scanWorker.stop()
        self.tcspcWorker.stop()
        self.xyzWorker.stop()


if __name__ == '__main__':

    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()
        
    #app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    gui = Frontend()
    
    #initialize devices
    
    port = tools.get_MiniLasEvoPort()
    print('MiniLasEvo diode laser port:', port)
    diodelaser = MiniLasEvo(port)
    
    #if camera wasnt closed properly just keep using it without opening new one
    try:
        cam = ids_cam.IDS_U3()
    except:
        pass
    
    ph = picoharp.PicoHarp300()
    
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)
    
    worker = Backend(adw, ph, cam, diodelaser)
    
    gui.make_connection(worker)
    worker.make_connection(gui)
    
    # initial parameters
    
    gui.scanWidget.emit_param()
    worker.scanWorker.emit_param()
    
    gui.minfluxWidget.emit_param()
#    gui.minfluxWidget.emit_param_to_backend()
#    worker.minfluxWorker.emit_param_to_frontend()
    
    gui.psfWidget.emit_param()
    
#    # GUI thread
#    
#    guiThread = QtCore.QThread()
#    gui.moveToThread(guiThread)
#    
#    guiThread.start()
    
    # psf thread
    
#    psfGUIThread = QtCore.QThread()
#    gui.psfWidget.moveToThread(psfGUIThread)
#    
#    psfGUIThread.start()
    
    # focus thread

    focusThread = QtCore.QThread()
    worker.xyzWorker.moveToThread(focusThread)
    worker.xyzWorker.viewtimer.moveToThread(focusThread)
    worker.xyzWorker.viewtimer.timeout.connect(worker.xyzWorker.update)

    focusThread.start()
    
    # focus GUI thread
    
#    focusGUIThread = QtCore.QThread()
#    gui.focusWidget.moveToThread(focusGUIThread)
#    
#    focusGUIThread.start()
       

    # tcspc thread
    
    tcspcWorkerThread = QtCore.QThread()
    worker.tcspcWorker.moveToThread(tcspcWorkerThread)
    worker.tcspcWorker.tcspcTimer.moveToThread(tcspcWorkerThread)
    worker.tcspcWorker.tcspcTimer.timeout.connect(worker.tcspcWorker.update)
    
    tcspcWorkerThread.start()
    
    # scan thread
    
    scanThread = QtCore.QThread()
    
    worker.scanWorker.moveToThread(scanThread)
    worker.scanWorker.viewtimer.moveToThread(scanThread)
    worker.scanWorker.viewtimer.timeout.connect(worker.scanWorker.update_view)

    scanThread.start()
    
    # minflux worker thread
    
    minfluxThread = QtCore.QThread()
    worker.minfluxWorker.moveToThread(minfluxThread)
    
    minfluxThread.start()
    
    # psf worker thread
    
#    psfThread = QtCore.QThread()
#    worker.psfWorker.moveToThread(psfThread)
#    worker.psfWorker.measTimer.moveToThread(psfThread)
#    worker.psfWorker.measTimer.timeout.connect(worker.psfWorker.measurement_loop)
#
#    psfThread.start()
    
    gui.showMaximized()
    app.exec_()
