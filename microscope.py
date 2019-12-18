# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:18:19 2018

@author: Luciano A. Masullo
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
import lantz.drivers.andor.ccd as ccd
import drivers.picoharp as picoharp

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from tkinter import Tk, filedialog

import drivers.ADwin as ADwin

import focus
import scan
import tcspc
import xy_tracking
import measurements.minflux as minflux
import measurements.psf as psf
import measurements.chechu as chechu

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
        self.chechuWidget = chechu.Frontend()

        self.psfMeasAction = QtGui.QAction('PSF measurement', self)
        self.psfMeasAction.setStatusTip('Routine to measure one MINFLUX PSF')
        fileMenu.addAction(self.psfMeasAction)
        
        self.psfMeasAction.triggered.connect(self.psf_measurement)
    
        self.minfluxMeasAction = QtGui.QAction('MINFLUX measurement', self)
        self.minfluxMeasAction.setStatusTip('Routine to perform a tcspc-MINFLUX measurement')
        fileMenu.addAction(self.minfluxMeasAction)
        
        self.minfluxMeasAction.triggered.connect(self.minflux_measurement)
        
        self.chechuMeasAction = QtGui.QAction('CHECHU measurement', self)
        self.chechuMeasAction.setStatusTip('Routine to measure CHECHU')
        fileMenu.addAction(self.chechuMeasAction)
        
        self.chechuMeasAction.triggered.connect(self.chechu_measurement)

        # GUI layout

        grid = QtGui.QGridLayout()
        self.cwidget.setLayout(grid)

        # Dock Area

        dockArea = DockArea()
        grid.addWidget(dockArea, 0, 0)

        ## scanner

        scanDock = Dock('scan', size=(1, 1))

        self.scanWidget = scan.Frontend()

        scanDock.addWidget(self.scanWidget)
        dockArea.addDock(scanDock, 'left')

        ## tcspc

        tcspcDock = Dock("Time-correlated single-photon counting")

        self.tcspcWidget = tcspc.Frontend()

        tcspcDock.addWidget(self.tcspcWidget)
        dockArea.addDock(tcspcDock, 'bottom', scanDock)

        ## focus lock

        focusDock = Dock("Focus Lock")

        self.focusWidget = focus.Frontend()

        focusDock.addWidget(self.focusWidget)
        dockArea.addDock(focusDock, 'right')

        ## xy tracking

        xyDock = Dock("xy drift control")

        self.xyWidget = xy_tracking.Frontend()

        xyDock.addWidget(self.xyWidget)
        dockArea.addDock(xyDock, 'top', focusDock)
#        dockArea.addDock(xyDock, 'top')

        # sizes to fit my screen properly

        self.scanWidget.setMinimumSize(1000, 550)
        self.xyWidget.setMinimumSize(800, 300)
        self.move(1, 1)
        
    def make_connection(self, backend):
        
        backend.zWorker.make_connection(self.focusWidget)
        backend.scanWorker.make_connection(self.scanWidget)
        backend.tcspcWorker.make_connection(self.tcspcWidget)
        backend.xyWorker.make_connection(self.xyWidget)
        
        backend.minfluxWorker.make_connection(self.minfluxWidget)
        backend.psfWorker.make_connection(self.psfWidget)
        backend.chechuWorker.make_connection(self.chechuWidget)

    def psf_measurement(self):

        self.psfWidget.show()
        
    def chechu_measurement(self):
        
        self.chechuWidget.show()
        
    def minflux_measurement(self):
        
        self.minfluxWidget.show()
        self.minfluxWidget.emit_filename()

    def closeEvent(self, *args, **kwargs):
        
        self.closeSignal.emit()

        super().closeEvent(*args, **kwargs)
        
        app.quit()        
        
class Backend(QtCore.QObject):
    
    askROIcenterSignal = pyqtSignal()
    moveToSignal = pyqtSignal(np.ndarray)
    tcspcStartSignal = pyqtSignal(str, int, int)
    xyzStartSignal = pyqtSignal()
    xyzEndSignal = pyqtSignal(str)
    xyMoveAndLockSignal = pyqtSignal(np.ndarray)
    
    def __init__(self, adw, ph, ccd, scmos, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.scanWorker = scan.Backend(adw)
        self.zWorker = focus.Backend(scmos, adw)
        self.tcspcWorker = tcspc.Backend(ph, adw)
        self.xyWorker = xy_tracking.Backend(ccd, adw)
        
        self.minfluxWorker = minflux.Backend(adw)
        self.psfWorker = psf.Backend()
        self.chechuWorker = chechu.Backend()
            
    def setup_minflux_connections(self):
        
        self.scanWorker.ROIcenterSignal.connect(self.minfluxWorker.get_ROI_center)
        
        self.minfluxWorker.tcspcPrepareSignal.connect(self.tcspcWorker.prepare_minflux)
        self.minfluxWorker.tcspcStartSignal.connect(self.tcspcWorker.measure_minflux)
        
        self.minfluxWorker.xyzStartSignal.connect(self.xyWorker.get_lock_signal)
        self.minfluxWorker.xyzStartSignal.connect(self.zWorker.get_lock_signal)
        
        # TO DO: check if this is compatible with both psf and minflux measurement
        
        self.minfluxWorker.moveToSignal.connect(self.xyWorker.get_move_signal)
        
        self.minfluxWorker.shutterSignal.connect(self.scanWorker.shutter_handler)
        self.minfluxWorker.shutterSignal.connect(self.xyWorker.toggle_tracking_shutter)
        
        self.tcspcWorker.tcspcDoneSignal.connect(self.minfluxWorker.get_tcspc_done_signal)
        
        self.minfluxWorker.xyzEndSignal.connect(self.xyWorker.get_end_measurement_signal)
        self.minfluxWorker.xyzEndSignal.connect(self.zWorker.get_end_measurement_signal)
        
    def setup_psf_connections(self):
        
        self.psfWorker.scanSignal.connect(self.scanWorker.get_scan_signal)
        self.psfWorker.xySignal.connect(self.xyWorker.single_xy_correction)
        self.psfWorker.zSignal.connect(self.zWorker.single_z_correction)
        self.psfWorker.xyStopSignal.connect(self.xyWorker.get_stop_signal)
        self.psfWorker.zStopSignal.connect(self.zWorker.get_stop_signal)
        self.psfWorker.moveToInitialSignal.connect(self.scanWorker.get_moveTo_initial_signal)
        self.psfWorker.shutterSignal.connect(self.scanWorker.shutter_handler)
                
        self.psfWorker.endSignal.connect(self.xyWorker.get_end_measurement_signal)
        self.psfWorker.endSignal.connect(self.zWorker.get_end_measurement_signal)
        
        self.scanWorker.frameIsDone.connect(self.psfWorker.get_scan_is_done)
        self.xyWorker.xyIsDone.connect(self.psfWorker.get_xy_is_done)
        self.zWorker.zIsDone.connect(self.psfWorker.get_z_is_done)
        
        
        
    def setup_chechu_connections(self):
        
        self.chechuWorker.scanSignal.connect(self.scanWorker.get_scan_signal_chechu)
#        self.chechuWorker.xySignal.connect(self.xyWorker.single_xy_correction)
        self.chechuWorker.zSignal.connect(self.zWorker.single_z_correction)
#        self.chechuWorker.xyStopSignal.connect(self.xyWorker.get_stop_signal)
        self.chechuWorker.zStopSignal.connect(self.zWorker.get_stop_signal)
        self.chechuWorker.moveToInitialSignal.connect(self.scanWorker.get_moveTo_initial_signal)
        
#        self.chechuWorker.endSignal.connect(self.xyWorker.get_end_measurement_signal)
        self.chechuWorker.endSignal.connect(self.zWorker.get_end_measurement_signal)
        
        self.scanWorker.frameIsDoneChechu.connect(self.chechuWorker.get_scan_is_done)
#        self.xyWorker.xyIsDone.connect(self.chechuWorker.get_xy_is_done)
        self.zWorker.zIsDone.connect(self.chechuWorker.get_z_is_done)
        
        
        self.chechuWorker.tcspcPrepareSignal.connect(self.tcspcWorker.prepare_chechu)
        self.chechuWorker.tcspcStartSignal.connect(self.tcspcWorker.measure_chechu)
        
    def make_connection(self, frontend):
        
        frontend.focusWidget.make_connection(self.zWorker)
        frontend.scanWidget.make_connection(self.scanWorker)
        frontend.tcspcWidget.make_connection(self.tcspcWorker)
        frontend.xyWidget.make_connection(self.xyWorker)
    
        frontend.minfluxWidget.make_connection(self.minfluxWorker)
        frontend.psfWidget.make_connection(self.psfWorker)
        frontend.chechuWidget.make_connection(self.chechuWorker)
    
        self.setup_minflux_connections()
        self.setup_psf_connections()
        self.setup_chechu_connections()
        
        frontend.scanWidget.paramSignal.connect(self.psfWorker.get_scan_parameters)
        frontend.scanWidget.paramSignal.connect(self.chechuWorker.get_scan_parameters)
        # TO DO: write this in a cleaner way, i. e. not in this section, not using frontend
        
        frontend.closeSignal.connect(self.stop)
        
    def stop(self):
        
        self.scanWorker.stop()
        self.zWorker.stop()
        self.tcspcWorker.stop()
        self.xyWorker.stop()

if __name__ == '__main__':

    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication(sys.argv)
    else:
        print('QApplication instance already exists: %s' % str(app))
        
    #app = QtGui.QApplication([])
    app.setStyle(QtGui.QStyleFactory.create('fusion'))
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    gui = Frontend()
    
    cam = uc480.UC480_Camera()
    andor = ccd.CCD()
    ph = picoharp.PicoHarp300()
    
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)
    
    worker = Backend(adw, ph, andor, cam)
    
    gui.make_connection(worker)
    worker.make_connection(gui)
    
    # initial parameters
    
    gui.scanWidget.emit_param()
    worker.scanWorker.emit_param()
    
    gui.minfluxWidget.emit_param()
#    gui.minfluxWidget.emit_param_to_backend()
#    worker.minfluxWorker.emit_param_to_frontend()
    
    gui.psfWidget.emit_param()
    
    gui.chechuWidget.emit_param()
    
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
    worker.zWorker.moveToThread(focusThread)
    worker.zWorker.focusTimer.moveToThread(focusThread)
    worker.zWorker.focusTimer.timeout.connect(worker.zWorker.update)

    focusThread.start()
    
    # focus GUI thread
    
#    focusGUIThread = QtCore.QThread()
#    gui.focusWidget.moveToThread(focusGUIThread)
#    
#    focusGUIThread.start()
    
#    # xy worker thread
#    
    xyThread = QtCore.QThread()
    worker.xyWorker.moveToThread(xyThread)
    worker.xyWorker.viewtimer.moveToThread(xyThread)
#    worker.xyWorker.viewtimer.timeout.connect(worker.xyWorker.update_view)
#    
    xyThread.start()
    
    # xy GUI thread
    
#    xyGUIThread = QtCore.QThread()
#    gui.xyWidget.moveToThread(xyGUIThread)
#    
#    xyGUIThread.start()

    # tcspc thread
    
    tcspcWorkerThread = QtCore.QThread()
    worker.tcspcWorker.moveToThread(tcspcWorkerThread)
    
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
    
    # minflux measurement connections
    
    gui.show()
    app.exec_()
