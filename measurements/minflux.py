# -*- coding: utf-8 -*-
"""
Inicia medidas de minflux con opcion a realizar patrones
"""

import numpy as np
import time
import os
from datetime import date, datetime

from pyqtgraph.Qt import QtCore, QtGui

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QGroupBox
from tkinter import Tk, filedialog

import tools.tools as tools
from swabian.backend import TCSPC_backend


class Frontend(QtGui.QFrame):
    filenameSignal = pyqtSignal(str)
    paramSignal = pyqtSignal(dict)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_gui()

    def emit_filename(self):
        """Emit filename.

        Es llamado por el front al abrir la ventana
        """
        filename = os.path.join(self.folderEdit.text(),
                                self.filenameEdit.text())
        today = str(date.today()).replace('-', '')
        filename = tools.getUniqueName(filename + '_' + today)
        self.filenameSignal.emit(filename)

    def emit_param(self):
        params = dict()
        filename = os.path.join(self.folderEdit.text(),
                                self.filenameEdit.text())
        params['measType'] = self.measType.currentText()
        params['acqtime'] = int(self.acqtimeEdit.text())
        params['filename'] = filename
        params['patternType'] = self.patternType.currentText()
        params['patternLength'] = float(self.lengthEdit.text())
        self.paramSignal.emit(params)

    def load_folder(self):
        try:
            root = Tk()
            root.withdraw()
            folder = filedialog.askdirectory(parent=root,
                                             initialdir=self.initialDir)
            root.destroy()
            if folder != '':
                self.folderEdit.setText(folder)
        except OSError:
            pass

    def toggle_parameters(self):
        if self.measType.currentText() == 'Predefined positions':
            self.patternType.show()
            self.lengthLabel.show()
            self.lengthEdit.show()
        else:
            self.patternType.hide()
            self.lengthLabel.hide()
            self.lengthEdit.hide()

    def setup_gui(self):
        self.setWindowTitle('MINFLUX measurement')
        grid = QtGui.QGridLayout()

        self.setLayout(grid)
        self.paramWidget = QGroupBox('Parameter')

        grid.addWidget(self.paramWidget, 0, 0)

        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        self.measLabel = QtGui.QLabel('Measurement type')

        self.measType = QtGui.QComboBox()
        self.measTypes = ['Standard', 'Predefined positions']
        self.measType.addItems(self.measTypes)

        self.patternType = QtGui.QComboBox()
        self.patternTypes = ['Row', 'Square', 'Triangle']
        self.patternType.addItems(self.patternTypes)

        self.lengthLabel = QtGui.QLabel('L [nm]')
        self.lengthEdit = QtGui.QLineEdit('30')

        self.patternType.hide()
        self.lengthLabel.hide()
        self.lengthEdit.hide()

        self.acqtimeLabel = QtGui.QLabel('Acq time [s]')
        self.acqtimeEdit = QtGui.QLineEdit('5')

        self.startButton = QtGui.QPushButton('Start')
        self.stopButton = QtGui.QPushButton('Stop')

        subgrid.addWidget(self.measLabel, 0, 0, 1, 2)
        subgrid.addWidget(self.measType, 1, 0, 1, 2)

        subgrid.addWidget(self.patternType, 2, 0, 1, 2)

        subgrid.addWidget(self.lengthLabel, 4, 0, 1, 1)
        subgrid.addWidget(self.lengthEdit, 4, 1, 1, 1)

        subgrid.addWidget(self.acqtimeLabel, 6, 0, 1, 1)
        subgrid.addWidget(self.acqtimeEdit, 6, 1, 1, 1)
        subgrid.addWidget(self.startButton, 7, 0, 1, 2)
        subgrid.addWidget(self.stopButton, 8, 0, 1, 2)

        # file/folder widget

        self.fileWidget = QGroupBox('Save options')
        self.fileWidget.setFixedHeight(155)
        self.fileWidget.setFixedWidth(150)

        # folder
        today = str(date.today()).replace('-', '')
        root = r'C:\\Data\\'
        folder = root + today

        try:
            os.mkdir(folder)
        except OSError:
            print(datetime.now(), '[minflux] Directory {} already exists'.format(folder))
        else:
            print(datetime.now(), '[minflux] Successfully created the directory {}'.format(folder))

        self.folderLabel = QtGui.QLabel('Folder')
        self.folderEdit = QtGui.QLineEdit(folder)
        self.browseFolderButton = QtGui.QPushButton('Browse')
        self.browseFolderButton.setCheckable(True)
        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('minflux')

        grid.addWidget(self.fileWidget, 0, 1)

        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)

        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)

        self.measType.currentIndexChanged.connect(self.toggle_parameters)

        self.folderEdit.textChanged.connect(self.emit_param)
        self.acqtimeEdit.textChanged.connect(self.emit_param)
        self.lengthEdit.textChanged.connect(self.emit_param)
        self.patternType.activated.connect(self.emit_param)

    def make_connection(self, backend):
        pass


class Backend(QtCore.QObject):

    tcspcPrepareSignal = pyqtSignal(str, int, int)
    tcspcStartSignal = pyqtSignal(str)

    xyzEndSignal = pyqtSignal(str)
    moveToSignal = pyqtSignal(np.ndarray)

    # paramSignal = pyqtSignal(np.ndarray, np.ndarray, int)
    shutterSignal = pyqtSignal(int, bool)

    saveConfigSignal = pyqtSignal(str)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.i = 0  # counter
        self.n = 1  # numero de posiciones en la figura
        self.pattern = np.array([0, 0])

        self.measTimer = QtCore.QTimer()
        self.measTimer.timeout.connect(self.loop)
        TCSPC_backend.sgnl_measure_end.connect(self.get_tcspc_done_signal)

    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        """
        Connection: [frontend] paramSignal
        """
        self.acqtime = params['acqtime']
        self.measType = params['measType']
        today = str(date.today()).replace('-', '')
        self.filename = params['filename'] + '_' + today
        self.patternType = params['patternType']
        self.patternLength = float(params['patternLength'])
        self.update_param()

    def update_param(self):
        l = self.patternLength
        h = np.sqrt(3/2)*l
        if self.measType == 'Predefined positions':
            if self.patternType == 'Row':
                self.pattern = np.array([[0, -l], [0, 0], [0, l]])
            if self.patternType == 'Square':
                self.pattern = np.array([[l/2, l/2], [l/2, -l/2],
                                        [-l/2, -l/2], [-l/2, l/2]])
            if self.patternType == 'Triangle':
                self.pattern = np.array([[0, (2/3)*h], [l/2, -(1/3)*h],
                                        [-l/2, -(1/3)*h]])
            self.n = np.shape(self.pattern)[0]
        else:
            self.pattern = np.array((0, 0,))
            self.n = 0

    def start(self):
        self.i = 0
        self.shutterSignal.emit(8, True)
        self.update_param()
        time.sleep(0.2)
        self.t0 = time.time()
        self.measTimer.start(self.acqtime * 1E3)  # resolucion pedorra
        TCSPC_backend.start_measure(self.filename)

    def loop(self):
        print(datetime.now(), '[minflux] fin loop', self.i)
        if self.i >= self.n:
            self.stop()
            print(datetime.now(), '[minflux] measurement ended')
        else:
            print("Moviendo para ", self.pattern[self.i])
            self.moveToSignal.emit(self.pattern[self.i])
            self.i += 1

    def stop(self):
        self.measTimer.stop()
        self.i = 1E8  # Flag para que el loop no ejecute si quedo una señal en cola
        TCSPC_backend.stop_measure()
        self.shutterSignal.emit(8, False)
        self.moveToSignal.emit(np.zeros((2,)))

    @pyqtSlot()
    def get_tcspc_done_signal(self):
        """
        Connection: [tcspc] tcspcDoneSignal
        """
        # make scan saving config file
        self.saveConfigSignal.emit(self.filename)
        self.xyzEndSignal.emit(self.filename)
        print(datetime.now(), '[minflux] measurement ended')

    def make_connection(self, frontend):
        frontend.paramSignal.connect(self.get_frontend_param)
#        frontend.filenameSignal.connect(self.get_filename)
        frontend.startButton.clicked.connect(self.start)
        frontend.stopButton.clicked.connect(self.stop)
