# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:38:16 2019

@author: Luciano A. Masullo
"""

import numpy as np
import os
from datetime import date, datetime
import time

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as _pg

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QGroupBox
from tkinter import Tk, filedialog

import tools.tools as tools
import tools.PSF_tools as psft
import imageio as iio
from tools import customLog  # NOQA

import logging as _lgn

_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)

π = np.pi

DEBUG = True

# Placeholder para encontrar fácil
FIX_L = 100.  # en nm
FIX_K = 4
N_COLS = 2

class Frontend(QtGui.QFrame):
    paramSignal = pyqtSignal(dict)
    """
    Signals
    """
    _plots: list = [] #  [_pg.PlotItem] = []  # plots de las donas
    _imitems: list = []  # [_pg.ImageItem] = []  # plots de las donas
    _images: list = [None, ] * FIX_K  #list[np.ndarray] = [None, ] * FIX_K  # data
    _centerplots: list = [None, ] * FIX_K  # Plots de los centros reales
    _perfectplots: list = [None, ] * FIX_K  # Plots de los centros ideales
    _nframes = None
    _ndonuts = None
    _backend = None  # reference to backend

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_gui()

    def emit_param(self):
        filename = os.path.join(self.folderEdit.text(),
                                self.filenameEdit.text())
        params = dict()
        params['label'] = self.doughnutLabel.text()
        params['nframes'] = self.NframesEdit.value() #!!! now making up number of frames per doughnut position,
                                                     #no longer total number of images!!!
        params['filename'] = filename
        params['folder'] = self.folderEdit.text()
        params['nDonuts'] = self.donutSpinBox.value()
        params['alignMode'] = self.activateModeCheckbox.isChecked()
        self._nframes = self.NframesEdit.value()
        self._ndonuts = self.donutSpinBox.value()
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

    @pyqtSlot(float, np.ndarray, int)
    def get_progress_signal(self, completed: float, image: np.ndarray, order: int):
        """Actualiza el progreso.

        Enviado por el back. Aprovechamos para actualizar los plots. Se recibe con %
        de 0% al iniciar una medida y con 100% al terminarla. En ambos casos, image y
        order pueden ser none

        Parameters
        ----------
            completed: int
                Porcentaje de avance
            image: numpy:ndarray
                Imagen recibida (si hay) o None si sólo hay que actualizar el %
            order: int
                Numero de medida. Va de 0 a nFrames * nDonuts -1, o bien -1 si
                no hay medida
        """
        try:
            self.progressBar.setValue(completed)
            if completed == 100:
                self.stopButton.setEnabled(False)
                self.startButton.setEnabled(True)
                if order == -1:
                    # measurement finished: process file
                    time.sleep(1.5)
                    self.process_measurement()
            if not (self._ndonuts and self._nframes):
                _lgr.error("No estan seteados numeros necesarios")
                return
            if order >= 0:
                n_img = order % self._nframes
                n_donut = order // self._nframes
                _lgr.debug("Actualizando imagen %s/%s de la dona %s", n_img+1, self._nframes,
                           n_donut)
                # PLACE
                self.update_donut_image(n_donut, image)
        except Exception as e:
            _lgr.error("Excepcion en gps: %s", e)

    def process_measurement(self):
        """Update and analyse the graphs."""
        if not self._backend:
            _lgr.error("No reference to backend")
            raise ValueError("Backend not found")
        base_filename = self._backend.lastFileName
        if not base_filename:
            _lgr.error("The backend does not know the filename")
            raise ValueError("No filename")
        if self._backend.data is None or len(self._backend.data) != self._ndonuts * self._nframes:
            _lgr.warning("These seems to be no data on the backend")
            return
        # cargar configuracion
        conf = tools.loadConfig(base_filename + ".txt")
        if not conf:
            _lgr.warning("Config file not found")
            return
        scale = conf.getfloat('Scan range (µm)') * 1E3
        ps = conf.getfloat('Pixel size (µm)') * 1E3
        _lgr.info("Pixel size is %s", ps)
        if abs(self._backend.data.shape[1] * ps - scale) > 1E-3:
            _lgr.warning("Diferencias de escala de %s",
                         (self._backend.data.shape[1] * ps - scale))
        centros_OK = psft.centers_minflux(FIX_L, FIX_K)
        for nd in range(self._ndonuts):
            start = nd * self._nframes
            avg = np.average(self._backend.data[start: start + self._nframes], axis=0)
            img = self.update_donut_image(nd, avg)
            _lgr.info("Antes de update scale")
            self._update_image_scale(img, ps, *avg.shape)
            _lgr.info("Antes de find center")
            centro = self._find_center(nd, ps, centros_OK[nd])
            _lgr.error("La dona %s se encuentra en %s y debería estar en %s",
                       nd, centro, centros_OK[nd])
            print(f"Correr la dona {nd} en {centro-centros_OK[nd]}")
            # Marcar diferencias en nm

    def _update_image_scale(self, img: _pg.ImageItem, pixel_size: float,
                            extent_x: int, extent_y: int):
        """Update scale of an image.

        Parameters
        ----------
        pixel_size : float
            pixel size in nm
        extent_x, extent_y: int
            Size of the image in x and y (por ahora son siempre iguales)

        Returns
        -------
        None.
        """
        tr = QtGui.QTransform()
        img.setTransform(tr.scale(pixel_size, pixel_size).translate(-extent_x/2, -extent_y/2))

    def _find_center(self, n_donut: int, pixel_size: float, centro: np.ndarray):
        """Find the center of a donut and update the image.

        Parameters
        ----------
        n_donut: int
            number of donut to analyze
        pixel_size: float
            size of each pixel in nm
        centro: array
            centro ideal

        Returns
        -------
        Centro de la dona (x0, y0) en nm respecto a un 0 central
        """
        plot = self._plots[n_donut]
        image = self._images[n_donut]
        xc, yc = psft.find_min(image, trim=15)
        _lgr.error("Antes de centerplots")
        if self._centerplots[n_donut] is not None:
            try:
                plot.removeItem(self._centerplots[n_donut])
                plot.removeItem(self._perfectplots[n_donut])
            except Exception as e:
                _lgr.error("Error updating center %s: %s", n_donut, e)
        sp = plot.plot([xc], [yc], pen=(200, 200, 200), symbolBrush=(255, 0, 0),
                       symbolPen='w', )
        self._centerplots[n_donut] = sp
        _lgr.error("Antes de segundo plot")
        ideal_plot = plot.plot([centro[0]], [centro[1]], pen=(200, 200, 200), symbolBrush=(0, 255, 0),
                       symbolPen='w', )
        self._perfectplots[n_donut] = ideal_plot
        tr = QtGui.QTransform()
        shift = [-_/2 for _ in image.shape]
        sp.setTransform(tr.scale(pixel_size, pixel_size).translate(*shift))
        # ideal_plot.setTransform(tr.scale(pixel_size, pixel_size).translate(*shift))
        # TODO: Fix si se independizan las resoluciones x e y
        return (np.array((xc, yc,)) - np.array([-_/2 for _ in image.shape])) * pixel_size

    def activate_alignmentmode(self, on):
        if on:
            self.shutter1Checkbox.setEnabled(True)
            self.shutter2Checkbox.setEnabled(True)
            self.shutter3Checkbox.setEnabled(True)
            self.shutter4Checkbox.setEnabled(True)
            _lgr.info('Alignment mode activated')
        else:
            self.shutter1Checkbox.setEnabled(False)
            self.shutter2Checkbox.setEnabled(False)
            self.shutter3Checkbox.setEnabled(False)
            self.shutter4Checkbox.setEnabled(False)
            _lgr.info('Alignment mode deactivated')

    def setup_gui(self):
        self.setWindowTitle('PSF measurement')

        self.resize(800, 600)

        grid = QtGui.QGridLayout()

        self.setLayout(grid)
        self.paramWidget = QGroupBox('Parameter')
        self.paramWidget.setMinimumHeight(250)
        self.paramWidget.setFixedWidth(175)

        grid.addWidget(self.paramWidget, 0, 0, 2, 1)

        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        self.NframesLabel = QtGui.QLabel('Frames per doughnut')
        self.NframesEdit = QtGui.QSpinBox()
        self.DonutNumLabel = QtGui.QLabel('Number of doughnuts')
        self.donutSpinBox = QtGui.QSpinBox()
        self.doughnutLabel = QtGui.QLabel('Doughnut label')
        self.doughnutEdit = QtGui.QLineEdit('Black, Blue, Yellow, Orange')
        self.filenameLabel = QtGui.QLabel('File name')
        self.filenameEdit = QtGui.QLineEdit('psf')
        self.startButton = QtGui.QPushButton('Start')
        self.stopButton = QtGui.QPushButton('Stop')
        self.stopButton.setEnabled(False)
        self.progressBar = QtGui.QProgressBar(self)

        self.NframesEdit.setValue(5)
        self.NframesEdit.setRange(1, 99)
        self.donutSpinBox.setValue(4)
        self.donutSpinBox.setMaximum(10)

        subgrid.addWidget(self.DonutNumLabel, 0, 0)
        subgrid.addWidget(self.donutSpinBox, 1, 0)
        subgrid.addWidget(self.NframesLabel, 2, 0)
        subgrid.addWidget(self.NframesEdit, 3, 0)
        subgrid.addWidget(self.doughnutLabel, 4, 0)
        subgrid.addWidget(self.doughnutEdit, 5, 0)
        subgrid.addWidget(self.filenameLabel, 6, 0)
        subgrid.addWidget(self.filenameEdit, 7, 0)
        subgrid.addWidget(self.progressBar, 8, 0)
        subgrid.addWidget(self.startButton, 9, 0)
        subgrid.addWidget(self.stopButton, 10, 0)

        # file/folder widget

        self.fileWidget = QGroupBox('Save options')
        self.fileWidget.setFixedHeight(155)
        self.fileWidget.setFixedWidth(150)

        # folder

        # TO DO: move this to backend

        today = str(date.today()).replace('-', '')
        root = r'C:\\Data\\'
        folder = root + today
        self.initialDir = folder

        try:
            os.mkdir(folder)
        except OSError:
            _lgr.info('Directory %s already exists', folder)
        else:
            _lgr.info('Successfully created the directory %s', folder)

        self.folderLabel = QtGui.QLabel('Folder')
        self.folderEdit = QtGui.QLineEdit(folder)
        self.browseFolderButton = QtGui.QPushButton('Browse')
        self.browseFolderButton.setCheckable(True)

        grid.addWidget(self.fileWidget, 0, 1, 1, 1)

        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)

        file_subgrid.addWidget(self.filenameLabel, 0, 0, 1, 2)
        file_subgrid.addWidget(self.filenameEdit, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 2, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 4, 0)

        # setup alignment mode widget
        self.alignWidget = QGroupBox('Alignment mode')
        self.alignWidget.setFixedHeight(110)
        self.alignWidget.setFixedWidth(150)

        grid.addWidget(self.alignWidget, 1, 1, 1, 1)

        align_subgrid = QtGui.QGridLayout()
        self.alignWidget.setLayout(align_subgrid)

        self.activateModeCheckbox = QtGui.QCheckBox('Mode Activated')
        self.shutter1Checkbox = QtGui.QCheckBox('1')
        self.shutter2Checkbox = QtGui.QCheckBox('2')
        self.shutter3Checkbox = QtGui.QCheckBox('3')
        self.shutter4Checkbox = QtGui.QCheckBox('4')

        self.checkboxGroup = QtGui.QButtonGroup(self)
        self.checkboxGroup.addButton(self.shutter1Checkbox)
        self.checkboxGroup.addButton(self.shutter2Checkbox)
        self.checkboxGroup.addButton(self.shutter3Checkbox)
        self.checkboxGroup.addButton(self.shutter4Checkbox)

        self.shutter1Checkbox.setEnabled(False)
        self.shutter2Checkbox.setEnabled(False)
        self.shutter3Checkbox.setEnabled(False)
        self.shutter4Checkbox.setEnabled(False)

        align_subgrid.addWidget(self.activateModeCheckbox, 0, 0, 1, 2)
        align_subgrid.addWidget(self.shutter1Checkbox, 1, 0)
        align_subgrid.addWidget(self.shutter2Checkbox, 2, 0)
        align_subgrid.addWidget(self.shutter3Checkbox, 1, 1)
        align_subgrid.addWidget(self.shutter4Checkbox, 2, 1)

        # Donut show window (ver analysis.py)
        try:
            _lgr.debug("Iniciando ventana donas")
            self.imageWidget = _pg.GraphicsLayoutWidget()
            for i in range(FIX_K):  # Debería ser K
                p = self.imageWidget.addPlot(title=f"Dona {i+1}")
                self._plots.append(p)
                img = _pg.ImageItem()
                p.addItem(img)
                self._imitems.append(img)
                p.setLabels(bottom='x/nm', left='y/nm')
                if ((i+1) % N_COLS) == 0:  # es lo que hay
                    self.imageWidget.nextRow()
                # p.setAspectLocked(True)
            grid.addWidget(self.imageWidget, 0, 2, 3, -1)
        except Exception as e:
            print("Excepción", e)

        # connections
        self.startButton.clicked.connect(self.emit_param)
        self.startButton.clicked.connect(lambda: self.stopButton.setEnabled(True))
        self.startButton.clicked.connect(lambda: self.startButton.setEnabled(False))
        self.stopButton.clicked.connect(lambda: self.startButton.setEnabled(True))
        self.browseFolderButton.clicked.connect(self.load_folder)
        self.activateModeCheckbox.clicked.connect(lambda: self.activate_alignmentmode(
            self.activateModeCheckbox.isChecked()))

    def make_connection(self, backend):
        """Connect slots to backend and save reference."""
        backend.progressSignal.connect(self.get_progress_signal)
        self._backend = backend

    def closeEvent(self, *args, **kwargs):
        self.progressBar.setValue(0)
        super().closeEvent(*args, **kwargs)

    def update_donut_image(self, donut_number: int, image: np.ndarray):
        """Actualiza la la data e imagen de la dona.

        Zero-based indexing

        Returns
        -------
           Created pyqtgraph.ImageItem
        """
        if donut_number > len(self._plots):
            _lgr.error("Invalid donut number: %s", donut_number)
            return
        _lgr.info("Actualizando imagen %s con una de intensidad %s",
                  donut_number, image.sum())
        imitem = self._imitems[donut_number]
        imitem.setImage(image)
        self._images[donut_number] = image
        self._plots[donut_number].autoRange()
        _lgr.debug("Fin actualizacion imagen %s", donut_number)
        return imitem


class Backend(QtCore.QObject):
    # bool 1: whether you feedback ON or OFF, bool 2: initial position
    xySignal = pyqtSignal(bool, bool)
    xyStopSignal = pyqtSignal(bool)
    # zSignal = pyqtSignal(bool, bool)
    # zStopSignal = pyqtSignal(bool)  # Removed since now there is a single worker
    endSignal = pyqtSignal(str)
    scanSignal = pyqtSignal(bool, str, np.ndarray)
    moveToInitialSignal = pyqtSignal()
    progressSignal = pyqtSignal(float, np.ndarray, int)
    shutterSignal = pyqtSignal(int, bool)
    saveConfigSignal = pyqtSignal(str)
    """
    Signals
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.i = 0
        self.xyIsDone = False
        # self.zIsDone = False
        self.scanIsDone = False
        self.measTimer = QtCore.QTimer()
        self.measTimer.timeout.connect(self.loop)

        self.checkboxID_old = 7
        self.alignMode = False
        self.lastFileName = None  # las filename used for saving
        self.data: np.ndarray | None = None  # data

    def start(self):
        self.i = 0
        self.xyIsDone = False
        # self.zIsDone = False
        self.scanIsDone = False
        try:
            self.progressSignal.emit(0, np.array([]), -1)
            self.shutterSignal.emit(7, False)
            self.shutterSignal.emit(11, False)
    
            _lgr.info('PSF measurement started')
    
            self.xyStopSignal.emit(True)
            # self.zStopSignal.emit(True)
    
            # open IR and tracking shutter
            self.shutterSignal.emit(5, True)
            self.shutterSignal.emit(6, True)
            self.moveToInitialSignal.emit()
    
            self.data = np.zeros((self.totalFrameNum, self.nPixels, self.nPixels))
            _lgr.debug('Data shape is %s', np.shape(self.data))
            self.xy_flag = True
            self.z_flag = True
            self.scan_flag = True
        except Exception as e:
            print("Excepcion en start:", e)
        self.measTimer.start(0)

    def stop(self):
        self.measTimer.stop()
        self.progressSignal.emit(100, np.array([]), -1)  # changed from 0
        self.shutterSignal.emit(8, False)

        # new filename indicating that getUniqueName() has already found filename
        # rerunning would only cause errors in files being saved by focus and xy_tracking
        attention_filename = '!' + self.filename
        self.endSignal.emit(attention_filename)

        self.xyStopSignal.emit(False)
        # self.zStopSignal.emit(False)

        self.export_data()
        self.progressSignal.emit(100, np.array([]), -1)  # changed from 0
        _lgr.debug('PSF measurement ended')

    def loop(self):
        """Check status on each tick.

        This would be best implemented using a state machine and no timer.
        """
        if self.i == 0:
            initial = True
        else:
            initial = False

        if self.xy_flag:
            self.xySignal.emit(True, initial)
            self.xy_flag = False
            _lgr.debug(' xy signal emitted (%s)', self.i)
        if self.xyIsDone:
            # if self.z_flag:
            #     self.zSignal.emit(True, initial)
            #     self.z_flag = False
            #     _lgr.debug('z signal emitted (%s)', self.i)

            # if self.zIsDone:
            if True:  # ahora la estabilización en Z es contiunua
                shutternum = self.i // self.nFrames + 1
                if self.scan_flag:
                    if not self.alignMode:
                        self.shutterSignal.emit(shutternum, True)
                        time.sleep(0.100)  # let the shutter move
                    initialPos = np.array([self.target_x, self.target_y,
                                           self.target_z], dtype=np.float64)
                    self.scanSignal.emit(True, 'frame', initialPos)
                    self.scan_flag = False
                    _lgr.debug('scan signal emitted (%s)', self.i)

                if self.scanIsDone:
                    if not self.alignMode:
                        self.shutterSignal.emit(shutternum, False)
                    completed = ((self.i+1)/self.totalFrameNum) * 100

                    self.xy_flag = True
                    self.z_flag = True
                    self.scan_flag = True
                    self.xyIsDone = False
                    # self.zIsDone = False
                    self.scanIsDone = False

                    self.data[self.i, :, :] = self.currentFrame
                    self.progressSignal.emit(completed, self.data[self.i, :, :], self.i)
                    _lgr.debug('PSF %s of %s', self.i+1, self.totalFrameNum)

                    if self.i < self.totalFrameNum-1:
                        self.i += 1
                    else:
                        self.stop()  # Incluye un progressSignal

    def export_data(self):
        fname = self.filename
        filename = tools.getUniqueName(fname)
        np.save(filename, self.data)
        self.data = np.array(self.data, dtype=np.float32)
        iio.mimwrite(filename + '.tiff', self.data)
        # make scan saving config file
        self.lastFileName = filename
        self.saveConfigSignal.emit(filename)

    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        self.label = params['label']
        self.nFrames = params['nframes']
        self.k = params['nDonuts']

        today = str(date.today()).replace('-', '')
        self.filename = tools.getUniqueName(params['filename'] + '_' + today)

        self.totalFrameNum = self.nFrames * self.k

        self.alignMode = params['alignMode']

    @pyqtSlot(bool, float, float, float)
    def get_xy_is_done(self, val, x, y, z):
        """
        Connection: [xy_tracking] xyIsDone
        """
        self.xyIsDone = True
        self.target_x = x
        self.target_y = y
        self.target_z = z

    # @pyqtSlot(bool, float)
    # def get_z_is_done(self, val, z):
    #     """
    #     Connection: [focus] zIsDone
    #     """
    #     self.zIsDone = True
    #     self.target_z = z

    @pyqtSlot(bool, np.ndarray)
    def get_scan_is_done(self, val, image):
        """
        Connection: [scan] scanIsDone
        """
        self.scanIsDone = True
        self.currentFrame = image

    @pyqtSlot(dict)
    def get_scan_parameters(self, params):
        # TO DO: this function is connected to the scan frontend, it should
        # be connected to a proper funciton in the scan backend
        self.nPixels = int(params['NofPixels'])
        # TO DO: build config file

    # button_clicked slot
    @pyqtSlot(QtGui.QAbstractButton)
    # @pyqtSlot(int)
    def checkboxGroup_selection(self, button_or_id):
        self.shutterSignal.emit(self.checkboxID_old, False)
        # if isinstance(button_or_id, QtGui.QAbstractButton):
        checkboxID = int(button_or_id.text())
        # print('Checkbox {} was selected'.format(checkboxID))
        self.shutterSignal.emit(checkboxID, True)

        self.checkboxID_old = checkboxID
        # elif isinstance(button_or_id, int):
        # print('"Id {}" was clicked'.format(button_or_id))

    def make_connection(self, frontend):
        frontend.startButton.clicked.connect(self.start)
        frontend.stopButton.clicked.connect(self.stop)
        frontend.paramSignal.connect(self.get_frontend_param)
        frontend.checkboxGroup.buttonClicked['QAbstractButton *'].connect(
            self.checkboxGroup_selection)
