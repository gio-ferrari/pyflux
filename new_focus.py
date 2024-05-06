# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2018

@authors: Luciano Masullo modified by Flor C. to use another ROI and IDS cam
"""
import numpy as np
import time

import scipy.ndimage as ndi
from datetime import date, datetime
import os

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.ptime as ptime
import qdarkstyle  # see https://stackoverflow.com/questions/48256772/dark-theme-for-in-qt-widgets

from PyQt5.QtCore import pyqtSignal, pyqtSlot
# from PyQt5.QtWidgets import QGroupBox
# import sys
# sys.path.append('C:\Program Files\Thorlabs\Scientific Imaging\ThorCam')
# install from https://instrumental-lib.readthedocs.io/en/stable/install.html
import tools.viewbox_tools as viewbox_tools
import tools.tools as tools

import scan
import drivers.ADwin as ADwin
import drivers.ids_cam as ids_cam

# ids_peak.Library.Initialize()

DEBUG = False
START_PIXEL_SIZE = 20  # el fit exacto es 13 y algo

def actuatorParameters(adwin, z_f, n_pixels_z=50, pixeltime=1000):
    if DEBUG:
        print("Inside actuatorParameters")
    z_f = tools.convert(z_f, 'XtoU')

    # adwin.Set_Par(33, n_pixels_z)
    adwin.Set_FPar(32, z_f)
    adwin.Set_FPar(36, tools.timeToADwin(pixeltime))

# Creo que se puede borrar
def zMoveTo(adwin, z_f):
    if DEBUG:
        print("Inside zMoveto")
    actuatorParameters(adwin, z_f)
    adwin.Start_Process(3)

class Frontend(QtGui.QFrame):
    changedROI = pyqtSignal(np.ndarray)  # sends new roi size
    closeSignal = pyqtSignal()
    saveDataSignal = pyqtSignal(bool)

    paramSignal = pyqtSignal(dict)

    """
    Signals

    - changedROI: #This is the signal called roiInfoSignal in xy_tracking_2
        To: [backend] get_new_roi #Named get_roi_info in xy_tracking_2

    - closeSignal:
        To: [backend] stop

    - saveDataSignal:
        To: [backend] get_save_data_state

    - paramSignal:
        To: [backend] get_frontend_param
    """
    def __init__(self, *args, **kwargs):
        if DEBUG:
            print("Inside init")

        super().__init__(*args, **kwargs)

        self.cropped = False
        self.roi = None
        self.setup_gui()

        x0 = 0
        y0 = 0
        x1 = 1280
        y1 = 1024

        value = np.array([x0, y0, x1, y1])
        self.changedROI.emit(value)  # Al PP porque el backend aun no esta

    def emit_param(self):
        """Le manda los parámetros al back."""
        if DEBUG:
            print("Inside emit_param")
        params = dict()  # Se crea diccionario vacío FC
        params['pxSize'] = float(self.pxSizeEdit.text())

        self.paramSignal.emit(params)

    def roi_method(self):
        """Crear el ROI de z."""
        if DEBUG:
            print("Inside roi_method")

        ROIpen = pg.mkPen(color='y')
        ROIpos = (512 -64, 512 -64) #cambio FC #Debería colocar las nuevas dimensiones? 1200,1920
        self.roi = viewbox_tools.ROI(140, self.vb, ROIpos,
                                     handlePos=(1, 0),
                                     handleCenter=(0, 1),
                                     scaleSnap=True,
                                     translateSnap=True,
                                     pen=ROIpen)
        self.update_ROI_buttons(True)

    def update_ROI_buttons(self, roi_exists: bool):
        """Activar o desactivar los botones segun si hay o no un ROI."""
        self.ROIbutton.setEnabled(not roi_exists)
        self.selectROIbutton.setEnabled(roi_exists)
        self.deleteROIbutton.setEnabled(roi_exists)

    def select_roi(self): #Analogo a emit_roi_info #Esta señal va al backend a get_new_roi
        """Marca al ROI elegido como válido y lo pasa al backend."""
        if DEBUG:
            print("Inside select_roi")
        self.getStats = True
        xmin, ymin = self.roi.pos()
        xmax, ymax = self.roi.pos() + self.roi.size()
        value = np.array([xmin, xmax, ymin, ymax])
        print("Coordinates of the selected roi: ", value)
        self.changedROI.emit(value)

#    def toggleFocus(self):
#        if self.lockButton.isChecked():
#            self.lockFocusSignal.emit(True)
#            # self.setpointLine = self.focusGraph.zPlot.addLine(y=self.setPoint, pen='r')
#        else:
#            self.lockFocusSignal.emit(False)

#    def delete_roi(self):
#        self.vb.removeItem(self.roi)
#        x0 = 0
#        y0 = 0
#        x1 = 1280
#        y1 = 1024
#
#        value = np.array([x0, y0, x1, y1])
#        self.changedROI.emit(value)
#        self.cropped = False
#
#        self.roi = None
#
#        print(datetime.now(), '[focus] ROI deleted')
#
#        self.deleteROIbutton.setEnabled(False)

    def delete_roi(self):
        self.vb.removeItem(self.roi)
        self.roi.hide()
        del self.roi
        self.roi = None
        self.ROIbutton.setEnabled(True)
        self.selectROIbutton.setEnabled(False)
        self.deleteROIbutton.setEnabled(False)
        print("Borramos ROI")

    @pyqtSlot(bool)
    def toggle_liveview(self, on):
        if DEBUG:
            print("Inside toggle_liveview")
        if on:
            self.liveviewButton.setChecked(True)
            print(datetime.now(), '[focus] focus live view started')
        else:
            self.liveviewButton.setChecked(False)
            self.img.setImage(np.zeros((1200, 1920)), autoLevels=False)
            print(datetime.now(), '[focus] focus live view stopped')

    def emit_save_data_state(self):
        if DEBUG:
            print("Inside emit_save_data_state")
        if self.saveDataBox.isChecked():
            self.saveDataSignal.emit(True)
        else:
            self.saveDataSignal.emit(False)

    def toggle_stats(self):
        if self.feedbackLoopBox.isChecked():
            self.focusMean = self.focusGraph.plot.addLine(y=self.setPoint,
                                                          pen='c')
        else:
            self.focusGraph.removeItem(self.focusMean)

    @pyqtSlot(np.ndarray)
    def get_image(self, img: np.ndarray):
        """Cambia la imagen recibida desde el backend."""
        if DEBUG:
            print(" Inside get_image ")
        self.img.setImage(img, autoLevels=False)

    @pyqtSlot(np.ndarray, np.ndarray)
    def get_data(self, time: np.ndarray, position: np.ndarray):
        """Actualiza el grafico."""
        if DEBUG:
            print("Inside get_data ")
        self.focusCurve.setData(time, position)
        # if self.feedbackLoopBox.isChecked():
        #     if DEBUG:
        #         print("self.feedbackLoopBox.isChecked() in get_data")
        #     if len(position) > 2:
        #         # print("len(position) is higher than 2")
        #         zMean = np.mean(position)
        #         zStDev = np.std(position)
        #         # TO DO: fix focus stats
        #         self.focusMean.setValue(zMean)
        #         self.focusStDev0.setValue(zMean - zStDev)
        #         self.focusStDev1.setValue(zMean + zStDev)

    @pyqtSlot(float)
    def get_setpoint(self, value):
        """Establece el setpoint Z desde el backend."""
        if DEBUG:
            print("Inside get_setpoint")
        self.setPoint = value
        print(datetime.now(), '[focus] set point', value, "nm")

        # TO DO: fix setpoint line
#        self.focusSetPoint = self.focusGraph.zPlot.addLine(y=self.setPoint,
#                                                           pen=pg.mkPen('r', width=2))
#        self.focusMean = self.focusGraph.zPlot.addLine(y=self.setPoint,
#                                                       pen='c')
#        self.focusStDev0 = self.focusGraph.zPlot.addLine(y=self.setPoint,
#                                                         pen='c')
#        
#        self.focusStDev1 = self.focusGraph.zPlot.addLine(y=self.setPoint,
#                                                         pen='c')

    def clear_graph(self):
        if DEBUG:
            print("Inside clear_graph")
        self.focusCurve.setData([], [])  # TODO: check
        # TO DO: fix setpoint line
#        self.focusGraph.zPlot.removeItem(self.focusSetPoint)
#        self.focusGraph.zPlot.removeItem(self.focusMean)
#        self.focusGraph.zPlot.removeItem(self.focusStDev0)
#        self.focusGraph.zPlot.removeItem(self.focusStDev1)

    @pyqtSlot(int, bool)
    def update_shutter(self, num: int, on: bool):
        """Update shutters status.

        setting of num-value:
            0 - signal send by scan-gui-button --> change state of all minflux shutters
            1...6 - shutter 1-6 will be set according to on-variable, i.e. either true or false; only 1-4 controlled from here
            7 - set all minflux shutters according to on-variable
            8 - set all shutters according to on-variable
        for handling of shutters 1-5 see [scan] and [focus]
        """
        if DEBUG:
            print("Inside update_shutter")
        if (num == 5) or (num == 8):
            self.shutterCheckbox.setChecked(on)

    def make_connection(self, backend):
        if DEBUG:
            print("Inside make_connection")
        backend.changedImage.connect(self.get_image)
        backend.changedData.connect(self.get_data)
        backend.changedSetPoint.connect(self.get_setpoint)
        backend.shuttermodeSignal.connect(self.update_shutter)
        backend.liveviewSignal.connect(self.toggle_liveview)

    def setup_gui(self):
        if DEBUG:
            print("Inside setup_gui")
        # Focus lock widget
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.setMinimumSize(2, 200)

        # LiveView Button
        self.liveviewButton = QtGui.QPushButton('Camera LIVEVIEW')
        self.liveviewButton.setCheckable(True)

        # turn ON/OFF feedback loop
        self.feedbackLoopBox = QtGui.QCheckBox('Feedback loop')

        # shutter button and label
        self.shutterLabel = QtGui.QLabel('Shutter open?')
        self.shutterCheckbox = QtGui.QCheckBox('IR laser')

        # Create ROI button
        # TODO: completely remove the ROI stuff from the code
        self.ROIbutton = QtGui.QPushButton('ROI')
        # Select ROI
        self.selectROIbutton = QtGui.QPushButton('Select ROI')
        # Delete ROI
        self.deleteROIbutton = QtGui.QPushButton('Delete ROI')
        self.update_ROI_buttons(False)

        # Save current frame button
        # self.currentFrameButton = QtGui.QPushButton('Save current frame')

        self.calibrationButton = QtGui.QPushButton('Calibrate')

        self.exportDataButton = QtGui.QPushButton('Export data')
        self.saveDataBox = QtGui.QCheckBox("Save data")
        self.clearDataButton = QtGui.QPushButton('Clear data')

        self.pxSizeLabel = QtGui.QLabel('Pixel size (nm)')
        self.pxSizeEdit = QtGui.QLineEdit(str(START_PIXEL_SIZE))  # Original: 10nm en focus.py
        self.focusPropertiesDisplay = QtGui.QLabel(' st_dev = 0  max_dev = 0')

        # gui connections
        self.saveDataBox.stateChanged.connect(self.emit_save_data_state)
        self.selectROIbutton.clicked.connect(self.select_roi)
        self.clearDataButton.clicked.connect(self.clear_graph)
        self.pxSizeEdit.textChanged.connect(self.emit_param)
        self.deleteROIbutton.clicked.connect(self.delete_roi)
        self.ROIbutton.clicked.connect(self.roi_method)

        # focus camera display
        self.camDisplay = pg.GraphicsLayoutWidget()
        self.camDisplay.setMinimumHeight(300)
        self.camDisplay.setMinimumWidth(300)

        self.vb = self.camDisplay.addViewBox(row=0, col=0)
        self.vb.setAspectLocked(True)
        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)

        # set up histogram for the liveview image
        self.hist = pg.HistogramLUTItem(image=self.img)
        # lut = viewbox_tools.generatePgColormap(cmaps.inferno)
        # self.hist.gradient.setColorMap(lut)
        self.hist.gradient.loadPreset('plasma')
        self.hist.vb.setLimits(yMin=0, yMax=10000)

        for tick in self.hist.gradient.ticks:
            tick.hide()

        self.camDisplay.addItem(self.hist, row=0, col=1)

        # focus lock graph
        self.focusGraph = pg.GraphicsWindow()
        self.focusGraph.setAntialiasing(True)

        self.focusGraph.statistics = pg.LabelItem(justify='right')
        self.focusGraph.addItem(self.focusGraph.statistics, row=0, col=0)
        self.focusGraph.statistics.setText('---')

        self.focusGraph.zPlot = self.focusGraph.addPlot(row=0, col=0)
        self.focusGraph.zPlot.setLabels(bottom=('Time', 's'),
                                        left= ('CM x position', 'px')) #('Z position', 'nm')
        self.focusGraph.zPlot.showGrid(x=True, y=True)
        self.focusCurve = self.focusGraph.zPlot.plot(pen='y')

        # self.focusSetPoint = self.focusGraph.plot.addLine(y=self.setPoint, pen='r')

        # GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)

        # parameters widget
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        # Widget size (widgets with buttons)
        self.paramWidget.setFixedHeight(330)
        self.paramWidget.setFixedWidth(140)

        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        subgrid.addWidget(self.calibrationButton, 7, 0, 1, 2)
        subgrid.addWidget(self.exportDataButton, 5, 0, 1, 2)
        subgrid.addWidget(self.clearDataButton, 6, 0, 1, 2)

        subgrid.addWidget(self.pxSizeLabel, 8, 0)
        subgrid.addWidget(self.pxSizeEdit, 8, 1)

        subgrid.addWidget(self.feedbackLoopBox, 9, 0)
        subgrid.addWidget(self.saveDataBox, 10, 0)
        # subgrid.addWidget(self.currentFrameButton, 13, 0)

        subgrid.addWidget(self.liveviewButton, 1, 0, 1, 2)
        subgrid.addWidget(self.ROIbutton, 2, 0, 1, 2)
        subgrid.addWidget(self.selectROIbutton, 3, 0, 1, 2)
        subgrid.addWidget(self.deleteROIbutton, 4, 0, 1, 2)

        subgrid.addWidget(self.shutterLabel, 11, 0)
        subgrid.addWidget(self.shutterCheckbox, 12, 0)

        grid.addWidget(self.paramWidget, 0, 1)
        grid.addWidget(self.focusGraph, 1, 0)
        grid.addWidget(self.camDisplay, 0, 0)

        # didnt want to work when being put at earlier point in this function
        self.liveviewButton.clicked.connect(
            lambda: self.toggle_liveview(self.liveviewButton.isChecked()))

    def closeEvent(self, *args, **kwargs):
        if DEBUG:
            print("Inside closeEvent")
        self.closeSignal.emit()
        time.sleep(1)
        # FIXME: No usar variables globales
        focusThread.exit()
        super().closeEvent(*args, **kwargs)
        # app.quit()  # FIXME: No usar variables globales


class Backend(QtCore.QObject):
    changedImage = pyqtSignal(np.ndarray)
    changedData = pyqtSignal(np.ndarray, np.ndarray)
    changedSetPoint = pyqtSignal(float)
    zIsDone = pyqtSignal(bool, float)
    shuttermodeSignal = pyqtSignal(int, bool)
    liveviewSignal = pyqtSignal(bool)
    focuslockpositionSignal = pyqtSignal(float)

    """
    Signals
    - changedImage:
        To: [frontend] get_image
    - changedData:
        To: [frontend] get_data
    - changedSetPoint:
        To: [frontend] get_setpoint
    - zIsDone:
        To: [psf] get_z_is_done
    - shuttermodeSignal:
        To: [frontend] update_shutter
    - focuslockpositionSignal:
        To: [scan] get current focus lock position
    """

    def __init__(self, camera, adw, *args, **kwargs):
        if DEBUG:
            print("Inside init in backend")
        super().__init__(*args, **kwargs)

        self.camera = camera
        #Ver si hace falta añadir la variable self.__acquisition_running
        self.adw = adw
        self.feedback_active = False
        self.cropped = False
        self.standAlone = False
        self.camON = False

        # Coordenadas del ROI en pixeles (enteros)
        self.ROIcoordinates: np.ndarray = np.array([0, 0, 0, 0], dtype=int)
        
        #Directorio para trabajar
        today = str(date.today()).replace('-', '') # TO DO: change to get folder from microscope
        root = 'C:\\Data\\'
        folder = root + today
        print("folder:", folder)
        subfolder = 'focus_mode'
        folder_with_subfolder = os.path.join(folder, subfolder)
        if not os.path.exists(folder_with_subfolder):
            os.makedirs(folder_with_subfolder)
        print("folder:", folder_with_subfolder)

        filename = r'focus'
        self.filename = os.path.join(folder_with_subfolder, filename)
        print("filename: ", self.filename)

        self.save_data_state = False

        self.npoints = 400  # Nro de puntos para graficar

        # Relacion entre pixel que se corre la señal y nm en z
        self.pxSize = START_PIXEL_SIZE  # TODO: check correspondence with GUI
        self.focusSignal = 0.  # pixel x del CM en el ROI (fraccionario)
        self.setPoint = 0  # nm del CM (focssig * pxsize)

        self.focusTime = 100  # periodo de actualización en ms
        self.focusTimer = QtCore.QTimer()

        self.reset()
        self.reset_data_arrays()

        if self.camera.open_device():
            self.camera.set_roi(16, 16, 1920, 1200)
            try:
                self.camera.alloc_and_announce_buffers()
                self.camera.start_acquisition()
            except Exception as e:
                print("Exception inicializando la cámara:", str(e))
                self.camera.destroy_all()
                raise
        else:
            self.camera.destroy_all()

    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        """Recibe los parámetros del frontend."""
        if DEBUG:
            print("Inside get_frotend_param")

        self.pxSize = params['pxSize']
        print(datetime.now(), ' [focus] got px size', self.pxSize, ' nm')

    def set_actuator_param(self, pixeltime=1000): #pixeltime in µs 
        """Manda a la adwin la posición buscada en z."""
        if DEBUG:
            print("Inside set_actuator_param")
        self.adw.Set_FPar(36, tools.timeToADwin(pixeltime))

        # set-up actuator initial param
        # TODO: make this more robust
        z_f = tools.convert(10, 'XtoU')
        self.adw.Set_FPar(32, z_f)
        # en teoría (Andi) no estaría usando esto porque en actuator_z, está comentado
        self.adw.Set_Par(30, 1)

    def actuator_z(self, z_f: float):
        """Envial al actuador a z_f (en µm)."""
        if DEBUG:
            print("Inside actuator_z")
        z_f = tools.convert(z_f, 'XtoU')  # XtoU' lenght  (um) to bits
        self.adw.Set_FPar(32, z_f) # Index = 32 (to choose process 3), Value = z_f, le asigna a setpointz (en process 3) el valor z_f
        # en teoría (Andi) no estaría usando esto porque en actuator_z, está comentado
        self.adw.Set_Par(30, 1)

    @pyqtSlot(bool)
    def liveview(self, value):
        if value:
            self.camON = True
            self.liveview_start()
        else:
            self.liveview_stop()
            self.camON = False

    def liveview_start(self):
        if DEBUG:
            print("Inside Liveview-start")
        # if self.camON:
        #     self.focusTimer.stop()
        #     self.camON = False
        self.camON = True
        self.focusTimer.start(self.focusTime)

    def liveview_stop(self):
        if DEBUG:
            print("Inside Liveview-stop")
        self.focusTimer.stop()
        print("cameraTimer: stopped")
        self.camON = False
        # x0 = 0
        # y0 = 0
        # x1 = 1200
        # y1 = 1920
        # val = np.array([x0, y0, x1, y1])
        # self.get_new_roi(val)

    @pyqtSlot(int, bool)
    def toggle_ir_shutter(self, num, val):
        if DEBUG:
            print("Inside toggle_ir_shutter")
        # TODO: change code to also update checkboxes in case of minflux measurement
        if (num == 5) or (num == 8):
            if val:
                tools.toggle_shutter(self.adw, 5, True)
                print(datetime.now(), '[focus] IR shutter opened')
            else:
                tools.toggle_shutter(self.adw, 5, False)
                print(datetime.now(), '[focus] IR shutter closed')

    @pyqtSlot(int, bool)
    def shutter_handler(self, num, on):
        if DEBUG:
            print("Inside shutter_handler")
        self.shuttermodeSignal.emit(num, on)

    @pyqtSlot(bool)
    def toggle_tracking(self, val):
        """Prepara el tracking, no la corrección.

        Connection: [frontend] trackingBeadsBox.stateChanged
        Description: toggles ON/OFF tracking of IR reflection from sample.
        Drift correction feedback loop is not automatically started.
        """
        if DEBUG:
            print("Inside toggle_tracking")
        # TODO: write track procedure like in xy-tracking
        self.startTime = time.time()
        if val is True:
            if DEBUG:
                print("Inside toggle_tracking in val is True")
            self.reset()
            self.reset_data_arrays()
            self.tracking_value = True
        if val is False:
            if DEBUG:
                print("Inside toggle_tracking in val is False")
            self.tracking_value = False

    @pyqtSlot(bool)
    def toggle_feedback(self, val, mode='continous'):
        """Toggle ON/OFF feedback.

        Mode is either 'continous' (TCSPC) or 'discrete' (scan imaging) for
        selecting the correction strategy.
        """
        if val:
            if DEBUG:
                print("Inside toggle_feedback in val is True")
            # Aquí capaz que puedo llamar a center of mass en lugar de hacerlo en setup_feedback
            # self.center_of_mass()
            self.reset()
            self.setup_feedback()
            self.update()
            self.feedback_active = True

            # set up and start actuator process
            if mode == 'continous':
                if DEBUG:
                    print("Inside toggle_feedback in mode continuous")
                self.set_actuator_param()
                # self.adw.Set_Par(39, 0)
                self.adw.Start_Process(3)
                print(datetime.now(), '[focus] Process 3 started')
            print(datetime.now(), ' [focus] Feedback loop ON')
        else:
            if DEBUG:
                print("Inside toggle_feedback in val is False")
            self.feedback_active = False
            if mode == 'continous':
                self.adw.Stop_Process(3)
                # self.adw.Set_Par(39, 1) # Par 39 stops Process 3 (see ADbasic code)
                print(datetime.now(), '[focus] Process 3 stopped')
            print(datetime.now(), ' [focus] Feedback loop OFF')

    # analogo a track (en parte) en xyz_tracking
    # Actúa una vez, cuando se inicia el feedback loop
    @pyqtSlot()
    def setup_feedback(self):
        """Prepara info interna para tracking."""
        if DEBUG:
            print("Inside setup_feedback")
        # Esto es imagen
        self.setPoint = self.focusSignal * self.pxSize  # define setpoint 
        # [self.focusSignal]= px que se mueve el reflejo en z
        # [pxSize] = nm/px en z (entra desde interfaz, sale de la calibración)
        # [self.setPoint] = nm

        # Esto es platina
        # current z position of the piezo
        self.initial_z = tools.convert(self.adw.Get_FPar(72), 'UtoX')
        self.target_z = self.initial_z  # set initial_z as target_z, µm
        # Es posible que esta línea no afecte a update_feedback
        self.changedSetPoint.emit(self.focusSignal)
        # TO DO: implement calibrated version of this

    def update_feedback(self, mode='continous'):
        """Actualiza los valores del feedback.

        Parameters
        ----------
        mode : str, optional
            Modo de feedback. The default is 'continous'.

        Returns
        -------
        None.
        """
        if DEBUG:
            print("Inside update_feedback")
        # Esto es imagen
        # dz es cuantos pixeles se corrio el centro (EN un eje) respecto al
        # original
        dz = self.focusSignal * self.pxSize - self.setPoint
        # [dz] = px*(nm/px) - nm =nm
        # print ("Image setpoint: ", self.setPoint, "nm")
        # print("New value in image", self.focusSignal*self.pxSize, "nm")

        threshold = 3  # in nm
        far_threshold = 16  # in nm
        correct_factor = .6
        security_thr = 200  # in nm
        correction = 0.0

        if np.abs(dz) > threshold:
            correction = dz
            if np.abs(dz) < far_threshold:
                correction *= correct_factor

        if np.abs(correction) > security_thr:
            print(datetime.now(), "[focus] Correction movement larger than 200 nm!!!!")
        else:
            # Esto es cuanto es el movimiento real de la platina
            print("self.target_z initial in piezo: ", self.target_z, "µm.")
            # TODO: VER SIGNOS
            self.target_z = self.target_z - correction/1000
            # [self.target_z] = µm + nm/1000 = µm
            print("self.target_z in piezo: ", self.target_z, "µm.")

            if mode == 'continous':
                self.actuator_z(self.target_z)
            if mode == 'discrete':
                # it's enough to have saved the value self.target_z
                print(datetime.now(), '[focus] discrete correction to',
                      self.target_z)

    def update_graph_data(self):
        """Update the data displayed in the gui graph."""
        if DEBUG:
            print("Inside update_graph_data")

        if self.ptr < self.npoints:
            self.data[self.ptr] = self.focusSignal
            self.time[self.ptr] = self.currentTime
            # Esta señal va a get_data
            self.changedData.emit(self.time[0:self.ptr + 1],
                                  self.data[0:self.ptr + 1])
        else:
            self.data[:-1] = self.data[1:]
            self.data[-1] = self.focusSignal
            self.time[:-1] = self.time[1:]
            self.time[-1] = self.currentTime
            self.changedData.emit(self.time, self.data)
        self.ptr += 1

    def update_stats(self):
        """Actualiza el la estadística.

        No se usa.
        """
        if DEBUG:
            print("Inside update_stats")
        # TO DO: fix this function
        signal = self.focusSignal
        if self.n == 1:
            self.mean = signal
            self.mean2 = self.mean**2
        else:
            self.mean += (signal - self.mean)/self.n
            self.mean2 += (signal**2 - self.mean2)/self.n

        # Stats
        self.std = np.sqrt(self.mean2 - self.mean**2)
        self.max_dev = np.max([self.max_dev,
                              self.focusSignal - self.setPoint])
        statData = 'std = {}    max_dev = {}'.format(np.round(self.std, 3),
                                                     np.round(self.max_dev, 3))
        self.gui.focusGraph.statistics.setText(statData)
        self.n += 1

    def update(self):
        """Actualiza todo. Es el nucleo del asunto."""
        if DEBUG:
            print("Inside update")
        self.acquire_data()
        self.update_graph_data()

        #  if locked, correct position
        if self.feedback_active:
            # self.updateStats()
            self.update_feedback()
        if self.save_data_state:
            self.time_array.append(self.currentTime)
            self.z_array.append(self.focusSignal)

    def acquire_data(self):  # Es update_view en otros códigos
        """Toma imagen y ubica CM."""
        if DEBUG:
            print("Inside acquire_data")
        # acquire image
        # This is a 2D array, (only R channel, to have other channel, or the
        # sum, go to ids_cam driver)
        self.image = self.camera.on_acquisition_timer()

        # This following lines are executed inside ids_cam driver, to change
        # this I should modify these lines there (depending on which one I prefer: R or R+G+B+A)
        # self.image = np.sum(raw_image, axis=2)  # sum the R, G, B images 
        # self.image = raw_image[:, :, 0] # take only R channel

        # WARNING: check if it is necessary to fix to match camera orientation
        # with piezo orientation
        self.image = np.rot90(self.image, k=3)  # Added by FC
        # Send image to gui
        # TODO: mover al final y mandar self.masscenter para que grafique en el front
        self.changedImage.emit(self.image)  # This signal goes to get_image
        # image sent to get_image. Type:  <class 'numpy.ndarray'>
        self.currentTime = ptime.time() - self.startTime
        self.center_of_mass()  # Esto da focusSignal

    def center_of_mass(self):
        """Actualiza el centro de masa de la imagen guardada."""
        xmin, xmax, ymin, ymax = self.ROIcoordinates
        zimage = self.image[xmin:xmax, ymin:ymax]

        self.masscenter = np.array(ndi.measurements.center_of_mass(zimage))

        # calculate z estimator
        self.focusSignal = self.masscenter[0]
        # self.focusSignal = np.sqrt(self.masscenter[0]**2 + self.masscenter[1]**2) #OJO aquí, a veces puede ser que vaya el signo menos, pero el original es signo mas
        # print("coord x: ", self.masscenter[0], "coord y: ", self.masscenter[1])
        # print("FocusSignal in center of mass:", self.focusSignal)
        self.currentTime = ptime.time() - self.startTime

    @pyqtSlot(bool, bool)
    def single_z_correction(self, feedback_val, initial):
        """Hace una sola corrección z."""
        if DEBUG:
            print("Inside single_z_correction")

        if initial:
            if not self.camON:
                self.camON = True
            self.reset()
            self.reset_data_arrays()
        self.acquire_data()
        self.update_graph_data()
        if initial:
            self.setup_feedback()
        else:
            self.update_feedback(mode='discrete')
        if self.save_data_state:
            self.time_array.append(self.currentTime)
            self.z_array.append(self.focusSignal)
        self.zIsDone.emit(True, self.target_z)

    def calibrate(self):
        if DEBUG:
            print("Inside calibrate")
        # TODO: fix calibration function  HECHO!
        fname = self.filename

        if fname[0] == '!':
            filename = fname[1:]
        else:
            filename = tools.getUniqueName(fname)
        filename = filename + '_calib_data.txt'
        print("archivo nombre: ", filename)
        
        self.z_calib_array = []
        self.cm_calib_array = []
        
        self.focusTimer.stop()
        time.sleep(0.100)

        nsteps = 40
        savedData = np.zeros((2, nsteps))
        zrange = .5  # in µm
        
        old_z_param = self.adw.Get_FPar(72)
        old_z = tools.convert(old_z_param, 'UtoX')
        calibData = np.zeros(nsteps)
        zData = np.linspace(old_z - zrange/2, old_z + zrange/2, nsteps)

        # zMoveTo(self.adw, zmin)
        self.adw.Start_Process(3)
        time.sleep(0.100)
        for i, z in enumerate(zData):
            # zMoveTo(self.adw, z)
            self.actuator_z(z)
            time.sleep(.125)
            self.update()
            calibData[i] = self.focusSignal
            np.save(f'C:\\Data\\20240426\\calib_closed_imagenes\\calib_1um_roi_chico\\img_{i}-{int(z*1000)}pm.npy', self.image)
        np.save('C:\\Data\\20240426\\calib_closed_imagenes\\calib_1um_roi_chico\\zData.npy', np.array(zData))
        np.save('C:\\Data\\20240426\\calib_closed_imagenes\\calib_1um_roi_chico\\calib.npy', np.array(calibData))
        for z, cm in zip(zData, calibData):
            self.z_calib_array.append(z)
            self.cm_calib_array.append(cm)
            
        self.adw.Set_FPar(32, old_z_param)
        time.sleep(0.200)
        self.adw.Stop_Process(3)
        self.focusTimer.start(self.focusTime)

        savedData[0, :] = np.array(self.z_calib_array)
        savedData[1, :] = np.array(self.cm_calib_array)

        np.savetxt(filename, savedData.T, header='z (px), cm (px)')

        print(datetime.now(), '[focus] Calibration z data exported to', filename)
        
    def reset(self):
        """Resetear información para graficar."""
        if DEBUG:
            print("Inside reset")
        self.data = np.zeros(self.npoints)
        self.time = np.zeros(self.npoints)
        self.ptr = 0
        self.startTime = ptime.time()

        # Creo que estos no se usan, son para stats
        self.max_dev = 0
        self.mean = self.focusSignal
        self.std = 0
        # print("focusSignal in reset: ", self.mean)
        self.n = 1

    def reset_data_arrays(self):
        """Resetea información para guardar."""
        if DEBUG:
            print("Inside reset_data_arrays")
        self.time_array = []
        self.z_array = []

    def export_data(self):
        """Graba info de z en un archivo."""
        if DEBUG:
            print("Inside export_data")
        fname = self.filename
        # case distinction to prevent wrong filenaming when starting minflux or
        # psf measurement
        if fname[0] == '!':
            filename = fname[1:]
        else:
            filename = tools.getUniqueName(fname)
        filename = filename + '_zdata.txt'

        size = np.size(self.z_array)
        savedData = np.zeros((2, size))

        savedData[0, :] = np.array(self.time_array)
        savedData[1, :] = np.array(self.z_array)

        np.savetxt(filename, savedData.T, header='t (s), z (px)')

        print(datetime.now(), '[focus] z data exported to', filename)

    @pyqtSlot()
    def get_stop_signal(self):
        """
        From: [psf]
        Description: stops liveview, tracking, feedback if they where running to
        start the psf measurement with discrete xy - z corrections
        """
        if DEBUG:
            print("Inside get_stop_signal")
        self.toggle_feedback(False)
        self.toggle_tracking(False)

        self.save_data_state = True  # TO DO: sync this with GUI checkboxes (Lantz typedfeat?)

        self.reset()
        self.reset_data_arrays()

    @pyqtSlot()    
    def get_lock_signal(self):
        if DEBUG:
            print("Inside get_lock_signal")
        if not self.camON:
            self.liveviewSignal.emit(True)
            print("self.liveviewSignal.emit(True) executed in get lock signal")
        self.reset_data_arrays()
        self.toggle_feedback(True)
        self.toggle_tracking(True)
        self.save_data_state = True

        # TO DO: fix updateGUIcheckboxSignal
        # self.updateGUIcheckboxSignal.emit(self.tracking_value,
        #                                   self.feedback_active,
        #                                   self.save_data_state)

        print(datetime.now(), '[focus] System focus locked')

    @pyqtSlot(np.ndarray)
    def get_new_roi(self, val):  # This is get_roi_info in other codes
        """Receive new ROI data from frontend.

        Connection: [frontend] changedROI
        Description: gets coordinates of the ROI in the GUI
        """
        if DEBUG:
            print("Inside get_new_roi")
        self.ROIcoordinates = val.astype(int)
        if DEBUG:
            print(datetime.now(), '[focus] got ROI coordinates')

    # FIXME: no veo que se llame en ningún luhgar
    @pyqtSlot(bool, str)
    def get_tcspc_signal(self, val, fname):
        """
        Get signal to start/stop xy position tracking and lock during
        tcspc acquisition. It also gets the name of the tcspc file to produce
        the corresponding z data file

        bool val
        True: starts the tracking and feedback loop
        False: stops saving the data and exports the data during tcspc measurement
        tracking and feedback are not stopped automatically
        """
        if DEBUG:
            print("Inside get_tcspc_signal")
        self.filename = fname
        if val is True:
            self.reset()
            self.reset_data_arrays()
            self.toggle_feedback(True)
            self.save_data_state = True
        else:
            self.export_data()
            self.save_data_state = False
        # TO DO: fix updateGUIcheckboxSignal
        # self.updateGUIcheckboxSignal.emit(self.tracking_value,
        #                                   self.feedback_active,
        #                                   self.save_data_state)

    # @pyqtSlot(bool, str)
    # def get_scan_signal(self, val, fname):
    #     """
    #     Get signal to stop continous xy tracking/feedback if active and to
    #     go to discrete xy tracking/feedback mode if required
    #     """
    #     if DEBUG:
    #         print("Inside get_scan_signal")
    #     pass

    @pyqtSlot(bool)
    def get_save_data_state(self, val):
        if DEBUG:
            print("Inside get_save_data_state")
        self.save_data_state = val

    @pyqtSlot(str)
    def get_end_measurement_signal(self, fname):
        """
        From: [minflux] or [psf]
        Description: at the end of the measurement exports the xy data
        """
        if DEBUG:
            print("Inside get_end_measurement_signal")
        self.filename = fname
        self.export_data()
        self.toggle_feedback(False)
        self.toggle_tracking(False)

        if self.camON:
            self.focusTimer.stop()
            self.liveviewSignal.emit(False)
            # print("self.liveviewSignal.emit(False) executed in get end measurement")

    def set_moveTo_param(self, x_f, y_f, z_f, n_pixels_x=128, n_pixels_y=128,
                         n_pixels_z=128, pixeltime=2000):
        if DEBUG:
            print("Inside set_moveTo_param")
        x_f = tools.convert(x_f, 'XtoU')
        y_f = tools.convert(y_f, 'XtoU')
        z_f = tools.convert(z_f, 'XtoU')
        self.adw.Set_Par(21, n_pixels_x)
        self.adw.Set_Par(22, n_pixels_y)
        self.adw.Set_Par(23, n_pixels_z)
        self.adw.Set_FPar(23, x_f)
        self.adw.Set_FPar(24, y_f)
        self.adw.Set_FPar(25, z_f)
        self.adw.Set_FPar(26, tools.timeToADwin(pixeltime))

    def moveTo(self, x_f, y_f, z_f):
        self.set_moveTo_param(x_f, y_f, z_f)
        self.adw.Start_Process(2)

    @pyqtSlot(float)
    def get_focuslockposition(self, position):
        """Actualiza en otro módulo la posición de lock de z.

        Conectado con SCAN.
        """
        if DEBUG:
            print("Inside get_focuslockposition")
        if position == -9999:
            position = self.setPoint
        else:
            position = self.focusSignal
        self.focuslockpositionSignal.emit(position)

    @pyqtSlot()
    def stop(self):
        """Cierra TODO antes de terminar el programa.

        viene del front.
        """
        if DEBUG:
            print("Inside stop")
        self.toggle_ir_shutter(8, False)
        time.sleep(1)
        self.focusTimer.stop()
        self.reset()
        if self.standAlone is True:
            # Go back to 0 position
            x_0 = 0
            y_0 = 0
            z_0 = 0
            self.moveTo(x_0, y_0, z_0)
        self.camera.destroy_all()

        print(datetime.now(), '[focus] Focus stopped')

        # clean up aux files from NiceLib
        try:
            os.remove(r'C:\Users\USUARIO\Documents\GitHub\pyflux\lextab.py')
            os.remove(r'C:\Users\USUARIO\Documents\GitHub\pyflux\yacctab.py')
        except Exception:
            pass

    def make_connection(self, frontend):
        if DEBUG:
            print("Inside make_connection in Backend")

        frontend.changedROI.connect(self.get_new_roi)
        frontend.closeSignal.connect(self.stop)
        # frontend.lockFocusSignal.connect(self.lock_focus)
        frontend.feedbackLoopBox.stateChanged.connect(
            lambda: self.toggle_feedback(frontend.feedbackLoopBox.isChecked()))
        frontend.saveDataSignal.connect(self.get_save_data_state)
        frontend.exportDataButton.clicked.connect(self.export_data)
        frontend.clearDataButton.clicked.connect(self.reset)
        frontend.clearDataButton.clicked.connect(self.reset_data_arrays)
        frontend.calibrationButton.clicked.connect(self.calibrate)
        frontend.shutterCheckbox.stateChanged.connect(
            lambda: self.toggle_ir_shutter(8, frontend.shutterCheckbox.isChecked()))
        frontend.paramSignal.connect(self.get_frontend_param)
        frontend.liveviewButton.clicked.connect(self.liveview)
        # frontend.currentFrameButton.clicked.connect(self.save_current_frame)


if __name__ == '__main__':
    if DEBUG:
        print("Inside main")

    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()

    # app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    print(datetime.now(), '[focus] Focus lock module running in stand-alone mode')

    # Initialize devices

    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)

    # if camera wasnt closed properly just keep using it without opening new one
    try:
        cam = ids_cam.IDS_U3()
    except Exception:
        print("Algo pasó inicializando la cámara")
        pass

    gui = Frontend()
    worker = Backend(cam, adw)
    worker.standAlone = True

    gui.make_connection(worker)
    worker.make_connection(gui)

    gui.emit_param()

    focusThread = QtCore.QThread()
    worker.moveToThread(focusThread)
    worker.focusTimer.moveToThread(focusThread)
    worker.focusTimer.timeout.connect(worker.update)

    focusThread.start()

    # initialize fpar_70, fpar_71, fpar_72 ADwin position parameters
    pos_zero = tools.convert(0, 'XtoU')

    worker.adw.Set_FPar(70, pos_zero)
    worker.adw.Set_FPar(71, pos_zero)
    worker.adw.Set_FPar(72, pos_zero)

    worker.moveTo(10, 10, 10)  # in µm

    gui.setWindowTitle('Focus lock')
    gui.resize(1500, 500)

    gui.show()
    app.exec_()
