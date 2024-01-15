# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2018

@authors: Luciano Masullo modified by Flor C. to use another ROI and IDS cam
"""

import numpy as np
import time
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from datetime import date, datetime
import os

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.ptime as ptime
import qdarkstyle # see https://stackoverflow.com/questions/48256772/dark-theme-for-in-qt-widgets

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QGroupBox
import sys
sys.path.append('C:\Program Files\Thorlabs\Scientific Imaging\ThorCam') #Ver si puedo quitar esto
# install from https://instrumental-lib.readthedocs.io/en/stable/install.html
import tools.viewbox_tools as viewbox_tools
import tools.tools as tools
import tools.PSF as PSF
import tools.colormaps as cmaps
from scipy import optimize as opt

import scan
import drivers.ADwin as ADwin
import drivers.ids_cam as ids_cam

#ids_peak.Library.Initialize()

FPS_LIMIT = 30

DEBUG = False

def actuatorParameters(adwin, z_f, n_pixels_z=50, pixeltime=1000):

    if DEBUG:
        print("Inside actuatorParameters")
    z_f = tools.convert(z_f, 'XtoU')

    adwin.Set_Par(33, n_pixels_z)
    adwin.Set_FPar(35, z_f)
    adwin.Set_FPar(36, tools.timeToADwin(pixeltime))

def zMoveTo(adwin, z_f):
    if DEBUG:
        print("Inside zMoveto")

    actuatorParameters(adwin, z_f)
    adwin.Start_Process(3)

class Frontend(QtGui.QFrame):
    
    if DEBUG:
        print("Inside Frontend")
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

        self.setup_gui()
        
        x0 = 0
        y0 = 0
        x1 = 1280 
        y1 = 1024 
            
        value = np.array([x0, y0, x1, y1])
        self.changedROI.emit(value)
        
    def emit_param(self):
        
        if DEBUG:
            print("Inside emit_param")
        params = dict() #Se crea diccionario vacío FC
        params['pxSize'] = float(self.pxSizeEdit.text())
        
        self.paramSignal.emit(params)

    def roi_method(self):
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
        self.ROIbutton.setChecked(False)
        self.selectROIbutton.setEnabled(True) #duda: debe ir esto?
        
    def select_roi(self): #Analogo a emit_roi_info #Esta señal va al backend a get_new_roi
        
        if DEBUG:
            print("Inside select_roi")
        #self.cropped = True
        self.getStats = True
        xmin, ymin = self.roi.pos()
        xmax, ymax = self.roi.pos() + self.roi.size()
        
        value = np.array([xmin, xmax, ymin, ymax])  
            
        #value = np.array([y0, x0, y1, x1])
        print("Coordinates of the selected roi: ", value)
            
        self.changedROI.emit(value)
    
        #self.vb.removeItem(self.roi)
        #elf.roi.hide()
        #self.roi = None
        
        #self.vb.enableAutoRange()
        
#    def toggleFocus(self):
#        
#        if self.lockButton.isChecked():
#            
#            self.lockFocusSignal.emit(True)
#
##            self.setpointLine = self.focusGraph.zPlot.addLine(y=self.setPoint, pen='r')
#            
#        else:
#            
#            self.lockFocusSignal.emit(False)
        
#    def delete_roi(self):
#        
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
            
    @pyqtSlot(bool)        
    def toggle_liveview(self, on):
        if DEBUG:
            print("Inside toggle_liveview")
        if on:
            self.liveviewButton.setChecked(True)
            print(datetime.now(), '[focus] focus live view started')
        else:
            self.liveviewButton.setChecked(False)
            self.select_roi()
            self.img.setImage(np.zeros((1200,1920)), autoLevels=False)

            print(datetime.now(), '[focus] focus live view stopped - line 202')
            
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
    def get_image(self, img):
        if DEBUG:
            print(" Inside get_image ")
        self.img.setImage(img, autoLevels=False)
        #croppedimg = img[0:300, 0:300]
        #self.img.setImage(croppedimg)  
            
    @pyqtSlot(np.ndarray, np.ndarray)
    def get_data(self, time, position):
        if DEBUG:
            print("Inside get_data ")
        
        self.focusCurve.setData(time, position)
             
        if self.feedbackLoopBox.isChecked():
            
            if DEBUG:
                print("self.feedbackLoopBox.isChecked() in get_data")
            if len(position) > 2:
                #print("len(position) is higher than 2")
        
                zMean = np.mean(position)
                zStDev = np.std(position)
                
                # TO DO: fix focus stats
                
#                self.focusMean.setValue(zMean)
#                self.focusStDev0.setValue(zMean - zStDev)
#                self.focusStDev1.setValue(zMean + zStDev)
      
    @pyqtSlot(float)          
    def get_setpoint(self, value):
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
        
        # TO DO: fix setpoint line
        
#        self.focusGraph.zPlot.removeItem(self.focusSetPoint)
#        self.focusGraph.zPlot.removeItem(self.focusMean)
#        self.focusGraph.zPlot.removeItem(self.focusStDev0)
#        self.focusGraph.zPlot.removeItem(self.focusStDev1)
        
        pass
    
    @pyqtSlot(int, bool)    
    def update_shutter(self, num, on):
        
        '''
        setting of num-value:
            0 - signal send by scan-gui-button --> change state of all minflux shutters
            1...6 - shutter 1-6 will be set according to on-variable, i.e. either true or false; only 1-4 controlled from here
            7 - set all minflux shutters according to on-variable
            8 - set all shutters according to on-variable
        for handling of shutters 1-5 see [scan] and [focus]
        '''
        if DEBUG:
            print("Inside update_shutter")
        if (num == 5)  or (num == 8):
            self.shutterCheckbox.setChecked(on)
            
    def make_connection(self, backend):
        if DEBUG:
            print("Inside make_connetion")
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
        #self.setMinimumSize(width, height)
        self.setMinimumSize(2,200)
        
        # LiveView Button

        self.liveviewButton = QtGui.QPushButton('Camera LIVEVIEW')
        self.liveviewButton.setCheckable(True)

        # turn ON/OFF feedback loop
        
        self.feedbackLoopBox = QtGui.QCheckBox('Feedback loop')
        
        #shutter button and label
        self.shutterLabel = QtGui.QLabel('Shutter open?')
        self.shutterCheckbox = QtGui.QCheckBox('IR laser')
        
        # Create ROI button
        
        # TODO: completely remove the ROI stuff from the code

        self.ROIbutton = QtGui.QPushButton('ROI')
        self.ROIbutton.setCheckable(True)
        
        # Select ROI
        self.selectROIbutton = QtGui.QPushButton('Select ROI')
        
        # Delete ROI
        self.deleteROIbutton = QtGui.QPushButton('Delete ROI')
        
        self.calibrationButton = QtGui.QPushButton('Calibrate')
        
        self.exportDataButton = QtGui.QPushButton('Export data')
        self.saveDataBox = QtGui.QCheckBox("Save data")
        self.clearDataButton = QtGui.QPushButton('Clear data')
        
        self.pxSizeLabel = QtGui.QLabel('Pixel size (nm)')
        self.pxSizeEdit = QtGui.QLineEdit('50') #Original: 10nm en focus.py
        self.focusPropertiesDisplay = QtGui.QLabel(' st_dev = 0  max_dev = 0')
        
#        self.deleteROIbutton.setEnabled(False)
#        self.selectROIbutton.setEnabled(False)

        #################################
        # stats widget
        
        self.statWidget = QGroupBox('Live statistics')   
        self.statWidget.setFixedHeight(300)
        # self.statWidget.setFixedWidth(240)
        self.statWidget.setFixedWidth(350)

        self.zstd_label = QtGui.QLabel('Z std (nm)')
        
        self.zstd_value = QtGui.QLabel('0')
        ################################

        
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

        self.hist = pg.HistogramLUTItem(image=self.img)   # set up histogram for the liveview image
        lut = viewbox_tools.generatePgColormap(cmaps.inferno)
        self.hist.gradient.setColorMap(lut)
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
 
#        self.focusSetPoint = self.focusGraph.plot.addLine(y=self.setPoint, pen='r')

        # GUI layout
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        
        # parameters widget
        
        self.paramWidget = QtGui.QFrame()
        self.paramWidget.setFrameStyle(QtGui.QFrame.Panel |
                                       QtGui.QFrame.Raised)
        #Widget size (widgets with buttons)
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
        
        #Create button        
        #self.ROIButton = QtGui.QPushButton('ROI')
#        self.ROIButton.setCheckable(True)
#        self.ROIButton.clicked.connect(lambda: self.roi_method())
        
        subgrid.addWidget(self.liveviewButton, 1, 0, 1, 2)
        subgrid.addWidget(self.ROIbutton, 2, 0, 1, 2)
        subgrid.addWidget(self.selectROIbutton, 3, 0, 1, 2)
        subgrid.addWidget(self.deleteROIbutton, 4, 0, 1, 2)
        
        subgrid.addWidget(self.shutterLabel, 11, 0)
        subgrid.addWidget(self.shutterCheckbox, 12, 0)
        
        grid.addWidget(self.paramWidget, 0, 1)
        grid.addWidget(self.focusGraph, 1, 0)
        grid.addWidget(self.camDisplay, 0, 0)
        
        #didnt want to work when being put at earlier point in this function
        self.liveviewButton.clicked.connect(lambda: self.toggle_liveview(self.liveviewButton.isChecked()))

    def closeEvent(self, *args, **kwargs):
        if DEBUG:
            print("Inside closeEvent")
        
        self.closeSignal.emit()
        time.sleep(1)
        
        focusThread.exit()
        super().closeEvent(*args, **kwargs)
        app.quit()
        
        
class Backend(QtCore.QObject):
    if DEBUG:
        print("Inside Backend")
    
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
        self.roi_area = np.zeros(4)
        
        today = str(date.today()).replace('-', '') # TO DO: change to get folder from microscope
        root = r'C:\\Data\\'
        folder = root + today
        
        filename = r'zdata.txt'
        self.filename = os.path.join(folder, filename)
        
        self.save_data_state = False
    
        self.npoints = 400
        
        
        self.pxSize = 50 #original 10nm FC  # in nm, TODO: check correspondence with GUI
        
        #self.sensorSize = np.array(image.shape) #Creo que esta linea no sirve para nada
        self.focusSignal = 0
        self.setPoint = 0
        
        # set focus update rate
        
        self.scansPerS = 10

        self.focusTime = 1000 / self.scansPerS
        self.focusTimer = QtCore.QTimer()
        
        self.reset()
        self.reset_data_arrays()
        
        if self.camera.open_device():
            self.camera.set_roi(16, 16, 1920, 1200)
            try:
                self.camera.alloc_and_announce_buffers()
                self.camera.start_acquisition()
            except Exception as e:
                print("Exception", str(e))
        else:
            self.camera.destroy_all()
        
    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        if DEBUG:
            print("Inside get_frotend_param")
        
        self.pxSize = params['pxSize']
        
        print(datetime.now(), ' [focus] got px size', self.pxSize, ' nm')
        
    def set_actuator_param(self, pixeltime=1000):

        print("Inside set_actuator_param")

        self.adw.Set_FPar(36, tools.timeToADwin(pixeltime))
        
        # set-up actuator initial param
    ##################NO entiendo esto
        z_f = tools.convert(10, 'XtoU') # TO DO: make this more robust #Cómo es esto de 10um para que convierta a bits
        self.adw.Set_FPar(32, z_f)
        self.adw.Set_Par(30, 1)
        print("z_f in set_actuator_param", z_f, "bits")
        
    def actuator_z(self, z_f):
        if DEBUG:
            print("Inside actuator_z")
        
        z_f = tools.convert(z_f, 'XtoU') # XtoU' lenght  (um) to bits
        #print("target_z es z_f in actuator z: ", z_f," bits")
        if DEBUG:
            print("z_f is self.target_z in actuator_z: ",z_f, "bits")
          
        self.adw.Set_FPar(32, z_f) # Index = 32 (to choose process 3), Value = z_f, le asigna a setpointz (en process 3) el valor z_f
        #Luego se asigna setpointz a currentz y ese valor se pasa al borne 6 de la ADwin
        #Luego ese valor currentz se asigna a fpar_72 y es el nuevo valor actual de z
        self.adw.Set_Par(30, 1)
        actual_z = tools.convert(self.adw.Get_FPar(72), 'UtoX')
        print("actual_z: ", actual_z, "um. Espero que este valor sea: ", self.initial_z, "um")
        #Si en esta linea son iguales, listo, sino probar restando dz/1000 en self.target_z
        #En base a esto recién se puede cambiar o no el signo en dz en xyz_tracking
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
        if self.camON:
            self.focusTimer.stop()
            self.camON = False
        self.camON = True
        self.focusTimer.start(self.focusTime)
        
    def liveview_stop(self):
        if DEBUG:
            print("Inside Liveview-stop")
        self.focusTimer.stop()
        print("cameraTimer: stopped")
        self.camON = False  
        x0 = 0
        y0 = 0
        x1 = 1200 
        y1 = 1920
            
        val = np.array([x0, y0, x1, y1])

        self.get_new_roi(val)
    
    @pyqtSlot(int, bool)
    def toggle_ir_shutter(self, num, val):
        if DEBUG:
            print("Inside toggle_ir_shutter")
        
        #TODO: change code to also update checkboxes in case of minflux measurement
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
        
        '''
        Connection: [frontend] trackingBeadsBox.stateChanged
        Description: toggles ON/OFF tracking of IR reflection from sample. 
        Drift correction feedback loop is not automatically started.
        
        '''
        if DEBUG:
            print("Inside toggle_tracking")
        #TODO: write track procedure like in xy-tracking
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
        ''' Toggles ON/OFF feedback for either continous (TCSPC) 
        or discrete (scan imaging) correction'''
        
        if val is True:
            if DEBUG:
                print("Inside toggle_feedback in val is True")
            #Aquí capaz que puedo llamar a center of mass en lugar de hacerlo en setup_feedback
            #self.center_of_mass()
            self.reset()
            self.setup_feedback()
            self.update()
            self.feedback_active = True
            
            # set up and start actuator process
            
            if mode == 'continous':
                if DEBUG:
                    print("Inside toggle_feedback in mode continuous")
            #Voy a comentar esto para ver si sigue funcionando, pero cual es su análogo?
                self.set_actuator_param()
#                self.adw.Set_Par(39, 0)
                self.adw.Start_Process(3)
                print(datetime.now(), '[focus] Process 3 started')
            
            print(datetime.now(), ' [focus] Feedback loop ON')
            
        if val is False:
            if DEBUG:
                print("Inside toggle_feedback in val is False")
            self.feedback_active = False
            
            if mode == 'continous':
            
                self.adw.Stop_Process(3)
#                self.adw.Set_Par(39, 1) # Par 39 stops Process 3 (see ADbasic code)
                print(datetime.now(), '[focus] Process 3 stopped')
                
            print(datetime.now(), ' [focus] Feedback loop OFF')

    
    @pyqtSlot()    
    def setup_feedback(self): #analogo a track (en parte) en xyz_tracking #Actúa una vez, cuando se inicia el feedback loop
        if DEBUG:
                print("Inside setup_feedback")
        ''' set up on/off feedback loop'''
        #Creo que podría anular la siguiente linea si va en toggle_feedback
        
        self.center_of_mass() #Esto se ejecuta para sacar self.focusSignal y configurar por primera vez el setpoint
        #Esto es imagen
        self.setPoint = self.focusSignal * self.pxSize # define setpoint 
        # [self.focusSignal]= px que se mueve el reflejo en z
        # [pxSize] = nm/px en z (entra desde interfaz, sale de la calibración)
        # [self.setPoint] = nm
        
        #Esto es platina
        self.initial_z = tools.convert(self.adw.Get_FPar(72), 'UtoX') # current z position of the piezo
        # self.adw.Get_FPar(72) toma la posicion en bits de la ADwin, luego la convierte a unidades de longitud (µm)
        self.target_z = self.initial_z # set initial_z as target_z, µm
        #print("Valor de focus signal en setup_feedback:", self.focusSignal," px. SetPoint: ", self.setPoint, "nm")
        #print("initial_z es target_z: ", self.initial_z, "µm")
        #print("esto salió de tomar la posicion del piezo en bits y convertir a um")
        self.changedSetPoint.emit(self.focusSignal) #Es posible que esta línea no afecte a update_feedback
        # antes de enviaba self.focusSignal en lugar de setPoint FC
        # TO DO: implement calibrated version of this
    
    def update_feedback(self, mode='continous'):
        if DEBUG:
                print("Inside update_feedback")
            
        #self.center_of_mass() #Esto se ejecuta para sacar self.focusSignal activamente
        #comento la linea anterior, ver qué cambia
        #Esto es imagen 
        
        dz = self.focusSignal * self.pxSize - self.setPoint #Este valor da positivo a veces y a veces negativo
        #[dz] = px*(nm/px) - nm =nm
        print ("Image setpoint: ", self.setPoint, "nm")
        print("New value in image", self.focusSignal*self.pxSize, "nm")
        print("dz: ", dz, "nm")
        print("self.initial_z in piezo: ", self.initial_z, "um")
        
        threshold = 7 # in nm
        far_threshold = 20 # in nm
        correct_factor = 1
        security_thr = 200 # in nm
        correction = 0.1
        
        # if np.abs(dz) > threshold:
            
        #     if np.abs(dz) < far_threshold:
                
        #         dz = correct_factor * dz
    
        if np.abs(dz) > security_thr:
            
            print(datetime.now(), '[focus] Correction movement larger than 200 nm, active correction turned OFF')
            
        else:
            #Esto es cuanto es el movimiento real de la platina
            self.target_z = self.initial_z - dz/1000  # conversion to µm #Creo que aquí está corrigiendo bien, le puse el signo menos 
            # [self.target_z] = µm + nm/1000 = µm
            print("self.target_z in piezo: ", self.target_z, "µm.")
                        
            if mode == 'continous':
                
                self.actuator_z(self.target_z)
                
            if mode == 'discrete':
                 
                # it's enough to have saved the value self.target_z
                print(datetime.now(), '[focus] discrete correction to', self.target_z)
            
    def update_graph_data(self):
        if DEBUG:
                print("Inside update_graph_data")
        ''' update of the data displayed in the gui graph '''
        
        if self.ptr < self.npoints:
            self.data[self.ptr] = self.focusSignal#* self.pxSize - self.setPoint  #Ahora se supone que focusSiganl no es cero
            #print("self.data[self.ptr]: ", self.data[self.ptr])
            self.time[self.ptr] = self.currentTime
            
            self.changedData.emit(self.time[0:self.ptr + 1], #Esta señal va a get_data
                                  self.data[0:self.ptr + 1])

        else:
            self.data[:-1] = self.data[1:]
            self.data[-1] = self.focusSignal
            #print("focusSignal in update_graph_data (in else): ", self.focusSignal)
            self.time[:-1] = self.time[1:]
            self.time[-1] = self.currentTime

            self.changedData.emit(self.time, self.data)
            
        self.ptr += 1
            
    def update_stats(self):
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
        if DEBUG:
                print("Inside update")
        
        self.acquire_data()
        self.update_graph_data()
        
        #  if locked, correct position
        
        if self.feedback_active:
            
#            self.updateStats()
            self.update_feedback()
            
        if self.save_data_state:
                        
            self.time_array.append(self.currentTime)
            self.z_array.append(self.focusSignal)
            
    def acquire_data(self): #Es update_view en otros códigos
        if DEBUG:
                print("Inside acquire_data")
                
        # acquire image
    
        self.image = self.camera.on_acquisition_timer() #This is a 2D array, (only R channel, to have other channel, or the sum, go to ids_cam driver)
        #This following lines are executed inside ids_cam driver, to change  this I should modify these lines there (depending on which one I prefer: R or R+G+B+A)
        #self.image = np.sum(raw_image, axis=2)  # sum the R, G, B images 
        #self.image = raw_image[:, :, 0] # take only R channel

        # WARNING: check if it is necessary to fix to match camera orientation with piezo orientation
        #find command for IDS, maybe in user manual
        # WARNING: fix to match camera orientation with piezo orientation
        #self.image = np.rot90(self.image, k=3) #Added by FC
        # Send image to gui
        self.changedImage.emit(self.image) # This signal goes to get_image
        #image sent to get_image. Type:  <class 'numpy.ndarray'>
        self.currentTime = ptime.time() - self.startTime
        self.center_of_mass() #Esto da focusSignal
        
    def center_of_mass(self):
        
        xmin, xmax, ymin, ymax = self.ROIcoordinates
        zimage = self.image[xmin:xmax, ymin:ymax]
        #print("zimage: ", zimage)
        
        # WARNING: extra rotation added to match the sensitive direction (hardware)
        
        #zimage = np.rot90(zimage, k=3)
        
        # calculate center of mass
        
        self.masscenter = np.array(ndi.measurements.center_of_mass(zimage))
        #print("center of mass: ", self.masscenter)
        
        # calculate z estimator
        
        self.focusSignal = np.sqrt(self.masscenter[0]**2 + self.masscenter[1]**2) #OJO aquí, a veces puede ser que vaya el signo menos, pero el original es signo mas
        #print("FocusSignal in center of mass:", self.focusSignal)       
        self.currentTime = ptime.time() - self.startTime
        
    @pyqtSlot(bool, bool)
    def single_z_correction(self, feedback_val, initial):
        if DEBUG:
                print("Inside single_z_correction")
        
        if initial:
                    
            if not self.camON:
                self.camON = True
                #--------------------------------
                #--------------------------------
                # OJO pensar esto
                #self.camera.start_live_video(framerate='20 Hz')
                time.sleep(0.200)
                    
            y0 = int(640-150)
            x0 = int(512-150)
            y1 = int(640+150)
            x1 = int(512+150)
            value = np.array([y0, x0, y1, x1])
            #self.camera._set_AOI(*value)
            
            self.reset()
            self.reset_data_arrays()
            
            time.sleep(0.200)
            
        
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
        # TO DO: fix calibration function
        
        self.focusTimer.stop()
        time.sleep(0.100)
        
        nsteps = 40
        xmin = 9.5  # in µm
        xmax = 10.5   # in µm
        xrange = xmax - xmin  
        
        calibData = np.zeros(40)
        xData = np.arange(xmin, xmax, xrange/nsteps)
        
        zMoveTo(self.actuator, xmin)
        
        time.sleep(0.100)
        
        for i in range(nsteps):
            
            zMoveTo(self.actuator, xmin + (i * 1/nsteps) * xrange)
            self.update()
            calibData[i] = self.focusSignal
            
        plt.plot(xData, calibData, 'o')
            
        time.sleep(0.200)
        
        self.focusTimer.start(self.focusTime)
    
            
    def reset(self):
        if DEBUG:
                print("Inside reset")
        
        self.data = np.zeros(self.npoints)
        self.time = np.zeros(self.npoints)
        self.ptr = 0
        self.startTime = ptime.time()

        self.max_dev = 0
        self.mean = self.focusSignal
        #print("focusSignal in reset: ", self.mean)
        self.std = 0
        self.n = 1
        
    def reset_data_arrays(self):
        if DEBUG:
                print("Inside reset_data_arrays")
        
        self.time_array = []
        self.z_array = []
        
    def export_data(self):
        if DEBUG:
                print("Inside export_data")
        
        fname = self.filename
        #case distinction to prevent wrong filenaming when starting minflux or psf measurement
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
        if DEBUG:
                print("Inside get_stop_signal")
        
        """
        From: [psf]
        Description: stops liveview, tracking, feedback if they where running to
        start the psf measurement with discrete xy - z corrections
        """
            
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
        
#        self.updateGUIcheckboxSignal.emit(self.tracking_value, 
#                                          self.feedback_active, 
#                                          self.save_data_state)
        
        print(datetime.now(), '[focus] System focus locked')
            
    @pyqtSlot(np.ndarray)
    def get_new_roi(self, val): #This is get_roi_info in other codes
        if DEBUG:
                print("Inside get_new_roi")
        '''
        Connection: [frontend] changedROI
        Description: gets coordinates of the ROI in the GUI
        
        '''
                
        self.ROIcoordinates = val.astype(int)
        #print("self.ROIcoordinates", self.ROIcoordinates)
        #print("TYPE self.ROIcoordinates", type(self.ROIcoordinates))
        if DEBUG:
            print(datetime.now(), '[focus] got ROI coordinates')
            
       # self.camera._set_AOI(*self.roi_area)
        
       # if DEBUG:
           # print(datetime.now(), '[focus] ROI changed to', self.camera._get_AOI())
    
    @pyqtSlot(bool, str)   
    def get_tcspc_signal(self, val, fname):
        if DEBUG:
                print("Inside get_tcspc_signal")
        
        """ 
        Get signal to start/stop xy position tracking and lock during 
        tcspc acquisition. It also gets the name of the tcspc file to produce
        the corresponding xy_data file
        
        bool val
        True: starts the tracking and feedback loop
        False: stops saving the data and exports the data during tcspc measurement
        tracking and feedback are not stopped automatically 
        
        """
        
        self.filename = fname
         
        if val is True:
            
            self.reset()
            self.reset_data_arrays()
            
            self.save_data_state = True
            self.toggle_feedback(True)
            self.save_data_state = True
            
        else:
            
            self.export_data()
            self.save_data_state = False
            
        # TO DO: fix updateGUIcheckboxSignal    
        
#        self.updateGUIcheckboxSignal.emit(self.tracking_value, 
#                                          self.feedback_active, 
#                                          self.save_data_state)
            
    @pyqtSlot(bool, str)   
    def get_scan_signal(self, val, fname):
        if DEBUG:
                print("Inside get_scan_signal")
        
        """ 
        Get signal to stop continous xy tracking/feedback if active and to
        go to discrete xy tracking/feedback mode if required
        """
        
    @pyqtSlot(bool)
    def get_save_data_state(self, val):
        if DEBUG:
                print("Inside get_save_data_state")
        
        self.save_data_state = val
        
    @pyqtSlot(str)    
    def get_end_measurement_signal(self, fname):
        if DEBUG:
                print("Inside get_end_measurement_signal")
        
        """ 
        From: [minflux] or [psf]
        Description: at the end of the measurement exports the xy data

        """ 
        
        self.filename = fname
        self.export_data()
        
        self.toggle_feedback(False)
        self.toggle_tracking(False)
        
        if self.camON:
            self.focusTimer.stop()
            self.liveviewSignal.emit(False)
            #print("self.liveviewSignal.emit(False) executed in get end measurement")
        
    def set_moveTo_param(self, x_f, y_f, z_f, n_pixels_x=128, n_pixels_y=128,
                         n_pixels_z=128, pixeltime=2000):

        if DEBUG:
            print("Inside set_moveTo_param")
        print("x_f in um before conversion: ", x_f)
        print("y_f in um before conversion: ", y_f)
        print("z_f in um before conversion: ", z_f)
        x_f = tools.convert(x_f, 'XtoU')
        y_f = tools.convert(y_f, 'XtoU')
        z_f = tools.convert(z_f, 'XtoU')
        print("x_f after conversion to ADwin units: ", x_f)
        print("y_f after conversion to ADwin units: ", y_f)
        print("z_f after conversion to ADwin units.: ", z_f)

        self.adw.Set_Par(21, n_pixels_x)
        self.adw.Set_Par(22, n_pixels_y)
        self.adw.Set_Par(23, n_pixels_z)

        self.adw.Set_FPar(23, x_f)
        self.adw.Set_FPar(24, y_f)
        self.adw.Set_FPar(25, z_f)

        self.adw.Set_FPar(26, tools.timeToADwin(pixeltime))

    def moveTo(self, x_f, y_f, z_f):
        if DEBUG:
                print("Inside moveTo - line 1173")
        #print("x_f, y_f, z_f: ", x_f, y_f, z_f)
        self.set_moveTo_param(x_f, y_f, z_f)
        self.adw.Start_Process(2)
        
    def gaussian_fit(self):
        if DEBUG:
                print("Inside gaussian_fit")
        
        # set main reference frame
        
        ymin, xmin, ymax, xmax = self.roi_area
        ymin_nm, xmin_nm, ymax_nm, xmax_nm = self.roi_area * self.pxSize
        
        # select the data of the image corresponding to the ROI

        array = self.image #[xmin:xmax, ymin:ymax]
        
        # set new reference frame
        
        xrange_nm = xmax_nm - xmin_nm
        yrange_nm = ymax_nm - ymin_nm
             
        x_nm = np.arange(0, xrange_nm, self.pxSize)
        y_nm = np.arange(0, yrange_nm, self.pxSize)
        
        (Mx_nm, My_nm) = np.meshgrid(x_nm, y_nm)
        
        # find max 
        
        argmax = np.unravel_index(np.argmax(array, axis=None), array.shape)
        
        x_center_id = argmax[0]
        y_center_id = argmax[1]
        
        # define area around maximum
    
        xrange = 10 # in px
        yrange = 10 # in px
        
        xmin_id = int(x_center_id-xrange)
        xmax_id = int(x_center_id+xrange)
        
        ymin_id = int(y_center_id-yrange)
        ymax_id = int(y_center_id+yrange)
        
        array_sub = array[xmin_id:xmax_id, ymin_id:ymax_id]
                
        xsubsize = 2 * xrange
        ysubsize = 2 * yrange
        
#        plt.imshow(array_sub, cmap=cmaps.parula, interpolation='None')
        
        x_sub_nm = np.arange(0, xsubsize) * self.pxSize
        y_sub_nm = np.arange(0, ysubsize) * self.pxSize

        [Mx_sub, My_sub] = np.meshgrid(x_sub_nm, y_sub_nm)
        
        # make initial guess for parameters
        
        bkg = np.min(array)
        A = np.max(array) - bkg
        σ = 400 # nm
        x0 = x_sub_nm[int(xsubsize/2)]
        y0 = y_sub_nm[int(ysubsize/2)]
        
        initial_guess_G = [A, x0, y0, σ, σ, bkg]
         
        poptG, pcovG = opt.curve_fit(PSF.gaussian2D, (Mx_sub, My_sub), 
                                     array_sub.ravel(), p0=initial_guess_G)
        
        # retrieve results

        poptG = np.around(poptG, 2)
    
        A, x0, y0, σ_x, σ_y, bkg = poptG
                
#        x = x0 + Mx_nm[xmin_id, ymin_id]
#        y = y0 + My_nm[xmin_id, ymin_id]
#        
##        self.currentx = x
##        self.currenty = y
#        
#        # if to avoid (probably) false localizations
#        
#        maxdist = 200 # in nm
#        
#        if self.initial is False:
#        
#            if np.abs(x - self.currentx) < maxdist and np.abs(y - self.currenty) < maxdist:
#        
#                self.currentx = x
#                self.currenty = y
#                
##                print(datetime.now(), '[xy_tracking] normal')
#                
#            else:
#                
#                pass
#                
#                print(datetime.now(), '[xy_tracking] max dist exceeded')
#        
#        else:
#            
#            self.currentx = x
#            self.currenty = y
            
#            print(datetime.now(), '[xy_tracking] else')
       
        
    @pyqtSlot(float)
    def get_focuslockposition(self, position):
        if DEBUG:
                print("Inside get_focuslockposition")
        
        if position == -9999:
            position = self.setPoint
        else:
            position = self.focusSignal
            
        self.focuslockpositionSignal.emit(position)
        
    @pyqtSlot()
    def stop(self):
        if DEBUG:
                print("Inside stop")
        self.toggle_ir_shutter(8, False)
        time.sleep(1)
        
        self.focusTimer.stop()
        
        #prevent system to throw weird errors when not being able to close the camera, see uc480.py --> close()
#        try:
        self.reset()
#        except:
#            pass
        
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
        except:
            pass
        
        
    def make_connection(self, frontend):
        if DEBUG:
                print("Inside make_connection in Backend")
          
        frontend.changedROI.connect(self.get_new_roi)
        frontend.closeSignal.connect(self.stop)
#        frontend.lockFocusSignal.connect(self.lock_focus)
        frontend.feedbackLoopBox.stateChanged.connect(lambda: self.toggle_feedback(frontend.feedbackLoopBox.isChecked()))
        frontend.saveDataSignal.connect(self.get_save_data_state)
        frontend.exportDataButton.clicked.connect(self.export_data)
        frontend.clearDataButton.clicked.connect(self.reset)
        frontend.clearDataButton.clicked.connect(self.reset_data_arrays)
        frontend.calibrationButton.clicked.connect(self.calibrate)
        frontend.shutterCheckbox.stateChanged.connect(lambda: self.toggle_ir_shutter(8, frontend.shutterCheckbox.isChecked()))
        frontend.paramSignal.connect(self.get_frontend_param)
        frontend.liveviewButton.clicked.connect(self.liveview)


if __name__ == '__main__':
    if DEBUG:
        print("Inside main")
    
    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()
        
    #app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    print(datetime.now(), '[focus] Focus lock module running in stand-alone mode')
    
    # Initialize devices
    
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)
    
    #if camera wasnt closed properly just keep using it without opening new one

    try:
        cam = ids_cam.IDS_U3()
    except:
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
    
    worker.moveTo(10, 10, 10) # in µm

    gui.setWindowTitle('Focus lock')
    gui.resize(1500, 500)

    gui.show()
    app.exec_()
        