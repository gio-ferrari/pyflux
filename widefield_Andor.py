# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:06:59 2024

@author: Florencia D. Choque
"""

import numpy as np
import time
import ctypes as ct
from datetime import date, datetime

import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime
from scipy import optimize as opt
from PIL import Image
import tifffile
import os

import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
import tools.PSF as PSF
import tools.tools as tools
import scan

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import QGroupBox
import qdarkstyle

#from pylablib.devices import Andor
from pylablib.devices.Andor import AndorSDK2
import drivers.ADwin as ADwin

DEBUG = True

PX_SIZE = 133.0 #px size of camera in nm

class Frontend(QtGui.QFrame):
    
    roiInfoSignal = pyqtSignal(int, np.ndarray)
    closeSignal = pyqtSignal()
    saveDataSignal = pyqtSignal(bool)
    saveImageSignal = pyqtSignal()
    startRecordingSignal = pyqtSignal(float)
    """
    Signals
             
    - roiInfoSignal:
         To: [backend] get_roi_info
        
    - closeSignal:
         To: [backend] stop
        
    - saveDataSignal:
         To: [backend] get_save_data_state
        
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.setup_gui()
        
        # initial ROI parameters        
        self.NofPixels = 50
        self.roi = None
        self.ROInumber = 0
        self.roilist = []

    def create_roi(self):
        ROIpen = pg.mkPen(color='r')
        ROIpos = (150,210) #These hardcoded values fit the confocal beam for minflux
        self.roi = viewbox_tools.ROI2(self.NofPixels, self.vb, ROIpos,
                                     handlePos=(1, 0),
                                     handleCenter=(0, 1),
                                     scaleSnap=True,
                                     translateSnap=True,
                                     pen=ROIpen, number=self.ROInumber)
        
        self.ROInumber += 1
        self.roilist.append(self.roi)
        self.ROIButton.setChecked(False)
    
    def emit_roi_info(self):
        roinumber = len(self.roilist)
        if roinumber == 0:
            print(datetime.now(), 'Please select a valid ROI')
        else:
            coordinates = np.zeros((4))
            for i in range(len(self.roilist)):
                xmin, ymin = self.roilist[i].pos()
                xmax, ymax = self.roilist[i].pos() + self.roilist[i].size()
                coordinates = np.array([xmin, xmax, ymin, ymax])  
            self.roiInfoSignal.emit(roinumber, coordinates)

    def delete_roi(self):
        for i in range(len(self.roilist)):
            self.vb.removeItem(self.roilist[i])
            self.roilist[i].hide()
        self.roilist = []
        self.delete_roiButton.setChecked(False)
        self.ROInumber = 0
     
    @pyqtSlot(bool)
    def toggle_liveview(self, on):
        if on:
            self.liveviewButton.setChecked(True)
            print(datetime.now(), 'Live view started')
        else:
            self.liveviewButton.setChecked(False)
            self.emit_roi_info() #Check if it is necessary to put this here
            self.img.setImage(np.zeros((512,512)), autoLevels=False)
            print(datetime.now(), 'Live view stopped')
        
    @pyqtSlot()  
    def get_roi_request(self):
        print(datetime.now(), 'got ROI request')
        self.emit_roi_info()
        
    @pyqtSlot(np.ndarray)
    def get_image(self, img):
        self.img.setImage(img, autoLevels=False)
        self.xaxis.setScale(scale=PX_SIZE/1000) #scale to µm
        self.yaxis.setScale(scale=PX_SIZE/1000) #scale to µm
        
    def on_save_button_clicked(self):
        self.saveImageSignal.emit()
        
    def emit_start_recording_signal(self):
        """Envía la señal para iniciar la grabación."""
        try:
            duration = float(self.durationEdit.text())
            self.startRecordingSignal.emit(duration)
            self.progressBar.setValue(0)  # Reinicia la barra de progreso
            self.progressBar.setVisible(True) 
        except ValueError:
            print("Por favor, ingresa un número válido para la duración.")
    
    @pyqtSlot(int)
    def update_progress(self, progress):
        """
        Actualiza la barra de progreso según el tiempo transcurrido de la grabación.
        """
        if progress is not None:
            self.progressBar.setValue(progress)
        if progress == 100:
            self.progressBar.setValue(100)
            print("Grabación finalizada.")

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
        if (num == 6)  or (num == 8):
            self.shutterCheckbox.setChecked(on)
                    
    @pyqtSlot(bool)
    def get_backend_states(self, savedata):
        self.saveDataBox.setChecked(savedata)            

    def emit_save_data_state(self):
        if self.saveDataBox.isChecked(): 
            self.saveDataSignal.emit(True)
            self.emit_roi_info() 
        else:
            self.saveDataSignal.emit(False)
        
    def make_connection(self, backend):
        backend.changedImage.connect(self.get_image)
        backend.updateGUIcheckboxSignal.connect(self.get_backend_states)
        backend.shuttermodeSignal.connect(self.update_shutter)
        backend.liveviewSignal.connect(self.toggle_liveview)
        backend.progressBarSignal.connect(self.update_progress)
        
    def setup_gui(self):
        # GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
    
        # Image widget layout
        imageWidget = pg.GraphicsLayoutWidget()
        imageWidget.setMinimumHeight(350)
        imageWidget.setMinimumWidth(350)
    
        # Setup axis for scaling
        self.xaxis = pg.AxisItem(orientation='bottom', maxTickLength=5)
        self.xaxis.showLabel(show=True)
        self.xaxis.setLabel('x', units='µm')
    
        self.yaxis = pg.AxisItem(orientation='left', maxTickLength=5)
        self.yaxis.showLabel(show=True)
        self.yaxis.setLabel('y', units='µm')
    
        self.vb = imageWidget.addPlot(axisItems={'bottom': self.xaxis, 
                                                 'left': self.yaxis})
        self.vb.setAspectLocked(True)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        imageWidget.setAspectLocked(True)
    
        gridItem = pg.GridItem()
        self.vb.addItem(gridItem)
        grid.addWidget(imageWidget, 0, 0, 2, 1)  # Spanning two rows
    
        # Set up histogram for the liveview image
        self.hist = pg.HistogramLUTItem(image=self.img)
        self.hist.gradient.loadPreset('magma')
        self.hist.vb.setLimits(yMin=0, yMax=5000)
        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)
    
        # Parameters widget
        self.paramWidget = QGroupBox('Parameters')
        self.paramWidget.setFixedHeight(260)
        self.paramWidget.setFixedWidth(250)
    
        param_layout = QtGui.QGridLayout()
        self.paramWidget.setLayout(param_layout)
    
        # Populate paramWidget
        self.liveviewButton = QtGui.QPushButton('Camera LIVEVIEW')
        self.liveviewButton.setCheckable(True)
    
        self.ROIButton = QtGui.QPushButton('ROI')
        self.ROIButton.setCheckable(True)
        self.ROIButton.clicked.connect(self.create_roi)
    
        self.selectROIbutton = QtGui.QPushButton('Select ROI')
        self.selectROIbutton.clicked.connect(self.emit_roi_info)
    
        self.delete_roiButton = QtGui.QPushButton('Delete ROIs')
        self.delete_roiButton.clicked.connect(self.delete_roi)
    
        self.saveFrameButton = QtGui.QPushButton('Save Frame')
        self.saveFrameButton.clicked.connect(self.on_save_button_clicked)
    
        self.selectAreaBox = QtGui.QCheckBox('Select Area to APD')
        self.selectAreaBox.stateChanged.connect(self.emit_roi_info)
    
        self.saveDataBox = QtGui.QCheckBox("Save data")
        self.saveDataBox.stateChanged.connect(self.emit_save_data_state)
    
        self.shutterLabel = QtGui.QLabel('Shutter open?')
        self.shutterCheckbox = QtGui.QCheckBox('MinilasEvo 640')
    
        param_layout.addWidget(self.liveviewButton, 0, 0)
        param_layout.addWidget(self.ROIButton, 1, 0)
        param_layout.addWidget(self.selectROIbutton, 2, 0)
        param_layout.addWidget(self.delete_roiButton, 3, 0)
        param_layout.addWidget(self.saveFrameButton, 4, 0)
        param_layout.addWidget(self.selectAreaBox, 1, 1)
        param_layout.addWidget(self.saveDataBox, 2, 1)
        param_layout.addWidget(self.shutterLabel, 7, 0)
        param_layout.addWidget(self.shutterCheckbox, 7, 1)
    
        # Video acquisition widget
        self.videoWidget = QGroupBox('Video Acquisition Settings')
    
        video_layout = QtGui.QGridLayout()
        self.videoWidget.setLayout(video_layout)
    
        self.durationLabel = QtGui.QLabel('Time [min]')
        self.durationEdit = QtGui.QLineEdit('1')
        self.startVideoButton = QtGui.QPushButton('Start Video')
        self.startVideoButton.setCheckable(True)
        self.startVideoButton.clicked.connect(self.emit_start_recording_signal)
    
        self.progressBar = QtGui.QProgressBar(self)
        self.progressBar.setRange(0, 100)  # range: (0-100%)
        self.progressBar.setValue(0)
    
        video_layout.addWidget(self.durationLabel, 0, 0)
        video_layout.addWidget(self.durationEdit, 0, 1)
        video_layout.addWidget(self.startVideoButton, 1, 0)
        video_layout.addWidget(self.progressBar, 1, 1)
    
        # Container widget for paramWidget and videoWidget
        side_widget = QtGui.QWidget()
        side_layout = QtGui.QVBoxLayout(side_widget)
        side_layout.addWidget(self.paramWidget)
        side_layout.addWidget(self.videoWidget)
    
        # Add side_widget to grid
        grid.addWidget(side_widget, 0, 1, 2, 1)  # Spanning two rows
        self.liveviewButton.clicked.connect(lambda: self.toggle_liveview(self.liveviewButton.isChecked()))

        
    # def setup_gui(self):
    #     # GUI layout
    #     grid = QtGui.QGridLayout()
    #     self.setLayout(grid)
        
    #     # parameters widget
    #     self.paramWidget = QGroupBox('Parameters')   
    #     self.paramWidget.setFixedHeight(260)
    #     self.paramWidget.setFixedWidth(250)
    #     grid.addWidget(self.paramWidget, 0, 1)
        
    #     # image widget layout
    #     imageWidget = pg.GraphicsLayoutWidget()
    #     imageWidget.setMinimumHeight(350)
    #     imageWidget.setMinimumWidth(350)
        
    #     # setup axis, for scaling see get_image()
    #     self.xaxis = pg.AxisItem(orientation='bottom', maxTickLength=5)
    #     self.xaxis.showLabel(show=True)
    #     self.xaxis.setLabel('x', units='µm')
        
    #     self.yaxis = pg.AxisItem(orientation='left', maxTickLength=5)
    #     self.yaxis.showLabel(show=True)
    #     self.yaxis.setLabel('y', units='µm')
        
    #     self.vb = imageWidget.addPlot(axisItems={'bottom': self.xaxis, 
    #                                              'left': self.yaxis})
    
    #     self.vb.setAspectLocked(True)
    #     self.img = pg.ImageItem()
    #     self.img.translate(-0.5, -0.5)
    #     self.vb.addItem(self.img)
    #     imageWidget.setAspectLocked(True)
        
    #     gridItem = pg.GridItem()
    #     self.vb.addItem(gridItem)
    #     grid.addWidget(imageWidget, 0, 0)
        
    #     # set up histogram for the liveview image
    #     self.hist = pg.HistogramLUTItem(image=self.img)
    #     self.hist.gradient.loadPreset('magma')
    #     self.hist.vb.setLimits(yMin=0, yMax=5000)
    #     for tick in self.hist.gradient.ticks:
    #         tick.hide()
    #     imageWidget.addItem(self.hist, row=0, col=1)
        
    #     ##To paramWidget
    #     # LiveView Button
    #     self.liveviewButton = QtGui.QPushButton('Camera LIVEVIEW')
    #     self.liveviewButton.setCheckable(True)
        
    #     # create ROI button
    #     self.ROIButton = QtGui.QPushButton('ROI')
    #     self.ROIButton.setCheckable(True)
    #     self.ROIButton.clicked.connect(self.create_roi)
        
    #     # select ROI
    #     self.selectROIbutton = QtGui.QPushButton('Select ROI')
    #     self.selectROIbutton.clicked.connect(self.emit_roi_info)
        
    #     # delete ROI button
    #     self.delete_roiButton = QtGui.QPushButton('Delete ROIs')
    #     self.delete_roiButton.clicked.connect(self.delete_roi)
        
    #     # save current frame
    #     self.saveFrameButton = QtGui.QPushButton('Save Frame')
    #     self.saveFrameButton.clicked.connect(self.on_save_button_clicked)
    #     # select area checkbox
    #     self.selectAreaBox = QtGui.QCheckBox('Select Area to APD')
    #     self.selectAreaBox.stateChanged.connect(self.emit_roi_info)

    #     # save data signal
    #     self.saveDataBox = QtGui.QCheckBox("Save data")
    #     self.saveDataBox.stateChanged.connect(self.emit_save_data_state)
        
    #     #shutter button and label
    #     self.shutterLabel = QtGui.QLabel('Shutter open?')
    #     self.shutterCheckbox = QtGui.QCheckBox('MinilasEvo 640')

    #     # buttons and param layout
    #     subgrid = QtGui.QGridLayout()
    #     self.paramWidget.setLayout(subgrid)

    #     subgrid.addWidget(self.liveviewButton, 0, 0)
    #     subgrid.addWidget(self.ROIButton, 1, 0)
    #     subgrid.addWidget(self.selectROIbutton, 2, 0)
    #     subgrid.addWidget(self.delete_roiButton, 3, 0)
    #     subgrid.addWidget(self.saveFrameButton, 4, 0)
    #     subgrid.addWidget(self.selectAreaBox, 1, 1)
    #     subgrid.addWidget(self.saveDataBox, 2, 1)
    #     subgrid.addWidget(self.shutterLabel, 7, 0)
    #     subgrid.addWidget(self.shutterCheckbox, 7, 1)
        
    #     ## To videoWidget
    #     self.videoWidget = QGroupBox('Video Acquisition Settings')
    #     self.durationLabel = QtGui.QLabel('Time [min]')
    #     self.durationEdit = QtGui.QLineEdit('1')
    #     self.startVideoButton = QtGui.QPushButton('Start Video')
    #     self.startVideoButton.setCheckable(True)
    #     self.startVideoButton.clicked.connect(self.emit_start_recording_signal)
        
    #     self.progressBar = QtGui.QProgressBar(self)
    #     self.progressBar.setRange(0, 100)  # range: (0-100%)
    #     self.progressBar.setValue(0)
    #     subgrid = QtGui.QGridLayout()
    #     self.videoWidget.setLayout(subgrid)
        
    #     subgrid.addWidget(self.durationLabel, 0, 0)
    #     subgrid.addWidget(self.durationEdit, 0, 1)
    #     subgrid.addWidget(self.startVideoButton, 1, 0)
    #     subgrid.addWidget(self.progressBar, 1, 1)
    #     grid.addWidget(self.videoWidget, 1, 0)
        

    #     self.liveviewButton.clicked.connect(lambda: self.toggle_liveview(self.liveviewButton.isChecked()))
            
    def closeEvent(self, *args, **kwargs):
        self.closeSignal.emit()
        super().closeEvent(*args, **kwargs)
        app.quit()
        
class Backend(QtCore.QObject):
    
    changedImage = pyqtSignal(np.ndarray)
    updateGUIcheckboxSignal = pyqtSignal(bool)
    shuttermodeSignal = pyqtSignal(int, bool)
    liveviewSignal = pyqtSignal(bool)
    progressBarSignal = pyqtSignal(int)
    """
    Signals
    
    - changedImage:
        To: [frontend] get_image
        
    - updateGUIcheckboxSignal:
        To: [frontend] get_backend_states
        
    - shuttermodeSignal:
        To: [frontend] update_shutter

    """
    def __init__(self, andor, adw, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.andor = andor
        self.adw = adw
        self.initialize_camera()
        self.setup_camera()
        
        # folder
        today = str(date.today()).replace('-', '')  # TO DO: change to get folder from microscope
        root = r'C:\\Data\\'
        folder = root + today
        
        filename = r'\widefield'
        self.filename = folder + filename
        
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.update_view)
        
        self.save_data_state = False
        self.camON = False
        self.image = None
        self.is_recording = False
        self.frames = []
        self.recording_start_time = None
        self.video_duration = 0 
        
    def setup_camera(self): #TODO: Check if it is necesary to move this to another module
        
        self.shape = (512, 512) # TO DO: change to 256 x 256
        self.expTime = 0.05   # in sec
        
        self.andor.set_exposure(self.expTime)
        self.andor.set_roi(0, 512, 0, 512, 1, 1) #This one could be a parameter like self.shape above
        print(datetime.now(), 'Andor FOV size = {}'.format(self.andor.get_roi()))

        # Temperature
        self.andor.set_cooler(True)
        self.andor.set_temperature(-20)   # in °C
        self.andor.set_acquisition_mode('cont')

        # Frame transfer mode
        self.andor.enable_frame_transfer_mode(True)
        # print(datetime.now(), 'Frame transfer mode =', self.andor.enable_frame_transfer_mode())
    
        #Horizontal Pixel Shift
        channel = 0 #channel_bitdepth = 14
        oamp = 0 #oamp_kind = 'Electron Multiplying' (output amplifier description)
        hsspeed = 1 # hsspeed_MHz = 5.0 (horizontal scan frequency corresponding to the given hsspeed index)
        preamp = 2 # preamp_gain=2.4000000953674316 #OJO: antes se usaba 4.7, con indice 2, chequear cómo se ve la imagen con un mayor preamp, puedo usar preamp = 2 y el cambio está en preamp_gain=5.099999904632568
    
        self.andor.set_amp_mode(channel, oamp, hsspeed, preamp)
        print("Current Amp mode: ", self.andor.get_amp_mode())
        
        # EM GAIN
        EM_gain = 10  #EM gain set to 100
        self.andor.set_EMCCD_gain(EM_gain) #Check I'm not sure about units (indexes???)
        print(datetime.now(), 'EM gain: ', self.andor.get_EMCCD_gain())
        
        # Vertical shift speed
        vert_shift_speed = 4 #µs
        self.andor.set_vsspeed(vert_shift_speed)
        print(datetime.now(), 'Current Vertical shift speed [µs]: ', self.andor.get_vsspeed())
            
    def initialize_camera(self):
        #self.andor.open() #Creo que esto no es necesario porque open se ejecuta cuando de inicializa el contructor
        print("Is Andor opened?", self.andor.is_opened())
        print(datetime.now(),'Device info:', self.andor.get_device_info())

    @pyqtSlot(int, bool)
    def toggle_tracking_shutter(self, num, val):
        #TODO: change code to also update checkboxes in case of minflux measurement
        if (num == 6)  or (num == 8):
            if val:
                tools.toggle_shutter(self.adw, 6, True)
                print(datetime.now(), '[xy_tracking] Tracking shutter opened')
            else:
                tools.toggle_shutter(self.adw, 6, False)
                print(datetime.now(), '[xy_tracking] Tracking shutter closed')
   
    @pyqtSlot(int, bool)
    def shutter_handler(self, num, on):
        self.shuttermodeSignal.emit(num, on)
        
    @pyqtSlot(bool)
    def liveview(self, value):
        '''
        Connection: [frontend] liveviewSignal
        Description: toggles start/stop the liveview of the camera.
        
        '''
        if value:
            self.camON = True
            self.liveview_start() 
        else:
            self.liveview_stop()
            self.camON = False
        
    def liveview_start(self):
        self.initial = True
        print("Andor current temperature: ", self.andor.get_temperature())
        print("Andor temperature status: ", self.andor.get_temperature_status())
        
        # Initial image
        print(datetime.now(), 'Acquisition mode:', self.andor.get_acquisition_mode())
        # self.andor.setup_shutter("open",ttl_mode=1) #No need to use shutter
        self.andor.start_acquisition()
        
        time.sleep(self.expTime * 2) #Capaz tenga que sacar esta linea
        self.andor.wait_for_frame() 
        self.image = self.andor.read_newest_image()
        self.changedImage.emit(self.image)
        self.viewtimer.start(200) # DON'T USE time.sleep() inside the update()
                                  # 400 ms ~ acq time + gaussian fit time
    
    def liveview_stop(self):
        self.viewtimer.stop()
        self.andor.stop_acquisition()
                    
    def update_view(self):
        """ Image update while in Liveview mode """
        if not self.andor.wait_for_frame():
            print("Wait for frame fallo")
            return
        self.image = self.andor.read_newest_image()
        if self.image is None:
            print("No hay imagen")
            return
        self.changedImage.emit(self.image)
        if self.is_recording:
            elapsed_time = time.time() - self.recording_start_time
            progress = int((elapsed_time / self.video_duration) * 100) #This is to update progress bar in front
            self.frames.append(self.image) 
            if elapsed_time >= self.video_duration:
                self.stop_recording()
            self.progressBarSignal.emit(progress)
        return None
    def start_recording(self, duration):
        """
        Inicia la grabación del video por la duración especificada.
        """
        if not self.is_recording:
            self.is_recording = True
            self.video_duration = duration*60 # min
            self.recording_start_time = time.time()
            self.frames = []
            print(f"Grabación iniciada por {duration} minutos.")
        else:
            print("Ya hay una grabación en curso.")

    def stop_recording(self):
        """
        Detiene la grabación y guarda el video.
        """
        if self.is_recording:
            self.is_recording = False
            self.save_video()
            print("Grabación finalizada.")

    def save_video(self):
        """
        Guarda los frames capturados como un stack TIFF.
        """
        if not self.frames:
            print("No se capturaron frames para guardar.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"_video_{timestamp}.tiff"
        tiff_filename = self.filename + filename
        tifffile.imsave(tiff_filename, self.frames) 
        print(f"Video guardado como {tiff_filename}")

    def save_current_frame(self):
        """Guarda la imagen actual en formatos .npy y .tiff."""
        filename = self.filename + '_frame'
        suffix = 1
        while os.path.exists(f"{filename}_{suffix}.npy") or os.path.exists(f"{filename}_{suffix}.tiff"):
            suffix += 1
        
        filename = f"{filename}_{suffix}"
        if self.image is not None:
            np.save(filename, self.image)
            Image.fromarray(self.image).save(f"{filename}.tiff")
            print(f"Imagen guardada en {filename}.tiff y {filename}.npy")
        else:
            print("No hay imagen disponible para guardar.")
        
    def set_actuator_param(self, pixeltime=1000):
        self.adw.Set_FPar(46, tools.timeToADwin(pixeltime))
        
        # set-up actuator initial param
        currentXposition = tools.convert(self.adw.Get_FPar(70), 'UtoX')
        currentYposition = tools.convert(self.adw.Get_FPar(71), 'UtoX')
        x_f = tools.convert(currentXposition, 'XtoU')
        y_f = tools.convert(currentYposition, 'XtoU')
        
        self.adw.Set_FPar(40, x_f)
        self.adw.Set_FPar(41, y_f)
        self.adw.Set_Par(40, 1)
        
    def actuator_xy(self, x_f, y_f):
        x_f = tools.convert(x_f, 'XtoU')
        y_f = tools.convert(y_f, 'XtoU')
        
        self.adw.Set_FPar(40, x_f)
        self.adw.Set_FPar(41, y_f)
        self.adw.Set_Par(40, 1)    
        
    def set_moveTo_param(self, x_f, y_f, z_f, n_pixels_x=128, n_pixels_y=128,
                         n_pixels_z=128, pixeltime=2000):

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

    def export_data(self):
        """
        Exports the x, y and t data into a .txt file
        """
#        fname = self.filename
##        filename = tools.getUniqueName(fname)    # TO DO: make compatible with psf measurement and stand alone
#        filename = fname + '_xydata.txt'
        fname = self.filename
        #case distinction to prevent wrong filenaming when starting minflux or psf measurement
        if fname[0] == '!':
            filename = fname[1:]
        else:
            filename = tools.getUniqueName(fname)
        filename = filename + '_xydata.txt'
        size = self.j
        savedData = np.zeros((3, size))
        savedData[0, :] = self.time_array[0:self.j]
        savedData[1, :] = self.x_array[0:self.j]
        savedData[2, :] = self.y_array[0:self.j]
        np.savetxt(filename, savedData.T,  header='t (s), x (nm), y(nm)') # transpose for easier loading
        print(datetime.now(), 'Data exported to', filename)

    @pyqtSlot(bool)
    def get_save_data_state(self, val):
        '''
        Connection: [frontend] saveDataSignal
        Description: gets value of the save_data_state variable, True -> save,
        Fals -> don't save
        '''
        self.save_data_state = val
        if DEBUG:
            print(datetime.now(), 'Save_data_state = {}'.format(val))
    
    @pyqtSlot(int, np.ndarray)
    def get_roi_info(self, N, coordinates_array):
        '''
        Connection: [frontend] roiInfoSignal
        Description: gets coordinates of the ROI in the GUI
        '''
        self.ROIcoordinates = coordinates_array.astype(int)
        if DEBUG:
            print(datetime.now(), '[Andor Widefield] got ROI coordinates')
            
    def make_connection(self, frontend):
        frontend.roiInfoSignal.connect(self.get_roi_info)
        frontend.closeSignal.connect(self.stop)
        frontend.saveDataSignal.connect(self.get_save_data_state)
        frontend.saveImageSignal.connect(self.save_current_frame)
        frontend.startRecordingSignal.connect(self.start_recording)
        frontend.liveviewButton.clicked.connect(self.liveview)
        
    @pyqtSlot()    
    def stop(self):
        self.viewtimer.stop()
        if self.camON:
            self.andor.stop_acquisition()
            print('Andor acquisition status: ', self.andor.get_status(), ". Idle: no acquisition")
        # self.andor.setup_shutter("closed",ttl_mode=0) #No need to modify shutter state
        self.andor.close()
        print(datetime.now(),"Is Andor opened? ", self.andor.is_opened())
        
        # Go back to 0 position
        x_0 = 0
        y_0 = 0
        z_0 = 0
        self.moveTo(x_0, y_0, z_0)
        self.toggle_tracking_shutter(8, False)

if __name__ == '__main__':

    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()
        
    #app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    andor = AndorSDK2.AndorSDK2Camera(fan_mode = "full") #Forma antigua
    #andor = Andor.AndorSDK2Camera(fan_mode = "full")
    
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)
    
    gui = Frontend()
    worker = Backend(andor, adw)
    
    gui.make_connection(worker)
    worker.make_connection(gui)
    
    # initialize fpar_70, fpar_71, fpar_72 ADwin position parameters
        
    pos_zero = tools.convert(0, 'XtoU')
        
    worker.adw.Set_FPar(70, pos_zero)
    worker.adw.Set_FPar(71, pos_zero)
    worker.adw.Set_FPar(72, pos_zero)
    
    worker.moveTo(10, 10, 10) # in µm
    
    time.sleep(0.200)
        
    gui.setWindowTitle('Widefield Andor')
    gui.show()
    app.exec_()
        
