# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:51:13 2023

@author: Florencia D. Choque based on xyz_tracking for RASTMIN by Luciano Masullo
Modified to work with new stabilization in p-MINFLUX, IDS_U3 cam and ADwin by Florencia D. Choque
Based on xyz_tracking by Luciano Masullo
"""

import numpy as np
import time
import scipy.ndimage as ndi
import ctypes as ct
import matplotlib.pyplot as plt
from datetime import date, datetime

from scipy import optimize as opt
from PIL import Image

import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
import tools.PSF as PSF
import tools.tools as tools

import scan

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtWidgets import QGroupBox

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime
import qdarkstyle

import drivers.ADwin as ADwin
import drivers.ids_cam as ids_cam #Is it necessary to modify expousure time and gain in driver ids_cam? FC

DEBUG = True
DEBUG1 = True
VIDEO = False
#to commit
PX_SIZE = 33.5 #px size of camera in nm #antes 80.0 para Andor
PX_Z = 100 # 202 nm/px for z in nm //Thorcam px size 25nm // IDS px size 50nm 

def actuatorParameters(adwin, z_f, n_pixels_z=50, pixeltime=1000): #funciones necesarias para calibrate

    z_f = tools.convert(z_f, 'XtoU')

    adwin.Set_Par(33, n_pixels_z)
    adwin.Set_FPar(35, z_f)
    adwin.Set_FPar(36, tools.timeToADwin(pixeltime))

def zMoveTo(adwin, z_f): #funciones necesarias para calibrate

    actuatorParameters(adwin, z_f)
    adwin.Start_Process(3)
    
class Frontend(QtGui.QFrame):
    #Chequear las señales aquí, hace falta changedROI (creo que es analogo a z_roiInfoSignal, cf), paramSignal???
    roiInfoSignal = pyqtSignal(str, int, list) #antes era (int, np.ndarray) cf xy_tracking Ver cómo afecta esto al procesamiento, porque parece estar bien al ser como en xyz_tracking

    z_roiInfoSignal = pyqtSignal(str, int, list) #señal, contrastar con focus.py
    closeSignal = pyqtSignal()
    saveDataSignal = pyqtSignal(bool)
    
    """
    Signals
             
    - roiInfoSignal:
         To: [backend] get_roi_info
         
    - z_roiInfoSignal:
         To: [backend] get_roi_info
        
    - closeSignal:
         To: [backend] stop
        
    - saveDataSignal:
         To: [backend] get_save_data_state
        
    """
    
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        # initial ROI parameters        
        
        self.ROInumber = 0
        self.roilist = [] #Una lista en la que se guardarán las coordenadas del ROI
        self.roi = None
        self.xCurve = None
        
        
        self.setup_gui()
        
      
    def craete_roi(self, roi_type):
        
        if DEBUG1:
             print("Estoy en craete_roi")
             
        if roi_type == 'xy':
        
            ROIpen = pg.mkPen(color='r')
    
            ROIpos = (512 - 64, 512 - 64)
            roi = viewbox_tools.ROI2(50, self.vb, ROIpos, handlePos=(1, 0),
                                     handleCenter=(0, 1),
                                     scaleSnap=True,
                                     translateSnap=True,
                                     pen=ROIpen, number=self.ROInumber)
            
            self.ROInumber += 1
            self.roilist.append(roi)
            self.xyROIButton.setChecked(False)
            
        if roi_type == 'z':
            
            ROIpen = pg.mkPen(color='y')
            
            ROIpos = (512 - 64, 512 - 64)
            self.roi_z = viewbox_tools.ROI2(140, self.vb, ROIpos,
                                            handlePos=(1, 0),
                                            handleCenter=(0, 1),
                                            scaleSnap=True,
                                            translateSnap=True,
                                            pen=ROIpen, number=self.ROInumber)
            
            self.zROIButton.setChecked(False)

    def emit_roi_info(self, roi_type):
        
        if DEBUG1:
             print("Estoy en emit_roi_info")
        
        if roi_type == 'xy':
        
            roinumber = len(self.roilist)
            
            if roinumber == 0:
                
                print(datetime.now(), '[xy_tracking] Please select a valid ROI for fiducial NPs tracking')
                
            else:
                
                coordinates = np.zeros((4))
                coordinates_list = []
                
                for i in range(len(self.roilist)):
                    
                    xmin, ymin = self.roilist[i].pos()
                    xmax, ymax = self.roilist[i].pos() + self.roilist[i].size()
            
                    coordinates = np.array([xmin, xmax, ymin, ymax])  
                    coordinates_list.append(coordinates)
                                                            
                self.roiInfoSignal.emit('xy', roinumber, coordinates_list)
                if DEBUG1:
                    print("Coordenadas xy: ", coordinates_list)
                    
        if roi_type == 'z':
            
            xmin, ymin = self.roi_z.pos()
            xmax, ymax = self.roi_z.pos() + self.roi_z.size()
            
            coordinates = np.array([xmin, xmax, ymin, ymax]) 
            coordinates_list = [coordinates]
            
            self.z_roiInfoSignal.emit('z', 0, coordinates_list)
            if DEBUG1:
                print("Coordenadas z: ", coordinates_list)
            
        if DEBUG1:
                print("roiInfoSignal.emit executed, signal from Frontend (function:emit_roi_info, to Backend:get_roi info FC")

# =============================================================================
#     def delete_roi_z(self): #elimina todas las ROI de la lista, antes era delete_roi
#     ############################################################
#     ########################################chequear esto para que funcione eliminando el roi de z
#         if DEBUG1:
#             print("EStoy en delete_roi_z")
#         
#         for i in range(len(self.roi_z)):
#             
#             self.vb.removeItem(self.roi_z)
#             self.roi_z.hide()
#             
#         self.roi_z = []
#         self.delete_roi_zButton.setChecked(False)
#         self.ROInumber = 0
# =============================================================================
    
    def delete_roi(self): #elimina solo la última ROI
                
        if DEBUG1:
            print("EStoy en delete_roi")
            
        self.vb.removeItem(self.roilist[-1])
        self.roilist[-1].hide()
        self.roilist = self.roilist[:-1]
        self.ROInumber -= 1
     
    @pyqtSlot(bool) #no toco esta función FC
    def toggle_liveview(self, on):
        if DEBUG1:
            print("EStoy en toggle_liveview")

        if on:
            self.liveviewButton.setChecked(True)
            print(datetime.now(), '[xy_tracking] Live view started')
        else:
            self.liveviewButton.setChecked(False)
            self.emit_roi_info('xy')
            self.img.setImage(np.zeros((512, 512)), autoLevels=False)
            print(datetime.now(), '[xy_tracking] Live view stopped')
        
    @pyqtSlot()  #Esta función no existe en xyz, chequear
    def get_roi_request(self):
        if DEBUG1:
            print("Estoy en get_roi_request")
        
        print(datetime.now(), '[xy_tracking] got ROI request')
        
        self.emit_roi_info('xy')
        
    @pyqtSlot(np.ndarray)
    def get_image(self, img):
        
#        if DEBUG:
#            print(datetime.now(),'[xy_tracking-frontend] got image signal')

        self.img.setImage(img, autoLevels=False)
        
        self.xaxis.setScale(scale=PX_SIZE/1000) #scale to µm
        self.yaxis.setScale(scale=PX_SIZE/1000) #scale to µm
        
        
    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray) #Cambió la señal changed_data 
    def get_data(self, tData, xData, yData, zData, avgIntData): #cambiaron los parámetros del slot get_data
        
        #print("xData: ",xData)
        #print("yData: ",yData)
        #print("zData: ",zData)

        N_NP = np.shape(xData)[1]
        #print("N_NP: ",N_NP)
        
        # x data
        
        for i in range(N_NP):
        
            self.xCurve[i].setData(tData, xData[:, i])
            
        self.xmeanCurve.setData(tData, np.mean(xData, axis=1))
        
        # y data
            
        for i in range(N_NP):
        
            self.yCurve[i].setData(tData, yData[:, i])
            
        self.ymeanCurve.setData(tData, np.mean(yData, axis=1))
        
        # z data
        
        self.zCurve.setData(tData, zData)
        
        # avg intensity data
        
        self.avgIntCurve.setData(avgIntData)
        
        # set xy 2D data
        
        self.xyDataItem.setData(np.mean(xData, axis=1), np.mean(yData, axis=1))
        #los cambios aquí tienen que verse reflejados en la gui, histogramas
        if len(xData) > 2:
            
            self.plot_ellipse(xData, yData)
            
            hist, bin_edges = np.histogram(zData, bins=60)
            self.zHist.setOpts(x=bin_edges[:-1], height=hist)
             
            xstd = np.std(np.mean(xData, axis=1))
            # print("mean x", np.mean(xData, axis=1))
            # print("size xData", np.size(xData))
            self.xstd_value.setText(str(np.around(xstd, 2)))
            
            ystd = np.std(np.mean(yData, axis=1))
            # print("mean y", np.mean(yData, axis=1))
            # print("size yData", np.size(yData))
            self.ystd_value.setText(str(np.around(ystd, 2)))
            
            zstd = np.std(zData)
            # print("size zData", np.size(zData))
            # print("std values x y z",xstd," ",ystd," ",zstd)
            self.zstd_value.setText(str(np.around(zstd, 2)))
        
    def plot_ellipse(self, x_array, y_array):
        
        pass
        
#            cov = np.cov(x_array, y_array)
#            
#            a, b, theta = tools.cov_ellipse(cov, q=.683)
#            
#            theta = theta + np.pi/2            
##            print(a, b, theta)
#            
#            xmean = np.mean(xData)
#            ymean = np.mean(yData)
#            
#            t = np.linspace(0, 2 * np.pi, 1000)
#            
#            c, s = np.cos(theta), np.sin(theta)
#            R = np.array(((c, -s), (s, c)))
#            
#            coord = np.array([a * np.cos(t), b * np.sin(t)])
#            
#            coord_rot = np.dot(R, coord)
#            
#            x = coord_rot[0] + xmean
#            y = coord_rot[1] + ymean
            
            # TO DO: fix plot of ellipse
            
#            self.xyDataEllipse.setData(x, y)
#            self.xyDataMean.setData([xmean], [ymean])

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
                    
    @pyqtSlot(bool, bool, bool)
    def get_backend_states(self, tracking, feedback, savedata): #Chequear si necesito esto, cf xyz

        self.trackingBeadsBox.setChecked(tracking)
        self.feedbackLoopBox.setChecked(feedback)
        self.saveDataBox.setChecked(savedata)            

    def emit_save_data_state(self):
        
        if self.saveDataBox.isChecked():
            
            self.saveDataSignal.emit(True)
            self.emit_roi_info() 
            
        else:
            
            self.saveDataSignal.emit(False)
        
    def make_connection(self, backend):
            
        backend.changedImage.connect(self.get_image)
        backend.changedData.connect(self.get_data)
        backend.updateGUIcheckboxSignal.connect(self.get_backend_states)
        backend.shuttermodeSignal.connect(self.update_shutter)
        backend.liveviewSignal.connect(self.toggle_liveview)
        print("liveviewSignal connected to toggle liveview - line 351")
        
    def setup_gui(self):
        
        # GUI layout
        
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        
        # parameters widget
        
        self.paramWidget = QGroupBox('XYZ-Tracking parameter')   
        self.paramWidget.setFixedHeight(350)
        self.paramWidget.setFixedWidth(270)
        
        #################################
        # stats widget
        
        self.statWidget = QGroupBox('Live statistics')   
        self.statWidget.setFixedHeight(300)
        # self.statWidget.setFixedWidth(240)
        self.statWidget.setFixedWidth(350)

        self.xstd_label = QtGui.QLabel('X std (nm)')
        self.ystd_label = QtGui.QLabel('Y std (nm)')
        self.zstd_label = QtGui.QLabel('Z std (nm)')
        
        self.xstd_value = QtGui.QLabel('0')
        self.ystd_value = QtGui.QLabel('0')
        self.zstd_value = QtGui.QLabel('0')
        #################################
        # image widget layout
        
        imageWidget = pg.GraphicsLayoutWidget()
        imageWidget.setMinimumHeight(350)
        imageWidget.setMinimumWidth(350)
        
        # setup axis, for scaling see get_image()
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
        self.vb.setAspectLocked(True)
        imageWidget.setAspectLocked(True)

        # set up histogram for the liveview image

        self.hist = pg.HistogramLUTItem(image=self.img)
        
       # lut = viewbox_tools.generatePgColormap(cmaps.parula) #chequear que hacen
       # self.hist.gradient.setColorMap(lut) #
       
#        self.hist.vb.setLimits(yMin=800, yMax=3000)

        ## TO DO: fix histogram range

        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)
        
        # xy drift graph (graph without a fixed range)
        
        self.xyzGraph = pg.GraphicsWindow()
    
#        self.xyzGraph.resize(200, 300)
        self.xyzGraph.setAntialiasing(True)
        
        self.xyzGraph.statistics = pg.LabelItem(justify='right')
        self.xyzGraph.addItem(self.xyzGraph.statistics)
        self.xyzGraph.statistics.setText('---')
        
        self.xyzGraph.xPlot = self.xyzGraph.addPlot(row=0, col=0) #no sé si esta linea va con row=0, parece que yo la modifiqué en xyz_tracking_flor y le puse 1
        self.xyzGraph.xPlot.setLabels(bottom=('Time', 's'),
                            left=('X position', 'nm'))   # TO DO: clean-up the x-y mess (they're interchanged)
        self.xyzGraph.xPlot.showGrid(x=True, y=True)
        self.xmeanCurve = self.xyzGraph.xPlot.plot(pen='w', width=40)
        
        
        self.xyzGraph.yPlot = self.xyzGraph.addPlot(row=1, col=0)
        self.xyzGraph.yPlot.setLabels(bottom=('Time', 's'),
                                     left=('Y position', 'nm'))
        self.xyzGraph.yPlot.showGrid(x=True, y=True)
        self.ymeanCurve = self.xyzGraph.yPlot.plot(pen='r', width=40)
        
        #####################añado plots de zCurve y avgIntCurve
        ########################################################
        self.xyzGraph.zPlot = self.xyzGraph.addPlot(row=2, col=0)
        self.xyzGraph.zPlot.setLabels(bottom=('Time', 's'),
                                     left=('Z position', 'nm'))
        self.xyzGraph.zPlot.showGrid(x=True, y=True)
        self.zCurve = self.xyzGraph.zPlot.plot(pen='y')
        
        self.xyzGraph.avgIntPlot = self.xyzGraph.addPlot(row=3, col=0)
        self.xyzGraph.avgIntPlot.setLabels(bottom=('Time', 's'),
                                           left=('Av. intensity', 'Counts'))
        self.xyzGraph.avgIntPlot.showGrid(x=True, y=True)
        self.avgIntCurve = self.xyzGraph.avgIntPlot.plot(pen='g')
        
        # xy drift graph (2D point plot)
        
        self.xyPoint = pg.GraphicsWindow()
        self.xyPoint.resize(400, 400)
        self.xyPoint.setAntialiasing(False)
        # self.xyPoint.setAspectLocked(True)
        
#        self.xyPoint.xyPointPlot = self.xyzGraph.addPlot(col=1)
#        self.xyPoint.xyPointPlot.showGrid(x=True, y=True)
        
        self.xyplotItem = self.xyPoint.addPlot()
        self.xyplotItem.showGrid(x=True, y=True)
        self.xyplotItem.setLabels(bottom=('X position', 'nm'),
                                  left=('Y position', 'nm'))#Cambio ejes FC
        self.xyplotItem.setAspectLocked(True) #Agregué FC
        
        self.xyDataItem = self.xyplotItem.plot([], pen=None, symbolBrush=(255,0,0), 
                                               symbolSize=5, symbolPen=None)
        
        self.xyDataMean = self.xyplotItem.plot([], pen=None, symbolBrush=(117, 184, 200), 
                                               symbolSize=5, symbolPen=None)
        
        self.xyDataEllipse = self.xyplotItem.plot(pen=(117, 184, 200))

        # z drift graph (1D histogram)
        x = np.arange(-30, 30)
        y = np.zeros(len(x))
        
        self.zHist = pg.BarGraphItem(x=x, height=y, width=0.6, brush='#3BC14A')

        self.zWin = self.xyPoint.addPlot()
        self.zWin.addItem(self.zHist)
        
        # LiveView Button

        self.liveviewButton = QtGui.QPushButton('Camera LIVEVIEW')
        self.liveviewButton.setCheckable(True)
        
        # create xy ROI button
    
        self.xyROIButton = QtGui.QPushButton('xy ROI')
        self.xyROIButton.setCheckable(True)
        self.xyROIButton.clicked.connect(lambda: self.craete_roi(roi_type='xy'))
        
        # create z ROI button
    
        self.zROIButton = QtGui.QPushButton('z ROI')
        self.zROIButton.setCheckable(True)
        self.zROIButton.clicked.connect(lambda: self.craete_roi(roi_type='z'))
        
        # select xy ROI
        
        self.selectxyROIbutton = QtGui.QPushButton('Select xy ROI')
        self.selectxyROIbutton.clicked.connect(lambda: self.emit_roi_info(roi_type='xy'))
        if DEBUG1:
            print("Conexiòn de xy con emit_roi_info existosa")
        
        # select z ROI
        
        self.selectzROIbutton = QtGui.QPushButton('Select z ROI')
        self.selectzROIbutton.clicked.connect(lambda: self.emit_roi_info(roi_type='z'))
        if DEBUG1:
            print("Conexiòn de z con emit_roi_info existosa")
        
        # delete ROI button
        
        self.delete_roiButton = QtGui.QPushButton('delete ROIs')
        self.delete_roiButton.clicked.connect(self.delete_roi)
        
        ##########Aquí se debería agregar delete_roi_zButton
        #self.delete_roi_zButton = QtGui.QPushButton('delete ROI z')
        #self.delete_roi_zButton.clicked.connect(self.delete_roi_z)
        
        # position tracking checkbox
        
        self.exportDataButton = QtGui.QPushButton('Export current data')

        # position tracking checkbox
        
        self.trackingBeadsBox = QtGui.QCheckBox('Track xy fiducials')
        self.trackingBeadsBox.stateChanged.connect(self.setup_data_curves) #agrego esta lìnea porque el tracking no funciona
        self.trackingBeadsBox.stateChanged.connect(self.emit_roi_info)
        
        #En xyz_tracking está la función def setup_data_curves en frontend aquí no, está relacionada con piezo? o es necesaria aquí?
        
        # position tracking checkbox
        
        # self.trackZbeamBox = QtGui.QCheckBox('Track z beam')
        # self.trackZbeamBox.stateChanged.connect(self.emit_roi_info)
        
        # turn ON/OFF feedback loop
        
        self.feedbackLoopBox = QtGui.QCheckBox('Feedback loop')

        # save data signal
        
        self.saveDataBox = QtGui.QCheckBox("Save data")
        self.saveDataBox.stateChanged.connect(self.emit_save_data_state)
        
        
        # button to clear the data
        
        self.clearDataButton = QtGui.QPushButton('Clear data')
        
        #shutter button and label
        #puedo usar esta idea para el confocal
        ##################################
        self.shutterLabel = QtGui.QLabel('Shutter open?')
        self.shutterCheckbox = QtGui.QCheckBox('473 nm laser')

        # Button to make custom pattern

        self.xyPatternButton = QtGui.QPushButton('Move') #es start pattern en linea 500 en xyz_tracking
        
        # buttons and param layout
        
        grid.addWidget(self.paramWidget, 0, 1)
        grid.addWidget(imageWidget, 0, 0)
        grid.addWidget(self.statWidget, 0, 2)
        
        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)

        subgrid.addWidget(self.liveviewButton, 0, 0)
        subgrid.addWidget(self.xyROIButton, 1, 0)
        subgrid.addWidget(self.zROIButton, 2, 0)
        subgrid.addWidget(self.selectxyROIbutton, 3, 0)
        subgrid.addWidget(self.selectzROIbutton, 4, 0)
        subgrid.addWidget(self.delete_roiButton, 5, 0)
        #también añadir algo así 
        #subgrid.addWidget(self.delete_roi_zButton, 6, 0)
        subgrid.addWidget(self.exportDataButton, 6, 0)
        subgrid.addWidget(self.clearDataButton, 7, 0)
        subgrid.addWidget(self.xyPatternButton, 8, 0)
        subgrid.addWidget(self.trackingBeadsBox, 1, 1)
        subgrid.addWidget(self.feedbackLoopBox, 2, 1)
        subgrid.addWidget(self.saveDataBox, 3, 1)
        subgrid.addWidget(self.shutterLabel, 9, 0)
        subgrid.addWidget(self.shutterCheckbox, 9, 1)
        
        ####################################################
        #Agrego FC grilla para estadística
        stat_subgrid = QtGui.QGridLayout()
        self.statWidget.setLayout(stat_subgrid)
        
        stat_subgrid.addWidget(self.xstd_label, 0, 0)
        stat_subgrid.addWidget(self.ystd_label, 1, 0)
        stat_subgrid.addWidget(self.zstd_label, 2, 0)
        stat_subgrid.addWidget(self.xstd_value, 0, 1)
        stat_subgrid.addWidget(self.ystd_value, 1, 1)
        stat_subgrid.addWidget(self.zstd_value, 2, 1)
        ########################################################
        
        grid.addWidget(self.xyzGraph, 1, 0)
        grid.addWidget(self.xyPoint, 1, 1, 1, 2) #######agrego 1,2 al final
        
        self.liveviewButton.clicked.connect(lambda: self.toggle_liveview(self.liveviewButton.isChecked()))
        print("liveviewButton connected to toggle liveview - line 431")
        

#aquí debería ir esta función de ser necesario, pruebo descomentando para saber si debe ir o no
    def setup_data_curves(self):
                     
         if self.trackingBeadsBox.isChecked():
             
             if self.xCurve is not None:
         
                 for i in range(len(self.roilist)): # remove previous curves
                 
                     self.xyzGraph.xPlot.removeItem(self.xCurve[i]) 
                     self.xyzGraph.yPlot.removeItem(self.yCurve[i]) 
                 
             self.xCurve = [0] * len(self.roilist)
             
             for i in range(len(self.roilist)):
                 self.xCurve[i] = self.xyzGraph.xPlot.plot(pen='w', alpha=0.3)
                 self.xCurve[i].setAlpha(0.3, auto=False)
                 
             self.yCurve = [0] * len(self.roilist)
             
             for i in range(len(self.roilist)):
                 self.yCurve[i] = self.xyzGraph.yPlot.plot(pen='r', alpha=0.3)
                 self.yCurve[i].setAlpha(0.3, auto=False) 
                     
         else:
             
             pass

        
    def closeEvent(self, *args, **kwargs):
        
        print('close in frontend')
        
        self.closeSignal.emit()
        super().closeEvent(*args, **kwargs)
        app.quit()
        
class Backend(QtCore.QObject):
    
    changedImage = pyqtSignal(np.ndarray)
    changedData = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    updateGUIcheckboxSignal = pyqtSignal(bool, bool, bool) #no se usa en xyz_tracking
    #changedSetPoint = pyqtSignal(float) #Debería añadir esta señal??? de focus.py
    xyIsDone = pyqtSignal(bool, float, float)  # signal to emit new piezo position after drift correction
    shuttermodeSignal = pyqtSignal(int, bool)
    liveviewSignal = pyqtSignal(bool)
    zIsDone = pyqtSignal(bool, float) #se emite para psf.py script
    focuslockpositionSignal = pyqtSignal(float) #se emite para scan.py
    
    """
    Signals
    
    - changedImage:
        To: [frontend] get_image
             
    - changedData:
        To: [frontend] get_data
        
    - updateGUIcheckboxSignal:
        To: [frontend] get_backend_states
        
    - xyIsDone:
        To: [psf] get_xy_is_done
        
    - zIsDone:
        To: [psf] get_z_is_done
        
    - shuttermodeSignal:
        To: [frontend] update_shutter
        
    - focuslockpositionSignal:
        To: [scan] get current focus lock position

    """

    def __init__(self, camera, adw, *args, **kwargs): #cambié andor por thorcam
        super().__init__(*args, **kwargs)
              
        self.camera = camera # no need to setup or initialize camera
        
        if VIDEO:
            self.video = []
        
        self.adw = adw
        
        # folder
        
        today = str(date.today()).replace('-', '')  # TO DO: change to get folder from microscope
        root = r'C:\\Data\\'
        folder = root + today
        print("Name of folder: ",folder)
        
        filename = r'\xydata'
        self.filename = folder + filename
        
        self.viewtimer = QtCore.QTimer() #Ojo: aquí coloqué viewtimer porque es el que se usa a lo largo del código, pero en xyz_tracking se usa view_timer
        #self.viewtimer.timeout.connect(self.update) #Línea original
        self.xyz_time = 200 # 200 ms per acquisition + fit + correction
        
        self.tracking_value = False
        self.save_data_state = False
        self.feedback_active = False
        self.camON = False
        self.roi_area = np.zeros(4) #Esta línea para qué?

        self.npoints = 1200
        self.buffersize = 30000
        
        self.currentx = 0 #Chequear estos dos atributos
        self.currenty = 0
        
        #self.reset() #comwnté FC
       # self.reset_data_arrays() #comenté FC
        
        self.counter = 0
        
        # saves displacement when offsetting setpoint for feedbackloop
        
        self.displacement = np.array([0.0, 0.0])
        self.pattern = False
        
        self.previous_image = None
        
        self.focusSignal = 0
        
        if self.camera.open_device():
            self.camera.set_roi(16, 16, 1920, 1200)
            try:
                self.camera.alloc_and_announce_buffers()
                self.camera.start_acquisition()
            except Exception as e:
                print("Exception", str(e))
        else:
            self.camera.destroy_all()
       
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
        if DEBUG:
            print("Inside Liveview-start")
        if self.camON:
            self.viewtimer.stop()
            self.camON = False
        self.camON = True
        self.viewtimer.start(self.xyz_time)
        
    def liveview_stop(self):
        if DEBUG:
            print("Inside Liveview-stop")
        self.viewtimer.stop()
        print("cameraTimer: stopped")
        self.camON = False  
        x0 = 0
        y0 = 0
        x1 = 1200 
        y1 = 1920
            
        val = np.array([x0, y0, x1, y1])
        #self.get_new_roi(val) #¿debo comentar esta línea como se hizo en xyz_tracking.py?
        #SI, Aquì no existe la función get_new_roi como en focus.py
                    
    def update(self):
        """ General update method """
        
        self.update_view()

        if self.tracking_value:
            
            t0 = time.time()
            self.track('xy')
            t1 = time.time()
        
            print('track xy took', (t1-t0)*1000, 'ms')
            
            t0 = time.time()
            self.track('z')
            t1 = time.time()
            
            print('track z took', (t1-t0)*1000, 'ms')
            
            t0 = time.time()
            self.update_graph_data()
            t1 = time.time()
            
            print('update graph data took', (t1-t0)*1000, 'ms')
            
            if self.feedback_active:
                    
                t0 = time.time()    
                self.correct()
                t1 = time.time()
                
                print('correct took', (t1-t0)*1000, 'ms')
                
        if self.pattern:
            val = (self.counter - self.initcounter)
            reprate = 50 #Antes era 10 para andor
            if (val % reprate == 0):
                self.make_tracking_pattern(val//reprate)
            
        self.counter += 1  # counter to check how many times this function is executed



    def update_view(self):
        """ Image update while in Liveview mode """
        
        # acquire image

        self.image = self.camera.on_acquisition_timer() #This is a 2D array, (only R channel)

        # WARNING: fix to match camera orientation with piezo orientation
        self.image = np.rot90(self.image, k=3) #Comment by FC
        
        if np.all(self.previous_image == self.image):
            
            print('WARNING: latest_frame equal to previous frame')
    
        self.previous_image = self.image
        
        if (VIDEO and self.save_data_state):
            self.video.append(self.image)

        # send image to gui
        self.changedImage.emit(self.image)
        
        
    def update_graph_data(self): #Incorporo cambios con vistas a añadir data actualizada de z
        """ Update the data displayed in the graphs """
        
        if self.ptr < self.npoints:
            self.xData[self.ptr, :] = self.x + self.displacement[0]
            self.yData[self.ptr, :] = self.y + self.displacement[1]
            self.zData[self.ptr] = self.z #Aparentemente es focusSignal
            self.avgIntData[self.ptr] = self.avgInt
            self.time[self.ptr] = self.currentTime
            
            self.changedData.emit(self.time[0:self.ptr + 1],
                                  self.xData[0:self.ptr + 1],
                                  self.yData[0:self.ptr + 1],
                                  self.zData[0:self.ptr + 1],
                                  self.avgIntData[0:self.ptr + 1])
            
        else:
            self.xData[:-1] = self.xData[1:]
            self.xData[-1, :] = self.x + self.displacement[0]
            self.yData[:-1] = self.yData[1:]
            self.yData[-1, :] = self.y + self.displacement[1]
            self.zData[:-1] = self.zData[1:]
            self.zData[-1] = self.z
            self.avgIntData[:-1] = self.avgIntData[1:]
            self.avgIntData[-1] = self.avgInt
            self.time[:-1] = self.time[1:]
            self.time[-1] = self.currentTime
            
            self.changedData.emit(self.time, self.xData, self.yData, 
                                  self.zData, self.avgIntData)

        self.ptr += 1
    
    @pyqtSlot(bool)
    def toggle_tracking(self, val): #esta función es igual a la de xyz_tracking porque es para xy únicamente
        
        '''
        Connection: [frontend] trackingBeadsBox.stateChanged
        Description: toggles ON/OFF tracking of fiducial fluorescent beads. 
        Drift correction feedback loop is not automatically started.
        
        '''

        
        self.startTime = time.time()
        
        if val is True:
            
            self.reset()
            self.reset_data_arrays()
            
            self.tracking_value = True
            self.counter = 0
            
            # initialize relevant xy-tracking arrays 
        
            size = len(self.roi_coordinates_list)
            
            self.currentx = np.zeros(size)
            self.currenty = np.zeros(size)
            self.x = np.zeros(size)
            self.y = np.zeros(size)
            
            if self.initial is True:
                
                self.initialx = np.zeros(size)
                self.initialy = np.zeros(size)
                    
        if val is False:
        
            self.tracking_value = False
            
    @pyqtSlot(bool)
    def toggle_feedback(self, val, mode='continous'): #Esta función es adecuada porque tiene en cuenta los procesos de de ADwin para drift xy 

        ''' 
        Connection: [frontend] feedbackLoopBox.stateChanged
        Description: toggles ON/OFF feedback for either continous (TCSPC) 
        or discrete (scan imaging) correction
        '''
        
        if val is True:
            #self.reset() #Qué efecto tendría colocar esta función aquí? Esto es según focus.py
            self.setup_feedback() #Añado esto por focus
            #self.update() ##Qué efecto tendría colocar esta función aquí? Esto es según focus.py
            self.feedback_active = True

            # set up and start actuator process
            
            if mode == 'continous':
            
                self.set_actuator_param()
                self.adw.Start_Process(4) #proceso para xy
                self.adw.Start_Process(3) #proceso para z
                print('process 4 status', self.adw.Process_Status(4))
                print(datetime.now(), '[focus] Process 4 started')
                print('process 3 status', self.adw.Process_Status(3))
                print(datetime.now(), '[focus] Process 3 started')
            
            if DEBUG:
                print(datetime.now(), '[xy_tracking] Feedback loop ON')
                print(datetime.now(), ' [focus] Feedback loop ON')
            
        if val is False:
            
            self.feedback_active = False
            
            if mode == 'continous':

                self.adw.Stop_Process(4)
                self.adw.Stop_Process(3)
                print(datetime.now(), '[xy_tracking] Process 4 stopped')
                print(datetime.now(), '[focus] Process 3 stopped')
                self.displacement = np.array([0.0, 0.0])

            
            if DEBUG:
                print(datetime.now(), '[xy_tracking] Feedback loop OFF')
                print(datetime.now(), ' [focus] Feedback loop OFF')
#            
#        self.updateGUIcheckboxSignal.emit(self.tracking_value, 
#                                          self.feedback_active, 
#                                          self.save_data_state)
    

    def center_of_mass(self):
        
        # set main reference frame
        
        xmin, xmax, ymin, ymax = self.zROIcoordinates
        
        # select the data of the image corresponding to the ROI

        zimage = self.image[xmin:xmax, ymin:ymax]
        
        # WARNING: extra rotation added to match the sensitive direction (hardware)
        
        zimage = np.rot90(zimage, k=3)
        
        # calculate center of mass
        
        self.m_center = np.array(ndi.measurements.center_of_mass(zimage))
        
        # calculate z estimator
        
        self.currentz = np.sqrt(self.m_center[0]**2 + self.m_center[1]**2) #Chequear si aquí conviene poner signo menos
        #Nota: self.currentz es self.focusSignal
        
    def gaussian_fit(self,roi_coordinates): #Le estoy agregando un parámetro (roi_coordinates) para que sea como en xyz_tracking
        
        # set main reference frame
        
        roi_coordinates = np.array(roi_coordinates, dtype=np.int)
        
        xmin, xmax, ymin, ymax = roi_coordinates
        xmin_nm, xmax_nm, ymin_nm, ymax_nm = roi_coordinates * PX_SIZE
        
        # select the data of the image corresponding to the ROI

        array = self.image[xmin:xmax, ymin:ymax]
        
        if np.size(array) == 0:
            
            print('WARNING: array is []')
        
        # set new reference frame
        
        xrange_nm = xmax_nm - xmin_nm
        yrange_nm = ymax_nm - ymin_nm
             
        x_nm = np.arange(0, xrange_nm, PX_SIZE)
        y_nm = np.arange(0, yrange_nm, PX_SIZE)
        
        (Mx_nm, My_nm) = np.meshgrid(x_nm, y_nm)
        
        # find max 
        
        argmax = np.unravel_index(np.argmax(array, axis=None), array.shape)
        
        x_center_id = argmax[0]
        y_center_id = argmax[1]
        
        # define area around maximum
    
        xrange = 15 # in px #en el código original era 10, pero lo cambié porque así está en xyz_tracking
        yrange = 15 # in px
        
        xmin_id = int(x_center_id-xrange)
        xmax_id = int(x_center_id+xrange)
        
        ymin_id = int(y_center_id-yrange)
        ymax_id = int(y_center_id+yrange)
        
        array_sub = array[xmin_id:xmax_id, ymin_id:ymax_id]
                
        xsubsize = 2 * xrange
        ysubsize = 2 * yrange
        
#        plt.imshow(array_sub, cmap=cmaps.parula, interpolation='None')
        
        x_sub_nm = np.arange(0, xsubsize) * PX_SIZE
        y_sub_nm = np.arange(0, ysubsize) * PX_SIZE

        [Mx_sub, My_sub] = np.meshgrid(x_sub_nm, y_sub_nm)
        
        # make initial guess for parameters
        
        bkg = np.min(array)
        A = np.max(array) - bkg
        σ = 200 # nm #antes era 130nm
        x0 = x_sub_nm[int(xsubsize/2)]
        y0 = y_sub_nm[int(ysubsize/2)]
        
        initial_guess_G = [A, x0, y0, σ, σ, bkg]
        
        if np.size(array_sub) == 0:
            
            print('WARNING: array_sub is []')
         
        poptG, pcovG = opt.curve_fit(PSF.gaussian2D, (Mx_sub, My_sub), 
                                     array_sub.ravel(), p0=initial_guess_G)
        
        perr = np.sqrt(np.diag(pcovG))
        
        print('perr', perr)
        
        # retrieve results

        poptG = np.around(poptG, 2)
    
        A, x0, y0, σ_x, σ_y, bkg = poptG
        
        x = x0 + Mx_nm[xmin_id, ymin_id]
        y = y0 + My_nm[xmin_id, ymin_id]
        
        currentx = x
        currenty = y
#ALERTA comento esta parte porque creo que hay incompatibilidad entre los nombres self.currentx y currentx y lo mismo para y
#Ver si en un futuro puedo implementar esto para que tenga en cuenta este caso
#        # if to avoid (probably) false localizations #notar que esta parte no se usa en xyz_tracking
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
#            
##            print(datetime.now(), '[xy_tracking] else')
        
        return currentx, currenty
        
        print('currentx: ', currentx, ' currenty: ',currenty)
            
    def track(self, track_type): #Añado parámetro para trabajar en xy y z
        
        """ 
        Function to track fiducial markers (Au NPs) from the selected ROI.
        The position of the NPs is calculated through an xy gaussian fit 
        If feedback_active = True it also corrects for drifts in xy
        If save_data_state = True it saves the xy data
        
        """
        
        # Calculate average intensity in the image to check laser fluctuations
        
        self.avgInt = np.mean(self.image)
        
        print('Average intensity', self.avgInt)
        
        # xy track routine of N=size fiducial AuNP

        if track_type == 'xy':
            
            for i, roi in enumerate(self.roi_coordinates_list):
                
                # try:
                #     roi = self.roi_coordinates_list[i]
                #     self.currentx[i], self.currenty[i] = self.gaussian_fit(roi)
                    
                # except(RuntimeError, ValueError):
                    
                #     print(datetime.now(), '[xy_tracking] Gaussian fit did not work')
                #     self.toggle_feedback(False)
                
                roi = self.roi_coordinates_list[i]
                print("roi:",roi, "type roi", type(roi))
                self.currentx[i], self.currenty[i] = self.gaussian_fit(roi)
                print("ejecutado con exito")
           
            if self.initial is True:
                
                for i, roi in enumerate(self.roi_coordinates_list):
                       
                    self.initialx[i] = self.currentx[i]
                    self.initialy[i] = self.currenty[i]
                    
                self.initial = False
            
            for i, roi in enumerate(self.roi_coordinates_list):
                    
                self.x[i] = self.currentx[i] - self.initialx[i]  # self.x is relative to initial pos
                self.y[i] = self.currenty[i] - self.initialy[i]
                
                self.currentTime = time.time() - self.startTime
                
            # print('x, y', self.x, self.y)
            # print('currentx, currenty', self.currentx, self.currenty)
                
            if self.save_data_state:
                
                self.time_array[self.j] = self.currentTime
                self.x_array[self.j, :] = self.x + self.displacement[0]
                self.y_array[self.j, :] = self.y + self.displacement[1]
                
                self.j += 1
                            
                if self.j >= (self.buffersize - 5):    # TODO: -5, arbitrary bad fix
                    
                    self.export_data()
                    self.reset_data_arrays()
                    
            #         print(datetime.now(), '[xy_tracking] Data array, longer than buffer size, data_array reset')
            
        # z track of the reflected IR beam   
        ####################### Revisar esto del trackeo en z, no puedo correlacionar con focus.py
            
        if track_type == 'z':
            
            self.center_of_mass()
            
            if self.initial_focus is True:
                
                self.initialz = self.currentz
                
                self.initial_focus = False
            
            self.z = (self.currentz - self.initialz) * PX_Z #self.z in nm
            print("self.z in track: ", self.z, " nm")
                
    def correct(self, mode='continous'):

        xmean = np.mean(self.x)
        ymean = np.mean(self.y)        

        dx = 0
        dy = 0
        dz = 0 #comparar con update_feedback
        
        threshold = 3 #antes era 5 con Andor
        z_threshold = 3
        far_threshold = 12 #ojo con estos parametros chequear focus
        correct_factor = 0.6
        
        security_thr = 0.35 # in µm
        
        if np.abs(xmean) > threshold:
            
            if dx < far_threshold: #TODO: double check this conditions (do they work?)
                
                dx = correct_factor * dx #TODO: double check this conditions (do they work?)
            
            dx = - (xmean)/1000 # conversion to µm
              # print('TEST','dx', dx)
            
        if np.abs(ymean) > threshold:
            
            if dy < far_threshold:
                
                dy = correct_factor * dy
            
            dy = - (ymean)/1000 # conversion to µm
            # print('TEST','dy', dy)
    
        if np.abs(self.z) > z_threshold:
                            
            dz = - (self.z)/1000 # conversion to µm
                
            if dz < far_threshold:
                    
                dz = correct_factor * dz
                
#                print('dz', dz)
        else:
            dz = - (self.z)/1000

        if dx > security_thr or dy > security_thr or dz > 2 * security_thr:
            
            print(datetime.now(), '[xyz_tracking] Correction movement larger than 200 nm, active correction turned OFF')
            self.toggle_feedback(False)
            
        else:
            
            # compensate for the mismatch between camera/piezo system of reference
            
            theta = np.radians(-3.7)   # 86.3 (or 3.7) is the angle between camera and piezo (measured)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c,-s), (s, c)))
            
            dy, dx = np.dot(R, np.asarray([dx, dy])) #ver si se puede arreglar esto añadiendo dz
            
            # add correction to piezo position
            
            currentXposition = tools.convert(self.adw.Get_FPar(70), 'UtoX') #Get_FPar(self, Index): Retorna el valor de una variable global de tipo float.
            currentYposition = tools.convert(self.adw.Get_FPar(71), 'UtoX')
            currentZposition = tools.convert(self.adw.Get_FPar(72), 'UtoX') #¿Está bien que sea key='UtoX'? FPar keeps track of z position of the piezo
            
            #print("self.z: ",self.z, " nm.")
            #print("dz: ",dz, " µm.")
            targetXposition = currentXposition + dx  
            targetYposition = currentYposition + dy
            targetZposition = currentZposition + dz  # in µm
            
            if mode == 'continous':
                #Le mando al actuador las posiciones x,y,z
                self.actuator_xyz(targetXposition, targetYposition, targetZposition) #aquí debería agregar targetZposition
                
            if mode == 'discrete':
                
#                self.moveTo(targetXposition, targetYposition, 
#                            currentZposition, pixeltime=10)
                
                self.target_x = targetXposition
                self.target_y = targetYposition
                self.target_z = targetZposition
            
    @pyqtSlot(bool, bool)
    def single_xy_correction(self, feedback_val, initial): 
        
        """
        From: [psf] xySignal
        Description: Starts acquisition of the camera and makes one single xy
        track and, if feedback_val is True, corrects for the drift
        """
        if DEBUG:
            print(datetime.now(), '[xy_tracking] Feedback {}'.format(feedback_val))
        
        if initial:
            self.toggle_feedback(True, mode='discrete')
            self.initial = initial
            print(datetime.now(), '[xy_tracking] initial', initial)
        
        if not self.camON:
            print(datetime.now(), 'liveview started')
            self.camON = True
            #self.camera.start_live_video(framerate='20 Hz') # Check how to do this part with new camera
            #CReo que aqui deberia ir algo así:
        # if self.camera.open_device():
        #     self.camera.set_roi(16, 16, 1920, 1200)
        #     try:
        #         self.camera.alloc_and_announce_buffers()
        #         self.camera.start_acquisition()
        #     except Exception as e:
        #         print("Exception", str(e))
        # else:
        #     self.camera.destroy_all()
        #La idea creo que es iniciar la adquisicion
            time.sleep(0.200)
            
        time.sleep(0.200)
        print("Estoy dentro de single_xy_correction")
        self.image = self.camera.on_acquisition_timer()
        self.changedImage.emit(self.image)
            
        self.camera.stop_acquisition() #???? Chequear esto lo estoy poniendo rapido como bosquejo
        self.camON = False
        
        self.track('xy')
        self.update_graph_data()
        self.correct(mode='discrete')
                
        target_x = np.round(self.target_x, 3)
        target_y = np.round(self.target_y, 3)
        
        print(datetime.now(), '[xy_tracking] discrete correction to', 
              target_x, target_y)
    
        self.xyIsDone.emit(True, target_x, target_y)
        
        if DEBUG:
            print(datetime.now(), '[xy_tracking] single xy correction ended') 
     
    @pyqtSlot(bool, bool)
    def single_z_correction(self, feedback_val, initial): #revisar los parámetros de esta funcion, la saqué de focus.py para que emita señal a psf
        
        if initial:
                    
            if not self.camON:
                self.camON = True
        #         self.camera.start_live_video(framerate='20 Hz')
        #Pensar esta parte, cómo sería para que funcionen de forma independiente
        # if self.camera.open_device():
        #     self.camera.set_roi(16, 16, 1920, 1200)
        #     try:
        #         self.camera.alloc_and_announce_buffers()
        #         self.camera.start_acquisition()
        #     except Exception as e:
        #         print("Exception", str(e))
        # else:
        #     self.camera.destroy_all()
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
            
        
        self.acquire_data() #No está definida aquí
        self.update_graph_data()
                
        if initial:
            
            self.setup_feedback()
            
        else:
        
            self.update_feedback(mode='discrete')
        
        if self.save_data_state:
            
            self.time_array.append(self.currentTime)
            self.z_array.append(self.focusSignal) #z_rrray lo agregué a reset_data_arrays. Aquí falta ver quien es focus signal en este caso, ver focus.py
                    
        self.zIsDone.emit(True, self.target_z)  
        
    def calibrate(self):
        
        # TO DO: fix calibration function
        
        self.viewtimer.stop()
        time.sleep(0.100)
        
        nsteps = 40
        xmin = 9.5  # in µm
        xmax = 10.5   # in µm
        xrange = xmax - xmin  
        
        calibData = np.zeros(40)
        xData = np.arange(xmin, xmax, xrange/nsteps) #aquí xData es una variable muda, se usa solo dos veces
        
        zMoveTo(self.actuator, xmin) #no entiendo de dónde sale este actuator
        
        time.sleep(0.100)
        
        for i in range(nsteps):
            
            zMoveTo(self.actuator, xmin + (i * 1/nsteps) * xrange)
            self.update()
            calibData[i] = self.focusSignal #quien es focus signal?
            
        plt.plot(xData, calibData, 'o')
            
        time.sleep(0.200)
        
        self.viewtimer.start(self.xyz_time)
    
    def set_actuator_param(self, pixeltime=1000): #configura los parámetros del actuador

        self.adw.Set_FPar(46, tools.timeToADwin(pixeltime)) 
        self.adw.Set_FPar(36, tools.timeToADwin(pixeltime)) #Añado para z, focus.py
        # set-up actuator initial param
        
        currentXposition = tools.convert(self.adw.Get_FPar(70), 'UtoX')
        currentYposition = tools.convert(self.adw.Get_FPar(71), 'UtoX')
        #por qué no debo colocar una linea similar para current z_position
    
        x_f = tools.convert(currentXposition, 'XtoU')
        y_f = tools.convert(currentYposition, 'XtoU')
        
        # set-up actuator initial param
    
        z_f = tools.convert(10, 'XtoU') #no estoy segura de esta linea #Añado para z, focus.py
        
        self.adw.Set_FPar(40, x_f)
        self.adw.Set_FPar(41, y_f)
        self.adw.Set_FPar(32, z_f) #Añado para z, focus.py
            
        self.adw.Set_Par(40, 1) #Set_Par(self, Index, Value): Establece una variable global de tipo long con el valor especificado.
        self.adw.Set_Par(30, 1) #Añado para z, focus.py
        
    def actuator_xyz(self, x_f, y_f, z_f):
        
#        print(datetime.now(), '[xy_tracking] actuator x, y =', x_f, y_f)
        
        x_f = tools.convert(x_f, 'XtoU')
        y_f = tools.convert(y_f, 'XtoU')
        z_f = tools.convert(z_f, 'XtoU') #Añado para z, focus.py
        
        self.adw.Set_FPar(40, x_f)
        self.adw.Set_FPar(41, y_f)
        self.adw.Set_FPar(32, z_f) #Añado para z, focus.py
        
        self.adw.Set_Par(40, 1)   
        self.adw.Set_Par(30, 1) #Añado para z, focus.py
            
    def set_moveTo_param(self, x_f, y_f, z_f, n_pixels_x=128, n_pixels_y=128,
                         n_pixels_z=128, pixeltime=2000):
        #Esta funcion se repite en xy_tracking y focus

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
        #Esta funcion se repite en xy_tracking y focus

        self.set_moveTo_param(x_f, y_f, z_f)
        self.adw.Start_Process(2)
            
    def reset(self):
        
        self.initial = True
        self.initial_focus = True
        
        try:
            self.xData = np.zeros((self.npoints, len(self.roi_coordinates_list)))
            self.yData = np.zeros((self.npoints, len(self.roi_coordinates_list)))
            
        except:
            
            self.xData = np.zeros(self.npoints)
            self.yData = np.zeros(self.npoints)
        
        self.zData = np.zeros(self.npoints)
        self.avgIntData = np.zeros(self.npoints)
        self.time = np.zeros(self.npoints)
        self.ptr = 0
        self.startTime = time.time()
        #self.j = 0  # iterator on the data array
        #-----------
        self.max_dev = 0 # Sale de focus.py se usa en update_stats
        self.std = 0
        self.n = 1
        #-----------
        self.changedData.emit(self.time, self.xData, self.yData, self.zData, 
                              self.avgIntData)
        
    def reset_data_arrays(self):
        
        self.time_array = np.zeros(self.buffersize, dtype=np.float16)
        
        self.x_array = np.zeros((self.buffersize, 
                                 len(self.roi_coordinates_list)), 
                                 dtype=np.float16)
        
        self.y_array = np.zeros((self.buffersize, 
                                 len(self.roi_coordinates_list)), 
                                 dtype=np.float16)
        self.z_array = []
        
        self.j = 0  # iterator on the data array
        
        
    def export_data(self): #Todavía tengo que modificar esta función para guardar data, la dejo tal cual esta por ahora
        
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
        
        print(datetime.now(), '[xy_tracking] xy data exported to', filename)

    @pyqtSlot(bool)    
    def get_stop_signal(self, stoplive): #Está en xy y en focus. Creo que no está conectado a nada
        
        """
        Connection: [psf] xyStopSignal
        Description: stops liveview, tracking, feedback if they where running to
        start the psf measurement with discrete xy - z corrections
        """
                
        self.toggle_feedback(False)
        self.toggle_tracking(False)
        
        self.reset()
        self.reset_data_arrays()
        
        self.save_data_state = True  # TO DO: sync this with GUI checkboxes (Lantz typedfeat?)
            
        if not stoplive:
            self.liveviewSignal.emit(False)
            
    def export_data(self):
        
        """
        Exports the x, y and t data into a .txt file
        """
        
        fname = self.folder + '/xy_data'
        
        #case distinction to prevent wrong filenaming when starting minflux or psf measurement
        if fname[0] == '!':
            filename = fname[1:]
        else:
            filename = tools.getUniqueName(fname)
        filename = filename + '.txt'
        
        size = self.j
        N_NP = len(self.roi_coordinates_list)
        
        savedData = np.zeros((size, 2*N_NP+1))

        savedData[:, 0] = self.time_array[0:self.j]
        savedData[:, 1:N_NP+1] = self.x_array[0:self.j, :]
        savedData[:, N_NP+1:2*N_NP+1] = self.y_array[0:self.j, :]
        
        np.savetxt(filename, savedData,  header='t (s), x (nm), y(nm)') # transpose for easier loading
        
        print(datetime.now(), '[xy_tracking] xy data exported to', filename)
        print('Exported data shape', np.shape(savedData))
        
        self.export_image()
        
        if VIDEO:
            
            tifffile.imwrite(fname + 'video' + '.tif', np.array(self.video))

    @pyqtSlot(bool)
    def get_save_data_state(self, val): #Dejo esta funcion como está, se repite en xy y en focus.py
        
        '''
        Connection: [frontend] saveDataSignal
        Description: gets value of the save_data_state variable, True -> save,
        False -> don't save
        
        '''
        
        self.save_data_state = val
        
        if DEBUG:
            print(datetime.now(), '[xy_tracking] save_data_state = {}'.format(val))
    
    @pyqtSlot(str, int, list) #antes se usaba roi_coordinates_array que se convertía a int en ROIcoordinates
    def get_roi_info(self, roi_type, N, coordinates_list): #Toma la informacion del ROI que viene de emit_roi_info en el frontend
        
        '''
        Connection: [frontend] roiInfoSignal , z_roiInfoSignal
        Description: gets coordinates of the ROI in the GUI
        
        '''
        if DEBUG1:
                print(datetime.now(), 'Estoy en get_roi_info')
                
        if roi_type == 'xy':
                            
            self.roi_coordinates_list = coordinates_list #LISTA
            if DEBUG1:
                print(datetime.now(), 'roi_coordinates_list: ',self.roi_coordinates_list)
        
            if DEBUG:
                print(datetime.now(), '[xy_tracking] got ROI coordinates list')
                
        if roi_type == 'z':
            
            self.zROIcoordinates = coordinates_list[0].astype(int) #ENTERO. coordinates_list en el caso de z se convierte a int poeque sólo tiene un valor
            if DEBUG1:
                print(datetime.now(), 'zROIcoordinates: ',self.zROIcoordinates)
     
    @pyqtSlot()    
    def get_lock_signal(self): #Dejo esta funcion como está
        
        '''
        Connection: [minflux] xyzStartSignal
        Description: activates tracking and feedback
        
        '''
        
        if not self.camON:
            self.liveviewSignal.emit(True)
        
        self.toggle_tracking(True)
        self.toggle_feedback(True)
        self.save_data_state = True
        #Esto está comentado en focus pero no en xy
        self.updateGUIcheckboxSignal.emit(self.tracking_value, 
                                          self.feedback_active, 
                                          self.save_data_state)
        
        if DEBUG:
            print(datetime.now(), '[xy_tracking] System xy locked')

    @pyqtSlot(np.ndarray, np.ndarray) 
    def get_move_signal(self, r, r_rel):            
        
        self.toggle_feedback(False)
#        self.toggle_tracking(True)
        
        self.updateGUIcheckboxSignal.emit(self.tracking_value, 
                                          self.feedback_active, 
                                          self.save_data_state)
        
        x_f, y_f, z_f = r

        self.actuator_xyz(x_f, y_f, z_f)
         
        if DEBUG:
            print(datetime.now(), '[xy_tracking] Moved to', r)
        
#        # Lock again
        
#        print(datetime.now(), '[xy_tracking] initial x and y', self.initialx, self.initialy)
#        print(datetime.now(), '[xy_tracking] dx, dy', r_rel)
##        self.initial = True # to lock at a new position, TO DO: fix relative position tracking
#        self.initialx = self.currentx - r_rel[0] * 1000 # r_rel to nm
#        self.initialy = self.currenty - r_rel[1] * 1000 # r_rel to nm
#        print(datetime.now(), '[xy_tracking] initial x and y', self.initialx, self.initialy)
        
#        self.toggle_feedback(True) # TO DO: fix each position lock
        

    def start_tracking_pattern(self):
        
        self.pattern = True
        self.initcounter = self.counter

    def make_tracking_pattern(self, step):
                
        if (step < 2) or (step > 5):
            return
        elif step == 2:
            dist = np.array([0.0, 20.0])
        elif step == 3:
            dist = np.array([20.0, 0.0])
        elif step == 4:
            dist = np.array([0.0, -20.0])
        elif step == 5:
            dist = np.array([-20.0, 0.0])
        
        
        self.initialx = self.initialx + dist[0]
        self.initialy = self.initialy + dist[1]
        self.displacement = self.displacement + dist
        
        print(datetime.now(), '[xy_tracking] Moved setpoint by', dist)
      
    @pyqtSlot(str)    
    def get_end_measurement_signal(self, fname):
        
        '''
        From: [minflux] xyzEndSignal or [psf] endSignal
        Description: at the end of the measurement exports the xy data

        '''
        
        self.filename = fname
        self.export_data()
        
        
        self.toggle_feedback(False) # TO DO: decide whether I want feedback ON/OFF at the end of measurement
        #check
        self.toggle_tracking(False)
        self.pattern = False
        
        self.reset()
        self.reset_data_arrays()
        #comparar con la funcion de focus: algo que ver con focusTimer
            
    @pyqtSlot(float)
    def get_focuslockposition(self, position):
        
        if position == -9999:
            position = self.setPoint
        else:
            position = self.focusSignal #Quien es focus signal
            
        self.focuslockpositionSignal.emit(position)
        
    def make_connection(self, frontend):
        if DEBUG:
            print("Connecting backend to frontend")
            
        frontend.roiInfoSignal.connect(self.get_roi_info)
        frontend.z_roiInfoSignal.connect(self.get_roi_info)
        frontend.closeSignal.connect(self.stop)
        frontend.saveDataSignal.connect(self.get_save_data_state)
        frontend.exportDataButton.clicked.connect(self.export_data)
        frontend.clearDataButton.clicked.connect(self.reset)
        frontend.clearDataButton.clicked.connect(self.reset_data_arrays)
        frontend.trackingBeadsBox.stateChanged.connect(lambda: self.toggle_tracking(frontend.trackingBeadsBox.isChecked()))
        frontend.shutterCheckbox.stateChanged.connect(lambda: self.toggle_tracking_shutter(8, frontend.shutterCheckbox.isChecked()))
        frontend.liveviewButton.clicked.connect(self.liveview)
        frontend.feedbackLoopBox.stateChanged.connect(lambda: self.toggle_feedback(frontend.feedbackLoopBox.isChecked()))
        frontend.xyPatternButton.clicked.connect(lambda: self.make_tracking_pattern(1)) #duda con esto, comparar con línea análoga en xyz_tracking
        
        #La función toggle_feedback se utiliza como un slot de PyQt y se conecta al evento stateChanged de un cuadro de verificación llamado feedbackLoopBox. Su propósito es activar o desactivar el feedback (retroalimentación) para la corrección continua en el modo especificado.
        
        # TO DO: clean-up checkbox create continous and discrete feedback loop
        
        # lambda function and gui_###_state are used to toggle both backend
        # states and checkbox status so that they always correspond 
        # (checked <-> active, not checked <-> inactive)
        
    @pyqtSlot()    
    def stop(self): #se repite en xy y focus
        self.toggle_tracking_shutter(8, False)
        time.sleep(1)
        
        self.viewtimer.stop()
        
        #prevent system to throw weird errors when not being able to close the camera, see uc480.py --> close()
        self.reset()
        
        # Go back to 0 position

        x_0 = 0
        y_0 = 0
        z_0 = 0

        self.moveTo(x_0, y_0, z_0)
        
        self.camera.destroy_all()
        print(datetime.now(), '[xyz_tracking] IDS camera shut down')
        

if __name__ == '__main__':

    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()
        
    #app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    # initialize devices
    
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)
    
    #if camera wasnt closed properly just keep using it without opening new one
    try:
        camera = ids_cam.IDS_U3()
    except:
        pass
    
    gui = Frontend()
    
    if DEBUG:
        print("gui class Frontend instanced, FC")
        
    worker = Backend(camera, adw)

    print("connection 1")
    gui.make_connection(worker)
    print("connection 2")
    worker.make_connection(gui)
    
    #Creamos un Thread y movemos el worker ahi, junto con sus timer, ahi realizamos la conexión
    
    xyThread = QtCore.QThread()
    worker.moveToThread(xyThread)
    worker.viewtimer.moveToThread(xyThread)
    worker.viewtimer.timeout.connect(worker.update)
    xyThread.start()

    # initialize fpar_70, fpar_71, fpar_72 ADwin position parameters
    
    pos_zero = tools.convert(0, 'XtoU')
        
    worker.adw.Set_FPar(70, pos_zero)
    worker.adw.Set_FPar(71, pos_zero)
    worker.adw.Set_FPar(72, pos_zero)
    
    worker.moveTo(10, 10, 10) # in µm
    
    time.sleep(0.200)
        
    gui.setWindowTitle('xyz drift correction test wirh IDS')
    gui.show()
    app.exec_()
        