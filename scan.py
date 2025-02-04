# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:18:19 2018

@author: Luciano A. Masullo
"""

import numpy as np
import time
from datetime import date, datetime
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import tools.tools as tools
import ctypes as ct
from PIL import Image
from tkinter import Tk, filedialog
import tifffile as tiff
import scipy.optimize as opt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter


from threading import Thread

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import qdarkstyle

from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QTabWidget, QGroupBox
from PyQt5 import QtTest


import tools.PSF as PSF

import drivers.ADwin as ADwin
import tools.viewbox_tools as viewbox_tools
import tools.colormaps as cmaps
from tools.lineprofile import linePlotWidget
from swabian import backend as swabian
from tools.PSF_tools import custom_parab, preprare_grid_forfit, parab_func, parag_param_guess, parab_analytical_min

from drivers.minilasevo import MiniLasEvo

π = np.pi

def setupDevice(adw):

    BTL = "ADwin11.btl"
    PROCESS_1 = "line_scan.TB1"
    PROCESS_2 = "moveto_xyz.TB2"
    PROCESS_3 = "actuator_z.TB3"
    PROCESS_4 = "actuator_xy.TB4"
    PROCESS_5 = "flipper.TB5"
    PROCESS_6 = "trace.TB6"
    PROCESS_7 = "shutters.TB7"
    
    btl = adw.ADwindir + BTL
    adw.Boot(btl)

    currdir = os.getcwd()
    process_folder = os.path.join(currdir, "processes")

    process_1 = os.path.join(process_folder, PROCESS_1)
    process_2 = os.path.join(process_folder, PROCESS_2)
    process_3 = os.path.join(process_folder, PROCESS_3)
    process_4 = os.path.join(process_folder, PROCESS_4)
    process_5 = os.path.join(process_folder, PROCESS_5)
    process_6 = os.path.join(process_folder, PROCESS_6)
    process_7 = os.path.join(process_folder, PROCESS_7)
    
    adw.Load_Process(process_1)
    adw.Load_Process(process_2)
    adw.Load_Process(process_3)
    adw.Load_Process(process_4)
    adw.Load_Process(process_5)
    adw.Load_Process(process_6)
    adw.Load_Process(process_7)
    
    
class Frontend(QtGui.QFrame):
    
    paramSignal = pyqtSignal(dict)
    closeSignal = pyqtSignal()
    liveviewSignal = pyqtSignal(bool, str)
    frameacqSignal = pyqtSignal(bool)
    fitPSFSignal = pyqtSignal(np.ndarray, str)

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

#        self.device = 'simulated'
#        self.device = 'nidaq'
#        self.device = 'ADwin'

        self.roi = None
        self.lineROI = None
        self.EBPscatter = [None, None, None, None]
        self.EBPcenters = np.zeros((4, 2))
        self.advanced = False
        self.EBPshown = True
        self.fitting = False
        self.image = np.zeros((128, 128))
        
        self.initialDir = r'C:\Data'
        
        # Define status icons dir
        self.ICON_RED_LED = 'icons\led-red-on.png'
        self.ICON_GREEN_LED = 'icons\green-led-on.png'
        
        # set up GUI

        self.setup_gui()
                
        # connections between changes in parameters and emit_param function
        
        self.NofPixelsEdit.textChanged.connect(self.emit_param)
        self.scanRangeEdit.textChanged.connect(self.emit_param)
        self.pxTimeEdit.textChanged.connect(self.emit_param)
        self.initialPosEdit.textChanged.connect(self.emit_param)
        self.auxAccEdit.textChanged.connect(self.emit_param)
        self.waitingTimeEdit.textChanged.connect(self.emit_param)
        self.detectorType.activated.connect(self.emit_param)
        self.scanMode.activated.connect(self.emit_param)
        self.filenameEdit.textChanged.connect(self.emit_param)
        self.xStepEdit.textChanged.connect(self.emit_param)
        self.yStepEdit.textChanged.connect(self.emit_param)
        self.zStepEdit.textChanged.connect(self.emit_param)
        self.moveToEdit.textChanged.connect(self.emit_param)
        self.powerEdit.textChanged.connect(self.emit_param)
                
    def emit_param(self):
        
        params = dict()
        
        params['detectorType'] = self.detectorType.currentText()
        params['scanType'] = self.scanMode.currentText()
        params['scanRange'] = float(self.scanRangeEdit.text())
        params['NofPixels'] = int(self.NofPixelsEdit.text())
        params['pxTime'] = float(self.pxTimeEdit.text())
        params['initialPos'] = np.array(self.initialPosEdit.text().split(' '),
                                        dtype=np.float64)
        params['a_aux_coeff'] = np.array(self.auxAccEdit.text().split(' '),
                                              dtype=np.float32)/100
        
        params['waitingTime'] = int(self.waitingTimeEdit.text())  # in µs
        params['fileName'] = os.path.join(self.folderEdit.text(),
                                          self.filenameEdit.text())
        params['moveToPos'] = np.array([float(x.strip()) for x in
                                        self.moveToEdit.text().split()],
                                       dtype=np.float16)
        
        params['xStep'] = float(self.xStepEdit.text())
        params['yStep'] = float(self.yStepEdit.text())
        params['zStep'] = float(self.zStepEdit.text())
        params['power'] = float(self.powerEdit.text())
        
        self.paramSignal.emit(params)
        
    @pyqtSlot(dict)
    def get_backend_param(self, params):
        
        frameTime = params['frameTime']
        pxSize = params['pxSize']
        maxCounts = params['maxCounts']
#        initialPos = np.round(params['initialPos'], 2)
        
#        print(datetime.now(), '[scan-frontend] got initialPos', initialPos)
        
        self.frameTimeValue.setText('{}'.format(np.around(frameTime, 2)))
        self.pxSizeValue.setText('{}'.format(np.around(1000 * pxSize, 3))) # in nm
        self.maxCountsValue.setText('{}'.format(maxCounts)) 
#        self.initialPosEdit.setText('{} {} {}'.format(*initialPos))
        
        self.pxSize = pxSize
     
    @pyqtSlot(np.ndarray)
    def get_image(self, image):
        """Update image received from backend."""
        # self.img.setImage(image, autoLevels=False)
        self.image = image
        self.img.setImage(self.image, autoLevels=False)
        self.xaxis.setScale(scale=self.pxSize)  # scale to µm
        self.yaxis.setScale(scale=self.pxSize)  # scale to µm

    @pyqtSlot(np.ndarray)
    def get_real_position(self, val):

      val = np.around(val, 3)
      self.moveToEdit.setText('{} {} {}'.format(*val))
        
    def main_roi(self):
        
        # TO DO: move this function to backend and implement "typedfeat" variables
        
        self.scanRangeEdit.setText('8')
        self.NofPixelsEdit.setText('80')
        self.pxTimeEdit.setText('500')
        self.initialPosEdit.setText('{} {} {}'.format(*[3, 3, 10]))
    
    def moveto_crosshair(self):

        xcenter = int(np.round(self.ch.vLine.value(), 0))
        ycenter = int(np.round(self.ch.hLine.value(), 0))

        halfsizex = int(self.roi.size()[0]/2)
        halfsizey = int(self.roi.size()[1]/2)        
        xmin = xcenter - halfsizex
        ymin = ycenter - halfsizey
        
        if self.roi != None:
            self.roi.setPos(xmin, ymin)
                    
            
    def toggle_advanced(self):
        
        if self.advanced:
            
            self.auxAccelerationLabel.show()
            self.auxAccEdit.show()
            self.waitingTimeLabel.show()
            self.waitingTimeEdit.show() 
            self.preview_scanButton.show()
            
            self.advanced = False
            
        else:
            
            self.auxAccelerationLabel.hide()
            self.auxAccEdit.hide()
            self.waitingTimeLabel.hide()
            self.waitingTimeEdit.hide() 
            self.preview_scanButton.hide()
            
            self.advanced = True

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

    def preview_scan(self):

        plt.figure('Preview scan plot x vs t')
        plt.plot(self.data_t_adwin[0:-1], self.data_x_adwin, 'go')
        plt.xlabel('t (ADwin time)')
        plt.ylabel('V (DAC units)')

        if np.max(self.data_x_adwin) > 2**16:

            plt.plot(self.data_t_adwin[0:-1],
                     2**16 * np.ones(np.size(self.data_t_adwin[0:-1])), 'r-')

    def toggle_liveview(self):

        if self.liveviewButton.isChecked():
            self.liveviewSignal.emit(True, 'liveview')
            if self.roi is not None:
                self.vb.removeItem(self.roi)
                self.roi.hide()
                self.ROIButton.setChecked(False)
            if self.lineROI is not None:
                self.vb.removeItem(self.lineROI)
                self.lplotWidget.hide()
                self.lineProfButton.setChecked(False)
                self.lineROI = None
            else:
                pass
        else:
            self.liveviewSignal.emit(False, 'liveview')
            self.emit_param()

    def toggle_frame_acq(self):

        if self.acquireFrameButton.isChecked():
            self.frameacqSignal.emit(True)
            if self.roi is not None:
                self.vb.removeItem(self.roi)
                self.roi.hide()
                self.ROIButton.setChecked(False)
                self.liveviewButton.setChecked(False)
            if self.lineROI is not None:
                self.vb.removeItem(self.lineROI)
                self.lplotWidget.hide()
                self.lineProfButton.setChecked(False)
                self.lineROI = None
            else:
                pass
        else:
            self.frameacqSignal.emit(False)

    def toggle_tg_ebp_meas(self):
        '''
        This function gets called when the button to measure the time-gated EBP is clicked.
        It emits the signal to start the measurement or, if the measurement is ongoing, to stop it.
        '''
        if self.measure_tg_EBP_button.isChecked():
            self.liveviewSignal.emit(True, 'timegated EBP meas')
            if self.roi is not None:
                self.vb.removeItem(self.roi)
                self.roi.hide()
                self.ROIButton.setChecked(False)
            if self.lineROI is not None:
                self.vb.removeItem(self.lineROI)
                self.lplotWidget.hide()
                self.lineProfButton.setChecked(False)
                self.lineROI = None
            else:
                pass
        else:
            self.liveviewSignal.emit(False, 'timegated EBP meas')
            self.emit_param()

    def toggle_psfscan_fitandmove(self):
        '''
        This function gets called when the button to scan and fit a PSF and move to its center is clicked.
        It emits the signal to start the scan or, if the measurement is ongoing, to stop it.
        Once the scan is done, the forward and back images will be fitted with a parabola and the piezo will move
        to the average of the centers of the two fits.
        '''
        if self.psfFitButton.isChecked():
            self.liveviewSignal.emit(True, 'psf scan fit and move')
            if self.roi is not None:
                self.vb.removeItem(self.roi)
                self.roi.hide()
                self.ROIButton.setChecked(False)
            if self.lineROI is not None:
                self.vb.removeItem(self.lineROI)
                self.lplotWidget.hide()
                self.lineProfButton.setChecked(False)
                self.lineROI = None
            else:
                pass
        else:
            self.liveviewSignal.emit(False, 'psf scan fit and move')
            self.emit_param()

    def line_profile(self):
        if self.lineROI is None:
            if self.roi is None:
                self.lineROI = pg.LineSegmentROI([[10, 64], [70, 64]], pen='r')
            else:
                xmin, ymin = self.roi.pos()
                xmax, ymax = self.roi.pos() + self.roi.size()
                y = int((ymax - ymin)/2) + ymin
                self.lineROI = pg.LineSegmentROI([[xmin, y], [xmax, y]], pen='r')
            self.vb.addItem(self.lineROI)
            self.lplotWidget.show()
        else:
            self.vb.removeItem(self.lineROI)
            if self.roi is None:
                self.lineROI = pg.LineSegmentROI([[10, 64], [70,64]], pen='r')
            else:
                xmin, ymin = self.roi.pos()
                xmax, ymax = self.roi.pos() + self.roi.size()
                y = int((ymax - ymin)/2) + ymin
                self.lineROI = pg.LineSegmentROI([[xmin, y], [xmax, y]], pen='r')
            self.vb.addItem(self.lineROI)
        self.lineROI.sigRegionChanged.connect(self.update_line_profile)

    def update_line_profile(self):
        data = self.lineROI.getArrayRegion(self.image, self.img)
        self.lplotWidget.linePlot.clear()
        #TODO: check how to make legend work that it is always deleted 
        #when plot is updated, otherwise the labels do overlap...
        #self.lplotWidget.linePlot.addLegend()

        px_to_nm = self.pxSize * 1000.0
        x = np.arange(np.size(data)) * px_to_nm
        self.lplotWidget.linePlot.plot(x, data, name='Data')
        
        # make 1D Gauss fit
        if self.lplotWidget.gauss:
            popt, pcov = self.fitGauss1D(data)    
            
            conv_fact = 2*np.sqrt(2*np.log(2))
            fwhm = conv_fact * popt[2]
            fwhm_uncert = conv_fact * pcov[2, 2]
            text = 'FWHM = ' + str(np.round(fwhm, 2)) + ' +/- ' + str(np.round(fwhm_uncert, 2)) + ' nm'      
           
            FWHMlabel = pg.TextItem()
            FWHMlabel.setPos(10, int(np.max(data)/2))
            FWHMlabel.setText(text)
            self.lplotWidget.linePlot.addItem(FWHMlabel)
            self.lplotWidget.linePlot.plot(x, PSF.gaussian1D(x, *popt), pen = pg.mkPen('b'), name='Gauss')
            
        #make 1D doughnut fit
        if self.lplotWidget.doughnut:
            popt, pcov = self.fitDoughnut1D(data)
            
            conv_fact =  1/np.sqrt(np.log(2))
            ptp = conv_fact * popt[2]
            ptp_uncert = conv_fact * pcov[2, 2]
            text = 'Peak-to-peak = ' + str(np.round(ptp, 2)) + ' +/- ' + str(np.round(ptp_uncert, 2)) + ' nm'      
           
            FWHMlabel = pg.TextItem()
            FWHMlabel.setPos(10, int(np.max(data)/2)+30)
            FWHMlabel.setText(text)
            self.lplotWidget.linePlot.addItem(FWHMlabel)
            self.lplotWidget.linePlot.plot(x, PSF.doughnut1D(x, *popt), pen = pg.mkPen('y'), name='Doughnut')
            
    def fitDoughnut1D(self, data):
        px_to_nm = self.pxSize * 1000.0
        x = np.arange(np.size(data)) * px_to_nm
        ofs0 = np.min(data)
        c0 = np.max(data) - ofs0
        peaks = find_peaks(data, distance=15, height=0.75*c0)
        dist = abs(peaks[0][1] - peaks[0][0])
        d = np.sqrt(np.log(2)) * dist * px_to_nm
        x0 = (peaks[0][0] + int(dist/2)) * px_to_nm
        guess = (c0, x0, d, ofs0)
        popt, pcov = opt.curve_fit(PSF.doughnut1D, x, data, p0=guess)

        return popt, pcov
    
    
    def fitGauss1D(self, data):
        px_to_nm = self.pxSize * 1000.0
        x = np.arange(np.size(data)) * px_to_nm
        x0 = np.argmax(data) * px_to_nm
        sigma0 = 400.0 / (2*np.sqrt(2*np.log(2))) #TODO soft-code this estimation
        ofs0 = np.min(data)
        c0 = np.max(data) - ofs0
        guess = (c0, x0, sigma0, ofs0)
        popt, pcov = opt.curve_fit(PSF.gaussian1D, x, data, p0=guess)

        return popt, pcov
        
    def toggle_ROI(self):
        
        self.select_ROIButton.setEnabled(True)
        ROIpen = pg.mkPen(color='y')
        npixels = int(self.NofPixelsEdit.text())

        if self.roi is None:

            ROIpos = (0.5 * npixels - 64, 0.5 * npixels - 64)
            self.roi = viewbox_tools.ROI(npixels/2, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True,
                                         pen=ROIpen)

        else:

            xmin, ymin = self.roi.pos()
            
            self.vb.removeItem(self.roi)
            self.roi.hide()

            ROIpos = (xmin, ymin)
            #ROIpos = (0.5 * npixels - 64, 0.5 * npixels - 64)
            self.roi = viewbox_tools.ROI(npixels/2, self.vb, ROIpos,
                                         handlePos=(1, 0),
                                         handleCenter=(0, 1),
                                         scaleSnap=True,
                                         translateSnap=True,
                                         pen=ROIpen)
            
        if self.EBProiButton.isChecked:
            self.EBProiButton.setChecked(False)
            
    def select_ROI(self):

        self.liveviewButton.setChecked(False)
        self.liveviewSignal.emit(False, 'liveview')
        self.crosshairCheckbox.setChecked(False)
        self.select_ROIButton.setEnabled(False)

        pxSize = self.pxSize
        initialPos = np.array(self.initialPosEdit.text().split(' '),
                              dtype=np.float64)

        newPos_µm = np.array(self.roi.pos()) * pxSize + initialPos[0:2]

        newPos_µm = np.around(newPos_µm, 2)

        self.initialPosEdit.setText('{} {} {}'.format(newPos_µm[0],
                                                      newPos_µm[1],
                                                      initialPos[2]))

        newRange_px = self.roi.size()[1]
        newRange_µm = pxSize * newRange_px
        newRange_µm = np.around(newRange_µm, 2)
        self.scanRangeEdit.setText('{}'.format(newRange_µm))

        self.emit_param()

    def emit_fit_ROI(self, actionType): # TO DO: change name from "fit" to a more general name including fit/move to center
      
      if self.roi is not None:
        
            xmin, ymin = self.roi.pos()
            xmax, ymax = self.roi.pos() + self.roi.size()
                        
            ymin, ymax = [int(self.NofPixelsEdit.text()) - ymax, 
                          int(self.NofPixelsEdit.text()) - ymin]
            
            coordinates = np.array([xmin, xmax, ymin, ymax])  
            

            print('FRONTEND roi ', coordinates)

          
            if actionType == 'fit':
          
                self.fitPSFSignal.emit(coordinates, 'fit')
                
            elif actionType == 'move':
                
                self.fitPSFSignal.emit(coordinates, 'move')
            
      else:
            
            print('[scan] no ROI for the fit was selected')
        
    def set_EBP(self):
        
        pxSize = self.pxSize
        ROIsize = np.array(self.roi.size())
        
        for i in range(4):
        
            if self.EBPscatter[i] is not None:
                
                self.vb.removeItem(self.EBPscatter[i])
        
#        array = self.roi.getArrayRegion(self.scworker.image, self.img)
        ROIsize = np.array(self.roi.size())
        ROIpos_µm = np.array(self.roi.pos()) * pxSize
            
        xmin = ROIpos_µm[0]
        xmax = ROIpos_µm[0] + ROIsize[0] * pxSize
        
        ymin = ROIpos_µm[1]
        ymax = ROIpos_µm[1] + ROIsize[1] * pxSize
        
        x0 = (xmax+xmin)/2
        y0 = (ymax+ymin)/2
        
        if self.EBPtype.currentText() == 'triangle':
        
            L = int(self.LEdit.text())/1000 # in µm
            θ = π * np.array([1/6, 5/6, 3/2])
            ebp = (L/2)*np.array([[0, 0], [np.cos(θ[0]), np.sin(θ[0])], 
                                 [np.cos(θ[1]), np.sin(θ[1])], 
                                 [np.cos(θ[2]), np.sin(θ[2])]])
            
            self.EBP = (ebp + np.array([x0, y0]))/pxSize
                                       
        print('[scan] EBP px', self.EBP)
            
        for i in range(4):
        
            if i == 0:
                mybrush = pg.mkBrush(255, 127, 80, 255)
                
            if i == 1:
                mybrush = pg.mkBrush(255, 255, 0, 255)
                
            if i == 2:
                mybrush = pg.mkBrush(135, 206, 235, 255)
                
            if i == 3:
                mybrush = pg.mkBrush(0, 0 ,0 , 255)
                
            self.EBPscatter[i] = pg.ScatterPlotItem([self.EBP[i][0]], 
                                                    [self.EBP[i][1]], 
                                                    size=10,
                                                    pen=pg.mkPen(None), 
                                                    brush=mybrush)            

            self.vb.addItem(self.EBPscatter[i])
        
        self.set_EBPButton.setChecked(False)
        
    def toggle_EBP(self):
        
        if self.EBPshown:
        
            for i in range(len(self.EBPscatter)):
                
                if self.EBPscatter[i] is not None:
                    self.vb.removeItem(self.EBPscatter[i])
                else:
                    pass
            
            self.EBPshown = False
            
        else:
            
            for i in range(len(self.EBPscatter)):
                
                if self.EBPscatter[i] is not None:
                    self.vb.addItem(self.EBPscatter[i])
                else:
                    pass
            
            self.EBPshown = True
    
        self.showEBPButton.setChecked(False)
    
    @pyqtSlot(int, bool)    
    def update_shutters(self, num, on):
        
        '''
        setting of num-value:
            0 - signal send by scan-gui-button --> change state of all minflux shutters
            1...6 - shutter 1-6 will be set according to on-variable, i.e. either true or false; only 1-4 controlled from here
            7 - set all minflux shutters according to on-variable
            8 - set all shutters according to on-variable
        '''
        
        if num == 0:
            if self.shutterButton.isChecked():
                if self.shutter1Checkbox.isChecked() == False:
                    self.shutter1Checkbox.setChecked(True)
                if self.shutter2Checkbox.isChecked() == False:
                    self.shutter2Checkbox.setChecked(True)
                if self.shutter3Checkbox.isChecked() == False:
                    self.shutter3Checkbox.setChecked(True)
                if self.shutter4Checkbox.isChecked() == False:
                    self.shutter4Checkbox.setChecked(True)    
            else:
                if self.shutter1Checkbox.isChecked():
                    self.shutter1Checkbox.setChecked(False)
                if self.shutter2Checkbox.isChecked():
                    self.shutter2Checkbox.setChecked(False)
                if self.shutter3Checkbox.isChecked():
                    self.shutter3Checkbox.setChecked(False)
                if self.shutter4Checkbox.isChecked():
                    self.shutter4Checkbox.setChecked(False)
        
        if num == 1:
            self.shutter1Checkbox.setChecked(on)
        
        if num == 2:
            self.shutter2Checkbox.setChecked(on)
                
        if num == 3:
            self.shutter3Checkbox.setChecked(on)
                
        if num == 4:
            self.shutter4Checkbox.setChecked(on)
                
        if (num == 7) or (num == 8):
            if self.shutterButton.isChecked():
                self.shutterButton.setChecked(False)
            self.shutter1Checkbox.setChecked(on)
            self.shutter2Checkbox.setChecked(on)
            self.shutter3Checkbox.setChecked(on)
            self.shutter4Checkbox.setChecked(on)
            
        if num == 11:
            self.diodeShutter.setChecked(on)
            
    @pyqtSlot(bool)    
    def update_led(self, emission):
        if emission:
            led = self.ICON_GREEN_LED
        else:
            led = self.ICON_RED_LED
            
        self.diodeemissionStatus.setPixmap(QtGui.QPixmap(led))
        self.diodeemissionStatus.setScaledContents(True)
        self.diodeemissionStatus.setFixedSize(20, 20)
        
        #whenever diodelaser status is changed,
        #the output power will be set to 0 mW for security reasons
        self.diodepowerSpinBox.setValue(0)
            
    def setup_gui(self):
                
        # set up axis items, scaling is performed in get_image()
        self.xaxis = pg.AxisItem(orientation='bottom', maxTickLength=5)
        self.xaxis.showLabel(show=True)
        self.xaxis.setLabel('x', units='µm')
        
        self.yaxis = pg.AxisItem(orientation='left', maxTickLength=5)
        self.yaxis.showLabel(show=True)
        self.yaxis.setLabel('y', units='µm')

        # image widget set-up and layout
        imageWidget = pg.GraphicsLayoutWidget()
        self.vb = imageWidget.addPlot(axisItems={'bottom': self.xaxis, 
                                                 'left': self.yaxis})
        self.lplotWidget = linePlotWidget()
        self.lplotWidget.get_scanConnection(self)
        self.lplotWidget.gauss = False
        self.lplotWidget.doughnut = False
        
        imageWidget.setFixedHeight(500)
        imageWidget.setFixedWidth(500)
        
        # Viewbox and image item where the liveview will be displayed

        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        
        #create crosshair
        #double .vb in order to get actual viewbox and not just the thingy 
        #we called viewbox ;)
        vbox = self.vb.vb
        self.ch = viewbox_tools.Crosshair(vbox)

        # set up histogram for the liveview image

        self.hist = pg.HistogramLUTItem(image=self.img)
        # lut = viewbox_tools.generatePgColormap(cmaps.parula)
        # self.hist.gradient.setColorMap(lut)
        self.hist.gradient.loadPreset('viridis')
        self.hist.vb.setLimits(yMin=0, yMax=10000)

        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)
        
        # widget with scanning parameters
        self.paramWidget = QGroupBox('Scan Settings')
    
        # LiveView Button

        self.liveviewButton = QtGui.QPushButton('Confocal Liveview')
        self.liveviewButton.setFont(QtGui.QFont('Helvetica', weight=QtGui.QFont.Bold))
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.setStyleSheet("font-size: 12px; background-color:rgb(180, 180, 180)")
        self.liveviewButton.clicked.connect(self.toggle_liveview)
        
        # Time gated EBP measurement button
        
        self.measure_tg_EBP_button = QtGui.QPushButton('Measure time-gated EBP')
        self.measure_tg_EBP_button.setFont(QtGui.QFont('Helvetica', weight=QtGui.QFont.Bold))
        self.measure_tg_EBP_button.setCheckable(True)
        #self.measure_tg_EBP_button.setStyleSheet("font-size: 12px; background-color:rgb(180, 180, 180)")
        self.measure_tg_EBP_button.clicked.connect(self.toggle_tg_ebp_meas)
        
        # ROI buttons

        self.ROIButton = QtGui.QPushButton('ROI')
        self.ROIButton.setCheckable(True)
        self.ROIButton.clicked.connect(self.toggle_ROI)

        self.select_ROIButton = QtGui.QPushButton('Select ROI')
        self.select_ROIButton.clicked.connect(self.select_ROI)
        self.select_ROIButton.setEnabled(False)

        
        #Shutters
        self.shutterLabel = QtGui.QLabel('Minflux shutter open?')
        self.shutter1Checkbox = QtGui.QCheckBox('1')
        self.shutter2Checkbox = QtGui.QCheckBox('2')
        self.shutter3Checkbox = QtGui.QCheckBox('3')
        self.shutter4Checkbox = QtGui.QCheckBox('4')
      
        # Shutter button
        
        self.shutterButton = QtGui.QPushButton('Open/ close all')
        self.shutterButton.setCheckable(True)
        self.shutterButton.clicked.connect(lambda: self.update_shutters(0, True))
        
        # Flipper button
        
        self.flipperButton = QtGui.QPushButton('Flipper 100x up/down')
        self.flipperButton.setCheckable(True)
        
        # Save current frame button

        self.currentFrameButton = QtGui.QPushButton('Save current frame')

        # preview scan button

        self.preview_scanButton = QtGui.QPushButton('Scan preview')
        self.preview_scanButton.setCheckable(True)
        self.preview_scanButton.clicked.connect(self.preview_scan)
        
        # toggle crosshair
        
        self.crosshairCheckbox = QtGui.QCheckBox('Crosshair')
        self.crosshairCheckbox.stateChanged.connect(self.ch.toggle)
        
        
        # move to center button
        
        self.moveToROIcenterButton = QtGui.QPushButton('Move to ROI center') 
        self.moveToROIcenterButton.clicked.connect(lambda: self.emit_fit_ROI('move'))
        
        # dougnhut fit
        
        self.psfFitButton = QtGui.QPushButton('PSF fit and move')
        self.psfFitButton.setCheckable(True)
        self.psfFitButton.clicked.connect(self.toggle_psfscan_fitandmove)
        
        # measure trace
        
        self.traceButton = QtGui.QPushButton('Measure trace')
        
        # main ROI button
        
        self.mainROIButton = QtGui.QPushButton('Go to main ROI') 
        self.mainROIButton.clicked.connect(self.main_roi)

        # line profile button
        
        self.lineProfButton = QtGui.QPushButton('Line profile')
        self.lineProfButton.setCheckable(True)
        self.lineProfButton.clicked.connect(self.line_profile)
        
        # edited scan button
        
        self.FBavScanButton = QtGui.QPushButton('F and B average scan')
        self.FBavScanButton.setCheckable(True)
        
        # move to crosshair button
        
        self.crosshairButton = QtGui.QPushButton('Move to Crosshair')
        self.crosshairButton.clicked.connect(self.moveto_crosshair)


        # Scanning parameters

        self.initialPosLabel = QtGui.QLabel('Initial Pos'
                                            ' [x0, y0, z0] (µm)')
        self.initialPosEdit = QtGui.QLineEdit('3 3 10')
        self.scanRangeLabel = QtGui.QLabel('Scan range (µm)')
        self.scanRangeEdit = QtGui.QLineEdit('8')
        self.pxTimeLabel = QtGui.QLabel('Pixel time (µs)')
        self.pxTimeEdit = QtGui.QLineEdit('500')
        self.NofPixelsLabel = QtGui.QLabel('Number of pixels')
        self.NofPixelsEdit = QtGui.QLineEdit('80')
        
        self.pxSizeLabel = QtGui.QLabel('Pixel size (nm)')
        self.pxSizeValue = QtGui.QLineEdit('')
        self.pxSizeValue.setReadOnly(True)
        self.frameTimeLabel = QtGui.QLabel('Frame time (s)')
        self.frameTimeValue = QtGui.QLineEdit('')
        self.frameTimeValue.setReadOnly(True)
        self.maxCountsLabel = QtGui.QLabel('Max counts per pixel')
        self.maxCountsValue = QtGui.QLineEdit('')
        self.frameTimeValue.setReadOnly(True)
        
        self.powerLabel = QtGui.QLabel('Power at BFP (µW)')
        self.powerEdit = QtGui.QLineEdit('0')
        
        self.advancedButton = QtGui.QPushButton('Advanced options')
        self.advancedButton.setCheckable(True)
        self.advancedButton.clicked.connect(self.toggle_advanced)
        
        self.auxAccelerationLabel = QtGui.QLabel('Aux acc'
                                                 ' (% of a_max)')
        self.auxAccEdit = QtGui.QLineEdit('1 1 1 1')
        self.waitingTimeLabel = QtGui.QLabel('Scan waiting time (µs)')
        self.waitingTimeEdit = QtGui.QLineEdit('0')
        
        self.toggle_advanced()
    

        # file/folder widget
        self.fileWidget = QGroupBox('Save options')        
        self.fileWidget.setMinimumWidth(180)
        self.fileWidget.setFixedHeight(110)

        # folder and buttons
        today = str(date.today()).replace('-', '')
        root = r'C:\\Data\\'
        folder = root + today
        
        try:  
            os.mkdir(folder)
        except OSError:  
            print(datetime.now(), '[scan] Directory {} already exists'.format(folder))
        else:  
            print(datetime.now(), '[scan] Successfully created the directory {}'.format(folder))
        
        self.filenameEdit = QtGui.QLineEdit('filename')
        self.folderLabel = QtGui.QLabel('Folder')
        self.folderEdit = QtGui.QLineEdit(folder)
        self.browseFolderButton = QtGui.QPushButton('Browse folder')
        self.browseFolderButton.setCheckable(True)
        self.browseFolderButton.clicked.connect(self.load_folder)

        # scan selection
        self.scanModeLabel = QtGui.QLabel('Scan type')

        self.scanMode = QtGui.QComboBox()
        self.scanModes = ['xy', 'xz', 'yz']
        self.scanMode.addItems(self.scanModes)
        
        self.detectorType = QtGui.QComboBox()
        self.dettypes = ['APD','photodiode']
        self.detectorType.addItems(self.dettypes)
        
        # diodelaser widget
        diodelaserWidget = QGroupBox('Diodelaser control')
        diodelaserWidget.setFixedHeight(108)   
        
        self.diodelaserButton = QtGui.QPushButton('Laser On')
        self.diodelaserButton.setCheckable(True)
        self.diodeemissionLabel = QtGui.QLabel('Emission')
        
        self.diodeemissionStatus = QtGui.QLabel()
        self.diodeemissionStatus.setPixmap(QtGui.QPixmap(self.ICON_RED_LED))
        self.diodeemissionStatus.setScaledContents(True)
        self.diodeemissionStatus.setFixedSize(20, 20)
        
        self.diodepowerLabel = QtGui.QLabel('Power [mW]')
        self.diodepowerSpinBox = QtGui.QSpinBox()
        self.diodepowerSpinBox.setRange(0, 78) #max value given by manual  
        
        self.diodeShutter = QtGui.QCheckBox('Open')
        
        diode_subgrid = QtGui.QGridLayout()
        diodelaserWidget.setLayout(diode_subgrid)
        
        diode_subgrid.addWidget(self.diodelaserButton, 0, 0)
        diode_subgrid.addWidget(self.diodeShutter, 0, 1)
        diode_subgrid.addWidget(self.diodeemissionLabel, 1, 0)
        diode_subgrid.addWidget(self.diodeemissionStatus, 1, 1)
        diode_subgrid.addWidget(self.diodepowerLabel, 2, 0)
        diode_subgrid.addWidget(self.diodepowerSpinBox, 2, 1)
      
        
        # widget with EBP parameters and buttons
        self.EBPWidget = QtGui.QFrame()
        
        EBPparamTitle = QtGui.QLabel('<h2><strong>Excitation Beam Pattern</strong></h2>')
        EBPparamTitle.setTextFormat(QtCore.Qt.RichText)
        
        self.EBProiButton = QtGui.QPushButton('EBP ROI')
        self.EBProiButton.setCheckable(True)
        self.EBProiButton.clicked.connect(self.toggle_ROI)
        
        self.showEBPButton = QtGui.QPushButton('show/hide EBP')
        self.showEBPButton.setCheckable(True)
        self.showEBPButton.clicked.connect(self.toggle_EBP)

        self.set_EBPButton = QtGui.QPushButton('set EBP')
        self.set_EBPButton.clicked.connect(self.set_EBP)
        
        # EBP selection
        self.EBPtypeLabel = QtGui.QLabel('EBP type')

        self.EBPtype = QtGui.QComboBox()
        self.EBPoptions = ['triangle', 'square']
        self.EBPtype.addItems(self.EBPoptions)
        
        self.Llabel = QtGui.QLabel('L (nm)')
        self.LEdit = QtGui.QLineEdit('100')
        
        # piezo navigation widget
        
        self.positioner = QtGui.QFrame()
        self.positioner.setFrameStyle(QtGui.QFrame.Panel |
                                      QtGui.QFrame.Raised)
        
        self.xUpButton = QtGui.QPushButton("(+x) ►")  # →
        self.xDownButton = QtGui.QPushButton("◄ (-x)")  # ←

        self.yUpButton = QtGui.QPushButton("(+y) ▲")  # ↑
        self.yDownButton = QtGui.QPushButton("(-y) ▼")  # ↓
        
        self.zUpButton = QtGui.QPushButton("(+z) ▲")  # ↑
        self.zDownButton = QtGui.QPushButton("(-z) ▼")  # ↓
        
        self.xStepLabel = QtGui.QLabel('x step (µm)')
        self.xStepEdit = QtGui.QLineEdit('0.050')
        
        self.yStepLabel = QtGui.QLabel('y step (µm)')
        self.yStepEdit = QtGui.QLineEdit('0.050')
        
        self.zStepLabel = QtGui.QLabel('z step (µm)')
        self.zStepEdit = QtGui.QLineEdit('0.050')
        
        # move to button

        self.moveToButton = QtGui.QPushButton('Move to')
        self.moveToLabel = QtGui.QLabel('Move to [x0, y0, z0] (µm)')
        self.moveToEdit = QtGui.QLineEdit('0 0 10')
        
        # lower widget displaying file and diodelaser widgets
        lowerWidget = QtGui.QFrame()
        lower_subgrid = QtGui.QGridLayout()
        lowerWidget.setLayout(lower_subgrid)
        lowerWidget.setMinimumWidth(450)
        lower_subgrid.addWidget(self.fileWidget, 0, 0)
        lower_subgrid.addWidget(diodelaserWidget, 0, 1)
        
        # scan GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        
        tabArea = QTabWidget()
        grid.addWidget(tabArea, 0, 0)
        
        paramTab = QtGui.QFrame()
        paramTabGrid = QtGui.QGridLayout()
        paramTabGrid.addWidget(self.paramWidget, 0, 0)
        paramTabGrid.addWidget(lowerWidget, 1, 0)
        paramTab.setLayout(paramTabGrid)

        tabArea.addTab(paramTab, "Scan parameters")
        tabArea.addTab(self.positioner, "Positioner")
        tabArea.addTab(self.EBPWidget, "EBP")
        
        grid.addWidget(imageWidget, 0, 1)
        
        # parameters widget layout

        subgrid = QtGui.QGridLayout()
        self.paramWidget.setLayout(subgrid)
                
        subgrid.addWidget(self.scanModeLabel, 2, 2)
        subgrid.addWidget(self.scanMode, 3, 2)
        subgrid.addWidget(self.detectorType, 4, 2)
        subgrid.addWidget(self.liveviewButton, 5, 2)
        subgrid.addWidget(self.currentFrameButton, 5, 3)
        subgrid.addWidget(self.measure_tg_EBP_button, 6, 2)
        
        subgrid.addWidget(self.flipperButton, 7, 2)
        #TODO check whether we keep this button
        subgrid.addWidget(self.crosshairCheckbox, 6, 3)
        subgrid.addWidget(self.crosshairButton, 7, 3)
        
        subgrid.addWidget(self.ROIButton, 8, 2)
        subgrid.addWidget(self.select_ROIButton, 9, 2)
        subgrid.addWidget(self.moveToROIcenterButton, 10, 2)
        subgrid.addWidget(self.mainROIButton, 11, 2)
        
        subgrid.addWidget(self.lineProfButton, 8, 3)
        subgrid.addWidget(self.FBavScanButton, 9, 3)
        subgrid.addWidget(self.psfFitButton, 10, 3)
        subgrid.addWidget(self.traceButton, 11, 3)
        
        subgrid.addWidget(self.initialPosLabel, 2, 0, 1, 2)
        subgrid.addWidget(self.initialPosEdit, 3, 0, 1, 2)
        subgrid.addWidget(self.scanRangeLabel, 4, 0)
        subgrid.addWidget(self.scanRangeEdit, 4, 1)
        subgrid.addWidget(self.pxTimeLabel, 5, 0)
        subgrid.addWidget(self.pxTimeEdit, 5, 1)
        subgrid.addWidget(self.NofPixelsLabel, 6, 0)
        subgrid.addWidget(self.NofPixelsEdit, 6, 1)
        
        subgrid.addWidget(self.pxSizeLabel, 7, 0)
        subgrid.addWidget(self.pxSizeValue, 7, 1)
        subgrid.addWidget(self.frameTimeLabel, 8, 0)
        subgrid.addWidget(self.frameTimeValue, 8, 1)
        subgrid.addWidget(self.maxCountsLabel, 9, 0)
        subgrid.addWidget(self.maxCountsValue, 9, 1)
        subgrid.addWidget(self.powerLabel, 10, 0)
        subgrid.addWidget(self.powerEdit, 10, 1)
        
        subgrid.addWidget(self.advancedButton, 11, 0)
        subgrid.addWidget(self.auxAccelerationLabel, 12, 0)
        subgrid.addWidget(self.auxAccEdit, 12, 1)
        subgrid.addWidget(self.waitingTimeLabel, 13, 0)
        subgrid.addWidget(self.waitingTimeEdit, 13, 1)
        subgrid.addWidget(self.preview_scanButton, 13, 2)
        
        subgrid.addWidget(self.shutterLabel, 14, 0)
        subgrid.addWidget(self.shutter1Checkbox, 15, 0)
        subgrid.addWidget(self.shutter2Checkbox, 15, 1)
        subgrid.addWidget(self.shutter3Checkbox, 15, 2)
        subgrid.addWidget(self.shutter4Checkbox, 15, 3)
        subgrid.addWidget(self.shutterButton, 16, 0)
        
        self.paramWidget.setFixedHeight(398)
        self.paramWidget.setMinimumWidth(450)
        
#        subgrid.setColumnMinimumWidth(1, 130)
#        subgrid.setColumnMinimumWidth(1, 50)
        
        # file/folder widget layout
        
        file_subgrid = QtGui.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)
        
        file_subgrid.addWidget(self.filenameEdit, 0, 0)
        #file_subgrid.addWidget(self.folderLabel, 1, 0)
        file_subgrid.addWidget(self.folderEdit, 2, 0)
        file_subgrid.addWidget(self.browseFolderButton, 1, 0)
        
        # EBP widget layout
        
        subgridEBP = QtGui.QGridLayout()
        self.EBPWidget.setLayout(subgridEBP)
        
        subgridEBP.addWidget(EBPparamTitle, 0, 0, 2, 4)
        
        subgridEBP.addWidget(self.EBProiButton, 2, 0, 1, 1)
        subgridEBP.addWidget(self.set_EBPButton, 3, 0, 1, 1)
        subgridEBP.addWidget(self.showEBPButton, 4, 0, 2, 1)
        subgridEBP.addWidget(self.EBPtypeLabel, 2, 1)
        subgridEBP.addWidget(self.EBPtype, 3, 1)
        subgridEBP.addWidget(self.Llabel, 4, 1)
        subgridEBP.addWidget(self.LEdit, 5, 1)
        
        self.EBPWidget.setFixedHeight(140)
        self.EBPWidget.setMinimumWidth(250)
        
        # Piezo navigation widget layout

        layout = QtGui.QGridLayout()
        self.positioner.setLayout(layout)
        
        positionerTitle = QtGui.QLabel('<h2><strong>Position</strong></h2>')
        positionerTitle.setTextFormat(QtCore.Qt.RichText)
        
        layout.addWidget(positionerTitle, 0, 0, 2, 3)
        layout.addWidget(self.xUpButton, 2, 4, 2, 1)
        layout.addWidget(self.xDownButton, 2, 2, 2, 1)
        
        layout.addWidget(self.xStepLabel, 0, 6)        
        layout.addWidget(self.xStepEdit, 1, 6)
        
        layout.addWidget(self.yUpButton, 1, 3, 2, 1)
        layout.addWidget(self.yDownButton, 3, 3, 2, 1)
        
        layout.addWidget(self.yStepLabel, 2, 6)        
        layout.addWidget(self.yStepEdit, 3, 6)

        layout.addWidget(self.zUpButton, 1, 5, 2, 1)
        layout.addWidget(self.zDownButton, 3, 5, 2, 1)
        
        layout.addWidget(self.zStepLabel, 4, 6)        
        layout.addWidget(self.zStepEdit, 5, 6)
        
        layout.addWidget(self.moveToLabel, 6, 1, 1, 3)
        layout.addWidget(self.moveToEdit, 7, 1, 1, 2)
        layout.addWidget(self.moveToButton, 8, 1, 1, 2)
        
        self.positioner.setFixedHeight(250)
        self.positioner.setMinimumWidth(400)
        
    # make connections between GUI and worker functions
            
    def make_connection(self, backend):
        
        backend.paramSignal.connect(self.get_backend_param)
        backend.imageSignal.connect(self.get_image)
        backend.realPositionSignal.connect(self.get_real_position)
        backend.shuttermodeSignal.connect(self.update_shutters)
        backend.diodelaserEmissionSignal.connect(self.update_led)
        backend.ebp_measurement_done.connect(lambda: self.measure_tg_EBP_button.setChecked(False))
        backend.psf_scanandfit_done.connect(lambda: self.psfFitButton.setChecked(False))
        
    def closeEvent(self, *args, **kwargs):

        # Emit close signal
        
        self.closeSignal.emit()
        scanThread.exit()
        super().closeEvent(*args, **kwargs)
        app.quit()



class Backend(QtCore.QObject):

    paramSignal = pyqtSignal(dict)
    imageSignal = pyqtSignal(np.ndarray)
    frameIsDone = pyqtSignal(bool, np.ndarray, int, int)
    ebp_measurement_done = pyqtSignal()
    psf_scanandfit_done = pyqtSignal()
    ROIcenterSignal = pyqtSignal(np.ndarray)
    realPositionSignal = pyqtSignal(np.ndarray)
    auxFitSignal = pyqtSignal()
    auxMoveSignal = pyqtSignal()
    shuttermodeSignal = pyqtSignal(int, bool)
    diodelaserEmissionSignal = pyqtSignal(bool)
    focuslockpositionSignal = pyqtSignal(float)

    """
    Signals
    
    - paramSignal:
         To: [frontend]
         Description: 
             
    - imageSignal:
         To: [frontend]
         Description: 
        
    - frameIsDone:
         To: [psf]
         Description: 
        
    - ROIcenterSignal:
         To: [minflux]
         Description:
        
    """
    
    def __init__(self, adwin, diodelaser, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.adw = adwin
        self.diodelas = diodelaser
        self.saveScanData = False
        self.feedback_active = False
        self.flipper_state = False
        self.laserstate = False        

        # full_scan: True --> full scan including aux parts
        # full_scan: False --> forward part of the scan

        self.full_scan = False
        self.FBaverage_scan = False
        self.time_gated_EBP = False

        # 5MHz is max count rate of the P. Elmer APD
        
        self.APDmaxCounts = 5*10**6   

        # Create a timer for the update of the liveview

        self.viewtimer = QtCore.QTimer()

        # Counter for the saved images

        self.imageNumber = 0

        # initialize flag for the linescan function

        self.flag = 0
        
        # initialize fpar_70, fpar_71, fpar_72 ADwin position parameters
        
        pos_zero = tools.convert(0, 'XtoU')
        
        self.adw.Set_FPar(70, pos_zero)
        self.adw.Set_FPar(71, pos_zero)
        self.adw.Set_FPar(72, pos_zero)
        
        # move to z = 10 µm

        self.moveTo(3, 3, 10)

        # initial directory

        self.initialDir = r'C:\Data'
        
        # initialize image
        
        self.image = None
        
    @pyqtSlot(dict)
    def get_frontend_param(self, params):
        # updates parameters according to what is input in the GUI
        self.detector = params['detectorType']
        self.scantype = params['scanType']
        self.scanRange = params['scanRange']
        self.NofPixels = int(params['NofPixels'])
        self.pxTime = params['pxTime']
        self.initialPos = params['initialPos']
        self.powerBFP = params['power']
        self.waitingTime = params['waitingTime']
        self.a_aux_coeff = params['a_aux_coeff']
        self.filename = params['fileName']
        self.moveToPos = params['moveToPos']
        self.xStep = params['xStep']
        self.yStep = params['yStep']
        self.zStep = params['zStep']
        # self.selectedCoord = params['ROIcoordinates']
        # print('[scan] selected ROI coordinates are:', self.selectedCoord)
        self.calculate_derived_param()

    @pyqtSlot(np.ndarray, str)
    def get_ROI_coords_and_fit(self, array, actionType):
        self.selectedCoord = array
        print('[scan] selected fit ROI coordinates are:', self.selectedCoord)
        if actionType == 'fit':
            self.auxFitSignal.emit()
        elif actionType == 'move':
            self.auxMoveSignal.emit()

    def calculate_derived_param(self):
        # TODO: check whether we can delete this.
        # caused saving zeros in some cases
        # self.image_to_save = self.image

        self.pxSize = self.scanRange/self.NofPixels   # in µm
        self.frameTime = self.NofPixels**2 * self.pxTime / 10**6
        self.maxCounts = int(self.APDmaxCounts/(1/(self.pxTime*10**-6)))
        self.linetime = (1/1000)*self.pxTime*self.NofPixels  # in ms

        #  aux scan parameters
        self.a_max = 4 * 10**-6  # in µm/µs^2

        if np.all(self.a_aux_coeff) <= 1:
            self.a_aux = self.a_aux_coeff * self.a_max
        else:
            self.a_aux[self.a_aux > 1] = self.a_max

        self.NofAuxPixels = 100

        self.waiting_pixels = int(self.waitingTime/self.pxTime)
        self.tot_pixels = (2 * self.NofPixels + 4 * self.NofAuxPixels +
                           self.waiting_pixels)

        # create scan signal
        self.dy = self.pxSize
        self.dz = self.pxSize

        (self.data_t, self.data_x,
         self.data_y) = tools.ScanSignal(self.scanRange,
                                         self.NofPixels,
                                         self.NofAuxPixels,
                                         self.pxTime,
                                         self.a_aux,
                                         self.dy,
                                         self.initialPos[0],
                                         self.initialPos[1],
                                         self.initialPos[2],
                                         self.scantype,
                                         self.waitingTime)

#        self.viewtimer_time = (1/1000) * self.data_t[-1]    # in ms

        self.viewtimer_time = 0  # timer will timeout as soon after it has executed all functions

        # Create blank image
        # full_scan = True --> size of the full scan including aux parts 
        # full_scan = False --> size of the forward part of the scan
        if self.full_scan is True:
            size = (self.tot_pixels, self.tot_pixels)
        else:
            size = (self.NofPixels, self.NofPixels)

#        self.blankImage = np.zeros(size)
        self.image = np.zeros(size)
        self.imageF = np.zeros(size)
        self.imageB = np.zeros(size)

        self.i = 0

        # load the new parameters into the ADwin system
        self.update_device_param()

        # emit calculated parameters
        self.emit_param()

    def emit_param(self):
        params = dict()
        params['frameTime'] = self.frameTime
        params['pxSize'] = self.pxSize
        params['maxCounts'] = self.maxCounts
#        params['initialPos'] = np.float64(self.initialPos)
        self.paramSignal.emit(params)

    def update_device_param(self):

        if self.detector == 'APD':
            self.adw.Set_Par(3, 0)  # Digital input (APD)
        if self.detector == 'photodiode':
            self.adw.Set_Par(3, 1)  # Analog input (photodiode)
        # select scan type
        if self.scantype == 'xy':
            self.adw.Set_FPar(10, 1)
            self.adw.Set_FPar(11, 2)
        if self.scantype == 'xz':
            self.adw.Set_FPar(10, 1)
            self.adw.Set_FPar(11, 6)
        if self.scantype == 'yz':
            self.adw.Set_FPar(10, 2)
            self.adw.Set_FPar(11, 6)
        #  initial positions x and y
        self.x_i = self.initialPos[0]
        self.y_i = self.initialPos[1]
        self.z_i = self.initialPos[2]

        self.x_offset = 0
        self.y_offset = 0
        self.z_offset = 0

        #  load ADwin parameters

        self.adw.Set_Par(1, self.tot_pixels)

        self.data_t_adwin = tools.timeToADwin(self.data_t)
        self.data_x_adwin = tools.convert(self.data_x, 'XtoU')
        self.data_y_adwin = tools.convert(self.data_y, 'XtoU')

        # repeat last element because time array has to have one more
        # element than position array

        dt = self.data_t_adwin[-1] - self.data_t_adwin[-2]

        self.data_t_adwin = np.append(self.data_t_adwin,
                                      (self.data_t_adwin[-1] + dt))

        # prepare arrays for conversion into ADwin-readable data

        self.time_range = np.size(self.data_t_adwin)
        self.space_range = np.size(self.data_x_adwin)

        self.data_t_adwin = np.array(self.data_t_adwin, dtype='int')
        self.data_x_adwin = np.array(self.data_x_adwin, dtype='int')
        self.data_y_adwin = np.array(self.data_y_adwin, dtype='int')

        self.data_t_adwin = list(self.data_t_adwin)
        self.data_x_adwin = list(self.data_x_adwin)
        self.data_y_adwin = list(self.data_y_adwin)

        self.adw.SetData_Long(self.data_t_adwin, 2, 1, self.time_range)
        self.adw.SetData_Long(self.data_x_adwin, 3, 1, self.space_range)
        self.adw.SetData_Long(self.data_y_adwin, 4, 1, self.space_range)

    def set_moveTo_param(self, x_f, y_f, z_f, n_pixels_x=128, n_pixels_y=128,
                         n_pixels_z=128, pixeltime=2000):

        x_f = tools.convert(x_f, 'XtoU')
        y_f = tools.convert(y_f, 'XtoU')
        z_f = tools.convert(z_f, 'XtoU')

        print("moviendo a ", x_f, y_f, z_f)

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

    def moveTo_action(self):
        self.moveTo(*self.moveToPos)

    def moveTo_roi_center(self):
        xi, xf, yi, yf = self.selectedCoord
        self.ROIcenter = self.initialPos + np.array([(xf+xi)/2, (yf+yi)/2, 0]) * self.pxSize

        # print('[scan] self.initialPos[0:2]', self.initialPos[0:2])
        print('[scan] moved to center of ROI:', self.ROIcenter, 'µm')

        self.moveTo(*self.ROIcenter)
        time.sleep(.3)
        self.ROIcenterSignal.emit(self.ROIcenter)

        self.trace_measurement()

    def psf_parab_fit_andmove(self):
        """
        This function is called once the single scan triggered by the psf fit and move button is done.
        It takes the two images, forward and back, resulting from the single scan and perform a parabolic fit.
        It then asks the piezo to go to the aritmetic average of the centers of the two fitted parabolas,
        where the molecule should be, with no offset.
        Before fitting, it guesses the position of the center of the PSF from the numerical minimum of the images.
        To avoid finding wrong minima, it performs a preliminary gaussian smoothing of the data.
        The smoothed image is used only to find the guess for the center, while the fit is performed on the raw one.
        it also checks that the final target position falls inside the ROI to avoid that a wrong fit sends the piezo far away. 
        """
        print("entering fit function")
        # size of the domain where to perform the fit
        self.radius_forfit_nm = 100
        # smoothing images to find minimum numerically
        self.imageF_smooth = gaussian_filter(self.imageF[5:-5,5:-5], sigma=3)
        self.imageB_smooth = gaussian_filter(self.imageB[5:-5,5:-5], sigma=3)
        # finding numerical minimum
        self.min_pos_psfF = np.unravel_index(np.argmin(self.imageF_smooth, axis=None), self.imageF_smooth.shape)
        self.min_pos_psfF = (self.min_pos_psfF[0] + 5, self.min_pos_psfF[1] + 5)
        self.min_pos_psfB = np.unravel_index(np.argmin(self.imageB_smooth, axis=None), self.imageB_smooth.shape)
        self.min_pos_psfB = (self.min_pos_psfB[0] + 5, self.min_pos_psfB[1] + 5)
        print("self.min_pos_psfF: ", self.min_pos_psfF)
        print("self.min_pos_psfB: ", self.min_pos_psfB)
        print("self.min_pos_psfF in nm: ", (self.min_pos_psfF[0] * self.pxSize, self.min_pos_psfF[1] * self.pxSize))
        print("self.min_pos_psfB in nm: ", (self.min_pos_psfB[0] * self.pxSize, self.min_pos_psfB[1] * self.pxSize))
        # computing the side of the square to cut the image for fit, in px
        self.half_side_square_forfit_px = int(self.radius_forfit_nm / (self.pxSize * 1e3))
        # getting the boundaries for cutting the images
        self.psfF_hole_xlims = [
            np.max((0, self.min_pos_psfF[0] - self.half_side_square_forfit_px)),
            np.min((self.min_pos_psfF[0] + self.half_side_square_forfit_px, self.NofPixels))
        ]
        self.psfF_hole_ylims = [
            np.max((0, self.min_pos_psfF[1] - self.half_side_square_forfit_px)),
            np.min((self.min_pos_psfF[1] + self.half_side_square_forfit_px, self.NofPixels))
        ]
        self.psfB_hole_xlims = [
            np.max((0, self.min_pos_psfB[0] - self.half_side_square_forfit_px)),
            np.min((self.min_pos_psfB[0] + self.half_side_square_forfit_px, self.NofPixels))
        ]
        self.psfB_hole_ylims = [
            np.max((0, self.min_pos_psfB[1] - self.half_side_square_forfit_px)),
            np.min((self.min_pos_psfB[1] + self.half_side_square_forfit_px, self.NofPixels))
        ]
        print("psfF_hole_xlims in nm: ", (self.psfF_hole_xlims[0] * self.pxSize, self.psfF_hole_xlims[1] * self.pxSize))
        print("psfF_hole_ylims in nm: ", (self.psfF_hole_ylims[0] * self.pxSize, self.psfF_hole_ylims[1] * self.pxSize))
        print("psfB_hole_xlims in nm: ", (self.psfB_hole_xlims[0] * self.pxSize, self.psfB_hole_xlims[1] * self.pxSize))
        print("psfB_hole_ylims in nm: ", (self.psfB_hole_ylims[0] * self.pxSize, self.psfB_hole_ylims[1] * self.pxSize))
        # recalculate position of the minima after cutting
        self.min_pos_psfF_hole = (self.min_pos_psfF[0] - self.psfF_hole_xlims[0], self.min_pos_psfF[1] - self.psfF_hole_ylims[0])
        self.min_pos_psfB_hole = (self.min_pos_psfB[0] - self.psfB_hole_xlims[0], self.min_pos_psfB[1] - self.psfB_hole_ylims[0])
        print("numerical min fwd img hole in px: ", self.min_pos_psfF_hole)
        print("numerical min bck img hole in px: ", self.min_pos_psfB_hole)
        print("numerical min fwd img hole in nm: ", (self.min_pos_psfF_hole[0] * self.pxSize, self.min_pos_psfF_hole[1] * self.pxSize))
        print("numerical min bck img hole in nm: ", (self.min_pos_psfB_hole[0] * self.pxSize, self.min_pos_psfB_hole[1] * self.pxSize))
        # getting the cut images
        self.psfF_hole_matrix = self.imageF[self.psfF_hole_xlims[0]:self.psfF_hole_xlims[1], self.psfF_hole_ylims[0]:self.psfF_hole_ylims[1]]
        self.psfB_hole_matrix = self.imageB[self.psfB_hole_xlims[0]:self.psfB_hole_xlims[1], self.psfB_hole_ylims[0]:self.psfB_hole_ylims[1]]
        self.psfF_hole_matrix_smooth = gaussian_filter(self.psfF_hole_matrix, sigma=2)
        self.psfB_hole_matrix_smooth = gaussian_filter(self.psfB_hole_matrix, sigma=2)
        # Initial guesses for fit
        p0_F = parag_param_guess(self.psfF_hole_matrix_smooth, self.min_pos_psfF_hole)
        p0_B = parag_param_guess(self.psfB_hole_matrix_smooth, self.min_pos_psfB_hole)
        print("initial guesses fwd dona: ", p0_F)
        print("initial guesses bck dona: ", p0_B)
        # preparing grids for fits
        grid_red_F = preprare_grid_forfit(
            self.psfF_hole_xlims[1] - self.psfF_hole_xlims[0],
            self.psfF_hole_ylims[1] - self.psfF_hole_ylims[0]
        )
        grid_red_B = preprare_grid_forfit(
            self.psfB_hole_xlims[1] - self.psfB_hole_xlims[0],
            self.psfB_hole_ylims[1] - self.psfB_hole_ylims[0]
        )
        # Fitting and calculating the fitted PSFs
        self.fitted_coeff_psfF, cov = opt.curve_fit(parab_func, grid_red_F, self.psfF_hole_matrix.ravel(), p0_F,
                                       bounds=([-np.inf, -np.inf, -np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))
        self.fitted_coeff_psfB, cov = opt.curve_fit(parab_func, grid_red_B, self.psfB_hole_matrix.ravel(), p0_B,
                                       bounds=([-np.inf, -np.inf, -np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))
        # finding minima of the fit analytically
        # Finding analytical minimum of the fitted parabolas
        print("params fwd dona fit: ", self.fitted_coeff_psfF)
        print("params bck dona fit: ", self.fitted_coeff_psfB)
        self.min_coords_fit_psfF_um = parab_analytical_min(*self.fitted_coeff_psfF, self.psfF_hole_xlims[0], self.psfF_hole_ylims[0], self.pxSize)
        self.min_coords_fit_psfB_um = parab_analytical_min(*self.fitted_coeff_psfB, self.psfB_hole_xlims[0], self.psfB_hole_ylims[0], self.pxSize)
        print("fwd parab analyt min: ", self.min_coords_fit_psfF_um)
        print("bck parab analyt min: ", self.min_coords_fit_psfB_um)

        # now we take the aritmetic average
        self.target_coords_inroi_um = (
            (self.min_coords_fit_psfF_um[0] + self.min_coords_fit_psfB_um[0]) / 2,
            (self.min_coords_fit_psfF_um[1] + self.min_coords_fit_psfB_um[1]) / 2,
        )
        print("target coords in roi:", self.target_coords_inroi_um)
        self.target_coords_abs_um = (
            self.initialPos[0] + self.target_coords_inroi_um[0],
            self.initialPos[1] + self.target_coords_inroi_um[1],
            self.initialPos[2]
        )
        print("target coords abs", self.target_coords_abs_um)
        # moving to target point
        self.moveTo(*self.target_coords_abs_um)

    def psf_fit_FandB_and_move(self):
        target_F = self.psf_fit(self.imageF_copy, d='F')
        target_B = self.psf_fit(self.imageB_copy, d='B')

        print('[scan] target_F', target_F)
        print('[scan] target_B', target_B)

        # target_position = (0.5*target_F + 0.5*target_B) + np.array([4*self.pxSize, 0, 0])
        target_position = (0.5*target_F + 0.5*target_B)
        print('[scan] target_position', target_position)
        self.moveTo(*target_position)
        self.ROIcenterSignal.emit(target_position)
        self.realPositionSignal.emit(target_position)
        time.sleep(.2)
        self.trace_measurement()
        self.shuttermodeSignal.emit(11, False)
        
    def psf_fit(self, image, d, function='gaussian'):
        
        self.shift_param = 0
        
        # set main reference frame (relative to the confocal image)
        
        px_size_nm = self.pxSize * 1000 # in nm
        
        xmin, xmax, ymin, ymax = np.array(self.selectedCoord, dtype=np.int)
        
        if d == 'B':
            
            xmin = xmin - self.shift_param
            xmax = xmax - self.shift_param
        
        elif d == 'F':
            
            pass
        
        else:
            
            print('[scan] Invalid direction of scan selected')
        
        # select the data of the image corresponding to the ROI

#        array = self.image_copy[xmin:xmax, ymin:ymax]
        array = image[ymin:ymax, xmin:xmax]
        # normalize image to countrate in kHz
        array = array/self.pxTime * 1000
        
        print('shape of array', array.shape)

        
        if array.shape[0] > array.shape[1]:
            
            xmax  = xmax + 1
            array = image[ymin:ymax, xmin:xmax]
            
        elif array.shape[1] > array.shape[0]:
            
            ymax = ymax + 1
            array = image[ymin:ymax, xmin:xmax]
            
        else:
            
            pass
        
        shape = array.shape
        
        print('[scan] shape of array', array.shape)
        
        # create x and y arrays and meshgrid
        
        xmin_nm, xmax_nm, ymin_nm, ymax_nm = np.array([xmin, xmax, ymin, ymax]) * px_size_nm
        
        print('[scan] xmin_nm, xmax_nm, ymin_nm, ymax_nm', xmin_nm, xmax_nm, ymin_nm, ymax_nm)
        
        extent = [xmin_nm + self.initialPos[0] * 1000, self.initialPos[0] * 1000 + xmax_nm,
                  self.initialPos[1] * 1000 + ymax_nm, self.initialPos[1] * 1000 + ymin_nm]
        
        if d == 'F':
        
            plt.figure('raw data psf - Forward')
            plt.imshow(array, cmap=cmaps.parula, interpolation='None', extent=extent)
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
            
        elif d == 'B':
            
            plt.figure('raw data psf - Backwards')
            plt.imshow(array, cmap=cmaps.parula, interpolation='None', extent=extent)
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
             
#        x_nm = np.arange(xmin_nm + px_size_nm/2, xmax_nm + px_size_nm/2, px_size_nm) # TO DO: check +/- px_size_nm/2
#        y_nm = np.arange(ymin_nm + px_size_nm/2, ymax_nm + px_size_nm/2, px_size_nm)
        size = array.shape[0]
        x_nm = np.linspace(xmin_nm + px_size_nm/2, xmax_nm + px_size_nm/2, size)
        y_nm = np.linspace(ymin_nm + px_size_nm/2, ymax_nm + px_size_nm/2, size)

        # print('[scan] x_nm', x_nm)
        print('[scan] x_nm shape', x_nm.shape)
        # print('[scan] y_nm', y_nm)
        print('[scan] y_nm shape', y_nm.shape)

        (Mx_nm, My_nm) = np.meshgrid(x_nm, y_nm)

        print('[scan] shape grid', Mx_nm.shape)

        # make initial guess for parameters
        offset = np.min(array)
        d = 300  # nm
        x0 = (xmin_nm + xmax_nm)/2
        y0 = (ymin_nm + ymax_nm)/2
        A = np.max(array)*d**2  # check this estimation ????

        if function == 'doughnut':
            initial_guess = [A, x0, y0, d, offset]
            popt, pcov = opt.curve_fit(PSF.doughnut2D, (Mx_nm, My_nm), array.ravel(), p0=initial_guess)
            # retrieve results
            print('[scan] doughnut fit parameters', popt)
            dougnutFit = PSF.doughnut2D((Mx_nm, My_nm), *popt).reshape(shape)
            plt.figure('doughnut fit')
            plt.imshow(dougnutFit, cmap=cmaps.parula, interpolation='None', extent=extent)
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
            x0_fit = popt[1]
            y0_fit = popt[2]
            doughnut_center = np.array([x0_fit, y0_fit, 0], dtype=np.float64)/1000 # in µm
            target = self.initialPos + doughnut_center 
            print('[scan] target', target)

        if function == 'gaussian':
          
          σ_x = 130
          σ_y = 130
          initial_guess = [A, x0, y0, σ_x, σ_y, offset]
          
          popt, pcov = opt.curve_fit(PSF.gaussian2D, (Mx_nm, My_nm), array.ravel(), p0=initial_guess)
        
          print('[scan] gaussian fit parameters', popt)
          
          gaussianFit = PSF.gaussian2D((Mx_nm, My_nm), *popt).reshape(shape)
          
          if d == 'F':
          
              plt.figure('gaussian fit - Forward')
              plt.imshow(gaussianFit, cmap=cmaps.parula, interpolation='None', extent=extent)
              plt.xlabel('x (nm)')
              plt.ylabel('y (nm)')
              
          elif d == 'B':
          
              plt.figure('gaussian fit - Backwards')
              plt.imshow(gaussianFit, cmap=cmaps.parula, interpolation='None', extent=extent)
              plt.xlabel('x (nm)')
              plt.ylabel('y (nm)')
          
          x0_fit = popt[1]
          y0_fit = popt[2]
          print('SCAN GAUSSIAN CENTER', x0_fit, y0_fit)
          print('SCAN INITIAL', self.initialPos)

          gaussian_center = np.array([x0_fit, y0_fit, 0], dtype=np.float64)/1000 # in µm
          target = self.initialPos + gaussian_center 
        return target

    @pyqtSlot()
    def get_moveTo_initial_signal(self):
        self.moveTo(*self.initialPos)

    def relative_move(self, axis, direction):
        if axis == 'x' and direction == 'up':
            newPos_µm = self.initialPos[0] - self.xStep
            newPos_µm = round(newPos_µm, 3)
            self.initialPos = np.array([newPos_µm, self.initialPos[1],
                                        self.initialPos[2]])
        if axis == 'x' and direction == 'down':
            newPos_µm = self.initialPos[0] + self.xStep
            newPos_µm = np.around(newPos_µm, 3)
            self.initialPos = np.array([newPos_µm, self.initialPos[1],
                                        self.initialPos[2]])
        if axis == 'y' and direction == 'up':
            newPos_µm = self.initialPos[1] + self.yStep
            newPos_µm = np.around(newPos_µm, 3)   
            self.initialPos = np.array([self.initialPos[0], newPos_µm,
                                        self.initialPos[2]])
        if axis == 'y' and direction == 'down':
            newPos_µm = self.initialPos[1] - self.yStep
            newPos_µm = np.around(newPos_µm, 3)
            self.initialPos = np.array([self.initialPos[0], newPos_µm,
                                        self.initialPos[2]])
        if axis == 'z' and direction == 'up':
            newPos_µm = self.initialPos[2] + self.zStep
            newPos_µm = np.around(newPos_µm, 3)
            self.initialPos = np.array([self.initialPos[0], self.initialPos[1], 
                                        newPos_µm])
        if axis == 'z' and direction == 'down':
            newPos_µm = self.initialPos[2] - self.zStep
            newPos_µm = np.around(newPos_µm, 3)
            self.initialPos = np.array([self.initialPos[0], self.initialPos[1], 
                                        newPos_µm])
        self.update_device_param()
        self.emit_param()

    @pyqtSlot(float)
    def get_focuslockposition(self, position):
        self.focuslockpos = position
        
    @pyqtSlot(str)
    def saveConfigfile(self, fname):
        
        self.focuslockpositionSignal.emit(-9999)
        self.focuslockpos = -0.0
        time.sleep(0.5)
        now = time.strftime("%c")
        tools.saveConfig(self, now, 'test', filename=fname)
        print('[scan] saved configfile', fname)

        
    def save_current_frame(self):
      
        self.save_FB = False
        
        # experiment parameters

        # get z-position of focus lock
        # in standalone mode it justs saves -0.0 as focus lock position 
        # sleeps 0.5s to catch return signal from focus.py 
        #TODO: find better solution than sleeping
        self.focuslockpos = -0.0
        self.focuslockpositionSignal.emit(0)
        time.sleep(0.5)
        
        name = tools.getUniqueName(self.filename)
        now = time.strftime("%c")
        tools.saveConfig(self, now, name)

        # save image
        data = self.image_to_save
        result = Image.fromarray(data.astype('uint16'))
        result.save(r'{}.tiff'.format(name))

        if self.save_FB is True:
            print('[scan] Saved current frame F and B', name)

            # save image F
            data = self.imageF_copy
            result = Image.fromarray(data.astype('uint16'))
            result.save(r'{} F.tiff'.format(name))
            # save image B
            data = self.imageB_copy
            result = Image.fromarray(data.astype('uint16'))
            result.save(r'{} B.tiff'.format(name))
        print('[scan] Saved current frame', name)

#        self.gui.currentFrameButton.setChecked(False)

    @pyqtSlot(bool, str, np.ndarray)
    def get_scan_signal(self, lvbool, mode, initialPos):
        """
        Connection: [psf] scanSignal
        Description: get drift-corrected initial position, calculates the
        derived parameters (and updates ADwin data)
        """
        self.initialPos = initialPos
        self.calculate_derived_param()

        self.liveview(lvbool, mode)

    def line_acquisition(self):
        """Scan and get line from ADwin."""

        self.adw.Start_Process(1)

        line_time = (1/1000) * self.data_t[-1]  # target linetime in ms
        wait_time = line_time * 1.05 # TO DO: optimize this, it should work with 1.00, or maybe even less?
                                     # it should even work without the time.sleep()

        time.sleep(wait_time/1000)  # in s
        line_data = self.adw.GetData_Long(1, 0, self.tot_pixels)

        return line_data

    @pyqtSlot(bool, str)
    def liveview(self, lvbool, mode):
        if lvbool:
            self.acquisitionMode = mode  # modes: 'liveview', 'frame'
            self.liveview_start()
        else:
            self.liveview_stop()

    def reset_position(self):
        if self.scantype == 'xy':
            self.z_i = tools.convert(self.adw.Get_FPar(72), 'UtoX')
            self.moveTo(self.x_i, self.y_i, self.z_i)
        elif self.scantype == 'xz':
            self.y_i = tools.convert(self.adw.Get_FPar(71), 'UtoX')
            self.moveTo(self.x_i, self.y_i + self.scanRange/2,
                        self.z_i - self.scanRange/2)
        elif self.scantype == 'yz':
            self.x_i = tools.convert(self.adw.Get_FPar(70), 'UtoX')
            self.moveTo(self.x_i + self.scanRange/2, self.y_i,
                        self.z_i - self.scanRange/2)
        else:
            print("Unknown scan type in reset_position")    
        # HOTFIX: para ver si el linescan y el moveto no se estan peleando
        time.sleep(.256)

    def liveview_start(self):
        # self.plot_scan()
        self.reset_position()
        if self.acquisitionMode == "timegated EBP meas":
            # open all shutters
            self.control_shutters(1, True)
            time.sleep(0.05)
            self.control_shutters(2, True)
            time.sleep(0.05)
            self.control_shutters(3, True)
            time.sleep(0.05)
            self.control_shutters(4, True)
            # start tcspc measurement
            swabian.TCSPC_backend.start_measure(
                "EBP_timegated_"
            )
        self.viewtimer.start(self.viewtimer_time)

    def liveview_stop(self):
        """Finish liveview scan."""
        self.viewtimer.stop()
        if self.acquisitionMode == "timegated EBP meas":
            # close all shutters
            self.control_shutters(1, False)
            time.sleep(0.05)
            self.control_shutters(2, False)
            time.sleep(0.05)
            self.control_shutters(3, False)
            time.sleep(0.05)
            self.control_shutters(4, False)
            # stop tcspc measurement
            swabian.TCSPC_backend.stop_measure()
            self.ebp_measurement_done.emit()
            name = swabian.make_unique_name("EBP_timegated_", True)
            now = time.strftime("%c")
            tools.saveConfig(self, now, name)
        if self.acquisitionMode == "psf scan fit and move":
            self.psf_scanandfit_done.emit()
        if self.acquisitionMode != "psf scan fit and move":    
            self.reset_position()

    def update_view(self):
        """Procesa click del timer."""
        if self.i < self.NofPixels:
            if self.scantype == 'xy':
                dy = tools.convert(self.dy, 'ΔXtoU')
                self.y_offset = int(self.y_offset + dy)
                self.adw.Set_FPar(2, self.y_offset)
            if self.scantype == 'xz' or self.scantype == 'yz':
                dz = tools.convert(self.dz, 'ΔXtoU')
                self.z_offset = int(self.z_offset + dz)
                self.adw.Set_FPar(2, self.z_offset)
            self.lineData = self.line_acquisition()

            if self.full_scan is True:
                self.image[:, self.i] = self.lineData
            elif self.FBaverage_scan is True:
                # display average of forward and backward image
                c0 = self.NofAuxPixels
                c1 = self.NofPixels

                lineData_F = self.lineData[c0:c0+c1]
                lineData_B = self.lineData[3*c0+c1:3*c0+2*c1]

                if self.i % 2 == 0:
                    self.image[:, self.i] = lineData_F
                if self.i % 2 != 0:
                    self.image[:, self.i] = lineData_B[::-1]
            else:
                # displays only forward image
                c0 = self.NofAuxPixels
                c1 = self.NofPixels

                lineData_F = self.lineData[c0:c0+c1]
                lineData_B = self.lineData[3*c0+c1:3*c0+2*c1]
                self.imageF[:, self.i] = lineData_F
                self.imageB[:, self.i] = lineData_B[::-1]
                self.image[:, self.i] = lineData_F
            # display image after every scanned line
            self.image_to_save = self.image
            self.imageF_copy = self.imageF     # TO DO: clean up with fit signal to avoid the copy image
            self.imageB_copy = self.imageB
            # self.image_copy = self.image

            # Podríamos siempre hacer que siempre tomemos la imagen completa y mandar
            # al frontend sólo la parte que importa. De mientras...
            self.imageSignal.emit(self.image)
            # print(datetime.now(), '[scan] Image emitted to frontend')

            self.i = self.i + 1
        else:
            print(datetime.now(), '[scan] Frame ended')
            self.i = 0
            self.y_offset = 0
            self.z_offset = 0
            self.reset_position()
            if self.acquisitionMode == 'frame':
                self.liveview_stop()
                self.frameIsDone.emit(True, self.image, self.NofAuxPixels, self.NofPixels)
            if self.acquisitionMode ==  'timegated EBP meas':
                self.liveview_stop()   
            if self.acquisitionMode ==  'psf scan fit and move':
                self.liveview_stop()
                if self.scantype == 'xy':
                    self.psf_parab_fit_andmove()
            self.update_device_param()

    @pyqtSlot(int, bool)
    def control_shutters(self, num, val):

        if val is True:
                        
            tools.toggle_shutter(self.adw, num, True)
            print(datetime.now(), '[scan] Shutter ' + str(num) + ' opened')
            
        if val is False:
                        
            tools.toggle_shutter(self.adw, num, False)
            print(datetime.now(), '[scan] Shutter ' + str(num) + ' closed')
                       
    @pyqtSlot(int, bool)
    def shutter_handler(self, num, on):
        self.shuttermodeSignal.emit(num, on)
        
    @pyqtSlot(bool)
    def toggle_flipper(self, val):
        
        if val is True:
            self.flipper_state = True
            self.adw.Set_Par(55, 1)
            self.adw.Start_Process(5)
            print('[scan] Flipper down')
        if val is False:
            self.flipper_state = False
            self.adw.Set_Par(55, 1)
            self.adw.Start_Process(5)
            print('[scan] Flipper up')

    @pyqtSlot(bool)        
    def toggle_FBav_scan(self, val):
        if val is True:
            self.FBaverage_scan = True
        if val is False:
            self.FBaverage_scan = False

    def emit_ROI_center(self):
        
        self.ROIcenterSignal.emit(self.ROIcenter)
        
        print('[scan] ROI center emitted')
        
    def trace_measurement(self):
      
        n = 100
        pxtime = 1000
        trace_data = self.trace_acquisition(Npoints=n, pixeltime=pxtime)
        
        time = np.arange(n)
        
        plt.style.use('dark_background')
        plt.figure()
        plt.plot(time, trace_data)
        plt.xlabel('Time (ms)')
        plt.ylabel('Count rate (kHz)')
        
    def trace_acquisition(self, Npoints, pixeltime):
      
        """ 
        Method to acquire a trace of photon counts at the current position.
        
        Npoints = number of points to be acquired (max = 1024)
        pixeltime = time per point (in μs)
        
        
        """
        
        # pixeltime in μs
      
        self.adw.Set_FPar(65, tools.timeToADwin(pixeltime))
        self.adw.Set_Par(60, Npoints+1)

        self.adw.Start_Process(6)
        
        trace_time = Npoints * (pixeltime/1000)  # target linetime in ms
        wait_time = trace_time * 1.05 # TO DO: optimize this, it should work with 1.00, or maybe even less?
                                     # it should even work without the time.sleep()
        
        time.sleep(wait_time/1000) # in s

        trace_data = self.adw.GetData_Long(6, 0, Npoints+1)
        
        trace_data = trace_data[1:]# TO DO: fix the high count error on first element

        return trace_data

    
    def enableDiodelaser(self, enable):
        
        diodelasID = self.diodelas.idn()
        
        if enable:
            self.laserstate = True
            self.diodelas.enabled = True
            print(datetime.now(), '[scan] Diodelaser started')
            print(datetime.now(), '[scan] Diodelaser-ID:', diodelasID)
            
            time.sleep(4) #according to manual, lasing will start 5s after turning-on
            ans1, ans2 = self.diodelas.status()
            i = 0
            while (ans2 != 'Laser system is active, radiation can be emitted'):
                time.sleep(0.5) #check every 0.5s whether emission started
                ans1, ans2 = self.diodelas.status()
                if i>=12:
                    break #interrupt loop after 10s of waiting for emission, preventing to find no end
                i += 1
                
            if i<12:
                self.diodelaserEmissionSignal.emit(True)
                print(datetime.now(), '[scan] Diodelaser emitting!')
            else:
                print(datetime.now(), '[scan] Diodelaser not able to emit radiation. Check status!')
            
        else:
            self.laserstate = False
            self.setpowerDiodelaser(0)
            self.diodelas.enabled = False
            self.diodelaserEmissionSignal.emit(False)
            print(datetime.now(), '[scan] Diodelaser disabled')
    
    def setpowerDiodelaser(self, value):
        if self.diodelas.enabled:
            self.diodelas.power = value
            print(datetime.now(), '[scan] Power of diodelaser set to', str(value), 'mW')
        
    def plot_scan(self):
        
        # save scan plot (x vs t)
        plt.figure()
        plt.title('Scan plot x vs t')
        plt.plot(self.data_t_adwin[0:-1], self.data_x_adwin, 'go')
        plt.xlabel('t (ADwin time)')
        plt.ylabel('V (DAC units)')
        
        c0 = self.NofAuxPixels
        c1 = self.NofPixels
        
        plt.plot(self.data_t_adwin[c0], self.data_x_adwin[c0], 'r*')
        plt.plot(self.data_t_adwin[c0+c1-1], self.data_x_adwin[c0+c1-1], 'r*')
                
        plt.plot(self.data_t_adwin[3*c0+c1], self.data_x_adwin[3*c0+c1], 'r*')
        plt.plot(self.data_t_adwin[3*c0+2*c1-1], self.data_x_adwin[3*c0+2*c1-1], 'r*')

        fname = tools.getUniqueName(self.filename)
        fname = fname + '_scan_plot'
        plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)

    def make_connection(self, frontend):
        frontend.liveviewSignal.connect(self.liveview)
        frontend.moveToROIcenterButton.clicked.connect(self.moveTo_roi_center)
        frontend.currentFrameButton.clicked.connect(self.save_current_frame)
        frontend.moveToButton.clicked.connect(self.moveTo_action)
        frontend.paramSignal.connect(self.get_frontend_param)
        frontend.closeSignal.connect(self.stop)
        frontend.traceButton.clicked.connect(self.trace_measurement)
        frontend.fitPSFSignal.connect(self.get_ROI_coords_and_fit)
        self.auxFitSignal.connect(self.psf_fit_FandB_and_move)
        self.auxMoveSignal.connect(self.moveTo_roi_center)
        frontend.shutter1Checkbox.stateChanged.connect(lambda: self.control_shutters(1, frontend.shutter1Checkbox.isChecked()))
        frontend.shutter2Checkbox.stateChanged.connect(lambda: self.control_shutters(2, frontend.shutter2Checkbox.isChecked()))
        frontend.shutter3Checkbox.stateChanged.connect(lambda: self.control_shutters(3, frontend.shutter3Checkbox.isChecked()))
        frontend.shutter4Checkbox.stateChanged.connect(lambda: self.control_shutters(4, frontend.shutter4Checkbox.isChecked()))
        frontend.diodeShutter.stateChanged.connect(lambda: self.control_shutters(11, frontend.diodeShutter.isChecked()))

        frontend.flipperButton.clicked.connect(lambda: self.toggle_flipper(frontend.flipperButton.isChecked()))
        frontend.FBavScanButton.clicked.connect(lambda: self.toggle_FBav_scan(frontend.FBavScanButton.isChecked()))
        frontend.diodelaserButton.clicked.connect(lambda: self.enableDiodelaser(frontend.diodelaserButton.isChecked()))
        frontend.diodepowerSpinBox.valueChanged.connect(lambda: self.setpowerDiodelaser(frontend.diodepowerSpinBox.value()))

        
        frontend.xUpButton.pressed.connect(lambda: self.relative_move('x', 'up'))
        frontend.xDownButton.pressed.connect(lambda: self.relative_move('x', 'down'))
        frontend.yUpButton.pressed.connect(lambda: self.relative_move('y', 'up'))
        frontend.yDownButton.pressed.connect(lambda: self.relative_move('y', 'down'))        
        frontend.zUpButton.pressed.connect(lambda: self.relative_move('z', 'up'))
        frontend.zDownButton.pressed.connect(lambda: self.relative_move('z', 'down'))
          
    def stop(self):
        
        self.shuttermodeSignal.emit(8, False)
        if self.flipper_state is True:
            self.toggle_flipper(False)
        self.liveview_stop()
        
        if self.laserstate:
            self.enableDiodelaser(False)
        self.diodelas.closeLaserPort()
        print(datetime.now(), '[scan] Serial port of diode laser closed')
        
        # Go back to 0 position

        x_0 = 0
        y_0 = 0
        z_0 = 0

        self.moveTo(x_0, y_0, z_0)
        

if __name__ == '__main__':

    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()
        
    #app.setStyle(QtGui.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        
    #initialize devices (diode laser and AdWin board)
    
    port = tools.get_MiniLasEvoPort()
    print('[scan] MiniLasEvo diode laser port:', port)
    diodelaser = MiniLasEvo(port)
    
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    setupDevice(adw)
    
    worker = Backend(adw, diodelaser)    
    gui = Frontend()
    
    worker.make_connection(gui)
    gui.make_connection(worker)
    
    gui.emit_param()
    worker.emit_param()
    
    scanThread = QtCore.QThread()
    worker.moveToThread(scanThread)
    worker.viewtimer.moveToThread(scanThread)
    worker.viewtimer.timeout.connect(worker.update_view)
    
    scanThread.start()

    gui.setWindowTitle('scan')
    gui.show()

    app.exec_()
