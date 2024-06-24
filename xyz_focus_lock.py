# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:51:13 2023.

@author: Florencia D. Choque based on xyz_tracking for RASTMIN by
Luciano Masullo
Modified to work with new stabilization in p-MINFLUX, IDS_U3 cam and ADwin by
Florencia D. Choque
"""
import os
import numpy as np
import time
import scipy.ndimage as ndi
from datetime import date, datetime

from scipy import optimize as opt
from tools import customLog  # NOQA Para inicializar el logging
import tools.viewbox_tools as viewbox_tools
import tools.PSF as PSF
import tools.tools as tools

import scan

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QCheckBox

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import qdarkstyle

import drivers.ADwin as ADwin
# Is it necessary to modify expousure time and gain in driver ids_cam? FC
import drivers.ids_cam as ids_cam

import logging as _lgn

_lgr = _lgn.getLogger(__name__)

VIDEO = False


PX_SIZE = 23.5  # px size of camera in nm #antes 80.0 para Andor #33.5
PX_Z = 16  # 20 nm/px for z in nm


# Posiblemente debería ir a un toolbox
class GroupedCheckBoxes:
    """Manages grouped CheckBoxes states."""

    def __init__(self, all_checkbox: QCheckBox, *other_checkboxes):
        """Init class.

        Parameters
        ----------
        all_checkbox : QCheckBox
            The checkbox that checks/unchecks all others.
        *other_checkboxes : TYPE
            All other checkboxes.
        """
        self.acb = all_checkbox
        self.others = other_checkboxes
        for _ in other_checkboxes:
            _.stateChanged.connect(self.on_state)
        all_checkbox.clicked.connect(self.on_click)

    def on_state(self, state: int):
        """Handle single items change."""
        self.acb.setChecked(all([_.isChecked() for _ in self.others]))

    def on_click(self, is_checked: bool):
        """Handle 'All' checkbox click."""
        for _ in self.others:
            _.setChecked(is_checked)


class Frontend(QtGui.QFrame):
    # Chequear las señales aquí, hace falta changedROI (creo que es analogo a
    # z_roiInfoSignal, cf), paramSignal???
    # antes era (int, np.ndarray) cf xy_tracking Ver cómo afecta esto al
    # procesamiento, porque parece estar bien al ser como en xyz_tracking
    roiInfoSignal = pyqtSignal(str, int, list)
    closeSignal = pyqtSignal()
    saveDataSignal = pyqtSignal(bool)
    """
    Signals
    - roiInfoSignal: emmited when a ROI changes
         To: [backend] get_roi_info
         Parameters: str ('xy', 'z'), int (nro de rois de ese tipo),
                     list de np.ndarrays([xmin, xmax, ymin, ymax])
    - closeSignal:
         To: [backend] stop
    - saveDataSignal:
         To: [backend] get_save_data_state
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initial ROI parameters
        self.ROInumber = 0  # siguiente ROIs xy a actualizar
        # Una lista en la que se guardarán los objetos ROI a graficar
        self.roilist: list = []  # list[viewbox_tools.ROI2]
        self.roi_z: viewbox_tools.ROI2 = None
        # lista de graficos de desplazamientos de cada fiduciaria
        self.xCurve: list = []  # list[pg.PlotDataItem]  una por ROI
        self.yCurve: list = []  # list[pg.PlotDataItem]  una por ROI

        self.setup_gui()

    def create_roi(self, roi_type: str):
        """Add a new roi ('xy' or 'z')."""
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
        elif roi_type == 'z':
            ROIpen = pg.mkPen(color='y')
            ROIpos = (512 - 64, 512 - 64)
            self.roi_z = viewbox_tools.ROI2(140, self.vb, ROIpos,
                                            handlePos=(1, 0),
                                            handleCenter=(0, 1),
                                            scaleSnap=True,
                                            translateSnap=True,
                                            pen=ROIpen, number='z')
            self.zROIButton.setEnabled(False)
        else:
            _lgr.error("Unknown ROI type asked: %s", roi_type)

    def emit_roi_info(self, roi_type):
        """Informar los valores de los ROI del tipo solicitado existentes."""
        if roi_type == 'xy':
            roinumber = len(self.roilist)
            if roinumber == 0:
                print(datetime.now(), '[xy_tracking] Please select a valid ROI'
                      ' for fiducial NPs tracking')
                return
            coordinates_list = []
            for i in range(len(self.roilist)):
                xmin, ymin = self.roilist[i].pos()
                xmax, ymax = self.roilist[i].pos() + self.roilist[i].size()
                coordinates = np.array([xmin, xmax, ymin, ymax])
                coordinates_list.append(coordinates)
            self.roiInfoSignal.emit('xy', roinumber, coordinates_list)
        elif roi_type == 'z':
            if self.roi_z is None:
                print(datetime.now(), 'Please select a valid ROI for z tracking')
                return
            xmin, ymin = self.roi_z.pos()
            xmax, ymax = self.roi_z.pos() + self.roi_z.size()
            coordinates = np.array([xmin, xmax, ymin, ymax])
            coordinates_list = [coordinates]

            self.roiInfoSignal.emit('z', 1, coordinates_list)
        else:
            _lgr.error("Unknown ROI type asked: %s", roi_type)

    def delete_roi(self):
        """Elimina la última ROI xy."""
        if not self.roilist:
            _lgr.info("No hay ROI que borrar")
            return
        roi = self.roilist.pop()
        self.vb.removeItem(roi)
        roi.hide()
        del roi
        self.ROInumber -= 1
        self.emit_roi_info('xy')

    @pyqtSlot(bool)
    def toggle_liveview(self, on):
        """Cambia el estado del botón al elegirlo.

        No realiza ninguna acción real: el mismo evento está conectado al
        backend.
        """
        if on:
            self.liveviewButton.setChecked(True)
            _lgr.info('Live view started')
        else:
            self.liveviewButton.setChecked(False)
            # self.emit_roi_info('xy')
            # self.emit_roi_info('z')
            # TODO: poner marca al agua para avisar que está OFF
            self.img.setImage(np.zeros((1200, 1920)), autoLevels=False)
            _lgr.info('Live view stopped')

    @pyqtSlot(np.ndarray)
    def get_image(self, img):
        """Recibe la imagen del back."""
        self.img.setImage(img, autoLevels=False)
        self.xaxis.setScale(scale=PX_SIZE/1000)  # scale to µm
        self.yaxis.setScale(scale=PX_SIZE/1000)  # scale to µm

    # Cambió la señal changed_data
    # cambiaron los parámetros del slot get_data
    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def get_data(self, tData, xData, yData, zData, avgIntData):
        """Recibir datos nuevos del backend."""
        N_NP = np.shape(xData)[1]
        if N_NP != len(self.roilist):
            _lgr.error("El número de ROIs y la info de xData no coinciden")

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
        # los cambios aquí tienen que verse reflejados en la gui, histogramas
        if len(xData) > 2:  # TODO: chequear esta parte
            # self.plot_ellipse(xData, yData)
            hist, bin_edges = np.histogram(zData, bins=60)
            self.zHist.setOpts(x=bin_edges[:-1], height=hist)
            xstd = np.std(np.mean(xData, axis=1))
            self.xstd_value.setText(str(np.around(xstd, 2)))
            ystd = np.std(np.mean(yData, axis=1))
            self.ystd_value.setText(str(np.around(ystd, 2)))
            zstd = np.std(zData)
            self.zstd_value.setText(str(np.around(zstd, 2)))

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def get_xy_data(self, tData, xData, yData):
        """Recibir datos nuevos xy del backend."""
        N_NP = np.shape(xData)[1]
        if N_NP != len(self.roilist):
            _lgr.error("El número de ROIs y la info de xData no coinciden")

        # x data
        for i in range(N_NP):
            self.xCurve[i].setData(tData, xData[:, i])
        self.xmeanCurve.setData(tData, np.mean(xData, axis=1))
        # y data
        for i in range(N_NP):
            self.yCurve[i].setData(tData, yData[:, i])
        self.ymeanCurve.setData(tData, np.mean(yData, axis=1))
        # set xy 2D data
        self.xyDataItem.setData(np.mean(xData, axis=1), np.mean(yData, axis=1))
        # los cambios aquí tienen que verse reflejados en la gui, histogramas
        if len(xData) > 2:  # TODO: chequear esta parte
            # self.plot_ellipse(xData, yData)
            xstd = np.std(np.mean(xData, axis=1))
            self.xstd_value.setText(str(np.around(xstd, 2)))
            ystd = np.std(np.mean(yData, axis=1))
            self.ystd_value.setText(str(np.around(ystd, 2)))

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def get_z_data(self, tData, zData, avgIntData):
        """Recibir datos nuevos z, intensidad del backend."""
        self.zCurve.setData(tData, zData)
        self.avgIntCurve.setData(avgIntData)

        hist, bin_edges = np.histogram(zData, bins=60)
        self.zHist.setOpts(x=bin_edges[:-1], height=hist)
        zstd = np.std(zData)
        self.zstd_value.setText(str(np.around(zstd, 2)))


    def plot_ellipse(self, x_array, y_array):
        pass
#            cov = np.cov(x_array, y_array)
#            a, b, theta = tools.cov_ellipse(cov, q=.683)
#            theta = theta + np.pi/2
#            print(a, b, theta)
#            xmean = np.mean(xData)
#            ymean = np.mean(yData)
#            t = np.linspace(0, 2 * np.pi, 1000)
#            c, s = np.cos(theta), np.sin(theta)
#            R = np.array(((c, -s), (s, c)))
#            coord = np.array([a * np.cos(t), b * np.sin(t)])
#            coord_rot = np.dot(R, coord)
#            x = coord_rot[0] + xmean
#            y = coord_rot[1] + ymean

#             TO DO: fix plot of ellipse
#            self.xyDataEllipse.setData(x, y)
#            self.xyDataMean.setData([xmean], [ymean])

    @pyqtSlot(int, bool)
    def update_shutter(self, num, on):
        """Toma acciones con los shutters que lo involucran. DESACTIVADO.

        Previamente el shutter 5 bloqueaba la señal z y el 6 la señal xy. Al
        mudar el setup sacamos estos shutters, así que esta función no hace
        nada (hoy por hoy)

        setting of num-value:
            0 - signal send by scan-gui-button --> change state of all minflux
                shutters
            1...6 - shutter 1-6 will be set according to on-variable, i.e.
                either true or false; only 1-4 controlled from here
            7 - set all minflux shutters according to on-variable
            8 - set all shutters according to on-variable
        for handling of shutters 1-5 see [scan] and [focus]
        """
        if (num == 5) or (num == 6) or (num == 8):
            self.shutterCheckbox.setChecked(on)

    # TODO: Chequear si necesito esto, cf xyz
    # FIXME reimplementar esto, evitando loops
    @pyqtSlot(bool, bool, bool)
    def get_backend_states(self, tracking, feedback, savedata):
        """Actualizar el frontend de acuerdo al estado del backend."""
        self.trackAllBox.setChecked(tracking)
        self.feedbackLoopBox.setChecked(feedback)
        self.saveDataBox.setChecked(savedata)

    def emit_save_data_state(self):
        """Informa al backend si hay que grabar data xy o no."""
        if self.saveDataBox.isChecked():
            self.saveDataSignal.emit(True)
        else:
            self.saveDataSignal.emit(False)

    def make_connection(self, backend):
        backend.changedImage.connect(self.get_image)
        backend.changedXYData.connect(self.get_xy_data)
        backend.changedZData.connect(self.get_z_data)
        backend.updateGUIcheckboxSignal.connect(self.get_backend_states)
        backend.shuttermodeSignal.connect(self.update_shutter)
        backend.liveviewSignal.connect(self.toggle_liveview)

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
        # lut = viewbox_tools.generatePgColormap(cmaps.parula)
        # self.hist.gradient.setColorMap(lut) #
        # self.hist.vb.setLimits(yMin=800, yMax=3000)
        # TO DO: fix histogram range

        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)

        # xyz drift graph (graph without a fixed range)
        self.xyzGraph = pg.GraphicsWindow()

#        self.xyzGraph.resize(200, 300)
        self.xyzGraph.setAntialiasing(True)

        self.xyzGraph.statistics = pg.LabelItem(justify='right')
        self.xyzGraph.addItem(self.xyzGraph.statistics)
        self.xyzGraph.statistics.setText('---')

        # no sé si esta linea va con row=0, parece que yo la modifiqué en
        # xyz_tracking_flor y le puse 1
        self.xyzGraph.xPlot = self.xyzGraph.addPlot(row=0, col=0)
        # TODO: clean-up the x-y mess (they're interchanged)
        self.xyzGraph.xPlot.setLabels(bottom=('Time', 's'),
                                      left=('X position', 'nm'))
        self.xyzGraph.xPlot.showGrid(x=True, y=True)
        self.xmeanCurve = self.xyzGraph.xPlot.plot(pen='w', width=40)

        self.xyzGraph.yPlot = self.xyzGraph.addPlot(row=1, col=0)
        self.xyzGraph.yPlot.setLabels(bottom=('Time', 's'),
                                      left=('Y position', 'nm'))
        self.xyzGraph.yPlot.showGrid(x=True, y=True)
        self.ymeanCurve = self.xyzGraph.yPlot.plot(pen='r', width=40)

        # añado plots de zCurve y avgIntCurve
        self.xyzGraph.zPlot = self.xyzGraph.addPlot(row=2, col=0)
        self.xyzGraph.zPlot.setLabels(bottom=('Time', 's'),
                                      left=('Z position', 'nm'))
        self.xyzGraph.zPlot.showGrid(x=True, y=True)
        self.zCurve = self.xyzGraph.zPlot.plot(pen='y', connect='finite')

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

        # self.xyPoint.xyPointPlot = self.xyzGraph.addPlot(col=1)
        # self.xyPoint.xyPointPlot.showGrid(x=True, y=True)

        self.xyplotItem = self.xyPoint.addPlot()
        self.xyplotItem.showGrid(x=True, y=True)
        self.xyplotItem.setLabels(bottom=('X position', 'nm'),
                                  left=('Y position', 'nm'))  # Cambio ejes FC
        self.xyplotItem.setAspectLocked(True)  # Agregué FC

        # FIXME: ver si esto esta funcionando
        self.xyDataItem = self.xyplotItem.plot([], pen=None,
                                               symbolBrush=(255, 0, 0),
                                               symbolSize=5, symbolPen=None)

        # Algo que nunca se implemento (ver plot_ellipse)
        # self.xyDataMean = self.xyplotItem.plot([], pen=None,
        #                                        symbolBrush=(117, 184, 200),
        #                                        symbolSize=5, symbolPen=None)
        # self.xyDataEllipse = self.xyplotItem.plot(pen=(117, 184, 200))

        # z drift graph (1D histogram)
        x = np.arange(-30, 30)
        y = np.zeros(len(x))

        self.zHist = pg.BarGraphItem(x=x, height=y, width=0.6, brush='#3BC14A')
        self.zWin = self.xyPoint.addPlot()
        self.zWin.addItem(self.zHist)

        # LiveView Button
        self.liveviewButton = QtGui.QPushButton('Camera LIVEVIEW')
        self.liveviewButton.setEnabled(True)
        self.liveviewButton.setCheckable(True)

        # create xy ROI button
        self.xyROIButton = QtGui.QPushButton('xy ROI')
        self.xyROIButton.setEnabled(True)
        self.xyROIButton.clicked.connect(
            lambda: self.create_roi(roi_type='xy'))

        # create z ROI button
        self.zROIButton = QtGui.QPushButton('z ROI')
        self.zROIButton.setEnabled(True)
        self.zROIButton.clicked.connect(lambda: self.create_roi(roi_type='z'))

        # select xy ROI
        self.selectxyROIbutton = QtGui.QPushButton('Select xy ROI')
        self.selectxyROIbutton.clicked.connect(
            lambda: self.emit_roi_info(roi_type='xy'))

        # select z ROI
        self.selectzROIbutton = QtGui.QPushButton('Select z ROI')
        self.selectzROIbutton.clicked.connect(
            lambda: self.emit_roi_info(roi_type='z'))

        # delete ROI button
        self.delete_roiButton = QtGui.QPushButton('delete ROIs')
        self.delete_roiButton.clicked.connect(self.delete_roi)

        # Aquí se debería agregar delete_roi_zButton
        # self.delete_roi_zButton = QtGui.QPushButton('delete ROI z')
        # self.delete_roi_zButton.clicked.connect(self.delete_roi_z)

        # export data checkbox
        self.exportDataButton = QtGui.QPushButton('Export current data')

        # position tracking checkbox
        # self.trackAllBox.stateChanged.connect(
        #     lambda: self.emit_roi_info(roi_type='xy'))
        trackgb = QGroupBox("Tracking")
        trackLayout = QHBoxLayout()
        trackgb.setLayout(trackLayout)
        self.trackAllBox = QtGui.QCheckBox('All')
        self.trackXYBox = QtGui.QCheckBox("xy")
        self.trackZBox = QtGui.QCheckBox("z")
        trackLayout.addWidget(self.trackAllBox)
        trackLayout.addWidget(self.trackXYBox)
        trackLayout.addWidget(self.trackZBox)

        self.trackManager = GroupedCheckBoxes(self.trackAllBox, self.trackXYBox,
                                              self.trackZBox,
                                              )
        self.trackXYBox.stateChanged.connect(
            lambda: self.emit_roi_info(roi_type='xy'))
        self.trackZBox.stateChanged.connect(
            lambda: self.emit_roi_info(roi_type='z'))
        self.trackAllBox.stateChanged.connect(self.setup_xy_data_curves)

        # TODO Ver como manejar el setup data curves
        # TODO: ver cómo manejar el emit_roi
        # TODO: ver cómo prender las cosas de a una

        # turn ON/OFF feedback loop
        # self.feedbackLoopBox = QtGui.QCheckBox('Feedback loop')
        feedbackgb = QGroupBox("Feedback")
        feedbackLayout = QHBoxLayout()
        feedbackgb.setLayout(feedbackLayout)
        self.feedbackAllBox = QtGui.QCheckBox('All')
        self.feedbackXYBox = QtGui.QCheckBox("xy")
        self.feedbackZBox = QtGui.QCheckBox("z")
        feedbackLayout.addWidget(self.feedbackAllBox)
        feedbackLayout.addWidget(self.feedbackXYBox)
        feedbackLayout.addWidget(self.feedbackZBox)
        self.feedbackManager = GroupedCheckBoxes(self.feedbackAllBox,
                                                  self.feedbackXYBox,
                                                  self.feedbackZBox,
                                                  )

        # self.feedbackXYBox.stateChanged.connect(
            # lambda: self.emit_roi_info(roi_type='xy'))
        # self.feedbackZBox.stateChanged.connect(
            # lambda: self.emit_roi_info(roi_type='z'))


        # save data signal
        self.saveDataBox = QtGui.QCheckBox("Save data")
        self.saveDataBox.stateChanged.connect(self.emit_save_data_state)

        # button to clear the data
        self.clearDataButton = QtGui.QPushButton('Clear data')

        # shutter button and label
        # puedo usar esta idea para el confocal
        ##################################
        self.shutterLabel = QtGui.QLabel('Shutter open?')
        self.shutterCheckbox = QtGui.QCheckBox('473 nm laser')

        # Button to make custom pattern
        # es start pattern en linea 500 en xyz_tracking
        self.xyPatternButton = QtGui.QPushButton('Move')

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
        # también añadir algo así
        # subgrid.addWidget(self.delete_roi_zButton, 6, 0)
        subgrid.addWidget(self.exportDataButton, 6, 0)
        subgrid.addWidget(self.clearDataButton, 7, 0)
        subgrid.addWidget(self.xyPatternButton, 8, 0)
        # subgrid.addWidget(self.trackAllBox, 1, 1)
        subgrid.addWidget(trackgb, 1, 1)
        # subgrid.addWidget(self.feedbackLoopBox, 2, 1)
        subgrid.addWidget(feedbackgb, 2, 1)
        subgrid.addWidget(self.saveDataBox, 3, 1)
        subgrid.addWidget(self.shutterLabel, 9, 0)
        subgrid.addWidget(self.shutterCheckbox, 9, 1)

        ####################################################
        # Agrego FC grilla para estadística
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
        grid.addWidget(self.xyPoint, 1, 1, 1, 2)  # agrego 1,2 al final

        self.liveviewButton.clicked.connect(
            lambda: self.toggle_liveview(self.liveviewButton.isChecked()))

    def setup_xy_data_curves(self):
        """Crear o borrar las curvas si hace falta.

        TODO: arreglar esta forma fea de hacer las cosas
        """
        if self.trackAllBox.isChecked():
            if self.xCurve:
                for i in range(len(self.roilist)):  # remove previous curves
                    self.xyzGraph.xPlot.removeItem(self.xCurve[i])
                    self.xyzGraph.yPlot.removeItem(self.yCurve[i])
                    # TODO: Promedio, z e AvgInt???
            self.xCurve = [0] * len(self.roilist)
            for i in range(len(self.roilist)):
                self.xCurve[i] = self.xyzGraph.xPlot.plot(pen='w', alpha=0.3)
                self.xCurve[i].setAlpha(0.3, auto=False)
            self.yCurve = [0] * len(self.roilist)
            for i in range(len(self.roilist)):
                self.yCurve[i] = self.xyzGraph.yPlot.plot(pen='r', alpha=0.3)
                self.yCurve[i].setAlpha(0.3, auto=False)

    def closeEvent(self, *args, **kwargs):
        _lgr.info('Close in frontend')
        self.closeSignal.emit()
        super().closeEvent(*args, **kwargs)
        # TODO: ESTO ES MUY FEO
        # TODO: Ver thread del backend
        app.quit()


class Backend(QtCore.QObject):
    changedImage = pyqtSignal(np.ndarray)
    changedXYData = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, )
    changedZData = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, )
    # no se usa en xyz_tracking
    updateGUIcheckboxSignal = pyqtSignal(bool, bool, bool)
    # changedSetPoint = pyqtSignal(float) #Debería añadir esta señal??? de focus.py
    xyIsDone = pyqtSignal(bool, float, float)  # signal to emit new piezo position after drift correction
    shuttermodeSignal = pyqtSignal(int, bool)
    liveviewSignal = pyqtSignal(bool)
    zIsDone = pyqtSignal(bool, float)  # se emite para psf.py script
    focuslockpositionSignal = pyqtSignal(float)  # se emite para scan.py
    """
    Signals

    - changedImage:
        To: [frontend] get_image
    - changedXYData, changedZData:
        To: [frontend] get_xy_data, get_z_data
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
    roi_coordinates_list: list = []  # list[np.ndarray] lista de tetradas de ROIs xy
    zROIcoordinates: np.ndarray = np.ndarray([0, 0, 0, 0], dtype=int)

    def __init__(self, camera, adw, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera = camera  # no need to setup or initialize camera
        if VIDEO:
            self.video = []
        self.adw = adw
        # folder
        # TODO: change to get folder from microscope
        today = str(date.today()).replace('-', '')
        root = 'C:\\Data\\'
        self.folder = root + today
        _lgr.debug("Name of folder: %s", self.folder)
        try:
            os.mkdir(self.folder)
        except FileExistsError:
            _lgr.info("The directory already existed: %s", self.folder)
        except Exception:
            _lgr.error("Creation of the directory %s failed", self.folder)
        else:
            _lgr.info("Successfully created the directory: %s", self.folder)
        xy_filename = 'xy_data'
        self.xy_filename = os.path.join(self.folder, xy_filename)
        z_filename = 'z_data'
        self.z_filename = os.path.join(self.folder, z_filename)
        # Se llama viewTimer pero es el unico para todo, no sólo para view
        # Ojo: aquí coloqué viewtimer porque es el que se usa a lo largo del
        # código, pero en xyz_tracking se usa view_timer
        self.viewtimer = QtCore.QTimer()
        # TODO: overload de movetothread para que se mueva con sus timers
        # self.viewtimer.timeout.connect(self.update)
        self.xyz_time = 200  # 200 ms per acquisition + fit + correction

        # Si trackear (NO si corregir)
        self.tracking_xy = False
        self.tracking_z = False
        self.save_data_state = False
        # Si corregir (implica trackear)
        self.feedback_xy = False
        self.feedback_z = False
        self.camON = False  # Si la cámara está prendida, para cámaras on/off

        self.npoints = 1200  # número de puntos a graficar
        self.buffersize = 30000  # tamano buffer correcciones xy y z

        # posicion actual del centro en cada ROI
        self.currentx: np.ndarray = np.zeros((1,))
        self.currenty: np.ndarray = np.zeros((1,))

        # Los llamamos en toggle_tracking
        self.reset_graph()
        self.reset_data_arrays()

        self.counter = 0  # Se usa para generar un patrón. Es cuántas veces se llamó a update

        # saves displacement when offsetting setpoint for feedbackloop
        # Estos son sólo para hacer un pattern de test
        self.displacement = np.array([0.0, 0.0])  # Solo para test pattern
        self.pattern = False

        self.previous_image = None  # para chequear que la imagen cambie
        self.currentz = 0.0  # Valor en pixeles dentro del roi del z

        if self.camera.open_device():
            self.camera.set_roi(16, 16, 1920, 1200)
            try:
                self.camera.alloc_and_announce_buffers()
                self.camera.start_acquisition()
            except Exception as e:
                print("Exception", str(e))
        else:
            self.camera.destroy_all()
            raise Exception("No pude abrir la cámara")

    @pyqtSlot(int, bool)
    def toggle_tracking_shutter(self, num, val):
        """Toma acciones con los shutters que lo involucran. DESACTIVADO.

        Previamente el shutter 5 bloqueaba la señal z y el 6 la señal xy. Al
        mudar el setup sacamos estos shutters, así que esta función no hace
        nada (hoy por hoy)
        """
        # TODO: change code to also update checkboxes in case of minflux
        # measurement
        if num == 8:
            to_update = (5, 6,)
        elif (num == 5) or (num == 6):
            to_update = (num,)
        else:
            to_update = tuple()
        for num in to_update:
            tools.toggle_shutter(self.adw, num, bool(val))
            _lgr.info('Tracking shutter %s %s', num, "opened" if val else "closed")

    @pyqtSlot(int, bool)
    def shutter_handler(self, num, on):
        self.shuttermodeSignal.emit(num, on)

    @pyqtSlot(bool)
    def liveview(self, value):
        """
        Connection: [frontend] liveviewSignal
        Description: toggles start/stop the liveview of the camera.
        """
        if value:
            self.camON = True
            self.liveview_start()
        else:
            self.liveview_stop()
            self.camON = False

    def liveview_start(self):
        if self.camON:
            self.viewtimer.stop()
            self.camON = False
        self.camON = True
        self.viewtimer.start(self.xyz_time)

    def liveview_stop(self):
        self.viewtimer.stop()
        _lgr.debug("cameraTimer: stopped")
        self.camON = False

    def update(self):
        """General update method.

        Trackea y corrige si esta configurado.
        """
        self.update_view()

        if self.tracking_xy:
            self.track('xy')
        if self.tracking_z:
            self.track('z')
        if self.tracking_xy or self.tracking_z:
            # Grabar salida
            if self.j >= self.buffersize or self.j_z >= (self.buffersize):
                self.export_data()
                self.reset_data_arrays()
            # FIXME: Ver cómo separar los trackings
            self.update_graph_data()
            if self.feedback_xy:
                self.correct_xy()
            if self.feedback_z:
                self.correct_z()

        # De acá para abajo es para hacer un patrón para algún test
        if self.pattern:
            val = (self.counter - self.initcounter)
            reprate = 10  # Antes era 10 para andor
            if (val % reprate == 0):
                self.make_tracking_pattern(val//reprate)
        # counter to check how many times this function is executed
        self.counter += 1

    def update_view(self):
        """Image/data update while in Liveview mode."""
        # acquire image
        # This is a 2D array, (only R channel)
        self.image = self.camera.on_acquisition_timer()
        # WARNING: fix to match camera orientation with piezo orientation
        self.image = np.rot90(self.image, k=3)  # Este es nuestro estandar
        if np.all(self.previous_image == self.image):
            _lgr.error('Latest_frame equal to previous frame')
        self.previous_image = self.image

        if (VIDEO and self.save_data_state):
            self.video.append(self.image)

        # send image to gui
        self.changedImage.emit(self.image)

    # Incorporo cambios con vistas a añadir data actualizada de z
    def update_graph_data(self):
        """Update the data displayed in the graphs and pass around."""
        if self.ptr < self.npoints:
            self.time[self.ptr] = self.currentTime
            if self.tracking_xy:
                self.xData[self.ptr, :] = self.x + self.displacement[0]
                self.yData[self.ptr, :] = self.y + self.displacement[1]
                self.changedXYData.emit(self.time[0:self.ptr + 1],
                                        self.xData[0:self.ptr + 1],
                                        self.yData[0:self.ptr + 1],
                                        )
            if self.tracking_z:
                self.zData[self.ptr] = self.z  # Es el delta z en nm respecto al inicial
                self.avgIntData[self.ptr] = self.avgInt
                self.changedZData.emit(self.time[0:self.ptr + 1],
                                       self.zData[0:self.ptr + 1],
                                       self.avgIntData[0:self.ptr + 1],
                                       )
        else:  # roll a mano
            self.time[:-1] = self.time[1:]
            self.time[-1] = self.currentTime
            if self.tracking_xy:
                self.xData[:-1] = self.xData[1:]
                self.xData[-1, :] = self.x + self.displacement[0]
                self.yData[:-1] = self.yData[1:]
                self.yData[-1, :] = self.y + self.displacement[1]
                self.changedXYData.emit(self.time, self.xData, self.yData,)
            if self.tracking_z:
                self.zData[:-1] = self.zData[1:]
                self.zData[-1] = self.z
                self.avgIntData[:-1] = self.avgIntData[1:]
                self.avgIntData[-1] = self.avgInt
                self.changedZData.emit(self.time, self.zData, self.avgIntData)

        self.ptr += 1

    # esta función es igual a la de xyz_tracking porque es para xy únicamente
    # Como z no es una lista, no hay nada especial acá
    @pyqtSlot(bool)
    def toggle_tracking(self, val):
        """Inicia el tracking de las marcas y de z (sin corregir).

        Connection: [frontend] trackingBeadsBox.stateChanged
        Description: toggles ON/OFF tracking of fiducial fluorescent beads.
        Drift correction feedback loop is not automatically started.
        """
        self.toggle_tracking_xy(val)
        self.toggle_tracking_z(val)

    @pyqtSlot(bool)
    def toggle_tracking_xy(self, val):
        """Inicia el tracking de las beads (sin corregir).

        Drift correction feedback loop is not automatically started.
        """
        if val is True:
            if self.tracking_xy:
                _lgr.info("Doble activación tracking xy")
                return
            self.reset_xy_graph()
            if not self.tracking_z:
                # self.reset_data_arrays()   mover a feedback
                self.reset_z_graph()  # Prepararlo para que exista, vacío
                self.reset_graph()  # esto manda los buffers al front
            self.tracking_xy = True
            self.counter = 0  # Como es para patrones lo pongo acá

            # initialize relevant xy-tracking arrays
            size = len(self.roi_coordinates_list)

            self.currentx = np.zeros(size)
            self.currenty = np.zeros(size)
            self.x = np.zeros(size)  # Deltas respecto a posicion inicial
            self.y = np.zeros(size)
            self.initialx = np.zeros(size)
            self.initialy = np.zeros(size)
            self._initialize_xy_positions()

        elif val is False:
            if not self.tracking_xy:
                _lgr.info("Doble desactivación tracking xy")
                return
            self.tracking_xy = False
        else:
            _lgr.error("Valor inválido pasado a toggle_tracking_z: %s", val)

    # @pyqtSlot(bool)
    def toggle_tracking_z(self, val):
        """Inicia el tracking del spot z (sin corregir).

        Drift correction feedback loop is not automatically started.
        """
        if val is True:
            if self.tracking_z:
                _lgr.info("Doble activación tracking z")
                return
            self.reset_z_graph()
            if not self.tracking_xy:
                self.reset_xy_graph()
                self.reset_graph()
                # self.reset_data_arrays()

            # init z position
            # si queremos prender el tracking, seguramente está el ROI definido,
            # etc.
            self.center_of_mass()  # Esto actualiza la posicion z medida
            self.initialz = self.currentz
            self.tracking_z = True
        elif val is False:
            if not self.tracking_z:
                _lgr.info("Doble desactivación tracking z")
                return
            self.tracking_z = False
        else:
            _lgr.error("Valor inválido pasado a toggle_tracking_z: %s", val)

    @pyqtSlot(bool)
    def toggle_feedback(self, val, mode='continous'):
        """Inicia y detiene los procesos de estabilizacion de la ADwin.

        Connection: [frontend] feedbackLoopBox.stateChanged
        Description: toggles ON/OFF feedback for either continous (TCSPC)
        or discrete (scan imaging) correction
        """
        if mode not in ('discrete', 'continous'):
            _lgr.warning("Invalid feedback mode: %s", mode)
        if type(val) is not bool:
            _lgr.warning("Toggling feedback mode not boolean; %s", type(val))

        # FIXME: feedback split
        self.set_xy_feedback(val, mode)
        self.set_z_feedback(val, mode)
        _lgr.debug('Feedback loop active: %s', val)
        # self.updateGUIcheckboxSignal.emit(self.tracking_value,
        #                                   self.feedback_active,
        #                                   self.save_data_state)

    def set_z_feedback(self, val, mode='continous'):
        """Inicia y detiene los procesos de estabilizacion de la ADwin para z."""
        if val is True:
            if self.feedback_z:  # esto es para evitar loops por la actualización de back a front
                _lgr.info("Dobre activación de z")
                return
            if mode == 'continous':  # set up and start actuator process
                self.set_z_actuator_param()
                self.adw.Start_Process(3)  # proceso para z
                _lgr.info('Process 3 started. Status: %s', self.adw.Process_Status(3))
                _lgr.debug('z Feedback loop ON')
                self.feedback_z = True
        elif val is False:
            if not self.feedback_z:
                _lgr.info("Dobre desactivación de z")
            self.feedback_z = False
            self.adw.Stop_Process(3)
            _lgr.info('Process 3 stopped. Status: %s', self.adw.Process_Status(3))
            _lgr.debug('z Feedback loop Off')
        else:
            _lgr.error("Deberías pasar un booleano, no: %s", val)

    def set_xy_feedback(self, val, mode='continous'):
        """Inicia y detiene los procesos de estabilizacion de la ADwin para xy."""
        if val is True:
            if self.feedback_xy:
                _lgr.info("Doble activacion feedback xy")
                return
            if (mode == 'continous') and (not self.tracking_xy):
                _lgr.warning("Requested XY feedback without tracking. Enabling tracking")
                self.toggle_tracking_xy(True)
            if mode == 'continous':  # set up and start actuator process
                self.set_xy_actuator_param()
                self.adw.Start_Process(4)  # proceso para xy
                _lgr.info('Process 4 started. Status: %s', self.adw.Process_Status(4))
                _lgr.debug('xy Feedback loop ON')
                self.feedback_xy = True
        elif val is False:
            if not self.feedback_xy:
                _lgr.info("Doble desactivacion feedback xy")
            self.feedback_xy = False
            # FIXME: check condition below
            if True:  # mode == 'continous':
                self.adw.Stop_Process(4)
                _lgr.info('Process 4 stopped. Status: %s', self.adw.Process_Status(4))
                self.displacement = np.array([0.0, 0.0])
                _lgr.debug('xy Feedback loop Off')
        else:
            _lgr.error("Deberías pasar un booleano, no: %s", val)

    def center_of_mass(self):
        """Calculate z image center of mass."""
        # set main reference frame
        xmin, xmax, ymin, ymax = self.zROIcoordinates
        # select the data of the image corresponding to the ROI
        zimage = self.image[xmin:xmax, ymin:ymax]
        # calculate center of mass
        self.m_center = np.array(ndi.measurements.center_of_mass(zimage))
        # calculate z estimator
        # TODO: copiar de new_focus
        self.currentz = self.m_center[0]

        # El estimador está en píxeles... fraccionarios

    # Le estoy agregando un parámetro (roi_coordinates) para que sea como en
    # xyz_tracking
    def gaussian_fit(self, roi_coordinates) -> (float, float):
        """Devuelve el centro del fiteo, en nm respecto al ROI.

        TODO: Creo que hay un ida y vuelta ridiculo entre nm y px
        """
        # set main reference frame
        roi_coordinates = np.array(roi_coordinates, dtype=np.int32)
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
        # en el código original era 10, pero lo cambié porque así está en xyz_tracking
        xrange = 15  # in px
        yrange = 15  # in px
        xmin_id = max(0, int(x_center_id - xrange))
        xmax_id = min(int(x_center_id + xrange), xmax-xmin)
        ymin_id = max(int(y_center_id - yrange), 0)
        ymax_id = min(int(y_center_id + yrange), ymax-ymin)

        array_sub = array[xmin_id:xmax_id, ymin_id:ymax_id]
        xsubsize = xmax_id - xmin_id  # 2 * xrange
        ysubsize = ymax_id - ymin_id  # 2 * yrange

        x_sub_nm = np.arange(0, xsubsize) * PX_SIZE
        y_sub_nm = np.arange(0, ysubsize) * PX_SIZE
        [Mx_sub, My_sub] = np.meshgrid(x_sub_nm, y_sub_nm)

        # make initial guess for parameters
        bkg = np.min(array)
        A = np.max(array) - bkg
        σ = 200  # nm, antes era 130nm
        x0 = x_sub_nm[int(xsubsize/2)]
        y0 = y_sub_nm[int(ysubsize/2)]

        initial_guess_G = [A, x0, y0, σ, σ, bkg]

        if np.size(array_sub) == 0:
            print('WARNING: array_sub is []')

        poptG, pcovG = opt.curve_fit(PSF.gaussian2D, (Mx_sub, My_sub),
                                     array_sub.ravel(), p0=initial_guess_G)
        perr = np.sqrt(np.diag(pcovG))
        # print('perr', perr)

        # retrieve results
        poptG = np.around(poptG, 2)
        A, x0, y0, σ_x, σ_y, bkg = poptG
        x = x0 + Mx_nm[xmin_id, ymin_id]
        y = y0 + My_nm[xmin_id, ymin_id]

        currentx = x
        currenty = y

        return currentx, currenty

    def _initialize_xy_positions(self):
        """Set the current xy positions as the starting ones."""
        self._fit_xy_rois()
        for i, roi in enumerate(self.roi_coordinates_list):
            self.initialx[i] = self.currentx[i]
            self.initialy[i] = self.currenty[i]

    def _fit_xy_rois(self):
        """Fitea lo ROIs xy con la última imagen obtenida."""
        for i, roi in enumerate(self.roi_coordinates_list):
            roi = self.roi_coordinates_list[i]
            try:
                self.currentx[i], self.currenty[i] = self.gaussian_fit(roi)
            except Exception as e:
                self.currentx[i] = self.initialx[i]
                self.currenty[i] = self.initialy[i]
                _lgr.warning("Error en Gaussian_fit: %e", e)

    def _check_xy_fit_limits(self):
        """Chequea que los fiteos no se hayan ido de límite.

        Si se fueron de límite, los pone como si estuviesen OK. Lo correcto sería que
        no los tenga en cuenta.

        FIXME: poner NANs y usar averages que lo tengan en cuenta
        """
        maxdist = 200  # in nm
        for i, roi in enumerate(self.roi_coordinates_list):
            if (np.abs(self.initialx[i] - self.currentx[i]) > maxdist
                    or np.abs(self.initialy[i] - self.currenty[i]) > maxdist):
                _lgr.warning('Max dist exceeded in xy ROI #%s', i)
                self.currentx[i] = self.initialx[i]
                self.currenty[i] = self.initialy[i]

    def track(self, track_type):  # Añado parámetro para trabajar en xy y z
        """Track fiducial marks and update shifts.

        Function to track fiducial markers and z reflection spot.
        The position of the markers  is calculated through an xy gaussian fit.
        The position of the z spot is calculated using center of mass

        If save_data_state = True it saves the xy data
        """
        # Calculate average intensity in the image to check laser fluctuations
        self.avgInt = np.mean(self.image)

        # xy track routine of N=size fiducial AuNP
        if track_type == 'xy':
            self._fit_xy_rois()
            self._check_xy_fit_limits()
            # self.x, y and z are relative to initial positions
            for i, roi in enumerate(self.roi_coordinates_list):
                self.x[i] = self.currentx[i] - self.initialx[i]
                self.y[i] = self.currenty[i] - self.initialy[i]
                self.currentTime = time.time() - self.startTime

            if self.save_data_state:
                self.time_array[self.j] = self.currentTime
                self.x_array[self.j, :] = self.x + self.displacement[0]
                self.y_array[self.j, :] = self.y + self.displacement[1]
                self.j += 1

        # z track of the reflected beam
        if track_type == 'z':
            self.center_of_mass()
            self.z = (self.currentz - self.initialz) * PX_Z  # self.z in nm
            if self.save_data_state:
                self.z_time_array[self.j_z] = time.time() - self.startTime
                self.z_array[self.j_z] = self.currentz
                self.j_z += 1
        ...


    def correct_xy(self, mode='continous'):
        """Corregir posicion xy."""
        # TODO: implementar PI

        xmean = np.mean(self.x)
        ymean = np.mean(self.y)

        dx = 0
        dy = 0

        # Thresholds en unidades de self.x, self.y (nm)
        threshold = 3
        xy_far_threshold = 12
        xy_correct_factor = 0.6

        security_thr = 0.35  # in µm

        # HINT: los signos están acorde a la platina y la rotación de la imagen
        if np.abs(xmean) > threshold:
            dx = xmean / 1000  # conversion to µm
            if abs(dx) < xy_far_threshold:
                dx *= xy_correct_factor

        if np.abs(ymean) > threshold:
            dy = -ymean / 1000  # conversion to µm
            if abs(dy) < xy_far_threshold:
                dy *= xy_correct_factor

        if (abs(dx) > security_thr or abs(dy) > security_thr):
            _lgr.error('xy Correction movement larger than 200 nm,'
                  ' active correction turned OFF')
            self.toggle_feedback(False, mode)
        else:
            # compensate for the mismatch between camera/piezo system of
            # reference
            # theta = np.radians(-3.7)   # 86.3 (or 3.7) is the angle between camera and piezo (measured)
            # c, s = np.cos(theta), np.sin(theta)
            # R = np.array(((c,-s), (s, c)))

            # dy, dx = np.dot(R, np.asarray([dx, dy])) #ver si se puede arreglar esto añadiendo dz

            # add correction to piezo position
            currentXposition = tools.convert(self.adw.Get_FPar(70), 'UtoX')
            currentYposition = tools.convert(self.adw.Get_FPar(71), 'UtoX')
            # Sólo xy
            targetZposition = tools.convert(self.adw.Get_FPar(72), 'UtoX')

            # TODO: chequear signos acá o arriba
            targetXposition = currentXposition + dx
            targetYposition = currentYposition + dy

            if mode == 'continous':
                # Le mando al actuador las posiciones x,y,z
                self.actuator_xyz(targetXposition, targetYposition,
                                  targetZposition)
            if mode == 'discrete':
                #self.moveTo(targetXposition, targetYposition, currentZposition, pixeltime=10)
                self.target_x = targetXposition
                self.target_y = targetYposition

    def correct_z(self, mode='continous'):
        """Corregir posicion z."""
        # TODO: implementar PI
        # TODO: implementar Discreto para PSF (para single correction)
        dz = 0

        # Thresholds en unidades de self.z (nm)
        z_threshold = 3
        z_far_threshold = 12
        z_correct_factor = .6

        security_thr = 0.35  # in µm

        if np.abs(self.z) > z_threshold:
            dz = -self.z / 1000
            if abs(dz) < z_far_threshold:
                dz *= z_correct_factor

        if abs(dz) > 2 * security_thr:
            print(datetime.now(),
                  '[xyz_tracking] Z Correction movement larger than 200 nm,'
                  ' active correction turned OFF')
            self.toggle_feedback(False, mode)
        else:
            # add correction to piezo position
            targetXposition = tools.convert(self.adw.Get_FPar(70), 'UtoX')
            targetYposition = tools.convert(self.adw.Get_FPar(71), 'UtoX')
            currentZposition = tools.convert(self.adw.Get_FPar(72), 'UtoX')

            # TODO: chequear signos acá o arriba
            targetZposition = currentZposition + dz  # in µm

            if mode == 'continous':
                # Le mando al actuador las posiciones x,y,z
                self.actuator_xyz(targetXposition, targetYposition,
                                  targetZposition)
            if mode == 'discrete':
                #self.moveTo(targetXposition, targetYposition, currentZposition, pixeltime=10)
                self.target_z = targetZposition

    @pyqtSlot(bool, bool)
    def single_xy_correction(self, feedback_val, initial):
        """
        From: [psf] xySignal
        feedback_val is unused
        """
        if initial:
            self.toggle_feedback(True, mode='discrete')
            _lgr.info("Initial xy single tracking")
        if not self.camON:
            print(datetime.now(), 'singlexy liveview started')
            self.camON = True
        time.sleep(0.200)

        self.update_view()
        if initial:
            self._initialize_xy_positions()

        self.track('xy')
        self.update_graph_data()
        self.correct_xy(mode='discrete')
        target_x = np.round(self.target_x, 3)
        target_y = np.round(self.target_y, 3)
        _lgr.info('Discrete correction to (%s, %s)', target_x, target_y)
        self.xyIsDone.emit(True, target_x, target_y)
        _lgr.debug('Single xy correction ended')

    @pyqtSlot(bool, bool)
    def single_z_correction(self, feedback_val, initial):
        if initial:
            if not self.camON:
                self.camON = True

        self.update_view()
        if initial:
            self.center_of_mass()  # Esto actualiza la posicion z medida
            self.initialz = self.currentz
        self.track('z')
        self.update_graph_data()
        self.correct_z(mode='discrete')
        # if self.save_data_state:
        #     self.time_array.append(self.currentTime)
        #     self.z_array[self.j_z] = self.currentz
        #     self.j_z += 1
        self.camON = False
        self.zIsDone.emit(True, self.target_z)

    def calibrate_z(self):
        self.viewtimer.stop()
        time.sleep(0.100)
        nsteps = 40
        zrange = 0.4  # en µm
        old_z_param = self.adw.Get_FPar(72)
        old_z = tools.convert(old_z_param, 'UtoX')
        calibData = np.zeros(nsteps)
        zData = np.linspace(old_z - zrange/2, old_z + zrange/2, nsteps)

        was_running = self.adw.Process_Status(3)
        if not was_running:
            self.adw.Start_Process(3)
        time.sleep(0.100)
        for i, z in enumerate(zData):
            z_f = tools.convert(z, 'XtoU')
            self.adw.Set_FPar(32, z_f)
            time.sleep(.125)  # Ojo con este parámetro
            self.update()
            calibData[i] = self.currentz
        #     np.save(f'C:\\Data\\img_{i}-{int(z*1000)}pm.npy', self.image)
        # np.save('C:\\Data\\zData.npy', np.array(zData))
        # np.save('C:\\Data\\calib.npy', np.array(calibData))
        print("Calibración")
        print("z\tcm")
        for z, cm in zip(zData, calibData):
            print(f"{z}\t{cm}")
        self.adw.Set_FPar(32, old_z_param)
        time.sleep(0.500)
        if not was_running:
            self.adw.Stop_Process(3)
        time.sleep(0.200)
        self.viewtimer.start(self.xyz_time)

    def set_actuator_param(self, pixeltime=1000):
        """Inicializar los parámetros antes de arrancar los scripts."""
        self.set_actuator_param_xy(pixeltime)
        self.set_actuator_param_z(pixeltime)

    def set_actuator_param_xy(self, pixeltime=1000):
        """Inicializar los parámetros xy antes de arrancar los scripts."""
        self.adw.Set_FPar(46, tools.timeToADwin(pixeltime))

        # set-up actuator initial param script
        # MoveTo usa un script que actualiza estos valores, podemos confiar
        currentXposition = tools.convert(self.adw.Get_FPar(70), 'UtoX')
        currentYposition = tools.convert(self.adw.Get_FPar(71), 'UtoX')
        x_f = tools.convert(currentXposition, 'XtoU')
        y_f = tools.convert(currentYposition, 'XtoU')

        # set-up actuator initial params
        self.adw.Set_FPar(40, x_f)
        self.adw.Set_FPar(41, y_f)

        # Para mi es igual a hacer esto:  VERIFICADO
        # self.adw.Set_FPar(40, self.adw.Get_FPar(70))
        # self.adw.Set_FPar(41, self.adw.Get_FPar(71))

        # Comento porque son cosas viejas (Andi)
        # self.adw.Set_Par(40, 1)

    def set_actuator_param_z(self, pixeltime=1000):
        """Inicializar los parámetros z antes de arrancar los scripts."""
        self.adw.Set_FPar(36, tools.timeToADwin(pixeltime))

        # set-up actuator initial param script
        # MoveTo usa un script que actualiza estos valores, podemos confiar
        currentZposition = tools.convert(self.adw.Get_FPar(72), 'UtoX')
        z_f = tools.convert(currentZposition, 'XtoU')

        # set-up actuator initial params
        self.adw.Set_FPar(32, z_f)

        # Para mi es igual a hacer esto: VERIFICADO
        # self.adw.Set_FPar(32, self.adw.Get_FPar(72))

        # Comento porque son cosas viejas (Andi)
        # self.adw.Set_Par(30, 1)


    def actuator_xyz(self, x_f, y_f, z_f):
        """Setear los parámetros de tracking de la adwin mientras corre.

        Estos parámetros son usados por los procesos:
            actuator_z.bas: (Proceso 3)
                FPar_32 es el setpoint z
            actuator_xy.bas: (Proceso 4)
                FPar_40 es el setpoint x
                FPar_41 es el setpoint y
        """
        x_f = tools.convert(x_f, 'XtoU')
        y_f = tools.convert(y_f, 'XtoU')
        z_f = tools.convert(z_f, 'XtoU')

        self.adw.Set_FPar(40, x_f)
        self.adw.Set_FPar(41, y_f)
        self.adw.Set_FPar(32, z_f)

        # Estas dos líneas son de los scripts viejos (ver .bas y .bak)
        # self.adw.Set_Par(40, 1)
        # self.adw.Set_Par(30, 1) #Añado para z, focus.py

    def set_moveTo_param(self, x_f, y_f, z_f, n_pixels_x=128, n_pixels_y=128,
                         n_pixels_z=128, pixeltime=2000):
        """Setea los parámatros para mover a una posición x,y,z.

        Ver moveTo y proceso 2 de ADwin.
        """
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
        """Move to specified position."""
        self.set_moveTo_param(x_f, y_f, z_f)
        self.adw.Start_Process(2)

    def reset_graph(self):
        """Prepare graphs buffers and internal graph data for new measurement.

        FIXED: Initial flags were set here
        """
        # self.initial = True
        # self.initial_focus = True

        # buffer para data z para graficar
        self.reset_other_graphs()
        self.time = np.zeros(self.npoints)
        self.ptr = 0  # Posición en los buffers de graficación
        self.startTime = time.time()

        self.changedData.emit(self.time, self.xData, self.yData, self.zData,
                              self.avgIntData)

    def reset_xy_graph(self):
        """Prepare xy graphs buffers for new measurement."""
        self.xData = np.full((self.npoints, len(self.roi_coordinates_list)), np.nan)
        self.yData = np.full((self.npoints, len(self.roi_coordinates_list)), np.nan)

    def reset_z_graph(self):
        """Prepare z graphs buffers and internal graph data for new measurement."""
        self.zData = np.full((self.npoints,), np.nan)

    def reset_other_graphs(self):
        """Reset data unrealted to xy and z."""
        self.avgIntData = np.full((self.npoints,), np.nan)

    def reset_data_arrays(self):
        """Reset/create buffers holding measured positions vs time."""
        self.time_array = np.zeros(self.buffersize, dtype=np.float16)
        self.z_time_array = np.zeros(self.buffersize, dtype=np.float16)
        self.x_array = np.zeros((self.buffersize,
                                 len(self.roi_coordinates_list)),
                                dtype=np.float16)
        self.y_array = np.zeros((self.buffersize,
                                 len(self.roi_coordinates_list)),
                                dtype=np.float16)
        # self.z_array = []  # TODO: hacer consistente con xy. z no se graba
        self.z_array = np.zeros((self.buffersize,), dtype=np.float16)
        self.j = 0  # iterator on the data arrays
        self.j_z = 0

    # Está en xy y en focus. Creo que no está conectado a nada (A PSF, Andi)
    @pyqtSlot(bool)
    def get_stop_signal(self, stoplive):
        """Para todo.

        Connection: [psf] xyStopSignal
        Description: stops liveview, tracking, feedback if they where running
        to start the psf measurement with discrete xy - z corrections
        """
        _lgr.debug("Got stop signal with value %s", stoplive)
        self.toggle_feedback(False)
        self.toggle_tracking(False)

        # TODO: Ver si no restringir a xy
        self.reset_graph()
        self.reset_data_arrays()

        # TODO: sync this with GUI checkboxes (Lantz typedfeat?)
        self.save_data_state = True

        if not stoplive:
            self.liveviewSignal.emit(False)

    def export_data(self):
        """Export t and x, y for each Roi data into a .txt file.

        TODO: ver info z
        """
        fname = self.xy_filename
        # case distinction to prevent wrong filenaming when starting minflux
        # or psf measurement
        if fname[0] == '!':
            basefilename = fname[1:]
        else:
            basefilename = tools.getUniqueName(fname)
        filename = basefilename + '.txt'

        size = self.j
        N_NP = len(self.roi_coordinates_list)

        savedData = np.zeros((size, 2*N_NP+1))

        savedData[:, 0] = self.time_array[0:self.j]
        savedData[:, 1:N_NP+1] = self.x_array[0:self.j, :]
        savedData[:, N_NP+1:2*N_NP+1] = self.y_array[0:self.j, :]

        np.savetxt(filename, savedData,  header="t(s) " +
                   " ".join(f"x{_}(nm)" for _ in range(N_NP)) +
                   " ".join(f"y{_}(nm)" for _ in range(N_NP)))
        _lgr.info('xy data exported to %s', filename)
        _lgr.debug('Exported data shape: %s', np.shape(savedData))

        # TODO: guardar frame final
        # self.export_image()

        # if VIDEO:
        #     tifffile.imwrite(fname + 'video' + '.tif', np.array(self.video))
        
        filename = self.z_filename + '.txt'

        size = self.j_z
        savedData = np.zeros((2, size))

        savedData[0, :] = self.z_time_array[0: self.j_z]
        savedData[1, :] = self.z_array[0: self.j_z]
        np.savetxt(filename, savedData.T, header='t (s), z (px)')
        _lgr.info('z data exported to %s', filename)

    # Dejo esta funcion como está, se repite en xy y en focus.py
    @pyqtSlot(bool)
    def get_save_data_state(self, val):
        """Setea si tiene o no que grabar datos xy.

        Connection: [frontend] saveDataSignal
        Description: gets value of the save_data_state variable, True -> save,
        False -> don't save
        """
        self.save_data_state = val
        _lgr.debug('save_data_state = %s', val)

    # antes se usaba roi_coordinates_array que se convertía a int en
    # ROIcoordinates
    @pyqtSlot(str, int, list)
    def get_roi_info(self, roi_type: str,
                     N: int,
                     coordinates_list: list):  # coordinates_list: list[np.ndarray]
        """Toma la informacion del ROI que viene de emit_roi_info en el frontend.

        Parameters
        ----------
            roi_type: str
                'xy' o 'z'
            N: int
                Número de ROIs en la lista. Al pp.
            coordinates_list: list of np.ndarray[4]
                Lista de (xmin, xmax, ymin, ymax) de cada ROI en pixeles

        Connection: [frontend] roiInfoSignal
        Description: gets coordinates of the ROI in the GUI
        """
        if roi_type == 'xy':
            self.roi_coordinates_list = coordinates_list[:]  # LISTA
        elif roi_type == 'z':
            self.zROIcoordinates = coordinates_list[0].astype(int)

    @pyqtSlot()
    def get_lock_signal(self):  # Dejo esta funcion como está
        """Activa tracking. No es muy claro qué función cumple.

        Connection: [minflux] xyzStartSignal
        Description: activates tracking and feedback
        """
        if not self.camON:
            self.liveviewSignal.emit(True)
        self.toggle_tracking(True)
        self.toggle_feedback(True)
        self.save_data_state = True
        # Esto está comentado en focus pero no en xy
        # self.updateGUIcheckboxSignal.emit(self.tracking_value,
        #                                   self.feedback_active,
        #                                   self.save_data_state)
        _lgr.debug('System xy locked')

    @pyqtSlot(np.ndarray, np.ndarray)
    def get_move_signal(self, r, r_rel):
        """Recibe de módulo Minflux para hacer patterns.

        TODO: entender qué bien qué hacer. Parece que recibe posiciones a las
        que moverse.
        TODO: si FPar_72 no está bien seteado esto se va a cualquier posición
        """
        self.toggle_feedback(False)
        # self.toggle_tracking(True)
        # self.updateGUIcheckboxSignal.emit(self.tracking_value,
        #                                   self.feedback_active,
        #                                   self.save_data_state)
        x_f, y_f = r
        z_f = tools.convert(self.adw.Get_FPar(72), 'UtoX')
        self.actuator_xyz(x_f, y_f, z_f)

    def start_tracking_pattern(self):
        """Se prepara para hacer un patrón.

        Ver módulo Minflux

        TODO: Terminar de entender
        """
        self.pattern = True
        self.initcounter = self.counter
        self.save_data_state = True

    def make_tracking_pattern(self, step):
        """Poner las posiciones de referencia en un cuadrado.

        TODO: Ver cómo se concilia con start_tracking_pattern.
        """
        if step < 2:
            return
        elif step == 2:
            dist = np.array([0.0, 20.0])
        elif step == 3:
            dist = np.array([20.0, 0.0])
        elif step == 4:
            dist = np.array([0.0, -20.0])
        elif step == 5:
            dist = np.array([-20.0, 0.0])
        else:
            self.pattern = False
            return

        self.initialx = self.initialx + dist[0]
        self.initialy = self.initialy + dist[1]
        self.displacement = self.displacement + dist

        _lgr.info('Moved setpoint by %s', dist)

    @pyqtSlot(str)
    def get_end_measurement_signal(self, fname):
        """Procesa un pedido de fin de medida.

        Description: at the end of the measurement exports the xy data
        Signals
        -------
            From: [minflux] xyzEndSignal or [psf] endSignal
        """
        self.xy_filename = fname
        self.export_data()
        # TODO: decide whether I want feedback ON/OFF at the end of measurement
        self.toggle_feedback(False)
        # check
        self.toggle_tracking(False)
        self.pattern = False
        self.reset_graph()
        self.reset_data_arrays()
        # comparar con la funcion de focus: algo que ver con focusTimer
        # TODO: ¿Seguro de que queremos apagar?
        if self.camON:
            self.viewtimer.stop()
            self.liveviewSignal.emit(False)

    @pyqtSlot(float)
    def get_focuslockposition(self, position):
        """Actualiza en otro módulo la posición de lock de z.

        Conectado con módulo SCAN. Sólo para registrar en el archivo de datos.
        """
        if position == -9999:
            # Lo mas parecido es z
            position = self.initialz * PX_Z
        else:
            position = self.currentz
        self.focuslockpositionSignal.emit(position)

    def make_connection(self, frontend):
        frontend.roiInfoSignal.connect(self.get_roi_info)
        frontend.closeSignal.connect(self.stop)
        frontend.saveDataSignal.connect(self.get_save_data_state)
        frontend.exportDataButton.clicked.connect(self.export_data)
        frontend.clearDataButton.clicked.connect(self.reset_graph)
        frontend.clearDataButton.clicked.connect(self.reset_data_arrays)
        # Falta botón de Calibrate con calibrate_z
        frontend.trackingBeadsBox.stateChanged.connect(
            lambda: self.toggle_tracking(frontend.trackingBeadsBox.isChecked())
            )
        frontend.shutterCheckbox.stateChanged.connect(
            lambda: self.toggle_tracking_shutter(
                8, frontend.shutterCheckbox.isChecked()))
        frontend.liveviewButton.clicked.connect(self.liveview)
        frontend.feedbackLoopBox.stateChanged.connect(
            lambda: self.toggle_feedback(frontend.feedbackLoopBox.isChecked()))
        frontend.xyPatternButton.clicked.connect(
            lambda: self.start_tracking_pattern
            )  # duda con esto, comparar con línea análoga en xyz_tracking

        # La función toggle_feedback se utiliza como un slot de PyQt y se
        # conecta al evento stateChanged de un cuadro de verificación llamado
        # feedbackLoopBox. Su propósito es activar o desactivar el feedback
        # (retroalimentación) para la corrección continua en el modo especificado.

        # TO DO: clean-up checkbox create continous and discrete feedback loop

        # lambda function and gui_###_state are used to toggle both backend
        # states and checkbox status so that they always correspond
        # (checked <-> active, not checked <-> inactive)

    @pyqtSlot()
    def stop(self):
        """Cierra TODO antes de terminar el programa.

        viene del front.
        """
        self.toggle_tracking_shutter(8, False)
        time.sleep(1)
        self.viewtimer.stop()
        # Go back to 0 position
        x_0 = 0
        y_0 = 0
        z_0 = 0
        self.moveTo(x_0, y_0, z_0)
        self.camera.destroy_all()
        _lgr.info('[xyz_tracking] IDS camera shut down')


if __name__ == '__main__':
    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # initialize devices
    DEVICENUMBER = 0x1
    adw = ADwin.ADwin(DEVICENUMBER, 1)
    scan.setupDevice(adw)

    # if camera wasnt closed properly just keep using it without opening new one
    try:
        camera = ids_cam.IDS_U3()
    except Exception:
        print("Excepcion inicializando la cámara... seguimos")

    gui = Frontend()
    worker = Backend(camera, adw)

    gui.make_connection(worker)
    worker.make_connection(gui)

    # Creamos un Thread y movemos el worker ahi, junto con sus timer, ahi
    # realizamos la conexión
    xyzThread = QtCore.QThread()
    worker.moveToThread(xyzThread)
    worker.viewtimer.moveToThread(xyzThread)
    # TODO: si este connect no se pasa al interior, no va a funcionar con otros
    # modulos. Yo haría una única función que mueva al worker y a todos sus
    # timers a un thread dado
    worker.viewtimer.timeout.connect(worker.update)
    xyzThread.start()

    # initialize fpar_70, fpar_71, fpar_72 ADwin position parameters
    pos_zero = tools.convert(0, 'XtoU')
    worker.adw.Set_FPar(70, pos_zero)
    worker.adw.Set_FPar(71, pos_zero)
    worker.adw.Set_FPar(72, pos_zero)

    worker.moveTo(10, 10, 10)  # in µm

    time.sleep(0.200)

    gui.setWindowTitle('xyz drift correction test with IDS')
    gui.show()
    app.exec_()
