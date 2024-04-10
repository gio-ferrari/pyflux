#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatically adjust donuts

Created on Mon Apr  8 15:24:27 2024

@author: azelcer
"""
import logging as _lgn
import numpy as _np
import scipy as _sp
from find_center import find_center
from PyQt5.QtWidgets import (QWidget, QPushButton, QApplication, QDialog,
                             QDialogButtonBox, QVBoxLayout, QLabel, QLineEdit)
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import pyqtgraph as pg

_scan_backend = int  # for testing
# from scan import Backend as _scan_backend
_lgn.basicConfig(format='%(levelname)s:%(message)s', level=_lgn.INFO)
_lgr = _lgn.getLogger(__name__)

class DonutScan(QDialog):
    # sgn_scan = pyqtSignal(str)
    _plots: list[pg.PlotItem] = []  # plots de las donas
    _images: list[_np.ndarray] = [None, ] * 4
    _centerplots: list = [None, ] * 4

    def __init__(self, parent, scanback: _scan_backend, *args, **kwargs):
        _lgr.debug("Iniciando DonutScan")
        super().__init__(parent, *args, **kwargs)
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.measure_next_donut)
        self.buttonBox.rejected.connect(self.cancel_measurement)
        self.layout = QVBoxLayout()
        message = QLabel("OK para arrancar, cancel no hace nada")
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        self._current_donut = 0
        self._max_donuts = 4
        self._scan = scanback
        self.Available = True
        # self._
        # conectar scanback.frameIsDone acá o en el start/stop
        # self._scan.frameIsDone.connect(self.frame_finished)
        self.abort = False
        self.show()
        win = pg.GraphicsLayoutWidget(show=True, title="Measured donuts")
        win.resize(600, 600)
        win.setMinimumSize(600, 600)
        
        # Enable antialiasing for prettier plots
        # pg.setConfigOptions(antialias=True)
        for i in range(4):
            self._plots.append(win.addPlot(title=f"Dona {i+1}"))
            if i == 1:  # es lo que hay
                win.nextRow()
                
        # p1 = win.addPlot(title="Basic array plotting", y=_np.random.normal(size=100))
        
        # p2 = win.addPlot(title="Multiple curves")
        # p2.plot(_np.random.normal(size=100), pen=(255,0,0), name="Red curve")
        # p2.plot(_np.random.normal(size=110)+5, pen=(0,255,0), name="Green curve")
        # p2.plot(_np.random.normal(size=120)+10, pen=(0,0,255), name="Blue curve")
        
        # p1
        self._donuts_window = win
        im = _np.array(iio.mimread("/home/azelcer/Devel/datos_test/donasPSF.tiff"))
        for i in range(min(im.shape[0], len(self._plots))):  # demasiado cuidadoso
            self.update_donut_image(i, im[i])
            self.update_donut_center(i)

    def update_donut_image(self, donut_number: int, image: _np.ndarray):
        """Actualiza la imagen de la dona.

        Zero-based indexing
        """
        if donut_number > len(self._plots):
            _lgr.error("Invalid donut number: %s", donut_number)
        plot = self._plots[donut_number]
        plot.clear()
        img = pg.ImageItem(image)
        plot.addItem(img)
        self._images[donut_number] = image

    def update_donut_center(self, donut_number: int):
        """Actualiza el plot del centro."""
        plot = self._plots[donut_number]
        xc, yc = find_center(self._images[donut_number], trim=30)
        if self._centerplots[donut_number] is not None:
            try:
                plot.removeItem[self._centerplots[donut_number]]
            except Exception as e:
                _lgr.error("Error updating center: %s", e)
        sp = plot.plot([xc], [yc], pen=(200, 200, 200),
                       symbolBrush=(255, 0, 0),
                       symbolPen='w',
                       )
        self._centerplots[donut_number] = sp

    @pyqtSlot(bool, _np.ndarray)
    def frame_finished(self, unknwn_param: bool, image: _np.ndarray):
        """Procesa cada dona."""
        _lgr.info("Dona %s escaneada.", self._current_donut)
        self.update_donut_image(self._current_donut, image)
        self.update_donut_center(self._current_donut)
        self._current_donut += 1
        if self.abort:
            _lgr.warning("Medidas abortadas")
            self.abort = False
            self.Available = True
            # TODO: restaurar estado shutters en vez de cerrarlos
            for ns in range(1, self._max_donuts + 1):
                self._scan.control_shutters(ns, False)
                self._current_donut = 0
            return
        if self.current_donut >= self._max_donuts:
            _lgr.info("Terminamos de medir donuts")
            self.Available = True
            # TODO: restaurar estado shutters en vez de cerrarlos
            for ns in range(1, self._max_donuts + 1):
                self._scan.control_shutters(ns, False)
                self._current_donut = 0
        else:
            self.measure_next_donut()

    @pyqtSlot()
    def cancel_measurement(self):
        self.abort = True
        _lgr.warning("Abortando medida...")
        self._scan.liveview_stop()
        _lgr.warning("No va a haber stop signal?")

    @pyqtSlot()
    def start_measurements(self):
        """Init donuts measurements."""
        if not self.Available:
            _lgr.warning("ME niego a empezar una nueva medida")
            return

        _lgr.info("Iniciamos medida donuts")
        self.Available = False
        self.measure_next_donut()

    def measure_next_donut(self):
        """Measure next donut, if not finished."""
        self.Available = False
        # TODO: guardar estado shutters
        if self.current_donut < self._max_donuts:
            n = self.current_donut + 1
            _lgr.info("Prendiendo laser %s", n)
            for ns in range(1, self._max_donuts + 1):
                self._scan.control_shutters(n, n == ns)
            _lgr.info("Escaneando dona %s...", n)
            self._scan.liveview(True, 'frame')
        ...


if __name__ == '__main__':
    import imageio as iio
    import matplotlib.pyplot as plt
    import sys
    # im = _np.array(iio.mimread("/home/azelcer/Devel/datos_test/donasPSF.tiff"))
    # for i in range(im.shape[0]):
    #     plt.figure(f"f{i}")
    #     # plt.contour(im[i])
    #     plt.imshow(im[i])
    #     yc, xc = find_center(im[i], trim=30)
    #     plt.scatter(xc, yc)
    #     print(xc, yc)
    ...
    app = QApplication(sys.argv)
    ex = DonutScan(None, 11)
    sys.exit(app.exec())
