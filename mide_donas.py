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
from PyQt5.QtWidgets import (QWidget, QPushButton, QApplication, QDialog,
                             QDialogButtonBox, QVBoxLayout, QLabel, QLineEdit)
from PyQt5.QtCore import pyqtSignal, pyqtSlot


from scan import Backend as _scan_backend
_lgn.basicConfig(format='%(levelname)s:%(message)s', level=_lgn.DEBUG)
_lgr = _lgn.getLogger(__name__)

class DonutScan(QDialog):
    # sgn_scan = pyqtSignal(str)

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
        self._scan.frameIsDone.connect(self.frame_finished)
        self.abort = False

    @pyqtSlot
    def frame_finished(self):
        """Procesa cada dona."""
        _lgr.info("Dona %s escaneada.", self._current_donut)
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


# def find_radial_center(image: _np.ndarray):
#     if len(image.shape) != 2:
#         raise ValueError("Invalid donut shape")
#     x, y = [_np.arange(_) for _ in image.shape]
#     xx, yy = _np.meshgrid(x, y)
#     interp = _sp.interpolate.RegularGridInterpolator((x, y), image,
#                                                      # bounds_error=False,
#                                                      fill_value=0.0)
#     vinterp = _np.vectorize(interp)

#     for xcenter in x[5:-5]:
#         for ycenter in y[5:-5]:
#             max_r = min(xcenter, x[-1]-xcenter, ycenter, y[-1]-ycenter)
#             print("max_r = ", max_r)
#             # arclen = 2*np.pi*r
#             angle = np.linspace(0, 2*_np.pi, 360, endpoint=False)
#             for r in np.linspace()
#             value = vinterp(xcenter + r * np.sin(angle),
#                             ycenter + r*np.cos(angle))


#     return xx, yy


# def analyze_donut(dona: _np.ndarray):
#     if len(dona.shape) != 2:
#         raise ValueError("Invalid donut shape")
#     xc, yc = _sp.ndimage.center_of_mass(dona)
    
#     pass



if __name__ == '__main__':
    import imageio as iio
    import matplotlib.pyplot as plt
    im = _np.array(iio.mimread("/home/azelcer/Devel/datos_test/donasPSF.tiff"))
    for i in range(im.shape[0]):
        # im[i] = im[i].T[:, ::-1]
        m = im[i].min()
        mask = im[i] < 2. * m
        im[i][mask] = 0
        plt.figure()
        plt.contourf(im[i])
        # Porque el swap de x e y?
        yc, xc = _sp.ndimage.center_of_mass(im[i])
        plt.scatter(xc, yc)
        print(xc, yc)
    ...

