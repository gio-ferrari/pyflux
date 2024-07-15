# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 14:46:50 2018

@author: USUARIO
"""

import ctypes
from ctypes import byref
import time
from datetime import datetime
import numpy as np
from unittest.mock import Mock
import logging as _lgn

_lgr = _lgn.getLogger(__name__)

LIB_VERSION = "3.0"
MAXDEVNUM = 8
MODE_T2 = 2
MODE_T3 = 3
TTREADMAX = 131072
# FLAG_OVERFLOW = 0x0040
FLAG_FIFOFULL = 0x0003
DEV_NUM = 0  # device number, works for only 1 PH device

_AVG_COUNTS_S = 15000


class libmock:
    """Mocks a Picoharp library.

    Simulates the response as if a fluorescant bead was moving periodically in
    the field.
    """

    _status = 'IDLE'
    _binning = None
    _offset = 0
    _last_read = time.time_ns()
    _buff = np.zeros((TTREADMAX,), dtype=np.uint32)
    _pos = 0
    _flags = 0

    def __init__(self, PSFs: np.ndarray, nmpp: float,
                 x_speed: float = 4.5, y_speed: float = 4.5,
                 x_amplitude: float = 20., y_amplitude: float = 40.):
        """Initialize a Mock Picoharp.

        Simulates a particle moving around 0.

        Parameters
        ----------
        PSFs : np.ndarray[4, M, M]
            PSFs.
        nmpp : float
            nanometer per pixel in the PSF.
        x_speed : float, optional
            Average speed in the x direction in nm/ms. The default is 4.5.
        y_speed : float, optional
            Average speed in the y direction in nm/ms. The default is 4.5.
        x_amplitude : float, optional
            Amplitude (in nm) of the movement in x. The default is 20.
        y_amplitude : float, optional
            Amplitude (in nm) of the movement in x. The default is 40.
        """
        self._PSFs = PSFs  # Copy?
        self._nmpp = nmpp
        self._x_amp = x_amplitude
        self._y_amp = y_amplitude
        self._x_freq = 0.5 * np.pi * x_speed / x_amplitude
        self._y_freq = 0.5 * np.pi * y_speed / y_amplitude
        # No le damos bola a nada
        # 0.5 s de periodo
        idx_pulses = 

    def PH_GetFlags(self, devidx: ctypes.c_int, flags,) -> int:
        flags.value = self._flags
        return 0

    def PH_StopMeas(self, devidx: ctypes.c_int) -> int:
        self._status = 'IDLE'
        # En realidad no habria que vaciar el buffer
        # self._pos = 0
        self._flags = 0
        return 0

    def PH_StartMeas(self, devidx: ctypes.c_int, tacq: ctypes.c_int):
        self._status = 'MEASURING'
        self._pos = 0
        self._last_read = time.time_ns()
        return 0

    def PH_SetBinning(self, devidx: ctypes.c_int, binning: ctypes.c_int):
        self._binning = binning
        return 0

    def PH_SetSyncOffset(self, devidx: ctypes.c_int, offset: ctypes.c_int):
        self._offset = offset
        return 0

    def PH_ReadFiFo(self, devidx: ctypes.c_int,
                    buffer,
                    count: ctypes.c_int,
                    nactual):
        # meter
        cur_time = time.time_ns()
        n_new_items = (cur_time - self._last_read) * _AVG_COUNTS_S / 1E9
        if n_new_items > TTREADMAX:
            self._status = "FIFOFULL"
            # setear flag
            self._flags = FLAG_FIFOFULL
        # llenar con lo que hay en el buffer
        # si el tiempo y el bufffer de llegada dan para mas, llenar con mas
        # Poner lo que sobra en el buffer y ajustar pos
        available = self._pos + ...
        # buffer._arr[] == ...
        self._last_read = time.time_ns()
        return 0


class PicoHarp300:

    def __init__(self, *args, **kwargs):
        # From phdefin.h
        # Variables to store information read from DLLs
        self.buffer = (ctypes.c_uint * TTREADMAX)()
        self.libVersion = ctypes.create_string_buffer(b"", 8)
        self.hwSerial = ctypes.create_string_buffer(b"", 8)
        self.hwPartno = ctypes.create_string_buffer(b"", 8)
        self.hwVersion = ctypes.create_string_buffer(b"", 8)
        self.hwModel = ctypes.create_string_buffer(b"", 16)
        self.errorString = ctypes.create_string_buffer(b"", 40)
        self.res = ctypes.c_double()
        self.countRate0 = ctypes.c_int()
        self.countRate1 = ctypes.c_int()
        self.flags = ctypes.c_int()
        self.nactual = ctypes.c_int()
        self.ctcDone = ctypes.c_int()
        self.warnings = ctypes.c_int()
        self.warningstext = ctypes.create_string_buffer(b"", 16384)

        # Measurement parameters, these are hardcoded since this is just a demo
        self.mode = MODE_T3  # observe suitable Syncdivider and Range!
        self.binning = 0  # you can change this, meaningful only in T3 mode
        self.offsetValue = 0  # you can change this, meaningful only in T3 mode
        self.tacq = 1000  # Measurement time in millisec, you can change this
        self.syncDiv = 1  # you can change this, observe mode! READ MANUAL!
        self.CFDZeroCross0 = 10  # you can change this (in mV)
        self.CFDLevel0 = 50  # you can change this (in mV)
        self.CFDZeroCross1 = 10  # you can change this (in mV)
        self.CFDLevel1 = 150  # you can change this (in mV)

        self.maxRes = 4  # max res of PicoHarp 300 in ps

        self._status = "UNINITIALIZED"
        self.lib = Mock()
        self.lib.PH_GetFlags()

    def getLibraryVersion(self):
        return "False Picoharp"

    def open(self):
        self._status = "OPEN"

    def getHardwareInfo(self):
        self.lib.PH_GetHardwareInfo(DEV_NUM, self.hwModel, self.hwPartno,
                                    self.hwVersion)
        return ["PHModel", "PHPartNo", "PHHwVers"]

    def setup_ph300(self):
        time.sleep(0.2)

    @property
    def binning(self):
        return self.binningValue

    @binning.setter
    def binning(self, value):
        self.binningValue = value

    @property
    def offset(self):
        return self.offsetValue

    @offset.setter
    def offset(self, value):
        self.offsetValue = value

    @property
    def resolution(self):
        return self.res.value

    @resolution.setter
    def resolution(self, value):
        self.binning = int(np.log(value/self.maxRes)/np.log(2))

    def countrate(self, channel):
        if channel not in (0, 1):
            raise ValueError(f"Invalid channel number: {channel}")
        return np.random.randint(1000)

    @property
    def syncDivider(self):
        return self.syncDiv

    @syncDivider.setter
    def syncDivider(self, value):
        self.syncDiv = value

    def stopTTTR(self):
        ...

    def initialize(self):
        self._status = "INITIALIZED"

    def finalize(self):
        self._status = "CLOSED"
