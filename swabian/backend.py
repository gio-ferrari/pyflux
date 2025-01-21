# -*- coding: utf-8 -*-
"""
TCSPC con tracking.

@author: azelcer
"""

import numpy as np
import atexit as _atexit
import tools.filenames as fntools
import logging as _lgn

from PyQt5.QtCore import pyqtSignal as _pyqtSignal, QObject as _QObject
import TimeTagger as _TimeTagger
import tools.swabiantools as _st
from tools.config_handler import TCSPInstrumentInfo as _TCSPInstrumentInfo
from typing import Union as _Union

from tools import analysis as _analysis
import configparser
from dataclasses import dataclass as _dataclass

from drivers.minflux_measurement import MinfluxMeasurement

_lgr = _lgn.getLogger(__name__)
_lgn.basicConfig(level=_lgn.DEBUG)


_MAX_EVENTS = 131072


@_dataclass
class PSFMetadata:
    """Metadata of PSFs."""

    scan_range: float  # en µm, 'Scan range (µm)'
    n_pixels:  int  # 'Number of pixels'
    px_size: float  # 'Pixel size (µm)'
    scan_type: str  # 'Scan type'


# está en tools
def loadConfig(filename) -> configparser.SectionProxy:
    """Load a config file and return just the parameters."""
    config = configparser.ConfigParser()
    config.read(filename)
    return config['Scanning parameters']


def _config_channels(tagger: _TimeTagger.TimeTagger, IInfo: _TCSPInstrumentInfo):
    """Set delays and filtering according to Info."""
    settings = []
    for APDi in IInfo.APD_info:
        # print(APDi)
        settings.append((APDi.channel, -APDi.delay,))
        _st.set_channel_level(tagger, APDi.channel, _st.SignalTypeEnum[APDi.signal_type])
    _st.set_channels_delay(tagger, settings)
    _st.set_channel_level(tagger, IInfo.laser_channel, _st.SignalTypeEnum[IInfo.laser_signal])
    tagger.setConditionalFilter(
        trigger=[APDi.channel for APDi in IInfo.APD_info],
        filtered=[IInfo.laser_channel]
    )


class __TCSPCBackend(_QObject):
    """Backend class for TCSPC.

    Not to be instantiated by the user.
    """

    _TCSPC_measurement = None
    _measurementGroup = None
    sgnl_measure_init = _pyqtSignal(str)
    sgnl_measure_end = _pyqtSignal()
    sgnl_new_data = _pyqtSignal(object, object, object)
    _NOPLACE = np.full((2,), np.nan)

    def __init__(self,
                 IInfo: _TCSPInstrumentInfo,
                 *args, **kwargs):
        """Receive device info."""
        super().__init__(*args, **kwargs)
        self._tagger = _TimeTagger.createTimeTagger(IInfo.serial)
        self.iinfo = IInfo
        self.period = IInfo.period
        _config_channels(self._tagger, self.iinfo)
        # TODO: Adjust latency
        self._tagger.setStreamBlockSize(max_events=_MAX_EVENTS, max_latency=5)

    def start_measure(self,
                      filename: str,
                      acq_time_s: _Union[float, None] = None, 
                      PSF: _Union[np.ndarray, None] = None,
                      PSF_info: _Union[PSFMetadata, None] = None) -> bool:
        """Start a measurement.

        Returns
        =======
            True if successful, False otherwise
        """
        if self._measurementGroup:
            _lgr.error("A measurement is already running")
            return False
        sorted_indexes = np.argsort(self.iinfo.shutter_delays)
        self._shutter_delays = [self.iinfo.shutter_delays[idx] for idx in sorted_indexes]
        if PSF and PSF_info:
            _lgr.info("Starting measurement with location")
            self._PSF = PSF[sorted_indexes]
            # FIXME
            SBR = 8
            self._locator = _analysis.MinFluxLocator(PSF, SBR, PSF_info.px_size * 1E3)
        else:
            _lgr.info("Starting measurement without location")
            self._locator = None
        # FIXME
        self.fname = filename
        self.currentfname = fntools.getUniqueName(self.fname)
        self._measurementGroup = _TimeTagger.SynchronizedMeasurements(self._tagger)
        self._TCSPC_measurement = MinfluxMeasurement(
            self._measurementGroup.getTagger(),
            self.iinfo.APD_info[0].channel,
            self.iinfo.laser_channel,
            self.iinfo.period,
            _MAX_EVENTS,
            self._shutter_delays,
            self.report,
        )
        self._file_measurement = _TimeTagger.FileWriter(
            self._measurementGroup.getTagger(), self.currentfname + '.ttbin',
            [self.iinfo.laser_channel] + [APDi.channel for APDi in self.iinfo.APD_info])
        self.sgnl_measure_init.emit(self.currentfname)
        if acq_time_s:
            self._measurementGroup.startFor(int(acq_time_s * 1E12))
        else:
            self._measurementGroup.start()

    def report(self, delta_t: np.ndarray, bins: np.ndarray, pos: tuple):
        """Receive data from Swabian driver and report."""
        if self._locator:
            try:
                new_pos = self._locator(bins)[1]
            except Exception as e:
                _lgr.error("Excepción %s reportando data: %s", type(e), e)
                self.stop_measure()
        else:
            new_pos = self._NOPLACE
        self.sgnl_new_data.emit(delta_t, bins, new_pos)

    def stop_measure(self) -> bool:
        """Stop measure.

        Returns
        =======
           True if successful
        """
        if self._measurementGroup:
            self._measurementGroup.stop()
            _lgr.info("Measurement finished")
            self._measurementGroup = None
            self._TCSPC_measurement = None
            self._file_measurement = None
            self.sgnl_measure_end.emit()
            return True
        _lgr.warning("Trying to stop while no measurement is active")
        return False

    def close(self):
        if self._tagger:
            _TimeTagger.freeTimeTagger(self._tagger)
            print("Cerrando tagger")
        self._tagger = None

    def __del__(self):
        self.close()


def __create_backend() -> __TCSPCBackend:
    tt_data = _st.get_tt_info()
    if not tt_data:
        _lgr.error("   ******* Enchufá el equipo *******")
        raise ValueError("POR FAVOR ENCHUFA EL EQUIPO")
    IInfo = None
    try:
        IInfo = _TCSPInstrumentInfo.load()
    except FileNotFoundError:
        _lgr.error("No configuration file found")
        raise
    if not (IInfo.serial in list(tt_data.keys())):
        _lgr.warning(
            "The configuration file is for a time tagger with a "
            "different serial number: will use the first one instead."
        )
        IInfo.serial = list(tt_data.keys())[0]
    _lgr.info(
        "%s TimeTaggers found. Using the one with with S#%s",
        len(tt_data),
        IInfo.serial,
    )
    return __TCSPCBackend(IInfo)


# exported object
TCSPC_backend = __create_backend()
__tagger = TCSPC_backend._tagger


# polémico
_atexit.register(TCSPC_backend.close)
