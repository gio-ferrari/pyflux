# -*- coding: utf-8 -*-
"""
TCSPC con tracking.

@author: azelcer
"""

import numpy as np
# import tools.filenames as fntools
import logging as _lgn
import pathlib as _plib

from PyQt5.QtCore import pyqtSignal as _pyqtSignal, QObject as _QObject
import TimeTagger as _TimeTagger
import tools.swabiantools as _st
from tools.config_handler import TCSPInstrumentInfo as _TCSPInstrumentInfo
from typing import Union as _Union

from tools import analysis as _analysis
from tools.metaclasses import _Singleton
import configparser
from dataclasses import dataclass as _dataclass

from drivers.minflux_measurement import MinfluxMeasurement
from datetime import datetime

_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


_MAX_EVENTS = 131072


def change_stem(path: _plib.Path, new_stem: str):
    """Trucho.

    Parche para python<3.9
    """
    return path.parent.joinpath(new_stem + ''.join(path.suffixes))


def make_unique_name(filename: str, append_date: bool = False):
    """Ensure that a filename does not exist."""
    base_path = _plib.Path(filename)
    original_stem = base_path.stem
    if append_date:
        now = datetime.now()
        extra = ('_' + now.date().isoformat().replace('-', '') + '-' +
                 now.time().isoformat().replace(':', '').split('.')[0] + '_')
        original_stem = base_path.stem + extra
        # base_path = base_path.with_stem(original_stem)
        base_path = change_stem(base_path, original_stem)
    n = 0
    final_name = _plib.Path(base_path)
    while final_name.exists():
        new_stem = original_stem + f"({n})"
        # final_name = base_path.with_stem(new_stem)
        final_name = change_stem(base_path, new_stem)
        n += 1
    return str(final_name)


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


def _get_instrument_info() -> _TCSPInstrumentInfo:
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
    return IInfo


class TCSPCBackend(metaclass=_Singleton):
    """Backend class for TCSPC.

    To be used as a context manager
    """

    class _Reporter(_QObject):
        sgnl_measure_init = _pyqtSignal(str)
        sgnl_measure_end = _pyqtSignal()
        sgnl_new_data = _pyqtSignal(object, int, object, object)

    _TCSPC_measurement = None
    _measurementGroup = None
    _NOPLACE = np.full((2,), np.nan)
    _reporter = _Reporter()
    sgnl_measure_init = _reporter.sgnl_measure_init
    sgnl_measure_end = _reporter.sgnl_measure_end
    sgnl_new_data = _reporter.sgnl_new_data

    def __init__(self, *args, **kwargs):
        """Prepare device info."""
        super().__init__(*args, **kwargs)
        self._tagger: _TimeTagger.TimeTagger = None
        IInfo = _get_instrument_info()
        self.iinfo = IInfo
        self.open()

    def __enter__(self):
        # self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def open(self):
        if self._tagger:
            _lgr.warning("Doble apertura de Swabian")
            return
        self._tagger = _TimeTagger.createTimeTagger(self.iinfo.serial)
        self.period = self.iinfo.period
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
        self.currentfname = make_unique_name(self.fname, append_date=True)
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
        self._reporter.sgnl_measure_init.emit(self.currentfname)
        if acq_time_s:
            self._measurementGroup.startFor(int(acq_time_s * 1E12))
        else:
            self._measurementGroup.start()

    def report(self, delta_t: np.ndarray, period_length: int, bins: np.ndarray, pos: tuple):
        """Receive data from Swabian driver and report."""
        if self._locator:
            try:
                new_pos = self._locator(bins)[1]
            except Exception as e:
                _lgr.error("Excepción %s reportando data: %s", type(e), e)
                self.stop_measure()
        else:
            new_pos = self._NOPLACE
        self._reporter.sgnl_new_data.emit(delta_t, period_length, bins, new_pos)

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
            self._reporter.sgnl_measure_end.emit()
            return True
        _lgr.warning("Trying to stop while no measurement is active")
        return False

    def close(self):
        if self._tagger:
            _TimeTagger.freeTimeTagger(self._tagger)
            _lgr.info("Cerrando tagger")
        self._tagger = None

    def __del__(self):
        self.close()
