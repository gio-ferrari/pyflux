# -*- coding: utf-8 -*-
"""
TCSPC con tracking.

@author: azelcer
"""


import numpy as np
import time
from datetime import date, datetime
import os
import tools.filenames as fntools
from tkinter import Tk, filedialog
import logging as _lgn

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QGroupBox
from PyQt5 import QtWidgets
import TimeTagger as _TimeTagger
import tools.swabiantools as _st
from tools.config_handler import TCSPInstrumentInfo


# import drivers.ADwin as ADwin
# from tools import analysis as _analysis
import configparser
from dataclasses import dataclass as _dataclass

import qdarkstyle

from drivers.minflux_measurement import MinfluxMeasurement

_lgr = _lgn.getLogger(__name__)
_lgn.basicConfig(level=_lgn.INFO)


_MAX_EVENTS = 131072
_N_BINS = 50


@_dataclass
class PSF_metadata:
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


def _config_channels(tagger: _TimeTagger.TimeTagger, IInfo: TCSPInstrumentInfo):
    """Set delays and filtering according to Info."""
    settings = []
    for APDi in IInfo.APD_info:
        print(APDi)
        settings.append((APDi.channel, -APDi.delay,))
        _st.set_channel_level(tagger, APDi.channel, _st.SignalTypeEnum[APDi.signal_type])
    _st.set_channels_delay(tagger, settings)
    _st.set_channel_level(IInfo.laser_channel, _st.SignalTypeEnum[IInfo.laser_signal])
    tagger.setConditionalFilter(
        trigger=[APDi.channel for APDi in IInfo.APD_info],
        filtered=[IInfo.laser_channel]
    )


class TCSPCFrontend(QtWidgets.QFrame):
    """Frontend para TCSPC con tracking."""

    # Signals
    paramSignal = pyqtSignal(list)
    measureSignal = pyqtSignal(np.ndarray, float)  # basta con esto de los eventos

    # Data
    _localizations = [[]]  # one list per shift
    _shifts = [(0, 0),]
    _PSF = None
    _config = None
    _measure: "TCSPCBackend" = None

    def __init__(self, IInfo: TCSPInstrumentInfo, *args, **kwargs):
        """No hace nada."""
        super().__init__(*args, **kwargs)

        # initial directory
        self.initialDir = r"C:\Data"
        self.iinfo = IInfo
        self.setup_gui()

        # FIXME: for developing only
        # self.period = self.iinfo.period
        self.period = int(50E3)
        self._init_data()
        # sorted_shutters = np.argsort(self.iinfo.shutter_delays)

    def _init_data(self):
        self._hist_data = np.histogram([], range=(0, self.period), bins=_N_BINS)

    def start_measurement(self):
        """Inicia la medida."""
        if self._PSF is None:
            print("PSFs needed to start measurements")
            return
        if self._config is None:
            print("config needed to start measurements")
            return
        self.clear_data()
        self._init_data()
        self.measureButton.setEnabled(False)
        self._measure = TCSPCBackend(self.tagger, self.iinfo, self.get_data)
        # TODO: sort PSFs
        self._measure.start_measure(self._PSF, self._config.px_size)

    def load_folder(self):
        """Muestra una ventana de selección de carpeta."""
        try:
            root = Tk()
            root.withdraw()
            folder = filedialog.askdirectory(parent=root, initialdir=self.initialDir)
            root.destroy()
            if folder:
                self.folderEdit.setText(folder)
        except OSError:
            pass

    def load_PSF(self):
        """Elegir archivo NPZ (no tiff)."""
        try:
            root = Tk()
            root.withdraw()
            psffile = filedialog.askopenfile(parent=root, title="Elegir PSF",
                                             initialdir=self.initialDir,
                                             filetypes=(("numpy", "*.npy"),),
                                             mode="rb")
            root.destroy()
            if psffile:
                try:
                    self._PSF = np.load(psffile, allow_pickle=False)
                    _lgr.info("Cargadas donas con forma %s", self._PSF.shape)
                    psffile.close()
                except Exception as e:
                    print("error abriendo PSF", e, type(e))
        except OSError:
            pass

    def load_config(self):
        """Elegir archivo de configuración."""
        try:
            root = Tk()
            root.withdraw()
            cfgfile = filedialog.askopenfilename(
                parent=root, title="Elegir configuración",
                initialdir=self.initialDir, filetypes=(("txt", "*.txt"),),
            )
            root.destroy()
            if cfgfile:
                try:
                    config = loadConfig(cfgfile)
                except Exception as e:
                    print("error configuración", e, type(e))
        except OSError as e:
            _lgr.error("Error '%s' abriendo archivo de configuracion: %s",
                       type(e), e)
            return
        metadata = PSF_metadata(
            float(config['Scan range (µm)']),
            int(config['Number of pixels']),
            float(config['Pixel size (µm)']),
            config['Scan type']
        )
        if metadata.scan_type != 'xy':
            _lgr.error("Scan invalido")
        _lgr.info("%s", metadata)
        self._config = metadata

    @pyqtSlot(np.ndarray)
    def get_data(self, delta_t: np.ndarray, binned: np.array):
        """Recieve new data and graph."""
        counts, bins = np.histogram(delta_t, range=(0, self.period), bins=50)  # TODO: choose proper binning
        self.histPlot.plot(bins[0:-1], counts)

    @pyqtSlot(float, float)
    def get_localization(self, pos_x, pos_y):
        """Receive a new localization from backend."""
        self._localizations[-1].append((pos_x, pos_y))
        # data = np.array(list(zip(*self._localizations[-1])))
        data = np.array(sum(self._localizations, []))
        # data = np.array(self._localizations)
        # shifts = np.array(self._shifts)
        # locs = data + shifts[np.newaxis, :]
        # TODO: usar numpy
        if len(self._localizations) != len(self._shifts):
            _lgr.error("El largo de los shifts y localizaciones no coincide")
        data = np.empty((sum(len(_) for _ in self._localizations), 2,))
        pos = 0
        try:
            for loc, s in zip(self._localizations, self._shifts):
                if len(loc):
                    data[pos: pos + len(loc)] = np.array(loc) + s
                    pos += len(loc)
        except Exception as e:
            print("**********")
            print(type(e), e, len(loc))
            print("===========")
            raise
        # data = np.vstack(self._localizations)
        self.posPlot.setData(data)

    @pyqtSlot(float, float)
    def get_shift(self, shift_x, shift_y):
        """Receive a shift signal from backend."""
        self._shifts.append((shift_x, shift_y,))
        self._localizations.append([])

    @pyqtSlot()
    def get_measure_end(self):
        """Receive a measure end signal from backend."""
        self.measureButton.setEnabled(True)

    def clear_data(self):
        """Clear all data and plots."""
        self.histPlot.clear()
        self.tracePlot.clear()
        self.posPlot.clear()
        self._localizations = [[]]
        self._shifts = [(0, 0),]

    def setup_gui(self):
        """Initialize the GUI."""
        # widget with tcspc parameters
        self.paramWidget = QGroupBox("TCSPC parameter")
        self.paramWidget.setFixedHeight(250)
        self.paramWidget.setFixedWidth(250)

        phParamTitle = QtWidgets.QLabel("<h2>TCSPC settings</h2>")
        phParamTitle.setTextFormat(QtCore.Qt.RichText)

        # widget to display data
        self.dataWidget = pg.GraphicsLayoutWidget()

        # file/folder widget
        self.fileWidget = QGroupBox("Save options")
        self.fileWidget.setFixedHeight(180)
        self.fileWidget.setFixedWidth(250)
        # Buttons
        self.prepareButton = QtWidgets.QPushButton("Prepare TTTR")
        self.measureButton = QtWidgets.QPushButton("Measure TTTR")
        self.stopButton = QtWidgets.QPushButton("Stop")
        self.exportDataButton = QtWidgets.QPushButton("Export data")
        self.clearButton = QtWidgets.QPushButton("Clear data")
        # TCSPC parameters labels and edits
        acqtimeLabel = QtWidgets.QLabel("Acquisition time [s]")
        self.acqtimeEdit = QtWidgets.QLineEdit("1")

        # self.channel0Label = QtWidgets.QLabel("Input 0 (sync) [kHz]")
        # self.channel0Value = QtWidgets.QLineEdit("")
        # self.channel0Value.setReadOnly(True)

        # self.channel1Label = QtWidgets.QLabel("Input 1 (APD) [kHz]")
        # self.channel1Value = QtWidgets.QLineEdit("")
        # self.channel1Value.setReadOnly(True)

        self.filenameEdit = QtWidgets.QLineEdit("filename")

        # microTime histogram and timetrace
        self.histPlot = self.dataWidget.addPlot(
            row=0, col=0, title="microTime histogram"
        )
        self.histPlot.setLabels(bottom=("ns"), left=("counts"))

        self.tracePlot = self.dataWidget.addPlot(row=1, col=0, title="Time trace")
        self.tracePlot.setLabels(bottom=("s"), left=("counts"))
        self.intplots: list[pg.PlotDataItem] = [
            self.tracePlot.plot(pen=_) for _ in range(4)]

        self.posPlotItem = self.dataWidget.addPlot(row=0, col=1, rowspan=2, title="Position")
        self.posPlotItem.showGrid(x=True, y=True)
        self.posPlotItem.setLabels(
            bottom=("X position", "nm"), left=("Y position", "nm")
        )
        self.posPlot = self.posPlotItem.plot([], pen=None,
                                             symbolBrush=(255, 0, 0),
                                             symbolSize=5, symbolPen=None)
        # folder
        # TO DO: move this to backend
        today = str(date.today()).replace("-", "")
        root = "C:\\Data\\"
        folder = root + today
        try:
            os.mkdir(folder)
        except OSError:
            _lgr.info("Directory %s already exists", folder)
        else:
            _lgr.info("Directory %s already exists", folder)

        self.folderLabel = QtWidgets.QLabel("Folder:")
        self.folderEdit = QtWidgets.QLineEdit(folder)
        self.browseFolderButton = QtWidgets.QPushButton("Browse")
        self.browseFolderButton.setCheckable(True)
        self.browsePSFButton = QtWidgets.QPushButton("PSF")
        self.browsePSFButton.setCheckable(True)
        self.browseConfigButton = QtWidgets.QPushButton("Config")
        self.browseConfigButton.setCheckable(True)

        # GUI connections
        self.measureButton.clicked.connect(self.start_measurement)
        self.browseFolderButton.clicked.connect(self.load_folder)
        self.browsePSFButton.clicked.connect(self.load_PSF)
        self.browseConfigButton.clicked.connect(self.load_config)
        self.clearButton.clicked.connect(self.clear_data)

        # self.acqtimeEdit.textChanged.connect(self.emit_param)

        # general GUI layout
        grid = QtWidgets.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.paramWidget, 0, 0)
        grid.addWidget(self.fileWidget, 1, 0)
        grid.addWidget(self.dataWidget, 0, 1, 2, 2)

        # param Widget layout
        subgrid = QtWidgets.QGridLayout()
        self.paramWidget.setLayout(subgrid)
        # subgrid.addWidget(phParamTitle, 0, 0, 2, 3)
        subgrid.addWidget(acqtimeLabel, 2, 0)
        subgrid.addWidget(self.acqtimeEdit, 2, 1)
        # subgrid.addWidget(self.channel0Label, 8, 0)
        # subgrid.addWidget(self.channel0Value, 8, 1)
        # subgrid.addWidget(self.channel1Label, 9, 0)
        # subgrid.addWidget(self.channel1Value, 9, 1)

        subgrid.addWidget(self.measureButton, 17, 0)
        subgrid.addWidget(self.prepareButton, 18, 0)
        subgrid.addWidget(self.stopButton, 17, 1)
        subgrid.addWidget(self.clearButton, 18, 1)

        file_subgrid = QtWidgets.QGridLayout()
        self.fileWidget.setLayout(file_subgrid)

        file_subgrid.addWidget(self.filenameEdit, 0, 0, 1, 2)
        file_subgrid.addWidget(self.folderLabel, 1, 0, 1, 2)
        file_subgrid.addWidget(self.folderEdit, 2, 0, 1, 2)
        file_subgrid.addWidget(self.browseFolderButton, 3, 0)
        file_subgrid.addWidget(self.browsePSFButton, 4, 0)
        file_subgrid.addWidget(self.browseConfigButton, 4, 1)

    def closeEvent(self, *args, **kwargs):

        # workerThread.exit()
        super().closeEvent(*args, **kwargs)
        # app.quit()


class TCSPCBackend:
    """Backend for TCSPC."""

    TCSPC_measurement = None
    measurementGroup = None

    def __init__(self, tagger, IInfo: TCSPInstrumentInfo, cb, *args, **kwargs):
        """Receive device and config info."""
        super().__init__(*args, **kwargs)
        self.tagger = tagger
        self.iinfo = IInfo
        self._cb = cb
        # self._PSF = self._PSF[np.argsort(self.iinfo.shutter_delays)]

    def start_measure(self, PSF: np.ndarray, nmppx: float):
        """Called from GUI."""
        self.fname = "Fantasia_filename"
        self.currentfname = fntools.getUniqueName(self.fname)
        _config_channels(tagger, self.iinfo)
        # TODO: Adjust latency
        tagger.setStreamBlockSize(max_events=_MAX_EVENTS, max_latency=20)
        self.measurementGroup = _TimeTagger.SynchronizedMeasurements(tagger)
        self.TCSPC_measurement = MinfluxMeasurement(
            self.measurementGroup.getTagger(),
            self.iinfo.APD_info[0].channel,
            self.iinfo.laser_channel,
            self.iinfo.period,
            _MAX_EVENTS,
            self.iinfo.shutter_delays,
            self.report_bins,
        )
        self.file_measurement = _TimeTagger.FileWriter(
            self.measurementGroup.getTagger(), 'filename.ttbin',
            [self.iinfo.laser_channel] + [APDi.channel for APDi in self.iinfo.APD_info])
        # self.measurementGroup.start()
        # Lo hacemos así por ahora
        self.measurementGroup.startFor(int(.5E12))

    def report_bins(self, delta_t: np.ndarray, bins: np.ndarray):
        try:
            self._cb(delta_t, bins)
        except Exception as e:
            _lgr.error("Excepción %s reportando data: %s", type(e), e)
            self.stop_measure()

    def stop_measure(self):
        """Called from GUI."""
        if self.measurementGroup:
            self.measurementGroup.stop()
            print("Measrumement finished")
        else:
            print("no measurement running")


if __name__ == "__main__":
    tt_data = _st.get_tt_info()
    if not tt_data:
        print("   ******* Enchufá el equipo *******")
        # raise ValueError("POR FAVOR ENCHUFA EL EQUIPO")
    IInfo = None
    try:
        IInfo = TCSPInstrumentInfo.load()
        serial = IInfo.serial
    except FileNotFoundError:
        _lgr.info("No configuration file found")
        # FIXME: for testing
        tt_data = {'aaaaa': {}}
        serial = list(tt_data.keys())[0]
    if not (serial in list(tt_data.keys())):
        _lgr.warning(
            "The configuration file is for a time tagger with a "
            "different serial number: will use the first one instead."
        )
        serial = list(tt_data.keys())[0]
    _lgr.info(
        "%s TimeTaggers found. Using the one with with S#%s",
        len(tt_data),
        serial,
    )
    tt_info = tt_data[serial]
    # with _TimeTagger.createTimeTagger() as tagger:
    tagger = None
    if True:
        if not QtWidgets.QApplication.instance():
            app = QtWidgets.QApplication([])
        else:
            app = QtWidgets.QApplication.instance()

        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        gui = TCSPCFrontend(IInfo)

        gui.setWindowTitle("Time-correlated single-photon counting with tracking")
        gui.show()
        gui.raise_()
        gui.activateWindow()
