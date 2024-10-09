# -*- coding: utf-8 -*-
"""
TCSPC con tracking.

@author: azelcer
"""


import numpy as np
from datetime import date
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
from tools import analysis as _analysis
import configparser
from dataclasses import dataclass as _dataclass

import qdarkstyle

from drivers.minflux_measurement import MinfluxMeasurement

_lgr = _lgn.getLogger(__name__)
_lgn.basicConfig(level=_lgn.INFO)


_MAX_EVENTS = 131072
_N_BINS = 50
_MAX_SAMPLES = int(60*200)  # si es cada 5 ms son 200 por segundo


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


def _config_channels(tagger: _TimeTagger.TimeTagger, IInfo: TCSPInstrumentInfo):
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


class TCSPCFrontend(QtWidgets.QFrame):
    """Frontend para TCSPC con tracking."""

    # Signals
    measureSignal = pyqtSignal(np.ndarray, np.ndarray, tuple)

    # Data
    _localizations_x = np.full((_MAX_SAMPLES,), 0)
    _localizations_y = np.full((_MAX_SAMPLES,), 0)
    _shifts = [(0, 0),]
    _intensities = np.zeros((4, _MAX_SAMPLES,), dtype=np.int32)
    _PSF = None
    _config: PSFMetadata = None
    _pos_vline: pg.InfiniteLine = None
    _measure: "TCSPCBackend" = None

    def __init__(self, tagger: _TimeTagger.TimeTagger, IInfo: TCSPInstrumentInfo, *args, **kwargs):
        """No hace nada."""
        super().__init__(*args, **kwargs)

        # initial directory
        self.tagger = tagger
        self.initialDir = r"C:\Data"
        self.iinfo = IInfo
        # FIXME: for developing only
        self.period = self.iinfo.period
        self._init_data()
        self.setup_gui()

    def _init_data(self):
        self._hist_data = list(np.histogram([], range=(0, self.period), bins=_N_BINS))
        self._intensities = np.zeros((4, _MAX_SAMPLES,), dtype=np.int32)
        self._last_pos = 0
        self._localizations_x = np.full((_MAX_SAMPLES,), 0)
        self._localizations_y = np.full((_MAX_SAMPLES,), 0)
        self._shifts = [(0, 0),]

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
        self._measure = TCSPCBackend(self.tagger, self.iinfo, self.callback, self._PSF,
                                     self._config)
        self._measure.start_measure()

    def stop_measurement(self):
        """Para la medida.

        Sin error checking por hora
        """
        self.measureButton.setEnabled(True)
        if not self._measure:
            print("Parece no haber measure")
            return
        self._measure.stop_measure()

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
        metadata = PSFMetadata(
            float(config['Scan range (µm)']),
            int(config['Number of pixels']),
            float(config['Pixel size (µm)']),
            config['Scan type']
        )
        if metadata.scan_type != 'xy':
            _lgr.error("Scan invalido")
        _lgr.info("%s", metadata)
        self._config = metadata

    def callback(self, delta_t: np.ndarray, binned: np.array, newpos: np.array):
        self.measureSignal.emit(delta_t, binned, newpos)

    @pyqtSlot(np.ndarray, np.ndarray, np.array)
    def get_data(self, delta_t: np.ndarray, binned: np.array, new_pos: np.array):
        """Receive new data and graph."""
        try:
            counts, bins = np.histogram(delta_t, range=(0, self.period), bins=_N_BINS)
            self._hist_data[0] += counts
            # self.histPlot.setData(bins[0:-1], self._hist_data[0])
            self._intensities[:, self._last_pos] = binned
            # print(self._last_pos)
            must_update = (self._last_pos % 100 == 0)
            if must_update:
                for plot, data in zip(self.intplots, self._intensities):
                    plot.setData(data)  #, connect="finite")
                self.trace_vline.setValue(self._last_pos)
                self.histPlot.setData(bins[0:-1], counts)
            self.add_localization(*new_pos, self._last_pos, must_update)
            print(new_pos)

            self._last_pos += 1
            if self._last_pos >= _MAX_SAMPLES:
                self._last_pos = 0
        except Exception as e:
            print(e, type(e))

    def add_localization(self, pos_x, pos_y, last_pos, update: bool):
        """Receive a new localization from backend."""
        self._localizations_x[last_pos] = pos_x  # + self._shifts[-1][0]
        self._localizations_y[last_pos] = pos_y  # + self._shifts[-1][1]
        if update:
            self.posPlot.setData(self._localizations_x, self._localizations_y)

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
        for p in self.intplots:
            p.clear()
        # self.tracePlot.addItem(self.trace_vline)

        self.posPlot.clear()
        self._init_data()

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

        # self.exportDataButton.clicked.connect(
        #     lambda: self.get_data(
        #         np.random.randint(0, 49999, 100),
        #         np.random.randint(0, 100, size=4), np.random.random(2)))

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
        self.histWidg = self.dataWidget.addPlot(
            row=0, col=0, title="microTime histogram", stepMode=True, fillLevel=0,
        )
        self.histWidg.setLabels(bottom=("ps"), left=("counts"))
        self.histPlot  = self.histWidg.plot()

        self.tracePlot = self.dataWidget.addPlot(row=1, col=0, title="Time trace")
        self.tracePlot.setLabels(bottom=("s"), left=("counts"))
        self.intplots: list[pg.PlotDataItem] = [
            self.tracePlot.plot(pen=_) for _ in range(4)
        ]
        self.trace_vline = pg.InfiniteLine(0)
        self.tracePlot.addItem(self.trace_vline)

        self.posPlotItem = self.dataWidget.addPlot(row=0, col=1, rowspan=2, title="Position")
        self.posPlotItem.showGrid(x=True, y=True)
        self.posPlotItem.setLabels(
            bottom=("X position", "nm"), left=("Y position", "nm")
        )
        self.posPlot: pg.PlotDataItem = self.posPlotItem.plot(
            [], [], pen=None, symbolBrush=(255, 0, 0), symbolSize=5, symbolPen=None
            )
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
        self.stopButton.clicked.connect(self.stop_measurement)
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

        subgrid.addWidget(self.exportDataButton, 16, 1)

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

    def __init__(self, tagger, IInfo: TCSPInstrumentInfo, cb, PSF: np.ndarray,
                 PSF_info: PSFMetadata, *args, **kwargs):
        """Receive device and config info."""
        super().__init__(*args, **kwargs)
        self.tagger = tagger
        self.iinfo = IInfo
        self._cb = cb
        # FIXME
        SBR = 8
        sorted_indexes = np.argsort(self.iinfo.shutter_delays)
        self._PSF = PSF[sorted_indexes]
        self._shutter_delays = [self.iinfo.shutter_delays[idx] for idx in sorted_indexes]
        self._locator = _analysis.MinFluxLocator(PSF, SBR, PSF_info.px_size * 1E3)

    def start_measure(self):
        """Called from GUI."""
        self.fname = "Fantasia_filename"
        self.currentfname = fntools.getUniqueName(self.fname)
        _config_channels(tagger, self.iinfo)
        # TODO: Adjust latency
        tagger.setStreamBlockSize(max_events=_MAX_EVENTS, max_latency=5)
        self.measurementGroup = _TimeTagger.SynchronizedMeasurements(tagger)
        self.TCSPC_measurement = MinfluxMeasurement(
            self.measurementGroup.getTagger(),
            self.iinfo.APD_info[0].channel,
            self.iinfo.laser_channel,
            self.iinfo.period,
            _MAX_EVENTS,
            self._shutter_delays,
            self.report,
        )
        self.file_measurement = _TimeTagger.FileWriter(
            self.measurementGroup.getTagger(), 'filename.ttbin',
            [self.iinfo.laser_channel] + [APDi.channel for APDi in self.iinfo.APD_info])
        self.measurementGroup.start()
        # Lo hacemos así por ahora
        # self.measurementGroup.startFor(int(15E12))

    def report(self, delta_t: np.ndarray, bins: np.ndarray, pos: tuple):
        try:
            new_pos = self._locator(bins)[1]
            self._cb(delta_t, bins, new_pos)
        except Exception as e:
            _lgr.error("Excepción %s reportando data: %s", type(e), e)
            self.stop_measure()

    def stop_measure(self):
        """Called from GUI."""
        if self.measurementGroup:
            self.measurementGroup.stop()
            print("Measurement finished")
        else:
            print("No measurement running")


if __name__ == "__main__":
    tt_data = _st.get_tt_info()
    if not tt_data:
        print("   ******* Enchufá el equipo *******")
        raise ValueError("POR FAVOR ENCHUFA EL EQUIPO")
    IInfo = None
    try:
        IInfo = TCSPInstrumentInfo.load()
        serial = IInfo.serial
    except FileNotFoundError:
        _lgr.info("No configuration file found")
        # FIXME: for testing
        # tt_data = {'aaaaa': {}}
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
    # FIXME: testing
    # tt_info = TCSPInstrumentInfo("serial", 0, 'NIM', int(50E3),
    #                              np.arange(0, 50000, 12500))
    with _TimeTagger.createTimeTagger() as tagger:
    # tagger = None
    # if True:
        if not QtWidgets.QApplication.instance():
            app = QtWidgets.QApplication([])
        else:
            app = QtWidgets.QApplication.instance()

        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        gui = TCSPCFrontend(tagger, IInfo)

        gui.setWindowTitle("Time-correlated single-photon counting with tracking")
        gui.show()
        gui.raise_()
        gui.activateWindow()
        app.exec_()
