# -*- coding: utf-8 -*-
"""
TCSPC con tracking

Es un engendro entre cosas PyQt heredadas y cosas GUI agnósticas hecha para simplificar

@author: azelcer
"""

import threading as _th
from collections import deque as _deque
from queue import Queue as _Queue

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

try:
    import drivers.picoharp as picoharp
except ImportError:
    print("Loading mock picoharp")
    import drivers.mock_picoharp as picoharp
import PicoHarp.Read_PTU as Read_PTU
# import drivers.ADwin as ADwin
from tools import analysis as _analysis
from PicoHarp import Read_PTU_fast as _rpf
import configparser
from dataclasses import dataclass as _dataclass

import qdarkstyle
import ctypes


# FIXME: should go to driver
TTREADMAX = 131072
FLAG_OVERFLOW = 0x0040
FLAG_FIFOFULL = 0x0003

_lgr = _lgn.getLogger(__name__)
_lgn.basicConfig(level=_lgn.INFO)


@_dataclass
class PSF_metadata:
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

class Frontend(QtWidgets.QFrame):
    """Frontend para TCSPC con tracking."""

    # Signals
    paramSignal = pyqtSignal(list)
    measureSignal = pyqtSignal(np.ndarray, float)  # basta con esto de los eventos

    # Data
    _localizations = [[]]  # one list per shift
    _shifts = [(0, 0),]
    _PSF = None
    _config = None

    def __init__(self, *args, **kwargs):
        """No hace nada."""
        super().__init__(*args, **kwargs)

        # initial directory
        self.initialDir = r"C:\Data"
        self.setup_gui()

    def start_measurement(self):
        """Inicia la medida."""
        if self._PSF is None:
            print("PSFs needed to start measurements")
            return
        if self._config is None:
            print("config needed to start measurements")
            return
        self.clear_data()
        self.measureButton.setEnabled(False)
        self.measureSignal.emit(self._PSF, float(self._config.px_size)*1E3)

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
                                             initialdir=self.initialDir, filetypes=(("numpy", "*.npy"),),
                                             mode="rb")
            root.destroy()
            if psffile:
                try:
                    self._PSF = np.load(psffile, allow_pickle=False)
                    _lgr.info("Cargadas donas con forma %s", self._PSF.shape)
                    psffile.close()
                except Exception as e:
                    print("error abriendo PSF", e , type(e))
        except OSError:
            pass

    def load_config(self):
        """Elegir archivo de configuración."""
        try:
            root = Tk()
            root.withdraw()
            cfgfile = filedialog.askopenfilename(parent=root, title="Elegir configuración",
                                             initialdir=self.initialDir, filetypes=(("txt", "*.txt"),),
                                             )
            root.destroy()
            if cfgfile:
                try:
                    config = loadConfig(cfgfile)
                except Exception as e:
                    print("error configuración", e , type(e))
        except OSError as e:
            print(e, type(e))
            pass
        metadata = PSF_metadata(float(config['Scan range (µm)']), int(config['Number of pixels']),
                                float(config['Pixel size (µm)']), config['Scan type'])
        if metadata.scan_type != 'xy':
            _lgr.error("Scan invalido")
        _lgr.info("%s", metadata)
        self._config = metadata


    def emit_param(self):
        # TO DO: change for dictionary
        filename = os.path.join(self.folderEdit.text(), self.filenameEdit.text())
        name = filename
        res = int(self.resolutionEdit.text())
        tacq = float(self.acqtimeEdit.text())
        folder = self.folderEdit.text()
        offset = float(self.offsetEdit.text())
        paramlist = [name, res, tacq, folder, offset]
        self.paramSignal.emit(paramlist)
        

    @pyqtSlot(np.ndarray)
    def get_data(self, data: np.ndarray):
        """Receve new data and graph."""
        relTime = np.ndarray((len(data),), dtype=np.uint64)
        absTime = np.ndarray((len(data),), dtype=np.uint64)
        # old_corr = self._oflcorrection
        n_rec, self._oflcorrection = _rpf.record2time(data, relTime, absTime, self._oflcorrection)
        counts, bins = np.histogram(relTime, bins=50) # TO DO: choose proper binning
        self.histPlot.plot(bins[0:-1], counts)
        #        plt.hist(relTime, bins=300)
        #        plt.xlabel('time (ns)')
        #        plt.ylabel('ocurrences')

        # np.append(self._abstime, absTime)
        # counts, time = np.histogram(absTime, bins=50)  # timetrace with 50 bins

        # binwidth = time[-1] / 50
        # timetrace_khz = counts / binwidth

        # self.tracePlot.plot(time[0:-1], timetrace_khz)
        # self.tracePlot.setLabels(bottom=("Time", "ms"), left=("Count rate", "kHz"))

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
        data = np.empty((sum(len(_) for _ in  self._localizations), 2,))
        pos = 0
        try:
            for l, s in zip(self._localizations, self._shifts):
                if len(l):
                    data[pos: pos + len(l)] = np.array(l) + s
                    pos += len(l)
        except:
            print("**********")
            print(len(l))
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
        self._lasttime = 0.0
        self._oflcorrection = np.uint64(0)
        self._abstime = np.zeros((1,), dtype=np.uint64)

    def make_connection(self, backend):
        """Make required connections with backend."""
        backend.sendDataSignal.connect(self.get_data)
        backend.localizationSignal.connect(self.get_localization)
        backend.shiftSignal.connect(self.get_shift)
        backend.measureEndSignal.connect(self.get_measure_end)

    def setup_gui(self):

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
        resolutionLabel = QtWidgets.QLabel("Resolution [ps]")
        self.resolutionEdit = QtWidgets.QLineEdit("16")
        offsetLabel = QtWidgets.QLabel("Offset [ns]")
        self.offsetEdit = QtWidgets.QLineEdit("3")

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

        self.acqtimeEdit.textChanged.connect(self.emit_param)
        self.offsetEdit.textChanged.connect(self.emit_param)
        self.resolutionEdit.textChanged.connect(self.emit_param)

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
        subgrid.addWidget(resolutionLabel, 4, 0)
        subgrid.addWidget(self.resolutionEdit, 4, 1)
        subgrid.addWidget(offsetLabel, 6, 0)
        subgrid.addWidget(self.offsetEdit, 6, 1)
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


class Backend(QtCore.QObject):

    sendDataSignal = pyqtSignal(np.ndarray,)
    tcspcDoneSignal = pyqtSignal()
    measureEndSignal = pyqtSignal()  # to self and front, triggered by working thread
    localizationSignal = pyqtSignal(float, float)  # int, int
    shiftSignal = pyqtSignal(float, float)  # int, int
    _measure_thread: "PicoHarpReaderThread" = None

    def __init__(self, ph_device, *args, **kwargs):
        """Receive picoharp driver device."""
        super().__init__(*args, **kwargs)
        self.ph = ph_device

    def update(self):
        """Placeholder."""
        self.measure_count_rate()

    def prepare_ph(self):
        """Initialize a measurement.

        TODO: Full of hardcoded values.
        """
        self.ph.open()
        self.ph.initialize()
        self.ph.setup_ph300()
        # this parameter must be set such that the count rate at channel 0
        # (sync) is equal or lower than 10MHz
        self.ph.syncDivider = 2
        self.ph.resolution = self.resolution  # desired resolution in ps
        # FIXME: no le hace caso al front en cuanto a resolucion:
        self.ph.lib.PH_SetBinning(
            ctypes.c_int(0), ctypes.c_int(2)
        )  # TO DO: fix this in a clean way (Check Library driver, 0= 4ps, 1 = 8ps, 2 = 16ps resolution)

        # FIXME: no le hace caso al front en cuanto offset
        self.ph.offset = int(self.offset * 1000)  # time in ps
        self.ph.lib.PH_SetSyncOffset(ctypes.c_int(0), ctypes.c_int(3000))

        self.ph.tacq = int(self.tacq * 1000)  # time in ms

        # necessarry sleeping time tcspc needs after ph.initialize()
        # see PicoQuant demo script
        time.sleep(0.2)

        _lgr.info("Resolution = %s ps", self.ph.resolution)
        _lgr.info("Acquisition time = %s ms",  self.ph.tacq)
        _lgr.info("Offset = %s ps", self.ph.offset)
        _lgr.info("TCSPC prepared for TTTR measurement")

    @pyqtSlot(np.ndarray, float)
    def measure(self, PSF: np.ndarray, nmppx: float):
        """Called from GUI."""
        t0 = time.time()
        self.prepare_ph()
        self.currentfname = fntools.getUniqueName(self.fname)

        t1 = time.time()
        _lgr.info("Starting the PH measurement took %s s", (t1 - t0))
        self._measure_thread = PicoHarpReaderThread(self.ph, "Lefilename", None,
                                                    PSF, nmppx, 18., self)
        self._measure_thread.start()

    @pyqtSlot(str, int, int)
    def prepare_minflux(self, fname, acqtime, n):
        """Called from another module (minflux)"""
        print(datetime.now(), "[tcspc] preparing minflux measurement")
        t0 = time.time()
        self.currentfname = fntools.getUniqueName(fname)
        self.prepare_ph()
        self.ph.tacq = acqtime * n * 1000  # TO DO: correspond to GUI !!!
        self.ph.lib.PH_SetBinning(
            ctypes.c_int(0), ctypes.c_int(2)
        )  # TO DO: fix this in a clean way (1 = 8 ps, 2 = 16 ps resolution)
        self.ph.lib.PH_SetSyncOffset(ctypes.c_int(0), ctypes.c_int(3000))
        t1 = time.time()
        print(
            datetime.now(),
            "[tcspc] preparing the PH measurement took {} s".format(t1 - t0),
        )

    @pyqtSlot()
    def measure_minflux(self):
        """Called from another module (minflux)"""
        self.ph.startTTTR(self.currentfname)
        print(datetime.now(), "[tcspc] minflux measurement started")
        while self.ph.measure_state != "done":
            pass
        self.tcspcDoneSignal.emit()
        self.export_data()

    def stop_measure(self):
        """Stop measurement thread."""
        if self._measure_thread:
            self._measure_thread.stop()
            _lgr.debug("Asked to stop measurement")
        else:
            _lgr.warning("Measurement not running")

    def export_data(self):
        """REWRITE"""
        inputfile = open(self.currentfname, "rb")  # TO DO: fix file selection
        print(datetime.now(), "[tcspc] opened {} file".format(self.currentfname))
        numRecords = self.ph.numRecords  # number of records
        globRes = 5e-8  # in ns, corresponds to sync @20 MHz, New NKT white laser
        timeRes = self.ph.resolution * 1e-12  # time resolution in s
        relTime, absTime = Read_PTU.readPT3(inputfile, numRecords)
        inputfile.close()
        print("max and min relTime", np.max(relTime), np.min(relTime))
        relTime = relTime * timeRes  # in real time units (s)
        self.relTime = relTime * 1e9  # in (ns)
        absTime = (
            absTime * globRes * 1e9
        )  # true time in (ns), 4 comes from syncDivider, 10 Mhz = 40 MHz / syncDivider
        self.absTime = absTime / 1e6  # in ms
        filename = self.currentfname + "_arrays.txt"
        datasize = np.size(self.absTime[self.absTime != 0])
        data = np.zeros((2, datasize))
        data[0, :] = self.relTime[self.absTime != 0]
        data[1, :] = self.absTime[self.absTime != 0]
        # self.plotDataSignal.emit(data[0, :], data[1, :])
        np.savetxt(filename, data.T)  # transpose for easier loading
        print(datetime.now(), "[tcspc] tcspc data exported")
        np.savetxt(
            self.currentfname + ".txt", []
        )  # TO DO: build config file, now empty file

    @pyqtSlot(list)
    def get_frontend_parameters(self, paramlist):

        print(datetime.now(), "[tcspc] got frontend parameters")
        self.fname = paramlist[0]
        self.resolution = paramlist[1]
        self.tacq = paramlist[2]
        self.folder = paramlist[3]
        self.offset = paramlist[4]

    def make_connection(self, frontend):

        frontend.paramSignal.connect(self.get_frontend_parameters)
        frontend.measureSignal.connect(self.measure)
        frontend.prepareButton.clicked.connect(self.prepare_ph)
        frontend.stopButton.clicked.connect(self.stop_measure)
        frontend.emit_param()  # TO DO: change such that backend has parameters defined from the start
        self.measureEndSignal.connect(self.cleanup_measurement)

    def stop(self):
        # TODO: call this function
        self.ph.finalize()

    def ack_measurement_end(self):
        """Signal to front (and ourselves) that measure ended."""
        self.measureEndSignal.emit()

    def ack_position(self, x: float, y: float):
        """Signal new localization to front."""
        self.localizationSignal.emit(x, y)

    def ack_shift(self, x: float, y: float):
        """Signal new shift to front."""
        self.shiftSignal.emit(x, y)

    def cleanup_measurement(self):
        self._measure_thread.join(10)
        if self._measure_thread.is_alive():
            _lgr.error("Measure_thread has not ended")
        # np.savetxt(self.currentfname + ".txt", [])
        # self.export_data()

        #     self.ph.lib.PH_ClearHistMem(ctypes.c_int(0),
        #                               ctypes.c_int(0))


class PicoHarpReaderThread(_th.Thread):

    _DEV_NUM = 0  # FIXME: HARDOCDED
    _stop_event: _th.Event = _th.Event()

    def __init__(self, ph: picoharp.PicoHarp300, fname: str, img_evt: _th.Event,
                 PSFs: np.ndarray, nmppx: float, SBR: float, signaler: QtCore.QObject,
                 *args, **kwargs):
        """Prepare.

        Parameters
        ----------
        ph : picoharp.PicoHarp300
            Driver.
        fname : str
            Filename to save to.
        img_evt : _th.Event
            event triggered when a XY image is ready.
        PSFs : np.ndarray
            Array of PSF for locating.
        nmppx: float
            nanometers per pixel on the PSFs
        SBR : float
            Signal to background ratio.
        signaler : QtCore.QObject
            Object to communicate with Qt. Must expose methods to send location and
            end of measurement events.

        Returns
        -------
        None.

        """
        super().__init__(*args, **kwargs)
        self.ph = ph.lib
        self._fname = fname
        self._img_evt = img_evt
        self._PSFs = PSFs
        self._nm_per_pixel = nmppx
        self._SBR = SBR
        self._signaler = signaler
        self._tacq = ph.tacq  # FIXME: esto debería poder ser infinito o venir del front

    def run(self):
        try:
            microbin_duration = 16E-3  # in ns, instrumental, match with front
            macrobin_duration = 12.5  # in ns, match with opticalfibers
            binwidth = int(np.round(macrobin_duration / microbin_duration))  # microbins per macrobin
            locator = _analysis.MinFluxLocator(self._PSFs, 16, self._nm_per_pixel)
            self.startTTTR_Track(self._fname, self._img_evt, locator, binwidth)
        except Exception as e:
            _lgr.error("Exception %s measuring: %s", type(e), e)
        self._signaler.ack_measurement_end()

    def stop(self):
        """External signal to stop measurement."""
        self._stop_event.set()

    def stop_PH_measure(self):
        self.ph.PH_StopMeas(ctypes.c_int(self._DEV_NUM))

    def startTTTR_Track(self, outputfilename: str, img_evt: _th.Event, locator,
                        binwidth: int):
        """Medida TTR lista para tracking.

        Estaría bien guardar los parámetros de medida con la medida, para poder
        procesar después.
        """

        MAXRECS = 512  # always a multiple of 512
        N_BUFFERS = 20  # initial number of buffers
        outputfile = open(outputfilename, "wb+")
        progress = 0

        # save real time for correlating with confocal images for FLIM
        f = open(outputfilename + "_ref_time_tcspc", "w+")
        f.write(str(datetime.now()) + "\n")
        f.write(str(time.time()) + "\n")
        buffers = [np.ndarray((MAXRECS,), np.dtype(ctypes.c_uint32))
                   for _ in range(N_BUFFERS)]
        buffer_q = _deque(buffers)
        data_q = _Queue()
        wt = WriterThread(outputfile, data_q, buffer_q, self._signaler)
        wt.start()
        measuring = True
        n_bins = 256# int(np.ceil(4096 / binwidth))  # (4096+bins-1) / /bins
        _lgr.info("Usando %s bines", n_bins)
        vec = np.zeros((n_bins,), dtype=np.uint64)
        flags = ctypes.c_int()
        ctcDone = ctypes.c_int()
        nactual = ctypes.c_int(0)  # actual number of data in buffer
        self.ph.PH_StartMeas(ctypes.c_int(self._DEV_NUM), ctypes.c_int(self._tacq))
        _lgr.info("Tracking TCSPC measurement started")

        # TODO borrar t0, idx, etc.
        t0 = time.time()
        idx = 0

        buf = buffer_q.pop()
        while measuring:
            self.ph.PH_GetFlags(ctypes.c_int(self._DEV_NUM), ctypes.byref(flags))
            if flags.value & FLAG_FIFOFULL > 0:
                _lgr.error("FiFo Overrun!")
                self.stop_PH_measure()
                measuring = False
                continue
            if nactual.value:
                try:
                    buf = buffer_q.pop()
                except IndexError:
                    _lgr.warning("Not enough buffers, voy a retrasarme un poco...")
                    buf = np.ndarray((MAXRECS,), np.dtype(ctypes.c_uint))
                    buffers.append(buf)
            self.ph.PH_ReadFiFo(
                ctypes.c_int(self._DEV_NUM),
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                ctypes.c_int(MAXRECS),
                ctypes.byref(nactual),
            )
            _lgr.info("records read: %s", nactual.value)
            ttt0 = time.time_ns()
            # ######################## testing
            # if time.time()-t0 > 7E-3:  #cada 7 ms
            #     self._signaler.ack_shift(*np.random.random_sample((2,)) + idx*20)
            #     t0 = time.time()
            #     idx += 1
            #  ###################

            if nactual.value > 0:
                _lgr.info("Current photon count: %s", nactual.value)
                data_q.put_nowait((nactual.value, buf, ))  # report async
                _rpf.all_in_one(buf[:nactual.value], vec, binwidth)
                # _lgr.info(f"{(time.time_ns()-ttt0)/1E6} all in one")
                try:
                    pos = locator(vec)
                    pos = pos[0]
                    # _lgr.info(f"{(time.time_ns()-ttt0)/1E6} ms locator")
                except Exception:
                    _lgr.info("Error en locator con vec= %s (pos=%s)", vec, pos)
                    raise
                self._signaler.ack_position(pos[0], pos[1])
                # if img_evt.is_set():
                #     move
                #     img_evt.clear()
                #     ...
                progress += nactual.value
            if True:  # else:
                self.ph.PH_CTCStatus(ctypes.c_int(self._DEV_NUM), ctypes.byref(ctcDone))
                if ctcDone.value > 0:
                    _lgr.info("TCSPC Done")
                    self.numRecords = progress
                    self.stop_PH_measure()
                    # save real time for correlating with confocal images
                    f.write(str(datetime.now()) + "\n")
                    f.write(str(time.time()) + "\n")
                    f.close()
                    _lgr.info("%s events recorded", self.numRecords)
                    measuring = False
            if self._stop_event.is_set():
                # do not use this as a condition as we need to do cleanup.
                measuring = False
            _lgr.info(f"{(time.time_ns()-ttt0)/1E6} ms / loop")
        data_q.put(None)
        wt.join()


class WriterThread(_th.Thread):
    """Thread que guarda a archivo en background y recicla los buffers."""

    data_queue: _Queue = None
    buffer_queue: _deque = None
    signaler: Backend = None
    fd = None

    def __init__(self, fd, data_q, buffer_q, signaler, *args, **kwargs):
        """Guarda los queues y el file descriptor."""
        super().__init__(*args, **kwargs)
        self.data_queue = data_q
        self.buffer_queue = buffer_q
        self.signaler = signaler
        self.fd = fd

    def run(self):
        """Graba lo recibido y recicla los buffers."""
        total = 0
        _lgr.info("Iniciando thread de escritura")
        nv = self.data_queue.get()
        while nv is not None:
            n_rec, buffer = nv
            self.fd.write(buffer.data[:n_rec])
            self.signaler.sendDataSignal.emit(np.array(buffer.data[:n_rec]))  # copiar porque reusamos
            self.buffer_queue.appendleft(buffer)
            total += n_rec
            nv = self.data_queue.get()
            # _lgr.warning("Grabamos algo")
        _lgr.info("Fin thread de escritura. %s registros escritos.", total)


##################   For dev
# from tools.PSF import doughnut2D
# import tools.PSF_tools as PSF_tools
# import matplotlib.pyplot as plt
# centros = PSF_tools.centers_minflux(100, 4)
# size = 80  # FOV in nm
# nmppx = 1  # in nm
# S = int(np.ceil(size/nmppx))
# d = np.linspace(-size/2, size/2, num=S)
# Mx, My = np.meshgrid(d, d)
# PSFs =np.empty((4, S, S))
# for dona, centro in zip(PSFs, centros):
#     dona[:] = doughnut2D((Mx, My), 1, centro[0], centro[1], 100, 0).reshape(S, S)
#     plt.figure()
#     plt.imshow(dona)
# np.save("donasfalsas", PSFs)

if __name__ == "__main__":

    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication([])
    else:
        app = QtWidgets.QApplication.instance()

    # app.setStyle(QtWidgets.QStyleFactory.create('fusion'))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    ph = picoharp.PicoHarp300()

    worker = Backend(ph)
    gui = Frontend()

    worker.make_connection(gui)
    gui.make_connection(worker)

    gui.setWindowTitle("Time-correlated single-photon counting")
    gui.show()
