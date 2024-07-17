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
# import tifffile as tiff
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


import qdarkstyle
import ctypes

# FIXME: should go to driver
TTREADMAX = 131072
FLAG_OVERFLOW = 0x0040
FLAG_FIFOFULL = 0x0003

_lgr = _lgn.getLogger(__name__)
_lgn.basicConfig(level=_lgn.INFO)


class Frontend(QtWidgets.QFrame):
    """Frontend para TCSPC con tracking."""

    # Signals
    paramSignal = pyqtSignal(list)
    measureSignal = pyqtSignal()

    def __init__(self, *args, **kwargs):
        """No hace nada."""
        super().__init__(*args, **kwargs)

        # initial directory
        self.initialDir = r"C:\Data"
        self.setup_gui()

    def start_measurement(self):
        """Inicia la medida."""
        self.measureButton.setEnabled(False)
        self.measureSignal.emit()

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

    @pyqtSlot(float, float)
    def get_backend_parameters(self, cts0, cts1):

        # conversion to kHz
        cts0_khz = cts0 / 1000
        cts1_khz = cts1 / 1000
        self.channel0Value.setText(("{}".format(cts0_khz)))
        self.channel1Value.setText(("{}".format(cts1_khz)))

    @pyqtSlot(np.ndarray, np.ndarray)
    def plot_data(self, relTime, absTime):

        self.clear_data()

        counts, bins = np.histogram(relTime, bins=50)  # TO DO: choose proper binning
        self.histPlot.plot(bins[0:-1], counts)
        #        plt.hist(relTime, bins=300)
        #        plt.xlabel('time (ns)')
        #        plt.ylabel('ocurrences')

        counts, time = np.histogram(absTime, bins=50)  # timetrace with 50 bins

        binwidth = time[-1] / 50
        timetrace_khz = counts / binwidth

        self.tracePlot.plot(time[0:-1], timetrace_khz)
        #        plt.plot(time[0:-1], timetrace)
        self.tracePlot.setLabels(bottom=("Time", "ms"), left=("Count rate", "kHz"))

        self.measureButton.setEnabled(True)

    def clear_data(self):

        self.histPlot.clear()
        self.tracePlot.clear()

    def make_connection(self, backend):

        backend.ctRatesSignal.connect(self.get_backend_parameters)
        backend.plotDataSignal.connect(self.plot_data)

    def setup_gui(self):

        # widget with tcspc parameters
        self.paramWidget = QGroupBox("TCSPC parameter")
        self.paramWidget.setFixedHeight(230)
        self.paramWidget.setFixedWidth(230)

        phParamTitle = QtWidgets.QLabel("<h2>TCSPC settings</h2>")
        phParamTitle.setTextFormat(QtCore.Qt.RichText)

        # widget to display data
        self.dataWidget = pg.GraphicsLayoutWidget()

        # file/folder widget
        self.fileWidget = QGroupBox("Save options")
        self.fileWidget.setFixedHeight(130)
        self.fileWidget.setFixedWidth(230)
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

        self.channel0Label = QtWidgets.QLabel("Input 0 (sync) [kHz]")
        self.channel0Value = QtWidgets.QLineEdit("")
        self.channel0Value.setReadOnly(True)

        self.channel1Label = QtWidgets.QLabel("Input 1 (APD) [kHz]")
        self.channel1Value = QtWidgets.QLineEdit("")
        self.channel1Value.setReadOnly(True)

        self.filenameEdit = QtWidgets.QLineEdit("filename")

        # microTime histogram and timetrace
        self.histPlot = self.dataWidget.addPlot(
            row=0, col=0, title="microTime histogram"
        )
        self.histPlot.setLabels(bottom=("ns"), left=("counts"))

        self.tracePlot = self.dataWidget.addPlot(row=1, col=0, title="Time trace")
        self.tracePlot.setLabels(bottom=("s"), left=("counts"))

        self.posPlot = self.dataWidget.addPlot(row=0, col=1, rowspan=2, title="Position")
        self.posPlot.showGrid(x=True, y=True)
        self.posPlot.setLabels(
            bottom=("X position", "nm"), left=("Y position", "nm")
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

        # GUI connections
        self.measureButton.clicked.connect(self.start_measurement)
        self.browseFolderButton.clicked.connect(self.load_folder)
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
        subgrid.addWidget(self.channel0Label, 8, 0)
        subgrid.addWidget(self.channel0Value, 8, 1)
        subgrid.addWidget(self.channel1Label, 9, 0)
        subgrid.addWidget(self.channel1Value, 9, 1)

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

    def closeEvent(self, *args, **kwargs):

        # workerThread.exit()
        super().closeEvent(*args, **kwargs)
        # app.quit()


class Backend(QtCore.QObject):

    ctRatesSignal = pyqtSignal(float, float)
    plotDataSignal = pyqtSignal(np.ndarray, np.ndarray)
    tcspcDoneSignal = pyqtSignal()
    measureEndSignal = pyqtSignal()  # to self and front, triggered by working thread
    localizationSignal = pyqtSignal(float, float)  # int, int
    _measure_thread: "PicoHarpReaderThread" = None

    def __init__(self, ph_device, *args, **kwargs):
        """Receive picoharp driver device."""
        super().__init__(*args, **kwargs)
        self.ph = ph_device

    def update(self):
        """Placeholder."""
        self.measure_count_rate()

    def measure_count_rate(self):
        """Sned countrate to front."""
        self.cts0 = self.ph.countrate(0)
        self.cts1 = self.ph.countrate(1)
        self.ctRatesSignal.emit(self.cts0, self.cts1)

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

    @pyqtSlot()
    def measure(self):
        """Called from GUI."""
        t0 = time.time()
        self.prepare_ph()
        self.currentfname = fntools.getUniqueName(self.fname)

        t1 = time.time()
        _lgr.info("Starting the PH measurement took %s s", (t1 - t0))
        self._measure_thread = PicoHarpReaderThread(self.ph, "Lefilename", None,
                                                    np.ones((4, 80, 80,)), 18., self)
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
            _lgr.debug("Asked to stop measure")
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
        self.plotDataSignal.emit(data[0, :], data[1, :])
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
                 PSFs: np.ndarray, SBR: float, signaler: QtCore.QObject,
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
        self._SBR = SBR
        self._signaler = signaler
        self._tacq = 1000  # FIXME: esto debería poder ser infinito o venir del front

    def run(self):
        try:
            microbin_duration = 16E-3  # in ns, instrumental, match with front
            macrobin_duration = 12.5  # in ns, match with opticalfibers
            binwidth = int(np.round(macrobin_duration / microbin_duration))  # microbins per macrobin
            self.startTTTR_Track(self._fname, self._img_evt, self._PSFs, self._SBR, binwidth)
        except Exception as e:
            _lgr.error("Exception %s measuring: %s", type(e), e)
        self._signaler.ack_measurement_end()

    def stop(self):
        """External signal to stop measurement."""
        self._stop_event.set()

    def stop_PH_measure(self):
        self.ph.PH_StopMeas(ctypes.c_int(self._DEV_NUM))

    def startTTTR_Track(self, outputfilename: str, img_evt: _th.Event, PSF, SBR,
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
        wt = WriterThread(outputfile, data_q, buffer_q)
        wt.start()
        locator = _analysis.MinFluxLocator(PSF, SBR)
        measuring = True
        self.ph.PH_StartMeas(ctypes.c_int(self._DEV_NUM), ctypes.c_int(self._tacq))
        _lgr.info("Tracking TCSPC measurement started")
        n_bins = int(np.ceil(4096 / binwidth))  # (4096+bins-1) / /bins
        vec = np.zeros((n_bins,), dtype=np.uint64)
        flags = ctypes.c_int()
        ctcDone = ctypes.c_int()
        nactual = ctypes.c_int()  # actual number of data in buffer

        while measuring:
            self.ph.PH_GetFlags(ctypes.c_int(self._DEV_NUM), ctypes.byref(flags))
            if flags.value & FLAG_FIFOFULL > 0:
                _lgr.error("FiFo Overrun!")
                self.stop_PH_measure()
                measuring = False
                continue
            try:
                buf = buffer_q.pop()
                print("got buffer")
            except IndexError:
                _lgr.warning("Not enough buffers, voy a retrasarme un poco...")
                buf = np.ndarray((MAXRECS,), np.dtype(ctypes.c_uint))
                buffers.append(buf)
                measuring = False
            self.ph.PH_ReadFiFo(
                ctypes.c_int(self._DEV_NUM),
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                ctypes.c_int(MAXRECS),
                ctypes.byref(nactual),
            )
            _lgr.info("records read: %s", nactual.value)

            if nactual.value > 0:
                _lgr.info("Current photon count: %s", nactual.value)
                data_q.put_nowait((nactual.value, buf, ))  # report async
                if True:  # img_evt.is_set():
                    _rpf.all_in_one(buf[:nactual.value], vec, binwidth)
                    pos = locator(vec)[0]
                    print(pos)
                    self._signaler.ack_position(pos[0], pos[1])
                    # move
                    # img_evt.clear()

                progress += nactual.value
            else:
                _lgr.info("Calling status")
                self.ph.PH_CTCStatus(ctypes.c_int(self._DEV_NUM), ctypes.byref(ctcDone))
                if ctcDone.value > 0:
                    _lgr.info("TCSPC Done")
                    self.numRecords = progress
                    self.stop_PH_measure()

                    # save real time for correlating with confocal images for FLIM
                    f.write(str(datetime.now()) + "\n")
                    f.write(str(time.time()) + "\n")
                    f.close()
                    _lgr.info("%s events recorded", self.numRecords)
                    measuring = False
            if self._stop_event.is_set():
                # do not use this as a condition as we need to do cleanup.
                measuring = False
        data_q.put(None)
        wt.join()


class WriterThread(_th.Thread):
    """Thread que guarda a archivo en background y recicla los buffers."""

    data_queue: _Queue = None
    buffer_queue: _deque = None
    fd = None

    def __init__(self, fd, data_q, buffer_q, *args, **kwargs):
        """Guarda los queues y el file descriptor."""
        super().__init__(*args, **kwargs)
        self.data_queue = data_q
        self.buffer_queue = buffer_q
        self.fd = fd

    def run(self):
        """Graba lo recibido y recicla los buffers."""
        total = 0
        _lgr.info("Iniciando thread de escritura")
        nv = self.data_queue.get()
        while nv is not None:
            n_rec, buffer = nv
            self.fd.write(buffer.data[:n_rec])
            self.buffer_queue.appendleft(buffer)
            total += n_rec
            nv = self.data_queue.get()
        _lgr.info("Fin thread de escritura. %s registros escritos.", total)


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
