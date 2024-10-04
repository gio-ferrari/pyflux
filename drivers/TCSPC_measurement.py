#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:58:46 2024

@author: azelcer
"""

import matplotlib.pyplot as plt
import TimeTagger
import numpy as np
import numba
import time


_MAX_EVENTS = 131072
_PERIOD = int(50E3)

class TCSPCMeasurement(TimeTagger.CustomMeasurement):
    """Swabian measurement that returns time differences."""

    _delays = np.empty(int(2E6), dtype=np.int32)
    _errors: int = 0
    _t_last_call = 0

    def __init__(self, tagger, APD_channel, laser_channel, period, max_events):
        super().__init__(tagger)
        self._APD_channel = APD_channel
        self._laser_channel = laser_channel
        self._period = period
        self._max_events = max_events

        # The method register_channel(channel) activates
        # data transmission from the Time Tagger to the PC
        # for the respective channels.
        self.register_channel(channel=APD_channel)
        self.register_channel(channel=laser_channel)

        self.clear_impl()

        # At the end of a CustomMeasurement construction,
        # we must indicate that we have finished.
        self.finalize_init()

    def __del__(self):
        # Inherited from example... del is not a good place to do this
        # The measurement must be stopped before deconstruction to avoid
        # concurrent process() calls.
        self.stop()

    def getData(self):
        # Locking this instance to guarantee that process() is not running in parallel
        # This ensures to return a consistent data.
        with self.mutex:
            return self._delays[:self._last_pos].copy()

    def clear_impl(self):
        # The lock is already acquired within the backend.
        self._last_pos = 0
        self._delays = np.empty(self._max_events, dtype=np.int32)

    def on_start(self):
        # The lock is already acquired within the backend.
        print("starting")
        self._t_last_call = time.time_ns()
        pass

    def on_stop(self):
        # The lock is already acquired within the backend.
        print("Finishing")
        print(f"{len(self._tiempos)} llamados a proc con un tiempo promedio de "
              f"{np.average(self._tiempos)/1E6} ms y un maximo de {max(self._tiempos)/1E6} ms")
        pass

    @staticmethod
    @numba.jit((numba.uint64)(
        numba.from_dtype(TimeTagger.CustomMeasurement.INCOMING_TAGS_DTYPE)[:],
        numba.int32[:], numba.int64, numba.int64, numba.int64, numba.int64,
        numba.int64[:]),
        nopython=True, nogil=True)
    def fast_process(tags: np.ndarray, data: np.ndarray,
            APD_channel: int, laser_channel: int, period: int, last_pos: int, errors: np.ndarray):
        last_start_timestamp = 0  # Puede perder un tag si quedaron en separados
        n_errors = 0
        for tag in tags:
            # tag.type can be: 0 - TimeTag, 1- Error, 2 - OverflowBegin, 3 -
            # OverflowEnd, 4 - MissedEvents (you can use the TimeTagger.TagType IntEnum)
            if tag['type'] != TimeTagger.TagType.TimeTag:
                # tag is not a TimeTag, so we are in an error state, e.g. overflow
                last_start_timestamp = 0
                n_errors += 1
            elif tag['channel'] == laser_channel and last_start_timestamp != 0:
                # valid event
                data[last_pos] = (tag['time'] - last_start_timestamp)
                if last_pos < data.size - 1:
                    last_pos += 1
                else:
                    n_errors += 1
            if tag['channel'] == APD_channel:
                last_start_timestamp = tag['time']
        errors[0] = n_errors
        return last_pos

    def process(self, incoming_tags, begin_time, end_time):
        """
        Main processing method for the incoming raw time-tags.

        The lock is already acquired within the backend.
        self.data is provided as reference, so it must not be accessed
        anywhere else without locking the mutex.

        Parameters
        ----------
        incoming_tags
            The incoming raw time tag stream provided as a read-only reference.
            The storage will be deallocated after this call, so you must not store a reference to
            this object. Make a copy instead.
            Please note that the time tag stream of all channels is passed to the process method,
            not only the ones from register_channel(...).
        begin_time
            Begin timestamp of the of the current data block.
        end_time
            End timestamp of the of the current data block.
        """
        # pero procesar acá mismo la info.
        t0 = time.time_ns()
        # print(f"{(t0 - self._t_last_call)/1E6} ms desde llamada anterior")
        # lst = self._last_pos
        # self._tiempos.append(t0 - self._t_last_call)
        errors = np.empty((1,), dtype=np.int64)
        self._last_pos = TCSPCMeasurement.fast_process(
            incoming_tags,
            self._delays,
            self._APD_channel,
            self._laser_channel,
            self._period,
            self._last_pos,
            errors)
        self._tiempos.append(time.time_ns()-t0)

        if errors[0]:
            print("hubo ", errors[0], "errores")


if __name__ == '__main__':

    with TimeTagger.createTimeTagger() as tagger:
        APD_CHANNEL = 4
        LASER_CHANNEL = 2
        tagger.setStreamBlockSize(max_events=_MAX_EVENTS, max_latency=2)
        tagger.setTriggerLevel(channel=APD_CHANNEL, voltage=2.5)
        test_time = 1E12
        # Set Test signal frecuency

        tagger.setConditionalFilter(trigger=[APD_CHANNEL], filtered=[LASER_CHANNEL])
        # We first have to create a SynchronizedMeasurements object to synchronize
        # several measurements
        with TimeTagger.SynchronizedMeasurements(tagger) as measurementGroup:
            TCSPC = TCSPCMeasurement(
                measurementGroup.getTagger(),
                APD_CHANNEL,
                LASER_CHANNEL,
                _PERIOD,
                _MAX_EVENTS,
                )
            cntrt_measurement = TimeTagger.Countrate(
                measurementGroup.getTagger(),
                [APD_CHANNEL],
                )
            print("Acquire data...\n")
            measurementGroup.startFor(int(test_time))
            measurementGroup.waitUntilFinished()
            data = TCSPC._delays
            last_pos = TCSPC._last_pos
            CPS = cntrt_measurement.get_data()
        tagger.clearConditionalFilter()
