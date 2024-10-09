#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:58:46 2024

@author: azelcer
"""

import TimeTagger
import numpy as np
import numba
import time


_MAX_EVENTS = 131072
_PERIOD = int(50E3)

class MinfluxMeasurement(TimeTagger.CustomMeasurement):
    """Swabian measurement that returns localizations."""

    _delays = np.empty((0,), dtype=np.int32)
    _bins = np.zeros((4,), dtype=np.int64)
    _errors: int = 0
    _last_time: int = 0

    def __init__(self, tagger, APD_channel, laser_channel, period, max_events,
                 delays, callback):
        super().__init__(tagger)
        self._APD_channel = APD_channel
        self._laser_channel = laser_channel
        self._period = period
        self._max_events = max_events
        # Hacemos la cuenta de los delays como vamos a calcularlos después
        # Exigimos que los delays estén ordenados. No los ordenamos acá para
        # forzar al caller a hacer bien
        if sorted(delays) != list(delays):
            raise ValueError("Delays are not sorted")
        self.shutter_delays = period - np.array(delays, dtype=np.int64)
        self._cb = callback

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

    # def getData(self):
    #     # Locking this instance to guarantee that process() is not running in parallel
    #     # This ensures to return a consistent data.
    #     with self.mutex:
    #         return self._delays[:self._last_pos].copy()

    def clear_impl(self):
        # The lock is already acquired within the backend.
        self._last_time = 0
        self._delays = np.empty(self._max_events, dtype=np.int32)
        self._bins[:] = 0

    def on_start(self):
        # The lock is already acquired within the backend.
        print("starting")
        pass

    def on_stop(self):
        # The lock is already acquired within the backend.
        print("Finishing")
        pass

    @staticmethod
    @numba.jit((numba.uint64)(
        numba.from_dtype(TimeTagger.CustomMeasurement.INCOMING_TAGS_DTYPE)[:],
        numba.int32[:], numba.int64, numba.int64,
        numba.int64[:]),
        nopython=True, nogil=True)
    def process_tags(tags: np.ndarray, data: np.ndarray,
                     APD_channel: int, laser_channel: int,
                     errors: np.ndarray):
        """Save time differences in data, return number of records."""
        n_errors = 0
        last_pos = 0
        last_timestamp = 0  # Puede ser que perdamos info, ponerlo en errors[1]
        for tag in tags:
            # tag.type can be: 0 - TimeTag, 1- Error, 2 - OverflowBegin, 3 -
            # OverflowEnd, 4 - MissedEvents (you can use the TimeTagger.TagType IntEnum)
            if tag['type'] != TimeTagger.TagType.TimeTag:
                # tag is not a TimeTag, so we are in an error state, e.g. overflow
                last_timestamp = 0
                n_errors += 1
            elif tag['channel'] == laser_channel and last_timestamp != 0:
                # valid event
                data[last_pos] = (tag['time'] - last_timestamp)
                if last_pos < data.size - 1:  # overkill
                    last_pos += 1
                else:
                    n_errors += 1
            if tag['channel'] == APD_channel:
                last_timestamp = tag['time']
        errors[0] = n_errors
        return last_pos

    @staticmethod
    @numba.jit((numba.int32[:], numba.int64[:], numba.int64[:]),
               nopython=True, nogil=True,) # parallel=True)
    def process_delays(data: np.ndarray, delays, bins):
        """Bin time differences in data, return number of records.

        WARNING: delays must be ORDERED.
        """
        bins[:] = 0
        for td in data:  #i in numba.prange(len(data)):  # hardcoded para 4
            # td = data[i]
            bins[3 - td // 12500] += 1
            # bins[0]+=1
            # if td <= delays[3]:
            #     bins[3] += 1
            # elif td <= delays[2]:
            #     bins[2] += 1
            # elif td <= delays[1]:
            #     bins[1] += 1
            # else:
            #     bins[0] += 1

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
        # TODO: meter todo el pipeline en una sola función
        errors = np.empty((2,), dtype=np.int64)
        n_times = MinfluxMeasurement.process_tags(
            incoming_tags,
            self._delays,
            self._APD_channel,
            self._laser_channel,
            errors)
        if errors[0]:
            print("hubo ", errors[0], "errores")
        # TODO: pasar esta línea a numba
        # n_times = max(n_times, 100)  # usar los últimos 100 fotones.
        MinfluxMeasurement.process_delays(self._delays[:n_times],
                                          self.shutter_delays, self._bins)
        self._cb(self._delays[:n_times].copy(), self._bins.copy(), (0, 0))


if __name__ == '__main__':
    delays = np.array([0, 2000, 12000, 25000])
    # test = MinfluxMeasurement(None, 4, 2, int(50E3), _MAX_EVENTS, delays, test)
    idelays = int(50E3) - delays
    data = np.random.randint(0, int(50E3), 100000, dtype=np.int32)
    bins = np.zeros((4,), dtype=np.int64)
    MinfluxMeasurement.process_delays(data, idelays, bins)
