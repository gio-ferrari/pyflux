# -*- coding: utf-8 -*-
"""

"""

import time
import numpy as np
import numba

def process_raw_data(data: np.ndarray, n_records: int):
    t0 = time.time()
    rv = inner(data[:n_records])
    tf = time.time()
    print(tf-t0)
    return rv


@numba.njit((None)(np.uint32[:]))
def inner(data):
    oflcorrection = 0
    T3WRAPAROUND = 65536

    numRecords = len(data)
    dtime_array = np.zeros(numRecords)
    truensync_array = np.zeros(numRecords)
    recnum = 0
    for d in data:
        nsync = d & 0b1111111111111111
        d = d >> 16
        dtime = d & 0b111111111111
        d = d >> 12
        channel = d  # & 0b1111 No necesario
        if channel == 0xF:  # Special record
            if dtime == 0:  # Not a marker, so overflow
                oflcorrection += T3WRAPAROUND
            else:  # got marker
                truensync = oflcorrection + nsync
        else:  # standard record, photon count
            truensync = oflcorrection + nsync
            dtime_array[recnum] = dtime
            truensync_array[recnum] = truensync
            recnum += 1
    return dtime_array[:recnum], truensync_array[:recnum]
