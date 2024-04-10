#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:31:08 2024

@author: azelcer
"""

import numpy as _np
from numba import njit as _njit


@_njit
def radial_sum(image: _np.ndarray):
    """Muy ineficiente: usar stencil.

    escrito como lo pensé: cero optimizacion
    """
    x = _np.arange(image.shape[0])
    y = _np.arange(image.shape[1])
    out = _np.zeros_like(image, dtype=_np.float64)
    for xp in x:
        max_x = min(xp, x[-1] - xp)
        for yp in y:
            max_y = min(yp, y[-1] - yp)
            if max_x == 0 and max_y == 0:
                out[xp, yp] = _np.inf
            else:
                nitems = 0
                value = 0
                for dy in range(1, max_y + 1):
                    xrange = _np.arange(-max_x, max_x + 1) if max_x else _np.arange(1)
                    for dx in xrange:
                        value += abs(image[xp+dx, yp+dy] - image[xp-dx, yp-dy])
                        nitems += 1
                for dx in _np.arange(max_x + 1):
                    value += abs(image[xp+dx, yp] - image[xp-dx, yp])
                    nitems += 1
                out[xp, yp] = value/nitems
        ...
    ...
    return out


def find_center(image: _np.ndarray, trim: int = 20) -> tuple:
    """Find the center of a radially symmetric feature.

    Parameters
    ----------
        image: numpy.ndarray
            2D array representing an image
        trim: int (optional)
            number or border pixels to exclude from the analysis

    Returns
    -------
        tuple of ints representing the (x, y) coordinates of the centers
    """
    im = image.astype(_np.float64)
    rv = radial_sum(im)
    trimmed = rv[trim: -trim, trim: -trim]
    ind = _np.unravel_index(_np.argmin(trimmed, axis=None), trimmed.shape)
    nidx = (ind[0] + trim, ind[1] + trim)
    return nidx


if __name__ == '__main__':
    import imageio as iio
    import matplotlib.pyplot as plt
    # image = _np.random.rand(*(5,5))
    im = _np.array(iio.mimread(r'/home/azelcer/Devel/datos_test/dona_haz2_naranja.tiff'))[0]
    # im = _np.array([
    #     [0.00, 0.00, 0.00, 0.00, 0.00],
    #     [0.00, 0.75, 1.00, 0.75, 0.00],
    #     [0.00, 1.00, 0.00, 1.00, 0.00],
    #     [0.00, 0.75, 1.00, 0.75, 0.00],
    #     [0.00, 0.00, 0.00, 0.00, 0.00],
    #     ])
    plt.figure("original")
    plt.imshow(im)
    nidx = find_center(im)
    nidx = list(nidx)
    nidx.reverse()
    plt.scatter(*nidx)
