#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:08:53 2024

@author: azelcer

Tools for working with PSF files
"""

import numpy as _np
import imageio as _iio
import logging as _lgn
import configparser as _cp
import numba as _nb
from typing import Tuple

# _lgn.basicConfig(level=_lgn.DEBUG)
_lgr = _lgn.getLogger(__name__)


@_nb.njit
def radial_sum(image: _np.ndarray):
    """Muy ineficiente: usar stencil.

    escrito como lo pense cero optimizacion
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
                # print(xp, yp, "NAN")
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
    im = image.astype(_np.float64)
    rv = radial_sum(im)
    trimmed = rv[trim: -trim, trim: -trim]
    ind = _np.unravel_index(_np.argmin(trimmed, axis=None), trimmed.shape)
    nidx = (ind[0] + trim, ind[1] + trim)
    return nidx


def find_min(image: _np.ndarray, trim: int = 20) -> tuple:
    trimmed = image[trim: -trim, trim: -trim]
    ind = _np.unravel_index(_np.argmin(trimmed, axis=None), trimmed.shape)
    nidx = (ind[0] + trim, ind[1] + trim)
    return nidx


def centers_minflux(L: float, k: int = 4):
    """Return center position of donuts respect to 0, 0.

    La dona 0 siempre el el medio, la última en (L/2, 0) y de ahí la numeración baja
    en sentido antihorario (Igual paper Balzarotti)
    """
    i = _np.arange(k)
    ebp = _np.zeros((k, 2), dtype=_np.float64)
    angles = i[:-1] / (k-1) * 2 * _np.pi
    ebp[1:][::-1, 0] = _np.sin(angles) * L / 2
    ebp[1:][::-1, 1] = _np.cos(angles) * L / 2
    return ebp

def preprare_grid_forfit(side_x, side_y):
    x_axis = _np.arange(0, side_x, 1)
    y_axis = _np.arange(0, side_y, 1)
    x_grid, y_grid = _np.meshgrid(x_axis, y_axis)
    grid = _np.vstack((x_grid.ravel(), y_grid.ravel()))
    return grid

def custom_parab(coord1, coord2, coeff00, coeff10, coeff01, coeff11, coeff20, coeff02):
    return coeff00 + coeff10 * coord1 + coeff01 * coord2 + coeff11 * coord1 * coord2 + coeff20 * coord1**2 + coeff02 * coord2**2

def parab_func(grid, c00, c10, c01, c11, c20, c02):
    """    
    Polynomial of deg 2 (parabola) function to fit PSFs.
    
    Inputs
    ----------
    grid : x,y array
    cij : coefficients of the polynomial function
    
    Returns
    -------
    q : polynomial evaluated in grid.
    
    """
        
    x, y = grid
    return custom_parab(x, y, c00, c10, c01, c11, c20, c02).ravel()

def parag_param_guess(psf_matrix, min_pos_psf):
    c00_0 = psf_matrix[0,0]
    c20_0 = (psf_matrix[0, min_pos_psf[1]] + psf_matrix[-1, min_pos_psf[1]])/(2*(psf_matrix.shape[0]/2)**2)
    c02_0 = (psf_matrix[min_pos_psf[0], 1] + psf_matrix[min_pos_psf[0], -1])/(2*(psf_matrix.shape[1]/2)**2)
    c10_0 = -2 * min_pos_psf[0] * c20_0
    c01_0 = -2 * min_pos_psf[1] * c20_0
    return [c00_0, c10_0, c01_0, 0, c20_0, c02_0]

def parab_determinant(
    c11: float,
    c20: float,
    c02: float
):
    return 4 * c20 * c02 - c11**2

def parab_analytical_min(
    c00: float,
    c10: float,
    c01: float,
    c11: float,
    c20: float,
    c02: float,
    offset_x: float,
    offset_y: float,
    pixel_size: float
):
    determinant = parab_determinant(c11, c20, c02)
    min_coords_fit = (
        ((-2 * c10 * c02 + c01 * c11) / determinant + offset_x) * pixel_size,
        ((-2 * c01 * c20 + c10 * c11) / determinant + offset_y) * pixel_size
        )
    
    return min_coords_fit

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    basename = "/tmp/psf_20240527_1"
    K = 4
    n_frames = 4
    # basename = "/tmp/dona1_real"
    imfilename = basename + ".tiff"
    mdfilename = basename + ".txt"
    data = _iio.imread(imfilename)
    _lgr.debug("Images shape: %s", data.shape)
    # _lgr.debug("Images metadata: %s", data.meta)

    config = _cp.ConfigParser()
    mdata = config.read(mdfilename, encoding="latin-1")
    sr = 1
    mpp = 1
    if not mdata:
        print("No pude cargar data")
    else:
        if 'Scanning parameters' not in config.sections():
            raise ValueError("Bad config file")
        sp = config['Scanning parameters']
        # FIXME: a veces usan mayúscula al principio
        sr = sp.getfloat('scan range (µm)')
        mpp = sp.getfloat('pixel size (µm)')
        print("Reported image side:", sp['number of pixels'], ", found: ", data[0].shape)
        print("Reported pixel Size:", mpp, ", found: ", sr/data[0].shape[0])
    

    _lgr.info("Promediando donas")
    reshaped = _np.reshape(data, (n_frames, K, data.shape[1], data.shape[2]))
    avdata = _np.average(reshaped, 1)

    cms = centers_minflux(40, 4)
    for k, d in enumerate(avdata):
        plt.figure()
        plt.imshow(d)
        centro = find_min(d)
        # print(centro)
        plt.scatter(centro[1], centro[0], c='r')
        cm = cms[k]
        plt.scatter(cm[1]+40, cm[0]+40, c='w')
    plt.figure("test")
    plt.scatter(cms[:,0],cms[:,1])
   
 
    # config['Scanning parameters'] = {
    #     'Date and time': dateandtime,
    #     'Initial Position [x0, y0, z0] (µm)': main.initialPos,
    #     'Focus lock position (px)': str(main.focuslockpos),
    #     'Scan range (µm)': main.scanRange,
    #     'Pixel time (µs)': main.pxTime,
    #     'Number of pixels': main.NofPixels,
    #     'a_max (µm/µs^2)': str(main.a_max),
    #     'a_aux [a0, a1, a2, a3] (% of a_max)': main.a_aux_coeff,
    #     'Pixel size (µm)': main.pxSize,
    #     'Frame time (s)': main.frameTime,
    #     'Scan type': main.scantype,
    #     'Power at BFP (µW)': main.powerBFP}
