# -*- coding: utf-8 -*-
"""
Fast analysis functions for p-Minflux localization and stabilization

@author: Andres Zelcer
"""

import numpy as _np
import logging as _lgn

_lgr = _lgn.getLogger(__name__)

try:
    import numba
    _NUMBA_PRESENT = True
except ImportError:
    _NUMBA_PRESENT = False


if _NUMBA_PRESENT:
    @numba.njit(nogil=True, parallel=True)
    def _loc_helper(n, logs, likelihood):
        likelihood[:] = n[0] * logs[0]
        for k in range(n.shape[0]-1):
            likelihood += n[k+1] * logs[k+1]
        # maximum likelihood estimator for the position
        i_max = _np.argmax(likelihood)
        return (i_max // likelihood.shape[0], i_max % likelihood.shape[0])


class MinFluxLocator:
    """Caching MINFLUX position estimator (using MLE)

    Precomputes everything that does not depend on photon counts

    Use
    ---
    >>> locator = MinFluxLocator(PSFs, background, pixel_size)
    >>> n = get_data_from_source()
    >>> indices, position, likelihood = locator(n)

    Uses numba if present.
    Performance is aprox 5x faster than that of non caching implementation when
    using plain numpy, and about 10x faster when numba is used.
    """

    def __init__(self, PSF: _np.ndarray, SBR: float, step_nm: float = 1.,
                 use_numba=True):
        """Precompute everything.

        Parameters
        ----------
        PSF: numpy.ndarray
            3-Dimensional array with EBP (K x size x size)
        SBR: float
            Estimated (exp) Signal to Bkgd Ratio
        step_nm: float
            Grid pixel size in nm
        """
        self.PSF = PSF
        self.SBR = SBR
        self.step_nm = step_nm
        self._pre_process()
        if use_numba:
            if _NUMBA_PRESENT:
                self._numba_precompile()
                self._estimate = self._numba_estimate
                _np.moveaxis(self.logs, 0, -1)  # use a faster ordering
            else:
                _lgr.warning("Numba requested but not present: ignoring")

    def _pre_process(self):
        """Do the actual precalculation."""
        K = _np.shape(self.PSF)[0]
        self.K = K
        SBR = self.SBR
        self.size = _np.shape(self.PSF)[1]
        size = self.size
        PSF = self.PSF / _np.sum(self.PSF, axis=0)
        # probabilitiy vector
        prob = _np.zeros((K, size, size))

        for i in _np.arange(K):
            prob[i, :, :] = (SBR / (SBR + 1.)) * PSF[i, :, :] + (1. / (SBR + 1.)) * (
                1 / K
            )

        # log-likelihood function
        self.logs = _np.zeros((K, size, size))
        self.LTot = _np.zeros((size, size,))
        for i in _np.arange(K):
            self.logs[i, :, :] = _np.log(prob[i, :, :])
        self.l_aux = _np.zeros((K, size, size))

    def _numba_precompile(self):  # Not the best way
        _loc_helper(_np.zeros((self.K,)), self.logs, self.LTot)

    def __call__(self, n):
        return self._estimate(n)

    def _estimate(self, n):
        """Estimate MINFLUX position.

        Parameters
        ----------
        n : numpy.ndarray
            Acquired photon collection (length K)

        Returns
        -------
        A 3-member tuple of:
            - position indices
            - position in nm
            - Likelihood function

        """
        # log-likelihood function
        for i in range(self.K):
            self.l_aux[i, :, :] = n[i] * self.logs[i]

        self.LTot = _np.sum(self.l_aux, axis=0)

        # maximum likelihood estimator for the position
        indrec = _np.unravel_index(_np.argmax(self.LTot, axis=None), self.LTot.shape)

        pos_estimator = pos_estimator = self._idx2pos(indrec)
        return indrec, pos_estimator, self.LTot

    def _numba_estimate(self, n):
        indrec = _loc_helper(n, self.logs, self.LTot)
        pos_estimator = self._idx2pos(indrec)
        return indrec, pos_estimator, self.LTot

    def _idx2pos(self, indices):
        """Convert indices to positions relative to center."""
        return (_np.array(indices, dtype=float) - self.size/2) * self.step_nm

    def _pos2idx(self, pos):
        """Convert positions to indexes."""
        x = int(pos[0] / self.step_nm + self.size / 2)
        y = int(pos[1] / self.step_nm + self.size / 2)
        x = min(max(x, 0), self.size - 1)
        y = min(max(y, 0), self.size - 1)
        return x, y
