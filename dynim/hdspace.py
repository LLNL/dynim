# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

from typing import List
import logging

import numpy as np
import faiss

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# A HD space used by samplers
# ------------------------------------------------------------------------------
class HDSpace(object):
    """A module to support an arbitrary high-dimensional space."""

    # --------------------------------------------------------------------------
    def __init__(self):
        self.dim = 0
        self.index = None

    def __str__(self) -> str:
        _type = 'unknown'
        if isinstance(self.index, faiss.IndexFlatL2):
            _type = 'exact'
        elif isinstance(self.index, faiss.IndexIVFFlat):
            _type = 'approx'
        else:
            assert False

        return 'HDSpace <D={}, type={}, npoints={}>'.format(self.dim, _type, self.count())

    def __repr__(self) -> str:
        return self.__str__()

    # --------------------------------------------------------------------------
    def count(self) -> int:
        return self.index.ntotal

    def empty(self) -> bool:
        return self.count() == 0

    # --------------------------------------------------------------------------
    def _assert_valid_points(self, points: np.ndarray) -> None:
        if not isinstance(points, np.ndarray):
            raise ValueError('Need numpy array of points')
        if points.dtype != np.float32:
            raise ValueError('Need float32 points')
        if len(points.shape) != 2:
            raise ValueError('Need points as a 2-rank vector of size (n,dim)')
        if points.shape[1] != self.dim:
            raise ValueError('Dimensionality mismatch: {} != (,{})'
                             .format(points.shape, self.dim))

    def _assert_valid_index(self) -> None:
        assert self.index is not None
        assert isinstance(self.index, (faiss.IndexFlatL2, faiss.IndexIVFFlat))
        if isinstance(self.index, faiss.IndexIVFFlat):
            assert self.index.is_trained

    # --------------------------------------------------------------------------
    # setup
    def setup(self, dim: int, idx_type: str = 'approx',
              nlist: int = 100, nprobes: int = 10) -> None:

        assert idx_type in ['approx', 'exact']
        assert dim > 0 and nlist > 0 and nprobes > 0

        self.dim = dim
        quantizer = faiss.IndexFlatL2(self.dim)

        # exact L2 distance metric
        if idx_type == 'exact':
            # quantizer is used as the index
            self.index = quantizer
            LOGGER.info('Initialized {}-D Space with {} metric'
                        .format(self.dim, idx_type))
            return

        # else
        # quantizer is used for mapping points to centers of voronoi cells
        self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist,
                                        faiss.METRIC_L2)
        self.index.nprobe = nprobes
        LOGGER.info('Initialized {}-D Space with {} metric ({}, {})'
                    .format(self.dim, idx_type, nlist, nprobes))

    # --------------------------------------------------------------------------
    # train the data-structure using some data
    # --------------------------------------------------------------------------
    def train(self, points: np.ndarray) -> None:

        if isinstance(self.index, faiss.IndexFlatL2):
            return

        assert not self.index.is_trained
        self._assert_valid_points(points)

        LOGGER.debug('Training with {} points'.format(points.shape))
        try:
            self.index.train(points)

        except Exception as e:
            LOGGER.error('Failed to train {}-D Space. Error = {}'
                         .format(self.dim, e))

        assert self.index.is_trained

    # --------------------------------------------------------------------------
    # restore the latent space
    # --------------------------------------------------------------------------
    def restore(self, fname: str) -> bool:

        try:
            self.index = faiss.read_index(fname)

        except Exception as e:
            LOGGER.error('Failed to restore HD Space. Error = {}'.format(e))
            return False

        self.dim = self.index.d
        LOGGER.info('Successfully restored {}-D Space with {} points from ({})'
                    .format(self.dim, self.count(), fname))
        return True

    # checkpoint the latent space
    def checkpoint(self, fname: str) -> bool:

        try:
            faiss.write_index(self.index, fname)

        except Exception as e:
            LOGGER.error('Failed to checkpoint {}-D Space. Error = {}'
                         .format(self.dim, e))
            return False

        LOGGER.info('Successfully checkpointed {}-D Space with {} points to ({})'
                    .format(self.dim, self.count(), fname))
        return True

    # --------------------------------------------------------------------------
    # add point(s) to the data-structure
    # --------------------------------------------------------------------------
    def add_points(self, points: np.ndarray) -> bool:

        self._assert_valid_index()
        self._assert_valid_points(points)

        LOGGER.debug('Adding {} points to {}-D Space'
                     .format(points.shape[0], self.dim))
        try:
            self.index.add(points)

        except Exception as e:
            LOGGER.error('Failed to add point to {}-D Space (error = {})'
                         .format(self.dim, e))
            return False

        return True

    # --------------------------------------------------------------------------
    # get k-nearest neighbors (starting from k0)
    # --------------------------------------------------------------------------
    def get_knn(self, points: np.ndarray, k: int, k0: int = 0) -> List[np.ndarray]:

        assert isinstance(k, int) and isinstance(k0, int)
        assert k >= 0 and k0 >= 0
        self._assert_valid_index()

        # ----------------------------------------------------------------------
        n = self.count()

        # dont make the query if the all the points will be excluded later
        if k0 >= n or n == 0:
            LOGGER.warning('Invalid request (k0={} > total={})'.format(k0, n))
            return []

        # ----------------------------------------------------------------------
        self._assert_valid_points(points)

        nquery = points.shape[0]
        LOGGER.debug('Getting ([{}, {}))-nn for {} query points (total = {})'
                     .format(k0, k + k0, nquery, n))
        try:
            k = min(k0 + k, n)
            dists, inds = self.index.search(points, k)

        except Exception as e:
            LOGGER.error('Failed to get knn from {}-D Space (error = {})'
                         .format(self.dim, e))
            return []

        assert dists.shape == (nquery, k) and inds.shape == (nquery, k)

        # ----------------------------------------------------------------------
        # remove any invalid values
        # although, should not be needed since we adjusted k to be <= n
        valids = np.bitwise_and(inds != -1, dists != np.finfo(dists.dtype).max)

        # pick valid knn after ignoring the first k0 valid points
        knn = []
        for i in range(nquery):
            vi = np.where(valids[i])[0]

            # check to see if we didn't find anything valid
            if vi.shape[0] == 0:
                knn.append((np.array([]), np.array([])))
                # knn.append((np.array([-1]), np.array([np.nan])))
            else:
                vi = vi[k0:]
                knn.append((inds[i][vi], dists[i][vi]))

        # the return value is a list of n tuples (n = number of query points)
        # each tuple contains (id, dist)
        # id    = a (k,1) ndarray of ids of nearest neighbors
        # dists = a (k,1) ndarray of dists of nearest neighbors
        # size may be smaller in case the system could not find k neighbors
        # either due to not having enough points or approximation error
        return knn

    # --------------------------------------------------------------------------
    # get mean distance to the nearest k points in the latent space
    #  (starting at k0)
    def get_knn_distance(self, points: np.ndarray, k: int, k0: int = 0) -> np.float32:

        self._assert_valid_points(points)
        nquery = points.shape[0]

        LOGGER.debug('Getting ([{}, {}))-nn dist for {} points (total = {})'
                     .format(k0, k + k0, nquery, self.count()))

        knn = self.get_knn(points, k, k0)
        mean_dists = -1. * np.ones(nquery).astype(np.float32)
        for i, _knn in enumerate(knn):
            # _knn is a tuple (ids, dists)
            mean_dists[i] = np.mean(_knn[1]) if _knn[1].shape[0] > 0 else np.nan

        return mean_dists

    # --------------------------------------------------------------------------
    # wrappers on the knn functions
    # --------------------------------------------------------------------------
    # get distance to the nearest point in the hd space
    def get_nn_distance(self, points: np.ndarray, k0: int = 0) -> np.ndarray:
        nn = self.get_knn(points, 1, k0)
        return np.array([_[1][0] if _[1].shape[0] > 0 else np.nan for _ in nn])

    # get id of the nearest neighbor
    def get_nn_id(self, points: np.ndarray, k0: int = 0) -> np.ndarray:
        nn = self.get_knn(points, 1, k0)
        return np.array([_[0][0] if _[1].shape[0] > 0 else -1 for _ in nn])


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
