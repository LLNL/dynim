# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import os
import sys
import logging
import numpy as np
from typing import List

from .hdspace import HDSpace
from .hdpoint import HDPoint
from .sampler import Sampler
from .utils import format_time, take_backup, backup_fname, read_history

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# An importance sampler of hd points
# ------------------------------------------------------------------------------
class SamplerImportance(Sampler):

    # --------------------------------------------------------------------------
    def __init__(self, name: str, workspace: str,
                 min_cands_b4_sel: int = 0, buffer_size: int = 0,
                 min_rand_b4_importance: int = 10):

        """ Initialize a HDSpace Sampler.
        Args:
            name (str):             a name given to this sampler
            workspace (str):        a directory where history and checkpoints will be written
            min_cands_b4_sel (int):  number of candidates to see before making any selection
            min_rand_b4_importance (int): number of random selections to make before switching to importance
            buffer_size (int):      size of the candidate buffer
        """
        assert isinstance(buffer_size, int) and int(buffer_size) >= 0
        assert isinstance(min_cands_b4_sel, int) and int(min_cands_b4_sel) >= 0
        assert isinstance(min_rand_b4_importance, int) and int(min_rand_b4_importance) >= 0

        self.type = 'SamplerImportance'
        super().__init__(name, workspace,
                         min_cands_b4_sel = min_cands_b4_sel,
                         buffer_size = buffer_size)

        self.min_rand_b4_importance = min_rand_b4_importance
        self.hdspace = None
        self.lchkpt = os.path.join(self.workspace, '{}.lspace.idx'.format(self.tag))

    # --------------------------------------------------------------------------
    # different ways of setting an HD space
    def set_hdspace(self, data: (str, HDSpace)) -> None:

        assert isinstance(data, (str, HDSpace))

        if isinstance(data, HDSpace):
            self.hdspace = data

        else:
            self.hdspace = HDSpace()
            self.hdspace.restore(data)

    # --------------------------------------------------------------------------
    # add these selections to hdspace
    def _add_selections(self, points: List[HDPoint]) -> None:

        if len(points) > 0:
            coords = np.array([p.coords for p in points])
            self.hdspace.add_points(coords)

    # --------------------------------------------------------------------------
    # update the ranks of the candidates
    def _update_ranks(self, k=10) -> None:

        n = len(self.candidates)

        # for the first few samples, just use random sampling
        # this is technically wrong!
        # if i get a request of N >> min_rand_b4_importance
        # i would still pick random ranks for all
        # but, if request ~= min_rand_b4_importance
        # we are probably going to be OK
        if len(self.selected) < self.min_rand_b4_importance:
            LOGGER.debug('Assigning random ranks to {} candidates'.format(n))
            for sample in self.candidates:
                sample.rank = np.random.uniform()

        # for the rest, rank = avg distance to [0, 0+k] nbrs
        else:
            LOGGER.debug('Assigning distance ranks to {} candidates'.format(n))

            coords = np.array([sample.coords for sample in self.candidates])
            ranks = self.hdspace.get_knn_distance(coords, k, 0)
            assert len(self.candidates) == len(ranks)

            for i, sample in enumerate(self.candidates):
                sample.rank = ranks[i]

    # --------------------------------------------------------------------------
    # get weights of selected samples
    # --------------------------------------------------------------------------
    def get_weights(self, normalize: bool= True):

        assert self.test()

        if len(self.selected) == 0:
            return [], []

        LOGGER.info('Computing weights: {}'.format(self))
        ids = np.array([s.id for s in self.selected])
        weights = np.ones(len(self.selected), dtype=np.float32)

        cands = [s.coords for s in self.candidates]
        cands += [s.coords for s in self.cached]
        cands += [s.coords for s in self.discarded]

        if len(cands) > 0:
            nnids = self.hdspace.get_nn_id(np.array(cands))
            for n in nnids:
                weights[n] += 1.

        if normalize:
            LOGGER.info('Computed weights: count = {}, sum = {}'
                        .format(weights.shape, weights.sum()))
            weights *= (weights.shape[0] / weights.sum())

        LOGGER.info('Computed weights: count = {}, sum = {}'
                    .format(weights.shape, weights.sum()))
        return ids, weights

    # --------------------------------------------------------------------------
    def checkpoint(self) -> None:
        super().checkpoint()
        take_backup(self.lchkpt)
        self.hdspace.checkpoint(self.lchkpt)

    def restore(self) -> bool:
        if not super().restore():
            LOGGER.warning('Failed to restore {}'.format(self.__str__()))
            return False
        self.hdspace.restore(self.lchkpt)
        assert self.test()
        return True

    # --------------------------------------------------------------------------
    def test(self) -> bool:
        # if len(self.selected) != self.hdspace.count():
        #   LOGGER.error('Inconsistent Sampler: lspace={}; {}'
        #   .format(self.hdspace.count(), self.__str__()))
        # return True        # TODO: remove this!
        return len(self.selected) == self.hdspace.count()

    # --------------------------------------------------------------------------
    def restore_from_history(self, coords_fetcher) -> None:

        assert callable(coords_fetcher)

        # ----------------------------------------------------------------------
        dadd, dsel, ddis = read_history(self.hists)

        # collect all the ids
        to_add = np.array([_[2].strip() for _ in dadd])
        to_sel = np.array([_[2].strip() for _ in dsel])
        to_dis = np.array([_[2].strip() for _ in ddis])

        # remove the selected and discarded from the list of added
        to_add = np.setdiff1d(to_add, to_dis)
        to_add = np.setdiff1d(to_add, to_sel)

        # get all the coordinates
        coords_add = coords_fetcher(to_add)
        coords_sel = coords_fetcher(to_sel)
        coords_dis = coords_fetcher(to_dis)

        if to_add.shape[0] > 0:
            assert coords_add.shape == (to_add.shape[0], self.hdspace.dim)
        if to_sel.shape[0] > 0:
            assert coords_sel.shape == (to_sel.shape[0], self.hdspace.dim)
        if to_dis.shape[0] > 0:
            assert coords_dis.shape == (to_dis.shape[0], self.hdspace.dim)

        # now, populate the datastructures
        candidates = np.array([])
        cached = [HDPoint(_id, _l) for _id, _l in zip(to_add, coords_add)]
        selected = [HDPoint(_id, _l) for _id, _l in zip(to_sel, coords_sel)]
        discarded = [HDPoint(_id, _l) for _id, _l in zip(to_dis, coords_dis)]

        # ----------------------------------------------------------------------
        # validate the history against current state
        def _validate(_d, _h, _tag):
            if len(_d) == 0:
                return
            _d = np.array([_.id for _ in _d])
            _h = np.array([_.id for _ in _h])

            _ab = np.setdiff1d(_d, _h)
            if _ab.shape[0] > 0:
                LOGGER.error('Inconsistent restore: Found {} {} points in data, '
                             'but not in history! {}'.format(_ab.shape, _tag, _ab))

            _ba = np.setdiff1d(_h, _d)
            if _ba.shape[0] > 0:
                LOGGER.info('Found {} {} points in history, '
                            'but not in data! {}'.format(_ba.shape, _tag, _ba))

        _validate(np.concatenate((self.cached, self.candidates)), cached, 'added')
        _validate(self.discarded, discarded, 'discarded')
        _validate(self.selected, selected, 'selected')

        # ----------------------------------------------------------------------
        # now, overwrite the data
        self.candidates = candidates
        self.cached = cached
        self.discarded = discarded
        self.selected = selected
        self._add_selections(self.selected)

        # ----------------------------------------------------------------------
        LOGGER.info('Restored from history: {}'.format(self.__str__()))
        assert len(self.selected) == self.hdspace.count()

# ------------------------------------------------------------------------------
