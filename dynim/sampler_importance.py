# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import os
import logging
import numpy as np
from typing import List

from .hdspace import HDSpace
from .hdpoint import HDPoint
from .sampler import Sampler
from .utils import take_backup, backup_fname, read_history, files_exist

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# An importance sampler of hd points
# ------------------------------------------------------------------------------
class SamplerImportance(Sampler):

    # --------------------------------------------------------------------------
    def __init__(self, name: str, workspace: str,
                 buffer_size: int = 0,
                 min_cands_b4_sel: int = 0,
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

        super().__init__(name, workspace,
                         buffer_size = buffer_size,
                         min_cands_b4_sel=min_cands_b4_sel)

        self.min_rand_b4_importance = min_rand_b4_importance
        self.hdspace = None
        self.lchkpt = os.path.join(self.workspace, f'{self.__tag__()}.lspace.idx')
        LOGGER.info(f'Initialized {self.__tag__()}')

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

        if len(points) == 0:
            return
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
            LOGGER.debug(f'Computing random ranks for ({n}) points')
            ranks = np.random.rand(n).astype(np.float32)

        # for the rest, rank = avg distance to [0, 0+k] nbrs
        else:
            LOGGER.debug(f'Computing importance ranks for ({n}) points')
            coords = np.array([sample.coords for sample in self.candidates])
            ranks = self.hdspace.get_knn_distance(coords, k, 0).astype(np.float32)

        # now apply these ranks to candidates
        self._apply_ranks_to_candidates(ranks)

    # --------------------------------------------------------------------------
    # get weights of selected samples
    # --------------------------------------------------------------------------
    def get_weights(self, normalize=True):

        assert self.test()
        if len(self.selected) == 0:
            return [], []

        LOGGER.info(f'Computing weights {self}')
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
            LOGGER.info(f'Computed weights: {weights.shape}, sum = {weights.sum()}')
            weights *= (weights.shape[0] / weights.sum())

        LOGGER.info(f'Computed weights: {weights.shape}, sum = {weights.sum()}')
        return ids, weights

    # --------------------------------------------------------------------------
    def checkpoint(self):

        super()._checkpoint()
        take_backup(self.lchkpt)
        self.hdspace.checkpoint(self.lchkpt)
        LOGGER.debug(f'Checkpointed {self}')

    def restore(self):

        # ----------------------------------------------------------------------
        def _restore_files(_files):
            if not files_exist(_files):
                return False

            try:
                self._restore(_files[0])
                self.hdspace.restore(_files[1])
            except Exception as e:
                LOGGER.error(e)
                return False

            if not self.test():
                LOGGER.error('Inconsistent restore!')
                return False
            return True

        # ----------------------------------------------------------------------
        files = [self.schkpt, self.lchkpt]

        success = False
        for i in range(2):
            success = _restore_files(files)
            if success:
                break
            files = [backup_fname(f) for f in files]

        success = success and self.test()
        if not success:
            LOGGER.info(f'Failed to restore')
            return False

        LOGGER.info(f'Restored {self}')
        return True

    # --------------------------------------------------------------------------
    def test(self):

        h = self.hdspace.count()
        if len(self.selected) != h:
            LOGGER.error(f'Inconsistent Sampler: lspace={h}; {self}')
            return False
        return True

    # --------------------------------------------------------------------------
    def restore_from_history(self, coords_fetcher):

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
                LOGGER.error(f'Inconsistent restore: '
                             f'Found {_ab.shape} {_tag} points in data, '
                             f'but not in history! {_ab}')

            _ba = np.setdiff1d(_h, _d)
            if _ba.shape[0] > 0:
                LOGGER.info(f'Found {_ba} {_tag} points in history, '
                            f'but not in data! {_ba}')

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
        LOGGER.info(f'Restored from history: {self}')
        assert len(self.selected) == self.hdspace.count()

# ------------------------------------------------------------------------------
