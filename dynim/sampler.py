# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import os
from typing import List, Union
import logging

import numpy as np
from .hdpoint import HDPoint
from .utils import format_time, take_backup, filter_list, write_history

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# A base class for sampling functionality
# ------------------------------------------------------------------------------
class Sampler(object):

    # --------------------------------------------------------------------------
    def __init__(self, name: str, workspace: str,
                 buffer_size: int = 0,
                 min_cands_b4_sel: int = 0):

        """ Initialize a Sampler.
        Args:
            workspace (str):    a workspace directory where
                                history and checkpoints will be written
            buffer_size (int):  size of the candidate buffer_size
            min_cands_b4_sel (int):  min number of samples before any selection
        """
        assert isinstance(name, str)
        assert isinstance(workspace, str)
        assert isinstance(buffer_size, int) and int(buffer_size) >= 0
        assert isinstance(min_cands_b4_sel, int) and int(min_cands_b4_sel) >= 0

        self.name = name
        self.workspace = workspace
        self.buffer_size = buffer_size
        self.min_cands_b4_sel = min_cands_b4_sel
        os.makedirs(self.workspace, exist_ok=True)

        # candidates for the next set of selection (sorted list of samples)
        self.candidates = np.array([])

        # samples received since last selection (list of samples)
        self.cached = []

        # samples that have been selected (in the order of selection)
        self.selected = []

        # samples that have been dropped (maintain for recalculation of weights)
        self.discarded = []

        # function to explicitly scale the rank
        self.rank_scaler = None

        # prepare to write the history and checkpoints
        self.schkpt = os.path.join(self.workspace, f'{self.__tag__()}.data.npz')
        self.hists = os.path.join(self.workspace, f'{self.__tag__()}.history.csv')

    # --------------------------------------------------------------------------
    def __type__(self):
        return type(self).__name__

    def __tag__(self):
        return f'{self.__type__()}-{self.name}'

    def __str__(self):
        return f'[{self.__type__()}<{self.name}>:' \
               f' {len(self.cached)} cached,' \
               f' {len(self.candidates)} candidates,' \
               f' {len(self.selected)} selected,' \
               f' {len(self.discarded)} discarded]'

    def __repr__(self):
        return self.__str__()

    def _state_dict(self):
        return {'ncached': len(self.cached),
                'ncandidates': len(self.candidates),
                'nselected': len(self.selected),
                'ndiscarded': len(self.discarded)}

    # --------------------------------------------------------------------------
    def num_candidates(self):
        return len(self.cached) + len(self.candidates)

    def num_selected(self):
        return len(self.selected)

    def set_rank_scaler(self, rank_scaler: callable) -> None:
        assert callable(rank_scaler)
        LOGGER.info(f'Setting rank scalar ({rank_scaler}) for ({self.__tag__()})')
        self.rank_scaler = rank_scaler

    # --------------------------------------------------------------------------
    def _apply_ranks_to_candidates(self, ranks):
        assert isinstance(ranks, np.ndarray)
        assert ranks.dtype == np.float32
        assert ranks.shape[0] == len(self.candidates)

        if self.rank_scaler is None:
            LOGGER.info(f'Applying {len(self.candidates)} ranks to candidates')
            for i, sample in enumerate(self.candidates):
                sample.rank = ranks[i]

        else:
            LOGGER.info(f'Scaling {len(self.candidates)} ranks using ({self.rank_scaler})')
            for i, sample in enumerate(self.candidates):
                sample.rank = np.float32(self.rank_scaler(sample.id, ranks[i]))

    # --------------------------------------------------------------------------
    # add new candidates to the sampler
    # --------------------------------------------------------------------------
    def add_candidates(self, points: List[HDPoint], do_filter=True) -> None:

        assert isinstance(points, (list, np.ndarray))
        assert all([isinstance(p, HDPoint) for p in points])

        n = len(points)
        if n == 0:
            return

        # ----------------------------------------------------------------------
        def _filter(_ids_to_filter, _ids_to_check_against, _tag):

            if len(_ids_to_check_against) == 0:
                return np.zeros(len(_ids_to_filter), dtype=bool)

            dup_flags = np.isin(_ids_to_filter, _ids_to_check_against, assume_unique=True)
            ndups = dup_flags.sum()
            if ndups > 0:
                dups = _ids_to_filter[dup_flags]
                LOGGER.warning(f'Rejecting {ndups} already {_tag} points: {dups}')
                write_history(self.hists, f'rejected:add_cands:already_{_tag}',
                              dups, self._state_dict())

            return dup_flags

        # ----------------------------------------------------------------------
        LOGGER.debug(f'Adding {n} candidates to ({self})')

        # filter the ids we have already seen
        if do_filter:
            pids = HDPoint.fetch_ids(points)
            dflags0 = _filter(pids, HDPoint.fetch_ids(self.selected), 'selected')
            dflags1 = _filter(pids, HDPoint.fetch_ids(self.discarded), 'discarded')
            dflags2 = _filter(pids, HDPoint.fetch_ids(self.cached), 'cached')
            dflags3 = _filter(pids, HDPoint.fetch_ids(self.candidates), 'candidates')

            dflags = np.logical_or.reduce((dflags0, dflags1, dflags2, dflags3))
            points = np.array(points)
            points = points[np.logical_not(dflags)]
            points = points.tolist()

        # finally, add the ones we have not seen before
        n = len(points)
        LOGGER.info(f'Added {n} candidates to ({self})')
        self.cached.extend(points)
        write_history(self.hists, 'added', points, self._state_dict())

    # --------------------------------------------------------------------------
    # update the state of the sampler
    # --------------------------------------------------------------------------
    def update(self):

        LOGGER.profile(f'Updating sampler ({self})')

        # move the cached candidates to actual candidate list
        self.candidates = np.append(self.candidates, self.cached)
        self.cached = []

        n = len(self.candidates)
        if n > 0:
            # update the rankings (to be implemented by the actual selector!)
            self._update_ranks()

            # sort the candidates based on ranking!
            ranks = np.array([p.rank for p in self.candidates])
            order = np.argsort(ranks)[::-1]
            self.candidates = self.candidates[order]

            # discard the tail if needed!
            if 0 < self.buffer_size < n:
                discard = list(self.candidates[self.buffer_size:])
                self.candidates = self.candidates[:self.buffer_size]
                self.discarded.extend(discard)

                # write discards
                write_history(self.hists, 'discarded', discard, self._state_dict())

        LOGGER.profile(f'Updated sampler ({self})')

    # --------------------------------------------------------------------------
    # select k new candidates
    # --------------------------------------------------------------------------
    def select(self, k: int, confirm_selection: bool = True) -> List[HDPoint]:

        assert isinstance(k, int)
        assert k >= 0
        if k == 0:
            return []

        if (self.min_cands_b4_sel > 0) and \
           (self.num_selected() == 0) and \
           (self.num_candidates() < self.min_cands_b4_sel):
            LOGGER.debug(f'Not selecting due to too few candidates from {self}')
            return []

        self.update()
        LOGGER.debug(f'Selecting {k} samples from {self}')

        # pick the top k candidates
        k = min(k, len(self.candidates))
        selection = list(self.candidates[:k])

        LOGGER.info(f'Selected {k} samples from {self}')

        if confirm_selection:
            self._confirm_selections(selection)
            LOGGER.info(f'Confirmed selection of {k} samples from {self}')

        return selection

    # --------------------------------------------------------------------------
    # confirm the selection of these points
    # --------------------------------------------------------------------------
    def confirm_selections(self, selection: Union[List[HDPoint], List[str]]) -> bool:

        assert isinstance(selection, list)

        is_ids = False
        is_samples = all([isinstance(p, HDPoint) for p in selection])
        if not is_samples:
            is_ids = all([isinstance(p, str) for p in selection])

        if not is_ids and not is_samples:
            raise AttributeError('confirm_selections expects a list of '
                                 'ids or list of samples')

        k = len(selection)
        if k == 0:
            return False

        LOGGER.debug(f'Confirming selection of {k} samples from {self}')

        # ----------------------------------------------------------------------
        # make sure that the given selection are the top candidates
        if is_samples:
            for i in range(k):
                if self.candidates[i] != selection[i]:
                    raise AttributeError(f'Cannot confirm selection: '
                                         f'[{i} = {selection[i]}] not a candidate')
        else:
            for i in range(k):
                if self.candidates[i].id != selection[i]:
                    raise AttributeError(f'Cannot confirm selection: '
                                         f'[{i} = {selection[i]}] not a candidate')

        # ----------------------------------------------------------------------
        selection = list(self.candidates[:k])

        # now, check that these are not previously selected
        for i in range(k):
            if selection[i] in self.selected:
                raise AttributeError(f'Cannot confirm selection: '
                                     f'[{i} = {selection[i]}] already selected')

            if selection[i] in self.discarded:
                raise AttributeError(f'Cannot confirm selection: '
                                     f'[{i} = {selection[i]}] already discarded')

        # ----------------------------------------------------------------------
        self._confirm_selections(selection)
        LOGGER.info(f'Confirmed selection of {k} samples from {self}')

    # --------------------------------------------------------------------------
    def _confirm_selections(self, selection: List[HDPoint]) -> bool:

        assert isinstance(selection, list)
        assert all([isinstance(p, HDPoint) for p in selection])

        k = len(selection)
        if k == 0:
            return False

        LOGGER.debug(f'Confirming selection of {k} samples')

        # mark the selections as sampled (to be implemented by the child class!)
        self._add_selections(selection)

        # remove selected from the candidate list
        self.candidates = self.candidates[k:]

        # and add to the selected list
        self.selected.extend(selection)

        # write selections
        write_history(self.hists, 'selected', selection, self._state_dict())

    # --------------------------------------------------------------------------
    def invalidate_candidates(self, validator):
        assert callable(validator)

        def _filter_valid(_data, tag):
            _ids = [_.id for _ in _data]
            _flags = [_ for _ in validator(_ids)]
            _valids = [_data[i] for i, _ in enumerate(_flags) if _]
            _invalids = [_data[i] for i, _ in enumerate(_flags) if not _]

            LOGGER.info(f'{tag} = {len(_data)}: '
                        f'valid = {len(_valids)}, invalid = {len(_invalids)}')
            return _valids, _invalids

        LOGGER.info(f'testing for invalid candidates for {self}')

        self.cached, _invalid_cached = _filter_valid(self.cached, 'cached')
        self.candidates, _invalid_cands = _filter_valid(self.candidates, 'candidates')
        LOGGER.info(f'after invaliding patches: {self}')

        if len(_invalid_cands) > 0:
            _hfile = self.hists
            write_history(_hfile, 'invalidated:cached', _invalid_cached, self._state_dict())
            write_history(_hfile, 'invalidated:candidates', _invalid_cands, self._state_dict())

        return len(_invalid_cands)

    # --------------------------------------------------------------------------
    # checkpoint and restore
    # --------------------------------------------------------------------------
    def _checkpoint(self):

        st = format_time()
        LOGGER.info(f'Checkpointing Sampler data {self} at {st}')

        # take backup of previous checkpoint
        take_backup(self.schkpt)
        np.savez_compressed(self.schkpt,
                            t=st, type=self.__type__(), name=self.name,
                            cached=self.cached, candidates=self.candidates,
                            selected=self.selected, discarded=self.discarded)

    # --------------------------------------------------------------------------
    def _restore(self, filename=None):

        if filename is None:
            filename = self.schkpt

        try:
            LOGGER.debug(f'Restoring from ({filename})')
            data = np.load(filename, allow_pickle=True)

            assert self.name == data['name']
            assert self.__type__() == data['type']

            self.cached = list(data['cached'])
            self.selected = list(data['selected'])
            self.candidates = np.array(data['candidates'])
            self.discarded = list(data['discarded'])
            data.close()

        except Exception as e:
            raise Exception(f'{type(e).__name__}: {e} ({filename})')

        LOGGER.info(f'Successfully restored {self}')
        return True

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
