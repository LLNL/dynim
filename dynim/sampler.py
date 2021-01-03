# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import os
import sys
import yaml
from typing import List, Union
import logging

import numpy as np
from .hdpoint import HDPoint
from .utils import format_time, backup_fname, take_backup, write_history

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# A base class for sampling functionality
# ------------------------------------------------------------------------------
class Sampler(object):

    # --------------------------------------------------------------------------
    def __init__(self, name: str, workspace: str,
                 min_cands_b4_sel: int = 0, buffer_size: int = 0):

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

        if not os.path.isdir(self.workspace):
            os.makedirs(self.workspace)

        # prepare to write the history and checkpoints
        self.tag = '{}-{}'.format(self.type, self.name)
        self.chkpts = {
            'state': os.path.join(self.workspace, '{}.state.chk'.format(self.tag)),
            'data':  os.path.join(self.workspace, '{}.data.npz'.format(self.tag))
        }
        self.hists = os.path.join(self.workspace, '{}.history.csv'.format(self.tag))

        # candidates for the next set of selection (sorted list of samples)
        self.candidates = np.array([])

        # samples received since last selection (list of samples)
        self.cached = []

        # samples that have been selected (in the order of selection)
        self.selected = []

        # samples that have been dropped (maintain for recalculation of weights)
        self.discarded = []

    # --------------------------------------------------------------------------
    def __str__(self):
        s = '{}<{}> '.format(self.type, self.name)
        s += '[{} cached, {} candidates, {} selected, {} discarded]'\
            .format(len(self.cached), len(self.candidates),
                    len(self.selected), len(self.discarded))
        return s

    def __repr__(self):
        return self.__str__()

    # --------------------------------------------------------------------------
    def _state_dict(self):
        return {'ncached': len(self.cached),
                'ncandidates': len(self.candidates),
                'nselected': len(self.selected),
                'ndiscarded': len(self.discarded)}

    def num_candidates(self):
        return len(self.cached) + len(self.candidates)

    def num_selected(self):
        return len(self.selected)

    @staticmethod
    def _filter(data_to_filter, filter_if_present_in_list):

        if len(data_to_filter) == 0 or len(filter_if_present_in_list) == 0:
            return data_to_filter, []

        # since filter_if_present_in_list is likely very large,
        # let's use numpy to find the ones that are already included
        ids_in_data = np.array([_.id for _ in data_to_filter])
        ids_in_list = np.array([_.id for _ in filter_if_present_in_list])
        is_present = np.intersect1d(ids_in_data, ids_in_list)

        # no intersection
        if is_present.shape[0] == 0:
            return data_to_filter, []

        # slower list comprehension to find the ones not present
        is_not_present = [_ for _ in data_to_filter if _.id not in is_present]
        return is_not_present, is_present.tolist()

    # --------------------------------------------------------------------------
    # add new candidates to the sampler
    # --------------------------------------------------------------------------
    def add_candidates(self, points: List[HDPoint]) -> None:

        assert isinstance(points, list)
        assert all([isinstance(p, HDPoint) for p in points])

        n = len(points)
        if n == 0:
            return

        LOGGER.debug('Adding {} candidates to ({})'.format(n, self.__str__()))

        # remove the ones we have already seen
        points, discarded = self._filter(points, self.selected)
        if len(discarded) > 0:
            LOGGER.warning('Rejecting {} already selected points: {}'
                           .format(len(discarded), discarded))
            write_history(self.hists, 'rejected:add_cands:already_selected',
                          discarded, self._state_dict())

        points, discarded = self._filter(points, self.discarded)
        if len(discarded) > 0:
            LOGGER.warning('Rejecting {} already discarded points: {}'
                           .format(len(discarded), discarded))
            write_history(self.hists, 'rejected:add_cands:already_discarded',
                          discarded, self._state_dict())

        points, discarded = self._filter(points, self.cached)
        if len(discarded) > 0:
            LOGGER.warning('Rejecting {} already cached points: {}'
                            .format(len(discarded), discarded))
            write_history(self.hists, 'rejected:add_cands:already_cached',
                          discarded, self._state_dict())

        points, discarded = self._filter(points, self.candidates)
        if len(discarded) > 0:
            LOGGER.warning('Rejecting {} already candidates points: {}'
                           .format(len(discarded), discarded))
            write_history(self.hists, 'rejected:add_cands:already_candidates',
                          discarded, self._state_dict())

        # finally, add the ones we have not seen before
        self.cached.extend(points)
        LOGGER.info('Added {} candidates to ({})'.format(n, self.__str__()))
        write_history(self.hists, 'added', points, self._state_dict())

    # --------------------------------------------------------------------------
    # update the state of the sampler
    # --------------------------------------------------------------------------
    def update(self):

        LOGGER.debug('Updating sampler ({})'.format(self.__str__()))

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

        LOGGER.debug('Updated sampler ({})'.format(self.__str__()))

    # --------------------------------------------------------------------------
    # select k new candidates
    # --------------------------------------------------------------------------
    def select(self, k: int, confirm_selection: bool = True) -> List[HDPoint]:

        assert isinstance(k, int)
        assert k >= 0
        if k == 0:
            return []

        if 0 < self.min_cands_b4_sel and self.num_candidates() < self.min_cands_b4_sel:
            LOGGER.debug('Not selecting due to too few candidates: {}'
                         .format(self.__str__()))
            return []

        self.update()
        LOGGER.debug('Selecting {} samples from ({})'.format(k, self.__str__()))

        # pick the top k candidates
        k = min(k, len(self.candidates))
        selection = list(self.candidates[:k])

        LOGGER.info('Selected {} samples from ({})'.format(k, self.__str__()))

        if confirm_selection:
            self._confirm_selections(selection)
            LOGGER.info('Confirmed selection of {} samples from ({})'
                        .format(k, self.__str__()))

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

        LOGGER.debug('Confirming selection of {} samples from ({})'
                     .format(k, self.__str__()))

        # ----------------------------------------------------------------------
        # make sure that the given selection are the top candidates
        if is_samples:
            for i in range(k):
                if self.candidates[i] != selection[i]:
                    raise AttributeError('Cannot confirm selection [{} = {}]; '
                                         'not found in candidates.'
                                         .format(i, selection[i]))
        else:
            for i in range(k):
                if self.candidates[i].id != selection[i]:
                    raise AttributeError('Cannot confirm selection [{} = {}]; '
                                         'not found in candidates.'
                                         .format(i, selection[i]))
        # ----------------------------------------------------------------------
        selection = list(self.candidates[:k])

        # now, check that these are not previously selected
        for i in range(k):
            if selection[i] in self.selected:
                raise AttributeError('Cannot confirm selection [{} = {}]; '
                                     'already selected.'
                                     .format(i, selection[i]))

            if selection[i] in self.discarded:
                raise AttributeError('Cannot confirm selection [{} = {}]; '
                                     'already discarded.'
                                     .format(i, selection[i]))

        # ----------------------------------------------------------------------
        self._confirm_selections(selection)
        LOGGER.info('Confirmed selection of {} samples from ({})'
                    .format(k, self.__str__()))

    # --------------------------------------------------------------------------
    def _confirm_selections(self, selection: List[HDPoint]) -> bool:

        assert isinstance(selection, list)
        assert all([isinstance(p, HDPoint) for p in selection])

        k = len(selection)
        if k == 0:
            return False

        LOGGER.debug('Confirming selection of {} samples'.format(k))

        # mark the selections as sampled (to be implemented by the child class!)
        self._add_selections(selection)

        # remove selected from the candidate list
        self.candidates = self.candidates[k:]

        # and add to the selected list
        self.selected.extend(selection)

        # write selections
        write_history(self.hists, 'selected', selection, self._state_dict())

    # --------------------------------------------------------------------------
    # checkpoint and restore
    # --------------------------------------------------------------------------
    def checkpoint(self):

        st = format_time()
        LOGGER.info('Checkpointing {} at {}'.format(self.__str__(), st))
        sys.stdout.flush()

        # take backup of previous checkpoints
        take_backup(self.chkpts['state'])
        take_backup(self.chkpts['data'])

        # save current state ast yaml
        with open(self.chkpts['state'], 'w') as outfile:
            state = dict(t=st, type=self.type, name=self.name,
                         ncached=len(self.cached),
                         nselected=len(self.selected),
                         ncandidates=len(self.candidates),
                         ndiscarded=len(self.discarded))

            yaml.dump(state, outfile)

        # save data as npz
        np.savez_compressed(self.chkpts['data'],
                            t=st, type=self.type, name=self.name,
                            cached=self.cached, candidates=self.candidates,
                            selected=self.selected, discarded=self.discarded)

        LOGGER.debug('Checkpointing done for Sampler')

    def restore(self):

        for k in self.chkpts.keys():
            if not os.path.isfile(self.chkpts[k]):
                LOGGER.debug('Checkpoint file {} does not exist!'
                             .format(self.chkpts[k]))
                return False

        # load yaml
        with open(self.chkpts['state'], 'r') as infile:
            state = yaml.load(infile, Loader=yaml.Loader)

        # restore data!
        LOGGER.debug('Restoring data from {}'.format(state['t']))
        sys.stdout.flush()

        try:
            data = np.load(self.chkpts['data'], allow_pickle=True)
            assert self.type == data['type']
            assert self.name == data['name']

        except Exception as e:
            LOGGER.error('Failed to restore {}. Error = {}'.format(self.chkpts['data'], e))
            raise e

        self.cached = list(data['cached'])
        self.selected = list(data['selected'])
        self.candidates = np.array(data['candidates'])
        self.discarded = list(data['discarded'])

        assert self.test()
        LOGGER.info('Restored {}'.format(self.__str__()))
        return True

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
