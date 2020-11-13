# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import os
import sys
import time
import yaml
from typing import List, Union
import logging

import numpy as np
from .hdpoint import HDPoint
from .utils import format_time, take_backup, write_history

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# A base class for sampling functionality
# ------------------------------------------------------------------------------
class Sampler(object):

    # --------------------------------------------------------------------------
    def __init__(self, name: str, workspace: str, buffer_size: int):

        """ Initialize a Sampler.
        Args:
            workspace (str):    a workspace directory where
                                history and checkpoints will be written
            buffer_size (int):  size of the candidate buffer_size
        """
        assert isinstance(name, str)
        assert isinstance(workspace, str)
        assert isinstance(buffer_size, int) and int(buffer_size >= 0)

        self.name = name
        self.workspace = workspace
        self.buffer_size = buffer_size

        if not os.path.isdir(self.workspace):
            os.makedirs(self.workspace)

        # prepare to write the history and checkpoints
        self.tag = '{}_{}'.format(self.name, self.type)
        self.chkpts = {
            'state': os.path.join(self.workspace, '{}.state.chk'.format(self.tag)),
            'data':  os.path.join(self.workspace, '{}.data.npz'.format(self.tag))
        }
        self.hists = {
            'selections': os.path.join(self.workspace, '{}.selections.csv'.format(self.tag)),
            'discards':  os.path.join(self.workspace, '{}.discards.csv'.format(self.tag))
        }

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
    def num_candidates(self):
        return len(self.cached) + len(self.candidates)

    def num_selected(self):
        return len(self.selected)

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
        self.cached.extend(points)
        LOGGER.info('Added {} candidates to ({})'.format(n, self.__str__()))

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

                # write selections and discards
                ts = time.time()
                write_history(self.hists['discards'], discard, ts)

        LOGGER.debug('Updated sampler ({})'.format(self.__str__()))

    # --------------------------------------------------------------------------
    # select k new candidates
    # --------------------------------------------------------------------------
    def select(self, k: int, confirm_selection: bool = True) -> List[HDPoint]:

        assert isinstance(k, int)
        assert k >= 0
        if k == 0:
            return []

        self.update()

        n = len(self.candidates)
        if n == 0:
            return []

        LOGGER.debug('Selecting {} samples from ({})'.format(k, self.__str__()))

        # pick the top k candidates
        k = min(k, n)
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

        # write selections and discards
        ts = time.time()
        write_history(self.hists['selections'], selection, ts)

    # --------------------------------------------------------------------------
    # checkpoint and restore
    # --------------------------------------------------------------------------
    def checkpoint(self):

        st = format_time(time.time())
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
        np.savez(self.chkpts['data'], t=st, type=self.type, name=self.name,
                 cached=self.cached, candidates=self.candidates,
                 selected=self.selected, discarded=self.discarded)

        LOGGER.debug('Checkpointing done for Sampler')

    def restore(self):

        for k in self.chkpts.keys():
            if not os.path.isfile(self.chkpts[k]):
                LOGGER.debug('Checkpoint file {} does not exist!'
                             .format(self.chkpts[k]))
                return

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
            LOGGER.error('Failed to restore. Error = {}'.format(e))
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
