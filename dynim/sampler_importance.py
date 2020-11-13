# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import os
import logging
import numpy as np

from .hdspace import HDSpace
from .sampler import Sampler
from .utils import take_backup

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# An importance sampler of hd points
# ------------------------------------------------------------------------------
class SamplerImportance(Sampler):

    # --------------------------------------------------------------------------
    def __init__(self, name: str, workspace: str,
                 buffer_size: int, n_minsamples: int):

        """ Initialize a HDSpace Sampler.
        Args:
            name (str):                 a name given to this sampler
            workspace (str):            a directory where history and checkpoints will be written
            n_minsamples (int):         number of random selections to make before switching to hdspace
            buffer_size (int):  size of the candidate buffer
        """
        assert isinstance(n_minsamples, int) and int(n_minsamples) >= 0

        self.type = 'SamplerImportance'
        super(SamplerImportance, self).__init__(name, workspace, buffer_size)

        self.nMinSamples = n_minsamples
        self.hdspace = None
        self.chkpt = os.path.join(self.workspace, '{}.lspace.idx'.format(self.tag))

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
    def _add_selections(self, points: np.ndarray) -> None:

        coords = np.array([p.coords for p in points])
        self.hdspace.add_points(coords)

    # --------------------------------------------------------------------------
    # update the ranks of the candidates
    def _update_ranks(self, k=10) -> None:

        n = len(self.candidates)

        # for the first few samples, just use random sampling
        if len(self.selected) < self.nMinSamples:
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
    def get_weights(self):

        LOGGER.info('Computing weights: {}'.format(self))

        weights = np.ones(len(self.selected), dtype=np.float32)

        cands = [s.coords for s in self.candidates]
        cands += [s.coords for s in self.cached]
        cands += [s.coords for s in self.discarded]

        nnids = self.hdspace.get_nn_id(np.array(cands))
        for n in nnids:
            weights[n] += 1.

        LOGGER.info('Computed weights: shape = {}, sum = {}'
                    .format(weights.shape, weights.sum()))
        return weights

    # --------------------------------------------------------------------------
    def checkpoint(self):
        super(SamplerImportance, self).checkpoint()

        take_backup(self.chkpt)
        self.hdspace.checkpoint(self.chkpt)

    def restore(self):
        super(SamplerImportance, self).restore()

        self.hdspace.restore(self.chkpt)

    # --------------------------------------------------------------------------
    def test(self):
        return True

# ------------------------------------------------------------------------------
