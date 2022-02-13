# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import numpy as np
import logging
from .sampler import Sampler

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# A random sampler of hd points
# ------------------------------------------------------------------------------
class SamplerRandom(Sampler):

    # --------------------------------------------------------------------------
    def __init__(self, name: str, workspace: str,
                 buffer_size: int = 0,
                 min_cands_b4_sel: int = 0):

        super().__init__(name, workspace,
                         buffer_size = buffer_size,
                         min_cands_b4_sel=min_cands_b4_sel,)
        LOGGER.info(f'Initialized {self.__tag__()}')

    # --------------------------------------------------------------------------
    # confirm the selection of these points
    def _add_selections(self, points: np.ndarray) -> None:
        pass

    # --------------------------------------------------------------------------
    # update the ranks of the candidates
    def _update_ranks(self) -> None:
        n = len(self.candidates)

        LOGGER.debug(f'Computing random ranks for ({n}) points')
        ranks = np.random.rand(n).astype(np.float32)

        # now apply these ranks to candidates
        self._apply_ranks_to_candidates(ranks)

    # --------------------------------------------------------------------------
    def checkpoint(self):
        super()._checkpoint()

    def restore(self):
        super()._restore()

    def test(self) -> bool:
        return True

# ------------------------------------------------------------------------------
