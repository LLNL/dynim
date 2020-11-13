# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import numpy as np
from .sampler import Sampler


# ------------------------------------------------------------------------------
# A random sampler of hd points
# ------------------------------------------------------------------------------
class SamplerRandom(Sampler):

    # --------------------------------------------------------------------------
    def __init__(self, name: str, workspace: str, buffer_size: int):

        self.type = 'SamplerRandom'
        super(SamplerRandom, self).__init__(name, workspace, buffer_size)

    # --------------------------------------------------------------------------
    # confirm the selection of these points
    def _add_selections(self, points: np.ndarray) -> None:
        pass

    # --------------------------------------------------------------------------
    # update the ranks of the candidates
    def _update_ranks(self) -> None:
        for sample in self.candidates:
            sample.rank = np.random.uniform()

    # --------------------------------------------------------------------------
    def test(self) -> bool:
        return True

# ------------------------------------------------------------------------------
