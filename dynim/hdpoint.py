# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import numpy as np


# ------------------------------------------------------------------------------
# A HD point used by samplers
# ------------------------------------------------------------------------------
class HDPoint:
    """A high-dimensional point."""

    def __init__(self, id: (int, str),
                 coords: np.ndarray,
                 rank:   np.float32 = np.float32('nan')) -> None:
        """Initialize a HD point.
        Args:
            id (int or string): ID
            coords (np array):  coordinates
        """
        assert isinstance(id, (int, str))
        assert isinstance(coords, np.ndarray)

        self.id = id                    # id
        self.coords = coords            # hd coordinates
        self.rank = rank                # ranking

    def __str__(self) -> str:
        return '<' + str(self.id) + ':' + str(self.rank) + '>'

    def __repr__(self) -> str:
        return self.__str__()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
