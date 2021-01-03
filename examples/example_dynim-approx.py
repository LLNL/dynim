#!/usr/bin/env python3

# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import logging
LOGGER = logging.getLogger()

import numpy as np
import dynim
from utils import create_data


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == '__main__':

    # --------------------------------------------------------------------------
    # initialize the logger
    FMT_LOG = '%(asctime)s - %(name)s:%(funcName)s:%(lineno)s - %(levelname)s - %(message)s'
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(FMT_LOG))

    LOGGER.setLevel(logging.DEBUG)
    LOGGER.addHandler(sh)

    # --------------------------------------------------------------------------
    # create a fake 2D dataset
    x, _, _, _ = create_data()      # only the first return value is relevant
    LOGGER.info('Created {} data'.format(x.shape))

    # let's split this data into three parts
    ndata, dim = x.shape[0], x.shape[1]
    a,b = ndata//2, ndata//2+ndata//4
    x1, x2, x3 = x[:a], x[a:b], x[b:]

    LOGGER.info('Split the data: {}, {}, and {}'.format(x1.shape, x2.shape, x3.shape))

    # --------------------------------------------------------------------------
    # here, we will create a dynim hdspace with approx metric
    # we will train this hdspace using some data and save (as a faiss index)

    # nlist and nprobes are the key parameters that define
    # how good the approximation is. see, e.g.,
    # https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    # https://github.com/facebookresearch/faiss/wiki/Faster-search
    h1 = dynim.HDSpace()
    h1.setup(dim=dim, idx_type='approx', nlist=100, nprobes=50)
    h1.train(x1)
    h1.checkpoint("./hdspace.idx")
    del h1

    # --------------------------------------------------------------------------
    # now, let's load this index file and add some points it
    h2 = dynim.HDSpace()
    h2.restore("./hdspace.idx")

    # add some points to this hdspace
    h2.add_points(x2)
    LOGGER.info('added {} points to hdspace: {}'.format(x2.shape, h2))

    # we could checkpoint the hdspace with points added, if we needed
    # h2.checkpoint("./hdspace.idx")

    # --------------------------------------------------------------------------
    # finally, let's make some queries to this hdspace

    # let's find the nearest neighbors for the remaining set of points
    # the functions returns [k0, k+k0] nearest neighbors
    # use k0=0 if query points (x3) have not been "added" to the hdspace
    # otherwise, use k0=1 to ignore the nearest neighbor (i.e., self)
    nn = h2.get_knn(x3, k=2, k0=1)

    # a shortcut exists if only the mean distance to knn is needed!
    ndist = h2.get_knn_distance(x3, k=2, k0=1)

    # the return value is a list of n tuples (n = number of query points)
    # each tuple contains (id, dist)
    # id    = a (k,1) ndarray of ids of nearest neighbors
    # dists = a (k,1) ndarray of dists of nearest neighbors
    # size may be smaller in case the system could not find k neighbors
    # either due to not having enough points or approximation error
    for i in range(x3.shape[0]):
        _nids = nn[i][0]
        _ndists = nn[i][1]

        print('knn of {} are: ids = {}, dists = {}; mean_dist = {}'
              .format(x3[i], _nids, _ndists, ndist[i]))

        if len(_ndists) > 0:
            assert np.isclose(_ndists.mean(), ndist[i])

# ------------------------------------------------------------------------------
