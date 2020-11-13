#!/usr/bin/env python3

# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import unittest
from os.path import dirname, realpath, join
import numpy as np
import dynim


# ------------------------------------------------------------------------------
class TestDynim(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.workspace = join(dirname(realpath(__file__)), 'test_workspace')
        self.samples = [dynim.HDPoint(i, np.random.rand(2).astype(np.float32))
                        for i in range(20)]
        super(TestDynim, self).__init__(*args, **kwargs)

    def test_sampler_random(self):
        sampler = dynim.SamplerRandom('test_id', self.workspace, 100)
        sampler.add_candidates(self.samples)
        self.assertTrue(len(sampler.select(5)) == 5)

    def test_sampler_importance(self):
        training_samples = np.random.rand(5000, 2).astype(np.float32)
        hdspace = dynim.HDSpace()
        hdspace.setup(2)
        hdspace.train(training_samples)

        sampler = dynim.SamplerImportance('test_id', self.workspace, 100, 10)
        sampler.set_hdspace(hdspace)
        sampler.add_candidates(self.samples)
        self.assertTrue(len(sampler.select(5)) == 5)


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()

# ------------------------------------------------------------------------------
