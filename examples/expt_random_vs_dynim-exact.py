#!/usr/bin/env python3

# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import os
import sys
import glob
import time
import timeit
import numpy as np
from matplotlib import pyplot as plt

import dynim
from utils import create_data, compute_densities, compute_kldiv
from utils import plot_kldiv, plot_densities, plot_kldiv_ensemble


# ------------------------------------------------------------------------------
# compute the random sampling
# ------------------------------------------------------------------------------
def draw_random_samples(x, niters, ksamples):

    print('> Computing random sampling: {} iterations of {} samples each...'
          .format(niters, ksamples))
    start_time = timeit.default_timer()

    # draw k samples each for n iterations
    nsampled = 0
    sampled = []
    weights = []
    xi = np.arange(x.shape[0])
    for i in range(niters):
        xidx = np.random.choice(xi, size=ksamples, replace=False)

        sampled.append(np.array(x[xidx]))
        nsampled += len(xidx)

        w = x.shape[0]/nsampled
        weights.append(w*np.ones(nsampled, dtype=np.float32))

        # delete these points from the list
        xi = np.setdiff1d(xi, xidx)

    elapsed = timeit.default_timer() - start_time
    print('took {:.04f} s'.format(elapsed))

    return sampled, weights


# ------------------------------------------------------------------------------
# compute the dynim sampling
# ------------------------------------------------------------------------------
def draw_dynimexact_samples(x, niters, ksamples):

    print('> Computing dynim sampling: {} iterations of {} samples each...'
          .format(niters, ksamples))
    start_time = timeit.default_timer()

    # create an hdspace and a dynim instance
    h = dynim.HDSpace()
    h.setup(dim=x.shape[1], idx_type='exact')

    sampler = dynim.SamplerImportance('dynim', 'test_workspace',
                                      buffer_size=x.shape[0] + 1,
                                      n_minsamples=ksamples)
    sampler.set_hdspace(h)

    # convert the data to dynim hdpoints
    pnts = [dynim.HDPoint(id=i, coords=x[i].astype(np.float32))
            for i in range(x.shape[0])]
    sampler.add_candidates(pnts)

    # draw k samples each for n iterations
    sampled = []
    weights = []
    for i in range(niters):
        selected = sampler.select(ksamples)     # returns a list of HD points
        xidx = [_.id for _ in selected]         # we need only the ids

        sampled.append(np.array(x[xidx]))
        weights.append(sampler.get_weights())
    elapsed = timeit.default_timer() - start_time
    print('took {:.04f} s'.format(elapsed))

    return sampled, weights


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def do_random_vs_dynim_exact(niters, ksamples, filename):
    """
    Compare random sampling with dynim sampling with exact L2 metric.

    (1)  create a random dataset (2 gaussians + 1 uniform)
    (2a) draw random samples
    (2b) draw dynim samples
    (3)  compute kde for all distributions
    (4)  write as npz
    """

    x, x1, x2, x3 = create_data()
    rsampled, rweights = draw_random_samples(x, niters, ksamples)
    dsampled, dweights = draw_dynimexact_samples(x, niters, ksamples)

    kde_x, kde_u, kde_d, kde_r, kde_dw, ncounts = compute_densities(x,
                                                                    dsampled,
                                                                    dweights,
                                                                    rsampled)

    np.savez_compressed(filename, true=x,
                        random=rsampled, random_weights=rweights,
                        dynim=dsampled, dynim_weights=dweights,
                        kde_x=kde_x, kde_u=kde_u,
                        kde_d=kde_d, kde_r=kde_r, kde_dw=kde_dw,
                        ncounts=ncounts)


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: {} [run_mode = expt / plot_each / plot_ensemble]'
              .format(sys.argv[0]))
        exit(1)

    run_mode = sys.argv[1]
    assert run_mode in ['expt', 'plot_each', 'plot_ensemble']

    DATA_PATH = './data_expts'

    # --------------------------------------------------------------------------
    # perform a (single, new) experiment
    if run_mode == 'expt':
        n, k = 500, 2
        filename = 'expt_{}.npz'.format(time.strftime("%Y-%m-%d-%H-%M"))
        filename = os.path.join(DATA_PATH, filename)
        do_random_vs_dynim_exact(n, k, filename)
        exit(0)

    # --------------------------------------------------------------------------
    # plot each experiment one by one
    if run_mode == 'plot_each':

        # which time-step to plot for each file?
        ts = -1

        # get all the available expt files
        expt_files = sorted(glob.glob(os.path.join(DATA_PATH, 'expt*npz')))
        for filename in expt_files:

            data = np.load(filename, allow_pickle=True)
            x = data['true']
            kde_x = data['kde_x']
            kde_u = data['kde_u']
            kde_d = data['kde_d']
            kde_r = data['kde_r']
            kde_dw = data['kde_dw']
            ncounts = data['ncounts']

            if 1:
                ks, klwx, kldu, klrx, klru = compute_kldiv(kde_x, kde_u,
                                                           kde_d, kde_r, kde_dw,
                                                           ncounts)
                plot_kldiv(ks, klwx, kldu, klrx, klru)
                plt.savefig(filename[:-4]+'_kld.pdf', bbox_inches='tight')

            if 1:
                xrng = [x[:, 0].min(), x[:, 0].max(),
                        x[:, 1].min(), x[:, 1].max()]
                plot_densities(kde_x, kde_r[ts], kde_d[ts], kde_dw[ts], xrng)
                plt.savefig(filename[:-4] + '_kde_t{}.pdf'.format(ts),
                            bbox_inches='tight')

            plt.close()
        exit(0)

    # --------------------------------------------------------------------------
    # plot the kld of the ensemble
    if run_mode == 'plot_ensemble':

        all_ks = []
        all_klwx = []
        all_kldu = []
        all_klrx = []
        all_klru = []

        expt_files = sorted(glob.glob(os.path.join(DATA_PATH, 'expt*npz')))
        for filename in expt_files:

            data = np.load(filename, allow_pickle=True)
            kde_x = data['kde_x']
            kde_u = data['kde_u']
            kde_d = data['kde_d']
            kde_r = data['kde_r']
            kde_dw = data['kde_dw']
            ncounts = data['ncounts']

            ks, klwx, kldu, klrx, klru = compute_kldiv(kde_x, kde_u,
                                                       kde_d, kde_r, kde_dw,
                                                       ncounts)

            all_ks.append(ks)
            all_klwx.append(klwx)
            all_kldu.append(kldu)
            all_klrx.append(klrx)
            all_klru.append(klru)

        # ----------------------------------------------------------------------
        plot_kldiv_ensemble(all_ks[0], all_klru, all_klrx, all_kldu, all_klwx)
        plt.savefig(os.path.join(DATA_PATH, 'expt_all_kld.pdf'),
                    bbox_inches='tight')
        exit(0)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
