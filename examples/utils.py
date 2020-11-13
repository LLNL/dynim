# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import timeit
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator


# ------------------------------------------------------------------------------
# create a 2d dataset
# ------------------------------------------------------------------------------
def create_data():

    # create two 2d gaussian distributions
    mu1, cov1 = np.array([6, 6]), np.array([[1, 0.8], [0.8, 3]])
    mu2, cov2 = np.array([0, 0]), np.array([[2, 0], [0, 1]])

    x1 = np.random.multivariate_normal(mean=mu1, cov=cov1, size=4800).astype(np.float32)
    x2 = np.random.multivariate_normal(mean=mu2, cov=cov2, size=4800).astype(np.float32)

    # and 1 uniform random
    x3 = np.random.rand(400, 2).astype(np.float32)
    x3 *= 12
    x3 -= [5, 2]

    # combine the data
    x = np.vstack((x1, x2, x3))
    return x, x1, x2, x3


# ------------------------------------------------------------------------------
# compute kde of a sampled set
# ------------------------------------------------------------------------------
def kde(data, positions, x_shp, weights=None):
    kernel = stats.gaussian_kde(np.vstack([data[:, 0], data[:, 1]]),
                                weights=weights)
    z = np.rot90(np.reshape(kernel(positions).T, x_shp))
    return z / z.sum()


def compute_densities(samples_all, samples_dynim, weights_dynim, samples_random):

    print('> Computing density...')
    start_time = timeit.default_timer()

    n = len(samples_random)
    assert n == len(samples_dynim) and n == len(weights_dynim)

    # --------------------------------------------------------------------------
    # linearize the selection list
    dcounts = np.cumsum(np.array([len(_) for _ in samples_dynim]))
    rcounts = np.cumsum(np.array([len(_) for _ in samples_random]))
    assert np.allclose(dcounts, rcounts)

    samples_dynim = np.array([item for sublist in samples_dynim for item in sublist])
    samples_random = np.array([item for sublist in samples_random for item in sublist])

    # --------------------------------------------------------------------------
    # compute a mesh on which to compute the densities
    rng = [[samples_all[:, 0].min(), samples_all[:, 0].max()],
           [samples_all[:, 1].min(), samples_all[:, 1].max()]]
    meshx, meshy = np.mgrid[rng[0][0]:rng[0][1]:100j, rng[1][0]:rng[1][1]:100j]
    positions = np.vstack([meshx.ravel(), meshy.ravel()])

    # --------------------------------------------------------------------------
    # density of the true distribution
    kde_all = kde(samples_all, positions, meshx.shape)

    # a uniform distribution
    kde_uniform = np.ones(kde_all.shape)
    kde_uniform /= kde_uniform.sum()

    # the 4x densities we will compute for each iteration
    kde_random, kde_dynim, kde_wdynim = [], [], []
    for i in range(1, n):
        assert weights_dynim[i].shape == (dcounts[i],)

        kde_random.append(kde(samples_random[:rcounts[i]], positions, meshx.shape))
        kde_dynim.append(kde(samples_dynim[:dcounts[i]], positions, meshx.shape))
        kde_wdynim.append(kde(samples_dynim[:dcounts[i]], positions, meshx.shape,
                              weights=weights_dynim[i]))

    elapsed = timeit.default_timer() - start_time
    print('took {:.04f} s'.format(elapsed))
    return kde_all, kde_uniform, kde_dynim, kde_random, kde_wdynim, dcounts


# ------------------------------------------------------------------------------
# compute KL divergence
# ------------------------------------------------------------------------------
def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def compute_kldiv(kde_orig, kde_uniform, kde_dynim, kde_random, kde_wdynim, nsamples):

    ks = []
    klrx, klru = [], []
    klwx, kldu = [], []

    n = len(kde_dynim)
    for i in range(n):
        ks.append(nsamples[i])
        klru.append(kl_divergence(kde_random[i], kde_uniform))
        kldu.append(kl_divergence(kde_dynim[i], kde_uniform))
        klrx.append(kl_divergence(kde_random[i], kde_orig))
        klwx.append(kl_divergence(kde_wdynim[i], kde_orig))

    return np.array(ks), np.array(klwx), np.array(kldu), np.array(klrx), np.array(klru)


# ------------------------------------------------------------------------------
# plot densities on a mesh
# ------------------------------------------------------------------------------
def plot_densities(kde_true, kde_random, kde_dynim, kde_wdynim, xrng,
                   cmap=plt.cm.gist_earth_r, vmax=None):

    if vmax is None:
        vmax = max([kde_true.max(), kde_random.max(),
                    kde_dynim.max(), kde_wdynim.max()])
    crng = [0, vmax]

    def _plot_img(ax, d, title, add_cbar=False):

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)

        im = ax.imshow(d, vmin=crng[0], vmax=crng[1], cmap=cmap, extent=xrng)

        if add_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='5%', pad=0.4)
            cbar = fig.colorbar(im, cax=cax, ticks=[0, 0.0005, 0.001], orientation='horizontal')
            cbar.ax.set_xticklabels(['0', '0.0005', '0.001'])
            cbar.ax.minorticks_off()
        return im

    fig, axs = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True,
                            figsize=(12, 4))

    im = _plot_img(axs[0], kde_true, 'true distribution')
    im = _plot_img(axs[1], kde_random, 'random distribution')
    im = _plot_img(axs[2], kde_dynim, 'dynim distribution')
    im = _plot_img(axs[3], kde_wdynim, 'weighted dynim distribution')

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.48])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, 0.0005, 0.001])
    cbar.ax.set_xticklabels(['0', '0.0005', '0.001'])
    cbar.ax.minorticks_off()


# ------------------------------------------------------------------------------
# plot KL divergence
# ------------------------------------------------------------------------------
def plot_kldiv(nsamples, wdyn_true, dyn_unif, rand_true, rand_unif):

    fig, axs = plt.subplots(1, 1, figsize=(8, 4))

    axs.plot(nsamples, rand_true, c='r', label='random vs true')
    axs.plot(nsamples, rand_unif, c='r', label='random vs uniform', linestyle='--')
    axs.plot(nsamples, wdyn_true, c='g', label='weighted vs true')
    axs.plot(nsamples, dyn_unif, c='g', label='dynim vs uniform', linestyle='--')

    axs.set_ylabel('kl divergeance')
    axs.set_xlabel('number of samples selected')
    axs.set_xlim([0, nsamples.max()])
    axs.set_ylim(bottom=0)
    plt.legend()


def plot_kldiv_ensemble(x, all_klru, all_klrx, all_kldu, all_klwx):

    data = [all_klru, all_klrx, all_kldu, all_klwx]
    labels = ['(1) random vs uniform', '(2) random vs true',
              '(3) dynim vs uniform', '(4) weighted dynim vs true']
    colors = ['#de2d26', '#a50f15', '#2ca25f', '#006d2c']
    styles = ['--', '-', '--', '-']

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    for i, y in enumerate(data):
        y = np.array(y)
        m, s = np.mean(y, axis=0), np.std(y, axis=0)

        ax.fill_between(x, m - s, m + s, color=colors[i], alpha=0.5)
        ax.plot(x, m, c=colors[i], linestyle=styles[i], label=labels[i])

    ax.set_ylabel('KL divergence')
    ax.set_xlabel('number of samples selected')
    ax.set_xlim(left=-5, right=1000)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    plt.legend()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
