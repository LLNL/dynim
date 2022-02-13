# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################

import numpy as np
import logging
LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
class BinningUtils:

    # --------------------------------------------------------------------------
    @staticmethod
    def create_cartesian_bins_1d(nxbins, xrange):
        return np.linspace(xrange[0], xrange[1], nxbins + 1)

    @staticmethod
    def create_cartesian_bins_2d(nxbins, nybins, xrange, yrange):
        xbins = BinningUtils.create_cartesian_bins_1d(nxbins, xrange)
        ybins = BinningUtils.create_cartesian_bins_1d(nybins, yrange)
        return xbins, ybins

    # --------------------------------------------------------------------------
    @staticmethod
    def create_spherical_bins():
        alt_bin_edges = np.deg2rad(np.array([0, 6, 18, 30, 42, 54, 66, 78, 90,
                                             102, 114, 126, 138, 150, 162, 174,
                                             180]))
        nbins_az = np.array([1, 6, 12, 18, 24, 28, 30, 32, 32,
                             30, 28, 24, 18, 12, 6, 1])

        az_bin_edges = [np.linspace(-np.pi, np.pi, n + 1) for n in nbins_az]
        return alt_bin_edges, az_bin_edges

    # --------------------------------------------------------------------------
    @staticmethod
    def create_polar_bins(nrad_bins, rad_range,
                          tht_range=np.array([-np.pi, np.pi]),
                          kfactor=1, do_verify=False):

        assert isinstance(nrad_bins, int)
        assert isinstance(rad_range, np.ndarray)
        assert isinstance(tht_range, np.ndarray)
        assert isinstance(kfactor, int)
        assert rad_range.shape == (2,)
        assert tht_range.shape == (2,)
        assert nrad_bins > 1
        assert kfactor > 0
        assert tht_range[0] >= -np.pi and tht_range[1] <= np.pi

        # compute radius bins
        rad_bin_edges = np.linspace(rad_range[0], rad_range[1], nrad_bins + 1)

        # split r'th radial bin into k*(2*r + 1) theta bins (gives equal area bins)
        tht_bin_edges = [np.linspace(tht_range[0], tht_range[1], kfactor*(2*r+1)+1)
                         for r in range(nrad_bins)]

        if do_verify:
            areas = []
            for ridx in range(nrad_bins):
                for tidx in range(tht_bin_edges[ridx].shape[0] - 1):
                    r0, r1 = rad_bin_edges[ridx], rad_bin_edges[ridx + 1]
                    t0, t1 = tht_bin_edges[ridx][tidx], tht_bin_edges[ridx][tidx + 1]

                    # area: 1/2 \theta (r2*r2 - r1*r1)
                    areas.append(0.5 * (t1 - t0) * (r1 * r1 - r0 * r0))
            assert np.isclose(min(areas), max(areas))

        return rad_bin_edges, tht_bin_edges

    # --------------------------------------------------------------------------
    @staticmethod
    def collect_bins(bin_edges0, bin_edges1):

        assert isinstance(bin_edges0, np.ndarray)
        assert isinstance(bin_edges1, list)
        assert bin_edges0.shape == (len(bin_edges1)+1,)
        assert all([isinstance(b, np.ndarray) for b in bin_edges1])

        # create four arrays to mark 4 corners of the bins
        bin_edges = []
        for ridx in range(bin_edges0.shape[0] - 1):

            bedges10 = bin_edges1[ridx]
            for tidx in range(bedges10.shape[0] - 1):
                bin_edges.append([bin_edges0[ridx], bin_edges0[ridx + 1],
                                  bedges10[tidx],   bedges10[tidx + 1]])

        return np.array(bin_edges)

    # --------------------------------------------------------------------------
    @staticmethod
    def get_bin_indices_1d(data, bin_edges, num_out_of_range_bins):

        assert isinstance(data, np.ndarray)
        assert isinstance(bin_edges, np.ndarray)
        assert len(data.shape) == 1
        assert len(bin_edges.shape) == 1
        assert isinstance(num_out_of_range_bins, int)
        assert num_out_of_range_bins in [0, 1, 2]

        LOGGER.info(f'Computing 1d bin indices for {data.shape} data '
                    f'(boundary type = {num_out_of_range_bins})')

        # empty data?
        if data.shape[0] == 0:
            return np.array([]).astype(np.uint8)

        # digitize returns an index for each data value, such that
        # bin_edges[i-1] <= x < bin_edges[i]
        # for x  < bin_edges[0],  i = 0                    (left out-of-range bin)
        # for x >= bin_edges[-1], i = bin_edges.shape + 1  (right out-of-range bin)
        bin_idxs = np.digitize(data, bin_edges)
        nbins = bin_edges.shape[0] - 1

        # remove out-of-range bins
        if num_out_of_range_bins == 0:
            assert bin_idxs.min() > 0 and bin_idxs.max() <= nbins
            bin_idxs -= 1

        # merge the left out-of-range bin into the right one
        elif num_out_of_range_bins == 1:
            assert bin_idxs.min() >= 0 and bin_idxs.max() <= nbins + 1
            bin_idxs[np.where(bin_idxs == 0)[0]] = nbins + 1
            bin_idxs -= 1

        # keep both out-of-range bins
        elif num_out_of_range_bins == 2:
            assert bin_idxs.min() >= 0 and bin_idxs.max() <= nbins + 1

        return bin_idxs.astype(np.uint8)

    @staticmethod
    def get_bin_indices_2d_spherical(data, bins_altitude, bins_azimuth):

        assert len(data.shape) == 2
        assert data.shape[1] == 2
        LOGGER.info(f'Computing 2d-spherical bin indices for {data.shape} data')

        ndata = data.shape[0]
        data_altitude, data_azimuthal = data[:, 0], data[:, 1]

        # altitude :: remove both out-of-range bins ([0,90] only)
        idxs_t = BinningUtils.get_bin_indices_1d(data_altitude, bins_altitude, 0)

        # azimuthal :: compute the array of bins
        idxs_r = np.array([np.nan for _ in range(ndata)])
        for tidx in range(bins_altitude.shape[0] - 1):
            # find the bins for data2 "with respect to" the bin in data1
            # these are the elements in bin "idx1" wrt column 1
            eidx = np.where(idxs_t == tidx)[0]

            # idxs_r for these elements should not be set
            assert np.all(np.isnan(idxs_r[eidx]))

            # compute the bin idxs
            idxs_r[eidx] = BinningUtils.get_bin_indices_1d(data_azimuthal[eidx],
                                                           bins_azimuth[tidx], 0)

        idxs_r = idxs_r.astype(np.uint8)
        return idxs_t, idxs_r

    # --------------------------------------------------------------------------
    @staticmethod
    def compute_histogram_from_bin_indices(bin_idxs, nbins):
        LOGGER.info(f'Computing histogram '
                    f'for {bin_idxs.shape} bin indices and {nbins} bins')
        hist = np.zeros(nbins, dtype=np.uint)
        vals, counts = np.unique(bin_idxs, return_counts=True)
        hist[vals] = counts
        return hist

    # --------------------------------------------------------------------------
    @staticmethod
    def plot_polar_histogram(ax, bin_corners, values, cmap='Purples', do_log=False):

        import matplotlib

        assert isinstance(ax, matplotlib.axes.Axes)
        assert isinstance(bin_corners, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert isinstance(do_log, bool)
        assert isinstance(cmap, (str, matplotlib.colors.Colormap))
        assert len(bin_corners.shape) == 2
        assert len(values.shape) == 1
        assert bin_corners.shape[1] == 4
        assert bin_corners.shape[0] == values.shape[0]

        # create colors
        if isinstance(cmap, str):
            cmap = matplotlib.cm.get_cmap(cmap)

        if do_log:
            norm = matplotlib.colors.LogNorm(1, values.max())
        else:
            norm = matplotlib.colors.Normalize(1, values.max())

        masked_values = np.ma.masked_equal(values, 0)
        cmap.set_bad(alpha = 0)
        colors = cmap(norm(masked_values))

        # plot stacked bars
        r0, r1 = bin_corners[:,0], bin_corners[:,1]
        t0, t1 = bin_corners[:,2], bin_corners[:,3]
        dt = t1-t0
        ax.bar(x = t0 + dt, height = r1-r0, width = dt, bottom = r0,
               color=colors) #, edgecolor='#bbbbbb', linewidth=0.5)
               #color=colors, edgecolor=colors, linewidth=0.5)

        # customize axes
        ax.grid(False)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                          labels=['0', '45', '90', '135', '180', '-135', '-90', '-45'])

        ax.set_rticks([])
        ax.set_rlim([r0[0], r1[-1]])
        ax.set_rlabel_position(180)

        # add colorbar
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        matplotlib.pyplot.colorbar(sm, orientation="horizontal", pad=0.05)

    # --------------------------------------------------------------------------
    @staticmethod
    def plot_polar_scatter(ax, rad, tht, rad_rng = None):

        import matplotlib

        assert isinstance(ax, matplotlib.axes.Axes)
        assert isinstance(rad, np.ndarray)
        assert isinstance(tht, np.ndarray)

        assert tht.min() >= -np.pi and tht.max() <= np.pi
        assert rad.min() >= 0

        if rad_rng is None:
            rad_rng = np.array([rad.min(), rad.max()])
        else:
            assert isinstance(rad_rng, np.ndarray)
            assert rad_rng.shape == (2,)
            assert rad.min() >= rad_rng[0] and rad.max() <= rad_rng[1]

        # now, plot!
        ax.scatter(tht, rad, s=1, alpha=0.1)

        # set up the axes!
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                          labels=['0', '45', '90', '135', '180', '-135', '-90', '-45'])

        ax.set_rticks([])
        ax.set_rlim(rad_rng)
        ax.set_rlabel_position(180)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
