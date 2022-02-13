# Copyright 2020 Lawrence Livermore National Security, LLC and other
# DynIm Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

################################################################################


import os
import numpy as np
import logging
from typing import List

from .hdspace import HDSpace
from .hdpoint import HDPoint
from .sampler import Sampler
from .binning import BinningUtils
from .utils import take_backup, backup_fname, files_exist

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# An importance sampler of hd points
# ------------------------------------------------------------------------------
class SamplerBinned(Sampler):

    KNOWN_BINNING_TYPES = ['1d-nonperiodic',
                           '2d-spherical',
                           '3d-spherical_and_cartesian']
    KNOWN_RANKING_TYPES = ['static', 'dynamic']

    def __init__(self, name: str, workspace: str,
                 novelty_factor: float,
                 binning_type: str,
                 min_cands_b4_sel: int = 0):

        self.ranking_type = 'dynamic'
        self.binning_type = binning_type
        assert self.ranking_type in SamplerBinned.KNOWN_RANKING_TYPES
        assert self.binning_type in SamplerBinned.KNOWN_BINNING_TYPES

        super().__init__(name, workspace, min_cands_b4_sel = min_cands_b4_sel)

        # ----------------------------------------------------------------------
        # --- set up the binned sampler
        # novelty_factor = 1:       complete novelty (flat curve)
        # novelty_factor = 0:       complete random (mimic distribution)
        # linear interpolation between the two!
        assert isinstance(novelty_factor, float)
        assert 0. <= novelty_factor <= 1.0
        self.novelty_factor = novelty_factor

        # for hdspace ranking within bins
        self.hdspace = None
        self.ranking_dim = 0
        self.lchkpt = os.path.join(self.workspace, f'{self.__tag__()}.lspace.idx')

        # ----------------------------------------------------------------------
        self.cand_histo, self.cand_binidxs = None, None
        self.sel_histo, self.sel_binidxs = None, None

        # ----------------------------------------------------------------------
        _s = 'unknown'
        if self.binning_type == '1d-nonperiodic':
            # dimer dists are measured from mid points
            # so add some margin
            xrange = [0, np.sqrt(2.)*15. + 3.]
            self.bins_x = BinningUtils.create_cartesian_bins_1d(5, xrange)
            self.nbins_x = self.bins_x.shape[0]-1

            self.nbins = self.nbins_x
            _s = f'x={self.nbins_x}'
            self.binning_dim = 1

            self.distribute_zero_map = {}
            for b in range(self.nbins):
                self.distribute_zero_map[b] = []
                for db in range(1,self.nbins):
                    b2a,b2b = b-db,b+db
                    if b2a >= 0 and b2b < self.nbins:
                        self.distribute_zero_map[b].append([b2a,b2b])
                    elif b2b < self.nbins:
                        self.distribute_zero_map[b].append([b2b])
                    elif b2a >= 0:
                        self.distribute_zero_map[b].append([b2a])

        # ----------------------------------------------------------------------
        elif self.binning_type in ['2d-spherical', '3d-spherical_and_cartesian']:

            # spherical bins = (bins_altitude, bins_azimuth)
            self.bins_altitude, self.bins_azimuth = BinningUtils.create_spherical_bins()
            self.nbins_altitude = self.bins_altitude.shape[0] - 1
            self.nbins_azimuth = [b.shape[0] - 1 for b in self.bins_azimuth]

            # count the number of azimuth bins prior to each altitude bin
            self.cbins_azimuth = np.concatenate((np.zeros(1), np.cumsum(self.nbins_azimuth))).astype(np.uint)
            assert sum(self.nbins_azimuth) == self.cbins_azimuth[-1]

            # that's it for 2d binning
            if self.binning_type == '2d-spherical':
                self.binning_dim = 2
                self.nbins = np.uint(self.cbins_azimuth[-1])
                _s = f'altitude={self.nbins_altitude}, azimuth={self.nbins_azimuth}'

            # 3d binning needs a z dimension also
            elif self.binning_type == '3d-spherical_and_cartesian':
                self.binning_dim = 3
                # then there will be two out of range bins: [< min] and [> max]
                self.bins_z = np.array([1.5, 2., 2.5, 3.])
                self.nbins_z = 2 + (self.bins_z.shape[0] - 1)

                self.nbins = np.uint(self.nbins_z * self.cbins_azimuth[-1])
                _s = f'z={self.nbins_z}, altitude={self.nbins_altitude}, azimuth={self.nbins_azimuth}'

        # ----------------------------------------------------------------------
        LOGGER.info(f'Initialized SamplerBinned({self.name}, {self.binning_type}, '
                    f'novelty={self.novelty_factor}) with '
                    f'{self.nbins} bins: ({_s})')

    # --------------------------------------------------------------------------
    def set_hdspace(self, data: (str, HDSpace)) -> None:

        assert isinstance(data, (str, HDSpace))
        if isinstance(data, HDSpace):
            self.hdspace = data
        else:
            self.hdspace = HDSpace()
            self.hdspace.restore(data)

        self.ranking_dim = self.hdspace.dim
        LOGGER.info(f'Assigned a {self.ranking_dim}-D hdspace for ranking')

    # --------------------------------------------------------------------------
    def _fetch_coords(self, points: List[HDPoint]):
        coords = np.array([p.coords for p in points])
        assert self.binning_dim + self.ranking_dim == coords.shape[1]
        if self.hdspace is None:
            return None, coords
        else:
            return np.array(coords[:, :-self.binning_dim]), \
                   np.array(coords[:, -self.binning_dim:])

    # --------------------------------------------------------------------------
    # add these selections to hdspace
    def _add_selections(self, points: List[HDPoint]) -> None:
        if len(points) == 0 or self.hdspace is None:
            return
        rcoords, bcoords = self._fetch_coords(points)
        self.hdspace.add_points(np.array(rcoords))

    # --------------------------------------------------------------------------
    def _update_ranks(self, k=10):

        nselections = len(self.selected)

        LOGGER.debug(f'Updating ranks: {self}')
        # ----------------------------------------------------------------------
        # Step 1: assign a global rank to all elements
        ccoords_ranking, ccoords_binning = self._fetch_coords(self.candidates)
        if ccoords_ranking is None:
             LOGGER.debug(f'fetched coordinates: {ccoords_binning.shape}')
        else:
            LOGGER.debug(f'fetched coordinates: {ccoords_ranking.shape}, {ccoords_binning.shape}')

        if self.hdspace is None or len(self.selected) == 0:
            ncandidates = len(self.candidates)
            LOGGER.debug(f'Computing random ranks for {ncandidates} points')
            cand_ranks = np.random.rand(ncandidates).astype(np.float32)
        else:
            LOGGER.debug(f'Computing importance ranks for {ccoords_ranking.shape} points')
            cand_ranks = self.hdspace.get_knn_distance(ccoords_ranking, k, 0).astype(np.float32)
            assert cand_ranks.max() >= 0, f'{self}: ranks cannot be negative'

        LOGGER.debug(f'cand_ranks: {cand_ranks.shape}, [{cand_ranks.min()}, {cand_ranks.max()}]')

        # ----------------------------------------------------------------------
        # Step 2: compute the histogram and bin_idxs
        #               for current candidates and previous selections
        LOGGER.profile('computing histogram of candidates')
        self.cand_histo, self.cand_binidxs = self.compute_histogram(ccoords_binning)
        LOGGER.debug(f'histogram of candidates: {self.cand_histo}')
        LOGGER.profile('computed histogram of candidates')

        # ----------------------------------------------------------------------
        # Step 3: compute the chunks (static or dynamic)
        if self.ranking_type == 'static':
            chunk_edges = SamplerBinned.create_chunks_static(self.cand_histo, self.novelty_factor)

        else:
            LOGGER.profile('computed histogram')
            if nselections == 0:
                self.sel_histo = np.zeros_like(self.cand_histo)
                self.sel_binidxs = None
            else:
                LOGGER.profile('computing histogram of selections')
                srcoords, sbcoords = self._fetch_coords(self.selected)
                self.sel_histo, self.sel_binidxs = self.compute_histogram(sbcoords)
                LOGGER.profile('computed histogram of selections')

            LOGGER.profile('computing dynamic chunking')
            chunk_edges = SamplerBinned.create_chunks_dynamic(self.cand_histo,
                                                              self.sel_histo,
                                                              self.novelty_factor)
            LOGGER.profile('computed dynamic chunking')

        # ----------------------------------------------------------------------
        # Step 4: update the ranks to incorporate chunk ids
        # basically, we will manipulate the ranks so that
        # global ranks preserve a chunk ordering
        # i.e., all elements in chunk 0 across all bins have higher rank
        # than any element in chunk 1 in any bin
        nbins, nchunks = chunk_edges.shape[0], chunk_edges.shape[1]-1

        rank_offset = cand_ranks.max() + cand_ranks.min()
        LOGGER.info(f'Updating ranks with offset={rank_offset}')
        LOGGER.profile('updating ranks')
        for bin_idx in range(nbins):

            elements_in_bin = np.where(self.cand_binidxs == bin_idx)[0]
            if elements_in_bin.shape[0] == 0:
                continue

            chunks_for_bins = chunk_edges[bin_idx]
            nchnks = chunks_for_bins.shape[0] - 1

            LOGGER.debug(f'bin {bin_idx} has {elements_in_bin.shape[0]} elements')
            LOGGER.debug(f'bin {bin_idx} has {chunks_for_bins} chunks')

            # descending sort of the global ranks in this bin
            rank_order_in_bin = np.argsort(-1 * cand_ranks[elements_in_bin])
            elements_bin_in_rank_order = elements_in_bin[rank_order_in_bin]

            # now go over each chunk in the rank order and update the ranks
            # for a chunk c, the rank becomes rank + (nchunks-c-1)*rank_offset
            # this splits the ranks into chunks,
            #   i.e., all elements in a given chunk (across all bins)
            #   have higher ranks than the next chunk
            for cidx in range(nchnks):
                c0,c1 = chunks_for_bins[cidx], chunks_for_bins[cidx+1]
                elements_in_chunk = elements_bin_in_rank_order[c0:c1]
                cand_ranks[elements_in_chunk] += (nchunks-cidx-1)*rank_offset

            LOGGER.debug(f'ranks_in_bins: {cand_ranks[elements_in_bin].shape}, '
                         f'[{cand_ranks[elements_in_bin].min()}, '
                         f'{cand_ranks[elements_in_bin].max()}]')

        # ----------------------------------------------------------------------
        LOGGER.debug(f'ranks: {cand_ranks.shape}, [{cand_ranks.min()}, {cand_ranks.max()}]')
        LOGGER.profile('updated ranks')

        # Step 5: simply assign these ranks to the hdpoints
        self._apply_ranks_to_candidates(cand_ranks)
        LOGGER.info('assigned all ranks!')

    # --------------------------------------------------------------------------
    def compute_histogram(self, data):

        if self.binning_type == '1d-nonperiodic':
            return self.compute_histograms_1d_nonperiodic(data)

        if self.binning_type == '2d-spherical':
            return self.compute_histograms_2d_spherical(data)

        elif self.binning_type == '3d-spherical_and_cartesian':
            return self.compute_histograms_3d_spherical_and_cartesian(data)

    # --------------------------------------------------------------------------
    def compute_histograms_1d_nonperiodic(self, data):

        assert len(data.shape) == 2
        assert data.shape[1] == 1

        dmin, dmax = data[:,0].min(), data[:,0].max()
        bmin, bmax = self.bins_x[0], self.bins_x[-1]
        assert dmin >= self.bins_x[0],  f'data ({dmin}) < bin range ({bmin})'
        assert dmax <= self.bins_x[-1], f'data ({dmax}) > bin range ({bmax})'

        bin_idxs = BinningUtils.get_bin_indices_1d(data[:,0], self.bins_x, 0)

        # now compute 1-d histogram using these linearized bin idxs
        hist = BinningUtils.compute_histogram_from_bin_indices(bin_idxs, self.nbins)
        return hist, bin_idxs

    # --------------------------------------------------------------------------
    def compute_histograms_2d_spherical(self, data):

        assert len(data.shape) == 2
        assert data.shape[1] == 2

        # get the bin indices for spherical binning
        idxs_t, idxs_r = \
            BinningUtils.get_bin_indices_2d_spherical(data[:,:2],
                                                      self.nbins_altitude,
                                                      self.nbins_azimuth)

        # linearize the bin indices across the two dimensions!
        def linearize_bins(t, r):
            return np.uint(self.cbins_azimuth[t] + r)

        bin_idxs = [linearize_bins(idxs_t[i], idxs_r[i]) for i in range(data.shape[0])]
        bin_idxs = np.array(bin_idxs)
        assert bin_idxs.max() < self.nbins

        # now compute 1-d histogram using these linearized bin idxs
        hist = BinningUtils.compute_histogram_from_bin_indices(bin_idxs, self.nbins)
        return hist, bin_idxs

    # --------------------------------------------------------------------------
    def compute_histograms_3d_spherical_and_cartesian(self, data):

        assert len(data.shape) == 2
        assert data.shape[1] == 3

        # get the bin indices for altitude and azimuth
        # using spherical binning
        idxs_t, idxs_r = \
            BinningUtils.get_bin_indices_2d_spherical(data[:,:2],
                                                      self.bins_altitude,
                                                      self.bins_azimuth)

        # get bin indices for height
        # maintain the left and right out-of-range bins separately
        idxs_d = BinningUtils.get_bin_indices_1d(data[:, 2], self.bins_z, 2)

        #  linearize the bin indices across the three dimensions!
        def linearize_bins(t, r, d):
            return np.uint(d * self.cbins_azimuth[-1] + self.cbins_azimuth[t] + r)

        bin_idxs = [linearize_bins(idxs_t[i], idxs_r[i], idxs_d[i]) for i in range(data.shape[0])]
        bin_idxs = np.array(bin_idxs)
        assert bin_idxs.max() < self.nbins

        # now compute 1-d histogram using these linearized bin idxs
        hist = BinningUtils.compute_histogram_from_bin_indices(bin_idxs, self.nbins)
        return hist, bin_idxs

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    def get_weights(self, normalize=True):

        assert '1d-nonperiodic' == self.binning_type and 1 == self.binning_dim,\
               'Implemented only for 1D'

        assert self.test()

        nsel = len(self.selected)
        ncands = len(self.candidates)
        if nsel == 0:
            return [], []

        LOGGER.info(f'Computing weights {self}')

        self._update_ranks()

        LOGGER.info(f'cand_binidxs: {self.cand_binidxs}')
        LOGGER.info(f'sel_binidxs:  {self.sel_binidxs}')
        assert nsel == len(self.sel_binidxs)
        assert ncands == len(self.cand_binidxs)

        ids = np.array([s.id for s in self.selected])

        if (self.cand_binidxs is None) or (self.sel_binidxs is None):
            weights = np.ones(len(self.selected), dtype=np.float32)

        else:
            # compute the number of candidates selected from each bin
            def number_in_bins(_, _tag):
                _n_in_bins = np.zeros(self.nbins, dtype=np.int)
                _u,_n = np.unique(_, return_counts=True)
                _n_in_bins[_u] = _n
                assert _n_in_bins.sum() == _.shape[0], f'mismatch in {_tag}: {_n_in_bins}, {_.shape[0]}'
                return _n_in_bins

            nsel_in_bins = number_in_bins(self.sel_binidxs, 'selected')
            ncands_in_bins = number_in_bins(self.cand_binidxs, 'candidates')
            LOGGER.debug(f'cands: {ncands_in_bins}')
            LOGGER.debug(f'sel: {nsel_in_bins}')

            # ------------------------------------------------------------------
            # for all bins with no selection,
            # need to assign their candidates to adjacent bins
            for b in range(self.nbins):
                if nsel_in_bins[b] != 0:
                    continue
                if ncands_in_bins[b] == 0:
                    continue

                LOGGER.debug(f'bin {b} has no selections. '
                             f'assigning candidates to {self.distribute_zero_map[b]}')

                for b2 in self.distribute_zero_map[b]:

                    # if there are two neigbors
                    if len(b2) == 2:
                        b2a,b2b = b2[0],b2[1]
                        if nsel_in_bins[b2a] > 0 and nsel_in_bins[b2b] > 0:
                            k1 = ncands_in_bins[b]//2
                            k2 = ncands_in_bins[b] - k1
                            LOGGER.debug(f'assigning {k1} and {k2} to bins {b2a} and {b2b}')
                            ncands_in_bins[b2a] += k1
                            ncands_in_bins[b2b] += k2
                            ncands_in_bins[b] = 0
                            break

                        elif nsel_in_bins[b2a] > 0 and nsel_in_bins[b2b] == 0:
                            k1 = ncands_in_bins[b]
                            LOGGER.debug(f'assigning {k1} to bin {b2a}')
                            ncands_in_bins[b2a] += k1
                            ncands_in_bins[b] = 0
                            break

                        elif nsel_in_bins[b2a] == 0 and nsel_in_bins[b2b] > 0:
                            k1 = ncands_in_bins[b]
                            LOGGER.debug(f'assigning {k1} to bin {b2b}')
                            ncands_in_bins[b2b] += k1
                            ncands_in_bins[b] = 0
                            break

                    # only one neighbor
                    elif len(b2) == 1:
                        b2a = b2[0]
                        if nsel_in_bins[b2a] > 0:
                            k1 = ncands_in_bins[b]
                            LOGGER.debug(f'assigning {k1} to bin {b2a}')
                            ncands_in_bins[b2a] += k1
                            ncands_in_bins[b] = 0
                            break

            # ------------------------------------------------------------------
            LOGGER.debug(f'cands: {ncands_in_bins}')
            LOGGER.debug(f'sel: {nsel_in_bins}')

            # ------------------------------------------------------------------
            # compute weights
            bin_weights = np.zeros(self.nbins_x)
            for b in range(self.nbins):
                if nsel_in_bins[b] == 0:
                    # if selected is 0, then candidates must be zero
                    assert ncands_in_bins[b] == 0, f'Failed to distribute correctly'
                else:
                    bin_weights[b] = (nsel_in_bins[b]+ncands_in_bins[b]) / nsel_in_bins[b]

            LOGGER.debug(f'bin_weights: {bin_weights}')

            # ------------------------------------------------------------------
            # now, assign them to the elements
            weights = bin_weights[self.sel_binidxs]
            LOGGER.debug(f'weights: {weights}')

            # the sum of bin weights should equal to all elements
            if not np.isclose(weights.sum(), nsel+ncands):
                LOGGER.error(f'cands={ncands_in_bins}, sel={nsel_in_bins}')
                LOGGER.error(f'bin_weights={bin_weights}')
                LOGGER.error(f'weights={weights} ({weights.sum()})')

                raise Exception(f'sum of weights should match total elements. '\
                                f'cands={nsel}, sel={ncands}; wsum={weights.sum()})')

        # ----------------------------------------------------------------------
        if normalize:
            LOGGER.info(f'Computed weights: {weights.shape}, sum = {weights.sum()}')
            weights *= (weights.shape[0] / weights.sum())

        LOGGER.info(f'Computed weights: {weights.shape}, sum = {weights.sum()}')
        return ids, weights

    # --------------------------------------------------------------------------
    def checkpoint(self):

        super()._checkpoint()
        if self.hdspace is not None:
            take_backup(self.lchkpt)
            self.hdspace.checkpoint(self.lchkpt)
        LOGGER.debug(f'Checkpointed {self}')

    def restore(self):

        # ----------------------------------------------------------------------
        def _restore_files(_files):
            if not files_exist(_files):
                return False

            try:
                self._restore(_files[0])
                if self.hdspace is not None:
                    self.hdspace.restore(_files[1])
            except Exception as e:
                LOGGER.error(e)
                return False

            if not self.test():
                LOGGER.error('Inconsistent restore!')
                return False
            return True

        # ----------------------------------------------------------------------
        if self.hdspace is None:
            files = [self.schkpt]
        else:
            files = [self.schkpt, self.lchkpt]

        success = False
        for i in range(2):
            success = _restore_files(files)
            if success:
                break
            files = [backup_fname(f) for f in files]
            continue

        success = success and self.test()
        if not success:
            LOGGER.info(f'Failed to restore')
            return False

        LOGGER.info(f'Restored {self}')
        return True

    def test(self):

        if self.hdspace is not None:
            h = self.hdspace.count()
            if len(self.selected) != h:
                LOGGER.error(f'Inconsistent Sampler: lspace={h}; {self}')
                return False
        return True

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    @staticmethod
    def fetch_first_chunk(N, S, l):

        Nsum = N.sum()

        # if there are no candidates
        if Nsum == 0:
            return np.zeros(N.shape).astype(np.uint)

        # if there is a single non-zero bin
        if (N > 0).sum() == 1:
            return N.astype(np.uint)

        Ssum = S.sum()
        NS = N + S

        # target distribution (considering N+S)
        # cannot pick anything if N+S = 0
        q = (1 - l) * NS / (Nsum + Ssum) + l / N.shape[0]
        q[NS == 0] = 0
        q /= q.sum()

        # create a chunk size that will create the target distribution
        # --- target = q*NS.sum()              (use q to divide N+S)
        # --- deficit = q*S.sum() - S          (use q to divide S)
        # --- chunk = target + deficit
        c = q * (Nsum + 2 * Ssum) - S

        # if the past was bad, we may be way far away from the target
        # for novelty, we want to have the chunk as small as possible
        # for random, as large as possible
        cnz = c > 0
        c[cnz] -= l * (c[cnz].min() - 1)

        # finally, a chunk cannot be larger than the input data
        c = np.minimum(np.ceil(c), N)
        assert c.min() >= 0
        return c.astype(np.uint)

    # --------------------------------------------------------------------------
    @staticmethod
    def create_chunks_dynamic(hist_cands, hist_sel, novelty_factor):

        LOGGER.info(f'Creating chunks for dynamic selection: '
                    f'novelty = {novelty_factor}, '
                    f'cands = {hist_cands.sum()}, selected = {hist_sel.sum()}')

        # compute size of each chunk for each bin
        # all bins have the same number of chunks
        # but sizes of these chunks will vary across bins
        N = np.copy(hist_cands)
        S = np.copy(hist_sel)

        # first figure out the size of each chunk
        chunk_sizes = []
        while N.sum() > 0:
            c = SamplerBinned.fetch_first_chunk(N, S, novelty_factor)
            chunk_sizes.append(c)
            N -= c
            S += c

        chunk_sizes = np.swapaxes(np.array(chunk_sizes), 0, 1)
        assert chunk_sizes.shape[0] == hist_cands.shape[0]

        # all bins have the same number of chunks
        nchunks = chunk_sizes.shape[1]
        LOGGER.info(f'number of chunks (per bin) = {nchunks}')

        # compute the edges of these chunk
        chunk_edges = np.cumsum(np.insert(chunk_sizes, 0, 0, axis=1), axis=1)
        LOGGER.info(f'Created chunks {chunk_edges.shape}')
        return chunk_edges

    @staticmethod
    def create_chunks_static(hist_cands, novelty_factor):

        LOGGER.info(f'Creating chunks for static selection: '
                    f'novelty = {novelty_factor}, cands = {hist_cands.sum()}')

        # depending upon the novelty factor, decide the chunk size
        # the same chunk (id) gets equal priority across all bins
        # to compute chunk size, we interpolate in the linear/log10 space
        # novelty_factor = 1:   chunk_sz = 1
        #       this will flatten the curve (i.e., novelty sampling)
        # novelty_factor = 0:   chunk_sz = bin_count
        #       this will mimic input distribution (i.e., random sampling)

        # compute the chunk size
        chunk_sizes = np.zeros_like(hist_cands).astype(np.float)
        nchunks = np.zeros_like(hist_cands).astype(np.uint)

        # compute only where bins are nonzero
        zidx = np.where(hist_cands == 0)[0]
        nzidx = np.where(hist_cands > 0)[0]

        if True:       # linear
            chunk_sizes[nzidx] = novelty_factor + (1.-novelty_factor) * hist_cands[nzidx]

        else:       # logarithmic
            chunk_sizes[nzidx] = 10. ** ((1.-novelty_factor) * np.log10(hist_cands[nzidx]))

        # number of chunks
        nchunks[nzidx] = np.ceil(hist_cands[nzidx] / chunk_sizes[nzidx]).astype(np.uint)

        # should have zero value if and only if bin_count = 0
        assert chunk_sizes[nzidx].min() > 0
        assert nchunks[nzidx].min() > 0
        if len(zidx) > 0:
            assert np.isclose(chunk_sizes[zidx].min(), 0.)
            assert (nchunks[zidx].min() == 0) and (nchunks[zidx].max() == 0)

        # min and max value of chunk
        assert np.all(chunk_sizes[nzidx] <= hist_cands[nzidx])
        assert np.all(nchunks[nzidx] <= hist_cands[nzidx])

        nchunks_all = int(1+nchunks.max())
        chunk_edges = np.zeros((hist_cands.shape[0], nchunks_all), dtype=np.uint)

        # create edges of these chunks
        nbins = hist_cands.shape[0]
        for bin in range(nbins):

            if nchunks[bin] == 0:
                c = np.zeros(1).astype(np.uint)
            else:
                c = chunk_sizes[bin] * np.arange(0, nchunks[bin] + 1)
                c = np.round(c).astype(np.uint)
                c[-1] = hist_cands[bin]

            chunk_edges[bin, :c.shape[0]] = c
            if c.shape[0] < nchunks_all:
                chunk_edges[bin, c.shape[0]:] = np.repeat(c[-1], nchunks_all-c.shape[0])

        LOGGER.info(f'Created chunks {chunk_edges.shape}')
        return chunk_edges

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
