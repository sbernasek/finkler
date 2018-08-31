import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from functools import reduce
from operator import add


class PerturbationTimeSeries:
    """
    Container for perturbation timeseries data, shaped P x S x N x T where:

    P = number of perturbations
    S = number of species
    N = number of trajectories
    T = number of timepoints

    """

    def __init__(self, t, states, labels=None):
        """
        Instantiate PerturbationTimeSeries object.

        Args:
        t (array of length T) - timepoints
        states (array of shape P x S x N x T) - state values
        labels (list of length S/2) - alphanumeric labels for each gene
        """

        self.t = t
        self.states = states

        # compute mean + variance for each set of trajectories
        self.mean = np.mean(states, axis=2)
        self.var = np.var(states, axis=2)

        # add labels
        if labels is None:
            labels = ['u', 'x', 'y', 'G']
        self.build_labeler(labels)

    def build_labeler(self, labels):
        """
        Define function for assigning alphanumeric labels to each species.

        Args:
        labels (list of length S/2) - alphanumeric labels for each gene
        """
        rlabel = lambda x: [l+' rna' for l in x]
        plabel = lambda x: [l+' protein' for l in x]
        self.labeler = np.vectorize(dict(enumerate(reduce(add, zip(rlabel(labels), plabel(labels))))).get)

    @staticmethod
    def from_timeseries_list(ts_list):
        """
        Instantiate from list of outputs from the SSA solver for each perturbation.

        Args:
        ts_list (list of TimeSeries, length P) - TimeSeries is the SSA solver output format in which states are shaped N x S x T. The states are reshaped here to match the format from GeneNetWeaver.
        """

        # get times (assumes all share same sampling frequency + duration)
        t = ts_list[0].t

        # reshape states to match GNW format (P, S, N, T)
        pad = lambda x: x.reshape(1, *x.shape)
        states = np.vstack([pad(ts.states) for ts in ts_list])
        states = np.swapaxes(states, 1, 2)

        return PerturbationTimeSeries(t, states)

    def nondimensionalize(self, T=1, X=100, Y=100, normalize=False):
        """ Returns non-dimensionalized version of timeseries. """

        # scale time
        nd_t = self.t / T

        # get rna/protein
        states = deepcopy(self.states)
        rna = states[:, ::2, :, :]
        protein = states[:, 1::2, :, :]

        # non-dimensionalize
        rna = rna / X
        protein = protein / Y

        # sort back into (r0, p0, ...rN, PN) order
        ind = np.array(reduce(add, zip(range(4), range(4, 8))))
        nd_states = np.concatenate((rna, protein), axis=1)[:, ind, :, :]

        # normalize
        if normalize:
            norm = rna.max()
            nd_states = nd_states / norm

        return PerturbationTimeSeries(nd_t, nd_states)

    def plot_perturbations(self, species, ax=None, **kw):
        """
        Plot mean trajectory of species under each perturbation.

        Args:
        species (int) - species index
        """

        # create axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))

        # plot perturbation contours
        for ptb in range(self.states.shape[0]):
            c = 'cmykrgb'[ptb]
            self._plot_mean(ax, ptb, species, c=c, **kw)

        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('{:s}'.format(self.labeler(species)), fontsize=11)
        ax.tick_params(labelsize=10)

    def _plot_samples(self, ax, ptb, species, c='k', **kw):
        """
        Plot simulation trajectories for a single perturbation of a species.

        Args:
        ptb (int) - perturbation index
        species (int) - species index
        """
        for trajectory in self.states[ptb, species, :, :]:
            ax.plot(self.t, trajectory, '-', c=c, lw=1, alpha=0.5, **kw)

    def _plot_mean(self, ax, ptb, species, c='k', interval=True, **kw):
        """
        Plot mean trajectory for a single perturbation of a species.

        Args:
        ptb (int) - perturbation index
        species (int) - species index
        """

        # plot SEM interval
        if interval:
            sem = np.sqrt(self.var[ptb, species, :]/self.states.shape[2])
            lbound = self.mean[ptb, species, :] - sem
            ubound = self.mean[ptb, species, :] + sem
            ax.fill_between(self.t, lbound, ubound, color=c, alpha=0.5)

        # plot mean
        trajectory = self.mean[ptb, species, :]
        ax.plot(self.t, trajectory, '-', c=c, lw=1, alpha=1, **kw)
