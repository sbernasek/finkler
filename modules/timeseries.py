import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from functools import reduce
from operator import add


class PerturbationTimeSeries:

    def __init__(self, t, states, labels=None):
        self.t = t
        self.states = states
        self.mean = np.mean(states, axis=2)
        self.var = np.var(states, axis=2)

        if labels is None:
            labels = ['u', 'x', 'y', 'G']
        self.build_labeler(labels)

    def build_labeler(self, labels):
        rlabel = lambda x: [l+' rna' for l in x]
        plabel = lambda x: [l+' protein' for l in x]
        self.labeler = np.vectorize(dict(enumerate(reduce(add, zip(rlabel(labels), plabel(labels))))).get)

    @staticmethod
    def from_timeseries_list(timeseries_list):

        t = timeseries_list[0].t

        # reshape states to match GNW format (P, S, N, T)
        pad = lambda x: x.reshape(1, *x.shape)
        states = np.vstack([pad(ts.states) for ts in timeseries_list])
        states = np.swapaxes(states, 1, 2)

        return PerturbationTimeSeries(t, states)

    def plot_perturbations(self, species, ax=None, **kw):

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
        """ Plot simulation trajectories. """
        for trajectory in self.states[ptb, species, :, :]:
            ax.plot(self.t, trajectory, '-', c=c, lw=1, alpha=0.5, **kw)

    def _plot_mean(self, ax, ptb, species, c='k', interval=True, **kw):

        # plot SEM interval
        if interval:
            sem = np.sqrt(self.var[ptb, species, :]/self.states.shape[2])
            lbound = self.mean[ptb, species, :] - sem
            ubound = self.mean[ptb, species, :] + sem
            ax.fill_between(self.t, lbound, ubound, color=c, alpha=0.5)

        # plot mean
        trajectory = self.mean[ptb, species, :]
        ax.plot(self.t, trajectory, '-', c=c, lw=1, alpha=1, **kw)
