import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from rxndiffusion.cells import Cell
from rxndiffusion.reactions import RegulatoryModule
from rxndiffusion.simulate import MonteCarloSimulation
from rxndiffusion.solver.signals import cSquarePulse
from .timeseries import PerturbationTimeSeries
from functools import reduce
from operator import add


class DimensionedSimulation:

    def __init__(self, gnw_sim, T=1, X=100, Y=100):

        # characteristic scales
        self.T = T
        self.X = X
        self.Y = Y

        # dimensionalize and un-normalize steady states
        ssr = self.scale_r(gnw_sim.ss_rna_xnoise)
        ssp = self.scale_p(gnw_sim.ss_p_xnoise)
        ss = np.hstack((ssr, ssp)) #/ gnw_sim.norm

        # rearrange states to simulator order
        ind = np.array(reduce(add, zip(range(4), range(4, 8))))
        ss = ss[ind]

        # build cell and instantiate simulator
        cell = gnw_sim.build_dimensioned_cell(T=T, X=X, Y=Y)
        self.sim = MonteCarloSimulation(cell, ic=ss)

        # store perturbations
        self.ptb = gnw_sim.ptb[:, 0]
        self.duration =  self.scale_T(gnw_sim.t.max())

    def scale_r(self, rna):
        return self.X * rna

    def scale_p(self, protein):
        return self.X * protein

    def scale_T(self, t):
        return self.T * t

    def run_perturbation(self, ptb=0, num_trials=3, duration=None):
        """
        Run dimensioned simulation of single perturbation.

        Args:
        ptb (float) - perturbation value
        num_trials (int) - number of stochastic trajectories

        Returns:
        ts (TimeSeries instance) - simulation output
        """

        # define perturbation signal
        if duration is None:
            duration = self.duration
        signal = cSquarePulse(t_on=0, t_off=duration/2, off=0, on=ptb)

        # run simulation
        ts = self.sim.run(input_function=signal,
                     num_trials=num_trials,
                     duration=duration,
                     dt=1)

        return ts

    def run(self, num_trials=3):
        """
        Run dimensioned simulation for all perturbations.

        Args:
        num_trials (int) - number of stochastic trajectories
        """
        ts = [self.run_perturbation(ptb, num_trials) for ptb in self.ptb]
        self.ts = PerturbationTimeSeries.from_timeseries_list(ts)

    def nondimensionalize(self, normalize=False):

        # scale time
        nd_t = self.ts.t / self.T

        # get rna/protein
        states = deepcopy(self.ts.states)
        rna = states[:, ::2, :, :]
        protein = states[:, 1::2, :, :]

        # non-dimensionalize
        rna = rna / self.X
        protein = protein / self.Y

        # sort back into (r0, p0, ...rN, PN) order
        ind = np.array(reduce(add, zip(range(4), range(4, 8))))
        nd_states = np.concatenate((rna, protein), axis=1)[:, ind, :, :]

        # normalize
        if normalize:
            norm = rna.max()
            nd_states = nd_states / norm

        return PerturbationTimeSeries(nd_t, nd_states)


class DimensionedCell(Cell):

    def __init__(self, T, X, Y):
        Cell.__init__(self)
        self.T = T
        self.X = X
        self.Y = Y

    @staticmethod
    def from_sbml(sbml, T=1, X=100, Y=100):

        cell = DimensionedCell(T, X, Y)

        # load dimensioned network
        network = DimensionalNetwork(sbml.rxns, T=T, X=X, Y=Y)

        # add each gene to network
        for g_id in 'uxyG':
            gene = network.genes[g_id]
            cell.add_gene(name=g_id, k=gene.k1, g1=gene.g0, g2=gene.g1)

        # add transcription reactions
        for g_id in 'uxyG':

            gene = network.genes[g_id]
            p_idx = cell.proteins[g_id]

            # format modifiers
            modules = []
            for kw in gene.modules:
                kw['modifiers'] = [cell.proteins[m] for m in kw['modifiers']]
                modules.append(RegulatoryModule(**kw))

            # add perturbation sensitivity
            if g_id == 'u':
                perturbed = True
            else:
                perturbed = False

            # add transcription reaction
            cell.add_transcription(g_id, modules, k=gene.k0, alpha=gene.alpha, perturbed=perturbed)

        return cell


class DimensionalNetwork:

    def __init__(self, rxns, **kw):
        self.genes = {k[0]: DimensionalGene(**kw) for k in rxns.keys() if 'synthesis' in k}
        #self.parse_degradations(rxns)
        self.parse_syntheses(rxns)

    def parse_degradations(self, rxns):
        for rxn_id, rxn in rxns.items():
            if 'degradation' in rxn_id:
                self.genes[rxn_id[0]].parse_degradation(rxn)

    def parse_syntheses(self, rxns):
        for rxn_id, rxn in rxns.items():
            if 'synthesis' in rxn_id:
                self.genes[rxn_id[0]].parse_synthesis(rxn, rxn_id)


class DimensionalGene:

    def __init__(self, T, X, Y):
        self.T = T
        self.X = X
        self.Y = Y

    def parse_degradation(self, rxn):
        parameters = rxn['parameters']

    def parse_synthesis(self, rxn, rxn_id):

        parameters = rxn['parameters']

        self.k0 = parameters['max'] / self.T * self.X
        self.g0 = self.k0 / self.X

        # protein synthesis/degradation
        self.k1 = parameters['maxTranslation'] * self.Y / (self.X * self.T)
        self.g1 = self.k1 * (self.X / self.Y)

        # store modules
        self.num_modules = len([p for p in parameters.keys() if 'numActivators' in p])
        self.parse_modules(rxn)

        # store alpha values
        self.num_alpha = len([p for p in parameters.keys() if 'a_' in p])
        self.alpha = [parameters['a_{:d}'.format(i)] for i in range(self.num_alpha)]

    def parse_modules(self, rxn):
        ind, modules = 1, []
        for module_id in range(1, self.num_modules+1):
            module = self.build_module_dict(rxn, module_id=module_id, ind=ind)
            modules.append(module)
            ind += (module['nA'] + module['nD'])
        self.modules = modules

    def build_module_dict(self, rxn, module_id=1, ind=0):

        # get activators/deactivators
        p = rxn['parameters']
        nA = int(p['numActivators_{:d}'.format(module_id)])
        nD = int(p['numDeactivators_{:d}'.format(module_id)])
        nI = nA + nD
        bindsAsComplex = bool(p['bindsAsComplex_{:d}'.format(module_id)])

        # get dissociation constants and hill coefficients
        k, n = [], []
        for i in range(ind, ind+nI):
            k.append(p['k_{:d}'.format(i)])
            n.append(p['n_{:d}'.format(i)])

        # re-dimensionalize k
        k = np.array(k, dtype=np.float64) * self.Y
        n = np.array(n, dtype=np.float64)

        # get modifiers and compile module dictionary
        modifiers = rxn['modifiers'][ind-1:ind+nI-1]
        module_dict = dict(modifiers=modifiers, nA=nA, nD=nD, bindsAsComplex=bindsAsComplex, k=k, n=n)

        return module_dict
