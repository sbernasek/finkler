import sys, os
from os.path import join
from glob import glob
import pandas as pd
import numpy as np
from functools import reduce
from operator import add
import xml.etree.ElementTree as ET

from .io import IO
from .sbml import SBML
from .dimensionalize import DimensionedCell, DimensionedSimulation
from .timeseries import PerturbationTimeSeries

class gnwData:

    def __init__(self, path):
        self.path = path
        self.network_paths = self.get_network_paths(path)

    def __getitem__(self, network_id):
        return self.load_network(self.network_paths[network_id], network_id)

    @staticmethod
    def get_network_paths(path):
        get_id = lambda x: int(x.rsplit('/', 1)[-1])
        return {get_id(p): p for p in glob(join(path, '*[0-9]'))}

    @staticmethod
    def load_network(network_path, network_id):
        return gnwNetwork(network_path, network_id)


class gnwNetwork:

    def __init__(self, path, network_id):
        self.path = path
        self.network_id = network_id
        self.simulations = self.get_simulations()

    def __getitem__(self, condition):
        return self.simulations[condition]

    def get_simulations(self):
        get_genotype = lambda x: x.rsplit('/', 1)[-1][:2]
        get_name = lambda x: '_'.join([str(self.network_id), get_genotype(x)])
        ps = [(p, get_name(p)) for p in glob(join(self.path, '*_sim'))]
        return {x[1][-2:]: gnwSimulation(*x) for x in ps}


class gnwSimulation:

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.load()
        self.s_mean = self.s.mean(axis=-2)
        self.sx_mean = self.sx.mean(axis=-2)

    def load(self):

        # path constructor
        pgen = lambda x: join(self.path, x.format(self.name))

        # network structure
        self.sbml = self.load_sbml(pgen('{:s}.xml'))
        #self.xml = self.load_xml(pgen('{:s}.xml'))
        #self.sbml = self.load_sbml(pgen('{:s}_sbml_params.pkl'))
        edges = self.load_structure(pgen('{:s}_goldstandard.tsv'))
        self.edges = pd.DataFrame(edges, columns=('from', 'to', 'exists'))
        conn = self.load_structure(pgen('{:s}_goldstandard_signed.tsv'))
        conn = pd.DataFrame(conn, columns=('from', 'to', 'sign'))
        self.connectivity = conn.set_index(keys=['from', 'to'])
        self.assign_edge_types()

        # perturbations
        ptb_path = pgen('{:s}_dream4_timeseries_perturbations.tsv')
        self.labels, self.ptb, self.P, self.N = self.load_ptb(ptb_path)

        # steady states
        self.ss_rna = self.load_ss(pgen('{:s}_wildtype.tsv'))
        self.ss_p = self.load_ss(pgen('{:s}_proteins_wildtype.tsv'))
        self.ss_rna_xnoise = self.load_ss(pgen('{:s}_noexpnoise_wildtype.tsv'))
        self.ss_p_xnoise = self.load_ss(pgen('{:s}_noexpnoise_proteins_wildtype.tsv'))

        # timeseries (with noise)
        t, sr = self.load_ts(pgen('{:s}_dream4_timeseries.tsv'))
        _, sp = self.load_ts(pgen('{:s}_proteins_dream4_timeseries.tsv'))
        self.t, self.s = t, np.concatenate((sr, sp), axis=1)

        # timeseries (before noise)
        _, srx = self.load_ts(pgen('{:s}_noexpnoise_dream4_timeseries.tsv'))
        _, spx = self.load_ts(pgen('{:s}_noexpnoise_proteins_dream4_timeseries.tsv'))
        self.sx = np.concatenate((srx, spx), axis=1)

        # load normalization
        self.norm = self.load_norm(pgen('{:s}_normalization_constant.tsv'))


    @staticmethod
    def load_sbml(path):
        return SBML(ET.parse(path).getroot())

    # @staticmethod
    # def load_sbml(sbml_path):
    #     return IO.read_pkl(sbml_path)

    # @staticmethod
    # def load_xml(xml_path):
    #     return ET.parse(xml_path).getroot()

    @staticmethod
    def load_structure(path):
        return IO.read_tsv(path)

    @staticmethod
    def load_norm(path):
        return float(IO.read_tsv(path)[0][0])

    @staticmethod
    def load_ss(path):
        ts = IO.read_tsv(path)
        tsheader = ts.pop(0)
        return np.array(ts[0], dtype=np.float64)

    @staticmethod
    def load_ptb(path):
        p = IO.read_tsv(path)
        pheader = p.pop(0)
        up, ind = np.unique(p, axis=0, return_index=True)
        up = up[np.argsort(ind)].astype(np.float64)
        P = len(up)
        N = len(p) // P
        return pheader, up, P, N

    def load_ts(self, path):
        ts = IO.read_tsv(path)
        tsheader = ts.pop(0)
        T = (len(ts) // (self.P*self.N) ) - 1
        S = len(tsheader)
        times, states = self.parse_timeseries(ts, self.P, self.N, T, S)
        return times, states

    @staticmethod
    def parse_timeseries(ts, P, N, T, S):
        """ Extract timeseries data as np array. """
        x = np.empty((P, N, T, S), dtype=np.float64)
        dT = (T+1)
        for i in range(P):
            for j in range(N):
                k = (i*N + j)
                x[i, j, :, :] = ts[slice(k*dT, (k*dT)+dT)][1:]
        x = np.swapaxes(np.swapaxes(x, 1, 3), 2, 3)
        times = x[:, 0, :, :][0, 0, :]
        states = x[:, 1:, :, :]
        return times, states

    def assign_edge_types(self):
        """ Add edge types to sbml reactions. """
        for rxn_id, rxn in self.sbml.rxns.items():
            product = rxn['products']
            if product != '_void_' and rxn['n_inputs'] > 0:
                et = {}
                for mod in rxn['modifiers']:
                    et[mod] = self.connectivity.loc[(mod, product)].values[0]
                rxn['edge_types'] = et

    def build_dimensioned_cell(self, **kw):
        """ Return dimensionalized network model. """
        return DimensionedCell.from_sbml(self.sbml, **kw)

    def dimensionalize(self, **kw):
        """ Return dimensionalized simulation. """
        return DimensionedSimulation(self, **kw)

    def get_timeseries(self):
        """ Returns timeseries object for plotting. """

        # sort into (r0, p0, ...rN, PN) order
        ind = np.array(reduce(add, zip(range(4), range(4, 8))))

        return PerturbationTimeSeries(self.t, self.sx[:, ind, :, :])


