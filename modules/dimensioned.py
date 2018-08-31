import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

# load solver components
from solver.cells import Cell
from solver.reactions import RegulatoryModule


class Network(Cell):
    """
    Class defines a SSA-solver compatible network.

    * See solver.cells for further documentation.
    """

    @staticmethod
    def dimensionalize_genes(rxns, **kw):
        genes = {}
        for rxn_id, rxn in rxns.items():
            if 'synthesis' in rxn_id:
                genes[rxn_id[0]] = Gene(rxn, rxn_id, **kw)
        return genes

    @staticmethod
    def from_sbml(sbml, T=1, X=100, Y=100):
        """
        Build solver-compatible network from gnw.SBML instance.

        Args:
        sbml (gnw.SBML instance) - network to reproduce
        T, X, Y (float) - time, mRNA level, and protein level scaling constants

        Returns:
        network (Network instance) - SSA solver compatible network
        """

        # instantiate network
        network = Network()

        # dimensionalize all genes
        genes = network.dimensionalize_genes(sbml.rxns, T=T, X=X, Y=Y)

        # add each dimensional gene to network
        for g_id in 'uxyG':
            gene = genes[g_id]
            network.add_gene(name=g_id, k=gene.k1, g1=gene.g0, g2=gene.g1)

        # add transcription reactions
        for g_id in 'uxyG':
            gene = genes[g_id]
            p_idx = network.proteins[g_id]

            # format modifiers
            modules = []
            for kw in gene.modules:
                modifiers = [network.proteins[m] for m in kw['modifiers']]
                kw['modifiers'] = modifiers
                modules.append(RegulatoryModule(**kw))

            # add perturbation sensitivity
            if g_id == 'u':
                perturbed = True
            else:
                perturbed = False

            # add transcription reaction
            network.add_transcription(g_id, modules, k=gene.k0, alpha=gene.alpha, perturbed=perturbed)

        return network


class Gene:
    """
    Class defines a dimensioned gene.

    Attributes:
    k0 (float) - transcription rate constant
    g0 (float) - transcript degradation constant
    k1 (float) - translation rate constant
    g1 (float) - protein degradation constant
    modules (list of RegulatoryModule instances)
    num_modules (int) - number of regulatory modules
    num_alpha (int) - number of alpha coefficients
    alpha (1-D array) - alpha coefficients
    """

    def __init__(self, rxn, rxn_id, **scaling):
        self.parse_synthesis(rxn, rxn_id, **scaling)

    def parse_synthesis(self, rxn, rxn_id, T=1, X=100, Y=100):

        parameters = rxn['parameters']

        self.k0 = parameters['max'] / T * X
        self.g0 = self.k0 / X

        # protein synthesis/degradation
        self.k1 = parameters['maxTranslation'] * Y / (X * T)
        self.g1 = self.k1 * (X / Y)

        # store modules
        self.num_modules = len([p for p in parameters.keys() if 'numActivators' in p])
        self.parse_modules(rxn, Y)

        # store alpha values
        self.num_alpha = len([p for p in parameters.keys() if 'a_' in p])
        self.alpha = [parameters['a_{:d}'.format(i)] for i in range(self.num_alpha)]

    def parse_modules(self, rxn, Y=100):
        ind, modules = 1, []
        for module_id in range(1, self.num_modules+1):
            module = self.build_module_dict(rxn, module_id, ind, Y=Y)
            modules.append(module)
            ind += (module['nA'] + module['nD'])
        self.modules = modules

    def build_module_dict(self, rxn, module_id, ind, Y=100):

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
        k = np.array(k, dtype=np.float64) * Y
        n = np.array(n, dtype=np.float64)

        # get modifiers and compile module dictionary
        modifiers = rxn['modifiers'][ind-1:ind+nI-1]
        module_dict = dict(modifiers=modifiers, nA=nA, nD=nD, bindsAsComplex=bindsAsComplex, k=k, n=n)

        return module_dict
