import numpy as np
from rxndiffusion.cells import Cell
from rxndiffusion.reactions import RegulatoryModule


class DimensionedCell(Cell):

    def __init__(self, T=1, X=10, Y=1000):
        Cell.__init__(self)
        self.T = T
        self.X = X
        self.Y = Y

    @staticmethod
    def from_sbml(sbml, T=1, X=10, Y=1000):

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

            # add transcription reaction
            cell.add_transcription(g_id, modules, k=gene.k0, alpha=gene.alpha)

        return cell


class DimensionalNetwork:

    def __init__(self, rxns, **kw):
        self.genes = {k[0]: DimensionalGene(**kw) for k in rxns.keys() if 'synthesis' in k}
        self.parse_degradations(rxns)
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

    def __init__(self, T=1, X=10, Y=1000):
        self.T = T
        self.X = X
        self.Y = Y

    def parse_degradation(self, rxn):
        self.g0 = rxn['parameters']['delta']/self.T
        self.k0 = self.g0 * self.X

    def parse_synthesis(self, rxn, rxn_id):

        parameters = rxn['parameters']

        # protein synthesis/degradation
        self.g1 = parameters['deltaProtein']/self.T
        self.k1 = self.g1 * self.Y / self.X

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
        for ind in range(ind, ind+nI):
            k.append(p['k_{:d}'.format(ind)])
            n.append(p['n_{:d}'.format(ind)])

        # re-dimensionalize k
        k = np.array(k, dtype=np.float64) * self.Y
        n = np.array(n, dtype=np.float64)

        # get modifiers and compile module dictionary
        modifiers = rxn['modifiers'][:nI]
        module_dict = dict(modifiers=modifiers, nA=nA, nD=nD, bindsAsComplex=bindsAsComplex, k=k, n=n)

        return module_dict
