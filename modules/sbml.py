

class SBML:
    """
    Compiles dictionary of reactions from XML file.

    Attributes:
    rxns (dict) - keys are reaction names, values are reaction parameters
    """

    def __init__(self, root):
        """
        Args:
        root (ElementTree) - root of SBML structure
        """

        self.root = root
        self.base = root.tag.strip('sbml')
        self.rxns = None
        self.compile_reactions()

    def compile_reactions(self):
        """ Parse XML to get reactions. """
        self.rxns = {}
        self._recurse(self.root, self._parse_model)

    @staticmethod
    def _recurse(parent, f):
        for child in parent:
            f(child)

    def _parse_rxn(self, node):
        """ Parse individual reaction element. """

        tag = node.tag.replace(self.base, '')
        rxn_id = node.attrib['id']
        rxn_dict = {'name': node.attrib['name']}

        # get reactants
        r = node.find(self.base+'listOfReactants')
        r = r.find(self.base+'speciesReference')
        rxn_dict['reactants'] = r.attrib['species']

        # get products
        p = node.find(self.base+'listOfProducts')
        p = p.find(self.base+'speciesReference')
        product = p.attrib['species']
        rxn_dict['products'] = product

        # get modifiers
        m = node.find(self.base+'listOfModifiers')
        if m is not None:
            m = m.findall(self.base+'modifierSpeciesReference')
            n_inputs = len(m)
            rxn_dict['modifiers'] = [mod.attrib['species'] for mod in m]
        else:
            n_inputs = 0
        rxn_dict['n_inputs'] = n_inputs

        # get parameters
        kl = node.find(self.base+'kineticLaw')
        params = kl.find(self.base+'listOfParameters')
        params = params.findall(self.base+'parameter')
        rxn_dict['parameters'] = {p.attrib['name']: float(p.attrib['value']) for p in params}

        # get edge logic
        if n_inputs > 1:
            logic = 'AND'
            if '+' in rxn_dict['name'].strip('{:s}: '.format(rxn_id)):
                logic = 'OR'
            rxn_dict['logic'] = logic

        # store rxn dictionary
        self.rxns[rxn_id] = rxn_dict

        # reporter
        #self._recurse(node, self._reporter)

    def _parse_model(self, node):
        tag = node.tag.replace(self.base, '')

        if tag == 'model':
            self._recurse(node, self._parse_model)

        if tag == 'listOfReactions':
            self._recurse(node, self._parse_rxn)

    def _reporter(self, node):
        """ Report contents of node and its children. """
        tag = node.tag.replace(self.base, '')
        print('TAG', tag)
        print('Attributes:', node.attrib)
        self._recurse(node, self._reporter)
