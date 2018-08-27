

class SBML:

    def __init__(self, root):
        self.root = root
        self.base = root.tag.strip('sbml')
        self.traverse()

    def traverse(self):
        self.rxns = {}
        self._recurse(self.root, self._parse_model)

    @staticmethod
    def _recurse(parent, f):
        for child in parent:
            f(child)

    def _parse_rxn(self, node):
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
        rxn_dict['products'] = p.attrib['species']

        # get modifiers
        m = node.find(self.base+'listOfModifiers')
        if m is not None:
            m = m.findall(self.base+'modifierSpeciesReference')
            rxn_dict['modifiers'] = [mod.attrib['species'] for mod in m]

        # get parameters
        kl = node.find(self.base+'kineticLaw')
        params = kl.find(self.base+'listOfParameters')
        params = params.findall(self.base+'parameter')
        rxn_dict['parameters'] = {p.attrib['name']: float(p.attrib['value']) for p in params}

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
        tag = node.tag.replace(self.base, '')
        print(tag)
        self._recurse(node, self._reporter)
