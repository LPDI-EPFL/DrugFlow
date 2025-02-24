from rdkit import Chem

from src import constants


def remove_dummy_atoms(rdmol, sanitize=False):
    # find exit atoms to be removed
    dummy_inds = []
    for a in rdmol.GetAtoms():
        if a.GetSymbol() == '*':
            dummy_inds.append(a.GetIdx())

    dummy_inds = sorted(dummy_inds, reverse=True)
    new_mol = Chem.EditableMol(rdmol)
    for idx in dummy_inds:
        new_mol.RemoveAtom(idx)
    new_mol = new_mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(new_mol)
    return new_mol


def build_molecule(coords, atom_types, bonds=None, bond_types=None,
                   atom_props=None, atom_decoder=None, bond_decoder=None):
    """
    Build RDKit molecule with given bonds
    :param coords: N x 3
    :param atom_types: N
    :param bonds: 2 x N_bonds
    :param bond_types: N_bonds
    :param atom_props: Dict, key: property name, value: list of float values (N,)
    :param atom_decoder: list
    :param bond_decoder: list
    :return: RDKit molecule
    """
    if atom_decoder is None:
        atom_decoder = constants.atom_decoder
    if bond_decoder is None:
        bond_decoder = constants.bond_decoder
    assert len(coords) == len(atom_types)
    assert bonds is None or bonds.size(1) == len(bond_types)

    mol = Chem.RWMol()
    for i, atom in enumerate(atom_types):
        element = atom_decoder[atom.item()]
        charge = None
        explicitHs = None

        if len(element) > 1 and element.endswith('H'):
            explicitHs = 1
            element = element[:-1]
        elif element.endswith('+'):
            charge = 1
            element = element[:-1]
        elif element.endswith('-'):
            charge = -1
            element = element[:-1]

        if element == 'NOATOM':
            # element = 'Xe'  # debug
            element = '*'

        a = Chem.Atom(element)

        if explicitHs is not None:
            a.SetNumExplicitHs(explicitHs)
        if charge is not None:
            a.SetFormalCharge(charge)

        if atom_props is not None:
            for k, vals in atom_props.items():
                a.SetDoubleProp(k, vals[i].item())

        mol.AddAtom(a)

    # add coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (coords[i, 0].item(),
                                 coords[i, 1].item(),
                                 coords[i, 2].item()))
    mol.AddConformer(conf)

    # add bonds
    if bonds is not None:
        for bond, bond_type in zip(bonds.T, bond_types):
            bond_type = bond_decoder[bond_type]
            src = bond[0].item()
            dst = bond[1].item()

            # try:
            if bond_type == 'NOBOND' or mol.GetAtomWithIdx(src).GetSymbol() == '*' or mol.GetAtomWithIdx(dst).GetSymbol() == '*':
                continue
            # except RuntimeError:
            #     from pdb import set_trace; set_trace()

            if mol.GetBondBetweenAtoms(src, dst) is not None:
                assert mol.GetBondBetweenAtoms(src, dst).GetBondType() == bond_type, \
                    "Trying to assign two different types to the same bond."
                continue

            if bond_type is None or src == dst:
                continue
            mol.AddBond(src, dst, bond_type)

    mol = remove_dummy_atoms(mol, sanitize=False)
    return mol
