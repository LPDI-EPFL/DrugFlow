import warnings

from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, UFFHasAllMoleculeParams

from src.data import sanifix


def uff_relax(mol, max_iter=200):
    """
    Uses RDKit's universal force field (UFF) implementation to optimize a
    molecule.
    """
    if not UFFHasAllMoleculeParams(mol):
        warnings.warn('UFF parameters not available for all atoms. '
                      'Returning None.')
        return None

    try:
        more_iterations_required = UFFOptimizeMolecule(mol, maxIters=max_iter)
        if more_iterations_required:
            warnings.warn(f'Maximum number of FF iterations reached. '
                          f'Returning molecule after {max_iter} relaxation steps.')

    except RuntimeError:
        return None

    return mol


def add_hydrogens(rdmol):
    return Chem.AddHs(rdmol, addCoords=(len(rdmol.GetConformers()) > 0))


def get_largest_fragment(rdmol):
    mol_frags = Chem.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
    largest_frag = max(mol_frags, default=rdmol, key=lambda m: m.GetNumAtoms())

    # try:
    #     Chem.SanitizeMol(largest_frag)
    # except ValueError:
    #     return None

    return largest_frag


def process_all(rdmol, largest_frag=True, adjust_aromatic_Ns=True, relax_iter=0):
    """
    Apply all filters and post-processing steps. Returns a new molecule.

    Returns:
        RDKit molecule or None if it does not pass the filters or processing
        fails
    """

    # Only consider non-trivial molecules
    if rdmol.GetNumAtoms() < 1:
        return None

    # Create a copy
    mol = Chem.Mol(rdmol)

    # try:
    #     Chem.SanitizeMol(mol)
    # except ValueError:
    #     warnings.warn('Sanitization failed. Returning None.')
    #     return None

    if largest_frag:
        mol = get_largest_fragment(mol)
        # if mol is None:
        #     return None

    if adjust_aromatic_Ns:
        mol = sanifix.fix_mol(mol)
        if mol is None:
            return None

    # if add_hydrogens:
    #     mol = add_hydrogens(mol)

    if relax_iter > 0:
        mol = uff_relax(mol, relax_iter)
        if mol is None:
            return None

    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        warnings.warn('Sanitization failed. Returning None.')
        return None

    return mol
