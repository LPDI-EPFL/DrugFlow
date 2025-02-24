import warnings

import torch
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import SanitizeFlags
from src.analysis.metrics import check_mol
from src import utils
from src.data.molecule_builder import build_molecule
from src.data.misc import protein_letters_1to3


# def pocket_to_rdkit(pocket, pocket_representation, atom_encoder=None,
#                     atom_decoder=None, aa_decoder=None, residue_decoder=None,
#                     aa_atom_index=None):
#
#     rdpockets = []
#     for i in torch.unique(pocket['mask']):
#
#         node_coord = pocket['x'][pocket['mask'] == i]
#         h = pocket['one_hot'][pocket['mask'] == i]
#
#         if pocket_representation == 'side_chain_bead':
#             coord = node_coord
#
#             node_types = [residue_decoder[b] for b in h[:, -len(residue_decoder):].argmax(-1)]
#             atom_types = ['C' if r == 'CA' else 'F' for r in node_types]
#
#         elif pocket_representation == 'CA+':
#             aa_types = [aa_decoder[b] for b in h.argmax(-1)]
#             side_chain_vec = pocket['v'][pocket['mask'] == i]
#
#             coord = []
#             atom_types = []
#             for xyz, aa, vec in zip(node_coord, aa_types, side_chain_vec):
#                 # C_alpha
#                 coord.append(xyz)
#                 atom_types.append('C')
#
#                 # all other atoms
#                 for atom_name, idx in aa_atom_index[aa].items():
#                     coord.append(xyz + vec[idx])
#                     atom_types.append(atom_name[0])
#
#             coord = torch.stack(coord, dim=0)
#
#         else:
#             raise NotImplementedError(f"{pocket_representation} residue representation not supported")
#
#         atom_types = torch.tensor([atom_encoder[a] for a in atom_types])
#         rdpockets.append(build_molecule(coord, atom_types, atom_decoder=atom_decoder))
#
#     return rdpockets
def pocket_to_rdkit(pocket, pocket_representation, atom_encoder=None,
                    atom_decoder=None, aa_decoder=None, residue_decoder=None,
                    aa_atom_index=None):

    rdpockets = []
    for i in torch.unique(pocket['mask']):

        node_coord = pocket['x'][pocket['mask'] == i]
        h = pocket['one_hot'][pocket['mask'] == i]
        atom_mask = pocket['atom_mask'][pocket['mask'] == i]

        pdb_infos = []

        if pocket_representation == 'side_chain_bead':
            coord = node_coord

            node_types = [residue_decoder[b] for b in h[:, -len(residue_decoder):].argmax(-1)]
            atom_types = ['C' if r == 'CA' else 'F' for r in node_types]

        elif pocket_representation == 'CA+':
            aa_types = [aa_decoder[b] for b in h.argmax(-1)]
            side_chain_vec = pocket['v'][pocket['mask'] == i]

            coord = []
            atom_types = []
            for resi, (xyz, aa, vec, am) in enumerate(zip(node_coord, aa_types, side_chain_vec, atom_mask)):

                # CA not treated differently with updated atom dictionary
                for atom_name, idx in aa_atom_index[aa].items():

                    if ~am[idx]:
                        warnings.warn(f"Missing atom {atom_name} in {aa}:{resi}")
                        continue

                    coord.append(xyz + vec[idx])
                    atom_types.append(atom_name[0])

                    info = Chem.AtomPDBResidueInfo()
                    # info.SetChainId('A')
                    info.SetResidueName(protein_letters_1to3[aa])
                    info.SetResidueNumber(resi + 1)
                    info.SetOccupancy(1.0)
                    info.SetTempFactor(0.0)
                    info.SetName(f' {atom_name:<3}')
                    pdb_infos.append(info)

            coord = torch.stack(coord, dim=0)

        else:
            raise NotImplementedError(f"{pocket_representation} residue representation not supported")

        atom_types = torch.tensor([atom_encoder[a] for a in atom_types])
        rdmol = build_molecule(coord, atom_types, atom_decoder=atom_decoder)

        if len(pdb_infos) == len(rdmol.GetAtoms()):
            for a, info in zip(rdmol.GetAtoms(), pdb_infos):
                a.SetPDBResidueInfo(info)

        rdpockets.append(rdmol)

    return rdpockets


def mols_to_pdbfile(rdmols, filename, flavor=0):
    pdb_str = ""
    for i, mol in enumerate(rdmols):
        pdb_str += f"MODEL{i + 1:>9}\n"
        block = Chem.MolToPDBBlock(mol, flavor=flavor)
        block = "\n".join(block.split("\n")[:-2])  # remove END
        pdb_str += block + "\n"
        pdb_str += f"ENDMDL\n"
    pdb_str += f"END\n"

    with open(filename, 'w') as f:
        f.write(pdb_str)

    return pdb_str


def mol_as_pdb(rdmol, filename=None, bfactor=None):

    _rdmol = Chem.Mol(rdmol)  # copy
    for a in _rdmol.GetAtoms():
        a.SetIsAromatic(False)
    for b in _rdmol.GetBonds():
        b.SetIsAromatic(False)

    if bfactor is not None:
        for a in _rdmol.GetAtoms():
            val = a.GetPropsAsDict()[bfactor]

            info = Chem.AtomPDBResidueInfo()
            info.SetResidueName('UNL')
            info.SetResidueNumber(1)
            info.SetName(f' {a.GetSymbol():<3}')
            info.SetIsHeteroAtom(True)
            info.SetOccupancy(1.0)
            info.SetTempFactor(val)
            a.SetPDBResidueInfo(info)

    pdb_str = Chem.MolToPDBBlock(_rdmol)

    if filename is not None:
        with open(filename, 'w') as f:
            f.write(pdb_str)

    return pdb_str


def draw_grid(molecules, mols_per_row=5, fig_size=(200, 200),
              label=check_mol,
              highlight_atom=lambda atom: False,
              highlight_bond=lambda bond: False):

    draw_mols = []
    marked_atoms = []
    marked_bonds = []
    for mol in molecules:
        draw_mol = Chem.Mol(mol)  # copy
        Chem.SanitizeMol(draw_mol, sanitizeOps=SanitizeFlags.SANITIZE_NONE)
        AllChem.Compute2DCoords(draw_mol)
        draw_mol = Draw.rdMolDraw2D.PrepareMolForDrawing(draw_mol,
                                                         kekulize=False)
        draw_mols.append(draw_mol)
        marked_atoms.append([a.GetIdx() for a in draw_mol.GetAtoms() if highlight_atom(a)])
        marked_bonds.append([b.GetIdx() for b in draw_mol.GetBonds() if highlight_bond(b)])

    drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    drawOptions.prepareMolsBeforeDrawing = False
    drawOptions.highlightBondWidthMultiplier = 20

    return Draw.MolsToGridImage(draw_mols,
                                molsPerRow=mols_per_row,
                                subImgSize=fig_size,
                                drawOptions=drawOptions,
                                highlightAtomLists=marked_atoms,
                                highlightBondLists=marked_bonds,
                                legends=[f'[{i}] {label(mol)}' for
                                         i, mol in enumerate(draw_mols)])
