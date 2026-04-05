import io
from itertools import accumulate, chain
from copy import deepcopy
import random
import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from torch_scatter import scatter_mean
from Bio.PDB import StructureBuilder, Chain, Model, Structure
from Bio.PDB.PICIO import read_PIC, write_PIC
from scipy.ndimage import gaussian_filter
from pdb import set_trace

from src.constants import FLOAT_TYPE, INT_TYPE
from src.constants import atom_encoder, bond_encoder, aa_encoder, residue_encoder, residue_bond_encoder, aa_atom_index
from src import utils
from src.data.misc import protein_letters_3to1, is_aa
from src.data.normal_modes import pdb_to_normal_modes
from src.data.nerf import get_nerf_params, ic_to_coords
import src.data.so3_utils as so3


class TensorDict(dict):
    def __init__(self, **kwargs):
        super(TensorDict, self).__init__(**kwargs)

    def _apply(self, func: str, *args, **kwargs):
        """ Apply function to all tensors. """
        for k, v in self.items():
            if torch.is_tensor(v):
                self[k] = getattr(v, func)(*args, **kwargs)
        return self

    # def to(self, device):
    #     for k, v in self.items():
    #         if torch.is_tensor(v):
    #             self[k] = v.to(device)
    #     return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')
    
    def to(self, device):
        return self._apply("to", device)
    
    def detach(self):
        return self._apply("detach")

    def __repr__(self):
        def val_to_str(val):
            if isinstance(val, torch.Tensor):
                # if val.isnan().any():
                #     return "(!nan)"
                return "%r" % list(val.size())
            if isinstance(val, list):
                return "[%r,]" % len(val)
            else:
                return "?"

        return f"{type(self).__name__}({', '.join(f'{k}={val_to_str(v)}' for k, v in self.items())})"


def collate_entity(batch):

    out = {}
    for prop in batch[0].keys():

        if prop == 'name':
            out[prop] = [x[prop] for x in batch]

        elif prop == 'size' or prop == 'n_bonds':
            out[prop] = torch.tensor([x[prop] for x in batch])

        elif prop == 'bonds':
            # index offset
            offset = list(accumulate([x['size'] for x in batch], initial=0))
            out[prop] = torch.cat([x[prop] + offset[i] for i, x in enumerate(batch)], dim=1)

        elif prop == 'residues':
            out[prop] = list(chain.from_iterable(x[prop] for x in batch))

        elif prop in {'mask', 'bond_mask'}:
            pass  # batch masks will be written later

        else:
            out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        # Create batch masks
        # make sure indices in batch start at zero (needed for torch_scatter)
        if prop == 'x':
            out['mask'] = torch.cat([i * torch.ones(len(x[prop]), dtype=torch.int64, device=x[prop].device)
                                     for i, x in enumerate(batch)], dim=0)
        if prop == 'bond_one_hot':
            # TODO: this is not necessary as it can be computed on-the-fly as bond_mask = mask[bonds[0]] or bond_mask = mask[bonds[1]]
            out['bond_mask'] = torch.cat([i * torch.ones(len(x[prop]), dtype=torch.int64, device=x[prop].device)
                                          for i, x in enumerate(batch)], dim=0)

    return out


def split_entity(
        batch,
        *,
        index_types={'bonds'}, 
        edge_types={'bond_one_hot', 'bond_mask'}, 
        no_split={'name', 'size', 'n_bonds'}, 
        skip={'fragments'},
        batch_mask=None, 
        edge_mask=None
    ):
    """ Splits a batch into items and returns a list. """

    batch_mask = batch["mask"] if batch_mask is None else batch_mask
    edge_mask = batch["bond_mask"] if edge_mask is None else edge_mask
    sizes = batch['size'] if 'size' in batch else torch.unique(batch_mask, return_counts=True)[1].tolist()

    batch_size = len(torch.unique(batch['mask']))
    out = {}
    for prop in batch.keys():
        if prop in skip:
            continue
        if prop in no_split:
            out[prop] = batch[prop]  # already a list

        elif prop in index_types:
            offsets = list(accumulate(sizes[:-1], initial=0))
            out[prop] = utils.batch_to_list_for_indices(batch[prop], edge_mask, offsets)

        elif prop in edge_types:
            out[prop] = utils.batch_to_list(batch[prop], edge_mask)

        else:
            out[prop] = utils.batch_to_list(batch[prop], batch_mask)

    out = [{k: v[i] for k, v in out.items()} for i in range(batch_size)]
    return out


def repeat_items(batch, repeats):
    batch_list = split_entity(batch)
    out = collate_entity([x for _ in range(repeats) for x in batch_list])
    return type(batch)(**out)


def get_side_chain_bead_coord(biopython_residue):
    """
    Places side chain bead at the location of the farthest side chain atom.
    """
    if biopython_residue.get_resname() == 'GLY':
        return None
    if biopython_residue.get_resname() == 'ALA':
        return biopython_residue['CB'].get_coord()

    ca_coord = biopython_residue['CA'].get_coord()
    side_chain_atoms = [a for a in biopython_residue.get_atoms() if
                        a.id not in {'N', 'CA', 'C', 'O'} and a.element != 'H']
    side_chain_coords = np.stack([a.get_coord() for a in side_chain_atoms])

    atom_idx = np.argmax(np.sum((side_chain_coords - ca_coord[None, :]) ** 2, axis=-1))

    return side_chain_coords[atom_idx, :]


def get_side_chain_vectors(res, index_dict, size=None):
    if size is None:
        size = max([x for aa in index_dict.values() for x in aa.values()]) + 1

    resname = protein_letters_3to1[res.get_resname()]

    out = np.zeros((size, 3))
    for atom in res.get_atoms():
        if atom.get_name() in index_dict[resname]:
            idx = index_dict[resname][atom.get_name()]
            out[idx] = atom.get_coord() - res['CA'].get_coord()
        # else:
        #     if atom.get_name() != 'CA' and not atom.get_name().startswith('H'):
        #         print(resname, atom.get_name())

    return out


def get_normal_modes(res, normal_mode_dict):
    nm = normal_mode_dict[(res.get_parent().id, res.id[1], 'CA')]  # (n_modes, 3)
    return nm


def get_torsion_angles(res, device=None):
    """
    Return the five chi angles. Missing angles are filled with zeros.
    """
    ANGLES = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']

    ic_res = res.internal_coord
    chi_angles = [ic_res.get_angle(chi) for chi in ANGLES]
    chi_angles = [chi if chi is not None else float('nan') for chi in chi_angles]

    return torch.tensor(chi_angles, device=device) * np.pi / 180


def apply_torsion_angles(res, chi_angles):
    """
    Set side chain torsion angles of a biopython residue object with
    internal coordinates.
    """
    ANGLES = ['chi1', 'chi2', 'chi3', 'chi4', 'chi5']

    chi_angles = chi_angles * 180 / np.pi

    # res.parent.internal_coord.build_atomArray()  # rebuild atom pointers

    ic_res = res.internal_coord
    for chi, angle in zip(ANGLES, chi_angles):
        if ic_res.pick_angle(chi) is None:
            continue
        ic_res.bond_set(chi, angle)

    res.parent.internal_to_atom_coordinates(verbose=False)
    # res.parent.internal_coord.init_atom_coords()
    # res.internal_coord.assemble()

    return res


def prepare_internal_coord(res):

    # Make new structure with a single residue
    new_struct = Structure.Structure('X')
    new_struct.header = {}
    new_model = Model.Model(0)
    new_struct.add(new_model)
    new_chain = Chain.Chain('X')
    new_model.add(new_chain)
    new_chain.add(res)
    res.set_parent(new_chain)  # update pointer

    # Compute internal coordinates
    new_chain.atom_to_internal_coordinates()

    pic_io = io.StringIO()
    write_PIC(new_struct, pic_io)
    return pic_io.getvalue()


def residue_from_internal_coord(ic_string):
    pic_io = io.StringIO(ic_string)
    struct = read_PIC(pic_io, quick=True)
    res = struct.child_list[0].child_list[0].child_list[0]
    res.parent.internal_to_atom_coordinates(verbose=False)
    return res


def prepare_pocket(biopython_residues, amino_acid_encoder, residue_encoder,
                   residue_bond_encoder, pocket_representation='side_chain_bead',
                   compute_nerf_params=False, compute_bb_frames=False,
                   nma_input=None):

    assert nma_input is None or pocket_representation == 'CA+', \
        "vector features are only supported for CA+ pockets"

    # sort residues
    biopython_residues = sorted(biopython_residues, key=lambda x: (x.parent.id, x.id[1]))

    if nma_input is not None:
        # preprocessed normal mode eigenvectors
        if isinstance(nma_input, dict):
            nma_dict = nma_input

        # PDB file
        else:
            nma_dict = pdb_to_normal_modes(str(nma_input))

    if pocket_representation == 'side_chain_bead':
        ca_coords = np.zeros((len(biopython_residues), 3))
        ca_types = np.zeros(len(biopython_residues), dtype='int64')
        side_chain_coords = []
        side_chain_aa_types = []
        edges = []  # CA-CA and CA-side_chain
        edge_types = []
        last_res_id = None
        for i, res in enumerate(biopython_residues):
            aa = amino_acid_encoder[protein_letters_3to1[res.get_resname()]]
            ca_coords[i, :] = res['CA'].get_coord()
            ca_types[i] = aa
            side_chain_coord = get_side_chain_bead_coord(res)
            if side_chain_coord is not None:
                side_chain_coords.append(side_chain_coord)
                side_chain_aa_types.append(aa)
                edges.append((i, len(ca_coords) + len(side_chain_coords) - 1))
                edge_types.append(residue_bond_encoder['CA-SS'])

            # add edges between contiguous CA atoms
            if i > 0 and res.id[1] == last_res_id + 1:
                edges.append((i - 1, i))
                edge_types.append(residue_bond_encoder['CA-CA'])

            last_res_id = res.id[1]

        # Coordinates
        side_chain_coords = np.stack(side_chain_coords)
        pocket_coords = np.concatenate([ca_coords, side_chain_coords], axis=0)
        pocket_coords = torch.from_numpy(pocket_coords)

        # Features
        amino_acid_onehot = F.one_hot(
            torch.cat([torch.from_numpy(ca_types), torch.tensor(side_chain_aa_types, dtype=torch.int64)], dim=0),
            num_classes=len(amino_acid_encoder)
        )
        side_chain_onehot = np.concatenate([
            np.tile(np.eye(1, len(residue_encoder), residue_encoder['CA']),
                    [len(ca_coords), 1]),
            np.tile(np.eye(1, len(residue_encoder), residue_encoder['SS']),
                    [len(side_chain_coords), 1])
        ], axis=0)
        side_chain_onehot = torch.from_numpy(side_chain_onehot)
        pocket_onehot = torch.cat([amino_acid_onehot, side_chain_onehot], dim=1)

        vector_features = None
        nma_features = None

        # Bonds
        edges = torch.tensor(edges).T
        edge_types = F.one_hot(torch.tensor(edge_types), num_classes=len(residue_bond_encoder))

    elif pocket_representation == 'CA+':
        ca_coords = np.zeros((len(biopython_residues), 3))
        ca_types = np.zeros(len(biopython_residues), dtype='int64')

        v_dim = max([x for aa in aa_atom_index.values() for x in aa.values()]) + 1
        vec_feats = np.zeros((len(biopython_residues), v_dim, 3), dtype='float32')
        nf_nma = 5
        nma_feats = np.zeros((len(biopython_residues), nf_nma, 3), dtype='float32')

        edges = []  # CA-CA and CA-side_chain
        edge_types = []
        last_res_id = None
        for i, res in enumerate(biopython_residues):
            aa = amino_acid_encoder[protein_letters_3to1[res.get_resname()]]
            ca_coords[i, :] = res['CA'].get_coord()
            ca_types[i] = aa

            vec_feats[i] = get_side_chain_vectors(res, aa_atom_index, v_dim)
            if nma_input is not None:
                nma_feats[i] = get_normal_modes(res, nma_dict)

            # add edges between contiguous CA atoms
            if i > 0 and res.id[1] == last_res_id + 1:
                edges.append((i - 1, i))
                edge_types.append(residue_bond_encoder['CA-CA'])

            last_res_id = res.id[1]

        # Coordinates
        pocket_coords = torch.from_numpy(ca_coords)

        # Features
        pocket_onehot = F.one_hot(torch.from_numpy(ca_types),
                                  num_classes=len(amino_acid_encoder))

        vector_features = torch.from_numpy(vec_feats)
        nma_features = torch.from_numpy(nma_feats)

        # Bonds
        if len(edges) < 1:
            edges = torch.empty(2, 0)
            edge_types = torch.empty(0, len(residue_bond_encoder))
        else:
            edges = torch.tensor(edges).T
            edge_types = F.one_hot(torch.tensor(edge_types),
                                   num_classes=len(residue_bond_encoder))

    else:
        raise NotImplementedError(
            f"Pocket representation '{pocket_representation}' not implemented")

    # pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in biopython_residues]

    pocket = {
        'x': pocket_coords.to(dtype=FLOAT_TYPE),
        'one_hot': pocket_onehot.to(dtype=FLOAT_TYPE),
        # 'ids': pocket_ids,
        'size': torch.tensor([len(pocket_coords)], dtype=INT_TYPE),
        'mask': torch.zeros(len(pocket_coords), dtype=INT_TYPE),
        'bonds': edges.to(INT_TYPE),
        'bond_one_hot': edge_types.to(FLOAT_TYPE),
        'bond_mask': torch.zeros(edges.size(1), dtype=INT_TYPE),
        'n_bonds': torch.tensor([len(edge_types)], dtype=INT_TYPE),
    }

    if vector_features is not None:
        pocket['v'] = vector_features.to(dtype=FLOAT_TYPE)

    if nma_input is not None:
        pocket['nma_vec'] = nma_features.to(dtype=FLOAT_TYPE)

    if compute_nerf_params:
        nerf_params = [get_nerf_params(r) for r in biopython_residues]
        nerf_params = {k: torch.stack([x[k] for x in nerf_params], dim=0)
                       for k in nerf_params[0].keys()}
        pocket.update(nerf_params)

    if compute_bb_frames:
        n_xyz = torch.from_numpy(np.stack([r['N'].get_coord() for r in biopython_residues]))
        ca_xyz = torch.from_numpy(np.stack([r['CA'].get_coord() for r in biopython_residues]))
        c_xyz = torch.from_numpy(np.stack([r['C'].get_coord() for r in biopython_residues]))
        pocket['axis_angle'], _ = get_bb_transform(n_xyz, ca_xyz, c_xyz)

    return pocket, biopython_residues


def encode_atom(rd_atom, atom_encoder):
    element = rd_atom.GetSymbol().capitalize()

    explicitHs = rd_atom.GetNumExplicitHs()
    if explicitHs == 1 and f'{element}H' in atom_encoder:
        return atom_encoder[f'{element}H']

    charge = rd_atom.GetFormalCharge()
    if charge == 1 and f'{element}+' in atom_encoder:
        return atom_encoder[f'{element}+']
    if charge == -1 and f'{element}-' in atom_encoder:
        return atom_encoder[f'{element}-']

    return atom_encoder[element]


def prepare_ligand(rdmol, atom_encoder, bond_encoder):

    # remove H atoms if not in atom_encoder
    if 'H' not in atom_encoder:
        rdmol = Chem.RemoveAllHs(rdmol, sanitize=False)

    # Coordinates
    ligand_coord = rdmol.GetConformer().GetPositions()
    ligand_coord = torch.from_numpy(ligand_coord)

    # Features
    ligand_onehot = F.one_hot(
        torch.tensor([encode_atom(a, atom_encoder) for a in rdmol.GetAtoms()]),
        num_classes=len(atom_encoder)
    )

    # Bonds
    adj = np.ones((rdmol.GetNumAtoms(), rdmol.GetNumAtoms())) * bond_encoder['NOBOND']
    for b in rdmol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        adj[i, j] = bond_encoder[str(b.GetBondType())]
        adj[j, i] = adj[i, j]  # undirected graph

    # molecular graph is undirected -> don't save redundant information
    bonds = np.stack(np.triu_indices(len(ligand_coord), k=1), axis=0)
    # bonds = np.stack(np.ones_like(adj).nonzero(), axis=0)
    bond_types = adj[bonds[0], bonds[1]].astype('int64')
    bonds = torch.from_numpy(bonds)
    bond_types = F.one_hot(torch.from_numpy(bond_types), num_classes=len(bond_encoder))

    ligand = {
        'x': ligand_coord.to(dtype=FLOAT_TYPE),
        'one_hot': ligand_onehot.to(dtype=FLOAT_TYPE),
        'mask': torch.zeros(len(ligand_coord), dtype=INT_TYPE),
        'bonds': bonds.to(INT_TYPE),
        'bond_one_hot': bond_types.to(FLOAT_TYPE),
        'bond_mask': torch.zeros(bonds.size(1), dtype=INT_TYPE),
        'size': torch.tensor([len(ligand_coord)], dtype=INT_TYPE),
        'n_bonds': torch.tensor([len(bond_types)], dtype=INT_TYPE),
    }

    return ligand


def process_raw_molecule_with_empty_pocket(rdmol):
    ligand = prepare_ligand(rdmol, atom_encoder, bond_encoder)
    pocket = {
        'x': torch.tensor([], dtype=FLOAT_TYPE),
        'one_hot': torch.tensor([], dtype=FLOAT_TYPE),
        'size': torch.tensor([], dtype=INT_TYPE),
        'mask': torch.tensor([], dtype=INT_TYPE),
        'bonds': torch.tensor([], dtype=INT_TYPE),
        'bond_one_hot': torch.tensor([], dtype=FLOAT_TYPE),
        'bond_mask': torch.tensor([], dtype=INT_TYPE),
        'n_bonds': torch.tensor([], dtype=INT_TYPE),
    }
    return ligand, pocket


def process_raw_pair(biopython_model, rdmol, dist_cutoff=None,
                     pocket_representation='side_chain_bead',
                     compute_nerf_params=False, compute_bb_frames=False,
                     nma_input=None, return_pocket_pdb=False):

    # Process ligand
    ligand = prepare_ligand(rdmol, atom_encoder, bond_encoder)

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in biopython_model.get_residues():

        # Remove non-standard amino acids and HETATMs
        if not is_aa(residue.get_resname(), standard=True):
            continue

        res_coords = torch.from_numpy(np.array([a.get_coord() for a in residue.get_atoms()]))
        if dist_cutoff is None or (((res_coords[:, None, :] - ligand['x'][None, :, :]) ** 2).sum(-1) ** 0.5).min() < dist_cutoff:
            pocket_residues.append(residue)

    pocket, pocket_residues = prepare_pocket(
        pocket_residues, aa_encoder, residue_encoder, residue_bond_encoder,
        pocket_representation, compute_nerf_params, compute_bb_frames, nma_input
    )

    if return_pocket_pdb:
        builder = StructureBuilder.StructureBuilder()
        builder.init_structure("")
        builder.init_model(0)
        pocket_struct = builder.get_structure()
        for residue in pocket_residues:
            chain = residue.get_parent().get_id()

            # init chain if necessary
            if not pocket_struct[0].has_id(chain):
                builder.init_chain(chain)

            # add residue
            pocket_struct[0][chain].add(residue)

        pocket['pocket_pdb'] = pocket_struct
    # if return_pocket_pdb:
    #     pocket['residues'] = [prepare_internal_coord(res) for res in pocket_residues]

    return ligand, pocket


class AppendVirtualNodes:
    def __init__(self, atom_encoder, bond_encoder, max_ligand_size, scale=1.0):
        self.max_size = max_ligand_size
        self.atom_encoder = atom_encoder
        self.bond_encoder = bond_encoder
        self.vidx = atom_encoder['NOATOM']
        self.bidx = bond_encoder['NOBOND']
        self.scale = scale

    def __call__(self, ligand, max_size=None, eps=1e-6):
        if max_size is None:
            max_size = self.max_size

        n_virt = max_size - ligand['size']

        C = torch.cov(ligand['x'].T)
        L = torch.linalg.cholesky(C + torch.eye(3) * eps)
        mu = ligand['x'].mean(0, keepdim=True)
        virt_coords = mu + torch.randn(n_virt, 3) @ L.T * self.scale

        # insert virtual atom column
        virt_one_hot = F.one_hot(torch.ones(n_virt, dtype=torch.int64) * self.vidx, num_classes=len(self.atom_encoder))
        virt_mask = torch.cat([torch.zeros(ligand['size'], dtype=bool), torch.ones(n_virt, dtype=bool)])

        ligand['x'] = torch.cat([ligand['x'], virt_coords])
        ligand['one_hot'] = torch.cat(([ligand['one_hot'], virt_one_hot]))
        ligand['virtual_mask'] = virt_mask
        ligand['size'] = max_size

        # Bonds
        new_bonds = torch.triu_indices(max_size, max_size, offset=1)

        bond_types = torch.ones(max_size, max_size, dtype=INT_TYPE) * self.bidx
        row, col = ligand['bonds']
        bond_types[row, col] = ligand['bond_one_hot'].argmax(dim=1)
        new_row, new_col = new_bonds
        bond_types = bond_types[new_row, new_col]

        ligand['bonds'] = new_bonds
        ligand['bond_one_hot'] = F.one_hot(bond_types, num_classes=len(self.bond_encoder)).to(ligand['bond_one_hot'].dtype)
        ligand['n_bonds'] = len(ligand['bond_one_hot'])

        return ligand


class AppendVirtualNodesInCoM:
    def __init__(self, atom_encoder, bond_encoder, add_min=0, add_max=10):
        self.atom_encoder = atom_encoder
        self.bond_encoder = bond_encoder
        self.vidx = atom_encoder['NOATOM']
        self.bidx = bond_encoder['NOBOND']
        self.add_min = add_min
        self.add_max = add_max

    def __call__(self, ligand):

        n_virt = random.randint(self.add_min, self.add_max)

        # all virtual coordinates in the CoM
        virt_coords = ligand['x'].mean(0, keepdim=True).repeat(n_virt, 1)

        # insert virtual atom column
        virt_one_hot = F.one_hot(torch.ones(n_virt, dtype=torch.int64) * self.vidx, num_classes=len(self.atom_encoder))
        virt_mask = torch.cat([torch.zeros(ligand['size'], dtype=bool), torch.ones(n_virt, dtype=bool)])

        ligand['x'] = torch.cat([ligand['x'], virt_coords])
        ligand['one_hot'] = torch.cat(([ligand['one_hot'], virt_one_hot]))
        ligand['virtual_mask'] = virt_mask
        ligand['size'] = len(ligand['x'])

        # Bonds
        new_bonds = torch.triu_indices(ligand['size'], ligand['size'], offset=1)

        bond_types = torch.ones(ligand['size'], ligand['size'], dtype=INT_TYPE) * self.bidx
        row, col = ligand['bonds']
        bond_types[row, col] = ligand['bond_one_hot'].argmax(dim=1)
        new_row, new_col = new_bonds
        bond_types = bond_types[new_row, new_col]

        ligand['bonds'] = new_bonds
        ligand['bond_one_hot'] = F.one_hot(bond_types, num_classes=len(self.bond_encoder)).to(ligand['bond_one_hot'].dtype)
        ligand['n_bonds'] = len(ligand['bond_one_hot'])

        return ligand


def rdmol_to_smiles(rdmol):
    mol = Chem.Mol(rdmol)
    Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


def get_n_nodes(lig_positions, pocket_positions, smooth_sigma=None):
    # Joint distribution of ligand's and pocket's number of nodes
    n_nodes_lig = [len(x) for x in lig_positions]
    n_nodes_pocket = [len(x) for x in pocket_positions]

    joint_histogram = np.zeros((np.max(n_nodes_lig) + 1,
                                np.max(n_nodes_pocket) + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(f'Original histogram: {np.count_nonzero(joint_histogram)}/'
          f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram, sigma=smooth_sigma, order=0, mode='constant',
            cval=0.0, truncate=4.0)

        print(f'Smoothed histogram: {np.count_nonzero(filtered_histogram)}/'
              f'{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled')

        joint_histogram = filtered_histogram

    return joint_histogram


# def get_type_histograms(lig_one_hot, pocket_one_hot, lig_encoder, pocket_encoder):
#
#     lig_one_hot = np.concatenate(lig_one_hot, axis=0)
#     pocket_one_hot = np.concatenate(pocket_one_hot, axis=0)
#
#     atom_decoder = list(lig_encoder.keys())
#     lig_counts = {k: 0 for k in lig_encoder.keys()}
#     for a in [atom_decoder[x] for x in lig_one_hot.argmax(1)]:
#         lig_counts[a] += 1
#
#     aa_decoder = list(pocket_encoder.keys())
#     pocket_counts = {k: 0 for k in pocket_encoder.keys()}
#     for r in [aa_decoder[x] for x in pocket_one_hot.argmax(1)]:
#         pocket_counts[r] += 1
#
#     return lig_counts, pocket_counts


def get_type_histogram(one_hot, type_encoder):

    one_hot = np.concatenate(one_hot, axis=0)

    decoder = list(type_encoder.keys())
    counts = {k: 0 for k in type_encoder.keys()}
    for a in [decoder[x] for x in one_hot.argmax(1)]:
        counts[a] += 1

    return counts


def get_residue_with_resi(pdb_chain, resi):
    res = [x for x in pdb_chain.get_residues() if x.id[1] == resi]
    assert len(res) == 1
    return res[0]


def get_pocket_from_ligand(pdb_model, ligand, dist_cutoff=8.0):

    if ligand.endswith(".sdf"):
        # ligand as sdf file
        rdmol = Chem.SDMolSupplier(str(ligand))[0]
        ligand_coords = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        resi = None
    else:
        # ligand contained in PDB; given in <chain>:<resi> format
        chain, resi = ligand.split(':')
        ligand = get_residue_with_resi(pdb_model[chain], int(resi))
        ligand_coords = torch.from_numpy(
            np.array([a.get_coord() for a in ligand.get_atoms()]))

    pocket_residues = []
    for residue in pdb_model.get_residues():
        if residue.id[1] == resi:
            continue  # skip ligand itself

        res_coords = torch.from_numpy(
            np.array([a.get_coord() for a in residue.get_atoms()]))
        if is_aa(residue.get_resname(), standard=True) \
                and torch.cdist(res_coords, ligand_coords).min() < dist_cutoff:
            pocket_residues.append(residue)

    return pocket_residues


def encode_residues(biopython_residues, type_encoder, level='atom',
                    remove_H=True):
    assert level in {'atom', 'residue'}

    if level == 'atom':
        entities = [a for res in biopython_residues for a in res.get_atoms()
                    if (a.element != 'H' or not remove_H)]
        types = [a.element.capitalize() for a in entities]
    else:
        entities = [res['CA'] for res in biopython_residues]
        types = [protein_letters_3to1[res.get_resname()] for res in biopython_residues]

    coord = torch.tensor(np.stack([e.get_coord() for e in entities]))
    one_hot = F.one_hot(torch.tensor([type_encoder[t] for t in types]),
                        num_classes=len(type_encoder))

    return coord, one_hot


def center_data(ligand, pocket):
    if pocket['x'].numel() > 0:
        pocket_com = pocket.center()
    else:
        pocket_com = scatter_mean(ligand['x'], ligand['mask'], dim=0)

    ligand['x'] = ligand['x'] - pocket_com[ligand['mask']]
    return ligand, pocket


def get_bb_transform(n_xyz, ca_xyz, c_xyz):
    """
    Compute translation and rotation of the canoncical backbone frame (triangle N-Ca-C) from a position with
    Ca at the origin, N on the x-axis and C in the xy-plane to the global position of the backbone frame

    Args:
        n_xyz: (n, 3)
        ca_xyz: (n, 3)
        c_xyz: (n, 3)

    Returns:
        axis-angle representation of the rotation, shape (n, 3)  # rotation matrix of shape (n, 3, 3)
        translation vector of shape (n, 3)
    """

    def rotation_matrix(angle, axis):
        axis_mapping = {'x': 0, 'y': 1, 'z': 2}
        axis = axis_mapping[axis]
        vector = torch.zeros(len(angle), 3)
        vector[:, axis] = 1
        # return axis_angle_to_matrix(angle * vector)
        return so3.matrix_from_rotation_vector(angle.view(-1, 1) * vector)

    translation = ca_xyz
    n_xyz = n_xyz - translation
    c_xyz = c_xyz - translation

    # Find rotation matrix that aligns the coordinate systems

    # rotate around y-axis to move N into the xy-plane
    theta_y = torch.arctan2(n_xyz[:, 2], -n_xyz[:, 0])
    Ry = rotation_matrix(theta_y, 'y')
    Ry = Ry.transpose(2, 1)
    n_xyz = torch.einsum('noi,ni->no', Ry, n_xyz)

    # rotate around z-axis to move N onto the x-axis
    theta_z = torch.arctan2(n_xyz[:, 1], n_xyz[:, 0])
    Rz = rotation_matrix(theta_z, 'z')
    Rz = Rz.transpose(2, 1)
    # print(torch.einsum('noi,ni->no', Rz, n_xyz))

    # n_xyz = torch.einsum('noi,ni->no', Rz.transpose(0, 2, 1), n_xyz)

    # rotate around x-axis to move C into the xy-plane
    c_xyz = torch.einsum('noj,nji,ni->no', Rz, Ry, c_xyz)
    theta_x = torch.arctan2(c_xyz[:, 2], c_xyz[:, 1])
    Rx = rotation_matrix(theta_x, 'x')
    Rx = Rx.transpose(2, 1)
    # print(torch.einsum('noi,ni->no', Rx, c_xyz))

    # Final rotation matrix
    Ry = Ry.transpose(2, 1)
    Rz = Rz.transpose(2, 1)
    Rx = Rx.transpose(2, 1)
    R = torch.einsum('nok,nkj,nji->noi', Ry, Rz, Rx)

    # return R, translation
    # return matrix_to_axis_angle(R), translation
    return so3.rotation_vector_from_matrix(R), translation


class Residues(TensorDict):
    """
    Dictionary-like container for residues that supports some basic transformations.
    """

    # all keys
    KEYS = {'x', 'one_hot', 'bonds', 'bond_one_hot', 'v', 'nma_vec', 'fixed_coord',
            'atom_mask', 'nerf_indices', 'length', 'theta', 'chi', 'ddihedral',
            'chi_indices', 'axis_angle', 'mask', 'bond_mask'}

    # coordinate-type values, shape (..., 3)
    COORD_KEYS = {'x', 'fixed_coord'}

    # vector-type values, shape (n_residues, n_feat, 3)
    VECTOR_KEYS = {'v', 'nma_vec'}

    # properties that change if the side chains and/or backbones are updated
    MUTABLE_PROPS_SS_AND_BB = {'v'}

    # properties that only change if the side chains are updated
    MUTABLE_PROPS_SS = {'chi'}

    # properties that only change if the backbones are updated
    MUTABLE_PROPS_BB = {'x', 'fixed_coord', 'axis_angle', 'nma_vec'}

    # properties that remain fixed in all cases
    IMMUTABLE_PROPS = {'mask', 'one_hot', 'bonds', 'bond_one_hot', 'bond_mask',
                       'atom_mask', 'nerf_indices', 'length', 'theta',
                       'ddihedral', 'chi_indices', 'name', 'size', 'n_bonds'}

    def copy(self):
        data = super().copy()
        return Residues(**data)

    def deepcopy(self):
        data = {k: v.clone() if torch.is_tensor(v) else deepcopy(v)
                for k, v in self.items()}
        return Residues(**data)

    def center(self):
        com = scatter_mean(self['x'], self['mask'], dim=0)
        self['x'] = self['x'] - com[self['mask']]
        self['fixed_coord'] = self['fixed_coord'] - com[self['mask']].unsqueeze(1)
        return com

    def set_empty_v(self):
        self['v'] = torch.tensor([], device=self['x'].device)

    @torch.no_grad()
    def set_chi(self, chi_angles):
        self['chi'][:, :5] = chi_angles
        nerf_params = {k: self[k] for k in ['fixed_coord', 'atom_mask',
                                            'nerf_indices', 'length', 'theta',
                                            'chi', 'ddihedral', 'chi_indices']}
        self['v'] = ic_to_coords(**nerf_params) - self['x'].unsqueeze(1)

    @torch.no_grad()
    def set_frame(self, new_ca_coord, new_axis_angle):
        bb_coord = self['fixed_coord']
        bb_coord = bb_coord - self['x'].unsqueeze(1)
        rotmat_before = so3.matrix_from_rotation_vector(self['axis_angle'])
        rotmat_after = so3.matrix_from_rotation_vector(new_axis_angle)
        rotmat_diff = rotmat_after @ rotmat_before.transpose(-1, -2)
        bb_coord = torch.einsum('boi,bai->bao', rotmat_diff, bb_coord)
        bb_coord = bb_coord + new_ca_coord.unsqueeze(1)

        self['x'] = new_ca_coord
        self['axis_angle'] = new_axis_angle
        self['fixed_coord'] = bb_coord
        self['v'] = torch.einsum('boi,bai->bao', rotmat_diff, self['v'])

    @staticmethod
    def empty(device):
        return Residues(
            x=torch.zeros(1, 3, device=device).float(),
            mask=torch.zeros(1, 1, device=device).long(),
            size=torch.zeros(1, device=device).long(),
        )


def randomize_tensors(tensor_dict, exclude_keys=None):
    """Replace tensors with random tensors with the same shape."""
    exclude_keys = set() if exclude_keys is None else set(exclude_keys)
    for k, v in tensor_dict.items():
        if isinstance(v, torch.Tensor) and k not in exclude_keys:
            if torch.is_floating_point(v):
                tensor_dict[k] = torch.randn_like(v)
            else:
                tensor_dict[k] = torch.randint_like(v, low=-42, high=42)
    return tensor_dict
