import warnings
from typing import Union, Iterable
import random
# import argparse
from argparse import Namespace

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import KekulizeException, AtomKekulizeException
import networkx as nx
from networkx.algorithms import isomorphism
from torch_scatter import scatter_add, scatter_mean


class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


#####


def sum_except_batch(x, indices):
    if len(x.size()) < 2:
        x = x.unsqueeze(-1)
    return scatter_add(x.sum(list(range(1, len(x.size())))), indices, dim=0)


def remove_mean_batch(x, batch_mask, dim_size=None):
    # Compute center of mass per sample
    mean = scatter_mean(x, batch_mask, dim=0, dim_size=dim_size)
    x = x - mean[batch_mask]
    return x, mean


def assert_mean_zero(x, batch_mask, thresh=1e-2, eps=1e-10):
    largest_value = x.abs().max().item()
    error = scatter_add(x, batch_mask, dim=0).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < thresh, f'Mean is not zero, relative_error {rel_error}'


def bvm(v, m):
    """
    Batched vector-matrix product of the form out = v @ m
    :param v: (b, n_in)
    :param m: (b, n_in, n_out)
    :return: (b, n_out)
    """
    # return (v.unsqueeze(1) @ m).squeeze()
    return torch.bmm(v.unsqueeze(1), m).squeeze(1)


def get_grad_norm(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
        norm_type: float = 2.0) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.)

    device = parameters[0].grad.device

    total_norm = torch.norm(torch.stack(
        [torch.norm(p.grad.detach(), norm_type).to(device) for p in
         parameters]), norm_type)

    return total_norm


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, 'w') as f:
        f.write(out)


def write_sdf_file(sdf_path, molecules, catch_errors=True, connected=False):
    with Chem.SDWriter(str(sdf_path)) as w:
        for mol in molecules:
            try:
                if mol is None:
                    raise ValueError("Mol is None.")
                w.write(get_largest_connected_component(mol) if connected else mol)

            except (RuntimeError, ValueError) as e:
                if not catch_errors:
                    raise e

                if isinstance(e, (KekulizeException, AtomKekulizeException)):
                    w.SetKekulize(False)
                    w.write(get_largest_connected_component(mol) if connected else mol)
                    w.SetKekulize(True)
                    warnings.warn(f"Mol saved without kekulization.")
                else:
                    # write empty mol to preserve the original order
                    w.write(Chem.Mol())
                    warnings.warn(f"Erroneous mol replaced with empty dummy.")


def get_largest_connected_component(mol):
    try:
        frags = Chem.GetMolFrags(mol, asMols=True)
        newmol = max(frags, key=lambda m: m.GetNumAtoms())
    except:
        newmol = mol
    return newmol


def write_chain(filename, rdmol_chain):
    with open(filename, 'w') as f:
        f.write("".join([Chem.MolToXYZBlock(m) for m in rdmol_chain]))


def combine_sdfs(sdf_list, out_file):
    all_content = []
    for sdf in sdf_list:
        with open(sdf, 'r') as f:
            all_content.append(f.read())
    combined_str = '$$$$\n'.join(all_content)
    with open(out_file, 'w') as f:
        f.write(combined_str)


def batch_to_list(data, batch_mask, keep_order=True):
    if keep_order:  # preserve order of elements within each sample
        data_list = [data[batch_mask == i]
                     for i in torch.unique(batch_mask, sorted=True)]
        return data_list

    # make sure batch_mask is increasing
    idx = torch.argsort(batch_mask)
    batch_mask = batch_mask[idx]
    data = data[idx]

    chunk_sizes = torch.unique(batch_mask, return_counts=True)[1].tolist()
    return torch.split(data, chunk_sizes)


def batch_to_list_for_indices(indices, batch_mask, offsets=None):
    # (2, n) -> (n, 2)
    split = batch_to_list(indices.T, batch_mask)

    # rebase indices at zero & (n, 2) -> (2, n)
    if offsets is None:
        warnings.warn("Trying to infer index offset from smallest element in "
                      "batch. This might be wrong.")
        split = [x.T - x.min() for x in split]
    else:
        # Typically 'offsets' would be accumulate(sizes[:-1], initial=0)
        assert len(offsets) == len(split) or indices.numel() == 0
        split = [x.T - offset for x, offset in zip(split, offsets)]

    return split


def num_nodes_to_batch_mask(n_samples, num_nodes, device):
    assert isinstance(num_nodes, int) or len(num_nodes) == n_samples

    if isinstance(num_nodes, torch.Tensor):
        num_nodes = num_nodes.to(device)

    sample_inds = torch.arange(n_samples, device=device)

    return torch.repeat_interleave(sample_inds, num_nodes)


def rdmol_to_nxgraph(rdmol):
    graph = nx.Graph()
    for atom in rdmol.GetAtoms():
        # Add the atoms as nodes
        graph.add_node(atom.GetIdx(), atom_type=atom.GetAtomicNum())

    # Add the bonds as edges
    for bond in rdmol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    return graph


def calc_rmsd(mol_a, mol_b):
    """ Calculate RMSD of two molecules with unknown atom correspondence. """
    graph_a = rdmol_to_nxgraph(mol_a)
    graph_b = rdmol_to_nxgraph(mol_b)

    gm = isomorphism.GraphMatcher(
        graph_a, graph_b,
        node_match=lambda na, nb: na['atom_type'] == nb['atom_type'])

    isomorphisms = list(gm.isomorphisms_iter())
    if len(isomorphisms) < 1:
        return None

    all_rmsds = []
    for mapping in isomorphisms:
        atom_types_a = [atom.GetAtomicNum() for atom in mol_a.GetAtoms()]
        atom_types_b = [mol_b.GetAtomWithIdx(mapping[i]).GetAtomicNum()
                        for i in range(mol_b.GetNumAtoms())]
        assert atom_types_a == atom_types_b

        conf_a = mol_a.GetConformer()
        coords_a = np.array([conf_a.GetAtomPosition(i)
                             for i in range(mol_a.GetNumAtoms())])
        conf_b = mol_b.GetConformer()
        coords_b = np.array([conf_b.GetAtomPosition(mapping[i])
                             for i in range(mol_b.GetNumAtoms())])

        diff = coords_a - coords_b
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        all_rmsds.append(rmsd)

    if len(isomorphisms) > 1:
        print("More than one isomorphism found. Returning minimum RMSD.")

    return min(all_rmsds)


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def disable_rdkit_logging():
    # RDLogger.DisableLog('rdApp.*')
    RDLogger.DisableLog('rdApp.info')
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')


# class Namespace(argparse.Namespace):
#     """Simple definition of a Namespace class that supports Namespace- and
#     dictionary-like access."""
#     def __getitem__(self, key):
#         # return vars(self)[key]
#         return self.__dict__[key]
#
#     def __setitem__(self, key, value):
#         self.__dict__[key] = value
#
#     def __getattr__(self, item):
#         """Supports other dictionary functionalities, e.g. get(), items(), etc."""
#         # return getattr(vars(self), item)
#         return getattr(self.__dict__, item)


def dict_to_namespace(input_dict):
    """ Recursively convert a nested dictionary into a Namespace object. """
    if isinstance(input_dict, dict):
        output_namespace = Namespace()
        output = output_namespace.__dict__
        for key, value in input_dict.items():
            output[key] = dict_to_namespace(value)
        return output_namespace

    elif isinstance(input_dict, Namespace):
        # recurse as Namespace might contain dictionaries
        return dict_to_namespace(input_dict.__dict__)

    else:
        return input_dict


def namespace_to_dict(x):
    """ Recursively convert a nested Namespace object into a dictionary. """
    if not (isinstance(x, Namespace) or isinstance(x, dict)):
        return x

    if isinstance(x, Namespace):
        x = vars(x)

    # recurse
    output = {}
    for key, value in x.items():
        output[key] = namespace_to_dict(value)
    return output
