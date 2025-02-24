from collections.abc import Iterable
from abc import abstractmethod
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.constants import INT_TYPE
from src.model.gvp import GVPModel, GVP, LayerNorm
from src.model.gvp_transformer import GVPTransformerModel
from src.constants import FLOAT_TYPE

from pdb import set_trace


def binomial_coefficient(n, k):
    # source: https://discuss.pytorch.org/t/n-choose-k-function/121974
    return ((n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()).exp()


def cycle_counts(adj):
    assert (adj.diag() == 0).all()
    assert (adj == adj.T).all()

    A = adj.float()
    d = A.sum(dim=-1)

    # Compute powers
    A2 = A @ A
    A3 = A2 @ A
    A4 = A3 @ A
    A5 = A4 @ A

    x3 = A3.diag() / 2
    x4 = (A4.diag() - d * (d - 1) - A @ d) / 2

    """ New (different from DiGress)
    case where correction is relevant:
    2   o
        |
    1,3 o--o 4
        | /
    0,5 o
    """
    # Triangle count matrix (indicates for each node i how many triangles it shares with node j)
    T = adj * A2
    x5 = (A5.diag() - 2 * T @ d - 4 * d * x3 - 2 * A @ x3 + 10 * x3) / 2

    # # TODO
    # A6 = A5 @ A
    #
    # # 4-cycle count matrix (indicates in how many shared 4-cycles i and j are 2 hops apart)
    # Q2 = binomial_coefficient(n=A2 - d.diag(), k=torch.tensor(2))
    #
    # # 4-cycle count matrix (indicates in how many shared 4-cycles i and j are 1 (and 3) hop(s) apart)
    # Q1 = A * (A3 - (d.view(-1, 1) + d.view(1, -1)) + 1)  # "+1" because link between i and j is subtracted twice
    #
    # x6 = ...
    # return torch.stack([x3, x4, x5, x6], dim=-1)

    return torch.stack([x3, x4, x5], dim=-1)


# TODO: also consider directional aggregation as in:
#  Beaini, Dominique, et al. "Directional graph networks."
#  International Conference on Machine Learning. PMLR, 2021.
def eigenfeatures(A, batch_mask, k=5):
    # TODO, see:
    # - https://github.com/cvignac/DiGress/blob/main/src/diffusion/extra_features.py
    # - https://arxiv.org/pdf/2209.14734.pdf (Appendix B.2)

    # split adjacency matrix
    batch = []
    for i in torch.unique(batch_mask, sorted=True):  # TODO: optimize (try to avoid loop)
        batch_inds = torch.where(batch_mask == i)[0]
        batch.append(A[torch.meshgrid(batch_inds, batch_inds, indexing='ij')])

    eigenfeats = [get_nontrivial_eigenvectors(adj)[:, :k] for adj in batch]
    # if there are less than k non-trivial eigenvectors
    eigenfeats = [torch.cat([
        x, torch.zeros(x.size(0), max(k - x.size(1), 0), device=x.device)], dim=-1)
        for x in eigenfeats]
    return torch.cat(eigenfeats, dim=0)


def get_nontrivial_eigenvectors(A, normalize_l=True, thresh=1e-5,
                                norm_eps=1e-12):
    """
    Compute eigenvectors of the graph Laplacian corresponding to non-zero
    eigenvalues.
    """
    assert (A == A.T).all(), "undirected graph"

    # Compute laplacian
    d = A.sum(-1)
    D = d.diag()
    L = D - A

    if normalize_l:
        D_inv_sqrt = (1 / (d.sqrt() + norm_eps)).diag()
        L = D_inv_sqrt @ L @ D_inv_sqrt

    # Eigendecomposition
    # eigenvalues are sorted in ascending order
    # eigvecs matrix contains eigenvectors as its columns
    eigvals, eigvecs = torch.linalg.eigh(L)

    # index of first non-trivial eigenvector
    try:
        idx = torch.nonzero(eigvals > thresh)[0].item()
    except IndexError:
        # recover if no non-trivial eigenvectors are found
        idx = eigvecs.size(1)

    return eigvecs[:, idx:]


class DynamicsBase(nn.Module):
    """
    Implements self-conditioning logic and basic functions
    """
    def __init__(
            self,
            predict_angles=False,
            predict_frames=False,
            add_cycle_counts=False,
            add_spectral_feat=False,
            self_conditioning=False,
            augment_residue_sc=False,
            augment_ligand_sc=False
    ):
        super().__init__()

        if not hasattr(self, 'predict_angles'):
            self.predict_angles = predict_angles

        if not hasattr(self, 'predict_frames'):
            self.predict_frames = predict_frames

        if not hasattr(self, 'add_cycle_counts'):
            self.add_cycle_counts = add_cycle_counts

        if not hasattr(self, 'add_spectral_feat'):
            self.add_spectral_feat = add_spectral_feat

        if not hasattr(self, 'self_conditioning'):
            self.self_conditioning = self_conditioning

        if not hasattr(self, 'augment_residue_sc'):
            self.augment_residue_sc = augment_residue_sc

        if not hasattr(self, 'augment_ligand_sc'):
            self.augment_ligand_sc = augment_ligand_sc

        if self.self_conditioning:
            self.prev_ligand = None
            self.prev_residues = None

    @abstractmethod
    def _forward(self, x_atoms, h_atoms, mask_atoms, pocket, t, bonds_ligand=None,
                 h_atoms_sc=None, e_atoms_sc=None, h_residues_sc=None):
        """
        Implement forward pass.
        Returns:
            - vel
            - h_final_atoms
            - edge_final_atoms
            - residue_angles
            - residue_trans
            - residue_rot
        """
        pass

    def make_sc_input(self, pred_ligand, pred_residues, sc_transform):

        if self.predict_confidence:
            h_atoms_sc = (torch.cat([pred_ligand['logits_h'], pred_ligand['uncertainty_vel'].unsqueeze(1)], dim=-1),
                          pred_ligand['vel'].unsqueeze(1))
        else:
            h_atoms_sc = (pred_ligand['logits_h'], pred_ligand['vel'].unsqueeze(1))
        e_atoms_sc = pred_ligand['logits_e']

        if self.predict_frames:
            h_residues_sc = (torch.cat([pred_residues['chi'], pred_residues['rot']], dim=-1),
                             pred_residues['trans'].unsqueeze(1))
        elif self.predict_angles:
            h_residues_sc = pred_residues['chi']
        else:
            h_residues_sc = None

        if self.augment_residue_sc and h_residues_sc is not None:
            if self.predict_frames:
                h_residues_sc = (h_residues_sc[0], torch.cat(
                    [h_residues_sc[1], sc_transform['residues'](pred_residues['chi'], pred_residues['trans'].squeeze(1), pred_residues['rot'])], dim=1))

            else:
                h_residues_sc = (h_residues_sc, sc_transform['residues'](pred_residues['chi']))

        if self.augment_ligand_sc:
            h_atoms_sc = (h_atoms_sc[0], torch.cat(
                [h_atoms_sc[1], sc_transform['atoms'](pred_ligand['vel'].unsqueeze(1))], dim=1))

        return h_atoms_sc, e_atoms_sc, h_residues_sc

    def forward(self, x_atoms, h_atoms, mask_atoms, pocket, t, bonds_ligand=None, sc_transform=None):
        """
        Implements self-conditioning as in https://arxiv.org/abs/2208.04202
        """

        h_atoms_sc, e_atoms_sc = None, None
        h_residues_sc = None

        if self.self_conditioning:

            # Sampling: use previous prediction in all but the first time step
            if not self.training and t.min() > 0.0:
                assert t.min() == t.max(), "currently only supports sampling at same time steps"
                assert self.prev_ligand is not None
                assert self.prev_residues is not None or not self.predict_frames

            else:
                # Create zero tensors
                zeros_ligand = {'logits_h': torch.zeros_like(h_atoms),
                                'vel': torch.zeros_like(x_atoms),
                                'logits_e': torch.zeros_like(bonds_ligand[1])}
                if self.predict_confidence:
                    zeros_ligand['uncertainty_vel'] = torch.zeros(
                        len(x_atoms), dtype=x_atoms.dtype, device=x_atoms.device)

                zeros_residues = {}
                if self.predict_angles:
                    zeros_residues['chi'] = torch.zeros((pocket['one_hot'].size(0), 5), device=pocket['one_hot'].device)
                if self.predict_frames:
                    zeros_residues['trans'] = torch.zeros((pocket['one_hot'].size(0), 3), device=pocket['one_hot'].device)
                    zeros_residues['rot'] = torch.zeros((pocket['one_hot'].size(0), 3), device=pocket['one_hot'].device)

                # Training: use 50% zeros and 50% predictions with detached gradients
                if self.training and random.random() > 0.5:
                    with torch.no_grad():
                        h_atoms_sc, e_atoms_sc, h_residues_sc = self.make_sc_input(
                            zeros_ligand, zeros_residues, sc_transform)

                        self.prev_ligand, self.prev_residues = self._forward(
                            x_atoms, h_atoms, mask_atoms, pocket, t, bonds_ligand,
                            h_atoms_sc, e_atoms_sc, h_residues_sc)

                # use zeros for first sampling step and 50% of training
                else:
                    self.prev_ligand = zeros_ligand
                    self.prev_residues = zeros_residues

            h_atoms_sc, e_atoms_sc, h_residues_sc = self.make_sc_input(
                self.prev_ligand, self.prev_residues, sc_transform)

        pred_ligand, pred_residues = self._forward(
            x_atoms, h_atoms, mask_atoms, pocket, t, bonds_ligand,
            h_atoms_sc, e_atoms_sc, h_residues_sc
        )

        if self.self_conditioning and not self.training:
            self.prev_ligand = pred_ligand.copy()
            self.prev_residues = pred_residues.copy()

        return pred_ligand, pred_residues

    def compute_extra_features(self, batch_mask, edge_indices, edge_types):

        feat = torch.zeros(len(batch_mask), 0, device=batch_mask.device)

        if not (self.add_cycle_counts or self.add_spectral_feat):
            return feat

        adj = batch_mask[:, None] == batch_mask[None, :]

        E = torch.zeros_like(adj, dtype=INT_TYPE)
        E[edge_indices[0], edge_indices[1]] = edge_types

        A = (E > 0).float()

        if self.add_cycle_counts:
            cycle_features = cycle_counts(A)
            cycle_features[cycle_features > 10] = 10  # avoid large values

            feat = torch.cat([feat, cycle_features], dim=-1)

        if self.add_spectral_feat:
            feat = torch.cat([feat, eigenfeatures(A, batch_mask)], dim=-1)

        return feat


class Dynamics(DynamicsBase):
    def __init__(self, atom_nf, residue_nf, joint_nf, bond_dict, pocket_bond_dict,
                 edge_nf, hidden_nf, act_fn=torch.nn.SiLU(), condition_time=True,
                 model='egnn', model_params=None,
                 edge_cutoff_ligand=None, edge_cutoff_pocket=None,
                 edge_cutoff_interaction=None,
                 predict_angles=False, predict_frames=False,
                 add_cycle_counts=False, add_spectral_feat=False,
                 add_nma_feat=False, self_conditioning=False,
                 augment_residue_sc=False, augment_ligand_sc=False,
                 add_chi_as_feature=False, angle_act_fn=False):
        super().__init__()
        self.model = model
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.hidden_nf = hidden_nf
        self.predict_angles = predict_angles
        self.predict_frames = predict_frames
        self.bond_dict = bond_dict
        self.pocket_bond_dict = pocket_bond_dict
        self.bond_nf = len(bond_dict)
        self.pocket_bond_nf = len(pocket_bond_dict)
        self.edge_nf = edge_nf
        self.add_cycle_counts = add_cycle_counts
        self.add_spectral_feat = add_spectral_feat
        self.add_nma_feat = add_nma_feat
        self.self_conditioning = self_conditioning
        self.augment_residue_sc = augment_residue_sc
        self.augment_ligand_sc = augment_ligand_sc
        self.add_chi_as_feature = add_chi_as_feature
        self.predict_confidence = False

        if self.self_conditioning:
            self.prev_vel = None
            self.prev_h = None
            self.prev_e = None
            self.prev_a = None
            self.prev_ca = None
            self.prev_rot = None

        lig_nf = atom_nf
        if self.add_cycle_counts:
            lig_nf = lig_nf + 3
        if self.add_spectral_feat:
            lig_nf = lig_nf + 5


        if not isinstance(joint_nf, Iterable):
            # joint_nf contains only scalars
            joint_nf = (joint_nf, 0)


        if isinstance(residue_nf, Iterable):
            _atom_in_nf = (lig_nf, 0)
            _residue_atom_dim = residue_nf[1]

            if self.add_nma_feat:
                residue_nf = (residue_nf[0], residue_nf[1] + 5)

            if self.self_conditioning:
                _atom_in_nf = (_atom_in_nf[0] + atom_nf, 1)

                if self.augment_ligand_sc:
                    _atom_in_nf = (_atom_in_nf[0], _atom_in_nf[1] + 1)

                if self.predict_angles:
                    residue_nf = (residue_nf[0] + 5, residue_nf[1])

                if self.predict_frames:
                    residue_nf = (residue_nf[0], residue_nf[1] + 2)

                if self.augment_residue_sc:
                    assert self.predict_angles
                    residue_nf = (residue_nf[0], residue_nf[1] + _residue_atom_dim)

            if self.add_chi_as_feature:
                residue_nf = (residue_nf[0] + 5, residue_nf[1])

            self.atom_encoder = nn.Sequential(
                GVP(_atom_in_nf, joint_nf, activations=(act_fn, torch.sigmoid)),
                LayerNorm(joint_nf, learnable_vector_weight=True),
                GVP(joint_nf, joint_nf, activations=(None, None)),
            )

            self.residue_encoder = nn.Sequential(
                GVP(residue_nf, joint_nf, activations=(act_fn, torch.sigmoid)),
                LayerNorm(joint_nf, learnable_vector_weight=True),
                GVP(joint_nf, joint_nf, activations=(None, None)),
            )

        else:
            # No vector-valued input features
            assert joint_nf[1] == 0

            # self-conditioning not yet supported
            assert not self.self_conditioning

            # Normal mode features are vectors
            assert not self.add_nma_feat

            if self.add_chi_as_feature:
                residue_nf += 5

            self.atom_encoder = nn.Sequential(
                nn.Linear(lig_nf, 2 * atom_nf),
                act_fn,
                nn.Linear(2 * atom_nf, joint_nf[0])
            )

            self.residue_encoder = nn.Sequential(
                nn.Linear(residue_nf, 2 * residue_nf),
                act_fn,
                nn.Linear(2 * residue_nf, joint_nf[0])
            )

        self.atom_decoder = nn.Sequential(
            nn.Linear(joint_nf[0], 2 * atom_nf),
            act_fn,
            nn.Linear(2 * atom_nf, atom_nf)
        )

        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, self.bond_nf)
        )

        _atom_bond_nf = 2 * self.bond_nf if self.self_conditioning else self.bond_nf
        self.ligand_bond_encoder = nn.Sequential(
            nn.Linear(_atom_bond_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, self.edge_nf)
        )

        self.pocket_bond_encoder = nn.Sequential(
            nn.Linear(self.pocket_bond_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, self.edge_nf)
        )

        out_nf = (joint_nf[0], 1)
        res_out_nf = (0, 0)
        if self.predict_angles:
            res_out_nf = (res_out_nf[0] + 5, res_out_nf[1])
        if self.predict_frames:
            res_out_nf = (res_out_nf[0], res_out_nf[1] + 2)
        self.residue_decoder = nn.Sequential(
            GVP(out_nf, out_nf, activations=(act_fn, torch.sigmoid)),
            LayerNorm(out_nf, learnable_vector_weight=True),
            GVP(out_nf, res_out_nf, activations=(None, None)),
        ) if res_out_nf != (0, 0) else None

        if angle_act_fn is None:
            self.angle_act_fn = None
        elif angle_act_fn == 'tanh':
            self.angle_act_fn = lambda x: np.pi * F.tanh(x)
        else:
            raise NotImplementedError(f"Angle activation {angle_act_fn} not available")

        # self.ligand_nobond_emb = nn.Parameter(torch.zeros(self.edge_nf))
        # self.pocket_nobond_emb = nn.Parameter(torch.zeros(self.edge_nf))
        self.cross_emb = nn.Parameter(torch.zeros(self.edge_nf),
                                      requires_grad=True)

        if condition_time:
            dynamics_node_nf = (joint_nf[0] + 1, joint_nf[1])
        else:
            print('Warning: dynamics model is NOT conditioned on time.')
            dynamics_node_nf = (joint_nf[0], joint_nf[1])

        if model == 'egnn':
            raise NotImplementedError
            # self.net = EGNN(
            #     in_node_nf=dynamics_node_nf[0], in_edge_nf=self.edge_nf,
            #     hidden_nf=hidden_nf, out_node_nf=joint_nf[0],
            #     device=model_params.device, act_fn=act_fn,
            #     n_layers=model_params.n_layers,
            #     attention=model_params.attention,
            #     tanh=model_params.tanh,
            #     norm_constant=model_params.norm_constant,
            #     inv_sublayers=model_params.inv_sublayers,
            #     sin_embedding=model_params.sin_embedding,
            #     normalization_factor=model_params.normalization_factor,
            #     aggregation_method=model_params.aggregation_method,
            #     reflection_equiv=model_params.reflection_equivariant,
            #     update_edge_attr=True
            # )
            # self.node_nf = dynamics_node_nf[0]

        elif model == 'gvp':
            self.net = GVPModel(
                node_in_dim=dynamics_node_nf, node_h_dim=model_params.node_h_dim,
                node_out_nf=joint_nf[0], edge_in_nf=self.edge_nf,
                edge_h_dim=model_params.edge_h_dim, edge_out_nf=hidden_nf,
                num_layers=model_params.n_layers,
                drop_rate=model_params.dropout,
                vector_gate=model_params.vector_gate,
                reflection_equiv=model_params.reflection_equivariant,
                d_max=model_params.d_max,
                num_rbf=model_params.num_rbf,
                update_edge_attr=True
            )

        elif model == 'gvp_transformer':
            self.net = GVPTransformerModel(
                node_in_dim=dynamics_node_nf,
                node_h_dim=model_params.node_h_dim,
                node_out_nf=joint_nf[0],
                edge_in_nf=self.edge_nf,
                edge_h_dim=model_params.edge_h_dim,
                edge_out_nf=hidden_nf,
                num_layers=model_params.n_layers,
                dk=model_params.dk,
                dv=model_params.dv,
                de=model_params.de,
                db=model_params.db,
                dy=model_params.dy,
                attn_heads=model_params.attn_heads,
                n_feedforward=model_params.n_feedforward,
                drop_rate=model_params.dropout,
                reflection_equiv=model_params.reflection_equivariant,
                d_max=model_params.d_max,
                num_rbf=model_params.num_rbf,
                vector_gate=model_params.vector_gate,
                attention=model_params.attention,
            )

        elif model == 'gnn':
            raise NotImplementedError
            # n_dims = 3
            # self.net = GNN(
            #     in_node_nf=dynamics_node_nf + n_dims, in_edge_nf=self.edge_emb_dim,
            #     hidden_nf=hidden_nf, out_node_nf=n_dims + dynamics_node_nf,
            #     device=model_params.device, act_fn=act_fn, n_layers=model_params.n_layers,
            #     attention=model_params.attention, normalization_factor=model_params.normalization_factor,
            #     aggregation_method=model_params.aggregation_method)

        else:
            raise NotImplementedError(f"{model} is not available")

        # self.device = device
        # self.n_dims = n_dims
        self.condition_time = condition_time

    def _forward(self, x_atoms, h_atoms, mask_atoms, pocket, t, bonds_ligand=None,
                h_atoms_sc=None, e_atoms_sc=None, h_residues_sc=None):
        """
        :param x_atoms:
        :param h_atoms:
        :param mask_atoms:
        :param pocket: must contain keys: 'x', 'one_hot', 'mask', 'bonds' and 'bond_one_hot'
        :param t:
        :param bonds_ligand: tuple - bond indices (2, n_bonds) & bond types (n_bonds, bond_nf)
        :param h_atoms_sc: additional node feature for self-conditioning, (s, V)
        :param e_atoms_sc: additional edge feature for self-conditioning, only scalar
        :param h_residues_sc: additional node feature for self-conditioning, tensor or tuple
        :return:
        """
        x_residues, h_residues, mask_residues = pocket['x'], pocket['one_hot'], pocket['mask']
        if 'bonds' in pocket:
            bonds_pocket = (pocket['bonds'], pocket['bond_one_hot'])
        else:
            bonds_pocket = None

        if self.add_chi_as_feature:
            h_residues = torch.cat([h_residues, pocket['chi'][:, :5]], dim=-1)

        if 'v' in pocket:
            v_residues = pocket['v']
            if self.add_nma_feat:
                v_residues = torch.cat([v_residues, pocket['nma_vec']], dim=1)
            h_residues = (h_residues, v_residues)

        if h_residues_sc is not None:
            # if self.augment_residue_sc:
            if isinstance(h_residues_sc, tuple):
                h_residues = (torch.cat([h_residues[0], h_residues_sc[0]], dim=-1),
                              torch.cat([h_residues[1], h_residues_sc[1]], dim=1))
            else:
                h_residues = (torch.cat([h_residues[0], h_residues_sc], dim=-1),
                              h_residues[1])

        # get graph edges and edge attributes
        if bonds_ligand is not None:
            # NOTE: 'bond' denotes one-directional edges and 'edge' means bi-directional
            ligand_bond_indices = bonds_ligand[0]

            # make sure messages are passed both ways
            ligand_edge_indices = torch.cat(
                [bonds_ligand[0], bonds_ligand[0].flip(dims=[0])], dim=1)
            ligand_edge_types = torch.cat([bonds_ligand[1], bonds_ligand[1]], dim=0)
            # edges_ligand = (ligand_edge_indices, ligand_edge_types)

            # add auxiliary features to ligand nodes
            extra_features = self.compute_extra_features(
                mask_atoms, ligand_edge_indices, ligand_edge_types.argmax(-1))
            h_atoms = torch.cat([h_atoms, extra_features], dim=-1)

        if bonds_pocket is not None:
            # make sure messages are passed both ways
            pocket_edge_indices = torch.cat(
                [bonds_pocket[0], bonds_pocket[0].flip(dims=[0])], dim=1)
            pocket_edge_types = torch.cat([bonds_pocket[1], bonds_pocket[1]], dim=0)
            # edges_pocket = (pocket_edge_indices, pocket_edge_types)

        if h_atoms_sc is not None:
            h_atoms = (torch.cat([h_atoms, h_atoms_sc[0]], dim=-1),
                       h_atoms_sc[1])

        if e_atoms_sc is not None:
            e_atoms_sc = torch.cat([e_atoms_sc, e_atoms_sc], dim=0)
            ligand_edge_types = torch.cat([ligand_edge_types, e_atoms_sc], dim=-1)

        # embed atom features and residue features in a shared space
        h_atoms = self.atom_encoder(h_atoms)
        e_ligand = self.ligand_bond_encoder(ligand_edge_types)

        if len(h_residues) > 0:
            h_residues = self.residue_encoder(h_residues)
            e_pocket = self.pocket_bond_encoder(pocket_edge_types)
        else:
            e_pocket = pocket_edge_types
            h_residues = (h_residues, h_residues)
            pocket_edge_indices = torch.tensor([[], []], dtype=torch.long, device=h_residues[0].device)
            pocket_edge_types = torch.tensor([[], []], dtype=torch.long, device=h_residues[0].device)

        if isinstance(h_atoms, tuple):
            h_atoms, v_atoms = h_atoms
            h_residues, v_residues = h_residues
            v = torch.cat((v_atoms, v_residues), dim=0)
        else:
            v = None

        edges, edge_feat = self.get_edges(
            mask_atoms, mask_residues, x_atoms, x_residues,
            bond_inds_ligand=ligand_edge_indices, bond_inds_pocket=pocket_edge_indices,
            bond_feat_ligand=e_ligand, bond_feat_pocket=e_pocket)

        # combine the two node types
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        mask = torch.cat([mask_atoms, mask_residues])

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[mask]
            h = torch.cat([h, h_time], dim=1)

        assert torch.all(mask[edges[0]] == mask[edges[1]])

        if self.model == 'egnn':
            # Don't update pocket coordinates
            update_coords_mask = torch.cat((torch.ones_like(mask_atoms),
                                            torch.zeros_like(mask_residues))).unsqueeze(1)
            h_final, vel, edge_final = self.net(
                h, x, edges,  batch_mask=mask, edge_attr=edge_feat,
                update_coords_mask=update_coords_mask)
            # vel = (x_final - x)

        elif self.model == 'gvp' or self.model == 'gvp_transformer':
            h_final, vel, edge_final = self.net(
                h, x, edges, v=v, batch_mask=mask, edge_attr=edge_feat)

        elif self.model == 'gnn':
            xh = torch.cat([x, h], dim=1)
            output = self.net(xh, edges, node_mask=None, edge_attr=edge_feat)
            vel = output[:, :3]
            h_final = output[:, 3:]

        else:
            raise NotImplementedError(f"Wrong model ({self.model})")

        # if self.condition_time:
        #     # Slice off last dimension which represented time.
        #     h_final = h_final[:, :-1]

        # decode atom and residue features
        h_final_atoms = self.atom_decoder(h_final[:len(mask_atoms)])

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final_atoms)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
                h_final_atoms[torch.isnan(h_final_atoms)] = 0.0
            else:
                raise ValueError("NaN detected in network output")

        # predict edge type
        ligand_edge_mask = (edges[0] < len(mask_atoms)) & (edges[1] < len(mask_atoms))
        edge_final = edge_final[ligand_edge_mask]
        edges = edges[:, ligand_edge_mask]

        # Symmetrize
        edge_logits = torch.zeros(
            (len(mask_atoms), len(mask_atoms), self.hidden_nf),
            device=mask_atoms.device)
        edge_logits[edges[0], edges[1]] = edge_final
        edge_logits = (edge_logits + edge_logits.transpose(0, 1)) * 0.5
        # edge_logits = edge_logits[lig_edge_indices[0], lig_edge_indices[1]]

        # return upper triangular elements only (matching the input)
        edge_logits = edge_logits[ligand_bond_indices[0], ligand_bond_indices[1]]
        # assert (edge_logits == 0).sum() == 0

        edge_final_atoms = self.edge_decoder(edge_logits)

        # Predict torsion angles
        residue_angles = None
        residue_trans, residue_rot = None, None
        if self.residue_decoder is not None:
            h_residues = h_final[len(mask_atoms):]
            vec_residues = vel[len(mask_atoms):].unsqueeze(1)
            residue_angles = self.residue_decoder((h_residues, vec_residues))
            if self.predict_frames:
                residue_angles, residue_frames = residue_angles
                residue_trans = residue_frames[:, 0, :].squeeze(1)
                residue_rot = residue_frames[:, 1, :].squeeze(1)
            if self.angle_act_fn is not None:
                residue_angles = self.angle_act_fn(residue_angles)

        # return vel[:len(mask_atoms)], h_final_atoms, edge_final_atoms, residue_angles, residue_trans, residue_rot
        pred_ligand = {'vel': vel[:len(mask_atoms)], 'logits_h': h_final_atoms, 'logits_e': edge_final_atoms}
        pred_residues = {'chi': residue_angles, 'trans': residue_trans, 'rot': residue_rot}
        return pred_ligand, pred_residues

    def get_edges(self, batch_mask_ligand, batch_mask_pocket, x_ligand,
                  x_pocket, bond_inds_ligand=None, bond_inds_pocket=None,
                  bond_feat_ligand=None, bond_feat_pocket=None, self_edges=False):

        # Adjacency matrix
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)

            # Add missing bonds if they got removed
            adj_ligand[bond_inds_ligand[0], bond_inds_ligand[1]] = True

        if self.edge_cutoff_p is not None and len(x_pocket) > 0:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)

            # Add missing bonds if they got removed
            adj_pocket[bond_inds_pocket[0], bond_inds_pocket[1]] = True

        if self.edge_cutoff_i is not None and len(x_pocket) > 0:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)

        adj = torch.cat((torch.cat((adj_ligand, adj_cross), dim=1),
                         torch.cat((adj_cross.T, adj_pocket), dim=1)), dim=0)

        if not self_edges:
            adj = adj ^ torch.eye(*adj.size(), out=torch.empty_like(adj))

        # # ensure that edge definition is consistent if bonds are provided (for loss computation)
        # if bond_inds_ligand is not None:
        #     # remove ligand edges
        #     adj[:adj_ligand.size(0), :adj_ligand.size(1)] = False
        #     edges = torch.stack(torch.where(adj), dim=0)
        #     # add ligand edges back with original definition
        #     edges = torch.cat([bond_inds_ligand, edges], dim=-1)
        # else:
        #     edges = torch.stack(torch.where(adj), dim=0)

        # Feature matrix
        ligand_nobond_onehot = F.one_hot(torch.tensor(
            self.bond_dict['NOBOND'], device=bond_feat_ligand.device),
            num_classes=self.ligand_bond_encoder[0].in_features)
        ligand_nobond_emb = self.ligand_bond_encoder(
            ligand_nobond_onehot.to(FLOAT_TYPE))
        feat_ligand = ligand_nobond_emb.repeat(*adj_ligand.shape, 1)
        feat_ligand[bond_inds_ligand[0], bond_inds_ligand[1]] = bond_feat_ligand

        if len(adj_pocket) > 0:
            pocket_nobond_onehot = F.one_hot(torch.tensor(
                self.pocket_bond_dict['NOBOND'], device=bond_feat_pocket.device),
                num_classes=self.pocket_bond_nf)
            pocket_nobond_emb = self.pocket_bond_encoder(
                pocket_nobond_onehot.to(FLOAT_TYPE))
            feat_pocket = pocket_nobond_emb.repeat(*adj_pocket.shape, 1)
            feat_pocket[bond_inds_pocket[0], bond_inds_pocket[1]] = bond_feat_pocket

            feat_cross = self.cross_emb.repeat(*adj_cross.shape, 1)

            feats = torch.cat((torch.cat((feat_ligand, feat_cross), dim=1),
                               torch.cat((feat_cross.transpose(0, 1), feat_pocket), dim=1)), dim=0)
        else:
            feats = feat_ligand

        # Return results
        edges = torch.stack(torch.where(adj), dim=0)
        edge_feat = feats[edges[0], edges[1]]

        return edges, edge_feat
