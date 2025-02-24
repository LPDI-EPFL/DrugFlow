from collections.abc import Iterable
from collections import defaultdict
from functools import partial
import functools
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.utils.hetero import check_add_self_loops
try:
    from torch_geometric.nn.conv.hgt_conv import group
except ImportError as e:
    from torch_geometric.nn.conv.hetero_conv import group

from src.model.dynamics import DynamicsBase
from src.model import gvp
from src.model.gvp import GVP, _rbf, _normalize, tuple_index, tuple_sum, _split, tuple_cat, _merge


class MyModuleDict(nn.ModuleDict):
    def __init__(self, modules):
        # a mapping (dictionary) of (string: module) or an iterable of key-value pairs of type (string, module)
        if isinstance(modules, dict):
            super().__init__({str(k): v for k, v in modules.items()})
        else:
            raise NotImplementedError

    def __getitem__(self, key):
        return super().__getitem__(str(key))

    def __setitem__(self, key, value):
        super().__setitem__(str(key), value)

    def __delitem__(self, key):
        super().__delitem__(str(key))


class MyHeteroConv(nn.Module):
    """
    Implementation from PyG 2.2.0 with minor changes.
    Override forward pass to control the final aggregation
    Ref.: https://pytorch-geometric.readthedocs.io/en/2.2.0/_modules/torch_geometric/nn/conv/hetero_conv.html
    """
    def __init__(self, convs, aggr="sum"):
        self.vo = {}
        for k, module in convs.items():
            dst = k[-1]
            if dst not in self.vo:
                self.vo[dst] = module.vo
            else:
                assert self.vo[dst] == module.vo

        # from the original implementation in PyTorch Geometric
        super().__init__()

        for edge_type, module in convs.items():
            check_add_self_loops(module, [edge_type])

        src_node_types = set([key[0] for key in convs.keys()])
        dst_node_types = set([key[-1] for key in convs.keys()])
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behaviour.")

        self.convs = ModuleDict({'__'.join(k): v for k, v in convs.items()})
        self.aggr = aggr

    def reset_parameters(self):
        for conv in self.convs.values():
            conv.reset_parameters()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'

    def forward(
            self,
            x_dict,
            edge_index_dict,
            *args_dict,
            **kwargs_dict,
    ):
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                holding graph connectivity information for each individual
                edge type.
            *args_dict (optional): Additional forward arguments of invididual
                :class:`torch_geometric.nn.conv.MessagePassing` layers.
            **kwargs_dict (optional): Additional forward arguments of
                individual :class:`torch_geometric.nn.conv.MessagePassing`
                layers.
                For example, if a specific GNN layer at edge type
                :obj:`edge_type` expects edge attributes :obj:`edge_attr` as a
                forward argument, then you can pass them to
                :meth:`~torch_geometric.nn.conv.HeteroConv.forward` via
                :obj:`edge_attr_dict = { edge_type: edge_attr }`.
        """
        out_dict = defaultdict(list)
        out_dict_edge = {}
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type

            str_edge_type = '__'.join(edge_type)
            if str_edge_type not in self.convs:
                continue

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append(
                        (value_dict.get(src, None), value_dict.get(dst, None)))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (value_dict.get(src, None),
                                   value_dict.get(dst, None))

            conv = self.convs[str_edge_type]

            if src == dst:
                out = conv(x_dict[src], edge_index, *args, **kwargs)
            else:
                out = conv((x_dict[src], x_dict[dst]), edge_index, *args,
                           **kwargs)

            if isinstance(out, (tuple, list)):
                out, out_edge = out
                out_dict_edge[edge_type] = out_edge

            out_dict[dst].append(out)

        for key, value in out_dict.items():
            out_dict[key] = group(value, self.aggr)
            out_dict[key] = _split(out_dict[key], self.vo[key])

        return out_dict if len(out_dict_edge) <= 0 else out_dict, out_dict_edge


class GVPHeteroConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    :param update_edge_attr: whether to compute an updated edge representation
    '''

    def __init__(self, in_dims, out_dims, edge_dims, in_dims_other=None,
                 n_layers=3, module_list=None, aggr="mean",
                 activations=(F.relu, torch.sigmoid), vector_gate=False,
                 update_edge_attr=False):
        super(GVPHeteroConv, self).__init__(aggr=aggr)

        if in_dims_other is None:
            in_dims_other = in_dims

        self.si, self.vi = in_dims
        self.si_other, self.vi_other = in_dims_other
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        self.update_edge_attr = update_edge_attr

        GVP_ = functools.partial(GVP,
                                 activations=activations,
                                 vector_gate=vector_gate)

        def get_modules(module_list, out_dims):
            module_list = module_list or []
            if not module_list:
                if n_layers == 1:
                    module_list.append(
                        GVP_((self.si + self.si_other + self.se, self.vi + self.vi_other + self.ve),
                             (self.so, self.vo), activations=(None, None)))
                else:
                    module_list.append(
                        GVP_((self.si + self.si_other + self.se, self.vi + self.vi_other + self.ve),
                             out_dims)
                    )
                    for i in range(n_layers - 2):
                        module_list.append(GVP_(out_dims, out_dims))
                    module_list.append(GVP_(out_dims, out_dims,
                                            activations=(None, None)))
            return nn.Sequential(*module_list)

        self.message_func = get_modules(module_list, out_dims)
        self.edge_func = get_modules(module_list, edge_dims) if self.update_edge_attr else None

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        elem_0, elem_1 = x
        if isinstance(elem_0, (tuple, list)):
            assert isinstance(elem_1, (tuple, list))
            x_s = (elem_0[0], elem_1[0])
            x_v = (elem_0[1].reshape(elem_0[1].shape[0], 3 * elem_0[1].shape[1]),
                   elem_1[1].reshape(elem_1[1].shape[0], 3 * elem_1[1].shape[1]))
        else:
            x_s, x_v = elem_0, elem_1
            x_v = x_v.reshape(x_v.shape[0], 3 * x_v.shape[1])

        message = self.propagate(edge_index, s=x_s, v=x_v, edge_attr=edge_attr)

        if self.update_edge_attr:
            if isinstance(x_s, (tuple, list)):
                s_i, s_j = x_s[1][edge_index[1]], x_s[0][edge_index[0]]
            else:
                s_i, s_j = x_s[edge_index[1]], x_s[edge_index[0]]

            if isinstance(x_v, (tuple, list)):
                v_i, v_j = x_v[1][edge_index[1]], x_v[0][edge_index[0]]
            else:
                v_i, v_j = x_v[edge_index[1]], x_v[edge_index[0]]

            edge_out = self.edge_attr(s_i, v_i, s_j, v_j, edge_attr)
            # return _split(message, self.vo), edge_out
            return message, edge_out
        else:
            # return _split(message, self.vo)
            return message

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)

    def edge_attr(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        return self.edge_func(message)


class GVPHeteroConvLayer(nn.Module):
    """
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param conv_dims: dictionary defining (src_dim, dst_dim, edge_dim) for each edge type
    """
    def __init__(self, conv_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 activations=(F.relu, torch.sigmoid), vector_gate=False,
                 update_edge_attr=False, ln_vector_weight=False):

        super(GVPHeteroConvLayer, self).__init__()
        self.update_edge_attr = update_edge_attr

        gvp_conv = partial(GVPHeteroConv,
                           n_layers=n_message,
                           aggr="sum",
                           activations=activations,
                           vector_gate=vector_gate,
                           update_edge_attr=update_edge_attr)

        def get_feedforward(n_dims):
            GVP_ = partial(GVP, activations=activations, vector_gate=vector_gate)

            ff_func = []
            if n_feedforward == 1:
                ff_func.append(GVP_(n_dims, n_dims, activations=(None, None)))
            else:
                hid_dims = 4 * n_dims[0], 2 * n_dims[1]
                ff_func.append(GVP_(n_dims, hid_dims))
                for i in range(n_feedforward - 2):
                    ff_func.append(GVP_(hid_dims, hid_dims))
                ff_func.append(GVP_(hid_dims, n_dims, activations=(None, None)))
            return nn.Sequential(*ff_func)

        # self.conv = HeteroConv({k: gvp_conv(*dims) for k, dims in conv_dims.items()}, aggr='sum')
        self.conv = MyHeteroConv({k: gvp_conv(*dims) for k, dims in conv_dims.items()}, aggr='sum')

        node_dims = {k[-1]: dims[1] for k, dims in conv_dims.items()}
        self.norm0 = MyModuleDict({k: gvp.LayerNorm(dims, ln_vector_weight) for k, dims in node_dims.items()})
        self.dropout0 = MyModuleDict({k: gvp.Dropout(drop_rate) for k, dims in node_dims.items()})
        self.ff_func = MyModuleDict({k: get_feedforward(dims) for k, dims in node_dims.items()})
        self.norm1 = MyModuleDict({k: gvp.LayerNorm(dims, ln_vector_weight) for k, dims in node_dims.items()})
        self.dropout1 = MyModuleDict({k: gvp.Dropout(drop_rate) for k, dims in node_dims.items()})

        if self.update_edge_attr:
            self.edge_norm0 = MyModuleDict({k: gvp.LayerNorm(dims[2], ln_vector_weight) for k, dims in conv_dims.items()})
            self.edge_dropout0 = MyModuleDict({k: gvp.Dropout(drop_rate) for k, dims in conv_dims.items()})
            self.edge_ff = MyModuleDict({k: get_feedforward(dims[2]) for k, dims in conv_dims.items()})
            self.edge_norm1 = MyModuleDict({k: gvp.LayerNorm(dims[2], ln_vector_weight) for k, dims in conv_dims.items()})
            self.edge_dropout1 = MyModuleDict({k: gvp.Dropout(drop_rate) for k, dims in conv_dims.items()})

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, node_mask_dict=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''

        dh_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict)

        if self.update_edge_attr:
            dh_dict, de_dict = dh_dict

            for k, edge_attr in edge_attr_dict.items():
                de = de_dict[k]

                edge_attr = self.edge_norm0[k](tuple_sum(edge_attr, self.edge_dropout0[k](de)))
                de = self.edge_ff[k](edge_attr)
                edge_attr = self.edge_norm1[k](tuple_sum(edge_attr, self.edge_dropout1[k](de)))

                edge_attr_dict[k] = edge_attr

        for k, x in x_dict.items():
            dh = dh_dict[k]
            node_mask = None if node_mask_dict is None else node_mask_dict[k]

            if node_mask is not None:
                x_ = x
                x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

            x = self.norm0[k](tuple_sum(x, self.dropout0[k](dh)))

            dh = self.ff_func[k](x)
            x = self.norm1[k](tuple_sum(x, self.dropout1[k](dh)))

            if node_mask is not None:
                x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
                x = x_

            x_dict[k] = x

        return (x_dict, edge_attr_dict) if self.update_edge_attr else x_dict


class GVPModel(torch.nn.Module):
    """
    GVP-GNN model
    inspired by: https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    and: https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/atom3d.py#L115
    """
    def __init__(self,
                 node_in_dim_ligand, node_in_dim_pocket,
                 edge_in_dim_ligand, edge_in_dim_pocket, edge_in_dim_interaction,
                 node_h_dim_ligand, node_h_dim_pocket,
                 edge_h_dim_ligand, edge_h_dim_pocket, edge_h_dim_interaction,
                 node_out_dim_ligand=None, node_out_dim_pocket=None,
                 edge_out_dim_ligand=None, edge_out_dim_pocket=None, edge_out_dim_interaction=None,
                 num_layers=3, drop_rate=0.1, vector_gate=False, update_edge_attr=False):

        super(GVPModel, self).__init__()

        self.update_edge_attr = update_edge_attr

        self.node_in = nn.ModuleDict({
            'ligand': GVP(node_in_dim_ligand, node_h_dim_ligand, activations=(None, None), vector_gate=vector_gate),
            'pocket': GVP(node_in_dim_pocket, node_h_dim_pocket, activations=(None, None), vector_gate=vector_gate),
        })
        # self.edge_in = MyModuleDict({
        #     ('ligand', 'ligand'): GVP(edge_in_dim_ligand, edge_h_dim_ligand, activations=(None, None), vector_gate=vector_gate),
        #     ('pocket', 'pocket'): GVP(edge_in_dim_pocket, edge_h_dim_pocket, activations=(None, None), vector_gate=vector_gate),
        #     ('ligand', 'pocket'): GVP(edge_in_dim_interaction, edge_h_dim_interaction, activations=(None, None), vector_gate=vector_gate),
        #     ('pocket', 'ligand'): GVP(edge_in_dim_interaction, edge_h_dim_interaction, activations=(None, None), vector_gate=vector_gate),
        # })
        self.edge_in = MyModuleDict({
            ('ligand', '', 'ligand'): GVP(edge_in_dim_ligand, edge_h_dim_ligand, activations=(None, None), vector_gate=vector_gate),
            ('pocket', '', 'pocket'): GVP(edge_in_dim_pocket, edge_h_dim_pocket, activations=(None, None), vector_gate=vector_gate),
            ('ligand', '', 'pocket'): GVP(edge_in_dim_interaction, edge_h_dim_interaction, activations=(None, None), vector_gate=vector_gate),
            ('pocket', '', 'ligand'): GVP(edge_in_dim_interaction, edge_h_dim_interaction, activations=(None, None), vector_gate=vector_gate),
        })

        # conv_dims = {
        #     ('ligand', 'ligand'): (node_h_dim_ligand, node_h_dim_ligand, edge_h_dim_ligand),
        #     ('pocket', 'pocket'): (node_h_dim_pocket, node_h_dim_pocket, edge_h_dim_pocket),
        #     ('ligand', 'pocket'): (node_h_dim_ligand, node_h_dim_pocket, edge_h_dim_interaction),
        #     ('pocket', 'ligand'): (node_h_dim_pocket, node_h_dim_ligand, edge_h_dim_interaction),
        # }
        conv_dims = {
            ('ligand', '', 'ligand'): (node_h_dim_ligand, node_h_dim_ligand, edge_h_dim_ligand),
            ('pocket', '', 'pocket'): (node_h_dim_pocket, node_h_dim_pocket, edge_h_dim_pocket),
            ('ligand', '', 'pocket'): (node_h_dim_ligand, node_h_dim_pocket, edge_h_dim_interaction, node_h_dim_pocket),
            ('pocket', '', 'ligand'): (node_h_dim_pocket, node_h_dim_ligand, edge_h_dim_interaction, node_h_dim_ligand),
        }

        self.layers = nn.ModuleList(
            GVPHeteroConvLayer(conv_dims,
                               drop_rate=drop_rate,
                               update_edge_attr=self.update_edge_attr,
                               activations=(F.relu, None),
                               vector_gate=vector_gate,
                               ln_vector_weight=True)
            for _ in range(num_layers))

        self.node_out = nn.ModuleDict({
            'ligand': GVP(node_h_dim_ligand, node_out_dim_ligand, activations=(None, None), vector_gate=vector_gate),
            'pocket': GVP(node_h_dim_pocket, node_out_dim_pocket, activations=(None, None), vector_gate=vector_gate) if node_out_dim_pocket is not None else None,
        })
        # self.edge_out = MyModuleDict({
        #     ('ligand', 'ligand'): GVP(edge_h_dim_ligand, edge_out_dim_ligand, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_ligand is not None else None,
        #     ('pocket', 'pocket'): GVP(edge_h_dim_pocket, edge_out_dim_pocket, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_pocket is not None else None,
        #     ('ligand', 'pocket'): GVP(edge_h_dim_interaction, edge_out_dim_interaction, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_interaction is not None else None,
        #     ('pocket', 'ligand'): GVP(edge_h_dim_interaction, edge_out_dim_interaction, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_interaction is not None else None,
        # })
        self.edge_out = MyModuleDict({
            ('ligand', '', 'ligand'): GVP(edge_h_dim_ligand, edge_out_dim_ligand, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_ligand is not None else None,
            ('pocket', '', 'pocket'): GVP(edge_h_dim_pocket, edge_out_dim_pocket, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_pocket is not None else None,
            ('ligand', '', 'pocket'): GVP(edge_h_dim_interaction, edge_out_dim_interaction, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_interaction is not None else None,
            ('pocket', '', 'ligand'): GVP(edge_h_dim_interaction, edge_out_dim_interaction, activations=(None, None), vector_gate=vector_gate) if edge_out_dim_interaction is not None else None,
        })

    def forward(self, node_attr, batch_mask, edge_index, edge_attr):

        # to hidden dimension
        for k in node_attr.keys():
            node_attr[k] = self.node_in[k](node_attr[k])

        for k in edge_attr.keys():
            edge_attr[k] = self.edge_in[k](edge_attr[k])

        # convolutions
        for layer in self.layers:
            out = layer(node_attr, edge_index, edge_attr)
            if self.update_edge_attr:
                node_attr, edge_attr = out
            else:
                node_attr = out

        # to output dimension
        for k in node_attr.keys():
            node_attr[k] = self.node_out[k](node_attr[k]) \
                if self.node_out[k] is not None else None

        if self.update_edge_attr:
            for k in edge_attr.keys():
                if self.edge_out[k] is not None:
                    edge_attr[k] = self.edge_out[k](edge_attr[k])

        return node_attr, edge_attr


class DynamicsHetero(DynamicsBase):
    def __init__(self, atom_nf, residue_nf, bond_dict, pocket_bond_dict,
                 condition_time=True,
                 num_rbf_time=None,
                 model='gvp',
                 model_params=None,
                 edge_cutoff_ligand=None,
                 edge_cutoff_pocket=None,
                 edge_cutoff_interaction=None,
                 predict_angles=False,
                 predict_frames=False,
                 add_cycle_counts=False,
                 add_spectral_feat=False,
                 add_nma_feat=False,
                 reflection_equiv=False,
                 d_max=15.0,
                 num_rbf_dist=16,
                 self_conditioning=False,
                 augment_residue_sc=False,
                 augment_ligand_sc=False,
                 add_chi_as_feature=False,
                 angle_act_fn=False,
                 add_all_atom_diff=False,
                 predict_confidence=False):

        super().__init__(
            predict_angles=predict_angles,
            predict_frames=predict_frames,
            add_cycle_counts=add_cycle_counts,
            add_spectral_feat=add_spectral_feat,
            self_conditioning=self_conditioning,
            augment_residue_sc=augment_residue_sc,
            augment_ligand_sc=augment_ligand_sc
        )

        self.model = model
        self.edge_cutoff_l = edge_cutoff_ligand
        self.edge_cutoff_p = edge_cutoff_pocket
        self.edge_cutoff_i = edge_cutoff_interaction
        self.bond_dict = bond_dict
        self.pocket_bond_dict = pocket_bond_dict
        self.bond_nf = len(bond_dict)
        self.pocket_bond_nf = len(pocket_bond_dict)
        # self.edge_dim = edge_dim
        self.add_nma_feat = add_nma_feat
        self.add_chi_as_feature = add_chi_as_feature
        self.add_all_atom_diff = add_all_atom_diff
        self.condition_time = condition_time
        self.predict_confidence = predict_confidence

        # edge encoding params
        self.reflection_equiv = reflection_equiv
        self.d_max = d_max
        self.num_rbf = num_rbf_dist


        # Output dimensions dimensions, always tuple (scalar, vector)
        _atom_out = (atom_nf[0], 1) if isinstance(atom_nf, Iterable) else (atom_nf, 1)
        _residue_out = (0, 0)

        if self.predict_confidence:
            _atom_out = tuple_sum(_atom_out, (1, 0))

        if self.predict_angles:
            _residue_out = tuple_sum(_residue_out, (5, 0))

        if self.predict_frames:
            _residue_out = tuple_sum(_residue_out, (3, 1))


        # Input dimensions dimensions, always tuple (scalar, vector)
        assert isinstance(atom_nf, int), "expected: element onehot"
        _atom_in = (atom_nf, 0)
        assert isinstance(residue_nf, Iterable), "expected: (AA-onehot, vectors to atoms)"
        _residue_in = tuple(residue_nf)
        _residue_atom_dim = residue_nf[1]

        if self.add_cycle_counts:
            _atom_in = tuple_sum(_atom_in, (3, 0))
        if self.add_spectral_feat:
            _atom_in = tuple_sum(_atom_in, (5, 0))

        if self.add_nma_feat:
            _residue_in = tuple_sum(_residue_in, (0, 5))

        if self.add_chi_as_feature:
            _residue_in = tuple_sum(_residue_in, (5, 0))

        if self.condition_time:
            self.embed_time = num_rbf_time is not None
            self.time_dim = num_rbf_time if self.embed_time else 1

            _atom_in = tuple_sum(_atom_in, (self.time_dim, 0))
            _residue_in = tuple_sum(_residue_in, (self.time_dim, 0))
        else:
            print('Warning: dynamics model is NOT conditioned on time.')

        if self.self_conditioning:
            _atom_in = tuple_sum(_atom_in, _atom_out)
            _residue_in = tuple_sum(_residue_in, _residue_out)

            if self.augment_ligand_sc:
                _atom_in = tuple_sum(_atom_in, (0, 1))

            if self.augment_residue_sc:
                assert self.predict_angles
                _residue_in = tuple_sum(_residue_in, (0, _residue_atom_dim))


        # Edge output dimensions, always tuple (scalar, vector)
        _edge_ligand_out = (self.bond_nf, 0)
        _edge_ligand_before_symmetrization = (model_params.edge_h_dim[0], 0)


        # Edge input dimensions dimensions, always tuple (scalar, vector)
        _edge_ligand_in = (self.bond_nf + self.num_rbf, 1 if self.reflection_equiv else 2)
        _edge_ligand_in = tuple_sum(_edge_ligand_in, _atom_in)  # src node
        _edge_ligand_in = tuple_sum(_edge_ligand_in, _atom_in)  # dst node

        if self_conditioning:
            _edge_ligand_in = tuple_sum(_edge_ligand_in, _edge_ligand_out)

        _n_dist_residue = _residue_atom_dim ** 2 if self.add_all_atom_diff else 1
        _edge_pocket_in = (_n_dist_residue * self.num_rbf + self.pocket_bond_nf, _n_dist_residue)
        _edge_pocket_in = tuple_sum(_edge_pocket_in, _residue_in)  # src node
        _edge_pocket_in = tuple_sum(_edge_pocket_in, _residue_in)  # dst node

        _n_dist_interaction = _residue_atom_dim if self.add_all_atom_diff else 1
        _edge_interaction_in = (_n_dist_interaction * self.num_rbf, _n_dist_interaction)
        _edge_interaction_in = tuple_sum(_edge_interaction_in, _atom_in)  # atom node
        _edge_interaction_in = tuple_sum(_edge_interaction_in, _residue_in)  # residue node


        # Embeddings for newly added edges
        _ligand_nobond_nf = self.bond_nf + _edge_ligand_out[0] if self.self_conditioning else self.bond_nf
        self.ligand_nobond_emb = nn.Parameter(torch.zeros(_ligand_nobond_nf), requires_grad=True)
        self.pocket_nobond_emb = nn.Parameter(torch.zeros(self.pocket_bond_nf), requires_grad=True)

        # for access in self-conditioning
        self.atom_out_dim = _atom_out
        self.residue_out_dim = _residue_out
        self.edge_out_dim = _edge_ligand_out

        if model == 'gvp':

            self.net = GVPModel(
                node_in_dim_ligand=_atom_in,
                node_in_dim_pocket=_residue_in,
                edge_in_dim_ligand=_edge_ligand_in,
                edge_in_dim_pocket=_edge_pocket_in,
                edge_in_dim_interaction=_edge_interaction_in,
                node_h_dim_ligand=model_params.node_h_dim,
                node_h_dim_pocket=model_params.node_h_dim,
                edge_h_dim_ligand=model_params.edge_h_dim,
                edge_h_dim_pocket=model_params.edge_h_dim,
                edge_h_dim_interaction=model_params.edge_h_dim,
                node_out_dim_ligand=_atom_out,
                node_out_dim_pocket=_residue_out,
                edge_out_dim_ligand=_edge_ligand_before_symmetrization,
                edge_out_dim_pocket=None,
                edge_out_dim_interaction=None,
                num_layers=model_params.n_layers,
                drop_rate=model_params.dropout,
                vector_gate=model_params.vector_gate,
                update_edge_attr=True
            )

        else:
            raise NotImplementedError(f"{model} is not available")

        assert _edge_ligand_out[1] == 0
        assert _edge_ligand_before_symmetrization[1] == 0
        self.edge_decoder = nn.Sequential(
            nn.Linear(_edge_ligand_before_symmetrization[0], _edge_ligand_before_symmetrization[0]),
            torch.nn.SiLU(),
            nn.Linear(_edge_ligand_before_symmetrization[0], _edge_ligand_out[0])
        )

        if angle_act_fn is None:
            self.angle_act_fn = None
        elif angle_act_fn == 'tanh':
            self.angle_act_fn = lambda x: np.pi * F.tanh(x)
        else:
            raise NotImplementedError(f"Angle activation {angle_act_fn} not available")

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

        # NOTE: 'bond' denotes one-directional edges and 'edge' means bi-directional
        # get graph edges and edge attributes
        if bonds_ligand is not None:

            ligand_bond_indices = bonds_ligand[0]

            # make sure messages are passed both ways
            ligand_edge_indices = torch.cat(
                [bonds_ligand[0], bonds_ligand[0].flip(dims=[0])], dim=1)
            ligand_edge_types = torch.cat([bonds_ligand[1], bonds_ligand[1]], dim=0)
            if e_atoms_sc is not None:
                e_atoms_sc = torch.cat([e_atoms_sc, e_atoms_sc], dim=0)

            # add auxiliary features to ligand nodes
            extra_features = self.compute_extra_features(
                mask_atoms, ligand_edge_indices, ligand_edge_types.argmax(-1))
            h_atoms = torch.cat([h_atoms, extra_features], dim=-1)

        if bonds_pocket is not None:
            # make sure messages are passed both ways
            pocket_edge_indices = torch.cat(
                [bonds_pocket[0], bonds_pocket[0].flip(dims=[0])], dim=1)
            pocket_edge_types = torch.cat([bonds_pocket[1], bonds_pocket[1]], dim=0)


        # Self-conditioning
        if h_atoms_sc is not None:
            h_atoms = (torch.cat([h_atoms, h_atoms_sc[0]], dim=-1), h_atoms_sc[1])

        if e_atoms_sc is not None:
            ligand_edge_types = torch.cat([ligand_edge_types, e_atoms_sc], dim=-1)

        if h_residues_sc is not None:
            # if self.augment_residue_sc:
            if isinstance(h_residues_sc, tuple):
                h_residues = (torch.cat([h_residues[0], h_residues_sc[0]], dim=-1),
                              torch.cat([h_residues[1], h_residues_sc[1]], dim=1))
            else:
                h_residues = (torch.cat([h_residues[0], h_residues_sc], dim=-1),
                              h_residues[1])

        if self.condition_time:
            if self.embed_time:
                t = _rbf(t.squeeze(-1), D_min=0.0, D_max=1.0, D_count=self.time_dim, device=t.device)
            if isinstance(h_atoms, tuple) :
                h_atoms = (torch.cat([h_atoms[0], t[mask_atoms]], dim=1), h_atoms[1]) 
            else: 
                h_atoms = torch.cat([h_atoms, t[mask_atoms]], dim=1)
            h_residues = (torch.cat([h_residues[0], t[mask_residues]], dim=1), h_residues[1])

        empty_pocket = (len(pocket['x']) == 0)

        # Process edges and encode in shared feature space
        edge_index_dict, edge_attr_dict = self.get_edges(
            x_atoms, h_atoms, mask_atoms, ligand_edge_indices, ligand_edge_types,
            x_residues, h_residues, mask_residues, pocket['v'], pocket_edge_indices, pocket_edge_types, 
            empty_pocket=empty_pocket
        )

        if not empty_pocket:
            node_attr_dict = {
                'ligand': h_atoms,
                'pocket': h_residues,
            }
            batch_mask_dict = {
                'ligand': mask_atoms,
                'pocket': mask_residues,
            }
        else:
            node_attr_dict = {'ligand': h_atoms}
            batch_mask_dict = {'ligand': mask_atoms}

        if self.model == 'gvp' or self.model == 'gvp_transformer':
            out_node_attr, out_edge_attr = self.net(
                node_attr_dict, batch_mask_dict, edge_index_dict, edge_attr_dict)

        else:
            raise NotImplementedError(f"Wrong model ({self.model})")

        h_final_atoms = out_node_attr['ligand'][0]
        vel = out_node_attr['ligand'][1].squeeze(-2)

        if torch.any(torch.isnan(vel)) or torch.any(torch.isnan(h_final_atoms)):
            if self.training:
                vel[torch.isnan(vel)] = 0.0
                h_final_atoms[torch.isnan(h_final_atoms)] = 0.0
            else:
                raise ValueError("NaN detected in network output")

        # predict edge type
        edge_final = out_edge_attr[('ligand', '', 'ligand')]
        edges = edge_index_dict[('ligand', '', 'ligand')]

        # Symmetrize
        edge_logits = torch.zeros(
            (len(mask_atoms), len(mask_atoms), edge_final.size(-1)),
            device=mask_atoms.device)
        edge_logits[edges[0], edges[1]] = edge_final
        edge_logits = (edge_logits + edge_logits.transpose(0, 1)) * 0.5

        # return upper triangular elements only (matching the input)
        edge_logits = edge_logits[ligand_bond_indices[0], ligand_bond_indices[1]]
        # assert (edge_logits == 0).sum() == 0

        edge_final_atoms = self.edge_decoder(edge_logits)

        pred_ligand = {'vel': vel, 'logits_e': edge_final_atoms}

        if self.predict_confidence:
            pred_ligand['logits_h'] = h_final_atoms[:, :-1]
            pred_ligand['uncertainty_vel'] = F.softplus(h_final_atoms[:, -1])
        else:
            pred_ligand['logits_h'] = h_final_atoms

        pred_residues = {}

        # Predict torsion angles
        if self.predict_angles and self.predict_frames:
            residue_s, residue_v = out_node_attr['pocket']
            pred_residues['chi'] = residue_s[:, :5]
            pred_residues['rot'] = residue_s[:, 5:]
            pred_residues['trans'] = residue_v.squeeze(1)

        elif self.predict_frames:
            pred_residues['rot'], pred_residues['trans'] = out_node_attr['pocket']
            pred_residues['trans'] = pred_residues['trans'].squeeze(1)

        elif self.predict_angles:
            pred_residues['chi'] = out_node_attr['pocket']

        if self.angle_act_fn is not None and 'chi' in pred_residues:
            pred_residues['chi'] = self.angle_act_fn(pred_residues['chi'])

        return pred_ligand, pred_residues

    def get_edges(self, x_ligand, h_ligand, batch_mask_ligand, edges_ligand, edge_feat_ligand,
                  x_pocket, h_pocket, batch_mask_pocket, atom_vectors_pocket, edges_pocket, edge_feat_pocket,
                  self_edges=False, empty_pocket=False):
        
        # Adjacency matrix
        adj_ligand = batch_mask_ligand[:, None] == batch_mask_ligand[None, :]
        adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
        adj_cross = batch_mask_ligand[:, None] == batch_mask_pocket[None, :]

        if self.edge_cutoff_l is not None:
            adj_ligand = adj_ligand & (torch.cdist(x_ligand, x_ligand) <= self.edge_cutoff_l)

            # Add missing bonds if they got removed
            adj_ligand[edges_ligand[0], edges_ligand[1]] = True

            if not self_edges:
                adj_ligand = adj_ligand ^ torch.eye(*adj_ligand.size(), out=torch.empty_like(adj_ligand))

        if self.edge_cutoff_p is not None and not empty_pocket:
            adj_pocket = adj_pocket & (torch.cdist(x_pocket, x_pocket) <= self.edge_cutoff_p)

            # Add missing bonds if they got removed
            adj_pocket[edges_pocket[0], edges_pocket[1]] = True

            if not self_edges:
                adj_pocket = adj_pocket ^ torch.eye(*adj_pocket.size(), out=torch.empty_like(adj_pocket))

        if self.edge_cutoff_i is not None and not empty_pocket:
            adj_cross = adj_cross & (torch.cdist(x_ligand, x_pocket) <= self.edge_cutoff_i)

        # ligand-ligand edge features
        edges_ligand_updated = torch.stack(torch.where(adj_ligand), dim=0)
        feat_ligand = self.ligand_nobond_emb.repeat(*adj_ligand.shape, 1)
        feat_ligand[edges_ligand[0], edges_ligand[1]] = edge_feat_ligand
        feat_ligand = feat_ligand[edges_ligand_updated[0], edges_ligand_updated[1]]
        feat_ligand = self.ligand_edge_features(h_ligand, x_ligand, edges_ligand_updated, batch_mask_ligand, edge_attr=feat_ligand)

        if not empty_pocket:
            # residue-residue edge features
            edges_pocket_updated = torch.stack(torch.where(adj_pocket), dim=0)
            feat_pocket = self.pocket_nobond_emb.repeat(*adj_pocket.shape, 1)
            feat_pocket[edges_pocket[0], edges_pocket[1]] = edge_feat_pocket
            feat_pocket = feat_pocket[edges_pocket_updated[0], edges_pocket_updated[1]]
            feat_pocket = self.pocket_edge_features(h_pocket, x_pocket, atom_vectors_pocket, edges_pocket_updated, edge_attr=feat_pocket)

            # ligand-residue edge features
            edges_cross = torch.stack(torch.where(adj_cross), dim=0)
            feat_cross = self.cross_edge_features(h_ligand, x_ligand, h_pocket, x_pocket, atom_vectors_pocket, edges_cross)

            edge_index = {
                ('ligand', '', 'ligand'): edges_ligand_updated,
                ('pocket', '', 'pocket'): edges_pocket_updated,
                ('ligand', '', 'pocket'): edges_cross,
                ('pocket', '', 'ligand'): edges_cross.flip(dims=[0]),
            }

            edge_attr = {
                ('ligand', '', 'ligand'): feat_ligand,
                ('pocket', '', 'pocket'): feat_pocket,
                ('ligand', '', 'pocket'): feat_cross,
                ('pocket', '', 'ligand'): feat_cross,
            }
        else:
            edge_index = {('ligand', '', 'ligand'): edges_ligand_updated}
            edge_attr = {('ligand', '', 'ligand'): feat_ligand}

        return edge_index, edge_attr

    def ligand_edge_features(self, h, x, edge_index, batch_mask=None, edge_attr=None):
        """
        :param h: (s, V)
        :param x:
        :param edge_index:
        :param batch_mask:
        :param edge_attr:
        :return: scalar and vector-valued edge features
        """
        row, col = edge_index
        coord_diff = x[row] - x[col]
        dist = coord_diff.norm(dim=-1)
        rbf = _rbf(dist, D_max=self.d_max, D_count=self.num_rbf,
                   device=x.device)

        if isinstance(h, tuple):
            edge_s = torch.cat([h[0][row], h[0][col], rbf], dim=1)
            edge_v = torch.cat([h[1][row], h[1][col], _normalize(coord_diff).unsqueeze(-2)], dim=1)
        else:
            edge_s = torch.cat([h[row], h[col], rbf], dim=1)
            edge_v = _normalize(coord_diff).unsqueeze(-2)

        # edge_s = rbf
        # edge_v = _normalize(coord_diff).unsqueeze(-2)

        if edge_attr is not None:
            edge_s = torch.cat([edge_s, edge_attr], dim=1)

        # self.reflection_equiv: bool, use reflection-sensitive feature based on
        #                        the cross product if False
        if not self.reflection_equiv:
            mean = scatter_mean(x, batch_mask, dim=0,
                                dim_size=batch_mask.max() + 1)
            row, col = edge_index
            cross = torch.cross(x[row] - mean[batch_mask[row]],
                                x[col] - mean[batch_mask[col]], dim=1)
            cross = _normalize(cross).unsqueeze(-2)

            edge_v = torch.cat([edge_v, cross], dim=-2)

        return torch.nan_to_num(edge_s), torch.nan_to_num(edge_v)

    def pocket_edge_features(self, h, x, v, edge_index, edge_attr=None):
        """
        :param h: (s, V)
        :param x:
        :param v:
        :param edge_index:
        :param edge_attr:
        :return: scalar and vector-valued edge features
        """
        row, col = edge_index

        if self.add_all_atom_diff:
            all_coord = v + x.unsqueeze(1)  # (nR, nA, 3)
            coord_diff = all_coord[row, :, None, :] - all_coord[col, None, :, :]  # (nB, nA, nA, 3)
            coord_diff = coord_diff.flatten(1, 2)
            dist = coord_diff.norm(dim=-1)  # (nB, nA^2)
            rbf = _rbf(dist, D_max=self.d_max, D_count=self.num_rbf, device=x.device)  # (nB, nA^2, rdb_dim)
            rbf = rbf.flatten(1, 2)
            coord_diff = _normalize(coord_diff)
        else:
            coord_diff = x[row] - x[col]
            dist = coord_diff.norm(dim=-1)
            rbf = _rbf(dist, D_max=self.d_max, D_count=self.num_rbf, device=x.device)
            coord_diff = _normalize(coord_diff).unsqueeze(-2)

        edge_s = torch.cat([h[0][row], h[0][col], rbf], dim=1)
        edge_v = torch.cat([h[1][row], h[1][col], coord_diff], dim=1)
        # edge_s = rbf
        # edge_v = coord_diff

        if edge_attr is not None:
            edge_s = torch.cat([edge_s, edge_attr], dim=1)

        return torch.nan_to_num(edge_s), torch.nan_to_num(edge_v)

    def cross_edge_features(self, h_ligand, x_ligand, h_pocket, x_pocket, v_pocket, edge_index):
        """
        :param h_ligand: (s, V)
        :param x_ligand:
        :param h_pocket: (s, V)
        :param x_pocket:
        :param v_pocket:
        :param edge_index: first row indexes into the ligand tensors, second row into the pocket tensors

        :return: scalar and vector-valued edge features
        """
        ligand_idx, pocket_idx = edge_index

        if self.add_all_atom_diff:
            all_coord_pocket = v_pocket + x_pocket.unsqueeze(1)  # (nR, nA, 3)
            coord_diff = x_ligand[ligand_idx, None, :] - all_coord_pocket[pocket_idx]  # (nB, nA, 3)
            dist = coord_diff.norm(dim=-1)  # (nB, nA)
            rbf = _rbf(dist, D_max=self.d_max, D_count=self.num_rbf, device=x_ligand.device)  # (nB, nA, rdb_dim)
            rbf = rbf.flatten(1, 2)
            coord_diff = _normalize(coord_diff)
        else:
            coord_diff = x_ligand[ligand_idx] - x_pocket[pocket_idx]
            dist = coord_diff.norm(dim=-1)  # (nB, nA)
            rbf = _rbf(dist, D_max=self.d_max, D_count=self.num_rbf, device=x_ligand.device)
            coord_diff = _normalize(coord_diff).unsqueeze(-2)

        if isinstance(h_ligand, tuple):
            edge_s = torch.cat([h_ligand[0][ligand_idx], h_pocket[0][pocket_idx], rbf], dim=1)
            edge_v = torch.cat([h_ligand[1][ligand_idx], h_pocket[1][pocket_idx], coord_diff], dim=1)
        else:
            edge_s = torch.cat([h_ligand[ligand_idx], h_pocket[0][pocket_idx], rbf], dim=1)
            edge_v = torch.cat([h_pocket[1][pocket_idx], coord_diff], dim=1)

        # edge_s = rbf
        # edge_v = coord_diff

        return torch.nan_to_num(edge_s), torch.nan_to_num(edge_v)
