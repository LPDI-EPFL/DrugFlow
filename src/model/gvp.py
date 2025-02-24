"""
Geometric Vector Perceptron implementation taken from:
https://github.com/drorlab/gvp-pytorch/blob/main/gvp/__init__.py
"""
import copy
import warnings

import torch, functools
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_mean


def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]


def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.

    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)

    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
           torch.randn(n, dims[1], 3, device=device)


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple
    representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    v = torch.reshape(x[..., -3 * nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3 * nv]
    return s, v


def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)


class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''

    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''

    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''

    def __init__(self, dims, learnable_vector_weight=False):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        self.vector_norm = VectorLayerNorm(self.v, learnable_vector_weight) \
            if self.v > 0 else None

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        # vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        # vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        # return self.scalar_norm(s), v / vn
        return self.scalar_norm(s), self.vector_norm(v)


class VectorLayerNorm(nn.Module):
    """
    Equivariant normalization of vector-valued features inspired by:
    Liao, Yi-Lun, and Tess Smidt.
    "Equiformer: Equivariant graph attention transformer for 3d atomistic graphs."
    arXiv preprint arXiv:2206.11990 (2022).
    Section 4.1, "Layer Normalization"
    """
    def __init__(self, n_channels, learnable_weight=True):
        super(VectorLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, n_channels, 1)) \
            if learnable_weight else None                            # (1, c, 1)

    def forward(self, x):
        """
        Computes LN(x) = ( x / RMS( L2-norm(x) ) ) * gamma
        :param x: input tensor (n, c, 3)
        :return: layer normalized vector feature
        """
        norm2 = _norm_no_nan(x, axis=-1, keepdims=True, sqrt=False)  # (n, c, 1)
        rms = torch.sqrt(torch.mean(norm2, dim=-2, keepdim=True))    # (n, 1, 1)
        x = x / rms                                                  # (n, c, 3)
        if self.gamma is not None:
            x = x * self.gamma
        return x


class GVPConv(MessagePassing):
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

    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean",
                 activations=(F.relu, torch.sigmoid), vector_gate=False,
                 update_edge_attr=False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        self.update_edge_attr = update_edge_attr

        GVP_ = functools.partial(GVP,
                                 activations=activations,
                                 vector_gate=vector_gate)

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve),
                         (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve),
                         out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                        activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

        self.edge_func = copy.deepcopy(self.message_func) \
            if self.update_edge_attr else None

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index,
                                 s=x_s,
                                 v=x_v.reshape(x_v.shape[0], 3 * x_v.shape[1]),
                                 edge_attr=edge_attr)

        if self.update_edge_attr:
            s_i, s_j = x_s[edge_index[0]], x_s[edge_index[1]]
            x_v = x_v.reshape(x_v.shape[0], 3 * x_v.shape[1])
            v_i, v_j = x_v[edge_index[0]], x_v[edge_index[1]]

            edge_out = self.edge_attr(s_i, v_i, s_j, v_j, edge_attr)
            return _split(message, self.vo), edge_out
        else:
            return _split(message, self.vo)

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


class GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    :param update_edge_attr: whether to compute an updated edge representation
    :param ln_vector_weight: whether to include a learnable weight in the vector
                             layer norm
    '''

    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False,
                 activations=(F.relu, torch.sigmoid), vector_gate=False,
                 update_edge_attr=False, ln_vector_weight=False):

        super(GVPConvLayer, self).__init__()
        assert not (update_edge_attr and autoregressive), "Not implemented"
        self.update_edge_attr = update_edge_attr
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                            aggr="add" if autoregressive else "mean",
                            activations=activations, vector_gate=vector_gate,
                            update_edge_attr=update_edge_attr)
        GVP_ = functools.partial(GVP,
                                 activations=activations,
                                 vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims, ln_vector_weight)
                                   for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        def get_feedforward(n_dims):
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

        self.ff_func = get_feedforward(node_dims)

        if self.update_edge_attr:
            self.edge_norm = nn.ModuleList([LayerNorm(edge_dims, ln_vector_weight)
                                            for _ in range(2)])
            self.edge_dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])
            self.edge_ff = get_feedforward(edge_dims)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`.
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node
                embeddings `x` will still be the base of the update and the
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''

        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward,
                          edge_attr_backward)
            )

            count = scatter_add(torch.ones_like(dst), dst,
                                dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(
                -1)

            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)

        if self.update_edge_attr:
            dh, de = dh
            edge_attr = self.edge_norm[0](tuple_sum(edge_attr, self.dropout[0](de)))
            de = self.edge_ff(edge_attr)
            edge_attr = self.edge_norm[1](tuple_sum(edge_attr, self.dropout[1](de)))

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))

        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return (x, edge_attr) if self.update_edge_attr else x


################################################################################
def _normalize(tensor, dim=-1, eps=1e-8):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True) + eps))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


class GVPModel(torch.nn.Module):
    """
    GVP-GNN model
    inspired by: https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    and: https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/atom3d.py#L115

    :param node_in_dim: node dimension in input graph, scalars or tuple (scalars, vectors)
    :param node_h_dim: node dimensions to use in GVP-GNN layers, tuple (s, V)
    :param node_out_nf: node dimensions in output graph, tuple (s, V)
    :param edge_in_nf: edge dimension in input graph (scalars)
    :param edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers,
        tuple (s, V)
    :param edge_out_nf: edge dimensions in output graph, tuple (s, V)
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    :param vector_gate: use vector gates in all GVPs
    :param reflection_equiv: bool, use reflection-sensitive feature based on the
        cross product if False
    :param d_max:
    :param num_rbf:
    :param update_edge_attr: bool, update edge attributes at each layer in a
        learnable way
    """
    def __init__(self, node_in_dim, node_h_dim, node_out_nf,
                 edge_in_nf, edge_h_dim, edge_out_nf,
                 num_layers=3, drop_rate=0.1, vector_gate=False,
                 reflection_equiv=True, d_max=20.0, num_rbf=16,
                 update_edge_attr=False):

        super(GVPModel, self).__init__()

        self.reflection_equiv = reflection_equiv
        self.update_edge_attr = update_edge_attr
        self.d_max = d_max
        self.num_rbf = num_rbf

        # node_in_dim = (node_in_dim, 1)
        if not isinstance(node_in_dim, tuple):
            node_in_dim = (node_in_dim, 0)

        edge_in_dim = (edge_in_nf + 2 * node_in_dim[0] + self.num_rbf, 1)
        if not self.reflection_equiv:
            edge_in_dim = (edge_in_dim[0], edge_in_dim[1] + 1)

        # self.W_v = nn.Sequential(
        #     GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=True),
        #     LayerNorm(node_h_dim)
        # )
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim, learnable_vector_weight=True),
            GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=vector_gate),
        )
        # self.W_e = nn.Sequential(
        #     GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=True),
        #     LayerNorm(edge_h_dim)
        # )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim, learnable_vector_weight=True),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=vector_gate),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate,
                         update_edge_attr=self.update_edge_attr,
                         activations=(F.relu, None), vector_gate=vector_gate,
                         ln_vector_weight=True)
                         # activations=(F.relu, torch.sigmoid))
            # GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate,
            #              update_edge_attr=self.update_edge_attr,
            #              activations=(nn.SiLU(), nn.SiLU()))
            for _ in range(num_layers))

        # self.W_v_out = GVP(node_h_dim, (node_out_nf, 1),
        #                    activations=(None, None), vector_gate=True)
        self.W_v_out = nn.Sequential(
            LayerNorm(node_h_dim, learnable_vector_weight=True),
            GVP(node_h_dim, (node_out_nf, 1), activations=(None, None), vector_gate=vector_gate),
        )
        # self.W_e_out = GVP(edge_h_dim, (edge_out_nf, 0),
        #                    activations=(None, None), vector_gate=True) \
        #     if self.update_edge_attr else None
        self.W_e_out = nn.Sequential(
            LayerNorm(edge_h_dim, learnable_vector_weight=True),
            GVP(edge_h_dim, (edge_out_nf, 0), activations=(None, None), vector_gate=vector_gate)
        ) if self.update_edge_attr else None

    def edge_features(self, h, x, edge_index, batch_mask=None, edge_attr=None):
        """
        :param h:
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

        edge_s = torch.cat([h[row], h[col], rbf], dim=1)
        edge_v = _normalize(coord_diff).unsqueeze(-2)

        if edge_attr is not None:
            edge_s = torch.cat([edge_s, edge_attr], dim=1)

        if not self.reflection_equiv:
            mean = scatter_mean(x, batch_mask, dim=0,
                                dim_size=batch_mask.max() + 1)
            row, col = edge_index
            cross = torch.cross(x[row] - mean[batch_mask[row]],
                                x[col] - mean[batch_mask[col]], dim=1)
            cross = _normalize(cross).unsqueeze(-2)

            edge_v = torch.cat([edge_v, cross], dim=-2)

        return torch.nan_to_num(edge_s), torch.nan_to_num(edge_v)

    def forward(self, h, x, edge_index, v=None, batch_mask=None, edge_attr=None):

        # h_v = (h, x.unsqueeze(-2))
        h_v = h if v is None else (h, v)
        h_e = self.edge_features(h, x, edge_index, batch_mask, edge_attr)

        h_v = self.W_v(h_v)
        h_e = self.W_e(h_e)

        for layer in self.layers:
            h_v = layer(h_v, edge_index, edge_attr=h_e)
            if self.update_edge_attr:
                h_v, h_e = h_v

        # h, x = self.W_v_out(h_v)
        # x = x.squeeze(-2)
        h, vel = self.W_v_out(h_v)
        # x = x + vel.squeeze(-2)

        if self.update_edge_attr:
            edge_attr = self.W_e_out(h_e)

        # return h, x, edge_attr
        return h, vel.squeeze(-2), edge_attr
