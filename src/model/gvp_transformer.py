import math
import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_std, scatter_min, scatter_max, scatter_softmax


# ## debug
# import sys
# from pathlib import Path
#
# basedir = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(basedir))
# ###

from src.model.gvp import GVP, _norm_no_nan, tuple_sum, Dropout, LayerNorm, \
    tuple_cat, tuple_index, _rbf, _normalize


def tuple_mul(tup, val):
    if isinstance(val, torch.Tensor):
        return (tup[0] * val, tup[1] * val.unsqueeze(-1))
    return (tup[0] * val, tup[1] * val)


class GVPBlock(nn.Module):
    def __init__(self, in_dims, out_dims, n_layers=1,
                 activations=(F.relu, torch.sigmoid), vector_gate=False,
                 dropout=0.0, skip=False, layernorm=False):
        super(GVPBlock, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        assert not skip or (self.si == self.so and self.vi == self.vo)
        self.skip = skip

        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        module_list = []
        if n_layers == 1:
            module_list.append(GVP_(in_dims, out_dims, activations=(None, None)))
        else:
            module_list.append(GVP_(in_dims, out_dims))
            for i in range(n_layers - 2):
                module_list.append(GVP_(out_dims, out_dims))
            module_list.append(GVP_(out_dims, out_dims, activations=(None, None)))

        self.layers = nn.Sequential(*module_list)

        self.norm = LayerNorm(out_dims, learnable_vector_weight=True) if layernorm else None
        self.dropout = Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`
        """

        dx = self.layers(x)

        if self.dropout is not None:
            dx = self.dropout(dx)

        if self.skip:
            x = tuple_sum(x, dx)
        else:
            x = dx

        if self.norm is not None:
            x = self.norm(x)

        return x


class GeometricPNA(nn.Module):
    def __init__(self, d_in, d_out):
        """ Map features to global features """
        super().__init__()
        si, vi = d_in
        so, vo = d_out
        self.gvp = GVPBlock((4 * si + 3 * vi, vi), d_out)

    def forward(self, x, batch_mask, batch_size=None):
        """ x: tuple (s, V) """
        s, v = x

        sm = scatter_mean(s, batch_mask, dim=0, dim_size=batch_size)
        smi = scatter_min(s, batch_mask, dim=0, dim_size=batch_size)[0]
        sma = scatter_max(s, batch_mask, dim=0, dim_size=batch_size)[0]
        sstd = scatter_std(s, batch_mask, dim=0, dim_size=batch_size)

        vnorm = _norm_no_nan(v)
        vm = scatter_mean(v, batch_mask, dim=0, dim_size=batch_size)
        vmi = scatter_min(vnorm, batch_mask, dim=0, dim_size=batch_size)[0]
        vma = scatter_max(vnorm, batch_mask, dim=0, dim_size=batch_size)[0]
        vstd = scatter_std(vnorm, batch_mask, dim=0, dim_size=batch_size)

        z = torch.hstack((sm, smi, sma, sstd, vmi, vma, vstd))
        out = self.gvp((z, vm))
        return out


class TupleLinear(nn.Module):
    def __init__(self, in_dims, out_dims, bias=True):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        assert self.si and self.so
        self.ws = nn.Linear(self.si, self.so, bias=bias)
        self.wv = nn.Linear(self.vi, self.vo, bias=bias) if self.vi and self.vo else None

    def forward(self, x):
        if self.vi:
            s, v = x

            s = self.ws(s)

            if self.vo:
                v = v.transpose(-1, -2)
                v = self.wv(v)
                v = v.transpose(-1, -2)

        else:
            s = self.ws(x)

            if self.vo:
                v = torch.zeros(s.size(0), self.vo, 3, device=s.device)

        return (s, v) if self.vo else s


class GVPTransformerLayer(nn.Module):
    """
    Full graph transformer layer with Geometric Vector Perceptrons.
    Inspired by
    - GVP: Jing, Bowen, et al. "Learning from protein structure with geometric vector perceptrons." arXiv preprint arXiv:2009.01411 (2020).
    - Transformer architecture: Vignac, Clement, et al. "Digress: Discrete denoising diffusion for graph generation." arXiv preprint arXiv:2209.14734 (2022).
    - Invariant point attention: Jumper, John, et al. "Highly accurate protein structure prediction with AlphaFold." Nature 596.7873 (2021): 583-589.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param global_dims: global feature dimension (n_scalar, n_vector)
    :param dk: key dimension, (n_scalar, n_vector)
    :param dv: node value dimension, (n_scalar, n_vector)
    :param de: edge value dimension, (n_scalar, n_vector)
    :param db: dimension of edge contribution to attention, int
    :param attn_heads: number of attention heads, int
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    :param attention: can be used to turn off the attention mechanism
    """

    def __init__(self, node_dims, edge_dims, global_dims, dk, dv, de, db,
                 attn_heads, n_feedforward=1, drop_rate=0.0,
                 activations=(F.relu, torch.sigmoid), vector_gate=False,
                 attention=True):

        super(GVPTransformerLayer, self).__init__()

        self.attention = attention

        dq = dk
        self.dq = dq
        self.dk = dk
        self.dv = dv
        self.de = de
        self.db = db

        self.h = attn_heads

        self.q = TupleLinear(node_dims, tuple_mul(dq, self.h), bias=False) if self.attention else None
        self.k = TupleLinear(node_dims, tuple_mul(dk, self.h), bias=False) if self.attention else None
        self.vx = TupleLinear(node_dims, tuple_mul(dv, self.h), bias=False)

        self.ve = TupleLinear(edge_dims, tuple_mul(de, self.h), bias=False)
        self.b = TupleLinear(edge_dims, (db * self.h, 0), bias=False) if self.attention else None

        m_dim = tuple_sum(tuple_mul(dv, self.h), tuple_mul(de, self.h))
        self.msg = GVPBlock(m_dim, m_dim, n_feedforward,
                            activations=activations, vector_gate=vector_gate)

        m_dim = tuple_sum(m_dim, global_dims)
        self.x_out = GVPBlock(m_dim, node_dims, n_feedforward,
                              activations=activations, vector_gate=vector_gate)
        self.x_norm = LayerNorm(node_dims, learnable_vector_weight=True)
        self.x_dropout = Dropout(drop_rate)

        e_dim = tuple_sum(tuple_mul(node_dims, 2), edge_dims, global_dims)
        if self.attention:
            e_dim = (e_dim[0] + 3 * attn_heads, e_dim[1])
        self.e_out = GVPBlock(e_dim, edge_dims, n_feedforward,
                              activations=activations, vector_gate=vector_gate)
        self.e_norm = LayerNorm(edge_dims, learnable_vector_weight=True)
        self.e_dropout = Dropout(drop_rate)

        self.pna_x = GeometricPNA(node_dims, node_dims)
        self.pna_e = GeometricPNA(edge_dims, edge_dims)
        self.y = GVP(global_dims, global_dims, activations=(None, None), vector_gate=vector_gate)
        _dim = tuple_sum(node_dims, edge_dims, global_dims)
        self.y_out = GVPBlock(_dim, global_dims, n_feedforward,
                              activations=activations, vector_gate=vector_gate)
        self.y_norm = LayerNorm(global_dims, learnable_vector_weight=True)
        self.y_dropout = Dropout(drop_rate)

    def forward(self, x, edge_index, batch_mask, edge_attr, global_attr=None,
                node_mask=None):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param batch_mask: array indicating different graphs
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param global_attr: tuple (s, V) of `torch.Tensor`
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        """

        row, col = edge_index
        n = len(x[0])
        batch_size = len(torch.unique(batch_mask))

        # Compute attention
        if self.attention:
            Q = self.q(x)
            K = self.k(x)
            b = self.b(edge_attr)

            qs, qv = Q  # (n, dq * h), (n, dq * h, 3)
            ks, kv = K  # (n, dq * h), (n, dq * h, 3)
            attn_s = (qs[row] * ks[col]).reshape(len(row), self.h, self.dq[0]).sum(dim=-1)  # (m, h)
            # NOTE: attn_v is the Frobenius inner product between vector-valued queries and keys of size [dq, 3]
            #  (generalizes the dot-product between queries and keys similar to Pocket2Mol)
            # TODO: double-check if this is correctly implemented!
            attn_v = (qv[row] * kv[col]).reshape(len(row), self.h, self.dq[1], 3).sum(dim=(-2, -1))  # (m, h)
            attn_e = b.reshape(b.size(0), self.h, self.db).sum(dim=-1)  # (m, h)

            attn = attn_s / math.sqrt(3 * self.dk[0]) + \
                   attn_v / math.sqrt(9 * self.dk[1]) + \
                   attn_e / math.sqrt(3 * self.db)
            attn = scatter_softmax(attn, row, dim=0)  # (m, h)
            attn = attn.unsqueeze(-1)  # (m, h, 1)

        # Compute new features
        Vx = self.vx(x)
        Ve = self.ve(edge_attr)

        mx = (Vx[0].reshape(Vx[0].size(0), self.h, self.dv[0]),  # (n, h, dv)
              Vx[1].reshape(Vx[1].size(0), self.h, self.dv[1], 3))  # (n, h, dv, 3)
        me = (Ve[0].reshape(Ve[0].size(0), self.h, self.de[0]),
              Ve[1].reshape(Ve[1].size(0), self.h, self.de[1], 3))

        mx = tuple_index(mx, col)
        if self.attention:
            mx = tuple_mul(mx, attn)
            me = tuple_mul(me, attn)

        _m = tuple_cat(mx, me)
        _m = (_m[0].flatten(1), _m[1].flatten(1, 2))
        m = self.msg(_m)  # (m, h * dv), (m, h * dv, 3)
        m = (scatter_mean(m[0], row, dim=0, dim_size=n),  # (n, h * dv)
             scatter_mean(m[1], row, dim=0, dim_size=n))  # (n, h * dv, 3)
        if global_attr is not None:
            m = tuple_cat(m, tuple_index(global_attr, batch_mask))
        X_out = self.x_norm(tuple_sum(x, self.x_dropout(self.x_out(m))))

        _e = tuple_cat(tuple_index(x, row), tuple_index(x, col), edge_attr)
        if self.attention:
            _e = (torch.cat([_e[0], attn_s, attn_v, attn_e], dim=-1), _e[1])
        if global_attr is not None:
            _e = tuple_cat(_e, tuple_index(global_attr, batch_mask[row]))
        E_out = self.e_norm(tuple_sum(edge_attr, self.e_dropout(self.e_out(_e))))

        _y = tuple_cat(self.pna_x(x, batch_mask, batch_size),
                       self.pna_e(edge_attr, batch_mask[row], batch_size))
        if global_attr is not None:
            _y = tuple_cat(_y, self.y(global_attr))
            y_out = self.y_norm(tuple_sum(global_attr, self.y_dropout(self.y_out(_y))))
        else:
            y_out = self.y_norm(self.y_dropout(self.y_out(_y)))

        if node_mask is not None:
            X_out[0][~node_mask], X_out[1][~node_mask] = tuple_index(x, ~node_mask)

        return X_out, E_out, y_out


class GVPTransformerModel(torch.nn.Module):
    """
    GVP-Transformer model

    :param node_in_dim: node dimension in input graph, scalars or tuple (scalars, vectors)
    :param node_h_dim: node dimensions to use in GVP-GNN layers, tuple (s, V)
    :param node_out_nf: node dimensions in output graph, tuple (s, V)
    :param edge_in_nf: edge dimension in input graph (scalars)
    :param edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers,
        tuple (s, V)
    :param edge_out_nf: edge dimensions in output graph, tuple (s, V)
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    :param reflection_equiv: bool, use reflection-sensitive feature based on the
        cross product if False
    :param d_max:
    :param num_rbf:
    :param vector_gate: use vector gates in all GVPs
    :param attention: can be used to turn off the attention mechanism
    """
    def __init__(self, node_in_dim, node_h_dim, node_out_nf, edge_in_nf,
                 edge_h_dim, edge_out_nf, num_layers, dk, dv, de, db, dy,
                 attn_heads, n_feedforward, drop_rate, reflection_equiv=True,
                 d_max=20.0, num_rbf=16, vector_gate=False, attention=True):

        super(GVPTransformerModel, self).__init__()

        self.reflection_equiv = reflection_equiv
        self.d_max = d_max
        self.num_rbf = num_rbf

        # node_in_dim = (node_in_dim, 1)
        if not isinstance(node_in_dim, tuple):
            node_in_dim = (node_in_dim, 0)

        edge_in_dim = (edge_in_nf + 2 * node_in_dim[0] + self.num_rbf, 1)
        if not self.reflection_equiv:
            edge_in_dim = (edge_in_dim[0], edge_in_dim[1] + 1)

        self.W_v = GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=vector_gate)
        self.W_e = GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=vector_gate)
        # self.W_v = nn.Sequential(
        #     LayerNorm(node_in_dim, learnable_vector_weight=True),
        #     GVP(node_in_dim, node_h_dim, activations=(None, None)),
        # )
        # self.W_e = nn.Sequential(
        #     LayerNorm(edge_in_dim, learnable_vector_weight=True),
        #     GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
        # )

        self.dy = dy
        self.layers = nn.ModuleList(
            GVPTransformerLayer(node_h_dim, edge_h_dim, dy, dk, dv, de, db,
                                attn_heads, n_feedforward=n_feedforward,
                                drop_rate=drop_rate, vector_gate=vector_gate,
                                activations=(F.relu, None), attention=attention)
            for _ in range(num_layers))

        self.W_v_out = GVP(node_h_dim, (node_out_nf, 1), activations=(None, None), vector_gate=vector_gate)
        self.W_e_out = GVP(edge_h_dim, (edge_out_nf, 0), activations=(None, None), vector_gate=vector_gate)
        # self.W_v_out = nn.Sequential(
        #     LayerNorm(node_h_dim, learnable_vector_weight=True),
        #     GVP(node_h_dim, (node_out_nf, 1), activations=(None, None)),
        # )
        # self.W_e_out = nn.Sequential(
        #     LayerNorm(edge_h_dim, learnable_vector_weight=True),
        #     GVP(edge_h_dim, (edge_out_nf, 0), activations=(None, None))
        # )

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

        bs = len(batch_mask.unique())

        # h_v = (h, x.unsqueeze(-2))
        h_v = h if v is None else (h, v)
        h_e = self.edge_features(h, x, edge_index, batch_mask, edge_attr)

        h_v = self.W_v(h_v)
        h_e = self.W_e(h_e)
        h_y = (torch.zeros(bs, self.dy[0], device=h.device),
               torch.zeros(bs, self.dy[1], 3, device=h.device))

        for layer in self.layers:
            h_v, h_e, h_y = layer(h_v, edge_index, batch_mask, h_e, h_y)

        # h, x = self.W_v_out(h_v)
        # x = x.squeeze(-2)
        h, vel = self.W_v_out(h_v)
        # x = x + vel.squeeze(-2)

        edge_attr = self.W_e_out(h_e)

        # return h, x, edge_attr
        return h, vel.squeeze(-2), edge_attr


if __name__ == "__main__":
    from src.model.gvp import randn
    from scipy.spatial.transform import Rotation

    def test_equivariance(model, nodes, edges, glob_feat):
        random = torch.as_tensor(Rotation.random().as_matrix(),
                                 dtype=torch.float32, device=device)

        with torch.no_grad():
            X_out, E_out, y_out = model(nodes, edges, glob_feat)
            n_v_rot, e_v_rot, y_v_rot = nodes[1] @ random, edges[1] @ random, glob_feat[1] @ random
            X_out_v_rot = X_out[1] @ random
            E_out_v_rot = E_out[1] @ random
            y_out_v_rot = y_out[1] @ random
            X_out_prime, E_out_prime, y_out_prime = model((nodes[0], n_v_rot), (edges[0], e_v_rot), (glob_feat[0], y_v_rot))

            assert torch.allclose(X_out[0], X_out_prime[0], atol=1e-5, rtol=1e-4)
            assert torch.allclose(X_out_v_rot, X_out_prime[1], atol=1e-5, rtol=1e-4)
            assert torch.allclose(E_out[0], E_out_prime[0], atol=1e-5, rtol=1e-4)
            assert torch.allclose(E_out_v_rot, E_out_prime[1], atol=1e-5, rtol=1e-4)
            assert torch.allclose(y_out[0], y_out_prime[0], atol=1e-5, rtol=1e-4)
            assert torch.allclose(y_out_v_rot, y_out_prime[1], atol=1e-5, rtol=1e-4)
            print("SUCCESS")


    n_nodes = 300
    n_edges = 10000
    batch_size = 6

    node_dim = (16, 8)
    edge_dim = (8, 4)
    global_dim = (4, 2)
    dk = (6, 3)
    dv = (7, 4)
    de = (5, 2)
    db = 10
    attn_heads = 9

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    nodes = randn(n_nodes, node_dim, device=device)
    edges = randn(n_edges, edge_dim, device=device)
    glob_feat = randn(batch_size, global_dim, device=device)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), device=device)
    batch_idx = torch.randint(0, batch_size, (n_nodes,), device=device)

    model = GVPTransformerLayer(node_dim, edge_dim, global_dim, dk, dv, de, db,
                                attn_heads, n_feedforward = 2,
                                drop_rate = 0.1).to(device).eval()

    model_fn = lambda h_V, h_E, h_y: model(h_V, edge_index, batch_idx, h_E, h_y)
    test_equivariance(model_fn, nodes, edges, glob_feat)
