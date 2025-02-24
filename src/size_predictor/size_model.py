from typing import Optional
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch_scatter import scatter_mean

from src.model.gvp import GVP, GVPModel, LayerNorm, GVPConvLayer
from src.model.gvp_transformer import GVPTransformerModel, GVPTransformerLayer
from src.constants import aa_decoder, residue_bond_encoder
from src.data.dataset import ProcessedLigandPocketDataset
import src.utils as utils


class SizeModel(pl.LightningModule):
    def __init__(
            self,
            max_size,
            pocket_representation,
            train_params,
            loss_params,
            eval_params,
            predictor_params,
    ):
        super(SizeModel, self).__init__()
        self.save_hyperparameters()

        assert pocket_representation == "CA+"
        self.pocket_representation = pocket_representation

        self.type = loss_params.type
        assert self.type in {'classifier', 'ordinal', 'regression'}

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_transform = None

        # Training parameters
        self.datadir = train_params.datadir
        self.batch_size = train_params.batch_size
        self.lr = train_params.lr
        self.num_workers = train_params.num_workers
        self.clip_grad = train_params.clip_grad
        if self.clip_grad:
            self.gradnorm_queue = utils.Queue()
            # Add large value that will be flushed.
            self.gradnorm_queue.add(3000)

        # Feature encoders/decoders
        self.aa_decoder = aa_decoder
        self.residue_bond_encoder = residue_bond_encoder

        # Set up the neural network
        self.edge_cutoff = predictor_params.edge_cutoff
        self.add_nma_feat = predictor_params.normal_modes
        self.max_size = max_size
        self.n_classes = max_size if self.type == 'ordinal' else max_size + 1
        backbone = predictor_params.backbone
        model_params = getattr(predictor_params, backbone + '_params')

        self.residue_nf = (len(self.aa_decoder), 0)
        if self.add_nma_feat:
            self.residue_nf = (self.residue_nf[0], self.residue_nf[1] + 5)

        out_nf = 1 if self.type == "regression" else self.n_classes

        if backbone == 'gvp_transformer':
            self.net = SizeGVPTransformer(
                node_in_dim=self.residue_nf,
                node_h_dim=model_params.node_h_dim,
                out_nf=out_nf,
                edge_in_nf=len(self.residue_bond_encoder),
                edge_h_dim=model_params.edge_h_dim,
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

        elif backbone == 'gvp_gnn':
            self.net = SizeGVPModel(
                node_in_dim=self.residue_nf,
                node_h_dim=model_params.node_h_dim,
                out_nf=out_nf,
                edge_in_nf=len(self.residue_bond_encoder),
                edge_h_dim=model_params.edge_h_dim,
                num_layers=model_params.n_layers,
                drop_rate=model_params.dropout,
                vector_gate=model_params.vector_gate,
                reflection_equiv=model_params.reflection_equivariant,
                d_max=model_params.d_max,
                num_rbf=model_params.num_rbf,
            )

        else:
            raise NotImplementedError(f"{backbone} is not available")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr,
                                 amsgrad=True, weight_decay=1e-12)

    def setup(self, stage: Optional[str] = None):

        if stage == 'fit':
            self.train_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'train.pt'),
                ligand_transform=None, catch_errors=True)
                # ligand_transform=self.data_transform, catch_errors=True)
            self.val_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'val.pt'), ligand_transform=None)
        elif stage == 'test':
            self.test_dataset = ProcessedLigandPocketDataset(
                Path(self.datadir, 'test.pt'), ligand_transform=None)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True,
                          num_workers=self.num_workers,
                          # collate_fn=self.train_dataset.collate_fn,
                          collate_fn=partial(self.train_dataset.collate_fn, ligand_transform=self.data_transform),
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.val_dataset.collate_fn,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.test_dataset.collate_fn,
                          pin_memory=True)

    def forward(self, pocket):

        # x: CA coordinates
        x, h, mask = pocket['x'], pocket['one_hot'], pocket['mask']

        edges = None
        if 'bonds' in pocket:
            edges = (pocket['bonds'], pocket['bond_one_hot'])

        v = None
        if self.add_nma_feat:
            v = pocket['nma_vec']

        if edges is not None:
            # make sure messages are passed both ways
            edge_indices = torch.cat(
                [edges[0], edges[0].flip(dims=[0])], dim=1)
            edge_types = torch.cat([edges[1], edges[1]], dim=0)

        edges, edge_feat = self.get_edges(
            mask, x, bond_inds=edge_indices, bond_feat=edge_types)

        assert torch.all(mask[edges[0]] == mask[edges[1]])

        out = self.net(h, x, edges, v=v, batch_mask=mask, edge_attr=edge_feat)

        if torch.any(torch.isnan(out)):
            # print("NaN detected in network output")
            # out[torch.isnan(out)] = 0.0
            if self.training:
                print("NaN detected in network output")
                out[torch.isnan(out)] = 0.0
            else:
                raise ValueError("NaN detected in network output")

        return out

    def get_edges(self, batch_mask, coord, bond_inds=None, bond_feat=None, self_edges=False):

        # Adjacency matrix
        adj = batch_mask[:, None] == batch_mask[None, :]

        if self.edge_cutoff is not None:
            adj = adj & (torch.cdist(coord, coord) <= self.edge_cutoff)

            # Add missing bonds if they got removed
            adj[bond_inds[0], bond_inds[1]] = True

        if not self_edges:
            adj = adj ^ torch.eye(*adj.size(), out=torch.empty_like(adj))

        # Feature matrix
        nobond_onehot = F.one_hot(torch.tensor(
            self.residue_bond_encoder['NOBOND'], device=bond_feat.device),
            num_classes=len(self.residue_bond_encoder)).float()
        # nobond_emb = self.residue_bond_encoder(nobond_onehot.to(FLOAT_TYPE))
        # feat = nobond_emb.repeat(*adj.shape, 1)
        feat = nobond_onehot.repeat(*adj.shape, 1)
        feat[bond_inds[0], bond_inds[1]] = bond_feat

        # Return results
        edges = torch.stack(torch.where(adj), dim=0)
        edge_feat = feat[edges[0], edges[1]]

        return edges, edge_feat

    def compute_loss(self, pred_logits, true_size):

        if self.type == "classifier":
            loss = F.cross_entropy(pred_logits, true_size)

        elif self.type == "ordinal":
            # each binary variable corresponds to P(x > i), i=0,...,(max_size-1)
            binary_labels = true_size.unsqueeze(1) > torch.arange(self.n_classes, device=true_size.device).unsqueeze(0)
            loss = F.binary_cross_entropy_with_logits(pred_logits, binary_labels.float())

        elif self.type == 'regression':
            loss = F.mse_loss(pred_logits.squeeze(), true_size.float())

        else:
            raise NotImplementedError()

        return loss

    def max_likelihood(self, pred_logits):

        if self.type == "classifier":
            pred = pred_logits.argmax(dim=-1)

        elif self.type == "ordinal":
            # convert probabilities from P(x > i), i=0,...,(max_size-1) to
            # P(i), i=0,...,max_size
            prop_greater = pred_logits.sigmoid()
            pred = torch.zeros((pred_logits.size(0), pred_logits.size(1) + 1),
                               device=pred_logits.device)
            pred[:, 0] = 1 - prop_greater[:, 0]
            pred[:, 1:-1] = prop_greater[:, :-1] - prop_greater[:, 1:]
            pred[:, -1] = prop_greater[:, -1]
            pred = pred.argmax(dim=-1)

        elif self.type == 'regression':
            pred = torch.clip(torch.round(pred_logits),
                              min=0, max=self.max_size)
            pred = pred.squeeze()

        else:
            raise NotImplementedError()

        return pred

    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{m}/{split}', value, batch_size=batch_size, **kwargs)

    def compute_metrics(self, pred_logits, target):

        pred = self.max_likelihood(pred_logits)

        accuracy = (pred == target).sum() / len(target)
        mse = torch.mean((target - pred).float()**2)

        acc_window3 = (torch.abs(target - pred) <= 1).sum() / len(target)
        acc_window5 = (torch.abs(target - pred) <= 2).sum() / len(target)

        return {'accuracy': accuracy,
                'mse': mse,
                'accuracy_window3': acc_window3,
                'accuracy_window5': acc_window5}

    def training_step(self, data, *args):

        ligand, pocket = data['ligand'], data['pocket']

        try:
            pred_logits = self.forward(pocket)
            true_size = ligand['size']

        except RuntimeError as e:
            # this is not supported for multi-GPU
            if self.trainer.num_devices < 2 and 'out of memory' in str(e):
                print('WARNING: ran out of memory, skipping to the next batch')
                return None
            else:
                raise e
        loss = self.compute_loss(pred_logits, true_size)

        # Compute metrics
        metrics = self.compute_metrics(pred_logits, true_size)
        self.log_metrics({'loss': loss, **metrics}, 'train',
                         batch_size=len(true_size), prog_bar=False)

        return loss

    def validation_step(self, data, *args):
        ligand, pocket = data['ligand'], data['pocket']

        pred_logits = self.forward(pocket)
        true_size = ligand['size']

        loss = self.compute_loss(pred_logits, true_size)

        # Compute metrics
        metrics = self.compute_metrics(pred_logits, true_size)
        self.log_metrics({'loss': loss, **metrics}, 'val', batch_size=len(true_size))

        return loss

    def configure_gradient_clipping(self, optimizer, optimizer_idx,
                                    gradient_clip_val, gradient_clip_algorithm):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + \
                        2 * self.gradnorm_queue.std()

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')


class SizeGVPTransformer(GVPTransformerModel):
    """
    GVP-Transformer model

    :param node_in_dim: node dimension in input graph, scalars or tuple (scalars, vectors)
    :param node_h_dim: node dimensions to use in GVP-GNN layers, tuple (s, V)
    :param out_nf: node dimensions of output feature, tuple (s, V)
    :param edge_in_nf: edge dimension in input graph (scalars)
    :param edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers,
        tuple (s, V)
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    :param reflection_equiv: bool, use reflection-sensitive feature based on the
        cross product if False
    :param d_max:
    :param num_rbf:
    :param vector_gate: use vector gates in all GVPs
    :param attention: can be used to turn off the attention mechanism
    """
    def __init__(self, node_in_dim, node_h_dim, out_nf, edge_in_nf,
                 edge_h_dim, num_layers, dk, dv, de, db, dy,
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

        self.dy = dy
        self.layers = nn.ModuleList(
            GVPTransformerLayer(node_h_dim, edge_h_dim, dy, dk, dv, de, db,
                                attn_heads, n_feedforward=n_feedforward,
                                drop_rate=drop_rate, vector_gate=vector_gate,
                                activations=(F.relu, None), attention=attention)
            for _ in range(num_layers))

        self.W_y_out = GVP(dy, (out_nf, 0), activations=(None, None), vector_gate=vector_gate)

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

        return self.W_y_out(h_y)


class SizeGVPModel(GVPModel):
    """
    GVP-GNN model
    inspired by: https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    and: https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/atom3d.py#L115

    :param node_in_dim: node dimension in input graph, scalars or tuple (scalars, vectors)
    :param node_h_dim: node dimensions to use in GVP-GNN layers, tuple (s, V)
    :param out_nf: node dimensions of output feature, tuple (s, V)
    :param edge_in_nf: edge dimension in input graph (scalars)
    :param edge_h_dim: edge dimensions to embed to before use in GVP-GNN layers,
        tuple (s, V)
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
    def __init__(self, node_in_dim, node_h_dim, out_nf,
                 edge_in_nf, edge_h_dim, num_layers=3, drop_rate=0.1,
                 vector_gate=False, reflection_equiv=True, d_max=20.0,
                 num_rbf=16):

        super(GVPModel, self).__init__()

        self.reflection_equiv = reflection_equiv
        self.d_max = d_max
        self.num_rbf = num_rbf

        if not isinstance(node_in_dim, tuple):
            node_in_dim = (node_in_dim, 0)

        edge_in_dim = (edge_in_nf + 2 * node_in_dim[0] + self.num_rbf, 1)
        if not self.reflection_equiv:
            edge_in_dim = (edge_in_dim[0], edge_in_dim[1] + 1)

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim, learnable_vector_weight=True),
            GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=vector_gate),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim, learnable_vector_weight=True),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=vector_gate),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate,
                         update_edge_attr=True, activations=(F.relu, None),
                         vector_gate=vector_gate, ln_vector_weight=True)
            for _ in range(num_layers))

        self.W_y_out = nn.Sequential(
            # LayerNorm(node_h_dim, learnable_vector_weight=True),
            # GVP(node_h_dim, node_h_dim, vector_gate=vector_gate),
            LayerNorm(node_h_dim, learnable_vector_weight=True),
            GVP(node_h_dim, (out_nf, 0), activations=(None, None), vector_gate=vector_gate),
        )

    def forward(self, h, x, edge_index, v=None, batch_mask=None, edge_attr=None):

        batch_size = len(torch.unique(batch_mask))

        h_v = h if v is None else (h, v)
        h_e = self.edge_features(h, x, edge_index, batch_mask, edge_attr)

        h_v = self.W_v(h_v)
        h_e = self.W_e(h_e)

        for layer in self.layers:
            h_v, h_e = layer(h_v, edge_index, edge_attr=h_e)

        # compute graph-level feature
        sm = scatter_mean(h_v[0], batch_mask, dim=0, dim_size=batch_size)
        vm = scatter_mean(h_v[1], batch_mask, dim=0, dim_size=batch_size)

        return self.W_y_out((sm, vm))
