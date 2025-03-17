import warnings
import tempfile
from typing import Optional, Union
from time import time
from pathlib import Path
from functools import partial
from itertools import accumulate
from argparse import Namespace

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.distributions.categorical import Categorical
import pytorch_lightning as pl
from torch_scatter import scatter_mean

import src.utils as utils
from src.constants import atom_encoder, atom_decoder, aa_encoder, aa_decoder, \
    bond_encoder, bond_decoder, residue_encoder, residue_bond_encoder, \
    residue_decoder, residue_bond_decoder, aa_atom_index, aa_atom_mask
from src.data.dataset import ProcessedLigandPocketDataset, ClusteredDataset, get_wds
from src.data import data_utils
from src.data.data_utils import AppendVirtualNodesInCoM, center_data, Residues, TensorDict, randomize_tensors
from src.model.flows import CoordICFM, TorusICFM, CoordICFMPredictFinal, TorusICFMPredictFinal, SO3ICFM
from src.model.markov_bridge import UniformPriorMarkovBridge, MarginalPriorMarkovBridge
from src.model.dynamics import Dynamics
from src.model.dynamics_hetero import DynamicsHetero
from src.model.diffusion_utils import DistributionNodes
from src.model.loss_utils import TimestepWeights, clash_loss
from src.analysis.visualization_utils import pocket_to_rdkit, mols_to_pdbfile
from src.analysis.metrics import MoleculeValidity, CategoricalDistribution, MolecularProperties
from src.data.molecule_builder import build_molecule
from src.data.postprocessing import process_all
from src.sbdd_metrics.metrics import FullEvaluator
from src.sbdd_metrics.evaluation import VALIDITY_METRIC_NAME, aggregated_metrics, collection_metrics
from tqdm import tqdm

# derive additional constants
aa_atom_mask_tensor = torch.tensor([aa_atom_mask[aa] for aa in aa_decoder])
aa_atom_decoder = {aa: {v: k for k, v in aa_atom_index[aa].items()} for aa in aa_decoder}
aa_atom_type_tensor = torch.tensor([[atom_encoder.get(aa_atom_decoder[aa].get(i, '-')[0], -42)
                                     for i in range(14)] for aa in aa_decoder])


def set_default(namespace, key, default_val):
    val = vars(namespace).get(key, default_val)
    setattr(namespace, key, val)


class DrugFlow(pl.LightningModule):
    def __init__(
            self,
            pocket_representation: str,
            train_params: Namespace,
            loss_params: Namespace,
            eval_params: Namespace,
            predictor_params: Namespace,
            simulation_params: Namespace,
            virtual_nodes: Union[list, None],
            flexible: bool,
            flexible_bb: bool = False,
            debug: bool = False,
            overfit: bool = False,
    ):
        super(DrugFlow, self).__init__()
        self.save_hyperparameters()

        # Set default parameters
        set_default(train_params, "sharded_dataset", False)
        set_default(train_params, "sample_from_clusters", False)
        set_default(train_params, "lr_step_size", None)
        set_default(train_params, "lr_gamma", None)
        set_default(train_params, "gnina", None)
        set_default(loss_params, "lambda_x", 1.0)
        set_default(loss_params, "lambda_clash", None)
        set_default(loss_params, "reduce", "mean")
        set_default(loss_params, "regularize_uncertainty", None)
        set_default(eval_params, "n_loss_per_sample", 1)
        set_default(eval_params, "n_sampling_steps", simulation_params.n_steps)
        set_default(predictor_params, "transform_sc_pred", False)
        set_default(predictor_params, "add_chi_as_feature", False)
        set_default(predictor_params, "augment_residue_sc", False)
        set_default(predictor_params, "augment_ligand_sc", False)
        set_default(predictor_params, "add_all_atom_diff", False)
        set_default(predictor_params, "angle_act_fn", None)
        set_default(simulation_params, "predict_confidence", False)
        set_default(simulation_params, "predict_final", False)
        set_default(simulation_params, "scheduler_chi", None)

        # Check for invalid configurations
        assert pocket_representation in {'side_chain_bead', 'CA+'}
        self.pocket_representation = pocket_representation

        assert flexible or not predictor_params.augment_residue_sc
        self.augment_residue_sc = predictor_params.augment_residue_sc \
            if 'augment_residue_sc' in predictor_params else False
        self.augment_ligand_sc = predictor_params.augment_ligand_sc \
            if 'augment_ligand_sc' in predictor_params else False

        assert not (flexible_bb and predictor_params.normal_modes), \
            "Normal mode eigenvectors are only meaningful for fixed backbones"
        assert (not flexible_bb) or flexible, \
            "Currently atom vectors aren't updated if flexible=False"

        assert not (simulation_params.predict_confidence and
                    (not predictor_params.heterogeneous_graph or simulation_params.predict_final))

        # Set parameters
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.virtual_nodes = virtual_nodes
        self.flexible = flexible
        self.flexible_bb = flexible_bb
        self.debug = debug
        self.overfit = overfit
        self.predict_confidence = simulation_params.predict_confidence

        if self.virtual_nodes:
            self.add_virtual_min = virtual_nodes[0]
            self.add_virtual_max = virtual_nodes[1]

        # Training parameters
        self.datadir = train_params.datadir
        self.receptor_dir = train_params.datadir
        self.batch_size = train_params.batch_size
        self.lr = train_params.lr
        self.lr_step_size = train_params.lr_step_size
        self.lr_gamma = train_params.lr_gamma
        self.num_workers = train_params.num_workers
        self.sample_from_clusters = train_params.sample_from_clusters
        self.sharded_dataset = train_params.sharded_dataset
        self.clip_grad = train_params.clip_grad
        if self.clip_grad:
            self.gradnorm_queue = utils.Queue()
            # Add large value that will be flushed.
            self.gradnorm_queue.add(3000)

        # Evaluation parameters
        self.outdir = eval_params.outdir
        self.eval_batch_size = eval_params.eval_batch_size
        self.eval_epochs = eval_params.eval_epochs
        # assert eval_params.visualize_sample_epoch % self.eval_epochs == 0
        self.visualize_sample_epoch = eval_params.visualize_sample_epoch
        self.visualize_chain_epoch = eval_params.visualize_chain_epoch
        self.sample_with_ground_truth_size = eval_params.sample_with_ground_truth_size
        self.n_loss_per_sample = eval_params.n_loss_per_sample
        self.n_eval_samples = eval_params.n_eval_samples
        self.n_visualize_samples = eval_params.n_visualize_samples
        self.keep_frames = eval_params.keep_frames
        self.gnina = train_params.gnina

        # Feature encoders/decoders
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.bond_encoder = bond_encoder
        self.bond_decoder = bond_decoder
        self.aa_encoder = aa_encoder
        self.aa_decoder = aa_decoder
        self.residue_encoder = residue_encoder
        self.residue_decoder = residue_decoder
        self.residue_bond_encoder = residue_bond_encoder
        self.residue_bond_decoder = residue_bond_decoder

        self.atom_nf = len(self.atom_decoder)
        self.residue_nf = len(self.aa_decoder)
        if self.pocket_representation == 'side_chain_bead':
            self.residue_nf += len(self.residue_encoder)
        if self.pocket_representation == 'CA+':
            self.aa_atom_index = aa_atom_index
            self.n_atom_aa = max([x for aa in aa_atom_index.values() for x in aa.values()]) + 1
            self.residue_nf = (self.residue_nf, self.n_atom_aa)  # (s, V)
        self.bond_nf = len(self.bond_decoder)
        self.pocket_bond_nf = len(self.residue_bond_decoder)
        self.x_dim = 3

        # Set up the neural network
        self.dynamics = self.init_model(predictor_params)

        # Initialize objects for each variable type
        if simulation_params.predict_final:
            self.module_x = CoordICFMPredictFinal(None)
            self.module_chi = TorusICFMPredictFinal(None, 5) if self.flexible else None
            if self.flexible_bb:
                raise NotImplementedError()
        else:
            self.module_x = CoordICFM(None)
            # self.module_chi = AngleICFM(None, 5) if self.flexible else None
            scheduler_args = None if simulation_params.scheduler_chi is None else vars(simulation_params.scheduler_chi)
            self.module_chi = TorusICFM(None, 5, scheduler_args) if self.flexible else None
            self.module_trans = CoordICFM(None) if self.flexible_bb else None
            self.module_rot = SO3ICFM(None) if self.flexible_bb else None

        if simulation_params.prior_h == 'uniform':
            self.module_h = UniformPriorMarkovBridge(self.atom_nf, loss_type=loss_params.discrete_loss)
        elif simulation_params.prior_h == 'marginal':
            self.register_buffer('prior_h', self.get_categorical_prop('atom'))  # add to module
            self.module_h = MarginalPriorMarkovBridge(self.atom_nf, self.prior_h, loss_type=loss_params.discrete_loss)

        if simulation_params.prior_e == 'uniform':
            self.module_e = UniformPriorMarkovBridge(self.bond_nf, loss_type=loss_params.discrete_loss)
        elif simulation_params.prior_e == 'marginal':
            self.register_buffer('prior_e', self.get_categorical_prop('bond'))  # add to module
            self.module_e = MarginalPriorMarkovBridge(self.bond_nf, self.prior_e, loss_type=loss_params.discrete_loss)


        # Loss parameters
        self.loss_reduce = loss_params.reduce
        self.lambda_x = loss_params.lambda_x
        self.lambda_h = loss_params.lambda_h
        self.lambda_e = loss_params.lambda_e
        self.lambda_chi = loss_params.lambda_chi if self.flexible else None
        self.lambda_trans = loss_params.lambda_trans if self.flexible_bb else None
        self.lambda_rot = loss_params.lambda_rot if self.flexible_bb else None
        self.lambda_clash = loss_params.lambda_clash
        self.regularize_uncertainty = loss_params.regularize_uncertainty

        if loss_params.timestep_weights is not None:
            weight_type = loss_params.timestep_weights.split('_')[0]
            kwargs = loss_params.timestep_weights.split('_')[1:]
            kwargs = {x.split('=')[0]: float(x.split('=')[1]) for x in kwargs}
            self.timestep_weights = TimestepWeights(weight_type, **kwargs)
        else:
            self.timestep_weights = None


        # Sampling
        self.T_sampling = eval_params.n_sampling_steps
        self.train_step_size = 1 / simulation_params.n_steps
        self.size_distribution = None  # initialized only if needed


        # Metrics, initialized only if needed
        self.train_smiles = None
        self.ligand_metrics = None
        self.molecule_properties = None
        self.evaluator = None
        self.ligand_atom_type_distribution = None
        self.ligand_bond_type_distribution = None

        # containers for metric aggregation
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def on_load_checkpoint(self, checkpoint):
        """
        This hook is only used for backward compatibility with checkpoints that
        did not save prior_h and prior_e in state_dict in the past
        """
        if hasattr(self, "prior_h") and "prior_h" not in checkpoint["state_dict"]:
            checkpoint["state_dict"]["prior_h"] = self.get_categorical_prop('atom')
        if hasattr(self, "prior_e") and "prior_e" not in checkpoint["state_dict"]:
            checkpoint["state_dict"]["prior_e"] = self.get_categorical_prop('bond')
        if "prior_e" in checkpoint["state_dict"] and not hasattr(self, "prior_e"):
            # NOTE: a very exotic case that happened to one model. Potentially can be removed in the future
            self.register_buffer("prior_e", self.get_categorical_prop('bond'))

    def init_model(self, predictor_params):

        model_type = predictor_params.backbone

        if 'heterogeneous_graph' in predictor_params and predictor_params.heterogeneous_graph:
            return DynamicsHetero(
                atom_nf=self.atom_nf,
                residue_nf=self.residue_nf,
                bond_dict=self.bond_encoder,
                pocket_bond_dict=self.residue_bond_encoder,
                model=model_type,
                num_rbf_time=predictor_params.__dict__.get('num_rbf_time'),
                model_params=getattr(predictor_params, model_type + '_params'),
                edge_cutoff_ligand=predictor_params.edge_cutoff_ligand,
                edge_cutoff_pocket=predictor_params.edge_cutoff_pocket,
                edge_cutoff_interaction=predictor_params.edge_cutoff_interaction,
                predict_angles=self.flexible,
                predict_frames=self.flexible_bb,
                add_cycle_counts=predictor_params.cycle_counts,
                add_spectral_feat=predictor_params.spectral_feat,
                add_nma_feat=predictor_params.normal_modes,
                reflection_equiv=predictor_params.reflection_equivariant,
                d_max=predictor_params.d_max,
                num_rbf_dist=predictor_params.num_rbf,
                self_conditioning=predictor_params.self_conditioning,
                augment_residue_sc=self.augment_residue_sc,
                augment_ligand_sc=self.augment_ligand_sc,
                add_chi_as_feature=predictor_params.add_chi_as_feature,
                angle_act_fn=predictor_params.angle_act_fn,
                add_all_atom_diff=predictor_params.add_all_atom_diff,
                predict_confidence=self.predict_confidence,
            )

        else:
            if predictor_params.__dict__.get('num_rbf_time') is not None:
                raise NotImplementedError("RBF time embedding not yet implemented")

            return Dynamics(
                atom_nf=self.atom_nf,
                residue_nf=self.residue_nf,
                joint_nf=predictor_params.joint_nf,
                bond_dict=self.bond_encoder,
                pocket_bond_dict=self.residue_bond_encoder,
                edge_nf=predictor_params.edge_nf,
                hidden_nf=predictor_params.hidden_nf,
                model=model_type,
                model_params=getattr(predictor_params, model_type + '_params'),
                edge_cutoff_ligand=predictor_params.edge_cutoff_ligand,
                edge_cutoff_pocket=predictor_params.edge_cutoff_pocket,
                edge_cutoff_interaction=predictor_params.edge_cutoff_interaction,
                predict_angles=self.flexible,
                predict_frames=self.flexible_bb,
                add_cycle_counts=predictor_params.cycle_counts,
                add_spectral_feat=predictor_params.spectral_feat,
                add_nma_feat=predictor_params.normal_modes,
                self_conditioning=predictor_params.self_conditioning,
                augment_residue_sc=self.augment_residue_sc,
                augment_ligand_sc=self.augment_ligand_sc,
                add_chi_as_feature=predictor_params.add_chi_as_feature,
                angle_act_fn=predictor_params.angle_act_fn,
            )

    def _load_histogram(self, type):
        """
        Load empirical categorical distributions of atom or bond types from disk.
        Returns None if the required file is not found.
        """
        assert type in {"atom", "bond"}
        filename = 'ligand_type_histogram.npy' if type == 'atom' else 'ligand_bond_type_histogram.npy'
        encoder = self.atom_encoder if type == 'atom' else self.bond_encoder
        hist_file = Path(self.datadir, filename)
        if not hist_file.exists():
            return None
        hist = np.load(hist_file, allow_pickle=True).item()
        return CategoricalDistribution(hist, encoder)

    def get_categorical_prop(self, type):
        hist = self._load_histogram(type)
        encoder = self.atom_encoder if type == 'atom' else self.bond_encoder
        # Note: default value ensures that code will crash if prior is not
        #  read from disk or loaded from checkpoint later on
        return torch.zeros(len(encoder)) * float("nan") if hist is None else torch.tensor(hist.p)

    def configure_optimizers(self):
        optimizers = [
            torch.optim.AdamW(self.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12),
        ]

        if self.lr_step_size is None or self.lr_gamma is None:
            lr_schedulers = []
        else:
            lr_schedulers = [
                torch.optim.lr_scheduler.StepLR(optimizers[0], step_size=self.lr_step_size, gamma=self.lr_gamma),
            ]
        return optimizers, lr_schedulers

    def setup(self, stage: Optional[str] = None):

        self.setup_sampling()

        if stage == 'fit':
            self.train_dataset = self.get_dataset(stage='train')
            self.val_dataset = self.get_dataset(stage='val')
            self.setup_metrics()
        elif stage == 'val':
            self.val_dataset = self.get_dataset(stage='val')
            self.setup_metrics()
        elif stage == 'test':
            self.test_dataset = self.get_dataset(stage='test')
            self.setup_metrics()
        elif stage == 'generation':
            pass
        else:
            raise NotImplementedError

    def get_dataset(self, stage, pocket_transform=None):

        # when sampling we don't append virtual nodes as we might need access to the ground truth size
        if self.virtual_nodes and stage == "train":
            ligand_transform = AppendVirtualNodesInCoM(
                atom_encoder, bond_encoder, add_min=self.add_virtual_min, add_max=self.add_virtual_max)
        else:
            ligand_transform = None

        # we want to know if something goes wrong on the validation or test set
        catch_errors = stage == "train"

        if self.sharded_dataset:
            return get_wds(
                data_path=self.datadir,
                stage='val' if self.debug else stage,
                ligand_transform=ligand_transform,
                pocket_transform=pocket_transform,
            )

        if self.sample_from_clusters and stage == "train":  # val/test should be deterministic
            return ClusteredDataset(
                pt_path=Path(self.datadir, 'val.pt' if self.debug else f'{stage}.pt'),
                ligand_transform=ligand_transform,
                pocket_transform=pocket_transform,
                catch_errors=catch_errors
            )

        return ProcessedLigandPocketDataset(
            pt_path=Path(self.datadir, 'val.pt' if self.debug else f'{stage}.pt'),
            ligand_transform=ligand_transform,
            pocket_transform=pocket_transform,
            catch_errors=catch_errors
        )

    def setup_sampling(self):
        # distribution of nodes
        histogram_file = Path(self.datadir, 'size_distribution.npy')  # TODO: store this in model checkpoint so that we can sample without this file
        size_histogram = np.load(histogram_file).tolist()
        self.size_distribution = DistributionNodes(size_histogram)

    def setup_metrics(self):
        # For metrics
        smiles_file = Path(self.datadir, 'train_smiles.npy')
        self.train_smiles = None if not smiles_file.exists() else np.load(smiles_file)

        self.ligand_metrics = MoleculeValidity()
        self.molecule_properties = MolecularProperties()
        self.evaluator = FullEvaluator(gnina=self.gnina, exclude_evaluators=['geometry', 'ring_count'])
        self.ligand_atom_type_distribution = self._load_histogram('atom')
        self.ligand_bond_type_distribution = self._load_histogram('bond')

    def train_dataloader(self):
        shuffle = None if self.overfit else False if self.sharded_dataset else True
        return DataLoader(self.train_dataset, self.batch_size, shuffle=shuffle,
                          sampler=SubsetRandomSampler([0]) if self.overfit else None,
                          num_workers=self.num_workers,
                          collate_fn=self.train_dataset.collate_fn,
                          # collate_fn=partial(self.train_dataset.collate_fn, ligand_transform=batch_transform),
                          pin_memory=True)

    def val_dataloader(self):
        if self.overfit:
            return self.train_dataloader()

        return DataLoader(self.val_dataset, self.eval_batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.val_dataset.collate_fn,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.eval_batch_size, shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.test_dataset.collate_fn,
                          pin_memory=True)

    def log_metrics(self, metrics_dict, split, batch_size=None, **kwargs):
        for m, value in metrics_dict.items():
            self.log(f'{m}/{split}', value, batch_size=batch_size, **kwargs)

    def aggregate_metrics(self, step_outputs, prefix):
        if 'timestep' in step_outputs[0]:
            timesteps = torch.cat([x['timestep'] for x in step_outputs]).squeeze()

        if 'loss_per_sample' in step_outputs[0]:
            losses = torch.cat([x['loss_per_sample'] for x in step_outputs])
            pearson_corr = torch.corrcoef(torch.stack([timesteps, losses], dim=0))[0, 1]
            self.log(f'corr_loss_timestep/{prefix}', pearson_corr, prog_bar=False)

        if 'eps_hat_norm' in step_outputs[0]:
            eps_norm = torch.cat([x['eps_hat_norm'] for x in step_outputs])
            pearson_corr = torch.corrcoef(torch.stack([timesteps, eps_norm], dim=0))[0, 1]
            self.log(f'corr_eps_timestep/{prefix}', pearson_corr, prog_bar=False)

    def on_train_epoch_end(self):
        self.aggregate_metrics(self.training_step_outputs, 'train')
        self.training_step_outputs.clear()

    # TODO: doesn't work in multi-GPU mode
    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     """
    #     Performs operations on data before it is transferred to the GPU.
    #     Hence, supports multiple dataloaders for speedup.
    #     """
    #     batch['pocket'] = Residues(**batch['pocket'])
    #     return batch

    # # TODO: try if this is compatible with DDP
    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     """
    #     Performs operations on data after it is transferred to the GPU.
    #     """
    #     batch['pocket'] = Residues(**batch['pocket'])
    #     batch['ligand'] = TensorDict(**batch['ligand'])
    #     return batch

    def get_sc_transform_fn(self, zt_chi, zt_x, t, z0_chi, ligand_mask, pocket):
        sc_transform = {}

        if self.augment_residue_sc:
            def pred_all_atom(pred_chi, pred_trans=None, pred_rot=None):
                temp_pocket = pocket.deepcopy()

                if pred_trans is not None and pred_rot is not None:
                    zt_trans = pocket['x']
                    zt_rot = pocket['axis_angle']
                    z1_trans_pred = self.module_trans.get_z1_given_zt_and_pred(
                        zt_trans, pred_trans, None, t, pocket['mask'])
                    z1_rot_pred = self.module_rot.get_z1_given_zt_and_pred(
                        zt_rot, pred_rot, None, t, pocket['mask'])
                    temp_pocket.set_frame(z1_trans_pred, z1_rot_pred)

                z1_chi_pred = self.module_chi.get_z1_given_zt_and_pred(
                    zt_chi[..., :5], pred_chi, z0_chi, t, pocket['mask'])
                temp_pocket.set_chi(z1_chi_pred)

                all_coord = temp_pocket['v'] + temp_pocket['x'].unsqueeze(1)
                return all_coord - pocket['x'].unsqueeze(1)

            sc_transform['residues'] = pred_all_atom

        if self.augment_ligand_sc:
            # sc_transform['atoms'] = partial(self.module_x.get_z1_given_zt_and_pred, zt=zs_x, z0=None, t=t, batch_mask=lig_mask)
            sc_transform['atoms'] = lambda pred: (self.module_x.get_z1_given_zt_and_pred(
                zt_x, pred.squeeze(1), None, t, ligand_mask) - zt_x).unsqueeze(1)

        return sc_transform

    def compute_loss(self, ligand, pocket, return_info=False):
        """
        Samples time steps and computes network predictions
        """
        # TODO: move somewhere else (like collate_fn)
        pocket = Residues(**pocket)

        # Center sample
        ligand, pocket = center_data(ligand, pocket)
        if pocket['x'].numel() > 0:
            pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        else:
            pocket_com = scatter_mean(ligand['x'], ligand['mask'], dim=0)

        # # Normalize pocket coordinates
        # pocket['x'] = self.module_x.normalize(pocket['x'])

        # Sample a timestep t for each example in batch
        t = torch.rand(ligand['size'].size(0), device=ligand['x'].device).unsqueeze(-1)

        # Noise
        z0_x = self.module_x.sample_z0(pocket_com, ligand['mask'])
        z0_h = self.module_h.sample_z0(ligand['mask'])
        z0_e = self.module_e.sample_z0(ligand['bond_mask'])
        zt_x = self.module_x.sample_zt(z0_x, ligand['x'], t, ligand['mask'])
        zt_h = self.module_h.sample_zt(z0_h, ligand['one_hot'], t, ligand['mask'])
        zt_e = self.module_e.sample_zt(z0_e, ligand['bond_one_hot'], t, ligand['bond_mask'])

        if self.flexible_bb:
            z0_trans = self.module_trans.sample_z0(pocket_com, pocket['mask'])
            z1_trans = pocket['x'].detach().clone()
            zt_trans = self.module_trans.sample_zt(z0_trans, z1_trans, t, pocket['mask'])

            z0_rot = self.module_rot.sample_z0(pocket['mask'])
            z1_rot = pocket['axis_angle'].detach().clone()
            zt_rot = self.module_rot.sample_zt(z0_rot, z1_rot, t, pocket['mask'])

            # update pocket
            pocket.set_frame(zt_trans, zt_rot)

        z0_chi, zt_chi = None, None
        if self.flexible:
            # residues = [data_utils.residue_from_internal_coord(ic) for ic in pocket['residues']]
            # residues = pocket['residues']
            # z1_chi = torch.stack([data_utils.get_torsion_angles(r, device=self.device) for r in residues], dim=0)
            z1_chi = pocket['chi'][:, :5].detach().clone()

            z0_chi = self.module_chi.sample_z0(pocket['mask'])
            zt_chi = self.module_chi.sample_zt(z0_chi, z1_chi, t, pocket['mask'])

            # internal to external coordinates
            pocket.set_chi(zt_chi)
        if pocket['x'].numel() == 0:
            pocket.set_empty_v()

        # Predict denoising
        sc_transform = self.get_sc_transform_fn(zt_chi, zt_x, t, z0_chi, ligand['mask'], pocket)
        # sc_transform = None
        pred_ligand, pred_residues = self.dynamics(
            zt_x, zt_h, ligand['mask'], pocket, t,
            bonds_ligand=(ligand['bonds'], zt_e), sc_transform=sc_transform
        )

        # Compute L2 loss
        if self.predict_confidence:
            loss_x = self.module_x.compute_loss(pred_ligand['vel'], z0_x, ligand['x'], t, ligand['mask'], reduce='none')

            # compute confidence regularization
            k = self.module_x.dim  # pred.size(-1)
            sigma = pred_ligand['uncertainty_vel']
            loss_x = loss_x / (2 * sigma ** 2) + k * torch.log(sigma)

            if self.regularize_uncertainty is not None:
                loss_x = loss_x + self.regularize_uncertainty * (pred_ligand['uncertainty_vel'] - 1) ** 2

            loss_x = self.module_x.reduce_loss(loss_x, ligand['mask'], reduce=self.loss_reduce)
        else:
            loss_x = self.module_x.compute_loss(pred_ligand['vel'], z0_x, ligand['x'], t, ligand['mask'], reduce=self.loss_reduce)

        # Loss for categorical variables
        t_next = torch.clamp(t + self.train_step_size, max=1.0)
        loss_h = self.module_h.compute_loss(pred_ligand['logits_h'], zt_h, ligand['one_hot'], ligand['mask'], t, t_next, reduce=self.loss_reduce)
        loss_e = self.module_e.compute_loss(pred_ligand['logits_e'], zt_e, ligand['bond_one_hot'], ligand['bond_mask'], t, t_next, reduce=self.loss_reduce)

        loss = self.lambda_x * loss_x + self.lambda_h * loss_h + self.lambda_e * loss_e
        if self.flexible:
            loss_chi = self.module_chi.compute_loss(pred_residues['chi'], z0_chi, z1_chi, zt_chi, t, pocket['mask'], reduce=self.loss_reduce)
            loss = loss + self.lambda_chi * loss_chi

        if self.flexible_bb:
            loss_trans = self.module_trans.compute_loss(pred_residues['trans'], z0_trans, z1_trans, t, pocket['mask'], reduce=self.loss_reduce)
            loss_rot = self.module_rot.compute_loss(pred_residues['rot'], z0_rot, z1_rot, zt_rot, t, pocket['mask'], reduce=self.loss_reduce)
            loss = loss + self.lambda_trans * loss_trans + self.lambda_rot * loss_rot

        if self.lambda_clash is not None and self.lambda_clash > 0:

            if self.flexible_bb:
                pred_z1_trans = self.module_trans.get_z1_given_zt_and_pred(zt_trans, pred_residues['trans'], z0_trans, t, pocket['mask'])
                pred_z1_rot = self.module_rot.get_z1_given_zt_and_pred(zt_rot, pred_residues['rot'], z0_rot, t, pocket['mask'])
                pocket.set_frame(pred_z1_trans, pred_z1_rot)

            if self.flexible:
                # internal to external coordinates
                pred_z1_chi = self.module_chi.get_z1_given_zt_and_pred(zt_chi, pred_residues['chi'], z0_chi, t, pocket['mask'])
                pocket.set_chi(pred_z1_chi)

            pocket_coord = pocket['x'].unsqueeze(1) + pocket['v']
            pocket_types = aa_atom_type_tensor[pocket['one_hot'].argmax(dim=-1)]
            pocket_mask = pocket['mask'].unsqueeze(-1).repeat((1, pocket['v'].size(1)))

            # Extract only existing atoms
            atom_mask = aa_atom_mask_tensor[pocket['one_hot'].argmax(dim=-1)]
            pocket_coord = pocket_coord[atom_mask]
            pocket_types = pocket_types[atom_mask]
            pocket_mask = pocket_mask[atom_mask]

            # pred_z1_x = pred_x + z0_x
            pred_z1_x = self.module_x.get_z1_given_zt_and_pred(zt_x, pred_ligand['vel'], z0_x, t, ligand['mask'])
            pred_z1_h = pred_ligand['logits_h'].argmax(dim=-1)
            loss_clash = clash_loss(pred_z1_x, pred_z1_h, ligand['mask'],
                                    pocket_coord, pocket_types, pocket_mask)
            loss = loss + self.lambda_clash * loss_clash

        if self.timestep_weights is not None:
            w_t = self.timestep_weights(t).squeeze()
            loss = w_t * loss

        loss = loss.mean(0)

        info = {
            'loss_x': loss_x.mean().item(),
            'loss_h': loss_h.mean().item(),
            'loss_e': loss_e.mean().item(),
        }
        if self.flexible:
            info['loss_chi'] = loss_chi.mean().item()
        if self.flexible_bb:
            info['loss_trans'] = loss_trans.mean().item()
            info['loss_rot'] = loss_rot.mean().item()
        if self.lambda_clash is not None:
            info['loss_clash'] = loss_clash.mean().item()
        if self.predict_confidence:
            sigma_x_mol = scatter_mean(pred_ligand['uncertainty_vel'], ligand['mask'], dim=0)
            info['pearson_sigma_x'] = torch.corrcoef(torch.stack([sigma_x_mol.detach(), t.squeeze()]))[0, 1].item()
            info['mean_sigma_x'] = sigma_x_mol.mean().item()
            entropy_h = Categorical(logits=pred_ligand['logits_h']).entropy()
            entropy_h_mol = scatter_mean(entropy_h, ligand['mask'], dim=0)
            info['pearson_entropy_h'] = torch.corrcoef(torch.stack([entropy_h_mol.detach(), t.squeeze()]))[0, 1].item()
            info['mean_entropy_h'] = entropy_h_mol.mean().item()
            entropy_e = Categorical(logits=pred_ligand['logits_e']).entropy()
            entropy_e_mol = scatter_mean(entropy_e, ligand['bond_mask'], dim=0)
            info['pearson_entropy_e'] = torch.corrcoef(torch.stack([entropy_e_mol.detach(), t.squeeze()]))[0, 1].item()
            info['mean_entropy_e'] = entropy_e_mol.mean().item()

        return (loss, info) if return_info else loss

    def training_step(self, data, *args):
        ligand, pocket = data['ligand'], data['pocket']
        try:
            loss, info = self.compute_loss(ligand, pocket, return_info=True)
        except RuntimeError as e:
            # this is not supported for multi-GPU
            if self.trainer.num_devices < 2 and 'out of memory' in str(e):
                print('WARNING: ran out of memory, skipping to the next batch')
                return None
            else:
                raise e

        log_dict = {k: v for k, v in info.items() if isinstance(v, float)
                    or torch.numel(v) <= 1}
        # if self.learn_nu:
        #     log_dict['nu_x'] = self.noise_schedules['x'].nu.item()
        #     log_dict['nu_h'] = self.noise_schedules['h'].nu.item()
        #     log_dict['nu_e'] = self.noise_schedules['e'].nu.item()

        self.log_metrics({'loss': loss, **log_dict}, 'train',
                         batch_size=len(ligand['size']))

        out = {'loss': loss, **info}
        self.training_step_outputs.append(out)
        return out

    def validation_step(self, data, *args):

        # Compute the loss N times and average to get a better estimate
        loss_list, info_list = [], []
        self.dynamics.train()  # TODO: this is currently necessary to make self-conditioning work
        for _ in range(self.n_loss_per_sample):
            loss, info = self.compute_loss(data['ligand'].copy(),
                                           data['pocket'].copy(),
                                           return_info=True)
            loss_list.append(loss.item())
            info_list.append(info)
        self.dynamics.eval()
        if len(loss_list) >= 1:
            loss = np.mean(loss_list)
            info = {k: np.mean([x[k] for x in info_list]) for k in info_list[0]}
            self.log_metrics({'loss': loss, **info}, 'val', batch_size=len(data['ligand']['size']))

        # Sample
        rdmols, rdpockets, _ = self.sample(
            data=data,
            n_samples=self.n_eval_samples,
            num_nodes="ground_truth" if self.sample_with_ground_truth_size else None,
        )

        out = {
            'ligands': rdmols,
            'pockets': rdpockets,
            'receptor_files': [Path(self.receptor_dir, 'val', x) for x in data['pocket']['name']]
        }
        self.validation_step_outputs.append(out)
        return out

    # def test_step(self, data, *args):
    #     self._shared_eval(data, 'test', *args)

    def on_validation_epoch_end(self):

        outdir = Path(self.outdir, f'epoch_{self.current_epoch}')

        rdmols = [m for x in self.validation_step_outputs for m in x['ligands']]
        rdpockets = [p for x in self.validation_step_outputs for p in x['pockets']]
        receptors = [r for x in self.validation_step_outputs for r in x['receptor_files']]
        self.validation_step_outputs.clear()

        ligand_atom_types = [atom_encoder[a.GetSymbol()] for m in rdmols for a in m.GetAtoms()]
        ligand_bond_types = []
        for m in rdmols:
            bonds = m.GetBonds()
            no_bonds = m.GetNumAtoms() * (m.GetNumAtoms() - 1) // 2 - m.GetNumBonds()
            ligand_bond_types += [bond_encoder['NOBOND']] * no_bonds
            for b in bonds:
                ligand_bond_types.append(bond_encoder[b.GetBondType().name])

        tic = time()
        results = self.analyze_sample(
            rdmols, ligand_atom_types, ligand_bond_types, receptors=(rdpockets if len(rdpockets) != 0 else None)
        )
        self.log_metrics(results, 'val')
        print(f'Evaluation took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_sample_epoch == 0:
            tic = time()

            outdir.mkdir(exist_ok=True, parents=True)

            # center for better visualization
            rdmols = rdmols[:self.n_visualize_samples]
            rdpockets = rdpockets[:self.n_visualize_samples]
            for m, p in zip(rdmols, rdpockets):
                center = m.GetConformer().GetPositions().mean(axis=0)
                for i in range(m.GetNumAtoms()):
                    x, y, z = m.GetConformer().GetPositions()[i] - center
                    m.GetConformer().SetAtomPosition(i, (x, y, z))
                for i in range(p.GetNumAtoms()):
                    x, y, z = p.GetConformer().GetPositions()[i] - center
                    p.GetConformer().SetAtomPosition(i, (x, y, z))

            # save molecule
            utils.write_sdf_file(Path(outdir, 'molecules.sdf'), rdmols)

            # save pocket
            utils.write_sdf_file(Path(outdir, 'pockets.sdf'), rdpockets)

            print(f'Sample visualization took {time() - tic:.2f} seconds')

        if (self.current_epoch + 1) % self.visualize_chain_epoch == 0:
            tic = time()
            outdir.mkdir(exist_ok=True, parents=True)

            if self.sharded_dataset:
                index = torch.randint(len(self.val_dataset), size=(1,)).item()
                for i, x in enumerate(self.val_dataset):
                    if i == index:
                        break
                batch = self.val_dataset.collate_fn([x])
            else:
                batch = self.val_dataset.collate_fn([self.val_dataset[torch.randint(len(self.val_dataset), size=(1,))]])
            batch['pocket'] = Residues(**batch['pocket']).to(self.device)
            pocket_copy = batch['pocket'].copy()

            if len(batch['pocket']['x']) > 0:
                ligand_chain, pocket_chain, info = self.sample_chain(batch['pocket'], self.keep_frames)
            else:
                num_nodes, _ = self.size_distribution.sample()
                ligand_chain, pocket_chain, info = self.sample_chain(batch['pocket'], self.keep_frames, num_nodes=num_nodes)

            # utils.write_sdf_file(Path(outdir, 'chain_pocket.sdf'), pocket_chain)
            # utils.write_chain(Path(outdir, 'chain_pocket.xyz'), pocket_chain)
            if self.flexible or self.flexible_bb:
                # insert ground truth at the beginning so that it's used by PyMOL to determine the connectivity
                ground_truth_pocket = pocket_to_rdkit(
                    pocket_copy, self.pocket_representation,
                    self.atom_encoder, self.atom_decoder,
                    self.aa_decoder, self.residue_decoder,
                    self.aa_atom_index
                )[0]
                ground_truth_ligand = build_molecule(
                    batch['ligand']['x'], batch['ligand']['one_hot'].argmax(1),
                    bonds=batch['ligand']['bonds'],
                    bond_types=batch['ligand']['bond_one_hot'].argmax(1),
                    atom_decoder=self.atom_decoder,
                    bond_decoder=self.bond_decoder
                )
                pocket_chain.insert(0, ground_truth_pocket)
                ligand_chain.insert(0, ground_truth_ligand)
                # pocket_chain.insert(0, pocket_chain[-1])
                # ligand_chain.insert(0, ligand_chain[-1])

            # save molecules
            utils.write_sdf_file(Path(outdir, 'chain_ligand.sdf'), ligand_chain)

            # save pocket
            mols_to_pdbfile(pocket_chain, Path(outdir, 'chain_pocket.pdb'))

            self.log_metrics(info, 'val')
            print(f'Chain visualization took {time() - tic:.2f} seconds')


    # NOTE: temporary fix of this Lightning bug:
    # https://github.com/Lightning-AI/pytorch-lightning/discussions/18110
    # Without it resume training has a strange behavior and fails
    @property
    def total_batch_idx(self) -> int:
        """Returns the current batch index (across epochs)"""
        # use `ready` instead of `completed` in case this is accessed after `completed` has been increased
        # but before the next `ready` increase
        return max(0, self.batch_progress.total.ready - 1)

    @property
    def batch_idx(self) -> int:
        """Returns the current batch index (within this epoch)"""
        # use `ready` instead of `completed` in case this is accessed after `completed` has been increased
        # but before the next `ready` increase
        return max(0, self.batch_progress.current.ready - 1)

    # def analyze_sample(self, rdmols, atom_types, bond_types, aa_types=None, receptors=None):
    #     out = {}

    #     # Distribution of node types
    #     kl_div_atom = self.ligand_atom_type_distribution.kl_divergence(atom_types) \
    #         if self.ligand_atom_type_distribution is not None else -1
    #     out['kl_div_atom_types'] = kl_div_atom

    #     # Distribution of edge types
    #     kl_div_bond = self.ligand_bond_type_distribution.kl_divergence(bond_types) \
    #         if self.ligand_bond_type_distribution is not None else -1
    #     out['kl_div_bond_types'] = kl_div_bond

    #     if aa_types is not None:
    #         kl_div_aa = self.pocket_type_distribution.kl_divergence(aa_types) \
    #             if self.pocket_type_distribution is not None else -1
    #         out['kl_div_residue_types'] = kl_div_aa

    #     # Post-process sample
    #     processed_mols = [process_all(m) for m in rdmols]

    #     # Other basic metrics
    #     results = self.ligand_metrics(rdmols)
    #     out['n_samples'] = results['n_total']
    #     out['Validity'] = results['validity']
    #     out['Connectivity'] = results['connectivity']
    #     out['valid_and_connected'] = results['valid_and_connected']

    #     # connected_mols = [get_largest_fragment(m) for m in rdmols]
    #     connected_mols = [process_all(m, largest_frag=True, adjust_aromatic_Ns=False, relax_iter=0) for m in rdmols]
    #     connected_mols = [m for m in connected_mols if m is not None]
    #     out.update(self.molecule_properties(connected_mols))

    #     # Repeat after post-processing
    #     results = self.ligand_metrics(processed_mols)
    #     out['validity_processed'] = results['validity']
    #     out['connectivity_processed'] = results['connectivity']
    #     out['valid_and_connected_processed'] = results['valid_and_connected']

    #     processed_mols = [m for m in processed_mols if m is not None]
    #     for k, v in self.molecule_properties(processed_mols).items():
    #         out[f"{k}_processed"] = v

    #     # Simple docking score
    #     if receptors is not None and self.gnina is not None:
    #         assert len(receptors) == len(rdmols)
    #         docking_results = compute_gnina_scores(rdmols, receptors, gnina=self.gnina)
    #         out.update(docking_results)

    #     # Clash score
    #     if receptors is not None:
    #         assert len(receptors) == len(rdmols)
    #         clashes = {
    #             'ligands': [legacy_clash_score(m) for m in rdmols],
    #             'pockets': [legacy_clash_score(p) for p in receptors],
    #             'between': [legacy_clash_score(m, p) for m, p in zip(rdmols, receptors)],
    #             'v2_ligands': [clash_score(m) for m in rdmols],
    #             'v2_pockets': [clash_score(p) for p in receptors],
    #             'v2_between': [clash_score(m, p) for m, p in zip(rdmols, receptors)]
    #         }
    #         for k, v in clashes.items():
    #             out[f'mean_clash_score_{k}'] = np.mean(v)
    #             out[f'frac_no_clashes_{k}'] = np.mean(np.array(v) <= 0.0)

    #     return out

    def analyze_sample(self, rdmols, atom_types, bond_types, aa_types=None, receptors=None):
        out = {}

        # Distribution of node types
        kl_div_atom = self.ligand_atom_type_distribution.kl_divergence(atom_types) \
            if self.ligand_atom_type_distribution is not None else -1
        out['kl_div_atom_types'] = kl_div_atom

        # Distribution of edge types
        kl_div_bond = self.ligand_bond_type_distribution.kl_divergence(bond_types) \
            if self.ligand_bond_type_distribution is not None else -1
        out['kl_div_bond_types'] = kl_div_bond

        if aa_types is not None:
            kl_div_aa = self.pocket_type_distribution.kl_divergence(aa_types) \
                if self.pocket_type_distribution is not None else -1
            out['kl_div_residue_types'] = kl_div_aa

        # Evaluation
        results = []
        if receptors is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                for mol, receptor in zip(tqdm(rdmols, desc='FullEvaluator'), receptors):
                    receptor_path = Path(tmpdir, 'receptor.pdb')
                    Chem.MolToPDBFile(receptor, str(receptor_path))
                    results.append(self.evaluator(mol, receptor_path))
        else:
            for mol in tqdm(rdmols, desc='FullEvaluator'):
                self.evaluator = FullEvaluator(pb_conf='mol')
                results.append(self.evaluator(mol))

        results = pd.DataFrame(results)
        agg_results = aggregated_metrics(results, self.evaluator.dtypes, VALIDITY_METRIC_NAME).fillna(0)
        agg_results['metric'] = agg_results['metric'].str.replace('.', '/')

        col_results = collection_metrics(results, self.train_smiles, VALIDITY_METRIC_NAME, exclude_evaluators='fcd')
        col_results['metric'] = 'collection/' + col_results['metric']

        all_results = pd.concat([agg_results, col_results])
        out.update(**dict(all_results[['metric', 'value']].values))

        return out

    def sample_zt_given_zs(self, zs_ligand, zs_pocket, s, t, delta_eps_x=None, uncertainty=None):

        sc_transform = self.get_sc_transform_fn(zs_pocket.get('chi'), zs_ligand['x'], s, None, zs_ligand['mask'], zs_pocket)
        pred_ligand, pred_residues = self.dynamics(
            zs_ligand['x'], zs_ligand['h'], zs_ligand['mask'], zs_pocket, s, bonds_ligand=(zs_ligand['bonds'], zs_ligand['e']),
            sc_transform=sc_transform
        )

        if delta_eps_x is not None:
            pred_ligand['vel'] = pred_ligand['vel'] + delta_eps_x

        zt_ligand = zs_ligand.copy()
        zt_ligand['x'] = self.module_x.sample_zt_given_zs(zs_ligand['x'], pred_ligand['vel'], s, t, zs_ligand['mask'])

        zt_ligand['h'] = self.module_h.sample_zt_given_zs(zs_ligand['h'], pred_ligand['logits_h'], s, t, zs_ligand['mask'])
        zt_ligand['e'] = self.module_e.sample_zt_given_zs(zs_ligand['e'], pred_ligand['logits_e'], s, t, zs_ligand['edge_mask'])

        zt_pocket = zs_pocket.copy()
        if self.flexible_bb:
            zt_trans_pocket = self.module_trans.sample_zt_given_zs(zs_pocket['x'], pred_residues['trans'], s, t, zs_pocket['mask'])
            zt_rot_pocket = self.module_rot.sample_zt_given_zs(zs_pocket['axis_angle'], pred_residues['rot'], s, t, zs_pocket['mask'])

            # update pocket in-place
            zt_pocket.set_frame(zt_trans_pocket, zt_rot_pocket)

        if self.flexible:
            zt_chi_pocket = self.module_chi.sample_zt_given_zs(zs_pocket['chi'][..., :5], pred_residues['chi'], s, t, zs_pocket['mask'])

            # update pocket in-place
            zt_pocket.set_chi(zt_chi_pocket)

        if self.predict_confidence:
            assert uncertainty is not None
            dt = (t - s).view(-1)[zt_ligand['mask']]
            uncertainty['sigma_x_squared'] += (dt * pred_ligand['uncertainty_vel']**2)
            uncertainty['entropy_h'] += (dt * Categorical(logits=pred_ligand['logits_h']).entropy())

        return zt_ligand, zt_pocket

    def simulate(self, ligand, pocket, timesteps, t_start, t_end=1.0,
                 return_frames=1, guide_log_prob=None):
        """
        Take a version of the ligand and pocket (at any time step t_start) and
        simulate the generative process from t_start to t_end.
        """

        assert 0 < return_frames <= timesteps
        assert timesteps % return_frames == 0
        assert 0.0 <= t_start < 1.0
        assert 0 < t_end <= 1.0
        assert t_start < t_end

        device = ligand['x'].device
        n_samples = len(pocket['size'])
        delta_t = (t_end - t_start) / timesteps

        # Initialize output tensors
        out_ligand = {
            'x': torch.zeros((return_frames, len(ligand['mask']), self.x_dim), device=device),
            'h': torch.zeros((return_frames, len(ligand['mask']), self.atom_nf), device=device),
            'e': torch.zeros((return_frames, len(ligand['edge_mask']), self.bond_nf), device=device)
        }
        if self.predict_confidence:
            out_ligand['sigma_x'] = torch.zeros((return_frames, len(ligand['mask'])), device=device)
            out_ligand['entropy_h'] = torch.zeros((return_frames, len(ligand['mask'])), device=device)
        out_pocket = {
            'x': torch.zeros((return_frames, len(pocket['mask']), 3), device=device),  # CA-coord
            'v': torch.zeros((return_frames, len(pocket['mask']), self.n_atom_aa, 3), device=device)  # difference vectors to all other atoms
        }

        cumulative_uncertainty = {
            'sigma_x_squared': torch.zeros(len(ligand['mask']), device=device),
            'entropy_h': torch.zeros(len(ligand['mask']), device=device)
        } if self.predict_confidence else None

        for i, t in enumerate(torch.linspace(t_start, t_end - delta_t, timesteps)):
            t_array = torch.full((n_samples, 1), fill_value=t, device=device)

            if guide_log_prob is not None:
                raise NotImplementedError('Not yet implemented for flow matching model')
                alpha_t = self.diffusion_x.schedule.alpha(self.gamma_x(t_array))

                with torch.enable_grad():
                    zt_x_ligand.requires_grad = True
                    g = guide_log_prob(t_array, x=ligand['x'], h=ligand['h'], batch_mask=ligand['mask'],
                                       bonds=ligand['bonds'], bond_types=ligand['e'])

                    # Compute gradient w.r.t. coordinates
                    grad_x_lig = torch.autograd.grad(g.sum(), inputs=ligand['x'])[0]

                    # clip gradients
                    g_max = 1.0
                    clip_mask = (grad_x_lig.norm(dim=-1) > g_max)
                    grad_x_lig[clip_mask] = \
                        grad_x_lig[clip_mask] / grad_x_lig[clip_mask].norm(
                            dim=-1, keepdim=True) * g_max

                delta_eps_lig = -1 * (1 - alpha_t[lig_mask]).sqrt() * grad_x_lig
            else:
                delta_eps_lig = None

            ligand, pocket = self.sample_zt_given_zs(
                ligand, pocket, t_array, t_array + delta_t, delta_eps_lig, cumulative_uncertainty)

            # save frame
            if (i + 1) % (timesteps // return_frames) == 0:
                idx = (i + 1) // (timesteps // return_frames)
                idx = idx - 1

                out_ligand['x'][idx] = ligand['x'].detach()
                out_ligand['h'][idx] = ligand['h'].detach()
                out_ligand['e'][idx] = ligand['e'].detach()
                if pocket['x'].numel() > 0:
                    out_pocket['x'][idx] = pocket['x'].detach()
                    out_pocket['v'][idx] = pocket['v'][:, :self.n_atom_aa, :].detach()
                if self.predict_confidence:
                    out_ligand['sigma_x'][idx] = cumulative_uncertainty['sigma_x_squared'].sqrt().detach()
                    out_ligand['entropy_h'][idx] = cumulative_uncertainty['entropy_h'].detach()

        # remove frame dimension if only the final molecule is returned
        out_ligand = {k: v.squeeze(0) for k, v in out_ligand.items()}
        out_pocket = {k: v.squeeze(0) for k, v in out_pocket.items()}

        return out_ligand, out_pocket

    def init_ligand(self, num_nodes_lig, pocket):
        device = pocket['x'].device

        n_samples = len(pocket['size'])
        lig_mask = utils.num_nodes_to_batch_mask(n_samples, num_nodes_lig, device)

        # only consider upper triangular matrix for symmetry
        lig_bonds = torch.stack(torch.where(torch.triu(
            lig_mask[:, None] == lig_mask[None, :], diagonal=1)), dim=0)
        lig_edge_mask = lig_mask[lig_bonds[0]]

        # Sample from Normal distribution in the pocket center
        pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)
        z0_x = self.module_x.sample_z0(pocket_com, lig_mask)
        z0_h = self.module_h.sample_z0(lig_mask)
        z0_e = self.module_e.sample_z0(lig_edge_mask)

        return TensorDict(**{
            'x': z0_x, 'h': z0_h, 'e': z0_e, 'mask': lig_mask,
            'bonds': lig_bonds, 'edge_mask': lig_edge_mask
        })

    def init_pocket(self, pocket):

        if self.flexible_bb:
            pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)
            z0_trans = self.module_trans.sample_z0(pocket_com, pocket['mask'])
            z0_rot = self.module_rot.sample_z0(pocket['mask'])

            # update pocket in-place
            pocket.set_frame(z0_trans, z0_rot)

        if self.flexible:
            z0_chi = self.module_chi.sample_z0(pocket['mask'])

            # # DEBUG ##
            # z0_chi = torch.stack([data_utils.get_torsion_angles(r, device=self.device) for r in pocket['residues']], dim=0)
            # ####

            # internal to external coordinates
            pocket.set_chi(z0_chi)

        if pocket['x'].numel() == 0:
            pocket.set_empty_v()

        return pocket

    def parse_num_nodes_spec(self, batch, spec=None, size_model=None):

        if spec == "2d_histogram" or spec is None:  # default option
            assert "pocket" in batch
            num_nodes = self.size_distribution.sample_conditional(
                n1=None, n2=batch['pocket']['size'])

            # make sure there is at least one potential bond
            num_nodes[num_nodes < 2] = 2

        elif isinstance(spec, (int, torch.Tensor)):
            num_nodes = spec

        elif spec == "ground_truth":
            assert "ligand" in batch
            num_nodes = batch['ligand']['size']

        elif spec == "nn_prediction":
            assert size_model is not None
            assert "pocket" in batch
            predictions = size_model.forward(batch['pocket'])
            predictions = torch.softmax(predictions, dim=-1)
            predictions[:, :5] = 0.0
            probabilities = predictions / predictions.sum(dim=1, keepdims=True)
            num_nodes = torch.distributions.Categorical(probabilities).sample()

        elif isinstance(spec, str) and spec.startswith("uniform"):
            # expected format: uniform_low_high
            assert "pocket" in batch
            left, right = map(int, spec.split("_")[1:])
            shape = batch['pocket']['size'].shape
            num_nodes = torch.randint(left, right + 1, shape, dtype=torch.long)

        else:
            raise NotImplementedError(f"Invalid size specification {spec}")

        if self.virtual_nodes:
            num_nodes += self.add_virtual_max

        return num_nodes

    @torch.no_grad()
    def sample(self, data, n_samples, num_nodes=None, timesteps=None,
               guide_log_prob=None, size_model=None, **kwargs):

        # TODO: move somewhere else (like collate_fn)
        data['pocket'] = Residues(**data['pocket'])

        timesteps = self.T_sampling if timesteps is None else timesteps

        if len(data['pocket']['x']) > 0:
            pocket = data_utils.repeat_items(data['pocket'], n_samples)
        else:
            pocket = Residues(**{key: value for key, value in data['pocket'].items()})
            pocket['name'] = pocket['name'] * n_samples
            pocket['size'] = pocket['size'].repeat(n_samples)
            pocket['n_bonds'] = pocket['n_bonds'].repeat(n_samples)

        _ligand = data_utils.repeat_items(data['ligand'], n_samples)
        _ligand = randomize_tensors(_ligand, exclude_keys=['size', 'name'])  # avoid data leakage

        batch = {"ligand": _ligand, "pocket": pocket}
        num_nodes = self.parse_num_nodes_spec(batch, spec=num_nodes, size_model=size_model)

        # Sample from prior
        if pocket['x'].numel() > 0:
            ligand = self.init_ligand(num_nodes, pocket)
        else:
            ligand = self.init_ligand(num_nodes, _ligand)
        pocket = self.init_pocket(pocket)

        # return prior samples
        if timesteps == 0:
            # Convert into rdmols
            rdmols = [build_molecule(coords=m['x'], 
                atom_types=m['h'].argmax(1), 
                bonds=m['bonds'], 
                bond_types=m['e'].argmax(1), 
                atom_decoder=self.atom_decoder, bond_decoder=self.bond_decoder) 
                for m in data_utils.split_entity(ligand.detach().cpu(), edge_types={"e", "edge_mask"}, edge_mask=ligand["edge_mask"])]

            rdpockets = pocket_to_rdkit(pocket, self.pocket_representation,
                                        self.atom_encoder, self.atom_decoder,
                                        self.aa_decoder, self.residue_decoder,
                                        self.aa_atom_index)

            return rdmols, rdpockets, _ligand['name']

        out_tensors_ligand, out_tensors_pocket = self.simulate(
            ligand, pocket, timesteps, 0.0, 1.0,
            guide_log_prob=guide_log_prob
        )

        # Build mol objects
        x = out_tensors_ligand['x'].detach().cpu()
        ligand_type = out_tensors_ligand['h'].argmax(1).detach().cpu()
        edge_type = out_tensors_ligand['e'].argmax(1).detach().cpu()
        lig_mask = ligand['mask'].detach().cpu()
        lig_bonds = ligand['bonds'].detach().cpu()
        lig_edge_mask = ligand['edge_mask'].detach().cpu()
        sizes = torch.unique(ligand['mask'], return_counts=True)[1].tolist()
        offsets = list(accumulate(sizes[:-1], initial=0))
        mol_kwargs = {
            'coords': utils.batch_to_list(x, lig_mask),
            'atom_types': utils.batch_to_list(ligand_type, lig_mask),
            'bonds': utils.batch_to_list_for_indices(lig_bonds, lig_edge_mask, offsets),
            'bond_types': utils.batch_to_list(edge_type, lig_edge_mask)
        }
        if self.predict_confidence:
            sigma_x = out_tensors_ligand['sigma_x'].detach().cpu()
            entropy_h = out_tensors_ligand['entropy_h'].detach().cpu()
            mol_kwargs['atom_props'] = [
                {'sigma_x': x[0], 'entropy_h': x[1]}
                for x in zip(utils.batch_to_list(sigma_x, lig_mask),
                             utils.batch_to_list(entropy_h, lig_mask))
            ]
        mol_kwargs = [{k: v[i] for k, v in mol_kwargs.items()}
                      for i in range(len(mol_kwargs['coords']))]

        # Convert into rdmols
        rdmols = [build_molecule(
            **m, atom_decoder=self.atom_decoder, bond_decoder=self.bond_decoder)
            for m in mol_kwargs
        ]

        out_pocket = pocket.copy()
        out_pocket['x'] = out_tensors_pocket['x']
        out_pocket['v'] = out_tensors_pocket['v']
        rdpockets = pocket_to_rdkit(out_pocket, self.pocket_representation,
                                    self.atom_encoder, self.atom_decoder,
                                    self.aa_decoder, self.residue_decoder,
                                    self.aa_atom_index)

        return rdmols, rdpockets, _ligand['name']

    @torch.no_grad()
    def sample_chain(self, pocket, keep_frames, num_nodes=None, timesteps=None,
                     guide_log_prob=None, **kwargs):

        # TODO: move somewhere else (like collate_fn)
        pocket = Residues(**pocket)

        info = {}

        timesteps = self.T_sampling if timesteps is None else timesteps

        # n_samples = 1
        # TODO: get batch_size differently
        assert len(pocket['mask'].unique()) <= 1, "sample_chain only supports a single sample"

        # # Pocket's initial center of mass
        # pocket_com_before = scatter_mean(pocket['x'], pocket['mask'], dim=0)

        num_nodes = self.parse_num_nodes_spec(batch={"pocket": pocket}, spec=num_nodes)

        # Sample from prior
        if pocket['x'].numel() > 0:
            ligand = self.init_ligand(num_nodes, pocket)
        else:
            dummy_pocket = Residues.empty(pocket['x'].device)
            ligand = self.init_ligand(num_nodes, dummy_pocket)

        pocket = self.init_pocket(pocket)

        out_tensors_ligand, out_tensors_pocket = self.simulate(
            ligand, pocket, timesteps, 0.0, 1.0, guide_log_prob=guide_log_prob, return_frames=keep_frames)

        # chain_lig = utils.reverse_tensor(chain_lig)
        # chain_pocket = utils.reverse_tensor(chain_pocket)
        # chain_bond = utils.reverse_tensor(chain_bond)

        info['traj_displacement_lig'] = torch.norm(out_tensors_ligand['x'][-1] - out_tensors_ligand['x'][0], dim=-1).mean()
        info['traj_rms_lig'] = out_tensors_ligand['x'].std(dim=0).mean()

        # # Repeat last frame to see final sample better.
        # chain_lig = torch.cat([chain_lig, chain_lig[-1:].repeat(10, 1, 1)], dim=0)
        # chain_pocket = torch.cat([chain_pocket, chain_pocket[-1:].repeat(10, 1, 1)], dim=0)
        # chain_bond = torch.cat([chain_bond, chain_bond[-1:].repeat(10, 1, 1)], dim=0)

        # Flatten
        assert keep_frames == out_tensors_ligand['x'].size(0) == out_tensors_pocket['x'].size(0)
        n_atoms = out_tensors_ligand['x'].size(1)
        n_bonds = out_tensors_ligand['e'].size(1)
        n_residues = out_tensors_pocket['x'].size(1)
        device = out_tensors_ligand['x'].device

        def flatten_tensor(chain):
            if len(chain.size()) == 3:  # l=0 values
                return chain.view(-1, chain.size(-1))
            elif len(chain.size()) == 4:  # vectors
                return chain.view(-1, chain.size(-2), chain.size(-1))
            else:
                warnings.warn(f"Could not flatten frame dimension of tensor with shape {list(chain.size())}")
                return chain

        out_tensors_ligand_flat = {k: flatten_tensor(chain) for k, chain in out_tensors_ligand.items()}
        out_tensors_pocket_flat = {k: flatten_tensor(chain) for k, chain in out_tensors_pocket.items()}
        # ligand_flat = chain_lig.view(-1, chain_lig.size(-1))
        # ligand_mask_flat = torch.arange(chain_lig.size(0)).repeat_interleave(chain_lig.size(1)).to(chain_lig.device)
        ligand_mask_flat = torch.arange(keep_frames).repeat_interleave(n_atoms).to(device)

        # # pocket_flat = chain_pocket.view(-1, chain_pocket.size(-1))
        # # pocket_v_flat = pocket['v'].repeat(100, 1, 1)
        # pocket_flat = chain_pocket.view(-1, chain_pocket.size(-2), chain_pocket.size(-1))
        # pocket_mask_flat = torch.arange(chain_pocket.size(0)).repeat_interleave(chain_pocket.size(1)).to(chain_pocket.device)
        pocket_mask_flat = torch.arange(keep_frames).repeat_interleave(n_residues).to(device)

        # bond_flat = chain_bond.view(-1, chain_bond.size(-1))
        # bond_mask_flat = torch.arange(chain_bond.size(0)).repeat_interleave(chain_bond.size(1)).to(chain_bond.device)
        bond_mask_flat = torch.arange(keep_frames).repeat_interleave(n_bonds).to(device)
        edges_flat = ligand['bonds'].repeat(1, keep_frames)

        # # Move generated molecule back to the original pocket position
        # pocket_com_after = scatter_mean(pocket_flat[:, 0, :], pocket_mask_flat, dim=0)
        # ligand_flat[:, :self.x_dim] += (pocket_com_before - pocket_com_after)[ligand_mask_flat]
        #
        # # Move pocket back as well (for visualization purposes)
        # pocket_flat[:, 0, :] += (pocket_com_before - pocket_com_after)[pocket_mask_flat]

        # Build ligands
        x = out_tensors_ligand_flat['x'].detach().cpu()
        ligand_type = out_tensors_ligand_flat['h'].argmax(1).detach().cpu()
        ligand_mask_flat = ligand_mask_flat.detach().cpu()
        bond_mask_flat = bond_mask_flat.detach().cpu()
        edges_flat = edges_flat.detach().cpu()
        edge_type = out_tensors_ligand_flat['e'].argmax(1).detach().cpu()
        offsets = torch.zeros(keep_frames, dtype=int)  # edges_flat is already zero-based
        molecules = list(
            zip(utils.batch_to_list(x, ligand_mask_flat),
                utils.batch_to_list(ligand_type, ligand_mask_flat),
                utils.batch_to_list_for_indices(edges_flat, bond_mask_flat, offsets),
                utils.batch_to_list(edge_type, bond_mask_flat)
                )
        )

        # Convert into rdmols
        ligand_chain = [build_molecule(
            *graph, atom_decoder=self.atom_decoder,
            bond_decoder=self.bond_decoder) for graph in molecules
        ]

        # Build pockets
        # as long as the pocket does not change during sampling, we can ust
        # write it once
        out_pocket = {
            'x': out_tensors_pocket_flat['x'],
            'one_hot': pocket['one_hot'].repeat(keep_frames, 1),
            'mask': pocket_mask_flat,
            'v': out_tensors_pocket_flat['v'],
            'atom_mask': pocket['atom_mask'].repeat(keep_frames, 1),
        } if self.flexible else pocket
        pocket_chain = pocket_to_rdkit(out_pocket, self.pocket_representation,
                                       self.atom_encoder, self.atom_decoder,
                                       self.aa_decoder, self.residue_decoder,
                                       self.aa_atom_index)

        return ligand_chain, pocket_chain, info

    # def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
    # def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
    def configure_gradient_clipping(self, optimizer, *args, **kwargs):

        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * stdev of the recent history.
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + \
                        2 * self.gradnorm_queue.std()

        # hard upper limit
        max_grad_norm = min(max_grad_norm, 10.0)

        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g['params']]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(optimizer, gradient_clip_val=max_grad_norm,
                            gradient_clip_algorithm='norm')

        if float(grad_norm) > max_grad_norm:
            print(f'Clipped gradient with value {grad_norm:.1f} '
                  f'while allowed {max_grad_norm:.1f}')
            grad_norm = max_grad_norm

        self.gradnorm_queue.add(float(grad_norm))