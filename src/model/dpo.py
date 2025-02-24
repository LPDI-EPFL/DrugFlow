from typing import Optional
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean

from src.constants import atom_encoder, bond_encoder
from src.model.lightning import DrugFlow, set_default
from src.data.dataset import ProcessedLigandPocketDataset, DPODataset
from src.data.data_utils import AppendVirtualNodesInCoM, Residues, center_data

class DPO(DrugFlow):
    def __init__(self, dpo_mode, ref_checkpoint_p, **kwargs):
        super(DPO, self).__init__(**kwargs)
        self.dpo_mode = dpo_mode
        self.dpo_beta = kwargs['loss_params'].dpo_beta if 'dpo_beta' in kwargs['loss_params'] else 100.0
        self.dpo_beta_schedule = kwargs['loss_params'].dpo_beta_schedule if 'dpo_beta_schedule' in kwargs['loss_params'] else 't'
        self.clamp_dpo = kwargs['loss_params'].clamp_dpo if 'clamp_dpo' in kwargs['loss_params'] else True
        self.dpo_lambda_dpo = kwargs['loss_params'].dpo_lambda_dpo if 'dpo_lambda_dpo' in kwargs['loss_params'] else 1
        self.dpo_lambda_w = kwargs['loss_params'].dpo_lambda_w if 'dpo_lambda_w' in kwargs['loss_params'] else 1
        self.dpo_lambda_l = kwargs['loss_params'].dpo_lambda_l if 'dpo_lambda_l' in kwargs['loss_params'] else 0.2
        self.dpo_lambda_h = kwargs['loss_params'].dpo_lambda_h if 'dpo_lambda_h' in kwargs['loss_params'] else kwargs['loss_params'].lambda_h
        self.dpo_lambda_e = kwargs['loss_params'].dpo_lambda_e if 'dpo_lambda_e' in kwargs['loss_params'] else kwargs['loss_params'].lambda_e
        self.ref_dynamics = self.init_model(kwargs['predictor_params'])
        state_dict = torch.load(ref_checkpoint_p)['state_dict']
        self.ref_dynamics.load_state_dict({k.replace('dynamics.',''): v for k, v in state_dict.items() if k.startswith('dynamics.')})
        print(f'Loaded reference model from {ref_checkpoint_p}')
        # initializing model params with ref model params
        self.dynamics.load_state_dict(self.ref_dynamics.state_dict())

    def get_dataset(self, stage, pocket_transform=None):

        # when sampling we don't append virtual nodes as we might need access to the ground truth size
        if self.virtual_nodes and stage == 'train':
            ligand_transform = AppendVirtualNodesInCoM(
                atom_encoder, bond_encoder, add_min=self.add_virtual_min, add_max=self.add_virtual_max)
        else:
            ligand_transform = None

        # we want to know if something goes wrong on the validation or test set
        catch_errors = stage == 'train'

        if self.sharded_dataset:
            raise NotImplementedError('Sharded dataset not implemented for DPO')

        if self.sample_from_clusters and stage == 'train':  # val/test should be deterministic
            raise NotImplementedError('Sampling from clusters not implemented for DPO')

        if stage == 'train':
            return DPODataset(
                Path(self.datadir, 'train.pt'),
                ligand_transform=None,
                pocket_transform=pocket_transform,
                catch_errors=True,
            )
        else:
            return ProcessedLigandPocketDataset(
                pt_path=Path(self.datadir, 'val.pt' if self.debug else f'{stage}.pt'),
                ligand_transform=ligand_transform,
                pocket_transform=pocket_transform,
                catch_errors=catch_errors,
            )


    def training_step(self, data, *args):
        ligand_w, ligand_l, pocket = data['ligand'], data['ligand_l'], data['pocket']
        loss, info = self.compute_dpo_loss(pocket, ligand_w=ligand_w, ligand_l=ligand_l, return_info=True)

        if torch.isnan(loss):
            print(f'For ligand pair , loss is NaN at epoch {self.current_epoch}. Info: {info}')
        
        log_dict = {k: v for k, v in info.items() if isinstance(v, float) or torch.numel(v) <= 1}
        self.log_metrics({'loss': loss, **log_dict}, 'train', batch_size=len(ligand_w['size']))

        out = {'loss': loss, **info}
        self.training_step_outputs.append(out)
        return out
    
    def validation_step(self, data, *args):
        return super().validation_step(data, *args)

    def compute_dpo_loss(self, pocket, ligand_w, ligand_l, return_info=False):
        t = torch.rand(ligand_w['size'].size(0), device=ligand_w['x'].device).unsqueeze(-1)

        if self.dpo_beta_schedule == 't':
            # from https://arxiv.org/pdf/2407.13981
            beta_t = (self.dpo_beta * t).squeeze()
        elif self.dpo_beta_schedule == 'const':
            beta_t = self.dpo_beta
        else:
            raise ValueError(f'Unknown DPO beta schedule: {self.dpo_beta_schedule}')

        loss_dict_w = self.compute_loss_single_pair(ligand_w, pocket, t)
        loss_dict_l = self.compute_loss_single_pair(ligand_l, pocket, t)
        info = {
            'loss_x_w': loss_dict_w['theta']['x'].mean().item(),
            'loss_h_w': loss_dict_w['theta']['h'].mean().item(),
            'loss_e_w': loss_dict_w['theta']['e'].mean().item(),
            'loss_x_l': loss_dict_l['theta']['x'].mean().item(),
            'loss_h_l': loss_dict_l['theta']['h'].mean().item(),
            'loss_e_l': loss_dict_l['theta']['e'].mean().item(),
        }
        if self.dpo_mode == 'single_dpo_comp':
            loss_w_theta = (
                loss_dict_w['theta']['x'] +
                self.dpo_lambda_h * loss_dict_w['theta']['h'] +
                self.dpo_lambda_e * loss_dict_w['theta']['e']
            )
            loss_w_ref = (
                loss_dict_w['ref']['x'] +
                self.dpo_lambda_h * loss_dict_w['ref']['h'] +
                self.dpo_lambda_e * loss_dict_w['ref']['e']
            )
            loss_l_theta = (
                loss_dict_l['theta']['x'] +
                self.dpo_lambda_h * loss_dict_l['theta']['h'] +
                self.dpo_lambda_e * loss_dict_l['theta']['e']
            )
            loss_l_ref = (
                loss_dict_l['ref']['x'] +
                self.dpo_lambda_h * loss_dict_l['ref']['h'] +
                self.dpo_lambda_e * loss_dict_l['ref']['e']
            )
            diff_w = loss_w_theta - loss_w_ref
            diff_l = loss_l_theta - loss_l_ref
            info['diff_w'] = diff_w.mean().item()
            info['diff_l'] = diff_l.mean().item()
            # print(diff)
            diff = -1 * beta_t * (diff_w - diff_l)
            loss = -1 * F.logsigmoid(diff)
        elif self.dpo_mode == 'single_dpo_comp_v3':
            diff_w_x = loss_dict_w['theta']['x'] - loss_dict_w['ref']['x']
            diff_w_h = loss_dict_w['theta']['h'] - loss_dict_w['ref']['h']
            diff_w_e = loss_dict_w['theta']['e'] - loss_dict_w['ref']['e']
            diff_l_x = loss_dict_l['theta']['x'] - loss_dict_l['ref']['x']
            diff_l_h = loss_dict_l['theta']['h'] - loss_dict_l['ref']['h']
            diff_l_e = loss_dict_l['theta']['e'] - loss_dict_l['ref']['e']
            info['diff_w_x'] = diff_w_x.mean().item()
            info['diff_w_h'] = diff_w_h.mean().item()
            info['diff_w_e'] = diff_w_e.mean().item()
            info['diff_l_x'] = diff_l_x.mean().item()
            info['diff_l_h'] = diff_l_h.mean().item()
            info['diff_l_e'] = diff_l_e.mean().item()
            
            # not used, just for logging
            _diff_w = diff_w_x + self.dpo_lambda_h * diff_w_h + self.dpo_lambda_e * diff_w_e
            _diff_l = diff_l_x + self.dpo_lambda_h * diff_l_h + self.dpo_lambda_e * diff_l_e
            info['diff_w'] = _diff_w.mean().item()
            info['diff_l'] = _diff_l.mean().item()

            diff_x = diff_w_x - diff_l_x
            diff_h = diff_w_h - diff_l_h
            diff_e = diff_w_e - diff_l_e
            info['diff_x'] = diff_x.mean().item()
            info['diff_h'] = diff_h.mean().item()
            info['diff_e'] = diff_e.mean().item()

            diff = -1 * beta_t * (diff_x + self.dpo_lambda_h * diff_h + self.dpo_lambda_e * diff_e)
            if self.clamp_dpo:
                diff = diff.clamp(-10, 10)
            info['dpo_arg_min'] = diff.min().item()
            info['dpo_arg_max'] = diff.max().item()
            info['dpo_arg_mean'] = diff.mean().item()
            dpo_loss = -1 * self.dpo_lambda_dpo * F.logsigmoid(diff)
            info['dpo_loss'] = dpo_loss.mean().item()
            
            loss_w_theta_reg = (
                loss_dict_w['theta']['x'] +
                self.lambda_h * loss_dict_w['theta']['h'] +
                self.lambda_e * loss_dict_w['theta']['e']
            )
            info['loss_w_theta_reg'] = loss_w_theta_reg.mean().item()
            loss_l_theta_reg = (
                loss_dict_l['theta']['x'] +
                self.lambda_h * loss_dict_l['theta']['h'] +
                self.lambda_e * loss_dict_l['theta']['e']
            )
            info['loss_l_theta_reg'] = loss_l_theta_reg.mean().item()
            dpo_reg = self.dpo_lambda_w * loss_w_theta_reg + \
                      self.dpo_lambda_l * loss_l_theta_reg
            info['dpo_reg'] = dpo_reg.mean().item()
            loss = dpo_loss + dpo_reg
        else:
            raise ValueError(f'Unknown DPO mode: {self.dpo_mode}')

        if self.timestep_weights is not None:
            w_t = self.timestep_weights(t).squeeze()
            loss = w_t * loss

        loss = loss.mean(0)
        
        print(f'Loss is {loss}, info is {info}')

        return (loss, info) if return_info else loss

    def compute_loss_single_pair(self, ligand, pocket, t):
        pocket = Residues(**pocket)

        # Center sample
        ligand, pocket = center_data(ligand, pocket)
        pocket_com = scatter_mean(pocket['x'], pocket['mask'], dim=0)

        # Noise
        z0_x = self.module_x.sample_z0(pocket_com, ligand['mask'])
        z0_h = self.module_h.sample_z0(ligand['mask'])
        z0_e = self.module_e.sample_z0(ligand['bond_mask'])
        zt_x = self.module_x.sample_zt(z0_x, ligand['x'], t, ligand['mask'])
        zt_h = self.module_h.sample_zt(z0_h, ligand['one_hot'], t, ligand['mask'])
        zt_e = self.module_e.sample_zt(z0_e, ligand['bond_one_hot'], t, ligand['bond_mask'])

        # Predict denoising
        sc_transform = self.get_sc_transform_fn(None, zt_x, t, None, ligand['mask'], pocket)

        pred_ligand, _ = self.dynamics(
            zt_x, zt_h, ligand['mask'], pocket, t,
            bonds_ligand=(ligand['bonds'], zt_e),
            sc_transform=sc_transform
        )

        # Reference model
        with torch.no_grad():
            ref_pred_ligand, _ = self.ref_dynamics(
                zt_x, zt_h, ligand['mask'], pocket, t,
                bonds_ligand=(ligand['bonds'], zt_e),
                sc_transform=sc_transform
            )

        # Compute L2 loss
        loss_x = self.module_x.compute_loss(pred_ligand['vel'], z0_x, ligand['x'], t, ligand['mask'], reduce=self.loss_reduce)
        ref_loss_x = self.module_x.compute_loss(ref_pred_ligand['vel'], z0_x, ligand['x'], t, ligand['mask'], reduce=self.loss_reduce)

        t_next = torch.clamp(t + self.train_step_size, max=1.0)

        loss_h = self.module_h.compute_loss(pred_ligand['logits_h'], zt_h, ligand['one_hot'], ligand['mask'], t, t_next, reduce=self.loss_reduce)
        ref_loss_h = self.module_h.compute_loss(ref_pred_ligand['logits_h'], zt_h, ligand['one_hot'], ligand['mask'], t, t_next, reduce=self.loss_reduce)
        loss_e = self.module_e.compute_loss(pred_ligand['logits_e'], zt_e, ligand['bond_one_hot'], ligand['bond_mask'], t, t_next, reduce=self.loss_reduce)
        ref_loss_e = self.module_e.compute_loss(ref_pred_ligand['logits_e'], zt_e, ligand['bond_one_hot'], ligand['bond_mask'], t, t_next, reduce=self.loss_reduce)

        return {
            'theta': {
                'x': loss_x,
                'h': loss_h,
                'e': loss_e,
            },
            'ref': {
                'x': ref_loss_x,
                'h': ref_loss_h,
                'e': ref_loss_e,
            }
        }
