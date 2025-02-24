import argparse
import sys
import yaml
import torch
import numpy as np
import pickle
from argparse import Namespace

from pathlib import Path

basedir = Path(__file__).resolve().parent.parent
sys.path.append(str(basedir))

from src import utils
from src.utils import dict_to_namespace, namespace_to_dict
from src.analysis.visualization_utils import mols_to_pdbfile, mol_as_pdb
from src.data.data_utils import TensorDict, Residues
from src.data.postprocessing import process_all
from src.model.lightning import DrugFlow
from src.sbdd_metrics.evaluation import compute_all_metrics_drugflow

from tqdm import tqdm
from pdb import set_trace


def combine(base_args, override_args):
    assert not isinstance(base_args, dict)
    assert not isinstance(override_args, dict)

    arg_dict = base_args.__dict__
    for key, value in override_args.__dict__.items():
        if key not in arg_dict or arg_dict[key] is None:  # parameter not provided previously
            print(f"Add parameter {key}: {value}")
            arg_dict[key] = value
        elif isinstance(value, Namespace):
            arg_dict[key] = combine(arg_dict[key], value)
        else:
            print(f"Replace parameter {key}: {arg_dict[key]} -> {value}")
            arg_dict[key] = value
    return base_args


def path_to_str(input_dict):
    for key, value in input_dict.items():
        if isinstance(value, dict):
            input_dict[key] = path_to_str(value)
        else:
            input_dict[key] = str(value) if isinstance(value, Path) else value
    return input_dict


def sample(cfg, model_params, samples_dir, job_id=0, n_jobs=1):
    print('Sampling...')
    model = DrugFlow.load_from_checkpoint(cfg.checkpoint, map_location=cfg.device, strict=False,
                                          **model_params)
    model.setup(stage='fit' if cfg.set == 'train' else cfg.set)
    model.eval().to(cfg.device)

    dataloader = getattr(model, f'{cfg.set}_dataloader')()
    print(f'Real batch size is {dataloader.batch_size * cfg.n_samples}')

    name2count = {}
    for i, data in enumerate(tqdm(dataloader)):
        if i % n_jobs != job_id:
            print(f'Skipping batch {i}')
            continue

        new_data = {
            'ligand': TensorDict(**data['ligand']).to(cfg.device),
            'pocket': Residues(**data['pocket']).to(cfg.device),
        }
        try:
            rdmols, rdpockets, names = model.sample(
                data=new_data,
                n_samples=cfg.n_samples,
                num_nodes=("ground_truth" if cfg.sample_with_ground_truth_size else None)
            )
        except Exception as e:
            if cfg.set == 'train':
                names = data['ligand']['name']
                print(f'Failed to sample for {names}: {e}')
                continue
            else:
                raise e

        for mol, pocket, name in zip(rdmols, rdpockets, names):
            name = name.replace('.sdf', '')
            idx = name2count.setdefault(name, 0)
            output_dir = Path(samples_dir, name)
            output_dir.mkdir(parents=True, exist_ok=True)
            if cfg.postprocess:
                mol = process_all(mol, largest_frag=True, adjust_aromatic_Ns=True, relax_iter=0)

            for prop in mol.GetAtoms()[0].GetPropsAsDict().keys():
                # compute avg uncertainty
                mol.SetDoubleProp(prop, np.mean([a.GetDoubleProp(prop) for a in mol.GetAtoms()]))

                # visualise local differences
                out_pdb_path = Path(output_dir, f'{idx}_ligand_{prop}.pdb')
                mol_as_pdb(mol, out_pdb_path, bfactor=prop)

            out_sdf_path = Path(output_dir, f'{idx}_ligand.sdf')
            out_pdb_path = Path(output_dir, f'{idx}_pocket.pdb')
            utils.write_sdf_file(out_sdf_path, [mol])
            mols_to_pdbfile([pocket], out_pdb_path)

            name2count[name] += 1


def evaluate(cfg, model_params, samples_dir):
    print('Evaluation...')
    data, table_detailed, table_aggregated = compute_all_metrics_drugflow(
        in_dir=samples_dir,
        gnina_path=model_params['train_params'].gnina,
        reduce_path=cfg.reduce,
        reference_smiles_path=Path(model_params['train_params'].datadir, 'train_smiles.npy'),
        n_samples=cfg.n_samples,
        exclude_evaluators=[] if cfg.exclude_evaluators is None else cfg.exclude_evaluators,
    )
    with open(Path(samples_dir, 'metrics_data.pkl'), 'wb') as f: 
        pickle.dump(data, f)
    table_detailed.to_csv(Path(samples_dir, 'metrics_detailed.csv'), index=False)
    table_aggregated.to_csv(Path(samples_dir, 'metrics_aggregated.csv'), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str)
    p.add_argument('--job_id', type=int, default=0, help='Job ID')
    p.add_argument('--n_jobs', type=int, default=1, help='Number of jobs')
    args = p.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = dict_to_namespace(cfg)

    utils.set_deterministic(seed=cfg.seed)
    utils.disable_rdkit_logging()

    model_params = torch.load(cfg.checkpoint, map_location=cfg.device)['hyper_parameters']
    if 'model_args' in cfg:
        ckpt_args = dict_to_namespace(model_params)
        model_params = combine(ckpt_args, cfg.model_args).__dict__

    ckpt_path = Path(cfg.checkpoint)
    ckpt_name = ckpt_path.parts[-1].split('.')[0]
    n_steps = model_params['simulation_params'].n_steps
    samples_dir = Path(cfg.sample_outdir, cfg.set, f'{ckpt_name}_T={n_steps}') or \
                  Path(ckpt_path.parent.parent, 'samples', cfg.set, f'{ckpt_name}_T={n_steps}')
    assert cfg.set in {'val', 'test', 'train'}
    samples_dir.mkdir(parents=True, exist_ok=True)

    # save configs
    with open(Path(samples_dir, 'model_params.yaml'), 'w') as f:
        yaml.dump(path_to_str(namespace_to_dict(model_params)), f)
    with open(Path(samples_dir, 'sampling_params.yaml'), 'w') as f:
        yaml.dump(path_to_str(namespace_to_dict(cfg)), f)

    if cfg.sample:
        sample(cfg, model_params, samples_dir, job_id=args.job_id, n_jobs=args.n_jobs)

    if cfg.evaluate:
        assert args.job_id == 0 and args.n_jobs == 1, 'Evaluation is not parallelised on GPU machines'
        evaluate(cfg, model_params, samples_dir)