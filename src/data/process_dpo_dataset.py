import argparse
from pathlib import Path
import numpy as np
import random
import shutil
from time import time
from collections import defaultdict
from Bio.PDB import PDBParser
from rdkit import Chem
import torch
from tqdm import tqdm
import pandas as pd
from itertools import combinations

import sys
basedir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(basedir))

from src.sbdd_metrics.metrics import REOSEvaluator, MedChemEvaluator, PoseBustersEvaluator, GninaEvalulator
from src.data.data_utils import process_raw_pair, rdmol_to_smiles

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smplsdir', type=Path, required=True)
    parser.add_argument('--metrics-detailed', type=Path, required=False)
    parser.add_argument('--ignore-missing-scores', action='store_true')
    parser.add_argument('--datadir', type=Path, required=True)
    parser.add_argument('--dpo-criterion', type=str, default='reos.all', 
                        choices=['reos.all', 'medchem.sa', 'medchem.qed', 'gnina.vina_efficiency','combined'])
    parser.add_argument('--basedir', type=Path, default=None)
    parser.add_argument('--pocket', type=str, default='CA+',
                        choices=['side_chain_bead', 'CA+'])
    parser.add_argument('--gnina', type=Path, default='gnina')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--normal_modes', action='store_true')
    parser.add_argument('--flex', action='store_true')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--toy_size', type=int, default=100)
    parser.add_argument('--n_pairs', type=int, default=5)
    args = parser.parse_args()
    return args

def scan_smpl_dir(samples_dir):
    samples_dir = Path(samples_dir)
    subdirs = []
    for subdir in tqdm(samples_dir.iterdir(), desc='Scanning samples'):
        if not subdir.is_dir():
            continue
        if not sample_dir_valid(subdir):
            continue
        subdirs.append(subdir)
    return subdirs

def sample_dir_valid(samples_dir):
    pocket = samples_dir / '0_pocket.pdb'
    if not pocket.exists():
        return False
    ligands = list(samples_dir.glob('*_ligand.sdf'))
    if len(ligands) < 2:
        return False
    for ligand in ligands:
        if ligand.stat().st_size == 0:
            return False
    return True

def return_winning_losing_smpl(score_1, score_2, criterion):
    if criterion == 'reos.all':
        if score_1 == score_2:
            return None
        return score_1 > score_2
    elif criterion == 'medchem.sa':
        if np.abs(score_1 - score_2) < 0.5:
            return None
        return score_1 < score_2
    elif criterion == 'medchem.qed':
        if np.abs(score_1 - score_2) < 0.1:
            return None
        return score_1 > score_2
    elif criterion == 'gnina.vina_efficiency':
        if np.abs(score_1 - score_2) < 0.1:
            return None
        return score_1 < score_2
    elif criterion == 'combined':
        score_reos_1, score_reos_2 = score_1['reos.all'], score_2['reos.all']
        score_sa_1, score_sa_2 = score_1['medchem.sa'], score_2['medchem.sa']
        score_qed_1, score_qed_2 = score_1['medchem.qed'], score_2['medchem.qed']
        score_vina_1, score_vina_2 = score_1['gnina.vina_efficiency'], score_2['gnina.vina_efficiency']
        if score_reos_1 == score_reos_2: return None
        # checking consistency
        reos_sign = score_reos_1 > score_reos_2
        sa_sign = score_sa_1 < score_sa_2
        qed_sign = score_qed_1 > score_qed_2
        vina_sign = score_vina_1 < score_vina_2
        signs = [reos_sign, sa_sign, qed_sign, vina_sign]
        if all(signs) or not any(signs): return signs[0]
        return None
    
def compute_scores(sample_dirs, evaluator, criterion, n_pairs=5, toy=False, toy_size=100, 
                   precomp_scores=None, ignore_missing_scores=False):
    samples = []
    pose_evaluator = PoseBustersEvaluator()
    pbar = tqdm(sample_dirs, desc='Computing scores for samples')
    
    for dir in pbar:
        pocket = dir / '0_pocket.pdb'
        ligands = list(dir.glob('*_ligand.sdf'))
        
        target_samples = []
        for lig_path in ligands:
            try:
                mol = Chem.SDMolSupplier(str(lig_path))[0]
                if mol is None:
                    continue
                smiles = rdmol_to_smiles(mol)
            except Exception as e:
                print('Failed to read ligand:', lig_path)
                continue
            
            if precomp_scores is not None and str(lig_path) in precomp_scores.index:
                mol_props = precomp_scores.loc[str(lig_path)].to_dict()
                if criterion == 'combined':
                    if not 'reos.all' in mol_props or not 'medchem.sa' in mol_props or not 'medchem.qed' in mol_props or not 'gnina.vina_efficiency' in mol_props:
                        print(f'Missing combined scores for ligand:', lig_path)
                        continue
                    mol_props['combined'] = {
                        'reos.all': mol_props['reos.all'],
                        'medchem.sa': mol_props['medchem.sa'],
                        'medchem.qed': mol_props['medchem.qed'],
                        'gnina.vina_efficiency': mol_props['gnina.vina_efficiency'],
                        'combined': mol_props['gnina.vina_efficiency']
                    }
            else:
                mol_props = {}
            if criterion not in mol_props:
                if ignore_missing_scores:
                    print(f'Missing {criterion} for ligand:', lig_path)
                    continue
                print(f'Recomputing {criterion} for ligand:', lig_path)
                try:
                    eval_res = evaluator.evaluate(mol)
                    criterion_cat = criterion.split('.')[0]
                    eval_res = {f'{criterion_cat}.{k}': v for k, v in eval_res.items()}
                    score = eval_res[criterion]
                except:
                    continue
            else:
                score = mol_props[criterion]

            if 'posebusters.all' not in mol_props:
                if ignore_missing_scores:
                    print('Missing PoseBusters for ligand:', lig_path)
                    continue
                print('Recomputing PoseBusters for ligand:', lig_path)
                try:
                    pose_eval_res = pose_evaluator.evaluate(lig_path, pocket)
                except:
                    continue
                if 'all' not in pose_eval_res or not pose_eval_res['all']:
                    continue
            else:
                pose_eval_res = mol_props['posebusters.all']
                if not pose_eval_res:
                    continue
            
            target_samples.append({
                'smiles': smiles,
                'score': score,
                'ligand_path': lig_path,
                'pocket_path': pocket
            })
        
        # Deduplicate by SMILES
        unique_samples = {}
        for sample in target_samples:
            if sample['smiles'] not in unique_samples:
                unique_samples[sample['smiles']] = sample
        unique_samples = list(unique_samples.values())
        if len(unique_samples) < 2:
            continue
        
        # Generate all possible pairs
        all_pairs = list(combinations(unique_samples, 2))
        
        # Calculate score differences and filter valid pairs
        valid_pairs = []
        for s1, s2 in all_pairs:
            sign = return_winning_losing_smpl(s1['score'], s2['score'], criterion)
            if sign is None:
                continue
            score_diff = abs(s1['score'] - s2['score']) if not criterion == 'combined' else \
                         abs(s1['score']['combined'] - s2['score']['combined'])
            if sign:
                valid_pairs.append((s1, s2, score_diff))
            elif sign is False:
                valid_pairs.append((s2, s1, score_diff))
        
        # Sort pairs by score difference (descending) and select top N pairs
        valid_pairs.sort(key=lambda x: x[2], reverse=True)
        used_ligand_paths = set()
        selected_pairs = []     
        for winning, losing, score_diff in valid_pairs:
            if winning['ligand_path'] in used_ligand_paths or losing['ligand_path'] in used_ligand_paths:
                continue
            
            selected_pairs.append((winning, losing, score_diff))
            used_ligand_paths.add(winning['ligand_path'])
            used_ligand_paths.add(losing['ligand_path'])
            
            if len(selected_pairs) == n_pairs:
                break   
        for winning, losing, _ in selected_pairs:
            d = {
                'score_w': winning['score'],
                'score_l': losing['score'],
                'pocket_p': winning['pocket_path'],
                'ligand_p_w': winning['ligand_path'],
                'ligand_p_l': losing['ligand_path']
            }
            if isinstance(winning['score'], dict):
                for k, v in winning['score'].items():
                    d[f'{k}_w'] = v
                d['score_w'] = winning['score']['combined']
            if isinstance(losing['score'], dict):
                for k, v in losing['score'].items():
                    d[f'{k}_l'] = v
                d['score_l'] = losing['score']['combined']
            samples.append(d)                
        
        pbar.set_postfix({'samples': len(samples)})
        
        if toy and len(samples) >= toy_size:
            break
    
    return samples

def main():
    args = parse_args()

    if 'reos' in args.dpo_criterion:
        evaluator = REOSEvaluator()
    elif 'medchem' in args.dpo_criterion:
        evaluator = MedChemEvaluator()
    elif 'gnina' in args.dpo_criterion:
        evaluator = GninaEvalulator(gnina=args.gnina)
    elif 'combined' in args.dpo_criterion:
        evaluator = None # for combined criterion, metrics have to be computed separately
        if args.metrics_detailed is None:
            raise ValueError('For combined criterion, detailed metrics file has to be provided')
        if not args.ignore_missing_scores:
            raise ValueError('For combined criterion, --ignore-missing-scores flag has to be set')
    else:
        raise ValueError(f"Unknown DPO criterion: {args.dpo_criterion}")
    
    # Make output directory
    dirname = f"dpo_{args.dpo_criterion.replace('.','_')}_{args.pocket}"
    if args.flex:
        dirname += '_flex'
    if args.normal_modes:
        dirname += '_nma'
    if args.toy:
        dirname += '_toy'
    processed_dir = Path(args.basedir, dirname)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if (processed_dir / f'samples_{args.dpo_criterion}.csv').exists():
        print(f"Samples already computed for criterion {args.dpo_criterion}, loading from file")
        samples = pd.read_csv(processed_dir / f'samples_{args.dpo_criterion}.csv')
        samples = [dict(row) for _, row in samples.iterrows()]
        print(f"Found {len(samples)} winning/losing samples")
    else:
        print('Scanning sample directory...')
        samples_dir = Path(args.smplsdir)
        # scan dir
        sample_dirs = scan_smpl_dir(samples_dir)
        if args.metrics_detailed:
            print(f'Loading precomputed scores from {args.metrics_detailed}')
            precomp_scores = pd.read_csv(args.metrics_detailed)
            precomp_scores = precomp_scores.set_index('sdf_file')
        else:
            precomp_scores = None
        print(f'Found {len(sample_dirs)} valid sample directories')
        print('Computing scores...')
        samples = compute_scores(sample_dirs, evaluator, args.dpo_criterion, 
                                 n_pairs=args.n_pairs, toy=args.toy, toy_size=args.toy_size,
                                    precomp_scores=precomp_scores,
                                    ignore_missing_scores=args.ignore_missing_scores)
        print(f'Found {len(samples)} winning/losing samples, saving to file')
        pd.DataFrame(samples).to_csv(Path(processed_dir, f'samples_{args.dpo_criterion}.csv'), index=False)

    data_split = {}
    data_split['train'] = samples
    if args.toy:
        data_split['train'] = random.sample(samples, min(args.toy_size, len(data_split['train'])))

    failed = {}
    train_smiles = []

    for split in data_split.keys():

        print(f"Processing {split} dataset...")

        ligands_w = defaultdict(list)
        ligands_l = defaultdict(list)
        pockets = defaultdict(list)

        tic = time()
        pbar = tqdm(data_split[split])
        for entry in pbar:

            pbar.set_description(f'#failed: {len(failed)}')

            pdbfile = Path(entry['pocket_p'])
            entry['ligand_p_w'] = Path(entry['ligand_p_w'])
            entry['ligand_p_l'] = Path(entry['ligand_p_l'])
            entry['ligand_w'] = Chem.SDMolSupplier(str(entry['ligand_p_w']))[0]
            entry['ligand_l'] = Chem.SDMolSupplier(str(entry['ligand_p_l']))[0]

            try:
                pdb_model = PDBParser(QUIET=True).get_structure('', pdbfile)[0]

                ligand_w, pocket = process_raw_pair(
                    pdb_model, entry['ligand_w'], pocket_representation=args.pocket,
                    compute_nerf_params=args.flex, compute_bb_frames=args.flex,
                    nma_input=pdbfile if args.normal_modes else None)
                ligand_l, _ = process_raw_pair(
                    pdb_model, entry['ligand_l'], pocket_representation=args.pocket,
                    compute_nerf_params=args.flex, compute_bb_frames=args.flex,
                    nma_input=pdbfile if args.normal_modes else None)

            except (KeyError, AssertionError, FileNotFoundError, IndexError,
                    ValueError, AttributeError) as e:
                failed[(split, entry['ligand_p_w'], entry['ligand_p_l'],  pdbfile)] \
                    = (type(e).__name__, str(e))
                continue

            nerf_keys = ['fixed_coord', 'atom_mask', 'nerf_indices', 'length', 'theta', 'chi', 'ddihedral', 'chi_indices']
            for k in ['x', 'one_hot', 'bonds', 'bond_one_hot', 'v', 'nma_vec'] + nerf_keys + ['axis_angle']:
                if k in ligand_w:
                    ligands_w[k].append(ligand_w[k])
                    ligands_l[k].append(ligand_l[k])
                if k in pocket:
                    pockets[k].append(pocket[k])

            smpl_n = pdbfile.parent.name
            pocket_file = f'{smpl_n}__{pdbfile.stem}.pdb'
            ligand_file_w = f'{smpl_n}__{entry["ligand_p_w"].stem}.sdf'
            ligand_file_l = f'{smpl_n}__{entry["ligand_p_l"].stem}.sdf'
            ligands_w['name'].append(ligand_file_w)
            ligands_l['name'].append(ligand_file_l)
            pockets['name'].append(pocket_file)
            train_smiles.append(rdmol_to_smiles(entry['ligand_w']))
            train_smiles.append(rdmol_to_smiles(entry['ligand_l']))

        data = {'ligands_w': ligands_w, 
                'ligands_l': ligands_l,
                'pockets': pockets}
        torch.save(data, Path(processed_dir, f'{split}.pt'))

        if split == 'train':
            np.save(Path(processed_dir, 'train_smiles.npy'), train_smiles)

        print(f"Processing {split} set took {(time() - tic) / 60.0:.2f} minutes")

    # cp stats from original dataset
    size_distr_p = Path(args.datadir, 'size_distribution.npy')
    type_histo_p = Path(args.datadir, 'ligand_type_histogram.npy')
    bond_histo_p = Path(args.datadir, 'ligand_bond_type_histogram.npy')
    metadata_p = Path(args.datadir, 'metadata.yml')
    shutil.copy(size_distr_p, processed_dir)
    shutil.copy(type_histo_p, processed_dir)
    shutil.copy(bond_histo_p, processed_dir)
    shutil.copy(metadata_p, processed_dir)

    # cp val and test .pt and dirs
    val_dir = Path(args.datadir, 'val')
    test_dir = Path(args.datadir, 'test')
    val_pt = Path(args.datadir, 'val.pt')
    test_pt = Path(args.datadir, 'test.pt')
    assert val_dir.exists() and test_dir.exists() and val_pt.exists() and test_pt.exists()
    if (processed_dir / 'val').exists():
        shutil.rmtree(processed_dir / 'val')
    if (processed_dir / 'test').exists():
        shutil.rmtree(processed_dir / 'test')
    shutil.copytree(val_dir, processed_dir / 'val')
    shutil.copytree(test_dir, processed_dir / 'test')
    shutil.copy(val_pt, processed_dir)
    shutil.copy(test_pt, processed_dir)

    # Write error report
    error_str = ""
    for k, v in failed.items():
        error_str += f"{'Split':<15}:  {k[0]}\n"
        error_str += f"{'Ligand W':<15}:  {k[1]}\n"
        error_str += f"{'Ligand L':<15}:  {k[2]}\n"
        error_str += f"{'Pocket':<15}:  {k[3]}\n"
        error_str += f"{'Error type':<15}:  {v[0]}\n"
        error_str += f"{'Error msg':<15}:  {v[1]}\n\n"

    with open(Path(processed_dir, 'errors.txt'), 'w') as f:
        f.write(error_str)

    with open(Path(processed_dir, 'dataset_config.txt'), 'w') as f:
        f.write(str(args))

if __name__ == '__main__':
    main()