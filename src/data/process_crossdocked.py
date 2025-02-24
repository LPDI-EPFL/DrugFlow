from pathlib import Path
from time import time
import argparse
import shutil
import random
import yaml
from collections import defaultdict

import torch
from tqdm import tqdm
import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem

import sys
basedir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(basedir))

from src.data.data_utils import process_raw_pair, get_n_nodes, get_type_histogram
from src.data.data_utils import rdmol_to_smiles
from src.constants import atom_encoder, bond_encoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=Path)
    parser.add_argument('--outdir', type=Path, default=None)
    parser.add_argument('--split_path', type=Path, default=None)
    parser.add_argument('--pocket', type=str, default='CA+',
                        choices=['side_chain_bead', 'CA+'])
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--val_size', type=int, default=100)
    parser.add_argument('--normal_modes', action='store_true')
    parser.add_argument('--flex', action='store_true')
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    random.seed(args.random_seed)

    datadir = args.basedir / 'crossdocked_pocket10/'

    # Make output directory
    dirname = f"processed_crossdocked_{args.pocket}"
    if args.flex:
        dirname += '_flex'
    if args.normal_modes:
        dirname += '_nma'
    if args.toy:
        dirname += '_toy'
    processed_dir = Path(args.basedir, dirname) if args.outdir is None else args.outdir
    processed_dir.mkdir(parents=True)

    # Read data split
    split_path = Path(args.basedir, 'split_by_name.pt') if args.split_path is None else args.split_path
    data_split = torch.load(split_path)

    # If there is no validation set, copy training examples (the validation set
    # is not very important in this application)
    if 'val' not in data_split:
        random.shuffle(data_split['train'])
        data_split['val'] = data_split['train'][-args.val_size:]
        data_split['train'] = data_split['train'][:-args.val_size]

    if args.toy:
        data_split['train'] = random.sample(data_split['train'], 100)

    failed = {}
    train_smiles = []

    n_samples_after = {}
    for split in data_split.keys():

        print(f"Processing {split} dataset...")

        ligands = defaultdict(list)
        pockets = defaultdict(list)

        tic = time()
        pbar = tqdm(data_split[split])
        for pocket_fn, ligand_fn in pbar:

            pbar.set_description(f'#failed: {len(failed)}')

            sdffile = datadir / f'{ligand_fn}'
            pdbfile = datadir / f'{pocket_fn}'

            try:
                pdb_model = PDBParser(QUIET=True).get_structure('', pdbfile)[0]

                rdmol = Chem.SDMolSupplier(str(sdffile))[0]

                ligand, pocket = process_raw_pair(
                    pdb_model, rdmol, pocket_representation=args.pocket,
                    compute_nerf_params=args.flex, compute_bb_frames=args.flex,
                    nma_input=pdbfile if args.normal_modes else None)

            except (KeyError, AssertionError, FileNotFoundError, IndexError,
                    ValueError, AttributeError) as e:
                failed[(split, sdffile, pdbfile)] = (type(e).__name__, str(e))
                continue

            nerf_keys = ['fixed_coord', 'atom_mask', 'nerf_indices', 'length', 'theta', 'chi', 'ddihedral', 'chi_indices']
            for k in ['x', 'one_hot', 'bonds', 'bond_one_hot', 'v', 'nma_vec'] + nerf_keys + ['axis_angle']:
                if k in ligand:
                    ligands[k].append(ligand[k])
                if k in pocket:
                    pockets[k].append(pocket[k])

            pocket_file = pdbfile.name.replace('_', '-')
            ligand_file = Path(pocket_file).stem + '_' + Path(sdffile).name.replace('_', '-')
            ligands['name'].append(ligand_file)
            pockets['name'].append(pocket_file)
            train_smiles.append(rdmol_to_smiles(rdmol))

            if split in {'val', 'test'}:
                pdb_sdf_dir = processed_dir / split
                pdb_sdf_dir.mkdir(exist_ok=True)

                # Copy PDB file
                pdb_file_out = Path(pdb_sdf_dir, pocket_file)
                shutil.copy(pdbfile, pdb_file_out)

                # Copy SDF file
                sdf_file_out = Path(pdb_sdf_dir, ligand_file)
                shutil.copy(sdffile, sdf_file_out)

        data = {'ligands': ligands, 'pockets': pockets}
        torch.save(data, Path(processed_dir, f'{split}.pt'))

        if split == 'train':
            np.save(Path(processed_dir, 'train_smiles.npy'), train_smiles)

        print(f"Processing {split} set took {(time() - tic) / 60.0:.2f} minutes")


    # --------------------------------------------------------------------------
    # Compute statistics & additional information
    # --------------------------------------------------------------------------
    train_data = torch.load(Path(processed_dir, f'train.pt'))

    # Maximum molecule size
    max_ligand_size = max([len(x) for x in train_data['ligands']['x']])

    # Joint histogram of number of ligand and pocket nodes
    pocket_coords = train_data['pockets']['x']
    ligand_coords = train_data['ligands']['x']
    n_nodes = get_n_nodes(ligand_coords, pocket_coords)
    np.save(Path(processed_dir, 'size_distribution.npy'), n_nodes)

    # Get histograms of ligand node types
    lig_one_hot = [x.numpy() for x in train_data['ligands']['one_hot']]
    ligand_hist = get_type_histogram(lig_one_hot, atom_encoder)
    np.save(Path(processed_dir, 'ligand_type_histogram.npy'), ligand_hist)

    # Get histograms of ligand edge types
    lig_bond_one_hot = [x.numpy() for x in train_data['ligands']['bond_one_hot']]
    ligand_bond_hist = get_type_histogram(lig_bond_one_hot, bond_encoder)
    np.save(Path(processed_dir, 'ligand_bond_type_histogram.npy'), ligand_bond_hist)

    # Write error report
    error_str = ""
    for k, v in failed.items():
        error_str += f"{'Split':<15}:  {k[0]}\n"
        error_str += f"{'Ligand':<15}:  {k[1]}\n"
        error_str += f"{'Pocket':<15}:  {k[2]}\n"
        error_str += f"{'Error type':<15}:  {v[0]}\n"
        error_str += f"{'Error msg':<15}:  {v[1]}\n\n"

    with open(Path(processed_dir, 'errors.txt'), 'w') as f:
        f.write(error_str)

    metadata = {
        'max_ligand_size': max_ligand_size
    }
    with open(Path(processed_dir, 'metadata.yml'), 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
