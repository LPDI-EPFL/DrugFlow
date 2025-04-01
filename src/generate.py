import argparse
import sys
import os
import warnings
import tempfile
import pandas as pd
import torch

from Bio.PDB import PDBParser
from pathlib import Path
from rdkit import Chem
from torch.utils.data import DataLoader
from functools import partial

basedir = Path(__file__).resolve().parent.parent
sys.path.append(str(basedir))
warnings.filterwarnings("ignore")

from src import utils
from src.data.dataset import ProcessedLigandPocketDataset
from src.data.data_utils import TensorDict, process_raw_pair
from src.model.lightning import DrugFlow
from src.sbdd_metrics.metrics import FullEvaluator

from tqdm import tqdm
from pdb import set_trace


def aggregate_metrics(table):
    agg_col = 'posebusters'
    total = 0
    table[agg_col] = 0
    for column in table.columns:
        if column.startswith(agg_col) and column != agg_col:
            table[agg_col] += table[column].fillna(0).astype(float)
            total += 1
    table[agg_col] = table[agg_col] / total

    agg_col = 'reos'
    total = 0
    table[agg_col] = 0
    for column in table.columns:
        if column.startswith(agg_col) and column != agg_col:
            table[agg_col] += table[column].fillna(0).astype(float)
            total += 1
    table[agg_col] = table[agg_col] / total

    agg_col = 'chembl_ring_systems'
    total = 0
    table[agg_col] = 0
    for column in table.columns:
        if column.startswith(agg_col) and column != agg_col and not column.endswith('smi'):
            table[agg_col] += table[column].fillna(0).astype(float)
            total += 1
    table[agg_col] = table[agg_col] / total
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--protein', type=str, required=True, help="Input PDB file.")
    p.add_argument('--ref_ligand', type=str, required=True, help="SDF file with reference ligand used to define the pocket.")
    p.add_argument('--checkpoint', type=str, required=True, help="Model checkpoint file.")
    p.add_argument('--molecule_size', type=str, required=False, default=None, help="Maximum number of atoms in the sampled molecules. Can be a single number or a range, e.g. '15,20'. If None, size will be sampled.")
    p.add_argument('--output', type=str, required=False, default='samples.sdf', help="Output file.")
    p.add_argument('--n_samples', type=int, required=False, default=10, help="Number of sampled molecules.")
    p.add_argument('--batch_size', type=int, required=False, default=32, help="Batch size.")
    p.add_argument('--pocket_distance_cutoff', type=float, required=False, default=8.0, help="Distance cutoff to define the pocket around the reference ligand.")
    p.add_argument('--n_steps', type=int, required=False, default=None, help="Number of denoising steps.")
    p.add_argument('--device', type=str, required=False, default='cuda:0', help="Device to use (mps, cuda:0).")
    p.add_argument('--datadir', type=Path, required=False, default=Path(basedir, 'src', 'default'), help="Needs to be specified to sample molecule sizes.")
    p.add_argument('--seed', type=int, required=False, default=42, help="Random seed.")
    p.add_argument('--filter', action='store_true', required=False, default=False, help="Apply basic filters and keep sampling until `n_samples` molecules passing these filters are found.")
    p.add_argument('--metrics_output', type=str, required=False, default=None, help="If provided, metrics will be computed and saved in csv format at this location.")
    p.add_argument('--gnina', type=str, required=False, default=None, help="Path to a gnina executable. Required for computing docking scores.")
    p.add_argument('--reduce', type=str, required=False, default=None, help="Path to a reduce executable. Required for computing interactions.")
    args = p.parse_args()

    utils.set_deterministic(seed=args.seed)
    utils.disable_rdkit_logging()

    if args.molecule_size is None and (args.datadir is None or not args.datadir.exists()):
        raise NotImplementedError(
            "Please provide a path to the processed dataset (using `--datadir`) "\
            "to infer the number of nodes. It contains the size distribution histogram."
        )
    
    if not args.filter:
        args.batch_size = min(args.batch_size, args.n_samples)
    
    # Select mps if available (Apple Silicon)
    if torch.backends.mps.is_available():
        args.device = 'mps'

    # Loading model
    chkpt_path = Path(args.checkpoint)
    chkpt_name = chkpt_path.parts[-1].split('.')[0]
    if torch.backends.mps.is_available():
        model = DrugFlow.load_from_checkpoint(args.checkpoint, map_location='cpu', strict=False) # load model to cpu
        model = model.to(args.device, dtype=torch.float32) # move model to mps with float32 type
    else:
        model = DrugFlow.load_from_checkpoint(args.checkpoint, map_location=args.device, strict=False)
    if args.datadir is not None:
        model.datadir = args.datadir

    model.setup(stage='generation')
    model.batch_size = model.eval_batch_size = args.batch_size
    model.eval().to(args.device)
    if args.n_steps is not None:
        model.T = args.n_steps

    # Loading size model
    size_model = None
    molecule_size = None
    molecule_size_boundaries = None
    if args.molecule_size is not None: 
        if args.molecule_size.isdigit():
            molecule_size = int(args.molecule_size)
            print(f'Will generate molecules of size {molecule_size}')
        else:
            boundaries = [x.strip() for x in args.molecule_size.split(',')]
            assert len(boundaries) == 2 and boundaries[0].isdigit() and boundaries[1].isdigit()
            left = int(boundaries[0])
            right = int(boundaries[1])
            molecule_size = f"uniform_{left}_{right}"
            print(f'Will generate molecules with numbers of atoms sampled from U({left}, {right})')

    # Preparing input
    pdb_model = PDBParser(QUIET=True).get_structure('', args.protein)[0]
    rdmol = Chem.SDMolSupplier(str(args.ref_ligand))[0]

    ligand, pocket = process_raw_pair(
        pdb_model, rdmol,
        dist_cutoff=args.pocket_distance_cutoff,
        pocket_representation=model.pocket_representation,
        compute_nerf_params=True,
        nma_input=args.protein if model.dynamics.add_nma_feat else None
    )
    ligand['name'] = 'ligand'
    dataset = [{'ligand': ligand, 'pocket': pocket} for _ in range(args.batch_size)]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size, 
        collate_fn=partial(ProcessedLigandPocketDataset.collate_fn, ligand_transform=None),
        pin_memory=True
    )

    # Start sampling
    smiles = set()
    sampled_molecules = []
    metrics = []
    Path(args.output).parent.absolute().mkdir(parents=True, exist_ok=True)
    print(f'Will generate {args.n_samples} samples')

    evaluator = FullEvaluator(gnina=args.gnina, reduce=args.reduce)

    with tqdm(total=args.n_samples) as pbar:
        while len(sampled_molecules) < args.n_samples:
            for i, data in enumerate(dataloader):
                new_data = {
                    'ligand': TensorDict(**data['ligand']).to(args.device),
                    'pocket': TensorDict(**data['pocket']).to(args.device),
                }
                rdmols, rdpockets, _ = model.sample(
                    new_data,
                    n_samples=1,
                    timesteps=args.n_steps,
                    num_nodes=molecule_size,
                )

                if args.filter or (args.metrics_output is not None):
                    results = []
                    with tempfile.TemporaryDirectory() as tmpdir:
                        for mol, receptor in zip(rdmols, rdpockets):
                            receptor_path = Path(tmpdir, 'receptor.pdb')
                            Chem.MolToPDBFile(receptor, str(receptor_path))
                            results.append(evaluator(mol, receptor_path))

                    table = pd.DataFrame(results)
                    table['novel'] = ~table['representation.smiles'].isin(smiles)
                    table = aggregate_metrics(table)
                    
                added_molecules = 0
                if args.filter:
                    table['passed_filters'] = (
                        (table['posebusters'] == 1) &
                        # (table['reos'] == 1) &
                        (table['chembl_ring_systems'] == 1) &
                        (table['novel'] == 1)
                    )
                    for i, (passed, smi) in enumerate(table[['passed_filters', 'representation.smiles']].values):
                        if passed:
                            sampled_molecules.append(rdmols[i])
                            smiles.add(smi)
                            added_molecules += 1

                    if args.metrics_output is not None:
                        metrics.append(table[table['passed_filters']])
                
                else:
                    sampled_molecules.extend(rdmols)
                    added_molecules = len(rdmols)
                    if args.metrics_output is not None:
                        metrics.append(table)

                pbar.update(added_molecules)

    # Write results
    utils.write_sdf_file(args.output, sampled_molecules)

    if args.metrics_output is not None:
        metrics = pd.concat(metrics)
        metrics.to_csv(args.metrics_output, index=False)
