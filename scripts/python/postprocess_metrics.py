import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

basedir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(basedir))

from src.data.data_utils import atom_encoder, bond_encoder, encode_atom
from src.sbdd_metrics.evaluation import VALIDITY_METRIC_NAME, aggregated_metrics, collection_metrics, get_data_type
from src.sbdd_metrics.metrics import FullEvaluator


DATA_TYPES = data_types = FullEvaluator().dtypes

MEDCHEM_PROPS = [
    'medchem.qed',
    'medchem.sa',
    'medchem.logp',
    'medchem.lipinski',
    'medchem.size',
    'medchem.n_rotatable_bonds',
    'energy.energy',
]

DOCKING_PROPS = [
    'gnina.vina_score',
    'gnina.gnina_score',
    'gnina.vina_efficiency',
    'gnina.gnina_efficiency',
]

RELEVANT_INTERACTIONS = [
    'interactions.HBAcceptor', 
    'interactions.HBDonor', 
    'interactions.HB', 
    'interactions.PiStacking', 
    'interactions.Hydrophobic',
    #
    'interactions.HBAcceptor.normalized', 
    'interactions.HBDonor.normalized', 
    'interactions.HB.normalized', 
    'interactions.PiStacking.normalized', 
    'interactions.Hydrophobic.normalized'
]


def compute_discrete_distributions(smiles, name):
    atom_counter = Counter()
    bond_counter = Counter()

    for smi in tqdm(smiles, desc=name):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.RemoveAllHs(mol, sanitize=False)
        for atom in mol.GetAtoms():
            try:
                encoded_atom = encode_atom(atom, atom_encoder=atom_encoder)
            except KeyError:
                continue
            atom_counter[encoded_atom] += 1
        for bond in mol.GetBonds():
            bond_counter[bond_encoder[str(bond.GetBondType())]] += 1

    atom_distribution = np.zeros(len(atom_encoder))
    bond_distribution = np.zeros(len(bond_encoder))

    for k, v in atom_counter.items():
        atom_distribution[k] = v
    for k, v in bond_counter.items():
        bond_distribution[k] = v

    atom_distribution = atom_distribution / atom_distribution.sum()
    bond_distribution = bond_distribution / bond_distribution.sum()

    return atom_distribution, bond_distribution


def flatten_distribution(data, name, table):
    aux = ['sample', 'sdf_file', 'pdb_file']
    method_distributions = defaultdict(list)

    sdf2sample2size = defaultdict(dict)
    for _, row in table.iterrows():
        sdf2sample2size[row['sdf_file']][int(row['sample'])] = row['medchem.size']

    for item in tqdm(data, desc=name):
        if item['medchem.valid'] is not True:
            continue

        if 'interactions.HBAcceptor' in item and 'interactions.HBDonor' in item:
            item['interactions.HB'] = item['interactions.HBAcceptor'] + item['interactions.HBDonor']
        
        new_entries = {}
        for key, value in item.items():
            if key.startswith('interactions'):
                size = sdf2sample2size.get(item['sdf_file'], dict()).get(int(item['sample']))
                if size is not None:
                    new_entries[key + '.normalized'] = value / size
        item.update(new_entries)
        
        for key, value in item.items():
            if value is None:
                continue
            if key in aux:
                continue
            if key == 'energy.energy' and abs(value) > 1000:
                continue
            
            if get_data_type(key, DATA_TYPES, default=type(value)) == list:
                method_distributions[key] += value
            else:
                method_distributions[key].append(value)
    
    return method_distributions


def prepare_baseline_data(root_path, baseline_name):
    metrics_detailed = pd.read_csv(f'{root_path}/metrics_detailed.csv')
    metrics_detailed = metrics_detailed[metrics_detailed['medchem.valid']]
    distributions = pickle.load(open(f'{root_path}/metrics_data.pkl', 'rb'))
    distributions = flatten_distribution(distributions, name=baseline_name, table=metrics_detailed)
    distributions['energy.energy'] = [v for v in distributions['energy.energy'] if -1000 <= v <= 1000]
    for prop in MEDCHEM_PROPS + DOCKING_PROPS:
        distributions[prop] = metrics_detailed[prop].dropna().values.tolist()

    smiles = metrics_detailed['representation.smiles']
    atom_distribution, bond_distribution = compute_discrete_distributions(smiles, name=baseline_name)
    discrete_distributions = {
        'atom_types': atom_distribution,
        'bond_types': bond_distribution,
    }

    return distributions, discrete_distributions


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', type=Path, required=True, help='Directory with samples')
    p.add_argument('--out_dir', type=str, required=True, help='Output directory')
    p.add_argument('--n_samples', type=int, required=False, default=None, help='N samples per target')
    p.add_argument('--reference_smiles', type=str, default=None, help='Path to the .npy file with reference SMILES (optional)')
    p.add_argument('--crossdocked_dir', type=str, required=False, default=None, help='Crossdocked data dir for computing distances between distributions')
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print('Combining data')
    data = []
    for file_path in tqdm(Path(args.in_dir).glob('metrics_data_*.pkl')):
        with open(file_path, 'rb') as f:
            d = pickle.load(f)
            if args.n_samples is not None:
                d = d[:args.n_samples]
            data += d
    with open(Path(args.out_dir, 'metrics_data.pkl'), 'wb') as f:
        pickle.dump(data, f)

    print('Combining detailed metrics')
    tables = []
    for file_path in tqdm(Path(args.in_dir).glob('metrics_detailed_*.csv')):
        table = pd.read_csv(file_path)
        if args.n_samples is not None:
            table = table.head(args.n_samples)
        tables.append(table)

    table_detailed = pd.concat(tables)
    table_detailed.to_csv(Path(args.out_dir, 'metrics_detailed.csv'), index=False)

    print('Computing aggregated metrics')
    evaluator = FullEvaluator(gnina='gnina', reduce='reduce')
    table_aggregated = aggregated_metrics(
        table_detailed,
        data_types=evaluator.dtypes,
        validity_metric_name=VALIDITY_METRIC_NAME
    )

    if args.reference_smiles is not None:
        reference_smiles = np.load(args.reference_smiles)
        col_metrics = collection_metrics(
            table=table_detailed,
            reference_smiles=reference_smiles,
            validity_metric_name=VALIDITY_METRIC_NAME,
            exclude_evaluators=[],
        )
        table_aggregated = pd.concat([table_aggregated, col_metrics])

    table_aggregated.to_csv(Path(args.out_dir, 'metrics_aggregated.csv'), index=False)

    # Computing distributions
    if args.crossdocked_dir is not None:

        # Loading training data distributions
        crossdocked_distributions = None
        crossdocked_discrete_distributions = None
        precomputed_distr_path = f'{args.crossdocked_dir}/crossdocked_distributions.pkl'
        precomputed_discrete_distr_path = f'{args.crossdocked_dir}/crossdocked_discrete_distributions.pkl'
        if os.path.exists(precomputed_distr_path) and os.path.exists(precomputed_discrete_distr_path):
            # Use precomputed distributions in case they exist
            with open(precomputed_distr_path, 'rb') as f:
                crossdocked_distributions = pickle.load(f)
            with open(precomputed_discrete_distr_path, 'rb') as f:
                crossdocked_discrete_distributions = pickle.load(f)
        else:
            assert os.path.exists(f'{args.crossdocked_dir}/metrics_detailed.csv')
            assert os.path.exists(f'{args.crossdocked_dir}/metrics_data.pkl')
            crossdocked_distributions, crossdocked_discrete_distributions = prepare_baseline_data(
                root_path=args.crossdocked_dir, 
                baseline_name='crossdocked'
            )
            # Save precomputed distributions for faster next runs
            with open(precomputed_distr_path, 'wb') as f:
                pickle.dump(crossdocked_distributions, f)
            with open(precomputed_discrete_distr_path, 'wb') as f:
                pickle.dump(crossdocked_discrete_distributions, f)

        # Selecting top-5 most frequent atom types, bond types, angles and torsions
        bonds = sorted([
            (k, len(v)) for k, v in crossdocked_distributions.items() 
            if k.startswith('geometry.') and sum(s.isalpha() for s in k.split('.')[1]) == 2
        ], key=lambda t: t[1], reverse=True)[:5]
        top_5_bonds = [t[0] for t in bonds]

        angles = sorted([
            (k, len(v)) for k, v in crossdocked_distributions.items() 
            if k.startswith('geometry.') and sum(s.isalpha() for s in k.split('.')[1]) == 3
        ], key=lambda t: t[1], reverse=True)[:5]
        top_5_angles = [t[0] for t in angles]

        # Loading distributions of samples
        distributions, discrete_distributions = prepare_baseline_data(args.out_dir, 'samples')

        # Computing distances between distributions
        distances = {'method': 'method',}
        relevant_columns = MEDCHEM_PROPS + DOCKING_PROPS + RELEVANT_INTERACTIONS + top_5_bonds + top_5_angles
        for metric in distributions.keys():
            if metric not in relevant_columns:
                continue

            ref = crossdocked_distributions.get(metric)
            # cur = distributions.get(metric)
            cur = [x for x in distributions.get(metric) if not pd.isna(x)]

            if ref is not None and cur is not None and len(cur) > 0:
                try:
                    distance = wasserstein_distance(ref, cur)
                except:
                    from pdb import set_trace; set_trace()
                num_ref = len(ref)
                num_cur = len(cur)
                distances[f'WD.{metric}'] = distance

        for metric in crossdocked_discrete_distributions.keys():
            ref = crossdocked_discrete_distributions.get(metric)
            cur = discrete_distributions.get(metric)
            if ref is not None and cur is not None:
                distance = jensenshannon(p=ref, q=cur)
                num_ref = len(ref)
                num_cur = len(cur)
                distances[f'JS.{metric}'] = distance

        dist_table = pd.DataFrame([distances])
        dist_table.to_csv(Path(args.out_dir, 'metrics_distances.csv'), index=False)