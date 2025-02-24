import os
import sys
import re

from pathlib import Path
from typing import Collection, List, Dict, Type

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics import FullEvaluator, FullCollectionEvaluator

AUXILIARY_COLUMNS = ['sample', 'sdf_file', 'pdb_file', 'subdir']
VALIDITY_METRIC_NAME = 'medchem.valid'


def get_data_type(key: str, data_types: Dict[str, Type], default=float) -> Type:
    found_data_type_key = None
    found_data_type_value = None
    for data_type_key, data_type_value in data_types.items():
        if re.match(data_type_key, key) is not None:
            if found_data_type_key is not None:
                raise ValueError(f'Multiple data type keys match [{key}]: {found_data_type_key}, {data_type_key}')

            found_data_type_value = data_type_value
            found_data_type_key = data_type_key

    if found_data_type_key is None:
        if default is None:
            raise KeyError(key)
        else:
            found_data_type_value = default

    return found_data_type_value


def convert_data_to_table(data: List[Dict], data_types: Dict[str, Type]) -> pd.DataFrame:
    """
    Converts data from `evaluate_drugflow` to a detailed table
    """
    table = []
    for entry in data:
        table_entry = {}
        for key, value in entry.items():
            if key in AUXILIARY_COLUMNS:
                table_entry[key] = value
                continue
            if get_data_type(key, data_types) != list:
                table_entry[key] = value
        table.append(table_entry)

    return pd.DataFrame(table)

def aggregated_metrics(table: pd.DataFrame, data_types: Dict[str, Type], validity_metric_name: str = None):
    """
    Args:
        table (pd.DataFrame): table with metrics computed for each sample
        data_types (Dict[str, Type]): dictionary with data types for each column
        validity_metric_name (str): name of the column that has validity metric

    Returns:
        agg_table (pd.DataFrame): table with columns ['metric', 'value', 'std']
    """
    aggregated_results = []

    # If validity column name is provided:
    #    1. compute validity on the entire data
    #    2. drop all invalid molecules to compute the rest
    if validity_metric_name is not None:
        aggregated_results.append({
            'metric': validity_metric_name,
            'value': table[validity_metric_name].fillna(False).astype(float).mean(),
            'std': None,
        })
        table = table[table[validity_metric_name]]

    # Compute aggregated metrics + standard deviations where applicable
    for column in table.columns:
        if column in AUXILIARY_COLUMNS + [validity_metric_name] or get_data_type(column, data_types) == str:
            continue
        with pd.option_context("future.no_silent_downcasting", True):
            if get_data_type(column, data_types) == bool:
                values = table[column].fillna(0).values.astype(float).mean()
                std = None
            else:
                values = table[column].dropna().values.astype(float).mean()
                std = table[column].dropna().values.astype(float).std()

        aggregated_results.append({
            'metric': column,
            'value': values,
            'std': std,
        })

    agg_table = pd.DataFrame(aggregated_results)
    return agg_table


def collection_metrics(
        table: pd.DataFrame,
        reference_smiles: Collection[str],
        validity_metric_name: str = None,
        exclude_evaluators: Collection[str] = [],
):
    """
    Args:
        table (pd.DataFrame): table with metrics computed for each sample
        reference_smiles (Collection[str]): list of reference SMILES (e.g. training set)
        validity_metric_name (str): name of the column that has validity metric
        exclude_evaluators (Collection[str]): Evaluator IDs to exclude

    Returns:
        col_table (pd.DataFrame): table with columns ['metric', 'value']
    """

    # If validity column name is provided drop all invalid molecules
    if validity_metric_name is not None:
        table = table[table[validity_metric_name]]

    evaluator = FullCollectionEvaluator(reference_smiles, exclude_evaluators=exclude_evaluators)
    smiles = table['representation.smiles'].values
    if len(smiles) == 0:
        print('No valid input molecules')
        return pd.DataFrame(columns=['metric', 'value'])

    collection_metrics = evaluator(smiles)
    results = [
        {'metric': key, 'value': value}
        for key, value in collection_metrics.items()
    ]

    col_table = pd.DataFrame(results)
    return col_table


def evaluate_drugflow_subdir(
        in_dir: Path,
        evaluator: FullEvaluator,
        desc: str = None,
        n_samples: int = None,
) -> List[Dict]:
    """
    Computes per-molecule metrics for a single directory of samples for one target
    """
    results = []
    valid_files = [
        int(fname.split('_')[0])
        for fname in os.listdir(in_dir)
        if fname.endswith('_ligand.sdf') and not fname.startswith('.')
    ]
    if len(valid_files) == 0:
        return pd.DataFrame()

    upper_bound = max(valid_files) + 1
    if n_samples is not None:
        upper_bound = min(upper_bound, n_samples)

    for i in tqdm(range(upper_bound), desc=desc, file=sys.stdout):
        in_mol = Path(in_dir, f'{i}_ligand.sdf')
        in_prot = Path(in_dir, f'{i}_pocket.pdb')
        res = evaluator(in_mol, in_prot)

        res['sample'] = i
        res['sdf_file'] = str(in_mol)
        res['pdb_file'] = str(in_prot)
        results.append(res)

    return results


def evaluate_drugflow(
        in_dir: Path,
        evaluator: FullEvaluator,
        n_samples: int = None,
        job_id: int = 0,
        n_jobs: int = 1,
) -> List[Dict]:
    """
    1. Computes per-molecule metrics for all single directories of samples
    2. Aggregates these metrics
    3. Computes additional collection metrics (if `reference_smiles_path` is provided)
    """
    data = []
    total_number_of_subdirs = len([path for path in in_dir.glob("[!.]*") if os.path.isdir(path)])
    i = 0
    for subdir in in_dir.glob("[!.]*"):
        if not os.path.isdir(subdir):
            continue

        i += 1
        if (i - 1) % n_jobs != job_id:
            continue

        curr_data = evaluate_drugflow_subdir(
            in_dir=subdir,
            evaluator=evaluator,
            desc=f'[{i}/{total_number_of_subdirs}] {str(subdir.name)}',
            n_samples=n_samples,
        )
        for entry in curr_data:
            entry['subdir'] = str(subdir)
            data.append(entry)

    return data


def compute_all_metrics_drugflow(
        in_dir: Path,
        gnina_path: Path,
        reduce_path: Path = None,
        reference_smiles_path: Path = None,
        n_samples: int = None,
        validity_metric_name: str = VALIDITY_METRIC_NAME,
        exclude_evaluators: Collection[str] = [],
        job_id: int = 0,
        n_jobs: int = 1,
):
    evaluator = FullEvaluator(gnina=gnina_path, reduce=reduce_path, exclude_evaluators=exclude_evaluators)
    data = evaluate_drugflow(in_dir=in_dir, evaluator=evaluator, n_samples=n_samples, job_id=job_id, n_jobs=n_jobs)
    table_detailed = convert_data_to_table(data, evaluator.dtypes)
    table_aggregated = aggregated_metrics(
        table_detailed,
        data_types=evaluator.dtypes,
        validity_metric_name=validity_metric_name
    )

    # Add collection metrics (uniqueness, novelty, FCD, etc.) if reference smiles are provided
    if reference_smiles_path is not None:
        reference_smiles = np.load(reference_smiles_path)
        col_metrics = collection_metrics(
            table=table_detailed,
            reference_smiles=reference_smiles,
            validity_metric_name=validity_metric_name,
            exclude_evaluators=exclude_evaluators
        )
        table_aggregated = pd.concat([table_aggregated, col_metrics])

    return data, table_detailed, table_aggregated
