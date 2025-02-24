import io
import random
import warnings
import torch
import webdataset as wds

from pathlib import Path
from torch.utils.data import Dataset

from src.data.data_utils import TensorDict, collate_entity
from src.constants import WEBDATASET_SHARD_SIZE, WEBDATASET_VAL_SIZE


class ProcessedLigandPocketDataset(Dataset):
    def __init__(self, pt_path, ligand_transform=None, pocket_transform=None,
                 catch_errors=False):

        self.ligand_transform = ligand_transform
        self.pocket_transform = pocket_transform
        self.catch_errors = catch_errors
        self.pt_path = pt_path

        self.data = torch.load(pt_path)

        # add number of nodes for convenience
        for entity in ['ligands', 'pockets']:
            self.data[entity]['size'] = torch.tensor([len(x) for x in self.data[entity]['x']])
            self.data[entity]['n_bonds'] = torch.tensor([len(x) for x in self.data[entity]['bond_one_hot']])

    def __len__(self):
        return len(self.data['ligands']['name'])

    def __getitem__(self, idx):
        data = {}
        data['ligand'] = {key: val[idx] for key, val in self.data['ligands'].items()}
        data['pocket'] = {key: val[idx] for key, val in self.data['pockets'].items()}
        try:
            if self.ligand_transform is not None:
                data['ligand'] = self.ligand_transform(data['ligand'])
            if self.pocket_transform is not None:
                data['pocket'] = self.pocket_transform(data['pocket'])
        except (RuntimeError, ValueError) as e:
            if self.catch_errors:
                warnings.warn(f"{type(e).__name__}('{e}') in data transform. "
                              f"Returning random item instead")
                # replace bad item with a random one
                rand_idx = random.randint(0, len(self) - 1)
                return self[rand_idx]
            else:
                raise e
        return data

    @staticmethod
    def collate_fn(batch_pairs, ligand_transform=None):

        out = {}
        for entity in ['ligand', 'pocket']:
            batch = [x[entity] for x in batch_pairs]

            if entity == 'ligand' and ligand_transform is not None:
                max_size = max(x['size'].item() for x in batch)
                # TODO: might have to remove elements from batch if processing fails, warn user in that case
                batch = [ligand_transform(x, max_size=max_size) for x in batch]

            out[entity] = TensorDict(**collate_entity(batch))

        return out


class ClusteredDataset(ProcessedLigandPocketDataset):
    def __init__(self, pt_path, ligand_transform=None, pocket_transform=None,
                 catch_errors=False):
        super().__init__(pt_path, ligand_transform, pocket_transform, catch_errors)
        self.clusters = list(self.data['clusters'].values())

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, cidx):
        cluster_inds = self.clusters[cidx]
        # idx = cluster_inds[random.randint(0, len(cluster_inds) - 1)]
        idx = random.choice(cluster_inds)
        return super().__getitem__(idx)

class DPODataset(ProcessedLigandPocketDataset):
    def __init__(self, pt_path, ligand_transform=None, pocket_transform=None,
                 catch_errors=False):
        self.ligand_transform = ligand_transform
        self.pocket_transform = pocket_transform
        self.catch_errors = catch_errors
        self.pt_path = pt_path

        self.data = torch.load(pt_path)

        if not 'pockets' in self.data:
            self.data['pockets'] = self.data['pockets_w']
        if not 'ligands' in self.data:
            self.data['ligands'] = self.data['ligands_w']

        if (
            len(self.data["ligands"]["name"])
            != len(self.data["ligands_l"]["name"])
            != len(self.data["pockets"]["name"])
        ):
            raise ValueError(
                "Error while importing DPO Dataset: Number of ligands winning, ligands losing and pockets must be the same"
            )

        # add number of nodes for convenience
        for entity in ['ligands', 'ligands_l', 'pockets']:
            self.data[entity]['size'] = torch.tensor([len(x) for x in self.data[entity]['x']])
            self.data[entity]['n_bonds'] = torch.tensor([len(x) for x in self.data[entity]['bond_one_hot']])

    def __len__(self):
        return len(self.data["ligands"]["name"])

    def __getitem__(self, idx):
        data = {}
        data['ligand'] = {key: val[idx] for key, val in self.data['ligands'].items()}
        data['ligand_l'] = {key: val[idx] for key, val in self.data['ligands_l'].items()}
        data['pocket'] = {key: val[idx] for key, val in self.data['pockets'].items()}
        try:
            if self.ligand_transform is not None:
                data['ligand'] = self.ligand_transform(data['ligand'])
                data['ligand_l'] = self.ligand_transform(data['ligand_l'])
            if self.pocket_transform is not None:
                data['pocket'] = self.pocket_transform(data['pocket'])
        except (RuntimeError, ValueError) as e:
            if self.catch_errors:
                warnings.warn(f"{type(e).__name__}('{e}') in data transform. "
                              f"Returning random item instead")
                # replace bad item with a random one
                rand_idx = random.randint(0, len(self) - 1)
                return self[rand_idx]
            else:
                raise e
        return data
    
    @staticmethod
    def collate_fn(batch_pairs, ligand_transform=None):

        out = {}
        for entity in ['ligand', 'ligand_l', 'pocket']:
            batch = [x[entity] for x in batch_pairs]

            if entity in ['ligand', 'ligand_l'] and ligand_transform is not None:
                max_size = max(x['size'].item() for x in batch)
                batch = [ligand_transform(x, max_size=max_size) for x in batch]

            out[entity] = TensorDict(**collate_entity(batch))

        return out

##########################################
############### WebDatasets ##############
##########################################

class ProteinLigandWebDataset(wds.WebDataset):
    @staticmethod
    def collate_fn(batch_pairs, ligand_transform=None):
        return ProcessedLigandPocketDataset.collate_fn(batch_pairs, ligand_transform)


def wds_decoder(key, value):
    return torch.load(io.BytesIO(value))


def preprocess_wds_item(data):
    out = {}
    for entity in ['ligand', 'pocket']:
        out[entity] = data['pt'][entity]
        for attr in ['size', 'n_bonds']:
            if torch.is_tensor(out[entity][attr]):
                assert len(out[entity][attr]) == 0
                out[entity][attr] = 0

    return out


def get_wds(data_path, stage, ligand_transform=None, pocket_transform=None):
    current_data_dir = Path(data_path, stage)
    shards = sorted(current_data_dir.glob('shard-?????.tar'), key=lambda s: int(s.name.split('-')[-1].split('.')[0]))
    min_shard = min(shards).name.split('-')[-1].split('.')[0]
    max_shard = max(shards).name.split('-')[-1].split('.')[0]
    total_size = (int(max_shard) - int(min_shard) + 1) * WEBDATASET_SHARD_SIZE if stage == 'train' else WEBDATASET_VAL_SIZE

    url = f'{data_path}/{stage}/shard-{{{min_shard}..{max_shard}}}.tar'
    ligand_transform_wrapper = lambda _data: _data
    pocket_transform_wrapper = lambda _data: _data

    if ligand_transform is not None:
        def ligand_transform_wrapper(_data):
            _data['pt']['ligand'] = ligand_transform(_data['pt']['ligand'])
            return _data
        
    if pocket_transform is not None:
        def pocket_transform_wrapper(_data):
            _data['pt']['pocket'] = pocket_transform(_data['pt']['pocket'])
            return _data

    return (
        ProteinLigandWebDataset(url, nodesplitter=wds.split_by_node)
        .decode(wds_decoder)
        .map(ligand_transform_wrapper)
        .map(pocket_transform_wrapper)
        .map(preprocess_wds_item)
        .with_length(total_size)
    )
