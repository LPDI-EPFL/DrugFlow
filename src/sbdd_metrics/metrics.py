import multiprocessing
import subprocess
import tempfile
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, Collection, Set, Optional
import signal
import numpy as np
import pandas as pd
from unittest.mock import patch
from scipy.spatial.distance import jensenshannon
from fcd import get_fcd
from posebusters import PoseBusters
from posebusters.modules.distance_geometry import _get_bond_atom_indices, _get_angle_atom_indices
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, KekulizeException, AtomKekulizeException
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from useful_rdkit_utils import REOS, RingSystemLookup, get_min_ring_frequency, RingSystemFinder

from .interactions import INTERACTION_LIST, prepare_ligand, read_protein, profile_detailed
from .sascorer import calculateScore

def timeout_handler(signum, frame):
    raise TimeoutError('Timeout')

BOND_SYMBOLS = {
    Chem.rdchem.BondType.SINGLE: '-',
    Chem.rdchem.BondType.DOUBLE: '=',
    Chem.rdchem.BondType.TRIPLE: '#',
    Chem.rdchem.BondType.AROMATIC: ':',
}


def is_nan(value):
    return value is None or pd.isna(value) or np.isnan(value)


def safe_run(func, timeout, **kwargs):
    def _run(f, q, **kwargs):
        r = f(**kwargs)
        q.put(r)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_run, kwargs={'f': func, 'q': queue, **kwargs})
    process.start()
    process.join(timeout)
    if process.is_alive():
        print(f"Function {func} didn't finish in {timeout} seconds. Terminating it.")
        process.terminate()
        process.join()
        return None
    elif not queue.empty():
        return queue.get()
    return None


class AbstractEvaluator:
    ID = None
    def __call__(self, molecule: Union[str, Path, Chem.Mol], protein: Union[str, Path] = None,
                 timeout=350):
        """
        Args:
            molecule (Union[str, Path, Chem.Mol]): input molecule
            protein (str): target protein
        
        Returns:
            metrics (dict): dictionary of metrics
        """
        RDLogger.DisableLog('rdApp.*')
        self.check_format(molecule, protein)

        # timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(timeout)
            results = self.evaluate(molecule, protein)
        except TimeoutError:
            print(f'Error when evaluating [{self.ID}]: Timeout after {timeout} seconds')
            signal.alarm(0)
            return {}
        except Exception as e:
            print(f'Error when evaluating [{self.ID}]: {e}')
            signal.alarm(0)
            return {}
        finally:
            signal.alarm(0)
        return self.add_id(results)

    def add_id(self, results):
        if self.ID is not None:
            return {f'{self.ID}.{key}': value for key, value in results.items()}
        else:
            return results

    @abstractmethod
    def evaluate(self, molecule: Union[str, Path, Chem.Mol], protein: Union[str, Path]) -> Dict[str, Union[int, float, str]]:
        raise NotImplementedError
    
    @staticmethod
    def check_format(molecule, protein):
        assert isinstance(molecule, (str, Path, Chem.Mol)), 'Supported molecule types: str, Path, Chem.Mol'
        assert protein is None or isinstance(protein, (str, Path)), 'Supported protein types: str'
        if isinstance(molecule, (str, Path)):
            supp = Chem.SDMolSupplier(str(molecule), sanitize=False)
            assert len(supp) == 1, 'Only one molecule per file is supported'

    @staticmethod
    def load_molecule(molecule):
        if isinstance(molecule, (str, Path)):
            return Chem.SDMolSupplier(str(molecule), sanitize=False)[0]
        return Chem.Mol(molecule)  # create copy to avoid overriding properties of the input molecule
    
    @staticmethod
    def save_molecule(molecule, sdf_path):
        if isinstance(molecule, (str, Path)):
            return molecule
        
        with Chem.SDWriter(str(sdf_path)) as w:
            try:
                w.write(molecule)
            except (RuntimeError, ValueError) as e:
                if isinstance(e, (KekulizeException, AtomKekulizeException)):
                    w.SetKekulize(False)
                    w.write(molecule)
                    w.SetKekulize(True)
                else:
                    w.write(Chem.Mol())
                    print('[AbstractEvaluator] Error when saving the molecule')
        
        return sdf_path
    
    @property
    def dtypes(self):
        return self.add_id(self._dtypes)
    
    @property
    @abstractmethod
    def _dtypes(self):
        raise NotImplementedError


class RepresentationEvaluator(AbstractEvaluator):
    ID = 'representation'

    def evaluate(self, molecule, protein=None):
        molecule = self.load_molecule(molecule)
        try:
            smiles = Chem.MolToSmiles(molecule)
        except:
            smiles = None

        return {'smiles': smiles}
    
    @property
    def _dtypes(self):
        return {'smiles': str}


class MolPropertyEvaluator(AbstractEvaluator):
    ID = 'mol_props'

    def evaluate(self, molecule, protein=None):
        molecule = self.load_molecule(molecule)
        return {k: v for k, v in molecule.GetPropsAsDict().items() if isinstance(v, float)}
    
    @property
    def _dtypes(self):
        return {'*': float}


class PoseBustersEvaluator(AbstractEvaluator):
    ID = 'posebusters'
    def __init__(self, pb_conf: str = 'dock'):
        self.posebusters = PoseBusters(config=pb_conf)

    @patch('rdkit.RDLogger.EnableLog', lambda x: None)
    @patch('rdkit.RDLogger.DisableLog', lambda x: None)
    def evaluate(self, molecule, protein=None):
        result = safe_run(self.posebusters.bust, timeout=20, mol_pred=molecule, mol_cond=protein)
        if result is None:
            return dict()
        
        with pd.option_context("future.no_silent_downcasting", True):
            result = dict(result.fillna(False).iloc[0])
        result['all'] = all([bool(value) if not is_nan(value) else False for value in result.values()])
        return result
    
    @property
    def _dtypes(self):
        return {'*': bool}
    

class GeometryEvaluator(AbstractEvaluator):
    ID = 'geometry'

    def evaluate(self, molecule, protein=None):
        mol = self.load_molecule(molecule)
        data = self.get_distances_and_angles(mol)
        return data

    @staticmethod
    def angle_repr(mol, triplet):
        i = mol.GetAtomWithIdx(triplet[0]).GetSymbol()
        j = mol.GetAtomWithIdx(triplet[1]).GetSymbol()
        k = mol.GetAtomWithIdx(triplet[2]).GetSymbol()
        ij = BOND_SYMBOLS[mol.GetBondBetweenAtoms(triplet[0], triplet[1]).GetBondType()]
        jk = BOND_SYMBOLS[mol.GetBondBetweenAtoms(triplet[1], triplet[2]).GetBondType()]

        # Unified (sorted) representation
        if i < k:
            return f'{i}{ij}{j}{jk}{k}'
        elif i > j:
            return f'{k}{jk}{j}{ij}{i}'
        elif ij <= jk:
            return f'{i}{ij}{j}{jk}{k}'
        else:
            return f'{k}{jk}{j}{ij}{i}'
    
    @staticmethod
    def bond_repr(mol, pair):
        i = mol.GetAtomWithIdx(pair[0]).GetSymbol()
        j = mol.GetAtomWithIdx(pair[1]).GetSymbol()
        ij = BOND_SYMBOLS[mol.GetBondBetweenAtoms(pair[0], pair[1]).GetBondType()]
        # Unified (sorted) representation
        return f'{i}{ij}{j}' if i <= j else f'{j}{ij}{i}'

    @staticmethod
    def get_bond_distances(mol, bonds):
        i, j = np.array(bonds).T
        x = mol.GetConformer().GetPositions()
        xi = x[i]
        xj = x[j]
        bond_distances = np.linalg.norm(xi - xj, axis=1)
        return bond_distances

    @staticmethod
    def get_angle_values(mol, triplets):
        i, j, k = np.array(triplets).T
        x = mol.GetConformer().GetPositions()
        xi = x[i]
        xj = x[j]
        xk = x[k]
        vji = xi - xj
        vjk = xk - xj
        angles = np.arccos((vji * vjk).sum(axis=1) / (np.linalg.norm(vji, axis=1) * np.linalg.norm(vjk, axis=1)))
        return np.degrees(angles)

    @staticmethod
    def get_distances_and_angles(mol):
        data = defaultdict(list)
        bonds = _get_bond_atom_indices(mol)
        distances = GeometryEvaluator.get_bond_distances(mol, bonds)
        for b, d in zip(bonds, distances):
            data[GeometryEvaluator.bond_repr(mol, b)].append(d)

        triplets = _get_angle_atom_indices(bonds)
        angles = GeometryEvaluator.get_angle_values(mol, triplets)
        for t, a in zip(triplets, angles):
            data[GeometryEvaluator.angle_repr(mol, t)].append(a)

        return data
    
    @property
    def _dtypes(self):
        return {'*': list}
    

class EnergyEvaluator(AbstractEvaluator):
    ID = 'energy'

    def evaluate(self, molecule, protein=None):
        molecule = self.load_molecule(molecule)
        try:
            energy = self.get_energy(molecule)
        except:
            energy = None
        return {'energy': energy}
    
    @staticmethod
    def get_energy(mol, conf_id=-1):
        mol = Chem.AddHs(mol, addCoords=True)
        uff = UFFGetMoleculeForceField(mol, confId=conf_id)
        e_uff = uff.CalcEnergy()
        return e_uff
    
    @property
    def _dtypes(self):
        return {'energy': float}
    

class InteractionsEvaluator(AbstractEvaluator):
    ID = 'interactions'

    def __init__(self, reduce='./reduce'):
        self.reduce = reduce

    @property
    def default_profile(self):
        return {i: 0 for i in INTERACTION_LIST}

    def evaluate(self, molecule, protein=None):
        molecule = self.load_molecule(molecule)
        profile = self.default_profile
        try:
            ligand_plf = prepare_ligand(molecule)
            protein_plf = read_protein(str(protein), reduce_exec=self.reduce)
            interactions = profile_detailed(ligand_plf, protein_plf)
            if not interactions.empty:
                profile.update(dict(interactions.interaction.value_counts()))
        except Exception:
            pass
        return profile
    
    @property
    def _dtypes(self):
        return {'*': int}


class GninaEvalulator(AbstractEvaluator):
    ID = 'gnina'
    def __init__(self, gnina):
        self.gnina = gnina

    def evaluate(self, molecule, protein=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            molecule = self.save_molecule(molecule, sdf_path=Path(tmpdir, 'molecule.sdf'))
            gnina_cmd = f'{self.gnina} -r {str(protein)} -l {str(molecule)} --minimize --seed 42 --no_gpu'
            gnina_result = subprocess.run(gnina_cmd, shell=True, capture_output=True, text=True)
            n_atoms = self.load_molecule(molecule).GetNumAtoms()

        gnina_scores = self.read_gnina_results(gnina_result)

        # Additionally computing ligand efficiency
        gnina_scores['vina_efficiency'] = gnina_scores['vina_score'] / n_atoms if n_atoms > 0 else None
        gnina_scores['gnina_efficiency'] = gnina_scores['gnina_score'] / n_atoms if n_atoms > 0 else None
        return gnina_scores
    
    @staticmethod
    def read_gnina_results(gnina_result):
        res = {
            'vina_score': None,
            'gnina_score': None,
            'minimisation_rmsd': None,
            'cnn_score': None,
        }
        if gnina_result.returncode != 0:
            print(gnina_result.stderr)
            return res

        for line in gnina_result.stdout.split('\n'):
            if line.startswith('Affinity'):
                res['vina_score'] = float(line.split(' ')[1].strip())
            if line.startswith('CNNaffinity'):
                res['gnina_score'] = float(line.split(' ')[1].strip())
            if line.startswith('CNNscore'):
                res['cnn_score'] = float(line.split(' ')[1].strip())
            if line.startswith('RMSD'):
                res['minimisation_rmsd'] = float(line.split(' ')[1].strip())

        return res
    
    @property
    def _dtypes(self):
        return {'*': float}


class MedChemEvaluator(AbstractEvaluator):
    ID = 'medchem'
    def __init__(self, connectivity_threshold=1.0):
        self.connectivity_threshold = connectivity_threshold

    def evaluate(self, molecule, protein=None):
        molecule = self.load_molecule(molecule)
        valid = self.is_valid(molecule)

        if valid:
            Chem.SanitizeMol(molecule)

        connected = None if not valid else self.is_connected(molecule)
        qed = None if not valid else self.calculate_qed(molecule)
        sa = None if not valid else self.calculate_sa(molecule)
        logp = None if not valid else self.calculate_logp(molecule)
        lipinski = None if not valid else self.calculate_lipinski(molecule)
        n_rotatable_bonds = None if not valid else self.calculate_rotatable_bonds(molecule)
        size = self.calculate_molecule_size(molecule)

        return {
            'valid': valid,
            'connected': connected,
            'qed': qed,
            'sa': sa,
            'logp': logp,
            'lipinski': lipinski,
            'size': size,
            'n_rotatable_bonds': n_rotatable_bonds,
        }

    @staticmethod
    def is_valid(rdmol):
        if rdmol.GetNumAtoms() < 1:
            return False

        _mol = Chem.Mol(rdmol)
        try:
            Chem.SanitizeMol(_mol)
        except ValueError:
            return False

        return True

    def is_connected(self, rdmol):
        if rdmol.GetNumAtoms() < 1:
            return False

        try:
            mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True)
            largest_frag = max(mol_frags, default=rdmol, key=lambda m: m.GetNumAtoms())
            return largest_frag.GetNumAtoms() / rdmol.GetNumAtoms() >= self.connectivity_threshold
        except:
            return False
    
    @staticmethod
    def calculate_qed(rdmol):
        try:
            return QED.qed(rdmol)
        except:
            return None

    @staticmethod
    def calculate_sa(rdmol):
        try:
            sa = calculateScore(rdmol)
            return sa
        except:
            return None

    @staticmethod
    def calculate_logp(rdmol):
        try:
            return Crippen.MolLogP(rdmol)
        except:
            return None

    @staticmethod
    def calculate_lipinski(rdmol):
        try:
            rule_1 = Descriptors.ExactMolWt(rdmol) < 500
            rule_2 = Lipinski.NumHDonors(rdmol) <= 5
            rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
            rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
            rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
            return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
        except:
            return None
        
    @staticmethod
    def calculate_molecule_size(rdmol):
        try:
            return rdmol.GetNumAtoms()
        except:
            return None
        
    @staticmethod
    def calculate_rotatable_bonds(rdmol):
        try:
            return Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol)
        except:
            return None
        
    @property
    def _dtypes(self):
        return {
            'valid': bool,
            'connected': bool,
            'qed': float,
            'sa': float,
            'logp': float,
            'lipinski': int,
            'size': int,
            'n_rotatable_bonds': int,
        }


class ClashEvaluator(AbstractEvaluator):
    ID = 'clashes'
    def __init__(self, margin=0.75, ignore={'H'}):
        self.margin = margin
        self.ignore = ignore

    def evaluate(self, molecule=None, protein=None):
        result = {
            'passed_clash_score_ligands': None,
            'passed_clash_score_pockets': None,
            'passed_clash_score_between': None,
        }
        if molecule is not None:
            molecule = self.load_molecule(molecule)
            clash_score = self.clash_score(molecule)
            result['clash_score_ligands'] = clash_score
            result['passed_clash_score_ligands'] = (clash_score == 0)

        if protein is not None:
            protein = Chem.MolFromPDBFile(str(protein), sanitize=False)
            clash_score = self.clash_score(protein)
            result['clash_score_pockets'] = clash_score
            result['passed_clash_score_pockets'] = (clash_score == 0)
        
        if molecule is not None and protein is not None:
            clash_score = self.clash_score(molecule, protein)
            result['clash_score_between'] = clash_score
            result['passed_clash_score_between'] = (clash_score == 0)
        
        return result
    
    def clash_score(self, rdmol1, rdmol2=None):
        """
        Computes a clash score as the number of atoms that have at least one
        clash divided by the number of atoms in the molecule.

        INTERMOLECULAR CLASH SCORE
        If rdmol2 is provided, the score is the percentage of atoms in rdmol1
        that have at least one clash with rdmol2.
        We define a clash if two atoms are closer than "margin times the sum of
        their van der Waals radii".

        INTRAMOLECULAR CLASH SCORE
        If rdmol2 is not provided, the score is the percentage of atoms in rdmol1
        that have at least one clash with other atoms in rdmol1.
        In this case, a clash is defined by margin times the atoms' smallest
        covalent radii (among single, double and triple bond radii). This is done
        so that this function is applicable even if no connectivity information is
        available.
        """

        intramolecular = rdmol2 is None
        if intramolecular:
            rdmol2 = rdmol1

        coord1, radii1 = self.coord_and_radii(rdmol1, intramolecular=intramolecular)
        coord2, radii2 = self.coord_and_radii(rdmol2, intramolecular=intramolecular)

        dist = np.sqrt(np.sum((coord1[:, None, :] - coord2[None, :, :]) ** 2, axis=-1))
        if intramolecular:
            np.fill_diagonal(dist, np.inf)

        clashes = dist < self.margin * (radii1[:, None] + radii2[None, :])
        clashes = np.any(clashes, axis=1)
        return np.mean(clashes)
    
    def coord_and_radii(self, rdmol, intramolecular):
        _periodic_table = Chem.GetPeriodicTable()
        _get_radius = _periodic_table.GetRcovalent if intramolecular else _periodic_table.GetRvdw

        coord = rdmol.GetConformer().GetPositions()
        radii = np.array([_get_radius(a.GetSymbol()) for a in rdmol.GetAtoms()])

        mask = np.array([a.GetSymbol() not in self.ignore for a in rdmol.GetAtoms()])
        coord = coord[mask]
        radii = radii[mask]

        assert coord.shape[0] == radii.shape[0]
        return coord, radii
    
    @property
    def _dtypes(self):
        return {
            'clash_score_ligands': float,
            'clash_score_pockets': float,
            'clash_score_between': float,
            'passed_clash_score_ligands': bool,
            'passed_clash_score_pockets': bool,
            'passed_clash_score_between': bool,
        }


class RingCountEvaluator(AbstractEvaluator):
    ID = 'ring_count'

    def evaluate(self, molecule, protein=None):
        _mol = self.load_molecule(molecule)

        # compute ring info if not yet available
        try:
            _mol.UpdatePropertyCache()
        except ValueError:
            return {}
        Chem.GetSymmSSSR(_mol)

        rings = _mol.GetRingInfo().AtomRings()
        ring_sizes = [len(r) for r in rings]

        ring_counts = defaultdict(int)
        for k in ring_sizes:
            ring_counts[f"num_{k}_rings"] += 1

        return ring_counts
    
    @property
    def _dtypes(self):
        return {'*': int}


class ChemblRingEvaluator(AbstractEvaluator):
    ID = 'chembl_ring_systems'

    def __init__(self):
        self.ring_system_lookup = RingSystemLookup.default()  # ChEMBL

    def evaluate(self, molecule, protein=None):

        results = {
            'min_ring_smi': None,
            'min_ring_freq_gt0_': None,
            'min_ring_freq_gt10_': None,
            'min_ring_freq_gt100_': None,
        }

        molecule = self.load_molecule(molecule)

        try:
            Chem.SanitizeMol(molecule)
            freq_list = self.ring_system_lookup.process_mol(molecule)
            freq_list = self.ring_system_lookup.process_mol(molecule)
        except ValueError:
            return results

        min_ring, min_freq = get_min_ring_frequency(freq_list)

        return {
            'min_ring_smi': min_ring,
            'min_ring_freq_gt0_': min_freq > 0,
            'min_ring_freq_gt10_': min_freq > 10,
            'min_ring_freq_gt100_': min_freq > 100,
        }
    
    @property
    def _dtypes(self):
        return {
            'min_ring_smi': str,
            'min_ring_freq_gt0_': bool,
            'min_ring_freq_gt10_': bool,
            'min_ring_freq_gt100_': bool,
        }
    

class REOSEvaluator(AbstractEvaluator):
    # Based on https://practicalcheminformatics.blogspot.com/2024/05/generative-molecular-design-isnt-as.html
    ID = 'reos'

    def __init__(self):
        self.reos = REOS()

    def evaluate(self, molecule, protein=None):
        
        molecule = self.load_molecule(molecule)
        try:
            Chem.SanitizeMol(molecule)
        except ValueError:
            return {rule_set: False for rule_set in self.reos.get_available_rule_sets()}
        
        results = {}
        for rule_set in self.reos.get_available_rule_sets():
            self.reos.set_active_rule_sets([rule_set])
            if rule_set == 'PW':
                self.reos.drop_rule('furans')

            reos_res = self.reos.process_mol(molecule)
            results[rule_set] = reos_res[0] == 'ok'

        results['all'] = all([bool(value) if not is_nan(value) else False for value in results.values()])
        return results
    
    @property
    def _dtypes(self):
        return {'*': bool}


class FullEvaluator(AbstractEvaluator):
    def __init__(
            self,
            pb_conf: str = 'dock',
            gnina: Optional[Union[Path, str]] = None, 
            reduce: Optional[Union[Path, str]] = None,
            connectivity_threshold: float = 1.0, 
            margin: float = 0.75, 
            ignore: Set[str] = {'H'},
            exclude_evaluators: Collection[str] = [],
    ):
        all_evaluators = [
            RepresentationEvaluator(),
            MolPropertyEvaluator(),
            PoseBustersEvaluator(pb_conf=pb_conf),
            MedChemEvaluator(connectivity_threshold=connectivity_threshold),
            ClashEvaluator(margin=margin, ignore=ignore),
            GeometryEvaluator(),
            RingCountEvaluator(),
            EnergyEvaluator(),
            ChemblRingEvaluator(),
            REOSEvaluator()
        ]
        if gnina is not None:
            all_evaluators.append(GninaEvalulator(gnina=gnina))
        else:
            print(f'Evaluator [{GninaEvalulator.ID}] is not included')
        if reduce is not None:
            all_evaluators.append(InteractionsEvaluator(reduce=reduce))
        else:
            print(f'Evaluator [{InteractionsEvaluator.ID}] is not included')

        self.evaluators = []
        for e in all_evaluators:
            if e.ID in exclude_evaluators:
                print(f'Excluded Evaluator [{e.ID}]')
            else:
                self.evaluators.append(e)

        print('Will use the follwoing evaluators:')
        for e in self.evaluators:
            print(f'- [{e.ID}]')


    def evaluate(self, molecule, protein):
        results = {}
        for evaluator in self.evaluators:
            results.update(evaluator(molecule, protein))
        return results
    
    @property
    def _dtypes(self):
        all_dtypes = {}
        for evaluator in self.evaluators:
            all_dtypes.update(evaluator.dtypes)
        return all_dtypes


########################################################################################
################################# Collection Metrics ###################################
########################################################################################


class AbstractCollectionEvaluator:
    ID = None
    def __call__(self, smiles: Collection[str], timeout=300):
        """
        Args:
            smiles (Collection[smiles]): input list of SMILES
        
        Returns:
            metrics (dict): dictionary of metrics
        """
        if self.ID is not None:
            print(f'Running CollectionEvaluator [{self.ID}]')

        RDLogger.DisableLog('rdApp.*')
        self.check_format(smiles)
        # timeout handler
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(timeout)
            results = self.evaluate(smiles)
        except TimeoutError:
            print(f'Error when evaluating [{self.ID}]: Timeout after {timeout} seconds')
            signal.alarm(0)
            return {}
        except Exception as e:
            print(f'Error when evaluating [{self.ID}]: {e}')
            signal.alarm(0)
            return {}
        finally:
            print(f'Finished CollectionEvaluator [{self.ID}]')
            signal.alarm(0)
        return results
    
    @staticmethod
    def check_format(smiles):
        assert len(smiles) > 0, 'List of input SMILES cannot be empty'
        assert isinstance(smiles, Collection), 'Only list of SMILES supported'
        assert isinstance(smiles[0], str), 'Only list of SMILES supported'
        
    
class UniquenessEvaluator(AbstractCollectionEvaluator):
    ID = 'uniqueness'
    def evaluate(self, smiles: Collection[str]):
        uniqueness = len(set(smiles)) / len(smiles)
        return {'uniqueness': uniqueness}
    

class NoveltyEvaluator(AbstractCollectionEvaluator):
    ID = 'novelty'
    def __init__(self, reference_smiles: Collection[str]):
        self.reference_smiles = set(list(reference_smiles))
        assert len(self.reference_smiles) > 0, 'List of refernce SMILES cannot be empty'

    def evaluate(self, smiles: Collection[str]):
        smiles = set(smiles)
        novel = [smi for smi in smiles if smi not in self.reference_smiles]
        novelty = len(novel) / len(smiles)
        return {'novelty': novelty}
    
def canonical_smiles(smiles):
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                yield Chem.MolToSmiles(mol)
        except:
            yield None

class FCDEvaluator(AbstractCollectionEvaluator):
    ID = 'fcd'
    def __init__(self, reference_smiles: Collection[str]):
        self.reference_smiles = list(reference_smiles)
        assert len(self.reference_smiles) > 0, 'List of refernce SMILES cannot be empty'

    def evaluate(self, smiles: Collection[str]):
        if len(smiles) > len(self.reference_smiles):
            print('Number of reference molecules should be greater than number of input molecules')
            return {'fcd': None}
        
        np.random.seed(42)
        reference_smiles = np.random.choice(self.reference_smiles, len(smiles), replace=False).tolist()
        reference_smiles_canonical = [w for w in canonical_smiles(reference_smiles) if w is not None]
        smiles_canonical = [w for w in canonical_smiles(smiles) if w is not None]
        fcd = get_fcd(reference_smiles_canonical, smiles_canonical)
        return {'fcd': fcd}


class RingDistributionEvaluator(AbstractCollectionEvaluator):
    ID = 'ring_system_distribution'

    def __init__(self, reference_smiles: Collection[str], jsd_on_k_most_freq: Collection[int] = ()):
        self.ring_system_finder = RingSystemFinder()
        self.ref_ring_dict = self.compute_ring_dict(reference_smiles)
        self.jsd_on_k_most_freq = jsd_on_k_most_freq

    def compute_ring_dict(self, molecules):

        ring_system_dict = defaultdict(int)

        for mol in tqdm(molecules, desc="Computing ring systems"):

            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)

            try:
                ring_system_list = self.ring_system_finder.find_ring_systems(mol, as_mols=True)
            except ValueError:
                print(f"WARNING[{type(self).__name__}]: error while computing ring systems; skipping molecule.")
                continue
            
            for ring in ring_system_list:
                inchi_key = Chem.MolToInchiKey(ring)
                ring_system_dict[inchi_key] += 1

        return ring_system_dict

    def precision(self, query_ring_dict):
        query_ring_systems = set(query_ring_dict.keys())
        ref_ring_systems = set(self.ref_ring_dict.keys())
        intersection = ref_ring_systems & query_ring_systems
        return len(intersection) / len(query_ring_systems) if len(query_ring_systems) > 0 else 0

    def recall(self, query_ring_dict):
        query_ring_systems = set(query_ring_dict.keys())
        ref_ring_systems = set(self.ref_ring_dict.keys())
        intersection = ref_ring_systems & query_ring_systems
        return len(intersection) / len(ref_ring_systems) if len(ref_ring_systems) > 0 else 0

    def jsd(self, query_ring_dict, k_most_freq=None):

        if k_most_freq is None:
            # example on the union of all ring systems
            sample_space = set(self.ref_ring_dict.keys()) | set(query_ring_dict.keys())
        else:
            # evaluate only on the k most common rings from the reference set
            sorted_rings = [k for k, v in sorted(self.ref_ring_dict.items(), key=lambda item: item[1], reverse=True)]
            sample_space = sorted_rings[:k_most_freq]

        p = np.zeros(len(sample_space))
        q = np.zeros(len(sample_space))

        for i, inchi_key in enumerate(sample_space):
            p[i] = self.ref_ring_dict.get(inchi_key, 0)
            q[i] = query_ring_dict.get(inchi_key, 0)

        # normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        return jensenshannon(p, q)

    def evaluate(self, smiles: Collection[str]):

        query_ring_dict = self.compute_ring_dict(smiles)

        out = {
            "precision": self.precision(query_ring_dict),
            "recall": self.recall(query_ring_dict),
            "jsd": self.jsd(query_ring_dict),
        }

        out.update(
            {f"jsd_{k}_most_freq": self.jsd(query_ring_dict, k_most_freq=k) for k in self.jsd_on_k_most_freq}
        )

        return out


class FullCollectionEvaluator(AbstractCollectionEvaluator):
    def __init__(self, reference_smiles: Collection[str], exclude_evaluators: Collection[str] = []):
        self.evaluators = [
            UniquenessEvaluator(),
            NoveltyEvaluator(reference_smiles=reference_smiles),
            FCDEvaluator(reference_smiles=reference_smiles),
            RingDistributionEvaluator(reference_smiles, jsd_on_k_most_freq=[10, 100, 1000, 10000]),
        ]
        for e in self.evaluators:
            if e.ID in exclude_evaluators:
                print(f'Excluding CollectionEvaluator [{e.ID}]')
                self.evaluators.remove(e)

    def evaluate(self, smiles):
        results = {}
        for evaluator in self.evaluators:
            results.update(evaluator(smiles))
        return results
