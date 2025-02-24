import subprocess

import numpy as np
import tempfile
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem import AtomKekulizeException, AtomValenceException, \
    KekulizeException, MolSanitizeException
from src.analysis.SA_Score.sascorer import calculateScore
from src.utils import write_sdf_file

from copy import deepcopy

from pdb import set_trace


class CategoricalDistribution:
    EPS = 1e-10

    def __init__(self, histogram_dict, mapping):
        histogram = np.zeros(len(mapping))
        for k, v in histogram_dict.items():
            histogram[mapping[k]] = v

        # Normalize histogram
        self.p = histogram / histogram.sum()
        self.mapping = deepcopy(mapping)

    def kl_divergence(self, other_sample):
        sample_histogram = np.zeros(len(self.mapping))
        for x in other_sample:
            # sample_histogram[self.mapping[x]] += 1
            sample_histogram[x] += 1

        # Normalize
        q = sample_histogram / sample_histogram.sum()

        return -np.sum(self.p * np.log(q / (self.p + self.EPS) + self.EPS))


def check_mol(rdmol):
    """
    See also: https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    """
    if rdmol is None:
        return 'is_none'

    _rdmol = Chem.Mol(rdmol)
    try:
        Chem.SanitizeMol(_rdmol)
        return 'valid'
    except ValueError as e:
        assert isinstance(e, MolSanitizeException)
        return type(e).__name__


def validity_analysis(rdmol_list):
    """
    For explanations, see: https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
    """

    result = {
        'AtomValenceException': 0,  # atoms in higher-than-allowed valence states
        'AtomKekulizeException': 0,
        'KekulizeException': 0,  # ring cannot be kekulized or aromatic bonds found outside of rings
        'other': 0,
        'valid': 0
    }

    for rdmol in rdmol_list:
        flag = check_mol(rdmol)

        try:
            result[flag] += 1
        except KeyError:
            result['other'] += 1

    assert sum(result.values()) == len(rdmol_list)

    return result


class MoleculeValidity:
    def __init__(self, connectivity_thresh=1.0):
        self.connectivity_thresh = connectivity_thresh

    def compute_validity(self, generated):
        """ generated: list of RDKit molecules. """
        if len(generated) < 1:
            return [], 0.0

        # Return copies of the valid molecules
        valid = [Chem.Mol(mol) for mol in generated if check_mol(mol) == 'valid']
        return valid, len(valid) / len(generated)

    def compute_connectivity(self, valid):
        """
        Consider molecule connected if its largest fragment contains at
        least <self.connectivity_thresh * 100>% of all atoms.
        :param valid: list of valid RDKit molecules
        """
        if len(valid) < 1:
            return [], 0.0

        for mol in valid:
            Chem.SanitizeMol(mol)  # all molecules should be valid

        connected = []
        for mol in valid:

            if mol.GetNumAtoms() < 1:
                continue

            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            except MolSanitizeException as e:
                print('Error while computing connectivity:', e)
                continue

            largest_frag = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            if largest_frag.GetNumAtoms() / mol.GetNumAtoms() >= self.connectivity_thresh:
                connected.append(largest_frag)

        return connected, len(connected) / len(valid)

    def __call__(self, rdmols, verbose=False):
        """
        :param rdmols: list of RDKit molecules
        """

        results = {}
        results['n_total'] = len(rdmols)

        valid, validity = self.compute_validity(rdmols)
        results['n_valid'] = len(valid)
        results['validity'] = validity

        connected, connectivity = self.compute_connectivity(valid)
        results['n_connected'] = len(connected)
        results['connectivity'] = connectivity
        results['valid_and_connected'] = results['n_connected'] / results['n_total']

        if verbose:
            print(f"Validity over {results['n_total']} molecules: {validity * 100 :.2f}%")
            print(f"Connectivity over {results['n_valid']} valid molecules: {connectivity * 100 :.2f}%")

        return results


class MolecularMetrics:
    def __init__(self, connectivity_thresh=1.0):
        self.connectivity_thresh = connectivity_thresh

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

        mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True)

        largest_frag = max(mol_frags, default=rdmol, key=lambda m: m.GetNumAtoms())
        if largest_frag.GetNumAtoms() / rdmol.GetNumAtoms() >= self.connectivity_thresh:
            return True
        else:
            return False

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        return sa

    @staticmethod
    def calculate_logp(rdmol):
        return Crippen.MolLogP(rdmol)

    @staticmethod
    def calculate_lipinski(rdmol):
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

    def __call__(self, rdmol):
        valid = self.is_valid(rdmol)

        if valid:
            Chem.SanitizeMol(rdmol)

        connected = None if not valid else self.is_connected(rdmol)
        qed = None if not valid else self.calculate_qed(rdmol)
        sa = None if not valid else self.calculate_sa(rdmol)
        logp = None if not valid else self.calculate_logp(rdmol)
        lipinski = None if not valid else self.calculate_lipinski(rdmol)

        return {
            'valid': valid,
            'connected': connected,
            'qed': qed,
            'sa': sa,
            'logp': logp,
            'lipinski': lipinski
        }


class Diversity:
    @staticmethod
    def similarity(fp1, fp2):
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def get_fingerprint(self, mol):
        # fp = AllChem.GetMorganFingerprintAsBitVect(
        #     mol, 2, nBits=2048, useChirality=False)
        fp = Chem.RDKFingerprint(mol)
        return fp

    def __call__(self, pocket_mols):

        if len(pocket_mols) < 2:
            return 0.0

        pocket_fps = [self.get_fingerprint(m) for m in pocket_mols]

        div = 0
        total = 0
        for i in range(len(pocket_fps)):
            for j in range(i + 1, len(pocket_fps)):
                div += 1 - self.similarity(pocket_fps[i], pocket_fps[j])
                total += 1

        return div / total


class MoleculeUniqueness:
    def __call__(self, smiles_list):
        """ smiles_list: list of SMILES strings. """
        if len(smiles_list) < 1:
            return 0.0

        return len(set(smiles_list)) / len(smiles_list)


class MoleculeNovelty:
    def __init__(self, reference_smiles):
        """
        :param reference_smiles: list of SMILES strings
        """
        self.reference_smiles = set(reference_smiles)

    def __call__(self, smiles_list):
        if len(smiles_list) < 1:
            return 0.0

        novel = [smi for smi in smiles_list if smi not in self.reference_smiles]
        return len(novel) / len(smiles_list)


class MolecularProperties:

    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        # return round((10 - sa) / 9, 2)  # from pocket2mol
        return sa

    @staticmethod
    def calculate_logp(rdmol):
        return Crippen.MolLogP(rdmol)

    @staticmethod
    def calculate_lipinski(rdmol):
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

    @classmethod
    def calculate_diversity(cls, pocket_mols):
        if len(pocket_mols) < 2:
            return 0.0

        div = 0
        total = 0
        for i in range(len(pocket_mols)):
            for j in range(i + 1, len(pocket_mols)):
                div += 1 - cls.similarity(pocket_mols[i], pocket_mols[j])
                total += 1
        return div / total

    @staticmethod
    def similarity(mol_a, mol_b):
        # fp1 = AllChem.GetMorganFingerprintAsBitVect(
        #     mol_a, 2, nBits=2048, useChirality=False)
        # fp2 = AllChem.GetMorganFingerprintAsBitVect(
        #     mol_b, 2, nBits=2048, useChirality=False)
        fp1 = Chem.RDKFingerprint(mol_a)
        fp2 = Chem.RDKFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def evaluate_pockets(self, pocket_rdmols, verbose=False):
        """
        Run full evaluation
        Args:
            pocket_rdmols: list of lists, the inner list contains all RDKit
                molecules generated for a pocket
        Returns:
            QED, SA, LogP, Lipinski (per molecule), and Diversity (per pocket)
        """

        for pocket in pocket_rdmols:
            for mol in pocket:
                Chem.SanitizeMol(mol)  # only evaluate valid molecules

        all_qed = []
        all_sa = []
        all_logp = []
        all_lipinski = []
        per_pocket_diversity = []
        for pocket in tqdm(pocket_rdmols):
            all_qed.append([self.calculate_qed(mol) for mol in pocket])
            all_sa.append([self.calculate_sa(mol) for mol in pocket])
            all_logp.append([self.calculate_logp(mol) for mol in pocket])
            all_lipinski.append([self.calculate_lipinski(mol) for mol in pocket])
            per_pocket_diversity.append(self.calculate_diversity(pocket))

        qed_flattened = [x for px in all_qed for x in px]
        sa_flattened = [x for px in all_sa for x in px]
        logp_flattened = [x for px in all_logp for x in px]
        lipinski_flattened = [x for px in all_lipinski for x in px]

        if verbose:
            print(f"{sum([len(p) for p in pocket_rdmols])} molecules from "
                  f"{len(pocket_rdmols)} pockets evaluated.")
            print(f"QED: {np.mean(qed_flattened):.3f} \pm {np.std(qed_flattened):.2f}")
            print(f"SA: {np.mean(sa_flattened):.3f} \pm {np.std(sa_flattened):.2f}")
            print(f"LogP: {np.mean(logp_flattened):.3f} \pm {np.std(logp_flattened):.2f}")
            print(f"Lipinski: {np.mean(lipinski_flattened):.3f} \pm {np.std(lipinski_flattened):.2f}")
            print(f"Diversity: {np.mean(per_pocket_diversity):.3f} \pm {np.std(per_pocket_diversity):.2f}")

        return all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity

    def __call__(self, rdmols):
        """
        Run full evaluation and return mean of each property
        Args:
            rdmols: list of RDKit molecules
        Returns:
            Dictionary with mean QED, SA, LogP, Lipinski, and Diversity values
        """

        if len(rdmols) < 1:
            return {'QED': 0.0, 'SA': 0.0, 'LogP': 0.0, 'Lipinski': 0.0,
                    'Diversity': 0.0}

        _rdmols = []
        for mol in rdmols:
            try:
                Chem.SanitizeMol(mol)  # only evaluate valid molecules
                _rdmols.append(mol)
            except ValueError as e:
                print("Tried to analyze invalid molecule")
        rdmols = _rdmols

        qed = np.mean([self.calculate_qed(mol) for mol in rdmols])
        sa = np.mean([self.calculate_sa(mol) for mol in rdmols])
        logp = np.mean([self.calculate_logp(mol) for mol in rdmols])
        lipinski = np.mean([self.calculate_lipinski(mol) for mol in rdmols])
        diversity = self.calculate_diversity(rdmols)

        return {'QED': qed, 'SA': sa, 'LogP': logp, 'Lipinski': lipinski,
                'Diversity': diversity}


def compute_gnina_scores(ligands, receptors, gnina):
    metrics = ['minimizedAffinity', 'minimizedRMSD', 'CNNscore', 'CNNaffinity', 'CNN_VS', 'CNNaffinity_variance']
    out = {m: [] for m in metrics}
    with tempfile.TemporaryDirectory() as tmpdir:
        for ligand, receptor in zip(tqdm(ligands, desc='Docking'), receptors):
            in_ligand_path = Path(tmpdir, 'in_ligand.sdf')
            out_ligand_path = Path(tmpdir, 'out_ligand.sdf')
            receptor_path = Path(tmpdir, 'receptor.pdb')
            write_sdf_file(in_ligand_path, [ligand], catch_errors=True)
            Chem.MolToPDBFile(receptor, str(receptor_path))
            if (
                    (not in_ligand_path.exists()) or
                    (not receptor_path.exists()) or
                    in_ligand_path.read_text() == '' or
                    receptor_path.read_text() == ''
            ):
                continue

            cmd = (
                f'{gnina} -r {receptor_path} -l {in_ligand_path} '
                f'--minimize --seed 42 -o {out_ligand_path} --no_gpu 1> /dev/null'
            )
            subprocess.run(cmd, shell=True)
            if not out_ligand_path.exists() or out_ligand_path.read_text() == '':
                continue

            mol = Chem.SDMolSupplier(str(out_ligand_path), sanitize=False)[0]
            for metric in metrics:
                out[metric].append(float(mol.GetProp(metric)))

    for metric in metrics:
        out[metric] = sum(out[metric]) / len(out[metric]) if len(out[metric]) > 0 else 0

    return out


def legacy_clash_score(rdmol1, rdmol2=None, margin=0.75):
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
    # source: https://en.wikipedia.org/wiki/Van_der_Waals_radius
    vdw_radii = {'N': 1.55, 'O': 1.52, 'C': 1.70, 'H': 1.10, 'S': 1.80, 'P': 1.80,
                 'Se': 1.90, 'K': 2.75, 'Na': 2.27, 'Mg': 1.73, 'Zn': 1.39, 'B': 1.92,
                 'Br': 1.85, 'Cl': 1.75, 'I': 1.98, 'F': 1.47}

    # https://en.wikipedia.org/wiki/Covalent_radius#Radii_for_multiple_bonds
    covalent_radii = {'H': 0.32, 'C': 0.60, 'N': 0.54, 'O': 0.53, 'F': 0.53, 'B': 0.73,
                      'Al': 1.11, 'Si': 1.02, 'P': 0.94, 'S': 0.94, 'Cl': 0.93, 'As': 1.06,
                      'Br': 1.09, 'I': 1.25, 'Hg': 1.33, 'Bi': 1.35}

    coord1 = rdmol1.GetConformer().GetPositions()

    if rdmol2 is None:
        radii1 = np.array([covalent_radii[a.GetSymbol()] for a in rdmol1.GetAtoms()])
        assert coord1.shape[0] == radii1.shape[0]

        dist = np.sqrt(np.sum((coord1[:, None, :] - coord1[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        clashes = dist < margin * (radii1[:, None] + radii1[None, :])

    else:
        coord2 = rdmol2.GetConformer().GetPositions()

        radii1 = np.array([vdw_radii[a.GetSymbol()] for a in rdmol1.GetAtoms()])
        assert coord1.shape[0] == radii1.shape[0]
        radii2 = np.array([vdw_radii[a.GetSymbol()] for a in rdmol2.GetAtoms()])
        assert coord2.shape[0] == radii2.shape[0]

        dist = np.sqrt(np.sum((coord1[:, None, :] - coord2[None, :, :]) ** 2, axis=-1))
        clashes = dist < margin * (radii1[:, None] + radii2[None, :])

    clashes = np.any(clashes, axis=1)
    return np.mean(clashes)


def clash_score(rdmol1, rdmol2=None, margin=0.75, ignore={'H'}):
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

    _periodic_table = AllChem.GetPeriodicTable()

    def _coord_and_radii(rdmol):
        coord = rdmol.GetConformer().GetPositions()
        radii = np.array([_get_radius(a.GetSymbol()) for a in rdmol.GetAtoms()])

        mask = np.array([a.GetSymbol() not in ignore for a in rdmol.GetAtoms()])
        coord = coord[mask]
        radii = radii[mask]

        assert coord.shape[0] == radii.shape[0]
        return coord, radii

    # INTRAMOLECULAR CLASH SCORE
    if intramolecular:
        rdmol2 = rdmol1
        _get_radius = _periodic_table.GetRcovalent  # covalent radii

    # INTERMOLECULAR CLASH SCORE
    else:
        _get_radius = _periodic_table.GetRvdw  # vdW radii

    coord1, radii1 = _coord_and_radii(rdmol1)
    coord2, radii2 = _coord_and_radii(rdmol2)

    dist = np.sqrt(np.sum((coord1[:, None, :] - coord2[None, :, :]) ** 2, axis=-1))
    if intramolecular:
        np.fill_diagonal(dist, np.inf)

    clashes = dist < margin * (radii1[:, None] + radii2[None, :])
    clashes = np.any(clashes, axis=1)
    return np.mean(clashes)
