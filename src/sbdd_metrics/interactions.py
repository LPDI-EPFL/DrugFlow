import prody
import prolif as plf
import pandas as pd
import subprocess

from io import StringIO
from prolif.fingerprint import Fingerprint
from prolif.plotting.complex3d import Complex3D
from prolif.residue import ResidueId
from prolif.ifp import IFP
from rdkit import Chem
from tqdm import tqdm


prody.confProDy(verbosity='none')


INTERACTION_LIST = [
    'Anionic', 'Cationic', # Salt Bridges ~400 kJ/mol
    'HBAcceptor', 'HBDonor', # Hydrogen bonds ~10 kJ/mol
    'XBAcceptor', 'XBDonor', # Halogen bonds ~5-30 kJ/mol
    'CationPi', 'PiCation', # 5-10 kJ/mol
    'PiStacking', # ~2-10 kJ/mol
    'Hydrophobic', # 1-10 kJ/mol
]

INTERACTION_ALIASES = {
    'Anionic': 'SaltBridge',
    'Cationic': 'SaltBridge',
    'HBAcceptor': 'HBAcceptor',
    'HBDonor': 'HBDonor',
    'XBAcceptor': 'HalogenBond',
    'XBDonor': 'HalogenBond',
    'CationPi': 'CationPi',
    'PiCation': 'PiCation',
    'PiStacking': 'PiStacking',
    'Hydrophobic': 'Hydrophobic',
}

INTERACTION_COLORS = {
    'SaltBridge': '#eba823',
    'HBDonor': '#3d5dfc',
    'HBAcceptor': '#3d5dfc',
    'HalogenBond': '#53f514',
    'CationPi': '#ff0000',
    'PiCation': '#ff0000',
    'PiStacking': '#e359d8',
    'Hydrophobic': '#c9c5c5',
}

INTERACTION_IMPORTANCE = ['SaltBridge', 'HydrogenBond', 'HBAcceptor', 'HBDonor', 'CationPi', 'PiCation', 'PiStacking', 'Hydrophobic']

REDUCE_EXEC = './reduce'

def remove_residue_by_atomic_number(structure, resnum, chain_id, icode):
    exclude_selection = f'not (chain {chain_id} and resnum {resnum} and icode {icode})'
    structure = structure.select(exclude_selection)
    return structure


def read_protein(protein_path, verbose=False, reduce_exec=REDUCE_EXEC):
    structure = prody.parsePDB(protein_path).select('protein')
    hydrogens = structure.select('hydrogen')
    if hydrogens is None or len(hydrogens) < len(set(structure.getResnums())):
        if verbose:
            print('Target structure is not protonated. Adding hydrogens...')

        reduce_cmd = f'{str(reduce_exec)} {protein_path}'
        reduce_result = subprocess.run(reduce_cmd, shell=True, capture_output=True, text=True)
        if reduce_result.returncode != 0:
            raise RuntimeError('Error during reduce execution:', reduce_result.stderr)

        pdb_content = reduce_result.stdout
        stream = StringIO()
        stream.write(pdb_content)
        stream.seek(0)
        structure = prody.parsePDBStream(stream).select('protein')

    # Select only one (largest) altloc
    altlocs = set(structure.getAltlocs())
    try:
        best_altloc = max(altlocs, key=lambda a: structure.select(f'altloc "{a}"').numAtoms())
        structure = structure.select(f'altloc "{best_altloc}"')
    except TypeError:
        # Strange thing that happens only once in the beginning sometimes...
        best_altloc = max(altlocs, key=lambda a: structure.select(f'altloc "{a}"').numAtoms())
        structure = structure.select(f'altloc "{best_altloc}"')
    
    return prepare_protein(structure, to_exclude=[], verbose=verbose)


def prepare_protein(input_structure, to_exclude=[], verbose=False):
    structure = input_structure.copy()

    # Remove residues with bad atoms
    if verbose and len(to_exclude) > 0:
        print(f'Removing {len(to_exclude)} residues...')
    for resnum, chain_id, icode in to_exclude:
        exclude_selection = f'not (chain {chain_id} and resnum {resnum})'
        structure = structure.select(exclude_selection)

    # Write new PDB content to the stream
    stream = StringIO()
    prody.writePDBStream(stream, structure)
    stream.seek(0)
    
    # Sanitize
    rdprot = Chem.MolFromPDBBlock(stream.read(), sanitize=False, removeHs=False)
    try:
        Chem.SanitizeMol(rdprot)
        plfprot = plf.Molecule(rdprot)
        return plfprot
    
    except Chem.AtomValenceException as e:
        atom_num = int(e.args[0].replace('Explicit valence for atom # ', '').split()[0])
        info = rdprot.GetAtomWithIdx(atom_num).GetPDBResidueInfo()
        resnum = info.GetResidueNumber()
        chain_id = info.GetChainId()
        icode = f'"{info.GetInsertionCode()}"'
        
        to_exclude_next = to_exclude + [(resnum, chain_id, icode)]
        if verbose:
            print(f'[{len(to_exclude_next)}] Removing broken residue with atom={atom_num}, resnum={resnum}, chain_id={chain_id}, icode={icode}')
        return prepare_protein(input_structure, to_exclude=to_exclude_next)


def prepare_ligand(mol):
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    ligand_plf = plf.Molecule.from_rdkit(mol)
    return ligand_plf


def sdf_reader(sdf_path, proress_bar=False):
    supp = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=False)
    for mol in tqdm(supp) if progress_bar else supp:
        yield prepare_ligand(mol)


def profile_detailed(
        ligand_plf, protein_plf, interaction_list=INTERACTION_LIST, ligand_name='ligand', protein_name='protein'
    ):

    fp = Fingerprint(interactions=interaction_list)
    fp.run_from_iterable(lig_iterable=[ligand_plf], prot_mol=protein_plf, progress=False)

    profile = []

    for ligand_residue in ligand_plf.residues:
        for protein_residue in protein_plf.residues:
            metadata = fp.metadata(ligand_plf[ligand_residue], protein_plf[protein_residue])
            for int_name, int_metadata in metadata.items():
                for int_instance in int_metadata:
                    profile.append({
                        'ligand': ligand_name,
                        'protein': protein_name,
                        'ligand_residue': str(ligand_residue),
                        'protein_residue': str(protein_residue),
                        'interaction': int_name,
                        'alias': INTERACTION_ALIASES[int_name],
                        'ligand_atoms': ','.join(map(str, int_instance['indices']['ligand'])),
                        'protein_atoms': ','.join(map(str, int_instance['indices']['protein'])),
                        'ligand_orig_atoms': ','.join(map(str, int_instance['parent_indices']['ligand'])),
                        'protein_orig_atoms': ','.join(map(str, int_instance['parent_indices']['protein'])),
                        'distance': int_instance['distance'],
                        'plane_angle': int_instance.get('plane_angle', None),
                        'normal_to_centroid_angle': int_instance.get('normal_to_centroid_angle', None),
                        'intersect_distance': int_instance.get('intersect_distance', None),
                        'intersect_radius': int_instance.get('intersect_radius', None),
                        'pi_ring': int_instance.get('pi_ring', None),
                    })

    return pd.DataFrame(profile)


def map_orig_atoms_to_new(atoms, mol):
    orig2new = dict()
    for atom in mol.GetAtoms():
        orig2new[atom.GetUnsignedProp("mapindex")] = atom.GetIdx()
    
    atoms = list(map(int, atoms.split(',')))
    new_atoms = ','.join(map(str, [orig2new[atom] for atom in atoms]))
    return new_atoms


def visualize(profile, ligand_plf, protein_plf):
    metadata = dict()

    for _, row in profile.iterrows():
        if 'ligand_atoms' not in row:
            row['ligand_atoms'] = map_orig_atoms_to_new(row['ligand_orig_atoms'], ligand_plf)
        if 'protein_atoms' not in row:
            row['protein_atoms'] = map_orig_atoms_to_new(row['protein_orig_atoms'], protein_plf[row['residue']])

        namenum, chain = row['residue'].split('.')
        name = namenum[:3]
        num = int(namenum[3:])
        protres = ResidueId(name=name, number=num, chain=chain)
        key = (ResidueId(name='UNL', number=1, chain=None), protres)

        metadata.setdefault(key, dict())
        interaction = {
            'indices': {
                'ligand': tuple(map(int, row['ligand_atoms'].split(','))),
                'protein': tuple(map(int, row['protein_atoms'].split(','))),
            },
            'parent_indices': {
                'ligand': tuple(map(int, row['ligand_atoms'].split(','))),
                'protein': tuple(map(int, row['protein_atoms'].split(','))),
            },
            'distance': row['distance'],
        }
        # if row['plane_angle'] is not None:
        #     interaction['plane_angle'] = row['plane_angle']
        # if row['normal_to_centroid_angle'] is not None:
        #     interaction['normal_to_centroid_angle'] = row['normal_to_centroid_angle']
        # if row['intersect_distance'] is not None:
        #     interaction['intersect_distance'] = row['intersect_distance']
        # if row['intersect_radius'] is not None:
        #     interaction['intersect_radius'] = row['intersect_radius']
        # if row['pi_ring'] is not None:
        #     interaction['pi_ring'] = row['pi_ring']

        metadata[key].setdefault(row['alias'], list()).append(interaction)
    
    ifp = IFP(metadata)
    fp = Fingerprint(interactions=INTERACTION_LIST, vicinity_cutoff=8.0)
    fp.ifp = {0: ifp}
    Complex3D.COLORS.update(INTERACTION_COLORS)
    v = fp.plot_3d(ligand_mol=ligand_plf, protein_mol=protein_plf, frame=0)
    return v