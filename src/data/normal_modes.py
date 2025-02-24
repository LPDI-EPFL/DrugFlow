import warnings
import numpy as np
import prody
prody.confProDy(verbosity='none')
from prody import parsePDB, ANM


def pdb_to_normal_modes(pdb_file, num_modes=5, nmax=5000):
    """
    Compute normal modes for a PDB file using an Anisotropic Network Model (ANM)
    http://prody.csb.pitt.edu/tutorials/enm_analysis/anm.html (accessed 01/11/2023)
    """
    protein = parsePDB(pdb_file, model=1).select('calpha')

    if len(protein) > nmax:
        warnings.warn("Protein is too big. Returning zeros...")
        eig_vecs = np.zeros((len(protein), 3, num_modes))

    else:
        # build Hessian
        anm = ANM('ANM analysis')
        anm.buildHessian(protein, cutoff=15.0, gamma=1.0)

        # calculate normal modes
        anm.calcModes(num_modes, zeros=False)

        # only use slowest modes
        eig_vecs = anm.getEigvecs()  # shape: (num_atoms * 3, num_modes)
        eig_vecs = eig_vecs.reshape(len(protein), 3, num_modes)
        # eig_vals = anm.getEigvals()  # shape: (num_modes,)

    nm_dict = {}
    for atom, nm_vec in zip(protein, eig_vecs):
        chain = atom.getChid()
        resi = atom.getResnum()
        name = atom.getName()
        nm_dict[(chain, resi, name)] = nm_vec.T

    return nm_dict


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import torch
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=Path)
    parser.add_argument('--outfile', type=Path, default=None)
    args = parser.parse_args()

    # Read data split
    split_path = Path(args.basedir, 'split_by_name.pt')
    data_split = torch.load(split_path)

    pockets = [x[0] for split in data_split.values() for x in split]

    all_normal_modes = {}
    for p in tqdm(pockets):
        pdb_file = Path(args.basedir, 'crossdocked_pocket10', p)

        try:
            nm_dict = pdb_to_normal_modes(str(pdb_file))
            all_normal_modes[p] = nm_dict
        except AttributeError as e:
            warnings.warn(str(e))

    np.save(args.outfile, all_normal_modes)
