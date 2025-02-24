"""
Natural Extension Reference Frame (NERF)

Inspiration for parallel reconstruction:
https://github.com/EleutherAI/mp_nerf and references therein

For atom names, see also:
https://www.ccpn.ac.uk/manual/v3/NEFAtomNames.html

References:
- https://onlinelibrary.wiley.com/doi/10.1002/jcc.20237 (NERF)
- https://onlinelibrary.wiley.com/doi/10.1002/jcc.26768 (for code)
"""

import warnings
import torch
import numpy as np

from src.data.misc import protein_letters_3to1
from src.constants import aa_atom_index, aa_atom_mask, aa_nerf_indices, aa_chi_indices, aa_chi_anchor_atom


# https://github.com/EleutherAI/mp_nerf/blob/master/mp_nerf/utils.py
def get_dihedral(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from:
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs:
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
        * c4: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) )


# https://github.com/EleutherAI/mp_nerf/blob/master/mp_nerf/utils.py
def get_angle(c1, c2, c3):
    """ Returns the angle in radians.
        Inputs:
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2

    # dont use acos since norms involved.
    # better use atan2 formula: atan2(cross, dot) from here:
    # https://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html

    # add a minus since we want the angle in reversed order - sidechainnet issues
    return torch.atan2( torch.norm(torch.cross(u1,u2, dim=-1), dim=-1),
                        -(u1*u2).sum(dim=-1) )


def get_nerf_params(biopython_residue):
    aa = protein_letters_3to1[biopython_residue.get_resname()]

    # Basic mask and index tensors
    atom_mask = torch.tensor(aa_atom_mask[aa], dtype=bool)
    nerf_indices = torch.tensor(aa_nerf_indices[aa], dtype=int)
    chi_indices = torch.tensor(aa_chi_indices[aa], dtype=int)

    fixed_coord = torch.zeros((5, 3))
    residue_coords = torch.zeros((14, 3))  # only required to compute internal coordinates during pre-processing
    atom_found = torch.zeros_like(atom_mask)
    for atom in biopython_residue.get_atoms():
        try:
            idx = aa_atom_index[aa][atom.get_name()]
            atom_found[idx] = True
        except KeyError:
            warnings.warn(f"{atom.get_name()} not found")
            continue

        residue_coords[idx, :] = torch.from_numpy(atom.get_coord())

        if atom.get_name() in ['N', 'CA', 'C', 'O', 'CB']:
            fixed_coord[idx, :] = torch.from_numpy(atom.get_coord())

    # Determine chi angles
    chi = torch.zeros(6)  # the last chi angle is a dummy and should always be zero
    for chi_idx, anchor in aa_chi_anchor_atom[aa].items():
        idx_a = nerf_indices[anchor, 2]
        idx_b = nerf_indices[anchor, 1]
        idx_c = nerf_indices[anchor, 0]

        coords_a = residue_coords[idx_a, :]
        coords_b = residue_coords[idx_b, :]
        coords_c = residue_coords[idx_c, :]
        coords_d = residue_coords[anchor, :]

        chi[chi_idx] = get_dihedral(coords_a, coords_b, coords_c, coords_d)

    # Compute remaining internal coordinates
    # (parallel version)
    idx_a = nerf_indices[:, 2]
    idx_b = nerf_indices[:, 1]
    idx_c = nerf_indices[:, 0]

    # update atom mask
    # remove atoms for which one or several parameters are missing/incorrect
    _atom_mask = atom_mask & atom_found & atom_found[idx_a] & atom_found[idx_b] & atom_found[idx_c]
    if not torch.all(_atom_mask == atom_mask):
        warnings.warn("Some atoms are missing for NERF reconstruction")
    atom_mask = _atom_mask

    coords_a = residue_coords[idx_a]
    coords_b = residue_coords[idx_b]
    coords_c = residue_coords[idx_c]
    coords_d = residue_coords

    length = torch.norm(coords_d - coords_c, dim=-1)
    theta = get_angle(coords_b, coords_c, coords_d)
    ddihedral = get_dihedral(coords_a, coords_b, coords_c, coords_d)

    # subtract chi angles from dihedrals
    ddihedral = ddihedral - chi[chi_indices]

    #     # (serial version)
    #     length = torch.zeros(14)
    #     theta = torch.zeros(14)
    #     ddihedral = torch.zeros(14)
    #     for i in range(5, 14):
    #         if not atom_mask[i]: # atom doesn't exist
    #             continue

    #         idx_a = nerf_indices[i, 2]
    #         idx_b = nerf_indices[i, 1]
    #         idx_c = nerf_indices[i, 0]

    #         coords_a = residue_coords[idx_a]
    #         coords_b = residue_coords[idx_b]
    #         coords_c = residue_coords[idx_c]
    #         coords_d = residue_coords[i]

    #         length[i] = torch.norm(coords_d - coords_c, dim=-1)
    #         theta[i] = get_angle(coords_b, coords_c, coords_d)
    #         ddihedral[i] = get_dihedral(coords_a, coords_b, coords_c, coords_d)

    #         # subtract chi angles from dihedrals
    #         ddihedral[i] = ddihedral[i] - chi[chi_indices[i]]

    return {
        'fixed_coord': fixed_coord,
        'atom_mask': atom_mask,
        'nerf_indices': nerf_indices,
        'length': length,
        'theta': theta,
        'chi': chi,
        'ddihedral': ddihedral,
        'chi_indices': chi_indices,
    }


# https://github.com/EleutherAI/mp_nerf/blob/master/mp_nerf/massive_pnerf.py#L38C1-L65C67
def mp_nerf_torch(a, b, c, l, theta, chi):
    """ Custom Natural extension of Reference Frame.
        Inputs:
        * a: (batch, 3) or (3,). point(s) of the plane, not connected to d
        * b: (batch, 3) or (3,). point(s) of the plane, not connected to d
        * c: (batch, 3) or (3,). point(s) of the plane, connected to d
        * theta: (batch,) or (float).  angle(s) between b-c-d
        * chi: (batch,) or float. dihedral angle(s) between the a-b-c and b-c-d planes
        Outputs: d (batch, 3) or (float). the next point in the sequence, linked to c
    """
    # safety check
    if not ( (-np.pi <= theta) * (theta <= np.pi) ).all().item():
        raise ValueError(f"theta(s) must be in radians and in [-pi, pi]. theta(s) = {theta}")
    # calc vecs
    ba = b-a
    cb = c-b
    # calc rotation matrix. based on plane normals and normalized
    n_plane  = torch.cross(ba, cb, dim=-1)
    n_plane_ = torch.cross(n_plane, cb, dim=-1)
    rotate   = torch.stack([cb, n_plane_, n_plane], dim=-1)
    rotate  /= torch.norm(rotate, dim=-2, keepdim=True)
    # calc proto point, rotate. add (-1 for sidechainnet convention)
    # https://github.com/jonathanking/sidechainnet/issues/14
    d = torch.stack([-torch.cos(theta),
                     torch.sin(theta) * torch.cos(chi),
                     torch.sin(theta) * torch.sin(chi)], dim=-1).unsqueeze(-1)
    # extend base point, set length
    return c + l.unsqueeze(-1) * torch.matmul(rotate, d).squeeze()


# inspired by: https://github.com/EleutherAI/mp_nerf/blob/master/mp_nerf/proteins.py#L323C5-L344C65
def ic_to_coords(fixed_coord, atom_mask, nerf_indices, length, theta, chi, ddihedral, chi_indices):
    """
    Run NERF in parallel for all residues.

    :param fixed_coord: (L, 5, 3) coordinates of (N, CA, C, O, CB) atoms, they don't depend on chi angles
    :param atom_mask: (L, 14) indicates whether atom exists in this residue
    :param nerf_indices: (L, 14, 3) indices of the three previous atoms ({c, b, a} for the NERF algorithm)
    :param length: (L, 14) bond length between this and previous atom
    :param theta: (L, 14) angle between this and previous two atoms
    :param chi: (L, 6) values of the 5 rotatable bonds, plus zero in last column
    :param ddihedral: (L, 14) angle offset to which chi is added
    :param chi_indices: (L, 14) indexes into the chi array
    :returns: (L, 14, 3) tensor with all coordinates, non-existing atoms are assigned CA coords
    """

    if not torch.all(chi[:, 5] == 0):
        chi[:, 5] = 0.0
        warnings.warn("Last column of 'chi' tensor should be zero. Overriding values.")
    assert torch.all(chi[:, 5] == 0)

    L, device = fixed_coord.size(0), fixed_coord.device
    coords = torch.zeros((L, 14, 3), device=device)
    coords[:, :5, :] = fixed_coord

    for i in range(5, 14):
        level_mask = atom_mask[:, i]
        #     level_mask = torch.ones(len(atom_mask), dtype=bool)

        length_i = length[level_mask, i]
        theta_i = theta[level_mask, i]

        #     dihedral_i = dihedral[level_mask, i]
        dihedral_i = chi[level_mask, chi_indices[level_mask, i]] + ddihedral[level_mask, i]

        idx_a = nerf_indices[level_mask, i, 2]
        idx_b = nerf_indices[level_mask, i, 1]
        idx_c = nerf_indices[level_mask, i, 0]

        coords[level_mask, i] = mp_nerf_torch(coords[level_mask, idx_a],
                                              coords[level_mask, idx_b],
                                              coords[level_mask, idx_c],
                                              length_i,
                                              theta_i,
                                              dihedral_i)

    if coords.isnan().any():
        warnings.warn("Side chain reconstruction error. Removing affected atoms...")

        # mask out affected atoms
        m, n, _ = torch.where(coords.isnan())
        atom_mask[m, n] = False
        coords[m, n, :] = 0.0

    # replace non-existing atom coords with CA coords (TODO: don't hard-code CA index)
    coords = atom_mask.unsqueeze(-1) * coords + \
             (~atom_mask.unsqueeze(2)) * coords[:, 1, :].unsqueeze(1)

    return coords
