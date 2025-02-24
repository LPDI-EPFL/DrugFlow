import torch
from torch_scatter import scatter_add, scatter_mean

from src.constants import atom_decoder, vdw_radii
_vdw_radii = {**vdw_radii}
_vdw_radii['NH'] = vdw_radii['N']
_vdw_radii['N+'] = vdw_radii['N']
_vdw_radii['O-'] = vdw_radii['O']
_vdw_radii['NOATOM'] = 0
vdw_radii_array = torch.tensor([_vdw_radii[a] for a in atom_decoder])


def clash_loss(ligand_coord, ligand_types, ligand_mask, pocket_coord,
               pocket_types, pocket_mask):
    """
    Computes a clash loss that penalizes interatomic distances smaller than the
    sum of van der Waals radii between atoms.
    """

    ligand_radii = vdw_radii_array[ligand_types].to(ligand_coord.device)
    pocket_radii = vdw_radii_array[pocket_types].to(pocket_coord.device)

    dist = torch.sqrt(torch.sum((ligand_coord[:, None, :] - pocket_coord[None, :, :]) ** 2, dim=-1))
    # dist[ligand_mask[:, None] != pocket_mask[None, :]] = float('inf')

    # compute linearly decreasing penalty
    # penalty = max(1 - 1/sum_vdw * d, 0)
    sum_vdw = ligand_radii[:, None] + pocket_radii[None, :]
    loss = torch.clamp(1 - dist / sum_vdw, min=0.0)  # (n_ligand, n_pocket)

    loss = scatter_add(loss, pocket_mask, dim=1)
    loss = scatter_mean(loss, ligand_mask, dim=0)
    loss = loss.diag()

    # # DEBUG (non-differentiable version)
    # dist = torch.sqrt(torch.sum((ligand_coord[:, None, :] - pocket_coord[None, :, :]) ** 2, dim=-1))
    # dist[ligand_mask[:, None] != pocket_mask[None, :]] = float('inf')
    # _loss = torch.clamp(1 - dist / sum_vdw, min=0.0)  # (n_ligand, n_pocket)
    # _loss = _loss.sum(dim=-1)
    # _loss = scatter_mean(_loss, ligand_mask, dim=0)
    # assert torch.allclose(loss, _loss)

    return loss


class TimestepSampler:
    def __init__(self, type='uniform', lowest_t=1, highest_t=500):
        assert type in {'uniform', 'sigmoid'}
        self.type = type
        self.lowest_t = lowest_t
        self.highest_t = highest_t

    def __call__(self, n, device=None):
        if self.type == 'uniform':
            t_int = torch.randint(self.lowest_t, self.highest_t + 1,
                                  size=(n, 1), device=device)

        elif self.type == 'sigmoid':
            weight_fun = lambda t: 1.45 * torch.sigmoid(-t * 10 / self.highest_t + 5) + 0.05

            possible_ts = torch.arange(self.lowest_t, self.highest_t + 1, device=device)
            weights = weight_fun(possible_ts)
            weights = weights / weights.sum()
            t_int = possible_ts[torch.multinomial(weights, n, replacement=True)].unsqueeze(-1)

        return t_int.float()


class TimestepWeights:
    def __init__(self, weight_type, a, b):
        if weight_type != 'sigmoid':
            raise NotImplementedError("Only sigmoidal loss weighting is available.")
        # self.weight_fn = lambda t: a * torch.sigmoid((-t + 0.5) * b) + (1 - a / 2)
        self.weight_fn = lambda t: a * torch.sigmoid((t - 0.5) * b) + (1 - a / 2)

    def __call__(self, t_array):
        # normalized t \in [0, 1]
        # return self.weight_fn(1 - t_array)
        return self.weight_fn(t_array)
