import math
import torch
import torch.nn.functional as F
import numpy as np


class DistributionNodes:
    def __init__(self, histogram):

        histogram = torch.tensor(histogram).float()
        histogram = histogram + 1e-3  # for numerical stability

        prob = histogram / histogram.sum()

        self.idx_to_n_nodes = torch.tensor(
            [[(i, j) for j in range(prob.shape[1])] for i in range(prob.shape[0])]
        ).view(-1, 2)

        self.n_nodes_to_idx = {tuple(x.tolist()): i
                               for i, x in enumerate(self.idx_to_n_nodes)}

        self.prob = prob
        self.m = torch.distributions.Categorical(self.prob.view(-1),
                                                 validate_args=True)

        self.n1_given_n2 = \
            [torch.distributions.Categorical(prob[:, j], validate_args=True)
             for j in range(prob.shape[1])]
        self.n2_given_n1 = \
            [torch.distributions.Categorical(prob[i, :], validate_args=True)
             for i in range(prob.shape[0])]

        # entropy = -torch.sum(self.prob.view(-1) * torch.log(self.prob.view(-1) + 1e-30))
        # entropy = self.m.entropy()
        # print("Entropy of n_nodes: H[N]", entropy.item())

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        num_nodes_lig, num_nodes_pocket = self.idx_to_n_nodes[idx].T
        return num_nodes_lig, num_nodes_pocket

    def sample_conditional(self, n1=None, n2=None):
        assert (n1 is None) ^ (n2 is None), \
            "Exactly one input argument must be None"

        m = self.n1_given_n2 if n2 is not None else self.n2_given_n1
        c = n2 if n2 is not None else n1

        return torch.tensor([m[i].sample() for i in c], device=c.device)

    def log_prob(self, batch_n_nodes_1, batch_n_nodes_2):
        assert len(batch_n_nodes_1.size()) == 1
        assert len(batch_n_nodes_2.size()) == 1

        idx = torch.tensor(
            [self.n_nodes_to_idx[(n1, n2)]
             for n1, n2 in zip(batch_n_nodes_1.tolist(), batch_n_nodes_2.tolist())]
        )

        # log_probs = torch.log(self.prob.view(-1)[idx] + 1e-30)
        log_probs = self.m.log_prob(idx)

        return log_probs.to(batch_n_nodes_1.device)

    def log_prob_n1_given_n2(self, n1, n2):
        assert len(n1.size()) == 1
        assert len(n2.size()) == 1
        log_probs = torch.stack([self.n1_given_n2[c].log_prob(i.cpu())
                                 for i, c in zip(n1, n2)])
        return log_probs.to(n1.device)

    def log_prob_n2_given_n1(self, n2, n1):
        assert len(n2.size()) == 1
        assert len(n1.size()) == 1
        log_probs = torch.stack([self.n2_given_n1[c].log_prob(i.cpu())
                                 for i, c in zip(n2, n1)])
        return log_probs.to(n2.device)


def cosine_beta_schedule_midi(timesteps, s=0.008, nu=1.0, clip=False):
    """
    Modified cosine schedule as proposed in https://arxiv.org/abs/2302.09048.
    Note: we use (t/T)^\nu not (t/T + s)^\nu as written in the MiDi paper
    We also divide by alphas_cumprod[0] as the original cosine schedule from
    https://arxiv.org/abs/2102.09672
    """
    device = nu.device if torch.is_tensor(nu) else None
    x = torch.linspace(0, timesteps, timesteps + 1, device=device)
    alphas_cumprod = torch.cos(0.5 * np.pi * ((x / timesteps)**nu + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    if clip:
        alphas_cumprod = torch.cat([torch.tensor([1.0], device=alphas_cumprod.device), alphas_cumprod])
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, min=0, max=0.999)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
    return alphas_cumprod


class CosineSchedule(torch.nn.Module):
    """
    nu=1.0 corresponds to the standard cosine schedule
    """

    def __init__(self, timesteps, nu=1.0, trainable=False, clip_alpha2_step=0.001):
        super(CosineSchedule, self).__init__()
        self.timesteps = timesteps
        self.trainable = trainable
        self.nu = nu
        assert 0.0 <= clip_alpha2_step < 1.0
        self.clip = clip_alpha2_step

        if self.trainable:
            self.nu = torch.nn.Parameter(torch.Tensor([nu]), requires_grad=True)
        else:
            self._alpha2 = self.alphas2
            self._gamma = torch.nn.Parameter(self.gammas, requires_grad=False)

    @property
    def alphas2(self):
        """
        Cumulative alpha squared.
        Called alpha_bar in: Nichol, Alexander Quinn, and Prafulla Dhariwal.
        "Improved denoising diffusion probabilistic models." PMLR, 2021.
        """
        if hasattr(self, '_alpha2'):
            return self._alpha2

        assert isinstance(self.nu, float) or ~self.nu.isnan()

        # our alpha is eqivalent to sqrt(alpha) from https://arxiv.org/abs/2102.09672, where the cosine schedule was introduced
        alphas2 = cosine_beta_schedule_midi(self.timesteps, nu=self.nu, clip=False)

        # avoid singularities near t=T
        alphas2 = torch.cat([torch.tensor([1.0], device=alphas2.device), alphas2])
        alphas2_step = alphas2[1:] / alphas2[:-1]
        alphas2_step = torch.clip(alphas2_step, min=self.clip, max=1.0)
        alphas2 = torch.cumprod(alphas2_step, dim=0)

        return alphas2

    @property
    def alphas2_t_given_tminus1(self):
        """
        Alphas for a single transition
        """
        alphas2 = torch.cat([torch.tensor([1.0]), self.alphas2])
        return alphas2[1:] / alphas2[:-1]

    @property
    def gammas(self):
        """
        Gammas as defined in appendix B of the EDM paper
        gamma_t = -(log alpha_t^2 - log sigma_t^2)
        """
        if hasattr(self, '_gamma'):
            return self._gamma

        alphas2 = self.alphas2
        sigmas2 = 1 - alphas2

        gammas = -(torch.log(alphas2) - torch.log(sigmas2))

        return gammas.float()

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gammas[t_int]

    @staticmethod
    def alpha(gamma):
        """ Computes alpha given gamma. """
        return torch.sqrt(torch.sigmoid(-gamma))

    @staticmethod
    def sigma(gamma):
        """ Computes sigma given gamma. """
        return torch.sqrt(torch.sigmoid(gamma))

    @staticmethod
    def SNR(gamma):
        """ Computes signal to noise ratio (alpha^2/sigma^2) given gamma. """
        return torch.exp(-gamma)

    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor):
        """
        Computes sigma_t_given_s, using gamma_t and gamma_s. Used during sampling.
        These are defined as:
            alpha_t_given_s = alpha_t / alpha_s,
            sigma_t_given_s = sqrt(1 - (alpha_t_given_s)^2 ).
        """
        sigma2_t_given_s = -torch.expm1(
            F.softplus(gamma_s) - F.softplus(gamma_t))

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = torch.clip(alpha_t_given_s, min=self.clip ** 0.5, max=1.0)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s
