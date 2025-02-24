from functools import reduce
import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add

from src.utils import bvm


class LinearSchedule:
    """
    We use the scheduling parameter \beta to linearly remove noise, i.e.
    \bar{\beta}_t = 1 - h (h: step size) with
    \bar{Q}_t = \bar{\beta}_t I + (1 - \bar{\beta}_t) 1_vec z1^T

    From this, it follows that for each step transition matrix, we have
    \beta_t = \bar{\beta}_t / \bar{\beta}_{t-h} = \frac{1-t}{1-t+h}
    """
    def __init__(self):
        super().__init__()

    def beta_bar(self, t):
        return 1 - t

    def beta(self, t, step_size):
        return (1 - t) / (1 - t + step_size)


class UniformPriorMarkovBridge:
    """
    Markov bridge model in which z0 is drawn from a uniform prior.
    Transitions are defined as:
    Q_t = \beta_t I + (1 - \beta_t) 1_vec z1^T
    where z1 is a one-hot representation of the final state.
    We follow the notation from [1] and multiply transition matrices from the
    right to one-hot state vectors.

    We use the scheduling parameter \beta to linearly remove noise, i.e.
    \bar{\beta}_t = 1 - h (h: step size) with
    \bar{Q}_t = \bar{\beta}_t I + (1 - \bar{\beta}_t) 1_vec z1^T

    From this, it follows that for each step transition matrix, we have
    \beta_t = \bar{\beta}_t / \bar{\beta}_{t-h} = \frac{1-t}{1-t+h}

    [1] Austin, Jacob, et al.
    "Structured denoising diffusion models in discrete state-spaces."
    Advances in Neural Information Processing Systems 34 (2021): 17981-17993.
    """
    def __init__(self, dim, loss_type='CE', step_size=None):
        assert loss_type in ['VLB', 'CE']
        self.dim = dim
        self.step_size = step_size  # required for VLB
        self.schedule = LinearSchedule()
        self.loss_type = loss_type
        super(UniformPriorMarkovBridge, self).__init__()

    @staticmethod
    def sample_categorical(p):
        """
        Sample from categorical distribution defined by probabilities 'p'
        :param p: (n, dim)
        :return: one-hot encoded samples (n, dim)
        """
        sampled = torch.multinomial(p, 1).squeeze(-1)
        return F.one_hot(sampled, num_classes=p.size(1)).float()

    def p_z0(self, batch_mask):
        return torch.ones((len(batch_mask), self.dim), device=batch_mask.device) / self.dim

    def sample_z0(self, batch_mask):
        """ Prior. """
        z0 = self.sample_categorical(self.p_z0(batch_mask))
        return z0

    def p_zt(self, z0, z1, t, batch_mask):
        Qt_bar = self.get_Qt_bar(t, z1, batch_mask)
        return bvm(z0, Qt_bar)

    def sample_zt(self, z0, z1, t, batch_mask):
        zt = self.sample_categorical(self.p_zt(z0, z1, t, batch_mask))
        return zt

    def p_zt_given_zs_and_z1(self, zs, z1, s, t, batch_mask):
        # 'z1' are one-hot "probabilities" for each class
        Qt = self.get_Qt(t, s, z1, batch_mask)
        # from pdb import set_trace; set_trace()
        q_zs_given_zt = bvm(zs, Qt)
        return q_zs_given_zt

    def p_zt_given_zs(self, zs, p_z1_hat, s, t, batch_mask):
        """
        Note that x can also represent a categorical distribution to compute
        transitions more efficiently at sampling time:
        p(z_t|z_s) = \sum_{\hat{z}_1} p(z_t | z_s, \hat{z}_1) * p(\hat{z}_1 | z_s)
                   = \sum_i z_s (\beta_t I + (1 - \beta_t) 1_vec z1_i^T) * \hat{p}_i
                   = \beta_t z_s I + (1 - \beta_t) z_s 1_vec \hat{p}^t
        """
        return self.p_zt_given_zs_and_z1(zs, p_z1_hat, s, t, batch_mask)

    def sample_zt_given_zs(self, zs, z1_logits, s, t, batch_mask):
        p_z1 = z1_logits.softmax(dim=-1)
        zt = self.sample_categorical(self.p_zt_given_zs(zs, p_z1, s, t, batch_mask))
        return zt

    def compute_loss(self, pred_logits, zs, z1, batch_mask, s, t, reduce='mean'):
        """ Compute loss per sample. """
        assert reduce in {'mean', 'sum', 'none'}

        if self.loss_type == 'CE':
            loss = F.cross_entropy(pred_logits, z1, reduction='none')

        else:  # VLB
            true_p_zs = self.p_zt_given_zs_and_z1(zs, z1, s, t, batch_mask)
            pred_p_zs = self.p_zt_given_zs(zs, pred_logits.softmax(dim=-1), s, t, batch_mask)
            loss = F.kl_div(pred_p_zs.log(), true_p_zs, reduction='none').sum(dim=-1)

        if reduce == 'mean':
            loss = scatter_mean(loss, batch_mask, dim=0)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0)

        return loss

    def get_Qt(self, t, s, z1, batch_mask):
        """ Returns one-step transition matrix from step s to step t. """

        beta_t_given_s = self.schedule.beta(t, t - s)
        beta_t_given_s = beta_t_given_s.unsqueeze(-1)[batch_mask]

        # Q_t = beta_t * I + (1 - beta_t) * ones (dot) z1^T
        Qt = beta_t_given_s * torch.eye(self.dim, device=t.device).unsqueeze(0) + \
             (1 - beta_t_given_s) * z1.unsqueeze(1)
             # (1 - beta_t_given_s) * (torch.ones(self.dim, 1, device=t.device) @ z1)

        # assert (Qt.sum(-1) == 1).all()

        return Qt

    def get_Qt_bar(self, t, z1, batch_mask):
        """ Returns transition matrix from step 0 to step t. """

        beta_bar_t = self.schedule.beta_bar(t)
        beta_bar_t = beta_bar_t.unsqueeze(-1)[batch_mask]

        # Q_t_bar = beta_bar * I + (1 - beta_bar) * ones (dot) z1^T
        Qt_bar = beta_bar_t * torch.eye(self.dim, device=t.device).unsqueeze(0) + \
                 (1 - beta_bar_t) * z1.unsqueeze(1)
                 # (1 - beta_bar_t) * (torch.ones(self.dim, 1, device=t.device) @ z1)

        # assert (Qt_bar.sum(-1) == 1).all()

        return Qt_bar


class MarginalPriorMarkovBridge(UniformPriorMarkovBridge):
    def __init__(self, dim, prior_p, loss_type='CE', step_size=None):
        self.prior_p = prior_p
        print('Marginal Prior MB')
        super(MarginalPriorMarkovBridge, self).__init__(dim, loss_type, step_size)

    def p_z0(self, batch_mask):
        device = batch_mask.device
        p = torch.ones((len(batch_mask), self.dim), device=device) * self.prior_p.view(1, -1).to(device)
        return p
