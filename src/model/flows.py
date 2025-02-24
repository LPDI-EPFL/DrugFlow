from abc import ABC
from abc import abstractmethod
import math
import torch
from torch_scatter import scatter_mean, scatter_add

import src.data.so3_utils as so3


class ICFM(ABC):
    """
    Abstract base class for all Independent-coupling CFM classes.
    Defines a common interface.
    Notation:
    - zt is the intermediate representation at time step t \in [0, 1]
    - zs is the noised representation at time step s < t

    # TODO: add interpolation schedule (not necessrily linear)
    """
    def __init__(self, sigma):
        self.sigma = sigma

    @abstractmethod
    def sample_zt(self, z0, z1, t, *args, **kwargs):
        """ TODO. """
        pass

    @abstractmethod
    def sample_zt_given_zs(self, *args, **kwargs):
        """ Perform update, typically using an explicit Euler step. """
        pass

    @abstractmethod
    def sample_z0(self, *args, **kwargs):
        """ Prior. """
        pass

    @abstractmethod
    def compute_loss(self, pred, z0, z1, *args, **kwargs):
        """ Compute loss per sample. """
        pass


class CoordICFM(ICFM):
    def __init__(self, sigma):
        self.dim = 3
        self.scale = 2.7
        super().__init__(sigma)

    def sample_zt(self, z0, z1, t, batch_mask):
        zt = t[batch_mask] * z1 + (1 - t)[batch_mask] * z0
        # zt = self.sigma * z0 + t[batch_mask] * z1 + (1 - t)[batch_mask] * z0  # TODO: do we have to compute Psi?
        return zt

    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask):
        """ Perform an explicit Euler step. """
        step_size = t - s
        zt = zs + step_size[batch_mask] * self.scale * pred
        return zt

    def sample_z0(self, com, batch_mask):
        """ Prior. """
        z0 = torch.randn((len(batch_mask), self.dim), device=batch_mask.device)

        # Move center of mass
        z0 = z0 + com[batch_mask]

        return z0

    def reduce_loss(self, loss, batch_mask, reduce):
        assert reduce in {'mean', 'sum', 'none'}

        if reduce == 'mean':
            loss = scatter_mean(loss / self.dim, batch_mask, dim=0)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0)

        return loss

    def compute_loss(self, pred, z0, z1, t, batch_mask, reduce='mean'):
        """ Compute loss per sample. """

        loss = torch.sum((pred - (z1 - z0) / self.scale) ** 2, dim=-1)

        return self.reduce_loss(loss, batch_mask, reduce)

    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """
        # z1 = z0 + pred
        z1 = zt + (1 - t)[batch_mask] * pred
        return z1


class TorusICFM(ICFM):
    """
    Following:
    Chen, Ricky TQ, and Yaron Lipman.
    "Riemannian flow matching on general geometries."
    arXiv preprint arXiv:2302.03660 (2023).
    """
    def __init__(self, sigma, dim, scheduler_args=None):
        super().__init__(sigma)
        self.dim = dim

        # Scheduler that determines the rate at which the geodesic distance decreases
        scheduler_args = scheduler_args or {}
        scheduler_args["type"] = scheduler_args.get("type", "linear")  # default
        scheduler_args["learn_scaled"] = scheduler_args.get("learn_scaled", False)  # default

        # linear scheduler: kappa(t) = 1-t (default)
        if scheduler_args["type"] == "linear":
            # equivalent to: 1 - kappa(t)
            self.flow_scaling = lambda t: t

            # equivalent to: -1 * d/dt kappa(t)
            self.velocity_scaling = lambda t: torch.ones_like(t)

        # exponential scheduler: kappa(t) = exp(-c*t)
        elif scheduler_args["type"] == "exponential":

            self.c = scheduler_args["c"]
            assert self.c > 0

            # equivalent to: 1 - kappa(t)
            self.flow_scaling = lambda t: 1 - torch.exp(-self.c * t)

            # equivalent to: -1 * d/dt kappa(t)
            self.velocity_scaling = lambda t: self.c * torch.exp(-self.c * t)

        # polynomial scheduler: kappa(t) = (1-t)^k
        elif scheduler_args["type"] == "polynomial":
            self.k = scheduler_args["k"]
            assert self.k > 0

            # equivalent to: 1 - kappa(t)
            self.flow_scaling = lambda t: 1 - (1 - t)**self.k

            # equivalent to: -1 * d/dt kappa(t)
            self.velocity_scaling = lambda t: self.k * (1 - t)**(self.k - 1)

        else:
            raise NotImplementedError(f"Scheduler {scheduler_args['type']} not implemented.")

        kappa_interval = self.flow_scaling(torch.tensor([0.0, 1.0]))
        if kappa_interval[0] != 0.0 or kappa_interval[1] != 1.0:
            print(f"Scheduler should satisfy kappa(0)=1 and kappa(1)=0. Found "
                  f"interval {kappa_interval.tolist()} instead.")

        # determines whether the scaled vector field is learned or the scheduler
        # is post-multiplied
        self.learn_scaled = scheduler_args["learn_scaled"]

    @staticmethod
    def wrap(angle):
        """ Maps angles to range [-\pi, \pi). """
        return ((angle + math.pi) % (2 * math.pi)) - math.pi

    def exponential_map(self, x, u):
        """
        :param x: point on the manifold
        :param u: point on the tangent space
        """
        return self.wrap(x + u)

    @staticmethod
    def logarithm_map(x, y):
        """
        :param x, y: points on the manifold
        """
        return torch.atan2(torch.sin(y - x), torch.cos(y - x))

    def sample_zt(self, z0, z1, t, batch_mask):
        """ expressed in terms of exponential and logarithm maps """

        # apply logarithm map
        # zt_tangent = t[batch_mask] * self.logarithm_map(z0, z1)
        zt_tangent = self.flow_scaling(t)[batch_mask] * self.logarithm_map(z0, z1)

        # apply exponential map
        return self.exponential_map(z0, zt_tangent)

    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """

        # estimate z1_tangent based on zt and pred only
        if self.learn_scaled:
            pred = pred / torch.clamp(self.velocity_scaling(t), min=1e-3)[batch_mask]

        z1_tangent = (1 - t)[batch_mask] * pred

        # exponential map
        return self.exponential_map(zt, z1_tangent)

    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask):
        """ Perform update, typically using an explicit Euler step. """

        step_size = t - s
        zt_tangent = step_size[batch_mask] * pred

        if not self.learn_scaled:
            zt_tangent = self.velocity_scaling(t)[batch_mask] * zt_tangent

        # exponential map
        return self.exponential_map(zs, zt_tangent)

    def sample_z0(self, batch_mask):
        """ Prior. """

        # Uniform distribution
        z0 = torch.rand((len(batch_mask), self.dim), device=batch_mask.device)

        return 2 * math.pi * z0 - math.pi

    def compute_loss(self, pred, z0, z1, zt, t, batch_mask, reduce='mean'):
        """ Compute loss per sample. """
        assert reduce in {'mean', 'sum', 'none'}
        mask = ~torch.isnan(z1)
        z1 = torch.nan_to_num(z1, nan=0.0)

        zt_dot = self.logarithm_map(z0, z1)
        if self.learn_scaled:
            # NOTE: potentially requires output magnitude to vary substantially
            zt_dot = self.velocity_scaling(t)[batch_mask] * zt_dot
        loss = mask * (pred - zt_dot) ** 2
        loss = torch.sum(loss, dim=-1)

        if reduce == 'mean':
            denom = mask.sum(dim=-1) + 1e-6
            loss = scatter_mean(loss / denom, batch_mask, dim=0)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0)
        return loss


class SO3ICFM(ICFM):
    """
    All rotations are assumed to be in axis-angle format.
    Mostly following descriptions from the FoldFlow paper:
    https://openreview.net/forum?id=kJFIH23hXb

    See also:
    https://geomstats.github.io/_modules/geomstats/geometry/special_orthogonal.html#SpecialOrthogonal
    https://geomstats.github.io/_modules/geomstats/geometry/lie_group.html#LieGroup
    """
    def __init__(self, sigma):
        super().__init__(sigma)

    def exponential_map(self, base, tangent):
        """
        Args:
            base: base point (rotation vector) on the manifold
            tangent: point in tangent space at identity
        Returns:
            rotation vector on the manifold
        """
        # return so3.exp_not_from_identity(tangent, base_point=base)
        return so3.compose_rotations(base, so3.exp(tangent))

    def logarithm_map(self, base, r):
        """
        Args:
            base: base point (rotation vector) on the manifold
            r: rotation vector on the manifold
        Return:
            point in tangent space at identity
        """
        # return so3.log_not_from_identity(r, base_point=base)
        return so3.log(so3.compose_rotations(-base, r))

    def sample_zt(self, z0, z1, t, batch_mask):
        """
        Expressed in terms of exponential and logarithm maps.
        Corresponds to SLERP interpolation: R(t) = R1 exp( t * log(R1^T R2) )
        (see https://lucaballan.altervista.org/pdfs/IK.pdf, slide 16)
        """

        # apply logarithm map
        zt_tangent = t[batch_mask] * self.logarithm_map(z0, z1)

        # apply exponential map
        return self.exponential_map(z0, zt_tangent)

    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """

        # estimate z1_tangent based on zt and pred only
        z1_tangent = (1 - t)[batch_mask] * pred

        # exponential map
        return self.exponential_map(zt, z1_tangent)

    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask):
        """ Perform update, typically using an explicit Euler step. """

        # # parallel transport vector field to lie algebra so3 (at identity)
        # # (FoldFlow paper, Algorithm 3, line 8)
        # # TODO: is this correct? is it necessary?
        # pred = so3.compose(so3.inverse(zs), pred)

        step_size = t - s
        zt_tangent = step_size[batch_mask] * pred

        # exponential map
        return self.exponential_map(zs, zt_tangent)

    def sample_z0(self, batch_mask):
        """ Prior. """
        return so3.random_uniform(n_samples=len(batch_mask), device=batch_mask.device)

    @staticmethod
    def d_R_squared_SO3(rot_vec_1, rot_vec_2):
        """
        Squared Riemannian metric on SO(3).
        Defined as d(R1, R2) = sqrt(0.5) ||log(R1^T R2)||_F
        where R1, R2 are rotation matrices.

        The following is equivalent if the difference between the rotations is
        expressed as a rotation vector \omega_diff:
        d(r1, r2) = ||\omega_diff||_2
        -----
        With the definition of the Frobenius matrix norm ||A||_F^2 = trace(A^H A):
        d^2(R1, R2) = 1/2 ||log(R1^T R2)||_F^2
                    = 1/2 || hat(R_d) ||_F^2
                    = 1/2 tr( hat(R_d)^T hat(R_d) )
                    = 1/2 * 2 * ||\omega||_2^2
        """

        # rot_mat_1 = so3.matrix_from_rotation_vector(rot_vec_1)
        # rot_mat_2 = so3.matrix_from_rotation_vector(rot_vec_2)
        # rot_mat_diff = rot_mat_1.transpose(-2, -1) @ rot_mat_2
        # return torch.norm(so3.log(rot_mat_diff, as_skew=True), p='fro', dim=(-2, -1))

        diff_rot = so3.compose_rotations(-rot_vec_1, rot_vec_2)
        return diff_rot.square().sum(dim=-1)

    def compute_loss(self, pred, z0, z1, zt, t, batch_mask, reduce='mean', eps=5e-2):
        """ Compute loss per sample. """
        assert reduce in {'mean', 'sum', 'none'}

        zt_dot = self.logarithm_map(zt, z1) / torch.clamp(1 - t, min=eps)[batch_mask]

        # TODO: do I need this?
        # pred_at_id = self.logarithm_map(zt, pred) / torch.clamp(1 - t, min=eps)[batch_mask]

        loss = torch.sum((pred - zt_dot)**2, dim=-1)  # TODO: is this the right loss in SO3?
        # loss = self.d_R_squared_SO3(zt_dot, pred)

        if reduce == 'mean':
            loss = scatter_mean(loss, batch_mask, dim=0)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0)

        return loss


#################
# Predicting z1 #
#################

class CoordICFMPredictFinal(CoordICFM):
    def __init__(self, sigma):
        self.dim = 3
        super().__init__(sigma)

    def sample_zt_given_zs(self, zs, z1_minus_zs_pred, s, t, batch_mask):
        """ Perform an explicit Euler step. """

        # step_size = t - s
        # zt = zs + step_size[batch_mask] * z1_minus_zs_pred / (1.0 - s)[batch_mask]

        # for numerical stability
        step_size = (t - s) / (1.0 - s)
        assert torch.all(step_size <= 1.0)
        # step_size = torch.clamp(step_size, max=1.0)
        zt = zs + step_size[batch_mask] * z1_minus_zs_pred
        return zt

    def compute_loss(self, z1_minus_zt_pred, z0, z1, t, batch_mask, reduce='mean'):
        """ Compute loss per sample. """
        assert reduce in {'mean', 'sum', 'none'}
        t = torch.clamp(t, max=0.9)
        zt = self.sample_zt(z0, z1, t, batch_mask)
        loss = torch.sum((z1_minus_zt_pred + zt - z1) ** 2, dim=-1) / torch.square(1 - t)[batch_mask].squeeze()

        if reduce == 'mean':
            loss = scatter_mean(loss / self.dim, batch_mask, dim=0)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0)

        return loss

    def get_z1_given_zt_and_pred(self, zt, z1_minus_zt_pred, z0, t, batch_mask):
        return z1_minus_zt_pred + zt


class TorusICFMPredictFinal(TorusICFM):
    """
    Following:
    Chen, Ricky TQ, and Yaron Lipman.
    "Riemannian flow matching on general geometries."
    arXiv preprint arXiv:2302.03660 (2023).
    """
    def __init__(self, sigma, dim):
        super().__init__(sigma, dim)

    def get_z1_given_zt_and_pred(self, zt, z1_tangent_pred, z0, t, batch_mask):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """

        # exponential map
        return self.exponential_map(zt, z1_tangent_pred)

    def sample_zt_given_zs(self, zs, z1_tangent_pred, s, t, batch_mask):
        """ Perform update, typically using an explicit Euler step. """

        # step_size = t - s
        # zt_tangent = step_size[batch_mask] * z1_tangent_pred / (1.0 - s)[batch_mask]

        # for numerical stability
        step_size = (t - s) / (1.0 - s)
        assert torch.all(step_size <= 1.0)
        # step_size = torch.clamp(step_size, max=1.0)
        zt_tangent = step_size[batch_mask] * z1_tangent_pred

        # exponential map
        return self.exponential_map(zs, zt_tangent)

    def compute_loss(self, z1_tangent_pred, z0, z1, t, batch_mask, reduce='mean'):
        """ Compute loss per sample. """
        assert reduce in {'mean', 'sum', 'none'}
        zt = self.sample_zt(z0, z1, t, batch_mask)
        t = torch.clamp(t, max=0.9)

        mask = ~torch.isnan(z1)
        z1 = torch.nan_to_num(z1, nan=0.0)
        loss = mask * (z1_tangent_pred - self.logarithm_map(zt, z1)) ** 2
        loss = torch.sum(loss, dim=-1) / torch.square(1 - t)[batch_mask].squeeze()

        if reduce == 'mean':
            denom = mask.sum(dim=-1) + 1e-6
            loss = scatter_mean(loss / denom, batch_mask, dim=0)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0)

        return loss
