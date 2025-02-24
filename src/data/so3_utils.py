import math
import torch


def _batch_trace(m):
    return torch.einsum('...ii', m)


def regularize(point, eps=1e-6):
    """
    Norm of the rotation vector should be between 0 and pi.
    Inverts the direction of the rotation axis if the value is between pi and 2 pi.
    Args:
        point, (n, 3)
    Returns:
        regularized point, (n, 3)
    """
    theta = torch.linalg.norm(point, axis=-1)

    # angle in [0, 2pi)
    theta_wrapped = theta % (2 * math.pi)
    inv_mask = theta_wrapped > math.pi

    # angle in [0, pi) & invert
    theta_wrapped[inv_mask] = -1 * (2 * math.pi - theta_wrapped[inv_mask])

    # apply
    theta = torch.clamp(theta, min=eps)
    point = point * (theta_wrapped / theta).unsqueeze(-1)
    assert not point.isnan().any()
    return point


def random_uniform(n_samples, device=None):
    """
    Follow geomstats implementation:
    https://geomstats.github.io/_modules/geomstats/geometry/special_orthogonal.html

    Args:
        n_samples: int
    Returns:
        rotation vectors, (n, 3)
    """
    random_point = (torch.rand(n_samples, 3, device=device) * 2 - 1) * math.pi
    random_point = regularize(random_point)

    return random_point


def hat(rot_vec):
    """
    Maps R^3 vector to a skew-symmetric matrix r (i.e. r \in R^{3x3} and r^T = -r).
    Since we have the identity rv = rot_vec x v for all v \in R^3, this is
    identical to a cross-product-matrix representation of rot_vec.
    rot_vec x v = hat(rot_vec)^T v
    See also:
    https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
    https://en.wikipedia.org/wiki/Hat_notation#Cross_product
    Args:
        rot_vec: (n, 3)
    Returns:
        skew-symmetric matrices (n, 3, 3)
    """
    basis = torch.tensor([
        [[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
        [[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]],
        [[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]]
    ], device=rot_vec.device)
    # basis = torch.tensor([
    #     [[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]],
    #     [[0., 0., -1.], [0., 0., 0.], [1., 0., 0.]],
    #     [[0., 1., 0.], [-1., 0., 0.], [0., 0., 0.]]
    # ], device=rot_vec.device)

    return torch.einsum('...i,ijk->...jk', rot_vec, basis)


def inv_hat(skew_mat):
    """
    Inverse of hat operation
    Args:
        skew_mat: skew-symmetric matrices (n, 3, 3)
    Returns:
        rotation vectors, (n, 3)
    """

    assert torch.allclose(-skew_mat, skew_mat.transpose(-2, -1), atol=1e-4), \
        f"Input not skew-symmetric (err={(-skew_mat - skew_mat.transpose(-2, -1)).abs().max():.4g})"

    # vec = torch.stack([
    #     skew_mat[:, 1, 2],
    #     skew_mat[:, 2, 1],
    #     skew_mat[:, 0, 1]
    # ], dim=1)

    vec = torch.stack([
        skew_mat[:, 2, 1],
        skew_mat[:, 0, 2],
        skew_mat[:, 1, 0]
    ], dim=1)

    return vec


def matrix_from_rotation_vector(axis_angle, eps=1e-6):
    """
    Args:
        axis_angle: (n, 3)
    Returns:
        rotation matrices, (n, 3, 3)
    """

    axis_angle = regularize(axis_angle)
    angle = axis_angle.norm(dim=-1)
    _norm = torch.clamp(angle, min=eps).unsqueeze(-1)
    skew_mat = hat(axis_angle / _norm)

    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    _id = torch.eye(3, device=axis_angle.device).unsqueeze(0)
    rot_mat = _id + \
              torch.sin(angle)[:, None, None] * skew_mat + \
              (1 - torch.cos(angle))[:, None, None] * torch.bmm(skew_mat, skew_mat)

    return rot_mat


class safe_acos(torch.autograd.Function):
    """
    Implementation of arccos that avoids NaN in backward pass.
    https://github.com/pytorch/pytorch/issues/8069#issuecomment-2041223872
    """
    EPS = 1e-4
    @classmethod
    def d_acos_dx(cls, x):
        x = torch.clamp(x, min=-1. + cls.EPS, max=1. - cls.EPS)
        return -1.0 / (1 - x**2).sqrt()

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.acos()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * safe_acos.d_acos_dx(input)


def rotation_vector_from_matrix(rot_mat, approx=1e-4):
    """
    Args:
        rot_mat: (n, 3, 3)
        approx: float, minimum angle below which an approximation will be used
            for numerical stability
    Returns:
        rotation vector, (n, 3)
    """

    # https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_rotation_matrix_to_axis%E2%80%93angle
    # https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Log_map_from_SO(3)_to_%F0%9D%94%B0%F0%9D%94%AC(3)

    # determine axis
    skew_mat = rot_mat - rot_mat.transpose(-2, -1)

    # determine the angle
    cos_angle = 0.5 * (_batch_trace(rot_mat) - 1)
    # arccos is only defined between -1 and 1
    assert torch.all(cos_angle.abs() <= 1 + 1e-6)
    cos_angle = torch.clamp(cos_angle, min=-1., max=1.)
    # abs_angle = torch.arccos(cos_angle)
    abs_angle = safe_acos.apply(cos_angle)

    # avoid numerical instability; use sin(x) \approx x for small x
    close_to_0 = abs_angle < approx
    _fac = torch.empty_like(abs_angle)
    _fac[close_to_0] = 0.5
    _fac[~close_to_0] = 0.5 * abs_angle[~close_to_0] / torch.sin(abs_angle[~close_to_0])

    axis_angle = inv_hat(_fac[:, None, None] * skew_mat)
    return regularize(axis_angle)


def get_jacobian(point, left=True, inverse=False, eps=1e-4):

    # # From Geomstats: https://geomstats.github.io/_modules/geomstats/geometry/special_orthogonal.html
    # jacobian = so3_vector.jacobian_translation(point, left)
    #
    # if inverse:
    #     jacobian = torch.linalg.inv(jacobian)

    # Right Jacobian defined as J_r(theta) = \partial exp([theta]_x) / \partial theta
    # https://math.stackexchange.com/questions/301533/jacobian-involving-so3-exponential-map-logr-expm
    # Source:
    # Chirikjian, Gregory S. Stochastic models, information theory, and Lie
    # groups, volume 2: Analytic methods and modern applications. Vol. 2.
    # Springer Science & Business Media, 2011. (page 40)
    # NOTE: the definitions of 'inverse' and 'left' in the book are the opposite
    #  of their meanings in Geomstats, whose functionality we're mimicking here.
    #  This explains the differences in the equations.
    angle_squared = point.square().sum(-1)
    angle = angle_squared.sqrt()
    skew_mat = hat(point)

    assert torch.all(angle <= math.pi)
    close_to_0 = angle < eps
    close_to_pi = (math.pi - angle) < eps

    angle = angle[:, None, None]
    angle_squared = angle_squared[:, None, None]

    if inverse:
        # _jacobian = torch.eye(3, device=point.device).unsqueeze(0) + \
        #            (1 - torch.cos(angle)) / angle_squared * skew_mat + \
        #            (angle - torch.sin(angle)) / angle ** 3 * (skew_mat @ skew_mat)

        _term1 = torch.empty_like(angle)
        _term1[close_to_0] = 0.5  # approximate with value at zero
        _term1[~close_to_0] = (1 - torch.cos(angle)) / angle_squared

        _term2 = torch.empty_like(angle)
        _term2[close_to_0] = 1 / 6  # approximate with value at zero
        _term2[~close_to_0] = (angle - torch.sin(angle)) / angle ** 3

        jacobian = torch.eye(3, device=point.device).unsqueeze(0) + \
                   _term1 * skew_mat + _term2 * (skew_mat @ skew_mat)
        # assert torch.allclose(jacobian, _jacobian, atol=1e-4)
    else:
        # _jacobian = torch.eye(3, device=point.device).unsqueeze(0) - 0.5 * skew_mat + \
        #            (1 / angle_squared - (1 + torch.cos(angle)) / (2 * angle * torch.sin(angle))) * (skew_mat @ skew_mat)

        _term1 = torch.empty_like(angle)
        _term1[close_to_0] = 1 / 12  # approximate with value at zero
        _term1[close_to_pi] = 1 / math.pi**2  # approximate with value at pi
        default = ~close_to_0 & ~close_to_pi
        _term1[default] = 1 / angle_squared[default] - \
                        (1 + torch.cos(angle[default])) / (2 * angle[default] * torch.sin(angle[default]))

        jacobian = torch.eye(3, device=point.device).unsqueeze(0) - \
                    0.5 * skew_mat + _term1 * (skew_mat @ skew_mat)
        # assert torch.allclose(jacobian, _jacobian, atol=1e-4)

    if left:
        jacobian = jacobian.transpose(-2, -1)

    return jacobian


def compose_rotations(rot_vec_1, rot_vec_2):
    rot_mat_1 = matrix_from_rotation_vector(rot_vec_1)
    rot_mat_2 = matrix_from_rotation_vector(rot_vec_2)
    rot_mat_out = torch.bmm(rot_mat_1, rot_mat_2)
    return rotation_vector_from_matrix(rot_mat_out)


def exp(tangent):
    """
    Exponential map at identity.
    Args:
        tangent: vector on the tangent space, (n, 3)
    Returns:
        rotation vector on the manifold, (n, 3)
    """
    # rotations are already represented by rotation vectors
    exp_from_identity = regularize(tangent)
    return exp_from_identity


def exp_not_from_identity(tangent_vec, base_point):
    """
    Exponential map at base point.
    Args:
        tangent_vec: vector on the tangent plane, (n, 3)
        base_point: base point on the manifold, (n, 3)
    Returns:
        new point on the manifold, (n, 3)
    """

    tangent_vec = regularize(tangent_vec)
    base_point = regularize(base_point)

    # Lie algebra is the tangent space at the identity element of a Lie group
    # -> to identity
    jacobian = get_jacobian(base_point, left=True, inverse=True)
    tangent_vec_at_id = torch.einsum("...ij,...j->...i", jacobian, tangent_vec)

    # exponential map from identity
    exp_from_identity = exp(tangent_vec_at_id)

    # -> back to base point
    return compose_rotations(base_point, exp_from_identity)


def log(rot_vec, as_skew=False):
    """
    Logarithm map from tangent space at the identity.
    Args:
        rot_vec: point on the manifold, (n, 3)
    Returns:
        vector on the tangent space, (n, 3)
    """
    # rotations are already represented by rotation vectors
    # log_from_id = regularize(rot_vec)
    log_from_id = rot_vec
    if as_skew:
        log_from_id = hat(log_from_id)
    return log_from_id


def log_not_from_identity(point, base_point):
    """
    Logarithm map of point from base point.
    Args:
        point: point on the manifold, (n, 3)
        base_point: base point on the manifold, (n, 3)
    Returns:
        vector on the tangent plane, (n, 3)
    """
    point = regularize(point)
    base_point = regularize(base_point)

    inv_base_point = -1 * base_point

    point_near_id = compose_rotations(inv_base_point, point)

    # logarithm map from identity
    log_from_id = log(point_near_id)

    jacobian = get_jacobian(base_point, inverse=False)
    tangent_vec_at_id = torch.einsum("...ij,...j->...i", jacobian, log_from_id)

    return tangent_vec_at_id


if __name__ == "__main__":

    import os
    os.environ['GEOMSTATS_BACKEND'] = "pytorch"
    import scipy.optimize  # does not seem to be imported correctly when just loading geomstats
    default_dtype = torch.get_default_dtype()
    from geomstats.geometry.special_orthogonal import SpecialOrthogonal
    torch.set_default_dtype(default_dtype)  # Geomstats changes default type when imported

    so3_vector = SpecialOrthogonal(n=3, point_type="vector")

    # decorator
    if torch.__version__ >= '2.0.0':
        GEOMSTATS_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        def geomstats_tensor_type(func):
            def inner(*args, **kwargs):
                with torch.device(GEOMSTATS_DEVICE):
                    out = func(*args, **kwargs)
                return out

            return inner
    else:
        GEOMSTATS_TENSOR_TYPE = 'torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor'

        # GEOMSTATS_TENSOR_TYPE = 'torch.cuda.DoubleTensor' if torch.cuda.is_available() else 'torch.DoubleTensor'
        def geomstats_tensor_type(func):
            def inner(*args, **kwargs):
                # tensor_type_before = TODO
                torch.set_default_tensor_type(GEOMSTATS_TENSOR_TYPE)
                out = func(*args, **kwargs)
                # torch.set_default_tensor_type(tensor_type_before)
                torch.set_default_tensor_type('torch.FloatTensor')
                return out

            return inner

    @geomstats_tensor_type
    def gs_matrix_from_rotation_vector(*args, **kwargs):
        return so3_vector.matrix_from_rotation_vector(*args, **kwargs)

    @geomstats_tensor_type
    def gs_rotation_vector_from_matrix(*args, **kwargs):
        return so3_vector.rotation_vector_from_matrix(*args, **kwargs)

    @geomstats_tensor_type
    def gs_exp_not_from_identity(*args, **kwargs):
        return so3_vector.exp_not_from_identity(*args, **kwargs)

    @geomstats_tensor_type
    def gs_log_not_from_identity(*args, **kwargs):
        # norm of the rotation vector will be between 0 and pi
        return so3_vector.log_not_from_identity(*args, **kwargs)

    @geomstats_tensor_type
    def compose(*args, **kwargs):
        return so3_vector.compose(*args, **kwargs)

    @geomstats_tensor_type
    def inverse(*args, **kwargs):
        return so3_vector.inverse(*args, **kwargs)

    @geomstats_tensor_type
    def gs_random_uniform(*args, **kwargs):
        return so3_vector.random_uniform(*args, **kwargs)


    #############
    # RUN TESTS #
    #############

    n = 16
    device = 'cuda' if torch.cuda.is_available() else None

    ### regularize ###

    # vec = (torch.rand(n, 3) * 2 - 1) * math.pi
    vec = (torch.rand(n, 3) * 4 - 2) * math.pi
    axis_angle = regularize(vec)
    assert torch.all(torch.cross(vec, axis_angle).norm(dim=-1) < 1e-5), "not all vectors collinear"
    assert torch.all(axis_angle.norm(dim=-1) < math.pi) & torch.all(axis_angle.norm(dim=-1) >= 0), "norm not between 0 and pi"


    ### matrix_from_rotation_vector ###

    rot_vec = random_uniform(16, device=device)
    assert torch.allclose(matrix_from_rotation_vector(rot_vec),
                          gs_matrix_from_rotation_vector(rot_vec), atol=1e-06)


    ### rotation_vector_from_matrix ###

    rot_vec = random_uniform(16, device=device)
    rot_mat = matrix_from_rotation_vector(rot_vec)
    assert torch.allclose(rotation_vector_from_matrix(rot_mat),
                          gs_rotation_vector_from_matrix(rot_mat), atol=1e-05)


    ### exp_not_from_identity ###

    tangent_vec = random_uniform(16, device=device)
    base_pt = random_uniform(16, device=device)
    my_val = exp_not_from_identity(tangent_vec, base_pt)
    gs_val = gs_exp_not_from_identity(tangent_vec, base_pt)
    assert torch.allclose(my_val, gs_val, atol=1e-03), (my_val - gs_val).abs().max()


    ### log_not_from_identity ###

    pt = random_uniform(16, device=device)
    base_pt = random_uniform(16, device=device)
    my_val = log_not_from_identity(pt, base_pt)
    gs_val = gs_log_not_from_identity(pt, base_pt)
    assert torch.allclose(my_val, gs_val, atol=1e-03), (my_val - gs_val).abs().max()


    print("All tests successful!")
