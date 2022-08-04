import torch

HAT_INV_SKEW_SYMMETRIC_TOL = 1e-5 # 허용치(유효범위)
# tensor : 계산 단순화를 위해 같은 성질의 여러 벡터를 한 행렬에 표시한 것

def allclose(mat1, mat2, tol=1e-6):
    # 두 텐서(transformation matrix, screw axis 등)의 모든 요소가 tol과 가까운지 확인하는 함수

    '''
    check is all elements of two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar
    '''
    return isclose(mat1, mat2, tol).all()
    # all : 모든 요소가 참이면 참 반환


def isclose(mat1, mat2, tol=1e-6):
    # 두 텐서의 각각의 요소에 대해  tol과 가까운지 확인
    """
    check element-wise if two tensors are close within some tolerance.

    Either tensot can be replaced by a scalar
    """
    return (mat1 - mat2).abs_().lt(tol)
    # 각 매트릭스의 차의 절댓값이 tol보다 작으면 참 반환

def outer(vecs1, vecs2):
    # 외적 계산이 목적

    # squeeze : (A x B x 1 x C x 1) 형태의 tensor에서 차원이 1인 부분을 제거하여 (A x B x C) 형태로 만들어 주는 것
    # unsqueeze : 지정한 dimension 자리에 사이즈가 1인 빈 공간을 채워주며 차원 확장

    # shape : 해당 차원의 사이즈 확인

    """Return the N x D x D outer products of a N x D batch of vectors,
    or return the D x D outer product of two D-dimensional vectors.
    """
    # Default batch size is 1
    if vecs1.dim() < 2:
        vecs1 = vecs1.unsqueeze(dim=0)

    if vecs2.dim() < 2:
        vecs2 = vecs2.unsqueeze(dim=0)

    if vecs1.shape[0] != vecs2.shape[0]:
        raise ValueError("Got inconsistent batch sizes {} and {}".format(
            vecs1.shape[0], vecs2.shape[0]))

    return torch.bmm(vecs1.unsqueeze(dim=2),
                     vecs2.unsqueeze(dim=2).transpose(2, 1)).squeeze_()
    # torch.bmm : batch matrix multiplication을 수행하는 함수로, 2개 이상의 차원을 지닌 텐서가 
    # 주어졌을 때 뒤의 2개 차원에 대해 행렬 곱을 수행하고 앞의 다른 차원은 미니배치로 취급한다. 
    # 따라서 앞의 차원들은 크기가 같아야 하고, 뒤의 2개 차원은 행렬 곱을 수행하기 위한 적절한 
    # 크기를 지녀야 한다. 

    # transpose : 두 개의 차원 교환



def trace(mat):
    # 정사각행렬의 주 대각성분의 합을 반환
    """Return the N traces of a batch of N square matrices,
    or return the trace of a square matrix."""
    # Default batch size is 1
    if mat.dim() < 3:
        mat = mat.unsqueeze(dim=0)

    # Element-wise multiply by identity and take the sum
    tr = (torch.eye(mat.shape[1], dtype=mat.dtype) * mat).sum(dim=1).sum(dim=1)
    #  eye : 단위 행렬 생성

    return tr.view(mat.shape[0])
    # view : 행렬로 결과를 보여줌


def matR_log_map(R, eps: float = 1e-4, cos_angle: bool = False):
    # rotation matrix에서 축 각도(lie algebra parameters) 반환

    """
    Returns the axis angle aka lie algebra parameters from a rotation matrix(SO3)
    """
    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = trace(R)

    if((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError(
            "A matrix has trace outside valid range [-1-eps, 3+eps]."
        )

    # clamp to valid range
    # clamp : 지정한 텐서를 대상으로 제안한 최솟값보다 작은 값은 최솟값으로, 최댓값보다 큰 값은 
    # 최댓값으로 변경

    rot_trace = torch.clamp(rot_trace, -1.0, 3.0)

    # phi ... rotation angle
    phi = 0.5 * (rot_trace - 1.0)
    phi = phi.acos()

    phi_valid = torch.clamp(phi.abs(), eps) * phi.sign()
    # sign : 숫자가 음수면 -1, 양수면 1, 0이면 0 반환

    log_rot_hat = (phi_valid / (2.0 * phi_valid.sin()))[:, None, None] * (
        R - R.permute(0, 2, 1)
    )
    # permute : 모든 차원의 순서 교환 가능
    log_rot = hat_inv(log_rot_hat)
    return log_rot


def lie_u_v(u, v, eps: float = 1e-4):
    # 단위벡터 u를 v로 회전시키기 위한 각도 매개변수를 찾음

    """
    find the axis angle parameters to rotate unit vector u onto unit vector v
    which is also the lie algebra parameters of corresponding SO3
    """
    w = torch.cross(u, v, dim=1)

    w_norm = torch.norm(w, p=2, dim=1)
    w_norm = torch.clamp(w_norm, eps)
    A = w / w_norm[:, None] * torch.mul(u, v).sum(dim=1).acos()[:, None]
    return A


def lie_exp_map(log_rot, eps: float = 1e-4):
    # 축 각도를 rotation matrix로 변환
    """
    Convert the lie algebra parameters to rotation matrices
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    # 로드리게스 포뮬라 적용
    skews = hat(log_rot)
    R = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * torch.bmm(skews, skews)
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )
    # print(R.shape)
    return R


def hat(v):
    # 3d 벡터들로 skew symmetric matrix 계산
    """
    compute the skew-symmetric matrices with a batch of 3d vectors.
    """
    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")
    h = v.new_zeros(N, 3, 3)
    x, y, z = v.unbind(1)
    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def hat_inv(h):
    # skew symmetric matrix로 3d 벡터 계산
    """
    compute the 3d-vectors with a batch of 3x3 skew-symmetric matrics.
    """
    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    ss_diff = (h + h.permute(0, 2, 1)).abs().max()
    if float(ss_diff) > HAT_INV_SKEW_SYMMETRIC_TOL:
        raise ValueError("One of input matrices not skew-symmetric.")

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v
