import time

import cv2
import numpy as np
import torch


def time_it(func):
    """
    A time logger, compute how much time was cost. Use as a decorator (@time_it).
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print('[TIMELOG] function \'{}\' cost time: {:.2f} ms.'.format(func.__name__, (end - start) * 1000))
        return result
    return wrapper


def crop_and_resize_square(img, out_size):
    """
    Crop the biggest square of the photo, which centered at the center of original photo,
    then resize it to the given size.
    Args:
        img(np.ndarray): shape of (H, W, C)
        out_size(int):
    Returns:
        np.ndarray: shape of (out_size, out_size, C)
    """
    h, w = img.shape[0], img.shape[1]
    if h < w:
        img = img[:, (w - h) // 2: (w + h) // 2, ...]
    else:
        img = img[(h - w) // 2: (h + w) // 2, ...]
    _out = cv2.resize(img, (out_size, out_size))
    return _out


def pad_and_resize_square(img, out_size):
    """ Pad the original image to square than resize it to the out_size
    The background is set to black by default
    Args:
        img (np.ndarray): shape of (H, W, C)
        out_size (int):
    Returns:
        np.ndarray: shape of (out_size, out_size, C)
    """
    if len(img.shape) == 3:    
        h, w, c = img.shape
        _out = np.zeros((out_size, out_size, c))
    elif len(img.shape) == 2:
        h, w = img.shape
        _out = np.zeros((out_size, out_size))
    else:
        raise 
    if h < w:
        newh = h * out_size // w 
        _out[(out_size - newh) // 2: (out_size + newh) // 2, ...] = cv2.resize(img, (w, newh))
    else:
        neww = w * out_size // h
        _out[:, (out_size - neww) // 2: (out_size + neww) // 2, ...] = cv2.reize(img, (neww, h))
    return _out


def make_grayscale(img):
    """
    Args:
        img (np.ndarray): shape of (H, W, 3)
    Returns:
        np.ndarray: shape of (H, W), non-[0, 0, 0] are set to 1
    """
    assert len(img.shape) == 3 and img.shape[2] == 3
    _out = np.zeros((img.shape[0], img.shape[1])).astype(np.float32)
    _out[img.sum(2) != 0] = 1
    return _out


def make_channels(img, conditions):
    """
    Convert the given image to a multichannel np.ndarray
    Args:
        img(np.ndarray): shape of (C=3, H, W)
        conditions (list of lambda): each element is a channel condition,
            which returns a (H, W) bool array.
    Returns:
        np.ndarray: dtype=np.float32, shape of (len(conditions), H, W), value of 0/1
    """
    return np.asarray([c(img) for c in conditions]).astype(np.float32)


def make_colorful(img, colors):
    """
    Convert the one-hot image to 3-colored image, if all channel are 0, set color to (0, 0, 0).
    Args:
        img(np.ndarray): shape of (C, H, W)
        colors(list of (tuple of int)): value should be 0~255. Each tuple is BGR color, colorize each channel.
    Returns:
        np.ndarray: shape(H, W, C=3), dtype=int
    """
    out = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        out[(img.argmax(0) == i) & (img[i] != 0)] = c
    return out


def cosine_similarity(x, y, dim=0):
    """
    Function like torch.nn.CosineSimilarity, but on np.ndarray.
    Args:
        x (np.ndarray): shape should be
        y (np.ndarray)
        dim (int): where to compute the similarity
    Return:
        np.ndarray: between [-1, 1], the greater the more similar, preserve dimensions other than dim.
    """
    assert x.shape == y.shape
    assert dim < len(x.shape)
    cos_angle = (x * y).sum(dim)
    cos_angle /= (x ** 2).sum(dim) ** 0.5
    cos_angle /= (y ** 2).sum(dim) ** 0.5
    return cos_angle


def largest_connected_area(img):
    """
    TODO: doesn't work.
    Reserve the largest component in each channel
    Args:
        img (np.ndarray): shape of (C, H, W), each channel is 0-1
    Returns:

    """
    out = np.zeros_like(img).astype(np.uint8)
    for i in range(len(img)):
        ch = (img[i] * 255).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ch, connectivity=8)
        out[i, labels == (stats[1:, cv2.CC_STAT_AREA].argmax() + 1)] = 1.0
    return out


def tensor_to_cv2(t):
    """
    Convert a torch.Tensor to a np.ndarray for cv2.imwrite.
    Args:
        t(torch.Tensor): float, 0-1, shape of (C, H, W)
    Returns:
        np.ndarray: dtype=int, 0-255, shape of (H, W, C(BGR))
    """
    t = t.detach().cpu().numpy() * 255
    if len(t.shape) == 3:
        t = t.transpose((1, 2, 0))
    t = t.astype(np.uint8)
    return t


def cv2_to_tensor(a):
    """
    Convert an image read by cv2 to a float tensor.
    Args:
        a(np.ndarray): int, shape of (H, W, C(BGR)) or (H, W)
    Returns:
        torch.Tensor: dtype=float32, 0-1, shape of (C, H, W)
    """
    a = a.astype(float) / 255.0
    if len(a.shape) == 3:
        a = a.transpose((2, 0, 1))
    a = torch.from_numpy(a).float()
    return a


def images_alpha_lighten(under, upper, alpha):
    """
    Lighten blend target with blend. Like Lighten + alpha=50% in Photoshop.
    Args:
        target: np.ndarray of (H, W, 3), 0-255 or torch.Tensor of (3, H, W), 0-1
        upper: np.ndarray of (H, W, 3), 0-255 or torch.Tensor of (3, H, W), 0-1
        alpha (float):
    Returns:
        np.ndarray:
    """
    if isinstance(under, np.ndarray) and isinstance(upper, np.ndarray):
        under_t = under.transpose((2, 0, 1))
        upper_t = upper.transpose((2, 0, 1))
        blended = alpha * upper_t + (1 - alpha) * under_t
        brighter = upper.sum(-1) > 0.01 * 255
        out = brighter * blended + (1 - brighter) * under_t
        out = out.transpose((1, 2, 0))
    elif isinstance(under, torch.Tensor) and isinstance(upper, torch.Tensor):
        blended = alpha * upper + (1 - alpha) * under
        brighter = upper.sum(0) > 0.01
        out = brighter * blended + (~brighter) * under
    return out


def quaternion_from_view_up(orig_view, orig_up, new_view, new_up):
    """
    Ideas of following quaternion related functions are from https://krasjet.github.io/quaternion/quaternion.pdf
    Args:
        orig_view (np.ndarray or list of float): dtype=float
        orig_up (np.ndarray or list of float): dtype=float
        new_view (np.ndarray or list of float): dtype=float
        new_up (np.ndarray or list of float): dtype=float
    Returns:
        (np.ndarray), a quaternion.
    """
    if isinstance(new_view, list):
        new_view = np.asarray(new_view).astype(float)
    if isinstance(new_up, list):
        new_up = np.asarray(new_up).astype(float)
    if isinstance(orig_view, list):
        orig_view = np.asarray(orig_view).astype(float)
    if isinstance(orig_up, list):
        orig_up = np.asarray(orig_up).astype(float)

    orig_view /= np.linalg.norm(orig_view)
    orig_up = np.cross(np.cross(orig_view, orig_up), orig_view)
    orig_up /= np.linalg.norm(orig_up)
    new_view /= np.linalg.norm(new_view)
    new_up = np.cross(np.cross(new_view, new_up), new_view)
    new_up /= np.linalg.norm(new_up)

    q0 = quaternion_from_2vec(orig_view, new_view, axis_if_oppo=(orig_up + orig_up) / 2)
    mid_up = quaternion_rotate_vec(orig_up, q0)
    q1 = quaternion_from_2vec(mid_up, new_up, axis_if_oppo=new_view)
    q = quaternion_mul(q1, q0)
    return q


def quaternion_from_2ob(u0, v0, u1, v1):
    """
    Compute the quaternion from 2 pairs of orthogonal basis.
    That is to say, rotate u0 to u1 and v0 to v1 simultaneously.
    TODO: try to do it one-step
    Args:
        u0 (np.ndarray): dtype=float, must be unit vector, must be orthogonal with v0.
        v0 (np.ndarray): dtype=float, must be unit vector, must be orthogonal with u0.
        u1 (np.ndarray): dtype=float, must be unit vector, must be orthogonal with v1.
        v1 (np.ndarray): dtype=float, must be unit vector, must be orthogonal with u1.
    Returns:
        (np.ndarray), a quaternion.
    """
    if np.dot(u0, u1) == np.dot(v0, v1) == -1.0:
        axis = np.cross(u0, v0)
        return np.asarray([
            0,
            axis[0],
            axis[1],
            axis[2]
        ])
    if np.dot(u0, u1) == 1.0:
        return quaternion_from_2vec(v0, v1, axis_if_oppo=u0)
    elif np.dot(u0, u1) == -1.0:
        return quaternion_from_2vec(u0, u1, axis_if_oppo=(v0 + v1) / 2)
    elif np.dot(v0, v1) == 1.0:
        return quaternion_from_2vec(u0, u1, axis_if_oppo=v0)
    elif np.dot(v0, v1) == -1.0:
        return quaternion_from_2vec(v0, v1, axis_if_oppo=(u0 + u1) / 2)
    else:
        nu = np.cross(np.cross(u0, u1), (u0 + u1) / 2)
        nv = np.cross(np.cross(v0, v1), (v0 + v1) / 2)
        axis = (u0 + u1) / 2
        axis /= np.linalg.norm(axis)
        return quaternion_from_2vec(nu, nv, axis_if_oppo=axis)


def quaternion_from_2vec(v0, v1, axis_if_oppo=None):
    """
    Compute a quaternion that rotate first vector to the other.
    Two special condition: v0 == v1, and v0 == -v1.
    For the first case, return
    Args:
        v0 (np.ndarray): shape of (3,), original vector.
        v1 (np.ndarray): shape of (3,), new vector.
        axis_if_oppo (np.ndarray): shape of (3,)
    Returns:
        np.ndarray: shape of (4,), a quaternion.
    """
    if np.linalg.norm(v0) == 0 or np.linalg.norm(v1) == 0:
        raise RuntimeError('The 2 vectors must not be zero vector!')
    cos_theta = np.dot(v0, v1) / np.linalg.norm(v0) / np.linalg.norm(v1)
    # TODO due to the precision, the theta is sometimes wrong (> 1.0 or < -1.0)
    cos_theta = np.clip(cos_theta, a_min=-1.0, a_max=1.0)
    if cos_theta == 1.0:
        axis = np.zeros_like(v0)
    elif cos_theta == -1.0:
        if axis_if_oppo is not None and np.linalg.norm(axis_if_oppo) != 0:
            axis = axis_if_oppo / np.linalg.norm(axis_if_oppo)
        else:
            raise RuntimeError('For opposite vectors, must input a unit vector as axis!')
    else:
        axis = np.cross(v0, v1)
        axis /= np.linalg.norm(axis)
    cos_half = ((cos_theta + 1) / 2) ** 0.5
    sin_half = ((1 - cos_theta) / 2) ** 0.5
    return np.asarray([
        cos_half,
        sin_half * axis[0],
        sin_half * axis[1],
        sin_half * axis[2]
    ])


def quaternion_rotate_vec(v, q):
    """
    Rotate a vector by a quaternion.
    If the quaternion is unit, it can be written as [cos(\theta / 2), sin(\theta / 2) u], where u is a vector.
    It means the vector v is rotated according to the axis u, and by angle of \theta.
    Args:
        v (np.ndarray): shape of (3,), a vector
        q (np.ndarray): shape of (4,), a quaternion
    Returns:
        np.ndarray: shape of (3,), a new vector.
    """
    assert v.shape == (3,)
    assert q.shape == (4,)
    v_ = np.asarray([0.0, v[0], v[1], v[2]])
    q_ = np.asarray([q[0], -q[1], -q[2], -q[3]])
    return quaternion_mul(q, quaternion_mul(v_, q_))[1:]


def quaternion_mul(q0, q1):
    """
    Multiply the first quaternion with the second. Note that it's not exchangeable.
    Args:
        q0 (np.ndarray): shape of (4,)
        q1 (np.ndarray): shape of (4,)
    Returns:
        np.ndarray: shape of (4,)
    """
    assert q0.shape == (4,)
    assert q1.shape == (4,)
    a, b, c, d = q0
    e, f, g, h = q1
    return np.asarray([
        a*e - b*f - c*g - d*h,
        b*e + a*f + c*h - d*g,
        c*e + a*g + d*f - b*h,
        d*e + a*h + b*g - c*f
    ])


def quaternion_to_matrix(q):
    """
    Convert a quaternion to a left matrix.
    Args:
        q (np.ndarray): shape of (4,)
    Returns:
        np.ndarray: shape of (3, 3)
    """
    a, b, c, d = q
    return np.asarray([
        [1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d],
        [2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b],
        [2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, 1 - 2 * b * b - 2 * c * c]
    ])


def matrix_from_7_parameters(param):
    """

    Args:
        param: [Qw, Qx, Qy, Qz, Tx, Ty, Tz]
    Returns:
        np.ndarray: shape of (4, 4)
    """
    quaternion = param[:4]
    translation = param[4:]
    a, b, c, d = quaternion
    x, y, z = translation
    return np.asarray([
        [1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d, x],
        [2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b, y],
        [2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d, 1 - 2 * b * b - 2 * c * c, z],
        [0, 0, 0, 1]
    ])


if __name__ == '__main__':
    view0 = np.asarray([0, 0, -1], dtype=float)
    up0 = np.asarray([0, 1, 0], dtype=float)
    view1 = np.asarray([
            -2.870878746927293e-17,
            -0.46885007970071213,
            2.6589809331329617
    ], dtype=float)
    up1 = np.asarray([0, 0, 1], dtype=float)
    qq = quaternion_from_view_up(
        view0, up0, view1, up1
    )
    view1_ = quaternion_rotate_vec(view0, qq)
    up1_ = quaternion_rotate_vec(up0, qq)
    print()
