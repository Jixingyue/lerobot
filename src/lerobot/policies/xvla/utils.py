import math

import numpy as np


def mat2quat(rmat):
    """
    将给定的旋转矩阵转换为四元数。

    Args:
        rmat (np.array): 3x3 旋转矩阵

    Returns:
        np.array: (x,y,z,w) float 类型的四元数
    """
    mat = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = mat[0, 0]
    m01 = mat[0, 1]
    m02 = mat[0, 2]
    m10 = mat[1, 0]
    m11 = mat[1, 1]
    m12 = mat[1, 2]
    m20 = mat[2, 0]
    m21 = mat[2, 1]
    m22 = mat[2, 2]
    # 对称矩阵 k
    k = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    k /= 3.0
    # 四元数是 k 的最大特征值所对应的特征向量
    w, v = np.linalg.eigh(k)
    inds = np.array([3, 0, 1, 2])
    q1 = v[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


def quat2axisangle(quat):
    """
    将四元数转换为轴角（axis-angle）格式。
    返回一个按其角度（弧度）缩放的单位向量方向。

    Args:
        quat (np.array): (x,y,z,w) vec4 float 类型的四元数

    Returns:
        np.array: (ax,ay,az) 轴角指数坐标
    """
    # 裁剪四元数
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # 这是（接近）零度旋转，立即返回
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def rotate6d_to_axis_angle(r6d):
    """
    r6d: np.ndarray，形状 (N, 6)
    return: np.ndarray，形状 (N, 3)，轴角向量
    """
    flag = 0
    if len(r6d.shape) == 1:
        r6d = r6d[None, ...]
        flag = 1

    a1 = r6d[:, 0:3]
    a2 = r6d[:, 3:6]

    # b1
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-6)

    # b2
    dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2_orth = a2 - dot_prod * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + 1e-6)

    # b3
    b3 = np.cross(b1, b2, axis=-1)

    rotation_matrix = np.stack([b1, b2, b3], axis=-1)  # shape: (N, 3, 3)

    axis_angle_list = []
    for i in range(rotation_matrix.shape[0]):
        quat = mat2quat(rotation_matrix[i])
        axis_angle = quat2axisangle(quat)
        axis_angle_list.append(axis_angle)

    axis_angle_array = np.stack(axis_angle_list, axis=0)  # shape: (N, 3)

    if flag == 1:
        axis_angle_array = axis_angle_array[0]

    return axis_angle_array


def mat_to_rotate6d(abs_action):
    if len(abs_action.shape) == 2:
        return np.concatenate([abs_action[:3, 0], abs_action[:3, 1]], axis=-1)
    elif len(abs_action.shape) == 3:
        return np.concatenate([abs_action[:, :3, 0], abs_action[:, :3, 1]], axis=-1)
    else:
        raise NotImplementedError


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """逐样本丢弃路径（Stochastic Depth，随机深度）（应用于残差块的主路径时）。

    这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，但是原来的名字有误导性，
    因为 'Drop Connect' 是另一篇论文中另一种形式的 dropout……
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ……我选择
    将层名和参数名改为 'drop path'，而不是把 DropConnect 作为层名、用
    'survival rate' 作为参数名。

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适用于不同维度的张量，而不仅仅是 2D ConvNet
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor
