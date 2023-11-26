"""
定义一些常见的 kernel
"""

import numpy as np


def kernel_linear():
    def _linear(x, y):
        return np.dot(x, y)

    return _linear


def kernel_poly(p=2):
    """
    多项式核函数
    :param p:
    :return:
    """
    def _poly(x, y):
        return np.power(np.dot(x, y) + 1, p)

    return _poly


def kernel_rbf(sigma=0.1):
    """
    高斯径向基核函数
    :param sigma:
    :return:
    """
    def _rbf(x, y):
        np_x = np.array(x)
        if np_x.ndim <= 1:
            return np.exp((-1 * np.dot(x - y, x - y) / (2 * sigma * sigma)))
        else:
            return np.exp((-1 * np.multiply(x - y, x - y).sum(axis=1) / (2 * sigma * sigma)))

    return _rbf
