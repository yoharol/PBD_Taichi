import taichi as ti
import numpy as np
import math


@ti.func
def get_col0_2d(a):
  return ti.Vector([a[0, 0], a[1, 0]])


@ti.func
def get_col1_2d(a):
  return ti.Vector([a[0, 1], a[1, 1]])


def np_2to3(array2d: np.array):
  return np.concatenate(
      (array2d, np.zeros(shape=(array2d.shape[0], 1), dtype=array2d.dtype)),
      axis=1)


def np_3to2(array3d: np.array):
  return array3d[..., :2]


SIGMA1D = 8.0 / 3.0
SIGMA2D = 40.0 / 7.0 / math.pi
SIGMA3D = 32.0 / math.pi


def kernel2d(r: float, h: float):
  q = r / h
  result = 0.0
  if 0.0 <= q and q <= 0.5:
    result = (6.0 * (q - 1.0) * math.pow(q, 2.0) + 1.0)
  elif q <= 1.0:
    result = 2.0 * math.pow((1.0 - q), 3.0)
  else:
    result = 0.0
  return result * SIGMA2D / math.pow(h, 2)


@ti.func
def sph_kernel3d(r: float, h: float):
  q = r / h
  result = 0.0
  if 0.0 <= q and q <= 0.5:
    result = (6.0 * (q - 1.0) * ti.pow(q, 2.0) + 1.0)
  elif q <= 1.0:
    result = 2.0 * ti.pow((1.0 - q), 3.0)
  else:
    result = 0.0
  return result * SIGMA3D / ti.pow(h, 2)