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


@ti.kernel
def isnan(veclist: ti.template()) -> ti.i32:
  have_nan = 0
  for i in veclist:
    if (ti.math.isnan(veclist[i][0]) or ti.math.isnan(veclist[i][1]) or
        ti.math.isnan(veclist[i][2])):
      print(i, veclist[i])
      have_nan = 1
  return have_nan


@ti.func
def get_cross_matrix(vec):
  return ti.Matrix.cols([[0.0, vec[2], -vec[1]], [-vec[2], 0.0, vec[0]],
                         [vec[1], -vec[0], 0.0]])


@ti.func
def rotvec_to_matrix(vec, angle):
  ans = ti.Matrix.identity(dt=ti.f32, n=3)
  if vec.norm_sqr() > 0.0:
    V = get_cross_matrix(vec.normalized())
    cos = ti.cos(angle)
    sin = ti.sin(angle)
    ans = ans + sin * V + (1 - cos) * (V @ V)
  return ans


@ti.func
def rotmatrix_to_vec(r):
  cos = (r[0, 0] + r[1, 1] + r[2, 2] - 1) / 2.0
  angle = 0.0
  if cos >= 1.0:
    angle = 1.0
  else:
    angle = ti.acos(cos)
  tmp = ti.sqrt((r[2, 1] - r[1, 2]) * (r[2, 1] - r[1, 2]) +
                (r[0, 2] - r[2, 0]) * (r[0, 2] - r[2, 0]) +
                (r[1, 0] - r[0, 1]) * (r[1, 0] - r[0, 1]))
  x = (r[2, 1] - r[1, 2]) / tmp
  y = (r[0, 2] - r[2, 0]) / tmp
  z = (r[1, 0] - r[0, 1]) / tmp
  vec = ti.Vector([x, y, z])
  if vec.norm_sqr() == 0.0:
    vec = ti.Vector([1.0, 0.0, 0.0])
  return vec.normalized() * angle


@ti.func
def get_rotmatrix(vec_from, vec_to):
  a = vec_from.normalized()
  b = vec_to.normalized()
  rot_dir = a.cross(b)
  r_cos = a.dot(b)
  r_sin = ti.sqrt(ti.math.clamp(1.0 - r_cos * r_cos, 0.0, 1.0))
  if r_sin > 1e-6:
    rot_dir = rot_dir / r_sin
  V = get_cross_matrix(rot_dir)
  return ti.Matrix.identity(dt=ti.f32, n=3) + r_sin * V + (1 - r_cos) * (V @ V)


@ti.func
def heat_rgb(value, minimum, maximum):
  ratio = 2.0 * (value - minimum) / (maximum - minimum)
  b = ti.max(0.0, 1.0 - ratio)
  r = ti.max(0.0, ratio - 1.0)
  g = 1.0 - b - r
  return ti.Vector([r, g, b])


@ti.func
def quat_to_rotate(qw, qx, qy, qz):
  return ti.Matrix([[
      1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qz * qw),
      2 * (qx * qz - qy * qw)
  ],
                    [
                        2 * (qx * qy - qz * qw), 1 - 2 * (qx * qx + qz * qz),
                        2 * (qy * qz + qx * qw)
                    ],
                    [
                        2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw),
                        1 - 2 * (qx * qx + qy * qy)
                    ]]).transpose()


def resize_array(arr: np.ndarray, x=1.0, y=1.0, z=1.0):
  arr[..., 0] *= x
  arr[..., 1] *= y
  arr[..., 2] *= z
  return arr


@ti.kernel
def get_closest_point(pos: ti.template(), x: ti.f32, y: ti.f32) -> ti.i32:
  index = -1
  mindis = 10000.0
  p = ti.Vector([x, y])
  ti.loop_config(serialize=True)
  for i in pos:
    d = p - pos[i]
    if d.norm() < mindis:
      mindis = d.norm()
      index = i

  return index