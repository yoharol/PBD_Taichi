import taichi as ti
import numpy as np
from OpenGL.GL import *

from utils import parser


class PointsData2D:

  def __init__(self,
               points: np.ndarray,
               weights: np.ndarray,
               scale=1.0,
               repose=(0.0, 0.0, 0.0)) -> None:

    self.n_points = points.shape[0]

    points = points * scale + np.array(repose, dtype=np.float32)
    self.points_np = points
    self.weights_np = weights

    self.c_p = ti.Vector.field(2, dtype=ti.f32, shape=self.n_points)
    self.c_p_ref = ti.Vector.field(2, dtype=ti.f32, shape=self.n_points)
    self.c_p_input = ti.Vector.field(2, dtype=ti.f32, shape=self.n_points)
    self.v_weights = ti.field(dtype=ti.f32, shape=weights.shape)

    self.c_A = ti.Matrix.field(2, 2, dtype=ti.f32, shape=self.n_points)
    self.c_A.from_numpy(
        np.eye(2, dtype=np.float32).repeat(self.n_points, 0).reshape(-1, 2, 2))
    self.c_b = ti.Vector.field(2, dtype=ti.f32, shape=self.n_points)
    self.c_b.from_numpy(np.zeros((self.n_points, 2), dtype=np.float32))

    self.c_p.from_numpy(points)
    self.c_p_ref.from_numpy(points)
    self.c_p_input.from_numpy(points)

    self.v_weights.from_numpy(weights)

    self.c_color = np.zeros((self.n_points, 3), dtype=np.float32)
    self.c_color[:] = np.array([1.0, 0.0, 0.0])

  def set_color(self,
                point_color=(0.0, 1.0, 0.0),
                fixed_color=(1.0, 0.0, 0.0),
                fixed=[]):
    self.c_color[:] = np.array(point_color)
    self.c_color[fixed] = np.array(fixed_color)


def load_points2d_data(tgfpath: str,
                       weightpath: str,
                       scale=1.0,
                       repose=(0.0, 0.0, 0.0)):
  points = parser.tgf_loader(tgfpath)
  weights = np.loadtxt(weightpath, delimiter=',')

  return PointsData2D(points, weights, scale, repose)


@ti.data_oriented
class PointLBS2D:

  def __init__(
      self,
      v_p: ti.MatrixField,  # vertex position
      v_p_ref: ti.MatrixField,  # vertex reference position
      v_weights: ti.Field,  # vertex weights
      v_invm: ti.Field,  # inverse mass
      c_p: ti.MatrixField,  # control point position
      c_p_ref: ti.MatrixField,  # control point reference position
  ) -> None:
    self.n_verts = v_p.shape[0]
    self.n_points = c_p.shape[0]

    self.v_p = v_p
    self.v_p_ref = v_p_ref
    self.v_p_rig = ti.Vector.field(2, dtype=ti.f32, shape=self.n_verts)
    self.v_p_rig.copy_from(self.v_p)
    self.v_weights = v_weights
    self.v_invm = v_invm
    self.c_p = c_p
    self.c_p_ref = c_p_ref
    self.c_p_ref_np = c_p_ref.to_numpy()

    self.c_rot = ti.Matrix.field(2, 2, dtype=ti.f32,
                                 shape=self.n_points)  # control point rotation
    for i in range(self.n_points):
      self.set_control_angle(i, 0.0)

  def set_control_pos(self, idx: int, pos: np.ndarray):
    self.c_p[idx][0] = pos[0]
    self.c_p[idx][1] = pos[1]

  def set_control_angle(self, idx: int, angle: float):
    self.c_rot[idx][0, 0] = np.cos(angle)
    self.c_rot[idx][0, 1] = -np.sin(angle)
    self.c_rot[idx][1, 0] = np.sin(angle)
    self.c_rot[idx][1, 1] = np.cos(angle)

  def set_control_pos_from_parent(self, idx: int, parent_idx: int,
                                  angle: float):
    p1 = self.c_p_ref_np[idx]
    p0 = self.c_p_ref_np[parent_idx]
    p = np.zeros(2, dtype=np.float32)
    p_delta_ref = p1 - p0
    p[0] = p_delta_ref[0] * np.cos(angle) - p_delta_ref[1] * np.sin(angle)
    p[1] = p_delta_ref[0] * np.sin(angle) + p_delta_ref[1] * np.cos(angle)
    self.c_p[idx][0] = p[0] + self.c_p[parent_idx][0]
    self.c_p[idx][1] = p[1] + self.c_p[parent_idx][1]

  @ti.kernel
  def lbs(self):
    for i in range(self.n_verts):
      self.v_p_rig[i] = ti.Vector.zero(ti.f32, 2)
      for j in range(self.n_points):
        self.v_p_rig[i] += self.v_weights[i, j] * (
            self.c_rot[j] @ (self.v_p_ref[i] - self.c_p_ref[j]) + self.c_p[j])

  def draw_display_points(self,
                          point_size=25.0,
                          point_color=(0.0, 0.9, 0.2),
                          fix_color=(0.9, 0.1, 0.0),
                          fix_point=[],
                          scale=1.0):
    trans = self.c_p.to_numpy()
    rot = self.c_rot.to_numpy()

    def t2f(idx):
      p = trans[idx]
      glVertex2f(p[0] * 2.0 - 1.0, p[1] * 2.0 - 1.0)

    glLineWidth(4 * scale)
    glBegin(GL_LINES)
    for i in range(self.n_points):
      glColor3f(1.0, 0.5, 0.0)
      t2f(i)
      p = trans[i] + rot[i] @ np.array([0.07 * scale, 0.0])
      glVertex2f(p[0] * 2.0 - 1.0, p[1] * 2.0 - 1.0)
      glColor3f(0.0, 0.5, 1.0)
      t2f(i)
      p = trans[i] + rot[i] @ np.array([0.0, 0.07 * scale])
      glVertex2f(p[0] * 2.0 - 1.0, p[1] * 2.0 - 1.0)
    glEnd()

    glPointSize(point_size * scale)
    glBegin(GL_POINTS)
    for i in range(self.n_points):
      if i in fix_point:
        glColor3f(fix_color[0], fix_color[1], fix_color[2])
        t2f(i)
      else:
        glColor3f(point_color[0], point_color[1], point_color[2])
        t2f(i)

    glEnd()