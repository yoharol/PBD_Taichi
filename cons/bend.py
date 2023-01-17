import numpy as np
import taichi as ti

from geom import gmesh


@ti.data_oriented
class Bend3D:

  def __init__(self, mesh: gmesh.TrianMesh, alpha=0.0) -> None:
    self.n = mesh.n_edge
    self.pos = mesh.v_p
    self.pos_ref = mesh.v_p_ref
    self.indices = mesh.e_i
    self.side_indices = mesh.e_sidei
    self.invm = mesh.v_invm
    self.alpha = alpha

    self.lambdaf = ti.field(dtype=ti.f32, shape=self.n)
    self.edge_rest_angle = ti.field(dtype=ti.f32, shape=self.n)

  @ti.kernel
  def compute_rest_angle(self):
    for k in range(self.n):
      if self.side_indices[k * 2 + 1] != -1:
        i1 = self.indices[k * 2]
        i2 = self.indices[k * 2 + 1]
        i3 = self.side_indices[k * 2]
        i4 = self.side_indices[k * 2 + 1]
        x1 = self.pos_ref[i1]
        x2 = self.pos_ref[i2]
        x3 = self.pos_ref[i3]
        x4 = self.pos_ref[i4]

        n1 = ((x2 - x1).cross(x3 - x1)).normalized()
        n2 = ((x2 - x1).cross(x4 - x1)).normalized()
        d = n1.dot(n2)
        self.edge_rest_angle[k] = ti.acos(d)

  def init_rest_status(self):
    self.compute_rest_angle()

  def preupdate_cons(self):
    self.lambdaf.fill(0.0)

  def update_cons(self, dt):
    self.sovle_cons(dt)

  @ti.kernel
  def sovle_cons(self, dt: ti.f32):
    for k in range(self.n):
      if self.side_indices[k * 2 + 1] != -1:
        i1 = self.indices[k * 2]
        i2 = self.indices[k * 2 + 1]
        i3 = self.side_indices[k * 2]
        i4 = self.side_indices[k * 2 + 1]
        x1 = self.pos[i1]
        x2 = self.pos[i2]
        x3 = self.pos[i3]
        x4 = self.pos[i4]
        p2 = x2 - x1
        p3 = x3 - x1
        p4 = x4 - x1
        n1 = (p2.cross(p3)).normalized()
        n2 = (p2.cross(p4)).normalized()
        d = n1.dot(n2)
        if d * d < 1.0:
          C = ti.acos(d) - self.edge_rest_angle[k]
          q3 = (p2.cross(n2) + n1.cross(p2) * d) / (p2.cross(p3)).norm()
          q4 = (p2.cross(n1) + n2.cross(p2) * d) / (p2.cross(p4)).norm()
          q2 = -(p3.cross(n2) + n1.cross(p3) * d) / (p2.cross(p3)).norm() - (
              p4.cross(n1) + n2.cross(p4) * d) / (p2.cross(p4)).norm()
          q1 = -q2 - q3 - q4

          coe1 = (self.invm[i1] * q1.norm_sqr() + self.invm[i2] * q2.norm_sqr()
                  + self.invm[i3] * q3.norm_sqr() +
                  self.invm[i4] * q4.norm_sqr()) / (1 - d * d)
          delta_lambda = -(C + self.alpha * self.lambdaf[k] /
                           (dt * dt)) / (coe1 + self.alpha / (dt * dt))
          coe2 = 1.0 / ti.sqrt(1 - d * d)
          self.pos[i1] += self.invm[i1] * coe2 * q1 * delta_lambda
          self.pos[i2] += self.invm[i2] * coe2 * q2 * delta_lambda
          self.pos[i3] += self.invm[i3] * coe2 * q3 * delta_lambda
          self.pos[i4] += self.invm[i4] * coe2 * q4 * delta_lambda
          self.lambdaf[k] += delta_lambda
