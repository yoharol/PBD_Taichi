import taichi as ti

from geom.gmesh import TrianMesh


@ti.data_oriented
class Volume:

  def __init__(
      self,
      v_p: ti.MatrixField,  # vertex positions
      v_p_ref: ti.MatrixField,  # vertex reference positions
      v_p_delta: ti.MatrixField,  # delta vertex positions
      f_i: ti.Field,  # face indices
      v_invm: ti.Field,  # vertex inverse mass
      dt,  # time step
      alpha=0.0  # inverse stiffness
  ) -> None:
    self.n = f_i.shape[0] // 3
    self.pos = v_p
    self.ref_pos = v_p_ref
    self.pos_delta = v_p_delta
    self.face_indices = f_i
    self.invm = v_invm

    self.rest_volume = ti.field(dtype=ti.f32, shape=())
    self.lambdaf = ti.field(dtype=ti.f32, shape=())
    self.alpha = alpha / (dt * dt)

  def init_rest_status(self):
    for k in range(self.n):
      i1 = self.face_indices[k * 3]
      i2 = self.face_indices[k * 3 + 1]
      i3 = self.face_indices[k * 3 + 2]
      p1 = self.ref_pos[i1]
      p2 = self.ref_pos[i2]
      p3 = self.ref_pos[i3]
      self.rest_volume[None] += p1.dot(p2.cross(p3))

  def preupdate_cons(self):
    self.lambdaf[None] = 0.0

  def update_cons(self):
    self.solve_cons()

  @ti.kernel
  def solve_cons(self):
    curr_volume = 0.0
    deriv = 0.0
    self.pos_delta.fill(ti.Vector([0.0, 0.0, 0.0]))
    for k in range(self.n):
      i1 = self.face_indices[k * 3]
      i2 = self.face_indices[k * 3 + 1]
      i3 = self.face_indices[k * 3 + 2]
      p1 = self.pos[i1]
      p2 = self.pos[i2]
      p3 = self.pos[i3]
      d1 = p2.cross(p3)
      d2 = p3.cross(p1)
      d3 = p1.cross(p2)
      deriv += self.invm[i1] * d1.norm_sqr() + self.invm[i2] * d2.norm_sqr(
      ) + self.invm[i3] * d3.norm_sqr()
      curr_volume += p1.dot(d1)

    C = curr_volume - self.rest_volume[None]
    delta_lambda = -(C + self.lambdaf[None] * self.alpha) / (deriv + self.alpha)
    self.lambdaf[None] += delta_lambda

    for k in range(self.n):
      i1 = self.face_indices[k * 3]
      i2 = self.face_indices[k * 3 + 1]
      i3 = self.face_indices[k * 3 + 2]
      p1 = self.pos[i1]
      p2 = self.pos[i2]
      p3 = self.pos[i3]
      d1 = p2.cross(p3)
      d2 = p3.cross(p1)
      d3 = p1.cross(p2)

      self.pos_delta[i1] += self.invm[i1] * delta_lambda * d1
      self.pos_delta[i2] += self.invm[i2] * delta_lambda * d2
      self.pos_delta[i3] += self.invm[i3] * delta_lambda * d3

    for k in self.pos:
      self.pos[k] += self.pos_delta[k]
