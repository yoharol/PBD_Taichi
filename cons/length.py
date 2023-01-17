import taichi as ti


@ti.data_oriented
class LengthCons:

  def __init__(self, model, alpha=0.0) -> None:
    self.n = model.n_edge

    self.pos = model.v_p
    self.pos_ref = model.v_p_ref
    self.indices = model.e_i
    self.invm = model.v_invm

    self.rest_length = ti.field(dtype=ti.f32, shape=self.n)
    self.length = ti.field(dtype=ti.f32, shape=self.n)
    self.lambdaf = ti.field(dtype=ti.f32, shape=self.n)
    self.alpha = alpha

  @ti.kernel
  def compute_rest_length(self):
    for k in range(self.n):
      i = self.indices[k * 2]
      j = self.indices[k * 2 + 1]
      xi = self.pos_ref[i]
      xj = self.pos_ref[j]
      self.rest_length[k] = (xi - xj).norm()

  def init_rest_status(self):
    self.compute_rest_length()

  def preupdate_cons(self):
    self.lambdaf.fill(0.0)

  def update_cons(self, dt):
    self.solve_cons(dt)

  @ti.kernel
  def solve_cons(self, dt: ti.f32):
    for k in range(self.n):
      i = self.indices[k * 2]
      j = self.indices[k * 2 + 1]
      xi = self.pos[i]
      xj = self.pos[j]
      xij = xi - xj
      self.length[k] = xij.norm()
      C = self.length[k] - self.rest_length[k]
      wi = self.invm[i]
      wj = self.invm[j]
      delta_lambda = -(C + self.alpha * self.lambdaf[k] /
                       (dt * dt)) / (wi + wj + self.alpha / (dt * dt))
      self.lambdaf[k] += delta_lambda
      xij = xij / xij.norm()
      self.pos[i] += wi * delta_lambda * xij.normalized()
      self.pos[j] += -wj * delta_lambda * xij.normalized()
