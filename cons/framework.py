import taichi as ti
from cons.cons import Constraint


@ti.data_oriented
class pbd_framework:

  def __init__(self, n_vert, v_p, g, dt) -> None:
    self.n_vert = n_vert
    self.v_p = v_p
    self.v_p_cache = ti.Vector.field(v_p.n, dtype=v_p.dtype, shape=n_vert)
    self.v_p_cache = ti.Vector.field(v_p.n, dtype=v_p.dtype, shape=n_vert)
    self.v_v = ti.Vector.field(v_p.n, dtype=v_p.dtype, shape=n_vert)
    self.dt = dt
    self.g = g

    self.cons_list = []
    self.n_cons = 0

  @ti.kernel
  def make_prediction(self):
    for k in range(self.n_vert):
      self.v_p_cache[k] = self.v_p[k]
      self.v_p[
          k] = self.v_p[k] + self.v_v[k] * self.dt + self.g * self.dt * self.dt

  @ti.kernel
  def update_vel(self):
    for k in range(self.n_vert):
      self.v_v[k] = (self.v_p[k] - self.v_p_cache[k]) / self.dt

  def add_cons(self, new_cons: Constraint):
    self.cons_list.append(new_cons)
    self.n_cons += 1

  def init_rest_status(self):
    for i in range(self.n_cons):
      self.cons_list[i].init_rest_status()

  def preupdate_cons(self):
    for i in range(self.n_cons):
      self.cons_list[i].preupdate_cons()

  def update_cons(self):
    for i in range(self.n_cons):
      self.cons_list[i].update_cons(self.dt)
