import taichi as ti
import geom.hashgrid


# Discrete Element Method
@ti.data_oriented
class DEM3D:

  def __init__(self,
               v_p: ti.MatrixField,
               v_p_cache: ti.MatrixField,
               hash3d: geom.hashgrid.HashGrid3D,
               r: float,
               fric_para=0.0) -> None:
    self.v_p = v_p
    self.v_p_cache = v_p_cache
    self.part_neib_count = hash3d.part_neib_count
    self.part_neib_list = hash3d.part_neib_list
    self.hashgrid = hash3d
    self.r = r
    assert self.r <= hash3d.neib
    self.fric_para = fric_para

  def init_rest_status(self):
    self.hashgrid.dem_update()
    self.solve_cons()

  def preupdate_cons(self):
    pass

  def update_cons(self):
    if self.fric_para == 0.0:
      self.solve_cons_no_friction()
    else:
      self.solve_cons()

  @ti.kernel
  def solve_cons_no_friction(self):
    for i in self.v_p:
      neib_count = self.part_neib_count[i]
      for index_j in range(neib_count):
        j = self.part_neib_list[i, index_j]
        p = self.v_p[i] - self.v_p[j]
        C = p.norm() - 2.0 * self.r
        if C < 0.0:
          p = p.normalized()
          self.v_p[i] -= p * C / 2.0
          self.v_p[j] += p * C / 2.0

  @ti.kernel
  def solve_cons(self):
    for i in self.v_p:
      neib_count = self.part_neib_count[i]
      for index_j in range(neib_count):
        j = self.part_neib_list[i, index_j]
        p = self.v_p[i] - self.v_p[j]
        C = p.norm() - 2.0 * self.r
        if C < 0.0:
          if p.norm() < 1e-6:
            p = p + self.r * 1e-2 * ti.Vector([
                ti.random(), ti.random(), ti.random()
            ])
          p = p.normalized()
          self.v_p[i] -= p * C / 2.0
          self.v_p[j] += p * C / 2.0
          v_i = self.v_p[i] - self.v_p_cache[i]
          v_j = self.v_p[j] - self.v_p_cache[j]
          v_i = v_i - v_i.dot(p) * p
          v_j = v_j - v_j.dot(p) * p
          v_avg = (v_i + v_j) / 2.0
          self.v_p[i] += self.fric_para * (v_avg - v_i)
          self.v_p[j] += self.fric_para * (v_avg - v_j)
