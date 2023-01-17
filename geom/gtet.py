from igl import read_mesh
import taichi as ti



def read_tet_mesh(filepath):
  v, t, f = read_mesh(filepath)
  n_v = v.shape[0]
  n_t = t.shape[0]
  n_f = f.shape[0]
  return n_v, n_t, n_f, v, t.flatten(), f.flatten()


@ti.data_oriented
class TetMesh:

  def __init__(self, filepath) -> None:
    n_v, n_t, n_f, v, t, f = read_tet_mesh(filepath)
    self.n_vert = n_v
    self.n_face = n_f
    self.n_tet = n_t

    self.v_p = ti.Vector.field(3, dtype=ti.f32, shape=n_v)
    self.v_p.from_numpy(v)
    self.t_i = ti.field(dtype=ti.i32, shape=self.n_tet * 4)
    self.t_i.from_numpy(t.flatten())
    self.f_i = ti.field(dtype=ti.i32, shape=self.n_fac * 3)
    self.f_i.from_numpy(f.flatten())

  def compute_mass(self, rho: float):
    self.v_invm = ti.field(dtype=ti.f32, shape=self.n_vert)
    self.t_mass = ti.field(dtype=ti.f32, shape=self.n_face)
    self.get_mass(rho)

  @ti.kernel
  def get_mass(self, rho: ti.f32):
    for k in range(self.n_tet):
      p1 = self.t_i[k * 4]
      p2 = self.t_i[k * 4 + 1]
      p3 = self.t_i[k * 4 + 2]
      p4 = self.t_i[k * 4 + 3]
      x1 = self.v_p[p1]
      x2 = self.v_p[p2]
      x3 = self.v_p[p3]
      x4 = self.v_p[p4]
      self.t_mass[k] = rho * ti.abs(((x4 - x1).dot(
          (x2 - x1).cross(x3 - x1))).norm()) / 6.0
      self.v_invm[p1] += self.t_mass[k] / 4.0
      self.v_invm[p2] += self.t_mass[k] / 4.0
      self.v_invm[p3] += self.t_mass[k] / 4.0
      self.v_invm[p4] += self.t_mass[k] / 4.0
    for k in range(self.n_vert):
      self.v_invm[k] = 1.0 / self.v_invm[k]
