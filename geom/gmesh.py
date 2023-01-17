from igl import read_triangle_mesh
import taichi as ti
import utils.geom2d as geom2d
import utils.mathlib as mathlib


def triangle_mesh_input(filepath):
  v, f = read_triangle_mesh(filepath)
  n_v = v.shape[0]
  n_f = f.shape[0]
  return n_v, n_f, v, f.flatten()


@ti.data_oriented
class TrianMesh:

  def __init__(self,
               filepath,
               dim=2,
               rho=1.0,
               get_edge=True,
               get_edgeside=True,
               get_edgeNeib=True,
               get_faceedge=True):
    n_vert, n_face, vert, face_indices = triangle_mesh_input(filepath)
    if dim == 2:
      vert = mathlib.np_3to2(vert)
    edge_indices, edge_sides, edge_neib, face_edges = geom2d.edge_extractor(
        face_indices)
    assert face_indices.ndim == 1
    n_edge = edge_indices.shape[0] // 2

    self.n_vert = n_vert
    self.n_edge = n_edge
    self.n_face = n_face

    self.v_p = ti.Vector.field(dim, dtype=ti.f32, shape=n_vert)
    self.v_p.from_numpy(vert)
    self.v_p_ref = ti.Vector.field(dim, dtype=ti.f32, shape=n_vert)
    self.v_p_ref.copy_from(self.v_p)
    self.f_i = ti.field(dtype=ti.i32, shape=n_face * 3)
    self.f_i.from_numpy(face_indices.flatten())

    if get_edge:
      self.e_i = ti.field(dtype=ti.i32, shape=n_edge * 2)
      self.e_i.from_numpy(edge_indices.flatten())
    if get_edgeside:
      self.e_sidei = ti.field(dtype=ti.i32, shape=n_edge * 2)
      self.e_sidei.from_numpy(edge_sides.flatten())
    if get_edgeNeib:
      self.e_neibi = ti.field(dtype=ti.i32, shape=n_edge * 2)
      self.e_neibi.from_numpy(edge_neib.flatten())
    if get_faceedge:
      self.f_edgei = ti.field(dtype=ti.i32, shape=n_face * 3)
      self.f_edgei.from_numpy(face_edges)

    self.compute_mass(rho)

  def compute_mass(self, rho: float):
    self.v_invm = ti.field(dtype=ti.f32, shape=self.n_vert)
    self.f_mass = ti.field(dtype=ti.f32, shape=self.n_face)
    self.get_mass(rho)

  @ti.kernel
  def get_mass(self, rho: ti.f32):
    for k in range(self.n_face):
      p1 = self.f_i[k * 3]
      p2 = self.f_i[k * 3 + 1]
      p3 = self.f_i[k * 3 + 2]
      x1 = self.v_p[p1]
      x2 = self.v_p[p2]
      x3 = self.v_p[p3]
      self.f_mass[k] = rho * 0.5 * ((x2 - x1).cross(x3 - x1)).norm()
      self.v_invm[p1] += self.f_mass[k] / 3.0
      self.v_invm[p2] += self.f_mass[k] / 3.0
      self.v_invm[p3] += self.f_mass[k] / 3.0
    for k in range(self.n_vert):
      self.v_invm[k] = 1.0 / self.v_invm[k]

  @ti.kernel
  def get_pos_by_index(self, n: ti.i32, index: ti.template(),
                       pos: ti.template()):
    for k in range(n):
      pos[k] = self.v_p_ref[index[k]]

  @ti.kernel
  def set_pos_by_index(self, n: ti.i32, index: ti.template(),
                       pos: ti.template()):
    for k in range(n):
      self.v_p[index[k]] = pos[k]

  @ti.kernel
  def set_fixed_point(self, n: ti.i32, index: ti.template()):
    for k in range(n):
      self.v_invm[index[k]] = 0.0
