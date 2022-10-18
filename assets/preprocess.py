import taichi as ti
import numpy as np
import os

ti.init(arch=ti.cpu, default_fp=ti.float32, default_ip=ti.int32)

working_path = os.path.dirname(__file__)

with open(os.path.join(working_path, './low_poly_bunny.npy'), 'rb') as f:
  vertices_pos = np.load(f)  # position of vertices
  tet_ids = np.load(f)  # composition of tetrahedrals
  edge_ids = np.load(f)  # composition of edges
  surface_ids = np.load(f)  # composition of surface faces

num_particles = vertices_pos.shape[0]
num_tets = tet_ids.shape[0]
num_edges = edge_ids.shape[0]
num_surfaces = surface_ids.shape[0]

tet = ti.field(dtype=int, shape=(num_tets, 4))
tet.from_numpy(tet_ids)
x = ti.Vector.field(3, dtype=float, shape=num_particles)
x.from_numpy(vertices_pos)
mass = ti.field(dtype=float, shape=num_particles)
mass.fill(0.0)
orders = ti.field(dtype=int, shape=num_particles)
orders.fill(0)
edge = ti.Vector.field(2, dtype=int, shape=num_edges)
edge.from_numpy(edge_ids)
rest_length = ti.field(dtype=float, shape=num_edges)


@ti.kernel
def compute_average_mass():
  for i in range(num_tets):
    e1 = x[tet[i, 1]] - x[tet[i, 0]]
    e2 = x[tet[i, 2]] - x[tet[i, 0]]
    e3 = x[tet[i, 3]] - x[tet[i, 0]]
    volume = abs((e1.cross(e2)).dot(e3))
    for j in range(4):
      mass[tet[i, j]] += volume / 4.0

  minv, maxv = 999.0, 0.0
  for i in range(num_particles):
    ti.atomic_min(minv, mass[i])
    ti.atomic_max(maxv, mass[i])
  print(minv, maxv)


@ti.kernel
def compute_max_order() -> int:
  for i in range(num_tets):
    for j in range(4):
      orders[tet[i, j]] += 1
  max_order = 0
  for i in range(num_particles):
    ti.atomic_max(max_order, orders[i])
  return max_order


@ti.kernel
def compute_rest_length():
  for i in range(num_edges):
    x_ij = x[edge[i][0]] - x[edge[i][1]]
    rest_length[i] = x_ij.norm()


compute_average_mass()
max_order = compute_max_order()
print(max_order)

vertices_mass = mass.to_numpy()
print(vertices_mass.shape, vertices_mass.dtype)

compute_rest_length()
rest = rest_length.to_numpy()

with open(os.path.join(working_path, './bunny_with_mass.npy'), 'wb') as f:
  for attr in [
      vertices_pos, tet_ids, edge_ids, surface_ids, vertices_mass, rest
  ]:
    np.save(f, attr)
  np.save(f, np.array([max_order]))