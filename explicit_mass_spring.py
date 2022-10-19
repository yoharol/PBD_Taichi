import taichi as ti
from pxr import Usd, UsdGeom
import numpy as np
import os
from assets.data_loader import TetData

ti.init(arch=ti.cpu, default_fp=ti.float32, default_ip=ti.int32)

working_dir = os.path.dirname(__file__)
data_path = os.path.join(working_dir, 'assets', 'bunny_with_mass.npy')
output_path = os.path.join(working_dir, 'outputs',
                           'explicit_bunny_animation.usdc')

initial_pos = ti.Vector([0.0, 1.3, 0.0])
frames = 120
frames_per_second = 30
substeps = 100
rho = 1.0
k = 600.0
dt = 1.0 / (frames_per_second * substeps)
externel_force = ti.Vector([0.0, -9.8, 0.0])
damping = 0.999

mesh = TetData(data_path, output_path, frames, frames_per_second)

particle_x = ti.Vector.field(3, float, shape=mesh.num_particles)
cache_x = ti.Vector.field(3, float, shape=mesh.num_particles)
particle_v = ti.Vector.field(3, float, shape=mesh.num_particles)
internel_force = ti.Vector.field(3, float, shape=mesh.num_particles)
mass = ti.field(float, shape=mesh.num_particles)
edges = ti.Vector.field(2, int, shape=mesh.num_edges)
rest_length = ti.field(float, shape=mesh.num_edges)

particle_x.from_numpy(mesh.vertices_pos)
cache_x.fill(ti.Vector([0.0, 0.0, 0.0]))
particle_v.fill(ti.Vector([0.0, 0.0, 0.0]))
mass.from_numpy(mesh.vertices_mass)
edges.from_numpy(mesh.edge_ids)
rest_length.from_numpy(mesh.rest_length)


@ti.kernel
def init_pos():
  for i in particle_x:
    particle_x[i] += initial_pos


@ti.kernel
def apply_externel_force():
  for i in particle_v:
    particle_v[i] += externel_force * dt


@ti.kernel
def compute_internel_force():
  internel_force.fill(ti.Vector([0.0, 0.0, 0.0]))
  for e in edges:
    i = edges[e][0]
    j = edges[e][1]
    x_ij = particle_x[i] - particle_x[j]
    f_ij = -k * (x_ij - rest_length[e] * x_ij / x_ij.norm()) / rest_length[e]
    internel_force[i] += f_ij
    internel_force[j] += (-1.0) * f_ij
  for i in particle_v:
    particle_v[i] += dt * internel_force[i] / 1.0


@ti.kernel
def forward_euler():
  for i in particle_x:
    cache_x[i] = particle_x[i] + particle_v[i] * dt

  for i in particle_x:
    if cache_x[i][1] < 0.0:
      cache_x[i][1] = 0.0

  for i in particle_x:
    particle_v[i] = ((cache_x[i] - particle_x[i]) / dt) * damping
    particle_x[i] = cache_x[i]


@ti.kernel
def solve_constraint():
  for i in particle_x:
    if particle_x[i][1] < 0.0:
      particle_x


init_pos()
for f in range(frames):
  for _ in range(substeps):
    compute_internel_force()
    apply_externel_force()
    forward_euler()
  mesh.render(particle_x.to_numpy(), f)

mesh.save()