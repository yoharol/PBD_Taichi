import taichi as ti
import numpy as np
from scipy import linalg
import glfw
from OpenGL.GL import *

from geom import gmesh, obj
from utils import parser, gl_mesh_viewer
from lbs import lbs
from cons import framework, deform2d, shape_matching

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

parent_folder = "assets/fish2bones"
usdpath = f"{parent_folder}/fish.usda"
texpath = f"{parent_folder}/Tex.png"
weightpath = f"{parent_folder}/Weights.csv"
tgfpath = f"{parent_folder}/fish.tgf"

scale = 0.8
repose = (0.1, 0.1)

verts, faces, uvs = parser.usd_parser(usdpath, "/root/mesh")
mesh = gmesh.TrianMesh(verts, faces, dim=2, rho=1.0, scale=scale, repose=repose)
mesh.set_texture_uv(uvs)

points = lbs.load_points2d_data(tgfpath, weightpath, scale=scale, repose=repose)
lbs = lbs.PointLBS2D(mesh.v_p, mesh.v_p_ref, points.v_weights, mesh.v_invm,
                     points.c_p, points.c_p_ref)

g = ti.Vector([0.0, 0.0])
fps = 60
substep = 5
sovle_step = 1
dt = 1.0 / (fps * substep)
xpbd = framework.pbd_framework(g=g,
                               n_vert=mesh.n_vert,
                               v_p=mesh.v_p,
                               dt=dt,
                               damp=0.993)
comp = shape_matching.ShapeMatching2D(v_p=mesh.v_p,
                                      v_p_ref=mesh.v_p_ref,
                                      v_p_rig=lbs.v_p_rig,
                                      v_invm=mesh.v_invm,
                                      v_weights=points.v_weights,
                                      dt=dt,
                                      alpha=1e-3)
xpbd.add_cons(comp)
xpbd.init_rest_status()

window = gl_mesh_viewer.OpenGLMeshRenderer2D("GLFW Viewer", res=(600, 600))
window.set_mesh(mesh.v_p.to_numpy(), mesh.uvs, mesh.faces_np, texpath, idx=0)
window.set_wireframe_mode(True, color=(1.0, 0.0, 0.0, 1.0), idx=0)
window.set_mesh(lbs.v_p_rig.to_numpy(), mesh.uvs, mesh.faces_np, texpath, idx=1)
window.set_wireframe_mode(True, color=(0.0, 0.3, 0.8, 1.0), idx=1)


def get_affine_matrix(v_p: np.ndarray, v_p_ref: np.ndarray, weights: np.ndarray,
                      mass: np.ndarray, c_pos: np.ndarray,
                      c_pos_ref: np.ndarray):
  n_vert = v_p.shape[0]
  n_control = weights.shape[1]
  D = np.zeros((n_control, 2, 2))
  B = np.zeros((n_control, 2, 2))
  for i in range(n_vert):
    for j in range(n_control):
      D[j] += weights[i, j] * mass[i] * np.outer(v_p[i] - c_pos[j],
                                                 v_p_ref[i] - c_pos_ref[j])
      B[j] += weights[i, j] * mass[i] * np.outer(v_p_ref[i] - c_pos_ref[j],
                                                 v_p_ref[i] - c_pos_ref[j])
  for j in range(n_control):
    D[j] = D[j] @ linalg.inv(B[j])
  for j in range(n_control):
    D[j], B[j] = linalg.polar(D[j])
  return D, B


def get_quad_affine_matrix(v_p: np.ndarray, v_p_ref: np.ndarray,
                           weights: np.ndarray, mass: np.ndarray,
                           c_pos: np.ndarray, c_pos_ref: np.ndarray):
  n_vert = v_p.shape[0]
  n_control = weights.shape[1]
  D = np.zeros((n_control, 2, 5))
  B = np.zeros((n_control, 5, 5))
  for i in range(n_vert):
    for j in range(n_control):
      q = v_p_ref[i] - c_pos_ref[j]
      quad_vec = np.array([
          q[i][0], q[i][1], q[i][0] * q[i][0], q[i][1] * q[i][1],
          q[i][0] * q[i][1]
      ])
      D[j] += weights[i, j] * mass[i] * np.outer(v_p[i] - c_pos[j], quad_vec)
      B[j] += weights[i, j] * mass[i] * np.outer(quad_vec, quad_vec)
  R = np.zeros((n_control, 2, 2))
  S = np.zeros((n_control, 2, 2))
  for j in range(n_control):
    D[j] = D[j] @ linalg.inv(B[j])
  for j in range(n_control):
    R[j], S[j] = linalg.polar(D[j, :, :2])
  return D, B


def quad_blend_skinning(v_p: np.ndarray, v_p_ref: np.ndarray,
                        weights: np.ndarray, c_pos: np.ndarray,
                        c_pos_ref: np.ndarray, D: np.ndarray):
  n_vert = v_p.shape[0]
  n_control = weights.shape[1]
  v_p_rig = np.zeros_like(v_p)
  for i in range(n_vert):
    for j in range(n_control):
      q = v_p_ref[i] - c_pos_ref[j]
      quad_vec = np.array([
          q[i][0], q[i][1], q[i][0] * q[i][0], q[i][1] * q[i][1],
          q[i][0] * q[i][1]
      ])
      v_p_rig[i] += weights[i, j] * (c_pos[j] + quad_vec @ D[j])


def set_movement():

  t = window.frame_count - 60

  if t > 0.0:
    angle0 = np.sin(t * 6.0 / 60) * 0.35
    angle1 = angle0 * 2.5
    lbs.set_control_angle(0, angle0)
    lbs.set_control_pos_from_parent(1, 0)
    lbs.set_control_angle(1, angle1)


beta = 0.0

while window.running:
  set_movement()
  lbs.lbs()
  window.update_mesh(lbs.v_p_rig.to_numpy(), idx=1)

  for sub in range(substep):
    xpbd.make_prediction()
    xpbd.preupdate_cons()
    R, S = get_affine_matrix(mesh.v_p.to_numpy(), mesh.v_p_ref.to_numpy(),
                             points.v_weights.to_numpy(),
                             1.0 / mesh.v_invm.to_numpy(),
                             points.c_p.to_numpy(), points.c_p_ref.to_numpy())
    rigged_R = lbs.c_rot.to_numpy()
    for j in range(lbs.n_points):
      R[j] = beta * rigged_R[j] @ S[j] + (1 - beta) * rigged_R[j]
    lbs.c_rot.from_numpy(R)
    lbs.lbs()
    xpbd.update_cons()
    xpbd.update_vel()

  window.pre_update()

  glClearColor(0.96, 0.96, 0.96, 1)

  window.update_mesh(lbs.v_p.to_numpy(), idx=0)

  window.data_render()
  lbs.draw_display_points(fix_point=[0, 1])

  window.show()

window.terminate()
