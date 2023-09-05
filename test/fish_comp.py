import taichi as ti
import numpy as np
import glfw
from OpenGL.GL import *

from geom import gmesh, obj
from utils import parser, gl_mesh_viewer
from lbs import lbs
from cons import framework, deform2d, comp

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
substep = 6
sovle_step = 2
dt = 1.0 / (fps * substep)
xpbd = framework.pbd_framework(g=g,
                               n_vert=mesh.n_vert,
                               v_p=mesh.v_p,
                               dt=dt,
                               damp=0.993)
deform = deform2d.Deform2D(dt=dt,
                           v_p=mesh.v_p,
                           v_p_ref=mesh.v_p_ref,
                           f_i=mesh.f_i,
                           v_invm=mesh.v_invm,
                           face_mass=mesh.f_mass,
                           hydro_alpha=1e-3,
                           devia_alpha=1e-2)
comp = comp.CompDynPoint2D(v_p=mesh.v_p,
                           v_p_ref=mesh.v_p_ref,
                           v_p_rig=lbs.v_p_rig,
                           v_invm=mesh.v_invm,
                           c_p=points.c_p,
                           c_p_ref=points.c_p_ref,
                           c_rot=lbs.c_rot,
                           v_weights=points.v_weights,
                           dt=dt,
                           alpha=1e-3,
                           alpha_fixed=4e-6,
                           fixed=[0, 1])
xpbd.add_cons(deform, 0)
xpbd.add_cons(comp, 1)
xpbd.init_rest_status()

window = gl_mesh_viewer.OpenGLMeshRenderer2D("GLFW Viewer", res=(600, 600))
window.set_mesh(mesh.v_p.to_numpy(), mesh.uvs, mesh.faces_np, texpath)

view_mode = [
    True,  # wireframe mode 
    True  # vertex position or lbs position
]


def set_view_mode(key, scancode, action, mods, ik=view_mode):
  if action == glfw.PRESS and key == glfw.KEY_F:
    ik[0] = not ik[0]
  if action == glfw.PRESS and key == glfw.KEY_R:
    ik[1] = not ik[1]


window.add_key_callback(set_view_mode)


def set_movement():

  t = window.get_time() - 4.0

  if t > 0.0:
    angle0 = np.sin(t * 6.0) * 0.35
    angle1 = angle0 * 2.0
    lbs.set_control_angle(0, angle0)
    lbs.set_control_pos_from_parent(1, 0, angle0)
    lbs.set_control_angle(1, angle1)


while window.running:
  set_movement()
  lbs.lbs()

  for sub in range(substep):
    xpbd.make_prediction()
    xpbd.preupdate_cons(0)
    xpbd.preupdate_cons(1)
    for _ in range(sovle_step):
      xpbd.update_cons(0)
    xpbd.update_cons(1)
    xpbd.update_vel()

  window.pre_update()

  if view_mode[0]:
    window.set_wireframe_mode(True, color=(1.0, 0.0, 0.0, 1.0))
  else:
    window.set_wireframe_mode(False)

  glClearColor(0.96, 0.96, 0.96, 1)

  if view_mode[1]:
    window.update_mesh(lbs.v_p.to_numpy())
  else:
    window.update_mesh(lbs.v_p_rig.to_numpy())

  window.data_render()
  lbs.draw_display_points(fix_point=[0, 1])

  window.show()

window.terminate()