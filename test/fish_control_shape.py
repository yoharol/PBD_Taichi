import taichi as ti
import numpy as np
import glfw
from OpenGL.GL import *

from geom import gmesh, obj
from utils import parser, gl_mesh_viewer

from lbs import lbs
from cons import framework, deform2d, comp, shape_matching

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

parent_folder = "assets/fish4bones"
usdpath = f"{parent_folder}/fish.usda"
texpath = f"{parent_folder}/Tex.png"
weightpath = f"{parent_folder}/Weights.csv"
tgfpath = f"{parent_folder}/fish.tgf"

scale = 1.0
repose = (0.0, 0.0)

verts, faces, uvs = parser.usd_parser(usdpath, "/root/mesh")
mesh = gmesh.TrianMesh(verts, faces, dim=2, rho=1.0, scale=scale, repose=repose)
mesh.set_texture_uv(uvs)

points = lbs.load_points2d_data(tgfpath, weightpath, scale=scale, repose=repose)
lbs = lbs.PointLBS2D(mesh.v_p, mesh.v_p_ref, points.v_weights, mesh.v_invm,
                     points.c_p, points.c_p_ref)

g = ti.Vector([0.0, 0.0])
fps = 60
substep = 6
sovle_step = 1
dt = 1.0 / (fps * substep)
xpbd = framework.pbd_framework(g=g,
                               n_vert=mesh.n_vert,
                               v_p=mesh.v_p,
                               dt=dt,
                               damp=0.99)
deform = deform2d.Deform2D(dt=dt,
                           v_p=mesh.v_p,
                           v_p_ref=mesh.v_p_ref,
                           f_i=mesh.f_i,
                           v_invm=mesh.v_invm,
                           face_mass=mesh.f_mass,
                           hydro_alpha=1e-3,
                           devia_alpha=1e-3)
shape = shape_matching.ShapeMatching2D(v_p=mesh.v_p,
                                       v_p_ref=mesh.v_p_ref,
                                       v_p_rig=lbs.v_p_rig,
                                       v_invm=mesh.v_invm,
                                       v_weights=points.v_weights,
                                       dt=dt,
                                       alpha=1e-3)
xpbd.add_cons(deform, 0)
xpbd.init_rest_status()

window = gl_mesh_viewer.OpenGLMeshRenderer2D("GLFW Viewer", res=(600, 600))
window.set_mesh(mesh.v_p.to_numpy(), mesh.uvs, mesh.faces_np, texpath, idx=0)
window.set_wireframe_mode(True, color=(1.0, 0.0, 0.0, 1.0), idx=0)
window.set_mesh(lbs.v_p_rig.to_numpy(), mesh.uvs, mesh.faces_np, texpath, idx=1)
window.set_wireframe_mode(False, idx=1)


def set_movement():

  t = window.frame_count - 60

  if t > 0.0:
    angle0 = np.sin(t * 8.0 / 60) * 0.3
    lbs.set_control_angle(0, angle0)
    lbs.set_control_pos_from_parent(1, 0)
    lbs.set_control_pos_from_parent(2, 1)
    lbs.set_control_pos_from_parent(3, 2)


while window.running:
  set_movement()
  lbs.lbs()
  for sub in range(substep):
    xpbd.make_prediction()
    xpbd.preupdate_cons(0)
    xpbd.update_cons(0)
    shape.update_selected_cons(0)
    for i in range(1, 4):
      lbs.set_control_pos_from_parent(i, i - 1)
      lbs.inverse_mixed(i, 0.9)
      lbs.lbs()
      shape.update_selected_cons(i)
    xpbd.update_vel()

  window.pre_update()

  window.update_mesh(lbs.v_p.to_numpy(), idx=0)
  window.update_mesh(lbs.v_p_rig.to_numpy(), idx=1)

  glClearColor(0.96, 0.96, 0.96, 1)

  window.data_render()
  lbs.draw_display_points(fix_point=[0])

  window.show()

window.terminate()
