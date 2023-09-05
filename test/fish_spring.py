import taichi as ti
import numpy as np
import glfw
from OpenGL.GL import *

from cons import framework, length
from utils import parser, gl_mesh_viewer
from geom import gmesh, obj

ti.init(arch=ti.cpu, cpu_max_num_threads=1)

parent_folder = "assets/fish2bones"
usdpath = f"{parent_folder}/fish.usda"
texpath = f"{parent_folder}/Tex.png"

verts, faces, uvs = parser.usd_parser(usdpath, "/root/mesh")
mesh = gmesh.TrianMesh(verts,
                       faces,
                       dim=2,
                       rho=1.0,
                       scale=0.8,
                       repose=(0.1, 0.2))
mesh.set_texture_uv(uvs)

g = ti.Vector([0.0, -1.5])
fps = 60
substep = 10
sovle_step = 1
dt = 1.0 / (fps * substep)

box2d = obj.BoundBox2D(bound_box=np.array([[0.0, 1.0], [0.0, 1.0]]),
                       padding=0.02,
                       bound_epsilon=1e-6)
xpbd = framework.pbd_framework(g=g,
                               n_vert=mesh.n_vert,
                               v_p=mesh.v_p,
                               dt=dt,
                               damp=1.0)
length = length.LengthCons(v_p=mesh.v_p,
                           v_p_ref=mesh.v_p_ref,
                           e_i=mesh.e_i,
                           v_invm=mesh.v_invm,
                           dt=dt,
                           alpha=1e-2)
xpbd.add_cons(length)
xpbd.add_collision(box2d.collision)
xpbd.init_rest_status()

window = gl_mesh_viewer.OpenGLMeshRenderer2D("GLFW Viewer", res=(700, 700))
window.set_mesh(mesh.v_p.to_numpy(), mesh.uvs, mesh.faces_np, texpath)

view_mode = [True]  # wireframe mode


def set_view_mode(key, scancode, action, mods, ik=view_mode):
  if action == glfw.PRESS and key == glfw.KEY_F:
    ik[0] = not ik[0]


window.add_key_callback(set_view_mode)

while window.running:

  for sub in range(substep):
    xpbd.make_prediction()
    xpbd.preupdate_cons()
    for _ in range(sovle_step):
      xpbd.update_cons()
    xpbd.update_vel()

  window.pre_update()

  if view_mode[0]:
    window.set_wireframe_mode(True, color=(1.0, 0.0, 0.0, 1.0))
  else:
    window.set_wireframe_mode(False)

  glClearColor(0.96, 0.96, 0.96, 1)
  window.update_mesh(mesh.v_p.to_numpy())

  window.data_render()
  window.show()

window.terminate()
