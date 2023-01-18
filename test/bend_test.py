import taichi as ti
import numpy as np
from cons import framework, bend
from geom import gmesh
import os

ti.init(arch=ti.cpu)

filepath = os.path.join(os.getcwd(), 'assets', 'libigl_models',
                        '2triangles.off')

trian_mesh = gmesh.TrianMesh(filepath, dim=3, rho=1.0)

g = ti.Vector([0.0, -2.0, 0.0])
fps = 60
substep = 20
dt = 1.0 / fps

pbd = framework.pbd_framework(g=g,
                              n_vert=trian_mesh.n_vert,
                              v_p=trian_mesh.v_p,
                              dt=dt)
bend_cons = bend.Bend3D(mesh=trian_mesh, dt=dt, alpha=1e-2)
pbd.add_cons(bend_cons)

window = ti.ui.Window("bend test", res=(600, 600), vsync=True)
scene = ti.ui.Scene()
canvas = window.get_canvas()

pos = ti.Vector([0.0, 0.0, 0.0])
camera = ti.ui.Camera()
camera.position(0.86, 3.13, 2.66)
camera.lookat(0.92, 2.43, 1.95)

pbd.init_rest_status()
bend_cons.edge_rest_angle.fill(ti.math.pi)

while window.running:

  if window.get_event(ti.ui.PRESS):
    if window.event.key == 'r':
      pbd.preupdate_cons()
    if window.event.key == 't':
      pbd.update_cons()

  camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
  scene.set_camera(camera)
  scene.ambient_light((0.8, 0.8, 0.8))
  scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
  scene.mesh(trian_mesh.v_p,
             trian_mesh.f_i,
             color=(0.8, 0.2, 0.1),
             show_wireframe=True)
  canvas.scene(scene)

  window.show()