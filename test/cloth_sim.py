import taichi as ti
import numpy as np
import os
from geom import gmesh
from cons import framework, length, bend
import time

ti.init(arch=ti.cpu)

filepath = os.path.join(os.getcwd(), 'assets', 'mesh', 'cloth_mesh.obj')
trian_mesh = gmesh.TrianMesh(filepath, dim=3, rho=1.0)

g = ti.Vector([0.0, -3.0, 5.0])
fps = 60
substep = 20
dt = 1.0 / fps

xpbd = framework.pbd_framework(g=g,
                               n_vert=trian_mesh.n_vert,
                               v_p=trian_mesh.v_p,
                               dt=dt)
length_cons = length.LengthCons(model=trian_mesh, alpha=1e-3)
bend_cons = bend.Bend3D(mesh=trian_mesh, alpha=1000.0)
xpbd.add_cons(length_cons)
xpbd.add_cons(bend_cons)
xpbd.init_rest_status()

window = ti.ui.Window("bend test", res=(600, 600), vsync=True)
scene = ti.ui.Scene()
canvas = window.get_canvas()

pos = ti.Vector([0.0, 0.0, 0.0])
camera = ti.ui.Camera()
camera.position(0.0, 1.5, 2.5)
camera.lookat(0.0, 0.5, 0.0)

cons_vert_i = ti.field(dtype=ti.i32, shape=2)
cons_vert_i.from_numpy(np.array([381, 399], dtype=int))
cons_vert_p = ti.Vector.field(3, dtype=ti.f32, shape=2)
trian_mesh.get_pos_by_index(n=2, index=cons_vert_i, pos=cons_vert_p)
trian_mesh.set_fixed_point(n=2, index=cons_vert_i)
cons_pos = cons_vert_p.to_numpy()
cons_pos_init = np.copy(cons_pos)

frame = 0

while window.running:
  begin_time = time.time()

  sys_time = frame / fps

  xpbd.make_prediction()

  cons_pos[1, 0] = cons_pos_init[1, 0] - np.math.sin(
      2 * np.math.pi * sys_time / 4.0)**2 * 0.5
  cons_vert_p.from_numpy(cons_pos)
  trian_mesh.set_pos_by_index(n=2, index=cons_vert_i, pos=cons_vert_p)

  xpbd.preupdate_cons()
  for _ in range(substep):
    xpbd.update_cons()

  xpbd.update_vel()

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

  spent_time = time.time() - begin_time

  if spent_time < 1.0 / fps:
    time.sleep(1.0 / fps - spent_time)

  frame += 1
