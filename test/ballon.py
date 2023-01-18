import taichi as ti
import os
from geom import gmesh
from cons import framework, length, bend, volume
from utils import renderer

ti.init(arch=ti.cpu)

filepath = os.path.join(os.getcwd(), 'assets', 'mesh', 'capsuleY_lowpoly.obj')
trian_mesh = gmesh.TrianMesh(filepath, dim=3, rho=1.0)

g = ti.Vector([0.0, -9.0, 0.0])
fps = 60
subsub = 10
substep = 2
dt = 1.0 / (fps * subsub)

xpbd = framework.pbd_framework(g=g,
                               n_vert=trian_mesh.n_vert,
                               v_p=trian_mesh.v_p,
                               dt=dt)
length_cons = length.LengthCons(model=trian_mesh, dt=dt, alpha=1e-4)
bend_cons = bend.Bend3D(mesh=trian_mesh, dt=dt, alpha=800.0)
volume_cons = volume.Volume(mesh=trian_mesh, dt=dt, alpha=0.0)
xpbd.add_cons(length_cons)
xpbd.add_cons(bend_cons)
xpbd.add_cons(volume_cons)
xpbd.init_rest_status()


@ti.kernel
def collision(minY: float, pos: ti.template()):
  for i in pos:
    if pos[i][1] < minY:
      pos[i][1] = minY


tirender = renderer.TaichiMeshRenderer("ballon sim",
                                       res=(600, 600),
                                       mesh=trian_mesh,
                                       fps=fps,
                                       cameraPos=(15.0, -2.0, -15.0),
                                       cameraLookat=(0.0, -2.0, 0.0),
                                       wireframe=True)

while tirender.window.running:
  for sub in range(subsub):
    xpbd.make_prediction()
    collision(-5.0, trian_mesh.v_p)
    xpbd.preupdate_cons()
    for _ in range(substep):
      xpbd.update_cons()
    xpbd.update_vel()

  tirender.render()