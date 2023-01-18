import taichi as ti
import numpy as np
import os
from geom import gmesh
from cons import framework, length, bend
from utils import renderer

ti.init(arch=ti.cpu)

#filepath = os.path.join(os.getcwd(), 'assets', 'mesh', 'cloth_mesh.obj')
filepath = os.path.join(os.getcwd(), 'assets', 'mesh', 'plane.obj')
trian_mesh = gmesh.TrianMesh(filepath, dim=3, rho=1.0)

g = ti.Vector([0.0, -3.0, 2.0])
fps = 60
subsub = 3
substep = 5
dt = 1.0 / (fps * subsub)

xpbd = framework.pbd_framework(g=g,
                               n_vert=trian_mesh.n_vert,
                               v_p=trian_mesh.v_p,
                               dt=dt)
length_cons = length.LengthCons(model=trian_mesh, dt=dt, alpha=0.0)
bend_cons = bend.Bend3D(mesh=trian_mesh, dt=dt, alpha=800.0)
xpbd.add_cons(length_cons)
xpbd.add_cons(bend_cons)
xpbd.init_rest_status()

tirender = renderer.TaichiMeshRenderer("cloth sim",
                                       res=(600, 600),
                                       mesh=trian_mesh,
                                       fps=fps,
                                       cameraPos=(1.5, 1.5, 4.5),
                                       cameraLookat=(0.0, 0.5, 1.0),
                                       wireframe=True)

cons_vert_i = ti.field(dtype=ti.i32, shape=2)
#cons_vert_i.from_numpy(np.array([381, 399], dtype=int))
cons_vert_i.from_numpy(np.array([9, 87], dtype=int))
cons_vert_p = ti.Vector.field(3, dtype=ti.f32, shape=2)
trian_mesh.get_pos_by_index(n=2, index=cons_vert_i, pos=cons_vert_p)
trian_mesh.set_fixed_point(n=2, index=cons_vert_i)
cons_pos = cons_vert_p.to_numpy()
cons_pos_init = np.copy(cons_pos)

while tirender.window.running:
  for sub in range(subsub):
    xpbd.make_prediction()

    cons_pos[1, 0] = cons_pos_init[1, 0] - np.math.sin(
        2 * np.math.pi * tirender.time / 5.0)**2 * 0.6
    cons_vert_p.from_numpy(cons_pos)
    trian_mesh.set_pos_by_index(n=2, index=cons_vert_i, pos=cons_vert_p)

    xpbd.preupdate_cons()
    for _ in range(substep):
      xpbd.update_cons()
    xpbd.update_vel()

  tirender.render()
