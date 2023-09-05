import taichi as ti
import numpy as np
import os
from geom import gmesh
from cons import framework, length, bend
from utils import renderer, parser

ti.init(arch=ti.cpu)

filepath = os.path.join(os.getcwd(), 'assets', 'mesh', 'plane.obj')
verts, faces = parser.obj_parser(filepath)
mesh = gmesh.TrianMesh(verts, faces, dim=3, rho=1.0)

g = ti.Vector([0.0, -3.0, 2.0])
fps = 60
substep = 15
solve_iters = 1
dt = 1.0 / (fps * substep)

xpbd = framework.pbd_framework(g=g, n_vert=mesh.n_vert, v_p=mesh.v_p, dt=dt)
length_cons = length.LengthCons(mesh.v_p,
                                mesh.v_p_ref,
                                mesh.e_i,
                                mesh.v_invm,
                                dt=dt,
                                alpha=0.0)
bend_cons = bend.Bend3D(mesh.v_p,
                        mesh.v_p_ref,
                        mesh.e_i,
                        mesh.e_sidei,
                        mesh.v_invm,
                        dt=dt,
                        alpha=200.0)
xpbd.add_cons(length_cons)
xpbd.add_cons(bend_cons)
xpbd.init_rest_status()

tirender = renderer.TaichiRenderer3D("cloth sim",
                                     res=(600, 600),
                                     fps=fps,
                                     cameraPos=(1.5, 1.5, 4.5),
                                     cameraLookat=(0.0, 0.5, 1.0))
tirender.add_scene_render_draw(mesh.get_render_draw())

cons_vert_i = ti.field(dtype=ti.i32, shape=2)
cons_vert_i.from_numpy(np.array([9, 87], dtype=int))
cons_vert_p = ti.Vector.field(3, dtype=ti.f32, shape=2)
mesh.get_pos_by_index(n=2, index=cons_vert_i, pos=cons_vert_p)
mesh.set_fixed_point(n=2, index=cons_vert_i)
cons_pos = cons_vert_p.to_numpy()
cons_pos_init = np.copy(cons_pos)

while tirender.window.running:
  for sub in range(substep):
    # init XPBD solver
    xpbd.make_prediction()

    # set fixed points
    cons_pos[1, 0] = cons_pos_init[1, 0] - np.math.sin(
        2 * np.math.pi * tirender.time / 5.0)**2 * 0.6
    cons_vert_p.from_numpy(cons_pos)
    mesh.set_pos_by_index(n=2, index=cons_vert_i, pos=cons_vert_p)

    # solve constraints
    xpbd.preupdate_cons()
    for _ in range(solve_iters):
      xpbd.update_cons()
    xpbd.update_vel()

  tirender.render()
