import taichi as ti
import os
from geom import gmesh, obj
from cons import framework, length, bend, volume
from utils import renderer, parser

ti.init(arch=ti.cpu)

filepath = os.path.join(os.getcwd(), 'assets', 'mesh', 'capsuleY_lowpoly.obj')
verts, faces = parser.obj_parser(filepath)
mesh = gmesh.TrianMesh(verts, faces, dim=3, rho=1.0)

g = ti.Vector([0.0, -9.0, 0.0])
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
                                alpha=1e-4)
bend_cons = bend.Bend3D(mesh.v_p,
                        mesh.v_p_ref,
                        mesh.e_i,
                        mesh.e_sidei,
                        mesh.v_invm,
                        dt=dt,
                        alpha=800.0)
volume_cons = volume.Volume(mesh.v_p,
                            mesh.v_p_ref,
                            mesh.v_p_delta,
                            mesh.f_i,
                            mesh.v_invm,
                            dt=dt,
                            alpha=0.0)
xpbd.add_cons(length_cons)
xpbd.add_cons(bend_cons)
xpbd.add_cons(volume_cons)
xpbd.init_rest_status()

ground = obj.Quad((10.0, 0.0, 0.0), (0.0, 0.0, -10.0), -5.0)
xpbd.add_collision(ground.collision)

tirender = renderer.TaichiRenderer3D("ballon sim",
                                     res=(600, 600),
                                     fps=fps,
                                     cameraPos=(10.0, 5.0, -10.0),
                                     cameraLookat=(0.0, -2.0, 0.0))
tirender.add_scene_render_draw(mesh.get_render_draw(color=(0.7, 0.2, 0.0)))
tirender.add_scene_render_draw(ground.get_render_draw())

while tirender.window.running:
  for sub in range(substep):
    xpbd.make_prediction()
    xpbd.preupdate_cons()
    for _ in range(solve_iters):
      xpbd.update_cons()
    xpbd.update_vel()

  tirender.render()