import taichi as ti
import numpy as np
import geom.hashgrid, geom.particle3d
import cons.dem3d, cons.framework
import geom.obj
import utils.renderer

ti.init(arch=ti.cpu)

N = 4000
dt = 1.0 / 60
radius = 0.04

bound_box = np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])

pos = np.array([[
    np.random.uniform(bound_box[0, 0], bound_box[0, 1]),
    np.random.uniform(bound_box[1, 0], bound_box[1, 1]),
    np.random.uniform(bound_box[2, 0], bound_box[2, 1])
] for i in range(N)])

position = ti.Vector.field(3, dtype=ti.f32, shape=N)
position.from_numpy(pos)
part3d = geom.particle3d.Particles3D(pos=position, radius=radius)
hash3d = geom.hashgrid.HashGrid3D(part3d.v_p,
                                  radius * 2.01,
                                  bound_box,
                                  neib=radius * 2.0,
                                  max_neib=64)

pbd = cons.framework.pbd_framework(n_vert=N,
                                   v_p=part3d.v_p,
                                   g=ti.Vector([0.0, -0.6, 0.0]),
                                   dt=dt)
dem_part = cons.dem3d.DEM3D(v_p=part3d.v_p,
                            v_p_cache=pbd.v_p_cache,
                            hash3d=hash3d,
                            r=radius,
                            fric_para=0.1)
pbd.add_cons(new_cons=dem_part)

box = geom.obj.BoundBox3D(bound_box=bound_box,
                          padding=radius,
                          bound_epsilon=1e-4)
pbd.add_collision(box.collision)

tirender = utils.renderer.TaichiRenderer3D("Discrete Element Method 3D Test",
                                           res=(800, 800),
                                           fps=60,
                                           cameraPos=(5.0, 5.0, 5.0),
                                           cameraLookat=(1.0, 1.0, 1.0))
tirender.add_scene_render_draw(box.get_render_draw())
tirender.add_scene_render_draw(part3d.get_render_draw())

pbd.init_rest_status()

while tirender.window.running:
  tirender.handle_input()
  pbd.make_prediction()
  pbd.preupdate_cons()
  hash3d.dem_update()
  for _ in range(10):
    pbd.update_cons()
  pbd.update_vel()
  tirender.render()