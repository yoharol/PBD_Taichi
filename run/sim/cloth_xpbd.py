import taichi as ti
import math
import numpy as np
import sim.cloth as cloth


def sim(ref_path, ref_prim, output_path, output_prim, export_path):

  cons_vert_index = np.array([380, 399])

  cloth_sim = cloth.ClothSim(ref_path,
                             ref_prim,
                             output_path,
                             output_prim,
                             export_path,
                             cons_vert_index,
                             render_color=True)

  cons_vert_pos = cloth_sim.GetPos(cons_vert_index)
  cons_vert_pos_initial = np.array(cons_vert_pos)

  for t in range(cloth_sim.frames):
    time = t / cloth_sim.fps
    cons_vert_pos[1, 0] = cons_vert_pos_initial[1, 0] - math.sin(
        2 * math.pi * time / 4.0)**2 * 0.5
    cloth_sim.SetConstaintPos(cons_vert_pos)

    cloth_sim.generate_prediction()
    for _ in range(cloth_sim.substeps):
      cloth_sim.solve_length_constraints()
      #cloth_sim.solve_angle_constraints()
    cloth_sim.update_vel()

    cloth_sim.render_frame(t)
  cloth_sim.save()
