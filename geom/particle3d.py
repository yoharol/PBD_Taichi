import taichi as ti
import numpy as np


@ti.data_oriented
class Particles3D:

  def __init__(self, pos: ti.MatrixField, radius: float, color_buffer=False):
    self.n = pos.shape[0]
    self.radius = radius
    self.v_p = pos
    self.color_buffer = color_buffer
    self.bound_box = ti.field(dtype=ti.f32, shape=(3, 2))
    if color_buffer:
      self.per_vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=self.n)

  @ti.kernel
  def get_bound_box(self):
    for k in ti.static(range(3)):
      self.bound_box[k, 0] = self.v_p[0][k] - self.radius
      self.bound_box[k, 1] = self.v_p[0][k] + self.radius
    for i in range(1, self.n):
      for k in ti.static(range(3)):
        ti.atomic_min(self.bound_box[k, 0], self.v_p[i][k] - self.radius)
        ti.atomic_max(self.bound_box[k, 1], self.v_p[i][k] + self.radius)

  def get_render_draw(self, color=(0.0, 0.7, 0.7)):
    if self.color_buffer:

      def render_draw(scene):
        scene.particles(self.v_p, self.radius, self.per_vertex_color)

      return render_draw
    else:

      def render_draw(scene):
        scene.particles(self.v_p, self.radius, color=color)

      return render_draw
