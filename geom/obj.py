import taichi as ti
import numpy as np


@ti.data_oriented
class Quad:

  def __init__(self, axis1, axis2, pos) -> None:
    self.axis = ti.Vector.field(3, dtype=ti.f32, shape=2)
    self.axis.from_numpy(np.array([axis1, axis2], dtype=np.single))
    self.v_p = ti.Vector.field(3, dtype=ti.f32, shape=4)
    self.pos = ti.field(dtype=ti.f32, shape=())
    self.pos[None] = pos
    self.f_i = ti.field(dtype=ti.i32, shape=6)
    self.f_i.from_numpy(np.array([0, 1, 2, 0, 2, 3], dtype=int))
    self.set_quad()

  @ti.kernel
  def set_quad(self):
    normal = self.axis[0].cross(self.axis[1]).normalized()

    self.v_p[
        0] = normal * self.pos[None] - 0.5 * self.axis[0] - 0.5 * self.axis[1]
    self.v_p[1] = self.v_p[0] + self.axis[0]
    self.v_p[2] = self.v_p[1] + self.axis[1]
    self.v_p[3] = self.v_p[0] + self.axis[1]

  def set_pos(self, pos: float):
    self.pos[None] = pos
    self.set_quad()

  @ti.kernel
  def collision(self, pos: ti.template()):
    normal = self.axis[0].cross(self.axis[1]).normalized()
    for i in pos:
      dis = pos[i].dot(normal) - self.pos[None]
      if dis < 0:
        pos[i] = pos[i] - normal * dis

  def get_render_draw(self, color=(0.5, 0.5, 0.5)):

    def render_draw(scene):
      scene.mesh(self.v_p, self.f_i, color=color)

    return render_draw


@ti.data_oriented
class BoundBox3D:

  def __init__(self,
               bound_box: np.ndarray,
               padding=0.0,
               bound_epsilon=0.0) -> None:
    self.bound_box = ti.field(dtype=ti.f32, shape=(3, 2))
    self.bound_box.from_numpy(bound_box)
    self.padding = padding
    self.bound_epsilon = bound_epsilon
    self.box_vert = ti.Vector.field(3, dtype=ti.f32, shape=6)
    self.box_edge = ti.field(dtype=ti.i32, shape=24)
    self.box_edge.from_numpy(
        np.array([
            0, 2, 2, 3, 0, 1, 1, 3, 0, 4, 2, 6, 3, 7, 1, 5, 4, 6, 6, 7, 4, 5, 5,
            7
        ]))
    self.set_box()

  def __init__(self,
               bound_box: ti.Field,
               padding=0.0,
               bound_epsilon=0.0) -> None:
    self.bound_box = bound_box
    print(self.bound_box.shape)
    self.padding = padding
    self.bound_epsilon = bound_epsilon
    self.box_vert = ti.Vector.field(3, dtype=ti.f32, shape=8)
    self.box_edge = ti.field(dtype=ti.i32, shape=24)
    self.box_edge.from_numpy(
        np.array([
            0, 2, 2, 3, 0, 1, 1, 3, 0, 4, 2, 6, 3, 7, 1, 5, 4, 6, 6, 7, 4, 5, 5,
            7
        ]))
    self.set_box()

  @ti.kernel
  def set_box(self):
    for I in ti.grouped(ti.ndrange(2, 2, 2)):
      index = I[0] * 4 + I[1] * 2 + I[2]
      for k in ti.static(range(3)):
        t = ti.cast(I[k], ti.f32)
        self.box_vert[index][k] = (
            1.0 - t) * self.bound_box[k, 0] + t * self.bound_box[k, 1]

  @ti.kernel
  def collision(self, pos: ti.template()):
    for i in pos:
      for k in ti.static(range(3)):
        if pos[i][k] < self.bound_box[k, 0] + self.padding:
          pos[i][k] = self.bound_box[
              k, 0] + self.padding + self.bound_epsilon * ti.random()
        if pos[i][k] > self.bound_box[k, 1] - self.padding:
          pos[i][k] = self.bound_box[
              k, 1] - self.padding - self.bound_epsilon * ti.random()

  def get_render_draw(self, color=(0.5, 0.5, 0.5), width=0.05):

    def render_draw(scene):
      scene.lines(self.box_vert, width, self.box_edge, color=color)

    return render_draw


@ti.data_oriented
class BoundBox2D:

  def __init__(self,
               bound_box: np.ndarray,
               padding=0.0,
               bound_epsilon=0.0) -> None:
    self.bound_box = ti.field(dtype=ti.f32, shape=(2, 2))
    self.bound_box.from_numpy(bound_box)
    self.padding = padding
    self.bound_epsilon = bound_epsilon
    self.box_vert = ti.Vector.field(2, dtype=ti.f32, shape=4)
    self.box_edge = ti.field(dtype=ti.i32, shape=8)
    self.box_edge.from_numpy(np.array([0, 1, 1, 2, 2, 3, 3, 0]))
    self.set_box()

  """def __init__(self,
               bound_box: ti.Field,
               padding=0.0,
               bound_epsilon=0.0) -> None:
    self.bound_box = bound_box
    print(self.bound_box.shape)
    self.padding = padding
    self.bound_epsilon = bound_epsilon
    self.box_vert = ti.Vector.field(3, dtype=ti.f32, shape=8)
    self.box_edge = ti.field(dtype=ti.i32, shape=24)
    self.box_edge.from_numpy(np.array([0, 1, 1, 2, 2, 3]))
    self.set_box()"""

  @ti.kernel
  def set_box(self):
    for I in ti.grouped(ti.ndrange(2, 2)):
      index = I[0] * 2 + I[1]
      for k in ti.static(range(2)):
        t = ti.cast(I[k], ti.f32)
        self.box_vert[index][k] = (
            1.0 - t) * self.bound_box[k, 0] + t * self.bound_box[k, 1]

  @ti.kernel
  def collision(self, pos: ti.template()):
    for i in pos:
      for k in ti.static(range(2)):
        if pos[i][k] < self.bound_box[k, 0] + self.padding:
          pos[i][k] = self.bound_box[
              k, 0] + self.padding + self.bound_epsilon * ti.random()
        if pos[i][k] > self.bound_box[k, 1] - self.padding:
          pos[i][k] = self.bound_box[
              k, 1] - self.padding - self.bound_epsilon * ti.random()

  """def get_render_draw(self, color=(0.5, 0.5, 0.5), width=0.05):

    def render_draw(scene):
      scene.lines(self.box_vert, width, self.box_edge, color=color)
    return render_draw"""
