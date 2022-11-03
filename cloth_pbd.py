import taichi as ti
import numpy as np
from assets.cloth_data_generator import ClothData
from pxr import Usd, UsdGeom
from os import path

ti.init(arch=ti.cpu)

cloth_size = [1.0, 1.0]
cloth_grid = [20, 20]
frame_rate = 30
substeps = 10
animation_time = 3.0

Alpha = 0.0

gravity = -3.0
wind = ti.Vector([0.0, 0.0, 6.0])

output_path = path.join(path.dirname(__file__), 'outputs')


@ti.data_oriented
class Cloth:

  def __init__(self) -> None:
    data = ClothData(cloth_grid[0], cloth_grid[1], cloth_size[0], cloth_size[1])

    self.w = cloth_grid[0]
    self.h = cloth_grid[1]

    self.g = ti.Vector([0.0, gravity, 0.0])
    self.wind = wind
    self.dt = 1.0 / frame_rate
    self.m = 1.0

    self.total_frames = frame_rate * int(animation_time)

    self.x = ti.Vector.field(3,
                             dtype=ti.f32,
                             shape=data.vertex_position.shape[0])
    self.x.from_numpy(data.vertex_position)

    self.faces = ti.Vector.field(3, dtype=ti.i32, shape=data.faces.shape[0])
    self.faces.from_numpy(data.faces)
    self.edges = ti.Vector.field(2, dtype=ti.i32, shape=data.edges.shape[0])
    self.edges.from_numpy(data.edges)
    self.fixed_indices = ti.field(dtype=ti.i32, shape=data.fixed_indices.shape)
    self.fixed_indices.from_numpy(data.fixed_indices)
    self.num_fixes = self.fixed_indices.shape[0]

    self.vertices_count = data.vertex_position.shape[0]
    self.faces_count = data.faces.shape[0]
    self.edges_count = data.edges.shape[0]

    self.v = ti.Vector.field(3,
                             dtype=ti.f32,
                             shape=data.vertex_position.shape[0])
    self.v.fill(ti.Vector([0.0, 0.0, 0.0]))
    self.cache_x = ti.Vector.field(3,
                                   dtype=ti.f32,
                                   shape=data.vertex_position.shape[0])
    self.rest_length = ti.field(dtype=ti.f32, shape=self.edges_count)
    self.edges_lambda = ti.field(dtype=ti.f32, shape=self.edges_count)

    self.stage = Usd.Stage.CreateNew(path.join(output_path, 'cloth.usd'))
    self.stage.SetStartTimeCode(1)
    self.stage.SetEndTimeCode(frame_rate * int(animation_time))
    self.stage.SetTimeCodesPerSecond(frame_rate)
    UsdGeom.Xform.Define(self.stage, '/root')
    self.cloth_mesh = UsdGeom.Mesh.Define(self.stage, '/root/cloth')
    self.cloth_mesh.GetPointsAttr().Set(data.vertex_position)
    self.cloth_mesh.GetFaceVertexIndicesAttr().Set(data.faces)
    self.cloth_mesh.GetFaceVertexCountsAttr().Set([3] * data.faces.shape[0])
    self.cloth_mesh.GetSubdivisionSchemeAttr().Set('none')
    self.cloth_mesh.GetDoubleSidedAttr().Set(True)

    self.compute_rest_length()

  def RenderFrame(self, timecode):
    self.cloth_mesh.GetPointsAttr().Set(value=self.x.to_numpy(), time=timecode)

  def Save(self):
    self.stage.GetRootLayer().Save()

  @ti.func
  def find_neighbour(self, edge_index):
    if edge_index >= (self.w - 1) * (self.h - 1) * 3:
      return -1, -1
    rect_index = int(edge_index / 3)
    column_index = rect_index % (self.w - 1)
    row_index = int(rect_index / (self.w - 1))
    edge_local_index = edge_index % 3
    if edge_local_index == 0:
      return rect_index * 2, rect_index * 2 + 1
    elif edge_local_index == 1:
      if row_index < self.h - 2:
        return rect_index * 2, (rect_index + self.w - 1) * 2
      else:
        return -1, -1
    elif edge_local_index == 2:
      if column_index < self.w - 2:
        return rect_index * 2 + 1, rect_index * 2 + 2
      else:
        return -1, -1

  @ti.func
  def get_face_normal(self, face_index):
    vec0 = self.x[self.faces[face_index][0]]
    vec1 = self.x[self.faces[face_index][1]]
    vec2 = self.x[self.faces[face_index][2]]

    return ((vec1 - vec0).cross(vec2 - vec0)).normalized()

  @ti.kernel
  def compute_rest_length(self):
    for i in self.edges:
      self.rest_length[i] = (self.x[self.edges[i][0]] -
                             self.x[self.edges[i][1]]).norm()

  @ti.kernel
  def generate_prediction(self):
    for i in self.x:
      self.cache_x[i] = self.x[i]

      fixed = False
      for l in range(self.num_fixes):
        if i == self.fixed_indices[l]:
          fixed = True
          break
      if not fixed:
        self.x[i] = self.x[i] + self.v[i] * self.dt + (
            self.g + self.wind) * self.dt * self.dt
    self.edges_lambda.fill(0.0)

  @ti.kernel
  def update_vel(self):
    for i in self.x:
      self.v[i] = (self.x[i] - self.cache_x[i]) / self.dt

  @ti.kernel
  def xpbd(self):
    for k in self.edges:
      i = self.edges[k][0]
      j = self.edges[k][1]
      x_ij = self.x[i] - self.x[j]
      C_ij = x_ij.norm() - self.rest_length[k]
      w_i = 1.0 / self.m
      w_j = 1.0 / self.m
      for l in range(self.num_fixes):
        if self.fixed_indices[l] == i:
          w_i = 0.0
        if self.fixed_indices[l] == j:
          w_j = 0.0
      delta_lambda = -(C_ij + Alpha * self.edges_lambda[k] /
                       (self.dt * self.dt)) / (w_i + w_j + Alpha /
                                               (self.dt * self.dt))
      self.edges_lambda[k] += delta_lambda
      x_ij = x_ij / x_ij.norm()
      self.x[i] += w_i * delta_lambda * x_ij
      self.x[j] += -w_j * delta_lambda * x_ij


def main():
  cloth = Cloth()

  for frame in range(cloth.total_frames):
    cloth.generate_prediction()
    for _ in range(substeps):
      cloth.xpbd()
    cloth.update_vel()
    cloth.RenderFrame(frame + 1)
  cloth.Save()


if __name__ == '__main__':
  main()
