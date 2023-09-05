import taichi as ti
import numpy as np


@ti.data_oriented
class HashGrid2D:

  def __init__(self, v_p, cell_size, bound_box, neib: float,
               max_neib: int) -> None:
    self.v_p = v_p
    self.N = v_p.shape[0]
    self.cell_size = cell_size
    self.neib = neib
    self.max_neib = max_neib
    assert (bound_box[1, 0] - bound_box[0, 0]) == (bound_box[1, 1] -
                                                   bound_box[0, 1])
    assert cell_size > neib
    self.bound_box = ti.field(2, dtype=ti.f32, shape=(2, 2))
    self.bound_box.from_numpy(bound_box)
    self.grid_count = (bound_box[1, 0] - bound_box[0, 0]) / cell_size + 1
    self.grid_num_particles = ti.field(dtype=ti.i32,
                                       shape=(self.grid_count, self.grid_count))
    self.grid_prefix = ti.field(dtype=ti.i32,
                                shape=(self.grid_count, self.grid_count))
    self.grid_tail = ti.field(dtype=ti.i32,
                              shape=(self.grid_count, self.grid_count))
    self.grid_curr = ti.field(dtype=ti.i32,
                              shape=(self.grid_count, self.grid_count))
    self.column_prefix = ti.field(dtype=ti.i32, shape=self.grid_count)
    self.grid_part_arr = ti.field(dtype=ti.i32, shape=self.N)

    self.part_neib_list = ti.field(dtype=ti.i32, shape=(self.N, self.max_neib))
    self.part_neib_count = ti.field(dtype=ti.i32, shape=self.N)

  @ti.func
  def get_grid_pos(self, input_pos):
    input_pos[0] = input_pos[0] - self.bound_box[0, 0]
    input_pos[1] = input_pos[1] - self.bound_box[1, 0]
    return ti.cast(input_pos / self.cell_size, ti.i32)

  @ti.kernel
  def dem_update(self):
    self.grid_num_particles.fill(0)
    self.column_prefix.fill(0)
    self.part_neib_count.fill(0)

    for i in self.v_p:
      cell_index = self.get_grid_pos(self.v_p[i])
      self.grid_num_particles[cell_index] += 1

    for i, j in self.grid_num_particles:
      self.column_prefix[i] += self.grid_num_particles[i, j]

    self.grid_prefix[0, 0] = 0
    ti.loop_config(serialize=True)
    for i in range(1, self.grid_count):
      self.grid_prefix[i, 0] = self.grid_prefix[i - 1,
                                                0] + self.column_prefix[i - 1]

    for i in range(self.grid_count):
      for j in range(self.grid_count):
        if j > 0:
          self.grid_prefix[i, j] = self.grid_prefix[
              i, j - 1] + self.grid_num_particles[i, j - 1]
        self.grid_tail[i,
                       j] = self.grid_prefix[i, j] + self.grid_num_particles[i,
                                                                             j]
        self.grid_curr[i, j] = self.grid_prefix[i, j]

    for i in self.v_p:
      cell_index = self.get_grid_pos(self.v_p[i])
      index = ti.atomic_add(self.grid_curr[cell_index], 1)
      self.grid_part_arr[index] = i

    for i in self.v_p:
      cell_index = self.get_grid_pos(self.v_p[i])

      x_begin = ti.max(cell_index[0] - 1, 0)
      x_end = ti.min(cell_index[0] + 2, self.grid_count)
      y_begin = ti.max(cell_index[1] - 1, 0)
      y_end = ti.min(cell_index[1] + 2, self.grid_count)

      for nei_i in range(x_begin, x_end):
        for nei_j in range(y_begin, y_end):
          begin_index = self.grid_prefix[nei_i, nei_j]
          tail_index = self.grid_tail[nei_i, nei_j]
          for index_j in range(begin_index, tail_index):
            j = self.grid_part_arr[i, index_j]
            if i != j and (self.v_p[i] - self.v_p[j]).norm() < self.neib:
              index = ti.atomic_add(self.part_neib_count[i], 1)
              self.part_neib_list[i, index] = j


@ti.data_oriented
class HashGrid3D:

  def __init__(self, v_p: ti.Field, cell_size: float, bound_box: np.ndarray,
               neib: float, max_neib: int) -> None:
    self.v_p = v_p
    self.N = v_p.shape[0]
    self.cell_size = cell_size
    self.neib = neib
    self.max_neib = max_neib
    assert cell_size > neib

    self.grid_count = ti.field(dtype=ti.i32, shape=3)

    self.grid_count.from_numpy(
        np.array([(bound_box[i, 1] - bound_box[i, 0]) / cell_size + 1
                  for i in range(3)],
                 dtype=int))
    self.bound_box = ti.field(dtype=ti.f32, shape=(3, 2))
    self.bound_box.from_numpy(bound_box)

    self.grid_num_particles = ti.field(dtype=ti.i32,
                                       shape=(self.grid_count[0],
                                              self.grid_count[1],
                                              self.grid_count[2]))
    self.grid_prefix = ti.field(dtype=ti.i32,
                                shape=(self.grid_count[0], self.grid_count[1],
                                       self.grid_count[2]))
    self.grid_tail = ti.field(dtype=ti.i32,
                              shape=(self.grid_count[0], self.grid_count[1],
                                     self.grid_count[2]))
    self.grid_curr = ti.field(dtype=ti.i32,
                              shape=(self.grid_count[0], self.grid_count[1],
                                     self.grid_count[2]))
    self.z_prefix = ti.field(dtype=ti.i32,
                             shape=(self.grid_count[0], self.grid_count[1]))
    self.grid_part_arr = ti.field(dtype=ti.i32, shape=self.N)

    self.part_neib_list = ti.field(dtype=ti.i32, shape=(self.N, self.max_neib))
    self.part_neib_count = ti.field(dtype=ti.i32, shape=self.N)

    self.dem_update()

  @ti.func
  def get_grid_pos(self, input_pos):
    input_pos[0] = input_pos[0] - self.bound_box[0, 0]
    input_pos[1] = input_pos[1] - self.bound_box[1, 0]
    input_pos[2] = input_pos[2] - self.bound_box[2, 0]
    return ti.cast(input_pos / self.cell_size, ti.i32)

  @ti.kernel
  def dem_update(self):
    self.grid_num_particles.fill(0)
    self.z_prefix.fill(0)
    self.part_neib_count.fill(0)

    for i in self.v_p:
      cell_index = self.get_grid_pos(self.v_p[i])
      self.grid_num_particles[cell_index] += 1

    for i, j, k in self.grid_num_particles:
      self.z_prefix[i, j] += self.grid_num_particles[i, j, k]

    self.grid_prefix[0, 0, 0] = 0
    ti.loop_config(serialize=True)
    for k in range(1, self.grid_count[0] * self.grid_count[1]):
      i = ti.cast(k / self.grid_count[1], ti.i32)
      j = k % self.grid_count[1]
      i0 = ti.cast((k - 1) / self.grid_count[1], ti.i32)
      j0 = (k - 1) % self.grid_count[1]
      self.grid_prefix[i, j,
                       0] = self.grid_prefix[i0, j0, 0] + self.z_prefix[i0, j0]

    for i, j in self.z_prefix:
      for k in range(self.grid_count[2]):
        if k > 0:
          self.grid_prefix[i, j, k] = self.grid_prefix[
              i, j, k - 1] + self.grid_num_particles[i, j, k - 1]
        self.grid_tail[
            i, j,
            k] = self.grid_prefix[i, j, k] + self.grid_num_particles[i, j, k]
        self.grid_curr[i, j, k] = self.grid_prefix[i, j, k]

    for i in self.v_p:
      cell_index = self.get_grid_pos(self.v_p[i])
      index = ti.atomic_add(self.grid_curr[cell_index], 1)
      self.grid_part_arr[index] = i

    for i in self.v_p:
      cell_index = self.get_grid_pos(self.v_p[i])
      x_begin = ti.max(cell_index[0] - 1, 0)
      x_end = ti.min(cell_index[0] + 2, self.grid_count[0])
      y_begin = ti.max(cell_index[1] - 1, 0)
      y_end = ti.min(cell_index[1] + 2, self.grid_count[1])
      z_begin = ti.max(cell_index[2] - 1, 0)
      z_end = ti.min(cell_index[2] + 2, self.grid_count[2])

      for nei_i in range(x_begin, x_end):
        for nei_j in range(y_begin, y_end):
          for nei_k in range(z_begin, z_end):
            for index_j in range(self.grid_prefix[nei_i, nei_j, nei_k],
                                 self.grid_tail[nei_i, nei_j, nei_k]):
              j = self.grid_part_arr[index_j]
              if i != j and (self.v_p[i] - self.v_p[j]).norm() < self.neib:
                index = ti.atomic_add(self.part_neib_count[i], 1)
                self.part_neib_list[i, index] = j

  @ti.kernel
  def dem_update_brutal(self):
    self.part_neib_count.fill(0)
    for i in self.v_p:
      for j in range(self.N):
        if i != j and (self.v_p[i] - self.v_p[j]).norm() < self.neib:
          index = ti.atomic_add(self.part_neib_count[i], 1)
          self.part_neib_list[i, index] = j
