import numpy as np


class ClothData:

  def __init__(self, width_count, height_count, width, height) -> None:
    array_x = np.mgrid[-width / 2.0:width / 2.0:complex(width_count)]
    array_y = np.mgrid[0:height:complex(height_count)]
    x, y = np.meshgrid(array_x, array_y)
    z = np.zeros_like(x)
    self.vertex_position = np.stack([x, y, z],
                                    axis=2).reshape(width_count * height_count,
                                                    3)
    self.faces = np.zeros(shape=((width_count - 1) * (height_count - 1) * 2, 3),
                          dtype=int)
    self.edges = np.zeros(shape=((width_count + height_count - 2) +
                                 (width_count - 1) * (height_count - 1) * 3, 2),
                          dtype=int)
    for i in range(height_count - 1):
      for j in range(width_count - 1):
        base_index = i * width_count + j
        rect_index = i * (width_count - 1) + j

        face_index = rect_index * 2
        self.faces[face_index] = np.array([
            base_index, base_index + width_count + 1, base_index + width_count
        ])
        self.faces[face_index + 1] = np.array(
            [base_index, base_index + 1, base_index + width_count + 1])

        edge_index = rect_index * 3
        self.edges[edge_index] = np.array(
            [base_index, base_index + width_count + 1])
        self.edges[edge_index + 1] = np.array(
            [base_index + width_count + 1, base_index + width_count])
        self.edges[edge_index + 2] = np.array(
            [base_index + width_count + 1, base_index + 1])

    add_edge_start_index = (width_count - 1) * (height_count - 1) * 3
    for j in range(width_count - 1):
      self.edges[add_edge_start_index + j] = np.array([j, j + 1])
    for i in range(height_count - 1):
      self.edges[add_edge_start_index + width_count - 1 + i] = np.array(
          [i * width_count, (i + 1) * width_count])

    self.fixed_indices = np.array(
        [width_count * (height_count - 1), width_count * height_count - 1],
        dtype=int)
