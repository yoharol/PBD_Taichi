import numpy as np


def generate_flat_cloth_mesh(width_count: int, height_count: int, width: float,
                             height: float):
  array_x = np.mgrid[-width / 2.0:width / 2.0:complex(width_count)]
  array_y = np.mgrid[0:height:complex(height_count)]
  x, y = np.meshgrid(array_x, array_y)
  z = np.zeros_like(x)
  vertex_position = np.stack([x, y, z],
                             axis=2).reshape(width_count * height_count, 3)
  faces = np.zeros(shape=((width_count - 1) * (height_count - 1) * 2, 3),
                   dtype=int)
  for i in range(height_count - 1):
    for j in range(width_count - 1):
      base_index = i * width_count + j
      rect_index = i * (width_count - 1) + j

      face_index = rect_index * 2
      faces[face_index] = np.array(
          [base_index, base_index + width_count + 1, base_index + width_count])
      faces[face_index + 1] = np.array(
          [base_index, base_index + 1, base_index + width_count + 1])

  return vertex_position, face_index
