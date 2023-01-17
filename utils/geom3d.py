import numpy as np


def extract_surface_faces(vertex_pos: np.ndarray, tet_indices: np.ndarray):
  raw_tet_faces = []
  raw_tet_id = []
  tet_count = tet_indices.shape[0]

  for i in range(tet_count):
    p1, p2, p3, p4 = np.sort(tet_indices[i])
    raw_tet_faces.append([p1, p2, p3])
    raw_tet_faces.append([p1, p2, p4])
    raw_tet_faces.append([p1, p3, p4])
    raw_tet_faces.append([p2, p3, p4])
    raw_tet_id.extend([i] * 4)
  tet_faces = np.array(raw_tet_faces, dtype=int)
  tet_id = np.array(raw_tet_id, dtype=int)
  sort_order = np.lexsort((tet_faces[:, 0], tet_faces[:, 1], tet_faces[:, 2]))
  tet_faces = tet_faces[sort_order]
  tet_id = tet_id[sort_order]
  assert tet_faces.shape == (tet_count * 4, 3)

  surface_faces_list = []
  i = 0
  while i < tet_faces.shape[0]:
    j = i + 1
    while j < tet_faces.shape[0] and (tet_faces[j] == tet_faces[i]).all():
      j += 1
    if j == i + 1:
      surface_faces_list.append(i)
    i = j

  surface_face_indicies = tet_faces[surface_faces_list]
  surface_face_tetids = tet_id[surface_faces_list]

  for i in range(surface_face_tetids.shape[0]):
    center = np.sum(vertex_pos[tet_indices[surface_face_tetids[i]]],
                    axis=0) / 4.0
    p1, p2, p3 = surface_face_indicies[i]
    x1, x2, x3 = vertex_pos[surface_face_indicies[i]]
    if np.dot(x1 - center, np.cross(x3 - x2, x2 - x1)) > 0:
      p2, p3 = p3, p2
    surface_face_indicies[i] = [p1, p2, p3]

  return surface_face_tetids, surface_face_indicies


def compute_vertex_mass(vert_pos: np.ndarray, tet_indices: np.ndarray):
  vert_mass = np.zeros(shape=vert_pos.shape[0], dtype=float)
  vert_order = np.zeros(shape=vert_pos.shape[0], dtype=int)
  tet_mass = np.zeros(shape=tet_indices.shape[0], dtype=float)
  for i in range(tet_indices.shape[0]):
    p1, p2, p3, p4 = tet_indices[i]
    x1, x2, x3, x4 = vert_pos[[p1, p2, p3, p4]]
    tet_mass[i] = np.abs(np.dot(np.cross(x2 - x1, x3 - x1), (x4 - x1))) / 6.0
    vert_mass[[p1, p2, p3, p4]] += tet_mass[i] / 4.0
    vert_order[[p1, p2, p3, p4]] += 1
  return vert_order, vert_mass, tet_mass