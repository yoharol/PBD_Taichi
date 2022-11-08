import numpy as np


def edge_extractor(face_indices: np.ndarray):
  face_count = face_indices.shape[0]

  raw_edge = []
  for i in range(face_count):
    for k1 in range(3):
      k2 = (k1 + 1) % 3
      p1 = face_indices[i, k1]
      p2 = face_indices[i, k2]
      raw_edge.append([min(p1, p2), max(p1, p2), i])

  raw_edge_np = np.array(raw_edge, dtype=int)
  raw_edge_np = raw_edge_np[np.lexsort((raw_edge_np[:, 0], raw_edge_np[:, 1]))]

  raw_edge_indices = []
  raw_edge_sides = []
  raw_edge_neib = []
  raw_face_edges = [[] for _ in range(face_count)]

  compare = lambda x, y: all(raw_edge_np[x, :2] == raw_edge_np[y, :2])

  i = 0
  while i < raw_edge_np.shape[0]:
    assert not (i > 0 and compare(i, i - 1))
    if i + 1 < raw_edge_np.shape[0] and compare(i, i + 1):
      raw_edge_indices.append([raw_edge_np[i, 0], raw_edge_np[i, 1]])
      raw_edge_neib.append([raw_edge_np[i, 2], raw_edge_np[i + 1, 2]])
      face_i1 = face_indices[raw_edge_np[i, 2]]
      face_i2 = face_indices[raw_edge_np[i + 1, 2]]
      raw_edge_sides.append([
          np.setdiff1d(face_i1, face_i2)[0],
          np.setdiff1d(face_i2, face_i1)[0]
      ])
      edge_index = len(raw_edge_indices) - 1
      raw_face_edges[raw_edge_np[i, 2]].append(edge_index)
      raw_face_edges[raw_edge_np[i + 1, 2]].append(edge_index)
      i += 2
    else:
      raw_edge_indices.append([raw_edge_np[i, 0], raw_edge_np[i, 1]])
      raw_edge_neib.append([raw_edge_np[i, 2], -1])
      raw_edge_sides.append([
          np.setdiff1d(face_indices[raw_edge_np[i, 2]], raw_edge_np[i, :2])[0],
          -1
      ])
      edge_index = len(raw_edge_indices) - 1
      raw_face_edges[raw_edge_np[i, 2]].append(edge_index)
      i += 1
  edge_indices = np.array(raw_edge_indices)
  edge_sides = np.array(raw_edge_sides)
  edge_neib = np.array(raw_edge_neib)
  face_edges = np.array(raw_face_edges)
  assert edge_indices.shape == edge_sides.shape and edge_sides.shape == edge_neib.shape
  assert face_edges.shape == (face_count, 3)
  return edge_indices, edge_sides, edge_neib, face_edges


def mesh_verify(edge_indices: np.ndarray, edge_sides: np.ndarray,
                edge_neib: np.ndarray, face_indices: np.ndarray,
                face_edges: np.ndarray):
  face_counts = face_indices.shape[0]
  edge_counts = edge_indices.shape[0]

  for i in range(face_counts):
    assert np.array([edge_indices[face_edges[i, k]] for k in range(3)
                    ]).flatten().sort() == np.repeat(face_indices[i], 2).sort()

  for i in range(edge_counts):
    if edge_neib[i, 1] == -1:
      assert edge_sides[i, 1] == -1
      assert (np.sort(np.concatenate(
          ([edge_indices[i],
            edge_sides[i,
                       [0]]]))) == np.sort(face_indices[edge_neib[i,
                                                                  0]])).all()
    else:
      assert (np.sort(
          np.concatenate(
              ([edge_indices[i], edge_sides[i]])).flatten()) == np.sort(
                  np.unique(
                      np.concatenate((face_indices[edge_neib[i, 0]],
                                      face_indices[edge_neib[i, 1]]))))).all()


def compute_rest_status(vert_pos: np.ndarray, edge_indices: np.ndarray,
                        edge_sides: np.ndarray):
  edge_count = edge_indices.shape[0]
  norm = lambda x: np.sqrt(np.sum(x**2))
  rest_length = np.zeros(shape=edge_count, dtype=np.float32)
  rest_angle = np.zeros(shape=edge_count, dtype=np.float32)
  for i in range(edge_count):
    p1, p2 = edge_indices[i]
    p3, p4 = edge_sides[i]
    x1, x2, x3, x4 = vert_pos[[p1, p2, p3, p4]]
    rest_length[i] = norm(x2 - x1)

    tmp_x1 = np.cross(x2 - x1, x3 - x1)
    tmp_x2 = np.cross(x2 - x1, x4 - x1)
    rest_angle[i] = np.arccos(
        np.dot(tmp_x1 / norm(tmp_x1), tmp_x2 / norm(tmp_x2)))
  return rest_length, rest_angle


def compute_vert_mass(vert_pos: np.ndarray, face_indices: np.ndarray):
  vert_count = vert_pos.shape[0]
  face_count = face_indices.shape[0]

  vert_order = np.zeros(shape=vert_count, dtype=int)
  vert_mass = np.zeros(shape=vert_count, dtype=np.float32)
  face_mass = np.zeros(shape=face_count, dtype=np.float32)

  norm = lambda x: np.sqrt(np.sum(x**2))

  for i in range(face_count):
    p1, p2, p3 = face_indices[i]
    x1, x2, x3 = vert_pos[[p1, p2, p3]]
    face_mass[i] = 0.5 * norm(np.cross(x2 - x1, x3 - x1))
    vert_mass[[p1, p2, p3]] += face_mass[i] / 3.0
    vert_order[[p1, p2, p3]] += 1

  return vert_order, vert_mass, face_mass
