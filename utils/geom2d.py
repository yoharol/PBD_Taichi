from utils.mathlib import *
import numpy as np
import taichi as ti


def edge_extractor(face_indices: np.ndarray):
  assert face_indices.ndim == 1, f"face indices wrong dim as{face_indices.ndim}"

  face_count = face_indices.shape[0] // 3

  raw_edge = []
  for i in range(face_count):
    for k1 in range(3):
      k2 = (k1 + 1) % 3
      p1 = face_indices[i * 3 + k1]
      p2 = face_indices[i * 3 + k2]
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
      face_i1 = face_indices[raw_edge_np[i, 2] * 3:raw_edge_np[i, 2] * 3 + 3]
      face_i2 = face_indices[raw_edge_np[i + 1, 2] *
                             3:raw_edge_np[i + 1, 2] * 3 + 3]
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
          np.setdiff1d(
              face_indices[raw_edge_np[i, 2] * 3:raw_edge_np[i, 2] * 3 + 3],
              raw_edge_np[i, :2])[0], -1
      ])
      edge_index = len(raw_edge_indices) - 1
      raw_face_edges[raw_edge_np[i, 2]].append(edge_index)
      i += 1
  edge_indices = np.array(
      raw_edge_indices)  # vert indices on each edge [n_edge, 2]
  edge_sides = np.array(
      raw_edge_sides)  # neib vert indices of each edge [n_edge, 2]
  edge_neib = np.array(
      raw_edge_neib)  # neib face indices of each edge [n_edge, 2]
  face_edges = np.array(raw_face_edges)  # edge indices of each face [n_face, 3]
  assert edge_indices.shape == edge_sides.shape and edge_sides.shape == edge_neib.shape
  assert face_edges.shape == (face_count, 3)
  return edge_indices.flatten(), edge_sides.flatten(), edge_neib.flatten(
  ), face_edges.flatten()
  #return edge_indices, face_edges


def compute_rest_length(vert_pos: np.ndarray, edge_indices: np.ndarray):
  edge_count = edge_indices.shape[0]
  norm = lambda x: np.sqrt(np.sum(x**2))
  rest_length = np.zeros(shape=edge_count, dtype=np.float32)
  for i in range(edge_count):
    p1, p2 = edge_indices[i]
    x1, x2 = vert_pos[[p1, p2]]
    rest_length[i] = norm(x2 - x1)

  return rest_length


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

  return vert_mass, face_mass


def distribute_weight(vert_pos: np.ndarray, node_pos: np.ndarray, h: float):
  n_vert = vert_pos.shape[0]
  n_node = node_pos.shape[0]

  weights = np.zeros(shape=(n_vert, n_node), dtype=np.float32)
  norm = lambda x: np.sqrt(np.sum(x**2))
  for i in range(n_vert):
    pos = vert_pos[i]
    min_dis = 100.0
    min_index = -1
    for j in range(n_node):
      dis = norm(pos - node_pos[j])
      if dis < min_dis:
        min_dis = dis
        min_index = j
      weights[i, j] = kernel2d(dis, h)
    if np.sum(weights[i]) == 0.0:
      weights[i, min_index] = 1.0
  for i in range(n_vert):
    sum_weight = np.sum(weights[i])
    assert sum_weight > 0.0
    weights[i] /= sum_weight
    assert np.abs(np.sum(weights[i]) - 1.0) < 1e-6
  return weights


def compute_spring_rest_states(vert_pos: np.ndarray, edge_indices: np.ndarray,
                               angle_indices: np.ndarray):
  n_edge = edge_indices.shape[0]
  n_angle = angle_indices.shape[0]

  rest_length = np.zeros(shape=n_edge, dtype=np.float32)
  rest_angle_cosine = np.zeros(shape=n_angle, dtype=np.float32)

  norm = lambda x: np.sqrt(np.sum(x**2))

  for i in range(n_edge):
    pos0 = vert_pos[edge_indices[i, 0]]
    pos1 = vert_pos[edge_indices[i, 1]]
    rest_length[i] = norm(pos0 - pos1)

  for i in range(n_angle):
    pos0 = vert_pos[angle_indices[i, 0]]
    pos1 = vert_pos[angle_indices[i, 1]]
    pos2 = vert_pos[angle_indices[i, 2]]
    rest_angle_cosine[i] = np.dot(
        pos1 - pos0, pos2 - pos0) / (norm(pos1 - pos0) * norm(pos2 - pos0))
    # print(i, np.arccos(rest_angle_cosine[i]) * 180.0 / np.pi)

  return rest_length, rest_angle_cosine


def compute_spring_node_mass(weights_bind: np.ndarray, vert_mass: np.ndarray):
  node_mass = np.zeros(shape=weights_bind.shape[1], dtype=np.float32)
  for i in range(weights_bind.shape[0]):
    for j in range(weights_bind.shape[1]):
      node_mass[j] += weights_bind[i, j] * vert_mass[i]
  return node_mass


@ti.data_oriented
class TriangleDistortMeasure:

  def __init__(self, verts_ref: np.ndarray, faces: np.ndarray) -> None:
    self.n = faces.size // 3
    self.pos = ti.Vector.field(2, dtype=ti.f32, shape=verts_ref.shape[0])
    self.pos_ref = ti.Vector.field(2, dtype=ti.f32, shape=verts_ref.shape[0])
    self.pos_ref.from_numpy(verts_ref)
    self.indices = ti.field(dtype=ti.i32, shape=faces.size)
    self.indices.from_numpy(faces.flatten())
    self.C = ti.field(dtype=ti.f32, shape=self.n)

  def update_verts(self, verts: np.ndarray):
    self.pos.from_numpy(verts)
    self.measure_distortion()

  @ti.kernel
  def measure_distortion(self):
    for k in range(self.n):
      a = self.indices[k * 3]
      b = self.indices[k * 3 + 1]
      c = self.indices[k * 3 + 2]
      x_1 = self.pos[a]
      x_2 = self.pos[b]
      x_3 = self.pos[c]
      r_1 = self.pos_ref[a]
      r_2 = self.pos_ref[b]
      r_3 = self.pos_ref[c]
      D = ti.Matrix.cols([x_1 - x_3, x_2 - x_3])
      B = ti.Matrix.cols([r_1 - r_3, r_2 - r_3]).inverse()
      F = D @ B
      C_H = F.determinant() - 1
      C_D = F.norm_sqr() - 2.0
      self.C[k] = C_H * C_H + C_D * C_D