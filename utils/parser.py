import numpy as np
from pxr import Usd, UsdGeom
import meshio


def usd_parser(filename: str, primpath: str):
  stage = Usd.Stage.Open(filename)
  mesh = UsdGeom.Mesh(stage.GetPrimAtPath(primpath))
  points = mesh.GetPointsAttr().Get()
  face_indices = mesh.GetFaceVertexIndicesAttr().Get()
  primVarApi = UsdGeom.PrimvarsAPI(mesh.GetPrim())
  uvAttr = primVarApi.GetPrimvar('st')
  uvs = uvAttr.Get()
  return np.array(points, dtype=np.float32), np.array(face_indices,
                                                      dtype=np.int32), np.array(
                                                          uvs, dtype=np.float32)


def obj_parser(filepath):
  mesh = meshio.read(filepath)
  v, f = mesh.points, mesh.cells_dict['triangle']
  return v, f.flatten()


def tgf_loader(filename: str):
  points = []

  with open(filename, 'r') as f:
    lines = f.readlines()
    state = 0
    for line in lines:
      line = line.replace('\n', '')
      if line[0] == '#':
        state += 1
      elif state == 0:
        point_data = str.split(line, sep=' ')
        p = [float(point_data[1]), float(point_data[2])]
        points.append(p)
  return np.array(points, dtype=np.float32)