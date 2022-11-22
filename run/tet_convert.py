import numpy as np
from pxr import Usd, UsdGeom
import sys, os
import format.tet as ftet
import geometry.tet_preprocess as tet_preprocess


def convert_raw_to_standard(data_path: str, data_prim_path: str,
                            output_path: str, output_prim_path: str):
  data = Usd.Stage.Open(data_path)
  meshGeom = UsdGeom.Mesh(data.GetPrimAtPath(data_prim_path))
  vertexPos = np.array(meshGeom.GetPointsAttr().Get(), dtype=np.float32)
  tetVertexIndices = np.array(meshGeom.GetFaceVertexIndicesAttr().Get(),
                              dtype=int)

  stage = Usd.Stage.CreateNew(output_path)
  output_geom = UsdGeom.Mesh.Define(stage, output_prim_path)
  output_prim = output_geom.GetPrim()
  build_standard_tet(tetVertexIndices, vertexPos, output_prim)
  stage.Save()


def build_standard_tet(tet_indices: np.ndarray, vert_pos: np.ndarray,
                       prim: Usd.Prim):
  if len(tet_indices.shape) == 1:
    tet_indices = tet_indices.reshape((tet_indices.shape[0] // 4, 4))
  surfaceTetIndices, faceVertexIndices = tet_preprocess.extract_surface_faces(
      vert_pos, tet_indices)
  vertexOrder, vertexMass, tetMass = tet_preprocess.compute_vertex_mass(
      vert_pos, tet_indices)
  ftet.SetCustomAttr(prim, ftet.TetCustomAttr.TetMassAttr, tetMass)
  ftet.SetCustomAttr(prim, ftet.TetCustomAttr.TetVertexIndicesAttr, tet_indices)
  ftet.SetCustomAttr(prim, ftet.TetCustomAttr.VertexMassAttr, vertexMass)
  ftet.SetCustomAttr(prim, ftet.TetCustomAttr.VertexOrderAttr, vertexOrder)
  ftet.SetCustomAttr(prim, ftet.TetCustomAttr.FaceTetIndicesAttr,
                     surfaceTetIndices)
  primGeom = UsdGeom.Mesh(prim)
  primGeom.GetPointsAttr().Set(vert_pos)
  primGeom.GetFaceVertexIndicesAttr().Set(faceVertexIndices.flatten())
  primGeom.GetFaceVertexCountsAttr().Set(
      np.array([3] * faceVertexIndices.shape[0], dtype=int))
  primGeom.GetSubdivisionSchemeAttr().Set('none')


def convert(from_path, from_prim, to_path, to_prim):
  convert_raw_to_standard(from_path, from_prim, to_path, to_prim)
