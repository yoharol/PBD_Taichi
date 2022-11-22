from pxr import Usd, UsdGeom, Tf, Sdf
import numpy as np
import os
import format.mesh as fmesh
import geometry.mesh_preprocess as mesh_preprocess


def convert_raw_to_standard(data_path: str, data_prim_path: str,
                            output_path: str, output_prim_path: str):
  stage = Usd.Stage.CreateNew(output_path)
  meshGeom = UsdGeom.Mesh.Define(stage, output_prim_path)
  meshPrim = meshGeom.GetPrim()
  relDataPath = os.path.relpath(data_path, start=os.path.dirname(output_path))
  meshPrim.GetReferences().AddReference(assetPath=relDataPath,
                                        primPath=data_prim_path)
  vertexPos = np.array(meshGeom.GetPointsAttr().Get(), dtype=np.float32)
  faceVertexIndices = np.array(meshGeom.GetFaceVertexIndicesAttr().Get(),
                               dtype=int)
  build_standard_mesh(vertexPos, faceVertexIndices, meshPrim)
  stage.Save()


def build_standard_mesh(vertexPos: np.ndarray, faceVertexIndices: np.ndarray,
                        prim: Usd.Prim):
  if len(faceVertexIndices.shape) == 1:
    faceVertexIndices = faceVertexIndices.reshape(
        (faceVertexIndices.shape[0] // 3, 3))
  edgeVerts, edgeSides, edgeNeibs, faceEdges = mesh_preprocess.edge_extractor(
      faceVertexIndices)
  edgeRestLength, edgeRestAngle = mesh_preprocess.compute_rest_status(
      vertexPos, edgeVerts, edgeSides)
  vertOrder, vertMass, faceMass = mesh_preprocess.compute_vert_mass(
      vertexPos, faceVertexIndices)
  fmesh.SetCustomAttr(prim, fmesh.MeshCustomAttr.VertexMass, vertMass)
  fmesh.SetCustomAttr(prim, fmesh.MeshCustomAttr.VertexOrder, vertOrder)
  fmesh.SetCustomAttr(prim, fmesh.MeshCustomAttr.EdgeVertexIndices, edgeVerts)
  fmesh.SetCustomAttr(prim, fmesh.MeshCustomAttr.EdgeSideIndices, edgeSides)
  fmesh.SetCustomAttr(prim, fmesh.MeshCustomAttr.EdgeNeibFaceIndices, edgeNeibs)
  fmesh.SetCustomAttr(prim, fmesh.MeshCustomAttr.EdgeRestLength, edgeRestLength)
  fmesh.SetCustomAttr(prim, fmesh.MeshCustomAttr.EdgeRestAngle, edgeRestAngle)
  fmesh.SetCustomAttr(prim, fmesh.MeshCustomAttr.FaceEdgeIndices, faceEdges)
  fmesh.SetCustomAttr(prim, fmesh.MeshCustomAttr.FaceMass, faceMass)
  UsdGeom.Mesh(prim).CreateDisplayColorPrimvar("vertex").Set(vertexPos**2)


def convert(from_path, from_prim, to_path, to_prim):
  convert_raw_to_standard(from_path, from_prim, to_path, to_prim)
