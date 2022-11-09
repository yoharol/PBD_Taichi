from pxr import Usd, UsdGeom, Vt, Sdf
import numpy as np
import global_var

from enum import Enum


class MeshCustomAttr(Enum):
  EdgeVertexIndices = 1
  EdgeSideIndices = 2
  EdgeNeibFaceIndices = 3
  FaceEdgeIndices = 4
  EdgeRestLength = 5
  EdgeRestAngle = 6
  VertexMass = 7
  VertexOrder = 8
  FaceMass = 9


MeshCustomAttrNames = {
    MeshCustomAttr.EdgeVertexIndices: 'edgeVertexIndices',
    MeshCustomAttr.EdgeSideIndices: 'edgeSideIndices',
    MeshCustomAttr.EdgeNeibFaceIndices: 'edgeNeibFaceIndices',
    MeshCustomAttr.FaceEdgeIndices: 'faceEdgeIndices',
    MeshCustomAttr.EdgeRestLength: 'edgeRestLength',
    MeshCustomAttr.EdgeRestAngle: 'edgeRestAngle',
    MeshCustomAttr.VertexMass: 'vertexMass',
    MeshCustomAttr.VertexOrder: 'vertexOrder',
    MeshCustomAttr.FaceMass: 'faceMass'
}

MeshCustomAttrDefineType = {
    MeshCustomAttr.EdgeVertexIndices: Sdf.ValueTypeNames.Int2Array,
    MeshCustomAttr.EdgeSideIndices: Sdf.ValueTypeNames.Int2Array,
    MeshCustomAttr.EdgeNeibFaceIndices: Sdf.ValueTypeNames.Int2Array,
    MeshCustomAttr.FaceEdgeIndices: Sdf.ValueTypeNames.Int3Array,
    MeshCustomAttr.EdgeRestLength: Sdf.ValueTypeNames.FloatArray,
    MeshCustomAttr.EdgeRestAngle: Sdf.ValueTypeNames.FloatArray,
    MeshCustomAttr.VertexMass: Sdf.ValueTypeNames.FloatArray,
    MeshCustomAttr.VertexOrder: Sdf.ValueTypeNames.IntArray,
    MeshCustomAttr.FaceMass: Sdf.ValueTypeNames.FloatArray
}

MeshCustomAttrSetType = {
    MeshCustomAttr.EdgeVertexIndices: Vt.Vec2iArray,
    MeshCustomAttr.EdgeSideIndices: Vt.Vec2iArray,
    MeshCustomAttr.EdgeNeibFaceIndices: Vt.Vec2iArray,
    MeshCustomAttr.FaceEdgeIndices: Vt.Vec3iArray,
    MeshCustomAttr.EdgeRestLength: Vt.FloatArray,
    MeshCustomAttr.EdgeRestAngle: Vt.FloatArray,
    MeshCustomAttr.VertexMass: Vt.FloatArray,
    MeshCustomAttr.VertexOrder: Vt.IntArray,
    MeshCustomAttr.FaceMass: Vt.FloatArray
}


class CompleteMesh:

  def __init__(self, mesh_filepath, mesh_primpath) -> None:
    self.stage = Usd.Stage.Open(mesh_filepath)
    self.mesh_prim = self.stage.GetPrimAtPath(mesh_primpath)
    self.mesh = UsdGeom.Mesh(self.mesh_prim)

  def GetCustomAttr(self, attr: MeshCustomAttr):
    return np.array(
        self.mesh_prim.GetAttribute(MeshCustomAttrNames[attr]).Get())

  def SetCustomAttr(self, attr: MeshCustomAttr, data: np.ndarray):
    self.mesh_prim.GetAttribute(MeshCustomAttrNames[attr]).Set(
        MeshCustomAttrSetType[attr].FromNumpy(data))

  def Save(self):
    self.stage.Save()
