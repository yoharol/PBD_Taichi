from pxr import Vt, Sdf
import numpy as np

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
    MeshCustomAttr.EdgeSideIndices: 'edgeSideVertexIndices',
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


def GetCustomAttr(prim, attrType: MeshCustomAttr):
  return np.array(prim.GetAttribute(MeshCustomAttrNames[attrType]).Get())


def SetCustomAttr(prim, attrType: MeshCustomAttr, data: np.ndarray):
  attr = prim.GetAttribute(MeshCustomAttrNames[attrType])
  if not attr.IsValid():
    attr = prim.CreateAttribute(MeshCustomAttrNames[attrType],
                                MeshCustomAttrDefineType[attrType], True,
                                Sdf.VariabilityUniform)
  attr.Set(MeshCustomAttrSetType[attrType].FromNumpy(data))
