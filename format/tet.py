from pxr import Sdf, Vt
from enum import Enum
import numpy as np


class TetCustomAttr(Enum):
  TetVertexIndicesAttr = 1
  VertexOrderAttr = 2
  VertexMassAttr = 3
  TetMassAttr = 4
  FaceTetIndicesAttr = 5


TetCustomAttrNames = {
    TetCustomAttr.TetVertexIndicesAttr: 'tetVertexIndicesAttr',
    TetCustomAttr.VertexOrderAttr: 'vertexOrderAttr',
    TetCustomAttr.VertexMassAttr: 'vertexMassAttr',
    TetCustomAttr.TetMassAttr: 'tetMassAttr',
    TetCustomAttr.FaceTetIndicesAttr: 'faceTetIndicesAttr'
}
TetCustomAttrDefineType = {
    TetCustomAttr.TetVertexIndicesAttr: Sdf.ValueTypeNames.Int4Array,
    TetCustomAttr.VertexOrderAttr: Sdf.ValueTypeNames.IntArray,
    TetCustomAttr.VertexMassAttr: Sdf.ValueTypeNames.FloatArray,
    TetCustomAttr.TetMassAttr: Sdf.ValueTypeNames.FloatArray,
    TetCustomAttr.FaceTetIndicesAttr: Sdf.ValueTypeNames.IntArray
}
TetCustomAttrSetType = {
    TetCustomAttr.TetVertexIndicesAttr: Vt.Vec4iArray,
    TetCustomAttr.VertexOrderAttr: Vt.IntArray,
    TetCustomAttr.VertexMassAttr: Vt.FloatArray,
    TetCustomAttr.TetMassAttr: Vt.FloatArray,
    TetCustomAttr.FaceTetIndicesAttr: Vt.IntArray
}


def GetCustomAttr(prim, attr: TetCustomAttr):
  return np.array(prim.GetAttribute(TetCustomAttrNames[attr]).Get())


def SetCustomAttr(prim, attrType: TetCustomAttr, data: np.ndarray):
  attr = prim.GetAttribute(TetCustomAttrNames[attrType])
  if not attr.IsValid():
    attr = prim.CreateAttribute(TetCustomAttrNames[attrType],
                                TetCustomAttrDefineType[attrType], True,
                                Sdf.VariabilityUniform)
  attr.Set(TetCustomAttrSetType[attrType].FromNumpy(data))