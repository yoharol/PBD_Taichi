from pxr import Usd, UsdGeom, Sdf, Vt
from enum import Enum
import numpy as np


class TetCustomAttr(Enum):
  TetVertexIndicesAttr = 1
  VertexOrderAttr = 2
  VertexMassAttr = 3
  TetMassAttr = 4


TetCustomAttrNames = {
    TetCustomAttr.TetVertexIndicesAttr: 'tetVertexIndicesAttr',
    TetCustomAttr.VertexOrderAttr: 'vertexOrderAttr',
    TetCustomAttr.VertexMassAttr: 'vertexMassAttr',
    TetCustomAttr.TetMassAttr: 'tetMassAttr'
}
TetCustomAttrDefineType = {
    TetCustomAttr.TetVertexIndicesAttr: Sdf.ValueTypeNames.Int4Array,
    TetCustomAttr.VertexOrderAttr: Sdf.ValueTypeNames.IntArray,
    TetCustomAttr.VertexMassAttr: Sdf.ValueTypeNames.FloatArray,
    TetCustomAttr.TetMassAttr: Sdf.ValueTypeNames.FloatArray
}
TetCustomAttrSetType = {
    TetCustomAttr.TetVertexIndicesAttr: Vt.Vec4iArray,
    TetCustomAttr.VertexOrderAttr: Vt.IntArray,
    TetCustomAttr.VertexMassAttr: Vt.FloatArray,
    TetCustomAttr.TetMassAttr: Vt.FloatArray
}


class TetModel:

  def __init__(self, mesh_filepath, mesh_primpath) -> None:
    self.stage = Usd.Stage.Open(mesh_filepath)
    self.mesh_prim = self.stage.GetPrimAtPath(mesh_primpath)
    self.mesh = UsdGeom.Mesh(self.mesh_prim)

  def GetCustomAttr(self, attr: TetCustomAttr):
    return np.array(self.mesh_prim.GetAttribute(TetCustomAttrNames[attr]).Get())

  def SetCustomAttr(self, attr: TetCustomAttr, data: np.ndarray):
    self.mesh_prim.GetAttribute(TetCustomAttrNames[attr]).Set(
        TetCustomAttrSetType[attr].FromNumpy(data))

  def Save(self):
    self.stage.Save()
