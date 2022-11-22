import taichi as ti
import numpy as np
from pxr import Usd, UsdGeom, Tf
import format.mesh as fmesh
import os


@ti.data_oriented
class ClothSim:

  def __init__(self,
               reference_path: str,
               reference_prim_path: str,
               output_path: str,
               output_prim: str,
               export_path: str,
               cons_vert_indices: np.ndarray,
               sim_time=4.0,
               fps=60,
               substeps=50,
               render_color=False) -> None:
    self.renderStage = Usd.Stage.CreateNew(output_path)
    self.outputPath = output_path
    self.meshGeom = UsdGeom.Mesh.Define(self.renderStage, output_prim)
    self.meshPrim = self.meshGeom.GetPrim()
    relDataPath = os.path.relpath(reference_path,
                                  start=os.path.dirname(output_path))
    self.meshPrim.GetReferences().AddReference(assetPath=relDataPath,
                                               primPath=reference_prim_path)
    self.export_path = export_path

    self.renderColor = render_color
    self.substeps = substeps
    self.frames = int(sim_time * fps)
    self.dt = 1.0 / fps
    self.fps = fps
    self.renderStage.SetStartTimeCode(1)
    self.renderStage.SetEndTimeCode(self.frames)
    self.renderStage.SetTimeCodesPerSecond(fps)

    self.stretchAlpha = 0.0
    self.bendAlpha = 0.0
    self.wind = ti.Vector([0.0, 0.0, 6.0])
    self.g = ti.Vector([0.0, -3.0, 0.0])

    self.vertexPosAttr = self.meshGeom.GetPointsAttr()
    self.colorPrimVar = self.meshGeom.GetDisplayColorPrimvar()

    vertPos = np.array(self.vertexPosAttr.Get(), dtype=np.float32)
    vertColor = np.array(self.colorPrimVar.Get(), dtype=np.float32)
    edgeNeib = fmesh.GetCustomAttr(self.meshPrim,
                                   fmesh.MeshCustomAttr.EdgeNeibFaceIndices)
    edgeSides = fmesh.GetCustomAttr(self.meshPrim,
                                    fmesh.MeshCustomAttr.EdgeSideIndices)
    restAngle = fmesh.GetCustomAttr(self.meshPrim,
                                    fmesh.MeshCustomAttr.EdgeRestAngle)
    restLength = fmesh.GetCustomAttr(self.meshPrim,
                                     fmesh.MeshCustomAttr.EdgeRestLength)
    edgeVertexIndices = fmesh.GetCustomAttr(
        self.meshPrim, fmesh.MeshCustomAttr.EdgeVertexIndices)
    vertMassInv = 1.0 / fmesh.GetCustomAttr(self.meshPrim,
                                            fmesh.MeshCustomAttr.VertexMass)

    for i in range(cons_vert_indices.shape[0]):
      vertMassInv[cons_vert_indices[i]] = 0.0

    self.consN = cons_vert_indices.shape[0]
    self.vertN = vertPos.shape[0]
    self.edgeN = edgeNeib.shape[0]
    self.vertX = ti.Vector.field(3, dtype=ti.f32, shape=self.vertN)
    self.consVertIndices = ti.field(dtype=ti.i32, shape=self.consN)
    self.consVertX = ti.Vector.field(3, dtype=ti.f32, shape=self.consN)
    self.vertColor = ti.Vector.field(3, dtype=ti.f32, shape=self.vertN)
    self.vertMassInv = ti.field(dtype=ti.f32, shape=self.vertN)
    self.edgeIndices = ti.field(dtype=ti.i32, shape=(self.edgeN, 2))
    self.edgeSides = ti.field(dtype=ti.i32, shape=(self.edgeN, 2))
    self.edgeRestLength = ti.field(dtype=ti.f32, shape=self.edgeN)
    self.edgeRestAngle = ti.field(dtype=ti.f32, shape=self.edgeN)
    self.edgeLengthLambda = ti.field(dtype=ti.f32, shape=self.edgeN)
    self.edgeAngleLambda = ti.field(dtype=ti.f32, shape=self.edgeN)

    self.cacheX = ti.Vector.field(3, dtype=ti.f32, shape=self.vertN)
    self.vertV = ti.Vector.field(3, dtype=ti.f32, shape=self.vertN)

    self.vertX.from_numpy(vertPos)
    self.consVertIndices.from_numpy(cons_vert_indices)
    self.cacheX.from_numpy(vertPos)
    self.vertV.fill(ti.Vector([0.0, 0.0, 0.0]))
    self.vertColor.from_numpy(vertColor)
    self.vertMassInv.from_numpy(vertMassInv)
    self.edgeIndices.from_numpy(edgeVertexIndices)
    self.edgeSides.from_numpy(edgeSides)
    self.edgeRestLength.from_numpy(restLength)
    #self.edgeRestAngle.from_numpy(restAngle)
    self.edgeRestAngle.fill(0.0)

  def render_frame(self, timecode):
    self.meshGeom.GetPointsAttr().Set(value=self.vertX.to_numpy(),
                                      time=timecode)
    if self.renderColor:
      self.meshGeom.GetDisplayColorPrimvar().Set(
          value=self.vertColor.to_numpy(), time=timecode)

  def save(self):
    self.renderStage.Save()
    flattened_layer = self.renderStage.Flatten()
    print("simulation complete")
    flattened_layer.Export(self.export_path)

  @ti.kernel
  def falseUpdate(self, time: ti.f32):
    for i in self.vertX:
      self.vertX[i][2] = self.cacheX[i][2] + 0.1 * ti.sin(
          2 * ti.math.pi * time / 2.0 + self.vertX[i][1])
      self.vertColor[i][1] = self.vertX[i][1] + time / 2.0
      self.vertColor[i][1] -= ti.floor(self.vertColor[i][1])

  @ti.kernel
  def UpdateConstaint(self):
    for i in range(self.consN):
      self.vertX[self.consVertIndices[i]] = self.consVertX[i]

  def GetPos(self, index_list: np.ndarray):
    return self.vertX.to_numpy()[index_list]

  def SetConstaintPos(self, updated_pos):
    assert updated_pos.shape == (self.consN, 3)
    self.consVertX.from_numpy(updated_pos)
    self.UpdateConstaint()

  @ti.kernel
  def generate_prediction(self):
    self.edgeAngleLambda.fill(0.0)
    self.edgeLengthLambda.fill(0.0)
    for i in self.vertX:
      self.cacheX[i] = self.vertX[i]

      self.vertX[i] = self.vertX[i] + self.vertV[i] * self.dt + (
          self.g + self.wind) * self.dt * self.dt

  @ti.kernel
  def update_vel(self):
    for i in self.vertV:
      tmp = (self.vertX[i] - self.cacheX[i]) / self.dt
      self.vertColor[i] = ti.Vector([(tmp - self.vertV[i]).norm(), 0.0, 0.0])
      self.vertV[i] = tmp

  @ti.kernel
  def solve_length_constraints(self):
    for k in range(self.edgeN):
      i = self.edgeIndices[k, 0]
      j = self.edgeIndices[k, 1]
      x_ij = self.vertX[i] - self.vertX[j]
      C_ij = x_ij.norm() - self.edgeRestLength[k]
      w_i = self.vertMassInv[i]
      w_j = self.vertMassInv[j]
      delta_lambda = -(C_ij + self.stretchAlpha * self.edgeLengthLambda[k] /
                       (self.dt * self.dt)) / (w_i + w_j + self.stretchAlpha /
                                               (self.dt * self.dt))
      self.edgeLengthLambda[k] += delta_lambda
      x_ij = x_ij / x_ij.norm()
      self.vertX[i] += w_i * delta_lambda * x_ij
      self.vertX[j] += -w_j * delta_lambda * x_ij

  @ti.kernel
  def solve_angle_constraints(self):
    for k in range(self.edgeN):
      if self.edgeSides[k, 1] != -1:
        i1 = self.edgeIndices[k, 0]
        i2 = self.edgeIndices[k, 1]
        i3 = self.edgeSides[k, 0]
        i4 = self.edgeSides[k, 1]
        x1 = self.vertX[i1]
        x2 = self.vertX[i2]
        x3 = self.vertX[i3]
        x4 = self.vertX[i4]

        n1 = ((x2 - x1).cross(x3 - x1)).normalized()
        n2 = ((x2 - x1).cross(x4 - x1)).normalized()
        d = n1.dot(n2)
        if d * d < 1.0:
          C = ti.acos(d) - self.edgeRestAngle[k]

          q3 = (x2.cross(n2) + n1.cross(x2) * d) / (x2.cross(x3)).norm()
          q4 = (x2.cross(n1) + n2.cross(x2) * d) / (x2.cross(x4)).norm()
          q2 = -(x3.cross(n2) + n1.cross(x3) * d) / (x2.cross(x3)).norm() - (
              x4.cross(n1) + n2.cross(x4) * d) / (x2.cross(x4)).norm()
          q1 = -q2 - q3 - q4

          coe1 = (self.vertMassInv[i1] * ti.pow(q1.norm(), 2) +
                  self.vertMassInv[i2] * ti.pow(q2.norm(), 2) +
                  self.vertMassInv[i3] * ti.pow(q3.norm(), 2) +
                  self.vertMassInv[i4] * ti.pow(q4.norm(), 2)) / (1 - d * d)
          delta_lambda = -(C + self.bendAlpha * self.edgeAngleLambda[k] /
                           (self.dt * self.dt)) / (coe1 + self.bendAlpha /
                                                   (self.dt * self.dt))
          coe2 = 1.0 / ti.sqrt(1 - d * d)
          #print(coe1, coe2, delta_lambda)
          self.vertX[
              i1] += self.vertMassInv[i1] * coe2 * q1 * delta_lambda * 0.01
          self.vertX[
              i2] += self.vertMassInv[i1] * coe2 * q2 * delta_lambda * 0.01
          self.vertX[
              i3] += self.vertMassInv[i1] * coe2 * q3 * delta_lambda * 0.01
          self.vertX[
              i4] += self.vertMassInv[i1] * coe2 * q4 * delta_lambda * 0.01
          #print(self.vertX[i1], self.vertX[i2], self.vertX[i3], self.vertX[i4])
          self.edgeAngleLambda[k] += delta_lambda
