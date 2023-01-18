import taichi as ti
from pxr import Usd, UsdGeom
from geom import gmesh
import numpy as np
import time


class TaichiMeshRenderer:

  def __init__(self,
               title: str,
               res,
               fps,
               mesh: gmesh.TrianMesh,
               cameraPos,
               cameraLookat,
               wireframe=False,
               vertColor=False) -> None:
    self.window = ti.ui.Window(title, res)
    self.canvas = self.window.get_canvas()
    self.scene = ti.ui.Scene()
    self.camera = ti.ui.Camera()
    self.camera.position(cameraPos[0], cameraPos[1], cameraPos[2])
    self.camera.lookat(cameraLookat[0], cameraLookat[1], cameraLookat[2])
    self.mesh = mesh

    self.frame = 0
    self.time = 0.0
    self.prev_time = time.time()
    self.fps = fps
    self.frame_dt = 1.0 / fps
    self.wireframe = wireframe
    self.vertColor = vertColor

  def render(self):
    self.camera.track_user_inputs(self.window,
                                  movement_speed=0.03,
                                  hold_key=ti.ui.RMB)
    self.scene.set_camera(self.camera)
    self.scene.ambient_light((0.8, 0.8, 0.8))
    self.scene.mesh(self.mesh.v_p,
                    self.mesh.f_i,
                    color=(0.8, 0.2, 0.1),
                    show_wireframe=self.wireframe)
    self.canvas.scene(self.scene)
    self.window.show()

    spent_time = time.time() - self.prev_time
    if spent_time < self.frame_dt:
      time.sleep(self.frame_dt - spent_time)
    self.prev_time = time.time()

    self.frame += 1
    self.time = self.frame / self.fps


class USDMeshRenderer:

  def __init__(self, filepath, totalframes, fps, mesh: gmesh.TrianMesh) -> None:
    self.stage = Usd.Stage.CreateNew(filepath)
    UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
    self.stage.SetStartTimeCode(1)
    self.stage.SetEndTimeCode(totalframes)
    self.stage.SetTimeCodesPerSecond(fps)

    self.totalframes = totalframes
    self.fps = fps
    self.mesh = mesh
    self.frame = 1
    self.time = 0.0

    self.rootXform = UsdGeom.Xform.Define(self.stage, '/root')
    self.meshGeom = UsdGeom.Mesh.Define(self.stage, '/root/mesh')

    self.meshGeom.GetPointsAttr().Set(self.mesh.v_p_ref.to_numpy())
    self.meshGeom.GetFaceVertexIndicesAttr().Set(self.mesh.f_i.to_numpy())
    self.meshGeom.GetFaceVertexCountsAttr().Set(self.mesh.v_p.shape[0] * [3])
    self.meshGeom.GetSubdivisionSchemeAttr().Set('none')

  def render(self):
    if self.frame > self.totalframes:
      return
    self.meshGeom.GetPointsAttr().Set(value=self.mesh.v_p.to_numpy(),
                                      time=self.frame)
    self.frame += 1
    self.time = self.frame / self.fps

  def add_ground(self, size):
    groundGeom = UsdGeom.Mesh.Define(self.stage, '/root/ground')
    groundPoints = np.array([[-size[0] / 2.0, 0.0, -size[1] / 2.0],
                             [-size[0] / 2.0, 0.0, size[1] / 2.0],
                             [size[0] / 2.0, 0.0, size[1] / 2.0],
                             [size[0] / 2.0, 0.0, -size[1] / 2.0]])
    groundGeom.GetPointsAttr().Set(groundPoints)
    groundGeom.GetFaceVertexIndicesAttr().Set(np.array([0, 1, 2, 0, 2, 3]))
    groundGeom.GetFaceVertexCountsAttr().Set(np.array([3, 3]))

  def save(self):
    self.stage.Save()
