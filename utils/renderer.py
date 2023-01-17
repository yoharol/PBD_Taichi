import taichi as ti
from pxr import Usd, UsdGeom
from geom import gmesh
import time


class TaichiRenderer:

  def __init__(self,
               title: str,
               res,
               fps,
               mesh: gmesh.TrianMesh,
               cameraPos,
               cameraLookat,
               wireframe=False,
               vertColor=False) -> None:
    self.window = ti.ui.Window(title, res, vsync=True)
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
