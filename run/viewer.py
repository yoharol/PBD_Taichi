import taichi as ti
from pxr import Usd, UsdGeom
import numpy as np


def usd_mesh_viewer(path: str, prim_path: str):
  stage = Usd.Stage.Open(path)
  mesh = UsdGeom.Mesh(stage.GetPrimAtPath(prim_path))
  startFrame = stage.GetStartTimeCode()
  endFrame = stage.GetEndTimeCode()
  fps = stage.GetTimeCodesPerSecond()
  window = ti.ui.Window("USD Viewer", (768, 768), vsync=True)
  canvas = window.get_canvas()
  canvas.set_background_color((0.1, 0.2, 0.2))
  scene = ti.ui.Scene()
  camera = ti.ui.Camera()
  frame = 1

  pointsAttr = mesh.GetPointsAttr()
  colorsAttr = mesh.GetDisplayColorPrimvar()
  indicesAttr = mesh.GetFaceVertexIndicesAttr()

  num_faces = np.array(indicesAttr.Get()).shape[0] // 3
  num_points = np.array(pointsAttr.Get()).shape[0]

  indices = ti.field(int, shape=num_faces * 3)
  vertices = ti.Vector.field(3, dtype=float, shape=num_points)
  colors = ti.Vector.field(3, dtype=float, shape=num_points)

  indices.from_numpy(np.array(indicesAttr.Get()))

  while window.running:
    if frame > endFrame:
      frame -= endFrame
    camera.position(0.0, 1.0, 3)
    camera.lookat(0.0, 0.5, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    vertices.from_numpy(np.array(pointsAttr.Get(frame)))
    colors.from_numpy(np.array(colorsAttr.Get(frame)))

    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)

    canvas.scene(scene)
    window.show()

    frame += 1
