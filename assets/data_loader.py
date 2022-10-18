import numpy as np
from pxr import Usd, UsdGeom


class TetData:

  def __init__(self,
               filepath,
               outputpath,
               frames=120,
               frames_per_second=60) -> None:

    with open(filepath, 'rb') as f:
      self.vertices_pos = np.load(f)  # position of vertices
      self.tet_ids = np.load(f)  # composition of tetrahedrals
      self.edge_ids = np.load(f)  # composition of edges
      self.surface_ids = np.load(f)  # composition of surface faces
      self.vertices_mass = np.load(f)  # volume(mass) of each vertex
      self.rest_length = np.load(f)
      self.max_order = np.load(f)[0]
    print("max order of single vertex", self.max_order)
    self.num_particles = self.vertices_pos.shape[0]
    self.num_tets = self.tet_ids.shape[0]
    self.num_edges = self.edge_ids.shape[0]
    self.num_surfaces = self.surface_ids.shape[0]

    self.stage = Usd.Stage.CreateNew(outputpath)
    self.stage.SetStartTimeCode(1)
    self.stage.SetEndTimeCode(frames)
    self.stage.SetTimeCodesPerSecond(frames_per_second)

    UsdGeom.Xform.Define(self.stage, '/root')
    self.geom = UsdGeom.Mesh.Define(self.stage, '/root/bunny')
    self.geom.GetFaceVertexIndicesAttr().Set(self.surface_ids)
    self.geom.GetFaceVertexCountsAttr().Set(self.num_surfaces * [3])
    self.geom.GetSubdivisionSchemeAttr().Set('none')

  def render(self, vertices_pos, timecode):
    assert vertices_pos.shape == self.vertices_pos.shape
    assert vertices_pos.dtype == self.vertices_pos.dtype
    self.geom.GetPointsAttr().Set(value=vertices_pos, time=timecode)
    extent = UsdGeom.Boundable.ComputeExtentFromPlugins(self.geom, timecode)
    self.geom.GetExtentAttr().Set(value=extent, time=timecode)

  def save(self):
    self.stage.GetRootLayer().Save()
