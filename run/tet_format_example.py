import numpy as np
from geometry_preprocess import tet_preprocess
from pxr import Usd, UsdGeom, Sdf, Vt
import os

vert_count = 4
face_count = 4
vert_pos = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32)
tet_indices = np.array([[0, 1, 2, 3]], dtype=int)

face_indices = tet_preprocess.extract_surface_faces(tet_indices)
vert_order, vert_mass, tet_mass = tet_preprocess.compute_vertex_mass(
    vert_pos, tet_indices)

stage = Usd.Stage.CreateNew(
    os.path.join(os.getcwd(), 'assets', 'tet_example.usda'))
UsdGeom.Xform.Define(stage, '/root')
mesh = UsdGeom.Mesh.Define(stage, '/root/tet')
mesh_prim = stage.GetPrimAtPath('/root/tet')

mesh.GetPointsAttr().Set(vert_pos)
mesh.GetFaceVertexIndicesAttr().Set(face_indices)

tetVertexIndicesAttr = mesh_prim.CreateAttribute('tetVertexIndicesAttr',
                                                 Sdf.ValueTypeNames.Int4Array,
                                                 True, Sdf.VariabilityUniform)
tetVertexIndicesAttr.Set(Vt.Vec4iArray.FromNumpy(tet_indices))
vertOrderAttr = mesh_prim.CreateAttribute('vertOrderAttr',
                                          Sdf.ValueTypeNames.IntArray, True,
                                          Sdf.VariabilityUniform)
vertOrderAttr.Set(Vt.IntArray.FromNumpy(vert_order))
vertMassAttr = mesh_prim.CreateAttribute('vertMassAttr',
                                         Sdf.ValueTypeNames.FloatArray, True,
                                         Sdf.VariabilityUniform)
vertMassAttr.Set(Vt.FloatArray.FromNumpy(vert_mass))
tetMassAttr = mesh_prim.CreateAttribute('tetMassAttr',
                                        Sdf.ValueTypeNames.FloatArray, True,
                                        Sdf.VariabilityUniform)
tetMassAttr.Set(Vt.FloatArray.FromNumpy(tet_mass))

stage.Save()