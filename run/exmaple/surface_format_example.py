"""
In this file we show the format of surface data
"""

import numpy as np
from pxr import Usd, UsdGeom, Sdf, Vt
from geometry import mesh_preprocess
import os

vert_count = 4
face_count = 4
vert_pos = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32)
vert_order = np.array([3] * 4, dtype=int)

face_indices = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int)
face_vert_counts = np.array([3] * 4, dtype=int)

stage = Usd.Stage.CreateNew(
    os.path.join(os.getcwd(), 'assets', 'mesh_format_example.usda'))
UsdGeom.Xform.Define(stage, '/root')
mesh = UsdGeom.Mesh.Define(stage, '/root/mesh')
mesh_prim = stage.GetPrimAtPath('/root/mesh')
stage.SetDefaultPrim(mesh_prim)

mesh.GetPointsAttr().Set(vert_pos)
mesh.GetFaceVertexIndicesAttr().Set(face_indices)
mesh.GetFaceVertexCountsAttr().Set(face_vert_counts)
mesh.GetSubdivisionSchemeAttr().Set('none')
# mesh.GetDoubleSidedAttr().Set(True)

# generate edge information
edge_indices, edge_sides, edge_neib, face_edges = mesh_preprocess.edge_extractor(
    face_indices)
edge_rest_length, edge_rest_angle = mesh_preprocess.compute_rest_status(
    vert_pos, edge_indices, edge_sides)

vert_order, vert_mass, face_mass = mesh_preprocess.compute_vert_mass(
    vert_pos, face_indices)

# work with custom usd types and numpy array:
# https://docs.omniverse.nvidia.com/prod_usd/prod_usd/python-snippets/data-types/convert-vtarray-numpy.html
edgeIndicesAttr = mesh_prim.CreateAttribute('edgeVertexIndices',
                                            Sdf.ValueTypeNames.Int2Array, True,
                                            Sdf.VariabilityUniform)
edgeIndicesAttr.Set(Vt.Vec2iArray.FromNumpy(edge_indices))
edgeSidesAttr = mesh_prim.CreateAttribute('edgeSideVertexIndices',
                                          Sdf.ValueTypeNames.Int2Array, True,
                                          Sdf.VariabilityUniform)
edgeSidesAttr.Set(Vt.Vec2iArray.FromNumpy(edge_sides))
edgeNeibAttr = mesh_prim.CreateAttribute('edgeNeibFaceIndices',
                                         Sdf.ValueTypeNames.Int2Array, True,
                                         Sdf.VariabilityUniform)
edgeNeibAttr.Set(Vt.Vec2iArray.FromNumpy(edge_neib))

faceEdgesAttr = mesh_prim.CreateAttribute('faceEdgeIndices',
                                          Sdf.ValueTypeNames.Int3Array, True,
                                          Sdf.VariabilityUniform)
faceEdgesAttr.Set(Vt.Vec3iArray.FromNumpy(face_edges))
edgeRestLengthAttr = mesh_prim.CreateAttribute('edgeRestLength',
                                               Sdf.ValueTypeNames.FloatArray,
                                               True, Sdf.VariabilityUniform)
edgeRestLengthAttr.Set(Vt.FloatArray.FromNumpy(edge_rest_length))
edgeRestAngleAttr = mesh_prim.CreateAttribute('edgeRestAngle',
                                              Sdf.ValueTypeNames.FloatArray,
                                              True, Sdf.VariabilityUniform)
edgeRestAngleAttr.Set(Vt.FloatArray.FromNumpy(edge_rest_angle))
vertMassAttr = mesh_prim.CreateAttribute('vertexMass',
                                         Sdf.ValueTypeNames.FloatArray, True,
                                         Sdf.VariabilityUniform)
vertMassAttr.Set(Vt.FloatArray.FromNumpy(vert_mass))
vertOrderAttr = mesh_prim.CreateAttribute('vertexOrder',
                                          Sdf.ValueTypeNames.IntArray, True,
                                          Sdf.VariabilityUniform)
vertOrderAttr.Set(Vt.IntArray.FromNumpy(vert_order))
faceMassAttr = mesh_prim.CreateAttribute('faceMass',
                                         Sdf.ValueTypeNames.FloatArray, True,
                                         Sdf.VariabilityUniform)
faceMassAttr.Set(Vt.FloatArray.FromNumpy(face_mass))

mesh_preprocess.mesh_verify(edge_indices, edge_sides, edge_neib, face_indices,
                            face_edges)

stage.Save()

import format.mesh

print(
    format.mesh.GetCustomAttr(mesh_prim,
                              format.mesh.MeshCustomAttr.EdgeVertexIndices))
print(
    format.mesh.GetCustomAttr(mesh_prim,
                              format.mesh.MeshCustomAttr.EdgeNeibFaceIndices))
print(
    format.mesh.GetCustomAttr(mesh_prim,
                              format.mesh.MeshCustomAttr.EdgeSideIndices))
print(
    format.mesh.GetCustomAttr(mesh_prim,
                              format.mesh.MeshCustomAttr.EdgeRestLength))
print(
    format.mesh.GetCustomAttr(mesh_prim,
                              format.mesh.MeshCustomAttr.EdgeRestAngle))
print(
    format.mesh.GetCustomAttr(mesh_prim,
                              format.mesh.MeshCustomAttr.FaceEdgeIndices))
print(format.mesh.GetCustomAttr(mesh_prim, format.mesh.MeshCustomAttr.FaceMass))
print(
    format.mesh.GetCustomAttr(mesh_prim, format.mesh.MeshCustomAttr.VertexMass))
print(
    format.mesh.GetCustomAttr(mesh_prim,
                              format.mesh.MeshCustomAttr.VertexOrder))
