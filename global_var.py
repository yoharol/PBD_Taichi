import os
from pxr import Vt, Sdf
from enum import Enum

WorkingDir = os.getcwd()

CompleteMeshPath = os.path.join(WorkingDir, 'assets', 'complete_mesh')
SimpleMeshPath = os.path.join(WorkingDir, 'assets', 'simple_mesh')
TetModelPath = os.path.join(WorkingDir, 'assets', 'tet_model')
OutputPath = os.path.join(WorkingDir, 'outputs')
