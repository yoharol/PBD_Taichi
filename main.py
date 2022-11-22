import hydra
from omegaconf import DictConfig
import os
from run.mesh_convert import convert
from run.viewer import usd_mesh_viewer
from run.sim import cloth_xpbd
import taichi as ti

arch = {'cpu': ti.x64, 'gpu': ti.cuda}


@hydra.main(version_base=None, config_path='', config_name='config')
def main(cfg: DictConfig):
  path_norm = lambda x: os.path.normpath(os.path.join(os.getcwd(), x))
  if cfg.mode == 'mesh_convert':
    convert(path_norm(cfg.mesh_convert.from_path), cfg.mesh_convert.from_prim,
            path_norm(cfg.mesh_convert.to_path), cfg.mesh_convert.to_prim)
  elif cfg.mode == 'tet_convert':
    convert(path_norm(cfg.tet_convert.from_path), cfg.tet_convert.from_prim,
            path_norm(cfg.tet_convert.to_path), cfg.tet_convert.to_prim)
  elif cfg.mode == 'cloth_xpbd':
    ti.init(arch=arch[cfg.ti.arch])
    cloth_xpbd.sim(path_norm(cfg.cloth_xpbd.ref_path), cfg.cloth_xpbd.ref_prim,
                   path_norm(cfg.cloth_xpbd.output_path),
                   cfg.cloth_xpbd.output_prim,
                   path_norm(cfg.cloth_xpbd.export_path))
  elif cfg.mode == 'mesh_viewer':
    ti.init(arch=ti.vulkan)
    usd_mesh_viewer(cfg.mesh_viewer.ref_path, cfg.mesh_viewer.ref_prim)


if __name__ == '__main__':
  main()
