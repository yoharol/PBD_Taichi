# PBD Taichi

Library for PBD and XPBD, aiming at including most type of constraints on mesh and tetrahedron.

- [x] Cloth simulation
- [x] Deformable simulaiton
- [X] Complementary Dynamics
- [X] Hash grid collision
- [ ] Rigid body simulation
- [ ] Rod

## Environments

```shell
pip install taichi 
pip install usd-core
pip install meshio
pip install pillow
pip install glfw
pip install PyOpenGL PyOpenGL_accelerate
``` 

## See results

Run scripts in `test` folder

For example:
```shell
python -m test.cloth_sim
```

1. **Cloth**

`python -m test.cloth_sim`

Constraints:
- Bending
- Length

<img src=".//preview//cloth.gif" alt="drawing" width="400"/>

2. **Balloon**

`python -m test.balloon`

Constraints:
- Bending
- Length
- Volume preserving

<img src=".//preview//balloon.gif" alt="drawing" width="400"/>

3. **Particles Collision**

`python -m test.dem3d`

Solve collision of large group of particles with hash grid

<img src=".//preview//DEM.gif" alt="drawing" width="400"/>

4. **Neo-Hookean Deformable Objects**

`python -m test.fish_deform`

Based on this [paper](https://matthias-research.github.io/pages/publications/neohookean.pdf) and the [live demo](https://matthias-research.github.io/pages/tenMinutePhysics/10-softBodies.html).

<img src=".//preview//fish_deform.gif" alt="drawing" width="400"/>

For comparison, there is a mass spring version.

`python -m test.fish_spring`

<img src=".//preview//fish_spring.gif" alt="drawing" width="400"/>

5. Complementary Dynamics

`python -m test.fish_comp`

Based on [complementary dynamics](https://www.dgp.toronto.edu/projects/complementary-dynamics/) and its [modified PBD version](https://yoharol.github.io/pages/control_pbd/).


<img src=".//preview//fish_comp.gif" alt="drawing" width="400"/>