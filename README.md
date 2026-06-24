# Diffuse Optics by GPU Parallelisation (DOGPUP)
Toolkit for fast parallelised time-domain diffuse optical tomography

- Version 1.2.0
- Author: Ifechi Ejidike

## Features

- Forward
    - Parallel solutions for multi-frequency FD diffusion approximation (DA)
    - TD and time-gated (TG) solutions from Fourier coefficient solutions of DA
- Reconstruction
    - Calculation of multi-frequency FD Jacobians/sensitivities
    - Calculation of TD and TG Jacobians
    - Weighted Levenburg-Marquardt reconstruction of absorption and scattering using TG/TD datatypes
- Data and mesh display

## Tutorials

Some MATLAB live script tutorials can be found in `tutorials` that provide an introuction on how to use the toolbox

## Requirements

### Hardware

NVIDIA CUDA-enabled GPU required. Tested with compute capability from `sm_61` to `sm_89` (GTX10xx to RTX40xx).

Source code for mex routines can be found in `DOGPUP/cuda_lib/mex_sources`, these can be recompiled on your machine if any compatability issues arise using `mexcuda <cuda_source.cu>`. See [MATLAB documentation](https://uk.mathworks.com/help/parallel-computing/run-mex-functions-containing-cuda-code.html) for guidance on this

### Software

**OS:** Windows, Linux

**MATLAB Release:** tested from `2023b` to `2025a`

**Packages:**
- Image Processing Toolbox for iso2mesh
- Parallel Computing Toolbox (**only for compliation** of MEX-files)

## How to Install

1. Clone main repo
2. Add `DOGPUP` to your path

## Changelog
- v1.2.0
    - More efficient solving, only dependent on cuda/mex binaries not MATLAB gpuArrays
    - Grid to mesh interpolation is more efficient
    - dMesh objects can now be written out and read in as JSON files
    - Functions have been renamed
        - `dMesh.flu_solve` -> `dMesh.get_fluence`
        - `dMesh.adj_flu_solve` -> `dMesh.get_adjoint`
        - `dMesh.meas_flu` -> `dMesh.get_detections`
        - `dMesh.plotfun_vol` -> `dMesh.plotfun_iso`
        - `dMesh.target_spots` -> `reconstruction\weighted_spots.m`
        - `reconstruction\wgthd_LM_recon.m` -> `reconstruction\dogpup_recon.m`
    - Plotting slices is now more user friendly, both grid data and mesh data work with `dMesh.plotfun_slice` now.
    - Data can be plotted on sliced mesh in 3D with `dMesh.plotfun_3d`

- v1.1.1
    - `dMesh.mesh2grid()` interpolation is now more efficient. Ignores gridpoints outside mesh which avoids bloat in variables defined on grid
    - Following the above change **meshes saved from older versions are no longer compatible**. Using `dMesh.mesh2grid()` to update their interpolation matrices fixes this
    - Option to define noise floor added in `id_thresh.m`. Important for noisy data 

- v1.1.0
    - IRF is now defined per channel rather than globally 
    - Added reduced scattering Jacobian generation
    - Added weighted scattering coefficient reconstruction


## Known issues and features to be added (in no particular order...)

- Examples with noisy data
- Source-detector placement optimisation routines

## Acknowledgement

The meshing utility is provided by iso2mesh v1.9.8 (Pot Stickers)* which is included, in full, in this repository. 
I would encourage users to check out the iso2mesh github page [here](https://github.com/fangq/iso2mesh).

All code in this repository ***EXCEPT*** for the code in `DOGPUP\meshing\iso2mesh` falls under the licensing desribed in the license file.

The iso2mesh toolbox interacts with external meshing tools the licensing of some of these tools mean that is is unsuitable for commerical use. Please see the iso2mesh repository for more information or `DOGPUP/meshing/iso2mesh/REAMDME.md`

*Anh Phong Tran, Shijie Yan and Qianqian Fang, (2020) "[Improving model-based fNIRS analysis using mesh-based anatomical and light-transport models](https://doi.org/10.1117/1.NPh.7.1.015008)," Neurophotonics, 7(1), 015008

*Qianqian Fang and David Boas, "[Tetrahedral mesh generation from volumetric binary and gray-scale images](https://iso2mesh.sourceforge.net/upload/ISBI2009_abstract_final_web.pdf)," Proceedings of IEEE International Symposium on Biomedical Imaging 2009, pp. 1142-1145, 2009
