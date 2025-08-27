# DOGPUP
## Diffuse Optics by GPU Parallelisation
Toolkit for fast parallelised time-domain diffuse optical tomography

- Version 1.0.0
- Author: Ifechi Ejidike

## Features

- Forward
    - Parallel solutions for multi-frequency FD diffusion approximation (DA)
    - TD and time-gated (TG) solutions from Fourier coefficient solutions of DA
- Reconstruction
    - Calculation of multi-frequency FD absorpton Jacobian/sensitivity
    - Calculation of TD and TG absorpton Jacobians
    - Weighted Levenburg-Marquardt reconstruction of absorption using TG/TD datatypes
- Data and mesh display

## Tutorials

Some MATLAB live script tutorials can be found in `tutorials` that provide an introuction on how to use the toolbox

## Requirements

### Hardware

NVIDIA CUDA-enabled GPU required. Tested with compute capability from `sm_61` to `sm_89` (GTX10xx to RTX40xx).

Source code for mex routines can be found in `DOGPUP/cuda_lib/mex_sources`, these can be recompiled on your machine if any compatability issues arise.

### Software

**OS**: Windows, Linux

**MATLAB Release**: tested from `2023b` to `2025a`

**Packages**: Image Processing Toolbox and Parallel Computing Toolbox

## How to Install

1. Clone main repo
2. Add `DOGPUP` to your path

## Changelog

N/A

## To be added

- Diffusion Jacobian generation
- Scattering coefficient reconstruction

## Acknowledgement

The meshing utility is provided by iso2mesh v1.9.8 (Pot Stickers)* which is included, in full, in this repository. 
I would encourage users to check out the iso2mesh github page [here](https://github.com/fangq/iso2mesh).

All code in this repository ***EXCEPT*** for the code in `DOGPUP\meshing\iso2mesh` falls under the licensing desribed in the license file.

The iso2mesh toolbox interacts with external meshing tools the licensing of some of these tools mean that is is unsuitable for commerical use. Please see the iso2mesh repository for more information or `DOGPUP/meshing/iso2mesh/REAMDME.md`

*Anh Phong Tran, Shijie Yan and Qianqian Fang, (2020) "[Improving model-based fNIRS analysis using mesh-based anatomical and light-transport models](https://doi.org/10.1117/1.NPh.7.1.015008)," Neurophotonics, 7(1), 015008

*Qianqian Fang and David Boas, "[Tetrahedral mesh generation from volumetric binary and gray-scale images](https://iso2mesh.sourceforge.net/upload/ISBI2009_abstract_final_web.pdf)," Proceedings of IEEE International Symposium on Biomedical Imaging 2009, pp. 1142-1145, 2009



