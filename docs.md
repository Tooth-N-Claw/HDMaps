# HDM_Python
[![Test package](https://github.com/frisbro303/HDM_Python/actions/workflows/test.yml/badge.svg)](https://github.com/frisbro303/HDM_Python/actions/workflows/test.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**A Python implementation of Horizontal Diffusion Maps (HDM), a manifold learning framework for data analysis of datasets with base-fiber structure.**


## What is HDMaps?

Horizontal Diffusion Maps (HDM) extend diffusion maps to collections of related data objects — shapes, images, point clouds — each carrying its own internal structure. HDM models the collection as a *fibre bundle*: a base manifold capturing how objects relate to one another, and a fibre over each point representing that object's structure as a noisy realization of a shared template. A random walk on the base is *lifted* across the fibres via correspondences between neighbouring objects, letting HDM both organize the objects and consistently register their internal structure into a shared coordinate system.

![Horizontal diffusion demo](media/hdm_demo.gif)

*A random walk on the base manifold (top), lifted through the fibres (bottom) — hopping between objects also moves to the corresponding point on each object's structure.*


## Installation
To install the latest development version of `HDM_Python` run:
```bash
pip install git+https://github.com/MorphMath/HDMaps
```

## Usage
To make effective use of this package the documentation, it is recommended to have a basic understanding of Horizontal Diffusions Maps,
as introduced in the paper: [The diffusion geometry of fibre bundles: Horizontal diffusion maps](https://www.sciencedirect.com/science/article/pii/S1063520318302215).

To get started using HDM_Python, add the following import to the top of your Python file:
```python
from HDM import hdm_embed, HDMConfig
```

## License

This software is licensed under the GPL-3.0 License. See the [LICENSE](https://github.com/frisbro303/SignDNE/blob/2347bf47a35affe612ac8d60e64805a3f1891951/LICENSE) file for details. 




