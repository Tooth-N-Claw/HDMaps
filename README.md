# HDMaps: Horizontal Diffusion Maps implemented in Python
[![Test package](https://github.com/MorphMath/HDMaps/actions/workflows/test.yml/badge.svg)](https://github.com/MorphMath/HDMaps/actions/workflows/test.yml)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)


**A Python implementation of Horizontal Diffusion Maps (HDM), a manifold learning framework for analyzing collections of data.**

![Horizontal diffusion demo](docs/media/hdm_demo.gif)
## What is HDMaps?

Horizontal Diffusion Maps (HDM) extend diffusion maps to collections of data related objects,
each carrying its own internal structure — think shapes, images, or point clouds that vary across a collection.
Instead of treating each object as a single point, HDM models the whole collection as a *fibre bundle*:
a base manifold of data objects, each attached to a fibre of internal points. A random walk on the base is *lifted* to a walk across the fibres, using known or estimated correspondences between neighbouring objects. This lets HDM both organize the data objects and consistently register the internal structure across them — recovering a shared coordinate system that treats each object as more than an isolated similarity score.

![Horizontal diffusion demo](docs/media/hdm_demo.gif)

*A random walk on the base manifold (top), then lifted through the fibres (bottom) — hopping between data objects also moves you to the corresponding point on each object's internal structure.*

## Installation
To install the latest development version of `HDMaps` run:
```bash
pip install git+https://github.com/MorphMath/HDMaps
```

## Usage and Documentation
The package provides an accessible implementation of Horizontal Diffusion Maps, as introduced in the paper: [The diffusion geometry of fibre bundles: Horizontal diffusion maps](https://www.sciencedirect.com/science/article/pii/S1063520318302215).

Detailed usage instructions are found in [docs.md](docs.md).

## License

This software is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
