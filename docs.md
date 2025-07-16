# HDM_Python
[![Test package](https://github.com/frisbro303/HDM_Python/actions/workflows/test.yml/badge.svg)](https://github.com/frisbro303/HDM_Python/actions/workflows/test.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**A Python implementation of Horizontal Diffusion Maps (HDM), a manifold learning framework for data analysis of datasets with base-fiber structure.**


## Installation
To install the latest development version of `HDM_Python` run:
```bash
pip install git+https://github.com/frisbro303/HDM_Python
```

## Usage
To make effective use of this package the documentation, it is recommended to have a basic understanding of Horizontal Diffusions Maps,
as introduced in the paper: [The diffusion geometry of fibre bundles: Horizontal diffusion maps](https://www.sciencedirect.com/science/article/pii/S1063520318302215).

To get started using HDM_Python, add the following import to the top of your Python file:
```python
from HDM import hdm_embed, HDMConfig
```
### The `hdm_embed`function
The primary interface of the package is the `hdm_embed` function, which embeds the data in a Euclidean space preserving the horizontal diffusion distance. The function signature is given below:

```python
def hdm_embed(
    config: HDMConfig = HDMConfig(),
    data_samples: Optional[list[np.ndarray]] = None,
    block_indices: Optional[np.ndarray] = None,
    base_kernel: Optional[coo_matrix] = None,
    fiber_kernel: Optional[coo_matrix] = None,
    base_distances: Optional[coo_matrix] = None,
    fiber_distances: Optional[coo_matrix] = None,
) -> np.ndarray:

    """
    Compute the Horizontal Diffusion Maps (HDM) embedding from input data.

    This function constructs and processes base and fiber kernels from the input data or 
    precomputed distances/kernels, normalizes the resulting joint kernel, and computes 
    a HDM embedding.

    Parameters:
        config (HDMConfig): Configuration object specifying HDM parameters.
        data_samples (list[np.ndarray], optional): List of data arrays (e.g., sampled fibers).
        block_indices (np.ndarray, optional): Block indices specifying data partitioning.
        base_kernel (coo_matrix, optional): Precomputed base kernel (spatial proximity).
        fiber_kernel (coo_matrix, optional): Precomputed fiber kernel (fiber similarity).
        base_distances (coo_matrix, optional): Precomputed base distances.
        fiber_distances (coo_matrix, optional): Precomputed fiber distances.

    Returns:
        np.ndarray: Diffusion coordinates from the joint HDM embedding.
    """

```
The embedding can be computed entirely from given data samples based on custom base and fiber metrics, or the user can provide either precomputed distances or kernels, or a mix of these.


**Important:** Distances and block indices are computed automatically only if `data_samples` are provided. If `data_samples` are omitted, precomputed kernels and block indices must be supplied explicitly.

See the [examples](https://github.com/frisbro303/HDM_Python/tree/main/examples) folder for usage demonstrations.


### Configuration

The `HDMConfig` class provides configuration parameters for controlling kernel computations and embedding:

- `base_epsilon` (float, default=0.04)  
  Bandwidth parameter for the base kernel.

- `fiber_epsilon` (float, default=0.08)  
  Bandwidth parameter for the fiber kernel.

- `num_eigenvectors` (int, default=4)  
  Number of eigenvectors (dimension of embedding) to compute.

- `device` (str or None, default="CPU")  
  Device to run computations on (e.g., `"CPU"` or `"GPU"`).

- `base_metric` (str, default="frobenius")  
  Metric used for base kernel distance computations.

- `fiber_metric` (str, default="euclidean")  
  Metric used for fiber kernel distance computations.

- `base_sparsity` (float, default=0.08)  
  Sparsity parameter for the base kernel (controls graph sparsification).

- `fiber_sparsity` (float, default=0.08)  
  Sparsity parameter for the fiber kernel.

### Example

```python
from hdm.utils import HDMConfig

config = HDMConfig(
    base_epsilon=0.05,
    fiber_epsilon=0.1,
    num_eigenvectors=3,
    device="CPU",
    base_metric="frobenius",
    fiber_metric="euclidean",
    base_sparsity=0.1,
    fiber_sparsity=0.1,
)
```


## License

This software is licensed under the GPL-3.0 License. See the [LICENSE](https://github.com/frisbro303/SignDNE/blob/2347bf47a35affe612ac8d60e64805a3f1891951/LICENSE) file for details. 




