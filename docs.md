
## Main Function: `hdm_embed`

The primary interface to the package is the `hdm_embed` function which computes the Horizontal Diffusion Map (HDM) embedding of fiber bundle data.

```python
def hdm_embed(
    data_samples: list[np.ndarray],
    config: HDMConfig = HDMConfig(),
    base_kernel: Optional[coo_matrix] = None,
    fiber_kernel: Optional[coo_matrix] = None,
    base_distances: Optional[coo_matrix] = None,
    fiber_distances: Optional[coo_matrix] = None,
) -> jnp.ndarray:
    """
    Compute the HDM embedding.

    Parameters:
    -----------
    data_samples : list of np.ndarray
        List of fiber data arrays, one per base point.
    config : HDMConfig, optional
        Configuration parameters for kernel and embedding.
    base_kernel : coo_matrix, optional
        Precomputed base kernel.
    fiber_kernel : coo_matrix, optional
        Precomputed fiber kernel.
    base_distances : coo_matrix, optional
        Base distances if kernel is not provided.
    fiber_distances : coo_matrix, optional
        Fiber distances if kernel is not provided.

    Returns:
    --------
    jnp.ndarray
        Diffusion coordinates representing the embedding.
    """
```

## Configuration: `HDMConfig`

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
