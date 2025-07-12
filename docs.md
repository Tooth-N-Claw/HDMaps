<a id="HDM/"></a>

# HDM/

<a id="HDM/.distances"></a>

# HDM/.distances

<a id="HDM/.HDM"></a>

# HDM/.HDM

<a id="HDM/.HDM.hdm_embed"></a>

#### hdm\_embed

```python
def hdm_embed(data_samples: list[np.ndarray],
              config: HDMConfig = HDMConfig(),
              base_kernel: Optional[coo_matrix] = None,
              fiber_kernel: Optional[coo_matrix] = None,
              base_distances: Optional[coo_matrix] = None,
              fiber_distances: Optional[coo_matrix] = None) -> np.ndarray
```



<a id="HDM/.tests.test_hdm"></a>

# HDM/.tests.test\_hdm

<a id="HDM/.tests.utils"></a>

# HDM/.tests.utils

<a id="HDM/.block_mul"></a>

# HDM/.block\_mul

<a id="HDM/.visualize"></a>

# HDM/.visualize

<a id="HDM/.validate"></a>

# HDM/.validate

<a id="HDM/.utils"></a>

# HDM/.utils

<a id="HDM/.utils.ensure_sparse"></a>

#### ensure\_sparse

```python
def ensure_sparse(matrix)
```

Converts a sparse scipy matrix to a jax BCOO

<a id="HDM/.utils.compute_block_indices"></a>

#### compute\_block\_indices

```python
def compute_block_indices(data_samples: list) -> jnp.ndarray
```

Compute cumulative start indices for a list of data samples.

<a id="HDM/.kernels"></a>

# HDM/.kernels

<a id="HDM/.kernels.compute_base_kernel"></a>

#### compute\_base\_kernel

```python
def compute_base_kernel(config: HDMConfig, data_samples: list[np.ndarray],
                        base_distances: BCOO, base_kernel: BCOO)
```



<a id="HDM/.kernels.compute_fiber_kernel"></a>

#### compute\_fiber\_kernel

```python
def compute_fiber_kernel(config: HDMConfig, data_samples: list[np.ndarray],
                         fiber_distances: BCOO, fiber_kernel: BCOO)
```



<a id="HDM/.spectral"></a>

# HDM/.spectral

<a id="HDM/.spectral.eigendecomposition"></a>

#### eigendecomposition

```python
def eigendecomposition(matrix: sparse.csr_matrix,
                       num_eigenvectors: int) -> tuple[np.ndarray, np.ndarray]
```

Perform eigendecomposition on a sparse matrix.
