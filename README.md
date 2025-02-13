# HDM_Python

# TODOS:

- order of loading ply files needs to be correct.

# Notes:

<!-- - input files, what formats should they be in? i.e. data_samples and maps -->
<!-- We just assume they are numpy arrays. We maybe provide tool for converting ply files into the numpy array. -->
<!-- -
- should input files also be able to be provided as a np.array or is it fine to assume path
- the paths, is there a specific way this should be done? like relative paths.
- Discuss subsampling procedure -->

- do he mean we should create our own errors, or just use build in like valueerror?
  e.g.:
  NegativeDistError - if the value returned from the base norm function name is non-positive.
  InputNumOfDistParam - the number of parameters in base norm function name doesn’t match.
  NonBoundedSubSample - subsample mapping is out of the range [0, 1].
  NegativeSparsity - the sparsity param base parameter is negative.

- How to solve the optimization problem for estimateDiffusionParamBase and estimateDiffusionParamFiber
- maps?
- how is the program used, should we provide interface, or are our code gonna be used and the arrays are provided, or is it by file paths?
- subsampling




Info gathered:
- Maybe use parquet for lazy loading of data, if it cannot be in main memory