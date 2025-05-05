
# Setup

- Install requirements with `pip install -r requirements.txt`
- If running on gpu, then install pytorch or/and cupy depending on which gpu versions you run. Note cupy requires nvidia gpu

# Running HDM on platyrrhine teeth

In order to run, add the data prepared by Rob. The data directory should be inside HDM_Python, and needs to be named "platyrrhine".
The platyrrhine folder should contain:

- Names.mat
- FinalDist.mat
- softMapMatrix.mat
- ReparametrizedOFF with the .off files inside.

By running `python src/run_HDM_teeth_CPU.py` from within the src folder, you should get the pringle below.

Proof of Concept Pringle (PCP):

![](pringle.png)

# Backends

## CPU

This version uses numpy, and is mainly single threaded
**Time:** `python src/run_HDM_teeth_CPU.py: 308.21s user 1.61s system 1105% cpu 28.018 total`

## GPU PyTorch

Pytorch is used, however due to poor sparse eigendecomposition support, it is rather slow.
**Time:** `python src/run_HDM_teeth_GPU_PyTorch.py  63.32s user 2.86s system 186% cpu 35.568 total`

## GPU PyTorch + CuPy

In an attempt to circumvent PyTorch's poor sparse eigendecomposition support, a hybrid version was made that uses CuPy for the eigendecomposition, speeding it up substantially. However there is overhead due to conversions between them.
**Time:** `python src/run_HDM_teeth_GPU_PyTorch_CuPy.py  40.65s user 1.52s system 303% cpu 13.876 total`

## GPU CuPy (Recommended)

A version using CuPy was made, due to its good support for sparse operations. This backend ended up being the fastest
**Time:** `python src/run_HDM_teeth_GPU_CuPy.py  13.77s user 1.85s system 133% cpu 11.703 total`
