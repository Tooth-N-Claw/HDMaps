import numpy as np

a = np.array([[1, 2], [3, 4]])

a[0:2, 0:2] *= 2
print(a)




# python src/run_HDM_teeth_CPU.py && python src/run_HDM_teeth_GPU_CuPy.py && python src/run_HDM_teeth_GPU_PyTorch.py && python src/run_HDM_teeth_GPU_PyTorch_CuPy.py && python src/run_HDM_wings_CPU.py && python src/run_HDM_wings_GPU_CuPy.py 