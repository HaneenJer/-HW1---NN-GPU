import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    rows = X.shape[0]
    cols = X.shape[1]
    C = np.zeros((rows,rows))
    for w in range(rows):
        for i in range(rows):
            for j in range(cols):
                C[w][i] += X[w][j] * X[i][w]
    return C
                
    raise NotImplementedError("To be implemented")


@njit
def matmul_transpose_numba(X):
    rows = X.shape[0]
    cols = X.shape[1]
    C = np.zeros((rows,rows))
    for w in range(rows):
        for i in range(rows):
            for j in range(cols):
                C[w][i] += X[w][j] * X[i][w]
                
    return C
                
    raise NotImplementedError("To be implemented")


def matmul_transpose_gpu(X):
    rows = X.shape[0]
    C = np.zeros((rows, rows))
    C_gpu = cuda.to_device(C)
    X_gpu = cuda.to_device(X)
    matmul_kernel[1, 1024](X_gpu, C_gpu)
    C = C_gpu.copy_to_host()
    
    return C
    
    raise NotImplementedError("To be implemented")

@cuda.jit
def matmul_kernel(A, C):
    rows = A.shape[0]
    cols = A.shape[1]
    init = cuda.threadIdx.x
    i = 0
    while (1024*i < rows*cols) :
        if (1024*i + init) < rows**2 :
            r = (1024*i+init)//rows
            c = (1024*i+init)%rows
            sum = 0
            for k in range(cols):
                sum += A[r][k]*A[c][k]
            C[r][c] = sum
        i += 1

    raise NotImplementedError("To be implemented")

#this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()
    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X,Xt)).repeat(3, 100))

    
    #print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    matmul_comparison()
