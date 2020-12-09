import numpy as np
from numba import cuda, njit, prange, float32
import timeit


def dist_cpu(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """
    pass
    sum1 = 0
    for i,j in zip(A,B):
        for x,y in zip(i,j):
            sum1 += ( abs(x - y) ** p ) 
   
    return (sum1**(1.0/p))
    
    raise NotImplementedError("To be implemented")
        


@njit(parallel=True)
def dist_numba(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """
    pass
    sum1 = 0
    for i in prange(1000):
        for j in prange(1000):
            sum1 += ( abs(A[i][j] - B[i][j]) ** p ) 
   
    return (sum1**(1.0/p))
    
    raise NotImplementedError("To be implemented")



def dist_gpu(A, B, p):
    """
     Returns
     -------
     np.array
         p-dist between A and B
     """
    pass
    C = np.zeros(1)
    
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    C_gpu = cuda.to_device(C)
    
    dist_kernel[1000, 1000](A_gpu, B_gpu, p, C_gpu)
    C = C_gpu.copy_to_host()
    
    return (C[0]**(1/p))
    
    raise NotImplementedError("To be implemented")
    

@cuda.jit
def dist_kernel(A, B, p, C):
    pass
    i = cuda.threadIdx.x
    j = cuda.blockIdx.x
    
    cuda.atomic.add(C, 0, ((abs(A[i][j] - B[i][j])) ** p))
  
   
#this is the comparison function - keep it as it is.
def dist_comparison():
    A = np.random.randint(0,256,(1000, 1000))
    B = np.random.randint(0,256,(1000, 1000))
    p = [1, 2]

    def timer(f, q):
        return min(timeit.Timer(lambda: f(A, B, q)).repeat(3, 20))


    for power in p:
        print('p=' + str(power))
        print('     [*] CPU:', timer(dist_cpu,power))
        print('     [*] Numba:', timer(dist_numba,power))
        print('     [*] CUDA:', timer(dist_gpu, power))

if __name__ == '__main__':
    dist_comparison()
