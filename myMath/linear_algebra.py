import numpy as np
class linear_system_solver:
    def __init__(self):
        pass

    def gaussian_elimination(self,coef_matrix,rhs):
        A=coef_matrix.copy()
        b=rhs.copy()
        n=b.size
        x=np.empty(n)
        for k in range(n-1):
            for i in range(k+1,n):
                mik=-A[i,k]/A[k,k]
                for j in range(k+1,n):
                    A[i,j]+=mik*A[k,j]
                b[i]+=mik*b[k]

        x[-1]=b[-1]/A[-1,-1]
        for k in range(n-2,-1,-1):
            tmp=0.0
            for j in range(k+1,n):
                tmp+=A[k,j]*x[j]
            x[k]=(b[k]-tmp)/A[k,k]
        return x



