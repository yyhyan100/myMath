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

    def jacobi(self,coef_matrix,rhs,init_guess=0,tol=0.0001):
        n=rhs.size
        x_new=np.empty(n)
        x_old=np.empty(n)
        x_old[:]=init_guess
        relative_error=np.empty(n)
        relative_error[:]=tol+1

        while np.max(relative_error)>=tol:
            for i in range(n):
                tmp=0.0
                for j in range(n):
                    if j !=i : tmp+=coef_matrix[i,j]*x_old[j]
                x_new[i]=(rhs[i]-tmp)/coef_matrix[i,i]
                if x_new[i]==0.0:
                    relative_error[i]=0 
                else:
                    relative_error[i]=abs((x_new[i]-x_old[i])/x_new[i])
            x_old[:]=x_new[:]
        return x_new
    
    def gauss_seidel(self,coef_matrix,rhs,init_guess=0,tol=0.0001):
        n=rhs.size
        x=np.empty(n)
        x[:]=init_guess
        relative_error=np.empty(n)
        relative_error[:]=tol+1

        while np.max(relative_error)>=tol:
            for i in range(n):
                tmp=0.0
                for j in range(n):
                    if j !=i : tmp+=coef_matrix[i,j]*x[j]
                tmp2=x[i]
                x[i]=(rhs[i]-tmp)/coef_matrix[i,i]
                if x[i]==0.0:
                    relative_error[i]=0 
                else:
                    relative_error[i]=abs((x[i]-tmp2)/x[i])
        return x

    def LU(self,A):
        n=A.shape[0] 
        L=np.zeros((n,n),dtype=np.double)
        U=L.copy()
        for i in range(n): 
            L[i,i]=1.0
        U[0,:]=A[0,:]
        L[1:,0]=A[1:,0]/U[0,0]
        for k in range(1,n):
            for j in range(k,n):
                tmp=0.0
                for p in range(k+1):
                    tmp+=L[k,p]*U[p,j]
                U[k,j]=A[k,j]-tmp
            for i in range(k+1,n):
                tmp=0.0
                for p in range(k+1):
                    tmp+=L[i,p]*U[p,k]
                L[i,k]=(A[i,k]-tmp)/U[k,k]
        return L,U

    def LU_solver(self, A, b):
        L,U=self.LU(A)
        n=A.shape[0] 
        
        y=np.zeros(n,dtype=np.double)
        y[0]=b[0]
        for i in range(1,n):
            tmp=0.0
            for k in range(i):
                tmp+=L[i,k]*y[k]
            y[i]=b[i]-tmp

        x=np.zeros(n,dtype=np.double)
        x[-1]=y[-1]/U[-1,-1]
        for i in range(n-2,-1,-1):
            tmp=0.0 
            for k in range(i+1,n):
                tmp+=U[i,k]*x[k]
            x[i]=(y[i]-tmp)/U[i,i]

        return x



