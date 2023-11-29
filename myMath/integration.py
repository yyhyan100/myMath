import numpy as np
class composite:
    def __init__(self):
        pass

    def trapezoid(self,f,a,b,n):
        dx=(b-a)/n
        tmp=0.0
        for j in range(1,n):
            tmp+=f(a+j*dx)
        tmp+=f(a)/2+f(b)/2
        tmp*=dx
        return tmp
    
    def simpson(self,f,a,b,n):
        dx=(b-a)/n
        tmp1=0.0
        tmp2=0.0
        for j in range(1,n,2):
            tmp1+=f(a+j*dx)
        tmp1*=4

        for j in range(2,n,2):
            tmp2+=f(a+j*dx)
        tmp2*=2
        tmp=tmp1+tmp2+f(a)+f(b)
        tmp*=dx/3
        return tmp
    
    def romberg(self,f,a,b,n, order=2):
        R=np.empty((order,order), np.float64)
        for i in range(order):
            R[i,0]=self.trapezoid(f,a,b,n*(2**i))
        
        for j in range(1,order):
            for i in range(j,order):
                tmp=4**j
                R[i,j]=(tmp*R[i,j-1]-R[i-1,j-1])/(tmp-1)
        return R[order-1,order-1]