import numpy as np
class composite:
    def __init__(self):
        pass

    def trap(self,f,a,b,n):
        dx=(b-a)/n
        tmp=0.0
        for j in range(1,n):
            tmp+=f(a+j*dx)
        tmp+=f(a)/2+f(b)/2
        tmp*=dx
        return tmp