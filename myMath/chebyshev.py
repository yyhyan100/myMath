from math import *
import numpy as np

class chebyshev_approximation:

    def __init__(self):
        self.c=None
    
    def get_zeros(self,N):
        return np.array([cos(pi*(2*k+1)/2/N) for k in range(N)])

    def train(self,X,Y):
        self.N=X.size 
        N=self.N
        self.c=np.zeros(N)
        self.c[0]=np.sum(Y)/N 
        for i in range(1,N):
            for k in range(N):
                self.c[i]+=Y[k]*cos(i*(2*k+1)*pi/2/N)
            self.c[i]*=2/N

    def predict(self,x_input):
        N=self.N
        m=x_input.size
        y2=np.zeros(m)
        y1=np.zeros(m)
        y0=np.zeros(m)
        for i in range(N-1,-1,-1):
            y0[:]=self.c[i]-y2[:]+2*x_input*y1[:]
            y2=y1.copy()
            y1=y0.copy()
        return y1-x_input*y2