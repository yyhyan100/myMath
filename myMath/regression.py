import numpy as np
from math import *
import numbers

class linear_regression_single:

    def __init__(self):
        self.a0=0.0
        self.a1=0.0

    def train(self,x,y):
        n=x.shape[0]
        xy_sum=np.sum(x*y)
        x_sum=np.sum(x)
        y_sum=np.sum(y)
        x2_sum=np.sum(x**2)
        self.a1=(n*xy_sum-x_sum*y_sum)/(n*x2_sum-x_sum**2)
        self.a0=(x2_sum*y_sum-x_sum*xy_sum)/(n*x2_sum-x_sum**2)

    def print_coef(self):
        print(f"a0={self.a0}, a1={self.a1}")

    def predict(self,x_input):
        return self.a0+self.a1*x_input
        
        
class poly_regression:

    def __init__(self,degree:int=1):
        self.degree=degree
        self.a=np.zeros(degree+1,dtype=np.float64)

    def train(self,x,y):
        m=self.degree+1
        x_sum=np.array([np.sum(x**i) for i in range(1,2*self.degree+1)])
        x_sum=np.concatenate([np.array([x.shape[0]]),x_sum])
        M=np.empty((m,m))
        for i in range(m):
            for j in range(m):
                M[i,j]=x_sum[i+j]

        b=np.empty((m,1))
        b[0,0]=np.sum(y)
        for i in range(1,m):
            b[i,0]=np.sum(x**i*y)

        self.a=np.matmul(np.linalg.inv(M), b).flatten()

    def print_coef(self):
        print(self.a)

    def predict(self,x_input):
        x=np.array([x_input**i for i in range(self.degree+1)])
        return np.sum(self.a*x)

class exp_regression:
# the model is y=a*exp(b*x), we need to find the coef a & b
    def __init__(self):
        self.a=None
        self.b=None

    def train(self,x,y):
        lny=np.log(y) 
        lr=linear_regression_single()
        lr.train(x,lny)
        self.a=exp(lr.a0)
        self.b=lr.a1

    def print_coef(self):
        print(f"a={self.a},b={self.b}")

    def predict(self,x_input):
        return self.a*np.exp(self.b*x_input)

class power_regression:
# the model is y=a*x^b, we need to find the coef a & b
    def __init__(self):
        self.a=None
        self.b=None

    def train(self,x,y):
        lnx=np.log(x)
        lny=np.log(y) 
        lr=linear_regression_single()
        lr.train(lnx,lny)
        self.a=exp(lr.a0)
        self.b=lr.a1

    def print_coef(self):
        print(f"a={self.a},b={self.b}")

    def predict(self,x_input):
        return self.a*x_input**self.b