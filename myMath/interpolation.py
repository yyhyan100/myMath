import numpy as np
class intp_1d:
    def __init__(self):
        pass

    def Lagrange(self,x,y,x_input):
        def _Li(x,x_input,i):
            n=len(x)
            ans1=1.0
            for j in range(n):
                if j!=i: ans1= ans1*(x_input-x[j])/(x[i]-x[j])
            return ans1
        y_output=0.0
        for i in range(len(x)):
            y_output += _Li(x,x_input,i)*y[i]
        return y_output
