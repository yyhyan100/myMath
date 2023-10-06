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

    def NewtonDD(self,x,y,x_input):
        n=len(x)
        D=np.empty((n,n),dtype=np.float64)
        D[:,0]=y

        for j in range(1,n):
            for i in range(j,n):
                D[i,j]=(D[i,j-1]-D[i-1,j-1])/(x[i]-x[i-j])

        y_output=0.0

        for i in range(1,n):
            tmp=1.0
            for j in range(i):
                tmp*=(x_input-x[j])
            y_output+=D[i,i]*tmp
        y_output+=D[0,0]
        return y_output