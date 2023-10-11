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
    
    def Hermite_2pts(self,x,y,yp,x_input):
        tmp1=(x_input-x[0])/(x[1]-x[0])
        tmp2=(x_input-x[1])/(x[0]-x[1])
        return (1+2*tmp1)*tmp2**2*y[0]+(1+2*tmp2)*tmp1**2*y[1]+\
             (x_input-x[0])*tmp2**2*yp[0]+(x_input-x[1])*tmp1**2*yp[1]
    
    def Hermite_3pts(self,x,y,yp,x_input):
        tmp=-(x-x_input)
        tp1=x[0]-x[1]
        tp2=x[0]-x[2]
        tp3=x[2]-x[1]
        return tmp[1]**2*tmp[2]*y[0]/tp1**2/tp2+\
            (1+tmp[1]/tp1+tmp[1]/tp3)*tmp[0]*tmp[2]*y[1]/tp1/tp3-\
            tmp[0]*tmp[1]**2*y[2]/tp2/tp3**2+\
            tmp[0]*tmp[1]*tmp[2]*yp/tp1/tp3