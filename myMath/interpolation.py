import numpy as np
import numbers
class intp_1d:
    def __init__(self):
        pass

    def Lagrange(self,x,y,x_input):
        if isinstance(x_input,numbers.Number): 
            intype_scalar=True
        else:
            intype_scalar=False
        if not isinstance(x_input,np.ndarray): x_input=np.array([x_input])

        def _Li(x,xin,i):
            n=len(x)
            ans1=1.0
            for j in range(n):
                if j!=i: ans1= ans1*(xin-x[j])/(x[i]-x[j])
            return ans1
        nn=len(x_input)
        y_output=np.zeros(nn,dtype=np.float64)
        for j in range(nn):
            for i in range(len(x)):
                y_output[j] += _Li(x,x_input[j],i)*y[i]
        if intype_scalar: 
            return y_output[0]
        else:
            return y_output

    def NewtonDD(self,x,y,x_input):
        if isinstance(x_input,numbers.Number): 
            intype_scalar=True
        else:
            intype_scalar=False
        if not isinstance(x_input,np.ndarray): x_input=np.array([x_input])
        n=len(x)
        D=np.empty((n,n),dtype=np.float64)
        D[:,0]=y

        for j in range(1,n):
            for i in range(j,n):
                D[i,j]=(D[i,j-1]-D[i-1,j-1])/(x[i]-x[i-j])

        nn=len(x_input)
        y_output=np.zeros(nn,dtype=np.float64)
        for k in range(nn):
            for i in range(1,n):
                tmp=1.0
                for j in range(i):
                    tmp*=(x_input[k]-x[j])
                y_output[k]+=D[i,i]*tmp
            y_output[k]+=D[0,0]

        if intype_scalar: 
            return y_output[0]
        else:
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
    
    def spline(self, x, y, x_input, bc=0):
        if isinstance(x_input,numbers.Number): 
            intype_scalar=True
        else:
            intype_scalar=False
        if not isinstance(x_input,np.ndarray): x_input=np.array([x_input])

        n=len(x)
        gpp=np.zeros(n,dtype=np.float64)
        b=np.zeros(n-2,dtype=np.float64)
        A=np.zeros((n-2,n-2),dtype=np.float64)
        for i in range(1,n-3):
            d_im1=x[i+1]-x[i]
            d_i=x[i+2]-x[i+1]
            A[i,i-1]=d_im1/6
            A[i,i]=(d_im1+d_i)/3
            A[i,i+1]=d_i/6
            b[i]=(y[i+2]-y[i+1])/d_i-(y[i+1]-y[i])/d_im1
        if bc==0: # free run out f''(x0)=0
            i=0
            d_im1=x[i+1]-x[i]
            d_i=x[i+2]-x[i+1]
            A[i,i]=(d_im1+d_i)/3
            A[i,i+1]=d_i/6
            b[i]=(y[i+2]-y[i+1])/d_i-(y[i+1]-y[i])/d_im1

            i=n-3
            d_im1=x[i+1]-x[i]
            d_i=x[i+2]-x[i+1]
            A[i,i-1]=d_im1/6
            A[i,i]=(d_im1+d_i)/3
            b[i]=(y[i+2]-y[i+1])/d_i-(y[i+1]-y[i])/d_im1
        else: # parabolic run out f''(x0)=f''(x1)
            i=0
            d_im1=x[i+1]-x[i]
            d_i=x[i+2]-x[i+1]
            A[i,i]=(d_im1+d_i)/3+d_im1/6
            A[i,i+1]=d_i/6
            b[i]=(y[i+2]-y[i+1])/d_i-(y[i+1]-y[i])/d_im1

            i=n-3
            d_im1=x[i+1]-x[i]
            d_i=x[i+2]-x[i+1]
            A[i,i-1]=d_im1/6
            A[i,i]=(d_im1+d_i)/3+d_i/6
            b[i]=(y[i+2]-y[i+1])/d_i-(y[i+1]-y[i])/d_im1

        gpp[1:n-1]=np.linalg.solve(A,b)
        if bc==0: # free run out f''(x0)=0
            gpp[0]=0
            gpp[n-1]=0
        else:
            gpp[0]=gpp[1]
            gpp[n-1]=gpp[n-2]

        nn=len(x_input)
        y_output=np.zeros(nn,dtype=np.float64)

        for j in range(nn):
            for i in range(1,n):
                if x_input[j]<=x[i]: break
            i-=1
            tmp1=x_input[j]-x[i]
            tmp2=x[i+1]-x_input[j]
            delta_i=x[i+1]-x[i]
            y_output[j]=gpp[i]*(tmp2**3/delta_i-delta_i*tmp2)/6+\
                gpp[i+1]*(tmp1**3/delta_i-delta_i*tmp1)/6+\
                y[i]*tmp2/delta_i+y[i+1]*tmp1/delta_i
        if intype_scalar: 
            return y_output[0]
        else:
            return y_output