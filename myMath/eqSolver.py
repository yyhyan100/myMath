class solver_1d:
    def __init__(self,tol=0.000001):
        self.tol=tol

    def bisection(self, f, a=0.0, b=1.0):
        err=self.tol+1
        xl=a 
        xu=b
        f_xl=f(xl) # fucntion value at x=xl
        while err>=self.tol:
            xm=(xl+xu)/2.0
            f_xm=f(xm) # fucntion value at the midpoint
            err=abs(f_xm)
            if f_xl*f_xm < 0:
                xu=xm 
            else:
                xl=xm
                f_xl=f_xm
        return xm

    def newton(self, f, fp,p0=0.0):
        xi=p0
        err1=self.tol+1
        while err1 > self.tol:
            xi1=xi-f(xi)/fp(xi)
            err1=abs(f(xi1))
            xi=xi1
        return xi1
    
    def secant(self, f, p0=0.0,p1=0.1):
        xim1=p0
        xi=p1
        err1=self.tol+1
        while err1 > self.tol:
            xi1=xi-f(xi)*(xi-xim1)/(f(xi)-f(xim1))
            err1=abs(f(xi1))
            xim1=xi
            xi=xi1
        return xi1

class solver_nd:
    def __init__(self,ndim=2,tol=0.000001):
        self.ndim=ndim
        self.tol=tol