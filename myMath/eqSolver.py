class eqSolver:
    def __init__(self,tol=0.000001):
        self.tol=tol

    def bisection(self, f, a=0.0, b=1.0):
        xm=(a+b)/2.0
        err1=abs(f(xm))
        while err1 > self.tol:
            if f(a)*f(xm) <= 0.0:
                b=xm
            else:
                a=xm 
            xm=(a+b)/2.0   
            err1=abs(f(xm))
        return xm

    def newton(self, f, fp,p0=0.0):
        xi=p0
        err1=self.tol+1
        while err1 > self.tol:
            xi1=xi-f(xi)/fp(xi)
            err1=abs(f(xi1))
            xi=xi1
        return xi1