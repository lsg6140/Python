import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint

def Sg_ode(ode,yhat,Q,k,time):
    # check whether y is vector or matrix
    try:
        if np.size(yhat) == np.size(yhat,0):
            y0 = yhat[0]
            N = np.size(yhat)
        else:
            y0 = yhat[:,0]
            N = np.size(yhat,1)

        p = np.size(k)
        Y,J,info = state_jacob_int(ode,y0,k,time)
        S,diff = objective_func(yhat,Y,Q,N)
        # calculation of S' = g
        g = np.zeros(p)
        for i in range(N):
            g -= J[i].T@Q@diff[:,i]
        return S, g
    except OverflowError:
        print("Problem with integration. Try with another parameters")
        return

def interpolate(a,b,phia,phib,dphia,n):
    d = b-a
    c = (phib-phia-d*dphia)/d**2
    if c>=5*n*np.eps*b:
        alpha = a-dphia/(2*c)
        d = 0.1*d
        alpha = min(max(alpha,a+d),b-d)
    else:
        alpha = (a+b)/2
    return alpha

def checkfgH(func,y,k):
    return func(y,k)

def dfdy_ode(ode,y,k,n):
    h = 1e-8
    y = y.astype(np.float)
    if np.isscalar(y):
        dfdy = (ode(y+h,k)-ode(y-h,k))/(2*h)
        return dfdy
    else:
        dfdy = np.empty((n,n))
        for i in range(n):
            yr = y.copy()
            yl = y.copy()
            yr[i] += h
            yl[i] -= h
            dfdy[i] = (ode(yr,k)-ode(yl,k))/(2*h)
        return dfdy.transpose()
    return

def dfdk_ode(ode,y,k,n,p):
    h = 1e-8
    k = k.astype(np.float)
    if p == 1:
        dfdk = (ode(y,k+h)-ode(y,k-h))/(2*h)
        return dfdk
    else:
        dfdk = np.empty((p,n))
        for i in range(p):
            kr = k.copy()
            kl = k.copy()
            kr[i] += h
            kl[i] -= h
            dfdk[i] = (ode(y,kr)-ode(y,kl))/(2*h)
        return dfdk.transpose()
    return

def phi_z(ode,z,k,n,p):
    y = z[0:n]
    J = z[n:].reshape((p,n)).transpose()
    phiz = np.empty(n*(p+1))
    dfdy = dfdy_ode(ode,y,k,n)
    dfdk = dfdk_ode(ode,y,k,n,p)
    dJdt = dfdy@J+dfdk
    phiz[0:n] = ode(y,k)
    phiz[n:] = dJdt.transpose().flatten()
    return phiz

def state_jacob_int(ode,y0,k,time):
    n = np.size(y0)
    p = np.size(k)
    N = np.size(time)
    # initial condition J0 = 0
    z0 = np.zeros(n*(p+1))
    z0[0:n] = y0
    def dzdt(t,z):
        return phi_z(ode,z,k,n,p)
    solution = solve_ivp(dzdt,[time[0],time[-1]],z0,method='Radau',t_eval=time)
    if solution.success == False:
        raise OverflowError("Integration by state_jacob_int failed")
    Z = solution.y
    Y = Z[0:n]
    J = Z[n:]
    Jt_i = np.hsplit(J,N)
    for i in range(N):
        Jt_i[i] = Jt_i[i].reshape(p,n).transpose()
    return Y,Jt_i,solution.success

def state_only_int(ode,y0,k,time):
    def dydt(t,y):
        return ode(y,k)
    solution = solve_ivp(dydt,[time[0],time[-1]],y0,method='Radau',t_eval=time)
    return solution.y,solution.success

def delta_k(J,Q,yhat,Y,p,N):
    if np.shape(yhat) != np.shape(Y):
        raise ValueError('size mismatch of yhat and Y')
    Hessian = np.zeros((p,p))
    gradient = np.zeros(p)
    for i in range(N):
        JQ = J[i].T@Q
        Hessian += JQ@J[i]
        gradient += JQ@(yhat[:,i]-Y[:,i])
    # solve using singluar value decomposition
    def svdsolve(a,b):
        u,s,v = np.linalg.svd(a)
        c = u.T@b
        w = np.linalg.solve(np.diag(s),c)
        x = v.T@w
        return x
    del_k = svdsolve(Hessian,gradient)
    return del_k

def objective_func(yhat,Y,Q,N):
    S = 0
    diff = yhat-Y
    if np.size(Q) == 1:
        S = np.sum(diff**2)
    else:
        for i in range(N):
            # S += np.dot(np.matmul(diff[:,i],Q),diff[:,i])
            S += diff[:,i]@Q@diff[:,i]
    return S, diff

def bisect(ode,yhat,Q,k,time,iter_max):
    # check whether y is 1-dimensional
    try:
        if np.size(yhat) == np.size(yhat,0):
            y0 = yhat[0]
            N = np.size(yhat)
        else:
            y0 = yhat[:,0]
            N = np.size(yhat,1)
        p = np.size(k)
        Y,J,suc = state_jacob_int(ode,y0,k,time)
        dk = delta_k(J,Q,yhat,Y,p,N)
        mu = 1.0
        S0 = objective_func(yhat,Y,Q,N)
        for j in range(iter_max):
            k_next = k + mu * dk
            Y_next,fos = state_only_int(ode,y0,k_next,time)
            if fos == False:
                mu /= 2
            else:
                S = objective_func(yhat,Y_next,Q,N)
                if S < S0:
                    break
                mu /= 2
        return Y,Y_next,J,dk,mu
    except OverflowError:
        print("Problem with integration. Try with another parameter")
        return
