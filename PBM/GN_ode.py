import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint

def dfdy_ode(func,y,k,n):
    h = 1e-8
    y = y.astype(np.float)
    if np.isscalar(y):
        dfdy = (func(y+h,k)-func(y-h,k))/(2*h)
        return dfdy
    else:
        dfdy = np.empty((n,n))
        for i in range(n):
            yr = y.copy()
            yl = y.copy()
            yr[i] += h
            yl[i] -= h
            dfdy[i] = (func(yr,k)-func(yl,k))/(2*h)
        return dfdy.transpose()
    return

def dfdk_ode(func,y,k,n,p):
    h = 1e-8
    k = k.astype(np.float)
    if p == 1:
        dfdk = (func(y,k+h)-func(y,k-h))/(2*h)
        return dfdk
    else:
        dfdk = np.empty((p,n))
        for i in range(p):
            kr = k.copy()
            kl = k.copy()
            kr[i] += h
            kl[i] -= h
            dfdk[i] = (func(y,kr)-func(y,kl))/(2*h)
        return dfdk.transpose()
    return

def phi_z(func,z,k,n,p):
    y = z[0:n]
    J = z[n:].reshape((p,n)).transpose()
    phiz = np.empty(n*(p+1))
    dfdy = dfdy_ode(func,y,k,n)
    dfdk = dfdk_ode(func,y,k,n,p)
    dJdt = np.matmul(dfdy,J)+dfdk
    phiz[0:n] = func(y,k)
    phiz[n:] = dJdt.transpose().flatten()
    return phiz

def state_jacob_int(func,y0,k,time):
    if np.isscalar(y0):
        n = 1
    else:
        n = np.size(y0)
    p = np.size(k)
    N = np.size(time)
    # initial condition J0 = 0
    z0 = np.zeros(n*(p+1))
    z0[0:n] = y0
    def dzdt(t,z):
        return phi_z(func,z,k,n,p)
    solution = solve_ivp(dzdt,[time[0],time[-1]],z0,method='Radau',t_eval=time)
    Z = solution.y
    Y = Z[0:n]
    J = Z[n:]
    Jt_i = np.hsplit(J,N)
    for i in range(N):
        Jt_i[i] = Jt_i[i].reshape(p,n).transpose()
    return Y,Jt_i

def state_only_int(func,y0,k,time):
    if np.isscalar(y0):
        def dydt(y,t):
            return func(y,k)
        solution = odeint(dydt,y0,time)
        return solution.transpose()
    else:
        def dydt(t,y):
            return func(y,k)
        solution = solve_ivp(dydt,[time[0],time[-1]],y0,method='Radau',t_eval=time)
        return solution.y
    return

def delta_k(J,Q,yhat,Y,p,N):
    Hessian = np.zeros((p,p))
    gradient = np.zeros(p)
    # Check whether y is scalar
    if np.size(Q) == 1:
        print('solve for scalar y')
        for i in range(N):
            Hessian += np.matmul(J[i].transpose(),J[i])
            gradient += np.dot(J[i].transpose(),yhat[i]-Y[:,i])
    else:
        print('solve for vector y')
        for i in range(N):
            JQ = np.matmul(J[i].transpose(),Q)
            Hessian += np.matmul(JQ,J[i])
            gradient += np.dot(JQ,yhat[:,i]-Y[:,i])
    del_k = np.linalg.solve(Hessian,gradient)
    return del_k

def chi_squared(yhat,Y,Q,N):
    S = 0
    diff = yhat-Y
    if np.size(Q) == 1:
        S = np.sum(diff**2)
    else:
        for i in range(N):
            S += np.dot(np.matmul(diff[:,i],Q),diff[:,i])
    return S

def bisect(func,yhat,Q,k,time,iter_max):
    # check whether y is 1-dimensional
    if np.size(yhat) == np.size(yhat,0):
        y0 = yhat[0]
        N = np.size(yhat)
    else:
        y0 = yhat[:,0]
        N = np.size(yhat,1)
    p = np.size(k)
    Y,J = state_jacob_int(func,y0,k,time)
    dk = delta_k(J,Q,yhat,Y,p,N)
    mu = 1.0
    S0 = chi_squared(yhat,Y,Q,N)
    for j in range(iter_max):
        k_temp = k + mu * dk
        Y_temp = state_only_int(func,y0,k_temp,time)
        S = chi_squared(yhat,Y_temp,Q,N)
        if S < S0:
            break
        mu /= 2
    return k_temp,Y,J
