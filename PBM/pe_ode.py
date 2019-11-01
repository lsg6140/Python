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
    dJdt = dfdy@J+dfdk
    phiz[0:n] = func(y,k)
    phiz[n:] = dJdt.transpose().flatten()
    return phiz

def state_jacob_int(func,y0,k,time):
    n = np.size(y0)
    p = np.size(k)
    N = np.size(time)
    # initial condition J0 = 0
    z0 = np.zeros(n*(p+1))
    z0[0:n] = y0
    def dzdt(t,z):
        return phi_z(func,z,k,n,p)
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

def state_only_int(func,y0,k,time):
    def dydt(t,y):
        return func(y,k)
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

def chi_squared(yhat,Y,Q,N):
    S = 0
    diff = yhat-Y
    if np.size(Q) == 1:
        S = np.sum(diff**2)
    else:
        for i in range(N):
            # S += np.dot(np.matmul(diff[:,i],Q),diff[:,i])
            S += diff[:,i]@Q@diff[:,i]
    return S

def bisect(func,yhat,Q,k,time,iter_max):
    # check whether y is 1-dimensional
    try:
        if np.size(yhat) == np.size(yhat,0):
            y0 = yhat[0]
            N = np.size(yhat)
        else:
            y0 = yhat[:,0]
            N = np.size(yhat,1)
        p = np.size(k)
        Y,J,suc = state_jacob_int(func,y0,k,time)
        dk = delta_k(J,Q,yhat,Y,p,N)
        mu = 1.0
        S0 = chi_squared(yhat,Y,Q,N)
        for j in range(iter_max):
            k_next = k + mu * dk
            Y_next,fos = state_only_int(func,y0,k_next,time)
            if fos == False:
                mu /= 2
            else:
                S = chi_squared(yhat,Y_next,Q,N)
                if S < S0:
                    break
                mu /= 2
        return Y,Y_next,J,dk,mu
    except OverflowError:
        print("Problem with integration. Try with another parameter")
        return

def optimal_step_size(Y,Y_next,yhat,J,dk,mu_a,Q,n,N):
    beta = np.zeros(4)
    for i in range(N):  
        Jdk = J[i] @ dk
        dy = yhat[:,i]-Y[:,i]
        r = (Y_next[:,i]-Y[:,i]-mu_a*Jdk)/mu_a**2
        rQ = r @ Q
        beta[0] += 2*rQ @ r
        beta[1] += 3*rQ @ Jdk
        beta[2] += Jdk.T @ Q @ Jdk-2*rQ @ dy
        beta[3] += -Jdk.T @ Q @ dy
    mu_opt = np.roots(beta)
    return mu_opt
