import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint

def Sg_ode(ode,yhat,Q,k,time):
    # check whether y is a scalar or a vector
    try:
        if np.size(yhat) == np.size(yhat,0):
            y0 = yhat[0]
            N = np.size(yhat)
        else:
            y0 = yhat[:,0]
            N = np.size(yhat,1)

        p = np.size(k)
        Y,J = state_jacob_int(ode,y0,k,time)
        S,r = objective_func(yhat,Y,Q,N)
        # calculation of S' = g
        g = np.zeros(p)
        for i in range(N):
            g -= J[i].T@Q@r[:,i]
        return S, g
    except OverflowError:
        print("Problem with integration. Try with another parameters")
        return

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
    if p == 1:
        dfdk = (ode(y,k+h)-ode(y,k-h))/(2*h)
        return dfdk.reshape(n,1)
    else:
        k = k.astype(np.float)
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
    return Y,Jt_i

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
        u,s,v = LA.svd(a)
        c = u.T@b
        w = LA.solve(np.diag(s),c)
        x = v.T@w
        return x
    del_k = svdsolve(Hessian,gradient)
    return del_k

def lm(ode,yhat,Q,k0,time,opts=[1e-3,1e-8,1e-8,1000]):
    # Input arguments

    # opts = [tau, tolg, tolk, max_iter]
    #
    # Outputs
    # output = [k,Y,info]
    # k : parameters
    # Y : results with k
    # info = [it,ter]
    # it : Number of iterations
    # ter : Termination criteria 1: gradient 2: change in h 3: maximum iteration

    try:
        stop = False
        nu = 2
        it = 0  
        rho = 0
        ter = 'm'
        if np.size(yhat) == np.size(yhat,0):
            y0 = yhat[0]
            N = np.size(yhat)
        else:
            y0 = yhat[:,0]
            N = np.size(yhat,1)
        p = np.size(k0)
        I = np.eye(p)
        k = k0
        print('Iteration | Objective function | Reduced gradient |   mu   |   rho')
        Y, J = state_jacob_int(ode,y0,k,time)
        S, r = objective_func(yhat,Y,Q,N)
        S0 = S
        H,g = Hg(J,Q,r,p,N)
        print('g is')
        print(g)
        K = np.diag(k)
        Hr = K@H@K
        gr = K@g
        gn = LA.norm(gr,np.inf)
        stop = bool(gn < opts[1])
        if stop:
            ter = 'g'
        mu = opts[0]*max(np.diag(Hr))
        while (not stop) and (it<=opts[3]):
            it += 1
            hr = svdsolve(Hr+mu*I,-gr)
            h = K@hr
            hn = LA.norm(h,np.inf)
            kn = LA.norm(k,np.inf)
            if hn <= opts[2]*(kn+opts[2]):
                stop = True
                ter = 'h'
            else:
                k_new = k + h
                Y, J = state_jacob_int(ode,y0,k_new,time)
                S, r = objective_func(yhat,Y,Q,N)
                den = h.T@(mu*h-g)/2
                print(*['L0 - L is ',den])
                rho = (S0 - S)/den
                if rho >0:
                    k = k_new 
                    K = np.diag(k)
                    S0 = S
                    print('S0 is ')
                    print(S0)
                    H, g = Hg(J,Q,r,p,N)
                    print('g is')
                    print(g)
                    Hr = K@H@K
                    gr = K@g
                    gn = LA.norm(gr,np.inf) 
                    if gn < opts[1]:
                        stop = True
                        ter = 'g'
                    mu *= max(1/3,1-(2*rho-1)**3)
                    nu = 2
                else:
                    mu *= nu
                    nu *= 2
            if rho == 0:
                print("{0:10d}|{1:20.4e}|{2:18.2e}|{3:8.1e}| Not calculated"
                      .format(it,S,gn,mu))
            else:
                print("{0:10d}|{1:20.4e}|{2:18.2e}|{3:8.1e}|{4:8.1e}"
                      .format(it,S,gn,mu,rho))
            print('Iter end\n\n\n')
        info = [it,ter]
        output = [k,Y,info]
        return output
    except OverflowError:
        print("Problem with integration. Try with another parameter")
        return

def objective_func(yhat,Y,Q,N):
    S = 0
    r = yhat-Y
    if np.size(Q) == 1:
        S = np.sum(r**2)
    else:
        for i in range(N):
            # S += np.dot(np.matmul(r[:,i],Q),r[:,i])
            S += r[:,i]@Q@r[:,i]
    return S, r

def Hg(J,Q,r,p,N):
    H = np.zeros((p,p))
    g = np.zeros(p)
    for i in range(N):
        JQ = J[i].T@Q
        H += JQ@J[i]
        g -= JQ@r[:,i]
    return H,g

def svdsolve(A,b):
    u,s,v = np.linalg.svd(A)
    c = u.T@b
    w = np.linalg.solve(np.diag(s),c)
    x = v.T@w
    return x
