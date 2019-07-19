import GN_ode as gn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def dehydro_ben(y,k):
    K1 = 0.242
    K2 = 0.428
    r1 = k[0]*(y[0]**2-y[1]*(2-2*y[0]-y[1])/(3*K1))
    r2 = k[1]*(y[0]*y[1]-(1-y[0]-2*y[1])*(2-2*y[0]-y[1])/(9*K2))
    dydt = np.empty(2)
    dydt[0] = -r1-r2
    dydt[1] = r1/2-r2
    return dydt

t = np.array([5.63,11.32,16.97,22.62,34.00,39.70,45.20,169.7])*1e-4
yhat = np.array([[0.828,0.704,0.622,0.565,0.499,0.482,0.470,0.443],
                   [0.0737,0.1130,0.1322,0.1400,0.1468,0.1477,0.1477,0.1476]])
y0 = yhat[:,0]
k = np.array([354.61,400.23])
k0 = np.array([10000,10000])
y,J = gn.state_jacob_int(dehydro_ben,y0,k,t)
Y,Jt = gn.state_jacob_int(dehydro_ben,y0,k0,t)

n = np.size(yhat,0)
N = np.size(yhat,1)
Q = np.eye(n)
S = gn.objective_func(yhat,Y,Q,N)
