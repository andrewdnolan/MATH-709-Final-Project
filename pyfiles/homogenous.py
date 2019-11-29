#!/usr/bin/env python3

import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import animation, rc

# Model Domain
# -----------------------------------------------------
nx  = 1000              # number of grid cells
L   = 10*1000           # domain length (m)
cfl = 0.9               # Counrant Number
T   = 2                 # length of simulation (s)

# Material Properties
# -----------------------------------------------------
C   = 2500              # velcoity (m/s)
rho = 2500              # density  (kg/m^3)
Z   = rho*C             # impedance (kg/(s m^2))
mu  = rho*C**2          # bluk modulus

# Inital Conditions
# -----------------------------------------------------
bet = 5e-6              # arugments in initial cond.
gam = 2e-5              # arugments in initial cond.
x0  = 1000              # postion of the initial conidition

# Heterogenous Set Up
# --------------------------------------------------------------------------
A = np.zeros((2,2,nx))
Z = np.zeros(nx)
c = np.zeros(nx)

# Spatially dependent velocity
c = c + C
c[int(nx/2):nx] = c[int(nx/2):nx]*3.

Z = rho*c

for i in range(1,nx):
    if i > nx/2:
        A[:,:,i] = np.array([[0, -mu],[-1/rho,0]])
    else:
        A[:,:,i] = np.array([[0, -4*mu],[-1/rho,0]])

# Initalizing spatial domain
x, dx = np.linspace(0,L,nx,retstep=True)

# Calculate time step based on CFL criterion
dt  = cfl*dx/c.max()
nt  = int(np.floor(T/dt))

# Initalize solution matrix
Q   = np.zeros((2,nx,nt))
#Q_n = np.zeros((2,nx,nt))

# Inital condition
# -----------------------------------------------------
Q[0,:,0]  = np.exp(-bet * (x-x0)**2) * np.cos(gam * (x - x0))


# Lax-Wendroff scheme
# -----------------------------------------------------
for n in range(nt-1):
    for j in range(1,nx-1):
        dQ1 = Q[:,j+1,n] - Q[:,j-1,n]
        dQ2 = Q[:,j+1,n] - 2*Q[:,j,n] + Q[:,j-1,n]
        Q[:,j,n+1] = Q[:,j,n] - 0.5*(dt/dx) * A[:,:,i].dot(dQ1) +\
                     0.5*(dt/dx)**2 * (A[:,:,i]@A[:,:,i]).dot(dQ2)

    Q[:,0,n+1]  = Q[:,-2,n+1]
    Q[:,-1,n+1]  = Q[:,1,n+1]
    
