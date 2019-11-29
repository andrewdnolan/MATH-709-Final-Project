#!/usr/bin/env python3

import numpy as np
from clawpack import pyclaw
from clawpack import riemann
from scipy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import animation, rc

# Material Properties
# -----------------------------------------------------
C   = 2500              # velcoity (m/s)
rho = 2500              # density  (kg/m^3)
Z   = rho*C             # impedance (kg/(s m^2))
mu  = rho*C**2          # bluk modulus

# Model Domain
# -----------------------------------------------------
nx  = 1000              # number of grid cells
L   = 10*1000           # domain length (m)
cfl = 0.9               # Counrant Number
T   = 5                 # length of simulation (s)

# Inital Conditions
# -----------------------------------------------------
bet = 5e-6              # arugments in initial cond.
gam = 2e-5              # arugments in initial cond.
x0  = 5000              # postion of the initial conidition

# Initalizing spatial domain
x, dx = np.linspace(0,L,nx,retstep=True)

# Calculate time step based on CFL criterion
dt  = cfl*dx/C
nt  = int(np.floor(T/dt))

# Initalize solution matrix
Q   = np.zeros((2,nx,nt))
Q_up = np.zeros((2,nx,nt))
#Q_n = np.zeros((2,nx,nt))

# Inital condition
# -----------------------------------------------------
Q[0,:,0]  = np.exp(-bet * (x-x0)**2) * np.cos(gam * (x - x0))
Q_up[0,:,0] = np.exp(-bet * (x-x0)**2) * np.cos(gam * (x - x0))



X    = np.array([[Z,-Z],[1,1]])             # Eigenvector Matrix
Xinv = 1/(2*Z) * np.array([[1,Z],[-1,Z]])   # Inverse of eigenvector Matrix
LamP = np.array([[ 0,0],[0,C]])             # Positive Lambda
LamM = np.array([[-C,0],[0,0]])             # Negative Lambda
AP   = X @ LamP @ Xinv                      # Postive decomposed A
AM   = X @ LamM @ Xinv                      # Negative decomposed A
A    = np.array([[0, -mu],[-1/rho,0]])      # Coefficent matrix A


# Upwind Method for 3D array
# -----------------------------------------------------
for n in range(nt-1):
    for j in range(1,nx-1):
        dQl = Q_up[:,j,n] - Q_up[:,j-1,n]
        dQr = Q_up[:,j+1,n] - Q_up[:,j,n]
        Q_up[:,j,n+1] = Q_up[:,j,n] - (dt/dx) * (AP.dot(dQl) + AM.dot(dQr))

    Q_up[:,0,n+1]  = Q_up[:,-2,n+1]
    Q_up[:,-1,n+1]  = Q_up[:,1,n+1]

# Lax-Wendroff scheme
# -----------------------------------------------------
for n in range(nt-1):
    for j in range(1,nx-1):
        dQ1 = Q[:,j+1,n] - Q[:,j-1,n]
        dQ2 = Q[:,j+1,n] - 2*Q[:,j,n] + Q[:,j-1,n]
        Q[:,j,n+1] = Q[:,j,n] - 0.5*(dt/dx) * A.dot(dQ1) +\
                     0.5*(dt/dx)**2 * (A@A).dot(dQ2)

    Q[:,0,n+1]  = Q[:,-2,n+1]
    Q[:,-1,n+1]  = Q[:,1,n+1]


# PyClaw
# -----------------------------------------------------
claw = pyclaw.Controller()  # initialize pyClaw Solver
claw.tfinal = T             # set intergration time equal to as defined above

claw.keep_copy = True       # Keep solution data in memory for plotting
claw.output_format = None   # Don't write solution data to file
claw.num_output_times = nt  # Write 50 output frames

riemann_solver = riemann.acoustics_1D
claw.solver = pyclaw.ClawSolver1D(riemann_solver)
claw.solver.all_bcs = pyclaw.BC.periodic
claw.solver.limiters = pyclaw.limiters.tvd.superbee

domain = pyclaw.Domain( (0.,), (L,), (nx,))
claw.solution = pyclaw.Solution(claw.solver.num_eqn,domain)

x_claw =domain.grid.x.centers
claw.solution.q[0,:] = np.exp(-bet * (x_claw-x0)**2) * np.cos(gam * (x_claw - x0))
claw.solution.q[1,:] = 0.

plt.plot(x_claw, claw.solution.q[0,:],'-o')

claw.solution.state.problem_data = {
                              'rho' : -rho,
                              'bulk': -mu,
                              'zz'  : -Z,
                              'cc'  : C
                              }
status = claw.run()
