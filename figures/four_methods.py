#!/usr/bin/env python3

import numpy as np
from scipy import linalg as LA
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

# Material Properties
# -----------------------------------------------------
C   = 2500              # velcoity (m/s)
rho = 2500              # density  (kg/m^3)
Z   = rho*C             # impedance (kg/(s m^2))
mu  = rho*C**2          # bluk modulus

# Model Domain
# -----------------------------------------------------
nx  = 800               # number of grid cells
L   = 10*1000           # domain length (m)
cfl = 0.5               # Counrant Number
T   = 2                 # length of simulation (s)

# Inital Conditions
# -----------------------------------------------------
bet = 5e-6              # arugments in initial cond.
gam = 2e-5              # arugments in initial cond.
x0  = 4000              # postion of the initial conidition

# Initalizing spatial domain
x, dx = np.linspace(0,L,nx,retstep=True)

# CFL criterion to ensure convergence
dt  = cfl*dx/C              # calculate dt based on CFL criteron
nt  = int(np.floor(T/dt))   # nt from dt calculated

# Initalize solution matrix
Q_lwf = np.zeros((2,nx,nt)) # Lax-Wefford
Q_up  = np.zeros((2,nx,nt)) # Upwind
Q_a   = np.zeros((2,nx,nt)) # Analytical

# Inital condition
# -----------------------------------------------------
Q_lwf[0,:,0] = np.exp(-bet * (x-x0)**2) * np.cos(gam * (x - x0))
Q_up[0,:,0]  = np.exp(-bet * (x-x0)**2) * np.cos(gam * (x - x0))
Q_a[0,:,0]   = np.exp(-bet * (x-x0)**2) * np.cos(gam * (x - x0))


# Riemann Problem
# -----------------------------------------------------
X    = np.array([[Z,-Z],[1,1]])             # Eigenvector Matrix
Xinv = 1/(2*Z) * np.array([[1,Z],[-1,Z]])   # Inverse of eigenvector Matrix
LamP = np.array([[ 0,0],[0,C]])             # Positive Lambda
LamM = np.array([[-C,0],[0,0]])             # Negative Lambda
AP   = X @ LamP @ Xinv                      # Postive decomposed A
AM   = X @ LamM @ Xinv                      # Negative decomposed A
A    = np.array([[0, -mu],[-1/rho,0]])      # Coefficent matrix A


# Time intergration loop
# -----------------------------------------------------
for n in range(nt-1):
    # Upwind Method for 3D array
    # -------------------------------------------------
    for j in range(1,nx-1):
        dQl = Q_up[:,j,n] - Q_up[:,j-1,n]
        dQr = Q_up[:,j+1,n] - Q_up[:,j,n]
        Q_up[:,j,n+1] = Q_up[:,j,n] - (dt/dx) * (AP.dot(dQl) + AM.dot(dQr))

    #Periodic Boundary Conditions
    Q_up[:,0,n+1]  = Q_up[:,-2,n+1]
    Q_up[:,-1,n+1]  = Q_up[:,1,n+1]

    # Lax-Wendroff scheme
    # --------------------------------------------------
    for j in range(1,nx-1):
        dQ1 = Q_lwf[:,j+1,n] - Q_lwf[:,j-1,n]
        dQ2 =Q_lwf[:,j+1,n] - 2*Q_lwf[:,j,n] + Q_lwf[:,j-1,n]
        Q_lwf[:,j,n+1] = Q_lwf[:,j,n] - 0.5*(dt/dx) * A.dot(dQ1) +\
                     0.5*(dt/dx)**2 * (A@A).dot(dQ2)

    #Periodic Boundary Conditions
    Q_lwf[:,0,n+1]   = Q_lwf[:,-2,n+1]
    Q_lwf[:,-1,n+1]  = Q_lwf[:,1,n+1]

    # Analytical Solution for 1:nt-1
    # --------------------------------------------------
    Q_a[0,:,n] = (1./2.)*(np.exp(-bet * (x[:]-x0 + C*n*dt)**2)*np.cos(gam*(x[:]-x0 + C*n*dt))\
        + np.exp(-bet * (x[:]-x0 - C*n*dt)**2)*np.cos(gam*(x[:]-x0 - C*n*dt)))
    Q_a[1,:,n] = (1./(2*Z))*(np.exp(-bet * (x[:]-x0 + C*n*dt)**2)*np.cos(gam*(x[:]-x0 + C*n*dt))\
        - np.exp(-bet * (x[:]-x0 - C*n*dt)**2)*np.cos(gam*(x[:]-x0 - C*n*dt)))

# Analytical Solution for nt
# --------------------------------------------------
Q_a[0,:,n] = (1./2.)*(np.exp(-bet * (x[:]-x0 + C*n*dt)**2)*np.cos(gam*(x[:]-x0 + C*n*dt))\
        + np.exp(-bet * (x[:]-x0 - C*n*dt)**2)*np.cos(gam*(x[:]-x0 - C*n*dt)))
Q_a[1,:,n] = (1./(2*Z))*(np.exp(-bet * (x[:]-x0 + C*n*dt)**2)*np.cos(gam*(x[:]-x0 + C*n*dt))\
        - np.exp(-bet * (x[:]-x0 - C*n*dt)**2)*np.cos(gam*(x[:]-x0 - C*n*dt)))

# PyClaw
# -----------------------------------------------------
claw = pyclaw.Controller()  # initialize pyClaw Solver
claw.tfinal = T             # set intergration time equal to as defined above

claw.keep_copy = True       # Keep solution data in memory for plotting
claw.output_format = None   # Don't write solution data to file
claw.num_output_times = nt  # frames equal to time intergation steps from above

riemann_solver = riemann.acoustics_1D                # eqivalent to elastic in 1D
claw.solver = pyclaw.ClawSolver1D(riemann_solver)
claw.solver.all_bcs = pyclaw.BC.periodic             # Periodic boundary condions
claw.solver.limiters = pyclaw.limiters.tvd.superbee  # Superbee slope limiter

domain = pyclaw.Domain( (0.,), (L,), (nx,))          # Same domain as above
claw.solution = pyclaw.Solution(claw.solver.num_eqn,domain)

x_claw =domain.grid.x.centers
claw.solution.q[0,:] = np.exp(-bet * (x_claw-x0)**2) * np.cos(gam * (x_claw - x0))
claw.solution.q[1,:] = 0.


claw.solution.state.problem_data = {                 # Material properties
                              'rho' : -rho,
                              'bulk': -mu,
                              'zz'  : -Z,
                              'cc'  : C
                              }
status = claw.run()

# Plotting Figure
# -----------------------------------------------------
fig, ax = plt.subplots(2,1,sharex=True)

ax[0].set_ylabel("Stress")
ax[1].set_ylabel("Velocity",labelpad = 11, rotation=270)
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
ax[1].get_yaxis().get_offset_text().set_x(1.1)

ax[1].set_xlabel("km")
ax[0].set_title('T = {:.3f}'.format(400*dt))
#BF5E58,#707BA7,#278F69,#A6782D


ax[0].plot(x, Q_a[0,:,400],c = 'k')
ax[1].plot(x, Q_a[1,:,400],c = 'k',label='Anlytical')

ax[0].plot(x, Q_up[0,:,400],c = '#BF5E58')
ax[1].plot(x, Q_up[1,:,400],c = '#BF5E58', label='Upwind')

ax[0].plot(x, Q_lwf[0,:,400],c = '#707BA7',ls='--')
ax[1].plot(x, Q_lwf[1,:,400],c = '#707BA7',ls='--',label='Law-Wefford')


ax[0].plot(x, claw.frames[400].q[0,:],c = '#278F69',ls=':')
ax[1].plot(x, claw.frames[400].q[1,:],c = '#278F69',ls=':',label='PyClaw')



ax[1].legend(ncol=2)

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig('./four_methods.eps')
