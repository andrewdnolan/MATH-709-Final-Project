# MATH 709: Finite Volume Solvers
An introduction to finite volume solvers

Numerical experiments using finite volume solvers to investigate the 1-D Elastic Wave Equations
$$
\begin{align*}    \frac{\partial}{\partial t}\sigma - \mu \frac{\partial}{\partial x} v &= 0 \\
    \ \frac{\partial}{\partial t}v - \frac{1}{\rho} \frac{\partial}{\partial x} \sigma &= 0
\end{align*}
$$

Repository Structure:
```
├── notebooks  
│   ├── homogenous_1D_elastic.ipynb
│   └── non_homogenous_1D_elastic.ipynb  
├── pyfiles  
│   ├── heterogenous.py  
│   └── homogenous.py  
└── report  
```
