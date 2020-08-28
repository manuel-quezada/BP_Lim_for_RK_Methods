# Bound-preserving limiting for Runge-Kutta methods

This repository contains the code to reproduce the results in "Bound-preserving convex limiting for high-order Runge-Kutta time discretizations of hyperbolic conservation laws" by D. Kuzmin, M. Quezada de Luna, D.I. Ketcheson and J. Grull. 

## Dependencies
This code requires Python 3 (we use 3.7.7) and the following libraries: 
<li>numpy</li>
<li>matplotlib</li>
<li>scipy</li>
<li>nodepy</li>
<li>A fortran compiler</li>
<li>f2py</li>

## Compilation
Inside the folder 'code', type make. 

## Running the examples
The code run\_problems.py calls functions inside time\_flux\_limiting.py to run the different problems in the manuscript. Inside run\_problems.py there are different 'sections' that can be activated via if False/True statements. Activate the different sections to create the different tables and figures in the manuscript. The following sections are inside run\_convergence.py:

<li> CONVERGENCE OF SEMI-DISCRETIZATION. This code produces the results in Section 5.1. </li>
<li> CONVERGENCE TO SMOOTH SOLUTIONS. If solution_type=0, the code produces the results in the first part of Section 5.2. If solution_type=3, the code produces the results in the first part of Section 5.3. </li>
<li> LINEAR ADVECTION WITH NON-SMOOTH SOLUTION. This code produces the results in the second part of Section 5.2. </li>
<li> NONLINEAR BURGERS AFTER SHOCK. This code produces the results in the second part of Section 5.3. </li>
<li> NONLINEAR ONE DIM KPP. This code produces the results in Section 5.4. </li>
