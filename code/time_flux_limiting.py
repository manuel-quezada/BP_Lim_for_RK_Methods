import sys 
import numpy as np 
import matplotlib.pyplot as plt
from nodepy import rk
from scipy.optimize import fsolve

import weno
weno = weno.weno

Min = np.minimum
Max = np.maximum
tol = 1E-10

#################################
# ***** INITIAL CONDITION ***** #
#################################
def get_init_condition(solution_type):
    # solution_type = 0: smooth
    #               = 1: three body problem
    #               = 2: Burgers with non-moving shock
    #               = 3: Burgers with moving shock
    #               = 4: one dim KPP
    if solution_type == 0:
        u_init = lambda x: np.exp(-100*(x-0.5)**2)
        uMin = 0.0
        uMax = 1.0
    elif solution_type == 1:
        arg_sqrt = lambda x: 1-((2*x-1.6)/0.2)**2        
        u_init = lambda x: (np.exp(-300*(2*x-0.3)**2)*(np.abs(2*x-0.3)<=0.25)
                            +1.0*(np.abs(2*x-0.9)<=0.2)
                            +np.sqrt(arg_sqrt(x)*(arg_sqrt(x)>=0))*(np.abs(2*x-1.6)<=0.2)
                            )
        uMin = 0.0
        uMax = 1.0
    elif solution_type == 2:
        u_init = lambda x: np.sin(2*np.pi*x)
        uMin = -1.0
        uMax = 1.0
    elif solution_type == 3:
        u_init = lambda x: 0.5 + np.sin(x)
        uMin = -0.5
        uMax = 1.5
    #
    elif solution_type == 4: 
        u_init = lambda x: 1.0*(x>0.35)
        uMin = 0.0
        uMax = 1.0
    #
    return u_init, uMin, uMax
#

##############################
# ***** EXACT SOLUTION ***** #
##############################
def func_solution_type_2(u,*args):
    x,t = args
    return np.sin(2*np.pi*(x-u*t))-u
#

def exact_solution_type_2(x,t):
    tc=1.0/(2*np.pi)
    if t<tc:
        uGuess = np.sin(2*np.pi*x)
        return fsolve(func_solution_type_2,uGuess,args=(x,t)) 
    else:
        xL = 0.5
        uL = 1.0/t*xL
        uGuess = (1.0/t*x)*(x<0.5) + (1.0/t*(x-0.5)-uL)*(x>=0.5)
        return fsolve(func_solution_type_2,uGuess,args=(x,t))
    #
#

def integrate(u,x):
    return np.trapz(u,x=x)/(2*np.pi)
#

def func_solution_type_3(u,*args):
    x,t = args
    sol = 0.5 + np.sin((x-u*t)) - u
    return sol
#

def find_root(u0,x,t):
    soln = fsolve(func_solution_type_3,u0,args=(x,t))
    return soln
#

def get_exact_solution_from_guess(x,u0,time):
    return find_root(u0,x,time)
#

def get_exact_solution(solution_type,x,t):
    if solution_type == 0:
        u_exact = np.exp(-100*(x-0.5)**2)
    elif solution_type == 1:
        arg_sqrt = lambda x: 1-((2*x-1.6)/0.2)**2        
        u_exact_lambda = lambda x: (np.exp(-300*(2*x-0.3)**2)*(np.abs(2*x-0.3)<=0.25)
                                    +1.0*(np.abs(2*x-0.9)<=0.2)
                                    +np.sqrt(arg_sqrt(x)*(arg_sqrt(x)>=0))*(np.abs(2*x-1.6)<=0.2)
                                    )
        u_exact = u_exact_lambda(x)
    elif solution_type == 2:
        u_exact = exact_solution_type_2(x,t)
    elif solution_type == 3:
        N = len(x)
        u_exact = np.zeros_like(x)
        mass = 0.5
        tc = 1.0
        solnL = np.zeros_like(u_exact)
        solnR = np.zeros_like(u_exact)
        if t<tc:
            # ***** before shock ***** #
            for i in range(N):
                u0=0.5+np.sin(x[i])
                u_exact[i] = find_root(u0,x[i],t)
        else:
            # ***** after shock ***** #
            for i in range(int(N)):
                if i==0:
                    solnL[i] = get_exact_solution_from_guess(x[i],0,t)
                    solnR[N-1-i] = get_exact_solution_from_guess(x[N-1-i],0,t)
                else:
                    solnL[i] = get_exact_solution_from_guess(x[i],solnL[i-1],t)
                    solnR[N-1-i] = get_exact_solution_from_guess(x[N-1-i],solnR[N-1-i+1],t)
                #
            #
            index=0
            aux=np.zeros(int(N)-int(N/2)) # assume shock is after x=pi
            for i in range(int(N/2),N):
                u_exact[:i] = solnL[:i]
                u_exact[i:] = solnR[i:]
                aux[index] = integrate(u_exact,x)
                index += 1
            #
            i_shock = np.argmin(np.abs(aux-mass)) + int(N/2)
            u_exact[:i_shock]=solnL[:i_shock]
            u_exact[i_shock:]=solnR[i_shock:]
        return u_exact
    elif solution_type == 4:
        # position of the shock
        xs = 0.25*np.sqrt(6)-0.15
        # end position of rarefaction
        xr = 0.35+1*0.5
        u_exact = (x>=xs)*(x<=xr)*((1-np.sqrt(3.0/8.0))/(xr-xs)*(x-xs)+np.sqrt(3.0/8.0))+1.0*(x>xr)
    #
    return u_exact
#

# ********************************************** #
# ***** APPLY PERIODIC BOUNDARY CONDITIONS ***** #
# ********************************************** #
def apply_bcs(q,nghost):
    q[:nghost] = q[-2*nghost:-nghost]
    q[-nghost:] = q[nghost:2*nghost]
#

# **************************** #
# ***** GET LOCAL BOUNDS ***** #
# **************************** #
def get_local_max(u,nghost):
    return Max(np.roll(u,-1), Max(u,np.roll(u,+1)))[nghost:-nghost]
#

def get_local_min(u,nghost):
    return Min(np.roll(u,-1), Min(u,np.roll(u,+1)))[nghost:-nghost]
#

# ***************************************** #
# ***** GET POLYNOMIAL RECONSTRUCTION ***** #
# ***************************************** #
def pw_poly_recon(q,nghost,order=5,weno_limiting=True):
    ql = np.zeros_like(q)
    qr = np.zeros_like(q)
    if weno_limiting:
        assert(order == 5)
        ql, qr = weno.weno5(q.reshape(1,len(q)),len(q)-2*nghost,nghost)
    elif order==1:
        ql[1:-1] = q[1:-1]
        qr[1:-1] = q[1:-1]
    elif order==3:
        ql[1:-1] = (2.*q[:-2] + 5.*q[1:-1] - q[2:])/6.
        qr[1:-1] = (-q[:-2] + 5.*q[1:-1] + 2.*q[2:])/6.
    elif order==5:
        ql[2:-2] = (-3.*q[:-4] + 27*q[1:-3] + 47*q[2:-2] - 13*q[3:-1] + 2*q[4:])/60.
        qr[2:-2] = (2.*q[:-4] - 13*q[1:-3] + 47*q[2:-2] + 27*q[3:-1] - 3*q[4:])/60.
    return ql.squeeze(), qr.squeeze()
#

# ************************ #
# ***** GET VELOCITY ***** #
# ************************ #
def get_velocity():
    return 1.0
#

# ************************* #
# ***** GET lambda_ij ***** #
# ************************* #
def get_lambda_max(solution_type, u, ul, ur, lambdaGmax=None):
    lmax_iph = np.zeros_like(u[1:-1])

    if solution_type in [0,1]:
        lmax_iph[:] = 1.0
    elif solution_type in [2,3]:
        if lambdaGmax is not None:
            lmax_iph[:] = lambdaGmax
        else:
            lmax_iph[:] = Max(Max(np.abs(u[1:-1]),np.abs(u[2:])),
                              Max(np.abs(ur[1:-1]),np.abs(ul[2:])))
    elif solution_type == 4:
        lmax_iph[:] = 1.0
    #
    return lmax_iph
#

# ****************** #
# ***** FLUXES ***** #
# ****************** #
def f_linear(q):
    v = get_velocity()
    return v*q
#

def f_burgers(q):
    return 0.5*q*q
#

def f_ge(q):
    return (0.25*q*(1-q))*(q<0.5) + (0.5*q*(q-1)+3.0/16)*(q>=0.5)
#

# ************************ #
# ***** FCT LIMITING ***** #
# ************************ #
def fct_limiting(flux,uLim,umin,umax,nghost,dx,dt,num_iter=1):
    # ***** Zalesak's FCT ***** #
    fstar_iph = np.zeros_like(flux)
    limited_flux_correction = np.zeros_like(flux[1:-1])
    for iter in range(num_iter):
        # Compute positive and negative fluxes #
        fPos = (flux[1:-1]>=0)*flux[1:-1] + (-flux[:-2]>=0)*(-flux[:-2])
        fNeg = (flux[1:-1]<0)*flux[1:-1]  + (-flux[:-2]<0)*(-flux[:-2])
        # Compute Rpos #
        QPos = dx/dt*(umax[1:-1]-uLim)
        fakeDen = fPos[:] + 1.0E15*(fPos[:]==0)
        ones = np.ones_like(QPos)
        Rpos = 1.0*(fPos[:]==0) + Min(ones, QPos/fakeDen)*(fPos[:]!=0)
        # Compute Rmin #
        QNeg = dx/dt*(umin[1:-1]-uLim)
        fakeDen = fNeg[:] + 1.0E15*(fNeg[:]==0)
        Rneg = 1.0*(fNeg[:]==0) + Min(ones, QNeg/fakeDen)*(fNeg[:]!=0) 
        # Compute limiters #
        LimR = (Min(Rpos,np.roll(Rneg,-1))*(flux[1:-1] >= 0) + 
                Min(Rneg,np.roll(Rpos,-1))*(flux[1:-1] < 0))
        LimL = (Min(Rpos,np.roll(Rneg,+1))*(-flux[:-2] >= 0) + 
                Min(Rneg,np.roll(Rpos,+1))*(-flux[:-2] < 0))
        # Apply the limiters #
        limiter_times_flux_correction = LimR*flux[1:-1]-LimL*flux[:-2]
        apply_bcs(limiter_times_flux_correction,1)
        # update output
        limited_flux_correction += limiter_times_flux_correction
        fstar_iph[1:-1] += LimR*flux[1:-1]
        apply_bcs(limited_flux_correction,1)
        apply_bcs(fstar_iph,nghost)
        # Update vectors for next iteration
        uLim += dt/dx * limiter_times_flux_correction
        apply_bcs(uLim,1)        
        # update flux for next iteration
        flux[1:-1] = flux[1:-1] - LimR*flux[1:-1]
        apply_bcs(flux,nghost)
    #
    return fstar_iph, limited_flux_correction

# *********************** #
# ***** MC LIMITING ***** #
# *********************** #
def mc_limiting(flux,uLim,dx,dt,dij,ubar_imh,ubar_iph,umin,umax,nghost):
    fstar_iph = np.zeros_like(flux)
    limited_flux_correction = np.zeros_like(flux[1:-1])
    # ***** Monolithic convex limiting ***** #
    fstar_iph[1:-1] = ( (flux[1:-1]>0)*Min(flux[1:-1], 
                                           2*dij[1:-1]*Min(umax[1:-1]-ubar_iph[1:-1],ubar_imh[2:]-umin[2:]))
                        +(flux[1:-1]<0)*Max(flux[1:-1],
                                            2*dij[1:-1]*Max(umin[1:-1]-ubar_iph[1:-1],ubar_imh[2:]-umax[2:])) 
                        )
    apply_bcs(fstar_iph,nghost)
    limited_flux_correction = fstar_iph[1:-1] - fstar_iph[:-2]
    uLim[:] += dt/dx * limited_flux_correction
    apply_bcs(uLim,1)
    return fstar_iph, limited_flux_correction
#

# ************************ #
# ***** GMC LIMITING ***** #
# ************************ #
def gmc_limiting(flux,uLim,dx,dt,dij,u,ubar_imh,ubar_iph,umin,umax,nghost,rkm_c=1.0,gamma=0.0):
    un = u
    # ***** Global monolithic convex limiting ***** #

    # create some data structures
    fstar_iph = np.zeros_like(flux)
    di = np.zeros_like(dij[1:-1])
    uBarL = np.zeros_like(di)

    # compute di
    di[1:-1] = 2*(dij[nghost:-nghost] + dij[nghost-1:-nghost-1])
    apply_bcs(di,1)
    
    # compute uBarL
    uBarL[1:-1] = 1/di[1:-1] * 2*(dij[nghost:-nghost]*ubar_iph[nghost:-nghost] +
                                  dij[nghost-1:-nghost-1]*ubar_imh[nghost:-nghost])
    apply_bcs(uBarL,1)
    check_bounds(uBarL,umin[1:-1],umax[1:-1],text='uBarL is out of bounds!')

    # Computte Q pos and neg
    QPos = rkm_c*(di*(umax[1:-1]-uBarL) + gamma*di*(umax[1:-1]-u[1:-1]))
    QNeg = rkm_c*(di*(umin[1:-1]-uBarL) + gamma*di*(umin[1:-1]-u[1:-1]))
    apply_bcs(QPos,1)
    apply_bcs(QNeg,1)
    
    # Compute positive and negative fluxes #
    fPos = (flux[1:-1]>=0)*flux[1:-1] + (-flux[:-2]>=0)*(-flux[:-2])
    fNeg = (flux[1:-1]<0)*flux[1:-1]  + (-flux[:-2]<0)*(-flux[:-2])
    apply_bcs(fPos,1)
    apply_bcs(fNeg,1)
        
    # Compute Rpos #
    fakeDen = fPos[:] + 1.0E15*(fPos[:]==0)
    ones = np.ones_like(QPos)
    Rpos = 1.0*(fPos[:]==0) + Min(ones, QPos/fakeDen)*(fPos[:]!=0)
    apply_bcs(Rpos,1)
        
    # Compute Rneg #
    fakeDen = fNeg[:] + 1.0E15*(fNeg[:]==0)
    Rneg = 1.0*(fNeg[:]==0) + Min(ones, QNeg/fakeDen)*(fNeg[:]!=0)
    apply_bcs(Rneg,1)
        
    # Compute limiters #
    LimR = (Min(Rpos,np.roll(Rneg,-1))*(flux[1:-1] >= 0) + 
            Min(Rneg,np.roll(Rpos,-1))*(flux[1:-1] < 0))
    LimL = (Min(Rpos,np.roll(Rneg,+1))*(-flux[:-2] >= 0) + 
            Min(Rneg,np.roll(Rpos,+1))*(-flux[:-2] < 0))
    apply_bcs(LimR,1)
    apply_bcs(LimL,1)
        
    # Apply the limiters #
    limiter_times_flux_correction = LimR*flux[1:-1]-LimL*flux[:-2]
    #limiter_times_flux_correction = flux[1:-1]-flux[:-2]
    uBarStar = uBarL + 1.0/(rkm_c*di) * limiter_times_flux_correction
    apply_bcs(uBarStar,1)

    # get output 
    fstar_iph[1:-1] += LimR*flux[1:-1]
    apply_bcs(fstar_iph,nghost)
    limited_flux_correction = fstar_iph[1:-1] - fstar_iph[:-2]    
    uLim[:] = u[1:-1] + rkm_c*dt/dx * di * (uBarStar-u[1:-1])
    
    # update vectors for the next iteration
    flux[1:-1] = flux[1:-1] - LimR*flux[1:-1]
    apply_bcs(flux,nghost)
    uBarL[:] = uBarStar
    apply_bcs(uBarL,1)
    if gamma==0:
        check_bounds(uBarL,umin[1:-1],umax[1:-1],text='uBarStar is out of bounds!')
    #
    return fstar_iph, limited_flux_correction
#

# ************************************************ #
# ***** SEMI-DISCRETIZATION VIA GMC LIMITERS ***** #
# ************************************************ #
def dudt_gmc(u,
             order=5,
             solution_type=0,
             weno_limiting=True,
             gamma=0.0,
             uMin=0.0,
             uMax=1.0):

    if solution_type in [0,1]:
        f = f_linear
    elif solution_type in [2,3]:
        f = f_burgers
    else:
        f = f_ge
    #

    # polynomial WENO reconstruction
    nghost = 2
    ul, ur = pw_poly_recon(u,nghost,order=order,weno_limiting=weno_limiting)
    apply_bcs(u,nghost)
    apply_bcs(ul,nghost)
    apply_bcs(ur,nghost)

    # compute dij 
    lmax_iph = np.zeros_like(u)
    lmax_iph[1:-1] = get_lambda_max(solution_type,u,ul,ur)
    apply_bcs(lmax_iph,nghost)
    dij = 0.5*lmax_iph

    #Compute bounds
    umax = np.zeros_like(u)
    umin = np.zeros_like(u)
    umax[:] = uMax
    umin[:] = uMin

    # Compute flux correction fij = HL-H
    H_L_iph = np.zeros_like(u)
    H_iph = np.zeros_like(u)
    H_L_iph[1:-1] = 0.5*(f(u[1:-1])+f(u[2:])) - dij[1:-1]*(u[2:] - u[1:-1])  
    H_iph[1:-1]   = 0.5*(f(ur[1:-1])+f(ul[2:])) - dij[1:-1]*(ul[2:]-ur[1:-1])
    flux = H_L_iph - H_iph

    # compute di
    di = np.zeros_like(dij[1:-1])
    di[1:-1] = 2*(dij[nghost:-nghost] + dij[nghost-1:-nghost-1])
    apply_bcs(di,1)

    # compute uBarL
    ubar_iph = np.zeros_like(u)
    ubar_imh = np.zeros_like(u)
    uBarL = np.zeros_like(u[1:-1])
    ubar_iph[1:-1] = 0.5*(u[1:-1] + u[2:]) - 0.5*(f(u[2:]) - f(u[1:-1]))/(2*dij[1:-1]+1E-15)
    ubar_imh[1:-1] = 0.5*(u[1:-1] + u[:-2]) + 0.5*(f(u[:-2]) - f(u[1:-1]))/(2*dij[:-2]+1E-15)
    uBarL[1:-1] = 1/di[1:-1] * 2*(dij[nghost:-nghost]*ubar_iph[nghost:-nghost] +
                                  dij[nghost-1:-nghost-1]*ubar_imh[nghost:-nghost])
    apply_bcs(uBarL,1)
    check_bounds(uBarL,umin[1:-1],umax[1:-1],text='uBarL is out of bounds!')

    # Computte Q pos and neg
    QPos = di*(umax[1:-1]-uBarL) + gamma*di*(umax[1:-1]-u[1:-1])
    QNeg = di*(umin[1:-1]-uBarL) + gamma*di*(umin[1:-1]-u[1:-1])
    apply_bcs(QPos,1)
    apply_bcs(QNeg,1)

    # Compute positive and negative fluxes #
    fPos = (flux[1:-1]>=0)*flux[1:-1] + (-flux[:-2]>=0)*(-flux[:-2])
    fNeg = (flux[1:-1]<0)*flux[1:-1]  + (-flux[:-2]<0)*(-flux[:-2])
    apply_bcs(fPos,1)
    apply_bcs(fNeg,1)
        
    # Compute Rpos #
    fakeDen = fPos[:] + 1.0E15*(fPos[:]==0)
    ones = np.ones_like(QPos)
    Rpos = 1.0*(fPos[:]==0) + Min(ones, QPos/fakeDen)*(fPos[:]!=0)
    apply_bcs(Rpos,1)
    
    # Compute Rneg #
    fakeDen = fNeg[:] + 1.0E15*(fNeg[:]==0)
    Rneg = 1.0*(fNeg[:]==0) + Min(ones, QNeg/fakeDen)*(fNeg[:]!=0)
    apply_bcs(Rneg,1)

    # Compute limiters #
    LimR = (Min(Rpos,np.roll(Rneg,-1))*(flux[1:-1] >= 0) + 
            Min(Rneg,np.roll(Rpos,-1))*(flux[1:-1] < 0))
    LimL = (Min(Rpos,np.roll(Rneg,+1))*(-flux[:-2] >= 0) + 
            Min(Rneg,np.roll(Rpos,+1))*(-flux[:-2] < 0))
    apply_bcs(LimR,1)
    apply_bcs(LimL,1)
    
    # Apply the limiters #
    limiter_times_flux_correction = LimR*flux[1:-1]-LimL*flux[:-2]
    uBarStar = uBarL + 1.0/di * limiter_times_flux_correction
    apply_bcs(uBarStar,1)

    return di * (uBarStar-u[1:-1])
#

# **************************************** #
# ***** SEMI-DISCRETIZATION VIA WENO ***** #
# **************************************** #
def dudt_weno(u,
              order=5,
              solution_type=0,
              weno_limiting=True):
   
    if solution_type in [0,1]:
        f = f_linear
    elif solution_type in [2,3]:
        f = f_burgers
    else:
        f = f_ge
    #

    # polynomial WENO reconstruction
    nghost = 2
    ul, ur = pw_poly_recon(u,nghost,order=order,weno_limiting=weno_limiting)
    apply_bcs(u,nghost)
    apply_bcs(ul,nghost)
    apply_bcs(ur,nghost)

    # compute dij 
    lmax_iph = np.zeros_like(u)
    lmax_iph[1:-1] = get_lambda_max(solution_type,u,ul,ur)
    apply_bcs(lmax_iph,nghost)
    dij = 0.5*lmax_iph
    
    # Compute weno flux 
    H_iph = np.zeros_like(u)
    H_iph[1:-1]   = 0.5*(f(ur[1:-1])+f(ul[2:])) - dij[1:-1]*(ul[2:]-ur[1:-1]) # LLF flux with high-order inputs

    return -(H_iph[1:-1] - H_iph[:-2])
#

# ************************************** #
# ***** GET SPATIAL DISCRETIZATION ***** #
# ************************************** #
def dudt(Q,
         Qn,
         order,
         dx,
         dt,
         solution_type=0,
         weno_limiting=True,
         gamma=0,
         bounds='global',
         uMin=0.0,
         uMax=1.0,
         dij=None,
         rkm_c=1.0,
         limit_space='None',
         slope_limiting=False):
    
    if solution_type in [0,1]:
        f = f_linear
    elif solution_type in [2,3]:
        f = f_burgers
    else:
        f = f_ge
    #

    # create data structures
    ubar_iph = np.zeros_like(Q)
    ubar_imh = np.zeros_like(Q)
    H_L_iph = np.zeros_like(Q)
    H_iph = np.zeros_like(Q)
    umax = np.zeros_like(Q)
    umin = np.zeros_like(Q)
    uDot = np.zeros_like(Q)
    fstar_iph = np.zeros_like(Q)
    u = Q.copy()

    # create poly reconstruction
    nghost = 2
    ul, ur = pw_poly_recon(u,nghost,order=order,weno_limiting=weno_limiting)
    apply_bcs(u,nghost)
    apply_bcs(ul,nghost)
    apply_bcs(ur,nghost)

    if bounds == 'local':
        umax[nghost:-nghost] = get_local_max(u,nghost)
        umin[nghost:-nghost] = get_local_min(u,nghost) 
    else:
        umax[:] = uMax + 0.0*dx
        umin[:] = uMin - 0.0*dx
    #

    if slope_limiting:
        # slope limiters
        ones = np.zeros_like(Q) + 1.0
        zeros = np.zeros_like(Q)
        QRef = Qn
        theta_l = ((ul-QRef>0) * Min((umax-QRef)/(ul-QRef),ones)
                   + (ul-QRef<0) * Min((umin-QRef)/(ul-QRef),ones)
                   + (ul-QRef==0) * 1.0)
        theta_r = ((ur-QRef>0) * Min((umax-QRef)/(ur-QRef),ones)
                   + (ur-QRef<0) * Min((umin-QRef)/(ur-QRef),ones)
                   + (ur-QRef==0) * 1.0)
        theta = Min(theta_l,theta_r)
        # check that slope limiters are between 0 and 1
        if (theta.max()>1.0+tol or theta.min()<0-tol):
            print ("theta is out of range: ",theta.min(),theta.max())
            sys.exit()
        else:
            theta = Min(Max(theta,zeros),ones)
        #
        ulsl = theta * (ul-QRef) + QRef
        ursl = theta * (ur-QRef) + QRef
        apply_bcs(ulsl,nghost)
        apply_bcs(ursl,nghost)
        check_bounds(ulsl,umin,umax,text='weno ul reconstruction is out of bounds')
        check_bounds(ursl,umin,umax,text='weno ur reconstruction is out of bounds')
        
        ul[:] = ulsl
        ur[:] = ursl
        
    # end of slope limiting 

    # get lambda max
    if dij is None:
        lmax_iph = np.zeros_like(Q)
        lmax_iph[1:-1] = get_lambda_max(solution_type,Q,ul,ur)
        apply_bcs(lmax_iph,nghost)
        dij = 0.5*lmax_iph
    #

    # For advection, these are just the upwind states:
    ubar_iph[1:-1] = 0.5*(u[1:-1] + u[2:]) - 0.5*(f(u[2:]) - f(u[1:-1]))/(2*dij[1:-1]+1E-15) # middle Riemann state from LLF
    ubar_imh[1:-1] = 0.5*(u[1:-1] + u[:-2]) + 0.5*(f(u[:-2]) - f(u[1:-1]))/(2*dij[:-2]+1E-15) # middle Riemann state from LLF
    H_L_iph[1:-1] = 0.5*(f(u[1:-1])+f(u[2:])) - dij[1:-1]*(u[2:] - u[1:-1])  # LLF flux with low-order inpust
    H_iph[1:-1]   = 0.5*(f(ur[1:-1])+f(ul[2:])) - dij[1:-1]*(ul[2:]-ur[1:-1]) # LLF flux with high-order inputs
    f_iph = H_L_iph - H_iph  # "correction" flux (with minus sign)
    #

    apply_bcs(umax,nghost)
    apply_bcs(umin,nghost)
    apply_bcs(f_iph,nghost)
    apply_bcs(H_iph,nghost)
    apply_bcs(H_L_iph,nghost)
    apply_bcs(ubar_iph,nghost)
    apply_bcs(ubar_imh,nghost)
    apply_bcs(u,nghost)

    # Compute low-order solution
    QL = Q[1:-1] - dt/dx * (H_L_iph[1:-1]-H_L_iph[:-2])
    apply_bcs(QL,1)

    # Check low-order solution in two different ways #
    # this is only for debugging
    if False:
        aux = Q[1:-1] + dt/dx * (2*dij[1:-1]*(ubar_iph[1:-1]-Q[1:-1]) + 2*dij[:-2]*(ubar_imh[1:-1]-Q[1:-1]))
        apply_bcs(aux,1)
        tmp = np.abs(aux-QL).min()
        if tmp>1E-15:
            print ("Low-order method is not the same when computed in the two different ways!")
            print (tmp)
            sys.exit()
    #############

    # For flux limiting #
    flux = np.copy(f_iph)
    spatial_flux_correction = np.zeros_like(flux[1:-1])
    num_iter = 1
    uLim = np.copy(QL)

    # FLUX LIMITING IN SPACE #
    if limit_space=='gmcl':
        fstar_iph[:], spatial_flux_correction[:] = gmc_limiting(flux,uLim,dx,dt,dij,Q,ubar_imh,ubar_iph,umin,umax,nghost,gamma=gamma)
    elif limit_space == 'mcl':
        fstar_iph[:], spatial_flux_correction[:] = mc_limiting(flux,uLim,dx,dt,dij,ubar_imh,ubar_iph,umin,umax,nghost)
    elif limit_space == 'fct':
        fstar_iph[:], spatial_flux_correction[:] = fct_limiting(flux,uLim,umin,umax,nghost,dx,dt,num_iter)
    #

    if limit_space != 'None':
        uDot[1:-1] = -1.0/dx * (H_L_iph[1:-1]-H_L_iph[:-2]) + 1.0/dx * spatial_flux_correction
    else:
        fstar_iph = f_iph
        uDot[1:-1] = -1.0/dx * (H_L_iph[1:-1]-H_L_iph[:-2] - (f_iph[1:-1] - f_iph[:-2]))
    #
    apply_bcs(uDot,nghost)
    return uDot, H_L_iph-fstar_iph, H_L_iph, dij, ubar_iph, ubar_imh, umax, umin
#

# ************************ #
# ***** CHECK BOUNDS ***** #
# ************************ #
def check_bounds(u,umin,umax,text=None):
    # upper bound 
    upper_bound = np.min(umax-u)<-tol
    lower_bound = np.min(u-umin)<-tol
    if upper_bound:
        print ("upper bound violated")
        print ("value, argument: ", np.min(umax-u), np.argmin(umax-u))
        if text is not None:
            print (text)
        sys.exit()
    if lower_bound:
        print ("lower bound violated")
        print ("value, argument: ", np.min(u-umin), np.argmin(u-umin))
        if text is not None:
            print (text)
        sys.exit()
    #
    # Clean round off errors. 
    # This is clipping, but only if the tol is fulfilled
    u[:] = Min(Max(umin[:],u[:]),umax[:])
#

# ************************* #
# ***** COMPUTE ERROR ***** #
# ************************* #
def compute_L1_error(q,x,dx,u_exact):
    # polynomial reconstruction (at mid point of the cells) #
    # Based on a fifth order polynomial reconstruction evaluated at the mid point of the cells
    # See the Mathematica file poly_rec.nb for details
    um = np.zeros_like(q)
    um = (9*q[:-4] - 116*q[1:-3] + 2134*q[2:-2] - 116*q[3:-1] + 9*q[4:])/1920.0
    mid_value_error = dx*np.sum(np.abs(um - u_exact))

    return mid_value_error
#

# *********************************** #
# ***** RUN SEMI-DISCRETIZATION ***** #
# *********************************** #
def test_semidiscretization(bounds='global',
                            order=5,
                            gamma=0,
                            m=100,
                            slope_limiting=False,
                            linearity=1, # nonlinear flux
                            weno_limiting=True):
    assert bounds in ['local', 'global']

    nghost = 2
    xlower = 0.0
    xupper = 1.0
    dx = (xupper-xlower)/(m)   # Size of 1 grid cell

    x = np.linspace(xlower-(2*nghost-1)*dx/2,xupper+(2*nghost-1)*dx/2,m+2*nghost)

    #####################
    # Initial condition #
    #####################
    # Note: the solution_type is hard coded to be 0
    u_init, uMin, uMax = get_init_condition(0)
    # NOTE: the initial condition must be given as cell averages of the exact solution
    from scipy.integrate import quad    
    Q = np.array([1/dx*quad(u_init,x[i]-dx/2.,x[i]+dx/2.)[0] for i in range(len(x))])
    apply_bcs(Q,nghost)

    f_gmc = dudt_gmc(Q,
                     order=order,
                     solution_type=2, #hard coded to use Burgers flux
                     weno_limiting=weno_limiting,
                     gamma=gamma,
                     uMin=0.0,
                     uMax=1.0)
    f_weno = dudt_weno(Q,
                       order=order,
                       solution_type=2,
                       weno_limiting=weno_limiting)
    # exact
    if linearity==0:
        F = -(np.exp(-100*(x+dx/2.0-0.5)**2) - np.exp(-100*(x-dx/2.0-0.5)**2))
    else:
        F = -0.5*(np.exp(-100*(x+dx/2.0-0.5)**2)**2 - np.exp(-100*(x-dx/2.0-0.5)**2)**2)
    #
    # compute errors
    E_gmc  = dx*np.sum(np.abs(F[nghost:-nghost]-f_gmc[1:-1]))
    E_weno = dx*np.sum(np.abs(F[nghost:-nghost]-f_weno[1:-1]))

    return (E_weno,E_gmc)
#

# ************************************** #
# ***** RUN TIME DEPENDENT PROBLEM ***** #
# ************************************** #
def test_advection(T=1,
                   bounds='global',
                   low_order=False,
                   limit_space='None',
                   limit_time='None',
                   stagewise_BP=None,
                   num_iter=1,
                   order=5,
                   gamma=0,
                   nu=0.5,
                   RKM='RK76',
                   m=100,
                   solution_type=0,
                   verbosity=True,
                   name_plot=None,
                   plot_exact_soln=False,
                   name_file=None,
                   weno_limiting=True):
    assert bounds in ['local', 'global']
    assert limit_space in ['fct', 'gmcl', 'mcl', 'None']
    assert limit_time in ['fct', 'gmcl', 'mcl', 'None']
    if stagewise_BP is not None:
        assert stagewise_BP in ['RK_stagewise_GMCL', 'RK_stagewise_FCT', 'clipped_stagewise_BP']
    #
    assert solution_type in [0,1,2,3,4]

    nghost = 2
    xlower = 0.0
    xupper = 1.0
    if solution_type == 3:
        xupper = 2*np.pi
    elif solution_type == 4:
        xupper = 2.0
        xlower = -1.0
    dx = (xupper-xlower)/(m)   # Size of 1 grid cell

    x = np.linspace(xlower-(2*nghost-1)*dx/2,xupper+(2*nghost-1)*dx/2,m+2*nghost)
    t = 0.      # Initial time

    dt = nu * dx  # Time step

    #####################
    # Initial condition #
    #####################
    u_init, uMin, uMax = get_init_condition(solution_type)
    # NOTE: the initial condition must be given as cell averages of the exact solution
    from scipy.integrate import quad    
    Q = np.array([1/dx*quad(u_init,x[i]-dx/2.,x[i]+dx/2.)[0] for i in range(len(x))])
    init_mass = dx*np.sum(Q[nghost:-nghost])
    apply_bcs(Q,nghost)

    ##################################
    # Define time integration scheme #
    ##################################
    if RKM == 'EE':
        rkm = rk.extrap(5)
    elif RKM == 'RK76':
        A=np.array([[0,0,0,0,0,0,0],
                    [1./3,0,0,0,0,0,0],
                    [0,2./3,0,0,0,0,0],
                    [1./12,1./3,-1./12,0,0,0,0],
                    [-1./16,18./16,-3./16,-6./16,0,0,0],
                    [0,9./8,-3./8,-6./8,4./8,0,0],
                    [9./44,-36./44,63./44,72./44,-64./44,0,0]])
        b=np.array([11./120,0,81./120,81./120,-32./120,-32./120,11./120])
        rkm = rk.ExplicitRungeKuttaMethod(A,b)
    else:
        rkm = rk.loadRKM(RKM)
    rkm = rkm.__num__()
    #import pdb; pdb.set_trace()

    t = 0. # current time
    b = rkm.b
    s = len(rkm)
    y = np.zeros((s, np.size(Q))) # stage values
    G = np.zeros((s, np.size(Q))) # stage derivatives
    H = np.zeros((s, np.size(Q))) # stage fluxes

    QMIN =  1E10
    QMAX = -1E10

    #############
    # Time loop #
    #############
    while t < T and not np.isclose(t, T):
        if t + dt > T:
            nu = nu*(T-t)/dt
            dt = T - t
        #

        Qn = np.copy(Q)
        ##########################
        # SPATIAL DISCRETIZATION #
        ##########################
        for i in range(s):
            y[i,:] = Q.copy()
            RK_flux = np.zeros_like(Q)
            for j in range(i):
                y[i,:] += rkm.A[i,j]*dt*G[j,:]
                RK_flux[:] += rkm.A[i,j]*H[j,:]
            if i==0:
                G[i,:], H[i,:], HL, dij, ubar_iph, ubar_imh, umax, umin = dudt(y[i,:],
                                                                               Qn,
                                                                               order,
                                                                               dx,
                                                                               dt,
                                                                               solution_type=solution_type,
                                                                               weno_limiting=weno_limiting, 
                                                                               gamma=gamma,
                                                                               bounds=bounds,
                                                                               uMin=uMin,
                                                                               uMax=uMax,
                                                                               dij=None,
                                                                               limit_space=limit_space)
            else:
                apply_bcs(y[i,:],nghost)
                # for stagewise BP limiting #
                if stagewise_BP == 'RK_stagewise_GMCL':
                    apply_bcs(RK_flux,nghost)
                    yLim = np.zeros_like(y[j,:])
                    gmc_limiting(rkm.c[i]*HL-RK_flux,
                                 yLim[1:-1],
                                 dx,dt,dij,Q,ubar_imh,ubar_iph,umin,umax,nghost,
                                 gamma=gamma,
                                 rkm_c=rkm.c[i])
                    apply_bcs(yLim,nghost)
                    y[i,:] = yLim
                    check_bounds(y[i,:],umin,umax,text='yLim intermediate solution is out of bounds!')
                elif stagewise_BP == 'RK_stagewise_FCT':
                    apply_bcs(RK_flux,nghost)
                    yLim = np.zeros_like(y[j,:])
                    yLim[1:-1] = Q[1:-1] - rkm.c[i]*dt/dx*(HL[1:-1]-HL[:-2])
                    apply_bcs(yLim,nghost)
                    fct_limiting(rkm.c[i]*HL-RK_flux,yLim[1:-1],umin,umax,nghost,dx,dt,num_iter=num_iter)
                    apply_bcs(yLim,nghost)
                    y[i,:] = yLim
                    check_bounds(y[i,:],umin,umax,text='yLim intermediate solution is out of bounds!')
                elif stagewise_BP == 'clipped_stagewise_BP':
                    y[i,:] = Max(y[i,:],umin)
                    y[i,:] = Min(y[i,:],umax)
                    check_bounds(y[i,:],umin,umax)
                #
                G[i,:], H[i,:], _, _, _, _, _, _ = dudt(y[i,:],
                                                        Qn,
                                                        order,
                                                        dx,
                                                        dt,
                                                        solution_type=solution_type,
                                                        weno_limiting=weno_limiting, 
                                                        gamma=gamma,
                                                        bounds=bounds,
                                                        uMin=uMin,
                                                        uMax=uMax,
                                                        dij=dij,
                                                        limit_space=limit_space)
            #
        # Get the flux correction in space and time
        fdot = HL[:] - sum([rkm.b[j]*H[j,:] for j in range(s)])

        ######################################
        # COMPUTE LOW-ORDER SOLUTION IN TIME #
        ######################################
        # low-order (in time) solution; i.e., forward euler
        uLim = Q[1:-1] - dt/dx*(HL[1:-1]-HL[:-2])
        apply_bcs(uLim,1)
        
        #########################
        # FLUX LIMITING IN TIME #
        #########################
        flux = np.copy(fdot)

        num_iter=1
        if low_order==False:
            if limit_time=='gmcl':
                _,limited_correction = gmc_limiting(flux,uLim,dx,dt,dij,Q,ubar_imh,ubar_iph,umin,umax,nghost,gamma=gamma)
            elif limit_time=='mcl':
                _,limited_correction = mc_limiting(flux,uLim,dx,dt,dij,ubar_imh,ubar_iph,umin,umax,nghost)
            elif limit_time=='fct':
                _,limited_correction = fct_limiting(flux,uLim,umin,umax,nghost,dx,dt,num_iter=num_iter)
            else: # no limiting in time
                uLim += dt/dx*(flux[1:-1]-flux[:-2])
            #

        ###################
        # Update solution #
        ###################
        Q[1:-1] = uLim 
        apply_bcs(Q,nghost)
        
        ################
        # CHECK BOUNDS #
        ################
        if limit_time != 'None' or low_order==True:
            check_bounds(uLim,umin[1:-1],umax[1:-1])

        # check conservation of mass
        mass = dx * np.sum(Q[nghost:-nghost])
        if (np.abs(init_mass-mass)>1E-12):
            print ("Loss in mass: ", init_mass-mass)

        ###############
        # update time #
        ###############
        t += dt
        QMIN = min(QMIN,Q[nghost:-nghost].min())
        QMAX = max(QMAX,Q[nghost:-nghost].max())
    #

    ##################
    # Exact solution #
    ##################
    u_exact = get_exact_solution(solution_type,x,t)
    if solution_type==4:
        xI_arg = (1000*(x<0)+np.abs(x)).argmin()
        xI = x[xI_arg]
        xE_arg = (1000*(x>1.0) + np.abs(x-1.0)).argmin()+1
        xE = x[xE_arg]
    #
    if solution_type==4:
        Q_tmp = np.zeros_like(Q[xI_arg-nghost:xE_arg+nghost])
        x_tmp = np.zeros_like(Q[xI_arg-nghost:xE_arg+nghost])
        Q_tmp[nghost:-nghost] = Q[xI_arg:xE_arg]
        x_tmp[nghost:-nghost] = x[xI_arg:xE_arg]
        apply_bcs(Q_tmp,nghost)
        apply_bcs(x,nghost)
        average_error = compute_L1_error(Q_tmp,x_tmp,dx,u_exact[xI_arg:xE_arg])
    else:
        average_error = compute_L1_error(Q,x,dx,u_exact[nghost:-nghost])

    ############
    # Plotting #
    ############
    #plt.plot(x[nghost:-nghost],Q[nghost:-nghost],'-r',lw=3)
    plt.plot(x[nghost:-nghost],Q[nghost:-nghost],lw=3)
    #plt.plot(x[nghost:-nghost],Q[nghost:-nghost],'.r',lw=3,mew=4, ms=4)
    if plot_exact_soln:
        plt.plot(x[nghost:-nghost],u_exact[nghost:-nghost],'--k',alpha=0.5,lw=3)
        if solution_type == 4:            
            plt.xlim([0.,1])
        #
    #

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if name_plot is None:
        plt.savefig('plot.png')
    else:
        plt.savefig(name_plot+'.png')
    #

    if verbosity:
        print('min(Q), max(Q) at EOS: ', np.min(Q[nghost:-nghost]),np.max(Q[nghost:-nghost]))
        print('min(Q), max(Q) during entire sim: ', QMIN, QMAX)

        print ("error via cell averages: ", average_error)
    #

    if name_file is not None:
        a= np.zeros((len(x[nghost:-nghost]),2))
        a[:,0] = x[nghost:-nghost]
        a[:,1] = Q[nghost:-nghost]
        np.savetxt(name_file+".csv", a, delimiter=",")

    return average_error, QMIN, QMAX
