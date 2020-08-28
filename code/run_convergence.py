import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from time_flux_limiting import *

# convergence to smooth solutions using SSP54, 
def run_convergence_to_smooth_solution(solution_type,
                                       m_values,
                                       RKM,
                                       gamma=0,
                                       limit_space='None',
                                       limit_time='None',
                                       stagewise_BP=None):
    print ("")
    index=0
    # baseline method (no limiting)
    for m in m_values:
        error,QMIN,QMAX = test_advection(T=final_time,
                                         bounds='global',
                                         stagewise_BP=stagewise_BP,
                                         # *** LIMITER IN SPACE *** #
                                         limit_space=limit_space,
                                         # *** LIMITER IN TIME *** #
                                         limit_time=limit_time,
                                         order=order,
                                         nu=0.4/(1+gamma),
                                         RKM=RKM,
                                         gamma=gamma,
                                         m=m,
                                         solution_type=solution_type,
                                         verbosity=False,
                                         plot_exact_soln=True if m==m_values[len(m_values)-1] else False,
                                         weno_limiting=True)
        errors[index] = error
        table[index,0]=m
        table[index,1]=error
        if index>0:
            table[index,2]=np.log(error/errors[index-1])/np.log(0.5)
        table[index,3]=min(QMIN-lower_bound, upper_bound-QMAX)
        index += 1
        print (index, ' out of ', len(m_values))
    #
    print(tabulate(table,
                   headers=['m', 'error', 'rate', 'delta'],
                   floatfmt='.2e'))
#

# ********** SOLUTION TYPE ********** #
# 0: linear advection with smooth data
# 1: linear advection with non-smooth data
# 2: burgers equation v1
# 3: burgers equation v2
# 4: one-dimensional KPP

# ******************************************************** #
# ********** CONVERGENCE OF SEMI-DISCRETIZATION ********** #
# ******************************************************** #
if False:
    m_values = np.array([25,50,100,200,400,800,1600])
    errors = np.zeros_like(m_values,'d')
    table = np.zeros((len(m_values),4))
    order = 5 # 1,3,5

    index=0
    linearity=1
    print ("")
    print ("********** Linearity type: ", linearity)
    for gamma in [0, 0.5, 1.0]:
        index=0
        errors_weno = np.zeros_like(m_values,'d')
        errors_gmc = np.zeros_like(m_values,'d')
        table = np.zeros((len(m_values),5))
        for m in m_values:
            E_weno, E_gmc = test_semidiscretization(order=order,
                                                    m=m,
                                                    weno_limiting=True,
                                                    gamma=gamma, 
                                                    slope_limiting=False,
                                                    linearity=linearity)
            errors_weno[index] = E_weno
            errors_gmc[index] = E_gmc
            table[index,0]=m
            table[index,1]=E_weno
            if index>0:
                table[index,2]=np.log(errors_weno[index]/errors_weno[index-1])/np.log(0.5)
            table[index,3]=E_gmc
            if index>0:
                table[index,4]=np.log(errors_gmc[index]/errors_gmc[index-1])/np.log(0.5)    
            index += 1
        #
        print ("GMC gamma value: ", gamma)
        print(tabulate(table,
                       headers=['m', 'WENO error', 'rate', 'GMC error', 'rate'],
                       floatfmt='.2e'))
    #
#

# ***************************************************** #
# ********** CONVERGENCE TO SMOOTH SOLUTIONS ********** #
# ***************************************************** #
solution_type=0

if False:
    # Select methods 
    SSP54=True
    ExERK5=True
    RK76=True
    SwRK76=True
    
    assert solution_type in [0,2,3]
    if solution_type == 0:
        lower_bound = 0.0
        upper_bound = 1.0
        final_time = 1.0
    elif solution_type==2:
        lower_bound = -1.0
        upper_bound = 1.0
        final_time = 0.1
    elif solution_type==3:
        lower_bound = -0.5
        upper_bound = 1.5
        final_time = 0.5
    #

    m_values = np.array([25,50,100,200,400,800])
    errors = np.zeros_like(m_values,'d')
    table = np.zeros((len(m_values),4))
    order = 5 # 1,3,5

    if SSP54:
        print ("")
        print ("********** CONVERGENCE WITH SSP **********")
        run_convergence_to_smooth_solution(solution_type,m_values,'SSP54',limit_space='None',limit_time='None',gamma=0.0)
        run_convergence_to_smooth_solution(solution_type,m_values,'SSP54',limit_space='gmcl',limit_time='None',gamma=0.0)
        run_convergence_to_smooth_solution(solution_type,m_values,'SSP54',limit_space='gmcl',limit_time='None',gamma=1.0)
    #

    if ExERK5:
        print ("")
        print ("********** CONVERGENCE WITH EXE-RK5 **********")
        run_convergence_to_smooth_solution(solution_type,m_values,'EE',limit_space='None',limit_time='None',gamma=0.0)
        run_convergence_to_smooth_solution(solution_type,m_values,'EE',limit_space='gmcl',limit_time='gmcl',gamma=0.0)
        run_convergence_to_smooth_solution(solution_type,m_values,'EE',limit_space='gmcl',limit_time='gmcl',gamma=1.0)
        run_convergence_to_smooth_solution(solution_type,m_values,'EE',limit_space='gmcl',limit_time='gmcl',gamma=2.0)
    #

    if RK76:
        print ("")
        print ("********** CONVERGENCE WITH RK76 **********")
        run_convergence_to_smooth_solution(solution_type,m_values,'RK76',limit_space='None',limit_time='None',gamma=0.0)
        run_convergence_to_smooth_solution(solution_type,m_values,'RK76',limit_space='None',limit_time='gmcl',gamma=0.0)
        run_convergence_to_smooth_solution(solution_type,m_values,'RK76',limit_space='None',limit_time='gmcl',gamma=1.0)
    #

    if SwRK76:
        print ("")
        print ("********** CONVERGENCE WITH Sw-RK76-GMC **********")
        run_convergence_to_smooth_solution(solution_type,m_values,'RK76',limit_space='None',limit_time='gmcl',gamma=0.0,stagewise_BP='RK_stagewise_GMCL')
        run_convergence_to_smooth_solution(solution_type,m_values,'RK76',limit_space='None',limit_time='gmcl',gamma=1.0,stagewise_BP='RK_stagewise_GMCL')
    #
#

# *************************************************************** #
# ********** LINEAR ADVECTION WITH NON-SMOOTH SOLUTION ********** #
# *************************************************************** #
if False:
    plt.clf()
    gamma=1.0
    order=5
    lower_bound = 0.0
    upper_bound = 1.0
    final_time = 1.0
    # SSP54
    names = [
        # SSP54
        'lin_adv_nsmooth_ssp_t1',
        'lin_adv_nsmooth_ssp_t100',
        'lin_adv_nsmooth_ssp_gmc_t1',
        'lin_adv_nsmooth_ssp_gmc_t100',
        # RK76
        'lin_adv_nsmooth_rk_t1',
        'lin_adv_nsmooth_rk_t100',
        'lin_adv_nsmooth_rk_gmc_t1',
        'lin_adv_nsmooth_rk_gmc_t100']
    limiters_space = ['None','None','gmcl','gmcl','None','None','None','None']
    limiters_time  = ['None','None','None','None','None','None','gmcl','gmcl']
    RKMethods = ['SSP54','SSP54','SSP54','SSP54','RK76','RK76','RK76','RK76']
    times = [1,100,1,100,1,100,1,100]
    for i, name in enumerate(names):        
        print ("**********")
        print (name)
        error,QMIN,QMAX = test_advection(T=times[i],
                                         bounds='global',
                                         # *** LIMITER IN SPACE *** #
                                         limit_space=limiters_space[i],
                                         # *** LIMITER IN TIME *** #
                                         limit_time=limiters_time[i],
                                         order=order,
                                         gamma=gamma,
                                         nu=0.4/(1+gamma),
                                         RKM=RKMethods[i],
                                         m=200,
                                         solution_type=1,
                                         verbosity=False,
                                         name_file=name,
                                         plot_exact_soln=True,
                                         weno_limiting=True)
        print ("delta: ", min(QMIN-lower_bound,upper_bound-QMAX))
    #
    # DO PLOTS #
    arg_sqrt = lambda x: 1-((2*x-1.6)/0.2)**2        
    ex = lambda x: (np.exp(-300*(2*x-0.3)**2)*(np.abs(2*x-0.3)<=0.25)
                    +1.0*(np.abs(2*x-0.9)<=0.2)
                    +np.sqrt(arg_sqrt(x)*(arg_sqrt(x)>=0))*(np.abs(2*x-1.6)<=0.2))
    # *** SSP54 *** #
    # T=1
    plt.clf()
    ssp = np.genfromtxt('lin_adv_nsmooth_ssp_t1.csv',delimiter=',')
    ssp_gmc = np.genfromtxt('lin_adv_nsmooth_ssp_gmc_t1.csv',delimiter=',')    
    plt.plot(ssp[:,0],ex(ssp[:,0]),'--k',alpha=0.75,lw=3)
    plt.plot(ssp[:,0],ssp[:,1],'-b',lw=3)
    plt.plot(ssp_gmc[:,0],ssp_gmc[:,1],'--r',lw=2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('lin_adv_nsmooth_ssp_gmc_t1.png')
    # T=100
    plt.clf()
    ssp = np.genfromtxt('lin_adv_nsmooth_ssp_t100.csv',delimiter=',')
    ssp_gmc = np.genfromtxt('lin_adv_nsmooth_ssp_gmc_t100.csv',delimiter=',')    
    plt.plot(ssp[:,0],ex(ssp[:,0]),'--k',alpha=0.75,lw=3)
    plt.plot(ssp[:,0],ssp[:,1],'-b',lw=3)
    plt.plot(ssp_gmc[:,0],ssp_gmc[:,1],'--r',lw=2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('lin_adv_nsmooth_ssp_gmc_t100.png')

    # *** RK76 *** #
    # T=1
    plt.clf()
    rk = np.genfromtxt('lin_adv_nsmooth_rk_t1.csv',delimiter=',')
    rk_gmc = np.genfromtxt('lin_adv_nsmooth_rk_gmc_t1.csv',delimiter=',')    
    plt.plot(rk[:,0],ex(rk[:,0]),'--k',alpha=0.75,lw=3)
    plt.plot(rk[:,0],rk[:,1],'-b',lw=3)
    plt.plot(rk_gmc[:,0],rk_gmc[:,1],'--r',lw=2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('lin_adv_nsmooth_rk_gmc_t1.png')
    # T=100
    plt.clf()
    rk = np.genfromtxt('lin_adv_nsmooth_rk_t100.csv',delimiter=',')
    rk_gmc = np.genfromtxt('lin_adv_nsmooth_rk_gmc_t100.csv',delimiter=',')    
    plt.plot(rk[:,0],ex(rk[:,0]),'--k',alpha=0.75,lw=3)
    plt.plot(rk[:,0],rk[:,1],'-b',lw=3)
    plt.plot(rk_gmc[:,0],rk_gmc[:,1],'--r',lw=2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('lin_adv_nsmooth_rk_gmc_t100.png')
#

# *************************************************** #
# ********** NONLINEAR BURGERS AFTER SHOCK ********** #
# *************************************************** #
if False:
    plt.clf()
    lower_bound = -0.5
    upper_bound = 1.5
    gamma=1.0
    order=5
    names = ['nonlin_ssp',
             'nonlin_ssp_gmc',
             'nonlin_rk',
             'nonlin_rk_gmc']
    limiters_space = ['None','gmcl','None','None']
    limiters_time  = ['None','None','None','gmcl']
    RKMethods = ['SSP54','SSP54','RK76','RK76']
    for i, name in enumerate(names):
        print ("**********")
        print (name)
        error,QMIN,QMAX = test_advection(T=2,
                                         bounds='global',
                                         # *** LIMITER IN SPACE *** #
                                         limit_space=limiters_space[i],
                                         # *** LIMITER IN TIME *** #
                                         limit_time=limiters_time[i],
                                         order=order,
                                         gamma=gamma,
                                         nu=0.4/(1+gamma),
                                         RKM=RKMethods[i],
                                         m=100,
                                         solution_type=3,
                                         verbosity=False,
                                         name_file=name,
                                         plot_exact_soln=True,
                                         weno_limiting=True)
        print ("delta: ", min(QMIN-lower_bound,upper_bound-QMAX))
    #
    # DO PLOTS #
    plt.clf()
    from time_flux_limiting import get_exact_solution
    # *** SSP54 *** #
    ssp = np.genfromtxt('nonlin_ssp.csv',delimiter=',')
    ssp_gmc = np.genfromtxt('nonlin_ssp_gmc.csv',delimiter=',')    
    u_exact = get_exact_solution(3,ssp[:,0],2)
    plt.plot(ssp[:,0],u_exact,'--k',alpha=0.75,lw=3)
    plt.plot(ssp[:,0],ssp[:,1],'-b',lw=3)
    plt.plot(ssp_gmc[:,0],ssp_gmc[:,1],'--r',lw=2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('nonlin_ssp_gmc.png')

    # *** SSP54 *** #
    plt.clf()
    rk = np.genfromtxt('nonlin_rk.csv',delimiter=',')
    rk_gmc = np.genfromtxt('nonlin_rk_gmc.csv',delimiter=',')    
    u_exact = get_exact_solution(3,rk[:,0],2)
    plt.plot(rk[:,0],u_exact,'--k',alpha=0.75,lw=3)
    plt.plot(rk[:,0],rk[:,1],'-b',lw=3)
    plt.plot(rk_gmc[:,0],rk_gmc[:,1],'--r',lw=2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('nonlin_rk_gmc.png')

# ******************************************* #
# ********** NONLINEAR ONE DIM KPP ********** #
# ******************************************* #
if False:
    plt.clf()
    names = ['non-WENO-ExRK_m', 'WENO-ExRK_m','SSP54-GMC_m','GMC-ExRK_m']
    limiters_space=['None','None','gmcl','None']
    limiters_time=['None','None','None','gmcl']
    RKMethods = ['RK76','RK76','SSP54','RK76']
    weno = [False,True,True,True]

    lower_bound = 0.0
    upper_bound = 1.0
    m_values = np.array([100,200,400,800,1600])
    errors = np.zeros_like(m_values,'d')
    table = np.zeros((len(m_values),4))
    gamma=1.0
    order=5
    for i, name in enumerate(names):
        index=0
        for m in m_values:
            error,QMIN,QMAX = test_advection(T=1,
                                             bounds='global',
                                             # *** LIMITER IN SPACE *** #
                                             limit_space=limiters_space[i],
                                             # *** LIMITER IN TIME *** #
                                             limit_time=limiters_time[i],
                                             order=order,
                                             gamma=gamma,
                                             nu=0.4/(1+gamma),
                                             RKM=RKMethods[i],
                                             m=m,
                                             solution_type=4,
                                             verbosity=False,
                                             name_file=name+str(3*m),
                                             plot_exact_soln= m==m_values[-1],
                                             weno_limiting=weno[i])
            errors[index] = error
            table[index,0]=m
            table[index,1]=error
            if index>0:
                table[index,2]=np.log(error/errors[index-1])/np.log(0.5)
            table[index,3]=min(QMIN-lower_bound, upper_bound-QMAX)
        
            index += 1
            print (index, ' out of ', len(m_values))
        #
        print(tabulate(table,
                       headers=['m', 'error', 'rate', 'delta'],
                       floatfmt='.2e'))
    # MAKE PLOTS 
    plt.clf()
    from make_kpp_plots import make_kpp_plots
    make_kpp_plots()
#
