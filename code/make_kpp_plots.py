import numpy as np
from matplotlib import pyplot as plt

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

m_values=[300,600,1200,2400,4800]
def make_kpp_plots():
    # ************************* #
    # ***** non-WENO-ExRK ***** #
    # ************************* #
    plt.clf()
    for m in m_values:
        data = np.genfromtxt("non-WENO-ExRK_m"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,lw=3)
    #

    # plot exact solution #
    # position of the shock
    x=np.linspace(0,1,1000)
    xs = 0.25*np.sqrt(6)-0.15
    # end position of rarefaction
    xr = 0.35+1*0.5
    u_exact = (x>=xs)*(x<=xr)*((1-np.sqrt(3.0/8.0))/(xr-xs)*(x-xs)+np.sqrt(3.0/8.0))+1.0*(x>xr)
    plt.plot(x,u_exact,'--k',alpha=0.5,lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([0,1])
    # INSET #
    ax1 = plt.gca()
    ax2 = plt.axes([0,0,1,1])
    ip= InsetPosition(ax1, [0.1,0.2,0.25,0.75])
    x1, x2, y1, y2 = 0.425, 0.525, 0.35, 0.8
    ax2.set_xlim(x1, x2)
    ax2.set_ylim(y1, y2)
    ax2.set_axes_locator(ip)
    mark_inset(ax1, ax2, loc1=4, loc2=1, fc="none", ec='0.75',lw=2)
    for m in m_values: 
        data = np.genfromtxt("non-WENO-ExRK_m"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        ax2.plot(x,u,lw=3)
        x=np.linspace(0,1,1000)
        plt.plot(x,u_exact,'--',color=[0.5,0.5,0.5],alpha=0.5,lw=3)
    #
    ax2.set_xticks([])
    ax2.set_yticks([])
    # save figure
    plt.savefig('non-WENO-ExRK.png')
    plt.clf()

    # ********************* #
    # ***** WENO-ExRK ***** #
    # ********************* #
    for m in m_values:
        data = np.genfromtxt("WENO-ExRK_m"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,lw=3)
    #

    # plot exact solution #
    # position of the shock
    x=np.linspace(0,1,1000)
    xs = 0.25*np.sqrt(6)-0.15
    # end position of rarefaction
    xr = 0.35+1*0.5
    u_exact = (x>=xs)*(x<=xr)*((1-np.sqrt(3.0/8.0))/(xr-xs)*(x-xs)+np.sqrt(3.0/8.0))+1.0*(x>xr)
    plt.plot(x,u_exact,'--k',alpha=0.5,lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([0,1])
    # INSET #
    ax1 = plt.gca()
    ax2 = plt.axes([0,0,1,1])
    ip= InsetPosition(ax1, [0.1,0.2,0.25,0.75])
    x1, x2, y1, y2 = 0.425, 0.525, 0.35, 0.8
    ax2.set_xlim(x1, x2)
    ax2.set_ylim(y1, y2)
    ax2.set_axes_locator(ip)
    mark_inset(ax1, ax2, loc1=4, loc2=1, fc="none", ec='0.75',lw=2)
    for m in m_values: 
        data = np.genfromtxt("WENO-ExRK_m"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        ax2.plot(x,u,lw=3)
        x=np.linspace(0,1,1000)
        plt.plot(x,u_exact,'--',color=[0.5,0.5,0.5],alpha=0.5,lw=3)
    #
    ax2.set_xticks([])
    ax2.set_yticks([])
    # save figure
    plt.savefig('WENO-ExRK.png')
    plt.clf()

    # ********************* #
    # ***** SSP54-GMC ***** #
    # ********************* #
    for m in m_values:
        data = np.genfromtxt("SSP54-GMC_m"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,lw=3)
    #

    # plot exact solution #
    # position of the shock
    x=np.linspace(0,1,1000)
    xs = 0.25*np.sqrt(6)-0.15
    # end position of rarefaction
    xr = 0.35+1*0.5
    u_exact = (x>=xs)*(x<=xr)*((1-np.sqrt(3.0/8.0))/(xr-xs)*(x-xs)+np.sqrt(3.0/8.0))+1.0*(x>xr)
    plt.plot(x,u_exact,'--k',alpha=0.5,lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([0,1])
    # INSET #
    ax1 = plt.gca()
    ax2 = plt.axes([0,0,1,1])
    ip= InsetPosition(ax1, [0.1,0.2,0.25,0.75])
    x1, x2, y1, y2 = 0.425, 0.525, 0.35, 0.8
    ax2.set_xlim(x1, x2)
    ax2.set_ylim(y1, y2)
    ax2.set_axes_locator(ip)
    mark_inset(ax1, ax2, loc1=4, loc2=1, fc="none", ec='0.75',lw=2)
    for m in m_values: 
        data = np.genfromtxt("SSP54-GMC_m"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        ax2.plot(x,u,lw=3)
        x=np.linspace(0,1,1000)
        plt.plot(x,u_exact,'--',color=[0.5,0.5,0.5],alpha=0.5,lw=3)
    #
    ax2.set_xticks([])
    ax2.set_yticks([])
    # save figure
    plt.savefig('SSP54-GMC.png')
    plt.clf()

    # ******************** #
    # ***** GMC-ExRK ***** #
    # ******************** #
    for m in m_values:
        data = np.genfromtxt("GMC-ExRK_m"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        plt.plot(x,u,lw=3)
    #

    # plot exact solution #
    # position of the shock
    x=np.linspace(0,1,1000)
    xs = 0.25*np.sqrt(6)-0.15
    # end position of rarefaction
    xr = 0.35+1*0.5
    u_exact = (x>=xs)*(x<=xr)*((1-np.sqrt(3.0/8.0))/(xr-xs)*(x-xs)+np.sqrt(3.0/8.0))+1.0*(x>xr)
    plt.plot(x,u_exact,'--k',alpha=0.5,lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim([0,1])
    # INSET #
    ax1 = plt.gca()
    ax2 = plt.axes([0,0,1,1])
    ip= InsetPosition(ax1, [0.1,0.2,0.25,0.75])
    x1, x2, y1, y2 = 0.425, 0.525, 0.35, 0.8
    ax2.set_xlim(x1, x2)
    ax2.set_ylim(y1, y2)
    ax2.set_axes_locator(ip)
    mark_inset(ax1, ax2, loc1=4, loc2=1, fc="none", ec='0.75',lw=2)
    for m in m_values: 
        data = np.genfromtxt("GMC-ExRK_m"+str(m)+".csv",delimiter=',')
        x = data[:,0]
        u = data[:,1]
        ax2.plot(x,u,lw=3)
        x=np.linspace(0,1,1000)
        plt.plot(x,u_exact,'--',color=[0.5,0.5,0.5],alpha=0.5,lw=3)
    #
    ax2.set_xticks([])
    ax2.set_yticks([])
    # save figure
    plt.savefig('GMC-ExRK.png')
    plt.clf()
