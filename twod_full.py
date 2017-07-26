"""
Full 2D neural field model

all evaluations on NxN matrix.

convolutions performed on 0,2pi x 0,2pi domain. plotted on -pi,pi x -pi,pi domain.

notes:
-

todo: 
-include methods to get and view slices, steady-state bumps

"""



import numpy as np
np.random.seed(0)

import matplotlib
#matplotlib.use("Agg")
#matplotlib.use("GTKAgg")

# for dynamic print updating
from sys import stdout
import sys
import getopt

#import twod_phase
import matplotlib.pylab as mp
import os
#import scipy as sp
import scipy as sp
from scipy.ndimage.measurements import center_of_mass
from scipy.interpolate import interp1d
import time
#from colorsys import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
import copy
import math


from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{bm} \usepackage{xcolor} \setlength{\parindent}{0pt}']
# anim
import matplotlib.pyplot as plt


import fourier_2d as f2d
from euler import ESolve



sin = np.sin
cos = np.cos
pi = np.pi
sqrt = np.sqrt
exp = np.exp

periodization_lower = -5
periodization_upper = 5


def usage():
    print "-l, --use-last\t\t: use last data from last sim"
    print "-v, --save-last\t\t: save last data of current sim"
    print "-s, --use-ss\t\t: use last saved steady-state data"
    print "-e, --save-ss\t\t: save solution as steady-state data"
    print "-r, --use-random\t: use random inits"
    print "-h, --help\t\t: help function"
    print "-p, --run-phase\t\t: run phase"
    print "-f, --run-full\t\t: run full"

def roll(Z,N):
    """
    roll by N/2
    """
    return np.roll(np.roll(Z,N/2,axis=1),N/2,axis=0)

def shift(Z,x,y):
    """
    shift surface Z by coordinates x,y
    """
    N,N = Z.shape
    Nx = int(N*x/(2*pi))
    Ny = int(N*y/(2*pi))
    return np.roll(np.roll(Z,Nx,axis=1),Ny,axis=0)

def plot_s(ax,Z,alpha=0.8):
    """
    define plotting function with new defaults
    """

    N,N = Z.shape
    x = np.linspace(-pi,pi,N)
    X,Y = np.meshgrid(x,x)


    ax.plot_surface(X,Y,Z,
                    rstride=2,
                    edgecolors="k",
                    cstride=2,
                    cmap="gray",
                    alpha=alpha,
                    linewidth=0.25)
    #ax.view_init(azim=30)

    ax.set_xlabel(r'$\bm{x}$')
    ax.set_ylabel(r'$\bm{y}$')

    ax.set_xlim(-pi,pi)
    ax.set_ylim(-pi,pi)
    ax.set_xticks(np.arange(-1,1+.5,.5)*pi)
    x_label = [r"$\bm{-\pi$}", r"$\bm{-\frac{\pi}{2}}$", r"$\bm{0}$", r"$\bm{\frac{\pi}{2}}$",   r"$\bm{\pi}$"]
    ax.set_xticklabels(x_label, fontsize=15)

    ax.set_yticks(np.arange(-1,1+.5,.5)*pi)
    y_label = [r"$\bm{-\pi$}", r"$\bm{-\frac{\pi}{2}}$", r"$\bm{0}$", r"$\bm{\frac{\pi}{2}}$",   r"$\bm{\pi}$"]
    ax.set_yticklabels(y_label, fontsize=15)

    return ax
    

class Sim(object):
    """
    general simulation parameters.
    grid size N. N_idx=N**2
    filenames
    """
    def __init__(self,
                 N=64,a=-pi,b=pi,
                 break_symm=False,
                 r=15.,ut=0.25,
                 mu=1.,
                 save_ss_bump=True):
        self.N = N
        self.K = N+1
        self.N_idx = self.N**2
        self.K_idx = self.K**2
        self.a = a # lower domain boundary
        self.b = b # upper domain boundary

        self.r = r
        self.ut = ut
        self.mu = mu
        self.X_double = np.linspace(self.a,self.b,2*N)
        self.XX_double,self.YY_double = np.meshgrid(self.X_double,self.X_double)

        self.X_quad = np.linspace(self.a,self.b,4*N)
        self.XX_quad,self.YY_quad = np.meshgrid(self.X_quad,self.X_quad)

        self.X = np.linspace(self.a,self.b,N)
        self.XX,self.YY = np.meshgrid(self.X,self.X)
        
        self.dir1 = 'bump_ss_2d_full'


class Kernel(Sim):
    """
    build kernel lookup table
    K: domain size
    X2: domain linspace
    se,si: kernel parameters 
    N_idx: N*N domain size
    Kamp: kernel amplitude
    pos: kernel position (keep at (0,0))
    plot_kernel: plot pinning function
    recompute_kernel: force function to rebuild pinning file
    """

    def __init__(self,se=2.,si=3.,ce=1.,Kamp=10.,
                 periodization_lower=-6,
                 periodization_upper=6,
                 plot_kernel=False,
                 recompute_kernel=False,
                 kernel_type='diff_gauss'):
        # fat diff_gause: se=2,si=3
        """
        kernel_type: 'cosine', 'diff_gauss', 'gauss-c', 'finite-.1'
        """
        
        Sim.__init__(self)
        
        self.se = se
        self.si = si
        self.ce = ce
        self.Kamp = Kamp

        self.periodization_lower = periodization_lower
        self.periodization_upper = periodization_upper
        self.plot_kernel = plot_kernel
        self.recompute_kernel = recompute_kernel

        self.kernel_type = kernel_type

        self.ker_dir2 = 'amp='+str(self.Kamp)+\
                        '_N='+str(self.N)+\
                        '_K='+str(self.K)+\
                        '_sige='+str(self.se)+\
                        '_sigi='+str(self.si)+\
                        '_kernel_type='+self.kernel_type

        self.kernelfile = self.dir1+'/'+self.ker_dir2+'.ker'

        self.X2 = np.linspace(self.a,self.b,self.K)
        self.XX2,self.YY2 = np.meshgrid(self.X2,self.X2)

        if (not os.path.exists(self.dir1+'/'+self.ker_dir2)):
            os.makedirs(self.dir1+'/'+self.ker_dir2)

        # load kernel if it exists. Else, build it.
        if os.path.isfile(self.kernelfile) and not(self.recompute_kernel):
            #print '* kernel file found. loading...'
            self.Ktable = np.loadtxt(self.kernelfile)
            #print '  ... done.'
        else:
            print '* kernel file not found.'
            self.Ktable = np.zeros((self.K,self.K))

            if self.kernel_type == 'diff_gauss':
                for i in range(self.K):
                    stdout.write("\r  ... building... %d%%" % int((100.*(i+1)/self.K)))
                    stdout.flush()
                    for j in range(self.K):
                        self.Ktable[i,j] = self.K_diff_p(self.X2[i],self.X2[j])

            elif self.kernel_type == 'gauss-c':
                for i in range(self.K):
                    stdout.write("\r  ... building... %d%%" % int((100.*(i+1)/self.K)))
                    stdout.flush()
                    for j in range(self.K):
                        self.Ktable[i,j] = self.K_gauss_p(self.X2[i],self.X2[j])-.1

            elif self.kernel_type == 'finite-.1':
                for i in range(self.K):
                    stdout.write("\r  ... building... %d%%" % int((100.*(i+1)/self.K)))
                    stdout.flush()
                    for j in range(self.K):
                        self.Ktable[i,j] = self.K_finite_p(self.X2[i],self.X2[j])-.1
                        #self.Ktable[i,j] = self.K_finite(self.X2[i],self.X2[j])-.1
                    
            elif self.kernel_type == 'cos':
                for i in range(self.K):
                    stdout.write("\r  ... building... %d%%" % int((100.*(i+1)/self.K)))
                    stdout.flush()
                    for j in range(self.K):
                        self.Ktable[i,j] = .275*self.K_cos(np.sqrt(self.X2[i]**2+self.X2[j]**2))
            else:
                raise ValueError('invalid kernel choice, '+str(self.kernel_type))
            print
            self.Ktable *= self.Kamp*((2*pi)**2)/self.N_idx
            print '* saving kernel...'
            np.savetxt(self.kernelfile,self.Ktable)
            print '  ... done.'

        #print '* getting kernel fft...'
        #Ktablefft = np.fft.fft2(Ktable,(N,N))
        #self.Ktablefft = np.fft.fft2(self.Ktable,s=(self.N,self.N))
        self.Ktablefft = np.fft.fft2(self.Ktable,s=(self.N,self.N))
        
        #print '  ... done.'
        
    def K_cos(self,x):
        return cos(x/self.ce)

    def K_cos_p(self,X,Y):
        """
        periodized kernel using cosine
        """
        tot = 0
        for n in np.arange(self.periodization_lower,self.periodization_upper+1,1):
            for m in np.arange(self.periodization_lower,self.periodization_upper+1,1):
                tot = tot + self.K_cos(sqrt( (X+n*2*pi)**2 + (Y+m*2*pi)**2))
        return tot

    def K_diff(self,x):
        """
        difference of gaussians
        """
        A=1./(sqrt(pi)*self.se);B=1./(sqrt(pi)*self.si)
        return (A*exp(-(x/self.se)**2) -
                B*exp(-(x/self.si)**2))
        
    def K_diff_p(self,X,Y):
        """
        periodized kernel using difference of gaussians
        """
        tot = 0
        for n in np.arange(self.periodization_lower,self.periodization_upper+1,1):
            for m in np.arange(self.periodization_lower,self.periodization_upper+1,1):
                tot = tot + self.K_diff(sqrt( (X+n*2*pi)**2 + (Y+m*2*pi)**2))
        return tot

    def K_finite(self,x):
        """
        exponential
        """
        if np.abs(x) < 1:
            return np.exp(-1/(1-x**2))
        else:
            return 0.

    def K_finite_p(self,X,Y):
        tot = 0
        for n in np.arange(self.periodization_lower,self.periodization_upper+1,1):
            for m in np.arange(self.periodization_lower,self.periodization_upper+1,1):
                tot = tot + self.K_finite(sqrt( (X+n*2*pi)**2 + (Y+m*2*pi)**2))
        return tot        
        
    def K_gauss(self,x):
        A=1./(sqrt(pi)*self.se)
        return A*exp(-(x/self.se)**2)

    def K_gauss_p(self,X,Y):
        """
        periodized kernel using difference of gaussians
        """
        tot = 0
        for n in np.arange(self.periodization_lower,self.periodization_upper+1,1):
            for m in np.arange(self.periodization_lower,self.periodization_upper+1,1):
                tot = tot + self.K_gauss(sqrt( (X+n*2*pi)**2 + (Y+m*2*pi)**2))
        return tot

    def plot(self):
        """
        plot kernel
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #ax.set_zlim(-5,10)
        ax.set_title("Kernel")
        ax = plot_s(ax,self.Ktable)

        return fig

class SteadyState(Kernel):
    """
    ss bump solution as a class.
    assuming no heterogeneity, centered at origin.
    """
    def __init__(self,
                 use_ss=True,
                 recompute_ss=False,
                 display_params=False,
                 break_symm=False,
                 save_ss_bump=True,
                 use_kernel=False,
                 ss_ushift1=0.,ss_ushift2=0.,
                 ss_zshift1=0.,ss_zshift2=0.,
                 ss_g=0.,ss_q=0.,
                 ss_dt=.1,ss_t0=0,ss_T=100,
                 ss_eps=0.005
             ):
        # defaults
        
        Sim.__init__(self)
        Kernel.__init__(self)
        
        
        self.use_kernel = use_kernel

        self.ss_ushift1 = ss_ushift1
        self.ss_ushift2 = ss_ushift2
        self.ss_zshift1 = ss_zshift1
        self.ss_zshift2 = ss_zshift2
        
        self.display_params = display_params
        self.ss_g = ss_g
        self.ss_q = ss_q
        self.ss_dt = ss_dt
        self.ss_t0 = ss_t0
        self.ss_T = ss_T
        self.ss_TN = int(self.ss_T/self.ss_dt)
        self.ss_eps = ss_eps

        self.ss_dir2 = 'amp='+str(self.Kamp)+\
                       '_N='+str(self.N)+\
                        '_K='+str(self.K)+\
                        '_sige='+str(self.se)+\
                        '_sigi='+str(self.si)+\
                        '_r='+str(self.r)+\
                        '_ut='+str(self.ut)+\
                        '_mu='+str(self.mu)+\
                        '_q='+str(self.ss_q)+\
                        '_g='+str(self.ss_g)+\
                        'N='+str(self.N)+\
                        '_dt='+str(self.ss_dt)+\
                        '_T='+str(self.ss_T)+\
                        '_eps='+str(self.ss_eps)

        self.ss_u_f = 'ss_bump_u.dat'
        self.ss_z_f = 'ss_bump_z.dat'

        self.ss_u_file = self.dir1+'/'+self.ss_dir2+'/'+self.ss_u_f
        self.ss_z_file = self.dir1+'/'+self.ss_dir2+'/'+self.ss_z_f

        if (not os.path.exists(self.dir1+'/'+self.ss_dir2)):
            os.makedirs(self.dir1+'/'+self.ss_dir2)


        self.recompute_ss = recompute_ss
        self.use_ss = use_ss
        self.save_ss_bump = save_ss_bump

        if self.ss_g > 0 and self.break_symm:
            self.break_val = (np.random.randn(N_idx)-.5)
            self.break_val2 = (np.random.randn(N_idx)-.5)
        else:
            self.break_val = 0
            self.break_val2 = 0

        #self.u0d,self.z0d = self.uz0d()

        if self.recompute_ss:
            self.u0ss,self.z0ss = self.u0b(self.Ktablefft,verbose=True)
        elif self.use_ss:
            if os.path.isfile(self.ss_u_file) and os.path.isfile(self.ss_z_file):
                self.u0ss = np.loadtxt(self.ss_u_file) + self.break_val
                self.z0ss = np.loadtxt(self.ss_z_file) + self.break_val2

            else: # if use_last not found, use the steady-state as init.
                self.u0ss,self.z0ss = self.u0b(self.Ktablefft)

        
        self.uy,self.ux = np.gradient(self.u0ss)

        self.df_u0b = f(self.u0ss,self.r,self.ut,d=True)

    def uz0d(self,verbose=False,nontrivial_z=False):
        """
        default initial conditions for u,z if no steady-state bump is found,
        or if a different initial condition is desired.
        -later implement other initial conditions
        """
        if verbose:
            print '\tbuilding initial condition'

        N = len(self.X)
        u0 = np.zeros((self.N,self.N))
        z0 = np.zeros((self.N,self.N))
        for i in range(self.N):
            for j in range(self.N):
                u0[i,j] = self.K_diff_p(self.X[i]-self.ss_ushift1,self.X[j]-self.ss_ushift2)
                if nontrivial_z:
                    z0[i,j] = self.K_diff_p(self.X[i]-self.ss_ushift1,self.X[j]-self.ss_ushift2)
        if verbose:
            print '\t... done'

        return u0*self.Kamp,z0*self.Kamp

    
    def u0b(self,Ktablefft,
            d=False,verbose=False):
        """
        if build_u0b, then use params to compute a steady-state bump by simulating the ODE
        if not build_u0b,check to see if there is a steady-state bump file and use it.
        if there is no steady-state bump file, build a steady-state bump.
        
        ss_u_file, ss_z_file: file and directory names
        d: gradient
        Ktablefft: circular FFT of kernel lookup table
        N: domain discretization
        params: tuple of parameters (mu,Kamp,se,si,r,ut,eps,q,g)
        """
        print '* building ss bump...'
    

        if verbose:
            print "SS BUMP WITH PARAMS"
            print "mu="+str(self.mu)+"; Kamp="+str(self.Kamp)+"; se="+str(self.se)+"; si="+str(self.si)
            print "r="+str(self.r)+"; ut="+str(self.ut)+"; eps="+str(self.ss_eps)+"; q="+str(self.ss_q)+"; g="+str(self.ss_g)
            print "save_ss_bump="+str(self.save_ss_bump)
        # domain
        
        z0d = np.zeros((self.N,self.N))
        init_for_ss = np.zeros((2,self.N,self.N))
        u0d,z0d = self.uz0d()
        init_for_ss[0,:,:] = u0d
        init_for_ss[1,:,:] = z0d
        
        
        sol = np.zeros((self.ss_TN,2,self.N,self.N))
        sol[0,:,:,:] = init_for_ss
    
        for i in range(1,self.ss_TN):

            sol[i,:,:,:] = sol[i-1,:,:,:] + self.ss_dt*rhs(sol[i-1,:,:,:],self.ss_dt*i,
                                                        self.ss_eps,self.mu,self.Ktablefft,0.,
                                                        self.ss_g,self.ss_q,self.r,self.ut,self.N)
            #print i, 'of', self.ss_TN
        #plot_surface_movie(X,sol,np.linspace(0,T,TN),1,movie=False,title='ss bump computation')
        print '  ... done.'
    
        if self.save_ss_bump:
            u0ss = sol[-1,0,:,:]
            z0ss = sol[-1,1,:,:]

            np.savetxt(self.ss_u_file,u0ss)
            np.savetxt(self.ss_z_file,z0ss)

        if d:
            # return gradient dy, dx
            uy,ux = np.gradient(u0ss)
            return u0ss,z0ss,ux,uy
        else:
            return u0ss,z0ss

    def plot(self,option='u0b'):
        """
        options: u0b_init,u0b,u0b_grad_x,u0b_grad_y
        """

        if option == "u0b_init":
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            #ax.set_zlim(-5,10)
            ax.set_title("steady-state bump init")
            ax = plot_s(ax,self.u0d)

        elif option == "u0b":
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            #ax.set_zlim(-5,10)
            ax.set_title("steady-state bump")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("u")

            ax = plot_s(ax,self.u0ss)

        elif option == "u0b_grad_x":

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            
            ax.set_title("du/dx")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("u")
            ax = plot_s(ax,self.ux)

        elif option == "u0b_grad_y":
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            
            ax.set_title("du/dy")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_zlabel(r"$\frac{\partial u}{\partial y}$")
            ax = plot_s(ax,self.uy)


        
        return fig


class SimDat(SteadyState):
    """
    simulation data
    """
    def __init__(self,g=0,q=0,t0=0,T=100,dt=.1,eps=0.005,
                 ushift1=0.,ushift2=0.,
                 zshift1=0.,zshift2=0.,
                 ishift1=0.,ishift2=0.,
                 pinning_function='ss',
                 initial_function='ss',
                 kernel_type='diff_gauss',

                 recompute_I=False,
                 recompute_ss=False,
                 recompute_kernel=False,
                 recompute_lc=False,
                 recompute_fq=False,

                 fq=False,
                 display_params=True,
                 use_last=False,
                 save_last=False,
                 use_random=False,
                 use_ss=False,
                 rand_small_pert=False
    ):

        """
        g: adaptation strength param
        q: pinning strength param
        t0: starting time
        T: ending time
        dt: time step
        eps: small epsilon.
        save_lc: True/False. Check if limit cycle exists in forward time. If so, save. Else, print no LC found.
        *shift: shift initial solution in the domain
        recompute*: recompute the data file for said function/data set.
        """

        Sim.__init__(self)
        Kernel.__init__(self,recompute_kernel=recompute_kernel,
                        kernel_type=kernel_type)
        SteadyState.__init__(self,recompute_ss=recompute_ss)


        self.use_last=use_last
        self.save_last = save_last
        self.use_random=use_random
        self.use_ss = use_ss
        
        self.recompute_I = recompute_I
        self.recompute_lc = recompute_lc
        self.recompute_fq = recompute_fq
        self.display_params = display_params

        self.rand_small_pert = rand_small_pert
        
        self.eps = eps
        self.g = g
        self.q = q
        self.dt = .1
        self.t0 = t0
        self.T = T
        self.TN = int(self.T/self.dt)
        self.t = np.linspace(self.t0,self.T+self.dt,self.TN)
        
        self.ushift1 = ushift1
        self.ushift2 = ushift2
        self.zshift1 = zshift1
        self.zshift2 = zshift2
        
        self.initial_function = initial_function
        self.pinning_function = pinning_function

        self.ishift1 = ishift1
        self.ishift2 = ishift2
        
        # convert the shift coordinates to indices
        self.diag_shiftx_z = int(self.N*(self.zshift1)/(2*pi))
        self.diag_shifty_z = int(self.N*(self.zshift2)/(2*pi))
        
        self.diag_shiftx_u = int(self.N*(self.ushift1)/(2*pi))
        self.diag_shifty_u = int(self.N*(self.ushift2)/(2*pi))

        # subdirectory for last data point (defunct?)
        self.lastdir = 'amp='+str(self.Kamp)+\
               '_N='+str(self.N)+\
                '_K='+str(self.K)+\
                '_sige='+str(self.se)+\
                '_sigi='+str(self.si)+\
                '_r='+str(self.r)+\
                '_ut='+str(self.ut)+\
                '_mu='+str(self.mu)+\
                '_q='+str(self.q)+\
                '_g='+str(self.g)+\
                'N='+str(self.N)+\
                '_dt='+str(self.dt)+\
                '_T='+str(self.T)+\
                '_eps='+str(self.eps)

        # subdirectory for LC and fq multipliers
        self.lcdir = self.dir1+'/amp='+str(self.Kamp)+\
                     '_N='+str(self.N)+\
                    '_K='+str(self.K)+\
                    '_sige='+str(self.se)+\
                    '_sigi='+str(self.si)+\
                    '_r='+str(self.r)+\
                    '_ut='+str(self.ut)+\
                    '_mu='+str(self.mu)+\
                    '_q='+str(self.q)+\
                    '_g='+str(self.g)+\
                    'N='+str(self.N)+\
                    '_dt='+str(self.dt)+\
                    '_eps='+str(self.eps)
        
        

        #self.savedir = self.dir1+'/'+self.lastdir+'/'
        self.savedir = self.dir1+'/'

        if (not os.path.exists(self.lcdir)):
            os.makedirs(self.lcdir)
        if (not os.path.exists(self.savedir)):
            os.makedirs(self.savedir)

        # filenames for solutions u,z, limit cycle in phase.
        self.filename_u = self.savedir+'u.dat'
        self.filename_z = self.savedir+'z.dat'
        self.filename_lc_th = self.lcdir+'/'+'lc.dat'

        #print self.filename_u

        if self.display_params:
            print "simulation parameters:"
            print "T = ",self.T
            print "mu =",self.mu,"; Kamp =",self.Kamp,"; se =",self.se
            print "si =",self.si,"; r =",self.r,"; ut =",self.ut
            print "N =",self.N,"; K =",self.K,"; N_idx =",self.N_idx

            print "eps =",self.eps,"; q =",self.q,"; g =",self.g
            print "ushift1 =",self.ushift1,"; ushift2 =",self.ushift2
            print "zshift1 =",self.zshift1,"; zshift2 =",self.zshift2
            print 'ishift1 =',self.ishift1,"; isfhit2 =",self.ishift2

        # load self.Itable
        self.load_pinning()
            
        # returns u[TN,N,N], z[TN,N,N], sol[TN,2,N,N]
        self.run_full_sim()

        if self.save_last:
            #print self.filename_u
            np.savetxt(self.filename_u,self.u[-1,:,:])
            np.savetxt(self.filename_z,self.z[-1,:,:])

        # get bump peak
        #self.x_locs,self.y_locs = self.get_peak(self.sol)
        self.x_locs,self.y_locs = self.get_center_mass_time(self.sol)
        self.th1 = self.x_locs # rename for convenience
        self.th2 = self.y_locs
        
        # put fq stuff here if needed 
        #self.load_limit_cycle() # gives access to self.lc_data.
        # organized by [time|th1|th2] over 1 period of LC if it exists.

        #self.load_fq()
        

        ####################################################################################
        ####################################################################################
        ### END INIT #######################################################################
        ####################################################################################
        ####################################################################################

    def run_full_sim(self):
        """
        run the full sim
        """
        self.init = np.zeros((2,self.N,self.N))
        #self.init[0,:,:] = self.u0ss + self.break_val


        file_not_found = False
        while True:

            if self.use_last and not(file_not_found):
                if os.path.isfile(self.filename_u) and\
                   os.path.isfile(self.filename_z):
                    print 'using last'
                    self.init[0,:,:] = np.loadtxt(self.filename_u)
                    self.init[1,:,:] = np.loadtxt(self.filename_z)
                    break
                else:
                    print 'init file not found'
                    file_not_found = True
            else:
                print 'using initial function = '+str(self.initial_function)
                if self.initial_function == 'ss':
                    self.init[0,:,:] = shift(self.u0ss,self.ushift1,self.ushift2)
                    self.init[1,:,:] = shift(self.z0ss,self.zshift1,self.zshift2)


                elif self.initial_function == 'gaussian_z0':
                    self.init[0,:,:] = shift(self.K_gauss_p(self.XX,self.YY),self.ushift1,self.ushift2)
                    
                elif self.initial_function == 'rand':
                    self.init[0,:,:] = np.random.randn(self.N,self.N)
                    self.init[1,:,:] = np.random.randn(self.N,self.N)
                    
                elif self.initial_function == 'ss_z0': # zero z but use ss bump
                    self.init[0,:,:] = shift(self.u0ss,self.ushift1,self.ushift2)
                    
                elif self.initial_function == 'rand_z0': # zero z but use random u
                    self.init[0,:,:] = np.random.randn(self.N,self.N)
                    
                else:
                    raise ValueError('no initial function'+str(self.initial_function))
                break


        
        self.sol = np.zeros((self.TN,2,self.N,self.N))

        self.sol[0,:,:,:] = self.init

        if self.rand_small_pert:
            self.sol[0,:,:,:] += np.random.randn(2,self.N,self.N)*.1

        print '* simulating...'
        # run Euler scheme
        for i in range(1,self.TN):
            self.sol[i,:,:,:] = self.sol[i-1,:,:,:] + self.dt*rhs(self.sol[i-1,:,:,:],self.dt*i,
                                                                  self.eps,self.mu,self.Ktablefft,self.Itable,
                                                                  self.g,self.q,self.r,self.ut,self.N)
        self.u = self.sol[:,0,:,:]
        self.z = self.sol[:,1,:,:]
        print '  ...done.'

    def load_limit_cycle(self):
        """
        if lc data exists, load. if DNE or recompute required, compute here.
        """
        file_not_found = False


        while True:
            if self.recompute_lc or file_not_found:
                """
                force recomputation of LC
                """
                self.compute_limit_cycle() # contains self.lc_data
                np.savetxt(self.filename_lc_th,self.lc_data)
                break

            else:
                if os.path.isfile(self.filename_lc_th):

                    lc_data = np.loadtxt(self.filename_lc_th)

                    self.lc_t = lc_data[:,0]
                    self.lc_th1 = lc_data[:,1]
                    self.lc_th2 = lc_data[:,2]
                    # check to see if file contains lc or not.
                    # non-lc parameter files have [-1,-1] as the data.

                    break
                else:
                    file_not_found = True

    def compute_limit_cycle(self):
        """
        if lc not found, or if recomputation requested, compute LC.

        algorithm:
        1. use existing data. if there are enough crossings, skip to 2. if there are not enough crossings detected, re-run with more time (print time). if there are enough crossings, skip to 2. else, quit.
        2. given that there are enough crossings, check periodicity by using the last period estimate and check if the solution comes back to the start (up to some tolerance, print this). if the tolerance check fails, quit. else go to 3.
        3. if a limit cycle exists, save the limit cycle solution data with a filename containing all parameter info in the format array=[time|theta1|theta2] (i.e. to plot theta1 over time i would use plot([array[:,0],array[:,1])).
        
        """
        
        tol = .01

        # first try finding crossings with current solution data.

        temp_th1 = copy.deepcopy(self.th1)
        temp_th2 = copy.deepcopy(self.th2)

        find_crossings_iter = 0 # count # of times attempted to find enough LC crossings
        max_find_crossings_iter = 1
        crossings_exist = True # assume true to start

        temp_TN = self.TN
        temp_sol = self.sol
        temp_t = self.t



        # step 1 use existing data.
        while True:
            # find ccw crossings on right
            crossing_idx_ccw = (temp_th1[1:]>0)*(temp_th2[1:]>0)*(temp_th2[:-1]<=0)
            crossing_idx_cw = (temp_th1[1:]>0)*(temp_th2[1:]<=0)*(temp_th2[:-1]>0)

            cross_fail = 0



            # check number of crossings in each direction
            if np.sum(crossing_idx_ccw) <= 5:
                print 'not enough crossings in ccw direction ('+str(np.sum(crossing_idx_ccw))+')'
                cross_fail += 1
            else:
                print 'enough candidate crossings found in ccw direction ('+str(np.sum(crossing_idx_ccw))+')'
                crossing_idx = crossing_idx_ccw
                break # break to leave loop and go to step 2

            if np.sum(crossing_idx_cw) <= 5:
                print 'not enough crossings in cw direction ('+str(np.sum(crossing_idx_cw))+')'
                cross_fail += 1
            else:
                print 'enough candidate crossings found in ccw direction ('+str(np.sum(crossing_idx_cw))+')'
                crossing_idx = crossing_idx_cw
                break # break to leave loop and go to step 2

            if find_crossings_iter >= max_find_crossings_iter:
                # if there was a limit cycle, it would have been detected in the 2nd pass above.
                # give up if limit cycle not found in 2nd pass.
                crossings_exist = False # gloabl var
                self.limit_cycle_exists = False # global var
                print 'no limit cycle found at step 1.', find_crossings_iter

                # save dummy file.
                break

            if cross_fail == 2 and (find_crossings_iter < max_find_crossings_iter):
                # if both crossing checks fail in step 1, run sim for longer
                # this should not run in the second pass (when find_crossings_iter >= 1)
                temp_T = 10000
                print 'not enough crossings. Re-initializing with additional time T='+str(temp_T)

                temp_TN = int(temp_T/self.dt)
                temp_sol = np.zeros((self.TN+temp_TN,2,self.N,self.N))
                
                temp_t = np.zeros(self.TN+temp_TN)

                temp_sol[:self.TN,:,:,:] = self.sol
                temp_t = np.linspace(self.t0,self.T+temp_T+self.dt,self.TN+temp_TN)

                print '* simulating...'
                # run Euler scheme
                for i in range(self.TN-1,self.TN+temp_TN):
                    temp_sol[i,:,:,:] = temp_sol[i-1,:,:,:] + self.dt*rhs(temp_sol[i-1,:,:,:],
                                                                          self.dt*i,
                                                                          self.eps,
                                                                          self.mu,
                                                                          self.Ktablefft,
                                                                          self.Itable,
                                                                          self.g,
                                                                          self.q,
                                                                          self.r,
                                                                          self.ut,
                                                                          self.N)

                temp_u = temp_sol[:,0,:,:]
                temp_z = temp_sol[:,1,:,:]

                temp_x_locs,temp_y_locs = self.get_center_mass_time(temp_sol)
                temp_th1 = temp_x_locs # rename for convenience
                temp_th2 = temp_y_locs

                find_crossings_iter += 1 # add 1 to number of longer sims run

        # step 2 check periodicity.
        if crossings_exist:
            print 'checking periodicity...'
            # get last idx #
            # http://stackoverflow.com/questions/34667282/numpy-where-detailed-step-by-step-explanation-examples
            final_idx = np.where(crossing_idx==1)[0][-1]

            # get approx period
            crossing_t = temp_t[1:][crossing_idx]
            period = crossing_t[-1]-crossing_t[-2]
            
            temp_TN = int(period/self.dt)

            # get approx init
            temp_sol2 = np.zeros((temp_TN,2,self.N,self.N))
            temp_sol2[0,:,:,:] = temp_sol[final_idx,:,:,:]

            # integrate for 1 period
            for i in range(1,temp_TN):
                temp_sol2[i,:,:,:] = temp_sol2[i-1,:,:,:] + self.dt*rhs(temp_sol2[i-1,:,:,:],
                                                                      self.dt*i,
                                                                      self.eps,
                                                                      self.mu,
                                                                      self.Ktablefft,
                                                                      self.Itable,
                                                                      self.g,
                                                                      self.q,
                                                                      self.r,
                                                                      self.ut,
                                                                      self.N)
                
            temp_x_locs,temp_y_locs = self.get_center_mass_time(temp_sol2)
            temp_th1 = temp_x_locs # rename for convenience
            temp_th2 = temp_y_locs
            
            if False:
                # just test plotting
                mp.figure()
                mp.plot(temp_th1,temp_th2)

                mp.figure()
                mp.matshow(temp_sol[final_idx,0,:,:])

                mp.figure()
                mp.plot(temp_th1)
                mp.plot(temp_th2)
                mp.show()

            # check tolerance
            err = (np.abs(temp_th1[-1]-temp_th1[0])+np.abs(temp_th2[-1]-temp_th2[0]))
            if err<tol:
                print 'limit cycle found!', 'tol =',np.abs(temp_th1[-1]-temp_th1[0])+np.abs(temp_th2[-1]-temp_th2[0])
                self.lc_t = np.linspace(0,period,temp_TN)
                self.lc_th1 = temp_th1
                self.lc_th2 = temp_th2
                
                self.lc_data = np.zeros((len(self.lc_t),3))

                self.lc_data[:,0] = self.lc_t
                self.lc_data[:,1] = self.lc_th1
                self.lc_data[:,2] = self.lc_th2
                
            else:
                print 'LIMIT CYCLE CANDIDATE DOES NOT EXIST, err =',err
                self.lc_data = np.zeros((2,3))
                self.limit_cycle_exists = False
                self.lc_t = [-1,-1]
                self.lc_th1 = [-1,-1]
                self.lc_th2 = [-1,-1]

        else: 
            print 'LIMIT CYCLE CANDIDATE DOES NOT EXIST'
            self.lc_data = np.zeros((2,3))
            self.limit_cycle_exists = False
            self.lc_t = [-1,-1]
            self.lc_th1 = [-1,-1]
            self.lc_th2 = [-1,-1]


        self.lc_data[:,0] = self.lc_t
        self.lc_data[:,1] = self.lc_th1
        self.lc_data[:,2] = self.lc_th2




    def idx2cartesian(self,idx_pair):
        """
        transform idx_pair into cartesian coordinates.
        idx_pair: tuple of non-integeter indices of center of mass of some square matrix.
        dom: define domain to get the proper transformation.
        """
        dim = len(idx_pair)
        coord = np.zeros(dim)


        # loop over each index value
        for i in range(dim):
            num = idx_pair[i]

            decim,integ = math.modf(num)
            #print i,decim,integ,

            #idx[i,:] = [1,np.floor(num),np.ceil(num),decim]
            # get weighted average:
            if i == 1:
                coord[i] = np.flipud(self.X)[int(np.floor(num))]*(1-decim)+np.flipud(self.X)[int(np.ceil(num))]*decim
            else:
                coord[i] = self.X[int(np.floor(num))]*(1-decim)+self.X[int(np.ceil(num))]*decim
        return coord
        
    def get_fourier_modes(self,table,threshold=2.):
        """
        extract largest fourier modes for table/array above threshold
        threshold is the non-normalized amplitude.
        i might implement a more specific function for odd/even functions later, but today is not that day.

        output: c (coefficients in matrix form), c_1d (coefficients in flattended list), idx (coefficient indiex tuples in 1d list)
        """
        #nx,ny = np.shape(table)
        #X, Y = np.meshgrid(np.linspace(self.a,self.b,nx),
        #                   np.linspace(self.a,self.b,ny))

        coeffs = np.fft.fft2(table)
        c,c_1d,f_idx = f2d.extract_coeff_all(coeffs,threshold=threshold,return_1d=True)
        
        self.c = c
        self.c_1d = c_1d
        self.f_idx = f_idx

        return self.c,self.c_1d,self.f_idx


    def get_center_mass(self,a,i=0):
        """
        get best center mass of 2d bump.
        a must be shifted to the origin using get_peak.
        assume periodic
        """

        """
        if i > 100:
            fig55 = plt.figure()
            ax55 = fig55.gca(projection='3d')
            ax55.set_title("initial bump solution")
            ax55 = plot_s(ax55,a)
            plt.show()
        """
        
        centermass_coords = center_of_mass(a)
        #print centermass_coords
        locs = self.idx2cartesian(centermass_coords[::-1])
        return locs[0],locs[1]


    def get_center_mass_time(self,sol):
        T_iter = len(sol[:,0,0,0])
        x_locs = np.zeros(T_iter)
        y_locs = np.zeros(T_iter)

        #w = np.exp(-2j*np.pi*np.arange(N) / N)
        #F1 = y.dot(w)
        for i in range(T_iter):
            
            locs1 = self.get_peak(sol[i,0,:,:],return_index=True)


            #print locs
            # get best center of mass approximation after centering at peak
            #locs2 = self.get_center_mass(np.roll(np.roll(sol[i,0,:,:],
            #                                             self.N/2-locs1[2],axis=0),
            #                                     self.N/2-locs1[3],axis=1),i=i)

            ang1 = np.angle(np.fft.fft(sol[i,0,locs1[2],:]).conj()[1])
            ang2 = np.angle(np.fft.fft(sol[i,0,:,locs1[3]]).conj()[1])

            if ang1 < 0:
                ang1 += 2*pi
            if ang2 < 0:
                ang2 += 2*pi

            #print locs2#,locs1
            # shift back to original position
            x_locs[i] = self.a+ang1#(self.b - self.a)*ang1/(2*pi)
            y_locs[i] = self.a+ang2#(self.b - self.a)*ang2/(2*pi)

            #print x_locs[i],y_locs[i]
            


        if self.q == 0.:
            xv = x_locs[-1]-x_locs[-2]#np.mean(np.gradient(x_locs[-10:],self.dt))
            yv = y_locs[-1]-y_locs[-2]#np.mean(np.gradient(y_locs[-10:],self.dt))
            print 'velocity components'+' (xv,yv)='+str(xv)+','+str(yv)+')'
            print 'velocity =',np.sqrt(xv**2 + yv**2)
            print 'velocity angle',np.arctan2(yv,xv)

        return x_locs,y_locs


    def get_peak(self,a,return_index=False):
        """
        get position of peak of 2d bump
        need sol to be square matrix.
        return cartesian coordinates
        """
        
        idx = np.argmax(np.reshape(a,self.N_idx))
        ix,iy = num2coord(idx,self.N)
        
        x_loc = self.X[ix]
        y_loc = self.X[iy]

        # convert to index starting from upper left before returing.

        if return_index:
            return x_loc,-y_loc,iy,ix
        else:
            return x_loc,y_loc


    def load_pinning(self):
        """
        build or load pinning function
        N: domain size
        X: domain linspace
        Ktablefft: kernel fft
        params: sim parameters
        pos: pinning position
        plot_I: plot pinning function
        recompute_I: force function to rebuild pinning file
        use_kernel: use saved kernel file as pinning function or rebuild.
        
        -load pinning file if exists (option kernel or not kernel)
        -if no pinning file, build (option kernel or not kernel)
        
        """

        # pinning filename
        self.I_f = 'r='+str(self.r)+'_ut='+str(self.ut)+'_N='+str(self.N)+\
                    '_pinning_function='+str(self.pinning_function)+'.i'

        self.Ifile = self.dir1+'/'+self.I_f

        #print self.Ifile

        if os.path.isfile(self.Ifile) and not(self.recompute_I):
            # logic:
    
            print '* Pinning file found. Loading...'
            I = np.loadtxt(self.Ifile)
            print '  ... done.'
            print 
            self.Itable = shift(I,self.ishift1,self.ishift2)



        else:
            # else, build functions. option kernel or not kernel
            I = np.zeros((self.N,self.N))
            if self.pinning_function == 'kernel':
                print '* Building pinning function using kernel...'
                for i in range(self.N):
                    stdout.write("\r  ... building... %d%%" % int((100.*(i+1)/self.N)))
                    stdout.flush()
                    for j in range(self.N):
                        I[i,j] = self.K_diff_p(self.X[i],self.X[j])
                print
                print '  ... done.'
                np.savetxt(self.Ifile,I)
                self.Itable = shift(I,ishift1,ishift2)


            elif self.pinning_function == 'ss':
                #print '* Building pinning function using ss bump...'
                temp = copy.deepcopy(self.u0ss)
                self.Itable = shift(temp,self.ishift1,self.ishift2)


            elif self.pinning_function == 'gaussian' or self.pinning_function == 'gaussian-.1':
                print '* Building pinning function using gaussian...'
                for i in range(self.N):
                    stdout.write("\r  ... building... %d%%" % int((100.*(i+1)/self.N)))
                    stdout.flush()
                    for j in range(self.N):
                        I[i,j] = self.K_gauss_p(self.X[i],self.X[j])
                print
                print '  ... done.'
                np.savetxt(self.Ifile,I)
                self.Itable = shift(I,self.ishift1,self.ishift2)
                
                print self.pinning_function
                if self.pinning_function == 'gaussian-.1':
                    self.Itable -= .1

            elif self.pinning_function == 'double':
                print '* Building double pinning function ...'
                #print '* Building pinning function using ss bump...'
                temp1 = copy.deepcopy(self.u0ss)
                temp2 = copy.deepcopy(self.u0ss)

                self.Itable = shift(temp2,-self.ishift1,-self.ishift2)+shift(temp1,self.ishift1,self.ishift2)

                np.savetxt(self.Ifile,self.Itable)
                
            else:
                raise ValueError('unknown pinning function option'+str(self.pinning_function))
        
        #print np.angle(np.fft.fft2(self.Itable))[:3,:3]
        
        #print self.get_center_mass(self.Itable),self.ishift1,self.ishift2
        
    def plot(self,option='phase_space',skip=200):
        """
        options: phase_space, phase_time, init, pinning, final, movie
        """

        fig  = plt.figure()
        if option == 'phase_time' or option == 'phase_space':
            ax = fig .add_subplot(111)

            if option == 'phase_time':
                ax.plot(self.t,self.th1)
                ax.plot(self.t,self.th2)

            elif option == 'phase_space':
                z = np.linspace(0,1,len(self.th1)+len(self.th1)/5)[len(self.th1)/5:]
                color = plt.cm.Greys(z)

                ax.scatter(self.th1,self.th2,edgecolor='',facecolor=color)
                #ax.plot(self.th1,self.th2,lw=2)
                ax.set_xlim(-pi,pi)
                ax.set_ylim(-pi,pi)

        elif option == "init":
            # initial bump
            ax = fig.gca(projection='3d')
            ax.set_title("initial bump solution")
            ax = plot_s(ax,self.sol[0,0,:,:])

        elif option == "pinning":
            ax = fig.gca(projection='3d')
            #ax.set_zlim(-5,10)
            ax.set_title("I; pinning")
            ax = plot_s(ax,self.Itable)

        elif option == 'final':
            #plot final bump position
            ax = fig.gca(projection='3d')
            ax.set_title("Final bump solution")
            ax = plot_s(ax,self.sol[-1,0,:,:])
            
        elif option == 'movie':
            plot_surface_movie(self.X,self.sol,self.t,skip,movie=False)

        return fig
            


def rhs(y_old,t,eps,mu,Ktablefft,Itable,g,q,r,ut,N):
    """
    y_old: solution values in shape(2*N*N,).
    First N*N entries are u, last N*N entries are z
    for manipulation, reshape to (2,N,N),
    where y_old[0,:,:] are u vals
    and y_old[1,:,:] are z vals

    t: time
    eps: epsilon
    mu: time scaling for adaptation
    K: 2D kernel lookup table
    
    Itable: N_idx array
    g,q,r,ut: parameters
    x,y: domain coordinates [0,2pi]?
    N: dimension of domain
    N_idx: neuron index, or total neuron number
    """

    output = np.zeros((2,N,N))
    u = y_old[0,:,:]
    z = y_old[1,:,:]
    
    # get f(u(x_1,x_2))
    fu = f(u,r,ut)

    # get convolution
    #Ktablefft = np.fliplr(np.flipud(Ktablefft))
    wf = np.real(np.fft.ifft2(Ktablefft*np.fft.fft2(fu)))

    k_rows, k_cols = fu.shape
    k_rows += 1
    k_cols += 1
    # the shifting is necessary.
    # http://stackoverflow.com/questions/18172653/convolving-a-periodic-image-with-python
    wf = np.roll(np.roll(wf, -(k_cols//2), axis=-1),
                 -(k_rows//2), axis=-2)

    output[0,:,:] = -u + wf + eps*(q*Itable - g*z)
    output[1,:,:] = eps*(u-z)*(1./mu)

    return output


def f(x,r=15.,ut=.25,d=False):
    """
    nonlinearity switch
    """
    a = np.exp(-r*(x-ut))
    if d:
        return (r*a)/(1.+a)**2
    else:
        return 1./(1.+a)

def num2coord(k,N):
    """
    return coordinate index pair given cell number
    I'm using the natural ordering starting from the bottom left
    N is the side length
    """
    x_idx = k%N
    y_idx = int(np.floor((k+0.5)/N))
    return (x_idx,y_idx)

def plot_surface_movie(X,sol,t,skip,movie=False,
                       file_prefix="mov/test",
                       file_suffix=".jpg",
                       title=""):
    """
    take fully computed surface solution and display the plot over time
    X: domain
    sol: full solution with u,z for all time
    TN: number of time steps
    skip: number of time steps to skip per plot display
    """
    TN = len(t)
    XX,YY = np.meshgrid(X,X)

    if movie:
        pass
    else:
        fig = plt.figure()
        fig2 = plt.figure()

        plt.ion()
        #plt.show()

        ax = fig.gca(projection='3d')
        ax2 = fig2.gca(projection='3d')

        #g1 = fig.add_subplot(111)

        #g2 = fig.add_subplot(111)
        #ax.set_zlim(-5,10)
        #ax2.set_zlim(-5,10)

        ax.set_title("u"+title)
        #ax2.set_title("z")
    
    #N = sol[0,0,:,0]
    lo = -3#lo = np.amin(np.reshape(sol[:,0,:,:],(TN*N*N)))
    hi = 5#hi = np.amax(np.reshape(sol[:,0,:,:],(TN*N*N)))
    for i in range(TN):
        k = i*skip
        
        
        if movie:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_zlim(lo,hi)
            ax.set_title("u")
            ax = plot_s(ax,sol[k,0,:,:])

            fig.savefig(file_prefix+str(i)+file_suffix)
            ax.text(0,0,11,"t="+str(t[k]))
            ax.text(0,0,10,"g="+str(g)+", q="+str(q)+", eps="+str(eps))
            plt.close()

        else:
            #g1.matshow(np.reshape(sol[i,:N_idx],(N,N)))
            ax.set_zlim(lo,hi)
            ax = plot_s(ax,sol[k,0,:,:])
            ax2 = plot_s(ax2,sol[k,1,:,:])

            plt.pause(.01)
            ax.clear()
            ax2.clear()

        stdout.write("\r Simulation Recapping... %d%%" % int((100.*(k+1)/len(t))))
        stdout.flush()
    print

    
    K = N+1
    N_idx = N*N

    if display_params:
        print "mu =",mu,"; Kamp =",Kamp,"; se =",se,";si =",si,"; r =",r,"; ut =",ut,"; eps =",eps,"; q =",q,"; g =",g,"; N =",N,"; K =",K,"; N_idx =",N_idx,"; ushift1 =",ushift1,"; ushift2 =",ushift2,"; zshift1 =",zshift1,"; zshift2 =",zshift2
    return mu,Kamp,se,si,r,ut,eps,q,g,N,K,N_idx,ushift1,ushift2,zshift1,zshift2


#def main(screen):
def main(argv):

    # process terminal flags
    try:
        opts, args = getopt.getopt(argv, "lvserhpf", ["use-last","save-last","use-ss","save-ss","use-random","help","run-phase","run-full"])

    except getopt.GetoptError:
        usage()
        sys.exit(2)

    use_last=False;save_last=False;use_ss=False;save_ss=False;use_random=False
    run_full=False;run_phase=False

    if opts == []:
        print "Please run using flags -p (phase model) and/or -f (full model)"
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):

            usage()
            sys.exit()
        else:
            if opt in ("-l","--use-last"):
                use_last = True
                print "use_last=True"
            elif opt in ('-v','--save-last'):
                save_last = True
                print "save_last=True"
            elif opt in ('-s','--use-ss'):
                use_ss = True
                print "use_ss=True"
            elif opt in ('-e','save-ss'):
                save_ss = True
                print "save_ss=True"
            elif opt in ('-r','use-random'):
                use_random = True
                print "use_random=True"
            elif opt in ('-p','run-phase'):
                run_phase = True
                print "run class phase=True"
            elif opt in ('-f','run-full'):
                run_full = True
                print "run class theta (full sim)=True"

    """
    ktest = Kernel(recompute_kernel=False,kernel_type='diff_gauss')
    ktest.plot()
    plt.show()

    u0b_test = SteadyState(recompute_ss=False)
    u0b_test.plot("u0b")
    u0b_test.plot("u0b_grad_x")
    u0b_test.plot("u0b_grad_y")

    plt.show()
    #phase = Phase(recompute_h=False,phase_option='approx2')
    """

    ### SIMULATION OPTIONS ###
    # 1==True, 0==False
    recompute_ss = False
    use_other_ss = False # use final state from other parameter values
    phase = False # run phase estimation
    break_symm = False # break symmetry in traveling bump regime 


    if run_full:

        # run simulation
        # z shift angle and radius
        #zshift_angle=pi/10.;zshift_rad=.4
        zshift_angle=pi/5.;zshift_rad=.3

        print 'initial angle',zshift_angle,'inital rad',zshift_rad
        ushift1=1.;ushift2=1.
        zshift1=ushift1-zshift_rad*np.cos(zshift_angle);zshift2=ushift2-zshift_rad*np.sin(zshift_angle)
        ishift1=0.;ishift2=0.

        # helix g=3.74

        simdata = SimDat(q=0.,g=1.3,
                         ushift1=ushift1,
                         ushift2=ushift2,
                         zshift1=zshift1,
                         zshift2=zshift2,
                         ishift1=ishift1,
                         ishift2=ishift2,
                         pinning_function='ss',
                         initial_function='ss',
                         recompute_kernel=False,
                         recompute_I=True,
                         save_last=save_last,
                         use_last=use_last,
                         T=5000,eps=.01)

        
        """
        c,c1,idx = simdata.get_fourier_modes(np.roll(np.roll(simdata.u0ss,simdata.N/2,axis=-1),simdata.N/2,axis=-2),threshold=100.)
        for i in range(len(c1)):
            print i,'&',np.real(c1[i])/(simdata.N**2),'&',idx[i],'\\\\'



        c,c1,idx = simdata.get_fourier_modes(simdata.Ktable,threshold=1.)
        for i in range(len(c1)):
            print i,'&',np.real(c1[i])/(simdata.K/(2*pi))**2,'&',idx[i],'\\\\'
        """
        
        #simdata.plot('u0b')
        simdata.plot('phase_space')
        simdata.plot('phase_time')
        #simdata.plot('pinning')
        #simdata.plot('init')
        #simdata.plot('final')
        #simdata.plot('movie')

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
