"""
2D neural field phase model

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
import collections
import matplotlib.pylab as mp
import os
#import scipy as sp
import scipy as sp
from scipy.integrate import odeint,dblquad
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import scipy.spatial as spatial
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hier
import time
#from colorsys import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
import copy
import math



from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])

# anim
import matplotlib.pyplot as plt


import fourier_2d as f2d
from euler import ESolve
from twod_full import SimDat as sd
from twod_full import f,plot_s
from lib import *

#sd = sd(display_params=False)


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


def shift(Z,x,y):
    """
    shift surface Z by coordinates x,y
    """
    N,N = Z.shape
    Nx = int(N*x/(2*pi))
    Ny = int(N*y/(2*pi))
    return np.roll(np.roll(Z,Nx,axis=1),Ny,axis=0)


class Phase(sd):
    """
    simulate phase equation
    """
    def __init__(self,
                 check_h=False,
                 check_j=False,
                 recompute_h=False,
                 recompute_j=False,
                 recompute_fq=True,
                 recompute_phase_lc=False,
                 compute_h_error=False,
                 new_phase_rhs=False,
                 low_memory=False,
                 use_last=False,
                 save_last=False,
                 pertx=False,
                 perty=False,
                 init_mode='polar',
                 dde_T=100,
                 dde_dt=.1,
                 dde_delay_t=20,
                 g=0.,q=0.,
                 x0=0,y0=0,
                 x1=0,y1=0,
                 dde_periodization_lower=-2,
                 dde_periodization_upper=2,
                 phase_option='full'):


        """
        compute_h_error: True or False. Compute the error between lookup table H_1 and Fourier approximation of H_1
        low_memory: if false, excludes all simulations that are memory-intensive. Some plots may not be available.
        
        """

        """
        Sim.__init__(self)
        Kernel.__init__(self)
        
        """
        sd.__init__(self,display_params=False)
        
        #SteadyState.__init__(self)

        self.init_mode = init_mode

        self.x0 = x0 # initial x-coordinate (1st pair)
        self.y0 = y0 # initial y-coordinate (1st pair)

        self.x1 = x1 # initial x-coordinate (2nd pair)
        self.y1 = y1 # initial y-coordinate (2nd pair)

        self.phase_option = phase_option
        self.new_phase_rhs=new_phase_rhs

        self.dde_periodization_lower = dde_periodization_lower
        self.dde_periodization_upper = dde_periodization_upper

        self.g = g
        self.q = q

        self.dde_T = dde_T
        self.dde_dt = dde_dt
        self.dde_delay_t = dde_delay_t
        self.dde_TN = int(self.dde_T/self.dde_dt)
        self.dde_t = np.linspace(0,self.dde_T+self.dde_dt,self.dde_TN)
        self.dde_delay_N = int(self.dde_delay_t/self.dde_dt)


        self.recompute_h = recompute_h
        self.recompute_j = recompute_j
        self.recompute_phase_lc = recompute_phase_lc
        self.recompute_fq = recompute_fq

        self.use_last = use_last
        self.save_last = save_last

        self.check_h = check_h
        self.check_j = check_j

        self.pertx = pertx
        self.perty = perty

        self.dde_dir = 'opt='+str(phase_option)+\
                       '_delayN='+str(self.dde_delay_N)+\
                        '_dt='+str(self.dde_dt)

        if (not os.path.exists(self.savedir+'/'+self.dde_dir)):
            os.makedirs(self.savedir+'/'+self.dde_dir)


        self.filename_th1 = self.savedir+'/'+self.dde_dir+'/th1_last.dat'
        self.filename_th2 = self.savedir+'/'+self.dde_dir+'/th2_last.dat'
        self.filename_thi_t = self.savedir+'/'+self.dde_dir+'/thi_t_last.dat'


        self.H1,self.H2 = self.H_i()
        self.J1,self.J2 = self.J_i()

        print '* Running phase_dde()...'
        self.th1_ph,self.th2_ph = self.phase_dde()

        print '  ... done.'
        
        if compute_h_error:
            err_h1,err_j1 = self.HJ_i_error()
            print 'H1_lookup vs H1_fourier error =',err_h1
            print 'J1_lookup vs J1_fourier error =',err_j1


    def h1_approx(self,x,y,sig=5.,a=.1):
        #return x*exp(-(x**2+y**2)**2)
        # based on numerics h1 seems to be y*exp
        if self.phase_option == 'approx2':
            return x*exp(-(x**2+y**2)**2/sig**2) - a*sin(x)
        else:
            return x*exp(-(x**2+y**2)**2/sig**2)

    def h2_approx(self,x,y):
        return self.h1_approx(y,x)

    def h1_approx_p(self,x,y):
        """
        periodized kernel using difference of gaussians
        """
        tot = 0
        for n in np.arange(self.dde_periodization_lower,self.dde_periodization_upper+1,1):
            for m in np.arange(self.dde_periodization_lower,self.dde_periodization_upper+1,1):
                tot = tot + self.h1_approx(x+n*2*pi,y+m*2*pi)
        return tot


    def h2_approx_p(self,x,y):
        """
        periodized kernel using difference of gaussians
        """
        tot = 0
        for n in np.arange(self.dde_periodization_lower,self.dde_periodization_upper+1,1):
            for m in np.arange(self.dde_periodization_lower,self.dde_periodization_upper+1,1):
                tot = tot + self.h2_approx(x+n*2*pi,y+m*2*pi)
        return tot


    def phase_dde(self):
        """
        the full integro-delay-differential equation using Euler's method
        x: x[:N], x[N:]. x history and y history, respectively. up to N time steps.    

        todo:
        -start with bard's approximations
        -once the bard method works, move on to Fourier stuff.

        I will use this guy's code if I have to:
        https://zulko.wordpress.com/2013/03/01/delay-differential-equations-easy-with-python/
        """

        file_not_found = False
        while True:

            if self.use_last and not(file_not_found):
                if os.path.isfile(self.filename_th1) and\
                   os.path.isfile(self.filename_th2):
                    print 'using last'
                    th1_0 = np.loadtxt(self.filename_th1)
                    th2_0 = np.loadtxt(self.filename_th2)
                    break
                else:
                    print 'init file not found'
                    file_not_found = True
            else:
                #np.random.seed(0)
                if self.init_mode == 'polar':
                    print 'using polar init'
                    r0 = self.x0#.36219
                    nu0 = self.y0#1.2458
                    th0 = np.linspace(0,-self.dde_delay_t,self.dde_delay_N)*nu0
                    th1_0 = r0*cos(th0)
                    th2_0 = r0*sin(th0)
                    
                elif self.init_mode == 'cartesian':
                    print 'using cartesian init'
                    init_angle = np.arctan2(self.y1-self.y0,self.x0-self.x1)
                    if init_angle < 0:
                        init_angle += 2*pi

                    print 'initial angle',init_angle
                    x_line = np.linspace(self.x0,self.x1,self.dde_delay_N)
                    y_line = np.linspace(self.y0,self.y1,self.dde_delay_N)
                    th1_0 = x_line
                    th2_0 = y_line
                    
                    if self.pertx:
                        print 'Reminder: added small perturbation to x init'
                        N = 20
                        th1_0[-N:]+=.01*np.exp(-np.linspace(0,N*self.dde_dt,N))
                        #th2_0[-150:-145]+=.01

                    if self.perty:
                        print 'Reminder: added small perturbation to y init'
                        th1_0[-150:-145]+=.01
                        #th2_0[-150:-145]+=.01


                    
                else:
                    raise ValueError('no initial choice'+str(self.init_mode))

                break
        
        th1 = np.zeros(self.dde_TN)
        th2 = np.zeros(self.dde_TN)

        th1[:self.dde_delay_N] = th1_0
        th2[:self.dde_delay_N] = th2_0

        # approximate the H function as a negative gaussian derivative: x*exp(-(x^2+y^2))

        # solve dde
        # for reference: H_1(x,y) = x*exp(-(x^2+y^2))
        # so H_1(th1(tau-s)-th1(tau),th2(tau-s)-th2(tau))

        n = np.arange(0,self.dde_delay_N,1)
        for i in range(self.dde_delay_N-1,self.dde_TN):
            if self.phase_option == 'approx' or self.phase_option == 'approx2':

                h1_val = self.h1_approx_p(th1[i-1-n]-th1[i-1],th2[i-1-n]-th2[i-1])
                h2_val = self.h2_approx_p(th1[i-1-n]-th1[i-1],th2[i-1-n]-th2[i-1])

                j1 = -self.h1_approx_p(th1[i-1],th2[i-1])
                j2 = -self.h2_approx_p(th1[i-1],th2[i-1])

            elif self.phase_option == 'full':
                h1_val = f2d.H1_fourier(th1[i-1-n]-th1[i-1],th2[i-1-n]-th2[i-1])
                h2_val = f2d.H2_fourier(th1[i-1-n]-th1[i-1],th2[i-1-n]-th2[i-1])
                
                j1 = -f2d.H1_fourier(th1[i-1],th2[i-1])
                j2 = -f2d.H2_fourier(th1[i-1],th2[i-1])

            elif self.phase_option == 'trunc':
                h1_val = self.h1(th1[i-1-n]-th1[i-1],th2[i-1-n]-th2[i-1],0.8)
                h2_val = self.h2(th1[i-1-n]-th1[i-1],th2[i-1-n]-th2[i-1],0.8)
                
                j1 = -self.h1(th1[i-1],th2[i-1],0.8)
                j2 = -self.h2(th1[i-1],th2[i-1],0.8)

                
            th1[i] = th1[i-1] + self.dde_dt*( -(1.*self.g)*np.sum(np.exp(-n*self.dde_dt)*h1_val)*self.dde_dt + self.q*j1 )
            th2[i] = th2[i-1] + self.dde_dt*( -(1.*self.g)*np.sum(np.exp(-n*self.dde_dt)*h2_val)*self.dde_dt + self.q*j2 )


            #if self.phase_option == 'approx':
            th1 = np.mod(th1+pi,2*pi)-pi
            th2 = np.mod(th2+pi,2*pi)-pi
            #elif self.phase_option == 'full':
        if self.q == 0:
            xv = th1[-1]-th1[-2]#np.mean(np.gradient(th1[-10:],self.dde_dt))
            yv = th2[-1]-th2[-2]#np.mean(np.gradient(th2[-10:],self.dde_dt))
            print 'velocity components'+' (xv,yv)='+str(xv)+','+str(yv)+')'
            print 'velocity =',np.sqrt(xv**2 + yv**2)
            final_angle = np.arctan2(yv,xv)
            if final_angle < 0:
                final_angle += 2*pi
            print 'velocity angle',final_angle


        if False:
            mp.figure()
            mp.plot(th1[-self.dde_delay_N:],th2[-self.dde_delay_N:])
            mp.show()

        if self.save_last:
            np.savetxt(self.filename_th1,th1[-self.dde_delay_N:])
            np.savetxt(self.filename_th2,th2[-self.dde_delay_N:])
            np.savetxt(self.filename_thi_t,self.dde_t[-self.dde_delay_N:])

        return th1,th2



    def phase_dde_v2(self,dde_TN,x0,y0,phase_option='full'):
        """
        v2 is the same as above, but with manual input params and improved control over initial conditions

        x0,y0: initial arrays up to self.dde_delay_N, self.dde_delay_t

        the full integro-delay-differential equation using Euler's method
        x: x[:N], x[N:]. x history and y history, respectively. up to N time steps.    

        """
        
        th1 = np.zeros(dde_TN)
        th2 = np.zeros(dde_TN)

        th1[:self.dde_delay_N] = x0
        th2[:self.dde_delay_N] = y0

        # approximate the H function as a negative gaussian derivative: x*exp(-(x^2+y^2))

        # solve dde
        # for reference: H_1(x,y) = x*exp(-(x^2+y^2))
        # so H_1(th1(tau-s)-th1(tau),th2(tau-s)-th2(tau))

        n = np.arange(0,self.dde_delay_N,1)
        for i in range(self.dde_delay_N-1,dde_TN):
            if phase_option == 'approx' or phase_option == 'approx2':

                h1_val = self.h1_approx_p(th1[i-1-n]-th1[i-1],th2[i-1-n]-th2[i-1])
                h2_val = self.h2_approx_p(th1[i-1-n]-th1[i-1],th2[i-1-n]-th2[i-1])

                j1 = -self.h1_approx_p(th1[i-1],th2[i-1])
                j2 = -self.h2_approx_p(th1[i-1],th2[i-1])

            elif phase_option == 'full':
                h1_val = f2d.H1_fourier(th1[i-1-n]-th1[i-1],th2[i-1-n]-th2[i-1])
                h2_val = f2d.H2_fourier(th1[i-1-n]-th1[i-1],th2[i-1-n]-th2[i-1])
                
                j1 = -f2d.H1_fourier(th1[i-1],th2[i-1])
                j2 = -f2d.H2_fourier(th1[i-1],th2[i-1])


            elif self.phase_option == 'trunc':
                h1_val = self.h1(th1[i-1-n]-th1[i-1],th2[i-1-n]-th2[i-1],0.8)
                h2_val = self.h2(th1[i-1-n]-th1[i-1],th2[i-1-n]-th2[i-1],0.8)
                
                j1 = -self.h1(th1[i-1],th2[i-1],0.8)
                j2 = -self.h2(th1[i-1],th2[i-1],0.8)

                
            th1[i] = th1[i-1] + self.dde_dt*( -(1.*self.g)*np.sum(np.exp(-n*self.dde_dt)*h1_val)*self.dde_dt + self.q*j1 )
            th2[i] = th2[i-1] + self.dde_dt*( -(1.*self.g)*np.sum(np.exp(-n*self.dde_dt)*h2_val)*self.dde_dt + self.q*j2 )

            th1 = np.mod(th1+pi,2*pi)-pi
            th2 = np.mod(th2+pi,2*pi)-pi

        return th1,th2


    def load_phase_lc(self):
        """
        if lc data exists, load. if DNE or recompute required, compute here.
        """
        file_not_found = False

        self.filename_lc_phase = self.lcdir+'/'+'lc_phase.dat'

        while True:
            if self.recompute_phase_lc or file_not_found:
                """
                force recomputation of LC
                """
                self.compute_phase_lc() # contains self.lc_phase_data
                np.savetxt(self.filename_lc_phase,self.lc_phase_data)
                break

            else:
                if os.path.isfile(self.filename_lc_phase):

                    lc_phase_data = np.loadtxt(self.filename_lc_phase)

                    self.lc_t_phase = lc_phase_data[:,0]
                    self.lc_th1_phase = lc_phase_data[:,1]
                    self.lc_th2_phase = lc_phase_data[:,2]
                    
                    self.lc_per = self.lc_t_phase[-1]
                    print 'limit cycle period', self.lc_per
                    # check to see if file contains lc or not.
                    # non-lc parameter files have [-1,-1] as the data.

                    if (lc_phase_data[0,0] == -1) and\
                       (lc_phase_data[0,1] == -1) and\
                       (lc_phase_data[0,2] == -1):
                        self.limit_cycle_exists = False
                    else:
                        self.limit_cycle_exists = True
                        
                        self.lc_th1_phase_fn = interp1d(self.lc_t_phase,self.lc_th1_phase)
                        self.lc_th2_phase_fn = interp1d(self.lc_t_phase,self.lc_th2_phase)

                    break
                else:
                    file_not_found = True

                            # make lookup tables for easier access and implementation

    def phase_lc(self,t,choice):
        if choice == 1:
            return self.lc_th1_phase_fn(np.mod(t,self.lc_per))
        if choice == 2:
            return self.lc_th2_phase_fn(np.mod(t,self.lc_per))


    def compute_phase_lc(self):
        """
        if lc not found, or if recomputation requested, compute LC.

        algorithm:
        1. use existing data. if there are enough crossings, skip to 2. if there are not enough crossings detected, re-run with more time (print time). if there are enough crossings, skip to 2. else, quit.
        2. given that there are enough crossings, check periodicity by using the last period estimate and check if the solution comes back to the start (up to some tolerance, print this). if the tolerance check fails, quit. else go to 3.
        3. if a limit cycle exists, save the limit cycle solution data with a filename containing all parameter info in the format array=[time|theta1|theta2] (i.e. to plot theta1 over time i would use plot([array[:,0],array[:,1])).
        
        """
        
        tol = .01

        # first try finding crossings with current solution data.

        temp_th1 = copy.deepcopy(self.th1_ph)
        temp_th2 = copy.deepcopy(self.th2_ph)

        find_crossings_iter = 0 # count # of times attempted to find enough LC crossings
        max_find_crossings_iter = 1
        crossings_exist = True # assume true to start

        temp_TN = self.dde_TN
        temp_t = self.dde_t



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
                temp_T = 100
                print 'not enough crossings. Re-initializing with additional time T='+str(temp_T)

                dde_TN = int(temp_T/self.dde_dt)
                temp_temp_th1 = np.zeros(dde_TN+temp_TN)
                temp_temp_th2 = np.zeros(dde_TN+temp_TN)
                
                x0 = temp_th1[-self.dde_delay_N:]
                y0 = temp_th2[-self.dde_delay_N:]

                temp_temp_th1[:self.dde_TN] = temp_th1
                temp_temp_th2[:self.dde_TN] = temp_th2

                temp_temp_th1[self.dde_TN:],temp_temp_th2[self.dde_TN:] = self.phase_dde_v2(dde_TN,x0,y0)

                temp_th1 = temp_temp_th1
                temp_th2 = temp_temp_th2

                find_crossings_iter += 1 # add 1 to number of longer sims run

        # step 2 check periodicity.
        if crossings_exist:
            print 'checking periodicity...'
            # get last idx #
            # http://stackoverflow.com/questions/34667282/numpy-where-detailed-step-by-step-explanation-examples
            final_idx = np.where(crossing_idx==1)[0][-1]

            # get approx period
            crossing_t = temp_t[1:][crossing_idx]
            period = crossing_t[-1]-crossing_t[-4]
            
            temp_TN = int(period/self.dde_dt)

            # get approx init
            temp_th1_2 = np.zeros(temp_TN)
            temp_th2_2 = np.zeros(temp_TN)

            x0 = temp_th1[(final_idx-self.dde_delay_N):final_idx]
            y0 = temp_th2[(final_idx-self.dde_delay_N):final_idx]

            print np.shape(x0)
            print len(temp_th1_2)
            print crossing_t


            #temp_th1_2[:self.dde_delay_N] = x0
            #temp_th2_2[:self.dde_delay_N] = y0


            # integrate for 1 period
            temp_th1_2,temp_th2_2 = self.phase_dde_v2(temp_TN,x0,y0)

            temp_th1 = temp_th1_2
            temp_th2 = temp_th2_2
            
            if False:
                # just test plotting
                mp.figure()
                mp.plot(temp_th1,temp_th2)

                mp.figure()
                mp.plot(temp_th1)
                mp.plot(temp_th2)
                mp.show()

            # check tolerance
            err = (np.abs(temp_th1[-1]-temp_th1[0])+np.abs(temp_th2[-1]-temp_th2[0]))
            if err<tol:
                print 'limit cycle found!', 'tol =',np.abs(temp_th1[-1]-temp_th1[0])+np.abs(temp_th2[-1]-temp_th2[0])



                self.lc_t_phase = np.linspace(0,period,temp_TN)
                self.lc_th1_phase = temp_th1
                self.lc_th2_phase = temp_th2

                self.lc_phase_data = np.zeros((len(self.lc_t_phase),3))                
                self.lc_phase_data[:,0] = self.lc_t_phase
                self.lc_phase_data[:,1] = self.lc_th1_phase
                self.lc_phase_data[:,2] = self.lc_th2_phase
                
            else:
                self.lc_phase_data = np.zeros((2,3))
                print 'LIMIT CYCLE CANDIDATE DOES NOT EXIST, err =',err
                self.limit_cycle_exists = False
                self.lc_t_phase = [-1,-1]
                self.lc_th1_phase = [-1,-1]
                self.lc_th2_phase = [-1,-1]

        else:
            self.lc_phase_data = np.zeros((2,3))
            print 'LIMIT CYCLE CANDIDATE DOES NOT EXIST'
            self.limit_cycle_exists = False
            self.lc_t_phase = [-1,-1]
            self.lc_th1_phase = [-1,-1]
            self.lc_th2_phase = [-1,-1]



        self.lc_phase_data[:,0] = self.lc_t_phase
        self.lc_phase_data[:,1] = self.lc_th1_phase
        self.lc_phase_data[:,2] = self.lc_th2_phase



    def load_fq(self):
        """
        if fq data exists, load. if DNE or recompute required, compute here.
        """
        file_not_found = False

        # load lc phase data
        self.load_phase_lc()
        
        self.filename_fq = self.lcdir+'/'+'fq.dat'

        if self.limit_cycle_exists:
            while True:
                if self.recompute_fq or file_not_found:
                    """
                    force recomputation of fq
                    """
                    self.compute_fq() # contains self.fq_data
                    np.savetxt(self.filename_fq,self.fq_data)
                    break

                else:
                    if os.path.isfile(self.filename_fq):

                        fq_data = np.loadtxt(self.filename_fq)

                        break
                    else:
                        file_not_found = True
        else:
            print 'no fq data to generate'


    def coeff_mat(self,t):
        """
        coefficient matrix for linearized system
        """
        h1x_th0,h1y_th0 = f2d.H1_fourier(self.phase_lc(t,1),self.phase_lc(t,2),d=True)
        #h2x_th0,h2y_th0 = f2d.H2_fourier_centered(self.phase_lc(t,1),self.phase_lc(t,2),d=True)
        h1x_0,h1y_0 = f2d.H1_fourier(0,0,d=True)
        #h2x_0,h2y_0 = f2d.H2_fourier_centered(0,0,d=True)

        #j1x_0=-h1x_0; j1y_0=-h1y_0
        #j2x_0=-h2x_0; j2y_0=-h2y_0

        #j1x_th0=-h1x_th0; j1y_th0=-h1y_th0
        #j2x_th0=-h2x_th0; j2y_th0=-h2y_th0

        
        #print h1x_th0,self.phase_lc(t,1),self.phase_lc(t,2)

        return np.array([[-self.q*h1x_th0 + self.g*h1x_0, -self.q*h1y_th0, -self.g, 0.],
                         [-self.q*h1y_th0, -self.q*h1x_th0 + self.g*h1x_0, 0., -self.g],
                         [h1x_0, 0.,   -1., 0.],
                         [0.,    h1x_0, 0., -1.]])

    def fq_rhs(self,y,t):
        th1=y[0]; th2=y[1]; a1=y[2]; b1=y[3]

        out = self.coeff_mat(t)

        c11=out[0,0]; c12=out[0,1]; c13=out[0,2]; c14=out[0,3]
        c21=out[1,0]; c22=out[1,1]; c23=out[1,2]; c24=out[1,3]
        c31=out[2,0]; c32=out[2,1]; c33=out[2,2]; c34=out[2,3]
        c41=out[3,0]; c42=out[3,1]; c43=out[3,2]; c44=out[3,3]
        
        #print c11, c22

        return np.array([c11*th1 + c12*th2 + c13*a1 + c14*b1,
                         c21*th1 + c22*th2 + c23*a1 + c24*b1,
                         c31*th1 + c32*th2 + c33*a1 + c34*b1,
                         c41*th1 + c42*th2 + c43*a1 + c44*b1])

    def compute_fq(self):
        """
        compute monodromy matrix and fq exp here
        """
        #phase_lc(t,choice)
        
        # integrate over one period, i.e., self.lc_per
        
        #self.lc_per


        t = np.linspace(0,self.lc_per,5000)
        start_X = np.zeros((4,4))
        end_X = np.zeros((4,4))
        

        print 'computing monodromy matrix'
        for i in range(4):
            print i+1,'of',4
            # make start B the identity
            start_X[i,i] = 1.
            
            # initialize accordingly
            init = np.zeros(4)
            init[i] = 1
            sol = odeint(self.fq_rhs,init,t)
            end_X[:,i] = sol[-1,:]

            if False:
                mp.figure()
                mp.plot(t,sol[:,0])
                mp.plot(t,sol[:,1])
                mp.plot(t,sol[:,2])
                mp.plot(t,sol[:,3])
                mp.show()
        
        #start_X_inv = np.linalg.inv(start_X)
        B = np.dot(start_X,end_X)

        self.fq_data = np.absolute(np.linalg.eig(B)[0])
        print self.fq_data
        #print np.absolute(np.linalg.eig(self.fq_data)[0])/self.lc_per



    def H_i(self):
        """
        interaction function of 2d bump
        need u0b,f, du/dx, du/dy, domain as lookup tables.
        u0b: ss bump
        df_u0b: f'(u0b)
        ux,uy: gradient of u0b
        x: domain
        i: h_i
        params: 
        check_h: set True to force compute H_i and verify saved H_i.
        """

        self.H_1_path = self.dir1+"/"+self.ss_dir2+"/"+"H_1.dat"
        self.H_2_path = self.dir1+"/"+self.ss_dir2+"/"+"H_2.dat"

        if (os.path.isfile(self.H_1_path) and os.path.isfile(self.H_2_path)) and not(self.recompute_h):
            print '* H_i files found. Loading...'
            H1 = np.loadtxt(self.H_1_path)
            H2 = np.loadtxt(self.H_2_path)
            print '  ... done.'

        else:
            print "* Computing H_i..."
            H1 = np.zeros((self.N,self.N))
            H2 = np.zeros((self.N,self.N))

            for i in range(self.N):
                stdout.write("\r  ... building... %d%%" % int((100.*(i+1)/self.N)))
                stdout.flush()
                for j in range(self.N):
                    temp1 = 0
                    temp2 = 0

                    # given x,y coordinates, compute double sum
                    # \int\int_\Omega f'(u0(y))\grad u0(y) u0(y+x)dy
                    for n in range(self.N):
                        for m in range(self.N):
                            #temp1 += self.ux[n,m]*self.df_u0b[n,m]*self.u0ss[np.mod(n+i+self.N/2,self.N),np.mod(m+j+self.N/2,self.N)]
                            #temp2 += self.uy[n,m]*self.df_u0b[n,m]*self.u0ss[np.mod(n+i+self.N/2,self.N),np.mod(m+j+self.N/2,self.N)]

                            temp1 += self.ux[n,m]*self.df_u0b[n,m]*self.u0ss[np.mod(n+i+self.N/2,self.N),np.mod(m+j+self.N/2,self.N)]
                            temp2 += self.uy[n,m]*self.df_u0b[n,m]*self.u0ss[np.mod(n+i+self.N/2,self.N),np.mod(m+j+self.N/2,self.N)]
                    H1[i,j] = temp1*((2*pi)**2)/(self.N**2)
                    H2[i,j] = temp2*((2*pi)**2)/(self.N**2)
            np.savetxt(self.H_1_path,H1)
            np.savetxt(self.H_2_path,H2)

        print
        return H1, H2


    def J_i(self):
        """
        pinning approximation
        interaction function of 2d bump
        need u0b,f, du/dx, du/dy, I, domain as lookup tables.
        u0b: ss bump
        df_u0b: f'(u0b)
        ux,uy: gradient of u0b
        x: domain
        I: pinning function
        params: 
        check_J: set True to force compute J_i and verify saved J_i.
        
        """
        self.J_1_path = self.dir1+"/"+self.ss_dir2+"/"+"J_1.dat"
        self.J_2_path = self.dir1+"/"+self.ss_dir2+"/"+"J_2.dat"

        if (os.path.isfile(self.J_1_path) and os.path.isfile(self.J_2_path)) and not(self.recompute_j):
            print '* J_i files found. Loading...'
            J1 = np.loadtxt(self.J_1_path)
            J2 = np.loadtxt(self.J_2_path)
            print '  ... done.'
            
        else:
            print "* Computing J_i..."
            J1 = np.zeros((self.N,self.N))
            J2 = np.zeros((self.N,self.N))
        
            for i in range(self.N):
                stdout.write("\r  ... building... %d%%" % int((100.*(i+1)/self.N)))
                stdout.flush()
                for j in range(self.N):
                    temp1 = 0
                    temp2 = 0

                    # given x,y coordinates, compute double sum
                    # \int\int_\Omega I(y)f'(u0(y+x))\grad u0(y+x)dy
                    for n in range(self.N):
                        for m in range(self.N):
                            temp1 += self.ux[np.mod(n+i+self.N/2,self.N),np.mod(m+j+self.N/2,self.N)]*self.df_u0b[np.mod(n+i+self.N/2,self.N),np.mod(m+j+self.N/2,self.N)]*self.Itable[n,m]
                            temp2 += self.uy[np.mod(n+i+self.N/2,self.N),np.mod(m+j+self.N/2,self.N)]*self.df_u0b[np.mod(n+i+self.N/2,self.N),np.mod(m+j+self.N/2,self.N)]*self.Itable[n,m]
                    J1[i,j] = temp1*((2*pi)**2)/(self.N**2)
                    J2[i,j] = temp2*((2*pi)**2)/(self.N**2)
            np.savetxt(self.J_1_path,J1)
            np.savetxt(self.J_2_path,J2)

        print
        return J1,J2

    def HJ_i_error(self):
        """
        compute error between H_i, J_i lookup tables and Fourier approximations.
        """
        
        """
        fig = plt.figure()        
        ax = fig.gca(projection='3d')
        ax.set_title("h1_table")
        ax = plot_s(ax,self.XX,self.YY,self.H1)
        
        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        ax2.set_title("h1_fourier")
        
        ax2 = plot_s(ax2,XX9,YY9,f2d.H1_fourier(XX9,YY9))

        plt.show()
        """

        # cut off last entry to match table.
        X9 = np.linspace(-pi,pi-2*pi/self.N,self.N)
        XX9, YY9 = np.meshgrid(X9,X9)
        
        err_h1 = np.amax(np.abs(self.H1 - f2d.H1_fourier(XX9,YY9)))
        err_j1 = np.amax(np.abs(self.J1 + f2d.H1_fourier(XX9,YY9)))
        return err_h1,err_j1


    def get_contour_verts(self,cn):
        """
        given contour plot, get the vertices.
        """
        
        contours = []
        # for each contour line
        for cc in cn.collections:
            paths = []
            # for each separate section of the contour line
            for pp in cc.get_paths():
                xy = []
                # for each segment of that section
                for vv in pp.iter_segments():
                    xy.append(vv[0])
                paths.append(np.vstack(xy))
            contours.append(paths)

        return contours

    def kappa_estimate(self,verbose=False):
        """
        check value of inner product (\nabla u_0,u^*).
        (u_0',u^*) = \int_0^{2\pi} \int_0^{2\pi} \nabla u_0(x,y)\cdot [f'(u_0(x,y)) \nabla u_0(x,y)] dx dy
        = \int_0^{2\pi} \int_0^{2\pi} f'(u_0(x,y)) \nabla u_0(x,y) \cdot \nabla u_0(x,y) dx dy
        = \int_0^{2\pi} \int_0^{2\pi} f'(u_0(x,y)) \| \nabla u_0(x,y) \|**2 dx dy
        
        ss0,ss1,ss_shift: ss bump solution
        """

        if verbose:
            print '\t building kappa'
        tot = 0

        for i in range(N):
            for j in range(N):
                tot += f(self.u0ss,self.r,self.ut,d=True)[i,j]*\
                       (self.ux[i,j]**2 + self.uy[i,j]**2)

        tot /= (self.N*self.N)

        if verbose:
            print '\t... done.'
            print '\t inner product kappa_2 \equiv (\\nabla u_0,u^*) =',tot

        return tot

    def twod_velocity(self,g,mode='phase',tol=5e-2,zero_vel_tol=1e-5):
        """
        return nu1,nu2 given g. b is fixed at b=0.8
        requires bifurcation data.
        g: bifurcation parameter. adaptation strength
        mode: 'phase' or 'full'. uses bifurcation data from full model or truncated phase model.

        zero_vel_tol: if velocity in axial direction, ignore. If one of the velocities is below this small number, it means the movement is axial. in this case, ignore.

        the parameters put into collect_disjoint_branches (remove_isolated, isolated_number, remove_redundant, etc) are finely tuned. do not change unless you know what you are doing!
        """

        if mode == 'phase':
            bif = np.loadtxt('twodphs3.ode.allinfo.dat')
            #bif = np.loadtxt('twod_wave_trunc_exist_all.dat')
            #bif2 = np.loadtxt('twod_wave_exist_br2.dat')
            
            # get all possible disjoint branches
            val,ty = collect_disjoint_branches(bif,remove_isolated=True,isolated_number=50)
        elif mode == 'full':
            bif = np.loadtxt('twod_wave_exist_v2.dat')
            #bif2 = np.loadtxt('twod_wave_exist_br2.dat')
            
            #bif_diag1 = np.loadtxt('twod_wave_exist_diag1.dat')
            bif_diag2 = np.loadtxt('twod_wave_exist_diag_v2.dat')
            
            # get all possible disjoint branches
            val,ty = collect_disjoint_branches(bif,remove_isolated=True,isolated_number=3,remove_redundant=False,N=10)
            val_di,ty_di = collect_disjoint_branches(bif_diag2,remove_isolated=True,isolated_number=3,remove_redundant=False,N=10)

            # combine dicts
            for key in val_di.keys():
                val[key+'d'] = val_di.pop(key)
            for key in ty_di.keys():
                ty[key+'d'] = ty_di.pop(key)
            
        for key in val.keys():
            g_data = val[key][:,0]
            v1_data = val[key][:,2]
            v2_data = val[key][:,3]

            smallest_diff = np.amin(np.abs(g-g_data))
            min_idx = np.argmin(np.abs(g-g_data))

            
            if (smallest_diff < tol) and\
               (np.abs(v1_data[min_idx])>zero_vel_tol) and\
               (np.abs(v2_data[min_idx])>zero_vel_tol) and\
               (np.abs(v2_data[min_idx] - v1_data[min_idx])>zero_vel_tol):

                tol = smallest_diff
                v1 = v1_data[min_idx]
                v2 = v2_data[min_idx]

        return v1,v2

    def parameteric_intersection(self):
        """
        get the first intersection between two parametric curves
        """

    def twod_velocity_v2(self,g,b,mode='trunc',
                         tol=5e-2,
                         diag_tol=5e-2,
                         zero_vel_tol=1e-5,
                         M_nu1 = 100,
                         M_nu2 = 100,
                         N = 200
                     ):
        """
        return nu1,nu2 given g and b.
        does not depend on any bifurcation data.
        g: bifurcation parameter. adaptation strength
        b: Fourier coefficient
        mode: 'trunc' or 'full'. uses full H function or truncated h function

        zero_vel_tol: if velocity in axial direction, ignore. If one of the velocities is below this small number, it means the movement is axial. in this case, ignore.

        solve:
        (1) 0 = -\nu_1 + g \int_0^\infty e^{-s} H_1(\nu_1 s,\nu_2 s) ds 
        (2) 0 = -\nu_2 + g \int_0^\infty e^{-s} H_2(\nu_1 s,\nu_2 s) ds

        """

        nu1 = np.linspace(0,3,M_nu1)
        nu2 = np.linspace(0,3,M_nu2)
        sint = np.linspace(0,N/10,N)
        
        nu1,nu2,sint = np.meshgrid(nu1,nu2,sint)

        N = np.shape(sint)[-1] # get size of integration variable array
        sint_pos = len(np.shape(sint))-1# get position of integration var
        
        # get limits of integration
        int_lo = sint[0,0,0]
        int_hi = sint[0,0,-1]
        dx = (int_hi-1.*int_lo)/N
        
        if mode == 'trunc':
            integrand1 = exp(-sint)*self.h1(nu1*sint,nu2*sint,b)
            integrand2 = exp(-sint)*self.h2(nu1*sint,nu2*sint,b)

        elif mode == 'full':
            integrand1 = exp(-sint)*f2d.H1_fourier(nu1*sint,nu2*sint)
            integrand2 = exp(-sint)*f2d.H2_fourier(nu1*sint,nu2*sint)
        else:
            raise ValueError('Invalid choice='+mode)

        eq1 = -nu1[:,:,0] + g*integrand1.sum(sint_pos)*dx
        eq2 = -nu2[:,:,0] + g*integrand2.sum(sint_pos)*dx
        
        # get contours

        fig = plt.figure()
        ax = fig.add_subplot(111)

        cs1 = ax.contour(nu1[:,:,0],nu2[:,:,0],eq1,levels=[0.])
        cs2 = ax.contour(nu1[:,:,0],nu2[:,:,0],eq2,levels=[0.])

        p1_all = cs1.collections[0].get_paths()
        p2_all = cs2.collections[0].get_paths()

        p1x_dict = {}
        p1y_dict = {}

        p2x_dict = {}
        p2y_dict = {}

        # this block of code will separate all branches into dictionaries.
        # redundant since we have two nontrivial curves.

        # gather nontrival zero contour from first equation
        for i in range(len(p1_all)):
            v = p1_all[i].vertices
            x = v[:,0]
            y = v[:,1]
            
            if (np.sum(np.abs(x)) <= zero_vel_tol) or (np.sum(np.abs(y)) <= zero_vel_tol):
                pass
            else:
                p1x_dict[str(i)] = x
                p1y_dict[str(i)] = y

        # gather nontrival zero contour from second equation
        for i in range(len(p2_all)):
            v = p2_all[i].vertices
            x = v[:,0]
            y = v[:,1]
            if (np.sum(np.abs(x)) <= zero_vel_tol) or (np.sum(np.abs(y)) <= zero_vel_tol):
                pass
            else:
                p2x_dict[str(i)] = x
                p2y_dict[str(i)] = y

        # warn user if there are more than 2 unique contours
        if (len(p1x_dict) > 1) or\
           (len(p1y_dict) > 1) or\
           (len(p2x_dict) > 1) or\
           (len(p2y_dict) > 1):
            raise ValueError('Warning: multiple zero contours detected. use the plot function in twod_velocity_v2')
            print 'there should be 1 zero contour for each existence equation'


        if (len(p1x_dict) < 1) or\
           (len(p1y_dict) < 1) or\
           (len(p2x_dict) < 1) or\
           (len(p2y_dict) < 1):
            raise RuntimeError('Warning: no contours detected. use the plot function in twod_velocity_v2')
            print 'there should be 1 zero contour for each existence equation'
        
        
        if False:
            mp.figure(5)
            for key in p1x_dict.keys():
                mp.plot(p1x_dict[key],p1y_dict[key])
            for key in p2x_dict.keys():
                mp.plot(p2x_dict[key],p2y_dict[key])

            
            mp.show()

        # find contour intersection. we only need the first.
        for key in p1x_dict.keys():
            x1 = p1x_dict[key]
            y1 = p1y_dict[key]
            
        for key in p2x_dict.keys():
            x2 = p2x_dict[key]
            y2 = p2y_dict[key]

        # create the interpolated functions
        t = np.linspace(0,1,len(x1))
        z = np.zeros((2,len(x1)))
        z[0,:] = x1
        z[1,:] = y1
        c1 = interp1d(t,z)

        t = np.linspace(0,1,len(x2))
        z = np.zeros((2,len(x2)))
        z[0,:] = x2
        z[1,:] = y2
        c2 = interp1d(t,z)

        def err(tt):
            t1 = tt[0]
            t2 = tt[1]
            return c1(t1)-c2(t2)

            

        try:
            t1,t2 = fsolve(err,x0=[.65,.75],factor=.01)
        except ValueError:
            print 'if you get the error, ValueError: A value in x_new is above the interpolation range. then modify starting times in def twod_velocity_v2 in twod_phase.py'

        v1,v2 = c1(t1)


        if False:
            mp.figure()
            z1 = c1(np.linspace(0,.6,10))
            x1 = z1[0,:]
            y1 = z1[1,:]
            mp.plot(x1,y1)

            z2 = c2(np.linspace(0,.9,10))
            x2 = z2[0,:]
            y2 = z2[1,:]
            mp.plot(x2,y2)


            mp.show()
        
        plt.clf()
        return v1,v2

        
    def h1(self,x,y,b,d=False):
        if d:
            return cos(x)*(1+b*cos(y)),-b*sin(x)*sin(y)
        else:
            return sin(x)*(1+b*cos(y))

    def h2(self,x,y,b,d=False):
        if d:
            return -b*sin(x)*sin(y),cos(y)*(1+b*cos(x))
        else:
            return sin(y)*(1+b*cos(x))

    def evans(self,lam,sint,g=2.):
        """
        evans function
        all meshgrids size/shape of (M,M,N)
        lam: complex number, or meshgrid on complex domain (M values)
        sint: integration variable. meshgrid on real domain (N values)
        nu1,nu2: velocity values
        """



        # get nu1,nu2 given g
        print 'reminder: implement g to nu1,nu2 conversion'

        # g=3
        #nu1=1.21;nu2=2.09

        # g=4
        #nu1=1.45;nu2=2.54

        b=.8

        def h1(x,y,d=False):
            if d:
                return cos(x)*(1+b*cos(y)),-b*sin(x)*sin(y)
            else:
                return sin(x)*(1+b*cos(y))

        def h2(x,y,d=False):
            if d:
                # sin(y)*(1+b*cos(x))
                return -b*sin(x)*sin(y),cos(y)*(1+b*cos(x))
            else:
                return sin(y)*(1+b*cos(x))
        
        Q1,Q2 = f2d.H1_fourier(-nu1*sint,-nu2*sint,d=True)
        Q3,Q4 = f2d.H2_fourier(-nu1*sint,-nu2*sint,d=True)
        
        #Q1,Q2 = h1(-nu1*sint,-nu2*sint,d=True)
        #Q3,Q4 = h2(-nu1*sint,-nu2*sint,d=True)
        
        # Q3,Q4 should be same as Q4,Q3=H1_fourier(-nu2*sint,-nu2*sint,d=True)
        
        sam = ( exp(-lam*sint)-1 ) / lam
        
        N = np.shape(sint)[-1] # get size of integration variable array
        sint_pos = len(np.shape(sint))-1# get position of integration var

        Qhat1 = (np.exp(-sint)*Q1*sam).sum(sint_pos)/N
        Qhat2 = (np.exp(-sint)*Q2*sam).sum(sint_pos)/N
        Qhat3 = (np.exp(-sint)*Q3*sam).sum(sint_pos)/N
        Qhat4 = (np.exp(-sint)*Q4*sam).sum(sint_pos)/N

        # return the complex valued functions

        return (1./g + Qhat1)*(1./g + Qhat4) - Qhat3*Qhat2


    def evans_v2(self,al,be,sint,g=2.5,b=0.8,return_intermediates=False,mode='trunc'):
        """
        evans function
        all meshgrids size/shape of (M,M,N)
        al,be: real and imaginary parts of some eigenvalue
        sint: integration variable. meshgrid on real domain (N values)
        """


        # get nu1,nu2 given g

        nu1,nu2=self.twod_velocity_v2(g,b,mode=mode)
        print 'velocity',nu1,nu2, "g="+str(g)+", b="+str(b)

        # g=4
        #nu1=1.45;nu2=2.54

        # g=3
        #nu1=1.21;nu2=2.09

        # g=2.5
        #nu1=1.0712;nu2=1.8395

        # g=2
        #nu1=.91067;nu2=1.5529

        # g=1.5
        #nu1=.70711;nu2=1.2247

        if mode == 'full':
            Q1,Q2 = f2d.H1_fourier(-nu1*sint,-nu2*sint,d=True)
            Q3,Q4 = f2d.H2_fourier(-nu1*sint,-nu2*sint,d=True)
        
        elif mode == 'trunc':
            Q1,Q2 = self.h1(-nu1*sint,-nu2*sint,b,d=True)
            Q3,Q4 = self.h2(-nu1*sint,-nu2*sint,b,d=True)
            #Q3,Q4 = h1(-nu2*sint,-nu1*sint,d=True)

        
        # Q3,Q4 should be same as Q4,Q3=H1_fourier(-nu2*sint,-nu2*sint,d=True)
        
        samp = exp(-al*sint)*cos(-be*sint) - 1
        samq = exp(-al*sint)*sin(-be*sint) 
        
        N = np.shape(sint)[-1] # get size of integration variable array
        sint_pos = len(np.shape(sint))-1# get position of integration var

        # get limits of integration
        int_lo = sint[0,0,0]
        int_hi = sint[0,0,-1]
        dx = (int_hi-1.*int_lo)/N


        ph1 = (np.exp(-sint)*Q1*samp).sum(sint_pos)*dx
        qh1 = (np.exp(-sint)*Q1*samq).sum(sint_pos)*dx

        ph2 = (np.exp(-sint)*Q2*samp).sum(sint_pos)*dx
        qh2 = (np.exp(-sint)*Q2*samq).sum(sint_pos)*dx

        ph3 = ph2
        qh3 = qh2

        ph4 = (np.exp(-sint)*Q4*samp).sum(sint_pos)*dx
        qh4 = (np.exp(-sint)*Q4*samq).sum(sint_pos)*dx


        # return the complex valued functions
        alf = al[:,:,0]
        bef = be[:,:,0]

        e_re = (g**2.)*(ph1*ph4 - qh1*qh4 - ph2*ph3 + qh2*qh3) + g*(alf*(ph1+ph4) - bef*(qh1+qh4)) + alf**2.-bef**2.
        e_im = (g**2.)*(ph1*qh4 + qh1*ph4 - qh2*ph3 - ph2*qh3) + g*(alf*(qh1+qh4) + bef*(ph1+ph4)) + 2.*alf*bef
        
        if return_intermediates:
            return e_re,e_im,ph1,qh1,ph2,qh2,ph3,qh3,ph4,qh4
        return e_re,e_im


    def evans_zero_alpha(self,g,b,al,be,sint,real=True,tol=1e-2):
        """
        return the real part of the input that yields a zero in the evans function.
        """



        e_re,e_im = self.evans_v2(al,be,sint,
                                  return_intermediates=False,g=g,b=b)
    
        fig = plt.figure()
        ax = fig.add_subplot(111)

        
        cs1 = ax.contour(al[:,:,0],be[:,:,0],e_re,levels=[0.])
        cs2 = ax.contour(al[:,:,0],be[:,:,0],e_im,levels=[0.])

        if False:
            intersection_example,contour_pts1,contour_pts2 = findIntersection(cs1,cs2,return_intermediates=True)
            

            plt.plot(contour_pts1[:,0]+.0001,contour_pts1[:,1]+.0001)
            plt.plot(contour_pts2[:,0],contour_pts2[:,1])
            plt.show()


        p1_all = cs1.collections[0].get_paths()
        p2_all = cs2.collections[0].get_paths()

        p1x_dict_raw = {}
        p1y_dict_raw = {}

        p2x_dict_raw = {}
        p2y_dict_raw = {}

        # this block of code will separate all branches into dictionaries.

        # gather nontrival zero contour from real part
        for i in range(len(p1_all)):
            v = p1_all[i].vertices
            x = v[:,0]
            y = v[:,1]

            p1x_dict_raw[str(i)] = x
            p1y_dict_raw[str(i)] = y

        # gather nontrival zero contour from imaginary part
        for i in range(len(p2_all)):
            v = p2_all[i].vertices
            x = v[:,0]
            y = v[:,1]

            p2x_dict_raw[str(i)] = x
            p2y_dict_raw[str(i)] = y


        if False:

            mp.figure(5)
            mp.title('original branches')
            for key in p1x_dict_raw.keys():
                mp.plot(p1x_dict_raw[key],p1y_dict_raw[key])
            for key in p2x_dict_raw.keys():
                mp.plot(p2x_dict_raw[key],p2y_dict_raw[key])

            
            #mp.show()
        



        # remove branches that cross the origin
        p1x_dict = {}
        p1y_dict = {}

        for key in p1x_dict_raw.keys():
            skipflag = False
            for i in range(len(p1x_dict_raw[key])):
                if np.abs(p1x_dict_raw[key][i]-p1y_dict_raw[key][i])<.01:
                    skipflag = True

            if not(skipflag):
                p1x_dict[key] = p1x_dict_raw[key]
                p1y_dict[key] = p1y_dict_raw[key]

        p2x_dict = {}
        p2y_dict = {}

        for key in p2x_dict_raw.keys():
            skipflag = False
            for i in range(len(p2x_dict_raw[key])):
                if np.abs(p2x_dict_raw[key][i]-p2y_dict_raw[key][i])<.01:
                    skipflag = True

            if not(skipflag):
                p2x_dict[key] = p2x_dict_raw[key]
                p2y_dict[key] = p2y_dict_raw[key]


        if False:
            mp.figure(6)
            mp.title('remaining branches')
            for key in p1x_dict.keys():
                mp.plot(p1x_dict[key],p1y_dict[key])
            for key in p2x_dict.keys():
                mp.plot(p2x_dict[key],p2y_dict[key])

            
            mp.show()


        # find contour intersection. if multiple mins found, take one with greater magnitude in complex plane
        # find minimia by taking differences
        
        min_xs = []
        min_ys = []

        for key1 in p1x_dict.keys():
            for key2 in p2x_dict.keys():
                rex = p1x_dict[key1]
                rey = p1y_dict[key1]
                
                imx = p2x_dict[key2]
                imy = p2y_dict[key2]

                if False:
                    mp.figure()
                    mp.plot(rex,rey,color='black',lw=3)
                    mp.plot(imx,imy,color='gray',lw=3)
                    
                    mp.show()


                newtol = tol
                for i in range(len(rex)):
                    diff_arr = (rex[i]-imx)**2. + (rey[i]-imy)**2.
                    minval = np.amin(diff_arr)

                    if minval < newtol:
                        newtol = minval
                        minx = rex[i]
                        miny = rey[i]
                        #print minx,miny
                min_xs.append(minx)
                min_ys.append(miny)

            """
            smallest_diff = np.amin(np.abs(g-g_data))
            min_idx = np.argmin(np.abs(g-g_data))
 
             
            if (smallest_diff < tol) and\
               (np.abs(v1_data[min_idx])>zero_vel_tol) and\
               (np.abs(v2_data[min_idx])>zero_vel_tol) and\
               (np.abs(v2_data[min_idx] - v1_data[min_idx])>zero_vel_tol):
 
                tol = smallest_diff
                v1 = v1_data[min_idx]
                v2 = v2_data[min_idx]
            """

        print 'minx,miny',min_xs,min_ys, 'for g,b=',g,b


        """
        for key in p1x_dict.keys():
            x1 = p1x_dict[key]
            y1 = p1y_dict[key]
            
        for key in p2x_dict.keys():
            x2 = p2x_dict[key]
            y2 = p2y_dict[key]

        # create the interpolated functions
        t = np.linspace(0,1,len(x1))
        z = np.zeros((2,len(x1)))
        z[0,:] = x1
        z[1,:] = y1
        c1 = interp1d(t,z)

        t = np.linspace(0,1,len(x2))
        z = np.zeros((2,len(x2)))
        z[0,:] = x2
        z[1,:] = y2
        c2 = interp1d(t,z)

        def err(tt):
            t1 = tt[0]
            t2 = tt[1]
            return c1(t1)-c2(t2)

        t1,t2 = fsolve(err,x0=[.6,.8],factor=.01)

        v1,v2 = c1(t1)


        if False:
            mp.figure()
            z1 = c1(np.linspace(0,.6,10))
            x1 = z1[0,:]
            y1 = z1[1,:]
            mp.plot(x1,y1)

            z2 = c2(np.linspace(0,.9,10))
            x2 = z2[0,:]
            y2 = z2[1,:]
            mp.plot(x2,y2)


            mp.show()
        
        """

        # if no intersections, return nan
        if (min_xs == []) or (min_ys == []):
            return np.nan

        # if two or more intersections, use the one with greatest magnitude in complex plane
        min_xs = np.array(min_xs)
        min_ys = np.array(min_ys)

        if (len(min_xs) >= 2):
            max_idx = np.argmax(min_xs**2. + min_yx**2.)
            min_xs = [min_xs[max_idx]]
            min_ys = [min_ys[max_idx]]
                

        if real:
            return min_xs[0]
        return min_xs[0],min_ys[0]



    def plot(self,option="h1"):

        fig = plt.figure()
        

        if option == 'h1':
            ax = fig.gca(projection='3d')
            ax.set_title("H1 (numerical)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            ax = plot_s(ax,self.H1)


        elif option == 'h2':
            ax = fig.gca(projection='3d')
            ax.set_title("H2 (numerical)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            #ax = plot_s(ax,self.XX,self.YY,self.H2)
            ax = plot_s(ax,self.H2)

        elif option == 'j1':
            ax = fig.gca(projection='3d')
            ax.set_title("J1 (numerical)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            #ax = plot_s(ax,self.XX,self.YY,self.J1)
            ax = plot_s(ax,self.J1)

        elif option == 'j2':
            ax = fig.gca(projection='3d')
            ax.set_title("J2 (numerical)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax = plot_s(ax,self.J2)

        elif option == 'h1_approx2':
            ax = fig.gca(projection='3d')
            ax.set_title("H1 (approx v2)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            z = np.sin(self.XX)*.2+self.h1_approx(self.XX,self.YY,sig=4.)
            ax = plot_s(ax,z/2.)

        elif option == 'evans':
            M_re = 300
            M_im = 300
            N = 200

            lam_re = np.linspace(-.25,1.,M_re)
            lam_im = np.linspace(-.01,2,M_im)
            sint = np.linspace(0,N/10,N)

            #LAM_re, LAM_im, SINT = np.meshgrid(lam_re,lam_im,sint,dtype=np.complex)
            LAM_re, LAM_im, SINT = np.meshgrid(lam_re,lam_im,sint)

            LAM_re_contour, LAM_im_contour = np.meshgrid(lam_re,lam_im)
            
            e_re,e_im = self.evans_v2(LAM_re,LAM_im,SINT,
                                      return_intermediates=False,g=4.,b=.4)


            ax = fig.add_subplot(111)

            #e_re = np.cos(2*LAM_re_contour*pi)*np.sin(LAM_im_contour*pi)
            #e_im = np.sin(2*LAM_re_contour*pi)*np.cos(LAM_re_contour*pi)

            cs_re = ax.contour(LAM_re_contour,LAM_im_contour,e_re,levels=[0.])
            cs_im = ax.contour(LAM_re_contour,LAM_im_contour,e_im,levels=[0.])

            p = cs_re.collections[0].get_paths()[0]
            v = p.vertices
            x = v[:,0]
            y = v[:,1]

            cs_re.collections[0].set_color('black')
            cs_re.collections[0].set_label('re')
            cs_re.collections[0].set_linewidths(2)

            cs_im.collections[0].set_color('gray')
            cs_im.collections[0].set_label('im')
            cs_im.collections[0].set_linewidths(2)

            ax.legend()



            # plot real and imag parts
        
        elif option == 'phase_time':
            ax = fig.add_subplot(111)
            ax.set_title("phase over time")
            ax.set_xlabel('t')
            ax.set_ylabel(r"$\theta$")
            ax.plot(self.dde_t,np.mod(self.th1_ph+pi,2*pi)-pi)
            ax.plot(self.dde_t,np.mod(self.th2_ph+pi,2*pi)-pi)

        elif option == 'phase_space':
            ax = fig.add_subplot(111)
            ax.set_title("phase in space")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            #ax.set_xlim(-pi,pi)
            #ax.set_ylim(-pi,pi)
            #ax.plot(np.mod(self.th1_ph+pi,2*pi)-pi,np.mod(self.th2_ph+pi,2*pi)-pi)
            
            #ax.set_xlim(-pi,pi)
            #ax.set_ylim(-pi,pi)
            ax.plot(np.mod(self.th1_ph+pi,2*pi)-pi,np.mod(self.th2_ph+pi,2*pi)-pi)

        elif option == 'h1_fourier':
            ax = fig.gca(projection='3d')
            ax.set_title("H1 (Fourier)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            #ax = plot_s(ax,self.XX,self.YY,self.J1)
            ax = plot_s(ax,f2d.H1_fourier(self.XX,self.YY))

        elif option == 'h2_fourier':
            ax = fig.gca(projection='3d')
            ax.set_title("H2 (Fourier)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            #ax = plot_s(ax,self.XX,self.YY,self.J1)
            ax = plot_s(ax,f2d.H2_fourier(self.XX,self.YY))


        elif option == 'h1_fourier_dx':
            ax = fig.gca(projection='3d')
            ax.set_title("dH1dx (Fourier)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            #ax = plot_s(ax,self.XX,self.YY,self.J1)
            dh1x,hd1y = f2d.H1_fourier(self.XX,self.YY,d=True)
            ax = plot_s(ax,dh1x)

        elif option == 'h1_fourier_dy':
            ax = fig.gca(projection='3d')
            ax.set_title("dH1dy (Fourier)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            #ax = plot_s(ax,self.XX,self.YY,self.J1)
            dh1x,hd1y = f2d.H1_fourier(self.XX,self.YY,d=True)
            ax = plot_s(ax,dh1y)

            
        elif option == 'h1_1d':
            ax = fig.add_subplot(111)
            ax.plot([-pi,pi],[0,0],color='black')

            ax.plot(self.X,-f2d.H1_fourier(self.X,self.X),label='-h1(x,x)',lw=3)
            ax.plot(self.X,-f2d.H1_fourier(-self.X,-self.X),label='-h1(-x,-x)',lw=3)
            ax.plot(self.X,-f2d.H1_fourier(self.X,-self.X),label='-h1(x,-x)',ls='--',lw=2)
            ax.plot(self.X,-f2d.H1_fourier(-self.X,self.X),label='-h1(-x,x)',ls='--',lw=2)

            ax.set_xlim(-pi,pi)
            
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles,labels)
            #ax.plot(self.X,f2d.H1_fourier(self.X,0))

        elif option == 'nullclines':
            
            # get contours
            # http://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour
            
            ax = fig.add_subplot(111)
            
            h1_x0_idx_vals = np.where(np.diff(np.sign(-f2d.H1_fourier(self.X,self.X))))[0]
            h1_x0_vals = self.X[h1_x0_idx_vals]
            
            t = np.linspace(0,100,1000)
            
            print h1_x0_vals
            
            for x0 in h1_x0_vals:
                # run sim, get solution
                sol = odeint(H_i_contour,[x0,x0],t)
                ax.plot(sol[:,0],sol[:,1])
            ax.plot(self.X,-f2d.H1_fourier(self.X,self.X))
            ax.set_xlim(-pi,pi)
            ax.set_ylim(-pi,pi)

            


        elif option == 'h1_centered_d':
            h1_dx,h1_dy = f2d.H1_fourier_centered(self.XX,self.YY,d=True)

            ax1 = fig.add_subplot(121,projection='3d')
            ax1.set_title("H1 dx (Fourier, centered)")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1 = plot_s(ax1,h1_dx)

            ax2 = fig.add_subplot(122,projection='3d')
            ax2.set_title("H1 dy (Fourier, centered)")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2 = plot_s(ax2,h1_dy)

            dx_val,dy_val = f2d.H1_fourier(0,0,d=True)
            
            print 'h1_fourier dx value at (0,0) =',dx_val
            print 'h1_fourier dy value at (0,0) =',dy_val

            dx_val,dy_val = f2d.H1_fourier(pi,pi,d=True)

            print 'h1_fourier dx value at (pi,pi) =',dx_val
            print 'h1_fourier dy value at (pi,pi) =',dy_val
            

        elif option == 'h1_fourier_d':
            h1_dx,h1_dy = f2d.H1_fourier(self.XX,self.YY,d=True)

            ax1 = fig.add_subplot(121,projection='3d')
            ax1.set_title("H1 dx (Fourier)")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1 = plot_s(ax1,h1_dx)

            ax2 = fig.add_subplot(122,projection='3d')
            ax2.set_title("H1 dy (Fourier)")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2 = plot_s(ax2,h1_dy)

            dx_val,dy_val = f2d.H1_fourier(0,0,d=True)
            
            print 'h1_fourier dx value at (0,0) =',dx_val
            print 'h1_fourier dy value at (0,0) =',dy_val

            dx_val,dy_val = f2d.H1_fourier(pi,pi,d=True)

            print 'h1_fourier dx value at (pi,pi) =',dx_val
            print 'h1_fourier dy value at (pi,pi) =',dy_val




        elif option == 'h2_fourier_d':
            h2_dx,h2_dy = f2d.H2_fourier(self.XX,self.YY,d=True)

            ax1 = fig.add_subplot(121,projection='3d')
            ax1.set_title("H2 dx (Fourier)")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1 = plot_s(ax1,h2_dx)

            ax2 = fig.add_subplot(122,projection='3d')
            ax2.set_title("H2 dy (Fourier)")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2 = plot_s(ax2,h2_dy)

            dx_val,dy_val = f2d.H2_fourier(0,0,d=True)
            
            print 'h2_fourier dx value at (0,0) =',dx_val
            print 'h2_fourier dy value at (0,0) =',dy_val

            dx_val,dy_val = f2d.H1_fourier(pi,pi,d=True)

            print 'h2_fourier dx value at (pi,pi) =',dx_val
            print 'h2_fourier dy value at (pi,pi) =',dy_val



        elif option == 'h2_fourier_dy':
            ax = fig.gca(projection='3d')
            ax.set_title("H1 dx (Fourier)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            #ax = plot_s(ax,self.XX,self.YY,self.J1)
            ax = plot_s(ax,f2d.H2_fourier(self.XX,self.YY,d=True))
            print 'h1_fourier dy value at (0,0) =',f2d.H2_fourier(0,0,d=True)
            print 'h1_fourier dy value at (pi,pi) =',f2d.H2_fourier(pi,pi,d=True)

        elif option == 'h1_approx':
            ax = fig.gca(projection='3d')
            ax.set_title("H1 approx_p. phase_option="+str(self.phase_option))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            #ax = plot_s(ax,self.XX,self.YY,self.J1)
            ax = plot_s(ax,self.h1_approx(self.XX,self.YY))

        elif option == 'h2_approx':
            ax = fig.gca(projection='3d')
            ax.set_title("H2 approx_p. phase_option="+str(self.phase_option))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            #ax = plot_s(ax,self.XX,self.YY,self.J1)
            ax = plot_s(ax,self.h2_approx_p(self.XX,self.YY))

            
        elif option == 'test':
            ax = fig.gca(projection='3d')
            ax.set_title("abs(x)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax = plot_s(ax,np.abs(self.XX))
        else:
            print 'option, ', option, ' not found.'

        return fig

def H_i_contour(z,t,i=1):
    x = z[0]
    y = z[1]
    
    # get derivatives of fourier
    if i == 1:
        h1x,h1y = f2d.H1_fourier(x,y,d=True)
        return -h1y,h1x
    elif i == 2:
        h2x,h2y = f2d.H2_fourier(x,y,d=True)
        return -h2y,h2x



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

    if run_phase:
        phase = Phase(x0=-2,x1=2.,y0=0,y1=0.0,
                      init_mode='cartesian',
                      q=0.,g=.989006,
                      dde_dt=.05,
                      dde_T=500,
                      phase_option='full',
                      recompute_h=False,recompute_j=False,
                      recompute_fq=False,recompute_phase_lc=False,
                      compute_h_error=False,
                      save_last=save_last,
                      use_last=use_last,
                  )
        
        #phase.plot("h1_fourier")
        #phase.plot("h2_fourier")

        #phase.plot('h1')
        #phase.plot('h2')

        #phase.plot('h1_approx2')
    
        #phase.plot("h1_fourier_d")
        #phase.plot("h1_fourier_centered_d")
        #phase.plot("h2_fourier_d")
        
        #phase.plot("nullclines")
        
        #phase.plot("h1_approx")
        #phase.plot("h2_approx")

        phase.plot("phase_time")
        phase.plot("phase_space")

        #phase.plot('evans')

        #phase.plot("h1_1d")

        #phase.plot('h1_fourier_dx')
        #phase.plot('h2_fourier_dx')
    
        #phase.plot("j1")
        #phase.plot("j2")
        
        #hettest = Heterogeneity()
        #hettest.plot()



    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
