"""
Nnumerical integration of neural field model + phase model.

L 137: figure out how to incorporate ss_numerical_u and ss_numerical_z cleanly.

Assume Cosine Kernel. ss bump classes/functions return bump amp and x,y shift. this assumption is invalid for non-cosine kernels.

execute this file in the working directory containing the directory of SS bump solution data

"""
import numpy as np;cos=np.cos;sin=np.sin;pi=np.pi
import matplotlib.pylab as mp
import os.path
#import scipy as sp
from scipy.integrate import odeint
import time
from colorsys import hsv_to_rgb
from scipy.linalg import blas as FB
from euler import *

np.random.seed(0)

# anim
from matplotlib import pyplot as plt
from matplotlib import animation

cos = np.cos
sin = np.sin
pi = np.pi

class Sim(object):
    """
    general simulation parameters.
    grid size N. N_idx=N**2
    filenames
    """
    def __init__(self,
                 N=240,a=0.,b=2*pi,
                 r=15.,ut=0.25,
                 mu=1.,
                 A=-.5,B=3.):

        # domain
        self.N = N
        self.a = a # lower domain boundary
        self.b = b # upper domain boundary

        self.A = A
        self.B = B

        self.r = r
        self.ut = ut
        self.mu = mu
        self.domain = np.linspace(a,b*(1-1./N),N)
        #self.domain = np.linspace(a,b,N)


    def f(self,x,d=False):
        """
        d: derivative flag
        """
        if d:
            a = np.exp(-self.r*(x-self.ut))
            return (self.r*a)/(1.+a)**2
        else:
            return 1./(1.+np.exp(-self.r*(x-self.ut)))

        
class SteadyState(Sim):
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
                 g=0.,q=0.,
                 ss_dt=.1,ss_t0=0,ss_T=5000,
                 eps=0.005,
                 ss0=-69,ss1=-69,ss_shift=0.,
                 Nkap=200,
                 kernel_factor=1.
             ):
        # defaults

        Sim.__init__(self)

        self.kernel_factor = kernel_factor
        self.display_params = display_params
        self.g = g
        self.q = q
        self.ss_dt = ss_dt
        self.ss_t0 = ss_t0
        self.ss_T = ss_T
        self.ss_TN = int(self.ss_T/self.ss_dt)
        self.ss_t = np.linspace(self.ss_t0,self.ss_T,self.ss_TN)
        self.eps = eps

        self.ss0 = ss0
        self.ss1 = ss1
        self.ss_shift = ss_shift

        self.recompute_ss = recompute_ss
        self.use_ss = use_ss
        self.save_ss_bump = save_ss_bump

        self.dir1 = 'bump_ss_1d_simple'
        self.ss_file_prefix = 'mu='+str(self.mu)+'_A='+str(self.A)+'_B='+str(self.B)+'_r='+str(self.r)+'_ut='+str(self.ut)+'_eps='+str(self.eps)+'_q='+str(self.q)+'_g='+str(self.g)+'_N='+str(self.N)+'_domain_left='+str(self.a)+'_domain_right='+str(self.b)+'_T='+str(self.ss_T)+'_dt='+str(self.ss_dt)
        self.parfile = self.ss_file_prefix + '_params.dat' # for ss parameters
        self.datfile_u = self.ss_file_prefix + '_ss_u.dat' # for ss data 
        self.datfile_z = self.ss_file_prefix + 'ss_z.dat' # for ss data 
        
        if (not os.path.exists(self.dir1)):
            os.makedirs(self.dir1)

        if self.g > 0 and self.break_symm:
            self.break_val = (np.random.randn(N_idx)-.5)
        else:
            self.break_val = 0

        self.Ivals = 0.


        # if recompute, run ss function and save
        # else if ss file exists, load to self.c0,self.c1
        # else, recompute and save.
        if recompute_ss:
            self.ss0, self.ss1, self.ss_shift = self.get_ss()
        else:
            # check if ss solution exists
            if os.path.isfile(self.dir1+'/'+self.parfile) and os.path.isfile(self.dir1+'/'+self.datfile_u) and os.path.isfile(self.dir1+'/'+self.datfile_z):
                # load ss solution if exists
                self.ss0,self.ss1,self.ss_shift = np.loadtxt(self.dir1+'/'+self.parfile)
                self.ss_numerical_u = np.loadtxt(self.dir1+'/'+self.datfile_u)
                self.ss_numerical_z = np.loadtxt(self.dir1+'/'+self.datfile_z)
            else:
                self.ss0, self.ss1, self.ss_shift = self.get_ss()
        #print 'ss values', '; ss0=',self.ss0, '; ss1=', self.ss1, '; ss_shift=',self.ss_shift

        self.Ivals = self.I(self.domain)

        # estimate inner produt kappa = (u_0',u^*)
        self.kap = self.kappa_estimate()

        # get H function amplitude (assume it is sine function)
        self.Hamp,self.H_numerical = self.get_H(return_data=True)
        
        # get J function parameters
        self.i0,self.i1,self.ishift,self.J_numerical = self.get_J(return_data=True)



    def rhs2(self,y,t,sim_factor=1.):
        """
        diffeq for full equation on interval [0,2pi]
        y: (solution estimate u,z)
        A,B: parms for K
        r,ut: parms for F
        q,g: parms for rhs
        """
        
        dy = np.zeros(2*self.N)
        
        u=y[:self.N];z=y[self.N:]
        fu = 1./(1.+np.exp(-self.r*(u-self.ut)))
        
        # new implementation with trig identities:
        wf0 = self.A*np.sum(fu)
        wf1 = self.B*cos(self.domain)*np.sum(cos(self.domain)*fu)
        wf2 = self.B*sin(self.domain)*np.sum(sin(self.domain)*fu)
        w = self.kernel_factor*(wf0 + wf1 + wf2)/self.N#(wf0 + wf1 + wf2)*(domain[-1]-domain[0])/N
        
        #dy[:N] = -u + w + eps*(q*Ivals - g*z)
        dy[:self.N] = sim_factor*(-u + w + self.eps*(self.q*self.Ivals- self.g*z))
        dy[self.N:] = sim_factor*(self.eps*(-z + u)/self.mu)
        
        return dy


    def rhs3(self,y,t,sim_factor=1.):
        """
        diffeq for full equation on interval [0,2pi]
        y: (solution estimate u,z)
        A,B: parms for K
        r,ut: parms for F
        q,g: parms for rhs
        """
        
        dy = np.zeros(2*self.N)
        
        u=y[:self.N];z=y[self.N:]
        fu = 1./(1.+np.exp(-self.r*(u-self.ut)))
        
        # new implementation with trig identities:
        wf0 = self.A*np.sum(fu)
        wf1 = self.B*cos(self.domain)*np.sum(cos(self.domain)*fu)
        wf2 = self.B*sin(self.domain)*np.sum(sin(self.domain)*fu)
        wf3 = self.B*cos(3*self.domain)*np.sum(cos(3*self.domain)*fu)
        wf4 = self.B*sin(3*self.domain)*np.sum(sin(3*self.domain)*fu)
        w = self.kernel_factor*(wf0 + wf1 + wf2 + wf3 + wf4)/self.N#(wf0 + wf1 + wf2)*(domain[-1]-domain[0])/N
        
        #dy[:N] = -u + w + eps*(q*Ivals - g*z)
        dy[:self.N] = sim_factor*(-u + w + self.eps*(self.q*self.Ivals- self.g*z))
        dy[self.N:] = sim_factor*(self.eps*(-z + u)/self.mu)
        
        return dy


    def oned_equivalent():
        """
        equivalent diffeq for full equation on interval [0,2pi]        
        
        """
        pass

    def I(self,x,use_ss=True):
        if use_ss:
            return self.u0b(x)
        else:
            return cos(x)

    def get_ss(self,return_data=False):
        """
        compute steady-state bump. always saves computed bump params (ignores numerics).
        """
        u0 = np.cos(self.domain)
        z0 = np.zeros(self.N)
        init = np.append(u0,z0)
        Ivals = self.I(self.domain)

        sol = ESolve(self.rhs2,init,self.ss_t)
        #sol = ESolve(self.rhs3,init,self.ss_t)

        self.ss_numerical_u = sol[-1,:self.N]
        self.ss_numerical_z = sol[-1,self.N:]

        # WLOG(?) use shifted sin: A*sin(x-c)
        peak_idx = np.argmax(self.ss_numerical_u) # solution peak idx

        # create cos function based on data
        ss0 = (np.amax(self.ss_numerical_u)+np.amin(self.ss_numerical_u))/2
        ss1 = (np.amax(self.ss_numerical_u)-np.amin(self.ss_numerical_u))/2

        # compare created sin function to data to get shift (of period 2pi)
        ss_shift = (np.argmax(ss0+ss1*cos(self.domain)) - peak_idx)*2.*pi/self.N
        
        # save data
        np.savetxt(self.dir1+'/'+self.parfile,(ss0,ss1,ss_shift))
        np.savetxt(self.dir1+'/'+self.datfile_u,self.ss_numerical_u)
        np.savetxt(self.dir1+'/'+self.datfile_z,self.ss_numerical_z)        
        
        if return_data:
            return self.ss_numerical_u,self.ss_numerical_z
        else:
            #self.ss0,self.ss1,self.ss_shift
            return ss0,ss1,ss_shift

    def u0b(self,x,d=False):
        """
        steady-state bump
        """
        if d:
            # derivative of bump
            return -self.ss1*sin(x+self.ss_shift)
        else:
            return self.ss0 + self.ss1*cos(x+self.ss_shift)

    def kappa_estimate(self):
        """
        check value of inner product (u_0',u^*).
        (u_0',u^*) = \int_0^{2\pi} u_0'(x) f'(u_0(x)) u_0'(x) dx
        I claimed this inner product is 1 in eq 59 log_youngmin.pdf. Generically it is not.

        ss0,ss1,ss_shift: ss bump solution
        """        
        tot = 0
        for i in range(self.N):
            tot += self.f(self.u0b(self.domain[i]),d=True)*\
                   self.u0b(self.domain[i],d=True)*\
                   self.u0b(self.domain[i],d=True)
        tot /= self.N

        #print 'value of inner product (u_0\',u^*):', tot
        return tot

    def get_H(self,return_data=False):
        """
        x: x \in [0,2pi]
        a1: amplitude of H (sine) function
        ss0,ss1,ss_shift: params of steady-state bump solution (get using u0b2)
        params: all other parameters for ss bump and f
        plotH: plot H function with estimated amplitude
        plotbump: plot ss-bump solution
        
        if parameters are put into the function, then an estimate of H is generated.
        if no parameters are put into the function, return a1*sin(x).
        ======================
        
        H is generically odd.  typically  a1*sin(x).
        
        (does not depend explicity/directly on kernel K, so parms A,B are not needed)
        
        r,ut are params for f. only used if params != None
        """

        ## plot H
        H_numerical = np.zeros(self.N)
        for k in range(self.N):
            tot = 0
            for j in range(self.N):
                tot += self.f(self.u0b(self.domain[j]),d=True)*\
                       self.u0b(self.domain[j],d=True)*\
                       self.u0b(self.domain[j]+self.domain[k]) # pg 175 Notebook#2

            H_numerical[k] = tot/self.N

        # return amplitude
        amp = np.amax(H_numerical)
        #print 
        #print "H(x) amplitude error:", np.amax(H) - np.abs(np.amin(H))
        #print 'H(x) parameter (amplitude) a1 =',amp
        if return_data:
            return amp,H_numerical
        else:
            return amp

    def H(self,x):
        return self.Hamp*sin(x)

    def get_J(self,return_data=False):
        J_numerical = np.zeros(self.N)
        for k in range(self.N):
            tot = 0
            for j in range(self.N):
                """
                tot += self.f(self.u0b(self.domain[k]+self.domain[j]),d=True)*\
                       self.u0b(self.domain[k]+self.domain[j],d=True)*\
                       self.I(self.domain[j]) # pg 199 Notebook#2
                """
                tot += self.f(self.u0b(self.domain[k]+self.domain[j]),d=True)*\
                       self.u0b(self.domain[k]+self.domain[j],d=True)*\
                       self.u0b(self.domain[j])
            J_numerical[k] = tot/self.N
        
        # create cos function based on data
        peak_idx = np.argmax(J_numerical) # solution peak idx
        i0 = (np.amax(J_numerical)+np.amin(J_numerical))/2
        i1 = (np.amax(J_numerical)-np.amin(J_numerical))/2

        # compare created cos function to data to get shift (out of period 2pi)
        ishift = (np.argmax(i0+i1*cos(self.domain)) - peak_idx)*2.*pi/self.N
        #print 
        #print 'i(x) parameters i0 =',i0, ';i1 =', i1, ';ishift =', ishift

        if return_data:
            return i0,i1,ishift,J_numerical
        else:
            return i0,i1,ishift

    def J(self,x):
        return self.i0+self.i1*cos(x+self.ishift)

    def params(self):
        """
        dump all params
        """
        print 
        print 'STEADY-STATE STABLE BUMP WITH PARAMTERS:'
        print 'mu=',self.mu, ';A=',self.A,';B=',self.B,';r=',self.r
        print 'ut=',self.ut, ';eps=',self.eps, ';q=',self.q
        print 'g=',self.g, ';N=',self.N, ';domain_left=',self.a, ';domain_right=',self.b
        print 'ss_T=',self.ss_T, ';ss_dt=',self.ss_dt, ';ss0=',self.ss0,';ss1=',self.ss1,';ss_shift=',self.ss_shift
        print 'Hamp=',self.Hamp, ';i0=',self.i0, ';i1=',self.i1,';ishift=',self.ishift
        print 'kap=',self.kap

    def plot(self,option='ss'):
        """
        option: 'ss', 'J', or 'H'
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if option == 'ss':
            ax.set_title('SS bump solution numerics (blue) vs theory (green). ss bump deriv (red)')
            ax.plot(self.domain,self.ss_numerical_u)
            ax.plot([self.domain[0],self.domain[-1]],[self.ut,self.ut])
            ax.plot(self.domain,self.u0b(self.domain))
            ax.plot(self.domain,self.u0b(self.domain,d=True))
    
        elif option == 'J':
            ax.set_title('J numerics (blue) vs ansatz (green)')
            ax.plot(self.domain,self.J_numerical)
            ax.plot(self.domain,self.i0+self.i1*cos(self.domain+self.ishift))
            #ax.plot([0,2*pi],[0,0],ls='--')

        elif option == 'H':
            ax.set_title('H numerics (blue) vs ansatz (green)')
            ax.plot(self.domain,self.H_numerical)
            ax.plot(self.domain,self.Hamp*sin(self.domain))
        
        return fig


class SimDat(SteadyState):
    def __init__(self,
                 ushift=0.,zshift=0.,
                 g=0.,q=0.,
                 dt=.05,t0=0,T=5000,
                 eps=0.01,
                 display_params=True,
                 phase=False,
                 save_ss=False,
                 use_ss=True,
                 save_last=False,
                 use_last=False,
                 use_random=False,
                 sim_factor=1.,
                 kernel_factor=1.
             ):

        """
        save_last: save last value of current sim
        use_last: use last value of previous sim
        use_ss: use steady-state bump as init
        use_random: use random initial conditions
        """

        SteadyState.__init__(self,kernel_factor=kernel_factor)

        
        # multiply simulations by x sim_factor
        self.sim_factor = sim_factor

        self.save_ss = save_ss
        self.save_last = save_last
        self.use_last = use_last
        self.use_random = use_random
        self.use_ss = use_ss

        self.eps = eps
        self.t0 = t0
        self.T = T
        self.dt = dt
        self.q = q
        self.g = g

        self.ushift = ushift
        self.zshift = zshift

        self.t = np.linspace(self.t0,self.T,int(self.T/self.dt))

        # filenames
        self.last_file_prefix = 'mu='+str(self.mu)+'_A='+str(self.A)+'_B='+str(self.B)+'_r='+str(self.r)+'_ut='+str(self.ut)+'_eps='+str(self.eps)+'_q='+str(self.q)+'_g='+str(self.g)+'_N='+str(self.N)+'_domain_left='+str(self.a)+'_domain_right='+str(self.b)+'_T='+str(self.T)+'_dt='+str(self.dt)

        #self.last_file_u = self.dir1+'/'+self.last_file_prefix + '_last_u.dat'
        #self.last_file_z = self.dir1+'/'+self.last_file_prefix + '_last_z.dat'

        self.last_file_u = self.dir1+'/'+'last_u.dat'
        self.last_file_z = self.dir1+'/'+'last_z.dat'

        self.filename_u = self.last_file_u
        self.filename_z = self.last_file_z
        
        self.phase = phase # run or do not run phase eqns

        # default solutions - set as ss, encourage bumps by shifting z bump
        # later add non-default solutions.
        
        self.run_full_sim()

        # get center of mass of bump solution
        cs = np.cos(self.domain)
        sn = np.sin(self.domain)

        # cosine/sine phase angle
        #cu = np.sum(cs*self.sol[:,:self.N],axis=1)
        #su = np.sum(sn*self.sol[:,:self.N],axis=1)

        self.cu = np.sum(cs*self.sol[:,:self.N],axis=1)
        self.su = np.sum(sn*self.sol[:,:self.N],axis=1)
        
        # get last position of z coordinate
        self.cz = np.sum(cs*self.sol[-1,self.N:])
        self.sz = np.sum(sn*self.sol[-1,self.N:])

        # center of mass
        self.ph_angle = np.arctan2(self.su,self.cu)
        self.ph_angle_z = np.arctan2(self.sz,self.cz)
        
        # save bump initial condition for later
        # save_ss overwrites because longer times lead to better ss values
        if self.save_last:# and not(os.path.isfile(ss_file_u) and os.path.isfile(ss_file_u)):
            # get final bump values. ss for long time
            self.bump_last_u = self.sol[-1,:self.N]
            self.bump_last_z = self.sol[-1,self.N:]
            np.savetxt(self.last_file_u,self.bump_last_u)
            np.savetxt(self.last_file_z,self.bump_last_z)

        #np.savetxt("chaos_simple1.dat",self.ph_angle)
        #self.compare = np.loadtxt("chaos_simple.dat")

        ## solve short phase estimation
        # domain on [0,2pi]
        # get initial conditions from numerics (replace the 3 lines below)
        
        self.c_num = (self.ph_angle[-1]-self.ph_angle[-2])/self.dt
        th0 = self.ph_angle[0]
        I10 = cos(th0)*self.mu/(1.-self.c_num**2)
        I20 = -cos(th0)*(self.c_num*self.mu**2)/((self.c_num*self.mu)**2+1)
        
        y0 = np.array([th0,I10,I20])
        #time.sleep(60)
        self.solph= odeint(self.phase_rhs_short,y0,self.t)

        # wave speed
        if self.q == 0.:
            self.c_theory_eqn = self.eps*np.sqrt(self.g*self.Hamp/(self.mu*self.kap)-(1./self.mu)**2)
            self.c_theory_num = np.abs(self.solph[-1,0]-self.solph[-2,0])/self.dt
        else:
            self.c_theory_eqn = -69
            self.c_theory_num = -69

    def run_full_sim(self):
        """
        run the sim
        and define initial conditions
        """


        self.init = np.zeros(2*self.N)
        #self.init[0,:,:] = self.u0ss + self.break_val

        file_not_found = False
        while True:
            if self.use_last and not(file_not_found):
                if os.path.isfile(self.filename_u) and\
                   os.path.isfile(self.filename_z):
                    print 'using last'
                    self.init[:self.N] = np.loadtxt(self.filename_u)
                    self.init[self.N:] = np.loadtxt(self.filename_z)
                    break
                else:
                    print 'init file not found'
                    file_not_found = True
            elif self.use_ss:
                print 'using initial function ss'
                #init = np.append(self.u0d,self.z0d)
                
                self.u0d = self.u0b(self.domain+self.ushift)
                self.z0d = self.u0b(self.domain+self.zshift)
                
                self.init[:self.N] = self.u0d
                self.init[self.N:] = self.z0d


                break
            else:
                print 'using random initial function'

                np.random.seed(0)
                self.u0d = np.random.randn(len(self.domain))/5.+.2
                self.z0d = np.random.randn(len(self.domain))/5.+.2
                
                self.init[:self.N] = self.u0d
                self.init[self.N:] = self.z0d

                break


        self.sol = odeint(self.rhs2,self.init,self.t,args=(self.sim_factor,))
        self.u = self.sol[:,:self.N]
        self.z = self.sol[:,self.N:]

            
    def params(self):
        """
        dump all params
        """
        print 
        print 'SIMULATION PARAMTERS:'
        print 'mu=',self.mu, ';A=',self.A,';B=',self.B,';r=',self.r
        print 'ut=',self.ut, ';eps=',self.eps, ';q=',self.q
        print 'g=',self.g, ';N=',self.N, ';domain_left=',self.a, ';domain_right=',self.b
        print 'T=',self.T, ';dt=',self.dt
        print 'ushift=',self.ushift,';zshift=',self.zshift
        if self.q == 0.:
            print 'c_num=',self.c_num, ';c_theory_eqn=',self.c_theory_eqn, ';c_theory_num=',self.c_theory_num

    
    def phase_rhs_short(self,y,t):
        """
        truncated phase model approximation (equivalent to above)
        derivation in pg 190-191 nb#2
        
        y: [theta,I_1,I_2], th \in [0,2pi]
        t: time
        mu,q,g: full sim parms
        Hamp: amplitude of H: H=A*sin(x)
        
        """
        
        # for readability
        th=y[0];I1=y[1];I2=y[2];A=self.Hamp

        rhs_th = (self.q*self.J(th)+self.g*A*( I2*sin(th)-I1*cos(th) )/self.mu)/self.kap 
        # 1/mu term in 2/2/2015 log_youngmin.pdf

        rhs_I1 = -I1/self.mu + sin(th)
        rhs_I2 = -I2/self.mu + cos(th)
        
        return self.eps*np.array([rhs_th,rhs_I1,rhs_I2])

    def oned_normal_form(self):
        """
        normal form calculations
        """
        


    def plot(self,option='phase_angle'):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('theory (green) vs numerics (black),g='+str(self.g)+',q='+str(self.q)+',eps='+str(self.eps)+',mu='+str(self.mu))

        if option == 'phase_angle':
            ax.set_xlabel('t')
            ax.set_ylabel(r'$\theta$')
            #ax.plot(self.t,np.mod(self.ph_angle+pi,2*pi)-pi,lw=3,color='black')
            ax.plot(self.t,self.ph_angle,lw=3,color='black')
            #ax.plot(self.t,self.compare,lw=2,color='red')
            if self.phase:
                ax.plot(self.t,-(np.mod(self.solph[:,0]+pi,2*pi)-pi),lw=3,color='green')
        elif option == 'theory':
            if self.phase:
                ax.plot(self.t,-(np.mod(self.solph[:,0]+pi,2*pi)-pi),lw=3,color='green')
            
        return fig




def main():
    
    """
    ss = SteadyState(recompute_ss=False)
    ss.params()
    ss.plot("ss")
    ss.plot("J")
    ss.plot("H")
    """

    ## solve full system
    # for chaos, use g=3.054,q=0.5 ???
    # for chaos, use g=2.65,q=0.5 (numerics)
    # for chaos g=2.661, q=0.5 (theory)
    sim = SimDat(g=2.65,q=.5,zshift=.01,phase=True,T=50000,kernel_factor=1.)
    sim.plot('phase_angle')
    #sim.plot('theory')
    sim.params()
    print sim.Hamp

    if True:

        # see readme.txt for info on data files
        #np.savetxt("chaos_simple_theory1.dat",np.mod(sim.solph[:,0]+pi,2*pi)-pi)
        np.savetxt("chaos_simple1_N="+str(sim.N)+".dat",sim.ph_angle)

        pass
    
    #plt.show()

    # show movie of full sim
    if False:
        #print elapsed, 'time elapsed'    
        #out = np.savetxt('u0.txt',sol[int(T/dt)-1,:N])
        # plot
        # clean this up, maybe write a dedicated movie script
        #mp.plot(t,sol[:,0])
        #mp.title('u0')
        #mp.ylabel('u0')
        #mp.show()
        
        fig = plt.figure(figsize=(11,5))
        plt.ion()
        plt.show()
        g1 = fig.add_subplot(121)
        g2 = fig.add_subplot(122)

        movdir = 'mov'

        sim.sol
        sim.cu
        sim.su
        
        for j in range(len(sim.t)):
            k = j*300
            #g1.matshow(np.reshape(sol[k,:N],(rN,rN)))
            g1.set_title("Solutions u (blue), z (green)")
            g1.plot(sim.domain,sim.sol[k,sim.N:],lw=3,color='green')
            g1.plot(sim.domain,sim.sol[k,:sim.N],lw=3,color='blue')
            LL = np.argmin(np.abs(sim.domain - np.mod(sim.ph_angle[k],2*np.pi)))
            g1.scatter(sim.domain[LL],sim.sol[k,LL],color='red',s=100)
            #print ph_angle[k]
            g1.set_xlabel('Domain')
            g1.set_ylabel('Activity')
            
            g1.set_ylim(-2,2)
            g1.set_xlim(0,2*pi)
            g1.text(.5,9,"t="+str(sim.t[k]))
            g1.text(.5,8,"g="+str(sim.g)+", eps="+str(sim.eps)+", q="+str(sim.q))
            
            g2.set_title("Peak Location on Unit Circle")
            g2.scatter(cos(sim.ph_angle[k]),sin(sim.ph_angle[k]),s=100,color='red')
            xx = np.linspace(-pi,pi,100)

            g2.plot(cos(xx),sin(xx), color='black', lw=3)
            g2.set_ylim(-1.2,1.2)
            g2.set_xlim(-1.2,1.2)
            
            #fig.savefig(movdir+"/test"+str(j)+".jpg")
            
            plt.pause(.01)
            print sim.t[k], 'of max t =',sim.t[-1]
            g1.clear()
            g2.clear()



    """
    fig = plt.figure(1)
    ax = plt.axes(xlim=(0, 1), ylim=(-5, 5))
    #ax = plt.axes()
    line, = ax.plot([], [], lw=2)
    #mp.ion()
    #mp.show()
    def init():
        line.set_data([], [])
        return line,
    def animate(j):
        #rgb = hsv_to_rgb( *(1.*j/Nt,1,1) )
        line.set_data(domain,sol[j,:len(u0)])
        #line.color(rgb)
        return line,
        
    #anim = animation.FuncAnimation(fig, animate, np.arange(1,len(sol[:len(sol[:,0]/2),0])), blit=True,interval=1)
    #anim = animation.FuncAnimation(fig, animate, np.arange(1,len(sol[:len(sol[:,0]/2),0])))
    anim = animation.FuncAnimation(fig, animate, len(sol[:len(sol[:,0]/2),0]), init_func=init, interval=20, blit=True)
    #for j in range(len(sol[:len(sol[:,0]/2),0])):

    mywriter = animation.AVConvWriter()
    #anim.save('basic_animation.mp4', extra_args=['-vcodec', 'libx264'],writer=mywriter)
    anim.save('basic_animation.mp4', fps=15,  writer=mywriter)
    """
    #mp.figure()
    #mp.plot(sol[:,0])
    #mp.plot(sol[0,:len(u0)])
    #mp.plot(sol[int(.1*T/dt),:len(u0)])
    #mp.plot(sol[int(.5*T/dt),:len(u0)])
    #mp.plot(sol[int(T/dt)-100,:len(u0)])
    #mp.plot(sol[int(T/dt)-1,:len(u0)])
    plt.show()
        
    #rgb[0]/=255.;rgb[1]/=255.;rgb[2]/=255.
        

if __name__ == "__main__":
    main()
