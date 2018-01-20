"""
1d neural field model with mexican hat kernel.

we use this code to find good parameters for the 2d domain.

with no pinning, stability is determined by
\lambda = g*H'(0)/\kappa - \alpha,

where \alpha = 1/\mu, and \kappa = (u_0\',u^*)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import matplotlib.pylab as mp

pi = np.pi
cos = np.cos
sin = np.sin
exp = np.exp
sqrt = np.sqrt

periodization_lower = -6
periodization_upper = 6

# 1d version of 2d_full.jl to get reasonable parameter range for steady-state bump.
# i can also use this code to avoid the turing bifurcation, which results in
# multiple bump solutions.



def rhs(y_old,t,eps,mu,Ktable,Ivals,g,q,r,ut,N):
    """
    y_old: solution values of shape (2*N).
    First N entries are u, last N entries are z

    t: time
    eps: epsilon
    mu: time scaling for adaptation
    Kvals: kernel lookup table
    
    Ivals: pinning lookup table
    g,q,r,ut: parameters
    x,y: domain coordinates [0,2pi]?
    N: size of domain

    """
    output = np.zeros(2*N)
    u = y_old[:N]#reshape(y_old,1,N)
    z = y_old[N:]

    fu = f(u,r,ut)

    # for each element in square domain, update rhs
    Kfft = np.fft.fft(Ktable,len(fu))
    fufft = np.fft.fft(fu)
    wf = np.fft.ifft(Kfft*fufft)

    output[:N] = -u + wf + eps*( -g*z)
    output[N:] = eps*(u - z)/mu
    
    return output

def f(x,r,ut):
    """
    """
    return 1./(1+exp(-r*(x-ut)))

def K_diff(x,se,si):
    """
    difference of gaussians
    """
    A=1./(sqrt(pi)*se);B=1./(sqrt(pi)*si)
    return (A*exp(-(x/se)**2) -
            B*exp(-(x/si)**2))

def K_gauss(x,s):
    A=1./(sqrt(pi)*s)
    return A*exp(-(x/s)**2)

def K_gauss_p(x,s):
    tot = 0
    for n in range(periodization_lower,periodization_upper+1):
        tot = tot + K_gauss(x+n*2*pi,s)
    return tot
    

def K_diff_p(x,se,si):
    """
    (periodic version)
    ricker wavelet https://en.wikipedia.org/wiki/Mexican_hat_wavelet
    """
    tot = 0
    for n in range(periodization_lower,periodization_upper+1):
        tot = tot + K_diff(x+n*2*pi,se,si)
    return tot

# parms
eps=.005;mu=1.;g=0.;q=0.
#r=25;ut=0.25
r=25.;ut=.25
N = 100 # domain size/neuron indexing
m = 50
se=1.;si=2. # excitation/inhibition
Kamp = 5. # kernel amplitude
#sig = .3 # 3 middle bumps flanked by half bumps
sig = 1.2#.65

# time
t0=0;T=5000
dt = .1
TN = int(T/dt)
t = np.linspace(t0,T,TN)

xstart=0;xend=2*pi
X = np.linspace(xstart,xend,N)
X2 = np.linspace(0,2*pi,N+1)

# pinning
Ivals = 0

leave_footprint = False # dump all parameters, plots to file
#numerics = True # simulate full numerics
save_ss = False # save last bump solution
use_ss = False # use previous bump solution (like XPP command IL)
break_symm = True # break symmetry in traveling bump regime (for default initial condition only)
perturb_ss_file = False # perturb the saved steady-state file (sometimes leads to double bump solutions. why?)
plot_init = True
#phase = False # run phase estimation    
#plot = True # plot bump
#movie = False # save movie of bump


ss_file_u = 'bump_ss_1d_full/py_bump_ss_u_r'+str(r)+'_ut'+str(ut)+\
             '_se'+str(se)+'_si'+str(si)+'_mu'+str(mu)+\
             '_q'+str(q)+'_g'+str(g)+'_N'+str(N)+\
             '_dt'+str(dt)+'_T'+str(T)+'_eps'+str(eps)+'.dat'
ss_file_z = 'bump_ss_1d_full/py_bump_ss_z_r'+str(r)+'_ut'+str(ut)+\
             '_se'+str(se)+'_si'+str(si)+'_mu'+str(mu)+\
             '_q'+str(q)+'_g'+str(g)+'_N'+str(N)+\
             '_dt'+str(dt)+'_T'+str(T)+'_eps'+str(eps)+'.dat'


if (g > 0) and (break_symm):
    seed = 0
    factor = 2.
    np.random.seed(seed)
    break_val = (np.random.randn(N)-.5)/factor
    break_val2 = (np.random.randn(N)-.5)/factor
    break_label = "np.random.randn(N)/"+str(factor)+", seed="+str(seed)
else:
    break_val = 0
    break_val2 = 0
    break_label = "0"

## build kernel
# use this if i want to do int K(x-y)f(u(y))dy
print "building periodized kernel"

kernel_label = "k_diff_p"
# Ktable for rhs_brute

#Ktable = np.zeros((1,N))
Ktable = K_diff_p(X2,se,si)*Kamp*2*pi/N
#Ktable = K_gauss_p(X2,se)*Kamp*2*pi/N

## check kernel
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_title("periodized kernel "+kernel_label)
ax2.plot(X2,Ktable,lw=5)

#plt.show()

# solution matrix
y = np.zeros((TN,2*N))

# default inits
init_label = "y[0,:N] = Kvals_test; y[0,N:] = 0"
u0d = Kamp*K_diff_p(X-pi,se,si)
z0d = np.zeros(N)

if plot_init:
    fig69 = plt.figure()
    ax69 = fig69.add_subplot(111)
    ax69.set_title("initial condition")
    ax69.plot(X,u0d,lw=5)

if use_ss:
    # load steady-state files if they exist
    if os.path.isfile(ss_file_u) and os.path.isfile(ss_file_z):
        # load numerical steady-state conditions.
        if perturb_ss_file:
            y[0,:N] = np.loadtxt(ss_file_u) + break_val
            y[0,N:] = np.loadtxt(ss_file_z) + break_val2
        else:
            y[0,:N] = np.loadtxt(ss_file_u)
            y[0,N:] = np.loadtxt(ss_file_z)
    else:
        # else load default initial conditions
        y[0,:N] = u0d
        y[0,N:] = z0d
        print "init choice:",init_label
        #y[0,N:] = z0d
else:
    y[0,:N] = u0d
    y[0,N:] = z0d
    print "init choice:",init_label

fig = plt.figure()
ax = fig.add_subplot(111)

# run sim
print "starting sim"

for i in range(1,TN):
    y[i,:] = y[i-1,:] + dt*rhs(y[i-1,:],i*dt,eps,mu,
                               Ktable,Ivals,g,q,r,ut,N)

if save_ss:
    bump_ss_u = y[-1,:N]
    bump_ss_z = y[-1,N:]
    np.savetxt(ss_file_u,bump_ss_u)
    np.savetxt(ss_file_z,bump_ss_z)

#title("Surface Plot")
#println(size(X))
#println(size(y[end,1:N]))
ax.set_title("ss bump")
ax.plot(X,y[-1,:N],lw=5)

fig22 = plt.figure()
ax22 = fig22.add_subplot(111)
ax22.set_title("activity of first neuron over time")
ax22.plot(t,y[:,0],lw=5)

# get peak of bump
#print "TN",TN
max_arg = np.argmax(y[:,:N],axis=1)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.set_title("bump peak phase angle over time")
ax3.plot(t,np.mod(X[max_arg]+pi,2*pi)-pi,lw=5)

if leave_footprint:
    # increment footprint index
    ff = open("footprint/N.txt","r+")
    a = int(f.readline())
    a += 1
    ff.close()
    ff = open("footprint/N.txt","w")
    ff.write(str(a))
    ff.close()
    #print a,type(a), np.shape(a)
    #np.savetxt("footprint/N.txt",a)

    import time
    localtime = time.asctime( time.localtime(time.time()) )
    data_dir = "footprint/"+str(localtime)+"_"+str(a)
    if (not os.path.exists(data_dir)):
        os.makedirs(data_dir)
        
    # output different kernel params depending on kernel choice
    if kernel_choice == 0:
        parstr = "sig="+str(sig)
    elif kernel_choice == 1:
        parstr = "se="+str(se)+"\n"+"si="+str(si)
    
    data_output = "SIMULATION PARAMS\n"+\
                  "eps="+str(eps)+"\n"+\
                    "mu="+str(mu)+"\n"+\
                    "g="+str(g)+"\n"+\
                    "q="+str(q)+"\n"+\
                    "r="+str(r)+"\n"+\
                    "ut="+str(ut)+"\n"+\
                    "N="+str(N)+"\n"+\
                    "Ivals="+str(Ivals)+"\n"+\
                    "use_ss="+str(use_ss)+"\n"+\
                    "save_ss="+str(save_ss)+"\n"+\
                    "\n"+\
                    "TIME PARAMS/INIT\n"+\
                    "t0="+str(t0)+"\n"+\
                    "T="+str(T)+"\n"+\
                    "dt="+str(dt)+"\n"+\
                    "TN="+str(TN)+"\n"+\
                    "break_symm="+str(break_symm)+"\n"+\
                    "break_val="+str(break_label)+"\n"+\
                    "init:"+str(init_label)+"\n"+\
                    "\n"+\
                    "DOMAIN\n"+\
                    "xstart="+str(xstart)+"\n"+\
                    "xend="+str(xend)+"\n"+\
                    "N="+str(N)+"\n"+\
                    "\n"+\
                    "KERNEL PARAMS"+"\n"+\
                    "kernel_choice="+str(kernel_choice)+"\n"+\
                    parstr+"\n"+\
                    "periodization_lower="+str(periodization_lower)+"\n"+\
                    "periodization_upper="+str(periodization_upper)+"\n"
                         
    f = open(data_dir+"/param_dump.txt","w")
    f.write(data_output)
    f.close()

    

    # save figs
    fig.savefig(data_dir+"/ss_bump.png")
    fig2.savefig(data_dir+"/kernel.png")
    fig22.savefig(data_dir+"/first_neur_activity_over_time.png")
    fig3.savefig(data_dir+"/bump_peak_over_time.png")
    fig69.savefig(data_dir+"/init.png")

plt.show()

