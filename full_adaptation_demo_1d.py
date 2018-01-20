"""
generate movie of the effects of pinning and adaptation. side-by-side with centroid and full bump solution in heatmap form

avconv -r 40 -start_number 1 -i test%d.png -b:v 1000k test.mp4
"""

import matplotlib.pyplot as plt
import numpy as np
from sys import stdout

import oned_simple as oned





sin = np.sin
cos = np.cos
pi = np.pi
sqrt = np.sqrt
exp = np.exp



def plot_surface_movie(simdata,skip,movie=False,
               file_prefix="mov/test",
               file_suffix=".png",
               title="",scene_number=1):
    """
    take fully computed surface solution and display the plot over time
    X: domain
    sol: full solution with u,z for all time
    TN: number of time steps
    skip: number of time steps to skip per plot display
    """
    sim = simdata # relabel for convenience

    TN = len(simdata.t)
    
    #N = sol[0,0,:,0]
    lo = -3#lo = np.amin(np.reshape(sol[:,0,:,:],(TN*N*N)))
    hi = 5#hi = np.amax(np.reshape(sol[:,0,:,:],(TN*N*N)))


    #plt.ion()

    total_iter = int(TN/skip)-1
    start_iter = (scene_number-1)*total_iter
    end_iter = (scene_number)*total_iter

    label = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$",   r"$2\pi$"]
    j=0
    for i in range(total_iter):
        
        k = i*skip
        if (i <= 20) or ( (i >= 30) and (i <= 40)):

            c1='red'
            bg1 = 'yellow'
        else:

            c1 = 'black'
            bg1 = 'white'

        fig = plt.figure(figsize=(7,3.2))
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        #ax2.set_xlim(-pi,pi)
        #ax2.set_ylim(-pi,pi)

        plt.suptitle('g='+str(simdata.g)+'; q='+str(simdata.q),color=c1,backgroundcolor=bg1)
        #fig = plt.figure()        
        #ax.set_title("u")

        ax.set_title('Solution u')
        ax.text(pi/2.,1.5,'T='+str(np.around(simdata.t[k],decimals=0)))
        ax2.set_title('Centroid')

        ax.plot(simdata.domain,simdata.u[k,:],lw=3)

        LL = np.argmin(np.abs(sim.domain - np.mod(sim.ph_angle[k],2*np.pi)))
        ax.scatter(sim.domain[LL],sim.sol[k,LL],color='red',s=100)

        ax2.scatter(cos(sim.ph_angle[k]),sin(sim.ph_angle[k]),s=100,color='red') # centroid
        xx = np.linspace(-pi,pi,100)
        ax2.plot(cos(xx),sin(xx), color='black', lw=3) # circle

        ax.set_xticks(np.arange(0,2+.5,.5)*pi)

        ax.set_xticklabels(label, fontsize=15)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        ax.tick_params(axis='x', which='major', pad=5)
        ax2.tick_params(axis='x', which='major', pad=5)

        ax.set_ylim(-2,2)
        ax.set_xlim(0,2*pi)

        ax2.set_ylim(-1.2,1.2)
        ax2.set_xlim(-1.2,1.2)

        j = start_iter+i

        fig.savefig(file_prefix+str(j)+file_suffix)
        plt.cla()
        plt.close()
        
        stdout.write("\r Simulation Recapping... %d%%" % int((100.*(k+1)/len(simdata.t))))
        stdout.flush()
    print



def main():

    if False:

        simdata = oned.SimDat(q=0.,g=0.,
                              zshift=.0,
                              save_last=False,
                              use_last=False,
                              phase=False,
                              use_ss=False,
                              use_random=True,
                              T=10,eps=.0)

        skip = 2
        plot_surface_movie(simdata,skip,movie=False,scene_number=1)

    if True:
        ## scenes 1/2 with stationary bump and traveling bump (q=0, g=0 and g=3)
        zshift_angle=pi/5.;zshift_rad=.3

        print 'initial angle',zshift_angle,'inital rad',zshift_rad
        ushift1=0.;ushift2=0.
        zshift1=ushift1-zshift_rad*np.cos(zshift_angle);zshift2=ushift2-zshift_rad*np.sin(zshift_angle)
        ishift1=0.;ishift2=0.

        simdata = oned.SimDat(q=0.,g=3.,
                              zshift=.01,
                              save_last=True,
                              use_last=False,
                              phase=False,
                              T=1000,eps=.01)

        skip = 50
        plot_surface_movie(simdata,skip,movie=False,scene_number=1)

        simdata = oned.SimDat(q=1.,g=3.,
                              save_last=True,
                              use_last=True,
                              phase=False,
                              T=1000,eps=.01)


        plot_surface_movie(simdata,skip,movie=False,scene_number=2)


        simdata = oned.SimDat(q=1.,g=5.,
                              save_last=False,
                              use_last=True,
                              phase=False,
                              T=1000,eps=.01)

        plot_surface_movie(simdata,skip,movie=False,scene_number=3)

    

if __name__ == "__main__":
    main()
