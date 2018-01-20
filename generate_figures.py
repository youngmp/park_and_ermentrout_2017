"""
Run to generate figures
Requires TeX; may need to install texlive-extra-utils on linux
Requires xppy and Py_XPPCall

the main() function at the end calls the preceding individual figure functions.

figures are saved as both png and pdf.

Copyright (c) 2016, Youngmin Park, Bard Ermentrout
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

# last compiled using python 2.7.6
# numpy version 1.8.2
# scipy version 0.13.3
# matplotlib version 1.3.1

import os
from sys import stdout
import numpy as np
import scipy as sp
import matplotlib
import copy
#from matplotlib.ticker import MultipleLocator
#import matplotlib.ticker as mticker

import matplotlib.colors as colors
from matplotlib import pyplot as plt
import matplotlib.pylab as mp
#import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import proj3d
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import axes3d

# 3d plotting is generated in twod_full_square.py, then beautified in this file.
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', serif=['Computer Modern Roman'])

matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{bm} \usepackage{xcolor} \setlength{\parindent}{0pt}']
matplotlib.rcParams.update({'figure.autolayout': True})

sizeOfFont = 20
fontProperties = {'weight' : 'bold', 'size' : sizeOfFont}


from scipy.interpolate import interp1d

import oned_simple
import fourier_2d as f2d
import twod_full as twod
import twod_phase as twodp
from xppy.utils import diagram #get xppy at https://github.com/jsnowacki/xppy
from xppcall import xpprun # get xppcall at https://github.com/youngmp/Py_XPPCALL
from lib import *


cos = np.cos
Cos = np.cos
sin = np.sin

Sin = np.sin
pi = np.pi;Pi=pi
sqrt = np.sqrt

Sqrt = np.sqrt
exp = np.exp
erfc = sp.special.erfc;Erfc=erfc

erf = sp.special.erf;Erf=erf
E = np.exp(1)#2.7182818284590452353602874713527
cosh = np.cosh;Cosh=cosh

arcsin = np.arcsin

x_label = [r"$\bm{-\pi$}", r"$\bm{-\frac{\pi}{2}}$", r"$\bm{0}$", r"$\bm{\frac{\pi}{2}}$",   r"$\bm{\pi}$"]
#x_label = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$",   r"$\pi$"]
x_label_short = [r"$\bm{-\pi}$", r"$\bm{0}$", r"$\bm{\pi}$"]

x_label2 = [r"$\bm{-\frac{\pi}{2}}$", r"$\bm{0}$", r"$\bm{\frac{\pi}{2}}$"]

blue2='#0066B2'
labelbg='#CCFF66'

AA = r'$\mathbf{A}$'
BB = r'$\mathbf{B}$'
CC = r'$\mathbf{C}$'
DD = r'$\mathbf{D}$'
EE = r'$\mathbf{E}$'
FF = r'$\mathbf{F}$'

formatter = matplotlib.ticker.FormatStrFormatter(r'$\mathbf{%g}$')


cmap = 'winter'
cmapmin = 1.
cmapmax = 0.

class MyAxes3D(axes3d.Axes3D):

    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.w_zaxis.get_ticklines() + self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        # set visibility of some features False 
        self.set_some_features_visibility(False)
        # draw the axes
        super(MyAxes3D, self).draw(renderer)
        # set visibility of some features True. 
        # This could be adapted to set your features to desired visibility, 
        # e.g. storing the previous values and restoring the values
        self.set_some_features_visibility(True)

        zaxis = self.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        # disable draw grid
        zaxis.axes._draw_grid = False

        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw :
            # draw zaxis on the left side
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw :
            # draw zaxis on the right side
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2], 
                             tmp_planes[1], tmp_planes[0], 
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes

        # disable draw grid
        zaxis.axes._draw_grid = draw_grid_old



def collect(x,y,use_nonan=True,lwstart=1.,lwend=5.,zorder=1.,cmapmax=1.,cmapmin=0.,cmap='copper'):
    """
    add desired line properties
    """
    x = np.real(x)
    y = np.real(y)
    
    x_nonan = x[(~np.isnan(x))*(~np.isnan(y))]
    y_nonan = y[(~np.isnan(x))*(~np.isnan(y))]
    
    if use_nonan:
        points = np.array([x_nonan, y_nonan]).T.reshape(-1, 1, 2)
    else:
        points = np.array([x, y]).T.reshape(-1, 1, 2)


    lwidths = np.linspace(lwstart,lwend,len(x_nonan))

    cmap = plt.get_cmap(cmap)
    #my_cmap = truncate_colormap(cmap,gshift/ga[-1],cmapmax)
    my_cmap = truncate_colormap(cmap,cmapmin,cmapmax)

    
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, linewidths=lwidths,cmap=my_cmap, norm=plt.Normalize(0.0, 1.0),zorder=zorder)
    
    #points = np.array([x, y]).T.reshape(-1, 1, 2)
    #segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    #lc = LineCollection(segments, cmap=plt.get_cmap('copper'),
    #                    linewidths=1+np.linspace(0,1,len(x)-1)
    #                    #norm=plt.Normalize(0, 1)
    #)
    
    lc.set_array(np.sqrt(x**2+y**2))
    #lc.set_array(y)
    
    return lc


def collect3d(v1a,ga,v2a,use_nonan=True):
    """
    set desired line properties
    """
    
    v1a = np.real(v1a)
    ga = np.real(ga)
    v2a = np.real(v2a)
    
    # remove nans for linewidth stuff later.
    ga_nonan = ga[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    v1a_nonan = v1a[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    v2a_nonan = v2a[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    
    
    if use_nonan:
        sol = np.zeros((len(ga_nonan),3))
        sol[:,0] = v1a_nonan
        sol[:,1] = ga_nonan
        sol[:,2] = v2a_nonan
    else:
        sol = np.zeros((len(ga),3))
        sol[:,0] = v1a
        sol[:,1] = ga
        sol[:,2] = v2a
        
    
    sol = np.transpose(sol)
    
    points = np.array([sol[0,:],sol[1,:],sol[2,:]]).T.reshape(-1,1,3)
    segs = np.concatenate([points[:-1],points[1:]],axis = 1)
    line3d = Line3DCollection(segs,linewidths=(1.+(v1a_nonan)/(.001+np.amax(v1a_nonan))*6.),colors='k')
    
    return line3d



def collect3d_colorgrad(v1a,ga,v2a,use_nonan=True,
                        lwstart=1.,
                        lwend=5.,
                        zorder=1.,
                        cmapmin=0.,
                        cmapmax=1.,
                        cmap='copper',
                        return3d=True):
    """
    set desired line properties. with color gradient. and width denotes g value
    """

    v1a = np.real(v1a)
    ga = np.real(ga)
    v2a = np.real(v2a)
    
    # remove nans for linewidth stuff later.
    ga_nonan = ga[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    v1a_nonan = v1a[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    v2a_nonan = v2a[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    

    lwidths = np.linspace(lwstart,lwend,len(ga_nonan))

    assert(len(lwidths) > 0)

    cmap = plt.get_cmap(cmap)
    #my_cmap = truncate_colormap(cmap,gshift/ga[-1],cmapmax)
    my_cmap = truncate_colormap(cmap,cmapmin,cmapmax)

    if use_nonan:
        sol = np.zeros((len(ga_nonan),3))
        sol[:,0] = v1a_nonan
        sol[:,1] = ga_nonan
        sol[:,2] = v2a_nonan
    else:
        sol = np.zeros((len(ga),3))
        sol[:,0] = v1a
        sol[:,1] = ga
        sol[:,2] = v2a

    
    # shift width and colormap
    #lwidths = (1.+(ga_nonan-gshift)/(.001+np.amax(ga_nonan-gshift))*lwfactor)

    if return3d:

        sol = np.transpose(sol)

        points = np.array([sol[0,:],sol[1,:],sol[2,:]]).T.reshape(-1,1,3)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        line3d = Line3DCollection(segs,linewidths=lwidths,
                                  cmap=my_cmap,zorder=zorder)
    
        line3d.set_array(ga_nonan)

        return line3d

    else:
        if use_nonan:
            points = np.array([sol[:,0], sol[:,2]]).T.reshape(-1, 1, 2)
        else:
            points = np.array([sol[:,0], sol[:,2]]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segs, linewidths=lwidths,
                            cmap=my_cmap,
                            zorder=zorder)
        lc.set_array(ga)
        return lc


def clean(x,y,smallscale=False,tol=.5):
    if smallscale:
        tol = .5
    else:
        tol = tol

    pos = np.where(np.abs(np.diff(y)) >= tol)[0]
    pos2 = np.where(np.abs(np.diff(x)) >= tol)[0]
    
    x[pos] = np.nan
    y[pos] = np.nan

    x[pos2] = np.nan
    y[pos2] = np.nan


    return x,y


def clean3d(x,y,z,smallscale=False,tol=.5):
    if smallscale:
        tol = .5
    else:
        tol = tol

    pos = np.where(np.abs(np.diff(y)) >= tol)[0]
    pos2 = np.where(np.abs(np.diff(x)) >= tol)[0]
    pos3 = np.where(np.abs(np.diff(z)) >= tol)[0]
    
    x[pos] = np.nan
    y[pos] = np.nan
    z[pos] = np.nan

    x[pos2] = np.nan
    y[pos2] = np.nan
    z[pos2] = np.nan


    x[pos3] = np.nan
    y[pos3] = np.nan
    z[pos3] = np.nan


    return x,y,z


def remove_redundant(x,y,tol=.01):

    pos = np.where(np.abs(np.diff(y)) < tol)[0]
    pos2 = np.where(np.abs(np.diff(x)) < tol)[0]
    
    x[pos] = np.nan
    y[pos] = np.nan

    x[pos2] = np.nan
    y[pos2] = np.nan

    return x,y


def remove_redundant_x(x,y,tol=.01):

    pos = np.where(np.abs(np.diff(x)) < tol)[0]
    
    x[pos] = np.nan
    y[pos] = np.nan

    return x,y

def remove_redundant_y(x,y,tol=.01):

    pos2 = np.where(np.abs(np.diff(x)) < tol)[0]

    x[pos2] = np.nan
    y[pos2] = np.nan

    return x,y


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    #http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def unlink_wrap(dat, lims=[-np.pi, np.pi], thresh = 0.95):
    # http://stackoverflow.com/questions/27138751/preventing-plot-joining-when-values-wrap-in-matplotlib-plots
    """
    Iterate over contiguous regions of `dat` (i.e. where it does not
    jump from near one limit to the other).

    This function returns an iterator object that yields slice
    objects, which index the contiguous portions of `dat`.

    This function implicitly assumes that all points in `dat` fall
    within `lims`.

    """
    jump = np.nonzero(np.abs(np.diff(dat)) > ((lims[1] - lims[0]) * thresh))[0]
    lasti = 0
    for ind in jump:
        yield slice(lasti, ind + 1)
        lasti = ind + 1
    yield slice(lasti, len(dat))


def beautify_phase(ax,xval,yval,per,dt,
                   back_idx=5,factor=1.,dashes='',
                   arrowsize=15,
                   arrows=2,
                   show_arrows=True,
                   gradient=False,
                   decorate=False,
                   lwstart=1.,
                   lwend=3.,
                   use_nonan=False,
                   fontsize=20,
                   xlo=-pi,xhi=pi,
                   ylo=-pi,yhi=pi,
                   ticks=True):
    """
    utility function:
    -add arrows on phase plane figure
    -thicken lines
    -change lines to black instead of default blue
    
    ax: axis plot object
    xval,yval: x-,y- values of plot
    per: period of solution, or total time
    dt: time step of sim
    back_idx: # of indices to stretch arrow back
    factor: modify radial distance (sometimes needed for small amplitude plots).
    """

    xval,yval = clean(xval,yval)

    if gradient:
        cmapmin = 1.
        cmapmax = 0.
    else:
        cmapmin=0.
        cmapmax=0.
        
    lyn = np.linspace(0,1,len(xval))
    ax.add_collection(collect3d_colorgrad(xval,lyn,yval,
                                          use_nonan=use_nonan,
                                          zorder=2,
                                          lwstart=lwstart,
                                          lwend=lwend,
                                          cmapmin=cmapmin,
                                          cmapmax=cmapmax,
                                          return3d=False,
                                          cmap=cmap))




    if show_arrows:
        idxlist = []
        for i in range(arrows):
            i = arrows - i
            idxlist.append(int((1.*i/arrows)*(per/dt))-2)
        print idxlist, per,len(xval)
            
        #idxlist = [int(0.*(per/dt)),int(1.*(per/dt)/2.)]# depends on period

        for j in idxlist:
            ax.annotate("",
                        xy=(factor*xval[j], factor*yval[j]), xycoords='data',
                        xytext=(factor*xval[j-back_idx], factor*yval[j-back_idx]), textcoords='data',
                        size=arrowsize,
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3",
                                        color='black')
                    )
    
    plt.setp(ax.lines,lw=3,color='k',dashes=dashes)

    if decorate:
        ax.set_xlim(xlo,xhi)
        ax.set_ylim(xlo,xhi)
        ax.set_xlabel(r'$\theta_1$',fontsize=fontsize)
        ax.set_ylabel(r'$\theta_2$',fontsize=fontsize)

        if ticks:
            ax.set_xticks(np.arange(-1,1+1,1)*pi)
            ax.set_yticks(np.arange(-1,1+1,1)*pi)


        ax.set_xticklabels(x_label_short,fontsize=fontsize)
        ax.set_yticklabels(x_label_short,fontsize=fontsize)
    else:
        #mp.figure()
        #mp.plot(xval,yval)
        #mp.plot(yval)
        #mp.show()
        #ax.set_xlim(np.amin(xval),np.amax(xval))
        #ax.set_ylim(np.amin(yval),np.amax(yval))
        ax.set_ylim(xlo,xhi)
        ax.set_xlabel('')
        ax.set_ylabel('')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        

    #ax.lines.set_linewidth(3)

    return ax


def ss_bump_fig():
    """
    plot steady-state bumps with arrows to bump peaks
    """
    
    fig = plt.figure(figsize=(10,5))
    
    ax1 = fig.add_subplot(121)
    dat = oned_simple.SimDat()

    ax1.set_title(AA,x=0,y=1.02)
    ax1.set_xlabel(r'$\bm{x}$',size=15)
    ax1.set_ylabel(r'\textbf{Activity}',size=15)
    ax1.plot(dat.domain-pi,np.roll(dat.u0b(dat.domain),dat.N/2),color='black',lw=3)
    
    # label peak 1d
    ax1.scatter(0,np.amax(dat.u0b(dat.domain)),edgecolor='black',facecolor='red',s=80,zorder=3)
    ax1.annotate(r'$\theta$', xy=(0, np.amax(dat.u0b(dat.domain))+.02), xycoords='data',
                 xytext=(0, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )
    # add text to peak
    
    ax1.set_xlim(-pi,pi)
    ax1.set_ylim(-1,1)

    ax2 = fig.add_subplot(122, projection='3d')
    dat = twod.SimDat()

    
    ax2 = twod.plot_s(ax2,dat.u0ss)

    # get/label peak 2d
    idx = np.argmax(np.reshape(dat.u0ss,dat.N_idx))
    peak_z = np.reshape(dat.u0ss,dat.N_idx)[idx]
    peak_x = np.reshape(dat.XX,dat.N_idx)[idx]
    peak_y = np.reshape(dat.YY,dat.N_idx)[idx]
    
    #ax2.scatter(peak_x+0.,peak_y+0.,peak_z,s=80,edgecolor='black',facecolor='white',zorder=1)
    ax2.plot([peak_x,peak_x],[peak_y,peak_y],[peak_z,peak_z+.01],marker='o',markersize=8,markeredgecolor='black',markerfacecolor='red',color='red',zorder=1)
    # http://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
    x2, y2, _ = proj3d.proj_transform(peak_x,peak_y,peak_z+.14,ax2.get_proj())
    ax2.annotate(r'$(\theta_1,\theta_2)$', xy=(x2,y2), xycoords='data',
                 xytext=(0, 30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )

    
    ax2.set_title(BB,x=0,y=1.1)
    ax2.set_xlabel(r'$\bm{x}$',size=15)
    ax2.set_ylabel(r'$\bm{y}$',size=15)
    # add text to peak

    plt.tight_layout()

    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)

    #ax2.xaxis.set_major_formatter(formatter)
    #ax2.yaxis.set_major_formatter(formatter)
    ax2.zaxis.set_major_formatter(formatter) 


    return fig


def oned_const_vel_bump(g=3.5,total=10000):
    """
    make figure for traveling bump
    """
    dat = oned_simple.SimDat(g=g,q=0,zshift=.1,T=total)    
    # get four bumps at four equal time intervals. use second half of sim,
    # use velocity to determine time intervals
    # Peaks of phase plot over time are at pi.
    # using second half of solution, subtract -(pi-.8*pi), find index of min

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$x$')
    
    ax.set_ylabel('t')
    
    #start_idx = len(dat.t)/2.
    #end_idx = int(1.5*start_idx)

    total_time_idx = dat.t[-1]/dat.dt

    pad = 10

    start_idx = np.argmin(np.mod(dat.ph_angle[total_time_idx/2:]+pi,2*pi)-pi)+total_time_idx/2+pad/2
    print start_idx
    edge_travel_time = (dat.b - dat.a)/dat.c_num # time it takes to go from -pi to pi
    edge_travel_idx = edge_travel_time/dat.dt-pad # total indices of travel time
    wraps = 5
    end_idx = start_idx+pad + wraps*(edge_travel_idx+pad)

    #idx = np.arange(start_idx,end_idx+1,1,dtype='int')
    
    cax = ax.matshow(np.roll(dat.sol[start_idx:end_idx,:dat.N],dat.N/2),cmap='gray',extent=[-pi,pi,dat.t[end_idx],dat.t[start_idx]])
    fig.colorbar(cax)
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position('bottom')
    
    
    for i in range(wraps):
        start_temp = start_idx+i*edge_travel_idx + pad*i
        end_temp = start_idx+(i+1)*edge_travel_idx
        idx_temp = np.arange(start_temp,end_temp+1,1,dtype='int')
        
        ax.plot(dat.ph_angle[idx_temp],np.linspace(dat.t[start_temp],dat.t[end_temp],len(idx_temp)),lw=3,color='black')
        ax.plot(-(np.mod(dat.solph[idx_temp+578,0]+pi,2*pi)-pi),np.linspace(dat.t[start_temp],dat.t[end_temp],len(idx_temp)),ls='--',lw=2,color='.65')
        

    print 'shifted oned const vel analytic by', 578, 'with dt=',dat.dt
    ax.set_aspect('auto')

    ax.set_xlim(-pi,pi)
    ax.set_ylim(dat.t[end_idx],dat.t[start_idx])

    ax.set_xticks(np.arange(-1,1+.5,.5)*pi)
    #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
    ax.set_xticklabels(x_label)
    
    plt.tight_layout()

    return fig


def oned_nonconst_vel_bump(g=3.,q=1.,shift=-700,sign=1,total=10000):
    """
    make figure for traveling bump
    """
    dat = oned_simple.SimDat(g=g,q=q,zshift=.1,T=total)    
    # period is approx 525 time units

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$x$')
    
    ax.set_ylabel('t')

    start_idx = len(dat.t)/2.
    end_idx = int(1.5*start_idx)

    idx = np.arange(start_idx,end_idx+1,1,dtype='int')
    
    cax = ax.matshow(np.roll(dat.sol[idx,:dat.N],dat.N/2),cmap='gray',extent=[-pi,pi,dat.t[end_idx],dat.t[start_idx]])
    fig.colorbar(cax)
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position('bottom')
    
    
    timearr = np.linspace(dat.t[start_idx],dat.t[end_idx],len(idx))
    for slc in unlink_wrap(dat.ph_angle[idx]):
        ax.plot(dat.ph_angle[idx][slc],timearr[slc],color='black',lw=3)
        
    modsolph = -(np.mod(dat.solph[idx+shift,0]+pi,2*pi)-pi)*sign
    for slc in unlink_wrap(modsolph):
        ax.plot(modsolph[slc],timearr[slc],ls='--',color='.65',lw=2)
    #ax.plot(dat.ph_angle[idx],np.linspace(dat.t[start_idx],dat.t[end_idx],len(idx)),color='black',lw=3)
    #ax.plot(-(np.mod(dat.solph[idx+shift,0]+pi,2*pi)-pi)*sign,np.linspace(dat.t[start_idx],dat.t[end_idx],len(idx)),ls='--',color='.65',lw=2)
    print 'shifted oned_nonconst_vel_bump ana by ', shift, 'where dt=',dat.dt
    
    ax.set_aspect('auto')

    ax.set_xlim(-pi,pi)
    ax.set_ylim(dat.t[end_idx],dat.t[start_idx])

    ax.set_xticks(np.arange(-1,1+.5,.5)*pi)

    ax.set_xticklabels(x_label)

    """

    ax.set_title('(b)',x=-.13)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\theta$')
    idx_beginning = int(dat.t[-1]/(1.2*dat.dt))
    ax.plot(dat.t[idx_beginning:],dat.ph_angle[idx_beginning:],color='black',lw=2)
    ax.plot(dat.t[idx_beginning:],-(np.mod(dat.solph[idx_beginning:,0]+pi,2*pi)-pi),color='gray',ls='--',lw=2)
    """

    plt.tight_layout()
    
    return fig


def oned_bump_combined():
    """
    display all 1d traveling bump figures in one.
    """

    fig = plt.figure(figsize=(10,4))

    """
    oned_const_vel
    #(oned_const_vel_bump,[],['oned_const_vel_bump_fig.pdf']),
    """
    #########################################################################################
    ax1 = fig.add_subplot(131)
    
    dat = oned_simple.SimDat(g=3.5,q=0.,zshift=.1,T=10000,phase=True)

    ax1.set_xlabel(r'$\bm{x}$',fontsize=15)
    ax1.set_ylabel(r'$\bm{t}$',fontsize=15)
    ax1.set_title(AA,x=0)
    
    #start_idx = len(dat.t)/2.
    #end_idx = int(1.5*start_idx)

    total_time_idx = dat.t[-1]/dat.dt

    pad = 10

    start_idx = np.argmin(np.mod(dat.ph_angle[total_time_idx/2:]+pi,2*pi)-pi)+total_time_idx/2+pad/2
    print start_idx
    edge_travel_time = (dat.b - dat.a)/dat.c_num # time it takes to go from -pi to pi
    edge_travel_idx = edge_travel_time/dat.dt-pad # total indices of travel time
    wraps = 5
    end_idx = start_idx+pad + wraps*(edge_travel_idx+pad)

    #idx = np.arange(start_idx,end_idx+1,1,dtype='int')
    
    cax = ax1.matshow(np.roll(dat.sol[start_idx:end_idx,:dat.N],dat.N/2),cmap='gray',extent=[-pi,pi,dat.t[end_idx],dat.t[start_idx]])
    #fig.colorbar(cax)
    ax1.xaxis.tick_bottom()
    ax1.xaxis.set_label_position('bottom')
    
    
    for i in range(wraps):
        start_temp = start_idx+i*edge_travel_idx + pad*i
        end_temp = start_idx+(i+1)*edge_travel_idx
        idx_temp = np.arange(start_temp,end_temp+1,1,dtype='int')
        
        ax1.plot(dat.ph_angle[idx_temp],np.linspace(dat.t[start_temp],dat.t[end_temp],len(idx_temp)),lw=3,color='black')
        ax1.plot(-(np.mod(dat.solph[idx_temp+578,0]+pi,2*pi)-pi),np.linspace(dat.t[start_temp],dat.t[end_temp],len(idx_temp)),ls='--',lw=2,color='#3399ff')
        

    print 'shifted oned const vel analytic by', 578, 'with dt=',dat.dt
    ax1.set_aspect('auto')

    ax1.set_xlim(-pi,pi)
    ax1.set_ylim(dat.t[end_idx],dat.t[start_idx])

    ax1.set_xticks(np.arange(-1,1+1,1)*pi)
    ax1.set_xticklabels(x_label_short)
    ax1.tick_params()


    #########################################################################################


    """
    ### oned_nonconst_vel1
    #(oned_nonconst_vel_bump,[],['oned_nonconst_vel_bump_fig.pdf']),
    """
    ax2 = fig.add_subplot(132)
    dat = oned_simple.SimDat(g=3.,q=1.,zshift=.1,T=10000,phase=True)
    # period is approx 525 time units
    shift = -1800
    sign = 1

    ax2.set_xlabel(r'$\bm{x}$',fontsize=15)
    ax2.set_title(BB,x=0)
    #ax2.set_ylabel(r'$t$')

    start_idx = len(dat.t)/2.
    end_idx = int(1.5*start_idx)

    idx = np.arange(start_idx,end_idx+1,1,dtype='int')
    
    cax = ax2.matshow(np.roll(dat.sol[idx,:dat.N],dat.N/2),cmap='gray',extent=[-pi,pi,dat.t[end_idx],dat.t[start_idx]])
    #fig.colorbar(cax)
    ax2.xaxis.tick_bottom()
    ax2.xaxis.set_label_position('bottom')
    
    
    timearr = np.linspace(dat.t[start_idx],dat.t[end_idx],len(idx))
    for slc in unlink_wrap(dat.ph_angle[idx]):
        ax2.plot(dat.ph_angle[idx][slc],timearr[slc],color='black',lw=3)
        
    modsolph = -(np.mod(dat.solph[idx+shift,0]+pi,2*pi)-pi)*sign
    for slc in unlink_wrap(modsolph):
        ax2.plot(modsolph[slc],timearr[slc],ls='--',lw=2,color='#3399ff')
    #ax.plot(dat.ph_angle[idx],np.linspace(dat.t[start_idx],dat.t[end_idx],len(idx)),color='black',lw=3)
    #ax.plot(-(np.mod(dat.solph[idx+shift,0]+pi,2*pi)-pi)*sign,np.linspace(dat.t[start_idx],dat.t[end_idx],len(idx)),ls='--',color='.65',lw=2)
    print 'shifted oned_nonconst_vel_bump ana by ', shift, 'where dt=',dat.dt
    
    ax2.set_aspect('auto')

    ax2.set_xlim(-pi,pi)
    ax2.set_ylim(dat.t[end_idx],dat.t[start_idx])

    ax2.set_xticks(np.arange(-1,1+1,1)*pi)
    ax2.set_xticklabels(x_label_short)
    ax2.tick_params()

    #########################################################################################
    """
    ### oned_nonconst_vel2
    #(oned_nonconst_vel_bump,[5.5,1.,-950,-1],['oned_nonconst_vel_bump_fig2.pdf']),
    """
    sign = -1
    shift = -950
    
    ax3 = fig.add_subplot(133)
    dat = oned_simple.SimDat(g=5.5,q=1.,zshift=.1,T=10000,phase=True)
    # period is approx 525 time units
    ax3.set_xlabel(r'$\bm{x}$',fontsize=15)
    ax3.set_title(CC,x=0)
    

    start_idx = len(dat.t)/2.
    end_idx = int(1.5*start_idx)

    idx = np.arange(start_idx,end_idx+1,1,dtype='int')
    
    cax = ax3.matshow(np.roll(dat.sol[idx,:dat.N],dat.N/2),cmap='gray',extent=[-pi,pi,dat.t[end_idx],dat.t[start_idx]])
    fig.colorbar(cax)
    ax3.xaxis.tick_bottom()
    ax3.xaxis.set_label_position('bottom')
    
    
    timearr = np.linspace(dat.t[start_idx],dat.t[end_idx],len(idx))
    for slc in unlink_wrap(dat.ph_angle[idx]):
        ax3.plot(dat.ph_angle[idx][slc],timearr[slc],color='black',lw=3)
        
    modsolph = -(np.mod(dat.solph[idx+shift,0]+pi,2*pi)-pi)*sign
    for slc in unlink_wrap(modsolph):
        ax3.plot(modsolph[slc],timearr[slc],ls='--',lw=2,color='#3399ff')
    #ax.plot(dat.ph_angle[idx],np.linspace(dat.t[start_idx],dat.t[end_idx],len(idx)),color='black',lw=3)
    #ax.plot(-(np.mod(dat.solph[idx+shift,0]+pi,2*pi)-pi)*sign,np.linspace(dat.t[start_idx],dat.t[end_idx],len(idx)),ls='--',color='.65',lw=2)
    print 'shifted oned_nonconst_vel_bump ana by ', shift, 'where dt=',dat.dt
    
    ax3.set_aspect('auto')

    ax3.set_xlim(-pi,pi)
    ax3.set_ylim(dat.t[end_idx],dat.t[start_idx])

    ax3.set_xticks(np.arange(-1,1+1,1)*pi)
    ax3.set_xticklabels(x_label_short)
    ax3.tick_params()

    #ax3.set_yticklabels([])
    
    plt.tight_layout()


    ax1.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)
    ax3.yaxis.set_major_formatter(formatter)


    return fig

def oned_pitchfork(g0=.1,g1=2.5,N=50):
    """
    get figure for 1d pitchfork bifurcation
    """
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$g$')
    ax.set_ylabel('Bump Velocity')

    glist = np.linspace(g0,g1,N)
    an_arr_plus = np.zeros(N) # analytic speed
    num_arr_plus = np.zeros(N) # numerical speed
    
    an_arr_minus = np.zeros(N) # analytic speed
    num_arr_minus = np.zeros(N) # numerical speed

    num_arr_zero = np.zeros(N)
    for i,g in enumerate(glist):
        dat = oned_simple.SimDat(g=g,q=0,zshift=.1)
        dat2 = oned_simple.SimDat(g=g,q=0,zshift=-.1)
        dat3 = oned_simple.SimDat(g=g,q=0,zshift=0.)
        
        dat.params()
        an_arr_plus[i] = dat.c_theory_eqn
        an_arr_minus[i] = -dat2.c_theory_eqn
        
        num_arr_plus[i] = dat.c_num
        num_arr_minus[i] = dat2.c_num

        
    ax.plot(glist,num_arr_plus,lw=3,color='black')
    ax.plot(glist,num_arr_minus,lw=3,color='black')
    
    ax.plot(glist,an_arr_plus,lw=2,linestyle='--',color='gray')
    #ax.scatter(glist,num_arr_plus,marker='x',s=80,color='black')

    ax.plot(glist,an_arr_minus,lw=2,linestyle='--',color='gray')
    #ax.scatter(glist,num_arr_minus,marker='x',s=80,color='black')

    ax.plot(glist,np.zeros(N),lw=2,linestyle='--',color='gray')

    ax.set_xlim(g0,g1)




    return fig

def oned_hopf(g0=1,g1=3.5,N=50):
    """
    Get limsup of simulation
    https://stackoverflow.com/questions/35149843/running-max-limsup-in-numpy-what-optimization/35150222#35150222?newreg=d630fa97367849d39f64defea1386dd2
    limsup code doesn't work as expected
    """
    
    # get index of peaks., get values of peaks. take last value.
    
    
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$g$')
    ax.set_ylabel('Oscillation Amplitude')

    glist = np.linspace(g0,g1,N)
    amp_num_plus = np.zeros(N) # numerical amplitude
    amp_ana_plus = np.zeros(N) # analytic amplitude

    amp_num_minus = np.zeros(N) # numerical amplitude
    amp_ana_minus = np.zeros(N) # analytic amplitude
    
    for i,g in enumerate(glist):
        dat = oned_simple.SimDat(g=g,q=1,zshift=.1,T=20000)
        dat.params()
                
        dat.plot('phase_angle')
        #plt.show()
        
        # get peak indices
        #get amplitude of last 20% of data
        N_num = len(dat.ph_angle)
        N_ana = len(dat.solph[:,0])
        
        amp_num_plus[i] = np.amax(dat.ph_angle[int(.8*N_num):])
        amp_ana_plus[i] = np.amax(dat.solph[:,0][int(.8*N_ana):])

        amp_num_minus[i] = np.amin(dat.ph_angle[int(.8*N_num):])
        amp_ana_minus[i] = np.amin(dat.solph[:,0][int(.8*N_ana):])

        """
        dsol_num = np.gradient(dat.ph_angle)
        dsol_ana = np.gradient(dat.solph[:,0])

        peak_idx_num = np.where(np.diff(np.sign(dsol_num)))[0][-1]
        peak_idx_ana = np.where(np.diff(np.sign(dsol_ana)))[0][-1]

        print peak_idx_num
        print peak_idx_ana
        
        # peak values. get last peak.
        amp_num[i] = dat.ph_angle[peak_idx_num]
        amp_ana[i] = dat.solph[:,0][peak_idx_ana]
        """
        
    ax.plot(glist,amp_num_plus,lw=3,color='black')
    ax.plot(glist,amp_num_minus,lw=3,color='black')
    #ax.scatter(glist,num_arr_plus,marker='x',s=80,color='black')

    ax.plot(glist,amp_ana_plus,lw=2,linestyle='--',color='gray')
    ax.plot(glist,amp_ana_minus,lw=2,linestyle='--',color='gray')

    ax.set_xlim(g0,g1)

    return fig

def oned_bifurcations():
    """
    combined hopf and pitchfork figure functions from above
    """
    fig = plt.figure(figsize=(10,4))

    subtitle_shift = -.0
    subtitle_shift_y = 1.05

    g0=.1;g1=2.5;N=50
    ax1 = fig.add_subplot(121)
    ax1.set_title(AA,x=subtitle_shift,y=subtitle_shift_y)
    ax1.set_xlabel(r'$g$')
    ax1.set_ylabel('Bump Velocity')

    glist = np.linspace(g0,g1,N)
    an_arr_plus = np.zeros(N) # analytic speed
    num_arr_plus = np.zeros(N) # numerical speed
    
    an_arr_minus = np.zeros(N) # analytic speed
    num_arr_minus = np.zeros(N) # numerical speed

    num_arr_zero = np.zeros(N)
    for i,g in enumerate(glist):
        dat = oned_simple.SimDat(g=g,q=0,zshift=.1)
        dat2 = oned_simple.SimDat(g=g,q=0,zshift=-.1)
        dat3 = oned_simple.SimDat(g=g,q=0,zshift=0.)
        
        dat.params()
        an_arr_plus[i] = dat.c_theory_eqn
        an_arr_minus[i] = -dat2.c_theory_eqn
        
        num_arr_plus[i] = dat.c_num
        num_arr_minus[i] = dat2.c_num

        del dat,dat2,dat3
        
    ax1.plot(glist,num_arr_plus,lw=3,color='black')
    ax1.plot(glist,num_arr_minus,lw=3,color='black')
    
    ax1.plot(glist,an_arr_plus,lw=2,linestyle='--',color='gray')
    ax1.plot(glist,an_arr_minus,lw=2,linestyle='--',color='gray')

    ax1.plot(glist,np.zeros(N),lw=2,linestyle='--',color='gray')

    ax1.set_xlim(g0,g1)


    g0=1;g1=3.5;N=50
    ax = fig.add_subplot(122)
    ax.set_title(BB,x=subtitle_shift,y=subtitle_shift_y)
    ax.set_xlabel(r'$g$')
    ax.set_ylabel('Oscillation Amplitude')

    glist = np.linspace(g0,g1,N)
    amp_num_plus = np.zeros(N) # numerical amplitude
    amp_ana_plus = np.zeros(N) # analytic amplitude

    amp_num_minus = np.zeros(N) # numerical amplitude
    amp_ana_minus = np.zeros(N) # analytic amplitude
    
    for i,g in enumerate(glist):
        dat = oned_simple.SimDat(g=g,q=1,zshift=.1,T=20000)
        dat.params()
                
        dat.plot('phase_angle')
        #plt.show()
        
        # get peak indices
        #get amplitude of last 20% of data
        N_num = len(dat.ph_angle)
        N_ana = len(dat.solph[:,0])
        
        amp_num_plus[i] = np.amax(dat.ph_angle[int(.8*N_num):])
        amp_ana_plus[i] = np.amax(dat.solph[:,0][int(.8*N_ana):])

        amp_num_minus[i] = np.amin(dat.ph_angle[int(.8*N_num):])
        amp_ana_minus[i] = np.amin(dat.solph[:,0][int(.8*N_ana):])

        del dat
        
    ax.plot(glist,amp_num_plus,lw=3,color='black')
    ax.plot(glist,amp_num_minus,lw=3,color='black')
    #ax.scatter(glist,num_arr_plus,marker='x',s=80,color='black')

    ax.plot(glist,amp_ana_plus,lw=2,linestyle='--',color='gray')
    ax.plot(glist,amp_ana_minus,lw=2,linestyle='--',color='gray')

    ax.set_xlim(g0,g1)

    
    return fig
    

def twod_full_fig(q=1,g=3.,zshift_angle=pi/4.,zshift_rad=.3,T=5000,factor=.5,increment=13):
    """
    peak dynamics of full model
    """

    
    print 'initial angle',zshift_angle,'inital rad',zshift_rad
    #ushift1=1.;ushift2=1.
    zshift1=ushift1-zshift_rad*np.cos(zshift_angle);zshift2=ushift2-zshift_rad*np.sin(zshift_angle)
    ishift1=0.;ishift2=0.

    ushift1=0.
    ushift2=0.

    #zshift1 = ushift1+.5#-.1
    #zshift2 = ushift2+1.#-.1

    eps = .005
    dat = twod.SimDat(q=q,g=g,T=T,zshift1=zshift1,zshift2=zshift2,ushift1=ushift1,ushift2=ushift2,eps=eps)
    
    # remove first half of sim to ignore transients    
    start_idx = int(dat.TN*factor)
    total_idx = dat.TN - start_idx
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)

    arrow_idx_increment = total_idx/increment
    back_idx = 2
    
    for i in range(start_idx,dat.TN-1):
        
        color = ((1.*total_idx - (i-start_idx))/total_idx)*.75

        if i%arrow_idx_increment == 0:
            ax.annotate("",
                        xy=(dat.th1[i], dat.th2[i]), xycoords='data',
                        xytext=(dat.th1[i-back_idx], dat.th2[i-back_idx]), textcoords='data',
                        size=22,
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3",
                                        color=str(color)),
                        
                    )

        #print color
        #ax.scatter(dat.th1[i],dat.th2[i],edgecolors='none',facecolors=str(color),s=(1-color)*30)



    #colors = np.arange(75,0,len(dat.th1[1:-1]))
    colors = np.linspace(.85,0.,len(dat.th1[start_idx:-1]))
    cmap = plt.get_cmap('gray')
    my_cmap = truncate_colormap(cmap,.0,.75)
    #my_cmap.set_under('w')
    size = (1-colors)*30
    
    #ax.set_title('g='+str(g)+'; q='+str(q)+'; eps='+str(eps))
    
    ax.scatter(dat.th1[start_idx:-1],dat.th2[start_idx:-1],edgecolors='none',c=colors,s=size,cmap=my_cmap)
    ax.scatter(dat.th1[-1],dat.th2[-1],marker="*",color='black',s=200,facecolors='white')
    ax.scatter(dat.th1[start_idx],dat.th2[start_idx],marker="o",color='black',s=50,facecolors='white')

    ax.set_xlim(-pi,pi)
    ax.set_ylim(-pi,pi)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')

    ax.set_xticks(np.arange(-1,1+.5,.5)*pi)
    ax.set_yticks(np.arange(-1,1+.5,.5)*pi)

    ax.set_xticklabels(x_label)
    ax.set_yticklabels(x_label)

    del dat
    return fig


def twod_phase_fig(q=1,g=4.2,T=104,factor=.71,increment=13,phase_option='approx'):
    """
    peak dynamics of phase model. full or approx.
    """
    ph = twodp.Phase(q=q,x0=1,y0=.01,g=g,dde_T=T,phase_option=phase_option)

    
    # remove first half of sim to ignore transients    
    start_idx = int(ph.dde_TN*factor)
    total_idx = ph.dde_TN - start_idx
    
    
    arrow_idx_increment = total_idx/increment
    back_idx = 1


    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)

    th1 = np.mod(ph.th1+pi,2*pi)-pi
    th2 = np.mod(ph.th2+pi,2*pi)-pi
    
    for i in range(start_idx,ph.dde_TN-1):
        
        color = ((1.*total_idx - (i-start_idx))/total_idx)*.75
        if i%arrow_idx_increment == 0:

            ax.annotate("",
                        xy=(th1[i], th2[i]), xycoords='data',
                        xytext=(th1[i-back_idx], th2[i-back_idx]), textcoords='data',
                        size=22,
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3",
                                        color=str(color)),
                    )

        #print color
        #ax.scatter(ph.th1[i],ph.th2[i],edgecolors='none',facecolors=str(color),s=(1-color)*30)
 
    
    # for speedup consider using http://stackoverflow.com/questions/17682216/scatter-plot-and-color-mapping-in-python

    colors = np.linspace(.75,0.,len(ph.th1[start_idx:-1]))
    cmap = plt.get_cmap('gray')
    my_cmap = truncate_colormap(cmap,.0,.75)

    size = (1-colors)*30
    ax.scatter(th1[start_idx:-1],th2[start_idx:-1],edgecolors='none',c=colors,s=size,cmap=my_cmap)
    ax.scatter(th1[-1],th2[-1],marker="*",color='black',s=200,facecolors='white')
    ax.scatter(th1[start_idx],th2[start_idx],marker="o",color='black',s=50,facecolors='white')


    ax.set_xlim(-pi,pi)
    ax.set_ylim(-pi,pi)
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')

    ax.set_xticks(np.arange(-1,1+.5,.5)*pi)
    ax.set_yticks(np.arange(-1,1+.5,.5)*pi)

    ax.set_xticklabels(x_label)
    ax.set_yticklabels(x_label)
    
    del ph
    return fig


    

def decorate_phase_space(ax,datax,datay,transients_factor,time_steps,arrow_increment,
                         back_idx=2,skip=1,cmap='gray'):
    """
    decorate the phase space plot of twod domain
    to be used with combined_phase_fig().
    ax: axis object
    datax,datay: data to be plotted.
    transients_factor: use .8 to skip 80% of the data and plot only the last 20%
    time_steps: total time steps (of full simulation)
    arrow_increment: increment of arrows in plot
    back_idx: look this many indices back to set arrow start point.
    skip: skip this many data points in scatter plots (helps reduce pdf size)
    """


    cmap = plt.get_cmap(cmap)
    my_cmap = truncate_colormap(cmap,.0,.75)

    # remove first half of sim to ignore transients    
    start_idx = int(time_steps*transients_factor)
    total_idx = time_steps - start_idx

    arrow_idx_increment = total_idx/arrow_increment
    
    for i in range(start_idx,time_steps-1):
        
        color = ((1.*total_idx - (i-start_idx))/total_idx)*.75

        if i%arrow_idx_increment == 0:
            ax.annotate("",
                        xy=(datax[i], datay[i]), xycoords='data',
                        xytext=(datax[i-back_idx], datay[i-back_idx]), textcoords='data',
                        size=22,
                        arrowprops=dict(arrowstyle="-|>",
                                        connectionstyle="arc3",
                                        color=str(color)),
                        
                    )



    colors = np.linspace(.85,0.,len(datax[start_idx:-1][::skip]))
    
    #my_cmap.set_under('w')
    size = (1-colors)*30
    
    #ax.set_title('g='+str(g)+'; q='+str(q)+'; eps='+str(eps))

    
    ax.scatter(datax[start_idx:-1][::skip],datay[start_idx:-1][::skip],edgecolors='none',c=colors,s=size,cmap=my_cmap)
    ax.scatter(datax[-1],datay[-1],marker="*",color='black',s=200,facecolors='white')
    ax.scatter(datax[start_idx],datay[start_idx],marker="o",color='black',s=50,facecolors='white')

    ax.set_xlim(-pi,pi)
    ax.set_ylim(-pi,pi)
    ax.set_xlabel(r'$\theta_1$',fontsize=20)
    ax.set_ylabel(r'$\theta_2$',fontsize=20)

    ax.set_xticks(np.arange(-1,1+.5,.5)*pi)
    ax.set_yticks(np.arange(-1,1+.5,.5)*pi)

    ax.set_xticklabels(x_label,fontsize=20)
    ax.set_yticklabels(x_label,fontsize=20)

    return ax

def combined_phase_fig(option ="limit_cycle"):
    """
    plot all three 2D full, 2D phase, 2D phase approx at once.
    """



    r0=1.;nu0=.01
    ushift1=0.
    ushift2=0.

    zshift_rad=.8;zshift_angle=pi/3.5
    zshift1=ushift1-zshift_rad*np.cos(zshift_angle);zshift2=ushift2-zshift_rad*np.sin(zshift_angle)
    ishift1=0.;ishift2=0.

    #zshift1 = ushift1+.4#-.1
    #zshift2 = ushift2+1.#-.1

    fig = plt.figure(figsize=(13,5))
    
    if option == "const":


        full_q=0;full_g=3;full_T=7000
        full_factor=.4;full_increment=10

        ph_full_q=0;ph_full_g=2.2;ph_full_dde_T=200
        ph_full_factor=.9;ph_full_increment=11

        ph_approx_q=0;ph_approx_g=2.5;ph_approx_dde_T=100
        ph_approx_factor=.8;ph_approx_increment=11

        arrows=5

    elif option == "limit_cycle":
        #(twod_full_fig, [2.,5.,5000,.84,13],['twod_full_fig_q=2_g=5.pdf']),
        #(twod_phase_fig,[1.,4.,300,.0,20,'full'],['twod_phase_full_fig_test']),
        #(twod_phase_fig,[1.,4.,110,.956,13,'approx'],['twod_phase_approx_fig_q=1_g=4.pdf']),

        full_q=2;full_g=5;full_T=5000
        full_factor=.839;full_increment=13

        ph_full_q=1;ph_full_g=3.;ph_full_dde_T=110
        ph_full_factor=.949;ph_full_increment=7

        ph_approx_q=.2;ph_approx_g=1.;ph_approx_dde_T=110
        ph_approx_factor=.9;ph_approx_increment=6

        arrows=2
    
    elif option == "non_const":
        #(twod_full_fig, [1.,5.,5000,.45,13],['twod_full_fig_q=1_g=5.pdf']),
        #(twod_phase_fig,[1.,5.,100,.85,5,'approx'],['twod_phase_approx_fig_q=1_g=5.pdf']),
        #(twod_phase_fig,[1.,5.,100,.8,12,'full'],['twod_phase_full_fig_q=1_g=5.pdf']),

        full_q=1.;full_g=5;full_T=5000
        full_factor=.4;full_increment=13 # plotting options

        ph_full_q=1.;ph_full_g=5.;ph_full_dde_T=200
        ph_full_factor=.92;ph_full_increment=5 # plotting options

        ph_approx_q=.5;ph_approx_g=4.5;ph_approx_dde_T=200
        ph_approx_factor=.93;ph_approx_increment=12 # plotting options

        arrows=6
    
    dat = twod.SimDat(q=full_q,g=full_g,T=full_T,zshift1=zshift1,zshift2=zshift2,ushift1=ushift1,ushift2=ushift2)
    ph_full = twodp.Phase(q=ph_full_q,x0=r0,y0=nu0,g=ph_full_g,dde_T=ph_full_dde_T,dde_dt=.03,phase_option='full')
    ph_approx = twodp.Phase(q=ph_approx_q,x0=3,y0=1.,g=ph_approx_g,dde_T=ph_approx_dde_T,dde_dt=.03,phase_option='trunc')

    dde_dt=.03
    
    subtitle_shift = -.0
    subtitle_shift_y = 1.05


    ## Plot full
    ax1 = fig.add_subplot(131)
    ax1.set_title(AA,x=subtitle_shift,y=subtitle_shift_y,fontsize=20)
    
    # ax,datax,datay,transients_factor,time_steps,arrow_increment,back_idx=2,skip=10
    #ax1 = decorate_phase_space(ax1,dat.th1,dat.th2,full_factor,dat.TN,full_increment,skip=40)

    skipN = int(full_factor*full_T/dat.dt)
    x = dat.th1[skipN:]
    y = dat.th2[skipN:]

    ax1 = beautify_phase(ax1,x,y,(1-full_factor)*full_T,dat.dt,
                         gradient=True,decorate=True,
                         lwstart=2.,lwend=6.,arrowsize=25,arrows=arrows)

    #ax2 = beautify_phase(ax2,xval,yval,per1,dt,gradient=True,arrowsize=10)

    ## Plot phase full
    ax2 = fig.add_subplot(132)
    ax2.set_title(BB,x=subtitle_shift,y=subtitle_shift_y,fontsize=20)

    th1 = np.mod(ph_full.th1_ph+pi,2*pi)-pi
    th2 = np.mod(ph_full.th2_ph+pi,2*pi)-pi


    skipN = int(ph_full_factor*ph_full_dde_T/dde_dt)
    x = th1[skipN:]
    y = th2[skipN:]


    #mp.figure()
    #mp.plot(ph_full.dde_t[ph_full.dde_delay_N+skipN:],x)
    #mp.plot(ph_full.dde_t[ph_full.dde_delay_N+skipN:],y)
    #mp.show()

    ax2 = beautify_phase(ax2,x,y,(1-ph_full_factor)*ph_full_dde_T,dde_dt,
                         gradient=True,decorate=True,
                         lwstart=2.,lwend=6.,arrowsize=25,arrows=arrows)
    #ax2 = decorate_phase_space(ax2,th1,th2,ph_full_factor,ph_full.dde_TN,ph_full_increment,back_idx=1)
    ax2.set_yticks([])
    ax2.set_ylabel('')
    
    ## Plot phase approx
    ax3 = fig.add_subplot(133)
    ax3.set_title(CC,x=subtitle_shift,y=subtitle_shift_y,fontsize=20)

    th1 = np.mod(ph_approx.th1_ph+pi,2*pi)-pi
    th2 = np.mod(ph_approx.th2_ph+pi,2*pi)-pi

    skipN = int(ph_approx_factor*ph_approx_dde_T/dde_dt)
    x = th1[skipN:]
    y = th2[skipN:]


    #ax3 = decorate_phase_space(ax3,th1,th2,ph_approx_factor,ph_approx.dde_TN,ph_approx_increment,back_idx=1)
    ax3 = beautify_phase(ax3,x,y,(1-ph_approx_factor)*ph_approx_dde_T,dde_dt,
                         gradient=True,decorate=True,
                         lwstart=2.,lwend=6.,arrowsize=25,arrows=arrows)

    ax3.set_yticks([])
    ax3.set_ylabel('')


    return fig

def twod_chaos_fig():
    """
    check if the neural field on the 2d domain is chaotic.

    spoiler: it's not. (checked up to 20000 time units with q=1,g=5 on full neural field model).
    """


    r0=1.;nu0=.01
    ushift1=0.
    ushift2=0.

    zshift_rad=.8;zshift_angle=pi/3.5
    zshift1=ushift1-zshift_rad*np.cos(zshift_angle);zshift2=ushift2-zshift_rad*np.sin(zshift_angle)
    ishift1=0.;ishift2=0.

    #zshift1 = ushift1+.4#-.1
    #zshift2 = ushift2+1.#-.1

    fig = plt.figure(figsize=(15,5))


    full_q=1.;full_g=5.;full_T=5000
    full_factor=.84;full_increment=13
    
    ph_full_q=1.;ph_full_g=5.;ph_full_dde_T=110
    ph_full_factor=.949;ph_full_increment=7
    
    ph_approx_q=1.;ph_approx_g=5.;ph_approx_dde_T=110
    ph_approx_factor=.953;ph_approx_increment=6

    dat = twod.SimDat(q=full_q,g=full_g,T=full_T,zshift1=zshift1,zshift2=zshift2,ushift1=ushift1,ushift2=ushift2)
    dat2 = twod.SimDat(q=full_q,g=full_g,T=full_T,zshift1=zshift1+.000001,zshift2=zshift2,ushift1=ushift1,ushift2=ushift2)


    #ph_full = twodp.Phase(q=ph_full_q,x0=r0,y0=nu0,g=ph_full_g,dde_T=ph_full_dde_T,dde_dt=.03,phase_option='full')
    #ph_approx = twodp.Phase(q=ph_approx_q,x0=3,y0=1.,g=ph_approx_g,dde_T=ph_approx_dde_T,dde_dt=.03,phase_option='approx')
    
    subtitle_shift = -.0
    subtitle_shift_y = 1.05


    ## Plot full
    ax1 = fig.add_subplot(131)
    ax1.set_title(AA,x=subtitle_shift,y=subtitle_shift_y,fontsize=20)
    
    # ax,datax,datay,transients_factor,time_steps,arrow_increment,back_idx=2,skip=10
    ax1 = decorate_phase_space(ax1,dat.th1,dat.th2,full_factor,dat.TN,full_increment,skip=40)
    ax1 = decorate_phase_space(ax1,dat2.th1,dat2.th2,full_factor,dat.TN,full_increment,skip=40,cmap=cmap)

    """
    ## Plot phase full
    ax2 = fig.add_subplot(132)
    ax2.set_title(r"\textbf{(b)}",x=subtitle_shift,y=subtitle_shift_y,fontsize=20)

    th1 = np.mod(ph_full.th1_ph+pi,2*pi)-pi
    th2 = np.mod(ph_full.th2_ph+pi,2*pi)-pi

    ax2 = decorate_phase_space(ax2,th1,th2,ph_full_factor,ph_full.dde_TN,ph_full_increment,back_idx=1)
    
    
    ## Plot phase approx
    ax3 = fig.add_subplot(133)
    ax3.set_title(r"\textbf{(c)}",x=subtitle_shift,y=subtitle_shift_y,fontsize=20)

    th1 = np.mod(ph_approx.th1_ph+pi,2*pi)-pi
    th2 = np.mod(ph_approx.th2_ph+pi,2*pi)-pi

    ax3 = decorate_phase_space(ax3,th1,th2,ph_approx_factor,ph_approx.dde_TN,ph_approx_increment,back_idx=1)
    """

    return fig



def HJ_i_fig():
    """
    plot H_i in first row
    J_i in second row
    """
    dat = twodp.Phase(recompute_h=False,recompute_j=False)
    H1,H2 = dat.H1,dat.H2
    J1,J2 = dat.J1,dat.J2

    fig = plt.figure(figsize=(10,10))

    subtitle_shift = -.0

    ax11 = fig.add_subplot(2,2,1,projection='3d')
    ax11.set_title(AA,x=subtitle_shift)
    #ax11.set_title(r"$H_1$")
    ax11 = twod.plot_s(ax11,H1)
    ax11.set_zlabel(r'$\bm{H_1}$',size=15)

    ax12 = fig.add_subplot(2,2,2,projection='3d')
    ax12.set_title(BB,x=subtitle_shift)
    #ax12.set_title(r"$H_2$")
    ax12 = twod.plot_s(ax12,H2)
    ax12.set_zlabel(r'$\bm{H_2}$',size=15)

    
    ax21 = fig.add_subplot(2,2,3,projection='3d')
    ax21.set_title(CC,x=subtitle_shift)
    #ax21.set_title(r"$J_1$")
    ax21 = twod.plot_s(ax21,J1)
    ax21.set_zlabel(r'$\bm{J_1}$',size=15)

    ax22 = fig.add_subplot(2,2,4,projection='3d')
    ax22.set_title(DD,x=subtitle_shift)
    #ax22.set_title(r"$J_2$")
    ax22 = twod.plot_s(ax22,J2)
    ax22.set_zlabel(r'\bm{$J_2}$',size=15)

    plt.tight_layout()


    ax11.zaxis.set_major_formatter(formatter)
    ax12.zaxis.set_major_formatter(formatter)
    ax21.zaxis.set_major_formatter(formatter)
    ax22.zaxis.set_major_formatter(formatter)
    
    return fig


def HJ_fig():
    """
    plot H in first row
    J in second row
    """
    dat = oned_simple.SteadyState()

    #dat.plot('J')
    #dat.plot('H')
    
    #plt.show()

    fig = plt.figure(figsize=(10,3))

    subtitle_shift = -0.05
    subtitle_shift_y = 1.1

    ax11 = fig.add_subplot(1,2,1)
    ax11.set_title(AA,x=subtitle_shift,y=subtitle_shift_y)

    newdom = np.linspace(-pi,pi,dat.N)

    ax11.plot(newdom,np.roll(dat.H_numerical,dat.N/2),color='black',lw=4,label=r'$H(\theta)$')
    ax11.plot(newdom,np.roll(dat.H(dat.domain),dat.N/2),color='#3399ff',ls='--',lw=3,label=r'$H(\theta)$ (approx.)')

    ax11.tick_params()
    #plot.tick_params(axis='both', which='major', labelsize=10)

    ax11.set_xticks(np.arange(-1,1+.5,.5)*pi)
    ax11.set_xticklabels(x_label)
    ax11.set_xlabel(r'$\theta$',size=15)
    
    ax11.set_xlim(-pi,pi)
    ax11.legend(loc=4)

    ax12 = fig.add_subplot(1,2,2)
    ax12.set_title(BB,x=subtitle_shift,y=subtitle_shift_y)

    ax12.plot(newdom,np.roll(dat.J_numerical,dat.N/2),color='black',lw=4,label=r'$J(\theta)$')
    ax12.plot(newdom,np.roll(dat.J(dat.domain),dat.N/2),color='#3399ff',ls='--',lw=3,label=r'$J(\theta)$ (approx.)')

    ax12.tick_params()
    ax12.set_xticks(np.arange(-1,1+.5,.5)*pi)
    ax12.set_xticklabels(x_label)
    ax12.set_xlabel(r'$\theta$',size=15)

    ax12.set_xlim(-pi,pi)
    ax12.legend(loc=3)

    plt.tight_layout()


    ax11.yaxis.set_major_formatter(formatter)
    ax12.yaxis.set_major_formatter(formatter) 


    
    return fig
    
    
def H_approx_fig():
    fig = plt.figure(figsize=(10,5))

    subtitle_shift = -.0

    dat = twodp.Phase()        
    h1_approx_p = dat.h1_approx_p(dat.XX,dat.YY)
    h2_approx_p = dat.h2_approx_p(dat.XX,dat.YY)

    dat2 = twodp.Phase(recompute_h=False,recompute_j=False)
    H1 = dat2.H1
    J1 = dat2.J1

    ax11 = fig.add_subplot(2,2,1,projection='3d')
    ax11 = twod.plot_s(ax11,h1_approx_p)
    ax11.set_title(AA,x=subtitle_shift)
    ax11.set_zlabel(r'$\hat H_1$')

    ax12 = fig.add_subplot(2,2,2,projection='3d')
    ax12 = twod.plot_s(ax12,-h1_approx_p)
    ax12.set_title(BB,x=subtitle_shift)
    ax12.set_zlabel(r'$\hat J_1$')

    ax21 = fig.add_subplot(2,2,3,projection='3d')
    ax21 = twod.plot_s(ax21,H1)
    ax21.set_title(CC,x=subtitle_shift)
    ax21.set_zlabel(r'$H_1$')

    ax22 = fig.add_subplot(2,2,4,projection='3d')
    ax22 = twod.plot_s(ax22,J1)
    ax22.set_title(DD,x=subtitle_shift)
    ax22.set_zlabel(r'$J_1$')


    return fig

def H_approx_nullclines():
    """
    plot level curves z=0. intersections denote existence of limit cycles.
    """

    ncx = np.loadtxt("nc_phase_approx_q=0.5_g=1.5_x_mesh=100.dat")
    ncy = np.loadtxt("nc_phase_approx_q=0.5_g=1.5_y_mesh=100.dat")

    #ncx = np.loadtxt("nc_phase_approx_q=1_g=3_x_mesh100.dat")
    #ncy = np.loadtxt("nc_phase_approx_q=1_g=3_y_mesh100.dat")

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(131)

    #ncy[:,0] = np.sort(ncy[:,0])
    #ncy[:,1] = ncy[:,1][np.argsort(ncy[:,0])]

    ncy[ncy[:,1]>.85]=np.nan

    #ncx,ncy = remove_redundant(ncx,ncy,tol=.01)

    index_to_order_x_by = ncx[:,1].argsort()
    index_to_order_y_by = ncy[:,0].argsort()

    ncx_ordered = ncx[index_to_order_x_by]
    ncy_ordered = ncy[index_to_order_y_by]

    #ncx = np.loadtxt("nc_phase_approx_q=1_g=3_x_mesh100.dat")
    #ncy = np.loadtxt("nc_phase_approx_q=1_g=3_y_mesh100.dat")

    #ncy[:,0] = np.sort(ncy[:,0])
    #ncy[:,1] = ncy[:,1][np.argsort(ncy[:,0])]

    #ax2.scatter(ncx[:,0],ncx[:,1],edgecolor='none',facecolor='green',s=15)
    #ax2.scatter(ncy[:,0],ncy[:,1],edgecolor='none',facecolor='blue',s=15)

    
    #ax2.plot(ncy[:,0],ncy[:,1],color='blue')
    ax.plot(ncx_ordered[:,0],ncx_ordered[:,1],color='green',lw=3)
    ax.plot(ncy_ordered[:,0],ncy_ordered[:,1],color='blue',lw=3)


    #ax.scatter(ncx[:,0],ncx[:,1],edgecolor='none',facecolor='green',s=15)
    #ax.scatter(ncy[:,0],ncy[:,1],edgecolor='none',facecolor='blue',s=15)

    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$\nu$')
    ax.set_title(r'$g=1.501$')


    # nullcline intersections (from XPP)
    r1 = .021405
    nu1 = .707
    
    r2 = 1.7474
    nu2 = .18074

    ax.scatter(r1,nu1,edgecolor='black',facecolor='white',s=60)
    ax.scatter(r2,nu2,edgecolor='black',facecolor='white',s=60)


    ax.annotate(r'$r='+str(r1)+r'$ \\ $\nu='+str(nu1)+r'$', xy=(r1+.1, nu1), xycoords='data',
                xytext=(40, 0), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc,angleA=0,armA=20,angleB=0,armB=15,rad=10"),
                )


    ax.annotate(r'$r='+str(r2)+r'$ \\ $\nu='+str(nu2)+r'$', xy=(r2-.02, nu2-.02), xycoords='data',
                xytext=(-60, -40), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc,angleA=0,armA=20,angleB=-130,armB=15,rad=7"),
                )



    ax.set_xlim(0,2)
    ax.set_ylim(0,1)


    ##### #PART 2


    #ncx = np.loadtxt("nc_phase_approx_q=0.5_g=1.75_x_mesh=100.dat")
    #ncy = np.loadtxt("nc_phase_approx_q=0.5_g=1.75_y_mesh=100.dat")

    ncx = np.loadtxt("nc_phase_approx_q=0.5_g=2_x_mesh=100.dat")
    ncy = np.loadtxt("nc_phase_approx_q=0.5_g=2_y_mesh=100.dat")

    ncy[ncy[:,1]>.85]=np.nan

    #ncx,ncy = remove_redundant(ncx,ncy,tol=.01)

    index_to_order_x_by = ncx[:,1].argsort()
    index_to_order_y_by = ncy[:,0].argsort()

    ncx_ordered = ncx[index_to_order_x_by]
    ncy_ordered = ncy[index_to_order_y_by]

    #ncx = np.loadtxt("nc_phase_approx_q=1_g=3_x_mesh100.dat")
    #ncy = np.loadtxt("nc_phase_approx_q=1_g=3_y_mesh100.dat")

    ax2 = fig.add_subplot(132)

    #ncy[:,0] = np.sort(ncy[:,0])
    #ncy[:,1] = ncy[:,1][np.argsort(ncy[:,0])]

    #ax2.scatter(ncx[:,0],ncx[:,1],edgecolor='none',facecolor='green',s=15)
    #ax2.scatter(ncy[:,0],ncy[:,1],edgecolor='none',facecolor='blue',s=15)

    
    #ax2.plot(ncy[:,0],ncy[:,1],color='blue')
    ax2.plot(ncx_ordered[:,0],ncx_ordered[:,1],color='green',lw=3)
    ax2.plot(ncy_ordered[:,0],ncy_ordered[:,1],color='blue',lw=3)

    ax2.set_xlabel(r'$r$')
    #ax2.set_ylabel(r'$\nu$')
    ax2.set_title(r'$g=2$')
    ax2.set_yticklabels([])


    # nullcline intersections (from XPP)
    r1 = 0.59458
    nu1 = 0.69031

    r2 = 1.4227
    nu2 = 0.33135
    #r1 = .41087
    #nu1 = .70345
    
    #r2 = 1.5752
    #nu2 = .25322

    ax2.scatter(r1,nu1,edgecolor='black',facecolor='white',s=60)
    ax2.scatter(r2,nu2,edgecolor='black',facecolor='white',s=60)


    ax2.annotate(r'$r='+str(r1)+r'$ \\ $\nu='+str(nu1)+r'$', xy=(r1, nu1-.025), xycoords='data',
                 xytext=(-40, -80), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->",
                                 connectionstyle="arc,angleA=0,armA=20,angleB=-90,armB=15,rad=10"),
             )


    ax2.annotate(r'$r='+str(1.4928)+r'$ \\ $\nu='+str(0.4808)+'$', xy=(r2-.02, nu2-.02), xycoords='data',
                 xytext=(-60, -40), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc,angleA=0,armA=20,angleB=-130,armB=15,rad=7"),
                )



    ax2.set_xlim(0,2)
    ax2.set_ylim(0,.8)


    ###### ## PART 3

    ncx = np.loadtxt("nc_phase_approx_q=0.5_g=2.44_x_mesh=100.dat")
    ncy = np.loadtxt("nc_phase_approx_q=0.5_g=2.44_y_mesh=100.dat")

    #ncx = np.loadtxt("nc_phase_approx_q=1_g=3_x_mesh100.dat")
    #ncy = np.loadtxt("nc_phase_approx_q=1_g=3_y_mesh100.dat")

    ax3 = fig.add_subplot(133)

    #ncy[:,0] = np.sort(ncy[:,0])
    #ncy[:,1] = ncy[:,1][np.argsort(ncy[:,0])]

    ax3.scatter(ncx[:,0],ncx[:,1],edgecolor='none',facecolor='green',s=15)
    ax3.scatter(ncy[:,0],ncy[:,1],edgecolor='none',facecolor='blue',s=15)

    ax3.set_xlabel(r'$r$')
    #ax3.set_ylabel(r'$\nu$')
    ax3.set_title(r'$g=2.44$')
    ax3.set_yticklabels([])

    # nullcline intersections (from XPP)
    r1 = .41087
    nu1 = .70345
    
    r2 = 1.5752
    nu2 = .25322

    #ax3.scatter(r1,nu1,edgecolor='black',facecolor='white',s=60)
    #ax3.scatter(r2,nu2,edgecolor='black',facecolor='white',s=60)

    """
    ax2.annotate(r'$r='+str(r1)+r'$ \\ $\nu='+str(nu1)+r'$', xy=(r1, nu1-.05), xycoords='data',
                xytext=(-30, -50), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc,angleA=0,armA=20,angleB=-90,armB=15,rad=10"),
                )


    ax2.annotate(r'$r='+str(1.4928)+r'$ \\ $\nu='+str(0.4808)+'$', xy=(r2-.02, nu2-.02), xycoords='data',
                 xytext=(-60, -40), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc,angleA=0,armA=20,angleB=-90,armB=15,rad=7"),
                )
    """


    ax3.set_xlim(0,2)
    ax3.set_ylim(0,1)



    
    return fig





def oned_phase_auto(choice='q1'):
    """
    1d bifurcation diagram of reduced system from auto
    see 1d.ode
    """

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)

    if choice == 'q1':
        filelist = ["bif_q1_gvary1.dat","bif_q1_gvary2a.dat","bif_q1_gvary2b.dat","bif_q1_gvary2c.dat"]
    elif choice == 'q0.5':
        filelist = ["bif_q0.5_gvary1.dat","bif_q0.5_gvary2a.dat","bif_q0.5_gvary2b.dat","bif_q0.5_gvary_travel.dat"]#,"bif_q0.5_gvary2c.dat"]

    branchidx = 0
    for filename in filelist:
        # get all branches
        bif_qg1 = np.loadtxt(filename)
        branchlist = np.unique(bif_qg1[:,-2])
        #if len(bif_qg1[0,:]==6):
        #    branchlist = np.unique(bif_qg1[:,-2])
        print branchlist

        # first branch

        stabe = False
        ustabe = False
        stabp = False
        ustabp = False

        for b in branchlist:
            b_idx = bif_qg1[:,-2] == b
            typelist = np.unique(bif_qg1[b_idx,-3])
            for t in typelist:
                t_idx = bif_qg1[:,-3] == t
                dat = bif_qg1[b_idx*t_idx,:]
                label = None

                if t == 1:
                    lw=3;color='red'
                    marker = None
                    if branchidx == 0 and not(stabe):
                        label='Stable Equilibrium'
                        ls = '-'
                        stab = True
                    else:
                        label=None

                elif t == 2:
                    lw=1;color='black'
                    marker = None
                    if branchidx == 0 and not(ustabe):
                        label='Unstable Equilibrium'
                        ls = '-'
                        ustabe = True
                    else:
                        label=None

                elif t == 3:
                    lw=3;color='green'
                    #marker = 'o'
                    marker = None
                    if branchidx == 0 and not(stabp):
                        label='Stable Periodic'
                        ls='-'
                        stabp = True
                    else:
                        label=None

                elif t == 4:
                    lw=1;color='blue'
                    #marker = 'o'
                    marker = None
                    if branchidx == 0 and not(ustabp):
                        label='Unstable Periodic'
                        ls='-'
                        ustabp = True
                    else:
                        label=None
                #print b,t
                alpha = 1

                me = 5

                if filename == "bif_q0.5_gvary_travel.dat":
                    ls = '--'
                else:
                    ls = '-'

                if filename == "bif_q1_gvary1.dat" or \
                   filename == "bif_q0.5_gvary1.dat" or \
                   filename == "bif_q0.5_gvary2a.dat" or \
                   filename == "bif_q0.5_gvary2b.dat":

                    ax.plot(clean(dat[:,0],dat[:,1])[0],clean(dat[:,0],dat[:,1])[1]+2*pi,lw=lw,color=color,marker=marker,markevery=me,ls=ls)
                    ax.plot(clean(dat[:,0],dat[:,2])[0],clean(dat[:,0],dat[:,2])[1]+2*pi,lw=lw,color=color,marker=marker,markevery=me,ls=ls)
                    if filename == "bif_q1_gvary1.dat" or \
                       filename == "bif_q0.5_gvary1.dat":

                        ax.plot(clean(dat[:,0],dat[:,1])[0],clean(dat[:,0],dat[:,1])[1]-2*pi,lw=lw,color=color,marker=marker,markevery=me,ls=ls)
                        ax.plot(clean(dat[:,0],dat[:,2])[0],clean(dat[:,0],dat[:,2])[1]-2*pi,lw=lw,color=color,marker=marker,markevery=me,ls=ls)

                """
                if filename == "bif_q1_gvary2a.dat" or \
                   filename == "bif_q0.5_gvary2a.dat" or \
                   filename == "bif_q0.5_gvary2c.dat":
                    label = None
                    alpha = 0.5
                else:
                    alpha = 1.
                """

                if filename == "bif_q0.5_gvary_travel.dat":
                    label = None
                    ax.plot(clean(dat[:,0],dat[:,1])[0],-(clean(dat[:,0],dat[:,1])[1]-2*pi-pi)+pi,
                            lw=lw,color=color,alpha=alpha,marker=marker,markevery=me,ls=ls)
                    ax.plot(clean(dat[:,0],dat[:,2])[0],clean(dat[:,0],dat[:,2])[1]-2*pi,
                            lw=lw,color=color,alpha=alpha,label=label,marker=marker,markevery=me,ls=ls)
                else:

                    ax.plot(clean(dat[:,0],dat[:,1])[0],clean(dat[:,0],dat[:,1])[1],
                            lw=lw,color=color,alpha=alpha,marker=marker,markevery=me,ls=ls)
                    ax.plot(clean(dat[:,0],dat[:,2])[0],clean(dat[:,0],dat[:,2])[1],
                            lw=lw,color=color,alpha=alpha,label=label,marker=marker,markevery=me,ls=ls)
            print branchidx, label
        branchidx += 1

    if choice == 'q0.5':
        ax.annotate("BP",color='teal',
                    xy=(2.36581,1.7138), xycoords='data',
                    xytext=(2, 2.5), textcoords='data',
                    size=15, va="center", ha="center",
                    arrowprops=dict(arrowstyle="->",
                                    relpos=(0., 0.),
                                    fc="w",color='teal'), 
                )

        ax.annotate("HB",color='orange',
                    xy=(1.5,0), xycoords='data',
                    xytext=(1.2, 1.2), textcoords='data',
                    size=15, va="center", ha="center",
                    arrowprops=dict(arrowstyle="->",
                                    relpos=(0., 0.),
                                    fc="w",color='orange'), 
                )
        
        ax.annotate("LP 2",color='purple',
                    xy=(2.65599, 11.8292-2*pi), xycoords='data',
                    xytext=(2.3, 2*pi-.3), textcoords='data',
                    size=15, va="center", ha="center",
                    arrowprops=dict(arrowstyle="->",
                                    relpos=(0., 0.),
                                    fc="w",color='purple'), 
                )
        
        #ax.annotate(r"\colorbox{blue!20}{{\color{yellow}LP Large}}",
        ax.annotate("LP 1",
                    xy=(2.20126, 0.847046), xycoords='data',
                    xytext=(2.7, .40746), textcoords='data',
                    size=15, va="center", ha="center",
                    arrowprops=dict(arrowstyle="->",
                                    relpos=(0., 0.),
                                    fc="w"), 
                )

    # unstable equilib
    ax.plot([0,5],[pi,pi],color='black')
    ax.plot([0,5],[-pi,-pi],color='black')
    
    # mark bistability
    ax.plot([2.20126,2.20126],[-10,10],color='black',ls=':')
    ax.plot([2.34017,2.34017],[-10,10],color='black',ls=':')
    
    # labels
    ax.set_xlabel(r'$\bm{g}$',size=15)
    ax.set_ylabel(r'$\theta$',size=15)
    
    # set y axis ticks to multiples of pi
    ax.set_ylim(-pi-.1,2*pi+.1)
    ax.set_xlim(0,5)
    
    ax.set_yticks(np.arange(-1,2+1.,1.)*pi)
    #y_label = [r"$-3\pi$", r"$-2\pi$",
    #           r"$-\pi$", r"$0$",
    #           r"$\pi$",r"$2\pi$",r"$3\pi$"]
    y_label = [r"$\bm{-\pi}$", r"$\bm{0}$",
               r"$\bm{\pi}$", r"$\bm{2\pi}$"]

    ax.set_yticklabels(y_label)
    
    ax.legend(loc='lower left',fontsize=10)

    """
    2 param bifurcation diagram from auto
    """

    #fig = plt.figure(figsize=(7.5,7.5))
    #fig = plt.figure()
    ax2 = fig.add_subplot(122)

    namelist = ['BP','HB','LP 2','LP 1']
    colorlist = ['teal','orange','purple','black']
    filelist = ["bif_gq_bp.dat","bif_gq_hb.dat","bif_gq_lp_travel.dat","bif_gq_lp_large.dat"]
    ls = ['-', '--', '-.', ':']

    i = 0
    for filename in filelist:
        # get all branches
        bif_qg1 = np.loadtxt(filename)
        branchlist = np.unique(bif_qg1[:,-2])
        #if len(bif_qg1[0,:]==6):
        #    branchlist = np.unique(bif_qg1[:,-2])
        print branchlist

        # first branch
        bidx = 0
        for b in branchlist:
            b_idx = bif_qg1[:,-2] == b
            typelist = np.unique(bif_qg1[b_idx,-3])
            for t in typelist:
                t_idx = bif_qg1[:,-3] == t
                dat = bif_qg1[b_idx*t_idx,:]
                if bidx == 0:
                    label = namelist[i]
                else:
                    label = None
                
                ax2.plot(clean(dat[:,0],dat[:,1])[0],clean(dat[:,0],dat[:,1])[1],
                        lw=2,color=colorlist[i],label=label,ls=ls[i])
                #ax2.plot(clean(dat[:,0],dat[:,2])[0],clean(dat[:,0],dat[:,2])[1],
                #        lw=2,color=colorlist[i],label=label)
            bidx += 1
        i += 1


    ax2.text(1.5,2.6,'1. Stationary Bump',rotation=0,size=15)
    ax2.text(3.3,2.5,'2. Sloshing Bump',rotation=37,size=15)
    ax2.text(3.5,1.55,'3. Bistability',rotation=27,size=15)
    ax2.text(3.2,.3,'4. Traveling Bump',rotation=0,size=15)

    ax2.plot([0,5],[.5,.5],color='gray')
    
    #ax2.text(1.5,.075,'reminder: added line from g=2 to g=1 for LP2')
    
    ax2.legend(loc='upper left',fontsize=10)
    ax2.set_xlabel(r'$\bm{g}$',size=15)
    ax2.set_ylabel(r'$\bm{q}$',size=15)

    ax2.set_xlim(1,5)




    ax.xaxis.set_major_formatter(formatter)
    #ax.yaxis.set_major_formatter(formatter)

    ax2.xaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter) 


    return fig



def draw_branches(ax,filelist,smallscale=False):

    branchidx = 0
    for filename in filelist:
        # get all branches
        bif_qg1 = np.loadtxt(filename)
        branchlist = np.unique(bif_qg1[:,-2])
        #if len(bif_qg1[0,:]==6):
        #    branchlist = np.unique(bif_qg1[:,-2])
        #print branchlist

        # first branch

        stabe = False
        ustabe = False
        stabp = False
        ustabp = False

        for b in branchlist:
            b_idx = bif_qg1[:,-2] == b
            typelist = np.unique(bif_qg1[b_idx,-3])
            #print typelist
            for t in typelist:
                print 'branch',b,'type',t
                t_idx = bif_qg1[:,-3] == t
                dat = bif_qg1[b_idx*t_idx,:]
                label = None

                if t == 1:
                    lw=3;color='red'
                    marker = None
                    if branchidx == 0 and not(stabe):
                        label='Stable Equilibrium'
                        ls = '-'
                        stab = True
                    else:
                        label=None

                elif t == 2:
                    lw=1;color='black'
                    marker = None
                    if branchidx == 0 and not(ustabe):
                        label='Unstable Equilibrium'
                        ls = '-'
                        ustabe = True
                    else:
                        label=None

                elif t == 3:
                    lw=3;color='green'
                    #marker = 'o'
                    marker = None
                    if branchidx == 0 and not(stabp):
                        label='Stable Periodic'
                        ls='-'
                        stabp = True
                    else:
                        label=None

                elif t == 4:
                    lw=1;color='blue'
                    #marker = 'o'
                    marker = None
                    if branchidx == 0 and not(ustabp):
                        label='Unstable Periodic'
                        ls='-'
                        ustabp = True
                    else:
                        label=None
                #print b,t
                alpha = 1

                me = 5

                if (filename == 'bif_full_q0.5_gvary2c.dat') or\
                   (filename == "bif_full_q0.5_gvary_travel.dat") or\
                   (filename == "bif_full_a2_q0.5_gvary2.dat") or\
                   (filename == "bif_full_a3_q0.5_gvary2.dat"):
                    ls='--'
                else:
                    ls='-'

                if filename == "bif_full_q1_gvary1.dat" or \
                   filename == "bif_full_q0.5_gvary1.dat" or \
                   filename == "bif_full_q0.5_gvary2a.dat" or \
                   filename == "bif_full_q0.5_gvary2b.dat":
                   

                    ax.plot(clean(dat[:,0],dat[:,1],smallscale=smallscale)[0],clean(dat[:,0],dat[:,1],smallscale=smallscale)[1]+2*pi,lw=lw,color=color,marker=marker,markevery=me,ls=ls)
                    ax.plot(clean(dat[:,0],dat[:,2],smallscale=smallscale)[0],clean(dat[:,0],dat[:,2],smallscale=smallscale)[1]+2*pi,lw=lw,color=color,marker=marker,markevery=me,ls=ls)
                elif filename == "bif_full_q1_gvary1.dat" or \
                     filename == "bif_full_q0.5_gvary1.dat" or \
                     filename == "bif_full_q0.5_gvary2c.dat":
                    #print 'gvary_2c',filename
                    ax.plot(clean(dat[:,0],dat[:,1],smallscale=smallscale)[0],clean(dat[:,0],dat[:,1],smallscale=smallscale)[1]-2*pi,lw=lw,color=color,marker=marker,markevery=me,ls=ls)
                    ax.plot(clean(dat[:,0],dat[:,2],smallscale=smallscale)[0],clean(dat[:,0],dat[:,2],smallscale=smallscale)[1]-2*pi,lw=lw,color=color,marker=marker,markevery=me,ls=ls)

                """
                if filename == "bif_q1_gvary2a.dat" or \
                   filename == "bif_q0.5_gvary2a.dat" or \
                   filename == "bif_q0.5_gvary2c.dat":
                    label = None
                    alpha = 0.5
                else:
                    alpha = 1.
                """
                


                if filename == "bif_full_q0.5_gvary_travel.dat" or\
                   filename == "bif_full_q0.5_gvary2c.dat":
                    label = None

                    ax.plot(clean(dat[:,0],dat[:,1],smallscale=smallscale)[0],-(clean(dat[:,0],dat[:,1],smallscale=smallscale)[1]-2*pi-pi)+pi,
                            lw=lw,color=color,alpha=alpha,marker=marker,markevery=me,ls=ls)
                    ax.plot(clean(dat[:,0],dat[:,2],smallscale=smallscale)[0],clean(dat[:,0],dat[:,2],smallscale=smallscale)[1]-2*pi,
                            lw=lw,color=color,alpha=alpha,label=label,marker=marker,markevery=me,ls=ls)
                else:

                    ax.plot(clean(dat[:,0],dat[:,1],smallscale=smallscale)[0],clean(dat[:,0],dat[:,1],smallscale=smallscale)[1],
                            lw=lw,color=color,alpha=alpha,marker=marker,markevery=me,ls=ls)
                    ax.plot(clean(dat[:,0],dat[:,2],smallscale=smallscale)[0],clean(dat[:,0],dat[:,2],smallscale=smallscale)[1],
                            lw=lw,color=color,alpha=alpha,label=label,marker=marker,markevery=me,ls=ls)

        branchidx += 1


    return ax
    

def draw_branches_twop(ax,filelist,namelist,colorlist,ls):

    i = 0

    for filename in filelist:
        # get all branches
        bif_qg1 = np.loadtxt(filename)
        branchlist = np.unique(bif_qg1[:,-2])
        #if len(bif_qg1[0,:]==6):
        #    branchlist = np.unique(bif_qg1[:,-2])
        print branchlist

        # first branch
        bidx = 0
        for b in branchlist:
            b_idx = bif_qg1[:,-2] == b
            typelist = np.unique(bif_qg1[b_idx,-3])
            for t in typelist:
                t_idx = bif_qg1[:,-3] == t
                dat = bif_qg1[b_idx*t_idx,:]
                if bidx == 0:
                    label = namelist[i]
                else:
                    label = None
                
                ax.plot(clean(dat[:,0],dat[:,1])[0],clean(dat[:,0],dat[:,1])[1],
                        lw=2,color=colorlist[i],label=label,ls=ls[i])
                #ax2.plot(clean(dat[:,0],dat[:,2])[0],clean(dat[:,0],dat[:,2])[1],
                #        lw=2,color=colorlist[i],label=label)
            bidx += 1
        i += 1
    return ax

def oned_full_auto():
    """
    1d bifurcation diagram from auto
    see numerical_bard_sep.ode
    """

    filelista2 = ["bif_full_a2_q0.5_gvary1.dat","bif_full_a2_q0.5_gvary2.dat","bif_full_a2_q0.5_gvary2b.dat"]
    filelista3 = ["bif_full_a3_q0.5_gvary1.dat","bif_full_a3_q0.5_gvary2.dat","bif_full_a3_q0.5_gvary2b.dat"]

    # data files obtained using numerical_bard_sep.ode

    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot2grid((2,2),(0,0))




    ax = draw_branches(ax,filelista2)

    # labels
    ax.set_xticks([])
    ax.set_ylabel(r'$\bm{a_1}$',size=15)
    
    # set y axis ticks to multiples of pi
    ax.set_ylim(-1.5,1)
    ax.set_xlim(0,5)

    ax.set_title(AA,x=0,y=1.05)
    
    #ax.legend(loc='lower left',fontsize=10)

    """
    ax.annotate("BP",color='teal',
    xy=(2.36581,1.7138), xycoords='data',
    xytext=(2, 2.5), textcoords='data',
    size=15, va="center", ha="center",
    arrowprops=dict(arrowstyle="->",
    relpos=(0., 0.),
    fc="w",color='teal'), 
    )
    """

    ax.annotate("HB",color='orange',
                xy=(1.50704,0.78737), xycoords='data',
                xytext=(.75, .5), textcoords='data',
                size=15, va="center", ha="center",
                arrowprops=dict(arrowstyle="->",
                                relpos=(0., 0.),
                                fc="w",color='orange'), 
            )

    # LP 2 label inset
    ax.annotate("LP",color='purple',
                   xy=(2.75, .796), xycoords='data',
                   xytext=(3.2, .4), textcoords='data',
                   size=15, va="center", ha="center",
                   arrowprops=dict(arrowstyle="->",
                                   relpos=(0., 0.),
                                   fc="w",color='purple'), 
               )
    
    
    #ax.annotate(r"\colorbox{blue!20}{{\color{yellow}LP Large}}",
    """
    ax.annotate("LP 1",
    xy=(2.20126, 0.847046), xycoords='data',
    xytext=(2.7, .40746), textcoords='data',
    size=15, va="center", ha="center",
    arrowprops=dict(arrowstyle="->",
    relpos=(0., 0.),
    fc="w"), 
    )
    """

    # inset
    axins = inset_axes(ax,
                       width="30%", # width = 30% of parent_bbox
                       height="50%", # height : 1 inch
                       loc=3)
    axins = draw_branches(axins,filelista2,smallscale=True)
    axins.set_xlim(2.1,2.9)
    axins.set_ylim(.78,.82)

    # bistability for inset
    axins.plot([2.25346,2.25346],[-2,2],ls=':',color='black')
    axins.plot([2.38425,2.38425],[-2,2],ls=':',color='black')

    # LP 2 label inset
    axins.annotate("LP",color='purple',
                   xy=(2.81, .796), xycoords='data',
                   xytext=(2.68, .785), textcoords='data',
                   size=15, va="center", ha="center",
                   arrowprops=dict(arrowstyle="->",
                                   relpos=(0., 0.),
                                   fc="w",color='purple'), 
               )

    
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    
    
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    # mark bistability
    ax.plot([2.25346,2.25346],[-2,2],ls=':',color='black')
    ax.plot([2.38425,2.38425],[-2,2],ls=':',color='black')


    ax2 = plt.subplot2grid((2,2),(1,0))
    ax2.set_title(BB,x=0,y=1.05)
    
    ax2 = draw_branches(ax2,filelista3)
    
    ax2.set_ylabel(r'$\bm{a_2}$',size=15)
    ax2.set_xlabel(r'$\bm{g}$',size=15)
    ax2.set_ylim(-1,.1)
    ax2.set_xlim(0,5)


    # inset
    axins = inset_axes(ax2,
                       width="30%", # width = 30% of parent_bbox
                       height=1., # height : 1 inch
                       loc=1)
    axins = draw_branches(axins,filelista3,smallscale=True)
    axins.set_xlim(2.1,3.)
    axins.set_ylim(-.83,-.73)

    # inset bistability
    axins.plot([2.25346,2.25346],[-2,2],ls=':',color='black')
    axins.plot([2.38425,2.38425],[-2,2],ls=':',color='black')
    
    # inset lp2
    axins.annotate("LP",color='purple',
                   xy=(2.84, -.779), xycoords='data',
                   xytext=(2.7, -.75), textcoords='data',
                   size=15, va="center", ha="center",
                   arrowprops=dict(arrowstyle="->",
                                   relpos=(0., 0.),
                                   fc="w",color='purple'), 
               )

    
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    plt.xticks(visible=False)
    plt.yticks(visible=False)


    mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")



    """
    ax2.annotate("BP",color='teal',
    xy=(2.36581,1.7138), xycoords='data',
    xytext=(2, 2.5), textcoords='data',
    size=15, va="center", ha="center",
    arrowprops=dict(arrowstyle="->",
    relpos=(0., 0.),
    fc="w",color='teal'), 
    )
    """

    ax2.annotate("HB",color='orange',
                xy=(1.50704,0.), xycoords='data',
                 xytext=(1., -.2), textcoords='data',
                size=15, va="center", ha="center",
                arrowprops=dict(arrowstyle="->",
                                relpos=(0., 0.),
                                fc="w",color='orange'), 
            )
    
    ax2.annotate("LP",color='purple',
                 xy=(2.84, -.779), xycoords='data',
                 xytext=(3., -.5), textcoords='data',
                size=15, va="center", ha="center",
                arrowprops=dict(arrowstyle="->",
                                relpos=(0., 0.),
                            fc="w",color='purple'), 
            )
    
    #ax2.annotate(r"\colorbox{blue!20}{{\color{yellow}LP Large}}",
    """
    ax2.annotate("LP 1",
    xy=(2.20126, 0.847046), xycoords='data',
    xytext=(2.7, .40746), textcoords='data',
    size=15, va="center", ha="center",
    arrowprops=dict(arrowstyle="->",
    relpos=(0., 0.),
    fc="w"), 
    )
    """
    
    


    #ax.set_xlabel(r'$\bm{g}$',size=15)

    
    # mark bistability
    ax2.plot([2.25346,2.25346],[-2,2],ls=':',color='black')
    ax2.plot([2.38425,2.38425],[-2,2],ls=':',color='black')

    
    

    ax3 = plt.subplot2grid((2,2),(0,1),rowspan=2)
    ax3.set_title(CC,x=0,y=1.02)

    namelist = ['HB','LP']#['BP','HB','LP 2','LP 1']
    colorlist = ['orange','purple']#['teal','orange','purple','black']
    filelist = ['bif_full_a2_gq_hb.dat','bif_full_a2_gq_lp2.dat']#,'bif_full_gq_lp2.dat']#["bif_gq_bp.dat","bif_gq_hb.dat","bif_gq_lp_travel.dat","bif_gq_lp_large.dat"]
    ls = ['--','-.','-', '--', '-.', ':']

    ax3 = draw_branches_twop(ax3,filelist,namelist,colorlist,ls)


    ax3.text(1.5,2.6,'1. Stationary Bump',rotation=0,size=15)
    ax3.text(3.3,2.5,'2. Sloshing Bump',rotation=37,size=15)
    #ax2.text(3.5,1.55,'3. Bistability',rotation=27,size=15)
    ax3.text(3.2,.3,'4. Traveling Bump',rotation=0,size=15)
    
    ax3.plot([0,5],[.5,.5],color='gray')
    
    #ax2.text(1.5,.075,'reminder: added line from g=2 to g=1 for LP2')
    
    #ax3.legend(loc='upper left',fontsize=8)
    ax3.set_xlabel(r'$\bm{g}$',size=15)
    ax3.set_ylabel(r'$\bm{q}$',size=15)

    ax3.set_xlim(1,5)
    ax3.legend(loc='upper left')


    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax2.xaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter) 

    ax3.xaxis.set_major_formatter(formatter)
    ax3.yaxis.set_major_formatter(formatter) 



    return fig


def root(ushift,g):
    """
    find slow limit cycle/wobbling bump
    """
    time = 882.4*5
    sim = oned_simple.SimDat(q=0.5,g=g,T=time,ushift=ushift,zshift=1e-5)
    max_loc = np.r_[True, sim.ph_angle[1:] > sim.ph_angle[:-1]] & np.r_[sim.ph_angle[:-1] > sim.ph_angle[1:], True]
    local_maxima = sim.ph_angle[max_loc][1:-1]
    diffraw = local_maxima[-2] - local_maxima[-3]

    print 'diffraw=',diffraw,'ushift=',ushift

    return diffraw


def oned_normal_form():
    """
    1d normal form calculation. probably incorrect. see bard's normal form calculation below.
    """
    fig = plt.figure(figsize=(8,4))
    #ax = fig.add_subplot(111)
    ax = plt.subplot2grid((1,2),(0,0))
    ax2 = plt.subplot2grid((1,2),(0,1))
    #ax3 = plt.subplot2grid((2,1),(1,1))#fig.add_subplot(132)


    ax.set_title(AA,x=0,y=1.05)

    gvals_long = np.linspace(1.5,2.,201) # use in theory


    # theory
    ss = oned_simple.SteadyState()
    mu = ss.kap
    Aprime = ss.Hamp

        
    # get a better approximation later.
    eps = .01
    period = eps*882.4 # period in tau (period in t times eps)
    #period = eps*441.2 # period in tau
    om = 2*pi/period

    # for a cosine kernel, H(x) = A'sin(x)
    h1 = Aprime*1.
    h3 = Aprime*(-1./6)
    be = 1.
    q = .5
    gstar = (mu*be - q*(-Aprime))/Aprime#+.00625

    print Aprime,om,gstar

    #B = 2*sqrt( -(be**2 + 4.*om**2)*h1/(gstar*h3) )/(6.*om)

    f1 = sqrt(be**2. + om**2.)
    #f2 = sqrt(be**2. + 4*om**2.)
    
    #B = 2*om*sqrt(h1*f2)/sqrt(h3*(12.*q*f1*f2-144.*gstar*om**4))
    B = 2.*(2.*sqrt(h1*om))/(f1*sqrt(h3*((12.*q)/om - (144.*gstar*om**3.)/(be**4. + 5.*be**2.*om**2. + 4.*om**4.))))
    amp = B*sqrt(gvals_long-gstar)

    #amp = sqrt(g-gstar)

    # numerics
    #gvals_short = [1.5,1.505,1.51,1.515]
    #gvals_short = np.linspace(1.5,1.75,41)
    gvals_short = np.arange(1.50625,2.,.00625)
    amp_num = np.zeros(len(gvals_short))
    i = 0
    ushift = 0.
    zshift = 1e-5

    savedir = 'hopf_data/'

    if (not os.path.exists(savedir)):
        os.makedirs(savedir)
        
    for g in gvals_short:
        time = 1500.#882.4*2
        print "g="+str(g)
        tol = 5e-4


        filename = 'osc_g='+str(g)+'.dat'
        if os.path.isfile(savedir+filename):
            local_max = float(open(savedir+filename,'r').readline())
            #print local_max
        else:
            #print
            #ss_time = sim.t[int(time/sim.dt/1.5):]

            sim = oned_simple.SimDat(q=0.5,g=g,T=time,ushift=0,zshift=.1,sim_factor=70)
            max_loc = np.r_[True, sim.ph_angle[1:] > sim.ph_angle[:-1]] & np.r_[sim.ph_angle[:-1] > sim.ph_angle[1:], True]
            local_maxima = sim.ph_angle[max_loc][1:-1]
            local_max = local_maxima[-1]
            file_ = open(savedir+filename,'w')
            file_.write(str(local_max))
            file_.close()
        amp_num[i] = local_max


        #mp.figure()
        #mp.plot(sim.t,sim.ph_angle)
        #mp.show()
        #ushift = sp.optimize.brentq(root,0,pi,args=(g,),rtol=1e-4)
        
        i += 1

    
    data = np.zeros((len(gvals_short),2))
    data[:,0] = gvals_short
    data[:,1] = amp_num

    """
    ax.annotate("("+str(data[20,0])+","+str(data[20,1])+")",
                 xy=(data[20,0], data[20,1]), xycoords='data',
                 xytext=(data[20,0]-.05, data[20,1]+.05), textcoords='data',
                 size=12,
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3")
             )
    """



    #np.savetxt("hopf_amplitude.dat")

    ax.plot(gvals_long,amp,color="#3399ff",ls='dashed',label="Theoretical",lw=3)
    ax.plot(gvals_short,amp_num,color="black",label="Numerical",lw=3)

    ax.set_xlim(data[:,0][0]-.01,data[:,0][-1]+.01)
    ax.set_ylabel(r"\textbf{Oscillation Amplitude (A)}")
    ax.set_xlabel(r"\textbf{Adaptation (g)}")

    ax.legend(loc=4)



    """
    PLOT SOLUTION ARRAY
    """

    dat = oned_simple.SimDat(g=1.55,q=.5,zshift=.1,T=10000,phase=True)
    # period is approx 525 time units
    shift = -700
    sign = 1

    ax2.set_xlabel(r'$x$')
    ax2.set_title(BB,x=0)
    #ax2.set_ylabel(r'$t$')

    start_idx = len(dat.t)/2.
    end_idx = int(1.5*start_idx)

    idx = np.arange(start_idx,end_idx+1,1,dtype='int')
    
    cax = ax2.matshow(np.roll(dat.sol[idx,:dat.N],dat.N/2),cmap='gray',extent=[-pi,pi,dat.t[end_idx],dat.t[start_idx]])
    #fig.colorbar(cax)
    ax2.xaxis.tick_bottom()
    ax2.xaxis.set_label_position('bottom')
    
    
    timearr = np.linspace(dat.t[start_idx],dat.t[end_idx],len(idx))
    for slc in unlink_wrap(dat.ph_angle[idx]):
        ax2.plot(dat.ph_angle[idx][slc],timearr[slc],color='black',lw=3)
        
    modsolph = -(np.mod(dat.solph[idx+shift,0]+pi,2*pi)-pi)*sign
    for slc in unlink_wrap(modsolph):
        ax2.plot(modsolph[slc],timearr[slc],ls='--',color='#3399ff',lw=3)
    #ax.plot(dat.ph_angle[idx],np.linspace(dat.t[start_idx],dat.t[end_idx],len(idx)),color='black',lw=3)
    #ax.plot(-(np.mod(dat.solph[idx+shift,0]+pi,2*pi)-pi)*sign,np.linspace(dat.t[start_idx],dat.t[end_idx],len(idx)),ls='--',color='.65',lw=2)
    print 'shifted oned_nonconst_vel_bump ana by ', shift, 'where dt=',dat.dt
    
    ax2.set_aspect('auto')

    ax2.set_xlim(-pi,pi)
    ax2.set_ylim(dat.t[end_idx],dat.t[start_idx])

    ax2.set_xticks(np.arange(-1,1+.5,.5)*pi)
    ax2.set_xticklabels(x_label)





    return fig


def oned_normal_form_bard():
    """
    oned normal form using bard's data
    """

    filename = 'diagram.dat'
    filename2 = 'diagram.25.dat'

    # get all branches
    bif_qg1 = np.loadtxt(filename)
    branchlist = [2]

    b_idx = bif_qg1[:,-2] == branchlist[0]
    typelist = [3]#np.unique(bif_qg1[b_idx,-3])
    #print typelist
    for t in typelist:
        t_idx = bif_qg1[:,-3] == t
        dat = bif_qg1[b_idx*t_idx,:]
        label = None

    # get all branches
    bif_qg2 = np.loadtxt(filename2)
    branchlist2 = [2]

    b_idx2 = bif_qg2[:,-2] == branchlist2[0]
    typelist2 = [3]#np.unique(bif_qg2[b_idx2,-3])
    #print typelist
    for t in typelist2:
        t_idx2 = bif_qg2[:,-3] == t
        dat2 = bif_qg2[b_idx2*t_idx2,:]
        label = None

    fig = plt.figure(figsize=(10,3))

    ax = fig.add_subplot(121)

    ax.plot(dat[:,0],dat[:,1],label='AUTO',lw=3,color='black')
    dom1 = np.linspace(2,2.5,100)
    ax.plot(dom1,2*np.sqrt((10./13.)*(dom1-2)),label='Normal Form',lw=3,ls='--',color='#3399ff')

    ax.set_title(AA,x=0,y=1.03)
    ax.set_ylabel(r"\textbf{Oscillation Amplitude}")
    ax.set_xlabel(r"\textbf{Adaptation ($g$)}")

    #ax.set_xlabel(r'Adaptation ($g$)')
    #ax.set_ylabel(r'Oscillation Amplitude')
    ax.set_xlim(2,2.5)
    ax.set_ylim(0,1.4)
    #ax.legend()

    ax.set_yticks(np.arange(0,1.6,.4))

    #ax12a.set_xticklabels(x_label_short,fontsize=10)
    #ax12a.set_yticklabels(x_label_short,fontsize=10)



    ax2 = fig.add_subplot(122)

    ax2.plot(dat2[:,0],dat2[:,1],label='AUTO',lw=3,color='black')
    dom2 = np.linspace(1.25,1.5,100)
    ax2.plot(dom2,2*np.sqrt(.5*(dom2-1.25)/.2175),label='Normal Form',lw=3,ls='--',color='#3399ff')

    ax2.set_title(BB,x=0,y=1.03)
    ax2.set_ylabel(r"\textbf{Oscillation Amplitude}")
    ax2.set_xlabel(r"\textbf{Adaptation ($g$)}")


    #ax2.set_xlabel(r'Adaptation ($g$)')
    #ax2.set_ylabel(r'Oscillation Amplitude')
    ax2.set_xlim(1.25,1.5)
    ax2.set_ylim(0,1.6)

    ax2.set_yticks(np.arange(0,2.,.4))
    ax2.legend(loc='lower right')

    return fig
    

def g_nu_fig():
    """
    plot g(nu)
    """
    N = 100
    nu = np.linspace(.00001,1,N)
    
    s = np.linspace(0,10.,100)
    ds = (s[-1]-s[0])/len(s)
    g1 = np.zeros(N)
    g2 = np.zeros(N)

    for i in range(N):
        tot = 0
        tot2 = 0
        # find integral of exp(-s)*H(nu s)
        for j in range(len(s)):
            tot += np.exp(-s[j])*(sin(nu[i]*s[j])-(.25)*sin(2.*nu[i]*s[j]))*ds#np.sin(nu[i]*s[j])*ds
            tot2 += np.exp(-s[j])*(sin(nu[i]*s[j]))*ds#np.sin(nu[i]*s[j])*ds

        g1[i] = nu[i]/tot
        g2[i] = nu[i]/tot2

    fig = plt.figure(figsize=(10,3))
    ax2 = fig.add_subplot(121)
    ax2.plot(nu,g2,lw=3,ls='-',color='black')

    ax2.set_title(AA,x=0,y=1.03)

    ax2.set_ylabel(r'$\bm{\Gamma(\nu)}$',size=15)
    ax2.set_xlabel(r'$\bm{\nu}$',size=15)
    ax2.tick_params(pad=5)
    ax2.set_yticks(np.arange(1,2+.4,.4))

    ax1 = fig.add_subplot(122)

    split_idx = np.argmin(g1)
    ax1.plot(nu[:split_idx],g1[:split_idx],lw=3,ls='--',color='black')
    ax1.plot(nu[split_idx:],g1[split_idx:],lw=3,ls='-',color='black')

    ax1.set_title(BB,x=0,y=1.03)    
    ax1.set_xlabel(r'$\bm{\nu}$',size=15)
    ax1.tick_params(pad=5)
    ax1.set_yticks(np.arange(1.8,2.6+.2,.2))



    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)

    ax2.xaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)

    
    return fig
    #plt.show()


def oned_chaos_fig():
    """
    """
    fig = plt.figure(figsize=(10,3))

    ax1 = fig.add_subplot(121)
    
    c1 = np.loadtxt("chaos_simple1.dat")
    c2 = np.loadtxt("chaos_simple2.dat")

    NT = len(c1)
    t = np.linspace(0,50000,NT)
    dt = t[-1]/NT
    

    start_t = 14000
    end_t = 20000

    sidx = int(start_t/dt)
    eidx = int(end_t/dt)


    for slc in unlink_wrap(c1[sidx:eidx]):
        ax1.plot(t[sidx:eidx][slc],c1[sidx:eidx][slc],color='black',lw=2)
    for slc in unlink_wrap(c2[sidx:eidx]):
        ax1.plot(t[sidx:eidx][slc],c2[sidx:eidx][slc],color='#3399ff',lw=2,ls='--',dashes=(5,1))

    ax1.set_ylabel(r'$\bm{\theta}$')
    ax1.set_xlabel(r'$\bm{t}$')
    ax1.set_xlim(start_t,end_t)
    ax1.set_ylim(-pi,pi)

    ax1.set_yticks(np.arange(-1,1+.5,.5)*pi)
    #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
    ax1.set_yticklabels(x_label)


    ax2 = fig.add_subplot(122)
    ct1 = np.loadtxt("chaos_simple_theory1.dat")
    ct2 = np.loadtxt("chaos_simple_theory2.dat")

    NTt = len(ct1)
    t2 = np.linspace(0,50000,NTt)
    dt2 = t2[-1]/NTt

    start_t2 = 34000
    end_t2 = 40000

    sidx2 = int(start_t2/dt2)
    eidx2 = int(end_t2/dt2)

    for slc in unlink_wrap(ct1[sidx2:eidx2]):
        ax2.plot(t2[sidx2:eidx2][slc],ct1[sidx2:eidx2][slc],color='black',lw=2)

    for slc in unlink_wrap(ct2[sidx2:eidx2]):
        ax2.plot(t2[sidx2:eidx2][slc],ct2[sidx2:eidx2][slc],color='#3399ff',lw=2,ls='--',dashes=(5,1))



    #ax2.set_ylabel(r'$\bm{\theta}$')
    ax2.set_xlabel(r'$\bm{t}$')
    ax2.set_xlim(start_t2,end_t2)
    ax2.set_ylim(-pi,pi)

    ax2.set_yticks(np.arange(-1,1+.5,.5)*pi)
    #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
    ax2.set_yticklabels(x_label)
    

    return fig


def oned_chaos_fig_finer():
    """
    same as oned_chaos_fig, but with finer time discretization
    """
    
    fig = plt.figure(figsize=(10,3))

    ax1 = fig.add_subplot(111)
    
    c1 = np.loadtxt("chaos_simple1_N=240.dat")
    c2 = np.loadtxt("chaos_simple2_N=240.dat")

    NT = len(c1)
    t = np.linspace(0,50000,NT)
    dt = t[-1]/NT
    

    start_t = 5000
    end_t = 25000

    sidx = int(start_t/dt)
    eidx = int(end_t/dt)


    for slc in unlink_wrap(c1[sidx:eidx]):
        ax1.plot(t[sidx:eidx][slc],c1[sidx:eidx][slc],color='black',lw=2)
    for slc in unlink_wrap(c2[sidx:eidx]):
        ax1.plot(t[sidx:eidx][slc],c2[sidx:eidx][slc],color='#3399ff',lw=2,ls='--',dashes=(5,1))

    ax1.set_ylabel(r'$\bm{\theta}$')
    ax1.set_xlabel(r'$\bm{t}$')
    ax1.set_xlim(start_t,end_t)
    ax1.set_ylim(-pi,pi)

    ax1.set_yticks(np.arange(-1,1+.5,.5)*pi)
    #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
    ax1.set_yticklabels(x_label)


    return fig
    
def oned_phase_chaos_fig_map():
    """
    return map figure for chaos in the reduced model 
    """


    fig = plt.figure(figsize=(5,5))

    ax1 = fig.add_subplot(111)
    
    
    dat = np.loadtxt('1d_phs_chaos_map.dat')
    x = dat[:,0]
    y = dat[:,1]
    
    nskip = 10

    ax1.scatter(x[::100],y[::100],s=1,edgecolor='none',color='black')

    """
    ax1a = inset_axes(ax1,
                      width="60%", # width = 30% of parent_bbox
                      height="40%", # height : 1 inch
                      loc=3)
    """


    ax1a = inset_axes(ax1,width=1, height=1,  loc=1, 
                      bbox_to_anchor=(0.74, 0.45), 
                      bbox_transform=ax1.figure.transFigure)

    ax1a.scatter(x,y,s=1,edgecolor='none',color='black')
    ax1a.set_xlim(-.3315,-.329)
    ax1a.set_ylim(.265,.3)
    
    ax1b = inset_axes(ax1,width=1, height=1,  loc=1, 
                      bbox_to_anchor=(0.65, 0.79), 
                      bbox_transform=ax1.figure.transFigure)

    ax1b.scatter(x,y,s=1,edgecolor='none',color='black')
    ax1b.set_xlim(-.739,-.736)
    ax1b.set_ylim(.24,.31)

    ax1a.set_xticks([])
    ax1a.set_yticks([])
    ax1b.set_xticks([])
    ax1b.set_yticks([])

    mark_inset(ax1, ax1a, loc1=2, loc2=4, fc="none", ec="0.5")
    mark_inset(ax1, ax1b, loc1=2, loc2=4, fc="none", ec="0.5")
    
    ax1.set_xlabel(r'$S(\tau)$')
    ax1.set_ylabel(r'$C(\tau)$')
    
    return fig
    

def twod_auto_3terms_fig_old():
    """
    twod bifurcation diagram for truncated h
    """
    #raw_data = np.loadtxt('twodphs_cys_wave_diagram_q=.125.dat')

    if True:
        raw_data = np.loadtxt('twodphs_sxs_wave_diagram_q=.125.dat')
        raw_data2 = np.loadtxt('twodphs_sxs_osc2_diagram_q=.125.dat')
        raw_data3 = np.loadtxt('twodphs_sxs_hopf_diagram_q=.125.dat')
        raw_data4 = np.loadtxt('twodphs_sxs_wave2_diagram_q=.125.dat')

    if False:
        raw_data = np.loadtxt('twodphs_cys_wave_diagram_q=.125.dat')
        raw_data2 = np.loadtxt('twodphs_cys_osc2_diagram_q=.125.dat')
        raw_data3 = np.loadtxt('twodphs_cys_hopf_diagram_q=.125.dat')
        raw_data4 = np.loadtxt('twodphs_cys_wave2_diagram_q=.125.dat')

    if False:
        raw_data = np.loadtxt('twodphs_x_wave_diagram_q=.125.dat')
        raw_data2 = np.loadtxt('twodphs_cys_osc2_diagram_q=.125.dat')
        raw_data3 = np.loadtxt('twodphs_x_hopf_diagram_q=.125.dat')
        raw_data4 = np.loadtxt('twodphs_x_wave2_diagram_q=.125.dat')

    
    data = diagram.read_diagram(raw_data)
    data2 = diagram.read_diagram(raw_data2)
    data3 = diagram.read_diagram(raw_data3)
    data4 = diagram.read_diagram(raw_data4)

    print np.shape(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot unstable fixed points
    ax.scatter(data[:,0],data[:,2],s=10,color='black')
    ax.scatter(data[:,0],data[:,6],s=10,color='black')

    ax.scatter(data3[:,0],data3[:,2],s=10,color='black')
    ax.scatter(data3[:,0],data3[:,6],s=10,color='black')


    # plot unstable periodic solutions
    ax.scatter(data[:,0],data[:,4],s=10,facecolor='none',edgecolor='blue')
    #ax.scatter(data[:,0],data[:,8],s=10,facecolor='none',edgecolor='blue')

    ax.scatter(data2[:,0],data2[:,4],s=10,facecolor='none',edgecolor='blue')
    ax.scatter(data2[:,0],data2[:,8],s=10,facecolor='none',edgecolor='blue')

    ax.scatter(data3[:,0],data3[:,4],s=10,facecolor='none',edgecolor='blue')
    ax.scatter(data3[:,0],data3[:,8],s=10,facecolor='none',edgecolor='blue')

    ax.scatter(data4[:,0],data4[:,4],s=10,facecolor='none',edgecolor='#0099ff')
    ax.scatter(data4[:,0],data4[:,8],s=10,facecolor='none',edgecolor='#0099ff')


    # plot stable fixed points
    ax.scatter(data[:,0],data[:,1],s=10,color='red')
    ax.scatter(data[:,0],data[:,5],s=10,color='red')

    ax.scatter(data3[:,0],data3[:,1],s=10,color='red')
    ax.scatter(data3[:,0],data3[:,5],s=10,color='red')

    # plot stable periodic solutions
    ax.scatter(data[:,0],data[:,3],s=20,color='green')
    ax.scatter(data[:,0],data[:,7],s=20,color='green')

    ax.scatter(data2[:,0],data2[:,3],s=20,color='green')
    ax.scatter(data2[:,0],data2[:,7],s=20,color='green')

    ax.scatter(data3[:,0],data3[:,3],s=20,color='green')
    ax.scatter(data3[:,0],data3[:,7],s=20,color='green')

    ax.scatter(data4[:,0],data4[:,3],s=20,color='#00cc00')
    ax.scatter(data4[:,0],data4[:,7],s=20,color='#00cc00')

    ax.set_xlim(.5,3)
    ax.set_ylim(-.01,1.)





    return fig


def get_switch_points(data):
    """
    given allinfo bifurcation diagram data, find all locations where stability changes.
    
    """
    
    pass

def twod_phase_auto_3terms_fig1():
    """
    twod bifurcation diagram for truncated h
    q = 0.01
    """
    #raw_data = np.loadtxt('twodphs_cys_wave_diagram_q=.125.dat')

    # all info data files are organized as follows:
    # Type, BR, 0, par1, par1/2, period, sv1 (high), ..., sv10 (high), sv1 (low),...,sv10(high), real/im eigenvalue pairs...

    # so I could use these values as initial conditions to plot.

    bif_data = np.loadtxt('twodphs_3_sxs_TR_q=.01.dat')
    init_data = np.loadtxt('twodphs_3_init_TR_q=.01.dat')

    # manually get index of sxs value
    idx = 5

    fig = plt.figure(figsize=(6,7))


    ### BIFURCATION DIAGRAM
    gs = gridspec.GridSpec(4, 4)
    gs.update(hspace=.75)
    gs.update(wspace=.3)
    ax11 = plt.subplot(gs[:3, :3])
    #ax11 = plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=2)
    #ax21 = plt.subplot2grid((4,4),(2,0),colspan=3,rowspan=1,sharex=ax11)

    #ax11 = plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=3)

    # pre-allocate for conversion to simple bifurcation data
    bif_data_simple = np.zeros((len(bif_data[:,0]),6))
    
    # remember write pts from auto gives: par, min, max, type, BR.

    # parameter value
    bif_data_simple[:,0] = bif_data[:,3]
    
    # max value
    # first get relative position of desired state variable
    bif_data_simple[:,1] = bif_data[:,5+idx]
    
    # min value
    bif_data_simple[:,2] = bif_data[:,5+idx+10]
    
    # type
    bif_data_simple[:,3] = bif_data[:,0]

    # branch
    bif_data_simple[:,4] = abs(bif_data[:,1])
    

    data = diagram.read_diagram(bif_data_simple)

    
    for i in range(1,len(data[0,:])):
        x = data[:,0]
        y = data[:,i]
        data[:,0],data[:,i]=clean(x,y,tol=.1)

        if (i == 4):
            print i
            x = data[:,0]
            y = data[:,i]
            data[:,0],data[:,i]=remove_redundant_x(x,y,tol=1e-7)

    # plot unstable fixed points
    #ax11.scatter(data[:,0],data[:,2],s=10,color='black')
    ax11.plot(data[:,0],data[:,2],color='black')

    # plot unstable periodic solutions
    #ax11.scatter(data[:,0],data[:,4],s=10,facecolor='none',edgecolor='blue')
    ax11.plot(data[:,0],data[:,4],color='blue')
    #ax.scatter(data[:,0],data[:,8],s=10,facecolor='none',edgecolor='blue')

    # plot stable fixed points
    #ax11.scatter(data[:,0],data[:,1],s=10,color='red')
    #ax11.scatter(data[:,0],data[:,5],s=10,color='red')
    ax11.plot(data[:,0],data[:,1],color='red')
    ax11.plot(data[:,0],data[:,5],color='red')


    # plot stable periodic solutions
    #ax11.scatter(data[:,0],data[:,3],s=5,color='green')
    #ax11.scatter(data[:,0],data[:,7],s=5,color='green')
    ax11.plot(data[:,0],data[:,3],color='green',lw=3,ls='--')
    ax11.plot(data[:,0],data[:,7],color='green',lw=3,ls='--')


    ax11.set_xlabel('$g$')
    ax11.xaxis.set_label_coords(0.5,-.03)
    ax11.set_ylabel('$sx$')

    ax11.set_xlim(.5,1)
    ax11.set_ylim(.75,1)

    # bifurcation diagram inset
    axins11 = inset_axes(ax11,
                       width="60%", # width = 30% of parent_bbox
                       height="40%", # height : 1 inch
                       loc=3)
    axins11.plot(data[:,0],data[:,4],color='blue')
    axins11.plot(data[:,0],data[:,3],color='green',lw=3,ls='--')
    axins11.plot(data[:,0],data[:,7],color='green',lw=3,ls='--')


    ax11.annotate("TR",
                  xy=(.9095,.79), xycoords='data',
                  xytext=(.9,.82), textcoords='data',
                  size=15,
                  color='red',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black')
                   )


    axins11.annotate("PD",
                     xy=(.5984,.9745), xycoords='data',
                     xytext=(.595,.973), textcoords='data',
                     size=15,
                     color='black',
                     arrowprops=dict(arrowstyle="-|>",
                                     connectionstyle="arc3",
                                     color='black')
                 )

    axins11.annotate("BP",
                     xy=(.59,.9795), xycoords='data',
                     xytext=(.588,.977), textcoords='data',
                     size=15,
                     color='blue',
                     arrowprops=dict(arrowstyle="-|>",
                                     connectionstyle="arc3",
                                     color='black')
                 )


    #axins11.scatter(data[:,0],data[:,4],s=10,facecolor='none',edgecolor='blue')
    #axins11.scatter(data[:,0],data[:,3],s=5,color='green')
    #axins11.scatter(data[:,0],data[:,7],s=5,color='green')
    #axins11.scatter(g,sxsval,color='purple')

    mark_inset(ax11, axins11, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    axins11.set_xlim(.586105,.600328)
    axins11.set_ylim(.972532,.984061)

    # LOOP OVER SAMPLE SOLUTIONS

    rlist = [957,998,1206,655,634,587,8]
    loclist = [(3,0),(3,1),(3,2),(0,3),(1,3),(2,3),(3,3)]
    labellist = [r'$\mathbf{A}$',r'$\mathbf{B}$',r'$\mathbf{C}$',r'$\mathbf{D}$',r'\textbf{E}',r'\textbf{F}',r'\textbf{G}']
    pos = []
    axlist = []

    for i in range(len(rlist)):

        rown = rlist[i]
        g = bif_data[rown,3]

        per = bif_data[rown,5]
        init = init_data[rown,5:]
        dt = .01

        #print g,per,bif_data[rown,6:6+10]
        print init_data[rown,2],init_data[rown,4],init

        npa, vn = xpprun('twodphs3.ode',
                         xppname='xppaut',
                         inits={'x':init[0],'y':init[1],
                                'cxs':init[2],'cys':init[3],
                                'sxs':init[4],'sys':init[5],
                                'sxsys':init[6],'sxcys':init[7],
                                'cxsys':init[8],'cxcys':init[9]},
                         parameters={'total':per,
                                     'g':g,
                                     'q':0.01,
                                     'dt':dt},
                         clean_after=True)

        t = npa[:,0]
        sv = npa[:,1:]

        idx = vn.index('sxs')    
        sxsval = bif_data[rown,6+idx]

        #axlist.append(plt.subplot2grid((4,4),loclist[i]))
        axlist.append(plt.subplot(gs[loclist[i][0],loclist[i][1]]))

        """
        ax41 = plt.subplot2grid((4,4),(3,0))
        ax42 = plt.subplot2grid((4,4),(3,1))
        ax43 = plt.subplot2grid((4,4),(3,2))
        ax14 = plt.subplot2grid((4,4),(0,3))
        ax24 = plt.subplot2grid((4,4),(1,3))
        ax34 = plt.subplot2grid((4,4),(2,3))
        ax44 = plt.subplot2grid((4,4),(3,3))
        """


        ### SAMPLE SOLUTIONS

        xval = np.mod(sv[:,vn.index('x')]+pi,2*pi)-pi
        yval = np.mod(sv[:,vn.index('y')]+pi,2*pi)-pi

        pos1 = np.where(np.abs(np.diff(xval)) >= 1)[0]
        pos2 = np.where(np.abs(np.diff(yval)) >= 1)[0]

        xval[pos1] = np.nan
        yval[pos2] = np.nan

        xval[pos2] = np.nan
        yval[pos2] = np.nan


        dashes = []
        print bif_data[rown,0]
        if abs(bif_data[rown,0]) == 4.:
            dashes = (5,2)

        axlist[i].plot(xval,yval,color='black',lw=2,dashes=dashes)


        # label 2 points with arrows
        back_idx = 5
        idxlist = [int(0.*(per/dt)),int(1.*(per/dt)/2.)]# depends on period

        for j in idxlist:
            axlist[i].annotate("",
                               xy=(xval[j], yval[j]), xycoords='data',
                               xytext=(xval[j-back_idx], yval[j-back_idx]), textcoords='data',
                               size=15,
                               arrowprops=dict(arrowstyle="-|>",
                                               connectionstyle="arc3",
                                               color='black')
            )


        axlist[i].set_xlim(-pi,pi)
        axlist[i].set_ylim(-pi,pi)
        axlist[i].tick_params(axis=u'both',which=u'both',length=0)


        axlist[i].set_xticks(np.arange(-1,1+1,1)*pi)
        axlist[i].set_yticks(np.arange(-1,1+1,1)*pi)
        x_label = [r"$-\pi$", r"$0$", r"$\pi$"]
        #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
        axlist[i].set_xticklabels(x_label)
        axlist[i].set_yticklabels(x_label)

        if i >= 3:
            axlist[i].yaxis.tick_right()
        if i < 3:
            axlist[i].set_yticklabels([])
        if i >= 3 and i < len(labellist)-1:
            axlist[i].set_xticklabels([])



        # annotations corresponding to solution plots
        axins11.annotate(labellist[i],
                         xy=(g, sxsval), xycoords='data',
                         xytext=(g, sxsval), textcoords='data',
                         size=12,
                         verticalalignment='top',
                         horizontalalignment='right',
                         backgroundcolor='yellow',
                         zorder=-1
                      #arrowprops=dict(arrowstyle="-|>",
                      #                connectionstyle="arc3",
                      #                color=str(color)),
                )

        ax11.annotate(labellist[i],
                      xy=(g, sxsval), xycoords='data',
                      xytext=(g, sxsval), textcoords='data',
                      size=12,
                      backgroundcolor='yellow',
                      zorder=-1
                      #arrowprops=dict(arrowstyle="-|>",
                      #                connectionstyle="arc3",
                      #                color=str(color)),
                )


        axlist[i].set_title(labellist[i])

        """
        ax11.annotate("",
                      xy=(th1[i], th2[i]), xycoords='data',
                      xytext=(th1[i-back_idx], th2[i-back_idx]), textcoords='data',
                      size=22,
                      arrowprops=dict(arrowstyle="-|>",
                                      connectionstyle="arc3",
                                      color=str(color)),
                )
        """

        #ax11.
        ax11.scatter(g,sxsval,s=10,color='black',marker='^')
        axins11.scatter(g,sxsval,s=10,color='black',marker='^')

        #ax2.plot(t,np.mod(sv[:,vn.index('x')],2*pi))
        #ax2.plot(t,np.mod(sv[:,vn.index('sxs')],2*pi))

        #ax2.plot(t,np.mod(npa[:,vn.index('sxs')],2*pi))
        #ax2.plot(t,np.mod(npa[:,vn.index('y')],2*pi))



    return fig




def get_and_clean_sol_phase_auto_3terms(init,per,g,dt,nskip=1):
    """
    helper function for 'twod_phase_auto_3terms_fig*'
    
    """

    npa, vn = xpprun('twodphs3.ode',
                     xppname='xppaut',
                     inits={'x':init[0],'y':init[1],
                            'cxs':init[2],'cys':init[3],
                            'sxs':init[4],'sys':init[5],
                            'sxsys':init[6],'sxcys':init[7],
                            'cxsys':init[8],'cxcys':init[9]},
                     parameters={'total':per,
                                 'g':g,
                                 'q':0.1,
                                 'dt':dt},
                     clean_after=True)

    t = npa[:,0]
    sv = npa[:,1:]
    
    idx = vn.index('sxs')    


    ### PLOT SAMPLE SOLUTIONS
    xval = np.mod(sv[:,vn.index('x')]+pi,2*pi)-pi
    yval = np.mod(sv[:,vn.index('y')]+pi,2*pi)-pi

    xval = xval[::nskip]
    yval = yval[::nskip]
    
    pos1 = np.where(np.abs(np.diff(xval)) >= 1)[0]
    pos2 = np.where(np.abs(np.diff(yval)) >= 1)[0]
    
    xval[pos1] = np.nan
    yval[pos2] = np.nan
    
    xval[pos2] = np.nan
    yval[pos2] = np.nan

    
    return xval,yval


def twod_phase_auto_3terms_fig2():
    """
    twod bifurcation diagram for truncated h
    q = 0.1
    """
    #raw_data = np.loadtxt('twodphs_cys_wave_diagram_q=.125.dat')

    # all info data files are organized as follows:
    # Type, BR, 0, par1, par1/2, period, sv1 (high), ..., sv10 (high), sv1 (low),...,sv10(high), real/im eigenvalue pairs...

    # so I could use these values as initial conditions to plot.

    bif_data = np.loadtxt('twodphs_3_HB_PD_q=.1_appended.dat')
    init_data = np.loadtxt('twodphs_3_init_HB_PD_q=.1_appended.dat')

    #bif_data = np.loadtxt('twodphs_3_HB_PD_q=.1_appended.dat')
    #init_data = np.loadtxt('twodphs_3_init_HB_PD_q=.1_appended.dat')

    # manually get index of sxs value
    idx = 5

    fig = plt.figure(figsize=(7,9))

    ### BIFURCATION DIAGRAM
    gs = gridspec.GridSpec(5, 4)
    gs.update(hspace=.4)
    gs.update(wspace=.3)
    ax11 = plt.subplot(gs[:3, :3])
    #ax11 = plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=2)
    #ax21 = plt.subplot2grid((4,4),(2,0),colspan=3,rowspan=1,sharex=ax11)
    ax21 = plt.subplot(gs[3:4,:3],sharex=ax11)

    # pre-allocate for conversion to simple bifurcation data
    bif_data_simple = np.zeros((len(bif_data[:,0]),6))
    
    # remember write pts from auto gives: par, min, max, type, BR.

    # parameter value
    bif_data_simple[:,0] = bif_data[:,3]
    
    # max value
    # first get relative position of desired state variable
    bif_data_simple[:,1] = bif_data[:,5+idx]
    
    # min value
    bif_data_simple[:,2] = bif_data[:,5+idx+10]
    
    # type
    bif_data_simple[:,3] = bif_data[:,0]

    # branch
    bif_data_simple[:,4] = abs(bif_data[:,1])

    data = diagram.read_diagram(bif_data_simple)
    
    for i in range(1,len(data[0,:])):
        x = data[:,0]
        y = data[:,i]
        data[:,0],data[:,i]=clean(x,y,tol=.1)

        if (i == 4):
            print i
            x = data[:,0]
            y = data[:,i]
            data[:,0],data[:,i]=remove_redundant_x(x,y,tol=1e-7)




    # plot unstable periodic solutions
    #ax11.scatter(data[:,0],data[:,4],s=10,facecolor='none',edgecolor='blue')
    #ax.scatter(data[:,0],data[:,8],s=10,facecolor='none',edgecolor='blue')

    ax11.plot(data[:,0],data[:,4],color=blue2,zorder=0)

    # plot stable periodic solutions
    ax11.plot(data[:,0],data[:,3],color='green',lw=3,zorder=0)
    #ax11.scatter(data[:,0],data[:,3],s=5,color='green')
    #ax11.scatter(data[:,0],data[:,7],s=5,color='green')

    # plot stable periodic solutions
    ax21.plot(data[:,0],data[:,3],color='green',lw=3,zorder=0)
    #ax21.scatter(data[:,0],data[:,3],s=5,color='green')
    #ax21.scatter(data[:,0],data[:,7],s=5,color='green')

    # plot unstable fixed points
    #ax21.scatter(data[:,0],data[:,2],s=10,color='black')
    ax21.plot(data[:,0],data[:,2],color='black',zorder=0)

    # plot stable fixed points
    ax21.plot(data[:,0],data[:,1],color='red',lw=3,zorder=0)
    #ax21.scatter(data[:,0],data[:,1],s=10,color='red')
    #ax21.scatter(data[:,0],data[:,5],s=10,color='red')

    #ax21.scatter(data[:,0],data[:,4],s=10,facecolor='none',edgecolor='blue')
    ax21.plot(data[:,0],data[:,4],color=blue2,lw=2,zorder=0)

    """
    # bifurcation diagram inset
    axins11 = inset_axes(ax11,
                       width="60%", # width = 30% of parent_bbox
                       height="40%", # height : 1 inch
                       loc=3)
    #axins11.scatter(data[:,0],data[:,4],s=10,facecolor='none',edgecolor='blue')
    #axins11.scatter(data[:,0],data[:,3],s=5,color='green')
    #axins11.scatter(data[:,0],data[:,7],s=5,color='green')
    #axins11.scatter(g,sxsval,color='purple')


    mark_inset(ax11, axins11, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.xticks(visible=False)
    plt.yticks(visible=False)
    """

    # bifurcation diagram inset
    """
    ax11.add_patch(
        patches.Rectangle(
            (.865, .7),
            (.88-.865),
            (1.05-.7),
            fill=False,
            alpha=.5
        )
    )
    ax11.text(.82,.66,'**',size=20)
    ax21.text(.82,.205,'**',size=20)
    """
    #ax11.plot([.865,.84],[.7,.4],color='gray')
    #ax11.plot([.88,2.1],[1.05,.4],color='gray')


    """
    axins21 = inset_axes(ax21,
                       width="78%",
                       height="78%",
                         loc=1)


    axins21.plot(data[:,0],data[:,4],color=blue2,zorder=0)
    axins21.plot(data[:,0],data[:,3],color='green',lw=3,zorder=0)

    #axins21.text(.866,1.04,'**',size=20)

    
    axins21.set_xlim(.865,.88)
    axins21.set_ylim(.7,1.1)

    axins21.set_xticks([])
    axins21.set_yticks([])

    axins21.spines['bottom'].set_color('gray')
    axins21.spines['top'].set_color('gray') 
    axins21.spines['right'].set_color('gray')
    axins21.spines['left'].set_color('gray')
    """

    #mark_inset(ax21, axins21, loc1=2, loc2=1, fc="none", ec="0.5")

    #plt.xticks(visible=False)
    #plt.yticks(visible=False)




    # label bifurcations

    ax11.annotate("LP1",
                  xy=(.8188,.9727), xycoords='data',
                  xytext=(.8188,1.05), textcoords='data',
                  size=15,
                  color='purple',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black')
                   )


    ax11.annotate("PD1",
                  xy=(1.64,.59), xycoords='data',
                  xytext=(1.2,.55), textcoords='data',
                  size=15,
                  color='black',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black')
                   )


    ax11.annotate("PD2",
                  xy=(1.75,.58), xycoords='data',
                  xytext=(1.5,.51), textcoords='data',
                  size=15,
                  color='black',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black')
                   )

    ax11.annotate("PD3",
                  xy=(1.796,.58), xycoords='data',
                  xytext=(1.7,.5), textcoords='data',
                  size=15,
                  color='black',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black')
                   )


    """
    ax11.annotate("LP*",
                  xy=(1.818,.59), xycoords='data',
                  xytext=(1.9,.5), textcoords='data',
                  size=15,
                  color='purple',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black')
                   )
    """

    ax11.annotate("BP1",
                  xy=(2.,.56), xycoords='data',
                  xytext=(2.1,.56), textcoords='data',
                  size=15,
                  color=blue2,
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black')
                   )

    ax11.annotate("LP2",
                  xy=(2.03,.59), xycoords='data',
                  xytext=(2.1,.65), textcoords='data',
                  size=15,
                  color='purple',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black')
                   )


    ax11.annotate('PD4',
                  xy=(1.738, .65), xycoords='data',
                  xytext=(1.5, .9), textcoords='data',
                  size=12,
                  color='.25',
                  zorder=-1,
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='.25'),
              )



    ax11.annotate('PD5',
                  xy=(1.738, .68), xycoords='data',
                  xytext=(1.7, .85), textcoords='data',
                  size=12,
                  color='.25',
                  zorder=-1,
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='.25'),
                  
              )


    ax21.annotate("HB",
                  xy=(.65,.0), xycoords='data',
                  xytext=(.7,.06), textcoords='data',
                  size=15,
                  color='orange',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black')
                   )

    
    """
    axins21.annotate("TR*",
                     xy=(.8761,.99), xycoords='data',
                     xytext=(.878,1.02), textcoords='data',
                     size=15,
                     color='red',
                     arrowprops=dict(arrowstyle="-|>",
                                     connectionstyle="arc3",
                                     color='black')
                 )
    """

    ax11.annotate("TR2",
                  xy=(1.18,.95), xycoords='data',
                  xytext=(1.4,1), textcoords='data',
                  size=15,
                  color='red',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black')
              )


    #ax11.xaxis.set_ticks_position('top')

    ax11.set_ylim(.48,1.1)
    ax21.set_ylim(-.01,.15)

    ax11.spines['bottom'].set_visible(False)
    ax21.spines['top'].set_visible(False)
    ax11.xaxis.tick_top()
    ax11.tick_params(labeltop='off')
    ax21.xaxis.tick_bottom()

    ax11.set_ylabel(r'$\bm{sx}$',fontsize=15)
    ax11.set_xlim(.51,2.3)

    #ax11.xaxis.set_ticks_position('top')
    #ax21.xaxis.set_ticks_position('bottom')

    ax21.set_xlabel(r'$\bm{g}$',fontsize=15)
    ax21.xaxis.set_label_coords(1.,0)


    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax11.transAxes, color='k', clip_on=False)
    ax11.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax11.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    
    kwargs.update(transform=ax21.transAxes)  # switch to the bottom axes
    ax21.plot((-d, +d), (1 - 4*d, 1 + 4*d), **kwargs)  # bottom-left diagonal
    ax21.plot((1 - d, 1 + d), (1 - 4*d, 1 + 4*d), **kwargs)  # bottom-right diagonal

    # fix scale on ax21
    ax21.set_yticks([0,.1])

    


    # draw patches and solutions where the attractor is "chaotic"
    rect1 = patches.Rectangle((.88,0),.24,2)
    rect2 = patches.Rectangle((1.19,0),.42,2)
    rect3 = patches.Rectangle((2.05,0),2,2)


    collection = PatchCollection([rect1,rect2,rect3],alpha=.4,color='gray',zorder=-1)
    collection2 = PatchCollection([rect1,rect2,rect3],alpha=.4,color='gray',zorder=-1)

    ax11.add_collection(collection)
    ax21.add_collection(collection2)

    #label the patches


    # loop over chaotic attractors
    loclist = [(2,3),(3,3),(4,3)]
    glist = [1.05,1.5,2.15]
    labellist = [r'\textbf{F}',r'\textbf{G}',r'\textbf{H}']
    totlist = [500,500,500]
    cutoff_ratio = [.09,.07,.05]
    labelcoord = [(1,.05),(1.35,.05),(2.1,0.05)]
    dt = .01

    pos = []
    axlist = []

    for i in range(len(loclist)):
        axlist.append(plt.subplot(gs[loclist[i][0],loclist[i][1]]))

        xval,yval = get_and_clean_sol_phase_auto_3terms(np.random.randn(18),totlist[i],glist[i],dt,nskip=1)

        xval = xval[-int(totlist[i]*cutoff_ratio[i]/dt):]
        yval = yval[-int(totlist[i]*cutoff_ratio[i]/dt):]


        lyn = np.linspace(0,1,len(xval))
        axlist[i].add_collection(collect3d_colorgrad(xval,lyn,yval,
                                                     use_nonan=False,
                                                     zorder=2,
                                                     lwstart=2,
                                                     lwend=2,
                                                     cmapmin=cmapmin,
                                                     cmapmax=cmapmax,
                                                     return3d=False,
                                                     cmap=cmap))
        axlist[i].set_title(labellist[i])

        # index to place arrow head
        shift = 40
        idxlist = [shift,int(1.*len(xval)/5)+shift,
                   int(2.*len(xval)/5)+shift,
                   int(3.*len(xval)/5)+shift,
                   int(4.*len(xval)/5)+shift]# depends on period
        factor = 1.
        back_idx = 1


        for j in idxlist:
            axlist[i].annotate("",
                               xy=(factor*xval[j], factor*yval[j]), xycoords='data',
                               xytext=(factor*xval[j-back_idx], factor*yval[j-back_idx]), textcoords='data',
                               size=15,
                               arrowprops=dict(arrowstyle="-|>",
                                               connectionstyle="arc3",
                                               color='black')
            )

        # label region
        ax21.annotate(labellist[i],
                      xy=labelcoord[i], xycoords='data',
                      xytext=labelcoord[i], textcoords='data',
                      size=12,
                      bbox=dict(boxstyle="round4,pad=.2", fc=(1.0, 0.7, 0.7)),
                      #backgroundcolor=labelbg,
                      zorder=-1
                  )


        # place label in bifurcation diagram
        #ax11.annotate(labellist[i],xy=(glist[i],.8),xytext=(glist[i],.8),size=12)

        axlist[i].set_xlim(-pi,pi)
        axlist[i].set_ylim(-pi,pi)
        axlist[i].tick_params(axis=u'both',which=u'both',length=0)



        
        if i < len(labellist)-1:
            print i, 'less than xlist'
            axlist[i].set_xticks([])
            axlist[i].set_xticklabels([])
            axlist[i].set_yticks(np.arange(-1,1+1,1)*pi)
            axlist[i].set_yticklabels(x_label_short)
        else:
            print i
            axlist[i].set_xticks(np.arange(-1,1+1,1)*pi)
            axlist[i].set_yticks(np.arange(-1,1+1,1)*pi)
            #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
            axlist[i].set_xticklabels(x_label_short)
            axlist[i].set_yticklabels(x_label_short)


        if i == 1:
            #axlist[i].set_ylim(-.5,.5)
            #axlist[i].set_yticks(np.arange(-.5,.5+.5,.5))
            #axlist[i].set_yticklabels(np.arange(-.5,.5+.5,.5))
            #axlist[i].set_yticks()
            pass

        axlist[i].yaxis.tick_right()



    # LOOP OVER SAMPLE SOLUTIONS

    rlist = [519,1372,886,1654,2349]#[957,998,1206,655,634,587,8]
    loclist = [(4,0),(4,1),(4,2),(0,3),(1,3),(2,3),(3,3),(4,3)]
    labellist = [r'$\mathbf{A}$',r'$\mathbf{B}$',r'$\mathbf{C}$',r'$\mathbf{D}$',r'\textbf{E}']
    pos = []
    axlist = []

    for i in range(len(rlist)):

        rown = rlist[i]
        g = bif_data[rown,3]

        per = bif_data[rown,5]
        init = init_data[rown,5:]




        #print g,per,bif_data[rown,6:6+10]
        print 'g=',g,'g=',init_data[rown,2],'per=',init_data[rown,4],'init=',init

        #axlist.append(plt.subplot2grid((4,4),loclist[i]))
        axlist.append(plt.subplot(gs[loclist[i][0],loclist[i][1]]))
        sxsval = bif_data[rown,5+idx]
        xval,yval = get_and_clean_sol_phase_auto_3terms(init,per,g,dt)


        #for slc in unlink_wrap(xval):
        #    axlist[i].plot(xval[slc],yval[slc],color='black',lw=2)
        dashes = []
        print bif_data[rown,0]
        if abs(bif_data[rown,0]) == 4.:
            dashes = (5,3)

        #axlist[i].plot(xval,yval,color='black',lw=2,dashes=dashes)
        lyn = np.linspace(0,1,len(xval))
        axlist[i].add_collection(collect3d_colorgrad(xval,lyn,yval,
                                                     use_nonan=False,
                                                     zorder=2,
                                                     lwstart=2,
                                                     lwend=2,
                                                     cmapmin=cmapmin,
                                                     cmapmax=cmapmax,
                                                     return3d=False,
                                                     cmap=cmap))

        # label 2 points with arrows
        back_idx = 10
        idxlist = [10,int(1.*(per/dt)/2.)]# depends on period

        if i == 0:
            factor = 1.3
            back_idx = 400
        else:
            factor = 1.
        
        for j in idxlist:
            axlist[i].annotate("",
                               xy=(factor*xval[j], factor*yval[j]), xycoords='data',
                               xytext=(factor*xval[j-back_idx], factor*yval[j-back_idx]), textcoords='data',
                               size=15,
                               arrowprops=dict(arrowstyle="-|>",
                                               connectionstyle="arc3",
                                               color='black')
            )

        axlist[i].set_xlim(-pi,pi)
        axlist[i].set_ylim(-pi,pi)
        axlist[i].tick_params(axis=u'both',which=u'both',length=0)

        axlist[i].set_xticks(np.arange(-1,1+1,1)*pi)
        axlist[i].set_yticks(np.arange(-1,1+1,1)*pi)

        #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
        axlist[i].set_xticklabels(x_label_short)
        axlist[i].set_yticklabels(x_label_short)

        if i >= 3:
            axlist[i].yaxis.tick_right()
        if i < 3:
            axlist[i].set_yticklabels([])
        if i >= 3:
            axlist[i].set_xticklabels([])

        # annotations corresponding to solution plots

        """
        ax21.annotate(labellist[i],
                      xy=(g, sxsval), xycoords='data',
                      xytext=(g, sxsval), textcoords='data',
                      size=12,
                      backgroundcolor='yellow',
                      zorder=-1
                      #arrowprops=dict(arrowstyle="-|>",
                      #                connectionstyle="arc3",
                      #                color=str(color)),
                )
        """
        
        ax11.annotate(labellist[i],
                      xy=(g, sxsval), xycoords='data',
                      xytext=(g+.01, sxsval+.01), textcoords='data',
                      size=12,
                      #backgroundcolor=labelbg,
                      bbox=dict(boxstyle="round4,pad=.2", fc=(1.0, 0.7, 0.7)),
                      zorder=-1
                      #arrowprops=dict(arrowstyle="-|>",
                      #                connectionstyle="arc3",
                      #                color=str(color)),
                )

        """
        axins21.annotate(labellist[i],
                         xy=(g, sxsval), xycoords='data',
                         xytext=(g, sxsval), textcoords='data',
                         size=12,
                         backgroundcolor='yellow',
                         zorder=-1,
                         horizontalalignment='right',
                         #arrowprops=dict(arrowstyle="-|>",
                         #                connectionstyle="arc3",
                         #                color=str(color)),
                )
        """


        axlist[i].set_title(labellist[i])

        #ax11.
        ax11.scatter(g,sxsval,s=50,color='black',marker='*',zorder=2)
        ax21.scatter(g,sxsval,s=50,color='black',marker='*',zorder=2)

        #axins21.scatter(g,sxsval,s=10,color='black',marker='^',zorder=2)


        #ax2.plot(t,np.mod(sv[:,vn.index('x')],2*pi))
        #ax2.plot(t,np.mod(sv[:,vn.index('sxs')],2*pi))

        #ax2.plot(t,np.mod(npa[:,vn.index('sxs')],2*pi))
        #ax2.plot(t,np.mod(npa[:,vn.index('y')],2*pi))



    
    ax11.xaxis.set_major_formatter(formatter)
    ax11.yaxis.set_major_formatter(formatter)

    return fig


def twod_phase_auto_5terms_fig():
    """
    twod bifurcation diagram for truncated h
    """
    #raw_data = np.loadtxt('twodphs_cys_wave_diagram_q=.125.dat')

    sv = 'cys'
    qval = '.5'

    raw_data = np.loadtxt('twodphs_5_'+sv+'_HB_q='+qval+'.dat')
    raw_data2 = np.loadtxt('twodphs_5_'+sv+'_TR_q='+qval+'.dat')

    
    data = diagram.read_diagram(raw_data)
    data2 = diagram.read_diagram(raw_data2)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot unstable fixed points
    ax.scatter(data[:,0],data[:,2],s=10,color='black')
    ax.scatter(data[:,0],data[:,6],s=10,color='black')

    #ax.scatter(data3[:,0],data3[:,2],s=10,color='black')
    #ax.scatter(data3[:,0],data3[:,6],s=10,color='black')


    # plot unstable periodic solutions
    ax.scatter(data[:,0],data[:,4],s=10,facecolor='none',edgecolor='blue')
    #ax.scatter(data[:,0],data[:,8],s=10,facecolor='none',edgecolor='blue')

    # plot traveling waves
    ax.scatter(data2[:,0],data2[:,4],s=10,facecolor='none',edgecolor='#0099ff')
    ax.scatter(data2[:,0],data2[:,8],s=10,facecolor='none',edgecolor='#0099ff')

    #ax.scatter(data3[:,0],data3[:,4],s=10,facecolor='none',edgecolor='blue')
    #ax.scatter(data3[:,0],data3[:,8],s=10,facecolor='none',edgecolor='blue')

    #ax.scatter(data4[:,0],data4[:,4],s=10,facecolor='none',edgecolor='#0099ff')
    #ax.scatter(data4[:,0],data4[:,8],s=10,facecolor='none',edgecolor='#0099ff')


    # plot stable fixed points
    ax.scatter(data[:,0],data[:,1],s=10,color='red')
    ax.scatter(data[:,0],data[:,5],s=10,color='red')

    #ax.scatter(data3[:,0],data3[:,1],s=10,color='red')
    #ax.scatter(data3[:,0],data3[:,5],s=10,color='red')

    # plot stable periodic solutions
    ax.scatter(data[:,0],data[:,3],s=20,color='green')
    ax.scatter(data[:,0],data[:,7],s=20,color='green')


    # plot stable traveling waves
    ax.scatter(data2[:,0],data2[:,3],s=20,color='#00cc00')
    ax.scatter(data2[:,0],data2[:,7],s=20,color='#00cc00')

    #ax.scatter(data3[:,0],data3[:,3],s=20,color='green')
    #ax.scatter(data3[:,0],data3[:,7],s=20,color='green')

    #ax.scatter(data4[:,0],data4[:,3],s=20,color='#00cc00')
    #ax.scatter(data4[:,0],data4[:,7],s=20,color='#00cc00')

    #ax.set_xlim(.5,3)
    #ax.set_ylim(.5,1.)


    ax.set_title('q='+qval)
    ax.set_ylabel(sv)


    return fig

def twod_phase_auto_3terms_2par():
    """
    twod, 2par bifurcation diagram
    """
    data = np.loadtxt('twodphs_3_2par.dat')
    data2 = np.loadtxt('twodphs_3_2par_TR.dat')

    # separate by branches
    # 6 = PD, 2 = LP, 3 = HB
    
    TR = data2[(data2[:,-1]==4)]
    PD = data[(data[:,-1]==6)*(data[:,-2]<9)]
    PD_gray = data[(data[:,-1]==6)*(data[:,-2]>=9)*(data[:,-2]<=11)]
    LP = data[data[:,-1]==2]
    HB = data[data[:,-1]==3]
    BP = data[data[:,-1]==5]

    # remove discontinuities
    TRx,TRy = clean(TR[:,0],TR[:,1],tol=.05)
    PDx,PDy = clean(PD[:,0],PD[:,1],tol=.05)
    PD_gx,PD_gy = clean(PD_gray[:,0],PD_gray[:,1],tol=.05)
    LPx,LPy = clean(LP[:,0],LP[:,1],tol=.05)
    HBx,HBy = clean(HB[:,0],HB[:,1],tol=.05)
    BPx,BPy = clean(BP[:,0],BP[:,1],tol=.05)
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)

    ax.plot([0,2],[.1,.1],color='gray')

    ax.plot(PD_gx,PD_gy,color='.35',lw=2,ls='--',dashes=(4,1))
    ax.plot(TRx,TRy,color='red',lw=2)
    ax.plot(PDx,PDy,color='black',lw=2)
    ax.plot(HBx,HBy,color='orange',ls='--',lw=2)
    ax.plot(LPx,LPy,color='purple',ls='-.',lw=2)
    ax.plot(BPx,BPy,color='blue',ls='',marker='1',ms=10,lw=2)
    
    ax.set_xlabel(r'$\bm{g}$')
    ax.set_ylabel(r'$\bm{q}$')


    # label regions
    
    ax.annotate('1. Stationary Bump',
                xy=(.5, .55), xycoords='data',
                xytext=(.5, .55), textcoords='data',
                size=12,
                zorder=-1
              )


    ax.annotate('2. Sloshing Bump',
                xy=(1.2, .35), xycoords='data',
                xytext=(1.2, .35), textcoords='data',
                size=12,
                zorder=-1
              )

    ax.annotate('3. Large-Slosh and Chaos',
                xy=(1.1, .15), xycoords='data',
                xytext=(1.1, .27), textcoords='data',
                size=12,
                zorder=-1,
                rotation=30
              )

    ax.annotate('4. Chaos',
                xy=(1.1, .15), xycoords='data',
                xytext=(1.3, .11), textcoords='data',
                size=12,
                zorder=2,
                rotation=22
              )


    # label branches
    ax.annotate('LP1',
                xy=(1.9, .35), xycoords='data',
                xytext=(1.9, .35), textcoords='data',
                size=12,
                color='purple',
                zorder=-1
              )

    ax.annotate('HB',
                xy=(1.15, .57), xycoords='data',
                xytext=(1.15, .57), textcoords='data',
                size=12,
                color='orange',
                zorder=-1
              )


    ax.annotate('PD1',
                xy=(1.616, .1), xycoords='data',
                xytext=(1.5, .13), textcoords='data',
                size=12,
                color='black',
                zorder=-1,
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3",
                                color='black'),

              )


    ax.annotate('PD2',
                xy=(1.728, .1), xycoords='data',
                xytext=(1.7, .15), textcoords='data',
                size=12,
                color='black',
                zorder=-1,
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3",
                                color='black'),

              )

    ax.annotate('PD3',
                xy=(1.81, .1), xycoords='data',
                xytext=(1.85, .18), textcoords='data',
                size=12,
                color='black',
                zorder=3,
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3",
                                color='black'),

              )

    ax.annotate('LP2',
                xy=(1.8, .06), xycoords='data',
                xytext=(1.8, .06), textcoords='data',
                size=12,
                color='purple',
                zorder=-1
              )

    ax.annotate('BP1',
                xy=(1., .06), xycoords='data',
                xytext=(1., .06), textcoords='data',
                size=12,
                color='blue',
                zorder=-1
              )


    ax.annotate('PD4,5',
                xy=(1.75, .1), xycoords='data',
                xytext=(1.6, .02), textcoords='data',
                size=12,
                color='.25',
                zorder=-1,
                arrowprops=dict(arrowstyle="wedge,tail_width=.7",
                                connectionstyle="arc3",
                                color='.25'),

              )

    ax.annotate('TR2',
                xy=(1.2, .125), xycoords='data',
                xytext=(1.2, .125), textcoords='data',
                size=12,
                color='red',
                zorder=-1
              )


    
    ax.set_xlim(.48,2.)
    ax.set_ylim(0,.6)

    return fig
    


def twod_full_auto_5terms_fig_old():
    """
    twod bifurcation diagram for truncated h
    q = 0.1
    """
    #raw_data = np.loadtxt('twodphs_cys_wave_diagram_q=.125.dat')

    # all info data files are organized as follows:
    # Type, BR, 0, par1, par1/2, period, sv1 (high), ..., sv10 (high), sv1 (low),...,sv10(high), real/im eigenvalue pairs...

    # so I could use these values as initial conditions to plot.

    #bif_data = np.loadtxt('twodphs_3_HB_PD_q=.1.dat')
    #init_data = np.loadtxt('twodphs_3_init_HB_PD_q=.1.dat')

    
    bif_data = np.loadtxt('full2dbranches1info.dat')
    init_data = np.loadtxt('full2dbranches1info_inits.dat')

    bif_data4 = np.loadtxt('full2dbranches2info.dat')
    init_data4 = np.loadtxt('full2dbranches2_inits.dat')

    bif_data3 = np.loadtxt('full2dbranches3info.dat')
    init_data3 = np.loadtxt('full2dbranches3_inits.dat')

    #bif_data2 = np.loadtxt('full2dbranches4info.dat')
    #init_data2 = np.loadtxt('full2dbranches4_inits.dat')

    bif_data2 = np.loadtxt('full2dbranches_refined.dat')
    init_data2 = np.loadtxt('full2dbranches_refined_inits.dat')


    # small g osc
    bif_data5 = np.loadtxt('full2dbranches5_smallg_osc.dat')
    init_data5 = np.loadtxt('full2dbranches5_smallg_osc_inits.dat')


    # manually get index of a11 value
    idx = 6

    fig = plt.figure(figsize=(7,7))

    ### BIFURCATION DIAGRAM
    gs = gridspec.GridSpec(4, 4)
    gs.update(hspace=.4)
    gs.update(wspace=.3)
    ax11 = plt.subplot(gs[:3, :3])
    #ax11 = plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=2)
    #ax21 = plt.subplot2grid((4,4),(2,0),colspan=3,rowspan=1,sharex=ax11)
    #ax21 = plt.subplot(gs[2,:3],sharex=ax11)

    # pre-allocate for conversion to simple bifurcation data
    bif_data_simple = np.zeros((len(bif_data[:,0]),6))
    bif_data_simple2 = np.zeros((len(bif_data2[:,0]),6))
    bif_data_simple3 = np.zeros((len(bif_data3[:,0]),6))
    bif_data_simple4 = np.zeros((len(bif_data4[:,0]),6))
    bif_data_simple5 = np.zeros((len(bif_data5[:,0]),6))
    
    # remember write pts from auto gives: par, min, max, type, BR.

    # parameter value
    bif_data_simple[:,0] = bif_data[:,3]
    bif_data_simple2[:,0] = bif_data2[:,3]
    bif_data_simple3[:,0] = bif_data3[:,3]
    bif_data_simple4[:,0] = bif_data4[:,3]
    bif_data_simple5[:,0] = bif_data5[:,3]
    
    # max value
    # first get relative position of desired state variable
    bif_data_simple[:,1] = bif_data[:,5+idx]
    bif_data_simple2[:,1] = bif_data2[:,5+idx]
    bif_data_simple3[:,1] = bif_data3[:,5+idx]
    bif_data_simple4[:,1] = bif_data4[:,5+idx]
    bif_data_simple5[:,1] = bif_data5[:,5+idx]
    
    # min value
    bif_data_simple[:,2] = bif_data[:,5+idx+18]
    bif_data_simple2[:,2] = bif_data2[:,5+idx+18]
    bif_data_simple3[:,2] = bif_data3[:,5+idx+18]
    bif_data_simple4[:,2] = bif_data4[:,5+idx+18]
    bif_data_simple5[:,2] = bif_data5[:,5+idx+18]
    
    # type
    bif_data_simple[:,3] = bif_data[:,0]
    bif_data_simple2[:,3] = bif_data2[:,0]
    bif_data_simple3[:,3] = bif_data3[:,0]
    bif_data_simple4[:,3] = bif_data4[:,0]
    bif_data_simple5[:,3] = bif_data5[:,0]

    # branch
    bif_data_simple[:,4] = abs(bif_data[:,1])
    bif_data_simple2[:,4] = abs(bif_data2[:,1])
    bif_data_simple3[:,4] = abs(bif_data3[:,1])
    bif_data_simple4[:,4] = abs(bif_data4[:,1])
    bif_data_simple5[:,4] = abs(bif_data5[:,1])


    for i in range(1,len(bif_data_simple[0,:])):
        x = bif_data_simple[:,0]
        y = bif_data_simple[:,i]
        bif_data_simple[:,0],bif_data_simple[:,i]=clean(x,y,tol=.1)
        
        x = bif_data_simple[:,0]
        y = bif_data_simple[:,i]
        bif_data_simple[:,0],bif_data_simple[:,i]=remove_redundant_x(x,y,tol=1e-7)


    data = diagram.read_diagram(bif_data_simple)
    data2 = diagram.read_diagram(bif_data_simple2)
    data3 = diagram.read_diagram(bif_data_simple3)
    data4 = diagram.read_diagram(bif_data_simple4)
    data5 = diagram.read_diagram(bif_data_simple5)

    # plot unstable fixed points
    ax11.plot(data[:,0],data[:,2],color='black')
    ax11.plot(data[:,0],data[:,6],color='black')

    ax11.scatter(data2[:,0],data2[:,2],s=1,color='black')
    ax11.scatter(data2[:,0],data2[:,6],s=1,color='black')

    #ax11.scatter(data3[:,0],data3[:,2],s=1,color='black')
    #ax11.scatter(data3[:,0],data3[:,6],s=1,color='black')

    ax11.scatter(data4[:,0],data4[:,2],s=1,color='black')
    ax11.scatter(data4[:,0],data4[:,6],s=1,color='black')
 
    
    # plot unstable periodic solutions
    ax11.scatter(data[:,0],data[:,4],s=10,facecolor='none',edgecolor='blue')
    ax11.scatter(data2[:,0],data2[:,4],s=10,facecolor='none',edgecolor='blue')
    #ax11.scatter(data3[:,0],data3[:,4],s=10,facecolor='none',edgecolor='blue')
    ax11.scatter(data4[:,0],data4[:,4],s=10,facecolor='none',edgecolor='blue')
    ax11.scatter(data5[:,0],data5[:,4],s=10,facecolor='none',edgecolor='blue')
    #ax.scatter(data[:,0],data[:,8],s=10,facecolor='none',edgecolor='blue')

    # plot stable periodic solutions
    ax11.scatter(data[:,0],data[:,3],s=5,color='green')
    ax11.scatter(data[:,0],data[:,7],s=5,color='green')

    ax11.scatter(data2[:,0],data2[:,3],s=5,color='green')
    ax11.scatter(data2[:,0],data2[:,7],s=5,color='green')

    #ax11.scatter(data3[:,0],data3[:,3],s=5,color='green')
    #ax11.scatter(data3[:,0],data3[:,7],s=5,color='green')

    ax11.scatter(data4[:,0],data4[:,3],s=5,color='green')
    ax11.scatter(data4[:,0],data4[:,7],s=5,color='green')


    ax11.scatter(data5[:,0],data5[:,3],s=5,color='green')
    ax11.scatter(data5[:,0],data5[:,7],s=5,color='green')


    # plot stable fixed points
    ax11.scatter(data[:,0],data[:,1],s=1,color='red')
    ax11.scatter(data[:,0],data[:,5],s=1,color='red')

    ax11.scatter(data2[:,0],data2[:,1],s=1,color='red')
    ax11.scatter(data2[:,0],data2[:,5],s=1,color='red')

    #ax11.scatter(data3[:,0],data3[:,1],s=1,color='red')
    #ax11.scatter(data3[:,0],data3[:,5],s=1,color='red')

    ax11.scatter(data4[:,0],data4[:,1],s=1,color='red')
    ax11.scatter(data4[:,0],data4[:,5],s=1,color='red')

    ax11.set_ylim(2.,4.)
    #ax21.set_ylim(-.01,.22)

    ax11.set_ylabel('$a_{11}$')
    #ax11.set_xlim(0,.3)

    #ax11.xaxis.set_ticks_position('top')
    #ax21.xaxis.set_ticks_position('bottom')

    ax11.set_xlabel('$g$')
    #ax21.xaxis.set_label_coords(1.,0)

    # LOOP OVER SAMPLE SOLUTIONS

    rlist = [64,234,23,4]
    loclist = [(3,0),(3,1),(3,2),(0,3),(1,3),(2,3),(3,3)]
    labellist = [r'$\mathbf{A}$',r'$\mathbf{B}$',r'$\mathbf{C}$',r'$\mathbf{D}$',r'\textbf{E}',r'\textbf{F}',r'\textbf{G}']
    pos = []
    axlist = []

    for i in range(len(rlist)):

        rown = rlist[i]

        if i == 1:
            g = bif_data[rown,3]
            per = bif_data[rown,5]
            init = init_data[rown,5:]
            print 'g=',g,'g=',init_data[rown,2],'per=',init_data[rown,4],'init=',init
        else:
            g = bif_data2[rown,3]
            per = bif_data2[rown,5]
            init = init_data2[rown,5:]
            print 'g=',g,'g=',init_data2[rown,2],'per=',init_data2[rown,4],'init=',init
        dt = .1

        #print g,per,bif_data[rown,6:6+10]


        npa, vn = xpprun('full2dbump.ode',
                         xppname='xppaut',
                         inits={'a0':init[0],'a10':init[1],
                                'a01':init[2],'b10':init[3],
                                'b01':init[4],'a11':init[5],
                                'b11':init[6],'c11':init[7],
                                'd11':init[8],
                                'e0':init[9],
                                'e10':init[10],'e01':init[11],
                                'e11':init[12],'f01':init[13],
                                'f10':init[14],'f11':init[15],
                                'g11':init[16],'h11':init[17]
                            },

                         parameters={'total':per,
                                     'g':g,
                                     'w':0.2,
                                     'tau':20.,
                                     'dt':dt},
                         clean_after=True)

        t = npa[:,0]
        sv = npa[:,1:]


        idx = vn.index('b11')
        #print idx, vn

        if i == 1:
            sxsval = bif_data[rown,5+idx]
        else:
            sxsval = bif_data2[rown,5+idx]

        #axlist.append(plt.subplot2grid((4,4),loclist[i]))
        axlist.append(plt.subplot(gs[loclist[i][0],loclist[i][1]]))

        ### PLOT SAMPLE SOLUTIONS

        b01 = sv[:,vn.index('b01')]
        a01 = sv[:,vn.index('a01')]
        b10 = sv[:,vn.index('b10')]
        a10 = sv[:,vn.index('a10')]

        xval = np.mod(np.arctan2(b01,a01)+5*pi,2*pi)-pi
        yval = np.mod(np.arctan2(b10,a10)+5*pi,2*pi)-pi

        pos1 = np.where(np.abs(np.diff(xval)) >= 1)[0]
        pos2 = np.where(np.abs(np.diff(yval)) >= 1)[0]

        xval[pos1] = np.nan
        yval[pos1] = np.nan

        xval[pos2] = np.nan
        yval[pos2] = np.nan

        #for slc in unlink_wrap(xval):
        #    axlist[i].plot(xval[slc],yval[slc],color='black',lw=2)
        dashes = []

        if i == 1:
            print bif_data[rown,0]
            if abs(bif_data[rown,0]) == 4.:
                dashes = (5,2)

        else:
            print bif_data2[rown,0]
            if abs(bif_data2[rown,0]) == 4.:
                dashes = (5,2)

        axlist[i].plot(xval,yval,color='black',lw=2,dashes=dashes)

        # label 2 points with arrows
        back_idx = 10
        idxlist = [10,int(1.*(per/dt)/2.)]# depends on period

        if i == 0:
            factor = 1.3
            back_idx = 400
        else:
            factor = 1.
        
        for j in idxlist:
            axlist[i].annotate("",
                               xy=(factor*xval[j], factor*yval[j]), xycoords='data',
                               xytext=(factor*xval[j-back_idx], factor*yval[j-back_idx]), textcoords='data',
                               size=15,
                               arrowprops=dict(arrowstyle="-|>",
                                               connectionstyle="arc3",
                                               color='black')
            )

        axlist[i].set_xlim(-pi,pi)
        axlist[i].set_ylim(-pi,pi)
        axlist[i].tick_params(axis=u'both',which=u'both',length=0)

        axlist[i].set_xticks(np.arange(-1,1+1,1)*pi)
        axlist[i].set_yticks(np.arange(-1,1+1,1)*pi)
        x_label = [r"$-\pi$", r"$0$", r"$\pi$"]
        #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
        axlist[i].set_xticklabels(x_label)
        axlist[i].set_yticklabels(x_label)

        if i >= 3:
            axlist[i].yaxis.tick_right()
        if i < 3:
            axlist[i].set_yticklabels([])
        if i >= 3 and i < len(labellist)-1:
            axlist[i].set_xticklabels([])

        # annotations corresponding to solution plots

        ax11.annotate(labellist[i],
                      xy=(g, sxsval), xycoords='data',
                      xytext=(g, sxsval), textcoords='data',
                      size=12,
                      backgroundcolor='yellow',
                      zorder=-1
                      #arrowprops=dict(arrowstyle="-|>",
                      #                connectionstyle="arc3",
                      #                color=str(color)),
                )


        axlist[i].set_title(labellist[i])

        #ax11.
        ax11.scatter(g,sxsval,s=10,color='black',marker='^')
        #axins11.scatter(g,sxsval,s=10,color='black',marker='^')

        #ax2.plot(t,np.mod(sv[:,vn.index('x')],2*pi))
        #ax2.plot(t,np.mod(sv[:,vn.index('sxs')],2*pi))

        #ax2.plot(t,np.mod(npa[:,vn.index('sxs')],2*pi))
        #ax2.plot(t,np.mod(npa[:,vn.index('y')],2*pi))


    return fig


def get_and_clean_sol_full_auto_5terms(init,per,g,dt):
    """
    helper function for 'twod_full_auto_5terms_fig'

    
    """
    npa, vn = xpprun('full2dbump.ode',
                     xppname='xppaut',
                     inits={'a0':init[0],'a10':init[1],
                            'a01':init[2],'a11':init[3],
                            'b10':init[4],'b01':init[5],
                            'b11':init[6],'c11':init[7],'d11':init[8],
                            'e0':init[9],'e10':init[10],
                            'e01':init[11],'e11':init[12],
                            'f10':init[13],'f01':init[14],
                            'f11':init[15],'g11':init[16],'h11':init[17]
                        },
                     parameters={'total':per,
                                 'g':g,
                                 'q':0.1,
                                 'eps':.05,
                                 'dt':dt},
                     clean_after=True)

    t = npa[:,0]
    sv = npa[:,1:]
    
    idx = vn.index('a11')
    #print idx, vn
    

    
    ### PLOT SAMPLE SOLUTIONS
    
    b01 = sv[:,vn.index('b01')]
    a01 = sv[:,vn.index('a01')]
    b10 = sv[:,vn.index('b10')]
    a10 = sv[:,vn.index('a10')]
    
    xval = np.mod(np.arctan2(b01,a01)+5*pi,2*pi)-pi
    yval = np.mod(np.arctan2(b10,a10)+5*pi,2*pi)-pi
    
    pos1 = np.where(np.abs(np.diff(xval)) >= 1)[0]
    pos2 = np.where(np.abs(np.diff(yval)) >= 1)[0]
    
    xval[pos1] = np.nan
    yval[pos1] = np.nan
    
    xval[pos2] = np.nan
    yval[pos2] = np.nan

    return xval,yval


def twod_full_auto_5terms_fig():
    """
    twod bifurcation diagram for truncated h
    q = 0.1
    """
    #raw_data = np.loadtxt('twodphs_cys_wave_diagram_q=.125.dat')

    # all info data files are organized as follows:
    # Type, BR, 0, par1, par1/2, period, sv1 (high), ..., sv10 (high), sv1 (low),...,sv10(high), real/im eigenvalue pairs...
    
    bif_data = np.loadtxt('full2dbump.ode.diagram.q=.1.dat')
    init_data = np.loadtxt('full2dbump.ode.diagram.q=.1.inits.dat')

    # manually get index of a11 value
    idx = 3

    fig = plt.figure(figsize=(7,6))

    ### BIFURCATION DIAGRAM
    gs = gridspec.GridSpec(4, 5)
    gs.update(hspace=.5)
    gs.update(wspace=.3)
    ax11 = plt.subplot(gs[:3, :4])
    #ax11 = plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=2)
    #ax21 = plt.subplot2grid((4,4),(2,0),colspan=3,rowspan=1,sharex=ax11)
    #ax21 = plt.subplot(gs[2,:3],sharex=ax11)

    # pre-allocate for conversion to simple bifurcation data
    bif_data_simple = np.zeros((len(bif_data[:,0]),6))
    
    # remember write pts from auto gives: par, min, max, type, BR.
    
    # parameter value
    bif_data_simple[:,0] = bif_data[:,3]
    
    # max value
    # first get relative position of desired state variable
    bif_data_simple[:,1] = bif_data[:,6+idx]
    
    # min value
    bif_data_simple[:,2] = bif_data[:,6+idx+18]
    
    # type
    bif_data_simple[:,3] = bif_data[:,0]
    
    # branch
    bif_data_simple[:,4] = abs(bif_data[:,1])
    

    for i in range(1,len(bif_data_simple[0,:])):
        x = bif_data_simple[:,0]
        y = bif_data_simple[:,i]
        bif_data_simple[:,0],bif_data_simple[:,i]=clean(x,y,tol=.5)
        
        #x = bif_data_simple[:,0]
        #y = bif_data_simple[:,i]
        #bif_data_simple[:,0],bif_data_simple[:,i]=remove_redundant_x(x,y,tol=1e-10)


    data = diagram.read_diagram(bif_data_simple)

    # plot unstable fixed points
    ax11.plot(data[:,0],data[:,2],color='black')
    ax11.plot(data[:,0],data[:,6],color='black') 
    
    # plot unstable periodic solutions
    #ax11.scatter(data[:,0],data[:,4],s=10,facecolor='none',edgecolor='blue')
    ax11.plot(data[:,0],data[:,4],color='#0066B2')

    # plot stable periodic solutions
    #ax11.scatter(data[:,0],data[:,3],s=5,color='green')
    #ax11.scatter(data[:,0],data[:,7],s=5,color='green')
    oscy = data[:,3][data[:,3]<1]
    oscx = data[:,0][data[:,3]<1]
    oscx,oscy = clean(oscx,oscy,tol=.1)
    ax11.plot(oscx,oscy,color='green',lw=3)

    wavey = data[:,3][data[:,3]>1]
    wavex = data[:,0][data[:,3]>1]

    ax11.plot(wavex,wavey,color='green',ls='--',lw=3)


    #ax11.plot(data[:,0],data[:,7],color='green',lw=2)

    # plot stable fixed points
    #ax11.scatter(data[:,0],data[:,1],s=1,color='red')
    #ax11.scatter(data[:,0],data[:,5],s=1,color='red')

    ax11.plot(data[:,0],data[:,1],color='red')
    ax11.plot(data[:,0],data[:,5],color='red')

    ax11.annotate("HB",
                  xy=(1.34,1.28), xycoords='data',
                  xytext=(1.2,1.4), textcoords='data',
                  size=15,
                  color='orange',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black')
                   )


    #ax11.set_ylim(2.,4.)
    #ax21.set_ylim(-.01,.22)

    ax11.set_ylabel(r'$\bm{a_{11}}$',fontsize=15)
    ax11.set_xlim(1.,2.5)
    ax11.set_ylim(0.,1.5)

    ax11.set_xlabel(r'$\bm{g}$',fontsize=15)
    ax11.xaxis.set_label_coords(1.,0)
    #ax21.xaxis.set_label_coords(1.,0)


    # draw patches and solutions where the attractor is "chaotic"
    rect1 = patches.Rectangle((1.3,0),.2,1.5)
    rect2 = patches.Rectangle((1.55,0),.4,1.5)
    rect3 = patches.Rectangle((2.1,0),.4,1.5)

    collection = PatchCollection([rect1,rect2,rect3],alpha=.4,color='gray',zorder=-1)

    ax11.add_collection(collection)

    # loop over chaotic attractors
    loclist = [(0,4),(1,4),(2,4)]
    glist = [1.4,1.7,2.4]
    labellist = [r'\textbf{E}',r'\textbf{F}',r'\textbf{G}']
    totlist = [5000,8000,8000]
    cutoff_ratio = [.3,.1,.06]
    dt = 1.

    pos = []
    axlist = []

    
    for i in range(len(loclist)):

        axlist.append(plt.subplot(gs[loclist[i][0],loclist[i][1]]))
        xval,yval = get_and_clean_sol_full_auto_5terms(np.random.randn(18),totlist[i],glist[i],dt)

        xval = xval[-int(totlist[i]*cutoff_ratio[i]/dt):]
        yval = yval[-int(totlist[i]*cutoff_ratio[i]/dt):]

        lyn = np.linspace(0,1,len(xval))
        axlist[i].add_collection(collect3d_colorgrad(xval,lyn,yval,
                                                     use_nonan=False,
                                                     zorder=2,
                                                     lwstart=2,
                                                     lwend=2,
                                                     cmapmin=cmapmin,
                                                     cmapmax=cmapmax,
                                                     return3d=False,
                                                     cmap=cmap))
        axlist[i].set_title(labellist[i])

        # index to place arrow head
        shift = 40
        idxlist = [shift,int(1.*len(xval)/5)+shift,
                   int(2.*len(xval)/5)+shift,
                   int(3.*len(xval)/5)+shift,
                   int(4.*len(xval)/5)+shift]# depends on period
        factor = 1.
        back_idx = 1

        for j in idxlist:
            axlist[i].annotate("",
                               xy=(factor*xval[j], factor*yval[j]), xycoords='data',
                               xytext=(factor*xval[j-back_idx], factor*yval[j-back_idx]), textcoords='data',
                               size=15,
                               arrowprops=dict(arrowstyle="-|>",
                                               connectionstyle="arc3",
                                               color='black')
            )

        # place label in bifurcation diagram
        ax11.annotate(labellist[i],xy=(glist[i],.8),
                      xycoords='data',xytext=(glist[i],.8),
                      textcoords='data',
                      size=12,
                      bbox=dict(boxstyle="round4,pad=.2", fc=(1.0, 0.7, 0.7)),
                      #backgroundcolor=labelbg,
                      zorder=-1
                  )


        axlist[i].set_xlim(-pi,pi)
        axlist[i].set_ylim(-pi,pi)
        axlist[i].tick_params(axis=u'both',which=u'both',length=0)

        axlist[i].set_xticks(np.arange(-1,1+1,1)*pi)
        axlist[i].set_yticks(np.arange(-1,1+1,1)*pi)
        #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
        axlist[i].set_xticklabels(x_label_short)
        axlist[i].set_yticklabels(x_label_short)



        if i >= 1:
            axlist[i].set_xticklabels([])

        if i == 1:
            axlist[i].set_ylim(-.5,.5)
            axlist[i].set_yticks(np.arange(-.5,.5+.5,.5))
            axlist[i].set_yticklabels(np.arange(-.5,.5+.5,.5))
            #axlist[i].set_yticks()

        axlist[i].yaxis.tick_right()
    # LOOP OVER SAMPLE SOLUTIONS

    rlist = [31,194,156,101]
    loclist = [(3,0),(3,1),(3,2),(3,3)]
    labellist = [r'$\mathbf{A}$',r'$\mathbf{B}$',r'$\mathbf{C}$',r'$\mathbf{D}$']

    start_idx = len(axlist)

    for i in range(len(rlist)):

        k = i + start_idx

        rown = rlist[i]
        g = bif_data[rown,3]
        per = bif_data[rown,5]
        init = init_data[rown,5:]
        print 'g=',g,'g=',init_data[rown,2],'per=',init_data[rown,4],'init=',init,loclist[i]

        dt = 1

        #print g,per,bif_data[rown,6:6+10]
    
        #axlist.append(plt.subplot2grid((4,4),loclist[i]))

        axlist.append(plt.subplot(gs[loclist[i][0],loclist[i][1]]))
        sxsval = bif_data[rown,6+idx]
        xval,yval = get_and_clean_sol_full_auto_5terms(init,per,g,dt)

        #for slc in unlink_wrap(xval):
        #    axlist[i].plot(xval[slc],yval[slc],color='black',lw=2)
        dashes = []

        print bif_data[rown,0]
        if abs(bif_data[rown,0]) == 4.:
            dashes = (5,2)


        #self.collect3d_colorgrad(v1a,ga,v2a)
        #axlist[i].plot(xval,yval,color='black',lw=2,dashes=dashes)

        lyn = np.linspace(0,1,len(xval))
        axlist[k].add_collection(collect3d_colorgrad(xval,lyn,yval,
                                                     use_nonan=False,
                                                     zorder=2,
                                                     lwstart=2,
                                                     lwend=2,
                                                     cmapmin=1.,
                                                     cmapmax=.0,
                                                     return3d=False,
                                                     cmap=cmap))


        # label 2 points with arrows
        back_idx = 10

        # index to place arrow head
        idxlist = [10,int(1.*(per/dt)/4),int(2.*(per/dt)/4),int(3.*(per/dt)/4)]# depends on period
        

        if i == 0:
            factor = 1.
            back_idx = 10
        else:
            factor = 1.
        
        for j in idxlist:
            axlist[k].annotate("",
                               xy=(factor*xval[j], factor*yval[j]), xycoords='data',
                               xytext=(factor*xval[j-back_idx], factor*yval[j-back_idx]), textcoords='data',
                               size=15,
                               arrowprops=dict(arrowstyle="-|>",
                                               connectionstyle="arc3",
                                               color='black')
            )





        ax11.annotate(labellist[i],xy=(g,sxsval),
                      xycoords='data',xytext=(g+.01,sxsval+.01),
                      textcoords='data',
                      size=12,
                      bbox=dict(boxstyle="round4,pad=.2", fc=(1.0, 0.7, 0.7)),
                      #backgroundcolor=labelbg,
                      zorder=-1
                  )


        axlist[k].set_title(labellist[i])


        # fix all subplots

        axlist[k].set_xlim(-pi,pi)
        axlist[k].set_ylim(-pi,pi)
        axlist[k].tick_params(axis=u'both',which=u'both',length=0)

        axlist[k].set_xticks(np.arange(-1,1+1,1)*pi)
        axlist[k].set_yticks(np.arange(-1,1+1,1)*pi)

        #x_label = [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$",   r"$\pi$"]
        axlist[k].set_xticklabels(x_label_short)
        axlist[k].set_yticklabels(x_label_short)

        if i >= 1:
            axlist[k].set_yticklabels([])

        # annotations corresponding to solution plots


        ax11.scatter(g,sxsval,s=50,color='black',marker='*',zorder=2)


    ax11.xaxis.set_major_formatter(formatter)
    ax11.yaxis.set_major_formatter(formatter)

    return fig


def get_solution(g,q,init,per,dt=1.):
    """
    get solution value. specialized function for twod_full_auto_5terms_2par.
    """
    npa, vn = xpprun('full2dbump.ode',
                     xppname='xppaut',
                     inits={'a0':init[0],'a10':init[1],
                            'a01':init[2],'a11':init[3],
                            'b10':init[4],'b01':init[5],
                            'b11':init[6],'c11':init[7],'d11':init[8],
                            'e0':init[9],'e10':init[10],
                            'e01':init[11],'e11':init[12],
                            'f10':init[13],'f01':init[14],
                            'f11':init[15],'g11':init[16],'h11':init[17]
                        },
                     parameters={'total':per,
                                 'g':g,
                                 'q':q,
                                 'eps':.05,
                                 'dt':dt},
                     clean_after=True)
    
    t = npa[:,0]
    sv = npa[:,1:]
    
    idx = vn.index('a11')
    #print idx, vn
    
    b01 = sv[:,vn.index('b01')]
    a01 = sv[:,vn.index('a01')]
    b10 = sv[:,vn.index('b10')]
    a10 = sv[:,vn.index('a10')]
    
    xval = np.mod(np.arctan2(b01,a01)+5*pi,2*pi)-pi
    yval = np.mod(np.arctan2(b10,a10)+5*pi,2*pi)-pi
    
    pos1 = np.where(np.abs(np.diff(xval)) >= 1)[0]
    pos2 = np.where(np.abs(np.diff(yval)) >= 1)[0]


    xval[pos1] = np.nan
    yval[pos1] = np.nan
    
    xval[pos2] = np.nan
    yval[pos2] = np.nan

    return xval,yval


def twod_full_auto_5terms_2par(subplots=4):
    """
    neural field with truncated mexican hat kernel.
    approximate 2 parameter dynamics.
    """

    # data
    bif_data = np.loadtxt('full2dbump.ode.diagram.q=.1.dat')
    init_data = np.loadtxt('full2dbump.ode.diagram.q=.1.inits.dat')

    rlist = [31,101,140]
    rown = rlist[0]
    dt = 1.
    init1 = init_data[rown,5:]


    label = [r"$-\pi$", r"$0$", r"$\pi$"]

    fig = plt.figure(figsize=(5,5))
    #ax = fig.add_subplot(111)

    # plot pretend lines
    plt.plot([.1,.5],[0,1],ls='--',color='orange',lw=2,label='HB') # HB
    plt.plot(np.linspace(.1,1,20),.3*(np.linspace(.1,1,20)-.1+((np.linspace(.1,1,20)-.1))**.5),ls='-.',color='purple',lw=2,label='LP') # LP

    plt.axis([0,1,0,1])




    # annotate regions
    plt.annotate('1. Stationary Bump',xy=(.01,.9))
    plt.annotate('2. Sloshing Bump',xy=(.6,.8))
    plt.annotate('3. Chaos \& Traveling Bump.',xy=(.5,.46),rotation=27)

    plt.tick_params(axis=u'both',which=u'both',length=0,size=0,labelbottom='off',labelleft='off')
    #plt.xlim(0,1)
    #plt.ylim(0,1)


    plt.xlabel(r'$\bm{g}$')
    plt.ylabel(r'$\bm{q}$')

    # point to where sloshing solution comes from
    plt.annotate('',xy=(.22,.2),xytext=(.4,.4),
                 xycoords='data',textcoords='data',
                 arrowprops=dict(arrowstyle="->",fc='black',connectionstyle="arc3,rad=0.04"))

    # point to where const. velocity solution comes from
    plt.annotate('',xy=(.6,0),xytext=(.55, .13),
                 xycoords='data',textcoords='data',
                 arrowprops=dict(arrowstyle="->",fc='black',connectionstyle="arc3,rad=-0.5"))



    ## stationary solution

    # this is an inset axes over the main axes
    ax1 = plt.axes([.2, .7, .1, .1])
    per2 = 10.
    g2 = .1
    q2 = 5.
    
    xval2,yval2 = get_solution(g2,q2,np.zeros(len(init1)),per2,dt=dt)
    plt.scatter(xval2,yval2,s=5,color='black')

    #n, bins, patches = plt.hist(s, 400, normed=1)
    #plt.title('Probability')
    plt.xticks([-pi,0,pi],x_label_short)
    plt.yticks([-pi,0,pi],x_label_short)
    plt.tick_params(bottom='off',left='off',right='off',top='off')


    ## sloshing solution (small slosh)
    ax2 = plt.axes([.4, .4, .1, .1])
    g1 = bif_data[rown,3]
    q1 = .1
    per1 = bif_data[rown,5]
    
    
    #print 'g=',g,'q=',q,'per=',init_data[rown,4],'init=',init    
    xval,yval = get_solution(g1,q1,init1,per1,dt=dt)
    
    """
    lyn = np.linspace(0,1,len(xval))
    ax2.add_collection(collect3d_colorgrad(xval,lyn,yval,
                                          use_nonan=False,
                                          zorder=2,
                                          lwstart=1,
                                          lwend=3,
                                          cmapmin=.5,
                                          cmapmax=.0,
                                          return3d=False,
                                          cmap='gray'))
    """

    ax2 = beautify_phase(ax2,xval,yval,per1,dt,gradient=True,arrowsize=10)

    #ax2.plot(xval,yval)
    #axins1 = beautify_phase(axins1,xval,yval,per1,dt)
    plt.tick_params(axis=u'both',which=u'both',length=0,bottom='off',left='off',right='off',top='off')
    plt.xlim(-pi,pi)
    plt.ylim(-pi,pi)
    plt.xticks(np.arange(-1,1+1,1)*pi,x_label_short)
    plt.yticks(np.arange(-1,1+1,1)*pi,x_label_short)

    plt.title(r'$\bm{g='+str(np.round(g1,1))+r'}$'+'\n'+r'$\bm{q='+str(q1)+r'}$',y=1,backgroundcolor='white',size=10)

    
    ## constant velocity
    ax4 = plt.axes([.5, .13, .1, .1])
    rown = rlist[2]
    g4 = 3.
    q4 = .0
    per4 = 600#bif_data[rown,5]
    init4 = np.random.randn(len(init1))
    dt = 1.
    #print 'g=',g,'q=',q,'per=',init_data[rown,4],'init=',init
    
    xval4,yval4 = get_solution(g4,q4,init4,per4,dt=dt)
    
    xval4 = xval4[int(300/dt):]
    yval4 = yval4[int(300/dt):]

    ax4 = beautify_phase(ax4,xval4,yval4,250,dt,gradient=True,arrowsize=10)
    #plt.plot(xval4,yval4)

    plt.tick_params(axis=u'both',which=u'both',length=0,bottom='off',left='off',right='off',top='off')
    plt.xlim(-pi,pi)
    plt.ylim(-pi,pi)
    plt.xticks(np.arange(-1,1+1,1)*pi,x_label_short)
    plt.yticks(np.arange(-1,1+1,1)*pi,x_label_short)

    plt.title(r'$\bm{g='+str(np.round(g4,1))+r'}$'+'\n'+r'$\bm{q='+str(q4)+r'}$',y=1,backgroundcolor='white',size=10)


    ## nonconstant velocity 

    ax3 = plt.axes([.8, .25, .1, .1])
    rown = rlist[2]
    g3 = 4.
    q3 = .5
    per3 = 2000#bif_data[rown,5]
    init3 = np.random.randn(len(init1))
    dt = 1.
    #print 'g=',g,'q=',q,'per=',init_data[rown,4],'init=',init
    
    xval3,yval3 = get_solution(g3,q3,init3,per3,dt=dt)
    
    xval3 = xval3[-int(500/dt):]
    yval3 = yval3[-int(500/dt):]
    
    #print xval3,yval3

    ax3 = beautify_phase(ax3,xval3,yval3,500,dt,gradient=True,arrowsize=10,arrows=5)
    #plt.plot(xval3,yval3)

    plt.tick_params(axis=u'both',which=u'both',length=0,bottom='off',left='off',right='off',top='off')
    plt.xlim(-pi,pi)
    plt.ylim(-pi,pi)
    plt.xticks(np.arange(-1,1+1,1)*pi,label)
    plt.yticks(np.arange(-1,1+1,1)*pi,label)

    plt.title(r'\bm{$g='+str(np.round(g3,1))+r'}$'+'\n'+r'$\bm{q='+str(q3)+r'}$',y=1,backgroundcolor='white',size=10)

    return fig


def fac(nu,s0=0.,s1=10.,sn=101):
    """
    nu/int e^-s H1(nu s,0)ds
    """

    sarr = np.linspace(s0,s1,sn)
    ds = (s1 - s0)/sn

    g_denom = 0.

    for s in sarr:
        g_denom += np.exp(-s)*f2d.H1_fourier(nu*s,0)*ds
        #print f2d.H1_fourier(nu*s,0)
    #print g_denom

    return nu/g_denom

def ix(nu,lam,choice=1,s0=0.,s1=10.,sn=101):

    sarr = np.linspace(s0,s1,sn)
    ds = (s1 - s0)/sn

    i = 0.
    for s in sarr:
        
        if choice == 1:
            h1x,h1y = f2d.H1_fourier(nu*s,0,d=True)
        else:
            h1x,h1y = f2d.H1_fourier(0,nu*s,d=True)

        i += np.exp(-s)*h1x*((np.exp(-lam*s)-1)/lam)*ds
    return i
    
def L(nu,lam,choice=1,s0=0.,s1=50.,sn=501):
    """
    L1 = 1 + g int_0^\infty e^-s \pa H_1/\pa x (nu s, 0) (e^{-\lambda s} - 1)/\lambda ds
    """
    sarr = np.linspace(s0,s1,sn)
    ds = (s1 - s0)/sn
    
    g_denom = 0.

    for s in sarr:
        g_denom += np.exp(-s)*f2d.H1_fourier(nu*s,0)*ds

        #print f2d.H1_fourier(nu*s,0)
    #print g_denom

    factor = nu/g_denom

    i = 0.
    for s in sarr:
        
        if choice == 1:
            h1x,h1y = f2d.H1_fourier(nu*s,0,d=True)
        else:
            h1x,h1y = f2d.H1_fourier(0,nu*s,d=True)

        i += np.exp(-s)*h1x*((np.exp(-lam*s)-1.)/lam)*ds

    
    return 1. + factor*i

def L_ana(nu,lam,choice=1):
    """
    analytic version of above. found using long but finite fourier truncation
    """
    print 'using analytic L1,L2'

    g = 1/(1.04609467947464/(1 + nu**2) + 0.06989538102574108/(1 + 4*nu**2) + (5.002469208435e-6)/(1 + 9*nu**2))

    if choice == 1:
        integral = -1.04609467947464/(1 + nu**2) + (1.04609467947464*(1 + lam))/((1 + lam)**2 + nu**2) - 0.06989538102574108/(1 + 4*nu**2) + (0.06989538102574108*(1 + lam))/((1 + lam)**2 + 4*nu**2) - (5.002469208435e-6)/(1 + 9*nu**2) + ((5.002469208435e-6)*(1 + lam))/((1 + lam)**2 + 9*nu**2)

    if choice == 2:
        integral = -0.6474559245027758 + 0.6474559245027758/(1. + lam) - 0.46310403788291776/(1 + nu**2) + (0.46310403788291776*(1 + lam))/((1 + lam)**2 + nu**2) - 0.005435100583895982/(1 + 4*nu**2) + (0.005435100583895982*(1 + lam))/((1 + lam)**2 + 4*nu**2)

    #return lam + g*integral
    return 1 + g*integral/lam
        
def wave_stbl_2d(choice='axial'):
    """
    compute eigenvalue as a function of traveling bump velocity in axial or diagonal directions.

    \lambda_1 &= -\frac{\nu}{\int_0^\infty e^{-s} H_1(\nu s, 0) ds} \int_0^\infty e^{-s} \frac{\pa H_1}{\pa x}(\nu s, 0)[e^{-\lambda_1 s}-1]ds,\\
    \lambda_2 &=  \frac{\nu}{\int_0^\infty e^{-s} H_1(\nu s, 0) ds} \int_0^\infty e^{-s} \frac{\pa H_1}{\pa y}(0, \nu s)[e^{-\lambda_2 s}-1]ds.
    """
    
    # import real and imaginary parts past the weird bifurcation
    bif = np.loadtxt('twod_wave_stble_test.ode.axial.allinfo.dat')
    

    dat_g = bif[:,3]
    dat_re = bif[:,6]
    dat_im = bif[:,7]
    
    
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(121)

    nu = np.linspace(.00,1.,200)
    lam = np.linspace(-1,1,200)

    X,Y = np.meshgrid(nu,lam,indexing='xy')

    #L(nu,lam,choice=1,s0=0,s1=10,sn=101):
    #z = L(X,Y)
    z = L_ana(X,Y)
    z[z>=5] = 5
    z[z<=-5] = -5

    #ax.plot([0,10],[0,0],color='gray')
    ax.plot(dat_g,dat_re,color='black',lw=2,label='Re')
    ax.plot(dat_g,dat_im,color='gray',lw=2)
    ax.plot(dat_g,-dat_im,color='gray',lw=2,label='Im')

    ax.plot([0,.33],[0,0],lw=2,color='gray')
    cs = ax.contour(X,Y,z,levels=[0.])

    # zero line
    ax.plot([0,1],[0,0],color='gray',ls='--')
    
    
    # remove other curves
    # customize desired curve
    cs.collections[0].set_color('black')
    cs.collections[0].set_linewidth(2)


    #cbar = plt.colorbar(cs)
    #cbar.add_lines(cs)
    ax.set_ylabel(r'$\lambda_1$')
    ax.set_xlabel(r'$\nu$')



    #ax.annotate(r'Stability of solution $\theta_1(\tau)=\nu\tau$',xy=(5.1,.75))
    #ax.annotate(r'Unstable',xy=(7.2,.05))
    #ax.annotate(r'Stable',xy=(7.2,-.12))

    ax2 = fig.add_subplot(122)

    #nu = np.linspace(0,1.5,150)
    lam = np.linspace(-.05,.25,150)

    X,Y = np.meshgrid(nu,lam,indexing='xy')

    z2 = L_ana(X,Y,choice=2)
    #z2 = L(X,Y,choice=2)
    z2[z2>=5] = 5
    z2[z2<=-5] = -5

    #ax2.plot([0,10],[0,0],color='gray')

    cs2 = ax2.contour(X,Y,z2,levels=[0.])
    cs2.collections[0].set_color('black')
    cs2.collections[0].set_linewidth(2)

    # zero line
    ax2.plot([0,1],[0,0],color='gray',ls='--')



    #ax2.annotate(r'Stability of solution $\theta_2(\tau)=0$',xy=(.1,.9))
    #ax2.annotate(r'Unstable',xy=(1.25,.019))
    #ax2.annotate(r'Stable',xy=(1.25,-.028))

    ax2.set_ylabel(r'$\lambda_2$')
    ax2.set_xlabel(r'$\nu$')

    ax.set_xlim(0,1.)
    ax2.set_xlim(0,1.)

    ax.set_ylim(-1.5,1.5)
    ax2.set_ylim(-.05,.25)

    ax.legend(loc=2)

    #ax2.set_ylabel(r'$\nu$')
    #p = cs.collections[0].get_paths()[1]
    #v = p.vertices

    #cbar2 = plt.colorbar(cs2)
    #cbar2.add_lines(cs2)

    #fig.set_clabel(cs, inline=1, fontsize=10)
    #print fig.__dict__

    #ax.plot(v[:,0],v[:,1],color='black')

    
    return fig



    
def L_trunc(nu,lam,choice=1,s0=0.,s1=50.,sn=501):
    """
    L1 = 1 + g int_0^\infty e^-s \pa H_1/\pa x (nu s, 0) (e^{-\lambda s} - 1)/\lambda ds
    """
    sarr = np.linspace(s0,s1,sn)
    ds = (s1 - s0)/sn
    
    g_denom = 0.

    for s in sarr:
        g_denom += np.exp(-s)*f2d.H1_fourier(nu*s,0)*ds

        #print f2d.H1_fourier(nu*s,0)
    #print g_denom

    factor = nu/g_denom

    i = 0.
    for s in sarr:
        
        if choice == 1:
            h1x,h1y = f2d.H1_fourier(nu*s,0,d=True)
        else:
            h1x,h1y = f2d.H1_fourier(0,nu*s,d=True)

        i += np.exp(-s)*h1x*((np.exp(-lam*s)-1.)/lam)*ds

    
    return 1. + factor*i

def L_ana_trunc(nu,lam,choice=1):
    """
    analytic version of above. found using long but finite fourier truncation
    """
    print 'using analytic L1,L2'

    g = 1/(1.04609467947464/(1 + nu**2) + 0.06989538102574108/(1 + 4*nu**2) + (5.002469208435e-6)/(1 + 9*nu**2))

    if choice == 1:
        integral = -1.04609467947464/(1 + nu**2) + (1.04609467947464*(1 + lam))/((1 + lam)**2 + nu**2) - 0.06989538102574108/(1 + 4*nu**2) + (0.06989538102574108*(1 + lam))/((1 + lam)**2 + 4*nu**2) - (5.002469208435e-6)/(1 + 9*nu**2) + ((5.002469208435e-6)*(1 + lam))/((1 + lam)**2 + 9*nu**2)

    if choice == 2:
        integral = -0.6474559245027758 + 0.6474559245027758/(1. + lam) - 0.46310403788291776/(1 + nu**2) + (0.46310403788291776*(1 + lam))/((1 + lam)**2 + nu**2) - 0.005435100583895982/(1 + 4*nu**2) + (0.005435100583895982*(1 + lam))/((1 + lam)**2 + 4*nu**2)

    return lam + g*integral



def wave_stbl_2d_trunc(choice='axial'):
    """
    compute eigenvalue as a function of traveling bump velocity in axial or diagonal directions.

    \lambda_1 &= -\frac{\nu}{\int_0^\infty e^{-s} H_1(\nu s, 0) ds} \int_0^\infty e^{-s} \frac{\pa H_1}{\pa x}(\nu s, 0)[e^{-\lambda_1 s}-1]ds,\\
    \lambda_2 &=  \frac{\nu}{\int_0^\infty e^{-s} H_1(\nu s, 0) ds} \int_0^\infty e^{-s} \frac{\pa H_1}{\pa y}(0, \nu s)[e^{-\lambda_2 s}-1]ds.
    """

    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(121)

    nu = np.linspace(.00,1.5,200)
    lam = np.linspace(-1,1,200)

    X,Y = np.meshgrid(nu,lam,indexing='xy')

    #L(nu,lam,choice=1,s0=0,s1=10,sn=101):
    #z = L(X,Y)
    z = L_ana(X,Y)
    z[z>=5] = 5
    z[z<=-5] = -5

    #ax.plot([0,10],[0,0],color='gray')
    cs = ax.contour(X,Y,z,levels=[0.])
    
    
    # remove other curves
    # customize desired curve
    cs.collections[0].set_color('black')
    cs.collections[0].set_linewidth(2)


    #cbar = plt.colorbar(cs)
    #cbar.add_lines(cs)
    ax.set_ylabel(r'$\lambda_1$')
    ax.set_xlabel(r'$\nu$')



    #ax.annotate(r'Stability of solution $\theta_1(\tau)=\nu\tau$',xy=(5.1,.75))
    #ax.annotate(r'Unstable',xy=(7.2,.05))
    #ax.annotate(r'Stable',xy=(7.2,-.12))

    ax2 = fig.add_subplot(122)

    #nu = np.linspace(0,1.5,150)
    lam = np.linspace(-.05,.25,150)

    X,Y = np.meshgrid(nu,lam,indexing='xy')

    z2 = L_ana(X,Y,choice=2)
    #z2 = L(X,Y,choice=2)
    z2[z2>=5] = 5
    z2[z2<=-5] = -5

    #ax2.plot([0,10],[0,0],color='gray')

    cs2 = ax2.contour(X,Y,z2,levels=[0.])

    cs2.collections[0].set_color('black')
    cs2.collections[0].set_linewidth(2)




    #ax2.annotate(r'Stability of solution $\theta_2(\tau)=0$',xy=(.1,.9))
    #ax2.annotate(r'Unstable',xy=(1.25,.019))
    #ax2.annotate(r'Stable',xy=(1.25,-.028))

    ax2.set_ylabel(r'$\lambda_2$')
    ax2.set_xlabel(r'$\nu$')

    ax.set_xlim(0,1.5)
    ax2.set_xlim(0,1.5)

    ax2.set_ylim(-.05,.25)

    #ax2.set_ylabel(r'$\nu$')
    #p = cs.collections[0].get_paths()[1]
    #v = p.vertices

    #cbar2 = plt.colorbar(cs2)
    #cbar2.add_lines(cs2)

    #fig.set_clabel(cs, inline=1, fontsize=10)
    #print fig.__dict__

    #ax.plot(v[:,0],v[:,1],color='black')
    
    return fig


def L_gauss(nu,lam,choice=1,sig=5.):
    """
    gaussian eigenvalue problem
    """

    tot = 0
    g_integral = 0
    spi = sqrt(pi)

    L1 = 1 + (5*nu**4*\
              ((4*lam*Sqrt(nu**2))/5. + (2*E**((4*Pi**2)/25.)*lam*Sqrt(nu**2))/5. + \
               Sqrt(Pi)*(E**(25/(4.*nu**2) + (4*Pi**2)/25.)*Erfc(5/(2.*Sqrt(nu**2))) - \
                         E**((25*(1 + lam)**2)/(4.*nu**2) + (4*Pi**2)/25.)*(1 + lam)**2*\
                         Erfc((5*(1 + lam))/(2.*Sqrt(nu**2))) + \
                         E**((25 - 4*nu*Pi)**2/(100.*nu**2))*Erfc((25 - 4*nu*Pi)/(10.*Sqrt(nu**2))) - \
                         E**((25*(1 + lam) - 4*nu*Pi)**2/(100.*nu**2))*(1 + lam)**2*\
                         Erfc((25*(1 + lam) - 4*nu*Pi)/(10.*Sqrt(nu**2))) + \
                         E**((25 + 4*nu*Pi)**2/(100.*nu**2))*Erfc((25 + 4*nu*Pi)/(10.*Sqrt(nu**2)))) - \
               E**((25*(1 + lam) + 4*nu*Pi)**2/(100.*nu**2))*(1 + lam)**2*Sqrt(Pi)*\
               Erfc((25*(1 + lam) + 4*nu*Pi)/(10.*Sqrt(nu**2)))))/\
        (lam*(nu**2)**1.5*(2*(2 + E**((4*Pi**2)/25.))*nu**2 - \
                           5*E**(25/(4.*nu**2) + (4*Pi**2)/25.)*Sqrt(nu**2)*Sqrt(Pi)*(1 + 2*Cosh((2*Pi)/nu)) + \
                           5*E**((25 - 4*nu*Pi)**2/(100.*nu**2))*nu*Sqrt(Pi)*\
                           (E**((2*Pi)/nu)*Erf(5/(2.*nu)) + Erf(5/(2.*nu) - (2*Pi)/5.)) + \
                           5*E**((25 + 4*nu*Pi)**2/(100.*nu**2))*nu*Sqrt(Pi)*Erf(5/(2.*nu) + (2*Pi)/5.)))
    

    L2 = 1 - (2*(nu**2)**1.5*Sqrt(Pi)*(25*(2 + E**((4*Pi**2)/25.)) - 16*Pi**2)*\
              (E**(25/(4.*nu**2) + (4*Pi**2)/25.)*Erfc(5/(2.*Sqrt(nu**2))) - \
               E**((25*(1 + lam)**2)/(4.*nu**2) + (4*Pi**2)/25.)*Erfc((5*(1 + lam))/(2.*Sqrt(nu**2))) + \
               E**((25 - 4*nu*Pi)**2/(100.*nu**2))*Erfc((25 - 4*nu*Pi)/(10.*Sqrt(nu**2))) - \
               E**((25*(1 + lam) - 4*nu*Pi)**2/(100.*nu**2))*\
               Erfc((25*(1 + lam) - 4*nu*Pi)/(10.*Sqrt(nu**2))) + \
               E**((25 + 4*nu*Pi)**2/(100.*nu**2))*Erfc((25 + 4*nu*Pi)/(10.*Sqrt(nu**2))) - \
               E**((25*(1 + lam) + 4*nu*Pi)**2/(100.*nu**2))*\
               Erfc((25*(1 + lam) + 4*nu*Pi)/(10.*Sqrt(nu**2)))))/\
        (125.*(2 + E**((4*Pi**2)/25.))*lam*\
         (2*(2 + E**((4*Pi**2)/25.))*nu**2 - \
          5*E**(25/(4.*nu**2) + (4*Pi**2)/25.)*Sqrt(nu**2)*Sqrt(Pi)*(1 + 2*Cosh((2*Pi)/nu)) + \
          5*E**((25 - 4*nu*Pi)**2/(100.*nu**2))*nu*Sqrt(Pi)*\
          (E**((2*Pi)/nu)*Erf(5/(2.*nu)) + Erf(5/(2.*nu) - (2*Pi)/5.)) + \
          5*E**((25 + 4*nu*Pi)**2/(100.*nu**2))*nu*Sqrt(Pi)*Erf(5/(2.*nu) + (2*Pi)/5.)))


    if choice == 1:
        return L1
    return L2
    


def wave_stbl_2d_gauss():
    """
    compute eigenvalue as a function of traveling bump velocity in axial or diagonal directions.
    the h function is given to be the negative derivative of the gaussian.

    \lambda_1 &= -\frac{\nu}{\int_0^\infty e^{-s} H_1(\nu s, 0) ds} \int_0^\infty e^{-s} \frac{\pa H_1}{\pa x}(\nu s, 0)[e^{-\lambda_1 s}-1]ds,\\
    \lambda_2 &=  \frac{\nu}{\int_0^\infty e^{-s} H_1(\nu s, 0) ds} \int_0^\infty e^{-s} \frac{\pa H_1}{\pa y}(0, \nu s)[e^{-\lambda_2 s}-1]ds.
    """

    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    nu = np.linspace(.001,2.,50)
    lam = np.linspace(-1,1,50)
    nu2 = np.linspace(.001,2.,50)
    lam2 = np.linspace(-1,1,50)

    X,Y = np.meshgrid(nu,lam,indexing='xy')
    X2,Y2 = np.meshgrid(nu2,lam2,indexing='xy')

    #L(nu,lam,choice=1,s0=0,s1=10,sn=101):
    z = L_gauss(X,Y)
    z2 = L_gauss(X2,Y2,choice=2)

    z[z>=5] = 5
    z[z<=-5] = -5

    z2[z2>=5] = 5
    z2[z2<=-5] = -5

    ax.plot([0,nu[-1]],[0,0],color='gray')
    ax2.plot([0,nu2[-1]],[0,0],color='gray')

    cs = ax.contour(X,Y,z,levels=[-0.00000001,0.,.00000001])
    cs2 = ax2.contour(X2,Y2,z2,levels=[-0.00000001,0.,.00000001])

    #cs = ax.contour(X,Y,z)
    #cs2 = ax2.contour(X2,Y2,z2)
    
    # remove other curves
    # customize desired curve
    cs.collections[0].set_color('black')
    cs.collections[1].set_color('black')
    cs.collections[2].set_color('black')
    cs2.collections[0].set_color('black')
    cs2.collections[1].set_color('black')
    cs2.collections[2].set_color('black')

    cs.collections[1].set_linewidth(2)
    cs2.collections[1].set_linewidth(2)

    ax.set_xlabel(r'$\nu$')
    ax2.set_xlabel(r'$\nu$')

    ax.set_ylabel(r'$\lambda_1$')
    ax2.set_ylabel(r'$\lambda_2$')
    
    return fig


def wave_exist_2d(choice='axial'):
    """
    plot zeros of -nu1 + G(nu1,nu2) and -nu2 + G(nu2,nu1)
    """

    nc1x = np.loadtxt('twod_wave_exist_nc_g=1x.dat')
    nc1y = np.loadtxt('twod_wave_exist_nc_g=1y.dat')

    nc2x = np.loadtxt('twod_wave_exist_nc_g=3x.dat')
    nc2y = np.loadtxt('twod_wave_exist_nc_g=3y.dat')

    # nc1 bifurcation values
    bif = np.loadtxt('twod_wave_exist_br1.dat')
    #bif2 = np.loadtxt('twod_wave_exist_br2.dat')

    bif_diag1 = np.loadtxt('twod_wave_exist_diag1.dat')
    bif_diag2 = np.loadtxt('twod_wave_exist_diag2.dat')


    # clean
    nc1xx,nc1xy = clean(nc1x[:,0],nc1x[:,1],tol=.05)
    nc1yx,nc1yy = clean(nc1y[:,0],nc1y[:,1],tol=.05)

    nc2xx,nc2xy = clean(nc2x[:,0],nc2x[:,1],tol=.1)
    nc2yx,nc2yy = clean(nc2y[:,0],nc2y[:,1],tol=.1)

    bifx,bify = clean(bif[:,3],bif[:,7],tol=1)
    bifx2,bify2 = clean(bif[:,3],bif[:,8],tol=.2)

    bif_diag1x,bif_diag1y = clean(bif_diag1[:,0],np.abs(bif_diag1[:,1]),tol=.2)
    bif_diag2x,bif_diag2y = clean(bif_diag2[:,0],np.abs(bif_diag2[:,1]),tol=.2)

    fig = plt.figure(figsize=(8,3))
    ax1 = fig.add_subplot(121)
    ax1.plot(nc1xx,nc1xy)
    ax1.plot(nc1yx,nc1yy)

    ax1.plot(nc2xx,nc2xy)
    ax1.plot(nc2yx,nc2yy)

    #ax1.plot(nc2yx,nc2yy)



    ax1.annotate(r'$g=1$',
                 xy=(.21, .21), xycoords='data',
                 xytext=(.3, .75), textcoords='data',
                 size=12,
                 zorder=2,
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
              )


    ax1.annotate(r'$g=3$',
                 xy=(.934, 1.37), xycoords='data',
                 xytext=(1.3, 1.7), textcoords='data',
                 size=12,
                 zorder=2,
                 verticalalignment='top',
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
              )

    ax1.annotate(r'$g=3$',
                 alpha=0.0,
                 xy=(1.16, 1.16), xycoords='data',
                 xytext=(1.3, 1.7), textcoords='data',
                 size=12,
                 zorder=2,
                 verticalalignment='top',
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
              )

    ax1.annotate(r'$g=3$',
                 alpha=0.0,
                 xy=(1.379,.923), xycoords='data',
                 xytext=(1.3, 1.7), textcoords='data',
                 size=12,
                 zorder=2,
                 verticalalignment='top',
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
              )

    axins1 = inset_axes(ax1,
                        width="30%",
                        height="30%",
                        loc=8)

    mark_inset(ax1, axins1, loc1=2, loc2=4, fc="none", ec="0.5")


    axins1.plot(nc1xx,nc1xy)
    axins1.plot(nc1yx,nc1yy)
    
    axins1.set_xlim(.1,.3)
    axins1.set_ylim(.1,.3)

    #mark_inset(ax21, axins21, loc1=2, loc2=1, fc="none", ec="0.5")

    plt.xticks(visible=False)
    plt.yticks(visible=False)



    ax2 = fig.add_subplot(122)
    ax2.plot(bifx,bify,color='black')
    ax2.plot(bifx2,bify2,color='black')

    ax2.plot(bif_diag1x,bif_diag1y,color='black')
    ax2.plot(bif_diag2x,bif_diag2y,color='black')

    ax2.plot([0,5],[0,0],color='black')

    ax2.annotate(r'$x$-axis direction',
                 xy=(1.04,.37),xycoords='data',textcoords='data',
                 xytext=(.6,.6),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'$y$-axis direction',
                 xy=(1.0,.0),xycoords='data',textcoords='data',
                 xytext=(.55,.33),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'$g^*$',
                 xy=(.9,.0),xycoords='data',textcoords='data',
                 xytext=(.8,.05),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Diagonal',
                 xy=(1.1,.32),xycoords='data',textcoords='data',
                 xytext=(1.4,.2),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Off-diagonal',
                 xy=(1.4,.41),xycoords='data',textcoords='data',
                 xytext=(1.5,.34),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Off-diagonal',
                 alpha=0.,
                 xy=(1.4,.62),xycoords='data',textcoords='data',
                 xytext=(1.5,.34),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )



    """
    import matplotlib.patches as patches
    ax2.add_patch(
        patches.Rectangle(
            (1.17,-3),1,6,
            color='red',
            alpha=.25
        )
    )
    """

    ax2.annotate('Multiple non-axial directions',xy=(3.68,.1),xycoords='data',textcoords='data',xytext=(3.,.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.plot([1.17,1.17],[-3,3],color='gray')
    ax2.plot([3.,3.],[-3,3],color='gray')    

    ax1.set_ylabel(r'$\nu_2$')
    ax1.set_xlabel(r'$\nu_1$')

    ax2.set_ylabel(r'$\nu_1$')
    ax2.set_xlabel(r'$g$')

    ax1.set_xlim(-.05,2.)
    ax1.set_ylim(-.05,2.)

    ax2.set_xlim(.5,2.)
    ax2.set_ylim(-.1,1)
    
    
    return fig

def wave_exist_2d_v2():

    # nc1 bifurcation values
    L1 = np.loadtxt('twod_wave_exist_br1.dat')
    L2 = np.loadtxt('twod_wave_exist_diag1.dat')

    M1 = np.loadtxt('twod_wave_exist_br2.dat')
    M2 = np.loadtxt('twod_wave_exist_diag2.dat')

    # clean
    bifx,bify = clean(L1[:,3],L1[:,7],tol=1)
    bifx2,bify2 = clean(bif[:,3],bif[:,8],tol=.2)

    bif_diag1x,bif_diag1y = clean(bif_diag1[:,0],np.abs(bif_diag1[:,1]),tol=.2)
    bif_diag2x,bif_diag2y = clean(bif_diag2[:,0],np.abs(bif_diag2[:,1]),tol=.2)

    fig = plt.figure(figsize=(8,3))
    ax1 = fig.add_subplot(121)
    ax1.plot(nc1xx,nc1xy)
    ax1.plot(nc1yx,nc1yy)

    ax1.plot(nc2xx,nc2xy)
    ax1.plot(nc2yx,nc2yy)




    plane1_z = 0.55
    plane2_z = 0.889

    g = np.linspace(0+.0*1j,2+0.*1j,1000)

    # nu1 branches
    L1 = Sqrt(-1 + 1.8*g)
    L2 = Sqrt(-5 + (4 + b)*g + Sqrt(9 + 6*(-4 + b)*g + (4 + b)**2*g**2))/(2.*Sqrt(2))
    L3 = Sqrt(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) - \
              Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + \
                   2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))/2.
    L4 = Sqrt(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) + \
              Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + \
                   2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))/2.
    L5 = 0.*g


    # nu2 branches
    M1 = 0.*g
    M2 = np.real(Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)*(-5 + (4 + b)*g + Sqrt(9 + 6*(-4 + b)*g + (4 + b)**2*g**2))))/(2.*Sqrt(2)*Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)))))
    M3 = np.real(Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)*(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) + Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + 2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))))/(2.*Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)))))
    M4 = np.real(Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)*(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) - Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + 2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))))/(2.*Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)))))
    M5 = Sqrt(-1 + 1.8*g)
    print M2

    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)


    # get plane intersection idx
    g_int_p1 = np.argmin(np.abs(g-plane1_z))
    g_int_p2 = np.argmin(np.abs(g-plane2_z))

    ax1.scatter(L1[g_int_p1],g[g_int_p1],M1[g_int_p1],color='black',s=20)
    ax1.scatter(L1[g_int_p2],g[g_int_p2],M1[g_int_p2],color='black',s=20)

    ax1.scatter(L2[g_int_p1],g[g_int_p1],M2[g_int_p1],color='black',s=20)
    ax1.scatter(L2[g_int_p2],g[g_int_p2],M2[g_int_p2],color='black',s=20)

    ax1.scatter(L3[g_int_p1],g[g_int_p1],M3[g_int_p1],color='black',s=20)
    ax1.scatter(L3[g_int_p2],g[g_int_p2],M3[g_int_p2],color='black',s=20)

    ax1.scatter(L4[g_int_p1],g[g_int_p1],M4[g_int_p1],color='black',s=20)
    ax1.scatter(L4[g_int_p2],g[g_int_p2],M4[g_int_p2],color='black',s=20)

    # plot curves in 3d
    ax1.plot(L1,g,M1,color='black',lw=2)
    ax1.plot(L2,g,M2,color='black',lw=2)
    ax1.plot(L3,g,M3,color='black',lw=2)
    ax1.plot(L4,g,M4,color='black',lw=2)
    ax1.plot(L5,g,M5,color='black',lw=2)

    # plot curves in 2d
    ax2.plot(g,L1,color='black',lw=2)
    ax2.plot(g,L2,color='black',lw=2)
    ax2.plot(g,L3,color='black',lw=2)
    ax2.plot(g,L4,color='black',lw=2)

    # plot bifurcation planes
    X,Y = np.meshgrid(np.linspace(g[0],g[-1],10),np.linspace(g[0],g[-1],10))
    ax1.plot_surface(X,0.*X+plane1_z,Y,alpha=.3,color='gray')
    ax1.plot_surface(X,0.*X+plane2_z,Y,alpha=.3,color='red')


    # plot bifurcation lines
    ax2.plot([plane1_z,plane1_z],[0,1.8],color='black',alpha=.5,lw=2)
    ax2.plot([plane2_z,plane2_z],[0,1.8],color='red',alpha=.5,lw=2)

    #ax1.plot([0,5],[0,0],color='black')


    ax2.annotate(r'$x$-axis direction',
                 xy=(.65,.4),xycoords='data',textcoords='data',
                 xytext=(.2,1.1),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'$y$-axis direction',
                 xy=(1.4,.0),xycoords='data',textcoords='data',
                 xytext=(1.3,.3),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'$g^*$',
                 xy=(.55,.0),xycoords='data',textcoords='data',
                 xytext=(.4,.07),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Diagonal',
                 xy=(1.6,1.05),xycoords='data',textcoords='data',
                 xytext=(1.6,.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Off-diagonal',
                 xy=(1.4,.63),xycoords='data',textcoords='data',
                 xytext=(1.3,1.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Off-diagonal',
                 alpha=0.,
                 xy=(1.4,1.14),xycoords='data',textcoords='data',
                 xytext=(1.3,1.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Multiple non-axial directions',xy=(3.68,.1),xycoords='data',textcoords='data',xytext=(3.,.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )




    ax1.view_init(20,-8)

    #ax1.plot([.89,.89],[-3,3],color='gray')
    #ax1.plot([3.,3.],[-3,3],color='gray')    

    ax1.set_xlabel(r'$\nu_2$')
    ax1.set_ylabel(r'$g$')
    ax1.set_zlabel(r'$\nu_1$')


    ax2.set_xlabel(r'$g$')
    ax2.set_ylabel(r'$\nu_1$')


    ax1.set_xlim(0,2.)
    ax1.set_ylim(0,2.)
    ax1.set_zlim(-.1,1.8)
    
    plt.show()
    return fig



def wave_exist_2d_trunc(b=.8):
    """
    plot zeros of -nu1 + G(nu1,nu2) and -nu2 + G(nu2,nu1)
    """
    g = np.linspace(.0,2,1000)
    
    L1 = Sqrt(-1 + 1.8*g)
    L2 = Sqrt(-5 + (4 + b)*g + Sqrt(9 + 6*(-4 + b)*g + (4 + b)**2*g**2))/(2.*Sqrt(2))
    L3 = Sqrt(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) - \
              Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + \
                   2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))/2.
    L4 = Sqrt(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) + \
              Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + \
                   2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))/2.


    fig = plt.figure(figsize=(5,3))
    ax1 = fig.add_subplot(111)

    ax1.plot(g,L1,color='black')
    ax1.plot(g,L2,color='black')
    ax1.plot(g,L3,color='black')
    ax1.plot(g,L4,color='black')


    ax1.plot([0,5],[0,0],color='black')

    ax1.annotate(r'$x$-axis direction',
                 xy=(.65,.4),xycoords='data',textcoords='data',
                 xytext=(.2,1.1),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax1.annotate(r'$y$-axis direction',
                 xy=(1.4,.0),xycoords='data',textcoords='data',
                 xytext=(1.3,.3),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax1.annotate(r'$g^*$',
                 xy=(.55,.0),xycoords='data',textcoords='data',
                 xytext=(.4,.07),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax1.annotate('Diagonal',
                 xy=(1.6,1.05),xycoords='data',textcoords='data',
                 xytext=(1.6,.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax1.annotate('Off-diagonal',
                 xy=(1.4,.63),xycoords='data',textcoords='data',
                 xytext=(1.3,1.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax1.annotate('Off-diagonal',
                 alpha=0.,
                 xy=(1.4,1.14),xycoords='data',textcoords='data',
                 xytext=(1.3,1.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax1.annotate('Multiple non-axial directions',xy=(3.68,.1),xycoords='data',textcoords='data',xytext=(3.,.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax1.plot([.89,.89],[-3,3],color='gray')
    #ax1.plot([3.,3.],[-3,3],color='gray')    

    ax1.set_ylabel(r'$\nu_1$')
    ax1.set_xlabel(r'$g$')

    ax1.set_xlim(0,2.)
    ax1.set_ylim(-.1,1.8)
    
    
    return fig



def wave_exist_2d_trunc_v2(b=.8):
    """
    plot zeros of -nu1 + G(nu1,nu2) and -nu2 + G(nu2,nu1)
    as a function of g

    """

    plane1_z = 0.55
    plane2_z = 0.889

    g = np.linspace(0+.0*1j,2+0.*1j,1000)

    # nu1 branches
    L1 = Sqrt(-1 + 1.8*g)
    L2 = Sqrt(-5 + (4 + b)*g + Sqrt(9 + 6*(-4 + b)*g + (4 + b)**2*g**2))/(2.*Sqrt(2))
    L3 = Sqrt(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) - \
              Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + \
                   2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))/2.
    L4 = Sqrt(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) + \
              Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + \
                   2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))/2.
    L5 = 0.*g


    # nu2 branches
    M1 = 0.*g
    M2 = np.real(Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)*(-5 + (4 + b)*g + Sqrt(9 + 6*(-4 + b)*g + (4 + b)**2*g**2))))/(2.*Sqrt(2)*Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)))))
    M3 = np.real(Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)*(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) + Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + 2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))))/(2.*Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)))))
    M4 = np.real(Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)*(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) - Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + 2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))))/(2.*Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)))))
    M5 = Sqrt(-1 + 1.8*g)
    print M2

    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)


    # get plane intersection idx
    g_int_p1 = np.argmin(np.abs(g-plane1_z))
    g_int_p2 = np.argmin(np.abs(g-plane2_z))

    ax1.scatter(L1[g_int_p1],g[g_int_p1],M1[g_int_p1],color='black',s=20)
    ax1.scatter(L1[g_int_p2],g[g_int_p2],M1[g_int_p2],color='black',s=20)

    ax1.scatter(L2[g_int_p1],g[g_int_p1],M2[g_int_p1],color='black',s=20)
    ax1.scatter(L2[g_int_p2],g[g_int_p2],M2[g_int_p2],color='black',s=20)

    ax1.scatter(L3[g_int_p1],g[g_int_p1],M3[g_int_p1],color='black',s=20)
    ax1.scatter(L3[g_int_p2],g[g_int_p2],M3[g_int_p2],color='black',s=20)

    ax1.scatter(L4[g_int_p1],g[g_int_p1],M4[g_int_p1],color='black',s=20)
    ax1.scatter(L4[g_int_p2],g[g_int_p2],M4[g_int_p2],color='black',s=20)

    # plot curves in 3d
    ax1.plot(L1,g,M1,color='black',lw=2)
    ax1.plot(L2,g,M2,color='black',lw=2)
    ax1.plot(L3,g,M3,color='black',lw=2)
    ax1.plot(L4,g,M4,color='black',lw=2)
    ax1.plot(L5,g,M5,color='black',lw=2)

    # plot curves in 2d
    ax2.plot(g,L1,color='black',lw=2)
    ax2.plot(g,L2,color='black',lw=2)
    ax2.plot(g,L3,color='black',lw=2)
    ax2.plot(g,L4,color='black',lw=2)

    # plot bifurcation planes
    X,Y = np.meshgrid(np.linspace(g[0],g[-1],10),np.linspace(g[0],g[-1],10))
    ax1.plot_surface(X,0.*X+plane1_z,Y,alpha=.3,color='gray')
    ax1.plot_surface(X,0.*X+plane2_z,Y,alpha=.3,color='red')


    # plot bifurcation lines
    ax2.plot([plane1_z,plane1_z],[0,1.8],color='black',alpha=.5,lw=2)
    ax2.plot([plane2_z,plane2_z],[0,1.8],color='red',alpha=.5,lw=2)

    #ax1.plot([0,5],[0,0],color='black')


    ax2.annotate(r'$x$-axis direction',
                 xy=(.65,.4),xycoords='data',textcoords='data',
                 xytext=(.2,1.1),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'$y$-axis direction',
                 xy=(1.4,.0),xycoords='data',textcoords='data',
                 xytext=(1.3,.3),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'$g^*$',
                 xy=(.55,.0),xycoords='data',textcoords='data',
                 xytext=(.4,.07),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Diagonal',
                 xy=(1.6,1.05),xycoords='data',textcoords='data',
                 xytext=(1.6,.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Off-diagonal',
                 xy=(1.4,.63),xycoords='data',textcoords='data',
                 xytext=(1.3,1.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Off-diagonal',
                 alpha=0.,
                 xy=(1.4,1.14),xycoords='data',textcoords='data',
                 xytext=(1.3,1.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Multiple non-axial directions',xy=(3.68,.1),xycoords='data',textcoords='data',xytext=(3.,.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )




    ax1.view_init(20,-8)

    #ax1.plot([.89,.89],[-3,3],color='gray')
    #ax1.plot([3.,3.],[-3,3],color='gray')    

    ax1.set_xlabel(r'$\nu_2$')
    ax1.set_ylabel(r'$g$')
    ax1.set_zlabel(r'$\nu_1$')


    ax2.set_xlabel(r'$g$')
    ax2.set_ylabel(r'$\nu_1$')


    ax1.set_xlim(0,2.)
    ax1.set_ylim(0,2.)
    ax1.set_zlim(-.1,1.8)
    
    plt.show()
    return fig

    

def wave_exist_2d_trunc_v3(b=.8):
    """
    plot zeros of -nu1 + G(nu1,nu2) and -nu2 + G(nu2,nu1)
    as a function of g

    """


    plane1_z = 0.53
    plane2_z = 0.88

    g = np.linspace(0+.0*1j,1.5+0.*1j,100)

    # nu1 branches
    L1 = Sqrt(-1 + 1.8*g)
    L2 = Sqrt(-5 + (4 + b)*g + Sqrt(9 + 6*(-4 + b)*g + (4 + b)**2*g**2))/(2.*Sqrt(2))
    L3 = Sqrt(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) - \
              Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + \
                   2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))/2.
    L4 = Sqrt(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) + \
              Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + \
                   2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))/2.
    L5 = 0.*g


    # nu2 branches
    M1 = 0.*g
    M2 = np.real(Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)*(-5 + (4 + b)*g + Sqrt(9 + 6*(-4 + b)*g + (4 + b)**2*g**2))))/(2.*Sqrt(2)*Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)))))
    M3 = np.real(Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)*(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) + Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + 2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))))/(2.*Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)))))
    M4 = np.real(Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)*(-6 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2) - Sqrt(-4*(-5 + Sqrt(16 + (2 + b)**2*g**2)) + 2*g*(-4 + b*(-2 + (2 + b)*g + Sqrt(16 + (2 + b)**2*g**2)))))))/(2.*Sqrt(-(g**3*(2 + g)*(5*(6 + b) + (1 + b)*(8 + 3*b)*g)))))
    M5 = Sqrt(-1 + 1.8*g)


    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)


    # get plane intersection idx
    g_int_p1 = np.argmin(np.abs(g-plane1_z))
    g_int_p2 = np.argmin(np.abs(g-plane2_z))

    # plot curves in 3d

    # prep for plotting with different line widths

    # add modified curves to figure

    ax1.add_collection3d(collect3d_colorgrad(L1,g,M1))
    ax1.add_collection3d(collect3d_colorgrad(L2,g,M2))
    ax1.add_collection3d(collect3d_colorgrad(L3,g,M3))
    ax1.add_collection3d(collect3d_colorgrad(L4,g,M4))
    ax1.add_collection3d(collect3d_colorgrad(L5,g,M5))

    # plot bifurcation planes
    X,Y = np.meshgrid(np.linspace(0,1.2,10),np.linspace(0,1.2,10))
    ax1.plot_surface(X,0.*X+plane1_z,Y,alpha=.2,color='gray')
    ax1.plot_surface(X,0.*X+plane2_z,Y,alpha=.2,color='red')

    # plot intersection points
    #    ax1.scatter(L1[g_int_p1],g[g_int_p1],M1[g_int_p1],color='black',s=30)
    ax1.plot([M3[g_int_p1]],[g[g_int_p1]],[L3[g_int_p1]],color='black',marker='o',markersize=8,zorder=10)
    #ax1.scatter(L1[g_int_p2],g[g_int_p2],M1[g_int_p2],color='red',s=30)
    ax1.plot([np.real(L1[g_int_p2])],[np.real(g[g_int_p2])],[np.real(M1[g_int_p2])],marker='o',markersize=8,markeredgecolor='none',zorder=100,color='red')

    ax1.plot([L2[g_int_p1]],[g[g_int_p1]],[M2[g_int_p1]],color='black',marker='o',markersize=8,zorder=10)
    #ax1.scatter(L2[g_int_p1],g[g_int_p1],M2[g_int_p1],color='black',s=35)
    ax1.plot([L2[g_int_p2]],[g[g_int_p2]],[M2[g_int_p2]],marker='o',markersize=7,markeredgecolor='none',zorder=100,color='red')

    ax1.plot([L3[g_int_p1]],[g[g_int_p1]],[M3[g_int_p1]],color='black',marker='o',markersize=8,zorder=10)
    ax1.scatter(L3[g_int_p1],g[g_int_p1],M3[g_int_p1],color='black',s=30)
    ax1.plot([L3[g_int_p2]],[g[g_int_p2]],[M3[g_int_p2]],marker='o',markersize=6,markeredgecolor='none',zorder=100,color='red')

    #ax1.scatter(L4[g_int_p1],g[g_int_p1],M4[g_int_p1],color='black',s=30)
    #ax1.scatter(L4[g_int_p2],g[g_int_p2],M4[g_int_p2],color='red',s=30)



    # plot curves in 2d + 2d projection in 3d plot
    
    #ax2.plot([L1[g_int_p1],M1[g_int_p1]],color='black',marker='o',markersize=8,zorder=10)
    #ax2.scatter(L1[g_int_p2],M1[g_int_p2],color='red',s=50,zorder=10)
    ax1.plot([L1[g_int_p1]],[1.5],[M1[g_int_p1]],marker='o',markeredgecolor='none',color='black',markersize=8,zorder=100)
    ax1.plot([L1[g_int_p2]],[1.5],[M1[g_int_p2]],marker='o',markeredgecolor='none',color='red',markersize=8,zorder=100)

    ax2.plot([L2[g_int_p1]],[M2[g_int_p1]],color='black',marker='o',markersize=8)
    ax2.scatter(L2[g_int_p2],M2[g_int_p2],color='red',s=70,zorder=10)
    ax1.plot([L2[g_int_p1]],[1.5],[M2[g_int_p1]],marker='o',markeredgecolor='none',color='black',markersize=8,zorder=100)
    ax1.plot([L2[g_int_p2]],[1.5],[M2[g_int_p2]],marker='o',markeredgecolor='none',color='red',markersize=8,zorder=100)

    ax2.scatter(L3[g_int_p1],M3[g_int_p1],color='black',s=70,zorder=10)
    ax2.scatter(L3[g_int_p2],M3[g_int_p2],color='red',s=70,zorder=10)
    ax1.plot([L3[g_int_p1]],[1.5],[M3[g_int_p1]],marker='o',markeredgecolor='none',color='black',markersize=8,zorder=100)
    ax1.plot([L3[g_int_p2]],[1.5],[M3[g_int_p2]],marker='o',markeredgecolor='none',color='red',markersize=8,zorder=100)

    ax2.scatter(L4[g_int_p1],M4[g_int_p1],color='black',s=70,zorder=10)
    ax2.scatter(L4[g_int_p2],M4[g_int_p2],color='red',s=70,zorder=10)
    ax1.plot([L4[g_int_p1]],[1.5],[M4[g_int_p1]],marker='o',markeredgecolor='none',color='black',markersize=8,zorder=100)
    ax1.plot([L4[g_int_p2]],[1.5],[M4[g_int_p2]],marker='o',markeredgecolor='none',color='red',markersize=8,zorder=100) 

    cmap = plt.get_cmap('gray_r')
    my_cmap = truncate_colormap(cmap,.0,.75)
        
    ax2.add_collection(collect(L1,M1,lwstart=3.,lwfactor=4.))
    ax1.add_collection3d(collect(L1,M1,lwstart=3.,lwfactor=4.),zs=1.5,zdir='y')
    #ax2.plot(L1,M1)
    ax2.add_collection(collect(L2,M2,lwstart=3.,lwfactor=4.))
    ax1.add_collection3d(collect(L2,M2,lwstart=3.,lwfactor=4.),zs=1.5,zdir='y')
    #ax2.plot(L2,M2)
    ax2.add_collection(collect(L3,M3,lwstart=3.,lwfactor=4.))
    ax1.add_collection3d(collect(L3,M3,lwstart=3.,lwfactor=4.),zs=1.5,zdir='y')
    #ax2.plot(L3,M3)
    ax2.add_collection(collect(L4,M4,lwstart=3.,lwfactor=4.))
    ax1.add_collection3d(collect(L4,M4,lwstart=3.,lwfactor=4.),zs=1.5,zdir='y')
    #ax2.plot(L4,M4)
    ax2.add_collection(collect(L5,M5,lwstart=3.,lwfactor=4.))
    ax1.add_collection3d(collect(L5,M5,lwstart=3.,lwfactor=4.),zs=1.5,zdir='y')
    #ax2.plot(L5,M5)



    #ax1.plot([0,5],[0,0],color='black')


    ax2.annotate(r'$x$-axis direction',
                 xy=(.65,.01),xycoords='data',textcoords='data',
                 xytext=(.45,.2),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'$y$-axis direction',
                 xy=(.01,.65),xycoords='data',textcoords='data',
                 xytext=(.1,.45),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'$g^*$',
                 xy=(.03,.015),xycoords='data',textcoords='data',
                 xytext=(.2,.07),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    """
    ax2.annotate(r'$g^*$',
                 alpha=0.,
                 xy=(.01,.01),xycoords='data',textcoords='data',
                 xytext=(.4,.07),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )
    """

    ax2.annotate('Diagonal direction',
                 xy=(.68,.7),xycoords='data',textcoords='data',
                 xytext=(.2,.65),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Off-diagonal\\direction',
                 xy=(1.4,.63),xycoords='data',textcoords='data',
                 xytext=(1.3,1.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Off-diagonal',
                 alpha=0.,
                 xy=(1.4,1.14),xycoords='data',textcoords='data',
                 xytext=(1.3,1.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Multiple non-axial directions',xy=(3.68,.1),xycoords='data',textcoords='data',xytext=(3.,.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )






    #ax1.plot([.89,.89],[-3,3],color='gray')
    #ax1.plot([3.,3.],[-3,3],color='gray')    

    ax1.view_init(20,-8)
    ax1.set_xlim(0,1.2)
    ax1.set_ylim(0,1.5)
    ax1.set_zlim(0.,1.2)


    ax1.set_xlabel(r'$\nu_1$')
    ax1.set_ylabel(r'$g$')
    ax1.set_zlabel(r'$\nu_2$')


    ax2.set_xlabel(r'$\nu_1$')
    ax2.set_ylabel(r'$\nu_2$')

    ax2.set_xlim(-.05,1.2)
    ax2.set_ylim(-.05,1.2)

    return fig



def wave_exist_2d_trunc_v4(b=.8):
    """
    plot zeros of -nu1 + G(nu1,nu2) and -nu2 + G(nu2,nu1)
    as a function of g

    """

    # get data
    # nc1 bifurcation values
    #bif = np.loadtxt('twod_wave_trunc_exist_all.dat')
    bif = np.loadtxt('twod_wave_trunc_exist_all_fixed.dat')
    #bif2 = np.loadtxt('twod_wave_exist_br2.dat')

    # get all possible disjoint branches
    val,ty = collect_disjoint_branches(bif,remove_isolated=True,isolated_number=50,redundant_threshold=1e-6,N=5)


    if False:
        mp.figure()
        for key in val.keys():
            mp.plot(val[key][:,1],val[key][:,2],label=key)

        mp.legend()
        mp.show()


    # fix branches to satisfy bounds

    # bound the values
    # .5 <= g <= 1.6
    # 0 <= vi <= .8

    gmin = 0.
    gmax = 1.5
    vimin = 0.
    vimax = 1.3

    # loop over each branch
    # add bounded guys to new dict val_final

    val_final = {}
    ty_final = {}
    for key in val.keys():
        g = val[key][:,0]
        v1 = val[key][:,2]
        v2 = val[key][:,3]


        idx = ((g>=gmin)*(g<=gmax)*
               (v1>=vimin)*(v1<=vimax)*
               (v2>=vimin)*(v2<=vimax))


        if (len(g[idx]) == 0) or\
           (len(v1[idx]) == 0) or\
           (len(v2[idx]) == 0):
            pass
        else:
            #print key,ty[key][0,1]
            val_final[key] = np.zeros((len(g[idx]),3))
            val_final[key][:,0] = g[idx]
            val_final[key][:,1] = v1[idx]
            val_final[key][:,2] = v2[idx]
            ty_final[key] = ty[key]
        
            #print key,ty_final[key]
    #bifx_raw=bif[:,3];bify_raw=bif[:,7]
    #bifx2_raw=bif[:,3];bify2_raw=bif[:,8]

    # use this plot to choose branches
    if False:
        mp.figure()
        for key in val_final.keys():
            mp.plot(val_final[key][:,1],val_final[key][:,2],label=key)
        mp.legend()
        mp.show()
        
    #print 
    #br14
    #br23
    #br48
    #br40


    vbr1=0.
    gbr1=.5555

    vbr2=.7746
    vdiagbr2=.546275
    gbr2=.8889


    plane1_z = 0.5555
    plane2_z = gbr2




    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_axes(MyAxes3D(ax1, 'l'))
    ax2 = fig.add_subplot(122)


    # add modified curves to figure
    for key in val_final.keys():
        g = val_final[key][:,0]
        v1 = val_final[key][:,1]
        v2 = val_final[key][:,2]

        if key == 'br4':
            ax1.add_collection3d(collect3d_colorgrad(v1,g,v2,use_nonan=False,zorder=4,
                                                     lwstart=2,lwend=4,
                                                     cmapmin=.3,cmapmax=.7))

            ax1.add_collection3d(collect3d_colorgrad(v2,g,v1,use_nonan=False,zorder=4,
                                                     lwstart=2,lwend=4,
                                                     cmapmin=.3,cmapmax=.7))


            ax2.add_collection(collect(v1,v2,use_nonan=False,lwstart=2.,lwend=4.,cmapmin=.3,cmapmax=.7,zorder=5))
            ax2.add_collection(collect(v2,v1,use_nonan=False,lwstart=2.,lwend=4.,cmapmin=.3,cmapmax=.7,zorder=5))

            #ax1.add_collection3d(collect(L3,M3,lwstart=3.,lwfactor=4.),zs=1.5,zdir='y')
            ax1.add_collection3d(collect(v1,v2,use_nonan=False,lwstart=2.,lwend=4.,cmapmin=.3,cmapmax=.7,zorder=5),zs=gmax,zdir='y')
            ax1.add_collection3d(collect(v2,v1,use_nonan=False,lwstart=2.,lwend=4.,cmapmin=.3,cmapmax=.7,zorder=5),zs=gmax,zdir='y')
            

        elif key == 'br24' or key == 'br2':
            ax1.add_collection3d(collect3d_colorgrad(v1,g,v2,use_nonan=False,zorder=4,
                                                     lwstart=4,lwend=5,
                                                     cmapmin=.7,cmapmax=1.))
            ax2.add_collection(collect(v1,v2,use_nonan=False,lwstart=4.,lwend=5.,cmapmin=.7,cmapmax=1.,zorder=5))

            ax1.add_collection3d(collect(v1,v2,use_nonan=False,lwstart=4.,lwend=5.,cmapmin=.7,cmapmax=1.,zorder=5),zs=gmax,zdir='y')

            
        elif key == 'br6':
            ax1.add_collection3d(collect3d_colorgrad(v1,g,v2,use_nonan=False,zorder=4,
                                                     lwstart=4,lwend=5,
                                                     cmapmin=.7,cmapmax=1.))

            ax1.add_collection3d(collect3d_colorgrad(v2,g,v1,use_nonan=False,zorder=4,
                                                     lwstart=4,lwend=5,
                                                     cmapmin=.7,cmapmax=1.))

            ax2.add_collection(collect(v1,v2,use_nonan=False,lwstart=4.,lwend=5.,cmapmin=.7,cmapmax=1.,zorder=5))
            ax2.add_collection(collect(v2,v1,use_nonan=False,lwstart=4.,lwend=5.,cmapmin=.7,cmapmax=1.,zorder=5))

            ax1.add_collection3d(collect(v1,v2,use_nonan=False,lwstart=4.,lwend=5.,cmapmin=.7,cmapmax=1.,zorder=5),zs=gmax,zdir='y')
            ax1.add_collection3d(collect(v2,v1,use_nonan=False,lwstart=4.,lwend=5.,cmapmin=.7,cmapmax=1.,zorder=5),zs=gmax,zdir='y')


    # manually add diagonal direction
    bif_data_diag = np.loadtxt('twod_trunc_diag_info.dat')

    g = bif_data_diag[:,0]
    v1 = bif_data_diag[:,1]
    v2 = bif_data_diag[:,2]
    idx2 = ((g>=gmin)*(g<=gmax)*
            (v1>=vimin)*(v1<=vimax)*
            (v2>=vimin)*(v2<=vimax))

    g = g[idx2]
    v1 = v1[idx2]
    v2 = v2[idx2]


    ax1.add_collection3d(collect3d_colorgrad(v1,g,v2,use_nonan=False,zorder=4,
                                             lwstart=2,lwend=5,
                                             cmapmin=.3,cmapmax=1.))

    ax2.add_collection(collect(v1,v2,use_nonan=False,lwstart=2.,lwend=5.,cmapmin=.3,cmapmax=1.,zorder=5))
    ax1.add_collection3d(collect(v1,v2,use_nonan=False,lwstart=2.,lwend=5.,cmapmin=.3,cmapmax=1.,zorder=5),zs=gmax,zdir='y')
        


    # plot beginning zero guy
    g = np.linspace(gmin,.555,10)
    ax1.add_collection3d(collect3d_colorgrad(0.*g,g,0.*g,use_nonan=False,zorder=2,
                                                     lwstart=1,lwend=2,
                                                     cmapmin=.1,cmapmax=.3))

    # plot bifurcation planes
    X,Y = np.meshgrid(np.linspace(0,vimax,10),np.linspace(0,vimax,10))
    Xhalf1,Yhalf1 = np.meshgrid(np.linspace(vbr1,vimax,10),np.linspace(vdiagbr2,vimax,10))
    Xhalf2,Yhalf2 = np.meshgrid(np.linspace(vbr1,vdiagbr2,10),np.linspace(0,vdiagbr2,10))
    Xhalf3,Yhalf3 = np.meshgrid(np.linspace(vdiagbr2,vimax,10),np.linspace(0,vdiagbr2,10))
    

    #ax1.plot_surface(Xhalf1,0.*Xhalf1+plane2_z,Yhalf1,alpha=.6,color='green',lw=0,edgecolor='none',zorder=1)

    ax1.plot_surface(X,0.*X+plane1_z,Y,alpha=.6,color='red',lw=0,edgecolor='none')
    ax1.plot_surface(Xhalf1,0.*X+plane2_z,Yhalf1,alpha=.6,color='green',lw=0,edgecolor='none',zorder=10)
    ax1.plot_surface(Xhalf2,0.*X+plane2_z,Yhalf2,alpha=.6,color='green',lw=0,edgecolor='none',zorder=-5)
    ax1.plot_surface(Xhalf3,0.*X+plane2_z,Yhalf3,alpha=.6,color='green',lw=0,edgecolor='none',zorder=1)
    #ax1.plot_surface(X,0.*X+plane2_z,Y,alpha=.6,color='green')

    # plot intersection points
    #ax1.plot([0.],[1.17],[.51],marker='o',markersize='6',color='red',markeredgecolor='none',zorder=100)
    ax1.plot([0.],[plane1_z],[0],marker='o',color='black',markersize=8,zorder=100,markeredgecolor='none')
    ax1.plot([vbr2],[plane2_z],[0],marker='o',color='red',markersize=8,zorder=100,markeredgecolor='none')
    ax1.plot([0],[plane2_z],[vbr2],marker='o',color='red',markersize=8,zorder=100,markeredgecolor='none')
    ax1.plot([vdiagbr2],[plane2_z],[vdiagbr2],marker='o',color='red',markersize=8,zorder=100,markeredgecolor='none')

    # plot projection
    ax1.plot([0.],[gmax],[0],marker='o',color='black',markersize=8,zorder=100,markeredgecolor='none')
    ax1.plot([vbr2],[gmax],[0],marker='o',color='red',markersize=8,zorder=100,markeredgecolor='none')
    ax1.plot([0],[gmax],[vbr2],marker='o',color='red',markersize=8,zorder=100,markeredgecolor='none')
    ax1.plot([vdiagbr2],[gmax],[vdiagbr2],marker='o',color='red',markersize=8,zorder=100,markeredgecolor='none')




    # plot curves in 2d + 2d projection in 3d plot
    zs = gmax
    

    # plot intersections on 2d
    ax2.scatter([0],[0],marker='o',color='black',s=70,zorder=100)
    ax2.scatter([vbr2],[0],marker='o',color='red',s=70,zorder=100)
    ax2.scatter([0],[vbr2],marker='o',color='red',s=70,zorder=100)
    ax2.scatter([vdiagbr2],[vdiagbr2],marker='o',color='red',s=70,zorder=100)


    #ax1.plot([0,5],[0,0],color='black')

    # g,nu1,nu2 annotation
    ax1.text(.01,gbr1-.15,.05,r'$\bm{g^*}$')
    ax1.text(.01,gbr2+.1,vimax-.1,r'$\bm{g^{**}}$')

    

    # nu1 vs nu2 annotation
    ax2.annotate(r'$\bm{x}$\textbf{-axis direction}',
                 xy=(.65,.01),xycoords='data',textcoords='data',
                 xytext=(.4,.02)
             )


    ax2.annotate(r'$\bm{y}$\textbf{-axis direction}',
                 xy=(.01,.4),xycoords='data',textcoords='data',
                 xytext=(.02,.5),
                 rotation=-90
             )


    ax2.annotate(r'$\bm{g^*}$',
                 xy=(.03,.015),xycoords='data',textcoords='data',
                 xytext=(.2,.07),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    gstarstarpos = (.7,.6)
    ax2.annotate(r'$\bm{g^{**}}$',
                 xy=(vbr2,.0),xycoords='data',textcoords='data',
                 xytext=gstarstarpos,
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
                 zorder=10
             )
    ax2.annotate(r'$\bm{g^{**}}$',
                 xy=(0,vbr2),xycoords='data',textcoords='data',
                 xytext=gstarstarpos,
                 alpha=0.,
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
                 zorder=10
             )
    ax2.annotate(r'$\bm{g^{**}}$',
                 xy=(vdiagbr2,vdiagbr2),xycoords='data',textcoords='data',
                 xytext=gstarstarpos,
                 alpha=0.,
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
                 zorder=10
             )



    """
    ax2.annotate(r'$g^*$',
                 alpha=0.,
                 xy=(.01,.01),xycoords='data',textcoords='data',
                 xytext=(.4,.07),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )
    """

    ax2.annotate(r'\textbf{Diagonal}',
                 xy=(.73,.7),xycoords='data',textcoords='data',
                 xytext=(.65,.85),
                 rotation=45
             )

    ax2.annotate(r'\textbf{Off-diagonal}',
                 xy=(1.2,.7),xycoords='data',textcoords='data',
                 xytext=(1.,1.),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate(r'\textbf{Off-diagonal}',
                 alpha=0.,
                 xy=(.7,1.2),xycoords='data',textcoords='data',
                 xytext=(1.,1.0),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate(r'\textbf{Multiple non-axial directions}',xy=(3.68,.1),xycoords='data',textcoords='data',xytext=(3.,.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )






    #ax1.plot([.89,.89],[-3,3],color='gray')
    #ax1.plot([3.,3.],[-3,3],color='gray')    

    ax1.view_init(20,-8)

    """
    tmp_planes = ax1.zaxis._PLANES 
    ax1.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                         tmp_planes[0], tmp_planes[1], 
                         tmp_planes[4], tmp_planes[5])
    """

    ax1.set_xlim(vimin,vimax)
    ax1.set_ylim(gmin,gmax)
    ax1.set_zlim(vimin,vimax)

    ax1.set_xlabel(r'$\\\bm{\nu_1}$',fontsize=15)
    ax1.set_ylabel(r'$\bm{g}$',fontsize=15)
    ax1.set_zlabel(r'$\\\bm{\nu_2}$',fontsize=15)


    ax2.set_xlabel(r'$\bm{\nu_1}$',fontsize=15)
    ax2.set_ylabel(r'$\bm{\nu_2}$',fontsize=15)


    ax2.set_xlim(-.05+vimin,vimax)
    ax2.set_ylim(-.05+vimin,vimax)


    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.zaxis.set_major_formatter(formatter)


    ax2.xaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)


    return fig

    



def wave_exist_2d_full_v2(b=.8):
    """
    plot zeros of -nu1 + G(nu1,nu2) and -nu2 + G(nu2,nu1)
    as a function of g
    
    use accurate fourier series

    """


    # get data
    # nc1 bifurcation values
    bif = np.loadtxt('twod_wave_exist_br1.dat')
    #bif2 = np.loadtxt('twod_wave_exist_br2.dat')

    bif_diag1 = np.loadtxt('twod_wave_exist_diag1.dat')
    bif_diag2 = np.loadtxt('twod_wave_exist_diag2.dat')


    # clean
    bifx,bify = clean(bif[:,3],bif[:,7],tol=.47)
    bifx2,bify2 = clean(bif[:,3],bif[:,8],tol=.47)

    bif_diag1x,bif_diag1y = clean(bif_diag1[:,0],np.abs(bif_diag1[:,1]),tol=.2)
    bif_diag2x,bif_diag2y = clean(bif_diag2[:,0],np.abs(bif_diag2[:,1]),tol=.2)


    # remove nans for calculating minima (usually nans are taken to be max/min vals, which is bad)
    bifx_nonan = bifx[(~np.isnan(bifx))*(~np.isnan(bify))]
    bify_nonan = bify[(~np.isnan(bifx))*(~np.isnan(bify))]

    bifx2_nonan = bifx2[(~np.isnan(bifx2))*(~np.isnan(bify2))]
    bify2_nonan = bify2[(~np.isnan(bifx2))*(~np.isnan(bify2))]

    bif_diag1x_nonan = bif_diag1x[(~np.isnan(bif_diag1x))*(~np.isnan(bif_diag1y))]
    bif_diag1y_nonan = bif_diag1y[(~np.isnan(bif_diag1x))*(~np.isnan(bif_diag1y))]

    bif_diag2x_nonan = bif_diag2x[(~np.isnan(bif_diag2x))*(~np.isnan(bif_diag2y))]
    bif_diag2y_nonan = bif_diag2y[(~np.isnan(bif_diag2x))*(~np.isnan(bif_diag2y))]


    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    plane1_z = .895
    plane2_z = 1.17

    # get plane intersection idx
    bifx_int_p1 = np.argmin(np.abs(bifx_nonan-plane1_z))
    bifx_int_p2 = np.argmin(np.abs(bifx_nonan-plane2_z))
    bifx2_int_p1 = np.argmin(np.abs(bifx2_nonan-plane1_z))
    bifx2_int_p2 = np.argmin(np.abs(bifx2_nonan-plane2_z))

    bif_diagx_int_p1 = np.argmin(np.abs(bif_diag1x_nonan-plane1_z))
    bif_diagx_int_p2 = np.argmin(np.abs(bif_diag1x_nonan-plane2_z))
    bif_diagx2_int_p1 = np.argmin(np.abs(bif_diag2x_nonan-plane1_z))
    bif_diagx2_int_p2 = np.argmin(np.abs(bif_diag2x_nonan-plane2_z))



    ## plot curves in 3d
    

    # plot off diagonal and axial curves
    v1a = bify2[(bify>=0)*(bify2>=0)*(bify<=1)*(bify2<=1)*(bifx<=2)]
    v2a = bify[(bify>=0)*(bify2>=0)*(bify<=1)*(bify2<=1)*(bifx<=2)]
    ga = bifx[(bify>=0)*(bify2>=0)*(bify<=1)*(bify2<=1)*(bifx<=2)]


    #v1b = bif_diag1y[(bif_diag1y>=0)*(bif_diag2y>=0)*(bif_diag1y<=1)*(bif_diag2y<=1)*(bif_diag1x<=2)]
    #v2b = bif_diag1y[(bif_diag1y>=0)*(bif_diag2y>=0)*(bif_diag1y<=1)*(bif_diag2y<=1)*(bif_diag1x<=2)]
    gb = np.linspace(np.amin(bif_diag1x[~np.isnan(bif_diag1x)]),np.amax(bif_diag1x[~np.isnan(bif_diag1x)]),20)

    
    # clean
    ga,v1a,v2a = clean3d(ga,v1a,v2a,tol=.47)

    # remove nans for linewidth stuff later.
    ga_nonan = ga[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    v1a_nonan = v1a[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]
    v2a_nonan = v2a[~np.isnan(ga)*(~np.isnan(v1a))*(~np.isnan(v2a))]


    # prep for plotting with different line widths
    sol = np.zeros((len(ga),3))
    sol[:,0] = v1a
    sol[:,1] = ga
    sol[:,2] = v2a

    sol = np.transpose(sol)

    points = np.array([sol[0,:],sol[1,:],sol[2,:]]).T.reshape(-1,1,3)
    segs = np.concatenate([points[:-1],points[1:]],axis = 1)
    line3d = Line3DCollection(segs,linewidths=(1.+(v1a_nonan)/np.amax(v1a_nonan)*3.),colors='k')

    # add modified curves to figure
    ax1.add_collection3d(line3d)

    # repleat above to capture remaining axial branch(es)
    # prep for plotting with different line widths
    sol = np.zeros((len(ga),3))
    sol[:,0] = v2a
    sol[:,1] = ga
    sol[:,2] = v1a

    sol = np.transpose(sol)

    points = np.array([sol[0,:],sol[1,:],sol[2,:]]).T.reshape(-1,1,3)
    segs = np.concatenate([points[:-1],points[1:]],axis = 1)
    line3d = Line3DCollection(segs,linewidths=(1.+(v2a_nonan)/np.amax(v2a_nonan)*3.),colors='k')

    # add modified curves to figure
    ax1.add_collection3d(line3d)
    

    # plot diagonal guys

    # prep for plotting with different line widths
    diagx = bif_diag2y[(bif_diag2y<=1)*(bif_diag2x<=2.)]
    diagy = bif_diag2x[(bif_diag2y<=1)*(bif_diag2x<=2.)]
    diagz = bif_diag2y[(bif_diag2y<=1)*(bif_diag2x<=2.)]

    diagx_nonan = diagx[~np.isnan(diagx)]

    sol = np.zeros((len(diagx),3))
    sol[:,0] = diagx
    sol[:,1] = diagy
    sol[:,2] = diagz

    sol = np.transpose(sol)

    points2 = np.array([sol[0,:],sol[1,:],sol[2,:]]).T.reshape(-1,1,3)
    segs2 = np.concatenate([points2[:-1],points2[1:]],axis = 1)
    line3d2 = Line3DCollection(segs2,linewidths=(1.+(diagx_nonan)/np.amax(diagx_nonan)*3.),colors='k')

    ax1.add_collection3d(line3d2)

    # plot zero solution
    ax1.plot([.0,0],[.5,plane1_z],[.0,0],color='black',lw=1)



    # plot bifurcation planes
    X,Y = np.meshgrid(np.linspace(0,1,10),np.linspace(0,1,10))
    ax1.plot_surface(X,0.*X+plane1_z,Y,alpha=.5,color='gray')
    ax1.plot_surface(X,0.*X+plane2_z,Y,alpha=.5,color='red')


    # plot plane intersections

    ax1.scatter(bify[bifx_int_p1],bifx[bifx_int_p1],bify2[bifx_int_p1],color='black',s=20)
    #ax1.scatter(bify[bifx_int_p2],bifx[bifx_int_p2],bify2[bifx_int_p2],color='black',s=20)


    #ax1.scatter(bif_diag2y_nonan[bif_diagx_int_p2],bif_diag1x_nonan[bif_diagx_int_p2],bif_diag1y_nonan[bif_diagx_int_p2],color='black',s=20)

    ax1.scatter(0,1.17,.51,color='red',s=20,zorder=10)
    ax1.scatter(.5,1.17,0.,color='red',s=40,zorder=10)
    ax1.scatter(.37,1.17,.37,color='red',s=50,zorder=10)





    """
    ax1.scatter(L1[g_int_p2],g[g_int_p2],M1[g_int_p2],color='black',s=20)

    ax1.scatter(L2[g_int_p1],g[g_int_p1],M2[g_int_p1],color='black',s=20)
    ax1.scatter(L2[g_int_p2],g[g_int_p2],M2[g_int_p2],color='black',s=20)

    ax1.scatter(L3[g_int_p1],g[g_int_p1],M3[g_int_p1],color='black',s=20)
    ax1.scatter(L3[g_int_p2],g[g_int_p2],M3[g_int_p2],color='black',s=20)

    ax1.scatter(L4[g_int_p1],g[g_int_p1],M4[g_int_p1],color='black',s=20)
    ax1.scatter(L4[g_int_p2],g[g_int_p2],M4[g_int_p2],color='black',s=20)
    """



    ## plot curves in 2d

    # bifurcation lines
    ax2.plot([plane1_z,plane1_z],[-1,1.8],color='black',alpha=.5,lw=2)
    ax2.plot([plane2_z,plane2_z],[-1,1.8],color='red',alpha=.5,lw=2)

    ax2.plot(bifx,bify,color='black')
    ax2.plot(bifx2,bify2,color='black')

    ax2.plot(bif_diag1x,bif_diag1y,color='black')
    ax2.plot(bif_diag2x,bif_diag2y,color='black')

    ax2.plot([0,5],[0,0],color='black')


    # label curves
    ax2.annotate(r'$x$-axis direction',
                 xy=(1.04,.37),xycoords='data',textcoords='data',
                 xytext=(.6,.6),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'$y$-axis direction',
                 xy=(1.0,.0),xycoords='data',textcoords='data',
                 xytext=(.55,.33),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'$g^*$',
                 xy=(.9,.0),xycoords='data',textcoords='data',
                 xytext=(.8,.05),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Diagonal',
                 xy=(1.1,.32),xycoords='data',textcoords='data',
                 xytext=(1.4,.2),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Off-diagonal',
                 xy=(1.4,.41),xycoords='data',textcoords='data',
                 xytext=(1.5,.34),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Off-diagonal',
                 alpha=0.,
                 xy=(1.4,.62),xycoords='data',textcoords='data',
                 xytext=(1.5,.34),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )



    # plot params
    ax1.view_init(20,-8)


    # set labels 

    ax1.set_xlabel(r'$\nu_2$')
    ax2.set_xlabel(r'$g$')

    ax1.set_ylabel(r'$g$')
    ax2.set_ylabel(r'$\nu_1$')

    ax1.set_zlabel(r'$\nu_1$')

    ax1.set_xlim(0.,1.)
    ax2.set_xlim(.5,2.)

    ax1.set_ylim(.5,2.)
    ax2.set_ylim(-.05,1.)

    ax1.set_zlim(0.,1.)


    #plt.show()
    return fig




    



def wave_exist_2d_full_v2_testing(b=.8):
    """
    testing to figure out how to implement lw changes in 1 plot.
    """

    
    #from matplotlib.collections import LineCollection

    from matplotlib.collections import PolyCollection
    from matplotlib import colors as mcolors

    fig = plt.figure()
    ax = fig.gca(projection='3d')


    xcoord = np.linspace(0,pi,100)
    ycoord = np.linspace(0,pi,100)
    xs = cos(xcoord)+6.
    ys = 3*sin(ycoord)+1.5
    zs = .25+0.*np.sqrt(xcoord**2.+ycoord**2.)
    

    sol = np.zeros((len(xs),3))
    sol[:,0] = xs
    sol[:,1] = ys
    sol[:,2] = zs

    sol = np.transpose(sol)

    points = np.array([sol[0,:],sol[1,:],sol[2,:]]).T.reshape(-1,1,3)


    segs = np.concatenate([points[:-1],points[1:]],axis = 1)
    line3d = Line3DCollection(segs,linewidths=ys)

    line3d.set_alpha(0.7)
    ax.add_collection3d(line3d)#, zs=zs)
    
    ax.set_xlabel('X')
    ax.set_xlim3d(0, 10)
    ax.set_ylabel('Y')
    ax.set_ylim3d(-1, 4)
    ax.set_zlabel('Z')
    ax.set_zlim3d(0, 1)
    
    plt.show()
    


    """
    from matplotlib.collections import LineCollection
    x=np.linspace(0,4*pi,10000)
    y=cos(x)
    lwidths=1+x[:-1]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, linewidths=lwidths,color='blue')
    fig,a = plt.subplots()
    a.add_collection(lc)
    a.set_xlim(0,4*pi)
    a.set_ylim(-1.1,1.1)
    fig.show()
    """

    return fig

    
    
    

def wave_exist_2d_full_v3(b=.8):
    """
    plot zeros of -nu1 + G(nu1,nu2) and -nu2 + G(nu2,nu1)
    as a function of g.

    g is shown implicitly as color/thickness.
    scatter dots included at countour lines to give an additional sense of depth
    
    use accurate fourier series

    """

    

    # get data
    # nc1 bifurcation values
    bif = np.loadtxt('twod_wave_exist_br1.dat')
    #bif2 = np.loadtxt('twod_wave_exist_br2.dat')

    bif_diag1 = np.loadtxt('twod_wave_exist_diag1.dat')
    bif_diag2 = np.loadtxt('twod_wave_exist_diag2.dat')

    # bound the values
    # .5 <= g <= 1.6
    # 0 <= vi <= .8

    gmin = .5
    gmax = 1.6
    vimin = 0.
    vimax = 1.



    bifx_raw=bif[:,3];bify_raw=bif[:,7]
    bifx2_raw=bif[:,3];bify2_raw=bif[:,8]

    bif_diagx_raw=bif_diag2[:,0];bif_diagy_raw=np.abs(bif_diag2[:,1])

    # get true/false arrays for entries satisfying the bounds
    bnd_idx_bool = ((bifx_raw>=gmin)*(bifx_raw<=gmax)*
                    (bify_raw>=vimin)*(bify_raw<=vimax)*
                    (bify2_raw>=vimin)*(bify2_raw<=vimax))

    print bifx2_raw[bnd_idx_bool]
    print bify2_raw[bnd_idx_bool]

    # get actual indices
    bnd_idx = np.arange(0,len(bnd_idx_bool),1)[bnd_idx_bool]
    
    # extract only 1 copy of a branch
    diff = 0
    i = 1

    final_bnd_idx = []
    """
    while diff <= 10:
        # as long as the next index is no more than 10 units, append.
        diff = np.abs(bnd_idx[i] - bnd_idx[i-1])
        final_bnd_idx.append(bnd_idx[i-1])
        i += 1
    """

    #final_bnd_idx = np.array(final_bnd_idx,dtype=int) # convert back to np array
    final_bnd_idx = bnd_idx_bool
    
    bifx_bndd = bifx_raw[final_bnd_idx]
    bify_bndd = bify_raw[final_bnd_idx]
    bifx2_bndd = bifx2_raw[final_bnd_idx]
    bify2_bndd = bify2_raw[final_bnd_idx]


    bif_diagx_bndd = bif_diagx_raw[(bif_diagx_raw>=gmin)*(bif_diagx_raw<=gmax)*
                                   (bif_diagy_raw>=vimin)*(bif_diagy_raw<=vimax)]
    bif_diagy_bndd = bif_diagy_raw[(bif_diagx_raw>=gmin)*(bif_diagx_raw<=gmax)*
                                   (bif_diagy_raw>=vimin)*(bif_diagy_raw<=vimax)]

    # clean
    bifx,bify = clean(bifx_bndd,bify_bndd,tol=.3)
    bifx2,bify2 = clean(bifx2_bndd,bify2_bndd,tol=.3)
    bif_diagx,bif_diagy = clean(bif_diagx_bndd,bif_diagy_bndd,tol=5)
    

    # clean
    #bifx,bify = clean(bif[:,3],bif[:,7],tol=.47)
    #bifx2,bify2 = clean(bif[:,3],bif[:,8],tol=.47)

    #bif_diag1x,bif_diag1y = clean(bif_diag1[:,0],np.abs(bif_diag1[:,1]),tol=.2)
    #bif_diag2x,bif_diag2y = clean(bif_diag2[:,0],np.abs(bif_diag2[:,1]),tol=.2)


    # create equivalent arrays without nans for calculating minima (usually nans are taken to be max/min vals, which is bad)
    bifx_nonan = bifx[(~np.isnan(bifx))*(~np.isnan(bify))]
    bify_nonan = bify[(~np.isnan(bifx))*(~np.isnan(bify))]

    bifx2_nonan = bifx2[(~np.isnan(bifx2))*(~np.isnan(bify2))]
    bify2_nonan = bify2[(~np.isnan(bifx2))*(~np.isnan(bify2))]

    bif_diagx_nonan = bif_diagx[(~np.isnan(bif_diagx))*(~np.isnan(bif_diagy))]
    bif_diagy_nonan = bif_diagy[(~np.isnan(bif_diagx))*(~np.isnan(bif_diagy))]


    plane1_z = .895
    plane2_z = 1.17

    # get plane intersection idx
    bifx_int_p1 = np.argmin(np.abs(bifx_nonan-plane1_z))
    bifx_int_p2 = np.argmin(np.abs(bifx_nonan-plane2_z))
    bifx2_int_p1 = np.argmin(np.abs(bifx2_nonan-plane1_z))
    bifx2_int_p2 = np.argmin(np.abs(bifx2_nonan-plane2_z))

    bif_diagx_int_p1 = np.argmin(np.abs(bif_diagx_nonan-plane1_z))
    bif_diagx_int_p2 = np.argmin(np.abs(bif_diagx_nonan-plane2_z))


    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(121,projection='3d')
    ax2 = fig.add_subplot(122)

    # prep for plotting with different line widths
    diagx = bif_diagy
    diagy = bif_diagx
    diagz = bif_diagy




    
    ## plot curves in 3d



    # plot off diagonal and axial curves
    
    # clean for 3d plot
    ga,v1a,v2a = clean3d(bifx,bify2,bify,tol=.391)

    # add modified curves to figure (non diagonal guys)
    #ax1.add_collection3d(collect3d_colorgrad(v1a,ga,v2a,use_nonan=False,zorder=2,lwstart=3,lwend=6,cmapmin=.2,cmapmax=1.))
    ax1.add_collection3d(collect3d_colorgrad(v2a,ga,v1a,use_nonan=False,zorder=2,
                                             cmapmin=.2,cmapmax=1.,
                                             lwstart=2,lwend=6.))
    ax1.add_collection3d(collect3d_colorgrad(v1a,ga,v2a,use_nonan=False,zorder=2,
                                             cmapmin=.2,cmapmax=1.,
                                             lwstart=2,lwend=6.))




    # plot diagonal guys
    ax1.add_collection3d(collect3d_colorgrad(diagx,diagy,diagz,use_nonan=False,zorder=2,
                                             cmapmin=.2,cmapmax=1.,
                                             lwstart=6.,lwend=2.))

    # plot hacky shit to fix clipping/zorder issue
    #ax1.plot([.55],[1.458],[.55],marker='s',color='#ffae6e',markeredgecolor='#ffae6e',zorder=10,markersize=5)
    #ax1.plot([.565],[1.46],[.565],marker='s',color='#ffae6e',markeredgecolor='#ffae6e',zorder=10,markersize=5)
    #ax1.plot([.32],[1.11],[.32],marker='s',color='#b77449',markeredgecolor='#b77449',zorder=10,markersize=3)
    
    print 'diagx,diagy,diagz',diagx,diagy,diagz

    # plot zero solution
    gt = np.linspace(.5,.9,10)
    ax1.add_collection3d(collect3d_colorgrad(0.*gt,gt,0.*gt,zorder=10,cmapmax=.4,lwend=2.))
    #ax1.plot([.0,0],[.5,plane1_z],[.0,0],color='black',lw=1)



    # plot bifurcation planes
    X,Y = np.meshgrid(np.linspace(0,.8,2),np.linspace(0,.8,2))
    ax1.plot_surface(X,0.*X+plane1_z,Y,alpha=.2,color='gray')
    ax1.plot_surface(X,0.*X+plane2_z,Y,alpha=.2,color='red')



    # plot plane intersections

    ax1.plot([bify[bifx_int_p1]],[bifx[bifx_int_p1]],[bify2[bifx_int_p1]],color='black',marker='o',markersize='8',zorder=100)
    #ax1.scatter(bify[bifx_int_p2],bifx[bifx_int_p2],bify2[bifx_int_p2],color='black',s=20)
    ax1.plot([0.],[1.17],[.51],marker='o',markersize='6',color='red',markeredgecolor='none',zorder=100)
    ax1.plot([.51],[1.17],[0.],marker='o',markersize='8',color='red',markeredgecolor='none',zorder=100)
    ax1.plot([.38],[1.17],[.38],marker='o',markersize='7',color='red',markeredgecolor='none',zorder=100)


    # plot projection of plane intersections
    ax1.plot([0.],[1.6],[.51],marker='o',markersize=8,color='red',markeredgecolor='none',zorder=5)
    ax1.plot([.51],[1.6],[0.],marker='o',markersize=8,color='red',markeredgecolor='none',zorder=100)
    ax1.plot([.38],[1.6],[.38],marker='o',markersize=8,color='red',markeredgecolor='none',zorder=2)
    ax1.plot([0],[1.6],[0],marker='o',markersize=8,color='black',markeredgecolor='none',zorder=2)


    ## plot curves in 2d
    zs = 1.6
    
    # axial guys
    ax2.add_collection(collect(bify,bify2,use_nonan=False,lwstart=3.,lwend=6.,cmapmin=.2,cmapmax=1.))
    ax2.add_collection(collect(bify2,bify,use_nonan=False,lwstart=3.,lwend=6.,cmapmin=.2,cmapmax=1.))
    ax1.add_collection3d(collect(bify,bify2,use_nonan=False,lwstart=3.,lwend=6.,cmapmin=.2,cmapmax=1.),zs=zs,zdir='y')
    ax1.add_collection3d(collect(bify2,bify,use_nonan=False,lwstart=3.,lwend=6.,cmapmin=.2,cmapmax=1.),zs=zs,zdir='y')

    # diagonal
    ax2.add_collection(collect(diagx,diagz,lwstart=3.,lwend=6,cmapmin=.2,cmapmax=1.))
    ax1.add_collection3d(collect(diagx,diagz,lwstart=3.,lwend=6,cmapmin=.2,cmapmax=1.),zs=zs,zdir='y')


    # bifurcation points lines
    ax2.scatter(0,.52,s=70,color='red',zorder=10) # axial intersection (y-axis)
    ax2.scatter(.52,0.,s=70,color='red',zorder=10) # axial intersection (x-axis)
    ax2.scatter(.38,.38,s=70,color='red',zorder=10) # diagonal intersection
    ax2.scatter(0.,0.,s=70,color='black',zorder=10) # diagonal intersection



    # label curves

    ax2.annotate(r'$x$-axis direction',
                 xy=(.6,.01),xycoords='data',textcoords='data',
                 xytext=(.6,.1),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'$y$-axis direction',
                 xy=(.01,.6),xycoords='data',textcoords='data',
                 xytext=(.1,.7),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'$g^*$',
                 xy=(.03,.015),xycoords='data',textcoords='data',
                 xytext=(.15,.05),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Diagonal',
                 xy=(1.1,.32),xycoords='data',textcoords='data',
                 xytext=(1.4,.2),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Off-diagonal',
                 xy=(1.4,.41),xycoords='data',textcoords='data',
                 xytext=(1.5,.34),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate('Off-diagonal',
                 alpha=0.,
                 xy=(1.4,.62),xycoords='data',textcoords='data',
                 xytext=(1.5,.34),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )






    ax1.view_init(20,-8)

    # set labels 

    ax1.set_xlabel(r'$\nu_1$')
    ax1.set_ylabel(r'$g$')
    ax1.set_zlabel(r'$\nu_2$')

    ax2.set_xlabel(r'$\nu_1$')
    ax2.set_ylabel(r'$\nu_2$')






    ax1.set_xlim(0.,.8)
    ax1.set_ylim(.5,1.6)
    ax1.set_zlim(0,.8)

    ax2.set_xlim(-.05,.8)
    ax2.set_ylim(-.05,.8)

    # plot params
    #ax1.view_init(20,-8)


    #plt.show()



    return fig


def truncate_branches(val,ty,gmin,gmax,vimin,vimax):
    val_final = {}
    ty_final = {}

    for key in val.keys():
        g = val[key][:,0]
        v1 = val[key][:,2]
        v2 = val[key][:,3]

        idx = ((g>=gmin)*(g<=gmax)*
               (v1>=vimin)*(v1<=vimax)*
               (v2>=vimin)*(v2<=vimax))


        if (len(g[idx]) == 0) or\
           (len(v1[idx]) == 0) or\
           (len(v2[idx]) == 0):
            pass
        else:
            print key,ty[key][0,1]
            val_final[key] = np.zeros((len(g[idx]),3))
            val_final[key][:,0] = g[idx]
            val_final[key][:,1] = v1[idx]
            val_final[key][:,2] = v2[idx]
            ty_final[key] = ty[key]
        

    return val_final,ty_final

def wave_exist_2d_full_v4(b=.8):

    # get data
    bif = np.loadtxt('twod_wave_exist_v2.dat')
    #bif2 = np.loadtxt('twod_wave_exist_br2.dat')

    #bif_diag1 = np.loadtxt('twod_wave_exist_diag1.dat')
    bif_diag2 = np.loadtxt('twod_wave_exist_diag_v2.dat')

    # get all possible disjoint branches
    val,ty = collect_disjoint_branches(bif,remove_isolated=True,isolated_number=3,remove_redundant=False,N=10)
    val_di,ty_di = collect_disjoint_branches(bif_diag2,remove_isolated=True,isolated_number=3,remove_redundant=False,N=10)



    plane1_z = .895
    plane2_z = 1.16


    if False:
        mp.figure()
        for key in val.keys():
            mp.plot(val[key][:,1],val[key][:,2],label=key)

        mp.legend()
        mp.show()


    # fix branches to satisfy bounds

    # bound the values
    # .5 <= g <= 1.6
    # 0 <= vi <= .8

    gmin = .7
    gmax = 1.6
    vimin = 0.
    vimax = .85


    val_final,ty_final = truncate_branches(val,ty,gmin,gmax,vimin,vimax)
    val_di_final,ty_di_final = truncate_branches(val_di,ty_di,gmin,gmax,vimin,vimax)
    


    # use this plot to choose branches
    if False:
        mp.figure()
        for key in val_final.keys():
            mp.plot(val_final[key][:,1],val_final[key][:,2],label=key)
        mp.legend()
        mp.show()


    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_axes(MyAxes3D(ax1, 'l'))
    ax2 = fig.add_subplot(122)


    # add modified curves to figure
    for key in val_final.keys():
        g = val_final[key][:,0]
        v1 = val_final[key][:,1]
        v2 = val_final[key][:,2]



        if key == 'br13' or key == 'br6':
            ax1.add_collection3d(collect3d_colorgrad(v1,g,v2,use_nonan=False,zorder=2,
                                                     lwstart=2,lwend=4,
                                                     cmapmin=.3,cmapmax=.7))


        elif key == 'br26' or key == 'br14' or key == 'br2' or key == 'br8':
            ax1.add_collection3d(collect3d_colorgrad(v1,g,v2,use_nonan=False,zorder=2,
                                                     lwstart=4,lwend=5,
                                                     cmapmin=.7,cmapmax=1.))


    for key in val_di_final.keys():
        g = val_di_final[key][:,0]
        v1 = val_di_final[key][:,1]
        v2 = val_di_final[key][:,2]

        ax1.add_collection3d(collect3d_colorgrad(v1,g,v2,use_nonan=False,zorder=2,
                                                 lwstart=2,lwend=5,
                                                 cmapmin=.3,cmapmax=1.))


    # plot beginning zero guy
    g = np.linspace(gmin,plane1_z,10)
    ax1.add_collection3d(collect3d_colorgrad(0.*g,g,0.*g,use_nonan=False,zorder=2,
                                                     lwstart=1,lwend=2,
                                                     cmapmin=.1,cmapmax=.3))

    # plot bifurcation planes
    X,Y = np.meshgrid(np.linspace(0,vimax,10),np.linspace(0,vimax,10))
    Xhalf1,Yhalf1 = np.meshgrid(np.linspace(0.,.5,10),np.linspace(0,.5,20))
    Xhalf2,Yhalf2 = np.meshgrid(np.linspace(.5,vimax,10),np.linspace(0,vimax,20))

    Xhalf1b,Yhalf1b = np.meshgrid(np.linspace(.0,.5,10),np.linspace(.5,vimax,20))
    #Xhalf2b,Yhalf2b = np.meshgrid(np.linspace(.5,vimax,10),np.linspace(0,vimax,20))



    ax1.plot_surface(X,0.*X+plane1_z,Y,alpha=.5,color='red',edgecolor='none')

    ax1.plot_surface(Xhalf1,0.*Xhalf1+plane2_z,Yhalf1,alpha=.6,color='green',lw=0,edgecolor='none',zorder=1)
    ax1.plot_surface(Xhalf2,0.*Xhalf2+plane2_z,Yhalf2,alpha=.6,color='green',lw=0,edgecolor='none',zorder=3)
    ax1.plot_surface(Xhalf1b,0.*Xhalf1b+plane2_z,Yhalf1b,alpha=.6,color='green',lw=0,edgecolor='none',zorder=3)

    #ax1.plot_surface(X2,0.*X2+plane2_z,Y2,alpha=.5,color='red',edgecolor='none')
    #ax1.plot_surface(X[X>3],0.*X[X>3]+plane2_z,Y[X>3],alpha=.5,color='red',edgecolor='none')

    # plot intersection points
    #ax1.plot([bify[bifx_int_p1]],[bifx[bifx_int_p1]],[bify2[bifx_int_p1]],color='black',marker='o',markersize='8',zorder=100)
    #ax1.scatter(bify[bifx_int_p2],bifx[bifx_int_p2],bify2[bifx_int_p2],color='black',s=20)
    ax1.plot([0.],[plane1_z],[.0],marker='o',markersize='8',color='black',markeredgecolor='none',zorder=100)
    ax1.plot([0.],[1.17],[.51],marker='o',markersize='8',color='red',markeredgecolor='none',zorder=100)
    ax1.plot([.51],[1.17],[0.],marker='o',markersize='8',color='red',markeredgecolor='none',zorder=100)
    ax1.plot([.38],[1.17],[.38],marker='o',markersize='8',color='red',markeredgecolor='none',zorder=100)


    # plot projection of plane intersections
    ax1.plot([0.],[1.6],[.51],marker='o',markersize=8,color='red',markeredgecolor='none',zorder=5)
    ax1.plot([.51],[1.6],[0.],marker='o',markersize=8,color='red',markeredgecolor='none',zorder=5)
    ax1.plot([.38],[1.6],[.38],marker='o',markersize=8,color='red',markeredgecolor='none',zorder=5)
    ax1.plot([0],[1.6],[0],marker='o',markersize=8,color='black',markeredgecolor='none',zorder=2)


    # plot curves in 2d + 2d projection in 3d plot
    zs = gmax
    
    for key in val_final.keys():
        g = val_final[key][:,0]
        v1 = val_final[key][:,1]
        v2 = val_final[key][:,2]


        if key == 'br13' or key == 'br6':
            ax2.add_collection(collect(v1,v2,use_nonan=False,zorder=3,
                                                   lwstart=2,lwend=4,
                                                   cmapmin=.3,cmapmax=1.))
            ax1.add_collection3d(collect(v1,v2,use_nonan=False,zorder=2,
                                                   lwstart=2,lwend=4,
                                                   cmapmin=.3,cmapmax=1.),zs=zs,zdir='y')
            

        elif key == 'br26' or key == 'br14' or key == 'br2' or key == 'br8':
            ax2.add_collection(collect(v1,v2,use_nonan=False,zorder=3,
                                       lwstart=4,lwend=5,
                                       cmapmin=.6,cmapmax=1.))
            ax1.add_collection3d(collect(v1,v2,use_nonan=False,zorder=2,
                                       lwstart=4,lwend=5,
                                       cmapmin=.6,cmapmax=1.),zs=zs,zdir='y')

    # bifurcation points
    ax2.scatter(0,.52,s=70,color='red',zorder=10) # axial intersection (y-axis)
    ax2.scatter(.52,0.,s=70,color='red',zorder=10) # axial intersection (x-axis)
    ax2.scatter(.38,.38,s=70,color='red',zorder=10) # diagonal intersection
    ax2.scatter(0.,0.,s=70,color='black',zorder=10) # diagonal intersection



    for key in val_di_final.keys():
        g = val_di_final[key][:,0]
        v1 = val_di_final[key][:,1]
        v2 = val_di_final[key][:,2]

        ax2.add_collection(collect(v1,v2,use_nonan=False,zorder=3,
                                   lwstart=2,lwend=5,
                                   cmapmin=.3,cmapmax=1.))
        ax1.add_collection3d(collect(v1,v2,use_nonan=False,zorder=2,
                                     lwstart=2,lwend=5,
                                     cmapmin=.3,cmapmax=1.),zs=zs,zdir='y')
        



    ax2.annotate(r'$\bm{x}$\textbf{-axis direction}',
                 xy=(.3,.01),xycoords='data',textcoords='data',
                 xytext=(.25,.02),
                 #arrowprops=dict(arrowstyle="-|>",
                 #                connectionstyle="arc3",
                 #                color='black'),
             )


    ax2.annotate(r'$\bm{y}$\textbf{-axis direction}',
                 xy=(.01,.3),xycoords='data',textcoords='data',
                 xytext=(.02,.45),
                 rotation=-90,
                 #arrowprops=dict(arrowstyle="-|>",
                 #                connectionstyle="arc3",
                 #                color='black'),
             )


    ax2.annotate(r'$\bm{g^*}$',
                 xy=(.03,.015),xycoords='data',textcoords='data',
                 xytext=(.2,.07),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )


    ax2.annotate(r'\textbf{Diagonal}',
                 xy=(.48,.5),xycoords='data',textcoords='data',
                 xytext=(.5,.63),
                 rotation=45
                 #arrowprops=dict(arrowstyle="-|>",
                 #                connectionstyle="arc3",
                 #                color='black'),
             )

    ax2.annotate(r'\textbf{Off-diagonal}',
                 xy=(.7,.55),xycoords='data',textcoords='data',
                 xytext=(.65,.75),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate(r'\textbf{Off-diagonal}',
                 alpha=0.,
                 xy=(.55,.7),xycoords='data',textcoords='data',
                 xytext=(.65,.75),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate(r'\textbf{Multiple non-axial directions}',xy=(3.68,.1),xycoords='data',textcoords='data',xytext=(3.,.5),
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    gstarstarpos=(.5,.4)
    ax2.annotate(r'$\bm{g^{**}}$',
                 xy=(.52,.0),xycoords='data',textcoords='data',
                 xytext=gstarstarpos,
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate(r'$\bm{g^{**}}$',
                 alpha=0.,
                 xy=(.0,.52),xycoords='data',textcoords='data',
                 xytext=gstarstarpos,
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )

    ax2.annotate(r'$\bm{g^{**}}$',
                 alpha=0.,
                 xy=(.38,.38),xycoords='data',textcoords='data',
                 xytext=gstarstarpos,
                 arrowprops=dict(arrowstyle="-|>",
                                 connectionstyle="arc3",
                                 color='black'),
             )



    # g,nu1,nu2 annotation
    ax1.text(.01,.8,.05,r'$\bm{g^*}$')
    ax1.text(.01,1.25,vimax-.1,r'$\bm{g^{**}}$')



    #ax1.plot([.89,.89],[-3,3],color='gray')
    #ax1.plot([3.,3.],[-3,3],color='gray')    

    ax1.view_init(20,-8)

    """
    tmp_planes = ax1.zaxis._PLANES 
    ax1.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3], 
                         tmp_planes[0], tmp_planes[1], 
                         tmp_planes[4], tmp_planes[5])
    """

    ax1.set_xlim(vimin,vimax)
    ax1.set_ylim(gmin,gmax)
    ax1.set_zlim(vimin,vimax)

    ax1.set_xlabel(r'$\\\bm{\nu_1}$',fontsize=15)
    ax1.set_ylabel(r'$\bm{g}$',fontsize=15)
    ax1.set_zlabel(r'$\\\bm{\nu_2}$',fontsize=15)


    ax2.set_xlabel(r'$\bm{\nu_1}$',fontsize=15)
    ax2.set_ylabel(r'$\bm{\nu_2}$',fontsize=15)

    ax2.set_xlim(-.05+vimin,vimax)
    ax2.set_ylim(-.05+vimin,vimax)


    ax1.set_xticks(np.arange(0,.8+.2,.2))

    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.zaxis.set_major_formatter(formatter)


    ax2.xaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)


    return fig





def twod_superfig():
    """
    create summary of dynamics
    """
    fig = plt.figure(figsize=(7,7))

    gs = gridspec.GridSpec(6, 6)
    axl = []
    #gs.update(hspace=.4)
    #gs.update(wspace=.3)

    # params
    g = np.zeros((6,6))
    q = np.zeros((6,6))
    
    # slosh, large slosh, const vel (per), const vel (nonper), nonconst vel (per), nonconst vel (chaos)

    # list of models
    models = ['2dfull','2dfulltrunc','2dphs','2dphstrunc','2dphsgauss']

    # 2d full
    g[0,:] = np.array([3., -1, 3., 3., -1., 3.])
    q[0,:] = np.array([1., -1, 0., 0., -1., 1.])

    # 2d full trunc
    
    
    for i in range(6):
        # loop over rows. each axl[i,:] is for plots of model i
        for j in range(6):
            # loop over params each axl[i,j] is plot j of model i
            axl.append(plt.subplot(gs[i,j]))
            
            
            

    """
    ax11 = plt.subplot(gs[0, 0])
    ax12 = plt.subplot(gs[0, 1])
    ax13 = plt.subplot(gs[0, 2])
    ax14 = plt.subplot(gs[0, 3])
    ax15 = plt.subplot(gs[0, 4])
    ax16 = plt.subplot(gs[0, 5])
    """
    
    
    
    #ax11 = plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=2)
    #ax21 = plt.subplot2grid((4,4),(2,0),colspan=3,rowspan=1,sharex=ax11)
    #ax21 = plt.subplot(gs[2,:3],sharex=ax11)


def evans(g=1.8):
    """
    compute the evans function for the non-axial non-diagonal case.
    """
    p = twodp.Phase()
    M_re = 300
    M_im = 300
    N = 200
    
    lam_re = np.linspace(-.25,.5,M_re)
    lam_im = np.linspace(-2,2,M_im)
    sint = np.linspace(0,N/10,N)
    
    #LAM_re, LAM_im, SINT = np.meshgrid(lam_re,lam_im,sint,dtype=np.complex)
    LAM_re, LAM_im, SINT = np.meshgrid(lam_re,lam_im,sint)
    
    LAM_re_contour, LAM_im_contour = np.meshgrid(lam_re,lam_im)
    
    #e_re,e_im = self.evans_v2(LAM_re,LAM_im,SINT)
    e_re,e_im = p.evans_v2(LAM_re,LAM_im,SINT,
                           return_intermediates=False,g=g)
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    #e_re = np.cos(2*LAM_re_contour*pi)*np.sin(LAM_im_contour*pi)
    #e_im = np.sin(2*LAM_re_contour*pi)*np.cos(LAM_re_contour*pi)
    
    cs_re = ax.contour(LAM_re_contour,LAM_im_contour,e_re,levels=[0.])
    cs_im = ax.contour(LAM_re_contour,LAM_im_contour,e_im,levels=[0.])
    
    cs_re.collections[0].set_color('black')
    cs_re.collections[0].set_label('re')
    cs_re.collections[0].set_linewidths(2)
    
    cs_im.collections[0].set_color('gray')
    cs_im.collections[0].set_label('im')
    cs_im.collections[0].set_linewidths(2)


    ax.text(-.24,1.7,'g='+str(g))
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    
    ax.legend()

    return fig


def add_subplot_axes(ax,rect,axisbg='w'):
    """
    https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
    """
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def get_solution_evans_2par(g,b,init,per,dt=.1,return_time=False):
    """
    get solution to truncated phase model. specialized function for evans_2par below
    """

    npa, vn = xpprun('twodphs3.ode',
                     xppname='xppaut',
                     inits={'x':init[0],'y':init[1],
                            'cxs':init[2],'cys':init[3],
                            'sxs':init[4],'sys':init[5],
                            'sxsys':init[6],'sxcys':init[7],
                            'cxsys':init[8],'cxcys':init[9]
                        },
                     parameters={'total':per,
                                 'g':g,
                                 'b':b,
                                 'q':0.,
                                 'eps':.01,
                                 'dt':dt},
                     clean_after=True)
    
    t = npa[:,0]
    sv = npa[:,1:]
    
    thx = sv[:,vn.index('x')]
    thy = sv[:,vn.index('y')]
    
    xval = np.mod(thx+5*pi,2*pi)-pi
    yval = np.mod(thy+5*pi,2*pi)-pi
    
    pos1 = np.where(np.abs(np.diff(xval)) >= 1)[0]
    pos2 = np.where(np.abs(np.diff(yval)) >= 1)[0]

    xval[pos1] = np.nan
    yval[pos1] = np.nan
    
    xval[pos2] = np.nan
    yval[pos2] = np.nan

    if return_time:
        return t,xval,yval
    else:
        return xval,yval


def evans_2par():
    """
    
    """

    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    
    dat_2par = np.loadtxt('evans.ode.2par_b_g.dat')

    x,y = clean(dat_2par[:,0],dat_2par[:,1])

    fig = plt.figure(figsize=(8,5))

    gs = gridspec.GridSpec(2, 3)
    ax11 = plt.subplot(gs[0,0])
    ax21 = plt.subplot(gs[1,0])
    ax12 = plt.subplot(gs[0:,1:])
    ax12a = inset_axes(ax12,width=1, height=1,  loc=1, 
                       bbox_to_anchor=(0.74, 0.45), 
                       bbox_transform=ax12.figure.transFigure)

    ax12b = inset_axes(ax12,width=1, height=1,  loc=1, 
                       bbox_to_anchor=(0.95, 0.79), 
                       bbox_transform=ax12.figure.transFigure)


    
    p = twodp.Phase()


    # evans function example 1
    ax11.set_title(AA)
    ax11.set_xlabel('Re',size=15)
    ax11.set_ylabel('Im',size=15)

    g1,b1 = dat_2par[100,:2]
    M_re = 300
    M_im = 300
    N = 200
    
    lam_re = np.linspace(-.25,1.,M_re)
    lam_im = np.linspace(-3,3,M_im)
    sint = np.linspace(0,N/10,N)
    
    #LAM_re, LAM_im, SINT = np.meshgrid(lam_re,lam_im,sint,dtype=np.complex)
    LAM_re, LAM_im, SINT = np.meshgrid(lam_re,lam_im,sint)
    
    LAM_re_contour, LAM_im_contour = np.meshgrid(lam_re,lam_im)
    
    #e_re,e_im = self.evans_v2(LAM_re,LAM_im,SINT)
    e_re,e_im = p.evans_v2(LAM_re,LAM_im,SINT,
                           return_intermediates=False,g=g1,b=b1)
    
    cs_re = ax11.contour(LAM_re_contour,LAM_im_contour,e_re,levels=[0.])
    cs_im = ax11.contour(LAM_re_contour,LAM_im_contour,e_im,levels=[0.])
    
    cs_re.collections[0].set_color('black')
    cs_re.collections[0].set_label('re')
    cs_re.collections[0].set_linewidths(2)
    
    cs_im.collections[0].set_color('gray')
    cs_im.collections[0].set_label('im')
    cs_im.collections[0].set_linewidths(2)

    # show imaginary axis + intersection
    ax11.axvline(x=0,ymin=-3,ymax=3,color='red',ls='--')
    ax11.scatter([0],[2.05],color='red',zorder=3)
    ax11.scatter([0],[-2.05],color='red',zorder=3)
    
    ax11.set_xlim(-.25,1)
    ax11.set_ylim(-3,3)

    ax11.legend()
    ax11.set_xticks([0,.5,1])
    #ax11.set_xticklabels([])

    

    # evans function example 2
    ax21.set_title(BB)
    ax21.set_xlabel('Re',size=15)
    ax21.set_ylabel('Im',size=15)


    g2,b2 = dat_2par[300,:2]
    
    #e_re,e_im = self.evans_v2(LAM_re,LAM_im,SINT)
    e_re,e_im = p.evans_v2(LAM_re,LAM_im,SINT,
                           return_intermediates=False,g=g2,b=b2)
    
    cs_re = ax21.contour(LAM_re_contour,LAM_im_contour,e_re,levels=[0.])
    cs_im = ax21.contour(LAM_re_contour,LAM_im_contour,e_im,levels=[0.])
    
    cs_re.collections[0].set_color('black')
    cs_re.collections[0].set_label('re')
    cs_re.collections[0].set_linewidths(2)
    
    cs_im.collections[0].set_color('gray')
    cs_im.collections[0].set_label('im')
    cs_im.collections[0].set_linewidths(2)

    # show imaginary axis + intersection
    ax21.axvline(x=0,ymin=-3,ymax=3,color='red',ls='--')
    ax21.scatter([0],[2.35],color='red',zorder=3)
    ax21.scatter([0],[-2.35],color='red',zorder=3)

    ax21.set_xticks([0,.5,1])
    ax21.set_ylim(-3,3)
    ax21.set_xlim(-.25,1)


    # 2 parameter
    ax12.plot(x,y,color='black',lw=2)
    ax12.fill(x,y,alpha=.3)


    ax12.axhline(y=0.8,xmin=0,xmax=35,color='gray',ls='--')

    ax12.text(20,1.3,r'Stable',fontsize=18)
    ax12.text(10,.9,r'Unstable',fontsize=18)

    ax12.scatter(g1,b1,marker='*',s=50)
    ax12.scatter(g2,b2,marker='*',s=50)

    ax12.annotate(r'$\mathbf{A}$',xy=(g1,b1),
                  xycoords='data',xytext=(g1+.5,b1+.01),
                  textcoords='data',
                  size=12,
                  bbox=dict(boxstyle="round4,pad=.2", fc=(1.0, 0.7, 0.7)),
                  #backgroundcolor=labelbg,
                  zorder=3
    )


    ax12.annotate(r'$\mathbf{B}$',xy=(g2,b2),
                  xycoords='data',xytext=(g2+.5,b2+.01),
                  textcoords='data',
                  size=12,
                  bbox=dict(boxstyle="round4,pad=.2", fc=(1.0, 0.7, 0.7)),
                  #backgroundcolor=labelbg,
                  zorder=3
    )

    # get solutions

    # 1st example solution (middle bottom, nonconst vel.)
    np.random.seed(0)
    tot=200
    dt=.01
    t,xval,yval = get_solution_evans_2par(3.,.9,np.random.randn(10),tot,dt=dt,return_time=True)


    #ax12a = beautify_phase(ax12a,t[int(-10/dt):],yval[-int(10/dt):],10,dt,gradient=False,arrowsize=10,decorate=False,arrows=9,use_nonan=False,show_arrows=True,xlo=-10,xhi=200,ylo=-200,yhi=100)
    #ax12a.plot(t[int(-15/dt):],yval[int(-15/dt):],lw=2,color='gray',dashes=(5,2),label=r'$\mathbf{\theta_2}')
    ax12a.plot(t[int(-15/dt):],xval[int(-15/dt):],lw=2,color='black',label=r'$\mathbf{\theta_1}')

    ax12a.set_xlabel(r'$\mathbf{\tau}$')
    ax12a.set_ylabel(r'$\mathbf{\theta_{1}}$')

    ax12a.set_ylim(-pi,pi)
    ax12a.set_xlim(t[int(-15/dt):][0],t[int(-15/dt):][-1])

    ax12a.set_xticks([])
    ax12a.set_yticks([])
    #ax12a.set_xticks(np.arange(-1,1+1,1)*pi)
    #ax12a.set_yticks(np.arange(-1,1+1,1)*pi)
    
    #ax12a.set_xticklabels(x_label_short,fontsize=10)
    #ax12a.set_yticklabels(x_label_short,fontsize=10)

    #ax12a = beautify_phase(ax12a,t[int(-25/dt):],yval[-int(25/dt):],25,dt,gradient=False,arrowsize=10,decorate=False,arrows=9,use_nonan=False,show_arrows=True,xlo=-100,xhi=10)

    if False:
        mp.figure()
        mp.plot(t,xval)
        mp.plot(t,yval)
        mp.show()


    ax12.annotate('',xy=(3.,.9),xytext=(15, .7),
                 xycoords='data',textcoords='data',
                 arrowprops=dict(arrowstyle="->",fc='black',connectionstyle="arc3,rad=-0.5"))


    # 2nd example solution (top right, const vel.)
    np.random.seed(0)
    tot=200
    dt=.01
    t,xval,yval = get_solution_evans_2par(15,1.3,np.random.randn(10),tot,dt=dt,return_time=True)


    #ax12a = beautify_phase(ax12a,t[int(-10/dt):],yval[-int(10/dt):],10,dt,gradient=False,arrowsize=10,decorate=False,arrows=9,use_nonan=False,show_arrows=True,xlo=-10,xhi=200,ylo=-200,yhi=100)
    #ax12b.plot(t[int(-5/dt):],yval[int(-5/dt):],lw=2,color='gray',dashes=(5,2),label=r'$\mathbf{\theta_2}')
    ax12b.plot(t[int(-5/dt):],xval[int(-5/dt):],lw=2,color='black',label=r'$\mathbf{\theta_1}')

    ax12b.set_xlabel(r'$\mathbf{\tau}$')
    ax12b.set_ylabel(r'$\mathbf{\theta_{1}}$')

    ax12b.set_ylim(-pi,pi)
    ax12b.set_xlim(t[int(-5/dt):][0],t[int(-5/dt):][-1])

    ax12b.set_xticks([])
    ax12b.set_yticks([])
    #ax12a.set_xticks(np.arange(-1,1+1,1)*pi)
    #ax12a.set_yticks(np.arange(-1,1+1,1)*pi)

    
    """
    tt=2.5 # total time to display
    ax12b = beautify_phase(ax12b,xval[-int(tt/dt):],yval[int(-tt/dt):],tt,dt,gradient=False,arrowsize=10,decorate=True,arrows=8,use_nonan=False,show_arrows=True,lwstart=2,lwend=2,fontsize=10)

    """

    ax12.annotate('',xy=(15,1.3),xytext=(30, 1.2),
                  xycoords='data',textcoords='data',
                  arrowprops=dict(arrowstyle="->",fc='black',connectionstyle="arc3,rad=0.5"))

    ax12.scatter(3.,.9,marker='*',s=50,color='blue',zorder=3)
    ax12.scatter(15,1.3,marker='*',s=50,color='blue')

    # get largest g value
    #min_idx_r2

    ax12.set_xlim(0,35)
    ax12.set_ylim(.5,1.45)
    ax12.set_xlabel(r'$\mathbf{g}$')
    ax12.set_ylabel(r'$\mathbf{b}$')

    return fig


def evans_full_cont():

    bif = np.loadtxt('evans_full.ode.all_info.dat')

    # remove divergent branches
    
    val_di,ty_di = collect_disjoint_branches(bif,remove_isolated=True,isolated_number=3,remove_redundant=False,N=10)
    
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)

    i = 0
    for key in val_di.keys():
        g = val_di[key][:,0]
        v1 = val_di[key][:,4]
        v2 = val_di[key][:,5]

        # ignore branches that extend below g=1.1
        if np.sum(g[g<=1.11])>0:
            pass
        # igore branches that do not extend past g=1.6
        elif np.sum(g[g>=1.6])==0:
            pass
        else:
            if i == 0:
                ax.plot(g,v1,color='black',label='Re',lw=2)
                ax.plot(g,v2,color='gray',label='Im',lw=2)
            else:
                ax.plot(g,v1,color='black',lw=2)
                ax.plot(g,v2,color='gray',lw=2)

        
        i += 1

    # plot zero line
    #ax.axhline(y=0,xmin=1,xmax=5,color='red',ls='--')
    ax.plot([1,5],[0,0],color='red',ls='--')
    
    ax.set_xlabel(r'$g$')
    ax.set_ylabel(r'$\alpha,\beta$')

    ax.set_xlim(1.2,5)
    ax.set_ylim(-.95,.5)

    ax.legend()

    return fig

def evans_full(g=1.5,lam_re_lo=-.2,lam_re_hi=.2,lam_im_lo=-.5,lam_im_hi=.5):
    """
    compute the evans function for the non-axial non-diagonal case.
    """

    p = twodp.Phase()
    M_re = 300
    M_im = 300
    N = 200
    
    lam_re = np.linspace(lam_re_lo,lam_re_hi,M_re)
    lam_im = np.linspace(lam_im_lo,lam_im_hi,M_im)
    sint = np.linspace(0,N/10,N)
    
    #LAM_re, LAM_im, SINT = np.meshgrid(lam_re,lam_im,sint,dtype=np.complex)
    LAM_re, LAM_im, SINT = np.meshgrid(lam_re,lam_im,sint)    
    LAM_re_contour, LAM_im_contour = np.meshgrid(lam_re,lam_im)
    
    #e_re,e_im = self.evans_v2(LAM_re,LAM_im,SINT)
    e_re,e_im = p.evans_v2(LAM_re,LAM_im,SINT,
                           return_intermediates=False,g=g,mode='full')

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    
    #e_re = np.cos(2*LAM_re_contour*pi)*np.sin(LAM_im_contour*pi)
    #e_im = np.sin(2*LAM_re_contour*pi)*np.cos(LAM_re_contour*pi)
    
    cs_re = ax.contour(LAM_re_contour,LAM_im_contour,e_re,levels=[0.])
    cs_im = ax.contour(LAM_re_contour,LAM_im_contour,e_im,levels=[0.])
    
    cs_re.collections[0].set_color('black')
    cs_re.collections[0].set_label('re')
    cs_re.collections[0].set_linewidths(2)
    
    cs_im.collections[0].set_color('gray')
    cs_im.collections[0].set_label('im')
    cs_im.collections[0].set_linewidths(2)


    

    ax.set_xlabel(r'Re')
    ax.set_ylabel(r'Im')
    
    #ax.set_xlim(1,5)
    #ax.set_ylim(-.5,1.)

    ax.legend()

    return fig



def evans_full_combined():
    """
    compute the evans function for the non-axial non-diagonal case.
    """


    bif = np.loadtxt('evans_full.ode.all_info.dat')
    # sample evans zeros
    reA = bif[420,8]
    imA = bif[420,9]

    reB = bif[244,8]
    imB = bif[244,9]
    

    fig = plt.figure(figsize=(8,5))

    gs = gridspec.GridSpec(2, 3)
    ax11 = plt.subplot(gs[0,0])
    ax21 = plt.subplot(gs[1,0])
    ax12 = plt.subplot(gs[0:,1:])

    axlist = [ax11,ax21]
    lam_re_lo = [-.1,-.5]
    lam_re_hi = [.1,.5]
    lam_im_lo = [-.1,-1.5]
    lam_im_hi = [.1,1.5]
    glist = [1.5,3.]
    title = [AA,BB]

    zeroRE = [reA,reB]
    zeroIM = [imA,imB]

    p = twodp.Phase()
    M_re = 300
    M_im = 300
    N = 200
    
    i = 0
    
    for ax in axlist:
        lam_re = np.linspace(lam_re_lo[i],lam_re_hi[i],M_re)
        lam_im = np.linspace(lam_im_lo[i],lam_im_hi[i],M_im)
        sint = np.linspace(0,N/10,N)
    
        #LAM_re, LAM_im, SINT = np.meshgrid(lam_re,lam_im,sint,dtype=np.complex)
        LAM_re, LAM_im, SINT = np.meshgrid(lam_re,lam_im,sint)    
        LAM_re_contour, LAM_im_contour = np.meshgrid(lam_re,lam_im)
    
        #e_re,e_im = self.evans_v2(LAM_re,LAM_im,SINT)
        e_re,e_im = p.evans_v2(LAM_re,LAM_im,SINT,
                               return_intermediates=False,g=glist[i],mode='full')
        
        cs_re = ax.contour(LAM_re_contour,LAM_im_contour,e_re,levels=[0.])
        cs_im = ax.contour(LAM_re_contour,LAM_im_contour,e_im,levels=[0.])
        
        cs_re.collections[0].set_color('black')
        cs_re.collections[0].set_label('re')
        cs_re.collections[0].set_linewidths(2)
        
        cs_im.collections[0].set_color('gray')
        cs_im.collections[0].set_label('im')
        cs_im.collections[0].set_linewidths(2)
        
        
        ax.set_xlabel(r'Re')
        ax.set_ylabel(r'Im')
        
        #ax.set_xlim(1,5)
        #ax.set_ylim(-.5,1.)
        
        if i == 0:
            ax.set_xticks([-.1,0,.1])
            ax.set_xlim(-.1,.1)
            ax.set_ylim(-.1,.1)
            #ax.legend()
        if i == 1:
            ax.set_xticks([-.4,0,.4])
            ax.set_xlim(-.5,.5)
            ax.set_ylim(-1.5,1.5)

        
        ax.scatter([zeroRE[i]],[zeroIM[i]],color='red',zorder=3)

        ax.set_title(title[i])

        


        i += 1
        







    val_di,ty_di = collect_disjoint_branches(bif,remove_isolated=True,isolated_number=3,remove_redundant=False,N=10)

    i = 0
    for key in val_di.keys():
        g = val_di[key][:,0]
        v1 = val_di[key][:,4]
        v2 = val_di[key][:,5]

        # ignore branches that extend below g=1.1
        if np.sum(g[g<=1.11])>0:
            pass
        # igore branches that do not extend past g=1.6
        elif np.sum(g[g>=1.6])==0:
            pass
        else:
            if i == 0:
                ax12.plot(g,v1,color='black',label='Re',lw=2)
                ax12.plot(g,v2,color='gray',label='Im',lw=2)
            else:
                ax12.plot(g,v1,color='black',lw=2)
                ax12.plot(g,v2,color='gray',lw=2)

        
        i += 1

    # mark locations of panels
    ax12.scatter([glist[0],glist[0]],[zeroRE[0],zeroIM[0]],color='red',zorder=3)
    ax12.scatter([glist[1],glist[1]],[zeroRE[1],zeroIM[1]],color='red',zorder=3)
    


    ax12.annotate(r'$\mathbf{A}$',
                  alpha=0.0,
                  xy=(glist[0], zeroIM[0]), xycoords='data',
                  xytext=(glist[0]+.5, .15), textcoords='data',
                  size=12,
                  zorder=2,

                  verticalalignment='top',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black'),
              )

    ax12.annotate(r'$\mathbf{A}$',
                  xy=(glist[0], zeroRE[0]), xycoords='data',
                  xytext=(glist[0]+.5, .15), textcoords='data',
                  size=12,
                  zorder=2,
                  bbox=dict(boxstyle="round4,pad=.2", fc=(1.0, 0.7, 0.7)),
                  verticalalignment='top',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black'),
              )



    ax12.annotate(r'$\mathbf{B}$',
                  alpha=0.0,
                  xy=(glist[1], zeroIM[1]), xycoords='data',
                  xytext=(glist[1]+.1, -.5), textcoords='data',
                  size=12,
                  zorder=2,
                  verticalalignment='top',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black'),
              )

    ax12.annotate(r'$\mathbf{B}$',
                  xy=(glist[1], zeroRE[1]), xycoords='data',
                  xytext=(glist[1]+.1, -.5), textcoords='data',
                  size=12,
                  zorder=2,
                  bbox=dict(boxstyle="round4,pad=.2", fc=(1.0, 0.7, 0.7)),
                  verticalalignment='top',
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3",
                                  color='black'),
              )



    # plot zero line
    #ax.axhline(y=0,xmin=1,xmax=5,color='red',ls='--')
    ax12.plot([1,5],[0,0],color='red',ls='--')
    
    ax12.set_xlabel(r'$g$')
    ax12.set_ylabel(r'$\alpha,\beta$')

    ax12.set_xlim(1.2,5)
    ax12.set_ylim(-.95,.5)

    ax12.legend()
    

    return fig


def twod_phase_3terms_chaos_fig():

    #data = np.loadtxt("twodphs3_chaos_qvary_g=1.6.dat")
    #data = np.loadtxt("twodphs3_chaos_qvary_g=1.6_1-20th.dat")
    data = np.loadtxt("twodphs3_chaos_gvary_q=0.1.dat")
    #data = np.loadtxt("twodphs3_chaos_gvary_q=0.1_1-100th.dat")
    # this file is organized as follows:
    # t, x,y,cxs,cys,sxs,sys,sxsys,sxcys,cxsys,cxcys,qq,gg
    

    if False:
        #np.savetxt("twodphs3_chaos_qvary_g=1.6_1-20th.dat",data)
        np.savetxt("twodphs3_chaos_gvary_q=0.1_1-100th.dat",data)


    fig = plt.figure(figsize=(6,4))
    #ax = fig.add_subplot(111, projection='3d')

    gs = gridspec.GridSpec(2,3)
    
    ax1 = fig.add_subplot(gs[0,:])
    ax2a = fig.add_subplot(gs[1,0])
    ax2b = fig.add_subplot(gs[1,1])
    ax2c = fig.add_subplot(gs[1,2])

    g = data[:,12]
    t = data[:,0]
    cys = data[:,4]
    sys = data[:,6]

    #cxs = data[:,3]
    #sxs = data[:,6]


    #ax1.scatter(g,cys,s=3,c=np.arange(len(cys)),cmap='winter',edgecolor='none')
    ax1.scatter(g,cys,s=3,edgecolor='none',color='black')
    ax1.set_xlim(np.amin(g),np.amax(g))
    ax1.set_ylim(-1,1)
    ax1.set_xlabel(r'$\bm{g}$',fontsize=15)
    ax1.set_ylabel(r'$\bm{cy}$',fontsize=15)
    #ax.set_ylim(-1,1)

    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)


    labellist = [r'\textbf{F}',r'\textbf{G}',r'\textbf{H}']
    labelcoord = [(1,0),(1.4,0),(2.2,0)]

    # label regions
    for i in range(len(labellist)):
        ax1.annotate(labellist[i],
                     xy=labelcoord[i], xycoords='data',
                     xytext=labelcoord[i], textcoords='data',
                     size=12,
                     bbox=dict(boxstyle="round4,pad=.2", fc=(1.0, 0.7, 0.7)),
                     #backgroundcolor=labelbg,
                     zorder=1
                 )



    pos1 = int(len(g)/12)
    pos2 = int(1.3*len(g)/4)
    pos3 = int(3*len(g)/4)

    ax2a.scatter(sys[g==g[pos1]],cys[g==g[pos1]],s=.2,color='black',edgecolor='none')
    ax2b.scatter(sys[g==g[pos2]],cys[g==g[pos2]],s=.2,color='black',edgecolor='none')
    ax2c.scatter(sys[g==g[pos3]],cys[g==g[pos3]],s=.2,color='black',edgecolor='none')


    ax1.axvline(g[pos1],color='red',lw=2)
    ax1.axvline(g[pos2],color='red',lw=2)
    ax1.axvline(g[pos3],color='red',lw=2)
    
    ax2a.set_xlim(-1,1)
    ax2a.set_ylim(-1,1)
    ax2b.set_xlim(-1,1)
    ax2b.set_ylim(-1,1)
    ax2c.set_xlim(-1,1)
    ax2c.set_ylim(-1,1)

    #plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    ax2a.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    ax2b.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
    ax2c.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    ax2a.set_title(r'$\bm{g='+str(np.round(g[pos1],2))+r'}$')
    ax2b.set_title(r'$\bm{g='+str(np.round(g[pos2],2))+r'}$')
    ax2c.set_title(r'$\bm{g='+str(np.round(g[pos3],2))+r'}$')

    ax2a.set_xlabel(r'$\bm{sy}$',fontsize=15)
    ax2a.set_ylabel(r'$\bm{cy}$',fontsize=15)

    ax2b.set_xlabel(r'$\bm{sy}$',fontsize=15)
    #ax2b.set_ylabel(r'$\bm{cy}$')

    ax2c.set_xlabel(r'$\bm{sy}$',fontsize=15)
    #ax2c.set_ylabel(r'$\bm{cy}$')


    
    
    return fig

def psi_fn(g,q,om):
    """
    wave lag function
    """
    return arcsin((-om+g*om-om**3.)/(q*(1+om**2.)))

def lurch_l1(g,q,psi,om):
    return -(2 - g + 2*om**2 + q*Cos(psi) + q*om**2*Cos(psi))/(3.*(1 + om**2)) - (2**0.3333333333333333*(-((2 - g + 2*om**2 + q*Cos(psi) + q*om**2*Cos(psi))**2/(1 + om**2)**2) + (3*(1 - g + 2*om**2 + g*om**2 + om**4 + 2*q*Cos(psi) + 2*q*om**2*Cos(psi)))/(1 + om**2)))/(3.*(-16/(1 + om**2)**3 + (24*g)/(1 + om**2)**3 - (12*g**2)/(1 + om**2)**3 + (2*g**3)/(1 + om**2)**3 - (48*om**2)/(1 + om**2)**3 + (48*g*om**2)/(1 + om**2)**3 - (12*g**2*om**2)/(1 + om**2)**3 - (48*om**4)/(1 + om**2)**3 + (24*g*om**4)/(1 + om**2)**3 - (16*om**6)/(1 + om**2)**3 + 18/(1 + om**2)**2 - (27*g)/(1 + om**2)**2 + (9*g**2)/(1 + om**2)**2 + (54*om**2)/(1 + om**2)**2 - (18*g*om**2)/(1 + om**2)**2 - (9*g**2*om**2)/(1 + om**2)**2 + (54*om**4)/(1 + om**2)**2 + (9*g*om**4)/(1 + om**2)**2 + (18*om**6)/(1 + om**2)**2 - 27*q*Cos(psi) - 27*q*om**2*Cos(psi) - (24*q*Cos(psi))/(1 + om**2)**3 + (24*g*q*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*Cos(psi))/(1 + om**2)**3 - (72*q*om**2*Cos(psi))/(1 + om**2)**3 + (48*g*q*om**2*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*om**2*Cos(psi))/(1 + om**2)**3 - (72*q*om**4*Cos(psi))/(1 + om**2)**3 + (24*g*q*om**4*Cos(psi))/(1 + om**2)**3 - (24*q*om**6*Cos(psi))/(1 + om**2)**3 + (45*q*Cos(psi))/(1 + om**2)**2 - (27*g*q*Cos(psi))/(1 + om**2)**2 + (99*q*om**2*Cos(psi))/(1 + om**2)**2 - (18*g*q*om**2*Cos(psi))/(1 + om**2)**2 + (63*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*g*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*q*om**6*Cos(psi))/(1 + om**2)**2 - (12*q**2*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 + (12*g*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 - (12*q**2*om**6*Cos(psi)**2)/(1 + om**2)**3 + (18*q**2*Cos(psi)**2)/(1 + om**2)**2 + (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**2 + (18*q**2*om**4*Cos(psi)**2)/(1 + om**2)**2 - (2*q**3*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**2*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**4*Cos(psi)**3)/(1 + om**2)**3 - (2*q**3*om**6*Cos(psi)**3)/(1 + om**2)**3 + Sqrt((-16/(1 + om**2)**3 + (24*g)/(1 + om**2)**3 - (12*g**2)/(1 + om**2)**3 + (2*g**3)/(1 + om**2)**3 - (48*om**2)/(1 + om**2)**3 + (48*g*om**2)/(1 + om**2)**3 - (12*g**2*om**2)/(1 + om**2)**3 - (48*om**4)/(1 + om**2)**3 + (24*g*om**4)/(1 + om**2)**3 - (16*om**6)/(1 + om**2)**3 + 18/(1 + om**2)**2 - (27*g)/(1 + om**2)**2 + (9*g**2)/(1 + om**2)**2 + (54*om**2)/(1 + om**2)**2 - (18*g*om**2)/(1 + om**2)**2 - (9*g**2*om**2)/(1 + om**2)**2 + (54*om**4)/(1 + om**2)**2 + (9*g*om**4)/(1 + om**2)**2 + (18*om**6)/(1 + om**2)**2 - 27*q*Cos(psi) - 27*q*om**2*Cos(psi) - (24*q*Cos(psi))/(1 + om**2)**3 + (24*g*q*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*Cos(psi))/(1 + om**2)**3 - (72*q*om**2*Cos(psi))/(1 + om**2)**3 + (48*g*q*om**2*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*om**2*Cos(psi))/(1 + om**2)**3 - (72*q*om**4*Cos(psi))/(1 + om**2)**3 + (24*g*q*om**4*Cos(psi))/(1 + om**2)**3 - (24*q*om**6*Cos(psi))/(1 + om**2)**3 + (45*q*Cos(psi))/(1 + om**2)**2 - (27*g*q*Cos(psi))/(1 + om**2)**2 + (99*q*om**2*Cos(psi))/(1 + om**2)**2 - (18*g*q*om**2*Cos(psi))/(1 + om**2)**2 + (63*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*g*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*q*om**6*Cos(psi))/(1 + om**2)**2 - (12*q**2*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 + (12*g*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 - (12*q**2*om**6*Cos(psi)**2)/(1 + om**2)**3 + (18*q**2*Cos(psi)**2)/(1 + om**2)**2 + (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**2 + (18*q**2*om**4*Cos(psi)**2)/(1 + om**2)**2 - (2*q**3*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**2*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**4*Cos(psi)**3)/(1 + om**2)**3 - (2*q**3*om**6*Cos(psi)**3)/(1 + om**2)**3)**2 + 4*(-((2 - g + 2*om**2 + q*Cos(psi) + q*om**2*Cos(psi))**2/(1 + om**2)**2) + (3*(1 - g + 2*om**2 + g*om**2 + om**4 + 2*q*Cos(psi) + 2*q*om**2*Cos(psi)))/(1 + om**2))**3))**0.3333333333333333) + (-16/(1 + om**2)**3 + (24*g)/(1 + om**2)**3 - (12*g**2)/(1 + om**2)**3 + (2*g**3)/(1 + om**2)**3 - (48*om**2)/(1 + om**2)**3 + (48*g*om**2)/(1 + om**2)**3 - (12*g**2*om**2)/(1 + om**2)**3 - (48*om**4)/(1 + om**2)**3 + (24*g*om**4)/(1 + om**2)**3 - (16*om**6)/(1 + om**2)**3 + 18/(1 + om**2)**2 - (27*g)/(1 + om**2)**2 + (9*g**2)/(1 + om**2)**2 + (54*om**2)/(1 + om**2)**2 - (18*g*om**2)/(1 + om**2)**2 - (9*g**2*om**2)/(1 + om**2)**2 + (54*om**4)/(1 + om**2)**2 + (9*g*om**4)/(1 + om**2)**2 + (18*om**6)/(1 + om**2)**2 - 27*q*Cos(psi) - 27*q*om**2*Cos(psi) - (24*q*Cos(psi))/(1 + om**2)**3 + (24*g*q*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*Cos(psi))/(1 + om**2)**3 - (72*q*om**2*Cos(psi))/(1 + om**2)**3 + (48*g*q*om**2*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*om**2*Cos(psi))/(1 + om**2)**3 - (72*q*om**4*Cos(psi))/(1 + om**2)**3 + (24*g*q*om**4*Cos(psi))/(1 + om**2)**3 - (24*q*om**6*Cos(psi))/(1 + om**2)**3 + (45*q*Cos(psi))/(1 + om**2)**2 - (27*g*q*Cos(psi))/(1 + om**2)**2 + (99*q*om**2*Cos(psi))/(1 + om**2)**2 - (18*g*q*om**2*Cos(psi))/(1 + om**2)**2 + (63*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*g*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*q*om**6*Cos(psi))/(1 + om**2)**2 - (12*q**2*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 + (12*g*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 - (12*q**2*om**6*Cos(psi)**2)/(1 + om**2)**3 + (18*q**2*Cos(psi)**2)/(1 + om**2)**2 + (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**2 + (18*q**2*om**4*Cos(psi)**2)/(1 + om**2)**2 - (2*q**3*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**2*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**4*Cos(psi)**3)/(1 + om**2)**3 - (2*q**3*om**6*Cos(psi)**3)/(1 + om**2)**3 + Sqrt((-16/(1 + om**2)**3 + (24*g)/(1 + om**2)**3 - (12*g**2)/(1 + om**2)**3 + (2*g**3)/(1 + om**2)**3 - (48*om**2)/(1 + om**2)**3 + (48*g*om**2)/(1 + om**2)**3 - (12*g**2*om**2)/(1 + om**2)**3 - (48*om**4)/(1 + om**2)**3 + (24*g*om**4)/(1 + om**2)**3 - (16*om**6)/(1 + om**2)**3 + 18/(1 + om**2)**2 - (27*g)/(1 + om**2)**2 + (9*g**2)/(1 + om**2)**2 + (54*om**2)/(1 + om**2)**2 - (18*g*om**2)/(1 + om**2)**2 - (9*g**2*om**2)/(1 + om**2)**2 + (54*om**4)/(1 + om**2)**2 + (9*g*om**4)/(1 + om**2)**2 + (18*om**6)/(1 + om**2)**2 - 27*q*Cos(psi) - 27*q*om**2*Cos(psi) - (24*q*Cos(psi))/(1 + om**2)**3 + (24*g*q*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*Cos(psi))/(1 + om**2)**3 - (72*q*om**2*Cos(psi))/(1 + om**2)**3 + (48*g*q*om**2*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*om**2*Cos(psi))/(1 + om**2)**3 - (72*q*om**4*Cos(psi))/(1 + om**2)**3 + (24*g*q*om**4*Cos(psi))/(1 + om**2)**3 - (24*q*om**6*Cos(psi))/(1 + om**2)**3 + (45*q*Cos(psi))/(1 + om**2)**2 - (27*g*q*Cos(psi))/(1 + om**2)**2 + (99*q*om**2*Cos(psi))/(1 + om**2)**2 - (18*g*q*om**2*Cos(psi))/(1 + om**2)**2 + (63*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*g*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*q*om**6*Cos(psi))/(1 + om**2)**2 - (12*q**2*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 + (12*g*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 - (12*q**2*om**6*Cos(psi)**2)/(1 + om**2)**3 + (18*q**2*Cos(psi)**2)/(1 + om**2)**2 + (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**2 + (18*q**2*om**4*Cos(psi)**2)/(1 + om**2)**2 - (2*q**3*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**2*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**4*Cos(psi)**3)/(1 + om**2)**3 - (2*q**3*om**6*Cos(psi)**3)/(1 + om**2)**3)**2 + 4*(-((2 - g + 2*om**2 + q*Cos(psi) + q*om**2*Cos(psi))**2/(1 + om**2)**2) + (3*(1 - g + 2*om**2 + g*om**2 + om**4 + 2*q*Cos(psi) + 2*q*om**2*Cos(psi)))/(1 + om**2))**3))**0.3333333333333333/(3.*2**0.3333333333333333)

def lurch_l2(g,q,psi,om):

    return -(2 - g + 2*om**2 + q*Cos(psi) + q*om**2*Cos(psi))/(3.*(1 + om**2)) + ((1 + 1j*Sqrt(3))*(-((2 - g + 2*om**2 + q*Cos(psi) + q*om**2*Cos(psi))**2/(1 + om**2)**2) + (3*(1 - g + 2*om**2 + g*om**2 + om**4 + 2*q*Cos(psi) + 2*q*om**2*Cos(psi)))/(1 + om**2)))/(3.*2**0.6666666666666666*(-16/(1 + om**2)**3 + (24*g)/(1 + om**2)**3 - (12*g**2)/(1 + om**2)**3 + (2*g**3)/(1 + om**2)**3 - (48*om**2)/(1 + om**2)**3 + (48*g*om**2)/(1 + om**2)**3 - (12*g**2*om**2)/(1 + om**2)**3 - (48*om**4)/(1 + om**2)**3 + (24*g*om**4)/(1 + om**2)**3 - (16*om**6)/(1 + om**2)**3 + 18/(1 + om**2)**2 - (27*g)/(1 + om**2)**2 + (9*g**2)/(1 + om**2)**2 + (54*om**2)/(1 + om**2)**2 - (18*g*om**2)/(1 + om**2)**2 - (9*g**2*om**2)/(1 + om**2)**2 + (54*om**4)/(1 + om**2)**2 + (9*g*om**4)/(1 + om**2)**2 + (18*om**6)/(1 + om**2)**2 - 27*q*Cos(psi) - 27*q*om**2*Cos(psi) - (24*q*Cos(psi))/(1 + om**2)**3 + (24*g*q*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*Cos(psi))/(1 + om**2)**3 - (72*q*om**2*Cos(psi))/(1 + om**2)**3 + (48*g*q*om**2*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*om**2*Cos(psi))/(1 + om**2)**3 - (72*q*om**4*Cos(psi))/(1 + om**2)**3 + (24*g*q*om**4*Cos(psi))/(1 + om**2)**3 - (24*q*om**6*Cos(psi))/(1 + om**2)**3 + (45*q*Cos(psi))/(1 + om**2)**2 - (27*g*q*Cos(psi))/(1 + om**2)**2 + (99*q*om**2*Cos(psi))/(1 + om**2)**2 - (18*g*q*om**2*Cos(psi))/(1 + om**2)**2 + (63*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*g*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*q*om**6*Cos(psi))/(1 + om**2)**2 - (12*q**2*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 + (12*g*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 - (12*q**2*om**6*Cos(psi)**2)/(1 + om**2)**3 + (18*q**2*Cos(psi)**2)/(1 + om**2)**2 + (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**2 + (18*q**2*om**4*Cos(psi)**2)/(1 + om**2)**2 - (2*q**3*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**2*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**4*Cos(psi)**3)/(1 + om**2)**3 - (2*q**3*om**6*Cos(psi)**3)/(1 + om**2)**3 + Sqrt((-16/(1 + om**2)**3 + (24*g)/(1 + om**2)**3 - (12*g**2)/(1 + om**2)**3 + (2*g**3)/(1 + om**2)**3 - (48*om**2)/(1 + om**2)**3 + (48*g*om**2)/(1 + om**2)**3 - (12*g**2*om**2)/(1 + om**2)**3 - (48*om**4)/(1 + om**2)**3 + (24*g*om**4)/(1 + om**2)**3 - (16*om**6)/(1 + om**2)**3 + 18/(1 + om**2)**2 - (27*g)/(1 + om**2)**2 + (9*g**2)/(1 + om**2)**2 + (54*om**2)/(1 + om**2)**2 - (18*g*om**2)/(1 + om**2)**2 - (9*g**2*om**2)/(1 + om**2)**2 + (54*om**4)/(1 + om**2)**2 + (9*g*om**4)/(1 + om**2)**2 + (18*om**6)/(1 + om**2)**2 - 27*q*Cos(psi) - 27*q*om**2*Cos(psi) - (24*q*Cos(psi))/(1 + om**2)**3 + (24*g*q*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*Cos(psi))/(1 + om**2)**3 - (72*q*om**2*Cos(psi))/(1 + om**2)**3 + (48*g*q*om**2*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*om**2*Cos(psi))/(1 + om**2)**3 - (72*q*om**4*Cos(psi))/(1 + om**2)**3 + (24*g*q*om**4*Cos(psi))/(1 + om**2)**3 - (24*q*om**6*Cos(psi))/(1 + om**2)**3 + (45*q*Cos(psi))/(1 + om**2)**2 - (27*g*q*Cos(psi))/(1 + om**2)**2 + (99*q*om**2*Cos(psi))/(1 + om**2)**2 - (18*g*q*om**2*Cos(psi))/(1 + om**2)**2 + (63*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*g*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*q*om**6*Cos(psi))/(1 + om**2)**2 - (12*q**2*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 + (12*g*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 - (12*q**2*om**6*Cos(psi)**2)/(1 + om**2)**3 + (18*q**2*Cos(psi)**2)/(1 + om**2)**2 + (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**2 + (18*q**2*om**4*Cos(psi)**2)/(1 + om**2)**2 - (2*q**3*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**2*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**4*Cos(psi)**3)/(1 + om**2)**3 - (2*q**3*om**6*Cos(psi)**3)/(1 + om**2)**3)**2 + 4*(-((2 - g + 2*om**2 + q*Cos(psi) + q*om**2*Cos(psi))**2/(1 + om**2)**2) + (3*(1 - g + 2*om**2 + g*om**2 + om**4 + 2*q*Cos(psi) + 2*q*om**2*Cos(psi)))/(1 + om**2))**3))**0.3333333333333333) - ((1 - 1j*Sqrt(3))*(-16/(1 + om**2)**3 + (24*g)/(1 + om**2)**3 - (12*g**2)/(1 + om**2)**3 + (2*g**3)/(1 + om**2)**3 - (48*om**2)/(1 + om**2)**3 + (48*g*om**2)/(1 + om**2)**3 - (12*g**2*om**2)/(1 + om**2)**3 - (48*om**4)/(1 + om**2)**3 + (24*g*om**4)/(1 + om**2)**3 - (16*om**6)/(1 + om**2)**3 + 18/(1 + om**2)**2 - (27*g)/(1 + om**2)**2 + (9*g**2)/(1 + om**2)**2 + (54*om**2)/(1 + om**2)**2 - (18*g*om**2)/(1 + om**2)**2 - (9*g**2*om**2)/(1 + om**2)**2 + (54*om**4)/(1 + om**2)**2 + (9*g*om**4)/(1 + om**2)**2 + (18*om**6)/(1 + om**2)**2 - 27*q*Cos(psi) - 27*q*om**2*Cos(psi) - (24*q*Cos(psi))/(1 + om**2)**3 + (24*g*q*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*Cos(psi))/(1 + om**2)**3 - (72*q*om**2*Cos(psi))/(1 + om**2)**3 + (48*g*q*om**2*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*om**2*Cos(psi))/(1 + om**2)**3 - (72*q*om**4*Cos(psi))/(1 + om**2)**3 + (24*g*q*om**4*Cos(psi))/(1 + om**2)**3 - (24*q*om**6*Cos(psi))/(1 + om**2)**3 + (45*q*Cos(psi))/(1 + om**2)**2 - (27*g*q*Cos(psi))/(1 + om**2)**2 + (99*q*om**2*Cos(psi))/(1 + om**2)**2 - (18*g*q*om**2*Cos(psi))/(1 + om**2)**2 + (63*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*g*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*q*om**6*Cos(psi))/(1 + om**2)**2 - (12*q**2*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 + (12*g*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 - (12*q**2*om**6*Cos(psi)**2)/(1 + om**2)**3 + (18*q**2*Cos(psi)**2)/(1 + om**2)**2 + (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**2 + (18*q**2*om**4*Cos(psi)**2)/(1 + om**2)**2 - (2*q**3*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**2*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**4*Cos(psi)**3)/(1 + om**2)**3 - (2*q**3*om**6*Cos(psi)**3)/(1 + om**2)**3 + Sqrt((-16/(1 + om**2)**3 + (24*g)/(1 + om**2)**3 - (12*g**2)/(1 + om**2)**3 + (2*g**3)/(1 + om**2)**3 - (48*om**2)/(1 + om**2)**3 + (48*g*om**2)/(1 + om**2)**3 - (12*g**2*om**2)/(1 + om**2)**3 - (48*om**4)/(1 + om**2)**3 + (24*g*om**4)/(1 + om**2)**3 - (16*om**6)/(1 + om**2)**3 + 18/(1 + om**2)**2 - (27*g)/(1 + om**2)**2 + (9*g**2)/(1 + om**2)**2 + (54*om**2)/(1 + om**2)**2 - (18*g*om**2)/(1 + om**2)**2 - (9*g**2*om**2)/(1 + om**2)**2 + (54*om**4)/(1 + om**2)**2 + (9*g*om**4)/(1 + om**2)**2 + (18*om**6)/(1 + om**2)**2 - 27*q*Cos(psi) - 27*q*om**2*Cos(psi) - (24*q*Cos(psi))/(1 + om**2)**3 + (24*g*q*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*Cos(psi))/(1 + om**2)**3 - (72*q*om**2*Cos(psi))/(1 + om**2)**3 + (48*g*q*om**2*Cos(psi))/(1 + om**2)**3 - (6*g**2*q*om**2*Cos(psi))/(1 + om**2)**3 - (72*q*om**4*Cos(psi))/(1 + om**2)**3 + (24*g*q*om**4*Cos(psi))/(1 + om**2)**3 - (24*q*om**6*Cos(psi))/(1 + om**2)**3 + (45*q*Cos(psi))/(1 + om**2)**2 - (27*g*q*Cos(psi))/(1 + om**2)**2 + (99*q*om**2*Cos(psi))/(1 + om**2)**2 - (18*g*q*om**2*Cos(psi))/(1 + om**2)**2 + (63*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*g*q*om**4*Cos(psi))/(1 + om**2)**2 + (9*q*om**6*Cos(psi))/(1 + om**2)**2 - (12*q**2*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 + (12*g*q**2*om**2*Cos(psi)**2)/(1 + om**2)**3 - (36*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 + (6*g*q**2*om**4*Cos(psi)**2)/(1 + om**2)**3 - (12*q**2*om**6*Cos(psi)**2)/(1 + om**2)**3 + (18*q**2*Cos(psi)**2)/(1 + om**2)**2 + (36*q**2*om**2*Cos(psi)**2)/(1 + om**2)**2 + (18*q**2*om**4*Cos(psi)**2)/(1 + om**2)**2 - (2*q**3*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**2*Cos(psi)**3)/(1 + om**2)**3 - (6*q**3*om**4*Cos(psi)**3)/(1 + om**2)**3 - (2*q**3*om**6*Cos(psi)**3)/(1 + om**2)**3)**2 + 4*(-((2 - g + 2*om**2 + q*Cos(psi) + q*om**2*Cos(psi))**2/(1 + om**2)**2) + (3*(1 - g + 2*om**2 + g*om**2 + om**4 + 2*q*Cos(psi) + 2*q*om**2*Cos(psi)))/(1 + om**2))**3))**0.3333333333333333)/(6.*2**0.3333333333333333)
    
def lurching_existence():
    """
    lurching waves.

    existence and stability equations comptued in mathematica "lurching.nb"    
    """
    
    om = np.linspace(-5,5,5000) # velocity
    q1 = 2. # pinning strength
    q2 = 3. # pinning strength
    q3 = 4. # pinning strength
    g=4.
    
    fig = plt.figure(figsize=(6,3))
    gs = gridspec.GridSpec(2,4)
    ax11 = plt.subplot(gs[:2,:2])
    ax13 = plt.subplot(gs[0,2:])
    ax23 = plt.subplot(gs[1,2:])
    #ax11 = fig.add_subplot(221)
    #ax12 = fig.add_subplot(222)
    
    #ax21 = fig.add_subplot(223)
    #ax22 = fig.add_subplot(224)

    p1 = psi_fn(g,q1,om)
    p2 = psi_fn(g,q2,om)
    p3 = psi_fn(g,q3,om)
    
    ax11.plot(om,np.real(p1),color='0.',lw=2,label=r'$q=2$')
    ax11.plot(om,np.real(p2),color='.4',lw=1.5,label=r'$q=3$')
    ax11.plot(om,np.real(p3),color='.75',label=r'$q=4$')

    om_im = np.linspace(-3,3+0*1j,100)
    L1 = lurch_l1(g,q1,psi_fn(g,q1,om_im),om_im)
    ax13.plot(np.real(om_im),np.real(L1),color='black')
    ax13.plot(np.real(om_im),np.imag(L1),color='red')

    L2 = lurch_l2(g,q1,psi_fn(g,q1,om_im),om_im)
    ax23.plot(np.real(om_im),np.real(L2),color='black')
    ax23.plot(np.real(om_im),np.imag(L2),color='red')

    # plot vertical lines indicating example parameter values
    ax13.plot([1,1],[-2,4],color='red',ls='--',zorder=-3,alpha=.7,dashes=(5,2))
    ax13.plot([0.5,0.5],[-2,4],color='red',ls='--',zorder=-3,alpha=.7,dashes=(5,2))

    ax23.plot([1,1],[-2,4],color='red',ls='--',zorder=-3,alpha=.7,dashes=(5,2))
    ax23.plot([0.5,0.5],[-2,4],color='red',ls='--',zorder=-3,alpha=.7,dashes=(5,2))

    ax11.set_ylabel(r'$\bm{\psi}$',fontsize=15)
    ax11.set_xlabel(r'$\bm{\Omega}$',fontsize=15)

    ax11.set_xlim(om[0],om[-1])
    ax11.set_ylim(-pi/2.,pi/2.)

    ax11.set_yticks(np.arange(-.5,.5+.5,.5)*pi)
    ax11.set_yticklabels(x_label2,fontsize=15)

    #ax13.set_xlim()
    ax13.set_ylim(-1.5,.1)

    ax23.set_xlabel(r'$\bm{\Omega}$',fontsize=15)

    ax13.set_xticks([])

    ax13.set_ylabel(r'$\bm{\lambda_1}$',labelpad=-3)
    ax23.set_ylabel(r'$\bm{\lambda_2,\overline\lambda_3}$',labelpad=-3)

    ax11.xaxis.set_major_formatter(formatter)
    #ax11.yaxis.set_major_formatter(formatter)

    ax13.xaxis.set_major_formatter(formatter)
    ax13.yaxis.set_major_formatter(formatter)

    ax23.xaxis.set_major_formatter(formatter)
    ax23.yaxis.set_major_formatter(formatter)

    ax13.locator_params(axis='y',nbins=3)
    ax23.locator_params(axis='y',nbins=3)



    ax11.legend(prop={'size':9})

    return fig


def lurching_example():
    """
    example of losso f stability
    """
    
    fig = plt.figure(figsize=(6,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)



    dat1 = oned_simple.SimDat(g=4.,q=2.,zshift=.1,T=2000,phase=True,Ivelocity=0.005)
    dat2 = oned_simple.SimDat(g=4.,q=2.,zshift=.1,T=2000,phase=True,Ivelocity=0.01)


    # clean numerics
    x1,y1 = clean(dat1.t,dat1.ph_angle)
    x2,y2 = clean(dat2.t,dat2.ph_angle)

    # clean theory

    xt1,yt1 = clean(dat1.t,-(np.mod(dat1.solph[:,0]+pi,2*pi)-pi))
    xt2,yt2 = clean(dat2.t,-(np.mod(dat2.solph[:,0]+pi,2*pi)-pi))
    #ax.plot(dat1.t,-(np.mod(self.solph[:,0]+pi,2*pi)-pi),lw=3,color='green')

    # plot numerics
    ax1.plot(x1,y1,color='k',lw=2)
    ax2.plot(x2,y2,color='k',lw=2)

    # plot theory #blue2='#0066B2'
    ax1.plot(xt1,yt1,color=blue2,lw=2)
    ax2.plot(xt2,yt2,color=blue2,lw=2)

    ax1.set_title(r'$\bm{\Omega=0.5}$')
    ax2.set_title(r'$\bm{\Omega=1}$')

    ax1.set_xlabel(r'$\bm{t}$')
    ax2.set_xlabel(r'$\bm{t}$')
    
    ax1.set_ylim(-pi/2.,pi/2.)
    ax2.set_ylim(-pi/2.,pi/2.)

    ax1.set_yticks(np.arange(-1,1+1,1)*pi)
    ax1.set_yticklabels(x_label_short,fontsize=15)

    ax2.set_yticks(np.arange(-1,1+1,1)*pi)
    ax2.set_yticklabels(x_label_short,fontsize=15)

    ax1.xaxis.set_major_formatter(formatter)
    ax2.xaxis.set_major_formatter(formatter)

    return fig
    

def generate_figure(function, args, filenames, dpi=100):
    # workaround for python bug where forked processes use the same random 
    # filename.
    #tempfile._name_sequence = None;

    fig = function(*args)

    if type(filenames) == list:
        for name in filenames:
            if name.split('.')[-1] == 'ps':
                fig.savefig(name, orientation='landscape',dpi=dpi)
            else:
                fig.savefig(name,dpi=dpi)
    else:
        if name.split('.')[-1] == 'ps':
            fig.savefig(filenames,orientation='landscape',dpi=dpi)
        else:
            fig.savefig(filenames,dpi=dpi)

    

def main():


    figures = [
        #(ss_bump_fig,[],["ss_bumps.pdf"]),
        #(HJ_fig, [],['oned_HJ_fig.pdf']),
        #(HJ_i_fig,[],['HJ_i_fig.pdf']),

        #(oned_full_auto,[],["1d_full_auto_q0p5_gvary.pdf"]),
        #(oned_phase_auto,['q0.5'],["1d_auto_q0p5_gvary.pdf"]),
        #(g_nu_fig,[],['g_nu.pdf']),

        #(oned_bump_combined,[],["oned_bump_combined.pdf"]), 
        #(oned_normal_form_bard,[],['1d_normal_form.pdf']),
        #(oned_chaos_fig,[],['oned_chaos.pdf']),

        #(oned_chaos_fig_finer,[],['oned_chaos_finer.pdf']),
        #(oned_phase_chaos_fig_map,[],['oned_phase_chaos_fig_map.pdf']),

        #(twod_full_auto_5terms_fig,[],['twod_full_auto_5terms.pdf']),
        #(twod_full_auto_5terms_2par,[],['twod_full_auto_5terms_2par.pdf']),

        #(twod_phase_auto_3terms_fig2,[],['twod_phase_auto_3terms_q=0p1_.pdf']), # if you are getting a stop iteration error, comment this line and continue.
        #(twod_phase_auto_3terms_2par,[],['twod_phase_auto_3terms_2par.pdf']),
        #(combined_phase_fig,["const"],["twod_const_velocity.pdf","twod_const_velocity.png"]),

        #(wave_exist_2d_full_v4,[],['twod_wave_exist_full_v4.pdf']),
        #(wave_exist_2d_trunc_v4,[],['twod_wave_exist_trunc_v4.pdf']),
        #(wave_stbl_2d,[],['twod_wave_stbl.pdf']),

        #(evans_full_combined,[],['evans_full_combined.pdf']),
        #(evans_2par,[],['evans_2par.pdf']),
        #(combined_phase_fig,["limit_cycle"],["twod_limit_cycle.pdf","twod_limit_cycle.png"]),

        #(combined_phase_fig,["non_const"],["twod_non_const.pdf","twod_non_const.png"]),
        #(twod_phase_3terms_chaos_fig,[],['twod_phase_3terms_chaos_fig.png'],500),

        ## figures for thesis defense
        (lurching_existence,[],['lurching_existence.pdf']),
        (lurching_example,[],['lurching_example.pdf']),
        
        ]


    for fig in figures:
        generate_figure(*fig)


if __name__ == "__main__":
    main()
