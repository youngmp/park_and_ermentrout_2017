"""
2d fourier playground

https://en.wikipedia.org/wiki/Fourier_series#Fourier_series_on_a_square
"""


#import twod_full_square as twod
import twod
import numpy as np
import matplotlib.pylab as mp
import matplotlib.pyplot as plt
import os
import time

#print dir(twod)

sin = np.sin
cos = np.cos
pi = np.pi
sqrt = np.sqrt
exp = np.exp

def bump_velocity_test(a,n,m,g,c1,c2):
    """
    testing/playground for bump velocity stuff
    see Teusday 12 April 2016 log 2, traveling bump notes in 2d
    -numerically approximate bump velocity
    -perform error check using zero velocity in one coordinate.
    """
    ntemp = n
    n = m
    m = ntemp
    
    # when c_1 = 0, c_2 \neq 0
    if c1 == 0 and c2 != 0:
        # we expect the first equation to be zero, the second to be 1
        # if the theory is sound, the first coordinate should always return 0
        # expected1, true1, expected2, true2
        return np.sum(a*m/(1+(c2*m)**2.)),0.,g*np.sum(a*n/(1.+(c2*n)**2.)),1.
    elif c1 != 0 and c2 == 0:
        pass
    elif c1 != 0 and c2 != 0:
        pass
    
def H1_fourier(x,y,d=False,h2_flag=False):
    """
    H_i function for 2d phase model, i=1,2
    2pi periodic. let domain be on -pi,pi.
    coeff list gathered from data using 64x64 grid.
    coeff list normalized manually at the end. Keep this in mind!!!
    
    """

    # from N = 64

    # gives excellent approximation 26 raw coeff (some freq and coeff are repeated)

    # these parameters are for se=1,si=2
    """
    idx_im = [(0, 1), (0, 2), (0, 3), (0, -3),
              (0, -2), (0, -1), (1, 1), (1, 2),
              (1, 3), (1, -3), (1, -2), (1, -1),
              (2, 1), (2, 2), (2, -2), (2, -1),
              (-2, 1), (-2, 2), (-2, -2), (-2, -1),
              (-1, 1), (-1, 2), (-1, 3), (-1, -3),
              (-1, -2), (-1, -1)]

    

    idx_im = np.array([0,1],
                      [0,2],
                      [0,3],
                      [0,-3],
                      [0,-2],
                      [0,-1],
                      [1,1],
                      [1,2],
                      [1,3],
                      [1,-3],
                      [1,-2],
                      [1,-1],
                      [2,1],
                      [2,2],
                      [2,-2],
                      [2,-1],
                      [-2,1],
                      [-2,2],
                      [-2,-2],
                      [-2,-1],
                      [-1,1],
                      [-1,2],
                      [-1,3],
                      [-1,-3],
                      [-1,-2],
                      [-1,-1])


    coeff_im = [-17.5606063462,
                -147.128891436,
                -20.3268644879,
                20.3268644879,
                147.128891436,
                17.5606063462,
                -113.536211995,
                -96.3291845099,
                -11.9628812875,
                11.9628812875,
                96.3291845099,
                113.536211995,
                -48.3080011014,
                -23.0911624865,
                23.0911624865,
                48.3080011014,
                -48.3080011014,
                -23.0911624865,
                23.0911624865,
                48.3080011014,
                -113.536211995,
                -96.3291845099,
                -11.9628812875,
                11.9628812875,
                96.3291845099,
                113.536211995]
    """
    coeff_im = [   2.99041641e-01,  -1.23427222e-02,   2.92404663e-07,  -2.92404663e-07,
                   1.23427222e-02,  -2.99041641e-01,  -1.10662060e-01 ,  2.55677958e-03,
                  -1.30119170e-07,   1.30119170e-07,  -2.55677958e-03,   1.10662060e-01,
                   1.34078963e-03,  -8.78193376e-06  , 1.40550933e-07,  -1.40550933e-07,
                   8.78193376e-06,  -1.34078963e-03 ,  1.34078963e-03,  -8.78193376e-06,
                   1.40550933e-07,  -1.40550933e-07,   8.78193376e-06,  -1.34078963e-03,
                  -1.10662060e-01,   2.55677958e-03,  -1.30119170e-07 ,  1.30119170e-07,
                  -2.55677958e-03,   1.10662060e-01]

    # i had to flip m and n to get the H_i fourier to match up with the numerics.

    # ORIGINAL (MARCH 14TH 2017 11:51PM)
    n = [ 1,  2,  3, -3, -2, -1,  1,  2,  3, -3, -2, -1,  1,  2,  3, -3, -2, -1,  1,  2,  3, -3, -2, -1,  1,
          2,  3, -3, -2, -1]
    m = [ 0, 0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2, -2, -2, -2, -2, -2, -2, -1,
          -1, -1, -1, -1, -1]

    #m = [ 1,  2,  3, -3, -2, -1,  1,  2,  3, -3, -2, -1,  1,  2,  3, -3, -2, -1,  1,  2,  3, -3, -2, -1,  1,
    #      2,  3, -3, -2, -1]
    #n = [ 0, 0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2, -2, -2, -2, -2, -2, -2, -1,
    #      -1, -1, -1, -1, -1]
    
    if False:
        for i in range(len(coeff_im)):
            print i, '&', coeff_im[i], '& ('+str(n[i])+', '+str(m[i])+')'


    tot = 0
    tot2 = 0

    a = pi
    if d == False:
        #return np.sum(-coeff_im*sin(x*idx_im[:,1] + y*idx_im[:,0]))/(64*64)

        for i in range(len(coeff_im)):
            tot += -coeff_im[i]*sin((x+a)*n[i] + (y+a)*m[i])
            #print 
        return tot

    else:
        for i in range(len(coeff_im)):
            # dh_1/dx
            tot += -coeff_im[i]*n[i]*cos((x+a)*n[i] + (y+a)*m[i])
            
            # dh_1/dy
            tot2 += -coeff_im[i]*m[i]*cos((x+a)*n[i] + (y+a)*m[i])

        if h2_flag:
            # if return derivative of h2, flip order
            return tot2,tot
        else:
            return tot,tot2




def H2_fourier(x,y,d=False):
    return H1_fourier(y,x,d=d,h2_flag=True)


def H1_fourier_centered(x,y,d=False,h2_flag=False):
    """
    H_i function for 2d phase model, i=1,2
    2pi periodic. let domain be on -pi,pi.
    coeff list gathered from data using 64x64 grid.
    coeff list normalized manually at the end. Keep this in mind!!!
    
    """

    # from N = 64

    # gives excellent approximation 26 raw coeff (some freq and coeff are repeated)

    # these parameters are for se=1,si=2
    coeff_im = [  -0.299041640592 , -0.0123427222227 , -2.92404662557e-07 ,
                  2.92404662711e-07 , 0.0123427222227 , 0.299041640592 ,
                  -0.110662059947 , -0.00255677958311 , -1.30119169782e-07 ,
                  1.30119169839e-07 , 0.00255677958311 , 0.110662059947 ,
                  -0.00134078962566 , -8.78193375763e-06 , -1.40550932909e-07 ,
                  1.40550932908e-07 , 8.78193375764e-06 , 0.00134078962566 ,
                  -0.00134078962566 , -8.78193375764e-06 , -1.40550932907e-07 ,
                  1.4055093291e-07 , 8.78193375763e-06 , 0.00134078962566 ,
                  -0.110662059947 , -0.00255677958311 , -1.30119169783e-07 ,
                  1.30119169839e-07 , 0.00255677958311 , 0.110662059947]

    # i had to flip m and n to get the H_i fourier to match up with the numerics.
    n = [1 , 2 , 3 ,-3 ,-2 ,-1 , 1 , 2 , 3 ,-3 ,-2 ,-1 , 1 , 2 , 3 ,-3 ,-2 ,-1 , 1 , 2 , 3 ,-3 ,-2 ,-1 , 1 , 2 , 3 ,-3 ,-2 ,-1]
    m = [0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 2 , 2 , 2 ,-2 ,-2 ,-2 ,-2 ,-2 ,-2 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1]
    
    if False:
        for i in range(len(coeff_im)):
            print i, '&', coeff_im[i], '& ('+str(n[i])+', '+str(m[i])+')'

    tot = 0
    tot2 = 0

    a = pi
    if d == False:
        #return np.sum(-coeff_im*sin(x*idx_im[:,1] + y*idx_im[:,0]))/(64*64)

        for i in range(len(coeff_im)):
            tot += -coeff_im[i]*sin(x*n[i] + y*m[i])
            #print 
        return tot

    else:
        for i in range(len(coeff_im)):
            # dh_1/dx
            tot += -coeff_im[i]*n[i]*cos(x*n[i] + y*m[i])
            
            # dh_1/dy
            tot2 += -coeff_im[i]*m[i]*cos(x*n[i] + y*m[i])

        if h2_flag:
            # if return derivative of h2, flip order
            return tot2,tot
        else:
            return tot,tot2


def H2_fourier_centered(x,y,d=False):
    return H1_fourier_centered(y,x,d=d,h2_flag=True)


def K_fourier(x,y):
    """
    difference of gaussians,
    sige = 2, sigi = 3
    """
    a00 = -18.71062565
    a01 = -7.56494081
    a10 = -7.56494081
    a11 = 4.36298017
    a_neg11 = 4.38344369

    return (a00 + 2*a01*cos(y+pi) +\
            2*a10*cos(x+pi) +\
            4*a11*cos(x+pi)*cos(y+pi))/(65*65)
    """
    return (a00 + 2*a01*cos(y+pi) +\
            2*a10*cos(x+pi) +\
            2*a11*cos(x+y+2*pi) + \
            2*a11*cos(x-y))/(65*65)
    """

def extract_coeff(f_coeffs,threshold=5e-3,re_or_im=None,return_data=False):
    """
    f_coeffs: full, untruncated coefficients. should be same size as original,
    discretized domain
    threshold: amplitude cutoff
    get relevant coefficients and extract corresponding indices

    todo: consider adding feature to return only relevant values of f_coeffs, 
    paired with tuples of idx_re and idx_im coordinates. reduces array sizes
    """

    nx,ny = np.shape(f_coeffs)

    idx_re = []
    idx_im = []

    # get coefficients above threshold
    a = np.real(f_coeffs)/(nx*ny)
    b = np.imag(f_coeffs)/(nx*ny)
    large_coeff_re = np.abs(a)>threshold
    large_coeff_im = np.abs(b)>threshold
    
    #print large_coeff_im

    #a[1-large_coeff_re] = 0
    #b[1-large_coeff_im] = 0

    # save coefficient indices to list
    if re_or_im == None:
        for i in range(nx):

            for j in range(ny):
                if large_coeff_re[i,j] == 1:
                    if i > 32:
                        i -= 64
                    if j > 32:
                        j -= 64
                    idx_re.append((i,j))
                if large_coeff_im[i,j] == 1:
                    if i > 32:
                        i -= 64
                    if j > 32:
                        j -= 64

                    idx_im.append((i,j))

        return a,b,idx_re, idx_im

    elif re_or_im == "re":
        idx_re_n = []
        idx_re_m = []
        idx_re_coeff = []
        for n in range(nx):
            for m in range(ny):
                if large_coeff_re[n,m] == 1:
                    idx_re_coeff.append(a[n,m])
                    idx_re_n.append(n)
                    idx_im_n.append(m)

        return np.array(idx_re_coeff),np.array(idx_re_n),np.array(idx_re_m)

    elif re_or_im == "im":
        idx_im_n = []
        idx_im_m = []
        idx_im_coeff = []
        for n in range(nx):
            for m in range(ny):
                if large_coeff_im[n,m] == 1:
                    idx_im_coeff.append(b[n,m])
                    idx_im_n.append(n)
                    idx_im_m.append(m)
        idx_im_n = np.array(idx_im_n)
        idx_im_m = np.array(idx_im_m)

        idx_im_n[idx_im_n>32] -= 64
        idx_im_m[idx_im_m>32] -= 64
        
        return np.array(idx_im_coeff),idx_im_n,idx_im_m


def extract_coeff_r(f_coeffs,threshold=1.,re_or_im=None,return_1d=False):
    """
    no normalization of coefficients
    f_coeffs: full, untruncated coefficients. should be same size as original,
    discretized domain
    threshold: amplitude cutoff
    get relevant coefficients and extract corresponding indices

    todo: consider adding feature to return only relevant values of f_coeffs, 
    paired with tuples of idx_re and idx_im coordinates. reduces array sizes
    """

    nx,ny = np.shape(f_coeffs)

    idx_re = []
    idx_im = []
    a_1d = []
    b_1d = []

    # get coefficients above threshold
    a = np.real(f_coeffs)
    b = np.imag(f_coeffs)

    large_coeff_re = np.abs(a)>=threshold
    large_coeff_im = np.abs(b)>=threshold

    large_coeff_re_complement = np.abs(a)<threshold
    large_coeff_im_complement = np.abs(b)<threshold

    a[large_coeff_re_complement] = 0
    b[large_coeff_im_complement] = 0
    
    #a[1-large_coeff_re] = 0
    #b[1-large_coeff_im] = 0

    # save coefficient indices to list
    if re_or_im == None:
        for i in range(nx):

            for j in range(ny):
                if (large_coeff_re[i,j] >= 1):
                    if i > 32:
                        i -= 64
                    if j > 32:
                        j -= 64
                    idx_re.append((i,j))
                    a_1d.append(a[i,j])

                if (large_coeff_im[i,j] >= 1):
                    if i > 32:
                        i -= 64
                    if j > 32:
                        j -= 64
                    idx_im.append((i,j))
                    b_1d.append(a[i,j])

        if return_1d:
            return a,b,a_1d,b_1d,idx_re, idx_im
        else:
            return a,b,idx_re, idx_im

    elif re_or_im == "re":
        idx_re_n = []
        idx_re_m = []
        idx_re_coeff = []
        for n in range(nx):
            for m in range(ny):
                if large_coeff_re[n,m] == 1:
                    idx_re_coeff.append(a[n,m])
                    idx_re_n.append(n)
                    idx_im_n.append(m)

        return np.array(idx_re_coeff),np.array(idx_re_n),np.array(idx_re_m)

    elif re_or_im == "im":
        idx_im_n = []
        idx_im_m = []
        idx_im_coeff = []
        for n in range(nx):
            for m in range(ny):
                if large_coeff_im[n,m] == 1:
                    idx_im_coeff.append(b[n,m])
                    idx_im_n.append(n)
                    idx_im_m.append(m)
        idx_im_n = np.array(idx_im_n)
        idx_im_m = np.array(idx_im_m)

        idx_im_n[idx_im_n>32] -= 64
        idx_im_m[idx_im_m>32] -= 64
        
        return np.array(idx_im_coeff),idx_im_n,idx_im_m



def extract_coeff_all(f_coeffs,threshold=1.,return_1d=False):
    """
    no normalization of coefficients
    f_coeffs: full, untruncated coefficients. should be same size as original,
    discretized domain
    threshold: amplitude cutoff
    get relevant coefficients and extract corresponding indices

    todo: consider adding feature to return only relevant values of f_coeffs, 
    paired with tuples of idx_re and idx_im coordinates. reduces array sizes
    """

    nx,ny = np.shape(f_coeffs)

    idx = []
    c_1d = []

    # get coefficients above threshold

    large_coeff = np.abs(f_coeffs)>=threshold
    large_coeff_complement = np.abs(f_coeffs)<threshold

    f_coeffs[large_coeff_complement] = 0

    # save coefficient indices to list
    for i in range(nx):        
        for j in range(ny):
            if (large_coeff[i,j] == True):
                if i > 32:
                    i -= nx
                if j > 32:
                    j -= ny
                idx.append((i,j))
                c_1d.append(f_coeffs[i,j])

    if return_1d:
        return f_coeffs,c_1d,idx
    else:
        return f_coeffs,idx


def dft2(f):
    """
    manual 2d dft
    nx by ny array f
    """
    nx,ny = np.shape(f)
    F = np.zeros((nx,ny),dtype=complex)
    for x in range(nx):
        for y in range(ny):
            tot = 0
            for i in range(nx):
                for j in range(ny):
                    tot += exp(-1j*((2*pi/nx)*x*i + (2*pi/ny)*y*j) )*f[i,j]
            F[x,y] = tot
    return F

def idft2(F,idx=None):
    """
    manual 2d idft
    nx by ny array F of fourier coefficients straight from fft2.
    idx = corresponding index terms (if performing truncation)
    """
    nx,ny = np.shape(F)
    
    f = np.zeros((nx,ny),dtype=complex)
    for x in range(nx):
        #print x
        for y in range(ny):
            tot = 0
            if idx == None:
                for i in range(nx):
                    for j in range(nx):
                        tot += F[i,j]*exp( 1j*((2*pi/nx)*x*i + (2*pi/ny)*y*j) )
            else:
                for (i,j) in idx:
                    tot += F[i,j]*exp( 1j*((2*pi/nx)*x*i + (2*pi/ny)*y*j) )
            f[x,y] = tot
    f /= nx*ny
    return f

def idft2_sin(F,idx_re,idx_im):
    """
    requre F "sparse"
    manual 2d idft, using compressed F
    nx by ny array a
    """
    a = np.real(F)
    b = np.imag(F)
    nx,ny = np.shape(F)

    f = np.zeros((nx,ny))
    
    xx = np.linspace(0,2*pi*(1.-1./nx),nx)
    yy = np.linspace(0,2*pi*(1.-1./ny),ny)

    n = 0
    m = 0
    for x in xx:
        m = 0
        for y in yy:
            tot = 0
            for i,j in idx_im:
                tot += -b[i,j]*sin(x*i + y*j)
            f[n,m] = tot
            m += 1
        n += 1

    f /= nx*ny    
    return f


def idft2_cos(F,idx):
    """
    requre F "sparse"
    manual 2d idft, using compressed F
    nx by ny array a
    """
    nx,ny = np.shape(F)

    a = np.real(F)

    f = np.zeros((nx,ny))

    for x in range(nx):
        #print x
        for y in range(ny):
            tot = 0
            if idx == None:
                for i in range(nx):
                    for j in range(nx):
                        tot += a[i,j]*cos( (2*pi/nx)*x*i + (2*pi/ny)*y*j )
            else:
                for (i,j) in idx:
                    tot += a[i,j]*cos( (2*pi/nx)*x*i + (2*pi/ny)*y*j )
                    #tot += F[i,j]*exp( 1j*((2*pi/nx)*x*i + (2*pi/ny)*y*j) )
            f[x,y] = tot
    f /= nx*ny
    return f




def idft2_trig(F,idx_re,idx_im):
    """
    full
    requre F "sparse"
    manual 2d idft, using compressed F
    nx by ny array a
    """
    
    """
    fig1 = plt.figure()
    nx,ny = np.shape(F)    
    X, Y = np.meshgrid(np.linspace(-pi,pi,nx),
                       np.linspace(-pi,pi,ny))

    ax1 = fig1.gca(projection='3d')
    approx = np.fft.ifft2(F)
    ax1.set_title('approx kernel test')
    ax1.plot_surface(X,Y,approx)
    """
    
    a = np.real(F)
    b = np.imag(F)
    nx,ny = np.shape(F)
    f = np.zeros((nx,ny))

    
    xx = np.linspace(0,2*pi*(1.-1./nx),nx)
    yy = np.linspace(0,2*pi*(1.-1./ny),ny)

    print nx,ny

    for n in range(len(xx)):
        for m in range(len(yy)):
            tot = 0
            for i,j in idx_re:
                tot += a[i,j]*cos(xx[n]*i)*cos(yy[m]*j)
            for i,j in idx_im:
                tot += -b[i,j]*sin(xx[n]*i)*sin(yy[m]*j)
                
            
            f[n,m] = tot#cos(xx[n]+yy[m])#tot

    f /= nx*ny
    
    return f

def gaussian_2d(XX,YY,dx=False,dy=False):
    """
    2d gaussian function
    """
    if not(dx) and not(dy):
        # no derivative
        return exp(-(XX**2+YY**2))

    elif not(dy) and dx:
        # x derivative
        return -XX*exp(-(XX**2+YY**2))

    elif not(dx) and dy:
        # y derivative
        return -YY*exp(-(XX**2+YY**2))

    else:
        raise Exception('dx=True, dy=True not implemented')


def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = np.perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def f1(nu,a,n,m,g,th):
    tot1 = 0
    for i in range(len(a)):
        num1 = -(cos(th)*n[i]+sin(th)*m[i])
        denom1 = 1 + (nu*(cos(th)*n[i]+sin(th)*m[i]))**2.
        tot1 += g*a[i]*num1/denom1
    tot1 += cos(th)
    return tot1
    
def f2(nu,a,n,m,g,th):
    tot2 = 0
    for i in range(len(a)):
        num2 = -(cos(th)*m[i]+sin(th)*n[i])
        denom2 = 1 + (nu*(cos(th)*m[i]+sin(th)*n[i]))**2.
        tot2 += g*a[i]*num2/denom2
    tot2 += sin(th)
    return tot2

def B(nu,th,a,n,m):
    tot2 = 0
    for i in range(len(a)):
        num2 = (cos(th)*n[i]+sin(th)*m[i])
        denom2 = 1 + (nu*(cos(th)*n[i]+sin(th)*m[i]))**2.
        tot2 += a[i]*num2/denom2
    return tot2


def C(nu,th,a,n,m):
    tot2 = 0
    for i in range(len(a)):
        num2 = (cos(th)*m[i]+sin(th)*n[i])
        denom2 = 1 + (nu*(cos(th)*m[i]+sin(th)*n[i]))**2.
        tot2 += a[i]*num2/denom2
    return tot2


def B1(nu,th,a,n,m):
    tot2 = 0
    for i in range(len(a)):
        num = a[i]*(n[i]+m[i])
        denom2 = 1 + (nu*(cos(th)*n[i]+sin(th)*m[i]))**2.
        tot2 += num/denom2
    return tot2


def sum_anm_m(nu,th,a,n,m):
    tot2 = 0
    for i in range(len(a)):
        num2 = a[i]*m[i]
        denom2 = 1 + (nu*(cos(th)*n[i]+sin(th)*m[i]))**2.
        tot2 += num2/denom2
    return tot2


def sum_amn_m(nu,th,a,n,m):
    tot2 = 0
    for i in range(len(a)):
        num2 = a[i]*m[i]
        denom2 = 1 + (nu*(cos(th)*m[i]+sin(th)*n[i]))**2.
        tot2 += num2/denom2
    return tot2



def C1(nu,th,a,n,m):
    tot2 = 0
    for i in range(len(a)):
        denom2 = 1 + (nu*(cos(th)*m[i]+sin(th)*n[i]))**2.
        tot2 += n[i]*a[i]/denom2
    return tot2

def B1test(nu,th,a,n,m):
    tot2 = 0
    for i in range(len(a)):
        denom2 = 1 + (nu*(cos(th)*n[i]+sin(th)*m[i]))**2.
        tot2 += m[i]*a[i]/denom2
    return tot2

def C1test(nu,th,a,n,m):
    tot2 = 0
    for i in range(len(a)):
        denom2 = 1 + (nu*(cos(th)*m[i]+sin(th)*n[i]))**2.
        tot2 += m[i]*a[i]/denom2
        #print 'term', i, 'num=',m[i]*a[i], '(n[i],m[i]) =', n[i],',',m[i], 'denom=',denom2[20:22]
    return tot2


def main():
    
    
    # the computed H functions are missing a zero function value.

    dat = twod.Phase(recompute_h=False,recompute_j=False)
    H1,H2 = dat.H1,dat.H2
    J1,J2 = dat.J1,dat.J2


    if False:
        fig22 = plt.figure()

        N,N = H1.shape
        x = np.linspace(-pi,pi,N)
        XX,YY = np.meshgrid(x,x)
        ax22 = fig22.gca(projection='3d')
        #ax.set_zlim(-5,10)
        ax22.set_title("approx")
        ax22.plot_surface(XX,YY,H1,
                          rstride=2,
                          edgecolors="k",
                          cstride=2,
                          cmap="gray",
                          alpha=0.8,
                          linewidth=0.25)

        plt.show()



    #H1,H2 = p2d.H_i(u0ss,df_u0b,ux,uy,X,params)
    #Nx,Ny = np.shape(H1t)
    #H1 = np.zeros((Nx,Ny+1))
    #H1[:,:-1] = H1t
    
    f_coeffs_H1 = np.fft.fft2(np.roll(np.roll(H1,int(dat.N/2),axis=-1),int(dat.N/2),axis=-2))

    #print '(0,1),(0,-1)',f_coeffs_H1[0,1],f_coeffs_H1[0,-1+64]
    #print '(1,0),(-1,0)',f_coeffs_H1[1,0],f_coeffs_H1[-1+64,0]
    #idx_re,idx_im = extract_coeff(f_coeffs_H1,threshold=1e-6)
    #print idx_re,idx_im

    if True:
        a,m,n = extract_coeff(f_coeffs_H1,threshold=1e-7,re_or_im="im")

    for i in range(len(a)):
        print i,' & ',a[i], ' & ', '('+str(n[i])+','+str(m[i])+')\\\\'

    #print a,m,n

    #print 'sum ak nk =', np.sum(a*n), 'B(0,0) =', B(0,0,a,n,m)
    #print 'sum ak nk =', np.sum(a*n), 'C(0,pi/2.) =', C(0,pi/2.,a,n,m)


    """
    th_vals = np.array([pi/2.])
    th_vals = np.linspace(0,2*pi,81)
    f_vals = np.zeros(len(th_vals))
    j = 0
    nu = np.linspace(0,30,5001)
    #g = 15

    lslist = ['-.','--','-']
    thlist = [pi/3.,pi/2.5,pi/2.01]

    fig = plt.figure()
    ax = fig.add_subplot(121)
    i = 0
    for th in thlist:
        ax.plot(nu,B(nu,th,a,n,m)/cos(th),color='blue',ls=lslist[i],label=str(th))
        ax.plot(nu,C(nu,th,a,n,m)/sin(th),color='red',ls=lslist[i])
        i += 1

    
    ax.set_xlabel('nu')
    ax.set_ylabel('g')
    ax.legend()
    #ax.set_ylim(-1,10)


    thlist = [.5,.1,.01]

    ax2 = fig.add_subplot(122)
    i = 0
    for th in thlist:
        ax2.plot(nu,B(nu,th,a,n,m)/cos(th),color='blue',ls=lslist[i],label=str(th))
        ax2.plot(nu,C(nu,th,a,n,m)/sin(th),color='red',ls=lslist[i])
        i += 1

    ax2.set_xlabel('nu')
    ax2.set_ylabel('g')
    ax2.legend()
    #ax2.set_ylim(-1,10)

    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111)
    #ax3.plot(cos(0)-8*B(nu,0,a,n,m))
    #ax3.plot(sin(0)-8*C(nu,0,a,n,m))

    #ax3.plot(B(nu,0,a,n,m))
    #ax3.plot(C(nu,0,a,n,m))

    
    gg = 3.
    pp = pi/2.

    print np.linalg.norm(sum_anm_m(nu,pp,a,n,m))
    print np.linalg.norm(sum_amn_m(nu,pp,a,n,m))
    #ax3.plot(nu,f1(nu,a,n,m,gg,pp),color='k')
    #ax3.plot(nu,f2(nu,a,n,m,3,pi/3.),color='k')

    #ax3.plot(nu,cos(pp)*(1-gg*B1(nu,pp,a,n,m)),color='r')
    #ax3.plot(nu,sin(pp)*(1-gg*C1(nu,pp,a,n,m)),color='r')

    #ax3.plot(nu,B1test(nu,pp,a,n,m))
    #ax3.plot(nu,C1test(nu,pp,a,n,m))
    
    print 'B()',
    """



    """
    th=0.
    mp.figure()
    #mp.plot(nu,B(nu,th,a,n,m)/cos(th),color='blue',ls='-.')
    mp.plot(nu,C(nu,th,a,n,m),color='blue',ls='-.')


    mp.show()

    for th in th_vals:
        
        
        tot1 = f1(nu,a,n,m,g,th)
        tot2 = f2(nu,a,n,m,g,th)
        diff = tot1-tot2
        norm = np.linalg.norm(tot1-tot2)

        crossing_idx_up = (diff[:-1]>0)*(diff[1:]<=0)
        crossing_idx_down = (diff[:-1]<=0)*(diff[1:]>0)

        crossing_value_up = tot1[crossing_idx_up]
        crossing_value_down = tot1[crossing_idx_down]

        #print crossing_value_up,crossing_value_down,th

        # if crossing_value != [], get interpolation
        # else, assign nan.

        if (np.size(crossing_value_up) == 0) and\
           (np.size(crossing_value_down) == 0):

            f_vals[j] = np.nan
        else:
            if np.size(crossing_value_up) == 1:
                f_vals[j] = crossing_value_up[0]
            elif np.size(crossing_value_down) == 1:
                f_vals[j] = crossing_value_down[0]
            elif (np.size(crossing_value_up) >= 2) or\
                 (np.size(crossing_value_down) >= 2):
                f_vals[j] = 0.
        
        j += 1

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    #mp.plot(nu,tot1)
    #mp.plot(nu,tot2)
    r = np.sqrt(th_vals**2 + f_vals**2)
    polar_angle = np.arctan2(f_vals,th_vals)
    ax1.plot((f_vals+1)*cos(th_vals),(f_vals+1)*sin(th_vals),lw=2)
    ax1.plot(cos(th_vals),sin(th_vals),color='k')
    ax1.plot([-1.2,1.2],[0,0],color='k')
    ax1.plot([0,0],[-1.2,1.2],color='k')
    ax1.plot([-1.2,1.2],[-1.2,1.2],color='k')
    ax1.plot([-1.2,1.2],[1.2,-1.2],color='k')

    ax2 = fig.add_subplot(122)
    ax2.plot(th_vals,f_vals)
    ax2.plot([0,2*pi],[0,0])
    
    #mp.xlim(0,2*pi)
    mp.xlabel('theta')


    mp.show()

    print 'check impl. fun. thm.', 
    print "# of coeff re,im", len(idx_re),len(idx_im)
    
    print 'coefficient values im'
    coeff_im = np.zeros(len(idx_im))
    k = 0
    for i,j in idx_im:
        #print "("+str(i)+","+str(j)+")",np.imag(f_coeffs_H1)[i,j]
        #print np.imag(f_coeffs_H1)[i,j]
        coeff_im[k] = np.imag(f_coeffs_H1)[i,j]
        k += 1
    

    XX,YY = np.meshgrid(np.linspace(-pi,pi*(1-2./N),N),
                        np.linspace(-pi,pi*(1-2./N),N),indexing='ij')
    AA,BB = np.meshgrid(np.linspace(0,2*pi*(1-2./N),N),
                        np.linspace(0,2*pi*(1-2./N),N),indexing='ij')
    XX3,YY3 = np.meshgrid(np.linspace(-pi,pi,32),
                          np.linspace(-pi,pi,32),indexing='ij')
    XX3t,YY3t = np.meshgrid(np.linspace(0,1,50),
                            np.linspace(-pi,0,50),indexing='ij')
    """

    if True:

        N,N = H1.shape
        x = np.linspace(-pi,pi,N)
        XX,YY = np.meshgrid(x,x)


        fig3 = plt.figure()
        ax3 = fig3.gca(projection='3d')
        ax3.set_title("H1 approx")

        ax3.plot_surface(XX,YY,H1_fourier_centered(XX,YY),
                         rstride=2,
                         edgecolors="k",
                         cstride=2,
                         cmap="gray",
                         alpha=0.8,
                         linewidth=0.25)
        """
        fig4 = plt.figure()
        ax4 = fig4.gca(projection='3d')
        ax4.set_title("H1")

        ax4.plot_surface(XX,YY,H1,
                         rstride=2,
                         edgecolors="k",
                         cstride=2,
                         cmap="gray",
                         alpha=0.8,
                         linewidth=0.25)
        """


        plt.show()


if __name__ == "__main__":
    main()
