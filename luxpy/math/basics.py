# -*- coding: utf-8 -*-
########################################################################
# <LUXPY: a Python package for lighting and color science.>
# Copyright (C) <2017>  <Kevin A.G. Smet> (ksmet1977 at gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#########################################################################
"""
Module with useful basic math functions
=======================================

 :normalize_3x3_matrix(): Normalize 3x3 matrix M to xyz0 -- > [1,1,1]

 :line_intersect(): | Line intersections of series of two line segments a and b. 
                    | https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

 :positive_arctan(): Calculates the positive angle (0°-360° or 0 - 2*pi rad.) 
                     from x and y.

 :dot23(): Dot product of a 2-d ndarray 
           with a (N x K x L) 3-d ndarray using einsum().

 :check_symmetric(): Checks if A is symmetric.

 :check_posdef(): Checks positive definiteness of a matrix via Cholesky.

 :symmM_to_posdefM(): | Converts a symmetric matrix to a positive definite one. 
                      | Two methods are supported:
                      |    * 'make': A Python/Numpy port of Muhammad Asim Mubeen's
                      |              matlab function Spd_Mat.m 
                      |       (https://nl.mathworks.com/matlabcentral/fileexchange/45873-positive-definite-matrix)
                      |    * 'nearest': A Python/Numpy port of John D'Errico's 
                      |                'nearestSPD' MATLAB code. 
                      |        (https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite)

 :bvgpdf(): Evaluate bivariate Gaussian probability density function (BVGPDF) 
            at (x,y) with center mu and inverse covariance matric, sigmainv.

 :mahalanobis2(): Evaluate the squared mahalanobis distance with center mu and 
                  shape and orientation determined by sigmainv. 

 :rms(): Calculates root-mean-square along axis.

 :geomean(): Calculates geometric mean along axis.

 :polyarea(): | Calculates area of polygon. 
              | (First coordinate should also be last)

 :erf(), erfinv(): erf-function and its inverse, direct import from scipy.special

 :cart2pol(): Converts Cartesian to polar coordinates.

 :pol2cart(): Converts polar to Cartesian coordinates.
 
 :cart2spher(): Converts Cartesian to spherical coordinates.
 
 :spher2cart(): Converts spherical to Cartesian coordinates.

 :magnitude_v():  Calculates magnitude of vector.

 :angle_v1v2():  Calculates angle between two vectors.

 :histogram(): | Histogram function that can take as bins either the center
               | (cfr. matlab hist) or bin-edges.

 :v_to_cik(): Calculate 2x2 '(covariance matrix)^-1' elements cik from v-format ellipse descriptor.

 :cik_to_v(): Calculate v-format ellipse descriptor from 2x2 'covariance matrix'^-1 cik.
 
 :fmod(): Floating point modulus, e.g.: fmod(theta, np.pi * 2) would keep an angle in [0, 2pi]b
 
 :remove_outliers(): Remove multivariate outliers from data when outside of alpha-level confidence ellipsoid.

 :fit_ellipse(): Fit an ellipse to supplied data points.
 
 :fit_cov_ellipse(): Fit an covariance ellipse to supplied data points.
 
 :interp1_sprague5(): Perform a 1-dimensional 5th order Sprague interpolation.
 
 :interp1(): Perform a 1-dimensional linear interpolation (wrapper around scipy.interpolate.InterpolatedUnivariateSpline).
 
 :ndinterp1(): Perform n-dimensional interpolation using Delaunay triangulation.
 
 :ndinterp1_scipy(): Perform n-dimensional interpolation using Delaunay triangulation (wrapper around scipy.interpolate.LinearNDInterpolator)
 
 :box_m(): Performs a Box M test on covariance matrices.
 
 :pitman_morgan(): Pitman-Morgan Test for the difference between correlated variances with paired samples.
     
 :stress(): Calculate STandardize-Residual-Sum-of-Squares (STRESS)
 
 :stress_F_test(): Perform F-test on significance of difference between STRESS A and STRESS B.
 
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
===============================================================================
"""

from luxpy.utils import np, sp, np2d, _EPS, asplit
from scipy.special import erf, erfinv
from scipy import stats
from scipy.interpolate import interp1d 

__all__  = ['normalize_3x3_matrix','symmM_to_posdefM','check_symmetric',
            'check_posdef','positive_arctan','line_intersect','erf', 'erfinv', 
            'histogram', 'pol2cart', 'cart2pol', 'spher2cart', 'cart2spher']
__all__ += ['bvgpdf','mahalanobis2','dot23', 'rms','geomean','polyarea']
__all__ += ['magnitude_v','angle_v1v2']
__all__ += ['v_to_cik', 'cik_to_v', 'fmod', 'remove_outliers','fit_ellipse','fit_cov_ellipse']
__all__ += ['in_hull','interp1_sprague5','interp1', 'ndinterp1','ndinterp1_scipy']
__all__ += ['box_m','pitman_morgan', 'stress','stress_F_test']


#------------------------------------------------------------------------------
def normalize_3x3_matrix(M, xyz0 = np.array([[1.0,1.0,1.0]])):
    """
    Normalize 3x3 matrix M to xyz0 -- > [1,1,1]
    
    | If M.shape == (1,9): M is reshaped to (3,3)
    
    Args:
        :M: 
            | ndarray((3,3) or ndarray((1,9))
        :xyz0: 
            | 2darray, optional 
        
    Returns:
        :returns: 
            | normalized matrix such that M*xyz0 = [1,1,1]
    """
    M = np2d(M)
    if M.shape[-1]==9:
        M = M.reshape(3,3)
    if xyz0.shape[0] == 1:
        return np.dot(np.diagflat(1/(np.dot(M,xyz0.T))),M)
    else:
        return np.concatenate([np.dot(np.diagflat(1/(np.dot(M,xyz0[1].T))),M) for i in range(xyz0.shape[0])],axis=0).reshape(xyz0.shape[0],3,3)

#------------------------------------------------------------------------------
def line_intersect(a1, a2, b1, b2):
    """
    Line intersections of series of two line segments a and b. 
        
    Args:
        :a1: 
            | ndarray (.shape  = (N,2)) specifying end-point 1 of line a
        :a2: 
            | ndarray (.shape  = (N,2)) specifying end-point 2 of line a
        :b1: 
            | ndarray (.shape  = (N,2)) specifying end-point 1 of line b
        :b2: 
            | ndarray (.shape  = (N,2)) specifying end-point 2 of line b
    
    Note: 
        N is the number of line segments a and b.
    
    Returns:
        :returns: 
            | ndarray with line-intersections (.shape = (N,2))
    
    References:
        1. https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    """
    T = np.array([[0.0, -1.0], [1.0, 0.0]])
    da = np.atleast_2d(a2 - a1)
    db = np.atleast_2d(b2 - b1)
    dp = np.atleast_2d(a1 - b1)
    dap = np.dot(da, T)
    denom = np.sum(dap * db, axis=1)
    num = np.sum(dap * dp, axis=1)
    return np.atleast_2d(num / denom).T * db + b1

#------------------------------------------------------------------------------
def positive_arctan(x,y, htype = 'deg'):
    """
    Calculate positive angle (0°-360° or 0 - 2*pi rad.) from x and y.
    
    Args:
        :x: 
            | ndarray of x-coordinates
        :y: 
            | ndarray of y-coordinates
        :htype:
            | 'deg' or 'rad', optional
            |   - 'deg': hue angle between 0° and 360°
            |   - 'rad': hue angle between 0 and 2pi radians
    
    Returns:
        :returns:
            | ndarray of positive angles.
    """
    if htype == 'deg':
        r2d = 180.0/np.pi
        h360 = 360.0
    else:
        r2d = 1.0
        h360 = 2.0*np.pi
    h = np.atleast_1d((np.arctan2(y,x)*r2d))
    h[np.where(h<0)] = h[np.where(h<0)] + h360
    return h


#------------------------------------------------------------------------------
def dot23(A,B, keepdims = False):
    """
    Dot product of a 2-d ndarray with a (N x K x L) 3-d ndarray 
    using einsum().
    
    Args:
        :A: 
            | ndarray (.shape = (M,N))
        :B: 
            | ndarray (.shape = (N,K,L))
        
    Returns:
        :returns: 
            | ndarray (.shape = (M,K,L))
    """
    if (len(A.shape)==2) & (len(B.shape)==3):
        dotAB = np.einsum('ij,jkl->ikl',A,B)
        if (len(B.shape)==3) & (keepdims == True):
            dotAB = np.expand_dims(dotAB,axis=1)
    elif (len(A.shape)==2) & (len(B.shape)==2):
        dotAB = np.einsum('ij,jk->ik',A,B)
        if (len(B.shape)==2) & (keepdims == True):
            dotAB = np.expand_dims(dotAB,axis=1)
            
    return dotAB

#------------------------------------------------------------------------------
def check_symmetric(A, atol = 1.0e-9, rtol = 1.0e-9):
    """
    Check if A is symmetric.
    
    Args:
        :A: 
            | ndarray
        :atol:
            | float, optional
            | The absolute tolerance parameter (see Notes of numpy.allclose())
        :rtol:
            | float, optional
            | The relative tolerance parameter (see Notes of numpy.allclose())
    
    Returns:
        :returns:
            | Bool
            | True: the array is symmetric within the given tolerance
    """
    return np.allclose(A, A.T, atol = atol, rtol = rtol)


def check_posdef(A, atol = 1.0e-9, rtol = 1.0e-9):
    """
    Checks positive definiteness of a matrix via Cholesky.
    
    Args:
        :A: 
            | ndarray
        :atol:
            | float, optional
            | The absolute tolerance parameter (see Notes of numpy.allclose())
        :rtol:
            | float, optional
            | The relative tolerance parameter (see Notes of numpy.allclose())
    
    Returns:
        :returns:
            | Bool
            | True: the array is positive-definite within the given tolerance

    """
    try:
        R = np.linalg.cholesky(A)
        if np.allclose(A, np.dot(R,R.T), atol = atol,rtol = rtol):
            return True
        else:
            return False
    except np.linalg.LinAlgError:
        return False


def symmM_to_posdefM(A = None, atol = 1.0e-9, rtol = 1.0e-9, method = 'make', forcesymm = True):
    """
    Convert a symmetric matrix to a positive definite one. 
    
    Args:
        :A: 
            | ndarray
        :atol:
            | float, optional
            | The absolute tolerance parameter (see Notes of numpy.allclose())
        :rtol:
            | float, optional
            | The relative tolerance parameter (see Notes of numpy.allclose())
        :method: 
            | 'make' or 'nearest', optional (see notes for more info)
        :forcesymm: 
            | True or False, optional
            | If A is not symmetric, force symmetry using: 
            |    A = numpy.triu(A) + numpy.triu(A).T - numpy.diag(numpy.diag(A))
    
    Returns:
        :returns:
            | ndarray with positive-definite matrix.
        
    Notes on supported methods:
        1. `'make': A Python/Numpy port of Muhammad Asim Mubeen's matlab function 
        Spd_Mat.m 
        <https://nl.mathworks.com/matlabcentral/fileexchange/45873-positive-definite-matrix>`_
        2. `'nearest': A Python/Numpy port of John D'Errico's `nearestSPD` 
        MATLAB code. 
        <https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite>`_
    """
    if A is not None:
        A = np2d(A)
        
        
        # Make sure matrix A is symmetric up to a certain tolerance:
        sn = check_symmetric(A, atol = atol, rtol = rtol) 
        if ((A.shape[0] != A.shape[1]) | (sn != True)):
            if (forcesymm == True)  &  (A.shape[0] == A.shape[1]):
                A = np.triu(A) + np.triu(A).T - np.diag(np.diag(A))
            else:
                raise Exception('symmM_to_posdefM(): matrix A not symmetric.')
        
        
        if check_posdef(A, atol = atol, rtol = rtol) == True:
            return A
        else:

            if method == 'make':

                # A Python/Numpy port of Muhammad Asim Mubeen's matlab function Spd_Mat.m
                #
                # See: https://nl.mathworks.com/matlabcentral/fileexchange/45873-positive-definite-matrix
                Val, Vec = np.linalg.eig(A) 
                Val = np.real(Val)
                Vec = np.real(Vec)
                Val[np.where(Val==0)] = _EPS #making zero eigenvalues non-zero
                p = np.where(Val<0)
                Val[p] = -Val[p] #making negative eigenvalues positive
                return   np.dot(Vec,np.dot(np.diag(Val) , Vec.T))
 
            
            elif method == 'nearest':
                
                 # A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
                 # credits [2].
                 #
                 # [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
                 #
                 # [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
                 # matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
                 #
                 # See: https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
                
                B = (A + A.T) / 2.0
                _, s, V = np.linalg.svd(B)

                H = np.dot(V.T, np.dot(np.diag(s), V))

                A2 = (B + H) / 2.0

                A3 = (A2 + A2.T) / 2.0

                if check_posdef(A3, atol = atol, rtol = rtol) == True:
                    return A3

                spacing = np.spacing(np.linalg.norm(A))
                I = np.eye(A.shape[0])
                k = 1
                while not check_posdef(A3, atol = atol, rtol = rtol):
                    mineig = np.min(np.real(np.linalg.eigvals(A3)))
                    A3 += I * (-mineig * k**2.0+ spacing)
                    k += 1

                return A3


#-----------------------------------------------------------------------------
def bvgpdf(x, y = None, mu = None, sigmainv = None):
    """
    Evaluate bivariate Gaussian probability density function (BVGPDF)
    
    Args:
        :x: 
            | scalar or list or ndarray (.ndim = 1 or 2) with 
            | x(y)-coordinates at which to evaluate bivariate Gaussian PD.
        :y: 
            | None or scalar or list or ndarray (.ndim = 1) with 
            | y-coordinates at which to evaluate bivariate Gaussian PD, optional.
            | If :y: is None, :x: should be a 2d array.
        :mu: 
            | None or ndarray (.ndim = 2) with center coordinates of 
            | bivariate Gaussian PD, optional. 
            | None defaults to ndarray([0,0]).
        :sigmainv:
            | None or ndarray with 'inverse covariance matrix', optional 
            | Determines the shape and orientation of the PD.
            | None default to numpy.eye(2).
     
    Returns:
         :returns:
             | ndarray with magnitude of BVGPDF(x,y)   
    
    """
    return np.exp(-0.5*mahalanobis2(x, y = y, mu = mu, sigmainv = sigmainv))

#------------------------------------------------------------------------------
def mahalanobis2(x, y = None, z = None, mu = None, sigmainv = None):
    """
    Evaluate the squared mahalanobis distance
    
    Args: 
        :x: 
            | scalar or list or ndarray (.ndim = 1 or 2) with x(y)-coordinates 
              at which to evaluate the mahalanobis distance squared.
        :y: 
            | None or scalar or list or ndarray (.ndim = 1) with y-coordinates 
              at which to evaluate the mahalanobis distance squared, optional.
            | If :y: is None, :x: should be a 2d array.
        :z: 
            | None or scalar or list or ndarray (.ndim = 1) with z-coordinates 
              at which to evaluate the mahalanobis distance squared, optional.
            | If :z: is None & :y: is None, then :x: should be a 2d array.
        :mu: 
            | None or ndarray (.ndim = 1) with center coordinates of the 
              mahalanobis ellipse, optional. 
            | None defaults to zeros(2) or zeros(3).
        :sigmainv:
            | None or ndarray with 'inverse covariance matrix', optional 
            | Determines the shape and orientation of the PD.
            | None default to np.eye(2) or eye(3).
    Returns:
         :returns: 
             | ndarray with magnitude of mahalanobis2(x,y[,z])

    """
    if (y is None) & (z is None):
        p = x.shape[-1]
    elif (z is None):
        p = x.shape[-1] if (y is None) else 2
    elif (z is not None):
        p = 3 if (y is not None) else 2
    
    if mu is None:
        mu = np.zeros(p)
    if sigmainv is None:
        sigmainv = np.eye(p)
    
    x = np2d(x)
    mu = np2d(mu)

    if (y is None) & (z is None):
        x = x - mu
        if p == 2:
            x, y = asplit(x)
        elif p==3:
            x, y, z = asplit(x)
    elif (z is None):
        if y is None:
            x = x - mu
            x, y = asplit(x)
        else:
            x = x - mu[...,0] # center data on mu 
            y = np2d(y) - mu[...,1] # center data on mu 
    elif (z is not None):
        if (y is not None):
            x = x - mu[0] # center data on mu 
            y = np2d(y) - mu[...,1] # center data on mu 
            z = np2d(z) - mu[...,2] # center data on mu 
        else:
            x = x - mu[...,0] # center data on mu 
            y = np2d(z) - mu[...,1] # center data on mu 
            
    if p == 2:
        return (sigmainv[0,0] * (x**2.0) + sigmainv[1,1] * (y**2.0) + 2.0*sigmainv[0,1]*(x*y))
    else:
        return (sigmainv[0,0] * (x**2.0) + sigmainv[1,1] * (y**2.0) + 2.0*sigmainv[0,1]*(x*y) + 
                sigmainv[2,2] * (z**2.0) + 2.0*sigmainv[0,2]*(x*z) +  2.0*sigmainv[1,2]*(y*z))




#------------------------------------------------------------------------------
def rms(data,axis = 0, keepdims = False):
    """
    Calculate root-mean-square along axis.
    
    Args:
        :data: 
            | list of values or ndarray
        :axis:
            | 0, optional
            | Axis along which to calculate rms.
        :keepdims:
            | False or True, optional
            | Keep original dimensions of array.
    
    Returns:
        :returns:
            | ndarray with rms values.
    """
    data = np2d(data)
    return np.sqrt(np.power(data,2).mean(axis=axis, keepdims = keepdims))

#-----------------------------------------------------------------------------
def geomean(data, axis = 0, keepdims = False):
    """
    Calculate geometric mean along axis.
    
    Args:
        :data:
            | list of values or ndarray
        :axis:
            | 0, optional
            | Axis along which to calculate geomean.
        :keepdims:
            | False or True, optional
            | Keep original dimensions of array.
    
    Returns:
        :returns:
            | ndarray with geomean values. 
    """
    data = np2d(data)
    return np.power(data.prod(axis=axis, keepdims = keepdims),1/data.shape[axis])
 
#------------------------------------------------------------------------------
def polyarea(x,y):
    """
    Calculates area of polygon. 
    
    | First coordinate should also be last.
    
    Args:
        :x: 
            | ndarray of x-coordinates of polygon vertices.
        :y: 
            | ndarray of x-coordinates of polygon vertices.     
    
    Returns:
        :returns:
            | float (area or polygon)
    
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1).T)-np.dot(y,np.roll(x,1).T))

#------------------------------------------------------------------------------
def cart2pol(x,y = None, htype = 'deg'):
    """
    Convert Cartesion to polar coordinates.
    
    Args:
        :x: 
            | float or ndarray with x-coordinates
        :y: 
            | None or float or ndarray with x-coordinates, optional
            | If None, y-coordinates are assumed to be in :x:.
        :htype:
            | 'deg' or 'rad, optional
            | Output type of theta.
    
    Returns:
        :returns: 
            | (float or ndarray of theta, float or ndarray of r) values
    """
    if y is None:
        y = x[...,1].copy()
        x = x[...,0].copy()
    return positive_arctan(x,y, htype = htype), np.sqrt(x**2 + y**2)

def pol2cart(theta, r = None, htype = 'deg'):
    """
    Convert Cartesion to polar coordinates.
    
    Args:
        :theta: 
            | float or ndarray with theta-coordinates
        :r: 
            | None or float or ndarray with r-coordinates, optional
            | If None, r-coordinates are assumed to be in :theta:.
        :htype:
            | 'deg' or 'rad, optional
            | Intput type of :theta:.
    
    Returns:
        :returns:
            | (float or ndarray of x, float or ndarray of y) coordinates 
    """
    if htype == 'deg':
        d2r = np.pi/180.0
    else:
        d2r = 1.0
    if r is None:
        r = theta[...,1].copy()
        theta = theta[...,0].copy()
    theta = theta*d2r
    return r*np.cos(theta), r*np.sin(theta)

#------------------------------------------------------------------------------
def spher2cart(theta, phi, r = 1., deg = True):
    """
    Convert spherical to cartesian coordinates.
    
    Args:
        :theta:
            | Float, int or ndarray
            | Angle with positive z-axis.
        :phi:
            | Float, int or ndarray
            | Angle around positive z-axis starting from x-axis.
        :r:
            | 1, optional
            | Float, int or ndarray
            | radius
            
    Returns:
        :x, y, z:
            | tuple of floats, ints or ndarrays
            | Cartesian coordinates
    """
    if deg == True:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
    x= r*np.sin(theta)*np.cos(phi)
    y= r*np.sin(theta)*np.sin(phi)
    z= r*np.cos(theta)
    return x,y,z

def cart2spher(x,y,z, deg = True):
    """
    Convert cartesian to spherical coordinates.
    
    Args:        
        :x, y, z:
            | tuple of floats, ints or ndarrays
            | Cartesian coordinates
    Returns:
        :theta:
            | Float, int or ndarray
            | Angle with positive z-axis.
        :phi:
            | Float, int or ndarray
            | Angle around positive z-axis starting from x-axis.
        :r:
            | 1, optional
            | Float, int or ndarray
            | radius

    """
    r = np.sqrt(x*x + y*y + z*z)
    phi = np.arctan2(y,x)
    phi[phi<0.] = phi[phi<0.] + 2*np.pi
    zdr = z/r
    zdr[zdr > 1.] = 1.
    zdr[zdr<-1.] = -1
    theta = np.arccos(zdr)
    if deg == True:
        theta = theta*180/np.pi
        phi = phi *180/np.pi
    return theta, phi, r   


#------------------------------------------------------------------------------
# magnitude of a vector
def magnitude_v(v):
    """
    Calculates magnitude of vector.
    
    Args:
        :v: 
            | ndarray with vector
 
    Returns:
        :magnitude:
            | ndarray 
    """
    magnitude = np.sqrt(v[:,0]**2 + v[:,1]**2)
    return magnitude


# angle between vectors
def angle_v1v2(v1,v2,htype = 'deg'):
    """
    Calculates angle between two vectors.
    
    Args:
        :v1: 
            | ndarray with vector 1
        :v2: 
            | ndarray with vector 2
        :htype:
            | 'deg' or 'rad', optional
            | Requested angle type.
    
    Returns:
        :ang: 
            | ndarray 
    """
    denom = magnitude_v(v1)*magnitude_v(v2)
    denom[denom==0.] = np.nan
    ang = np.arccos(np.sum(v1*v2,axis=1)/denom)
    if htype == 'deg':
        ang = ang*180/np.pi
    return ang
    
#------------------------------------------------------------------------------
def histogram(a, bins=10, bin_center = False, range=None, normed=False, weights=None, density=None):
    """
    Histogram function that can take as bins either the center (cfr. matlab hist) or bin-edges.
    
    Args: 
        :bin_center:
            | False, optional
            | False: if :bins: int, str or sequence of scalars:
            |       default to numpy.histogram (uses bin edges).
            | True: if :bins: is a sequence of scalars:
            |         bins (containing centers) are transformed to edges
            |         and nump.histogram is run. 
            |         Mimicks matlab hist (uses bin centers).
        
    Note:
        For other armuments and output, see ?numpy.histogram
        
    Returns:
        :returns:
            | ndarray with histogram
    """
    if (isinstance(bins, list) |  isinstance(bins, np.ndarray)) & (bin_center == True):
        if len(bins) == 1:
            edges = np.hstack((bins[0],np.inf))
        else:
            centers = bins
            d = np.diff(centers)/2
            edges = np.hstack((centers[0]-d[0], centers[:-1] + d, centers[-1] + d[-1]))
            edges[1:] = edges[1:] + np.finfo(float).eps
        return np.histogram(a, bins=edges, range=range, normed=normed, weights=weights, density=density)

    else:
        return np.histogram(a, bins=bins, range=range, normed=normed, weights=weights, density=density)

#------------------------------------------------------------------------------
def v_to_cik(v, inverse = False):
    """
    Calculate 2x2 '(covariance matrix)^-1' elements cik 
    
    Args:
        :v: 
            | (Nx5) np.ndarray
            | ellipse parameters [Rmax,Rmin,xc,yc,theta]
        :inverse:
            | If True: return inverse of cik.
    
    Returns:
        :cik: 
            | 'Nx2x2' (covariance matrix)^-1
    
    Notes:
        | cik is not actually a covariance matrix,
        | only for a Gaussian or normal distribution!

    """
    v = np.atleast_2d(v)
    g11 = (1/v[:,0]*np.cos(v[:,4]))**2 + (1/v[:,1]*np.sin(v[:,4]))**2
    g22 = (1/v[:,0]*np.sin(v[:,4]))**2 + (1/v[:,1]*np.cos(v[:,4]))**2
    g12 = (1/v[:,0]**2 - 1/v[:,1]**2)*np.sin(v[:,4])*np.cos(v[:,4])
    cik = np.zeros((g11.shape[0],2,2))

    for i in range(g11.shape[0]):
        cik[i,:,:] = np.vstack((np.hstack((g11[i],g12[i])), np.hstack((g12[i],g22[i]))))
        if inverse == True:
            cik[i,:,:] = np.linalg.inv(cik[i,:,:])
    return cik
#------------------------------------------------------------------------------

def cik_to_v(cik, xyc = None, inverse = False):
    """
    Calculate v-format ellipse descriptor from 2x2 'covariance matrix'^-1 cik 
    
    Args:
        :cik: 
            | 'Nx2x2' (covariance matrix)^-1
        :inverse:
            | If True: input is inverse of cik.
              
            
    Returns:
        :v: 
            | (Nx5) np.ndarray
            | ellipse parameters [Rmax,Rmin,xc,yc,theta]

    Notes:
        | cik is not actually the inverse covariance matrix,
        | only for a Gaussian or normal distribution!

    """
    if cik.ndim < 3:
        cik = cik[None,...]
    
    if inverse == True:
        for i in range(cik.shape[0]):
            cik[i,:,:] = np.linalg.inv(cik[i,:,:])
            
    g11 = cik[:,0,0]
    g22 = cik[:,1,1] 
    g12 = cik[:,0,1]

    theta = 0.5*np.arctan2(2*g12,(g11-g22)) + (np.pi/2)*(g12<0)
    #theta = theta2 + (np.pi/2)*(g12<0)
    #theta2 = theta
    cottheta = np.cos(theta)/np.sin(theta) #np.cot(theta)
    cottheta[np.isinf(cottheta)] = 0

    a = 1/np.sqrt((g22 + g12*cottheta))
    b = 1/np.sqrt((g11 - g12*cottheta))

    # ensure largest ellipse axis is first (correct angle):
    c = b>a; a[c], b[c], theta[c] = b[c],a[c],theta[c]+np.pi/2

    v = np.vstack((a, b, np.zeros(a.shape), np.zeros(a.shape), theta)).T
    
    # add center coordinates:
    if xyc is not None:
        v[:,2:4] = xyc
    
    return v

def fmod(x, y):
    """
    Floating point modulus 
    
    | e.g., fmod(theta, np.pi * 2) would keep an angle in [0, 2pi]

    Args:
        :x:
            | angle to restrict
        :y: 
            | end of  interval [0, y] to restrict to
    
    Returns:
        :r: floating point modulus
    """
    r = x
    while(r < 0):
        r = r + y
    while(r > y):
        r = r - y
    return r

def remove_outliers(data, alpha = 0.01):
    """
    Remove multivariate outliers from data when outside of alpha-level confidence ellipsoid.
    
    Args:
        :data:
            | Nxp ndarray with multivariate data (N samples, p variables)
        :alpha:
            | 0.01, optional
            | Significance level of confidence ellipsoid marking the boundary for outliers.
            
    Return:
        :data:
            | (N-... x p) ndarray with multivariate data; outliers removed.
    """
    # delete outliers:    
    datac = data.mean(axis=0)
    cov_ = np.cov(data.T)
    f = stats.chi2.ppf(1-alpha, data.shape[1])
    D = mahalanobis2(data, mu = datac, sigmainv = np.linalg.inv(cov_)/f)**0.5
    datan = data.copy()
    datan = datan[D<=1]
    return datan

def fit_ellipse(xy, center_on_mean_xy = False):
    """
    Fit an ellipse to supplied data points.

    Args:
        :xy: 
            | coordinates of points to fit (Nx2 array)
        :center_on_mean_xy:
            | False, optional
            | Center ellipse on mean of xy 
            | (otherwise it might be offset due to solving 
            | the contrained minization problem: aT*S*a, see ref below.)
            
    Returns:
        :v:
            | vector with ellipse parameters [Rmax,Rmin, xc,yc, theta (rad.)]
            
    Reference:
        1. Fitzgibbon, A.W., Pilu, M., and Fischer R.B., 
        Direct least squares fitting of ellipsees, 
        Proc. of the 13th Internation Conference on Pattern Recognition, 
        pp 253–257, Vienna, 1996.
    """
    # remove centroid:
#    center = xy.mean(axis=0)
#    xy = xy - center
    
    # Fit ellipse:
    x, y = xy[:,0:1], xy[:,1:2]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S, C = np.dot(D.T, D), np.zeros([6, 6])
    C[0, 2], C[2, 0], C[1, 1] = 2, 2, -1
    U, s, V = np.linalg.svd(np.dot(np.linalg.inv(S), C))
    e = U[:, 0]
#    E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
#    n = np.argmax(np.abs(E))
#    e = V[:,n]
        
    # get ellipse axis lengths, center and orientation:
    b, c, d, f, g, a = e[1] / 2, e[2], e[3] / 2, e[4] / 2, e[5], e[0]
    
    # get ellipse center:
    num = b * b - a * c
    if num == 0:
        xc = 0
        yc = 0
    else:
        xc = ((c * d - b * f) / num) 
        yc = ((a * f - b * d) / num) 
    
    # get ellipse orientation:
    theta = np.arctan2(np.array(2 * b), np.array((a - c))) / 2
#    if b == 0:
#        if a > c:
#            theta = 0
#        else:
#            theta = np.pi/2
#    else:
#        if a > c:
#            theta = np.arctan2(2*b,(a-c))/2
#        else:
#            theta =  np.arctan2(2*b,(a-c))/2 + np.pi/2
        
    # axis lengths:
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    a, b  = np.sqrt((up / down1)), np.sqrt((up / down2))


    # assert that a is the major axis (otherwise swap and correct angle)
    if(b > a):
        b, a = a, b
        # ensure the angle is betwen 0 and 2*pi
        theta = fmod(theta, 2.0 * np.pi)
        
    if center_on_mean_xy == True:
        xc,yc = xy.mean(axis=0)

    return np.hstack((a, b, xc, yc, theta))


def fit_cov_ellipse(xy, alpha = 0.05, pdf = 'chi2', SE = False, 
                    robust = False, robust_alpha = 0.01):
    """
    Fit covariance ellipse to xy data.
    
    Args:
        :xy: 
            | coordinates of points to fit (Nx2 array)
        :alpha:
            | 0.05, optional
            | alpha significance level 
            | (e.g alpha = 0.05 for 95% confidence ellipse)
        :pdf:
            | chi2, optional
            | - 'chi2': Rescale using Chi2-distribution
            | - 't': Rescale using Student t-distribution
            | - 'norm': Rescale using normal-distribution
            | - None: don't rescale using pdf, use alpha as scalefactor (cfr. alpha* 1SD or alpha * 1SE)
        :SE:
            | False, optional
            | If false, fit standard error ellipse at alpha significance level
            | If true, fit standard deviation ellipse at alpha significance level
        :robust:
            | False, optional
            | If True: remove outliers beyond the confidence ellipsoid before calculating
            |          the covariances.
        :robust_alpha:
            | 0.01, optional
            | Significance level of confidence ellipsoid marking the boundary for outliers.
            
    Returns:
        :v:
            | vector with ellipse parameters [Rmax,Rmin, xc,yc, theta (rad.)]
    """

    # delete outliers:    
    if robust == True:
        xy = remove_outliers(xy, alpha = robust_alpha)
    
    xyc = xy.mean(axis=0)
    cov_ = np.cov(xy.T)
    
    
    cik = np.linalg.inv(cov_)
    
    if pdf == 'chi2':
        f = stats.chi2.ppf(1-alpha, xy.shape[1])
    elif pdf == 't':
        f = stats.t.ppf(1-alpha, xy.shape[0]-1)
    elif pdf =='norm':
        f = stats.norm.ppf(1-alpha)
    elif pdf == 'sample':
        p = xy.shape[1]
        n = xy.shape[0]
        f = (p*(n-1)/(n-p)*stats.f.ppf(1-alpha,p,n-p))
    else:
        f = alpha #  -> fraction of Mahalanobis distance
        
    if SE == True:
        f = f/xy.shape[0]
        
    v = cik_to_v(cik/f, xyc=xyc)
    return v

#------------------------------------------------------------------------------
def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    Args:
        :p: 
            | NxK coordinates of N points in K dimensions
        :hull:
            | Either a scipy.spatial.Delaunay object or the MxK array of the 
            | coordinates of M points in K dimensions for which Delaunay 
            | triangulation will be computed
            
    Returns:
        :bool:
            | boolean ndarray with True for in-gamut and False for out-of-gamut points
    """
    if not isinstance(hull,sp.spatial.Delaunay):
        hull = sp.spatial.Delaunay(hull)
    return hull.find_simplex(p)>=0

#------------------------------------------------------------------------------
_SPRAGUE_COEFFICIENTS = np.array([
                                 [884, -1960, 3033, -2648, 1080, -180],
                                 [508, -540, 488, -367, 144, -24],
                                 [-24, 144, -367, 488, -540, 508],
                                 [-180, 1080, -2648, 3033, -1960, 884],
                                 ]).T / 209.0
def interp1_sprague5(x, y, xn, extrap = (np.nan, np.nan)):
    """ 
    Perform a 1-dimensional 5th order Sprague interpolation.
    
    Args:
        :x:
            | ndarray with n-dimensional coordinates.
        :y: 
            | ndarray with values at coordinates in x.
        :xn:
            | ndarray of new coordinates.
        :extrap:
            | (np.nan, np.nan) or string, optional
            | If tuple: fill with values in tuple (<x[0],>x[-1])
            | If string:  ('zeros','linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous','next')
            |           for more info on the other options see: scipy.interpolate.interp1d?
    Returns:
        :yn:
            | ndarray with values at new coordinates in xn.
    """
    # Do extrapolation:
    if ((xn<x[0]) | (xn>x[-1])).any(): # extrapolation needed !
        if isinstance(extrap,tuple):
            if extrap[0] == extrap[1]: 
                yne = np.ones((y.shape[0],len(xn)))*extrap[0]
            else:
                yne = np.zeros((y.shape[0],len(xn)))
                yne[:,(xn<x[0])] = extrap[0]
                yne[:,(xn>x[-1])] = extrap[1]
        elif isinstance(extrap,str):
            yne = interp1d(x, y, kind = extrap, bounds_error = False, fill_value = 'extrapolate')(xn)
        else:
            raise Exception('Invalid option for extrap argument. Only tuple and string allowed.')
        xn_x = xn[(xn>=x[0]) & (xn<=x[-1])]
    else:
        xn_x = xn
        yne = None
     
    # Check equal x-spacing:
    dx = np.diff(x)
    if np.all(dx == dx[0]):
        dx = dx[0] 
    else:
        raise Exception('Elements in x are not equally spaced!')
        
    # Extrapolate x, y with required additional elements for Sprague to work:
    xe = np.hstack((x[0] - 2*dx, x[0] - dx, x, x[-1] + dx, x[-1] + 2*dx))
    
    y = np.atleast_2d(y)
    ye1 = (y[:, :6] @ _SPRAGUE_COEFFICIENTS[:,0])[:,None]
    ye2 = (y[:, :6] @ _SPRAGUE_COEFFICIENTS[:,1])[:,None]
    ye3 = (y[:,-6:] @ _SPRAGUE_COEFFICIENTS[:,2])[:,None]
    ye4 = (y[:,-6:] @ _SPRAGUE_COEFFICIENTS[:,3])[:,None]
    ye = np.hstack((ye1,ye2,y,ye3,ye4)).T
    
    
    # Evaluate at xn_x (no extrapolation!!):
    i = np.searchsorted(xe, xn_x) - 1
    X = np.atleast_2d((xn_x - xe[i]) / (xe[i + 1] - xe[i])).T

    a0 = ye[i]
    a1 = ((2 * ye[i - 2] - 16 * ye[i - 1] + 16 * ye[i + 1] - 2 * ye[i + 2]) / 24)  
    a2 = ((-ye[i - 2] + 16 * ye[i - 1] - 30 * ye[i] + 16 * ye[i + 1] - ye[i + 2]) / 24) 
    a3 = ((-9 * ye[i - 2] + 39 * ye[i - 1] - 70 * ye[i] + 66 * ye[i + 1] - 33 * ye[i + 2] + 7 * ye[i + 3]) / 24)
    a4 = ((13 * ye[i - 2] - 64 * ye[i - 1] + 126 * ye[i] - 124 * ye[i + 1] + 61 * ye[i + 2] - 12 * ye[i + 3]) / 24)
    a5 = ((-5 * ye[i - 2] + 25 * ye[i - 1] - 50 * ye[i] + 50 * ye[i + 1] - 25 * ye[i + 2] + 5 * ye[i + 3]) / 24)

    yn = (a0 + a1*X + a2*X**2 + a3*X**3 + a4*X**4 + a5*X**5).T
    
    if yne is None:
        return yn
    else:
        yne[:,(xn>=x[0]) & (xn<=x[-1])] = yn
        return yne

#------------------------------------------------------------------------------
def interp1(X,Y,Xnew, kind = 'linear', ext = 'extrapolate', w = None, bbox=[None, None], check_finite = False):
    """
    Perform a 1-dimensional linear interpolation (wrapper around scipy.interpolate.InterpolatedUnivariateSpline).
    
    Args:
        :X: 
            | ndarray with n-dimensional coordinates (last axis represents dimension)
        :Y: 
            | ndarray with values at coordinates in X
        :Xnew: 
            | ndarray of new coordinates (last axis represents dimension)
        :kind:
            | str or int,  optional
            | if str: kind is 'translated' to an int value for input to scipy.interpolate.InterpolatedUnivariateSpline()
            | supported options for str: 'linear', 'quadratic', 'cubic', 'quartic', 'quintic'
        :other args:
            | see scipy.interpolate.InterpolatedUnivariateSpline()
        
    Returns:
        :Ynew:
            | ndarray with new values at coordinates in Xnew
    """
    k = ['linear', 'quadratic', 'cubic', 'quartic', 'quintic'].index(kind) + 1
    if ext == 'nearest': ext = 'const'
    return sp.interpolate.InterpolatedUnivariateSpline(X,Y, ext = ext, k = k, w = w, bbox = bbox, check_finite = check_finite)(Xnew)
#------------------------------------------------------------------------------
def ndinterp1_scipy(X,Y,Xnew, fill_value = np.nan,  rescale = False):    
    """
    Perform a n-dimensional linear interpolation (wrapper around scipy.interpolate.LinearNDInterpolator).
    
    Args:
        :X: 
            | ndarray with n-dimensional coordinates (last axis represents dimension)
        :Y: 
            | ndarray with values at coordinates in X
        :Xnew: 
            | ndarray of new coordinates (last axis represents dimension)
        :fill_value: 
            | float, optional
            | Value used to fill in for requested points outside of the
            | convex hull of the input points.  If not provided, then
            | the default is ``nan``.
        :rescale:
            | bool, optional
            | Rescale points to unit cube before performing interpolation.
            | This is useful if some of the input dimensions have
            | incommensurable units and differ by many orders of magnitude.
        
    Returns:
        :Ynew:
            | ndarray with new values at coordinates in Xnew
    """
    return sp.interpolate.LinearNDInterpolator(X,Y, fill_value = fill_value,  rescale = rescale).__call__(Xnew)

def ndinterp1(X, Y, Xnew):
    """
    Perform nd-dimensional linear interpolation using Delaunay triangulation.
    
    Args:
        :X: 
            | ndarray with n-dimensional coordinates (last axis represents dimension).
        :Y: 
            | ndarray with values at coordinates in X.
        :Xnew: 
            | ndarray of new coordinates (last axis represents dimension).
            | When outside of the convex hull of X, then a best estimate is 
            | given based on the closest vertices.
        
    Returns:
        :Ynew:
            | ndarray with new values at coordinates in Xnew.
    """
    #get dimensions:
    n = Xnew.shape[-1]
    # create an object with triangulation
    tri = sp.spatial.Delaunay(X) 
    # find simplexes that contain interpolated points
    s = tri.find_simplex(Xnew)
    # get the vertices for each simplex
    v = tri.vertices[s]
    # get transform matrices for each simplex (see explanation bellow)
    m = tri.transform[s]
    # for each interpolated point p, mutliply the transform matrix by 
    # vector p-r, where r=m[:,n,:] is one of the simplex vertices to which 
    # the matrix m is related to (again, see below)
    b = np.einsum('ijk,ik->ij', m[:,:n,:n], Xnew-m[:,n,:])
    
    # get the weights for the vertices; `b` contains an n-dimensional vector
    # with weights for all but the last vertices of the simplex
    # (note that for n-D grid, each simplex consists of n+1 vertices);
    # the remaining weight for the last vertex can be copmuted from
    # the condition that sum of weights must be equal to 1
    w = np.c_[b, 1-b.sum(axis=1)]
    
    # normalize weigths:
    w = w/w.sum(axis=1, keepdims=True)
    
    # interpolate:
    if Y[v].ndim == 3:
        Ynew = np.einsum('ijk,ij->ik', Y[v], w)
    else:
        Ynew = np.einsum('ij,ij->i', Y[v], w)
        
    return Ynew

def box_m(*X, ni = None, verbosity = 0, robust = False, robust_alpha = 0.01):
    """
    Perform Box's M test (p>=2) to check equality of covariance matrices or Bartlett's test (p==1) for equality of variances.
    
    Args:
        :X: 
            | A number  (k groups) or list of 2d-ndarrays (rows: samples, cols: variables) with data.
            | or a number of 2d-ndarrays with covariance matrices (supply ni!)
        :ni:
            | None, optional
            | If None: X contains data, else, X contains covariance matrices.
        :verbosity: 
            | 0, optional
            | If 1: print results.
        :robust:
            | False, optional
            | If True: remove outliers beyond the confidence ellipsoid before calculating
            |          the covariances.
        :robust_alpha:
            | 0.01, optional
            | Significance level of confidence ellipsoid marking the boundary for outliers.
    
    Returns:
        :statistic:
            | F or chi2 value (see len(dfs))
        :pval:
            | p-value
        :df:
            | degrees of freedom.
            | if len(dfs) == 2: F-test was used.
            | if len(dfs) == 1: chi2 approx. was used.
    
    Notes:
        1. If p==1: Reduces to Bartlett's test for equal variances.
        2. If (ni>20).all() & (p<6) & (k<6): then a more appropriate chi2 test is used in a some cases.
    """

    k = len(X) # groups
    p = np.atleast_2d(X[0]).shape[1] # variables
    if p == 1: # for p == 1: only variance!
        det = lambda x: np.array(x)
    else:
        det = lambda x: np.linalg.det(x)
    if ni is None: # samples in each group
        
        # remove outliers before calculation of box M:
        if robust == True:
            X = [remove_outliers(Xi, alpha = robust_alpha) for Xi in X]
            
        ni = np.array([Xi.shape[0] for Xi in X])
        Si = np.array([np.cov(Xi.T) for Xi in X])
        if p == 1:
            Si = np.atleast_2d(Si).T
    else:
        Si = np.array([Xi for Xi in X]) # input are already cov matrices!
        ni = np.array(ni)
        if ni.shape[0] == 1:
            ni = ni*np.ones((k,))
        
    N = ni.sum()
    S = np.array([(ni[i]-1)*Si[i] for i in range(len(ni))]).sum(axis=0)/(N - k)

    M = (N-k)*np.log(det(S)) - ((ni-1)*np.log(det(Si))).sum()
    if p == 1:
        M = M[0]
    A1 = (2*p**2 + 3*p -1)/(6*(p+1)*(k-1))*((1/(ni-1)) - 1/(N - k)).sum()
    v1 = p*(p+1)*(k-1)/2
    A2 = (p-1)*(p+2)/(6*(k-1))*((1/(ni-1)**2) - 1/(N - k)**2).sum()

    if (A2 - A1**2) > 0:
        v2 = (v1 + 2)/(A2 - A1**2)
        b = v1/(1 - A1 -(v1/v2))
        Fv1v2 = M/b
        statistic = Fv1v2
        pval = 1.0 - sp.stats.f.cdf(Fv1v2,v1,v2)
        dfs = [v1,v2]
        
        if verbosity == 1:
            print('M = {:1.4f}, F = {:1.4f}, df1 = {:1.1f}, df2 = {:1.1f}, p = {:1.4f}'.format(M,Fv1v2,v1,v2,pval))
    else:
        v2 = (v1 + 2)/(A1**2 - A2)
        b = v2/(1 - A1 + (2/v2))
        Fv1v2 = v2*M/(v1*(b - M))
        statistic = Fv1v2
        pval = 1.0 - sp.stats.f.cdf(Fv1v2,v1,v2)
        dfs = [v1,v2]

        if (ni>20).all() & (p<6) & (k<6): #use Chi2v1
            chi2v1 = M*(1-A1)
            statistic = chi2v1
            pval = 1.0 - sp.stats.chi2.cdf(chi2v1,v1)
            dfs = [v1]
            if verbosity == 1:
                print('M = {:1.4f}, chi2 = {:1.4f}, df1 = {:1.1f}, p = {:1.4f}'.format(M,chi2v1,v1,pval))

        else:
            if verbosity == 1:
                print('M = {:1.4f}, F = {:1.4f}, df1 = {:1.1f}, df2 = {:1.1f}, p = {:1.4f}'.format(M,Fv1v2,v1,v2,pval))

    return statistic, pval, dfs

def pitman_morgan(X,Y, verbosity = 0):
    """
    Pitman-Morgan Test for the difference between correlated variances with paired samples.
     
    Args:
        :X,Y: 
            | ndarrays with data.
        :verbosity: 
            | 0, optional
            | If 1: print results. 
            
    Returns:
        :tval:
            | statistic
        :pval:
            | p-value
        :df:
            | degree of freedom.
        :ratio:
            | variance ratio var1/var2 (with var1 > var2).

    Note:
        1. Based on Gardner, R.C. (2001). Psychological Statistics Using SPSS for Windows. New Jersey, Prentice Hall.
        2. Python port from matlab code by Janne Kauttonen (https://nl.mathworks.com/matlabcentral/fileexchange/67910-pitmanmorgantest-x-y; accessed Sep 26, 2019)
    """
    N = X.shape[0]
    var1, var2 = X.var(axis=0),Y.var(axis=0)
    cor = np.corrcoef(X,Y)[0,1]
    
    # must have var1 > var2:
    if var1 < var2:
        var1, var2 = var2, var1

    ratio = var1/var2
    
    # formulas from Garder (2001, p.57):
    numerator1_S1minusS2 = var1-var2
    numerator2_SQRTnminus2 = np.sqrt(N-2)
    numerator3 = numerator1_S1minusS2*numerator2_SQRTnminus2
    denominator1_4timesS1timesS2 = 4*var1*var2
    denominator2_rSquared = cor**2
    denominator3_1minusrSquared = 1.0 - denominator2_rSquared
    denominator4_4timesS1timesS2div1minusrSquared = denominator1_4timesS1timesS2*denominator3_1minusrSquared
    denominator5 = np.sqrt(denominator4_4timesS1timesS2div1minusrSquared)
    df = N-2
    if denominator5 == 0:
        denominator5 = _EPS
    tval = numerator3/denominator5
    
    # compute stats:
    p = 2*(1.0-sp.stats.t.cdf(tval,df))
    if verbosity == 1:
        print('tval = {:1.4f}, df = {:1.1f}, p = {:1.4f}'.format(tval,df, p))

    return tval, p, df, ratio

def stress(DE,DV, axis = 0, max_scale = 100):
    """
    Calculate STandardize-Residual-Sum-of-Squares (STRESS).
    
    Args:
        :DE, DV: 
            | ndarrays of data to be compared.
        :axis:
            | 0, optional
            | axis with samples
        :max_scale:
            | 100, optional
            | Maximum of scale.
            
    Returns:
        :stress:
            | nadarray with stress value(s).
    
    Reference:
        1. `Melgosa, M., García, P. A., Gómez-Robledo, L., Shamey, R., Hinks, D., Cui, G., & Luo, M. R. (2011). 
        Notes on the application of the standardized residual sum of squares index 
        for the assessment of intra- and inter-observer variability in color-difference experiments. 
        Journal of the Optical Society of America A, 28(5), 949–953. 
        <https://doi.org/10.1364/JOSAA.28.000949>`_
    """
    F = (DE**2).sum(axis = axis, keepdims = True)/(DE*DV).sum(axis = axis, keepdims = True)
    return max_scale*(((DE - F*DV)**2).sum(axis = axis, keepdims = True)/(F**2*DV**2).sum(axis = axis, keepdims = True))**0.5

def stress_F_test(stressA, stressB, N, alpha = 0.05):
    """ 
    Perform F-test on significance of difference between STRESS A and STRESS B.
    
    Args:
        :stressA, stressB:
            | ndarray with stress(es) values for A and B
        :N:
            | int or ndarray with number of samples used to determine stress values.
        :alpha:
            | 0.05, optional
            | significance level
            
    Returns:
        :Fstats:
            | Dictionary with keys:
            | - 'p': p-values
            | - 'F':  F-values
            | - 'Fc': critcal values
            | - 'H': string reporting on significance of A compared to B.
    """
    N = N*np.ones(stressA.shape[0])
    Fvs = np.nan*np.ones_like(stressA)
    ps = Fvs.copy()
    Fcs = Fvs.copy()
    H = []
    i = 0
    for stA, stB in zip(stressA,stressB):
        Ni = N[i]
        Fvs[i] = stA**2/stB**2
        ps[i] = stats.f.sf(Fvs[i], Ni-1, Ni-1)
        Fcs[i] = stats.f.ppf(q = alpha/2, dfn = Ni - 1, dfd = Ni-1)
        if Fvs[i] < Fcs[i]:
            H_ = "A significantly better than B"
        elif Fvs[i] > 1/Fcs[i]:
            H_ = "A significantly poorer than B"
        elif (Fcs[i] <= Fvs[i]) & (Fvs[i] < 1):
            H_ = "A insignificantly better than B"
        elif (1 < Fvs[i]) & (Fvs[i] <= 1/Fcs[i]):
            H_ = "A insignificanty poorer than B"
        elif (Fvs[i] == 1):
            H_ = "A equals B"
        H.append(H_)
        i+=1
    Fstats = {'p': ps, 'F': Fvs, 'Fc': Fcs, 'H': H}
    return Fstats