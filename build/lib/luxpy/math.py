# -*- coding: utf-8 -*-
"""
###############################################################################
# Module with useful math functions
###############################################################################
#
# line_intersect(): Line intersections of series of two line segments a and b. 
#                  From https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
#
# positive_arctan(): Calculates the positive angle (0°-360° or 0 - 2*pi rad.) from x and y.
#
# dot23(): Dot product of a 2-d numpy.ndarray with a (N x K x L) 3-d numpy.array using einsum().
#
# check_symmetric(): Checks if A is symmetric.
#
# check_posdef(): Checks positive definiteness of a matrix via Cholesky.
#
# symmM_to_posdefM(): Converts a symmetric matrix to a positive definite one. Two methods are supported:
#                   * 'make': A Python/Numpy port of Muhammad Asim Mubeen's matlab function Spd_Mat.m (https://nl.mathworks.com/matlabcentral/fileexchange/45873-positive-definite-matrix)
#                   * 'nearest': A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code. (https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite)
#
# bvgpdf(): Evaluate bivariate Gaussian probability density function (BVGPDF) at (x,y) with center mu and inverse covariance matric, sigmainv.
#
# mahalanobis2(): Evaluate the squared mahalanobis distance with center mu and shape and orientation determined by sigmainv. 
#
# rms(): Calculates root-mean-square along axis.
#
# geomean(): Calculates geometric mean along axis.
#
# polyarea(): Calculates area of polygon. (First coordinate should also be last)
#
# erf(): erf-function, direct import from scipy.special
#------------------------------------------------------------------------------

Created on Tue Jun 27 11:50:32 2017

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import *
from scipy.special import erf, erfinv
__all__ = ['symmM_to_posdefM','check_symmetric','check_posdef','positive_arctan','line_intersect','erf', 'erfinv']
__all__+= ['bvgpdf','mahalanobis2','dot23', 'rms','geomean','polyarea']


#------------------------------------------------------------------------------
def line_intersect(a1, a2, b1, b2):
    """
    Line intersections of series of two line segments a and b. 
        
    Args:
        :a1: numpy.ndarray (.shape  = (N,2)) specifying end-point 1 of line a
        :a2: numpy.ndarray (.shape  = (N,2)) specifying end-point 2 of line a
        :b1: numpy.ndarray (.shape  = (N,2)) specifying end-point 1 of line b
        :b2: numpy.ndarray (.shape  = (N,2)) specifying end-point 2 of line b
    
        Note: N is the number of line segments a and b.
    
    Returns:
        :returns: numpy.ndarray with line-intersections (.shape = (N,2))
    
    Notes:
        See: https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
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
        :x: numpy.ndarray of x-coordinates
        :y: numpy.ndarray of y-coordinates
        :htype: 'deg' or 'rad', optional
            'deg': hue angle between 0° and 360°
            'rad': hue angle between 0 and 2pi radians
    Returns:
        :returns: numpy.ndarray of positive angles.
    """
    if htype == 'deg':
        r2d = 180.0/np.pi
        h360 = 360.0
    else:
        r2d = 1.0
        h360 = 2.0*np.pi
    h = np.arctan2(y,x)*r2d
    h[np.where(h<0)] = h[np.where(h<0)] + h360
    return h


#------------------------------------------------------------------------------
def dot23(A,B, keepdims = False):
    """
    Dot product of a 2-d numpy.ndarray with a (N x K x L) 3-d numpy.array using einsum().
    
    Args:
        :A: numpy.ndarray (.shape = (M,N))
        :B: numpy.ndarray (.shape = (N,K,L))
        
    Returns:
        :returns: numpy.ndarray (.shape = (M,K,L))
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
        :A: numpy.ndarray
        :atol: float, optional
            The absolute tolerance parameter (see Notes of numpy.allclose())
        :rtol: float, optional
            The relative tolerance parameter (see Notes of numpy.allclose())
    
    Returns:
        :returns: Bool
            True: if the array is symmetric within the given tolerance; False otherwise
    """
    return np.allclose(A, A.T, atol = atol, rtol = rtol)


def check_posdef(A, atol = 1.0e-9, rtol = 1.0e-9):
    """
    Checks positive definiteness of a matrix via Cholesky.
    
    Args:
        :A: numpy.ndarray
        :atol: float, optional
            The absolute tolerance parameter (see Notes of numpy.allclose())
        :rtol: float, optional
            The relative tolerance parameter (see Notes of numpy.allclose())
    
    Returns:
        :returns: Bool
            True: if the array is positive-definite within the given tolerance; False otherwise

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
        :A: numpy.ndarray
        :atol: float, optional
            The absolute tolerance parameter (see Notes of numpy.allclose())
        :rtol: float, optional
            The relative tolerance parameter (see Notes of numpy.allclose())
        :method: 'make' or 'nearest', optional (see notes for more info)
        :forcesymm: True or False, optional
            If A is not symmetric, force symmetry using: A = numpy.triu(A) + numpy.triu(A).T - numpy.diag(numpy.diag(A))
    
    Returns:
        :returns: numpy.ndarray with positive-definite matrix.
        
    Notes on supported methods:
        'make': A Python/Numpy port of Muhammad Asim Mubeen's matlab function Spd_Mat.m (https://nl.mathworks.com/matlabcentral/fileexchange/45873-positive-definite-matrix)
        'nearest': A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code. (https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite)
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
    Evaluate bivariate Gaussian probability density function (BVGPDF) at (x,y) with center mu and inverse covariance matric, sigmainv.
    
    Args:
        :x: scalar or list or numpy.ndarray (.ndim = 1 or 2) with x(y)-coordinates at which to evaluate bivariate Gaussian PD.
        :y: None or scalar or list or numpy.ndarray (.ndim = 1) with y-coordinates at which to evaluate bivariate Gaussian PD, optional.
            If :y: is None, :x: should be a 2d array.
        :mu: None or numpy.ndarray (.ndim = 2) with center coordinates of bivariate Gaussian PD, optional. 
            None defaults to numpy.array([0,0]).
        :sigmainv: None or numpyndarray with 'inverse covariance matrix', optional 
            Determines the shape and orientation of the PD.
            None default to numpy.eye(2).
     Returns:
         :returns: numpy.ndarray with magnitude of BVGPDF(x,y)   
    
    """
    return np.exp(-0.5*mahalanobis2(x,y = y, mu = mu, sigmainv= sigmainv))

#------------------------------------------------------------------------------
def mahalanobis2(x, y = None, mu = None,sigmainv = None):
    """
    Evaluate the squared mahalanobis distance with center mu and shape and orientation determined by sigmainv. 
    
    Args:
        :x: scalar or list or numpy.ndarray (.ndim = 1 or 2) with x(y)-coordinates at which to evaluate the mahalanobis distance squared.
        :y: None or scalar or list or numpy.ndarray (.ndim = 1) with y-coordinates at which to evaluate the mahalanobis distance squared, optional.
            If :y: is None, :x: should be a 2d array.
        :mu: None or numpy.ndarray (.ndim = 2) with center coordinates of the mahalanobis ellipse, optional. 
            None defaults to numpy.array([0,0]).
        :sigmainv: None or numpyndarray with 'inverse covariance matrix', optional 
            Determines the shape and orientation of the PD.
            None default to numpy.eye(2).
     Returns:
         :returns: numpy.ndarray with magnitude of mahalanobis2(x,y)   

    
    """
    if mu is None:
        mu = np.zeros(2)
    if sigmainv is None:
        sigmainv = np.eye(2)
    
    x = np2d(x)

    if y is not None:
        x = x - mu[0] # center data on mu 
        y = np2d(y) - mu[1] # center data on mu 
    else:
        x = x - mu # center data on mu    
        x, y = asplit(x)
    return (sigmainv[0,0] * (x**2.0) + sigmainv[1,1] * (y**2.0) + 2.0*sigmainv[0,1]*(x*y))



#------------------------------------------------------------------------------
def rms(data,axis = 0, keepdims = False):
    """
    Calculate root-mean-square along axis.
    
    Args:
        :data: list of values or numpy.ndarray
        :axis: 0, optional
            Axis along which to calculate rms.
        :keepdims: False or True, optional
            Keep original dimensions of array.
    
    Returns:
        :returns: numpy.ndarray with rms values.
    """
    data = np2d(data)
    return np.sqrt(np.power(data,2).mean(axis=axis, keepdims = keepdims))

#-----------------------------------------------------------------------------
def geomean(data, axis = 0, keepdims = False):
    """
    Calculate geometric mean along axis.
    
    Args:
        :data: list of values or numpy.ndarray
        :axis: 0, optional
            Axis along which to calculate geomean.
        :keepdims: False or True, optional
            Keep original dimensions of array.
    
    Returns:
        :returns: numpy.ndarray with geomean values. 
    """
    data = np2d(data)
    return np.power(data.prod(axis=axis, keepdims = keepdims),1/data.shape[axis])
 
#------------------------------------------------------------------------------
def polyarea(x,y):
    """
    Calculates area of polygon. 
    
    First coordinate should also be last.
    
    Args:
        :x: numpy.ndarray of x-coordinates of polygon vertices.
        :y: numpy.ndarray of x-coordinates of polygon vertices.     
    
    Returns:
        :returns: float (area or polygon)
    
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1).T)-np.dot(y,np.roll(x,1).T))