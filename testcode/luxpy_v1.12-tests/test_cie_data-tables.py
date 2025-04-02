# Test data from https://cie.co.at/data-tables (16/12/2024 9:45 CET) vs luxpy data
##################################################################################

#------luxpy import-------------------------------------------------------
import os 
if "code\\py" not in os.getcwd(): os.chdir("./code/py/")
root = "../../" if "code\\py" in os.getcwd() else  "./"

import luxpy as lx
print('luxpy version:', lx.__version__,'\n')


#-----other imports-------------------------------------------------------
import copy
import numpy as np
import scipy as sp
from scipy import interpolate
import matplotlib.pyplot as plt 

def deal_with_nan_input(x):
    if type(x[0]) == list:
        x = np.array([[float(xij) for xij in xi] for xi in x if xi != ['']])
    return x 

def plot_dcmfs(cmf_lx, cmf_cie):
    plt.figure()
    cmf_lx_cie = cmf_lx - cmf_cie
    cmf_lx_cie[0] = cmf_cie[0]
    axh = lx.plot_cmfs(cmf_lx_cie)
    plt.ylim([cmf_lx_cie[1:].min()*1.05, cmf_lx_cie[1:].max()*1.05])
    return axh, cmf_lx_cie

def compare_cmfs(cmf_lx, cmf_cie, spd = lx._CIE_D65, K = 683, relative_xyz = False, nans_to_zero = True):
    
    dcmf = None
    if cmf_lx is None:
        cmf_cie_5nm = cmf_cie[:,::5]
        cmf_lx = lx.cie_interp(cmf_cie_5nm, cmf_cie[0], datatype = 'cmf', force_scipy_interpolator = True, scipy_interpolator = 'InterpolatedUnivariateSpline')

        axh, dcmf = plot_dcmfs(cmf_lx, cmf_cie)

    
    if nans_to_zero:
        cmf_lx[np.isnan(cmf_lx)] = 0.0
        cmf_cie[np.isnan(cmf_cie)] = 0.0 

    diff_max = np.abs(cmf_lx[1:] - cmf_cie[1:]).max(axis = 1)
    print('Max. CMF diff: ', diff_max)
    diff_wls_x, diff_wls_y, diff_wls_z = None, None, None
    if diff_max.any()>0:
        plt.figure()
        if (np.abs(cmf_lx[1] - cmf_cie[1])>0).any():
            diff_wls_x = cmf_lx[0,np.abs(cmf_lx[1] - cmf_cie[1])>0]
            diff_x_lx = cmf_lx[1,np.abs(cmf_lx[1] - cmf_cie[1])>0]
            diff_x_cie = cmf_cie[1,np.abs(cmf_lx[1] - cmf_cie[1])>0]
            print('Wavelength(s) with difference in xbar: ', diff_wls_x)
            plt.plot(diff_wls_x, diff_x_lx, 'r.')
            plt.plot(diff_wls_x, diff_x_cie, 'r+')
        if (np.abs(cmf_lx[2] - cmf_cie[2])>0).any():
            diff_wls_y = cmf_lx[0,np.abs(cmf_lx[2] - cmf_cie[2])>0]
            diff_y_lx = cmf_lx[2,np.abs(cmf_lx[2] - cmf_cie[2])>0]
            diff_y_cie = cmf_cie[2,np.abs(cmf_lx[2] - cmf_cie[2])>0]
            print('Wavelength(s) with difference in ybar: ', diff_wls_y)
            plt.plot(diff_wls_y, diff_y_lx, 'g.')
            plt.plot(diff_wls_y, diff_y_cie, 'g+')
        if (np.abs(cmf_lx[3] - cmf_cie[3])>0).any():
            diff_wls_z = cmf_lx[0,np.abs(cmf_lx[3] - cmf_cie[3])>0]
            diff_z_lx = cmf_lx[3,np.abs(cmf_lx[3] - cmf_cie[3])>0]
            diff_z_cie = cmf_cie[3,np.abs(cmf_lx[3] - cmf_cie[3])>0]
            print('Wavelength(s) with difference in zbar: ', diff_wls_z)
            plt.plot(diff_wls_z, diff_z_lx, 'b.')
            plt.plot(diff_wls_z, diff_z_cie, 'b+')
    xyz_lx = lx.spd_to_xyz(spd, cieobs = cmf_lx, K = K, relative = relative_xyz)
    xyz_cie = lx.spd_to_xyz(spd, cieobs = cmf_cie, K = K, relative = relative_xyz)
    print('Max. XYZ diff.: ', np.abs(xyz_lx - xyz_cie).max(axis = 0))
    return dcmf

#----------------------------------------------------------------------------------------------------------------
# Lagrange interpolation code from: https://stackoverflow.com/questions/78007883/obtaining-a-list-of-the-coefficients-from-the-lagrange-interpolation-in-python
from math import * 
def P(l,x:float):   # l is the list of coeffcients from lowest to highest degree
    res=0
    for n,i in enumerate(l):
        res+=i*x**n    # evaluate the equation in x
    return res

def Lagrange(xtab,ytab):    # return the Lagrange interpolation
    X=np.poly1d([1,0])
    P=0
    for i in range(len(ytab)):
        Li=1
        for j in range(len(ytab)):
            if i==j:
                continue
            else :
                Li=Li*((X-xtab[j])/(xtab[i]-xtab[j]))
        P+=Li*ytab[i]
    return P

def points(l,N,h):   # return  a list of points of a known polynomial randomized a little by h
    xtab=np.linspace(-10,10,N)
    ytab=np.zeros(N)
    for k,x in enumerate(xtab):
        ytab[k]=P(l,x)
    for i,y in enumerate(ytab):
        ytab[i]= y + np.random.uniform(-h,h)
    return xtab,ytab

def interp1_lagrange(xtab,ytab, xtab_new):
    L = Lagrange(xtab,ytab)
    return np.polyval(L,xtab_new)

#------------------------------------------------------
# From: https://www.geeksforgeeks.org/lagranges-interpolation/
# function to interpolate the given data points
# using Lagrange's formula
# xi -> corresponds to the new data point
# whose value is to be obtained
# n -> represents the number of known data points
def interp_lagrange(x, y, xn) -> float:
    n = x.shape[0]

    # Initialize result
    result = 0.0
    for i in range(n):

        # Compute individual terms of above formula
        term = y[i]
        for j in range(n):
            if j != i:
                term = term * (xn - x[j]) / (x[i] - x[j])

        # Add current term to result
        result += term

    return result

#-----------------------------------------------------------------------------
#From: https://gist.github.com/aurelienpierre/1d9826e7db078e048bf437e516a7a4b2
#from sympy import *
import sympy
from sympy import matrices
from sympy.polys.polyfuncs import horner
import numpy
import pylab
import warnings

def Lagrange_interpolation(points, variable=None):
    """
    Compute the Lagrange interpolation polynomial.
    
    :var points: A numpy n×2 ndarray of the interpolations points
    :var variable: None, float or ndarray
    :returns:   * P the symbolic expression
                * Y the evaluation of the polynomial if `variable` is float or ndarray
                
    """
    points = points.T 
    x = sympy.Symbol("x")
    L = matrices.zeros(1, points.shape[0])
    i = 0

    for p in points:
        numerator = 1
        denominator = 1
        other_points = numpy.delete(points, i, 0)

        for other_p in other_points:
            numerator = numerator * (x - other_p[0])
            denominator = denominator * (p[0] - other_p[0])

        L[i] = numerator / denominator
        i = i+1
        
    # The Horner factorization will reduce chances of issues with floats approximations
    P = horner(L.multiply(points[..., 1])[0])
    Y = None
    
    try: 
        Y = sympy.lambdify(x, P, 'numpy')
        Y = Y(variable)
            
    except:
        warnings.warn("No input variable given - polynomial evaluation skipped")
            
    return P,Y

#--------------------------------------------------------------------------------
#Adapted From: https://www.math.ntnu.no/emner/TMA4125/2021v/lectures/LagrangeInterpolation.pdf
def cardinal(xdata, x):
    """
    cardinal(xdata, x):
    In: xdata, array with the nodes x_i.
    x, array or a scalar of values in which the cardinal functions are evaluated.
    Return: l: a list of arrays of the cardinal functions evaluated in x.
    """
    n = xdata.shape[0] # Number of evaluation points x
    l = []
    for i in range(n): # Loop over the cardinal functions
        li = np.ones(x.shape[0])
        li_ = 1
        for j in range(n): # Loop to make the product for l_i
            if i != j:
                li = li*(x-xdata[j])/(xdata[i]-xdata[j])
        l.append(li) # Append the array to the list
    return l


def lagrange(ydata, l, idx):
    """
    lagrange(ydata, l):
    In: ydata, array of the y-values of the interpolation points.
    l, a list of the cardinal functions, given by cardinal(xdata, x)
    Return: An array with the interpolation polynomial.
    """
    poly = 0
    for i in range(len(l)):
        poly = poly + ydata[idx[i]]*l[i]
    return poly

def cardinal(xdata, x, k = 3):
    """
    cardinal(xdata, x):
    In: xdata, array with the nodes x_i.
    x, array or a scalar of values in which the cardinal functions are evaluated.
    Return: l: a list of arrays of the cardinal functions evaluated in x.
    """
    indices = np.searchsorted(xdata, x, side='left')
    ks = np.arange(k+1) - (k+1)//2
    idx = indices + ks[...,None]
    idx[:,idx[0]<0] = idx[:,idx[0]<0] - idx[0].min()
    n = xdata.shape[0]-1
    idx[:,idx[-1]>n] = idx[:,idx[-1]>n] - (idx[-1].max() - n)
    l = []
    for i in range(k+1):
        li = np.ones(x.shape[0])
        idxi = idx[i]
        for j in range(k+1):
            if i != j:
                idxj = idx[j]
                li = li*(x-xdata[idxj])/(xdata[idxi]-xdata[idxj])
        l.append(li) # Append the array to the list
    return l, idx
    
def interp_lagrange(x,y,xn, k = 3):
    l, idx = cardinal(x, xn, k = k) # Find the cardinal functions evaluated in x
    #p = lagrange(y, l, idx)
    p = (y[...,idx]*np.array(l)).sum(axis=int(y.ndim>1)) # multiply yi (points corresponding to xi for each of the k+1 lagrange polynomials) with lagrange polymials and sum
    return p

if __name__ == '__main__':
    #-------------------------------------------------------------------------
    # cmfs:
    #cmf_1931_2_lx = deal_with_nan_input(lx.utils.getdata('./luxpy/data/cmfs/ciexyz_1931_2.dat')).T
    cmf_1931_2_cie = deal_with_nan_input(lx.utils.getdata('../../data/cmfs/cie/CIE_xyz_1931_2deg.csv')).T
    #cmf_1931_2_cvrl_5nm = deal_with_nan_input(lx.utils.getdata('../../data/cmfs/cvrl.org/ciexyz31.csv')).T
    cmf_1931_2_cie_5nm = deal_with_nan_input(lx.utils.getdata('../../data/cmfs/cie/CIE_015_4_Data_1931_2_5nm.csv')).T
    print('\nCIE 1931 2°:')
    #compare_cmfs(cmf_1931_2_lx, cmf_1931_2_cie, relative_xyz = True, nans_to_zero = True)
    #compare_cmfs(None, cmf_1931_2_cie, relative_xyz = True, nans_to_zero = True)
    
    #--------------------
    # Test impact of different interpolators (1nm->5nm, 380 nm - 780 nm):
    # cmf_1931_2_cie_5nm_np_interp = np.vstack((cmf_1931_2_cie_5nm[0],np.array([np.interp(cmf_1931_2_cie_5nm[0],cmf_1931_2_cie[0],cmf_1931_2_cie[i+1]) for i in range(3)])))
    # cmf_1931_2_cie_5nm_sp_interp1d = np.vstack((cmf_1931_2_cie_5nm[0],np.array([interpolate.interp1d(cmf_1931_2_cie[0],cmf_1931_2_cie[i+1], kind = 'linear')(cmf_1931_2_cie_5nm[0]) for i in range(3)])))
    # cmf_1931_2_cie_5nm_sp_interpolatedunivariatespline = np.vstack((cmf_1931_2_cie_5nm[0],np.array([interpolate.InterpolatedUnivariateSpline(cmf_1931_2_cie[0],cmf_1931_2_cie[i+1], k = 1)(cmf_1931_2_cie_5nm[0]) for i in range(3)])))
    #compare_cmfs(cmf_1931_2_cie_5nm, np.round(cmf_1931_2_cie_5nm_np_interp,6), relative_xyz = True, nans_to_zero = True)
    #compare_cmfs(cmf_1931_2_cie_5nm, np.round(cmf_1931_2_cie_5nm_sp_interp1d,6), relative_xyz = True, nans_to_zero = True)
    # compare_cmfs(cmf_1931_2_cie_5nm_np_interp, cmf_1931_2_cie_5nm_sp_interp1d, relative_xyz = True, nans_to_zero = True)
    # compare_cmfs(cmf_1931_2_cie_5nm_sp_interpolatedunivariatespline, cmf_1931_2_cie_5nm_sp_interp1d, relative_xyz = True, nans_to_zero = True)
    
    #=======================
    # Test impact of different interpolators (5nm->1nm, 380 nm - 780 nm):
    wl = lx.getwlr([380,780,1])
    cmf_1931_2_cie_1nm_np_interp = np.vstack((wl,np.array([np.interp(wl,cmf_1931_2_cie_5nm[0],cmf_1931_2_cie_5nm[i+1]) for i in range(3)])))
    cmf_1931_2_cie_1nm_sp_interp1d = np.vstack((wl,np.array([interpolate.interp1d(cmf_1931_2_cie_5nm[0],cmf_1931_2_cie_5nm[i+1], kind = 'linear')(wl) for i in range(3)])))
    cmf_1931_2_cie_1nm_sp_interpolatedunivariatespline = np.vstack((wl,np.array([interpolate.InterpolatedUnivariateSpline(cmf_1931_2_cie_5nm[0],cmf_1931_2_cie_5nm[i+1], k = 1)(wl) for i in range(3)])))
    cmf_1931_2_cie_1nm_sp_interpolatedunivariatespline_cubic = np.vstack((wl,np.array([interpolate.InterpolatedUnivariateSpline(cmf_1931_2_cie_5nm[0],cmf_1931_2_cie_5nm[i+1], k = 3)(wl) for i in range(3)])))
    cmf_1931_2_cie_1nm_lx_sprague5 = np.vstack((wl,np.array([lx.math.interp1_sprague5(cmf_1931_2_cie_5nm[0],cmf_1931_2_cie_5nm[i+1], wl)[0] for i in range(3)])))
    cmf_1931_2_cie_1nm_lx_lagrange = np.vstack((wl,np.array([interp_lagrange(cmf_1931_2_cie_5nm[0],cmf_1931_2_cie_5nm[i+1], wl) for i in range(3)])))
    #from scipy.interpolate import lagrange
    #cmf_1931_2_cie_1nm_sp_lagrange = np.vstack((wl,np.array([lagrange(cmf_1931_2_cie_5nm[0],cmf_1931_2_cie_5nm[i+1])(wl) for i in range(3)])))

    digits = 13
    #-------------------
    # Test if linear interpolation can genereta the 5 nm data published by the CIE:
    # compare_cmfs(np.round(cmf_1931_2_cie[:,20:-50],digits), np.round(cmf_1931_2_cie_1nm_np_interp,digits), relative_xyz = True, nans_to_zero = True)
    # compare_cmfs(np.round(cmf_1931_2_cie[:,20:-50],digits), np.round(cmf_1931_2_cie_1nm_sp_interpolatedunivariatespline,digits), relative_xyz = True, nans_to_zero = True)
    # plot_dcmfs(np.round(cmf_1931_2_cie[:,20:-50],digits), np.round(cmf_1931_2_cie_1nm_sp_interpolatedunivariatespline,digits))
    # print('None of the 3 linear interpolators can generate the 5 nm data published by CIE!')
    
    #-------------------
    # Test if the 3 different linear interpolators agree:
    #compare_cmfs(cmf_1931_2_cie_1nm_np_interp, cmf_1931_2_cie_1nm_sp_interp1d, relative_xyz = True, nans_to_zero = True)
    #compare_cmfs(cmf_1931_2_cie_1nm_sp_interpolatedunivariatespline, cmf_1931_2_cie_1nm_sp_interp1d, relative_xyz = True, nans_to_zero = True)
    #plot_dcmfs(cmf_1931_2_cie_1nm_sp_interpolatedunivariatespline, cmf_1931_2_cie_1nm_sp_interp1d)
    #print('InterpolatedUnivariateSpline with k = 1 (linear) differs from the other two, but only very little of the order of 1e-16')

    #-------------------
    # Test if perhaps interpolation from 5 nm to 1 nm was done using a cubic interpolation:
    # compare_cmfs(np.round(cmf_1931_2_cie[:,20:-50],digits), np.round(cmf_1931_2_cie_1nm_sp_interpolatedunivariatespline_cubic,digits), relative_xyz = True, nans_to_zero = True)
    # plot_dcmfs(np.round(cmf_1931_2_cie[:,20:-50],digits), np.round(cmf_1931_2_cie_1nm_sp_interpolatedunivariatespline_cubic,digits))
    # print('A cubic interpolation cannot generate the 5 nm data published by CIE!')
    
    #-------------------
    # Test if interpolation from 5 nm to 1 nm was done using a Sprague interpolation:
    #compare_cmfs(np.round(cmf_1931_2_cie[:,20:-50],digits), np.round(cmf_1931_2_cie_1nm_lx_sprague5,digits), relative_xyz = True, nans_to_zero = True)
    #plot_dcmfs(np.round(cmf_1931_2_cie[:,20:-50],digits), np.round(cmf_1931_2_cie_1nm_lx_sprague5,digits))
    #print('A sprague interpolation cannot generate the 5 nm data published by CIE!')
    
    #-------------------
    # Test if interpolation from 5 nm to 1 nm was done using a Lagrange interpolation:
    compare_cmfs(np.round(cmf_1931_2_cie[:,20:-50],digits), np.round(cmf_1931_2_cie_1nm_lx_lagrange,digits), relative_xyz = True, nans_to_zero = True)
    axh, dcmf = plot_dcmfs(np.round(cmf_1931_2_cie[:,20:-50],digits), np.round(cmf_1931_2_cie_1nm_lx_lagrange,digits))
    print('A Lagrange interpolation fails ??? -> cannot generate the 5 nm data published by CIE!')


    #===================
    # Test agreement for other CMF sets between lx data (downloaded from CIE in the past) and the current data published at: cie.co.at/data-tables    
    # cmf_1964_10_lx = deal_with_nan_input(lx.utils.getdata('./luxpy/data/cmfs/ciexyz_1964_10.dat')).T
    # cmf_1964_10_cie = deal_with_nan_input(lx.utils.getdata('../../data/cmfs/cie/CIE_xyz_1964_10deg.csv')).T
    # print('\nCIE 1964 10°:')
    # compare_cmfs(cmf_1964_10_lx, cmf_1964_10_cie, relative_xyz = True, nans_to_zero = True)
    # compare_cmfs(None, cmf_1964_10_cie, relative_xyz = True, nans_to_zero = True)

    # cmf_2006_2_lx = deal_with_nan_input(lx.utils.getdata('./luxpy/data/cmfs/ciexyz_2006_2.dat')).T
    # cmf_2006_2_cie = deal_with_nan_input(lx.utils.getdata('../../data/cmfs/cie/CIE_cfb_stv_2deg.csv')).T
    # print('\nCIE 2006 2°:')
    # compare_cmfs(cmf_2006_2_lx, cmf_2006_2_cie, relative_xyz = True, nans_to_zero = True)
    #compare_cmfs(None, cmf_2006_2_cie, relative_xyz = True, nans_to_zero = True)

    # cmf_2006_10_lx = deal_with_nan_input(lx.utils.getdata('./luxpy/data/cmfs/ciexyz_2006_10.dat')).T
    # cmf_2006_10_cie = deal_with_nan_input(lx.utils.getdata('../../data/cmfs/cie/CIE_cfb_stv_10deg.csv')).T
    # print('\nCIE 2006 10°:')
    # compare_cmfs(cmf_2006_10_lx, cmf_2006_10_cie, relative_xyz = True, nans_to_zero = True)
    #compare_cmfs(None, cmf_2006_10_cie, relative_xyz = True, nans_to_zero = True)