# -*- coding: utf-8 -*-
"""
Standalone Robertson1968 implementation 
=======================================
 
 (includes correction near slope-sign-change of iso-temperature-lines)

 :cct_to_xyz(): Calculates xyz from CCT, Duv by estimating the line perpendicular to the planckian locus (=iso-T line).

 :cct_to_xyz(): Calculates xyz from CCT, Duv [_CCT_MIN < CCT < _CCT_MAX]
 
 
References:
   1. `Robertson, A. R. (1968). 
   Computation of Correlated Color Temperature and Distribution Temperature. 
   Journal of the Optical Society of America,  58(11), 1528–1535. 
   <https://doi.org/10.1364/JOSA.58.001528>`_
   
   2. Smet K.A.G., Royer M., Baxter D., Bretschneider E., Esposito E., Houser K., Luedtke W., Man K., Ohno Y. (2022),
   Recommended method for determining the correlated color temperature and distance from the Planckian Locus of a light source
   (in preparation, LEUKOS?)
   
   3. Baxter D., Royer M., Smet K.A.G. (2022)
   Modifications of the Robertson Method for Calculating Correlated Color Temperature to Improve Accuracy and Speed
   (in preparation, LEUKOS?)
   
Created on Thu June 8 15:36:29 2022

@author: ksmet1977 [at] gmail.com
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get path to module:
_PATH = os.path.dirname(__file__);""" Absolute path to module """ 
_SEP = os.sep; """ Operating system separator """

__all__ = ['_CCT_LUT_WL3','_CCT_AVOID_ZERO_DIV','_CCT_AVOID_INF','_CCT_MIN','_CCT_MAX',
          '_CCT_FAST_DUV','_CCT_MAX_ITER','_CCT_SPLIT_CALC_AT_N',
          '_CCT_LUT_PATH','_CCT_LUT','_CCT_LIST_OF_CIEOBS_LUTS','_CCT_CIEOBS',
          '_BB','_WL3',
          'save_pkl','load_pkl','get_tcs4','calculate_lut',
          'xyz_to_cct','xyz_to_duv','cct_to_xyz']

#==============================================================================
# define global variables:
#==============================================================================
_CCT_LUT_WL3 = [360,830,1]
_CCT_AVOID_ZERO_DIV = 1e-100
_CCT_AVOID_INF = 1/_CCT_AVOID_ZERO_DIV

_CCT_MIN = 450 
_CCT_MAX = 1e11 # don't set to higher value to avoid overflow and errors

_CCT_FAST_DUV = True # use a fast, but slightly less accurate Duv calculation with Newton-Raphson
_CCT_MAX_ITER = 10
_CCT_SPLIT_CALC_AT_N = 25 # some tests show that this seems to be the fastest (for 1000 conversions)

_CCT_LUT_PATH = _PATH + _SEP + 'luts'+ _SEP #folder with cct lut data
_CCT_LUT_CALC = False
_CCT_LUT = {}
_CCT_USE_EXTENDED_LUT = False # True doesn't really lead to a gain in computatation time

_CCT_LIST_OF_CIEOBS_LUTS = ['1931_2','1964_10','2015_2','2015_10']
_CCT_CIEOBS = '1931_2' # default CIE observer


#--------------------------------------------------------------------------------------------------
# set coefficients for blackbody radiators (c2 rounded to 1.4388e-2 as defiend for ITS-90 International Temperature Scale):
_BB = {'c1' : 3.74177185e-16, 'c2' : np.round(1.4387768775e-2,6),'n': 1.000, 'na': 1.00028, 'c' : 299792458, 'h' : 6.62607015e-34, 'k' : 1.380649e-23} # blackbody c1,c2 & n standard values (h,k,c from NIST, CODATA2018)


#==============================================================================
# Function definitions:
#==============================================================================

#------------------------------------------------------------------------------
# Luxpy copies:
#------------------------------------------------------------------------------
_WL3 = _CCT_LUT_WL3
def save_pkl(filename, obj): 
    """ 
    Save an object in a pickle file.
    
    Args:
        :filename:
            | str with filename of pickle file.
        :obj:
            | python object to save
    
    Returns:
        :None:
    """
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_pkl(filename):
    """ 
    Load the object in a pickle file.
    
    Args:
        :filename:
            | str with filename of pickle file.
        
    Returns:
        :obj:
            | loaded python object
    """
    obj = None
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


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


def getwlr(wl3 = None):
    """
    Get/construct a wavelength range from a 3-vector (start, stop, spacing).
    
    Args:
        :wl3: 
            | list[start, stop, spacing], optional 
            | (defaults to luxpy._WL3)

    Returns:
        :returns: 
            | ndarray (.shape = (n,)) with n wavelengths ranging from
            | start to stop, with wavelength interval equal to spacing.
    """
    if wl3 is None: wl3 = _WL3
    
    # Wavelength definition:
    wl = wl3 if (len(wl3) != 3) else np.arange(wl3[0], wl3[1] + wl3[2], wl3[2]) # define wavelengths from [start = l0, stop = ln, spacing = dl]
    return wl

#------------------------------------------------------------------------------
def getwld(wl):
    """
    Get wavelength spacing. 
    
    Args:
        :wl: 
            | ndarray with wavelengths
        
    Returns:
        :returns: 
            | - float:  for equal wavelength spacings
            | - ndarray (.shape = (n,)): for unequal wavelength spacings
    """
    d = np.diff(wl)
    # dl = (np.hstack((d[0],d[0:-1]/2.0,d[-1])) + np.hstack((0.0,d[1:]/2.0,0.0)))
    dl = np.hstack((d[0],(d[0:-1] + d[1:])/2.0,d[-1]))
    if (dl == dl.mean()).all(): dl = dl[0]
    return dl

#------------------------------------------------------------------------------
# Load CMFs:
#------------------------------------------------------------------------------
_CMF = load_pkl(os.path.join(_CCT_LUT_PATH,'CMFs.pkl'))

#------------------------------------------------------------------------------
# XYZ to chromaticity conversion functions  
#------------------------------------------------------------------------------
def xyz_to_Yxy(xyz, **kwargs):
    """
    Convert XYZ tristimulus values CIE Yxy chromaticity values.

    Args:
        :xyz: 
            | ndarray with tristimulus values

    Returns:
        :Yxy: 
            | ndarray with Yxy chromaticity values
            |  (Y value refers to luminance or luminance factor)
    """
    xyz = np.atleast_2d(xyz)
    Yxy = np.empty(xyz.shape)
    sumxyz = xyz[...,0] + xyz[...,1] + xyz[...,2]
    Yxy[...,0] = xyz[...,1]
    Yxy[...,1] = xyz[...,0] / sumxyz
    Yxy[...,2] = xyz[...,1] / sumxyz
    return Yxy


def Yxy_to_xyz(Yxy, **kwargs):
    """
    Convert CIE Yxy chromaticity values to XYZ tristimulus values.

    Args:
        :Yxy: 
            | ndarray with Yxy chromaticity values
            |  (Y value refers to luminance or luminance factor)

    Returns:
        :xyz: 
            | ndarray with tristimulus values
    """
    Yxy = np.atleast_2d(Yxy)
    xyz = np.empty(Yxy.shape)
    xyz[...,1] = Yxy[...,0]
    xyz[...,0] = Yxy[...,0]*Yxy[...,1]/Yxy[...,2]
    xyz[...,2] = Yxy[...,0]*(1.0-Yxy[...,1]-Yxy[...,2])/Yxy[...,2]
    
    return xyz

# Direct definition causes slight rounding errors with luxpy implementation
# def xyz_to_Yuv60(xyz):
#     """
#     Convert XYZ tristimulus values CIE 1960 Y,u,v chromaticity values.

#     Args:
#         :xyz: 
#             | ndarray with tristimulus values

#     Returns:
#         :Yuv: 
#             | ndarray with CIE 1960 Y,u,v chromaticity values
#             |  (Y value refers to luminance or luminance factor)
#     """
#     xyz = np.atleast_2d(xyz)
#     Yuv = np.empty(xyz.shape)
#     denom = xyz[...,0] + 15.0*xyz[...,1] + 3.0*xyz[...,2]
#     Yuv[...,0] = xyz[...,1]
#     Yuv[...,1] = 4.0*xyz[...,0] / denom
#     Yuv[...,2] = 6.0*xyz[...,1] / denom
#     return Yuv


# def Yuv60_to_xyz(Yuv60):
#     """
#     Convert CIE 1960 Y,u,v chromaticity values to XYZ tristimulus values.

#     Args:
#         :Yuv: 
#             | ndarray with CIE 1960 Y,u,v chromaticity values
#             |  (Y value refers to luminance or luminance factor)

#     Returns:
#         :xyz: 
#             | ndarray with tristimulus values
#     """
#     Yuv = np.atleast_2d(Yuv60)
#     xyz = np.empty(Yuv.shape)
#     xyz[...,1] = Yuv[...,0]
#     xyz[...,0] = Yuv[...,0]*(3.0*Yuv[...,1])/(2.0*Yuv[...,2])
#     xyz[...,2] = Yuv[...,0]*(4.0 - Yuv[...,1] - 10.0*Yuv[...,2])/(2.0*Yuv[...,2])
#     return xyz

def xyz_to_Yuv76(xyz,**kwargs):
    """
    Convert XYZ tristimulus values CIE 1976 Y,u',v' chromaticity values.

    Args:
        :xyz: 
            | ndarray with tristimulus values

    Returns:
        :Yuv: 
            | ndarray with CIE 1976 Y,u',v' chromaticity values
            |  (Y value refers to luminance or luminance factor)
    """
    xyz = np.atleast_2d(xyz)
    Yuv = np.empty(xyz.shape)
    denom = xyz[...,0] + 15.0*xyz[...,1] + 3.0*xyz[...,2]
    Yuv[...,0] = xyz[...,1]
    Yuv[...,1] = 4.0*xyz[...,0] / denom
    Yuv[...,2] = 9.0*xyz[...,1] / denom
    return Yuv

def Yuv76_to_xyz(Yuv, **kwargs):
    """
    Convert CIE 1976 Y,u',v' chromaticity values to XYZ tristimulus values.

    Args:
        :Yuv: 
            | ndarray with CIE 1976 Y,u',v' chromaticity values
            |  (Y value refers to luminance or luminance factor)

    Returns:
        :xyz: 
            | ndarray with tristimulus values
    """
    Yuv = np.atleast_2d(Yuv)
    xyz = np.empty(Yuv.shape)
    xyz[...,1] = Yuv[...,0]
    xyz[...,0] = Yuv[...,0]*(9.0*Yuv[...,1])/(4.0*Yuv[...,2])
    xyz[...,2] = Yuv[...,0]*(12.0 - 3.0*Yuv[...,1] - 20.0*Yuv[...,2])/(4.0*Yuv[...,2])
    return xyz


def xyz_to_Yuv60(xyz,**kwargs):
    """
    Convert XYZ tristimulus values CIE 1960 Y,u,v chromaticity values.

    Args:
        :xyz: 
            | ndarray with tristimulus values

    Returns:
        :Yuv: 
            | ndarray with CIE 1960 Y,u,v chromaticity values
            |  (Y value refers to luminance or luminance factor)
    """
    Yuv = xyz_to_Yuv76(xyz,**kwargs)
    Yuv[...,2] *= 2/3 
    return Yuv


def Yuv60_to_xyz(Yuv, **kwargs):
    """
    Convert CIE 1976 Y,u,v chromaticity values to XYZ tristimulus values.

    Args:
        :Yuv: 
            | ndarray with CIE 1976 Yu'v' chromaticity values
            |  (Y value refers to luminance or luminance factor)

    Returns:
        :xyz: 
            | ndarray with tristimulus values
    """
    Yuv = np.atleast_2d(Yuv.copy())
    Yuv[...,2] *= 3/2 
    return Yuv76_to_xyz(Yuv,**kwargs)


#------------------------------------------------------------------------------
# XYZ calculation, CMFs and CMFs conversion (apply color space transform already to CMFs!)
#------------------------------------------------------------------------------
import luxpy as lx
def _get_xyzbar_wl_dl(cieobs):
    """
    Get the xyzbar CMF set corresponding to cieobs.
    Returns an ndarray with the cmfs (stripped of wavelengths), ndarray with
    wavelengths and an ndarray with the wavelength differences.
    """
    # get requested cmf set:
    cmf = _CMF[cieobs]['bar'].copy() if isinstance(cieobs,str) else cieobs.copy()
    wl, cmf = cmf[0], cmf[1:]
    dl = getwld(wl)*1.0  # get wavelength difference
    c = ~(((cmf[1:]==0).sum(0)==3))
    cmf[:,c] += _CCT_AVOID_ZERO_DIV # avoid nan's in uvwvbar
    return cmf, wl, dl

def _convert_xyzbar_to_uvwbar(xyzbar):
    """
    Convert the xyzbar (no wl on row 0!) CMF set to a CMF set representing a different 
    color space/ chromaticity diagram (integration leads to new tristimulus values)
    Returns an ndarray (no wl on row 0!) of new CMFs.
    """
    # convert to cspace based cmfs (Eq.6-7):
    Yuvbar = xyz_to_Yuv60(xyzbar.T) # convert to chromaticity format from xyz (cfr. cmf) format
    uvwbar = Yxy_to_xyz(Yuvbar).T # convert from chromaticity format (Vuv) to tristimulus (UVW) format and take transpose (=spectra)
    return uvwbar


#------------------------------------------------------------------------------
# Planckian spectrum and colorimetric calculations:
#------------------------------------------------------------------------------
def _get_BB_BBp_BBpp(T, wl, out = 'BB,BBp,BBpp'):
    """ 
    Get the blackbody radiatior spectrum, and the spectra corresponding to 
    the first and second derivatives to Tc of the blackbody radiator.
    """
    BBp,BBpp = None,None
    T = np.atleast_2d(T)*1.0 # force float
    wlt = wl*1.0e-9
    c_wl_T = _BB['c2']/(wlt*T)
    exp = np.exp(c_wl_T)
    exp[np.isinf(exp)] = _CCT_AVOID_INF
    
    # avoid div by inf or zero:
    exp_min_1 = exp - 1.0
    exp_min_1[exp_min_1==0] = _CCT_AVOID_ZERO_DIV
    exp_min_1[np.isinf(exp_min_1)] = _CCT_AVOID_INF
        
    BB = _BB['c1']*(wlt**(-5))*(1/(exp_min_1))
    BB[np.isinf(BB)] = _CCT_AVOID_INF
        
    if ('BBp' in out) | ('BBpp' in out): 
        
        exp_min_1_squared = exp_min_1**2
        
        # avoid div by inf or zero:
        exp_min_1_squared[np.isinf(exp_min_1_squared)] = _CCT_AVOID_INF # avoid warning "invalid value encountered in true_divide"
        exp_min_1_squared[exp_min_1_squared == 0.0] = _CCT_AVOID_ZERO_DIV
        
        exp_frac = exp/exp_min_1_squared

        BBp = (_BB['c1']*_BB['c2']*(T**(-2))*(wlt**(-6)))*exp_frac

        
    if 'BBpp' in out:
        exp_plus_1 = exp + 1.0
        BBpp = (BBp/T) * (c_wl_T * (exp_plus_1 / exp_min_1)  - 2) 
        
        
    return BB, BBp, BBpp


def _get_tristim_of_BB_BBp_BBpp(T, xyzbar, wl, dl, out = 'BB,BBp,BBpp'):
    """ 
    Get the tristimulus values for CMF set xyzbar of the blackbody radiatior spectra
    and the spectra corresponding to the first and second derivatives to Tc 
    of the blackbody radiator.
    """
    xyzp,xyzpp = None, None
    BB, BBp, BBpp =  _get_BB_BBp_BBpp(T, wl, out = out)
    #cnd = np.ones((BB.shape[-1],),dtype=bool)#((xyzbar>0).sum(0)>0).T # keep only wavelengths where not all 3 cmfs are equal (to avoid nan's for 2015 cmfs which are defined only between 390 and 830 nm)
    xyz = ((BB * dl) @ xyzbar.T)

    if 'BBp' in out.split(','): 
        xyzp = ((BBp * dl) @ xyzbar.T)
        xyzp[np.isinf(xyzp)] = _CCT_AVOID_INF # # avoid warning "invalid value encountered in subtract" when calculating li

    if 'BBpp' in out.split(','): 
        xyzpp = ((BBpp * dl) @ xyzbar.T)
        xyzpp[np.isinf(xyzpp)] = _CCT_AVOID_INF

    return T, xyz, xyzp, xyzpp


def _get_uv_uvp_uvpp(T, uvwbar, wl, dl, out = 'BB,BBp,BBpp'):
    """ 
    Get the (u,v), (u',v') and (u",v") coordinates of one or more Planckians
    with specified Tc. uvwbar (no wavelengths on row0, these are supplied seperately
    in wl, with wavelength spacing in dl) is the cmf set corresponding to the tristimulus values
    of the chosen chromaticity diagram or color space to do the CCT calculations in.
    See: Li et al. (2016). Accurate method for computing correlated color temperature. Optics Express, 24(13), 14066–14078.

    """
    # calculate U,V,W (Eq. 6) and U',V',W' (Eq.10) [Robertson,1986] and U",V",W" [Li,2016; started from XYZ, but this is equivalent]:
    T, UVW, UVWp, UVWpp = _get_tristim_of_BB_BBp_BBpp(T, uvwbar, wl, dl, out = out)
    T = T[:,None]
    
    # get u,v & u',v' and u",v":
    S = (UVW[...,0] + UVW[...,1] + UVW[...,2])[:,None]
    uv = (UVW[...,:2] / (S + _CCT_AVOID_ZERO_DIV))
    u,v = uv[...,0:1], uv[...,1:2]

    if UVWp is not None:
        Sp = (UVWp[...,0] + UVWp[...,1] + UVWp[...,2])[:,None]
        PQ = (UVWp[...,:2] * S - UVW[...,:2] * Sp)
        uvp = (PQ / ((S**2) + _CCT_AVOID_ZERO_DIV))
        up,vp = uvp[...,0:1], uvp[...,1:2]
    else:
        up,vp = None, None
        
    if (UVWpp is not None) & (UVWp is not None):
        Spp = (UVWpp[...,0] + UVWpp[...,1] + UVWpp[...,2])[:,None]
        PQp = (UVWpp[...,:2] * S - UVW[...,:2] * Spp)
        uvpp = ((PQp * S - 2 * PQ *Sp) / ((S**3) + _CCT_AVOID_ZERO_DIV))
        upp,vpp = uvpp[...,0:1], uvpp[...,1:2]
    else:
        upp, vpp = None, None
    
    return T, u, v, up, vp, upp, vpp, (UVW, UVWp, UVWpp)


#------------------------------------------------------------------------------
# Calculate %, K LUT
#------------------------------------------------------------------------------

def get_tcs4(tc4, cct_min = _CCT_MIN, cct_max = _CCT_MAX):
    """ 
    Generate list of Tc of Planckians from (Tmin, Tmax inclusive, Tincrement, unit) 
    
    Args:
        :tc4:
            | 4-element list or tuple
            | Elements are: [Tmin, Tmax inclusive, Tincrement, unit]
            |  Unit specifies unit of the Tc interval, i.e. it determines the
            |       type of scale in which the spacing of the Tc are done.
            |  Unit options are:
            |   - '%': equal relative Tc spacing (in %, cfr. (Ti+1 - Ti-1)/Ti-1).
            |   - 'K' equal absolute Tc spacing (in K, cfr. (Ti+1 - Ti-1).
            |   - '%-1': equal relative reciprocal Tc (MK-1 = mired).
            |   - 'K-1': equal absolute reciprocal Tc (MK-1 = mired).
            |  If the 'increment' element is negative, it actually represents
            |  the number of intervals between Tmin, Tmax (included).
        :cct_min:
            | _CCT_MIN, optional        
            | Limit Tc's to a minimum value of cct_min
        :cct_max:
            | _CCT_MAX, optional
            | Limit Tc's to a maximum value of cct_max

    Returns:
        :Tcs:
            | ndarray [N,1] of ccts. 
    """
    
    (T0,Tn,dT),u = (np.atleast_1d(tc4_i) for tc4_i in tc4[:-1]), tc4[-1] # min, max, interval, unit
    
    # Get n from third element of input 4-vector:
    if ((dT<0).any() & (dT>=0).any()):
        raise Exception('3e element [dT,n] in 4-vector tc4 contains negatives AND positives! Should be only 1 type.')
    else:
        n = np.abs(dT) if (dT<0).all() else None # dT contains number of tcs between T0 and Tn, not dT
         
    # calculate Ts for different unit types:
    if 'K' in u:
        if n is None:
            n = (((Tn-T0)//dT) + (((Tn-T0)%dT)!=0)).max() # get n from dT
        else: 
            dT = (Tn - T0)/n # get dT from n
            n = n.max() 
            
        Ts = (T0[:,None] + np.arange(0, n + 1,1)*dT[:,None]).T # to include Tn
    
    elif '%' in u:
        if n is None:
            p = 1 + dT/100
            n = (((np.log(Tn/T0) / np.log(p))).max()) # get n from dT
        else:
            p = (Tn/T0)**(1/n) # get p = (1+dT/100) from n
            n = n.max()
        Ts = T0*p**np.arange(0, n + 1, 1)[:,None]
    
    if '-1' in u:
        Ts[Ts==0] = _CCT_AVOID_ZERO_DIV
        Ts = 1e6/Ts[::-1] # scale was in mireds, so flip it
        
    Ts[(Ts<cct_min)] = cct_min
    Ts[(Ts>cct_max)] = cct_max # limit to a maximum cct to avoid overflow/error and/or increase speed.    

    return Ts 



def calculate_lut(ccts, cieobs, lut_vars = ['T','uv','uvp','uvpp','iso-T-slope'],
                  cct_min = _CCT_MIN, cct_max = _CCT_MAX):
    """
    Function that calculates a LUT for the specified calculation method 
    for the input ccts. Calculation is performed for CMF set specified in 
    cieobs and in the chromaticity diagram in cspace. 
    
    Args:
        :ccts: 
            | ndarray [Nx1] or str or 4-element tuple
            | If ndarray: list of ccts for which to (re-)calculate the LUTs.
            | If str: path to file containing CCTs (no header; sep = ',')
            | If 4-element tuple: generate ccts from (Tmin, Tmax, increment, unit) specifier 
        :cieobs: 
            | None or str, optional
            | str specifying cmf set.
        :lut_vars:
            | ['T','uv','uvp','uvpp','iso-T-slope'], optional
            | Data the lut should contain. Must follow this order 
            | and minimum should be ['T']
        :cct_min:
            | _CCT_MIN, optional        
            | Limit Tc's to a minimum value of cct_min
        :cct_max:
            | _CCT_MAX, optional
            | Limit Tc's to a maximum value of cct_max
            
    Returns:
        :returns: 
            :lut:
                | ndarray with T, u, v, u', v', u", v", slope (note ':1st deriv., ":2nd deriv.).
                                            
    """   
    # Get ndarray with Planckian temperatures, Ts:
    if isinstance(ccts, str):
        ccts = pd.read_csv(ccts, names = None, index_col = None, header = None, sep = ',').values
    elif isinstance(ccts, tuple):
        if len(ccts) == 4:
            ccts = get_tcs4(ccts, cct_min = cct_min, cct_max = cct_max)
            
    if ('uv' not in lut_vars) & ('uvp' not in lut_vars) & ('uvpp' not in lut_vars) & ('iso-T-slope' not in lut_vars):
        # no need to calculate anything, only Tcs needed
        return np.atleast_2d(ccts)
    
    # Determine what to calculate:
    outBB = 'BB'
    if ('uvp' in lut_vars) | ('iso-T-slope' in lut_vars): 
        outBB = outBB + 'uvp'
    if ('uvpp' in lut_vars): outBB = outBB + 'uvpp'
 
    # get requested cmf set:
    xyzbar, wl, dl = _get_xyzbar_wl_dl(cieobs)
        
    # convert to cspace based cmfs (Eq.6-7):
    uvwbar = _convert_xyzbar_to_uvwbar(xyzbar) 
    
    # calculate U,V,W (Eq. 6) and U',V',W' (Eq.10) [Robertson,1986] and U",V",W" [Li,2016; started from XYZ, but this is equivalent]:
    #Ti, UVW, UVWp, UVWpp = _get_tristim_of_BB_BBp_BBpp(ccts, uvwbar, wl, dl, out = 'BB,BBp,BBpp')
    _, u, v, up, vp, upp, vpp, (UVW, UVWp, UVWpp) = _get_uv_uvp_uvpp(ccts, uvwbar, wl, dl, out = 'BB,BBp,BBpp')
    Ti = ccts
    if 'uv' in lut_vars: uvi = np.hstack((u,v))
    if 'uvp' in lut_vars: uvpi = np.hstack((up,vp))
    if 'uvpp' in lut_vars: uvppi = np.hstack((upp,vpp))
    
    
    # calculate li, mi (= slope of iso-T-lines):
    if 'iso-T-slope' in lut_vars:
        
        R = UVW.sum(axis=-1, keepdims = True) # for Ohno, 2014 & Robertson, 1968 & Li, 2016
        if UVWp is not None: Rp = UVWp.sum(axis=-1, keepdims = True) # for Robertson, 1968 & Li, 2016
        # if UVWpp is not None: Rpp = UVWpp.sum(axis=-1, keepdims = True) # for Li, 2016

        num = (UVWp[:,1:2]*R - UVW[:,1:2]*Rp) 
        denom = (UVWp[:,:1]*R - UVW[:,:1]*Rp)
        
        # avoid div by zero:
        num[(num == 0)] += _CCT_AVOID_ZERO_DIV
        denom[(denom == 0)] += _CCT_AVOID_ZERO_DIV
    
        li = num/denom  
        li = li + np.sign(li)*_CCT_AVOID_ZERO_DIV # avoid division by zero
        mi = -1.0/li # slope of isotemperature lines
        
        sq1pmi2 = (1+mi**2)**0.5
        sign_mimip1 = np.sign(mi*np.vstack((mi[1:],mi[-1:]))) # use same value for last one
    else:
        mi = None
    
    # # get u,v & u',v' and u",v":
    # uvi = UVW[:,:2]/R
    # if UVWp is not None: uvpi = UVWp[:,:2]/Rp
    # if UVWpp is not None: uvppi = UVWpp[:,:2]/Rpp
    
    # construct output (use comple if structure to avoid creating intermediate arrays for optimal speed):
    if  ('uvp' in lut_vars) & ('uvpp' in lut_vars) & ('iso-T-slope' in lut_vars):
        if  ('sqrt(1+iso-T-slope**2)' in lut_vars) & ('sign(iso-T-slope_i*i+1)' in lut_vars):
            lut = np.hstack((Ti,uvi,uvpi,uvppi,mi,sq1pmi2,sign_mimip1))
        else:
            lut = np.hstack((Ti,uvi,uvpi,uvppi,mi))
        
    elif ('uvp' not in lut_vars) & ('uvpp' not in lut_vars) & ('iso-T-slope' not in lut_vars):
        lut = np.hstack((Ti,uvi))
        
    elif ('uvp' in lut_vars) & ('uvpp' not in lut_vars) & ('iso-T-slope' in lut_vars):
        if  ('sqrt(1+iso-T-slope**2)' in lut_vars) & ('sign(iso-T-slope_i*i+1)' in lut_vars):
            lut = np.hstack((Ti,uvi, uvpi, mi, sq1pmi2, sign_mimip1))
        else:
            lut = np.hstack((Ti,uvi, uvpi, mi))
        
    elif ('uvp' in lut_vars) & ('uvpp' not in lut_vars) & ('iso-T-slope' not in lut_vars):
        lut = np.hstack((Ti,uvi, uvpi))
    elif ('uvp' in lut_vars) & ('uvpp' in lut_vars) & ('iso-T-slope' not in lut_vars):
        lut = np.hstack((Ti,uvi, uvpi, uvppi))
        
    elif ('uvp' not in lut_vars) & ('uvpp' in lut_vars) & ('iso-T-slope' in lut_vars):
        if  ('sqrt(1+iso-T-slope**2)' in lut_vars) & ('sign(iso-T-slope_i*i+1)' in lut_vars):
            lut = np.hstack((Ti,uvi, uvppi, mi, sq1pmi2, sign_mimip1))
        else:
            lut = np.hstack((Ti,uvi, uvppi, mi))

    elif ('uvp' not in lut_vars) & ('uvpp' in lut_vars) & ('iso-T-slope' not in lut_vars):
        lut = np.hstack((Ti,uvi, uvppi))
           
    elif ('uvp' not in lut_vars) & ('uvpp' not in lut_vars) & ('iso-T-slope' in lut_vars):

        if  ('sqrt(1+iso-T-slope**2)' in lut_vars) & ('sign(iso-T-slope_i*i+1)' in lut_vars):
            lut = np.hstack((Ti,uvi, mi, sq1pmi2, sign_mimip1))
        else:
            lut = np.hstack((Ti,uvi, mi))

    return lut 

#------------------------------------------------------------------------------
# Duv calculation
#------------------------------------------------------------------------------

def _get_Duv_for_T_from_uvBB(u,v, uBB0, vBB0):
    """ 
    Calculate Duv from (u,v) coordinates of estimated Tc.
    """
    # Get duv: 
    du, dv = u - uBB0, v - vBB0
    Duv = (du**2 + dv**2)**0.5 

    # find sign of duv:
    theta = positive_arctan(du,dv,htype='deg')
    theta[theta>180] = theta[theta>180] - 360
    Duv *= (np.sign(theta))
    return Duv

def _get_Duv_for_T(u,v, T, wl, cieobs, uvwbar,  dl,
                   uBB = None, vBB = None):
    """ 
    Calculate Duv from T by generating a planckian and
    calculating the Euclidean distance to the point (u,v) and
    determing the sign as the v coordinate difference between 
    the test point and the planckian.
    """
    if (uBB is None)  & (vBB is None):
        _,UVWBB,_,_ = _get_tristim_of_BB_BBp_BBpp(T, uvwbar, wl, dl, out='BB')
        uvBB = xyz_to_Yxy(UVWBB)#[...,1:]
        uBB, vBB = uvBB[...,1,None], uvBB[...,2,None]
    
    # Get duv: 
    return _get_Duv_for_T_from_uvBB(u, v, uBB, vBB)

#------------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------------
    

#------------------------------------------------------------------------------
def _get_pns_from_x(x, idx, i = None, m0p = 'm0p'):
    """ 
    Get idx-1, idx and idx +1 from array. 
    Returns [Nx1] ndarray with N = len(idx).
    """
    if 'm' in m0p: 
        idx_m1 = idx-1
        idx_m1[idx_m1 == -1] = 0 # otherwise wraparound error happens
    
    if x.shape[-1]==1:
        if m0p == 'm0p': 
            return x[idx_m1], x[idx], x[idx+1]
        elif m0p == '0p':
            return x[idx], x[idx+1]
        elif m0p == 'mp':
            return x[idx_m1], x[idx+1]
        elif m0p == '0':
            return x[idx]
        elif m0p == 'm0':
            return x[idx_m1], x[idx]
    else:
        if 'p' in m0p: 
            idx_p1 = idx + 1
        if i is None: i = np.arange(idx.shape[0])
        if m0p == 'm0p': 
            return x[idx_m1,i][:,None], x[idx,i][:,None], x[idx_p1,i][:,None]
        elif m0p == '0p':
            return x[idx,i][:,None], x[idx_p1,i][:,None]
        elif m0p == 'mp':
            return x[idx_m1,i][:,None], x[idx_p1,i][:,None]
        elif m0p == '0':
            return x[idx,i][:,None]
        elif m0p == 'm0':
            return x[idx_m1,i][:,None], x[idx,i][:,None]
   
    
def _deal_with_lut_end_points(pn, TBB, out_of_lut = None):
    ce = pn == (TBB.shape[0]-1) # end point
    cb = pn==0 # begin point
    if out_of_lut is None: out_of_lut = (cb | ce)[:,None]
    pn[cb] =  1 # begin point 
    ce = pn == (TBB.shape[0]-1) # end point double-check !!
    pn[ce] = (TBB.shape[0] - 2) # end of lut (results in TBB_0==TBB_p1 -> (1/TBB_0)-(1/TBB_p1)) == 0 !

    return pn, out_of_lut

#------------------------------------------------------------------------------
# Newton-Raphson estimator (cfr. Li, 2016):
#------------------------------------------------------------------------------       
def _get_newton_raphson_estimated_Tc(u, v, T0, atol = 0.1, rtol = 1e-5,
                                     cieobs = None, wl = None, xyzbar = None, uvwbar = None,
                                     max_iter = _CCT_MAX_ITER, fast_duv = _CCT_FAST_DUV):
    """
    Get an estimate of the CCT using the Newton-Raphson method (as specified in 
    Li et al., 2016). (u,v) are the test coordinates. T0 is a first estimate of the Tc.
    atol and rtol are the absolute and relative tolerance values that are aimed at (if
    possible the error on the estimation should smaller than or close to these values,
    once one is achieved the algorithm stops). wl contains the wavelengths of the 
    Planckians, cieobs is the CIE cmfs set to be used (or use xyzbar; at least one
    must be given). uvwbar the already converted cmf set. If this one is not None
    then any input in cieobs or xyzbar is ignored. Max-iter specifies
    the maximum number of iterations (avoid potential infinite loops or cut the
    optimization short). When fast_duv is True (default) a faster method is used, but this
    only sufficiently accurate when the estimated CCT is 1 K or less than the
    true value. 
    
    Reference:
        1. `Li, C., Cui, G., Melgosa, M., Ruan,X., Zhang, Y., Ma, L., Xiao, K., & Luo, M. R. (2016).
        Accurate method for computing correlated color temperature. 
        Optics Express, 24(13), 14066–14078. 
        <https://doi.org/10.1364/OE.24.014066>`_
    """
    # process NR input:
    if uvwbar is None:
        if (xyzbar is None) & (cieobs is not None):
            xyzbar, wl, dl = _get_xyzbar_wl_dl(cieobs)
        elif (xyzbar is None) & (cieobs is None):
            raise Exception('Must supply xyzbar or cieobs or uvwbar !!!')
        uvwbar = _convert_xyzbar_to_uvwbar(xyzbar)
       
    if uvwbar.shape[0] == 4:
        wl = uvwbar[0]
        dl = getwld(wl)
        uvwbar = uvwbar[1:]
    else:
        if wl is not None:
            dl = getwld(wl)
        else:
            raise Exception('Must supply wl for uvwbar !!!')

    i = 0
    T = T0
    while True & (i <= max_iter):
        
        T[T < _CCT_MIN] = _CCT_MIN # avoid infinities & convergence problems
        
        # Get (u,v), (u',v'), (u",v"):
        _, uBB, vBB, upBB, vpBB, uppBB, vppBB, _ = _get_uv_uvp_uvpp(T, uvwbar, wl, dl, out = 'BB,BBp,BBpp')

        # Calculate DT (ratio of f' and abs(f"):
        du, dv = (u - uBB), (v - vBB) # pre-calculate for speed
        DT = -(du*upBB + dv*vpBB) / np.abs((upBB**2)-du*uppBB + (vpBB**2)-dv*vppBB)

        # DT[DT>T] = _CCT_MIN # avoid convergence problems
        T = T - DT 

        if (np.abs(DT) < atol).all() | (np.abs(DT)/T < rtol).all():
            break
        i+=1
        
    # get Duv:
    if ~(fast_duv & (np.abs(DT)<=1.0).all()):
        _, uBB, vBB, _, _, _, _, _ = _get_uv_uvp_uvpp(T, uvwbar, wl, dl, out = 'BB') # update one last time
    
    Duv = _get_Duv_for_T_from_uvBB(u,v, uBB, vBB)
    return T, Duv


#==============================================================================
# Robertson 1968:
#==============================================================================

#------------------------------------------------------------------------------
# Load or calculate LUTs: 
#------------------------------------------------------------------------------
_CCT_LUT_PATH_FILE = os.path.join(_CCT_LUT_PATH, 'robertson1968.pkl')
if _CCT_LUT_CALC:
    _CCT_LUT['lut_type_def'] = (1000.0, 41000.0, 1, '%') # default LUT
    if not _CCT_USE_EXTENDED_LUT: 
        _CCT_LUT['lut_vars'] = ['T','uv','iso-T-slope']
    else: 
        _CCT_LUT['lut_vars'] = ['T','uv','iso-T-slope','sqrt(1+iso-T-slope**2)','sign(iso-T-slope_i*i+1)']

    # Calculate:
    _CCT_LUT['luts'] = {}
    for cieobs in _CCT_LIST_OF_CIEOBS_LUTS:
        _CCT_LUT['luts'][cieobs] = calculate_lut(_CCT_LUT['lut_type_def'], 
                                                 cieobs, 
                                                 lut_vars = _CCT_LUT['lut_vars'],
                                                 cct_min = _CCT_MIN, cct_max = _CCT_MAX)
    if not os.path.exists(_CCT_LUT_PATH): os.makedirs(_CCT_LUT_PATH, exist_ok = True)
    save_pkl(_CCT_LUT_PATH_FILE, _CCT_LUT)
else:
    # load:
    _CCT_LUT = load_pkl(_CCT_LUT_PATH_FILE)


#------------------------------------------------------------------------------
# Robertson1968 base (core) method
#------------------------------------------------------------------------------
def _uv_to_Tx_robertson1968(u, v, lut, lut_n_cols, ns = 4, out_of_lut = None,
                            fast_duv = _CCT_FAST_DUV, **kwargs):
    """ 
    Calculate Tx from u,v and lut using Robertson 1968.
    (lut_n_cols specifies the number of columns in the lut for 'robertson1968')
    """
    Duvx = None 
    idx_sources = np.arange(u.shape[0], dtype = np.int32) # source/conversion index
    
    # get uBB, vBB, mBB from lut:
    TBB, uBB, vBB, mBB  = lut[:,0::lut_n_cols], lut[:,1::lut_n_cols], lut[:,2::lut_n_cols], lut[:,3::lut_n_cols]
    if not _CCT_USE_EXTENDED_LUT: 
        # calculate distances to coordinates in lut (Eq. 4 in Robertson, 1968):
        di = ((v.T - vBB) - mBB * (u.T - uBB)) / ((1 + mBB**2)**(0.5))
    else:
        sqrt1pm2,sign_mimip1 = lut[:,4::lut_n_cols], lut[:,5::lut_n_cols]
        di = ((v.T - vBB) - mBB * (u.T - uBB)) / sqrt1pm2
        
    pn = (((v.T - vBB)**2 + (u.T - uBB)**2)).argmin(axis=0)
        
    # Get di_0, mBB_0 values to check sign of di_0 * mBB_0 -> if positive (right of apex): [j,j+1] -> [j-1,j]
    di_0 = _get_pns_from_x(di, pn, i = idx_sources, m0p = '0')
    mBB_0 = _get_pns_from_x(mBB, pn, i = idx_sources, m0p = '0')

    # Deal with endpoints of lut + create intermediate variables to save memory:
    pn, out_of_lut = _deal_with_lut_end_points(pn, TBB, out_of_lut)

    # Deal with positive slopes of iso-T lines
    c = (di_0*mBB_0 < 0)[:,0]
    pn[c] = pn[c] - 1
    
    # Get final values required for T calculation:
    mBB_0, mBB_p1 = _get_pns_from_x(mBB, pn, i = idx_sources, m0p = '0p')
    TBB_0, TBB_p1 = _get_pns_from_x(TBB, pn, i = idx_sources, m0p = '0p')
    di_0, di_p1 = _get_pns_from_x(di, pn, i = idx_sources, m0p = '0p')
    if _CCT_USE_EXTENDED_LUT: sign_mimip1_0 = _get_pns_from_x(sign_mimip1, pn, i = idx_sources, m0p = '0')

    
    # Estimate Tc (Robertson, 1968): 
    if not _CCT_USE_EXTENDED_LUT: 
        sign = np.sign(mBB_0*mBB_p1) # Solve issue of zero-crossing of slope of planckian locus:
    else: 
        sign = sign_mimip1_0
    slope = (di_0/((di_0 - sign*di_p1) + _CCT_AVOID_ZERO_DIV))
    Tx = (((1/TBB_0) + slope * ((1/TBB_p1) - (1/TBB_0)))**(-1))#".copy()

    # Estimate Duv from approximation of u,v coordinates of Tx:
    if fast_duv:
        uBB_0, uBB_p1 = _get_pns_from_x(uBB, pn, i = idx_sources, m0p = '0p')
        vBB_0, vBB_p1 = _get_pns_from_x(vBB, pn, i = idx_sources, m0p = '0p')
        ux, vx = (uBB_0 + slope * (uBB_p1 - uBB_0)), (vBB_0 + slope * (vBB_p1 - vBB_0))
        Duvx = _get_Duv_for_T_from_uvBB(u, v, ux, vx)
        
    return Tx, Duvx, out_of_lut


#------------------------------------------------------------------------------
# xyz_to_cct wrapper function:
#------------------------------------------------------------------------------
def xyz_to_cct(xyzw, is_uv_input = False, cieobs = _CCT_CIEOBS, out = 'cct',
               lut = None, apply_newton_raphson = False,
               rtol = 1e-10, atol = 0.1,  max_iter = _CCT_MAX_ITER, 
               split_calculation_at_N = _CCT_SPLIT_CALC_AT_N, use_fast_duv = _CCT_FAST_DUV):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv(distance above (> 0) or below ( < 0) the Planckian locus) using  
    Robertson's 1968 search method.
        
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        :is_uv_input:
            | False, optional
            | If True: xyzw contain uv input data, not xyz data!
        :cieobs: 
            | _CCT_CIEOBS, optional
            | CMF set used to calculated xyzw.
        :out: 
            | 'cct' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)
        :rtol: 
            | 1e-10, float, optional
            | Stop search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop search when cct a absolute tolerance (K) is reached.
        :lut:
            | None, optional
            | Look-Up-Table with Ti, u,v,u',v',u",v",slope values of Planckians. 
            | Options:
            |  - None: defaults to the lut specified in _CCT_LUT['lut_type_def'].
            |  - tuple: new lut will be generated from scratch using the info in the tuple.
            |  - ndarray [Nx1]: list of luts for which to generate a lut
            |  - ndarray [Nxn] with n>3: pre-calculated lut (last col must contain slope of the isotemperature lines).
        :apply_newton_raphson:
            | False, optional
            | If False: use only the Robertson1968 base method. 
            |           Accuracy depends on CCT of test source and the location
            |           and spacing of the CCTs in the list.
            | If True:  improve estimate of base method using a follow-up 
            |           newton-raphson method.
            |           When the CCT for multiple source is calculated in one go,
            |           then the atol and rtol values have to be met for all! 
        :max_iter:
            | _CCT_MAX_ITER, optional
            | Maximum number of iterations used by the cascading-lut or newton-raphson methods.
        :split_calculation_at_N:
            | _CCT_SPLIT_CALC_AT_N, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :use_fast_duv:
            | _CCT_FAST_DUV, optional
            | If True: use a fast estimator of the Duv 
            |   (one that avoids calculation of Planckians and uses the former
            |    best estimate's u,v coordinates. This method is accurate enough
            |    when the atol is small enough -> as long as abs(T-T_former)<=1K
            |    the Duv estimate should be ok.)

    Returns:
        :returns: 
            | ndarray with:
            |    cct: out == 'cct' (or 1)
            |    duv: out == 'duv' (or -1)
            |    cct, duv: out == 'cct,duv' (or 2)
            |    [cct,duv]: out == "[cct,duv]" (or -2) 
            
    Note: 
        1. Out-of-lut CCTs are encoded as negative CCTs (with as absolute value
        the value of the closest CCT from the lut.)
    
    References:
        1.  `Robertson, A. R. (1968). 
        Computation of Correlated Color Temperature and Distribution Temperature. 
        Journal of the Optical Society of America,  58(11), 1528–1535. 
        <https://doi.org/10.1364/JOSA.58.001528>`_
        
        2. Baxter D., Royer M., Smet K.A.G. (2022)
        Modifications of the Robertson Method for Calculating Correlated Color Temperature to Improve Accuracy and Speed
        (in preparation, LEUKOS?)
         
        3. `Li, C., Cui, G., Melgosa, M., Ruan, X., Zhang, Y., Ma, L., Xiao, K., & Luo, M. R. (2016).
        Accurate method for computing correlated color temperature. 
        Optics Express, 24(13), 14066–14078. 
        <https://doi.org/10.1364/OE.24.014066>`_

    """      
    # Get chromaticity coordinates u,v from xyzw:
    uvw = xyz_to_Yuv60(xyzw)[:,1:3]  if is_uv_input == False else xyzw[:,0:2] # xyz contained uv !!! (needed to efficiently determine f_corr)
    
    # pre-calculate wl, dl, uvwbar for later use (will also determine wl if None !):
    xyzbar, wl, dl = _get_xyzbar_wl_dl(cieobs)
    uvwbar = _convert_xyzbar_to_uvwbar(xyzbar) if apply_newton_raphson else None # might not be needed (only required in Duv calculation)
    
    # Get or generate requested lut:
    if lut is None:
        lut = _CCT_LUT['luts'][cieobs].copy()
    elif isinstance(lut, (tuple,np.ndarray)):
        if isinstance(lut,tuple): # only list of ccts! or tuple -> generate new lut!
            new_lut = True
        elif (lut.shape[-1] == 1):
            new_lut = True 
        else:
            new_lut = False
        if new_lut:
            lut = calculate_lut(lut, cieobs, lut_vars = _CCT_LUT['lut_vars'],
                                cct_min = _CCT_MIN, cct_max = _CCT_MAX)
            
    lut_n_cols = lut.shape[-1] # store now, as this will change later
    
    # prepare split of input data to speed up calculation:
    n = xyzw.shape[0] # number of conversions
    ccts, duvs = np.zeros((n,1)), np.zeros((n,1))
    n_ii = split_calculation_at_N if split_calculation_at_N is not None else n
    N_ii = n//n_ii + 1*((n%n_ii)>0)
    
    # loop of splitted data:
    for ii in range(N_ii):
        out_of_lut = None

        # get data for split ii:
        uv = uvw[n_ii*ii:n_ii*ii+n_ii] if (ii < (N_ii-1)) else uvw[n_ii*ii:]
        u, v = uv[:,0,None], uv[:,1,None]
        
    
        # get Tx estimate, out_of_lut boolean array using Robertson1968:
        Tx, Duvx, out_of_lut = _uv_to_Tx_robertson1968(u, v, lut, lut_n_cols, 
                                                       ns = lut_n_cols, 
                                                       out_of_lut = out_of_lut)  

        
        if apply_newton_raphson:
            Tx, Duvx = _get_newton_raphson_estimated_Tc(u, v, Tx, wl = wl, uvwbar = uvwbar,
                                                        atol = atol, rtol = rtol, 
                                                        max_iter = max_iter,
                                                        fast_duv = use_fast_duv)           

        # Calculate Duvx if not already done:
        if Duvx is None:
            if uvwbar is None: uvwbar = _convert_xyzbar_to_uvwbar(xyzbar) # only done once!
            Duvx = _get_Duv_for_T(u,v, Tx, wl, cieobs, uvwbar, dl)

        # signal out-of-lut CCTs with a negative sign:
        if out_of_lut is not None: Tx = Tx*(-1)**out_of_lut

        # Add freshly calculated Tx, Duvx to storage array:
        if (ii < (N_ii-1)): 
            ccts[n_ii*ii:n_ii*ii+n_ii] = Tx
            duvs[n_ii*ii:n_ii*ii+n_ii] = Duvx
        else: 
            ccts[n_ii*ii:] = Tx
            duvs[n_ii*ii:] = Duvx

    # Regulate output:
    if (out == 'cct') | (out == 1):
        return ccts
    elif (out == 'duv') | (out == -1):
        return duvs
    elif (out == 'cct,duv') | (out == 2):
        return ccts, duvs
    elif (out == "[cct,duv]") | (out == -2):
        return np.hstack((ccts,duvs))   
    else:
        raise Exception('Unknown output requested')

#---------------------------------------------------------------------------------------------------
def xyz_to_duv(xyzw, out = 'duv', **kwargs):
    """
    Wraps xyz_to_cct, but with duv output. For kwargs info, see xyz_to_cct.
    """
    return xyz_to_cct(xyzw, out = out, **kwargs)
        
#---------------------------------------------------------------------------------------------------
def cct_to_xyz(ccts, duv = None, cct_offset = None, cieobs = _CCT_CIEOBS):
    """
    Convert correlated color temperature (550 K <= CCT <= 1e11 K) and 
    Duv (distance above (>0) or below (<0) the Planckian locus) to 
    XYZ tristimulus values.
    
    | Finds xyzw_estimated by determining the iso-temperature line 
    |   (= line perpendicular to the Planckian locus): 
    |   Option 1 (fastest):
    |       First, the angle between the coordinates corresponding to ccts 
    |       and ccts-cct_offset are calculated, then 90° is added, and finally
    |       the new coordinates are determined, while taking sign of duv into account.
    |   Option 2 (slowest, about 55% slower):
    |       Calculate the slope of the iso-T-line directly using the Planckian
    |       spectrum and its derivative.
     
    Args:
        :ccts: 
            | ndarray [N,1] of cct values
        :duv: 
            | None or ndarray [N,1] of duv values, optional
            | Note that duv can be supplied together with cct values in :ccts: 
            | as ndarray with shape [N,2].
        :cct_offset:
            | None, optional
            | If None: use option 2 (direct iso-T slope calculation, more accurate,
            |                        but slower: about 1.55 slower)
            | else: use option 1 (estimate slope from 90° + angle of small cct_offset)
        :cieobs: 
            | _CCT_CIEOBS, optional
            | CMF set used to calculated xyzw.
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
            | If None: use same wavelengths as CMFs in :cieobs:.
        
    Returns:
        :returns: 
            | ndarray with estimated XYZ tristimulus values
    
    Note:
        1. If duv is not supplied (:ccts:.shape is (N,1) and :duv: is None), 
        source is assumed to be on the Planckian locus.
        2. Minimum CCT is 550 K (lower than 550 K, some negative Duv values
        will result in coordinates outside of the Spectrum Locus !!!)
    """
    # make ccts a min. 2d np.array:
    if isinstance(ccts,list):
        ccts = np.atleast_2d(np.array(ccts)).T
    else:
        ccts = np.atleast_2d(ccts) 
    
    if len(ccts.shape)>2:
        raise Exception('cct_to_xyz(): Input ccts.shape must be <= 2 !')
    
    # get cct and duv arrays from :ccts:
    cct = np.atleast_2d(ccts[:,0,None])

    if (duv is None) & (ccts.shape[1] == 2):
        duv = np.atleast_2d(ccts[:,1,None])
    if (duv is None) & (ccts.shape[1] == 1):
        duv = np.zeros_like(ccts)
    elif duv is not None:
        duv = np.atleast_2d(duv)

    xyzbar,wl, dl = _get_xyzbar_wl_dl(cieobs)
    if cct_offset is not None:
        # estimate iso-T-line from estimated slope using small cct offset:
        #-----------------------------------------------------------------
        _,xyzBB,_,_ = _get_tristim_of_BB_BBp_BBpp(np.vstack((cct, cct-cct_offset,cct+cct_offset)),xyzbar,wl,dl,out='BB') 
        YuvBB = xyz_to_Yuv60(xyzBB)

        N = (xyzBB.shape[0])//3
        YuvBB_centered = (YuvBB[N:] - np.vstack((YuvBB[:N],YuvBB[:N])))
        theta = np.arctan2(YuvBB_centered[...,2:3],YuvBB_centered[...,1:2])
        theta = (theta[:N] + (theta[N:] - np.pi*np.sign(theta[N:])))/2 # take average for increased accuracy
        theta = theta + np.pi/2*np.sign(duv) # add 90° to obtain the direction perpendicular to the blackbody locus
        u, v = YuvBB[:N,1:2] + np.abs(duv)*np.cos(theta), YuvBB[:N,2:3] + np.abs(duv)*np.sin(theta)

    else:
        # estimate iso-T-line from calculated slope:
        #-------------------------------------------
        uvwbar = _convert_xyzbar_to_uvwbar(xyzbar)
        _,UVW,UVWp,_ = _get_tristim_of_BB_BBp_BBpp(cct,uvwbar,wl,dl,out='BB,BBp') 
        
        R = UVW.sum(axis=-1, keepdims = True) 
        Rp = UVWp.sum(axis=-1, keepdims = True) 
        num = (UVWp[:,1:2]*R - UVW[:,1:2]*Rp) 
        denom = (UVWp[:,:1]*R - UVW[:,:1]*Rp)
        num[(num == 0)] += _CCT_AVOID_ZERO_DIV
        denom[(denom == 0)] += _CCT_AVOID_ZERO_DIV
        li = num/denom  
        li = li + np.sign(li)*_CCT_AVOID_ZERO_DIV # avoid division by zero
        mi = -1.0/li # slope of isotemperature lines

        YuvBB = xyz_to_Yxy(UVW)
        u, v = YuvBB[:,1:2] + np.sign(mi) * duv*(1/((1+mi**2)**0.5)), YuvBB[:,2:3] + np.sign(mi)* duv*((mi)/(1+mi**2)**0.5)    
        
    Yuv = np.hstack((100*np.ones_like(u),u,v))
    return Yuv60_to_xyz(Yuv)

#==============================================================================

if __name__ == '__main__':
    import luxpy as lx 
    
    cieobs = '1931_2'
    
    # cieobs = '2015_10'

    # ------------------------------
    # Setup some tests:
    
    # test 1:
    cct = 5000   
    duvs = np.array([[-0.05,-0.025,0,0.025,0.05]]).T
    # duvs = np.array([[-0.03]]).T
    ccts = np.array([[cct]*duvs.shape[0]]).T
    
    # test 2:
    # duv = -0.04
    # duvs = np.array([[0,*[duv]*(ccts.shape[0]-1)]]).T
    # ccts = np.array([[1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000, 3500, 4500.0, 5500, 6500, 15500,25500,35500,45500,50500]]).T

    # test 3:
    # ccts = np.array([[1625.92608972303,1626.26, 3500, 4500.0, 5500, 6500, 15500,25500,35500,45500,50500]]).T
    # duvs = np.array([[duv]*ccts.shape[0]]).T
    # duvs[0] = 0.0037117089512229
    
    cctsduvs_t = np.hstack((ccts,duvs))

    # # # Test 4 (from disk):
    # # Test 4 (from disk): 'ref_cct_duv_1500-40000K.csv' or 'test_rob_error.csv'
    # cctsduvs_t = pd.read_csv('../ref_cct_duv_1500-40000K.csv',header='infer',index_col=None).values
    # cctsduvs_t = cctsduvs_t[cctsduvs_t[:,0] <= 40000,:2]
    # # cctsduvs_t = cctsduvs_t[(cctsduvs_t[:,0] >= 2000) & (cctsduvs_t[:,0] <= 20000),:2]
    # # cctsduvs_t = cctsduvs_t[(cctsduvs_t[:,1] >= -0.03) & (cctsduvs_t[:,1] <= 0.03),:2]
    # ccts, duvs = cctsduvs_t[:,:1], cctsduvs_t[:,1:]
    
    
    #--------------------------------
    # Backward transform from CCT,Duv to xyz to generate test xyz for forward tf:
    cct_offset = None
    print('cct_to_xyz:')
    xyz = cct_to_xyz(ccts = ccts, duv = duvs, cieobs = cieobs, cct_offset = cct_offset)
    
    
    #--------------------------------
    # Forward transform from xyz to CCT,Duv using Robertson 1968:
    modes = ['robertson1968']
    lut = _CCT_LUT['luts']['1931_2'].copy()
    #lut_ = lx._CCT_LUT['robertson1968']['luts']['Yuv60']['1931_2'][((1000.0,51000.0,0.5,'%'),)][0]
    lut = (1000.0,41000.0,1,'%')
    # lut = None
    for mode in modes:
        print('mode:',mode)
        print('xyz_to_cct')
        cctsduvs = xyz_to_cct(xyz, atol = 0.1, rtol = 1e-10,cieobs = cieobs, out = '[cct,duv]', 
                              apply_newton_raphson = False, lut = lut,split_calculation_at_N=None)
    
    # Out of LUT conversions are coded with a negative CCT, so make positive again before calculating error:
    cctsduvs_ = cctsduvs.copy();cctsduvs_[:,0] = np.abs(cctsduvs_[:,0]) # outof gamut ccts are encoded as negative!!
    
    #--------------------------------
    # Close loop: Backward transform from CCT,Duv (obtained from forward tf) to xyz
    print('cct_to_xyz2')
    xyz_ = cct_to_xyz(cctsduvs_, cieobs = cieobs, cct_offset = cct_offset)
    
    #--------------------------------
    # Calculate CCT,Duv and XYZ errors:
    print('cctsduvs_t:\n',cctsduvs_t)
    print('cctsduvs:\n', cctsduvs)
    print('Dcctsduvs:\n', cctsduvs_ - cctsduvs_t)
    print('Dxyz:\n', xyz - xyz_)
    
    #---------------------------------
    # Make a plot of the errors:    
    fig,ax = plt.subplots(1,2,figsize=(14,8))
    d = np.abs(cctsduvs_ - cctsduvs_t) # error array
    for i in range(2):
        ax[i].plot(ccts[:,0], d[:,i],'o')
        #ax[i].plot(lut[:,0], np.zeros_like(lut),'r.')
        #ax[i].plot(lut[:,0], lut[:,-1],'g.-')
        ax[i].set_ylim([-d[:,i].max()*1.1,d[:,i].max()*1.1])
    
    # plt.close('all')
    xyz_to_cct_r = lambda xyz: xyz_to_cct(xyz, atol = 0.1, rtol = 1e-10,cieobs = cieobs, out = '[cct,duv]', 
                          apply_newton_raphson = False, lut = lut,split_calculation_at_N=None)
    