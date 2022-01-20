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
cct: Module with functions related to correlated color temperature calculations
===============================================================================
 These methods supersede earlier methods in cct_legacy.y (prior to Nov 2021)

 :_CCT_MAX: (= 1e11 K), max. value that does not cause overflow problems. 
 
 :_CCT_MIN: (= 550 K), min. value that does not cause underflow problems.
 
 :_CCT_FALLBACK_N: Number of intervals to divide an ndarray with CCTs.
 
 :_CCT_FALLBACK_UNIT: Type of scale (units) an ndarray will be subdivided.

 :_CCT_LUT_PATH: Folder with Look-Up-Tables (LUT) for correlated color 
                 temperature calculations. 

 :_CCT_LUT: Dict with pre-calculated LUTs with structure LUT[mode][cspace][cieobs][lut i].
 
 :_CCT_LUT_CALC: Boolean determining whether to force LUT calculation, even if
                 the LUT.pkl files can be found in ./data/cctluts/.
 
 :_CCT_LUT_RESOLUTION_REDUCTION_FACTOR: number of subdivisions when performing
                                        a cascading lut calculation to zoom-in 
                                        progressively on the CCT (until a certain 
                                        tolerance is met)
                 
 :_CCT_CSPACE: default chromaticity space to calculate CCT and Duv in.
 
 :_CCT_CSPACE_KWARGS: nested dict with cspace parameters for forward and backward modes. 
 
 :get_tcs4(): Get an ndarray of Tc's obtained from a list or tuple of tc4 4-vectors.
 
 :calculate_lut(): Function that calculates the LUT for the input ccts.
 
 :generate_luts(): Generate a number of luts and store them in a nested dictionary.
                    (Structure: lut[cspace][cieobs][lut type])

 :xyz_to_cct(): Calculates CCT, Duv from XYZ (wraps a variety of methods)

 :xyz_to_duv(): Calculates Duv, (CCT) from XYZ (wrapper around xyz_to_cct, but with Duv output.)
                
 :cct_to_xyz(): Calculates xyz from CCT, Duv by estimating the line perpendicular to the planckian locus (=iso-T line).

 :cct_to_xyz(): Calculates xyz from CCT, Duv [_CCT_MIN < CCT < _CCT_MAX]

 :xyz_to_cct_mcamy1992(): | Calculates CCT from XYZ using Mcamy model:
                          | `McCamy, Calvin S. (April 1992). 
                            Correlated color temperature as an explicit function of 
                            chromaticity coordinates. 
                            Color Research & Application. 17 (2): 142–144. 
                            <http://onlinelibrary.wiley.com/doi/10.1002/col.5080170211/abstract>`_

 :xyz_to_cct_hernandez1999(): | Calculate CCT from XYZ using Hernández-Andrés et al. model.
                              | `Hernández-Andrés, Javier; Lee, RL; Romero, J (September 20, 1999). 
                                Calculating Correlated Color Temperatures Across the 
                                Entire Gamut of Daylight and Skylight Chromaticities. 
                                Applied Optics. 38 (27): 5703–5709. PMID 18324081. 
                                <https://www.osapublishing.org/ao/abstract.cfm?uri=ao-38-27-5703>`_

 :xyz_to_cct_ohno2014(): | Calculates CCT, Duv from XYZ using a Ohno's 2014 LUT method.
                         | `Ohno Y. (2014)
                           Practical use and calculation of CCT and Duv. 
                           Leukos. 2014 Jan 2;10(1):47-55.
                           <http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020>`_
                       
 :xyz_to_cct_zhang2019():  | Calculates CCT, Duv from XYZ using Zhang's 2019 golden-ratio search algorithm
                           | `Zhang, F. (2019). 
                              High-accuracy method for calculating correlated color temperature with 
                              a lookup table based on golden section search. 
                              Optik, 193, 163018. 
                              <https://doi.org/https://doi.org/10.1016/j.ijleo.2019.163018>`_
                 
 :xyz_to_cct_robertson1968(): | Calculates CCT, Duv from XYZ using a Robertson's 1968 search method.
                              | `Robertson, A. R. (1968). 
                                Computation of Correlated Color Temperature and Distribution Temperature. 
                                Journal of the Optical Society of America,  58(11), 1528–1535. 
                                <https://doi.org/10.1364/JOSA.58.001528>`_
  
 :xyz_to_cct_li2016(): | Calculates CCT, Duv from XYZ using a Li's 2019 Newton-Raphson method.
                       | `Li, C., Cui, G., Melgosa, M., Ruan, X., Zhang, Y., Ma, L., Xiao, K., & Luo, M. R. (2016).
                         Accurate method for computing correlated color temperature. 
                         Optics Express, 24(13), 14066–14078. 
                         <https://doi.org/10.1364/OE.24.014066>`_                        
                                
 :xyz_to_cct_fibonacci(): | Calculates CCT, Duv from XYZ using a Fibonacci search method.
                  
 :cct_to_mired(): Converts from CCT to Mired scale (or back).

===============================================================================
"""

import os
import copy 
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# load some methods already programmed in luxpy:
from luxpy import (math, _BB, _WL3, _CIEOBS, _CMF, 
                   cie_interp, spd_to_xyz, 
                   getwlr, getwld,
                   xyz_to_Yxy, Yxy_to_xyz, xyz_to_Yuv60, Yuv60_to_xyz, 
                   xyz_to_Yuv, Yuv_to_xyz, cri_ref, 
                   )
from luxpy.utils import _PKG_PATH, _SEP, np2d, np2dT, getdata, save_pkl, load_pkl
from luxpy.color.ctf.colortf import colortf

__all__ = ['_CCT_MAX','_CCT_MIN','_CCT_CSPACE','_CCT_CSPACE_KWARGS',
           '_CCT_LUT_PATH','_CCT_LUT', '_CCT_LUT_RESOLUTION_REDUCTION_FACTOR',
           '_CCT_FALLBACK_N', '_CCT_FALLBACK_UNIT',
           'cct_to_mired','xyz_to_cct_mcamy1992', 'xyz_to_cct_hernandez1999',
           'xyz_to_cct_robertson1968','xyz_to_cct_ohno2014',
           'xyz_to_cct_li2016','xyz_to_cct_zhang2019', 'xyz_to_cct_fibonacci',
           'xyz_to_cct','cct_to_xyz', 'calculate_lut', 'generate_luts', 'get_tcs4',
           '_get_lut', '_generate_tcs', '_generate_lut','_generate_lut_ohno2014']


#==============================================================================
# define global variables:
#==============================================================================
_CCT_AVOID_ZERO_DIV = 1e-100
_CCT_AVOID_INF = 1/_CCT_AVOID_ZERO_DIV

_CCT_MAX = 1e11 # don't set to higher value to avoid overflow and errors
_CCT_MIN = 550

_CCT_CSPACE = 'Yuv60'
_CCT_CSPACE_KWARGS = {'fwtf':{}, 'bwtf':{}}

_CCT_FALLBACK_N = 50
_CCT_FALLBACK_UNIT = 'K-1'
_CCT_MAX_ITER = 100
_CCT_SPLIT_CALC_AT_N = 25 # some tests show that this seems to be the fastest (for 1000 conversions)

_CCT_LUT_PATH = _PKG_PATH + _SEP + 'data'+ _SEP + 'cctluts' + _SEP #folder with cct lut data
_CCT_LUT_PATH_LX_REPO = 'https://raw.github.com/ksmet1977/luxpy/master/luxpy/data/cctluts/' # luxpy repo url where cctluts are stored 
_CCT_LUT_CALC = False
_CCT_LUT = {}
_CCT_UV_TO_TX_FCNS = {}
_CCT_LUT_RESOLUTION_REDUCTION_FACTOR = 4 # for when cascading luts are used (d(Tm1,Tp1)-->divide in _CCT_LUT_RESOLUTION_REDUCTION_FACTOR segments)

_CCT_FAST_DUV = True # use a fast, but slightly less accurate Duv calculation with Newton-Raphson
_CCT_VERBOSITY_LUT_GENERATION = 1

# flow control parameters:
#-------------------------
_CCT_LIST_OF_MODE_LUTS = ['robertson1968','ohno2014','zhang2019','fibonacci'] # only for the ones in this list are LUTS pre-generated (->_CCT_LUT)
_CCT_LIST_OF_CIEOBS_LUTS = ['1931_2', '1964_10', '2015_2', '2015_10'] # only for the ones in this list are LUTS pre-generated (->_CCT_LUT)


_CCT_LUT_MIN, _CCT_LUT_MAX = 1000.0, 51000
# _CCT_SHARED_LUT_TYPES = [((10,100,10,'K-1'),(100,625,25,'K-1'),True),
#                         ((1,1,1,'K-1'),(25,1025,25,'K-1'),False),
#                         ((_CCT_LUT_MIN, _CCT_LUT_MAX, 15, '%'),),
#                         ((_CCT_LUT_MIN, _CCT_LUT_MAX, 1, '%'),),
#                         ((_CCT_LUT_MIN, _CCT_LUT_MAX, 0.75, '%'),),
#                         ((_CCT_LUT_MIN, _CCT_LUT_MAX, 0.50, '%'),),
#                         ((_CCT_LUT_MIN, _CCT_LUT_MAX, 0.25, '%'),),
#                         ((_CCT_LUT_MIN, _CCT_LUT_MAX, 100.0, 'K'),),
#                         ((_CCT_LUT_MIN, _CCT_LUT_MAX, 50.0, 'K'),),
#                         ((_CCT_LUT_MIN, _CCT_LUT_MAX, 25.0, 'K'),),
#                         ((_CCT_LUT_MIN, _CCT_LUT_MAX, 10.0, 'K'),),
#                         ((_CCT_LUT_MIN, _CCT_LUT_MAX, 1, 'K'),),
#                         ((_CCT_LUT_MIN, _CCT_LUT_MAX, 0.1, 'K'),)] # possible LUT types, will be added to the default lut for each mode upon LUT generation 

_CCT_SHARED_LUT_TYPES = [((_CCT_LUT_MIN, _CCT_LUT_MAX, 10.0, 'K'),),((_CCT_LUT_MIN, _CCT_LUT_MAX, 0.50, '%'),)] # shared among all modes

_CCT_LUT_ONE_NPY_PER_MODE = True # switch between generating one npy file per mode (3-.. columns) vs one for all modes (8 columns)

#==============================================================================
# define general helper functions:
#==============================================================================


#------------------------------------------------------------------------------
# cspace related functions:
#------------------------------------------------------------------------------

def _process_cspace(cspace, cspace_kwargs = None, cust_str = 'cspace'):
    """ 
    Process cspace and cspace_kwargs input. 
    
    Returns dict with keys:
        - 'str': cspace string
        - 'fwtf': lambda function for xyz_to_cspace forward transform (cspace_kwargs are already processed).
        - 'bwtf': lambda function for cspace_to_xyz backward transform (cspace_kwargs are already processed).
    """
    if cspace_kwargs is None: cspace_kwargs = {'fwtf':{},'bwtf':{}}
    if 'fwtf' not in cspace_kwargs: cspace_kwargs['fwtf'] = {}
    if 'bwtf' not in cspace_kwargs: cspace_kwargs['bwtf'] = {}
    if isinstance(cspace,str): 
        cspace_str = cspace
        if cspace == 'Yuv60':
            cspace_fw = xyz_to_Yuv60 # don't use lambda function for speed
            cspace_bw = Yuv60_to_xyz 
        elif (cspace == 'Yuv') | (cspace == 'Yuv76'):
            cspace_fw = xyz_to_Yuv # don't use lambda function for speed
            cspace_bw = Yuv_to_xyz 
        else:
            cspace_fw = (lambda xyz,**kwargs: colortf(xyz,tf = 'xyz>' + cspace_str, fwtf = cspace_kwargs['fwtf']))
            cspace_bw = (lambda Yuv,**kwargs: colortf(Yuv,tf = cspace_str + '>xyz', bwtf = cspace_kwargs['bwtf']))
    elif isinstance(cspace,tuple):
        if len(cspace) == 3: 
            cspace_str = cspace[2]
            cspace_fw = cspace[0]
            cspace_bw = cspace[1]
        elif len(cspace) == 2:
            if isinstance(cspace[1],str):
                cspace_str = cspace[1]
                cspace_fw = cspace[0]
                cspace_bw = None
            else:
                cspace_str = cust_str
                cspace_fw = cspace[0]
                cspace_bw = cspace[1]
        else:
            cspace_str = cust_str
            cspace_fw = cspace[0]
            cspace_bw = None
    elif isinstance(cspace,dict):
        cspace_str = cspace['str'] if ('str' in cspace) else cust_str
        if 'fwtf' not in cspace: 
            raise Exception("'fwtf' key with forward xyz_to_cspace function must be supplied !!!")
        else:
            cspace_fw = cspace['fwtf'] 
        cspace_bw = cspace['bwtf']  if ('bwtf' in cspace) else None
    else:
        cspace_str = cust_str
        cspace_fw = cspace
        cspace_bw = None
    
    # create cspace dict:
    cspace_dict = {'str': cspace_str} 
    cspace_dict['fwtf'] = cspace_fw if (len(cspace_kwargs['fwtf']) == 0) else (lambda xyz,**kwargs: cspace_fw(xyz,**cspace_kwargs['fwtf']))
    cspace_dict['bwtf'] = cspace_bw if (len(cspace_kwargs['bwtf']) == 0) else (lambda xyz,**kwargs: cspace_fw(xyz,**cspace_kwargs['fwtf']))

    return cspace_dict, cspace_str

def _get_xyzbar_wl_dl(cieobs, wl = None):
    """
    Get the interpolated (to wl) xyzbar CMF set corresponding to cieobs.
    Returns an ndarray with the cmfs (stripped of wavelengths), ndarray with
    wavelengths and an ndarray with the wavelength differences.
    """
    # get requested cmf set:
    if isinstance(cieobs,str):
        cmf = _CMF[cieobs]['bar'].copy()
    else:
        cmf = cieobs.copy()
    wl = cmf[0] if wl is None else getwlr(wl)
    dl = getwld(wl)*1.0
    cmf =  cie_interp(cmf, wl, kind = 'cmf', negative_values_allowed = False)[1:]
    c = ~(((cmf[1:]==0).sum(0)==3))
    cmf[:,c] += _CCT_AVOID_ZERO_DIV # avoid nan's in uvwvbar
    return cmf, wl, dl

def _convert_xyzbar_to_uvwbar(xyzbar, cspace_dict):
    """
    Convert the xyzbar (no wl on row 0!) CMF set to a CMF set representing a different 
    color space/ chromaticity diagram (integration leads to new tristimulus values)
    Returns an ndarray (no wl on row 0!) of new CMFs.
    """
    # convert to cspace based cmfs (Eq.6-7):
    Yuvbar = cspace_dict['fwtf'](xyzbar.T.copy()) # convert to chromaticity format from xyz (cfr. cmf) format
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
    T = np2d(T)*1.0 # force float
    wlt = wl*1.0e-9
    c_wl_T = _BB['c2']/(wlt*T)
    exp = np.exp(c_wl_T)
    exp[np.isinf(exp)] = _CCT_AVOID_INF
    
    # avoid div by inf or zero:
    exp_min_1, exp_plus_1 = exp - 1.0, exp + 1.0
    exp_min_1[exp_min_1==0] = _CCT_AVOID_ZERO_DIV
    exp_min_1[np.isinf(exp_min_1)] = _CCT_AVOID_INF
    exp_min_1_squared = exp_min_1**2
    exp_min_1_squared[np.isinf(exp_min_1_squared)] = _CCT_AVOID_INF # avoid warning "invalid value encountered in true_divide"
    exp_min_1_squared[exp_min_1_squared == 0.0] = _CCT_AVOID_ZERO_DIV
    exp_frac = exp/exp_min_1_squared
    
    BB = _BB['c1']*(wlt**(-5))*(1/(exp_min_1))
    BB[np.isinf(BB)] = _CCT_AVOID_INF
    if 'BBp' in out: 
        BBp = (_BB['c1']*_BB['c2']*(T**(-2))*(wlt**(-6)))*exp_frac
    if 'BBpp' in out:
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


#------------------------------------------------------------------------------
# Tc list generation functions:
#------------------------------------------------------------------------------

def _get_tcs4(tc4, uin = None, out = 'Ts',
              fallback_unit = _CCT_FALLBACK_UNIT, 
              fallback_n = _CCT_FALLBACK_N):

    (T0,Tn,dT),u = (np.atleast_1d(tc4_i) for tc4_i in tc4[:-1]), tc4[-1] # min, max, interval, unit
    
    # Get n from third element:
    if ((dT<0).any() & (dT>=0).any()):
        raise Exception('3e element [dT,n] in 4-vector tc4 contains negatives AND positives! Should be only 1 type.')
    else:
        n = np.abs(dT) if (dT<0).all() else None # dT contains number of tcs between T0 and Tn, not dT
 
    # special 'au' case
    if 'au' in u:
        u = fallback_unit 
        if n is None: n = fallback_n # not-None n force the use of n in what follows !!
    
    # Tmin, Tmax input different from unit u:
    if uin is not None:
        if uin != u:
            if (('-1' in uin) & ('-1' not in u)) | (('-1' not in uin) & ('-1' in u)):
                T0, Tn = 1e6/Tn[::-1], 1e6/T0[::-1] # generate scale in mireds (input was always in Tc)
    
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
        Ts = 1e6/Ts[::-1] # scale was in mireds 
    
    if out == 'Ts': 
        return Ts
    elif out == 'Ts,T0,Tn,dT,u,n':
        return Ts,T0,Tn,dT,u,n

def get_tcs4(tc4, uin = None, seamless_stitch = True,
             fallback_unit = _CCT_FALLBACK_UNIT, 
             fallback_n = _CCT_FALLBACK_N):
    """ 
    Get an ndarray of Tc's obtained from a list or tuple of tc4 4-vectors.
    
    Args:
        :tc4:
            | list or tuple of 4-vectors.
            |  e.g. (tc4_1, tc4_2, tc4_3,...) or (tc4_1, tc4_2, tc4_3,..., bool::seamless_stitch)
            | When the last element of the list/tuple is a bool, then this specifies
            |  how the Tc arrays generated for each of the 4-vector elements need to be
            |  stitched together. This overrides the seamless_stitch input argument.
            | Vector elements are: 
            |    [Tmin, Tmax inclusive, Tinterval(or number of intervals), unit]
            |  Unit specifies unit of the Tc interval, i.e. it determines the
            |       type of scale in which the spacing of the Tc are done.
            |  Unit options are:
            |   - '%': equal relative Tc spacing (in %, cfr. (Ti+1 - Ti-1)/Ti-1).
            |   - 'K' equal absolute Tc spacing (in K, cfr. (Ti+1 - Ti-1).
            |   - '%-1': equal relative reciprocal Tc (MK-1 = mired).
            |   - 'K-1': equal absolute reciprocal Tc (MK-1 = mired).
            |  If the 'interval' element is negative, it actually represents
            |  the number of intervals between Tmin, Tmax (included).
        :uin:
            | None, optional
            | Unit of input Tmin, Tmax (by default it is assumed to be the same
            | as the scale 'unit').
        :seamless_stitch:
            | True, optional
            | Determines how the Tc arrays generated for each of the 4-vector 
            | elements are stitched together. Is overriden by the presence of a 
            | bool as last list/tuple element in :tc4:.
            | For a seamless stitch, all units for all 4-vectors should be the same!!
        :fallback_unit:
            | _CCT_FALLBACK_UNIT, optional
            | Unit to fall back on when the input unit in tc4 (of first list) is 'au'.
            | As there is no common distancing of the unit types ['K','%','%-1','K-1']
            | the Tc's are generated by dividing the min-max range into 
            | a number of divisions, specified by the negative 3 element (or when
            | positive or NaN, the number of divisions is set by :fallback_divisions:)
        :fallback_n:
            | _CCT_FALLBACK_N, optional
            | Number of divisions the min-max range is divided into, in the 
            | fallback case in which unit=='au' and the 3e 4-vector element 
            | is NaN or positive.
            
    Returns:
        :tcs:
            | ndarray with Tcs
            
    """
    
    # make tupleof depth 2 if not already:
    if isinstance(tc4[-1],str): 
         tc4 = tuple([tc4])
    
    # use seamless stitch from tuple/list of tc4s:
    if isinstance(tc4[-1],bool): 
        seamless_stitch = tc4[-1]
        tc4 = tc4[:-1]
    
    # check all units (should be the same!!):
    if seamless_stitch & (len(tc4)>1):
        units = np.array([(tc4_i[3]==tc4[0][3]) for tc4_i in tc4])
        if (~units).any():
            raise Exception('Seamless stitching for unequal units not supported.')
    
    # loop over all tc4s and 'stitch' them together.
    Ts = None    
    for i,tc4_i in enumerate(tc4):
        tc4_i = list(tc4_i)
        if seamless_stitch & (i>0): 
            if ('-1' in tc4_i[-1]):
                tc4_i[0] = 1e6/Ts[0,0] # change T0
            else:
                tc4_i[0] = Ts[-1,0] # change T0

        Ts_i = _get_tcs4(tc4_i, uin = uin, 
                         fallback_unit = fallback_unit,
                         fallback_n = fallback_n)
        
        if (i == 0):
            Ts = Ts_i  
        else:
            if '-1' in tc4_i[-1]: # avoid overlap
                Ts = np.vstack((Ts_i[Ts_i[:,0]<Ts[0,0],:],Ts))
            else:
                Ts = np.vstack((Ts,Ts_i[Ts_i[:,0]>Ts[-1,0],:]))
    return Ts
          

def _generate_tcs(tc4, uin = None, seamless_stitch = True, cct_max = _CCT_MAX, cct_min = _CCT_MIN, 
                  fallback_unit = _CCT_FALLBACK_UNIT, 
                  fallback_n = _CCT_FALLBACK_N,
                  resample_ndarray = False):
    """ 
    Get an ndarray of Tc's obtained from a list or tuple of tc4 4-vectors (or ndarray).
        
    Args:
        :tc4:
            | list or tuple of 4-vectors or ndarray.
            | If ndarray: return tc4 limited to a cct_min-cct_max range (do nothing else).
            | If list/tuple: e.g. (tc4_1, tc4_2, tc4_3,...) or (tc4_1, tc4_2, tc4_3,..., bool::seamless_stitch)
            | When the last element of the list/tuple is a bool, then this specifies
            |  how the Tc arrays generated for each of the 4-vector elements need to be
            |  stitched together. This overrides the seamless_stitch input argument.
            | Vector elements are: 
            |    [Tmin, Tmax inclusive, Tinterval(or number of intervals), unit]
            |  Unit specifies unit of the Tc interval, i.e. it determines the
            |       type of scale in which the spacing of the Tc are done.
            |  Unit options are:
            |   - '%': equal relative Tc spacing (in %, cfr. (Ti+1 - Ti-1)/Ti-1).
            |   - 'K' equal absolute Tc spacing (in K, cfr. (Ti+1 - Ti-1).
            |   - '%-1': equal relative reciprocal Tc (MK-1 = mired).
            |   - 'K-1': equal absolute reciprocal Tc (MK-1 = mired).
            |  If the 'interval' element is negative, it actually represents
            |   the number of intervals between Tmin, Tmax (included).
        :uin:
            | None, optional
            | Unit of input Tmin, Tmax (by default it is assumed to be the same
            | as the scale 'unit').
        :seamless_stitch:
            | True, optional
            | Determines how the Tc arrays generated for each of the 4-vector 
            | elements are stitched together. Is overriden by the presence of a 
            | bool as last list/tuple element in :tc4:.
            | For a seamless stitch, all units for all 4-vectors should be the same!!
        :cct_max:
            | _CCT_MAX, optional
            | Limit Tc's to a maximum value of cct_max
        :cct_min:
            | _CCT_MIN, optional
            | Limit Tc's to a minimum value of cct_max
        :fallback_unit:
            | _CCT_FALLBACK_UNIT, optional
            | Unit to fall back on when the input unit in tc4 (of first list) is 'au'.
            | As there is no common distancing of the unit types ['K','%','%-1','K-1']
            | the Tc's are generated by dividing the min-max range into 
            | a number of divisions, specified by the negative 3 element (or when
            | positive or NaN, the number of divisions is set by :fallback_divisions:)
        :fallback_n:
            | _CCT_FALLBACK_N, optional
            | Number of divisions the min-max range is divided into, in the 
            | fallback case in which unit=='au' and the 3e 4-vector element 
            | is NaN or positive.
        :resample_ndarray:
            | False, optional
            | If False: do not resample Tc's of an ndarray input for tc4
            | else: divide min-max range in fallback_n intervals. Uses fallback_unit
            | to determine the scale for the resampling.
    
    Returns:
        :tcs:
            | ndarray with Tcs
            
    """

    # Get ccts for lut generation:
    if not isinstance(tc4,np.ndarray):

        Ts = get_tcs4(tc4, uin = uin, seamless_stitch = seamless_stitch,
                      fallback_unit = fallback_unit, 
                      fallback_n = fallback_n)

    else:

        if resample_ndarray:
            T0 = tc4[:,0].min()
            T1 = tc4[:,0].max()
            n = -fallback_n
            u = fallback_unit
            tc4_a = (T0,T1,n,u)
            
            Ts = get_tcs4(tc4_a, uin = 'K')
            
        else:
            Ts = tc4[:,:1] # actually stores ccts already! [Nx1] with N>2 !

    Ts[(Ts<cct_min)] = cct_min
    Ts[(Ts>cct_max)] = cct_max # limit to a maximum cct to avoid overflow/error and/or increase speed.    
    return Ts              
 
    
def _get_lut_characteristics(lut, force_au = True, tuple_depth_2 = True):
    """ 
    Guesses the interval, unit and wavelength range from lut array.
     (slow, so avoid use: set force_au to True !!)
    """
    if force_au:
        if tuple_depth_2: 
            return ((lut[:,0].min(),lut[:,0].max(), np.nan, 'au'),)
        else:
            return (lut[:,0].min(),lut[:,0].max(), np.nan, 'au')
    else:
        
        lut_units = np.array(['K','%','K-1','%-1'])
        
        T = lut[:,:1]
        T1 = np.roll(T,1)
        T01 = T[1:]
        T11 = T1[1:]
        
        # cfr. 'K'
        dT = (np.round(np.abs(T - T1),8)[1:])
        intsT = np.unique(dT)
        minmaxT = [[T01[(dT==intsT[i])].min(),T01[(dT==intsT[i])].max()] for i in range(len(intsT))]
        minmaxT[0][0] = T[0,0]
        
        # cfr '%:
        dTr = np.round((T01-T11)/T11,8)*100
        intsTr = np.unique(dTr)
        minmaxTr = [[T01[(dTr==intsTr[i])].min(),T01[(dTr==intsTr[i])].max()] for i in range(len(intsTr))]
        minmaxTr[0][0] = T[0,0]
        
        # cfr. 'K-1':
        dRD = (np.round(np.abs(1e6/T01 - 1e6/T11),8))
        intsRD = np.unique(dRD)
        minmaxRD = [[1e6/T01[(dRD==intsRD[i])].max(),1e6/T01[(dRD==intsRD[i])].min()] for i in range(len(intsRD))]
        minmaxRD[-1][1] = 1e6/T[0,0]
          
        # cfr. '%-1':
        dRDr = np.round((T01-T11)/T11,8)*100
        intsRDr = np.unique(dRDr)
        minmaxRDr = [[1e6/T01[(dRDr==intsRDr[i])].max(),1e6/T01[(dRDr==intsRDr[i])].min()] for i in range(len(intsRDr))]
        minmaxRDr[-1][1] = 1e6/T[0,0]
        
        # find minimum lengths & units for which the min max order is ok: 
        len_ints = np.array([intsT.shape[0],intsTr.shape[0],intsRD.shape[0],intsRDr.shape[0]])
        if len_ints.min()>1: 
            minmax_order_ok = np.array([minmaxT[0][1]<=minmaxT[1][0],
                                    minmaxTr[0][1]<=minmaxTr[1][0],
                                    minmaxRD[0][1]<=minmaxRD[1][0],
                                    minmaxRDr[0][1]<=minmaxRDr[1][0]])
        else:
            minmax_order_ok = np.ones((4,),dtype = bool)
        
        # determine unit:
        len_order_ok = ((len_ints == len_ints.min()) & minmax_order_ok)
        lut_unit = lut_units[len_order_ok]
        if len(lut_unit) == 0: lut_unit = 'au'
        # for code testing:
        # return {'K':(dT,intsT,minmaxT),
        #         '%':(dTr,intsTr,minmaxTr),
        #         'K-1':(dRD,intsRD,minmaxRD),
        #         '%-1':(dRDr,intsRDr,minmaxRDr),
        #         'len_ints':len_ints,
        #         'minmax_order_ok':minmax_order_ok,
        #         'len_order_ok' : len_order_ok,
        #         'lut_unit':lut_unit}
        
        if lut_unit[0] == 'K':
            dT, lut_unit, Tminmax = intsT,lut_unit[0],minmaxT
        elif lut_unit[0] == '%':
            dT, lut_unit, Tminmax = intsTr,lut_unit[0],minmaxTr 
        elif lut_unit[0] == 'K-1':
            dT, lut_unit, Tminmax = intsRD,lut_unit[0],minmaxRD
        elif lut_unit[0] == '%-1':
            dT, lut_unit, Tminmax = intsRDr,lut_unit[0],minmaxRDr
        else:
            dT, lut_unit, Tminmax = np.nan, 'au', [[T.min(),T.max()]]
        if not np.isnan(dT).any(): dT = dT[0] if (dT.shape[0] == 1) else tuple(dT)
        Tminmax = Tminmax[0] if (len(Tminmax) == 1) else Tminmax 
        
        # format output:
        if isinstance(dT,(float,int)): 
            if tuple_depth_2: 
                return ((*Tminmax, dT, lut_unit),)
            else:
                return (*Tminmax, dT, lut_unit)
        else: 
            tmp = (*np.array(Tminmax).T, dT, (lut_unit,)*len(dT)) 
            tmp = np.array(tmp,dtype=object).T
            return (*tuple((tuple(tmp[i]) for i in range(tmp.shape[0]))),lut_unit!='au')
        
 
#------------------------------------------------------------------------------
# LUT generation functions:
#------------------------------------------------------------------------------

def calculate_lut(ccts, cieobs, wl = None, lut_vars = ['T','uv','uvp','uvpp','iso-T-slope'],
                  cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS):
    """
    Function that calculates a LUT for the specified calculation method 
    for the input ccts. Calculation is performed for CMF set specified in 
    cieobs and in the chromaticity diagram in cspace. 
    
    Args:
        :ccts: 
            | ndarray [Nx1] or str
            | list of ccts for which to (re-)calculate the LUTs.
            | If str, ccts contains path/filename.dat to list.
        :cieobs: 
            | None or str, optional
            | str specifying cmf set.
        :wl: 
            | None, optional
            | Generate luts based on Planckians with wavelengths (range). 
            | If None: use same wavelengths as CMFs in :cieobs:.
        :lut_vars:
            | ['T','uv','uvp','uvpp','iso-T-slope'], optional
            | Data the lut should contain. Must follow this order 
            | and minimum should be ['T']
        :cspace:
            | _CCT_SPACE, optional
            | Color space to do calculations in. 
            | Options: 
            |    - cspace string: 
            |        e.g. 'Yuv60' for use with luxpy.colortf()
            |    - tuple with forward (i.e. xyz_to..) [and backward (i.e. ..to_xyz)] functions 
            |      (and an optional string describing the cspace): 
            |        e.g. (forward, backward) or (forward, backward, cspace string) or (forward, cspace string) 
            |    - dict with keys: 'fwtf' (foward), 'bwtf' (backward) [, optional: 'str' (cspace string)]
            |  Note: if the backward tf is not supplied, optimization in cct_to_xyz() is done in the CIE 1976 u'v' diagram
        :cspace_kwargs:
            | _CCT_CSPACE_KWARGS, optional
            | Parameter nested dictionary for the forward and backward transforms.
    
    Returns:
        :returns: 
            :lut:
                | ndarray with T, u, v, u', v', u", v", slope (note ':1st deriv., ":2nd deriv.).
                                            
    """    
    if isinstance(ccts, str):
        ccts = getdata(ccts)
    
        
    if ('uv' not in lut_vars) & ('uvp' not in lut_vars) & ('uvpp' not in lut_vars) & ('iso-T-slope' not in lut_vars):
        # no need to calculate anything, only Tcs needed
        return np2d(ccts)
    
    # Determine what to calculate:
    outBB = 'BB'
    if ('uvp' in lut_vars) | ('iso-T-slope' in lut_vars): 
        outBB = outBB + 'uvp'
    if ('uvpp' in lut_vars): outBB = outBB + 'uvpp'
 

    # get requested cmf set:
    xyzbar, wl, dl = _get_xyzbar_wl_dl(cieobs, wl)
    
    # process cspace input:
    cspace_dict, cspace_str = _process_cspace(cspace, cspace_kwargs)
    
    # convert to cspace based cmfs (Eq.6-7):
    uvwbar = _convert_xyzbar_to_uvwbar(xyzbar, cspace_dict) 
    
    # calculate U,V,W (Eq. 6) and U',V',W' (Eq.10) [Robertson,1986] and U",V",W" [Li,2016; started from XYZ, but this is equivalent]:
    Ti, UVW, UVWp, UVWpp = _get_tristim_of_BB_BBp_BBpp(ccts, uvwbar, wl, dl, out = 'BB,BBp,BBpp')

    # calculate li, mi:
    R = UVW.sum(axis=-1, keepdims = True) # for Ohno, 2014 & Robertson, 1968 & Li, 2016
    if UVWp is not None: Rp = UVWp.sum(axis=-1, keepdims = True) # for Robertson, 1968 & Li, 2016
    if UVWpp is not None: Rpp = UVWpp.sum(axis=-1, keepdims = True) # for Li, 2016

    # avoid div by zero:
    if 'iso-T-slope' in lut_vars:
        num = (UVWp[:,1:2]*R - UVW[:,1:2]*Rp) 
        denom = (UVWp[:,:1]*R - UVW[:,:1]*Rp)
        num[(num == 0)] += _CCT_AVOID_ZERO_DIV
        denom[(denom == 0)] += _CCT_AVOID_ZERO_DIV
    
        li = num/denom  
        li = li + np.sign(li)*_CCT_AVOID_ZERO_DIV # avoid division by zero
        mi = -1.0/li # slope of isotemperature lines
    else:
        mi = None
    
    # get u,v & u',v' and u",v":
    uvi = UVW[:,:2]/R
    if UVWp is not None: uvpi = UVWp[:,:2]/Rp
    if UVWpp is not None: uvppi = UVWpp[:,:2]/Rpp
    
    # construct output (use comple if structure to avoid creating intermediate arrays for optimal speed):
    if   ('uvp' in lut_vars) & ('uvpp' in lut_vars) & ('iso-T-slope' in lut_vars):
        lut = np.hstack((Ti,uvi,uvpi,uvppi,mi))
    elif ('uvp' not in lut_vars) & ('uvpp' not in lut_vars) & ('iso-T-slope' not in lut_vars):
        lut = np.hstack((Ti,uvi))
        
    elif ('uvp' in lut_vars) & ('uvpp' not in lut_vars) & ('iso-T-slope' in lut_vars):
        lut = np.hstack((Ti,uvi, uvpi, mi))
    elif ('uvp' in lut_vars) & ('uvpp' not in lut_vars) & ('iso-T-slope' not in lut_vars):
        lut = np.hstack((Ti,uvi, uvpi))
    elif ('uvp' in lut_vars) & ('uvpp' in lut_vars) & ('iso-T-slope' not in lut_vars):
        lut = np.hstack((Ti,uvi, uvpi, uvppi))
        
    elif ('uvp' not in lut_vars) & ('uvpp' in lut_vars) & ('iso-T-slope' in lut_vars):
        lut = np.hstack((Ti,uvi, uvppi, mi))
    elif ('uvp' not in lut_vars) & ('uvpp' in lut_vars) & ('iso-T-slope' not in lut_vars):
        lut = np.hstack((Ti,uvi, uvppi))
           
    elif ('uvp' not in lut_vars) & ('uvpp' not in lut_vars) & ('iso-T-slope' in lut_vars):
        lut = np.hstack((Ti,uvi,mi))

    return lut 
    
def _generate_lut(tc4, uin = None, seamless_stitch = True, 
                  fallback_unit = _CCT_FALLBACK_UNIT, fallback_n = _CCT_FALLBACK_UNIT,
                  resample_ndarray = False,
                  cct_max = _CCT_MAX, cct_min = _CCT_MIN,
                  wl = None, cieobs = _CIEOBS, lut_vars = ['T','uv','uvp','uvpp','iso-T-slope'],
                  cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                  **kwargs):
    """ 
    Get an ndarray LUT for Tc's obtained from a list or tuple of tc4 4-vectors (or ndarray).
        
    Args:
        :tc4:
            | list or tuple of 4-vectors or ndarray.
            | If ndarray: return tc4 limited to a cct_min-cct_max range (do nothing else).
            | If list/tuple: e.g. (tc4_1, tc4_2, tc4_3,...) or (tc4_1, tc4_2, tc4_3,..., bool::seamless_stitch)
            | When the last element of the list/tuple is a bool, then this specifies
            |  how the Tc arrays generated for each of the 4-vector elements need to be
            |  stitched together. This overrides the seamless_stitch input argument.
            | Vector elements are: 
            |    [Tmin, Tmax inclusive, Tinterval(or number of intervals), unit]
            |  Unit specifies unit of the Tc interval, i.e. it determines the
            |       type of scale in which the spacing of the Tc are done.
            |  Unit options are:
            |   - '%': equal relative Tc spacing (in %, cfr. (Ti+1 - Ti-1)/Ti-1).
            |   - 'K' equal absolute Tc spacing (in K, cfr. (Ti+1 - Ti-1).
            |   - '%-1': equal relative reciprocal Tc (MK-1 = mired).
            |   - 'K-1': equal absolute reciprocal Tc (MK-1 = mired).
            |  If the 'interval' element is negative, it actually represents
            |   the number of intervals between Tmin, Tmax (included).
        :uin:
            | None, optional
            | Unit of input Tmin, Tmax (by default it is assumed to be the same
            | as the scale 'unit').
        :seamless_stitch:
            | True, optional
            | Determines how the Tc arrays generated for each of the 4-vector 
            | elements are stitched together. Is overriden by the presence of a 
            | bool as last list/tuple element in :tc4:.
            | For a seamless stitch, all units for all 4-vectors should be the same!!
        :cct_max:
            | _CCT_MAX, optional
            | Limit Tc's to a maximum value of cct_max
        :cct_min:
            | _CCT_MIN, optional
            | Limit Tc's to a minimum value of cct_max
        :fallback_unit:
            | _CCT_FALLBACK_UNIT, optional
            | Unit to fall back on when the input unit in tc4 (of first list) is 'au'.
            | As there is no common distancing of the unit types ['K','%','%-1','K-1']
            | the Tc's are generated by dividing the min-max range into 
            | a number of divisions, specified by the negative 3 element (or when
            | positive or NaN, the number of divisions is set by :fallback_divisions:)
        :fallback_n:
            | _CCT_FALLBACK_N, optional
            | Number of divisions the min-max range is divided into, in the 
            | fallback case in which unit=='au' and the 3e 4-vector element 
            | is NaN or positive.
        :resample_tc4_array:
            | False, optional
            | If False: do not resample Tc's of an ndarray input for tc4
            | else: divide min-max range in fallback_n intervals. Uses fallback_unit
            | to determine the scale for the resampling.
        :wl:
            | None, optional
            | Wavelength for Planckian spectrum generation.
            | If None: use same wavelengths as CMFs in :cieobs:.
        :cieobs:
            | [_CIEOBS] or list, optional
            | Generate a LUT for each one in the list.
            | If None: generate for all cmfs in _CMF.
        :lut_vars:
            | ['T','uv','uvp','uvpp','iso-T-slope'], optional
            | Data the lut should contain. Must follow this order 
            | and minimum should be ['T']
        :cspace,cspace_kwargs:
            | Lists with the cspace and cspace_kwargs for which luts will be generated.
            | Default is single chromaticity diagram in _CCT_CSPACE.

    Returns:
        :lut:
            | List with an ndarray with in the columns whatever is specified in 
            | lut_vars (Tc and uv are always present!).
            | Default lut_vars =  ['T','uv','uvp','uvpp','iso-T-slope']
            | - Tc: (in K)
            | - u,v: chromaticity coordinates of planckians
            | - u'v': chromaticity coordinates of 1st derivative of the planckians.
            | - u",v": chromaticity coordinates of 2nd derivative of the planckians.
            | - slope of isotemperature lines (calculated as in Robertson, 1968).
        :lut_kwargs:
            | {},
            | Dictionary with additional parameters related to the generation of the
            | lut.
            
    """    
    # get tcs:
    Ts = _generate_tcs(tc4, uin = uin, seamless_stitch = seamless_stitch,
                       fallback_unit = fallback_unit, 
                       fallback_n = fallback_n,
                       cct_min = cct_min, cct_max = cct_max,
                       resample_ndarray = resample_ndarray)
     
    if (len(lut_vars) == 1) & (lut_vars[0] == 'T'):
        return list([Ts,{}]) #no need to do anythin, except output lut containining Tcs only
    else:
        
        # reshape for input in calculate_luts:
        n_sources = Ts.shape[-1]
        
        if n_sources > 1:
            Ts = np.reshape(Ts,(-1,1))
    
        # calculate lut:
        lut = calculate_lut(ccts = Ts, wl = wl, cieobs = cieobs, lut_vars = lut_vars,
                            cspace = cspace, cspace_kwargs = cspace_kwargs)
    
        
        # reshape lut back:
        if n_sources > 1:
            lut = np.reshape(lut,(-1,lut.shape[-1]*n_sources))
          
        
        return list([lut, {}])

    
def _get_lut(lut, 
             uin = None, seamless_stitch = True, 
             fallback_unit = _CCT_FALLBACK_UNIT, fallback_n = _CCT_FALLBACK_N,
             resample_ndarray = False, cct_max = _CCT_MAX, cct_min = _CCT_MIN,
             luts_dict = None, lut_type_def = None, lut_vars = ['T','uv','uvp','uvpp','iso-T-slope'],
             cieobs =  _CIEOBS, cspace_str = None, wl = None, ignore_unequal_wl = False, 
             lut_generator_fcn = _generate_lut, lut_generator_kwargs = {},
             cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
             **kwargs):
    """ 
    Get an ndarray LUT from various sources.
        
    Args:
        :lut:
            | Look-Up-Table with Ti, u,v,u',v',u",v",slope values of Planckians, or
            | whatever quantities are specified in lut_vars ('T','uv' is always part of the lut).
            | Options: 
            | - list: must have two elements: [lut,lut_kwargs]
            | - None: lut from luts_dict with lut_type_def as key
            | - str: lut from luts_dict at key :lut:
            | - ndarray [Nxn, with n>1]: precalculated lut (only processing will be to keep it with cct_min-cct_max range)
            | - ndarray [Nx1]: list of Tc's from which a new lut will be calculated.
            | - tuple of 4-vectors: used as key in luts_dict or to generate new lut from scratch
            |    4-vector info: 
            |       + format: e.g. (tc4_1, tc4_2, tc4_3,...) or (tc4_1, tc4_2, tc4_3,..., bool::seamless_stitch)
            |       + When the last element of the list/tuple is a bool, then this specifies
            |         how the Tc arrays generated for each of the 4-vector elements need to be
            |         stitched together. This overrides the seamless_stitch input argument.
            |       + Vector elements are: 
            |           [Tmin, Tmax inclusive, Tinterval(or number of intervals), unit]
            |         Unit specifies unit of the Tc interval, i.e. it determines the
            |         type of scale in which the spacing of the Tc are done.
            |         Unit options are:
            |           - '%': equal relative Tc spacing (in %, cfr. (Ti+1 - Ti-1)/Ti-1).
            |           - 'K' equal absolute Tc spacing (in K, cfr. (Ti+1 - Ti-1).
            |           - '%-1': equal relative reciprocal Tc (MK-1 = mired).
            |           - 'K-1': equal absolute reciprocal Tc (MK-1 = mired).
            |         If the 'interval' element is negative, it actually represents
            |         the number of intervals between Tmin, Tmax (included).
        :uin:
            | None, optional
            | Unit of input Tmin, Tmax (by default it is assumed to be the same
            | as the scale 'unit') in Tc generation from tuple.
        :seamless_stitch:
            | True, optional
            | Determines how the Tc arrays generated for each of the 4-vector 
            | elements are stitched together. Is overriden by the presence of a 
            | bool as last list/tuple element in :tc4:.
            | For a seamless stitch, all units for all 4-vectors should be the same!!
        :cct_max:
            | _CCT_MAX, optional
            | Limit Tc's to a maximum value of cct_max
        :cct_min:
            | _CCT_MIN, optional
            | Limit Tc's to a minimum value of cct_max
        :fallback_unit:
            | _CCT_FALLBACK_UNIT, optional
            | Unit to fall back on when the input unit in tc4 (of first list) is 'au'.
            | As there is no common distancing of the unit types ['K','%','%-1','K-1']
            | the Tc's are generated by dividing the min-max range into 
            | a number of divisions, specified by the negative 3 element (or when
            | positive or NaN, the number of divisions is set by :fallback_divisions:)
        :fallback_n:
            | _CCT_FALLBACK_N, optional
            | Number of divisions the min-max range is divided into, in the 
            | fallback case in which unit=='au' and the 3e 4-vector element 
            | is NaN or positive.
        :resample_tc4_array:
            | False, optional
            | If False: do not resample Tc's of an ndarray input for tc4
            | else: divide min-max range in fallback_n intervals. Uses fallback_unit
            | to determine the scale for the resampling.
        :wl:
            | None, optional
            | Wavelength for Planckian spectrum generation.
            | If None: use same wavelengths as CMFs in :cieobs:.
        :cieobs:
            | _CIEOBS, optional
            | CMF set used to convert Planckian spectra to chromaticity coordinates
        :lut_type_def:
            | None, placeholder
            | Default lut (tuple key) to read from luts_dict.
        :luts_dict:
            | None, optional
            | Dictionary of pre-calculated luts for various cspaces and cmf sets.
            |  Must have structure luts_dict[cspace][cieobs][lut_label] with the
            |   lut part of a two-element list [lut, lut_kwargs]. It must contain
            |   at the top-level a key 'wl' containing the wavelengths of the 
            |   Planckians used to generate the luts in this dictionary.
            | If None: the default dict for the mode is used 
            |   (e.g. _CCT_LUT['ohno2014']['lut_type_def'], for mode=='ohno2014').    
        :lut_vars:
            | ['T','uv','uvp','uvpp','iso-T-slope'], optional
            | Data the lut should contain. Must follow this order 
            | and minimum should be ['T']
        :cspace,cspace_kwargs:
            | Lists with the cspace and cspace_kwargs for which luts will be generated.
            | Default is single chromaticity diagram in _CCT_CSPACE.
        :ignore_unequal_wl:
            | False, optional
            | If True: ignore any differences in the wavelengths used to calculate
            |   the lut (cfr. Planckians) from the luts_dict and the requested 
            |   wavelengths in :wl:
        :lut_generator_fcn:
            | _generate_lut, optional
            | Lets a user specify his own lut generation function (must output a list of 1 lut). 
            | Default is the general function. There is a specific one for
            | Ohno's 2014 method as that one requires a different correction factor
            | for each lut for the parabolic solutions. This optimized value is specified in the 
            | second list index. (see _generate_lut_ohno2014()).
        :lut_generator_kwargs:
            | {}, optional
            | Dict with keyword arguments specific to the (user) lut_generator_fcn.
            |  (e.g. {'f_corr':0.9991} for _generate_lut_ohno2014())  

    Returns:
        :lut:
            | List with an ndarray with in the columns whatever is specified in 
            | lut_vars (Tc and uv are always present!).
            | Default lut_vars =  ['T','uv','uvp','uvpp','iso-T-slope']
            | - Tc: (in K)
            | - u,v: chromaticity coordinates of planckians
            | - u'v': chromaticity coordinates of 1st derivative of the planckians.
            | - u",v": chromaticity coordinates of 2nd derivative of the planckians.
            | - slope of isotemperature lines (calculated as in Robertson, 1968).
        :lut_kwargs:
            | {}
            | Dictionary with additional parameters related to the generation of the
            | lut.
            
    """  
    # get cspace info:
    cspace_dict, cspace_str = _process_cspace(cspace, cspace_kwargs)

    lut_kwargs = lut_generator_kwargs # default settings, but can be overriden later depending on the lut input argument

    if isinstance(lut,list): # should contain [(tuple,list,ndarray)::lut,dict::lut_kwargs]
        lut_list_error = True 
        if (len(lut) == 2):
            if (isinstance(lut[-1],dict)): # test for lut_kwargs presence 
                lut_kwargs = lut[1]
                lut = lut[0]
                lut_list_error = False
 
        if lut_list_error: 
            raise Exception("""When lut input is a list, the first element 
contains the lut as a tuple/list, tuple/list of 
tuples/lists or as an ndarray; the second element
should be a dictionary 'lut_kwargs'.""")
    
    
    luts_dict_empty = False 
    # lut_from_dict = False
    lut_from_tuple = False
    # lut_from_str = False
    lut_from_Tcs = False
    lut_from_array = False 
    unequal_wl = False
    lut_tuple = None

    # print(luts_dict[cspace_str][cieobs].keys(),lut)
    if lut is None: lut = lut_type_def # use default type in luts_dict
 
    # further process lut (1st element of input lut): 
    if isinstance(lut, (tuple,str)): # lut is key in luts_dict, if not generate new lut from scratch
        lut_from_tuple = True
        lut_tuple = lut # keep copy to use later as key

        if luts_dict is not None: # luts_dict is None: generate a new lut from scratch
            if ('wl' not in luts_dict): 
                luts_dict_empty = True # if not present luts_dict must be empty 
            else:
                if cieobs not in luts_dict['wl']:
                    luts_dict_empty = True # is empty for this cieobs
                    
            if cieobs in luts_dict[cspace_str]:
                if lut in luts_dict[cspace_str][cieobs]: # read from luts_dict
                    lut, lut_kwargs = copy.deepcopy(luts_dict[cspace_str][cieobs][lut])
                    lut_from_tuple = False
    
    elif isinstance(lut, np.ndarray): # lut is either pre-calculated lut or a list with Tcs for which a lut needs to be generated
        lut_from_array = True
        if lut.ndim == 1:
            lut = lut[:,None] # make 2D
        if lut.shape[-1] == 1:
            lut_from_Tcs = True
            lut_from_array = False # signal that the lut doesn't exist yet!

    # If pre-calculated from luts_dict, check if wavelengths agree.
    # When directly entered as ndarray there is no way to check, 
    # so assume unequal_wl==False:
    if ignore_unequal_wl == False:
        if (luts_dict is not None) & (luts_dict_empty == False):
            if not np.array_equal(luts_dict['wl'][cieobs], wl):
                unequal_wl = True    

    if (unequal_wl  | luts_dict_empty| lut_from_tuple | lut_from_Tcs | resample_ndarray):
    
        if cspace_dict is None: raise Exception('No cspace dict or other given !')
    
    
        if (not lut_from_array) | (resample_ndarray):

            lut, lut_kwargs = lut_generator_fcn(lut, 
                                                uin = uin,
                                                seamless_stitch = seamless_stitch,
                                                fallback_unit = fallback_unit,
                                                fallback_n = fallback_n,
                                                resample_ndarray = resample_ndarray,
                                                cct_max = cct_max,
                                                cct_min = cct_min,
                                                lut_vars = lut_vars,
                                                wl = wl, 
                                                cieobs = cieobs, 
                                                cspace = cspace_dict,
                                                cspace_kwargs = None,
                                                **lut_kwargs)
            
            if luts_dict is not None:
                if cieobs not in luts_dict['wl']:
                    luts_dict['wl'][cieobs] = wl
                if cieobs not in luts_dict[cspace_str]:
                    luts_dict[cspace_str][cieobs] = {} # create empty dict for new cieobs
                if lut_tuple is not None: 
                    luts_dict[cspace_str][cieobs][lut_tuple] = [lut, lut_kwargs] # store for later use

        else:
            lut = lut[(lut[:,0]>=cct_min) & (lut[:,0]<=cct_max),:]
            

    return list([lut, lut_kwargs])
  

def generate_luts(types = [None], seamless_stitch = True,
                  fallback_unit = _CCT_FALLBACK_UNIT, fallback_n = _CCT_FALLBACK_N,
                  cct_min = _CCT_MIN, cct_max = _CCT_MAX,
                  lut_file = None, load = False, lut_path = _CCT_LUT_PATH, save_luts = True, 
                  wl = None, cieobs = [_CIEOBS], 
                  lut_vars = ['T','uv','uvp','uvpp','iso-T-slope'],
                  cspace = [_CCT_CSPACE], cspace_kwargs = [_CCT_CSPACE_KWARGS],
                  verbosity = 0, lut_generator_fcn = _generate_lut, 
                  lut_generator_kwargs = {}):
    """
    Generate a number of luts and store them in a nested dictionary.
    Structure: lut[cspace][cieobs][lut type].
    
    Args:
        :lut_file:
            | None, optional
            | string specifying the filename to save the lut (as .pkl) to.
            | If None: don't save anything when generated (i.e. load==False).
        :load:
            | True, optional
            | If True: load previously generated dictionary.
            | If False: generate from scratch.
        :lut_path:
            | _CCT_LUT_PATH, optional
            | Path to file.
        :wl:
            | None, optional
            | Wavelength for Planckian spectrum generation.
            | If None: use same wavelengths as CMFs in :cieobs:.
        :cieobs:
            | [_CIEOBS] or list, optional
            | Generate a LUT for each one in the list.
            | If None: generate for all cmfs in _CMF.
        :types:
            | [None], optional
            | List of lut specifiers of format [(Tmin,Tmax,Tinterval,unit),...]
            | If units are in MK-1 then the range is also!
            |  Unit options are:
            |  - '%': equal relative Tc spacing (in %, cfr. (Ti+1 - Ti-1)/Ti-1).
            |  - 'K' equal absolute Tc spacing (in K, cfr. (Ti+1 - Ti-1).
            |  - '%-1': equal relative reciprocal Tc (MK-1 = mired).
            |  - 'K-1': equal absolute reciprocal Tc (MK-1 = mired).
            | If the last element of the list is a bool, then the way the different
            | lists of Tcs generated by each list element can be set. If True:
            | the Tcs will be 'seamlessly' stitched together (this does have an
            | an impact on the min-max range of each Tc set) so that there are no
            | discontinuities in terms of the intervals.
        :seamless_stitch:
            | True, optional
            | When stitching (creating) LUTs composed of several CCT ranges with different
            | intervals, these do not always 'match' well, in the sense that discontinuities
            | might be generated. This can be avoided (at the expense of possibly slightly changed ranges)
            | by setting the :seamless_stitch: argument to True. Is overriden when
            | the last element in the lut list is a boolean.
        :cct_max:
            | _CCT_MAX, optional
            | Limit Tc's to a maximum value of cct_max
        :cct_min:
            | _CCT_MIN, optional
            | Limit Tc's to a minimum value of cct_max
        :fallback_unit:
            | _CCT_FALLBACK_UNIT, optional
            | Unit to fall back on when the input unit in tc4 (of first list) is 'au'.
            | As there is no common distancing of the unit types ['K','%','%-1','K-1']
            | the Tc's are generated by dividing the min-max range into 
            | a number of divisions, specified by the negative 3 element (or when
            | positive or NaN, the number of divisions is set by :fallback_divisions:)
        :fallback_n:
            | _CCT_FALLBACK_N, optional
            | Number of divisions the min-max range is divided into, in the 
            | fallback case in which unit=='au' and the 3e 4-vector element 
            | is NaN or positive.
        :lut_vars:
            | ['T','uv','uvp','uvpp','iso-T-slope'], optional
            | Data the lut should contain. Must follow this order 
            | and minimum should be ['T']
        :cspace,cspace_kwargs:
            | Lists with the cspace and cspace_kwargs for which luts will be generated.
            | Default is single chromaticity diagram in _CCT_CSPACE.
        :verbosity:
            | 0, optional
            | If > 0: give some intermediate feedback while generating luts.
        :lut_generator_fcn:
            | _generate_lut, optional
            | Lets a user specify his own lut generation function (must output a list of 1 lut). 
            | Default is the general function. There is a specific one for
            | Ohno's 2014 method as that one requires a different correction factor
            | for each lut for the parabolic solutions. This optimized value is specified in the 
            | second list index. (see _generate_lut_ohno2014()).
        :lut_generator_kwargs:
            | {}, optional
            | Dict with keyword arguments specific to the (user) lut_generator_fcn.
            |  (e.g. {'f_corr':0.9991} for _generate_lut_ohno2014())
            
        Returns:
            :dict:
                | Dictionary with luts for the specified mode, cieobs(s) and cspace(s).
                | Structure: lut[cspace][cieobs][lut type]
                | At the upper dict level there is also a key 'wl' which contains a dict with keys 
                | the cieobs and with values the wavelengths used to calculate the Planckians for
                | each lut for the specified cieobs; as well as a key with the lut_vars
                | The luts contains as data the variables as specified in lut_vars:
                | - T: (in K)
                | - uv: chromaticity coordinates of planckians
                | - uvp: chromaticity coordinates of 1st derivative of the planckians.
                | - uvpp: chromaticity coordinates of 2nd derivative of the planckians.
                | - iso-T-slope: slope of isotemperature lines (calculated as in Robertson, 1968).
    """
    luts = {'lut_vars' : lut_vars} 
    # lut_units = ['%','K','%-1','K-1','au']

    # Calculate luts:
    if (load == False):
        #luts['wl'] = wl
        luts['wl'] = {} # store wavelengths of the cieobs
        
        for i, (cspace_i,cspace_kwargs_i) in enumerate(zip(cspace,cspace_kwargs)):
            
            cspace_dict_i,_ = _process_cspace(cspace_i, cspace_kwargs = cspace_kwargs_i)
            cspace_str_i = cspace_dict_i['str']
            luts[cspace_i] = {'cspace' : cspace_i, 'cspace_kwargs' : cspace_kwargs_i, 'cspace_dict': cspace_dict_i}
            
            if cieobs is None: cieobs = _CMF['types']
            
            for j,cieobs_j in enumerate(cieobs):
                
                luts['wl'][cieobs_j] = _CMF[cieobs_j]['bar'][0] if (wl is None) else getwlr(wl) # store wavelengths
                
                ftmp = lambda lut: lut_generator_fcn(lut, 
                                                     seamless_stitch = seamless_stitch, 
                                                     fallback_unit = fallback_unit, 
                                                     fallback_n = fallback_n,
                                                     cct_max = cct_max, 
                                                     cct_min = cct_min,
                                                     wl = luts['wl'][cieobs_j], 
                                                     cieobs = cieobs_j, 
                                                     cspace =  cspace_dict_i, 
                                                     cspace_kwargs = None,
                                                     lut_vars = lut_vars,
                                                     **lut_generator_kwargs)
                luts[cspace_i][cieobs_j] = {}
                
                for type_k in types:
                    # ensure full tuple depth for use as key:
                    if not isinstance(type_k[-1],str): # at least depth 2 (<--last element is str for depth 1)
                        tmp = list((tuple(type_kl) for type_kl in type_k if not isinstance(type_kl,bool)))
                        if (isinstance(type_k[-1],bool)): seamless_stitch = type_k[-1]
                        if (len(tmp)>1): tmp = (*tmp,seamless_stitch) # 

                    tmp = (tuple(tmp))
                    if verbosity > 0:
                        print('Generating lut with type = {} in cspace = {:s} for cieobs = {:s}'.format(type_k,cspace_str_i,cieobs_j))
                    
                    if (cieobs_j != 'cie_std_dev_obs_f1') & (cieobs_j != '1951_20_scotopic'):
                        luts[cspace_i][cieobs_j][tmp] = list(ftmp(tmp))
                    else:
                        luts[cspace_i][cieobs_j][tmp] = []
                    

                # save to disk (do intermediate saves):
                if (lut_file is not None) & (save_luts == True):
                    if (i == 0) & (j == 0): 
                        if not os.path.exists(lut_path):
                            os.makedirs(lut_path,exist_ok=True)
                    file_path = os.path.join(lut_path,lut_file)
                    if verbosity > 0:
                        print('Saving dict with luts in {:s}'.format(file_path))                                                 
                    save_pkl(file_path,luts)
        luts = load_pkl(file_path)
    else:
        if lut_file is not None:
            file_path = os.path.join(lut_path, lut_file)
            if verbosity > 1:
                print('Loading dict with luts in {:s}'.format(file_path))                                                 
            luts = load_pkl(file_path)
        else:
            raise Exception('Trying to load lut file but no lut_file has been supplied.')
    return luts

def _copy_luts(mode, cspace = _CCT_CSPACE, cieobs = ['2006_2','2006_10'], 
               cieobs_src = ['2015_2', '2015_10'], lut = _CCT_LUT):
    """ Copy luts for specific cieobs keys to equivalent cieobs keys"""
    cieobs_in_lut = list(lut[mode]['luts'][cspace].keys())
    for cieobs_i, cieobs_src_i in zip(cieobs, cieobs_src):
        if cieobs_src_i in cieobs_in_lut:
            lut[mode]['luts']['wl'][cieobs_i] = lut[mode]['luts']['wl'][cieobs_src_i]
            lut[mode]['luts'][cspace][cieobs_i] = lut[mode]['luts'][cspace][cieobs_src_i]

def _unique_types(lut_types):
    """ Get list of unique tuple lut_type specifiers"""
    utypes = []
    for type_lut in lut_types:
         if type_lut not in utypes:
             utypes.append(type_lut) 
    return utypes

def _sample_lut_vars(lut_vars, all_modes_luts, is_lut_kwargs_empty_mode = True):
    """ From a full set of lut_vars (8 columns in lut), select only those required by a mode"""
    lut = {'lut_vars' : lut_vars,
           'wl' : all_modes_luts['wl'],
           }
    
    # get lut_vars indices:
    lut_vars_n = {'T':1, 'uv':2, 'uvp':2,'uvpp':2,'iso-T-slope':1}
    lut_vars_indices = []
    cnt = 0
    for lut_vars_i in lut_vars: 
        lut_vars_indices = lut_vars_indices + list(cnt+np.arange(0,lut_vars_n[lut_vars_i]))
        cnt+=lut_vars_n[lut_vars_i]
    
    # sample request lut_vars:
    for cspace_i in all_modes_luts.keys():
        if (cspace_i == 'wl') | (cspace_i == 'lut_vars'):
            pass
        else:
            lut[cspace_i] = {}
            for cieobs_j in all_modes_luts[cspace_i].keys():
                if 'cspace' in cieobs_j:
                    lut[cspace_i][cieobs_j] = all_modes_luts[cspace_i][cieobs_j]
                else:
                    lut[cspace_i][cieobs_j] = {}
                    for lut_k in  all_modes_luts[cspace_i][cieobs_j].keys():
                        lut_array = all_modes_luts[cspace_i][cieobs_j][lut_k][0][:,lut_vars_indices]
                        lut_kwargs = {} if is_lut_kwargs_empty_mode else all_modes_luts[cspace_i][cieobs_j][lut_k][1]
                        lut[cspace_i][cieobs_j][lut_k] = [lut_array, lut_kwargs]
    return lut

def _lut_to_float64(luts):
    """ undo *1000 for storage as float32 """
    
    # get lut_vars indices:
    for cspace_i in luts.keys():
        if not ((cspace_i == 'wl') | (cspace_i == 'lut_vars')):
            for cieobs_j in luts[cspace_i].keys():
                if 'cspace' not in cieobs_j:
                    for lut_k in  luts[cspace_i][cieobs_j].keys():
                        lut_array = luts[cspace_i][cieobs_j][lut_k][0]
                        lut_kwargs = luts[cspace_i][cieobs_j][lut_k][1]
                        luts[cspace_i][cieobs_j][lut_k] = [lut_array, lut_kwargs]
    return luts


def _download_luts_from_github(modes = None, url = _CCT_LUT_PATH_LX_REPO):
    """ Download lut(s) from the luxpy github repo """
    import requests 
    from io import BytesIO
    if modes is None: modes = _CCT_LIST_OF_MODE_LUTS
    for mode in modes:
        r = requests.get(os.path.join(url,mode+'_luts.pkl'))
        lut = pickle.load(BytesIO(r.content))
        save_pkl(os.path.join(_CCT_LUT_PATH,mode+'_luts.pkl'),lut)
    
        
def _initialize_lut(mode, lut_types, force_calc = _CCT_LUT_CALC, wl = None):
    """ Pre-generate / load from disk / download from github some LUTs for a specific mode """
    if (mode in _CCT_LIST_OF_MODE_LUTS) & _CCT_LUT_ONE_NPY_PER_MODE:
        lut_exists = os.path.exists(os.path.join(_CCT_LUT_PATH,'{:s}_luts.pkl'.format(mode)))
        if (not lut_exists) & (force_calc == False):
            try:
                print("LUT pickle file for mode '{:s}' doesn't exist. Trying download from luxpy github repo.".format(mode))
                _download_luts_from_github(modes=[mode])
                lut_exists = True
            except:
                lut_exists = False
                print("Couldn't download LUTs from luxpy github repo. Will generate from scratch. This might take a while.")
        _CCT_LUT[mode]['luts'] = generate_luts(types = lut_types,
                                                lut_file = '{:s}_luts.pkl'.format(mode), 
                                                load =  (lut_exists & (force_calc==False)), 
                                                lut_path = _CCT_LUT_PATH, 
                                                wl = wl, cieobs = _CCT_LIST_OF_CIEOBS_LUTS,
                                                cspace = [_CCT_CSPACE], cspace_kwargs = [_CCT_CSPACE_KWARGS],
                                                lut_vars = _CCT_LUT[mode]['lut_vars'],
                                                verbosity = _CCT_VERBOSITY_LUT_GENERATION,
                                                lut_generator_fcn = _CCT_LUT[mode]['_generate_lut'])
        _CCT_LUT[mode]['luts'] = _lut_to_float64(_CCT_LUT[mode]['luts'])
        _copy_luts(mode, lut = _CCT_LUT) # 2015_2 -> 2006_2, 2015_10 -> 2006_10


def _add_lut_endpoints(x):
    """ Replicates endpoints of lut to avoid out-of-bounds issues """
    return np.vstack((x[:1],x,x[-1:]))


    

#------------------------------------------------------------------------------
# Other helper functions
#------------------------------------------------------------------------------

def _get_Duv_for_T_from_uvBB(u,v, uBB0, vBB0):
    """ 
    Calculate Duv from uv coordinates of estimated Tc.
    """
    # Get duv: 
    du, dv = u - uBB0, v - vBB0
    Duv = (du**2 + dv**2)**0.5 
    # return Duv*(np.sign(du*dv)*(-np.sign(dv)) +
    #         (Duv>0)*(np.sign(dv)*(du==0) - np.sign(du)*(dv==0)))
    # find sign of duv:
    theta = math.positive_arctan(du,dv,htype='deg')
    theta[theta>180] = theta[theta>180] - 360
    Duv *= (np.sign(theta))
    return Duv

def _get_Duv_for_T(u,v, T, wl, cieobs, cspace_dict, uvwbar = None, dl = None,
                   uBB = None, vBB = None):
    """ 
    Calculate Duv from T by generating a planckian and
    calculating the Euclidean distance to the point (u,v) and
    determing the sign as the v coordinate difference between 
    the test point and the planckian.
    """
    if (uBB is None)  & (vBB is None):
        
        if (uvwbar is not None) & (dl is not None):
            _,UVWBB,_,_ = _get_tristim_of_BB_BBp_BBpp(T, uvwbar, wl, dl, out='BB')
            uvBB = xyz_to_Yxy(UVWBB)[...,1:]
        else:
            BB = cri_ref(T, ref_type = ['BB'], wl3 = wl)
            xyzBB = spd_to_xyz(BB, cieobs = cieobs, relative = True)
            uvBB = cspace_dict['fwtf'](xyzBB)[...,1:]
        uBB, vBB = uvBB[...,0:1], uvBB[...,1:2]
    
    # Get duv: 
    return _get_Duv_for_T_from_uvBB(u, v, uBB, vBB)
        

# def _plot_triangular_solution(u,v,uBB,vBB,TBB,pn):
#     """
#     Make a plot of the geometry of the test (u,v) and the
#     3 points i-1, i, i+1. Helps for testing and understanding coded algorithms.
#     """
#     import luxpy as lx
    
#     plt.plot(u,v,'ro')
#     # pnl = np.hstack((pn-2,pn-1,pn,pn+1,pn+2))
#     # plt.plot(uBB[pnl],vBB[pnl],'k.-')
#     plt.plot(uBB[pn-1],vBB[pn-1],'cv')
#     plt.plot(uBB[pn+1],vBB[pn+1],'m^')
#     plt.plot(np.vstack((u,uBB[pn-1])), np.vstack((v,vBB[pn-1])), 'c')
#     plt.plot(np.vstack((u,uBB[pn+1])), np.vstack((v,vBB[pn+1])), 'm')
#     plt.plot(np.vstack((uBB[pn-1],uBB[pn+1])), np.vstack((vBB[pn-1],vBB[pn+1])), 'g')
#     # for i in range(TBB.shape[0]):
#     #     plt.text(uBB[i],vBB[i],'{:1.0f}K'.format(TBB[i,0]))
#     lx.plotSL(axh=plt.gca(),cspace='Yuv60')

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

#==============================================================================
# define original versions of various cct methods:
#==============================================================================

#------------------------------------------------------------------------------
# Mcamy, 1992:
#------------------------------------------------------------------------------
_CCT_LUT['mcamy1992'] = {'lut_vars': None, 'lut_type_def': None, 'luts':None, '_generate_lut':None}
def _xyz_to_cct_mcamy(xyzw):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) using 
    the mccamy approximation.
    
    | Only valid for approx. 3000 < T < 9000, if < 6500, error < 2 K.
    
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        
    Returns:
        :cct: 
            | ndarray of correlated color temperatures estimates
            
    References:
        1. `McCamy, Calvin S. (April 1992). 
        "Correlated color temperature as an explicit function of 
        chromaticity coordinates".
        Color Research & Application. 17 (2): 142–144.
        <http://onlinelibrary.wiley.com/doi/10.1002/col.5080170211/abstract>`_
    """
    Yxy = xyz_to_Yxy(xyzw)
    n = (Yxy[:,1]-0.3320)/(Yxy[:,2]-0.1858)
    return  np2d(-449.0*(n**3) + 3525.0*(n**2) - 6823.3*n + 5520.33).T

def xyz_to_cct_mcamy1992(xyzw, cieobs = '1931_2', wl = None, out = 'cct',
                         cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) using 
    the mccamy approximation (!!! only valid for CIE 1931 2° input !!!).
    
    | Only valid for approx. 3000 < T < 9000, if < 6500, error < 2 K 
    
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        :cieobs: 
            | '1931_2', optional
            | CMF set used to calculated xyzw. 
            | Note: since the parameter values in Mcamy's equation were optimized,
            |   using the 1931 2° CMFs, this is only valid for that CMF set.
            |   It can be changed, but will only impact the calculation of Duv and
            |   thereby causing a potential mismatch/error. Change at own discretion.
        :out: 
            | 'cct' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators when determining Duv.
            |  (!!CCT is determined using a fixed set of equations optimized for the 1931 2° CMFS!!)
        :cspace:
            | _CCT_SPACE, optional
            | Color space to do calculations in. 
            | Options: 
            |    - cspace string: 
            |        e.g. 'Yuv60' for use with luxpy.colortf()
            |    - tuple with forward (i.e. xyz_to..) [and backward (i.e. ..to_xyz)] functions 
            |      (and an optional string describing the cspace): 
            |        e.g. (forward, backward) or (forward, backward, cspace string) or (forward, cspace string) 
            |    - dict with keys: 'fwtf' (foward), 'bwtf' (backward) [, optional: 'str' (cspace string)]
            |  Note: if the backward tf is not supplied, optimization in cct_to_xyz() is done in the CIE 1976 u'v' diagram
        :cspace_kwargs:
            | _CCT_CSPACE_KWARGS, optional
            | Parameter nested dictionary for the forward and backward transforms.
        
    Returns:
        :cct: 
            | ndarray of correlated color temperatures estimates
            
    References:
        1. `McCamy, Calvin S. (April 1992). 
        "Correlated color temperature as an explicit function of 
        chromaticity coordinates".
        Color Research & Application. 17 (2): 142–144.
        <http://onlinelibrary.wiley.com/doi/10.1002/col.5080170211/abstract>`_

    """
    ccts = _xyz_to_cct_mcamy(xyzw)
    cspace_dict,_ = _process_cspace(cspace, cspace_kwargs = cspace_kwargs)
    uv = cspace_dict['fwtf'](xyzw)[:,1:]
    u,v = uv[:,0,None], uv[:,1,None]
    duvs = _get_Duv_for_T(u, v, ccts, wl, cieobs, cspace_dict)
    
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


#------------------------------------------------------------------------------
# Hernandez-Andres, 1999:
#------------------------------------------------------------------------------
_CCT_LUT['hernandez1999'] = {'lut_vars': None, 'lut_type_def': None, 'luts':None, '_generate_lut':None}
def _xyz_to_cct_HA(xyzw, verbosity = 1):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT). 
       
    | According to paper small error from 3000 - 800 000 K
    
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        
    Returns:
        :cct: 
            | ndarray of correlated color temperatures estimates
    
    References:
        1. `Hernández-Andrés, Javier; Lee, RL; Romero, J (September 20, 1999). 
        Calculating Correlated Color Temperatures Across the Entire Gamut 
        of Daylight and Skylight Chromaticities.
        Applied Optics. 38 (27), 5703–5709. P
        <https://www.osapublishing.org/ao/abstract.cfm?uri=ao-38-27-5703>`_
            
    Notes: 
        According to paper small error from 3000 - 800 000 K, but a test with 
        Planckians showed errors up to 20% around 500 000 K; 
        e>0.05 for T>200 000, e>0.1 for T>300 000, ...
    """
    if len(xyzw.shape)>2:
        raise Exception('xyz_to_cct_HA(): Input xyzw.ndim must be <= 2 !')
        
    out_of_range_code = np.nan
    xe = [0.3366, 0.3356]
    ye = [0.1735, 0.1691]
    A0 = [-949.86315, 36284.48953]
    A1 = [6253.80338, 0.00228]
    t1 = [0.92159, 0.07861]
    A2 = [28.70599, 5.4535*1e-36]
    t2 = [0.20039, 0.01543]
    A3 = [0.00004, 0.0]
    t3 = [0.07125,1.0]
    cct_ranges = np.array([[3000.0,50000.0],[50000.0,800000.0]])
    
    Yxy = xyz_to_Yxy(xyzw)
    CCT = np.ones((1,Yxy.shape[0]))*out_of_range_code
    for i in range(2):
        n = (Yxy[:,1]-xe[i])/(Yxy[:,2]-ye[i])
        CCT_i = np2d(np.array(A0[i] + A1[i]*np.exp(np.divide(-n,t1[i])) + A2[i]*np.exp(np.divide(-n,t2[i])) + A3[i]*np.exp(np.divide(-n,t3[i]))))
        p = (CCT_i >= (1.0-0.05*(i == 0))*cct_ranges[i][0]) & (CCT_i < (1.0+0.05*(i == 0))*cct_ranges[i][1])
        CCT[p] = CCT_i[p].copy()
        p = (CCT_i < (1.0-0.05)*cct_ranges[0][0]) #smaller than smallest valid CCT value
        CCT[p] = -1
   
    if ((np.isnan(CCT.sum()) == True) | (np.any(CCT == -1))) & (verbosity == 1):
        print("Warning: xyz_to_cct_HA(): one or more CCTs out of range! --> (CCT < 3 kK,  CCT >800 kK) coded as (-1, NaN) 's")
    return CCT.T

def xyz_to_cct_hernandez1999(xyzw, cieobs = '1931_2', wl = None, out = 'cct',
                             cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) using 
    the mccamy approximation (!!! only valid for CIE 1931 2° input !!!).
    
    | According to paper small error from 3000 - 800 000 K
    
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        :cieobs: 
            | '1931_2', optional
            | CMF set used to calculated xyzw. 
            | Note: since the parameter values in the HA equations were optimized,
            |   using the 1931 2° CMFs, this is only valid for that CMF set.
            |   It can be changed, but will only impact the calculation of Duv and
            |   thereby causing a potential mismatch/error. Change at own discretion.
        :out: 
            | 'cct' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators when determining Duv.
            |  (!!CCT is determined using a fixed set of equations optimized for the 1931 2° CMFS!!)
        :cspace:
            | _CCT_SPACE, optional
            | Color space to do calculations in. 
            | Options: 
            |    - cspace string: 
            |        e.g. 'Yuv60' for use with luxpy.colortf()
            |    - tuple with forward (i.e. xyz_to..) [and backward (i.e. ..to_xyz)] functions 
            |      (and an optional string describing the cspace): 
            |        e.g. (forward, backward) or (forward, backward, cspace string) or (forward, cspace string) 
            |    - dict with keys: 'fwtf' (foward), 'bwtf' (backward) [, optional: 'str' (cspace string)]
            |  Note: if the backward tf is not supplied, optimization in cct_to_xyz() is done in the CIE 1976 u'v' diagram
        :cspace_kwargs:
            | _CCT_CSPACE_KWARGS, optional
            | Parameter nested dictionary for the forward and backward transforms.
        
    Returns:
        :cct: 
            | ndarray of correlated color temperatures estimates
            
    References:
        1. `Hernández-Andrés, Javier; Lee, RL; Romero, J (September 20, 1999). 
        Calculating Correlated Color Temperatures Across the Entire Gamut 
        of Daylight and Skylight Chromaticities.
        Applied Optics. 38 (27), 5703–5709. P
        <https://www.osapublishing.org/ao/abstract.cfm?uri=ao-38-27-5703>`_
    """
    ccts = _xyz_to_cct_HA(xyzw)
    cspace_dict,_ = _process_cspace(cspace, cspace_kwargs = cspace_kwargs)
    uv = cspace_dict['fwtf'](xyzw)[:,1:]
    u,v = uv[:,0,None], uv[:,1,None]
    duvs = _get_Duv_for_T(u, v, ccts, wl, cieobs, cspace_dict)
    
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

#------------------------------------------------------------------------------
# Newton-Raphson estimator (cfr. Li, 2016):
#------------------------------------------------------------------------------       
def _get_uv_uvp_uvpp(T, uvwbar, wl, dl, out = 'BB,BBp,BBpp'):
    """ 
    Get the (u,v), (u',v') and (u",v") coordinates of one or more Planckians
    with specified Tc. uvwbar (no wavelengths on row0, these are supplied seperately
    in wl, with wavelength spacing in dl) is the cmf set corresponding to the tristimulus values
    of the chosen chromaticity diagram or color space to do the CCT calculations in.
    
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
    
    return T, u, v, up, vp, upp, vpp


def _get_newton_raphson_estimated_Tc(u, v, T0, wl = None, atol = 0.1, rtol = 1e-5,
                                     cieobs = None, xyzbar = None, uvwbar = None,
                                     cspace_dict = None, max_iter = _CCT_MAX_ITER,
                                     fast_duv = _CCT_FAST_DUV):
    """
    Get an estimate of the CCT using the Newton-Raphson method (as specified in 
    Li et al., 2016). (u,v) are the test coordinates. T0 is a first estimate of the Tc.
    atol and rtol are the absolute and relative tolerance values that are aimed at (if
    possible the error on the estimation should smaller than or close to these values,
    once one is achieved the algorithm stops). wl contains the wavelengths of the 
    Planckians, cieobs is the CIE cmfs set to be used (or use xyzbar; at least one
    must be given). uvwbar the already converted cmf set. If this one is not none
    than any input in cieobs or xyzbar is ignored. cspace_dict must be supplied
    when uvwbar is None (needed for color space conversion!). Max-iter specifies
    the maximum number of iterations (avoid potential infinite loops or cut the
    optimization short). When fast_duv is True (dfeault) a faster method is used, but this
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
        if (cspace_dict is not None):
            if (xyzbar is None) & (cieobs is not None):
                xyzbar, wl, dl = _get_xyzbar_wl_dl(cieobs, wl)
            elif (xyzbar is None) & (cieobs is None):
                raise Exception('Must supply xyzbar or cieobs or uvwbar !!!')
            uvwbar = _convert_xyzbar_to_uvwbar(xyzbar, cspace_dict)
        else:
            raise Exception('Must supply cspace_dict if uvwbar is None. How to convert xyzbar if not supplied ?')

    if uvwbar.shape[0] == 4:
        wl = uvwbar[0]
        dl = getwld(wl)
        uvwbar = uvwbar[1:]
    else:
        dl = getwld(wl)

    
    i = 0
    T = T0
    while True & (i <= max_iter):
           
        # Get (u,v), (u',v'), (u",v"):
        _, uBB, vBB, upBB, vpBB, uppBB, vppBB = _get_uv_uvp_uvpp(T, uvwbar, wl, dl, out = 'BB,BBp,BBpp')

        # Calculate DT (ratio of f' and abs(f"):
        du, dv = (u - uBB), (v - vBB) # pre-calculate for speed
        DT = -(du*upBB + dv*vpBB) / np.abs((upBB**2)-du*uppBB + (vpBB**2)-dv*vppBB)

        DT[DT>T] = _CCT_MIN # avoid convergence problems
        T = T - DT 

        if (np.abs(DT) < atol).all() | (np.abs(DT)/T < rtol).all():
            break
        i+=1
        
    # get Duv:
    if ~(fast_duv & (np.abs(DT)<=1.0).all()):
        _, uBB, vBB, _, _, _, _ = _get_uv_uvp_uvpp(T, uvwbar, wl, dl, out = 'BB') # update one last time
    
    Duv = _get_Duv_for_T_from_uvBB(u,v, uBB, vBB)
    return T, Duv

#------------------------------------------------------------------------------
# Cascade lut estimator:
#------------------------------------------------------------------------------       
def _get_loop_i_lut_for_cascading_lut(Tx, TBB_m1, TBB_p1, out_of_lut,
                                      atol, rtol, 
                                      cascade_idx, lut_char, lut_resolution_reduction_factor,
                                      mode, luts_dict, cieobs, wl, cspace_str, cspace_dict,
                                      ignore_wl_diff,
                                      lut_generator_fcn = _generate_lut,
                                      lut_generator_kwargs = {},
                                      **kwargs):
    """
    Get a new updated lut with reduced min-max range for a cascading lut calculation.
    """

    # cl cannot recover from out-of-lut, so if all are out-of-lut, no use continuing!
    if (out_of_lut is not None) and (out_of_lut.all()): 
        return -1, None # ,None because expected output (from _generate_lut is 2):

    # get overall min, max Ts over all xyzw test points:
    Ts_m1p1 =  np.hstack((TBB_m1,TBB_p1))  
    Ts_min, Ts_max = Ts_m1p1.min(axis=-1),Ts_m1p1.max(axis=-1)
    
    dTs = np.abs(Ts_max - Ts_min)

    if (dTs<= atol).all() | (np.abs(dTs/Tx) <= rtol).all():
        Tx =  ((Ts_min + Ts_max)/2)[:,None] 
        return -1, Tx 
    else:
        
        lut_int = lut_char[0][2]

        if np.isnan(lut_int): 
            lut_cl = 1e6/np.linspace(1e6/Ts_max,1e6/Ts_min,lut_resolution_reduction_factor)
        else:
            lut_unit = lut_char[0][3]
            lut_int = lut_int/lut_resolution_reduction_factor**(cascade_idx + 1)
            
            # dTs = np.abs((Ts_max-Ts_min)) if ('-1' not in lut_unit) else np.abs((1e6/Ts_max - 1e6/Ts_min))
            # if (dTs < lut_int).any(): 
            #     lut_int = lut_int*np.ones_like(dTs) 
            #     lut_int[dTs < lut_int] = dTs[dTs< lut_int]/lut_resolution_reduction_factor

            lut_cl = [Ts_min,Ts_max,lut_int,lut_unit] if ('-1' not in lut_unit) else [1e6/Ts_max,1e6/Ts_min,lut_int,lut_unit]
            
        
        #return (lut, lut_kwargs) tuple:
        lut_vars = lut_generator_kwargs.pop('lut_vars') if 'lut_vars' in lut_generator_kwargs else _CCT_LUT[mode]['lut_vars']
        return lut_generator_fcn(lut_cl, seamless_stitch = True, 
                                 fallback_unit = _CCT_FALLBACK_UNIT, 
                                 fallback_n = _CCT_FALLBACK_N,
                                 resample_ndarray = False, 
                                 luts_dict = luts_dict, cieobs = cieobs, 
                                 lut_type_def = _CCT_LUT[mode]['lut_type_def'],
                                 cspace_str = cspace_str, wl = wl, cspace = cspace_dict, 
                                 cspace_kwargs = None, ignore_unequal_wl = ignore_wl_diff, 
                                 lut_vars = lut_vars,
                                 **lut_generator_kwargs)

def _get_cascading_lut_Tx(mode, u, v, lut, lut_n_cols, lut_char, lut_resolution_reduction_factor,
                          luts_dict, cieobs, wl, cspace_str, cspace_dict, ignore_wl_diff,
                          max_iter = _CCT_MAX_ITER, mode_kwargs = {}, atol = 0.1, rtol = 1e-5, 
                          Tx = None, Duvx = None, out_of_lut = None, TBB_l = None, TBB_r = None,
                          **kwargs):
    """
    Determine Tx using a specified mode from u,v input using a cascading lut, 
    i.e. lut progressively decreasing in min-max range, zooming-in on the 
    'true' Tx value. lut_n_cols should specify the number of columns in the lut
    for the specified mode. _uv_to_Tx_mode is a function that calculates the Tx
    for the specific mode from u,v. It should have the following interface:
    _uv_to_Tx_mode(u,v,lut,lut_n_cols, ns = 0, out_of_lut = None, 
                   Tx = None, out_of_lut = None, TBB_l = None, TBB_r = None)
    and return the Tx,Duvx, out_of_lut and a tuple with Tleft (TBB_m1), Tright (TBB_p1). 
    Duvx can be None if method doesn't naturally provide an estimate.
    Skips first calculation of Tx when (Tx0, out_of_lut, TBB_l, TBB_r) are None
    (i.e. already calculated).
    """
    # cascading lut:
    cascade_i = 0
    lut_i = lut # cascading lut will be updated later in the while loop
    _uv_to_Tx_mode = _CCT_UV_TO_TX_FCNS[mode]
    lut_generator_fcn = _CCT_LUT[mode]['_generate_lut']
    lut_generator_kwargs = copy.copy(mode_kwargs[mode])
    
    # some mode specific processing to prepare lut_generator_kwargs:
    lut_generator_kwargs.update({'f_corr':1}) # only required when mode = 'ohno2014'
    if 'wl' in lut_generator_kwargs: lut_generator_kwargs.pop('wl') # for mode == 'zhang2019': 'wl' is also part of this dict!
    # if 'lut_vars' in lut_generator_kwargs: lut_generator_kwargs.pop('lut_vars') # for mode == 'zhang2019': 'wl' is also part of this dict!
    
    while True & (cascade_i < max_iter):
        
        # needed to get correct columns from updated lut_i:
        N = lut_i.shape[-1]//lut_n_cols
        ns = lut_n_cols #np.arange(0,N*lut_n_cols,lut_n_cols,dtype=int)

        # get Tx estimate, out_of_lut boolean array, and (TBB_m1, TBB_p1 or equivalent):
        if ((Tx is None) & (out_of_lut is None) & (TBB_l is None) & (TBB_r is None)) | (cascade_i > 0):
            Tx, Duvx, out_of_lut, (TBB_l,TBB_r) = _uv_to_Tx_mode(u, v, lut_i, lut_n_cols, 
                                                                 ns = ns, out_of_lut = out_of_lut,
                                                                 **{**mode_kwargs[mode],**{'max_iter':1}}) # cl takes over, so max_iter should be 1

        if cascade_i == 0: Tx0 = Tx.copy() # keep copy of first estimate

        # Update lut for next cascade (ie decrease min-max range):
        tmp = _get_loop_i_lut_for_cascading_lut(Tx, TBB_l, TBB_r, out_of_lut,
                                                atol, rtol, 
                                                cascade_i, lut_char, lut_resolution_reduction_factor,
                                                mode, luts_dict, cieobs, wl, cspace_str, cspace_dict,
                                                ignore_wl_diff,
                                                lut_generator_fcn = lut_generator_fcn,
                                                lut_generator_kwargs = lut_generator_kwargs,
                                                )

        if (tmp[0] is -1): 
            if tmp[1] is not None: Tx = tmp[1]
            break
        else:
            lut_i = tmp[0]
            lut_kwargs = tmp[1]

        cascade_i+=1 # to stop cascade loop after max_iter iterations
    
    Tx[out_of_lut] = Tx0[out_of_lut] # restore originals as cl_lut might have messed up these Tx   

    return Tx, Duvx


#------------------------------------------------------------------------------
# General _xyz_to_cct structure:
#------------------------------------------------------------------------------
def _xyz_to_cct(xyzw, mode, is_uv_input = False, cieobs = _CIEOBS, wl = None, out = 'cct',
                lut = None, luts_dict = None, ignore_wl_diff = False,
                force_tolerance = True, tol_method = 'newton-raphson', atol = 0.1, rtol = 1e-5, 
                max_iter = _CCT_MAX_ITER, force_au = False, 
                split_calculation_at_N = _CCT_SPLIT_CALC_AT_N, lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
                cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                duv_parabolic_threshold = 0.002, 
                first_guess_mode = 'robertson1968',
                use_fast_duv = _CCT_FAST_DUV,
                **kwargs):
    """ 
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv (distance above (>0) or below (<0) the Planckian locus) using a number
    of modes (methods). 
    
    General fucntion for calculation of 'robertson1968', 'ohno2014', 'li2016' and 'zhang2019'
    (for info on arguments, see docstring of xyz_to_cct)
    """
    
    # allow for seperate max_iter for mode and follow-up tol_method:
    if not isinstance(max_iter,(list,tuple)):
        max_iter = (max_iter,max_iter) #(mode, tol_method)
    
    # Deal with mode == 'li2016': 
    if 'li2016' in mode: 
        if ':' in mode: # for 'li2016:first_guess_mode' format
            p = mode.index(':')
            first_guess_mode = mode[p+1:]
        mode = first_guess_mode # overwrite mode with first_guess_mode as this is the one to use together with LUTs
        force_tolerance = True # must be True, otherwise li2016 == first_guess_mode output!
        tol_method = 'newton-raphson' # by default, no need for addditional lut cascading   
    
    # Process cspace-parameters:
    cspace_dict, cspace_str = _process_cspace(cspace, cspace_kwargs)
    
    # Get chromaticity coordinates u,v from xyzw:
    uvw = cspace_dict['fwtf'](xyzw)[:,1:3]  if is_uv_input == False else xyzw[:,0:2] # xyz contained uv !!! (needed to efficiently determine f_corr)
    
    # pre-calculate wl,dl,uvwbar for later use (will also determine wl if None !):
    xyzbar, wl, dl = _get_xyzbar_wl_dl(cieobs, wl)
    uvwbar = _convert_xyzbar_to_uvwbar(xyzbar, cspace_dict)
    
    # Get or generate requested lut
    # (if wl doesn't match those in _CCT_LUT[mode] , 
    # a new recalculated lut will be generated):
    if (luts_dict is None): 
        luts_dict = _CCT_LUT[mode]['luts']

    lut, lut_kwargs = _get_lut(lut, 
                               fallback_unit = _CCT_FALLBACK_UNIT, 
                               fallback_n = _CCT_FALLBACK_N,
                               resample_ndarray = False, 
                               luts_dict = luts_dict, cieobs = cieobs, 
                               lut_type_def = _CCT_LUT[mode]['lut_type_def'],
                               cspace_str = cspace_str, wl = wl, cspace = cspace_dict, 
                               cspace_kwargs = None, ignore_unequal_wl = ignore_wl_diff, 
                               lut_generator_fcn = _CCT_LUT[mode]['_generate_lut'],
                               lut_vars = _CCT_LUT[mode]['lut_vars'])
    
    # Prepare some parameters for forced tolerance:
    if force_tolerance: 
        if (tol_method == 'newton-raphson') | (tol_method == 'nr'):
            pass
        elif (tol_method == 'cascading-lut') | (tol_method == 'cl'): 
            lut_char = _get_lut_characteristics(lut, force_au = force_au)
        else:
            raise Exception('Tolerance method = {:s} not implemented.'.format(tol_method))
    
    lut_n_cols = lut.shape[-1] # store now, as this will change later
    
    # prepare split of input data to speed up calculation:
    n = xyzw.shape[0]
    ccts = np.zeros((n,1))
    duvs = np.zeros((n,1))
    n_ii = split_calculation_at_N if split_calculation_at_N is not None else n
    N_ii = n//n_ii + 1*((n%n_ii)>0)

    lut_vars = _CCT_LUT[mode]['lut_vars']  if 'lut_vars' not in kwargs else kwargs['lut_vars'] 
    # prepare mode_kwargs (i.e. extra kwargs for _uv_to_Tx_mode() required by specific modes):
    mode_kwargs = {'robertson1968': {},
                   'zhang2019' : {'uvwbar' : uvwbar, 'wl' : wl, 'dl' : dl, 'lut_vars' : lut_vars,
                                  'max_iter' : max_iter[0], 'atol' : atol, 'rtol' : rtol},
                   'ohno2014' : {**lut_kwargs, **{'duv_parabolic_threshold' : duv_parabolic_threshold}},
                   'fibonacci' : {'uvwbar' : uvwbar, 'wl' : wl, 'dl' : dl, 'lut_vars' : lut_vars,
                                  'max_iter' : max_iter[0], 'atol' : atol, 'rtol' : rtol},
                   'none' : {'uvwbar' : uvwbar, 'wl' : wl, 'dl' : dl, 'lut_vars' : lut_vars,
                             'max_iter' : max_iter, 'atol' : atol, 'rtol' : rtol}
                   } 

    # loop of splitted data:
    for ii in range(N_ii):
        out_of_lut = None

        # get data for split ii:
        uv = uvw[n_ii*ii:n_ii*ii+n_ii] if (ii < (N_ii-1)) else uvw[n_ii*ii:]
        u, v = uv[:,0,None], uv[:,1,None]
    
        # get Tx estimate, out_of_lut boolean array, and (TBB_m1, TBB_p1 or equivalent):
        Tx, Duvx, out_of_lut, (TBB_l, TBB_r) = _CCT_UV_TO_TX_FCNS[mode](u, v, lut, lut_n_cols, 
                                                                        ns = lut_n_cols, 
                                                                        out_of_lut = out_of_lut,
                                                                        **mode_kwargs[mode])  


        if force_tolerance:
            if (tol_method == 'cascading-lut') | (tol_method == 'cl'): 

                Tx,Duvx = _get_cascading_lut_Tx(mode, u, v, lut, lut_n_cols, lut_char, lut_resolution_reduction_factor,
                                                luts_dict, cieobs, wl, cspace_str, cspace_dict, ignore_wl_diff, 
                                                max_iter = max_iter[1], mode_kwargs = mode_kwargs, atol = atol, rtol = rtol,
                                                Tx = Tx, Duvx = Duvx, out_of_lut = out_of_lut, TBB_l = TBB_l, TBB_r = TBB_r,
                                                )

   
            elif (tol_method == 'newton-raphson') | (tol_method == 'nr'):

                Tx, Duvx = _get_newton_raphson_estimated_Tc(u, v, Tx, wl = wl, uvwbar = uvwbar,
                                                            atol = atol, rtol = rtol, max_iter = max_iter[1],
                                                            fast_duv = use_fast_duv)           

        if Duvx is None:
            Duvx = _get_Duv_for_T(u,v, Tx, wl, cieobs, cspace_dict, 
                                  uvwbar = uvwbar, dl = dl)

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



#------------------------------------------------------------------------------
# Robertson 1968:
#------------------------------------------------------------------------------
_CCT_LUT['robertson1968'] = {}
_CCT_LUT['robertson1968']['lut_type_def'] = ((_CCT_LUT_MIN, _CCT_LUT_MAX, 0.50, '%'),) # default LUT, must be in all_modes
_CCT_LUT['robertson1968']['lut_vars'] = ['T','uv','iso-T-slope']
_CCT_LUT['robertson1968']['_generate_lut'] = _generate_lut 

def _uv_to_Tx_robertson1968(u, v, lut, lut_n_cols, ns = 4, out_of_lut = None,
                            fast_duv = _CCT_FAST_DUV, **kwargs):
    """ 
    Calculate Tx from u,v and lut using Robertson 1968.
    (lut_n_cols specifies the number of columns in the lut for 'robertson1968')
    """
    Duvx = None 
    idx_sources = np.arange(u.shape[0],dtype=np.int32)
    
    # get uBB, vBB, mBB from lut:
    TBB, uBB, vBB, mBB  = lut[:,0::lut_n_cols], lut[:,1::lut_n_cols], lut[:,2::lut_n_cols], lut[:,-1::lut_n_cols].copy()
    mBB[mBB>0] = -mBB[mBB>0]
    
    # calculate distances to coordinates in lut (Eq. 4 in Robertson, 1968):
    di = ((v.T - vBB) - mBB * (u.T - uBB)) / ((1 + mBB**2)**(0.5))
    pn = (((v.T - vBB)**2 + (u.T - uBB)**2)).argmin(axis=0)
    

    # Deal with endpoints of lut + create intermediate variables 
    # to save memory:
    pn, out_of_lut = _deal_with_lut_end_points(pn, TBB, out_of_lut)
    TBB_m1, TBB_0, TBB_p1 = _get_pns_from_x(TBB, pn, i = idx_sources, m0p = 'm0p')
    # uBB_0, uBB_p1 = _get_pns_from_x(uBB, pn, m0p = '0p')
    # vBB_0, vBB_p1 = _get_pns_from_x(vBB, pn, m0p = '0p')

    di_0, di_p1 = _get_pns_from_x(di, pn, i = idx_sources, m0p = '0p')

    # Estimate Tc (Robertson, 1968): 
    slope = (di_0/((di_0 - di_p1) + _CCT_AVOID_ZERO_DIV))
    Tx = ((((1/TBB_0) + slope * ((1/TBB_p1) - (1/TBB_0)))**(-1)))#".copy()

    if fast_duv:
        uBB_0, uBB_p1 = _get_pns_from_x(uBB, pn, i = idx_sources, m0p = '0p')
        vBB_0, vBB_p1 = _get_pns_from_x(vBB, pn, i = idx_sources, m0p = '0p')
        # ux = ((((1/uBB_0)+slope*((1/uBB_p1) - (1/uBB_0)))**(-1)))#.copy()
        # vx = ((((1/vBB_0)+slope*((1/vBB_p1) - (1/vBB_0)))**(-1)))#.copy()
        ux = (uBB_0 + slope * (uBB_p1 - uBB_0))#.copy()
        vx = (vBB_0 + slope * (vBB_p1 - vBB_0))#.copy()
        Duvx = _get_Duv_for_T_from_uvBB(u, v, ux, vx)
    return Tx, Duvx, out_of_lut, (TBB_m1, TBB_p1)

_CCT_UV_TO_TX_FCNS['robertson1968'] = _uv_to_Tx_robertson1968

def xyz_to_cct_robertson1968(xyzw, cieobs = _CIEOBS, out = 'cct', is_uv_input = False, wl = None, 
                            atol = 0.1, rtol = 1e-5, force_tolerance = True, tol_method = 'newton-raphson', 
                            lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
                            split_calculation_at_N = _CCT_SPLIT_CALC_AT_N, max_iter = _CCT_MAX_ITER,
                            cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                            lut = None, luts_dict = None, ignore_wl_diff = False,
                            use_fast_duv = _CCT_FAST_DUV,
                            **kwargs):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv(distance above (> 0) or below ( < 0) the Planckian locus) using  
    Robertson's 1968 search method.
        
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set used to calculated xyzw.
        :out: 
            | 'cct' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)
        :is_uv_input:
            | False, optional
            | If True: xyzw contain uv input data, not xyz data!
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
            | If None: use same wavelengths as CMFs in :cieobs:.
        :rtol: 
            | 1e-5, float, optional
            | Stop search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop search when cct a absolute tolerance (K) is reached.
        :force_tolerance:
            | True, optional
            | If False: search only using the list of CCTs in the used lut. 
            |           Only one loop of the full algorithm is performed. 
            |           Accuracy depends on CCT of test source and the location
            |           and spacing of the CCTs in the list.
            | If True:  search will use adjacent CCTs to test source to create a new LUT,
            |           (repeat the algoritm at higher resolution, progessively zooming in
            |            toward the ground-truth) for tol_method == 'cl'; when 
            |           tol_method == 'nr' a newton-raphson method is used.
            |           Because the CCT for multiple source is calculated in one go,
            |           the atol and rtol values have to be met for all! 
        :tol_method:
            | 'newton-raphson', optional
            | (Additional) method to try and achieve set tolerances. 
            | Options: 
            | - 'cl', 'cascading-lut': use increasingly higher CCT-resolution
            |       to 'zoom-in' on the ground-truth.
            | - 'nr', 'newton-raphson': use the method as described in Li, 2016.
        :lut_resolution_reduction_factor:
            | _CCT_LUT_RESOLUTION_REDUCTION_FACTOR, optional
            | Number of times the interval spanned by the adjacent Tc in a search or lut
            | method is downsampled (the search process will then start again)
        :max_iter:
            | _CCT_MAX_ITER, optional
            | Maximum number of iterations used by the cascading-lut or newton-raphson methods.
        :split_calculation_at_N:
            | _CCT_SPLIT_CALC_AT_N, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :lut:
            | None, optional
            | Look-Up-Table with Ti, u,v,u',v',u",v",slope values of Planckians. 
            | Options:
            |  - None: defaults to the lut specified in _CCT_LUT['robertson1968']['lut_type_def'].
            |  - list (lut,lut_kwargs): use this pre-calculated lut 
            |       (add additional kwargs for the lut_generator_fcn(), defaults to None if omitted)
            |  - tuple: must be key (label) in :luts_dict: (pre-calculated dict of luts),
            |           if not: then a new lut will be generated from scratch using the info in the tuple.
            |  - str: must be key (label) in :luts_dict: (pre-calculated dict of luts)
            |  - ndarray [Nx1]: list of luts for which to generate a lut
            |  - ndarray [Nxn] with n>3: pre-calculated lut (last col must contain slope of the isotemperature lines).
        :luts_dict:
            | None, optional
            | Dictionary of pre-calculated luts for various cspaces and cmf sets.
            |  Must have structure luts_dict[cspace][cieobs][lut_label] with the
            |   lut part of a two-element list [lut, lut_kwargs]. It must contain
            |   at the top-level a key 'wl' containing the wavelengths of the 
            |   Planckians used to generate the luts in this dictionary.
            | If None: luts_dict defaults to _CCT_LUT['robertson1968']['luts']   
        :cspace:
            | _CCT_SPACE, optional
            | Color space to do calculations in. 
            | Options: 
            |    - cspace string: 
            |        e.g. 'Yuv60' for use with luxpy.colortf()
            |    - tuple with forward (i.e. xyz_to..) [and backward (i.e. ..to_xyz)] functions 
            |      (and an optional string describing the cspace): 
            |        e.g. (forward, backward) or (forward, backward, cspace string) or (forward, cspace string) 
            |    - dict with keys: 'fwtf' (foward), 'bwtf' (backward) [, optional: 'str' (cspace string)]
            |  Note: if the backward tf is not supplied, optimization in cct_to_xyz() is done in the CIE 1976 u'v' diagram
        :cspace_kwargs:
            | _CCT_CSPACE_KWARGS, optional
            | Parameter nested dictionary for the forward and backward transforms.
        :ignore_wl_diff:
            | False, optional
            | When getting a lut from the dictionary, if differences are
            | detected in the wavelengts of the lut and the ones used to calculate any
            | plankcians then a new lut should be generated. Seting this to True ignores
            | these differences and proceeds anyway.
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
         
        2. `Li, C., Cui, G., Melgosa, M., Ruan, X., Zhang, Y., Ma, L., Xiao, K., & Luo, M. R. (2016).
        Accurate method for computing correlated color temperature. 
        Optics Express, 24(13), 14066–14078. 
        <https://doi.org/10.1364/OE.24.014066>`_

    """
    return _xyz_to_cct(xyzw, mode = 'robertson1968',
                       cieobs = cieobs, out = out, wl = wl, is_uv_input = is_uv_input, 
                       cspace = cspace, cspace_kwargs = cspace_kwargs,
                       atol = atol, rtol = rtol, force_tolerance = force_tolerance,
                       tol_method = tol_method, max_iter = max_iter,  
                       lut_resolution_reduction_factor = lut_resolution_reduction_factor,
                       split_calculation_at_N = split_calculation_at_N, 
                       lut = lut, luts_dict = luts_dict, 
                       ignore_wl_diff = ignore_wl_diff, 
                       use_fast_duv = use_fast_duv,
                       **kwargs)

# pre-generate / load from disk / load from github some LUTs for Robertson1968:
_initialize_lut(mode = 'robertson1968',lut_types = _unique_types([_CCT_LUT['robertson1968']['lut_type_def'],((10,100,10,'K-1'),(100,625,25,'K-1'),True)] + _CCT_SHARED_LUT_TYPES))


#------------------------------------------------------------------------------
# Zhang 2019:
#------------------------------------------------------------------------------
_CCT_LUT['zhang2019'] = {} 
_CCT_LUT['zhang2019']['lut_type_def'] = ((1,1,1,'K-1'),(25,1025,25,'K-1'),False)
_CCT_LUT['zhang2019']['lut_vars'] = ['T','uv'] # use a lut with 'uv', without it the speed is not much different and ability to more finely tuned detect out-of-luts is gone
_CCT_LUT['zhang2019']['_generate_lut'] = _generate_lut 


def _uv_to_Tx_zhang2019(u, v, lut, lut_n_cols, ns = 0, out_of_lut = None, 
                        max_iter = _CCT_MAX_ITER, uvwbar = None, wl = None, dl = None, 
                        atol = 0.1, rtol = 1e-5, lut_vars = ['T','uv'],
                        **kwargs):
    """ 
    Calculate Tx from u,v and lut using Zhang 2019.
    """
    s = (5.0**0.5 - 1.0)/2.0
    
    # get uBB, vBB from lut:
    TBB = lut[:,0::lut_n_cols] 
    idx_sources = np.arange(u.shape[0],dtype=np.int32)
    if 'uv' in lut_vars:
        uBB, vBB = lut[:,1::lut_n_cols], lut[:,2::lut_n_cols]

        # calculate distances to coordinates in lut and find minimum:
        pn = (((v.T - vBB)**2 + (u.T - uBB)**2)).argmin(axis=0)
    
        # Deal with endpoints of lut + create intermediate variables 
        # to save memory:
        pn, out_of_lut = _deal_with_lut_end_points(pn, TBB, out_of_lut = out_of_lut)
      
        TBB_m1, TBB_p1 = _get_pns_from_x(TBB, pn, i = idx_sources, m0p = 'mp')
    
    else:
        TBB_m1,TBB_p1 = TBB[:1],TBB[-1:]

        
    # get RTm-1 (RTl) and RTm+1 (RTr):
    RTl = 1e6/TBB_m1
    RTr = 1e6/TBB_p1

    # calculate RTa, RTb:
    RTa = RTl + (1.0 - s) * (RTr - RTl)
    RTb = RTl + s * (RTr - RTl) 
    
    Tx_a, Tx_b = 1e6/RTa, 1e6/RTb
    # # RTx = ((RTa+RTb)/2)
    # # _plot_triangular_solution(u,v,uBB,vBB,TBB,pn)
    
      
    j = 0
    while True & (j < max_iter): # loop part of zhang optimization process
        
        # calculate BBa BBb:
        # BBab = cri_ref(np.vstack([cct_to_mired(RTa), cct_to_mired(RTx), cct_to_mired(RTb)]), ref_type = ['BB'], wl3 = wl)
        # BBab = cri_ref(np.vstack([cct_to_mired(RTa), cct_to_mired(RTb)]), ref_type = ['BB'], wl3 = wl)
        if (uvwbar is not None) & (wl is not None) & (dl is not None):
            TBB = np.vstack([Tx_a, Tx_b])
            _,UVWBBab,_,_ = _get_tristim_of_BB_BBp_BBpp(TBB, uvwbar, wl, dl, out='BB')
        else:
            raise Exception('uvwbar, wl & dl must all be not None !!!')
        
        # calculate xyzBBab:
        # xyzBBab = spd_to_xyz(BBab, cieobs = cieobs, relative = True)
    
        # get cspace coordinates of BB and input xyz:
        # uvBBab_ = cspace_dict['fwtf'](XYZBBab)[...,1:]
        uvBBab = xyz_to_Yxy(UVWBBab)[...,1:]

        # N = uvBBab.shape[0]//3 
        # uBBa, vBBa = uvBBab[:N,0:1], uvBBab[:N,1:2]
        # uBBx, vBBx = uvBBab[N:2*N,0:1], uvBBab[N:2*N,1:2]
        # uBBb, vBBb = uvBBab[2*N:,0:1], uvBBab[2*N:,1:2]
       
        N = uvBBab.shape[0]//2 
        # uBBa, vBBa = uvBBab[:N,0:1], uvBBab[:N,1:2]
        # uBBb, vBBb = uvBBab[N:,0:1], uvBBab[N:,1:2]
        
        # find distance in UCD of BBab to input:
        DEuv = ((uvBBab[...,0:1] - u.T)**2 + (uvBBab[...,1:2] - v.T)**2) # no need for **0.5
        DEuv_a, DEuv_b = DEuv[:N], DEuv[N:] 

        c = (DEuv_a < DEuv_b)[:,0]
        
        # when DEuv_a < DEuv_b:
        RTr[c] = RTb[c]
        RTb[c] = RTa[c]
        #DEuv_b[c] = DEuv_a[c]
        RTa[c] = (RTl[c] + (1.0 - s) * (RTr[c] - RTl[c]))
        
        # when DEuv_a >= DEuv_b:
        RTl[~c] = RTa[~c]
        RTa[~c] = RTb[~c]
        #DEuv_a[~c] = DEuv_b[~c]
        RTb[~c] = (RTl[~c] + s * (RTr[~c] - RTl[~c]))
        
        # Calculate CCTs from RTa and RTb:
        Tx_a, Tx_b = 1e6/RTa, 1e6/RTb
        Tx = 1e6/((RTa+RTb)/2)
        dTx = np.abs(Tx_a - Tx_b)
        if (((dTx <= atol).all() | ((dTx/Tx) <= rtol).all())):
            break
        j+=1

    # uBB = np.vstack((uBBa,uBBx,uBBb))
    # vBB = np.vstack((vBBa,vBBx,vBBb))
    # TBB = np.vstack((Tx_a,Tx,Tx_b))
    # pn = np.array([1])
    # _plot_triangular_solution(u,v,uBB,vBB,TBB,pn)
    
    return Tx, None, out_of_lut, (1e6/RTl,1e6/RTr)

_CCT_UV_TO_TX_FCNS['zhang2019'] = _uv_to_Tx_zhang2019

def xyz_to_cct_zhang2019(xyzw, cieobs = _CIEOBS, out = 'cct', is_uv_input = False, wl = None, 
                        atol = 0.1, rtol = 1e-5, force_tolerance = True, tol_method = 'newton-raphson', 
                        lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
                        split_calculation_at_N = _CCT_SPLIT_CALC_AT_N, max_iter = _CCT_MAX_ITER,
                        cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                        lut = None, luts_dict = None, ignore_wl_diff = False,
                        use_fast_duv = _CCT_FAST_DUV,
                        **kwargs):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv(distance above (> 0) or below ( < 0) the Planckian locus) using the 
    golden-ratio search method described in Zhang et al. (2019).
        
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set used to calculated xyzw.
        :out: 
            | 'cct' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)
        :is_uv_input:
            | False, optional
            | If True: xyzw contain uv input data, not xyz data!
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
            | If None: use same wavelengths as CMFs in :cieobs:.
        :rtol: 
            | 1e-5, float, optional
            | Stop search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop search when cct a absolute tolerance (K) is reached.
        :force_tolerance:
            | True, optional
            | If False: search only using the list of CCTs in the used lut. 
            |           Only one loop of the full algorithm is performed. 
            |           Accuracy depends on CCT of test source and the location
            |           and spacing of the CCTs in the list.
            | If True:  search will use adjacent CCTs to test source to create a new LUT,
            |           (repeat the algoritm at higher resolution, progessively zooming in
            |            toward the ground-truth) for tol_method == 'cl'; when 
            |           tol_method == 'nr' a newton-raphson method is used.
            |           Because the CCT for multiple source is calculated in one go,
            |           the atol and rtol values have to be met for all! 
        :tol_method:
            | 'newton-raphson', optional
            | (Additional) method to try and achieve set tolerances. 
            | Options: 
            | - 'cl', 'cascading-lut': use increasingly higher CCT-resolution
            |       to 'zoom-in' on the ground-truth.
            | - 'nr', 'newton-raphson': use the method as described in Li, 2016.
        :lut_resolution_reduction_factor:
            | _CCT_LUT_RESOLUTION_REDUCTION_FACTOR, optional
            | Number of times the interval spanned by the adjacent Tc in a search or lut
            | method is downsampled (the search process will then start again)
        :max_iter:
            | _CCT_MAX_ITER, optional
            | Maximum number of iterations used by the cascading-lut or newton-raphson methods.
        :split_calculation_at_N:
            | _CCT_SPLIT_CALC_AT_N, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :lut:
            | None, optional
            | Look-Up-Table with Ti, u,v,u',v',u",v",slope values of Planckians. 
            | Options:
            |  - None: defaults to the lut specified in _CCT_LUT['zhang2019']['lut_type_def'].
            |  - list (lut,lut_kwargs): use this pre-calculated lut 
            |       (add additional kwargs for the lut_generator_fcn(), defaults to None if omitted)
            |  - tuple: must be key (label) in :luts_dict: (pre-calculated dict of luts),
            |           if not: then a new lut will be generated from scratch using the info in the tuple.
            |  - str: must be key (label) in :luts_dict: (pre-calculated dict of luts)
            |  - ndarray [Nx1]: list of luts for which to generate a lut
            |  - ndarray [Nxn] with n>3: pre-calculated lut (last col must contain slope of the isotemperature lines).
        :luts_dict:
            | None, optional
            | Dictionary of pre-calculated luts for various cspaces and cmf sets.
            |  Must have structure luts_dict[cspace][cieobs][lut_label] with the
            |   lut part of a two-element list [lut, lut_kwargs]. It must contain
            |   at the top-level a key 'wl' containing the wavelengths of the 
            |   Planckians used to generate the luts in this dictionary.
            | If None: luts_dict defaults to _CCT_LUT['zhang2019']['luts']    
        :cspace:
            | _CCT_SPACE, optional
            | Color space to do calculations in. 
            | Options: 
            |    - cspace string: 
            |        e.g. 'Yuv60' for use with luxpy.colortf()
            |    - tuple with forward (i.e. xyz_to..) [and backward (i.e. ..to_xyz)] functions 
            |      (and an optional string describing the cspace): 
            |        e.g. (forward, backward) or (forward, backward, cspace string) or (forward, cspace string) 
            |    - dict with keys: 'fwtf' (foward), 'bwtf' (backward) [, optional: 'str' (cspace string)]
            |  Note: if the backward tf is not supplied, optimization in cct_to_xyz() is done in the CIE 1976 u'v' diagram
        :cspace_kwargs:
            | _CCT_CSPACE_KWARGS, optional
            | Parameter nested dictionary for the forward and backward transforms.
        :ignore_wl_diff:
            | False, optional
            | When getting a lut from the dictionary, if differences are
            | detected in the wavelengts of the lut and the ones used to calculate any
            | plankcians then a new lut should be generated. Seting this to True ignores
            | these differences and proceeds anyway.
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
        1. `Zhang, F. (2019). 
        High-accuracy method for calculating correlated color temperature with 
        a lookup table based on golden section search. 
        Optik, 193, 163018. 
        <https://doi.org/https://doi.org/10.1016/j.ijleo.2019.163018>`_
         
        2. `Li, C., Cui, G., Melgosa, M., Ruan, X., Zhang, Y., Ma, L., Xiao, K., & Luo, M. R. (2016).
        Accurate method for computing correlated color temperature. 
        Optics Express, 24(13), 14066–14078. 
        <https://doi.org/10.1364/OE.24.014066>`_

    """
    return _xyz_to_cct(xyzw, mode = 'zhang2019', cieobs = cieobs, out = out, wl = wl, is_uv_input = is_uv_input, 
                       cspace = cspace, cspace_kwargs = cspace_kwargs,
                       atol = atol, rtol = rtol, force_tolerance = force_tolerance,
                       tol_method = tol_method, max_iter = max_iter,  
                       lut_resolution_reduction_factor = lut_resolution_reduction_factor,
                       split_calculation_at_N = split_calculation_at_N, 
                       lut = lut, luts_dict = luts_dict, 
                       ignore_wl_diff = ignore_wl_diff, 
                       use_fast_duv = use_fast_duv,
                       **kwargs)


# pre-generate / load from disk / load from github some LUTs for Zhang2019:
_initialize_lut(mode = 'zhang2019',lut_types = _unique_types([_CCT_LUT['zhang2019']['lut_type_def']] + _CCT_SHARED_LUT_TYPES))


#------------------------------------------------------------------------------
# Ohno 2014 related functions:
#------------------------------------------------------------------------------
_CCT_LUT['ohno2014'] = {'luts':None}
_CCT_LUT['ohno2014']['lut_type_def'] = ((_CCT_LUT_MIN, _CCT_LUT_MAX, 10.0, 'K'),)
_CCT_LUT['ohno2014']['lut_vars'] = ['T','uv']

def _generate_lut_ohno2014(lut, 
                           uin = None, seamless_stitch = True, 
                           fallback_unit = _CCT_FALLBACK_UNIT, fallback_n = _CCT_FALLBACK_N,
                           resample_ndarray = False,
                           cct_max = _CCT_MAX, cct_min = _CCT_MIN,
                           luts_dict = None, lut_type_def = None, lut_vars = ['T','uv'],
                           cieobs =  _CIEOBS, cspace_str = None, wl = None, ignore_unequal_wl = False, 
                           lut_generator_fcn = _generate_lut, lut_generator_kwargs = {},
                           cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                           f_corr = None, ignore_f_corr_is_None = False,
                           ignore_wl_diff = False, 
                           **kwargs):
    """
    Lut generator function for ohno2014. 
    
    Args:
        :...: 
            | see docstring for _generate_lut
        :f_corr:
            | Tc,x correction factor for the parabolic solution in Ohno2014.
            |   If None, it will be recalculated (note that it depends on the lut) for increased accuracy.
        :ignore_f_corr_is_None: 
            |   If True, ignore f_corr is None, i.e. don't re-calculate f_corr.
                             
    Returns:
        :lut: 
            | an ndarray with the lut 
        :dict:
            | a dictionary with the (re-optmized) value for f_corr and for ignore_f_cor_is_None.)
    """    

    # generate lut:
    lut, _ = _generate_lut(lut, uin = uin, seamless_stitch = seamless_stitch, 
                        fallback_unit = fallback_unit, fallback_n = fallback_n,
                        resample_ndarray = resample_ndarray, cct_max = cct_max, cct_min = cct_min,
                        wl = wl, cieobs = cieobs, lut_vars = lut_vars,
                        cspace =  cspace, cspace_kwargs = cspace_kwargs)        
    
    # No f_corr needed for high resolution luts:
    if (np.round(np.diff(lut[:,0]).min(),6) == np.round(np.diff(lut[:,0]).max(),6) == np.round(lut[-1,0]-lut[-2,0],6)): # K scale
        if lut[-1,0]-lut[-2,0] <= 10:
            f_corr = 1
    
    # Get correction factor for Tx in parabolic solution:
    if (f_corr is None): 
        if (ignore_f_corr_is_None == False):
            f_corr = get_correction_factor_for_Tx(lut, 
                                                  uin = uin, 
                                                  seamless_stitch = seamless_stitch, 
                                                  fallback_unit = _CCT_FALLBACK_UNIT,
                                                  fallback_n = lut.shape[0]*4,
                                                  resample_ndarray = True,
                                                  cct_max = cct_max, 
                                                  cct_min = cct_min,
                                                  wl = wl, cieobs = cieobs, 
                                                  lut_vars = lut_vars,
                                                  cspace =  cspace, 
                                                  cspace_kwargs = cspace_kwargs,
                                                  ignore_wl_diff = ignore_wl_diff)

        else: 
            f_corr = 1.0 # use this a backup value


    return list([lut, {'f_corr':f_corr,'ignore_f_corr_is_None':ignore_f_corr_is_None}])

_CCT_LUT['ohno2014']['_generate_lut'] = _generate_lut_ohno2014

def get_correction_factor_for_Tx(lut, lut_fine = None,
                                 uin = None, seamless_stitch = True, 
                                 fallback_unit = _CCT_FALLBACK_UNIT, fallback_n = _CCT_FALLBACK_N,
                                 resample_ndarray = True,
                                 cct_max = _CCT_MAX, cct_min = _CCT_MIN,
                                 luts_dict = None, lut_type_def = None, lut_vars = ['T','uv'],
                                 cieobs =  _CIEOBS, cspace_str = None, wl = None, ignore_unequal_wl = False, 
                                 lut_generator_fcn = _generate_lut, lut_generator_kwargs = {},
                                 cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                                 f_corr = None, ignore_f_corr_is_None = False,
                                 ignore_wl_diff = False,
                                 verbosity = 0):
    """ 
    Ohno's 2014 parabolic solution uses a correction factor to correct the
    calculated CCT. However, this factor depends on the lut used. This function
    optimizes a new correction factor. Not using the right f_corr can lead to errors
    of several Kelvin. (it generates a finer resolution lut and optimizes the correction
                        factor such that predictions of the working lut for eacg of the
                        entries in this fine-resolution lut is minimized.)
    
    Args:
        :lut:
            | ndarray with lut to optimize factor for.
        :...: 
            | see docstring for _generate_lut
        :f_corr:
            | Tc,x correction factor for the parabolic solution in Ohno2014.
            |   If None, it will be recalculated (note that it depends on the lut) for increased accuracy.
        :ignore_f_corr_is_None: 
            |   If True, ignore f_corr is None, i.e. don't re-calculate f_corr.
                             

    Returns:
         :f_corr:
             | Tc,x correction factor.
    """

    # Generate a finer resolution lut to estimate the f_corr correction factor:
    if lut_fine is None:
        lut_fine, _ = _generate_lut(lut, uin = uin, seamless_stitch = seamless_stitch, 
                                    fallback_unit = fallback_unit, fallback_n = fallback_n,
                                    resample_ndarray = resample_ndarray, cct_max = cct_max, cct_min = cct_min,
                                    wl = wl, cieobs = cieobs, 
                                    cspace =  cspace, cspace_kwargs = cspace_kwargs,
                                    lut_vars = _CCT_LUT['ohno2014']['lut_vars'])

    # define shorthand lambda fcn:
    TxDuvx_p = lambda x: _xyz_to_cct(lut_fine[:,1:3], mode = 'ohno2014', 
                                     lut = [lut, {'f_corr': np.round(x,5)}], 
                                     is_uv_input = True, 
                                     force_tolerance = False, tol_method = None,
                                     out = '[cct,duv]',
                                     duv_parabolic_threshold = 0, # force use of parabolic
                                     lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
                                     wl = wl, cieobs = cieobs, ignore_wl_diff = ignore_wl_diff,
                                     cspace = cspace, cspace_kwargs = cspace_kwargs,
                                     luts_dict = _CCT_LUT['ohno2014']['luts'],
                                     )[1:-1,:]
 
    T = lut_fine[1:-1,0] 
    Duv = 0.0
    
    # define objective function:
    def optfcn(x, T, Duv, out = 'F'):
        f_corr = np.abs(x[0])
        if f_corr > 100: f_corr = 100 # limit search, avoid invalid values
        TxDuvx = TxDuvx_p(f_corr)
        Tx,Duvx = np.abs(TxDuvx[:,0]),TxDuvx[:,1] # abs needed as out_of_lut's are encode as negative!
        dT2 = (T-Tx)**2
        dDuv2 = (Duv - Duvx)**2
        F = (dT2/1000**2 + dDuv2).mean()
        
        if out == 'F':
            return F
        else:
            return eval(out)
        
    # setup and run optimization:    
    x0 = np.array([1])
    options = {'maxiter': 1e3, 'maxfev': 1e3,  'xatol': 1e-6, 'fatol': 1e-6}
    res = minimize(optfcn, x0, args = (T,Duv,'F'),method = 'Nelder-Mead', options = options)
    f_corr = np.round(res['x'][0],5)
    F, dT2, dDuv2, Tx, Duvx = optfcn(res['x'], T, Duv, out = 'F,dT2,dDuv2,Tx,Duvx')
    
    if verbosity > 1: 
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].plot(T,dT2**0.5)
        ax[0].set_title('dT (f_corr = {:1.5f})'.format(f_corr))
        ax[0].set_ylabel('dT')
        ax[0].set_xlabel('T (K)')
        ax[1].plot(T,dDuv2**0.5)
        ax[1].set_title('dDuv (f_corr = {:1.5f})'.format(f_corr))
        ax[1].set_ylabel('dDuv')
        ax[1].set_xlabel('T (K)')
        
    if verbosity > 0: 
        print('    f_corr = {:1.5f}: rmse dT={:1.4f}, dDuv={:1.6f}'.format(f_corr, dT2.mean()**0.5, dDuv2.mean()**0.5))

    return f_corr


def _uv_to_Tx_ohno2014(u, v, lut, lut_n_cols, ns = 0, out_of_lut = None, 
                       f_corr = 1.0, duv_parabolic_threshold = 0.002,
                       **kwargs):
    """ 
    Calculate Tx from u,v and lut using Ohno2014.
    """ 
    # get uBB, vBB from lut:
    TBB, uBB, vBB  = lut[:,0::lut_n_cols], lut[:,1::lut_n_cols], lut[:,2::lut_n_cols]
    idx_sources = np.arange(u.shape[0],dtype=np.int32)
    
    # calculate distances to coordinates in lut:
    di = ((u.T - uBB)**2 + (v.T - vBB)**2)**0.5
    pn = di.argmin(axis=0)

    # Deal with endpoints of lut + create intermediate variables 
    # to save memory:
    pn, out_of_lut = _deal_with_lut_end_points(pn, TBB, out_of_lut)

    TBB_m1, TBB_0, TBB_p1 = _get_pns_from_x(TBB, pn, i = idx_sources)
    uBB_m1, uBB_p1 = _get_pns_from_x(uBB, pn, i = idx_sources, m0p = 'mp')
    vBB_m1, vBB_p1 = _get_pns_from_x(vBB, pn, i = idx_sources, m0p = 'mp')
    di_m1, di_0, di_p1 = _get_pns_from_x(di, pn, i = idx_sources)

    #---------------------------------------------
    # Triangular solution:        
    l = ((uBB_p1 - uBB_m1)**2 + (vBB_p1 - vBB_m1)**2)**0.5
    l[l==0] += -_CCT_AVOID_ZERO_DIV 
    x = (di_m1**2 - di_p1**2 + l**2) / (2*l)
    # uTx = uBB_m1 + (uBB_p1 - uBB_m1)*(x/l)
    vTx = vBB_m1 + (vBB_p1 - vBB_m1)*(x/l)
    Txt = TBB_m1 + (TBB_p1 - TBB_m1) * (x/l) 
    Duvxt = (di_m1**2 - x**2)
    Duvxt[Duvxt<0] = 0
    Duvxt = (Duvxt**0.5)*np.sign(v - vTx)
    # _plot_triangular_solution(u,v,uBB,vBB,TBB,pn)


    #---------------------------------------------
    # Parabolic solution:
    X = (TBB_p1 - TBB_0) * (TBB_m1 - TBB_p1) * (TBB_0-TBB_m1)
    X[X==0] += _CCT_AVOID_ZERO_DIV
    a = (TBB_m1 * (di_p1 - di_0) + TBB_0 * (di_m1 - di_p1) + TBB_p1 * (di_0 - di_m1)) / X
    a[a==0] += _CCT_AVOID_ZERO_DIV
    b = -((TBB_m1**2) * (di_p1 - di_0) + (TBB_0**2) * (di_m1 - di_p1) + (TBB_p1**2) * (di_0 - di_m1)) / X
    c = -(di_m1 * (TBB_p1 - TBB_0)  * TBB_p1 * TBB_0  +\
          di_0  * (TBB_m1 - TBB_p1) * TBB_m1 * TBB_p1 +\
          di_p1 * (TBB_0 - TBB_m1)  * TBB_0 * TBB_m1) / X
    Txp = -b/(2*a)
    Txp_corr = Txp * f_corr # correction factor depends on the LUT !!!!! (0.0.99991 is for 1% Table I in paper, for smaller % correction factor is not needed)
    Txp = Txp_corr
    Duvxp = np.sign(v - vTx)*(a*Txp**2 + b*Txp + c)

    # Select triangular (threshold=0), parabolic (threshold=inf) or 
    # combined solution:
    Tx, Duvx = Txt, Duvxt 
    cnd = np.abs(Duvx) >= duv_parabolic_threshold
    Tx[cnd], Duvx[cnd]= Txp[cnd], Duvxp[cnd]
            
    return Tx, Duvx, out_of_lut, (TBB_m1,TBB_p1)

_CCT_UV_TO_TX_FCNS['ohno2014']= _uv_to_Tx_ohno2014
    
def xyz_to_cct_ohno2014(xyzw, cieobs = _CIEOBS, out = 'cct', is_uv_input = False, wl = None, 
                        atol = 0.1, rtol = 1e-5, force_tolerance = True, tol_method = 'newton-raphson', 
                        lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
                        duv_parabolic_threshold = 0.002,
                        split_calculation_at_N = _CCT_SPLIT_CALC_AT_N, max_iter = _CCT_MAX_ITER,
                        cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                        lut = None, luts_dict = None, ignore_wl_diff = False,
                        use_fast_duv = _CCT_FAST_DUV,
                        **kwargs):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv (distance above (>0) or below (<0) the Planckian locus) 
    using Ohno's 2014 method. 
        
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set used to calculated xyzw.
        :out: 
            | 'cct' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)
        :is_uv_input:
            | False, optional
            | If True: xyzw contain uv input data, not xyz data!
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
            | If None: use same wavelengths as CMFs in :cieobs:.
        :rtol: 
            | 1e-5, float, optional
            | Stop search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop search when cct a absolute tolerance (K) is reached.
        :force_tolerance:
            | True, optional
            | If False: search only using the list of CCTs in the used lut. 
            |           Only one loop of the full algorithm is performed. 
            |           Accuracy depends on CCT of test source and the location
            |           and spacing of the CCTs in the list.
            | If True:  search will use adjacent CCTs to test source to create a new LUT,
            |           (repeat the algoritm at higher resolution, progessively zooming in
            |            toward the ground-truth) for tol_method == 'cl'; when 
            |           tol_method == 'nr' a newton-raphson method is used.
            |           Because the CCT for multiple source is calculated in one go,
            |           the atol and rtol values have to be met for all! 
        :tol_method:
            | 'newton-raphson', optional
            | (Additional) method to try and achieve set tolerances. 
            | Options: 
            | - 'cl', 'cascading-lut': use increasingly higher CCT-resolution
            |       to 'zoom-in' on the ground-truth.
            | - 'nr', 'newton-raphson': use the method as described in Li, 2016.
        :lut_resolution_reduction_factor:
            | _CCT_LUT_RESOLUTION_REDUCTION_FACTOR, optional
            | Number of times the interval spanned by the adjacent Tc in a search or lut
            | method is downsampled (the search process will then start again)
        :duv_parabolic_threshold:
            | 0.002, optional
            | Threshold for use of the parabolic solution 
            |  (if larger then use parabolic, else use triangular solution)
        :max_iter:
            | _CCT_MAX_ITER, optional
            | Maximum number of iterations used by the cascading-lut or newton-raphson methods.
        :split_calculation_at_N:
            | _CCT_SPLIT_CALC_AT_N, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :lut:
            | None, optional
            | Look-Up-Table with Ti, u,v,u',v',u",v",slope values of Planckians. 
            | Options:
            |  - None: defaults to the lut specified in _CCT_LUT['ohno2014']['lut_type_def'].
            |  - list (lut,lut_kwargs): use this pre-calculated lut 
            |       (add additional kwargs for the lut_generator_fcn(), defaults to None if omitted)
            |  - tuple: must be key (label) in :luts_dict: (pre-calculated dict of luts),
            |           if not: then a new lut will be generated from scratch using the info in the tuple.
            |  - str: must be key (label) in :luts_dict: (pre-calculated dict of luts)
            |  - ndarray [Nx1]: list of luts for which to generate a lut
            |  - ndarray [Nxn] with n>3: pre-calculated lut (last col must contain slope of the isotemperature lines).
        :luts_dict:
            | None, optional
            | Dictionary of pre-calculated luts for various cspaces and cmf sets.
            |  Must have structure luts_dict[cspace][cieobs][lut_label] with the
            |   lut part of a two-element list [lut, lut_kwargs]. It must contain
            |   at the top-level a key 'wl' containing the wavelengths of the 
            |   Planckians used to generate the luts in this dictionary.
            | If None: luts_dict defaults to _CCT_LUT['ohno2014']['luts']    
        :cspace:
            | _CCT_SPACE, optional
            | Color space to do calculations in. 
            | Options: 
            |    - cspace string: 
            |        e.g. 'Yuv60' for use with luxpy.colortf()
            |    - tuple with forward (i.e. xyz_to..) [and backward (i.e. ..to_xyz)] functions 
            |      (and an optional string describing the cspace): 
            |        e.g. (forward, backward) or (forward, backward, cspace string) or (forward, cspace string) 
            |    - dict with keys: 'fwtf' (foward), 'bwtf' (backward) [, optional: 'str' (cspace string)]
            |  Note: if the backward tf is not supplied, optimization in cct_to_xyz() is done in the CIE 1976 u'v' diagram
        :cspace_kwargs:
            | _CCT_CSPACE_KWARGS, optional
            | Parameter nested dictionary for the forward and backward transforms.
        :ignore_wl_diff:
            | False, optional
            | When getting a lut from the dictionary, if differences are
            | detected in the wavelengts of the lut and the ones used to calculate any
            | plankcians then a new lut should be generated. Seting this to True ignores
            | these differences and proceeds anyway.
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
        1. `Ohno Y. Practical use and calculation of CCT and Duv. 
        Leukos. 2014 Jan 2;10(1):47-55.
        <http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020>`_
         
        2. `Li, C., Cui, G., Melgosa, M., Ruan, X., Zhang, Y., Ma, L., Xiao, K., & Luo, M. R. (2016).
        Accurate method for computing correlated color temperature. 
        Optics Express, 24(13), 14066–14078. 
        <https://doi.org/10.1364/OE.24.014066>`_
        
    """ 
    return _xyz_to_cct(xyzw, mode = 'ohno2014', cieobs = cieobs, out = out, wl = wl, is_uv_input = is_uv_input, 
                       cspace = cspace, cspace_kwargs = cspace_kwargs,
                       atol = atol, rtol = rtol, force_tolerance = force_tolerance,
                       tol_method = tol_method, max_iter = max_iter,  
                       lut_resolution_reduction_factor = lut_resolution_reduction_factor,
                       split_calculation_at_N = split_calculation_at_N, 
                       lut = lut, luts_dict = luts_dict, 
                       ignore_wl_diff = ignore_wl_diff, 
                       duv_parabolic_threshold = duv_parabolic_threshold,
                       use_fast_duv = use_fast_duv,
                       **kwargs)

# pre-generate / load from disk / load from github some LUTs for Ohno2014:
_initialize_lut(mode = 'ohno2014', lut_types = _unique_types([_CCT_LUT['ohno2014']['lut_type_def'], ((_CCT_LUT_MIN, _CCT_LUT_MAX, 1.0, '%'),)] + _CCT_SHARED_LUT_TYPES))


#------------------------------------------------------------------------------
# Li 2016:
#------------------------------------------------------------------------------
_CCT_LUT['li2016'] = {'lut_vars': None, 'lut_type_def': None, 'luts':None,'_generate_lut':None}


def xyz_to_cct_li2016(xyzw, cieobs = _CIEOBS, out = 'cct', is_uv_input = False, wl = None, 
                      atol = 0.1, rtol = 1e-5, max_iter = _CCT_MAX_ITER, 
                      split_calculation_at_N = _CCT_SPLIT_CALC_AT_N,
                      lut = None, luts_dict = None, ignore_wl_diff = False,
                      lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
                      cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                      first_guess_mode = 'robertson1968', fgm_kwargs = {}, 
                      use_fast_duv = _CCT_FAST_DUV,
                      **kwargs):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv(distance above (> 0) or below ( < 0) the Planckian locus) using the 
    Newton-Raphson method described in Li et al. (2016).
        
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set used to calculated xyzw.
        :out: 
            | 'cct' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)
        :is_uv_input:
            | False, optional
            | If True: xyzw contain uv input data, not xyz data!
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
            | If None: use same wavelengths as CMFs in :cieobs:.
        :rtol: 
            | 1e-5, float, optional
            | Stop method when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop method when cct a absolute tolerance (K) is reached.
        :max_iter:
            | _CCT_MAX_ITER, optional
            | Maximum number of iterations used newton-raphson methods.
        :lut_resolution_reduction_factor:
            | _CCT_LUT_RESOLUTION_REDUCTION_FACTOR, optional
            | Number of times the interval spanned by the adjacent Tc in a search or lut
            | method is downsampled (the search process will then start again)
        :split_calculation_at_N:
            | _CCT_SPLIT_CALC_AT_N, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :lut:
            | None, optional
            | Look-Up-Table with Ti, u,v,u',v',u",v",slope values of Planckians. 
            | Options:
            |  - None: defaults to the lut specified in _CCT_LUT[first_guess_mode]['lut_type_def'].
            |  - list (lut,lut_kwargs): use this pre-calculated lut 
            |       (add additional kwargs for the lut_generator_fcn(), defaults to None if omitted)
            |  - tuple: must be key (label) in :luts_dict: (pre-calculated dict of luts),
            |           if not: then a new lut will be generated from scratch using the info in the tuple.
            |  - str: must be key (label) in :luts_dict: (pre-calculated dict of luts)
            |  - ndarray [Nx1]: list of luts for which to generate a lut
            |  - ndarray [Nxn] with n>3: pre-calculated lut (last col must contain slope of the isotemperature lines).
        :luts_dict:
            | None, optional
            | Dictionary of pre-calculated luts for various cspaces and cmf sets.
            |  Must have structure luts_dict[cspace][cieobs][lut_label] with the
            |   lut part of a two-element list [lut, lut_kwargs]. It must contain
            |   at the top-level a key 'wl' containing the wavelengths of the 
            |   Planckians used to generate the luts in this dictionary.
            | If None: luts_dict defaults to _CCT_LUT[first_guess_mode]['luts']    
        :cspace:
            | _CCT_SPACE, optional
            | Color space to do calculations in. 
            | Options: 
            |    - cspace string: 
            |        e.g. 'Yuv60' for use with luxpy.colortf()
            |    - tuple with forward (i.e. xyz_to..) [and backward (i.e. ..to_xyz)] functions 
            |      (and an optional string describing the cspace): 
            |        e.g. (forward, backward) or (forward, backward, cspace string) or (forward, cspace string) 
            |    - dict with keys: 'fwtf' (foward), 'bwtf' (backward) [, optional: 'str' (cspace string)]
            |  Note: if the backward tf is not supplied, optimization in cct_to_xyz() is done in the CIE 1976 u'v' diagram
        :cspace_kwargs:
            | _CCT_CSPACE_KWARGS, optional
            | Parameter nested dictionary for the forward and backward transforms.
        :ignore_wl_diff:
            | False, optional
            | When getting a lut from the dictionary, if differences are
            | detected in the wavelengts of the lut and the ones used to calculate any
            | plankcians then a new lut should be generated. Seting this to True ignores
            | these differences and proceeds anyway.
        :first_guess_mode:
            | 'robertson1968', optional
            | Method used to get an approximate (first guess) estimate of the cct,
            | after which the newton-raphson method is started.
            | Options: 'robertson1968', 'ohno2014', 'zhang2019'
        :fgm_kwargs:
            | Dict with keyword arguments for the selected first_guess_mode.
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
        1. Out-of-lut (of first_guess_mode) CCTs are encoded as negative CCTs (with as absolute value
        the value of the closest CCT from the lut.)
    
    References:
        1. `Li, C., Cui, G., Melgosa, M., Ruan, X., Zhang, Y., Ma, L., Xiao, K., & Luo, M. R. (2016).
        Accurate method for computing correlated color temperature. 
        Optics Express, 24(13), 14066–14078. 
        <https://doi.org/10.1364/OE.24.014066>`_
        
        2.  `Robertson, A. R. (1968). 
        Computation of Correlated Color Temperature and Distribution Temperature. 
        Journal of the Optical Society of America,  58(11), 1528–1535. 
        <https://doi.org/10.1364/JOSA.58.001528>`_
    """  
    return _xyz_to_cct(xyzw, mode = 'li2016', cieobs = cieobs, out = out, 
                       wl = wl, is_uv_input = is_uv_input, 
                       cspace = cspace, cspace_kwargs = cspace_kwargs,
                       atol = atol, rtol = rtol, max_iter = max_iter,  
                       lut_resolution_reduction_factor = lut_resolution_reduction_factor,
                       split_calculation_at_N = split_calculation_at_N, 
                       lut = lut, luts_dict = luts_dict, 
                       ignore_wl_diff = ignore_wl_diff, 
                       first_guess_mode = first_guess_mode,
                       use_fast_duv = use_fast_duv,
                       **{**fgm_kwargs,**kwargs})

#------------------------------------------------------------------------------
# Fibonacci search on lut:
#------------------------------------------------------------------------------
_CCT_LUT['fibonacci'] = {} 
_CCT_LUT['fibonacci']['lut_type_def'] = ((_CCT_LUT_MIN,_CCT_LUT_MAX,0.1,'K'),)
_CCT_LUT['fibonacci']['lut_vars'] = ['T','uv'] # with 'uv' is faster than calculating the BB on the spot !
_CCT_LUT['fibonacci']['_generate_lut'] = _generate_lut 

    
def _fib_poly(a = None,b=None,t=None,d = None):
    f = [0,1]
    if d is None: 
        d = (b - a)/t
    n = 0
    while f[-1] < d:
        n+=1
        f.append(f[-1]+f[-2])
    return f[1:], n 

def _uv_to_Tx_fibonacci(u, v, lut, lut_n_cols, ns = 0, out_of_lut = None, 
                        max_iter = _CCT_MAX_ITER, uvwbar = None, wl = None, dl = None, 
                        atol = 0.1, rtol = 1e-5, lut_vars = ['T','uv'],
                        fast_duv = _CCT_FAST_DUV,
                        **kwargs):
    """ 
    Calculate Tx from u,v and lut using a fibonacci search of the lut.
    """
    # get uBB, vBB from lut:
    TBB = lut[:,0::lut_n_cols]
    if 'uv' in lut_vars: 
        uBB, vBB  = lut[:,1::lut_n_cols], lut[:,2::lut_n_cols] 

    # get list of fibonacci nrs:
    fib, n = _fib_poly(d = TBB.shape[0])
    
    a, b = np.zeros((u.shape[0],), dtype = int), np.ones((u.shape[0],), dtype = int)*(TBB.shape[0]-1) # work in indices in lut
    for k in range(1,n+1):
        
        d = np.ceil((b - a) * (fib[n - k] / fib[n - k + 1])).astype(int)  # indices -> ints 
        x1 = (b - d)
        x2 = (a + d)

        Tx_a, Tx_b = TBB[a], TBB[b]  
        if fast_duv:
            ux_a, ux_b = uBB[a], uBB[b]
            vx_a, vx_b = vBB[a], vBB[b]
        dTx = np.round(np.abs(Tx_a - Tx_b),11)
        Tx =  (Tx_a + Tx_b)/2

        if (((dTx <= atol).all() | ((dTx/Tx) <= rtol).all())) | (k == n + 2):
            break
        
        # preparation for next loop:
        if 'uv' not in lut_vars: 
            if (uvwbar is not None) & (wl is not None) & (dl is not None):
                tc_x12 = TBB[(x1,x2),].reshape((2*x1.shape[0],1)) 
                _,UVWBB_x12,_,_ = _get_tristim_of_BB_BBp_BBpp(tc_x12, uvwbar, wl, dl, out='BB')
            else:
                raise Exception('uvwbar, wl & dl must all be not None !!!')
            
            uvBB_x12 = (UVWBB_x12/UVWBB_x12.sum(axis=-1,keepdims=True))[...,1:]
            N = uvBB_x12.shape[0]//2 
            uBB_x1, vBB_x1 = uvBB_x12[:N,0:1],  uvBB_x12[:N,1:2]
            uBB_x2, vBB_x2 = uvBB_x12[N:,0:1],  uvBB_x12[N:,1:2]
        
        else:
            uBB_x1, vBB_x1 = uBB[x1], vBB[x1]
            uBB_x2, vBB_x2 = uBB[x2], vBB[x2]
        
        d2_1 = (((v - vBB_x1)**2 + (u - uBB_x1)**2)) 
        d2_2 = (((v - vBB_x2)**2 + (u - uBB_x2)**2)) 
       
        cs = (d2_1 <= d2_2)[:,0]
        cl = (d2_1 > d2_2)[:,0]
        b[cs] = x2[cs] # when d2_1 < d2_2
        a[cl] = x1[cl] # when d2_2 < d2_1
        
    
    if fast_duv:
        Duvx = _get_Duv_for_T_from_uvBB(u, v, (ux_a+ux_b)/2, (vx_a+vx_b)/2)
    else:
        Duvx = None
    
    Tx_min, Tx_max = Tx_a, Tx_b
    if out_of_lut is None: 
        closest_endpoint = -np.vstack((np.abs(Tx - TBB[0,0]).T,np.abs(Tx - TBB[-1,0]).T)).argmin(axis=0)
        TBB_closest_end = TBB[closest_endpoint]
        out_of_lut = (np.abs(Tx - TBB_closest_end) <= 2*np.abs(TBB_closest_end - TBB[closest_endpoint - 1 + 3*(closest_endpoint==0)]))

    return Tx, Duvx, out_of_lut, (Tx_min,Tx_max)

_CCT_UV_TO_TX_FCNS['fibonacci'] = _uv_to_Tx_fibonacci

def xyz_to_cct_fibonacci(xyzw, cieobs = _CIEOBS, out = 'cct', is_uv_input = False, wl = None, 
                        atol = 0.1, rtol = 1e-5, force_tolerance = True, tol_method = 'newton-raphson', 
                        lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
                        split_calculation_at_N = _CCT_SPLIT_CALC_AT_N, max_iter = _CCT_MAX_ITER,
                        cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                        lut = None, luts_dict = None, ignore_wl_diff = False,
                        use_fast_duv = _CCT_FAST_DUV,
                        **kwargs):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv(distance above (> 0) or below ( < 0) the Planckian locus) using a 
    Fibonacci search.
        
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set used to calculated xyzw.
        :out: 
            | 'cct' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)
        :is_uv_input:
            | False, optional
            | If True: xyzw contain uv input data, not xyz data!
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
            | If None: use same wavelengths as CMFs in :cieobs:.
        :rtol: 
            | 1e-5, float, optional
            | Stop search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop search when cct a absolute tolerance (K) is reached.
        :force_tolerance:
            | True, optional
            | If False: search only using the list of CCTs in the used lut.
            |           Only one loop of the full algorithm is performed. 
            |           Accuracy depends on CCT of test source and the location
            |           and spacing of the CCTs in the list.
            | If True:  search will use adjacent CCTs to test source to create a new LUT,
            |           (repeat the algoritm at higher resolution, progessively zooming in
            |            toward the ground-truth) for tol_method == 'cl'; when 
            |           tol_method == 'nr' a newton-raphson method is used.
            |           Because the CCT for multiple source is calculated in one go,
            |           the atol and rtol values have to be met for all! 
        :tol_method:
            | 'newton-raphson', optional
            | (Additional) method to try and achieve set tolerances. 
            | Options: 
            | - 'cl', 'cascading-lut': use increasingly higher CCT-resolution
            |       to 'zoom-in' on the ground-truth.
            | - 'nr', 'newton-raphson': use the method as described in Li, 2016.
        :lut_resolution_reduction_factor:
            | _CCT_LUT_RESOLUTION_REDUCTION_FACTOR, optional
            | Number of times the interval spanned by the adjacent Tc in a search or lut
            | method is downsampled (the search process will then start again)
        :max_iter:
            | _CCT_MAX_ITER, optional
            | Maximum number of iterations used by the cascading-lut or newton-raphson methods.
        :split_calculation_at_N:
            | _CCT_SPLIT_CALC_AT_N, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :lut:
            | None, optional
            | Look-Up-Table with Ti, u,v,u',v',u",v",slope values of Planckians. 
            | Options:
            |  - None: defaults to the lut specified in _CCT_LUT['fibonacci']['lut_type_def'].
            |  - list (lut,lut_kwargs): use this pre-calculated lut 
            |       (add additional kwargs for the lut_generator_fcn(), defaults to None if omitted)
            |  - tuple: must be key (label) in :luts_dict: (pre-calculated dict of luts),
            |           if not: then a new lut will be generated from scratch using the info in the tuple.
            |  - str: must be key (label) in :luts_dict: (pre-calculated dict of luts)
            |  - ndarray [Nx1]: list of luts for which to generate a lut
            |  - ndarray [Nxn] with n>3: pre-calculated lut (last col must contain slope of the isotemperature lines).
        :luts_dict:
            | None, optional
            | Dictionary of pre-calculated luts for various cspaces and cmf sets.
            |  Must have structure luts_dict[cspace][cieobs][lut_label] with the
            |   lut part of a two-element list [lut, lut_kwargs]. It must contain
            |   at the top-level a key 'wl' containing the wavelengths of the 
            |   Planckians used to generate the luts in this dictionary.
            | If None: luts_dict defaults to _CCT_LUT['fibonacci']['luts']    
        :cspace:
            | _CCT_SPACE, optional
            | Color space to do calculations in. 
            | Options: 
            |    - cspace string: 
            |        e.g. 'Yuv60' for use with luxpy.colortf()
            |    - tuple with forward (i.e. xyz_to..) [and backward (i.e. ..to_xyz)] functions 
            |      (and an optional string describing the cspace): 
            |        e.g. (forward, backward) or (forward, backward, cspace string) or (forward, cspace string) 
            |    - dict with keys: 'fwtf' (foward), 'bwtf' (backward) [, optional: 'str' (cspace string)]
            |  Note: if the backward tf is not supplied, optimization in cct_to_xyz() is done in the CIE 1976 u'v' diagram
        :cspace_kwargs:
            | _CCT_CSPACE_KWARGS, optional
            | Parameter nested dictionary for the forward and backward transforms.
        :ignore_wl_diff:
            | False, optional
            | When getting a lut from the dictionary, if differences are
            | detected in the wavelengts of the lut and the ones used to calculate any
            | plankcians then a new lut should be generated. Seting this to True ignores
            | these differences and proceeds anyway.
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
        1. Out-of-lut CCTs (or close to) are encoded as negative CCTs (with as absolute value
        the value of the closest CCT from the lut.)
    """
    return _xyz_to_cct(xyzw, mode = 'fibonacci', cieobs = cieobs, out = out, wl = wl, is_uv_input = is_uv_input, 
                       cspace = cspace, cspace_kwargs = cspace_kwargs,
                       atol = atol, rtol = rtol, force_tolerance = force_tolerance,
                       tol_method = tol_method, max_iter = max_iter,  
                       lut_resolution_reduction_factor = lut_resolution_reduction_factor,
                       split_calculation_at_N = split_calculation_at_N, 
                       lut = lut, luts_dict = luts_dict, 
                       ignore_wl_diff = ignore_wl_diff,
                       use_fast_duv = use_fast_duv,
                       **kwargs)


# pre-generate / load from disk / load from github some LUTs for fibonacci based search:
_initialize_lut(mode = 'fibonacci', lut_types = _unique_types([_CCT_LUT['fibonacci']['lut_type_def']] + _CCT_SHARED_LUT_TYPES))


    
#------------------------------------------------------------------------------
# none method:
#------------------------------------------------------------------------------
_CCT_LUT['none'] = {}
_CCT_LUT['none']['lut_type_def'] = ((_CCT_LUT_MIN, _CCT_LUT_MAX, 0.50, '%'),) # default LUT
_CCT_LUT['none']['lut_vars'] = ['T','uv']
_CCT_LUT['none']['_generate_lut'] = _generate_lut 

def _uv_to_Tx_none(u, v, lut, lut_n_cols, ns = 0, out_of_lut = None, **kwargs):
    """ 
    Basically, do (almost) nothing to test common code in _xyz_to_cct.
    """
    Tx, Duvx = 5000.31415, 0.0 # set Duv to something different from None (this way Duv is not calculated later one)
    ones = np.ones_like(u)
    out_of_lut = np.zeros_like(u,dtype=bool)
    return Tx*ones, Duvx, out_of_lut, ((Tx-100)*ones, (Tx+100)*ones)

_CCT_UV_TO_TX_FCNS['none'] = _uv_to_Tx_none

if ('robertson1968' in _CCT_LIST_OF_MODE_LUTS) & _CCT_LUT_ONE_NPY_PER_MODE:
    _CCT_LUT['none']['luts'] = copy.deepcopy(_CCT_LUT['robertson1968']['luts'])
    

if _CCT_LUT_ONE_NPY_PER_MODE == False:
    #------------------------------------------------------------------------------
    # Generate luts for all modes 
    # (this takes up more memory due to 8-column luts, than only using the 
    #  desired luts for each mode, if shared luts of 3 columns is more than 8/3 
    #  than it makes sense to use a single lut dict for all. With the 0.1 K lut
    # the shared approach results in a npy file of around 60 Mb for 4 cieobs,
    # while using the 0.1 K LUT only where needed, i.e. Fibonaci, the npy files
    # together are only around 23 Mb, only very large for Fibonacci. Additional
    # advantage of separating the luts for each mode is that the Fibonacci could
    # excluded from the luxpy distribution and downloaded when needed.)
    #-----------------------------------------------------------------------------------------------------------------------------
    # Generate 1 lut dictionary for all modes 
    # (later copy and adjust as needed on a mode per mode basis)
    # (reduce memory and storage consumption and speed up re-calculation of luts):
    _CCT_LUT['all_modes'] = {}
    _CCT_LUT['all_modes']['lut_type_def'] = ((_CCT_LUT_MIN, _CCT_LUT_MAX, 10.0, 'K'),) # default LUT
    _CCT_LUT['all_modes']['lut_vars'] = ['T','uv','uvp','uvpp','iso-T-slope']
    _CCT_LUT['all_modes']['_generate_lut'] = _generate_lut 
    
    all_modes_luts_exist = os.path.exists(os.path.join(_CCT_LUT_PATH,'all_modes_luts.pkl'))
    _CCT_LUT['all_modes']['luts'] = generate_luts(types = _unique_types([_CCT_LUT['all_modes']['lut_type_def']] + _CCT_SHARED_LUT_TYPES),
                                                  lut_file ='all_modes_luts.pkl', 
                                                  load =  (all_modes_luts_exist & (_CCT_LUT_CALC==False)), 
                                                  lut_path = _CCT_LUT_PATH, 
                                                  wl = None, cieobs = _CCT_LIST_OF_CIEOBS_LUTS,
                                                  cspace = [_CCT_CSPACE], cspace_kwargs = [_CCT_CSPACE_KWARGS],
                                                  lut_vars = _CCT_LUT['all_modes']['lut_vars'],
                                                  verbosity = _CCT_VERBOSITY_LUT_GENERATION,
                                                  lut_generator_fcn = _CCT_LUT['ohno2014']['_generate_lut'])
    
    _copy_luts('all_modes', lut = _CCT_LUT) # 2015_2 -> 2006_2, 2015_10 -> 2006_10
    #------------------------------------------------------------------------------
    
    # Sample requested lut_vars from general 'all_modes' luts for each mode:
    _CCT_LUT['none']['luts'] = _sample_lut_vars(_CCT_LUT['none']['lut_vars'], _CCT_LUT['all_modes']['luts'])
    _CCT_LUT['robertson1968']['luts'] = _sample_lut_vars(_CCT_LUT['robertson1968']['lut_vars'], _CCT_LUT['all_modes']['luts'])
    _CCT_LUT['zhang2019']['luts'] = _sample_lut_vars(_CCT_LUT['zhang2019']['lut_vars'], _CCT_LUT['all_modes']['luts'])
    _CCT_LUT['ohno2014']['luts'] = _sample_lut_vars(_CCT_LUT['ohno2014']['lut_vars'], _CCT_LUT['all_modes']['luts'])
    _CCT_LUT['fibonacci']['luts'] = _sample_lut_vars(_CCT_LUT['fibonacci']['lut_vars'], _CCT_LUT['all_modes']['luts'])



#==============================================================================
# General wrapper function for the various methods: xyz_to_cct()
#==============================================================================

def xyz_to_cct(xyzw, mode = 'robertson1968',
               cieobs = _CIEOBS, out = 'cct', is_uv_input = False, wl = None, 
               atol = 0.1, rtol = 1e-5, force_tolerance = True, tol_method = 'newton-raphson', 
               lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
               split_calculation_at_N = _CCT_SPLIT_CALC_AT_N, max_iter = _CCT_MAX_ITER,
               cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
               lut = None, luts_dict = None, ignore_wl_diff = False,
               duv_parabolic_threshold = 0.002,
               first_guess_mode = 'robertson1968', fgm_kwargs = {},
               use_fast_duv = _CCT_FAST_DUV,
               **kwargs):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv (distance above (>0) or below (<0) the Planckian locus) using a number
    of modes (methods). 
        
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        :mode:
            | 'robertson1968', optional
            | String with name of method to use.
            | Options: 'robertson1968', 'ohno2014', 'li2016', 'zhang2019', 'fibonacci',
            |       (also, but see note below: 'mcamy1992', 'hernandez1999')
            | Note: first_guess_mode for li2016 can also be specified using a ':' separator,
            |        e.g. 'li2016:robertson1968'
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set used to calculated xyzw.
        :out: 
            | 'cct' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)
        :is_uv_input:
            | False, optional
            | If True: xyzw contain uv input data, not xyz data!
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
            | If None: use same wavelengths as CMFs in :cieobs:.
        :rtol: 
            | 1e-5, float, optional
            | Stop search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop search when cct a absolute tolerance (K) is reached.
        :force_tolerance:
            | True, optional
            | If False: search only using the list of CCTs in the used lut. 
            |           Only one loop of the full algorithm is performed. 
            |           Accuracy depends on CCT of test source and the location
            |           and spacing of the CCTs in the list.
            | If True:  search will use adjacent CCTs to test source to create a new LUT,
            |           (repeat the algoritm at higher resolution, progessively zooming in
            |            toward the ground-truth) for tol_method == 'cl'; when 
            |           tol_method == 'nr' a newton-raphson method is used.
            |           Because the CCT for multiple source is calculated in one go,
            |           the atol and rtol values have to be met for all! 
        :tol_method:
            | 'newton-raphson', optional
            | (Additional) method to try and achieve set tolerances. 
            | Options: 
            | - 'cl', 'cascading-lut': use increasingly higher CCT-resolution
            |       to 'zoom-in' on the ground-truth. (not for mode == 'li2016')
            | - 'nr', 'newton-raphson': use the method as described in Li, 2016.
        :lut_resolution_reduction_factor:
            | _CCT_LUT_RESOLUTION_REDUCTION_FACTOR, optional
            | Number of times the interval spanned by the adjacent Tc in a search or lut
            | method is downsampled (the search process will then start again)
        :max_iter:
            | _CCT_MAX_ITER, optional
            | Maximum number of iterations used by the cascading-lut or newton-raphson methods.
        :split_calculation_at_N:
            | _CCT_SPLIT_CALC_AT_N, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :lut:
            | None, optional
            | Look-Up-Table with Ti, u,v,u',v',u",v",slope values of Planckians. 
            | Options:
            |  - None: defaults to the lut specified in _CCT_LUT[mode]['lut_type_def'].
            |  - list (lut,lut_kwargs): use this pre-calculated lut 
            |       (add additional kwargs for the lut_generator_fcn(), defaults to None if omitted)
            |  - tuple: must be key (label) in :luts_dict: (pre-calculated dict of luts),
            |           if not: then a new lut will be generated from scratch using the info in the tuple.
            |  - str: must be key (label) in :luts_dict: (pre-calculated dict of luts)
            |  - ndarray [Nx1]: list of luts for which to generate a lut
            |  - ndarray [Nxn] with n>3: pre-calculated lut (last col must contain slope of the isotemperature lines).
        :luts_dict:
            | None, optional
            | Dictionary of pre-calculated luts for various cspaces and cmf sets.
            |  Must have structure luts_dict[cspace][cieobs][lut_label] with the
            |   lut part of a two-element list [lut, lut_kwargs]. It must contain
            |   at the top-level a key 'wl' containing the wavelengths of the 
            |   Planckians used to generate the luts in this dictionary.
            | If None: the default dict for the mode is used 
            |   (e.g. _CCT_LUT['ohno2014']['lut_type_def'], for mode=='ohno2014').    
        :cspace:
            | _CCT_SPACE, optional
            | Color space to do calculations in. 
            | Options: 
            |    - cspace string: 
            |        e.g. 'Yuv60' for use with luxpy.colortf()
            |    - tuple with forward (i.e. xyz_to..) [and backward (i.e. ..to_xyz)] functions 
            |      (and an optional string describing the cspace): 
            |        e.g. (forward, backward) or (forward, backward, cspace string) or (forward, cspace string) 
            |    - dict with keys: 'fwtf' (foward), 'bwtf' (backward) [, optional: 'str' (cspace string)]
            |  Note: if the backward tf is not supplied, optimization in cct_to_xyz() is done in the CIE 1976 u'v' diagram
        :cspace_kwargs:
            | _CCT_CSPACE_KWARGS, optional
            | Parameter nested dictionary for the forward and backward transforms.
        :ignore_wl_diff:
            | False, optional
            | When getting a lut from the dictionary, if differences are
            | detected in the wavelengts of the lut and the ones used to calculate any
            | plankcians then a new lut should be generated. Seting this to True ignores
            | these differences and proceeds anyway.
        :duv_parabolic_threshold:
            | 0.002, optional  (cfr. mode == 'ohno2014')
            | Threshold for use of the parabolic solution 
            |  (if larger then use parabolic, else use triangular solution)
        :first_guess_mode:
            | 'robertson1968', optional (cfr. mode == 'li2016')
            | Method used to get an approximate (first guess) estimate of the cct,
            | after which the newton-raphson method is started.
            | Options: 'robertson1968', 'ohno2014', 'zhang2019'
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
        1. Using the 'mcamy1992' and 'hernandez1999' options will result in additional
        errors when cieobs is different from '1931_2' as for these options the CCT 
        is determined using a fixed set of equations optimized for the 1931 2° CMFs!!
        The only impact will be on the calculation of the Duv from the CCT. That does
        depend on the settings of cieobs and cspace! Change at own discretion.
        2. Out-of-lut CCTs are encoded as negative CCTs (with as absolute value
        the value of the closest CCT from the lut.)
    
    References:
        1.  `Robertson, A. R. (1968). 
        Computation of Correlated Color Temperature and Distribution Temperature. 
        Journal of the Optical Society of America,  58(11), 1528–1535. 
        <https://doi.org/10.1364/JOSA.58.001528>`_
         
        2. `Ohno Y. Practical use and calculation of CCT and Duv. 
        Leukos. 2014 Jan 2;10(1):47-55.
        <http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020>`_
        
        3. `Zhang, F. (2019). 
        High-accuracy method for calculating correlated color temperature with 
        a lookup table based on golden section search. 
        Optik, 193, 163018. 
        <https://doi.org/https://doi.org/10.1016/j.ijleo.2019.163018>`_
         
        3. `Li, C., Cui, G., Melgosa, M., Ruan, X., Zhang, Y., Ma, L., Xiao, K., & Luo, M. R. (2016).
        Accurate method for computing correlated color temperature. 
        Optics Express, 24(13), 14066–14078. 
        <https://doi.org/10.1364/OE.24.014066>`_
        
        4. `McCamy, Calvin S. (April 1992). 
        "Correlated color temperature as an explicit function of 
        chromaticity coordinates".
        Color Research & Application. 17 (2): 142–144.
        <http://onlinelibrary.wiley.com/doi/10.1002/col.5080170211/abstract>`_
        
        5. `Hernández-Andrés, Javier; Lee, RL; Romero, J (September 20, 1999). 
        Calculating Correlated Color Temperatures Across the Entire Gamut 
        of Daylight and Skylight Chromaticities.
        Applied Optics. 38 (27), 5703–5709. P
        <https://www.osapublishing.org/ao/abstract.cfm?uri=ao-38-27-5703>`_
    """  
    if (mode != 'mcamy1992') & (mode != 'hernandez1999'):
        
        return _xyz_to_cct(xyzw, mode, cieobs = cieobs, out = out, wl = wl, is_uv_input = is_uv_input, 
                           cspace = cspace, cspace_kwargs = cspace_kwargs,
                           atol = atol, rtol = rtol, force_tolerance = force_tolerance,
                           tol_method = tol_method, max_iter = max_iter,  
                           lut_resolution_reduction_factor = lut_resolution_reduction_factor,
                           split_calculation_at_N = split_calculation_at_N, 
                           lut = lut, luts_dict = luts_dict, 
                           ignore_wl_diff = ignore_wl_diff, 
                           duv_parabolic_threshold = duv_parabolic_threshold,
                           first_guess_mode = first_guess_mode,
                           use_fast_duv = use_fast_duv,
                           **kwargs)
    
    elif mode == 'mcamy1992':
        return xyz_to_cct_mcamy1992(xyzw,cieobs = cieobs,wl = wl,out = out,
                                    cspace = cspace,cspace_kwargs = cspace_kwargs)
    
    elif mode == 'hernandez1999':
        return xyz_to_cct_hernandez1999(xyzw,cieobs = cieobs,wl = wl,out = out,
                                    cspace = cspace,cspace_kwargs = cspace_kwargs)
    
    else:
        raise Exception ('Request mode = {:s} not implemented.'.format(mode))

#---------------------------------------------------------------------------------------------------
def xyz_to_duv(xyzw, out = 'duv', **kwargs):
    """
    Wraps xyz_to_cct, but with duv output. For kwargs info, see xyz_to_cct.
    """
    return xyz_to_cct(xyzw, out = out, **kwargs)
        
#---------------------------------------------------------------------------------------------------
def cct_to_xyz(ccts, duv = None, cct_offset = None, cieobs = _CIEOBS, wl = None,
               cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS):
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
            | luxpy._CIEOBS, optional
            | CMF set used to calculated xyzw.
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
            | If None: use same wavelengths as CMFs in :cieobs:.
        :cspace:
            | _CCT_SPACE, optional
            | Color space to do calculations in. 
            | Options: 
            |    - cspace string: 
            |        e.g. 'Yuv60' for use with luxpy.colortf()
            |    - tuple with forward (i.e. xyz_to..) [and backward (i.e. ..to_xyz)] functions 
            |      (and an optional string describing the cspace): 
            |        e.g. (forward, backward) or (forward, backward, cspace string) or (forward, cspace string) 
            |    - dict with keys: 'fwtf' (foward), 'bwtf' (backward) [, optional: 'str' (cspace string)]
            |  Note: if the backward tf is not supplied, optimization in cct_to_xyz() is done in the CIE 1976 u'v' diagram
        :cspace_kwargs:
            | _CCT_CSPACE_KWARGS, optional
            | Parameter nested dictionary for the forward and backward transforms.
        
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
        ccts = np2dT(np.array(ccts))
    else:
        ccts = np2d(ccts) 
    
    if len(ccts.shape)>2:
        raise Exception('cct_to_xyz(): Input ccts.shape must be <= 2 !')
    
    # get cct and duv arrays from :ccts:
    cct = np2d(ccts[:,0,None])

    if (duv is None) & (ccts.shape[1] == 2):
        duv = np2d(ccts[:,1,None])
    if (duv is None) & (ccts.shape[1] == 1):
        duv = np.zeros_like(ccts)
    elif duv is not None:
        duv = np2d(duv)

    cspace_dict,_ = _process_cspace(cspace, cspace_kwargs)
    if cspace_dict['bwtf'] is None:
        raise Exception('cct_to_xyz_fast requires the backward cspace transform to be defined !!!')

    xyzbar,wl, dl = _get_xyzbar_wl_dl(cieobs, wl = wl)
    if cct_offset is not None:
        # estimate iso-T-line from estimated slope using small cct offset:
        #-----------------------------------------------------------------
        _,xyzBB,_,_ = _get_tristim_of_BB_BBp_BBpp(np.vstack((cct, cct-cct_offset,cct+cct_offset)),xyzbar,wl,dl,out='BB') 
        YuvBB = cspace_dict['fwtf'](xyzBB)

        N = (xyzBB.shape[0])//3
        YuvBB_centered = (YuvBB[N:] - np.vstack((YuvBB[:N],YuvBB[:N])))
        theta = np.arctan2(YuvBB_centered[...,2:3],YuvBB_centered[...,1:2])
        theta = (theta[:N] + (theta[N:] - np.pi*np.sign(theta[N:])))/2 # take average for increased accuracy
        theta = theta + np.pi/2*np.sign(duv) # add 90° to obtain the direction perpendicular to the blackbody locus
        u, v = YuvBB[:N,1:2] + np.abs(duv)*np.cos(theta), YuvBB[:N,2:3] + np.abs(duv)*np.sin(theta)

    else:
        # estimate iso-T-line from calculated slope:
        #-------------------------------------------
        uvwbar = _convert_xyzbar_to_uvwbar(xyzbar,cspace_dict)
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
   
    # plt.plot(YuvBB[...,1],YuvBB[...,2],'gx')
    # lx.plotSL(cspace='Yuv60',axh=plt.gca())
    # plt.plot(u,v,'b+')    
    
    Yuv = np.hstack((100*np.ones_like(u),u,v))
    return cspace_dict['bwtf'](Yuv)

#------------------------------------------------------------------------------
def cct_to_mired(data):
    """
    Convert cct to Mired scale (or back). 

    Args:
        :data: 
            | ndarray with cct or Mired values.

    Returns:
        :returns: 
            | ndarray ((10**6) / data)
    """
    return np.divide(10**6,data)
  


#==============================================================================

if __name__ == '__main__':
    import luxpy as lx 
    import imp 
    imp.reload(lx)
    
    cieobs = '1931_2'
    
    BB = cri_ref([3000,5000,9000,15000,25000], ref_type = 'BB', wl3 = _WL3)
    
    xyz = spd_to_xyz(BB, cieobs = cieobs)
    
    cct = 5000
    duv = -0.05
    
    # duvs = np.array([[0.05,0.025,0,-0.025,-0.05]]).T
    duvs = np.array([[-0.05,-0.025,0,0.025,0.05]]).T
    ccts = np.array([[cct]*duvs.shape[0]]).T
    # ccts = np.array([[cct, cct, cct+1000,cct+100,cct+10, cct+1, cct+0.1, cct+0.01]]).T
    # duvs = np.array([[0,*[duv]*(ccts.shape[0]-1)]]).T
    ccts = np.array([[3500, 4500.0, 5500, 6500, 15500,25500,35500,45500,50500]]).T
    duvs = np.array([[duv]*ccts.shape[0]]).T
    
    # ccts, duvs = np.array([[2801.15]]),np.array([[-0.0]])
    # ccts, duvs = np.array([[19080.549294416654]]),np.array([[-0.040182320893775464]])
    # ccts, duvs = np.array([[9245.398073947294]]),np.array([[0.024538833865705142]])
    # ccts, duvs = np.array([[10396.099312731212]]),np.array([[0.0059537886422764115]])
    cctsduvs_t = np.hstack((ccts,duvs))

    
    cct_offset = None
    # plt.figure()
    xyz = cct_to_xyz(ccts = ccts, duv = duvs, cieobs = cieobs, wl = _WL3, cct_offset = cct_offset)
    
    # fig,ax=plt.subplots(1,1)    
    # plt.xlim([0.15,0.3])
    # plt.ylim([0.25,0.4])
    # lx.plotSL(cspace='Yuv60',cieobs=cieobs,axh=ax)
    # BB = cri_ref([cct-1,cct-0.1,cct,cct+0.5,cct+1], ref_type = 'BB', wl3 = _WL3)
    # xyzBB = spd_to_xyz(BB, cieobs = cieobs)
    
    Yuv = lx.xyz_to_Yuv60(xyz)
    # YuvBB = lx.xyz_to_Yuv60(xyzBB)
    # plt.plot(Yuv[...,1],Yuv[...,2],'rp')
    # plt.plot(YuvBB[...,1],YuvBB[...,2],'g+-')
    # plt.xlim([0.25056,0.25058])
    # plt.ylim([0.34758,0.34760])

    

    # xyz = np.array([[100,100,100]])
    modes = ['fibonacci']#,'ohno2014','zhang2019','fibonacci']
    for mode in modes:
        print('mode:',mode)
        cctsduvs = xyz_to_cct(xyz, atol = 0.1, rtol = 1e-10,cieobs = cieobs, out = '[cct,duv]', wl = _WL3, 
                              mode = mode, force_tolerance=True, 
                              tol_method='nr',
                              lut=_CCT_LUT[mode]['lut_type_def'],#((_CCT_LUT_MIN,_CCT_LUT_MAX,0.1,'K'),),
                              split_calculation_at_N=None)
    
    
    
    # cctsduvs2 = xyz_to_cct_li2016(xyz, rtol=1e-6, cieobs = cieobs, out = '[cct,duv]',force_tolerance=True)
    cctsduvs_ = cctsduvs.copy();cctsduvs_[:,0] = np.abs(cctsduvs_[:,0]) # outof gamut ccts are encoded as negative!!
    xyz_ = cct_to_xyz(cctsduvs_, cieobs = cieobs, wl = _WL3,cct_offset = cct_offset)
    print('cctsduvs_t:\n',cctsduvs_t)
    print('cctsduvs:\n', cctsduvs)
    print('Dcctsduvs:\n', cctsduvs_ - cctsduvs_t)
    print('Dxyz:\n', xyz - xyz_)
    print(cctsduvs_t[0,0],cctsduvs[0,0], cctsduvs[0,0]-cctsduvs_t[0,0])
# if __name__ == '__main__':
#     from cct_methods_analysis import get_cctduv_grid
#     ccts = np.array([2000,2250,2500,2750,3000,3500,4000,4500,5000,6000,7000,8000,9000,
#                  1e4,1.25e4,1.5e4,2e4,3e4,4e4,5e4])
#     duvs = np.array([-0.05,-0.025,0,0.025,0.05])
#     rnd_grid_offsets = (1,1,10,0) # (100% of half-distance between CCTs, 100% of half-distance between duvs, 10 random points per original grid, random seed)
    
#     error_minmaxCCT = [1500,50000]
#     error_minmaxDuv = [-0.05,0.05]
    
#     xyz, cctsduvs_t, cctsduvs_t0 = get_cctduv_grid(ccts = ccts, duvs = duvs, rnd_grid_offsets = rnd_grid_offsets)
#     cnd = (cctsduvs_t[:,0]>=error_minmaxCCT[0]) & (cctsduvs_t[:,0]<=error_minmaxCCT[1]) & (cctsduvs_t[:,1]>=error_minmaxDuv[0]) & (cctsduvs_t[:,1]<=error_minmaxDuv[1])
#     xyz = xyz[cnd,:]
#     cctsduvs_t = cctsduvs_t[cnd,:]
#     cctsduvs_t0 = cctsduvs_t0[cnd,:]
    
#     cctsduvs = xyz_to_cct(xyz, atol = 0.1, rtol = 1e-10,cieobs = cieobs, out = '[cct,duv]', wl = _WL3, 
#                           mode='robertson1968', force_tolerance=False, 
#                           tol_method='nr',lut=None,split_calculation_at_N=None)
#     cctsduvs_ = cctsduvs.copy();cctsduvs_[:,0] = np.abs(cctsduvs_[:,0]) # outof gamut ccts are encoded as negative!!

#     print('max error', np.abs(cctsduvs_[:,0]-cctsduvs_t[:,0]).max())
    
    