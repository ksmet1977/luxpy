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

 :_CCT_LUT_PATH: Folder with Look-Up-Tables (LUT) for correlated color 
                 temperature calculations. 

 :_CCT_LUT: Dict with LUTs structure LUT[mode][cspace][cieobs][lut i].
 
 :_CCT_LUT_CALC: Boolean determining whether to force LUT calculation, even if
                 the LUT.npy files can be found in ./data/cctluts/.
                 
 :_CCT_LUT: Dict with all pre-calculated LUTs.
 
 :_CCT_LUT_RESOLUTION_REDUCTION_FACTOR: number of subdivisions when performing
                                        a cascading lut calculation to zoom-in 
                                        progressively on the CCT (until a certain 
                                        tolerance is met)
                 
 :_CCT_CSPACE: default chromaticity space to calculate CCT and Duv in.
 
 :_CCT_CSPACE_KWARGS: nested dict with cspace parameters for forward and backward modes. 
 
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
                                
                                             
 :cct_to_mired(): Converts from CCT to Mired scale (or back).

===============================================================================
"""

import os
import copy 
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import luxpy as lx

# load some methods already programmed in luxpy:
from luxpy import (math, _BB, _WL3, _CIEOBS, _CMF, 
                   cie_interp, spd_to_xyz, 
                   getwlr, getwld,
                   xyz_to_Yxy, Yxy_to_xyz, xyz_to_Yuv60, Yuv60_to_xyz, 
                   xyz_to_Yuv, Yuv_to_xyz, cri_ref, 
                   )
from luxpy.utils import _PKG_PATH, _SEP, np2d, np2dT, getdata
from luxpy.color.ctf.colortf import colortf

__all__ = ['_CCT_MAX','_CCT_MIN','_CCT_CSPACE','_CCT_CSPACE_KWARGS',
           '_CCT_LUT_PATH','_CCT_LUT', '_CCT_LUT_RESOLUTION_REDUCTION_FACTOR',
           'cct_to_mired','xyz_to_cct_mcamy1992', 'xyz_to_cct_hernandez1999',
           'xyz_to_cct_robertson1968','xyz_to_cct_ohno2014',
           'xyz_to_cct_li2016','xyz_to_cct_zhang2019',
            'xyz_to_cct','cct_to_xyz', 'calculate_lut', 'generate_luts',
            '_get_lut','_generate_lut','_generate_lut_ohno2014']

_AVOID_ZERO_DIV = 1e-300
_AVOID_INF = 1/_AVOID_ZERO_DIV

#==============================================================================
# define general helper functions:
#==============================================================================
_CCT_MAX = 1e11 # don't set to higher value to avoid overflow and errors
_CCT_MIN = 550
_CCT_CSPACE = 'Yuv60'
_CCT_CSPACE_KWARGS = {'fwtf':{}, 'bwtf':{}}
_CCT_LUT_PATH = _PKG_PATH + _SEP + 'data'+ _SEP + 'cctluts' + _SEP #folder with cct lut data
_CCT_LUT_CALC = False
_CCT_LUT = {}
# _CCT_LUT_COL_NUM = 8 # T, (u, v), (u', v'; 1st deriv.), (u", v"; 2nd deriv.), slope of iso-T-lines
_CCT_LUT_RESOLUTION_REDUCTION_FACTOR = 4 # for when cascading luts are used (d(Tm1,Tp1)-->divide in _CCT_LUT_RESOLUTION_REDUCTION_FACTOR segments)
verbosity_lut_generation = 1

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
    cmf[:,c] = cmf[:,c] + (_AVOID_ZERO_DIV) # avoid nan's in uvwvbar
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

    # avoid div by inf or zero:
    exp_min_1, exp_plus_1 = exp - 1.0, exp + 1.0
    exp_min_1[exp_min_1==0] = (_AVOID_ZERO_DIV)
    exp_min_1_squared = exp_min_1**2
    exp_min_1_squared[np.isinf(exp_min_1_squared)] = _AVOID_INF # avoid warning "invalid value encountered in true_divide"
    exp_min_1_squared[exp_min_1_squared == 0.0] = _AVOID_ZERO_DIV
    exp_frac = exp/exp_min_1_squared
    
    BB = _BB['c1']*(wlt**(-5))*(1/(exp_min_1))
    BB[np.isinf(BB)] = _AVOID_INF
    
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
    cnd = np.ones((BB.shape[-1],),dtype=bool)#((xyzbar>0).sum(0)>0).T # keep only wavelengths where not all 3 cmfs are equal (to avoid nan's for 2015 cmfs which are defined only between 390 and 830 nm)
    xyz = ((BB * dl)[:,cnd] @ xyzbar[:,cnd].T)
    if 'BBp' in out.split(','): 
        xyzp = ((BBp * dl)[:,cnd] @ xyzbar[:,cnd].T)
        xyzp[np.isinf(xyzp)] = _AVOID_INF # # avoid warning "invalid value encountered in subtract" when calculating li
    if 'BBpp' in out.split(','): 
        xyzpp = ((BBpp * dl)[:,cnd] @ xyzbar[:,cnd].T)
        xyzpp[np.isinf(xyzpp)] = _AVOID_INF
    return T, xyz, xyzp, xyzpp
   
#------------------------------------------------------------------------------  
def calculate_lut(ccts, cieobs, wl = _WL3, lut_vars = ['T','uv','uvp','uvpp','iso-T-slope'],
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
            | _WL3, optional
            | Generate luts based on Planckians with wavelengths (range). 
        :lut_vars:
            | ['T','uv','uvp','uvpp','iso-T-slope'], optional
            | Data the lut should contain. Must follow this order 
            | and minimum should be ['T','uv']
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
    # Determine what to calculate:
    outBB = 'BB'
    if ('uvp' in lut_vars) | ('iso-T-slope' in lut_vars): 
        outBB = outBB + 'uvp'
    if ('uvpp' in lut_vars): outBB = outBB + 'uvpp'
 
    
    if isinstance(ccts, str):
        ccts = getdata(ccts)

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
        num[(num == 0)] += _AVOID_ZERO_DIV
        denom[(denom == 0)] += _AVOID_ZERO_DIV
    
        li = num/denom  
        li = li + np.sign(li)*_AVOID_ZERO_DIV # avoid division by zero
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

#------------------------------------------------------------------------------
def _process_lut_label(label = None, lut_int = None, lut_unit = None, lut_min_max = None):
    """
    Convert string or tuple to tuple with the interval, unit and range of the lut.
    If these values are supplied as input argument then a label will be created.
    The returned label will be a string if input label is None (of = type str),
    if type tuple then a tuple.
    """
    if (label is None) | (label is str): # compose
        if (lut_int is not None) & (lut_unit is not None):
            label = '{:1.2g}_{:s}'.format(lut_int,lut_unit)
        else:
            raise Exception('lut_int and lut_unit are minimum requirements for lut label construction')
        if lut_min_max is not None:
            if len(lut_min_max)==2:
                label = label + ',[{:1.1f}-{:1.1f}]'.format(*lut_min_max)
        return label
    elif label == tuple:
        if lut_min_max is not None:
            return (lut_int,lut_unit, tuple(lut_min_max))
        else:
            return (lut_int,lut_unit)
    else: # decompose:
        if isinstance(label,str):
            p = label.find('_')
            lut_int = float(label[:p])
            p2 = label.find(',')
            if p2 == -1:
                lut_unit = label[(p+1):]
                lut_min_max = []
            else:
                lut_unit = label[(p+1):p2]
                t = label[p2+2:-1]
                p3 = t.find('-')
                lut_min_max = [float(t[:p3]), float(t[p3+1:])]
        else:
            lut_int, lut_unit = label[0], label[1]
            lut_min_max = label[2] if len(label) > 2 else []
        return lut_int, lut_unit, lut_min_max
            
def _get_lut_characteristics(lut):
    """ 
    Guesses the interval, unit and wavelength range from lut array.
    """
    T = lut[:,0]
    Tminmax = [T.min(),T.max()]
    T = T[:3]
    T1 = np.roll(T,1)
    dT = np.abs((T-T1))[1:]
    dTr = (dT/T1[1:])
    dRD = np.abs((1e6/T - 1e6/T1))[1:]
    dRDr = (dRD/(1e6/T1[1:]))
    if (np.isclose(dT[0], dT[1])) & (~np.isclose(dTr[0],dTr[1])):
        dT,lut_unit = dT[0],'K'
    elif (~np.isclose(dT[0], dT[1])) & (np.isclose(dTr[0],dTr[1])):
        dT,lut_unit = 100*dTr[0],'%'
    elif (np.isclose(dRD[0], dRD[1])) & (~np.isclose(dRDr[0], dRDr[1])):
        dT, lut_unit = dRD[0], 'K-1'
    elif (~np.isclose(dRD[0], dRD[1])) & (np.isclose(dRDr[0], dRDr[1])):
        dT, lut_unit = 100*dRDr[0], '%-1'
    else:
        dT, lut_unit = np.nan, 'au'
    return np.round(dT,6), lut_unit, Tminmax, 
        

def _add_lut_endpoints(x):
    """ Replicates endpoints of lut to avoid out-of-bounds issues """
    return np.vstack((x[:1],x,x[-1:]))
               

def _generate_lut(lut_int = 1, lut_unit = '%', lut_min_max = [1000,5e4], cct_max = _CCT_MAX, cct_min = _CCT_MIN,
                  wl = _WL3, cieobs = _CIEOBS, lut_vars = ['T','uv','uvp','uvpp','iso-T-slope'],
                  cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                  **kwargs):
    """
    Generate a lut with a specific interval in the specified units over the specified min-max range 
    (not larger than cct_max or smaller than cct_min, whatever the input in lut_min_max!).
    
    Planckians are computed for wavelength interval wl and cmf set in cieobs and in the color space
    specified in cspace (additional arguments for these chromaticity functions can be supplied using
    the cspace_kwargs).
    
    Unit options are:
    - '%': equal relative Tc spacing (in %, cfr. (Ti+1 - Ti-1)/Ti-1).
    - 'K' equal absolute Tc spacing (in K, cfr. (Ti+1 - Ti-1).
    - '%-1': equal relative reciprocal Tc (MK-1 = mired).
    - 'K-1': equal absolute reciprocal Tc (MK-1 = mired).
    - 'au': arbitrary interval (generate by supplying an ndarray of Tc in lut_min_max).
    
    Returns single element list with an ndarray with in the columns whatever is specified in 
    lut_vars (Tc and uv are always present!). Default lut_vars =  ['T','uv','uvp','uvpp','iso-T-slope']
    - Tc: (in K)
    - u,v: chromaticity coordinates of planckians
    - u'v': chromaticity coordinates of 1st derivative of the planckians.
    - u",v": chromaticity coordinates of 2nd derivative of the planckians.
    - slope of isotemperature lines (calculated as in Robertson, 1968).
    """    
    # Get ccts for lut generation:
    if len(lut_min_max) == 2: 
        lut_min_max = np.array(lut_min_max)
        T0 = lut_min_max[0]
        Tn = lut_min_max[1]
        if '%' in lut_unit:
            # p = int((np.log(Tn/T0) / np.log(1 + lut_int/100)) + 1.0) 
            p = (((np.log(Tn/T0) / np.log(1 + lut_int/100)) + 1.0).max())
            Ts = T0*(1 + lut_int/100)**np.arange(-1,p + 1,1)[:,None]
        elif 'K' in lut_unit:
            fT = lambda m,M,d: m+(np.arange(0,(((M-m)//d)+1*(((M-m)%d)!=0)+1).max(),1)*d)[:,None]
            
            
            # fT = lambda m,M,d: np.linspace(m,m+int((M-m)/d)*d-(((M-m)%d)==0)*d+d,int((M-m)/d)+2)
            Ts = fT(T0, Tn, lut_int)
            # Ts = np.arange(lut_min_max[0],lut_min_max[1]+lut_int,lut_int)
        if '-1' in lut_unit:
            Ts[Ts==0] = 1e6/cct_max
            Ts = 1e6/Ts[::-1] # scale was in mireds        
    else:
        Ts = lut_min_max # actually stores ccts already! [Nx1] with N>2 !
    
    Ts[(Ts<cct_min)] = cct_min
    Ts[(Ts>cct_max)] = cct_max # limit to a maximum cct to avoid overflow/error and/or increase speed.    

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

  
def generate_luts(lut_file = None, load = False, lut_path = _CCT_LUT_PATH, 
                  wl = _WL3, cieobs = [_CIEOBS], 
                  types = ['15_%','1_%','0.25_%','1000_K'],
                  lut_min_max = [1e3,5e4], lut_vars = ['T','uv','uvp','uvpp','iso-T-slope'],
                  cspace = [_CCT_CSPACE], cspace_kwargs = [_CCT_CSPACE_KWARGS],
                  verbosity = 0, lut_generator_fcn = _generate_lut, 
                  lut_generator_kwargs = {}):
    """
    Generate a number of luts and store them in a nested dictionary.
    Structure: lut[cspace][cieobs][lut type].
    
    Args:
        :lut_file:
            | None, optional
            | string specifying the filename to save the lut (as.npy) to.
            | If None: don't save anything when generated (i.e. load==False).
        :load:
            | True, optional
            | If True: load previously generated dictionary.
            | If False: generate from scratch.
        :lut_path:
            | _CCT_LUT_PATH, optional
            | Path to file.
        :wl:
            | _WL3, optional
            | Wavelength for Planckian spectrum generation.
        :cieobs:
            | [_CIEOBS] or list, optional
            | Generate a LUT for each one in the list.
            | If None: generate for all cmfs in _CMF.
        :types:
            | ['15_%','1_%','0.25_%','1000_K'] or list, optional
            | List of lut intervals and units as strings (note '_' between
            | between value and unit!) or as tuples (eg. (15, '%') or (15,'%',(1e3,1e5))).
            | The latter would be '15_%,[1e3-1e5]' in string format.
            | If the min_max range is given, then this one will be used, otherwise
            | the range in :lut_min_max: is used. If units are in MK-1 then the range is also!
            |  Unit options are:
            |  - '%': equal relative Tc spacing (in %, cfr. (Ti+1 - Ti-1)/Ti-1).
            |  - 'K' equal absolute Tc spacing (in K, cfr. (Ti+1 - Ti-1).
            |  - '%-1': equal relative reciprocal Tc (MK-1 = mired).
            |  - 'K-1': equal absolute reciprocal Tc (MK-1 = mired).
            |  - 'au': arbitrary interval (generate by supplying an ndarray of Tc in lut_min_max).
            |
        :lut_min_max:
            | [1e3,5e4], optional
            | Aim for the min and max of the Tc (K or MK-1) range.
        :lut_vars:
            | ['T','uv','uvp','uvpp','iso-T-slope'], optional
            | Data the lut should contain. Must follow this order 
            | and minimum should be ['T','uv']
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
            :[lut]:
                | List of 1 ndarray with the lut.
                | The lut contains as data specified in lut_vars:
                | - T: (in K)
                | - uv: chromaticity coordinates of planckians
                | - uvp: chromaticity coordinates of 1st derivative of the planckians.
                | - uvpp: chromaticity coordinates of 2nd derivative of the planckians.
                | - iso-T-slope: slope of isotemperature lines (calculated as in Robertson, 1968).
    """
    luts = {'lut_vars' : lut_vars} 
    lut_units = ['%','K','%-1','K-1','au']
    
    # Calculate luts:
    if (load == False):
        types = [_process_lut_label(type) for type in types]
        luts['wl'] = wl
        for cspace_i,cspace_kwargs_i in zip(cspace,cspace_kwargs):
            cspace_dict_i,_ = _process_cspace(cspace_i, cspace_kwargs = cspace_kwargs_i)
            cspace_str_i = cspace_dict_i['str']
            luts[cspace_i] = {'cspace' : cspace_i, 'cspace_kwargs' : cspace_kwargs_i, 'cspace_dict': cspace_dict_i}
            if cieobs is None: cieobs = _CMF['types']
            for cieobs_j in cieobs:
                ftmp = lambda lut_int, lut_unit, lut_min_max: lut_generator_fcn(lut_int = lut_int, lut_unit = lut_unit, wl = wl, lut_min_max = lut_min_max, lut_vars = lut_vars, cieobs = cieobs_j, cspace = cspace_dict_i, cspace_kwargs = None, **lut_generator_kwargs)
                luts[cspace_i][cieobs_j] = {}
                for type_k in types:
                    tmp = list(type_k)
                    if len(tmp[-1]) == 0: tmp[-1] = lut_min_max
                    tmp[-1] = tuple(tmp[-1])
                    type_k = copy.deepcopy(tmp)
                    # if ('%' in type_k): tmp[0] *= 100
                    if verbosity > 0:
                        print('Generating lut with type = {} in cspace = {:s} for cieobs = {:s}'.format(type_k,cspace_str_i,cieobs_j))
                    
                    luts[cspace_i][cieobs_j][tuple(type_k)] = list(ftmp(tmp[0],tmp[1],tmp[2]))
        
        # save to disk:
        if lut_file is not None:
            file_path = os.path.join(lut_path,lut_file)
            if verbosity > 0:
                print('Saving dict with luts in {:s}'.format(file_path))                                                 
            np.save(file_path,luts)
    else:
        if lut_file is not None:
            file_path = os.path.join(lut_path, lut_file)
            if verbosity > 1:
                print('Loading dict with luts in {:s}'.format(file_path))                                                 
            luts = np.load(file_path,allow_pickle = True)[()]
        else:
            raise Exception('Trying to load lut file but no lut_file has been supplied.')
    return luts

def _get_lut(lut, luts_dict = None, cieobs = None, cspace_str = None, 
             default_lut_type = None, lut_vars = ['T','uv','uvp','uvpp','iso-T-slope'],
             ignore_unequal_wl = False, lut_generator_fcn = _generate_lut,
             lut_generator_kwargs = {},
             **kwargs):
    """ 
    Helper function to make coding specific xyz_to_cct_MODE functions
    more easy. It basically processes input of the :lut: argument in several 
    xyz_to_cct_MODE functions to make code in those functions cleaner. lut can 
    be the first element of list (other element lut_kwargs (i.e. lut_generator_kwargs)
    must be a dictionary with keys of any other arguments one wants to pass along
    to the :lut_generator_fcn:), if supplied it will overwrite any input via 
    lut_generator_kwargs. Lut[0] can be string or tuple specifying the type in
    a (global) nested dictionary with luts or it can be a dictionary itself 
    with all keywords required for the :lut_generator_fcn:. :cieobs: and 
    :cspace_str: are strings (keys) in the lut dictionary. the :default_lut_type:
    is used whenever :lut: is set to None by a user.
    For the user_defined lut_generator function, lut_generator_fcn kwargs can be supplied
    using the lut_generator_kwargs argument. If not supplied the fcn-defaults will be used or
    whatever is in lut_kwargs (if 2n element in lut-list).
    When :ignore_unequal_wl: no new lut will be generated when the wavelengths of the lut
    do not match the ones specified in kwargs (the keyword arguments to 
    the :lut_generator_fcn:, excluding mandatory arguments :cspace_str: 
    (which is required for indexing in the lut dictionary), like :cieobs:,
    the latter is also required when generating a new lut using the
    generator function.) When supplying cspace information to the function
    use the dictionary format (with cspace_kwargs alreadty taken into account.)
    :lut_vars: sepcifies what the lut should contain. By default it contains 
    ['T','uv','uvp','uvpp','iso-T-slope'] and minimally it should contain ['T','uv'].    
    """
    luts_dict_empty = False
    lut_is_cct_list = False
    lut_is_precalc_lut = False
    unequal_wl = False
    # cspace_dict = kwargs.pop('cspace_dict') if 'cspace_dict' in kwargs else None  
    if 'cspace' in kwargs:
        cspace_dict, cspace_str = _process_cspace(kwargs.pop('cspace'),
                                                  kwargs.pop('cspace_kwargs',None))
    

    if len(lut_generator_kwargs)==0:
        lut_kwargs = {} if 'lut_kwargs' not in kwargs else kwargs['lut_kwargs']
    else:
        lut_kwargs = lut_generator_kwargs

    wl = kwargs.pop('wl') if 'wl' in kwargs else _WL3  # use defaults if not given
        
    # get single lut:
    if isinstance(lut,list): # if list, second element contains additional kwargs for the lut_generator_fcn
        lut_kwargs = lut[1] if len(lut) > 1 else None 
        lut = lut[0]
  
    if lut is None: # use default type in luts_dict
        if luts_dict is None:
            raise Exception('User must supply a dictionary with luts when lut is None!')
        if ('wl' not in luts_dict): luts_dict_empty = True # if not present luts_dict must be empty 
        lut, lut_kwargs = copy.deepcopy(luts_dict[cspace_str][cieobs][default_lut_type])
        
    elif isinstance(lut,str) | (isinstance(lut,tuple)): # str or tuples specify the (lut_int,lut_unit,lut_min_max)
        if luts_dict is None: # luts_dict is None: generate a new lut from scratch
            if isinstance(lut,str): # prepare variable for input into _generate_lut
                lut_int,lut_unit,lut_min_max = _process_lut_label(lut)
                lut = (lut_int, lut_unit, lut_min_max)
            #raise Exception('User must supply a dictionary with luts when lut is a string key or tuple key!')
        else:
            if ('wl' not in luts_dict): luts_dict_empty = True # if not present luts_dict must be empty 
            if lut in luts_dict[cspace_str][cieobs]: # read from luts_dict
                lut, lut_kwargs = copy.deepcopy(luts_dict[cspace_str][cieobs][lut])
            else: # if not in luts_dict: generate a new lut from scratch
                if isinstance(lut,str): # prepare variable for input into _generate_lut
                    lut_int,lut_unit,lut_min_max = _process_lut_label(lut)
                    lut = (lut_int, lut_unit, lut_min_max)
 
    elif isinstance(lut,np.ndarray):
        if lut.ndim == 1:
            lut = lut[:,None] # make 2D
        if lut.shape[-1]==1:
            lut_is_cct_list = True
        else:
            lut_is_precalc_lut = True
           
    if ignore_unequal_wl == False:
        if luts_dict is not None:
            if not np.array_equal(luts_dict['wl'],wl):
                unequal_wl = True
    
    if ((unequal_wl) | (isinstance(lut, dict) | isinstance(lut, tuple) | (luts_dict_empty) | (lut_is_cct_list))) & (lut_is_precalc_lut==False):
    
        if cspace_dict is None: raise Exception('No cspace dict or other given !')
        
        # create dict with lut_kwargs for input in the generator function:
        # if not isinstance(lut_kwargs,dict) & (len(lut_generator_kwargs)>0):
        #     if not isinstance(lut_kwargs,list): lut_kwargs = [lut_kwargs] 
        #     lut_kwargs_keys = lut_generator_kwargs.keys()
        #     lut_kwargs = dict(zip(lut_kwargs_keys,lut_kwargs))

        
        if isinstance(lut, dict): 
            if ('cieobs' not in lut): lut['cieobs'] = cieobs
            if ('wl' not in lut): lut['wl'] = wl
            if ('lut_vars' not in lut): lut['lut_vars'] = lut_vars
            if ('lut_kwargs') not in lut: 
                lut = {**lut,**lut_kwargs}
            if ('cspace' not in lut): 
                lut['cspace'] = cspace_dict
                lut['cspace_kwargs'] = None
            lut, lut_kwargs = lut_generator_fcn(**lut)
        
        elif isinstance(lut, tuple):
            lut, lut_kwargs = lut_generator_fcn(lut_int = lut[0],
                                                lut_unit = lut[1],
                                                lut_min_max = lut[2],
                                                cieobs = cieobs, wl = wl,
                                                cspace = cspace_dict,
                                                lut_vars = lut_vars,
                                                cspace_kwargs = None,
                                                **lut_kwargs)
        else:
            # generator function must always have to returns!
            # Take care of wavelength difference or when lut is actually a cct_list:
            ccts = lut[:,:1]
            lut, lut_kwargs = lut_generator_fcn(lut_min_max = ccts,
                                                cieobs = cieobs, wl = wl,
                                                cspace = cspace_dict,
                                                lut_vars = lut_vars,
                                                cspace_kwargs = None,
                                                **lut_kwargs)
    
    return list([lut, lut_kwargs])


#------------------------------------------------------------------------------
def _get_Duv_for_T(u,v, T, wl, cieobs, cspace_dict, uvwbar = None, dl = None):
    """ 
    Calculate Duv from T by generating a planckian and
    calculating the Euclidean distance to the point (u,v) and
    determing the sign as the v coordinate difference between 
    the test point and the planckian.
    """
    # Get duv: 
    if (uvwbar is not None) & (dl is not None):
        _,UVWBB,_,_ = _get_tristim_of_BB_BBp_BBpp(T, uvwbar, wl, dl, out='BB')
        uvBB = xyz_to_Yxy(UVWBB)
    else:
        BB = cri_ref(T, ref_type = ['BB'], wl3 = wl)
        xyzBB = spd_to_xyz(BB, cieobs = cieobs, relative = True)
        uvBB = cspace_dict['fwtf'](xyzBB)[...,1:]
    
    
    uBB, vBB = uvBB[...,0:1], uvBB[...,1:2]
    uBB_c, vBB_c = (u - uBB), (v - vBB)
    duvs = (uBB_c**2 + vBB_c**2)**0.5
    # find sign of duv:
    theta = math.positive_arctan(uBB_c,vBB_c,htype='deg')
    theta[theta>180] = theta[theta>180] - 360
    duvs *= (np.sign(theta))
    # duvs *= (np.sign(vBB_c))
    
    return duvs

def _plot_triangular_solution(u,v,uBB,vBB,TBB,pn):
    """
    Make a plot of the geometry of the test (u,v) and the
    3 points i-1, i, i+1. Helps for testing and understanding coded algorithms.
    """
    plt.plot(u,v,'ro')
    # pnl = np.hstack((pn-2,pn-1,pn,pn+1,pn+2))
    # plt.plot(uBB[pnl],vBB[pnl],'k.-')
    plt.plot(uBB[pn-1],vBB[pn-1],'cv')
    plt.plot(uBB[pn+1],vBB[pn+1],'m^')
    plt.plot(np.vstack((u,uBB[pn-1])), np.vstack((v,vBB[pn-1])), 'c')
    plt.plot(np.vstack((u,uBB[pn+1])), np.vstack((v,vBB[pn+1])), 'm')
    plt.plot(np.vstack((uBB[pn-1],uBB[pn+1])), np.vstack((vBB[pn-1],vBB[pn+1])), 'g')
    # for i in range(TBB.shape[0]):
    #     plt.text(uBB[i],vBB[i],'{:1.0f}K'.format(TBB[i,0]))
    lx.plotSL(axh=plt.gca(),cspace='Yuv60')

def _get_pns_from_x(x, idx):
    """ 
    Get idx-1, idx and idx +1 from array. 
    Returns [Nx1] ndarray with N = len(idx).
    """
    if x.shape[-1]==1:
        return x[idx-1], x[idx], x[idx+1]
    else:
        diag = np.diag
        return diag(x[idx-1])[:,None], diag(x[idx])[:,None], diag(x[idx+1])[:,None]


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

def xyz_to_cct_mcamy1992(xyzw, cieobs = '1931_2', wl = _WL3, out = 'cct',
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

def xyz_to_cct_hernandez1999(xyzw, cieobs = '1931_2', wl = _WL3, out = 'cct',
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
    S = (UVW[...,0] + UVW[...,1] + UVW[...,2])
    u = (UVW[...,0] / (S + _AVOID_ZERO_DIV))[:,None]
    v = (UVW[...,1] / (S + _AVOID_ZERO_DIV))[:,None]
    
    if UVWp is not None:
        Sp = (UVWp[...,0] + UVWp[...,1] + UVWp[...,2])
        P = (UVWp[...,0] * S - UVW[...,0] * Sp)
        Q = (UVWp[...,1] * S - UVW[...,1] * Sp)
        up = (P / ((S**2) + _AVOID_ZERO_DIV))[:,None]
        vp = (Q / ((S**2) + _AVOID_ZERO_DIV))[:,None]
    else:
        up,vp = None, None
        
    if (UVWpp is not None) & (UVWp is not None):
        Spp = (UVWpp[...,0] + UVWpp[...,1] + UVWpp[...,2])
        Pp = (UVWpp[...,0] * S - UVW[...,0] * Spp)
        Qp = (UVWpp[...,1] * S - UVW[...,1] * Spp)
        upp = ((Pp * S - 2*P*Sp) / ((S**3) + _AVOID_ZERO_DIV))[:,None]
        vpp = ((Qp * S - 2*Q*Sp) / ((S**3) + _AVOID_ZERO_DIV))[:,None]
    else:
        upp, vpp = None, None
    
    return T, u, v, up, vp, upp, vpp

def _get_newton_raphson_estimated_Tc(u, v, T0, wl = _WL3, atol = 0.1, rtol = 1e-5,
                                     cieobs = None, xyzbar = None, uvwbar = None,
                                     cspace_dict = None, max_iter = 100):
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
    optimization short).
    
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
            uvwbar, wl, dl = _convert_xyzbar_to_uvwbar(xyzbar, cspace_dict)
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

        # Calculate f' and f":
        fp = -2*((u - uBB)*upBB + (v - vBB)*vpBB)
        fpp = 2*((upBB**2)-(u - uBB)*uppBB + (vpBB**2)-(v - vBB)*vppBB)
        
        # Calculate DT:
        DT = fp/np.abs(fpp)
        DT[DT>T] = _CCT_MIN # avoid convergence problems
        T = T - DT

        if (np.abs(DT) < atol).all() | (np.abs(DT)/T < rtol).all():
            break
        i+=1
        
    # get Duv:
    _, uBB, vBB, _, _, _, _ = _get_uv_uvp_uvpp(T, uvwbar, wl, dl, out = 'BB')
    u_c, v_c = (u - uBB), (v - vBB)
    Duv = (u_c**2 + v_c**2)**0.5
    
    # find sign of duv:
    theta = math.positive_arctan(u_c,v_c,htype='deg')
    theta[theta>180] = theta[theta>180] - 360
    Duv *= np.sign(theta)
    return T, Duv


#------------------------------------------------------------------------------
# Robertson 1968:
#------------------------------------------------------------------------------
_CCT_LUT['robertson1968'] = {}
_CCT_LUT['robertson1968']['lut_type_def'] = (25.0, 'K-1', (0.0, 625.0)) # default LUT 
_CCT_LUT['robertson1968']['lut_vars'] = ['T','uv','iso-T-slope']
_CCT_LUT['robertson1968']['_generate_lut'] = _generate_lut 

def xyz_to_cct_robertson1968(xyzw, cieobs = _CIEOBS, out = 'cct', is_uv_input = False, wl = _WL3, 
                            atol = 0.1, rtol = 1e-5, force_tolerance = True, tol_method = 'newton-raphson', 
                            lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
                            split_calculation_at_N = 10, max_iter = 100,
                            cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                            lut = None, luts_dict = None, ignore_wl_diff = False,
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
            | If False: search only using the list of CCTs in the used lut,
            |           or suplied using :cct_search_list: or :mk_search_list:. 
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
            | 100, optional
            | Maximum number of iterations used by the cascading-lut or newton-raphson methods.
        :split_calculation_at_N:
            | 10, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :lut:
            | None, optional
            | Look-Up-Table with Ti, u,v,u',v',u",v",slope values of Planckians. 
            | Options:
            |  - None: defaults to the lut specified in _CCT_LUT['robertson1968']['lut_type_def'].
            |  - list (lut,lut_kwargs): use this pre-calculated lut 
            |       (add additional kwargs for the lut_generator_fcn(), defaults to None if omitted)
            |  - str or tuple: must be key (label) in :luts_dict: (pre-calculated dict of luts)
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
    
    # Process cspace-parameters:
    cspace_dict, cspace_str = _process_cspace(cspace, cspace_kwargs)
    
    # Get chromaticity coordinates u,v from xyzw:
    uvw = cspace_dict['fwtf'](xyzw)[:,1:3]  if is_uv_input == False else xyzw[:,0:2] # xyz contained uv !!! (needed to efficiently determine f_corr)
    
    # Get or generate requested lut
    # (if wl doesn't match those in _CCT_LUT['robertson1968'] , 
    # a new recalculated lut will be generated):
    if luts_dict is None: luts_dict = _CCT_LUT['robertson1968']['luts']

    lut,_ = _get_lut(lut, luts_dict = luts_dict, 
                     default_lut_type = _CCT_LUT['robertson1968']['lut_type_def'], cieobs = cieobs, 
                     cspace_str = cspace_str, wl = wl, cspace = cspace_dict, 
                     cspace_kwargs = None, ignore_unequal_wl = ignore_wl_diff,
                     lut_vars = _CCT_LUT['robertson1968']['lut_vars'])
    
    # pre-calculate wl,dl,uvwbar for later use:
    xyzbar, wl, dl = _get_xyzbar_wl_dl(cieobs, wl)
    uvwbar = _convert_xyzbar_to_uvwbar(xyzbar, cspace_dict)
    
    # Prepare some parameters for forced tolerance:
    if force_tolerance: 
        if (tol_method == 'cascading-lut') | (tol_method == 'cl'): 
            lut_int, lut_unit, lut_min_max = _get_lut_characteristics(lut)
        elif (tol_method == 'newton-raphson') | (tol_method == 'nr'):
            max_iter_nr = max_iter 
            max_iter = 1 # no need to run multiple times, all estimation done by NR
        else:
            raise Exception('Tolerance method = {:s} not implemented.'.format(tol_method))
    
    lut_n_cols = lut.shape[-1] # store now, as this will change later
    
    # prepare split of input data to speed up calculation:
    n = xyzw.shape[0]
    ccts = np.zeros((n,1))
    duvs = np.zeros((n,1))
    n_ii = split_calculation_at_N if split_calculation_at_N is not None else n
    N_ii = n//n_ii + 1*((n%n_ii)>0)

    # loop of splitted data:
    for ii in range(N_ii):
        
        # get data for split ii:
        uv = uvw[n_ii*ii:n_ii*ii+n_ii] if (ii < (N_ii-1)) else uvw[n_ii*ii:]
        # xyz = xyzw[n_ii*ii:n_ii*ii+n_ii] if (ii < (N_ii-1)) else xyzw[n_ii*ii:]
        u, v = uv[:,0,None], uv[:,1,None]
    
        i = 0
        lut_i = lut # cascading lut will be updated later in while loop
        Duvx = None
        
        while True & (i < max_iter):
  
            # needed to get correct columns from lut_i:
            N = lut_i.shape[-1]//lut_n_cols
            ns = np.arange(0,N*lut_n_cols,lut_n_cols,dtype=int)
            
            # get uBB, vBB, mBB from lut:
            TBB, uBB, vBB, mBB  = lut_i[:,ns], lut_i[:,ns+1], lut_i[:,ns+2], lut_i[:,ns+(lut_n_cols-1)]
            mBB[mBB>0] = -mBB[mBB>0]
            
            # calculate distances to coordinates in lut (Eq. 4 in Robertson, 1968):
            di = ((v.T - vBB) - mBB * (u.T - uBB)) / ((1 + mBB**2)**(0.5))
            pn = (((v.T - vBB)**2 + (u.T - uBB)**2)**0.5).argmin(axis=0)

            # Deal with endpoints of lut + create intermediate variables 
            # to save memory:
            ce = pn == (TBB.shape[0]-1) # end point
            cb = pn==0 # begin point
            if i == 0: out_of_lut = (cb | ce)[:,None]
            pn[ce] = (TBB.shape[0] - 2) # end of lut (results in TBB_0==TBB_p1 -> (1/TBB_0)-(1/TBB_p1)) == 0 !
            pn[cb] =  1 # begin point

            # TBB = _add_lut_endpoints(TBB)
            # uBB = _add_lut_endpoints(uBB)
            # vBB = _add_lut_endpoints(vBB)
            # di = _add_lut_endpoints(di)
            # pn += 1

            # if i == 0: out_of_lut = ((pn==1) | (pn==TBB.shape[0]-2))[:,None]
          
            TBB_m1, TBB_0, TBB_p1 = _get_pns_from_x(TBB, pn)
            uBB_m1, uBB_0, uBB_p1 = _get_pns_from_x(uBB, pn)
            vBB_m1, vBB0, vBB_p1 = _get_pns_from_x(vBB, pn)
            di_m1, di_0, di_p1 = _get_pns_from_x(di, pn)

            # Estimate Tc (Robertson, 1968): 
            Tx = ((((1/TBB_0)+(di_0/((di_0 - di_p1) + _AVOID_ZERO_DIV))*((1/TBB_p1) - (1/TBB_0)))**(-1))).copy()
            
            if force_tolerance == False:
                break 
            else:
                
                if (tol_method == 'cascading-lut') | (tol_method == 'cl'):            
                    Ts_m1p1 =  np.hstack((TBB_m1,TBB_p1)) 
                    Ts_min, Ts_max = Ts_m1p1.min(axis=-1),Ts_m1p1.max(axis=-1)
                    dTs = np.abs(Ts_max - Ts_min)

                    if (dTs<= atol).all() | (np.abs(dTs/Tx) <= rtol).all():
                        Tx =  ((Ts_min + Ts_max)/2)[:,None] 
                        break 
                    else:
                        if np.isnan(lut_int): 
                            lut_min_max = 1e6/np.linspace(1e6/Ts_max,1e6/Ts_min,lut_resolution_reduction_factor)
                        else:
                            lut_min_max = [Ts_min,Ts_max] if ('-1' not in lut_unit) else [1e6/Ts_max,1e6/Ts_min]
                       
                        lut_i = _generate_lut(lut_int/lut_resolution_reduction_factor**(i+1),
                                              lut_min_max = lut_min_max, wl = wl,
                                              lut_unit = lut_unit, cieobs = cieobs,   
                                              cspace = cspace_dict, cspace_kwargs = None,
                                              lut_vars = _CCT_LUT['robertson1968']['lut_vars'])[0]

            
                elif (tol_method == 'newton-raphson') | (tol_method == 'nr'):
                    Tx, Duvx = _get_newton_raphson_estimated_Tc(u, v, Tx, wl = wl, uvwbar = uvwbar,
                                                                atol = atol, rtol = rtol, max_iter = max_iter_nr)           

            i+=1 # stop cascade loop
            
            
        if Duvx is None:
            Duvx = _get_Duv_for_T(u,v, Tx, wl, cieobs, cspace_dict, uvwbar = uvwbar, dl = dl)
        Tx = Tx*(-1)**out_of_lut
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

# pre-generate some LUTs for Robertson1968:
robertson1968_luts_exist = os.path.exists(os.path.join(_CCT_LUT_PATH,'robertson1968_luts.npy'))
_CCT_LUT['robertson1968']['luts'] = generate_luts('robertson1968_luts.npy', 
                                                load = (robertson1968_luts_exist & (_CCT_LUT_CALC==False)), 
                                                lut_path = _CCT_LUT_PATH, 
                                                wl = _WL3, cieobs = None, 
                                                types = [_CCT_LUT['robertson1968']['lut_type_def']],
                                                cspace = [_CCT_CSPACE], cspace_kwargs = [_CCT_CSPACE_KWARGS],
                                                lut_vars = _CCT_LUT['robertson1968']['lut_vars'],
                                                verbosity = verbosity_lut_generation)

# #-----------------------------------------------------------------------------
# # test code for different input formats if :lut::
# key = (25.0, 'K-1', (0.0, 625.0))
# cctduvs = xyz_to_cct_robertson1968(np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]),out='[cct,duv]',force_tolerance=True)
# print('cctduvs',cctduvs)   
# cctduvs = xyz_to_cct_robertson1968(np.array([[100,100,100],[9.5047e+01, 1.0000e+02, 1.0888e+02]]),out='[cct,duv]',force_tolerance=True)
# cctduvs2 = xyz_to_cct_robertson1968(np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]),
#                                 out='[cct,duv]',force_tolerance=False, lut = key) 
# print('cctduvs2',cctduvs2) 
# lut = _CCT_LUT['robertson1968']['luts']['Yuv60']['1931_2'][key][0]
# lut2 = _generate_lut(lut_int = 25.0, lut_unit='K-1',lut_min_max=lut[:,:1])[0]
# cctduvs3 = xyz_to_cct_robertson1968(np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]),
#                                 out='[cct,duv]',force_tolerance=True, lut = lut2)
# print('cctduvs3',cctduvs3) 
# ccts = _CCT_LUT['robertson1968']['luts']['Yuv60']['1931_2'][key][0][:,:1]
# cctduvs4 = xyz_to_cct_robertson1968(np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]),
#                                 out='[cct,duv]',force_tolerance=False, lut = ccts )
# print('cctduvs4',cctduvs4) 
# list_2 = _CCT_LUT['robertson1968']['luts']['Yuv60']['1931_2'][key]
# cctduvs5 = xyz_to_cct_robertson1968(np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]),
#                                 out='[cct,duv]',force_tolerance=False, lut = list_2 )
# print('cctduvs5',cctduvs5) 
# list_1 = [_CCT_LUT['robertson1968']['luts']['Yuv60']['1931_2'][key][0]]
# cctduvs6 = xyz_to_cct_robertson1968(np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]),
#                                 out='[cct,duv]',force_tolerance=False, lut = list_1 )
# print('cctduvs6',cctduvs6) 
# raise Exception('')

#------------------------------------------------------------------------------
# Zhang 2019:
#------------------------------------------------------------------------------
_CCT_LUT['zhang2019'] = {} 
_CCT_LUT['zhang2019']['lut_type_def'] = (25.0, 'K-1', (1.0, 1025+25.0))
_CCT_LUT['zhang2019']['lut_vars'] = ['T','uv']
_CCT_LUT['zhang2019']['_generate_lut'] = _generate_lut 

def xyz_to_cct_zhang2019(xyzw, cieobs = _CIEOBS, out = 'cct', is_uv_input = False, wl = None, 
                        atol = 0.1, rtol = 1e-5, force_tolerance = True, tol_method = 'newton-raphson', 
                        lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
                        split_calculation_at_N = 10, max_iter = 100,
                        cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                        lut = None, luts_dict = None, ignore_wl_diff = False,
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
            | If False: search only using the list of CCTs in the used lut,
            |           or suplied using :cct_search_list: or :mk_search_list:. 
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
            | 100, optional
            | Maximum number of iterations used by the cascading-lut or newton-raphson methods.
        :split_calculation_at_N:
            | 10, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :lut:
            | None, optional
            | Look-Up-Table with Ti, u,v,u',v',u",v",slope values of Planckians. 
            | Options:
            |  - None: defaults to the lut specified in _CCT_LUT['zhang2019']['lut_type_def'].
            |  - list [lut,lut_kwargs]: use this pre-calculated lut 
            |       (add additional kwargs for the lut_generator_fcn(), defaults to None if omitted)
            |  - str or tuple: must be key (label) in :luts_dict: (pre-calculated dict of luts)
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
    
    # Process cspace-parameters:
    cspace_dict, cspace_str = _process_cspace(cspace, cspace_kwargs)
    
    # Get chromaticity coordinates u,v from xyzw:
    uvw = cspace_dict['fwtf'](xyzw)[:,1:3]  if is_uv_input == False else xyzw[:,0:2] # xyz contained uv !!! (needed to efficiently determine f_corr)
    
    # Get or generate requested lut
    # (if wl doesn't match those in _CCT_LUT['zhang2019']['luts'], 
    # a new recalculated lut will be generated):
    if luts_dict is None: luts_dict = _CCT_LUT['zhang2019']['luts']
    lut, _ = _get_lut(lut, luts_dict = luts_dict, cieobs = cieobs, 
                      default_lut_type = _CCT_LUT['zhang2019']['lut_type_def'], 
                      cspace_str = cspace_str, wl = wl, cspace = cspace_dict, 
                      cspace_kwargs = None, ignore_unequal_wl = ignore_wl_diff,
                      lut_vars = _CCT_LUT['zhang2019']['lut_vars'])
    
    # lut = lut[::-1] # should be increasing CCT !
    
    # pre-calculate wl,dl,uvwbar for later use:
    xyzbar, wl, dl = _get_xyzbar_wl_dl(cieobs, wl)
    uvwbar = _convert_xyzbar_to_uvwbar(xyzbar, cspace_dict)
    
    # Prepare some parameters for forced tolerance:
    max_iter_i = max_iter
    if force_tolerance: 
        if (tol_method == 'cascading-lut') | (tol_method == 'cl'): 
            lut_int, lut_unit, lut_min_max = _get_lut_characteristics(lut)
        elif (tol_method == 'newton-raphson') | (tol_method == 'nr'):
            max_iter_nr = max_iter
            max_iter_i = 1 # no need to run multiple times, all estimation done by NR (no need for cascading loop)
        else:
            raise Exception('Tolerance method = {:s} not implemented.'.format(tol_method))
            
    lut_n_cols = lut.shape[-1] # store now, as this will change later
    
    # prepare split of input data to speed up calculation:
    n = xyzw.shape[0]
    ccts = np.zeros((n,1))
    duvs = np.zeros((n,1))
    n_ii = split_calculation_at_N if split_calculation_at_N is not None else n
    N_ii = n//n_ii + 1*((n%n_ii)>0)

    # loop of splitted data:
    for ii in range(N_ii):
        
        # get data for split ii:
        uv = uvw[n_ii*ii:n_ii*ii+n_ii] if (ii < (N_ii-1)) else uvw[n_ii*ii:]
        # xyz = xyzw[n_ii*ii:n_ii*ii+n_ii] if (ii < (N_ii-1)) else xyzw[n_ii*ii:]
        u, v = uv[:,0,None], uv[:,1,None]
    
        i = 0
        lut_i = lut # cascading lut will be updated later in while loop
        Duvx = None
        while True & (i < max_iter_i):
                
            # needed to get correct columns from lut_i:
            N = lut_i.shape[-1]//lut_n_cols
            ns = np.arange(0,N*lut_n_cols,lut_n_cols,dtype=int)
            
            # get uBB, vBB from lut:
            TBB, uBB, vBB  = lut_i[:,ns], lut_i[:,ns+1], lut_i[:,ns+2]
            # MKBB = cct_to_mired(TBB)
        
            # calculate distances to coordinates in lut (Eq. 4 in Robertson, 1968):
            di = (((v.T - vBB)**2 + (u.T - uBB)**2)**0.5)
            pn = di.argmin(axis=0)
    
            # Deal with endpoints of lut + create intermediate variables 
            # to save memory:
            ce = pn == (TBB.shape[0]-1) # end point
            cb = pn==0 # begin point
            if i == 0: out_of_lut = (cb | ce)[:,None]
            pn[ce] = (TBB.shape[0] - 2) # end of lut (results in TBB_0==TBB_p1 -> (1/TBB_0)-(1/TBB_p1)) == 0 !
            pn[cb] =  1 # begin point

            # TBB = _add_lut_endpoints(TBB)
            # uBB = _add_lut_endpoints(uBB)
            # vBB = _add_lut_endpoints(vBB)
            # di = _add_lut_endpoints(di)
            # pn += 1

            # if i == 0: out_of_lut = ((pn==1) | (pn==TBB.shape[0]-2))[:,None]
          
            TBB_m1, TBB_0, TBB_p1 = _get_pns_from_x(TBB, pn)
            uBB_m1, uBB_0, uBB_p1 = _get_pns_from_x(uBB, pn)
            vBB_m1, vBB0, vBB_p1 = _get_pns_from_x(vBB, pn)


            # get RTm-1 (RTl) and RTm+1 (RTr):
            RTl = 1e6/TBB_m1
            RTr = 1e6/TBB_p1

            # calculate RTa, RTb:
            s = (5.0**0.5 - 1.0)/2.0
            RTa = RTl + (1.0 - s) * (RTr - RTl)
            RTb = RTl + s * (RTr - RTl)
            
            Tx_a, Tx_b = cct_to_mired(RTa), cct_to_mired(RTb)
            # RTx = ((RTa+RTb)/2)
            # _plot_triangular_solution(u,v,uBB,vBB,TBB,pn)
            j = 0
            while True & (j < max_iter):
                
                # calculate BBa BBb:
                # BBab = cri_ref(np.vstack([cct_to_mired(RTa), cct_to_mired(RTx), cct_to_mired(RTb)]), ref_type = ['BB'], wl3 = wl)
                # BBab = cri_ref(np.vstack([cct_to_mired(RTa), cct_to_mired(RTb)]), ref_type = ['BB'], wl3 = wl)
                _,UVWBBab,_,_ = _get_tristim_of_BB_BBp_BBpp(np.vstack([cct_to_mired(RTa), cct_to_mired(RTb)]), uvwbar, wl, dl, out='BB')
                
                # calculate xyzBBab:
                # xyzBBab = spd_to_xyz(BBab, cieobs = cieobs, relative = True)
            
                # get cspace coordinates of BB and input xyz:
                # uvBBab = cspace_dict['fwtf'](xyzBBab)[...,1:]
                uvBBab = xyz_to_Yxy(UVWBBab)
                # N = uvBBab.shape[0]//3 
                # uBBa, vBBa = uvBBab[:N,0:1], uvBBab[:N,1:2]
                # uBBx, vBBx = uvBBab[N:2*N,0:1], uvBBab[N:2*N,1:2]
                # uBBb, vBBb = uvBBab[2*N:,0:1], uvBBab[2*N:,1:2]
                N = uvBBab.shape[0]//2 
                uBBa, vBBa = uvBBab[:N,0:1], uvBBab[:N,1:2]
                uBBb, vBBb = uvBBab[N:,0:1], uvBBab[N:,1:2]
                
                # find distance in UCD of BBab to input:
                DEuv_a = ((uBBa - u)**2 + (vBBa - v)**2)**0.5
                DEuv_b = ((uBBb - u)**2 + (vBBb - v)**2)**0.5
                
                c = (DEuv_a < DEuv_b)[:,0]
                
                # when DEuv_a < DEuv_b:
                RTr[c] = RTb[c]
                RTb[c] = RTa[c]
                DEuv_b[c] = DEuv_a[c]
                RTa[c] = (RTl[c] + (1.0 - s) * (RTr[c] - RTl[c])).copy()
                
                # when DEuv_a >= DEuv_b:
                RTl[~c] = RTa[~c]
                RTa[~c] = RTb[~c]
                DEuv_a[~c] = DEuv_b[~c]
                RTb[~c] = (RTl[~c] + s * (RTr[~c] - RTl[~c]))
                
                # Calculate CCTs from RTa and RTb:
                Tx_a, Tx_b = cct_to_mired(RTa), cct_to_mired(RTb)

                Tx = cct_to_mired((RTa+RTb)/2)
                dTx = np.abs(Tx_a - Tx_b)
                # print(j,Tx, dTx,RTx,RTa-RTb)
                if (((dTx <= atol).all() | ((dTx/Tx) <= rtol).all())):
                    break
                j+=1
                
            # uBB = np.vstack((uBBa,uBBx,uBBb))
            # vBB = np.vstack((vBBa,vBBx,vBBb))
            # TBB = np.vstack((Tx_a,Tx,Tx_b))
            # pn = np.array([1])
            # _plot_triangular_solution(u,v,uBB,vBB,TBB,pn)

            if force_tolerance == False:
                break 
            else:
                # RTa[out_of_lut] = 1e6/((1e6/RTa[out_of_lut]) - 100)
                # RTb[out_of_lut] = 1e6/((1e6/RTb[out_of_lut]) + 100)
                if (tol_method == 'cascading-lut') | (tol_method == 'cl'):
                    Ts_m1p1 =  1e6/np.hstack((RTa,RTb)) 
                    Ts_min, Ts_max = Ts_m1p1.min(axis=-1),Ts_m1p1.max(axis=-1)
                    
                    dTs = np.abs(Ts_max - Ts_min)
                    if (dTs<= atol).all() | (np.abs(dTs/Tx) <= rtol).all():
                        Tx =  ((Ts_min + Ts_max)/2)[:,None] 
                        break 
                    else:
                        if np.isnan(lut_int): 
                            lut_min_max = 1e6/np.linspace(1e6/Ts_max,1e6/Ts_min,lut_resolution_reduction_factor)
                        else:
                            lut_min_max = [Ts_min,Ts_max] if ('-1' not in lut_unit) else [1e6/Ts_max,1e6/Ts_min]

                        lut_i = _generate_lut(lut_int/lut_resolution_reduction_factor**(i+1),
                                              lut_min_max = lut_min_max, wl = wl,
                                              lut_unit = lut_unit, cieobs = cieobs,   
                                              cspace = cspace_dict, cspace_kwargs = None,
                                              lut_vars = _CCT_LUT['zhang2019']['lut_vars'])[0]
                        
                        
            
                elif (tol_method == 'newton-raphson') | (tol_method == 'nr'):
                    Tx, Duvx = _get_newton_raphson_estimated_Tc(u, v, Tx, wl = wl, uvwbar = uvwbar,
                                                                atol = atol, rtol = rtol, max_iter = max_iter_nr)
                           
            
            i+=1 # stop cascade loop
            
            
        if Duvx is None:
            Duvx = _get_Duv_for_T(u,v, Tx, wl, cieobs, cspace_dict, uvwbar = uvwbar, dl = dl)
        
        Tx = Tx*(-1)**out_of_lut
        
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


# pre-generate some LUTs for Zhang2019:
zhang2019_luts_exist = os.path.exists(os.path.join(_CCT_LUT_PATH,'zhang2019_luts.npy'))
_CCT_LUT['zhang2019']['luts'] = generate_luts('zhang2019_luts.npy', 
                                    load = (zhang2019_luts_exist & (_CCT_LUT_CALC==False)), 
                                    lut_path = _CCT_LUT_PATH, 
                                    wl = _WL3, cieobs = None, 
                                    types = [_CCT_LUT['zhang2019']['lut_type_def']],
                                    cspace = [_CCT_CSPACE], cspace_kwargs = [_CCT_CSPACE_KWARGS],
                                    lut_vars = _CCT_LUT['zhang2019']['lut_vars'],
                                    verbosity = verbosity_lut_generation)

#-----------------------------------------------------------------------------
# # test code for different input formats if :lut::
# print('Zhang2019')
# cctduvs = xyz_to_cct_zhang2019(np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]),out='[cct,duv]',force_tolerance=True)
# print('cctduvs',cctduvs)   
# # cctduvs = xyz_to_cct_zhang2019(np.array([[100,100,100],[9.5047e+01, 1.0000e+02, 1.0888e+02]]),out='[cct,duv]',force_tolerance=True)
# cctduvs2 = xyz_to_cct_zhang2019(np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]),
#                                 out='[cct,duv]',force_tolerance=True, lut = (25.0, 'K-1', (1.0, 1025+25.0)))
# print('cctduvs2',cctduvs2) 
# lut = _CCT_LUT['zhang2019']['luts']['Yuv60']['1931_2'][(25.0, 'K-1', (1.0, 1025+25.0))][0]
# lut2 = _generate_lut(lut_int = 25.0, lut_unit='K-1',lut_min_max=lut[:,:1])[0]
# cctduvs3 = xyz_to_cct_zhang2019(np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]),
#                                 out='[cct,duv]',force_tolerance=True, lut = lut2 )
# print('cctduvs3',cctduvs3) 
# ccts = _CCT_LUT['zhang2019']['luts']['Yuv60']['1931_2'][(25.0, 'K-1', (1.0, 1025+25.0))][0][:,:1]
# cctduvs4 = xyz_to_cct_zhang2019(np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]),
#                                 out='[cct,duv]',force_tolerance=False, lut = ccts )
# print('cctduvs4',cctduvs4) 
# list_2 = _CCT_LUT['zhang2019']['luts']['Yuv60']['1931_2'][(25.0, 'K-1', (1.0, 1025+25.0))]
# cctduvs5 = xyz_to_cct_zhang2019(np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]),
#                                 out='[cct,duv]',force_tolerance=False, lut = list_2 )
# print('cctduvs5',cctduvs5) 
# list_1 = [_CCT_LUT['zhang2019']['luts']['Yuv60']['1931_2'][(25.0, 'K-1', (1.0, 1025+25.0))][0]]
# cctduvs6 = xyz_to_cct_zhang2019(np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]),
#                                 out='[cct,duv]',force_tolerance=False, lut = list_1 )
# print('cctduvs6',cctduvs6) 
# raise Exception('')



#------------------------------------------------------------------------------
# Ohno 2014:
#------------------------------------------------------------------------------
_CCT_LUT['ohno2014'] = {'luts':None}
_CCT_LUT['ohno2014']['lut_type_def'] = (1.0, '%', (1000.0, 50000.0))
_CCT_LUT['ohno2014']['lut_vars'] = ['T','uv']

def get_correction_factor_for_Tx(lut, 
                                 lut_int = 1, lut_unit = '%', lut_min_max = [1000,5e4], 
                                 wl = _WL3, cieobs = _CIEOBS, ignore_wl_diff = False,
                                 cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
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
        :lut_int:
            | 1, optional  
            | CCT interval (see lut_unit for more info)
        :lut_unit:
            | '%', optional 
            | String specifier for the units in the lut.
            | Unit options are:
            | - '%': equal relative Tc spacing (in %, cfr. (Ti+1 - Ti-1)/Ti-1).
            | - 'K' equal absolute Tc spacing (in K, cfr. (Ti+1 - Ti-1).
            | - '%-1': equal relative reciprocal Tc (MK-1 = mired).
            | - 'K-1': equal absolute reciprocal Tc (MK-1 = mired).
            | - 'au': arbitrary interval (generate by supplying an ndarray of Tc in lut_min_max).
        :lut_min_max:
            | [1e3,5e4], optional
            | Minimum and maximum values (in units in lut_unit) of the lut.
            | (note that the actual values are allways in K !!!)
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set used to calculated xyzw.
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
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
        :verbosity:
            | 0, optional
            | If > 9; print some intermediate (status) output.
        :ignore_wl_diff:
            | False, optional
            | When getting a lut from the dictionary, if differences are
            | detected in the wavelengts of the lut and the ones used to calculate any
            | plankcians then a new lut should be generated. Seting this to True ignores
            | these differences and proceeds anyway.
     
        Returns:
         :f_corr:
             | Tc,x correction factor.
    """
    # Generate a finer resolution lut to estimate the f_corr correction factor:
    lut_fine,_ = _generate_lut(lut_int = lut_int, lut_unit = lut_unit, lut_min_max = lut_min_max, 
                             wl = wl, cieobs = cieobs, cspace = cspace, 
                             cspace_kwargs = cspace_kwargs,
                             lut_vars = _CCT_LUT['ohno2014']['lut_vars'])

    # define shorthand lambda fcn:
    TxDuvx_p = lambda x: xyz_to_cct_ohno2014(lut_fine[:,1:3], lut = [lut, {'f_corr':np.round(x,5)}], is_uv_input = True, 
                                            force_tolerance = False, out = '[cct,duv]',
                                            duv_parabolic_threshold = 0, # force use of parabolic
                                            lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
                                            wl = wl, cieobs = cieobs, ignore_wl_diff = ignore_wl_diff,
                                            cspace = cspace, cspace_kwargs = cspace_kwargs,
                                            luts_dict = _CCT_LUT['ohno2014']['luts'])[1:-1,:]
 
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
    


def _generate_lut_ohno2014(lut_int = 1, lut_unit = True, lut_min_max = [1000,5e4], 
                           wl = _WL3, cieobs = _CIEOBS, cspace = _CCT_CSPACE, 
                           cspace_kwargs = _CCT_CSPACE_KWARGS, ignore_wl_diff = False,
                           f_corr = None, ignore_f_corr_is_None = False,
                           lut_vars = ['T','uv']):
    """
    Generate a lut with a specific interval in the specified units over the specified min-max range.
    
    Planckians are computed for wavelength interval wl and cmf set in cieobs and in the color space
    specified in cspace (additional arguments for these chromaticity functions can be supplied using
    the cspace_kwargs). [=  for _generate_lut with some additions to allow for the
    Tc,x correction factor for the parabolic solution in Ohno2014]
                         
    Returns ndarray with lut and an optimized f_corr factor (when f_corr is set to None 
    in the input, else use whatever is set in f_corr.)
    """    
    # generate lut:
    lut = _generate_lut(lut_int = lut_int, lut_unit = lut_unit, lut_min_max = lut_min_max,
                        wl = wl, cieobs = cieobs, cspace = cspace, 
                        cspace_kwargs = cspace_kwargs, lut_vars = lut_vars)[0]        

    # Get correction factor for Tx in parabolic solution:
    if (f_corr is None): 
        if (ignore_f_corr_is_None == False):
            f_corr = get_correction_factor_for_Tx(lut, lut_int = lut_int/4, 
                                                  lut_unit = lut_unit, 
                                                  lut_min_max = lut_min_max, 
                                                  wl = wl, cieobs = cieobs, 
                                                  cspace = cspace, 
                                                  cspace_kwargs = cspace_kwargs,
                                                  ignore_wl_diff = ignore_wl_diff)
        else: 
            f_corr = 1.0 # use this a backup value


    return list([lut, {'f_corr':f_corr,'ignore_f_corr_is_None':ignore_f_corr_is_None}])

_CCT_LUT['ohno2014']['_generate_lut'] = _generate_lut_ohno2014
    
def xyz_to_cct_ohno2014(xyzw, cieobs = _CIEOBS, out = 'cct', is_uv_input = False, wl = None, 
                        atol = 0.1, rtol = 1e-5, force_tolerance = True, tol_method = 'newton-raphson', 
                        lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
                        duv_parabolic_threshold = 0.002,
                        split_calculation_at_N = 10, max_iter = 100,
                        cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                        lut = None, luts_dict = None, ignore_wl_diff = False,
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
            | If False: search only using the list of CCTs in the used lut,
            |           or suplied using :cct_search_list: or :mk_search_list:. 
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
            | 100, optional
            | Maximum number of iterations used by the cascading-lut or newton-raphson methods.
        :split_calculation_at_N:
            | 10, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :lut:
            | None, optional
            | Look-Up-Table with Ti, u,v,u',v',u",v",slope values of Planckians. 
            | Options:
            |  - None: defaults to the lut specified in _CCT_LUT['ohno2014']['lut_type_def'].
            |  - list [lut,lut_kwargs]: use this pre-calculated lut 
            |       (add additional kwargs for the lut_generator_fcn(), defaults to None if omitted)
            |  - str or tuple: must be key (label) in :luts_dict: (pre-calculated dict of luts)
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
    
    # Process cspace-parameters:
    cspace_dict, cspace_str = _process_cspace(cspace, cspace_kwargs)
    
    # Get chromaticity coordinates u,v from xyzw:
    uvw = cspace_dict['fwtf'](xyzw)[:,1:3]  if is_uv_input == False else xyzw[:,0:2] # xyz contained uv !!! (needed to efficiently determine f_corr)
    
    # Get or generate requested lut
    # (if wl doesn't match those in _CCT_LUT['ohno2014']['luts'], 
    # a new recalculated lut will be generated):
    if luts_dict is None: luts_dict = _CCT_LUT['ohno2014']['luts']
    lut, lut_kwargs = _get_lut(lut, luts_dict = luts_dict, cieobs = cieobs, 
                               default_lut_type = _CCT_LUT['ohno2014']['lut_type_def'], 
                               cspace_str = cspace_str, wl = wl, cspace = cspace_dict, 
                               cspace_kwargs = None, ignore_unequal_wl = ignore_wl_diff, 
                               lut_generator_fcn = _CCT_LUT['ohno2014']['_generate_lut'],
                               lut_vars = _CCT_LUT['ohno2014']['lut_vars'])

    f_corr = lut_kwargs['f_corr']
    
    
    # Prepare some parameters for forced tolerance:
    if force_tolerance: 
        if (tol_method == 'cascading-lut') | (tol_method =='cl'): 
            lut_int, lut_unit, lut_min_max  = _get_lut_characteristics(lut)
        elif (tol_method == 'newton-raphson') | (tol_method == 'nr'):
            xyzbar, wl, dl = _get_xyzbar_wl_dl(cieobs, wl)
            uvwbar = _convert_xyzbar_to_uvwbar(xyzbar, cspace_dict)
            max_iter_nr = max_iter 
            max_iter = 1 # no need to run multiple times, all estimation done by NR
        else:
            raise Exception('Tolerance method = {:s} not implemented.'.format(tol_method))
    
    lut_n_cols = lut.shape[-1] # store now, as this will change later
    
    # prepare split of input data to speed up calculation:
    n = xyzw.shape[0]
    ccts = np.zeros((n,1))
    duvs = np.zeros((n,1))
    n_ii = split_calculation_at_N if split_calculation_at_N is not None else n
    N_ii = n//n_ii + 1*((n%n_ii)>0)

    # loop of splitted data:
    for ii in range(N_ii):
        
        # get data for split ii:
        uv = uvw[n_ii*ii:n_ii*ii+n_ii] if (ii < (N_ii-1)) else uvw[n_ii*ii:]
        # xyz = xyzw[n_ii*ii:n_ii*ii+n_ii] if (ii < (N_ii-1)) else xyzw[n_ii*ii:]
        u, v = uv[:,0,None], uv[:,1,None]
    
        i = 0
        lut_i = lut # cascading lut will be updated later in while loop ()
        while True & (i < max_iter):

            # needed to get correct columns from lut_i:
            N = lut_i.shape[-1]//lut_n_cols
            ns = np.arange(0,N*lut_n_cols,lut_n_cols,dtype=int)

            
            # get uBB, vBB from lut:
            TBB, uBB, vBB  = lut_i[:,ns], lut_i[:,ns+1], lut_i[:,ns+2]

            # calculate distances to coordinates in lut:
            di = ((u.T - uBB)**2 + (v.T - vBB)**2)**0.5
            pn = di.argmin(axis=0)
    
            # Deal with endpoints of lut + create intermediate variables 
            # to save memory:
            ce = pn == (TBB.shape[0]-1) # end point
            cb = pn==0 # begin point
            if i == 0: out_of_lut = (cb | ce)[:,None]
            pn[ce] = (TBB.shape[0] - 2) # end of lut (results in TBB_0==TBB_p1 -> (1/TBB_0)-(1/TBB_p1)) == 0 !
            pn[cb] =  1 # begin point
                     
            # TBB_m1, TBB_0, TBB_p1 = _get_pns_from_x(TBB, pn)
            # uBB_m1, uBB_0, uBB_p1 = _get_pns_from_x(uBB, pn)
            # vBB_m1, vBB0, vBB_p1 = _get_pns_from_x(vBB, pn)
            # di_m1, di_0, di_p1 = _get_pns_from_x(di, pn)

            # TBB = _add_lut_endpoints(TBB)
            # uBB = _add_lut_endpoints(uBB)
            # vBB = _add_lut_endpoints(vBB)
            # di = _add_lut_endpoints(di)

            # pn += 1
            # if i == 0: out_of_lut = ((pn==1) | (pn==TBB.shape[0]))[:,None]
            TBB_m1, TBB_0, TBB_p1 = _get_pns_from_x(TBB, pn)
            uBB_m1, uBB_0, uBB_p1 = _get_pns_from_x(uBB, pn)
            vBB_m1, vBB0, vBB_p1 = _get_pns_from_x(vBB, pn)
            di_m1, di_0, di_p1 = _get_pns_from_x(di, pn)

            #---------------------------------------------
            # Triangular solution:        
            l = ((uBB_p1 - uBB_m1)**2 + (vBB_p1 - vBB_m1)**2)**0.5
            l[l==0] += -_AVOID_ZERO_DIV 
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
            X[X==0] += _AVOID_ZERO_DIV
            a = (TBB_m1 * (di_p1 - di_0) + TBB_0 * (di_m1 - di_p1) + TBB_p1 * (di_0 - di_m1)) / X
            a[a==0] += _AVOID_ZERO_DIV
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
            
            
            if force_tolerance == False:
                break 
            else:
                
                if (tol_method == 'cascading-lut') | (tol_method == 'cl'):
                                
                    Ts_m1p1 =  np.hstack((TBB_m1,TBB_p1)) 
                    Ts_min, Ts_max = Ts_m1p1.min(axis=-1),Ts_m1p1.max(axis=-1)
                    
                    dTs = np.abs(Ts_max - Ts_min)
                    if (dTs<= atol).all() | (np.abs(dTs/Tx) <= rtol).all():
                        Tx =  ((Ts_min + Ts_max)/2)[:,None] 
                        break 
                    else:
                        lut_min_max = [Ts_min,Ts_max] if ('-1' not in lut_unit) else [1e6/Ts_max,1e6/Ts_min]

                        lut_i, lut_kwargs = _generate_lut_ohno2014(lut_int/lut_resolution_reduction_factor**(i+1),
                                                                   lut_min_max = lut_min_max, 
                                                                   lut_unit = lut_unit, cieobs = cieobs, 
                                                                   wl = wl, f_corr = 1, ignore_wl_diff = ignore_wl_diff,
                                                                   cspace = cspace_dict, cspace_kwargs = None,
                                                                   lut_vars = _CCT_LUT['ohno2014']['lut_vars'])
                        f_corr = lut_kwargs['f_corr']
                        
                elif (tol_method == 'newton-raphson') | (tol_method == 'nr'):
                    Tx, Duvx = _get_newton_raphson_estimated_Tc(u, v, Tx, wl = wl, uvwbar = uvwbar,
                                                                atol = atol, rtol = rtol, max_iter = max_iter_nr)
                    
            i+=1 # stop cascade loop
        
        Tx = Tx*(-1)**out_of_lut
        
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
 
   
 
# pre-generate some LUTs for Ohno 2014:
ohno2014_luts_exist = os.path.exists(os.path.join(_CCT_LUT_PATH,'ohno2014_luts.npy'))
_CCT_LUT['ohno2014']['luts'] = generate_luts('ohno2014_luts.npy', 
                                load = (ohno2014_luts_exist & (_CCT_LUT_CALC==False)), 
                                lut_path = _CCT_LUT_PATH, 
                                wl = _WL3, cieobs = None, 
                                types = [_CCT_LUT['ohno2014']['lut_type_def'],'15_%','0.25_%', '100_K'],
                                lut_min_max = [1e3,5e4],
                                cspace = [_CCT_CSPACE], cspace_kwargs = [_CCT_CSPACE_KWARGS],
                                lut_vars = _CCT_LUT['ohno2014']['lut_vars'],
                                verbosity = verbosity_lut_generation, 
                                lut_generator_fcn = _CCT_LUT['ohno2014']['_generate_lut'])
         
#------------------------------------------------------------------------------
# Li 2016:
#------------------------------------------------------------------------------
_CCT_LUT['li2016'] = {'lut_vars': None, 'lut_type_def': None, 'luts':None,'_generate_lut':None}
def xyz_to_cct_li2016(xyzw, cieobs = _CIEOBS, out = 'cct', is_uv_input = False, wl = None, 
                      atol = 0.1, rtol = 1e-5, max_iter = 100, split_calculation_at_N = 10, 
                      cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                      first_guess_mode = 'robertson1968', fgm_kwargs = {}, **kwargs):
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
            | 100, optional
            | Maximum number of iterations used newton-raphson methods.
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
        :first_guess_mode:
            | 'robertson1968', optional
            | Method used to get an approximate (first guess) estimate of the cct,
            | after which the newton-raphson method is started.
            | Options: 'robertson1968', 'ohno2014', 'zhang2019'
        :fgm_kwargs:
            | Dict with keyword arguments for the selected first_guess_mode.
            
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
    
    # Process cspace-parameters:
    cspace_dict, cspace_str = _process_cspace(cspace, cspace_kwargs)
    
    # Get xyzbar cmfs:
    xyzbar, wl, dl = _get_xyzbar_wl_dl(cieobs, wl)
    
    # Convert xyz cmfs to uvw cmfs:
    uvwbar = _convert_xyzbar_to_uvwbar(xyzbar, cspace_dict)
    
    # Get chromaticity coordinates u,v from xyzw:
    uvw = cspace_dict['fwtf'](xyzw)[:,1:3] if (is_uv_input == False) else xyzw[:,0:2] # xyz contained uv !!! (needed to efficiently determine f_corr)
    
    # prepare split of input data to speed up calculation:
    n = xyzw.shape[0]
    ccts = np.zeros((n,1))
    duvs = np.zeros((n,1))
    n_ii = split_calculation_at_N if split_calculation_at_N is not None else n
    N_ii = n//n_ii + 1*((n%n_ii)>0)

    # loop of splitted data:
    for ii in range(N_ii):
        
        # get data for split ii:
        uv = uvw[n_ii*ii:n_ii*ii+n_ii] if (ii < (N_ii-1)) else uvw[n_ii*ii:]
        # xyz = xyzw[n_ii*ii:n_ii*ii+n_ii] if (ii < (N_ii-1)) else xyzw[n_ii*ii:]
        u, v = uv[:,0,None], uv[:,1,None]
        
        # get first estimate of Tx using recommended method = Robertson, 1968:
        if 'force_tolerance' in fgm_kwargs: fgm_kwargs.pop('force_tolerance')
        if first_guess_mode == 'robertson1968': 
            
            Tx0 = xyz_to_cct_robertson1968(uv, is_uv_input = True, out = 'cct', cieobs = cieobs,
                                           wl = wl, cspace = cspace_dict, cspace_kwargs = None,
                                           atol = atol, rtol = rtol, force_tolerance = False, 
                                           max_iter = max_iter, **fgm_kwargs)
        elif first_guess_mode == 'ohno2014': 
            Tx0 = xyz_to_cct_ohno2014(uv, is_uv_input = True, out = 'cct', cieobs = cieobs,
                                      wl = wl, cspace = cspace_dict, cspace_kwargs = None,
                                      atol = atol, rtol = rtol, force_tolerance = False,
                                      max_iter = max_iter, **fgm_kwargs)
        elif first_guess_mode == 'zhang2019':
            Tx0 = xyz_to_cct_zhang2019(uv, is_uv_input = True, out = 'cct', cieobs = cieobs,
                                       wl = wl, cspace = cspace_dict, cspace_kwargs = None,
                                       atol = atol, rtol = rtol, force_tolerance = False,
                                       max_iter = max_iter, **fgm_kwargs)
        else:
            raise Exception ('Request first_guess_mode = {:s} not implemented.'.format(first_guess_mode))
        
        # Apply Newton–Raphson's method (use abs(Tx0) as first_guess, as out_of_gamuts are encodes as negative CCTs):
        Tx, Duvx = _get_newton_raphson_estimated_Tc(u, v, np.abs(Tx0), wl = wl, atol = atol, rtol = rtol,
                                                    cieobs = cieobs, xyzbar = xyzbar, uvwbar = uvwbar,
                                                    cspace_dict = cspace_dict, max_iter = max_iter)
       
        Tx = Tx*np.sign(Tx0) # put out_of_gamut encoding back in.
        
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
  
    
#==============================================================================
# General wrapper function for the various methods: xyz_to_cct()
#==============================================================================

def xyz_to_cct(xyzw, mode = 'ohno2014',
               cieobs = _CIEOBS, out = 'cct', is_uv_input = False, wl = None, 
               atol = 0.1, rtol = 1e-5, force_tolerance = True, tol_method = 'newton-raphson', 
               lut_resolution_reduction_factor = _CCT_LUT_RESOLUTION_REDUCTION_FACTOR,
               split_calculation_at_N = 10, max_iter = 100,
               cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
               lut = None, luts_dict = None, ignore_wl_diff = False,
               duv_parabolic_threshold = 0.002,
               first_guess_mode = 'robertson1968', fgm_kwargs = {},
               **kwargs):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv (distance above (>0) or below (<0) the Planckian locus) using a number
    of modes (methods). 
        
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        :mode:
            | 'ohno2014', optional
            | String with name of method to use.
            | Options: 'robertson1968', 'ohno2014', 'li2016', 'zhang2019'
            |       (also, but see note below: 'mcamy1992', 'hernandez1999')
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
            | If False: search only using the list of CCTs in the used lut,
            |           or suplied using :cct_search_list: or :mk_search_list:. 
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
            | 100, optional
            | Maximum number of iterations used by the cascading-lut or newton-raphson methods.
        :split_calculation_at_N:
            | 10, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :lut:
            | None, optional
            | Look-Up-Table with Ti, u,v,u',v',u",v",slope values of Planckians. 
            | Options:
            |  - None: defaults to the lut specified in _CCT_LUT['ohno2014']['lut_type_def'].
            |  - list [lut,lut_kwargs]: use this pre-calculated lut 
            |       (add additional kwargs for the lut_generator_fcn(), defaults to None if omitted)
            |  - str or tuple: must be key (label) in :luts_dict: (pre-calculated dict of luts)
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
        :fgm_kwargs:
            | Dict with keyword arguments for the selected first_guess_mode in Li's 2016 method.
            
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
    # get first estimate of Tx using recommended method = Robertson, 1968:
    if mode == 'robertson1968': 
        return xyz_to_cct_robertson1968(xyzw, cieobs = cieobs, out = out, wl = wl, is_uv_input = is_uv_input, 
                                        cspace = cspace, cspace_kwargs = cspace_kwargs,
                                        atol = atol, rtol = rtol, force_tolerance = force_tolerance,
                                        tol_method = tol_method, max_iter = max_iter,  
                                        lut_resolution_reduction_factor = lut_resolution_reduction_factor,
                                        split_calculation_at_N = split_calculation_at_N, 
                                        lut = lut, luts_dict = luts_dict, 
                                        ignore_wl_diff = ignore_wl_diff, 
                                        **kwargs)
    elif mode == 'ohno2014': 
        return xyz_to_cct_ohno2014(xyzw, cieobs = cieobs, out = out, wl = wl, is_uv_input = is_uv_input, 
                                   cspace = cspace, cspace_kwargs = cspace_kwargs,
                                   atol = atol, rtol = rtol, force_tolerance = force_tolerance,
                                   tol_method = tol_method, max_iter = max_iter,  
                                   lut_resolution_reduction_factor = lut_resolution_reduction_factor,
                                   split_calculation_at_N = split_calculation_at_N, 
                                   lut = lut, luts_dict = luts_dict, 
                                   ignore_wl_diff = ignore_wl_diff, 
                                   duv_parabolic_threshold = duv_parabolic_threshold,
                                   **kwargs)
    elif mode == 'zhang2019':
        return xyz_to_cct_zhang2019(xyzw, cieobs = cieobs, out = out, wl = wl, is_uv_input = is_uv_input, 
                                    cspace = cspace, cspace_kwargs = cspace_kwargs,
                                    atol = atol, rtol = rtol, force_tolerance = force_tolerance,
                                    tol_method = tol_method, max_iter = max_iter,  
                                    lut_resolution_reduction_factor = lut_resolution_reduction_factor,
                                    split_calculation_at_N = split_calculation_at_N, 
                                    lut = lut, luts_dict = luts_dict, 
                                    ignore_wl_diff = ignore_wl_diff, 
                                    **kwargs)
    elif mode == 'li2016':
        return xyz_to_cct_li2016(xyzw, cieobs = cieobs, out = out, wl = wl, is_uv_input = is_uv_input, 
                                 cspace = cspace, cspace_kwargs = cspace_kwargs,
                                 atol = atol, rtol = rtol, force_tolerance = force_tolerance,
                                 tol_method = tol_method, max_iter = max_iter,  
                                 lut_resolution_reduction_factor = lut_resolution_reduction_factor,
                                 split_calculation_at_N = split_calculation_at_N, 
                                 lut = lut, luts_dict = luts_dict, 
                                 ignore_wl_diff = ignore_wl_diff, 
                                 first_guess_mode = first_guess_mode,
                                 fgm_kwargs = fgm_kwargs,
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
        num[(num == 0)] += _AVOID_ZERO_DIV
        denom[(denom == 0)] += _AVOID_ZERO_DIV
        li = num/denom  
        li = li + np.sign(li)*_AVOID_ZERO_DIV # avoid division by zero
        mi = -1.0/li # slope of isotemperature lines

        YuvBB = xyz_to_Yxy(UVW)
        u, v = YuvBB[:,1:2] + np.sign(mi) * duv*(1/((1+mi**2)**0.5)), YuvBB[:,2:3] + np.sign(mi)* duv*((mi)/(1+mi**2)**0.5)
   
    # plt.plot(YuvBB[...,1],YuvBB[...,2],'gx')
    # lx.plotSL(cspace='Yuv60',axh=plt.gca())
    # plt.plot(u,v,'b+')    
    
    Yuv = np.hstack((100*np.ones_like(u),u,v))
    return cspace_dict['bwtf'](Yuv)

    

if __name__ == '__main__':
    import luxpy as lx 
    import imp 
    imp.reload(lx)
    
    cieobs = '1931_2'
    
    BB = cri_ref([3000,5000,9000,15000], ref_type = 'BB', wl3 = _WL3)
    
    xyz = spd_to_xyz(BB, cieobs = cieobs)
    
    cct = 6500
    duvs = np.array([[0.05,0.025,0,-0.025,-0.05]]).T
    duvs = np.array([[-0.03,0.03]]).T
    ccts = np.array([[cct]*duvs.shape[0]]).T
    cctsduvs_t = np.hstack((ccts,duvs))
    
    cct_offset = None
    plt.figure()
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
    cctsduvs = xyz_to_cct(xyz, rtol = 1e-6,cieobs = cieobs, out = '[cct,duv]', wl = _WL3, 
                          mode='ohno2014',force_tolerance=True,tol_method='cl',lut=[(25,'K-1',(15,625)),{'f_corr':0.4}])
    # cctsduvs2 = xyz_to_cct_li2016(xyz, rtol=1e-6, cieobs = cieobs, out = '[cct,duv]',force_tolerance=True)
    cctsduvs_ = cctsduvs.copy();cctsduvs_[:,0] = np.abs(cctsduvs_[:,0]) # outof gamut ccts are encoded as negative!!
    xyz_ = cct_to_xyz(cctsduvs_, cieobs = cieobs, wl = _WL3,cct_offset = cct_offset)
    print('cctsduvs_t:\n',cctsduvs_t)
    print('cctsduvs:\n', cctsduvs)
    print('Dcctsduvs:\n', cctsduvs - cctsduvs_t)
    print('Dxyz:\n', xyz - xyz_)

