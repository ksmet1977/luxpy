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

 :_CCT_MAX: (= 1e11), max. value that does not cause overflow problems. 

 :_CCT_LUT_PATH: Folder with Look-Up-Tables (LUT) for correlated color 
                 temperature calculation followings Ohno's method.

 :_CCT_LUT: Dict with LUTs.
 
 :_CCT_LUT_CALC: Boolean determining whether to force LUT calculation, even if
                 the LUT can be found in ./data/cctluts/.
                 
 :_CCT_CSPACE: default chromaticity space to calculate CCT and Duv in.
 
 :_CCT_CSPACE_KWARGS: nested dict with cspace parameters for forward and backward modes. 
 
 
 :_CCT_SEARCH_METHOD: string with default search method.
 
 :_OHNO2014_FALLBACK_MODE: string with fallback method when Ohno's 2014 LUT algorithm has out-of-lut values.
 
 :_CCT_SEARCH_LIST_OHNO2014:  ndarray with default CCTs to start Ohno's 2014 LUT algorithms.
 
 :_MK_SEARCH_LIST_OHNO2014: ndarray with default CCTs (in mired) to start Ohno's 2014 LUT algorithms.
 
 :_CCT_SEARCH_LIST_ROBERTSON1968:  ndarray with default CCTs to start Robertson's 1968 search algorithms.
 
 :_MK_SEARCH_LIST_ROBERTSON1968: ndarray with default CCTs (in mired) to start Robertson's 1968 search algorithms.
 
 :_CCT_SEARCH_LIST_BRUTEFORCE:  ndarray with default CCTs to start the brute-force search algorithms.
 
 :_MK_SEARCH_LIST_BRUTEFORCE: ndarray with default CCTs (in mired) to start the brute-force search algorithms.
 
 :_CCT_SEARCH_LIST_PW_LIN:  ndarray with (piecewise) linearly spaced  CCTs to start the search algorithms.
 
 :_MK_SEARCH_LIST_PW_LIN: ndarray with (piecewise) linearly spaced CCTs (in mired) to start the search algorithms.

 
 :calculate_lut(): Function that calculates the LUT for the ccts stored in 
                   ./data/cctluts/cct_lut_cctlist.dat or given as input 
                   argument. Calculation is performed for CMF set specified in
                   cieobs. Adds a new (temprorary) field to the _CCT_LUT dict.

 :calculate_luts(): Function that recalculates (and overwrites) LUTs in 
                    ./data/cctluts/ for the ccts stored in 
                    ./data/cctluts/cct_lut_cctlist_{lut_mode}.dat or given as input 
                    argument. Calculation is performed for all CMF sets listed 
                    in _CMF['types'].

 :xyz_to_cct(): | Calculates CCT, Duv from XYZ 
                | wrapper for xyz_to_cct_ohno2014() & xyz_to_cct_search()

 :xyz_to_duv(): | Calculates Duv, (CCT) from XYZ
                | wrapper for xyz_to_cct_ohno2014() & xyz_to_cct_search()
                
 :xyz_to_cct_search(): Calculates CCT, Duv from XYZ using brute-force search 
                       algorithm or Zhang's 2019 golden-ratio or Robertson's 1968 method.

 :cct_to_xyz_fast(): Calculates xyz from CCT, Duv by estimating 
                     the line perpendicular to the planckian locus.

 :cct_to_xyz(): Calculates xyz from CCT, Duv [100 K < CCT < _CCT_MAX]

 :xyz_to_cct_mcamy(): | Calculates CCT from XYZ using Mcamy model:
                      | `McCamy, Calvin S. (April 1992). 
                        Correlated color temperature as an explicit function of 
                        chromaticity coordinates. 
                        Color Research & Application. 17 (2): 142–144. 
                        <http://onlinelibrary.wiley.com/doi/10.1002/col.5080170211/abstract>`_

 :xyz_to_cct_HA(): | Calculate CCT from XYZ using Hernández-Andrés et al. model.
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
                       
 :xyz_to_cct_search_zhang2019():  | Calculates CCT, Duv from XYZ using Zhang's 2019 golden-ratio search algorithm
                                  | `Zhang, F. (2019). 
                                    High-accuracy method for calculating correlated color temperature with 
                                    a lookup table based on golden section search. 
                                    Optik, 193, 163018. 
                                    <https://doi.org/https://doi.org/10.1016/j.ijleo.2019.163018>`_
                     
 :xyz_to_cct_search_robertson1968(): | Calculates CCT, Duv from XYZ using a Robertson's 1968 search method.
                                     | `Robertson, A. R. (1968). 
                                        Computation of Correlated Color Temperature and Distribution Temperature. 
                                        Journal of the Optical Society of America,  58(11), 1528–1535. 
                                        <https://doi.org/10.1364/JOSA.58.001528>`_
                                                     
 :cct_to_mired(): Converts from CCT to Mired scale (or back).

===============================================================================
"""
#from . import _CCT_LUT_CALC
import copy
import os
from luxpy import  (_BB, _WL3, _CMF, _CIEOBS, math, 
                    getwlr, getwld, spd_to_xyz, cie_interp,
                    cri_ref, blackbody,  
                    xyz_to_Yxy, Yxy_to_xyz, 
                    xyz_to_Yuv, Yuv_to_xyz, 
                    xyz_to_Yuv60, Yuv60_to_xyz)
from luxpy.utils import np, pd, sp, _PKG_PATH, _SEP, _EPS, np2d, np2dT, getdata, dictkv
from luxpy.color.ctf.colortf import colortf

_CCT_MAX = 1e11 # maximum value that does not cause overflow problems

#------------------------------------------------------------------------------
_CCT_LUT_PATH = _PKG_PATH + _SEP + 'data'+ _SEP + 'cctluts' + _SEP #folder with cct lut data

#------------------------------------------------------------------------------
# Definition of some default search lists for use with brute-force or golden-ratio (Zhang,2019) or Robertson1968 based search methods:
_CCT_SEARCH_LIST_PW_LIN = np.array([[50,100,500,1000,2000,3000,4000,5000,6000,10000, 20000,50000, 7.5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10, 5e10, 1e11, _CCT_MAX]]).T
_MK_SEARCH_LIST_PW_LIN = 1e6/_CCT_SEARCH_LIST_PW_LIN
# _MK_SEARCH_LIST_ROBERTSON1968 = np.hstack((np.arange(1e-308,20,1),np.arange(20,100,10),np.arange(100,625,25),np.arange(625,1000,100),np.arange(1000,2200,200)))
_MK_SEARCH_LIST_ROBERTSON1968 = np.hstack((np.arange(1e-300,20,1),np.arange(20,50,2),np.arange(50,100,10),np.arange(100,625,25),np.arange(625,1000,100),np.arange(1000,2400,200)))
_CCT_SEARCH_LIST_ROBERTSON1968 = 1e6/_MK_SEARCH_LIST_ROBERTSON1968
_CCT_SEARCH_LIST_ROBERTSON1968[0] = _CCT_MAX
pd.DataFrame(_CCT_SEARCH_LIST_ROBERTSON1968).to_csv('{}cct_lut_cctlist_{:s}.dat'.format(_CCT_LUT_PATH, 'robertson1968'),header=None,float_format='%1.9e',index=False)
_CCT_SEARCH_LIST_ROBERTSON1968 = getdata('{}cct_lut_cctlist_{:s}.dat'.format(_CCT_LUT_PATH, 'robertson1968'))
_CCT_SEARCH_LIST_ROBERTSON1968[np.isinf(_CCT_SEARCH_LIST_ROBERTSON1968)] = _CCT_MAX # avoid overflow problems causing calculation of wrong CCTS!!
_MK_SEARCH_LIST_ROBERTSON1968 = 1e6/_CCT_SEARCH_LIST_ROBERTSON1968
_CCT_SEARCH_LIST_OHNO2014 = getdata('{}cct_lut_cctlist_{:s}.dat'.format(_CCT_LUT_PATH, 'ohno2014'))
_CCT_SEARCH_LIST_OHNO2014[np.isinf(_CCT_SEARCH_LIST_OHNO2014)] = _CCT_MAX # avoid overflow problems causing calculation of wrong CCTS!!
_MK_SEARCH_LIST_OHNO2014 = 1e6/_CCT_SEARCH_LIST_OHNO2014
_MK_SEARCH_LIST_ZHANG2019 = np2d(np.arange(1.0,1025+25,25)).T
_CCT_SEARCH_LIST_ZHANG2019 = 1e6/_MK_SEARCH_LIST_ZHANG2019
_MK_SEARCH_LIST_BRUTEFORCE  = np2d(np.hstack((np.arange(1e6/_CCT_MAX,1+0.1,0.1),
                              _MK_SEARCH_LIST_ZHANG2019[1:,0],
                              np.arange(1026.0,1025+50*21,50)[1:],
                              1e6/np.arange(450,0,-100)))).T
_CCT_SEARCH_LIST_BRUTEFORCE = 1e6/_MK_SEARCH_LIST_BRUTEFORCE
_CCT_SEARCH_METHOD = 'robertson1968'
_OHNO2014_FALLBACK_MODE = _CCT_SEARCH_METHOD


_CCT_LUT_CALC = False # True: (re-)calculates LUTs for ccts in .cctluts/cct_lut_cctlist.dat
_CCT_CSPACE = 'Yuv60' # chromaticity diagram to perform CCT, Duv calculations in
_CCT_CSPACE_KWARGS = {'fwtf':{},'bwtf':{}} # any required parameters in the xyz_to_cspace() funtion
__all__ = ['_CCT_LUT_CALC', '_CCT_MAX','_CCT_CSPACE', '_CCT_CSPACE_KWARGS']

__all__ +=['_CCT_SEARCH_LIST_OHNO2014','_MK_SEARCH_LIST_OHNO2014',
           '_CCT_SEARCH_LIST_ROBERTSON1968','_MK_SEARCH_LIST_ROBERTSON1968',
           '_CCT_SEARCH_LIST_ZHANG2019', '_MK_SEARCH_LIST_ZHANG2019',
           '_CCT_SEARCH_LIST_PW_LIN', '_MK_SEARCH_LIST_PW_LIN',
           '_CCT_SEARCH_LIST_BRUTEFORCE','_MK_SEARCH_LIST_BRUTEFORCE',
           '_CCT_SEARCH_METHOD','_OHNO2014_FALLBACK_MODE']

__all__ += ['_CCT_LUT_PATH', 'cct_to_mired',
            '_CCT_LUT','calculate_lut', 'calculate_luts', 
            'xyz_to_cct','xyz_to_duv', 'cct_to_xyz_fast', 'cct_to_xyz',
            'xyz_to_cct_ohno2014', 'xyz_to_cct_search_zhang2019','xyz_to_cct_search_robertson1968',
            'xyz_to_cct_search', 'xyz_to_cct_search_bf_fast', 'xyz_to_cct_search_bf_robust',
            'xyz_to_cct_HA','xyz_to_cct_mcamy']



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
def _process_cspace_input(cspace, cspace_kwargs = None, cust_str = 'cspace'):
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

    return cspace_dict
        

#------------------------------------------------------------------------------ 
def _get_tristim_of_BB_and_BBprime(T, xyzbar, wl):
    """ Get the tristimulus values for CMF set xyzbar of the blackbody radiatior spectra
    and the spectra corresponding to the blackbody radiator derivated to Tc.
    """
    T = np2d(T)
    wl = wl*1e-9
    dl = getwld(wl)
    exp = np.exp(_BB['c2']/(wl*T))
    
    # avoid div by inf or zero:
    exp_min_1 = exp - 1.0
    exp_min_1[exp_min_1==0] = (1e-308)
    exp_min_1_squared = (exp-1.0)**2
    exp_min_1_squared[np.isinf(exp_min_1_squared)] = 1e308 # avoid warning "invalid value encountered in true_divide"
    exp_min_1_squared[exp_min_1_squared == 0.0] = 1e-308
    exp_frac = exp/exp_min_1_squared
    
    BB = _BB['c1']*(wl**(-5))*(1/(exp_min_1))
    BB[np.isinf(BB)] = 1e308
    BBprime = (_BB['c1']*_BB['c2']*(T**(-2))*(wl**(-6)))*exp_frac
    cnd = ((xyzbar>0).sum(0)==3).T # keep only wavelengths where not all 3 cmfs are equal (to avoid nan's for 2015 cmfs which are defined only between 390 and 830 nm)
    xyz = ((BB * dl)[:,cnd] @ xyzbar[:,cnd].T)
    xyzprime = ((BBprime * dl)[:,cnd] @ xyzbar[:,cnd].T)
    xyzprime[np.isinf(xyzprime)] = 1e308/3 # # avoid warning "invalid value encountered in subtract" when calculating li
    return T, xyz, xyzprime
   
   
_CCT_LUT = {'ohno2014':{},'robertson1968':{}}
def calculate_lut(lut_mode, ccts = None, cieobs = None, add_to_lut = False, wl = _WL3, 
                  cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS, cct_max = _CCT_MAX):
    """
    Function that calculates a LUT for the specified calculation method 
    for the ccts stored in '_CCT_LUT_PATH/cct_lut_cctlist_{lut_mode}.dat',
    or given as input argument.
    Calculation is performed for CMF set specified in cieobs and in the
    chromaticity diagram in cspace. 
    Adds a new (temporary) field to the nested _CCT_LUT dict.
    
    Args:
        :lut_mode:
            | string with calculation mode requiring a pre-calculated lut for speed
            | Options: 'ohno2014', 'robertson1968'
        :ccts: 
            | ndarray [Nx1] or str, optional
            | list of ccts for which to (re-)calculate the LUTs.
            | If str, ccts contains path/filename.dat to list.
        :cieobs: 
            | None or str, optional
            | str specifying cmf set.
        :wl: 
            | _WL3, optional
            | Generate luts based on Planckians with wavelengths (range). 
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
        :cct_max:
            | _CCT_MAX, optional
            | Maximum CCT, anything higher will be set to _CCT_MAX
    Returns:
        :returns: 
            | ndarray with cct and duv.
    """

    if ccts is None:
        ccts = getdata('{}cct_lut_cctlist_{}.dat'.format(_CCT_LUT_PATH, lut_mode))
    elif isinstance(ccts,str):
        ccts = getdata(ccts)       

    # get requested cmf set:
    if isinstance(cieobs,str):
        cmf = _CMF[cieobs]['bar'].copy()
    else:
        cmf = cieobs.copy()
    wl = getwlr(wl)
    cmf = cie_interp(cmf, wl, kind = 'cmf', negative_values_allowed=False)
    
    # process cspace input:
    cspace_dict = _process_cspace_input(cspace, cspace_kwargs = cspace_kwargs)
    cspace_str = cspace_dict['str']
    
    # convert to cspace based cmfs (Eq.6-7):
    Yuvbar = cspace_dict['fwtf'](cmf[1:].T) # convert to chromaticity format from xyz (cfr. cmf) format
    uvwbar = Yxy_to_xyz(Yuvbar).T # convert from chromaticity format (Vuv) to tristimulus (UVW) format and take transpose (=spectra)
    
    # calculate U,V,W (Eq. 6) and U',V',W' (Eq.10):
    Ti, UVW, UVWprime = _get_tristim_of_BB_and_BBprime(ccts, uvwbar, wl)

    # calculate li, mi:
    R = UVW.sum(axis=-1, keepdims = True)
    Rprime = UVWprime.sum(axis=-1, keepdims = True)

    # avoid div by zero:
    num = (UVWprime[:,1:2]*R - UVW[:,1:2]*Rprime) 
    denom = (UVWprime[:,:1]*R - UVW[:,:1]*Rprime)
    num[(num == 0)]+=1e-308
    denom[(denom == 0)]+=1e-308
    
    li = num/denom
    mi = -1.0/li
    uvi = UVW[:,:2]/R
    
    lut = np.hstack((Ti,uvi,mi))

    if add_to_lut == True:
        if cspace_str not in _CCT_LUT[lut_mode].keys(): _CCT_LUT[lut_mode][cspace_str] = {} # create nested dict if required
        _CCT_LUT[lut_mode][cspace_str][cieobs] = lut

    return lut 
   

def calculate_luts(lut_mode, ccts = None, wl = _WL3, 
                   save_path = _CCT_LUT_PATH, save_luts = True, add_cspace_str = True,
                   cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS):
    """
    Function that recalculates (and overwrites) LUTs in save_path
    for the ccts stored in '_CCT_LUT_PATH/cct_lut_cctlist_{lut_mode}.dat',
    or given as input argument. Calculation is performed for all CMF sets listed 
    in _CMF['types'].
    
    Args:
        :lut_mode:
            | string with calculation mode requiring a pre-calculated lut for speed
            | Options: 'ohno2014', 'robertson1968'
        :ccts: 
            | ndarray [Nx1] or str, optional
            | List of ccts for which to (re-)calculate the LUTs.
            | If str, ccts contains path/filename.dat to list.
        :wl: 
            | _WL3, optional
            | Generate luts based on Planckians with wavelengths (range). 
        :save_path:
            | _CCT_LUT_PATH, optional
            | Path to save luts to.
        :save_luts:
            | True, optional
            | If True: save luts to folder specified in save_luts.
        :add_cspace_str:
            | True, optional
            | If True: Add a string specifying the cspace to the filename of the saved luts.
            |           (if cspace is a function, 'cspace' is added)
            | Else: if add_cspace_str is itself a string, add this, else add nothing.
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
         | None
    
    Reference:
        1. `Ohno Y.  (2014)
        Practical use and calculation of CCT and Duv. 
        Leukos. 2014 Jan 2;10(1):47-55.
        <http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020>`_
    """
    luts = {}
    
    cspace_dict = _process_cspace_input(cspace, cspace_kwargs)
    cspace_string = cspace_dict['str']
    cspace_str = '_' + cspace_string if add_cspace_str else '' 

    # if add_cspace_str:
    #     cspace_str = '_' + cspace if isinstance(cspace,str) else ''
    # else:
    #     cspace_str = '_' + add_cspace_str if isinstance(add_cspace_str,str) else ''
        
    for ii, cieobs in enumerate(sorted(_CMF['types'])):
        print("Calculating CCT LUT (for {:s}) for CMF set {} & cspace {}".format(lut_mode, cieobs, cspace_string))
        cctuv = calculate_lut(lut_mode, ccts = ccts, cieobs = cieobs, add_to_lut = False, wl = wl, cspace = cspace_dict, cspace_kwargs = None)
        if save_luts:  
            pd.DataFrame(cctuv).to_csv('{}cct_lut_{:s}_{}{}.dat'.format(save_path,lut_mode,cieobs,cspace_str), header=None, index=None, float_format = '%1.9e')
        if cspace_string not in luts.keys(): luts[cspace_string] = {} # create nested dict if required
        luts[cspace_string][cieobs] = cctuv
    return luts

def initialize_lut(lut_mode):
    """ Initialize LUT dicts """
 
    if _CCT_LUT_CALC == True:
        _CCT_LUT[lut_mode] = calculate_luts(lut_mode, wl = _WL3, cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS)  
        
    # Initialize _CCT_LUT dict:
    cspace_dict = _process_cspace_input(_CCT_CSPACE, _CCT_CSPACE_KWARGS)
    cspace_string = cspace_dict['str']
    cspace_str = '_' + cspace_string  
 
    try:
        _CCT_LUT[lut_mode][cspace_string] = dictkv(keys = sorted(_CMF['types']), values = [getdata('{}cct_lut_{}_{}{}.dat'.format(_CCT_LUT_PATH,lut_mode,sorted(_CMF['types'])[i],cspace_str),kind='np') for i in range(len(_CMF['types']))],ordered = False)
           
    except:
        calculate_luts(lut_mode, wl = _WL3, cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS)  
        _CCT_LUT[lut_mode][cspace_string] = dictkv(keys = sorted(_CMF['types']), values = [getdata('{}cct_lut_{}_{}{}.dat'.format(_CCT_LUT_PATH,lut_mode,sorted(_CMF['types'])[i],cspace_str),kind='np') for i in range(len(_CMF['types']))],ordered = False)

# Initizalize all LUT modes:
for lut_mode in _CCT_LUT.keys(): 
    initialize_lut(lut_mode)

   
#------------------------------------------------------------------------------
# CCT calculation methods:
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Mcamy, 1992:
#-------------
def xyz_to_cct_mcamy(xyzw):
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


#------------------------------------------------------------------------------
# Hernandez-Andres, 1999:
#------------------------
def xyz_to_cct_HA(xyzw, verbosity = 1):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT). 
    
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

#------------------------------------------------------------------------------
# Search algorithms: 
#------------------------------------------------------------------------------      
def _process_cct_mk_search_lists(cct_search_list = None, 
                                 mk_search_list = None, 
                                 upper_cct_max = _CCT_MAX):
    """ 
    Get cct_search_list and mk_search_list.
    
    Args:
        :cct_search_list:
            | None,optional
            | Options: 'bf-search','pw-linear','zhang2019','robertson1968'
            | None defaults to _CCT_SEARCH_METHOD
        :mk_search_list:
            | None, optional
            | If not None: overrides cct_search_list
            
    Returns:
        :cct_search_list,mk_search_list:
            | ndarrays
    """
    # Get search lists:
    if mk_search_list is not None:
        if isinstance(mk_search_list,str):
            cct_search_list = mk_search_list
        else:
            cct_search_list = cct_to_mired(mk_search_list)
    
    if cct_search_list is None:
        cct_search_list = _CCT_SEARCH_METHOD
        
    if isinstance(cct_search_list,str):
        if ('bf-search' in cct_search_list.lower()):
            cct_search_list = _CCT_SEARCH_LIST_BRUTEFORCE
            mk_search_list = _MK_SEARCH_LIST_BRUTEFORCE
        elif cct_search_list.lower() == 'pw-linear':
            cct_search_list = _CCT_SEARCH_LIST_PW_LIN
            mk_search_list = _MK_SEARCH_LIST_PW_LIN
        elif ('zhang' in cct_search_list.lower()):
            mk_search_list = _MK_SEARCH_LIST_ZHANG2019
            cct_search_list = _CCT_SEARCH_LIST_ZHANG2019
        elif ('robertson' in cct_search_list.lower()):
            mk_search_list = _MK_SEARCH_LIST_ROBERTSON1968
            cct_search_list = _CCT_SEARCH_LIST_ROBERTSON1968
        else:
            raise Exception('cct_search_list = {:s} not supported'.format(cct_search_list))
    else:
        if mk_search_list is None: 
            mk_search_list = cct_to_mired(cct_search_list)

    if upper_cct_max is not None: 
        mk_search_list = mk_search_list[cct_search_list<=upper_cct_max][:,None]
        cct_search_list = cct_search_list[cct_search_list<=upper_cct_max][:,None]

    return cct_search_list, mk_search_list

def xyz_to_cct_search(xyzw, cieobs = _CIEOBS, out = 'cct', wl = None, mode = 'zhang2019',
                      rtol = 1e-5, atol = 0.1, force_tolerance = True,
                      cct_search_list = None, mk_search_list = None, 
                      split_zhang_calculation_at_N = 100, upper_cct_max = _CCT_MAX,
                      cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS, 
                      approx_cct_temp = True, lut = None):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv(distance above (> 0) or below ( < 0) the Planckian locus) by a 
    brute-force (robust or fast) search method or Zhang's (2019) golden-ratio search method
    or Robertson's (1968) method. 
    
    Wrapper around xyz_to_cct_search_bf_robust(), xyz_to_cct_search_bf_fast(),
    xyz_to_cct_search_zhang2019() and xyz_to_cct_search_robertson1968().
    
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
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
        :mode:
            | 'zhang2019', optional
            | Options:
            |   -'bf-robust': use xyz_to_cct_search_bf_robust()
            |   -'bf-fast': use xyz_to_cct_search_bf_fast()
            |   -'zhang2019': use xyz_to_cct_search_zhang2019()
            |   -'robertson1968': use xyz_to_cct_search_robertson2019()
        :rtol: 
            | 1e-5, float, optional
            | Stop search when a relative cct tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | search process and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop search when an absolute cct tolerance (K) is reached.
        :force_tolerance:
            | True, optional
            | Accuracy of the calculations depends on the CCT of test source 
            |   and the location and spacing of initial CCTs used to start the search,
            |   or the LUT based method.
            | If True:  search process will continue until the tolerance is
            |           reached for ALL sources in xyzw! 
            | If False: search process might stop early (depending on the chosen mode).
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit search to this cct.
            | Not used in 'robertson2019' !
        :cct_search_list:
            | None, optional
            | List of ccts to obtain a first guess for the cct of the input xyz
            | for the 'brute-force-search-robust', 'zhang2019', 'robertson1968' fallback methods, or
            | when HA estimation fails in the 'brute-force-search-fast' fallback algorithm 
            | due to out-of-range ccts.
            | Options:
            |   - 'default' or None: defaults to the mode in _CCT_SEARCH_METHOD.
            |   - 'bf-search': defaults to _CCT_SEARCH_LIST_BRUTEFORCE
            |   - 'zhang2019': defaults to _CCT_SEARCH_LIST_ZHANG2019
            |   - 'pw_linear': defaults to _CCT_SEARCH_PW_LIN
            |   - 'robertson1968': defaults to _CCT_SEARCH_LIST_ROBERTSON1968
        :mk_search_list:
            | None, optional
            | Input cct_search_list directly in MK (mired) scale.
            | None: does nothing, but when not None input overwrites cct_search_list !
        :split_calculation_at_N:
            | 100, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
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
        :approx_cct_temp: 
            | True, optional
            | If True: use xyz_to_cct_HA() to get a first estimate of cct to 
            |          speed up search.
        :lut:
            | None, optional
            | LUT for mode == 'robertson1968'. If None: use _CCT_LUT['robertson1968']

            
    Returns:
        :returns: 
            | ndarray with:
            |    cct: out == 'cct' (or 1)
            |    duv: out == 'duv' (or -1)
            |    cct, duv: out == 'cct,duv' (or 2)
            |    [cct,duv]: out == "[cct,duv]" (or -2) 
    
    """
    if (mode == 'brute-force-search-robust') | (mode == 'brute-force-robust') | (mode == 'bf-robust'):
        return xyz_to_cct_search_bf_robust(xyzw, cieobs = cieobs, out = out, wl = wl, 
                                           rtol = rtol, atol = atol, force_tolerance = force_tolerance, 
                                           upper_cct_max = upper_cct_max, 
                                           cct_search_list = cct_search_list, mk_search_list = mk_search_list,
                                           cspace = cspace, cspace_kwargs = cspace_kwargs)
    
    elif (mode == 'brute-force-search-fast') | (mode == 'brute-force-fast') | (mode == 'bf-fast') | (mode == 'search'): # 'search' for legacy reasons
        return xyz_to_cct_search_bf_fast(xyzw, cieobs = cieobs, out = out, wl = wl, 
                                         rtol = rtol, atol = atol, force_tolerance = force_tolerance, 
                                         upper_cct_max = upper_cct_max, 
                                         cct_search_list = cct_search_list, mk_search_list = mk_search_list,
                                         cspace = cspace, cspace_kwargs = cspace_kwargs, 
                                         approx_cct_temp = approx_cct_temp)
    
    elif (mode == 'zhang2019'):
        return xyz_to_cct_search_zhang2019(xyzw, cieobs = cieobs, out = out, wl = wl, 
                                       rtol = rtol, atol = atol, force_tolerance = force_tolerance, 
                                       upper_cct_max = upper_cct_max, split_calculation_at_N = split_zhang_calculation_at_N, 
                                       cct_search_list = cct_search_list, mk_search_list = mk_search_list,
                                       cspace = cspace, cspace_kwargs = cspace_kwargs)
    
    elif (mode == 'robertson1968'):
        return xyz_to_cct_search_robertson1968(xyzw, cieobs = cieobs, out = out, wl = wl, 
                                               rtol = rtol, atol = atol, force_tolerance = force_tolerance,
                                               upper_cct_max = upper_cct_max, lut = lut,
                                               cct_search_list = cct_search_list, mk_search_list = mk_search_list,
                                               cspace = cspace, cspace_kwargs = cspace_kwargs)

    else:
        raise Exception('Unrecognize cct search mode: {:s}'.format(mode))
                
#------------------------------------------------------------------------------
# Brute force search
#------------------------------------------------------------------------------
def _find_closest_ccts(uvw, cieobs = _CIEOBS, ccts = None, wl = _WL3,
                       cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS):
    """
    Find closest cct from a list and the two surrounding ccts. (helper function 
    for the brute-force search methods)
    """
    if ccts is None:
        raise Exception('Ndarray with ccts must be supplied in _find_closes_ccts() helper function.')
    
    max_cct = ccts[-1]
    
    cspace_dict = _process_cspace_input(cspace, cspace_kwargs)
    
    uv = cspace_dict['fwtf'](spd_to_xyz(cri_ref(ccts, wl3 = wl, ref_type = ['BB']), cieobs = cieobs))[:,1:3]
    # uv = np.empty((ccts.shape[0],2)) 
    # for i,cct in enumerate(ccts):
    #   uv[i,:] = cspace_dict['fwtf'](spd_to_xyz(blackbody(cct, wl3 = wl), cieobs = cieobs))[:,1:3]
    #uv[:,1] *= (2.0/3.0) # get CIE 1960 v
    
    dc2=((uv[...,None]-uvw.T[None,...])**2).sum(axis=1)
    q = dc2.argmin(axis=0)

    if q.shape[0]>1:
        dv = ((uv[q,1]-uvw[...,1]))
        ccts_i = ccts[np.vstack((q,q))]
        ccts_i[:,q==0] = ccts[np.vstack((q,q+1))[:,q==0]]
        ccts_i[:,q==len(ccts)-1] = ccts[np.vstack((q,q-1))[:,q==len(ccts)-1]]
        ccts_i[1,(q==len(ccts)-1) & (dv > 0)] = max_cct
        ccts_i[:,(q>0) & (q<len(ccts)-1)] = ccts[np.vstack((q-1,q+1))[:,(q>0) & (q<len(ccts)-1)]]
    else:
        q = q[0]
        if q == 0:
            ccts_i = ccts[[q,q+1]]
        elif q == (len(ccts) - 1):
            if (uv[q,1] - uvw[:,1]) <= 0: # point lies within last bin
                ccts_i = ccts[[q-1,q]]
            else:
                ccts_i = np.hstack((ccts[q],max_cct))
        else:
            ccts_i = ccts[[q-1,q+1]]
        ccts_i = np2d(ccts_i).T

    return ccts_i.mean(axis=0,keepdims=True).T, ccts_i.T

def xyz_to_cct_search_bf_robust(xyzw, cieobs = _CIEOBS, out = 'cct', wl = None, 
                                rtol = 1e-5, atol = 0.1, force_tolerance = True, 
                                upper_cct_max = _CCT_MAX, cct_search_list = 'bf-search', mk_search_list = None,
                                cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv(distance above (> 0) or below ( < 0) the Planckian locus) by a 
    brute-force search. 

    | The algorithm uses an approximate cct_temp as starting point then 
    | constructs, a 4-step section of the blackbody (Planckian) locus on which 
    | to find the minimum distance to the 1960 uv (or other) chromaticity of 
    | the test source. The approximate starting point is found by generating 
    | the uv chromaticity values of a set blackbody radiators spread across the
    | locus in a ~ 50 K to ~ _CCT_MAX K range (larger CCT's cause instability of the 
    | chromaticity points due to floating point errors), looking for the closest
    | blackbody radiator and then calculating the mean of the two surrounding ones.
    | The default cct search list is given in _CCT_SEARCH_LIST_BRUTEFORCE.

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
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
        :rtol: 
            | 1e-5, float, optional
            | Stop search when a relative cct tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | search process and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop search when an absolute cct tolerance (K) is reached.
        :force_tolerance:
            | True, optional
            | Accuracy of the calculations depends on the CCT of test source 
            |   and the location and spacing of initial CCTs used to start the search,
            |   or the LUT based method.
            | If True:  search process will continue until the tolerance is
            |           reached for ALL sources in xyzw! 
            | If False: search process might stop early (depending on the chosen mode).
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit search to this cct.
        :cct_search_list:
            | None, optional
            | List of ccts to obtain a first guess for the cct of the input xyz
            | for the 'brute-force-search-robust', 'zhang2019', 'robertson1968' fallback methods, or
            | when HA estimation fails in the 'brute-force-search-fast' fallback algorithm 
            | due to out-of-range ccts.
            | Options:
            |   - 'default' or None: defaults to the mode in _CCT_SEARCH_METHOD.
            |   - 'bf-search': defaults to _CCT_SEARCH_LIST_BRUTEFORCE
            |   - 'zhang2019': defaults to _CCT_SEARCH_LIST_ZHANG2019
            |   - 'pw_linear': defaults to _CCT_SEARCH_PW_LIN
            |   - 'robertson1968': defaults to _CCT_SEARCH_LIST_ROBERTSON1968
        :mk_search_list:
            | None, optional
            | Input cct_search_list directly in MK (mired) scale.
            | None: does nothing, but when not None input overwrites cct_search_list !
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
            | ndarray with:
            |    cct: out == 'cct' (or 1)
            |    duv: out == 'duv' (or -1)
            |    cct, duv: out == 'cct,duv' (or 2)
            |    [cct,duv]: out == "[cct,duv]" (or -2) 
    
    Notes:
        1. This function is more accurate, but slower than xyz_to_cct_ohno!
        Note that cct must be between 50 K - _CCT_MAX K 
        (very large cct take a long time!!!)
    """

    xyzw = np2d(xyzw)   

    if len(xyzw.shape)>2:
        raise Exception('xyz_to_cct_search_bf_robust(): Input xyzw.shape must be <= 2 !')
       
    cspace_dict = _process_cspace_input(cspace, cspace_kwargs)     
   
    # get 1960 u,v of test source:
    Yuvt = cspace_dict['fwtf'](np.squeeze(xyzw)) # remove possible 1-dim 
    ut = Yuvt[:,1,None] # get CIE 1960 u
    vt = Yuvt[:,2,None] # get CIE 1960 v

    # Initialize arrays:
    ccts = np.zeros((xyzw.shape[0],1))
    ccts.fill(np.nan)
    duvs = ccts.copy()
    
    # get cct_search_list:
    cct_search_list, _ = _process_cct_mk_search_lists(cct_search_list =cct_search_list, 
                                                      mk_search_list = mk_search_list,
                                                      upper_cct_max=upper_cct_max)
        
    #calculate preliminary estimates within range in cct_search_list:
    ccts_est, cctranges = _find_closest_ccts(np.hstack((ut,vt)), cieobs = cieobs, 
                                             ccts = cct_search_list[:,0], wl = wl,
                                             cspace = cspace_dict, cspace_kwargs = None)

    cct_scale_fun = lambda x: x
    cct_scale_ifun = lambda x: x
    
    # Loop through all ccts:        
    for i in range(xyzw.shape[0]):

        #initialize CCT search parameters:
        cct = np.nan
        duv = np.nan
        ccttemp = ccts_est[i,0].copy()
        dT = np.abs(np.diff(cctranges[i,:]))/2     
        
        nsteps = 4 
        signduv = 1.0 
        delta_cct = dT
        reached_CCT_MAX = False

        while ((((delta_cct*2/ccttemp) >= rtol) & (delta_cct*2 >= atol)) & (reached_CCT_MAX == False)) & (force_tolerance == True):# keep converging on CCT 

            #generate range of ccts:
            ccts_i = cct_scale_ifun(np.linspace(cct_scale_fun(ccttemp)-dT,cct_scale_fun(ccttemp)+dT,nsteps+1))
            ccts_i[ccts_i < 100.0] = 100.0 # avoid nan's in calculation
            reached_CCT_MAX = True if (ccts_i>_CCT_MAX).any() else False
            ccts_i[ccts_i > _CCT_MAX] = _CCT_MAX

            # Generate BB:
            BB = cri_ref(ccts_i,wl3 = wl,ref_type = ['BB'],cieobs = cieobs)
            
            # Calculate xyz:
            xyz = spd_to_xyz(BB, cieobs = cieobs)
    
            # Convert to CIE 1960 u,v:
            Yuv = cspace_dict['fwtf'](np.squeeze(xyz)) # remove possible 1-dim 
            u = Yuv[:,1,None] # get CIE 1960 u
            v = Yuv[:,2,None] # get CIE 1960 v
            
            # Calculate distance between list of uv's and uv of test source:
            dc = ((ut[i] - u)**2 + (vt[i] - v)**2)**0.5

            if np.isnan(dc.min()) == False:
                q = dc.argmin()
                if np.size(q) > 1: #to minimize calculation time: only calculate median when necessary
                    cct = np.median(ccts_i[q,0])
                    duv = np.median(dc[q,0])
                    q = np.median(q)
                    q = int(q) #must be able to serve as index
                else:
                     cct = ccts_i[q]
                     duv = dc[q]

                if (q == 0):
                    ccttemp = cct_scale_ifun(np.array(cct_scale_fun(cct)) + 2*dT/nsteps)[0]
                    delta_cct = abs(cct - ccttemp)
                    #dT = 2.0*dT/nsteps
                    continue # look in higher section of planckian locus
                    
                if (q == np.size(ccts_i)-1):
                    ccttemp = cct_scale_ifun(np.array(cct_scale_fun(cct)) - 2*dT/nsteps)[0]
                    delta_cct = abs(cct - ccttemp)
                    #dT = 2.0*dT/nsteps
                    continue # look in lower section of planckian locus
                
                if (q > 0) & (q < (np.size(ccts_i)-1)):
                    dT = 2*dT/nsteps
                    
                    # get Duv sign:
                    d_p1m1 = ((u[q+1] - u[q-1])**2.0 + (v[q+1] - v[q-1])**2.0)**0.5
    
                    x = (dc[q-1]**2.0 - dc[q+1]**2.0 + d_p1m1**2.0)/2.0*d_p1m1
                    vBB = v[q-1] + ((v[q+1] - v[q-1]) * (x / d_p1m1))
                    signduv = np.sign(vt[i]-vBB)

                #calculate max. difference with previous intermediate solution:
                delta_cct = dT
               
                # ccttemp = (np.array(cct) + ccttemp)/2 #%set new intermediate CCT
                ccttemp = np.array(cct) #%set new intermediate CCT

            else:
                ccttemp = np.nan 
                cct = np.nan
                duv = np.nan
              
        duvs[i] = signduv*abs(duv)
        ccts[i] = cct + (np.pi*reached_CCT_MAX/10) # to signal out-of-rangness

    # Regulate output:
    if (out == 'cct') | (out == 1):
        return np2d(ccts)
    elif (out == 'duv') | (out == -1):
        return np2d(duvs)
    elif (out == 'cct,duv') | (out == 2):
        return np2d(ccts), np2d(duvs)
    elif (out == "[cct,duv]") | (out == -2):
        return np.hstack((ccts,duvs))       


def xyz_to_cct_search_bf_fast(xyzw, cieobs = _CIEOBS, out = 'cct', wl = None, 
                               rtol = 1e-5, atol = 0.1, force_tolerance = True, 
                               upper_cct_max = _CCT_MAX, cct_search_list = 'bf-search', mk_search_list = None, 
                               cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                               approx_cct_temp = True):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv(distance above (> 0) or below ( < 0) the Planckian locus) by a 
    brute-force search. 

    | The algorithm uses an approximate cct_temp (HA approx., see xyz_to_cct_HA) 
    |  as starting point or uses the middle of the allowed cct-range 
    |  (~ 100 K - _CCT_MAX K, higher causes overflow) on a log-scale, then constructs 
    |  a 4-step section of the blackbody (Planckian) locus on which to find the
    |  minimum distance to the 1960 uv (or other) chromaticity of the test source.
    | If HA fails then another approximate starting point is found by generating 
    | the uv chromaticity values of a set blackbody radiators spread across the
    | locus in a ~ 50 K to _CCT_MAX K range (larger CCT's cause instability of the 
    | chromaticity points due to floating point errors), looking for the closest
    | blackbody radiator and then calculating the mean of the two surrounding ones.
    | The default cct list is given in _CCT_SEARCH_LIST_BRUTEFORCE.

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
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
        :rtol: 
            | 1e-5, float, optional
            | Stop search when a relative cct tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | search process and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop search when an absolute cct tolerance (K) is reached.
        :force_tolerance:
            | True, optional
            | Accuracy of the calculations depends on the CCT of test source 
            |   and the location and spacing of initial CCTs used to start the search,
            |   or the LUT based method.
            | If True:  search process will continue until the tolerance is
            |           reached for ALL sources in xyzw! 
            | If False: search process might stop early (depending on the chosen mode).
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit search to this cct.
        :cct_search_list:
            | None, optional
            | List of ccts to obtain a first guess for the cct of the input xyz
            | for the 'brute-force-search-robust', 'zhang2019', 'robertson1968' fallback methods, or
            | when HA estimation fails in the 'brute-force-search-fast' fallback algorithm 
            | due to out-of-range ccts.
            | Options:
            |   - 'default' or None: defaults to the mode in _CCT_SEARCH_METHOD.
            |   - 'bf-search': defaults to _CCT_SEARCH_LIST_BRUTEFORCE
            |   - 'zhang2019': defaults to _CCT_SEARCH_LIST_ZHANG2019
            |   - 'pw_linear': defaults to _CCT_SEARCH_PW_LIN
            |   - 'robertson1968': defaults to _CCT_SEARCH_LIST_ROBERTSON1968
        :mk_search_list:
            | None, optional
            | Input cct_search_list directly in MK (mired) scale.
            | None: does nothing, but when not None input overwrites cct_search_list !
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
        :approx_cct_temp: 
            | True, optional
            | If True: use xyz_to_cct_HA() to get a first estimate of cct to 
            |          speed up search.
            
    Returns:
        :returns: 
            | ndarray with:
            |    cct: out == 'cct' (or 1)
            |    duv: out == 'duv' (or -1)
            |    cct, duv: out == 'cct,duv' (or 2)
            |    [cct,duv]: out == "[cct,duv]" (or -2) 
    
    Notes:
        This program is more accurate, but slower than xyz_to_cct_ohno!
        Note that cct must be between 1e3 K - 1e20 K 
        (very large cct take a long time!!!)
    """
 
    xyzw = np2d(xyzw)   
    
    if len(xyzw.shape)>2:
        raise Exception('xyz_to_cct_search_bf_fast(): Input xyzw.shape must be <= 2 !')
    
    cspace_dict = _process_cspace_input(cspace, cspace_kwargs)

    # get 1960 u,v of test source:
    Yuvt = cspace_dict['fwtf'](np.squeeze(xyzw)) # remove possible 1-dim 
    ut = Yuvt[:,1,None] #.take([1],axis = axis_of_v3t) # get CIE 1960 u
    vt = Yuvt[:,2,None] #.take([2],axis = axis_of_v3t) # get CIE 1960 v

    # Initialize arrays:
    ccts = np.zeros((xyzw.shape[0],1))
    ccts.fill(np.nan)
    duvs = ccts.copy()

    # get cct_searh_list:
    cct_search_list, _ = _process_cct_mk_search_lists(cct_search_list =cct_search_list, 
                                                      mk_search_list = mk_search_list,
                                                      upper_cct_max=upper_cct_max)
    cct_search_list = cct_search_list[:,0]
    
    # calculate preliminary solution(s):
    if (approx_cct_temp == True):
        ccts_est = xyz_to_cct_HA(xyzw, verbosity = 0)
        procent_estimates = np.array([[3000.0, 100000.0,0.05],[100000.0,200000.0,0.1],[200000.0,300000.0,0.25],[300000.0,400000.0,0.4],[400000.0,600000.0,0.4],[600000.0,800000.0,0.4],[800000.0,np.inf,0.25]])
        ccts_est[np.isnan(ccts_est)] =  -2 # recode to avoid "RuntimeWarning: invalid value encountered in less"
        if ((np.isnan(ccts_est).any()) | (ccts_est == -2).any() | (ccts_est == -1).any()) | ((ccts_est < procent_estimates[0,0]).any() | (ccts_est > procent_estimates[-2,1]).any()):
            
            #calculate preliminary estimates in 50 K to _CCT_MAX range or whatever is given in cct_search_list:
            ccts_est, cct_ranges = _find_closest_ccts(np.hstack((ut,vt)), cieobs = cieobs, wl = wl, ccts = cct_search_list, 
                                                      cspace = cspace_dict, cspace_kwargs = None)
            not_in_estimator_range = True
            ccts_est[(ccts_est>upper_cct_max)[:,0],:] = upper_cct_max

        else:
            cct_ranges = None
            not_in_estimator_range = False

    else:
        upper_cct = np.array(upper_cct_max)
        lower_cct = np.array(10.0**2)
        cct_scale_fun = lambda x: np.log10(x)
        cct_scale_ifun = lambda x: np.power(10.0,x)
        dT = (cct_scale_fun(upper_cct) - cct_scale_fun(lower_cct))/2
        ccttemp = np.array([cct_scale_ifun(cct_scale_fun(lower_cct) + dT)])
        ccts_est = np2d(ccttemp*np.ones((xyzw.shape[0],1)))
        dT_approx_cct_False = dT.copy()
        not_in_estimator_range = False


    # Loop through all ccts:        
    for i in range(xyzw.shape[0]):

        #initialize CCT search parameters:
        cct = np.nan
        duv = np.nan
        ccttemp = ccts_est[i,0].copy()

        # Take care of (-1, NaN)'s from xyz_to_cct_HA signifying (CCT < lower, CCT > upper) bounds:
        approx_cct_temp_temp = approx_cct_temp
        if (approx_cct_temp == True):
            cct_scale_fun = lambda x: x
            cct_scale_ifun = lambda x: x
            if  (not_in_estimator_range == False) & ((ccttemp != -1) & (np.isnan(ccttemp) == False)) & (ccttemp >= procent_estimates[0,0]) & (ccttemp <= procent_estimates[-2,1]): # within validity range of CCT estimator-function
                for ii in range(procent_estimates.shape[0]):
                    if (ccttemp >= (1.0-0.05*(ii == 0))*procent_estimates[ii,0]) & (ccttemp < (1.0+0.05*(ii == 0))*procent_estimates[ii,1]):
                        procent_estimate = procent_estimates[ii,2]
                        break

                dT = np.multiply(ccttemp,procent_estimate) # determines range around CCTtemp (25% around estimate) or 100 K

            else:
                dT = np.abs(np.diff(cct_ranges[i,:]))/2

            delta_cct = dT
        else:
            dT = dT_approx_cct_False
            delta_cct = cct_scale_ifun(cct_scale_fun(ccttemp) + dT) - ccttemp
      
        nsteps = 4 
        signduv = 1.0 
        reached_CCT_MAX = False

        rtols = np.ones((5,))*_CCT_MAX
        cnt = 0

        while ((((delta_cct*2) >= atol) & ((delta_cct*2/ccttemp) >= rtol)) & (reached_CCT_MAX == False)) & (force_tolerance == True):# keep converging on CCT 

            #generate range of ccts:
            ccts_i = cct_scale_ifun(np.linspace(cct_scale_fun(ccttemp)-dT,cct_scale_fun(ccttemp)+dT,nsteps+1))
            ccts_i[ccts_i < 100.0] = 100.0 # avoid nan's in calculation
            reached_CCT_MAX = True if ((ccts_i>_CCT_MAX).any() & (approx_cct_temp == True)) else False
            ccts_i[ccts_i > _CCT_MAX] = _CCT_MAX # avoid nan's in calculation

            # Generate BB:
            BB = cri_ref(ccts_i,wl3 = wl,ref_type = ['BB'],cieobs = cieobs)
            
            # Calculate xyz:
            xyz = spd_to_xyz(BB,cieobs = cieobs)
    
            # Convert to CIE 1960 u,v:
            Yuv = cspace_dict['fwtf'](np.squeeze(xyz)) # remove possible 1-dim 
            u = Yuv[:,1,None] # get CIE 1960 u
            v = Yuv[:,2,None] # get CIE 1960 v
            
            # Calculate distance between list of uv's and uv of test source:
            dc = ((ut[i] - u)**2 + (vt[i] - v)**2)**0.5
            
            if np.isnan(dc.min()) == False:
                q = dc.argmin()
                
                if np.size(q) > 1: #to minimize calculation time: only calculate median when necessary
                    cct = np.median(ccts_i[q])
                    duv = np.median(dc[q])
                    q = np.median(q)
                    q = int(q) #must be able to serve as index
                else:
                     cct = ccts_i[q]
                     duv = dc[q]

                if (q == 0):
                    ccttemp = cct_scale_ifun(np.array(cct_scale_fun(cct)) + 2*dT/nsteps)
                                        
                    delta_cct = abs(cct - ccttemp)
                    if cnt > 4: # to ensure loop breaks when chromaticity is outside of valid range
                        if np.diff(rtols).mean() < rtol:
                            cnt = 0
                            break
                    cnt += 1
                    
                    continue # look in higher section of planckian locus
                    
                if (q == np.size(ccts_i)-1):
                    ccttemp = cct_scale_ifun(np.array(cct_scale_fun(cct)) - 2*dT/nsteps)
                    #dT = 2.0*dT/nsteps
                    
                    delta_cct = abs(cct - ccttemp)
                    if cnt > 4: # to ensure loop breaks when chromaticity is outside of valid range
                        if np.diff(rtols).mean() < rtol:
                            cnt = 0
                            break
                    cnt += 1
                    continue # look in lower section of planckian locus
                    
                if (q > 0) & (q < np.size(ccts_i)-1):
                    dT = 2*dT/nsteps
                    # get Duv sign:
                    d_p1m1 = ((u[q+1] - u[q-1])**2.0 + (v[q+1] - v[q-1])**2.0)**0.5
    
                    x = (dc[q-1]**2.0 - dc[q+1]**2.0 + d_p1m1**2.0)/2.0*d_p1m1
                    vBB = v[q-1] + ((v[q+1] - v[q-1]) * (x / d_p1m1))
                    signduv =np.sign(vt[i]-vBB)
            
                #calculate difference with previous intermediate solution:
                delta_cct = dT#abs(cct - ccttemp)

                ccttemp = np.array(cct) #set new intermediate CCT
                approx_cct_temp = approx_cct_temp_temp
                if (cnt > 4): cnt = 0  
                rtols[cnt] = delta_cct*2/ccttemp
            else:
                ccttemp = np.nan 
                cct = np.nan
                duv = np.nan

        duvs[i] = signduv*abs(duv)
        ccts[i] = cct + (np.pi*reached_CCT_MAX/10)
    
    # Regulate output:
    if (out == 'cct') | (out == 1):
        return np2d(ccts)
    elif (out == 'duv') | (out == -1):
        return np2d(duvs)
    elif (out == 'cct,duv') | (out == 2):
        return np2d(ccts), np2d(duvs)
    elif (out == "[cct,duv]") | (out == -2):
        return np.hstack((ccts,duvs))

#------------------------------------------------------------------------------
# Zhang, 2019
#------------------------------------------------------------------------------
def xyz_to_cct_search_zhang2019(xyzw, cieobs = _CIEOBS, out = 'cct', wl  = None, 
                                rtol = 1e-5, atol = 0.1, force_tolerance = True, split_calculation_at_N = 100,
                                cct_search_list = 'bf-search', mk_search_list = None, upper_cct_max = _CCT_MAX,
                                cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS):
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
            | Accuracy of the calculations depends on the CCT of test source 
            |   and the location and spacing of initial CCTs used to start the search,
            |   or the LUT based method.
            | If True:  search process will continue until the tolerance is
            |           reached for ALL sources in xyzw! 
            | If False: search process might stop early (depending on the chosen mode).
        :split_calculation_at_N:
            | 100, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit golden-ratio search to this cct.
        :cct_search_list:
            | None, optional
            | List of ccts to obtain a first guess for the cct of the input xyz.
            | Options:
            |   - 'default' or None: defaults to the mode in _CCT_SEARCH_METHOD.
            |   - 'bf-search': defaults to _CCT_SEARCH_LIST_BRUTEFORCE
            |   - 'zhang2019': defaults to _CCT_SEARCH_LIST_ZHANG2019
            |   - 'pw_linear': defaults to _CCT_SEARCH_PW_LIN
            |   - 'robertson1968': defaults to _CCT_SEARCH_LIST_ROBERTSON1968
        :mk_search_list:
            | None, optional
            | Input cct_search_list directly in MK (mired) scale.
            | None: does nothing, but when not None input overwrites cct_search_list !
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
            | ndarray with:
            |    cct: out == 'cct' (or 1)
            |    duv: out == 'duv' (or -1)
            |    cct, duv: out == 'cct,duv' (or 2)
            |    [cct,duv]: out == "[cct,duv]" (or -2) 
    
    References:
        1. `Zhang, F. (2019). 
        High-accuracy method for calculating correlated color temperature with 
        a lookup table based on golden section search. 
        Optik, 193, 163018. 
        <https://doi.org/https://doi.org/10.1016/j.ijleo.2019.163018>`_
    """
    
    xyzw = np2d(xyzw)   
    
    if len(xyzw.shape)>2:
        raise Exception('xyz_to_cct_search(): Input xyzw.shape must be <= 2 !')
    
    cspace_dict = _process_cspace_input(cspace, cspace_kwargs)
    
    # get search_lists:
    cct_search_list, mk_search_list = _process_cct_mk_search_lists(cct_search_list = cct_search_list, 
                                                                   mk_search_list = mk_search_list, 
                                                                   upper_cct_max = upper_cct_max)
    
    # dirty solution to code that was originally programmed for vectors:
    cct_search_list, mk_search_list = cct_search_list[:,0], mk_search_list[:,0]
    
    # get BB radiator spectra:
    BB = cri_ref(cct_search_list, ref_type = ['BB'], wl3 = wl)
    
    # convert BB spectra to xyz:
    xyzBB = spd_to_xyz(BB, cieobs = cieobs, relative = True)
        
    # get cspace coordinates of BB xyz:
    uvBB = cspace_dict['fwtf'](xyzBB)[...,1:]
    
    # # store cct, MK and uv in LUT:
    # lut = np.vstack((cct_search_list,mk_search_list, uvBB.T)).T 
    # lut = np.vstack((lut[0],lut,lut[-1]))
    mk_search_list = np.hstack((mk_search_list[0],mk_search_list,mk_search_list[-1]))
    
    # get cspace coordinates of input xyzw:
    uvw = cspace_dict['fwtf'](xyzw)[...,1:] 
    
    # prepare split of input data to speed up calculation:
    n = xyzw.shape[0]
    ccts = np.zeros((n,))
    duvs = np.zeros((n,))
    n_i = split_calculation_at_N if split_calculation_at_N is not None else n
    N_i = n//n_i + 1*((n%n_i)>0)
    
    # loop of splitted data:
    for i in range(N_i):
        
        # get data for split i:
        uv = uvw[n_i*i:n_i*i+n_i] if (i < (N_i-1)) else uvw[n_i*i:]

        # find distance in UCD of BB to input:
        uBB, vBB = uvBB[...,0:1],uvBB[...,1:2]
        u, v = uv[...,0:1],uv[...,1:2]
        DEuv = ((uBB - u.T)**2 + (vBB - v.T)**2)**0.5
        
        # find minimum in distance table:
        p0 = DEuv.argmin(axis=0) + 1 # + 1 to index in lut
        
        # get RTm-1 (RTl) and RTm+1 (RTr):
        RTl = mk_search_list[p0 - 1]
        RTr = mk_search_list[p0 + 1]
        
        # calculate RTa, RTb:
        s = (5**0.5 - 1)/2
        RTa = RTl + (1.0 - s) * (RTr - RTl)
        RTb = RTl + s * (RTr - RTl)
        
        while True:
            # calculate BBa BBb:
            BBab = cri_ref(np.hstack([cct_to_mired(RTa), cct_to_mired(RTb)]), ref_type = ['BB'], wl3 = wl)
            
            # calculate xyzBBab:
            xyzBBab = spd_to_xyz(BBab, cieobs = cieobs, relative = True)
        
            # get cspace coordinates of BB and input xyz:
            uvBBab = cspace_dict['fwtf'](xyzBBab)[...,1:]
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
            RTa[c] = RTl[c] + (1.0 - s) * (RTr[c] - RTl[c])
            
            # when DEuv_a >= DEuv_b:
            RTl[~c] = RTa[~c]
            RTa[~c] = RTb[~c]
            DEuv_a[~c] = DEuv_b[~c]
            RTb[~c] = RTl[~c] + s * (RTr[~c] - RTl[~c])
            
            # Calculate CCTs from RTa and RTb:
            ccts_a, ccts_b = cct_to_mired(RTa), cct_to_mired(RTb)
            ccts_i = cct_to_mired((RTa+RTb)/2)
            dccts = np.abs(ccts_a - ccts_b)
            if ((dccts <= atol).all() | ((dccts/ccts_i) <= rtol).all()) | (force_tolerance == False):
                break
    
        # Get duv: 
        BB_i = cri_ref(ccts_i, ref_type = ['BB'], wl3 = wl)
        xyzBB_i = spd_to_xyz(BB_i, cieobs = cieobs, relative = True)
        uvBB_i = cspace_dict['fwtf'](xyzBB_i)[...,1:]
        uBB_i, vBB_i = uvBB_i[...,0:1], uvBB_i[...,1:2]
        uBB_c, vBB_c = (u - uBB_i), (v - vBB_i)
        duvs_i = (uBB_c**2 + vBB_c**2)**0.5
        
        # find sign of duv:
        theta = math.positive_arctan(uBB_c,vBB_c,htype='deg')
        theta[theta>180] = theta[theta>180] - 360
        # lx.plotSL(cieobs=cieobs,cspace='Yuv60')
        # plt.plot(u,v,'ro')
        # plt.plot(uBB,vBB,'bx')
        duvs_i *= np.sign(theta)

        if (i < (N_i-1)): 
            ccts[n_i*i:n_i*i+n_i] = ccts_i
            duvs[n_i*i:n_i*i+n_i] = duvs_i[:,0]
        else: 
            ccts[n_i*i:] = ccts_i
            duvs[n_i*i:] = duvs_i[:,0]
    
    # Regulate output:
    if (out == 'cct') | (out == 1):
        return np2d(ccts).T
    elif (out == 'duv') | (out == -1):
        return np2d(duvs).T
    elif (out == 'cct,duv') | (out == 2):
        return np2d(ccts).T, np2d(duvs).T
    elif (out == "[cct,duv]") | (out == -2):
        return np.vstack((ccts,duvs)).T   
    else:
        raise Exception('Unknown output requested')

#------------------------------------------------------------------------------
# Robertson 1968:
#------------------------------------------------------------------------------
def xyz_to_cct_search_robertson1968(xyzw, cieobs = _CIEOBS, out = 'cct', wl = None, 
                                    atol = 0.1, rtol = 1e-5, force_tolerance = True,
                                    cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS,
                                    lut = None, cct_search_list = 'robertson1968', mk_search_list = None, 
                                    upper_cct_max = _CCT_MAX, 
                                    ):
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
            | If False: search only using the list of CCTs in _CCT_LUT, or in lut,
            |           or suplied using :cct_search_list: or :mk_search_list:. 
            |           Only one loop is performed. Accuracy depends on CCT of test
            |           source and the location and spacing of the LUT-CCTs in the list.
            | If True:  search will use adjacent CCTs to test source to create a new LUT,
            |           after which the search process repeats until the tolerance is
            |           reached for ALL sources in xyzw!
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit golden-ratio search to this cct.
        :lut:
            | None, optional
            | Pre-calculated LUT. If None use _CCT_LUT['robertson1968']
        :cct_search_list:
            | None, optional
            | List of ccts to obtain a first guess for the cct of the input xyz.
            |   (creates new LUT)
            | Options:
            |   - 'default' or None: defaults to the mode in _CCT_SEARCH_METHOD.
            |   - 'bf-search': defaults to _CCT_SEARCH_LIST_BRUTEFORCE
            |   - 'zhang2019': defaults to _CCT_SEARCH_LIST_ZHANG2019
            |   - 'pw_linear': defaults to _CCT_SEARCH_PW_LIN
            |   - 'robertson1968': defaults to _CCT_SEARCH_LIST_ROBERTSON1968
        :mk_search_list:
            | None, optional
            | Input cct_search_list directly in MK (mired) scale.
            | None: does nothing, but when not None input overwrites cct_search_list !
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
            | ndarray with:
            |    cct: out == 'cct' (or 1)
            |    duv: out == 'duv' (or -1)
            |    cct, duv: out == 'cct,duv' (or 2)
            |    [cct,duv]: out == "[cct,duv]" (or -2) 
    
    References:
         1.  `Robertson, A. R. (1968). 
         Computation of Correlated Color Temperature and Distribution Temperature. 
         Journal of the Optical Society of America,  58(11), 1528–1535. 
         <https://doi.org/10.1364/JOSA.58.001528>`_

    """
    # get requested cmf set:
    if isinstance(cieobs,str):
        cmf = _CMF[cieobs]['bar'].copy()
    else:
        cmf = cieobs.copy()
    cmf = cie_interp(cmf, wl, kind = 'cmf', negative_values_allowed=True)
    
    # process cspace input:
    cspace_dict = _process_cspace_input(cspace, cspace_kwargs = cspace_kwargs)
    cspace_str = cspace_dict['str']
    
    # get search_lists:
    cct_search_list, mk_search_list = _process_cct_mk_search_lists(cct_search_list = cct_search_list, 
                                                                    mk_search_list = mk_search_list, 
                                                                    upper_cct_max = upper_cct_max)

    # load / create LUT:
    if lut is None:   
        lut = copy.deepcopy(_CCT_LUT['robertson1968'])
    else:
        if not isinstance(lut,dict):
            lut = {cspace_str: {cieobs:lut}}
            
    if cspace_str not in lut.keys():
        lut[cspace_str] = {cieobs : calculate_lut('robertson1968', cct_search_list, cieobs = cieobs, wl = wl,
                                                  cspace = cspace_dict, cspace_kwargs = None)}
    if cieobs not in lut[cspace_str]:
        lut[cspace_str][cieobs] = calculate_lut('robertson1968', cct_search_list, cieobs = cieobs, wl = wl,
                                                cspace = cspace_dict, cspace_kwargs = None)
    # print(lut[cspace_str][cieobs][:,:1].shape,cct_search_list.shape)
    # print(np.hstack((lut[cspace_str][cieobs][:,:1],cct_search_list)))
    if not np.array_equal(lut[cspace_str][cieobs][:,:1],cct_search_list):
        print('Generating Robertson1968 LUT for cct_search_list != default list.')
        lut[cspace_str][cieobs] = calculate_lut('robertson1968', cct_search_list, cieobs = cieobs, wl = wl,
                                                cspace = cspace_dict, cspace_kwargs = None)
       
    lut_i = lut[cspace_str][cieobs]
       
    # calculate chromaticity coordinates of input xyzw:
    Yuv = cspace_dict['fwtf'](xyzw)
    u = Yuv[:,1,None] # get CIE 1960 u
    v = Yuv[:,2,None] # get CIE 1960 v
    i = 0
    while True:
        N = lut_i.shape[-1]//4
        ns = np.arange(0,N*4,4,dtype=int)
        
        # get uBB, vBB, mBB from lut:
        TBB = lut_i[:,ns]
        uBB = lut_i[:,ns+1]
        vBB = lut_i[:,ns+2]
        mBB =  lut_i[:,ns+3] # slope
 
        # calculate distances to coordinates in lut (Eq. 4):
        di = ((v.T - vBB) - mBB * (u.T - uBB)) / ((1 + mBB**2)**(0.5))
        # dip1 = np.roll(di,-1,0)
        di0 = ((v.T - vBB)**2 + (u.T - uBB)**2)
        
        # find adjacent Ti's (i.e. dj/dj+1<0):
        # pn = np.where((di/dip1) < 0)[0]#[u.shape[0]:] # results in multiple solutions for single CCT!!
        pn = (di0.argmin(axis=0))
        # import matplotlib.pyplot as plt
        # plt.plot(uBB,vBB,'b+-')
        # plt.plot(uBB[pn],vBB[pn],'mx')
        # plt.plot(u,v,'ro')
    
        # Estimate Tc:
        c = ((pn)>=(TBB.shape[0]-1)) # check out-of-luts
        pn[c] = (TBB.shape[0] - 2)
        ccts_i = (np.diag(((1/TBB[pn])+(di[pn]/(di[pn]-di[pn+1]))*((1/TBB[pn+1]) - (1/TBB[pn])))**(-1)))[:,None].copy()
        if c.any(): ccts_i[c] = -1 # indicate out of lut
        
        # break loop if required tolerance is reached:
        if force_tolerance:
            ni = 10
            # update lut_i:
            pn[(pn-1)<0] = 1
            pn[(pn+1)>TBB.shape[0]] = TBB.shape[0] - 1
            ccts_i_mM =  np.hstack((TBB[pn-1],TBB[pn+1]))
            ccts_min, ccts_max = ccts_i_mM.min(axis=-1),ccts_i_mM.max(axis=-1)
            cct_search_list_i = 1e6/np.linspace(1e6/ccts_max,1e6/ccts_min,ni)
            cct_search_list_i = np.reshape(cct_search_list_i,(-1,1)) # reshape for easy input in calculate lut
            ccts_im1 = ccts_i # update previous cct
            lut_i = calculate_lut('robertson1968', cct_search_list_i, cieobs = cieobs, wl = wl,
                                  cspace = cspace_dict, cspace_kwargs = None)
            lut_i = np.reshape(lut_i, (ni,-1))
            
            if i == 0:
                ccts_im1 = ccts_i  # initialize
                i+=1
                continue
            
            dccts = np.abs(ccts_i - ccts_im1)
            if (dccts <= atol).all() | ((dccts/ccts_i) <= rtol).all():
                break
            i+=1
            
        else:
            break
        
    # Final ccts:
    ccts = ccts_i
    
    # Get duv: 
    BB_i = cri_ref(ccts_i, ref_type = ['BB'], wl3 = wl)
    xyzBB_i = spd_to_xyz(BB_i, cieobs = cieobs, relative = True)
    uvBB_i = cspace_dict['fwtf'](xyzBB_i)[...,1:]
    uBB_i, vBB_i = uvBB_i[...,0:1], uvBB_i[...,1:2]
    uBB_c, vBB_c = (u - uBB_i), (v - vBB_i)
    duvs_i = (uBB_c**2 + vBB_c**2)**0.5
    
    # find sign of duv:
    theta = math.positive_arctan(uBB_c,vBB_c,htype='deg')
    theta[theta>180] = theta[theta>180] - 360
    duvs_i *= np.sign(theta)
    duvs = duvs_i


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
# Ohno 2014
#------------------------------------------------------------------------------
def xyz_to_cct_ohno2014(xyzw, cieobs = _CIEOBS, out = 'cct', wl = None, 
                        rtol = 1e-5, atol = 0.1, force_tolerance = True, 
                        force_out_of_lut = True, fallback_mode = _OHNO2014_FALLBACK_MODE, split_zhang_calculation_at_N = 100, 
                        cct_search_list = None, mk_search_list = None, upper_cct_max = _CCT_MAX, approx_cct_temp = True, 
                        cctuv_lut = None, cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS):
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
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
        :force_out_of_lut: 
            | True, optional
            | If True and cct is out of range of the LUT, then switch to 
            | the selected search fallback_mode, else return numpy.nan values.
        :fallback_mode:
            | _OHNO2014_FALLBACK_MODE, optional
            | Fallback mode for out-of-lut input. 
            | Options:
            |  - 'zhang2019': use xyz_to_cct_zhang2019()
            |  - 'brute-force-search-robust': use xyz_to_cct_search_bf_robust()
            |  - 'brute-force-search-fast': use xyz_to_cct_search_bf_fast()
            |  - 'robertson1968': use xyz_to_cct_search_robertson1968()
        :split_zhang_calculation_at_N:
            | 100, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :rtol: 
            | 1e-5, float, optional
            | Stop search when  a relative cct tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop search when cct a absolute tolerance (K) is reached.
        :force_tolerance:
            | True, optional
            | Accuracy of the calculations depends on the CCT of test source 
            |   and the location and spacing of initial CCTs used to start the search,
            |   or the LUT based method.
            | If True:  search process will continue until the tolerance is
            |           reached for ALL sources in xyzw! 
            | If False: search process might stop early (depending on the chosen mode).
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit brute-force or golden-ratio search to this cct.
        :approx_cct_temp: 
            | True, optional
            | If True: use xyz_to_cct_HA() to get a first estimate of cct to 
            |           speed up the fast brute-force search.
        :cct_search_list:
            | None, optional
            | List of ccts to obtain a first guess for the cct of the input xyz
            | for the 'brute-force-search-robust', 'zhang2019', 'robertson1968' fallback methods, or
            | when HA estimation fails in the 'brute-force-search-fast' fallback algorithm 
            | due to out-of-range ccts.
            | Options:
            |   - 'default' or None: defaults to the mode in _CCT_SEARCH_METHOD.
            |   - 'bf-search': defaults to _CCT_SEARCH_LIST_BRUTEFORCE
            |   - 'zhang2019': defaults to _CCT_SEARCH_LIST_ZHANG2019
            |   - 'pw_linear': defaults to _CCT_SEARCH_PW_LIN
            |   - 'robertson1968': defaults to _CCT_SEARCH_LIST_ROBERTSON1968
        :mk_search_list:
            | None, optional
            | Input cct_search_list directly in MK (mired) scale.
            | None: does nothing, but when not None input overwrites cct_search_list !        
        :cctuv_lut:
            | None, optional
            | CCT+uv look-up-table to use.
            | If None: use luxpy._CCT_LUT['ohno2014']
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
            | ndarray with:
            |    cct: out == 'cct' (or 1)
            |    duv: out == 'duv' (or -1)
            |    cct, duv: out == 'cct,duv' (or 2)
            |    [cct,duv]: out == "[cct,duv]" (or -2) 
            
    Note:
        Default LUTs are stored in the folder specified in _CCT_LUT_PATH.
        
    Reference:
        1. `Ohno Y. Practical use and calculation of CCT and Duv. 
        Leukos. 2014 Jan 2;10(1):47-55.
        <http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020>`_
    """

    xyzw = np2d(xyzw)  

    if len(xyzw.shape)>2:
        raise Exception('xyz_to_cct_ohno(): Input xyzwa.ndim must be <= 2 !')
      
    cspace_dict = _process_cspace_input(cspace, cspace_kwargs)
    cspace_string = cspace_dict['str']
    
    # get 1960 u,v of test source:
    Yuv = cspace_dict['fwtf'](xyzw) # remove possible 1-dim 
    axis_of_v3 = len(Yuv.shape)-1 # axis containing color components
    u = Yuv[:,1,None] # get CIE 1960 u
    v = Yuv[:,2,None] # get CIE 1960 v

    uv = np2d(np.concatenate((u,v),axis = axis_of_v3))
    
    # get search_lists:
    cct_search_list, mk_search_list = _process_cct_mk_search_lists(cct_search_list = cct_search_list, 
                                                                   mk_search_list = mk_search_list, 
                                                                   upper_cct_max = upper_cct_max)
    
    # load cct & uv from LUT:
    if cctuv_lut is None:   
        cctuv_lut = _CCT_LUT['ohno2014']
    else:
        if not isinstance(cctuv_lut,dict):
            cctuv_lut = {cspace_string: {cieobs:cctuv_lut}}
    if cspace_string not in cctuv_lut.keys():
        cctuv_lut[cspace_string] = {cieobs : calculate_lut('ohno2014',ccts = None, cieobs = cieobs, add_to_lut = False, wl = wl,
                                                         cspace = cspace_dict, cspace_kwargs = None)}
    if cieobs not in cctuv_lut[cspace_string]:
        cctuv_lut[cspace_string][cieobs] = calculate_lut('ohno2014',ccts = None, cieobs = cieobs, add_to_lut = False, wl = wl,
                                                         cspace = cspace_dict, cspace_kwargs = None)
    
    cct_LUT = cctuv_lut[cspace_string][cieobs][:,0,None] 
    uv_LUT = cctuv_lut[cspace_string][cieobs][:,1:3] 
    
    # calculate CCT of each uv:
    CCT = np.zeros(uv.shape[0])
    CCT.fill(np.nan) # initialize with NaN's
    Duv = CCT.copy() # initialize with NaN's
    idx_m = 0
    idx_M = uv_LUT.shape[0]-1
    for i in range(uv.shape[0]):
        out_of_lut = False
        delta_uv = (((uv_LUT - uv[i])**2.0).sum(axis = 1))**0.5 # calculate distance of uv with uv_LUT
        idx_min = delta_uv.argmin() # find index of minimum distance 

        # find Tm, delta_uv and u,v for 2 points surrounding uv corresponding to idx_min:
        if idx_min == idx_m:
            idx_min_m1 = idx_min
            out_of_lut = True
        else:
            idx_min_m1 = idx_min - 1
        if idx_min == idx_M:
            idx_min_p1 = idx_min
            out_of_lut = True
        else:
            idx_min_p1 = idx_min + 1
        

        if (out_of_lut == True) & (force_out_of_lut == True): # calculate using search-function

            cct_i, Duv_i = xyz_to_cct_search(xyzw[i:i+1,:], cieobs = cieobs, wl = wl, mode = fallback_mode,
                                             rtol = rtol, atol = atol, force_tolerance = force_tolerance,
                                             out = 'cct,duv', upper_cct_max = upper_cct_max, 
                                             cct_search_list = cct_search_list, mk_search_list = mk_search_list,
                                             split_zhang_calculation_at_N = split_zhang_calculation_at_N,
                                             cspace = cspace_dict, cspace_kwargs = None,
                                             approx_cct_temp = approx_cct_temp, lut = None)
     
            CCT[i] = cct_i
            Duv[i] = Duv_i
            continue
        elif (out_of_lut == True) & (force_out_of_lut == False):
            CCT[i] = np.nan
            Duv[i] = np.nan
            
            
        cct_m1 = cct_LUT[idx_min_m1] # - 2*_EPS
        delta_uv_m1 = delta_uv[idx_min_m1]
        uv_m1 = uv_LUT[idx_min_m1]
        cct_p1 = cct_LUT[idx_min_p1] 
        delta_uv_p1 = delta_uv[idx_min_p1]
        uv_p1 = uv_LUT[idx_min_p1]

        cct_0 = cct_LUT[idx_min]
        delta_uv_0 = delta_uv[idx_min]

        # calculate uv distance between Tm_m1 & Tm_p1:
        delta_uv_p1m1 = ((uv_p1[0] - uv_m1[0])**2.0 + (uv_p1[1] - uv_m1[1])**2.0)**0.5

        # Triangular solution:
        x = ((delta_uv_m1**2)-(delta_uv_p1**2)+(delta_uv_p1m1**2))/(2*delta_uv_p1m1)
        Tx = cct_m1 + ((cct_p1 - cct_m1) * (x / delta_uv_p1m1))
        #uBB = uv_m1[0] + (uv_p1[0] - uv_m1[0]) * (x / delta_uv_p1m1)
        vBB = uv_m1[1] + (uv_p1[1] - uv_m1[1]) * (x / delta_uv_p1m1)

        Tx_corrected_triangular = Tx*0.99991
        signDuv = np.sign(uv[i][1]-vBB)
        Duv_triangular = signDuv*np.atleast_1d(((delta_uv_m1**2.0) - (x**2.0))**0.5)

                                
        # Parabolic solution (only when Duv_triangular above Threshold or when two ccts are equal)
        Threshold = 0.002
        if ((cct_0 == cct_p1) | (cct_0 == cct_m1)) | (np.abs(Duv_triangular) < Threshold):
            CCT[i] = Tx_corrected_triangular
            Duv[i] = Duv_triangular
        else:
            a = delta_uv_m1/((cct_m1 - cct_0 )*(cct_m1 - cct_p1) + _EPS)
            b = delta_uv_0/((cct_0 - cct_m1 )*(cct_0 - cct_p1 ) + _EPS)
            c = delta_uv_p1/((cct_p1 - cct_m1 )*(cct_p1 - cct_0) + _EPS)
            A = a + b + c
            B = -(a*(cct_p1 + cct_0) + b*(cct_p1 + cct_m1) + c*(cct_0 + cct_m1))
            C = (a*cct_p1*cct_0) + (b*cct_p1*cct_m1) + (c*cct_0*cct_m1)
            Tx = -B/(2*A + _EPS)
            Tx_corrected_parabolic = Tx*0.99991
            Duv_parabolic = signDuv*(A*np.power(Tx_corrected_parabolic,2) + B*Tx_corrected_parabolic + C)

            CCT[i] = Tx_corrected_parabolic
            Duv[i] = Duv_parabolic
            

    # Regulate output:
    if (out == 'cct') | (out == 1):
        return np2dT(CCT)
    elif (out == 'duv') | (out == -1):
        return np2dT(Duv)
    elif (out == 'cct,duv') | (out == 2):
        return np2dT(CCT), np2dT(Duv)
    elif (out == "[cct,duv]") | (out == -2):
        return np.vstack((CCT,Duv)).T

xyz_to_cct_ohno = xyz_to_cct_ohno2014 # for legacy reasons

#---------------------------------------------------------------------------------------------------
def cct_to_xyz_fast(ccts, duv = None, cct_resolution = 0.1, cieobs = _CIEOBS, wl = None,
               cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS):
    """
    Convert correlated color temperature (CCT) and Duv (distance above (>0) or 
    below (<0) the Planckian locus) to XYZ tristimulus values.
    
    | Finds xyzw_estimated by estimating the line perpendicular to the Planckian lcous: 
    |    First, the angle between the coordinates corresponding to ccts 
    |    and ccts-cct_resolution are calculated, then 90° is added, and finally
    |    the new coordinates are determined, while taking sign of duv into account.   
     
    Args:
        :ccts: 
            | ndarray [N,1] of cct values
        :duv: 
            | None or ndarray [N,1] of duv values, optional
            | Note that duv can be supplied together with cct values in :ccts: 
            | as ndarray with shape [N,2].
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
        If duv is not supplied (:ccts:.shape is (N,1) and :duv: is None), 
        source is assumed to be on the Planckian locus.
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

    cspace_dict = _process_cspace_input(cspace, cspace_kwargs)
    if cspace_dict['bwtf'] is None:
        raise Exception('cct_to_xyz_fast requires the backward cspace transform to be defined !!!')
    
    BB = cri_ref(np.vstack((cct, cct-cct_resolution,cct+cct_resolution)), wl3 = wl, ref_type = ['BB'])
    xyzBB = spd_to_xyz(BB, cieobs = cieobs)
    YuvBB = cspace_dict['fwtf'](xyzBB)
    N = (BB.shape[0]-1)//3
    YuvBB_centered = (YuvBB[N:] - np.vstack((YuvBB[:N],YuvBB[:N])))
    theta = math.positive_arctan(YuvBB_centered[...,1:2], YuvBB_centered[...,2:3],htype='rad') 
    theta = (theta[:N] + (theta[N:] - np.pi))/2 # take average for increased accuracy
    theta = theta + np.pi/2*np.sign(duv) # add 90° to obtain the direction perpendicular to the blackbody locus
    u, v = YuvBB[:N,1:2] + np.abs(duv)*np.cos(theta), YuvBB[:N,2:3] + np.abs(duv)*np.sin(theta)
    Yuv = np.hstack((100*np.ones_like(u),u,v))
    return cspace_dict['bwtf'](Yuv)

def cct_to_xyz(ccts, duv = None, cieobs = _CIEOBS, wl = None, mode = 'ohno2014', 
               fallback_mode_for_ohno2014 = _OHNO2014_FALLBACK_MODE, split_zhang_calculation_at_N = 100,
               force_fast_mode = True, cct_resolution_of_fast_mode = 0.1, out = None, 
               rtol = 1e-5, atol = 0.1, force_tolerance = True, 
               force_out_of_lut = True, upper_cct_max = _CCT_MAX, 
               approx_cct_temp = True, cct_search_list = None, mk_search_list = None,
               cctuv_lut = None, cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS):
    """
    Convert correlated color temperature (CCT) and Duv (distance above (>0) or 
    below (<0) the Planckian locus) to XYZ tristimulus values.
    
    | Finds xyzw_estimated by minimization of:
    |    
    |    F = numpy.sqrt(((100.0*(cct_min - cct)/(cct))**2.0) 
    |         + (((duv_min - duv)/(duv))**2.0))
    |    
    | with cct,duv the input values and cct_min, duv_min calculated using 
    | luxpy.xyz_to_cct(xyzw_estimated,...). This requires calculating xyz_to_cct()
    | which is done using the method specified in mode (if not 'fast').
    |
    | or 
    |
    | Finds xyzw_estimated by estimating the line perpendicular to the Planckian lcous: 
    |    First, the angle between the coordinates corresponding to ccts 
    |    and ccts-cct_resolution are calculated, then 90° is added, and finally
    |    the new coordinates are determined, while taking sign of duv into account.   

    Args:
        :ccts: 
            | ndarray of cct values
        :duv: 
            | None or ndarray of duv values, optional
            | Note that duv can be supplied together with cct values in :ccts: 
            | as ndarray with shape (N,2)
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set used to calculated xyzw.
        :mode: 
            | 'ohno2014', optional
            | Determines what method to use.
            | Options:
            | - 'fast': use cct_to_xyz_fast() for a direct search.
            | - 'ohno2014': use xyz_to_cct_ohno2014() in inverse search.
            | - 'search': use xyz_to_cct_search_bf_fast() in inverse search, 
            | - 'bf-fast' or 'brute-force-search-fast': use xyz_to_cct_search_bf_fast() in inverse search.
            | - 'bf-robust' or 'brute-force-search-robust': use xyz_to_cct_search_bf_robust() for inverse search.
            | - 'zhang2019': use xyz_to_cct_search_zhang2019() for inverse search.
            | - 'robertson1968': use xyz_to_cct_search_robertson1968() for inverse search.
        :split_zhang_calculation_at_N:
            | 100, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
        :force_fast_mode:
            | True, optional
            | Try the fast approach (i.e. cct_to_xyz_fast()). This overrides the 
            | method specified in 'mode'. (mode method only used when cspace 
            |                              does not have backward transform!)
        :cct_resolution_of_fast_mode:
            | 0.1, optional
            | The CCT resolution of the fast mode.
        :out: 
            | None (or 1), optional
            | If not None or 1: output a ndarray that contains estimated 
            | xyz and minimization results: 
            |    (cct_min, duv_min, F_min (objective fcn value))
            | (Defaults to None when fast mode is used !!!)
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
        :rtol: 
            | 1e-5, float, optional
            | Stop search when a relative cct tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop brute-force search when an absolute cct tolerance (K) is reached.
        :force_tolerance:
            | True, optional
            | Accuracy of the calculations depends on the CCT of test source 
            |   and the location and spacing of initial CCTs used to start the search,
            |   or the LUT based method.
            | If True:  search process will continue until the tolerance is
            |           reached for ALL sources in xyzw! 
            | If False: search process might stop early (depending on the chosen mode).
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit brute-force search to this cct.
        :approx_cct_temp: 
            | True, optional
            | If True: use xyz_to_cct_HA() to get a first estimate of cct to 
            |           speed up the fast brute-force search.
        :cct_search_list:
            | None, optional
            | List of ccts to obtain a first guess for the cct of the input xyz
            | for the 'brute-force-search-robust', 'zhang2019', 'robertson1968' fallback methods, or
            | when HA estimation fails in the 'brute-force-search-fast' fallback algorithm 
            | due to out-of-range ccts.
            | Options:
            |   - 'default' or None: defaults to the mode in _CCT_SEARCH_METHOD.
            |   - 'bf-search': defaults to _CCT_SEARCH_LIST_BRUTEFORCE
            |   - 'zhang2019': defaults to _CCT_SEARCH_LIST_ZHANG2019
            |   - 'pw_linear': defaults to _CCT_SEARCH_PW_LIN
            |   - 'robertson1968': defaults to _CCT_SEARCH_LIST_ROBERTSON1968
        :mk_search_list:
            | None, optional
            | Input cct_search_list directly in MK (mired) scale.
            | None: does nothing, but when not None input overwrites cct_search_list !         
        :force_out_of_lut: 
            | True, optional
            | If True and cct is out of range of the LUT, then switch to 
            | brute-force search method, else return numpy.nan values.
        :fallback_mode_for_ohno2014:
            | _OHNO2014_FALLBACK_MODE, optional
            | Fallback mode for out-of-lut input when mode == 'ohno2014'. 
            | Options:
            |  - 'robertson1968': use xyz_to_cct_search_robertson1968()
            |  - 'zhang2019': use xyz_to_cct_search_zhang2019()
            |  - 'bf-robust' or 'brute-force-search-robust': use xyz_to_cct_search_bf_robust()
            |  - 'bf-fast' or 'brute-force-search-fast': use xyz_to_cct_search_bf_fast()
        :cctuv_lut:
            | None, optional
            | CCT+uv look-up-table to use.
            | If None: use luxpy._CCT_LUT
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
        If duv is not supplied (:ccts:.shape is (N,1) and :duv: is None), 
        source is assumed to be on the Planckian locus.
    """
    cspace_dict = _process_cspace_input(cspace, cspace_kwargs)
    cspace_string = cspace_dict['str']
   
    # Set up requested mode:
    mode_bak = mode
    if force_fast_mode: mode = 'fast'
    if mode == 'fast':
        if cspace_dict['bwtf'] is None: # Exception not needed: use xyz_to_Yuv and Yuv_to_xyz in optimization
            mode = 'ohno2014' if mode_bak == 'fast' else mode_bak # use fall-back method
    
    if mode == 'fast':
        return cct_to_xyz_fast(ccts, duv = duv, 
                               cct_resolution = cct_resolution_of_fast_mode, 
                               cieobs = cieobs, wl = wl,
                               cspace = cspace_dict, cspace_kwargs = None)
    else:
        
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
        elif duv is not None:
            duv = np2d(duv)
        
        # pre-load or pre-create LUT:
        if (mode == 'robertson1968') | (mode == 'ohno2014'):
            if cctuv_lut is None:   
                cctuv_lut = _CCT_LUT[mode]
            else:
                if not isinstance(cctuv_lut,dict):
                    cctuv_lut = {cspace_string: {cieobs:cctuv_lut}}
            if cspace_string not in cctuv_lut.keys():
                cctuv_lut[cspace_string] = {cieobs : calculate_lut(mode, ccts = None, cieobs = cieobs, add_to_lut = False, wl = wl,
                                                                   cspace = cspace_dict, cspace_kwargs = None)}
            if cieobs not in cctuv_lut[cspace_string]:
                cctuv_lut[cspace_string][cieobs] = calculate_lut(mode, ccts = None, cieobs = cieobs, add_to_lut = False, wl = wl,
                                                                 cspace = cspace_dict, cspace_kwargs = None)
        else:
            cctuv_lut = None
    
        # get search_lists:
        cct_search_list, mk_search_list = _process_cct_mk_search_lists(cct_search_list = cct_search_list, 
                                                                       mk_search_list = mk_search_list, 
                                                                       upper_cct_max = upper_cct_max)

    
        # get estimates of approximate xyz values in case duv = None:
        BB = cri_ref(ccts = cct, wl3 = wl, ref_type = ['BB'])
        xyz_est = spd_to_xyz(data = BB, cieobs = cieobs, out = 1)
        results = np.zeros([ccts.shape[0],3]);results.fill(np.nan) 
    
        if duv is not None:
            
            # optimization/minimization setup:
            def objfcn(uv_offset, uv0, cct, duv, out = 1):#, cieobs = cieobs, wl = wl, mode = mode):
                uv0 = np2d(uv0 + uv_offset)
                Yuv0 = np.concatenate((np2d([100.0]), uv0),axis=1)
                xyz0 = cspace_dict['bwtf'](Yuv0) if cspace_dict['bwtf'] is not None else Yuv_to_xyz(Yuv0)
                cct_min, duv_min = xyz_to_cct(xyz0,cieobs = cieobs, out = 'cct,duv',
                                              wl = wl, mode = mode, 
                                              fallback_mode_for_ohno2014 = fallback_mode_for_ohno2014, 
                                              split_zhang_calculation_at_N = split_zhang_calculation_at_N,
                                              rtol = rtol, atol = atol, force_tolerance = force_tolerance,
                                              force_out_of_lut = force_out_of_lut, 
                                              upper_cct_max = upper_cct_max, 
                                              approx_cct_temp = approx_cct_temp,
                                              cct_search_list = cct_search_list,
                                              mk_search_list = mk_search_list,
                                              cctuv_lut = cctuv_lut,
                                              cspace = cspace_dict, 
                                              cspace_kwargs = None)
                
                F = np.sqrt(((100.0*(cct_min[0] - cct[0])/(cct[0]))**2.0) + (((duv_min[0] - duv[0])/(duv[0]))**2.0))
                if out == 'F':
                    return F
                else:
                    return np.concatenate((cct_min, duv_min, np2d(F)),axis = 1) 
                
            # loop through each xyz_est:
            for i in range(xyz_est.shape[0]):
                xyz0 = xyz_est[i]
                cct_i = cct[i]
                duv_i = duv[i]
                cct_min, duv_min =  xyz_to_cct(xyz0,cieobs = cieobs, out = 'cct,duv',wl = wl, 
                                               mode = mode,
                                               fallback_mode_for_ohno2014 = fallback_mode_for_ohno2014,
                                               split_zhang_calculation_at_N = split_zhang_calculation_at_N,
                                               rtol = rtol, atol = atol, force_tolerance = force_tolerance, 
                                               force_out_of_lut = force_out_of_lut, 
                                               upper_cct_max = upper_cct_max, 
                                               approx_cct_temp = approx_cct_temp,
                                               cct_search_list = cct_search_list,
                                               mk_search_list = mk_search_list,
                                               cctuv_lut = cctuv_lut,
                                               cspace = cspace_dict,
                                               cspace_kwargs = None)
                
                if np.abs(duv[i]) > _EPS:
                    # find xyz:
                    Yuv0 = cspace_dict['fwtf'](xyz0) if cspace_dict['bwtf'] is not None else xyz_to_Yuv(xyz0)
                    uv0 = Yuv0[0] [1:3]
    
                    OptimizeResult = sp.optimize.minimize(fun = objfcn,x0 = np.zeros((1,2)), args = (uv0,cct_i, duv_i, 'F'), method = 'Nelder-Mead',options={"maxiter":np.inf, "maxfev":np.inf, 'xatol': 0.000001, 'fatol': 0.000001})
                    betas = OptimizeResult['x']
                    #betas = np.zeros(uv0.shape)
                    if out is not None:
                        results[i] = objfcn(betas,uv0,cct_i, duv_i, out = 3)
                    
                    uv0 = np2d(uv0 + betas)
                    Yuv0 = np.concatenate((np2d([100.0]),uv0),axis=1)
                    xyz_est[i] = cspace_dict['bwtf'](Yuv0) if cspace_dict['bwtf'] is not None else Yuv_to_xyz(Yuv0)
                
                else:
                    xyz_est[i] = xyz0
          
        if (out is None) | (out == 1):
            return xyz_est
        else:
            # Also output results of minimization:
            return np.concatenate((xyz_est,results),axis = 1)  


#-------------------------------------------------------------------------------------------------   
# general CCT-wrapper function
def xyz_to_cct(xyzw, cieobs = _CIEOBS, out = 'cct',mode = 'ohno2014', wl = None, 
               rtol = 1e-5, atol = 0.1, force_tolerance = True, 
               force_out_of_lut = True, fallback_mode_for_ohno2014 = _OHNO2014_FALLBACK_MODE,
               split_zhang_calculation_at_N = 100,
               upper_cct_max = _CCT_MAX, approx_cct_temp = True, 
               cct_search_list = None, mk_search_list = None,
               cctuv_lut = None, cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS): 
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and
    Duv (distance above (>0) or below (<0) the Planckian locus)
    using a brute-force search method, or Zhang's 2019 golden-ratio search method, 
    or Ohno's 2014 Look-Up-Table method or Robertson's 1968 search method. 
    
    | Wrapper function for use with luxpy.colortf().
    
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
        :mode: 
            | 'ohno2014', optional
            | Determines what method to use.
            | Options:
            | - 'fast': use cct_to_xyz_fast() for a direct search.
            | - 'ohno2014': use xyz_to_cct_ohno2014() in inverse search.
            | - 'search': use xyz_to_cct_search_bf_fast() in inverse search, 
            | - 'bf-fast' or 'brute-force-search-fast': use xyz_to_cct_search_bf_fast() in inverse search.
            | - 'bf-robust' or 'brute-force-search-robust': use xyz_to_cct_search_bf_robust() for inverse search.
            | - 'zhang2019': use xyz_to_cct_search_zhang2019() for inverse search.
            | - 'robertson1968': use xyz_to_cct_search_robertson1968() for inverse search.
        :split_zhang_calculation_at_N:
            | 100, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
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
            | Accuracy of the calculations depends on the CCT of test source 
            |   and the location and spacing of initial CCTs used to start the search,
            |   or the LUT based method.
            | If True:  search process will continue until the tolerance is
            |           reached for ALL sources in xyzw! 
            | If False: search process might stop early (depending on the chosen mode).
        :force_out_of_lut: 
            | True, optional
            | If True and cct is out of range of the LUT, then switch to 
            | the selected fallback_mode, else return numpy.nan values.
        :fallback_mode_for_ohno2014:
            | 'zhang2019', optional
            | Fallback mode for out-of-lut input when mode == 'ohno2014'. 
            | Options:
            |  - 'robertson1968': use xyz_to_cct_search_robertson1968()
            |  - 'zhang2019': use xyz_to_cct_zhang2019()
            |  - 'brute-force-search-robust': use xyz_to_cct_search_bf_robust()
            |  - 'brute-force-search-fast': use xyz_to_cct_search_bf_fast()
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit brute-force or golden-ratio search to this cct.
        :approx_cct_temp: 
            | True, optional
            | If True: use xyz_to_cct_HA() to get a first estimate of cct to 
            |  speed up the brute-force search. Only for 'fast' code option.
        :cct_search_list:
            | None, optional
            | List of ccts to obtain a first guess for the cct of the input xyz
            | for the 'brute-force-search-robust', 'zhang2019', 'robertson1968' fallback methods, or
            | when HA estimation fails in the 'brute-force-search-fast' fallback algorithm 
            | due to out-of-range ccts.
            | Options:
            |   - 'default' or None: defaults to the mode in _CCT_SEARCH_METHOD.
            |   - 'bf-search': defaults to _CCT_SEARCH_LIST_BRUTEFORCE
            |   - 'zhang2019': defaults to _CCT_SEARCH_LIST_ZHANG2019
            |   - 'pw_linear': defaults to _CCT_SEARCH_PW_LIN
            |   - 'robertson1968': defaults to _CCT_SEARCH_LIST_ROBERTSON1968
        :mk_search_list:
            | None, optional
            | Input cct_search_list directly in MK (mired) scale.
            | None: does nothing, but when not None input overwrites cct_search_list !
        :cctuv_lut:
            | None, optional
            | CCT+uv look-up-table to use.
            | If None: use luxpy._CCT_LUT
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
            | ndarray with:
            |   cct: out == 'cct' (or 1)
            | Optional: 
            |     duv: out == 'duv' (or -1), 
            |    cct, duv: out == 'cct,duv' (or 2), 
            |    [cct,duv]: out == "[cct,duv]" (or -2)
    """
    if (mode.lower() == 'ohno2014') | (mode.lower() == 'ohno'):
        return xyz_to_cct_ohno(xyzw = xyzw, cieobs = cieobs, out = out, wl  = wl, 
                               rtol = rtol, atol = atol, force_tolerance = force_tolerance, 
                               force_out_of_lut = force_out_of_lut, fallback_mode = fallback_mode_for_ohno2014,
                               split_zhang_calculation_at_N = split_zhang_calculation_at_N,
                               cct_search_list = cct_search_list, mk_search_list = mk_search_list, 
                               upper_cct_max = upper_cct_max, approx_cct_temp = approx_cct_temp,
                               cctuv_lut = cctuv_lut, cspace = cspace, cspace_kwargs = cspace_kwargs)
    else:
        return xyz_to_cct_search(xyzw = xyzw, cieobs = cieobs, out = out, wl  = wl, mode = mode,
                                  rtol = rtol, atol = atol, force_tolerance = force_tolerance, 
                                  split_zhang_calculation_at_N = split_zhang_calculation_at_N,
                                  cct_search_list = cct_search_list, mk_search_list = mk_search_list, 
                                  upper_cct_max = upper_cct_max, lut = cctuv_lut,
                                  cspace = cspace, cspace_kwargs = cspace_kwargs)


def xyz_to_duv(xyzw, cieobs = _CIEOBS, out = 'duv',mode = 'ohno2014', wl = None,
               rtol = 1e-5, atol = 0.1, force_tolerance = True, 
               force_out_of_lut = True, fallback_mode_for_ohno2014 = _OHNO2014_FALLBACK_MODE, 
               split_zhang_calculation_at_N = 100,
               upper_cct_max = _CCT_MAX, approx_cct_temp = True,  
               cct_search_list = None, mk_search_list = None,
               cctuv_lut = None, cspace = _CCT_CSPACE, cspace_kwargs = _CCT_CSPACE_KWARGS): 
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and
    Duv (distance above (>0) or below (<0) the Planckian locus)
    using a brute-force search method, or Zhang's golden-ratio search method, 
    or Ohno's Look-Up-Table method. 
    
    | Wrapper function for use with luxpy.colortf().
    
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set used to calculated xyzw.
        :out: 
            | 'duv' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)
        :mode: 
            | 'ohno2014', optional
            | Determines what method to use.
            | Options:
            | - 'fast': use cct_to_xyz_fast() for a direct search.
            | - 'ohno2014': use xyz_to_cct_ohno2014() in inverse search.
            | - 'search': use xyz_to_cct_search_bf_fast() in inverse search, 
            | - 'bf-fast' or 'brute-force-search-fast': use xyz_to_cct_search_bf_fast() in inverse search.
            | - 'bf-robust' or 'brute-force-search-robust': use xyz_to_cct_search_bf_robust() for inverse search.
            | - 'zhang2019': use xyz_to_cct_search_zhang2019() for inverse search.
            | - 'robertson1968': use xyz_to_cct_search_robertson1968() for inverse search.
        :split_zhang_calculation_at_N:
            | 100, optional
            | Split calculation when xyzw.shape[0] > split_calculation_at_N. 
            | Splitting speeds up the calculation. If None: no splitting is done.
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
            | Accuracy of the calculations depends on the CCT of test source 
            |   and the location and spacing of initial CCTs used to start the search,
            |   or the LUT based method.
            | If True:  search process will continue until the tolerance is
            |           reached for ALL sources in xyzw! 
            | If False: search process might stop early (depending on the chosen mode).
        :force_out_of_lut: 
            | True, optional
            | If True and cct is out of range of the LUT, then switch to 
            | the selected fallback_mode, else return numpy.nan values.
        :fallback_mode_for_ohno2014:
            | _OHNO2014_FALLBACK_MODE, optional
            | Fallback mode for out-of-lut input when mode == 'ohno2014'. 
            | Options:
            |  - 'robertson1968': use xyz_to_cct_search_robertson1968()
            |  - 'zhang2019': use xyz_to_cct_zhang2019()
            |  - 'brute-force-search-robust': use xyz_to_cct_search_bf_robust()
            |  - 'brute-force-search-fast': use xyz_to_cct_search_bf_fast()
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit brute-force or golden-ratio search to this cct.
        :approx_cct_temp: 
            | True, optional
            | If True: use xyz_to_cct_HA() to get a first estimate of cct to 
            |  speed up the brute-force search. Only for 'fast' code option.
        :cct_search_list:
            | None, optional
            | List of ccts to obtain a first guess for the cct of the input xyz
            | for the 'brute-force-search-robust', 'zhang2019', 'robertson1968' fallback methods, or
            | when HA estimation fails in the 'brute-force-search-fast' fallback algorithm 
            | due to out-of-range ccts.
            | Options:
            |   - 'default' or None: defaults to the mode in _CCT_SEARCH_METHOD.
            |   - 'bf-search': defaults to _CCT_SEARCH_LIST_BRUTEFORCE
            |   - 'zhang2019': defaults to _CCT_SEARCH_LIST_ZHANG2019
            |   - 'pw_linear': defaults to _CCT_SEARCH_PW_LIN
            |   - 'robertson1968': defaults to _CCT_SEARCH_LIST_ROBERTSON1968
        :mk_search_list:
            | None, optional
            | Input cct_search_list directly in MK (mired) scale.
            | None: does nothing, but when not None input overwrites cct_search_list !
        :cctuv_lut:
            | None, optional
            | CCT+uv look-up-table to use.
            | If None: use luxpy._CCT_LUT
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
            | ndarray with:
            |   cct: out == 'cct' (or 1)
            | Optional: 
            |     duv: out == 'duv' (or -1), 
            |    cct, duv: out == 'cct,duv' (or 2), 
            |    [cct,duv]: out == "[cct,duv]" (or -2)
            
    Returns:
        :returns:
            | ndarray with:
            |   duv: out == 'duv' (or -1)
            | Optional: 
            |     duv: out == 'duv' (or -1), 
            |     cct, duv: out == 'cct,duv' (or 2), 
            |     [cct,duv]: out == "[cct,duv]" (or -2)
    """

    if (mode.lower() == 'ohno2014') | (mode.lower() == 'ohno'):
        return xyz_to_cct_ohno2014(xyzw = xyzw, cieobs = cieobs, out = out, wl  = wl, 
                               rtol = rtol, atol = atol, force_tolerance = force_tolerance,
                               force_out_of_lut = force_out_of_lut, fallback_mode = fallback_mode_for_ohno2014,
                               split_zhang_calculation_at_N = split_zhang_calculation_at_N,
                               cct_search_list = cct_search_list, mk_search_list = mk_search_list, 
                               upper_cct_max = upper_cct_max, approx_cct_temp = approx_cct_temp,
                               cctuv_lut = cctuv_lut, cspace = cspace, cspace_kwargs = cspace_kwargs)
    else:
        return xyz_to_cct_search(xyzw = xyzw, cieobs = cieobs, out = out, wl  = wl, mode = mode,
                                  rtol = rtol, atol = atol, force_tolerance = force_tolerance, 
                                  split_zhang_calculation_at_N = split_zhang_calculation_at_N,
                                  cct_search_list = cct_search_list, mk_search_list = mk_search_list, 
                                  upper_cct_max = upper_cct_max, lut = cctuv_lut,
                                  cspace = cspace, cspace_kwargs = cspace_kwargs)


   

