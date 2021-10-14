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

 :_CCT_MAX: (= 1e12), max. value that does not cause overflow problems. 

 :_CCT_LUT_PATH: Folder with Look-Up-Tables (LUT) for correlated color 
                 temperature calculation followings Ohno's method.

 :_CCT_LUT: Dict with LUTs.
 
 :_CCT_LUT_CALC: Boolean determining whether to force LUT calculation, even if
                 the LUT can be fuond in ./data/cctluts/.

 :calculate_lut(): Function that calculates the LUT for the ccts stored in 
                   ./data/cctluts/cct_lut_cctlist.dat or given as input 
                   argument. Calculation is performed for CMF set specified in
                   cieobs. Adds a new (temprorary) field to the _CCT_LUT dict.

 :calculate_luts(): Function that recalculates (and overwrites) LUTs in 
                    ./data/cctluts/ for the ccts stored in 
                    ./data/cctluts/cct_lut_cctlist.dat or given as input 
                    argument. Calculation is performed for all CMF sets listed 
                    in _CMF['types'].

 :xyz_to_cct(): | Calculates CCT, Duv from XYZ 
                | wrapper for xyz_to_cct_ohno() & xyz_to_cct_search()

 :xyz_to_duv(): Calculates Duv, (CCT) from XYZ
                wrapper for xyz_to_cct_ohno() & xyz_to_cct_search()

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

 :xyz_to_cct_ohno(): | Calculates CCT, Duv from XYZ using a LUT following:
                     | `Ohno Y. (2014)
                       Practical use and calculation of CCT and Duv. 
                       Leukos. 2014 Jan 2;10(1):47-55.
                       <http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020>`_

 :xyz_to_cct_search(): Calculates CCT, Duv from XYZ using brute-force search 
                       algorithm (between 1e2 K - _CCT_MAX K)

 :cct_to_mired(): Converts from CCT to Mired scale (or back).

===============================================================================
"""
#from . import _CCT_LUT_CALC

from luxpy import  _WL3, _CMF, _CIEOBS, spd_to_xyz, cri_ref, blackbody, xyz_to_Yxy, xyz_to_Yuv,Yuv_to_xyz
from luxpy.utils import np, pd, sp, _PKG_PATH, _SEP, _EPS, np2d, np2dT, getdata, dictkv

_CCT_MAX = 1e11 # maximum value that does not cause overflow problems
_CCT_LUT_CALC = False # True: (re-)calculates LUTs for ccts in .cctluts/cct_lut_cctlist.dat
__all__ = ['_CCT_LUT_CALC', '_CCT_MAX']

__all__ += ['_CCT_LUT','_CCT_LUT_PATH', 'calculate_lut', 'calculate_luts', 'xyz_to_cct','xyz_to_duv', 'cct_to_xyz',
            'cct_to_mired','xyz_to_cct_ohno','xyz_to_cct_search','xyz_to_cct_search_fast', 'xyz_to_cct_search_robust',
            'xyz_to_cct_HA','xyz_to_cct_mcamy']

#------------------------------------------------------------------------------
_CCT_LUT_PATH = _PKG_PATH + _SEP + 'data'+ _SEP + 'cctluts' + _SEP #folder with cct lut data
_CCT_LUT = {}



#--------------------------------------------------------------------------------------------------
# load / calculate CCT LUT:
def calculate_lut(ccts = None, cieobs = None, add_to_lut = True, wl = _WL3):
    """
    Function that calculates LUT for the ccts stored in 
    ./data/cctluts/cct_lut_cctlist.dat or given as input argument.
    Calculation is performed for CMF set specified in cieobs. 
    Adds a new (temprorary) field to the _CCT_LUT dict.
    
    Args:
        :ccts: 
            | ndarray or str, optional
            | list of ccts for which to (re-)calculate the LUTs.
            | If str, ccts contains path/filename.dat to list.
        :cieobs: 
            | None or str, optional
            | str specifying cmf set.
        :wl: 
            | _WL3, optional
            | Generate luts based on Planckians with wavelengths (range). 
            
    Returns:
        :returns: 
            | ndarray with cct and duv.
        
    Note:
        Function changes the global variable: _CCT_LUT!
    """
    if ccts is None:
        ccts = getdata('{}cct_lut_cctlist.dat'.format(_CCT_LUT_PATH))
    elif isinstance(ccts,str):
        ccts = getdata(ccts)
        
    Yuv = np.zeros((ccts.shape[0],2));Yuv.fill(np.nan)
    for i,cct in enumerate(ccts):
        Yuv[i,:] = xyz_to_Yuv(spd_to_xyz(blackbody(cct, wl3 = wl), cieobs = cieobs))[:,1:3]
    u = Yuv[:,0,None] # get CIE 1960 u
    v = (2.0/3.0)*Yuv[:,1,None] # get CIE 1960 v
    cctuv = np.hstack((ccts,u,v))
    if add_to_lut == True:
        _CCT_LUT[cieobs] = cctuv
    return cctuv 
    
def calculate_luts(ccts = None, wl = _WL3, save_luts = True):
    """
    Function that recalculates (and overwrites) LUTs in ./data/cctluts/ 
    for the ccts stored in ./data/cctluts/cct_lut_cctlist.dat or given as 
    input argument. Calculation is performed for all CMF sets listed 
    in _CMF['types'].
    
    Args:
        :ccts: 
            | ndarray or str, optional
            | List of ccts for which to (re-)calculate the LUTs.
            | If str, ccts contains path/filename.dat to list.
        :wl: 
            | _WL3, optional
            | Generate luts based on Planckians with wavelengths (range). 
        :save_luts:
            | True, optional
            | If True: save luts to folder './data/cctluts/'
            
    Returns:
         | None
        
    """
    luts = {}
    for ii, cieobs in enumerate(sorted(_CMF['types'])):
        print("Calculating CCT LUT for CMF set: {}".format(cieobs))
        cctuv = calculate_lut(ccts = ccts, cieobs = cieobs, add_to_lut = False, wl = wl)
        if save_luts:  
            pd.DataFrame(cctuv).to_csv('{}cct_lut_{}.dat'.format(_CCT_LUT_PATH,cieobs), header=None, index=None, float_format = '%1.9e')
        luts[cieobs] = cctuv
    return luts
        
if _CCT_LUT_CALC == True:
    _CCT_LUT = calculate_luts(wl = _WL3)  

# Initialize _CCT_LUT dict:
try:
    _CCT_LUT = dictkv(keys = sorted(_CMF['types']), values = [getdata('{}cct_lut_{}.dat'.format(_CCT_LUT_PATH,sorted(_CMF['types'])[i]),kind='np') for i in range(len(_CMF['types']))],ordered = False)
except:
    calculate_luts()  
    _CCT_LUT = dictkv(keys = sorted(_CMF['types']), values = [getdata('{}cct_lut_{}.dat'.format(_CCT_LUT_PATH,sorted(_CMF['types'])[i]),kind='np') for i in range(len(_CMF['types']))],ordered = False)
      


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

def _find_closest_ccts(uvw, cieobs = _CIEOBS, ccts = None, wl = _WL3):
    """
    Find closest cct from a list and the two surrounding ccts.
    """
    if ccts is None:
        ccts=np.array([50,100,500,1000,2000,3000,4000,5000,6000,10000,
                       20000,50000,1e5,1e6, 1e7, 1e8,1e9, 1e10, 1e11, _CCT_MAX])
    
    max_cct = ccts[-1]
    
    uv = np.empty((ccts.shape[0],2))
    for i,cct in enumerate(ccts):
        uv[i,:] = xyz_to_Yuv(spd_to_xyz(blackbody(cct, wl3 = wl), cieobs = cieobs))[:,1:3]
    uv[:,1] *= (2.0/3.0) # get CIE 1960 v
    
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
      
def xyz_to_cct_search(xyzw, cieobs = _CIEOBS, out = 'cct',wl = None, rtol = 1e-5, atol = 0.1, 
                      upper_cct_max = _CCT_MAX, approx_cct_temp = True, fast = True, cct_search_list = None):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv(distance above (> 0) or below ( < 0) the Planckian locus) by a 
    brute-force search. 
    
    Wrapper around xyz_to_cct_search_fast() and xyz_to_cct_search_fast()
    
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
            | Stop brute-force search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | brute-force search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop brute-force search when cct a absolute tolerance (K) is reached.
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit brute-force search to this cct.
        :cct_search_list:
            | None, optional
            | list of ccts to obtain a first guess for the cct of the input xyz.
            | None defaults to: [50,100,500,1000,2000,3000,4000,5000,6000,10000,
            |                  20000,50000,1e5,1e6, 1e7, 1e8,1e9, 1e10, 1e11, _CCT_MAX]
            | Only for 'robust' code option.
        :approx_cct_temp: 
            | True, optional
            | If True: use xyz_to_cct_HA() to get a first estimate of cct to 
            |  speed up search.
            | Only for 'fast' code option.
        :fast:
            | True, optional
            | Use fast brute-force search, i.e. xyz_to_cct_search_fast()
        :cct_search_list:
            | None, optional
            | list of ccts to obtain a first guess for the cct of the input xyz 
            | when HA estimation fails due to out-of-range cct or when fast == False.
            | None defaults to: [50,100,500,1000,2000,3000,4000,5000,6000,10000,
            |                  20000,50000,1e5,1e6, 1e7, 1e8,1e9, 1e10, 1e11, _CCT_MAX]

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
    if fast == False:
        return xyz_to_cct_search_robust(xyzw, cieobs = cieobs, out = out, wl = wl, 
                                 rtol = rtol, atol = atol, upper_cct_max = upper_cct_max, 
                                 cct_search_list = cct_search_list)
    else:
        return xyz_to_cct_search_fast(xyzw, cieobs = cieobs, out = out,wl = wl, 
                               rtol = rtol, atol = atol, upper_cct_max = upper_cct_max, 
                               approx_cct_temp = approx_cct_temp, cct_search_list = cct_search_list)
    
  
def xyz_to_cct_search_robust(xyzw, cieobs = _CIEOBS, out = 'cct',wl = None, rtol = 1e-5, atol = 0.1, 
                      upper_cct_max = _CCT_MAX, cct_search_list = None):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv(distance above (> 0) or below ( < 0) the Planckian locus) by a 
    brute-force search. 

    | The algorithm uses an approximate cct_temp as starting point 
    | then constructs, a 4-step section of the blackbody (Planckian) locus 
    | on which to find the minimum distance to the 1960 uv chromaticity of 
    | the test source. The approximate starting point is found by generating 
    | the uv chromaticity values of a set blackbody radiators spread across the
    | locus in a 50 K to _CCT_MAX K range (larger CCT's cause instability of the 
    | chromaticity points due to floating point errors), looking for the closest
    | blackbody radiator and then calculating the mean of the two surrounding ones.
    | The default cct list is [50,100,500,1000,2000,3000,4000,5000,6000,10000,
    |                          20000,50000,1e5,1e6, 1e7, 1e8,1e9, 1e10, 1e11, _CCT_MAX].


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
            | Stop brute-force search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | brute-force search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop brute-force search when cct a absolute tolerance (K) is reached.
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit brute-force search to this cct.
        :cct_search_list:
            | None, optional
            | list of ccts to obtain a first guess for the cct of the input xyz.
            | None defaults to: [50,100,500,1000,2000,3000,4000,5000,6000,10000,
            |                  20000,50000,1e5,1e6, 1e7, 1e8,1e9, 1e10, 1e11, _CCT_MAX]

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
        raise Exception('xyz_to_cct_search(): Input xyzw.shape must be <= 2 !')
       
    # get 1960 u,v of test source:
    Yuvt = xyz_to_Yuv(np.squeeze(xyzw)) # remove possible 1-dim + convert xyzw to CIE 1976 u',v'
    ut = Yuvt[:,1,None] # get CIE 1960 u
    vt = (2/3)*Yuvt[:,2,None] # get CIE 1960 v

    # Initialize arrays:
    ccts = np.zeros((xyzw.shape[0],1));ccts.fill(np.nan)
    duvs = ccts.copy()
        
    #calculate preliminary estimates in 50 K to _CCT_MAX range or whatever is given in cct_search_list:
    ccts_est, cctranges = _find_closest_ccts(np.hstack((ut,vt)), cieobs = cieobs, 
                                             ccts = cct_search_list, wl = wl)
    
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

        while (((delta_cct*2/ccttemp) >= rtol) & (delta_cct*2 >= atol)) & (reached_CCT_MAX == False):# & (ccttemp<upper_cct_max)):# keep converging on CCT 

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
            Yuv = xyz_to_Yuv(np.squeeze(xyz)) # remove possible 1-dim + convert xyz to CIE 1976 u',v'
            u = Yuv[:,1,None] # get CIE 1960 u
            v = (2.0/3.0)*Yuv[:,2,None] # get CIE 1960 v
            
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
                    signduv =np.sign(vt[i]-vBB)

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

def xyz_to_cct_search_fast(xyzw, cieobs = _CIEOBS, out = 'cct',wl = None, 
                           rtol = 1e-5, atol = 0.1, upper_cct_max = _CCT_MAX, 
                           approx_cct_temp = True, cct_search_list = None):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv(distance above (> 0) or below ( < 0) the Planckian locus) by a 
    brute-force search. 

    | The algorithm uses an approximate cct_temp (HA approx., see xyz_to_cct_HA) 
    |  as starting point or uses the middle of the allowed cct-range 
    |  (1e2 K - _CCT_MAX K, higher causes overflow) on a log-scale, then constructs 
    |  a 4-step section of the blackbody (Planckian) locus on which to find the
    |  minimum distance to the 1960 uv chromaticity of the test source.
    | If HA fails then another approximate starting point is found by generating 
    | the uv chromaticity values of a set blackbody radiators spread across the
    | locus in a 50 K to _CCT_MAX K range (larger CCT's cause instability of the 
    | chromaticity points due to floating point errors), looking for the closest
    | blackbody radiator and then calculating the mean of the two surrounding ones.
    | The default cct list is [50,100,500,1000,2000,3000,4000,5000,6000,10000,
    |                          20000,50000,1e5,1e6, 1e7, 1e8,1e9, 1e10, 1e11, _CCT_MAX].


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
            | Stop brute-force search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | brute-force search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop brute-force search when cct a absolute tolerance (K) is reached.
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit brute-force search to this cct.
            | Note that values > _CCT_MAX give overflow problems.
        :approx_cct_temp: 
            | True, optional
            | If True: use xyz_to_cct_HA() to get a first estimate of cct to 
              speed up search.
        :cct_search_list:
            | None, optional
            | list of ccts to obtain a first guess for the cct of the input xyz 
            | when HA estimation fails due to out-of-range cct.
            | None defaults to: [50,100,500,1000,2000,3000,4000,5000,6000,10000,
            |                  20000,50000,1e5,1e6, 1e7, 1e8,1e9, 1e10, 1e11, _CCT_MAX]

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
        raise Exception('xyz_to_cct_search(): Input xyzw.shape must be <= 2 !')
       
    # get 1960 u,v of test source:
    Yuvt = xyz_to_Yuv(np.squeeze(xyzw)) # remove possible 1-dim + convert xyzw to CIE 1976 u',v'
    #axis_of_v3t = len(Yuvt.shape)-1 # axis containing color components
    ut = Yuvt[:,1,None] #.take([1],axis = axis_of_v3t) # get CIE 1960 u
    vt = (2/3)*Yuvt[:,2,None] #.take([2],axis = axis_of_v3t) # get CIE 1960 v

    # Initialize arrays:
    ccts = np.zeros((xyzw.shape[0],1));ccts.fill(np.nan)
    duvs = ccts.copy()

    #calculate preliminary solution(s):
    if (approx_cct_temp == True):
        ccts_est = xyz_to_cct_HA(xyzw, verbosity = 0)
        procent_estimates = np.array([[3000.0, 100000.0,0.05],[100000.0,200000.0,0.1],[200000.0,300000.0,0.25],[300000.0,400000.0,0.4],[400000.0,600000.0,0.4],[600000.0,800000.0,0.4],[800000.0,np.inf,0.25]])
        ccts_est[np.isnan(ccts_est)] =  -2 # recode to avoid "RuntimeWarning: invalid value encountered in less"
        if ((np.isnan(ccts_est).any()) | (ccts_est == -2).any() | (ccts_est == -1).any()) | ((ccts_est < procent_estimates[0,0]).any() | (ccts_est > procent_estimates[-2,1]).any()):
            
            #calculate preliminary estimates in 50 K to _CCT_MAX range or whatever is given in cct_search_list:
            ccts_est, cct_ranges = _find_closest_ccts(np.hstack((ut,vt)), cieobs = cieobs, wl = wl)
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

        while (((delta_cct*2) >= atol) & ((delta_cct*2/ccttemp) >= rtol)) & (reached_CCT_MAX == False):# keep converging on CCT 

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
            Yuv = xyz_to_Yuv(np.squeeze(xyz)) # remove possible 1-dim + convert xyz to CIE 1976 u',v'

            u = Yuv[:,1,None] # get CIE 1960 u
            v = (2.0/3.0)*Yuv[:,2,None] # get CIE 1960 v
            
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

def xyz_to_cct_ohno(xyzw, cieobs = _CIEOBS, out = 'cct', wl = None, rtol = 1e-5, atol = 0.1, 
                    force_out_of_lut = True, upper_cct_max = _CCT_MAX, 
                    approx_cct_temp = True, cct_search_list = None, fast_search = True,
                    cctuv_lut = None):
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and 
    Duv (distance above (>0) or below (<0) the Planckian locus) 
    using Ohno's method. 
    
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
            | Stop brute-force search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | brute-force search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop brute-force search when cct a absolute tolerance (K) is reached.
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit brute-force search to this cct.
        :approx_cct_temp: 
            | True, optional
            | If True: use xyz_to_cct_HA() to get a first estimate of cct to 
            |  speed up search.
            | Only for 'fast' code option.
        :fast_search:
            | True, optional
            | Use fast brute-force search, i.e. xyz_to_cct_search_fast()
        :cct_search_list:
            | None, optional
            | list of ccts to obtain a first guess for the cct of the input xyz 
            | when HA estimation fails due to out-of-range cct.
            | None defaults to: [50,100,500,1000,2000,3000,4000,5000,6000,10000,
            |                  20000,50000,1e5,1e6, 1e7, 1e8,1e9, 1e10, 1e11, _CCT_MAX]
        :force_out_of_lut: 
            | True, optional
            | If True and cct is out of range of the LUT, then switch to 
            | brute-force search method, else return numpy.nan values.
        :cctuv_lut:
            | None, optional
            | CCT+uv look-up-table to use.
            | If None: use luxpy._CCT_LUT
        
    Returns:
        :returns: 
            | ndarray with:
            |    cct: out == 'cct' (or 1)
            |    duv: out == 'duv' (or -1)
            |    cct, duv: out == 'cct,duv' (or 2)
            |    [cct,duv]: out == "[cct,duv]" (or -2) 
            
    Note:
        Default LUTs are stored in ./data/cctluts/
        
    Reference:
        1. `Ohno Y. Practical use and calculation of CCT and Duv. 
        Leukos. 2014 Jan 2;10(1):47-55.
        <http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020>`_
    """

    xyzw = np2d(xyzw)  

    if len(xyzw.shape)>2:
        raise Exception('xyz_to_cct_ohno(): Input xyzwa.ndim must be <= 2 !')
      
    # get 1960 u,v of test source:
    Yuv = xyz_to_Yuv(xyzw) # remove possible 1-dim + convert xyzw to CIE 1976 u',v'
    axis_of_v3 = len(Yuv.shape)-1 # axis containing color components
    u = Yuv[:,1,None] # get CIE 1960 u
    v = (2.0/3.0)*Yuv[:,2,None] # get CIE 1960 v

    uv = np2d(np.concatenate((u,v),axis = axis_of_v3))
    
    # load cct & uv from LUT:
    if cctuv_lut is None:   
        cctuv_lut = _CCT_LUT
    else:
        if not isinstance(cctuv_lut,dict):
            cctuv_lut = {cieobs:cctuv_lut}
    if cieobs not in cctuv_lut:
        cctuv_lut[cieobs] = calculate_lut(ccts = None, cieobs = cieobs, add_to_lut = False, wl = wl)
    cct_LUT = cctuv_lut[cieobs][:,0,None] 
    uv_LUT = cctuv_lut[cieobs][:,1:3] 
    
    # calculate CCT of each uv:
    CCT = np.zeros(uv.shape[0]);CCT.fill(np.nan) # initialize with NaN's
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
            cct_i, Duv_i = xyz_to_cct_search(xyzw[i:i+1,:], cieobs = cieobs, wl = wl, rtol = rtol, atol = atol,
                                             out = 'cct,duv',upper_cct_max = upper_cct_max, 
                                             approx_cct_temp = approx_cct_temp, cct_search_list = cct_search_list,
                                             fast = fast_search)
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


#---------------------------------------------------------------------------------------------------
def cct_to_xyz(ccts, duv = None, cieobs = _CIEOBS, wl = None, mode = 'lut', out = None, 
               rtol = 1e-5, atol = 0.1, force_out_of_lut = True, upper_cct_max = _CCT_MAX, 
               approx_cct_temp = True, fast_search = True, cct_search_list = None,
               cctuv_lut = None):
    """
    Convert correlated color temperature (CCT) and Duv (distance above (>0) or 
    below (<0) the Planckian locus) to XYZ tristimulus values.
    
    | Finds xyzw_estimated by minimization of:
    |    
    |    F = numpy.sqrt(((100.0*(cct_min - cct)/(cct))**2.0) 
    |         + (((duv_min - duv)/(duv))**2.0))
    |    
    | with cct,duv the input values and cct_min, duv_min calculated using 
    | luxpy.xyz_to_cct(xyzw_estimated,...).
    
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
            | 'lut' or 'search', optional
            | Determines what method to use.
        :out: 
            | None (or 1), optional
            | If not None or 1: output a ndarray that contains estimated 
            | xyz and minimization results: 
            | (cct_min, duv_min, F_min (objective fcn value))
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
        :rtol: 
            | 1e-5, float, optional
            | Stop brute-force search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | brute-force search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop brute-force search when cct a absolute tolerance (K) is reached.
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit brute-force search to this cct.
        :approx_cct_temp: 
            | True, optional
            | If True: use xyz_to_cct_HA() to get a first estimate of cct to 
            |  speed up search.
            | Only for 'fast' code option.
        :fast_search:
            | True, optional
            | Use fast brute-force search, i.e. xyz_to_cct_search_fast()
        :cct_search_list:
            | None, optional
            | list of ccts to obtain a first guess for the cct of the input xyz 
            | when HA estimation fails due to out-of-range cct or when fast_search == False.
            | None defaults to: [50,100,500,1000,2000,3000,4000,5000,6000,10000,
            |                  20000,50000,1e5,1e6, 1e7, 1e8,1e9, 1e10, 1e11, _CCT_MAX]
        :force_out_of_lut: 
            | True, optional
            | If True and cct is out of range of the LUT, then switch to 
            | brute-force search method, else return numpy.nan values.
        :cctuv_lut:
            | None, optional
            | CCT+uv look-up-table to use.
            | If None: use luxpy._CCT_LUT
        
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
    elif duv is not None:
        duv = np2d(duv)

    #get estimates of approximate xyz values in case duv = None:
    BB = cri_ref(ccts = cct, wl3 = wl, ref_type = ['BB'])
    xyz_est = spd_to_xyz(data = BB, cieobs = cieobs, out = 1)
    results = np.zeros([ccts.shape[0],3]);results.fill(np.nan) 

    if duv is not None:
        
        # optimization/minimization setup:
        def objfcn(uv_offset, uv0, cct, duv, out = 1):#, cieobs = cieobs, wl = wl, mode = mode):
            uv0 = np2d(uv0 + uv_offset)
            Yuv0 = np.concatenate((np2d([100.0]), uv0),axis=1)
            cct_min, duv_min = xyz_to_cct(Yuv_to_xyz(Yuv0),cieobs = cieobs, out = 'cct,duv',
                                          wl = wl, mode = mode, rtol = rtol, atol = atol, 
                                          force_out_of_lut = force_out_of_lut, 
                                          upper_cct_max = upper_cct_max, 
                                          approx_cct_temp = approx_cct_temp,
                                          cct_search_list = cct_search_list,
                                          fast_search = fast_search,
                                          cctuv_lut = cctuv_lut)
            
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
                                           mode = mode, rtol = rtol, atol = atol, 
                                           force_out_of_lut = force_out_of_lut, 
                                           upper_cct_max = upper_cct_max, 
                                           approx_cct_temp = approx_cct_temp,
                                           cct_search_list = cct_search_list,
                                           fast_search = fast_search,
                                           cctuv_lut = cctuv_lut)
            
            if np.abs(duv[i]) > _EPS:
                # find xyz:
                Yuv0 = xyz_to_Yuv(xyz0)
                uv0 = Yuv0[0] [1:3]

                OptimizeResult = sp.optimize.minimize(fun = objfcn,x0 = np.zeros((1,2)), args = (uv0,cct_i, duv_i, 'F'), method = 'Nelder-Mead',options={"maxiter":np.inf, "maxfev":np.inf, 'xatol': 0.000001, 'fatol': 0.000001})
                betas = OptimizeResult['x']
                #betas = np.zeros(uv0.shape)
                if out is not None:
                    results[i] = objfcn(betas,uv0,cct_i, duv_i, out = 3)
                
                uv0 = np2d(uv0 + betas)
                Yuv0 = np.concatenate((np2d([100.0]),uv0),axis=1)
                xyz_est[i] = Yuv_to_xyz(Yuv0)
            
            else:
                xyz_est[i] = xyz0
      
    if (out is None) | (out == 1):
        return xyz_est
    else:
        # Also output results of minimization:
        return np.concatenate((xyz_est,results),axis = 1)  


#-------------------------------------------------------------------------------------------------   
# general CCT-wrapper function
def xyz_to_cct(xyzw, cieobs = _CIEOBS, out = 'cct',mode = 'lut', wl = None, rtol = 1e-5, atol = 0.1, 
               force_out_of_lut = True, upper_cct_max = _CCT_MAX, 
               approx_cct_temp = True, fast_search = True, cct_search_list = None,
               cctuv_lut = None): 
    """
    Convert XYZ tristimulus values to correlated color temperature (CCT) and
    Duv (distance above (>0) or below (<0) the Planckian locus)
    using either the brute-force search method or Ohno's method. 
    
    | Wrapper function for use with luxpy.colortf().
    
    Args:
        :xyzw:
            | ndarray of tristimulus values
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set used to calculated xyzw.
        :mode: 
            | 'lut' or 'search', optional
            | Determines what method to use.
        :out: 
            | 'cct' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
        :rtol: 
            | 1e-5, float, optional
            | Stop brute-force search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | brute-force search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop brute-force search when cct a absolute tolerance (K) is reached.
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit brute-force search to this cct.
        :approx_cct_temp: 
            | True, optional
            | If True: use xyz_to_cct_HA() to get a first estimate of cct to 
            |  speed up search.
            | Only for 'fast' code option.
        :fast_search:
            | True, optional
            | Use fast brute-force search, i.e. xyz_to_cct_search_fast()
        :cct_search_list:
            | None, optional
            | list of ccts to obtain a first guess for the cct of the input xyz 
            | when HA estimation fails due to out-of-range cct or when fast_search == False.
            | None defaults to: [50,100,500,1000,2000,3000,4000,5000,6000,10000,
            |                  20000,50000,1e5,1e6, 1e7, 1e8,1e9, 1e10, 1e11, _CCT_MAX]
        :force_out_of_lut: 
            | True, optional
            | If True and cct is out of range of the LUT, then switch to 
            | brute-force search method, else return numpy.nan values.
        :cctuv_lut:
            | None, optional
            | CCT+uv look-up-table to use.
            | If None: use luxpy._CCT_LUT
        
    Returns:
        :returns: 
            | ndarray with:
            |   cct: out == 'cct' (or 1)
            | Optional: 
            |     duv: out == 'duv' (or -1), 
            |    cct, duv: out == 'cct,duv' (or 2), 
            |    [cct,duv]: out == "[cct,duv]" (or -2)
    """
    if (mode == 'lut') | (mode == 'ohno'):
        return xyz_to_cct_ohno(xyzw = xyzw, cieobs = cieobs, out = out, rtol = rtol, atol = atol, force_out_of_lut = force_out_of_lut, wl = wl,
                               upper_cct_max = upper_cct_max, approx_cct_temp = approx_cct_temp, cct_search_list = cct_search_list, fast_search = fast_search, cctuv_lut = cctuv_lut)
    elif (mode == 'search'):
        return xyz_to_cct_search(xyzw = xyzw, cieobs = cieobs, out = out, wl = wl, rtol = rtol, atol = atol, upper_cct_max = upper_cct_max, approx_cct_temp = approx_cct_temp, cct_search_list = cct_search_list, fast = fast_search)


def xyz_to_duv(xyzw, cieobs = _CIEOBS, out = 'duv', mode = 'lut', wl = None,
               rtol = 1e-5, atol = 0.1, force_out_of_lut = True, upper_cct_max = _CCT_MAX, 
               approx_cct_temp = True, fast_search = True, cct_search_list = None,
               cctuv_lut = None): 
    """
    Convert XYZ tristimulus values to Duv (distance above (>0) or below (<0) 
    the Planckian locus) and correlated color temperature (CCT) values
    using either the brute-force search method or Ohno's method. 
    
    | Wrapper function for use with luxpy.colortf().
    
    Args:
        :xyzw: 
            | ndarray of tristimulus values
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set used to calculated xyzw.
        :mode: 
            | 'lut' or 'search', optional
            | Determines what method to use.
        :out: 
            | 'duv' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)
        :wl: 
            | None, optional
            | Wavelengths used when calculating Planckian radiators.
        :rtol: 
            | 1e-5, float, optional
            | Stop brute-force search when cct a relative tolerance is reached.
            | The relative tolerance is calculated as dCCT/CCT_est, 
            | with CCT_est the current intermediate estimate in the 
            | brute-force search and with dCCT the difference between
            | the present and former estimates.
        :atol: 
            | 0.1, optional
            | Stop brute-force search when cct a absolute tolerance (K) is reached.
        :upper_cct_max: 
            | _CCT_MAX, optional
            | Limit brute-force search to this cct.
        :approx_cct_temp: 
            | True, optional
            | If True: use xyz_to_cct_HA() to get a first estimate of cct to 
            |  speed up search.
            | Only for 'fast' code option.
        :fast_search:
            | True, optional
            | Use fast brute-force search, i.e. xyz_to_cct_search_fast()
        :cct_search_list:
            | None, optional
            | list of ccts to obtain a first guess for the cct of the input xyz 
            | when HA estimation fails due to out-of-range cct or when fast_search == False.
            | None defaults to: [50,100,500,1000,2000,3000,4000,5000,6000,10000,
            |                  20000,50000,1e5,1e6, 1e7, 1e8,1e9, 1e10, 1e11, _CCT_MAX]
        :force_out_of_lut: 
            | True, optional
            | If True and cct is out of range of the LUT, then switch to 
            | brute-force search method, else return numpy.nan values.
        :cctuv_lut:
            | None, optional
            | CCT+uv look-up-table to use.
            | If None: use luxpy._CCT_LUT
        
    Returns:
        :returns:
            | ndarray with:
            |   duv: out == 'duv' (or -1)
            | Optional: 
            |     duv: out == 'duv' (or -1), 
            |     cct, duv: out == 'cct,duv' (or 2), 
            |     [cct,duv]: out == "[cct,duv]" (or -2)
    """
    if (mode == 'lut') | (mode == 'ohno'):
        return xyz_to_cct_ohno(xyzw = xyzw, cieobs = cieobs, out = out, rtol = rtol, atol = atol, force_out_of_lut = force_out_of_lut, wl = wl,
                               upper_cct_max = upper_cct_max, approx_cct_temp = approx_cct_temp, cct_search_list = cct_search_list, fast_search = fast_search, cctuv_lut = cctuv_lut)
    elif (mode == 'search'):
        return xyz_to_cct_search(xyzw = xyzw, cieobs = cieobs, out = out, wl = wl, rtol = rtol, atol = atol, upper_cct_max = upper_cct_max, approx_cct_temp = approx_cct_temp, cct_search_list = cct_search_list, fast = fast_search)
   
   
#-------------------------------------------------------------------------------------------------   
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