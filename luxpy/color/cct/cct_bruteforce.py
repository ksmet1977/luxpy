"""
# Brute-force CCT & Duv calculator
#-------------------------------------------------------------------------

 :_get_ccts_for_lut_bf(): Calculates CCTs for a LUT.

 :generate_lut_bf(): Calculate a Look-Up-Table for CCT & Duv calculations.

 :xyz_to_cct_bruteforce: Calculate CCT, Duv from XYZ using a brute-force technique.

""" 
# #------luxpy import-------------------------------------------------------
# import os 
# if "code\\py" not in os.getcwd(): os.chdir("./code/py/")
# root = "../../" if "code\\py" in os.getcwd() else  "./"

# import luxpy as lx
# print('luxpy version:', lx.__version__,'\n')

__all__ = ['_CCT_LUT_BRUTEFORCE_1931_2_uv60','_get_ccts_for_lut_bf', 'generate_lut_bf', 'xyz_to_cct_bruteforce']

from luxpy import _CMF, cie_interp, blackbody, spd_to_xyz, spd_to_xyz_barebones, getwlr, xyz_to_Yuv60, Yxy_to_xyz, _CMF 
from luxpy.color.cct.cct import _get_newton_raphson_estimated_Tc, _process_cspace
from luxpy.color.ctf.colortf import colortf
from luxpy import math

import numpy as np

def _get_ccts_for_lut_bf(start = 1000, end = 41000, interval = 0.25, unit = '%'):
    if unit == '%':
        n = np.ceil(np.log(end/start)/np.log(1+interval/100))+1
        ccts = start*(1+interval/100)**np.arange(n)
        ccts = np.hstack((start*(1+interval/100)**(-1), ccts))
        ccts[ccts==0] = 1e-16
    elif unit == 'K':
        n = np.ceil((end - start)/interval)
        ccts = start + (np.arange(n+1)-1)*interval
        ccts[ccts==0] = 1e-16
    elif unit == 'K-1':
        n = np.ceil((1e6/start - 1e6/end)/interval) + 2
        ccts = (1e6/start - (np.arange(n)-1)*interval)
        ccts[ccts==0] = 1e-16
        ccts=1e6/ccts
    elif unit == '%-1':
        n = np.ceil(np.log((1e6/start)/(1e6/end))/np.log(1+interval/100)) + 1
        ccts = (1e6/start)/(1+interval/100)**np.arange(n)
        ccts[ccts==0] = 1e-16
        ccts=1e6/ccts
        ccts = np.hstack((1e6/((1e6/start)*(1+interval/100)),ccts))
    return ccts

def _check_sampling_interval(unit, interval, down_sampling_factor):
    """ Avoid division by zero crashing the code when LUT CCT spacing interval become too small. """
    if '%' in unit:
        return np.log(1+interval/(down_sampling_factor*100)) > 0
    elif 'K' in unit:
        return (interval/down_sampling_factor) > 0
    else:
        raise Exception('Unknown unit')

def _get_start_end(cctmin, interval, unit):
    interval = 2*interval
    if unit == '%':
        start = (cctmin/(1+interval/100))
        end = (cctmin*(1+interval/100))
    elif unit == 'K':
        start = cctmin - interval 
        end = cctmin + interval 
    elif unit == 'K-1':
        start =  1e6/(1e6/(cctmin) + interval) 
        end =  1e6/(1e6/(cctmin) - interval) 
    elif unit == '%-1':
        start = 1e6/((1e6/cctmin)*(1+interval/100))
        end = 1e6/((1e6/cctmin)/(1+interval/100))
    return start, end

def generate_lut_bf(ccts = None, start = 1000, end = 41000, interval = 0.25, unit = '%', 
                 wl = None, cmfs = "1931_2", cspace = 'Yuv60', cspace_kwargs = {}):
    """
    Calculate a Look-Up-Table for CCT & Duv calculations.

    Args:
        :ccts:
            | None, optional
            | If not None: use this specific list or ndarray of CCTs.
        :start:
            | 1000, optional
            | Start in CCT (LUT also has one lower CCT)
        :end:
            | 41000, optional
            | End at this CCT (LUT also has a higher CCT)
        :interval:
            | 0.25, optional
            | Interval to go from one to the next CCT in the LUT
            | (:unit: determines exactly how much this number increases the CCT)
        :unit:
            | '%', optional
            | Options:
            |   - '%': cct[i+1] = cct[i]*(1 + interval/100)
            |   - 'K': cct[i+1] = cct[i] + interval
            |   - '%-1': 1e6/cct[i+1] = (1e6/cct[i])*(1 + interval/100)
            |   - 'K-1': 1e6/cct[i+1] = 1e6/cct[i] + interval
        :wl:
            | None, optional
            | If None: use same wavelengths as from cmf set to generate blackbody radiators
        :cmf:
            | "1931_2", optional
            | String specifying or ndarray with CMF set.
        :cspace:
            | 'Yuv60', optional
            | String specifying the color or chromaticity space to calculate 
            | the distance to the blackbody locus in. 
            | (uses luxpy.colortf)
        :cspace_kwargs:
            | {}, optional
            | A dict with any kwargs for the xyz_to_space function 
            | (cfr. luxpy.colortf(xyz, fwtf = cspace_kwargs)).

    Returns:
        :lut:
            | ndarray [nx3] with CCT, u, v coordinates 
            |   (or whatever equivalent coordinates for the selected cspace)
    """
    if ccts is None: 
        ccts = _get_ccts_for_lut_bf(start = start, end = end, interval = interval, unit = unit)

    if isinstance(cmfs, str): cmfs = _CMF[cmfs]["bar"].copy()
    wl = getwlr(wl) if (wl is not None) else cmfs[0]
    if not np.array_equal(wl, cmfs[0]): cmfs = cie_interp(cmfs, wl, datatype = 'cmf')

    BBs = np.zeros((len(ccts),len(wl)))
    for i, cct in enumerate(ccts):
        BBs[i] = blackbody(cct, wl3 = wl)[1]
    BBs = np.vstack((wl,BBs))
    #xyz = spd_to_xyz(BBs, cmfs, relative = True)
    xyz,_ = spd_to_xyz_barebones(BBs, cmfs, relative = True)
    xyz = xyz[0] # needed when using spd_to_xyz_barebones
    #print('luxpy: xyz0', xyz[0,0,0],xyz[0,0,1],xyz[0,0,2])
    Yuv60 = colortf(xyz, tf = cspace, fwtf = cspace_kwargs) # default is cspace = 'Yuv60'
    lut = np.hstack((np.atleast_2d(ccts).T,Yuv60[...,1:]))
    return lut

_CCT_LUT_BRUTEFORCE_1931_2_uv60 = generate_lut_bf(ccts = None, start = 1000, end = 41000, interval = 0.25, unit = '%', 
                                                  wl = None, cmfs = "1931_2",cspace = 'Yuv60')

def _get_Duv_for_T_from_uvBB(u,v, uBB0, vBB0, sign_only = False):
    """ 
    Calculate Duv from (u,v) coordinates of estimated Tc.
    """
    # Get duv: 
    du, dv = u - uBB0, v - vBB0

    # find sign of duv:
    theta = math.positive_arctan(du,dv,htype='deg')
    theta[theta>180] = theta[theta>180] - 360
    sign = np.sign(theta)
    if sign_only: 
        return sign
    else:
        return sign*(du**2 + dv**2)**0.5 
    
    
def xyz_to_cct_bruteforce(xyz, wl = None, cmfs = "1931_2", atol = 1e-15, rtol = 1e-20, n_max = 1e4, down_sampling_factor = 10,
                          ccts = None, start = 1000, end = 41000, interval = 0.25, unit = '%',
                          cspace = 'Yuv60', cspace_kwargs = {}, lut = _CCT_LUT_BRUTEFORCE_1931_2_uv60,
                          use_newton_raphson = False, fast_duv = True, out = 'cct'):
    """
    Calculates the CCT (and Duv) value for a set of tristimulus values using 
    brute-force approach. The method start by generating a large LUT, finds 
    CCT with a minimum distance to the blackbody locus. Then further iterates 
    over smaller and smaller CCT-ranges by generating new LUTs, until the solution
    converges to a specified tolerance or until a maximum number of iterations 
    is reached. 

    Args:
        :xyz:
            | ndarray of tristimulus values XYZ. [nx3]
        :wl:
            | None, optional
            | If None: use same wavelengths as from cmf set to generate blackbody radiators for LUT.
        :cmf:
            | "1931_2", optional
            | String specifying or ndarray with CMF set to use for LUT computation.
        :atol:
            | 0.1, optional
            | Absolute tolerance in Kelvin. If the difference between the two surrounding CCTs is smaller than tol, 
            | the brute-force search stops.
        :n_max:
            | 1000, optional
            | Maximum number of iterations that a more detailed LUT is generated.
            | If the number of iterations > n_max, the brute-force search stops.
        :down_sampling_factor:
            | 10, optional
            | Value by which the original interval is further downsampled at each iteration.
        :ccts:
            | None, optional
            | If not None: use this specific list or ndarray of CCTs.
        :start:
            | 1000, optional
            | Start in CCT (LUT also has one lower CCT)
        :end:
            | 41000, optional
            | End at this CCT (LUT also has a higher CCT)
        :interval:
            | 0.25, optional
            | Interval to go from one to the next CCT in the LUT
            | (:unit: determines exactly how much this number increases the CCT)
        :unit:
            | '%', optional
            | Options:
            |   - '%': cct[i+1] = cct[i]*(1 + interval/100)
            |   - 'K': cct[i+1] = cct[i] + interval
            |   - '%-1': 1e6/cct[i+1] = (1e6/cct[i])*(1 + interval/100)
            |   - 'K-1': 1e6/cct[i+1] = 1e6/cct[i] + interval
        :cspace:
            | 'Yuv60', optional
            | String specifying the color or chromaticity space to calculate 
            | the distance to the blackbody locus in. 
            | (uses luxpy.colortf)
        :cspace_kwargs:
            | {}, optional
            | A dict with any kwargs for the xyz_to_space function 
            | (cfr. luxpy.colortf(xyz, fwtf = cspace_kwargs)).
        :lut:
            | _CCT_LUT_BRUTEFORCE_1931_2_uv60, optonal
            | Pre-calculated LUT: Lut for CIE uv 1960 coordinates and CIE 1931 2° CMFs.  
            | If not None, this LUT is used instead of generating a new one (= only starting lut).
        :use_newton_raphson:
            | False, optional
            | If True: use Newton-Raphson method to find the exact CCT. 
            | Much faster than brute-force search.
        :fast_duv:
            | True, optional
        :out: 
            | 'cct' (or 1), optional
            | Determines what to return.
            | Other options: 'duv' (or -1), 'cct,duv'(or 2), "[cct,duv]" (or -2)

            
     Returns:
        :CCT_Duv:
            | ndarray(s) with:
            |    cct: out == 'cct' (or 1)
            |    duv: out == 'duv' (or -1)
            |    cct, duv: out == 'cct,duv' (or 2)
            |    [cct,duv]: out == "[cct,duv]" (or -2) 
    """

    uv60 = colortf(xyz, tf = cspace, fwtf = cspace_kwargs)[...,1:] # default is cspace = 'Yuv60'


    if lut is None: 
        lut = generate_lut_bf(ccts = ccts, start = start, end = end, interval = interval, unit = unit, wl = wl, cmfs = cmfs,
                              cspace = cspace, cspace_kwargs = cspace_kwargs)

    duv = ((lut[:,1:][:,None,:]-uv60)**2).sum(axis=-1)**0.5
    cctmin = lut[duv.argmin(0)][:,0]
    if use_newton_raphson:
        # Newton-Raphson method to find the exact CCT
        #xyzbar = _CMF[cmfs]["bar"].copy() if isinstance(cmfs,str) else cmfs 
        # Process cspace-parameters:
        cspace_dict, cspace_str = _process_cspace(cspace, cspace_kwargs)
        cctduv = np.hstack(_get_newton_raphson_estimated_Tc(uv60[...,0:1], uv60[...,1:2], np.atleast_2d(cctmin).T, 
                                                wl = wl, atol = atol, rtol = rtol,
                                                cieobs = cmfs, xyzbar = None,
                                                cspace_dict = cspace_dict, 
                                                max_iter = n_max, fast_duv = fast_duv))


        
    else:
        if wl is not None: wl = getwlr(wl)
        start, end = _get_start_end(cctmin, interval, unit)
        start_end = zip(start,end)
        cctduv = np.nan*np.ones_like(uv60)
        for i,start_end_i in enumerate(start_end):
            start_i, end_i = start_end_i
            delta_cct_i = end_i - start_i
            n = 0
            interval_i = interval/down_sampling_factor
            while True & _check_sampling_interval(unit, interval_i, 1):
                n += 1
                lut_i = generate_lut_bf(start = start_i, end = end_i, interval = interval_i, unit = unit, wl = wl, cmfs = cmfs, 
                                        cspace = cspace, cspace_kwargs = cspace_kwargs)
                duv_i = ((lut_i[:,1:][:,None,:]-uv60[i:i+1,:])**2).sum(axis=-1)**0.5
                cctmin_i = lut_i[duv_i.argmin(0)][0,0]
                #if i < 1: print('luxpy: i,n:',i,n,duv_i.min(0)[0],duv_i.argmin(0)[0],start_i,end_i,cctmin_i,interval_i)
                start_i, end_i = _get_start_end(cctmin_i, interval_i, unit)
                delta_cct_i = end_i - start_i
                interval_i = interval_i/down_sampling_factor
                if (delta_cct_i <= atol) | (delta_cct_i/cctmin_i <= rtol) | (n >= n_max): break
                
            u,v = uv60[i:i+1,0], uv60[i:i+1,1]
            uBB0, vBB0 = lut_i[duv_i.argmin(0)][0,1], lut_i[duv_i.argmin(0)][0,2]
            sign = _get_Duv_for_T_from_uvBB(u,v, uBB0, vBB0, sign_only = True)
            cctduv[i] = np.hstack((cctmin_i,sign*duv_i.min(0)[0])) 
    
    # Regulate output:
    if (out == 'cct') | (out == 1):
        return cctduv[:,:1]
    elif (out == 'duv') | (out == -1):
        return cctduv[:,1:]
    elif (out == 'cct,duv') | (out == 2):
        return cctduv[:,:1],cctduv[:,1:]
    elif (out == "[cct,duv]") | (out == -2):
        return cctduv   
    else:
        raise Exception('Unknown output requested')


if __name__ == '__main__':
    import luxpy as lx 

    spds = np.round(lx._IESTM3018['S']['data'].copy(),15)
    spd = spds[[0,1,1,1,1]]

    cmf2,K2 = lx._CMF["1931_2"]["bar"].copy(),lx._CMF["1931_2"]["K"]
    cmf2_ = lx.xyzbar(cieobs="1931_2")
    print('cmf2 == cmf2_ ?:', (((cmf2-cmf2_)==0).all()))

    cctduv_calculator = np.array([6425.14268404576,0.00719612617474981])

    force_tolerance = False

    xyz1 = lx.spd_to_xyz(spd,cieobs='1931_2',relative=False)

    cctduv = xyz_to_cct_bruteforce(xyz1, wl = None, cmfs = "1931_2", n_max = 100, down_sampling_factor = 2,
                                start = 1100, end = 41000, interval = 25, unit ='K-1', cspace = 'Yuv60')
    print('cctduv: \n',cctduv)


