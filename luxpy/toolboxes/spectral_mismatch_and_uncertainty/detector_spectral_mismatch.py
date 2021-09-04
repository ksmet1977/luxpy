# -*- coding: utf-8 -*-
"""
Module for detector spectral responsivity spectral mismatch calculations
========================================================================

 :f1prime(): Determine the f1prime spectral mismatch index.
 
 :get_spectral_mismatch_correction_factors(): Determine the spectral mismatch factors.

Reference:
    1. Krüger, U. et al. GENERAL V(λ) MISMATCH - INDEX HISTORY, CURRENT STATE, NEW IDEAS
    
Created on Wed Aug 25 13:02:00 2021

@author: ksmet1977 [at] gmail.com
"""
from luxpy.spectrum import _CIE_ILLUMINANTS, _CMF, cie_interp, getwlr, getwld
import numpy as np

__all__ = ['f1prime','get_spectral_mismatch_correction_factors']

def f1prime(s_detector, S_C = 'A', 
            cieobs = '1931_2', s_target_index = 2,
            wlr = [380,780, 1], interp_kind = 'linear', 
            out = 'f1p'):
    """
    Determine the f1prime spectral mismatch index.
    
    Args:
        :s_detector:
            | ndarray with detector spectral responsivity (first row = wavelengths)
        :S_C:
            | 'A', optional
            | Standard 'calibration' illuminant.
            | string specifying the illuminant to use from the luxpy._CIE_ILLUMINANTS dict 
            | or ndarray with standard illuminant spectral data.
        :cieobs:
            | '1931_2', optional
            | string with CIE standard observer color matching functions to use (from luxpy._CMF)
            | or ndarray with CMFs (s_target_index > 0) 
            | or target spectral responsivity (s_target_index == 0)
            | (first row contains the wavelengths).
        :s_target_index:
            | 2, optional
            | if > 0: index into CMF set (1->'xbar', 2->'ybar'='Vlambda', 3->'zbar')
            | if == 0: cieobs is expected to contain an ndarray with the target spectral responsivity. 
        :wlr:
            | [380,780,1], optional
            | Wavelength range (ndarray or [start, stop, spacing]).
        :interp_kind:
            | 'linear', optional
            | Interpolation type to use when interpolating function to specified wavelength range.
        :out:
            | 'f1p', optional
            | Specify requested output of function, 
            |    e.g. 'f1p,s_rel' also outputs the normalized target spectral responsitivity. 
    Returns:
        :f1p:
            | ndarray (vector) with f1prime values for each of the spectral responsivities in s_detector.
    """
    
    # Get illuminant:
    if isinstance(S_C,str): S_C = _CIE_ILLUMINANTS[S_C].copy()
    
    # Get target function from cieobs: 
    if s_target_index == 0: s_target_index = 1
    s_target = _CMF[cieobs]['bar'][[0,s_target_index]].copy() if isinstance(cieobs, str) else cieobs[[0,s_target_index]].copy()
    
    # Interpolate to desired wavelength range:
    #wlr = getwlr(wlr) # get wavelength range from array or '3-vector'
    wlr = s_detector[0] # get wavelength range from the detector data
    dl = getwld(wlr) # wavelength differences (unequal wavelength spacings are taken into account)
    s_detector = cie_interp(s_detector, wlr, kind = interp_kind)[1:]
    s_target = cie_interp(s_target, wlr, kind = interp_kind)[1:]
    S_C = cie_interp(S_C, wlr, kind = interp_kind)[1:]
    
    # Calculate s_rel:
    s_rel = (s_target @ (S_C*dl).T) / (s_detector @ (S_C*dl).T) * s_detector

    # Calculate fprime1:
    dl = dl*np.ones_like(wlr) # ensure array like for matrix multiplication
    f1p = (np.abs(s_rel - s_target) @ dl) / (s_target @ dl)
    
    if out == 'f1p':
        return f1p
    elif out == 's_rel':
        return s_rel 
    elif out == 'f1p,s_rel':
        return f1p, s_rel
    elif out == 'f1p,s_rel,s_target,wlr,dl':
        return f1p,s_rel,s_target,wlr,dl
    else:
        return eval(out)
    
def get_spectral_mismatch_correction_factors(S_Z, s_detector, S_C = 'A', 
                                          cieobs = '1931_2', s_target_index = 2,
                                          wlr = [380,780, 1], interp_kind = 'linear', 
                                          out = 'F'):
    """
    Determine the spectral mismatch factors.
    
    Args:
        :S_Z:
            | ndarray with spectral power distribution of measured light source (first row = wavelengths).
        :s_detector:
            | ndarray with detector spectral responsivity (first row = wavelengths)
        :S_C:
            | 'A', optional
            | Standard 'calibration' illuminant.
            | string specifying the illuminant to use from the luxpy._CIE_ILLUMINANTS dict 
            | or ndarray with standard illuminant spectral data.
        :cieobs:
            | '1931_2', optional
            | string with CIE standard observer color matching functions to use (from luxpy._CMF)
            | or ndarray with CMFs (s_target_index > 0) 
            | or target spectral responsivity (s_target_index == 0)
            | (first row contains the wavelengths).
        :s_target_index:
            | 2, optional
            | if > 0: index into CMF set (1->'xbar', 2->'ybar'='Vlambda', 3->'zbar')
            | if == 0: cieobs is expected to contain an ndarray with the target spectral responsivity. 
        :wlr:
            | [380,780,1], optional
            | Wavelength range (ndarray or [start, stop, spacing]).
            | If None: use the wavelength range of S_Z.
        :interp_kind:
            | 'linear', optional
            | Interpolation type to use when interpolating function to specified wavelength range.
        :out:
            | 'F', optional
            | Specify requested output of function, 
            |    e.g. 'F,f1p' also outputs the f1prime spectral mismatch index. 
    Returns:
        :F:
            | ndarray with correction factors for each of the mesured spectra (rows)
            | and spectral responsivities in s_detector (columns).
    """
    if wlr is None:
        wlr = S_Z[0] # use wavelengths of measured spectra!
    
    # get f1p & s_rel
    f1p, s_rel, s_target, wlr, dl = f1prime(s_detector, S_C = S_C, 
                                            cieobs = cieobs, s_target_index = s_target_index,
                                            wlr = wlr, interp_kind = interp_kind, 
                                            out = 'f1p,s_rel,s_target,wlr,dl')
    
    # Interpolate measured spectrum to desired wavelength range:
    S_Z = cie_interp(S_Z, wlr, kind = interp_kind)[1:]
    
    # Calculate spectral mismatch correction factors:
    F = (S_Z @ (s_target*dl).T) / (S_Z @ (s_rel*dl).T)
    
    if out == 'F':
        return F
    elif out == 'F,f1p':
        return F,f1p
    else:
        return eval(out)

    
if __name__ == '__main__':
    
    s_detector = _CMF['1931_2']['bar'][[0,2,3]].copy()
    f1p = f1prime(s_detector)
    
    s_detector = _CMF['1964_10']['bar'][[0,1,2,3]].copy()
    f1p = f1prime(s_detector)
    
    S_Z = np.vstack((_CIE_ILLUMINANTS['D65'],_CIE_ILLUMINANTS['C'][1:]))
    F = get_spectral_mismatch_correction_factors(S_Z,s_detector)
    print(F.shape)