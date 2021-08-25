# -*- coding: utf-8 -*-
"""
Module for detector spectral responsivity spectral mismatch calculations
========================================================================

 :f1prime(): Determine the f1prime spectral mismatch index.

Reference:
    1. Krüger, U. et al. GENERAL V(λ) MISMATCH - INDEX HISTORY, CURRENT STATE, NEW IDEAS
    
Created on Wed Aug 25 13:02:00 2021

@author: ksmet1977 [at] gmail.com
"""
from luxpy.spectrum import _CIE_ILLUMINANTS, _CMF, cie_interp, getwlr, getwld
import numpy as np

__all__ = ['f1prime']

def f1prime(s_detector, S_illuminant = 'A', cieobs = '1931_2', s_target_index = 2, wlr = [380,780, 1], interp_kind = 'linear'):
    """
    Determine the f1prime spectral mismatch index.
    
    Args:
        :s_detector:
            | ndarray with detector spectral responsivity (first row = wavelengths)
        :S_illuminant:
            | 'A', optional
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
            | Wavelength range (ndarray or [start, stop, spacing]) .
        :interp_kind:
            | 'linear', optional
            | Interpolation type to use when interpolating function to specified wavelength range.
            
    Returns:
        :f1p:
            | ndarray (vector) with f1prime values for each of the spectral responsivities in s_detector.
    """
    
    # Get illuminant:
    if isinstance(S_illuminant,str): S_illuminant = _CIE_ILLUMINANTS[S_illuminant].copy()
    
    # Get target function from cieobs: 
    if s_target_index == 0: s_target_index = 1
    s_target = _CMF[cieobs]['bar'][[0,s_target_index]].copy() if isinstance(cieobs, str) else cieobs[[0,s_target_index]].copy()
    
    # Interpolate to desired wavelength range:
    wlr = getwlr(wlr) # get wavelength range from array or '3-vector'
    dl = getwld(wlr) # wavelength differences (unequal wavelength spacings are taken into account)
    s_detector = cie_interp(s_detector, wlr, kind = interp_kind)[1:]
    s_target = cie_interp(s_target, wlr, kind = interp_kind)[1:]
    S_illuminant = cie_interp(S_illuminant, wlr, kind = interp_kind)[1:]
    
    # Calculate s_rel:
    s_rel = (s_target @ (S_illuminant*dl).T) / (s_detector @ (S_illuminant*dl).T) * s_detector

    # Calculate fprime1:
    dl = dl*np.ones_like(wlr) # ensure array like for matrix multiplication
    f1p = (np.abs(s_rel - s_target) @ dl) / (s_target @ dl)
    
    return f1p
    
if __name__ == '__main__':
    
    s_detector = _CMF['1931_2']['bar'][[0,2,3]].copy()
    f1p = f1prime(s_detector)
    
