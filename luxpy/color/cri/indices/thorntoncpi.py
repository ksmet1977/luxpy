# -*- coding: utf-8 -*-
"""
Thornton's Color Preference Index
=================================

 :spd_to_thornton_cpi(): Calculate Thornton's Color Preference Index (CPI).

Created on Fri Oct  2 20:33:31 2020

@author: ksmet1977 at gmail.com
"""
import numpy as np 

from luxpy import _RFL
from luxpy.color.cri.utils.helpers import spd_to_jab_t_r

_RFL_CPI = _RFL['cri']['cie-13.3-1995']['8']['1nm']

#add white so we have access to the white point chromaticity for a Judd-type translational CAT
_RFL_CPIw = np.vstack((_RFL_CPI,np.ones((1,_RFL_CPI.shape[1])))) 

__all__ = ['spd_to_thornton_cpi']

def spd_to_thornton_cpi(spd, interp_settings = None):
    """
    Calculate Thornton's Color Preference Index (CPI).
    
    Args:
        :spd:
            | nd array with spectral power distribution(s) of the test light source(s).
            
    Returns:
        :cpi:
            | ndarray with CPI values.
        
    Reference:
        1. `Thornton, W. A. (1974). A Validation of the Color-Preference Index.
        Journal of the Illuminating Engineering Society, 4(1), 48–52. 
        <https://doi.org/10.1080/00994480.1974.10732288>`_
    """
    
    # sample 1976 u'v' coordinates for test and reference 
    # using CIE Ra calculation engine 
    # (only cspace, sampleset (with added white), and catf have changed; 
    # catf = None so no CAT is applied as Thornton CPI, 
    # like Judd's Flattery index, uses a Judd-type translational CAT)
    Yuv_t, Yuv_r = spd_to_jab_t_r(spd, cri_type='ciera',
                                  cspace = {'type':'Yuv','xyzw':None},
                                  catf = None,
                                  sampleset = _RFL_CPIw,
                                  interp_settings = interp_settings)
    
    # Convert to 1960 UCS:
    Yuv_t[...,2]*=(2/3)
    Yuv_r[...,2]*=(2/3)
    
    # Perform Judd-type translational CAT with white point stored in last row:
    Yuv_t[...,1:] -= Yuv_t[...,1:][-1]
    Yuv_r[...,1:] -= Yuv_r[...,1:][-1]

    # Remove last row (white point):
    Yuv_t = Yuv_t[:-1,...]
    Yuv_r = Yuv_r[:-1,...]

    # Define preferred chromaticity shifts for 8 CIE CRI samples:
    # (*5, because Thorton uses full preferred shifts unlike Judd's Flattery Index)
    uv_shifts = np.array([[0.0020,0.0008],
                          [0.0000,0.0000],
                          [-0.0020,0.0008],
                          [-0.0020,0.0010],
                          [-0.0020,-0.0004],
                          [-0.0012,-0.0020],
                          [0.0008,-0.0020],
                          [0.0020,-0.0010]])*5

    # Calculate chromaticity difference between test and shifted ref coordinates:
    DE = 800*(((Yuv_t[...,1:] - (Yuv_r[...,1:] + uv_shifts[:,None,:]))**2).sum(axis=-1))**0.5
    
    # Calculate CPI:
    CPI = 156 - 7.317*DE.mean(axis=0) # in Thornton 1974 we find 7.18, but then CPI(D65)!=100    
    return CPI

if __name__ == '__main__':
    import luxpy as lx
    F4 = lx.cie_interp(lx._CIE_F4, wl_new = lx.getwlr([360,830,1]), datatype = 'spd')
    D65 = lx.cie_interp(lx._CIE_D65, wl_new = lx.getwlr([360,830,1]), datatype = 'spd')
    spds = np.vstack((F4, D65[1:,:]))
    
    cpi1 = spd_to_thornton_cpi(F4)
    cpi2 = spd_to_thornton_cpi(spds)