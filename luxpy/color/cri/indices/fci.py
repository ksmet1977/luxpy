# -*- coding: utf-8 -*-
"""
Feeling of Contrast Index
=========================

 :spd_to_fci(): Calculate Feeling of Contrast Index (FCI).


Created on Fri Oct  2 16:37:13 2020

@author: ksmet1977 at gmail.com
"""
import numpy as np 

from luxpy import cat, cam, spd_to_xyz, _RFL, _CIE_D65, xyz_to_lab 
from luxpy.utils import asplit

# Get RFLs and calculate fixed reference D65 XYZs for speed:
_RFL_FCI = _RFL['cri']['fci']
_XYZ_D65_REF, _XYZW_D65_REF = spd_to_xyz(_CIE_D65, cieobs = '1931_2', relative = True, rfl = _RFL_FCI, out = 2)

__all__ = ['spd_to_fci']

    
def _polyarea3D(xyz):
    
    x,y,z = asplit(xyz)
    
    RY = np.sqrt((x[0] - x[1])**2 + (y[0]-y[1])**2 + (z[0]-z[1])**2)
    YG = np.sqrt((x[1] - x[2])**2 + (y[1]-y[2])**2 + (z[1]-z[2])**2)
    GR = np.sqrt((x[2] - x[0])**2 + (y[2]-y[0])**2 + (z[2]-z[0])**2)
    RB = np.sqrt((x[0] - x[3])**2 + (y[0]-y[3])**2 + (z[0]-z[3])**2)
    BG = np.sqrt((x[2] - x[3])**2 + (y[2]-y[3])**2 + (z[2]-z[3])**2)
    S1 = (RY+YG+GR)/2
    S2 = (RB+BG+GR)/2
    GA1 = np.sqrt(S1*(S1-RY)*(S1-YG)*(S1-GR))
    GA2 = np.sqrt(S2*(S2-RB)*(S2-BG)*(S2-GR))
    GA = GA1 + GA2
    return GA
    
    
def spd_to_fci(spd, use_cielab = True):
    """
    Calculate Feeling of Contrast Index (FCI).
    
    Args:
        :spd:
            | ndarray with spectral power distribution(s) of the test light source(s).
        :use_cielab:
            |  True, optional
            | True: use original formulation of FCI, which adopts a CIECAT94 
            | chromatic adaptation transform followed by a conversion to 
            | CIELAB coordinates before calculating the gamuts.
            | False: use CIECAM02 coordinates and embedded CAT02 transform.
            
    Returns:
        :fci:
            | ndarray with FCI values.
            
    References:
        1. `Hashimoto, K., Yano, T., Shimizu, M., & Nayatani, Y. (2007). 
        New method for specifying color-rendering properties of light sources 
        based on feeling of contrast. 
        Color Research and Application, 32(5), 361–371. 
        <http://dx.doi.org/10.1002/col.20338>`_
    """
    
    # get xyz:
    xyz, xyzw = spd_to_xyz(spd, cieobs = '1931_2', 
                           relative = True, 
                           rfl = _RFL_FCI, out = 2)

    # set condition parameters:
    D = 1
    Yb = 20
    La = Yb*1000/np.pi/100
    
    if use_cielab:
        # apply ciecat94 chromatic adaptation transform:
        xyzc = cat.apply_ciecat94(xyz, xyzw = xyzw, 
                                  E = 1000, Yb = 20, D = D,
                                  cat94_old = True) # there is apparently an updated version with an alpha incomplete adaptation factor and noise = 0.1; However, FCI doesn't use that version. 
        
        # convert to cielab:
        lab = xyz_to_lab(xyzc, xyzw = _XYZW_D65_REF)
        labd65 = np.repeat(xyz_to_lab(_XYZ_D65_REF, xyzw = _XYZW_D65_REF),lab.shape[1],axis=1)
    else:
        f = lambda xyz, xyzw: cam.xyz_to_jabC_ciecam02(xyz, xyzw = xyzw, La = 1000*20/np.pi/100, Yb = 20, surround = 'avg')
        lab = f(xyz, xyzw)
        labd65 = np.repeat(f(_XYZ_D65_REF, _XYZW_D65_REF),lab.shape[1],axis=1)

    fci = 100*(_polyarea3D(lab) / _polyarea3D(labd65))**1.5
    
    return fci

if __name__ == '__main__':
    import luxpy as lx
    F6 = lx.cie_interp(lx._CIE_ILLUMINANTS['F6'], wl_new = lx.getwlr([360,830,1]), kind = 'spd')
    F4 = lx.cie_interp(lx._CIE_F4, wl_new = lx.getwlr([360,830,1]), kind = 'spd')
    D65 = lx.cie_interp(lx._CIE_D65, wl_new = lx.getwlr([360,830,1]), kind = 'spd')
    spds = np.vstack((F6,F4[1:,:], D65[1:,:]))
    
    fci1a = spd_to_fci(F6, True)
    print(fci1a)
    fci1b = spd_to_fci(F6,False)
    print(fci1b)
    fci2 = spd_to_fci(spds)
    print(fci2)
