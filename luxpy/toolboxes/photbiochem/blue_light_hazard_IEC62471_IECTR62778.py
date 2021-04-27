# -*- coding: utf-8 -*-
"""
Module for Blue light hazard calculations
=========================================

 :_BLH: Blue Light Hazard function
 
 :spd_to_blh_eff(): Calculate Blue Light Hazard efficacy (K) or efficiency (eta) of radiation.


References:
        1. IEC 62471:2006, 2006, Photobiological safety of lamps and lamp systems.
        2. IEC TR 62778, 2014, Application of IEC 62471 for the assessment of blue light hazard to light sources and luminaires.


Created on Tue Apr 27 12:39:51 2021

@author: ksmet1977 [at] gmail.com
"""
import numpy as np
from luxpy import getwlr, getwld, cie_interp, _CMF, _CIEOBS, vlbar

__all__ = ['_BLH', 'spd_to_blh_eff']

def _get_BLH():
    
    BLH1 = np.array([[380,385,390,395,400,405,410,415,420,425,430,435,440,445,450,455,460,465,470,475,480,485,490,495,500],
                     [0.01,0.013,0.025,0.05,0.1,0.2,0.4,0.8,0.9,0.95,0.98,1,1,0.97,0.94,0.9,0.8,0.7,0.62,0.55,0.45,0.4,0.22,0.16,10**((450-500)/50)]])
    BLH1 = cie_interp(BLH1, wl_new = getwlr([360, 500, 1]), kind = 'linear')
    BLH2 = getwlr([501,600,1])
    BLH2 = np.vstack((BLH2, 10**((450-BLH2)/50)))
    BLH3 = getwlr([601,700,1])
    BLH3 = np.vstack((BLH3, [0.001]*BLH3.shape[0]))
    BLH4 = getwlr([701,830,1])
    BLH4 = np.vstack((BLH4, [0.0]*BLH4.shape[0]))
    return np.hstack((BLH1,BLH2,BLH3,BLH4))

_BLH = _get_BLH()


def spd_to_blh_eff(spd, efficacy = True, cieobs = _CIEOBS, scr = 'dict', K = None):
    """
    Calculate Blue Light Hazard efficacy (K) or efficiency (eta) of radiation.
   
    Args:
        :S: 
            | ndarray with spectral data
        :cieobs: 
            | str, optional
            | Sets the type of Vlambda function to obtain.
        :scr: 
            | 'dict' or array, optional
            | - 'dict': get from ybar from _CMF
            | - 'array': ndarray in :cieobs:
            | Determines whether to load cmfs from file (./data/cmfs/) 
            | or from dict defined in .cmf.py
            | Vlambda is obtained by collecting Ybar.
        :K: 
            | None, optional
            |   e.g.  K  = 683 lm/W for '1931_2' (relative == False) 
            |   or K = 100/sum(spd*dl)        (relative == True)
            
    Returns:
        :eff:
            | ndarray with blue light hazard efficacy or efficiency of radiation values.
            
    References:
        1. IEC 62471:2006, 2006, Photobiological safety of lamps and lamp systems.
        2. IEC TR 62778, 2014, Application of IEC 62471 for the assessment of blue light hazard to light sources and luminaires.
    """
    blh = cie_interp(_BLH, wl_new = spd[0], kind = 'linear')    
    dl = getwld(spd[0])
    if efficacy:
        Vl = vlbar(cieobs = cieobs, scr = scr, wl_new = spd[0])
        if K is None:
            if scr == 'dict':
                K = _CMF[cieobs]['K']
            else:
                K = 683
        return ((spd[1:] @ (blh[1:]*dl).T).T / (K*(spd[1:] @ (Vl[1:]*dl).T).T))[0]
    else:
        return ((spd[1:] @ (blh[1:]*dl).T).T / (spd[1:]*dl).sum(axis = 1))[0]
        
if __name__ == '__main__':
    import luxpy as lx
    spd = np.vstack((lx._CIE_D65,lx._CIE_A[1:]))     

    K = spd_to_blh_eff(spd, efficacy = True, cieobs = '1931_2',
                       scr = 'dict', K = None)
    eta = spd_to_blh_eff(spd, efficacy = False, cieobs = '1931_2',
                         scr = 'dict', K = None)
    print('K: ', K)
    print('eta: ', eta)


