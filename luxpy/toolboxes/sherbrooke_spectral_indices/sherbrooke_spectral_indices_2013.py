# -*- coding: utf-8 -*-
"""
Module for the calculation of the Melatonin Suppression Index (MSI), 
the Induced Photosynthesis Index (IPI) and the Star Light Index (SLI)
---------------------------------------------------------------------

 :spd_to_msi(): calculate Melatonin Suppression Index from spectrum.
 
 :spd_to_ipi(): calculate Induced Photosynthesis Index from spectrum.
 
 :spd_to_sli(): calculate Star Light Index from spectrum.

References: 
    1. Aubé M, Roby J, Kocifaj M (2013) 
    Evaluating Potential Spectral Impacts of Various Artificial Lights on Melatonin Suppression, Photosynthesis, and Star Visibility. 
    PLoS ONE 8(7): e67798
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0067798

Created on Fri Jun 11 13:46:33 2021

@author: ksmet1977 [at] gmail dot com
"""
import os
import numpy as np

from luxpy import _CIE_D65, cie_interp, getwlr
from luxpy.utils import _PKG_PATH, getdata


_WLR_RANGE = [380, 730] # for 2013 indices (see ref. 1)
_VSCOT_VM_MSAS_PAS = getdata(os.path.join(_PKG_PATH,'toolboxes','sherbrooke_spectral_indices','data','msas_pas_functions_2013.csv'),header = 'infer').T

__all__ = ['spd_to_msi','spd_to_ipi','spd_to_sli','_VSCOT_VM_MSAS_PAS','_WLR_RANGE']

def _limit_wlr_range(spd, wlr_range = _WLR_RANGE, wlr_interval = None):
    if wlr_interval is None:
        return spd[:,(spd[0]>=wlr_range[0]) & (spd[0]<=wlr_range[1])]
    else:
        return cie_interp(spd, getwlr([wlr_range[0],wlr_range[1],wlr_interval]), kind = 'linear')

_VSCOT_VM_MSAS_PAS = _limit_wlr_range(_VSCOT_VM_MSAS_PAS, wlr_range = _WLR_RANGE)

def _spd_to_x(spd, idx, force_5nm_interval = True):
    """ 
    Calculate x Index from spectrum.
    
    Args:
        :spd:
            | ndarray with spectral data (first row are wavelengths)
        :idx:
            | index in _VSCOT_VM_MSAS_PAS
        :force_5nm_interval:
            | True, optional
            | If True: interpolate spd to 5nm wavelengths intervals, else: keep as in spd.
    
    Returns:
        :index:
            | ndarray with requested index values for each input spectrum.
    """
    # limit spectrum to desired wavelength range:
    wlr_interval = 5 if force_5nm_interval else None
    spd = _limit_wlr_range(spd, wlr_interval = wlr_interval)
    
    # get VM & requested action spectrum interpolated to spd wavelengths:
    vm_acs = cie_interp(_VSCOT_VM_MSAS_PAS[[0,2,idx]], spd[0], kind = 'linear')
    vm, acs = vm_acs[1], vm_acs[2]
    
    # get D65 ref spectrum interpolated to spd wavelengths:
    D65 = cie_interp(_CIE_D65, spd[0], kind = 'linear')
    
    # normalize spectrum and D65 using VM function:
    spd[1:] = spd[1:]/(spd[1:]*vm).sum(axis = 1, keepdims = True)
    D65[1:] = D65[1:]/(D65[1:]*vm).sum(axis = 1, keepdims = True)
    
    # x = ratio of integrated actionspectrum-weighted spd and D65:
    return (spd[1:]*acs).sum(axis = 1, keepdims = True) / (D65[1:]*acs).sum(axis = 1, keepdims = True)



def spd_to_msi(spd, force_5nm_interval = True):
    """ 
    Calculate Melatonin Suppression Index from spectrum.
    
    Args:
        :spd:
            | ndarray with spectral data (first row are wavelengths)
        :force_5nm_interval:
            | True, optional
            | If True: interpolate spd to 5nm wavelengths intervals, else: keep as in spd.
    
    Returns:
        :msi:
            | ndarray with Melatonin Suppression Index values for each input spectrum.
    """
    return _spd_to_x(spd, 3, force_5nm_interval = force_5nm_interval)


def spd_to_ipi(spd, force_5nm_interval = True):
    """ 
    Calculate Induced Photosynthesis Index from spectrum.
    
    Args:
        :spd:
            | ndarray with spectral data (first row are wavelengths)
        :force_5nm_interval:
            | True, optional
            | If True: interpolate spd to 5nm wavelengths intervals, else: keep as in spd.
    
    Returns:
        :msi:
            | ndarray with Induced Photosynthesis Index values for each input spectrum.
    """
    return _spd_to_x(spd, 4, force_5nm_interval = force_5nm_interval)

   
def spd_to_sli(spd, force_5nm_interval = True):
    """ 
    Calculate Star Light Index from spectrum.
    
    Args:
        :spd:
            | ndarray with spectral data (first row are wavelengths)
        :force_5nm_interval:
            | True, optional
            | If True: interpolate spd to 5nm wavelengths intervals, else: keep as in spd.
    
    Returns:
        :msi:
            | ndarray with Star Light Index values for each input spectrum.
    """
    return _spd_to_x(spd, 1, force_5nm_interval = force_5nm_interval)
    
if __name__ == '__main__':
    
    # Read test spectrum:
    cree = getdata(os.path.join(_PKG_PATH,'toolboxes','sherbrooke_spectral_indices','data','LED_CREE_BR30.csv')).T
    
    msi = spd_to_msi(cree)
    print('MSI:', msi)
    
    ipi = spd_to_ipi(cree)
    print('IPI:', ipi)
    
    sli = spd_to_sli(cree)
    print('SLI:', sli)

    
    