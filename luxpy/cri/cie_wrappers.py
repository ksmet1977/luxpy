# -*- coding: utf-8 -*-
"""
###############################################################################
# Module with CIE 13.3-1995 Ra and CIE 224-2017 Rf color fidelity indices.
###############################################################################
    
CIE13.3-1995. Method of Measuring and Specifying 
                Colour Rendering Properties of Light Sources 
                (Vol. CIE13.3-19). Vienna, Austria: CIE. (1995).
                
CIE224:2017. CIE 2017 Colour Fidelity Index for accurate scientific use. 
                Vienna, Austria: CIE. (2017).
   
###############################################################################
 
# spd_to_ciera(): the 'ciera' color rendition (fidelity) metric (CIE 13.3-1995)    

# spd_to_ciera_133_1995 = spd_to_ciera   

# spd_to_cierf(): the 'cierf' color rendition (fidelity) metric (CIE224-2017). 

# spd_to_cierf_224_2017 = spd_to_cierf
               
               
#------------------------------------------------------------------------------                     
Created on Sun Apr 15 11:55:09 2018

@author: kevin.smet
"""

from .helpers import spd_to_cri

__all__ = ['spd_to_ciera', 'spd_to_cierf',
           'spd_to_ciera_133_1995','spd_to_cierf_224_2017']

#------------------------------------------------------------------------------
def spd_to_ciera(SPD, out = 'Rf', wl = None):
    """
    Wrapper function the 'ciera' color rendition (fidelity) metric (CIE 13.3-1995). 
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or numpy.ndarray with CIE13.3 Ra for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] CIE13.3-1995. (1995). Method of Measuring and Specifying Colour Rendering Properties of Light Sources (Vol. CIE13.3-19). Vienna, Austria: CIE.

    """
    return spd_to_cri(SPD, cri_type = 'ciera', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cierf(SPD, out = 'Rf', wl = None):
    """
    Wrapper function the 'cierf' color rendition (fidelity) metric (CIE224-2017). 
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or numpy.ndarray with CIE224-2017 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    References:
        ..[1] CIE224:2017. (2017). CIE 2017 Colour Fidelity Index for accurate scientific use. Vienna, Austria.
    
    """
    return spd_to_cri(SPD, cri_type = 'cierf', out = out, wl = wl)


# Additional callers:
spd_to_ciera_133_1995 = spd_to_ciera
spd_to_cierf_224_2017 = spd_to_cierf
