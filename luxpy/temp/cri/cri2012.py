# -*- coding: utf-8 -*-
"""
###############################################################################
# Module with CRI2012 color fidelity index.
###############################################################################
    
Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
    CRI2012: A proposal for updating the CIE colour rendering index. 
    Lighting Research and Technology, 45, 689–709. 
    Retrieved from http://lrt.sagepub.com/content/45/6/689
    
###############################################################################
# spd_to_cri2012(): the 'cri2012' color rendition (fidelity) metric
                    with the spectally uniform HL17 mathematical sampleset. 
                    
# spd_to_cri2012_hl17(): the 'cri2012' color rendition (fidelity) metric
                    with the spectally uniform HL17 mathematical sampleset.  
                    
# spd_to_cri2012_hl1000(): the 'cri2012' color rendition (fidelity) metric
                    with the spectally uniform HL1000 sampleset. 
                    
# spd_to_cri2012(): the 'cri2012' color rendition (fidelity) metric
                    with the Real-210 sampleset. 
                    (normally for special color rendering indices)
                    
#------------------------------------------------------------------------------                    
Created on Sun Apr 15 11:35:10 2018

@author: kevin.smet
"""

from .helpers import spd_to_cri
__all__ =['spd_to_cri2012', 'spd_to_cri2012_hl17', 'spd_to_cri2012_hl1000', 'spd_to_cri2012_real210']

#------------------------------------------------------------------------------
def spd_to_cri2012(SPD, out = 'Rf', wl = None):
    """
    Wrapper function for the 'cri2012' color rendition (fidelity) metric
    with the spectally uniform HL17 mathematical sampleset.

    Args:
        :SPD: ndarray with spectral data (can be multiple SPDs, 
              first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or ndarray with CRI2012 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
            
    References:
        ..[1] Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
            CRI2012: A proposal for updating the CIE colour rendering index. 
            Lighting Research and Technology, 45, 689–709. 
            Retrieved from http://lrt.sagepub.com/content/45/6/689
    """
    return spd_to_cri(SPD, cri_type = 'cri2012', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cri2012_hl17(SPD, out = 'Rf', wl = None):
    """
    Wrapper function for the 'cri2012' color rendition (fidelity) metric
    with the spectally uniform HL17 mathematical sampleset.
    
    Args:
        :SPD: ndarray with spectral data (can be multiple SPDs, 
              first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or ndarray with CRI2012 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
            CRI2012: A proposal for updating the CIE colour rendering index. 
            Lighting Research and Technology, 45, 689–709. 
            Retrieved from http://lrt.sagepub.com/content/45/6/689
    """
    return spd_to_cri(SPD, cri_type = 'cri2012-hl17', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cri2012_hl1000(SPD, out = 'Rf', wl = None):
    """
    Wrapper function for the 'cri2012' color rendition (fidelity) metric
    with the spectally uniform Hybrid HL1000 sampleset.
    
    Args:
        :SPD: ndarray with spectral data (can be multiple SPDs, 
              first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or ndarray with CRI2012 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
            CRI2012: A proposal for updating the CIE colour rendering index. 
            Lighting Research and Technology, 45, 689–709. 
            Retrieved from http://lrt.sagepub.com/content/45/6/689
    """
    return spd_to_cri(SPD, cri_type = 'cri2012-hl1000', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cri2012_real210(SPD, out = 'Rf', wl = None):
    """
    Wrapper function the 'cri2012' color rendition (fidelity) metric 
    with the Real-210 sampleset (normally for special color rendering indices).
    
    Args:
        :SPD: ndarray with spectral data (can be multiple SPDs, 
              first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or ndarray with CRI2012 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
            CRI2012: A proposal for updating the CIE colour rendering index. 
            Lighting Research and Technology, 45, 689–709. 
            Retrieved from http://lrt.sagepub.com/content/45/6/689
    
    """
    return spd_to_cri(SPD, cri_type = 'cri2012-real210', out = out, wl = wl)

