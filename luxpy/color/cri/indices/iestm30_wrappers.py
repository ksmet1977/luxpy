# -*- coding: utf-8 -*-
"""
###############################################################################
# Module with IES (TM30) color fidelity and gamut area indices.
###############################################################################
    
..[1] IES. (2015). 
        IES-TM-30-15: Method for Evaluating Light Source Color Rendition. 
        New York, NY: The Illuminating Engineering Society of North America.
                
..[2] David, A., Fini, P. T., Houser, K. W., Ohno, Y., Royer, M. P., 
      Smet, K. A. G., … Whitehead, L. (2015). 
        Development of the IES method for evaluating the color rendition 
        of light sources. 
        Optics Express, 23(12), 15888–15906. 
        https://doi.org/10.1364/OE.23.015888
   
###############################################################################
 
# spd_to_iesrf(): the 'iesrf' color fidelity index (latest version)    

# spd_to_iesrg(): the 'iesrg' color gamut area index (latest version)    
    
# spd_to_iesrf_tm30(): the 'iesrf' color fidelity index (latest version)    

# spd_to_iesrg_tm30(): the 'iesrg' color gamut area index (latest version)    

    
# spd_to_iesrf_tm30_15(): the 'iesrf' color fidelity index (TM30-15) 
        
# spd_to_iesrg_tm30_15(): the 'iesrg' color gamut area index (TM30-15)    
        
        
# spd_to_iesrf_tm30_18(): the 'iesrf' color fidelity index (TM30-18)    

# spd_to_iesrg_tm30_18(): the 'iesrg' color gamut area index (TM30-18)    
               
               
#------------------------------------------------------------------------------  
Created on Sun Apr 15 12:14:23 2018

@author: kevin.smet
"""

from ..utils.helpers import spd_to_cri, spd_to_rg

__all__ = ['spd_to_iesrf','spd_to_iesrg',
           'spd_to_iesrf_tm30','spd_to_iesrg_tm30',
           'spd_to_iesrf_tm30_15','spd_to_iesrg_tm30_15',
           'spd_to_iesrf_tm30_18','spd_to_iesrg_tm30_18']

#------------------------------------------------------------------------------
def spd_to_iesrf_tm30_15(SPD, out = 'Rf', wl = None, cri_type = 'iesrf_tm30_15'):
    """
    Wrapper function for the 'iesrf' color fidelity index (IES TM30-15). 
    
    Args:
        :SPD: ndarray with spectral data (can be multiple SPDs, 
             first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or ndarray with IES TM30_15 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] IES. (2015). 
                IES-TM30-15: Method for Evaluating Light Source Color Rendition 
                New York, NY: The Illuminating Eng. Soc. of North America.
        ..[2] David, A., Fini, P. T., Houser, K. W., Ohno, Y., Royer, M. P., 
              Smet, K. A. G., … Whitehead, L. (2015). 
                Development of the IES method for evaluating the color 
                rendition of light sources. 
                Optics Express, 23(12), 15888–15906. 
                https://doi.org/10.1364/OE.23.015888
    
    """
    return spd_to_cri(SPD, cri_type = cri_type, out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_iesrg_tm30_15(SPD, out = 'Rg', wl = None, cri_type ='iesrf-tm30-15'):
    """
    Wrapper function for the 'spd_to_rg' color gamut area index (IES TM30-15). 
    
    Args:
        :SPD: ndarray with spectral data (can be multiple SPDs, first axis are 
              the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            None: default to no interpolation
        :out:  'Rg' or str, optional
            Specifies requested output (e.g. 'Rg,Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or ndarray with IES TM30_15 Rg for :out: 'Rg'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] IES. (2015). 
                IES-TM30-15: Method for Evaluating Light Source Color Rendition
                New York, NY: The Illuminating Eng. Soc. of North America.
        ..[2] David, A., Fini, P. T., Houser, K. W., Ohno, Y., Royer, M. P., 
              Smet, K. A. G., … Whitehead, L. (2015). 
                Development of the IES method for evaluating the color 
                rendition of light sources. 
                Optics Express, 23(12), 15888–15906. 
                https://doi.org/10.1364/OE.23.015888
    
    """
    return spd_to_rg(SPD, cri_type = cri_type, out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_iesrf_tm30_18(SPD, out = 'Rf', wl = None, cri_type = 'iesrf-tm30-18'):
    """
    Wrapper function for the 'iesrf' color fidelity index (IES TM30-18). 
    
    Args:
        :SPD: ndarray with spectral data (can be multiple SPDs, 
              first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or ndarray with IES TM30_18 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] IES. (2015). 
                IES-TM30-15: Method for Evaluating Light Source Color Rendition
                New York, NY: The Illuminating Eng. Soc. of North America.
        ..[2] David, A., Fini, P. T., Houser, K. W., Ohno, Y., Royer, M. P., 
              Smet, K. A. G., … Whitehead, L. (2015). 
                Development of the IES method for evaluating the color 
                rendition of light sources. 
                Optics Express, 23(12), 15888–15906. 
                https://doi.org/10.1364/OE.23.015888
    
    """
    return spd_to_cri(SPD, cri_type = cri_type, out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_iesrg_tm30_18(SPD, out = 'Rg', wl = None, cri_type ='iesrf-tm30-18'):
    """
    Wrapper function for the 'spd_to_rg' color gamut area index (IES TM30-18). 
    
    Args:
        :SPD: ndarray with spectral data (can be multiple SPDs, first axis are 
              the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            None: default to no interpolation
        :out:  'Rg' or str, optional
            Specifies requested output (e.g. 'Rg,Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or ndarray with IES TM30_18 Rg for :out: 'Rg'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] IES. (2015). 
                IES-TM30-15: Method for Evaluating Light Source Color Rendition 
                New York, NY: The Illuminating Eng. Soc. of North America.
        ..[2] David, A., Fini, P. T., Houser, K. W., Ohno, Y., Royer, M. P., 
              Smet, K. A. G., … Whitehead, L. (2015). 
                Development of the IES method for evaluating the color 
                rendition of light sources. 
                Optics Express, 23(12), 15888–15906. 
                https://doi.org/10.1364/OE.23.015888
    
    """
    return spd_to_rg(SPD, cri_type = cri_type, out = out, wl = wl)

# additional (latest version) callers:
spd_to_iesrf_tm30 = spd_to_iesrf_tm30_18
spd_to_iesrg_tm30 = spd_to_iesrg_tm30_18
spd_to_iesrf = spd_to_iesrf_tm30_18
spd_to_iesrg = spd_to_iesrg_tm30_18