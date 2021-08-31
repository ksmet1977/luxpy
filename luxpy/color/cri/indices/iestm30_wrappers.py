# -*- coding: utf-8 -*-
"""
Module with IES (TM30) color fidelity and gamut area indices.
=============================================================
   
 :spd_to_iesrf(): the 'iesrf' color fidelity index (latest version)    

 :spd_to_iesrg(): the 'iesrg' color gamut area index (latest version)    
    
 :spd_to_iesrf_tm30(): the 'iesrf' color fidelity index (latest version)    

 :spd_to_iesrg_tm30(): the 'iesrg' color gamut area index (latest version)    

    
 :spd_to_iesrf_tm30_15(): the 'iesrf' color fidelity index (TM30-15) 
        
 :spd_to_iesrg_tm30_15(): the 'iesrg' color gamut area index (TM30-15)    
        
        
 :spd_to_iesrf_tm30_18(): the 'iesrf' color fidelity index (TM30-18)    

 :spd_to_iesrg_tm30_18(): the 'iesrg' color gamut area index (TM30-18) 

 :spd_to_iesrf_tm30_20(): the 'iesrf' color fidelity index (TM30-20 = TM30-18 !!)    

 :spd_to_iesrg_tm30_20(): the 'iesrg' color gamut area index (TM30-20 = TM30-18 !!)    
               
References:
    
    1. `IES TM30, Method for Evaluating Light Source Color Rendition. 
    New York, NY: The Illuminating Engineering Society of North America.
    <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
    
    2. `A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, 
    “Development of the IES method for evaluating the color rendition of light sources,” 
    Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015. 
    <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888>`_
    
    3. `K. A. G. Smet, A. David, and L. Whitehead, 
    “Why color space uniformity and sample set spectral uniformity are essential for color rendering measures,” 
    LEUKOS, vol. 12, no. 1–2, pp. 39–50, 2016 
    <http://www.tandfonline.com/doi/abs/10.1080/15502724.2015.1091356>`_               

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from ..utils.helpers import spd_to_cri, spd_to_rg

__all__ = ['spd_to_iesrf','spd_to_iesrg',
           'spd_to_iesrf_tm30','spd_to_iesrg_tm30',
           'spd_to_iesrf_tm30_15','spd_to_iesrg_tm30_15',
           'spd_to_iesrf_tm30_18','spd_to_iesrg_tm30_18',
           'spd_to_iesrf_tm30_20','spd_to_iesrg_tm30_20']

#------------------------------------------------------------------------------
def spd_to_iesrf_tm30_15(SPD, out = 'Rf', wl = None, cri_type = 'iesrf-tm30-15'):
    """
    Wrapper function for the 'iesrf' color fidelity index (IES TM30-15). 
    
    Args:
        :SPD: 
            | ndarray with spectral data (can be multiple SPDs, 
            | first axis are the wavelengths)
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation
        :out: 
            | 'Rf' or str, optional
            | Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns:
            | float or ndarray with IES TM30-15 Rf for :out: 'Rf'
            | Other output is also possible by changing the :out: str value.
    
    References:
        1. `IES TM30 (99, 4880 spectrally uniform samples) 
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
        
        2. `A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, 
        “Development of the IES method for evaluating the color rendition of light sources,” 
        Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015. 
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888>`_
        
        3. `K. A. G. Smet, A. David, and L. Whitehead, 
        “Why color space uniformity and sample set spectral uniformity are essential for color rendering measures,” 
        LEUKOS, vol. 12, no. 1–2, pp. 39–50, 2016 
        <http://www.tandfonline.com/doi/abs/10.1080/15502724.2015.1091356>`_ 
    
    """
    return spd_to_cri(SPD, cri_type = cri_type, out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_iesrg_tm30_15(SPD, out = 'Rg', wl = None, cri_type ='iesrf-tm30-15'):
    """
    Wrapper function for the 'spd_to_rg' color gamut area index (IES TM30-15). 
    
    Args:
        :SPD: 
            | ndarray with spectral data (can be multiple SPDs, 
            | first axis are the wavelengths)
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation
        :out: 
            | 'Rg' or str, optional
            | Specifies requested output (e.g. 'RgRf,Rfi,cct,duv') 
    
    Returns:
        :returns:
            | float or ndarray with IES TM30-15 Rg for :out: 'Rg'
            | Other output is also possible by changing the :out: str value.
    
    References:
        1. `IES TM30 (99, 4880 spectrally uniform samples) 
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
        
        2. `A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, 
        “Development of the IES method for evaluating the color rendition of light sources,” 
        Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015. 
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888>`_
       
        3. `K. A. G. Smet, A. David, and L. Whitehead, 
        “Why color space uniformity and sample set spectral uniformity are essential for color rendering measures,” 
        LEUKOS, vol. 12, no. 1–2, pp. 39–50, 2016 
        <http://www.tandfonline.com/doi/abs/10.1080/15502724.2015.1091356>`_ 
    
    """
    return spd_to_rg(SPD, cri_type = cri_type, out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_iesrf_tm30_18(SPD, out = 'Rf', wl = None, cri_type = 'iesrf-tm30-18'):
    """
    Wrapper function for the 'iesrf' color fidelity index (IES TM30-18). 
    
    Args:
        :SPD: 
            | ndarray with spectral data (can be multiple SPDs, 
            | first axis are the wavelengths)
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation
        :out: 
            | 'Rf' or str, optional
            | Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns:
            | float or ndarray with IES TM30-18 Rf for :out: 'Rf'
            | Other output is also possible by changing the :out: str value.
    
    References:
        1. `IES TM30 (99, 4880 spectrally uniform samples) 
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
        
        2. `A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, 
        “Development of the IES method for evaluating the color rendition of light sources,” 
        Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015. 
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888>`_
        
        3. `K. A. G. Smet, A. David, and L. Whitehead, 
        “Why color space uniformity and sample set spectral uniformity are essential for color rendering measures,” 
        LEUKOS, vol. 12, no. 1–2, pp. 39–50, 2016 
        <http://www.tandfonline.com/doi/abs/10.1080/15502724.2015.1091356>`_ 
    
    """
    return spd_to_cri(SPD, cri_type = cri_type, out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_iesrg_tm30_18(SPD, out = 'Rg', wl = None, cri_type ='iesrf-tm30-18'):
    """
    Wrapper function for the 'spd_to_rg' color gamut area index (IES TM30-18). 
    
    Args:
        :SPD: 
            | ndarray with spectral data (can be multiple SPDs, 
            | first axis are the wavelengths)
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation
        :out: 
            | 'Rg' or str, optional
            | Specifies requested output (e.g. 'Rg,Rf,Rfi,cct,duv') 
    
    Returns:
        :returns:
            | float or ndarray with IES TM30-18 Rg for :out: 'Rg'
            | Other output is also possible by changing the :out: str value.
    
    References:
        1. `IES TM30 (99, 4880 spectrally uniform samples) 
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
        
        2. `A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, 
        “Development of the IES method for evaluating the color rendition of light sources,” 
        Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015. 
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888>`_
        
        3. `K. A. G. Smet, A. David, and L. Whitehead, 
        “Why color space uniformity and sample set spectral uniformity are essential for color rendering measures,” 
        LEUKOS, vol. 12, no. 1–2, pp. 39–50, 2016 
        <http://www.tandfonline.com/doi/abs/10.1080/15502724.2015.1091356>`_ 
    
    """
    return spd_to_rg(SPD, cri_type = cri_type, out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_iesrf_tm30_20(SPD, out = 'Rf', wl = None, cri_type = 'iesrf-tm30-20'):
    """
    Wrapper function for the 'iesrf' color fidelity index (IES TM30-20 = TM30-18). 
    
    Args:
        :SPD: 
            | ndarray with spectral data (can be multiple SPDs, 
            | first axis are the wavelengths)
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation
        :out: 
            | 'Rf' or str, optional
            | Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns:
            | float or ndarray with IES TM30-20 Rf for :out: 'Rf'
            | Other output is also possible by changing the :out: str value.
    
    References:
        1. `IES TM30 (99, 4880 spectrally uniform samples) 
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
        
        2. `A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, 
        “Development of the IES method for evaluating the color rendition of light sources,” 
        Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015. 
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888>`_
        
        3. `K. A. G. Smet, A. David, and L. Whitehead, 
        “Why color space uniformity and sample set spectral uniformity are essential for color rendering measures,” 
        LEUKOS, vol. 12, no. 1–2, pp. 39–50, 2016 
        <http://www.tandfonline.com/doi/abs/10.1080/15502724.2015.1091356>`_ 
    
    """
    return spd_to_cri(SPD, cri_type = cri_type, out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_iesrg_tm30_20(SPD, out = 'Rg', wl = None, cri_type ='iesrf-tm30-20'):
    """
    Wrapper function for the 'spd_to_rg' color gamut area index (IES TM30-18 = TM30-20). 
    
    Args:
        :SPD: 
            | ndarray with spectral data (can be multiple SPDs, 
            | first axis are the wavelengths)
        :wl: 
            | None, optional
            | Wavelengths (or [start, end, spacing]) to interpolate the SPDs to. 
            | None: default to no interpolation
        :out: 
            | 'Rg' or str, optional
            | Specifies requested output (e.g. 'Rg,Rf,Rfi,cct,duv') 
    
    Returns:
        :returns:
            | float or ndarray with IES TM30-20 Rg for :out: 'Rg'
            | Other output is also possible by changing the :out: str value.
    
    References:
        1. `IES TM30 (99, 4880 spectrally uniform samples) 
        <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
        
        2. `A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, 
        “Development of the IES method for evaluating the color rendition of light sources,” 
        Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015. 
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888>`_
        
        3. `K. A. G. Smet, A. David, and L. Whitehead, 
        “Why color space uniformity and sample set spectral uniformity are essential for color rendering measures,” 
        LEUKOS, vol. 12, no. 1–2, pp. 39–50, 2016 
        <http://www.tandfonline.com/doi/abs/10.1080/15502724.2015.1091356>`_ 
    
    """
    return spd_to_rg(SPD, cri_type = cri_type, out = out, wl = wl)


# additional (latest version) callers:
spd_to_iesrf_tm30 = spd_to_iesrf_tm30_20
spd_to_iesrg_tm30 = spd_to_iesrg_tm30_20
spd_to_iesrf = spd_to_iesrf_tm30_20
spd_to_iesrg = spd_to_iesrg_tm30_20