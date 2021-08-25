# -*- coding: utf-8 -*-

"""
spectrum: sub-package supporting basic spectral calculations
============================================================

spectrum/cmf.py
---------------

    :luxpy._CMF: 
      | Dict with keys 'types' and x
      | x are dicts with keys 'bar', 'K', 'M'
      |
      | * luxpy._CMF['types']  = ['1931_2','1964_10',
      |                           '2006_2','2006_10','2015_2','2015_10',
      |                           '1931_2_judd1951','1931_2_juddvos1978',
      |                           '1951_20_scotopic']
      | * luxpy._CMF[x]['bar'] = numpy array with CMFs for type x 
      |                          between 360 nm and 830 nm (has shape: (4,471))
      | * luxpy._CMF[x]['K']   = Constant converting Watt to lumen for CMF type x.
      | * luxpy._CMF[x]['M']   = XYZ to LMS conversion matrix for CMF type x.
      |                          Matrix is numpy array with shape: (3,3)
      | * luxpy._CMF[x]['N']   = XYZ to RGB conversion matrix for CMF type x.
      |                          Matrix is numpy array with shape: (3,3)
                            
     Notes:
         
        1. All functions have been expanded (when necessary) using zeros to a 
            full 360-830 range. This way those wavelengths do not contribute 
            in the calculation, AND are not extrapolated using the closest 
            known value, as per CIE recommendation.

        2. There is no XYZ to LMS conversion matrices defined for the 
            1931 2° Judd corrected (1951) cmf sets.
            The Hunt-Pointer-Estevez conversion matrix of the 1931 2° is 
            therefore used as an approximation!
            
        3. The XYZ to LMS conversion matrix M for the Judd-Vos XYZ CMFs is the one
            that converts to the 1979 Smith-Pokorny cone fundamentals.
            
        4. The XYZ to LMS conversion matrix for the 1964 10° XYZ CMFs is set
            to the one of the CIE 2006 10° cone fundamentals, as not matrix has
            been officially defined for this CMF set.
            
        4. The K lm to Watt conversion factors for the Judd and Judd-Vos cmf 
            sets have been set to 683.002 lm/W (same as for standard 1931 2°).
            
        5. The 1951 scoptopic V' function has been replicated in the 3 
            xbar, ybar, zbar columns to obtain a data format similar to the 
            photopic color matching functions. 
            This way V' can be called in exactly the same way as other V 
            functions can be called from the X,Y,Z cmf sets. 
            The K value has been set to 1700.06 lm/W and the conversion matrix 
            has been filled with NaN's.
            
        6. The '2015_x' (with x = 2 or 10) are the same XYZ-CMFs as stored in '2006_x'.
        
        7. _CMF[x]['M'] for x equal to '2006_2' (='2015_2') or '2006_10' (='2015_10') is NOT 
            normalized to illuminant E! These are the original matrices 
            as defined by [1] & [2].
            
        8. _CMF[x]['N'] stores known or calculated conversion matrices from
            xyz to rgb. If not available, N has been filled with NaNs.




spectrum/spectral.py
--------------------

 :_WL3: Default wavelength specification in vector-3 format: 
        numpy.array([start, end, spacing])

 :_INTERP_TYPES: Dict with interpolation types associated with various types of
                 spectral data according to CIE recommendation:  

 :_S_INTERP_TYPE: Interpolation type for light source spectral data

 :_R_INTERP_TYPE: Interpolation type for reflective/transmissive spectral data
 
 :_C_INTERP_TYPE: Interpolation type for CMF and cone-fundamental spectral data


 :getwlr(): Get/construct a wavelength range from a (start, stop, spacing) 
            3-vector.

 :getwld(): Get wavelength spacing of numpy.ndarray with wavelengths.

 :spd_normalize(): Spectrum normalization (supports: area, max, lambda, 
                   radiometric, photometric and quantal energy units).

 :cie_interp(): Interpolate / extrapolate spectral data following standard 
                [`CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_]

 :spd(): | All-in-one function that can:
         |  1. Read spectral data from data file or take input directly as 
            pandas.dataframe or ndarray.
         |  2. Convert spd-like data from ndarray to pandas.dataframe and back.
         |  3. Interpolate spectral data.
         |  4. Normalize spectral data.

 :xyzbar(): Get color matching functions.
        
 :vlbar(): Get Vlambda function.
 
 :vlbar_cie_mesopic(): Get CIE mesopic luminous efficiency function Vmesm according to CIE191:2010

 :get_cie_mesopic_adaptation(): Get the mesopic adaptation state according to CIE191:2010

 :spd_to_xyz(): Calculates xyz tristimulus values from spectral data. 
            
 :spd_to_ler():  Calculates Luminous efficacy of radiation (LER) 
                 from spectral data.

 :spd_to_power(): Calculate power of spectral data in radiometric, photometric
                  or quantal energy units.
         
 :detect_peakwl(): Detect peak wavelengths and fwhm of peaks in spectrum spd.
         
  
spectrum/spectral_databases.py
------------------------------

 :_S_PATH: Path to light source spectra data.

 :_R_PATH: Path to with spectral reflectance data

 :_IESTM3015: Database with spectral reflectances related to and light source 
            spectra contained excel calculator of IES TM30-15 publication.
            
 :_IESTM3018: Database with spectral reflectances related to and light source 
            spectra contained excel calculator of IES TM30-18 publication.

 :_IESTM3015_S: Database with only light source spectra contained in the 
              IES TM30-15 excel calculator.
              
 :_IESTM3018_S: Database with only light source spectra contained in the 
              IES TM30-18 excel calculator.

 :_CIE_ILLUMINANTS: | Database with CIE illuminants: 
                    | * 'E', 'D65', 'A', 'C',
                    | * 'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
                      'F7', 'F8', 'F9', 'F10', 'F11', 'F12'
                      
 :_CIE_E, _CIE_D65, _CIE_A, _CIE_C, _CIE_F4: Some CIE illuminants for easy use.

 :_CRI_RFL: | Database with spectral reflectance functions for various 
              color rendition calculators:
            | * `CIE 13.3-1995 (8, 14 munsell samples) <http://www.cie.co.at/index.php/index.php?i_ca_id=303>`_
            | * `CIE 224:2015 (99 set) <http://www.cie.co.at/index.php?i_ca_id=1027>`_
            | * `CRI2012 (HL17 & HL1000 spectrally uniform and 210 real samples) <http://journals.sagepub.com/doi/abs/10.1177/1477153513481375>`_
            | * `IES TM30 (99, 4880 sepctrally uniform samples) <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition>`_
            | * `MCRI (10 familiar object set) <http://www.sciencedirect.com/science/article/pii/S0378778812000837>`_
            | * `CQS (v7.5 and v9.0 sets) <http://spie.org/Publications/Journal/10.1117/1.3360335>`_

 :_MUNSELL: Database (dict) with 1269 Munsell spectral reflectance functions 
            and Value (V), Chroma (C), hue (h) and (ab) specifications.
           
 :_RFL: | Database (dict) with RFLs, including:
        | * all those in _CRI_RFL, 
        | * the 1269 Matt Munsell samples (see also _MUNSELL),
        | * the 24 Macbeth ColorChecker samples,
        | * the 215 samples proposed by Opstelten, J.J. , 1983, The establishment of a representative set of test colours
        |   for the specification of the colour rendering properties of light sources, CIE-20th session, Amsterdam. 
        | * the 114120 RFLs from capbone.com/spectral-reflectance-database/
    
spectrum/illuminants.py
-----------------------

 :_BB: Dict with constants for blackbody radiator calculation 
       constant are (c1, c2, n, na, c, h, k). 

 :_S012_DAYLIGHTPHASE: ndarray with CIE S0,S1, S2 curves for daylight 
        phase calculation (linearly interpolated to 1 nm).
        
 :_CRI_REF_TYPES: Dict with blackbody to daylight transition (mixing) ranges for
                 various types of reference illuminants used in color rendering
                 index calculations.
        
 :blackbody(): Calculate blackbody radiator spectrum.
 
 :_DAYLIGHT_LOCI_PARAMETERS: dict with parameters for daylight loci for various CMF sets; used by daylightlocus().
 
 :_DAYLIGHT_M12_COEFFS: dict with coefficients in weights M1 & M2 for daylight phases for various CMF sets.
 
 :get_daylightloci_parameters(): Get parameters for the daylight loci functions xD(1000/CCT) and yD(xD); used by daylightlocus().

 :get_daylightphase_Mi_coeffs(): Get coefficients of Mi weights of daylight phase for specific cieobs following Judd et al. (1964).

 :_get_daylightphase_Mi_values(): Get daylight phase coefficients M1, M2 following Judd et al. (1964).         

 :_get_daylightphase_Mi(): Get daylight phase coefficients M1, M2 following Judd et al. (1964)            
 
 :daylightlocus(): Calculates daylight chromaticity from cct. 

 :daylightphase(): Calculate daylight phase spectrum.
         
 :cri_ref(): Calculates a reference illuminant spectrum based on cct for color 
             rendering index calculations.
            (`CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_, 
             `cie224:2017, CIE 2017 Colour Fidelity Index for accurate scientific use. (2017), ISBN 978-3-902842-61-9. <http://www.cie.co.at/index.php?i_ca_id=1027>`_,
             `IES-TM-30-15: Method for Evaluating Light Source Color Rendition. New York, NY: The Illuminating Engineering Society of North America. <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
 
 :spd_to_indoor(): Convert spd to indoor variant by multiplying it with the CIE spectral transmission for glass. 

spectrum/spdx_iestm2714.py
--------------------------

 :_SPDX_TEMPLATE: template dictionary for SPDX data.
 
 :read_spdx(): Read xml file or convert xml string with spdx data to dictionary.
     
 :write_spdx(): Convert spdx dictionary to xml string (and write to .spdx file)


References
----------

    1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_

    2. `CIE, and CIE (2006). 
    Fundamental Chromaticity Diagram with Physiological Axes - Part I.(Vienna: CIE).
    <http://www.cie.co.at/publications/fundamental-chromaticity-diagram-physiological-axes-part-1>`_
    
    3. `cie224:2017, CIE 2017 Colour Fidelity Index for accurate scientific use. (2017),
    ISBN 978-3-902842-61-9. 
    <http://www.cie.co.at/index.php?i_ca_id=1027>`_
    
    4. `IES-TM-30-15: Method for Evaluating Light Source Color Rendition. 
    New York, NY: The Illuminating Engineering Society of North America. 
    <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_
    
    5. Judd, D. B., MacAdam, D. L., Wyszecki, G., Budde, H. W., Condit, H. R., Henderson, S. T., & Simonds, J. L. (1964). Spectral Distribution of Typical Daylight as a Function of Correlated Color Temperature. J. Opt. Soc. Am., 54(8), 1031–1040. https://doi.org/10.1364/JOSA.54.001031

    6. http://www.ies.org/iestm2714
    
spectrum/detector_spectral_mismatch.py
--------------------------------------

 :f1prime(): Determine the f1prime spectral mismatch index.
 
 :get_spectral_mismatch_correct_factors(): Determine the spectral mismatch factors.


Reference
---------
    1. Krüger, U. et al. GENERAL V(λ) MISMATCH - INDEX HISTORY, CURRENT STATE, NEW IDEAS
    
===============================================================================
"""
from .basics import *
__all__ = basics.__all__ 

from .spdx_ietm2714 import read_spdx, write_spdx, _SPDX_TEMPLATE
__all__ += ['read_spdx', 'write_spdx', '_SPDX_TEMPLATE']

from .detector_spectral_mismatch import f1prime, get_spectral_mismatch_correction_factors
__all__ += ['f1prime','get_spectral_mismatch_correction_factors'] 