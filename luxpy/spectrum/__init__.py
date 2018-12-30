# -*- coding: utf-8 -*-

"""
spectrum: sub-package supporting basic spectral calculations
==================================================================

spectrum/cmf.py
---------------

  :luxpy._CMF: | Dict with keys 'types' and x
               | x are dicts with keys 'bar', 'K', 'M'
 
     | * luxpy._CMF['types']  = ['1931_2','1964_10','2006_2','2006_10',
                                 '1931_2_judd1951','1931_2_juddvos1978',
                                 '1951_20_scotopic']
     | * luxpy._CMF[x]['bar'] = numpy array with CMFs for type x 
                                between 360 nm and 830 nm (has shape: (4,471))
     | * luxpy._CMF[x]['K']   = Constant converting Watt to lumen for CMF type x.
     | * luxpy._CMF[x]['M']   = XYZ to LMS conversion matrix for CMF type x.
                                Matrix is numpy arrays with shape: (3,3)
                            
     Notes:
         
        1. All functions have been expanded (when necessary) using zeros to a 
            full 360-830 range. This way those wavelengths do not contribute 
            in the calculation, AND are not extrapolated using the closest 
            known value, as per CIE recommendation.

        2. There are no XYZ to LMS conversion matrices defined for the 
            1964 10°, 1931 2° Judd corrected (1951) 
            and 1931 2° Judd-Vos corrected (1978) cmf sets.
            The Hunt-Pointer-Estevez conversion matrix of the 1931 2° is 
            therefore used as an approximation!
            
        3. The K lm to Watt conversion factors for the Judd and Judd-Vos cmf 
            sets have been set to 683.002 lm/W (same as for standard 1931 2°).
            
        4. The 1951 scoptopic V' function has been replicated in the 3 
            xbar, ybar, zbar columns to obtain a data format similar to the 
            photopic color matching functions. 
            This way V' can be called in exactly the same way as other V 
            functions can be called from the X,Y,Z cmf sets. 
            The K value has been set to 1700.06 lm/W and the conversion matrix 
            to np.eye().


spectrum/spectral.py
--------------------

 :_WL3: Default wavelength specification in vector-3 format: 
        numpy.array([start, end, spacing])

 :_BB: Dict with constants for blackbody radiator calculation 
       constant are (c1, c2, n, na, c, h, k). 

 :_S012_DAYLIGHTPHASE: numpy.ndarray with CIE S0,S1, S2 curves for daylight 
        phase calculation.

 :_INTERP_TYPES: Dict with interpolation types associated with various types of
                 spectral data according to CIE recommendation:  

 :_S_INTERP_TYPE: Interpolation type for light source spectral data

 :_R_INTERP_TYPE: Interpolation type for reflective/transmissive spectral data

 :_CRI_REF_TYPE: Dict with blackbody to daylight transition (mixing) ranges for
                 various types of reference illuminants used in color rendering
                 index calculations.

 :getwlr(): Get/construct a wavelength range from a (start, stop, spacing) 
            3-vector.

 :getwld(): Get wavelength spacing of numpy.ndarray with wavelengths.

 :spd_normalize(): Spectrum normalization (supports: area, max, lambda, 
                   radiometric, photometric and quantal energy units).

 :cie_interp(): Interpolate / extrapolate spectral data following standard 
                [CIE15:2004, “Colorimetry,” CIE, Vienna, Austria, 2004.]

 :spd(): | All-in-one function that can:
         |  1. Read spectral data from data file or take input directly as 
            pandas.dataframe or numpy.array.
         |  2. Convert spd-like data from numpy.array to pandas.dataframe and back.
         |  3. Interpolate spectral data.
         |  4. Normalize spectral data.

 :xyzbar(): Get color matching functions.
        
 :vlbar(): Get Vlambda function.

 :spd_to_xyz(): Calculates xyz tristimulus values from spectral data. 
            
 :spd_to_ler():  Calculates Luminous efficacy of radiation (LER) 
                 from spectral data.

 :spd_to_power(): Calculate power of spectral data in radiometric, photometric
                  or quantal energy units.
         
 :blackbody(): Calculate blackbody radiator spectrum.
             
 :daylightlocus(): Calculates daylight chromaticity from cct. 

 :daylightphase(): Calculate daylight phase spectrum         
         
 :cri_ref(): Calculates a reference illuminant spectrum based on cct for color 
             rendering index calculations.
            (`CIE15:2004CIE15:2004, “Colorimetry,” CIE, Vienna, Austria, 2004. <http://www.cie.co.at/index.php/index.php?i_ca_id=304)>`_, 
             `cie224:2017, CIE 2017 Colour Fidelity Index for accurate scientific use. (2017), ISBN 978-3-902842-61-9. <http://www.cie.co.at/index.php?i_ca_id=1027>`_,
             `IES-TM-30-15: Method for Evaluating Light Source Color Rendition. New York, NY: The Illuminating Engineering Society of North America. <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_

 
spectrum/spectral_databases.py
------------------------------

 :_S_PATH: Path to light source spectra data.

 :_R_PATH: Path to with spectral reflectance data

 :_IESTM3015: Database with spectral reflectances related to and light source 
            spectra contained excel calculator of IES TM30-15 publication.

 :_IESTM3015_S: Database with only light source spectra contained in the 
              IES TM30-15 excel calculator.

 :_CIE_ILLUMINANTS: | Database with CIE illuminants: 
                    | * 'E', 'D65', 'A', 'C',
                    | * 'F1', 'F2', 'F3', 'F4', 'F5', 'F6',
                      'F7', 'F8', 'F9', 'F10', 'F11', 'F12'

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
    
    
    
References
----------

    1. `CIE15-2004 (2004). 
    Colorimetry 
    (Vienna, Austria: CIE) 
    <http://www.cie.co.at/index.php/index.php?i_ca_id=304>`_

    2. `CIE, and CIE (2006). 
    Fundamental Chromaticity Diagram with Physiological Axes - Part I.(Vienna: CIE).
    <http://www.cie.co.at/publications/fundamental-chromaticity-diagram-physiological-axes-part-1>`_
    
    3. `cie224:2017, CIE 2017 Colour Fidelity Index for accurate scientific use. (2017),
    ISBN 978-3-902842-61-9. 
    <http://www.cie.co.at/index.php?i_ca_id=1027>`_
    
    4. `IES-TM-30-15: Method for Evaluating Light Source Color Rendition. 
    New York, NY: The Illuminating Engineering Society of North America. 
    <https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/>`_

===============================================================================
"""
from .basics import *
__all__ = basics.__all__ 