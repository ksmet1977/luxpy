
# Python toolbox for lighting and color science
![alt text][logo]

[logo]: https://github.com/ksmet1977/luxpy/blob/master/images/LUXPY__logo.jpg

* Author: K. A.G. Smet (ksmet1977 at gmail.com)
* Version: 1.2.06
* Date: April 12, 2018
* License: [GPLv3](https://github.com/ksmet1977/luxpy/blob/master/LICENSE.md)


-------------------------------------------------------------------------------
## Module overview:
    
    0.1.  helpers/ 
            helpers.py (imported directly into luxpy namespace, details see end of this file)
    
    0.2.  math/ 
            math.py (imported as math into the luxpy namespace, details see end of this file)
            optimizers.py (imported in math name space)
            
    1.  spectral/ 
            cmf.py
            spectral.py
            spectral_databases.py
            individual_observer_cmf_model.py (imports into indvcmf namespace)
            
    2.  ctf/ 
            colortransforms.py (imported directly into luxpy namespace)
            colortf.py (imported directly into luxpy namespace)
            
    3.  cct/ 
            cct.py (imported directly into luxpy namespace)
            
    4.  cat/ 
            chromaticadaptation.py (imported in luxpy namespace as .cat)
            
    5.  cam/ 
            colorappearancemodels.py (imported in luxpy namespace as .cam)
              cam_02_X.py
              cam15u.py
              sww2016.py

    6.  deltaE/ 
            colordifferences.py (imported in luxpy namespace as .deltaE)
            
    7.  cri/ 
            colorrendition.py (imported in luxpy namespace as .cri)
                DE_scalers.py
                helpers.py
                init_cri_defaults_database.py
                indices.py
                    cie_wrappers.py
                    ies_wrappers.py
                    cri2012.py
                    mcri.py
                    cqs.py
                graphics.py
                ies_tm30_metrics.py
                ies_tm30_graphics.py
                VF_PX_models.py (imported in .cri as .VFPX)
                    vectorshiftmodel.py
                    pixelshiftmodel.py
            
    8. graphics/ 
            plotters.py (imported directly into luxpy namespace)
            
    9. classes/ 
            SPD (imported directly into luxpy namespace)
            CDATA, XYZ, LAB (imported directly into luxpy namespace)
            
    10. ciephotbio/
            cie_tn003_2015.py (imported into luxpy namespace as .ciephotbio)
            
    11. spdbuild/
            spd_builder.py (imported into luxpy namespace as .spdbuild)
    
 
-------------------------------------------------------------------------------
## \__init__.py
     
     Loads above modules and sets some default global (specified with '_') variables/constants:
     * _PKG_PATH (absolute path to luxpy package)
     * _SEP (operating system operator)
     * _EPS = 7./3 - 4./3 -1 (machine epsilon)
     * _CIEOBS = '1931_2' (default CIE observer color matching function)
     * _CSPACE = 'Yuv' (default color space / chromaticity diagram)
 
More info:

    ?luxpy
 
-------------------------------------------------------------------------------
## 1. spectral/ cmf.py

### _CMF:
    
    * luxpy._CMF: Dict with keys 'types' and x
                 x are dicts with keys 'bar', 'K', 'M'
 
     + luxpy._CMF['types'] = ['1931_2','1964_10','2006_2','2006_10',
                             '1931_2_judd1951','1931_2_juddvos1978',
                             '1951_20_scotopic']
     + luxpy._CMF[x]['bar'] = numpy array with CMFs for type x 
                             between 360 nm and 830 nm (has shape: (4,471))
     + luxpy._CMF[x]['K'] = Constant converting Watt to lumen for CMF type x.
     + luxpy._CMF[x]['M'] = XYZ to LMS conversion matrix for CMF type x.
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
    
References:
 1. [CIE15:2004, “Colorimetry,” CIE, Vienna, Austria, 2004.](http://www.cie.co.at/index.php/index.php?i_ca_id=304)

 2. CIE, and CIE (2006). 
     Fundamental Chromaticity Diagram with Physiological Axes - Part I 
     (Vienna: CIE).
     
For more info:

    ?luxpy._CMF

## 1. spectral/ spectral.py

### _WL3:
Default wavelength specification in vector-3 format: [start, end, spacing]

### _BB:
Dict with constants for blackbody radiator calculation (c1, c2, n, n_air, c, h and k) 
* [CIE15:2004, “Colorimetry,” CIE, Vienna, Austria, 2004.](http://www.cie.co.at/index.php/index.php?i_ca_id=304)

### _S012_DAYLIGHTPHASE: 
CIE S0,S1, S2 curves for daylight phase calculation. 
* [CIE15:2004, “Colorimetry,” CIE, Vienna, Austria, 2004.](http://www.cie.co.at/index.php/index.php?i_ca_id=304)

### _INTERP_TYPES:
Dict with interpolation types associated with various types of spectral data according to CIE recommendation
* [CIE15:2004, “Colorimetry,” CIE, Vienna, Austria, 2004.](http://www.cie.co.at/index.php/index.php?i_ca_id=304)

### _S_INTERP_TYPE:
Interpolation type for light source spectral data

### _R_INTERP_TYPE:
Interpolation type for reflective/transmissive spectral data

### _CRI_REF_TYPE:
Dict with blackbody to daylight transition (mixing) ranges for various types of reference illuminants used in color rendering index calculations.

### getwlr():
Get/construct a wavelength range from a 3-vector (start, stop, spacing).

### getwld():
Get wavelength spacing of np.array input.

### normalize_spd():
Spectrum normalization (supports: area, max, lambda, radiometric, photometric and quantal energy units)

### cie_interp():
Interpolate / extrapolate spectral data following standard [CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304).

### spd():
All-in-one function that can:
 * Read spectral data from data file or take input directly as pandas.dataframe or numpy.array.
 * Convert spd-like data from numpy.array to pandas.dataframe and back.
 * Interpolate spectral data.
 * Normalize spectral data.

### xyzbar():
Get color matching functions.

### vlbar():
Get Vlambda function.

### spd_to_xyz():
Calculates xyz from spectral data. ([CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304)).

### spd_to_ler():
Calculates Luminous efficacy of radiation (LER) from spectral data.([CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304)).

# spd_to_power(): 
Calculate power of spectral data in radiometric, photometric or quantal energy units.([CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304)).

### blackbody():
Calculate blackbody radiator spectrum for correlated color temperature (cct). ([CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304))

### daylightlocus():
Calculates daylight chromaticity from cct. ([CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304)).

### daylightphase():
Calculate daylight phase spectrum for correlated color temperature (cct). ([CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304)) 

### cri_ref():
Calculates a reference illuminant spectrum based on cct for color rendering index calculations.
([CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304), 
[cie224:2017, CIE 2017 Colour Fidelity Index for accurate scientific use. (2017), ISBN 978-3-902842-61-9](http://www.cie.co.at/index.php?i_ca_id=1027),
[IES-TM-30-15: Method for Evaluating Light Source Color Rendition. New York, NY: The Illuminating Engineering Society of North America.](https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/)
) 

For more info:

    ?luxpy.spectral
    ?luxpy.spd_to_xyz()
    etc.

## 1. spectral/ spectral_databases.py

### _S_PATH:
Path to light source spectra data.

### _R_PATH:
Path to spectral reflectance data

### _IESTM30:
Database with spectral reflectances related to and light source spectra contained excel calculator of [IES TM30-15](https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/) publication.

### _CIE_ILLUMINANTS:
Database with CIE illuminants:
* 'E', 'D65', 'A', 'C', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12' 

### _CRI_RFL:
Database with spectral reflectance functions for various color rendition calculators
* 'cie-13.3-1995': [CIE 13.3-1995 (8, 14 munsell samples)](http://www.cie.co.at/index.php/index.php?i_ca_id=303), 
* 'cie-224-2017': [CIE 224:2015 (99 set)](http://www.cie.co.at/index.php?i_ca_id=1027)
* 'cri2012': [CRI2012 (HL17 & HL1000 spetcrally uniform and 210 real samples)](http://journals.sagepub.com/doi/abs/10.1177/1477153513481375))
* 'ies-tm30-15': [IES TM30 (99, 4880 sepctrally uniform samples)](https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/)
* 'mcri': [MCRI (10 familiar object set)](http://www.sciencedirect.com/science/article/pii/S0378778812000837)
* 'cqs': [CQS (v7.5 and v9.0 sets)](http://spie.org/Publications/Journal/10.1117/1.3360335)

### _MUNSELL:
Database with 1269 Munsell spectral reflectance functions and Value (V), Chroma (C), hue (h) and (ab) specifications.

For more info:

    ?luxpy.spectral_databases


## 1. spectral/ individual_observer_cmf_model.py (.indvcmf)

### cie2006cmfsEx(): 
Generate Individual Observer CMFs (cone fundamentals) based on CIE2006 cone 
fundamentals and published literature on observer variability in color matching
and in physiological parameters.

### getMonteCarloParam(): 
Get dict with normally-distributed physiological factors for a population of observers.
                            
### getUSCensusAgeDist(): 
Get US Census Age Distribution.

### genMonteCarloObs(): 
Monte-Carlo generation of individual observer color matching functions (cone fundamentals) for a certain age and field size.

### getCatObs(): 
Generate cone fundamentals for categorical observers.

### get_lms_to_xyz_matrix(): 
Calculate lms to xyz conversion matrix for a specific field size.
                            
### lmsb_to_xyzb(): 
Convert from LMS cone fundamentals to XYZ CMF.

### add_to_cmf_dict(): 
Add set of cmfs to _CMF dict.

References:
    1. Asano Y, Fairchild MD, and Blondé L (2016). 
        Individual Colorimetric Observer Model. 
        PLoS One 11, 1–19.
    2. Asano Y, Fairchild MD, Blondé L, and Morvan P (2016). 
        Color matching experiment for highlighting interobserver variability. 
        Color Res. Appl. 41, 530–539.
    3. CIE, and CIE (2006). Fundamental Chromaticity Diagram with Physiological Axes - Part I 
        (Vienna: CIE).
    4. [Asano's Individual Colorimetric Observer Model](https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php)

## 2. ctf/ colortransforms.py
Module with basic colorimetric functions (xyz_to_chromaticity, chromaticity_to_xyz conversions):
Note that colorimetric data is always located in the last axis of the data arrays.
(see also xyz specification in __doc__ string of luxpy.spd_to_xyz())

### xyz_to_Yxy(), Yxy_to_xyz(): 
CIE xyz <--> CIE Yxy 

### xyz_to_Yuv(), Yuv_to_xyz(): 
CIE xyz <--> CIE 1976 Yu'v' 

### xyz_to_wuv(), wuv_to_xyz(): 
CIE 1964 W*U*V* <--> CIE xyz 

###	 xyz_to_xyz():	
CIE xyz <--> CIE xyz (forward = inverse)

###	 xyz_to_lms(), lms_to_xyz:	
CIE xyz <--> CIE lms (cone fundamentals)

###	 xyz_to_lab(), lab_to_xyz(): 
CIE xyz <--> CIELAB 

###	 lab_to_xyz(), xyz_to_luv(): 
CIE xyz <--> CIELUV 

###  xyz_to_Vrb_mb(), Vrb_mb_to_xyz():  
CIE xyz <--> Macleod-Boyton type coordinates (V,r,b) = (V,l,s) with V = L + M, l=L/V, m = M/V (related to luminance)

###   xyz_to_ipt(), ipt_to_xyz():   
CIE xyz <--> IPT ()
* [F. Ebner and M. D. Fairchild, “Development and testing of a color space (IPT) with improved hue uniformity,” in IS&T 6th Color Imaging Conference, 1998, pp. 8–13.](http://www.ingentaconnect.com/content/ist/cic/1998/00001998/00000001/art00003?crawler=true)

###  xyz_to_Ydlep(), Ydlep_to_xyz(): 
CIE xyz <--> Y, dominant / complementary wavelength (dl, compl. wl: specified by < 0) and excitation purity (ep)

References:
   1. CIE15-2004 (2004). Colorimetry (Vienna, Austria: CIE)
   2. Ebner F, and Fairchild MD (1998). 
        Development and testing of a color space (IPT) with improved hue uniformity. 
        In IS&T 6th Color Imaging Conference, (Scottsdale, Arizona, USA), pp. 8–13.
   3. MacLeod DI, and Boynton RM (1979). 
        Chromaticity diagram showing cone excitation by stimuli of equal luminance. 
        J. Opt. Soc. Am. 69, 1183–1186.

For more info:

    ?luxpy.colortransforms
    ?luxpy.xyz_to_Yuv()
    etc.
    
## 2. ctf/ colortf.py

### _COLORTF_DEFAULT_WHITE_POINT: 
XYZ values (numpy.ndarray) of default white point (equi-energy white) 
for color transformations using colortf if none is supplied.

### colortf():
Calculates conversion between any two color spaces for which functions xyz_to_...() and ..._to_xyz() are defined.

For more info:

    ?luxpy.colortf
    etc.    

## 3. cct/ cct.py

### _CCT_LUT_PATH:
Path to Look-Up-Tables (LUT) for correlated color temperature calculation followings [Ohno's method](http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020).

### _CCT_LUT:
Dict with LUTs for cct calculations.

### calculate_lut(): 
Function that calculates LUT for the ccts stored in ./data/cctluts/cct_lut_cctlist.dat 
or given as input argument. Calculation is performed for CMF set specified in cieobs. 
Adds a new (temprorary) field to the _CCT_LUT dict.

### calculate_luts(): 
Function that recalculates (and overwrites) LUTs in ./data/cctluts/ for the ccts 
stored in ./data/cctluts/cct_lut_cctlist.dat or given as input argument. 
Calculation is performed for all CMF sets listed in _CMF['types'].

### xyz_to_cct(): 
Calculates CCT,Duv from XYZ, wrapper for ..._ohno() & ..._search()

### xyz_to_duv(): 
Calculates Duv, (CCT) from XYZ, wrapper for ..._ohno() & ..._search()

### cct_to_xyz(): 
Calculates xyz from CCT, Duv [100 K < CCT < 10**20]

### xyz_to_cct_mcamy(): 
Calculates CCT from XYZ using Mcamy model:
* [McCamy, Calvin S. (April 1992). "Correlated color temperature as an explicit function of chromaticity coordinates". Color Research & Application. 17 (2): 142–144.](http://onlinelibrary.wiley.com/doi/10.1002/col.5080170211/abstract)

### xyz_to_cct_HA(): 
Calculate CCT from XYZ using Hernández-Andrés et al. model .
* [Hernández-Andrés, Javier; Lee, RL; Romero, J (September 20, 1999). Calculating Correlated Color Temperatures Across the Entire Gamut of Daylight and Skylight Chromaticities. Applied Optics. 38 (27): 5703–5709. PMID 18324081. doi:10.1364/AO.38.005703](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-38-27-5703)

### xyz_to_cct_ohno(): 
Calculates CCT,Duv from XYZ using LUT following:
* [Ohno Y. Practical use and calculation of CCT and Duv. Leukos. 2014 Jan 2;10(1):47-55.](http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020)

### xyz_to_cct_search(): 
Calculates CCT,Duv from XYZ using brute-force search algorithm (between 1e2 K - 1e20 K on a log scale)

### cct_to_mired(): 
Converts from CCT to Mired scale (or back)

For more info:

    ?luxpy.cct
    ?luxpy.xyz_to_cct()
    etc.

## 4. cat/ chromaticadaptation.py (.cat)

### cat._WHITE_POINT:   
Default adopted white point

### cat._LA:   
Default luminance of adapting field

### cat._MCATS: 
Default chromatic adaptation sensor spaces
* 'hpe': Hunt-Pointer-Estevez: R. W. G. Hunt, The Reproduction of Colour: Sixth Edition, 6th ed. Chichester, UK: John Wiley & Sons Ltd, 2004.
* 'cat02': from ciecam02: [CIE159-2004, “A Colour Apperance Model for Color Management System: CIECAM02,” CIE, Vienna, 2004.](http://onlinelibrary.wiley.com/doi/10.1002/col.20198/abstract)
* 'cat02-bs':  cat02 adjusted to solve yellow-blue problem (last line = [0 0 1]): [Brill MH, Süsstrunk S. Repairing gamut problems in CIECAM02: A progress report. Color Res Appl 2008;33(5), 424–426.](http://onlinelibrary.wiley.com/doi/10.1002/col.20432/abstract)
* 'cat02-jiang': cat02 modified to solve yb-probem + purple problem: [Jun Jiang, Zhifeng Wang,M. Ronnier Luo,Manuel Melgosa,Michael H. Brill,Changjun Li, Optimum solution of the CIECAM02 yellow–blue and purple problems, Color Res Appl 2015: 40(5), 491-503.](http://onlinelibrary.wiley.com/doi/10.1002/col.21921/abstract)
* 'kries'
* 'judd-1945': from [CIE16-2004](http://www.cie.co.at/index.php/index.php?i_ca_id=436), Eq.4, a23 modified from 0.1 to 0.1020 for increased accuracy
* 'bfd': bradford transform :  [G. D. Finlayson and S. Susstrunk, “Spectral sharpening and the Bradford transform,” 2000, vol. Proceeding, pp. 236–242.](https://infoscience.epfl.ch/record/34077)
* sharp': sharp transform:  [S. Süsstrunk, J. Holm, and G. D. Finlayson, “Chromatic adaptation performance of different RGB sensors,” IS&T/SPIE Electronic Imaging 2001: Color Imaging, vol. 4300. San Jose, CA, January, pp. 172–183, 2001.](http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=903890)
* 'cmc':  [C. Li, M. R. Luo, B. Rigg, and R. W. G. Hunt, “CMC 2000 chromatic adaptation transform: CMCCAT2000,” Color Res. Appl., vol. 27, no. 1, pp. 49–58, 2002.](http://onlinelibrary.wiley.com/doi/10.1002/col.10005/abstract)
* 'ipt':  [F. Ebner and M. D. Fairchild, “Development and testing of a color space (IPT) with improved hue uniformity,” in IS&T 6th Color Imaging Conference, 1998, pp. 8–13.](http://www.ingentaconnect.com/content/ist/cic/1998/00001998/00000001/art00003?crawler=true)
* 'lms':
* bianco':  [S. Bianco and R. Schettini, “Two new von Kries based chromatic adaptation transforms found by numerical optimization,” Color Res. Appl., vol. 35, no. 3, pp. 184–192, 2010.](http://onlinelibrary.wiley.com/doi/10.1002/col.20573/full)
* bianco-pc':  [S. Bianco and R. Schettini, “Two new von Kries based chromatic adaptation transforms found by numerical optimization,” Color Res. Appl., vol. 35, no. 3, pp. 184–192, 2010.](http://onlinelibrary.wiley.com/doi/10.1002/col.20573/full)
* 'cat16': [C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a.](http://onlinelibrary.wiley.com/doi/10.1002/col.22131/abstract)

### cat.check_dimensions():  
Check if dimensions of data and xyzw match.

### cat.get_transfer_function():  
Calculate the chromatic adaptation diagonal matrix transfer function Dt. 
Default = 'vonkries' (others: 'rlab', see Fairchild 1990)

### cat.smet2017_D(): 
Calculate the degree of adaptation based on chromaticity. 
* [Smet, K.A.G.*, Zhai, Q., Luo, M.R., Hanselaer, P., (2017), Study of chromatic adaptation using memory color matches, Part I: neutral illuminants, Opt. Express, 25(7), pp. 7732–7748.](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-7-7732&origin=search)
* [Smet, K.A.G.*, Zhai, Q., Luo, M.R., Hanselaer, P., (2017), Study of chromatic adaptation using memory color matches, Part II: colored illuminants, Opt. Express, 25(7), pp. 8350-8365.](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-7-8350&origin=search)

### cat.get_degree_of_adaptation(): 
Calculates the degree of adaptation. 
D passes either right through or D is calculated following some 
D-function (Dtype) published in literature (cat02, cat16, cmccat, smet2017, manual).
 
### cat.parse_x1x2_parameters():    
Local helper function that parses input parameters and makes them the target_shape for easy calculation 

### cat.apply(): 
Calculate corresponding colors by applying a von Kries chromatic adaptation
transform (CAT), i.e. independent rescaling of 'sensor sensitivity' to data
to adapt from current adaptation conditions (1) to the new conditions (2). 

For more info:

    ?luxpy.cat
    ?luxpy.cat.apply()
    etc.

## 5. cam/ colorappearancemodels.py (cam)

### cam._UNIQUE_HUE_DATA: 
Database of unique hues with corresponding Hue quadratures and eccentricity factors
(ciecam02, cam16, ciecam97s, cam15u)

### cam._SURROUND_PARAMETERS: 
Database of surround parameters c, Nc, F and FLL for ciecam02, cam16, ciecam97s and cam15u.

### cam._NAKA_RUSHTON_PARAMETERS: 
Database with parameters (n, sig, scaling and noise) for the Naka-Rushton function: 
scaling * ((data^n) / ((data^n) + (sig^n))) + noise

### cam._CAM_02_X_UCS_PARAMETERS: 
Database with parameters specifying the conversion from ciecam02/cam16 to
 cam[x]ucs (uniform color space), 
 cam[x]lcd (large color diff.), 
 cam[x]scd (small color diff).

### cam._CAM15U_PARAMETERS: 
Database with CAM15u model parameters.

### cam._CAM_SWW_2016_PARAMETERS: 
Database with cam_sww_2016 parameters (model by Smet, Webster and Whitehead published in JOSA A in 2016)

### cam._CAM_DEFAULT_TYPE: 
Default CAM type str specifier.

### cam._CAM_DEFAULT_WHITE_POINT: 
Default internal reference white point (xyz)

### cam._CAM_DEFAULT_CONDITIONS:
Default CAM model parameters for model in cam._CAM_DEFAULT_TYPE

### cam._CAM_AXES: 
Dict with list[str,str,str] containing axis labels of defined cspaces.

### cam.naka_rushton(): 
Applies a Naka-Rushton function to the input (forward and inverse available)
 
### cam.hue_angle(): 
Calculates a positive hue_angle

### cam.hue_quadrature(): 
Calculates the hue_quadrature from the hue and the parameters in the _unique_hue_data database

### cam.cam_structure_ciecam02_cam16(): 
Basic structure of both the ciecam02 and cam16 models. Has 'forward' (xyz --> color attributes) and 'inverse' (color attributes --> xyz) modes.
* [N. Moroney, M. D. Fairchild, R. W. G. Hunt, C. Li, M. R. Luo, and T. Newman, (2002), "The CIECAM02 color appearance model,” IS&T/SID Tenth Color Imaging Conference. p. 23, 2002.](http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf)
* [C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, (2017), “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a.](http://onlinelibrary.wiley.com/doi/10.1002/col.22131/abstract)

### cam.ciecam02(): 
Calculates ciecam02 output (wrapper for cam_structure_ciecam02_cam16 with specifics of ciecam02)
* [N. Moroney, M. D. Fairchild, R. W. G. Hunt, C. Li, M. R. Luo, and T. Newman, (2002), "The CIECAM02 color appearance model,” IS&T/SID Tenth Color Imaging Conference. p. 23, 2002.](http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf)

### cam.cam16(): 
Calculates cam16 output (wrapper for cam_structure_ciecam02_cam16 with specifics of cam16)
* [C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, (2017), “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a.](http://onlinelibrary.wiley.com/doi/10.1002/col.22131/abstract)
 
### cam.camucs_structure(): 
Basic structure to go to ucs, lcd and scd color spaces (forward + inverse available)
* [M. R. Luo, G. Cui, and C. Li, “Uniform colour spaces based on CIECAM02 colour appearance model,” Color Res. Appl., vol. 31, no. 4, pp. 320–330, 2006.](http://onlinelibrary.wiley.com/doi/10.1002/col.20227/abstract)

### cam.cam02ucs(): 
Calculates ucs (or lcd, scd) output based on ciecam02 (forward + inverse available)
* [M. R. Luo, G. Cui, and C. Li, “Uniform colour spaces based on CIECAM02 colour appearance model,” Color Res. Appl., vol. 31, no. 4, pp. 320–330, 2006.](http://onlinelibrary.wiley.com/doi/10.1002/col.20227/abstract)

### cam.cam16ucs(): 
Calculates ucs (or lcd, scd) output based on cam16 (forward + inverse available)
* [C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, (2017), “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a.](http://onlinelibrary.wiley.com/doi/10.1002/col.22131/abstract)
* [M. R. Luo, G. Cui, and C. Li, “Uniform colour spaces based on CIECAM02 colour appearance model,” Color Res. Appl., vol. 31, no. 4, pp. 320–330, 2006.](http://onlinelibrary.wiley.com/doi/10.1002/col.20227/abstract)

### cam.cam15u(): 
Calculates the output for the CAM15u model for self-luminous unrelated stimuli.
* [M. Withouck, K. A. G. Smet, W. R. Ryckaert, and P. Hanselaer, (2015), “Experimental driven modelling of the color appearance of unrelated self-luminous stimuli: CAM15u,”  Opt. Express, vol. 23, no. 9, pp. 12045–12064.](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-9-12045&origin=search)
* [M. Withouck, K. A. G. Smet, and P. Hanselaer, (2015), “Brightness prediction of different sized unrelated self-luminous stimuli,” Opt. Express, vol. 23, no. 10, pp. 13455–13466.](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-10-13455&origin=search)

### cam_sww16(): 
A simple principled color appearance model based on a mapping of the Munsell color system.
This function implements the JOSA A (parameters = 'JOSA') published model. 
* [Smet, K. A. G., Webster, M. A., & Whitehead, L. A. (2016). A simple principled approach for modeling and understanding uniform color metrics. Journal of the Optical Society of America A, 33(3), A319–A331.](https://www.osapublishing.org/josaa/abstract.cfm?URI=josaa-33-3-a319)


### specific wrappers in the xyz_to_...() and ..._to_xyz() format:
 * 'xyz_to_jabM_ciecam02', 'jabM_ciecam02_to_xyz',
 * 'xyz_to_jabC_ciecam02', 'jabC_ciecam02_to_xyz',
 * 'xyz_to_jabM_cam16', 'jabM_cam16_to_xyz',
 * 'xyz_to_jabC_cam16', 'jabC_cam16_to_xyz',
 * 'xyz_to_jab_cam02ucs', 'jab_cam02ucs_to_xyz', 
 * 'xyz_to_jab_cam02lcd', 'jab_cam02lcd_to_xyz',
 * 'xyz_to_jab_cam02scd', 'jab_cam02scd_to_xyz', 
 * 'xyz_to_jab_cam16ucs', 'jab_cam16ucs_to_xyz',
 * 'xyz_to_jab_cam16lcd', 'jab_cam16lcd_to_xyz',
 * 'xyz_to_jab_cam16scd', 'jab_cam16scd_to_xyz', 
 * 'xyz_to_qabW_cam15u', 'qabW_cam15u_to_xyz'
 * 'xyz_to_lAb_cam_sww16', 'lab_cam_sww16_to_xyz'

These wrapper functions are imported directly into the luxpy namespace. 

For more info:

    ?luxpy.cam
    ?luxpy.cam.cam_sww16()
    etc.

## 6. deltaE/ colordifference.py (.deltaE)

### process_DEi(): 
Process color difference input DEi for output (helper fnc).

### DE_camucs(): 
Calculate color appearance difference DE using camucs type model.

### DE_2000(): 
Calculate DE2000 color difference.

### DE_cspace():  
Calculate color difference DE in specific color space.


## 7. cri/ colorrendition.py (.cri)

### _CRI_TYPE_DEFAULT: 
Default cri_type.

### _CRI_DEFAULTS: 
Default parameters for color fidelity and gamut area metrics 
        
    (major dict has 9 keys: 
            sampleset [str/dict], 
            ref_type [str], 
            cieobs [str], 
            avg [fcn handle], 
            scale [dict], 
            cspace [dict], 
            catf [dict], 
            rg_pars [dict], 
            cri_specific_pars [dict])
     
     Supported cri-types:
    * 'ciera','ciera-8','ciera-14','cierf',
    * 'iesrf','iesrf-tm30-15','iesrf-tm30-18',
    * 'cri2012','cri2012-hl17','cri2012-hl1000','cri2012-real210',
    * 'mcri',
    * 'cqs-v7.5','cqs-v9.0'

            
### process_cri_type_input(): 
load a cri_type dict but overwrites any keys that have a non-None input in calling function.



### cri.linear_scale():  
Linear color rendering index scale from [CIE13.3-1974/1995](http://www.cie.co.at/index.php/index.php?i_ca_id=303):   Ri,a = 100 - c1*DEi,a. (c1 = 4.6)

### cri.log_scale(): 
Log-based color rendering index scale from [Davis & Ohno (2010)](http://spie.org/Publications/Journal/10.1117/1.3360335):  Ri,a = 10 * ln(exp((100 - c1*DEi,a)/10) + 1)

### cri.psy_scale():
Psychometric based color rendering index scale from CRI2012 ([Smet et al. 2013, LRT](http://journals.sagepub.com/doi/abs/10.1177/1477153513481375)):  Ri,a = 100 * (2 / (exp(c1*abs(DEi,a)**(c2) + 1))) ** c3



### gamut_slicer(): 
Slices the gamut in nhbins slices and provides normalization of test gamut to reference gamut.

### jab_to_rg(): 
Calculates gamut area index, Rg.

### jab_to_rhi(): 
Calculate hue bin measures: 

     *Rfhi (local (hue bin) color fidelity)
     *Rcshi (local chroma shift) 
     *Rhshi (local hue shift)

### spd_to_jab_t_r(): 
Calculates jab color values for a sample set illuminated with test source and its reference illuminant.
                  
### spd_to_rg(): 
Calculates the color gamut index of spectral data for a sample set illuminated with test source (data) with respect to some reference illuminant.

### spd_to_DEi(): 
Calculates color difference (~fidelity) of spectral data between sample set illuminated with test source (data) and some reference illuminant.

### optimize_scale_factor(): 
Optimize scale_factor of cri-model in cri_type such that average Rf for a set of light sources is the same as that of a target-cri (default: 'ciera')

### spd_to_cri(): 
Calculates the color rendering fidelity index (CIE Ra, CIE Rf, IES Rf, CRI2012 Rf) of spectral data. Can also output Rg, Rfhi, Rcshi, Rhshi, cct, duv, ...

### wrapper functions for fidelity type metrics:
* cri.spd_to_ciera(), spd_to_ciera_133_1995():
    * [CIE13.3-1995, “Method of Measuring and Specifying Colour Rendering Properties of Light Sources,” CIE, Vienna, Austria, 1995.,ISBN 978 3 900734 57 2](http://www.cie.co.at/index.php/index.php?i_ca_id=303)
* cri.spd_to_cierf(): [latest version]
* spd_to_cierf_224_2017():
    * [cie224:2017, CIE 2017 Colour Fidelity Index for accurate scientific use. (2017), ISBN 978-3-902842-61-9](http://www.cie.co.at/index.php?i_ca_id=1027)
* cri.spd_to_iesrf() [latest version]
* spd_to_iesrf_tm30_15():[TM30-15, 2015 version]
    * [IES TM30 (99, 4880 sepctrally uniform samples)](https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/)
    * [A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, “Development of the IES method for evaluating the color rendition of light sources,” Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015.](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888)
    * [K. A. G. Smet, A. David, and L. Whitehead, “Why color space uniformity and sample set spectral uniformity are essential for color rendering measures,” LEUKOS, vol. 12, no. 1–2, pp. 39–50, 2016](http://www.tandfonline.com/doi/abs/10.1080/15502724.2015.1091356)
* spd_to_iesrf_tm30_18():[TM30-18, 2018 version]
* cri.spd_to_cri2012(), cri.spd_to_cri2012_hl17(), cri.spd_to_cri2012_hl1000(), cri.spd_to_cri2012_real210
    * [K. Smet, J. Schanda, L. Whitehead, and R. Luo, “CRI2012: A proposal for updating the CIE colour rendering index,” Light. Res. Technol., vol. 45, pp. 689–709, 2013](http://journals.sagepub.com/doi/abs/10.1177/1477153513481375)

### wrapper functions for gamut area type metrics:
* cri.spd_to_iesrg() [latest version]
* spd_to_iesrg_tm30(): [latest version]
* spd_to_iesrg_tm30_15(): [TM30-15, 2015 version]
    * [IES TM30 (99, 4880 sepctrally uniform samples)](https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/)
* spd_to_iesrg_tm30_18(): [TM30-18, 2018 version]

### cri.spd_to_mcri(): 
Calculates the memory color rendition index, Rm:  
* [K. A. G. Smet, W. R. Ryckaert, M. R. Pointer, G. Deconinck, and P. Hanselaer, (2012) “A memory colour quality metric for white light sources,” Energy Build., vol. 49, no. C, pp. 216–225.](http://www.sciencedirect.com/science/article/pii/S0378778812000837)

### cri.spd_to_cqs(): 
Versions 7.5 and 9.0 are supported.  
* [W. Davis and Y. Ohno, “Color quality scale,” (2010), Opt. Eng., vol. 49, no. 3, pp. 33602–33616.](http://spie.org/Publications/Journal/10.1117/1.3360335)


### plot_hue_bins(): 
Makes basis plot for Color Vector Graphic (CVG).

### plot_ColorVectorGraphic():
Plots Color Vector Graphic (see IES TM30).


### colorrendition_VF_PX_models module (.cri.VFPX)
* VF: Implements a Vector Field model to calculate the base color shift generated by a light source.
and to calculate a Metameric uncertainty index
* PX: Implements a Color Space Pixelation method to assess light source induced color shifts across color space.

For more info:

    ?luxpy.cri.VFPX

### spd_to_ies_tm30_metrics(): 
Calculates IES TM30 metrics from spectral data.

### plot_cri_graphics(): 
Plot_cri_graphics(): Plots graphical information on color rendition properties based on spectral data input or dict with pre-calculated measures.


For more info on .cri module:

    ?luxpy.cri
    ?luxpy.cri.spd_to_cri()
    etc.

## 8. graphics/ plotters.py

### plot_color_data():
Plot color data (local helper function)

### plotDL():
Plot daylight locus. 

### plotBB():
Plot blackbody locus. 

### plotSL():
Plot spectrum locus. (plotBB() and plotDL() are also called, but can be turned off).

### plotceruleanline():
Plot cerulean (yellow (577 nm) - blue (472 nm)) line (Kuehni, CRA, 2014: Table II: spectral lights) [Kuehni, R. G. (2014). Unique hues and their stimuli—state of the art. Color Research & Application, 39(3), 279–287](https://doi.org/10.1002/col.21793)

### plotUH():
Plot unique hue lines from color space center point xyz0. (Kuehni, CRA, 2014: uY,uB,uG: Table II: spectral lights; uR: Table IV: Xiao data) [Kuehni, R. G. (2014). Unique hues and their stimuli—state of the art. Color Research & Application, 39(3), 279–287](https://doi.org/10.1002/col.21793)

### plotcircle():
Plot one or more concentric circles.

For more info:

    ?luxpy.plotters
    ?luxpy.plotDL()
    etc.


## 9. classes/ SPD.py and CDATA.py (classes SPD, CDATA, XYZ, LAB)

### class SPD
Class with the following methods:


-------------------------------------------------------------------------------
## 0.1.  helpers/ helpers.py 

### np2d():
Make a tuple, list or numpy array at least 2d array.

### np2dT():
Make a tuple, list or numpy array at least 2d array and tranpose.

### np3d():
Make a tuple, list or numpy array at least 3d array.

### np3dT():
Make a tuple, list or numpy array at least 3d array and tranpose (swap) first two axes.

### put_args_in_db():
Takes the **args with not-None input values of a function fcn and overwrites the values of the corresponding keys in Dict db.
See put_args_in_db? for more info.

### getdata():
Get data from csv-file or convert between pandas dataframe (kind = 'df') and numpy 2d-array (kind = 'np').

### dictkv():
Easy input of of keys and values into dict (both should be iterable lists).

### OD():
Provides a nice way to create OrderedDict "literals".

### meshblock():
Create a meshed block (similar to meshgrid, but axis = 0 is retained) to enable fast blockwise calculation.

### aplit():
Split np.array data on (default = last) axis.

### ajoin():
Join tuple of np.array data on (default = last) axis.

### broadcast_shape():
Broadcasts shapes of data to a target_shape. Useful for block/vector calculation in which nupy fails to broadcast correctly.

## todim():  
Expand x to dimensions that are broadcast-compatable with shape_ of another array.

For more info:

    ?luxpy.helpers
    ?luxpy.np2d()
    etc.


## 0.2.  math/ math.py (math)

### normalize_3x3_matrix():  
Normalize 3x3 matrix M to xyz0 -- > [1,1,1]

### math.line_intersect():
Line intersections of series of two line segments a and b.
* From [https://stackoverflow.com/questions/3252194/numpy-and-line-intersections](https://stackoverflow.com/questions/3252194/numpy-and-line-intersections)

### math.positive_arctan():
Calculates positive angle (0°-360° or 0 - 2*pi rad.) from x and y.

### math.dot23():
Dot product of a 2-d numpy.ndarray with a (N x K x L) 3-d numpy.array using einsum().

### math.check_symmetric():
Checks if A is symmetric.

### math.check_posdef():
Checks positive definiteness of a matrix via Cholesky.

### math.symmM_to_posdefM():
Converts a symmetric matrix to a positive definite one. Two methods are supported:
* 'make': A Python/Numpy port of Muhammad Asim Mubeen's matlab function Spd_Mat.m (https://nl.mathworks.com/matlabcentral/fileexchange/45873-positive-definite-matrix)
* 'nearest': A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code. (https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite)

### math.bvgpdf():
Calculates bivariate Gaussian (PD) function, with center mu and shape and orientation determined by sigmainv. 

### math.mahalanobis2():
Calculates mahalanobis.^2 distance with center mu and shape and orientation determined by sigmainv. 

### math.rms():
Calculates root-mean-square along axis.

### math.geomean():
Calculates geometric mean along axis.

### math.polyarea():
Calculates area of polygon. (First coordinate should also be last)

### math.erf(), math.erfinv(): 
erf-function (and inverse), direct import from scipy.special

# cart2pol(): 
Converts Cartesian to polar coordinates.

# pol2cart(): 
Converts polar to Cartesian coordinates.

# magnitude_v():  
Calculates magnitude of vector.

# angle_v1v2():  
Calculates angle between two vectors.

For more info:

    ?luxpy.math
    ?luxpy.math.bvgpdf()
    etc.
