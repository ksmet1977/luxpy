# luxpy: Python package for lighting and color science

* Author: K. A.G. Smet
* Version: 1.1.5
* Date: Sep 11, 2017
* License: GPLv3

-------------------------------------------------------------------------------
## Overview of modules (in order as loaded in __init__):

    0.1.  helpers.py (imported directly into luxpy namespace, details see end of this file)
    0.2.  math.py (imported as math into the luxpy namespace, details see end of this file)
    
    1.  cmf.py
    2.  spectral.py
    3.  spectral_databases
    4.  colortransforms.py (imported directly into luxpy namespace)
    5.  cct.py (imported directly into luxpy namespace)
    6.  chromaticadaptation.py (imported in luxpy namespace as .cat)
    7.  colorappearancemodels.py (imported in luxpy namespace as .cam)
    8.  colortf.py (imported directly into luxpy namespace)
    9.  colorrenditionindices.py (imported in luxpy namespace as .cri)
    10. plotters.py (imported directly into luxpy namespace)
    
 
-------------------------------------------------------------------------------
## 1. cmf.py

### _cmf:
Dict with info on several sets of color matching functions:
 * '1931_2', '1964_10','2006_2','2006_10' (CIE 1931 2°, CIE 1964 10°, CIE 2006 2° and CIE 2006 10° observers)
 * Dict keys are: 'types, 'K' (lm/W), 'M' (xyz -- > lms conversion matrix), 'bar' (color matching functions, downloaded from cvrl.org)

## 2. spectral.py

### _wl3:
Default wavelength specification in vector-3 format: [start, end, spacing]

### _BB:
Constants for blackbody radiator calculation (c1, c2, n) 
* [CIE15:2004, “Colorimetry,” CIE, Vienna, Austria, 2004.](http://www.cie.co.at/index.php/index.php?i_ca_id=304)


### _S012_daylightphase: 
CIE S0,S1, S2 curves for daylight phase calculation. 
* [CIE15:2004, “Colorimetry,” CIE, Vienna, Austria, 2004.](http://www.cie.co.at/index.php/index.php?i_ca_id=304)

### _interp_types:
Dict with interpolation types associated with various types of spectral data according to CIE recommendation
* [CIE15:2004, “Colorimetry,” CIE, Vienna, Austria, 2004.](http://www.cie.co.at/index.php/index.php?i_ca_id=304)

### _S_interp_type:
Interpolation type for light source spectral data

### _R_interp_type:
Interpolation type for reflective/transmissive spectral data

### _cri_ref_type:
Dict with blackbody to daylight transition (mixing) ranges for various types of reference illuminants used in color rendering index calculations.

### getwlr():
Get/construct a wavelength range from a 3-vector (start, stop, spacing), output is a (n,)-vector.

### getwld():
Get wavelength spacing of np.array input.

### normalize_spd():
Spectrum normalization (supports: area, max and lambda)

### cie_interp():
Interpolate / extrapolate (i.e. flat: replicate closest known values) following [CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304).

### spd():
All-in-one function: convert spd-like data from np.array to pd.dataframe, interpolate (use wl and interpolation like in cie_interp), normalize.

### xyzbar():
Get color matching functions from file (./data/cmfs/) 
Load cmfs from file or get from dict defined in .cmf.py.

### spd_to_xyz():
Calculates xyz from spd following [CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304).
If rfl input argument is not None, the first input argument is assumed to contain light source spectra illuminating the spectral reflection functions contained in 'rfl'.
Output will be [N x M x 3], with N number of light source spectra, M, number of spectral reflectance function and last axis referring to xyz. 

### blackbody():
Calculate blackbody radiator spd for correlated color temperature = cct. ([CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304))
Input cct must be float (for multiple cct, use cri_ref() with ref_type = 'BB').

### daylightlocus():
Calculates daylight chromaticity for cct ([CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304)).

### daylightphase():
Calculate daylight phase spd for correlated color temperature = cct. ([CIE15:2004](http://www.cie.co.at/index.php/index.php?i_ca_id=304)) 
Default lower cct-limit is 4000 K, but can be turned off by setting 'force_daylight_below4000K' to True
Input cct must be float (for multiple cct, use cri_ref() with ref_type = 'DL').

### cri_ref():
Calculates a reference illuminant for cri calculation based on cct. Type and CIE observer can be set in resp. ref_type and cieobs. 
Input cct can be np.array, in which case output is 2d-array of spectra.



## 3. spectral_databases.py

### _S_dir:
Folder with light source spectra data.

### _R_dir:
Folder with spectral reflectance data

### _iestm30:
Database with spectral reflectances related to and light source spectra contained excel calculator of [IES TM30-15](https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/) publication.

### _cie_illuminants:
Database with CIE illuminants:
* 'E', 'D65', 'A', 'C', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12' 

### _cri_rfl:
Database with spectral reflectance functions for various color rendition calculators
* 'cie-13.3-1995': [CIE 13.3-1995 (8, 14 munsell samples)](http://www.cie.co.at/index.php/index.php?i_ca_id=303), 
* 'cie-224-2017': [CIE 224:2015 (99 set)](http://www.cie.co.at/index.php?i_ca_id=1027)
* 'cri2012': [CRI2012 (HL17 & HL1000 spetcrally uniform and 210 real samples)](http://journals.sagepub.com/doi/abs/10.1177/1477153513481375))
* 'ies-tm30-15': [IES TM30 (99, 4880 sepctrally uniform samples)](https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/)
* 'mcri': [MCRI (10 familiar object set)](http://www.sciencedirect.com/science/article/pii/S0378778812000837)
* 'cqs': [CQS (v7.5 and v9.0 sets)](http://spie.org/Publications/Journal/10.1117/1.3360335)

### _munsell:
Database with 1269 Munsell spectral reflectance functions + Value (V), Chroma (C), hue (h) and (ab) specifications.

## 4. colortransforms.py
Module with basic colorimetric functions (xyz_to_chromaticity, chromaticity_to_xyz conversions):
### xyz_to_Yxy(), Yxy_to_xyz(): 
CIE xyz <--> CIE Yxy 
### xyz_to_Yuv(), Yuv_to_xyz(): 
CIE xyz <--> CIE 1976 Yu'v' 
### Yxy_to_Yuv(), Yuv_to_Yxy(): 
CIE 1976 Yu'v' <--> CIE Yxy 
###	 xyz_to_xyz():	
CIE xyz <--> CIE xyz (forward = inverse)
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

## 5. cct.py

### _cct_lut_dir:
Folder with Look-Up-Tables (LUT) for correlated color temperature calculation followings [Ohno's method](http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020).

### _cct_LUT:
Dict with LUT.

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

###	 cct_to_mired(): 
Converts from CCT to Mired scale (or back)

## 6. chromaticadaptation.py (cat)

### cat._xyz0:   
Default adopted white point

### cat._mcats: 
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
Check if dimensions of data and xyzw match. If xyzw.shape[0] > 1 then len(data.shape) > 2 & (data.shape[0] = xyzw.shape[0]).

### cat.normalize_mcat():  
Normalize mcat matrix to xyz0 -- > [1,1,1]

### cat.get_transfer_function():  
Calculate the chromatic adaptation diagonal matrix transfer function Dt. 
Default = 'vonkries' (others: 'rlab')

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
Applies a von kries (independent rescaling of 'sensor sensitivity' = diag. tf.) to adapt from 
current adaptation conditions (1) to the new conditions (2). 


## 7. colorappearancemodels.py (cam)
### cam._unique_hue_data: 
Database of unique hues with corresponding Hue quadratures and eccentricity factors
(ciecam02, cam16, ciecam97s, cam15u)

### cam._surround_parameters: 
Database of surround parameters c, Nc, F and FLL for ciecam02, cam16, ciecam97s and cam15u.

### cam._naka_rushton_parameters: 
Database with parameters (n, sig, scaling and noise) for the Naka-Rushton function: 
scaling * ((data^n) / ((data^n) + (sig^n))) + noise

### cam._camucs_parameters: 
Database with parameters specifying the conversion from ciecam02/cam16 to cam[x]ucs (uniform color space), cam[x]lcd (large color diff.), cam[x]scd (small color diff).

### cam._cam15u_parameters: 
Database with CAM15u model parameters.

### cam._cam_sww_2016_parameters: 
Database with cam_sww_2016 parameters (model by Smet, Webster and Whitehead published in JOSA A in 2016)

### cam._cam_default_white_point: 
Default internal reference white point (xyz)

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

### cam_sww_2016(): 
Calculates output for the principled color appearance model developped by Smet, Webster and Whitehead 
that is based on a mapping of the Munsell color system.
This function implements the JOSA A (parameters = 'JOSA') published model (with a correction for the parameter in Eq.4 of Fig. 11: 0.952 --> -0.952 and the delta_ac and delta_bc white-balance shifts in Eq. 5e & 5f should be: -0.028 & 0.821),
as well as using a set of other parameters providing a better fit (parameters = 'best fit'). 
* [K. Smet, M. Webster, and L. Whitehead, “A Simple Principled Approach for Modeling and Understanding Uniform Color Metrics,” HHS Public Access, vol. 8, no. 12, pp. 1699–1712, 2015.](https://www.osapublishing.org/josaa/abstract.cfm?URI=josaa-33-3-a319)


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
* 'xyz_to_lab_cam_sww_2016', 'lab_cam_sww_2016_to_xyz'

These functions are imported directly into the luxpy namespace. 


## 8. colortf.py
### colortf():
Calculates conversion between any two color spaces for which xyz_to_...() and ..._to_xyz() exists.


## 9. colorrenditionindices.py (cri)

### cri._cri_defaults: 
Default settings for different color rendition indices: (major dict has 9 keys (04-Jul-2017): 
* sampleset [str/dict],  ref_type [str], cieobs [str], avg [fcn handle], scale [dict], cspace [dict], catf [dict], rg_pars [dict], cri_specific_pars [dict]

Supported cri-types:
* 'ciera', 'ciera-8', 'ciera-14', 'cierf', 'iesrf', 'cri2012', 'cri2012-hl17', 'cri2012-hl1000', 'cri2012-real210', 'cqs-v7.5', 'cqs-v9.0', mcri'

### cri.linear_scale():  
Linear color rendering index scale from [CIE13.3-1974/1995](http://www.cie.co.at/index.php/index.php?i_ca_id=303):   Ri,a = 100 - c1*DEi,a. (c1 = 4.6)

### cri.log_scale(): 
Log-based color rendering index scale from [Davis & Ohno (2010)](http://spie.org/Publications/Journal/10.1117/1.3360335):  Ri,a = 10 * ln(exp((100 - c1*DEi,a)/10) + 1)

### cri.psy_scale():
Psychometric based color rendering index scale from CRI2012 ([Smet et al. 2013, LRT](http://journals.sagepub.com/doi/abs/10.1177/1477153513481375)):  Ri,a = 100 * (2 / (exp(c1*abs(DEi,a)**(c2) + 1))) ** c3

### cri.process_cri_type_input(): 
Load a cri_type dict but overwrites any keys that have a non-None input in calling function

### cri.gamut_slicer(): 
Slices the gamut in nhbins slices and provides normalization of test gamut to reference gamut.

### cri.jab_to_rg(): 
Calculates gamut area index, Rg based on hue-ordered jabt and jabr input (first element must also be last)

### cri.spd_to_jab_t_r(): 
Calculates jab color values for a sample set illuminated with test source and its reference illuminant.
(inputs are subset of spd_o_cri()) 
                  
### cri.spd_to_rg(): 
Calculates the color gamut index of data (= np.array([[wl,spds]]) (data_axis = 0) for a sample set illuminated with test source (data) with respect to some reference illuminant.
(inputs are subset of spd_o_cri())
                                          
### cri.spd_to_DEi(): 
Calculates color difference (~fidelity) of data (= np.array([[wl,spds]]) (data_axis = 0) between sample set illuminated with test source (data) and some reference illuminant.
(inputs are subset of spd_o_cri())
                             
### cri.optimize_scale_factor():
Optimize scale_factor of cri-model in cri_type such that average Ra for a set of light sources is the same as that of a target-cri (default: 'ciera').

### cri.spd_to_cri(): 
Calculates color rendition (~fidelity) index of data (= np.array([[wl,spds]]) (data_axis = 0) free choice of:
* out = output requested (e.g. 'Ra', 'Ri' or 'Ra,Ri', or 'Ra, Ri, cct', ...; default = 'Ra', 'a' stands for average --> general color rendition index, i for individual regardless of cri_type

* wl: wavelengths (or [start, end, spacing]) to interpolate the SPD's in data argument to. Default = None (no interpolation) 

* cri_type: str input specifying dict with default settings or user defined dict with parameters specifying color rendering index specifics (see e.g. luxpy.cri._cri_defaults['cierf'])
non-None input arguments to function will override defaults in cri_type dict

* cri_type keys / further function arguments:
    1. sampleset: np.array([[wl,rfl]]) or str for built-in rfl-set
    2. ref_type: reference illuminant type ('BB' : Blackbody radiatiors, 'DL': daylightphase, 'ciera': used in [CIE CRI-13.3-1995](http://www.cie.co.at/index.php/index.php?i_ca_id=303), 'cierf': used in [CIE 224-2017](http://www.cie.co.at/index.php?i_ca_id=1027), 'iesrf': used in [TM30-15](https://www.ies.org/store/technical-memoranda/ies-method-for-evaluating-light-source-color-rendition/), ...)
    3. cieobs: dict:
         * 'xyz': cie observer for calculating xyz of samples and white 
         * 'cct': cie observer for calculating cct
    4. cspace: 
         * 'type': color space used to calculate color differences
         * 'xyzw': white point of color space, (None: use xyzw of test / reference (after chromatic adaptation, if specified))
         * ' ...' : other possible parameters needed for color space calculation
    5. catf: None: don't apply a cat (other than perhaps the one built into the colorspace), 
    OR dict:
         * 'D': degree of adaptation
         * 'mcat': sensor matrix specification,
         * 'xyzw': (None: use xyzw of reference otherwise transform both test and ref to xyzw)
    6. avg: averaging function (handle) for color differences, DEi (e.g. numpy.mean, .math.rms, .math.geomean)
    7. scale
         * 'fcn': function handle to type of cri scale, e.g.  linear_scale, log_scale, psy_scale
         * 'cfactor': factors used in scaling function: if True: cfactor will be optimized to minimize the rms between the Ra's of the requested metric and some target metric specified in:
              * opt_cri_type:  str (one of the preset _cri_defaults) or dict (dict must contain all keys as normal), default = 'ciera' (if 'opt_cri_type' -key not in 'scale' dict)
              * opt_spd_set: set of light source spds used to optimize cfactor, default = 'F1-F12' (if 'opt_spd_set' -key not in 'scale' dict)
    8. opt_scale_factor: True: optimize c-factor, else do nothing and use value of cfactor in 'scale'.    
    9. cri_specific_pars: other parameters specific to type of cri, e.g. maxC for CQS calculations
    10. rg_pars: dict containing:
         * 'nhbins' (int): number of hue bins to divide the gamut in
         * 'start_hue' (float,°): hue at which to start slicing
         * 'normalize_gamut' (bool): normalize gamut or not before calculating a gamut area index Rg. 

### wrapper functions for fidelity type metrics:
* cri.spd_to_ciera()
    * [[1] CIE13.3-1995, “Method of Measuring and Specifying Colour Rendering Properties of Light Sources,” CIE, Vienna, Austria, 1995.,ISBN 978 3 900734 57 2](http://www.cie.co.at/index.php/index.php?i_ca_id=303)
* cri.spd_to_cierf()
    * [cie224:2017, CIE 2017 Colour Fidelity Index for accurate scientific use. (2017), ISBN 978-3-902842-61-9](http://www.cie.co.at/index.php?i_ca_id=1027)
* cri.spd_to_iesrf()
    * [A. David, P. T. Fini, K. W. Houser, Y. Ohno, M. P. Royer, K. A. G. Smet, M. Wei, and L. Whitehead, “Development of the IES method for evaluating the color rendition of light sources,” Opt. Express, vol. 23, no. 12, pp. 15888–15906, 2015.](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-12-15888)
    * [K. A. G. Smet, A. David, and L. Whitehead, “Why color space uniformity and sample set spectral uniformity are essential for color rendering measures,” LEUKOS, vol. 12, no. 1–2, pp. 39–50, 2016](http://www.tandfonline.com/doi/abs/10.1080/15502724.2015.1091356)
* cri.spd_to_cri2012(), cri.spd_to_cri2012_hl17(), cri.spd_to_cri2012_hl1000(), cri.spd_to_cri2012_real210
    * [K. Smet, J. Schanda, L. Whitehead, and R. Luo, “CRI2012: A proposal for updating the CIE colour rendering index,” Light. Res. Technol., vol. 45, pp. 689–709, 2013](http://journals.sagepub.com/doi/abs/10.1177/1477153513481375)

### cri.spd_to_mcri(): 
Calculates the memory color rendition index, Rm:  
* [K. A. G. Smet, W. R. Ryckaert, M. R. Pointer, G. Deconinck, and P. Hanselaer, (2012) “A memory colour quality metric for white light sources,” Energy Build., vol. 49, no. C, pp. 216–225.](http://www.sciencedirect.com/science/article/pii/S0378778812000837)

### cri.spd_to_cqs(): 
Versions 7.5 and 9.0 are supported.  
* [W. Davis and Y. Ohno, “Color quality scale,” (2010), Opt. Eng., vol. 49, no. 3, pp. 33602–33616.](http://spie.org/Publications/Journal/10.1117/1.3360335)

## 10. plotters.py

### plot_color_data():
Plot color data (local helper function)

### plotDL():
Plot daylight locus (for 'ccts', default = 4000 K to 1e19 K) for 'cieobs' in 'cspace'.

### plotBB():
Plot blackbody locus (for 'ccts', default = 4000 K to 1e19 K) for 'cieobs' in 'cspace'.

### plotSL():
Plot spectrum locus for 'cieobs' in 'cspace'. plotBB and plotDL are also called, but can be turned off.

### plotceruleanline():
Plot cerulean (yellow (577 nm) - blue (472 nm)) line (Kuehni, CRA, 2013: Table II: spectral lights).

### plotUH():
Plot unique hue line from centerpoint xyz0 (Kuehni, CRA, 2013: uY,uB,uG: Table II: spectral lights; uR: Table IV: Xiao data)

-------------------------------------------------------------------------------
## 0.1.  helpers.py 

### np2d():
Make a tupple, list or numpy array at least 2d array.

### np2dT():
Make a tupple, list or numpy array at least 2d array and tranpose.

### np3d():
Make a tupple, list or numpy array at least 3d array.

### np3dT():
Make a tupple, list or numpy array at least 3d array and tranpose (swap) first two axes.

### put_args_in_db():
Overwrites values in dict db with 'not-None' input arguments from function (obtained with built-in locals()).
See put_args_in_db? for more info.

### getdata():
Get data from csv-file or convert between pandas dataframe (kind = 'df') and numpy 2d-array (kind = 'np').

### dictkv():
Easy input of of keys and values into dict (both should be iterable lists).

### OD():
Provides a nice way to create OrderedDict "literals".

### meshblock():
Create a meshed black (similar to meshgrid, but axis = 0 is retained) to enable fast blockwise calculation.

### aplit():
Split np.array data on (default = last) axis.

### ajoin():
Join tupple of np.array data on (default = last) axis.

### broadcast_shape():
Broadcasts shapes of data to a target_shape, expand_2d_to_3d if not None and data.ndim == 2, axis0,1_repeats specify how many times data much be repeated along axis (default = same axis size).
Useful for block/vector calculation in which nupy fails to broadcast correctly.


## 0.2.  math.py 

### line_intersect():
Line intersection of two line segments a and b (Nx2) specified by their end points 1,2.
* From [https://stackoverflow.com/questions/3252194/numpy-and-line-intersections](https://stackoverflow.com/questions/3252194/numpy-and-line-intersections)

### positive_arctan():
Calculates positive angle (0°-360° or 0 - 2*pi rad.) from x and y.

### dot23():
Dot product of a (M x N) 2-d np.array with a (N x K x L) 3-d np.array using einsum().

### check_symmetric():
Checks if A is symmetric (returns bool).

### check_posdef():
Checks positive definiteness of matrix.
Returns true when input is positive-definite, via Cholesky

### symmM_to_posdefM():
Converts a symmetric matrix to a positive definite one. Two methods are supported:
* 'make': A Python/Numpy port of Muhammad Asim Mubeen's matlab function Spd_Mat.m (https://nl.mathworks.com/matlabcentral/fileexchange/45873-positive-definite-matrix)
* 'nearest': A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code. (https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite)

### bvgpdf():
Calculates bivariate Gaussian (PD) function, with center mu and shape and orientation determined by sigmainv. 

### rms():
Calculates root-mean-square along axis.

### geomean():
Calculates geometric mean along axis.

### polyarea():
Calculates area of polygon. (first coordinate should also be last)

### erf(), erfinv(): 
erf-function (and inverse), direct import from scipy.special