# -*- coding: utf-8 -*-
"""
cct: Module with functions related to correlated color temperature calculations
===============================================================================
 These methods supersede earlier methods in cct_legacy.y (prior to Nov 2021)

 :_CCT_MAX: (= 1e11 K), max. value that does not cause overflow problems. 
 
 :_CCT_MIN: (= 550 K), min. value that does not cause underflow problems.
 
 :_CCT_FALLBACK_N: Number of intervals to divide an ndarray with CCTs.
 
 :_CCT_FALLBACK_UNIT: Type of scale (units) an ndarray will be subdivided.

 :_CCT_LUT_PATH: Folder with Look-Up-Tables (LUT) for correlated color 
                 temperature calculations. 

 :_CCT_LUT: Dict with pre-calculated LUTs with structure LUT[mode][cspace][cieobs][lut i].
 
 :_CCT_LUT_CALC: Boolean determining whether to force LUT calculation, even if
                 the LUT.pkl files can be found in ./data/cctluts/.
 
 :_CCT_LUT_RESOLUTION_REDUCTION_FACTOR: number of subdivisions when performing
                                        a cascading lut calculation to zoom-in 
                                        progressively on the CCT (until a certain 
                                        tolerance is met)
                 
 :_CCT_CSPACE: default chromaticity space to calculate CCT and Duv in.
 
 :_CCT_CSPACE_KWARGS: nested dict with cspace parameters for forward and backward modes. 
 
 :get_tcs4(): Get an ndarray of Tc's obtained from a list or tuple of tc4 4-vectors.
 
 :calculate_lut(): Function that calculates the LUT for the input ccts.
 
 :generate_luts(): Generate a number of luts and store them in a nested dictionary.
                    (Structure: lut[cspace][cieobs][lut type])

 :xyz_to_cct(): Calculates CCT, Duv from XYZ (wraps a variety of methods)

 :xyz_to_duv(): Calculates Duv, (CCT) from XYZ (wrapper around xyz_to_cct, but with Duv output.)
                
 :cct_to_xyz(): Calculates xyz from CCT, Duv by estimating the line perpendicular to the planckian locus (=iso-T line).

 :cct_to_xyz(): Calculates xyz from CCT, Duv [_CCT_MIN < CCT < _CCT_MAX]

 :xyz_to_cct_mcamy1992(): | Calculates CCT from XYZ using Mcamy model:
                          | `McCamy, Calvin S. (April 1992). 
                            Correlated color temperature as an explicit function of 
                            chromaticity coordinates. 
                            Color Research & Application. 17 (2): 142–144. 
                            <http://onlinelibrary.wiley.com/doi/10.1002/col.5080170211/abstract>`_

 :xyz_to_cct_hernandez1999(): | Calculate CCT from XYZ using Hernández-Andrés et al. model.
                              | `Hernández-Andrés, Javier; Lee, RL; Romero, J (September 20, 1999). 
                                Calculating Correlated Color Temperatures Across the 
                                Entire Gamut of Daylight and Skylight Chromaticities. 
                                Applied Optics. 38 (27): 5703–5709. PMID 18324081. 
                                <https://www.osapublishing.org/ao/abstract.cfm?uri=ao-38-27-5703>`_

 :xyz_to_cct_ohno2014(): | Calculates CCT, Duv from XYZ using a Ohno's 2014 LUT method.
                         | `Ohno Y. (2014)
                           Practical use and calculation of CCT and Duv. 
                           Leukos. 2014 Jan 2;10(1):47-55.
                           <http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020>`_
                       
 :xyz_to_cct_zhang2019():  | Calculates CCT, Duv from XYZ using Zhang's 2019 golden-ratio search algorithm
                           | `Zhang, F. (2019). 
                              High-accuracy method for calculating correlated color temperature with 
                              a lookup table based on golden section search. 
                              Optik, 193, 163018. 
                              <https://doi.org/https://doi.org/10.1016/j.ijleo.2019.163018>`_
                 
 :xyz_to_cct_robertson1968(): | Calculates CCT, Duv from XYZ using a Robertson's 1968 search method.
                              | `Robertson, A. R. (1968). 
                                Computation of Correlated Color Temperature and Distribution Temperature. 
                                Journal of the Optical Society of America,  58(11), 1528–1535. 
                                <https://doi.org/10.1364/JOSA.58.001528>`_
  
 :xyz_to_cct_li2016(): | Calculates CCT, Duv from XYZ using a Li's 2019 Newton-Raphson method.
                       | `Li, C., Cui, G., Melgosa, M., Ruan, X., Zhang, Y., Ma, L., Xiao, K., & Luo, M. R. (2016).
                         Accurate method for computing correlated color temperature. 
                         Optics Express, 24(13), 14066–14078. 
                         <https://doi.org/10.1364/OE.24.014066>`_                        
                                
 :xyz_to_cct_fibonacci(): | Calculates CCT, Duv from XYZ using a Fibonacci search method.
                  
 :cct_to_mired(): Converts from CCT to Mired scale (or back).
 
 :xyz_to_cct_ohno2011(): Calculate cct and Duv from CIE 1931 2° xyz following Ohno (CORM 2011).

 :cct_legacy: module with old (pre Nov 2021 cct conversion functions)
 
===============================================================================
"""

from .cct import *
__all__ = cct.__all__

from .cctduv_ohno_CORM2011 import *
__all__ += cctduv_ohno_CORM2011.__all__

from luxpy.color.cct import cct_legacy  
__all__ += ['cct_legacy']




