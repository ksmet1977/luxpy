# -*- coding: utf-8 -*-
"""
cct: Module with functions related to correlated color temperature calculations
===============================================================================
 :_CCT_MAX: (= 1e12), max. value that does not cause overflow problems. 

 :_CCT_LUT_PATH: Folder with Look-Up-Tables (LUT) for correlated color 
                 temperature calculation followings Ohno's method.

 :_CCT_LUT: Dict with LUTs.
 
 :_CCT_LUT_CALC: Boolean determining whether to force LUT calculation, even if
                 the LUT can be found in ./data/cctluts/.
                 
 :_CCT_CSPACE: default chromaticity space to calculate CCT and Duv in.
 
 :_CCT_CSPACE_KWARGS: nested dict with cspace parameters for forward and backward modes. 

 :calculate_lut(): Function that calculates the LUT for the ccts stored in 
                   ./data/cctluts/cct_lut_cctlist.dat or given as input 
                   argument. Calculation is performed for CMF set specified in
                   cieobs. Adds a new (temprorary) field to the _CCT_LUT dict.

 :calculate_luts(): Function that recalculates (and overwrites) LUTs in 
                    ./data/cctluts/ for the ccts stored in 
                    ./data/cctluts/cct_lut_cctlist.dat or given as input 
                    argument. Calculation is performed for all CMF sets listed 
                    in _CMF['types'].

 :xyz_to_cct(): | Calculates CCT, Duv from XYZ 
                | wrapper for xyz_to_cct_ohno() & xyz_to_cct_search()

 :xyz_to_duv(): | Calculates Duv, (CCT) from XYZ
                | wrapper for xyz_to_cct_ohno() & xyz_to_cct_search()

 :cct_to_xyz_fast(): Calculates xyz from CCT, Duv by estimating 
                     the line perpendicular to the planckian locus.

 :cct_to_xyz(): Calculates xyz from CCT, Duv [100 K < CCT < _CCT_MAX]

 :xyz_to_cct_mcamy(): | Calculates CCT from XYZ using Mcamy model:
                      | `McCamy, Calvin S. (April 1992). 
                        Correlated color temperature as an explicit function of 
                        chromaticity coordinates. 
                        Color Research & Application. 17 (2): 142–144. 
                        <http://onlinelibrary.wiley.com/doi/10.1002/col.5080170211/abstract>`_

 :xyz_to_cct_HA(): | Calculate CCT from XYZ using Hernández-Andrés et al. model.
                   | `Hernández-Andrés, Javier; Lee, RL; Romero, J (September 20, 1999). 
                     Calculating Correlated Color Temperatures Across the 
                     Entire Gamut of Daylight and Skylight Chromaticities. 
                     Applied Optics. 38 (27): 5703–5709. PMID 18324081. 
                     <https://www.osapublishing.org/ao/abstract.cfm?uri=ao-38-27-5703>`_

 :xyz_to_cct_ohno(): | Calculates CCT, Duv from XYZ using a LUT following:
                     | `Ohno Y. (2014)
                       Practical use and calculation of CCT and Duv. 
                       Leukos. 2014 Jan 2;10(1):47-55.
                       <http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020>`_

 :xyz_to_cct_search(): Calculates CCT, Duv from XYZ using brute-force search 
                       algorithm (between 1e2 K - _CCT_MAX K)
                       
 :cct_to_mired(): Converts from CCT to Mired scale (or back).
 
 :xyz_to_cct_ohno2011(): Calculate cct and Duv from CIE 1931 2° xyz following Ohno (CORM 2011).

===============================================================================
"""

from .cct import *
__all__ = cct.__all__

from .cctduv_ohno_CORM2011 import *
__all__ += cctduv_ohno_CORM2011.__all__




