# -*- coding: utf-8 -*-
"""

cam: sub-package with color appearance models
=============================================

 :_UNIQUE_HUE_DATA: database of unique hues with corresponding 
                             Hue quadratures and eccentricity factors 
                             for ciecam02, cam16, ciecam97s, cam15u)

 :_SURROUND_PARAMETERS: database of surround param. c, Nc, F and FLL 
                                 for ciecam02, cam16, ciecam97s and cam15u.

 :_NAKA_RUSHTON_PARAMETERS: | database with parameters 
                                       (n, sig, scaling and noise) 
                                       for the Naka-Rushton function: 
                                     | scaling * ((data**n) / ((data**n) + (sig**n))) + noise

 :_CAM_02_X_UCS_PARAMETERS: | database with parameters specifying the conversion 
                              from ciecam02/cam16 to:
                            |    cam[x]ucs (uniform color space), 
                            |    cam[x]lcd (large color diff.), 
                            |    cam[x]scd (small color diff).
                            
 :_CAM15U_PARAMETERS: database with CAM15u model parameters.
 
 :_CAM_SWW16_PARAMETERS: cam_sww16 model parameters.

 :_CAM_DEFAULT_WHITE_POINT: Default internal reference white point (xyz)

 :_CAM_DEFAULT_TYPE: Default CAM type str specifier.

 :_CAM_DEFAULT_MCAT: Default MCAT specifier.

 :_CAM_02_X_DEFAULT_CONDITIONS: Default CAM model parameters for model 
                                in cam._CAM_DEFAULT_TYPE

 :_CAM_AXES: dict with list[str,str,str] containing axis labels 
                  of defined cspaces.

 :naka_rushton(): applies a Naka-Rushton function to the input
 
 :hue_angle(): calculates a positive hue angle

 :hue_quadrature(): calculates the Hue quadrature from the hue.

 :cam_structure_ciecam02_cam16(): | basic structure of ciecam02 and cam16 models.
                                  | Has 'forward' (xyz --> color attributes) 
                                    and 'inverse' (color attributes --> xyz) modes.

 :ciecam02(): | calculates ciecam02 output 
              | (wrapper for cam_structure_ciecam02_cam16 with specifics 
                of ciecam02): 
              | `N. Moroney, M. D. Fairchild, R. W. G. Hunt, C. Li, M. R. Luo, and T. Newman, 
                “The CIECAM02 color appearance model,” 
                IS&T/SID Tenth Color Imaging Conference. p. 23, 2002. <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_

 :cam16(): | calculates cam16 output 
           | (wrapper for cam_structure_ciecam02_cam16 with specifics 
             of cam16):  
           | `C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, 
             “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” 
             Color Res. Appl., p. n/a–n/a. <http://onlinelibrary.wiley.com/doi/10.1002/col.22131/abstract>`_

 :camucs_structure(): basic structure to go to ucs, lcd and scd color spaces 
                      (forward + inverse available)

 :cam02ucs(): | calculates ucs (or lcd, scd) output based on ciecam02 
                (forward + inverse available)
              | `M. R. Luo, G. Cui, and C. Li, 
                “Uniform colour spaces based on CIECAM02 colour appearance model,” 
                Color Res. Appl., vol. 31, no. 4, pp. 320–330, 2006.
                <http://onlinelibrary.wiley.com/doi/10.1002/col.20227/abstract>`_

 :cam16ucs(): | calculates ucs (or lcd, scd) output based on cam16 
                (forward + inverse available)
              | `C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, 
                “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” 
                Color Res. Appl., p. n/a–n/a. <http://onlinelibrary.wiley.com/doi/10.1002/col.22131/abstract>`_

 :cam15u(): | calculates the output for the CAM15u model for self-luminous unrelated stimuli. 
            | `M. Withouck, K. A. G. Smet, W. R. Ryckaert, and P. Hanselaer, 
              “Experimental driven modelling of the color appearance of 
              unrelated self-luminous stimuli: CAM15u,” 
              Opt. Express, vol. 23, no. 9, pp. 12045–12064, 2015.
              <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-9-12045&origin=search>`_
            | `M. Withouck, K. A. G. Smet, and P. Hanselaer, (2015), 
              “Brightness prediction of different sized unrelated self-luminous stimuli,” 
              Opt. Express, vol. 23, no. 10, pp. 13455–13466. 
              <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-10-13455&origin=search>`_
            
 :cam_sww16(): A simple principled color appearance model based on a mapping 
               of the Munsell color system.

 :wrappers:
      | 'xyz_to_jabM_ciecam02', 'jabM_ciecam02_to_xyz',
      | 'xyz_to_jabC_ciecam02', 'jabC_ciecam02_to_xyz',
      | 'xyz_to_jabM_cam16', 'jabM_cam16_to_xyz',
      | 'xyz_to_jabC_cam16', 'jabC_cam16_to_xyz',
      | 'xyz_to_jab_cam02ucs', 'jab_cam02ucs_to_xyz', 
      | 'xyz_to_jab_cam02lcd', 'jab_cam02lcd_to_xyz',
      | 'xyz_to_jab_cam02scd', 'jab_cam02scd_to_xyz', 
      | 'xyz_to_jab_cam16ucs', 'jab_cam16ucs_to_xyz',
      | 'xyz_to_jab_cam16lcd', 'jab_cam16lcd_to_xyz',
      | 'xyz_to_jab_cam16scd', 'jab_cam16scd_to_xyz',
      | 'xyz_to_qabW_cam15u', 'qabW_cam15u_to_xyz',
      | 'xyz_to_lAb_cam_sww16', 'lab_cam_sww16_to_xyz'



.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from .colorappearancemodels import *
__all__ = colorappearancemodels.__all__