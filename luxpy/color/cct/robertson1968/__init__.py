# -*- coding: utf-8 -*-
"""
Standalone (no luxpy required) module with (updated, 2022) Robertson1968 CCT functions
======================================================================================
 
(includes correction near slope-sign-change of iso-temperature-lines)

 :cct_to_xyz(): Calculates xyz from CCT, Duv by estimating the line perpendicular to the planckian locus (=iso-T line).

 :cct_to_xyz(): Calculates xyz from CCT, Duv [_CCT_MIN < CCT < _CCT_MAX]
 
 
References:
   1. `Robertson, A. R. (1968). 
   Computation of Correlated Color Temperature and Distribution Temperature. 
   Journal of the Optical Society of America,  58(11), 1528–1535. 
   <https://doi.org/10.1364/JOSA.58.001528>`_
   
   2. Smet K.A.G., Royer M., Baxter D., Bretschneider E., Esposito E., Houser K., Luedtke W., Man K., Ohno Y. (2022),
   Recommended method for determining the correlated color temperature and distance from the Planckian Locus of a light source
   (in preparation, LEUKOS?)
   
   3. Baxter D., Royer M., Smet K.A.G. (2022)
   Modifications of the Robertson Method for Calculating Correlated Color Temperature to Improve Accuracy and Speed
   (in preparation, LEUKOS?)
   
===============================================================================
"""

from .robertson1968 import *
__all__ = robertson1968.__all__





