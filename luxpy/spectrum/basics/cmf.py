# -*- coding: utf-8 -*-
########################################################################
# <LUXPY: a Python package for lighting and color science.>
# Copyright (C) <2017>  <Kevin A.G. Smet> (ksmet1977 at gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#########################################################################
"""
Module for Color Matching Functions (CMF) and Vlambda (=Ybar)
=============================================================

cmf.py
------

  :luxpy._CMF: | Dict with keys 'types' and x
               | x are dicts with keys 'bar', 'K', 'M'
 
     | * luxpy._CMF['types']  = ['1931_2','1964_10','2006_2','2006_10',
     |                           '1931_2_judd1951','1931_2_juddvos1978',
     |                           '1951_20_scotopic']
     | * luxpy._CMF[x]['bar'] = numpy array with CMFs for type x 
     |                          between 360 nm and 830 nm (has shape: (4,471))
     | * luxpy._CMF[x]['K']   = Constant converting Watt to lumen for CMF type x.
     | * luxpy._CMF[x]['M']   = XYZ to LMS conversion matrix for CMF type x.
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
            
        3. The XYZ to LMS conversion matrix for the Judd-Vos XYZ CMFs is the one
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
            to np.eye().
        
        6. _CMF[x]['M'] for x equal to '2006_2' or '2006_10' is NOT 
            normalized to illuminant E! These are the original matrices 
            as defined by [1] & [2].

    
References
----------

    1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_

    2. `CIE, and CIE (2006). 
    Fundamental Chromaticity Diagram with Physiological Axes - Part I.(Vienna: CIE).
    <http://www.cie.co.at/publications/fundamental-chromaticity-diagram-physiological-axes-part-1>`_

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""


###################################################################################################
# luxpy parameters
###################################################################################################

#--------------------------------------------------------------------------------------------------
from luxpy import math
from luxpy.utils import np
__all__ = ['_CMF']


#--------------------------------------------------------------------------------------------------
# load all cmfs and set up nested dict:
_CMF_TYPES = ['1931_2','1964_10','2006_2','2006_10','1931_2_judd1951','1931_2_juddvos1978','1951_20_scotopic','cie_std_dev_obs_f1']
_CMF_K_VALUES = [683.002, 683.599, 683.358, 683.144, 683.002, 683.002, 1700.06, 0.0] 


_CMF_M_1931_2 = np.array([     # definition of 3x3 matrices to convert from xyz to Hunt-Pointer-Estevez lms
[0.38971,0.68898,-0.07868],
[-0.22981,1.1834,0.04641],
[0.0,0.0,1.0]
 ])


_CMF_M_1931_2_JUDDVOS1978 = np.array([ # definition of 3x3 matrices to convert from Judd-Vos xyz to Smith-Pokorny lms
[0.15514,0.54312,-0.03286],
[-0.15514,0.45684,0.032386],
[0.0,0.00801,1.0]
 ])
         

_CMF_M_2006_2 = np.linalg.inv(np.array([[1.94735469, -1.41445123, 0.36476327],
                                        [0.68990272, 0.34832189, 0],
                                        [0, 0, 1.93485343]]))

_CMF_M_2006_10 = np.linalg.inv(np.array([[1.93986443, -1.34664359, 0.43044935],
                                        [0.69283932, 0.34967567, 0],
                                        [0, 0, 2.14687945]]))

# Note that for the following, no conversion has been defined, so the 2006 10° matrix is used:          
_CMF_M_1964_10 = _CMF_M_2006_10.copy()

# Note that for the following, no conversion has been defined, so the 1931 HPE matrix is used:          
_CMF_M_1931_2_JUDD1951 = _CMF_M_1931_2.copy()
         


# Scotopic conversion matrix has been set as the identity matrix (V' was replicated in the Xb,Yb,Zb columns)     
_CMF_M_1951_20_SCOTOPIC = np.eye(3)   
_CMF_M_cie_std_dev_obs_f1 = np.eye(3)   
   
_CMF_M_list = [_CMF_M_1931_2,_CMF_M_1964_10,_CMF_M_2006_2,_CMF_M_2006_10, _CMF_M_1931_2_JUDD1951, _CMF_M_1931_2_JUDDVOS1978, _CMF_M_1951_20_SCOTOPIC,_CMF_M_cie_std_dev_obs_f1]

_CMF = {'types': _CMF_TYPES}
for i, cmf_type in enumerate(_CMF_TYPES): # store all in single nested dict
    _CMF[cmf_type]  = {'bar':  []}
    _CMF[cmf_type]['K'] = _CMF_K_VALUES[i]
    _CMF[cmf_type]['M'] = _CMF_M_list[i] 



