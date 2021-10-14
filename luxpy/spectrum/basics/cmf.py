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
         
        1. All CMF functions are extrapolated using the CIE recommended method 
            to a full 360-830 range. See luxpy.cie_interp for more info on the default
            extrapolation method used.

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
from luxpy.utils import np
__all__ = ['_CMF']


#--------------------------------------------------------------------------------------------------
# load cmf data in nested _CMF dict:
_CMF_TYPES = ['1931_2','1964_10','2006_2','2006_10','2015_2','2015_10','1931_2_judd1951','1931_2_juddvos1978','1951_20_scotopic','cie_std_dev_obs_f1']
_CMF_K_VALUES = [683.002, 683.599, 683.358, 683.144, 683.358, 683.144, 683.002, 683.002, 1700.06, 0.0] 
_CMF = {'types': _CMF_TYPES}

#------------------------------------------------------------------------------
# CIE 1931 2°:
_CMF_M_1931_2 = np.array([     # M : definition of 3x3 matrices to convert from xyz to Hunt-Pointer-Estevez lms
[0.38971,0.68898,-0.07868],
[-0.22981,1.1834,0.04641],
[0.0,0.0,1.0]
 ])

_CMF_N_1931_2 = np.array([     # N : definition of 3x3 matrices to convert from rgb to xyz 
[2.76888,1.75175,1.13016],
[1.00000,4.59070,0.06010],
[0.00000,0.05651,5.59427]
 ])
_CMF_N_1931_2 = np.linalg.inv(_CMF_N_1931_2) # store as xyz to rgb !



#------------------------------------------------------------------------------
# 1951 Judd 2°:
# Note that for the following, no conversion has been defined, 
# so the 1931 matrices are used as only small changes have been made (to Vl below 460 nm):          
_CMF_M_1931_2_JUDD1951 = _CMF_M_1931_2.copy() # store as xyz to rgb !
_CMF_N_1931_2_JUDD1951 = _CMF_N_1931_2.copy() # store as xyz to rgb !   


#------------------------------------------------------------------------------
# 1978 Judd Vos 2°:
_CMF_M_1931_2_JUDDVOS1978 = np.array([ # definition of 3x3 matrices to convert from Judd-Vos xyz to Smith-Pokorny lms (from Smith-Pokorny 1975)
[0.15514,0.54312,-0.03286],
[-0.15514,0.45684,0.032386],
[0.0,0.00801,1.0]
 ])

_CMF_M_1931_2_JUDDVOS1978_Vos78 = np.array([ # definition of 3x3 matrices to convert from Judd-Vos xyz to lms (from Vos 1978)
[0.15514,0.54312,-0.0370161],
[-0.15514,0.45684, 0.0296946],
[0.0,0.00801,0.0073215]
 ])
_CMF_N_1931_2_JUDDVOS1978 = _CMF_N_1931_2.copy() # not defined, so take the one for CIE 1931 2°



#------------------------------------------------------------------------------
# 2015 CMFs based on 2006 cone fundamentals (2°):
_CMF_M_2006_2 = np.linalg.inv(np.array([[1.94735469, -1.41445123, 0.36476327], # from CIE15:2018
                                        [0.68990272, 0.34832189, 0],
                                        [0, 0, 1.93485343]]))

_CMF_N_2006_2 = np.ones((3,3))*np.nan # N : definition of 3x3 matrices to convert from rgb to xyz (not defined, so NaNs)


#------------------------------------------------------------------------------
# 2015 CMFs based on 2006 cone fundamentals (10°):
_CMF_M_2006_10 = np.linalg.inv(np.array([[1.93986443, -1.34664359, 0.43044935], # from CIE15:2018
                                        [0.69283932, 0.34967567, 0],
                                        [0, 0, 2.14687945]]))

_CMF_N_2006_10 = np.array([ # N : definition of 3x3 matrices to convert from rgb to xyz (calculated from published xyz-to-lms and rgb-to-lms matrices, CIE TC1-36)
[3.161850764,-0.698441888,-0.572538921],
[-0.522270801,1.29543215,0.046547295],
[0.005536941,-0.01373374,0.469326311]
 ])



#------------------------------------------------------------------------------
# CIE 1964 10°:
# Note that for the following, no conversion has been defined, 
# so the 2006 10° matrix is used, as the two CMFs are very similar:          
_CMF_M_1964_10 = _CMF_M_2006_10.copy()

_CMF_N_1964_10 = np.array([     # N : definition of 3x3 matrices to convert from rgb to xyz 
[0.341080,0.189145,0.387529],
[0.139058,0.837460,0.073160],
[0.00000,0.039553,1.026200]
 ])
_CMF_N_1964_10 = np.linalg.inv(_CMF_N_1964_10) # store as xyz to rgb !


#------------------------------------------------------------------------------
# CIE 1951 20° scotopic observer:
# Scotopic conversion matrix has been filled with NaNs (V' was replicated in the Xb,Yb,Zb columns)     
_CMF_M_1951_20_SCOTOPIC = np.ones((3,3))*np.nan   # xyz to lms
_CMF_N_1951_20_SCOTOPIC = np.ones((3,3))*np.nan   # xyz to rgb
_CMF_M_cie_std_dev_obs_f1 = np.ones((3,3))*np.nan # xyz to lms
_CMF_N_cie_std_dev_obs_f1 = np.ones((3,3))*np.nan # xyz to rgb
   

#------------------------------------------------------------------------------
# Note that _CMF_..2006_.. occurs twice as these are the same as the 2015 XYZ-CMFs
_CMF_M_list = [_CMF_M_1931_2,_CMF_M_1964_10,_CMF_M_2006_2,_CMF_M_2006_10, _CMF_M_2006_2,_CMF_M_2006_10,_CMF_M_1931_2_JUDD1951, _CMF_M_1931_2_JUDDVOS1978, _CMF_M_1951_20_SCOTOPIC,_CMF_M_cie_std_dev_obs_f1]
_CMF_N_list = [_CMF_N_1931_2,_CMF_N_1964_10,_CMF_N_2006_2,_CMF_N_2006_10, _CMF_N_2006_2,_CMF_N_2006_10,_CMF_N_1931_2_JUDD1951, _CMF_N_1931_2_JUDDVOS1978, _CMF_N_1951_20_SCOTOPIC,_CMF_N_cie_std_dev_obs_f1]


_CMF = {'types': _CMF_TYPES}
for i, cmf_type in enumerate(_CMF_TYPES): # store all in single nested dict
    _CMF[cmf_type]  = {'bar':  []}
    _CMF[cmf_type]['K'] = _CMF_K_VALUES[i]
    _CMF[cmf_type]['M'] = _CMF_M_list[i] 
    _CMF[cmf_type]['N'] = _CMF_N_list[i] 



