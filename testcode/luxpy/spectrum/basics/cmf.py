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

    
References
----------

    1. `CIE15-2004 (2004). 
    Colorimetry 
    (Vienna, Austria: CIE) 
    <http://www.cie.co.at/index.php/index.php?i_ca_id=304>`_

    2. `CIE, and CIE (2006). 
    Fundamental Chromaticity Diagram with Physiological Axes - Part I.(Vienna: CIE).
    <http://www.cie.co.at/publications/fundamental-chromaticity-diagram-physiological-axes-part-1>`_

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""


###################################################################################################
# luxpy parameters
###################################################################################################

#--------------------------------------------------------------------------------------------------
from luxpy import np, odict
__all__ = ['_CMF']



#--------------------------------------------------------------------------------------------------
# load all cmfs and set up nested dict:
_CMF_TYPES = ['1931_2','1964_10','2006_2','2006_10','1931_2_judd1951','1931_2_juddvos1978','1951_20_scotopic','cie_std_dev_obs_f1']
_CMF_K_VALUES = [683.002, 683.6, 683.002, 683.002, 683.002, 683.002, 1700.06, 0.0] 

#def _dictkv(keys=None,values=None, ordered = True): 
#    # Easy input of of keys and values into dict (both should be iterable lists)
#    if ordered is True:
#        return odict(zip(keys,values))
#    else:
#        return dict(zip(keys,values))


_CMF_M_1931_2=np.array([     # definition of 3x3 matrices to convert from xyz to lms
[0.38971,0.68898,-0.07868],
[-0.22981,1.1834,0.04641],
[0.0,0.0,1.0]
 ])
_CMF_M_2006_2=np.array([
[0.21057582,0.85509764,-0.039698265],
[-0.41707637,1.1772611,0.078628251],
[0.0,0.0,0.51683501]
])
_CMF_M_2006_10=np.array([
[0.21701045,0.83573367,-0.043510597],
[-0.42997951,1.2038895,0.086210895],
[0.0,0.0,0.46579234]
])
    
# Note that for the following, no conversion has been defined, so the 1931 HPE matrix is used:    
_CMF_M_1964_10=np.array([
[0.38971,0.68898,-0.07868],
[-0.22981,1.1834,0.04641],
[0.0,0.0,1.0]
]) 
_CMF_M_1931_2_JUDD1951=np.array([
[0.38971,0.68898,-0.07868],
[-0.22981,1.1834,0.04641],
[0.0,0.0,1.0]
]) 
_CMF_M_1931_2_JUDDVOS1978=np.array([
[0.38971,0.68898,-0.07868],
[-0.22981,1.1834,0.04641],
[0.0,0.0,1.0]
]) 

# Scotopic conversion matrix has been set as the identity matrix (V' was replicated in the Xb,Yb,Zb columns)     
_CMF_M_1951_20_SCOTOPIC = np.eye(3)   

_CMF_M_cie_std_dev_obs_f1 = np.eye(3)   
   

_CMF_M_list = [_CMF_M_1931_2,_CMF_M_1964_10,_CMF_M_2006_2,_CMF_M_2006_10, _CMF_M_1931_2_JUDD1951, _CMF_M_1931_2_JUDDVOS1978, _CMF_M_1951_20_SCOTOPIC,_CMF_M_cie_std_dev_obs_f1]


#_CMF_K = _dictkv(keys = _CMF_TYPES, values = _CMF_K_VALUES, ordered = True) # K-factors for calculating absolute tristimulus values
 
#_CMF_M = _dictkv(keys = _CMF_TYPES, values= _CMF_M_list, ordered = True)

_CMF = {'types': _CMF_TYPES}
for i, cmf_type in enumerate(_CMF_TYPES): # store all in single nested dict
    _CMF[cmf_type]  = {'bar':  []}
    _CMF[cmf_type]['K'] = _CMF_K_VALUES[i]
    _CMF[cmf_type]['M'] = _CMF_M_list[i] 
			


