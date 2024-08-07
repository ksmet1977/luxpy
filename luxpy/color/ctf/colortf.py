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
###################################################
 Extension of basic colorimetry module
###################################################
 
Global internal variables:
    
 :_COLORTF_DEFAULT_WHITE_POINT: ndarray with XYZ values of default white point 
                                (equi-energy white) for color transformation 
                                if none is supplied.

Functions:

 :colortf(): Calculates conversion between any two color spaces ('cspace')
              for which functions xyz_to_cspace() and cspace_to_xyz() are defined.

===============================================================================
"""
import numpy as np

from luxpy import *

__all__ = ['_COLORTF_DEFAULT_WHITE_POINT','colortf']


_COLORTF_DEFAULT_WHITE_POINT = np.array([100.0, 100.0, 100.0]) # ill. E white point

#------------------------------------------------------------------------------------------------
def colortf(data, tf = _CSPACE, fwtf = {}, bwtf = {}, **kwargs):
    """
    Wrapper function to perform various color transformations.
    
    Args:
        :data: 
            | ndarray
        :tf: 
            | _CSPACE or str specifying transform type, optional
            |     E.g. tf = 'spd>xyz' or 'spd>Yuv' or 'Yuv>cct' 
            |      or 'Yuv' or 'Yxy' or ...
            |  If tf is for example 'Yuv', it is assumed to be a transformation 
            |  of type: 'xyz>Yuv'
        :fwtf: 
            | dict with parameters (keys) and values required 
            | by some color transformations for the forward transform: 
            |  i.e. 'xyz>...'
        :bwtf:
            | dict with parameters (keys) and values required 
            | by some color transformations for the backward transform: 
            |  i.e. '...>xyz'

    Returns:
        :returns: 
            | ndarray with data transformed to new color space
        
    Note:
        For the forward transform ('xyz>...'), one can input the keyword 
        arguments specifying the transform parameters directly without having 
        to use the dict :fwtf: (should be empty!) 
        [i.e. kwargs overwrites empty fwtf dict]
    """
    tf = tf.split('>')
    if len(tf) == 1:
        if not bool(fwtf):
            fwtf = kwargs
        return globals()['{}_to_{}'.format('xyz', tf[0])](data,**fwtf)
    else:
        if not bool(fwtf):
            fwtf = kwargs
        bwfcn = globals()['{}_to_{}'.format(tf[0], 'xyz')]
        fwfcn = globals()['{}_to_{}'.format('xyz', tf[1])]
        return fwfcn(bwfcn(data,**bwtf),**fwtf)   
