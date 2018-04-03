# -*- coding: utf-8 -*-
"""
###############################################################################
# _COLORTF_DEFAULT_WHITE_POINT: numpy.ndarray with XYZ values of default white point 
#                               (equi-energy white) for color transformation if none is supplied.
#
# colortf(): Calculates conversion between any two color spaces 
#            for which functions xyz_to_...() and ..._to_xyz() are defined.
###############################################################################

Created on Fri Jun 30 18:34:34 2017

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import *
__all__ = ['_COLORTF_DEFAULT_WHITE_POINT','colortf']


_COLORTF_DEFAULT_WHITE_POINT = np.array([100.0, 100.0, 100.0]) # ill. E white point

#------------------------------------------------------------------------------------------------
def colortf(data, tf = 'Yuv>Yxy', tfa0 = {}, tfa1 = {}):
    """
    Wrapper function to perform various color transformations.
    
    Args:
        :data: numpy.ndarray
        :tf: str specifying transform type, optional
            E.g. tf = 'spd>xyz' or 'spd>Yuv' or 'Yuv>cct' or ...
            If tf is for example 'Yuv' it is assumed to be a transformation of type: 'xyz>Yuv'
        :tfa0: dict with parameters (keys) and values required by some color transformations ('...>xyz')
        :tfa1: dict with parameters (keys) and values required by some color transformations ('xyz>...')

    Returns:
        :returns: numpy.ndarray with data transformed to new color space
    """
    data = np2d(data)
    tf = tf.split('>')
    if len(tf)>1:
        for ii in range(len(tf)):    
            if (ii%2 == 1):
                out_ = tf[ii]
                in_ = 'xyz'
            else:
                out_ = 'xyz'
                in_ = tf[ii]
            data = eval('{}_to_{}(data,**tfa{})'.format(in_,out_,ii))
	
    else:
        data = eval('{}_to_{}(data,**tfa0)'.format('xyz',tf[0]))    
    return data   