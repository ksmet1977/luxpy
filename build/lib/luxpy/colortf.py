# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:34:34 2017

@author: kevin.smet
"""
###############################################################################
# Calculates conversion between any two color spaces 
# for which xyz_to_...() and ..._to_xyz() exists.
###############################################################################


from luxpy import *
__all__ = ['_colortf_default_white_point','colortf']


_colortf_default_white_point = np.array([100.0, 100.0, 100.0]) # ill. E white point

#------------------------------------------------------------------------------------------------
def colortf(data, tf = 'Yuv>Yxy', tfa0 = {}, tfa1 = {}):
    """
    Wrapper function to perform various color transformations, e.g. tf = 'spd>xyz', 'spd>Yuv', 'Yuv>cct',...
    Provide additional keyword arguments for the '..>xyz' and 'xyz>..' transformations as dicts in respectively, tfa1 & tfa2 
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