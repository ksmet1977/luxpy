# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 10:43:24 2017

@author: kevin.smet
"""
###################################################################################################
# luxpy parameters
###################################################################################################

#--------------------------------------------------------------------------------------------------
from luxpy import *
__all__ = ['_cmf']



#--------------------------------------------------------------------------------------------------
# load all cmfs and set up nested dict:
_cmf_types = ['1931_2','1964_10','2006_2','2006_10']
def dictkv(keys=None,values=None, ordered = True): 
    # Easy input of of keys and values into dict (both should be iterable lists)
    if ordered is True:
        return odict(zip(keys,values))
    else:
        return dict(zip(keys,values))

_cmf_K = dictkv(keys = _cmf_types, values = [683.0,683.6,683.0,683.0],ordered = True) # K-factors for calculating absolute tristimulus values

_cmf_M_1931_2=np.array([     # definition of 3x3 matrices to convert from xyz to lms
[0.38971,0.68898,-0.07868],
[-0.22981,1.1834,0.04641],
[0.0,0.0,1.0]
 ])
_cmf_M_1964_10=np.array([
[0.38971,0.68898,-0.07868],
[-0.22981,1.1834,0.04641],
[0.0,0.0,1.0]
])
_cmf_M_2006_2=np.array([
[0.21057582,0.85509764,-0.039698265],
[-0.41707637,1.1772611,0.078628251],
[0.0,0.0,0.51683501]
])
_cmf_M_2006_10=np.array([
[0.21701045,0.83573367,-0.043510597],
[-0.42997951,1.2038895,0.086210895],
[0.0,0.0,0.46579234]
])
_cmf_M = dictkv(keys = _cmf_types, values= [_cmf_M_1931_2,_cmf_M_1964_10,_cmf_M_2006_2,_cmf_M_2006_10],ordered = True)

_cmf = {'types': _cmf_types,'bar': [], 'K': _cmf_K, 'M':_cmf_M} # store all in single nested dict
			


