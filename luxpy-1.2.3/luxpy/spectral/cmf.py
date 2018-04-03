# -*- coding: utf-8 -*-
"""
luxpy module loading sets of color matching functions.

 * luxpy._CMF: Dict with keys 'types', 'bar', 'K', 'M'
 
     + luxpy._CMF['types'] = ['1931_2','1964_10','2006_2','2006_10']
     + luxpy._CMF['bar'] = Dict with CMFs for each of the types between 360 to 830 nm
                             keys are cmf_types, values are CMF numpy arrays with shape: (4,471)
     + luxpy._CMF['K'] = Dict with constants converting Watt to lumen for specified CMF   
                            keys are cmf_types, values are scalars
     + luxpy._CMF['M'] = Dict with XYZ to LMS conversion matrices
                            keys are cmf_types, values are numpy arrays with shape: (3,3)
                            
Created on Sat Jun 17 10:43:24 2017

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
###################################################################################################
# luxpy parameters
###################################################################################################

#--------------------------------------------------------------------------------------------------
from luxpy import np, odict
__all__ = ['_CMF']



#--------------------------------------------------------------------------------------------------
# load all cmfs and set up nested dict:
_CMF_TYPES = ['1931_2','1964_10','2006_2','2006_10']

def _dictkv(keys=None,values=None, ordered = True): 
    # Easy input of of keys and values into dict (both should be iterable lists)
    if ordered is True:
        return odict(zip(keys,values))
    else:
        return dict(zip(keys,values))

_CMF_K = _dictkv(keys = _CMF_TYPES, values = [683.0,683.6,683.0,683.0],ordered = True) # K-factors for calculating absolute tristimulus values

_CMF_M_1931_2=np.array([     # definition of 3x3 matrices to convert from xyz to lms
[0.38971,0.68898,-0.07868],
[-0.22981,1.1834,0.04641],
[0.0,0.0,1.0]
 ])
_CMF_M_1964_10=np.array([
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
_CMF_M = _dictkv(keys = _CMF_TYPES, values= [_CMF_M_1931_2,_CMF_M_1964_10,_CMF_M_2006_2,_CMF_M_2006_10],ordered = True)

_CMF = {'types': _CMF_TYPES,'bar': [], 'K': _CMF_K, 'M':_CMF_M} # store all in single nested dict
			


