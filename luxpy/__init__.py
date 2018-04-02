# -*- coding: utf-8 -*-
"""
Package for color science, colorimetric and color appearance calculations.

Loads the following luxpy modules:
    
    0.1.  helpers.py (imported directly into luxpy namespace, details see end of this file)
    0.2.  math.py (imported as math into the luxpy namespace, details see end of this file)
    
    1.  cmf.py
    2.  spectral.py
    3.  spectral_databases
    4.  colortransforms.py (imported directly into luxpy namespace)
    5.  cct.py (imported directly into luxpy namespace)
    6.  chromaticadaptation.py (imported in luxpy namespace as .cat)
    7.  colorappearancemodels.py (imported in luxpy namespace as .cam)
    8.  colortf.py (imported directly into luxpy namespace)
    9.  colorrenditionindices.py (imported in luxpy namespace as .cri)
    10. plotters.py (imported directly into luxpy namespace)

Loads the following global default constants:
    
 * _PKG_PATH (absolute path to luxpy package)
 * _SEP (operating system operator)
 * _EPS = 7./3 - 4./3 -1 (machine epsilon)
 * _CIEOBS = '1931_2' (default CIE observer color matching function)
 * _CSPACE = 'Yuv' (default color space / chromaticity diagram)
 
Note. In luxpy 'global constants' start with '_'

Created on Sat Jun 17 15:44:10 2017

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
###################################################################################################
# Set up basic luxpy parameters
###################################################################################################

#--------------------------------------------------------------------------------------------------
# module imports for use across luxpy:
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize
import warnings
import os
from collections import OrderedDict as odict
import matplotlib.pyplot as plt

__all__ = ['plt','np','pd','os','warnings','interpolate','minimize','odict']



#--------------------------------------------------------------------------------------------------
# os related:
_PKG_PATH = os.path.dirname(__file__)  # Get absolute path to package: 
_SEP = os.sep # operating system separator
__all__ += ['_PKG_PATH','_SEP']

#--------------------------------------------------------------------------------------------------
# set some general parameters:
_EPS = 7./3 - 4./3 -1 # get machine epsilon
__all__+=['_EPS']

#--------------------------------------------------------------------------------------------------
# set default colorimetric observer
_CIEOBS = '1931_2' #(CMF selection)
_CSPACE = 'Yuv'
__all__+=['_CIEOBS','_CSPACE']




#--------------------------------------------------------------------------------------------------
# Load luxpy specific modules:

# Load helper module:
from . helpers import *
__all__ += helpers.__all__

# Load math module:
from luxpy import math  
__all__ += ['math']  

# Load cmfs, part 1 (prepare basic data dict, no actual cmfs)
from . cmf import *
__all__ += cmf.__all__
#
# Load spectral module:
from . spectral import *
__all__ += spectral.__all__

## Set xyzbar in _CMF dict:
_CMF['bar'] = {_CMF['types'][i] : (xyzbar(cieobs = _CMF['types'][i], scr = 'file', kind = 'np')) for i in range(len(_CMF['types']))}

# load spd and rfl data in /spd/:
from . spectral_databases import _R_PATH, _S_PATH, _CIE_ILLUMINANTS, _IESTM30, _CRI_RFL, _MUNSELL
__all__ += ['_R_PATH', '_S_PATH', '_CRI_RFL', '_CIE_ILLUMINANTS', '_IESTM30','_MUNSELL'] 

# Load color/chromaticty transforms module:
from . colortransforms import *
__all__ += colortransforms.__all__

# Load correlated color temperature module:
from . cct import *
__all__ += cct.__all__

# Load chromatic adaptation module:
from luxpy import chromaticadaptation as cat
__all__ += ['cat']  

# Load color appearance model functions:
from luxpy import colorappearancemodels as cam
__all__ += ['cam']  

# load cam wrapper functions for use with colortf() from .colortransforms module:
from luxpy.colorappearancemodels import xyz_to_jabM_ciecam02, jabM_ciecam02_to_xyz, xyz_to_jabC_ciecam02, jabC_ciecam02_to_xyz,xyz_to_jabM_cam16, jabM_cam16_to_xyz, xyz_to_jabC_cam16, jabC_cam16_to_xyz, xyz_to_jab_cam02ucs, jab_cam02ucs_to_xyz, xyz_to_jab_cam02lcd, jab_cam02lcd_to_xyz,xyz_to_jab_cam02scd, jab_cam02scd_to_xyz, xyz_to_jab_cam16ucs, jab_cam16ucs_to_xyz, xyz_to_jab_cam16lcd, jab_cam16lcd_to_xyz,xyz_to_jab_cam16scd, jab_cam16scd_to_xyz, xyz_to_qabW_cam15u, qabW_cam15u_to_xyz, xyz_to_lab_cam_sww_2016, lab_cam_sww_2016_to_xyz

__all__ += ['xyz_to_jabM_ciecam02', 'jabM_ciecam02_to_xyz', 'xyz_to_jabC_ciecam02', 'jabC_ciecam02_to_xyz',
            'xyz_to_jabM_cam16', 'jabM_cam16_to_xyz', 'xyz_to_jabC_cam16', 'jabC_cam16_to_xyz',
            'xyz_to_jab_cam02ucs', 'jab_cam02ucs_to_xyz', 'xyz_to_jab_cam02lcd', 'jab_cam02lcd_to_xyz','xyz_to_jab_cam02scd', 'jab_cam02scd_to_xyz', 
            'xyz_to_jab_cam16ucs', 'jab_cam16ucs_to_xyz', 'xyz_to_jab_cam16lcd', 'jab_cam16lcd_to_xyz','xyz_to_jab_cam16scd', 'jab_cam16scd_to_xyz', 
            'xyz_to_qabW_cam15u', 'qabW_cam15u_to_xyz',
            'xyz_to_lab_cam_sww_2016', 'lab_cam_sww_2016_to_xyz']

from .colortf import colortf
__all__+=['colortf']

from luxpy import colorrendition as cri
__all__ += ['cri'] 


from . plotters import *
__all__ += plotters.__all__

#from .cri_graphics import *
#__all__ += cri_graphics.__all__