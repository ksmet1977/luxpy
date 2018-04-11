# -*- coding: utf-8 -*-
"""
########################################################################################
# Package for color science, colorimetric and color appearance attribute calculations. #
########################################################################################

    For an overview of constants, functions, classes and modules of the luxpy package,
    see: "http://github.com/ksmet1977/luxpy/blob/master/README.md"

    * Author: K. A.G. Smet (ksmet1977 at gmail.com)
    * Version: 1.2.06
    * Date: April 11, 2018
    * License: GPLv3

########################################################################################

#------------------------------------------------------------------------------ 
Loads the following luxpy modules and classes:
    
    0.1.  helpers/ helpers.py (imported directly into luxpy namespace, details see end of this file)
    0.2.  math/ math.py (imported as math into the luxpy namespace, details see end of this file)
    
    1.  spectral/ cmf.py
    2.  spectral/ spectral.py
    3.  spectral/ spectral_databases
    4.  ctf/ colortransforms.py (imported directly into luxpy namespace)
    5.  cct/ cct.py (imported directly into luxpy namespace)
    6.  cat/ chromaticadaptation.py (imported in luxpy namespace as .cat)
    7.  cam/ colorappearancemodels.py (imported in luxpy namespace as .cam)
    8.  ctf/ colortf.py (imported directly into luxpy namespace)
    9.  cri/ colorrenditionindices.py (imported in luxpy namespace as .cri)
    10. graphics/ plotters.py (imported directly into luxpy namespace)
    11. classes/ SPD (imported directly into luxpy namespace)
    12. classes/ CDATA, XYZ, LAB (imported directly into luxpy namespace)

#------------------------------------------------------------------------------ 
Loads the following global default constants:
    
 * _PKG_PATH (absolute path to luxpy package)
 * _SEP (operating system operator)
 * _EPS = 7./3 - 4./3 -1 (machine epsilon)
 * _CIEOBS = '1931_2' (default CIE observer color matching function)
 * _CSPACE = 'Yuv' (default color space / chromaticity diagram)
 
#------------------------------------------------------------------------------ 
 Note: In luxpy 'global constants' start with '_'



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
from mpl_toolkits.mplot3d import Axes3D
import colorsys

__all__ = ['plt','Axes3D','np','pd','os','warnings','interpolate','minimize','odict']



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
from .helpers import *
__all__ += helpers.__all__

# Load math module:
from .math  import math as math 
__all__ += ['math']  

# Load spectral module:
from .spectral import *
__all__ += spectral.__all__

# Load color/chromaticty transforms module:
from .ctf import *
__all__ += ctf.__all__

# Load correlated color temperature module:
from .cct import *
__all__ += cct.__all__

# Load chromatic adaptation module:
from .cat import chromaticadaptation as cat
__all__ += ['cat']  

# Load color appearance model module:
from .cam import colorappearancemodels as cam
__all__ += ['cam']  

# load cam wrapper functions for use with colortf() from .colortransforms module:
from .cam.colorappearancemodels import _CAM_AXES, xyz_to_jabM_ciecam02, jabM_ciecam02_to_xyz, xyz_to_jabC_ciecam02, jabC_ciecam02_to_xyz,xyz_to_jabM_cam16, jabM_cam16_to_xyz, xyz_to_jabC_cam16, jabC_cam16_to_xyz, xyz_to_jab_cam02ucs, jab_cam02ucs_to_xyz, xyz_to_jab_cam02lcd, jab_cam02lcd_to_xyz,xyz_to_jab_cam02scd, jab_cam02scd_to_xyz, xyz_to_jab_cam16ucs, jab_cam16ucs_to_xyz, xyz_to_jab_cam16lcd, jab_cam16lcd_to_xyz,xyz_to_jab_cam16scd, jab_cam16scd_to_xyz, xyz_to_qabW_cam15u, qabW_cam15u_to_xyz, xyz_to_lab_cam_sww_2016, lab_cam_sww_2016_to_xyz

__all__ += ['_CAM_AXES', 'xyz_to_jabM_ciecam02', 'jabM_ciecam02_to_xyz', 'xyz_to_jabC_ciecam02', 'jabC_ciecam02_to_xyz',
            'xyz_to_jabM_cam16', 'jabM_cam16_to_xyz', 'xyz_to_jabC_cam16', 'jabC_cam16_to_xyz',
            'xyz_to_jab_cam02ucs', 'jab_cam02ucs_to_xyz', 'xyz_to_jab_cam02lcd', 'jab_cam02lcd_to_xyz','xyz_to_jab_cam02scd', 'jab_cam02scd_to_xyz', 
            'xyz_to_jab_cam16ucs', 'jab_cam16ucs_to_xyz', 'xyz_to_jab_cam16lcd', 'jab_cam16lcd_to_xyz','xyz_to_jab_cam16scd', 'jab_cam16scd_to_xyz', 
            'xyz_to_qabW_cam15u', 'qabW_cam15u_to_xyz',
            'xyz_to_lab_cam_sww_2016', 'lab_cam_sww_2016_to_xyz']

_CSPACE_AXES = {**_CSPACE_AXES, **_CAM_AXES} # merge _CAM_AXES dict with _CSPACE_AXES dict

# Extend colot transform module:
from .ctf.colortf import *
__all__+=['colortf']

# Load color rendition module:
from .cri import colorrendition as cri
__all__ += ['cri'] 

# Load some basic graphics functions:
from .graphics.plotters import *
__all__ += graphics.__all__

# Load classes:
from .classes.SPD import SPD
from .classes.CDATA import CDATA, XYZ, LAB
