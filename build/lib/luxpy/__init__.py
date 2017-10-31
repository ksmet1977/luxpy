# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:44:10 2017

@author: kevin.smet
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
_pckg_dir = os.path.dirname(__file__)  # Get absolute path to package: 
_sep = os.sep # operating system separator
__all__ += ['_pckg_dir','_sep']

#--------------------------------------------------------------------------------------------------
# set some general parameters:
_eps = 7./3 - 4./3 -1 # get machine epsilon
__all__+=['_eps']

#--------------------------------------------------------------------------------------------------
# set default colorimetric observer
_cieobs = '1931_2' #(CMF selection)
_cspace = 'Yuv'
__all__+=['_cieobs','_cspace']




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

## Set xyzbar in _cmf dict:
_cmf['bar'] = {_cmf['types'][i] : (xyzbar(cieobs = _cmf['types'][i], scr = 'file', kind = 'np')) for i in range(len(_cmf['types']))}

# load spd and rfl data in /spd/:
from . spectral_databases import _R_dir, _S_dir, _cie_illuminants, _iestm30, _cri_rfl, _munsell
__all__ += ['_R_dir', '_S_dir', '_cri_rfl', '_cie_illuminants', '_iestm30','_munsell'] 

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

from luxpy import colorrenditionindices as cri
__all__ += ['cri'] 

from . plotters import *
__all__ += plotters.__all__