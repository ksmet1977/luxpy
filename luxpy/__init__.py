# -*- coding: utf-8 -*-
"""
###############################################################################
# LuxPy: a package for lighting and color science
###############################################################################

    * Author: K. A.G. Smet (ksmet1977 at gmail.com)
    * Version: 1.3.00
    * Date: May 1, 2018
    * License: GPLv3

###############################################################################


###############################################################################
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

#------------------------------------------------------------------------------ 
Loads the following luxpy sub-packages, modules and classes:
    
	/utils
		/ helpers
         helpers.py
		/.math
			math.py
			optimizers.py
	
	/spectrum
		cmf.py
		spectral.py
		spectral_databases.py
	
	/color
		colortransformations.py
		cct.py
		/.cat
			chromaticadaptation.py	
		/.cam
			colorappearancemodels.py
				cam_02_X.py
				cam15u.py
				sww16.py
		colortf.py
		/.deltaE
			colordifferences.py
      /.cri
    		colorrendition.py
			/utils
				DE_scalers.py
				helpers.py
				init_cri_defaults_database.py
				graphics.py
			/indices
				indices.py
					cie_wrappers.py
					ies_wrappers.py
					cri2012.py
					mcri.py
					cqs.py
			/ies_tm30
				ies_tm30_metrics.py
				ies_tm30_graphics.py
			/.VFPX
				VF_PX_models.py (imported in .cri as .VFPX)
					vectorshiftmodel.py
					pixelshiftmodel.py
		/utils
			plotters.py
		
		
	/classes
		SPD.py
		CDATA.py
		
	/data
		/cmfs
		/spds
		/rfls
		/cctluts

		
	/toolboxes
		
		/.ciephotbio
			cie_tn003_2015.py
			/data
			
		/.indvcmf
			individual_observer_cmf_model.py
			/data
		
		/.spdbuild
			spd_builder.py
			
		/.hypspcsim
			hyperspectral_img_simulator.py

#------------------------------------------------------------------------------ 
Loads the following global default constants:
    
 * _PKG_PATH (absolute path to luxpy package)
 * _SEP (operating system operator)
 * _EPS = 7./3 - 4./3 -1 (machine epsilon)
 * _CIEOBS = '1931_2' (default CIE observer color matching function)
 * _CSPACE = 'Yuv' (default color space / chromaticity diagram)
 
#------------------------------------------------------------------------------ 
 Note: In luxpy 'global constants' start with '_'


#------------------------------------------------------------------------------ 
Created on Sat Jun 17 15:44:10 2017

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
###############################################################################
# Initialze LuxPy
###############################################################################

#==============================================================================
# Import required modules
#==============================================================================

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import warnings
import os
from collections import OrderedDict as odict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys
import cv2

__all__ = ['plt','Axes3D','np','pd','os','warnings','interpolate','minimize',
           'cKDTree','odict','colorsys','cv2']


#==============================================================================
# Set up basic luxpy parameters
#==============================================================================


#------------------------------------------------------------------------------
# Setup output format of print:
np.set_printoptions(formatter={'float': lambda x: "{0:0.4e}".format(x)}) 

#------------------------------------------------------------------------------
# os related:
_PKG_PATH = os.path.dirname(__file__)  # Get absolute path to package: 
_SEP = os.sep # operating system separator
__all__ += ['_PKG_PATH','_SEP']

#------------------------------------------------------------------------------
# set some general parameters:
_EPS = 7./3 - 4./3 -1 # get machine epsilon
__all__+=['_EPS']

#------------------------------------------------------------------------------
# set default colorimetric observer
_CIEOBS = '1931_2' #(CMF selection)
_CSPACE = 'Yuv'
__all__+=['_CIEOBS','_CSPACE']




#==============================================================================
# Load luxpy specific modules:
#==============================================================================


#----------------------------------------
# From /utils:
#----------------------------------------
#   load module with basic helper fcns:
from .utils.helpers import *


#   load math sub_package:
from .utils.math import math as math


#----------------------------------------
# From /spectrum:
#----------------------------------------

#   Load spectral module:
from .spectrum.basics import *


#----------------------------------------
# From /color:
#----------------------------------------

#   Load color/chromaticty transforms module:
from .color.ctf.colortransforms import *

#   Load correlated color temperature module:
from .color.cct import *

#   Load chromatic adaptation module:
from .color.cat import chromaticadaptation as cat

#   Load color appearance model module:
from .color.cam import colorappearancemodels as cam

#   load cam wrapper functions for use with colortf() from .colortransforms module:
from .color.cam import (_CAM_AXES, 
                  xyz_to_jabM_ciecam02, jabM_ciecam02_to_xyz, 
                  xyz_to_jabC_ciecam02, jabC_ciecam02_to_xyz,
                  xyz_to_jabM_cam16, jabM_cam16_to_xyz, 
                  xyz_to_jabC_cam16, jabC_cam16_to_xyz, 
                  xyz_to_jab_cam02ucs, jab_cam02ucs_to_xyz, 
                  xyz_to_jab_cam02lcd, jab_cam02lcd_to_xyz,
                  xyz_to_jab_cam02scd, jab_cam02scd_to_xyz, 
                  xyz_to_jab_cam16ucs, jab_cam16ucs_to_xyz, 
                  xyz_to_jab_cam16lcd, jab_cam16lcd_to_xyz,
                  xyz_to_jab_cam16scd, jab_cam16scd_to_xyz, 
                  xyz_to_qabW_cam15u, qabW_cam15u_to_xyz, 
                  xyz_to_lab_cam_sww16, lab_cam_sww16_to_xyz)

#   Merge _CAM_AXES dict with _CSPACE_AXES dict:
_CSPACE_AXES = {**_CSPACE_AXES, **_CAM_AXES} 

#   Extend color transform module:
__all__ = [x for x in dir() if x[:2]!='__'] # to give color.ctf.colortf access to globals()
from .color.ctf.colortf import *

#   Load DE (color difference) module:
from .color.deltaE import colordifferences as deltaE

#   Load some basic graphics functions:
from .color.utils.plotters import *

#   Load color rendition sub-package:
from .color.cri import colorrendition as cri

#----------------------------------------
# From /classes:
#----------------------------------------
from .classes.SPD import SPD
from .classes.CDATA import CDATA, XYZ, LAB


#----------------------------------------
# From /toolboxes:
#----------------------------------------
#   load ciephotbio sub_package:
from .toolboxes.ciephotbio import cie_tn003_2015 as ciephotbio

#   load Asano Individual Observer lms-CMF model:
from .toolboxes.indvcmf import individual_observer_cmf_model as indvcmf

#   Load spdbuild sub_package:
from .toolboxes.spdbuild import spdbuilder as spdbuild

#   Load hypspcim sub_package:
from .toolboxes.hypspcim import hyperspectral_img_simulator as hypspcim

# Setup __all__:
__all__ = [x for x in dir() if x[:2]!='__']
#print(__all__)


###############################################################################
