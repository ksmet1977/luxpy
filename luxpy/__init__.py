# -*- coding: utf-8 -*-
"""
LuxPy: a package for lighting and color science
===============================================

    * Author: K.A.G. Smet (ksmet1977 at gmail.com)
    * Version: 1.6.8
    * Date: May 11, 2021
    * License: GPLv3

    * DOI: https://doi.org/10.5281/zenodo.1298963
    * Cite: `Smet, K. A. G. (2019). Tutorial: The LuxPy Python Toolbox for Lighting and Color Science. LEUKOS, 1–23. DOI: 10.1080/15502724.2018.1518717 <https://www.tandfonline.com/doi/full/10.1080/15502724.2018.1518717>`_ 

License
-------
Copyright (C) <2017><Kevin A.G. Smet> (ksmet1977 at gmail.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.


LuxPy package structure
-----------------------
run: luxpy.utils.show_luxpy_tree()


Imported core packages/modules/functions:
----------------------------------------
 * os, warnings, colorsys, itertools, copy, time, tkinter, ctypes, platform, subprocess, pathlib, importlib
 * collections.OrderedDict.odict
 * mpl_toolkits.mplot3d.Axes3D
 
 
Imported 3e party dependencies (automatic install):
---------------------------------------------------
 * numpy, scipy, matplotlib.pyplot, pandas, imageio
 * pyswarms (luxpy tries a forced pip install if not already installed)
 
Imported 3e party dependencies (requiring manual install):
----------------------------------------------------------
To control Ocean Optics spectrometers with spectro toolbox:
 * import seabreeze (conda install -c poehlmann python-seabreeze)
 * pip install pyusb (for use with 'pyseabreeze' backend of python-seabreeze)


Global constants
----------------
The package uses several global constants that set the default state/behaviour
of the calculations. LuxPy 'global constants' start with '`_`' and are in an 
all `_ALL_CAPITAL` format. 

E.g.:
 * _CIEOBS = '1931_2' (default CIE observer color matching function)
 * _CSPACE = 'Yuv' (default color space / chromaticity diagram)
 
**!!DO NOT CHANGE THESE CONSTANTS!!**
"""
###############################################################################
# Initialze LuxPy
###############################################################################
# Package info:
__VERSION__ = 'v1.6.8'; """Current version"""
__version__ = __VERSION__
__DATE__ = '11-May-2021'; """release date"""

__COPYRIGHT__ = 'Copyright (C) 2017-2021 - Kevin A.G. Smet'; """copyright info"""

__AUTHOR__ = 'Kevin A.G. Smet'; """Package author"""
__EMAIL__ = 'ksmet1977 at gmail.com'; """contact info"""
__URL__ = 'github.com/ksmet1977/luxpy/'; """package url"""
__LICENSE__ = 'GPLv3'; """ License """
__DOI__ = ['https://doi.org/10.5281/zenodo.1298963', 'https://doi.org/10.1080/15502724.2018.1518717']; """ DOIs: zenodo, Leukos """
__CITE__ = '`Smet, K. A. G. (2019). Tutorial: The LuxPy Python Toolbox for Lighting and Color Science. LEUKOS, 1–23. DOI: 10.1080/15502724.2018.1518717 <https://www.tandfonline.com/doi/full/10.1080/15502724.2018.1518717>`_'; """ Citation info """ 
__all__ = ['__version__','__VERSION__','__AUTHOR__','__EMAIL__', '__URL__','__DATE__',
           '__COPYRIGHT__','__LICENSE__','__DOI__','__CITE__']

#==============================================================================
# Import required modules
#==============================================================================
# Imports are done in utils module

# keep track of required packages over all of luxpy:
# core: should be in core
# other: commonly used packages
# special: more special packages that are imported (or tried to be) on use of module
__REQUIRED__={'core':['os','warnings','pathlib','importlib',
                      'collections.OrderedDict.odict','mpl_toolkits.mplot3d.Axes3D',
                      'colorsys','itertools','copy','time','tkinter','ctypes',
                      'platform','subprocess',
                      'cProfile', 'pstats', 'io'],
              'other':['numpy','scipy','matplotlib.pyplot','pandas','imageio'],
              'special':['seabreeze', 'seabreeze.spectrometers','pyswarms']}
# (some imports for spectro toolbox are done there to avoid dependency 
# on manual install requirements)
__all__ += ['__REQUIRED__']

#==============================================================================
# Set up basic luxpy parameters
#==============================================================================

#------------------------------------------------------------------------------
import numpy as np
# Setup output format of print:
np.set_printoptions(formatter={'float': lambda x: "{0:0.4e}".format(x)}) 

# Setup numpy warnings:
np.seterr(over = 'ignore', under = 'ignore')


#------------------------------------------------------------------------------
# set default colorimetric observer
_CIEOBS = '1931_2'; """ Default CMF set """ 
_CSPACE = 'Yuv'; """ Default color space """
__all__+=['_CIEOBS','_CSPACE']


#==============================================================================
# Load luxpy specific modules:
#==============================================================================

#----------------------------------------
# From /utils:
#----------------------------------------

#   load module with basic utility fcns:
import luxpy.utils 
from luxpy.utils import _PKG_PATH
__all__ += ['utils','_PKG_PATH']


#----------------------------------------
# From /math:
#----------------------------------------

#   load math sub_package:
import luxpy.math
__all__ += ['math']


#----------------------------------------
# From /spectrum:
#----------------------------------------

#   Load spectral module:
from luxpy.spectrum import *
__all__ += spectrum.__all__


#----------------------------------------
# From /color:
#----------------------------------------

#   Load color/chromaticty transforms module:
from luxpy.color.ctf.colortransforms import *
__all__ += color.ctf.colortransforms.__all__

#   Load correlated color temperature module:
from luxpy.color.cct import *
__all__ += color.cct.__all__

#   Load chromatic adaptation module:
from luxpy.color.cat import chromaticadaptation as cat
__all__ += ['cat']

#   Load whiteness metric module:
from luxpy.color.whiteness.smet_white_loci import *
__all__ += color.whiteness.smet_white_loci.__all__

#   Load color appearance model module:
from luxpy.color import cam 
__all__ += ['cam']


#   load cam wrapper functions for use with colortf() from .colortransforms module:
from luxpy.color.cam import (_CAM_AXES, 
                  xyz_to_jabM_ciecam02, jabM_ciecam02_to_xyz, 
                  xyz_to_jabC_ciecam02, jabC_ciecam02_to_xyz,
                  xyz_to_jabM_ciecam16, jabM_ciecam16_to_xyz, 
                  xyz_to_jabC_ciecam16, jabC_ciecam16_to_xyz, 
                  xyz_to_jabz,          jabz_to_xyz,
                  xyz_to_jabM_zcam,      jabM_zcam_to_xyz, 
                  xyz_to_jabC_zcam,     jabC_zcam_to_xyz, 
                  xyz_to_jab_cam02ucs,  jab_cam02ucs_to_xyz, 
                  xyz_to_jab_cam02lcd,  jab_cam02lcd_to_xyz,
                  xyz_to_jab_cam02scd,  jab_cam02scd_to_xyz, 
                  xyz_to_jab_cam16ucs,  jab_cam16ucs_to_xyz, 
                  xyz_to_jab_cam16lcd,  jab_cam16lcd_to_xyz,
                  xyz_to_jab_cam16scd,  jab_cam16scd_to_xyz, 
                  xyz_to_qabW_cam15u,   qabW_cam15u_to_xyz, 
                  xyz_to_lab_cam_sww16, lab_cam_sww16_to_xyz,
                  xyz_to_qabM_cam18sl,  qabM_cam18sl_to_xyz,
                  xyz_to_qabS_cam18sl,  qabS_cam18sl_to_xyz)

__all__ += ['_CAM_AXES', 
          'xyz_to_jabM_ciecam02', 'jabM_ciecam02_to_xyz', 
          'xyz_to_jabC_ciecam02', 'jabC_ciecam02_to_xyz',
          'xyz_to_jabM_ciecam16', 'jabM_ciecam16_to_xyz', 
          'xyz_to_jabC_ciecam16', 'jabC_ciecam16_to_xyz',
          'xyz_to_jabz',          'jabz_to_xyz',
          'xyz_to_jabM_zcam',     'jabM_zcam_to_xyz', 
          'xyz_to_jabC_zcam',     'jabC_zcam_to_xyz', 
          'xyz_to_jab_cam02ucs', 'jab_cam02ucs_to_xyz', 
          'xyz_to_jab_cam02lcd', 'jab_cam02lcd_to_xyz',
          'xyz_to_jab_cam02scd', 'jab_cam02scd_to_xyz', 
          'xyz_to_jab_cam16ucs', 'jab_cam16ucs_to_xyz', 
          'xyz_to_jab_cam16lcd', 'jab_cam16lcd_to_xyz',
          'xyz_to_jab_cam16scd', 'jab_cam16scd_to_xyz', 
          'xyz_to_qabW_cam15u',  'qabW_cam15u_to_xyz', 
          'xyz_to_lab_cam_sww16', 'lab_cam_sww16_to_xyz',
          'xyz_to_qabM_cam18sl', 'qabM_cam18sl_to_xyz',
          'xyz_to_qabS_cam18sl', 'qabS_cam18sl_to_xyz']


#   Merge _CAM_AXES dict with _CSPACE_AXES dict:
_CSPACE_AXES = {**_CSPACE_AXES, **_CAM_AXES}; """ Dictionary with color space axes labels for plotting """


#   Extend color transform module:
#__all__ = [x for x in dir() if x[:2]!='__'] # to give color.ctf.colortf access to globals()
from luxpy.color.ctf.colortf import *
__all__ += color.ctf.colortf.__all__

#   Load some basic graphics functions:
from luxpy.color.utils.plotters import *
__all__ += color.utils.plotters.__all__

#   Load DE (color difference) module:
from luxpy.color import deltaE 
__all__ += ['deltaE']

#   Load color rendition sub-package:
from luxpy.color.cri import colorrendition as cri
__all__ += ['cri']



#----------------------------------------
# Import some class functionality:
#----------------------------------------
from luxpy.color.CDATA import CDATA, XYZ, LAB
__all__ += ['CDATA', 'XYZ', 'LAB']
from luxpy.spectrum.SPD import SPD
__all__ += ['SPD']


#----------------------------------------
# From /toolboxes:
#----------------------------------------
list_of_toolboxes = ['photbiochem','indvcmf','spdbuild','hypspcim','iolidfiles','spectro','rgb2spec','dispcal']
try:
    #   load ciephotbio sub_package:
    from luxpy.toolboxes import photbiochem 
    __all__ += ['photbiochem']
    
    #   load Asano Individual Observer lms-CMF model:
    from luxpy.toolboxes import indvcmf 
    __all__ += ['indvcmf']
    
    #   Load spdbuild sub_package:
    from luxpy.toolboxes import spdbuild 
    __all__ += ['spdbuild']
    
    #   Load hypspcim sub_package:
    from luxpy.toolboxes import hypspcim 
    __all__ += ['hypspcim']
    
    #   Load hypspcim sub_package:
    from luxpy.toolboxes import iolidfiles 
    __all__ += ['iolidfiles']
    
    #   Load spectro sub_package:
    from luxpy.toolboxes import spectro 
    __all__ += ['spectro']
    
    #   Load rgb2spec sub_package:
    from luxpy.toolboxes import rgb2spec
    __all__ += ['rgb2spec']
    
    #   Load dispcal sub_package:
    from luxpy.toolboxes import dispcal
    __all__ += ['dispcal']
except:
    pass

###############################################################################
