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
###################################################################################################
# Sub-package with color appearance models
###################################################################################################
# _UNIQUE_HUE_DATA: database of unique hues with corresponding Hue quadratures and eccentricity factors
#                   (ciecam97s, ciecam02, cam16, cam15u)
#
# _SURROUND_PARAMETERS: database of surround parameters c, Nc, F and FLL for ciecam02, cam16, ciecam97s and cam15u.
#
# _NAKA_RUSHTON_PARAMETERS: database with parameters (n, sig, scaling and noise) for the Naka-Rushton function: scaling * ((data**n) / ((data**n) + (sig**n))) + noise
#
# _CAM_02_X_UCS_PARAMETERS: database with parameters specifying the conversion from ciecam02/cam16 to cam[x]ucs (uniform color space), cam[x]lcd (large color diff.), cam[x]scd (small color diff).
#
# _CAM15U_PARAMETERS: database with CAM15u model parameters.
#
# _CAM_SWW16_PARAMETERS: database with cam_sww16 parameters (model by Smet, Webster and Whitehead published in JOSA A in 2016)
# 
# _CAM_DEFAULT_WHITE_POINT: Default internal reference white point (xyz)
#
# _CAM_DEFAULT_TYPE: Default CAM type str specifier.
#
# _CAM_DEFAULT_CONDITIONS: Default CAM model parameters for model in cam._CAM_DEFAULT_TYPE
#
# _CAM_AXES: dict with list[str,str,str] containing axis labels of defined cspaces.
#
# naka_rushton(): applies a Naka-Rushton function to the input
# 
# hue_angle(): calculates a positive hue angle
#
# hue_quadrature(): calculates the Hue quadrature from the hue.
#
# cam_structure_ciecam02_cam16(): basic structure of both the ciecam02 and cam16 models. Has 'forward' (xyz --> color attributes) and 'inverse' (color attributes --> xyz) modes.
#
# ciecam02(): calculates ciecam02 output (wrapper for cam_structure_ciecam02_cam16 with specifics of ciecam02):  N. Moroney, M. D. Fairchild, R. W. G. Hunt, C. Li, M. R. Luo, and T. Newman, “The CIECAM02 color appearance model,” IS&T/SID Tenth Color Imaging Conference. p. 23, 2002.
#
# cam16(): calculates cam16 output (wrapper for cam_structure_ciecam02_cam16 with specifics of cam16):  C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a.
# 
# camucs_structure(): basic structure to go to ucs, lcd and scd color spaces (forward + inverse available)
#
# cam02ucs(): calculates ucs (or lcd, scd) output based on ciecam02 (forward + inverse available): M. R. Luo, G. Cui, and C. Li, “Uniform colour spaces based on CIECAM02 colour appearance model,” Color Res. Appl., vol. 31, no. 4, pp. 320–330, 2006.
#
# cam16ucs(): calculates ucs (or lcd, scd) output based on cam16 (forward + inverse available):  C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a.
#
# cam15u(): calculates the output for the CAM15u model for self-luminous unrelated stimuli. : M. Withouck, K. A. G. Smet, W. R. Ryckaert, and P. Hanselaer, “Experimental driven modelling of the color appearance of unrelated self-luminous stimuli: CAM15u,” Opt. Express, vol. 23, no. 9, pp. 12045–12064, 2015.
#
# cam_sww16(): calculates output for the CAM developed by Smet, Webster and Whitehead:  Smet, K. A. G., Webster, M. A., & Whitehead, L. A. (2016). A simple principled approach for modeling and understanding uniform color metrics. Journal of the Optical Society of America A, 33(3), A319–A331. https://doi.org/10.1364/JOSAA.33.00A319
#
# specific wrappers in the xyz_to_...() and ..._to_xyz() format:
# 'xyz_to_jabM_ciecam02', 'jabM_ciecam02_to_xyz',
# 'xyz_to_jabC_ciecam02', 'jabC_ciecam02_to_xyz',
# 'xyz_to_jabM_cam16', 'jabM_cam16_to_xyz',
# 'xyz_to_jabC_cam16', 'jabC_cam16_to_xyz',
# 'xyz_to_jab_cam02ucs', 'jab_cam02ucs_to_xyz', 
# 'xyz_to_jab_cam02lcd', 'jab_cam02lcd_to_xyz',
# 'xyz_to_jab_cam02scd', 'jab_cam02scd_to_xyz', 
# 'xyz_to_jab_cam16ucs', 'jab_cam16ucs_to_xyz',
# 'xyz_to_jab_cam16lcd', 'jab_cam16lcd_to_xyz',
# 'xyz_to_jab_cam16scd', 'jab_cam16scd_to_xyz', 
# 'xyz_to_qabW_cam15u', 'qabW_cam15u_to_xyz'
# 'xyz_to_lAb_cam_sww16', 'lab_cam_sww16_to_xyz'
#------------------------------------------------------------------------------

Created on Sun Jun 25 09:55:05 2017

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from .colorappearancemodels import *