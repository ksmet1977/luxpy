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
Module for display characterization
===================================
 :_PATH_DATA: path to package data folder   

 :_RGB:  set of RGB values that work quite well for display characterization
   
 :_XYZ: example set of measured XYZ values corresponding to the RGB values in _RGB
   
 :calibrate(): Calculate TR parameters/lut and conversion matrices
   
 :calibration_performance(): Check calibration performance (cfr. individual and average color differences for each stimulus). 

 :rgb_to_xyz(): Convert input rgb to xyz
    
 :xyz_to_rgb(): Convert input xyz to rgb
     
 :DisplayCalibration(): Calculate TR parameters/lut and conversion matrices and store in object.

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from .displaycalibration import *
__all__ = displaycalibration.__all__
