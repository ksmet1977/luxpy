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
Module for color difference calculations
========================================

 :process_DEi(): Process color difference input DEi for output (helper fnc).

 :DE_camucs(): Calculate color appearance difference DE using camucs type model.

 :DE_2000(): Calculate DE2000 color difference.

 :DE_cspace():  Calculate color difference DE in specific color space.
 
 :get_macadam_ellipse(): Estimate n-step MacAdam ellipse at CIE x,y coordinates  
 
 :get_brown1957_ellipse(): Estimate n-step Brown (1957) ellipse at CIE x,y coordinates.  
 
 :get_gij_fmc(): Get gij matrices describing the discrimination ellipses for Yxy using FMC-1 or FMC-2.

 :get_fmc_discrimination_ellipse(): Get n-step discrimination ellipse(s) in v-format (R,r, xc, yc, theta) for Yxy using FMC-1 or FMC-2.
  
"""
from .colordifferences import *
__all__ = colordifferences.__all__

from .discriminationellipses import *
__all__ += discriminationellipses.__all__