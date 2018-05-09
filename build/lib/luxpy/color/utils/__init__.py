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
Module with functions related to plotting of color data
=======================================================

 :plot_color_data(): Plot color data (local helper function)

 :plotDL(): Plot daylight locus. 

 :plotBB(): Plot blackbody locus. 

 :plotSL(): | Plot spectrum locus.  
            | (plotBB() and plotDL() are also called, but can be turned off).

 :plotcerulean(): | Plot cerulean (yellow (577 nm) - blue (472 nm)) line 
                  | (Kuehni, CRA, 2014: Table II: spectral lights)
                  | `Kuehni, R. G. (2014). 
                    Unique hues and their stimuli—state of the art. 
                    Color Research & Application, 39(3), 279–287.
                    <https://doi.org/10.1002/col.21793>`_

 :plotUH(): | Plot unique hue lines from color space center point xyz0. 
            | (Kuehni, CRA, 2014: uY,uB,uG: Table II: spectral lights; 
            | uR: Table IV: Xiao data) 
            | `Kuehni, R. G. (2014). 
              Unique hues and their stimuli—state of the art. 
              Color Research & Application, 39(3), 279–287.
              <https://doi.org/10.1002/col.21793>`_
    
 :plotcircle(): Plot one or more concentric circles.

===============================================================================
"""
from .plotters import *
__all__ = plotters.__all__