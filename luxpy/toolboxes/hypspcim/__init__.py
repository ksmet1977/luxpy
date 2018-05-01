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
###############################################################################
# Module for hyper spectral image simulation
###############################################################################
# _HYPSPCIM_PATH: path to module

# _HYPSPCIM_DEFAULT_IMAGE: path + filename to default image

# xyz_to_rfl(): approximate spectral reflectance of xyz based on k nearest 
                neighbour interpolation of samples from a standard reflectance 
                set.

# render_image(): Render image under specified light source spd.

#------------------------------------------------------------------------------
Created on Sun Apr 29 16:30:05 2018

@author: kevin.smet
"""
from .hyperspectral_img_simulator import *
