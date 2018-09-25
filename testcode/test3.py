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
Module for 
==================================================================


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import luxpy as lx # package for color science calculations 
import matplotlib.pyplot as plt # package for plotting
import numpy as np # fundamental package for scientific computing 
import timeit # package for timing functions

cieobs = '1964_10' # set CIE observer, i.e. cmf set
ccts = [3000,4000,4500, 6000] # define M = 4 CCTs
ref_types = ['BB','DL','cierf','DL'] # define reference illuminant types

# calculate reference illuminants:
REF = lx.cri_ref(ccts, ref_type = ref_types, norm_type = 'lambda', norm_f = 600)

TCS8 = lx._CRI_RFL['cie-13.3-1995']['8'] # 8 TCS from CIE 13.3-1995
xyz_TCS8_REF = lx.spd_to_xyz(REF, cieobs = cieobs, rfl = TCS8, relative = True) 
xyz_TCS8_REF_2, xyz_REF_2 = lx.spd_to_xyz(REF, cieobs = cieobs, rfl = TCS8, relative = True, out = 2)

Yuv_REF_2 = lx.xyz_to_Yuv(xyz_REF_2)
axh = lx.plotSL(cspace = 'Yuv', cieobs = cieobs, show = False,\
                 BBL = True, DL = True, diagram_colors = True)

# Step 2:
Y, u, v = np.squeeze(lx.asplit(Yuv_REF_2)) # splits array along last axis

# Step 3:
lx.plot_color_data(u, v, formatstr = 'go', label = 'Yuv_REF_2')