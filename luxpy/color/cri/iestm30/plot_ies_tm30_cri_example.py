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
Created on Sun Apr  1 10:55:02 2018

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

import luxpy as lx
import numpy as np
import matplotlib.pyplot as plt
import colorsys

plot_iestm30_output = True

SPDs = lx._IESTM3015['S']['data']

SPD = SPDs[:3]

plot_CF = True
plot_VF = False
plot_SF = False
plot_bin_colors = True
vf_plot_bin_colors = False

#Nspds = SPD.shape[0] - 1
#
#cri_iestm30_defaults = lx.cri._CRI_DEFAULTS['iesrf']
#rg_pars_iesttm30 =cri_iestm30_defaults['rg_pars']
#
## Calculate metrics:
#out = 'Rf,Rg,cct,duv,Rfi,jabt,jabr, Rfhi,Rcshi,Rhshi'
#spd_to_iestm30 = lambda x: lx.cri.spd_to_cri(x, cri_type = cri_iestm30_defaults, out = out)
#Rf,Rg,cct,duv,Rfi,jabt,jabr,Rfhi,Rcshi,Rhshi = spd_to_iestm30(SPD)
#
#______________________________________________________________________________
# Plot IES TM30 output:
if plot_iestm30_output == True:
    
    # plot graphic output for SPD:
    axtype ='polar'
    normalized_chroma_ref = 100
    data = lx.cri.plot_cri_graphics(SPD, cri_type = 'iesrf', plot_VF = plot_VF, plot_CF = plot_CF, plot_SF = plot_SF, plot_bin_colors = plot_bin_colors, vf_plot_bin_colors = vf_plot_bin_colors, axtype = axtype, ax = None, plot_center_lines = False, plot_edge_lines = True, scalef = normalized_chroma_ref*1, force_CVG_layout = True)
                
    plt.show()