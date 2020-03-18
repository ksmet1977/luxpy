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
Module for discrimination ellipses
==================================================
 :get_macadam_ellipse(): Estimate n-step MacAdam ellipse at CIE x,y coordinates  
 
 :get_gij_fmc(): Get gij matrices describing the discrimination ellipses for Yxy using FMC-1 or FMC-2.

 :get_fmc_discrimination_ellipse(): Get n-step discrimination ellipse(s) in v-format (R,r, xc, yc, theta) for Yxy using FMC-1 or FMC-2.

 
References:
    1. MacAdam DL. Visual Sensitivities to Color Differences in Daylight*. J Opt Soc Am. 1942;32(5):247-274.
    2. Chickering, K.D. (1967), Optimization of the MacAdam-Modified 1965 Friele Color-Difference Formula, 57(4):537-541
    3. Chickering, K.D. (1971), FMC Color-Difference Formulas: Clarification Concerning Usage, 61(1):118-122
    
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import (np, plt, math, Yxy_to_xyz, plotSL, plot_chromaticity_diagram_colors, plotellipse)
from .macadamellipses import get_macadam_ellipse
from .frieleellipses import (get_gij_fmc, get_fmc_discrimination_ellipse)

__all__ = ['get_discrimination_ellipse','get_macadam_ellipse','get_gij_fmc','get_fmc_discrimination_ellipse']


def get_discrimination_ellipse(Yxy = np.array([[100,1/3,1/3]]), etype = 'fmc2', nsteps = 10, k_neighbours = 3, average_cik = True, Y = None):
    """
    Get discrimination ellipse(s) in v-format (R,r, xc, yc, theta) for Yxy using an interpolation of the MacAdam ellipses or using FMC-1 or FMC-2.
    
    Args:
        :Yxy:
            | 2D ndarray with [Y,]x,y coordinate centers. 
            | If Yxy.shape[-1]==2: Y is added using the value from the Y-input argument.
        :etype:
            | 'fmc2', optional
            | Type color discrimination ellipse estimation to use.
            | options: 'macadam', 'fmc1', 'fmc2' 
            |  - 'macadam': interpolate covariance matrices of closest MacAdam ellipses (see: get_macadam_ellipse?).
            |  - 'fmc1': use FMC-1 from ref 2 (see get_fmc_discrimination_ellipse?).
            |  - 'fmc2': use FMC-1 from ref 3 (see get_fmc_discrimination_ellipse?).
        :nsteps:
            | 10, optional
            | Set multiplication factor for ellipses 
            | (nsteps=1 corresponds to approximately 1 MacAdam step, 
            | for FMC-2, Y also has to be 10.69, see note below).
        :k_neighbours:
            | 3, optional
            | Only for option 'macadam'.
            | Number of nearest ellipses to use to calculate ellipse at xy 
        :average_cik:
            | True, optional
            | Only for option 'macadam'.
            | If True: take distance weighted average of inverse 
            |   'covariance ellipse' elements cik. 
            | If False: average major & minor axis lengths and 
            |   ellipse orientation angles directly.
        :Y:
            | None, optional
            | Only for option 'fmc2'(see note below).
            | If not None: Y = 10.69 and overrides values in Yxy. 
    
    Note:
        1. FMC-2 is almost identical to FMC-1 is Y = 10.69!; see [3]
    
    References:
       1. MacAdam DL. Visual Sensitivities to Color Differences in Daylight*. J Opt Soc Am. 1942;32(5):247-274.
       2. Chickering, K.D. (1967), Optimization of the MacAdam-Modified 1965 Friele Color-Difference Formula, 57(4):537-541
       3. Chickering, K.D. (1971), FMC Color-Difference Formulas: Clarification Concerning Usage, 61(1):118-122
    """
    if Yxy.shape[-1] == 2:
        Yxy = np.hstack((100*np.ones((Yxy.shape[0],1)),Yxy))
    if Y is not None:
        Yxy[...,0] = Y
    if etype == 'macadam':
        return get_macadam_ellipse(xy = Yxy[...,1:], k_neighbours = k_neighbours, nsteps = nsteps, average_cik = average_cik)
    else:
        return get_fmc_discrimination_ellipse(Yxy = Yxy, etype = etype, nsteps = nsteps, Y = Y)
    
if __name__ == '__main__':
    Yxy1 = np.array([[100,1/3,1/3]])
    Yxy2 = np.array([[100,1/3, 1/3],[50,1/3,1/3]])
    gij_11 = get_gij_fmc(Yxy1,etype = 'fmc1', ellipsoids=False)
    gij_12 = get_gij_fmc(Yxy2,etype = 'fmc1', ellipsoids=False)
    
    # Get MacAdam ellipses:
    v_mac = get_macadam_ellipse(xy = None)
    xys = v_mac[:,2:4]
    
    # Get discrimination ellipses for MacAdam centers using FMC-1 & FMC-2:
    v_mac_0 = get_fmc_discrimination_ellipse(Yxy = xys, etype = 'macadam', nsteps = 10)
    v_mac_1 = get_discrimination_ellipse(Yxy = xys, etype = 'fmc1', nsteps = 10)
    v_mac_2 = get_discrimination_ellipse(Yxy = xys, etype = 'fmc2', nsteps = 10, Y = 10.69)
    
    # Plot results:
    cspace = 'Yxy'
    #axh = plot_chromaticity_diagram_colors(cspace = cspace)
    axh = plotSL(cspace = cspace, cieobs = '1931_2', show = False, diagram_colors = False)
    axh = plotellipse(v_mac_0, show = True, axh = axh, cspace_in = None, cspace_out = cspace,plot_center = False, center_color = 'r', out = 'axh', line_style = ':', line_color ='r',line_width = 1.5)
    plotellipse(v_mac_1, show = True, axh = axh, cspace_in = None, cspace_out = cspace,line_color = 'b', line_style = ':', plot_center = True, center_color = 'k')
    plotellipse(v_mac_2, show = True, axh = axh, cspace_in = None, cspace_out = cspace,line_color = 'g', line_style = '--', plot_center = True, center_color = 'k')

    if cspace == 'Yuv':
        axh.set_xlim([0,0.6])
        axh.set_ylim([0,0.6])
    plt.plot(xys[:,0],xys[:,1],'r.')
    

