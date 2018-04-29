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
# Module with functions related to correlated color temperature calculations
###############################################################################
#
# _CCT_LUT_CALC: True: (re-)calculates LUTs for ccts in .cctluts/cct_lut_cctlist.dat and overwrites existing files.
#
# _CCT_LUT_PATH: Folder with Look-Up-Tables (LUT) for correlated color temperature calculation followings Ohno's method.
#
# _CCT_LUT: Dict with LUT.
#
# xyz_to_cct(): Calculates CCT,Duv from XYZ, wrapper for ..._ohno() & ..._search()
#
# xyz_to_duv(): Calculates Duv, (CCT) from XYZ, wrapper for ..._ohno() & ..._search()
#
# cct_to_xyz(): Calculates xyz from CCT, Duv [100 K < CCT < 10**20]
#
# xyz_to_cct_mcamy(): Calculates CCT from XYZ using Mcamy model:
#                   *[McCamy, Calvin S. (April 1992). "Correlated color temperature as an explicit function of chromaticity coordinates". Color Research & Application. 17 (2): 142–144.](http://onlinelibrary.wiley.com/doi/10.1002/col.5080170211/abstract)
#
# xyz_to_cct_HA(): Calculate CCT from XYZ using Hernández-Andrés et al. model .
#                  * [Hernández-Andrés, Javier; Lee, RL; Romero, J (September 20, 1999). Calculating Correlated Color Temperatures Across the Entire Gamut of Daylight and Skylight Chromaticities. Applied Optics. 38 (27): 5703–5709. PMID 18324081. doi:10.1364/AO.38.005703](https://www.osapublishing.org/ao/abstract.cfm?uri=ao-38-27-5703)
#
# xyz_to_cct_ohno(): Calculates CCT, Duv from XYZ using LUT following:
#                   * [Ohno Y. Practical use and calculation of CCT and Duv. Leukos. 2014 Jan 2;10(1):47-55.](http://www.tandfonline.com/doi/abs/10.1080/15502724.2014.839020)
#
# xyz_to_cct_search(): Calculates CCT, Duv from XYZ using brute-force search algorithm (between 1e2 K - 1e20 K on a log scale)
#
# cct_to_mired(): Converts from CCT to Mired scale (or back)
#
#------------------------------------------------------------------------------

Created on Wed Jun 28 22:52:28 2017

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

_CCT_LUT_CALC = False # True: (re-)calculates LUTs for ccts in .cctluts/cct_lut_cctlist.dat
__all__ = ['_CCT_LUT_CALC']

from .cct import *
__all__ += cct.__all__