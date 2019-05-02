# -*- coding: utf-8 -*-
########################################################################
# <spectro: a Python package for spectral measurement.>
# Copyright (C) <2019>  <Kevin A.G. Smet> (ksmet1977 at gmail.com)
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
Package for spectral measurements
=================================

Supported devices:
------------------

 * JETI: specbos 1211, ...
 * OceanOptics: QEPro, QE65Pro, QE65000, USB2000, USB650,...
 
 :get_spd(): wrapper function to measure a spectral power distribution using a spectrometer of one of the supported manufacturers. 
 
Notes
-----
 1. For info on the input arguments of get_spd(), see help for each identically named function in each of the subpackages. 
 2. The use of jeti spectrometers requires access to some dll files (delivered with this package).
 3. The use of oceanoptics spectrometers requires the manual installation of pyseabreeze, 
 as well as some other 'manual' settings. See help for oceanoptics sub-package. 

 
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from  .spectro import *	
__all__ = spectro.__all__
		



