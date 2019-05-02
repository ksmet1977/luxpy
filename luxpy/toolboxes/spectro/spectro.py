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

#from .jeti import jeti as jeti
#from .oceanoptics import oceanoptics as oceanoptics

__all__ = ['get_spd']

def get_spd(manufacturer = 'jeti', **kwargs):
	"""
	Measure a spectral power distribution using a spectrometer of one of the supported manufacturers. 
	
	Args:
		| For info on the input arguments of get_spd(), 
		| see help for each identically named function 
		| in each of the subpackage. 
		
	Returns:
		| ndarray with wavelengths (1st row) and spectral radiance (2nd row). 
		| (+ additional output as specified in :out:).
	"""
    
	if manufacturer == 'jeti':
        # import inside function to ensure that the module only get loaded 
        # when needed to avoid having to have working installations for the 
        # other manufacturers:
		from .jeti import jeti as jeti 
		return jeti.get_spd(**kwargs)
    
    
	elif manufacturer == 'oceanoptics':
        # import inside function to ensure that the module only get loaded 
        # when needed to avoid having to have working installations for the 
        # other manufacturers:
		from .oceanoptics import oceanoptics as oceanoptics
		return oceanoptics.get_spd(**kwargs)