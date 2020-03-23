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

 * JETI: specbos 1211, etc.
 * OceanOptics: QEPro, QE65Pro, QE65000, USB2000, USB650,etc.
 
 :init(manufacturer): import module for specified manufacturer; make sure everything (drivers, external packages, ...) required is installed! 
 
 :get_spd(): wrapper function to measure a spectral power distribution using a spectrometer of one of the supported manufacturers. 
 
Notes
-----
 1. For info on the input arguments of get_spd(), see help for each identically named function in each of the subpackages. 
 2. The use of jeti spectrometers requires access to some dll files (delivered with this package).
 3. The use of oceanoptics spectrometers requires the manual installation of pyseabreeze, as well as some other 'manual' settings. See help for oceanoptics sub-package. 

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

__all__ = ['init','get_spd']

def init(manufacturer):
    """
    Import module for specified manufacturer. Make sure everything (drivers, external packages, ...) required is installed!
    """
    if (manufacturer not in globals()):#(eval(manufacturer) is None):
        if manufacturer == 'jeti':
            # import inside function to ensure that the module only get loaded 
            # when needed to avoid having to have working installations for the 
            # other manufacturers:
            try:
                from .jeti import jeti as jeti
                return jeti
            except:
                Warning('Could not load jeti sub-module into spectro.')
 
        
        elif manufacturer == 'oceanoptics':
            # import inside function to ensure that the module only get loaded 
            # when needed to avoid having to have working installations for the 
            # other manufacturers:
            try:
                from .oceanoptics import oceanoptics as oceanoptics
                return oceanoptics
            except:
                Warning('Could not load oceanoptics sub-module into spectro. Make python-seabreeze, pyubs, etc. is installed correctly.')
        else:
            raise Exception('Unsupported manufacturer!')

  
def get_spd(manufacturer = 'jeti', dvc = 0, Tint = 0, autoTint_max = None, close_device = True, out = 'spd', **kwargs):
    """
    Measure a spectral power distribution using a spectrometer of one of the supported manufacturers. 
    
    Args:
        :manufacturer:
            | 'jeti' or 'oceanoptics', optional
            | Manufacturer of spectrometer (ensures the correct module is loaded).
        :dvc:
            | 0 or int or spectrometer handle, optional
            | If int: function will try to initialize the spectrometer to 
            |         obtain a handle. The int represents the device 
            |         number in a list of all detected devices of the manufacturer.
        :Tint:
            | 0 or Float, optional
            | Integration time in seconds. (if 0: find best integration time, but < autoTint_max).
        :autoTint_max:
            | Limit Tint to this value when Tint = 0.
        :close_device:
            | True, optional
            | Close spectrometer after measurement.
            | If 'dvc' not in out.split(','): always close!!!
        :out:
            | "spd" or e.g. "spd,dvc,Errors", optional
            | Requested return.
        :kwargs:
            | For info on additional input (keyword) arguments of get_spd(), 
            | see help for each identically named function in each of the subpackages. 
    
    Returns:
        :spd:
            | ndarray with spectrum. (row 0: wavelengths, row1: values)
        :dvc:
            | Device handle, if succesfull open (_ERROR: failure, nan: closed)
        :Errors:
            | Dict with error messages.
    """
    spec = init(manufacturer)
    return spec.get_spd(dvc = dvc, Tint = Tint, autoTint_max = autoTint_max, 
                        close_device = close_device, out = out, **kwargs)