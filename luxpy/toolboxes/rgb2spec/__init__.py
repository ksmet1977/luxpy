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
Module for RGB to spectrum conversions

 :_BASESPEC_SMITS: Default dict with base spectra for white, cyan, magenta, yellow, blue, green and red for each intent ('rfl' or 'spd')
 :rgb_to_spec_smits(): Convert an array of RGB values to a spectrum using a smits like conversion as implemented in mitsuba (July 10, 2019)
 :convert(): Convert an array of RGB values to a spectrum (wrapper around rgb_to_spec_smits(), future: implement other methods)

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from luxpy import _WL3
from .smits_mitsuba import *
__all__ = smits_mitsuba.__all__
__all__ += ['convert']

def convert(rgb, method = 'smits_mtsb', intent = 'rfl',  bitdepth = 8, wlr = _WL3, rgb2spec = None):
    """
    Convert an array of RGB values to a spectrum.
    
    Args:
        :rgb: 
            | ndarray of list of rgb values
        :method:
            | 'smits_mtsb', optional
            | Method to use for conversion:
            |  - 'smits_mtsb': use a smits like conversion as implemented in mitsuba.
        :intent:
            | 'rfl' (or 'spd'), optional
            | type of requested spectrum conversion .
        :bitdepth:
            | 8, optional
            | bit depth of rgb values
        :wlr: 
            | _WL3, optional
            | desired wavelength (nm) range of spectrum.
        :rgb2spec:
            | None, optional
            | Dict with base spectra for white, cyan, magenta, yellow, blue, green and red for each intent.
            | If None: use _BASESPEC_SMITS.
        
    Returns:
        :spec: 
            | ndarray with spectrum or spectra (one for each rgb value, first row are the wavelengths)
    """
    return rgb_to_spec_smits(rgb, intent = intent,  bitdepth = bitdepth, wlr = wlr, rgb2spec = rgb2spec)


