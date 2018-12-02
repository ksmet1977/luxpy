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
Module for reading and writing IES and LDT files.
=================================================

 :read_lamp_data: Read in light intensity distribution and other lamp data from LDT or IES files.

    Notes:
        1.Only basic support. Writing is not yet implemented.
        2.Reading IES files is based on Blender's ies2cycles.py
        3.This was implemented to build some uv-texture maps for rendering and only tested for a few files.
        4. Use at own risk. No warranties.
        
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from .io_lid_files import *
__all__ = io_lid_files.__all__
