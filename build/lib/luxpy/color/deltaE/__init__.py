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
Module for color difference calculations
========================================

 :process_DEi(): Process color difference input DEi for output (helper fnc).

 :DE_camucs(): Calculate color appearance difference DE using camucs type model.

 :DE_2000(): Calculate DE2000 color difference.

 :DE_cspace():  Calculate color difference DE in specific color space.
"""
from .colordifferences import *
__all__ = colordifferences.__all__