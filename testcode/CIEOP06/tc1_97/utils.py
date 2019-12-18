#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils: Utility functions for CIE functions provided by CIE TC 1-97.

Copyright (C) 2012-2018 Ivar Farup and Jan Henrik Wold

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import sys
import os
import inspect

def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    Parameters
    ----------
    relative : string
        The relative path name.

    Returns
    -------
    absolute : string
        The absolute path name.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = os.path.join(sys._MEIPASS, 'tc1_97')
    except Exception:
        base_path = os.path.dirname(os.path.abspath(inspect.getsourcefile(resource_path)))

    return os.path.join(base_path, relative_path)
