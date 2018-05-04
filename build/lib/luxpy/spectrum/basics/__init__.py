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


# Load cmfs, part 1 (prepare basic data dict, no actual cmfs)
from .cmf import *
__all__ = cmf.__all__

# Load spectral module:
from .spectral import *
__all__ += spectral.__all__

## Set xyzbar in _CMF dict:
#_CMF['bar'] = {_CMF['types'][i] : (xyzbar(cieobs = _CMF['types'][i], scr = 'file', kind = 'np')) for i in range(len(_CMF['types']))}
for i, cmf_type in enumerate(_CMF['types']): # store all in single nested dict
    _CMF[cmf_type]['bar'] =  xyzbar(cieobs = cmf_type, scr = 'file', kind = 'np')

# load spd and rfl data in /spd/:
from .spectral_databases import _R_PATH, _S_PATH, _CIE_ILLUMINANTS, _IESTM30, _CRI_RFL, _MUNSELL
__all__ += ['_R_PATH', '_S_PATH', '_CRI_RFL', '_CIE_ILLUMINANTS', '_IESTM30','_MUNSELL'] 
