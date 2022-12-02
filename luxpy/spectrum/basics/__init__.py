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
from luxpy.utils import np

# load spd and rfl data in /spd/:
from .spectral_databases import (_R_PATH, _S_PATH, _CIE_ILLUMINANTS, 
                                  _CIE_E, _CIE_D65, _CIE_A, _CIE_B, _CIE_C, _CIE_F4,_CIE_L41,
                                  _CIE_F_SERIES, _CIE_F3_SERIES,_CIE_HP_SERIES,_CIE_LED_SERIES,
                                  _IESTM3015, _IESTM3018, 
                                  _CIE_GLASS_ID, _CRI_RFL, _RFL, _MUNSELL)

# Load cmfs, part 1 (prepare basic data dict, no actual cmfs)
from .cmf import *
__all__ = cmf.__all__


# Load spectral module:
from .spectral import *
__all__ += spectral.__all__

## Set xyzbar in _CMF dict (note that any missing wavelength data is filled in with nan's):
for i, cmf_type in enumerate(_CMF['types']): # store all in single nested dict
    _CMF[cmf_type]['bar'] =  xyzbar(cieobs = cmf_type, scr = 'file', kind = 'np',extrap_values = (np.nan,np.nan))

# add 'all' key to _CIE_ILLUMINANTS that  contains all CIE_ILLUMINANTS in a stack:
_CIE_ILLUMINANTS['all'] = np.vstack((_CIE_E[0,:],np.array([cie_interp(_CIE_ILLUMINANTS[x],_CIE_E[0,:],kind = 'spd', extrap_values = 'ext')[1,:] for x in _CIE_ILLUMINANTS['types'] if 'series' not in x])))
_CIE_S = _CIE_ILLUMINANTS

# load additional functions to determine daylight loci for specific CMF sets:
from .illuminants import *
__all__ += illuminants.__all__

__all__ += ['_R_PATH','_S_PATH', '_CIE_ILLUMINANTS','_CIE_S',
            '_CIE_E', '_CIE_D65', '_CIE_A', '_CIE_B', '_CIE_C', '_CIE_F4','_CIE_L41',
            '_CIE_F_SERIES', '_CIE_F3_SERIES','_CIE_HP_SERIES','_CIE_LED_SERIES',
            '_IESTM3015','_IESTM3018',
            '_CIE_GLASS_ID', '_CRI_RFL','_RFL', '_MUNSELL']