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
from luxpy import np

# Load cmfs, part 1 (prepare basic data dict, no actual cmfs)
from .cmf import *
__all__ = cmf.__all__


# load spd and rfl data in /spd/:
from .spectral_databases import (_R_PATH, _S_PATH, _CIE_ILLUMINANTS, _CIE_GLASS_ID,
                                 _IESTM3015, _IESTM3018, 
                                 _CIE_E, _CIE_D65, _CIE_A, _CIE_B, _CIE_C, _CIE_F4,
                                 _CIE_F_1_12,_CIE_F3_1_15,_CIE_HP_1_5,_CIE_LED_B1toB5_BH1_RGB1_V1_V2,_CIE_LED,
                                 _CRI_RFL, _RFL, _MUNSELL)

# Load spectral module:
from .spectral import *
__all__ += spectral.__all__

## Set xyzbar in _CMF dict:
for i, cmf_type in enumerate(_CMF['types']): # store all in single nested dict
    _CMF[cmf_type]['bar'] =  xyzbar(cieobs = cmf_type, scr = 'file', kind = 'np')

# add 'all' key to _CIE_ILLUMINANTS that  contains all CIE_ILLUMINANTS in a stack:
_CIE_ILLUMINANTS['all'] = np.vstack((_CIE_E[0,:],np.array([cie_interp(_CIE_ILLUMINANTS[x],_CIE_E[0,:],kind='linear')[1,:] for x in _CIE_ILLUMINANTS['types']])))

__all__ += ['_R_PATH','_S_PATH', '_CIE_ILLUMINANTS', '_CIE_GLASS_ID', 
            '_IESTM3015','_IESTM3018',
            '_CIE_E', '_CIE_D65', '_CIE_A', '_CIE_B', '_CIE_C', '_CIE_F4',
            '_CIE_F_1_12','_CIE_F3_1_15','_CIE_HP_1_5','_CIE_LED_B1toB5_BH1_RGB1_V1_V2','_CIE_LED',
            '_CRI_RFL','_RFL', '_MUNSELL']