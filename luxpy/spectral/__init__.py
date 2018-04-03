# -*- coding: utf-8 -*-

# Load cmfs, part 1 (prepare basic data dict, no actual cmfs)
from .cmf import *
__all__ = cmf.__all__

# Load spectral module:
from .spectral import *
__all__ += spectral.__all__

## Set xyzbar in _CMF dict:
_CMF['bar'] = {_CMF['types'][i] : (xyzbar(cieobs = _CMF['types'][i], scr = 'file', kind = 'np')) for i in range(len(_CMF['types']))}

# load spd and rfl data in /spd/:
from .spectral_databases import _R_PATH, _S_PATH, _CIE_ILLUMINANTS, _IESTM30, _CRI_RFL, _MUNSELL
__all__ += ['_R_PATH', '_S_PATH', '_CRI_RFL', '_CIE_ILLUMINANTS', '_IESTM30','_MUNSELL'] 

