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
Module for color rendition and color quality metrics
====================================================

utils/init_cri_defaults_database.py
-----------------------------------

 :_CRI_TYPE_DEFAULT: Default cri_type.

 :_CRI_DEFAULTS: default parameters for color fidelity and gamut area metrics 
                 (major dict has 12 keys (18-Dec-2024): 
                 sampleset [str/dict], 
                 ref_type [str], 
                 calculation_wavelength_range [list],
                 cieobs [Dict], 
                 cct_mode [str],
                 avg [fcn handle], 
                 rf_from_avg_rounded_rfi [bool],
                 scale [dict], 
                 cspace [dict], 
                 catf [dict], 
                 rg_pars [dict], 
                 cri_specific_pars [dict])
                 
                * Supported cri-types:
                    * 'ciera','ciera-8','ciera-14','cierf',
                    * 'iesrf','iesrf-tm30-15','iesrf-tm30-18',
                    * 'cri2012','cri2012-hl17','cri2012-hl1000','cri2012-real210',
                    * 'mcri',
                    * 'cqs-v7.5','cqs-v9.0'
                    * 'fci'
                    * 'thornton_cpi'

 :process_cri_type_input(): load a cri_type dict but overwrites any keys that 
                            have a non-None input in calling function.

            
utils/DE_scalers.py
-------------------

 :linear_scale(): | Linear color rendering index scale from CIE13.3-1974/1995:
                  | Rfi,a = 100 - c1*DEi,a. (c1 = 4.6)

 :log_scale(): | Log-based color rendering index scale from Davis & Ohno (2009):
               | Rfi,a = 10 * ln(exp((100 - c1*DEi,a)/10) + 1)

 :psy_scale(): | Psychometric based color rendering index scale from Smet et al. (2013):
               | Rfi,a = 100 * (2 / (exp(c1*abs(DEi,a)**(c2) + 1))) ** c3


utils/helpers.py
----------------

 :_get_hue_bin_data(): Slice gamut spanned by the sample jabt, jabr and calculate hue-bin data.

 :_hue_bin_data_to_rxhj(): Calculate hue bin measures: Rcshj, Rhshj, Rfhj, DEhj
     
 :_hue_bin_data_to_rfi(): Get sample color differences DEi and calculate color fidelity values Rfi.

 :_hue_bin_data_to_rg():  Calculates gamut area index, Rg.

 :spd_to_jab_t_r(): Calculates jab color values for a sample set illuminated
                    with test source and its reference illuminant.

 :spd_to_rg(): Calculates the color gamut index of spectral data 
               for a sample set illuminated with test source (data) 
               with respect to some reference illuminant.

 :spd_to_DEi(): Calculates color difference (~fidelity) of spectral data 
                between sample set illuminated with test source (data) 
                and some reference illuminant.

 :optimize_scale_factor(): Optimize scale_factor of cri-model in cri_type 
                           such that average Rf for a set of light sources is 
                           the same as that of a target-cri (default: 'ciera')

 :spd_to_cri(): Calculates the color rendering fidelity index 
                (CIE Ra, CIE Rf, IES Rf, CRI2012 Rf) of spectral data. 
                Can also output Rg, Rfhi, Rcshi, Rhshi, cct, duv, ...


            
indices/ciewrappers.py & ieswrappers.py
---------------------------------------  

 :wrapper_functions_for_fidelity_type_metrics:
      | spd_to_ciera(): CIE 13.3 1995 version 
      | spd_to_ciera_133_1995(): CIE 13.3 1995 version
      | spd_to_cierf(): latest version
      | spd_to_cierf_224_2017(): CIE224-2017 version

      | spd_to_iesrf(): latest version
      | spd_to_iesrf_tm30(): latest version
      | spd_to_iesrf_tm30_15(): TM30-15 version
      | spd_to_iesrf_tm30_18(): TM30-18 version
      | spd_to_iesrf_tm30_20(): TM30-20 version (= TM30-18)

      | spd_to_cri2012()
      | spd_to_cri2012_hl17()
      | spd_to_cri2012_hl1000()
      | spd_to_cri2012_real210()

 :wrapper_functions_for_gamut_area_metrics:
      | spd_to_iesrg(): latest version
      | spd_to_iesrg_tm30(): latest version
      | spd_to_iesrg_tm30_15(): TM30-15 version
      | spd_to_iesrg_tm30_18(): TM30-18 version
      | spd_to_iesrg_tm30_20(): TM30-20 version (= TM30-18)


indices/mcri.py
---------------

 :spd_to_mcri(): | Calculates the memory color rendition index, Rm:  
                 | K. A. G. Smet, W. R. Ryckaert, M. R. Pointer, G. Deconinck, and P. Hanselaer, (2012) 
                 | “A memory colour quality metric for white light sources,” 
                 | Energy Build., vol. 49, no. C, pp. 216–225.

indices/cqs.py
--------------

 :spd_to_cqs(): | versions 7.5 and 9.0 are supported.  
                | W. Davis and Y. Ohno, 
                | “Color quality scale,” (2010), 
                | Opt. Eng., vol. 49, no. 3, pp. 33602–33616.
                            
indices/fci.py
--------------

 :spd_to_fci(): | Calculates the Feeling of Contrast Index, FCI:  
                | Hashimoto, K., Yano, T., Shimizu, M., & Nayatani, Y. (2007). 
                | New method for specifying color-rendering properties of light sources 
                | based on feeling of contrast. 
                | Color Research and Application, 32(5), 361–371. 
                
indices/thorntoncpi.py
----------------------

 :spd_to_thornton_cpi(): | Calculates Thornton's Color Preference Index, CPI:  
                         | Thornton, W. A. (1974). A Validation of the Color-Preference Index.
                         | Journal of the Illuminating Engineering Society, 4(1), 48–52. 
        

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import _CRI_RFL

#from ..utils.DE_scalers import linear_scale, log_scale, psy_scale

from ..utils.init_cri_defaults_database import _CRI_TYPE_DEFAULT, _CRI_DEFAULTS

from ..utils.helpers import (_get_hue_bin_data, spd_to_jab_t_r, spd_to_rg,
                            spd_to_DEi, optimize_scale_factor, spd_to_cri,
                            _hue_bin_data_to_rxhj, _hue_bin_data_to_rfi, 
                            _hue_bin_data_to_rg)
from .cie_wrappers import (spd_to_ciera, spd_to_cierf, 
                         spd_to_ciera_133_1995, spd_to_cierf_224_2017)
from .iestm30_wrappers import (spd_to_iesrf, spd_to_iesrg, 
                         spd_to_iesrf_tm30, spd_to_iesrg_tm30,
                         spd_to_iesrf_tm30_15, spd_to_iesrg_tm30_15, 
                         spd_to_iesrf_tm30_18, spd_to_iesrg_tm30_18,
                         spd_to_iesrf_tm30_20, spd_to_iesrg_tm30_20)
from .cri2012 import spd_to_cri2012, spd_to_cri2012_hl17, spd_to_cri2012_hl1000, spd_to_cri2012_real210
from .mcri import _MCRI_DEFAULTS, spd_to_mcri
from .cqs import _CQS_DEFAULTS, spd_to_cqs
from .fci import spd_to_fci
from .thorntoncpi import spd_to_thornton_cpi



__all__  = ['_CRI_RFL','_CRI_TYPE_DEFAULT','_CRI_DEFAULTS']

__all__ += ['_get_hue_bin_data','spd_to_jab_t_r','spd_to_rg', 'spd_to_DEi', 
           'optimize_scale_factor','spd_to_cri',
           '_hue_bin_data_to_rxhj', '_hue_bin_data_to_rfi', '_hue_bin_data_to_rg']

__all__ += ['spd_to_ciera', 'spd_to_cierf',
           'spd_to_ciera_133_1995','spd_to_cierf_224_2017']

__all__ += ['spd_to_iesrf','spd_to_iesrg',
           'spd_to_iesrf_tm30','spd_to_iesrg_tm30',
           'spd_to_iesrf_tm30_15','spd_to_iesrg_tm30_15',
           'spd_to_iesrf_tm30_18','spd_to_iesrg_tm30_18',
           'spd_to_iesrf_tm30_20','spd_to_iesrg_tm30_20']

__all__ += ['spd_to_cri2012','spd_to_cri2012_hl17','spd_to_cri2012_hl1000','spd_to_cri2012_real210']

__all__ += ['spd_to_mcri']

__all__ += ['spd_to_cqs']

__all__ += ['spd_to_fci']

__all__ += ['spd_to_thornton_cpi']


# Update _CRI_DEFAULTS from .init_cri_defaults_database (already filled with cie/ies Rf and Rg type metric defaults):
_CRI_DEFAULTS['cqs-v7.5'] = _CQS_DEFAULTS['cqs-v7.5']
_CRI_DEFAULTS['cri_types'].append('cqs-v7.5')
_CRI_DEFAULTS['cqs-v9.0'] = _CQS_DEFAULTS['cqs-v9.0']
_CRI_DEFAULTS['cri_types'].append('cqs-v9.0')

_CRI_DEFAULTS['mcri'] = _MCRI_DEFAULTS
_CRI_DEFAULTS['cri_types'].append('mcri')

_CRI_DEFAULTS['fci'] = {'sampleset' : "_CRI_RFL['fci']['R']"}
_CRI_DEFAULTS['cri_types'].append('fci')

_CRI_DEFAULTS['thornton_cpi'] = {'sampleset' : "_CRI_RFL['cie-13.3-1995']['8']"}
_CRI_DEFAULTS['cri_types'].append('thornton_cpi')

   

