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
cri: sub-package suppporting color rendition calculations (colorrendition.py)
=============================================================================

utils/init_cri_defaults_database.py
-----------------------------------

 :_CRI_TYPE_DEFAULT: Default cri_type.

 :_CRI_DEFAULTS: default parameters for color fidelity and gamut area metrics 
                 (major dict has 9 keys (04-Jul-2017): 
                 sampleset [str/dict], 
                 ref_type [str], 
                 cieobs [str], 
                 avg [fcn handle], 
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


utils/graphics.py
-----------------

 :plot_hue_bins(): Makes basis plot for Color Vector Graphic (CVG).

 :plot_ColorVectorGraphic(): Plots Color Vector Graphic (see IES TM30).


indices/indices.py
------------------

 :wrapper_functions_for_fidelity_type_metrics:
      | spd_to_ciera(): CIE 13.3 1995 version 
      | spd_to_ciera_133_1995(): CIE 13.3 1995 version
      | spd_to_cierf(): latest version
      | spd_to_cierf_224_2017(): CIE224-2017 version

      | spd_to_iesrf(): latest version
      | spd_to_iesrf_tm30(): latest version
      | spd_to_iesrf_tm30_15(): TM30-15 version
      | spd_to_iesrf_tm30_18(): TM30-18 version

      | spd_to_cri2012()
      | spd_to_cri2012_hl17()
      | spd_to_cri2012_hl1000()
      | spd_to_cri2012_real210()

 :wrapper_functions_for_gamut_area_metrics:
      | spd_to_iesrg(): latest version
      | spd_to_iesrg_tm30(): latest version
      | spd_to_iesrg_tm30_15(): TM30-15 version
      | spd_to_iesrg_tm30_18(): TM30-18 version


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


iestm30/graphics.py
-------------------
                       
 :spd_to_ies_tm30_metrics(): Calculates IES TM30 metrics from spectral data
 
 :plot_cri_graphics(): Plots graphical information on color rendition 
                       properties based on spectral data input or dict with 
                       pre-calculated measures.
                       
 :_tm30_process_spd(): Calculate all required parameters for plotting from spd using cri.spd_to_cri()

 :plot_tm30_cvg(): Plot TM30 Color Vector Graphic (CVG).
 
 :plot_tm30_Rfi(): Plot Sample Color Fidelity values (Rfi).
 
 :plot_tm30_Rxhj(): Plot Local Chroma Shifts (Rcshj), Local Hue Shifts (Rhshj) and Local Color Fidelity values (Rfhj).

 :plot_tm30_Rcshj(): Plot Local Chroma Shifts (Rcshj).

 :plot_tm30_Rhshj(): Plot Local Hue Shifts (Rhshj).

 :plot_tm30_Rfhj(): Plot Local Color Fidelity values (Rfhj).

 :plot_tm30_spd(): Plot test SPD and reference illuminant, both normalized to the same luminous power.

 :plot_tm30_report(): Plot a figure with an ANSI/IES-TM30 color rendition report.
 
 
 :plot_cri_graphics(): Plots graphical information on color rendition 
                       properties based on spectral data input or dict with 
                       pre-calculated measures (cusom design). 
                       Includes Metameric uncertainty index Rt and vector-fields
                       of color rendition shifts.


iestm30/metrics.py
------------------

:spd_to_ies_tm30_metrics(): Calculates IES TM30 metrics from spectral data + Metameric Uncertainty + Vector Fields


iestm30/metrics_fast.py
-----------------------

 :_cri_ref(): Calculate multiple reference illuminant spectra based on ccts for color rendering index calculations.

 :_xyz_to_jab_cam02ucs(): Calculate CAM02-UCS J'a'b' coordinates from xyz tristimulus values of sample and white point.

 :spd_tom_tm30(): Calculate tm30 measures from spd.
 
 * Created for faster spectral optimization based on ANSI/IES-TM30 measures

 
VFPX
----

 :Module_for_VectorField_and_Pixelation_CRI models.
  
 * see ?luxpy.cri.VFPX


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from .utils.DE_scalers import linear_scale, log_scale, psy_scale

from .utils.helpers import (_get_hue_bin_data, spd_to_jab_t_r, spd_to_rg,
                            spd_to_DEi, optimize_scale_factor, spd_to_cri,
                            _hue_bin_data_to_rxhj, _hue_bin_data_to_rfi, 
                            _hue_bin_data_to_rg)

from .indices.indices import *

from .utils.graphics import *

from .VFPX import VF_PX_models as VFPX


from .iestm30.metrics import spd_to_ies_tm30_metrics
from .iestm30.graphics import (_tm30_process_spd,plot_tm30_cvg,
                                             plot_tm30_Rfi,plot_tm30_Rxhj,
                                             plot_tm30_Rcshj, plot_tm30_Rhshj,
                                             plot_tm30_Rfhj, plot_tm30_spd,
                                             plot_tm30_report,spd_to_tm30_report,
                                             plot_cri_graphics)
from .iestm30.metrics_fast import spd_to_tm30 as spd_to_tm30_fast
from .iestm30.metrics_fast import _cri_ref as cri_ref_fast
from .iestm30.metrics_fast import _xyz_to_jab_cam02ucs as xyz_to_jab_cam02ucs_fast


# .utils/DE_scalers:
__all__ = ['linear_scale', 'log_scale', 'psy_scale']


# .utils/helpers:
__all__ += ['_get_hue_bin_data','spd_to_jab_t_r','spd_to_rg', 'spd_to_DEi', 
           'optimize_scale_factor','spd_to_cri',
           '_hue_bin_data_to_rxhj', '_hue_bin_data_to_rfi', '_hue_bin_data_to_rg']


# .utils/indices:
__all__ += ['spd_to_ciera', 'spd_to_cierf',
           'spd_to_ciera_133_1995','spd_to_cierf_224_2017']
__all__ += ['spd_to_iesrf','spd_to_iesrg',
           'spd_to_iesrf_tm30','spd_to_iesrg_tm30',
           'spd_to_iesrf_tm30_15','spd_to_iesrg_tm30_15',
           'spd_to_iesrf_tm30_18','spd_to_iesrg_tm30_18']
__all__ += ['spd_to_cri2012','spd_to_cri2012_hl17','spd_to_cri2012_hl1000','spd_to_cri2012_real210']
__all__ += ['spd_to_mcri']
__all__ += ['spd_to_cqs']


# .utils/graphics:
__all__ += ['plot_hue_bins','plot_ColorVectorGraphic']


# VF_PX_models:
__all__ += ['VFPX']


# .iestm30/metrics:
__all__ += ['spd_to_ies_tm30_metrics']

# .iestm30/graphics:
__all__ += ['_tm30_process_spd','plot_tm30_cvg','plot_tm30_Rfi',
           'plot_tm30_Rxhj','plot_tm30_Rcshj', 'plot_tm30_Rhshj', 
           'plot_tm30_Rfhj', 'plot_tm30_spd',
           'plot_tm30_report','spd_to_tm30_report',
           'plot_cri_graphics']

# .iestm30/metrics_fast:
__all__ += ['spd_to_tm30_fast','cri_ref_fast','xyz_to_jab_cam02ucs_fast']