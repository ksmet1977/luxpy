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
###############################################################################
# Module for color rendition calculations and graphical output
###############################################################################

# --- init_cri_defaults_database.py -------------------------------------------
# _CRI_TYPE_DEFAULT: Default cri_type.

# _CRI_DEFAULTS: default parameters for color fidelity and gamut area metrics 
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
            
# process_cri_type_input(): load a cri_type dict but overwrites any keys that 
            have a non-None input in calling function

            
# --- DE_scalers.py ------------ ----------------------------------------------
# linear_scale(): Linear color rendering index scale from CIE13.3-1974/1995:  
                     Rfi,a = 100 - c1*DEi,a. (c1 = 4.6)

# log_scale(): Log-based color rendering index scale from Davis & Ohno (2009):  
                    Rfi,a = 10 * ln(exp((100 - c1*DEi,a)/10) + 1)

# psy_scale(): Psychometric based color rendering index scale from Smet et al. (2013):  
                    Rfi,a = 100 * (2 / (exp(c1*abs(DEi,a)**(c2) + 1))) ** c3


# --- helpers.py --------------- ----------------------------------------------         
# gamut_slicer(): Slices the gamut in nhbins slices and provides normalization 
                    of test gamut to reference gamut.

# jab_to_rg(): Calculates gamut area index, Rg.

# jab_to_rhi(): Calculate hue bin measures: 
                    Rfhi (local (hue bin) color fidelity)
                    Rcshi (local chroma shift) 
                    Rhshi (local hue shift)

# spd_to_jab_t_r(): Calculates jab color values for a sample set illuminated 
                    with test source and its reference illuminant.
                  
# spd_to_rg(): Calculates the color gamut index of spectral data 
                for a sample set illuminated with test source (data) 
                with respect to some reference illuminant.

# spd_to_DEi(): Calculates color difference (~fidelity) of spectral data 
                between sample set illuminated with test source (data) 
                and some reference illuminant.

# optimize_scale_factor(): Optimize scale_factor of cri-model in cri_type 
                            such that average Rf for a set of light sources is 
                            the same as that of a target-cri (default: 'ciera')

# spd_to_cri(): Calculates the color rendering fidelity index 
                (CIE Ra, CIE Rf, IES Rf, CRI2012 Rf) of spectral data. 
                Can also output Rg, Rfhi, Rcshi, Rhshi, cct, duv, ...

            
# --- RfRg_indices.py----------------------------------------------------------  
# wrapper functions for fidelity type metrics:
      spd_to_ciera(): CIE 13.3 1995 version 
      spd_to_ciera_133_1995(): CIE 13.3 1995 version
      spd_to_cierf(): latest version
      spd_to_cierf_224_2017(): CIE224-2017 version
      
      spd_to_iesrf(): latest version
      spd_to_iesrf_tm30(): latest version
      spd_to_iesrf_tm30_15(): TM30-15 version
      spd_to_iesrf_tm30_18(): TM30-18 version
      
      spd_to_cri2012()
      spd_to_cri2012_hl17()
      spd_to_cri2012_hl1000()
      spd_to_cri2012_real210()

# wrapper functions for gamuta area metrics:
      spd_to_iesrg(): latest version
      spd_to_iesrg_tm30(): latest version
      spd_to_iesrg_tm30_15(): TM30-15 version
      spd_to_iesrg_tm30_18(): TM30-18 version


# --- mcri.py ----------------------------------------------------------------  
# spd_to_mcri(): Calculates the memory color rendition index, Rm:  K. A. G. Smet, W. R. Ryckaert, M. R. Pointer, G. Deconinck, and P. Hanselaer, (2012) “A memory colour quality metric for white light sources,” Energy Build., vol. 49, no. C, pp. 216–225.

# --- cqs.py ------------------------------------------------------------------  
# spd_to_cqs(): versions 7.5 and 9.0 are supported.  W. Davis and Y. Ohno, “Color quality scale,” (2010), Opt. Eng., vol. 49, no. 3, pp. 33602–33616.   


#------------------------------------------------------------------------------
#
# plot_hue_bins(): Makes basis plot for Color Vector Graphic (CVG).
#
# plot_ColorVectorGraphic(): Plots Color Vector Graphic (see IES TM30).
#
# plot_cri_graphics(): Plot graphical information on color rendition properties.
#
#------------------------------------------------------------------------------
#
#
# Module for VectorField and Pixelation CRI models.
# see ?luxpy.cri.VFPX
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
Created on Mon Apr  2 03:35:33 2018

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from .colorrendition import *