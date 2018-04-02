# -*- coding: utf-8 -*-
"""

###############################################################################
# Module for color rendition calculations and graphical output
###############################################################################
# _CRI_DEFAULTS: default settings for different color rendition indices: (major dict has 9 keys (04-Jul-2017): sampleset [str/dict], ref_type [str], cieobs [str], avg [fcn handle], scale [dict], cspace [dict], catf [dict], rg_pars [dict], cri_specific_pars [dict])
#               types supported: 'ciera','ciera-8','ciera-14','cierf','iesrf','cri2012','cri2012-hl17', 'cri2012-hl1000','cri2012-real210','cqs-v7.5', 'cqs-v9.0', mcri'
#
# linear_scale():  Linear color rendering index scale from CIE13.3-1974/1995:   Rfi,a = 100 - c1*DEi,a. (c1 = 4.6)
#
# log_scale(): Log-based color rendering index scale from Davis & Ohno (2009):  Rfi,a = 10 * ln(exp((100 - c1*DEi,a)/10) + 1)
#
# psy_scale():  Psychometric based color rendering index scale from CRI2012 (Smet et al. 2013, LRT):  Rfi,a = 100 * (2 / (exp(c1*abs(DEi,a)**(c2) + 1))) ** c3
#
# process_cri_type_input(): load a cri_type dict but overwrites any keys that have a non-None input in calling function
#
# gamut_slicer(): Slices the gamut in nhbins slices and provides normalization of test gamut to reference gamut.
#
# jab_to_rg(): Calculates gamut area index, Rg based on hue-ordered jabt and jabr input (first element must also be last)
#
# spd_to_jab_t_r(): Calculates jab color values for a sample set illuminated with test source and its reference illuminant.
#                   
# spd_to_rg(): Calculates the color gamut index of data (= np.array([[wl,spds]]) (data_axis = 0) for a sample set illuminated with test source (data) with respect to some reference illuminant.
#
# spd_to_DEi(): Calculates color difference (~fidelity) of data (= np.array([[wl,spds]]) (data_axis = 0) between sample set illuminated with test source (data) and some reference illuminant.
#
# optimize_scale_factor(): Optimize scale_factor of cri-model in cri_type such that average Rf for a set of light sources is the same as that of a target-cri (default: 'ciera').
#
# spd_to_cri(): Calculates the color rendering fidelity index (CIE Ra, CIE Rf, IES Rf, CRI2012 Rf) of spectral data. 
#
# wrapper functions for fidelity type metrics:
#     spd_to_ciera(), spd_to_cierf(), spd_to_iesrf(), spd_to_cri2012(), spd_to_cri2012_hl17(), spd_to_cri2012_hl1000(), spd_to_cri2012_real210
#
# wrapper functions for gamuta area metrics:
#      spd_to_iesrf(),
#
# spd_to_mcri(): Calculates the memory color rendition index, Rm:  K. A. G. Smet, W. R. Ryckaert, M. R. Pointer, G. Deconinck, and P. Hanselaer, (2012) “A memory colour quality metric for white light sources,” Energy Build., vol. 49, no. C, pp. 216–225.
#
# spd_to_cqs(): versions 7.5 and 9.0 are supported.  W. Davis and Y. Ohno, “Color quality scale,” (2010), Opt. Eng., vol. 49, no. 3, pp. 33602–33616.   
#
#------------------------------------------------------------------------------
#
# plot_hue_bins(): Makes basis plot for Color Vector Graphic (CVG).
#
# plot_ColorVectorGraphic(): Plots Color Vector Graphic (see IES TM30).
#
# plot_cri_grpahics(): Plot graphical information on color rendition properties.
#------------------------------------------------------------------------------

Created on Mon Apr  2 03:35:33 2018

@author: kevin.smet
"""
from .colorrendition_indices import *
from .colorrendition_graphics import *

__all__ =  ['_CRI_DEFAULTS','linear_scale','log_scale','psy_scale','gamut_slicer','jab_to_rg','spd_to_rg','spd_to_DEi','spd_to_cri']
__all__ += ['spd_to_ciera','spd_to_cierf','spd_to_iesrf','spd_to_iesrg','spd_to_cri2012','spd_to_cri2012_hl17','spd_to_cri2012_hl1000','spd_to_cri2012_real210']
__all__ += ['spd_to_mcri', 'spd_to_cqs']

__all__ += ['plot_hue_bins','plot_ColorVectorGraphic','plot_cri_graphics']
