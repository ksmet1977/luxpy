# -*- coding: utf-8 -*-
"""
Module with Smet et al. (2018) neutral white loci
=================================================
 
 :_UW_NEUTRALITY_PARAMETERS_SMET2014: dict with parameters of the unique white models in Smet et al. (2014)

 :xyz_to_neutrality_smet2018(): Calculate degree of neutrality using the unique white model in Smet et al. (2014) or the normalized (max = 1) degree of chromatic adaptation model from Smet et al. (2017).

 :cct_to_neutral_loci_smet2018():  Calculate the most neutral appearing Duv10 in and the degree of neutrality for a specified CCT using the models in Smet et al. (2018).
 
References
----------
    1. `Smet, K. A. G. (2018). 
    Two Neutral White Illumination Loci Based on Unique White Rating and Degree of Chromatic Adaptation. 
    LEUKOS, 14(2), 55–67.  
    <https://doi.org/10.1080/15502724.2017.1385400>`_
    
    2. `Smet, K., Deconinck, G., & Hanselaer, P., (2014), 
    Chromaticity of unique white in object mode. 
    Optics Express, 22(21), 25830–25841. 
    <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-22-21-25830>`_
    
    3. `Smet, K.A.G.*, Zhai, Q., Luo, M.R., Hanselaer, P., (2017), 
    Study of chromatic adaptation using memory color matches, 
    Part II: colored illuminants, 
    Opt. Express, 25(7), pp. 8350-8365.
    <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-7-8350&origin=search)>`_

Added August 02, 2019.
"""

from .smet_white_loci import *
__all__ = smet_white_loci.__all__




