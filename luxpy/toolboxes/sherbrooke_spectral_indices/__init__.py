# -*- coding: utf-8 -*-
"""
Module for the calculation of the Melatonin Suppression Index (MSI), 
the Induced Photosynthesis Index (IPI) and the Star Light Index (SLI)
---------------------------------------------------------------------

 :spd_to_msi(): calculate Melatonin Suppression Index from spectrum.
 
 :spd_to_ipi(): calculate Induced Photosynthesis Index from spectrum.
 
 :spd_to_sli(): calculate Star Light Index  from spectrum.

References: 
    1. Aubé M, Roby J, Kocifaj M (2013) 
    Evaluating Potential Spectral Impacts of Various Artificial Lights on Melatonin Suppression, Photosynthesis, and Star Visibility. 
    PLoS ONE 8(7): e67798
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0067798

Created on Fri Jun 11 13:46:33 2021

@author: ksmet1977 [at] gmail dot com
"""
from .sherbrooke_spectral_indices_2013 import *
__all__ = sherbrooke_spectral_indices_2013.__all__