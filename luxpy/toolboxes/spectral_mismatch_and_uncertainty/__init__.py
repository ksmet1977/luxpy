# -*- coding: utf-8 -*-
"""
Toolbox for spectral mismatch and measurement uncertainty calculations
======================================================================


spectral_mismatch_and_uncertainty/detector_spectral_mismatch.py
---------------------------------------------------------------

 :f1prime(): Determine the f1prime spectral mismatch index.
 
 :get_spectral_mismatch_correct_factors(): Determine the spectral mismatch factors.


Reference
---------
    1. Krüger, U. et al. GENERAL V(λ) MISMATCH - INDEX HISTORY, CURRENT STATE, NEW IDEAS (TechnoTeam)
    
===============================================================================

Created on Tue Aug 31 10:46:02 2021

@author: ksmet1977 [at] gmail.com
"""

from .detector_spectral_mismatch import f1prime, get_spectral_mismatch_correction_factors
__all__ = ['f1prime','get_spectral_mismatch_correction_factors'] 

