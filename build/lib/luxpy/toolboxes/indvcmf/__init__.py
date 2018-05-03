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
Module for Individual Observer lms-CMFs (Asano, 2016)
=====================================================
    
 :_INDVCMF_DATA_PATH: path to data files
 
 :_INDVCMF_DATA: Dict with required data
 
 :_INDVCMF_STD_DEV_ALL_PARAM: Dict with std. dev. model parameters
 
 :_INDVCMF_CATOBSPFCTR: Categorical observer parameters.
 
 :_INDVCMF_M_10d: xyz to 10° lms conversion matrix.
 
 :_WL_CRIT: critical wavelength above which interpolation of S-cone data fails.
 
 :_WL: wavelengths of spectral data.

    
 :cie2006cmfsEx(): Generate Individual Observer CMFs (cone fundamentals) 
                   based on CIE2006 cone fundamentals and published literature 
                   on observer variability in color matching and 
                   in physiological parameters.

 :getMonteCarloParam(): Get dict with normally-distributed physiological 
                        factors for a population of observers.
                            
 :getUSCensusAgeDist(): Get US Census Age Distribution

 :genMonteCarloObs(): Monte-Carlo generation of individual observer 
                      color matching functions (cone fundamentals) for a
                      certain age and field size.

 :getCatObs(): Generate cone fundamentals for categorical observers.

 :get_lms_to_xyz_matrix(): Calculate lms to xyz conversion matrix for a 
                           specific field size.
                            
 :lmsb_to_xyzb(): Convert from LMS cone fundamentals to XYZ CMF.

 :add_to_cmf_dict(): Add set of cmfs to _CMF dict.
 


References
----------
 1. `Asano Y, Fairchild MD, and Blondé L (2016). 
 Individual Colorimetric Observer Model. 
 PLoS One 11, 1–19. 
 <http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0145671>`_
 
 2. `Asano Y, Fairchild MD, Blondé L, and Morvan P (2016). 
 Color matching experiment for highlighting interobserver variability. 
 Color Res. Appl. 41, 530–539. 
 <https://onlinelibrary.wiley.com/doi/abs/10.1002/col.21975>`_
 
 3. `CIE, and CIE (2006). 
 Fundamental Chromaticity Diagram with Physiological Axes - Part I 
 (Vienna: CIE). 
 <http://www.cie.co.at/publications/fundamental-chromaticity-diagram-physiological-axes-part-1>`_ 
 
 4. `Asano's Individual Colorimetric Observer Model 
 <https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php>`_

 
Note
----
Port of Matlab code from:
https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php
(Accessed April 20, 2018)           

"""
from .individual_observer_cmf_model import *
__all__ = individual_observer_cmf_model.__all__