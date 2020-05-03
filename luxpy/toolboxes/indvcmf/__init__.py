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
Module for Individual Observer lms-CMFs (Asano, 2016 and CIE TC1-97)
====================================================================
    
 :_DATA_PATH: path to data files
 
 :_DATA: Dict with required data
 
 :_DSRC_STD_DEF: default data source for stdev of physiological data ('matlab', 'germany')
 
 :_DSRC_LMS_ODENS_DEF: default data source for lms absorbances and optical densities ('asano', 'cietc197')
 
 :_LMS_TO_XYZ_METHOD: default method to calculate lms to xyz conversion matrix ('asano', 'cietc197')
 
 :_WL_CRIT: critical wavelength above which interpolation of S-cone data fails.
 
 :_WL: default wavelengths of spectral data in INDVCMF_DATA.
 
 :load_database(): Load a database with parameters and data required by the Asano model.
 
 :init():   Initialize: load database required for Asano Individual Observer Model 
            into the default _DATA dict and set some options for rounding, 
            sign. figs and chopping small value to zero; for source data to use for 
            spectral data for LMS absorp. and optical densities, ... 
            
 :query_state(): print current settings for global variables.
 
 :compute_cmfs(): Generate Individual Observer CMFs (cone fundamentals) 
                  based on CIE2006 cone fundamentals and published literature 
                  on observer variability in color matching and 
                  in physiological parameters (Use of Asano optical data and model; 
                  or of CIE TC1-91 data and 'variability'-extended model possible).
 
 :cie2006cmfsEx(): Generate Individual Observer CMFs (cone fundamentals) 
                   based on CIE2006 cone fundamentals and published literature 
                   on observer variability in color matching and 
                   in physiological parameters. (Use of Asano optical data and model; 
                   or of CIE TC1-91 data and 'variability'-extended model possible)
 
 :getMonteCarloParam(): Get dict with normally-distributed physiological 
                        factors for a population of observers.
                            
 :getUSCensusAgeDist(): Get US Census Age Distribution
 
 :genMonteCarloObs(): Monte-Carlo generation of individual observer 
                      color matching functions (cone fundamentals) for a
                      certain age and field size.
 
 :getCatObs(): Generate cone fundamentals for categorical observers.
 
 :get_lms_to_xyz_matrix(): Calculate lms to xyz conversion matrix for a specific field 
                           size determined as a weighted combination of the 2° and 10° matrices.
 
 :lmsb_to_xyzb(): Convert from LMS cone fundamentals to XYZ CMFs using conversion
                  matrix determined as a weighted combination of the 2° and 10° matrices.
 
 :add_to_cmf_dict(): Add set of cmfs to _CMF dict.
 
 :plot_cmfs(): Plot cmf set.

References
----------
 1. `Asano Y, Fairchild MD, and Blondé L (2016). 
 Individual Colorimetric Observer Model. 
 PLoS One 11, 1–19. 
 <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0145671>`_
 
 2. `Asano Y, Fairchild MD, Blondé L, and Morvan P (2016). 
 Color matching experiment for highlighting interobserver variability. 
 Color Res. Appl. 41, 530–539. 
 <https://onlinelibrary.wiley.com/doi/abs/10.1002/col.21975>`_
 
 3. `CIE TC1-36 (2006). 
 Fundamental Chromaticity Diagram with Physiological Axes - Part I 
 (Vienna: CIE). 
 <https://www.cie.co.at/publications/fundamental-chromaticity-diagram-physiological-axes-part-1>`_ 
 
 4. `Asano's Individual Colorimetric Observer Model 
 <https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php>`_
 
 5. `CIE TC1-97 cmf functions python code developed by Ivar Farup and Jan Hendrik Wold.
 <https://github.com/ifarup/ciefunctions>`_
 
Notes
-----
    1. Port of Matlab code from: 
    https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php
    (Accessed April 20, 2018)  
    2. Adjusted/extended following CIE TC1-97 Python code (and data):
    github.com/ifarup/ciefunctions (Copyright (C) 2012-2017 Ivar Farup and Jan Henrik Wold)     
    (Accessed Dec 18, 2019)

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from .individual_observer_cmf_model import *
__all__ = individual_observer_cmf_model.__all__