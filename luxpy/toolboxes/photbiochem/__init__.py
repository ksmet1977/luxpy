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
Module for calculating CIE (TN003:2015) photobiological quantities
==================================================================
(Eesc, Eemc, Eelc, Eez, Eer and Esc, Emc, Elc, Ez, Er)

+---------------+----------------+---------------------+---------------------+----------+-------------+
| Photoreceptor |  Photopigment  | Spectral efficiency | Quantity            | Q-symbol | Unit symbol |
|               |  (label, α)    | sα(λ)               | (α-opic irradiance) | (Ee,α)   |             |
+===============+================+=====================+=====================+==========+=============+
|    s-cone     | photopsin (sc) |       cyanolabe     |      cyanopic       |   Ee,sc  |    W.m−2    |
+---------------+----------------+---------------------+---------------------+----------+-------------+
|    m-cone     | photopsin (mc) |       chlorolabe    |      chloropic      |   Ee,mc  |    W.m−2    |
+---------------+----------------+---------------------+---------------------+----------+-------------+
|    l-cone     | photopsin (lc) |       erythrolabe   |      erythropic     |   Ee,lc  |    W.m−2    |
+---------------+----------------+---------------------+---------------------+----------+-------------+
|    ipRGC      | melanopsin (z) |       melanopic     |      melanopic      |   Ee,z   |    W.m−2    |
+---------------+----------------+---------------------+---------------------+----------+-------------+
|    rod        | rhodopsin (r)  |       rhodopic      |      rhodopic       |   Ee,r   |    W.m−2    |
+---------------+----------------+---------------------+---------------------+----------+-------------+


| CIE recommends that the α-opic irradiance is determined by convolving the spectral
| irradiance, Ee,λ(λ) (W⋅m−2), for each wavelength, with the action spectrum, sα(λ), 
| where sα(λ) is normalized to one at its peak:
| 
|    Ee,α = ∫ Ee,λ(λ) sα(λ) dλ 
|
| where the corresponding units are W⋅m−2 in each case. 
| 
| The equivalent luminance is calculated as:
|     
|     E,α = Km ⋅ ∫ Ee,λ(λ) sα(λ) dλ ⋅ ∫ V(λ) dλ / ∫ sα(λ) dλ
| 
| To avoid ambiguity, the weighting function used must be stated, so, for example, 
| cyanopic refers to the cyanopic irradiance weighted using 
| the s-cone or ssc(λ) spectral efficiency function.

 :_PHOTORECEPTORS: ['l-cone', 'm-cone','s-cone', 'rod', 'iprgc']
 :_Ee_SYMBOLS: ['Ee,lc','Ee,mc', 'Ee,sc','Ee,r',  'Ee,z']
 :_E_SYMBOLS: ['E,lc','E,mc', 'E,sc','E,r',  'E,z']
 :_Q_SYMBOLS: ['Q,lc','Q,mc', 'Q,sc','Q,r',  'Q,z']
 :_Ee_UNITS: ['W⋅m−2'] * 5
 :_E_UNITS: ['lux'] * 5
 :_Q_UNITS: ['photons/m2/s'] * 5 
 :_QUANTITIES: | list with actinic types of irradiance, illuminance
               | ['erythropic', 
               |  'chloropic',
               |  'cyanopic',
               |  'rhodopic',
               |  'melanopic'] 
 
 :_ACTIONSPECTRA: ndarray with alpha-actinic action spectra. (stored in file:
                  './data/cie_tn003_2015_SI_action_spectra.dat')

 :spd_to_aopicE(): Calculate alpha-opic irradiance (Ee,α) and equivalent 
                   luminance (Eα) values for the l-cone, m-cone, s-cone, 
                   rod and iprgc (α) photoreceptor cells following 
                   CIE technical note TN 003:2015.

References:
      1. `CIE-TN003:2015 (2015). 
      Report on the first international workshop on 
      circadian and neurophysiological photometry, 2013 
      (Vienna, Austria).
      <http://www.cie.co.at/publications/report-first-international-workshop-circadian-and-neurophysiological-photometry-2013>`_
      (http://files.cie.co.at/785_CIE_TN_003-2015.pdf)
      

Module for calculation of cyanosis index (AS/NZS 1680.2.5:1997)
===============================================================
 
 :_COI_OBS: Default CMF set for calculations
 :_COI_CSPACE: Default color space (CIELAB)
 :_COI_RFL_BLOOD: ndarray with reflectance spectra of 100% and 50% 
                   oxygenated blood
 :spd_to_COI_ASNZS1680: Calculate the Cyanosis Observartion Index (COI) 
                        [ASNZS 1680.2.5-1995] 

Reference:
    AS/NZS1680.2.5 (1997). INTERIOR LIGHTING PART 2.5: HOSPITAL AND MEDICAL TASKS.

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from .cie_tn003_2015 import *
__all__ = cie_tn003_2015.__all__

from .ASNZS_1680_2_5_1997_COI import *
__all__ += ASNZS_1680_2_5_1997_COI.__all__

from .circadian_CS_CLa_lrc import *
__all__ += circadian_CS_CLa_lrc.__all__
