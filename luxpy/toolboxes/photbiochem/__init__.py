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
Module for calculating CIE (S026:2018) photobiological quantities
==================================================================
(Eelc, Eemc, Eesc, Eer, Eez, and Elc, Emc, Esc, Er, Ez)

+---------------+----------------+---------------------+---------------------+----------+-------------+
| Photoreceptor |  Photopigment  | Spectral efficiency | Quantity            | Q-symbol | Unit symbol |
|               |  (label, α)    | sα(λ)               | (α-opic irradiance) | (Ee,α)   |             |
+===============+================+=====================+=====================+==========+=============+
|    l-cone     | photopsin (lc) |       erythrolabe   |      erythropic     |   Ee,lc  |    W.m−2    |
+---------------+----------------+---------------------+---------------------+----------+-------------+
|    m-cone     | photopsin (mc) |       chlorolabe    |      chloropic      |   Ee,mc  |    W.m−2    |
+---------------+----------------+---------------------+---------------------+----------+-------------+
|    s-cone     | photopsin (sc) |       cyanolabe     |      cyanopic       |   Ee,sc  |    W.m−2    |
+---------------+----------------+---------------------+---------------------+----------+-------------+
|    rod        | rhodopsin (r)  |       rhodopic      |      rhodopic       |   Ee,r   |    W.m−2    |
+---------------+----------------+---------------------+---------------------+----------+-------------+
|    ipRGC      | melanopsin (z) |       melanopic     |      melanopic      |   Ee,z   |    W.m−2    |
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
                  './data/cie_S026_2018_SI_action_spectra_CIEToolBox_v1.049.dat')

 :spd_to_aopicE(): Calculate alpha-opic irradiance (Ee,α) and equivalent 
                   luminance (Eα) values for the l-cone, m-cone, s-cone, 
                   rod and iprgc (α) photoreceptor cells following 
                   CIE S026:2018.
                   
                   
 :spd_to_aopicEDI(): Calculate alpha-opic equivalent daylight (D65) illuminance (lx)
                     for the l-cone, m-cone, s-cone, rod and iprgc (α) photoreceptor cells.

     
References:
      1. `CIE-S026:E2018 (2018). 
      CIE System for Metrology of Optical Radiation for ipRGC-Influenced Responses to Light 
      (Vienna, Austria).
      <https://cie.co.at/publications/cie-system-metrology-optical-radiation-iprgc-influenced-responses-light-0>`_
      (https://files.cie.co.at/CIE%20S%20026%20alpha-opic%20Toolbox%20User%20Guide.pdf)

     

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



Module for Blue light hazard calculations
=========================================

 :_BLH: Blue Light Hazard function
 
 :spd_to_blh_eff(): Calculate Blue Light Hazard efficacy (K) or efficiency (eta) of radiation.


References:
        1. IEC 62471:2006, 2006, Photobiological safety of lamps and lamp systems.
        2. IEC TR 62778, 2014, Application of IEC 62471 for the assessment of blue light hazard to light sources and luminaires.



.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from .cie_s026_2018 import *
__all__ = cie_s026_2018.__all__

from .ASNZS_1680_2_5_1997_COI import *
__all__ += ASNZS_1680_2_5_1997_COI.__all__

from .circadian_CS_CLa_lrc import *
__all__ += circadian_CS_CLa_lrc.__all__

from .blue_light_hazard_IEC62471_IECTR62778 import *
__all__ += blue_light_hazard_IEC62471_IECTR62778.__all__


