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
###################################################################################################
# Module for calculating CIE (TN003:2015) photobiological quantities: 
#  Eesc, Eemc, Eelc, Eez, Eer and Esc, Emc, Elc, Ez, Er
###################################################################################################

+---------------------------------------------------------------------------------------------------+
|Photoreceptor|  Photopigment  | Spectral efficiency |      Quantity       | Q-symbol | Unit symbol |
|             |   (label, α)   |        sα(λ)        | (α-opic irradiance) |  (Ee,α)  |             |
+---------------------------------------------------------------------------------------------------+
|   s-cone    | photopsin (sc) |       cyanolabe     |      cyanopic       |   Ee,sc  |    W⋅m−2    |
|   m-cone    | photopsin (mc) |       chlorolabe    |      chloropic      |   Ee,mc  |    W⋅m−2    |
|   l-cone    | photopsin (lc) |       erythrolabe   |      erythropic     |   Ee,lc  |    W⋅m−2    |
|   ipRGC     | melanopsin (z) |       melanopic     |      melanopic      |   Ee,z   |    W⋅m−2    |
|    rod      | rhodopsin (r)  |        rhodopic     |      rhodopic       |   Ee,r   |    W⋅m−2    |
+---------------------------------------------------------------------------------------------------+

CIE recommends that the α-opic irradiance is determined by convolving the spectral
irradiance, Ee,λ(λ) (W⋅m−2), for each wavelength, with the action spectrum, sα(λ), 
where sα(λ) is normalized to one at its peak:

    Ee,α = ∫ Ee,λ(λ) sα(λ) dλ 

where the corresponding units are W⋅m−2 in each case. 

The equivalent luminance is calculated as:
    
    E,α = Km ⋅ ∫ Ee,λ(λ) sα(λ) dλ ⋅ ∫ V(λ) dλ / ∫ sα(λ) dλ

To avoid ambiguity, the weighting function used must be stated, so, for example, 
cyanopic refers to the cyanopic irradiance weighted using 
the s-cone or ssc(λ) spectral efficiency function.
----------------------------------------------------------------------------------------------------


----------------------------------------------------------------------------------------------------

Created on Tue Apr 17 12:25:29 2018

@author: kevin.smet
"""
from .cie_tn003_2015 import *