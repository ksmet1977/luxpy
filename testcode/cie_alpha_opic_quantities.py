# -*- coding: utf-8 -*-
"""
###################################################################################################
# Test script for calculating CIE photobiological quantities: Eesc, Eemc, Eelc, Eez, Eer.
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

   
if __name__ == '__main__':
    import luxpy as lx 
    import numpy as np
    sid = np.vstack((lx.getwlr([378,782,1]),0.1*np.ones((1,lx.getwlr([378,782,1]).shape[0]))))
    #sid= lx.blackbody(3000,wl3=lx.getwlr([378,782,1]))
    
    Ees,Es = lx.ciephotbio.spd_to_aopicE(sid, E = 100, sid_units = 'uW/cm2')
    
    print('Ees:')
    print(Ees)
    print('Es:')
    print(Es)