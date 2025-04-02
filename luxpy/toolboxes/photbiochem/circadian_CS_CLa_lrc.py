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
Module for Circadian Light (CLa) and Stimulus (CS) calculations (LRC)
=====================================================================

:_LRC_CLA_CS_CONST: dict with model parameters and spectral data.

:spd_to_CS_CLa_lrc(): Calculate Circadian Stimulus (CS) 
                        and Circadian Light (CLA: Rea et al 2012, CLA2.0: Rea et al 2021, 2022.
                                             
:CLa_to_CS(): Convert Circadian Light to Circadian Stimulus.


Definitions
-----------

 1. **Circadian Stimulus (CS)** is the calculated effectiveness of the 
spectrally weighted irradiance at the cornea from threshold (CS = 0.1) 
to saturation (CS = 0.7), assuming a fixed duration of exposure of 1 hour.

 2. **Circadian Light (CLA, CLA2.0)** is the irradiance at the cornea weighted 
 to reflect the spectral sensitivity of the human circadian system as measured 
 by acute melatonin suppression after a 1-hour exposure. (see note below on CLA2.0)


References
----------
    1. `LRC Online Circadian stimulus calculator (CLa1.0, 2012)
        <http://www.lrc.rpi.edu/cscalculator/>`_
    
    2. `LRC Excel based Circadian stimulus calculator (CLa1.0, 2012). 
        <http://www.lrc.rpi.edu/resources/CSCalculator_2017_10_03_Mac.xlsm>`_
    
    3. `Rea MS, Figueiro MG, Bierman A, and Hamner R (2012). 
        Modelling the spectral sensitivity of the human circadian system. 
        Light. Res. Technol. 44, 386–396.  
        <https://doi.org/10.1177/1477153511430474>`_
            
    4. `Rea MS, Figueiro MG, Bierman A, and Hamner R (2012). 
        Erratum: Modeling the spectral sensitivity of the human circadian system 
        (Lighting Research and Technology (2012) 44:4 (386-396)). 
        Light. Res. Technol. 44, 516.
        <https://doi.org/10.1177/1477153512467607>`_
        
    5. `Rea, M. S., Nagare, R., & Figueiro, M. G. (2021). 
        Modeling Circadian Phototransduction: Quantitative Predictions of Psychophysical Data. 
        Frontiers in Neuroscience, 15, 44. 
        <https://doi.org/10.3389/fnins.2021.615322>`_
        
    6. `Rea, M. S., Nagare, R., & Figueiro, M. G. (2022). 
        Corrigendum: Modeling Circadian Phototransduction: Quantitative Predictions of Psychophysical Data. 
        Frontiers in Neuroscience, 16. 
        <https://www.frontiersin.org/article/10.3389/fnins.2022.849800>`_
        
    7. `LRC Online Circadian stimulus calculator  (CLa2.0, 2021)
        <https://docs.light-health.org/cscalc>`_

    8. `Github code: LRC Online Circadian stimulus calculator (CLa2.0, accessed Nov. 5, 2021)
        <https://github.com/Light-and-Health-Research-Center/cscalc>`_
Note:
----
    1. The 2005 CLA formulation did not include a physiologically based 
    threshold term for the ipRGC-melanopsin response. The CLA formulation was
    therefore revised to CLA2.0, wherein the ipRGC-melanopsin response is 
    directly modulated by a threshold term involving both rods and cones that, 
    through the AII amacrine neuron, elevates the threshold response of
    the M1 ipRGCs to light. [Rea, 2021]

Also see notes in doc_string of spd_to_CS_CLa_lrc()

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import numpy as np 

from luxpy import _CIE_ILLUMINANTS, getwld, cie_interp
from luxpy.utils import _PKG_PATH, _SEP, getdata

__all__=['_LRC_CLA_CS_CONST','spd_to_CS_CLa_lrc','CLa_to_CS']

# Get 2012 efficiency functions (originally from excel calculator):
_LRC_CLA_CS_EFF_FCN_2012 = getdata(_PKG_PATH + _SEP + 'toolboxes' + _SEP + 'photbiochem' + _SEP  + 'data' + _SEP + 'LRC2012_CS_CLa_efficiency_functions.dat', header = 'infer', verbosity = 0).T

# Get 2021 efficiency functions (original data from github repository, and pre-generated using LRC2021_CS_CLa_efficiency_functions.py):
_LRC_CLA_CS_EFF_FCN_2021 = getdata(_PKG_PATH + _SEP + 'toolboxes' + _SEP + 'photbiochem' + _SEP  + 'data' + _SEP + 'LRC2021_CS_CLa_efficiency_functions.dat', header = 'infer', verbosity = 0).T


_LRC_CLA_CS_CONST = {'CLa_2012' : {'Norm' : 1622, 'k': 0.2616, 'a_b_y':0.6201, 'a_rod' : 3.2347, 'RodSat' : 6.52,\
                           'Vphotl': _LRC_CLA_CS_EFF_FCN_2012[1], 'Vscotl': _LRC_CLA_CS_EFF_FCN_2012[2], \
                           'Vl_mpl': _LRC_CLA_CS_EFF_FCN_2012[3], 'Scl_mpl': _LRC_CLA_CS_EFF_FCN_2012[4],\
                           'Mcl' : _LRC_CLA_CS_EFF_FCN_2012[5], 'WL' : _LRC_CLA_CS_EFF_FCN_2012[0]},\
                  'CLa1.0':   {'Norm' : 1547.9, 'k': 0.2616, 'a_b_y':0.7, 'a_rod' : 3.3, 'RodSat' : 6.5215,\
                           'Vphotl': _LRC_CLA_CS_EFF_FCN_2012[1], 'Vscotl': _LRC_CLA_CS_EFF_FCN_2012[2], \
                           'Vl_mpl': _LRC_CLA_CS_EFF_FCN_2012[3], 'Scl_mpl': _LRC_CLA_CS_EFF_FCN_2012[4],\
                           'Mcl' : _LRC_CLA_CS_EFF_FCN_2012[5], 'WL' : _LRC_CLA_CS_EFF_FCN_2012[0]},
                  'CLa2.0':   {'Norm' : 1548, 'k': 0.2616, 'a_b_y':0.21, 'a_rod' : (2.30,1.60), 'RodSat' : 6.5215, 'g' : (1.00,0.16),\
                                'Vphotl': _LRC_CLA_CS_EFF_FCN_2021[1], 'Vscotl': _LRC_CLA_CS_EFF_FCN_2021[2], \
                                'Vl_mpl': _LRC_CLA_CS_EFF_FCN_2021[3], 'Scl_mpl': _LRC_CLA_CS_EFF_FCN_2021[4],\
                                'Mcl' : _LRC_CLA_CS_EFF_FCN_2021[5], 'Scl':_LRC_CLA_CS_EFF_FCN_2021[6], \
                                'ybarl':_LRC_CLA_CS_EFF_FCN_2021[7],'WL' : _LRC_CLA_CS_EFF_FCN_2021[0]}}
_LRC_CLA_CS_CONST['CLa'] = _LRC_CLA_CS_CONST['CLa1.0'] # for backward compatibility


def fCLa(wl, Elv, integral, Norm = None, k = None, a_b_y = None, a_rod = None, RodSat = None,\
        Vphotl = None, Vscotl = None, Vl_mpl = None, Scl_mpl = None, Mcl = None, WL = None):
    """
    Local helper function that calculate CLa from El based on Eq. 1 
    in Rea et al (2012).
    
    Args:
        The various model parameters as described in the paper and contained 
        in the dict _LRC_CONST.
        
    Returns:
        ndarray with CLa values.
        
    References:
        1. `Rea MS, Figueiro MG, Bierman A, and Hamner R (2012). 
        Modelling the spectral sensitivity of the human circadian system. 
        Light. Res. Technol. 44, 386–396.  
        <https://doi.org/10.1177/1477153511430474>`_
            
        2. `Rea MS, Figueiro MG, Bierman A, and Hamner R (2012). 
        Erratum: Modeling the spectral sensitivity of the human circadian system 
        (Lighting Research and Technology (2012) 44:4 (386-396)). 
        Light. Res. Technol. 44, 516.
        <https://doi.org/10.1177/1477153512467607>`_
         
        
    """
    dl = getwld(wl) 
    
    # Calculate piecewise function in Eq. 1 in Rea et al. 2012:
    
    #calculate value of condition function (~second term of 1st fcn):
    cond_number = integral(Elv*Scl_mpl*dl) - k*integral(Elv*Vl_mpl*dl)

    # Calculate second fcn:
    fcn2 = integral(Elv*Mcl*dl)

    # Calculate last term of 1st fcn:
    fcn1_3 = a_rod * (1 - np.exp(-integral(Vscotl*Elv*dl)/RodSat))

    # Satisfying cond. is effectively adding fcn1_2 and fcn1_3 to fcn1_1:
    CLa = Norm*(fcn2 + 1*(cond_number>=0)*(a_b_y*cond_number - fcn1_3))
    
    return CLa
    
def fCLa2d0(wl, Elv, integral, Norm = None, k = None, a_b_y = None, a_rod = None, RodSat = None,\
            Vphotl = None, Vscotl = None, Vl_mpl = None, Scl_mpl = None, Mcl = None, WL = None,\
            Scl = None, ybarl = None, g = None):
    """
    Local helper function that calculate CLa2.0 from El based on the correction of Eq. 3 
    in Rea et al (2021, 2022).
    
    Args:
        The various model parameters as described in the paper and contained 
        in the dict _LRC_CONST.
        
    Returns:
        ndarray with CLa 2.0 values.
        
    References:
        1. `Rea, M.S., Nagare, R., & Figueiro, M.G. (2021). 
        Modeling Circadian Phototransduction: Quantitative Predictions of Psychophysical Data. 
        Frontiers in Neuroscience, 15, 44. 
        <https://doi.org/10.3389/fnins.2021.615322>`_
        
        2. `LRC Online Circadian Stimulus calculator (CLa2.0, 2021)
        <https://docs.light-health.org/cscalc>`_
        
        3. `Rea, M. S., Nagare, R., & Figueiro, M. G. (2022). 
        Corrigendum: Modeling Circadian Phototransduction: Quantitative Predictions of Psychophysical Data. 
        Frontiers in Neuroscience, 16. 
        <https://www.frontiersin.org/article/10.3389/fnins.2022.849800>`_
        
        4. `Github code: LRC Online Circadian stimulus calculator (CLa2.0, accessed Nov. 5, 2021)
        <https://github.com/Light-and-Health-Research-Center/cscalc>`_

    Note:
        1. The equations and parameter values used are those in the 
        github repository and corrigendum.
    """
    dl = getwld(wl) 
    
    # Calculate piecewise function in Eq. 3 in Rea et al. 2021 
    # -> use corrected in corrigendum from 2022:
    
    # calculate value of condition function (b-y):
    cond_number = integral(Elv*Scl_mpl*dl) - k*integral(Elv*Vl_mpl*dl)

    # Calculate 1st term of fcn 1 and 2:
    term1 = integral(Elv*Mcl*dl) # github code: = cs1
    term1 = term1*((term1>=0)*1) # is in code on github
    
    # Calculate first factor of rod contribution for rod 1 & 2:
    num = integral(Vscotl*Elv*dl)
    denom1 = integral(Elv*Vl_mpl*dl) # Note that Vl/mac is used here to get correct results, while this is NOT what is in the paper (but it is in the github code!)
    denom2 = integral(Elv*Scl_mpl*dl) # Note that Scl/mac is used here to get correct results, while this is NOT what is in the paper (but it is in the github code!)
    rod_contr_11 = num / (denom1 + g[0]*denom2) # github code: rod_mel
    rod_contr_12 = num / (denom1 + g[1]*denom2) # github code: rod_bminusY
    
    # Calculate second factor of rod contribution:
    rod_contr_2 = (1 - np.exp(-integral(Vscotl*Elv*dl)/RodSat))
    
    # Total rod contribution for rod 1 & 2:
    rod_contr1 = a_rod[0] * rod_contr_11 * rod_contr_2 # github code: = rodmel
    rod_contr2 = a_rod[1] * rod_contr_12 * rod_contr_2 # github code: = rod
    
    # first part of fcn (in pw-fcn 1 & 2):
    fcn1 = term1 - rod_contr1 # github code: = cs1 - rodmel
    
    # second part of fcn (not part of pw_fcn 2):
    fcn2 = a_b_y*cond_number
    fcn2 = fcn2*((fcn2>=0)*1) # is in code on github
    fcn2 = fcn2 - rod_contr2 # github code: cs2 - rod
    
    # Satisfying cond. is effectively adding fcn1_2 and fcn1_3 to fcn1_1:
    CLa2d0 = Norm*(fcn1 + (1*(cond_number>=0))*fcn2)
    CLa2d0[CLa2d0<0] = 0
    return CLa2d0

def interpolate_efficiency_functions(wl, cs_cl_lrs):
    """
    Interpolate all spectral data in dict cs_cl_lrs to new wavelength range.
    """
    
    for key in cs_cl_lrs:
        if key[-1] == 'l': #signifies l for spectral data
            temp = np.vstack((cs_cl_lrs['WL'],cs_cl_lrs[key])) # construct [wl,S] data
            cs_cl_lrs[key] = cie_interp(temp,wl, kind = 'linear', negative_values_allowed=True,extrap_values=0.0)[1:] # interpolate and store in dict
    cs_cl_lrs['WL'] = wl # store new wavelength range
    
    return  cs_cl_lrs

def CLa_to_CS(CLa, t = 1, f = 1, forward = True):
    """ 
    Convert Circadian Light to Circadian Stimulus (and back).
    
    Args:
        :CLa:
            | ndarray with Circadian Light values
            | or Circadian Stimulus values (if forward == False)
        :t:
            | 1.0, optional
            | The duration factor (in hours): a continuous value from 0.5 to 3.0
        :f:
            | 1.0, optional
            | The spatial distribution factor: a discrete value (2, 1, or 0.5)
            | depending upon the spatial distribution of the light source.
            | Default = 1 (for t = 1 h, CS is equal to the 2012 version).
            | Options:
            | - 2.0: full visual field, as with a Ganzfeld.
            | - 1.0: central visual field, as with a discrete light box on a desk.
            | - 0.5: superior visual field, as from ceiling mounted down-light fixtures.
        :forward:
            | True, optional
            | If True: convert CLa to CS values.
            | If False: convert CS values to CLa values.
            
    Returns:
         :CS:
             | ndarray with CS values or with CLa values (if forward == False)
             
    References:
        1. `Rea MS, Figueiro MG, Bierman A, and Hamner  R (2012). 
        Modelling the spectral sensitivity of the human circadian system. 
        Light. Res. Technol. 44, 386–396.  
        <https://doi.org/10.1177/1477153511430474>`_
            
        2. `Rea MS, Figueiro MG, Bierman A, and Hamner R (2012). 
        Erratum: Modeling the spectral sensitivity of the human circadian system 
        (Lighting Research and Technology (2012) 44:4 (386-396)). 
        Light. Res. Technol. 44, 516.
        <https://doi.org/10.1177/1477153512467607>`_
        
        3. `Rea, M. S., Nagare, R., & Figueiro, M.G. (2021). 
        Modeling Circadian Phototransduction: Quantitative Predictions of Psychophysical Data. 
        Frontiers in Neuroscience, 15, 44. 
        <https://doi.org/10.3389/fnins.2021.615322>`_     
            
        4. `LRC Online Circadian Stimulus calculator (CLa2.0, 2021)
        <https://docs.light-health.org/cscalc>`_

    """
    if forward: 
        return 0.7 * (1 - (1/(1 + (t*f*CLa/355.7)**1.1026)))
    else:
        return (355.7 / (t * f)) * (1 / (1 - CLa / 0.7) - 1)**(1 / 1.1026)
        
def spd_to_CS_CLa_lrc(El = None, version = 'CLa2.0', E = None, 
                      sum_sources = False, interpolate_sources = True,
                      t_CS = 1.0, f_CS = 1.0):
    """
    Calculate Circadian Stimulus (CS) and Circadian Light (CLa, CLa2.0).
    
    
    Args:
        :El:
            | ndarray, optional
            | Defaults to D65
            | light source spectral irradiance distribution
        :version:
            | 'CLa2.0', optional
            | CLa version to calculate 
            | Options: 
            | - 'CLa1.0': Rea et al. 2012
            | - 'CLa2.0': Rea et al. 2021, 2022
        :E: 
            | None, float or ndarray, optional
            | Illuminance of light sources.
            | If None: El is used as is, otherwise El is renormalized to have
            | an illuminance equal to E.
        :sum_sources:
            | False, optional
            |   - False: calculate CS (1.0,2.0) and CLa (1.0, 2.0) for all sources in El array.
            |   - True: sum sources in El to a single source and perform calc.
        :interpolate_sources:
            | True, optional
            |  - True: El is interpolated to wavelength range of efficiency 
            |          functions (as in LRC calculator). 
            |  - False: interpolate efficiency functions to source range. 
            |           Source interpolation is not recommended due to possible
            |           errors for peaky spectra. 
            |           (see CIE15-2018, "Colorimetry").
        :t_CS:
            | 1.0, optional
            | The duration factor (in hours): a continuous value from 0.5 to 3.0
        :f_CS:
            | 1.0, optional
            | The spatial distribution factor: a discrete value (2, 1, or 0.5)
            | depending upon the spatial distribution of the light source.
            | Default = 1 (for t = 1 h, CS is equal to the 2012 version).
            | Options:
            | - 2.0: full visual field, as with a Ganzfeld.
            | - 1.0: central visual field, as with a discrete light box on a desk.
            | - 0.5: superior visual field, as from ceiling mounted down-light fixtures.
            
    Returns:
        :CS:
            | ndarray with Circadian stimulus values
        :CLa:
            | ndarray with Circadian Light values
            
    Notes on CLa1.0 (2012 version):
        1. The original 2012 (E.q. 1) had set the peak wavelength of the 
        melanopsin at 480 nm. Rea et al. later published a corrigendum with 
        updated model parameters for k, a_{b-y} and a_rod. The comparison table
        between showing values calculated for a number of sources with the old
        and updated parameters were very close (~1 unit voor CLa). 
        
        2. In that corrrection paper they did not mention a change in the
        factor (1622) that multiplies the (sum of) the integral(s) in Eq. 1. 
        HOWEVER, the excel calculator released in 2017 and the online 
        calculator show that factor to have a value of 1547.9. The change in
        values due to the new factor is much larger than their the updated 
        mentioned in note 1!
        
        3. For reasons of consistency the calculator uses the latest model 
        parameters, as could be read from the excel calculator. They values 
        adopted are: multiplier 1547.9, k = 0.2616, a_{b-y} = 0.7 and 
        a_rod = 3.3. 
        
        4. The parameter values to convert CLa to CS were also taken from the 
        2017 excel calculator.
        
    Notes on CLa2.0 (2021 version):
        1. In the original model, 1000 lux of CIE Illuminant A resulted in a 
        CLA = 1000. In the revised model, a photopic illuminance of 1000 lux 
        from CIE Illuminant A (approximately that of an incandescent lamp 
        operated at 2856 K) results in a CLA 2.0 = 813. The value of 813 CLA 2.0
        should be used by those wishing to calibrate instrumentation designed 
        to report CLA 2.0 and CS. CLA 2.0 values can still be used to approximate
        the photopic illuminance, in lux, from a nonspecific "white" light source.
        For comparison, CLA 2.0 values should be multiplied by 1.23 to estimate
        the equivalent photopic illuminance from CIE Illuminant A, or by 0.66 
        to estimate the equivalent photopic illuminance from CIE Illuminant D65
        (an approximation of daylight with a CCT of 6500 K).
        
        2. Nov. 6, 2021: To get a value of CLa2.0 = 813, Eq. 3 from the paper 
        must be adjusted to also divide by the transmision of the 
        macula ('mp' in paper) the S-cone and Vlambda functions prior to 
        calculating the integrals in the denominators of the first factor after
        the a_rod_1 and a_rod_2 scalars! Failure to do so results in a CLa2.0 
        of 800, instead of the reported 813 by the online calculator. 
        Verification of the code on github shows indeed that these denominators
        are calculated by using the macular transmission divided S-cone and 
        Vlambda functions. Is this an error in the code or in the paper?
        
        3. Feb. 22, 2022: A corrigendum has been released for Eq. 3 in the original
        paper, where the normalization is indeed done.
        
        4. Feb. 22, 2022: While the rodsat value in the corrigendum is defined as 6.50 W/m²,
        this calculator uses the value as used in the online calculator: 6.5215 W/m².
        (see `code base on github: <https://github.com/Light-and-Health-Research-Center/cscalc>`_)
        
        
        
    References:
        
        1. `LRC Online Circadian stimulus calculator 
        <http://www.lrc.rpi.edu/cscalculator/>`_
        
        2. `LRC Excel based Circadian stimulus calculator. 
        <http://www.lrc.rpi.edu/resources/CSCalculator_2017_10_03_Mac.xlsm>`_
        
        3. `Rea MS, Figueiro MG, Bierman A, and Hamner R (2012). 
        Modelling the spectral sensitivity of the human circadian system. 
        Light. Res. Technol. 44, 386–396.  
        <https://doi.org/10.1177/1477153511430474>`_
            
        4. `Rea MS, Figueiro MG, Bierman A, and Hamner R (2012). 
        Erratum: Modeling the spectral sensitivity of the human circadian system 
        (Lighting Research and Technology (2012) 44:4 (386-396)). 
        Light. Res. Technol. 44, 516.
        <https://doi.org/10.1177/1477153512467607>`_
        
        5. `Rea, M. S., Nagare, R., & Figueiro, M. G. (2021). 
        Modeling Circadian Phototransduction: Quantitative Predictions of Psychophysical Data. 
        Frontiers in Neuroscience, 15, 44. 
        <https://doi.org/10.3389/fnins.2021.615322>`_
        
        6. `Rea, M. S., Nagare, R., & Figueiro, M. G. (2022). 
        Corrigendum: Modeling Circadian Phototransduction: Quantitative Predictions of Psychophysical Data. 
        Frontiers in Neuroscience, 16. 
        <https://www.frontiersin.org/article/10.3389/fnins.2022.849800>`_
        
        7. `LRC Online Circadian stimulus calculator (CLa2.0, 2021)
        <https://docs.light-health.org/cscalc>`_
        
        8. `Github code: LRC Online Circadian stimulus calculator (CLa2.0, accessed Nov. 5, 2021)
        <https://github.com/Light-and-Health-Research-Center/cscalc>`_
        
    """
    # Create copy of dict with model parameters and spectral data:
    cs_cl_lrs = _LRC_CLA_CS_CONST[version].copy()
    
    # Interpolate efficiency functions to light source wl-range:
    if interpolate_sources is False:
        cs_cl_lrs = interpolate_efficiency_functions(El[0], cs_cl_lrs)
    else:
        El = cie_interp(El, cs_cl_lrs['WL'], datatype = 'spd')
    
    # Get wavelength spacing:
    dl = getwld(El[0])  
    
    # Separate wavelengths and data:
    wl = El[0]
    Elv = El[1:].copy()
      
    # define integral function:
    # from scipy import integrate # lazy import
    # integral = lambda x: integrate.trapz(x, x = wl, axis = -1) 
    integral = lambda x: np.sum(x,  axis = -1) 
    
    # Rescale El to E (if not None):
    if E is not None:

        Vlambda_version = 'ybarl' if version == 'CLa2.0' else 'Vphotl' 
        K = 683 if version == 'CLa2.0' else 683.002 
        
        # Calculate current E value of El:
        E_cv = np.atleast_2d(K * integral(cs_cl_lrs[Vlambda_version]*Elv*dl))

        # Rescale El to supplied E:
        Elv = (E/E_cv).T*Elv
        
         
                
    # Sum all sources in array if requested:
    if sum_sources == True:
        Elv = Elv.sum(axis = 0, keepdims = True)  
    
    # Calculate Circadian light using model param. and spectral data:
    if (version == 'CLa') | (version == 'CLa1.0'): 
        CLa = fCLa(wl, Elv, integral, **cs_cl_lrs)
    
    elif (version == 'CLa2.0'):
        CLa = fCLa2d0(wl, Elv, integral, **cs_cl_lrs)
    else:
        raise Exception('Unsupported CLa version!')
    
    # Calculate Circadian stimulus:
    CS = CLa_to_CS(CLa, t = t_CS, f = f_CS)
    
    return CS, CLa

        
if __name__ == '__main__':

    import luxpy as lx
    
    E = 1000
    El = _CIE_ILLUMINANTS['A'].copy() 
    El = El[:,(El[0]>=380) & (El[0]<=730) & ((El[0]%2)==0)]
    # El = El[:,(El[0]>=380) & (El[0]<=730)]# & ((El[0]%2)==0)]
    
    # Ela = lx.utils.loadtxt('./data/Aa.dat',header=None).T
    # Elr = lx.utils.loadtxt('./data/Ar.dat',header=None).T
    # El = np.vstack((Ela,Elr[1:]))
    
    CS, CLa = spd_to_CS_CLa_lrc(El = El, E = E, version = 'CLa1.0',\
                                sum_sources = False, interpolate_sources = False)
    print('out CLa1.0')
    print('Cs: ', CS)
    print('CLa: ',CLa)
    
    CS, CLa = spd_to_CS_CLa_lrc(El = El, E = E, version = 'CLa2.0',\
                                sum_sources = False, interpolate_sources = False)
    print('\nout Cla2.0')
    print('Cs: ', CS) # should be 0.114 according to web-calculator
    print('CLa: ',CLa) # should be 80 according to web-calculator