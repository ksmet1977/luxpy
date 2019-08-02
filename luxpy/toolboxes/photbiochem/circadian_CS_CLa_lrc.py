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

# _LRC_CLA_CS_CONST: dict with model parameters and spectral data.

# spd_to_CS_CLa_lrc(): Calculate Circadian Stimulus (CS) and Circadian Light 
                       [LRC: Rea et al 2012]



Definitions
-----------

1. **Circadian Stimulus (CS)** is the calculated effectiveness of the 
spectrally weighted irradiance at the cornea from threshold (CS = 0.1) 
to saturation (CS = 0.7), assuming a fixed duration of exposure of 1 hour.

2. **Circadian Light (CLA)** is the irradiance at the cornea weighted to reflect 
the spectral sensitivity of the human circadian system as measured by acute 
melatonin suppression after a 1-hour exposure.


References
----------
    1. `LRC Online Circadian stimulus calculator <http://www.lrc.rpi.edu/cscalculator/>`_
    
    2. `LRC Excel based Circadian stimulus calculator. <http://www.lrc.rpi.edu/resources/CSCalculator_2017_10_03_Mac.xlsm>`_
    
    3. `Rea MS, Figueiro MG, Bierman A, and Hamner R (2012). 
        Modelling the spectral sensitivity of the human circadian system. 
        Light. Res. Technol. 44, 386–396.  
        <https://doi.org/10.1177/1477153511430474>`_
            
    4. `Rea MS, Figueiro MG, Bierman A, and Hamner R (2012). 
        Erratum: Modeling the spectral sensitivity of the human circadian system 
        (Lighting Research and Technology (2012) 44:4 (386-396)). 
        Light. Res. Technol. 44, 516.
        <https://doi.org/10.1177/1477153512467607>`_


Also see notes in doc_string of spd_to_CS_CLa_lrc()

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from luxpy import np, sp, _PKG_PATH, _SEP, _CIE_ILLUMINANTS, getdata, getwld, cie_interp, _IESTM3015, blackbody
from luxpy import spd_to_power

__all__=['_LRC_CLA_CS_CONST','spd_to_CS_CLa_lrc']


_LRC_CLA_CS_EFF_FCN = getdata(_PKG_PATH + _SEP + 'toolboxes' + _SEP + 'photbiochem' + _SEP  + 'data' + _SEP + 'LRC2012_CS_CLa_efficiency_functions.dat', header = 'infer', kind ='np', verbosity = 0).T


_LRC_CLA_CS_CONST = {'CLa_2012' : {'Norm' : 1622, 'k': 0.2616, 'a_b_y':0.6201, 'a_rod' : 3.2347, 'RodSat' : 6.52,\
                           'Vphotl': _LRC_CLA_CS_EFF_FCN[1], 'Vscotl': _LRC_CLA_CS_EFF_FCN[2], \
                           'Vl_mpl': _LRC_CLA_CS_EFF_FCN[3], 'Scl_mpl': _LRC_CLA_CS_EFF_FCN[4],\
                           'Mcl' : _LRC_CLA_CS_EFF_FCN[5], 'WL' : _LRC_CLA_CS_EFF_FCN[0]},\
                  'CLa':   {'Norm' : 1547.9, 'k': 0.2616, 'a_b_y':0.7, 'a_rod' : 3.3, 'RodSat' : 6.5215,\
                           'Vphotl': _LRC_CLA_CS_EFF_FCN[1], 'Vscotl': _LRC_CLA_CS_EFF_FCN[2], \
                           'Vl_mpl': _LRC_CLA_CS_EFF_FCN[3], 'Scl_mpl': _LRC_CLA_CS_EFF_FCN[4],\
                           'Mcl' : _LRC_CLA_CS_EFF_FCN[5], 'WL' : _LRC_CLA_CS_EFF_FCN[0]}}



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
    

def interpolate_efficiency_functions(wl, cs_cl_lrs):
    """
    Interpolate all spectral data in dict cs_cl_lrs to new wavelength range.
    """
    
    for key in cs_cl_lrs:
        if key[-1] == 'l': #signifies l for spectral data
            temp = np.vstack((cs_cl_lrs['WL'],cs_cl_lrs[key])) # construct [wl,S] data
            cs_cl_lrs[key] = cie_interp(temp,wl, kind = 'cmf', negative_values_allowed=True)[1:] # interpolate and store in dict
    cs_cl_lrs['WL'] = wl # store new wavelength range
    
    return  cs_cl_lrs

        
def spd_to_CS_CLa_lrc(El = None, E = None, \
                          sum_sources = False, interpolate_sources = True):
    """
    Calculate Circadian Stimulus (CS) and Circadian Light [LRC: Rea et al 2012].
    
    
    Args:
        :El:
            | ndarray, optional
            | Defaults to D65
            | light source spectral irradiance distribution
        :E: 
            | None, float or ndarray, optional
            | Illuminance of light sources.
            | If None: El is used as is, otherwise El is renormalized to have
              an illuminance equal to E.
        :sum_sources:
            | False, optional
            |   - False: calculate CS and CLa for all sources in El array.
            |   - True: sum sources in El to a single source and perform calc.
        :interpolate_sources:
            | True, optional
            |  - True: El is interpolated to wavelength range of efficiency 
            |          functions (as in LRC calculator). 
            |  - False: interpolate efficiency functions to source range. 
            |           Source interpolation is not recommended due to possible
            |           errors for peaky spectra. 
            |           (see CIE15-2004, "Colorimetry").
            
    Returns:
        :CS:
            | ndarray with Circadian stimulus values
        :CLa:
            | ndarray with Circadian Light values
            
    Notes:
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
        
    References:
        
        1. `LRC Online Circadian stimulus calculator <http://www.lrc.rpi.edu/cscalculator/>`_
        
        2. `LRC Excel based Circadian stimulus calculator. <http://www.lrc.rpi.edu/resources/CSCalculator_2017_10_03_Mac.xlsm>`_
        
        3. `Rea MS, Figueiro MG, Bierman A, and Hamner R (2012). 
        Modelling the spectral sensitivity of the human circadian system. 
        Light. Res. Technol. 44, 386–396.  
        <https://doi.org/10.1177/1477153511430474>`_
            
        4. `Rea MS, Figueiro MG, Bierman A, and Hamner R (2012). 
        Erratum: Modeling the spectral sensitivity of the human circadian system 
        (Lighting Research and Technology (2012) 44:4 (386-396)). 
        Light. Res. Technol. 44, 516.
        <https://doi.org/10.1177/1477153512467607>`_
    """
    # Create copy of dict with model parameters and spectral data:
    cs_cl_lrs = _LRC_CLA_CS_CONST['CLa'].copy()
    
 
    # Interpolate efficiency functions to light source wl-range:
    if interpolate_sources is False:
        cs_cl_lrs = interpolate_efficiency_functions(El[0], cs_cl_lrs)
    else:
        El = cie_interp(El, cs_cl_lrs['WL'], kind = 'spd')
    
    # Get wavelength spacing:
    dl = getwld(El[0])        
    
    # Separate wavelengths and data:
    wl = El[0]
    Elv = El[1:].copy()
      
    # define integral function:
    integral = lambda x: sp.integrate.trapz(x, x = wl, axis = -1) 
    #integral = lambda x: np.sum(x,  axis = -1) 
    
    # Rescale El to E (if not None):
    if E is not None:

        # Calculate current E value of El:
        E_cv = np.atleast_2d(683.002 * integral(cs_cl_lrs['Vphotl']*Elv*dl))

        # Rescale El to supplied E:
        Elv = (E/E_cv).T*Elv
            
                
    # Sum all sources in array if requested:
    if sum_sources == True:
        Elv = Elv.sum(axis = 0, keepdims = True)  
    
    # Calculate Circadian light using model param. and spectral data:
    CLa = fCLa(wl, Elv, integral,  **cs_cl_lrs)
    
    # Calculate Circadian stimulus:
    CS = 0.7 * (1 - (1/(1 + (CLa/355.7)**1.1026)))
    
    return CS, CLa

        
if __name__ == '__main__':
     
    E = 100
    El = _CIE_ILLUMINANTS['A'].copy() 
    El = El[:,(El[0]>=380) & (El[0]<=730) & ((El[0]%2)==0)]
    CS, CLa = spd_to_CS_CLa_lrc(El = El, E = E, \
                                    sum_sources = False, interpolate_sources = False)
    print('out')
    print('Cs: ', CS)
    print('CLa: ',CLa)