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
Module for calculating CIE (S026:2018 & TN003:2015) photobiological quantities
==============================================================================
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
| 
| Notes:
| 1. that in CIE S026 the l-cone, m-cone, s-cone -opic quantities are 
| based on the CIE 2006 L,M,S 10° cone fundamentals!
| 2. This differs from the approach in Lucas et al. (2014), 
| where the cone spectral sensitivity functions are based on an opsin template, and
| are denote by the terms 'erythropic', 'chloropic' and 'cyanopic', resp.
| 3. For rhopic responses, the sensitivity curve is the V'(lambda) curve from ISO 23539 / CIE S 010.
| 4. The melanopic spectral sensitivity curve is the same shape as the Nz(lambda) function 
| in Lucas et al. (2014)
| 5. In CIE S026: luminous radiation is limited to a 380 nm - 780 nm range, and any summing is done
| over the range with a 1 nm spacing!

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
 
 :_ACTIONSPECTRA: | ndarray with default CIE-S026:2018 alpha-actinic action spectra. (stored in file:
                  | './data/cie_S026_2018_SI_action_spectra_CIEToolBox_v1.049a_20_11.csv', last downloaded from cie.at.co/data-tables/ on Dec 18, 2024)
     
 :_ACTIONSPECTRA_CIES026: | ndarray with alpha-actinic action spectra. (stored in file:
                          | './data/cie_S026_2018_SI_action_spectra_CIEToolBox_v1.049a_20_11.csv', last downloaded from cie.at.co/data-tables/ on Dec 18, 2024)

 :_ACTIONSPECTRA_CIETN003: | ndarray with CIE-TN003:2015 alpha-actinic action spectra. (stored in file:
                           | './data/cie_tn003_2015_SI_action_spectra.dat')


 :spd_to_aopicE(): | Calculate alpha-opic irradiance (Ee,α) and equivalent 
                   | luminance (Eα) values for the l-cone, m-cone, s-cone, 
                   | rod and iprgc (α) photoreceptor cells following 
                   | CIE S026:2018 (= default actionspectra) or CIE TN003:2015.
                   
                   
 :spd_to_aopicEDI(): | Calculate alpha-opic equivalent daylight (D65) illuminance (lx)
                     | for the l-cone, m-cone, s-cone, rod and iprgc (α) photoreceptor cells.

 :spd_to_aopicDER(): | Calculate α-opic Daylight (D65) Efficacy Ratio
                     | for the l-cone, m-cone, s-cone, rod and iprgc (α) photoreceptor cells.

 :spd_to_aopicELR(): | Calculate α-opic Efficacy of Luminous Radiation
                     | for the l-cone, m-cone, s-cone, rod and iprgc (α) photoreceptor cells.


References:
      1. `CIE-S026:E2018 (2018). 
      CIE System for Metrology of Optical Radiation for ipRGC-Influenced Responses to Light 
      (Vienna, Austria).
      <https://cie.co.at/publications/cie-system-metrology-optical-radiation-iprgc-influenced-responses-light-0>`_
      (https://files.cie.co.at/CIE%20S%20026%20alpha-opic%20Toolbox%20User%20Guide.pdf)

      2. `CIE-TN003:2015 (2015). 
      Report on the first international workshop on 
      circadian and neurophysiological photometry, 2013 
      (Vienna, Austria).
      <http://www.cie.co.at/publications/report-first-international-workshop-circadian-and-neurophysiological-photometry-2013>`_
      (http://files.cie.co.at/785_CIE_TN_003-2015.pdf)

      3. `Lucas RJ, Peirson SN, Berson DM, Brown TM, Cooper HM, Czeisler CA, Figueiro MG, 
      Gamlin PD, Lockley SW, O'Hagan JB, Price LL, Provencio I, Skene DJ, Brainard GC. (2014). 
      Measuring and using light in the melanopsin age. 
      Trends Neurosci. 2014 Jan;37(1):1-9. doi: 10.1016/j.tins.2013.10.004.<https://doi.org/10.1016/j.tins.2013.10.004>`_

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import numpy as np 

from luxpy import _CIEOBS, _CIE_D65, _CIE_E, _BB, spd, getwld, vlbar, spd_to_power, spd_normalize, cie_interp
from luxpy.utils import  _PKG_PATH, _SEP, getdata

__all__ = ['_PHOTORECEPTORS','_QUANTITIES', 
           '_ACTIONSPECTRA','_ACTIONSPECTRA_CIES026','_ACTIONSPECTRA_CIETN003',
           '_KM_CORRECTION_FACTOR',
           '_Ee_SYMBOLS', '_E_SYMBOLS', '_Q_SYMBOLS', 
           '_Ee_UNITS', '_E_UNITS', '_Q_UNITS', 
           'spd_to_aopicE','spd_to_aopicEDI','spd_to_aopicDER','spd_to_aopicELR',
           'spd_to_aopicX']


_PHOTORECEPTORS = ['l-cone', 'm-cone','s-cone', 'rod', 'iprgc']
_Ee_SYMBOLS =  ['Ee,lc','Ee,mc', 'Ee,sc','Ee,r',  'Ee,z']
_E_SYMBOLS =  ['E,lc','E,mc', 'E,sc','E,r',  'E,z']
_Q_SYMBOLS =  ['Q,lc','Q,mc', 'Q,sc','Q,r',  'Q,z']
_Ee_UNITS = ['W⋅m−2', 'W⋅m−2', 'W⋅m−2', 'W⋅m−2', 'W⋅m−2'] 
_E_UNITS = ['lux', 'lux', 'lux', 'lux', 'lux'] 
_Q_UNITS = ['photons/m2/s', 'photons/m2/s', 'photons/m2/s', 'photons/m2/s', 'photons/m2/s'] 
_QUANTITIES = ['erythropic', 'chloropic','cyanopic','rhodopic','melanopic'] #irradiance, illuminance

#_ACTIONSPECTRA_CIES026 = getdata(_PKG_PATH + _SEP + 'toolboxes' + _SEP + 'photbiochem' + _SEP  + 'data' + _SEP + 'cie_S026_2018_SI_action_spectra_CIEToolBox_v1.049a_20_11.csv', header = 'infer', verbosity = 0).T[[0,3,2,1,4,5]] # [[0,3,2,1,4,5]]-> put in l,m,s order
#_ACTIONSPECTRA_CIES026 = getdata(_PKG_PATH + _SEP + 'toolboxes' + _SEP + 'photbiochem' + _SEP  + 'data' + _SEP + 'cie_S026_2018_SI_action_spectra_LUOXapp_download04062025.csv', header = 'infer', verbosity = 0).T[[0,3,2,1,4,5]] # [[0,3,2,1,4,5]]-> put in l,m,s order
_ACTIONSPECTRA_CIES026 = getdata(_PKG_PATH + _SEP + 'toolboxes' + _SEP + 'photbiochem' + _SEP  + 'data' + _SEP + 'S026_Table2_Data_downloaded_04062025.csv', header = 'infer', verbosity = 0).T[[0,3,2,1,4,5]] # [[0,3,2,1,4,5]]-> put in l,m,s order
_ACTIONSPECTRA_CIES026[np.isnan(_ACTIONSPECTRA_CIES026)] = 0.0 # unknown data stored as NaN,-> convert to 0.0 so summing does not add any contribution
_ACTIONSPECTRA_CIETN003 = getdata(_PKG_PATH + _SEP + 'toolboxes' + _SEP + 'photbiochem' + _SEP  + 'data' + _SEP + 'cie_tn003_2015_SI_action_spectra.dat', header = 'infer', verbosity = 0).T
_ACTIONSPECTRA = _ACTIONSPECTRA_CIES026

_ELR_D65_LUOX_APP_GITHUB = np.array([[1.62890776589039,1.45582633881653,0.817289644883213,1.4497035760559,1.32621318911359]])

# Calculate correction factor for Km in standard air:
na = _BB['na'] # n for standard air
c = _BB['c'] # m/s light speed
lambdad = c/(na*54*1e13)/(1e-9) # 555 nm lambda in standard air
_KM_CORRECTION_FACTOR = 1/(1 - (1 - 0.9998567)*(lambdad - 555)) # correction factor for Km in standard air

def _limit_spd_wl_range(spd, force_1nm_spacing = False):
    if force_1nm_spacing:
        return cie_interp(spd, [380,780,1], 'spd')
    else:
        cnd = (spd[0]>=380) & (spd[0]<=780)
        return spd[:,cnd].copy()

def spd_to_aopicE(sid, Ee = None, E = None, Q = None, cieobs = _CIEOBS, K = None,
                  sid_units = 'W/m2', out = 'Eeas', actionspectra = 'CIE-S026',
                  interp_settings = None, use_pusa = False, force_1nm_sid_spacing = True):
    """
    Calculate alpha-opic irradiance (Ee,α) values (W/m²) for the l-cone, m-cone, 
    s-cone, rod and iprgc (α) photoreceptor cells following CIE S026:2018.
    
    Args:
        :sid: 
            | numpy.ndarray with retinal spectral irradiance in :sid_units: 
            | (if 'uW/cm2', sid will be converted to SI units 'W/m2')
        :Ee: 
            | None, optional
            | If not None: normalize :sid: to an irradiance of :Ee:
        :E: 
            | None, optional
            | If not None: normalize :sid: to an illuminance of :E:
            | Note that E is calculate using a Km factor corrected to standard air.
        :Q: 
            | None, optional
            | If not None: Normalize :sid: to a quantal energy of :Q:
        :cieobs:
            | _CIEOBS or str, optional
            | Type of cmf set to use for photometric units.
        :sid_units:
            | 'W/m2', optional
            | Other option 'uW/m2', input units of :sid:
        :out: 
            | 'Eeas' or str, optional
            | Determines values to return. 
            | (to get also get equivalent illuminance Eα set :out: to 'Eeas,Eas')
        :actionspectra:
            | 'CIE-S026', optional
            | Actionspectra to use in calculation 
            | options: 
            | - 'CIE-S026': will use action spectra as defined in CIE S026 
            | - 'CIE-TN003': will use action spectra as defined in CIE TN003
        :force_1nm_sid_spacing:
            | True, optional
            | If True: when limiting the sid wavelength range to 380 nm -780 nm,
            |   also, force the wavelength spacing to 1 nm by interpolating the sid.
            
    Returns:
        :returns: 
            | Eeas a numpy.ndarray with the α-opic irradiance
            | of all spectra in :sid: in SI-units (W/m²). 
            |
            | (other choice can be set using :out:)
            
    References:
          1. `CIE-S026:E2018 (2018). 
          CIE System for Metrology of Optical Radiation for ipRGC-Influenced Responses to Light 
          (Vienna, Austria).
          <https://cie.co.at/publications/cie-system-metrology-optical-radiation-iprgc-influenced-responses-light-0>`_
          (https://files.cie.co.at/CIE%20S%20026%20alpha-opic%20Toolbox%20User%20Guide.pdf)
    
          2. `CIE-TN003:2015 (2015). 
          Report on the first international workshop on 
          circadian and neurophysiological photometry, 2013 
          (Vienna, Austria).
          <http://www.cie.co.at/publications/report-first-international-workshop-circadian-and-neurophysiological-photometry-2013>`_
          (http://files.cie.co.at/785_CIE_TN_003-2015.pdf)
    """
    outlist = out.split(',')

    sid = _limit_spd_wl_range(sid, force_1nm_spacing = force_1nm_sid_spacing)
    
    # Convert to Watt/m²:
    if sid_units == 'uW/cm2':
        sid[1:] = sid[1:]/100

    elif sid_units == 'W/m2':
        pass
    else:
        raise Exception("spd_to_aopicE(): {} unsupported units for SID.".format(sid_units))
    
    
    # Normalize sid to Ee:
    ptype = 'pusa' if use_pusa else 'pu'
    if Ee is not None:
        sid = spd_normalize(sid.copy(), norm_type = 'ru', norm_f = Ee,
                            cieobs = cieobs, K = K,
                            interp_settings = interp_settings)  
    elif E is not None:
        sid = spd_normalize(sid.copy(), norm_type = ptype, norm_f = E, 
                            cieobs = cieobs, K = K,
                            interp_settings = interp_settings) 
    elif Q is not None:
        sid = spd_normalize(sid.copy(), norm_type = 'qu', norm_f = Q,
                            cieobs = cieobs, K = K, 
                            interp_settings = interp_settings) 
        
    
    # Get sid irradiance (W/m²):
    if 'Ee' in outlist:
        Ee = spd_to_power(sid, cieobs = cieobs, K = K, ptype = 'ru', 
                          interp_settings = interp_settings)
    
    # Get sid illuminance (lx):
    if 'E' in outlist:
        E = spd_to_power(sid, cieobs = cieobs, K = K, ptype = ptype,
                         interp_settings = interp_settings) #photometric units (if requested, i.e. 'pusa'->Km corrected to standard air)
    
    # Get sid quantal energy (photons/m²/s):
    if 'Q' in outlist:
        Q = spd_to_power(sid, cieobs = cieobs, K = K,  ptype = 'qu',
                         interp_settings = interp_settings)

    # select requested actionspectra:
    if actionspectra == 'CIE-TN003':
        actionspectra = _ACTIONSPECTRA_CIETN003
    elif actionspectra == 'CIE-S026':
        actionspectra = _ACTIONSPECTRA_CIES026
    else:
        pass # must be numpy array !

    # get SI actinic action spectra, sa:
    sa = spd(actionspectra, wl = sid[0], datatype = 'cmf', #norm_type = 'max', 
             interp_settings = interp_settings)
    
    # get wavelength spacing:
    dl = getwld(sid[0])
    
    # Calculate all alpha-opics Ee's:
    Eeas = (np.dot((sa[1:]*dl),sid[1:].T)).T

    # Calculate equivalent alpha-opic E's:
    if 'Eas' in outlist:
        if isinstance(cieobs, str): 
            src = 'dict'
        else:
            src = 'vltype' # if str -> cieobs is an array
            if K is None: raise Exception('If cieobs is an array, Km must be explicitely supplied')
        
        Vl, Km = vlbar(cieobs = cieobs, K = K, src = src, wl_new = sid[0], out = 2, 
                       interp_settings = interp_settings)
        if K is None: K = Km
        Km_correction_factor = _KM_CORRECTION_FACTOR if use_pusa else 1.0
        Eas = K*Km_correction_factor*Eeas*(Vl[1].sum()/sa[1:].sum(axis = 1))

    #Prepare output:
    if out == 'Eeas':
        return Eeas
    elif out == 'Eeas,E':
        return Eeas,E
    elif out == 'Eeas,E,sid':
        return Eeas,E, sid
    elif out == 'Eeas,Eas':
        return Eeas,Eas
    elif out == 'Eeas,Eas,E':
        return Eeas,Eas,E
    elif out == 'Eas':
        return Eas
    else:
        eval(out)
        
#----------------------------------------------------------------------------------
def spd_to_aopicELR(sid, cieobs = _CIEOBS, K = None, sid_units = 'W/m2',
                    actionspectra = 'CIE-S026', use_pusa = False, 
                    out = 'ELR', force_1nm_sid_spacing = True, interp_settings = None):
    """
    Calculate α-opic Efficacy of Luminous Radiation (W/lm)
    for the l-cone, m-cone, s-cone, rod and iprgc (α) photoreceptor cells.
    
    Args:
        :sid: 
            | numpy.ndarray with retinal spectral irradiance in :sid_units: 
            | (if 'uW/cm2', sid will be converted to SI units 'W/m2')
        :cieobs:
            | _CIEOBS or str, optional
            | Type of cmf set to use for photometric units.
        :sid_units:
            | 'W/m2', optional
            | Other option 'uW/m2', input units of :sid:
        :actionspectra:
            | 'CIE-S026', optional
            | Actionspectra to use in calculation 
            | options: 
            | - 'CIE-S026': will use action spectra as defined in CIE S026 
            | - 'CIE-TN003': will use action spectra as defined in CIE TN003
        :out: 
            | 'ELR' or str, optional
            | Determines values to return.
        :force_1nm_sid_spacing:
            | True, optional
            | If True: when limiting the sid wavelength range to 380 nm -780 nm,
            |   also, force the wavelength spacing to 1 nm by interpolating the sid.
            
    Returns:
        :returns: 
            | ndarray with the α-opic Efficacy of Luminous Radiation (W/lm) with the 
            | for the l-cone, m-cone, s-cone, rod and iprgc photoreceptors
            | of all spectra in :sid: in SI-units. 
    """
    # Calculate all alpha-opic irradiance values:
    Eeas, Ev, sid = spd_to_aopicE(sid, cieobs = cieobs, K = K, sid_units = sid_units, out = 'Eeas,E,sid', actionspectra = actionspectra,
                             use_pusa = use_pusa, force_1nm_sid_spacing = force_1nm_sid_spacing, interp_settings = interp_settings) # calculate all alpha-opic values 
    
    # Calculate α-opic Efficacy of Luminous Radiation: 
    ELR = Eeas / Ev 

    if out.lower() == 'elr':
        return ELR
    elif (out.lower() == 'elr,ev') | (out.lower() == 'elr,e'):
        return ELR, Ev 
    elif (out.lower() == 'elr,ev,sid') | (out.lower() == 'elr,e,sid'):
        return ELR, Ev, sid
    else:
        raise Exception(f'Unknown requested output: {out}')

        
#--------------------------------------------------------------------------------------
def spd_to_aopicDER(sid, cieobs = _CIEOBS, K = None, sid_units = 'W/m2',
                    actionspectra = 'CIE-S026', ref = 'D65', use_pusa = False,
                    out = 'DER', force_1nm_sid_spacing = True, interp_settings = None):
    """
    Calculate α-opic Daylight (D65) Efficacy Ratio (= α-opic Daylight (D65) Efficiency)
    for the l-cone, m-cone, s-cone, rod and iprgc (α) photoreceptor cells.
    
    Args:
        :sid: 
            | numpy.ndarray with retinal spectral irradiance in :sid_units: 
            | (if 'uW/cm2', sid will be converted to SI units 'W/m2')
        :cieobs:
            | _CIEOBS or str, optional
            | Type of cmf set to use for photometric units.
        :sid_units:
            | 'W/m2', optional
            | Other option 'uW/m2', input units of :sid:
        :actionspectra:
            | 'CIE-S026', optional
            | Actionspectra to use in calculation 
            | options: 
            | - 'CIE-S026': will use action spectra as defined in CIE S026 
            | - 'CIE-TN003': will use action spectra as defined in CIE TN003
        :ref:
            | 'D65', optional
            | Reference (daylight) spectrum to use. ('D65' or 'E' or ndarray)
        :out: 
            | 'DER' or str, optional
            | Determines values to return.
        :force_1nm_sid_spacing:
            | True, optional
            | If True: when limiting the sid wavelength range to 380 nm -780 nm,
            |   also, force the wavelength spacing to 1 nm by interpolating the sid.
            
    Returns:
        :returns: 
            | ndarray with the α-opic Daylight Efficacy Ratio with the 
            | for the l-cone, m-cone, s-cone, rod and iprgc photoreceptors
            | of all spectra in :sid: in SI-units. 
    """
    #----------------------------------
    # Get ELR for spd:
    ELR, Ev, sid = spd_to_aopicELR(sid, cieobs = cieobs, K = K, sid_units = sid_units,
                    actionspectra = actionspectra, use_pusa = use_pusa, 
                    force_1nm_sid_spacing = force_1nm_sid_spacing, out = 'ELR,Ev,sid',
                    interp_settings = interp_settings)
    

    #----------------------------------
    # Get and interpolate reference illuminant spectrum:
    ELR_ref = None
    if isinstance(ref,str):
        if (ref == 'D65'):
            ref = _CIE_D65 
            ELR_ref = None#_ELR_D65_LUOX_APP_GITHUB/1000 # only works for D65 (and uses Km = 683.0015478, which is not mentioned in CIES026:2018)!
        else:
            ref = _CIE_E

    
    if ELR_ref is None: 
        ref = cie_interp(ref, sid[0], 'spd') # make ref same wavelength range as spd!

        #----------------------------------
        # Get ELR for ref spd:
        
        # Excel calculator uses 683.0015478 lm/W overall, the javascript only uses it
        # when pre-calculating the ELR(D65), for the remainder it uses 683.002.
        # The CIE S026 publication only uses 683.002 lm/W!
        # K = 683.0015478 # needed to match excel calculator toolbox (and javascript pre-calculated ELR(D65) values)) !!!
        ELR_ref = spd_to_aopicELR(ref, cieobs = cieobs, K = K, sid_units = sid_units,
                                actionspectra = actionspectra, use_pusa = use_pusa,
                                force_1nm_sid_spacing = force_1nm_sid_spacing,
                                interp_settings = interp_settings)
        
        # In CIE S026 these values are rounded to 4 digits for D65. 
        # However, when only rounding the ELR_ref, the the DER would not be 1.0 anymore
        # for D65! There is no mention of rounding ELR for the spectrum (nor actually any
        # explicit mention for the D65). So we leave it unrounded.
        # ELR_ref = np.round(ELR_ref, 4) 
    

    #----------------------------------
    # Calculate DER:
    DER = ELR/ELR_ref 

    # Return DER:
    if out.lower() == 'der':
        return DER
    elif (out.lower() == 'der,ev') | (out.lower() == 'der,e'):
        return DER, Ev
    else:
        raise Exception(f'Unknown requested output: {out}')


#------------------------------------------------------------------------------
def spd_to_aopicEDI(sid, Ee = None, E = None, Q = None, 
                    cieobs = _CIEOBS, K = None, sid_units = 'W/m2',
                    actionspectra = 'CIE-S026', ref = 'D65', 
                    out = 'EDI', use_pusa = False,
                    force_1nm_sid_spacing = True,
                    interp_settings = None):
    """
    Calculate alpha-opic equivalent daylight (D65) illuminance (lux)
    for the l-cone, m-cone, s-cone, rod and iprgc (α) photoreceptor cells.
    
    Args:
        :sid: 
            | numpy.ndarray with retinal spectral irradiance in :sid_units: 
            | (if 'uW/cm2', sid will be converted to SI units 'W/m2')
        :Ee: 
            | None, optional
            | If not None: normalize :sid: to an irradiance of :Ee:
        :E: 
            | None, optional
            | If not None: normalize :sid: to an illuminance of :E:
            | Note that E is calculate using a Km factor corrected to standard air.
        :Q: 
            | None, optional
            | If not None: nNormalize :sid: to a quantal energy of :Q:
        :cieobs:
            | _CIEOBS or str, optional
            | Type of cmf set to use for photometric units.
        :sid_units:
            | 'W/m2', optional
            | Other option 'uW/m2', input units of :sid:
        :actionspectra:
            | 'CIE-S026', optional
            | Actionspectra to use in calculation 
            | options: 
            | - 'CIE-S026': will use action spectra as defined in CIE S026 
            | - 'CIE-TN003': will use action spectra as defined in CIE TN003
        :ref:
            | 'D65', optional
            | Reference (daylight) spectrum to use. ('D65' or 'E' or ndarray)
        :out: 
            | 'EDI' or str, optional
            | Determines values to return.
        :force_1nm_sid_spacing:
            | True, optional
            | If True: when limiting the sid wavelength range to 380 nm -780 nm,
            |   also, force the wavelength spacing to 1 nm by interpolating the sid.
            
    Returns:
        :returns: 
            | ndarray with the α-opic Equivalent Daylight Illuminance (lux) with the 
            | for the l-cone, m-cone, s-cone, rod and iprgc photoreceptors
            | of all spectra in :sid: in SI-units. 
    """
    # Eeas, Ev = spd_to_aopicE(sid, cieobs = cieobs, K = K, Ee = Ee, E = E, Q = Q, sid_units = sid_units, out = 'Eeas,E', actionspectra = actionspectra,
    #                         interp_settings = interp_settings) # calculate all alpha-opic values 
    # if isinstance(ref,str):
    #     ref = _CIE_D65 if (ref == 'D65') else _CIE_E
    # ref = cie_interp(ref, wl_new = sid[0], datatype = 'spd', interp_settings = interp_settings) # make ref same wavelength range as spd!
    # #Eeas_ref, Ev_ref = spd_to_aopicE(ref, cieobs = cieobs, K = K,  out = 'Eeas,E', actionspectra = actionspectra, interp_settings = interp_settings) # calculate all alpha-opic Irradiance and illuminance values for  ref spectrum (= D65 for CIE S026) 
    # #Ev_ref = spd_to_power(ref, ptype = 'pusa', cieobs = cieobs, K = K, interp_settings = interp_settings)[:,0] # calculate photometric (illuminance) value for ref spectrum (= D65 for CIE S026)
    # #a_edi = Eeas * (Ev_ref/Eeas_ref) # calculate MEDI
    # ELR_ref = spd_to_aopicELR(ref, cieobs = cieobs, K = K, sid_units = sid_units, actionspectra = actionspectra, interp_settings = interp_settings)
    # a_edi = Eeas / ELR_ref
    # if out == 'a_edi':
    #     return a_edi
    # elif out == 'a_edi,Ev':
    #     return a_edi, Ev
    # else:
    #     return eval(out)
        
    #----------------------------------
    # Calculate alpha-opic Daylight (D65) Efficacy Ratio (= alpha-opic Daylight (D65) Efficiency)
    DER, Ev = spd_to_aopicDER(sid, cieobs = cieobs, K = K, sid_units =  sid_units, out = 'DER,Ev',
                    actionspectra = actionspectra, ref = ref, use_pusa = use_pusa,
                    force_1nm_sid_spacing = force_1nm_sid_spacing,
                    interp_settings = interp_settings)
   
    #----------------------------------
    # Calculate alpha-opic equivalent daylight (D65) illuminance:
    EDI = DER * Ev
    
    #----------------------------------
    # Regulate output:
    if out.lower() == 'edi':
        return EDI
    elif (out.lower() == 'edi,ev') | (out.lower() == 'edi,e'):
        return EDI, Ev
    else:
         raise Exception(f'Unknown requested output: {out}')


        
#==============================================================================================
def spd_to_aopicX(sid, Xtype = 'E', out = None, sid_units = 'W/m2', 
                  Ee = None, E = None, Q = None, 
                  actionspectra = 'CIE-S026', ref = 'D65', use_pusa = False,
                  cieobs = _CIEOBS, K = None,
                  force_1nm_sid_spacing = True, 
                  interp_settings = None):
    """
    Calculate various alpha-opic quantites for the l-cone, m-cone, 
    s-cone, rod and iprgc (α) photoreceptor cells following CIE S026:2018.
    
    Args:
        :sid: 
            | numpy.ndarray with retinal spectral irradiance in :sid_units: 
            | (if 'uW/cm2', sid will be converted to SI units 'W/m2')
        :Xtype:
            | 'E', optional
            | Type of alpha-opic quantity to calculate.
            | Options:
            |   - 'E'   : alpha-opic irradiance (Ee,α) values (W/m²)
            |   - 'EDI' : alpha-opic equivalent daylight (D65) illuminance (lux)
            |   - 'DER' : alpha-opic Daylight (D65) Efficacy Ratio (= alpha-opic Daylight (D65) Efficiency)
            |   - 'ELR' : alpha-opic Efficacy of Luminous Radiation (W/lm)
        :sid_units:
            | 'W/m2', optional
            | Other option 'uW/m2', input units of :sid:
        :Ee: 
            | None, optional
            | If not None: normalize :sid: to an irradiance of :Ee:
        :E: 
            | None, optional
            | If not None: normalize :sid: to an illuminance of :E:
            | Note that E is calculate using a Km factor corrected to standard air.
        :Q: 
            | None, optional
            | If not None: Normalize :sid: to a quantal energy of :Q:
        :out: 
            | None or str, optional
            | Determines values to return for the specific function. 
            | (e.g. to get  alphaopic irradiance Ee and 
            |  equivalent alpha-opic illuminance E, set :out: to 'Eeas,Eas' for Xtype == 'E')
        :actionspectra:
            | 'CIE-S026', optional
            | Actionspectra to use in calculation 
            | options: 
            | - 'CIE-S026': will use action spectra as defined in CIE S026 
        :ref:
            | 'D65', optional
            | Reference (daylight) spectrum to use with specific Xtype quantities. 
            | Options: 'D65' or 'E' or ndarray
        :cieobs:
            | _CIEOBS, optional
            | CMF set to use to get Vlambda.
        :K:
            | None, optional
            | Photopic Luminous Efficacy (lm/W)
            | If None: use the one stored in _CMF[cmf]['K']
        :force_1nm_sid_spacing:
            | True, optional
            | If True: when limiting the sid wavelength range to 380 nm -780 nm,
            |   also, force the wavelength spacing to 1 nm by interpolating the sid.
            
    Returns:
        :returns: 
            | Eeas a numpy.ndarray with the α-opic irradiance
            | of all spectra in :sid: in SI-units (W/m²). 
            |
            | (other choice can be set using :out:)
            
    References:
          1. `CIE-S026:E2018 (2018). 
          CIE System for Metrology of Optical Radiation for ipRGC-Influenced Responses to Light 
          (Vienna, Austria).
          <https://cie.co.at/publications/cie-system-metrology-optical-radiation-iprgc-influenced-responses-light-0>`_
          (https://files.cie.co.at/CIE%20S%20026%20alpha-opic%20Toolbox%20User%20Guide.pdf)
    
          2. `CIE-TN003:2015 (2015). 
          Report on the first international workshop on 
          circadian and neurophysiological photometry, 2013 
          (Vienna, Austria).
          <http://www.cie.co.at/publications/report-first-international-workshop-circadian-and-neurophysiological-photometry-2013>`_
          (http://files.cie.co.at/785_CIE_TN_003-2015.pdf)
    """
        
    if Xtype == 'E':
        if out is None: out = 'Eeas'
        return spd_to_aopicE(sid, Ee = Ee, E = E, Q = Q, 
                             sid_units = sid_units, out = out, 
                             actionspectra = actionspectra,
                             cieobs = cieobs, K = K,
                             use_pusa = use_pusa, 
                             force_1nm_sid_spacing = force_1nm_sid_spacing,
                             interp_settings = interp_settings)
    elif Xtype == 'EDI':
        if out is None: out = 'EDI'
        return spd_to_aopicEDI(sid, Ee = Ee, E = E, Q = Q, 
                               sid_units = sid_units, out = out, 
                               actionspectra = actionspectra, 
                               ref = ref, cieobs = cieobs, K = K,
                               use_pusa = use_pusa,
                               force_1nm_sid_spacing = force_1nm_sid_spacing,
                               interp_settings = interp_settings)
    elif Xtype == 'DER':
        return spd_to_aopicDER(sid, sid_units = sid_units,
                               actionspectra = actionspectra, ref = ref,
                               cieobs = cieobs, K = K,
                               use_pusa = use_pusa,
                               force_1nm_sid_spacing = force_1nm_sid_spacing,
                               interp_settings = interp_settings)
    elif Xtype == 'ELR':
        return spd_to_aopicELR(sid,  sid_units = sid_units,
                               actionspectra = actionspectra, 
                               cieobs = cieobs, K = K,
                               use_pusa = use_pusa,
                               force_1nm_sid_spacing = force_1nm_sid_spacing,
                               interp_settings = interp_settings)
    else:
        raise Exception(f'Unknown Xtype: {Xtype}')

