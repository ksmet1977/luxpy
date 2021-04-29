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
 
 :_ACTIONSPECTRA: ndarray with default CIE-S026:2018 alpha-actinic action spectra. (stored in file:
                  './data/cie_S026_2018_SI_action_spectra_CIEToolBox_v1.049.dat')
     
 :_ACTIONSPECTRA_CIES026: ndarray with alpha-actinic action spectra. (stored in file:
                  './data/cie_S026_2018_SI_action_spectra_CIEToolBox_v1.049.dat')

 :_ACTIONSPECTRA_CIETN003: ndarray with CIE-TN003:2015 alpha-actinic action spectra. (stored in file:
                  './data/cie_tn003_2015_SI_action_spectra.dat')


 :spd_to_aopicE(): Calculate alpha-opic irradiance (Ee,α) and equivalent 
                   luminance (Eα) values for the l-cone, m-cone, s-cone, 
                   rod and iprgc (α) photoreceptor cells following 
                   CIE S026:2018 (= default actionspectra) or CIE TN003:2015.
                   
                   
 :spd_to_aopicEDI(): Calculate alpha-opic equivalent daylight (D65) illuminance (lx)
                     for the l-cone, m-cone, s-cone, rod and iprgc (α) photoreceptor cells.

     
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

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import _CIEOBS, _CIE_D65, _BB, spd, getwld, vlbar, spd_to_power, spd_normalize, cie_interp
from luxpy.utils import np, _PKG_PATH, _SEP, getdata

__all__ = ['_PHOTORECEPTORS','_QUANTITIES', 
           '_ACTIONSPECTRA','_ACTIONSPECTRA_CIES026','_ACTIONSPECTRA_CIETN003',
           'Km_correction_factor',
           '_Ee_SYMBOLS', '_E_SYMBOLS', '_Q_SYMBOLS', 
           '_Ee_UNITS', '_E_UNITS', '_Q_UNITS', 
           'spd_to_aopicE','spd_to_aopicEDI']


_PHOTORECEPTORS = ['l-cone', 'm-cone','s-cone', 'rod', 'iprgc']
_Ee_SYMBOLS =  ['Ee,lc','Ee,mc', 'Ee,sc','Ee,r',  'Ee,z']
_E_SYMBOLS =  ['E,lc','E,mc', 'E,sc','E,r',  'E,z']
_Q_SYMBOLS =  ['Q,lc','Q,mc', 'Q,sc','Q,r',  'Q,z']
_Ee_UNITS = ['W⋅m−2', 'W⋅m−2', 'W⋅m−2', 'W⋅m−2', 'W⋅m−2'] 
_E_UNITS = ['lux', 'lux', 'lux', 'lux', 'lux'] 
_Q_UNITS = ['photons/m2/s', 'photons/m2/s', 'photons/m2/s', 'photons/m2/s', 'photons/m2/s'] 
_QUANTITIES = ['erythropic', 'chloropic','cyanopic','rhodopic','melanopic'] #irradiance, illuminance

_ACTIONSPECTRA_CIES026 = getdata(_PKG_PATH + _SEP + 'toolboxes' + _SEP + 'photbiochem' + _SEP  + 'data' + _SEP + 'cie_S026_2018_SI_action_spectra_CIEToolBox_v1.049.dat', header = 'infer', kind ='np', verbosity = 0).T
_ACTIONSPECTRA_CIETN003 = getdata(_PKG_PATH + _SEP + 'toolboxes' + _SEP + 'photbiochem' + _SEP  + 'data' + _SEP + 'cie_tn003_2015_SI_action_spectra.dat', header = 'infer', kind ='np', verbosity = 0).T
_ACTIONSPECTRA = _ACTIONSPECTRA_CIES026

# Calculate correction factor for Km in standard air:
na = _BB['na'] # n for standard air
c = _BB['c'] # m/s light speed
lambdad = c/(na*54*1e13)/(1e-9) # 555 nm lambda in standard air
Km_correction_factor = 1/(1 - (1 - 0.9998567)*(lambdad - 555)) # correction factor for Km in standard air


def spd_to_aopicE(sid, Ee = None, E = None, Q = None, cieobs = _CIEOBS, sid_units = 'W/m2', out = 'Eeas,Eas', actionspectra = 'CIE-S026'):
    """
    Calculate alpha-opic irradiance (Ee,α) and equivalent luminance (Eα) values
    for the l-cone, m-cone, s-cone, rod and iprgc (α) photoreceptor cells 
    following CIE S026:2018.
    
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
            | 'Eeas, Eas' or str, optional
            | Determines values to return.
        :actionspectra:
            | 'CIES026', optional
            | Actionspectra to use in calculation (options: 'CIE-S026', 'CIE-TN003')
            
    Returns:
        :returns: 
            | (Eeas, Eas) with Eeas and Eas resp. numpy.ndarrays with the 
            | α-opic irradiance and equivalent illuminance values 
            | of all spectra in :sid: in SI-units. 
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
    
    # Convert to Watt/m²:
    if sid_units == 'uW/cm2':
        sid[1:] = sid[1:]/100

    elif sid_units == 'W/m2':
        pass
    else:
        raise Exception("spd_to_aopicE(): {} unsupported units for SID.".format(sid_units))
    
    
    # Normalize sid to Ee:
    if Ee is not None:
        sid = spd_normalize(sid.copy(), norm_type = 'ru', norm_f = Ee)  
    elif E is not None:
        sid = spd_normalize(sid.copy(), norm_type = 'pusa', norm_f = E) 
    elif Q is not None:
        sid = spd_normalize(sid.copy(), norm_type = 'qu', norm_f = Q) 
        
    
    # Get sid irradiance (W/m²):
    if 'Ee' in outlist:
        Ee = spd_to_power(sid, cieobs = cieobs, ptype = 'ru')
    
    # Get sid illuminance (lx):
    if 'E' in outlist:
        E = spd_to_power(sid, cieobs = cieobs, ptype = 'pusa') #photometric units (Km corrected to standard air)
    
    # Get sid quantal energy (photons/m²/s):
    if 'Q' in outlist:
        Q = spd_to_power(sid, cieobs = cieobs, ptype = 'qu')

    # select requested actionspectra:
    if actionspectra == 'CIE-TN003':
        actionspectra = _ACTIONSPECTRA_CIETN003
    else:
        actionspectra = _ACTIONSPECTRA_CIES026

    # get SI actinic action spectra, sa:
    sa = spd(actionspectra, wl = sid[0], interpolation = 'cmf', norm_type = 'max')
    
    # get wavelength spacing:
    dl = getwld(sid[0])
    
    # Calculate all alpha-opics Ee's:
    Eeas = (np.dot((sa[1:]*dl),sid[1:].T)).T

    # Calculate equivalent alpha-opic E's:
    Vl, Km = vlbar(cieobs = cieobs, wl_new = sid[0], out = 2)
    Eas = Km*Km_correction_factor*Eeas*(Vl[1].sum()/sa[1:].sum(axis = 1))

    #Prepare output:
    if out == 'Eeas,Eas':
        return Eeas,Eas
    elif out == 'Eeas':
        return Eeas
    elif out == 'Eas':
        return Eas
    else:
        eval(out)
        
        
#------------------------------------------------------------------------------

def spd_to_aopicEDI(sid, Ee = None, E = None, Q = None, 
                    cieobs = _CIEOBS, sid_units = 'W/m2',
                    actionspectra = 'CIE-S026'):
    """
    Calculate alpha-opic equivalent daylight (D65) illuminance (lx)
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
            | 'CIES026', optional
            | Actionspectra to use in calculation (options: 'CIE-S026', 'CIE-TN003')

            
    Returns:
        :returns: 
            | ndarray with the α-opic Equivalent Daylight Illuminance with the 
            | for the l-cone, m-cone, s-cone, rod and iprgc photoreceptors
            | of all spectra in :sid: in SI-units. 
    """
    Eeas = spd_to_aopicE(sid, cieobs = cieobs, Ee = Ee, E = E, Q = Q, sid_units = sid_units, actionspectra = actionspectra)[0] # calculate all alpha-opic values and select last one (melanopic)
    D65 = cie_interp(_CIE_D65, wl_new = sid[0], kind = 'spd')
    Eeas_D65 = spd_to_aopicE(D65, cieobs = cieobs, actionspectra = actionspectra)[0] # calculate all alpha-opic values for D65 and select last one (melanopic)
    Ev_D65 = spd_to_power(D65, ptype = 'pusa', cieobs = cieobs)[:,0] # calculate photometric (illuminance) value for D65
    a_edi = Eeas * (Ev_D65/Eeas_D65) # calculate MEDI
    return a_edi

        
    
