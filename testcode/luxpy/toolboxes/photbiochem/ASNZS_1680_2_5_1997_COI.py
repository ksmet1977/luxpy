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
from luxpy import np, math, deltaE, _PKG_PATH, _SEP, _CIE_ILLUMINANTS, getdata, spd_to_xyz, blackbody, xyz_to_cct


__all__ = ['_COI_RFL_BLOOD','_COI_CIEOBS','_COI_CSPACE','spd_to_COI_ASNZS1680']


# Reflectance spectra of 100% and 50% oxygenated blood
_COI_RFL_BLOOD = getdata(_PKG_PATH + _SEP + 'toolboxes' + _SEP + 'photbiochem' + _SEP  + 'data' + _SEP + 'ASNZS_1680.2.5_1997_cyanosisindex_100_50.dat', header = None, kind ='np', verbosity = 0).T

_COI_CIEOBS = '1931_2' # default CMF set

_COI_CSPACE = 'lab'

_COI_REF = blackbody(4000, )

def spd_to_COI_ASNZS1680(S = None, tf = _COI_CSPACE, cieobs = _COI_CIEOBS, out = 'COI,cct', extrapolate_rfl = False):
    """
    Calculate the Cyanosis Observation Index (COI) [ASNZS 1680.2.5-1995].
    
    Args:
        :S:
            | ndarray with light source spectrum (first column are wavelengths).
        :tf:
            | _COI_CSPACE, optional
            | Color space in which to calculate the COI.
            | Default is CIELAB.
        :cieobs: 
            | _COI_CIEOBS, optional
            | CMF set to use. 
            | Default is '1931_2'.
        :out: 
            | 'COI,cct' or str, optional
            | Determines output.
        :extrapolate_rfl:
            | False, optional
            | If False: 
            |  limit the wavelength range of the source to that of the standard
            |  reflectance spectra for the 50% and 100% oxygenated blood.
            
    Returns:
        :COI:
            | ndarray with cyanosis indices for input sources.
        :cct:
            | ndarray with correlated color temperatures.
            
    Note:
        Clause 7.2 of the ASNZS 1680.2.5-1995. standard mentions the properties
        demanded of the light source used in region where visual conditions 
        suitable to the detection of cyanosis should be provided:
        
            1. The correlated color temperature (CCT) of the source should be from 
            3300 to 5300 K.
                
            2. The cyanosis observation index should not exceed 3.3

    """
    
    if S is None: #use default
        S = _CIE_ILLUMINANTS['F4']
    
    if extrapolate_rfl == False: # _COI_RFL do not cover the full 360-830nm range.
        wl_min = _COI_RFL_BLOOD[0].min()
        wl_max = _COI_RFL_BLOOD[0].max()
        S = S[:,np.where((S[0] >= wl_min) & (S[0] <= wl_max))[0]]

    # Calculate reference spd:
    Sr = blackbody(4000, wl3 = S[0]) # same wavelength range

    # Calculate xyz of blood under test source and ref. source:
    xyzt,xyzwt = spd_to_xyz(S, rfl = _COI_RFL_BLOOD, relative = True, cieobs = cieobs, out = 2)
    xyzr,xyzwr = spd_to_xyz(Sr, rfl = _COI_RFL_BLOOD, relative = True, cieobs = cieobs, out = 2)

    # Calculate color difference between blood under test and ref.
    DEi = deltaE.DE_cspace(xyzt,xyzr, xyzwt = xyzwt, xyzwr = xyzwr, tf = tf)
    
    # Calculate Cyanosis Observation Index:
    COI = np.nanmean(DEi, axis = 0)[:,None]
    
    
    # Calculate cct, if requested:
    if 'cct' in out.split(','):
        cct, duv = xyz_to_cct(xyzwt, cieobs = cieobs, out = 2)

    # manage output:
    if out == 'COI':
        return COI
    elif out == 'COI,cct':
        return COI, cct
    else:
        return eval(out)

    
if __name__ == '__main__':
    # test
    S = np.vstack((_CIE_ILLUMINANTS['A'],_CIE_ILLUMINANTS['F4'][1:],_CIE_ILLUMINANTS['F5'][1:]))
    coi, cct = spd_to_COI_ASNZS1680(S,extrapolate_rfl=True)
    
    