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
Module supporting basic spectral calculations.
==============================================

 :_WL3: Default wavelength specification in vector-3 format: 
        ndarray([start, end, spacing])

 :_INTERP_REFERENCE:  Sets the specific interpolation for spectrum types: ['spd','cmf','rfl','none'] 

 :_INTERP_SETTINGS_ALL: Nested Dict with interpolation settings per spectral type ['spd','cmf','rfl','none'] for various interp_reference keys.

 :_INTERP_SETTINGS: Nested Dict with interpolation settings per spectral type ['spd','cmf','rfl','none'].

 :_INTERP_TYPES: Dict with interpolation types associated with various types of
                 spectral data according to CIE recommendation:  

 :_S_INTERP_TYPE: Interpolation type for light source spectral data

 :_R_INTERP_TYPE: Interpolation type for reflective/transmissive spectral data

 :_C_INTERP_TYPE: Interpolation type for CMF and cone-fundamental spectral data

 :getwlr(): Get/construct a wavelength range from a (start, stop, spacing) 
            3-vector.

 :getwld(): Get wavelength spacing of ndarray with wavelengths.

 :spd_normalize(): Spectrum normalization (supports: area, max, lambda, 
                   radiometric, photometric and quantal energy units).
                   
 :cie_interp(): Interpolate / extrapolate spectral data following standard 
                [`CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_]

 :spd(): | All-in-one function that can:
         |  1. Read spectral data from data file or take input directly as ndarray.
         |  2. Interpolate spectral data.
         |  3. Normalize spectral data.

 :xyzbar(): Get color matching functions.
        
 :vlbar(): Get Vlambda function.
 
 :vlbar_cie_mesopic(): Get CIE mesopic luminous efficiency function Vmesm according to CIE191:2010

 :get_cie_mesopic_adaptation(): Get the mesopic adaptation state according to CIE191:2010

 :spd_to_xyz_legacy(): Calculates xyz tristimulus values from spectral data. (luxpy version <= 1.11.4)

 :spd_to_xyz_barebones(): Calculates xyz tristimulus values from equal wavelength spectral data (no additional processing) 

 :spd_to_xyz(): Calculates xyz tristimulus values from spectral data. 
            
 :spd_to_ler():  Calculates Luminous efficacy of radiation (LER) 
                 from spectral data.

 :spd_to_power(): Calculate power of spectral data in radiometric, photometric
                  or quantal energy units.
         
 :detect_peakwl(): Detect peak wavelengths and fwhm of peaks in spectrum spd. 
 
 :create_spectral_interpolator(): Create an interpolator of kind for spectral data S. 
 
 :wls_shift(): Wavelength-shift array S over shft wavelengths.

References
----------

    1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
        
    2. `CIE 191:2010 Recommended System for Mesopic Photometry Based on Visual Performance.
    (ISBN 978-3-901906-88-6), http://cie.co.at/publications/recommended-system-mesopic-photometry-based-visual-performance>`_
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

#--------------------------------------------------------------------------------------------------
import copy
import numpy as np 

from luxpy import  _CIEOBS, math
from luxpy.utils import _PKG_PATH, _SEP, np2d, getdata, _EPS

from .cmf import _CMF
__all__ = ['_BB','_WL3','_INTERP_TYPES','_S_INTERP_TYPE', '_R_INTERP_TYPE','_C_INTERP_TYPE',
           '_SPECTRUM_TYPES','_INTERP_REFERENCE','_INTERP_SETTINGS','_INTERP_SETTINGS_ALL', 
           'getwlr','getwld','spd_normalize','spectral_interp','cie_interp','spd','xyzbar', 'vlbar', 
           'vlbar_cie_mesopic', 'get_cie_mesopic_adaptation',
           'spd_to_xyz', 'spd_to_xyz_barebones','spd_to_ler', 'spd_to_power', 'detect_peakwl',
           'create_spectral_interpolator','wls_shift',
           'spd_to_xyz_legacy']


#--------------------------------------------------------------------------------------------------
# set standard SPD wavelength interval interval and spacing
_WL3 = [360.0,830.0,1.0]
    
#--------------------------------------------------------------------------------------------------
# set coefficients for blackbody radiators (c2 rounded to 1.4388e-2 as defiend for ITS-90 International Temperature Scale):
_BB = {'c1' : 3.74177185e-16, 'c2' : np.round(1.4387768775e-2,6),'n': 1.000, 'na': 1.00028, 'c' : 299792458, 'h' : 6.62607015e-34, 'k' : 1.380649e-23, 'e' : 1.602176634e-19} # blackbody c1,c2 & n standard values (h,k,c,e from NIST, CODATA2018); 'e' = electron charge in Coulomb


#--------------------------------------------------------------------------------------------------
# Define interpolation types (conform CIE15:20xx): 
_SPECTRUM_TYPES = ['spd','cmf','rfl','none']
_INTERP_REFERENCE = 'CIE15:2018'
_INTERP_SETTINGS_ALL = {'CIE15:2018' : {'spd'  : {'itype' : 'cubic',  'etype' : 'linear', 'fill_value' : None, 'negative_values_allowed' : False},
                                    'cmf'  : {'itype' : 'linear', 'etype' : 'linear', 'fill_value' : None, 'negative_values_allowed' : False},
                                    'rfl'  : {'itype' : 'cubic',  'etype' : 'linear', 'fill_value' : None, 'negative_values_allowed' : False},
                                    'none' : {'itype' : 'linear', 'etype' : 'linear', 'fill_value' : None, 'negative_values_allowed' : False}
                                    },
                    'CIE15:2004' : {'spd'  : {'itype' : 'cubic',  'etype' : 'linear',     'fill_value' : None, 'negative_values_allowed' : False},
                                    'cmf'  : {'itype' : 'linear', 'etype' : 'linear',     'fill_value' : None, 'negative_values_allowed' : False},
                                    'rfl'  : {'itype' : 'cubic',  'etype' : 'linear',     'fill_value' : None, 'negative_values_allowed' : False},
                                    'none' : {'itype' : 'linear', 'etype' : 'linear',     'fill_value' : None, 'negative_values_allowed' : False}
                                    },
                    'general' : {'force_scipy_interpolator' : False, 'scipy_interpolator' : 'interp1d',
                                 'sprague_allowed' : False, 'sprague_method' : 'spargue_cie224_2017', 
                                 'choose_most_efficient_interpolator' : False,
                                 'interp_log' : False, 'extrap_log' : False}
                    }
_INTERP_SETTINGS = copy.deepcopy(_INTERP_SETTINGS_ALL[_INTERP_REFERENCE])
_INTERP_SETTINGS['general'] = _INTERP_SETTINGS_ALL['general']
_INTERP_TYPES = {'linear' : ['xyzbar','cmf','lms','undefined','Dxx'],'cubic': ['S', 'spd','SPD','Le','rfl','RFL','r','R'],'none':None}
_INTERP_TYPES['sprague5'] = _INTERP_TYPES['cubic']
_INTERP_TYPES['sprague_cie224_2017'] = _INTERP_TYPES['cubic']
_INTERP_TYPES['lagrange5'] = _INTERP_TYPES['cubic']
_S_INTERP_TYPE = 'cubic' # -> cie_interp(): changes this to Sprague5 for equal wavelength spacings, if explicitely allowed (slower code)!
_R_INTERP_TYPE = 'cubic' # -> cie_interp(): changes this to Sprague5 for equal wavelength spacings, if explicitely allowed (slower code) !
_C_INTERP_TYPE = 'linear'
                    
                    

 


#--------------------------------------------------------------------------------------------------
def getwlr(wl3 = None):
    """
    Get/construct a wavelength range from a 3-vector (start, stop, spacing).
    
    Args:
        :wl3: 
            | list[start, stop, spacing], optional 
            | (defaults to luxpy._WL3)

    Returns:
        :returns: 
            | ndarray (.shape = (n,)) with n wavelengths ranging from
            | start to stop, with wavelength interval equal to spacing.
    """
    if wl3 is None: wl3 = _WL3
    
    # Wavelength definition:
    # wl = wl3 if (len(wl3) != 3) else np.linspace(wl3[0],wl3[1],int(np.floor((wl3[1]-wl3[0]+wl3[2])/wl3[2]))) # define wavelengths from [start = l0, stop = ln, spacing = dl]
    wl = wl3 if (len(wl3) != 3) else np.arange(wl3[0], wl3[1] + wl3[2], wl3[2]) # define wavelengths from [start = l0, stop = ln, spacing = dl]
    return wl

#------------------------------------------------------------------------------
def getwld(wl):
    """
    Get wavelength spacing. 
    
    Args:
        :wl: 
            | ndarray with wavelengths
        
    Returns:
        :returns: 
            | - float:  for equal wavelength spacings
            | - ndarray (.shape = (n,)): for unequal wavelength spacings
    """
    d = np.diff(wl)
    # dl = (np.hstack((d[0],d[0:-1]/2.0,d[-1])) + np.hstack((0.0,d[1:]/2.0,0.0)))
    dl = np.hstack((d[0],(d[0:-1] + d[1:])/2.0,d[-1]))
    # if np.array_equal(dl,dl.mean()*np.ones(dl.shape)): dl = dl[0]
    if (dl == dl.mean()).all(): dl = dl[0]
    return dl


#------------------------------------------------------------------------------
def spd_normalize(data, norm_type = None, norm_f = 1, wl = True, cieobs = _CIEOBS, K = None,
                  interp_settings = None):
    """
    Normalize a spectral power distribution (SPD).
    
    Args:
        :data: 
            | ndarray
        :norm_type: 
            | None, optional 
            |       - 'lambda': make lambda in norm_f equal to 1
            |       - 'area': area-normalization times norm_f
            |       - 'max': max-normalization times norm_f
            |       - 'ru': to :norm_f: radiometric units 
            |       - 'pu': to :norm_f: photometric units 
            |       - 'pusa': to :norm_f: photometric units (with Km corrected
            |                             to standard air, cfr. CIE TN003-2015)
            |       - 'qu': to :norm_f: quantal energy units
        :norm_f:
            | 1, optional
            | Normalization factor that determines the size of normalization 
            | for 'max' and 'area' 
            | or which wavelength is normalized to 1 for 'lambda' option.
        :wl: 
            | True or False, optional 
            | If True, the first column of data contains wavelengths.
        :cieobs:
            | _CIEOBS or str or ndarray, optional
            | Type of cmf set to use for normalization using photometric units 
            | (norm_type == 'pu')
        :K:
            | None, optional
            | Luminous efficacy of radiation.
            | Must be supplied if cieobs is an array for norm_type == 'pu'
    
    Returns:
        :returns: 
            | ndarray with normalized data.
    """
    if norm_type is not None:
        if not isinstance(norm_type,list): norm_type = [norm_type]
        
        if norm_f is not None:
            if not isinstance(norm_f,list): norm_f = [norm_f]
                
        if ('lambda' in norm_type) | ('qu' in norm_type):
            wl = True # for lambda & 'qu' normalization wl MUST be first column
            wlr = data[0]
            
        if (('area' in norm_type) | ('ru' in norm_type) | ('pu' in norm_type) | ('pusa' in norm_type)) & (wl == True):
            dl = getwld(data[0])
        else:
            dl = 1 #no wavelengths provided
            
        offset = int(wl)
        for i in range(data.shape[0]-offset):  
            norm_type_ = norm_type[i] if (len(norm_type)>1) else norm_type[0]

            if norm_f is not None:
                norm_f_ = norm_f[i] if (len(norm_f)>1) else norm_f[0]
            else:
                norm_f_ = 560.0 if (norm_type_ == 'lambda') else 1.0
      
            if norm_type_=='max':
                data[i+offset]=norm_f_*data[i+offset]/np.max(data[i+offset])
            elif norm_type_=='area':
                data[i+offset]=norm_f_*data[i+offset]/(np.sum(data[i+offset])*dl)
            elif norm_type_=='lambda':
                wl_index = np.abs(wlr-norm_f_).argmin()
                data[i+offset]=data[i+offset]/data[i+offset][wl_index]
            elif (norm_type_ == 'ru') | (norm_type_ == 'pu') | (norm_type == 'pusa') | (norm_type_ == 'qu'):
                rpq_power = spd_to_power(data[[0,i+offset],:], cieobs = cieobs, K = K, ptype = norm_type_,
                                         interp_settings = interp_settings)
                data[i+offset] = (norm_f/rpq_power)*data[i+offset]
            else:
                data[i+offset]=data[i+offset]/norm_f_
    return data


#--------------------------------------------------------------------------------------------------
def spectral_interp(data, wl_new, stype = 'cmf', 
                    interp_settings = copy.deepcopy(_INTERP_SETTINGS),
                    itype = None, etype = None, fill_value = None,
                    negative_values_allowed = False,
                    delete_nans = True,
                    force_scipy_interpolator = False,
                    scipy_interpolator = 'InterpolatedUnivariateSpline',
                    interp_log = False, extrap_log = False,
                    choose_most_efficient_interpolator = False,
                    verbosity = 0):
    """
    Perform a 1-dimensional interpolation of spectral data
        
    Args:
        :data: 
            | ndarray with (n+1,N)-dimensional spectral data (0-row: wavelengths, remaining n rows: data)
        :wl_new: 
            | ndarray of new wavelengths (N,)
        :stype:
            | None, optional 
            | Type of spectral data: None or ('spd', 'cmf', 'rfl')
            | If None: itype, etype and fill_value kwargs should not be none!
        :itype:
            | None or str,  optional
            | supported options for str: 'linear', 'quadratic', 'cubic'
            | If None: use value in interp_settings.
        :etype:
            | None, or str, optional
            | options: 
            |   - 'extrapolate','ext': use method specified in :itype: to extrapolate.
            |   - 'zeros': out-of-bounds values are filled with zeros
            |   - 'const': out-of-bounds values are filled with nearest value
            |   - 'fill_value': value of tuple (2,) of values is used to fill out-of-bounds values
            |   - 'linear','quadratic','cubic': use of of these methods (slows down function 
            |       if this method is different from the one in :itype:)
            | If None: use value in intp_settings.
        :fill_value:
            | None or str or float or int or tupple, optional
            | If etype == 'fill_value': use fill_value to set lower- and upper-out-of-bounds values when extrapolating
            | ('extrapolate' when etype requires extrapolation)
            | If None: use value in interp_settings.
        :negative_values_allowed: 
            | False, optional
            | If False: negative values are clipped to zero.
        :delete_nans:
            | True, optional
            | If NaNs are present, remove them and (and try to) interpolate without them.
        :force_scipy_interpolator:
            | False, optional
            | If False: numpy.interp function is used for linear interpolation when no or linear extrapolation is used/required (fast!). 
        :scipy_interpolator:
            | 'InterpolatedUnivariateSpline', optional
            | options: 'InterpolatedUnivariateSpline', 'interp1d'
        :w,bbox,check_finite:
            | see scipy.interpolate.InterpolatedUnivariateSpline()
        :interp_log:
            | Perform interpolation method ('linear', 'quadratic', or 'cubic') in log space.
        :extrap_log:
            | Perform extrapolation method ('linear', 'quadratic', or 'cubic') in log space.
 
    Returns:
        :data_new:
            | ndarray with interpolated (n+1,N)-dimensional spectral data 
            | (0-row: wavelengths, remaining n rows: interpolated data)
    
    Note:
        1. 'numpy.interp' is fastest (but only works for linear interpolation and linear or no extrapolation)
        2. For linear interpolation: 'interp1d' is faster for Y (N,...) with N > 1, else 'InterpolatedUnivariateSpline' is faster
        3. For 'cubic' interpolation: 'InterpolatedUnivariateSpline' is faster for Y (N,...) with N > 1, else 'interp1d' is faster
    """    
    if wl_new.ndim == 2: wl_new = wl_new[0]
    if np.array_equal(wl_new, data[0]): return data 
    
    # Split wavelengths and data:
    wl, data = data[0], data[1:]

    # Deal with possible override of dict keys by kwargs:
    if stype is not None: 
        if itype is not None: interp_settings[stype]['itype'] = itype
        if etype is not None: interp_settings[stype]['etype'] = etype
        if fill_value is not None: interp_settings[stype]['fill_value'] = fill_value
    else:
        stype = 'none'
        interp_settings[stype]['itype'] = itype
        interp_settings[stype]['etype'] = etype
        interp_settings[stype]['fill_value'] = fill_value

    # Interpolate & extrapolate:
    if interp_settings[stype]['itype'] == 'sprague5':
        if (interp_settings[stype]['etype'] == 'fill_value') & ((interp_settings[stype]['fill_value'] != 'ext') | (interp_settings[stype]['fill_value'] != 'extrapolate')):
            extrap = interp_settings[stype]['fill_value']
        else:
            extrap = interp_settings[stype]['etype']
        if verbosity > 0: print('Interpolation/Extrapolation: using luxpy.math.interp1_sprague5.')
        datan = np.vstack((wl_new, math.interp1_sprague5(wl, data, wl_new, 
                                                extrap = extrap, 
                                                force_scipy_interpolator = force_scipy_interpolator,
                                                scipy_interpolator = scipy_interpolator,
                                                delete_nans = delete_nans,
                                                choose_most_efficient_interpolator = choose_most_efficient_interpolator)))
    elif interp_settings[stype]['itype'] == 'sprague_cie224_2017':
        if (interp_settings[stype]['etype'] == 'fill_value') & ((interp_settings[stype]['fill_value'] != 'ext') | (interp_settings[stype]['fill_value'] != 'extrapolate')):
            extrap = interp_settings[stype]['fill_value']
        else:
            extrap = interp_settings[stype]['etype']
        if verbosity > 0: print('Interpolation/Extrapolation: using luxpy.math.interp1_sprague_cie224_2017.')
        datan = np.vstack((wl_new, math.interp1_sprague_cie224_2017(wl, data, wl_new, 
                                                extrap = extrap, 
                                                force_scipy_interpolator = force_scipy_interpolator,
                                                scipy_interpolator = scipy_interpolator,
                                                delete_nans = delete_nans,
                                                choose_most_efficient_interpolator = choose_most_efficient_interpolator)))

    elif interp_settings[stype]['itype'][:8] == 'lagrange':
        k_lagrange = int(interp_settings[stype]['itype'][8:]) if (len(interp_settings[stype]['itype']) > 8) else 5 
        if (interp_settings[stype]['etype'] == 'fill_value') & ((interp_settings[stype]['fill_value'] != 'ext') | (interp_settings[stype]['fill_value'] != 'extrapolate')):
            extrap = interp_settings[stype]['fill_value']
        else:
            extrap = interp_settings[stype]['etype']
        if verbosity > 0: print('Interpolation/Extrapolation: using luxpy.math.interp1_lagrange.')
        datan = np.vstack((wl_new, math.interp1_lagrange(wl, data, wl_new, k = k_lagrange,
                                                extrap = extrap, 
                                                force_scipy_interpolator = force_scipy_interpolator,
                                                scipy_interpolator = scipy_interpolator,
                                                delete_nans = delete_nans,
                                                choose_most_efficient_interpolator = choose_most_efficient_interpolator)))

    else:
        #print('::',interp_settings[stype]['itype'],interp_settings[stype]['etype'],interp_settings[stype]['fill_value'])
        datan = np.vstack((wl_new, math.interp1(wl, data, wl_new,
                                        kind = interp_settings[stype]['itype'],
                                        ext = interp_settings[stype]['etype'],
                                        fill_value = interp_settings[stype]['fill_value'],
                                        force_scipy_interpolator = force_scipy_interpolator,
                                        scipy_interpolator = scipy_interpolator,
                                        delete_nans = delete_nans,
                                        interp_log = interp_log,
                                        extrap_log = extrap_log,
                                        choose_most_efficient_interpolator = choose_most_efficient_interpolator,
                                        verbosity = verbosity)))
        
    # No negative values allowed for spectra:    
    if negative_values_allowed == False:
        if np.any(datan): datan[datan<0.0] = 0.0

    return datan

def _get_itype_sprague_interp_options(itype, sprague_method, wl, wl_new):
    dwl = np.diff(wl)
    if np.all(dwl == dwl[0]):
        dwl_new = np.diff(wl_new)
        if np.all(dwl_new == dwl_new[0]):
            itype = sprague_method # force recommended 5th order Sprague interpolation or Sprague interpolation defined in CIE224-2017 for equal wavelength spacings when kind was a spectral data type. The latter is only available when downsampling 5:1 (e.g. 5 nm -> 1 nm)!
            if (dwl[0]/dwl_new[0] != 5): itype = 'sprague5' # 'sprague_cie224_2017 is only for 5 nm -> 1 nm  
        elif ('sprague' in itype):
            raise Exception('Sprague interpolation requires an equal wavelength spacing!')
    elif ('sprague' in itype):
            raise Exception('Sprague interpolation requires an equal wavelength spacing!')
    return itype

#--------------------------------------------------------------------------------------------------
def cie_interp(data, wl_new, datatype = 'none',
               interp_settings = copy.deepcopy(_INTERP_SETTINGS), 
               kind = None, 
               extrap_kind = None, extrap_values = None,
               sprague_allowed = None, sprague_method = 'sprague_cie224_2017',
               negative_values_allowed = None,
               interp_log = None, extrap_log = None,
               force_scipy_interpolator = None, scipy_interpolator = None,
               choose_most_efficient_interpolator = None,
               verbosity = 0):
    """
    Interpolate / extrapolate spectral data following standard CIE15-2018.
    
    | The kind of interpolation depends on the spectrum type defined in :datatype: 
    |  (or in :kind: for legacy puprposes-> overrules :datatype:). 
    
    Args:
        :data: 
            | ndarray with spectral data 
            | (.shape = (number of spectra + 1, number of original wavelengths))
        :wl_new: 
            | None or ndarray with new wavelengths or [start wavelength, stop wavelength, wavelength interval]
            | If None: no interpolation is done, a copy of the original data is returned.
        :datatype: 
            | 'spd' (light source) or 'rfl' (reflectance) or 'cmf' (color matching functions) or 'none' (undefined), optional
            | Specifies a type of spectral data. 
            | Is used to select the interpolation and extrapolation defaults, specified
            | in :interp_settings:. 
        :interp_settings:
            | _INTERP_SETTINGS or dict, optional
            | Dictionary of dictionaries (see _INTERP_SETTINGS), with at least a key entry 
            | with the interpolation and extrapolation settings for the type specified in 
            | :datatype: (or :kind: if string with spectrum datatype) and one key entry 'none'
            | ('none' is used in case :extrap_kind: is None or 'ext').
            | 
        :kind: 
            | None, optional
            | - If None: the value from interp_settings is used, based on the value of :datatype:.
            | - If :kind: is a spectrum type (see :interp_settings:), the correct 
            |     interpolation type is automatically chosen based on the values in :interp_settings:
            |       (The use of the slow(er) 'sprague5' or 'sprague_cie224_2017' can be toggled on using :sprague_allowed:).
            | - Or :kind: can be 'linear', 'quadratic', 'cubic' (or 'sprague5', or 'sprague_cie224_2017, or 'lagrange5'). 
            |       (see luxpy.spectral_interp?) 
        :sprague_allowed:
            | None, optional
            | If None: the value from interp_settings is used.
            | If True: When kind is a spectral data type that corresponds to 'cubic' interpolation,
            |    then a cubic spline interpolation will be used in case of 
            |    unequal wavelength spacings, otherwise a 5th order Sprague or Sprague as defined in CIE224-2017 will be used.
            | If False: always use 'cubic', don't use 'sprague5' or 'sprague_cie224_2017'. 
            |           This is the default, as differences are minimal and 
            |           use of the 'sprague' functions is a lot slower ('sprague5' = slowest )!
        :sprague_method:
            | 'sprague_cie224_2017', optional
            | Specific sprague method used for interpolation. (Only for equal spacings, 'sprague_cie224_2017' also on for 5 nm -> 1nm)
            | - options: 'sprague5' (use luxpy.math.interp1_sprague5), 'sprague_cie224_2017' (use luxpy.interp1_sprague_cie224_2017)
        :negative_values_allowed: 
            | None, optional
            | If None: the value from interp_settings is used.
            | If False: negative values are clipped to zero.
        :extrap_kind:
            | None, optional
            | If None or 'ext': use the method specified interp_settings[datatype].
            | If 'kind' or 'itype':
            |   - If possible, use the same method as the interpolation method 
            |         (only for 'linear', 'quadratic', 'cubic'), 
            |   - otherwise: use the method specified :interp_settings['none']:.
            | Other options: 'linear' (or 'cie167:2005'), 'quadratic' (or 'cie15:2018'), 
            |           'nearest' (or 'cie15:2004' or 'const' or 'flat'), 'cubic', 'fill_value' (use value(s)n in extrap_values)
            |   - If 'linear','quadratic','cubic': slow down of function 
            |       in case this method is different from the interpolation method used.
            | CIE15:2018 states that based on a 2017 paper by Wang that 'quadratic' is 'better'. 
            | However, no significant difference was found between 'quadratic' and 'linear' methods.
            | Also see note 1 below, for why the CIE67:2005 recommended 'linear' extrapolation
            | is set as the default.
        :extrap_values:
            | None, optional
            | If float or list or ndarray, use those values to fill extrapolated value(s) when :extrap_kind:S == 'fill_value'.
        :extrap_log:
            | None, optional
            | If None: the value from interp_settings is used.
            | If True: extrap the log of the spectral values 
            |     (not CIE recommended but in most cases seems to give a 
            |     more realistic estimate, but can sometimes seriously fail, 
            |     especially for the 'quadratic' extrapolation case (see note 1)!!!)
            | If any zero or negative values are present in a spectrum, then the log is NOT taken.
        :interp_log:
            | None, optional
            | If None: the value from interp_settings is used.
            | Take log before interpolating the spectral data, afterwards take exp of interpolated data.
            | If any zero or negative values are present in a spectrum, then the log is NOT taken.
        :force_scipy_interpolator:
            | None, optional
            | If None: the value from interp_settings is used.
            | If False: numpy.interp function is used for linear interpolation when no or linear extrapolation is used/required (fast!). 
        :scipy_interpolator:
            | None, optional
            | If None: the value from interp_settings is used.
            | options: 'InterpolatedUnivariateSpline', 'interp1d'

    Returns:
        :returns: 
            | ndarray of interpolated spectral data.
            | (.shape = (number of spectra + 1, number of wavelength in wl_new))
    
    Notes:
        | 1. Type of extrapolation: 'quadratic' vs 'linear'; impact of extrapolating log spectral values:
        |       Using a 'linear' or 'quadratic' extrapolation, as mentioned in 
        |       CIE167:2005 and CIE15:2018, resp., can lead to extreme large values 
        |       when setting :extrap_log: (not CIE recommended) to True. 
        |       A quick test with the IES TM30 spectra (400 nm - 700 nm, 5 nm spacing) 
        |       shows that 'linear' is better than 'quadratic' in terms of 
        |       mean, median and max DEu'v' with the original spectra (380 nm - 780 nm, 5 nm spacing).
        |       This conferms the recommendation from CIE167:2005 to use 'linear' extrapolation.
        |       Setting :extrap_log: to True reduces the median, but inflates the mean due to some
        |       extremely large DEu'v' values. However, the increase in mean and max DEu'v' is much 
        |       larger for the 'quadratic' case, suggesting that 'linear' extrapolation 
        |       is likely a more suitable recommendation. When using a 1 nm spacing
        |       'linear' is more similar to 'quadratic' when :extrap_log: is False, otherwise 'linear'
        |       remains the 'best'. Hence the choice to use the CIE167:2005 recommended linear extrapolation as default!
    """
    if (wl_new is None):
        return data.copy()
    else:

        # Make sure interp_settings is dict with minimum required entries
        if interp_settings is None: 
            interp_settings = _INTERP_SETTINGS
        else: 
            if ('general' not in interp_settings): interp_settings['general'] = copy.deepcopy(_INTERP_SETTINGS['general'])
            if ('none' not in interp_settings): interp_settings['none'] = copy.deepcopy(_INTERP_SETTINGS['none'])
            if (((kind is None) | (extrap_kind is None)) & (datatype not in interp_settings)):
                raise Exception("Interpolation and extrapolation methods for specified datatype not in interp_settings.")
            
        # Wavelength definition:
        wl_new = getwlr(wl_new)
        wl = data[0]

        # Set interpolation type based on data type:
        if kind is None: 
            itype = interp_settings[datatype]['itype']
        elif kind in _SPECTRUM_TYPES:
            datatype = kind # override with kind for legacy purposes.
            itype = interp_settings[datatype]['itype']
        elif kind in ('linear','quadratic','cubic'): 
            itype = kind
        elif kind in ('sprague5', 'sprague_cie224_2017', 'lagrange5'):
            itype = kind
            if 'sprague' in kind: sprague_method = kind
        else:
            raise Exception("Unsupported interpolation type, kind = {:s}.\n Options = None or 'linear', 'quadratic', 'cubic'  or 'spd', 'cmf', 'rfl'.".format(kind)) 
        if (itype == 'cubic'):
            if sprague_allowed: 
                itype = _get_itype_sprague_interp_options(itype, sprague_method, wl, wl_new)
        elif ('sprague' in itype):
            itype = _get_itype_sprague_interp_options(itype, sprague_method, wl, wl_new)
   
        # Set extrapolation type based on CIE report:
        if extrap_kind is None: 
            etype = interp_settings[datatype]['etype']
        elif extrap_kind[:3] == 'ext':
            etype = interp_settings[datatype]['etype']
        elif (extrap_kind == 'itype') | (extrap_kind == 'kind'):
            if itype in ('linear','quadratic','cubic'): 
                etype = itype
            else:
                etype = interp_settings['none']['etype']
        elif extrap_kind in ('nearest','flat','zeros','const','linear','quadratic','cubic','fill_value'):
            etype = extrap_kind
        elif extrap_kind == 'cie167:2005':
            etype = 'linear'
        elif extrap_kind == 'cie15:2018':
            etype = 'quadratic'
        elif extrap_kind == 'cie15:2004':
            etype = 'nearest'
        else:
            raise Exception("Unsupported extrapolation type, extrap_kind = {}.\n - Options: None or 'nearest',[= 'flat', 'const'],'zeros','linear','quadratic','cubic','fill_value'".format(extrap_kind))
        
        if (extrap_values is None): extrap_values = interp_settings[datatype]['fill_value']
        if (negative_values_allowed is None): negative_values_allowed = interp_settings[datatype]['negative_values_allowed'] 
        if (force_scipy_interpolator is None): force_scipy_interpolator = interp_settings['general']['force_scipy_interpolator']
        if (scipy_interpolator is None): scipy_interpolator = interp_settings['general']['scipy_interpolator'] 
        if (interp_log is None): interp_log = interp_settings['general']['interp_log'] 
        if (extrap_log is None): extrap_log = interp_settings['general']['extrap_log']

        if (choose_most_efficient_interpolator is None): choose_most_efficient_interpolator = interp_settings['general']['choose_most_efficient_interpolator']
 
        return spectral_interp(data, wl_new, stype = None, 
                                itype = itype, etype = etype, fill_value = extrap_values,
                                negative_values_allowed = negative_values_allowed,
                                delete_nans = True,
                                force_scipy_interpolator = force_scipy_interpolator,
                                scipy_interpolator = scipy_interpolator,
                                interp_log = interp_log, extrap_log = extrap_log,
                                choose_most_efficient_interpolator = choose_most_efficient_interpolator,
                                verbosity = verbosity)

#--------------------------------------------------------------------------------------------------
def spd(data = None, wl = None, \
        interp_settings = None, \
        kind = None, extrap_kind = None, extrap_values = None, \
        sep = ',',header = None, datatype = 'spd', \
        norm_type = None, norm_f = None, **kwargs):
    """
    | All-in-one function that can:
    |    1. Read spectral data from data file or take input directly as ndarray.
    |    2. Interpolate spectral data.
    |    3. Normalize spectral data.
            
    Args:
        :data: 
            | - str with path to file containing spectral data
            | - ndarray with spectral data
            | (.shape = (number of spectra + 1, number of original wavelengths))
        :wl: 
            | None, optional
            | New wavelength range for interpolation. 
            | If None: no interpolation will be done.
        :kind:
            | None, optional
            | - None: use defaults in interp_settings for specified datatype.
            | - str with interpolation type or spectrum type (if spectrum type: overrides anything set in :datatype:)
        :extrap_kind:
            | None, optional
            | - None: use defaults in interp_settings for specified datatype.
            | - str with extrapolation type
        :extrap_values:
            | None, optional
            | Controls extrapolation. See cie_interp.
        :header: 
            | None or 'infer', optional
            | - None: no header in file
            | - 'infer': infer headers from file
        :sep: 
            | ',' or '\t' or other char, optional
            | Column separator in case :data: specifies a data file. 
        :datatype': 
            | 'spd' (light source) or 'rfl' (reflectance) or 'cmf' (color matching functions) or 'none' (undefined), optional
            | Specifies a type of spectral data. 
            | Is used to determine interpolation and extrapolation defaults. 
        :norm_type: 
            | None, optional 
            |       - 'lambda': make lambda in norm_f equal to 1
            |       - 'area': area-normalization times norm_f
            |       - 'max': max-normalization times norm_f
            |       - 'ru': to :norm_f: radiometric units 
            |       - 'pu': to :norm_f: photometric units 
            |       - 'pusa': to :norm_f: photometric units (with Km corrected
            |                             to standard air, cfr. CIE TN003-2015)
            |       - 'qu': to :norm_f: quantal energy units
        :norm_f:
            | 1, optional
            | Normalization factor that determines the size of normalization 
            |  for 'max' and 'area' 
            |  or which wavelength is normalized to 1 for 'lambda' option.
    
    Returns:
        :returns: 
            | ndarray with interpolated and/or normalized spectral data.
    """
    transpose = True if isinstance(data,str) else False #when spd comes from file -> transpose (columns in files should be different spectra)
         
    # Data input:
    if data is not None:
        if (wl is None) & (norm_type is None):
            data = getdata(data = data, sep = sep, header = header, datatype = datatype, copy = True)
            if (transpose == True): data = data.T
        else:
            data = getdata(data = data, sep = sep, header = header, datatype = datatype, copy = True)#interpolation requires np-array as input
            if (transpose == True): data = data.T
            # if kind in _SPECTRUM_TYPES: datatype = kind
            # if kind is None: kind = interp_settings[datatype]['itype'] 
            # if extrap_kind is None: extrap_kind = interp_settings[datatype]['etype']
            # if extrap_values is None: extrap_values = interp_settings[datatype]['fill_value']
            if interp_settings is None: interp_settings = copy.deepcopy(_INTERP_SETTINGS)
            if kind in _SPECTRUM_TYPES: datatype = kind
            if kind is not None: interp_settings[datatype]['itype'] = kind
            if extrap_kind is not None: interp_settings[datatype]['etype'] = extrap_kind
            if extrap_values is not None: interp_settings[datatype]['fill_value'] = extrap_values

            data = cie_interp(data = data, wl_new = wl, datatype = datatype, interp_settings = interp_settings)
            data = spd_normalize(data, norm_type = norm_type, norm_f = norm_f, wl = True, interp_settings = interp_settings)
        
    else:
        # Wavelength definition:
        if wl is None: wl = _WL3
        data = np2d(getwlr(wl))
       
    # convert to desired kind:
    data = getdata(data = data, datatype = datatype, copy = False) # already copy when data is not None, else new anyway
        
    return data


#--------------------------------------------------------------------------------------------------
def xyzbar(cieobs = _CIEOBS, src = 'dict', wl_new = None, 
           interp_settings = None,
           kind = None, extrap_kind = None, extrap_values = None):
    """
    Get color matching functions.  
    
    Args:
        :cieobs: 
            | luxpy._CIEOBS, optional
            | Sets the type of color matching functions to load.
        :src: 
            | 'dict' or 'file', optional
            | Determines whether to load cmfs from file (./data/cmfs/) 
            | or from dict defined in .cmf.py
        :wl: 
            | None, optional
            | New wavelength range for interpolation. 
            | If None: no interpolation is done.
        :kind:
            | None, optional
            | - None: use defaults in interp_settings for "cmf" datatype.
            | - str with interpolation type
        :extrap_kind:
            | None, optional
            | - None: use defaults in interp_settings for specified datatype.
            | - str with extrapolation type
        :extrap_values:
            | None, optional
            | Controls extrapolation. See cie_interp.

    Returns:
        :returns: 
            | ndarray with CMFs 
        
            
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    if src == 'file':
        dict_or_file = _PKG_PATH + _SEP + 'data' + _SEP + 'cmfs' + _SEP + 'ciexyz_' + cieobs + '.dat'
    elif src == 'dict':
        dict_or_file = _CMF[cieobs]['bar']
    elif src == 'cieobs':
        dict_or_file = cieobs #can be file or data itselfµ
    if extrap_values is None: extrap_values = (np.nan, np.nan)
    return spd(data = dict_or_file, wl = wl_new, datatype = 'cmf', 
                interp_settings = interp_settings,
                kind = kind, extrap_kind = extrap_kind, extrap_values = extrap_values)

#--------------------------------------------------------------------------------------------------
def vlbar(cieobs = _CIEOBS, K = None, src = 'dict', wl_new = None, 
          interp_settings = None, 
          kind = None, extrap_kind = None, extrap_values = None, 
          out = 1):
    """
    Get Vlambda functions.  
    
    Args:
        :cieobs: 
            | str or ndarray, optional
            | If str: Sets the type of Vlambda function to obtain.
        :K:
            | None, optional
            | Luminous efficacy of radiation.
            | Must be supplied if cieobs is an array
        :src: 
            | 'dict' or array, optional
            | - 'dict': get from ybar from _CMF
            | - 'array': ndarray in :cieobs:
            | Determines whether to load cmfs from file (./data/cmfs/) 
            | or from dict defined in .cmf.py
            | Vlambda is obtained by collecting Ybar.
        :wl: 
            | None, optional
            | New wavelength range for interpolation. 
            | If None: no interpolation is done.
        :kind:
            | None, optional
            | - None: use defaults in interp_settings for "cmf" datatype.
            | - str with interpolation type
        :extrap_kind:
            | None, optional
            | - None: use defaults in interp_settings for specified datatype.
            | - str with extrapolation type
        :extrap_values:
            | None, optional
            | Controls extrapolation. See cie_interp.
        :out: 
            | 1 or 2, optional
            |     1: returns Vlambda
            |     2: returns (Vlambda, Km)
    
    Returns:
        :returns: 
            | ndarray with Vlambda of type :cieobs: 
        
            
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    if src == 'dict':
        dict_or_file = _CMF[cieobs]['bar'][[0,2],:] 
        K = _CMF[cieobs]['K']
    elif src == 'vltype':
        dict_or_file = cieobs #can be file or data itself
        if K is None: K = 1
    if extrap_values is None: extrap_values = (np.nan, np.nan)
    Vl = spd(data = dict_or_file, wl = wl_new, datatype = 'cmf', 
             interp_settings = interp_settings,
             kind = kind, extrap_kind = extrap_kind, extrap_values = extrap_values)
    if Vl.shape[0] > 2: Vl = Vl[[0,2]]
    if out == 2:
        return Vl, K
    else:
        return Vl

#--------------------------------------------------------------------------------------------------
def vlbar_cie_mesopic(m = [1], wl_new = None, out = 1,
                      Lp = None, Ls = None, SP = None,
                      interp_settings = None, 
                      kind = None, extrap_kind = None, extrap_values = None):
    """
    Get CIE mesopic luminous efficiency function Vmesm according to CIE191:2010
    
    Args:
        :m:
            | float or list or ndarray with mesopic adaptation coefficients
        :wl: 
            | None, optional
            | New wavelength range for interpolation. 
            | If None: no interpolation is done.
        :out: 
            | 1 or 2, optional
            |     1: returns Vmesm
            |     2: returns (Vmes, Kmesm)
        :Lp: 
            | None, optional
            | float or ndarray with photopic adaptation luminance
            | If not None: use this (and SP or Ls) to calculate the 
            | mesopic adaptation coefficient
        :Ls: 
            | None, optional
            | float or ndarray with scotopic adaptation luminance
            | If None: SP must be supplied.
        :SP:
            | None, optional
            | S/P ratio
            | If None: Ls must be supplied.
        :kind:
            | None, optional
            | - None: use defaults in interp_settings for "cmf" datatype.
            | - str with interpolation type
        :extrap_kind:
            | None, optional
            | - None: use defaults in interp_settings for specified datatype.
            | - str with extrapolation type
        :extrap_values:
            | None, optional
            | Controls extrapolation. See cie_interp.
        
    Returns:
        :Vmes: 
            | ndarray with mesopic luminous efficiency function 
            | for adaptation coefficient(s) m
        :Kmes:
            | ndarray with luminous efficacies of 555 nm monochromatic light
            | for for adaptation coefficient(s) m
    
    Reference:
        1. `CIE 191:2010 Recommended System for Mesopic Photometry Based on Visual Performance.
        (ISBN 978-3-901906-88-6  ), <http://cie.co.at/publications/recommended-system-mesopic-photometry-based-visual-performance>`_
    """
    if (Lp is not None) & ((Ls is not None) | (SP is not None)):
        Lmes, m = get_cie_mesopic_adaptation(Lp = Lp, Ls = Ls, SP = SP)
    else:
        Lmes = None
    m = np.atleast_2d(m).T
    m[m<0] = 0
    m[m>1] = 1
    Vl = vlbar(cieobs='1931_2', interp_settings = interp_settings)#, wl_new = wl_new, interp_settings = interp_settings, kind = kind, extrap_kind = extrap_kind, extrap_values = extrap_values)
    Vlp = vlbar(cieobs='1951_20_scotopic', interp_settings = interp_settings)#, wl_new = wl_new, interp_settings = interp_settings, kind = kind, extrap_kind = extrap_kind, extrap_values = extrap_values)
    Vlmes= m*(Vl[1:,:]) + (1-m)*(Vlp[1:,:])
    Vlmes = np.vstack((Vl[:1,:],Vlmes))
    Kmes = 683/Vlmes[1:,Vlmes[0,:] == 555]
    Vlmes[1:,:] = Vlmes[1:,:]/Vlmes[1:,:].max(axis=1,keepdims=True) # normalize to max = 1
    
    Vlmes = spd(data = Vlmes, wl = wl_new, datatype = 'cmf',
                interp_settings = interp_settings, 
                kind = kind, extrap_kind = extrap_kind, extrap_values = extrap_values,
                norm_type = 'max', norm_f = 1)
    
    if out == 2:
        return Vlmes, Kmes
    elif out == 1:
        return Vlmes
    else:
        return eval(out)

def get_cie_mesopic_adaptation(Lp, Ls = None, SP = None):
    """
    Get the mesopic adaptation state according to CIE191:2010
    
    Args:
        :Lp: 
            | float or ndarray with photopic adaptation luminance
        :Ls: 
            | None, optional
            | float or ndarray with scotopic adaptation luminance
            | If None: SP must be supplied.
        :SP:
            | None, optional
            | S/P ratio
            | If None: Ls must be supplied.
            
    Returns:
        :Lmes: 
            | mesopic adaptation luminance
        :m: 
            | mesopic adaptation coefficient
    Reference:
        1. `CIE 191:2010 Recommended System for Mesopic Photometry Based on Visual Performance.
        (ISBN 978-3-901906-88-6 ), <http://cie.co.at/publications/recommended-system-mesopic-photometry-based-visual-performance>`_
    """
    Lp = np.atleast_1d(Lp)
    Ls = np.atleast_1d(Ls)
    SP = np.atleast_1d(SP)
    if not (None in SP):
        Ls = Lp*SP
    elif not (None in Ls):
        SP = Ls/Lp
    else:
        raise Exception('Either the S/P ratio or the scotopic luminance Ls must be supplied in addition to the photopic luminance Lp')
    m = np.ones_like(Ls)*np.nan
    Lmes = m.copy()
    for i in range(Lp.shape[0]):
        mi_ = 0.5
        fLmes = lambda m, Lp, SP: ((m*Lp) + (1-m)*SP*683/1699)/(m + (1-m)*683/1699)
        fm = lambda m, Lp, SP: 0.767 + 0.3334*np.log10(fLmes(m, Lp, SP))
        mi = fm(mi_, Lp[i],SP[i])
        while True:
            if np.isclose(mi,mi_): break
            mi_ = mi
            mi = fm(mi_, Lp[i],SP[i])
        m[i] = mi
        Lmes[i] = fLmes(mi, Lp[i],SP[i])
    return Lmes, m

#--------------------------------------------------------------------------------------------------
def spd_to_xyz_legacy(data,  relative = True, rfl = None, cieobs = _CIEOBS, K = None, out = None, cie_std_dev_obs = None):
    """
    Calculates xyz tristimulus values from spectral data.
       
    Args: 
        :data: 
            | ndarray with spectral data
            | (.shape = (number of spectra + 1, number of wavelengths))
            | Note that :data: is never interpolated, only CMFs and RFLs. 
            | This way interpolation errors due to peaky spectra are avoided. 
            | Conform CIE15-2018.
        :relative: 
            | True or False, optional
            | Calculate relative XYZ (Yw = 100) or absolute XYZ (Y = Luminance)
        :rfl: 
            | ndarray with spectral reflectance functions.
            | Will be interpolated if wavelengths do not match those of :data:
        :cieobs:
            | luxpy._CIEOBS or str, optional
            | Determines the color matching functions to be used in the 
            | calculation of XYZ.
        :K: 
            | None, optional
            |   e.g.  K  = 683 lm/W for '1931_2' (relative == False) 
            |   or K = 100/sum(spd*dl)        (relative == True)
        :out:
            | None or 1 or 2, optional
            | Determines number and shape of output. (see :returns:)
        :cie_std_dev_obs: 
            | None or str, optional
            | - None: don't use CIE Standard Deviate Observer function.
            | - 'f1': use F1 function.
    
    Returns:
        :returns:
            | If rfl is None:
            |    If out is None: ndarray of xyz values 
            |        (.shape = (data.shape[0],3))
            |    If out == 1: ndarray of xyz values 
            |        (.shape = (data.shape[0],3))
            |    If out == 2: (ndarray of xyz, ndarray of xyzw) values
            |        Note that xyz == xyzw, with (.shape = (data.shape[0],3))
            | If rfl is not None:
            |   If out is None: ndarray of xyz values 
            |         (.shape = (rfl.shape[0],data.shape[0],3))
            |   If out == 1: ndarray of xyz values 
            |       (.shape = (rfl.shape[0]+1,data.shape[0],3))
            |        The xyzw values of the light source spd are the first set 
            |        of values of the first dimension. The following values 
            |       along this dimension are the sample (rfl) xyz values.
            |    If out == 2: (ndarray of xyz, ndarray of xyzw) values
            |        with xyz.shape = (rfl.shape[0],data.shape[0],3)
            |        and with xyzw.shape = (data.shape[0],3)
             
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    
    data = np2d(data) if isinstance(data,np.ndarray) else getdata(data) # convert to np format and ensure 2D-array

    # get wl spacing:
    dl = getwld(data[0])
    
    # get cmf,k for cieobs:
    if isinstance(cieobs,str):
        if K is None: K = _CMF[cieobs]['K']
        src = 'dict'
    else:
        src = 'cieobs'
        if (K is None) & (relative == False): K = 1
    
    # Interpolate to wl of data:
    cmf = xyzbar(cieobs = cieobs, src = src, wl_new = data[0]) 
    
    # Add CIE standard deviate observer function to cmf if requested:
    if cie_std_dev_obs is not None:
        cmf_cie_std_dev_obs = xyzbar(cieobs = 'cie_std_dev_obs_' + cie_std_dev_obs.lower(), src = src, wl_new = data[0])
        cmf[1:] = cmf[1:] + cmf_cie_std_dev_obs[1:] 
    
    # Rescale xyz using k or 100/Yw:
    if relative == True: K = 100.0/np.dot(data[1:],cmf[2,:]*dl)

    # Interpolate rfls to lambda range of spd and calculate xyz:
    if rfl is not None: 
        rfl = cie_interp(data=np2d(rfl),wl_new = data[0],kind = 'rfl')
        rfl = np.concatenate((np.ones((1,data.shape[1])),rfl[1:])) #add rfl = 1 for light source spectrum
        xyz = K*np.array([np.dot(rfl,(data[1:]*cmf[i+1,:]*dl).T) for i in range(3)])#calculate tristimulus values
        rflwasnotnone = 1
    else:
        rfl = np.ones((1,data.shape[1]))
        xyz = (K*(np.dot((cmf[1:]*dl),data[1:].T))[:,None,:])
        rflwasnotnone = 0
    xyz = np.transpose(xyz,[1,2,0]) #order [rfl,spd,xyz]
    
    # Setup output:
    if out == 2:
        xyzw = xyz[0,...]
        xyz = xyz[rflwasnotnone:,...]
        if rflwasnotnone == 0: xyz = np.squeeze(xyz,axis = 0)
        return xyz,xyzw
    elif out == 1:
        if rflwasnotnone == 0: xyz = np.squeeze(xyz,axis = 0)
        return xyz
    else: 
        xyz = xyz[rflwasnotnone:,...]
        if rflwasnotnone == 0: xyz = np.squeeze(xyz,axis = 0)
        return xyz
        
def spd_to_xyz_barebones(spd, cmf, K = 1.0, relative = True, rfl = None, wl = None, matmul = True):
    """
    Calculate tristimulus values from equal wavelength spectral data.

    Args:
        :spd: 
            | ndarray with (N+1,number of wavelengths)-dimensional spectral data (0-row: wavelengths, remaining n rows: data)
        :cmf:
            | color matching functions (3+1,number of wavelengths). (0-row: spectral wavelengths)
        :K: 
            | 1.0, optional
            |   e.g.  K  = 683 lm/W for '1931_2' (relative == False) 
            |   or K = 100/sum(spd*dl)        (relative == True)
        :relative:
            | False, optional
            | If False: use K, else calculate K = 100 ./ Yw
        :rfl: 
            | None, optional 
            | If not None, must be ndarray with (M+1,number of wavelengths)-dimensional spectral reflectance data (0-row: wavelengths, remaining n rows: data)
        :wl: 
            | None, optional 
            | If None: first row of all spectral data are the wavelengths, else wl is ndarray with corresponding wavelengths of shape (number of wavelength,).
        :matmul:
            | True, optional
            | If True: use matrix multiplication and broadcasting to calculate tristimulus values, else use sumproduct with loop over cmfs.

    Returns:
        :XYZ, XYZw:
            | ndarrays with tristimulus values (X,Y,Z are on last dimension)
            |  - XYZ: tristim. values of all rfls (if rfl is None: same as XYZw) [M,N,3]
            |  - XYZw: tristim. values of all white points (purely spds are used) [N,3] 

    """
    if rfl is None: 
        rfl = np.ones(((wl is None)+ 2,cmf.shape[-1]))
        
    if wl is None: 
        wl, spd, cmf, rfl = spd[0], spd[1:], cmf[1:], rfl[1:]
    
    dl = getwld(wl) 
        
    # Compute the xyz values
    if matmul: 
        xyz = (((dl* rfl[:,None,:]) * spd[None]) @ cmf.T)
    else:
        dl_x_rfl_x_s = dl*rfl[:,None,:]*spd[None]
        xyz = np.transpose(([np.inner(dl_x_rfl_x_s,cmf[i]) for i in range(cmf.shape[0])]),(2,1,0))

    if relative: K = 100 / xyz[:1,:,1:2]
    
    xyz *= K 

    # Setup output:
    return xyz[1:,...], xyz[0,...]

    
    
def spd_to_xyz(spds, cieobs = _CIEOBS, K = None, relative = True, rfl = None,
               out = None, cie_std_dev_obs = None,
               rounding = None, matmul = True, 
               interpolate_to = 'spd',
               interp_settings = _INTERP_SETTINGS,
               kind = None, extrap_kind = None, extrap_values = None,
               negative_values_allowed = None, 
               sprague_allowed = None,
               sprague_method = 'sprague_cie224_2017',
               force_scipy_interpolator = None,
               scipy_interpolator = None,
               choose_most_efficient_interpolator = None, 
               verbosity = 0):
    """
    Calculate tristimulus values from spectral data.

    Args:
        :spds: 
            | ndarray with (N+1,number of wavelengths)-dimensional spectral data (0-row: wavelengths, remaining n rows: data)
        :cieobs:
            | luxpy._CIEOBS or str or ndarray, optional
            | Determines the color matching functions to be used in the 
            | calculation of XYZ.
            | If ndarray: color matching functions (3+1,number of wavelengths). (0-row: spectral wavelengths)
        :K: 
            | None, optional
            |   e.g.  K  = 683 lm/W for '1931_2' (relative == False) 
            |   or K = 100/sum(spd*dl)        (relative == True)
        :relative:
            | True, optional
            | If False: use K, else calculate K = 100 ./ Yw
        :rfl: 
            | None, optional 
            | If not None, must be ndarray with (M+1,number of wavelengths)-dimensional spectral reflectance data (0-row: wavelengths, remaining n rows: data)
        :out:
            | None or 1 or 2, optional
            | Determines number and shape of output. (see :returns:)
        :cie_std_dev_obs: 
            | None or str, optional
            | - None: don't use CIE Standard Deviate Observer function.
            | - 'f1': use F1 function.
        :matmul:
            | True, optional
            | If True: use matrix multiplication and broadcasting to calculate tristimulus values, else use sumproduct with loop over cmfs.
        :rounding:
            | None, optional
            | if not None: round xyz output to this many decimals. (see math.round for more options).
        :interpolate_to:
            | 'spd', optional
            | Interpolate other spectral data to the wavelengths of specified spectral type.
            | Options: 'spd' or 'cmf'
        :interp_settings:
            | Nested Dict with interpolation settings per spectral type ['spd','cmf','rfl','none'].
            | Keys per spectrum type: 
            |   - 'itype': str
            |              supported options for str: 'linear', 'quadratic', 'cubic'
            |   - 'etype': str
            |              supported options: 
            |                   + 'extrapolate'
            |                   + 'zeros': out-of-bounds values are filled with zeros
            |                   + 'const': out-of-bounds values are filled with nearest value
            |                   + 'fill_value': value of tuple (2,) of values is used to fill out-of-bounds values
            |   - 'fill_value': str or float or int or tupple, optional
            |              If ext == 'fill_value': use fill_value to set lower- and upper-out-of-bounds values when extrapolating
            |               ('extrapolate' when etype requires extrapolation)
        :negative_values_allowed: 
            | None, optional
            | If False: after interpolation/extrapolation, any negative values are clipped to zero.
            | If None: use the value in the interp_settings dictionary.
        :force_scipy_interpolator:
            | None, optional
            | If False: numpy.interp function is used for linear interpolation when no or linear extrapolation is used/required (fast!). 
            | If None: use the value in the interp_settings dictionary.
        :scipy_interpolator:
            | None, optional
            | options: 'InterpolatedUnivariateSpline', 'interp1d'
            | If None: use the value in the interp_settings dictionary.
        :choose_most_efficient_interpolator:
            | None, optional
            | If True: Choose most efficient interpolator
            | If None: use the value in the interp_settings dictionary.
        
    Returns:
        :returns:
            | If rfl is None:
            |    If out is None: ndarray of xyz values 
            |        (.shape = (data.shape[0],3))
            |    If out == 1: ndarray of xyz values 
            |        (.shape = (data.shape[0],3))
            |    If out == 2: (ndarray of xyz, ndarray of xyzw) values
            |        Note that xyz == xyzw, with (.shape = (data.shape[0],3))
            | If rfl is not None:
            |   If out is None: ndarray of xyz values 
            |         (.shape = (rfl.shape[0],data.shape[0],3))
            |   If out == 1: ndarray of xyz values 
            |       (.shape = (rfl.shape[0]+1,data.shape[0],3))
            |        The xyzw values of the light source spd are the first set 
            |        of values of the first dimension. The following values 
            |       along this dimension are the sample (rfl) xyz values.
            |    If out == 2: (ndarray of xyz, ndarray of xyzw) values
            |        with xyz.shape = (rfl.shape[0],data.shape[0],3)
            |        and with xyzw.shape = (data.shape[0],3)
    
     References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    
    spds = np2d(spds) if isinstance(spds,np.ndarray) else getdata(spds) # convert to np format and ensure 2D-array

    # get cmf, K data:
    if isinstance(cieobs,str):
        if K is None: K = _CMF[cieobs]['K']
        cmf = _CMF[cieobs]['bar']
    else:
        if (K is None) & (relative == False): K = 1
        cmf = cieobs

    # fintp = lambda data, wl, stype: spectral_interp(data, wl, stype, interp_settings = interp_settings,
    #                                                 delete_nans = True, 
    #                                                 negative_values_allowed = negative_values_allowed,
    #                                                 force_scipy_interpolator = force_scipy_interpolator,
    #                                                 scipy_interpolator = scipy_interpolator,
    #                                                 choose_most_efficient_interpolator = choose_most_efficient_interpolator)
    
    fintp = lambda data, wl, stype: cie_interp(data, wl, datatype = stype, interp_settings = interp_settings,
                                                kind = kind, extrap_kind = extrap_kind, extrap_values = extrap_values,
                                                sprague_allowed = sprague_allowed, sprague_method = sprague_method,
                                                negative_values_allowed = negative_values_allowed,
                                                interp_log = False, extrap_log = False,
                                                force_scipy_interpolator = force_scipy_interpolator,
                                                scipy_interpolator = scipy_interpolator,
                                                choose_most_efficient_interpolator = choose_most_efficient_interpolator, 
                                                verbosity = verbosity)
    
    # Interpolate spectral input:
    if interpolate_to == 'spd':
        wl = spds[0]
        # Interpolate cmf set
        if verbosity > 0: print('Interpolate/Extrapolate cmfs')
        cmf = fintp(cmf, wl, 'cmf')[1:]
        s = spds[1:]
    else:
        wl = cmf[0]
        # Interpolate spd set
        if verbosity > 0: print('Interpolate/Extrapolate spds')
        s = fintp(spds, wl, 'spd')[1:]
        cmf = cmf[1:]
    if rfl is not None:
        eew = np.ones_like(wl)
        rflwasnotnone = True
        # Interpolate rfl set
        if verbosity > 0: print('Interpolate/Extrapolate rfls')
        rfl = fintp(rfl, wl, 'rfl')[1:]
        rfl = np.vstack((eew, rfl))
    else:
        eew = None
        rflwasnotnone = False
        rfl = np.ones((2,wl.shape[0]))

    # Add CIE standard deviate observer function to cmf if requested:
    if cie_std_dev_obs is not None:
        cmf_cie_std_dev_obs = _CMF['cie_std_dev_obs_' + cie_std_dev_obs.lower()]
        cmf_cie_std_dev_obs = fintp(cmf_cie_std_dev_obs, wl, 'cmf')[1:]
        cmf = cmf + cmf_cie_std_dev_obs

    # Compute the xyz values
    xyz, xyzw = spd_to_xyz_barebones(s, cmf, K = K, relative = relative, rfl = rfl, wl = wl, matmul = matmul)

    # Setup output:
    if out == 2:
        if rflwasnotnone == False: xyz = np.squeeze(xyz,axis = 0)
        return math.round((xyz,xyzw), rounding)
    elif out == 1:
        if rflwasnotnone == False: xyz = np.squeeze(xyz,axis = 0)
        return math.round(xyz, rounding)
    else: 
        #xyz = xyz[rflwasnotnone:,...]
        if rflwasnotnone == False: xyz = np.squeeze(xyz,axis = 0)
        return math.round(xyz, rounding)
    
#------------------------------------------------------------------------------
def spd_to_ler(data, cieobs = _CIEOBS, K = None,
               interp_settings = None, kind = None, extrap_kind = None, extrap_values = None):
    """
    Calculates Luminous efficacy of radiation (LER) from spectral data.
       
    Args: 
        :data: 
            | ndarray with spectral data
            | (.shape = (number of spectra + 1, number of wavelengths))
            | Note that :data: is never interpolated, only CMFs and RFLs. 
            | This way interpolation errors due to peaky spectra are avoided. 
            | Conform CIE15-2018.
        :cieobs: 
            | luxpy._CIEOBS, optional
            | Determines the color matching function set used in the 
            | calculation of LER. For cieobs = '1931_2' the ybar CMF curve equals
            | the CIE 1924 Vlambda curve.
        :K: 
            | None, optional
            |   e.g.  K  = 683 lm/W for '1931_2'
      
    Returns:
        :ler: 
            | ndarray of LER values. 
             
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    
    if isinstance(cieobs,str):    
        if K == None: K = _CMF[cieobs]['K']
        Vl = vlbar(cieobs = cieobs, src = 'dict',wl_new = data[0], interp_settings = interp_settings, 
                   kind = kind, extrap_kind = extrap_kind, extrap_values = extrap_values)[1:2] #also interpolate to wl of data
    else:
        Vl = spd(wl = data[0], data = cieobs, datatype = 'cmf', interp_settings = interp_settings, 
                 kind = kind, extrap_kind = extrap_kind, extrap_values = extrap_values)[1:2]
        if K is None: raise Exception("spd_to_ler: User defined Vlambda, but no K scaling factor has been supplied.")
    dl = getwld(data[0])
    return ((K * np.dot((Vl*dl),data[1:].T))/np.sum(data[1:]*dl, axis = data.ndim-1)).T

#------------------------------------------------------------------------------
def spd_to_power(data, ptype = 'ru', cieobs = _CIEOBS, K = None,
                 interp_settings = None, kind = None, extrap_kind = None, extrap_values = None):
    """
    Calculate power of spectral data in radiometric, photometric 
    or quantal energy units.
    
    Args:
        :data: 
            | ndarray with spectral data
        :ptype: 
            | 'ru' or str, optional
            | str: - 'ru': in radiometric units 
            |      - 'pu': in photometric units 
            |      - 'pusa': in photometric units with Km corrected 
            |                to standard air (cfr. CIE TN003-2015)
            |      - 'qu': in quantal energy units
        :cieobs: 
            | _CIEOBS or str or ndarray, optional
            | Type of cmf set to use for photometric units.
        :K:
            | None, optional
            | Luminous efficacy of radiation, must be supplied if cieobs is an array.
    
    Returns:
        returns: 
            | ndarray with normalized spectral data (SI units)
    """
    # get wavelength spacing:
    dl = getwld(data[0])
    
    if ptype == 'ru': #normalize to radiometric units
        p = np2d(np.dot(data[1:],dl*np.ones(data.shape[1]))).T

    elif ptype == 'pusa': # normalize in photometric units with correction of Km to standard air
    
        # Calculate correction factor for Km in standard air:
        na = _BB['na'] # n for standard air
        c = _BB['c'] # m/s light speed
        lambdad = c/(na*54*1e13)/(1e-9) # 555 nm lambda in standard air
        Km_correction_factor = 1/(1 - (1 - 0.9998567)*(lambdad - 555)) # correction factor for Km in standard air

        # Get Vlambda and Km (for E):
        if isinstance(cieobs, str): 
            src = 'dict'
        else:
            src = 'vltype' # if str -> cieobs is an array
            if K is None: raise Exception('If cieobs is an array, Km must be explicitely supplied')
        
        Vl, Km = vlbar(cieobs = cieobs, K = K, src = src, wl_new = data[0], out = 2, 
                       interp_settings = interp_settings, kind = kind, extrap_kind = extrap_kind, extrap_values = extrap_values)
        if K is None: K = Km
        K *= Km_correction_factor
        p = K*np2d(np.dot(data[1:],dl*Vl[1])).T
        
    elif ptype == 'pu': # normalize in photometric units
    
        # Get Vlambda and Km (for E):
        if isinstance(cieobs, str): 
            src = 'dict'
        else:
            src = 'vltype' # if str -> cieobs is an array
            if K is None: raise Exception('If cieobs is an array, Km must be explicitely supplied')
        
        Vl, Km = vlbar(cieobs = cieobs, K = K, src = src, wl_new = data[0], out = 2, 
                       interp_settings = interp_settings, kind = kind, extrap_kind = extrap_kind, extrap_values = extrap_values)
        if K is None: K = Km
        p = K*np2d(np.dot(data[1:],dl*Vl[1])).T

    
    elif ptype == 'qu': # normalize to quantual units

        # Get Quantal conversion factor:
        fQ = ((1e-9)/(_BB['h']*_BB['c']))
        p = np2d(fQ*np.dot(data[1:],dl*data[0])).T

    return p

#------------------------------------------------------------------------------
def detect_peakwl(spd, n = 1,verbosity = 1, **kwargs):
    """
    Detect primary peak wavelengths and fwhm in spectrum spd.
    
    Args:
        :spd:
            | ndarray with spectral data (2xN). 
            | First row should be wavelengths.
        :n:
            | 1, optional
            | The number of peaks to try to detect in spd. 
        :verbosity:
            | Make a plot of the detected peaks, their fwhm, etc.
        :kwargs:
            | Additional input arguments for scipy.signal.find_peaks.
    Returns:
        :prop:
            | list of dictionaries with keys: 
            | - 'peaks_idx' : index of detected peaks
            | - 'peaks' : peak wavelength values (nm)
            | - 'heights' : height of peaks
            | - 'fwhms' : full-width-half-maxima of peaks
            | - 'fwhms_mid' : wavelength at the middle of the fwhm-range of the peaks (if this is different from the values in 'peaks', then their is some non-symmetry in the peaks)
            | - 'fwhms_mid_heights' : height at the middle of the peak
    """
    from scipy import signal, interpolate # lazy import
    
    props = []
    ips_to_spd_fit = np.polyfit(np.arange(spd.shape[1]),spd[0],1)
    for i in range(spd.shape[0]-1):
        peaks_, prop_ = signal.find_peaks(spd[i+1,:], **kwargs)
        prominences = signal.peak_prominences(spd[i+1,:], peaks_)[0]
        peaks = [peaks_[prominences.argmax()]]
        prominences[prominences.argmax()] = 0
        for j in range(n-1):
            peaks.append(peaks_[prominences.argmax()])
            prominences[prominences.argmax()] = 0
        peaks = np.sort(np.array(peaks))
        peak_heights = spd[i+1,peaks]
        _, width_heights, left_ips, right_ips = signal.peak_widths(spd[i+1,:], peaks, rel_height=0.5)
        #left_ips, right_ips = left_ips + spd[0,0], right_ips + spd[0,0]
        left_ips, right_ips = np.polyval(ips_to_spd_fit, left_ips), np.polyval(ips_to_spd_fit, right_ips)
        widths = (right_ips - left_ips)
        
    
        # get middle of fwhm and calculate peak position and height:
        mpeaks = left_ips + widths/2
        hmpeaks = interpolate.InterpolatedUnivariateSpline(spd[0,:],spd[i+1,:])(mpeaks)
    
        prop = {'peaks_idx' : peaks,'peaks' : spd[0,peaks], 'heights' : peak_heights,
                'fwhms' : widths, 'fwhms_mid' : mpeaks, 'fwhms_mid_heights' : hmpeaks}
        props.append(prop)
        if verbosity == 1:
            print('Peak properties:', prop)
            results_half = (widths, width_heights, left_ips, right_ips)
            import matplotlib.pyplot as plt # lazy import
            plt.plot(spd[0,:],spd[i+1,:],'b-',label = 'spectrum')
            plt.plot(spd[0,peaks],spd[i+1,peaks],'ro', label = 'peaks')
            plt.hlines(*results_half[1:], color="C2", label = 'FWHM range of peaks')
            plt.plot(mpeaks,hmpeaks,'gd', label = 'middle of FWHM range')
    if verbosity == 1: plt.show()
    return props

#------------------------------------------------------------------------------
def create_spectral_interpolator(S, wl = None, kind = 1, ext = 0):
    """ 
    Create an interpolator of kind for spectral data S. 
    
    Args:
        :S:
            | Spectral data array
            | Row 0 should contain wavelengths if :wl:  is None.
        :wl:
            | None, optional 
            | Wavelengths
            | If wl is None: row 0 of S should contain wavelengths.
        :kind:
            | 1, optional
            | Order of spline functions used in interpolator (1<=kind<=5)
            | Interpolator = scipy.interpolate.InterpolatedUnivariateSpline
            
    Returns:
        :interpolators:
            | List of interpolator functions for each row in S (minus wl-row if present).
    
    Note:
        1. Nan's, +infs, -infs will be ignored when generating the interpolators. 
    """
    from scipy import interpolate # lazy import
    
    if S.ndim == 1:
        S = S[None,:] # make 2d
    if wl is None:
        wl = S[0]
        S = S[1:]
    interpolators = []
    for i in range(S.shape[0]):
        indices = np.logical_not(np.isnan(S[i]) | np.isneginf(S[i]) |  np.isposinf(S[i]))
        interpolators.append(interpolate.InterpolatedUnivariateSpline(wl[indices],S[i][indices], k = kind, ext = ext))
    return interpolators

def wls_shift(shfts, log_shft = False, wl = None, S = None, interpolators = None, kind = 1, ext = 0):
    """ 
    Wavelength-shift array S over shft wavelengths.
    
    Args:
        :shfts:
            | array with wavelength shifts.
        :log_shft:
            | False, optional
            | If True: shift in log10 wavelength space.
        :wl:
            | None, optional 
            | Wavelengths to return
            | If wl is None: S will be used and row 0 should contain wavelengths.
        :S:
            | None, optional
            | Spectral data array.
            | Row 0 should contain wavelengths if :wl:  is None.
            | If None: interpolators should be precalculated + wl must contain wavelength array !
        :interpolators:
            | None, optional
            | Pre-calculated interpolators for the (non-wl) rows in S.
            | If None: will be generated from :S: (which should contain wavelengths on row 0) 
            | with specified :kind: using scipy.interpolate.InterpolatedUnivariateSpline
            | If not None and S is not None: interpolators take precedence
        :kind:
            | 1, optional
            | Order of spline functions used in interpolator (1<=kind<=5)
            
    Returns:
        :wavelength_shifted:
            | array with wavelength-shifted S (or interpolators) evaluated at wl.
            | (row 0 contains) 
    
    Note:
        1. Nan's, +infs, -infs will be ignored when generating the interpolators. 
    """
    if wl is None:
        if S is None:
            raise Exception('Either wl or S with wavelengths on row 0 must be supplied')
        wl = S[0] 
        if (S.shape[0] == 1) & (interpolators is None): 
            raise Exception("Interpolators are not supplied and S contains no data to generate them")
        N = S.shape[0] - 1
    else:
        if (S is None) & (interpolators is None):
            raise Exception('Either the interpolators for S or S itself must be supplied')
        elif (S is not None): 
            if S.ndim == 1: 
                S = S[None,:] # make 2d
                N = 1
                if (S.shape[1] == wl.shape[0]):
                    raise Exception("Number of wavelengths in S doesn't match that in wl")
                S = np.vstack((wl,S))
                N = S.shape[0] - 1
            else:
                pass # S is assumed to contain wavelengths on row 0
        else: # (S is not None and interpolators is not None) or (S is None and interpolators is not None)
            N = None
        
    if (interpolators is None) | (S is not None): 
        interpolators = create_spectral_interpolator(S, kind = kind, ext = ext)
    else: 
        if not isinstance(interpolators, (list,tuple)):
            if N is None: N = len(shfts)
            interpolators = [interpolators]*N
            
            
    # Peak Wavelength Shift:
    if not log_shft: 
        wl_shifted = (wl - np.atleast_1d(shfts)[:,None])
    else:
        wl_shifted = 10**(np.log10(wl) - np.log10(np.atleast_1d(shfts) + 1e-308)[:,None])

    peak_shft = np.empty(wl_shifted.shape)
    for i in range(peak_shft.shape[0]): 
        peak_shft[i] = interpolators[i](wl_shifted[i])
    
    return np.vstack((wl,peak_shft))
