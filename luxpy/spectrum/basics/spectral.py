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
         |  1. Read spectral data from data file or take input directly as 
            pandas.dataframe or ndarray.
         |  2. Convert spd-like data from ndarray to pandas.dataframe and back.
         |  3. Interpolate spectral data.
         |  4. Normalize spectral data.

 :xyzbar(): Get color matching functions.
        
 :vlbar(): Get Vlambda function.
 
 :vlbar_cie_mesopic(): Get CIE mesopic luminous efficiency function Vmesm according to CIE191:2010

 :get_cie_mesopic_adaptation(): Get the mesopic adaptation state according to CIE191:2010

 :spd_to_xyz(): Calculates xyz tristimulus values from spectral data. 
            
 :spd_to_ler():  Calculates Luminous efficacy of radiation (LER) 
                 from spectral data.

 :spd_to_power(): Calculate power of spectral data in radiometric, photometric
                  or quantal energy units.
         
 :detect_peakwl(): Detect peak wavelengths and fwhm of peaks in spectrum spd. 

References
----------

    1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
        
    2. `CIE 191:2010 Recommended System for Mesopic Photometry Based on Visual Performance.
    (ISBN 978-3-901906-88-6), http://cie.co.at/publications/recommended-system-mesopic-photometry-based-visual-performance>`_
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

#--------------------------------------------------------------------------------------------------
from luxpy import  _CIEOBS, math
from luxpy.utils import np, pd, sp, plt, _PKG_PATH, _SEP, np2d, getdata, _EPS

from .cmf import _CMF
from scipy import signal
__all__ = ['_BB','_WL3','_INTERP_TYPES','_S_INTERP_TYPE', '_R_INTERP_TYPE','_C_INTERP_TYPE',
           'getwlr','getwld','spd_normalize','cie_interp','spd','xyzbar', 'vlbar', 
           'vlbar_cie_mesopic', 'get_cie_mesopic_adaptation',
           'spd_to_xyz', 'spd_to_ler', 'spd_to_power', 'detect_peakwl']


#--------------------------------------------------------------------------------------------------
# set standard SPD wavelength interval interval and spacing
_WL3 = [360.0,830.0,1.0]
    
#--------------------------------------------------------------------------------------------------
# set coefficients for blackbody radiators:
_BB = {'c1' : 3.74183e-16, 'c2' : 1.4388*0.01,'n': 1.000, 'na': 1.00028, 'c' : 299792458, 'h' : 6.626070040e-34, 'k' : 1.38064852e-23} # blackbody c1,c2 & n standard values


#--------------------------------------------------------------------------------------------------
# Define interpolation types (conform CIE15:20xx): 
_INTERP_TYPES = {'linear' : ['xyzbar','cmf','lms','undefined','Dxx'],'cubic': ['S', 'spd','SPD','Le','rfl','RFL','r','R'],'none':None}
_INTERP_TYPES['sprague5'] = _INTERP_TYPES['cubic']
_S_INTERP_TYPE = 'cubic' # -> cie_interp(): changeds this to Sprague5 for equal wavelength spacings !
_R_INTERP_TYPE = 'cubic' # -> cie_interp(): changeds this to Sprague5 for equal wavelength spacings !
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
    wl = wl3 if (len(wl3) != 3) else np.linspace(wl3[0],wl3[1],int(np.floor((wl3[1]-wl3[0]+wl3[2])/wl3[2]))) # define wavelengths from [start = l0, stop = ln, spacing = dl]
    
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
    dl = (np.hstack((d[0],d[0:-1]/2.0,d[-1]))+np.hstack((0.0,d[1:]/2.0,0.0)))
    if np.array_equal(dl,dl.mean()*np.ones(dl.shape)): dl = dl[0]
    return dl


#------------------------------------------------------------------------------
def spd_normalize(data, norm_type = None, norm_f = 1, wl = True, cieobs = _CIEOBS):
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
            | _CIEOBS or str, optional
            | Type of cmf set to use for normalization using photometric units 
            | (norm_type == 'pu')
    
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
                rpq_power = spd_to_power(data[[0,i+offset],:], cieobs = cieobs, ptype = norm_type_)
                data[i+offset] = (norm_f/rpq_power)*data[i+offset]
            else:
                data[i+offset]=data[i+offset]/norm_f_
    return data


#--------------------------------------------------------------------------------------------------
def cie_interp(data, wl_new, kind = None, sprague5_allowed = False, negative_values_allowed = False,
               extrap_values = 'ext', extrap_kind = 'linear', extrap_log = False):
    """
    Interpolate / extrapolate spectral data following standard CIE15-2018.
    
    | The kind of interpolation depends on the spectrum type defined in :kind:. 
    
    Args:
        :data: 
            | ndarray with spectral data 
            | (.shape = (number of spectra + 1, number of original wavelengths))
        :wl_new: 
            | ndarray with new wavelengths
        :kind: 
            | None, optional
            |   - If :kind: is None, return original data.
            |   - If :kind: is a spectrum type (see _INTERP_TYPES), the correct 
            |     interpolation type is automatically chosen 
            |       (The use of the slow(er) 'sprague5' can be toggled on using :sprague5_allowed:).
            |   - Or :kind: can be any interpolation type supported by 
            |     scipy.interpolate.interp1d (or luxpy.math.interp1 if nan's are present!!)
            |     or can be 'sprague5' (uses luxpy.math.interp1_sprague5).  
        :sprague5_allowed:
            | False, optional
            | If True: When kind is a spectral data type from _INTERP_TYPES['cubic'],
            |    then a cubic spline interpolation will be used in case of 
            |    unequal wavelength spacings, otherwise a 5th order Sprague will be used.
            | If False: always use 'cubic', don't use 'sprague5'. 
            |           This is the default, as differences are minimal and 
            |           use of the 'sprague5' function is a lot slower!
        :negative_values_allowed: 
            | False, optional
            | If False: negative values are clipped to zero.
        :extrap_values:
            | 'ext', optional
            | If 'ext': extrapolate using 'linear' ('cie167:2005'), 'quadratic' ('cie15:2018') 
            |           'nearest' ('cie15:2004') recommended or other (e.g. 'cubic') methods.
            | If None: use CIE15:2004 recommended 'nearest value' approach when extrapolating.
            | If float or list or ndarray, use those values to fill extrapolated value(s).
        :extrap_kind:
            | 'linear', optional
            | Extrapolation method used when :extrap_values: is set to 'ext'. 
            | Options: 'linear' ('cie167:2005'), 'quadratic' ('cie15:2018'), 
            |           'nearest' ('cie15:2004'), 'cubic'
            | CIE15:2018 states that based on a 2017 paper by Wang that 'quadratic' is 'better'. 
            | However, no significant difference was found between 'quadratic' and 'linear' methods.
            | Also see note 1 below, for why the CIE67:2005 recommended 'linear' extrapolation
            | is set as the default.
        :extrap_log:
            | False, optional
            | If True: extrap the log of the spectral values 
            |     (not CIE recommended but in most cases seems to give a 
            |     more realistic estimate, but can sometimes seriously fail, 
            |     especially for the 'quadratic' extrapolation case (see note 1)!!!)
    
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
    if (kind is not None):
        # Wavelength definition:
        wl_new = getwlr(wl_new)
        
        if (not np.array_equal(data[0],wl_new)) | np.isnan(data).any():
       
            extrap_values = np.atleast_1d(extrap_values)
            
            # define wl, S, wl_new:
            wl = np.array(data[0])
            S = data[1:]
            wl_new = np.array(wl_new)
            
            # Set interpolation type based on data type:
            if kind in _INTERP_TYPES['linear']:
                kind = 'linear'
            elif kind in _INTERP_TYPES['cubic']:
                kind = 'cubic'
                if sprague5_allowed: 
                    dwl = np.diff(wl)
                    if np.all(dwl == dwl[0]):
                        kind = 'sprague5' # force recommended 5th order Sprague interpolation for equal wavelength spacings when kind was a spectral data type!

            # Set extrapolation type based on CIE report:
            if extrap_kind == 'cie167:2005':
                extrapolation_kind = 'linear'
            elif extrap_kind == 'cie15:2018':
                extrapolation_kind = 'quadratic'
            elif extrap_kind == 'cie15:2004':
                extrapolation_kind = 'nearest'
            else:
                extrapolation_kind = extrap_kind

            # Interpolate each spectrum in S: 
            N = S.shape[0]
            nan_indices = np.isnan(S)
            
            # Interpolate all spectra:
            all_rows = np.arange(S.shape[0])
            rows_with_nans = np.where(nan_indices.sum(axis=1))[0]
            rows_with_no_nans = np.setdiff1d(all_rows,rows_with_nans)
            # rows_with_no_nans = np.where(~np.isnan(S.sum(axis=1)))

            Si = np.zeros([N,wl_new.shape[0]])
            Si.fill(np.nan)
            if (rows_with_no_nans.size>0): 

                S_no_nans = S[rows_with_no_nans]

                # prepare + do 'ext' extrapolation:
                if (extrap_values[0] is None) | (((type(extrap_values[0])==np.str_)|(type(extrap_values[0])==str)) and (extrap_values[0][:3]=='ext')): 
                    fill_value = (0,0)
                    if extrap_log:
                        Si_ext_no_nans = np.exp(np.atleast_2d(sp.interpolate.interp1d(wl, np.log(S_no_nans + _EPS), kind = extrapolation_kind, bounds_error = False, fill_value = 'extrapolate')(wl_new)))
                    else:
                        Si_ext_no_nans = np.atleast_2d(sp.interpolate.interp1d(wl, S_no_nans, kind = extrapolation_kind, bounds_error = False, fill_value = 'extrapolate')(wl_new))
                else:
                    fill_value, Si_ext_no_nans = (extrap_values[0],extrap_values[-1]), None

                # interpolate:
                if kind != 'sprague5':
                    Si_no_nans = sp.interpolate.interp1d(wl, S_no_nans, kind = kind, bounds_error = False, fill_value = fill_value)(wl_new)
                else:
                    Si_no_nans = math.interp1_sprague5(wl, S_no_nans, wl_new, extrap = fill_value)

                # Add extrapolated part to the interpolate part (which had extrapolated fill values set to zero)
                if Si_ext_no_nans is not None: 
                    Si_ext_no_nans[:,(wl_new >= wl[0]) & (wl_new <= wl[-1])] = 0
                    Si_no_nans += Si_ext_no_nans
                
                Si[rows_with_no_nans] = Si_no_nans
                
            # In case there are NaN's:
            if rows_with_nans.size > 0:

                # looping required as some values are NaN's:
                for i in rows_with_nans:

                    nonan_indices = np.logical_not(nan_indices[i])
                    wl_nonan = wl[nonan_indices]
                    S_i_nonan = S[i][nonan_indices]
                    
                    if (kind != 'sprague5'): 
                        Si_nonan = math.interp1(wl_nonan, S_i_nonan, wl_new, kind = kind, ext = 'extrapolate')
#                       Si_nonan = sp.interpolate.interp1d(wl_nonan, S_i_nonan, kind = kind, bounds_error = False, fill_value = 'extrapolate')(wl_new)
                    else:
                        # check wavelength spacing constancy:
                        dwl_nonan = np.diff(wl_nonan)
                        if np.all(dwl_nonan == dwl_nonan[0]):
                            Si_nonan = math.interp1_sprague5(wl_nonan, S_i_nonan, wl_new, extrap = (0,0))
                        else:
                            # fall back to 'cubic interpolation!:
                            Si_nonan = math.interp1(wl_nonan, S_i_nonan, wl_new, kind = 'cubic', ext = 'extrapolate')
  
                    # Do extrapolation:
                    if (extrap_values[0] is None) | (((type(extrap_values[0])==np.str_)|(type(extrap_values[0])==str)) and (extrap_values[0][:3]=='ext')): 
                        if extrapolation_kind != 'nearest':
                            if extrap_log:
                                Si_nonan_ext = np.exp(math.interp1(wl_nonan,np.log(S_i_nonan + _EPS), wl_new, kind = extrapolation_kind, ext = 'extrapolate'))
                            else:
                                Si_nonan_ext = math.interp1(wl_nonan,S_i_nonan, wl_new, kind = extrapolation_kind, ext = 'extrapolate')
                        else: # do nearest neighbour extrapolation
                            Si_nonan_ext = np.zeros((wl_new.size))
                            Si_nonan_ext[wl_new<wl_nonan[0]] = S_i_nonan[0]
                            Si_nonan_ext[wl_new>wl_nonan[-1]] = S_i_nonan[-1]
                        Si_nonan_ext[(wl_new >= wl_nonan[0]) & (wl_new <= wl_nonan[-1])] = 0
                        Si_nonan[(wl_new<wl_nonan[0]) | (wl_new>wl_nonan[-1])] = 0
                        Si_nonan = Si_nonan + Si_nonan_ext

                    else:
                        Si_nonan[wl_new<wl_nonan[0]] = extrap_values[0]
                        Si_nonan[wl_new>wl_nonan[-1]] = extrap_values[-1]  
                    
                    # add to array:
                    Si[i] = Si_nonan              
                
            # No negative values allowed for spectra:    
            if negative_values_allowed == False:
                if np.any(Si): Si[Si<0.0] = 0.0
            
            # Add wavelengths to data array: 
            return np.vstack((wl_new,Si))  
    
    return data.copy()

#--------------------------------------------------------------------------------------------------
def spd(data = None, interpolation = None, kind = 'np', wl = None,\
        columns = None, sep = ',',header = None, datatype = 'S', \
        norm_type = None, norm_f = None):
    """
    | All-in-one function that can:
    |    1. Read spectral data from data file or take input directly 
         as pandas.dataframe or ndarray.
    |    2. Convert spd-like data from ndarray to pandas.dataframe and back.
    |    3. Interpolate spectral data.
    |    4. Normalize spectral data.
            
    Args:
        :data: 
            | - str with path to file containing spectral data
            | - ndarray with spectral data
            | - pandas.dataframe with spectral data
            | (.shape = (number of spectra + 1, number of original wavelengths))
        :interpolation:
            | None, optional
            | - None: don't interpolate
            | - str with interpolation type or spectrum type
        :kind: 
            | str ['np','df'], optional 
            | Determines type(:returns:), np: ndarray, df: pandas.dataframe
        :wl: 
            | None, optional
            | New wavelength range for interpolation. 
            | Defaults to wavelengths specified by luxpy._WL3.
        :columns: 
            | -  None or list[str] of column names for dataframe, optional
        :header: 
            | None or 'infer', optional
            | - None: no header in file
            | - 'infer': infer headers from file
        :sep: 
            | ',' or '\t' or other char, optional
            | Column separator in case :data: specifies a data file. 
        :datatype': 
            | 'S' (light source) or 'R' (reflectance) or other, optional
            | Specifies a type of spectral data. 
            | Is used when creating column headers when :column: is None.
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
              for 'max' and 'area' 
              or which wavelength is normalized to 1 for 'lambda' option.
    
    Returns:
        :returns: 
            | ndarray or pandas.dataframe 
            | with interpolated and/or normalized spectral data.
    """
    transpose = True if isinstance(data,str) else False #when spd comes from file -> transpose (columns in files should be different spectra)
         
    # Wavelength definition:
    wl = getwlr(wl)
    
    # Data input:
    if data is not None:
        if (interpolation is None) & (norm_type is None):
            data = getdata(data = data, kind = 'np', columns = columns, sep = sep, header = header, datatype = datatype, copy = True)
            if (transpose == True): data = data.T
        else:
            data = getdata(data = data, kind = 'np', columns = columns, sep = sep, header = header, datatype = datatype, copy = True)#interpolation requires np-array as input
            if (transpose == True): data = data.T
            data = cie_interp(data = data, wl_new = wl,kind = interpolation)
            data = spd_normalize(data,norm_type = norm_type, norm_f = norm_f, wl = True)
        
        if isinstance(data,pd.DataFrame): columns = data.columns #get possibly updated column names

    else:
        data = np2d(wl)
  
     
    if ((data.shape[0] - 1) == 0): columns = None #only wavelengths
       
    if kind == 'df':  data = data.T
        
    # convert to desired kind:
    data = getdata(data = data,kind = kind, columns = columns, datatype = datatype, copy = False) # already copy when data is not None, else new anyway
        
    return data


#--------------------------------------------------------------------------------------------------
def xyzbar(cieobs = _CIEOBS, scr = 'dict', wl_new = None, kind = 'np'):
    """
    Get color matching functions.  
    
    Args:
        :cieobs: 
            | luxpy._CIEOBS, optional
            | Sets the type of color matching functions to load.
        :scr: 
            | 'dict' or 'file', optional
            | Determines whether to load cmfs from file (./data/cmfs/) 
            | or from dict defined in .cmf.py
        :wl: 
            | None, optional
            | New wavelength range for interpolation. 
            | Defaults to wavelengths specified by luxpy._WL3.
        :kind: 
            | str ['np','df'], optional 
            | Determines type(:returns:), np: ndarray, df: pandas.dataframe

    Returns:
        :returns: 
            | ndarray or pandas.dataframe with CMFs 
        
            
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    if scr == 'file':
        dict_or_file = _PKG_PATH + _SEP + 'data' + _SEP + 'cmfs' + _SEP + 'ciexyz_' + cieobs + '.dat'
    elif scr == 'dict':
        dict_or_file = _CMF[cieobs]['bar']
    elif scr == 'cieobs':
        dict_or_file = cieobs #can be file or data itself
    return spd(data = dict_or_file, wl = wl_new, interpolation = 'linear', kind = kind, columns = ['wl','xb','yb','zb'])

#--------------------------------------------------------------------------------------------------
def vlbar(cieobs = _CIEOBS, scr = 'dict', wl_new = None, kind = 'np', out = 1):
    """
    Get Vlambda functions.  
    
    Args:
        :cieobs: 
            | str, optional
            | Sets the type of Vlambda function to obtain.
        :scr: 
            | 'dict' or array, optional
            | - 'dict': get from ybar from _CMF
            | - 'array': ndarray in :cieobs:
            | Determines whether to load cmfs from file (./data/cmfs/) 
            | or from dict defined in .cmf.py
            | Vlambda is obtained by collecting Ybar.
        :wl: 
            | None, optional
            | New wavelength range for interpolation. 
            | Defaults to wavelengths specified by luxpy._WL3.
        :kind: 
            | str ['np','df'], optional 
            | Determines type(:returns:), np: ndarray, df: pandas.dataframe
        :out: 
            | 1 or 2, optional
            |     1: returns Vlambda
            |     2: returns (Vlambda, Km)
    
    Returns:
        :returns: 
            | dataframe or ndarray with Vlambda of type :cieobs: 
        
            
    References:
        1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
    """
    if scr == 'dict':
        dict_or_file = _CMF[cieobs]['bar'][[0,2],:] 
        K = _CMF[cieobs]['K']
    elif scr == 'vltype':
        dict_or_file = cieobs #can be file or data itself
        K = 1
    Vl = spd(data = dict_or_file, wl = wl_new, interpolation = 'linear', kind = kind, columns = ['wl','Vl'])

    if out == 2:
        return Vl, K
    else:
        return Vl

#--------------------------------------------------------------------------------------------------
def vlbar_cie_mesopic(m = [1], wl_new = None, kind = 'np', out = 1,
                      Lp = None, Ls = None, SP = None):
    """
    Get CIE mesopic luminous efficiency function Vmesm according to CIE191:2010
    
    Args:
        :m:
            | float or list or ndarray with mesopic adaptation coefficients
        :wl: 
            | None, optional
            | New wavelength range for interpolation. 
            | Defaults to wavelengths specified by luxpy._WL3.
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
    Vl = vlbar(cieobs='1931_2')
    Vlp = vlbar(cieobs='1951_20_scotopic')
    Vlmes= m*(Vl[1:,:]) + (1-m)*(Vlp[1:,:])
    Vlmes = np.vstack((Vl[:1,:],Vlmes))
    Kmes = 683/Vlmes[1:,Vlmes[0,:] == 555]
    Vlmes[1:,:] = Vlmes[1:,:]/Vlmes[1:,:].max(axis=1,keepdims=True) # normalize to max = 1
    
    if kind == 'df':
        columns = ['wl']
        for i in range(m.size):
            columns.append('Vmes{:0.2f}'.format(m[i,0]))
    else:
        columns = ['wl',['Vmes']*m.size]
    Vlmes = spd(data = Vlmes, wl = wl_new, interpolation = 'linear', 
                norm_type = 'max', norm_f = 1, kind = kind, columns = columns)
    
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
def spd_to_xyz(data,  relative = True, rfl = None, cieobs = _CIEOBS, K = None, out = None, cie_std_dev_obs = None):
    """
    Calculates xyz tristimulus values from spectral data.
       
    Args: 
        :data: 
            | ndarray or pandas.dataframe with spectral data
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
    
    data = getdata(data,kind = 'np') if isinstance(data,pd.DataFrame) else np2d(data) # convert to np format and ensure 2D-array

    # get wl spacing:
    dl = getwld(data[0])
    
    # get cmf,k for cieobs:
    if isinstance(cieobs,str):
        if K is None: K = _CMF[cieobs]['K']
        scr = 'dict'
    else:
        scr = 'cieobs'
        if (K is None) & (relative == False): K = 1
    
    # Interpolate to wl of data:
    cmf = xyzbar(cieobs = cieobs, scr = scr, wl_new = data[0], kind = 'np') 
    
    # Add CIE standard deviate observer function to cmf if requested:
    if cie_std_dev_obs is not None:
        cmf_cie_std_dev_obs = xyzbar(cieobs = 'cie_std_dev_obs_' + cie_std_dev_obs.lower(), scr = scr, wl_new = data[0], kind = 'np')
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
    
#------------------------------------------------------------------------------
def spd_to_ler(data, cieobs = _CIEOBS, K = None):
    """
    Calculates Luminous efficacy of radiation (LER) from spectral data.
       
    Args: 
        :data: 
            | ndarray or pandas.dataframe with spectral data
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
        Vl = vlbar(cieobs = cieobs, scr = 'dict',wl_new = data[0], kind = 'np')[1:2] #also interpolate to wl of data
    else:
        Vl = spd(wl = data[0], data = cieobs, interpolation = 'cmf', kind = 'np')[1:2]
        if K is None: raise Exception("spd_to_ler: User defined Vlambda, but no K scaling factor has been supplied.")
    dl = getwld(data[0])
    return ((K * np.dot((Vl*dl),data[1:].T))/np.sum(data[1:]*dl, axis = data.ndim-1)).T

#------------------------------------------------------------------------------
def spd_to_power(data, ptype = 'ru', cieobs = _CIEOBS):
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
            | _CIEOBS or str, optional
            | Type of cmf set to use for photometric units.
    
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
        Vl, Km = vlbar(cieobs = cieobs, wl_new = data[0], out = 2)
        Km *= Km_correction_factor
        p = Km*np2d(np.dot(data[1:],dl*Vl[1])).T
        
    elif ptype == 'pu': # normalize in photometric units
    
        # Get Vlambda and Km (for E):
        Vl, Km = vlbar(cieobs = cieobs, wl_new = data[0], out = 2)
        p = Km*np2d(np.dot(data[1:],dl*Vl[1])).T

    
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
    props = []
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
        widths, width_heights, left_ips, right_ips = signal.peak_widths(spd[i+1,:], peaks, rel_height=0.5)
        left_ips, right_ips = left_ips + spd[0,0], right_ips + spd[0,0]
    
        # get middle of fwhm and calculate peak position and height:
        mpeaks = left_ips + widths/2
        hmpeaks = sp.interpolate.interp1d(spd[0,:],spd[i+1,:])(mpeaks)
    
        prop = {'peaks_idx' : peaks,'peaks' : spd[0,peaks], 'heights' : peak_heights,
                'fwhms' : widths, 'fwhms_mid' : mpeaks, 'fwhms_mid_heights' : hmpeaks}
        props.append(prop)
        if verbosity == 1:
            print('Peak properties:', prop)
            results_half = (widths, width_heights, left_ips, right_ips)
            plt.plot(spd[0,:],spd[i+1,:],'b-',label = 'spectrum')
            plt.plot(spd[0,peaks],spd[i+1,peaks],'ro', label = 'peaks')
            plt.hlines(*results_half[1:], color="C2", label = 'FWHM range of peaks')
            plt.plot(mpeaks,hmpeaks,'gd', label = 'middle of FWHM range')
    return props

