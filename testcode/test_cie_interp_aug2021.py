# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 16:02:03 2021

@author: u0032318
"""
from luxpy import  _CIEOBS, _CMF,math, getwlr, _INTERP_TYPES
from luxpy.utils import np, pd, sp, plt, _PKG_PATH, _SEP, np2d, getdata, _EPS


from scipy import signal

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
            rows_with_nans = np.where(nan_indices.sum(axis=1))[0]
            if (rows_with_nans.size > 0): # at least 1 row has at least one NaN, so don't interpolate using scipy's interp1d as NaN's in one row also affect interpolation of other rows!
                Si = np.zeros([N,wl_new.shape[0]])
                Si.fill(np.nan)
            else:
                # prepare + do 'ext' extrapolation:
                if (extrap_values[0] is None) | (((type(extrap_values[0])==np.str_)|(type(extrap_values[0])==str)) and (extrap_values[0][:3]=='ext')): 
                    fill_value = (0,0)
                    if extrap_log:
                        Si_ext = np.exp(np.atleast_2d(sp.interpolate.interp1d(wl, np.log(S + _EPS), kind = extrapolation_kind, bounds_error = False, fill_value = 'extrapolate')(wl_new)))
                    else:
                        Si_ext = np.atleast_2d(sp.interpolate.interp1d(wl, S, kind = extrapolation_kind, bounds_error = False, fill_value = 'extrapolate')(wl_new))
                else:
                    fill_value, Si_ext = (extrap_values[0],extrap_values[-1]), None

                # interpolate:
                if kind != 'sprague5':
                    Si = sp.interpolate.interp1d(wl, S, kind = kind, bounds_error = False, fill_value = fill_value)(wl_new)
                else:
                    Si = math.interp1_sprague5(wl, S, wl_new, extrap = fill_value)

                # Add extrapolated part to the interpolate part (which had extrapolated fill values set to zero)
                if Si_ext is not None: 
                    Si_ext[:,(wl_new >= wl[0]) & (wl_new <= wl[-1])] = 0
                    Si = Si + Si_ext
                
                   
            # In case there are NaN's:
            if nan_indices.any():

                # looping required as some values are NaN's:
                # for i in rows_with_nans: # this line would ideally work, so interpolation should only be done for those rows with NaNs, but once there is a single NaN in the array scipy's interp1d outputs NaNs for all rows!
                for i in range(S.shape[0]):

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


if __name__ == '__main__':
    import luxpy as lx
    import matplotlib.pyplot as plt
    spd1 = lx._CIE_D65.copy()
    spd2 = lx._CIE_A.copy()
    spds1_ = np.vstack((spd1[:,20:-50:5].copy(),20+((spd2[1,20:-50:5])**1.2).copy()))
    spds1 = np.vstack((spd1,20+(spd2[1:])**1.2))
    spds1_[1:,[5,10,15]] = np.nan
    spds1_i = cie_interp(spds1_,spds1[0], kind = 'spd',extrap_values = 'ext',extrap_kind='cie167:2005')
    
    plt.plot(spds1[0],spds1[1:].T,linestyle='-')
    plt.plot(spds1_i[0],spds1_i[1:].T,linestyle='--')
    
    
