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
Module for class functionality for spectral data: class SPD
===========================================================

SPD fields 
----------

 :self.wl: wavelengths of spectral data
    
 :self.value: values of spectral data
    
 :self.dtype: spectral data type ('S' light source, or 'R', reflectance, ...),
              used to determine interpolation method following CIE15-2017.
    
 :self.shape: self.value.shape

 :self.N = self.shape[0] (number of spectra in instance)
    

SPD methods 
-----------

 :self.read_csv_(): Reads spectral data from file.

 :self.plot(): Make a plot of the spectral data in SPD instance.

 :self.mean(): Take mean of all spectra in SPD instance

 :self.sum(): Sum all spectra in SPD instance.

 :self.dot(): Take dot product with instance of SPD.

 :self.add(): Add instance of SPD.

 :self.sub(): Subtract instance of SPD.

 :self.mul(): Multiply instance of SPD.

 :self.div(): Divide by instance of SPD.

 :self.pow(): Raise SPD instance to power n.

 :self.get_(): Get spd as ndarray in instance of SPD.

 :self.setwlv(): Store spd ndarray in fields wl and values of instance of SPD.

 :self.getwld_(): Get wavelength spacing of SPD instance. 

 :self.normalize(): Normalize spectral power distributions in SPD instance.

 :self.cie_interp(): Interpolate / extrapolate spectral data following CIE15-2018.

 :self.to_xyz(): Calculates xyz tristimulus values from spectral data 
                 and return as instance of class XYZ.

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import numpy as np

from luxpy import _CIEOBS, spd_to_xyz, cie_interp, getwld, spd_normalize, plot_spectrum_colors
from luxpy.utils import getdata
from luxpy.color.CDATA import XYZ 

class SPD:
    
    def __init__(self, spd = None, wl = None, ax0iswl = True, dtype = 'S', \
                 wl_new = None, interp_method = 'auto', negative_values_allowed = False, extrap_values = 'ext',\
                 norm_type = None, norm_f = 1,\
                 header = None, sep = ','):
        """
        Initialize instance of SPD.
        
        Args:
            :spd: 
                | None or ndarray or str, optional
                | If None: self.value is initialized with zeros.
                | If str: spd contains filename.
                | If ndarray: ((wavelength, spectra)) or (spectra). 
                |     If latter, :wl: should contain the wavelengths.
            :wl: 
                | None or ndarray, optional
                | Wavelengths.
                | Either specified as a 3-vector ([start, stop, spacing]) 
                | or as full wavelength array.
            :a0iswl:
                | True, optional
                | Signals that first axis of :spd: contains wavelengths.
            :dtype:
                | 'S', optional
                | Type of spectral object (e.g. 'S' for source spectrum, 'R' for
                  reflectance spectra, etc.)
                | See SPD._INTERP_TYPES for more options. 
                | This is used to automatically determine the correct kind of
                  interpolation method according to CIE15-2018.
            :wl_new: 
                | None or ndarray with wavelength range, optional
                | If None: don't interpolate, else perform interpolation.
            :interp_method:
                | - 'auto', optional
                | If 'auto': method is determined based on :dtype:
            :negative-values_allowed:
                | False, optional (for cie_interp())
                | Spectral data can not be negative. Values < 0 are therefore 
                  clipped when set to False.
            :extrap_values:
                | 'ext', optional
                | float or list or ndarray with values to extrapolate
                | If 'ext' or 'cie15:2018': use CIE15:2018 recommended quadratic extrapolation.
                | If None or 'cie15:2004': use CIE15:2004 recommended 'closest value' approach.
            :norm_type:
                | None or str, optional
                | - 'lambda': make lambda in norm_f equal to 1
                | - 'area': area-normalization times norm_f
                | - 'max': max-normalization times norm_f
            :norm_f:
                | 1, optional
                | Normalization factor determines size of normalization 
                | for 'max' and 'area' 
                | or which wavelength is normalized to 1 for 'lambda' option.
        """
        if spd is not None:
            if isinstance(spd, str):
                spd = SPD.read_csv_(self, file = spd,header = header, sep = sep)
            if ax0iswl == True:
                self.wl = spd[0]
                self.value = spd[1:]
            else:
                self.wl = wl
                if (self.wl.size == 3):
                    self.wl = np.arange(self.wl[0],self.wl[1]+1,self.wl[2])
                self.value = spd
            if self.value.shape[1] != self.wl.shape[0]:
                raise Exception('SPD.__init__(): Dimensions of wl and spd do not match.' )
        else:
            if (wl is None):
                self.wl = SPD._WL3
            else:
                self.wl = wl
            if (self.wl.size == 3):
                self.wl = np.arange(self.wl[0],self.wl[1]+1,self.wl[2])
            self.value = np.zeros((1,self.wl.size))
        
        self.wl = self.wl
        self.dtype = dtype
        self.shape = self.value.shape
        self.N = self.shape[0]

        
        if wl_new is not None:
            if interp_method == 'auto':
                interp_method = dtype
            self.cie_interp(wl_new, kind = interp_method, negative_values_allowed = negative_values_allowed, extrap_values = extrap_values)
        if norm_type is not None:
            self.normalize(norm_type = norm_type, norm_f = norm_f)
    
    def read_csv_(self, file, header = None, sep = ','):
        """
        Reads spectral data from file.
        
        Args:
            :file:
                | filename 
            :header:
                | None or 'infer', optional
                | If 'infer': headers will be inferred from file itself.
                | If None: no headers are expected from file.
            :sep: 
                | ',', optional
                | Column separator.
        
        Returns:
            :returns:
                | ndarray with spectral data (first row are wavelengths)
            
        Note:
            Spectral data in file should be organized in columns with the first
            column containing  the wavelengths.
        """
        #return pd.read_csv(file, names = None, index_col = None, header = header, sep = sep).values.T
        return getdata(file, header = header, sep = sep).T

    def plot(self, ylabel = 'Spectrum', wavelength_bar = True, *args,**kwargs):
        """
        Make a plot of the spectral data in SPD instance.
        
        Returns:
            :returns:
                | handle to current axes.
        """
        import matplotlib.pyplot as plt # lazy import
        plt.plot(self.wl, self.value.T, *args,**kwargs)
        if wavelength_bar == True:
            Smax = np.nanmax(self.value)
            axh = plot_spectrum_colors(spd = None,spdmax=Smax, axh = plt.gca(), wavelength_height = -0.05)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel(ylabel)
        return plt.gca()
    
    def mean(self):
        """
        Take mean of all spectra in SPD instance.
        """
        self.value = np.nanmean(self.value, axis = 0, keepdims = True)
        self.shape = self.value.shape
        self.N = self.shape[0]
        return self
        
    def sum(self):
        """
        Sum all spectra in SPD instance.
        """
        self.value = np.nansum(self.value, axis = 0, keepdims = True)
        self.shape = self.value.shape
        self.N = self.shape[0]
        return self
        
    def dot(self, S):
        """
        Take dot product with instance of SPD.
        """
        if isinstance(S,SPD):
            self.value = np.dot(self.value,S.value)
        else:
            self.value = np.dot(self.value,S)
        self.shape = self.value.shape
        self.N = self.shape[0]
        return self
            
    def add(self, S):
        """
        Add instance of SPD.
        """
        self.value += S.value 
        return self
    
    def sub(self, S):
        """
        Subtract instance of SPD.
        """
        self.value -= S.value 
        return self
        
    def mul(self, S):
        """
        Multiply by instance of SPD.
        """
        self.value *= S.value 
        return self
        
    def div(self, S):
        """
        Divide by instance of SPD.
        """
        self.value /= S.value 
        return self
    
    def pow(self, n):
        """
        Raise SPD instance to power n.
        """
        self.value **= n
        return self
    
    def get_(self):
        """
        Get spd as ndarray in instance of SPD.
        """
        return np.vstack((self.wl, self.value))
    
    def setwlv(self,spd):
        """
        Store spd ndarray in fields wl and values of instance of SPD.
        """
        self.wl = spd[0]
        self.value = spd[1:]
        return self
     
    #--------------------------------------------------------------------------------------------------
      
    def getwld_(self):
        """
        Get wavelength spacing of SPD instance. 
                   
        Returns:
            :returns:
                | float:  for equal wavelength spacings
                |     ndarray (.shape = (n,)): for unequal wavelength spacings
        """
        return getwld(self.wl)

        
    #------------------------------------------------------------------------------
    def normalize(self, norm_type = None, norm_f = 1, cieobs = _CIEOBS):
        """
        Normalize spectral power distributions in SPD instance.
        
        Args:
            :norm_type:
                | None, optional 
                |   - 'lambda': make lambda in norm_f equal to 1
                |   - 'area': area-normalization times norm_f
                |   - 'max': max-normalization times norm_f
                |   - 'ru': to :norm_f: radiometric units 
                |   - 'pu': to :norm_f: photometric units 
                |   - 'pusa': to :norm_f: photometric units (with Km corrected
                |                         to standard air, cfr. CIE TN003-2015)
                |   - 'qu': to :norm_f: quantal energy units
            :norm_f: 
                | 1, optional
                | Determines size of normalization for 'max' and 'area' 
                  or which wavelength is normalized to 1 for 'lambda' option.
            :cieobs:
                | _CIEOBS or str, optional
                | Type of cmf set to use for normalization using photometric 
                  units (norm_type == 'pu')
        """
        self.value = spd_normalize(self.get_(), norm_type = norm_type, norm_f = norm_f, cieobs = cieobs)[1:]
        return self

    #--------------------------------------------------------------------------------------------------
    def cie_interp(self,wl_new, kind = 'auto', sprague5_allowed = False, negative_values_allowed = False, 
                   extrap_values = 'ext', extrap_kind = 'linear', extrap_log = False):
        """
        Interpolate / extrapolate spectral data following standard CIE15-2018.
        
        | The interpolation type depends on the spectrum type defined in :kind:. 
        
        Args:
            :wl_new: 
                | ndarray with new wavelengths
            :kind:
                | 'auto', optional
                | If :kind: is None, return original data.
                | If :kind: is a spectrum type (see _INTERP_TYPES), the correct 
                |     interpolation type if automatically chosen.
                |       (The use of the slow(er) 'sprague5' can be toggled on using :sprague5_allowed:).
                | If kind = 'auto': use self.dtype
                | Or :kind: can be any interpolation type supported by 
                |     luxpy.math.interp1
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
                | If False: negative values are clipped to zero
            :extrap_values:
                | 'ext', optional
                | If 'ext': extrapolate using 'linear' ('cie167:2005' r), 'quadratic' ('cie15:2018') 
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
                | (.shape = (number of spectra+1, number of wavelength in wl_new))
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
        if (kind == 'auto') & (self.dtype is not None):
            kind = self.dtype
        spd = cie_interp(self.get_(), wl_new, kind = kind, sprague5_allowed = sprague5_allowed,
                         negative_values_allowed = negative_values_allowed, 
                         extrap_values = extrap_values,
                         extrap_kind = extrap_kind,
                         extrap_log = extrap_log)
        self.wl = spd[0]
        self.value = spd[1:]
        self.shape = self.value.shape
        return self
    
    #--------------------------------------------------------------------------------------------------

    def to_xyz(self,  relative = True, rfl = None, cieobs = _CIEOBS, out = None):
        """
        Calculates xyz tristimulus values from spectral data 
        and return as instance of XYZ.
           
        Args: 
            :relative:
                | True or False, optional
                | Calculate relative XYZ (Yw = 100) 
                  or absolute XYZ (Y = Luminance)
            :rfl: 
                | ndarray with spectral reflectance functions.
                | Will be interpolated if wavelengths don't match those of :data:
            :cieobs:
                | luxpy._CIEOBS, optional
                | Determines the color matching functions to be used in the 
                  calculation of XYZ.
            :out: 
                | None or 1 or 2, optional
                | Determines number and shape of output. (see :returns:)
        
        Returns:
            :returns:
                | luxpy.XYZ instance with ndarray .value field:
                | If rfl is None:
                |     If out is None: ndarray of xyz values 
                |                     (.shape = (data.shape[0],3))
                |     If out == 1: ndarray of xyz values 
                |                     (.shape = (data.shape[0],3))
                |     If out == 2: (ndarray of xyz , ndarray of xyzw) values
                |         Note that xyz == xyzw, with (.shape=(data.shape[0],3))
                | If rfl is not None:
                |     If out is None: ndarray of xyz values 
                |                     (.shape = (rfl.shape[0],data.shape[0],3))
                |     If out == 1: ndarray of xyz values 
                |                 (.shape = (rfl.shape[0]+1,data.shape[0],3))
                |         The xyzw values of the light source spd are the first 
                |         set of values of the first dimension. 
                |         The following values along this dimension are the 
                |         sample (rfl) xyz values.
                |     If out == 2: (ndarray of xyz, ndarray of xyzw) values
                |         with xyz.shape = (rfl.shape[0],data.shape[0],3)
                |         and with xyzw.shape = (data.shape[0],3)
                 
        References:
            1. `CIE15:2018, “Colorimetry,” CIE, Vienna, Austria, 2018. <https://doi.org/10.25039/TR.015.2018>`_
        """
        if (out == 2):
            xyz, xyzw = spd_to_xyz(self.get_(),  relative = relative, rfl = rfl, cieobs = cieobs, out = out)
            xyz = XYZ(xyz, relative = relative, cieobs = cieobs)
            xyzw = XYZ(xyzw, relative = relative, cieobs = cieobs)
            return xyz, xyzw
        else:
            xyz = spd_to_xyz(self.get_(),  relative = relative, rfl = rfl, cieobs = cieobs, out = out)
            xyz = XYZ(xyz, relative = relative, cieobs = cieobs)
            return xyz

