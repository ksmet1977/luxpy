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
              used to determine interpolation method following CIE15-2004.
    
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

 :self.cie_interp(): Interpolate / extrapolate spectral data following CIE15-2004.

 :self.to_xyz(): Calculates xyz tristimulus values from spectral data 
                 and return as instance of class XYZ.

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import _CIEOBS, _WL3, _BB, _S012_DAYLIGHTPHASE, _INTERP_TYPES, _S_INTERP_TYPE, _R_INTERP_TYPE, _CRI_REF_TYPE, _CRI_REF_TYPES
from luxpy import spd_to_xyz, cie_interp, getwlr, getwld, spd_normalize
from luxpy import np, pd, plt, interpolate
from .CDATA import XYZ, LAB

class SPD:
    
    def __init__(self, spd = None, wl = None, ax0iswl = True, dtype = 'S', \
                 wl_new = None, interp_method = 'auto', negative_values_allowed = False, \
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
                  interpolation method according to CIE15-2004.
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
            self.cie_interp(wl_new, kind = interp_method, negative_values_allowed = negative_values_allowed)
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
        return pd.read_csv(file, names = None, index_col = None, header = header, sep = sep).values.T

    def plot(self, ylabel = 'Spectrum', *args,**kwargs):
        """
        Make a plot of the spectral data in SPD instance.
        
        Returns:
            :returns:
                | handle to current axes.
        """
        plt.plot(self.wl, self.value.T, *args,**kwargs)
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
            self.value = np.dot(self.value,M)
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
    def cie_interp(self,wl_new, kind = 'auto', negative_values_allowed = False):
        """
        Interpolate / extrapolate spectral data following standard CIE15-2004.
        
        | The interpolation type depends on the spectrum type defined in :kind:. 
        | Extrapolation is always done by replicate the closest known values.
        
        Args:
            :wl_new: 
                | ndarray with new wavelengths
            :kind:
                | 'auto', optional
                | If :kind: is None, return original data.
                | If :kind: is a spectrum type (see _INTERP_TYPES), the correct 
                |     interpolation type if automatically chosen.
                | If kind = 'auto': use self.dtype
                | Or :kind: can be any interpolation type supported 
                  by scipy.interpolate.interp1d
            :negative_values_allowed:
                | False, optional
                | If False: negative values are clipped to zero
        
        Returns:
            :returns:
                | ndarray of interpolated spectral data.
                | (.shape = (number of spectra+1, number of wavelength in wl_new))
        """
        if (kind == 'auto') & (self.dtype is not None):
            kind = self.dtype
        spd = cie_interp(self.get_(), wl_new, kind = kind, negative_values_allowed = negative_values_allowed)
        self.wl = spd[0]
        self.value = spd[1:]
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
            1. `CIE15:2004, 
            “Colorimetry,” CIE, Vienna, Austria, 2004.
            <http://www.cie.co.at/index.php/index.php?i_ca_id=304)>`_
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

