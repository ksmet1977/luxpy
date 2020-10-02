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
Module supporting class functionality for colorimetric data (CDATA, XYZ, LAB) 
=============================================================================

CDATA fields 
------------

 :self.relative: relative (True) or absolute (False) colorimetric data.  
    
 :self.value: values of spectral data
    
 :self.dtype: colorimetric data type ('xyz', 'Yuv', 'lab', ...)
    
 :self.shape: self.value.shape

 :self.cieobs: CMF set used to determine colorimetric data from spectral data.
   
 
CDATA methods
-------------

 :self.get_values_(): Get values from data and return ndarray. 

 :self.split_(): Split .value along last axis and return list of ndarrays.

 :self.join(): Join data along last axis and return instance.
 
 :self.take_(): Applies numpy.take on .value field.
 
 :self.getax_(): Get elements in .value field along specific axis
 
 :self.dot(): Take dot product with instance.
 
 :self.add(): Add data to instance value field.
 
 :self.sub(): Subtract data from instance value field.
 
 :self.mul(): Multiply data with instance value field.
 
 :self.div(): Divide instance value field by data.
 
 :self.pow(): Raise instance value field to power.
 
 :self.broadcast(): Broadcast instance value field to shape of data.
 
 :self.get_S(): | Get spectral data related to light sources. 
                | (cfr. axis = 1 in xyz ndarrays).
          
 :self.get_R():  | Get spectral data related to reflectance samples.
                 | (cfr. axis = 0 in xyz ndarrays).
            
 :self.get_subset(): | Get spectral data related to specific light source and reflectance data
                     | (cfr. axis = 1 and axis = 0 in xyz ndarrays).



XYZ fields 
----------

Same as CDATA, XYZ inherits from CDATA 



XYZ methods
-----------

 :self.ctf(): Convert XYZ tristimulus values to color space coordinates.
 
 :self.plot(): Plot tristimulus or cone fundamental values.

 :self.to_cspace(): Convert XYZ tristimulus values to ...
                    (Method wrappers for all xyz_to_cspace type functions)
  

          
LAB fields 
----------

| Same as CDATA, LAB inherits from CDATA 
| AND, additionally the following dict field with keys related to color space parameters:
|     
|     self.cspace_par = {}
|     self.cspace_par['cieobs'] = self.cieobs
|   
|    
| # specific to some chromaticity / color space transforms:   
|  
|     self.cspace_par['xyzw'] = xyzw
|     self.cspace_par['M'] = M
|     self.cspace_par['scaling'] = scaling
|     
| # specific to some CAM transforms:
| 
|     self.cspace_par['Lw'] = Lw
|     self.cspace_par['Yw'] = Yw
|     self.cspace_par['Yb'] = Yb
|     self.cspace_par['conditions'] = conditions
|     self.cspace_par['yellowbluepurplecorrect'] = yellowbluepurplecorrect
|     self.cspace_par['mcat'] = mcat
|     self.cspace_par['ucstype'] = ucstype
|     self.cspace_par['fov'] = fov
|     self.cspace_par['parameters'] = parameters


LAB methods
-----------

 :self.ctf(): Convert color space coordinates to XYZ tristimulus values.
 
 :self.to_xyz(): Convert color space coordinates to XYZ tristimulus values. 
 
 :self.plot(): Plot color coordinates.
 

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import _CMF, _COLORTF_DEFAULT_WHITE_POINT, _CIEOBS, _CSPACE, _CSPACE_AXES
from luxpy.color.cam.colorappearancemodels import _CAM_DEFAULT_WHITE_POINT, _CAM_DEFAULT_CONDITIONS 
from luxpy import (xyz_to_Yxy, xyz_to_Yuv, xyz_to_wuv, xyz_to_lab, xyz_to_luv, xyz_to_Vrb_mb, xyz_to_ipt, xyz_to_Ydlep, xyz_to_xyz, xyz_to_lms, 
                   Yxy_to_xyz, Yuv_to_xyz, lab_to_xyz, luv_to_xyz, Vrb_mb_to_xyz, ipt_to_xyz, Ydlep_to_xyz, lms_to_xyz, 
                   xyz_to_jabM_ciecam02, jabM_ciecam02_to_xyz, xyz_to_jabC_ciecam02, jabC_ciecam02_to_xyz, 
                   xyz_to_jabM_ciecam16, jabM_ciecam16_to_xyz, xyz_to_jabC_ciecam16, jabC_ciecam16_to_xyz, 
                   xyz_to_jabz, jabz_to_xyz, xyz_to_jabM_camjabz, jabM_camjabz_to_xyz, xyz_to_jabC_camjabz, jabC_camjabz_to_xyz, 
                   xyz_to_jab_cam02ucs, jab_cam02ucs_to_xyz, xyz_to_jab_cam02lcd, jab_cam02lcd_to_xyz, xyz_to_jab_cam02scd, jab_cam02scd_to_xyz,
                   xyz_to_jab_cam16ucs, jab_cam16ucs_to_xyz, xyz_to_jab_cam16lcd, jab_cam16lcd_to_xyz, xyz_to_jab_cam16scd, jab_cam16scd_to_xyz, 
                   xyz_to_qabW_cam15u, qabW_cam15u_to_xyz, 
                   xyz_to_lab_cam_sww16, lab_cam_sww16_to_xyz,
                   xyz_to_qabM_cam18sl, qabM_cam18sl_to_xyz, xyz_to_qabS_cam18sl, qabS_cam18sl_to_xyz,
                   xyz_to_srgb, srgb_to_xyz, 
                   xyz_to_jabz, jabz_to_xyz,
                   colortf)
from luxpy.utils import plt, np, todim, put_args_in_db

class CDATA():
    
    def __init__(self, value = None, relative = True, cieobs = _CIEOBS, dtype = 'xyz'):
        if value is not None:
            self.value = value.copy()
        self.relative = relative
        self.dtype = dtype
        self.cieobs = cieobs
        self.shape = self.value.shape
    
    def getvalues_(self,data):
        """
        Get values from data and return ndarray.
        """
        if isinstance(data, XYZ):
            data = data.value
        return data
    
    def split_(self):
        """
        Split .value along last axis and return list of ndarrays.
        """
        return [self.value[...,i] for i in range(self.value.shape[-1])]
     
    def join(self,data):
        """
        Join data along last axis and return instance.
        """
        if data[0].ndim == 2: #faster implementation
            self.value = np.transpose(np.concatenate(data,axis=0).reshape((np.hstack((len(data),data[0].shape)))),(1,2,0))
        elif data[0].ndim == 1:
            self.value = np.concatenate(data,axis=0).reshape((np.hstack((len(data),data[0].shape)))).T
        else:
            self.value = np.hstack(data)[0]
        return self
    
    def take_(self, indices, axis=None, out=None, mode='raise'):
        """
        Applies numpy.take on .value field.
        """
        return np.take(self.value, indices=indices, axis=axis, out=out, mode=mode)  
    
    def getax_(self, indices = 0, axis = None):
        """
        Get elements in .value field along specific axis
        """
        if (axis is None):
            return self.value[...,indices]
        elif axis == 0:
            return self.value[indices, ...]
        elif axis == 1:
            return self.value[:,indices,...]
        elif (axis == 2) & (self.value.ndim>=2):
            return self.value[...,indices]
   
    def dot(self, M):
        """
        Take dot product with instance.
        """
        self.value = np.dot(M, self.value.T).T
        self.shape = self.value.shape
        return self
    
    def add(self, data):
        """
        Add data to instance value field.
        """
        self.value += self.getvalues_(data)
        return self
       
    def sub(self, data):
        """
        Subtract data from instance value field.
        """
        self.value -= self.getvalues_(data)
        return self
           
    def mul(self, data):
        """
        Multiply data with instance value field.
        """
        self.value *= self.getvalues_(data)
        return self
              
    def div(self, data):
        """
        Divide instance value field by data.
        """
        self.value /= self.getvalues_(data)
        return self
    
    def pow(self, n):
        """
        Raise instance value field to power.
        """
        self.value **= n
        return self
    
    def broadcast(self,data, add_axis = 1, equal_shape = False):
        """
        Broadcast instance value field to shape of data.
        """
        self.value = todim(self.value,data.shape, add_axis = add_axis, equal_shape = equal_shape)
        self.shape = self.value.shape
        return self
    
    def get_S(self, idx = 0):
        """
        Get spectral data related to light sources. 
        | (cfr. axis = 1 in xyz ndarrays).
        
        Args:
            :idx:
                | 0, optional
                | Index of light source related spectral data.
        
        Returns:
            :returns:
                | luxpy.CDATA instance with only selected spectral data.
        """
        if self.value.ndim == 3:
            self.value = self.value[:,idx,:]
        else:
            self.value = self.value[idx,:]
        self.shape = self.value.shape
        return self
    
    def get_R(self, idx = 0):
        """
        Get spectral data related to reflectance samples.
        | (cfr. axis = 0 in xyz ndarrays).
        
        Args:
            :idx: 
                | 0, optional
                | Index of reflectance sample related spectral data.
        
        Returns:
            :returns: 
                | luxpy.CDATA instance with only selected spectral data.
        """
        self.value = self.value[idx,...]
        self.shape = self.value.shape
        return self
    
    def get_subset(self, idx_R = None, idx_S = None):
        """
        Get spectral data related to specific light source and reflectance data
        | (cfr. axis = 1 and axis = 0 in xyz ndarrays).
        
        Args:
            :idx_S: 
                | None, optional
                | Index of light source related spectral data.
                | None: selects all.
            :idx_R:
                | None, optional
                | Index of reflectance sample related spectral data.
                | None selects all.
        Returns:
            :returns:
                | luxpy.CDATA instance with only selected spectral data.
            
        Note: 
            If ndim < 3: selection is based on :idx_R:
        """
        if idx_S is None:
            idx_S = np.arange(self.value.shape[1])
        if idx_R is None:
            idx_R = np.arange(self.value.shape[0])  
        if self.value.ndim == 3:
            self.value = self.value[idx_R,idx_S,:]
        else:
            self.value = self.value[idx_R,...]
        self.shape = self.value.shape
        return self
    
        
###############################################################################

class XYZ(CDATA):
    
    def __init__(self, value = None, relative = True, cieobs = _CIEOBS, dtype = 'xyz'):
        super().__init__(value = value, relative = relative, cieobs = cieobs, dtype = dtype)

    
    def ctf(self, dtype = _CSPACE, **kwargs):
        """
        Convert XYZ tristimulus values to color space coordinates.
        
        Args:
            :dtype:
                | _CSPACE or str, optional
                | Convert to this color space.
            :**kwargs:
                | additional input arguments required for 
                | color space transformation.
                | See specific luxpy function for more info 
                |     (e.g. ?luxpy.xyz_to_lab)
        
        Returns:
            :returns: 
                | luxpy.LAB with .value field that is a ndarray 
                |     with color space coordinates 

        """
#        return LAB(value = colortf(self.value, tf = dtype, fwtf = kwargs), relative = self.relative, cieobs = self.cieobs, dtype = dtype, **kwargs)
        return LAB(value = getattr(self,'to_{:s}'.format(dtype))(**kwargs).value, relative = self.relative, cieobs = self.cieobs, dtype = dtype, **kwargs)


    def plot(self,  ax = None, title = None, **kwargs):
        """
        Plot tristimulus or cone fundamental values.
        
        Args:
            :ax: 
                | None or axes handles, optional
                | None: create new figure axes, else use :ax: for plotting.
            :title:
                | None or str, optional
                | Give plot a title.
            :**kwargs: 
                | additional arguments for use with 
                | matplotlib.pyplot.scatter
                
        Returns:
            :gca:
                | handle to current axes.
        """
        X,Y,Z = self.split_()
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        if self.dtype == 'xyz':
            ax.scatter(X, Z, Y, **kwargs)
            ax.set_xlabel(_CSPACE_AXES[self.dtype][0])
            ax.set_ylabel(_CSPACE_AXES[self.dtype][2])
            ax.set_zlabel(_CSPACE_AXES[self.dtype][1])
        elif self.dtype == 'lms':
            ax.scatter(X, Y, Z, **kwargs)
            ax.set_xlabel(_CSPACE_AXES[self.dtype][0])
            ax.set_ylabel(_CSPACE_AXES[self.dtype][1])
            ax.set_zlabel(_CSPACE_AXES[self.dtype][2])
        if title is not None:
            ax.set_title(title)
            
        return plt.gca()
    
    
    #------------------------------------------------------------------------------
    #---chromaticity and color space ----------------------------------------------
    #------------------------------------------------------------------------------
    def to_Yxy(self):
        """ 
        Convert XYZ tristimulus values CIE Yxy chromaticity values.
            
        Returns:
            :Yxy: 
                | luxpy.LAB with .value field that is a ndarray 
                | with Yxy chromaticity values. 
                | (Y value refers to luminance or luminance factor)
        """
        return LAB(value = xyz_to_Yxy(self.value), relative = self.relative, cieobs = self.cieobs, dtype = 'Yxy')
 
    
    
    def to_Yuv(self,**kwargs):
        """ 
        Convert XYZ tristimulus values CIE 1976 Yu'v' chromaticity values.

            
        Returns:
            :Yuv: 
                | luxpy.LAB with .value field that is a ndarray 
                | with CIE 1976 Yu'v' chromaticity values.
                | (Y value refers to luminance or luminance factor)
           """
        return LAB(value = xyz_to_Yuv(self.value), relative = self.relative, cieobs = self.cieobs, dtype = 'Yuv')
    
    
    
    def to_wuv(self, xyzw = _COLORTF_DEFAULT_WHITE_POINT):
        """ 
        Convert XYZ tristimulus values CIE 1964 U*V*W* color space.
         
        Args:
            :xyzw:
                | ndarray with tristimulus values of white point, optional
                | Defaults to luxpy._COLORTF_DEFAULT_WHITE_POINT
            
        Returns:
            :wuv: 
                | luxpy.LAB with .value field that is a ndarray 
                | with W*U*V* values.
           """
        return LAB(value = xyz_to_wuv(self.value, xyzw = xyzw), relative = self.relative, cieobs = self.cieobs, dtype = 'wuv', xyzw = xyzw)


    def to_lms(self):
        """ 
        Convert XYZ tristimulus values or LMS cone fundamental responses 
        to LMS cone fundamental responses.
            
        Returns:
            :lms: 
                | luxpy.XYZ with .value field that is a ndarray 
                | with LMS cone fundamental responses.
        """
        if self.dtype == 'lms':
            return self
        elif self.dtype == 'xyz':
            return XYZ(value = xyz_to_lms(self.value, cieobs = self.cieobs, M = None), relative = self.relative, cieobs = self.cieobs, dtype = 'lms')
    
     
    def to_xyz(self):
        """ 
        Convert XYZ tristimulus values or LMS cone fundamental responses 
        to XYZ tristimulus values.
            
        Returns:
            :xyz: 
                | luxpy.XYZ with .value field that is a ndarray 
                | with XYZ tristimulus values.
        """
        if self.dtype == 'xyz':
            return self
        elif self.dtype == 'lms' :
            return XYZ(value = lms_to_xyz(self.value, cieobs = self.cieobs, M = None), relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')
 
    
    
    def to_lab(self, xyzw = None, cieobs = _CIEOBS):
        """ 
        Convert XYZ tristimulus values to CIE 1976 L*a*b* (CIELAB) coordinates.
         
        Args:
            :xyzw: 
                | None or ndarray with xyz values of white point, optional
                | None defaults to xyz of CIE D65 using the :cieobs: observer.
            :cieobs:
                | luxpy._CIEOBS, optional
                | CMF set to use when calculating xyzw.
            
        Returns:
            :lab: 
                | luxpy.LAB with .value field that is a ndarray 
                | with CIE 1976 L*a*b* (CIELAB) color coordinates
        """
        return LAB(value = xyz_to_lab(self.value, xyzw = xyzw, cieobs = cieobs), relative = self.relative, dtype = 'lab', xyzw = xyzw, cieobs = cieobs)


            
    def to_luv(self, xyzw = None, cieobs = _CIEOBS):
        """ 
        Convert XYZ tristimulus values to CIE 1976 L*u*v* (CIELUV) coordinates.
         
        Args:
            :xyzw: 
                | None or ndarray with xyz values of white point, optional
                | None defaults to xyz of CIE D65 using the :cieobs: observer.
            :cieobs:
                | luxpy._CIEOBS, optional
                | CMF set to use when calculating xyzw.
            
        Returns:
            :luv: 
                | luxpy.LAB with .value field that is a ndarray 
                | with CIE 1976 L*u*v* (CIELUV) color coordinates
        """
        return LAB(value = xyz_to_luv(self.value, xyzw = xyzw, cieobs = cieobs), relative = self.relative, dtype = 'luv', xyzw = xyzw, cieobs = cieobs)
 

    
    def to_Vrb_mb(self, cieobs = _CIEOBS, scaling = [1,1], M = None):
        """ 
        Convert XYZ tristimulus values to V,r,b (Macleod-Boynton) coordinates.
        
        | Macleod Boynton: V = R+G, r = R/V, b = B/V 
        | Note that R,G,B ~ L,M,S
         
        Args:
            :cieobs:
                | luxpy._CIEOBS, optional
                | CMF set to use when calculating xyzw.
            :scaling:
                | list of scaling factors for r and b dimensions.
            :M: 
                | None, optional
                | Conversion matrix for going from XYZ to RGB (LMS) 
                | If None, :cieobs: determines the M (function does inversion)
                
        Returns:
            :Vrb: 
                | luxpy.LAB with .value field that is a ndarray
                | ndarray with V,r,b (Macleod-Boynton) color coordinates
        """
        return LAB(value = xyz_to_Vrb_mb(self.value, cieobs = cieobs, scaling = scaling, M = M), relative = self.relative, dtype = 'Vrb_mb', cieobs = cieobs, scaling = scaling, M = M)
    
         
    
    def to_ipt(self, cieobs = _CIEOBS, xyzw = None, M = None):
        """ 
        Convert XYZ tristimulus values to IPT color coordinates.
         
        | I: Lightness axis, P, red-green axis, T: yellow-blue axis.
         
        Args:
            :xyzw: 
                | None or ndarray with xyz values of white point, optional
                | None defaults to xyz of CIE D65 using the :cieobs: observer.
            :cieobs: 
                | luxpy._CIEOBS, optional
                | CMF set to use when calculating xyzw for rescaling Mxyz2lms 
                | (only when not None).
            :M: 
                | None, optional
                | None defaults to conversion matrix determined by :cieobs:
            
        Returns:
            :ipt: 
                | luxpy.LAB with .value field that is a ndarray
                | with IPT color coordinates
            
        Note: 
            :xyz: is assumed to be under D65 viewing conditions!! 
            | If necessary perform chromatic adaptation !!
        """
        return LAB(value = xyz_to_ipt(self.value, cieobs = cieobs, xyzw = xyzw, M = M), relative = self.relative,  dtype = 'ipt', cieobs = cieobs, xyzw = xyzw, M = M)
        
    

    def to_Ydlep(self, cieobs = _CIEOBS, xyzw = _COLORTF_DEFAULT_WHITE_POINT):
        """ 
        Convert XYZ values to Y, dominant (complementary) wavelength 
        and excitation purity.
         
        Args:
            :xyzw: 
                | None or ndarray with xyz values of white point, optional
                | None defaults to xyz of CIE D65 using the :cieobs: observer.
            :cieobs:
                | luxpy._CIEOBS, optional
                | CMF set to use when calculating spectrum locus coordinates.
            
        Returns:
            :Ydlep: 
                | ndarray with Y, dominant (complementary) wavelength 
                | and excitation purity
        """
        return LAB(value = xyz_to_Ydlep(self.value, cieobs = cieobs, xyzw = xyzw), relative = self.relative, dtype = 'Ydlep', cieobs = cieobs, xyzw = xyzw)
  

    def to_srgb(self, gamma = 2.4):
        """
        Calculates IEC:61966 sRGB values from xyz.

        Args:
            :xyz: 
                | ndarray with relative tristimulus values.
            :gamma: 
                | 2.4, optional
                | compression in sRGB

        Returns:
            :rgb: 
                | ndarray with R,G,B values (uint8).
        """
        return LAB(value = xyz_to_srgb(self.value, gamma = gamma), relative = self.relative, cieobs = self.cieobs, dtype = 'srgb')

    
    def to_jabz(self, ztype = 'jabz'):
        """ 
        Convert XYZ tristimulus values to Jz,az,bz color coordinates.

        Args:
            :xyz: 
                | ndarray with absolute tristimulus values (Y in cd/m²!)
            :ztype:
                | 'jabz', optional
                | String with requested return:
                | Options: 'jabz', 'iabz'
                
        Returns:
            :jabz: 
                | ndarray with Jz,az,bz color coordinates

        Notes:
         | 1. :xyz: is assumed to be under D65 viewing conditions! If necessary perform chromatic adaptation!
         |
         | 2a. Jz represents the 'lightness' relative to a D65 white with luminance = 10000 cd/m² 
         |      (note that Jz that not exactly equal 1 for this high value, but rather for 102900 cd/m2)
         | 2b.  az, bz represent respectively a red-green and a yellow-blue opponent axis 
         |      (but note that a D65 shows a small offset from (0,0))

        Reference:
            1. `Safdar, M., Cui, G., Kim,Y. J., and  Luo,M. R. (2017).
                Perceptually uniform color space for image signals including high dynamic range and wide gamut.
                Opt. Express, vol. 25, no. 13, pp. 15131–15151, Jun. 2017.
                <http://www.opticsexpress.org/abstract.cfm?URI=oe-25-13-15131>`_    
        """
        return LAB(value = xyz_to_jabz(self.value, ztype = ztype), relative = self.relative, cieobs = self.cieobs, dtype = 'jabz')

    
    #------------------------------------------------------------------------------
    #---color appearance space coordinates-----------------------------------------
    #------------------------------------------------------------------------------
    
    def to_jabM_ciecam02(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
        """
        See ?luxpy.xyz_to_jabM_ciecam02
        """
        value = xyz_to_jabM_ciecam02(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jabM_ciecam02', xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)



    def to_jabC_ciecam02(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
        """
        See ?luxpy.xyz_to_jabC_ciecam02
        """
        value = xyz_to_jabC_ciecam02(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jabC_ciecam02', xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)


    
    def to_jab_cam02ucs(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
        """
        See ?luxpy.xyz_to_jab_cam02ucs
        """
        value = xyz_to_jab_cam02ucs(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jab_cam02ucs', xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)



    def to_jab_cam02lcd(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
        """
        See ?luxpy.xyz_to_jab_cam02lcd
        """
        value = xyz_to_jab_cam02lcd(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jab_cam02lcd', xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)



    def to_jab_cam02scd(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
        """
        See ?luxpy.xyz_to_jab_cam02scd
        """
        value = xyz_to_jab_cam02scd(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jab_cam02scd', xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)

    
    
    #------------------------------------------------------------------------------
    def to_jabM_ciecam16(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS,  mcat = 'cat16'):
        """
        See ?luxpy.xyz_to_jabM_ciecam16
        """
        value = xyz_to_jabM_ciecam16(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jabM_ciecam16', xyzw = xyzw, Yw = Yw, conditions = conditions, mcat = mcat)



    def to_jabC_ciecam16(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, mcat = 'cat16'):
        """
        See ?luxpy.xyz_to_jabC_ciecam16
        """
        value = xyz_to_jabC_ciecam16(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jabC_ciecam16', xyzw = xyzw, Yw = Yw, conditions = conditions, mcat = mcat)


    
    def to_jab_cam16ucs(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, mcat = 'cat16'):
        """
        See ?luxpy.xyz_to_jab_cam02ucs
        """
        value = xyz_to_jab_cam02ucs(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jab_cam162ucs', xyzw = xyzw, Yw = Yw, conditions = conditions,  mcat = mcat)



    def to_jab_cam16lcd(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS,  mcat = 'cat16'):
        """
        See ?luxpy.xyz_to_jab_cam16lcd
        """
        value = xyz_to_jab_cam02lcd(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jab_cam16lcd', xyzw = xyzw, Yw = Yw, conditions = conditions, mcat = mcat)



    def to_jab_cam16scd(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS,  mcat = 'cat16'):
        """
        See ?luxpy.xyz_to_jab_cam16scd
        """
        value = xyz_to_jab_cam16scd(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions,  mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jab_cam16scd', xyzw = xyzw, Yw = Yw, conditions = conditions, mcat = mcat)

    #------------------------------------------------------------------------------
    def to_jabM_cam_jabz(self, xyzw = None,  conditions = _CAM_DEFAULT_CONDITIONS,  mcat = 'cat16'):
        """
        See ?luxpy.xyz_to_jabM_camjabz
        """
        value = xyz_to_jabM_camjabz(self.value, xyzw = xyzw, cieobs = self.cieobs,  conditions = conditions, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jabM_camjabz', xyzw = xyzw, conditions = conditions, mcat = mcat)



    def to_jabC_camjabz6(self, xyzw = _CAM_DEFAULT_WHITE_POINT, conditions = _CAM_DEFAULT_CONDITIONS, mcat = 'cat16'):
        """
        See ?luxpy.xyz_to_jabC_camjabz
        """
        value = xyz_to_jabC_camjabz(self.value, xyzw = xyzw, cieobs = self.cieobs, conditions = conditions, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jabC_camjabz', xyzw = xyzw, conditions = conditions, mcat = mcat)

   
    #--------------------------------------------------------------------------
    def to_qabW_cam15u(self, fov = 10.0, parameters = None):
        """
        See ?luxpy.xyz_to_qabW_cam15u
        """
        value = xyz_to_qabW_cam15u(self.value, fov = fov, parameters = parameters)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'qabW_cam15u', fov = fov, parameters = parameters)


    #------------------------------------------------------------------------------
    def to_lab_cam_sww_2016(self, xyzw = None, Yb = 20.0, Lw = 400.0, relative = True, parameters = None, inputtype = 'xyz', cieobs = '2006_10'):
        """
        See ?luxpy.xyz_to_lab_cam_sww_2016
        """
        value = xyz_to_lab_cam_sww16(self.value, dataw = xyzw, Yb = Yb, Lw = Lw, relative = relative, parameters = parameters, inputtype = inputtype, cieobs = cieobs)
        return LAB(value = value, dtype = 'cam_sww_2016', xyzw = xyzw, Yb = Yb, Lw = Lw, relative = relative, parameters = parameters, cieobs = cieobs)
  
    #--------------------------------------------------------------------------
    def to_qabS_cam18sl(self, xyz, xyzb = None, Lb = [100], fov = 10.0, parameters = None):
        """
        See ?luxpy.xyz_to_qabS_cam18sl
        """
        value = xyz_to_qabS_cam18sl(self.value, xyzb = xyzb, Lb = Lb, fov = fov, parameters = parameters)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'qabS_cam18sl', xyzb = xyzb, Lb = Lb, fov = fov, parameters = parameters)
   
    #--------------------------------------------------------------------------
    def to_qabM_cam18sl(self, xyz, xyzb = None, Lb = [100], fov = 10.0, parameters = None):
        """
        See ?luxpy.xyz_to_qabM_cam18sl
        """
        value = xyz_to_qabM_cam18sl(self.value, xyzb = xyzb, Lb = Lb, fov = fov, parameters = parameters)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'qabM_cam18sl', xyzb = xyzb, Lb = Lb, fov = fov, parameters = parameters) 
   
###############################################################################
    
class LAB(CDATA):
  
    def __init__(self, value = None, relative = True, cieobs = _CIEOBS, dtype = 'lab', \
                 xyzw = None, M = None, scaling = None, \
                 Lw = None, Yw = None, Yb = None, conditions = None, yellowbluepurplecorrect = None, mcat = None, ucstype = None,\
                 fov = None, parameters = None):
        super().__init__(value = value, relative = relative, cieobs = cieobs, dtype = dtype)
        
        self.cspace_par = {}
        self.cspace_par['cieobs'] = self.cieobs
        
        # specific to some chromaticity / color space transforms    
        self.cspace_par['xyzw'] = xyzw
        self.cspace_par['M'] = M
        self.cspace_par['scaling'] = scaling
        
        # specific to some CAM transforms:
        self.cspace_par['Lw'] = Lw
        self.cspace_par['Yw'] = Yw
        self.cspace_par['Yb'] = Yb
        self.cspace_par['conditions'] = conditions
        self.cspace_par['yellowbluepurplecorrect'] = yellowbluepurplecorrect
        self.cspace_par['mcat'] = mcat
        self.cspace_par['ucstype'] = ucstype
        self.cspace_par['fov'] = fov
        self.cspace_par['parameters'] = parameters
    
    
    def ctf(self, **kwargs):
        """
        Convert color space coordinates to XYZ tristimulus values.
        
        Args:
            :dtype: 
                | 'xyz'
                | Convert to this color space.
            :**kwargs: 
                | additional input arguments required for 
                | color space transformation.
                | See specific luxpy function for more info 
                |   (e.g. ?luxpy.xyz_to_lab)
        
        Returns:
            :returns:
                | luxpy.XYZ with .value field that is a ndarray 
                  with tristimulus values 
        """
        db = put_args_in_db(self.cspace_par,locals().copy()) 
        return XYZ(value = colortf(self.value, tf = '{:s}>xyz'.format(self.dtype), bwtf = db), relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')

    
    def plot(self, plt_type = '3d', ax = None, title = None, **kwargs):
        """
        Plot color coordinates.
        
        Args:
            :plt_type: 
                | '3d' or 3 or '2d or 2, optional
                |   -'3d' or 3: plot all 3 dimensions (lightness and chromaticity)
                |   -'2d' or 2: plot only chromaticity dimensions.
            :ax: 
                | None or axes handles, optional
                | None: create new figure axes, else use :ax: for plotting.
            :title: 
                | None or str, optional
                | Give plot a title.
            :**kwargs:
                | additional arguments for use with 
                | matplotlib.pyplot.scatter
                
        Returns:
            :gca: 
                | handle to current axes.
        """
        L,a,b = self.split_()
        if ax is None:
            fig = plt.figure()

        if (plt_type == '2d') | (plt_type == 2):
            if ax is None:
                ax = fig.add_subplot(111)
            ax.scatter(a, b, **kwargs)
        else:
            if ax is None:
                ax = fig.add_subplot(111, projection='3d')
            ax.scatter(a, b, L, **kwargs)
            ax.set_zlabel(_CSPACE_AXES[self.dtype][0])
        ax.set_xlabel(_CSPACE_AXES[self.dtype][1])
        ax.set_ylabel(_CSPACE_AXES[self.dtype][2])
        
        if title is not None:
            ax.set_title(title)
        
        return plt.gca()
    
    
    #------------------------------------------------------------------------------
    #---chromaticity coordinates---------------------------------------------------
    #------------------------------------------------------------------------------
        
    def to_xyz(self,**kwargs):
        """
        Convert color space coordinates to XYZ tristimulus values.
        """
        db = put_args_in_db(self.cspace_par,locals().copy()) 
        return XYZ(value = colortf(self.value, tf = '{:s}>xyz'.format(self.dtype),bwtf = db), relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')
    
