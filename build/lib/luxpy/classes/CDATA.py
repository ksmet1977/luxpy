# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:51:37 2018

@author: kevin.smet
"""

from .. import _CMF, _COLORTF_DEFAULT_WHITE_POINT, _CIEOBS, _CSPACE, _CSPACE_AXES
from ..cam.colorappearancemodels import _CAM_DEFAULT_WHITE_POINT, _CAM_DEFAULT_CONDITIONS 
from .. import xyz_to_Yxy, xyz_to_Yuv, xyz_to_wuv, xyz_to_lab, xyz_to_luv, xyz_to_Vrb_mb, xyz_to_ipt, xyz_to_Ydlep, xyz_to_xyz, xyz_to_lms
from .. import Yxy_to_xyz, Yuv_to_xyz, lab_to_xyz, luv_to_xyz, Vrb_mb_to_xyz, ipt_to_xyz, Ydlep_to_xyz, lms_to_xyz
from .. import xyz_to_jabM_ciecam02, jabM_ciecam02_to_xyz, xyz_to_jabC_ciecam02, jabC_ciecam02_to_xyz
from .. import xyz_to_jabM_cam16, jabM_cam16_to_xyz, xyz_to_jabC_cam16, jabC_cam16_to_xyz
from .. import xyz_to_jab_cam02ucs, jab_cam02ucs_to_xyz, xyz_to_jab_cam02lcd, jab_cam02lcd_to_xyz, xyz_to_jab_cam02scd, jab_cam02scd_to_xyz 
from .. import xyz_to_jab_cam16ucs, jab_cam16ucs_to_xyz, xyz_to_jab_cam16lcd, jab_cam16lcd_to_xyz, xyz_to_jab_cam16scd, jab_cam16scd_to_xyz
from .. import xyz_to_qabW_cam15u, qabW_cam15u_to_xyz
from .. import xyz_to_lab_cam_sww_2016, lab_cam_sww_2016_to_xyz

from .. import plt, np, todim


class CDATA():
    
    def __init__(self, value = None, relative = True, cieobs = _CIEOBS, dtype = 'xyz'):
        if value is not None:
            self.value = value.copy()
        self.relative = relative
        self.dtype = dtype
        self.cieobs = cieobs
        self.shape = self.value.shape
    
    def getvalues_(self,data):
        if isinstance(data, XYZ):
            data = data.value
        return data
    
    def split_(self):
        return [self.value[...,i] for i in range(self.value.shape[-1])]
     
    def join(self,data):
        if data[0].ndim == 2: #faster implementation
            self.value = np.transpose(np.concatenate(data,axis=0).reshape((np.hstack((len(data),data[0].shape)))),(1,2,0))
        elif data[0].ndim == 1:
            self.value = np.concatenate(data,axis=0).reshape((np.hstack((len(data),data[0].shape)))).T
        else:
            self.value = np.hstack(data)[0]
        return self
    
    def take_(self, indices, axis=None, out=None, mode='raise'):
        return np.take(self.value, indices=indices, axis=axis, out=out, mode=mode)  
    
    def getax_(self, indices = 0, axis = None):
        if (axis is None):
            return self.value[...,indices]
        elif axis == 0:
            return self.value[indices, ...]
        elif axis == 1:
            return self.value[:,indices,...]
        elif (axis == 2) & (self.value.ndim>=2):
            return self.value[...,indices]
   
    def dot(self, M):
        self.value = np.dot(M, self.value.T).T
        self.shape = self.value.shape
        return self
    
    def add(self, data):
        self.value += self.getvalues_(data)
        return self
       
    def sub(self, data):
        self.value -= self.getvalues_(data)
        return self
           
    def mul(self, data):
        self.value *= self.getvalues_(data)
        return self
              
    def div(self, data):
        self.value /= self.getvalues_(data)
        return self
    
    def pow(self, n):
        self.value **= n
        return self
    
    def broadcast(self,data, add_axis = 1, equal_shape = False):
        self.value = todim(self.value,data, add_axis = add_axis, equal_shape = equal_shape)
        self.shape = self.value.shape
        return self
    
    def get_S(self, idx = 0):
        """
        Get spectral data related to light sources (cfr. axis = 1 in xyz numpy.ndarrays).
        
        Args:
            :idx: 0, optional
                Index of light source related spectral data.
        
        Returns:
            :returns: luxpy.CDATA instance with only selected spectral data.
        """
        if self.value.ndim == 3:
            self.value = self.value[:,idx,:]
        else:
            self.value = self.value[idx,:]
        self.shape = self.value.shape
        return self
    
    def get_R(self, idx = 0):
        """
        Get spectral data related to reflectance samples (cfr. axis = 0 in xyz numpy.ndarrays).
        
        Args:
            :idx: 0, optional
                Index of reflectance sample related spectral data.
        
        Returns:
            :returns: luxpy.CDATA instance with only selected spectral data.
        """
        self.value = self.value[idx,...]
        self.shape = self.value.shape
        return self
    
    def get_subset(self, idx_R = None, idx_S = None):
        """
        Get spectral data related to specific light sources and reflectance data
        (cfr. axis = 1 and axis = 0 in xyz numpy.ndarrays).
        
        Args:
            :idx_S: None, optional
                Index of light source related spectral data.
                None: selects all.
            :idx_R: None, optional
                Index of reflectance sample related spectral data.
                None selects all.
        Returns:
            :returns: luxpy.CDATA instance with only selected spectral data.
            
        Note: 
            If ndim < 3: selection is based on :idx_R:
        """
        if idx_S is None:
            idx_S = np.arange(self.value.shape[1])
        if idx_R is None:
            idx_R = np.arange(self.value.shape[0])  
        print(self.value.ndim)
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

    
    def colortf(self, dtype = _CSPACE, **kwargs):
        """
        Convert XYZ tristimulus values to color space coordinates.
        
        Args:
            :dtype: _CSPACE or str, optional
                Convert to this color space.
            :**kwargs: additional input arguments required for color space transformation.
                See specific luxpy function for more info (e.g. ?luxpy.xyz_to_lab)
        
        Returns:
            :returns: luxpy.LAB with .value field that is a numpy.array 
                    with color space coordinates 

        """
        return LAB(value = getattr(self,'to_{:s}'.format(dtype))(**kwargs).value, relative = self.relative, cieobs = self.cieobs, dtype = dtype, **kwargs)

    def plot(self,  ax = None, title = None, **kwargs):
        """
        Plot tristimulus or cone fundamental values.
        
        Args:
            :ax: None or axes handles, optional
                None: create new figure axes, else use :ax: for plotting.
            :title: None or str, optional
                Give plot a title.
            :**kwargs: additional arguments for use with matplotlib.pyplot.scatter
                
        Returns:
            :gca: handle to current axes.
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
            :Yxy: luxpy.LAB with .value field that is a numpy.array 
                    with Yxy chromaticity values (Y value refers to luminance or luminance factor)
        """
        return LAB(value = xyz_to_Yxy(self.value), relative = self.relative, cieobs = self.cieobs, dtype = 'Yxy')
 
    
    
    def to_Yuv(self):
        """ 
        Convert XYZ tristimulus values CIE 1976 Yu'v' chromaticity values.

            
        Returns:
            :Yuv: luxpy.LAB with .value field that is a numpy.array 
                with CIE 1976 Yu'v' chromaticity values (Y value refers to luminance or luminance factor)
    	  """
        return LAB(value = xyz_to_Yuv(self.value), relative = self.relative, cieobs = self.cieobs, dtype = 'Yuv')
    
    
    
    def to_wuv(self, xyzw = _COLORTF_DEFAULT_WHITE_POINT):
        """ 
        Convert XYZ tristimulus values CIE 1964 U*V*W* color space.
         
        Args:
            :xyzw: numpy.array with tristimulus values of white point, optional
                Defaults to luxpy._COLORTF_DEFAULT_WHITE_POINT
            
        Returns:
            :wuv: luxpy.LAB with .value field that is a numpy.array with W*U*V* values 
    	  """
        return LAB(value = xyz_to_wuv(self.value, xyzw = xyzw), relative = self.relative, cieobs = self.cieobs, dtype = 'wuv', xyzw = xyzw)


    def to_lms(self):
        """ 
    	  Convert XYZ tristimulus values or LMS cone fundamental responses 
          to LMS cone fundamental responses.
            
        Returns:
            :lms: luxpy.XYZ with .value field that is a numpy.array with LMS cone fundamental responses	
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
            :xyz: luxpy.XYZ with .value field that is a numpy.array with tristimulus values	
        """
        if self.dtype == 'xyz':
            return self
        elif self.dtype == 'lms' :
            return XYZ(value = lms_to_xyz(self.value, cieobs = self.cieobs, M = None), relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')
 
    
    
    def to_lab(self, xyzw = None, cieobs = _CIEOBS):
        """ 
        Convert XYZ tristimulus values to CIE 1976 L*a*b* (CIELAB) color coordinates.
         
        Args:
            :xyzw: None or numpy.array with tristimulus values of white point, optional
                None defaults to xyz of CIE D65 using the :cieobs: observer.
            :cieobs: luxpy._CIEOBS, optional
                CMF set to use when calculating xyzw.
            
        Returns:
            :lab: luxpy.LAB with .value field that is a numpy.array 
                    with CIE 1976 L*a*b* (CIELAB) color coordinates
        """
        return LAB(value = xyz_to_lab(self.value, xyzw = xyzw, cieobs = cieobs), relative = self.relative, dtype = 'lab', xyzw = xyzw, cieobs = cieobs)


            
    def to_luv(self, xyzw = None, cieobs = _CIEOBS):
        """ 
        vert XYZ tristimulus values to CIE 1976 L*u*v* (CIELUV) color coordinates.
         
        Args:
            :xyzw: None or numpy.array with tristimulus values of white point, optional
                None defaults to xyz of CIE D65 using the :cieobs: observer.
            :cieobs: luxpy._CIEOBS, optional
                CMF set to use when calculating xyzw.
            
        Returns:
            :luv: luxpy.LAB with .value field that is a numpy.array 
                    with CIE 1976 L*u*v* (CIELUV) color coordinates
        """
        return LAB(value = xyz_to_luv(self.value, xyzw = xyzw, cieobs = cieobs), relative = self.relative, dtype = 'luv', xyzw = xyzw, cieobs = cieobs)
 

    
    def to_Vrb_mb(self, cieobs = _CIEOBS, scaling = [1,1], M = None):
        """ 
    	  Convert XYZ tristimulus values to V,r,b (Macleod-Boynton) color coordinates.
        
        Macleod Boynton: V = R+G, r = R/V, b = B/V 
        Note that R,G,B ~ L,M,S
         
        Args:
            :cieobs: luxpy._CIEOBS, optional
                CMF set to use when calculating xyzw.
            :scaling: list of scaling factors for r and b dimensions.
            :M: None, optional
                Conversion matrix for going from XYZ to RGB (LMS) 
                If None, :cieobs: determines the M (function does inversion)
                
        Returns:
            :Vrb: luxpy.LAB with .value field that is a numpy.array
                    numpy.array with V,r,b (Macleod-Boynton) color coordinates
        """
        return LAB(value = xyz_to_Vrb_mb(self.value, cieobs = cieobs, scaling = scaling, M = M), relative = self.relative, dtype = 'Vrb_mb', cieobs = cieobs, scaling = scaling, M = M)
    
         
    
    def to_ipt(self, cieobs = _CIEOBS, xyzw = None, M = None):
        """ 
    	  Convert XYZ tristimulus values to IPT color coordinates.
         
        I: Lightness axis, P, red-green axis, T: yellow-blue axis.
         
        Args:
            :xyzw: None or numpy.array with tristimulus values of white point, optional
                None defaults to xyz of CIE D65 using the :cieobs: observer.
            :cieobs: luxpy._CIEOBS, optional
                CMF set to use when calculating xyz0 for rescaling Mxyz2lms (only when not None).
            :M: None, optional
                None defaults to xyz2lms conversion matrix determined by :cieobs:
            
        Returns:
            :ipt: luxpy.LAB with .value field that is a numpy.array
                    with IPT color coordinates
            
        Note: 
            :xyz: is assumed to be under D65 viewing conditions!! 
            If necessary perform chromatic adaptation !!
        """
        return LAB(value = xyz_to_ipt(self.value, cieobs = cieobs, xyzw = xyzw, M = M), relative = self.relative,  dtype = 'ipt', cieobs = cieobs, xyzw = xyzw, M = M)
        
    

    def to_Ydlep(self, cieobs = _CIEOBS, xyzw = _COLORTF_DEFAULT_WHITE_POINT):
        """ 
    	  Convert XYZ tristimulus values to Y, dominant (complementary) wavelength and excitation purity.
         
        Args:
            :xyzw: None or numpy.array with tristimulus values of white point, optional
                None defaults to xyz of CIE D65 using the :cieobs: observer.
            :cieobs: luxpy._CIEOBS, optional
                CMF set to use when calculating spectrum locus coordinates.
            
        Returns:
            :Ydlep: numpy.array with Y, dominant (complementary) wavelength and excitation purity
        """
        return LAB(value = xyz_to_Ydlep(self.value, cieobs = cieobs, xyzw = xyzw), relative = self.relative, dtype = 'Ydlep', cieobs = cieobs, xyzw = xyzw)
  
    
    
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
    def to_jabM_cam16(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS,  mcat = 'cat16'):
        """
        See ?luxpy.xyz_to_jabM_cam16
        """
        value = xyz_to_jabM_cam16(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jabM_cam16', xyzw = xyzw, Yw = Yw, conditions = conditions, mcat = mcat)



    def to_jabC_cam16(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, mcat = 'cat16'):
        """
        See ?luxpy.xyz_to_jabC_cam16
        """
        value = xyz_to_jabC_cam16(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, mcat = mcat)
        return LAB(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'jabC_cam16', xyzw = xyzw, Yw = Yw, conditions = conditions, mcat = mcat)


    
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
        value = xyz_to_lab_cam_sww_2016(self.value, dataw = xyzw, Yb = Yb, Lw = Lw, relative = relative, parameters = parameters, inputtype = inputtype, cieobs = cieobs)
        return LAB(value = value, dtype = 'cam_sww_2016', xyzw = xyzw, Yb = Yb, Lw = Lw, relative = relative, parameters = parameters, cieobs = cieobs)
          
###############################################################################
    
class LAB(CDATA):
  
    def __init__(self, value = None, relative = True, cieobs = _CIEOBS, dtype = 'lab', \
                 xyzw = None, M = None, scaling = None, \
                 Lw = None, Yw = None, Yb = None, conditions = None, yellowbluepurplecorrect = None, mcat = None, ucstype = None,\
                 fov = None, parameters = None):
        super().__init__(value = value, relative = relative, dtype = dtype)
        
        self.cieobs = cieobs
        
        # specific to some chromaticity / color space transforms    
        self.xyzw = xyzw
        self.M = M
        self.scaling = scaling
        
        # specific to some CAM transforms:
        self.Lw = Lw
        self.Yw = Yw
        self.Yb = Yb
        self.conditions = conditions
        self.yellowbluepurplecorrect = yellowbluepurplecorrect
        self.mcat = mcat
        self.ucstype = ucstype
        self.fov = fov
        self.parameters = parameters
    
    
    def colortf(self, **kwargs):
        """
        Convert color space coordinates to XYZ tristimulus values.
        
        Args:
            :dtype: 'xyz'
                Convert to this color space.
            :**kwargs: additional input arguments required for color space transformation.
                See specific luxpy function for more info (e.g. ?luxpy.xyz_to_lab)
        
        Returns:
            :returns: luxpy.XYZ with .value field that is a numpy.array 
                    with tristimulus values 
        """
        return self.to_xyz(**kwargs)
    
    def plot(self, plt_type = '3d', ax = None, title = None, **kwargs):
        """
        Plot tristimulus or cone fundamental values.
        
        Args:
            :plt_type: '3d' or 3 or '2d or 2, optional
                '3d' or 3: plot all 3 dimensions (lightness/luminance and chromaticity)
                '2d' or 2: plot only chromaticity dimensions.
            :ax: None or axes handles, optional
                None: create new figure axes, else use :ax: for plotting.
            :title: None or str, optional
                Give plot a title.
            :**kwargs: additional arguments for use with matplotlib.pyplot.scatter
                
        Returns:
            :gca: handle to current axes.
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
        return XYZ(value = getattr(self,'{:s}_to_xyz'.format(self.dtype))(**kwargs), relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')
    
    
    
    def Yxy_to_xyz(self,**kwargs):
        """ 
        Convert CIE Yxy chromaticity values to XYZ tristimulus values.
             
        Returns:
            :xyz: numpy.array with tristimulus values
    	  """
        return Yxy_to_xyz(self.value)



    def Yuv_to_xyz(self,**kwargs):
        """ 
        Convert CIE 1976 Yu'v' chromaticity values to XYZ tristimulus values.
             
        Returns:
            :xyz: numpy.array with tristimulus values
    	  """
        return Yuv_to_xyz(self.value)
  
    
    
    def wuv_to_xyz(self, xyzw = _COLORTF_DEFAULT_WHITE_POINT, **kwargs):
        """ 
        Convert CIE 1976 L*a*b* color coordinates to XYZ tristimulus values.
        
        Args:
            :xyzw: numpy.array with tristimulus values of white point, optional
                Defaults to luxpy._COLORTF_DEFAULT_WHITE_POINT
        
        Returns:
            :xyz: numpy.array with tristimulus values
    	  """
        return wuv_to_xyz(self.value, xyzw = xyzw)
    
    
    
    def lab_to_xyz(self, xyzw = None, cieobs = _CIEOBS, **kwargs):
        """ 
        Convert CIE 1976 L*a*b* color coordinates to XYZ tristimulus values.
        
        Args:
            :xyzw: None or numpy.array with tristimulus values of white point, optional
                None defaults to xyz of CIE D65 using the :cieobs: observer.
            :cieobs: luxpy._CIEOBS, optional
                CMF set to use when calculating xyzw.
        
        Returns:
            :xyz: numpy.array with tristimulus values
    	  """
        return lab_to_xyz(self.value, xyzw = xyzw, cieobs = cieobs)
    
    
    
    def luv_to_xyz(self, xyzw = None, cieobs = _CIEOBS, **kwargs):
        """ 
        Convert CIE 1976 L*u*v* color coordinates to XYZ tristimulus values.
        
        Args:
            :xyzw: None or numpy.array with tristimulus values of white point, optional
                None defaults to xyz of CIE D65 using the :cieobs: observer.
            :cieobs: luxpy._CIEOBS, optional
                CMF set to use when calculating xyzw.
        
        Returns:
            :xyz: numpy.array with tristimulus values
    	  """
        return luv_to_xyz(self.value, xyzw = xyzw, cieobs = cieobs)
    
    
    
    def Vrb_mb_to_xyz(self, cieobs = _CIEOBS, scaling = [1,1], M = None, Minverted = False, **kwargs):
        """ 
        Convert V,r,b (Macleod-Boynton) color coordinates to XYZ tristimulus values.
        
        Macleod Boynton: V = R+G, r = R/V, b = B/V 
        Note that R,G,B ~ L,M,S
         
        Args:
            :cieobs: luxpy._CIEOBS, optional
                CMF set to use when getting the default M, the xyz to lms conversion matrix.
            :scaling: list of scaling factors for r and b dimensions.
            :M: None, optional
                Conversion matrix for going from XYZ to RGB (LMS) 
                If None, :cieobs: determines the M (function does inversion)
            :Minverted: False, optional
                Bool that determines whether M should be inverted.
        
        Returns:
            :xyz: numpy.array with tristimulus values
    	  """
        return Vrb_mb_to_xyz(self.value, cieobs = cieobs, scaling = scaling, M = M, Minverted = Minverted)



    def ipt_to_xyz(self, cieobs = _CIEOBS, xyzw = None, M = None, **kwargs):
        """ 
        Convert XYZ tristimulus values to IPT color coordinates.
             
        I: Lightness axis, P, red-green axis, T: yellow-blue axis.
     
        
        Args:
            :xyzw: None or numpy.array with tristimulus values of white point, optional
                None defaults to xyz of CIE D65 using the :cieobs: observer.
            :cieobs: luxpy._CIEOBS, optional
                CMF set to use when calculating xyzw for rescaling M (only when not None).
            :M: None, optional
                None defaults to xyz to lms conversion matrix determined by :cieobs:
        
        Returns:
            :xyz: numpy.array with tristimulus values
    	  """
        return ipt_to_xyz(self.value, xyzw = xyzw, cieobs = cieobs, M = M)
    
    
    
    def Ydlep_to_xyz(self, cieobs = _CIEOBS, xyzw = _COLORTF_DEFAULT_WHITE_POINT, **kwargs):
        """ 
    	 Convert Y, dominant (complementary) wavelength and excitation purity to XYZ tristimulus values.
         
        Args:
            :Ydlep: numpy.array with Y, dominant (complementary) wavelength and excitation purity
            :xyzw: None or numpy.array with tristimulus values of white point, optional
                None defaults to xyz of CIE D65 using the :cieobs: observer.
            :cieobs: luxpy._CIEOBS, optional
                CMF set to use when calculating spectrum locus coordinates.       
        
        Returns:
            :xyz: numpy.array with tristimulus values
    	  """
        return Ydlep_to_xyz(self.value, xyzw = xyzw, cieobs = cieobs)
    
    
    
    #------------------------------------------------------------------------------
    #---color appearance space coordinates-----------------------------------------
    #------------------------------------------------------------------------------
    
    def jabM_ciecam02_to_xyz(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02',**kwargs):
        """
        See ?luxpy.jabM_ciecam02_to_xyz
        """
        value = jabM_ciecam02_to_xyz(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return XYZ(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')



    def jabC_ciecam02_to_xyz(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02',**kwargs):
        """
        See ?luxpy.jabC_ciecam02_to_xyz
        """
        value = jabC_ciecam02_to_xyz(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return XYZ(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')


    
    def jab_cam02ucs_to_xyz(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02',**kwargs):
        """
        See ?luxpy.jab_cam02ucs_to_xyz
        """
        value = jab_cam02ucs_to_xyz(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return XYZ(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')



    def jab_cam02lcd_to_xyz(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02',**kwargs):
        """
        See ?luxpy.jab_cam02lcd_to_xyz
        """
        value = jab_cam02lcd_to_xyz(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return XYZ(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')



    def jab_cam02scd_to_xyz(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02',**kwargs):
        """
        See ?luxpy.jab_cam02scd_to_xyz
        """
        value = jab_cam02scd_to_xyz(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return XYZ(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')
 
    
    
    def jabM_cam16_to_xyz(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat16',**kwargs):
        """
        See ?luxpy.jabM_cam16_to_xyz
        """
        value = jabM_cam16_to_xyz(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return XYZ(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')



    def jabC_cam16_to_xyz(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat16',**kwargs):
        """
        See ?luxpy.jabC_cam16_to_xyz
        """
        value = jabC_cam16_to_xyz(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return XYZ(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')
    
    
    
    def jab_cam16ucs_to_xyz(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat16',**kwargs):
        """
        See ?luxpy.jab_cam16ucs_to_xyz
        """
        value = jab_cam02ucs_to_xyz(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return XYZ(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')



    def jab_cam16lcd_to_xyz(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat16',**kwargs):
        """
        See ?luxpy.jab_cam16lcd_to_xyz
        """
        value = jab_cam16lcd_to_xyz(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return XYZ(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')



    def jab_cam16scd_to_xyz(self, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat16',**kwargs):
        """
        See ?luxpy.jab_cam16scd_to_xyz
        """
        value = jab_cam16scd_to_xyz(self.value, xyzw = xyzw, Yw = Yw, conditions = conditions, yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        return XYZ(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')
    
    
    
    def qabW_cam15u_to_xyz(self, fov = 10.0, parameters = None,**kwargs):
        """
        See ?luxpy.qabW_cam15u_to_xyz
        """
        value = qabW_cam15u_to_xyz(self.value, fov = fov, parameters = parameters)
        return XYZ(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')
 
    
    
    def lab_cam_sww_2016_to_xyz(self, xyzw = None, Yb = 20.0, Lw = 400.0, relative = True, parameters = None, inputtype = 'xyz', cieobs = '2006_10',**kwargs):
        """
        See ?luxpy.lab_cam_sww_2016_to_xyz
        """
        value = lab_cam_sww_2016_to_xyz(self.value, dataw = xyzw, Yb = Yb, Lw = Lw, relative = relative, parameters = parameters, inputtype = inputtype, cieobs = cieobs)
        return XYZ(value = value, relative = self.relative, cieobs = self.cieobs, dtype = 'xyz')


       
     
     
