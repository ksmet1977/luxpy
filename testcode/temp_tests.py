# -*- coding: utf-8 -*-
"""
Test to check if code changes increase performance of color transform functions.
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

import numpy as np
from luxpy import _CIEOBS, todim, _COLORTF_DEFAULT_WHITE_POINT,_IPT_M, spd_to_xyz, np2d, np3d, _CIE_ILLUMINANTS, math, xyz_to_Yxy, asplit, ajoin, _CMF

def xyz_to_ipt(xyz, cieobs = _CIEOBS, xyzw = None, M = None, **kwargs):
    """
    Convert XYZ tristimulus values to IPT color coordinates.

    | I: Lightness axis, P, red-green axis, T: yellow-blue axis.

    Args:
        :xyz: 
            | ndarray with tristimulus values
        :xyzw: 
            | None or ndarray with tristimulus values of white point, optional
            | None defaults to xyz of CIE D65 using the :cieobs: observer.
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set to use when calculating xyzw for rescaling M
              (only when not None).
        :M: | None, optional
            | None defaults to xyz to lms conversion matrix determined by :cieobs:

    Returns:
        :ipt: 
            | ndarray with IPT color coordinates

    Note:
        :xyz: is assumed to be under D65 viewing conditions! If necessary 
              perform chromatic adaptation !

    Reference:
        1. `Ebner F, and Fairchild MD (1998).
           Development and testing of a color space (IPT) with improved hue uniformity.
           In IS&T 6th Color Imaging Conference, (Scottsdale, Arizona, USA), pp. 8–13.
           <http://www.ingentaconnect.com/content/ist/cic/1998/00001998/00000001/art00003?crawler=true>`_
    """
    xyz = np2d(xyz)

    # get M to convert xyz to lms and apply normalization to matrix or input your own:
    if M is None:
        M = _IPT_M['xyz2lms'][cieobs].copy() # matrix conversions from xyz to lms
        if xyzw is None:
            xyzw = spd_to_xyz(_CIE_ILLUMINANTS['D65'],cieobs = cieobs, out = 1)[0]/100.0
        else:
            xyzw = xyzw/100.0
        M = math.normalize_3x3_matrix(M,xyzw)

    # get xyz and normalize to 1:
    xyz = xyz/100.0

    # convert xyz to lms:
    if np.ndim(M)==2:
        if len(xyz.shape) == 3:
            lms = np.einsum('ij,klj->kli', M, xyz)
        else:
            lms = np.einsum('ij,lj->li', M, xyz)
    else:
        if len(xyz.shape) == 3: # second dim of xyz must match dim of 1st of M and 1st dim of xyzw
            lms = np.concatenate([np.einsum('ij,klj->kli', M[i], xyz[:,i:i+1,:]) for i in range(M.shape[0])],axis=1)
        else: # first dim of xyz must match dim of 1st of M and 1st dim of xyzw
            lms = np.concatenate([np.einsum('ij,lj->li', M[i], xyz[i:i+1,:]) for i in range(M.shape[0])],axis=0)
        
    #lms = np.dot(M,xyz.T).T

    #response compression: lms to lms'
    lmsp = lms**0.43
    p = np.where(lms<0.0)
    lmsp[p] = -np.abs(lms[p])**0.43

    # convert lms' to ipt coordinates:
    if len(xyz.shape) == 3:
        ipt = np.einsum('ij,klj->kli', _IPT_M['lms2ipt'], lmsp)
    else:
        ipt = np.einsum('ij,lj->li', _IPT_M['lms2ipt'], lmsp)

    return ipt

def xyz_to_ipt2(xyz, cieobs = _CIEOBS, xyzw = None, M = None, **kwargs):
    """
    Convert XYZ tristimulus values to IPT color coordinates.

    | I: Lightness axis, P, red-green axis, T: yellow-blue axis.

    Args:
        :xyz: 
            | ndarray with tristimulus values
        :xyzw: 
            | None or ndarray with tristimulus values of white point, optional
            | None defaults to xyz of CIE D65 using the :cieobs: observer.
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set to use when calculating xyzw for rescaling M
              (only when not None).
        :M: | None, optional
            | None defaults to xyz to lms conversion matrix determined by :cieobs:

    Returns:
        :ipt: 
            | ndarray with IPT color coordinates

    Note:
        :xyz: is assumed to be under D65 viewing conditions! If necessary 
              perform chromatic adaptation !

    Reference:
        1. `Ebner F, and Fairchild MD (1998).
           Development and testing of a color space (IPT) with improved hue uniformity.
           In IS&T 6th Color Imaging Conference, (Scottsdale, Arizona, USA), pp. 8–13.
           <http://www.ingentaconnect.com/content/ist/cic/1998/00001998/00000001/art00003?crawler=true>`_
    """
    xyz = np2d(xyz)

    # get M to convert xyz to lms and apply normalization to matrix or input your own:
    if M is None:
        M = _IPT_M['xyz2lms'][cieobs].copy() # matrix conversions from xyz to lms
        if xyzw is None:
            xyzw = spd_to_xyz(_CIE_ILLUMINANTS['D65'],cieobs = cieobs, out = 1)[0]/100.0
        else:
            xyzw = xyzw/100.0
        M = math.normalize_3x3_matrix(M,xyzw)

    # get xyz and normalize to 1:
    xyz = xyz/100.0

    # convert xyz to lms:
    if np.ndim(M)==2:
        if len(xyz.shape) == 3:
            lms = np.einsum('ij,klj->kli', M, xyz)
        else:
            lms = np.einsum('ij,lj->li', M, xyz)
    else:
        if len(xyz.shape) == 3: # second dim of xyz must match dim of 1st of M and 1st dim of xyzw
            lms = np.concatenate([np.einsum('ij,klj->kli', M[i], xyz[:,i:i+1,:]) for i in range(M.shape[0])],axis=1)
        else: # first dim of xyz must match dim of 1st of M and 1st dim of xyzw
            lms = np.concatenate([np.einsum('ij,lj->li', M[i], xyz[i:i+1,:]) for i in range(M.shape[0])],axis=0)
        
    #lms = np.dot(M,xyz.T).T

    #response compression: lms to lms'
    lmsp = lms**0.43
    p = np.where(lms<0.0)
    lmsp[p] = -np.abs(lms[p])**0.43

    # convert lms' to ipt coordinates:
    if len(xyz.shape) == 3:
        ipt = np.einsum('ij,klj->kli', _IPT_M['lms2ipt'], lmsp)
    else:
        ipt = np.einsum('ij,lj->li', _IPT_M['lms2ipt'], lmsp)

    return ipt


def xyz_to_Vrb_mb(xyz, cieobs = _CIEOBS, scaling = [1,1], M = None, **kwargs):
    """
    Convert XYZ tristimulus values to V,r,b (Macleod-Boynton) color coordinates.

    | Macleod Boynton: V = R+G, r = R/V, b = B/V
    | Note that R,G,B ~ L,M,S

    Args:
        :xyz: 
            | ndarray with tristimulus values
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set to use when getting the default M, which is
              the xyz to lms conversion matrix.
        :scaling:
            | list of scaling factors for r and b dimensions.
        :M: 
            | None, optional
            | Conversion matrix for going from XYZ to RGB (LMS)
            |   If None, :cieobs: determines the M (function does inversion)

    Returns:
        :Vrb: 
            | ndarray with V,r,b (Macleod-Boynton) color coordinates

    Reference:
        1. `MacLeod DI, and Boynton RM (1979).
           Chromaticity diagram showing cone excitation by stimuli of equal luminance.
           J. Opt. Soc. Am. 69, 1183–1186.
           <https://www.osapublishing.org/josa/abstract.cfm?uri=josa-69-8-1183>`_
    """
    xyz = np2d(xyz)

    X,Y,Z = asplit(xyz)
    if M is None:
        M = _CMF[cieobs]['M']
    R, G, B = [M[i,0]*X + M[i,1]*Y + M[i,2]*Z for i in range(3)]
    V = R + G
    r = R / V * scaling[0]
    b = B / V * scaling[1]
    return ajoin((V,r,b))

def xyz_to_Vrb_mb2(xyz, cieobs = _CIEOBS, scaling = [1,1], M = None, **kwargs):
    """
    Convert XYZ tristimulus values to V,r,b (Macleod-Boynton) color coordinates.

    | Macleod Boynton: V = R+G, r = R/V, b = B/V
    | Note that R,G,B ~ L,M,S

    Args:
        :xyz: 
            | ndarray with tristimulus values
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set to use when getting the default M, which is
              the xyz to lms conversion matrix.
        :scaling:
            | list of scaling factors for r and b dimensions.
        :M: 
            | None, optional
            | Conversion matrix for going from XYZ to RGB (LMS)
            |   If None, :cieobs: determines the M (function does inversion)

    Returns:
        :Vrb: 
            | ndarray with V,r,b (Macleod-Boynton) color coordinates

    Reference:
        1. `MacLeod DI, and Boynton RM (1979).
           Chromaticity diagram showing cone excitation by stimuli of equal luminance.
           J. Opt. Soc. Am. 69, 1183–1186.
           <https://www.osapublishing.org/josa/abstract.cfm?uri=josa-69-8-1183>`_
    """
    xyz = np2d(xyz)

    if M is None:
        M = _CMF[cieobs]['M']
        
    if len(xyz.shape) == 3:
        RGB = np.einsum('ij,klj->kli', M, xyz)
    else:
        RGB = np.einsum('ij,lj->li', M, xyz)
    Vrb = xyz.astype(np.float)        
    Vrb[...,0] = RGB[...,0] + RGB[...,1]
    Vrb[...,1] = RGB[...,0] / Vrb[...,0] * scaling[0]
    Vrb[...,2] = RGB[...,2] / Vrb[...,0] * scaling[1]
    return Vrb

def Vrb_mb_to_xyz(Vrb,cieobs = _CIEOBS, scaling = [1,1], M = None, Minverted = False, **kwargs):
    """
    Convert V,r,b (Macleod-Boynton) color coordinates to XYZ tristimulus values.

    | Macleod Boynton: V = R+G, r = R/V, b = B/V
    | Note that R,G,B ~ L,M,S

    Args:
        :Vrb: 
            | ndarray with V,r,b (Macleod-Boynton) color coordinates
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set to use when getting the default M, which is
              the xyz to lms conversion matrix.
        :scaling:
            | list of scaling factors for r and b dimensions.
        :M: 
            | None, optional
            | Conversion matrix for going from XYZ to RGB (LMS)
            |   If None, :cieobs: determines the M (function does inversion)
        :Minverted:
            | False, optional
            | Bool that determines whether M should be inverted.

    Returns:
        :xyz: 
            | ndarray with tristimulus values

    Reference:
        1. `MacLeod DI, and Boynton RM (1979).
           Chromaticity diagram showing cone excitation by stimuli of equal luminance.
           J. Opt. Soc. Am. 69, 1183–1186.
           <https://www.osapublishing.org/josa/abstract.cfm?uri=josa-69-8-1183>`_
    """
    Vrb = np2d(Vrb)

    V,r,b = asplit(Vrb)
    R = r*V / scaling[0]
    B = b*V / scaling[1]
    G = V-R
    if M is None:
        M = _CMF[cieobs]['M']
    if Minverted == False:
        M = np.linalg.inv(M)
    X, Y, Z = [M[i,0]*R + M[i,1]*G + M[i,2]*B for i in range(3)]
    return ajoin((X,Y,Z))


def Vrb_mb_to_xyz2(Vrb,cieobs = _CIEOBS, scaling = [1,1], M = None, Minverted = False, **kwargs):
    """
    Convert V,r,b (Macleod-Boynton) color coordinates to XYZ tristimulus values.

    | Macleod Boynton: V = R+G, r = R/V, b = B/V
    | Note that R,G,B ~ L,M,S

    Args:
        :Vrb: 
            | ndarray with V,r,b (Macleod-Boynton) color coordinates
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set to use when getting the default M, which is
              the xyz to lms conversion matrix.
        :scaling:
            | list of scaling factors for r and b dimensions.
        :M: 
            | None, optional
            | Conversion matrix for going from XYZ to RGB (LMS)
            |   If None, :cieobs: determines the M (function does inversion)
        :Minverted:
            | False, optional
            | Bool that determines whether M should be inverted.

    Returns:
        :xyz: 
            | ndarray with tristimulus values

    Reference:
        1. `MacLeod DI, and Boynton RM (1979).
           Chromaticity diagram showing cone excitation by stimuli of equal luminance.
           J. Opt. Soc. Am. 69, 1183–1186.
           <https://www.osapublishing.org/josa/abstract.cfm?uri=josa-69-8-1183>`_
    """
    Vrb = np2d(Vrb)
    RGB = Vrb.astype(np.float)
    RGB[...,0] = Vrb[...,1]*Vrb[...,0] / scaling[0]
    RGB[...,2] = Vrb[...,2]*Vrb[...,0] / scaling[1]
    RGB[...,1] = Vrb[...,0] - RGB[...,0]
    if M is None:
        M = _CMF[cieobs]['M']
    if Minverted == False:
        M = np.linalg.inv(M)
    
    if len(RGB.shape) == 3:
        return np.einsum('ij,klj->kli', M, RGB)
    else:
        return np.einsum('ij,lj->li', M, RGB)


def xyz_to_luv(xyz, xyzw = None, cieobs = _CIEOBS, **kwargs):
    """
    Convert XYZ tristimulus values to CIE 1976 L*u*v* (CIELUV) coordinates.

    Args:
        :xyz: 
            | ndarray with tristimulus values
        :xyzw:
            | None or ndarray with tristimulus values of white point, optional
            | None defaults to xyz of CIE D65 using the :cieobs: observer.
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set to use when calculating xyzw.

    Returns:
        :luv: 
            | ndarray with CIE 1976 L*u*v* (CIELUV) color coordinates
    """
    xyz = np2d(xyz)

    if xyzw is None:
        xyzw = spd_to_xyz(_CIE_ILLUMINANTS['D65'],cieobs = cieobs)

    # make xyzw same shape as xyz:
    xyzw = todim(xyzw, xyz.shape)

    # Calculate u',v' of test and white:
    Y,u,v = asplit(xyz_to_Yuv(xyz))
    Yw,uw,vw = asplit(xyz_to_Yuv(xyzw))

    #uv1976 to CIELUV
    YdivYw = Y / Yw
    L = 116.0*YdivYw**(1.0/3.0) - 16.0
    p = np.where(YdivYw <= (6.0/29.0)**3.0)
    L[p] = ((29.0/3.0)**3.0)*YdivYw[p]
    u = 13.0*L*(u-uw)
    v = 13.0*L*(v-vw)

    return ajoin((L,u,v))

def xyz_to_luv2(xyz, xyzw = None, cieobs = _CIEOBS, **kwargs):
    """
    Convert XYZ tristimulus values to CIE 1976 L*u*v* (CIELUV) coordinates.

    Args:
        :xyz: 
            | ndarray with tristimulus values
        :xyzw:
            | None or ndarray with tristimulus values of white point, optional
            | None defaults to xyz of CIE D65 using the :cieobs: observer.
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set to use when calculating xyzw.

    Returns:
        :luv: 
            | ndarray with CIE 1976 L*u*v* (CIELUV) color coordinates
    """
    xyz = np2d(xyz)

    if xyzw is None:
        xyzw = spd_to_xyz(_CIE_ILLUMINANTS['D65'],cieobs = cieobs)

    # Calculate u',v' of test and white:
    Yuv = xyz_to_Yuv(xyz)
    Yuvw = xyz_to_Yuv(todim(xyzw, xyz.shape)) # todim: make xyzw same shape as xyz

    #uv1976 to CIELUV
    luv = xyz.astype(np.float)
    YdivYw = Yuv[...,0] / Yuvw[...,0]
    luv[...,0] = 116.0*YdivYw**(1.0/3.0) - 16.0
    p = np.where(YdivYw <= (6.0/29.0)**3.0)
    luv[...,0][p] = ((29.0/3.0)**3.0)*YdivYw[p]
    luv[...,1] = 13.0*luv[...,0]*(Yuv[...,1]-Yuvw[...,1])
    luv[...,2] = 13.0*luv[...,0]*(Yuv[...,2]-Yuvw[...,2])
    return luv

def luv_to_xyz(luv, xyzw = None, cieobs = _CIEOBS, **kwargs):
    """
    Convert CIE 1976 L*u*v* (CIELUVB) coordinates to XYZ tristimulus values.

    Args:
        :luv: 
            | ndarray with CIE 1976 L*u*v* (CIELUV) color coordinates
        :xyzw:
            | None or ndarray with tristimulus values of white point, optional
            | None defaults to xyz of CIE D65 using the :cieobs: observer.
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set to use when calculating xyzw.

    Returns:
        :xyz: 
            | ndarray with tristimulus values
    """
    luv = np2d(luv)


    if xyzw is None:
        xyzw = spd_to_xyz(_CIE_ILLUMINANTS['D65'],cieobs = cieobs)

    # Make xyzw same shape as luv:
    Yuvw = todim(xyz_to_Yuv(xyzw), luv.shape, equal_shape = True)

    # Get Yw, uw,vw:
    Yw,uw,vw = asplit(Yuvw)

    # calculate u'v' from u*,v*:
    L,u,v = asplit(luv)
    up,vp = [(x / (13*L)) + xw for (x,xw) in ((u,uw),(v,vw))]
    up[np.where(L == 0.0)] = 0.0
    vp[np.where(L == 0.0)] = 0.0

    fy = (L + 16.0) / 116.0
    Y = Yw*(fy**3.0)
    p = np.where((Y/Yw) < ((6.0/29.0)**3.0))
    Y[p] = Yw[p]*(L[p]/((29.0/3.0)**3.0))

    return Yuv_to_xyz(ajoin((Y,up,vp)))

def luv_to_xyz2(luv, xyzw = None, cieobs = _CIEOBS, **kwargs):
    """
    Convert CIE 1976 L*u*v* (CIELUVB) coordinates to XYZ tristimulus values.

    Args:
        :luv: 
            | ndarray with CIE 1976 L*u*v* (CIELUV) color coordinates
        :xyzw:
            | None or ndarray with tristimulus values of white point, optional
            | None defaults to xyz of CIE D65 using the :cieobs: observer.
        :cieobs: 
            | luxpy._CIEOBS, optional
            | CMF set to use when calculating xyzw.

    Returns:
        :xyz: 
            | ndarray with tristimulus values
    """
    luv = np2d(luv)

    if xyzw is None:
        xyzw = spd_to_xyz(_CIE_ILLUMINANTS['D65'],cieobs = cieobs)

    # Make xyzw same shape as luv and convert to Yuv:
    Yuvw = todim(xyz_to_Yuv(xyzw), luv.shape, equal_shape = True)

    # calculate u'v' from u*,v*:
    Yuv = luv.astype(np.float)
    Yuv[...,1:3] = (luv[...,1:3] / (13*luv[...,0])) + Yuvw[...,1:3]
    Yuv[Yuv[...,0]==0,1:3] = 0

    Yuv[...,0] = Yuvw[...,0]*(((luv[...,0] + 16.0) / 116.0)**3.0)
    p = np.where((Yuv[...,0]/Yuvw[...,0]) < ((6.0/29.0)**3.0))
    Yuv[...,0][p] = Yuvw[...,0][p]*(luv[...,0][p]/((29.0/3.0)**3.0))

    return Yuv_to_xyz(Yuv)



def wuv_to_xyz(wuv,xyzw = _COLORTF_DEFAULT_WHITE_POINT, **kwargs):
    """
    Convert CIE 1964 U*V*W* color space coordinates to XYZ tristimulus values.

    Args:
        :wuv: 
            | ndarray with W*U*V* values
        :xyzw: 
            | ndarray with tristimulus values of white point, optional
              (Defaults to luxpy._COLORTF_DEFAULT_WHITE_POINT)

    Returns:
        :xyz: 
            | ndarray with tristimulus values
	 """
    wuv = np2d(wuv)
    xyzw = np2d(xyzw)

    Yuvw = xyz_to_Yuv(xyzw) # convert to cie 1976 u'v'
    Yw, uw, vw = asplit(Yuvw)
    vw = (2.0/3.0)*vw # convert to cie 1960 u, v
    W,U,V = asplit(wuv)
    Y = ((W + 17.0) / 25.0)**3.0
    u = uw + U/(13.0*W)
    v = (vw + V/(13.0*W)) * (3.0/2.0)
    Yuv = ajoin((Y,u,v)) # = 1976 u',v'
    return Yuv_to_xyz(Yuv)


def wuv_to_xyz2(wuv,xyzw = _COLORTF_DEFAULT_WHITE_POINT, **kwargs):
    """
    Convert CIE 1964 U*V*W* color space coordinates to XYZ tristimulus values.

    Args:
        :wuv: 
            | ndarray with W*U*V* values
        :xyzw: 
            | ndarray with tristimulus values of white point, optional
              (Defaults to luxpy._COLORTF_DEFAULT_WHITE_POINT)

    Returns:
        :xyz: 
            | ndarray with tristimulus values
	 """
    wuv = np2d(wuv)
    Yuvw = xyz_to_Yuv2(xyzw) # convert to cie 1976 u'v'
    Yuv = wuv.astype(np.float)
    Yuv[...,0] = ((wuv[...,0] + 17.0) / 25.0)**3.0
    Yuv[...,1] = Yuvw[...,1] + wuv[...,1]/(13.0*wuv[...,0])
    Yuv[...,2] = Yuvw[...,2] + wuv[...,2]/(13.0*wuv[...,0]) * (3.0/2.0) # convert to cie 1960 u, v
    return Yuv_to_xyz2(Yuv)


def xyz_to_wuv(xyz, xyzw = _COLORTF_DEFAULT_WHITE_POINT, **kwargs):
    """
    Convert XYZ tristimulus values CIE 1964 U*V*W* color space.

    Args:
        :xyz: 
            | ndarray with tristimulus values
        :xyzw: 
            | ndarray with tristimulus values of white point, optional
              (Defaults to luxpy._COLORTF_DEFAULT_WHITE_POINT)

    Returns:
        :wuv: 
            | ndarray with W*U*V* values
    """
    xyz = np2d(xyz)
    xyzw = np2d(xyzw)
    Yuv = xyz_to_Yuv(xyz) # convert to cie 1976 u'v'
    Yuvw = xyz_to_Yuv(xyzw)
    Y, u, v = asplit(Yuv)
    Yw, uw, vw = asplit(Yuvw)
    W = 25.0*(Y**(1/3)) - 17.0
    U = 13.0*W*(u - uw)
    V = 13.0*W*(v - vw)*(2.0/3.0) # *(2/3) to  convert to cie 1960 u, v
    return ajoin((W,U,V))

def xyz_to_wuv2(xyz, xyzw = _COLORTF_DEFAULT_WHITE_POINT, **kwargs):
    """
    Convert XYZ tristimulus values CIE 1964 U*V*W* color space.

    Args:
        :xyz: 
            | ndarray with tristimulus values
        :xyzw: 
            | ndarray with tristimulus values of white point, optional
              (Defaults to luxpy._COLORTF_DEFAULT_WHITE_POINT)

    Returns:
        :wuv: 
            | ndarray with W*U*V* values
    """
    Yuv = xyz_to_Yuv2(xyz) # convert to cie 1976 u'v'
    Yuvw = xyz_to_Yuv2(xyzw)
    wuv = xyz.astype(np.float)
    wuv[...,0] = 25.0*(Yuv[...,0]**(1/3)) - 17.0
    wuv[...,1] = 13.0*wuv[...,0]*(Yuv[...,1] - Yuvw[...,1])
    wuv[...,2] = 13.0*wuv[...,0]*(Yuv[...,2] - Yuvw[...,2])*(2.0/3.0) #*(2/3) to convert to cie 1960 u, v
    return wuv

def xyz_to_Yxy(xyz, **kwargs):
    """
    Convert XYZ tristimulus values CIE Yxy chromaticity values.

    Args:
        :xyz: 
            | ndarray with tristimulus values

    Returns:
        :Yxy: 
            | ndarray with Yxy chromaticity values
              (Y value refers to luminance or luminance factor)
    """
    xyz = np2d(xyz)
    X,Y,Z = asplit(xyz)
    sumxyz = X + Y + Z
    x = X / sumxyz
    y = Y / sumxyz
    return ajoin((Y,x,y))


def Yxy_to_xyz(Yxy, **kwargs):
    """
    Convert CIE Yxy chromaticity values to XYZ tristimulus values.

    Args:
        :Yxy: 
            | ndarray with Yxy chromaticity values
              (Y value refers to luminance or luminance factor)

    Returns:
        :xyz: 
            | ndarray with tristimulus values
    """
    Yxy = np2d(Yxy)

    Y,x,y = asplit(Yxy)
    X = Y*x/y
    Z = Y*(1.0-x-y)/y
    return ajoin((X,Y,Z))


def xyz_to_Yxy2(xyz, **kwargs):
    """
    Convert XYZ tristimulus values CIE Yxy chromaticity values.

    Args:
        :xyz: 
            | ndarray with tristimulus values

    Returns:
        :Yxy: 
            | ndarray with Yxy chromaticity values
              (Y value refers to luminance or luminance factor)
    """
    xyz = np2d(xyz)
    Yxy = xyz.astype(np.float)
    sumxyz = xyz[...,0] + xyz[...,1] + xyz[...,2]
    Yxy[...,0] = xyz[...,1]
    Yxy[...,1] = xyz[...,0] / sumxyz
    Yxy[...,2] = xyz[...,1] / sumxyz
    return Yxy


def Yxy_to_xyz2(Yxy, **kwargs):
    """
    Convert CIE Yxy chromaticity values to XYZ tristimulus values.

    Args:
        :Yxy: 
            | ndarray with Yxy chromaticity values
              (Y value refers to luminance or luminance factor)

    Returns:
        :xyz: 
            | ndarray with tristimulus values
    """
    Yxy = np2d(Yxy)
    xyz = Yxy.astype(np.float)
    xyz[...,1] = Yxy[...,0]
    xyz[...,0] = Yxy[...,0]*Yxy[...,1]/Yxy[...,2]
    xyz[...,2] = Yxy[...,0]*(1.0-Yxy[...,1]-Yxy[...,2])/Yxy[...,2]
    return xyz



def xyz_to_lab(xyz, xyzw = None, cieobs = _CIEOBS, **kwargs):
    """
    Convert XYZ tristimulus values to CIE 1976 L*a*b* (CIELAB) coordinates.

    Args:
        :xyz: 
            | ndarray with tristimulus values
        :xyzw:
            | None or ndarray with tristimulus values of white point, optional
            | None defaults to xyz of CIE D65 using the :cieobs: observer.
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set to use when calculating xyzw.

    Returns:
        :lab: 
            | ndarray with CIE 1976 L*a*b* (CIELAB) color coordinates
    """
    xyz = np2d(xyz)

    if xyzw is None:
        xyzw = spd_to_xyz(_CIE_ILLUMINANTS['D65'], cieobs = cieobs)

    # get and normalize (X,Y,Z) to white point:
    XYZr = xyz/xyzw

    # Apply cube-root compression:
    fXYZr = XYZr**(1.0/3.0)

    # Check for T/Tn <= 0.008856: (Note (24/116)**3 = 0.008856)
    pqr = XYZr<=(24/116)**3

    # calculate f(T) for T/Tn <= 0.008856: (Note:(1/3)*((116/24)**2) = 841/108 = 7.787)
    fXYZr[pqr] = ((841/108)*XYZr[pqr]+16.0/116.0)

    # calculate L*, a*, b*:
    L = 116.0*(fXYZr[...,1]) - 16.0
    L[pqr[...,1]] = 903.3*XYZr[pqr[...,1],1]
    a = 500.0*(fXYZr[...,0]-fXYZr[...,1])
    b = 200.0*(fXYZr[...,1]-fXYZr[...,2])
    return ajoin((L,a,b))

def xyz_to_lab2(xyz, xyzw = None, cieobs = _CIEOBS, **kwargs):
    """
    Convert XYZ tristimulus values to CIE 1976 L*a*b* (CIELAB) coordinates.

    Args:
        :xyz: 
            | ndarray with tristimulus values
        :xyzw:
            | None or ndarray with tristimulus values of white point, optional
            | None defaults to xyz of CIE D65 using the :cieobs: observer.
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set to use when calculating xyzw.

    Returns:
        :lab: 
            | ndarray with CIE 1976 L*a*b* (CIELAB) color coordinates
    """
    xyz = np2d(xyz)

    if xyzw is None:
        xyzw = spd_to_xyz(_CIE_ILLUMINANTS['D65'], cieobs = cieobs)

    # get and normalize (X,Y,Z) to white point:
    XYZr = xyz/xyzw

    # Apply cube-root compression:
    fXYZr = XYZr**(1.0/3.0)

    # Check for T/Tn <= 0.008856: (Note (24/116)**3 = 0.008856)
    pqr = XYZr<=(24/116)**3

    # calculate f(T) for T/Tn <= 0.008856: (Note:(1/3)*((116/24)**2) = 841/108 = 7.787)
    fXYZr[pqr] = ((841/108)*XYZr[pqr]+16.0/116.0)

    # calculate L*, a*, b*:
    Lab = xyz.astype(np.float)
    Lab[...,0] = 116.0*(fXYZr[...,1]) - 16.0
    Lab[pqr[...,1],0] = 903.3*XYZr[pqr[...,1],1]
    Lab[...,1] = 500.0*(fXYZr[...,0]-fXYZr[...,1])
    Lab[...,2] = 200.0*(fXYZr[...,1]-fXYZr[...,2])
    return Lab

def lab_to_xyz(lab, xyzw = None, cieobs = _CIEOBS, **kwargs):
    """
    Convert CIE 1976 L*a*b* (CIELAB) color coordinates to XYZ tristimulus values.

    Args:
        :lab: 
            | ndarray with CIE 1976 L*a*b* (CIELAB) color coordinates
        :xyzw:
            | None or ndarray with tristimulus values of white point, optional
            | None defaults to xyz of CIE D65 using the :cieobs: observer.
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set to use when calculating xyzw.

    Returns:
        :xyz: 
            | ndarray with tristimulus values
    """
    lab = np2d(lab)

    if xyzw is None:
        xyzw = spd_to_xyz(_CIE_ILLUMINANTS['D65'],cieobs = cieobs)

    # make xyzw same shape as data:
    xyzw = xyzw*np.ones(lab.shape)

    # set knee point of function:
    k=(24/116) #(24/116)**3**(1/3)

    # get L*, a*, b* and Xw, Yw, Zw:
    L,a,b = asplit(lab)
    Xw,Yw,Zw = asplit(xyzw)

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b/200.0

    # apply 3rd power:
    X,Y,Z = [xw*(x**3.0) for (x,xw) in ((fx,Xw),(fy,Yw),(fz,Zw))]

    # Now calculate T where T/Tn is below the knee point:
    p,q,r = [np.where(x<k) for x in (fx,fy,fz)]
    X[p],Y[q],Z[r] = [np.squeeze(xw[xp]*((x[xp] - 16.0/116.0) / (841/108))) for (x,xw,xp) in ((fx,Xw,p),(fy,Yw,q),(fz,Zw,r))]

    return ajoin((X,Y,Z))

def lab_to_xyz2(lab, xyzw = None, cieobs = _CIEOBS, **kwargs):
    """
    Convert CIE 1976 L*a*b* (CIELAB) color coordinates to XYZ tristimulus values.

    Args:
        :lab: 
            | ndarray with CIE 1976 L*a*b* (CIELAB) color coordinates
        :xyzw:
            | None or ndarray with tristimulus values of white point, optional
            | None defaults to xyz of CIE D65 using the :cieobs: observer.
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set to use when calculating xyzw.

    Returns:
        :xyz: 
            | ndarray with tristimulus values
    """
    lab = np2d(lab)

    if xyzw is None:
        xyzw = spd_to_xyz(_CIE_ILLUMINANTS['D65'],cieobs = cieobs)

    # make xyzw same shape as data:
    xyzw = xyzw*np.ones(lab.shape)

    # get L*, a*, b* and Xw, Yw, Zw:
    fXYZ = lab.astype(np.float)
    fXYZ[...,1] = (lab[...,0] + 16.0) / 116.0
    fXYZ[...,0] = lab[...,1] / 500.0 + fXYZ[...,1]
    fXYZ[...,2] = fXYZ[...,1] - lab[...,2]/200.0

    # apply 3rd power:
    xyz = (fXYZ**3.0)*xyzw

    # Now calculate T where T/Tn is below the knee point:
    pqr = fXYZ<=(24/116) #(24/116)**3**(1/3)
    xyz[pqr] = np.squeeze(xyzw[pqr]*((fXYZ[pqr] - 16.0/116.0) / (841/108)))

    return xyz


def xyz_to_Yuv(xyz,**kwargs):
    """
    Convert XYZ tristimulus values CIE 1976 Yu'v' chromaticity values.

    Args:
        :xyz: 
            | ndarray with tristimulus values

    Returns:
        :Yuv: 
            | ndarray with CIE 1976 Yu'v' chromaticity values
              (Y value refers to luminance or luminance factor)
    """
    xyz = np2d(xyz)

    X,Y,Z = asplit(xyz)
    denom = X + 15.0*Y + 3.0*Z
    u = 4.0*X / denom
    v = 9.0*Y / denom
    return ajoin((Y,u,v))

def xyz_to_Yuv2(xyz,**kwargs):
    """
    Convert XYZ tristimulus values CIE 1976 Yu'v' chromaticity values.

    Args:
        :xyz: 
            | ndarray with tristimulus values

    Returns:
        :Yuv: 
            | ndarray with CIE 1976 Yu'v' chromaticity values
              (Y value refers to luminance or luminance factor)
    """
    xyz = np2d(xyz)
    Yuv = xyz.astype(np.float)#*1.0 #*1.0 to ensure float and have a copy
    denom = xyz[...,0] + 15.0*xyz[...,1] + 3.0*xyz[...,2]
    Yuv[...,0] = xyz[...,1]
    Yuv[...,1] = 4.0*xyz[...,0] / denom
    Yuv[...,2] = 9.0*xyz[...,1] / denom
    return Yuv

def Yuv_to_xyz2(Yuv, **kwargs):
    """
    Convert CIE 1976 Yu'v' chromaticity values to XYZ tristimulus values.

    Args:
        :Yuv: 
            | ndarray with CIE 1976 Yu'v' chromaticity values
              (Y value refers to luminance or luminance factor)

    Returns:
        :xyz: 
            | ndarray with tristimulus values
    """
    Yuv = np2d(Yuv)
    xyz = Yuv.astype(np.float)
    xyz[...,1] = Yuv[...,0]
    xyz[...,0] = Yuv[...,0]*(9.0*Yuv[...,1])/(4.0*Yuv[...,2])
    xyz[...,2] = Yuv[...,0]*(12.0 - 3.0*Yuv[...,1] - 20.0*Yuv[...,2])/(4.0*Yuv[...,2])
    return xyz

def Yuv_to_xyz(Yuv, **kwargs):
    """
    Convert CIE 1976 Yu'v' chromaticity values to XYZ tristimulus values.

    Args:
        :Yuv: 
            | ndarray with CIE 1976 Yu'v' chromaticity values
              (Y value refers to luminance or luminance factor)

    Returns:
        :xyz: 
            | ndarray with tristimulus values
    """
    Yuv = np2d(Yuv)

    Y,u,v = asplit(Yuv)
    X = Y*(9.0*u)/(4.0*v)
    Z = Y*(12.0 - 3.0*u - 20.0*v)/(4.0*v)
    return ajoin((X,Y,Z))

def xyz_to_Ydlep(xyz, cieobs = _CIEOBS, xyzw = _COLORTF_DEFAULT_WHITE_POINT, flip_axes = False, **kwargs):
    """
    Convert XYZ tristimulus values to Y, dominant (complementary) wavelength
    and excitation purity.

    Args:
        :xyz:
            | ndarray with tristimulus values
        :xyzw:
            | None or ndarray with tristimulus values of a single (!) native white point, optional
            | None defaults to xyz of CIE D65 using the :cieobs: observer.
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set to use when calculating spectrum locus coordinates.
        :flip_axes:
            | False, optional
            | If True: flip axis 0 and axis 1 in Ydelep to increase speed of loop in function.
            |          (single xyzw with is not flipped!)
    Returns:
        :Ydlep: 
            | ndarray with Y, dominant (complementary) wavelength
              and excitation purity
    """
    
    xyz3 = np3d(xyz).copy().astype(np.float)

    # flip axis so that shortest dim is on axis0 (save time in looping):
    if (xyz3.shape[0] < xyz3.shape[1]) & (flip_axes == True):
        axes12flipped = True
        xyz3 = xyz3.transpose((1,0,2))
    else:
        axes12flipped = False

    # convert xyz to Yxy:
    Yxy = xyz_to_Yxy(xyz3)
    Yxyw = xyz_to_Yxy(xyzw)

    # get spectrum locus Y,x,y and wavelengths:
    SL = _CMF[cieobs]['bar']

    wlsl = SL[0]
    Yxysl = xyz_to_Yxy(SL[1:4].T)[:,None]

    # center on xyzw:
    Yxy = Yxy - Yxyw
    Yxysl = Yxysl - Yxyw
    Yxyw = Yxyw - Yxyw

    #split:
    Y, x, y = asplit(Yxy)
    Yw,xw,yw = asplit(Yxyw)
    Ysl,xsl,ysl = asplit(Yxysl)

    # calculate hue:
    h = math.positive_arctan(x,y, htype = 'deg')

    hsl = math.positive_arctan(xsl,ysl, htype = 'deg')

    hsl_max = hsl[0] # max hue angle at min wavelength
    hsl_min = hsl[-1] # min hue angle at max wavelength

    dominantwavelength = np.zeros(Y.shape)
    purity = dominantwavelength.copy()
    for i in range(xyz3.shape[1]):

            # find index of complementary wavelengths/hues:
            pc = np.where((h[:,i] >= hsl_max) & (h[:,i] <= hsl_min + 360.0)) # hue's requiring complementary wavelength (purple line)
            h[:,i][pc] = h[:,i][pc] - np.sign(h[:,i][pc] - 180.0)*180.0 # add/subtract 180° to get positive complementary wavelength

            # find 2 closest hues in sl:
            #hslb,hib = meshblock(hsl,h[:,i:i+1])
            hib,hslb = np.meshgrid(h[:,i:i+1],hsl)
            dh = np.abs(hslb-hib)
            q1 = dh.argmin(axis=0) # index of closest hue
            dh[q1] = 1000.0
            q2 = dh.argmin(axis=0) # index of second closest hue

            dominantwavelength[:,i] = wlsl[q1] + np.divide(np.multiply((wlsl[q2] - wlsl[q1]),(h[:,i] - hsl[q1,0])),(hsl[q2,0] - hsl[q1,0])) # calculate wl corresponding to h: y = y1 + (y2-y1)*(x-x1)/(x2-x1)
            dominantwavelength[:,i][pc] = - dominantwavelength[:,i][pc] #complementary wavelengths are specified by '-' sign

            # calculate excitation purity:
            x_dom_wl = xsl[q1,0] + (xsl[q2,0] - xsl[q1,0])*(h[:,i] - hsl[q1,0])/(hsl[q2,0] - hsl[q1,0]) # calculate x of dom. wl
            y_dom_wl = ysl[q1,0] + (ysl[q2,0] - ysl[q1,0])*(h[:,i] - hsl[q1,0])/(hsl[q2,0] - hsl[q1,0]) # calculate y of dom. wl
            d_wl = (x_dom_wl**2.0 + y_dom_wl**2.0)**0.5 # distance from white point to sl
            d = (x[:,i]**2.0 + y[:,i]**2.0)**0.5 # distance from white point to test point
            purity[:,i] = d/d_wl

            # correct for those test points that have a complementary wavelength
            # calculate intersection of line through white point and test point and purple line:
            xy = np.vstack((x[:,i],y[:,i])).T
            xyw = np.hstack((xw,yw))
            xypl1 = np.hstack((xsl[0,None],ysl[0,None]))
            xypl2 = np.hstack((xsl[-1,None],ysl[-1,None]))
            da = (xy-xyw)
            db = (xypl2-xypl1)
            dp = (xyw - xypl1)
            T = np.array([[0.0, -1.0], [1.0, 0.0]])
            dap = np.dot(da,T)
            denom = np.sum(dap * db,axis=1,keepdims=True)
            num = np.sum(dap * dp,axis=1,keepdims=True)
            xy_linecross = (num/denom) *db + xypl1
            d_linecross = np.atleast_2d((xy_linecross[:,0]**2.0 + xy_linecross[:,1]**2.0)**0.5).T#[0]
            purity[:,i][pc] = d[pc]/d_linecross[pc][:,0]
    Ydlep = np.dstack((xyz3[:,:,1],dominantwavelength,purity))

    if axes12flipped == True:
        Ydlep = Ydlep.transpose((1,0,2))
    else:
        Ydlep = Ydlep.transpose((0,1,2))
    return Ydlep.reshape(xyz.shape)



def Ydlep_to_xyz(Ydlep, cieobs = _CIEOBS, xyzw = _COLORTF_DEFAULT_WHITE_POINT, flip_axes = False, **kwargs):
    """
    Convert Y, dominant (complementary) wavelength and excitation purity to XYZ
    tristimulus values.

    Args:
        :Ydlep: 
            | ndarray with Y, dominant (complementary) wavelength
              and excitation purity
        :xyzw: 
            | None or narray with tristimulus values of a single (!) native white point, optional
            | None defaults to xyz of CIE D65 using the :cieobs: observer.
        :cieobs:
            | luxpy._CIEOBS, optional
            | CMF set to use when calculating spectrum locus coordinates.
        :flip_axes:
            | False, optional
            | If True: flip axis 0 and axis 1 in Ydelep to increase speed of loop in function.
            |          (single xyzw with is not flipped!)
    Returns:
        :xyz: 
            | ndarray with tristimulus values
    """

    Ydlep3 = np3d(Ydlep).copy().astype(np.float)

    # flip axis so that longest dim is on first axis  (save time in looping):
    if (Ydlep3.shape[0] < Ydlep3.shape[1]) & (flip_axes == True):
        axes12flipped = True
        Ydlep3 = Ydlep3.transpose((1,0,2))
    else:
        axes12flipped = False

    # convert xyzw to Yxyw:
    Yxyw = xyz_to_Yxy(xyzw)
    Yxywo = Yxyw.copy()

    # get spectrum locus Y,x,y and wavelengths:
    SL = _CMF[cieobs]['bar']
    wlsl = SL[0,None].T
    Yxysl = xyz_to_Yxy(SL[1:4].T)[:,None]

    # center on xyzw:
    Yxysl = Yxysl - Yxyw
    Yxyw = Yxyw - Yxyw

    #split:
    Y, dom, pur = asplit(Ydlep3)
    Yw,xw,yw = asplit(Yxyw)
    Ywo,xwo,ywo = asplit(Yxywo)
    Ysl,xsl,ysl = asplit(Yxysl)

    # loop over longest dim:
    x = np.zeros(Y.shape)
    y = x.copy()
    for i in range(Ydlep3.shape[1]):

        # find closest wl's to dom:
        #wlslb,wlib = meshblock(wlsl,np.abs(dom[i,:])) #abs because dom<0--> complemtary wl
        wlib,wlslb = np.meshgrid(np.abs(dom[:,i]),wlsl)

        dwl = np.abs(wlslb-wlib)
        q1 = dwl.argmin(axis=0) # index of closest wl
        dwl[q1] = 10000.0
        q2 = dwl.argmin(axis=0) # index of second closest wl

        # calculate x,y of dom:
        x_dom_wl = xsl[q1,0] + (xsl[q2,0] - xsl[q1,0])*(np.abs(dom[:,i]) - wlsl[q1,0])/(wlsl[q2,0] - wlsl[q1,0]) # calculate x of dom. wl
        y_dom_wl = ysl[q1,0] + (ysl[q2,0] - ysl[q1,0])*(np.abs(dom[:,i]) - wlsl[q1,0])/(wlsl[q2,0] - wlsl[q1,0]) # calculate y of dom. wl

        # calculate x,y of test:
        d_wl = (x_dom_wl**2.0 + y_dom_wl**2.0)**0.5 # distance from white point to dom
        d = pur[:,i]*d_wl
        hdom = math.positive_arctan(x_dom_wl,y_dom_wl,htype = 'deg')
        x[:,i] = d*np.cos(hdom*np.pi/180.0)
        y[:,i] = d*np.sin(hdom*np.pi/180.0)

        # complementary:
        pc = np.where(dom[:,i] < 0.0)
        hdom[pc] = hdom[pc] - np.sign(dom[:,i][pc] - 180.0)*180.0 # get positive hue angle

        # calculate intersection of line through white point and test point and purple line:
        xy = np.vstack((x_dom_wl,y_dom_wl)).T
        xyw = np.vstack((xw,yw)).T
        xypl1 = np.vstack((xsl[0,None],ysl[0,None])).T
        xypl2 = np.vstack((xsl[-1,None],ysl[-1,None])).T
        da = (xy-xyw)
        db = (xypl2-xypl1)
        dp = (xyw - xypl1)
        T = np.array([[0.0, -1.0], [1.0, 0.0]])
        dap = np.dot(da,T)
        denom = np.sum(dap * db,axis=1,keepdims=True)
        num = np.sum(dap * dp,axis=1,keepdims=True)
        xy_linecross = (num/denom) *db + xypl1
        d_linecross = np.atleast_2d((xy_linecross[:,0]**2.0 + xy_linecross[:,1]**2.0)**0.5).T[:,0]
        x[:,i][pc] = pur[:,i][pc]*d_linecross[pc]*np.cos(hdom[pc]*np.pi/180)
        y[:,i][pc] = pur[:,i][pc]*d_linecross[pc]*np.sin(hdom[pc]*np.pi/180)
    Yxy = np.dstack((Ydlep3[:,:,0],x + xwo, y + ywo))
    if axes12flipped == True:
        Yxy = Yxy.transpose((1,0,2))
    else:
        Yxy = Yxy.transpose((0,1,2))
    return Yxy_to_xyz(Yxy).reshape(Ydlep.shape)

