# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 22:48:09 2017

@author: kevin.smet
"""

# -*- coding: utf-8 -*-

###################################################################################################
# functions related to basic colorimetry
###################################################################################################
from luxpy import *
#from luxpy.chromaticadaptation import normalize_mcat

__all__ = ['_cspace_axes', '_ipt_M','xyz_to_Yxy','Yxy_to_xyz','xyz_to_Yuv','Yuv_to_xyz',
           'xyz_to_wuv','wuv_to_xyz','xyz_to_xyz','xyz_to_lab','lab_to_xyz','xyz_to_luv','luv_to_xyz',
           'xyz_to_Vrb_mb','Vrb_mb_to_xyz','xyz_to_ipt','ipt_to_xyz','xyz_to_Ydlep','Ydlep_to_xyz']


#------------------------------------------------------------------------------
# COLORIMETRIC functions:
# Chromaticity / colorspace functions:
#   xyz_to_Yxy(): self-explanatory
#   Yxy_to_xyz():      "
#   Yxy_to_Yuv():      "
#   Yuv_to_Yxy():      "
#   xyz_to_Yuv():      "
#   Yuv_to_xyz():      "
#	   xyz_to_xyz():	   "
#	   xyz_to_lab():	   "
#	   lab_to_xyz():	   "
#	   xyz_to_luv():	   "
#	   luv_to_xyz():	   "
#   xyz_to_Vrb_mb():   convert xyz to macleod boyton type coordinates (r,b) = (l,s)
#   Vrb_mb_to_xyz():   convert macleod boyton type coordinates (r,b) = (l,s) to xyz
#   xyz_to_ipt():   self-explanatory
#   ipt_to_xyz():  
#   xyz_to_Ydlep(): convert xyz to Y, dominant wavelength (dl) and excitation purity (ep)
#   Ydlep_to_xyz(): convert Y, dominant wavelength (dl) and excitation purity (ep) to xyz
#
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Database with cspace-axis strings (for plotting):
_cspace_axes = {'Yxy': ['Y / L (cd/m²)', 'x', 'y']}
_cspace_axes['Yuv'] = ['Y / L (cd/m²)', "u'", "v'"]
_cspace_axes['xyz'] = ['X', 'Y', 'Z']
_cspace_axes['lab'] = ['L*', "a*", "b*"]
_cspace_axes['luv'] = ['L*', "u*", "u*"]
_cspace_axes['ipt'] = ['I', "P", "T"]
_cspace_axes['wuv'] = ['W*', "U*", "V*"]
_cspace_axes['Vrb_mb'] = ['V (Macleod-Boyton)', "r (Macleod-Boyton)", "b (Macleod-Boyton)"]
_cspace_axes['cct'] = ['', 'cct','duv']


# pre-calculate matrices for conversion of xyz to lms and back for use in xyz_to_ipt() and ipt_to_xyz(): 
_ipt_M = {'lms2ipt': np.array([[0.4000,0.4000,0.2000],[4.4550,-4.8510,0.3960],[0.8056,0.3572,-1.1628]]),
                              'xyz2lms' : {x : normalize_3x3_matrix(_cmf['M'][x],spd_to_xyz(_cie_illuminants['D65'],cieobs = x)) for x in sorted(_cmf['M'].keys())}}
_colortf_default_white_point = np.array([100.0, 100.0, 100.0]) # ill. E white point

#------------------------------------------------------------------------------
#---chromaticity coordinates---------------------------------------------------
#------------------------------------------------------------------------------
def xyz_to_Yxy(data):
    """ 
	 Convert data = np.array([[x,y,z]]) tristimulus values to np.array([[Y,x,y]]).
	 """
    data = np2d(data)
#    sumxyz = data[...,0] + data[...,1] + data[...,2]
#    return ajoin((data[...,1],data[...,0]/sumxyz,data[...,1]/sumxyz))
    X,Y,Z = asplit(data) 
    sumxyz = X + Y + Z 
    x = X / sumxyz
    y = Y / sumxyz
    return ajoin((Y,x,y))
   

def Yxy_to_xyz(data):
    """ 
	 Convert data = np.array([[Y,x,y]]) chromaticity coordinates 
	 to np.array([[x,y,z]]) tristimulus values.
	 """
    data = np2d(data)
    
    Y,x,y = asplit(data) 
    X = Y*x/y
    Z = Y*(1.0-x-y)/y
    return ajoin((X,Y,Z))

def xyz_to_Yuv(data):
    """ 
	 Convert data = np.array([[x,y,z]]) tristimulus values to CIE 1976 np.array([[Y,u',v']]).
	 """
    data = np2d(data)
    
    X,Y,Z = asplit(data)
    denom = X + 15.0*Y + 3.0*Z
    u = 4.0*X / denom
    v = 9.0*Y / denom
    return ajoin((Y,u,v))

def Yuv_to_xyz(data):
    """ 
	 Convert data = np.array([[Y,u',v']]) CIE 1976 chromaticity coordinates 
	 to np.array([[x,y,z]]) tristimulus values.
	 """	
    data = np2d(data)
    
    Y,u,v = asplit(data)
    X = Y*(9.0*u)/(4.0*v)
    Z = Y*(12.0 - 3.0*u - 20.0*v)/(4.0*v)
    return ajoin((X,Y,Z))


def xyz_to_wuv(data, xyzw = _colortf_default_white_point):
    """ 
	 Convert data = np.array([[x,y,z]]) tristimulus values to CIE 1964 np.array([[W*,U*,V*]]).
	 """
    data = np2d(data)
    xyzw = np2d(xyzw)
    Yuv = xyz_to_Yuv(data) # convert to cie 1976 u'v'
    Yuvw = xyz_to_Yuv(xyzw)
    Y, u, v = asplit(Yuv)
    Yw, uw, vw = asplit(Yuvw)
    v = (2.0/3.0)*v # convert to cie 1960 u, v
    vw = (2.0/3.0)*vw # convert to cie 1960 u, v
    W = 25.0*(Y**(1/3)) - 17.0
    U = 13.0*W*(u - uw)
    V = 13.0*W*(v - vw)
    return ajoin((W,U,V))    

def wuv_to_xyz(data,xyzw = _colortf_default_white_point):
    """ 
	 Convert data = np.array([[W*,U*,V*]]) CIE 1964 cooridnates to np.array([[x,y,z]]) tristimulus values.
	 """
    data = np2d(data)
    xyzw = np2d(xyzw)
    
    Yuvw = xyz_to_Yuv(xyzw) # convert to cie 1976 u'v'
    Yw, uw, vw = asplit(Yuvw)
    vw = (2.0/3.0)*vw # convert to cie 1960 u, v
    W,U,V = asplit(data)
    Y = ((W + 17.0) / 25.0)**3.0
    u = uw + U/(13.0*W) 
    v = (vw + V/(13.0*W)) * (3.0/2.0)
    Yuv = ajoin((Y,u,v)) # = 1976 u',v'
    return Yuv_to_xyz(Yuv)
    
 
def xyz_to_xyz(data):
    """
	 Convert np.array([[x,y,z]]) tristimulus values to itself: dummy function for colortf().
	 """
    data = np2d(data)
    
    return data 


def xyz_to_lab(data,xyzw = None, cieobs = _cieobs):
    """ 
    Convert data = np.array([[x,y,z]]) tristimulus values to CIELab np.array([[L*,a*,b*]]).
    Default white point xyzw is D65 using CIE _cieobs observer. 
    """
    data = np2d(data)
       
    if xyzw is None:
        xyzw = spd_to_xyz(_cie_illuminants['D65'],cieobs = cieobs)
    
    # get and normalize (X,Y,Z) to white point:
    XYZr = data/xyzw
       
    # Apply cube-root compression:
    fXYZr = XYZr**(1.0/3.0)
    
    # Check for T/Tn <= 0.008856:
    pqr = XYZr<=0.008856

    # calculate f(T) for T/Tn <= 0.008856:
    fXYZr[pqr] = (7.787*XYZr[pqr]+16.0/116.0)
    
    # calculate L*, a*, b*:
    L = 116.0*(fXYZr[...,1]) - 16.0
    L[pqr[...,1]] = 903.3*XYZr[pqr[...,1],1]
    a = 500.0*(fXYZr[...,0]-fXYZr[...,1])
    b = 200.0*(fXYZr[...,1]-fXYZr[...,2])
    return ajoin((L,a,b))

#def xyz_to_lab2(data,xyzw = None, cieobs = _cieobs):
#    """ 
#    Convert data = np.array([[x,y,z]]) tristimulus values to CIELab np.array([[L*,a*,b*]]).
#    Default white point xyzw is D65 using CIE _cieobs observer. 
#    """
#    data = np2d(data)
#       
#    if xyzw is None:
#        xyzw = spd_to_xyz(_cie_illuminants['D65'],cieobs = cieobs)
#    
#    # get and normalize (X,Y,Z) to white point:
#    Xr,Yr,Zr = asplit(data/xyzw)
#       
#    # Apply cube-root compression:
#    fX,fY,fZ = [x**(1.0/3.0) for x in (Xr,Yr,Zr)]
#    
#    # Check for T/Tn <= 0.008856:
#    p,q,r = [np.where(x<=0.008856) for x in (Xr,Yr,Zr)]
#
#    # calculate f(T) for T/Tn <= 0.008856:
#    fX[p], fY[q] ,fZ[r] = [(7.787*x+16.0/116.0) for x in (Xr[p],Yr[q],Zr[r])]  
#
#    # calculate L*, a*, b*:
#    L = 116.0*(Yr**(1.0/3.0)) - 16.0
#    L[q] = 903.3*Yr[q]
#    a = 500.0*(fX-fY)
#    b = 200.0*(fY-fZ)
#    return ajoin((L,a,b))



def lab_to_xyz(data,xyzw = None, cieobs = _cieobs):
    """ 
    Convert data = CIELab np.array([[L*,a*,b*]]) to np.array([[x,y,z]]) tristimulus values.
    Default white point xyzw is D65 using CIE _cieobs observer. 
    """
    data = np2d(data)
         
    if xyzw is None:
        xyzw = spd_to_xyz(_cie_illuminants['D65'],cieobs = cieobs)
    
    # make xyzw same shape as data:
    #xyzw = todim(xyzw, data.shape, equal_shape = True)
    xyzw = xyzw*np.ones(data.shape)
    
    # set knee point of function:
    k=0.008856**(1/3)
    
    # get L*, a*, b* and Xw, Yw, Zw:
    L,a,b = asplit(data)
    Xw,Yw,Zw = asplit(xyzw)
   
    fy = (L + 16.0) / 116.0 
    fx = a / 500.0 + fy
    fz = fy - b/200.0

    # apply 3rd power:
    X,Y,Z = [xw*(x**3.0) for (x,xw) in ((fx,Xw),(fy,Yw),(fz,Zw))]

    # Now calculate T where T/Tn is below the knee point:
    p,q,r = [np.where(x<k) for x in (fx,fy,fz)]   
    X[p],Y[q],Z[r] = [np.squeeze(xw[xp]*((x[xp] - 16.0/116.0) / 7.787)) for (x,xw,xp) in ((fx,Xw,p),(fy,Yw,q),(fz,Zw,r))]
 
    return ajoin((X,Y,Z))  

def xyz_to_luv(data,xyzw = None, cieobs = _cieobs):
    """ 
    Convert data = np.array([[x,y,z]]) tristimulus values to CIELuv np.array([[L*,u*,v*]]).
    Default white point xyzw is D65 using CIE _cieobs observer. 
    """
    data = np2d(data)
    
    if xyzw is None:
        xyzw = spd_to_xyz(_cie_illuminants['D65'],cieobs = cieobs)
    
    # make xyzw same shape as data:
    xyzw = todim(xyzw, data.shape)
    
    # Calculate u',v' of test and white:
    Y,u,v = asplit(xyz_to_Yuv(data))
    Yw,uw,vw = asplit(xyz_to_Yuv(xyzw))
    
    #uv1976 to CIELUV
    YdivYw = Y / Yw
    L = 116.0*YdivYw**(1.0/3.0) - 16.0
    p = np.where(YdivYw <= (6.0/29.0)**3.0)
    L[p] = ((29.0/3.0)**3.0)*YdivYw[p]
    u = 13.0*L*(u-uw)
    v = 13.0*L*(v-vw)

    return ajoin((L,u,v))

def luv_to_xyz(data,xyzw = None, cieobs = _cieobs):
    """ 
    Convert data = CIELuv np.array([[L*,u*,v*]]) to np.array([[x,y,z]]) tristimulus values.
    Default white point xyzw is D65 using CIE _cieobs observer. 
    """
    data = np2d(data)
    
    
    if xyzw is None:
        xyzw = spd_to_xyz(_cie_illuminants['D65'],cieobs = cieobs)
    
    # Make xyzw same shape as data:
    Yuvw = todim(xyz_to_Yuv(xyzw), data.shape, equal_shape = True)
    
    # Get Yw, uw,vw:
    Yw,uw,vw = asplit(Yuvw)

    # calculate u'v' from u*,v*:
    L,u,v = asplit(data)
    up,vp = [(x / (13*L)) + xw for (x,xw) in ((u,uw),(v,vw))]
    up[np.where(L == 0.0)] = 0.0
    vp[np.where(L == 0.0)] = 0.0
    
    fy = (L + 16.0) / 116.0
    Y = Yw*(fy**3.0)
    p = np.where((Y/Yw) < ((6.0/29.0)**3.0))
    Y[p] = Yw[p]*(L[p]/((29.0/3.0)**3.0))

    return Yuv_to_xyz(ajoin((Y,up,vp)))
 
#-------------------------------------------------------------------------------------------------   
def xyz_to_Vrb_mb(data,cieobs = _cieobs, scaling = [1,1],M = None):
    """ 
    Convert data = np.array([[x,y,z]]) tristimulus values to Macleod-Boynton coordinates np.array([[V,r,b]]), 
    with V = R+G, r = R/V, b = B/V (note that R,G,B ~ L,M,S)
    If M is None: cieobs determines the conversion matrix M for going from XYZ to RGB (LMS).
    """
    data = np2d(data)
    
    X,Y,Z = asplit(data)
    if M is None:
        M = _cmf['M'][cieobs]
    R, G, B = [M[i,0]*X + M[i,1]*Y + M[i,2]*Z for i in range(3)]
    V = R + G
    r = R / V
    b = B / V
    return ajoin((V,r,b))

     
def Vrb_mb_to_xyz(data,cieobs = _cieobs, scaling = [1,1],M = None, Minverted = False):
    """ 
    Convert data = np.array([[V,r,b]]) Macleod-Boynton coordinates to np.array([[x,y,z]]) tristimulus values 
    Macleod Boynton: V = R+G, r = R/V, b = B/V (note that R,G,B ~ L,M,S)
    If M is None: cieobs determines the conversion matrix M for going from XYZ to RGB (LMS) (function does inversion).
    """
    data = np2d(data)
    
    V,r,b = asplit(data)
    R = r*V
    B = b*V
    G = V-R
    if M is None:
        M = _cmf['M'][cieobs]
    if Minverted == False:
        M = np.linalg.inv(M)
    X, Y, Z = [M[i,0]*R + M[i,1]*G + M[i,2]*B for i in range(3)]
    return ajoin((X,Y,Z))

def xyz_to_ipt(data,cieobs = _cieobs, xyz0 = None, Mxyz2lms = None):
    """
    Convert data = np.array([[x,y,z]]) relative (Ymax = 100) tristimulus values to I,P,T coordinates (I: lightness, P: red-green, T: yellow-blue).
    Note: data must be under D65 viewing conditions!! If necessary perform chromatic adaptation !!
    """
    data = np2d(data)
    
    # get M to convert xyz to lms and apply normalization to matrix or input your own:
    if Mxyz2lms is None:
        M = _ipt_M['xyz2lms'][cieobs] # matrix conversions from xyz to lms
        if xyz0 is None:
            xyz0 = spd_to_xyz(_cie_illuminants['D65'],cieobs = cieobs, out = 1)[0]/100.0
        else:
            xyz0 = xyz0/100.0    
        M = normalize_3x3_matrix(M,xyz0)
    else:
        M = Mxyz2lms

    # get xyz and normalize to 1:
    xyz = data/100.0
    
    # convert xyz to lms:
    if len(xyz.shape) == 3:
        lms = np.einsum('ij,klj->kli', M, xyz)
    else:
        lms = np.einsum('ij,lj->li', M, xyz)
    #lms = np.dot(M,xyz.T).T

    #response compression: lms to lms'
    lmsp = lms**0.43
    p = np.where(lms<0.0)
    lmsp[p] = -np.abs(lms[p])**0.43
    
    # convert lms' to ipt coordinates:
    if len(xyz.shape) == 3:
        ipt = np.einsum('ij,klj->kli', _ipt_M['lms2ipt'], lmsp)
    else:
        ipt = np.einsum('ij,lj->li', _ipt_M['lms2ipt'], lmsp)    
    #ipt = np.dot(_ipt_M['lms2ipt'],lmsp.T).T
    return ipt

def ipt_to_xyz(data,cieobs = _cieobs, xyz0 = None, Mxyz2lms = None):
    """
    Convert data = np.array([[I,P,T]]) coordinates (I: lightness, P: red-green, T: yellow-blue) to np.array([[x,y,z]]) relative (Ymax = 100) tristimulus values.
    Note: data is assumed to be under D65 viewing conditions!! If necessary perform chromatic adaptation !!
    """
    data = np2d(data)
    
    # get M to convert xyz to lms and apply normalization to matrix or input your own:
    if Mxyz2lms is None:
        M = _ipt_M['xyz2lms'][cieobs] # matrix conversions from xyz to lms
        if xyz0 is None:
            xyz0 = spd_to_xyz(_cie_illuminants['D65'],cieobs = cieobs, out = 1)[0]/100.0
        else:
            xyz0 = xyz0/100.0    
        M = normalize_3x3_matrix(M,xyz0)
    else:
        M = Mxyz2lms
    
    # convert from ipt to lms':
    if len(data.shape) == 3:
        lmsp = np.einsum('ij,klj->kli', np.linalg.inv(_ipt_M['lms2ipt']), data)
    else:
        lmsp = np.einsum('ij,lj->li', np.linalg.inv(_ipt_M['lms2ipt']), data)
    #lmsp = np.dot(np.linalg.inv(_ipt_M['lms2ipt']),data.T).T

    # reverse response compression: lms' to lms
    lms = lmsp**(1.0/0.43)
    p = np.where(lmsp<0.0)
    lms[p] = -np.abs(lmsp[p])**(1.0/0.43)

    # convert from lms to xyz:
    if len(data.shape) == 3:
        xyz = np.einsum('ij,klj->kli', np.linalg.inv(M), lms)
    else:
        xyz = np.einsum('ij,lj->li', np.linalg.inv(M), lms)    
    #xyz = np.dot(np.linalg.inv(M),lms.T).T
    xyz = xyz * 100.0
    xyz[np.where(xyz<0.0)] = 0.0
    
    return xyz
    
#------------------------------------------------------------------------------
def xyz_to_Ydlep(data, cieobs = _cieobs, xyzw = np2d([100.0,100.0,100.0])):
    """
    Calculates Y, dominant (complementary) wavelength and excitation purity.
    """
    
    data3 = np3d(data).copy()
    
       # flip axis so that shortest dim is on axis0 (save time in looping):
    if data3.shape[0] > data3.shape[1]:
        axes12flipped = True
        data3 = data3.transpose((1,0,2))
    else:
        axes12flipped = False
    
    # convert xyz to Yxy:
    Yxy = xyz_to_Yxy(data3)
    Yxyw = xyz_to_Yxy(xyzw)

    # get spectrum locus Y,x,y and wavelengths:
    SL = _cmf['bar'][cieobs]

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
    for i in range(data3.shape[1]):

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
            d_linecross = np.atleast_2d((xy_linecross[:,0]**2.0 + xy_linecross[:,1]**2.0)**0.5).T
            purity[:,i][pc] = d[pc]/d_linecross[pc]
    Ydlep = np.dstack((data3[:,:,1],dominantwavelength,purity))  
    
    if axes12flipped == True:
        Ydlep = Ydlep.transpose((1,0,2))
    else:
        Ydlep = Ydlep.transpose((0,1,2))
    return Ydlep.reshape(data.shape)

    

def Ydlep_to_xyz(data, cieobs = _cieobs, xyzw = np2d([100.0,100.0,100.0])):
    """
    Calculates xyz from Y, dominant (complementary) wavelength and excitation purity.
    """
    
    
    data3 = np3d(data).copy()
    
    # flip axis so that shortest dim is on axis0 (save time in looping):
    if data3.shape[2] > data3.shape[1]:
        axes12flipped = True
        data3 = data3.transpose((1,0,2))
    else:
        axes12flipped = False
    
    # convert xyzw to Yxyw:
    Yxyw = xyz_to_Yxy(xyzw)
    Yxywo = Yxyw.copy()

    # get spectrum locus Y,x,y and wavelengths:
    SL = _cmf['bar'][cieobs]
    wlsl = SL[0,None].T
    Yxysl = xyz_to_Yxy(SL[1:4].T)[:,None]
    
    # center on xyzw:
    Yxysl = Yxysl - Yxyw
    Yxyw = Yxyw - Yxyw
    
    #split:
    Y, dom, pur = asplit(data3)
    Yw,xw,yw = asplit(Yxyw)
    Ywo,xwo,ywo = asplit(Yxywo)
    Ysl,xsl,ysl = asplit(Yxysl)
    
    # loop over longest dim:
    x = np.zeros(Y.shape)
    y = x.copy()
    for i in range(data3.shape[1]):

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
        d_linecross = np.atleast_2d((xy_linecross[:,0]**2.0 + xy_linecross[:,1]**2.0)**0.5).T
        x[:,i][pc] = pur[:,i][pc]*d_linecross[pc]*np.cos(hdom[pc]*np.pi/180)
        y[:,i][pc] = pur[:,i][pc]*d_linecross[pc]*np.sin(hdom[pc]*np.pi/180)
    Yxy = np.dstack((data3[:,:,0],x + xwo, y + ywo))
    if axes12flipped == True:
        Yxy = Yxy.transpose((1,0,2))
    else:
        Yxy = Yxy.transpose((0,1,2))
    return Yxy_to_xyz(Yxy).reshape(data.shape)
    
