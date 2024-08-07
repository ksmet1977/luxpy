# -*- coding: utf-8 -*-
"""
ZCAM: Jzazbz-based color appearance model
=========================================

 :_UNIQUE_HUE_DATA: dictionary with unique hue data 
            
 :_SURROUND_PARAMETERS: dictionary with surround param. c, Nc, F and FLL for 'avg','dim' and 'dark' conditions

 :_NAKA_RUSHTON_PARAMETERS: | dictionary with parameters (n, sig, scaling and noise) 
                            | for the Naka-Rushton function: 
                            |   NK(x) = sign(x) * scaling * ((abs(x)**n) / ((abs(x)**n) + (sig**n))) + noise

 :_DEFAULT_WHITE_POINT: Default internal reference white point (xyz)

 :_DEFAULT_CONDITIONS: Default CAM model parameters 

 :_AXES: dict with list[str,str,str] containing axis labels of defined cspaces.

 :run(): Run the Jzazbz-based  color appearance model in forward or backward modes.
 
 :zcam(): Run the Jzazbz-based color appearance model in forward or backward modes.

 :specific_wrappers_in_the_'xyz_to_cspace()' and 'cpsace_to_xyz()' format:
      | 'xyz_to_jabM_zcam', 'jabM_zcam_to_xyz',
      | 'xyz_to_jabC_zcam', 'jabC_zcam_to_xyz',
      
References
----------
    1. `Safdar, M., Cui, G., Kim,Y. J., and  Luo, M. R. (2017).
    Perceptually uniform color space for image signals including high dynamic range and wide gamut.
    Opt. Express, vol. 25, no. 13, pp. 15131–15151, Jun. 2017. 
    <https://www.opticsexpress.org/abstract.cfm?URI=oe-25-13-15131>`_
    
    2. `Safdar, M., Hardeberg, J., Cui, G., Kim, Y. J., and Luo, M. R. (2018).
    A Colour Appearance Model based on Jzazbz Colour Space, 
    26th Color and Imaging Conference (2018), Vancouver, Canada, November 12-16, 2018, pp96-101.
    <https://doi.org/10.2352/ISSN.2169-2629.2018.26.96>`_

    3. Safdar, M., Hardeberg, J.Y., Luo, M.R. (2021) 
    "ZCAM, a psychophysical model for colour appearance prediction", 
    Optics Express. 29(4), 6036-6052, <https://doi.org/10.1364/OE.413659>`_
    
Created on Wed Sep 30 21:58:11 2020

@author: ksmet1977 at gmail.com
"""
import numpy as np

from luxpy import _CIEOBS, _CIE_D65, spd_to_xyz
from luxpy.utils import np2d, ajoin
from luxpy import cat
from luxpy.color.cam.utils import hue_angle, hue_quadrature

__all__ = ['run', 'zcam',
           '_AXES','_UNIQUE_HUE_DATA','_DEFAULT_WHITE_POINT',
           '_SURROUND_PARAMETERS']

__all__ += ['xyz_to_jabz', 'jabz_to_xyz',
            'xyz_to_jabM_zcam', 'jabM_zcam_to_xyz', 
            'xyz_to_jabC_zcam', 'jabC_zcam_to_xyz']

_UNIQUE_HUE_DATA = {'hues': 'red yellow green blue red'.split(), 
                    'i': [0,1,2,3,4], 
                    'hi':[33.44, 89.29, 146.30,238.36,393.44],
                    'ei':[0.68,0.64,1.52,0.77,0.68],
                    'Hi':[0.0,100.0,200.0,300.0,400.0]}

_SURROUND_PARAMETERS =  {'surrounds': ['avg', 'dim', 'dark'], 
                         'avg' : {'c':0.69, 'Nc':1.0, 'F':1.0,'FLL': 1.0}, 
                         'dim' : {'c':0.59, 'Nc':0.9, 'F':0.9,'FLL':1.0} ,
                         'dark' : {'c':0.525, 'Nc':0.8, 'F':0.8,'FLL':1.0}}

_PQ_PARAMETERS = {'c1':3424/2**12, 'c2': 2413/2**7, 'c3':2392/2**7,'n':2610/2**14,'p':1.7*2523/2**5}

# Define X',Y',Z' to L,M,S conversion matrix:
_M_XYZP_TO_LMS = np.array([[0.41478972, 0.579999, 0.0146480],
                           [-0.2015100, 1.120649, 0.0531008],
                           [-0.0166008, 0.264800, 0.6684799]])

# Define L',M',S' to Izab conversion matrix (original Ja,az,bz color space model):
_M_LMSP_TO_IAB_JABZ = np.array([[0.5, 0.5, 0],
                           [3.524000, -4.066708, 0.542708],
                           [0.199076, 1.096799, -1.295875]])

# Define L',M',S' to Izab conversion matrix (ZCAM model):
_M_LMSP_TO_IAB_ZCAM = np.array([[0, 1, 0],
                           [3.524000, -4.066708, 0.542708],
                           [0.199076, 1.096799, -1.295875]])
    
_DEFAULT_CONDITIONS = {'La': 100.0, 'Yb': 20.0, 'surround': 'avg','D': 1.0, 'Dtype': None}

_DEFAULT_WHITE_POINT = '_CIE_D65'

# Plotting ease:
_AXES = {'jabz' : ['Jz','az','bz']}
_AXES['jabM_zcam'] = ["Jz (zcam)", "azM (zcam)", "bzM (zcam)"]
_AXES['jabC_zcam'] = ["Jz (zcam)", "azC (zcam)", "bzC (zcam)"] 


def xyz_to_jabz(xyz, ztype = 'jabz', use_zcam_parameters = False, **kwargs):
    """
    Convert XYZ tristimulus values to Jz,az,bz color coordinates.

    Args:
        :xyz: 
            | ndarray with absolute tristimulus values (Y in cd/m²!)
        :ztype:
            | 'jabz', optional
            | String with requested return:
            | Options: 'jabz', 'iabz'
        :use_zcam_parameters:
            | False, optional
            | ZCAM uses a slightly different values (see notes)
    
    Returns:
        :jabz: 
            | ndarray with Jz (or Iz), az, bz color coordinates

    Notes:
     | 1. :xyz: is assumed to be under D65 viewing conditions! If necessary perform chromatic adaptation!
     |
     | 2a. Jz represents the 'lightness' relative to a D65 white with luminance = 10000 cd/m² 
     |      (note that Jz that not exactly equal 1 for this high value, but rather for 102900 cd/m2)
     | 2b. az, bz represent respectively a red-green and a yellow-blue opponent axis 
     |      (but note that a D65 shows a small offset from (0,0))
     | 3. ZCAM: calculates Iz as M' - epsilon (instead L'/2 + M'/2 as in Iz,az,bz color space!).

    Reference:
        1. `Safdar, M., Cui, G., Kim,Y. J., and Luo, M. R. (2017).
        Perceptually uniform color space for image signals including high dynamic range and wide gamut.
        Opt. Express, vol. 25, no. 13, pp. 15131–15151, June 2017. 
        <http://www.opticsexpress.org/abstract.cfm?URI=oe-25-13-15131>`_
        
        2. Safdar, M., Hardeberg, J.Y., Luo, M.R. (2021) 
        ZCAM, a psychophysical model for colour appearance prediction, 
        Optics Express. 29(4), 6036-6052, <https://doi.org/10.1364/OE.413659>`_
    """
    xyz = np2d(xyz)
    
    # Setup X,Y,Z to X',Y',Z' transform as matrix:
    b = 1.15 
    g = 0.66

    M_to_xyzp = np.array([[b, 0, 1 - b],[1 - g, g, 0],[0, 0, 1]])
    
    # Premultiply _M_XYZP_TO_LMS and M_to_lms:
    M =  _M_XYZP_TO_LMS @ M_to_xyzp
    
    # Transform X,Y,Z to L,M,S:
    if len(xyz.shape) == 3:
        lms = np.einsum('ij,klj->kli', M, xyz)
    else:
        lms = np.einsum('ij,lj->li', M, xyz)

    # response compression: lms to lms'
    c1, c2, c3, n, p = [_PQ_PARAMETERS[x] for x in sorted(_PQ_PARAMETERS.keys())]
    lmsp = ((c1 + c2*(lms/10000)**n)/(1 + c3*(lms/10000)**n))**p

    # Transform L',M',S' to Iabz:
    if use_zcam_parameters:
        epsilon = 3.7035226210190005e-11
        M = _M_LMSP_TO_IAB_ZCAM 
    else:
        epsilon = 0
        M = _M_LMSP_TO_IAB_JABZ

    if len(lms.shape) == 3:
        Iabz = np.einsum('ij,klj->kli', M, lmsp)
    else:
        Iabz = np.einsum('ij,lj->li', M, lmsp)

    Iabz[...,0]-=epsilon # correct to ensure zero output

    # convert Iabz' to Jabz coordinates:
    if ztype == 'jabz':
        Iabz[...,0] = ((1-0.56)*Iabz[...,0]/(1-0.56*Iabz[...,0])) - 1.6295499532821566e-11
    return Iabz

def jabz_to_xyz(jabz, ztype = 'jabz', use_zcam_parameters = False, **kwargs):
    """
    Convert Jz,az,bz color coordinates to XYZ tristimulus values.

    Args:
        :jabz: 
            | ndarray with Jz,az,bz color coordinates
        :ztype:
            | 'jabz', optional
            | String with requested return:
            | Options: 'jabz', 'iabz'
        :use_zcam_parameters:
            | False, optional
            | ZCAM uses a slightly different values (see notes)
            
    Returns:
        :xyz: 
            | ndarray with tristimulus values

    Note:
     | 1. :xyz: is assumed to be under D65 viewing conditions! If necessary perform chromatic adaptation!
     |
     | 2a. Jz represents the 'lightness' relative to a D65 white with luminance = 10000 cd/m² 
     |      (note that Jz that not exactly equal 1 for this high value, but rather for 102900 cd/m2)
     | 2b.  az, bz represent respectively a red-green and a yellow-blue opponent axis 
     |      (but note that a D65 shows a small offset from (0,0))
     | 3. ZCAM: calculates Iz as M' - epsilon (instead L'/2 + M'/2 as in Iz,az,bz color space!).

    Reference:
        1. `Safdar, M., Cui, G., Kim,Y. J., and Luo, M. R. (2017).
        Perceptually uniform color space for image signals including high dynamic range and wide gamut.
        Opt. Express, vol. 25, no. 13, pp. 15131–15151, June, 2017.
        <http://www.opticsexpress.org/abstract.cfm?URI=oe-25-13-15131>`_
        
        2. Safdar,  M., Hardeberg, J.Y., Luo, M.R. (2021) 
        ZCAM, a psychophysical model for colour appearance prediction, 
        Optics Express. 29(4), 6036-6052, <https://doi.org/10.1364/OE.413659>`_
    """
    jabz = np2d(jabz)
    
    # Convert Jz to Iz:
    if ztype == 'jabz':
        jabz[...,0] = (jabz[...,0] + 1.6295499532821566e-11)/(1 - 0.56*(1 - (jabz[...,0] + 1.6295499532821566e-11)))

    # Convert Iabz to lmsp:
    # Transform L',M',S' to Iabz:
    if use_zcam_parameters:
        epsilon = 3.70352262101900054e-11
        M = _M_LMSP_TO_IAB_ZCAM 
    else:
        epsilon = 0
        M = _M_LMSP_TO_IAB_JABZ
    jabz[...,0] += epsilon
    if len(jabz.shape) == 3:
        lmsp = np.einsum('ij,klj->kli', np.linalg.inv(M), jabz)
    else:
        lmsp = np.einsum('ij,lj->li', np.linalg.inv(M), jabz)
        
    # Convert lmsp to lms:
    c1, c2, c3, n, p = [_PQ_PARAMETERS[x] for x in sorted(_PQ_PARAMETERS.keys())]
    lms = 10000*((c1 - lmsp**(1/p)) / ((c3*lmsp**(1/p)) - c2))**(1/n)
    
    # Convert lms to xyz:
    # Setup X',Y',Z' from X,Y,Z transform as matrix:
    b = 1.15 
    g = 0.66 
    M_to_xyzp = np.array([[b, 0, 1 - b],[1 - g, g, 0],[0, 0, 1]])
    
    # Premultiply M_to_xyzp and M_to_lms and invert:
    M = np.linalg.inv(_M_XYZP_TO_LMS @ M_to_xyzp)
    
    # Transform L,M,S to X,Y,Z:
    if len(jabz.shape) == 3:
        xyz = np.einsum('ij,klj->kli', M, lms)
    else:
        xyz = np.einsum('ij,lj->li', M, lms)
        
    return xyz


def run(data, xyzw = None, outin = 'J,aM,bM', cieobs = _CIEOBS,
            conditions = None, forward = True, 
            mcat = 'cat02', apply_cat_to_whitepoint = False, **kwargs):
    """ 
    Run the Jz,az,bz based color appearance model in forward or backward modes.
    
    Args:
        :data:
            | ndarray with relative sample xyz values (forward mode) or J'a'b' coordinates (inverse mode)
        :xyzw:
            | ndarray with relative white point tristimulus values
            | None defaults to D65
        :cieobs:
            | _CIEOBS, optional
            | CMF set to use when calculating :xyzw: if this is None.
        :conditions:
            | None, optional
            | Dictionary with viewing condition parameters for:
            |       La, Yb, D and surround.
            |  surround can contain:
            |      - str (options: 'avg','dim','dark') or 
            |      - dict with keys c, Nc, F.
            | None results in:
            |   {'La':100, 'Yb':20, 'D':1, 'surround':'avg'}
        :forward:
            | True, optional
            | If True: run in CAM in forward mode, else: inverse mode.
        :outin:
            | 'J,aM,bM', optional
            | String with requested output (e.g. "J,aM,bM,M,h") [Forward mode]
            | - attributes: 'J': lightness,'Q': brightness,
            |               'M': colorfulness,'C': chroma, 's': saturation,
            |               'h': hue angle, 'H': hue quadrature/composition,
            |               'Wz': whiteness, 'Kz':blackness, 'Sz': saturation, 'V': vividness
            | String with inputs in data [inverse mode]. 
            | Input must have data.shape[-1]==3 and last dim of data must have 
            | the following structure for inverse mode: 
            |  * data[...,0] = J or Q,
            |  * data[...,1:] = (aM,bM) or (aC,bC) or (aS,bS) or (M,h) or (C, h), ...
        :mcat:
            | 'cat02', optional
            | Specifies CAT sensor space.
            | - options:
            |    - None defaults to 'cat02'
            |    - str: see see luxpy.cat._MCATS.keys() for options 
            |         (details on type, ?luxpy.cat)
            |    - ndarray: matrix with sensor primaries
        :apply_cat_to_whitepoint: 
            | False, optional
            | Apply a CAT to the white point.
            | However, ZCAM as published doesn't do this for some reason.
    Returns:
        :camout: 
            | ndarray with color appearance correlates (forward mode) 
            |  or 
            | XYZ tristimulus values (inverse mode)
     
    References:
        1. `Safdar, M., Cui, G., Kim,Y. J., and  Luo, M. R.(2017).
        Perceptually uniform color space for image signals including high dynamic range and wide gamut.
        Opt. Express, vol. 25, no. 13, pp. 15131–15151, Jun. 2017. 
        <https://www.opticsexpress.org/abstract.cfm?URI=oe-25-13-15131>`_
    
        2. `Safdar, M., Hardeberg, J., Cui, G., Kim, Y. J., and Luo, M. R.(2018).
        A Colour Appearance Model based on Jzazbz Colour Space, 
        26th Color and Imaging Conference (2018), Vancouver, Canada, November 12-16, 2018, pp96-101.
        <https://doi.org/10.2352/ISSN.2169-2629.2018.26.96>`_
        
        3. Safdar, M.,  Hardeberg, J.Y., Luo, M.R. (2021) 
        ZCAM, a psychophysical model for colour appearance prediction, 
        Optics Express. 29(4), 6036-6052, <https://doi.org/10.1364/OE.413659>`_
    """
    #print("WARNING: Z-CAM is as yet unpublished and under development, so parameter values might change! (07 Oct, 2020")
    outin = outin.split(',') if isinstance(outin,str) else outin
    
    #--------------------------------------------
    # Get condition parameters:
    if conditions is None:
        conditions = _DEFAULT_CONDITIONS

    D, Dtype, La, Yb, surround = (conditions[x] for x in sorted(conditions.keys()))
    
    surround_parameters =  _SURROUND_PARAMETERS
    if isinstance(surround, str):
        surround = surround_parameters[surround]
    F, FLL, Nc, c = [surround[x] for x in sorted(surround.keys())]
 
    # Define cone/chromatic adaptation sensor space:  
    if (mcat is None):
        mcat = cat._MCATS['cat02']
    elif isinstance(mcat,str):
        mcat = cat._MCATS[mcat]
    invmcat = np.linalg.inv(mcat)     
    
    #--------------------------------------------
    # Get white point of D65 fro chromatic adaptation transform (CAT)
    # xyzw_d65 = np.array([[9.5047e+01, 1.0000e+02, 1.08883e+02]]) if cieobs == '1931_2'  else  spd_to_xyz(_CIE_D65, cieobs = cieobs)
    xyzw_d65 = np.array([[95.0429, 100, 108.89]]) if cieobs == '1931_2'  else  spd_to_xyz(_CIE_D65, cieobs = cieobs)

   
    #--------------------------------------------
    # Get default white point:
    if xyzw is None:
        xyzw = xyzw_d65.copy()
 
    
    #--------------------------------------------
    # calculate condition dependent parameters:
    Yw = xyzw[...,1].T
    FL = 0.171 * La**(1/3) * (1 - np.exp(-48/9*La)) # luminance adaptation factor
    
    n = Yb/Yw 
    Fb = n**0.5 # background factor
    Fs = c # surround factor 
    #--------------------------------------------
    # Calculate degree of chromatic adaptation:
    if D is None:
        D = F*(1.0-(1.0/3.6)*np.exp((-La-42.0)/92.0))

    #===================================================================
    # WHITE POINT transformations (common to forward and inverse modes):   
  
    
    #--------------------------------------------
    # Apply CAT to white point:
    if apply_cat_to_whitepoint: 
        xyzwc = cat.apply_vonkries1(xyzw, xyzw1 = xyzw, xyzw2 = xyzw_d65, 
                                D = D, mcat = mcat, invmcat = invmcat,
                                use_Yw = True)
    else: 
        xyzwc = xyzw # use original unadapted white point in further calculations
    
    
    #--------------------------------------------
    # Get Iz,az,bz coordinates:
    iabzw = xyz_to_jabz(xyzwc, ztype = 'iabz', use_zcam_parameters = True)
   
    
    # Get brightness of white point:
    Qw = 2700 * (iabzw[...,0]**(1.6*Fs/(Fb**0.12))) * (Fs**2.2) * (Fb**0.5) * (FL**0.2)
    
    #===================================================================
    # STIMULUS transformations:
    
    #--------------------------------------------
    # massage shape of data for broadcasting:
    original_ndim = data.ndim
    if data.ndim == 2: data = data[:,None]
    
    if forward:
        # Apply CAT to D65:
        xyzc =  cat.apply_vonkries1(data, xyzw1 = xyzw, xyzw2 = xyzw_d65, 
                                    D = D, mcat = mcat, invmcat = invmcat,
                                    use_Yw = True)
 
        # Get Iz,az,bz coordinates:
        iabz = xyz_to_jabz(xyzc, ztype = 'iabz', use_zcam_parameters = True)
        Iz, az, bz = iabz[...,0],iabz[...,1], iabz[...,2]

        #--------------------------------------------
        # calculate hue h and eccentricity factor, et:
        h = hue_angle(iabz[...,1],iabz[...,2], htype = 'deg')
        ez = 1.015 + np.cos((h + 89.038)*np.pi/180)

        
        #-------------------------------------------- 
        # calculate Hue quadrature (if requested in 'out'):
        if 'H' in outin:    
            H = hue_quadrature(h, unique_hue_data = _UNIQUE_HUE_DATA)
        else:
            H = None
            
        #-------------------------------------------- 
        # calculate brightness, Q:
        Q = 2700 * (iabz[...,0]**(1.6*Fs/(Fb**0.12))) * (Fs**2.2) * (Fb**0.5) * (FL**0.2)
            
        #--------------------------------------------   
        # calculate lightness, J:
        J = 100.0 * (Q/Qw)
         
        #-------------------------------------------- 
        # calculate colorfulness, M:
        M = 100*((iabz[...,1]**2.0 + iabz[...,2]**2.0)**0.37)*(ez**0.068)*(FL**0.2) / ((iabzw[...,0]**0.78) * (Fb**0.1))

        #-------------------------------------------- 
        # calculate chroma, C:
        C = 100*M/Qw

        #--------------------------------------------         
        # calculate saturation, s:
        s = 100.0* ((M/Q)**0.5) * (FL)**0.6
        S = s # make extra variable, just in case 'S' is called
        
        
        #--------------------------------------------         
        # calculate whiteness, W:
        if ('Wz' in outin) | ('aWz' in outin):
            Wz = 100 - ((100-J)**2 + C**2)**0.5
        
        #--------------------------------------------         
        # calculate blackness, K:
        if ('Kz' in outin) | ('aKz' in outin):
            Kz = 100 - 0.8*(J**2 + 8*C**2)**0.5
            
        #--------------------------------------------         
        # calculate saturation, S:
        if ('Sz' in outin) | ('aSz' in outin):
            Sz = S
            
        #--------------------------------------------         
        # calculate vividness, V:
        if ('Vz' in outin) | ('aVz' in outin):
            Vz = ((J - 58)**2 + 3.4*C**2)**0.5
        
        
        
        #--------------------------------------------            
        # calculate cartesian coordinates:
        if ('aS' in outin):
             aS = s*np.cos(h*np.pi/180.0)
             bS = s*np.sin(h*np.pi/180.0)
        
        if ('aC' in outin):
             aC = C*np.cos(h*np.pi/180.0)
             bC = C*np.sin(h*np.pi/180.0)
             
        if ('aM' in outin):
             aM = M*np.cos(h*np.pi/180.0)
             bM = M*np.sin(h*np.pi/180.0)
             
        if ('aKz' in outin):
             aKz = Kz*np.cos(h*np.pi/180.0)
             bKz = Kz*np.sin(h*np.pi/180.0)
             
        if ('aVz' in outin):
             aVz = Vz*np.cos(h*np.pi/180.0)
             bVz = Vz*np.sin(h*np.pi/180.0)
             
        if ('aWz' in outin):
             aWz = Wz*np.cos(h*np.pi/180.0)
             bWz = Wz*np.sin(h*np.pi/180.0)
             
        if ('aSz' in outin):
             aSz = Sz*np.cos(h*np.pi/180.0)
             bSz = Sz*np.sin(h*np.pi/180.0)
             
         
        #-------------------------------------------- 
        if outin != ['J','aM','bM']:
            camout = eval('ajoin(('+','.join(outin)+'))')
        else:
            camout = ajoin((J,aM,bM))
        
        if (camout.shape[1] == 1) & (original_ndim < 3):
            camout = camout[:,0,:]

        return camout
    
    elif forward == False:
        #--------------------------------------------
        # Get Lightness J and brightness Q from data:
        if ('J' in outin[0]):
            J = data[...,0].copy()
            Q = Qw*(J/100)
        elif ('Q' in outin[0]):
            Q = data[...,0].copy()
            J = 100.0* (Q/Qw)
        else:
            raise Exception('No lightness or brightness values in data[...,0]. Inverse CAM-transform not possible!')
            
        #--------------------------------------------
        # calculate achromatic signal, Iz:
        Iz = (Q/(2700*(Fs**2.2)*(Fb**0.5)*(FL**0.2)))**((Fb**0.12)/(1.6*Fs))
            
        
        #-------------------------------------------- 
        # calculate Hue quadrature (if requested in 'out'):
        if 'H' in outin:    
            h = hue_quadrature(data[...,outin.index('H')], unique_hue_data = _UNIQUE_HUE_DATA, forward = False)
        
        
        #--------------------------------------------    
        if 'a' in outin[1]: 
            # calculate hue h:
            h = hue_angle(data[...,1],data[...,2], htype = 'deg')
        
            #--------------------------------------------
            # calculate Colorfulness M or Chroma C or Saturation s from a,b:
            MCs = (data[...,1]**2.0 + data[...,2]**2.0)**0.5  
        elif 'H' in outin:    
            h = hue_quadrature(data[...,outin.index('H')], unique_hue_data = _UNIQUE_HUE_DATA, forward = False)
            MCs = data[...,1] 
        elif 'h' in outin:
            h = data[...,2]
            MCs = data[...,1]  
        else:
            raise Exception('No (a,b) or hue angle or Hue quadrature data in input!')
        

        if ('aS' in outin) | ('S' in outin):
            M = Q * (MCs / FL**0.6 / 100)**2 
            C = 100*M/Qw
         
        if ('aM' in outin) | ('M' in outin): 
            C = 100*MCs/Qw
            
        if ('aC' in outin) | ('C' in outin): # convert C to M:
            C = MCs
            
        
        if ('Wz' in outin) | ('aWz' in outin): #whiteness
            C = (((100-MCs))**2 - (100 - J)**2)**0.5
        
        if ('Kz' in outin) | ('aKz' in outin): # blackness
            C = ((1/8)*(((100-MCs)/0.8)**2 - J**2))**0.5
            
        if ('Sz' in outin) | ('aSz' in outin):  # saturation
            C = (Q / Qw) * MCs**2 / FL**1.2 / 100 
            
        if ('Vz' in outin) | ('aVz' in outin):  # vividness
            C = ((MCs**2 - (J - 58)**2)/3.4)**0.5
            
                
        #--------------------------------------------
        # Calculate colorfulness, M:
        M = Qw * C / 100
        
        #--------------------------------------------    
        # calculate eccentricity factor, et:
        # ez = 1.014 + np.cos(h*np.pi/180 + 89.038)
        ez = 1.015 + np.cos((h + 89.038)*np.pi/180)
        
        #--------------------------------------------
        # calculate t (=sqrt(a**2+b**2)) from M:
        t = (((M/100) * (iabzw[...,0]**0.78) * (Fb**0.1))/((ez**0.068)*(FL**0.2)))**(1/0.37/2)

        #--------------------------------------------
        # Calculate az, bz:
        az = t*np.cos(h*np.pi/180)
        bz = t*np.sin(h*np.pi/180)
                  
        #--------------------------------------------
        # join values and convert to xyz:
        xyzc = jabz_to_xyz(ajoin((Iz,az,bz)), ztype = 'iabz', use_zcam_parameters = True)


        #-------------------------------------------
        # Apply CAT from D65:
        xyz =  cat.apply_vonkries1(xyzc, xyzw_d65, xyzw, D = D, 
                                   mcat = mcat, invmcat = invmcat,
                                   use_Yw = True)

        return xyz
    
        
#------------------------------------------------------------------------------
# wrapper functions for use with colortf():
#------------------------------------------------------------------------------
zcam = run
def xyz_to_jabM_zcam(data, xyzw = _DEFAULT_WHITE_POINT, cieobs = _CIEOBS,
                         conditions = None, mcat = 'cat02', apply_cat_to_whitepoint = False, **kwargs):
    """
    Wrapper function for zcam forward mode with J,aM,bM output.
    
    | For help on parameter details: ?luxpy.cam.zcam 
    """
    return zcam(data, xyzw = xyzw, cieobs = cieobs, conditions = conditions, forward = True, outin = 'J,aM,bM', mcat = mcat, apply_cat_to_whitepoint = apply_cat_to_whitepoint)
   

def jabM_zcam_to_xyz(data, xyzw = _DEFAULT_WHITE_POINT, cieobs = _CIEOBS,
                         conditions = None, mcat = 'cat02', apply_cat_to_whitepoint = False, **kwargs):
    """
    Wrapper function for zcam inverse mode with J,aM,bM input.
    
    | For help on parameter details: ?luxpy.cam.zcam 
    """
    return zcam(data, xyzw = xyzw, cieobs = cieobs, conditions = conditions, forward = False, outin = 'J,aM,bM', mcat = mcat, apply_cat_to_whitepoint = apply_cat_to_whitepoint)



def xyz_to_jabC_zcam(data, xyzw = _DEFAULT_WHITE_POINT, cieobs = _CIEOBS,
                         conditions = None, mcat = 'cat02', apply_cat_to_whitepoint = False, **kwargs):
    """
    Wrapper function for zcam forward mode with J,aC,bC output.
    
    | For help on parameter details: ?luxpy.cam.zcam 
    """
    return zcam(data, xyzw = xyzw, cieobs = cieobs, conditions = conditions, forward = True, outin = 'J,aC,bC', mcat = mcat, apply_cat_to_whitepoint = apply_cat_to_whitepoint)
 

def jabC_zcam_to_xyz(data, xyzw = _DEFAULT_WHITE_POINT, cieobs = _CIEOBS,
                         conditions = None, mcat = 'cat02', apply_cat_to_whitepoint = False, **kwargs):
    """
    Wrapper function for zcam inverse mode with J,aC,bC input.
    
    | For help on parameter details: ?luxpy.cam.zcam 
    """
    return zcam(data, xyzw = xyzw, cieobs = cieobs, conditions = conditions, forward = False, outin = 'J,aC,bC', mcat = mcat, apply_cat_to_whitepoint = apply_cat_to_whitepoint)
    
  
#==============================================================================  
if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    # Code test
    #--------------------------------------------------------------------------
    
    _cam = run
    
    import matplotlib.pyplot as plt 
    import numpy as np
    
    import luxpy as lx
    
    # Prepare some illuminant data:
    C = lx._CIE_ILLUMINANTS['C'].copy()
    Ill1 = C
    Ill2 = np.vstack((C,lx.cie_interp(lx._CIE_ILLUMINANTS['D65'],C[0],kind='spd')[1:],C[1:,:]*2,C[1:,:]*3))
    
    # Prepare some sample data:
    rflM = lx._MUNSELL['R'].copy()
    rflM = lx.cie_interp(rflM,C[0], kind='rfl')
    
    # Setup some model parameters:
    cieobs = '2006_10'
    Lw = 400
    
    # Create Lw normalized data:
    # Normalize to Lw:
    def normalize_to_Lw(Ill, Lw, cieobs, rflM):
        xyzw = lx.spd_to_xyz(Ill, cieobs = cieobs, relative = False)
        for i in range(Ill.shape[0]-1):
            Ill[i+1] = Lw*Ill[i+1]/xyzw[i,1]
        IllM = []
        for i in range(Ill.shape[0]-1):
            IllM.append(np.vstack((Ill1[0],Ill[i+1]*rflM[1:,:])))
        IllM = np.transpose(np.array(IllM),(1,0,2))
        return Ill, IllM
    Ill1, Ill1M = normalize_to_Lw(Ill1, Lw, cieobs, rflM)
    Ill2, Ill2M = normalize_to_Lw(Ill2, Lw, cieobs, rflM)
    
    n = 6
    xyz1, xyzw1 = lx.spd_to_xyz(Ill1, cieobs = cieobs, relative = True, rfl = rflM, out = 2)
    xyz1 = xyz1[:n,0,:]
    Ill1M = Ill1M[:(n+1),0,:]
    
    xyz2, xyzw2 = lx.spd_to_xyz(Ill2, cieobs = cieobs, relative = True, rfl = rflM, out = 2)
    xyz2 = xyz2[:n,:,:]
    Ill2M = Ill2M[:(n+1),:,:]
    
    # Module output plot:
    # _cam_o = lambda xyz, xyzw, forward: lx.xyz_to_jabz(xyz)
    xyz, xyzw = lx.spd_to_xyz(Ill1, cieobs = cieobs, relative = True, rfl = rflM, out = 2)
    jabch = _cam(xyz, xyzw = xyzw, forward = True, outin = 'J,aM,bM')
    # out_ = _cam_o(xyz, xyzw = xyzw, forward = True)
    plt.figure()
    plt.plot(jabch[...,1],jabch[...,2],'c.')
    # plt.plot(out_[...,1],out_[...,2],'r.')
    plt.axis('equal')


    out = 'J,aM,bM,M,C,s,h'.split(',')
    # Single data for sample and illuminant:
    # test input to _simple_cam():
    print('\n\n1: xyz in:')
    out_1 = _cam(xyz1, xyzw = xyzw1, forward = True, outin = out)
    xyz_1 = _cam(out_1[...,:3], xyzw = xyzw1, forward = False, outin = out[:3])
    print((xyz1 - xyz_1).sum())
    
    
    # Multiple data for sample and illuminants:
    print('\n\n2: xyz in:')
    out_2 = _cam(xyz2, xyzw = xyzw2, forward = True, outin = out)
    xyz_2 = _cam(out_2[...,:3], xyzw = xyzw2, forward = False, outin = out[:3])
    print((xyz2 - xyz_2).sum())
        
    
    # Single data for sample, multiple illuminants:
    print('\n\n3: xyz in:')
    out_3 = _cam(xyz1, xyzw = xyzw2, forward = True, outin = out)
    xyz_3 = _cam(out_3[...,:3], xyzw = xyzw2, forward = False, outin = out[:3])
    print((xyz1 - xyz_3[:,0,:]).sum())
    
    # test of origin of space:
    xyzw_d65 = lx.spd_to_xyz(lx._CIE_D65, cieobs='1931_2')
    # xyzw_d65 = np.array([[95.047, 100.0, 108.888]])
    out_d65 = run(xyzw_d65,xyzw=xyzw_d65,outin='h,Q,J,M,C,Sz,Vz,Kz,Wz',conditions={'La':264,'Yb':100,'D':1,'Dtype':None,'surround':'avg'},mcat='cat02')
    print('origin test', out_d65)
    print('Origin is not at az=bz=0 !!!')
    # user_conditions = {'D': 1, 'Dtype': None,\
    #                'La': 500.0, 'Yb': 20.0, 'surround': 'avg'}

    # jabM_zcam = lx.xyz_to_jabM_zcam(xyz[:,0,:], xyzw = xyzw)
    # jabC_zcam = lx.xyz_to_jabC_zcam(xyz[:,0,:], xyzw = xyzw)
    # print("JabM_zcam (default viewing conditions) = ", jabM_zcam)
    
    # jabM_zcam_user_vc = lx.xyz_to_jabM_zcam(xyz[:,0,:], xyzw = xyzw, conditions = user_conditions)
    # jabC_zcam_user_vc = lx.xyz_to_jabC_zcam(xyz[:,0,:], xyzw = xyzw, conditions = user_conditions)

    # print("JabM_zcam (user defined viewing conditions) = ", jabM_zcam_user_vc)    
        
    
    # fig, axs = plt.subplots(1,2,figsize=(12,6));

    # axs[0].plot(jabM_zcam[...,1:2],jabM_zcam[...,2:3],'b.', label = 'zcam')
    # axs[0].plot(jabM_zcam_user_vc[...,1:2],jabM_zcam_user_vc[...,2:3],'g.', label = 'zcam (user cond.)')
    # axs[0].set_xlabel('azM (zcam)')
    # axs[0].set_ylabel('bzM (zcam)')
    
    # axs[1].plot(jabC_zcam[...,1:2],jabC_zcam[...,2:3],'rx', label = 'zcam')
    # axs[1].plot(jabC_zcam_user_vc[...,1:2],jabC_zcam_user_vc[...,2:3],'c.', label = 'zcam (user cond.)')
    # axs[1].set_xlabel('azC (zcam)')
    # axs[1].set_ylabel('bzC (zcam)')
    
    
if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    # Code test using examples in Table of supplement to paper
    #--------------------------------------------------------------------------
    
   
    import luxpy as lx
    import matplotlib.pyplot as plt

    np.set_printoptions(formatter={'float_kind':"{:.6f}".format})

    # Test code with examples in Table of supplement to paper:
    xyzt =np.array([[185,206,163]])
    xyzw =np.array([[256,264,202]])
    out1 = run(xyzt,xyzw=xyzw,outin='h,H,Q,J,M,C,Sz,Vz,Kz,Wz',conditions={'La':264,'Yb':100,'D':None,'Dtype':'cat02','surround':'avg'}, mcat='cat02')
    print('\nout1', out1)
    expected_1 = np.array([[196.3524, 237.6401, 321.3464, 92.25, 10.53, 3.0216, 19.1314, 34.7022, 25.2994, 91.6837]]) # from supplementary material of Z-CAM paper
    print('expected1: ', expected_1)
    print('diff1: ', out1 - expected_1)
    
    xyzt =np.array([[89,96,120]])
    xyzw =np.array([[256,264,202]])
    out2 = run(xyzt,xyzw=xyzw,outin='h,Q,J,aM,bM,M,C,Sz,Vz,Kz,Wz',conditions={'La':264,'Yb':100,'D':None,'Dtype':None,'surround':'avg'},mcat='cat02')
    print('\nout2', out2)
    
    print('\n !!!! REMARK !!!!')
    print('Output does not match perfectly with example in supplementary material of paper Safdar et al (2021).')
    print('Differences start with output for jz,az,bz; especially b seems to be a bit more off.')
    print('Parameters in model are as in paper, so differences are likely caused by CAT, in particular the exact whitepoint of D65')
    print("Using close but off values for the xyz of D65 gets the results closer.")

    #print(run(run(np.array([[50, 10, 100]]), outin = "J,C,H", forward = True), outin = "J,C,H", forward = False))
  