# -*- coding: utf-8 -*-
"""
Jzazbz-based color appearance model
===================================

 :_UNIQUE_HUE_DATA: dictionary with unique hue data 
            
 :_SURROUND_PARAMETERS: dictionary with surround param. c, Nc, F and FLL for 'avg','dim' and 'dark' conditions

 :_NAKA_RUSHTON_PARAMETERS: | dictionary with parameters (n, sig, scaling and noise) 
                            | for the Naka-Rushton function: 
                            |   NK(x) = sign(x) * scaling * ((abs(x)**n) / ((abs(x)**n) + (sig**n))) + noise

 :_DEFAULT_WHITE_POINT: Default internal reference white point (xyz)

 :_DEFAULT_CONDITIONS: Default CAM model parameters 

 :_AXES: dict with list[str,str,str] containing axis labels of defined cspaces.

 :run(): Run the Jzazbz-based  color appearance model in forward or backward modes.
 
 :camjabz(): Run the Jzazbz-based color appearance model in forward or backward modes.

 :specific_wrappers_in_the_'xyz_to_cspace()' and 'cpsace_to_xyz()' format:
      | 'xyz_to_jabM_camjabz', 'jabM_camjabz_to_xyz',
      | 'xyz_to_jabC_camjabz', 'jabC_camjabz_to_xyz',
      
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


Created on Wed Sep 30 21:58:11 2020

@author: ksmet1977 at gmail.com
"""


from luxpy import math, _CIEOBS, _CIE_D65, spd_to_xyz
from luxpy.utils import np, np2d, asplit, ajoin
from luxpy import cat
from luxpy.color.cam.utils import hue_angle, hue_quadrature

__all__ = ['run', 'camjabz',
           '_AXES','_UNIQUE_HUE_DATA','_DEFAULT_WHITE_POINT',
           '_SURROUND_PARAMETERS']

__all__ += ['xyz_to_jabz', 'jabz_to_xyz',
            'xyz_to_jabM_camjabz', 'jabM_camjabz_to_xyz', 
            'xyz_to_jabC_camjabz', 'jabC_camjabz_to_xyz']

_UNIQUE_HUE_DATA = {'hues': 'red yellow green blue red'.split(), 
                    'i': [0,1,2,3,4], 
                    'hi':[33.34, 89.29, 146.30,238.36,393.44],
                    'ei':[0.68,0.64,1.52,0.77,0.68],
                    'Hi':[0.0,100.0,200.0,300.0,400.0]}

_SURROUND_PARAMETERS =  {'surrounds': ['avg', 'dim', 'dark'], 
                         'avg' : {'c':0.69, 'Nc':1.0, 'F':1.0,'FLL': 1.0}, 
                         'dim' : {'c':0.59, 'Nc':0.9, 'F':0.9,'FLL':1.0} ,
                         'dark' : {'c':0.525, 'Nc':0.8, 'F':0.8,'FLL':1.0}}


_DEFAULT_CONDITIONS = {'La': 100.0, 'Yb': 20.0, 'surround': 'avg','D': 1.0, 'Dtype': None}

_DEFAULT_WHITE_POINT = '_CIE_D65'

# Plotting ease:
_AXES = {'jabz' : ['Jz','az','bz']}
_AXES['jabM_camjabz'] = ["Jz (camjabz)", "azM (camjabz)", "bzM (camjabz)"]
_AXES['jabC_camjabz'] = ["Jz (camjabz)", "azC (camjabz)", "bzC (camjabz)"] 


def xyz_to_jabz(xyz, ztype = 'jabz', **kwargs):
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
            | ndarray with Jz (or Iz), az, bz color coordinates

    Notes:
     | 1. :xyz: is assumed to be under D65 viewing conditions! If necessary perform chromatic adaptation!
     |
     | 2a. Jz represents the 'lightness' relative to a D65 white with luminance = 10000 cd/m² 
     |      (note that Jz that not exactly equal 1 for this high value, but rather for 102900 cd/m2)
     | 2b. az, bz represent respectively a red-green and a yellow-blue opponent axis 
     |      (but note that a D65 shows a small offset from (0,0))

    Reference:
        1. `Safdar, M., Cui, G., Kim,Y. J., and Luo, M. R. (2017).
        Perceptually uniform color space for image signals including high dynamic range and wide gamut.
        Opt. Express, vol. 25, no. 13, pp. 15131–15151, June 2017. 
        <http://www.opticsexpress.org/abstract.cfm?URI=oe-25-13-15131>`_
    """
    xyz = np2d(xyz)
    
    # Setup X,Y,Z to X',Y',Z' transform as matrix:
    b = 1.15
    g = 0.66
    M_to_xyzp = np.array([[b, 0, 1 - b],[1 - g, g, 0],[0, 0, 1]])

    # Define X',Y',Z' to L,M,S conversion matrix:
    M_to_lms = np.array([[0.41478972, 0.579999, 0.0146480,],
                  [-0.2015100, 1.120649, 0.0531008],
                  [-0.0166008, 0.264800, 0.6684799]])
    
    # Premultiply M_to_xyzp and M_to_lms:
    M =  M_to_lms @ M_to_xyzp
    
    # Transform X,Y,Z to L,M,S:
    if len(xyz.shape) == 3:
        lms = np.einsum('ij,klj->kli', M, xyz)
    else:
        lms = np.einsum('ij,lj->li', M, xyz)
    
    # response compression: lms to lms'
    lmsp = ((3424/(2**12) + (2413/(2**7))*(lms/10000)**(2610/(2**14)))/(1 + (2392/(2**7))*((lms/10000)**(2610/(2**14)))))**(1.7*2523/(2**5))

    # Transform L',M',S' to Iabz:
    M = np.array([[0.5, 0.5, 0],
                  [3.524000, -4.066708, 0.542708],
                  [0.199076, 1.096799, -1.295875]])
    if len(lms.shape) == 3:
        Iabz = np.einsum('ij,klj->kli', M, lmsp)
    else:
        Iabz = np.einsum('ij,lj->li', M, lmsp)

    # convert Iabz' to Jabz coordinates:

    Iabz[...,0] = ((1-0.56)*Iabz[...,0]/(1-0.56*Iabz[...,0])) - 1.6295499532821566e-11
    return Iabz

def jabz_to_xyz(jabz, ztype = 'jabz', **kwargs):
    """
    Convert Jz,az,bz color coordinates to XYZ tristimulus values.

    Args:
        :jabz: 
            | ndarray with Jz,az,bz color coordinates
        :ztype:
            | 'jabz', optional
            | String with requested return:
            | Options: 'jabz', 'iabz'
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

    Reference:
        1. `Safdar, M., Cui, G., Kim,Y. J., and Luo, M. R. (2017).
        Perceptually uniform color space for image signals including high dynamic range and wide gamut.
        Opt. Express, vol. 25, no. 13, pp. 15131–15151, June, 2017.
        <http://www.opticsexpress.org/abstract.cfm?URI=oe-25-13-15131>`_
    """
    jabz = np2d(jabz)
    
    # Convert Jz to Iz:
    jabz[...,0] = (jabz[...,0] + 1.6295499532821566e-11)/(1 - 0.56*(1 - (jabz[...,0] + 1.6295499532821566e-11)))

    # Convert Iabz to lmsp:
    M = np.linalg.inv(np.array([[0.5, 0.5, 0],
                  [3.524000, -4.066708, 0.542708],
                  [0.199076, 1.096799, -1.295875]]))
    
    if len(jabz.shape) == 3:
        lmsp = np.einsum('ij,klj->kli', M, jabz)
    else:
        lmsp = np.einsum('ij,lj->li', M, jabz)
        
    # Convert lmsp to lms:

    lms = 10000*(((3424/2**12) - lmsp**(1/(1.7*2523/2**5))) / (((2392/2**7)*lmsp**(1/(1.7*2523/2**5))) - (2413/2**7)))**(1/(2610/(2**14)))
    
    # Convert lms to xyz:
    # Setup X',Y',Z' from X,Y,Z transform as matrix:
    b = 1.15
    g = 0.66
    M_to_xyzp = np.array([[b, 0, 1 - b],[1 - g, g, 0],[0, 0, 1]])
    
    # Define X',Y',Z' to L,M,S conversion matrix:
    M_to_lms = np.array([[0.41478972, 0.579999, 0.0146480],
                  [-0.2015100, 1.120649, 0.0531008],
                  [-0.0166008, 0.264800, 0.6684799]])
    
    # Premultiply M_to_xyzp and M_to_lms and invert:
    M = M_to_lms @ M_to_xyzp
    M = np.linalg.inv(M)
    
    # Transform L,M,S to X,Y,Z:
    if len(jabz.shape) == 3:
        xyz = np.einsum('ij,klj->kli', M, lms)
    else:
        xyz = np.einsum('ij,lj->li', M, lms)
        
    return xyz


def run(data, xyzw = None, outin = 'J,aM,bM', cieobs = _CIEOBS,
            conditions = None, forward = True, mcat = 'cat16', **kwargs):
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
            | String with inputs in data. 
            | Input must have data.shape[-1]==3 and last dim of data must have 
            | the following structure: 
            |  * data[...,0] = J or Q,
            |  * data[...,1:] = (aM,bM) or (aC,bC) or (aS,bS)
        :mcat:
            | 'cat16', optional
            | Specifies CAT sensor space.
            | - options:
            |    - None defaults to 'cat16'
            |    - str: see see luxpy.cat._MCATS.keys() for options 
            |         (details on type, ?luxpy.cat)
            |    - ndarray: matrix with sensor primaries
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
    """
    outin = outin.split(',') if isinstance(outin,str) else outin
    
    #--------------------------------------------
    # Get condition parameters:
    if conditions is None:
        conditions = _DEFAULT_CONDITIONS
    print(conditions)
    D, Dtype, La, Yb, surround = (conditions[x] for x in sorted(conditions.keys()))
    
    surround_parameters =  _SURROUND_PARAMETERS
    if isinstance(surround, str):
        surround = surround_parameters[conditions['surround']]
    F, FLL, Nc, c = [surround[x] for x in sorted(surround.keys())]
 
    # Define cone/chromatic adaptation sensor space:  
    if (mcat is None) | (mcat == 'cat16'):
        mcat = cat._MCATS['cat16']
    elif isinstance(mcat,str):
        mcat = cat._MCATS[mcat]
    invmcat = np.linalg.inv(mcat)     
    
    #--------------------------------------------
    # Get white point of D65 fro chromatic adaptation transform (CAT)
    xyzw_d65 = np.array([[9.5047e+01, 1.0000e+02, 1.0888e+02]]) if cieobs == '1931_2'  else  spd_to_xyz(_CIE_D65, cieobs = cieobs)
    
    
    #--------------------------------------------
    # Get default white point:
    if xyzw is None:
        xyzw = xyzw_d65.copy()
 
    
    #--------------------------------------------
    # calculate condition dependent parameters:
    Yw = xyzw[...,1].T
    k = 1.0 / (5.0*La + 1.0)
    FL = 0.2*(k**4.0)*(5.0*La) + 0.1*((1.0 - k**4.0)**2.0)*((5.0*La)**(1.0/3.0)) # luminance adaptation factor
    n = Yb/Yw 
    Nbb = 0.725*(1/n)**0.2   
    z = 1.48 + FLL*n**0.5
     
    #--------------------------------------------
    # Calculate degree of chromatic adaptation:
    if D is None:
        D = F*(1.0-(1.0/3.6)*np.exp((-La-42.0)/92.0))
  
    #===================================================================
    # WHITE POINT transformations (common to forward and inverse modes):   
  
    
    #--------------------------------------------
    # Apply CAT to white point:
    xyzwc = cat.apply_vonkries1(xyzw, xyzw, xyzw_d65, D = D, mcat = mcat, invmcat = invmcat)
    
    
    #--------------------------------------------
    # Get Iz,az,bz coordinates:
    iabzw = xyz_to_jabz(xyzwc, ztype = 'iabz')
    
    
    
    #===================================================================
    # STIMULUS transformations:
    
    #--------------------------------------------
    # massage shape of data for broadcasting:
    original_ndim = data.ndim
    if data.ndim == 2: data = data[:,None]
    
    if forward:
        # Apply CAT to D65:
        xyzc =  cat.apply_vonkries1(data, xyzw, xyzw_d65, D = D, mcat = mcat, invmcat = invmcat)

        # Get Iz,az,bz coordinates:
        iabz = xyz_to_jabz(xyzc, ztype = 'iabz')
 
        #--------------------------------------------
        # calculate hue h and eccentricity factor, et:
        h = hue_angle(iabz[...,1],iabz[...,2], htype = 'deg')
        et = 1.01 + np.cos(h*np.pi/180 + 1.55)
        
        #-------------------------------------------- 
        # calculate Hue quadrature (if requested in 'out'):
        if 'H' in outin:    
            H = hue_quadrature(h, unique_hue_data = _UNIQUE_HUE_DATA)
        else:
            H = None
        
        #--------------------------------------------   
        # calculate lightness, J:
        if ('J' in outin) | ('Q' in outin) | ('C' in outin) | ('M' in outin) | ('s' in outin) | ('aS' in outin) | ('aC' in outin) | ('aM' in outin):
            J = 100.0* (iabz[...,0] / iabzw[...,0])**(c*z)
         
        #-------------------------------------------- 
        # calculate brightness, Q:
        if ('Q' in outin) | ('s' in outin) | ('aS' in outin):
            Q = 192.5 * (J/c) * (FL**0.64)
          
        #-------------------------------------------- 
        # calculate chroma, C:
        if ('C' in outin) | ('M' in outin) | ('s' in outin) | ('aS' in outin) | ('aC' in outin) | ('aM' in outin):
            C = ((1/n)**0.074)*((iabz[...,1]**2.0 + iabz[...,2]**2.0)**0.37) * (et**0.067)

             
        #-------------------------------------------- 
        # calculate colorfulness, M:
        if ('M' in outin) | ('s' in outin) | ('aM' in outin) | ('aS' in outin):
            M = 1.42*C*FL**0.25
        
        #--------------------------------------------         
        # calculate saturation, s:
        if ('s' in outin) | ('aS' in outin):
            s = 100.0* (M/Q)**0.5
        
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
        # Get Lightness J from data:
        if ('J' in outin[0]):
            J = data[...,0].copy()
        elif ('Q' in outin[0]):
            Q = data[...,0].copy()
            J = c*(Q / (192.25 * FL**0.64))
        else:
            raise Exception('No lightness or brightness values in data[...,0]. Inverse CAM-transform not possible!')
            
            
        #--------------------------------------------    
        if 'a' in outin[1]: 
            # calculate hue h:
            h = hue_angle(data[...,1],data[...,2], htype = 'deg')
        
            #--------------------------------------------
            # calculate Colorfulness M or Chroma C or Saturation s from a,b:
            MCs = (data[...,1]**2.0 + data[...,2]**2.0)**0.5   
        else:
            h = data[...,2]
            MCs = data[...,1]   
        
        
        if ('aS' in outin):
            Q = 192.5 * (J/c) * (FL**0.64)
            M = Q*(MCs/100.0)**2.0 
            C = M/(1.42*FL**0.25)
         
        if ('aM' in outin): # convert M to C:
            C = MCs/(1.42*FL**0.25)
        
        if ('aC' in outin):
            C = MCs
        
        #--------------------------------------------
        # calculate achromatic signal, Iz:
        Iz = iabzw[...,0]*(J/100.0)**(1.0/(c*z))
        
        #--------------------------------------------    
        # calculate eccentricity factor, et:
        et = 1.01 + np.cos(h*np.pi/180 + 1.55)

        #--------------------------------------------
        # calculate t (=a**2+b**2) from C:
        t = (n**0.074 * C * (1/et)**0.067)**(1/0.37)        
        
        #--------------------------------------------
        # Calculate az, bz:
        az = (t / (1 + np.tan(h*np.pi/180)**2))**0.5
        bz = az*np.tan(h*np.pi/180)
                  
        #--------------------------------------------
        # join values and convert to xyz:
        xyzc = jabz_to_xyz(ajoin((Iz,az,bz)), ztype = 'iabz')


        #-------------------------------------------
        # Apply CAT from D65:
        xyz =  cat.apply_vonkries1(xyzc, xyzw_d65, xyzw, D = D, mcat = mcat, invmcat = invmcat)

        return xyz
    
        
#------------------------------------------------------------------------------
# wrapper functions for use with colortf():
#------------------------------------------------------------------------------
camjabz = run
def xyz_to_jabM_camjabz(data, xyzw = _DEFAULT_WHITE_POINT, cieobs = _CIEOBS,
                         conditions = None, mcat = 'cat16', **kwargs):
    """
    Wrapper function for camjabz forward mode with J,aM,bM output.
    
    | For help on parameter details: ?luxpy.cam.camjabz 
    """
    return camjabz(data, xyzw = xyzw, cieobs = cieobs, conditions = conditions, forward = True, outin = 'J,aM,bM', mcat = mcat)
   

def jabM_camjabz_to_xyz(data, xyzw = _DEFAULT_WHITE_POINT, cieobs = _CIEOBS,
                         conditions = None, mcat = 'cat16', **kwargs):
    """
    Wrapper function for camjabz inverse mode with J,aM,bM input.
    
    | For help on parameter details: ?luxpy.cam.camjabz 
    """
    return camjabz(data, xyzw = xyzw, cieobs = cieobs, conditions = conditions, forward = False, outin = 'J,aM,bM', mcat = mcat)



def xyz_to_jabC_camjabz(data, xyzw = _DEFAULT_WHITE_POINT, cieobs = _CIEOBS,
                         conditions = None, mcat = 'cat16', **kwargs):
    """
    Wrapper function for camjabz forward mode with J,aC,bC output.
    
    | For help on parameter details: ?luxpy.cam.camjabz 
    """
    return camjabz(data, xyzw = xyzw, cieobs = cieobs, conditions = conditions, forward = True, outin = 'J,aC,bC', mcat = mcat)
 

def jabC_camjabz_to_xyz(data, xyzw = _DEFAULT_WHITE_POINT, cieobs = _CIEOBS,
                         conditions = None, mcat = 'cat16', **kwargs):
    """
    Wrapper function for camjabz inverse mode with J,aC,bC input.
    
    | For help on parameter details: ?luxpy.cam.camjabz 
    """
    return camjabz(data, xyzw = xyzw, cieobs = cieobs, conditions = conditions, forward = False, outin = 'J,aC,bC', mcat = mcat)
    
  
#==============================================================================  
if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    # Code test
    #--------------------------------------------------------------------------
    
    _cam = run
    
    import luxpy as lx
    from luxpy.utils import np, plt
    
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
    import matplotlib.pyplot as plt
    _cam_o = lambda xyz, xyzw, forward: lx.xyz_to_jabz(xyz)
    xyz, xyzw = lx.spd_to_xyz(Ill1, cieobs = cieobs, relative = True, rfl = rflM, out = 2)
    jabch = _cam(xyz, xyzw = xyzw, forward = True, outin = 'J,aM,bM')
    out_ = _cam_o(xyz, xyzw = xyzw, forward = True)
    plt.figure()
    plt.plot(jabch[...,1],jabch[...,2],'c.')
    plt.plot(out_[...,1],out_[...,2],'r.')
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
    
    
     
    user_conditions = {'D': 1, 'Dtype': None,\
                   'La': 500.0, 'Yb': 20.0, 'surround': 'avg'}

    jabM_camjabz = lx.xyz_to_jabM_camjabz(xyz[:,0,:], xyzw = xyzw)
    jabC_camjabz = lx.xyz_to_jabC_camjabz(xyz[:,0,:], xyzw = xyzw)
    print("JabM_camjabz (default viewing conditions) = ", jabM_camjabz)
    
    jabM_camjabz_user_vc = lx.xyz_to_jabM_camjabz(xyz[:,0,:], xyzw = xyzw, conditions = user_conditions)
    jabC_camjabz_user_vc = lx.xyz_to_jabC_camjabz(xyz[:,0,:], xyzw = xyzw, conditions = user_conditions)

    print("JabM_camjabz (user defined viewing conditions) = ", jabM_camjabz_user_vc)    
        
     
    fig, axs = plt.subplots(1,2,figsize=(12,6));

    axs[0].plot(jabM_camjabz[...,1:2],jabM_camjabz[...,2:3],'b.', label = 'camjabz')
    axs[0].plot(jabM_camjabz_user_vc[...,1:2],jabM_camjabz_user_vc[...,2:3],'g.', label = 'camjabz (user cond.)')
    axs[0].set_xlabel('azM (camjabz)')
    axs[0].set_ylabel('bzM (camjabz)')
    
    axs[1].plot(jabC_camjabz[...,1:2],jabC_camjabz[...,2:3],'rx', label = 'camjabz')
    axs[1].plot(jabC_camjabz_user_vc[...,1:2],jabC_camjabz_user_vc[...,2:3],'c.', label = 'camjabz (user cond.)')
    axs[1].set_xlabel('azC (camjabz)')
    axs[1].set_ylabel('bzC (camjabz)')
    
    