# -*- coding: utf-8 -*-
"""
CIECAM16 color appearance model
===============================

 :_UNIQUE_HUE_DATA: dictionary with unique hue data 
            
 :_SURROUND_PARAMETERS: dictionary with surround param. c, Nc, F and FLL for 'avg','dim' and 'dark' conditions

 :_NAKA_RUSHTON_PARAMETERS: | dictionary with parameters (n, sig, scaling and noise) 
                            | for the Naka-Rushton function: 
                            |   NK(x) = sign(x) * scaling * ((abs(x)**n) / ((abs(x)**n) + (sig**n))) + noise

 :_DEFAULT_WHITE_POINT: Default internal reference white point (xyz)

 :_DEFAULT_CONDITIONS: Default CAM model parameters 

 :_AXES: dict with list[str,str,str] containing axis labels of defined cspaces.

 :run(): Run the CIECAM16 color appearance model in forward or backward modes.
 
 :ciecam16(): Run the CIECAM16 color appearance model in forward or backward modes.

 :specific_wrappers_in_the_'xyz_to_cspace()' and 'cpsace_to_xyz()' format:
      | 'xyz_to_jabM_ciecam16', 'jabM_ciecam16_to_xyz',
      | 'xyz_to_jabC_ciecam16', 'jabC_ciecam16_to_xyz',
      

Created on Wed Sep 30 21:58:11 2020

@author: ksmet1977 at gmail.com
"""


from luxpy import math
from luxpy.utils import np, asplit, ajoin
from luxpy import cat
from luxpy.color.cam.utils import hue_angle, hue_quadrature, naka_rushton

__all__ = ['run', 'ciecam16',
           '_AXES','_UNIQUE_HUE_DATA','_DEFAULT_WHITE_POINT',
           '_SURROUND_PARAMETERS', '_NAKA_RUSHTON_PARAMETERS']

__all__ += ['xyz_to_jabM_ciecam16', 'jabM_ciecam16_to_xyz', 
            'xyz_to_jabC_ciecam16', 'jabC_ciecam16_to_xyz']

_UNIQUE_HUE_DATA = {'hues': 'red yellow green blue red'.split(), 
                    'i': [0,1,2,3,4], 
                    'hi':[20.14, 90.0, 164.25,237.53,380.14],
                    'ei':[0.8,0.7,1.0,1.2,0.8],
                    'Hi':[0.0,100.0,200.0,300.0,400.0]}

_SURROUND_PARAMETERS =  {'surrounds': ['avg', 'dim', 'dark'], 
                         'avg' : {'c':0.69, 'Nc':1.0, 'F':1.0,'FLL': 1.0}, 
                         'dim' : {'c':0.59, 'Nc':0.9, 'F':0.9,'FLL':1.0} ,
                         'dark' : {'c':0.525, 'Nc':0.8, 'F':0.8,'FLL':1.0}}

_NAKA_RUSHTON_PARAMETERS = {'n':0.42, 'sig': 27.13**(1/0.42), 'scaling': 400.0, 'noise': 0.1}

_DEFAULT_CONDITIONS = {'La': 100.0, 'Yb': 20.0, 'surround': 'avg','D': 1.0, 'Dtype': None}

_DEFAULT_WHITE_POINT = np.array([[100.0,100.0,100.0]])

# Plotting ease:
_AXES = {'jabM_ciecam16' : ["J (ciecam16)", "aM (ciecam16)", "bM (ciecam16)"]}
_AXES['jabC_ciecam16'] = ["J (ciecam16)", "aC (ciecam16)", "bC (ciecam16)"] 



# Main function:
def run(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None, outin = 'J,aM,bM', 
        conditions = None, forward = True, mcat = 'cat16'):
    """ 
    Run CIECAM16 color appearance model in forward or backward modes.
    
    Args:
        :data:
            | ndarray with relative sample xyz values (forward mode) or J'a'b' coordinates (inverse mode)
        :xyzw:
            | ndarray with relative white point tristimulus values 
        :Yw: 
            | None, optional
            | Luminance factor of white point.
            | If None: xyz (in data) and xyzw are entered as relative tristimulus values 
            |          (normalized to Yw = 100). 
            | If not None: input tristimulus are absolute and Yw is used to
            |              rescale the absolute values to relative ones 
            |              (relative to a reference perfect white diffuser 
            |               with Ywr = 100). 
            | Yw can be < 100 for e.g. paper as white point. If Yw is None, it 
            | is assumed that the relative Y-tristimulus value in xyzw 
            | represents the luminance factor Yw.
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
            | String with inputs in data [inverse mode]. 
            | Input must have data.shape[-1]==3 and last dim of data must have 
            | the following structure for inverse mode: 
            |  * data[...,0] = J or Q,
            |  * data[...,1:] = (aM,bM) or (aC,bC) or (aS,bS) or (M,h) or (C, h), ...
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
        1. `C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, (2017), 
        “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” 
        Color Res. Appl., p. n/a–n/a.
        <http://onlinelibrary.wiley.com/doi/10.1002/col.22131/abstract>`_
    """
    outin = outin.split(',') if isinstance(outin,str) else outin
    
    #--------------------------------------------
    # Get condition parameters:
    if conditions is None:
        conditions = _DEFAULT_CONDITIONS
    D, Dtype, La, Yb, surround = (conditions[x] for x in sorted(conditions.keys()))

    surround_parameters =  _SURROUND_PARAMETERS
    if isinstance(surround, str):
        surround = surround_parameters[conditions['surround']]
    F, FLL, Nc, c = [surround[x] for x in sorted(surround.keys())]
           
    #--------------------------------------------
    # Define cone/chromatic adaptation sensor space:  
    if (mcat is None) | (mcat == 'cat16'):
        mcat = cat._MCATS['cat16']
    elif isinstance(mcat,str):
        mcat = cat._MCATS[mcat]
    
    #--------------------------------------------
    # pre-calculate some matrices:
    invmcat = np.linalg.inv(mcat)
    
    #--------------------------------------------
    # Set Yw:
    if Yw is not None:
        Yw = (Yw*np.ones_like(xyzw2[...,1:2]).T)
    else:
        Yw = xyzw[...,1:2].T
    
    #--------------------------------------------
    # calculate condition dependent parameters:
    k = 1.0 / (5.0*La + 1.0)
    FL = 0.2*(k**4.0)*(5.0*La) + 0.1*((1.0 - k**4.0)**2.0)*((5.0*La)**(1.0/3.0)) # luminance adaptation factor
    n = Yb/Yw 
    Nbb = 0.725*(1/n)**0.2   
    Ncb = Nbb
    z = 1.48 + FLL*n**0.5
    yw = xyzw[...,1:2].T # original Y in xyzw (pre-transposed)
    
    #--------------------------------------------
    # Calculate degree of chromatic adaptation:
    if D is None:
        D = F*(1.0-(1.0/3.6)*np.exp((-La-42.0)/92.0))
        
    #===================================================================
    # WHITE POINT transformations (common to forward and inverse modes):
    
    #--------------------------------------------
    # Normalize white point (keep transpose for next step):
    xyzw = Yw*xyzw.T/yw    
    
    #--------------------------------------------
    # transform from xyzw to cat sensor space:
    rgbw = math.dot23(mcat, xyzw)
    
    #--------------------------------------------  
    # apply von Kries cat:
    rgbwc = ((D*Yw/rgbw) + (1 - D))*rgbw # factor 100 from ciecam16 is replaced with Yw[i] in cam16, but see 'note' in Fairchild's "Color Appearance Models" (p291 ni 3ed.)

    #--------------------------------------------
    # convert from cat16 sensor space to cone sensors:
    rgbwp = rgbwc.T # in ciecam16 the 'cone sensors' and 'cat sensor' are the same

    
    #--------------------------------------------
    # apply Naka_rushton repsonse compression to white:
    NK = lambda x, forward: naka_rushton(x, scaling = 400, n = 0.42, sig = 27.13**(1/0.42), noise = 0.1, forward = forward)
    pw = np.where(rgbwp<0)
    rgbwpa = NK(FL*rgbwp/100.0, True)
    rgbwpa[pw] = 0.1 - (NK(FL*np.abs(rgbwp[pw])/100.0, True) - 0.1)
    
    #--------------------------------------------
    # Calculate achromatic signal of white:
    Aw =  (2.0*rgbwpa[...,0] + rgbwpa[...,1] + (1.0/20.0)*rgbwpa[...,2] - 0.305)*Nbb
    
    # massage shape of data for broadcasting:
    original_ndim = data.ndim
    if data.ndim == 2: data = data[:,None]

    #===================================================================
    # STIMULUS transformations 
    if forward:
        
        #--------------------------------------------
        # Normalize xyz (keep transpose for matrix multiplication in next step):
        xyz = (Yw/yw)[...,None]*data.T
        
        #--------------------------------------------
        # transform from xyz to cone/cat sensor space:
        rgb = math.dot23(mcat, xyz)
        
        #--------------------------------------------  
        # apply von Kries cat:
        rgbc = ((D*Yw/rgbw)[...,None] + (1 - D))*rgb # factor 100 from ciecam16 is replaced with Yw[i] in cam16, but see 'note' in Fairchild's "Color Appearance Models" (p291 ni 3ed.)
        
        #--------------------------------------------
        # convert from cat16 sensor space to cone sensors:
        rgbp = rgbc.T # in ciecam16 the 'cone sensors' and 'cat sensor' are the same
        
        #--------------------------------------------
        # apply Naka_rushton repsonse compression:        
        p = np.where(rgbp<0)
        rgbpa = NK(FL*rgbp/100.0, forward)
        rgbpa[p] = 0.1 - (NK(FL*np.abs(rgbp[p])/100.0, forward) - 0.1)
        
        #--------------------------------------------
        # Calculate achromatic signal:
        A  =  (2.0*rgbpa[...,0] + rgbpa[...,1] + (1.0/20.0)*rgbpa[...,2] - 0.305)*Nbb
                
        #--------------------------------------------
        # calculate initial opponent channels:
        a = rgbpa[...,0] - 12.0*rgbpa[...,1]/11.0 + rgbpa[...,2]/11.0
        b = (1.0/9.0)*(rgbpa[...,0] + rgbpa[...,1] - 2.0*rgbpa[...,2])

        #--------------------------------------------
        # calculate hue h and eccentricity factor, et:
        h = hue_angle(a,b, htype = 'deg')
        et = (1.0/4.0)*(np.cos(h*np.pi/180 + 2.0) + 3.8)
        
        #-------------------------------------------- 
        # calculate Hue quadrature (if requested in 'out'):
        if 'H' in outin:    
            H = hue_quadrature(h, unique_hue_data = _UNIQUE_HUE_DATA)
        else:
            H = None
        
        #--------------------------------------------   
        # calculate lightness, J:
        J = 100.0* (A / Aw)**(c*z)
         
        #-------------------------------------------- 
        # calculate brightness, Q:
        Q = (4.0/c)* ((J/100.0)**0.5) * (Aw + 4.0)*(FL**0.25)
          
        #-------------------------------------------- 
        # calculate chroma, C:
        t = ((50000.0/13.0)*Nc*Ncb*et*((a**2.0 + b**2.0)**0.5)) / (rgbpa[...,0] + rgbpa[...,1] + (21.0/20.0*rgbpa[...,2]))
        C = (t**0.9)*((J/100.0)**0.5) * (1.64 - 0.29**n)**0.73
               
        #-------------------------------------------- 
        # calculate colorfulness, M:
        M = C*FL**0.25
        
        #--------------------------------------------         
        # calculate saturation, s:
        s = 100.0* (M/Q)**0.5
        S = s # make extra variable, jsut in case 'S' is called
        
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
            J = 100.0*(Q / ((Aw + 4.0)*(FL**0.25)*(4.0/c)))**2.0
        else:
            raise Exception('No lightness or brightness values in data. Inverse CAM-transform not possible!')
            
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
        
        
        if ('S' in outin[1]):
            Q = (4.0/c)* ((J/100.0)**0.5) * (Aw + 4.0)*(FL**0.25)
            M = Q*(MCs/100.0)**2.0 
            C = M/(FL**0.25)
         
        if ('M' in outin[1]): # convert M to C:
            C = MCs/(FL**0.25)
        
        if ('C' in outin[1]):
            C = MCs
            
        #--------------------------------------------
        # calculate t from J, C:
        t = (C / ((J/100.0)**(1.0/2.0) * (1.64 - 0.29**n)**0.73))**(1.0/0.9)

        #--------------------------------------------
        # calculate eccentricity factor, et:
        et = (np.cos(h*np.pi/180.0 + 2.0) + 3.8) / 4.0
        
        #--------------------------------------------
        # calculate achromatic signal, A:
        A = Aw*(J/100.0)**(1.0/(c*z))

        #--------------------------------------------
        # calculate temporary cart. co. at, bt and p1,p2,p3,p4,p5:
        at = np.cos(h*np.pi/180.0)
        bt = np.sin(h*np.pi/180.0)
        p1 = (50000.0/13.0)*Nc*Ncb*et/t
        p2 = A/Nbb + 0.305
        p3 = 21.0/20.0
        p4 = p1/bt
        p5 = p1/at

        #--------------------------------------------
        #q = np.where(np.abs(bt) < np.abs(at))[0]
        q = (np.abs(bt) < np.abs(at))

        b = p2*(2.0 + p3) * (460.0/1403.0) / (p4 + (2.0 + p3) * (220.0/1403.0) * (at/bt) - (27.0/1403.0) + p3*(6300.0/1403.0))
        a = b * (at/bt)
        
        a[q] = p2[q]*(2.0 + p3) * (460.0/1403.0) / (p5[q] + (2.0 + p3) * (220.0/1403.0) - ((27.0/1403.0) - p3*(6300.0/1403.0)) * (bt[q]/at[q]))
        b[q] = a[q] * (bt[q]/at[q])
        
        #--------------------------------------------
        # calculate post-adaptation values
        rpa = (460.0*p2 + 451.0*a + 288.0*b) / 1403.0
        gpa = (460.0*p2 - 891.0*a - 261.0*b) / 1403.0
        bpa = (460.0*p2 - 220.0*a - 6300.0*b) / 1403.0
        
        #--------------------------------------------
        # join values:
        rgbpa = ajoin((rpa,gpa,bpa))

        #--------------------------------------------
        # decompress signals:
        rgbp = (100.0/FL)*NK(rgbpa, forward)


        #--------------------------------------------
        # convert from to cone sensors to cat16 sensor space:
        rgbc = rgbp.T # in ciecam16, cat and cone sensor spaces are the same
   
                        
        #--------------------------------------------
        # apply inverse von Kries cat:
        rgb = rgbc / ((D*Yw/rgbw)[...,None] + (1.0 - D))
        
        #--------------------------------------------
        # transform from cat sensor space to xyz:
        xyz = math.dot23(invmcat,rgb)
        
        #--------------------------------------------
        # unnormalize xyz:
        xyz = ((yw/Yw)[...,None]*xyz).T 
        
        return xyz
  
#------------------------------------------------------------------------------
# wrapper functions for use with colortf():
#------------------------------------------------------------------------------
ciecam16 = run
def xyz_to_jabM_ciecam16(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                         conditions = None, mcat = 'cat16', **kwargs):
    """
    Wrapper function for ciecam16 forward mode with J,aM,bM output.
    
    | For help on parameter details: ?luxpy.cam.ciecam16 
    """
    return ciecam16(data, xyzw = xyzw, Yw = Yw, conditions = conditions, forward = True, outin = 'J,aM,bM', mcat = mcat)
   

def jabM_ciecam16_to_xyz(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                         conditions = None, mcat = 'cat16', **kwargs):
    """
    Wrapper function for ciecam16 inverse mode with J,aM,bM input.
    
    | For help on parameter details: ?luxpy.cam.ciecam16 
    """
    return ciecam16(data, xyzw = xyzw, Yw = Yw, conditions = conditions, forward = False, outin = 'J,aM,bM', mcat = mcat)



def xyz_to_jabC_ciecam16(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                         conditions = None, mcat = 'cat16', **kwargs):
    """
    Wrapper function for ciecam16 forward mode with J,aC,bC output.
    
    | For help on parameter details: ?luxpy.cam.ciecam16 
    """
    return ciecam16(data, xyzw = xyzw, Yw = Yw, conditions = conditions, forward = True, outin = 'J,aC,bC', mcat = mcat)
 

def jabC_ciecam16_to_xyz(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                         conditions = None, mcat = 'cat16', **kwargs):
    """
    Wrapper function for ciecam16 inverse mode with J,aC,bC input.
    
    | For help on parameter details: ?luxpy.cam.ciecam16 
    """
    return ciecam16(data, xyzw = xyzw, Yw = Yw, conditions = conditions, forward = False, outin = 'J,aC,bC', mcat = mcat)
    
  
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
    _cam_o = lambda xyz, xyzw, forward: lx.xyz_to_jabM_ciecam16(xyz,xyzw)
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
    
    
    
    