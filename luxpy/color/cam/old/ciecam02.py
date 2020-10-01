# -*- coding: utf-8 -*-
"""
CIECAM02 color appearance model
===============================

 :run(): Run the CIECAM02 color appearance model in forward or backward modes.

Created on Wed Sep 30 14:17:02 2020

@author: u0032318
"""


from luxpy import math, np
from luxpy.utils import asplit, ajoin
from luxpy.color.cam.utils import hue_angle, hue_quadrature, naka_rushton


def run(data, xyzw, out = 'J,aM,bM', conditions = None, forward = True):
    """ 
    Run CIECAM02 color appearance model in forward or backward modes.
    
    Args:
        :data:
            | ndarray with relative sample xyz values (forward mode) or J'a'b' coordinates (inverse mode)
        :xyzw:
            | ndarray with relative white point tristimulus values  
        :conditions:
            | None, optional
            | Dictionary with viewing conditions.
            | None results in:
            |   {'La':100, 'Yb':20, 'D':1, 'surround':'avg'}
            | For more info see luxpy.cam.ciecam02()?
        :forward:
            | True, optional
            | If True: run in CAM in forward mode, else: inverse mode.
        :out:
            | 'J,aM,bM', optional
            | String with requested output (e.g. "J,aM,bM,M,h") [Forward mode]
            | String with inputs in data. 
            | Input must have data.shape[-1]==3 and last dim of data must have 
            | the following structure: 
            |  * data[...,0] = J or Q,
            |  * data[...,1:] = (aM,bM) or (aC,bC) or (aS,bS)
    Returns:
        :camout:
            | ndarray with Jab coordinates or whatever correlates requested in out.
    
    Note:
        * This is a simplified, less flexible, but faster version than the main ciecam02().
    
    References:
        1. `N. Moroney, M. D. Fairchild, R. W. G. Hunt, C. Li, M. R. Luo, and T. Newman, (2002), 
        "The CIECAM02 color appearance model,” 
        IS&T/SID Tenth Color Imaging Conference. p. 23, 2002.
        <http://rit-mcsl.org/fairchild/PDFs/PRO19.pdf>`_
    """
    outin = out.split(',') if isinstance(out,str) else out
    
    #--------------------------------------------
    # Get/ set conditions parameters:
    if conditions is not None:
        surround_parameters =  {'surrounds': ['avg', 'dim', 'dark'], 
                                'avg' : {'c':0.69, 'Nc':1.0, 'F':1.0,'FLL': 1.0}, 
                                'dim' : {'c':0.59, 'Nc':0.9, 'F':0.9,'FLL':1.0} ,
                                'dark' : {'c':0.525, 'Nc':0.8, 'F':0.8,'FLL':1.0}}
        La = conditions['La']
        Yb = conditions['Yb']
        D = conditions['D']
        surround = conditions['surround']
        if isinstance(surround, str):
            surround = surround_parameters[conditions['surround']]
        F, FLL, Nc, c = [surround[x] for x in sorted(surround.keys())]
    else:
        # set defaults:
        La, Yb, D, F, FLL, Nc, c = 100, 20, 1, 1, 1, 1, 0.69
        
    #--------------------------------------------
    # Define sensor space and cat matrices:        
    mhpe = np.array([[0.38971,0.68898,-0.07868],
                     [-0.22981,1.1834,0.04641],
                     [0.0,0.0,1.0]]) # Hunt-Pointer-Estevez sensors (cone fundamentals)
    
    mcat = np.array([[0.7328, 0.4296, -0.1624],
                       [ -0.7036, 1.6975,  0.0061],
                       [ 0.0030, 0.0136,  0.9834]]) # CAT02 sensor space
    
    #--------------------------------------------
    # pre-calculate some matrices:
    invmcat = np.linalg.inv(mcat)
    mhpe_x_invmcat = np.dot(mhpe,invmcat)
    if not forward: mcat_x_invmhpe = np.dot(mcat,np.linalg.inv(mhpe))
    
    #--------------------------------------------
    # calculate condition dependent parameters:
    Yw = xyzw[...,1:2].T
    k = 1.0 / (5.0*La + 1.0)
    FL = 0.2*(k**4.0)*(5.0*La) + 0.1*((1.0 - k**4.0)**2.0)*((5.0*La)**(1.0/3.0)) # luminance adaptation factor
    n = Yb/Yw 
    Nbb = 0.725*(1/n)**0.2   
    Ncb = Nbb
    z = 1.48 + FLL*n**0.5
    
    if D is None:
        D = F*(1.0-(1.0/3.6)*np.exp((-La-42.0)/92.0))
        
    #===================================================================
    # WHITE POINT transformations (common to forward and inverse modes):
    
    #--------------------------------------------
    # transform from xyzw to cat sensor space:
    rgbw = mcat @ xyzw.T
    
    #--------------------------------------------  
    # apply von Kries cat:
    rgbwc = ((D*Yw/rgbw) + (1 - D))*rgbw # factor 100 from ciecam02 is replaced with Yw[i] in cam16, but see 'note' in Fairchild's "Color Appearance Models" (p291 ni 3ed.)

    #--------------------------------------------
    # convert from cat02 sensor space to cone sensors (hpe):
    rgbwp = (mhpe_x_invmcat @ rgbwc).T
    
    #--------------------------------------------
    # apply Naka_rushton repsonse compression to white:
    NK = lambda x, forward: naka_rushton(x, scaling = 400, n = 0.42, sig = 27.13**(1/0.42), noise = 0.1, forward = forward)
    
    rgbwpa = NK(FL*rgbwp/100.0, True)
    pw = np.where(rgbwp<0)
    rgbwpa[pw] = 0.1 - (NK(FL*np.abs(rgbwp[pw])/100.0, True) - 0.1)
    
    #--------------------------------------------
    # Calculate achromatic signal of white:
    Aw =  (2.0*rgbwpa[...,0] + rgbwpa[...,1] + (1.0/20.0)*rgbwpa[...,2] - 0.305)*Nbb
    
    # massage shape of data for broadcasting:
    if data.ndim == 2: data = data[:,None]

    #===================================================================
    # STIMULUS transformations 
    if forward:
        
        #--------------------------------------------
        # transform from xyz to cat sensor space:
        rgb = math.dot23(mcat, data.T)
        
        #--------------------------------------------  
        # apply von Kries cat:
        rgbc = ((D*Yw/rgbw)[...,None] + (1 - D))*rgb # factor 100 from ciecam02 is replaced with Yw[i] in cam16, but see 'note' in Fairchild's "Color Appearance Models" (p291 ni 3ed.)
        
        #--------------------------------------------
        # convert from cat02 sensor space to cone sensors (hpe):
        rgbp = math.dot23(mhpe_x_invmcat,rgbc).T
        
        #--------------------------------------------
        # apply Naka_rushton repsonse compression:        
        rgbpa = NK(FL*rgbp/100.0, forward)
        p = np.where(rgbp<0)
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
            H = hue_quadrature(h, unique_hue_data = 'ciecam02')
        else:
            H = None
        
        #--------------------------------------------   
        # calculate lightness, J:
        if ('J' in outin) | ('Q' in outin) | ('C' in outin) | ('M' in outin) | ('s' in outin) | ('aS' in outin) | ('aC' in outin) | ('aM' in outin):
            J = 100.0* (A / Aw)**(c*z)
         
        #-------------------------------------------- 
        # calculate brightness, Q:
        if ('Q' in outin) | ('s' in outin) | ('aS' in outin):
            Q = (4.0/c)* ((J/100.0)**0.5) * (Aw + 4.0)*(FL**0.25)
          
        #-------------------------------------------- 
        # calculate chroma, C:
        if ('C' in outin) | ('M' in outin) | ('s' in outin) | ('aS' in outin) | ('aC' in outin) | ('aM' in outin):
            t = ((50000.0/13.0)*Nc*Ncb*et*((a**2.0 + b**2.0)**0.5)) / (rgbpa[...,0] + rgbpa[...,1] + (21.0/20.0*rgbpa[...,2]))
            C = (t**0.9)*((J/100.0)**0.5) * (1.64 - 0.29**n)**0.73
               
        #-------------------------------------------- 
        # calculate colorfulness, M:
        if ('M' in outin) | ('s' in outin) | ('aM' in outin) | ('aS' in outin):
            M = C*FL**0.25
        
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
        
        if camout.shape[1] == 1:
            camout = camout[:,0,:]

        
        return camout
        
    elif forward == False:

                       
            #--------------------------------------------
            # Get Lightness J from data:
            if ('J' in outin):
                J = data[...,0].copy()
            elif ('Q' in outin):
                Q = data[...,0].copy()
                J = 100.0*(Q / ((Aw + 4.0)*(FL**0.25)*(4.0/c)))**2.0
            else:
                raise Exception('No lightness or brightness values in data. Inverse CAM-transform not possible!')
                
                
            #--------------------------------------------
            # calculate hue h:
            h = hue_angle(data[...,1],data[...,2], htype = 'deg')
            
            #--------------------------------------------
            # calculate Colorfulness M or Chroma C or Saturation s from a,b:
            MCs = (data[...,1]**2.0 + data[...,2]**2.0)**0.5    
            
            
            if ('aS' in outin):
                Q = (4.0/c)* ((J/100.0)**0.5) * (Aw + 4.0)*(FL**0.25)
                M = Q*(MCs/100.0)**2.0 
                C = M/(FL**0.25)
             
            if ('aM' in outin): # convert M to C:
                C = MCs/(FL**0.25)
            
            if ('aC' in outin):
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
            # convert from to cone sensors (hpe) cat02 sensor space:
            rgbc = math.dot23(mcat_x_invmhpe,rgbp.T)
                            
            #--------------------------------------------
            # apply inverse von Kries cat:
            rgb = rgbc / ((D*Yw/rgbw)[...,None] + (1.0 - D))
            
            #--------------------------------------------
            # transform from cat sensor space to xyz:
            xyz = math.dot23(invmcat,rgb).T
            
            
            return xyz
        
if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    # Code test
    #--------------------------------------------------------------------------
    _cam = run
    
    import luxpy as lx
    from luxpy import np
    
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
    _cam_o = lambda xyz, xyzw, forward: lx.xyz_to_jabM_ciecam02(xyz,xyzw)
    xyz, xyzw = lx.spd_to_xyz(Ill1, cieobs = cieobs, relative = True, rfl = rflM, out = 2)
    jabch = _cam(xyz, xyzw = xyzw, forward = True, out = 'J,aM,bM')
    out_ = _cam_o(xyz, xyzw = xyzw, forward = True)
    plt.figure()
    plt.plot(jabch[...,1],jabch[...,2],'c.')
    plt.plot(out_[...,1],out_[...,2],'r.')
    plt.axis('equal')



    out = 'J,aM,bM,M,C,s,h'.split(',')
    # Single data for sample and illuminant:
    # test input to _simple_cam():
    print('\n\n1: xyz in:')
    out_1 = _cam(xyz1, xyzw = xyzw1, forward = True, out = out)
    xyz_1 = _cam(out_1[...,:3], xyzw = xyzw1, forward = False, out = out[:3])
    print((xyz1 - xyz_1).sum())
    
    
    # Multiple data for sample and illuminants:
    print('\n\n2: xyz in:')
    out_2 = _cam(xyz2, xyzw = xyzw2, forward = True, out = out)
    xyz_2 = _cam(out_2[...,:3], xyzw = xyzw2, forward = False, out = out[:3])
    print((xyz2 - xyz_2).sum())
        
    
    # Single data for sample, multiple illuminants:
    print('\n\n3: xyz in:')
    out_3 = _cam(xyz1, xyzw = xyzw2, forward = True, out = out)
    xyz_3 = _cam(out_3[...,:3], xyzw = xyzw2, forward = False, out = out[:3])
    print((xyz1 - xyz_3[:,0,:]).sum())
    
    
    
    