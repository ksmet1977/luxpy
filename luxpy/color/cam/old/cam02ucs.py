# -*- coding: utf-8 -*-
"""
CAM02UCS color appearance difference space
==========================================

 :run(): Run the CAM02-UCS[,-LCD,-SDC] color appearance difference model in forward or backward modes.
 
Created on Wed Sep 30 14:35:05 2020

@author: ksmet1977 at gmail.com
"""
from luxpy.utils import asplit, ajoin
from luxpy.color.cam.ciecam02 import run as ciecam02
import numpy as np

def run(data, xyzw, conditions = None, ucs_type = 'ucs', forward = True):
    """ 
    Run the CAM02-UCS[,-LCD,-SDC] color appearance difference model in forward or backward modes.
    
    Args:
        :data:
            | ndarray with sample xyz values (forward mode) or J'a'b' coordinates (inverse mode)
        :xyzw:
            | ndarray with white point tristimulus values  
        :conditions:
            | None, optional
            | Dictionary with viewing conditions.
            | None results in:
            |   {'La':100, 'Yb':20, 'D':1, 'surround':'avg'}
            | For more info see luxpy.cam.ciecam02()?
        :ucs_type:
            | 'ucs', optional
            | String with type of color difference appearance space
            | options: 'ucs', 'scd', 'lcd'
        :forward:
            | True, optional
            | If True: run in CAM in forward mode, else: inverse mode.

    Returns:
        :camout:
            | ndarray with J'a'b' coordinates or whatever correlates requested in out.
    
    Note:
        * This is a simplified, less flexible, but faster version than the main cam02ucs().
    """
    # get ucs parameters:
    if isinstance(ucs_type,str):
        ucs_pars = {'ucs':{'KL': 1.0, 'c1':0.007,'c2':0.0228},
                    'lcd':{'KL': 0.77, 'c1':0.007,'c2':0.0053}, 
                    'scd':{'KL': 1.24, 'c1':0.007,'c2':0.0363}}
        ucs = ucs_pars[ucs_type]
    else:
        ucs = ucs_type
    KL, c1, c2 =  ucs['KL'], ucs['c1'], ucs['c2']
    
    if forward == True:
        
        # run ciecam02 to get JMh:
        data = ciecam02(data, xyzw, out = 'J,M,h', conditions = conditions, forward = True) 
        
        camout = np.zeros_like(data) # for output

        #--------------------------------------------
        # convert to cam02ucs J', aM', bM':
        camout[...,0] = (1.0 + 100.0*c1)*data[...,0] / (1.0 + c1*data[...,0])
        Mp = (1.0/c2) * np.log(1.0 + c2*data[...,1])
        camout[...,1] = Mp * np.cos(data[...,2]*np.pi/180)
        camout[...,2] = Mp * np.sin(data[...,2]*np.pi/180)
        
        return camout
    
    else:
        #--------------------------------------------
        # convert cam02ucs J', aM', bM' to xyz:
            
        # calc CAM02 hue angle
        #Jp, aMp, bMp = asplit(data)
        h = np.arctan2(data[...,2],data[...,1])

        # calc CAM02 and CIECAM02 colourfulness
        Mp = (data[...,1]**2.0 + data[...,2]**2.0)**0.5
        M = (np.exp(c2*Mp) - 1.0) / c2
        
        # calculate ciecam02 aM, bM:
        aM = M * np.cos(h)
        bM = M * np.sin(h)

        # calc CAM02 lightness
        J = data[...,0] / (1.0 + (100.0 - data[...,0]) * c1)

        
         # run ciecam02 in inverse mode to get xyz:
        return ciecam02(ajoin((J,aM,bM)), xyzw, out = 'J,aM,bM', conditions = conditions, forward = False) 

        

if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    # Code test
    #--------------------------------------------------------------------------
    _cam = run
    
    import luxpy as lx
    import numpy as np
    
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
    _cam_o = lambda xyz, xyzw, forward: lx.cri.xyz_to_jab_cam02ucs_fast(xyz,xyzw, ucs = True)
    _cam_o2 = lambda xyz, xyzw, forward: lx.xyz_to_jab_cam02ucs(xyz,xyzw)

    xyz, xyzw = lx.spd_to_xyz(Ill1, cieobs = cieobs, relative = True, rfl = rflM, out = 2)
    jabch = _cam(xyz, xyzw = xyzw, forward = True)
    out_ = _cam_o(xyz, xyzw = xyzw, forward = True)
    out_2 = _cam_o2(xyz, xyzw = xyzw, forward = True)
    plt.figure()
    plt.plot(jabch[...,1],jabch[...,2],'c.')
    plt.plot(out_[...,1],out_[...,2],'r.')
    plt.plot(out_2[...,1],out_2[...,2],'gx')
    plt.axis('equal')



    # Single data for sample and illuminant:
    # test input to _simple_cam():
    print('\n\n1: xyz in:')
    out_1 = _cam(xyz1, xyzw = xyzw1, forward = True)
    xyz_1 = _cam(out_1[...,:3], xyzw = xyzw1, forward = False)
    print((xyz1 - xyz_1).sum())
    
    
    # Multiple data for sample and illuminants:
    print('\n\n2: xyz in:')
    out_2 = _cam(xyz2, xyzw = xyzw2, forward = True)
    xyz_2 = _cam(out_2[...,:3], xyzw = xyzw2, forward = False)
    print((xyz2 - xyz_2).sum())
        
    
    # Single data for sample, multiple illuminants:
    print('\n\n3: xyz in:')
    out_3 = _cam(xyz1, xyzw = xyzw2, forward = True)
    xyz_3 = _cam(out_3[...,:3], xyzw = xyzw2, forward = False)
    print((xyz1 - xyz_3[:,0,:]).sum())