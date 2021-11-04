# -*- coding: utf-8 -*-
"""
CAM16UCS color appearance difference space
==========================================

 :_CAM_UCS_PARAMETERS: | dictionary with parameters specifying the conversion 
                       | from ciecam16 to:
                       |    - cam16ucs (uniform color space), 
                       |    - cam16lcd (large color diff.), 
                       |    - cam16scd (small color diff).

 :_DEFAULT_WHITE_POINT: Default internal reference white point (xyz)

 :_DEFAULT_CONDITIONS: Default CAM model parameters 

 :_AXES: dict with list[str,str,str] containing axis labels of defined cspaces.

 :run(): Run the CAM16-UCS[,-LCD,-SDC] color appearance difference model in forward or backward modes.
 
 :cam16ucs(): Run the CAM16-UCS[,-LCD,-SDC] color appearance difference model in forward or backward modes.
     
 :specific_wrappers_in_the_'xyz_to_cspace()' and 'cpsace_to_xyz()' format:
      | 'xyz_to_jab_cam16ucs', 'jab_cam16ucs_to_xyz', 
      | 'xyz_to_jab_cam16lcd', 'jab_cam16lcd_to_xyz',
      | 'xyz_to_jab_cam16scd', 'jab_cam16scd_to_xyz', 

Created on Wed Sep 30 22:15:57 2020

@author: ksmet1977 at gmail.com
"""
from luxpy.utils import np, asplit, ajoin
from luxpy.color.cam.ciecam16 import run as ciecam16


__all__ = ['run','cam16ucs','_CAM_UCS_PARAMETERS', 
           '_DEFAULT_WHITE_POINT','_DEFAULT_WHITE_POINT','_AXES']

__all__ += ['xyz_to_jab_cam16ucs', 'jab_cam16ucs_to_xyz', 
            'xyz_to_jab_cam16lcd', 'jab_cam16lcd_to_xyz',
            'xyz_to_jab_cam16scd', 'jab_cam16scd_to_xyz', 
            ]

_CAM_UCS_PARAMETERS = {'none':{'KL': 1.0, 'c1':0,'c2':0},
                   'ucs': {'KL': 1.0, 'c1':0.007,'c2':0.0228},
                   'lcd': {'KL': 0.77, 'c1':0.007,'c2':0.0053},
                   'scd': {'KL': 1.24, 'c1':0.007,'c2':0.0363}}

_AXES = {'jab_cam16ucs' :  ["J' (cam16ucs)", "a' (cam16ucs)", "b' (cam16ucs)"]} 
_AXES['jab_cam16lcd'] = ["J' (cam16lcd)", "a' (cam16lcd)", "b' (cam16lcd)"] 
_AXES['jab_cam16scd'] = ["J' (cam16scd)", "a' (cam16scd)", "b' (cam16scd)"] 

_DEFAULT_CONDITIONS = {'La': 100.0, 'Yb': 20.0, 'surround': 'avg','D': 1.0, 'Dtype': None}

_DEFAULT_WHITE_POINT = np.array([[100.0,100.0,100.0]])

def run(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None, conditions = None, 
        naka_rushton_parameters = None, unique_hue_data = None,
        ucstype = 'ucs', forward = True, mcat = 'cat16'):
    """ 
    Run the CAM16-UCS[,-LCD,-SDC] color appearance difference model in forward or backward modes.
    
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
            | For more info see luxpy.cam.ciecam16()?
        :naka_rushton_parameters:
            | None, optional
            | If None: use _NAKA_RUSHTON_PARAMETERS
        :unique_hue_data:
            | None, optional
            | If None: use _UNIQUE_HUE_DATA
        :ucstype:
            | 'ucs', optional
            | String with type of color difference appearance space
            | options: 'ucs', 'scd', 'lcd'
        :forward:
            | True, optional
            | If True: run in CAM in forward mode, else: inverse mode.
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
            | ndarray with J'a'b' coordinates (forward mode) 
            |  or 
            | XYZ tristimulus values (inverse mode)
    
    References:
        1. `M.R. Luo, G. Cui, and C. Li, 
        'Uniform colour spaces based on CIECAM02 colour appearance model,' 
        Color Res. Appl., vol. 31, no. 4, pp. 320–330, 2006.
        <http://onlinelibrary.wiley.com/doi/10.1002/col.20227/abstract)>`_
    """
    # get ucs parameters:
    if isinstance(ucstype,str):
        ucs_pars = _CAM_UCS_PARAMETERS
        ucs = ucs_pars[ucstype]
    else:
        ucs = ucstype
    KL, c1, c2 =  ucs['KL'], ucs['c1'], ucs['c2']
    
    # set conditions to use in CIECAM16 (overrides None-default in ciecam16() !!!)
    if conditions is None:
        conditions = _DEFAULT_CONDITIONS
    
    if forward == True:
        
        # run ciecam16 to get JMh:
        data = ciecam16(data, xyzw, outin = 'J,M,h', conditions = conditions,
                        naka_rushton_parameters = naka_rushton_parameters,
                        unique_hue_data = unique_hue_data,
                        forward = True, mcat = mcat) 
        
        camout = np.zeros_like(data) # for output

        #--------------------------------------------
        # convert to cam16ucs J', aM', bM':
        camout[...,0] = (1.0 + 100.0*c1)*data[...,0] / (1.0 + c1*data[...,0])
        Mp = ((1.0/c2) * np.log(1.0 + c2*data[...,1])) if (c2 != 0) else data[...,1]
        camout[...,1] = Mp * np.cos(data[...,2]*np.pi/180)
        camout[...,2] = Mp * np.sin(data[...,2]*np.pi/180)
        
        return camout
    
    else:
        #--------------------------------------------
        # convert cam16ucs J', aM', bM' to xyz:
            
        # calc ciecam16 hue angle
        #Jp, aMp, bMp = asplit(data)
        h = np.arctan2(data[...,2],data[...,1])

        # calc cam16ycs and ciecam16 colourfulness
        Mp = (data[...,1]**2.0 + data[...,2]**2.0)**0.5
        M = ((np.exp(c2*Mp) - 1.0) / c2) if (c2 != 0) else Mp
        
        # calculate ciecam16 aM, bM:
        aM = M * np.cos(h)
        bM = M * np.sin(h)

        # calc ciecam16 lightness
        J = data[...,0] / (1.0 + (100.0 - data[...,0]) * c1)

        
         # run ciecam16 in inverse mode to get xyz:
        return ciecam16(ajoin((J,aM,bM)), xyzw, outin = 'J,aM,bM', 
                        conditions = conditions, 
                        naka_rushton_parameters = naka_rushton_parameters,
                        unique_hue_data = unique_hue_data,
                        forward = False, mcat = mcat) 

#------------------------------------------------------------------------------
# wrapper functions for use with colortf():
cam16ucs = run
def xyz_to_jab_cam16ucs(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                        conditions = None, naka_rushton_parameters = None,
                        unique_hue_data = None,
                        mcat = 'cat16', **kwargs):
    """
    Wrapper function for cam16ucs forward mode with J,aM,bM output.
    
    | For help on parameter details: ?luxpy.cam.cam16ucs 
    """
    return cam16ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, 
                    naka_rushton_parameters = naka_rushton_parameters,
                    unique_hue_data = unique_hue_data,
                    forward = True, ucstype = 'ucs', mcat = mcat)
                
def jab_cam16ucs_to_xyz(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                        conditions = None, naka_rushton_parameters = None,
                        unique_hue_data = None,
                        mcat = 'cat16', **kwargs):
    """
    Wrapper function for cam16ucs inverse mode with J,aM,bM input.
    
    | For help on parameter details: ?luxpy.cam.cam16ucs 
    """
    return cam16ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, 
                    naka_rushton_parameters = naka_rushton_parameters,
                    unique_hue_data = unique_hue_data,
                    forward = False, ucstype = 'ucs', mcat = mcat)



def xyz_to_jab_cam16lcd(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                        conditions = None, naka_rushton_parameters = None,
                        unique_hue_data = None,
                        mcat = 'cat16', **kwargs):
    """
    Wrapper function for cam16ucs forward mode with J,aMp,bMp output and ucstype = lcd.
    
    | For help on parameter details: ?luxpy.cam.cam16ucs 
    """
    return cam16ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, 
                    naka_rushton_parameters = naka_rushton_parameters,
                    unique_hue_data = unique_hue_data,
                    forward = True, ucstype = 'lcd', mcat = mcat)
                
def jab_cam16lcd_to_xyz(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                        conditions = None, naka_rushton_parameters = None, 
                        unique_hue_data = None,
                        mcat = 'cat16', **kwargs):
    """
    Wrapper function for cam16ucs inverse mode with J,aMp,bMp input and ucstype = lcd.
    
    | For help on parameter details: ?luxpy.cam.cam16ucs 
    """
    return cam16ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, 
                    naka_rushton_parameters = naka_rushton_parameters,
                    unique_hue_data = unique_hue_data,
                    forward = False, ucstype = 'lcd', mcat = mcat)



def xyz_to_jab_cam16scd(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                        conditions = None, naka_rushton_parameters = None,
                        unique_hue_data = None,
                        mcat = 'cat16', **kwargs):
    """
    Wrapper function for cam16ucs forward mode with J,aMp,bMp output and ucstype = scd.
    
    | For help on parameter details: ?luxpy.cam.cam16ucs 
    """
    return cam16ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, 
                    naka_rushton_parameters = naka_rushton_parameters,
                    unique_hue_data = unique_hue_data,
                    forward = True, ucstype = 'scd', mcat = mcat)
                
def jab_cam16scd_to_xyz(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                        conditions = None, naka_rushton_parameters = None,
                        unique_hue_data = None,
                        mcat = 'cat16', **kwargs):
    """
    Wrapper function for cam16ucs inverse mode with J,aMp,bMp input and ucstype = scd.
    
    | For help on parameter details: ?luxpy.cam.cam16ucs 
    """
    return cam16ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, 
                    naka_rushton_parameters = naka_rushton_parameters,
                    unique_hue_data = unique_hue_data,
                    forward = False, ucstype = 'scd', mcat = mcat)
        

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
    _cam_o = lambda xyz, xyzw, forward: lx.xyz_to_jab_cam16ucs(xyz,xyzw)

    xyz, xyzw = lx.spd_to_xyz(Ill1, cieobs = cieobs, relative = True, rfl = rflM, out = 2)
    jabch = _cam(xyz, xyzw = xyzw, forward = True)
    out_ = _cam_o(xyz, xyzw = xyzw, forward = True)

    plt.figure()
    plt.plot(jabch[...,1],jabch[...,2],'c.')
    plt.plot(out_[...,1],out_[...,2],'r.')
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