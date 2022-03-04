# -*- coding: utf-8 -*-
"""
CAM02UCS color appearance difference space
==========================================

 :_CAM_UCS_PARAMETERS: | dictionary with parameters specifying the conversion 
                       | from ciecam02 to:
                       |    - cam02ucs (uniform color space), 
                       |    - cam02lcd (large color diff.), 
                       |    - cam02scd (small color diff).

 :_DEFAULT_WHITE_POINT: Default internal reference white point (xyz)

 :_DEFAULT_CONDITIONS: Default CAM model parameters 

 :_AXES: dict with list[str,str,str] containing axis labels of defined cspaces.

 :run(): Run the CAM02-UCS[,-LCD,-SDC] color appearance difference model in forward or backward modes.
 
 :cam02ucs(): Run the CAM02-UCS[,-LCD,-SDC] color appearance difference model in forward or backward modes.
     
 :specific_wrappers_in_the_'xyz_to_cspace()' and 'cpsace_to_xyz()' format:
      | 'xyz_to_jab_cam02ucs', 'jab_cam02ucs_to_xyz', 
      | 'xyz_to_jab_cam02lcd', 'jab_cam02lcd_to_xyz',
      | 'xyz_to_jab_cam02scd', 'jab_cam02scd_to_xyz', 

Created on Wed Sep 30 14:35:05 2020

@author: ksmet1977 at gmail.com
"""
from luxpy.utils import np, asplit, ajoin
from luxpy.color.cam.ciecam02 import run as ciecam02


__all__ = ['run','cam02ucs','_CAM_UCS_PARAMETERS', 
           '_DEFAULT_WHITE_POINT','_DEFAULT_WHITE_POINT','_AXES']

__all__ += ['xyz_to_jab_cam02ucs', 'jab_cam02ucs_to_xyz', 
            'xyz_to_jab_cam02lcd', 'jab_cam02lcd_to_xyz',
            'xyz_to_jab_cam02scd', 'jab_cam02scd_to_xyz', 
            ]

_CAM_UCS_PARAMETERS = {'none':{'KL': 1.0, 'c1':0,'c2':0},
                   'ucs':{'KL': 1.0, 'c1':0.007,'c2':0.0228},
                   'lcd':{'KL': 0.77, 'c1':0.007,'c2':0.0053},
                   'scd':{'KL': 1.24, 'c1':0.007,'c2':0.0363}}

_AXES = {'jab_cam02ucs' :  ["J' (cam02ucs)", "a' (cam02ucs)", "b' (cam02ucs)"]} 
_AXES['jab_cam02lcd'] = ["J' (cam02lcd)", "a' (cam02lcd)", "b' (cam02lcd)"] 
_AXES['jab_cam02scd'] = ["J' (cam02scd)", "a' (cam02scd)", "b' (cam02scd)"] 

_DEFAULT_CONDITIONS = {'La': 100.0, 'Yb': 20.0, 'surround': 'avg','D': 1.0, 'Dtype': None}

_DEFAULT_WHITE_POINT = np.array([[100.0,100.0,100.0]])

def run(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None, conditions = None, 
        naka_rushton_parameters = None, unique_hue_data = None, 
        ucstype = 'ucs', forward = True,
        yellowbluepurplecorrect = False, mcat = 'cat02'):
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
        :yellowbluepurplecorrect:
            | False, optional
            | If False: don't correct for yellow-blue and purple problems in ciecam02. 
            | If 'brill-suss': 
            |       for yellow-blue problem, see: 
            |          - Brill [Color Res Appl, 2006; 31, 142-145] and 
            |          - Brill and Süsstrunk [Color Res Appl, 2008; 33, 424-426] 
            | If 'jiang-luo': 
            |       for yellow-blue problem + purple line problem, see:
            |          - Jiang, Jun et al. [Color Res Appl 2015: 40(5), 491-503] 
        :mcat:
            | 'cat02', optional
            | Specifies CAT sensor space.
            | - options:
            |    - None defaults to 'cat02' 
            |         (others e.g. 'cat02-bs', 'cat02-jiang',
            |         all trying to correct gamut problems of original cat02 matrix)
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
    
    # set conditions to use in CIECAM02 (overrides None-default in ciecam02() !!!)
    if conditions is None:
        conditions = _DEFAULT_CONDITIONS
    
    if forward == True:
        
        # run ciecam02 to get JMh:
        data = ciecam02(data, xyzw, outin = 'J,M,h', conditions = conditions, 
                        forward = True, mcat = mcat, 
                        naka_rushton_parameters = naka_rushton_parameters,
                        unique_hue_data = unique_hue_data,
                        yellowbluepurplecorrect = yellowbluepurplecorrect) 
        
        camout = np.zeros_like(data) # for output

        #--------------------------------------------
        # convert to cam02ucs J', aM', bM':
        camout[...,0] = (1.0 + 100.0*c1)*data[...,0] / (1.0 + c1*data[...,0])
        Mp = ((1.0/c2) * np.log(1.0 + c2*data[...,1])) if (c2 != 0) else data[...,1]
        camout[...,1] = Mp * np.cos(data[...,2]*np.pi/180)
        camout[...,2] = Mp * np.sin(data[...,2]*np.pi/180)
        
        return camout
    
    else:
        #--------------------------------------------
        # convert cam02ucs J', aM', bM' to xyz:
            
        # calc ciecam02 hue angle
        #Jp, aMp, bMp = asplit(data)
        h = np.arctan2(data[...,2],data[...,1])

        # calc cam02ucs and CIECAM02 colourfulness
        Mp = (data[...,1]**2.0 + data[...,2]**2.0)**0.5
        M = ((np.exp(c2*Mp) - 1.0) / c2) if (c2 != 0) else Mp
        
        # calculate ciecam02 aM, bM:
        aM = M * np.cos(h)
        bM = M * np.sin(h)

        # calc ciecam02 lightness
        J = data[...,0] / (1.0 + (100.0 - data[...,0]) * c1)

        
         # run ciecam02 in inverse mode to get xyz:
        return ciecam02(ajoin((J,aM,bM)), xyzw, outin = 'J,aM,bM', 
                        conditions = conditions, 
                        naka_rushton_parameters = naka_rushton_parameters,
                        unique_hue_data = unique_hue_data,
                        forward = False, mcat = mcat,
                        yellowbluepurplecorrect = yellowbluepurplecorrect) 

#------------------------------------------------------------------------------
# wrapper functions for use with colortf():
cam02ucs = run
def xyz_to_jab_cam02ucs(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                        conditions = None, naka_rushton_parameters = None,
                        unique_hue_data = None,
                        yellowbluepurplecorrect = None, 
                        mcat = 'cat02', **kwargs):
    """
    Wrapper function for cam02ucs forward mode with J,aM,bM output.
    
    | For help on parameter details: ?luxpy.cam.cam02ucs 
    """
    return cam02ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, 
                    naka_rushton_parameters = naka_rushton_parameters,
                    unique_hue_data = unique_hue_data,
                    forward = True, ucstype = 'ucs', 
                    yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
                
def jab_cam02ucs_to_xyz(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                        conditions = None, naka_rushton_parameters = None,
                        unique_hue_data = None,
                        yellowbluepurplecorrect = None, 
                        mcat = 'cat02', **kwargs):
    """
    Wrapper function for cam02ucs inverse mode with J,aM,bM input.
    
    | For help on parameter details: ?luxpy.cam.cam02ucs 
    """
    return cam02ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, 
                    naka_rushton_parameters = naka_rushton_parameters,
                    unique_hue_data = unique_hue_data,
                    forward = False, ucstype = 'ucs', 
                    yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)



def xyz_to_jab_cam02lcd(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                        conditions = None, naka_rushton_parameters = None, 
                        unique_hue_data = None,
                        yellowbluepurplecorrect = None, 
                        mcat = 'cat02', **kwargs):
    """
    Wrapper function for cam02ucs forward mode with J,aMp,bMp output and ucstype = lcd.
    
    | For help on parameter details: ?luxpy.cam.cam02ucs 
    """
    return cam02ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, 
                    naka_rushton_parameters = naka_rushton_parameters,
                    unique_hue_data = unique_hue_data,
                    forward = True, ucstype = 'lcd', 
                    yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
                
def jab_cam02lcd_to_xyz(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                        conditions = None, naka_rushton_parameters = None,
                        unique_hue_data = None,
                        yellowbluepurplecorrect = None, 
                        mcat = 'cat02', **kwargs):
    """
    Wrapper function for cam02ucs inverse mode with J,aMp,bMp input and ucstype = lcd.
    
    | For help on parameter details: ?luxpy.cam.cam02ucs 
    """
    return cam02ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, 
                    naka_rushton_parameters = naka_rushton_parameters,
                    unique_hue_data = unique_hue_data,
                    forward = False, ucstype = 'lcd', 
                    yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)



def xyz_to_jab_cam02scd(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                        conditions = None, naka_rushton_parameters = None,
                        unique_hue_data = None,
                        yellowbluepurplecorrect = None, 
                        mcat = 'cat02', **kwargs):
    """
    Wrapper function for cam02ucs forward mode with J,aMp,bMp output and ucstype = scd.
    
    | For help on parameter details: ?luxpy.cam.cam02ucs 
    """
    return cam02ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions,
                    naka_rushton_parameters = naka_rushton_parameters,
                    unique_hue_data = unique_hue_data,
                    forward = True, ucstype = 'scd', 
                    yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
                
def jab_cam02scd_to_xyz(data, xyzw = _DEFAULT_WHITE_POINT, Yw = None,
                        conditions = None, naka_rushton_parameters = None,
                        unique_hue_data = None,
                        yellowbluepurplecorrect = None, 
                        mcat = 'cat02', **kwargs):
    """
    Wrapper function for cam02ucs inverse mode with J,aMp,bMp input and ucstype = scd.
    
    | For help on parameter details: ?luxpy.cam.cam02ucs 
    """
    return cam02ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, 
                    naka_rushton_parameters = naka_rushton_parameters,
                    unique_hue_data = unique_hue_data,
                    forward = False, ucstype = 'scd', 
                    yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
        

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
    _cam_o = lambda xyz, xyzw, forward: lx.xyz_to_jab_cam02ucs(xyz,xyzw)

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