# -*- coding: utf-8 -*-
"""
Module with CAM15u color appearance model
=========================================

 :_CAM_15U_AXES: dict with list[str,str,str] containing axis labels 
                  of defined cspaces.
                  
 :_CAM15U_PARAMETERS: database with CAM15u model parameters.
 
 :_CAM15U_UNIQUE_HUE_DATA: database of unique hues with corresponding 
                           Hue quadratures and eccentricity factors 
                           for ciecam02, cam16, ciecam97s, cam15u)

 :_CAM15U_SURROUND_PARAMETERS: database of surround param. c, Nc, F and FLL 
                               for cam15u.

 :_CAM15U_NAKA_RUSHTON_PARAMETERS: | database with parameters 
                                     (n, sig, scaling and noise) 
                                     for the Naka-Rushton function: 
                                   | scaling * ((data**n) / ((data**n) + (sig**n))) + noise
                                   
 :cam15u(): | calculates the output for the CAM15u model for self-luminous unrelated stimuli. 
            | `M. Withouck, K. A. G. Smet, W. R. Ryckaert, and P. Hanselaer, 
              “Experimental driven modelling of the color appearance of 
              unrelated self-luminous stimuli: CAM15u,” 
              Opt. Express, vol. 23, no. 9, pp. 12045–12064, 2015.
              <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-9-12045&origin=search>`_
            | `M. Withouck, K. A. G. Smet, and P. Hanselaer, (2015), 
            “Brightness prediction of different sized unrelated self-luminous stimuli,” 
            Opt. Express, vol. 23, no. 10, pp. 13455–13466. 
            <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-10-13455&origin=search>`_
"""
from luxpy import np, _CMF, _CIE_ILLUMINANTS, _MUNSELL, np2d, spd_to_xyz, asplit, ajoin,cie_interp
from luxpy.color.cam.colorappearancemodels import hue_angle, hue_quadrature

_CAM15U_AXES = {'qabW_cam15u' : ["Q (cam15u)", "aW (cam15u)", "bW (cam15u)"]} 

_CAM15U_UNIQUE_HUE_DATA = {'hues': 'red yellow green blue red'.split(), 'i': np.arange(5.0), 'hi':[20.14, 90.0, 164.25,237.53,380.14],'ei':[0.8,0.7,1.0,1.2,0.8],'Hi':[0.0,100.0,200.0,300.0,400.0]}

_CAM15U_PARAMETERS = {'k': [666.7, 782.3,1444.6],'cp': 1.0/3, 'cA':3.22 ,'cAlms':[2.0, 1.0, 1/20] ,'ca' : 1.0, 'calms':[1.0,-12/11,1/11],'cb': 0.117, 'cblms': [1.0, 1.0,-2.0], 'unique_hue_data':_CAM15U_UNIQUE_HUE_DATA, 'cM': 135.52, 'cHK': [2.559,0.561], 'cW': [2.29,2.68], 'cfov': 0.271, 'Mxyz2rgb': np.array([[0.211831, 0.815789, -0.042472],[-0.492493, 1.378921, 0.098745],[0.0, 0.0, 0.985188]])}

_CAM15U_NAKA_RUSHTON_PARAMETERS = {'n':None, 'sig': None, 'scaling': None, 'noise': None}

_CAM15U_SURROUND_PARAMETERS = {'surrounds': ['dark'], 'dark' : {'c': None, 'Nc':None,'F':None,'FLL':None}}

__all__ = ['cam15u','_CAM15U_AXES','_CAM15U_UNIQUE_HUE_DATA', '_CAM15U_PARAMETERS','_CAM15U_NAKA_RUSHTON_PARAMETERS', '_CAM15U_SURROUND_PARAMETERS']

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------            
def cam15u(data, fov = 10.0, inputtype = 'xyz', direction = 'forward', outin = 'Q,aW,bW', parameters = None):
    """
    Convert between CIE 2006 10°  XYZ tristimulus values (or spectral data) 
    and CAM15u color appearance correlates.
    
    Args:
        :data: 
            | ndarray of CIE 2006 10°  XYZ tristimulus values or spectral data
              or color appearance attributes
        :fov: 
            | 10.0, optional
            | Field-of-view of stimulus (for size effect on brightness)
        :inputtpe:
            | 'xyz' or 'spd', optional
            | Specifies the type of input: 
            |     tristimulus values or spectral data for the forward mode.
        :direction:
            | 'forward' or 'inverse', optional
            |   -'forward': xyz -> cam15u
            |   -'inverse': cam15u -> xyz 
        :outin:
            | 'Q,aW,bW' or str, optional
            | 'Q,aW,bW' (brightness and opponent signals for amount-of-neutral)
            |  other options: 'Q,aM,bM' (colorfulness) and 'Q,aS,bS' (saturation)
            | Str specifying the type of 
            |     input (:direction: == 'inverse') and 
            |     output (:direction: == 'forward')
        :parameters:
            | None or dict, optional
            | Set of model parameters.
            |   - None: defaults to luxpy.cam._CAM15U_PARAMETERS 
            |    (see references below)
    
    Returns:
        :returns: 
            | ndarray with color appearance correlates (:direction: == 'forward')
            |  or 
            | XYZ tristimulus values (:direction: == 'inverse')

    References: 
        1. `M. Withouck, K. A. G. Smet, W. R. Ryckaert, and P. Hanselaer, 
        “Experimental driven modelling of the color appearance of 
        unrelated self-luminous stimuli: CAM15u,” 
        Opt. Express, vol. 23, no. 9, pp. 12045–12064, 2015.
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-9-12045&origin=search>`_
        2. `M. Withouck, K. A. G. Smet, and P. Hanselaer, (2015), 
        “Brightness prediction of different sized unrelated self-luminous stimuli,” 
        Opt. Express, vol. 23, no. 10, pp. 13455–13466. 
        <https://www.osapublishing.org/oe/abstract.cfm?uri=oe-23-10-13455&origin=search>`_  
     """
    
    if parameters is None:
        parameters = _CAM15U_PARAMETERS
        
    outin = outin.split(',')    
        
    #unpack model parameters:
    Mxyz2rgb, cA, cAlms, cHK, cM, cW, ca, calms, cb, cblms, cfov, cp, k, unique_hue_data = [parameters[x] for x in sorted(parameters.keys())]
    
    
    # precomputations:
    invMxyz2rgb = np.linalg.inv(Mxyz2rgb)
    MAab = np.array([cAlms,calms,cblms])
    invMAab = np.linalg.inv(MAab)
    
     #initialize data and camout:
    data = np2d(data)
    if len(data.shape)==2:
        data = np.expand_dims(data, axis = 0) # avoid looping if not necessary
    
    if (data.shape[0] > data.shape[1]): # loop over shortest dim.
        flipaxis0and1 = True
        data = np.transpose(data, axes = (1,0,2))
    else:
        flipaxis0and1 = False

    
    dshape = list(data.shape)
    dshape[-1] = len(outin) # requested number of correlates
    if (inputtype != 'xyz') & (direction == 'forward'):
        dshape[-2] = dshape[-2] - 1 # wavelength row doesn't count & only with forward can the input data be spectral
    camout = np.nan*np.ones(dshape)

    for i in range(data.shape[0]):
        
        if (inputtype != 'xyz') & (direction == 'forward'):
            xyz = spd_to_xyz(data[i], cieobs = '2006_10', relative = False)
            lms = np.dot(_CMF['2006_10']['M'],xyz.T).T # convert to l,m,s
            rgb = (lms / _CMF['2006_10']['K']) * k # convert to rho, gamma, beta
        elif (inputtype == 'xyz') & (direction == 'forward'):
            rgb = np.dot(Mxyz2rgb,data[i].T).T

        if direction == 'forward':
            
            # apply cube-root compression:
            rgbc = rgb**(cp)
            
            # calculate achromatic and color difference signals, A, a, b:
            Aab = np.dot(MAab, rgbc.T).T
            A,a,b = asplit(Aab)
            A = cA*A
            a = ca*a
            b = cb*b

            # calculate colorfullness like signal M:
            M = cM*((a**2.0 + b**2.0)**0.5)

            # calculate brightness Q:
            Q = A + cHK[0]*M**cHK[1] # last term is contribution of Helmholtz-Kohlrausch effect on brightness
                      
            # calculate saturation, s:
            s = M / Q
            
            # calculate amount of white, W:
            W = 100.0 / (1.0 + cW[0]*(s**cW[1]))

            #  adjust Q for size (fov) of stimulus (matter of debate whether to do this before or after calculation of s or W, there was no data on s, M or W for different sized stimuli: after)
            Q = Q*(fov/10.0)**cfov
            
            # calculate hue, h and Hue quadrature, H:
            h = hue_angle(a,b, htype = 'deg')

            if 'H' in outin:
                H = hue_quadrature(h, unique_hue_data = unique_hue_data)
            else:
                H = None

            # calculate cart. co.:
            if 'aM' in outin:
                aM = M*np.cos(h*np.pi/180.0)
                bM = M*np.sin(h*np.pi/180.0)
            
            if 'aS' in outin:
                aS = s*np.cos(h*np.pi/180.0)
                bS = s*np.sin(h*np.pi/180.0)
            
            if 'aW' in outin:
                aW = W*np.cos(h*np.pi/180.0)
                bW = W*np.sin(h*np.pi/180.0)
            
    
            if (outin != ['Q','aW','bW']):
                camout[i] =  eval('ajoin(('+','.join(outin)+'))')
            else:
                camout[i] = ajoin((Q,aW,bW))
    
        
        elif direction == 'inverse':

            # get Q, M and a, b depending on input type:        
            if 'aW' in outin:
                Q,a,b = asplit(data[i])
                Q = Q / ((fov/10.0)**cfov) #adjust Q for size (fov) of stimulus back to that 10° ref
                W = (a**2.0 + b**2.0)**0.5
                s = (((100 / W) - 1.0)/cW[0])**(1.0/cW[1])
                M = s*Q
                
            
            if 'aM' in outin:
                Q,a,b = asplit(data[i])
                Q = Q / ((fov/10.0)**cfov) #adjust Q for size (fov) of stimulus back to that 10° ref
                M = (a**2.0 + b**2.0)**0.5
            
            if 'aS' in outin:
                Q,a,b = asplit(data[i])
                Q = Q / ((fov/10.0)**cfov) #adjust Q for size (fov) of stimulus back to that 10° ref
                s = (a**2.0 + b**2.0)**0.5
                M = s*Q
                      
            if 'h' in outin:
                Q, WsM, h = asplit(data[i])
                Q = Q / ((fov/10.0)**cfov) #adjust Q for size (fov) of stimulus back to that 10° ref
                if 'W' in outin:
                     s = (((100.0 / WsM) - 1.0)/cW[0])**(1.0/cW[1])
                     M = s*Q
                elif 's' in outin:
                     M = WsM*Q
                elif 'M' in outin:
                     M = WsM
            
            # calculate achromatic signal, A from Q and M:
            A = Q - cHK[0]*M**cHK[1]
            A = A/cA
            
            # calculate hue angle:
            h = hue_angle(a,b, htype = 'rad')
            
            # calculate a,b from M and h:
            a = (M/cM)*np.cos(h)
            b = (M/cM)*np.sin(h)
            a = a/ca
            b = b/cb

            # create Aab:
            Aab = ajoin((A,a,b))    
            
            # calculate rgbc:
            rgbc = np.dot(invMAab, Aab.T).T    
            
            # decompress rgbc to rgb:
            rgb = rgbc**(1/cp)
            
            
            # convert rgb to xyz:
            xyz = np.dot(invMxyz2rgb,rgb.T).T 
            
            camout[i] = xyz
    
    if flipaxis0and1 == True: # loop over shortest dim.
        camout = np.transpose(camout, axes = (1,0,2))

    
    if camout.shape[0] == 1:
        camout = np.squeeze(camout,axis = 0)
    
    return camout
 

#------------------------------------------------------------------------------
def xyz_to_qabW_cam15u(xyz, fov = 10.0, parameters = None, **kwargs):
    """
    Wrapper function for cam15u forward mode with 'Q,aW,bW' output.
    
    | For help on parameter details: ?luxpy.cam.cam15u
    """
    return cam15u(xyz, fov = fov, direction = 'forward', inputtype = 'xyz', outin = 'Q,aW,bW', parameters = parameters)
                
def qabW_cam15u_to_xyz(qab, fov = 10.0, parameters = None, **kwargs):
    """
    Wrapper function for cam15u inverse mode with 'Q,aW,bW' input.
    
    | For help on parameter details: ?luxpy.cam.cam15u
    """
    return cam15u(qab, fov = fov, direction = 'inverse', inputtype = 'xyz', outin = 'Q,aW,bW', parameters = parameters)


#------------------------------------------------------------------------------
if __name__ == '__main__':
    C = _CIE_ILLUMINANTS['C'].copy()
    C = np.vstack((C,cie_interp(_CIE_ILLUMINANTS['D65'],C[0],kind='spd')[1:]))
    M = _MUNSELL.copy()
    rflM = M['R']
    cieobs = '2006_10'
    
    # Normalize to Lw:
    Lw = 100
    xyzw2 = spd_to_xyz(C, cieobs = cieobs, relative = False)
    for i in range(C.shape[0]-1):
        C[i+1] = Lw*C[i+1]/xyzw2[i,1]

    
    xyz, xyzw = spd_to_xyz(C, cieobs = cieobs, relative = True, rfl = rflM, out = 2)
    qab = xyz_to_qabW_cam15u(xyzw, fov = 10.0)
    qab2 = cam15u(C, fov = 10.0, direction = 'forward', inputtype = 'spd', outin = 'Q,aW,bW', parameters = None)
           
    xyz_ = qabW_cam15u_to_xyz(qab, fov = 10.0)
    print(qab)
    print(qab2)
    print(xyzw-xyz_)
