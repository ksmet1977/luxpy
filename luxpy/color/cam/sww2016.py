# -*- coding: utf-8 -*-
"""
Module with Smet, Webster and Whitehead 2016 CAM.
=================================================

 :_CAM_SWW16_AXES: dict with list[str,str,str] containing axis labels 
                   of defined cspaces.
                   
 :_CAM_SWW16_PARAMETERS: cam_sww16 model parameters.
 
 :cam_sww16(): A simple principled color appearance model based on a mapping 
               of the Munsell color system.

References:
    1. `Smet, K. A. G., Webster, M. A., & Whitehead, L. A. (2016). 
    A simple principled approach for modeling and understanding uniform color metrics. 
    Journal of the Optical Society of America A, 33(3), A319–A331. 
    <https://doi.org/10.1364/JOSAA.33.00A319>`_
    .. 
"""

from luxpy import np, math, _CIE_ILLUMINANTS, _MUNSELL, _CMF, np2d, put_args_in_db, spd_to_xyz, getwlr, cie_interp,asplit, ajoin

_CAM_SWW16_AXES = {'lab_cam_sww16' : ["L (lab_cam_sww16)", "a (lab_cam_sww16)", "b (lab_cam_sww16)"]}

_CAM_SWW16_PARAMETERS = {'JOSA': {'cLMS': [1.0,1.0,1.0], 'lms0': [4985.0,5032.0,4761.0] , 'Cc': 0.252, 'Cf': -0.4, 'clambda': [0.5, 0.5, 0.0], 'calpha': [1.0, -1.0, 0.0], 'cbeta': [0.5, 0.5, -1.0], 'cga1': [26.1, 34.0], 'cgb1': [6.76, 10.9], 'cga2': [0.587], 'cgb2': [-0.952], 'cl_int': [14.0,1.0], 'cab_int': [4.99,65.8], 'cab_out' : [-0.1,-1.0], 'Ccwb': None, 'Mxyz2lms': [[ 0.21701045,  0.83573367, -0.0435106 ],[-0.42997951,  1.2038895 ,  0.08621089],[ 0.,  0.,  0.46579234]]}}
_CAM_SWW16_PARAMETERS['best-fit-JOSA'] = {'cLMS': [1.0,1.0,1.0], 'lms0': [4208.0,  4447.0,  4199.0] , 'Cc': 0.243, 'Cf': -0.269, 'clambda': [0.5, 0.5, 0.0], 'calpha': [1.0, -1.0, 0.0], 'cbeta': [0.5, 0.5, -1.0], 'cga1': [22.38, 26.42], 'cgb1': [5.36, 9.61], 'cga2': [0.668], 'cgb2': [-1.214], 'cl_int': [15.0, 1.04], 'cab_int': [5.85,65.86], 'cab_out' : [-1.008,-1.037], 'Ccwb': 0.80, 'Mxyz2lms': [[ 0.21701045,  0.83573367, -0.0435106 ],[-0.42997951,  1.2038895 ,  0.08621089],[ 0.,  0.,  0.46579234]]}
_CAM_SWW16_PARAMETERS['best-fit-all-Munsell'] = {'cLMS': [1.0,1.0,1.0], 'lms0': [5405.0, 5617.0,  5520.0] , 'Cc': 0.206, 'Cf': -0.128, 'clambda': [0.5, 0.5, 0.0], 'calpha': [1.0, -1.0, 0.0], 'cbeta': [0.5, 0.5, -1.0], 'cga1': [38.26, 43.35], 'cgb1': [8.97, 16.18], 'cga2': [0.512], 'cgb2': [-0.896], 'cl_int': [19.3, 0.99], 'cab_int': [5.87,63.24], 'cab_out' : [-0.545,-0.978], 'Ccwb': 0.736, 'Mxyz2lms': [[ 0.21701045,  0.83573367, -0.0435106 ],[-0.42997951,  1.2038895 ,  0.08621089],[ 0.,  0.,  0.46579234]]}

__all__ = ['_CAM_SWW16_AXES','_CAM_SWW16_PARAMETERS','cam_sww16','xyz_to_lab_cam_sww16','lab_cam_sww16_to_xyz']


def _update_parameter_dict(args, parameters = None, cieobs = '2006_10', 
                          match_to_conversionmatrix_to_cieobs = True):
    """
    Get parameter dict and update with values in args dict. 
    Also replace the xyz-to-lms conversion matrix with the one corresponding 
    to cieobs and normalize it to illuminant E.
    """
    if parameters is None:
        parameters = _CAM_SWW16_PARAMETERS['JOSA']
    if isinstance(parameters,str):
        parameters = _CAM_SWW16_PARAMETERS[parameters]
    parameters = put_args_in_db(parameters,args)  #overwrite parameters with other (not-None) args input 
    if match_to_conversionmatrix_to_cieobs == True:
        parameters['Mxyz2lms'] = _CMF[cieobs]['M'].copy()
    parameters['Mxyz2lms'] = math.normalize_3x3_matrix(parameters['Mxyz2lms'], np.array([[1.0, 1.0, 1.0]])) # normalize matrix for xyz-> lms conversion to ill. E
    return parameters


def _setup_default_adaptation_field(dataw = None, Lw = 400,
                                   inputtype = 'xyz', relative = True,
                                   cieobs = '2006_10'):
    """
    Setup theh default illuminant C adaptation field with Lw = 400 cd/m² for selected CIE observer.
    
    Args:
        :dataw: 
            | None or ndarray, optional
            | Input tristimulus values or spectral data of white point.
            | None defaults to the use of CIE illuminant C.
        :Lw:
            | 400.0, optional
            | Luminance (cd/m²) of white point.
        :inputtype:
            | 'xyz' or 'spd', optional
            | Specifies the type of input: 
            |     tristimulus values or spectral data for the forward mode.
        :relative:
            | True or False, optional
            | True: xyz tristimulus values are relative (Yw = 100)
        :cieobs:
            | '2006_10', optional
            | CMF set to use to perform calculations where spectral data 
              is involved (inputtype == 'spd'; dataw = None)
            | Other options: see luxpy._CMF['types']
    
    Returns:
        :dataw:
            | Ndarray with default adaptation field data (spectral or xyz)
    """
    if (dataw is None):
        dataw = _CIE_ILLUMINANTS['C'].copy() # get illuminant C
        xyzw = spd_to_xyz(dataw, cieobs = cieobs,relative=False) # get abs. tristimulus values
        if relative == False: #input is expected to be absolute
            dataw[1:] = Lw*dataw[1:]/xyzw[:,1:2] # dataw = Lw*dataw # make absolute
        else:
            dataw = dataw # make relative (Y=100)
        if inputtype == 'xyz':
            dataw = spd_to_xyz(dataw, cieobs = cieobs, relative = relative)
    return dataw

def _massage_input_and_init_output(data, dataw, 
                                  inputtype = 'xyz', direction = 'forward'):
    """
    Redimension input data to ensure most they have the appropriate sizes for easy and efficient looping.
    |
    | 1. Convert data and dataw to atleast_2d ndarrays
    | 2. Make axis 1 of dataw have 'same' dimensions as data
    | 3. Make dataw have same lights source axis size as data
    | 4. Flip light source axis to axis=0 for efficient looping
    | 5. Initialize output array camout to 'same' shape as data
    
    Args:
        :data: 
            | ndarray with input tristimulus values 
            | or spectral data 
            | or input color appearance correlates
            | Can be of shape: (N [, xM], x 3), whereby: 
            | N refers to samples and M refers to light sources.
            | Note that for spectral input shape is (N x (M+1) x wl) 
        :dataw: 
            | None or ndarray, optional
            | Input tristimulus values or spectral data of white point.
            | None defaults to the use of CIE illuminant C.
        :inputtype:
            | 'xyz' or 'spd', optional
            | Specifies the type of input: 
            |     tristimulus values or spectral data for the forward mode.
        :direction:
            | 'forward' or 'inverse', optional
            |   -'forward': xyz -> cam
            |   -'inverse': cam -> xyz 
    """
    # Convert data and dataw to atleast_2d ndarrays:
    data = np2d(data).copy() # stimulus data (can be upto NxMx3 for xyz, or [N x (M+1) x wl] for spd))
    dataw = np2d(dataw).copy() # white point (can be upto Nx3 for xyz, or [(N+1) x wl] for spd)
    originalshape = data.shape # to restore output to same shape

    # Make axis 1 of dataw have 'same' dimensions as data:         
    if (data.ndim == 2): 
        data = np.expand_dims(data, axis = 1)  # add light source axis 1 
    
    # Flip light source dim to axis 0:
    data = np.transpose(data, axes = (1,0,2))
    
    dataw = np.expand_dims(dataw, axis = 1)  # add extra axis to move light source to axis 0 

    # Make dataw have same lights source dimension size as data:
    if inputtype == 'xyz': 
        if dataw.shape[0] == 1: 
            dataw = np.repeat(dataw,data.shape[0],axis=0)     
        if (data.shape[0] == 1) & (dataw.shape[0]>1): 
            data = np.repeat(data,dataw.shape[0],axis=0)     
    else:
        dataw = np.array([np.vstack((dataw[:1,0,:],dataw[i+1:i+2,0,:])) for i in range(dataw.shape[0]-1)])
        if (data.shape[0] == 1) & (dataw.shape[0]>1):
            data = np.repeat(data,dataw.shape[0],axis=0) 
        
    # Initialize output array:
    dshape = list((data).shape)
    dshape[-1] = 3 # requested number of correlates: l_int, a_int, b_int
    if (inputtype != 'xyz') & (direction == 'forward'):
        dshape[-2] = dshape[-2] - 1 # wavelength row doesn't count & only with forward can the input data be spectral
    camout = np.nan*np.ones(dshape)
    return data, dataw, camout, originalshape


def _massage_output_data_to_original_shape(camout, originalshape):
    """
    Massage output data to restore original shape of input.
    """
    # Flip light source dim back to axis 1:
    camout = np.transpose(camout, axes = (1,0,2))

    if len(originalshape) < 3:
        if camout.shape[1] == 1:
            camout = np.squeeze(camout,axis = 1)

    return camout

def _get_absolute_xyz_xyzw(data, dataw, i = 0, Lw= 400, direction = 'forward', 
                          cieobs = '2006_10', inputtype = 'xyz', relative = True):
    """
    Calculate absolute xyz tristimulus values of stimulus and white point 
    from spectral input or convert relative xyz values to absolute ones.
    
    Args:
        :data: 
            | ndarray with input tristimulus values 
            | or spectral data 
            | or input color appearance correlates
            | Can be of shape: (N [, xM], x 3), whereby: 
            | N refers to samples and M refers to light sources.
            | Note that for spectral input shape is (N x (M+1) x wl) 
        :dataw: 
            | None or ndarray, optional
            | Input tristimulus values or spectral data of white point.
            | None defaults to the use of CIE illuminant C.
        :i:
            | 0, optional
            | row number in data and dataw ndarrays.
        :Lw:
            | 400.0, optional
            | Luminance (cd/m²) of white point.
        :inputtype:
            | 'xyz' or 'spd', optional
            | Specifies the type of input: 
            |     tristimulus values or spectral data for the forward mode.
        :direction:
            | 'forward' or 'inverse', optional
            |   -'forward': xyz -> cam
            |   -'inverse': cam -> xyz 
        :relative:
            | True or False, optional
            | True: xyz tristimulus values are relative (Yw = 100)
        :cieobs:
            | '2006_10', optional
            | CMF set to use to perform calculations where spectral data 
              is involved (inputtype == 'spd'; dataw = None)
            | Other options: see luxpy._CMF['types']
    """
    xyzw_abs = None
    
    # Spectral input:
    if (inputtype != 'xyz'):    
        
        # make spectral data in `dataw` absolute:        
        if relative == True:
            xyzw_abs = spd_to_xyz(dataw[i], cieobs = cieobs, relative = False)
            dataw[i,1:,:] = Lw*dataw[i,1:,:]/xyzw_abs[0,1] 
        
        # Calculate absolute xyzw:
        xyzwi = spd_to_xyz(dataw[i], cieobs = cieobs, relative = False)

        # make spectral data in `data` absolute:
        if (direction == 'forward'): # no xyz data or spectra in data if == 'inverse'!!!
            if relative == True:
                data[i,1:,:] = Lw*data[i,1:,:]/xyzw_abs[0,1]
                
            # Calculate absolute xyz of test field:    
            xyzti = spd_to_xyz(data[i,...], cieobs = cieobs, relative = False) 
            
        else:
            xyzti = None
        
    # XYZ input:
    elif (inputtype == 'xyz'):

        # make xyz data in `dataw` absolute: 
        if relative == True: 
            xyzw_abs = dataw[i].copy()
            dataw[i] = Lw*dataw[i]/xyzw_abs[:,1]   
        xyzwi = dataw[i]

        if (direction == 'forward'):
            if relative == True:
                # make xyz data in `data` absolute: 
                data[i] = Lw*data[i]/xyzw_abs[:,1] # make absolute
            xyzti = data[i]
        else:
            xyzti = None # not needed in inverse model
        
    return xyzti, xyzwi, xyzw_abs
 

def cam_sww16(data, dataw = None, Yb = 20.0, Lw = 400.0, Ccwb = None,
              relative = True,  inputtype = 'xyz', direction = 'forward',
              parameters = None, cieobs = '2006_10',
              match_to_conversionmatrix_to_cieobs = True):
    """
    A simple principled color appearance model based on a mapping of 
    the Munsell color system.
    
    | This function implements the JOSA A (parameters = 'JOSA') published model. 
    
    Args:
        :data: 
            | ndarray with input tristimulus values 
            | or spectral data 
            | or input color appearance correlates
            | Can be of shape: (N [, xM], x 3), whereby: 
            | N refers to samples and M refers to light sources.
            | Note that for spectral input shape is (N x (M+1) x wl) 
        :dataw: 
            | None or ndarray, optional
            | Input tristimulus values or spectral data of white point.
            | None defaults to the use of CIE illuminant C.
        :Yb: 
            | 20.0, optional
            | Luminance factor of background (perfect white diffuser, Yw = 100)
        :Lw:
            | 400.0, optional
            | Luminance (cd/m²) of white point.
        :Ccwb:
            | None,  optional
            | Degree of cognitive adaptation (white point balancing)
            | If None: use [..,..] from parameters dict.
        :relative:
            | True or False, optional
            | True: xyz tristimulus values are relative (Yw = 100)
        :parameters:
            | None or str or dict, optional
            | Dict with model parameters.
            |    - None: defaults to luxpy.cam._CAM_SWW_2016_PARAMETERS['JOSA']
            |    - str: 'best-fit-JOSA' or 'best-fit-all-Munsell'
            |    - dict: user defined model parameters 
            |            (dict should have same structure)
        :inputtype:
            | 'xyz' or 'spd', optional
            | Specifies the type of input: 
            |     tristimulus values or spectral data for the forward mode.
        :direction:
            | 'forward' or 'inverse', optional
            |   -'forward': xyz -> cam_sww_2016
            |   -'inverse': cam_sww_2016 -> xyz 
        :cieobs:
            | '2006_10', optional
            | CMF set to use to perform calculations where spectral data 
              is involved (inputtype == 'spd'; dataw = None)
            | Other options: see luxpy._CMF['types']
        :match_to_conversionmatrix_to_cieobs:
            | When channging to a different CIE observer, change the xyz-to_lms
            | matrix to the one corresponding to that observer. If False: use 
            | the one set in parameters or _CAM_SWW16_PARAMETERS
    
    Returns:
        :returns: 
            | ndarray with color appearance correlates (:direction: == 'forward')
            |  or 
            | XYZ tristimulus values (:direction: == 'inverse')
    
    Notes:
        | This function implements the JOSA A (parameters = 'JOSA') 
          published model. 
        | With:
        |    1. A correction for the parameter 
        |         in Eq.4 of Fig. 11: 0.952 --> -0.952 
        |         
        |     2. The delta_ac and delta_bc white-balance shifts in Eq. 5e & 5f 
        |         should be: -0.028 & 0.821 
        |  
        |     (cfr. Ccwb = 0.66 in: 
        |         ab_test_out = ab_test_int - Ccwb*ab_gray_adaptation_field_int))
             
    References:
        1. `Smet, K. A. G., Webster, M. A., & Whitehead, L. A. (2016). 
        A simple principled approach for modeling and understanding uniform color metrics. 
        Journal of the Optical Society of America A, 33(3), A319–A331. 
        <https://doi.org/10.1364/JOSAA.33.00A319>`_

    """
    #--------------------------------------------------------------------------
    # Get model parameters:
    #--------------------------------------------------------------------------
    args = locals().copy() 
    parameters = _update_parameter_dict(args, parameters = parameters,
                                       match_to_conversionmatrix_to_cieobs = match_to_conversionmatrix_to_cieobs)
      
    #unpack model parameters:
    Cc, Ccwb, Cf, Mxyz2lms, cLMS, cab_int, cab_out, calpha, cbeta,cga1, cga2, cgb1, cgb2, cl_int, clambda, lms0  = [parameters[x] for x in sorted(parameters.keys())]


    #--------------------------------------------------------------------------
    # Setup default adaptation field:   
    #--------------------------------------------------------------------------
    dataw = _setup_default_adaptation_field(dataw = dataw, Lw = Lw,
                                           inputtype = inputtype, relative = relative,
                                           cieobs = cieobs)

    #--------------------------------------------------------------------------
    # Redimension input data to ensure most appropriate sizes 
    # for easy and efficient looping and initialize output array:
    #--------------------------------------------------------------------------
    data, dataw, camout, originalshape = _massage_input_and_init_output(data, dataw, 
                                                                       inputtype = inputtype, 
                                                                       direction = direction)
    
    
    #--------------------------------------------------------------------------
    # Do precomputations needed for both the forward and inverse model,
    # and which do not depend on sample or light source data:
    #--------------------------------------------------------------------------
    Mxyz2lms = np.dot(np.diag(cLMS),Mxyz2lms) # weight the xyz-to-lms conversion matrix with cLMS (cfr. stage 1 calculations)   
    invMxyz2lms = np.linalg.inv(Mxyz2lms) # Calculate the inverse lms-to-xyz conversion matrix
    MAab = np.array([clambda,calpha,cbeta]) # Create matrix with scale factors for L, M, S for quick matrix multiplications
    invMAab = np.linalg.inv(MAab) # Pre-calculate its inverse to avoid repeat in loop.


    #--------------------------------------------------------------------------
    # Apply forward/inverse model by looping over each row (=light source dim.)
    # in data:
    #--------------------------------------------------------------------------
    N = data.shape[0]
    for i in range(N):
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  START FORWARD MODE and common part of inverse mode
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        #-----------------------------------------------------------------------------
        # Get absolute tristimulus values for stimulus field and white point for row i:
        #-----------------------------------------------------------------------------
        xyzt, xyzw, xyzw_abs = _get_absolute_xyz_xyzw(data, dataw, i = i, Lw = Lw, direction = direction, 
                                           cieobs = cieobs, inputtype = inputtype, relative = relative)
        

        #-----------------------------------------------------------------------------
        # stage 1: calculate photon rates of stimulus and white white, and
        # adapting field: i.e. lmst, lmsw and lmsf
        #-----------------------------------------------------------------------------
        # Convert to white point l,m,s:
        lmsw = 683.0*np.dot(Mxyz2lms,xyzw.T).T /_CMF[cieobs]['K']
        
        # Calculate adaptation field and convert to l,m,s:
        lmsf = (Yb/100.0)*lmsw 
        
        # Calculate lms of stimulus 
        # or put adaptation lmsf in test field lmst for later use in inverse-mode (no xyz in 'inverse' mode!!!):
        lmst = (683.0*np.dot(Mxyz2lms,xyzt.T).T /_CMF[cieobs]['K']) if (direction == 'forward') else lmsf


        #-----------------------------------------------------------------------------
        # stage 2: calculate cone outputs of stimulus lmstp
        #-----------------------------------------------------------------------------
        lmstp = math.erf(Cc*(np.log(lmst/lms0) + Cf*np.log(lmsf/lms0))) # stimulus test field
        lmsfp = math.erf(Cc*(np.log(lmsf/lms0) + Cf*np.log(lmsf/lms0))) # adaptation field

        
        # add adaptation field lms temporarily to lmstp for quick calculation
        lmstp = np.vstack((lmsfp,lmstp)) 
        
        
        #-----------------------------------------------------------------------------
        # stage 3: calculate optic nerve signals, lam*, alphp, betp:
        #-----------------------------------------------------------------------------
        lstar, alph, bet = asplit(np.dot(MAab, lmstp.T).T)

        alphp = cga1[0]*alph
        alphp[alph<0] = cga1[1]*alph[alph<0]
        betp = cgb1[0]*bet
        betp[bet<0] = cgb1[1]*bet[bet<0]

        
        #-----------------------------------------------------------------------------
        #  stage 4: calculate recoded nerve signals, alphapp, betapp:
        #-----------------------------------------------------------------------------
        alphpp = cga2[0]*(alphp + betp)
        betpp = cgb2[0]*(alphp - betp)


        #-----------------------------------------------------------------------------
        #  stage 5: calculate conscious color perception:
        #-----------------------------------------------------------------------------
        lstar_int = cl_int[0]*(lstar + cl_int[1])
        alph_int = cab_int[0]*(np.cos(cab_int[1]*np.pi/180.0)*alphpp - np.sin(cab_int[1]*np.pi/180.0)*betpp)
        bet_int = cab_int[0]*(np.sin(cab_int[1]*np.pi/180.0)*alphpp + np.cos(cab_int[1]*np.pi/180.0)*betpp)
        lstar_out = lstar_int


        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  stage 5 continued but SPLIT IN FORWARD AND INVERSE MODES:
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        #--------------------------------------
        # FORWARD MODE TO PERCEPTUAL SIGNALS:
        #--------------------------------------
        if direction == 'forward':
            if Ccwb is None:
                alph_out = alph_int - cab_out[0]
                bet_out = bet_int -  cab_out[1]
                
                
            else:
                Ccwb = Ccwb*np.ones((2))
                Ccwb[Ccwb<0.0] = 0.0
                Ccwb[Ccwb>1.0] = 1.0
                
                # white balance shift using adaptation gray background (Yb=20%), with Ccw: degree of adaptation:
                alph_out = alph_int - Ccwb[0]*alph_int[0] 
                bet_out = bet_int -  Ccwb[1]*bet_int[0]

            # stack together and remove adaptation field from vertical stack
            # camout is an ndarray with perceptual signals:
            camout[i] = np.vstack((lstar_out[1:],alph_out[1:],bet_out[1:])).T 
        
        
        #--------------------------------------
        # INVERSE MODE FROM PERCEPTUAL SIGNALS:
        #--------------------------------------    
        elif direction == 'inverse':
             
            # stack cognitive pre-adapted adaptation field signals (first on stack) together:
            labf_int = np.hstack((lstar_int[0],alph_int[0],bet_int[0]))
            
            # get lstar_out, alph_out & bet_out for data 
            #(contains model perceptual signals in inverse mode!!!):
            lstar_out, alph_out, bet_out = asplit(data[i])

            #------------------------------------------------------------------------
            #  Inverse stage 5: undo cortical white-balance:
            #------------------------------------------------------------------------
            if Ccwb is None:
                alph_int = alph_out + cab_out[0]
                bet_int = bet_out +  cab_out[1]
            else:
                Ccwb = Ccwb*np.ones((2))
                Ccwb[Ccwb<0.0] = 0.0
                Ccwb[Ccwb>1.0] = 1.0
                
                #  inverse white balance shift using adaptation gray background (Yb=20%), with Ccw: degree of adaptation
                alph_int = alph_out + Ccwb[0]*alph_int[0]
                bet_int = bet_out +  Ccwb[1]*bet_int[0]
            

            alphpp = (1.0 / cab_int[0]) * (np.cos(-cab_int[1]*np.pi/180.0)*alph_int - np.sin(-cab_int[1]*np.pi/180.0)*bet_int)
            betpp = (1.0 / cab_int[0]) * (np.sin(-cab_int[1]*np.pi/180.0)*alph_int + np.cos(-cab_int[1]*np.pi/180.0)*bet_int)
            lstar_int = lstar_out
            lstar = (lstar_int /cl_int[0]) - cl_int[1] 

            
            #---------------------------------------------------------------------------
            #  Inverse stage 4: pre-adapted perceptual signals to recoded nerve signals:
            #---------------------------------------------------------------------------
            alphp = 0.5*(alphpp/cga2[0] + betpp/cgb2[0])  # <-- alphpp = (Cga2.*(alphp+betp));
            betp = 0.5*(alphpp/cga2[0] - betpp/cgb2[0]) # <-- betpp = (Cgb2.*(alphp-betp));


            #---------------------------------------------------------------------------
            #  Inverse stage 3: recoded nerve signals to optic nerve signals:
            #---------------------------------------------------------------------------
            alph = alphp/cga1[0]
            bet = betp/cgb1[0]
            sa = np.sign(cga1[1])
            sb = np.sign(cgb1[1])
            alph[(sa*alphp)<0.0] = alphp[(sa*alphp)<0] / cga1[1] 
            bet[(sb*betp)<0.0] = betp[(sb*betp)<0] / cgb1[1] 
            lab = ajoin((lstar, alph, bet))
            

            #---------------------------------------------------------------------------
            #  Inverse stage 2: optic nerve signals to cone outputs:
            #---------------------------------------------------------------------------
            lmstp = np.dot(invMAab,lab.T).T 
            lmstp[lmstp<-1.0] = -1.0
            lmstp[lmstp>1.0] = 1.0


            #---------------------------------------------------------------------------
            #  Inverse stage 1: cone outputs to photon rates:
            #---------------------------------------------------------------------------

            lmstp = math.erfinv(lmstp) / Cc - Cf*np.log(lmsf/lms0)
            lmst = np.exp(lmstp) * lms0

            #---------------------------------------------------------------------------
            #  Photon rates to absolute or relative tristimulus values:
            #---------------------------------------------------------------------------
            xyzt =  np.dot(invMxyz2lms,lmst.T).T  *(_CMF[cieobs]['K']/683.0)
            if relative == True:
                xyzt = (100/Lw) * xyzt

            # store in same named variable as forward mode:
            camout[i] = xyzt
    
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #  END inverse mode 
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    return _massage_output_data_to_original_shape(camout, originalshape)
        
#------------------------------------------------------------------------------
def xyz_to_lab_cam_sww16(xyz, xyzw = None, Yb = 20.0, Lw = 400.0, Ccwb = None, relative = True,\
                         parameters = None, inputtype = 'xyz', cieobs = '2006_10', **kwargs):
    """
    Wrapper function for cam_sww16 forward mode with 'xyz' input.
    
    | For help on parameter details: ?luxpy.cam.cam_sww16
    """
    return cam_sww16(xyz, dataw = xyzw, Yb = Yb, Lw = Lw, relative = relative, parameters = parameters, inputtype = 'xyz', direction = 'forward', cieobs = cieobs)
                
def lab_cam_sww16_to_xyz(lab, xyzw = None, Yb = 20.0, Lw = 400.0, Ccwb = None, relative = True, \
                         parameters = None, inputtype = 'xyz', cieobs = '2006_10', **kwargs):
    """
    Wrapper function for cam_sww16 inverse mode with 'xyz' input.
    
    | For help on parameter details: ?luxpy.cam.cam_sww16
    """
    return cam_sww16(lab, dataw = xyzw, Yb = Yb, Lw = Lw, relative = relative, parameters = parameters, inputtype = 'xyz', direction = 'inverse', cieobs = cieobs)





#------------------------------------------------------------------------------
def test_model():

    import pandas as pd
    import luxpy as lx

    # Read selected set of Munsell samples and LMS10(lambda):
    M = pd.read_csv('Munsell_LMS_nonlin_Nov18_2015_version.dat',header=None,sep='\t').values
    YLMS10_ = pd.read_csv('YLMS10_LMS_nonlin_Nov18_2015_version.dat',header=None,sep='\t').values
    Y10_ = YLMS10_[[0,1],:].copy()
    LMS10_ = YLMS10_[[0,2,3,4],:].copy()
    
    # Calculate lms:
    Y10 = cie_interp(_CMF['1964_10']['bar'].copy(),getwlr([400,700,5]),kind='cmf')[[0,2],:]
    XYZ10_lx = _CMF['2006_10']['bar'].copy()
    XYZ10_lx = cie_interp(XYZ10_lx,getwlr([400,700,5]),kind='cmf')
    LMS10_lx = np.vstack((XYZ10_lx[:1,:],np.dot(math.normalize_3x3_matrix(_CMF['2006_10']['M'],np.array([[1,1,1]])),XYZ10_lx[1:,:])))
    LMS10 = cie_interp(LMS10_lx,getwlr([400,700,5]),kind='cmf')
    
    #LMS10 = np.vstack((XYZ10[:1,:],np.dot(lx.math.normalize_3x3_matrix(_CMF['2006_10']['M'],np.array([[1,1,1]])),XYZ10_lx[1:,:])))

    #LMS10[1:,:] = LMS10[1:,:]/LMS10[1:,:].sum(axis=1,keepdims=True)*Y10[1:,:].sum() 
    
    # test python model vs excel calculator:
    def spdBB(CCT = 5500, wl = [400,700,5], Lw = 25000, cieobs = '1964_10'):
        wl = getwlr(wl)
        dl = wl[1] - wl[0]
        spd = 2*np.pi*6.626068E-34*(299792458**2)/((wl*0.000000001)**5)/(np.exp(6.626068E-34*299792458/(wl*0.000000001)/1.3806503E-23/CCT)-1)
        spd = Lw*spd/(dl*683*(spd*cie_interp(_CMF[cieobs]['bar'].copy(),wl,kind='cmf')[2,:]).sum())
        return np.vstack((wl,spd))
    
    # Create long term and applied spds:
    spd5500 = spdBB(5500, Lw = 25000, wl = [400,700,5], cieobs = '1964_10')
    spd6500 = spdBB(6500, Lw = 400, wl = [400,700,5], cieobs = '1964_10')
    
    # Calculate lms0 as a check:
    clms = np.array([0.98446776, 0.98401909, 0.98571412]) # correction factor for slight differences in _CMF and the cmfs from the excel calculator
    lms0 = 5*683*(spd5500[1:]*LMS10[1:,:]*0.2).sum(axis=1).T
    
    
    # Full excel parameters for testing:
    parameters = {'cLMS':np.array([1,1,1]), 'lms0': np.array([4985.02802565,5032.49518502,4761.27272226])*1,
                   'Cc': 0.251617118325755, 'Cf': -0.4, 'clambda': [0.5, 0.5, 0.0], 
                   'calpha': [1.0, -1.0, 0.0], 'cbeta': [0.5, 0.5, -1.0], 
                   'cga1': [26.1047711317923, 33.9721745703298], 'cgb1': [6.76038379211498, 10.9220216677629], 
                   'cga2': [0.587271269247578], 'cgb2': [-0.952412544980473], 
                   'cl_int': [14.0035243121804,1.0], 'cab_int': [4.99218965716342,65.7869547646456], 
                   'cab_out' : [-0.1,-1.0], 'Ccwb': None, 
                   'Mxyz2lms': [[ 0.21701045,  0.83573367, -0.0435106 ],
                                [-0.42997951,  1.2038895 ,  0.08621089],
                                [ 0.,  0.,  0.46579234]]}
                   
    # Note cLMS is a relative scaling factor between CIE2006 10° and 1964 10°:
#    clms = np.array([1.00164919, 1.00119269, 1.0029173 ]) = (Y10[1:,:].sum(axis=1)/LMS10[1:,:].sum(axis=1))*(406.98099078/400)
                    
    #parameters =_CAM_SWW16_PARAMETERS['JOSA']
    # Calculate Munsell spectra multiplied with spd6500:
    spd6500xM = np.vstack((spd6500[:1,:],spd6500[1:,:]*M[1:,:]))
               
    # Test spectral input:
    print('SPD INPUT -----')
    jab = cam_sww16(spd6500xM, dataw = spd6500, Yb = 20.0, Lw = 400.0, Ccwb = 1,
                      relative = True,  inputtype = 'spd', direction = 'forward',
                      parameters = parameters, cieobs = '2006_10',
                      match_to_conversionmatrix_to_cieobs = True)
    
#    # Test xyz input:
    print('\nXYZ INPUT -----')
    xyz = lx.spd_to_xyz(spd6500xM,cieobs='2006_10',relative=False)
    xyzw = lx.spd_to_xyz(spd6500,cieobs='2006_10',relative=False)
    xyz2,xyzw2 = lx.spd_to_xyz(spd6500,cieobs='2006_10',relative=False,rfl=M,out=2)
     

    print(xyzw)
    jab = cam_sww16(xyz, dataw = xyzw, Yb = 20.0, Lw = 400, Ccwb = 1,
                      relative = True,  inputtype = 'xyz', direction = 'forward',
                      parameters = parameters, cieobs = '2006_10',
                      match_to_conversionmatrix_to_cieobs = True)

#------------------------------------------------------------------------------
if __name__ == '__main__0':
    test_model()
    
if __name__ == '__main__':
    
    C = _CIE_ILLUMINANTS['C'].copy()
    C = np.vstack((C,cie_interp(_CIE_ILLUMINANTS['D65'],C[0],kind='spd')[1:],C[1:,:]*2,C[1:,:]*3))
    M = _MUNSELL.copy()
    rflM = M['R']
    rflM = cie_interp(rflM,C[0],kind='rfl')
    cieobs = '2006_10'
    Lw = 400
    Yb = 20
    
    # Normalize to Lw:
    xyzw2 = spd_to_xyz(C, cieobs = cieobs, relative = False)
    for i in range(C.shape[0]-1):
        C[i+1] = Lw*C[i+1]/xyzw2[i,1]
    CM = []
    for i in range(C.shape[0]-1):
        CM.append(np.vstack((C[0],C[i+1]*rflM[1:,:])))
    CM = np.transpose(np.array(CM),(1,0,2))
    
    xyz, xyzw = spd_to_xyz(C, cieobs = cieobs, relative = True, rfl = rflM, out = 2)
    xyz = xyz[:4,0,:]
    CM = CM[:5,0,:]
    #xyzw = np.vstack((xyzw[:1,:],xyzw[:1,:]))
    xyzw = xyzw[:1,...]
    C = C[:2,:]

    print('xyz in:')
    lab = cam_sww16(xyz, dataw = xyzw, Yb = Yb, Lw = Lw, Ccwb = 1, relative = True, \
              parameters = None, inputtype = 'xyz', direction = 'forward', \
              cieobs = cieobs)
    print(lab)
    
    print('spd in:')
    lab2 = cam_sww16(CM, dataw = C[:2,:], Yb = Yb, Lw = Lw, Ccwb = 1, relative = True, \
              parameters = None, inputtype = 'spd', direction = 'forward', \
              cieobs = cieobs)
    print(lab2)
    
    print('inverse xyz in')
    xyz_ = cam_sww16(lab, dataw = xyzw, Yb = Yb, Lw = Lw, Ccwb = 1, relative = True, \
              parameters = None, inputtype = 'xyz', direction = 'inverse', \
              cieobs = cieobs)
    print(xyz_)
    
    print('inverse spd in')
    xyz_2 = cam_sww16(lab2, dataw = C[:2,:], Yb = Yb, Lw = Lw, Ccwb = 1, relative = True, \
              parameters = None, inputtype = 'spd', direction = 'inverse', \
              cieobs = cieobs)
    print(xyz_2)
    
    print('\ndiff xyz in: ', xyz-xyz_)
    print((xyz-xyz_)/xyz)
    print('diff spd in: ', xyz-xyz_2,(xyz-xyz_2)/xyz)
