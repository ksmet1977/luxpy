# -*- coding: utf-8 -*-
"""
# Helper fcns for CAM (absolute/relative, spd/xyz, lamp included, etc.)
=======================================================================

 :_update_parameter_dict(): Get parameter dict and update with values in args dict

 :_setup_default_adaptation_field(): Setup a default illuminant adaptation field with Lw = 100 cd/m² for selected CIE observer.

 :_massage_input_and_init_output(): Redimension input data to ensure most they have the appropriate sizes for easy and efficient looping.

 :_massage_output_data_to_original_shape(): Massage output data to restore original shape of original CAM input.
 
 :_get_absolute_xyz_xyzw(): Calculate absolute xyz tristimulus values of stimulus and white point from spectral input or convert relative xyz values to absolute ones.
 
 :_simple_cam(): An example CAM illustration the usage of the functions in luxpy.cam.helpers 
     

Notes:
    1. These functions help take care of some recurring steps in building 
    a CAM that allows for both xyz or spectral input, input that is eihter
    absolute or relative, or that has multiple inputs for the whitepoints.
    2. For example usage, see the code of _simple_cam() by typing:
    luxpy.cam._simple_cam??

Created on Wed Sep 30 09:35:37 2020

@author: ksmet1977@gmail.com
"""

from luxpy import (math, _CIE_ILLUMINANTS, _CMF, spd_to_xyz)
from luxpy.utils import (np, np2d, put_args_in_db)

_CAM_DEFAULT_CIEOBS = '2006_10' # place holder for real deal in calling module

__all__ = ['_update_parameter_dict','_setup_default_adaptation_field',
           '_massage_input_and_init_output','_massage_output_data_to_original_shape',
           '_get_absolute_xyz_xyzw','_simple_cam']


def _update_parameter_dict(args, 
                           parameters = {},
                           cieobs = _CAM_DEFAULT_CIEOBS, 
                           match_conversionmatrix_to_cieobs = False,
                           Mxyz2lms_whitepoint = None):
    """
    Get parameter dict and update with values in args dict. 
     | Also replace the xyz-to-lms conversion matrix with the one corresponding 
     | to cieobs and normalize it to illuminant E.
     
    Args:
        :args: 
            | dictionary with updated values. 
            | (get by placing 'args = locals().copy()' immediately after the start
            | of the function from which the update is called, 
            | see _simple_cam() code for an example.)
        :parameters:
            | dictionary with all (adjustable) parameter values used by the model   
        :cieobs: 
            | String with the CIE observer CMFs (one of _CMF['types'] of the input data
            | Is used to get the Mxyz2lms matrix when  match_conversionmatrix_to_cieobs == True)
        :match_conversionmatrix_to_cieobs:
            | False, optional
            | If False: keep the Mxyz2lms in the parameters dict
        :Mxyz2lms_whitepoint:
            | None, optional
            | If not None: update the Mxyz2lms key in the parameters dict
            | so that the conversion matrix is the one in _CMF[cieobs]['M'], 
            | in other such that it matches the cieobs of the input data.
    
    Returns:
        :parameters:
            | updated dictionary with model parameters for further use in the CAM.
            
    Notes:
        For an example on the use, see code _simple_cam() (type: _simple_cam??)
    """
    parameters = put_args_in_db(parameters,args)  #overwrite parameters with other (not-None) args input 
    if match_conversionmatrix_to_cieobs == True:
        parameters['Mxyz2lms'] = _CMF[cieobs]['M'].copy()
    if Mxyz2lms_whitepoint is None:
        Mxyz2lms_whitepoint = np.array([[1.0, 1.0, 1.0]])
    parameters['Mxyz2lms'] = math.normalize_3x3_matrix(parameters['Mxyz2lms'], 
                                                       Mxyz2lms_whitepoint) # normalize matrix for xyz-> lms conversion to ill. E
    return parameters


def _setup_default_adaptation_field(dataw = None, Lw = 100, 
                                    cie_illuminant = 'D65',
                                    inputtype = 'xyz', relative = True,
                                    cieobs = _CAM_DEFAULT_CIEOBS):
    """
    Setup a default illuminant adaptation field with Lw = 100 cd/m² for selected CIE observer.
    
    Args:
        :dataw: 
            | None or ndarray, optional
            | Input tristimulus values or spectral data of white point.
            | None defaults to the use of the illuminant specified in :cie_illuminant:.
        :cie_illuminant:
            | 'D65', optional
            | String corresponding to one of the illuminants (keys) 
            | in luxpy._CIE_ILLUMINANT
            | If ndarray, then use this one.
            | This is ONLY USED WHEN dataw is NONE !!!
        :Lw:
            | 100.0, optional
            | Luminance (cd/m²) of white point.
        :inputtype:
            | 'xyz' or 'spd', optional
            | Specifies the type of input: 
            |     tristimulus values or spectral data for the forward mode.
        :relative:
            | True or False, optional
            | True: xyz tristimulus values are relative (Yw = 100)
        :cieobs:
            | _CAM_DEFAULT_CIEOBS, optional
            | CMF set to use to perform calculations where spectral data 
            | is involved (inputtype == 'spd'; dataw = None)
            | Other options: see luxpy._CMF['types']
    
    Returns:
        :dataw:
            | Ndarray with default adaptation field data (spectral or xyz)
            
    Notes:
        For an example on the use, see code _simple_cam() (type: _simple_cam??)
    """
    if (dataw is None):
        if cie_illuminant is None:
            cie_illuminant = 'D65'
        if isinstance(cie_illuminant,str):
            cie_illuminant = _CIE_ILLUMINANTS[cie_illuminant]
        dataw = cie_illuminant.copy() # get illuminant 
        xyzw = spd_to_xyz(dataw, cieobs = cieobs,relative=False) # get abs. tristimulus values
        if relative == False: #input is expected to be absolute
            dataw[1:] = Lw*dataw[1:]/xyzw[:,1:2] # dataw = Lw*dataw # make absolute
        else:
            dataw = dataw # make relative (Y=100)
        if inputtype == 'xyz':
            dataw = spd_to_xyz(dataw, cieobs = cieobs, relative = relative)
    return dataw

def _massage_input_and_init_output(data, dataw, 
                                   inputtype = 'xyz', 
                                   direction = 'forward',
                                   n_out = 3):
    """
    Redimension input data to ensure most they have the appropriate sizes for easy and efficient looping.
    |
    | 1. Convert data and dataw to atleast_2d ndarrays
    | 2. Make axis 1 of dataw have 'same' dimensions as data
    | 3. Make dataw have same lights source axis size as data
    | 4. Flip light source axis to axis=0 for efficient looping
    | 5. Initialize output array camout to 'same' shape as data but with camout.shape[-1] == n_out
    
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
        :n_out:
            | 3, optional
            | output size of last dimension of camout 
            | (e.g. n_out=3 for j,a,b output or n_out = 5 for J,M,h,a,b output)
            
    Returns:
        :data:
            | ndarray with reshaped data
        :dataw:
            | ndarray with reshaped dataw
        :camout:
            | NaN filled ndarray for output of CAMv (camout.shape[-1] == Nout) 
        :originalshape:
            | original shape of data
            
    Notes:
        For an example on the use, see code _simple_cam() (type: _simple_cam??)
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
    if n_out is not None:
        dshape = list((data).shape)
        dshape[-1] = n_out # requested number of correlates: e.g. j,a,b
        if (inputtype != 'xyz') & (direction == 'forward'):
            dshape[-2] = dshape[-2] - 1 # wavelength row doesn't count & only with forward can the input data be spectral
        camout = np.zeros(dshape);
        camout.fill(np.nan)
    else:
        camout = None
    return data, dataw, camout, originalshape


def _massage_output_data_to_original_shape(data, originalshape):
    """
    Massage output data to restore original shape of original CAM input.
    
    Notes:
        For an example on the use, see code _simple_cam() (type: _simple_cam??)
    """
    # Flip light source dim back to axis 1:
    data = np.transpose(data, axes = (1,0,2))

    if len(originalshape) < 3:
        if data.shape[1] == 1:
            data = np.squeeze(data,axis = 1)

    return data

def _get_absolute_xyz_xyzw(data, dataw, i = 0, Lw= 100, direction = 'forward', 
                          cieobs = _CAM_DEFAULT_CIEOBS, inputtype = 'xyz', 
                          relative = True):
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
            | row number in data and dataw ndarrays 
            | (for loops across illuminant dimension after dimension reshape
            | with _massage_output_data_to_original_shape).
        :Lw:
            | 100.0, optional
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
            | _CAM_DEFAULT_CIEOBS, optional
            | CMF set to use to perform calculations where spectral data 
              is involved (inputtype == 'spd'; dataw = None)
            | Other options: see luxpy._CMF['types']
          
    Returns:
        :xyzti:
            | in forward mode : ndarray with relative or absolute sample xyz for data[i] 
            | in inverse mode: None
        :xyzwi:
            | ndarray with relative or absolute white point for dataw[i]
        :xyzw_abs:
            | ndarray with absolute xyz for white point for dataw[i]
            
    Notes:
        For an example on the use, see code _simple_cam() (type: _simple_cam??)
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

def _simple_cam(data, dataw = None, Lw = 100.0, relative = True, 
         inputtype = 'xyz', direction = 'forward', cie_illuminant = 'D65',
         parameters = {'cA': 1,
                       'ca':np.array([1,-1,0]),
                       'cb':(1/3)*np.array([0.5,0.5,-1]),
                       'n': 1/3, 
                       'Mxyz2lms': _CMF['1931_2']['M'].copy()},
         cieobs = '2006_10', match_to_conversionmatrix_to_cieobs = True):
    """
    An example CAM illustration the usage of the functions in luxpy.cam.helpers 
    
    | Note that this example uses NO chromatic adaptation 
    | and SIMPLE compression, opponent and correlate processing.
    | THIS IS ONLY FOR ILLUSTRATION PURPOSES !!!

    Args:
        :data: 
            | ndarray with input:
            |  - tristimulus values 
            | or
            |  - spectral data 
            | or 
            |  - input color appearance correlates
            | Can be of shape: (N [, xM], x 3), whereby: 
            | N refers to samples and M refers to light sources.
            | Note that for spectral input shape is (N x (M+1) x wl) 
        :dataw: 
            | None or ndarray, optional
            | Input tristimulus values or spectral data of white point.
            | None defaults to the use of :cie_illuminant:
        :cie_illuminant:
            | 'D65', optional
            | String corresponding to one of the illuminants (keys) 
            | in luxpy._CIE_ILLUMINANT
            | If ndarray, then use this one.
            | This is ONLY USED WHEN dataw is NONE !!!
        :Lw:
            | 100.0, optional
            | Luminance (cd/m²) of white point.
        :relative:
            | True or False, optional
            | True: data and dataw input is relative (i.e. Yw = 100)
        :parameters:
            | {'cA': 1, 'ca':np.array([1,-1,0]), 'cb':(1/3)*np.array([0.5,0.5,-1]),
            |  'n': 1/3, 'Mxyz2lms': _CMF['1931_2']['M'].copy()}
            | Dict with model parameters 
            | (For illustration purposes of match_conversionmatrix_to_cieobs, 
            |  the conversion matrix luxpy._CMF['1931_2']['M'] does NOT match
            |  the default observer specification of the input data in :cieobs: !!!)
        :inputtype:
            | 'xyz' or 'spd', optional
            | Specifies the type of input: 
            |     tristimulus values or spectral data for the forward mode.
        :direction:
            | 'forward' or 'inverse', optional
            |   -'forward': xyz -> cam
            |   -'inverse': cam -> xyz 
        :cieobs:
            | '2006_10', optional
            | CMF set to use to perform calculations where spectral data 
            | is involved (inputtype == 'spd'; dataw = None)
            | Other options: see luxpy._CMF['types']
        :match_conversionmatrix_to_cieobs:
            | True, optional
            | When changing to a different CIE observer, change the xyz_to_lms
            | matrix to the one corresponding to that observer. 
            | Set to False to keep the one in the parameter dict!
    
    Returns:
        :returns: 
            | ndarray with:
            | - color appearance correlates (:direction: == 'forward')
            |  or 
            | - XYZ tristimulus values (:direction: == 'inverse')
    """
    #--------------------------------------------------------------------------
    # Get model parameters:
    #--------------------------------------------------------------------------
    args = locals().copy() # gets all local variables (i.e. the function arguments)

    parameters = _update_parameter_dict(args, 
                                        parameters = parameters,
                                        cieobs = cieobs,
                                        match_conversionmatrix_to_cieobs = match_to_conversionmatrix_to_cieobs,
                                        Mxyz2lms_whitepoint = np.array([[1,1,1]]))
      
    #unpack model parameters:
    (Mxyz2lms,cA, ca, cb, n)  = [parameters[x] for x in sorted(parameters.keys())]


    #--------------------------------------------------------------------------
    # Setup default white point / adaptation field:   
    #--------------------------------------------------------------------------
    dataw = _setup_default_adaptation_field(dataw = dataw, 
                                            Lw = Lw,
                                            cie_illuminant = 'C',
                                            inputtype = inputtype, 
                                            relative = relative,
                                            cieobs = cieobs)

    #--------------------------------------------------------------------------
    # Redimension input data to ensure most appropriate sizes 
    # for easy and efficient looping and initialize output array:
    #--------------------------------------------------------------------------
    n_out = 5 # this example outputs 5 'correlates': J, a, b, C, h
    (data, dataw, 
     camout, originalshape) = _massage_input_and_init_output(data, 
                                                             dataw, 
                                                             inputtype = inputtype, 
                                                             direction = direction,
                                                             n_out = n_out)
    
    
    #--------------------------------------------------------------------------
    # Do precomputations needed for both the forward and inverse model,
    # and which do not depend on sample or light source data:
    #--------------------------------------------------------------------------
    # Create matrix with scale factors for L, M, S 
    # for quick matrix multiplications to obtain neural signals:
    MAab = np.array([[cA,cA,cA],ca,cb]) 
    
    if direction == 'inverse':
        invMxyz2lms = np.linalg.inv(Mxyz2lms) # Calculate the inverse lms-to-xyz conversion matrix
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
        # Get tristimulus values for stimulus field and white point for row i:
        #-----------------------------------------------------------------------------
        # Note that xyzt will contain a None in case of inverse mode !!!
        xyzt, xyzw, xyzw_abs = _get_absolute_xyz_xyzw(data, 
                                                      dataw,
                                                      i = i, 
                                                      Lw = Lw, 
                                                      direction = direction, 
                                                      cieobs = cieobs, 
                                                      inputtype = inputtype, 
                                                      relative = relative)
        
        #---------------------------------------------------------------------
        # stage 1 (white point): calculate lms values of white:
        #----------------------------------------------------------------------
        lmsw = np.dot(Mxyz2lms, xyzw.T).T 
        
        #------------------------------------------------------------------
        # stage 2 (white): apply simple chromatic adaptation:
        #------------------------------------------------------------------
        lmsw_a = lmsw/lmsw
        
        #----------------------------------------------------------------------
        # stage 3 (white point): apply simple compression to lms values
        #----------------------------------------------------------------------
        lmsw_ac = lmsw_a**n
        
        #----------------------------------------------------------------------
        # stage 4 (white point): calculate achromatic A, and opponent signals a,b):
        #----------------------------------------------------------------------
        Aabw = np.dot(MAab, lmsw_ac.T).T

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # SPLIT CALCULATION STEPS IN FORWARD AND INVERSE MODES:
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        if direction == 'forward':
            #------------------------------------------------------------------
            # stage 1 (stimulus): calculate lms values 
            #------------------------------------------------------------------
            lms = np.dot(Mxyz2lms, xyzt.T).T 
            
            #------------------------------------------------------------------
            # stage 2 (stimulus): apply simple chromatic adaptation:
            #------------------------------------------------------------------
            lms_a = lms/lmsw
        
            #------------------------------------------------------------------
            # stage 3 (stimulus): apply simple compression to lms values 
            #------------------------------------------------------------------
            lms_ac = lms_a**n
            
            #------------------------------------------------------------------
            # stage 3 (stimulus): calculate achromatic A, and opponent signals a,b:
            #------------------------------------------------------------------
            Aab = np.dot(MAab, lms_ac.T).T

            #------------------------------------------------------------------
            # stage 4 (stimulus): calculate J, C, h
            #------------------------------------------------------------------
            J = Aab[...,0]/Aabw[...,0]
            C = (Aab[...,1]**2 + Aab[...,2]**2)**0.5
            h = math.positive_arctan(Aab[...,1],Aab[...,2])
            
            # # stack together:
            camout[i] = np.vstack((J, Aab[...,1], Aab[...,2], C, h)).T 
        
        #--------------------------------------
        # INVERSE MODE FROM PERCEPTUAL SIGNALS:
        #--------------------------------------    
        elif direction == 'inverse':
            pass
        
    return _massage_output_data_to_original_shape(camout, originalshape)
    

if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    # Code test
    #--------------------------------------------------------------------------
    
    import luxpy as lx
    from luxpy import np
    
    # Prepare some illuminant data:
    C = _CIE_ILLUMINANTS['C'].copy()
    Ill1 = C
    Ill2 = np.vstack((C,lx.cie_interp(_CIE_ILLUMINANTS['D65'],C[0],kind='spd')[1:],C[1:,:]*2,C[1:,:]*3))
    
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

    # Single data for sample and illuminant:
    # test input to _simple_cam():
    print('\n\n1: xyz in:')
    jabch1_a = _simple_cam(xyz1, dataw = xyzw1, Lw = Lw, relative = True, \
                 inputtype = 'xyz', direction = 'forward', cieobs = cieobs)
    print(jabch1_a)
    
    print('1: spd in:')
    jabch1_b = _simple_cam(Ill1M, dataw = Ill1, Lw = Lw, relative = True, \
                  inputtype = 'spd', direction = 'forward', cieobs = cieobs)
    print(jabch1_b)
    print('DELTA(xyz,spd): Single, Single: ', (jabch1_a-jabch1_b).sum())
    
    
    
    # Multiple data for sample and illuminants:
    print('\n\n2: xyz in:')
    jabch2_a = _simple_cam(xyz2, dataw = xyzw2, Lw = Lw, relative = True, \
                 inputtype = 'xyz', direction = 'forward', cieobs = cieobs)
    print(jabch2_a)
    
    print('2: spd in:')
    jabch2_b = _simple_cam(Ill2M, dataw = Ill2, Lw = Lw, relative = True, \
                  inputtype = 'spd', direction = 'forward', cieobs = cieobs)
    print(jabch2_b)
    print('DELTA(xyz,spd): Multi, Multi: ', (jabch2_a-jabch2_b).sum())
    
    
    # Single data for sample, multiple illuminants:
    print('\n\n3: xyz in:')
    jabch3_a = _simple_cam(xyz1, dataw = xyzw2, Lw = Lw, relative = True, \
                 inputtype = 'xyz', direction = 'forward', cieobs = cieobs)
    print(jabch3_a)
    
    print('3: spd in:')
    jabch3_b = _simple_cam(Ill1M, dataw = Ill2, Lw = Lw, relative = True, \
                  inputtype = 'spd', direction = 'forward', cieobs = cieobs)
    print(jabch3_b)
    print('DELTA(xyz,spd): Single, Multi: ', (jabch3_a-jabch3_b).sum())

    # Module output plot:
    import matplotlib.pyplot as plt
    xyz, xyzw = lx.spd_to_xyz(Ill1, cieobs = cieobs, relative = True, rfl = rflM, out = 2)
    jabch = _simple_cam(xyz, dataw = xyzw, Lw = Lw, relative = True, \
                 inputtype = 'xyz', direction = 'forward', cieobs = cieobs)
    plt.figure()
    plt.plot(jabch[...,1],jabch[...,2],'b.')
    plt.axis('equal')