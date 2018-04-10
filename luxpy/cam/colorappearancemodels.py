# -*- coding: utf-8 -*-
"""

###################################################################################################
# Module with color appearance models
###################################################################################################
# _UNIQUE_HUE_DATA: database of unique hues with corresponding Hue quadratures and eccentricity factors
#                   (ciecam02, cam16, ciecam97s, cam15u)
#
# _SURROUND_PARAMETERS: database of surround parameters c, Nc, F and FLL for ciecam02, cam16, ciecam97s and cam15u.
#
# _NAKA_RUSHTON_PARAMETERS: database with parameters (n, sig, scaling and noise) for the Naka-Rushton function: scaling * ((data**n) / ((data**n) + (sig**n))) + noise
#
# _CAMUCS_PARAMETERS: database with parameters specifying the conversion from ciecam02/cam16 to cam[x]ucs (uniform color space), cam[x]lcd (large color diff.), cam[x]scd (small color diff).
#
# _CAM15U_PARAMETERS: database with CAM15u model parameters.
#
# _CAM_SWW_2016_PARAMETERS: database with cam_sww_2016 parameters (model by Smet, Webster and Whitehead published in JOSA A in 2016)
# 
# _CAM_DEFAULT_WHITE_POINT: Default internal reference white point (xyz)
#
# _CAM_DEFAULT_TYPE: Default CAM type str specifier.
#
# _CAM_AXES: dict with list[str,str,str] containing axis labels of defined cspaces
#
# naka_rushton(): applies a Naka-Rushton function to the input
# 
# hue_angle(): calculates a positive hue angle
#
# hue_quadrature(): calculates the Hue quadrature from the hue.
#
# cam_structure_ciecam02_cam16(): basic structure of both the ciecam02 and cam16 models. Has 'forward' (xyz --> color attributes) and 'inverse' (color attributes --> xyz) modes.
#
# ciecam02(): calculates ciecam02 output (wrapper for cam_structure_ciecam02_cam16 with specifics of ciecam02):  N. Moroney, M. D. Fairchild, R. W. G. Hunt, C. Li, M. R. Luo, and T. Newman, “The CIECAM02 color appearance model,” IS&T/SID Tenth Color Imaging Conference. p. 23, 2002.
#
# cam16(): calculates cam16 output (wrapper for cam_structure_ciecam02_cam16 with specifics of cam16):  C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a.
# 
# camucs_structure(): basic structure to go to ucs, lcd and scd color spaces (forward + inverse available)
#
# cam02ucs(): calculates ucs (or lcd, scd) output based on ciecam02 (forward + inverse available): M. R. Luo, G. Cui, and C. Li, “Uniform colour spaces based on CIECAM02 colour appearance model,” Color Res. Appl., vol. 31, no. 4, pp. 320–330, 2006.
#
# cam16ucs(): calculates ucs (or lcd, scd) output based on cam16 (forward + inverse available):  C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a.
#
# cam15u(): calculates the output for the CAM15u model for self-luminous unrelated stimuli. : M. Withouck, K. A. G. Smet, W. R. Ryckaert, and P. Hanselaer, “Experimental driven modelling of the color appearance of unrelated self-luminous stimuli: CAM15u,” Opt. Express, vol. 23, no. 9, pp. 12045–12064, 2015.
#
# cam_sww_2016(): calculates output for the CAM developed by Smet, Webster and Whitehead:  Smet, K. A. G., Webster, M. A., & Whitehead, L. A. (2016). A simple principled approach for modeling and understanding uniform color metrics. Journal of the Optical Society of America A, 33(3), A319–A331. https://doi.org/10.1364/JOSAA.33.00A319
#
# specific wrappers in the xyz_to_...() and ..._to_xyz() format:
# 'xyz_to_jabM_ciecam02', 'jabM_ciecam02_to_xyz',
# 'xyz_to_jabC_ciecam02', 'jabC_ciecam02_to_xyz',
# 'xyz_to_jabM_cam16', 'jabM_cam16_to_xyz',
# 'xyz_to_jabC_cam16', 'jabC_cam16_to_xyz',
# 'xyz_to_jab_cam02ucs', 'jab_cam02ucs_to_xyz', 
# 'xyz_to_jab_cam02lcd', 'jab_cam02lcd_to_xyz',
# 'xyz_to_jab_cam02scd', 'jab_cam02scd_to_xyz', 
# 'xyz_to_jab_cam16ucs', 'jab_cam16ucs_to_xyz',
# 'xyz_to_jab_cam16lcd', 'jab_cam16lcd_to_xyz',
# 'xyz_to_jab_cam16scd', 'jab_cam16scd_to_xyz', 
# 'xyz_to_qabW_cam15u', 'qabW_cam15u_to_xyz'
# 'xyz_to_LAb_cam_sww_2016', 'lab_cam_sww_2016_to_xyz'
#------------------------------------------------------------------------------

Created on Sun Jun 25 09:55:05 2017

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from .. import np, math, cat, _CIEOBS, _CIE_ILLUMINANTS, np2d, np2dT, np3d, put_args_in_db, spd_to_xyz, asplit, ajoin


__all__ = ['_CAM_AXES', '_UNIQUE_HUE_DATA','_SURROUND_PARAMETERS','_NAKA_RUSHTON_PARAMETERS','_CAMUCS_PARAMETERS','_CAM15U_PARAMETERS','_CAM_SWW_2016_PARAMETERS', '_CAM_DEFAULT_WHITE_POINT', '_CAM_DEFAULT_CONDITIONS','hue_angle', 'hue_quadrature','naka_rushton','ciecam02','cam02ucs','cam15u','cam_sww_2016']
__all__ += ['xyz_to_jabM_ciecam02', 'jabM_ciecam02_to_xyz', 'xyz_to_jabC_ciecam02', 'jabC_ciecam02_to_xyz',
            'xyz_to_jabM_cam16', 'jabM_cam16_to_xyz', 'xyz_to_jabC_cam16', 'jabC_cam16_to_xyz',
            'xyz_to_jab_cam02ucs', 'jab_cam02ucs_to_xyz', 'xyz_to_jab_cam02lcd', 'jab_cam02lcd_to_xyz','xyz_to_jab_cam02scd', 'jab_cam02scd_to_xyz', 
            'xyz_to_jab_cam16ucs', 'jab_cam16ucs_to_xyz', 'xyz_to_jab_cam16lcd', 'jab_cam16lcd_to_xyz','xyz_to_jab_cam16scd', 'jab_cam16scd_to_xyz', 
            'xyz_to_qabW_cam15u', 'qabW_cam15u_to_xyz',
            'xyz_to_lab_cam_sww_2016', 'lab_cam_sww_2016_to_xyz']


_CAM_AXES = {'jabM_ciecam02' : ["J (ciecam02)", "aM (ciecam02)", "bM (ciecam02)"]}
_CAM_AXES['jabC_ciecam02'] = ["J (ciecam02)", "aC (ciecam02)", "bC (ciecam02)"] 
_CAM_AXES['jabM_cam16'] = ["J (cam16)", "aM (cam16)", "bM (cam16)"]
_CAM_AXES['jabC_cam16'] = ["J (cam16)", "aC (cam16)", "bC (cam16)"] 
_CAM_AXES['jab_cam02ucs'] = ["J' (cam02ucs)", "a' (cam02ucs)", "b' (cam02ucs)"] 
_CAM_AXES['jab_cam02lcd'] = ["J' (cam02lcd)", "a' (cam02lcd)", "b' (cam02lcd)"] 
_CAM_AXES['jab_cam02scd'] = ["J' (cam02scd)", "a' (cam02scd)", "b' (cam02scd)"] 
_CAM_AXES['qabW_cam15u'] = ["Q (cam15u)", "aW (cam15u)", "bW (cam15u)"] 
_CAM_AXES['lab_cam_sww_2016'] = ["L (lab_cam_sww_2016)", "a (lab_cam_sww_2016)", "b (lab_cam_sww_2016)"] 

_UNIQUE_HUE_DATA = {'parameters': 'hues i hi ei Hi'.split()}
_UNIQUE_HUE_DATA['models'] = 'ciecam02 ciecam97s cam15u cam16'.split()
_UNIQUE_HUE_DATA['ciecam02'] = {'hues': 'red yellow green blue red'.split(), 'i': np.arange(5.0), 'hi':[20.14, 90.0, 164.25,237.53,380.14],'ei':[0.8,0.7,1.0,1.2,0.8],'Hi':[0.0,100.0,200.0,300.0,400.0]}
_UNIQUE_HUE_DATA['ciecam97s'] = _UNIQUE_HUE_DATA['ciecam02']
_UNIQUE_HUE_DATA['cam15u'] = _UNIQUE_HUE_DATA['ciecam02']
_UNIQUE_HUE_DATA['cam16'] = {'hues': 'red yellow green blue red'.split(), 'i': np.arange(5.0), 'hi':[20.14, 90.0, 164.25,237.53,380.14],'ei':[0.8,0.7,1.0,1.2,0.8],'Hi':[0.0,100.0,200.0,300.0,400.0]}

_SURROUND_PARAMETERS = {'parameters': 'c Nc F FLL'.split()}
_SURROUND_PARAMETERS['models'] = 'ciecam02 ciecam97s cam15u cam16'.split()
_SURROUND_PARAMETERS['ciecam02'] =  {'surrounds': ['avg', 'dim', 'dark'], 'avg' : {'c':0.69, 'Nc':1.0, 'F':1.0,'FLL': 1.0}, 'dim' : {'c':0.59, 'Nc':0.9, 'F':0.9,'FLL':1.0} ,'dark' : {'c':0.525, 'Nc':0.8, 'F':0.8,'FLL':1.0}}
_SURROUND_PARAMETERS['ciecam97s'] = {'surrounds': ['avg', 'avg,stim>4°','dim', 'dark','cutsheet'], 'avg' : {'c':0.69, 'Nc':1.0, 'F':1.0,'FLL': 1.0}, 'avg,stim>4°' : {'c':0.69, 'Nc':1.0, 'F':1.0,'FLL': 0.0}, 'dim' : {'c':0.59, 'Nc':1.1, 'F':0.9,'FLL':1.0} ,'dark' : {'c':0.525, 'Nc':0.8, 'F':0.9,'FLL':1.0},'cutsheet': {'c':0.41, 'Nc':0.8, 'F':0.9,'FLL':1.0}}
_SURROUND_PARAMETERS['cam15u'] =  {'surrounds': ['dark'], 'dark' : {'c': None, 'Nc':None,'F':None,'FLL':None}}
_SURROUND_PARAMETERS['cam16'] =  {'surrounds': ['avg', 'dim', 'dark'], 'avg' : {'c':0.69, 'Nc':1.0, 'F':1.0,'FLL': 1.0}, 'dim' : {'c':0.59, 'Nc':0.9, 'F':0.9,'FLL':1.0} ,'dark' : {'c':0.525, 'Nc':0.8, 'F':0.8,'FLL':1.0}}

_NAKA_RUSHTON_PARAMETERS = {'parameters': 'n sig scaling noise'.split()}
_NAKA_RUSHTON_PARAMETERS['models'] = 'ciecam02 ciecam97s cam15u cam16'.split()
_NAKA_RUSHTON_PARAMETERS['ciecam02'] = {'n':0.42, 'sig': 27.13**(1/0.42), 'scaling': 400.0, 'noise': 0.1}
_NAKA_RUSHTON_PARAMETERS['ciecam97s'] = {'n':0.73, 'sig': 2.0**(1/0.73), 'scaling': 40.0, 'noise': 1.0}
_NAKA_RUSHTON_PARAMETERS['cam15u'] = {'n':None, 'sig': None, 'scaling': None, 'noise': None}
_NAKA_RUSHTON_PARAMETERS['cam16'] = {'n':0.42, 'sig': 27.13**(1/0.42), 'scaling': 400.0, 'noise': 0.1}

_CAMUCS_PARAMETERS = {'ciecam02': {'ucs':{'KL': 1.0, 'c1':0.007,'c2':0.0228},'lcd':{'KL': 0.77, 'c1':0.007,'c2':0.0053}, 'scd':{'KL': 1.24, 'c1':0.007,'c2':0.0363}}}
_CAMUCS_PARAMETERS['cam16'] = {'ucs':{'KL': 1.0, 'c1':0.007,'c2':0.0228},'lcd':{'KL': 0.77, 'c1':0.007,'c2':0.0053}, 'scd':{'KL': 1.24, 'c1':0.007,'c2':0.0363}}


_CAM15U_PARAMETERS = {'k': [666.7, 782.3,1444.6],'cp': 1.0/3, 'cA':3.22 ,'cAlms':[2.0, 1.0, 1/20] ,'ca' : 1.0, 'calms':[1.0,-12/11,1/11],'cb': 0.117, 'cblms': [1.0, 1.0,-2.0], 'unique_hue_data':_UNIQUE_HUE_DATA['cam15u'], 'cM': 135.52, 'cHK': [2.559,0.561], 'cW': [2.29,2.68], 'cfov': 0.271, 'Mxyz2rgb': np.array([[0.211831, 0.815789, -0.042472],[-0.492493, 1.378921, 0.098745],[0.0, 0.0, 0.985188]])}

_CAM_SWW_2016_PARAMETERS = {'JOSA': {'cLMS': [1.0,1.0,1.0], 'lms0': [4985.0,5032.0,4761.0] , 'Cc': 0.252, 'Cf': -0.4, 'clambda': [0.5, 0.5, 0.0], 'calpha': [1.0, -1.0, 0.0], 'cbeta': [0.5, 0.5, -1.0], 'cga1': [26.1, 34.0], 'cgb1': [6.76, 10.9], 'cga2': [0.587], 'cgb2': [-0.952], 'cl_int': [14.0,1.0], 'cab_int': [4.99,65.8], 'cab_out' : [-0.1,-1.0], 'Ccwb': None, 'Mxyz2lms': [[ 0.21701045,  0.83573367, -0.0435106 ],[-0.42997951,  1.2038895 ,  0.08621089],[ 0.,  0.,  0.46579234]]}}
_CAM_SWW_2016_PARAMETERS['best-fit-JOSA'] = {'cLMS': [1.0,1.0,1.0], 'lms0': [4208.0,  4447.0,  4199.0] , 'Cc': 0.243, 'Cf': -0.269, 'clambda': [0.5, 0.5, 0.0], 'calpha': [1.0, -1.0, 0.0], 'cbeta': [0.5, 0.5, -1.0], 'cga1': [22.38, 26.42], 'cgb1': [5.36, 9.61], 'cga2': [0.668], 'cgb2': [-1.214], 'cl_int': [15.0, 1.04], 'cab_int': [5.85,65.86], 'cab_out' : [-1.008,-1.037], 'Ccwb': 0.80, 'Mxyz2lms': [[ 0.21701045,  0.83573367, -0.0435106 ],[-0.42997951,  1.2038895 ,  0.08621089],[ 0.,  0.,  0.46579234]]}
_CAM_SWW_2016_PARAMETERS['best-fit-all-Munsell'] = {'cLMS': [1.0,1.0,1.0], 'lms0': [5405.0, 5617.0,  5520.0] , 'Cc': 0.206, 'Cf': -0.128, 'clambda': [0.5, 0.5, 0.0], 'calpha': [1.0, -1.0, 0.0], 'cbeta': [0.5, 0.5, -1.0], 'cga1': [38.26, 43.35], 'cgb1': [8.97, 16.18], 'cga2': [0.512], 'cgb2': [-0.896], 'cl_int': [19.3, 0.99], 'cab_int': [5.87,63.24], 'cab_out' : [-0.545,-0.978], 'Ccwb': 0.736, 'Mxyz2lms': [[ 0.21701045,  0.83573367, -0.0435106 ],[-0.42997951,  1.2038895 ,  0.08621089],[ 0.,  0.,  0.46579234]]}

_CAM_DEFAULT_TYPE = 'ciecam02'
_CAM_DEFAULT_WHITE_POINT = np2d([100.0, 100.0, 100.0]) # ill. E white point
_CAM_DEFAULT_CONDITIONS = {'La': 100.0, 'Yb': 20.0, 'surround': 'avg','D': 1.0, 'Dtype':None}

def naka_rushton(data, sig = 2.0, n = 0.73, scaling = 1.0, noise = 0.0, cam = None, direction = 'forward'):
    """
    Apply a Naka-Rushton response compression (n) + adaptive shift (sig) to data.
    
    NK(x) = scaling * ((x**n) / ((x**n) + (sig**n))) + noise
    
    Args:
        :data: float or numpy.ndarray
        :sig: 2.0, optional
            Semi-saturation constant. Value for which NK(:data:) is 1/2
        :n: 0.73, optional
            Compression power.
        :scaling: 1.0, optional
            Maximum value of NK-function.
        :noise: 0.0, optional
            Cone excitation noise.
        :cam: None or str, optional
            Use NK parameters values specific to the color appearance model in :cam:.
            See luxpy.cam._NAKA_RUSHTON_PARAMETERS['models'] for supported types.
        :direction: 'forward' or 'inverse', optional
            Perform either NK(x) or NK(x)**(-1).
    
    Returns:
        :returns: float or numpy.ndarray with NK-(de)compressed input :x:        
    """
    if cam is not None: #override input
        n = _NAKA_RUSHTON_PARAMETERS[cam]['n']
        sig = _NAKA_RUSHTON_PARAMETERS[cam]['sig']
        scaling = _NAKA_RUSHTON_PARAMETERS[cam]['scaling']
        noise = _NAKA_RUSHTON_PARAMETERS[cam]['noise']
        
    if direction == 'forward':
        return scaling * ((data**n) / ((data**n) + (sig**n))) + noise
    elif direction =='inverse':
        Ip =  sig*(((np.abs(data-noise))/(scaling-np.abs(data-noise))))**(1/n)
        if not np.isscalar(Ip):
            p = np.where(data < noise)
            Ip[p] = -Ip[p]
        else:
            if data < noise:
                Ip = -Ip
        return Ip

def hue_angle(a,b, htype = 'deg'):
    """
    Calculate positive hue angle (0°-360° or 0 - 2*pi rad.) from opponent signals a and b.
    
    Args:
        :a: numpy.ndarray of a-coordinates
        :b: numpy.ndarray of b-coordinates
        :htype: 'deg' or 'rad', optional
            'deg': hue angle between 0° and 360°
            'rad': hue angle between 0 and 2pi radians
    Returns:
        :returns: numpy.ndarray of positive hue angles.
    """
    return math.positive_arctan(a,b, htype = htype)

def hue_quadrature(h, unique_hue_data = None):
    """
    Get hue quadrature H from h.
    
    Args:
        :h: float or list[float] or numpy.ndarray with hue data in degrees (!).
        :unique_hue data: None or str or dict, optional
            - None: H = h.
            - str: CAM specifier that gets parameters from luxpy.cam._UNIQUE_HUE_DATA
                (For supported models, see luxpy.cam._UNIQUE_HUE_DATA['models'])
            - dict: user specified unique hue data (see luxpy.cam._UNIQUE_HUE_DATA for expected structure)
    
    Returns:
        :H: float or list[float] or numpy.ndarray of Hue quadrature value(s).
    """
    if unique_hue_data is None:
        return h
    elif isinstance(unique_hue_data,str):
        unique_hue_data = _UNIQUE_HUE_DATA[unique_hue_data]
    
    squeezed = 0
    if h.shape[0] == 1:
        h = np.squeeze(h,axis = 0)
        squeezed = 1
    
    hi = unique_hue_data['hi']
    Hi = unique_hue_data['Hi']
    ei = unique_hue_data['ei']
    h[h<hi[0]] = h[h<hi[0]] + 360.0
    h_hi = np.repeat(np.atleast_2d(h),repeats=len(hi),axis = 1)
    hi_h = np.repeat(np.atleast_2d(hi),repeats=h.shape[0],axis = 0)
    d = h_hi-hi_h
    d[d<0] = 1000.0
    p = d.argmin(axis=1)
    p[p==(len(hi)-1)] = 0 # make sure last unique hue data is not selected
    H = np.array([Hi[pi] + (100.0*(h[i]-hi[pi])/ei[pi])/((h[i]-hi[pi])/ei[pi] + (hi[pi+1] - h[i])/ei[pi+1]) for (i,pi) in enumerate(p)])
    if squeezed == 1:
        H = np.expand_dims(H,axis=0)
    return H


def cam_structure_ciecam02_cam16(data, xyzw, camtype = _CAM_DEFAULT_TYPE, mcat = None, Yw = np2d(100), conditions = _CAM_DEFAULT_CONDITIONS, direction = 'forward', outin = 'J,aM,bM', yellowbluepurplecorrect = False):
    """
    Convert between XYZ tristsimulus values and ciecam02 /cam16 color appearance correlates.
    
    Args:
        :data: numpy.ndarray with input tristimulus values or input color appearance correlates
            Can be of shape: (N [, xM], x 3), N specifies samples, M specifies light sources.
        :xyzw: numpy.ndarray with tristimulus values of white point(s)
            Can be multiple by specifying a Mx3 numpy.ndarray, instead of 1x3.
        :Yw: luxpy.np2d(100), optional
            Luminance factor of white point.
            Is normally 100 for perfect white diffuser, < 100 for e.g. paper as white point.         
        :camtype: luxpy.cam._CAM_DEFAULT_TYPE, optional
            Str specifier for CAM type to use, options are 'ciecam02' or 'cam16'.
        :mcat: None or str or numpy.ndarray, optional
            Specifies CAT sensor space.
            - None defaults to 'cat02' (others e.g. 'cat02-bs', 'cat02-jiang', all trying to correct gamut problems of original cat02 matrix)
            - str: see see luxpy.cat._MCATS.keys() for options (details on type, ?luxpy.cat)
            - numpy.ndarray: matrix with sensor primaries
        :condition: luxpy.cam._CAM_DEFAULT_CONDITIONS, optional
            Dict specifying condition parameters, D, La, surround ([c,Nc,F]), Yb
            Can be user defined, but dict must have same structure.
        :direction: 'forward' or 'inverse', optional
            -'forward': xyz -> ciecam02 / cam16
            -'inverse': ciecam02 / cam16 -> xyz 
                (input data must be (J or Q, aM, bM) or (J or Q, aC,bC) or (J or Q, aS, bS) !!)
        :outin: 'J,aM,bM' or str, optional
            Str specifying the type of input (:direction: == 'inverse') and output (:direction: == 'forward')
        :yellowbluepurplecorrect: True or False, optional
            Correct for yellow-blue and purple problems in ciecam02 
            (Is not used in cam16 because cat16 solves issues)
    
    Returns:
        :returns: numpy.ndarray with color appearance correlates (:direction: == 'forward') or XYZ tristimulus values (:direction: == 'inverse')
    
    References:
        ..[1] N. Moroney, M. D. Fairchild, R. W. G. Hunt, C. Li, M. R. Luo, and T. Newman, “The CIECAM02 color appearance model,” IS&T/SID Tenth Color Imaging Conference. p. 23, 2002.
        ..[2] C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a.
    """
    outin = outin.split(',')
      
    #initialize data, xyzw, conditions and camout:
    data = np2d(data)
    data_original_shape = data.shape

    #prepare input parameters related to viewing conditions.
    if data.ndim == 2:
        data = data[:,None] # add dim to avoid looping
    
    if xyzw.ndim < 3:
        if xyzw.shape[0]==1:
            xyzw = (xyzw*np.ones(data.shape[1:]))[None] #make xyzw same looping size
        elif (xyzw.shape[0] == data.shape[1]):
            xyzw = xyzw[None]
        else:
            xyzw = xyzw[:,None]
    
    if isinstance(conditions,dict):
        conditions = np.repeat(conditions,data.shape[1]) #create condition dict for each xyzw

    Yw = np2d(Yw)
    if Yw.shape[0]==1:
        Yw = np.repeat(Yw,data.shape[1])
#    if len(mcat)==1:
#        mcat = np.repeat(mcat,data.shape[1]) #create mcat for each xyzw

    dshape = list(data.shape)
    dshape[-1] = len(outin) # requested number of correlates
    camout = np.nan*np.ones(dshape)
    
    # get general data:
    if camtype == 'ciecam02':
        if (mcat is None) | (mcat == 'cat02'):
            mcat = cat._MCATS['cat02']
            if yellowbluepurplecorrect == 'brill-suss':
                mcat = cat._MCATS['cat02-bs']  # for yellow-blue problem, Brill [Color Res Appl 2006;31:142-145] and Brill and Süsstrunk [Color Res Appl 2008;33:424-426] 
            elif yellowbluepurplecorrect == 'jiang-luo':
                mcat = cat._MCATS['cat02-jiang-luo'] # for yellow-blue problem + purple line problem
        elif isinstance(mcat,str):
            mcat = cat._MCATS[mcat]
        invmcat = np.linalg.inv(mcat)
        mhpe = cat._MCATS['hpe']
        mhpe_x_invmcat = np.dot(mhpe,invmcat)
        if direction == 'inverse':
            mcat_x_invmhpe = np.dot(mcat,np.linalg.inv(mhpe))
    elif camtype =='cam16':
        if mcat is None:
            mcat = cat._MCATS['cat16']
        elif isinstance(mcat,str):
            mcat = cat._MCATS[mcat]
            
        invmcat = np.linalg.inv(mcat)
    else:
            raise Exception('.cam.cam_structure_ciecam02_cam16(): Unrecognized camtype')
    
    # loop through all xyzw:
    for i in range(xyzw.shape[1]):
        # Get condition parameters:

        D, Dtype, La, Yb, surround = [conditions[i][x] for x in sorted(conditions[i].keys())] # unpack dictionary
        if isinstance(surround,str):
            surround = _SURROUND_PARAMETERS[camtype][surround] #if surround is not a dict of F,Nc,c values --> get from _SURROUND_PARAMETERS
        F, FLL, Nc, c = [surround[x] for x in sorted(surround.keys())]
 
     
        # calculate condition dependent parameters:
        k = 1.0 / (5.0*La + 1.0)
        FL = 0.2*(k**4.0)*(5.0*La) + 0.1*((1.0 - k**4.0)**2.0)*((5.0*La)**(1.0/3.0)) # luminance adaptation factor
        n = Yb/Yw[i] 
        Nbb = 0.725*(1/n)**0.2   
        Ncb = Nbb
        z = 1.48 + FLL*n**0.5
        yw = xyzw[:,i,1,None]
        xyzwi = Yw[i]*xyzw[:,i]/yw # normalize xyzw

        # calculate D:
        if D is None:
            D = F*(1.0-(1.0/3.6)*np.exp((-La-42.0)/92.0))

        # transform from xyzw to cat sensor space:
        rgbw = np.dot(mcat,xyzwi.T)

        # apply von Kries cat to white:
        rgbwc = ((100.0*D/rgbw) + (1 - D))*rgbw # factor 100 from ciecam02 is replaced with Yw[i] in cam16, but see 'note' in Fairchild's "Color Appearance Models" (p291 ni 3ed.)

        
        if camtype == 'ciecam02':
            # convert white from cat02 sensor space to cone sensors (hpe):
            rgbwp = np.dot(mhpe_x_invmcat,rgbwc).T
        elif camtype == 'cam16':
            rgbwp = rgbwc.T # in cam16, cat and cone sensor spaces are the same

        pw = np.where(rgbwp<0)
        
        if (yellowbluepurplecorrect == 'brill-suss')  & (camtype == 'ciecam02'): # Brill & Susstrunck approach, for purple line problem
            rgbwp[pw]=0.0
        
        # apply repsonse compression to white:
        rgbwpa = naka_rushton(FL*rgbwp/100.0, cam = camtype)
        rgbwpa[pw] = 0.1 - (naka_rushton(FL*np.abs(rgbwp[pw])/100.0, cam = camtype) - 0.1)
        
        # split white into separate cone signals:
        rwpa, gwpa, bwpa = asplit(rgbwpa)
        
        # Calculate achromatic signal:
        Aw =  (2.0*rwpa + gwpa + (1.0/20.0)*bwpa - 0.305)*Nbb
        
        if (direction == 'forward'):
        
            # calculate stimuli:
            xyzi = Yw[i]*data[:,i]/yw # normalize xyzw
            
            # transform from xyz to cat02 sensor space:
            rgb = np.dot(mcat,xyzi.T)

            # apply von Kries cat:
            rgbc = ((100.0*D/rgbw) + (1 - D))*rgb

            if camtype == 'ciecam02':
                # convert from cat02 sensor space to cone sensors (hpe):
                rgbp = np.dot(mhpe_x_invmcat,rgbc).T
            elif camtype == 'cam16':
                rgbp = rgbc.T # in cam16, cat and cone sensor spaces are the same
            p = np.where(rgbp<0)
            
            if (yellowbluepurplecorrect == 'brill-suss') & (camtype=='ciecam02'): # Brill & Susstrunck approach, for purple line problem
                rgbp[p]=0.0

            # apply repsonse compression:
            rgbpa = naka_rushton(FL*rgbp/100.0, cam = camtype)
            rgbpa[p] = 0.1 - (naka_rushton(FL*np.abs(rgbp[p])/100.0, cam = camtype) - 0.1)
            
            # split into separate cone signals:
            rpa, gpa, bpa = asplit(rgbpa)
            
            # calculate initial opponent channels:
            a = rpa - 12.0*gpa/11.0 + bpa/11.0
            b = (1.0/9.0)*(rpa + gpa - 2.0*bpa)
        
            # calculate hue h:
            h = hue_angle(a,b, htype = 'deg')
            
            # calculate eccentricity factor et:
            et = (1.0/4.0)*(np.cos(h*np.pi/180.0 + 2.0) + 3.8)
            
            # calculate Hue quadrature (if requested in 'out'):
            if 'H' in outin:    
                H = hue_quadrature(h, unique_hue_data = camtype)
            else:
                H = None
            
            # Calculate achromatic signal:
            A =  (2.0*rpa + gpa + (1.0/20.0)*bpa - 0.305)*Nbb
            
            # calculate lightness, J:
            if ('J' in outin) | ('Q' in outin) | ('C' in outin) | ('M' in outin) | ('s' in outin) | ('aS' in outin) | ('aC' in outin) | ('aM' in outin):
                J = 100.0* (A / Aw)**(c*z)
            
            # calculate brightness, Q:
            if ('Q' in outin) | ('s' in outin) | ('aS' in outin):
                Q = (4.0/c)* ((J/100.0)**0.5) * (Aw + 4.0)*(FL**0.25)
            
            # calculate chroma, C:
            if ('C' in outin) | ('M' in outin) | ('s' in outin) | ('aS' in outin) | ('aC' in outin) | ('aM' in outin):
                t = ((50000.0/13.0)*Nc*Ncb*et*((a**2.0 + b**2.0)**0.5)) / (rpa + gpa + (21.0/20.0*bpa))
                C = (t**0.9)*((J/100.0)**0.5) * (1.64 - 0.29**n)**0.73
           
  
            # calculate colorfulness, M:
            if ('M' in outin) | ('s' in outin) | ('aM' in outin) | ('aS' in outin):
                M = C*FL**0.25
             
            # calculate saturation, s:
            if ('s' in outin) | ('aS' in outin):
                s = 100.0* (M/Q)**0.5
                
            # calculate cartesion coordinates:
            if ('aS' in outin):
                 aS = s*np.cos(h*np.pi/180.0)
                 bS = s*np.sin(h*np.pi/180.0)
            
            if ('aC' in outin):
                 aC = C*np.cos(h*np.pi/180.0)
                 bC = C*np.sin(h*np.pi/180.0)
                 
            if ('aM' in outin):
                 aM = M*np.cos(h*np.pi/180.0)
                 bM = M*np.sin(h*np.pi/180.0)
                 
            if outin != ['J','aM','bM']:
                out_i = eval('ajoin(('+','.join(outin)+'))')
            else:
                out_i = ajoin((J,aM,bM))

            camout[:,i] = out_i
            
            
        elif (direction == 'inverse'):
            
            # input = J, a, b:
            J, aMCs, bMCs = asplit(data[:,i])
            
            # calculate hue h:
            h = hue_angle(aMCs,bMCs, htype = 'deg')
            
            # calculate M or C or s from a,b:
            MCs = (aMCs**2.0 + bMCs**2.0)**0.5    
            
            
            if ('Q' in outin):
                Q = J.copy()
                J = 100.0*(Q / ((Aw + 4.0)*(FL**0.25)*(4.0/c)))**2.0
            
            if ('aS' in outin):
                Q = (4.0/c)* ((J/100.0)**0.5) * (Aw + 4.0)*(FL**0.25)
                M = Q*(MCs/100.0)**2.0 
                C = M/(FL**0.25)
             
            if ('aM' in outin): # convert M to C:
                C = MCs/(FL**0.25)
            
            if ('aC' in outin):
                C = MCs
                
            # calculate t from J, C:
            t = (C / ((J/100.0)**(1.0/2.0) * (1.64 - 0.29**n)**0.73))**(1.0/0.9)
            
            # calculate eccentricity factor, et:
            et = (np.cos(h*np.pi/180.0 + 2.0) + 3.8) / 4.0
            
            # calculate achromatic signal, A:
            A = Aw*(J/100.0)**(1.0/(c*z))
            
            # calculate temporary cart. co. at, bt and p1,p2,p3,p4,p5:
            at = np.cos(h*np.pi/180.0)
            bt = np.sin(h*np.pi/180.0)
            p1 = (50000.0/13.0)*Nc*Ncb*et/t
            p2 = A/Nbb + 0.305
            p3 = 21.0/20.0
            p4 = p1/bt
            p5 = p1/at

            q = np.where(np.abs(bt) < np.abs(at))


            b = p2*(2.0 + p3) * (460.0/1403.0) / (p4 + (2.0 + p3) * (220.0/1403.0) * (at/bt) - (27.0/1403.0) + p3*(6300.0/1403.0))
            a = b * (at/bt)

            a[q] = p2[q]*(2.0 + p3) * (460.0/1403.0) / (p5[q] + (2.0 + p3) * (220.0/1403.0) - ((27.0/1403.0) - p3*(6300.0/1403.0)) * (bt[q]/at[q]))
            b[q] = a[q] * (bt[q]/at[q])
            
            # calculate post-adaptation values
            rpa = (460.0*p2 + 451.0*a + 288.0*b) / 1403.0
            gpa = (460.0*p2 - 891.0*a - 261.0*b) / 1403.0
            bpa = (460.0*p2 - 220.0*a - 6300.0*b) / 1403.0

            # join values:
            rgbpa = ajoin((rpa,gpa,bpa))
            
            # decompress signals:
            rgbp = (100.0/FL)*naka_rushton(rgbpa, cam = camtype, direction = 'inverse')
           
            if (yellowbluepurplecorrect == 'brill-suss') & (camtype == 'ciecam02'): # Brill & Susstrunck approach, for purple line problem
                p = np.where(rgbp<0.0)
                rgbp[p]=0.0
            
            if  (camtype == 'ciecam02'):
                # convert from to cone sensors (hpe) cat02 sensor space:
                rgbc = np.dot(mcat_x_invmhpe,rgbp.T)
            elif (camtype == 'cam16'):
                rgbc = rgbp.T # in cam16, cat and cone sensor spaces are the same
                     
            # apply inverse von Kries cat:
            rgb = rgbc/ ((100.0*D/rgbw) + (1.0 - D))
 
            # transform from cat sensor space to xyz:
            xyzi = np.dot(invmcat,rgb).T
            
            # unnormalize data:
            xyzi = xyzi*yw/Yw[i] 
            #xyzi[xyzi<0] = 0
            
            camout[:,i] = xyzi
    
    # return to original shape:
    if len(data_original_shape) == 2:
        camout = camout[:,0]   
   
    return camout    


    
    
#---------------------------------------------------------------------------------------------------------------------
def ciecam02(data, xyzw, mcat = 'cat02', Yw = np2d(100.0), conditions = _CAM_DEFAULT_CONDITIONS, direction = 'forward', outin = 'J,aM,bM', yellowbluepurplecorrect = False):
    """
    Convert between XYZ tristsimulus values and ciecam02 color appearance correlates.
    
    Wrapper for luxpy.cam.cam_structure_ciecam02_cam16() designed specifically for camtype = 'ciecam02.
    
    Args:
        :data: numpy.ndarray with input tristimulus values or input color appearance correlates
            Can be of shape: (N [, xM], x 3), N specifies samples, M specifies light sources.
        :xyzw: numpy.ndarray with tristimulus values of white point(s)
            Can be multiple by specifying a Mx3 numpy.ndarray, instead of 1x3.
        :Yw: luxpy.np2d(100), optional
            Luminance factor of white point.
            Is normally 100 for perfect white diffuser, < 100 for e.g. paper as white point.         
        :mcat: 'cat02' or str or numpy.ndarray, optional
            Specifies CAT sensor space.
            - None defaults to 'cat02' (others e.g. 'cat02-bs', 'cat02-jiang', all trying to correct gamut problems of original cat02 matrix)
            - str: see see luxpy.cat._MCATS.keys() for options (details on type, ?luxpy.cat)
            - numpy.ndarray: matrix with sensor primaries
        :condition: luxpy.cam._CAM_DEFAULT_CONDITIONS, optional
            Dict specifying condition parameters, D, La, surround ([c,Nc,F]), Yb
            Can be user defined, but dict must have same structure.
        :direction: 'forward' or 'inverse', optional
            -'forward': xyz -> ciecam02 
            -'inverse': ciecam02 -> xyz 
                (input data must be (J or Q, aM, bM) or (J or Q, aC,bC) or (J or Q, aS, bS) !!)
        :outin: 'J,aM,bM' or str, optional
            Str specifying the type of input (:direction: == 'inverse') and output (:direction: == 'forward')
        :yellowbluepurplecorrect: False or True, optional
            Correct for yellow-blue and purple problems in ciecam02 
    
    Returns:
        :returns: numpy.ndarray with color appearance correlates (:direction: == 'forward') or XYZ tristimulus values (:direction: == 'inverse')
    
    References:
        ..[1] N. Moroney, M. D. Fairchild, R. W. G. Hunt, C. Li, M. R. Luo, and T. Newman, “The CIECAM02 color appearance model,” IS&T/SID Tenth Color Imaging Conference. p. 23, 2002.
    
    """
    return cam_structure_ciecam02_cam16(data, xyzw, camtype = 'ciecam02', mcat = mcat, Yw = Yw, conditions = conditions, direction = direction, outin = outin, yellowbluepurplecorrect = yellowbluepurplecorrect)


#---------------------------------------------------------------------------------------------------------------------
def cam16(data, xyzw, mcat = 'cat16', Yw = np2d(100.0), conditions = _CAM_DEFAULT_CONDITIONS, direction = 'forward', outin = 'J,aM,bM'):
    """
    Convert between XYZ tristsimulus values and cam16 color appearance correlates.
    
    Wrapper for luxpy.cam.cam_structure_ciecam02_cam16() designed specifically for camtype = 'cam16'.
    
    Args:
        :data: numpy.ndarray with input tristimulus values or input color appearance correlates
            Can be of shape: (N [, xM], x 3), N specifies samples, M specifies light sources.
        :xyzw: numpy.ndarray with tristimulus values of white point(s)
            Can be multiple by specifying a Mx3 numpy.ndarray, instead of 1x3.
        :Yw: luxpy.np2d(100), optional
            Luminance factor of white point.
            Is normally 100 for perfect white diffuser, < 100 for e.g. paper as white point.         
        :mcat: 'cat16' or str or numpy.ndarray, optional
            Specifies CAT sensor space.
            - None defaults to back 'cat02' ! (others e.g. 'cat02-bs', 'cat02-jiang', all trying to correct gamut problems of original cat02 matrix)
            - str: see see luxpy.cat._MCATS.keys() for options (details on type, ?luxpy.cat)
            - numpy.ndarray: matrix with sensor primaries
        :condition: luxpy.cam._CAM_DEFAULT_CONDITIONS, optional
            Dict specifying condition parameters, D, La, surround ([c,Nc,F]), Yb
            Can be user defined, but dict must have same structure.
        :direction: 'forward' or 'inverse', optional
            -'forward': xyz -> cam16 
            -'inverse': cam16 -> xyz 
                (input data must be (J or Q, aM, bM) or (J or Q, aC,bC) or (J or Q, aS, bS) !!)
        :outin: 'J,aM,bM' or str, optional
            Str specifying the type of input (:direction: == 'inverse') and output (:direction: == 'forward')
    
    Returns:
        :returns: numpy.ndarray with color appearance correlates (:direction: == 'forward') or XYZ tristimulus values (:direction: == 'inverse')
    
    References:
        ..[1] C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a.
 
    """    
    return cam_structure_ciecam02_cam16(data, xyzw, camtype = 'cam16', mcat = mcat, Yw = Yw, conditions = conditions, direction = direction, outin = outin, yellowbluepurplecorrect = False)

#---------------------------------------------------------------------------------------------------------------------
def camucs_structure(data, xyzw = _CAM_DEFAULT_WHITE_POINT, camtype = 'ciecam02', mcat = None, Yw = np2d(100.0), conditions = _CAM_DEFAULT_CONDITIONS, direction = 'forward', ucstype = 'ucs', yellowbluepurplecorrect = False):
    """
    Convert between XYZ tristsimulus values and camucs type color appearance correlates.
    
    Wrapper for luxpy.cam.cam_structure_ciecam02_cam16() with additional compression of color attributes
    for the following case:
        - 'ucs': uniform color space
        - 'lcd': large color differences
        - 'scd': small color differences
    
    Args:
        :data: numpy.ndarray with input tristimulus values or input color appearance correlates
            Can be of shape: (N [, xM], x 3), N specifies samples, M specifies light sources.
        :xyzw: numpy.ndarray with tristimulus values of white point(s)
            Can be multiple by specifying a Mx3 numpy.ndarray, instead of 1x3.
        :Yw: luxpy.np2d(100), optional
            Luminance factor of white point.
            Is normally 100 for perfect white diffuser, < 100 for e.g. paper as white point.         
        :camtype: luxpy.cam._CAM_DEFAULT_TYPE, optional
            Str specifier for CAM type to use, options are 'ciecam02' or 'cam16'.
        :mcat: None or str or numpy.ndarray, optional
            Specifies CAT sensor space.
            - None defaults to 'cat02' (others e.g. 'cat02-bs', 'cat02-jiang', all trying to correct gamut problems of original cat02 matrix)
            - str: see see luxpy.cat._MCATS.keys() for options (details on type, ?luxpy.cat)
            - numpy.ndarray: matrix with sensor primaries
        :condition: luxpy.cam._CAM_DEFAULT_CONDITIONS, optional
            Dict specifying condition parameters, D, La, surround ([c,Nc,F]), Yb
            Can be user defined, but dict must have same structure.
        :direction: 'forward' or 'inverse', optional
            -'forward': xyz -> camucs
            -'inverse': camucs -> xyz 
                (input data must be (J or Q, aM, bM) or (J or Q, aC,bC) or (J or Q, aS, bS) !!)
        :yellowbluepurplecorrect: True or False, optional
            Correct for yellow-blue and purple problems in ciecam02 
            (Is not used in cam16 because cat16 solves issues)
        :ucstype: 'ucs' or 'lcd' or 'scd', optional
            Str specifier for which type of color attribute compression parameters to use.
             ('ucs': uniform color space, 'lcd', large color differences, 'scd': small color differences)
             
    Returns:
        :returns: numpy.ndarray with color appearance correlates (:direction: == 'forward') or XYZ tristimulus values (:direction: == 'inverse')
    
    References:
        ..[1] M. R. Luo, G. Cui, and C. Li, “Uniform colour spaces based on CIECAM02 colour appearance model,” Color Res. Appl., vol. 31, no. 4, pp. 320–330, 2006.
    
    """
    if mcat == None:
        if camtype == 'ciecam02':
            mcat = 'cat02'
        elif camtype == 'cam16':
            mcat = 'cat16'

    # get cam02 parameters:
    KL, c1, c2 = [_CAMUCS_PARAMETERS[camtype][ucstype][x] for x in sorted(_CAMUCS_PARAMETERS[camtype][ucstype].keys())]
    
    if direction == 'forward':
        
        # calculate ciecam02 J, aM,bM:
        J, aM, bM = asplit(cam_structure_ciecam02_cam16(data, xyzw = xyzw, camtype = camtype, Yw = Yw, conditions = conditions, direction = 'forward', outin = 'J,aM,bM',yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat))

        # convert to cam02ucs J', aM', bM':
        M  = (aM**2.0 + bM**2.0)**0.5 
        h=np.arctan2(bM,aM)
        Jp = (1.0 + 100.0*c1)*J / (1.0 + c1*J)
        Mp = (1.0/c2) * np.log(1.0 + c2*M)
        aMp = Mp*np.cos(h)
        bMp = Mp*np.sin(h)
        
        return ajoin((Jp,aMp,bMp))
        
    elif direction == 'inverse':
        
    # convert J',aM', bM' to ciecam02 J,aM,bM:
        # calc CAM02 hue angle
        Jp,aMp,bMp = asplit(data)
        h=np.arctan2(bMp,aMp)

        # calc CAM02 and CIECAM02 colourfulness
        Mp = (aMp**2.0+bMp**2.0)**0.5
        M = (np.exp(c2*Mp) - 1.0) / c2
        
        # calculate ciecam02 aM, bM:
        aM = M*np.cos(h)
        bM = M*np.sin(h)

        # calc CAM02 lightness
        J = Jp/(1.0 + (100.0 - Jp)*c1)
        
        data = ajoin((J,aM,bM))
        
        # calculate xyz from ciecam02 J,aM,bM
        return cam_structure_ciecam02_cam16(data, xyzw = xyzw, camtype = camtype, Yw = Yw, conditions = conditions, direction = 'inverse', outin = 'J,aM,bM',yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
     
#---------------------------------------------------------------------------------------------------------------------
def cam02ucs(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = np2d(100.0), conditions = _CAM_DEFAULT_CONDITIONS, direction = 'forward', ucstype = 'ucs', yellowbluepurplecorrect = False, mcat = 'cat02'):
    """
    Convert between XYZ tristsimulus values and cam02ucs type color appearance correlates.
    
    Wrapper for luxpy.cam.camucs_structure() specifically designed for 'ciecam02' + 'ucs'
    
    Args:
        :data: numpy.ndarray with input tristimulus values or input color appearance correlates
            Can be of shape: (N [, xM], x 3), N specifies samples, M specifies light sources.
        :xyzw: numpy.ndarray with tristimulus values of white point(s)
            Can be multiple by specifying a Mx3 numpy.ndarray, instead of 1x3.
        :Yw: luxpy.np2d(100), optional
            Luminance factor of white point.
            Is normally 100 for perfect white diffuser, < 100 for e.g. paper as white point.         
        :mcat: 'cat02' or None or str or numpy.ndarray, optional
            Specifies CAT sensor space.
            - None defaults to 'cat02' (others e.g. 'cat02-bs', 'cat02-jiang', all trying to correct gamut problems of original cat02 matrix)
            - str: see see luxpy.cat._MCATS.keys() for options (details on type, ?luxpy.cat)
            - numpy.ndarray: matrix with sensor primaries
        :condition: luxpy.cam._CAM_DEFAULT_CONDITIONS, optional
            Dict specifying condition parameters, D, La, surround ([c,Nc,F]), Yb
            Can be user defined, but dict must have same structure.
        :direction: 'forward' or 'inverse', optional
            -'forward': xyz -> cam02ucs
            -'inverse': cam02ucs -> xyz 
                (input data must be (J or Q, aM, bM) or (J or Q, aC,bC) or (J or Q, aS, bS) !!)
        :yellowbluepurplecorrect: True or False, optional
            Correct for yellow-blue and purple problems in ciecam02 
            (Is not used in cam16 because cat16 solves issues)
        :ucstype: 'ucs' or 'lcd' or 'scd', optional
            Str specifier for which type of color attribute compression parameters to use.
             ('ucs': uniform color space, 'lcd', large color differences, 'scd': small color differences)
             
    Returns:
        :returns: numpy.ndarray with color appearance correlates (:direction: == 'forward') or XYZ tristimulus values (:direction: == 'inverse')
    
    References:
        ..[1] M. R. Luo, G. Cui, and C. Li, “Uniform colour spaces based on CIECAM02 colour appearance model,” Color Res. Appl., vol. 31, no. 4, pp. 320–330, 2006.
    
    """
    return camucs_structure(data, xyzw = xyzw, camtype = 'ciecam02', mcat = mcat, Yw = Yw, conditions = conditions, direction = direction, ucstype = ucstype, yellowbluepurplecorrect = yellowbluepurplecorrect)

 #---------------------------------------------------------------------------------------------------------------------
def cam16ucs(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = np2d(100.0), conditions = _CAM_DEFAULT_CONDITIONS, direction = 'forward', ucstype = 'ucs',  mcat = 'cat16'):
    """
    Convert between XYZ tristsimulus values and cam16ucs type color appearance correlates.
    
    Wrapper for luxpy.cam.camucs_structure() specifically designed for 'cam16' + 'ucs'
    
    Args:
        :data: numpy.ndarray with input tristimulus values or input color appearance correlates
            Can be of shape: (N [, xM], x 3), N specifies samples, M specifies light sources.
        :xyzw: numpy.ndarray with tristimulus values of white point(s)
            Can be multiple by specifying a Mx3 numpy.ndarray, instead of 1x3.
        :Yw: luxpy.np2d(100), optional
            Luminance factor of white point.
            Is normally 100 for perfect white diffuser, < 100 for e.g. paper as white point.         
        :mcat: 'cat16' or None or str or numpy.ndarray, optional
            Specifies CAT sensor space.
            - None defaults to 'cat02' (others e.g. 'cat02-bs', 'cat02-jiang', all trying to correct gamut problems of original cat02 matrix)
            - str: see see luxpy.cat._MCATS.keys() for options (details on type, ?luxpy.cat)
            - numpy.ndarray: matrix with sensor primaries
        :condition: luxpy.cam._CAM_DEFAULT_CONDITIONS, optional
            Dict specifying condition parameters, D, La, surround ([c,Nc,F]), Yb
            Can be user defined, but dict must have same structure.
        :direction: 'forward' or 'inverse', optional
            -'forward': xyz -> ciecam02 / cam16
            -'inverse': ciecam02 / cam16 -> xyz 
                (input data must be (J or Q, aM, bM) or (J or Q, aC,bC) or (J or Q, aS, bS) !!)
        :yellowbluepurplecorrect: True or False, optional
            Correct for yellow-blue and purple problems in ciecam02 
            (Is not used in cam16 because cat16 solves issues)
        :ucstype: 'ucs' or 'lcd' or 'scd', optional
            Str specifier for which type of color attribute compression parameters to use.
             ('ucs': uniform color space, 'lcd', large color differences, 'scd': small color differences)
             
    Returns:
        :returns: numpy.ndarray with color appearance correlates (:direction: == 'forward') or XYZ tristimulus values (:direction: == 'inverse')
    
    References:
        ..[1] M. R. Luo, G. Cui, and C. Li, “Uniform colour spaces based on CIECAM02 colour appearance model,” Color Res. Appl., vol. 31, no. 4, pp. 320–330, 2006.
        ..[2] C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a.

    """
    return camucs_structure(data, xyzw = xyzw, camtype = 'cam16', mcat = mcat, Yw = Yw, conditions = conditions, direction = direction, ucstype = ucstype, yellowbluepurplecorrect = False)
  
     
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------            
def cam15u(data, fov = 10.0, inputtype = 'xyz', direction = 'forward', outin = 'Q,aW,bW', parameters = None):
    """
    Convert between CIE 2006 10°  XYZ tristimulus values (or spectral data) and CAM15u color appearance correlates.
    
    Args:
        :data: numpy.ndarray of CIE 2006 10°  XYZ tristimulus values or spectral data or color appearance attributes
        :fov: 10.0, optional
            Field-of-view of stimulus (to take size effect on brightness into account)
        :inputtpe: 'xyz' or 'spd', optional
            Specifies the type of input: tristimulus values or spectral data for the forward mode.
        :direction: 'forward' or 'inverse', optional
            -'forward': xyz -> cam15u
            -'inverse': cam15u -> xyz 
        :outin: 'Q,aW,bW' or str, optional
            'Q,aW,bW' (brightness and opponent signals for amount-of-neutral/white)
            other options: 'Q,aM,bM' (colorfulness) and 'Q,aS,bS' (saturation)
            Str specifying the type of input (:direction: == 'inverse') and output (:direction: == 'forward')
        :parameters: None or dict, optional
            Set of model parameters.
            - None: defaults to luxpy.cam._CAM15U_PARAMETERS (see references below)
    
    Returns:
        :returns: numpy.ndarray with color appearance correlates (:direction: == 'forward') or XYZ tristimulus values (:direction: == 'inverse')

    
    References: 
        ..[1] Withouck M., Smet K.A.G, Ryckaert WR, Hanselaer P. (2015). 
                Experimental driven modelling of the color appearance of unrelated self-luminous stimuli: CAM15u. 
                Optics Express,  23 (9), 12045-12064. 
        ..[2] Withouck, M., Smet, K., Hanselaer, P. (2015). 
                Brightness prediction of different sized unrelated self-luminous stimuli. 
                Optics Express, 23 (10), 13455-13466    
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
    dshape = list(data.shape)
    dshape[-1] = len(outin) # requested number of correlates
    camout = np.nan*np.ones(dshape)
   
    for i in range(data.shape[0]):
        
        if (inputtype != 'xyz') & (direction == 'forward'):
            xyz = spd_to_xyz(data[i], cieobs = '2006_10', relative = False)
            lms = np.dot(_CMF['M']['2006_10'],xyz.T).T # convert to l,m,s
            rgb = (lms / _CMF['K']['2006_10']) * k # convert to rho, gamma, beta
        elif (inputtype == 'xyz') & (direction == 'forward'):
            rgb = np.dot(Mxyz2rgb,data[i].T).T
       
        if direction == 'forward':

            # apply cube-root compression:
            rgbc = rgb**(cp)
        
            # calculate achromatic and color difference signals, A, a, b:
            Aab = np.dot(MAab, rgbc.T).T
            A,a,b = asplit(Aab)

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
            
            # calculate hue angle:
            h = hue_angle(a,b, htype = 'rad')
            
            # calculate a,b from M and h:
            a = (M/cM)*np.cos(h)
            b = (M/cM)*np.sin(h)

            # create Aab:
            Aab = ajoin((A,a,b))    
            
            # calculate rgbc:
            rgbc = np.dot(invMAab, Aab.T).T    
            
            # decompress rgbc to rgb:
            rgb = rgbc**(1/cp)
            
            
            # convert rgb to xyz:
            xyz = np.dot(invMxyz2rgb,rgb.T).T 
            
            camout[i] = xyz
    
    if camout.shape[0] == 1:
        camout = np.squeeze(camout,axis = 0)
    
    return camout
 
#------------------------------------------------------------------------------
def cam_sww_2016(data, dataw = None, Yb = 20.0, Lw = 400.0, relative = True, parameters = None, inputtype = 'xyz', direction = 'forward', cieobs = '2006_10'):
    """
    A simple principled color appearance model based on a mapping of the Munsell color system.
    
    This function implements the JOSA A (parameters = 'JOSA') published model. 
    
    Args:
        :data: numpy.ndarray with input tristimulus values or spectral data or input color appearance correlates
            Can be of shape: (N [, xM], x 3), N specifies samples, M specifies light sources.
            Note that for spectral input shape is (N x (M+1) x wl) 
        :dataw: None or numpy.ndarray, optional
            Input tristimulus values or spectral data of white point.
            None defaults to the use of CIE illuminant C.
        :Yb: 20.0, optional
            Luminance factor of background (perfect white diffuser, Yw = 100)
        :Lw: 400.0, optional
            Luminance (cd/m²) of white point.
        :relative: True or False, optional
            True: xyz tristimulus values are relative (Yw = 100)
        :parameters: None or str or dict, optional
            Dict with model parameters.
                - None: defaults to luxpy.cam._CAM_SWW_2016_PARAMETERS['JOSA']
                - str: 'best-fit-JOSA' or 'best-fit-all-Munsell'
                - dict: user defined model parameters (dict should have same structure)
        :inputtpe: 'xyz' or 'spd', optional
            Specifies the type of input: tristimulus values or spectral data for the forward mode.
        :direction: 'forward' or 'inverse', optional
            -'forward': xyz -> cam_sww_2016
            -'inverse': cam_sww_2016 -> xyz 
        :cieobs: '2006_10', optional
            CMF set to use to perform calculations where spectral data is involved (inputtype == 'spd'; dataw = None)
            Other options: see luxpy._CMF['types']
    
    Returns:
        :returns: numpy.ndarray with color appearance correlates (:direction: == 'forward') or XYZ tristimulus values (:direction: == 'inverse')
    
    Notes:
        This function implements the JOSA A (parameters = 'JOSA') published model. 
        (with a correction for the parameter in Eq.4 of Fig. 11: 0.952 --> -0.952 
         and the delta_ac and delta_bc white-balance shifts in Eq. 5e & 5f should be: -0.028 & 0.821 
         (cfr. Ccwb = 0.66 in: ab_test_out = ab_test_int - Ccwb*ab_gray_adaptation_field_int)),
             
    References:
        ..[1] Smet, K. A. G., Webster, M. A., & Whitehead, L. A. (2016). 
            A simple principled approach for modeling and understanding uniform color metrics. 
            Journal of the Optical Society of America A, 33(3), A319–A331. 
            https://doi.org/10.1364/JOSAA.33.00A319

    """
    # get model parameters
    args = locals().copy() 
    if parameters is None:
        parameters = _CAM_SWW_2016_PARAMETERS['JOSA']
    if isinstance(parameters,str):
        parameters = _CAM_SWW_2016_PARAMETERS[parameters]
    parameters = put_args_in_db(parameters,args)  #overwrite parameters with other (not-None) args input 
   
    #unpack model parameters:
    Cc, Ccwb, Cf, Mxyz2lms, cLMS, cab_int, cab_out, calpha, cbeta,cga1, cga2, cgb1, cgb2, cl_int, clambda, lms0  = [parameters[x] for x in sorted(parameters.keys())]

    
    # setup default adaptation field:   
    if (dataw is None):
        dataw = _CIE_ILLUMINANTS['C'] # get illuminant C
        xyzw = spd_to_xyz(dataw, cieobs = cieobs,relative=False) # get abs. tristimulus values
        if relative == False: #input is expected to be absolute
            dataw[1:] = Lw*dataw[1:]/xyzw[0,1] #dataw = Lw*dataw # make absolute
        else:
            dataw = dataw # make relative (Y=100)
        if inputtype == 'xyz':
            dataw = spd_to_xyz(dataw, cieobs = cieobs, relative = relative)

    # precomputations:
    Mxyz2lms = np.dot(np.diag(cLMS),math.normalize_3x3_matrix(Mxyz2lms, np.array([1, 1, 1]))) # normalize matrix for xyz-> lms conversion to ill. E weighted with cLMS   
    invMxyz2lms = np.linalg.inv(Mxyz2lms)
    MAab = np.array([clambda,calpha,cbeta])
    invMAab = np.linalg.inv(MAab)
    
    #initialize data and camout:
    data = np2d(data).copy() # stimulus data (can be upto NxMx3 for xyz, or [N x (M+1) x wl] for spd))
    dataw = np2d(dataw).copy() # white point (can be upto Nx3 for xyz, or [(N+1) x wl] for spd)

    if len(data.shape)==2: # avoid looping if not necessary
        data = np.expand_dims(data, axis = 0) 

    if dataw.shape[0] == 1: #make dataw have same lights source dimension size as data
        if data.ndim == 3: # data array contains rfl data under several light sources
            dataw = np.repeat(dataw,data.shape[1],axis=0)
        else: # data array contains only either ligthsource or rfl data
            dataw = np.repeat(dataw,data.shape[0],axis=0)
            
    dshape = list(data.shape)
    dshape[-1] = 3 # requested number of correlates: l_int, a_int, b_int
    camout = np.nan*np.ones(dshape)

    # apply forward/inverse model for each row in data:
    for i in range(data.shape[0]):
        
        # stage 1: calculate photon rates of stimulus and adapting field, lmst & lmsf:
        if (inputtype != 'xyz') :
            if relative == True:
                dataw[i+1] = Lw*dataw[i+1]/100.0 # make absolute
            xyzw = spd_to_xyz(np.vstack((dataw[0],dataw[i+1])), cieobs = cieobs, relative = False)/_CMF['K'][cieobs]
            lmsf = (Yb/100.0)*683.0*np.dot(Mxyz2lms,xyzw.T).T # calculate adaptation field and convert to l,m,s

            if (direction == 'forward'):
                if relative == True:
                    data[i] = Lw*data[i]/100.0 # make absolute
                xyzt = spd_to_xyz(data[i], cieobs = cieobs, relative = False)/_CMF['K'][cieobs] 
                lmst = 683.0*np.dot(Mxyz2lms,xyzt.T).T # convert to l,m,s
            else:
                lmst = lmsf # put lmsf in lmst for inverse-mode
                
        elif (inputtype == 'xyz'):
            if relative == True: 
                dataw[i] = Lw*dataw[i]/100.0 # make absolute
            lmsw = 683.0* np.dot(Mxyz2lms, dataw[i].T).T /_CMF['K'][cieobs]  # convert to lms
            lmsf = (Yb/100.0)*lmsw

            if (direction == 'forward'):
                if relative == True:
                    data[i] = Lw*data[i]/100.0 # make absolute
                lmst = 683.0* np.dot(Mxyz2lms, data[i].T).T /_CMF['K'][cieobs] # convert to lms
            else:
                 lmst = lmsf # put lmsf in lmst for inverse-mode
                 
             
        # stage 2: calculate cone outputs of stimulus lmstp
        lmstp = math.erf(Cc*(np.log(lmst/lms0) + Cf*np.log(lmsf/lms0)))
        lmsfp = math.erf(Cc*(np.log(lmsf/lms0) + Cf*np.log(lmsf/lms0)))
        lmstp = np.vstack((lmsfp,lmstp)) # add adaptation field lms temporarily to lmsp for quick calculation

        # stage 3: calculate optic nerve signals, lam*, alphp, betp:
        lstar,alph, bet = asplit(np.dot(MAab, lmstp.T).T)
        alphp = cga1[0]*alph
        alphp[alph<0] = cga1[1]*alph[alph<0]
        betp = cgb1[0]*bet
        betp[bet<0] = cgb1[1]*bet[bet<0]
        
        # stage 4: calculate recoded nerve signals, alphapp, betapp:
        alphpp = cga2[0]*(alphp + betp)
        betpp = cgb2[0]*(alphp - betp)

        # stage 5: calculate conscious color perception:
        lstar_int = cl_int[0]*(lstar + cl_int[1])
        alph_int = cab_int[0]*(np.cos(cab_int[1]*np.pi/180.0)*alphpp - np.sin(cab_int[1]*np.pi/180.0)*betpp)
        bet_int = cab_int[0]*(np.sin(cab_int[1]*np.pi/180.0)*alphpp + np.cos(cab_int[1]*np.pi/180.0)*betpp)
        lstar_out = lstar_int
        
        if direction == 'forward':
            if Ccwb is None:
                alph_out = alph_int - cab_out[0]
                bet_out = bet_int -  cab_out[1]
            else:
                Ccwb[Ccwb<0.0] = 0.0
                Ccwb[Ccwb>1.0] = 1.0
                alph_out = alph_int - Ccwb[0]*alph_int[0] # white balance shift using adaptation gray background (Yb=20%), with Ccw: degree of adaptation
                bet_out = bet_int -  Ccwb[1]*bet_int[0]
                
            camout[i] = np.vstack((lstar_out[1:],alph_out[1:],bet_out[1:])).T # stack together and remove adaptation field from vertical stack
        
        elif direction == 'inverse':
            labf_int = np.hstack((lstar_int[0],alph_int[0],bet_int[0]))
            
            # get lstar_out, alph_out & bet_out for data:
            lstar_out, alph_out, bet_out = asplit(data[i])
            
            # stage 5 inverse: 
            # undo cortical white-balance:
            if Ccwb is None:
                alph_int = alph_out + cab_out[0]
                bet_int = bet_out +  cab_out[1]
            else:
                Ccwb[Ccwb<0.0] = 0.0
                Ccwb[Ccwb>1.0] = 1.0
                alph_int = alph_int + Ccwb[0]*alph_out[0] #  inverse white balance shift using adaptation gray background (Yb=20%), with Ccw: degree of adaptation
                bet_int = bet_int +  Ccwb[1]*bet_out[0]
            
            lstar_int = lstar_out
            alphpp = (1.0 / cab_int[0]) * (np.cos(-cab_int[1]*np.pi/180.0)*alph_int - np.sin(-cab_int[1]*np.pi/180.0)*bet_int)
            betpp = (1.0 / cab_int[0]) * (np.sin(-cab_int[1]*np.pi/180.0)*alph_int + np.cos(-cab_int[1]*np.pi/180.0)*bet_int)
            lstar_int = lstar_out
            lstar = (lstar_int /cl_int[0]) - cl_int[1] 
             
            # stage 4 inverse:
            alphp = 0.5*(alphpp/cga2[0] + betpp/cgb2[0])  # <-- alphpp = (Cga2.*(alphp+betp));
            betp = 0.5*(alphpp/cga2[0] - betpp/cgb2[0]) # <-- betpp = (Cgb2.*(alphp-betp));

            # stage 3 invers:
            alph = alphp/cga1[0]
            bet = betp/cgb1[0]
            sa = np.sign(cga1[1])
            sb = np.sign(cgb1[1])
            alph[(sa*alphp)<0.0] = alphp[(sa*alphp)<0] / cga1[1] 
            bet[(sb*betp)<0.0] = betp[(sb*betp)<0] / cgb1[1] 
            lab = ajoin((lstar, alph, bet))
            
            # stage 2 inverse:
            lmstp = np.dot(invMAab,lab.T).T 
            lmstp[lmstp<-1.0] = -1.0
            lmstp[lmstp>1.0] = 1.0

            lmstp = math.erfinv(lmstp) / Cc - Cf*np.log(lmsf/lms0)
            lmst = np.exp(lmstp) * lms0
            
            # stage 1 inverse:
            xyzt =  np.dot(invMxyz2lms,lmst.T).T   
            
            if relative == True:
                xyzt = (100.0/Lw) * xyzt
            
            camout[i] = xyzt
    
    if camout.shape[0] == 1:
        camout = np.squeeze(camout,axis = 0)
        
    return camout

       
#------------------------------------------------------------------------------
# wrapper function for use with colortf():
def xyz_to_jabM_ciecam02(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
    """
    Wrapper function for ciecam02 forward mode with J,aM,bM output.
    
    For help on parameter details: ?luxpy.cam.ciecam02 
    """
    return ciecam02(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'forward', outin = 'J,aM,bM', yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
   
def jabM_ciecam02_to_xyz(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
    """
    Wrapper function for ciecam02 inverse mode with J,aM,bM input.
    
    For help on parameter details: ?luxpy.cam.ciecam02 
    """
    return ciecam02(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'inverse', outin = 'J,aM,bM', yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)



def xyz_to_jabC_ciecam02(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
    """
    Wrapper function for ciecam02 forward mode with J,aC,bC output.
    
    For help on parameter details: ?luxpy.cam.ciecam02 
    """
    return ciecam02(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'forward', outin = 'J,aC,bC', yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
 
def jabC_ciecam02_to_xyz(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
    """
    Wrapper function for ciecam02 inverse mode with J,aC,bC input.
    
    For help on parameter details: ?luxpy.cam.ciecam02 
    """
    return ciecam02(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'inverse', outin = 'J,aC,bC', yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)


              
def xyz_to_jab_cam02ucs(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
    """
    Wrapper function for cam02ucs forward mode with J,aM,bM output.
    
    For help on parameter details: ?luxpy.cam.cam02ucs 
    """
    return cam02ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'forward', ucstype = 'ucs', yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
                
def jab_cam02ucs_to_xyz(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
    """
    Wrapper function for cam02ucs inverse mode with J,aM,bM input.
    
    For help on parameter details: ?luxpy.cam.cam02ucs 
    """
    return cam02ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'inverse', ucstype = 'ucs', yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)



def xyz_to_jab_cam02lcd(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
    """
    Wrapper function for cam02ucs forward mode with J,aMp,bMp output and ucstype = lcd.
    
    For help on parameter details: ?luxpy.cam.cam02ucs 
    """
    return cam02ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'forward', ucstype = 'lcd', yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
                
def jab_cam02lcd_to_xyz(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
    """
    Wrapper function for cam02ucs inverse mode with J,aMp,bMp input and ucstype = lcd.
    
    For help on parameter details: ?luxpy.cam.cam02ucs 
    """
    return cam02ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'inverse', ucstype = 'lcd', yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)



def xyz_to_jab_cam02scd(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
    """
    Wrapper function for cam02ucs forward mode with J,aMp,bMp output and ucstype = scd.
    
    For help on parameter details: ?luxpy.cam.cam02ucs 
    """
    return cam02ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'forward', ucstype = 'scd', yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)
                
def jab_cam02scd_to_xyz(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, yellowbluepurplecorrect = None, mcat = 'cat02'):
    """
    Wrapper function for cam02ucs inverse mode with J,aMp,bMp input and ucstype = scd.
    
    For help on parameter details: ?luxpy.cam.cam02ucs 
    """
    return cam02ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'inverse', ucstype = 'scd', yellowbluepurplecorrect = yellowbluepurplecorrect, mcat = mcat)



#------------------------------------------------------------------------------
def xyz_to_jabM_cam16(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS,  mcat = 'cat16'):
    """
    Wrapper function for cam16 forward mode with J,aM,bM output.
    
    For help on parameter details: ?luxpy.cam.cam16 
    """
    return cam16(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'forward', outin = 'J,aM,bM',  mcat = mcat)
   
def jabM_cam16_to_xyz(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS,  mcat = 'cat16'):
    """
    Wrapper function for cam16 inverse mode with J,aM,bM input.
    
    For help on parameter details: ?luxpy.cam.cam16 
    """
    return cam16(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'inverse', outin = 'J,aM,bM',  mcat = mcat)


def xyz_to_jabC_cam16(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS,  mcat = 'cat16'):
    """
    Wrapper function for cam16 forward mode with J,aC,bC output.
    
    For help on parameter details: ?luxpy.cam.cam16 
    """
    return cam16(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'forward', outin = 'J,aC,bC',  mcat = mcat)
   
def jabC_cam16_to_xyz(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS,  mcat = 'cat16'):
    """
    Wrapper function for cam16 inverse mode with J,aC,bC input.
    
    For help on parameter details: ?luxpy.cam.cam16 
    """
    return cam16(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'inverse', outin = 'J,aC,bC',  mcat = mcat)


              
def xyz_to_jab_cam16ucs(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS,  mcat = 'cat16'):
    """
    Wrapper function for cam16ucs forward mode with J,aM,bM output and ucstype = 'ucs'.
    
    For help on parameter details: ?luxpy.cam.cam16ucs 
    """
    return cam16ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'forward', ucstype = 'ucs', mcat = mcat)
                
def jab_cam16ucs_to_xyz(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, mcat = 'cat16'):
    """
    Wrapper function for cam16ucs inverse mode with J,aM,bM input and ucstype = 'ucs'.
    
    For help on parameter details: ?luxpy.cam.cam16ucs 
    """
    return cam16ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'inverse', ucstype = 'ucs', mcat = mcat)


def xyz_to_jab_cam16lcd(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS,  mcat = 'cat16'):
    """
    Wrapper function for cam16ucs forward mode with J,aM,bM output and ucstype = 'lcd'.
    
    For help on parameter details: ?luxpy.cam.cam16ucs 
    """
    return cam16ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'forward', ucstype = 'lcd', mcat = mcat)
                
def jab_cam16lcd_to_xyz(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, mcat = 'cat16'):
    """
    Wrapper function for cam16ucs inverse mode with J,aM,bM input and ucstype = 'lcd'.
    
    For help on parameter details: ?luxpy.cam.cam16ucs 
    """
    return cam16ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'inverse', ucstype = 'lcd', mcat = mcat)



def xyz_to_jab_cam16scd(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS,  mcat = 'cat16'):
    """
    Wrapper function for cam16ucs forward mode with J,aM,bM output and ucstype = 'scd'.
    
    For help on parameter details: ?luxpy.cam.cam16ucs 
    """
    return cam16ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'forward', ucstype = 'scd', mcat = mcat)
                
def jab_cam16scd_to_xyz(data, xyzw = _CAM_DEFAULT_WHITE_POINT, Yw = 100.0, conditions = _CAM_DEFAULT_CONDITIONS, mcat = 'cat16'):
    """
    Wrapper function for cam16ucs inverse mode with J,aM,bM input  and ucstype = 'scd'. 
    
    For help on parameter details: ?luxpy.cam.cam16ucs 
    """
    return cam16ucs(data, xyzw = xyzw, Yw = Yw, conditions = conditions, direction = 'inverse', ucstype = 'scd', mcat = mcat)




#------------------------------------------------------------------------------
def xyz_to_qabW_cam15u(data, fov = 10.0, parameters = None):
    """
    Wrapper function for cam15u forward mode with 'Q,aW,bW' output.
    
    For help on parameter details: ?luxpy.cam.cam15u
    """
    return cam15u(data, fov = fov, direction = 'forward', outin = 'Q,aW,bW', parameters = parameters)
                
def qabW_cam15u_to_xyz(data, fov = 10.0, parameters = None):
    """
    Wrapper function for cam15u inverse mode with 'Q,aW,bW' input.
    
    For help on parameter details: ?luxpy.cam.cam15u
    """
    return cam15u(data, fov = fov, direction = 'inverse', outin = 'Q,aW,bW', parameters = parameters)
             
#------------------------------------------------------------------------------
def xyz_to_lab_cam_sww_2016(data, dataw = None, Yb = 20.0, Lw = 400.0, relative = True, parameters = None, inputtype = 'xyz', cieobs = '2006_10'):
    """
    Wrapper function for cam_sww_2016 forward mode with 'xyz' input.
    
    For help on parameter details: ?luxpy.cam.cam_sww_2016
    """
    return cam_sww_2016(data, dataw = dataw, Yb = Yb, Lw = Lw, relative = relative, parameters = parameters, inputtype = 'xyz', direction = 'forward', cieobs = cieobs)
                
def lab_cam_sww_2016_to_xyz(data, dataw = None, Yb = 20.0, Lw = 400.0, relative = True, parameters = None, inputtype = 'xyz', cieobs = '2006_10'):
    """
    Wrapper function for cam_sww_2016 inverse mode with 'xyz' input.
    
    For help on parameter details: ?luxpy.cam.cam_sww_2016
    """
    return cam_sww_2016(data, dataw = dataw, Yb = Yb, Lw = Lw, relative = relative, parameters = parameters, inputtype = 'xyz', direction = 'inverse', cieobs = cieobs)
          
        
    
        
        
        
        
        
        
        
               
            
            
            
            
            
            
                
            
        
        
        
        
        
    