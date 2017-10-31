# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:12:28 2017

@author: kevin.smet
"""


###################################################################################################
# chromatic adaptation models
###################################################################################################

# _white_point :   default adopted white point
# _La :  default luminance of the adaptation field
#
# _ mcats: default chromatic adaptation sensor spaces
#        - 'hpe': Hunt-Pointer-Estevez: R. W. G. Hunt, The Reproduction of Colour: Sixth Edition, 6th ed. Chichester, UK: John Wiley & Sons Ltd, 2004.
#        - 'cat02': from ciecam02: CIE159-2004, “A Colour Apperance Model for Color Management System: CIECAM02,” CIE, Vienna, 2004.
#        - 'cat02-bs':  cat02 adjusted to solve yellow-blue problem (last line = [0 0 1]): #Brill MH, Süsstrunk S. Repairing gamut problems in CIECAM02: A progress report. Color Res Appl 2008;33(5), 424–426.
#        - 'cat02-jiang': cat02 modified to solve yb-probem + purple problem: Jun Jiang, Zhifeng Wang,M. Ronnier Luo,Manuel Melgosa,Michael H. Brill,Changjun Li, Optimum solution of the CIECAM02 yellow–blue and purple problems, Color Res Appl 2015: 40(5), 491-503 
#        - 'kries'
#        - 'judd-1945': from CIE16-2004, Eq.4, a23 modified from 0.1 to 0.1020 for increased accuracy
#        - 'bfd': bradford transform :  G. D. Finlayson and S. Susstrunk, “Spectral sharpening and the Bradford transform,” 2000, vol. Proceeding, pp. 236–242.
#        - sharp': sharp transform:  S. Süsstrunk, J. Holm, and G. D. Finlayson, “Chromatic adaptation performance of different RGB sensors,” IS&T/SPIE Electronic Imaging 2001: Color Imaging, vol. 4300. San Jose, CA, January, pp. 172–183, 2001.
#        - 'cmc':  C. Li, M. R. Luo, B. Rigg, and R. W. G. Hunt, “CMC 2000 chromatic adaptation transform: CMCCAT2000,” Color Res. Appl., vol. 27, no. 1, pp. 49–58, 2002.
#        - 'ipt':  F. Ebner and M. D. Fairchild, “Development and testing of a color space (IPT) with improved hue uniformity,” in IS&T 6th Color Imaging Conference, 1998, pp. 8–13.
#        - 'lms':
#        - bianco':  S. Bianco and R. Schettini, “Two new von Kries based chromatic adaptation transforms found by numerical optimization,” Color Res. Appl., vol. 35, no. 3, pp. 184–192, 2010.
#        - bianco-pc':  S. Bianco and R. Schettini, “Two new von Kries based chromatic adaptation transforms found by numerical optimization,” Color Res. Appl., vol. 35, no. 3, pp. 184–192, 2010.
#        -'cat16': C. Li, Z. Li, Z. Wang, Y. Xu, M. R. Luo, G. Cui, M. Melgosa, M. H. Brill, and M. Pointer, “Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS,” Color Res. Appl., p. n/a–n/a.
#
# check_dimensions():  Check if dimensions of data and xyzw match. If xyzw.shape[0] > 1 then len(data.shape) > 2 & (data.shape[0] = xyzw.shape[0]).
#
# get_transfer_function():  Calculate the chromatic adaptation diagonal matrix transfer function Dt.  
#                           Default = 'vonkries' (others: 'rlab')
#
# smet2017_D(): Calculate the degree of adaptation based on chromaticity. 
#               See: Smet, K.A.G.*, Zhai, Q., Luo, M.R., Hanselaer, P., (2017),
#               Study of chromatic adaptation using memory color matches, Part II: colored illuminants, 
#               Opt. Express, 25(7), pp. 8350-8365
#
# get_degree_of_adaptation(): Calculates the degree of adaptation. 
#                             D passes either right through or D is calculated following some 
#                             D-function (Dtype) published in literature (cat02, cat16, cmccat, smet2017, manual).
#
# 
# parse_x1x2_parameters():    local helper function that parses input parameters and makes them the target_shape for easy calculation 
#
# apply(): Applies a von kries (independent rescaling of 'sensor sensitivity' = diag. tf.) to adapt from 
#          current adaptation conditions (1) to the new conditions (2). 

       
from luxpy import *

#__all__ = ['_white_point','_La', '_mcats','normalize_3x3_matrix','degree_of_adaptation_D','transferfunction_Dt','degree_of_adaptation_D','parse_x1x2_parameters','apply','smet2017_D']

_white_point = np2d([100,100,100]) #default adopted white point
_La = 100.0 #cd/m²

_mcats = {x : _cmf['M'][x] for x in _cmf['types']}
_mcats['hpe'] = _mcats['1931_2']
_mcats['cat02'] = np2d([[0.7328, 0.4296, -0.1624],[ -0.7036, 1.6975,  0.0061],[ 0.0030, 0.0136,  0.9834]])
_mcats['cat02-bs'] =np2d([[0.7328, 0.4296, -0.1624],[ -0.7036, 1.6975,  0.0061],[ 0.0, 0.0,  1.0]]) #Brill MH, Süsstrunk S. Repairing gamut problems in CIECAM02: A progress report. Color Res Appl 2008;33(5), 424–426.
_mcats['cat02-jiang-luo'] =np2d([[0.556150, 0.556150, -0.112300],[-0.507327, 1.404878, 0.102449],[0.0, 0.0, 1.0]]) # Jun Jiang, Zhifeng Wang,M. Ronnier Luo,Manuel Melgosa,Michael H. Brill,Changjun Li, Optimum solution of the CIECAM02 yellow–blue and purple problems, Color Res Appl 2015: 40(5), 491-503 
_mcats['kries'] = np2d([[0.40024, 0.70760, -0.08081],[-0.22630, 1.16532,  0.04570],[ 0.0,       0.0,        0.91822]])
_mcats['judd-1945'] = np2d([[0.0, 1.0, 0.0],[-0.460,1.360,0.102],[0.0,0.0,1.0]]) # from CIE16-2004, Eq.4, a23 modified from 0.1 to 0.1020 for increased accuracy
_mcats['bfd'] = np2d([[0.8951,0.2664,-0.1614],[-0.7502,1.7135,0.0367],[0.0389,-0.0685,1.0296]]) #also used in ciecam97s
_mcats['sharp'] = np2d([[1.2694,-0.0988,-0.1706],[-0.8364,1.8006,0.0357],[0.0297,-0.0315,1.0018]])
_mcats['cmc'] = np2d([[0.7982,  0.3389,	-0.1371],[-0.5918,  1.5512,	 0.0406],[0.0008,  0.0239,	 0.9753]])
_mcats['ipt'] = np2d([[0.4002,  0.7075, -0.0807],[-0.2280,  1.1500,  0.0612],[ 0.0,       0.0,       0.9184]])
_mcats['lms'] = np2d([[0.2070,   0.8655,  -0.0362],[-0.4307,   1.1780,   0.0949],[ 0.0865,  -0.2197,   0.4633]])
_mcats['bianco'] = np2d([[0.8752, 0.2787, -0.1539],[-0.8904, 1.8709, 0.0195],[-0.0061, 0.0162, 0.9899]]) # %Bianco, S. & Schettini, R., Two New von Kries Based Chromatic Adaptation Transforms Found by Numerical Optimization. Color Res Appl 2010, 35(3), 184-192
_mcats['biaco-pc'] = np2d([[0.6489, 0.3915, -0.0404],[-0.3775, 1.3055,  0.0720],[-0.0271, 0.0888, 0.9383]])
_mcats['cat16'] = np2d([[0.401288, 0.650173, -0.051461],[-0.250268, 1.204414, 0.045854],[-0.002079, 0.048952, 0.953127]])

def check_dimensions(data,xyzw, caller = 'cat.apply()'):
    """
    Check if dimensions of data and xyzw match. If xyzw.shape[0] > 1 then len(data.shape) > 2 & (data.shape[0] = xyzw.shape[0]).
    """
    xyzw = np2d(xyzw)
    data = np2d(data)
    if ((xyzw.shape[0]> 1)  & (data.shape[0] != xyzw.shape[0]) & (len(data.shape) == 2)):# | ((xyzw.shape[0]> 1) & (len(data.shape) == 2)):
        raise Exception('{}: Cannot match dim of xyzw with data: xyzw0.shape[0]>1 & != data.shape[0]'.format(caller))

#------------------------------------------------------------------------------
def get_transfer_function(cattype = 'vonkries', catmode = '1>0>2',lmsw1 = None,lmsw2 = None,lmsw0 = _white_point,D10 = 1.0,D20 = 1.0,La1=_La,La2=_La,La0 = _La):
    """
    Calculate the chromatic adaptation diagonal matrix transfer function Dt.
    Default = 'vonkries' (others: 'rlab')
    """

    if (catmode is None) & (cattype == 'vonkries'):
        if (lmsw1 is not None) & (lmsw2 is not None):
            catmode = '1>0>2'
        elif (lmsw2 is None) & (lmsw1 is not None): # apply one-step CAT: 1-->0
            catmode = '1>0'
        elif (lmsw1 is None) & (lmsw2 is not None):
            catmode = '0>2' # apply one-step CAT: 0-->2

    if cattype == 'vonkries':
        # Determine von Kries transfer function Dt:
        if (catmode == '1>0>2'): #2-step (forward + backward)
            Dt = (D10*lmsw0/lmsw1 + (1-D10)) / (D20*lmsw0/lmsw2 + (1-D20))
        elif (catmode == '1>0'): # 1-step (forward)
            Dt = (D10*lmsw0/lmsw1 + (1-D10))
        elif (catmode == '0>2'):# 1-step (backward)
            Dt = 1/(D20*lmsw0/lmsw2 + (1-D20))
        elif (catmode == '1>2'):# 1-step directly between 1>2
            Dt = (D10*lmsw2/lmsw1 + (1-D10))

    elif cattype == 'rlab': # Farchild 1990
        lmsw1divlmsw0 = (lmsw1/lmsw0).T
        lmsw2divlmsw0 = (lmsw2/lmsw0).T
        lmse1 = 3*lmsw1divlmsw0/lmsw1divlmsw0.sum(axis = 0)
        lmse2 = 3*lmsw2divlmsw0/lmsw2divlmsw0.sum(axis = 0)
        La1p = La1**(1/3.0)
        La2p = La2**(1/3.0)
        lmsp1 = (1 + La1p + lmse1) / (1 + La1p + 1/lmse1)
        lmsp2 = (1 + La2p + lmse2) / (1 + La2p + 1/lmse2)
        Dt =    ((lmsw2 / lmsw1).T * (lmsp1 + D10*(1 - lmsp1)) / (lmsp2 + D20*(1 - lmsp2))).T

    return Dt     
 
#------------------------------------------------------------------------------
def smet2017_D(xyzw, Dmax = None, cieobs = '1964_10'):
    """
    Calculate the degree of adaptation based on chromaticity. 
    See: Smet, K.A.G.*, Zhai, Q., Luo, M.R., Hanselaer, P., (2017),
    Study of chromatic adaptation using memory color matches, Part II: colored illuminants, 
    Opt. Express, 25(7), pp. 8350-8365
    """
    
    # Convert xyzw to log-compressed Macleod_Boyton coordinates:
    Vl, rl, bl = asplit(np.log(xyz_to_Vrb_mb(xyzw,cieobs = cieobs)))

    # apply Dmodel (technically only for cieobs = '1964_10')
    pD = (1.0e7)*np.array([0.021081326530436, 4.751255762876845, -0.000000071025181, -0.000000063627042, -0.146952821492957, 3.117390441655821]) #D model parameters for gaussian model in log(MB)-space (july 2016) 
    if Dmax is None:
        Dmax = 0.6539 # max D obtained under experimental conditions (probably too low due to dark surround leading to incomplete chromatic adaptation even for neutral illuminants resulting in background luminance (fov~50°) of 760 cd/m²)
    return Dmax*math.bvgpdf(x= rl, y=bl, mu = pD[2:4], sigmainv = np.linalg.inv(np.array([[pD[0],pD[4]],[pD[4],pD[1]]])))**pD[5]


#------------------------------------------------------------------------------
def get_degree_of_adaptation(Dtype = None, **kwargs):
    """
    Calculates the degree of adaptation. 
    D passes either right through or D is calculated following some D-function (Dtype)
    published in literature.
    """
    try:
        if Dtype is None:
            PAR = ["D"]
            D = np.array([kwargs['D']])
        elif Dtype == 'manual':
            PAR = ["D"]
            D = np.array([kwargs['D']])
        elif (Dtype == 'cat02') | (Dtype == 'cat16'):
            PAR = ["F, La"]
            F = kwargs['F']

            if isinstance(F,str): #CIECAM02 / CAT02 surround based F values 
                if (F == 'avg') | (F == 'average'):
                    F = 1
                elif (F == 'dim'):
                    F = 0.9
                elif (F == 'dark'):
                    F = 0.8
                elif (F == 'disp') | (F == 'display'):
                    F = 0.0
                else:
                    F = eval(F)
           
            F = np.array([F]) 
            La = np.array([kwargs['La']])
            D = F*(1-(1/3.6)*np.exp((-La-42)/92))
        elif Dtype == 'cmc':
            PAR = ["La, La0, order"]
            order = np.array([kwargs['order']])
            if order == '1>0':
                La1 = np.array([kwargs['La']])
                La0 = np.array([kwargs['La0']])
            elif order == '0>2':
                La0 = np.array([kwargs['La']])
                La1 = np.array([kwargs['La0']])
            elif order == '1>2':
                La1 = np.array([kwargs['La']])
                La0 = np.array([kwargs['La2']])
            D = 0.08*np.log10(La1+La0)+0.76-0.45*(La1-La0)/(La1+La0)
           
        elif 'smet2017': 
            PAR = ['xyzw','Dmax'] 
            xyzw = np.array([kwargs['xyzw']])
            Dmax = np.array([kwargs['Dmax']])
            D = smet2017_Dmodel(xyzw,Dmax = Dmax)
        else:
            PAR = ["? user defined"]
            D = np.array(eval(Dtype))

        D[np.where(D<0)] = 0
        D[np.where(D>1)] = 1

    except:
        raise Exception('degree_of_adaptation_D(): **kwargs does not contain the necessary parameters ({}) for Dtype = {}'.format(PAR,Dtype))

    return D


##------------------------------------------------------------------------------
#def parse_x1x2_parameters2(x,target_shape, catmode,expand_2d_to_3d = None, default = [1.0,1.0]):
#   """
#   Parse input parameters x and make them the target_shape for easy calculation 
#   (input in main function can now be a single value valid for all xyzw or an array 
#   with a different value for each xyzw)
#   """
#   if x is None:
#        x10 = broadcast_shape(default[0],target_shape = target_shape, expand_2d_to_3d = None, axis0_repeats=1)
#        if (catmode == '1>0>2') | (catmode == '1>2'):
#            x20 = broadcast_shape(default[1],target_shape = target_shape, expand_2d_to_3d = None,axis0_repeats=1)
#        else:
#            x20 = broadcast_shape(None,target_shape = target_shape, expand_2d_to_3d = None,axis0_repeats=1)
#   else:
#        x = np2d(x)
#        if (catmode == '1>0>2') |(catmode == '1>2'):
#            if x.shape[-1] == 2:
#                x10 = broadcast_shape(x[:,0,None],target_shape = target_shape, expand_2d_to_3d = None,axis0_repeats=1)
#                x20 = broadcast_shape(x[:,1,None],target_shape = target_shape, expand_2d_to_3d = None,axis0_repeats=1)
#            else:
#                 x10 = broadcast_shape(x,target_shape = target_shape, expand_2d_to_3d = None,axis0_repeats=1)
#                 x20 = x10.copy()
#        elif catmode == '1>0':
#            x10 = broadcast_shape(x[:,0,None],target_shape = target_shape, expand_2d_to_3d = None,axis0_repeats=1)
#            x20 = broadcast_shape(None,target_shape = target_shape, expand_2d_to_3d = None,axis0_repeats=1)
#   return x10, x20

#------------------------------------------------------------------------------
def parse_x1x2_parameters(x,target_shape, catmode,expand_2d_to_3d = None, default = [1.0,1.0]):
   """
   Parse input parameters x and make them the target_shape for easy calculation 
   (input in main function can now be a single value valid for all xyzw or an array 
   with a different value for each xyzw)
   """
   if x is None:
        x10 = np.ones(target_shape)*default[0]
        if (catmode == '1>0>2') | (catmode == '1>2'):
            x20 = np.ones(target_shape)*default[1]
        else:
            x20 = np.ones(target_shape)*np.nan
   else:
        x = np2d(x)
        if (catmode == '1>0>2') |(catmode == '1>2'):
            if x.shape[-1] == 2:
                x10 = np.ones(target_shape)*x[...,0]
                x20 = np.ones(target_shape)*x[...,1]
            else:
                 x10 = np.ones(target_shape)*x
                 x20 = x10.copy()
        elif catmode == '1>0':
            x10 = np.ones(target_shape)*x[...,0]
            x10 = np.ones(target_shape)*np.nan
   return x10, x20

#------------------------------------------------------------------------------
def apply(data, catmode = '1>0>2', cattype = 'vonkries', xyzw1 = None,xyzw2 = None,xyzw0 = None, D = None,mcat = ['cat02'], normxyz0 = None, outtype = 'xyz', La = None, F = None, Dtype = None):
    """
    Apply a von kries (independent rescaling of 'sensor sensitivity' = diag. tf.) to data = np.array([[x,y,z]]) (can be nxmx3)
    to adapt from current adaptation conditions (1) (can be nx3) to the new conditions (2) can be (nx3). D is the degree of adaptation (D10, D02).
    mcat is the matrix defining the sensor space. If La is not None: xyz are relative, if La is None: xyz are rel. or abs. 
    """
        
    if (xyzw1 is None) & (xyzw2 is None):
        return data # do nothing
    
    else:
        
        # Make data 2d:
        data = np2d(data)
        data_original_shape = data.shape
        if data.ndim < 3:
            target_shape = np.hstack((1,data.shape))
            data = data*np.ones(target_shape)
        else:
            target_shape = data.shape

        target_shape = data.shape

        # initialize xyzw0:
        if (xyzw0 is None): # set to iLL.E
            xyzw0 = np2d([100.0,100.0,100.0])
        xyzw0 = np.ones(target_shape)*xyzw0
        La0 = xyzw0[...,1,None]

        
        # Determine cat-type (1-step or 2-step) + make input same shape as data for block calculations:
        expansion_axis = np.abs(1*(len(data_original_shape)==2)-1)
        if ((xyzw1 is not None) & (xyzw2 is not None)):
            xyzw1 = xyzw1*np.ones(target_shape) 
            xyzw2 = xyzw2*np.ones(target_shape)
            default_La12 = [xyzw1[...,1,None],xyzw2[...,1,None]]
            
        elif (xyzw2 is None) & (xyzw1 is not None): # apply one-step CAT: 1-->0
            catmode = '1>0' #override catmode input
            xyzw1 = xyzw1*np.ones(target_shape)
            default_La12 = [xyzw1[...,1,None],La0]
            
        elif (xyzw1 is None) & (xyzw2 is not None):
            raise Exception("von_kries(): cat transformation '0>2' not supported, use '1>0' !")

        # Get or set La (La == None: xyz are absolute or relative, La != None: xyz are relative):  
        target_shape_1 = tuple(np.hstack((target_shape[:-1],1)))
        La1, La2 = parse_x1x2_parameters(La,target_shape = target_shape_1, catmode = catmode, expand_2d_to_3d = expansion_axis, default = default_La12)
        # Set degrees of adaptation, D10, D20:  (note D20 is degree of adaptation for 2-->0!!)
        D10, D20 = parse_x1x2_parameters(D,target_shape = target_shape_1, catmode = catmode, expand_2d_to_3d = expansion_axis)

        # Set F surround in case of Dtype == 'cat02':
        F1, F2 =  parse_x1x2_parameters(F,target_shape = target_shape_1, catmode = catmode, expand_2d_to_3d = expansion_axis)
            
        # Make xyz relative to go to relative xyz0:
        if La is None:
            data = 100*data/La1
            xyzw1 = 100*xyzw1/La1
            xyzw0 = 100*xyzw0/La0
            if (catmode == '1>0>2') | (catmode == '1>2'):
                xyzw2 = 100*xyzw2/La2 


        # transform data (xyz) to sensor space (lms) and perform cat:
        xyzc = np.ones(data.shape)*np.nan
        mcat = np.array(mcat)
        if (mcat.shape[0] != data.shape[1]) & (mcat.shape[0]==1):
            mcat = np.repeat(mcat,data.shape[1],axis = 0)
        elif (mcat.shape[0] != data.shape[1]) & (mcat.shape[0]>1):
            raise Exception('von_kries(): mcat.shape[0] > 1 and does not match data.shape[0]!')

        for i in range(xyzc.shape[1]):
            # get cat sensor matrix:
            if  mcat[i].dtype == np.float64:
                mcati = mcat[i]
            else:
                mcati = _mcats[mcat[i]]
            
            # normalize sensor matrix:
            if normxyz0 is not None:
                mcati = normalize_3x3_matrix(mcati, xyz0 = normxyz0)

            # convert from xyz to lms:
            lms = np.dot(mcati,data[:,i].T).T
            lmsw0 = np.dot(mcati,xyzw0[:,i].T).T
            if (catmode == '1>0>2') | (catmode == '1>0'):
                lmsw1 = np.dot(mcati,xyzw1[:,i].T).T
                Dpar1 = dict(D = D10[:,i], F = F1[:,i] , La = La1[:,i], La0 = La0[:,i], order = '1>0')
                D10[:,i] = get_degree_of_adaptation(Dtype = Dtype, **Dpar1) #get degree of adaptation depending on Dtype
                lmsw2 = None # in case of '1>0'
                
            if (catmode == '1>0>2'):
                lmsw2 = np.dot(mcati,xyzw2[:,i].T).T
                Dpar2 = dict(D = D20[:,i], F = F2[:,i] , La = La2[:,i], La0 = La0[:,i], order = '0>2')

                D20[:,i] = get_degree_of_adaptation(Dtype = Dtype, **Dpar2) #get degree of adaptation depending on Dtype

            if (catmode == '1>2'):
                lmsw1 = np.dot(mcati,xyzw1[:,i].T).T
                lmsw2 = np.dot(mcati,xyzw2[:,i].T).T
                Dpar12 = dict(D = D10[:,i], F = F1[:,i] , La = La1[:,i], La2 = La2[:,i], order = '1>2')
                D10[:,i] = get_degree_of_adaptation(Dtype = Dtype, **Dpar12) #get degree of adaptation depending on Dtype


            # Determine transfer function Dt:
            Dt = get_transfer_function(cattype = cattype, catmode = catmode,lmsw1 = lmsw1,lmsw2 = lmsw2,lmsw0 = lmsw0,D10 = D10[:,i], D20 = D20[:,i], La1 = La1[:,i], La2 = La2[:,i])

            # Perform cat:
            lms = np.dot(np.diagflat(Dt[0]),lms.T).T
            
            # Make xyz, lms 'absolute' again:
            if (catmode == '1>0>2'):
                lms = (La2[:,i]/La1[:,i])*lms
            elif (catmode == '1>0'):
                lms = (La0[:,i]/La1[:,i])*lms
            elif (catmode == '1>2'):
                lms = (La2[:,i]/La1[:,i])*lms
            
            # transform back from sensor space to xyz (or not):
            if outtype == 'xyz':
                xyzci = np.dot(np.linalg.inv(mcati),lms.T).T
                xyzci[np.where(xyzci<0)] = _eps
                xyzc[:,i] = xyzci
            else:
                xyzc[:,i] = lms
                
        # return data to original shape:
        if len(data_original_shape) == 2:
            xyzc = xyzc[0]
   
        return xyzc
