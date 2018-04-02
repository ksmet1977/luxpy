# -*- coding: utf-8 -*-
"""
###############################################################################
# Module for color rendition and color quality metrics
###############################################################################
#
# _CRI_DEFAULTS: default settings for different color rendition indices: (major dict has 9 keys (04-Jul-2017): sampleset [str/dict], ref_type [str], cieobs [str], avg [fcn handle], scale [dict], cspace [dict], catf [dict], rg_pars [dict], cri_specific_pars [dict])
#               types supported: 'ciera','ciera-8','ciera-14','cierf','iesrf','cri2012','cri2012-hl17', 'cri2012-hl1000','cri2012-real210','cqs-v7.5', 'cqs-v9.0', mcri'
#
# linear_scale():  Linear color rendering index scale from CIE13.3-1974/1995:   Rfi,a = 100 - c1*DEi,a. (c1 = 4.6)
#
# log_scale(): Log-based color rendering index scale from Davis & Ohno (2009):  Rfi,a = 10 * ln(exp((100 - c1*DEi,a)/10) + 1)
#
# psy_scale():  Psychometric based color rendering index scale from CRI2012 (Smet et al. 2013, LRT):  Rfi,a = 100 * (2 / (exp(c1*abs(DEi,a)**(c2) + 1))) ** c3
#
# process_cri_type_input(): load a cri_type dict but overwrites any keys that have a non-None input in calling function
#
# gamut_slicer(): Slices the gamut in nhbins slices and provides normalization of test gamut to reference gamut.
#
# jab_to_rg(): Calculates gamut area index, Rg based on hue-ordered jabt and jabr input (first element must also be last)
#
# spd_to_jab_t_r(): Calculates jab color values for a sample set illuminated with test source and its reference illuminant.
#                   
# spd_to_rg(): Calculates the color gamut index of data (= np.array([[wl,spds]]) (data_axis = 0) for a sample set illuminated with test source (data) with respect to some reference illuminant.
#
# spd_to_DEi(): Calculates color difference (~fidelity) of data (= np.array([[wl,spds]]) (data_axis = 0) between sample set illuminated with test source (data) and some reference illuminant.
#
# optimize_scale_factor(): Optimize scale_factor of cri-model in cri_type such that average Rf for a set of light sources is the same as that of a target-cri (default: 'ciera').
#
# spd_to_cri(): Calculates the color rendering fidelity index (CIE Ra, CIE Rf, IES Rf, CRI2012 Rf) of spectral data. 
#
# wrapper functions for fidelity type metrics:
#     spd_to_ciera(), spd_to_cierf(), spd_to_iesrf(), spd_to_cri2012(), spd_to_cri2012_hl17(), spd_to_cri2012_hl1000(), spd_to_cri2012_real210
#
# wrapper functions for gamuta area metrics:
#      spd_to_iesrf(),
#
# spd_to_mcri(): Calculates the memory color rendition index, Rm:  K. A. G. Smet, W. R. Ryckaert, M. R. Pointer, G. Deconinck, and P. Hanselaer, (2012) “A memory colour quality metric for white light sources,” Energy Build., vol. 49, no. C, pp. 216–225.
#
# spd_to_cqs(): versions 7.5 and 9.0 are supported.  W. Davis and Y. Ohno, “Color quality scale,” (2010), Opt. Eng., vol. 49, no. 3, pp. 33602–33616.   
#
#------------------------------------------------------------------------------

Created on Fri Jun 30 00:10:59 2017

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import *
from luxpy.colorappearancemodels import hue_angle
from luxpy.math import polyarea

__all__ = ['_CRI_DEFAULTS','linear_scale','log_scale','psy_scale','gamut_slicer','jab_to_rg','spd_to_rg','spd_to_DEi','spd_to_cri']
__all__ +=['spd_to_ciera','spd_to_cierf','spd_to_iesrf','spd_to_iesrg','spd_to_cri2012','spd_to_cri2012_hl17','spd_to_cri2012_hl1000','spd_to_cri2012_real210']
__all__+=['spd_to_mcri', 'spd_to_cqs']



#------------------------------------------------------------------------------
# define cri scale functions:
def linear_scale(data, scale_factor = [4.6], scale_max = 100.0): # defaults from cie-13.3-1995 cri
    """
    Linear color rendering index scale from CIE13.3-1974/1995: 
        Rfi,a = 100 - c1*DEi,a. (c1 = 4.6)
        
    Args:
        :data: float or list[floats] or numpy.ndarray 
        :scale_factor: [4.6] or list[float] or numpy.ndarray, optional
            Rescales color differences before subtracting them from :scale_max:
        :scale_max: 100.0, optional
            Maximum value of linear scale
    
    Returns:
        :returns: float or list[floats] or numpy.ndarray 
    
    References:
        ..[1] CIE13-1965. (1965). Method of measuring and specifying colour rendering properties of light sources. CIE 13. Paris, France: CIE.
    
    """
    return scale_max - scale_factor[0]*data

def log_scale(data, scale_factor = [6.73], scale_max = 100.0): # defaults from cie-224-2017 cri
    """
    Log-based color rendering index scale from Davis & Ohno (2009): 
        Rfi,a = 10 * ln(exp((100 - c1*DEi,a)/10) + 1).
                    
    Args:
        :data: float or list[floats] or numpy.ndarray 
        :scale_factor: [6.73] or list[float] or numpy.ndarray, optional
            Rescales color differences before subtracting them from :scale_max:
            Note that the default value is the one from cie-224-2017.
        :scale_max: 100.0, optional
            Maximum value of linear scale
    
    Returns:
        :returns: float or list[floats] or numpy.ndarray
        
    References:
        ..[1] Davis, W., & Ohno, Y. (2009). Approaches to color rendering measurement. 
                Journal of Modern Optics, 56(13), 1412–1419. 
        ..[2] CIE224:2017. (2017). CIE 2017 Colour Fidelity Index for accurate scientific use. Vienna, Austria.

    """
    return 10.0*np.log(np.exp((scale_max - scale_factor[0]*data)/10.0) + 1.0)

def psy_scale(data, scale_factor = [1.0/55.0, 3.0/2.0, 2.0], scale_max = 100.0): # defaults for cri2012
    """
    Psychometric based color rendering index scale from CRI2012: 
        Rfi,a = 100 * (2 / (exp(c1*abs(DEi,a)**(c2) + 1))) ** c3.
        
    Args:
        :data: float or list[floats] or numpy.ndarray 
        :scale_factor: [1.0/55.0, 3.0/2.0, 2.0] or list[float] or numpy.ndarray, optional
            Rescales color differences before subtracting them from :scale_max:
            Note that the default value is the one from (Smet et al. 2013, LRT).
        :scale_max: 100.0, optional
            Maximum value of linear scale
    
    Returns:
        :returns: float or list[floats] or numpy.ndarray
        
    References:
        ..[1] Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
            CRI2012: A proposal for updating the CIE colour rendering index. 
            Lighting Research and Technology, 45, 689–709. 
            Retrieved from http://lrt.sagepub.com/content/45/6/689    
        
    """
    return scale_max*np.power(2.0 / (np.exp(scale_factor[0]*np.power(np.abs(data),scale_factor[1])) + 1.0), scale_factor[2])

#------------------------------------------------------------------------------
# create default settings for different color rendition indices: (major dict has 9 keys (04-Jul-2017): sampleset [str/dict], ref_type [str], cieobs [str], avg [fcn handle], scale [dict], cspace [dict], catf [dict], rg_pars [dict], cri_specific_pars [dict])
_CRI_DEFAULTS = {'cri_types' : ['ciera','ciera-8','ciera-14','cierf','iesrf','iesrf-tm30-15','iesrf-tm30-18','cri2012','cri2012-hl17','cri2012-hl1000','cri2012-real210','mcri','cqs-v7.5','cqs-v9.0']}
_CRI_DEFAULTS['ciera'] = {'sampleset' : "_CRI_RFL['cie-13.3-1995']['8']", 'ref_type' : 'ciera', 'cieobs' : {'xyz': '1931_2', 'cct' : '1931_2'}, 'avg' : np.mean, 'scale' :{'fcn' : linear_scale, 'cfactor' : [4.6]}, 'cspace' : {'type':'wuv', 'xyzw' : None}, 'catf': {'xyzw':None, 'mcat':'judd-1945','D':1.0,'La':None,'cattype':'vonkries','Dtype':None, 'catmode' : '1>2'}, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False}, 'cri_specific_pars' : None}
_CRI_DEFAULTS['ciera-8'] = _CRI_DEFAULTS['ciera'].copy()
_CRI_DEFAULTS['ciera-14'] = _CRI_DEFAULTS['ciera'].copy() 
_CRI_DEFAULTS['ciera-14']['sampleset'] = "_CRI_RFL['cie-13.3-1995']['14']"
_CRI_DEFAULTS['cierf'] = {'sampleset' : "_CRI_RFL['cie-224-2017']['99']['5nm']", 'ref_type' : 'cierf', 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 'avg' : np.mean, 'scale' : {'fcn' : log_scale, 'cfactor' : [6.73]}, 'cspace' : {'type' : 'jab_cam02ucs' , 'xyzw': None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : None},'catf': None, 'rg_pars' : {'nhbins': 8, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100}, 'cri_specific_pars' : None}
_CRI_DEFAULTS['iesrf'] = {'sampleset' : "_CRI_RFL['ies-tm30-18']['99']['5nm']", 'ref_type' : 'iesrf', 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 'avg' : np.mean, 'scale' :{'fcn' : log_scale, 'cfactor' : [6.73]}, 'cspace' : {'type': 'jab_cam02ucs', 'xyzw':None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : None},'catf': None, 'rg_pars' : {'nhbins': 16, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100}, 'cri_specific_pars' : None}
_CRI_DEFAULTS['iesrf-tm30-15'] = {'sampleset' : "_CRI_RFL['ies-tm30-15']['99']['5nm']", 'ref_type' : 'iesrf', 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 'avg' : np.mean, 'scale' :{'fcn' : log_scale, 'cfactor' : [7.54]}, 'cspace' : {'type': 'jab_cam02ucs', 'xyzw':None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : None},'catf': None, 'rg_pars' : {'nhbins': 16, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100}, 'cri_specific_pars' : None}
_CRI_DEFAULTS['iesrf-tm30-18'] = {'sampleset' : "_CRI_RFL['ies-tm30-18']['99']['5nm']", 'ref_type' : 'iesrf', 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 'avg' : np.mean, 'scale' :{'fcn' : log_scale, 'cfactor' : [6.73]}, 'cspace' : {'type': 'jab_cam02ucs', 'xyzw':None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : None},'catf': None, 'rg_pars' : {'nhbins': 16, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100}, 'cri_specific_pars' : None}
_CRI_DEFAULTS['cri2012'] = {'sampleset' : "_CRI_RFL['cri2012']['HL17']", 'ref_type' : 'ciera', 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 'avg' : math.rms, 'scale' : {'fcn': psy_scale, 'cfactor' : [1/55, 3/2, 2]}, 'cspace' : {'type': 'jab_cam02ucs', 'xyzw':None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : 'brill-suss'},'catf': None, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100}, 'cri_specific_pars' : None}
_CRI_DEFAULTS['cri2012-hl17'] = _CRI_DEFAULTS['cri2012'].copy()
_CRI_DEFAULTS['cri2012-hl1000'] = {'sampleset' : "_CRI_RFL['cri2012']['HL1000']", 'ref_type' : 'ciera','cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 'avg' : math.rms,'scale': {'fcn' : psy_scale, 'cfactor' : [1/50, 3/2, 2]}, 'cspace' : {'type' : 'jab_cam02ucs','xyzw':None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : 'brill-suss'},'catf': None, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False,'normalized_chroma_ref' : 100}, 'cri_specific_pars' : None}
_CRI_DEFAULTS['cri2012-real210'] = {'sampleset' : "_CRI_RFL['cri2012']['Real210']",'ref_type' : 'ciera', 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'},'avg' : math.rms, 'scale' : {'fcn' : psy_scale, 'cfactor' : [2/45, 3/2, 2]},'cspace' : {'type': 'jab_cam02ucs', 'xyzw':None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : 'brill-suss'}, 'catf': None, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100}, 'cri_specific_pars' : None, 'normalized_chroma_ref' : 100}
_CRI_DEFAULTS['cqs-v7.5'] = {'sampleset' : "_CRI_RFL['cqs']['v7.5']",'ref_type' : 'ciera', 'cieobs' : {'xyz': '1931_2', 'cct' : '1931_2'}, 'avg' : math.rms, 'scale' : {'fcn' : log_scale, 'cfactor' : [2.93, 3.10, 3.78]}, 'cspace' : {'type': 'lab', 'xyzw' : None}, 'catf': {'xyzw': None,'mcat':'cmc','D':None,'La':[1000.0,1000.0],'cattype':'vonkries','Dtype':'cmc', 'catmode' : '1>2'}, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100}, 'cri_specific_pars' : {'maxC': None}}
_CRI_DEFAULTS['cqs-v9.0'] = {'sampleset' : "_CRI_RFL['cqs']['v9.0']", 'ref_type' : 'ciera','cieobs' : {'xyz': '1931_2', 'cct' : '1931_2'}, 'avg' : math.rms, 'scale' : {'fcn' : log_scale, 'cfactor' : [3.03, 3.20, 3.88]}, 'cspace' : {'type': 'lab', 'xyzw' : None}, 'catf': {'xyzw': None,'mcat':'cmc','D':None,'La':[1000.0,1000.0],'cattype':'vonkries','Dtype':'cmc', 'catmode' : '1>2'}, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100}, 'cri_specific_pars' : {'maxC': 10.0}}

_CRI_DEFAULTS['mcri'] = {'sampleset': "_CRI_RFL['mcri']", 'ref_type' : None, 'cieobs' : {'xyz' : '1964_10', 'cct': '1931_2'}, 'avg': math.geomean, 'scale' : {'fcn': psy_scale, 'cfactor': [21.7016,   4.2106,   2.4154]}, 'cspace': {'type': 'ipt', 'Mxyz2lms': [[ 0.400070,	0.707270,	-0.080674],[-0.228111, 1.150561,	0.061230],[0.0, 0.0,	0.931757]]}, 'catf': {'xyzw': [94.81,  100.00,  107.32], 'mcat': 'cat02', 'cattype': 'vonkries', 'F':1, 'Yb': 20.0,'Dtype':None, 'catmode' : '1>2'}, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False, 'normalized_chroma_ref' : 100}, 'cri_specific_pars' : {'similarity_ai' : np.array([[-0.09651, 0.41354, 40.64, 16.55, -0.17],[0.16548, 0.38877, 58.27,	20.37,	-0.59],[0.32825, 0.49673, 35.97	, 18.05,-6.04],[0.02115, -0.13658, 261.62, 110.99, -44.86], [-0.12686,	-0.22593, 99.06, 55.90, -39.86],[ 0.18488, 0.01172, 58.23, 62.55,	-22.86],[-0.03440,	0.23480, 94.71,	32.12, 2.90],[ 0.04258, 0.05040, 205.54,	53.08,	-35.20], [0.15829,  0.13624, 90.21,  70.83,	-19.01],[-0.01933,	-0.02168,	742.97, 297.66,	-227.30]])}}


#------------------------------------------------------------------------------
def process_cri_type_input(cri_type, args, callerfunction = ''):
    """
    Processes cri_type input in a function (helper function).
    
    This function replaces the values of keys in the cri_type dict with the corresponding not-None values in args.
    
    Args:
        :cri_type: str or dict
            Database with CRI model parameters.
        :args: arguments from a caller function
        :callerfunction: str with function the args originated from
        
    Returns:
        :cri_type: dict with database of CRI model parameters.
    """
    if isinstance(cri_type,str):
        if (cri_type in _CRI_DEFAULTS['cri_types']):
            cri_type = _CRI_DEFAULTS[cri_type].copy()
        else:
            raise Exception('.{}(): Unrecognized cri_type: {}'.format(callerfunction,cri_type))
    elif not isinstance(cri_type,dict):
        raise Exception('.{}(): cri_type is not a dict !'.format(callerfunction))
            
    cri_type = put_args_in_db(cri_type,args)
    return cri_type    


#------------------------------------------------------------------------------
def gamut_slicer(jab_test,jab_ref, out = 'jabt,jabr', nhbins = None, start_hue = 0.0, normalize_gamut = True, normalized_chroma_ref = 100, close_gamut = False):
    """
    Slices the gamut in hue bins.
    
    Args:
        :jab_test: numpy.ndarray with Cartesian color coordinates (e.g. Jab) of the samples under the test SPD
        :jab_ref:  numpy.ndarray with Cartesian color coordinates (e.g. Jab) of the samples under the reference SPD
        :out: 'jabt,jabr' or str, optional
            Specifies which variables to output as numpy.ndarray
        :nhbins: None or int, optional
            - None: defaults to using the sample hues themselves as 'bins'. In other words, the number of bins will be equal to the number of samples.
            - float: number of bins to slice the sample gamut in.
        :start_hue: 0.0 or float, optional
            Hue angle to start bin slicing
        :normalize_gamut: True or False, optional
            True normalizes the gamut of test to that of ref (perfect agreement results in circle).
        :normalized_chroma_ref: 100.0 or float, optional
            Controls the size (chroma/radius) of the normalization circle/gamut.
        :close_gamut: False or True, optional
            True appends the first jab coordinates to the end of the output (for plotting closed gamuts)
    
    Returns:
        :returns: numpy.ndarray with average jabt,jabr of each hue bin. (.shape = (number of hue bins, 3))
            (or whatever is specified in :out:) 
        
    """

    # make 3d for easy looping:
    test_original_shape = jab_test.shape

    if len(test_original_shape)<3:
        jab_test = jab_test[None]# add axis 0 #broadcast_shape(jab_test,target_shape = None,expand_2d_to_3d = 0) # expand 2-array to 3-array by adding '0'-axis
        jab_ref = jab_ref[None]# add axis 0 #broadcast_shape(jab_ref,target_shape = None,expand_2d_to_3d = 0) # expand 2-array to 3-array by adding '0'-axis
    
    #initialize Jabt, Jabr
    test_shape = list(jab_test.shape)
    if nhbins is not None:
        nhbins = np.int(nhbins)
        test_shape[0] = nhbins + close_gamut*1
    else:
        test_shape[0] = test_shape[0] + close_gamut*1
    jabt = np.zeros(test_shape)
    jabr = jabt.copy()
    binnr = jab_test[...,0].copy()
    DEi = jabt[...,0].copy()
    for ii in range(jab_test.shape[1]):
          
        # calculate hue angles:
        ht = cam.hue_angle(jab_test[:,ii,1],jab_test[:,ii,2], htype='rad')
        hr = cam.hue_angle(jab_ref[:,ii,1],jab_ref[:,ii,2], htype='rad')

        if nhbins is None:
            Ir = np.argsort(hr)
            jabtii = jab_test[Ir,ii,:]
            jabrii = jab_ref[Ir,ii,:]
            nhbins = (jabtii.shape[0])
            DEi[...,ii] =  np.sqrt(np.power((jabtii - jabtii),2).sum(axis = jabtii.ndim -1))
        else:
            
            #divide huecircle/data in n hue slices:
            hbins = np.floor(((hr - start_hue*np.pi/180)/2/np.pi) * nhbins) # because of start_hue bin range can be different from 0 : n-1
            hbins[hbins>=nhbins] = hbins[hbins>=nhbins] - nhbins # reset binnumbers to 0 : n-1 range
            hbins[hbins < 0] = (nhbins - 2) - hbins[hbins < 0] # reset binnumbers to 0 : n-1 range

            jabtii = np.zeros((nhbins,3))
            jabrii = np.zeros((nhbins,3))
            for i in range(nhbins):
                if i in hbins:
                    jabtii[i,:] = jab_test[hbins==i,ii,:].mean(axis = 0)
                    jabrii[i,:] = jab_ref[hbins==i,ii,:].mean(axis = 0)
                    DEi[i,ii] =  np.sqrt(np.power((jab_test[hbins==i,ii,:] - jab_ref[hbins==i,ii,:]),2).sum(axis = jab_test[hbins==i,ii,:].ndim -1)).mean(axis = 0)
        
        if normalize_gamut == True:
            #renormalize jabtii using jabrii:
            Ct = np.sqrt(jabtii[:,1]**2 + jabtii[:,2]**2)
            Cr = np.sqrt(jabrii[:,1]**2 + jabrii[:,2]**2)
            ht = cam.hue_angle(jabtii[:,1],jabtii[:,2], htype = 'rad')
            hr = cam.hue_angle(jabrii[:,1],jabrii[:,2], htype = 'rad')
        
            # calculate rescaled chroma of test:
            C = normalized_chroma_ref*(Ct/Cr) 
        
            # calculate normalized cart. co.: 
            jabtii[:,1] = C*np.cos(ht)
            jabtii[:,2] = C*np.sin(ht)
            jabrii[:,1] = normalized_chroma_ref*np.cos(hr)
            jabrii[:,2] = normalized_chroma_ref*np.sin(hr)
        
        if close_gamut == True:
            jabtii = np.vstack((jabtii,jabtii[0,:])) # to create closed curve when plotting
            jabrii = np.vstack((jabrii,jabrii[0,:])) # to create closed curve when plotting

        jabt[:,ii,:] = jabtii
        jabr[:,ii,:] = jabrii
        binnr[:,ii] = hbins

    # circle coordinates for plotting:
    hc = np.arange(360.0)*np.pi/180.0
    jabc = np.ones((hc.shape[0],3))*100
    jabc[:,1] = normalized_chroma_ref*np.cos(hc)
    jabc[:,2] = normalized_chroma_ref*np.sin(hc)

    if len(test_original_shape) == 2:
        jabt = jabt[:,0]
        jabr = jabr[:,0]

    if out == 'jabt,jabr':
        return jabt, jabr
    elif out == 'jabt,jabr,DEi':
        return jabt, jabr, DEi
    elif out == 'jabt,jabr,DEi,binnr':
        return jabt, jabr, DEi, binnr
    else:
        return eval(out)        
 
#------------------------------------------------------------------------------
def jab_to_rg(jabt,jabr, max_scale = 100, ordered_and_sliced = False, nhbins = None, start_hue = 0.0, normalize_gamut = True, normalized_chroma_ref = 100, out = 'Rg,jabt,jabr'):
    """
    Calculates gamut area index, Rg.
    
    Args:
        :jabt: numpy.ndarray with Cartesian color coordinates (e.g. Jab) of the samples under the test SPD
        :jabr:  numpy.ndarray with Cartesian color coordinates (e.g. Jab) of the samples under the reference SPD
        :max_scale: 100.0, optional
           Value of Rg when Rf = max_scale (i.e. DEavg = 0)
        :ordered_and_sliced: False or True, optional
           False: Hue ordering will be done with lux.cri.gamut_slicer().
           True: user is responsible for hue-ordering and closing gamut (i.e. first element in :jab: equals the last).
        :nhbins: None or int, optional
            - None: defaults to using the sample hues themselves as 'bins'. In other words, the number of bins will be equal to the number of samples.
            - float: number of bins to slice the sample gamut in.
        :start_hue: 0.0 or float, optional
            Hue angle to start bin slicing
        :normalize_gamut: True or False, optional
            True normalizes the gamut of test to that of ref (perfect agreement results in circle).
        :normalized_chroma_ref: 100.0 or float, optional
            Controls the size (chroma/radius) of the normalization circle/gamut.
        :out: 'Rg,jabt,jabr' or str, optional
            Specifies which variables to output as numpy.ndarray

    Returns: 
        :Rg: float or numpy.ndarray with gamut area indices Rg.
    """    
    # slice, order and normalize jabt and jabr:
    if ordered_and_sliced == False: 
        jabt, jabr, DEi = gamut_slicer(jabt,jabr, out = 'jabt,jabr,DEi', nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, normalized_chroma_ref = normalized_chroma_ref, close_gamut = True)
 
    # make 3d:
    test_original_shape = jabt.shape
    if len(test_original_shape)<3:
        jabt = jabt[None] #broadcast_shape(jabt,target_shape = None,expand_2d_to_3d = 0) # expand 2-array to 3-array by adding '0'-axis
        jabr = jabt[None] #broadcast_shape(jabr,target_shape = None,expand_2d_to_3d = 0) # expand 2-array to 3-array by adding '0'-axis
    
    # calculate Rg for each spd:
    Rg = np.zeros((1,jabt.shape[1]))

    for ii in range(jabt.shape[1]):
        Rg[:,ii] = max_scale*polyarea(jabt[:,ii,1],jabt[:,ii,2])/polyarea(jabr[:,ii,1],jabr[:,ii,2]) # calculate Rg =  gamut area ratio of test and ref
    
    if out == 'Rg':
        return Rg
    elif (out == 'Rg,jabt,jabr'):
        return Rg, jabt, jabr
    elif (out == 'Rg,jabt,jabr,DEi'):
        return Rg, jabt, jabr, DEi
    else:
        return eval(out)

#------------------------------------------------------------------------------
def spd_to_jab_t_r(SPD, cri_type = 'cierf', out = 'jabt,jabr', wl = None, sampleset = None, ref_type = None, cieobs  = None, cspace = None, catf = None, cri_specific_pars = None):
    """
    Calculates jab color values for a sample set illuminated with test source SPD and its reference illuminant.
        
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :out:  'jabt,jabr' or str, optional
            Specifies requested output (e.g. 'jabt,jabr' or 'jabt,jabr,cct,duv') 
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :cri_type: 'cierf' or str or dict, optional
            -'str: specifies dict with default cri model parameters (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            - dict: user defined model parameters (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] for required structure)
            Note that any non-None input arguments to the function will override default values in cri_type dict.
            
        :sampleset: None or numpy.ndarray or str, optional
            Specifies set of spectral reflectance samples for cri calculations.
                - None defaults to standard set for metric specified by cri_type.
                - numpy.ndarray: user defined set of spectral reflectance functions (.shape = (N+1, number of wavelengths); first axis are wavelengths)
        :ref_type: None or str or numpy.ndarray, optional
            Specifies type of reference illuminant type.
                - None: defaults to metric_specific reference illuminant in accordance with cri_type.
                - str: 'BB' : Blackbody radiatiors, 'DL': daylightphase, 
                        'ciera': used in CIE CRI-13.3-1995, 
                        'cierf': used in CIE 224-2017, 
                        'iesrf': used in TM30-15, ...
                - numpy.ndarray: user defined reference SPD
        :cieobs: None or dict, optional
            Specifies which CMF sets to use for the calculation of the sample XYZs and the CCT (for reference illuminant calculation).
            None defaults to the one specified in :cri_type: dict.    
                - key: 'xyz': str specifying CMF set for calculating xyz of samples and white 
                - key: 'cct': str specifying CMF set for calculating cct
        :cspace:  None or dict, optional
            Specifies which color space to use.
            None defaults to the one specified in  :cri_type: dict.  
                - key: 'type': str specifying color space used to calculate color differences
                - key: 'xyzw': None or numpy.ndarray with white point of color space
                     If None: use xyzw of test / reference (after chromatic adaptation, if specified)
                - other keys specify other possible parameters needed for color space calculation, 
                    see lx.cri._CRI_DEFAULTS['iesrf']['cspace'] for details. 
        :catf: None or dict, optional
            Perform explicit CAT before converting to color space coordinates.
                - None: don't apply a cat (other than perhaps the one built into the colorspace) 
                - dict: with CAT parameters:
                    - key: 'D': numpy.ndarray with degree of adaptation
                    - key: 'mcat': numpy.ndarray with sensor matrix specification
                    - key: 'xyzw': None or numpy.ndarray with white point
                        None: use xyzw of reference otherwise transform both test and ref to xyzw
        :cri_specific_pars: None or dict, optional
            Specifies other parameters specific to type of cri (e.g. maxC for CQS calculations)
                - None: default to the one specified in  :cri_type: dict. 
                - dict: user specified parameters. 
                    See for example luxpy.cri._CRI_DEFAULTS['mcri']['cri_specific_pars'] for its use.
    Returns:
        :returns: (numpy.ndarray, numpy.ndarray) with jabt and jabr data for :out: 'jabt,jabr'
            Other output is also possible by changing the :out: str value.
    """
   
    #Override input parameters with data specified in cri_type:
    args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_jab_t_r')
    avg, catf, cieobs, cri_specific_pars, cspace, ref_type, rg_pars, sampleset, scale = [cri_type[x] for x in sorted(cri_type.keys())] 
   
    # make SPD atleast_2d:
    SPD = np2d(SPD)

    if wl is not None: 
        SPD = spd(data = SPD, interpolation = _S_INTERP_TYPE, kind = 'np', wl = wl)
      
    # obtain sampleset:
    if isinstance(sampleset,str):
        sampleset = eval(sampleset)
    
    # A. calculate reference illuminant:
    # A.a. get xyzw:
    xyztw = spd_to_xyz(SPD, cieobs = cieobs['cct'], rfl = None, out = 1)

    # A.b. get cct:
    cct, duv = xyz_to_cct(xyztw, cieobs = cieobs['cct'], out = 'cct,duv',mode = 'lut')
    
    # A.c. get reference ill.:
    Sr = cri_ref(cct, ref_type = ref_type, cieobs = cieobs['xyz'], wl3 = SPD[0])
    
    # B. calculate xyz and xyzw of data (spds) and Sr:
    xyzti, xyztw = spd_to_xyz(SPD, cieobs = cieobs['xyz'], rfl = sampleset, out = 2)
    xyzri, xyzrw = spd_to_xyz(Sr, cieobs = cieobs['xyz'], rfl = sampleset, out = 2)

    # C. apply chromatic adaptation for non-cam/lab cspaces:
    if catf is not None:
        D_cat, Dtype_cat, La_cat, catmode_cat, cattype_cat, mcat_cat, xyzw_cat = [catf[x] for x in sorted(catf.keys())]
        
        #if not isinstance(D_cat,list): D_cat = [D_cat]
        if xyzw_cat is None: #transform from xyzwt --> xyzwr
            xyzti = cat.apply(xyzti, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw, xyzw0 = None, xyzw2 = xyzrw, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyztw = cat.apply(xyztw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw, xyzw0 = None, xyzw2 = xyzrw, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyzri = cat.apply(xyzri, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyzrw, xyzw0 = None, xyzw2 = xyzrw, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyzrw = cat.apply(xyzrw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyzrw, xyzw0 = None, xyzw2 = xyzrw, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
        else: # transform both xyzwr and xyzwt to xyzw_cat
            xyzti = cat.apply(xyzti, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw, xyzw0 = None, xyzw2 = xyzw_cat, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyztw = cat.apply(xyztw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw, xyzw0 = None, xyzw2 = xyzw_cat, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyzri = cat.apply(xyzri, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyzrw, xyzw0 = None, xyzw2 = xyzw_cat, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)
            xyzrw = cat.apply(xyzrw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyzrw, xyzw0 = None, xyzw2 = xyzw_cat, D = D_cat, La = La_cat, mcat = [mcat_cat], Dtype = Dtype_cat)

    # D. convert xyz to colorspace, cam or chromaticity co. lab (i.e. lab, ipt, Yuv, jab, wuv,..):
    # D.a. broadcast xyzw to shape of xyzi:
    xyztw = xyztw[None] #cat.broadcast_shape(xyztw,target_shape = xyzti.shape,expand_2d_to_3d = None, axis1_repeats = 1) 
    xyzrw = xyzrw[None] #cat.broadcast_shape(xyzrw,target_shape = xyzri.shape,expand_2d_to_3d = None, axis1_repeats = 1) 

    cspace_pars = cspace.copy()
    cspace_pars.pop('type')
    if 'xyzw' in cspace_pars.keys(): 
        if cspace_pars['xyzw'] is None: 
            cspace_pars['xyzw'] = xyztw # enter test whitepoint
    jabt = colortf(xyzti, tf = cspace['type'], tfa0 = cspace_pars)
    
    cspace_pars = cspace.copy()
    cspace_pars.pop('type')
    if 'xyzw' in cspace_pars.keys(): 
        if cspace_pars['xyzw'] is None: 
            cspace_pars['xyzw'] = xyzrw # enter ref. whitepoint
    jabr = colortf(xyzri, tf = cspace['type'], tfa0 = cspace_pars)    
    del cspace_pars


    # E. Regulate output:
    if out == 'jabt,jabr':
        return jabt, jabr
    elif out == 'jabt,jabr,cct,duv':
        return jabt,jabr,cct,duv
    else:
        eval(out)

#------------------------------------------------------------------------------
def jab_to_rhi(jabt, jabr, DEi, cri_type = 'cierf', start_hue = None, nhbins = None, scale_factor = None, scale_fcn = None, use_bin_avg_DEi = True):
    """
    Calculate hue bin measures: Rfhi, Rcshi and Rhshi.
    
    Rfhi: local (hue bin) color fidelity  
    Rcshi: local chroma shift
    Rhshi: local hue shift

    (See IES TM30)
    
    Args:
        :jabt: numpy.ndarray with jab coordinates under test SPD
        :jabr: numpy.ndarray with jab coordinates under reference SPD
        :DEi: numpy.ndarray with DEi (from gamut_slicer()).
        :use_bin_avg_DEi: True, optional
            Note that following IES-TM30 DEi from gamut_slicer() is obtained by averaging the DEi per hue bin (True), and
            NOT by averaging the jabt and jabr per hue  bin and then calculating the DEi (False).
        :nhbins: int, number of hue bins to slice gamut (None use the one specified in :cri_type: dict).
        :start_hue: float (°), hue at which to start slicing
        :scale_fcn: function handle to type of cri scale, 
            e.g. 
            * linear()_scale --> (100 - scale_factor*DEi), 
            * log_scale --> (cfr. Ohno's CQS), 
            * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
        :scale_factor: factors used in scaling function
        
    Returns:
        :returns: numpy.ndarrays of Rfhi, Rcshi and Rhshi
        
    References:
        ..[1] IES. (2015). IES-TM-30-15: Method for Evaluating Light Source Color Rendition. 
                New York, NY: The Illuminating Engineering Society of North America.

    """
    if isinstance(cri_type, str): 
        args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
        cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.jab_to_rhi')
    

    # Get scale factor and function:
    if (scale_factor is None):
        scale_factor = cri_type['scale']['cfactor']
    if (scale_fcn is None):
        scale_fcn = cri_type['scale']['fcn']
    if (start_hue is None):
        start_hue = cri_type['rg_pars']['start_hue']
    if (nhbins is None):
        nhbins = cri_type['rg_pars']['nhbins']
     
    # A. Local Color Fidelity, Rfhi:
    if use_bin_avg_DEi == False:
        DEi = np.power((bjabt - bjabr), 2).sum(axis = len(bjabt.shape)-1,keepdims = False)**0.5
    Rfhi = scale_fcn(DEi,scale_factor)
    
    # B.Local chroma shift and hue shift, [Rcshi, Rhshi]:
    # B.1 relative paths:
    Cr = np.sqrt((jabr[...,1:3]**2).sum(axis = jabr[...,1:3].ndim-1))
    da = (jabt[...,1] - jabr[...,1])/Cr
    db = (jabt[...,2] - jabr[...,2])/Cr

    # B.2 Reference unit circle:
    dhbins = 2*np.pi/nhbins
    hbincenters = np.arange(start_hue + dhbins/2, 2*np.pi, dhbins)
    arc = np.cos(hbincenters)[:,None]
    brc = np.sin(hbincenters)[:,None]
    
    # B.3 calculate local chroma shift, Rcshi:
    Rcshi = da * arc + db * brc
    
    # B.4 calculate local hue shift, Rcshi:
    Rhshi = db * arc - da * brc
    
    return Rfhi, Rcshi, Rhshi 


#------------------------------------------------------------------------------
def spd_to_rg(SPD, cri_type = 'cierf', out = 'Rg', wl = None, sampleset = None, ref_type = None, cieobs  = None, avg = None, cspace = None, catf = None, cri_specific_pars = None, rg_pars = {'nhbins' : None,  'normalize_gamut' : True,'start_hue' : 0, 'normalized_chroma_ref' : 100}):
    """
    Calculates the color gamut index, Rg, of spectral data. 
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :out:  'Rg' or str, optional
            Specifies requested output (e.g. 'Rg,cct,duv') 
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :cri_type: 'cierf' or str or dict, optional
            -'str: specifies dict with default cri model parameters (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            - dict: user defined model parameters (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] for required structure)
            Note that any non-None input arguments to the function will override default values in cri_type dict.
        :sampleset: None or numpy.ndarray or str, optional
            Specifies set of spectral reflectance samples for cri calculations.
                - None defaults to standard set for metric specified by cri_type.
                - numpy.ndarray: user defined set of spectral reflectance functions (.shape = (N+1, number of wavelengths); first axis are wavelengths)
        :ref_type: None or str or numpy.ndarray, optional
            Specifies type of reference illuminant type.
                - None: defaults to metric_specific reference illuminant in accordance with cri_type.
                - str: 'BB' : Blackbody radiatiors, 'DL': daylightphase, 
                        'ciera': used in CIE CRI-13.3-1995, 
                        'cierf': used in CIE 224-2017, 
                        'iesrf': used in TM30-15, ...
                - numpy.ndarray: user defined reference SPD
        :cieobs: None or dict, optional
            Specifies which CMF sets to use for the calculation of the sample XYZs and the CCT (for reference illuminant calculation).
            None defaults to the one specified in :cri_type: dict.    
                - key: 'xyz': str specifying CMF set for calculating xyz of samples and white 
                - key: 'cct': str specifying CMF set for calculating cct
        :cspace:  None or dict, optional
            Specifies which color space to use.
            None defaults to the one specified in  :cri_type: dict.  
                - key: 'type': str specifying color space used to calculate color differences
                - key: 'xyzw': None or numpy.ndarray with white point of color space
                     If None: use xyzw of test / reference (after chromatic adaptation, if specified)
                - other keys specify other possible parameters needed for color space calculation, 
                    see lx.cri._CRI_DEFAULTS['iesrf']['cspace'] for details. 
        :catf: None or dict, optional
            Perform explicit CAT before converting to color space coordinates.
                - None: don't apply a cat (other than perhaps the one built into the colorspace) 
                - dict: with CAT parameters:
                    - key: 'D': numpy.ndarray with degree of adaptation
                    - key: 'mcat': numpy.ndarray with sensor matrix specification
                    - key: 'xyzw': None or numpy.ndarray with white point
                        None: use xyzw of reference otherwise transform both test and ref to xyzw
        :cri_specific_pars: None or dict, optional
            Specifies other parameters specific to type of cri (e.g. maxC for CQS calculations)
                - None: default to the one specified in  :cri_type: dict. 
                - dict: user specified parameters. 
                    See for example luxpy.cri._CRI_DEFAULTS['mcri']['cri_specific_pars'] for its use.
        :rg_pars: {'nhbins' : None, 'start_hue' : 0, 'normalize_gamut' : True}, optional
            Dict containing specifying parameters for slicing the gamut.
                - key: 'nhbins': int, number of hue bins to slice gamut (None use the one specified in :cri_type: dict).
                - key: 'start_hue': float (°), hue at which to start slicing
                - key: 'normalize_gamut': True or False: normalize gamut or not before calculating a gamut area index Rg. 
                - key: 'normalized_chroma_ref': 100.0 or float, optional
                    Controls the size (chroma/radius) of the normalization circle/gamut.
        :avg: None or fcn handle, optional
            Averaging function (handle) for color differences, DEi (e.g. numpy.mean, .math.rms, .math.geomean)
            None use the one specified in :cri_type: dict.
        :scale: None or dict, optional
            Specifies scaling of color differences to obtain CRI.
                - None use the one specified in :cri_type: dict.
                - dict: user specified dict with scaling parameters.
                    - key: 'fcn': function handle to type of cri scale, 
                            e.g. 
                            * linear()_scale --> (100 - scale_factor*DEi), 
                            * log_scale --> (cfr. Ohno's CQS), 
                            * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
                    - key: 'cfactor': factors used in scaling function, 
                          If None: 
                              Scaling factor value(s) will be optimized to minimize 
                              the rms between the Rf's of the requested metric 
                              and some target metric specified in:
                                  - key: 'opt_cri_type':  str 
                                      * str: one of the preset _CRI_DEFAULTS
                                      * dict: user speciied (dict must contain all keys as normal)
                                     Note that if key not in :scale: dict, then 'opt_cri_type' is added with default setting = 'ciera'.
                                  - key: 'opt_spd_set': numpy.ndarray with set of light source spds used to optimize cfactor 
                                     Note that if key not in :scale: dict, then default = 'F1-F12'.

    Returns:
        :returns: float or numpy.ndarray with Rg for :out: 'Rg'
            Other output is also possible by changing the :out: str value.
            
    References:
        ..[1] IES. (2015). IES-TM-30-15: Method for Evaluating Light Source Color Rendition. 
                New York, NY: The Illuminating Engineering Society of North America.
        ..[2] David, A., Fini, P. T., Houser, K. W., Ohno, Y., Royer, M. P., Smet, K. A. G., … Whitehead, L. (2015). 
            Development of the IES method for evaluating the color rendition of light sources. 
            Optics Express, 23(12), 15888–15906. 
            https://doi.org/10.1364/OE.23.015888
    """
    #Override input parameters with data specified in cri_type:
    args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_rg')

    #avg, catf, cieobs, cieobs_cct, cri_specific_pars, cspace, cspace_pars, ref_type, rg_pars, sampleset, scale_factor, scale_fcn = [cri_type[x] for x in sorted(cri_type.keys())] 

       
    # calculate Jabt of test and Jabr of the reference illuminant corresponding to test: 
    jabt, jabr,cct,duv = spd_to_jab_t_r(SPD, cri_type = cri_type, out = 'jabt,jabr,cct,duv', wl = wl) 

    
    # calculate gamut area index:
    rg_pars = cri_type['rg_pars']
    #rg_pars = put_args_in_db(cri_type['rg_pars'],rg_pars)#{'nhbins':nhbins,'start_hue':start_hue,'normalize_gamut':normalize_gamut}) #override with not-None input from function
    nhbins, normalize_gamut, normalized_chroma_ref, start_hue  = [rg_pars[x] for x in sorted(rg_pars.keys())]
    Rg, jabt_binned, jabr_binned, DEi_binned = jab_to_rg(jabt,jabr, ordered_and_sliced = False, nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, out = 'Rg,jabt,jabr,DEi')
    Rg = np2d(Rg)
    
   
    if (out == 'Rg'):
        return Rg
    elif (out == 'Rg,jabt,jabr'):
        return Rg, jabt_binned,jabr_binned
    elif (out == 'Rg,jabt,jabr,DEi'):
        return Rg, jabt_binned,jabr_binned,DEi_binned
    else:
        return eval(out)


#------------------------------------------------------------------------------
def spd_to_DEi(SPD, cri_type = 'cierf', out = 'DEi', wl = None, sampleset = None, ref_type = None, cieobs = None, avg = None, cspace = None, catf = None, cri_specific_pars = None):
    """
    Calculates color differences (~fidelity), DEi, of spectral data.
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :out:  'DEi' or str, optional
            Specifies requested output (e.g. 'DEi,DEa,cct,duv') 
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :cri_type: 'cierf' or str or dict, optional
            -'str: specifies dict with default cri model parameters (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            - dict: user defined model parameters (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] for required structure)
            Note that any non-None input arguments to the function will override default values in cri_type dict.
        :sampleset: None or numpy.ndarray or str, optional
            Specifies set of spectral reflectance samples for cri calculations.
                - None defaults to standard set for metric specified by cri_type.
                - numpy.ndarray: user defined set of spectral reflectance functions (.shape = (N+1, number of wavelengths); first axis are wavelengths)
        :ref_type: None or str or numpy.ndarray, optional
            Specifies type of reference illuminant type.
                - None: defaults to metric_specific reference illuminant in accordance with cri_type.
                - str: 'BB' : Blackbody radiatiors, 'DL': daylightphase, 
                        'ciera': used in CIE CRI-13.3-1995, 
                        'cierf': used in CIE 224-2017, 
                        'iesrf': used in TM30-15, ...
                - numpy.ndarray: user defined reference SPD
        :cieobs: None or dict, optional
            Specifies which CMF sets to use for the calculation of the sample XYZs and the CCT (for reference illuminant calculation).
            None defaults to the one specified in :cri_type: dict.    
                - key: 'xyz': str specifying CMF set for calculating xyz of samples and white 
                - key: 'cct': str specifying CMF set for calculating cct
        :cspace:  None or dict, optional
            Specifies which color space to use.
            None defaults to the one specified in  :cri_type: dict.  
                - key: 'type': str specifying color space used to calculate color differences
                - key: 'xyzw': None or numpy.ndarray with white point of color space
                     If None: use xyzw of test / reference (after chromatic adaptation, if specified)
                - other keys specify other possible parameters needed for color space calculation, 
                    see lx.cri._CRI_DEFAULTS['iesrf']['cspace'] for details. 
        :catf: None or dict, optional
            Perform explicit CAT before converting to color space coordinates.
                - None: don't apply a cat (other than perhaps the one built into the colorspace) 
                - dict: with CAT parameters:
                    - key: 'D': numpy.ndarray with degree of adaptation
                    - key: 'mcat': numpy.ndarray with sensor matrix specification
                    - key: 'xyzw': None or numpy.ndarray with white point
                        None: use xyzw of reference otherwise transform both test and ref to xyzw
        :cri_specific_pars: None or dict, optional
            Specifies other parameters specific to type of cri (e.g. maxC for CQS calculations)
                - None: default to the one specified in  :cri_type: dict. 
                - dict: user specified parameters. 
                    See for example luxpy.cri._CRI_DEFAULTS['mcri']['cri_specific_pars'] for its use.

    Returns:
        :returns: float or numpy.ndarray with DEi for :out: 'DEi'
            Other output is also possible by changing the :out: str value.
    """
    #Override input parameters with data specified in cri_type:
    args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_DEi')

    # calculate Jabt of test and Jabr of the reference illuminant corresponding to test: 
    #Jabt, Jabr, cct, duv = spd_to_jab_t_r(data, cri_type = cri_type, out = 'Jabt,Jabr,cct,duv', wl = wl, sampleset = sampleset, cieobs  = cieobs, cieobs_cct = cieobs_cct, cspace = cspace, catf = catf, ref_type = ref_type, cspace_pars = cspace_pars,cri_specific_pars = cri_specific_pars)
    jabt, jabr, cct, duv = spd_to_jab_t_r(SPD, cri_type = cri_type, out = 'jabt,jabr,cct,duv', wl = wl)
      
    # E. calculate DEi, DEa:
    avg = cri_type['avg']
    DEi = np.power((jabt - jabr),2).sum(axis = len(jabt.shape)-1,keepdims = False)**0.5
    #DEi = np.power((jabt - jabr),2).sum(axis = len(Jabt.shape)-1,keepdims = True)**0.5
    DEa = avg(DEi, axis = 0) #len(Jabt.shape)-2)
    DEa = np2d(DEa)
  
     # output:
    if (out != 'DEi'):
        return  eval(out)
    else:
        return DEi

      
#------------------------------------------------------------------------------
def optimize_scale_factor(cri_type, opt_scale_factor, scale_fcn, avg) :
    """
    Optimize scale_factor of cri-model in cri_type such that average Rf for a set of light sources is the same as that of a target-cri (default: 'ciera').
    
    Args:
        :cri_type: 'cierf' or str or dict
            -'str: specifies dict with default cri model parameters (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            - dict: user defined model parameters (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] for required structure)
        :opt_scale: True or False
            True: optimize scaling-factor, else do nothing and use value of scaling-factor in :scale: dict.   
        :scale_fcn: function handle to type of cri scale
            e.g. 
            * linear()_scale --> (100 - scale_factor*DEi), 
            * log_scale --> (cfr. Ohno's CQS), 
            * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
        :avg: None or fcn handle
            Averaging function (handle) for color differences, DEi (e.g. numpy.mean, .math.rms, .math.geomean)
            None use the one specified in :cri_type: dict.

    Returns:
        :scaling_factor: numpy.ndarray

    """
    if  np.any(opt_scale_factor):
        if 'opt_cri_type' not in cri_type['scale'].keys(): 
            opt_cri_type = _CRI_DEFAULTS['ciera'] # use CIE Ra-13.3-1995 as target
        if 'opt_spd_set' not in cri_type['scale'].keys(): 
            opt_spd_set = _IESTM30['S']['data'][0:13] # use CIE F1-F12
        scale_fcn_opt = opt_cri_type ['scale']['fcn']
        scale_factor_opt = opt_cri_type ['scale']['cfactor']
        avg_opt = opt_cri_type ['avg']
        DEa_opt = spd_to_DEi(opt_spd_set, out ='DEa', cri_type = opt_cri_type) # DEa using target cri
        Rf_opt = avg(scale_fcn_opt(DEa_opt,scale_factor_opt))
        
        DEa = spd_to_DEi(opt_spd_set, out ='DEa', cri_type = cri_type) # DEa using current cri

        
        # optimize scale_factor to minimize rms difference:
        sf = cri_type['scale']['cfactor'] # get scale_factor of cri_type to determine len and non-optimized factors

        if (isinstance(sf,float)): #(isinstance(1.0*sf,float))
            sf = [sf]
        if isinstance(opt_scale_factor, bool):
            opt_scale_factor = [opt_scale_factor] 
        if (len(opt_scale_factor)==1) & (len(sf) == 1):
            x0 = 1
            optfcn = lambda x : math.rms(avg(scale_fcn(DEa,x)) - Rf_opt,axis=1) # optimize the only cfactor
        elif (len(opt_scale_factor)==1) & (len(sf) > 1):     
            x0 = 1
            optfcn = lambda x : math.rms(avg(scale_fcn(DEa,np.hstack( (x,sf[1:]) ))) - Rf_opt,axis=1) # optimize the first cfactor (for scale_factor input of len = 1)
        else:
            x0 = np.ones(np.sum(opt_scale_factor))
            optfcn = lambda x : math.rms(avg(scale_fcn(DEa,np.hstack( (x,sf[np.invert(opt_scale_factor)]) ))) - Rf_opt,axis=1) # optimize first N 'True' cfactor (for scale_factor input of len = n>=N)
        
        optresult = minimize(fun = optfcn, x0 = x0, args=(), method = 'Nelder-Mead')
        scale_factor = optresult['x']
        
        #Reconstruct 'scale_factor' from optimized and fixed parts:
        if (len(opt_scale_factor)==1) & (len(sf) == 1):
            pass #only cfactor
        elif (len(opt_scale_factor)==1) & (len(sf) > 1):     
            scale_factor = np.hstack( (scale_factor,sf[1:]) )
        else:
          scale_factor = np.hstack( (scale_factor,sf[np.invert(opt_scale_factor)]) ) # optimize first N 'True' cfactor (for scale_factor input of len = n>=N)

    else:
        scale_factor = cri_type['scale']['cfactor']
    return scale_factor

#------------------------------------------------------------------------------
def spd_to_cri(SPD, cri_type = 'cierf', out = 'Rf', wl = None, sampleset = None, ref_type = None, cieobs = None, avg = None, scale = None, opt_scale_factor = False, cspace = None, catf = None, cri_specific_pars = None, rg_pars = None):
    """
    Calculates the color rendering fidelity index, Rf, of spectral data. 
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :cri_type: 'cierf' or str or dict, optional
            -'str: specifies dict with default cri model parameters (for supported types, see luxpy.cri._CRI_DEFAULTS['cri_types'])
            - dict: user defined model parameters (see e.g. luxpy.cri._CRI_DEFAULTS['cierf'] for required structure)
            Note that any non-None input arguments to the function will override default values in cri_type dict.
        :sampleset: None or numpy.ndarray or str, optional
            Specifies set of spectral reflectance samples for cri calculations.
                - None defaults to standard set for metric specified by cri_type.
                - numpy.ndarray: user defined set of spectral reflectance functions (.shape = (N+1, number of wavelengths); first axis are wavelengths)
        :ref_type: None or str or numpy.ndarray, optional
            Specifies type of reference illuminant type.
                - None: defaults to metric_specific reference illuminant in accordance with cri_type.
                - str: 'BB' : Blackbody radiatiors, 'DL': daylightphase, 
                        'ciera': used in CIE CRI-13.3-1995, 
                        'cierf': used in CIE 224-2017, 
                        'iesrf': used in TM30-15, ...
                - numpy.ndarray: user defined reference SPD
        :cieobs: None or dict, optional
            Specifies which CMF sets to use for the calculation of the sample XYZs and the CCT (for reference illuminant calculation).
            None defaults to the one specified in :cri_type: dict.    
                - key: 'xyz': str specifying CMF set for calculating xyz of samples and white 
                - key: 'cct': str specifying CMF set for calculating cct
        :cspace:  None or dict, optional
            Specifies which color space to use.
            None defaults to the one specified in  :cri_type: dict.  
                - key: 'type': str specifying color space used to calculate color differences
                - key: 'xyzw': None or numpy.ndarray with white point of color space
                     If None: use xyzw of test / reference (after chromatic adaptation, if specified)
                - other keys specify other possible parameters needed for color space calculation, 
                    see lx.cri._CRI_DEFAULTS['iesrf']['cspace'] for details. 
        :catf: None or dict, optional
            Perform explicit CAT before converting to color space coordinates.
                - None: don't apply a cat (other than perhaps the one built into the colorspace) 
                - dict: with CAT parameters:
                    - key: 'D': numpy.ndarray with degree of adaptation
                    - key: 'mcat': numpy.ndarray with sensor matrix specification
                    - key: 'xyzw': None or numpy.ndarray with white point
                        None: use xyzw of reference otherwise transform both test and ref to xyzw
        :cri_specific_pars: None or dict, optional
            Specifies other parameters specific to type of cri (e.g. maxC for CQS calculations)
                - None: default to the one specified in  :cri_type: dict. 
                - dict: user specified parameters. 
                    See for example luxpy.cri._CRI_DEFAULTS['mcri']['cri_specific_pars'] for its use.
        :rg_pars: {'nhbins' : None, 'start_hue' : 0, 'normalize_gamut' : True}, optional
            Dict containing specifying parameters for slicing the gamut.
                - key: 'nhbins': int, number of hue bins to slice gamut (None use the one specified in :cri_type: dict).
                - key: 'start_hue': float (°), hue at which to start slicing
                - key: 'normalize_gamut': True or False: normalize gamut or not before calculating a gamut area index Rg. 
                - key: 'normalized_chroma_ref': 100.0 or float, optional
                    Controls the size (chroma/radius) of the normalization circle/gamut.
        :avg: None or fcn handle, optional
            Averaging function (handle) for color differences, DEi (e.g. numpy.mean, .math.rms, .math.geomean)
            None use the one specified in :cri_type: dict.
        :scale: None or dict, optional
            Specifies scaling of color differences to obtain CRI.
                - None use the one specified in :cri_type: dict.
                - dict: user specified dict with scaling parameters.
                    - key: 'fcn': function handle to type of cri scale, 
                            e.g. 
                            * linear()_scale --> (100 - scale_factor*DEi), 
                            * log_scale --> (cfr. Ohno's CQS), 
                            * psy_scale (Smet et al.'s cri2012,See: LRT 2013)
                    - key: 'cfactor': factors used in scaling function, 
                          If None: 
                              Scaling factor value(s) will be optimized to minimize 
                              the rms between the Rf's of the requested metric 
                              and some target metric specified in:
                                  - key: 'opt_cri_type':  str 
                                      * str: one of the preset _CRI_DEFAULTS
                                      * dict: user speciied (dict must contain all keys as normal)
                                     Note that if key not in :scale: dict, then 'opt_cri_type' is added with default setting = 'ciera'.
                                  - key: 'opt_spd_set': numpy.ndarray with set of light source spds used to optimize cfactor 
                                     Note that if key not in :scale: dict, then default = 'F1-F12'.
        :opt_scale: True or False, optional
            True: optimize scaling-factor, else do nothing and use value of scaling-factor in :scale: dict.   
    
    Returns:
        :returns: float or numpy.ndarray with Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
            
    References:
        ..[1] IES. (2015). IES-TM-30-15: Method for Evaluating Light Source Color Rendition. 
                New York, NY: The Illuminating Engineering Society of North America.
        ..[2] David, A., Fini, P. T., Houser, K. W., Ohno, Y., Royer, M. P., Smet, K. A. G., … Whitehead, L. (2015). 
                Development of the IES method for evaluating the color rendition of light sources. 
                Optics Express, 23(12), 15888–15906. 
                https://doi.org/10.1364/OE.23.015888
        ..[3] CIE224:2017. (2017). CIE 2017 Colour Fidelity Index for accurate scientific use. Vienna, Austria.
        ..[4] Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
                CRI2012: A proposal for updating the CIE colour rendering index. 
                Lighting Research and Technology, 45, 689–709. 
                Retrieved from http://lrt.sagepub.com/content/45/6/689    
        ..[5] CIE13.3-1995. (1995). Method of Measuring and Specifying Colour Rendering Properties of Light Sources (Vol. CIE13.3-19). Vienna, Austria: CIE.
    """
    
    #Override input parameters with data specified in cri_type:
    args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_cri')
    
    # unpack some keys:
    if (opt_scale_factor is None) | (opt_scale_factor is False):
        scale_factor = cri_type['scale']['cfactor']
    scale_fcn = cri_type['scale']['fcn']
    avg = cri_type['avg']  
    
    # Input parsing: optimize scale_factor for input based on F1-F12 (default) if scale_factor is NaN or None:
    scale_factor = optimize_scale_factor(cri_type,opt_scale_factor, scale_fcn, avg)

    if np.isnan(scale_factor).any():
        raise Exception ('Unable to optimize scale_factor.')
#         raise Exception('03-jul-2017: Provide scale_factor(s). Automatic scale_factor search under development.')

    # A. get DEi of for ciera and of requested cri metric for spds in or specified by scale_factor_optimization_spds':
    DEi, jabt, jabr, cct, duv = spd_to_DEi(SPD, out = 'DEi,jabt,jabr,cct,duv', cri_type = cri_type)
    if ('Rg' in out.split(',')) | ('Rfhi' in out.split(',')) | ('Rhshi' in out.split(',')) | ('Rcshi' in out.split(',')):
        # calculate gamut area index:
        rg_pars = cri_type['rg_pars'] 
        nhbins, normalize_gamut, normalized_chroma_ref, start_hue = [rg_pars[x] for x in sorted(rg_pars.keys())]
        Rg, jabt_binned, jabr_binned, DEi_binned = jab_to_rg(jabt,jabr, ordered_and_sliced = False, nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, out = 'Rg,jabt,jabr,DEi')
    else:
        jabt_binned, jabr_binned, DEi_binned = None, None, None
        
    # B. convert DE to color rendering index:
    Rfi = scale_fcn(DEi,scale_factor)
    Rf = np2d(scale_fcn(avg(DEi,axis = 0),scale_factor))
    
    # C. Calculate Rfhi, Rhshi and Rcshi:
    if ('Rfhi' in out.split(',')) | ('Rhshi' in out.split(',')) | ('Rcshi' in out.split(',')):
        Rfhi, Rcshi, Rhshi = jab_to_rhi(jabt = jabt_binned[:-1,...], jabr = jabr_binned[:-1,...], DEi = DEi_binned, cri_type = cri_type, scale_factor = scale_factor, scale_fcn = scale_fcn, use_bin_avg_DEi = True) # [:-1,...] removes last row from jab as this was added to close the gamut. 
 
    if (out == 'Rf'):
        return Rf
    elif (out == 'Rg'):
        return Rg
    else:
        return eval(out)

    
#------------------------------------------------------------------------------
def spd_to_ciera(SPD, out = 'Rf', wl = None):
    """
    Wrapper function the 'ciera' color rendition (fidelity) metric (CIE 13.3-1995). 
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or numpy.ndarray with CIE13.3 Ra for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] CIE13.3-1995. (1995). Method of Measuring and Specifying Colour Rendering Properties of Light Sources (Vol. CIE13.3-19). Vienna, Austria: CIE.

    """
    return spd_to_cri(SPD, cri_type = 'ciera', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cierf(SPD, out = 'Rf', wl = None):
    """
    Wrapper function the 'cierf' color rendition (fidelity) metric (CIE224-2017). 
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or numpy.ndarray with CIE224-2017 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    References:
        ..[1] CIE224:2017. (2017). CIE 2017 Colour Fidelity Index for accurate scientific use. Vienna, Austria.
    
    """
    return spd_to_cri(SPD, cri_type = 'cierf', out = out, wl = wl)


#------------------------------------------------------------------------------
def spd_to_iesrf(SPD, out = 'Rf', wl = None):
    """
    Wrapper function the 'iesrf' color rendition (fidelity) metric (IES TM30-15). 
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or numpy.ndarray with IES TM30_15 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] IES. (2015). IES-TM-30-15: Method for Evaluating Light Source Color Rendition. 
                New York, NY: The Illuminating Engineering Society of North America.
        ..[2] David, A., Fini, P. T., Houser, K. W., Ohno, Y., Royer, M. P., Smet, K. A. G., … Whitehead, L. (2015). 
                Development of the IES method for evaluating the color rendition of light sources. 
                Optics Express, 23(12), 15888–15906. 
                https://doi.org/10.1364/OE.23.015888
    
    """
    return spd_to_cri(SPD, cri_type = 'iesrf', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_iesrg(SPD, out = 'Rg', wl = None):
    """
    Wrapper function the 'spd_to_rg' color rendition gamut area metric (IES TM30-15). 
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :out:  'Rg' or str, optional
            Specifies requested output (e.g. 'Rg,Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or numpy.ndarray with IES TM30_15 Rg for :out: 'Rg'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] IES. (2015). IES-TM-30-15: Method for Evaluating Light Source Color Rendition. 
                New York, NY: The Illuminating Engineering Society of North America.
        ..[2] David, A., Fini, P. T., Houser, K. W., Ohno, Y., Royer, M. P., Smet, K. A. G., … Whitehead, L. (2015). 
                Development of the IES method for evaluating the color rendition of light sources. 
                Optics Express, 23(12), 15888–15906. 
                https://doi.org/10.1364/OE.23.015888
    
    """
    return spd_to_rg(SPD, cri_type = 'iesrf', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cri2012(SPD, out = 'Rf', wl = None):
    """
    Wrapper function the 'cri2012' color rendition (fidelity) metric
    with the spectally uniform HL17 mathematical sampleset.

    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or numpy.ndarray with CRI2012 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
            
    References:
        ..[1] Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
                CRI2012: A proposal for updating the CIE colour rendering index. 
                Lighting Research and Technology, 45, 689–709. 
                Retrieved from http://lrt.sagepub.com/content/45/6/689
    """
    return spd_to_cri(SPD, cri_type = 'cri2012', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cri2012_hl17(SPD, out = 'Rf', wl = None):
    """
    Wrapper function the 'cri2012' color rendition (fidelity) metric
    with the spectally uniform HL17 mathematical sampleset.
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or numpy.ndarray with CRI2012 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
                CRI2012: A proposal for updating the CIE colour rendering index. 
                Lighting Research and Technology, 45, 689–709. 
                Retrieved from http://lrt.sagepub.com/content/45/6/689
    """
    return spd_to_cri(SPD, cri_type = 'cri2012-hl17', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cri2012_hl1000(SPD, out = 'Rf', wl = None):
    """
    Wrapper function the 'cri2012' color rendition (fidelity) metric
    with the spectally uniform Hybrid HL1000 sampleset.
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or numpy.ndarray with CRI2012 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
                CRI2012: A proposal for updating the CIE colour rendering index. 
                Lighting Research and Technology, 45, 689–709. 
                Retrieved from http://lrt.sagepub.com/content/45/6/689
    """
    return spd_to_cri(SPD, cri_type = 'cri2012-hl1000', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cri2012_real210(SPD, out = 'Rf', wl = None):
    """
    Wrapper function the 'cri2012' color rendition (fidelity) metric 
    with the Real-210 sampleset (normally for special color rendering indices).
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :SPD:. 
            None: default to no interpolation
        :out:  'Rf' or str, optional
            Specifies requested output (e.g. 'Rf,Rfi,cct,duv') 
    
    Returns:
        :returns: float or numpy.ndarray with CRI2012 Rf for :out: 'Rf'
            Other output is also possible by changing the :out: str value.
    
    References:
        ..[1] Smet, K., Schanda, J., Whitehead, L., & Luo, R. (2013). 
                CRI2012: A proposal for updating the CIE colour rendering index. 
                Lighting Research and Technology, 45, 689–709. 
                Retrieved from http://lrt.sagepub.com/content/45/6/689
    
    """
    return spd_to_cri(SPD, cri_type = 'cri2012-real210', out = out, wl = wl)


###############################################################################
def spd_to_mcri(SPD, D = 0.9, E = None, Yb = 20.0, out = 'Rm', wl = None):
    """
    Calculates the MCRI or Memory Color Rendition Index, Rm
    
    Args: 
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :D: 0.9, optional
            Degree of adaptation.
        :E: None, optional
            Illuminance in lux (used to calculate La = (Yb/100)*(E/pi) to calculate D following the 'cat02' model). 
             If None: the degree is determined by :D:
             If (:E: is not None) & (:Yb: is None):  :E: is assumed to contain the adapting field luminance La.
        :Yb: 20.0, optional
            Luminance factor of background. (used in the calculation of La from E)
        :out:  'Rm' or str, optional
            Specifies requested output (e.g. 'Rm,Rmi,cct,duv') 
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :data:. 
            None: default to no interpolation   
    
    Returns:
        :returns: float or numpy.ndarray with MCRI Rm for :out: 'Rm'
            Other output is also possible by changing the :out: str value.        
          
    References:
        ..[1] Smet, K. A. G., Ryckaert, W. R., Pointer, M. R., Deconinck, G., & Hanselaer, P. (2012)
                A memory colour quality metric for white light sources. 
                Energy and Buildings, 49(C), 216–225. 
                https://doi.org/10.1016/j.enbuild.2012.02.008
    """
    SPD = np2d(SPD)
    
    if wl is not None: 
        data = spd(data = SPD, interpolation = _S_INTERP_TYPE, kind = 'np', wl = wl)
    
    
    # unpack metric default values:
    cri_type = 'mcri'
    avg, catf, cieobs, cri_specific_pars, cspace, ref_type, rg_pars, sampleset, scale = [_CRI_DEFAULTS[cri_type][x] for x in sorted(_CRI_DEFAULTS[cri_type].keys())] 
    similarity_ai = cri_specific_pars['similarity_ai']
    Mxyz2lms = cspace['Mxyz2lms'] 
    scale_fcn = scale['fcn']
    scale_factor = scale['cfactor']
    sampleset = eval(sampleset)
    
    
    # A. calculate xyz:
    xyzti, xyztw = spd_to_xyz(SPD, cieobs = cieobs['xyz'],  rfl = sampleset, out = 2)

    # B. perform chromatic adaptation to adopted whitepoint of ipt color space, i.e. D65:
    if catf is not None:
        Dtype_cat, F, Yb, catmode_cat, cattype_cat, mcat_cat, xyzw_cat = [catf[x] for x in sorted(catf.keys())]
        
        # calculate degree of adaptationn D:
        if E is not None:
            if Yb is not None:
                La = (Yb/100.0)*(E/np.pi)
            else:
                La = E
            D = cat.get_degree_of_adaptation(Dtype = Dtype_cat, F = F, La = La)
        if (E is None) and (D is None):
            D = 1.0 # set degree of adaptation to 1 !
        if D > 1.0: D = 1.0
        if D < 0.6: D = 0.6 # put a limit on the lowest D
        
        # apply cat:
        xyzti = cat.apply(xyzti, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw,xyzw0 = None, xyzw2 = xyzw_cat, D = D, mcat = [mcat_cat], Dtype = Dtype_cat)
        xyztw = cat.apply(xyztw, cattype = cattype_cat, catmode = catmode_cat, xyzw1 = xyztw,xyzw0 = None, xyzw2 = xyzw_cat, D = D, mcat = [mcat_cat], Dtype = Dtype_cat)
     
    # C. convert xyz to ipt and split:
    ipt = xyz_to_ipt(xyzti, cieobs = cieobs['xyz'], Mxyz2lms = Mxyz2lms) #input matrix as published in Smet et al. 2012, Energy and Buildings
    I,P,T = asplit(ipt)  

    # D. calculate specific (hue dependent) similarity indicators, Si:
    if len(xyzti.shape) == 3:
        ai = np.expand_dims(similarity_ai, axis = 1)
    else: 
        ai = similarity_ai
    a1,a2,a3,a4,a5 = asplit(ai)
    mahalanobis_d2 = (a3*np.power((P - a1),2.0) + a4*np.power((T - a2),2.0) + 2.0*a5*(P-a1)*(T-a2))
    if (len(mahalanobis_d2.shape)==3) & (mahalanobis_d2.shape[-1]==1):
        mahalanobis_d2 = mahalanobis_d2[:,:,0].T
    Si = np.exp(-0.5*mahalanobis_d2)

    # E. calculate general similarity indicator, Sa:
    Sa = avg(Si, axis = 0,keepdims = True)

    # F. rescale similarity indicators (Si, Sa) with a 0-1 scale to memory color rendition indices (Rmi, Rm) with a 0 - 100 scale:
    Rmi = scale_fcn(np.log(Si),scale_factor = scale_factor)
    Rm = np2d(scale_fcn(np.log(Sa),scale_factor = scale_factor))

    # G. calculate Rg (polyarea of test / polyarea of memory colours):
    if 'Rg' in out.split(','):
        I = I[...,None] #broadcast_shape(I, target_shape = None,expand_2d_to_3d = 0)
        a1 = a1[:,None]*np.ones(I.shape)#broadcast_shape(a1, target_shape = None,expand_2d_to_3d = 0)
        a2 = a2[:,None]*np.ones(I.shape) #broadcast_shape(a2, target_shape = None,expand_2d_to_3d = 0)
        a12 = np.concatenate((a1,a2),axis=2) #broadcast_shape(np.hstack((a1,a2)), target_shape = ipt.shape,expand_2d_to_3d = 0)
        ipt_mc = np.concatenate((I,a12),axis=2)
        nhbins, normalize_gamut, normalized_chroma_ref, start_hue  = [rg_pars[x] for x in sorted(rg_pars.keys())]
    
        Rg = jab_to_rg(ipt,ipt_mc, ordered_and_sliced = False, nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut)


    if (out != 'Rm'):
        return  eval(out)
    else:
        return Rm
    
#-----------------------------------------------------------------------------
def  spd_to_cqs(SPD, version = 'v9.0', out = 'Qa',wl = None):
    """
    Calculates CQS Qa (Qai) or Qf (Qfi) or Qp (Qpi) for versions v9.0 or v7.5.
    
    Args:
        :SPD: numpy.ndarray with spectral data (can be multiple SPDs, first axis are the wavelengths)
        :version: 'v9.0' or 'v7.5', optional
        :out:  'Qa' or str, optional
            Specifies requested output (e.g. 'Qa,Qai,Qf,cct,duv') 
        :wl: None, optional
            Wavelengths (or [start, end, spacing]) to interpolate the SPD's in :data:. 
            None: default to no interpolation   
    
    Returns:
        :returns: float or numpy.ndarray with CQS Qa for :out: 'Qa'
            Other output is also possible by changing the :out: str value. 
    
    References:
        ..[1] Davis, W., & Ohno, Y. (2010). Color quality scale. 
                Optical Engineering, 49(3), 33602–33616.
    
    """  
    outlist = out.split()    
    if isinstance(version,str):
        cri_type = 'cqs-' + version
    elif isinstance(version, dict):
        cri_type = version
     
    # calculate DEI, labti, labri and get cspace_pars and rg_pars:
    DEi, labti, labri, cct, duv, cri_type = spd_to_DEi(SPD, cri_type = cri_type, out = 'DEi,Jabt,Jabr,cct,duv,cri_type', wl = wl)
    
    # further unpack cri_type:
    scale_fcn = cri_type['scale']['fcn']     
    scale_factor = cri_type['scale']['cfactor']    
    avg = cri_type['avg']
    cri_specific_pars = cri_type['cri_specific_pars'] 
    rg_pars = cri_type['rg_pars'] 
    
    # get maxC: to limit chroma-enhancement:
    maxC = cri_specific_pars['maxC']
    
    # make 3d:
    test_original_shape = labti.shape
    if len(test_original_shape)<3:
        labti = labti[:,None] #broadcast_shape(labti,target_shape = None,expand_2d_to_3d = 0) # expand 2-array to 3-array by adding '0'-axis
        labri = labri[:,None] #broadcast_shape(labri,target_shape = None,expand_2d_to_3d = 0) # expand 2-array to 3-array by adding '0'-axis
        DEi = DEi[:,None] #broadcast_shape(DEi,target_shape = None,expand_2d_to_3d = 0) # expand 2-array to 3-array by adding '0'-axis
        cct = cct[:,None] #broadcast_shape(cct,target_shape = None,expand_2d_to_3d = 0) # expand 2-array to 3-array by adding '0'-axis
  
    # calculate Rg for each spd:
    Qf = np.zeros((1,labti.shape[1]))
    Qfi = np.zeros((labti.shape[0],labti.shape[1]))
    
    if version == 'v7.5':
        GA = (9.2672*(1.0e-11))*cct**3.0  - (8.3959*(1.0e-7))*cct**2.0 + 0.00255*cct - 1.612 
    elif version == 'v9.0':
        GA = np.ones(cct.shape)
    else:
        raise Exception ('.cri.spd_to_cqs(): Unrecognized CQS version.')
      
    if ('Qf' in outlist) | ('Qfi' in outlist):

        # loop of light source spds
        for ii in range(labti.shape[1]):
            Qfi[:,ii] = GA[ii]*scale_fcn(DEi[:,ii],[scale_factor[0]])
            Qf[:,ii] = GA[ii]*scale_fcn(avg(DEi[:,ii,None],axis = 0),[scale_factor[0]])

    if ('Qa' in outlist) | ('Qai' in outlist) | ('Qp' in outlist) | ('Qpi' in outlist):
        Qa = Qf.copy()
        Qai = Qfi.copy()
        Qp = Qf.copy()
        Qpi = Qfi.copy()
         # loop of light source spds
        for ii in range(labti.shape[1]):
            # calculate deltaC 
            deltaC = np.sqrt(np.power(labti[:,ii,1:3],2).sum(axis = 1,keepdims=True)) - np.sqrt(np.power(labri[:,ii,1:3],2).sum(axis = 1,keepdims=True)) 
            
            # limit chroma increase:
            DEi_Climited = DEi[:,ii,None].copy()
            if maxC is None:
                maxC = 10000.0
            limitC = np.where(deltaC >= maxC)
            DEi_Climited[limitC] = maxC
            p_deltaC_pos = np.where(deltaC>0.0)
            DEi_Climited[p_deltaC_pos] = np.sqrt(DEi[:,ii,None][p_deltaC_pos]**2.0 - deltaC[p_deltaC_pos]**2.0) # increase in chroma is not penalized!

            if ('Qa' in outlist) | ('Qai' in outlist):
                Qai[:,ii,None] = GA[ii]*scale_fcn(DEi_Climited,[scale_factor[1]])
                Qa[:,ii] = GA[ii]*scale_fcn(avg(DEi_Climited,axis = 0),[scale_factor[1]])
                
            if ('Qp' in outlist) | ('Qpi' in outlist):
                deltaC_pos = deltaC * (deltaC >= 0.0)
                deltaCmu = np.mean(deltaC * (deltaC >= 0.0))
                Qpi[:,ii,None] = GA[ii]*scale_fcn((DEi_Climited - deltaC_pos),[scale_factor[2]]) # or ?? np.sqrt(DEi_Climited**2 - deltaC_pos**2) ??
                Qp[:,ii] = GA[ii]*scale_fcn((avg(DEi_Climited, axis = 0) - deltaCmu),[scale_factor[2]])

    if ('Qg' in outlist):
        Qg = Qf.copy()
        for ii in range(labti.shape[1]):
            Qg[:,ii] = 100.0*polyarea(labti[:,ii,1],labti[:,ii,2])/polyarea(labri[:,ii,1],labri[:,ii,2]) # calculate Rg =  gamut area ratio of test and ref

     
    if out == 'Qa':
        return Qa
    else:
        return eval(out)
