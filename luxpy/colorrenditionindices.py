# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 00:10:59 2017

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:42:10 2017

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
###############################################################################
# Color rendition and color quality metrics
###############################################################################
#
# _cri_defaults: default settings for different color rendition indices: (major dict has 9 keys (04-Jul-2017): sampleset [str/dict], ref_type [str], cieobs [str], avg [fcn handle], scale [dict], cspace [dict], catf [dict], rg_pars [dict], cri_specific_pars [dict])
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
# spd_to_cri(): Calculates color rendition (~fidelity) index of data (= np.array([[wl,spds]]) (data_axis = 0) free choice of:
#     * out = output requested (e.g. 'Rf', 'Rfi' or 'Rf,Rfi', or 'Rf, Rfi, cct', ...; default = 'Rf', with 'Rf' general color fidelity index, Rfi individual color fidelity indices
#        * wl: wavelengths (or [start, end, spacing]) to interpolate the SPD's in data argument to. Default = None (no interpolation) 
#        * cri_type: str input specifying dict with default settings or user defined dict with parameters specifying color rendering index specifics (see e.g. luxpy.cri._cri_defaults['cierf'])
#                    non-None input arguments to function will override defaults in cri_type dict
#        * cri_type keys / further function arguments:
#            - sampleset: np.array([[wl,rfl]]) or str for built-in rfl-set
#            - ref_type: reference illuminant type ('BB' : Blackbody radiatiors, 'DL': daylightphase, 'ciera': used in CIE CRI-13.3-1995, 'cierf': used in CIE 224-2017, 'iesrf': used in TM30-15, ...)
#            - cieobs: dict: 
#                + 'xyz': cie observer for calculating xyz of samples and white 
#                + 'cct': cie observer for calculating cct
#
#            - cspace: 
#                + 'type': color space used to calculate color differences
#                + 'xyzw': white point of color space, (None: use xyzw of test / reference (after chromatic adaptation, if specified))
#                + ' ...' : other possible parameters needed for color space calculation
#            - catf: None: don't apply a cat (other than perhaps the one built into the colorspace), 
#                   OR dict:
#                       - 'D': degree of adaptation
#                       - 'mcat': sensor matrix specification,
#                       - 'xyzw': (None: use xyzw of reference otherwise transform both test and ref to xyzw)
#            - avg: averaging function (handle) for color differences, DEi (e.g. numpy.mean, .math.rms, .math.geomean)
#            - scale
#                + 'fcn': function handle to type of cri scale, 
#                    e.g. 
#                    linear()_scale --> (100 - scale_factor*DEi), 
#                    log_scale --> (cfr. Ohno's CQS), 
#                    psy_scale (Smet et al.'s cri2012,See: LRT 2013)
#                + 'cfactor': factors used in scaling function, 
#                          if True: 
#                              will be optimized to minimize the rms between the Rf's of the requested metric and some target metric specified in:
#                                  + opt_cri_type:  str (one of the preset _cri_defaults) or dict (dict must contain all keys as normal)
#                                        default = 'ciera' (if 'opt_cri_type' -key not in 'scale' dict)
#                                  + opt_spd_set: set of light source spds used to optimize cfactor 
#                                        default = 'F1-F12' (if 'opt_spd_set' -key not in 'scale' dict)
#            - opt_scale_factor: True: optimize c-factor, else do nothing and use value of cfactor in 'scale'.    
#            - cri_specific_pars: other parameters specific to type of cri, e.g. maxC for CQS calculations
#            - rg_pars: dict containing:
#                + 'nhbins' (int): number of hue bins to divide the gamut in
#                + 'start_hue' (float,°): hue at which to start slicing
#                + 'normalize_gamut' (bool): normalize gamut or not before calculating a gamut area index Rg. 
#
# wrapper functions for fidelity type metrics:
#               spd_to_ciera(), spd_to_cierf(), spd_to_iesrf(), spd_to_cri2012(), spd_to_cri2012_hl17(), spd_to_cri2012_hl1000(), spd_to_cri2012_real210
#
# spd_to_mcri(): Calculates the memory color rendition index, Rm:  K. A. G. Smet, W. R. Ryckaert, M. R. Pointer, G. Deconinck, and P. Hanselaer, (2012) “A memory colour quality metric for white light sources,” Energy Build., vol. 49, no. C, pp. 216–225.
#
# spd_to_cqs(): versions 7.5 and 9.0 are supported.  W. Davis and Y. Ohno, “Color quality scale,” (2010), Opt. Eng., vol. 49, no. 3, pp. 33602–33616.   
#
#
from luxpy import *
from luxpy.colorappearancemodels import hue_angle
from luxpy.math import polyarea

__all__ = ['cie_ra','_cri_defaults','linsear_scale','log_scale','psy_scale','gamut_slicer','jab_to_rg','spd_to_rg','spd_to_DEi','spd_to_cri']
__all__ +=['spd_to_ciera','spd_to_cierf','spd_to_iesrf','spd_to_cri2012','spd_to_cri2012_hl17','spd_to_cri2012_hl1000','spd_to_cri2012_real201']
__all__+=['spd_to_mcri', 'spd_to_cqs']



#------------------------------------------------------------------------------
# define cri scale functions:
def linear_scale(data, scale_factor = [4.6], scale_max = 100.0): # defaults from cie-13.3-1995 cri
    """
    Linear color rendering index scale from CIE13.3-1974/1995: 
        Rfi,a = 100 - c1*DEi,a. (c1 = 4.6)
    """
    return scale_max - scale_factor[0]*data

def log_scale(data, scale_factor = [6.73], scale_max = 100.0): # defaults from cie-224-2017 cri
    """
    Log-based color rendering index scale from Davis & Ohno (2009): 
        Rfi,a = 10 * ln(exp((100 - c1*DEi,a)/10) + 1).
    """
    return 10.0*np.log(np.exp((scale_max - scale_factor[0]*data)/10.0) + 1.0)

def psy_scale(data, scale_factor = [1.0/55.0, 3.0/2.0, 2.0], scale_max = 100.0): # defaults for cri2012
    """
    Psychometric based color rendering index scale from CRI2012 (Smet et al. 2013, LRT): 
        Rfi,a = 100 * (2 / (exp(c1*abs(DEi,a)**(c2) + 1))) ** c3.
    """
    return scale_max*np.power(2.0 / (np.exp(scale_factor[0]*np.power(np.abs(data),scale_factor[1])) + 1.0), scale_factor[2])

#------------------------------------------------------------------------------
# create default settings for different color rendition indices: (major dict has 9 keys (04-Jul-2017): sampleset [str/dict], ref_type [str], cieobs [str], avg [fcn handle], scale [dict], cspace [dict], catf [dict], rg_pars [dict], cri_specific_pars [dict])
_cri_defaults = {'cri_types' : ['ciera','ciera-8','ciera-14','cierf','iesrf','cri2012','cri2012-hl17','cri2012-hl1000','cri2012-real210','mcri','cqs-v7.5','cqs-v9.0']}
_cri_defaults['ciera'] = {'sampleset' : "_CRI_RFL['cie-13.3-1995']['8']", 'ref_type' : 'ciera', 'cieobs' : {'xyz': '1931_2', 'cct' : '1931_2'}, 'avg' : np.mean, 'scale' :{'fcn' : linear_scale, 'cfactor' : [4.6]}, 'cspace' : {'type':'wuv', 'xyzw' : None}, 'catf': {'xyzw':None, 'mcat':'judd-1945','D':1.0,'La':None,'cattype':'vonkries','Dtype':None, 'catmode' : '1>2'}, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False}, 'cri_specific_pars' : None}
_cri_defaults['ciera-8'] = _cri_defaults['ciera'].copy()
_cri_defaults['ciera-14'] = _cri_defaults['ciera'].copy() 
_cri_defaults['ciera-14']['sampleset'] = "_CRI_RFL['cie-13.3-1995']['14']"
_cri_defaults['cierf'] = {'sampleset' : "_CRI_RFL['cie-224-2017']['99']['5nm']", 'ref_type' : 'cierf', 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 'avg' : np.mean, 'scale' : {'fcn' : log_scale, 'cfactor' : [6.73]}, 'cspace' : {'type' : 'jab_cam02ucs' , 'xyzw': None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : None},'catf': None, 'rg_pars' : {'nhbins': 8.0, 'start_hue':0.0, 'normalize_gamut': False}, 'cri_specific_pars' : None}
_cri_defaults['iesrf'] = {'sampleset' : "_CRI_RFL['ies-tm30-15']['99']['5nm']", 'ref_type' : 'iesrf', 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 'avg' : np.mean, 'scale' :{'fcn' : log_scale, 'cfactor' : [7.54]}, 'cspace' : {'type': 'jab_cam02ucs', 'xyzw':None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : None},'catf': None, 'rg_pars' : {'nhbins': 16.0, 'start_hue':0.0, 'normalize_gamut': False}, 'cri_specific_pars' : None}
_cri_defaults['cri2012'] = {'sampleset' : "_CRI_RFL['cri2012']['HL17']", 'ref_type' : 'ciera', 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 'avg' : math.rms, 'scale' : {'fcn': psy_scale, 'cfactor' : [1/55, 3/2, 2]}, 'cspace' : {'type': 'jab_cam02ucs', 'xyzw':None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : 'brill-suss'},'catf': None, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False}, 'cri_specific_pars' : None}
_cri_defaults['cri2012-hl17'] = _cri_defaults['cri2012'].copy()
_cri_defaults['cri2012-hl1000'] = {'sampleset' : "_CRI_RFL['cri2012']['HL1000']", 'ref_type' : 'ciera','cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'}, 'avg' : math.rms,'scale': {'fcn' : psy_scale, 'cfactor' : [1/50, 3/2, 2]}, 'cspace' : {'type' : 'jab_cam02ucs','xyzw':None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : 'brill-suss'},'catf': None, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False}, 'cri_specific_pars' : None}
_cri_defaults['cri2012-real210'] = {'sampleset' : "_CRI_RFL['cri2012']['Real210']",'ref_type' : 'ciera', 'cieobs' : {'xyz': '1964_10', 'cct' : '1931_2'},'avg' : math.rms, 'scale' : {'fcn' : psy_scale, 'cfactor' : [2/45, 3/2, 2]},'cspace' : {'type': 'jab_cam02ucs', 'xyzw':None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : 'brill-suss'}, 'catf': None, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False}, 'cri_specific_pars' : None}
_cri_defaults['cqs-v7.5'] = {'sampleset' : "_CRI_RFL['cqs']['v7.5']",'ref_type' : 'ciera', 'cieobs' : {'xyz': '1931_2', 'cct' : '1931_2'}, 'avg' : math.rms, 'scale' : {'fcn' : log_scale, 'cfactor' : [2.93, 3.10, 3.78]}, 'cspace' : {'type': 'lab', 'xyzw' : None}, 'catf': {'xyzw': None,'mcat':'cmc','D':None,'La':[1000.0,1000.0],'cattype':'vonkries','Dtype':'cmc', 'catmode' : '1>2'}, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False}, 'cri_specific_pars' : {'maxC': None}}
_cri_defaults['cqs-v9.0'] = {'sampleset' : "_CRI_RFL['cqs']['v9.0']", 'ref_type' : 'ciera','cieobs' : {'xyz': '1931_2', 'cct' : '1931_2'}, 'avg' : math.rms, 'scale' : {'fcn' : log_scale, 'cfactor' : [3.03, 3.20, 3.88]}, 'cspace' : {'type': 'lab', 'xyzw' : None}, 'catf': {'xyzw': None,'mcat':'cmc','D':None,'La':[1000.0,1000.0],'cattype':'vonkries','Dtype':'cmc', 'catmode' : '1>2'}, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False}, 'cri_specific_pars' : {'maxC': 10.0}}

_cri_defaults['mcri'] = {'sampleset': "_CRI_RFL['mcri']", 'ref_type' : None, 'cieobs' : {'xyz' : '1964_10', 'cct': '1931_2'}, 'avg': math.geomean, 'scale' : {'fcn': psy_scale, 'cfactor': [21.7016,   4.2106,   2.4154]}, 'cspace': {'type': 'ipt', 'Mxyz2lms': [[ 0.400070,	0.707270,	-0.080674],[-0.228111, 1.150561,	0.061230],[0.0, 0.0,	0.931757]]}, 'catf': {'xyzw': [94.81,  100.00,  107.32], 'mcat': 'cat02', 'cattype': 'vonkries', 'F':1, 'Yb': 20.0,'Dtype':None, 'catmode' : '1>2'}, 'rg_pars' : {'nhbins': None, 'start_hue':0.0, 'normalize_gamut': False}, 'cri_specific_pars' : {'similarity_ai' : np.array([[-0.09651, 0.41354, 40.64, 16.55, -0.17],[0.16548, 0.38877, 58.27,	20.37,	-0.59],[0.32825, 0.49673, 35.97	, 18.05,-6.04],[0.02115, -0.13658, 261.62, 110.99, -44.86], [-0.12686,	-0.22593, 99.06, 55.90, -39.86],[ 0.18488, 0.01172, 58.23, 62.55,	-22.86],[-0.03440,	0.23480, 94.71,	32.12, 2.90],[ 0.04258, 0.05040, 205.54,	53.08,	-35.20], [0.15829,  0.13624, 90.21,  70.83,	-19.01],[-0.01933,	-0.02168,	742.97, 297.66,	-227.30]])}}


#------------------------------------------------------------------------------
def process_cri_type_input(cri_type, args, callerfunction = ''):
    if isinstance(cri_type,str):
        if (cri_type in _cri_defaults['cri_types']):
            cri_type = _cri_defaults[cri_type].copy()
        else:
            raise Exception('.{}(): Unrecognized cri_type: {}'.format(callerfunction,cri_type))
    elif not isinstance(cri_type,dict):
        raise Exception('.{}(): cri_type is not a dict !'.format(callerfunction))
            
    cri_type = put_args_in_db(cri_type,args)
    return cri_type    


#------------------------------------------------------------------------------
def gamut_slicer(jab_test,jab_ref, out = 'jabt,jabr', nhbins = None, start_hue = 0.0, normalize_gamut = True, normalized_chroma_ref = 100, close_gamut = False):
    """
    Slices the gamut in nhbins slices.
        * normalize is True: normalizes the gamut of test to that of ref (perfect agreement results in circle). 
        * normalized_chroma_ref controls the size of the normalization circle/gamut.
        * close_gamut: appends the first jab coordinates at the end of the output (for plotting closed gamuts) 
    """
    # make 3d for easy looping:
    test_original_shape = jab_test.shape

    if len(test_original_shape)<3:
        jab_test = jab_test[None]# add axis 0 #broadcast_shape(jab_test,target_shape = None,expand_2d_to_3d = 0) # expand 2-array to 3-array by adding '0'-axis
        jab_ref = jab_ref[None]# add axis 0 #broadcast_shape(jab_ref,target_shape = None,expand_2d_to_3d = 0) # expand 2-array to 3-array by adding '0'-axis
    
    #initialize Jabt, Jabr
    test_shape = list(jab_test.shape)
    if nhbins is not None:
        test_shape[0] = nhbins + close_gamut*1
    else:
        test_shape[0] = test_shape[0] + close_gamut*1
    Jabt = np.zeros(test_shape)
    Jabr = Jabt.copy()
    for ii in range(jab_test.shape[1]):
          
        # calculate hue angles:
        ht = cam.hue_angle(jab_test[:,ii,1],jab_test[:,ii,2], htype='rad')
        hr = cam.hue_angle(jab_ref[:,ii,1],jab_ref[:,ii,2], htype='rad')

        if nhbins is None:
            Ir = np.argsort(hr)
            jabt = jab_test[Ir,ii,:]
            jabr = jab_ref[Ir,ii,:]
            nhbins = (jabt.shape[0])

        else:
            
            #divide huecircle/data in n hue slices:
            hbins = np.floor(((hr - start_hue*np.pi/180)/2/np.pi) * nhbins) # because of start_hue bin range can be different from 0 : n-1
            hbins[hbins>=nhbins] = hbins[hbins>=nhbins] - nhbins # reset binnumbers to 0 : n-1 range
            hbins[hbins < 0] = (nhbins - 2) - hbins[hbins < 0] # reset binnumbers to 0 : n-1 range
            jabt = np.zeros((nhbins,3))
            jabr = np.zeros((nhbins,3))
            for i in range(nhbins):
                if i in hbins:
                    jabt[i,:] = jab_test[hbins==i,ii,:].mean(axis = 0)
                    jabr[i,:] = jab_ref[hbins==i,ii,:].mean(axis = 0)

        if normalize_gamut == True:
            #renormalize jabt using jabr:
            Ct = np.sqrt(jabt[:,1]**2 + jabt[:,2]**2)
            Cr = np.sqrt(jabr[:,1]**2 + jabr[:,2]**2)
            ht = cam.hue_angle(jabt[:,1],jabt[:,2], htype = 'rad')
            hr = cam.hue_angle(jabr[:,1],jabr[:,2], htype = 'rad')
        
            # calculate rescaled chroma of test:
            C = normalized_chroma_ref*(Ct/Cr) 
        
            # calculate normalized cart. co.: 
            jabt[:,2] = C*np.cos(ht)
            jabt[:,3] = C*np.sin(ht)
            jabr[:,2] = normalized_chroma_ref*np.cos(hr)
            jabr[:,3] = normalized_chroma_ref*np.sin(hr)
        
        if close_gamut == True:
            jabt = np.vstack((jabt,jabt[0,:])) # to create closed curve when plotting
            jabr = np.vstack((jabr,jabr[0,:])) # to create closed curve when plotting

        Jabt[:,ii,:] = jabt
        Jabr[:,ii,:] = jabr

    # circle coordinates for plotting:
    hc = np.arange(360.0)*np.pi/180.0
    jabc = np.zeros((hc.shape[0],2))
    jabc[:,0] = normalized_chroma_ref*np.cos(hc)
    jabc[:,0] = normalized_chroma_ref*np.sin(hc)

    if len(test_original_shape) == 2:
        Jabt = Jabt[:,0]
        Jabr = Jabr[:,0]

    if out == 'Jabt,Jabr':
        return Jabt, Jabr
    else:
        return eval(out)        
 
#------------------------------------------------------------------------------
def jab_to_rg(jabt,jabr, max_scale = 100, ordered_and_sliced = False, nhbins = None, start_hue = 0.0, normalize_gamut = True, normalized_chroma_ref = 100):
    """
    Calculates gamut area index, Rg based on hue-ordered jabt and jabr input (first element must also be last).
    """    
    # slice, order and normalize jabt and jabr:
    if ordered_and_sliced == False: 
        jabt, jabr = gamut_slicer(jabt,jabr, out = 'Jabt,Jabr', nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, normalized_chroma_ref = normalized_chroma_ref, close_gamut = True)

    # make 3d:
    test_original_shape = jabt.shape
    if len(test_original_shape)<3:
        jabt = jabt[None] #broadcast_shape(jabt,target_shape = None,expand_2d_to_3d = 0) # expand 2-array to 3-array by adding '0'-axis
        jabr = jabt[None] #broadcast_shape(jabr,target_shape = None,expand_2d_to_3d = 0) # expand 2-array to 3-array by adding '0'-axis
    
    # calculate Rg for each spd:
    Rg = np.zeros((1,jabt.shape[1]))

    for ii in range(jabt.shape[1]):
        Rg[:,ii] = max_scale*polyarea(jabt[:,ii,1],jabt[:,ii,2])/polyarea(jabr[:,ii,1],jabr[:,ii,2]) # calculate Rg =  gamut area ratio of test and ref
    return Rg

#------------------------------------------------------------------------------
def spd_to_jab_t_r(data, cri_type = 'cierf', out = 'Jabt,Jabr', wl = None, sampleset = None, ref_type = None, cieobs  = None, cspace = None, catf = None, cri_specific_pars = None):
    """
    Calculates jab color values for a sample set illuminated with test source (data) and its reference illuminant.
        * out = output requested (e.g. 'Jabt,Jabr' or 'Jabt,Jabr, cct, duv') 
        * wl: wavelengths (or [start, end, spacing]) to interpolate the SPD's in data argument to. Default = None (no interpolation) 
        * cri_type: str input specifying dict with default settings or user defined dict with parameters specifying color rendering index specifics (see e.g. luxpy.cri._cri_defaults['cierf'])
                    non-None input arguments to function will override defaults in cri_type dict
        * cri_type keys / further function arguments:
            - sampleset: np.array([[wl,rfl]]) or str for built-in rfl-set
            - ref_type: reference illuminant type ('BB' : Blackbody radiatiors, 'DL': daylightphase, 'ciera': used in CIE CRI-13.3-1995, 'cierf': used in CIE 224-2017, 'iesrf': used in TM30-15, ...)
            - cieobs: dict: 
                + 'xyz': cie observer for calculating xyz of samples and white 
                + 'cct': cie observer for calculating cct

            - cspace: 
                + 'type': color space used to calculate color differences
                + 'xyzw': white point of color space, (None: use xyzw of test / reference (after chromatic adaptation, if specified))
                + ' ...' : other possible parameters needed for color space calculation
            - catf: None: don't apply a cat (other than perhaps the one built into the colorspace), 
                   OR dict:
                       - 'D': degree of adaptation
                       - 'mcat': sensor matrix specification,
                       - 'xyzw': (None: use xyzw of reference otherwise transform both test and ref to xyzw)
                           
            - cri_specific_pars: other parameters specific to type of cri, e.g. maxC for CQS calculations
    """
   
    #Override input parameters with data specified in cri_type:
    args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_jab_t_r')
    avg, catf, cieobs, cri_specific_pars, cspace, ref_type, rg_pars, sampleset, scale = [cri_type[x] for x in sorted(cri_type.keys())] 
   
    # make data atleast_2d:
    data = np2d(data)

    if wl is not None: 
        data = spd(data = data, interpolation = _S_INTERP_TYPE, kind = 'np', wl = wl)
      
    # obtain sampleset:
    if isinstance(sampleset,str):
        sampleset = eval(sampleset)
    
    # A. calculate reference illuminant:
    # A.a. get xyzw:
    xyztw = spd_to_xyz(data, cieobs = cieobs['cct'], rfl = None, out = 1)

    # A.b. get cct:
    cct, duv = xyz_to_cct(xyztw, cieobs = cieobs['cct'], out = 'cct,duv',mode = 'lut')
    
    # A.c. get reference ill.:
    Sr = cri_ref(cct, ref_type = ref_type, cieobs = cieobs['xyz'], wl3 = data[0])
    
    # B. calculate xyz and xyzw of data (spds) and Sr:
    xyzti, xyztw = spd_to_xyz(data, cieobs = cieobs['xyz'], rfl = sampleset, out = 2)
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
    Jabt = colortf(xyzti, tf = cspace['type'], tfa0 = cspace_pars)
    
    cspace_pars = cspace.copy()
    cspace_pars.pop('type')
    if 'xyzw' in cspace_pars.keys(): 
        if cspace_pars['xyzw'] is None: 
            cspace_pars['xyzw'] = xyzrw # enter ref. whitepoint
    Jabr = colortf(xyzri, tf = cspace['type'], tfa0 = cspace_pars)    
    del cspace_pars


    # E. Regulate output:
    if out == 'Jabt,Jabr':
        return Jabt, Jabr
    elif out == 'Jabt,Jabr,cct,duv':
        return Jabt,Jabr,cct,duv
    else:
        eval(out)


#------------------------------------------------------------------------------
def spd_to_rg(data, cri_type = 'cierf', out = 'Rg', wl = None, sampleset = None, ref_type = None, cieobs  = None, avg = None, cspace = None, catf = None, cri_specific_pars = None, rg_pars = {'nhbins' : None, 'start_hue' : 0, 'normalize_gamut' : True}):
    """
    Calculates the color gamut index of data (= np.array([[wl,spds]]) (data_axis = 0) for a sample set illuminated with test source (data) with respect to some reference illuminant.
    For use in color rendition calculation with free choice of :
        * out = output requested (e.g. 'Rg')
        * wl: wavelengths (or [start, end, spacing]) to interpolate the SPD's in data argument to. Default = None (no interpolation) 
        * cri_type: str input specifying dict with default settings or user defined dict with parameters specifying color rendering index specifics (see e.g. luxpy.cri._cri_defaults['cierf'])
                    non-None input arguments to function will override defaults in cri_type dict
        * cri_type keys / further function arguments:
            - sampleset: np.array([[wl,rfl]]) or str for built-in rfl-set
            - ref_type: reference illuminant type ('BB' : Blackbody radiatiors, 'DL': daylightphase, 'ciera': used in CIE CRI-13.3-1995, 'cierf': used in CIE 224-2017, 'iesrf': used in TM30-15, ...)
            - cieobs: dict: 
                + 'xyz': cie observer for calculating xyz of samples and white 
                + 'cct': cie observer for calculating cct

            - cspace: 
                + 'type': color space used to calculate color differences
                + 'xyzw': white point of color space, (None: use xyzw of test / reference (after chromatic adaptation, if specified))
                + ' ...' : other possible parameters needed for color space calculation
            - catf: None: don't apply a cat (other than perhaps the one built into the colorspace), 
                   OR dict:
                       - 'D': degree of adaptation
                       - 'mcat': sensor matrix specification,
                       - 'xyzw': (None: use xyzw of reference otherwise transform both test and ref to xyzw)
            - avg: averaging function (handle) for color differences, DEi (e.g. numpy.mean, .math.rms, .math.geomean)
            - scale
                + 'fcn': function handle to type of cri scale, 
                    e.g. 
                    linear()_scale --> (100 - scale_factor*DEi), 
                    log_scale --> (cfr. Ohno's CQS), 
                    psy_scale (Smet et al.'s cri2012,See: LRT 2013)
                + 'cfactor': factors used in scaling function, 
                          if None: 
                              will be optimized to minimize the rms between the Rf's of the requested metric and some target metric specified in:
                                  + opt_cri_type:  str (one of the preset _cri_defaults) or dict (dict must contain all keys as normal)
                                        default = 'ciera' (if 'opt_cri_type' -key not in 'scale' dict)
                                  + opt_spd_set: set of light source spds used to optimize cfactor 
                                        default = 'F1-F12' (if 'opt_spd_set' -key not in 'scale' dict)
    
            - cri_specific_pars: other parameters specific to type of cri, e.g. maxC for CQS calculations
            - rg_pars: dict containing:
                + 'nhbins' (int): number of hue bins to divide the gamut in
                + 'start_hue' (float,°): hue at which to start slicing
                + 'normalize_gamut' (bool): normalize gamut or not before calculating a gamut area index Rg. 
    """
    #Override input parameters with data specified in cri_type:
    args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_rg')

    #avg, catf, cieobs, cieobs_cct, cri_specific_pars, cspace, cspace_pars, ref_type, rg_pars, sampleset, scale_factor, scale_fcn = [cri_type[x] for x in sorted(cri_type.keys())] 

       
    # calculate Jabt of test and Jabr of the reference illuminant corresponding to test: 
    #jabti, jabri,cct,duv = spd_to_jab_t_r(data, cri_type = cri_type, out = 'Jabt,Jabr,cct,duv', wl = wl, sampleset = sampleset, cieobs  = cieobs, cieobs_cct = cieobs_cct, cspace = cspace, catf = catf, ref_type = ref_type, cspace_pars = cspace_pars, cri_specific_pars = cri_specific_pars)
    jabti, jabri,cct,duv = spd_to_jab_t_r(data, cri_type = cri_type, out = 'Jabt,Jabr,cct,duv', wl = wl) 

    
    # calculate gamut area index:
    rg_pars = cri_type['rg_pars']
    #rg_pars = put_args_in_db(cri_type['rg_pars'],rg_pars)#{'nhbins':nhbins,'start_hue':start_hue,'normalize_gamut':normalize_gamut}) #override with not-None input from function
    nhbins, start_hue, normalize_gamut = [rg_pars[x] for x in sorted(rg_pars.keys())]
    Rg = np2d(jab_to_rg(jabti,jabri, ordered_and_sliced = False, nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut))
   
    if out == 'Rg':
        return Rg
    else:
        return eval(out)




#------------------------------------------------------------------------------
def spd_to_DEi(data, cri_type = 'cierf', out = 'DEi', wl = None, sampleset = None, ref_type = None, cieobs = None, avg = None, cspace = None, catf = None, cri_specific_pars = None):
    """
    Calculates color difference (~fidelity) of data (= np.array([[wl,spds]]) (data_axis = 0) between sample set illuminated with test source (data) and some reference illuminant.
        * out = output requested (e.g. 'DEa', 'DEi' or 'DEa,DEii', or 'DEa, DEi, cct', ...; default = 'DEi'
        * wl: wavelengths (or [start, end, spacing]) to interpolate the SPD's in data argument to. Default = None (no interpolation) 
        * cri_type: str input specifying dict with default settings or user defined dict with parameters specifying color rendering index specifics (see e.g. luxpy.cri._cri_defaults['cierf'])
                    non-None input arguments to function will override defaults in cri_type dict
        * cri_type keys / further function arguments:
            - sampleset: np.array([[wl,rfl]]) or str for built-in rfl-set
            - ref_type: reference illuminant type ('BB' : Blackbody radiatiors, 'DL': daylightphase, 'ciera': used in CIE CRI-13.3-1995, 'cierf': used in CIE 224-2017, 'iesrf': used in TM30-15, ...)
            - cieobs: dict: 
                + 'xyz': cie observer for calculating xyz of samples and white 
                + 'cct': cie observer for calculating cct

            - cspace: 
                + 'type': color space used to calculate color differences
                + 'xyzw': white point of color space, (None: use xyzw of test / reference (after chromatic adaptation, if specified))
                + ' ...' : other possible parameters needed for color space calculation
            - catf: None: don't apply a cat (other than perhaps the one built into the colorspace), 
                   OR dict:
                       - 'D': degree of adaptation
                       - 'mcat': sensor matrix specification,
                       - 'xyzw': (None: use xyzw of reference otherwise transform both test and ref to xyzw)
            - avg: averaging function (handle) for color differences, DEi (e.g. numpy.mean, .math.rms, .math.geomean)
            - cri_specific_pars: other parameters specific to type of cri, e.g. maxC for CQS calculations
    """
    #Override input parameters with data specified in cri_type:
    args = locals().copy() # get dict with keyword input arguments to function (used to overwrite non-None input arguments present in cri_type dict)
    cri_type = process_cri_type_input(cri_type, args, callerfunction = 'cri.spd_to_DEi')

    # calculate Jabt of test and Jabr of the reference illuminant corresponding to test: 
    #Jabt, Jabr, cct, duv = spd_to_jab_t_r(data, cri_type = cri_type, out = 'Jabt,Jabr,cct,duv', wl = wl, sampleset = sampleset, cieobs  = cieobs, cieobs_cct = cieobs_cct, cspace = cspace, catf = catf, ref_type = ref_type, cspace_pars = cspace_pars,cri_specific_pars = cri_specific_pars)
    Jabt, Jabr, cct, duv = spd_to_jab_t_r(data, cri_type = cri_type, out = 'Jabt,Jabr,cct,duv', wl = wl)
      
    # E. calculate DEi, DEa:
    avg = cri_type['avg']
    DEi = np.power((Jabt - Jabr),2).sum(axis = len(Jabt.shape)-1,keepdims = False)**0.5
    #DEi = np.power((Jabt - Jabr),2).sum(axis = len(Jabt.shape)-1,keepdims = True)**0.5
    DEa = avg(DEi, axis = 0) #len(Jabt.shape)-2)
    DEa = np2d(DEa)
  
     # output:
    if (out != 'DEi'):
        return  eval(out)
    else:
        return DEi

      
#------------------------------------------------------------------------------
def optimize_scale_factor(cri_type,opt_scale_factor, scale_fcn, avg) :
    """
    Optimize scale_factor of cri-model in cri_type such that average Rf for a set of light sources is the same as that of a target-cri (default: 'ciera').
    """
    if  np.any(opt_scale_factor):
        if 'opt_cri_type' not in cri_type['scale'].keys(): 
            opt_cri_type = _cri_defaults['ciera'] # use CIE Ra-13.3-1995 as target
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
def spd_to_cri(data, cri_type = 'cierf', out = 'Rf', wl = None, sampleset = None, ref_type = None, cieobs = None, avg = None, scale = None, opt_scale_factor = False, cspace = None, catf = None, cri_specific_pars = None, rg_pars = None):
    """
    Calculates color rendition (~fidelity) index of data (= np.array([[wl,spds]]) (data_axis = 0) free choice of :
        * out = output requested (e.g. 'Rf', 'Rfi' or 'Rf,Rfi', or 'Rf, Rfi, cct', ...; default = 'Rf', with 'Rf' general color fidelity index, Rfi individual color fidelity indices
        * wl: wavelengths (or [start, end, spacing]) to interpolate the SPD's in data argument to. Default = None (no interpolation) 
        * cri_type: str input specifying dict with default settings or user defined dict with parameters specifying color rendering index specifics (see e.g. luxpy.cri._cri_defaults['cierf'])
                    non-None input arguments to function will override defaults in cri_type dict
        * cri_type keys / further function arguments:
            - sampleset: np.array([[wl,rfl]]) or str for built-in rfl-set
            - ref_type: reference illuminant type ('BB' : Blackbody radiatiors, 'DL': daylightphase, 'ciera': used in CIE CRI-13.3-1995, 'cierf': used in CIE 224-2017, 'iesrf': used in TM30-15, ...)
            - cieobs: dict: 
                + 'xyz': cie observer for calculating xyz of samples and white 
                + 'cct': cie observer for calculating cct

            - cspace: 
                + 'type': color space used to calculate color differences
                + 'xyzw': white point of color space, (None: use xyzw of test / reference (after chromatic adaptation, if specified))
                + ' ...' : other possible parameters needed for color space calculation
            - catf: None: don't apply a cat (other than perhaps the one built into the colorspace), 
                   OR dict:
                       - 'D': degree of adaptation
                       - 'mcat': sensor matrix specification,
                       - 'xyzw': (None: use xyzw of reference otherwise transform both test and ref to xyzw)
            - avg: averaging function (handle) for color differences, DEi (e.g. numpy.mean, .math.rms, .math.geomean)
            - scale
                + 'fcn': function handle to type of cri scale, 
                    e.g. 
                    linear()_scale --> (100 - scale_factor*DEi), 
                    log_scale --> (cfr. Ohno's CQS), 
                    psy_scale (Smet et al.'s cri2012,See: LRT 2013)
                + 'cfactor': factors used in scaling function, 
                          if True: 
                              will be optimized to minimize the rms between the Rf's of the requested metric and some target metric specified in:
                                  + opt_cri_type:  str (one of the preset _cri_defaults) or dict (dict must contain all keys as normal)
                                        default = 'ciera' (if 'opt_cri_type' -key not in 'scale' dict)
                                  + opt_spd_set: set of light source spds used to optimize cfactor 
                                        default = 'F1-F12' (if 'opt_spd_set' -key not in 'scale' dict)
            - opt_scale_factor: True: optimize c-factor, else do nothing and use value of cfactor in 'scale'.    
            - cri_specific_pars: other parameters specific to type of cri, e.g. maxC for CQS calculations
            - rg_pars: dict containing:
                + 'nhbins' (int): number of hue bins to divide the gamut in
                + 'start_hue' (float,°): hue at which to start slicing
                + 'normalize_gamut' (bool): normalize gamut or not before calculating a gamut area index Rg. 
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
    DEi, Jabt, Jabr, cct, duv = spd_to_DEi(data, out = 'DEi,Jabt,Jabr,cct,duv', cri_type = cri_type)
    if 'Rg' in out.split(','):
        ##DEi, Jabt, Jabr, cct, duv = spd_to_DEi(data, out = 'DEi,Jabt,Jabr,cct, duv', cri_type = cri_type, sampleset = sampleset, cieobs  = cieobs, cieobs_cct = cieobs_cct, cspace = cspace, catf = catf, ref_type = ref_type, avg = avg, cspace_pars = cspace_pars, cri_specific_pars = cri_specific_pars)
        #DEi, Jabt, Jabr, cct, duv = spd_to_DEi(data, out = 'DEi,Jabt,Jabr,cct,duv', cri_type = cri_type)

        # calculate gamut area index:
        rg_pars = cri_type['rg_pars']    
        nhbins, start_hue, normalize_gamut = [rg_pars[x] for x in sorted(rg_pars.keys())]
        Rg = jab_to_rg(Jabt,Jabr, ordered_and_sliced = False, nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut)

    #else:
        ##DEi, cct,duv = spd_to_DEi(data, out = 'DEi,cct,duv', cri_type = cri_type, sampleset = sampleset, cieobs  = cieobs, cieobs_cct = cieobs_cct, cspace = cspace, catf = catf, ref_type = ref_type, avg = avg, cspace_pars = cspace_pars,cri_specific_pars = cri_specific_pars)
        #DEi, Jabt,Jabr, cct, duv = spd_to_DEi(data, out = 'DEi,Jabt,Jabr,cct,duv', cri_type = cri_type)
        
    # B. convert DE to color rendering index:

    Rfi = scale_fcn(DEi,scale_factor)
    Rf = np2d(scale_fcn(avg(DEi,axis = 0),scale_factor))
      
 
    if (out == 'Rf'):
        return Rf
    elif (out == 'Rg'):
        return Rg
    else:
        return eval(out)

    
#------------------------------------------------------------------------------
def spd_to_ciera(data, out = 'Rf', wl = None):
    """
    Wrapper function the 'ciera' color rendition (fidelity) metric. 
    """
    return spd_to_cri(data, cri_type = 'ciera', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cierf(data, out = 'Rf', wl = None):
    """
    Wrapper function the 'cierf' color rendition (fidelity) metric. 
    """
    return spd_to_cri(data, cri_type = 'cierf', out = out, wl = wl)


#------------------------------------------------------------------------------
def spd_to_iesrf(data, out = 'Rf', wl = None):
    """
    Wrapper function the 'iesrf' color rendition (fidelity) metric. 
    """
    return spd_to_cri(data, cri_type = 'iesrf', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cri2012(data, out = 'Rf', wl = None):
    """
    Wrapper function the 'cri2012' color rendition (fidelity) metric
    with the spectally uniform HL17 mathematical sampleset.
    """
    return spd_to_cri(data, cri_type = 'cri2012', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cri2012_hl17(data, out = 'Rf', wl = None):
    """
    Wrapper function the 'cri2012' color rendition (fidelity) metric
    with the spectally uniform HL17 mathematical sampleset.
    """
    return spd_to_cri(data, cri_type = 'cri2012-hl17', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cri2012_hl1000(data, out = 'Rf', wl = None):
    """
    Wrapper function the 'cri2012' color rendition (fidelity) metric
    with the spectally uniform Hybrid HL1000 sampleset.
    """
    return spd_to_cri(data, cri_type = 'cri2012-hl1000', out = out, wl = wl)

#------------------------------------------------------------------------------
def spd_to_cri2012_real210(data, out = 'Rf', wl = None):
    """
    Wrapper function the 'cri2012' color rendition (fidelity) metric 
    with the Real-210 sampleset (normally for special color rendering indices).
    """
    return spd_to_cri(data, cri_type = 'cri2012-real210', out = out, wl = wl)


###############################################################################
# MCRI: Memory Color Rendition Index, Rm: (See Smet et al. 2012, Energy & Buildings, 49 (2012) 216–225)
def spd_to_mcri(data, D = 0.9, E = None, Yb = 20.0, out = 'Rm', wl = None):
    """
    MCRI: Memory Color Rendition Index, Rm: (See Smet et al. 2012, Energy & Buildings, 49 (2012) 216–225)
    Input: 
        * data: spectral power distribution(s)
        * D: degree of adaptation (default = 0.9)
        * E: Illuminance in lux (used to calculate La = (Yb/100)*(E/pi) to calculate D following the 'cat02' model). 
             If E is None: the degree is determined by input parameter D
             If (E is not None) & (Yb is None): then E is assumed to contain the adapting field luminance La.
        * wl: wavelengths (or [start, end, spacing]) to interpolate the SPD's in data argument to. Default = None (no interpolation) 
    Output:
        * out: determines output. default = Rm (general mcri index), more / other output can be specified as e.g. 'Rm, Rmi' or 'Rmi'.  
    """
    data = np2d(data)
    
    if wl is not None: 
        data = spd(data = data, interpolation = _S_INTERP_TYPE, kind = 'np', wl = wl)
    
    
    # unpack metric default values:
    cri_type = 'mcri'
    avg, catf, cieobs, cri_specific_pars, cspace, ref_type, rg_pars, sampleset, scale = [_cri_defaults[cri_type][x] for x in sorted(_cri_defaults[cri_type].keys())] 
    similarity_ai = cri_specific_pars['similarity_ai']
    Mxyz2lms = cspace['Mxyz2lms'] 
    scale_fcn = scale['fcn']
    scale_factor = scale['cfactor']
    sampleset = eval(sampleset)
    
    
    # A. calculate xyz:
    xyzti, xyztw = spd_to_xyz(data, cieobs = cieobs['xyz'],  rfl = sampleset, out = 2)

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
    a1 = a1[:,None] #broadcast_shape(a1, target_shape = None,expand_2d_to_3d = 0)
    a2 = a2[:,None] #broadcast_shape(a2, target_shape = None,expand_2d_to_3d = 0)
    #ipt = ipt[:,None] #broadcast_shape(ipt, target_shape = None,expand_2d_to_3d = 0)
    I = I[:,None] #broadcast_shape(I, target_shape = None,expand_2d_to_3d = 0)
    a12 = np.concatenate((a1,a2),axis=2) #broadcast_shape(np.hstack((a1,a2)), target_shape = ipt.shape,expand_2d_to_3d = 0)
    #a12 = broadcast_shape(a12, target_shape = ipt.shape,expand_2d_to_3d = 0)
    ipt_mc = np.concatenate((I,a12),axis=2)
    #ipt_mc = broadcast_shape(ipt_mc, target_shape = None,expand_2d_to_3d = 0)
    ipt_test = ipt #broadcast_shape(ipt, target_shape = None,expand_2d_to_3d = 0)
    nhbins, start_hue, normalize_gamut = [rg_pars[x] for x in sorted(rg_pars.keys())]
    Rg = jab_to_rg(ipt_test,ipt_mc, ordered_and_sliced = False, nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut)


    if (out != 'Rm'):
        return  eval(out)
    else:
        return Rm
    
#-----------------------------------------------------------------------------
def  spd_to_cqs(data, version = 'v9.0', out = 'Qa',wl = None):
    """
    Calculates CQS Qa (Qai) or Qf (Qfi) or Qp (Qpi) for versions v9.0 (default) or v7.5.
    """  
    outlist = out.split()    
    if isinstance(version,str):
        cri_type = 'cqs-' + version
    elif isinstance(version, dict):
        cri_type = version
     
    # calculate DEI, labti, labri and get cspace_pars and rg_pars:
    DEi, labti, labri, cct, duv, cri_type = spd_to_DEi(data, cri_type = cri_type, out = 'DEi,Jabt,Jabr,cct,duv,cri_type', wl = wl)
    
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
