# -*- coding: utf-8 -*-
"""
###############################################################################
# Module for Individual Observer lms-CMFs (Asano, 2016)
###############################################################################
 Port of Matlab code from:
     https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php
     (Accessed April 20, 2018)
#------------------------------------------------------------------------------

# cie2006cmfsEx(): Generate Individual Observer CMFs (cone fundamentals) 
                    based on CIE2006 cone fundamentals and published literature 
                    on observer variability in color matching and 
                    in physiological parameters.

# getMonteCarloParam(): Get dict with normally-distributed physiological factors 
                            for a population of observers.
                            
# getUSCensusAgeDist(): Get US Census Age Distribution

# genMonteCarloObs(): Monte-Carlo generation of individual observer 
                            color matching functions (cone fundamentals) for a
                            certain age and field size.

# getCatObs(): Generate cone fundamentals for categorical observers.

# get_lms_to_xyz_matrix(): Calculate lms to xyz conversion matrix for specific fieldsize.
                            
# lmsb_to_xyzb(): Convert from LMS cone fundamentals to XYZ color matching functions.

# add_to_cmf_dict(): Add set of cmfs to _CMF dict.

#------------------------------------------------------------------------------

    References:
        1. Asano Y, Fairchild MD, and Blondé L (2016). 
            Individual Colorimetric Observer Model. 
            PLoS One 11, 1–19.
        2. Asano Y, Fairchild MD, Blondé L, and Morvan P (2016). 
            Color matching experiment for highlighting interobserver variability. 
            Color Res. Appl. 41, 530–539.
        3. CIE, and CIE (2006). Fundamental Chromaticity Diagram with Physiological Axes - Part I 
            (Vienna: CIE).
            
#------------------------------------------------------------------------------
Created on Thu Apr 19 13:29:15 2018

@author: kevin.smet
"""
from luxpy import np, pd, interpolate, math, _PKG_PATH, _SEP, _CMF, dictkv,  xyzbar, spd, getdata, getwlr

from luxpy import plt

__all__ = ['_INDVCMF_DATA_PATH','_INDVCMF_DATA','_INDVCMF_STD_DEV_ALL_PARAM','_INDVCMF_CATOBSPFCTR', '_INDVCMF_M_2d', '_INDVCMF_M_10d']
__all__ +=['cie2006cmfsEx','getMonteCarloParam','genMonteCarloObs','getCatObs']


_INDVCMF_DATA_PATH = _PKG_PATH + _SEP + 'data' + _SEP + 'indvcmfs' + _SEP

# Load data from files:
_INDVCMF_DATA = {}
_INDVCMF_DATA['rmd'] = getdata(_INDVCMF_DATA_PATH  + 'cie2006_RelativeMacularDensity.dat', header = None).T 
_INDVCMF_DATA['LMSa'] = getdata(_INDVCMF_DATA_PATH  + 'cie2006_Alms.dat', header = None).T 
_INDVCMF_DATA['docul'] = getdata(_INDVCMF_DATA_PATH  + 'cie2006_docul.dat', header = None).T 
_INDVCMF_DATA['USCensus2010population'] = getdata(_INDVCMF_DATA_PATH  + 'USCensus2010Population.dat', header = 'infer',verbosity = 0).T 
_INDVCMF_DATA['CatObsPfctr'] = getdata(_INDVCMF_DATA_PATH  + 'CatObsPfctr.csv', header = None).T 

# Store var of. physiological parameters in dict:
_INDVCMF_STD_DEV_ALL_PARAM = {}
_INDVCMF_STD_DEV_ALL_PARAM['od_lens'] = 19.1 # from matlab code
_INDVCMF_STD_DEV_ALL_PARAM['od_macula'] = 37.2
_INDVCMF_STD_DEV_ALL_PARAM['od_L'] = 17.9
_INDVCMF_STD_DEV_ALL_PARAM['od_M'] = 17.9
_INDVCMF_STD_DEV_ALL_PARAM['od_S'] = 14.7
_INDVCMF_STD_DEV_ALL_PARAM['shft_L'] = 4.0
_INDVCMF_STD_DEV_ALL_PARAM['shft_M'] = 3.0
_INDVCMF_STD_DEV_ALL_PARAM['shft_S'] = 2.5

## from website (corrected values from Germany (GE) data):
## (corrected in genMonteCarloObs)
#_INDVCMF_STD_DEV_ALL_PARAM_GE['od_lens'] = 18.7 
#_INDVCMF_STD_DEV_ALL_PARAM_GE['od_macula'] = 36.5
#_INDVCMF_STD_DEV_ALL_PARAM_GE['od_L'] = 9.0
#_INDVCMF_STD_DEV_ALL_PARAM_GE['od_M'] = 9.0
#_INDVCMF_STD_DEV_ALL_PARAM_GE['od_S'] = 7.4
#_INDVCMF_STD_DEV_ALL_PARAM_GE['shft_L'] = 2.0
#_INDVCMF_STD_DEV_ALL_PARAM_GE['shft_M'] = 1.5
#_INDVCMF_STD_DEV_ALL_PARAM_GE['shft_S'] = 1.3

# Define dict with Iteratively Derived Cat.Obs.:
t_data = getdata(_INDVCMF_DATA_PATH  + 'CatObsPfctr.csv', header = None).T
dict_values = [t_data[:,i+1] for i in range(t_data.shape[1]-1)]
dict_keys = list(_INDVCMF_STD_DEV_ALL_PARAM.keys())
_INDVCMF_CATOBSPFCTR = dict(zip(dict_keys, dict_values))
_INDVCMF_CATOBSPFCTR['age'] = t_data[:,0] 


# Matrices for conversion from LMS cone fundamentals to XYZ CMFs:
# (https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php)
# For 2-degree, the 3x3 matrix is:

_INDVCMF_M_2d = np.array([[0.4151, -0.2424, 0.0425],
                          [0.1355, 0.0833, -0.0043],
                          [-0.0093, 0.0125, 0.2136]])

# For 10-degree, the 3x3 matrix is:
_INDVCMF_M_10d = np.array([[0.4499, -0.2630, 0.0460],
                           [0.1617, 0.0726, -0.0011],
                           [-0.0036, 0.0054, 0.2291]])



_WL_CRIT = 620 # Asano: 620 nm: wavelenght at which interpolation fails for S-cones
_WL = getwlr([390,780,5]) # wavelength range of specrtal data in _INDVCMF_DATA

def cie2006cmfsEx(age = 32,fieldsize = 10, wl = None,\
                  var_od_lens = 0, var_od_macula = 0, \
                  var_od_L = 0, var_od_M = 0, var_od_S = 0,\
                  var_shft_L = 0, var_shft_M = 0, var_shft_S = 0,\
                  out = 'LMS', allow_negative_values = False):
    """
    Generate Individual Observer CMFs (cone fundamentals) 
    based on CIE2006 cone fundamentals and published literature 
    on observer variability in color matching and in physiological parameters.
    
    Args:
        :age: 32 or float or int, optional
            Observer age
        :fieldsize: 10, optional
            Field size of stimulus in degrees (between 2° and 10°).
        :wl: None, optional
            Interpolation/extraplation of :LMS: output to specified wavelengths.
            None: output original _WL = np.array([390,780,5])
        :var_od_lens: 0, optional
            Std Dev. in peak optical density [%] of lens.
        :var_od_macula: 0, optional
            Std Dev. in peak optical density [%] of macula.
        :var_od_L: 0, optional
            Std Dev. in peak optical density [%] of L-cone.
        :var_od_M: 0, optional
            Std Dev. in peak optical density [%] of M-cone.
        :var_od_S: 0, optional
            Std Dev. in peak optical density [%] of S-cone.
        :var_shft_L: 0, optional
            Std Dev. in peak wavelength shift [nm] of L-cone. 
        :var_shft_L: 0, optional
            Std Dev. in peak wavelength shift [nm] of M-cone.  
        :var_shft_S: 0, optional
            Std Dev. in peak wavelength shift [nm] of S-cone. 
        :out: 'LMS' or , optional
            Determines output.
        :allow_negative_values: False, optional
            Cone fundamentals or color matching functions should not have negative values.
                If False: X[X<0] = 0.
            
    Returns:
        :returns: 
            - 'LMS' : numpy.ndarray with individual observer area-normalized cone fundamentals.
                Wavelength have been added.
                
            [- 'trans_lens': numpy.ndarray with lens transmission (no wavelengths added, no interpolation)
             - 'trans_macula': numpy.ndarray with macula transmission (no wavelengths added, no interpolation)
             - 'sens_photopig' : numpy.ndarray with photopigment sens. (no wavelengths added, no interpolation)]
            
    References:
        1. Asano Y, Fairchild MD, and Blondé L (2016). 
            Individual Colorimetric Observer Model. 
            PLoS One 11, 1–19.
        2. Asano Y, Fairchild MD, Blondé L, and Morvan P (2016). 
            Color matching experiment for highlighting interobserver variability. 
            Color Res. Appl. 41, 530–539.
        3. CIE, and CIE (2006). Fundamental Chromaticity Diagram with Physiological Axes - Part I 
            (Vienna: CIE).
    """
    fs = fieldsize
    rmd = _INDVCMF_DATA['rmd'].copy() 
    LMSa = _INDVCMF_DATA['LMSa'].copy() 
    docul = _INDVCMF_DATA['docul'].copy() 
    
    # field size corrected macular density:
    pkOd_Macula = 0.485*np.exp(-fs/6.132) * (1 + var_od_macula/100) # varied peak optical density of macula
    corrected_rmd = rmd*pkOd_Macula
    
    # age corrected lens/ocular media density: 
    if (age <= 60):
        correct_lomd = docul[:1] * (1 + 0.02*(age-32)) + docul[1:2]
    else:
        correct_lomd = docul[:1] * (1.56 + 0.0667*(age-60)) + docul[1:2]
    correct_lomd = correct_lomd * (1 + var_od_lens/100) # varied overall optical density of lens
    
    # Peak Wavelength Shift:
    wl_shifted = np.empty(LMSa.shape)
    wl_shifted[0] = _WL + var_shft_L 
    wl_shifted[1] = _WL + var_shft_M 
    wl_shifted[2] = _WL + var_shft_S 
    
    LMSa_shft = np.empty(LMSa.shape)
    kind = 'cubic'
    LMSa_shft[0] = interpolate.interp1d(wl_shifted[0],LMSa[0], kind = kind, bounds_error = False, fill_value = "extrapolate")(_WL)
    LMSa_shft[1] = interpolate.interp1d(wl_shifted[1],LMSa[1], kind = kind, bounds_error = False, fill_value = "extrapolate")(_WL)
    LMSa_shft[2] = interpolate.interp1d(wl_shifted[2],LMSa[2], kind = kind, bounds_error = False, fill_value = "extrapolate")(_WL)
#    LMSa[2,np.where(_WL >= _WL_CRIT)] = 0 #np.nan # Not defined above 620nm
#    LMSa_shft[2,np.where(_WL >= _WL_CRIT)] = 0
    
    ssw = np.hstack((0,np.sign(np.diff(LMSa_shft[2,:])))) #detect poor interpolation (sign switch due to instability)
    LMSa_shft[2,np.where((ssw >= 0) & (_WL > 560))] = np.nan
    
    
    # corrected LMS (no age correction):
    pkOd_L = (0.38 + 0.54*np.exp(-fs/1.333)) * (1 + var_od_L/100) # varied peak optical density of L-cone
    pkOd_M = (0.38 + 0.54*np.exp(-fs/1.333)) * (1 + var_od_M/100) # varied peak optical density of M-cone
    pkOd_S = (0.30 + 0.45*np.exp(-fs/1.333)) * (1 + var_od_S/100) # varied peak optical density of S-cone
    
    alpha_lms = 0. * LMSa_shft
    alpha_lms[0] = 1 - 10**(-pkOd_L*(10**LMSa_shft[0]))
    alpha_lms[1] = 1 - 10**(-pkOd_M*(10**LMSa_shft[1]))
    alpha_lms[2] = 1 - 10**(-pkOd_S*(10**LMSa_shft[2]))
    
    # this fix is required because the above math fails for alpha_lms[2,:]==0
    alpha_lms[2,np.where(_WL >= _WL_CRIT)] = 0 
    
    # Corrected to Corneal Incidence:
    lms_barq = alpha_lms * (10**(-corrected_rmd - correct_lomd))*np.ones(alpha_lms.shape)

    # Corrected to Energy Terms:
    lms_bar = lms_barq * _WL

    # Set NaN values to zero:
    lms_bar[np.isnan(lms_bar)] = 0
    
    # normalized:
    LMS = 100 * lms_bar / np.nansum(lms_bar, axis = 1, keepdims = True)

    
    # Output extra:
    trans_lens = 10**(-correct_lomd) 
    trans_macula = 10**(-corrected_rmd) 
    sens_photopig = alpha_lms * _WL 

    # Add wavelengths:
    LMS = np.vstack((_WL,LMS))
    
    if ('xyz' in out.lower().split(',')):
        LMS = lmsb_to_xyzb(LMS, fieldsize, out = 'xyz', allow_negative_values = allow_negative_values)
        out = out.replace('xyz','LMS').replace('XYZ','LMS')
    if ('lms' in out.lower().split(',')):
        out = out.replace('lms','LMS')
   
    # Interpolate/extrapolate:
    if wl is None:
        interpolation = None
    else:
        interpolation = 'cubic'
    LMS = spd(LMS, wl = wl, interpolation = interpolation, norm_type = 'area')
    
    if (out == 'LMS'):
        return LMS
    elif (out == 'LMS,trans_lens,trans_macula,sens_photopig'):
        return LMS,trans_lens, trans_macula, sens_photopig
    elif (out == 'LMS,trans_lens,trans_macula,sens_photopig,LMSa'):
        return LMS, trans_lens, trans_macula, sens_photopig, LMSa
    else:
        return eval(out)

def getMonteCarloParam(n_obs = 1, stdDevAllParam = _INDVCMF_STD_DEV_ALL_PARAM.copy()):
    """
    Get dict with normally-distributed physiological factors for a population of observers.
    
    Args:
        :n_obs: 1, optional
            Number of individual observers in population.
        :stdDevAllParam: _INDVCMF_STD_DEV_ALL_PARAM, optional
            Dict with parameters for:
                ['od_lens', 'od_macula', 'od_L', 'od_M', 'od_S', 'shft_L', 'shft_M', 'shft_S']
    
    Returns:
        :returns: dict with n_obs randomly drawn parameters.
    """

    varParam = {}
    for k in list(stdDevAllParam.keys()):
        varParam[k] = stdDevAllParam[k] * np.random.randn(n_obs)
  
        # limit varAllParam so that it doesn't create negative val for 
        # lens, macula, pkod_LMS:
        if (k == 'od_lens') | (k == 'od_macula') | (k == 'od_L') | (k == 'od_M') | (k == 'od_S'):
            varParam[k][np.where(varParam[k] < -100)] = -100
        
    return varParam  
  
def getUSCensusAgeDist():
    """
    Get US Census Age Distribution
    """
    t_num = _INDVCMF_DATA['USCensus2010population'] 
    
    list_AgeCensus = t_num[0] 
    freq_AgeCensus = np.round(t_num[1]/1000) # Reduce # of populations to manageable number, this doesn't change probability
    
    # Remove age < 10 and 70 < age:
    freq_AgeCensus[:10] = 0
    freq_AgeCensus[71:] = 0
      
    list_Age = [] 
    for k in range(len(list_AgeCensus)):
        list_Age = np.hstack((list_Age, np.repeat(list_AgeCensus[k],freq_AgeCensus[k]))) 

    return list_Age    

def genMonteCarloObs(n_obs = 1, fieldsize = 10, list_Age = [32], out = 'LMS', wl = None, allow_negative_values = False):
    """
    Monte-Carlo generation of individual observer cone fundamentals.
    
    Args: 
        :n_obs: 1, optional
            Number of observer CMFs to generate.
        :list_Age: list of observer ages or str, optional
            Defaults to 32 (cfr. CIE2006 CMFs)
            If 'us_census': use US population census of 2010 to generate list_Age.
        :fieldsize: fieldsize in degrees (between 2° and 10°), optional
            Defaults to 10°.
        :out: 'LMS' or str, optional
            Determines output.
        :wl: None, optional
            Interpolation/extraplation of :LMS: output to specified wavelengths.
            None: output original _WL = np.array([390,780,5])
        :allow_negative_values: False, optional
            Cone fundamentals or color matching functions should not have negative values.
                If False: X[X<0] = 0.
    
    Returns:
        :returns: LMS [,var_age, vAll] 
            - LMS: numpy.ndarray with population LMS functions.
            - var_age: numpy.ndarray with population observer ages.
            - vAll: dict with population physiological factors (see .keys()) 
            
    References:
        1. Asano Y, Fairchild MD, and Blondé L (2016). 
            Individual Colorimetric Observer Model. 
            PLoS One 11, 1–19.
        2. Asano Y, Fairchild MD, Blondé L, and Morvan P (2016). 
            Color matching experiment for highlighting interobserver variability. 
            Color Res. Appl. 41, 530–539.
        3. CIE, and CIE (2006). Fundamental Chromaticity Diagram with Physiological Axes - Part I 
            (Vienna: CIE).
    """

    # Scale down StdDev by scalars optimized using Asano's 75 observers 
    # collected in Germany:
    stdDevAllParam = _INDVCMF_STD_DEV_ALL_PARAM.copy()
    scale_factors = [0.98, 0.98, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    scale_factors = dict(zip(list(stdDevAllParam.keys()), scale_factors))
    stdDevAllParam = {k : v*scale_factors[k] for (k,v) in stdDevAllParam.items()}

    # Get Normally-distributed Physiological Factors:
    vAll = getMonteCarloParam(n_obs = n_obs) 
     
    if list_Age is 'us_census':
        list_Age = getUSCensusAgeDist()
    
    # Generate Random Ages with the same probability density distribution 
    # as color matching experiment:
    sz_interval = 1 
    list_AgeRound = np.round(np.array(list_Age)/sz_interval ) * sz_interval
    h = math.histogram(list_AgeRound, bins = np.unique(list_AgeRound), bin_center = True)[0]
    p = h/h.sum() # probability density distribution

    var_age = np.random.choice(np.unique(list_AgeRound), \
                               size = n_obs, replace = True,\
                               p = p)
    
    # Set requested wavelength range:
    if wl is not None:
        wl = getwlr(wl3 = wl)
    else:
        wl = _WL
        
    LMS_All = np.nan*np.ones((3+1, wl.shape[0],n_obs))
    for k in range(n_obs):
        t_LMS, t_trans_lens, t_trans_macula, t_sens_photopig = cie2006cmfsEx(age = var_age[k], fieldsize = fieldsize, wl = wl,\
                                                                          var_od_lens = vAll['od_lens'][k], var_od_macula = vAll['od_macula'][k], \
                                                                          var_od_L = vAll['od_L'][k], var_od_M = vAll['od_M'][k], var_od_S = vAll['od_S'][k],\
                                                                          var_shft_L = vAll['shft_L'][k], var_shft_M = vAll['shft_M'][k], var_shft_S = vAll['shft_S'][k],\
                                                                          out = 'LMS,trans_lens,trans_macula,sens_photopig')                                    
        LMS_All[:,:,k] = t_LMS
        
#        listout = out.split(',')
#        if ('trans_lens' in listout) | ('trans_macula' in listout) | ('trans_photopig' in listout):
#            trans_lens[:,k] = t_trans_lens
#            trans_macula[:,k] = t_trans_macula 
#            sens_photopig[:,:,k] = t_sens_photopig 

    if n_obs == 1:
        LMS_All = np.squeeze(LMS_All, axis = 2)
	
    if ('xyz' in out.lower().split(',')):
        LMS_All = lmsb_to_xyzb(LMS_All, fieldsize, out = 'xyz', allow_negative_values = allow_negative_values)
        out = out.replace('xyz','LMS').replace('XYZ','LMS')
    if ('lms' in out.lower().split(',')):
        out = out.replace('lms','LMS')

    if (out == 'LMS'):
        return LMS_All
    elif (out == 'LMS,var_age,vAll'):
        return LMS_All, var_age, vAll 
    else:
        return eval(out)

        
def getCatObs(n_cat = 10, fieldsize = 2, out = 'LMS', wl = None, allow_negative_values = False):
    """
    Generate cone fundamentals for categorical observers.
    
    Args: 
        :n_cat: 10, optional
            Number of observer CMFs to generate.
        :fieldsize: fieldsize in degrees (between 2° and 10°), optional
            Defaults to 10°.
        :out: 'LMS' or str, optional
            Determines output.
        :wl: None, optional
            Interpolation/extraplation of :LMS: output to specified wavelengths.
            None: output original _WL = np.array([390,780,5])
        :allow_negative_values: False, optional
            Cone fundamentals or color matching functions should not have negative values.
                If False: X[X<0] = 0.
    
    Returns:
        :returns: LMS [,var_age, vAll] 
            - LMS: numpy.ndarray with population LMS functions.
            - var_age: numpy.ndarray with population observer ages.
            - vAll: dict with population physiological factors (see .keys()) 
    Notes:
        Categorical observers are observer functions that would represent 
        color-normal populations. They are finite and discrete as opposed to 
        observer functions generated from the individual colorimetric observer 
        model. Thus, they would offer more convenient and practical approaches
        for the personalized color imaging workflow and color matching analyses.
        Categorical observers were derived in two steps. 
        At the first step, 10000 observer functions were generated from the 
        individual colorimetric observer model using Monte Carlo simulation. 
        At the second step, the cluster analysis, a modified k-medoids algorithm,
        was applied to the 10000 observers minimizing the squared Euclidean 
        distance in cone fundamentals space, and categorical observers were 
        derived iteratively. Since the proposed categorical observers are 
        defined by their physiological parameters and ages, their CMFs can be 
        derived for any target field size.

        Categorical observers were ordered by the importance; 
        the first categorical observer vas the average observer equivalent to 
        CIEPO06 with 38 year-old for a given field size, followed by the second
        most important categorical observer, the third, and so on.
        
        (see: https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php)
    """
    # Use Iteratively Derived Cat.Obs.:
    var_age = _INDVCMF_CATOBSPFCTR['age'].copy()
    vAll = _INDVCMF_CATOBSPFCTR.copy()
    vAll.pop('age')

    # Set requested wavelength range:
    if wl is not None:
        wl = getwlr(wl3 = wl)
    else:
        wl = _WL

    LMS_All = np.nan*np.ones((3+1,_WL.shape[0],n_cat)) 
    for k in range(n_cat):
        t_LMS = cie2006cmfsEx(age = var_age[k],fieldsize = fieldsize, wl = wl,\
                              var_od_lens = vAll['od_lens'][k],\
                              var_od_macula = vAll['od_macula'][k],\
                              var_od_L = vAll['od_L'][k],\
                              var_od_M = vAll['od_M'][k],\
                              var_od_S = vAll['od_S'][k],\
                              var_shft_L = vAll['shft_L'][k],\
                              var_shft_M = vAll['shft_M'][k],\
                              var_shft_S = vAll['shft_S'][k],\
                              out = 'LMS')
        
        LMS_All[:,:,k] = t_LMS 
    
    LMS_All[np.where(LMS_All < 0)] = 0
    
    if n_cat == 1:
        LMS_All = np.squeeze(LMS_All, axis = 2)
	
    if ('xyz' in out.lower().split(',')):
        LMS_All = lmsb_to_xyzb(LMS_All, fieldsize, out = 'xyz', allow_negative_values = allow_negative_values)
        out = out.replace('xyz','LMS').replace('XYZ','LMS')
    if ('lms' in out.lower().split(',')):
        out = out.replace('lms','LMS')
        
    if (out == 'LMS'):
        return LMS_All
    elif (out == 'LMS,var_age,vAll'):
        return LMS_All,var_age,vAll 
    else:
        return eval(out)

def get_lms_to_xyz_matrix(fieldsize = 10):
    """
    Get the lms to xyz conversion matrix for specific fieldsize.
    
    Args:
        :fieldsize: fieldsize in degrees (between 2° and 10°), optional
            Defaults to 10°.
            
    Returns:
        :M: numpy array with conversion matrix.
    
    Note: 
        For intermediate field sizes (2° < field size < 10°) the conversion matrix
        is calculated by linear interpolation between 
        the _INDVCMF_M_2d and _INDVCMF_M_10d matrices.
    """
    a = (10-fieldsize)/(10-2)
    if a < 2:
        a = 2
    elif a > 10:
        a = 10        
    return _INDVCMF_M_2d*(1 - a) + a*_INDVCMF_M_10d

def lmsb_to_xyzb(lms, fieldsize = 10, out = 'XYZ', allow_negative_values = False):
    """
    Convert from LMS cone fundamentals to XYZ color matching functions.
    
    Args:
        :lms: numpy.ndarray with lms cone fundamentals, optional
        :fieldsize: fieldsize in degrees, optional
            Defaults to 10°.
        :out: 'xyz' or str, optional
            Determines output.
        :allow_negative_values: False, optional
            XYZ color matching functions should not have negative values.
                If False: xyz[xyz<0] = 0.
    Returns:
        :returns: LMS 
            - LMS: numpy.ndarray with population XYZ color matching functions.    
    
    Note: 
        For intermediate field sizes (2° < field size < 10°) a conversion matrix
        is calculated by linear interpolation between 
        the _INDVCMF_M_2d and _INDVCMF_M_10d matrices.
    """
    wl = lms[None,0] #store wavelengths
    M = get_lms_to_xyz_matrix(fieldsize = fieldsize)
    if lms.ndim > 2:
        xyz = np.vstack((wl,math.dot23(M,lms[1:,...], keepdims = False)))
    else:
        xyz = np.vstack((wl,np.dot(M,lms[1:,...])))
    if allow_negative_values == False:
        xyz[np.where(xyz < 0)] = 0
    return xyz

def add_to_cmf_dict(bar = None, cieobs = 'indv', K = 683, M = np.eye(3)):
    """
    Add set of cmfs to _CMF dict.
    
    Args:
        :bar: None, optional
            Set of CMFs. None: initializes to empty ndarray.
        :cieobs: 'indv' or str, optional
            Name of CMF set.
        :K: 683 (lm/W), optional
            Conversion factor from radiometric to photometric quantity.
        :M: np.eye, optional
            Matrix for lms to xyz conversion.

    """
    if bar is None:
        wl3 = getwlr(_WL3)
        bar = np.vstack((wl3,np.empty((3,wl3.shape[0]))))
    _CMF['types'].append(cieobs)
    _CMF[cieobs] = {'bar' : bar}
    _CMF[cieobs]['K'] = K
    _CMF[cieobs]['M'] = M
    #return _CMF
    
    
if __name__ == '__main__':
    
    outcmf = 'lms'
    
    out = outcmf + ',trans_lens,trans_macula,sens_photopig,LMSa'
    LMS, trans_lens, trans_macula, sens_photopig, LMSa = cie2006cmfsEx(out = out)
    
    plt.figure()
    plt.plot(LMS[0],LMS[1], color ='r', linestyle='--')
    plt.plot(LMS[0],LMS[2], color ='g', linestyle='--')
    plt.plot(LMS[0],LMS[3], color ='b', linestyle='--')
    plt.title('cie2006cmfsEx(...)')
    plt.show()

    out = outcmf + ',var_age,vAll'

    LMS_All, var_age, vAll = genMonteCarloObs(n_obs = 10, fieldsize = 10, list_Age = [32], out = out)
    plt.figure()
    plt.plot(LMS_All[0],LMS_All[1], color ='r', linestyle='-')
    plt.plot(LMS_All[0],LMS_All[2], color ='g', linestyle='-')
    plt.plot(LMS_All[0],LMS_All[3], color ='b', linestyle='-')
    plt.title('genMonteCarloObs(...)')
    plt.show()
    
    LMS_All_US, var_age_US, vAll_US = genMonteCarloObs(n_obs = 10, fieldsize = 10, out = out, list_Age = 'us_census')
    plt.figure()
    plt.plot(LMS_All_US[0],LMS_All_US[1], color ='r', linestyle='-')
    plt.plot(LMS_All_US[0],LMS_All_US[2], color ='g', linestyle='-')
    plt.plot(LMS_All_US[0],LMS_All_US[3], color ='b', linestyle='-')
    plt.title("genMonteCarloObs(..., list_Age = 'use_census')")
    plt.show()
    
    LMS_All_CatObs, var_age_CatObs, vAll_CatObs  = getCatObs(n_cat = 10, fieldsize = 2, out = out)
    plt.figure()
    plt.plot(LMS_All_CatObs[0],LMS_All_CatObs[1], color ='r', linestyle='-')
    plt.plot(LMS_All_CatObs[0],LMS_All_CatObs[2], color ='g', linestyle='-')
    plt.plot(LMS_All_CatObs[0],LMS_All_CatObs[3], color ='b', linestyle='-')
    plt.title('getCatObs(...)')
    plt.show()
    
#    XYZ_All_CatObs = lmsb_to_xyzb(LMS_All_CatObs, fieldsize = 3)
#    plt.figure()
#    plt.plot(wl[:,None],XYZ_All_CatObs[0,:,:], color ='r', linestyle='-')
#    plt.plot(wl[:,None],XYZ_All_CatObs[1,:,:], color ='g', linestyle='-')
#    plt.plot(wl[:,None],XYZ_All_CatObs[2,:,:], color ='b', linestyle='-')
#    plt.title('getCatObs XYZ')
#    plt.show()
    
    # Calculate new set of CMFs and calculate xyzw and cct, duv:
    from luxpy import spd_to_xyz, _CIE_ILLUMINANTS, xyz_to_cct_ohno
    XYZb_All_CatObs, _, _  = getCatObs(n_cat = 1, fieldsize = 10, out = out)
    add_to_cmf_dict(bar = XYZb_All_CatObs, cieobs = 'CatObs1', K = 683) 
    xyz2 = spd_to_xyz(_CIE_ILLUMINANTS['F4'], cieobs = '1931_2')
    xyz1 = spd_to_xyz(_CIE_ILLUMINANTS['F4'], cieobs = 'CatObs1')
    cct2,duv2 = xyz_to_cct_ohno(xyz2, cieobs = '1931_2', out = 'cct,duv')
    cct1,duv1 = xyz_to_cct_ohno(xyz1, cieobs = 'CatObs1', out = 'cct,duv')
    print('cct,duv using 1931_2: {:1.0f} K, {:1.4f}'.format(cct2[0,0],duv2[0,0]))
    print('cct,duv using CatObs1: {:1.0f} K, {:1.4f}'.format(cct1[0,0],duv1[0,0]))