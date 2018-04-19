# -*- coding: utf-8 -*-
"""
###############################################################################
# Module for Individual Observer lms-CMFs (Asano, 2016)
###############################################################################

# cie2006cmfsEx(): Generate Individual Observer CMFs (cone fundamentals) 
                    based on CIE2006 cone fundamentals and published literature 
                    on observer variability in color matching and 
                    in physiological parameters.

# fnc_MonteCarloParam(): Get dict with normally-distributed physiological factors 
                            for a population of observers.

# fnc_genMonteCarloObs(): Monte-Carlo generation of individual observer 
                            color matching functions (cone fundamentals) for a
                            certain age and field size.

# fnc_genMonteCarloObs_USCensusAgeDist(): Monte-Carlo generation of individual 
                                            observer cone fundamentals using
                                            US Census Data for list_Age.

# fnc_getCatObs(): Generate cone fundamentals for categorical observers.


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
from luxpy import np, pd, interpolate, math, _PKG_PATH, _SEP, dictkv,  xyzbar, getdata, getwlr

from luxpy import plt

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
_INDVCMF_STD_DEV_ALL_PARAM['od_lens'] = 19.1
_INDVCMF_STD_DEV_ALL_PARAM['od_macula'] = 37.2
_INDVCMF_STD_DEV_ALL_PARAM['od_L'] = 17.9
_INDVCMF_STD_DEV_ALL_PARAM['od_M'] = 17.9
_INDVCMF_STD_DEV_ALL_PARAM['od_S'] = 14.7
_INDVCMF_STD_DEV_ALL_PARAM['shft_L'] = 4.0
_INDVCMF_STD_DEV_ALL_PARAM['shft_M'] = 3.0
_INDVCMF_STD_DEV_ALL_PARAM['shft_S'] = 2.5

# Define dict with Iteratively Derived Cat.Obs.:
t_data = getdata(_INDVCMF_DATA_PATH  + 'CatObsPfctr.csv', header = None).T
dict_values = [t_data[:,i+1] for i in range(t_data.shape[1]-1)]
dict_keys = list(_INDVCMF_STD_DEV_ALL_PARAM.keys())
_INDVCMF_CATOBSPFCTR = dict(zip(dict_keys, dict_values))
_INDVCMF_CATOBSPFCTR['age'] = t_data[:,0] 

_WL_CRIT = 620 # Asano: 620 nm: wavelenght at which interpolation fails for S-cones
wl = getwlr([390,780,5]) # wavelength range of specrtal data in _INDVCMF_DATA

def cie2006cmfsEx(age = 32,fieldsize = 10,\
                  var_od_lens = 0, var_od_macula = 0, \
                  var_od_L = 0, var_od_M = 0, var_od_S = 0,\
                  var_shft_L = 0, var_shft_M = 0, var_shft_S = 0,\
                  out = 'LMS'):
    """
    Generate Individual Observer CMFs (cone fundamentals) 
    based on CIE2006 cone fundamentals and published literature 
    on observer variability in color matching and in physiological parameters.
    
    Args:
        :age: 32 or float or int, optional
            Observer age
        :fieldsize: 10, optional
            Field size of stimulus in degrees.
        :var_od_lens: 0, optional
            Variance in peak optical density of lens.
        :var_od_macula: 0, optional
            Variance in peak optical density of macula.
        :var_od_L: 0, optional
            Variance in peak optical density of L-cone.
        :var_od_M: 0, optional
            Variance in peak optical density of M-cone.
        :var_od_S: 0, optional
            Variance in peak optical density of S-cone.
        :var_shft_L: 0, optional
            Variance in peak wavelength shift of L-cone. 
        :var_shft_L: 0, optional
            Variance in peak wavelength shift of M-cone.  
        :var_shft_S: 0, optional
            Variance in peak wavelength shift of S-cone. 
        :out: 'LMS' or , optional
            Determines output.
            
    Returns:
        :returns: 
            - 'LMS' : numpy.ndarray with individual observer area-normalized cone fundamentals.
            - 'trans_lens': numpy.ndarray with lens transmission
            - 'trans_macula': numpy.ndarray with macula transmission
            - 'sens_photopig' : numpy.ndarray with photopigment sens.
            
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
    wl_shifted[0] = wl + var_shft_L 
    wl_shifted[1] = wl + var_shft_M 
    wl_shifted[2] = wl + var_shft_S 
    
    LMSa_shft = np.empty(LMSa.shape)
    kind = 'cubic'
    LMSa_shft[0] = interpolate.interp1d(wl_shifted[0],LMSa[0], kind = kind, bounds_error = False, fill_value = "extrapolate")(wl)
    LMSa_shft[1] = interpolate.interp1d(wl_shifted[1],LMSa[1], kind = kind, bounds_error = False, fill_value = "extrapolate")(wl)
    LMSa_shft[2] = interpolate.interp1d(wl_shifted[2],LMSa[2], kind = kind, bounds_error = False, fill_value = "extrapolate")(wl)
#    LMSa[2,np.where(wl >= _WL_CRIT)] = 0 #np.nan # Not defined above 620nm
#    LMSa_shft[2,np.where(wl >= _WL_CRIT)] = 0
    
    ssw = np.hstack((0,np.sign(np.diff(LMSa_shft[2,:])))) #detect poor interpolation (sign switch due to instability)
    LMSa_shft[2,np.where((ssw >= 0) & (wl > 560))] = np.nan
    
    
    # corrected LMS (no age correction):
    pkOd_L = (0.38 + 0.54*np.exp(-fs/1.333)) * (1 + var_od_L/100) # varied peak optical density of L-cone
    pkOd_M = (0.38 + 0.54*np.exp(-fs/1.333)) * (1 + var_od_M/100) # varied peak optical density of M-cone
    pkOd_S = (0.30 + 0.45*np.exp(-fs/1.333)) * (1 + var_od_S/100) # varied peak optical density of S-cone
    
    alpha_lms = 0. * LMSa_shft
    alpha_lms[0] = 1 - 10**(-pkOd_L*(10**LMSa_shft[0]))
    alpha_lms[1] = 1 - 10**(-pkOd_M*(10**LMSa_shft[1]))
    alpha_lms[2] = 1 - 10**(-pkOd_S*(10**LMSa_shft[2]))
    
    # this fix is required because the above math fails for alpha_lms[2,:]==0
    alpha_lms[2,np.where(wl >= _WL_CRIT)] = 0 
    
    # Corrected to Corneal Incidence:
    lms_barq = alpha_lms * (10**(-corrected_rmd - correct_lomd))*np.ones(alpha_lms.shape)

    # Corrected to Energy Terms:
    lms_bar = lms_barq * wl

    # Set NaN values to zero:
    lms_bar[np.isnan(lms_bar)] = 0
    
    # normalized:
    LMS = 100 * lms_bar / np.nansum(lms_bar, axis = 1, keepdims = True)

    # Output extra:
    trans_lens = 10**(-correct_lomd) 
    trans_macula = 10**(-corrected_rmd) 
    sens_photopig = alpha_lms * wl 

    if (out == 'LMS'):
        return LMS
    elif (out == 'LMS,trans_lens,trans_macula,sens_photopig'):
        return LMS,trans_lens, trans_macula, sens_photopig
    elif (out == 'LMS,trans_lens,trans_macula,sens_photopig,LMSa'):
        return LMS, trans_lens, trans_macula, sens_photopig, LMSa
    else:
        return eval(out)

def fnc_MonteCarloParam(n_population = 1, stdDevAllParam = _INDVCMF_STD_DEV_ALL_PARAM.copy()):
    """
    Get dict with normally-distributed physiological factors for a population of observers.
    
    Args:
        :n_population: 1, optional
            Number of individual observers in population.
        :stdDevAllParam: _INDVCMF_STD_DEV_ALL_PARAM, optional
            Dict with parameters for:
                ['od_lens', 'od_macula', 'od_L', 'od_M', 'od_S', 'shft_L', 'shft_M', 'shft_S']
    
    Returns:
        :returns: dict with n_population randomly drawn parameters.
    """

    varParam = {}
    for k in list(stdDevAllParam.keys()):
        varParam[k] = stdDevAllParam[k] * np.random.randn(n_population)
  
        # limit varAllParam so that it doesn't create negative val for 
        # lens, macula, pkod_LMS:
        if (k == 'od_lens') | (k == 'od_macula') | (k == 'od_L') | (k == 'od_M') | (k == 'od_S'):
            varParam[k][np.where(varParam[k] < -100)] = -100
        
    return varParam  
  

def fnc_genMonteCarloObs(n_population = 1, fieldsize = 10, list_Age = [32], out = 'LMS'):
    """
    Monte-Carlo generation of individual observer cone fundamentals.
    
    Args: 
        :n_population: 1, optional
            Number of observer CMFs to generate.
        :list_Age: list of observer ages, optional
            Defaults to 32 (cfr. CIE2006 CMFs)
        :fieldsize: fieldsize in degrees, optional
            Defaults to 10°.
        :out: 'LMS' or str, optional
            Determines output.
    
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
    vAll = fnc_MonteCarloParam(n_population = n_population) 
      
    # Generate Random Ages with the same probability density distribution 
    # as color matching experiment:
    sz_interval = 1 
    list_AgeRound = np.round(np.array(list_Age)/sz_interval ) * sz_interval
    h = math.histogram(list_AgeRound, bins = np.unique(list_AgeRound), bin_center = True)[0]
    p = h/h.sum() # probability density distribution

    var_age = np.random.choice(np.unique(list_AgeRound), \
                               size = n_population, replace = True,\
                               p = p)

    LMS_All = np.nan*np.ones((3, wl.shape[0],n_population))
    for k in range(n_population):
        t_LMS, t_trans_lens, t_trans_macula, t_sens_photopig = cie2006cmfsEx(age = var_age[k], fieldsize = fieldsize,\
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

    if (out == 'LMS'):
        return LMS_All
    elif (out == 'LMS,var_age,vAll'):
        return LMS_All, var_age, vAll 
    else:
        return eval(out)

def fnc_genMonteCarloObs_USCensusAgeDist(n_population = 1, fieldsize = 10, out = 'LMS'):
    """
    Monte-Carlo generation of individual observer cone fundamentals using
    US Census Data for list_Age.
    
    Args: 
        :n_population: 1, optional
            Number of observer CMFs to generate.
        :fieldsize: fieldsize in degrees, optional
            Defaults to 10°.
        :out: 'LMS' or str, optional
            Determines output.
    
    Returns:
        :returns: LMS [,var_age, vAll] 
            - LMS: numpy.ndarray with population LMS functions.
            - var_age: numpy.ndarray with population observer ages.
            - vAll: dict with population physiological factors (see .keys()) 
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

    LMS_All, var_age, vAll = fnc_genMonteCarloObs(n_population = n_population, list_Age = list_Age, fieldsize = fieldsize, out = out) 
    
    if (out == 'LMS'):
        return LMS_All
    elif (out == 'LMS,var_age,vAll'):
        return LMS_All,var_age,vAll 
    else:
        return eval(out)
        
def fnc_getCatObs(n_cat = 10, fieldsize = 2, out = 'LMS'):
    """
    Generate cone fundamentals for categorical observers.
    
    Args: 
        :n_cat: 10, optional
            Number of observer CMFs to generate.
        :fieldsize: fieldsize in degrees, optional
            Defaults to 10°.
        :out: 'LMS' or str, optional
            Determines output.
    
    Returns:
        :returns: LMS [,var_age, vAll] 
            - LMS: numpy.ndarray with population LMS functions.
            - var_age: numpy.ndarray with population observer ages.
            - vAll: dict with population physiological factors (see .keys()) 
    
    """
    # Use Iteratively Derived Cat.Obs.:
    var_age = _INDVCMF_CATOBSPFCTR['age'].copy()
    vAll = _INDVCMF_CATOBSPFCTR.copy()
    vAll.pop('age')


    LMS_All = np.nan*np.ones((3,wl.shape[0],n_cat)) 
    for k in range(n_cat):
        t_LMS = cie2006cmfsEx(age = var_age[k],fieldsize = fieldsize,\
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
    
    if (out == 'LMS'):
        return LMS_All
    elif (out == 'LMS,var_age,vAll'):
        return LMS_All,var_age,vAll 
    else:
        return eval(out)


if __name__ == '__main__':
    LMS, trans_lens, trans_macula, sens_photopig, LMSa = cie2006cmfsEx(out = 'LMS,trans_lens,trans_macula,sens_photopig,LMSa')
    
    plt.figure()
    plt.plot(wl[:,None],LMS[0].T, color ='r', linestyle='--')
    plt.plot(wl[:,None],LMS[1].T, color ='g', linestyle='--')
    plt.plot(wl[:,None],LMS[2].T, color ='b', linestyle='--')
    plt.title('cie2006cmfsEx')
    plt.show()

    LMS_All, var_age, vAll = fnc_genMonteCarloObs(n_population = 10, fieldsize = 10, list_Age = [32], out='LMS,var_age,vAll')
    plt.figure()
    plt.plot(wl[:,None],LMS_All[0,:,:], color ='r', linestyle='-')
    plt.plot(wl[:,None],LMS_All[1,:,:], color ='g', linestyle='-')
    plt.plot(wl[:,None],LMS_All[2,:,:], color ='b', linestyle='-')
    plt.title('fnc_genMonteCarloObs')
    plt.show()
    
    LMS_All_US, var_age_US, vAll_US = fnc_genMonteCarloObs_USCensusAgeDist(n_population = 10, fieldsize = 10, out='LMS,var_age,vAll')
    plt.figure()
    plt.plot(wl[:,None],LMS_All_US[0,:,:], color ='r', linestyle='-')
    plt.plot(wl[:,None],LMS_All_US[1,:,:], color ='g', linestyle='-')
    plt.plot(wl[:,None],LMS_All_US[2,:,:], color ='b', linestyle='-')
    plt.title('fnc_genMonteCarloObs_USCensusAgeDist')
    plt.show()
    
    LMS_All_CatObs, var_age_CatObs, vAll_CatObs  = fnc_getCatObs(n_cat = 10, fieldsize = 2, out = 'LMS,var_age,vAll')
    plt.figure()
    plt.plot(wl[:,None],LMS_All_CatObs[0,:,:], color ='r', linestyle='-')
    plt.plot(wl[:,None],LMS_All_CatObs[1,:,:], color ='g', linestyle='-')
    plt.plot(wl[:,None],LMS_All_CatObs[2,:,:], color ='b', linestyle='-')
    plt.title('fnc_getCatObs')
    plt.show()