# -*- coding: utf-8 -*-
"""
###############################################################################
# Module for Individual Observer CMFs (Asano, 2016)
###############################################################################

#------------------------------------------------------------------------------
Created on Thu Apr 19 13:29:15 2018

@author: kevin.smet
"""
from luxpy import np, pd, interpolate, _PKG_PATH, _SEP, xyzbar, getdata, getwlr

from luxpy import plt

_INDVCMF_DATA_PATH = _PKG_PATH + _SEP + 'data' + _SEP + 'indvcmfs' + _SEP

_INDVCMF_DATA = {}
_INDVCMF_DATA['rmd'] = getdata(_INDVCMF_DATA_PATH  + 'cie2006_RelativeMacularDensity.dat', header = None).T 
_INDVCMF_DATA['LMSa'] = getdata(_INDVCMF_DATA_PATH  + 'cie2006_Alms.dat', header = None).T 
_INDVCMF_DATA['docul'] = getdata(_INDVCMF_DATA_PATH  + 'cie2006_docul.dat', header = None).T 
_INDVCMF_DATA['USCensus2010population'] = getdata(_INDVCMF_DATA_PATH  + 'USCensus2010Population.dat', header = 'infer',verbosity = 0).T 

_INDVCMF_STD_DEV_ALL_PARAM = {}
_INDVCMF_STD_DEV_ALL_PARAM['od_lens'] = 19.1
_INDVCMF_STD_DEV_ALL_PARAM['od_macula'] = 37.2
_INDVCMF_STD_DEV_ALL_PARAM['od_L'] = 17.9
_INDVCMF_STD_DEV_ALL_PARAM['od_M'] = 17.9
_INDVCMF_STD_DEV_ALL_PARAM['od_S'] = 14.7
_INDVCMF_STD_DEV_ALL_PARAM['shft_L'] = 4.0
_INDVCMF_STD_DEV_ALL_PARAM['shft_M'] = 3.0
_INDVCMF_STD_DEV_ALL_PARAM['shft_S'] = 2.5


wl = getwlr([390,780,5]) # wavelength range of specrtal data in _INDVCMF_DATA

def cie2006cmfsEx(age = 32,fieldsize = 10,\
                  var_od_lens = 0, var_od_macula = 0, \
                  var_od_L = 0, var_od_M = 0, var_od_S = 0,\
                  var_shft_L = 0, var_shft_M = 0, var_shft_S = 0):

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
    LMSa_shft[0] = interpolate.interp1d(wl_shifted[0],LMSa[0], kind = 'cubic', bounds_error = False)(wl)
    LMSa_shft[1] = interpolate.interp1d(wl_shifted[1],LMSa[1], kind = 'cubic', bounds_error = False)(wl)
    LMSa_shft[2] = interpolate.interp1d(wl_shifted[2],LMSa[2], kind = 'cubic', bounds_error = False)(wl)
    LMSa_shft[2,np.where(wl >= 620)] = 0
    LMSa[2,np.where(wl >= 620)] = np.nan # Note defined above 620nm

    
    # corrected LMS (no age correction):
    pkOd_L = (0.38 + 0.54*np.exp(-fs/1.333)) * (1 + var_od_L/100) # varied peak optical density of L-cone
    pkOd_M = (0.38 + 0.54*np.exp(-fs/1.333)) * (1 + var_od_M/100) # varied peak optical density of M-cone
    pkOd_S = (0.30 + 0.45*np.exp(-fs/1.333)) * (1 + var_od_S/100) # varied peak optical density of S-cone
    
    alpha_lms = 0. * LMSa_shft
    alpha_lms[0] = 1 - 10**(-pkOd_L*(10**LMSa_shft[0]))
    alpha_lms[1] = 1 - 10**(-pkOd_M*(10**LMSa_shft[1]))
    alpha_lms[2] = 1 - 10**(-pkOd_S*(10**LMSa_shft[2]))
    
    
    # this fix is required because the above math fails for alpha_lms(3,:)==0
    alpha_lms[2,np.where(wl >= 620)] = 0 
    
    # Corrected to Corneal Incidence:
    lms_barq = alpha_lms * (10**(-corrected_rmd - correct_lomd))*np.ones(alpha_lms.shape)
    
    # Corrected to Energy Terms:
    lms_bar = lms_barq * wl
    
    # normalized:
    LMS = 100 * lms_bar / lms_bar.sum(axis = 1, keepdims = True)
    
    # Output extra:
    trans_lens = 10**(-correct_lomd) 
    trans_macula = 10**(-corrected_rmd) 
    sens_photopig = alpha_lms * wl #repmat(wl,1,3)

    return LMS, trans_lens, trans_macula, sens_photopig, LMSa

def fnc_MonteCarloParam(n_population = 1, stdDevAllParam = _INDVCMF_STD_DEV_ALL_PARAM.copy()):
    """
    Get dict with normally-distributed physiological factors for a population of observers.
    """

    varParam = {}
    for k in stdDevAllParam:
        varParam[k] = stdDevAllParam[k] * np.random.randn(n_population)
  
        # limit varAllParam so that it doesn't create negative val for 
        # lens, macula, pkod_LMS:
        if (k == 'od_lens') | (k == 'od_macla') | (k == 'od_L') | (k == 'od_M') | (k == 'od_S'):
            varParam[k][np.where(varParam[k] < -100)] = -100
        
        return varParam  
  
def histogram(a, bins=10, bin_center = False, range=None, normed=False, weights=None, density=None):
    """
    Histogram function that can take as bins either the center (cfr. matlab hist)
    or bin-edges.
    
    Args: 
        :bin_center: False, optional
            False: if :bins: int, str or sequence of scalars:
                    default to numpy.histogram (uses bin edges).
            True: if :bins: is a sequence of scalars:
                    bins (containing centers) are transformed to edges
                    and nump.histogram is run. 
                    Mimicks matlab hist (uses bin centers).
        
    Note for other armuments and output, see ?numpy.histogram
    """
    if (isinstance(bins, int) |  isinstance(bins, str)) | (bin_center == False):
        return np.histogram(a, bins=bins, range=range, normed=normed, weights=weights, density=density)
    else (bin_center == True) & ((isinstance(bins, list) | (isinstance(bins, ndarray)) :
        if len(bins) == 1:
            bins = np.hstack
        centers = bins
        d = np.diff(centers)/2
        edges = np.hstack((centers[0]-d[0], centers[:-1] + d, centers[-1] + d[-1]))
        edges[1:] = edges[1:] + np.finfo(float).eps
        print('ok')
        print(edges)
        return np.histogram(a, bins=edges, range=range, normed=normed, weights=weights, density=density)


def fnc_genMonteCarloObs(n_population, fieldsize = 10, list_Age = [32]):
    """
    Monte-Carlo generation of individual observer color matching functions.
    
    Args: 
        :list_Age: list of observer ages, optional
            Defaults to 32 (cfr. CIE2006 CMFs)
        :fieldsize: fieldsize in degrees, optional
            Defaults to 10°.
    
    Returns:
        :returns: LMS_All, var_age, vAll 
            - LMS_All: numpy.ndarray with population LMS functions.
            - var_age: numpy.ndarray with population observer ages.
            - vAll: dict with population physiological factors (see .keys()) 
    """

    # Scale down StdDev by scalars optimized using Asano's 75 observers 
    # collected in Germany:
    scale_factors = [0.98, 0.98, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    stdDevAllParam = _INDVCMF_STD_DEV_ALL_PARAM.copy()
    stdDevAllParam = [stdDevAllParam[v]*scale_factors[i] for (i,v) in enumerate(stdDevAllParam.keys())]
    
    # Get Normally-distributed Physiological Factors:
    vAll = fnc_MonteCarloParam(n_population = n_population) 
    
    # Generate Random Ages with the same probability density distribution 
    # as color matching experiment:
    
    sz_interval = 1 
    list_AgeRound = np.round(np.array(list_Age)/sz_interval ) * sz_interval
    p = histogram(list_AgeRound, bins = np.unique(list_AgeRound), bin_center = True)
    print(p)
    var_age = np.random.choice(np.unique(list_AgeRound), \
                               size = n_population, replace = True, \
                               p = histogram(list_AgeRound, bins = np.unique(list_AgeRound), bin_center = True))
    
    LMS_All = nan*np.ones((n_population, wl.shape[0], 3))
    for k in range(n_population):
        t_LMS, trans_lens, trans_macula, sens_photopig, _ = cie2006cmfsEx(age = var_age[k], fieldsize = fieldsize,\
                                                                          var_od_lens = vAll[k,0], var_od_macula = vAll[k,1], \
                                                                          var_od_L = vAll[k,2], var_od_M = vAll[k,3], var_od_S = vAll[k,4],\
                                                                          var_shft_L = vAll[k,5], var_shft_M = vAll[k,6], var_shft_S = vAll[k,7])                                    
        LMS_All[k,:,:] = t_LMS
        # trans_lens_All[k,:] = trans_lens
        # trans_macula_All[k,:] = trans_macula 
        # sens_photopig_All[k,:,:] = sens_photopig 
    
    return LMS_All, var_age, vAll 





if __name__ == '__main__':
    LMS, trans_lens, trans_macula, sens_photopig, LMSa = cie2006cmfsEx()
    
    plt.figure()
    plt.plot(wl[:,None],LMSa[0].T, color ='r', linestyle='-')
    plt.plot(wl[:,None],LMS[0].T, color ='r', linestyle='--')
    plt.plot(wl[:,None],LMSa[1].T, color ='g', linestyle='-')
    plt.plot(wl[:,None],LMS[1].T, color ='g', linestyle='--')
    plt.plot(wl[:,None],LMSa[2].T, color ='b', linestyle='-')
    plt.plot(wl[:,None],LMS[2].T, color ='b', linestyle='--')
    plt.show()

    LMS_All, var_age, vAll = fnc_genMonteCarloObs(n_population = 2, fieldsize = 10, list_Age = [32])
    plt.figure()
    plt.plot(wl[:,None],LMS_All[...,0].T, color ='r', linestyle='-')
    plt.plot(wl[:,None],LMS_All[...,1].T, color ='g', linestyle='-')
    plt.plot(wl[:,None],LMS_All[...,2].T, color ='b', linestyle='-')
    plt.show()