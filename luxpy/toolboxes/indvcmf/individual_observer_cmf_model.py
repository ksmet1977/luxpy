# -*- coding: utf-8 -*-
"""
Module for Individual Observer lms-CMFs (Asano, 2016 and CIE TC1-97)
====================================================================
    
 :_DATA_PATH: path to data files
 
 :_DATA: Dict with required data
 
 :_DSRC_STD_DEF: default data source for stdev of physiological data ('matlab', 'germany')
 
 :_DSRC_LMS_ODENS_DEF: default data source for lms absorbances and optical densities ('asano', 'cietc197')
 
 :_LMS_TO_XYZ_METHOD: default method to calculate lms to xyz conversion matrix ('asano', 'cietc197')
 
 :_WL_CRIT: critical wavelength above which interpolation of S-cone data fails.
 
 :_WL: default wavelengths of spectral data in INDVCMF_DATA.
 
 :load_database(): Load a database with parameters and data required by the Asano model.
 
 :init():   Initialize: load database required for Asano Individual Observer Model 
            into the default _DATA dict and set some options for rounding, 
            sign. figs and chopping small value to zero; for source data to use for 
            spectral data for LMS absorp. and optical densities, ... 
            
 :query_state(): print current settings for global variables.
 
 :compute_cmfs(): Generate Individual Observer CMFs (cone fundamentals) 
                  based on CIE2006 cone fundamentals and published literature 
                  on observer variability in color matching and 
                  in physiological parameters (Use of Asano optical data and model; 
                  or of CIE TC1-91 data and 'variability'-extended model possible).
 
 :cie2006cmfsEx(): Generate Individual Observer CMFs (cone fundamentals) 
                   based on CIE2006 cone fundamentals and published literature 
                   on observer variability in color matching and 
                   in physiological parameters. (Use of Asano optical data and model; 
                   or of CIE TC1-91 data and 'variability'-extended model possible)
 
 :getMonteCarloParam(): Get dict with normally-distributed physiological 
                        factors for a population of observers.
                            
 :getUSCensusAgeDist(): Get US Census Age Distribution
 
 :genMonteCarloObs(): Monte-Carlo generation of individual observer 
                      color matching functions (cone fundamentals) for a
                      certain age and field size.
 
 :getCatObs(): Generate cone fundamentals for categorical observers.
 
 :get_lms_to_xyz_matrix(): Calculate lms to xyz conversion matrix for a specific field 
                           size determined as a weighted combination of the 2° and 10° matrices.
 
 :lmsb_to_xyzb(): Convert from LMS cone fundamentals to XYZ CMFs using conversion
                  matrix determined as a weighted combination of the 2° and 10° matrices.
 
 :add_to_cmf_dict(): Add set of cmfs to _CMF dict.
 
 :plot_cmfs(): Plot cmf set.

References
----------
 1. `Asano Y, Fairchild MD, and Blondé L (2016). 
 Individual Colorimetric Observer Model. 
 PLoS One 11, 1–19. 
 <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0145671>`_
 
 2. `Asano Y, Fairchild MD, Blondé L, and Morvan P (2016). 
 Color matching experiment for highlighting interobserver variability. 
 Color Res. Appl. 41, 530–539. 
 <https://onlinelibrary.wiley.com/doi/abs/10.1002/col.21975>`_
 
 3. `CIE TC1-36 (2006). 
 Fundamental Chromaticity Diagram with Physiological Axes - Part I 
 (Vienna: CIE). 
 <https://www.cie.co.at/publications/fundamental-chromaticity-diagram-physiological-axes-part-1>`_ 
 
 4. `Asano's Individual Colorimetric Observer Model 
 <https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php>`_
 
 5. `CIE TC1-97 cmf functions python code developed by Ivar Farup and Jan Hendrik Wold.
 <https://github.com/ifarup/ciefunctions>`_
 
Notes
-----
    1. Port of Matlab code from: 
    https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php
    (Accessed April 20, 2018)  
    2. Adjusted/extended following CIE TC1-97 Python code (and data):
    github.com/ifarup/ciefunctions (Copyright (C) 2012-2017 Ivar Farup and Jan Henrik Wold)     
    (Accessed Dec 18, 2019)

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from luxpy import math, _WL3, _CMF, spd, getwlr, getwld, cie_interp, spd_to_power, xyz_to_Yxy, spd_normalize
from luxpy.utils import np, pd, sp, plt, _PKG_PATH, _SEP, getdata
import warnings

__all__ = ['_DATA','_DSRC_STD_DEF', '_DSRC_LMS_ODENS_DEF','_LMS_TO_XYZ_METHOD']
__all__ += ['load_database','init','query_state']
__all__ += ['cie2006cmfsEx','getMonteCarloParam','genMonteCarloObs','getCatObs']
__all__ += ['compute_cmfs','add_to_cmf_dict','plot_cmfs']


_DATA_PATH = _PKG_PATH + _SEP + 'toolboxes' + _SEP + 'indvcmf' + _SEP + 'data' + _SEP  
#_DATA_PATH = './data/' # for testing

global _WL, _DATA, _USE_MY_ROUND, _USE_SIGN_FIGS, _USE_CHOP, _DSRC_STD_DEF,_DSRC_LMS_ODENS_DEF, _LMS_TO_XYZ_METHOD, _DATA
_DATA = None
_DSRC_STD_DEF = 'matlab' # default data source for stdev of physiological data
_DSRC_LMS_ODENS_DEF = 'cietc197' # default data source for lms absorbances and optical densities
_LMS_TO_XYZ_METHOD = 'cietc197' # default method to calculate lms to xyz conversion matrix


_WL_CRIT = 620 # Asano: 620 nm: wavelenght at which interpolation fails for S-cones
_WL_ASANO = getwlr([390,780,5]) # wavelength range of spectral data in _DATA
_WL_CIETC197 = getwlr([390, 830, 0.1]) # wavelength range of spectral data in _DATA
if _DSRC_LMS_ODENS_DEF == 'asano':
    _WL = _WL_ASANO
elif _DSRC_LMS_ODENS_DEF == 'cietc197':
    _WL = _WL_CIETC197
    
#=============================================================================  
#  Utility functions
#=============================================================================
_USE_MY_ROUND = True
_USE_SIGN_FIGS = True
_USE_CHOP = True

def my_round(x, n=0):
    """
    Round array x to n decimal points using round half away from zero.
    This function is needed because the rounding specified in the CIE
    recommendation is different from the standard rounding scheme in python
    (which is following the IEEE recommendation).
    Args:
        :x: 
            | ndarray
            | Array to be rounded
        :n:
            | int
            | Number of decimal points
    Returns:
        :y: 
            | ndarray
            | Rounded array
    """
    if _USE_MY_ROUND:
        s = np.sign(x)
        return s*np.floor(np.absolute(x)*10**n + 0.5)/10**n
    else:
        return x

def sign_figs(x, n=0):
    """
    Round x to n significant figures (not decimal points).
    This function is needed because the rounding specified in the CIE
    recommendation is different from the standard rounding scheme in python
    (which is following the IEEE recommendation). Uses my_round (above).
    Args:
        :x: 
            | int, float or ndarray
            | Number or array to be rounded.
    Returns;
        :t:
            | float or ndarray
            | Rounded number or array.
    """
    if _USE_SIGN_FIGS:
        if type(x) == float or type(x) == int:
            if x == 0.:
                return 0
            else:
                exponent = np.ceil(np.log10(x))
                return 10**exponent * my_round(x / 10**exponent, n)
        exponent = x.copy()
        exponent[x == 0] = 0
        exponent[x != 0] = np.ceil(np.log10(np.abs(x[x != 0])))
        return 10**exponent * my_round(x / 10**exponent, n)
    else:
        return x


def chop(arr, epsilon=1e-14):
    """
    Chop values smaller than epsilon in absolute value to zero.
    Similar to Mathematica function.
    Args:
        :arr:
            | float or ndarray
            | Array or number to be chopped.
        :epsilon:
            | float
            | Minimum number.
    Returns:
        :chopped:
            | float or ndarray
            | Chopped numbers.
    """
    if _USE_CHOP:
        if isinstance(arr, float) or isinstance(arr, int):
            chopped = arr
            if np.abs(chopped) < epsilon:
                chopped = 0
            return chopped
        chopped = arr.copy()                    # initialise to arr values
        chopped[np.abs(chopped) < epsilon] = 0  # set too low values to zero
        return chopped
    else:
        return arr

#=============================================================================  
#  Function/class determining/constituting the database
#=============================================================================

def _load_asano_misc_data(path=None):
    """
    Load misc. data required by Asano model.
    
    Args:
        :path:
            | None, optional
            | Path where data files are stored (If None: look in ./data/ folder under toolbox path)
   
    Returns:
        :data:
            | dict with data for 'USCensus2010population', 'CatObsPfctr', 'M' (lms to xyz conversion matrices)
    """
    if path is None:
        path = _DATA_PATH 
        
    # Load data from files:
    data = {}
    data['USCensus2010population'] = getdata(path + 'asano_USCensus2010Population.dat', header = 'infer',verbosity = 0).T 
    #data['CatObsPfctr'] = getdata(path  + 'CatObsPfctr.dat', header = None).T 
    
    # Define dict with Iteratively Derived Cat.Obs.:
    t_data = getdata(path  + 'asano_CatObsPfctr.dat', header = None).T
    dict_values = [t_data[:,i+1] for i in range(t_data.shape[1]-1)]
    dict_keys = ['od_lens', 'od_macula', 'od_L', 'od_M', 'od_S', 'shft_L', 'shft_M', 'shft_S']
    data['CatObsPfctr'] = dict(zip(dict_keys, dict_values))
    data['CatObsPfctr']['age'] = t_data[:,0] 
    
    # Matrices for conversion from LMS cone fundamentals to XYZ CMFs:
    # (https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php)
    # For 2-degree, the 3x3 matrix is:
    data['M'] = {}
    data['M']['2d'] = np.array([[0.4151, -0.2424, 0.0425],
                          [0.1355, 0.0833, -0.0043],
                          [-0.0093, 0.0125, 0.2136]])

    # For 10-degree, the 3x3 matrix is:
    data['M']['10d']  = np.array([[0.4499, -0.2630, 0.0460],
                               [0.1617, 0.0726, -0.0011],
                               [-0.0036, 0.0054, 0.2291]])
    return data

def _load_asano_std_data(dsrc = None):
    """
    Load standard deviation data for Asano Individual Observer Model.
    
    Args:
        :dsrc:
            | None, optional
            | Data source ('matlab' code, or 'germany') for stdev data on physiological factors.
            | None defaults to string in _DSRC_DEF
    
    Returns:
        :data:
            | dict with data for 'od_lens', 'od_macula', 'od_L', 'od_M', 'od_S', 'shft_L', 'shft_M', 'shft_S'
    """
    if dsrc is None:
        dsrc = _DSRC_STD_DEF
    
    # Store var of. physiological parameters in dict:
    data = {}
    if (dsrc == 'matlab') | (dsrc == 'germany'):
        # from matlab code:
        data['od_lens'] = 19.1 
        data['od_macula'] = 37.2
        data['od_L'] = 17.9
        data['od_M'] = 17.9
        data['od_S'] = 14.7
        data['shft_L'] = 4.0
        data['shft_M'] = 3.0
        data['shft_S'] = 2.5
    else:
        raise Exception('Unknown data source (options: %s, %s)'.format('matlab', 'germany'))
        
    if (dsrc == 'germany'):
        # Scale down StdDev by scalars optimized using Asano's 75 observers 
        # collected in Germany:
        scale_factors = [0.98, 0.98, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        scale_factors = dict(zip(list(data.keys()), scale_factors))
        data = {k : v*scale_factors[k] for (k,v) in data.items()}
   
    data['dsrc'] = dsrc
    
    return data

def _load_asano_lms_and_odensities(wl=None, path=None):
    """
    Load LMS absorbance data and optical density data from Asano.
    
    Args:
        :wl:
            | [390, 780,5], optional
            | Wavelength range to interpolate data to.
        :path:
            | None, optional
            | Path where data files are stored (If None: look in ./data/ folder under toolbox path)
            
    Returns:
        :data:
            | dict with:
            | - LMS absorbances ('LMSa') 
            | - relative macular pigment density ('rmd') 
            | - ocular media optical density ('docul')
    """
    if path is None:
        path = _DATA_PATH 
    if wl is None:
        wl = _WL_ASANO
    # Load data from files:
    data = {}
    wls = getwlr([390,780,5])
    data['wls'] = wl
    data['rmd'] = getdata(path + 'asano_cie2006_RelativeMacularDensity.dat', header = None).T 
    data['docul'] = getdata(path  + 'asano_cie2006_docul.dat', header = None).T 
    data['LMSa'] = getdata(path + 'asano_cie2006_Alms.dat', header = None).T 
    for key in data.keys():
        if key != 'wls':
            data[key] = cie_interp(np.vstack((wls,data[key])), wl, kind='linear',negative_values_allowed=True)
    return data
    
def _docul_fine(ocular_sum_32, docul2):
    """
    Calculate the two parts of the expression for the optical density of the 
    ocular media as function of age.
    
    Args:
        :ocular_sum_32:
            | ndarray
            | Sum of two ocular functions
        :docul2:
            | ndarray
    Returns:
        :docul1_fine:
            | ndarray
            | The calculatedd values for part 1, docul1, tabulated  
            | with high resolution
        :docul2_fine:
            | ndarray
            | The calculatedd values for part 2, docul2, tabulated  
            | with high resolution
    """
    docul2_pad = np.zeros((75, 2))             # initialize
    docul2_pad[:, 0] = np.arange(460, 835, 5)  # fill
    docul2_pad[:, 1] = 0                       # fill
    docul2 = np.concatenate((docul2, docul2_pad))
    spl = sp.interpolate.InterpolatedUnivariateSpline(docul2[:, 0], docul2[:, 1])
    docul2_fine = ocular_sum_32.copy()
    docul2_fine[:, 1] = spl(ocular_sum_32[:, 0])
    docul1_fine = ocular_sum_32.copy()
    docul1_fine[:, 1] = ocular_sum_32[:, 1] - docul2_fine[:, 1]
    return np.hstack((docul1_fine, docul2_fine[:,1:]))

def _load_cietc197_lms_and_odensities(wl=None, path = None):
    """
    Load LMS absorbance data and optical density data from CIE TC1-97.
        
    Args:
        :wl:
            | [390, 830, 0.1], optional
            | Wavelength range to interpolate data to.
        :path:
            | None, optional
            | Path where data files are stored (If None: look in ./data/ folder under toolbox path)
            
    Returns:
        :data:
            | dict with:
            | - LMS absorbances ('LMSa') 
            | - relative macular pigment density ('rmd') 
            | - ocular media optical density ('docul')
    """
    if path is None:
        path = _DATA_PATH 
    if wl is None:
        wl = _WL_CIETC197
    # data from tc197:
    tmp = pd.read_csv(path  + 'cietc197_absorbances0_1nm.dat', header=None).values[:, [0, 2, 3, 4, 5, 6]].T
    isnan = np.isnan(tmp[3,:]) # find isnan for Sbar (missing values -> need to be set at -inf)
    first_isnan_wl = tmp[0,np.where(isnan)[0][0]] # find wavelength at which first isnan occurs for Sbar
    tmp = cie_interp(tmp,wl, kind = 'linear',negative_values_allowed = True)#, extrap_values = 'ext')
    
    absorbance = tmp[[0, 1, 2, 3],:] #LMS absorbance
    absorbance[-1,tmp[0]>first_isnan_wl] = -np.inf # set to -inf.

    macula_rel = tmp[[0, 5],:] 
    macula_rel[1,:] /= 0.35  # div by 0.35 since macula at 2° has a maximum of 0.35 at 460 (at 5nm step)
    docul2 = pd.read_csv(path  + 'cietc197_docul2.dat', header=None)
    ocular_sum_32 = tmp[[0, 4],:].T  # 32 years only!
    docul2 = _docul_fine(ocular_sum_32, docul2)
    docul2 = cie_interp(docul2.T,wl, kind = 'linear',negative_values_allowed = True, extrap_values = 'ext')
    data = {'wls': wl, 'rmd': macula_rel, 'docul':docul2, 'LMSa': absorbance}
    return data

def load_database(wl = None, dsrc_std = None, dsrc_lms_odens = None, path = None):
    """
    Load database required for Asano Individual Observer Model.
    
    Args:
        :wl:
            | None, optional
            | Wavelength range to interpolate data to.
            | None defaults to the wavelength range associated with data in :dsrc_lms_odens:
        :path:
            | None, optional
            | Path where data files are stored (If None: look in ./data/ folder under toolbox path)
        :dsrc_std:
            | None, optional
            | Data source ('matlab' code, or 'germany') for stdev data on physiological factors.
            | None defaults to string in _DSRC_STD_DEF
        :dsrc_lms_odens:
            | None, optional
            | Data source ('asano', 'cietc197') for LMS absorbance and optical density data.
            | None defaults to string in _DSRC_LMS_ODENS_DEF
    Returns:
        :data:
            | dict with data for:
            | - 'LMSa': LMS absorbances 
            | - 'rmd': relative macular pigment density 
            | - 'docul': ocular media optical density 
            | - 'USCensus2010population': data (age and numbers) on a 2010 US Census 
            | - 'CatObsPfctr': dict with iteratively derived Categorical Observer physiological stdevs.
            | - 'M2d': Asano 2° lms to xyz conversion matrix 
            | - 'M10d': Asano 10° lms to xyz conversion matrix 
            | - standard deviations on physiological parameters: 'od_lens', 'od_macula', 'od_L', 'od_M', 'od_S', 'shft_L', 'shft_M', 'shft_S'
    """
    data = _load_asano_misc_data(path=path)
    data['stdev'] = _load_asano_std_data(dsrc=dsrc_std)
    if dsrc_lms_odens is None:
        dsrc_lms_odens = _DSRC_LMS_ODENS_DEF
    if dsrc_lms_odens == 'asano':
        data['odata'] = _load_asano_lms_and_odensities(wl=wl, path=path)
    elif dsrc_lms_odens == 'cietc197':
        data['odata'] = _load_cietc197_lms_and_odensities(wl=wl, path=path)
    else:
        data['odata'] = {}
        raise Exception("Unknown source string for optical data (options: %s, %s)".format('asano', 'cietc197'))
    data['odata']['dsrc'] = dsrc_lms_odens
    return data

#Actually load default database
_DATA = load_database()

def init(wl = None, dsrc_std = None, dsrc_lms_odens = None, lms_to_xyz_method = None,
         use_sign_figs = True, use_my_round = True,use_chop = True, 
         path = None, out = None, verbosity = 1):
    """
    Initialize: load database required for Asano Individual Observer Model 
    into the default _DATA dict and set some options for rounding, 
    sign. figs and chopping small value to zero; for source data to use for 
    spectral data for LMS absorp. and optical desnities, ... 
    
    Args:
        :wl:
            | None, optional
            | Wavelength range to interpolate data to.
            | None defaults to the wavelength range associated with data in :dsrc_lms_odens:
        :dsrc_std:
            | None, optional
            | Data source ('matlab' code, or 'germany') for stdev data on physiological factors.
            | None defaults to string in _DSRC_STD_DEF
        :dsrc_lms_odens:
            | None, optional
            | Data source ('asano', 'cietc197') for LMS absorbance and optical density data.
            | None defaults to string in _DSRC_LMS_ODENS_DEF
        :lms_to_xyz_method:
            | None, optional
            | Method to use to determine lms-to-xyz conversion matrix (options: 'asano', 'cietc197')
        :use_my_round:
            | True, optional
            | If True: use my_rounding() conform CIE TC1-91 Python code 'ciefunctions'. (slows down code)
            | by setting _USE_MY_ROUND.
        :use_sign_figs:
            | True, optional
            | If True: use sign_figs() conform CIE TC1-91 Python code 'ciefunctions'. (slows down code)
            | by setting _USE_SIGN_FIGS.
        :use_chop:
            | True, optional
            | If True: use chop() conform CIE TC1-91 Python code 'ciefunctions'. (slows down code)
            | by setting _USE_CHOP.
        :path:
            | None, optional
            | Path where data files are stored (If None: look in ./data/ folder under toolbox path)
        :out:
            | None, optional
            | If None: only set global variables, do not output _DATA.copy()
        :verbosity:
            | 1, optional
            | Print new state of global settings.
            
    Returns:
        :data:
            | if out is not None: return a dict with dict with data for:
            | - 'LMSa': LMS absorbances 
            | - 'rmd': relative macular pigment density 
            | - 'docul': ocular media optical density 
            | - 'USCensus2010population': data (age and numbers) on a 2010 US Census 
            | - 'CatObsPfctr': dict with iteratively derived Categorical Observer physiological stdevs.
            | - 'M2d': Asano 2° lms to xyz conversion matrix 
            | - 'M10d': Asano 10° lms to xyz conversion matrix 
            | - standard deviations on physiological parameters: 'od_lens', 'od_macula', 'od_L', 'od_M', 'od_S', 'shft_L', 'shft_M', 'shft_S'
    """
    global _WL, _DATA, _USE_MY_ROUND, _USE_SIGN_FIGS, _USE_CHOP, _DSRC_STD_DEF,_DSRC_LMS_ODENS_DEF, _LMS_TO_XYZ_METHOD
    
    _USE_MY_ROUND = use_my_round
    _USE_SIGN_FIGS = use_sign_figs
    _USE_CHOP = use_chop
    
    if dsrc_std is not None:
        _DSRC_STD_DEF = dsrc_std
    else:
        _DSRC_STD_DEF = 'matlab'
        
    if dsrc_lms_odens is not None:
        _DSRC_LMS_ODENS_DEF = dsrc_lms_odens
    else:
        _DSRC_LMS_ODENS_DEF = 'cietc197'
    
    if lms_to_xyz_method is not None:
        _LMS_TO_XYZ_METHOD = lms_to_xyz_method
    else:
        _LMS_TO_XYZ_METHOD = 'cietc197'
    
    if _DSRC_LMS_ODENS_DEF == 'asano':
        _WL = _WL_ASANO
    elif _DSRC_LMS_ODENS_DEF == 'cietc197':
        _WL = _WL_CIETC197
    
    _DATA = load_database(wl = wl, dsrc_std = dsrc_std, dsrc_lms_odens = dsrc_lms_odens, path = path)
    
    if verbosity > 0:
        query_state()
    if out is not None:
        return _DATA.copy()
    else:
        return out
    
def query_state():
    """
    Print current settings for 'global variables'.
    """
    
    print("\nCurrent state of 'global variables/settings': \n")
    print("\t_USE_MY_ROUND = ", _USE_MY_ROUND)
    print("\t_USE_SIGN_FIGS = ", _USE_SIGN_FIGS)
    print("\t_USE_CHOP = ", _USE_CHOP)
    print("\n\t_WL = [{:1.2f}, {:1.2f}, {:1.2f}, ..., {:1.2f}, {:1.2f}, {:1.2f}]".format(_WL[0],_WL[1],_WL[2],_WL[-3],_WL[-2],_WL[-1]))
    print("\t_DSRC_STD_DEF = ", _DSRC_STD_DEF)
    print("\t_DSRC_LMS_ODENS_DEF = ", _DSRC_LMS_ODENS_DEF)
    print("\t_LMS_TO_XYZ_METHOD = ", _LMS_TO_XYZ_METHOD)
    print("\n\tDatabase dsrc_std = ", _DATA['stdev']['dsrc'])
    print("\tDatabase dsrc_lms_odens = ", _DATA['odata']['dsrc'])
    print("\n")

    if not np.array_equal(_WL, _DATA['odata']['wls']):
        warnings.warn("Note that ['odata']['wls']) does not match database setting.")
    if _DATA['stdev']['dsrc'] != _DSRC_STD_DEF: 
        warnings.warn('Note that _STD_DEF does not match database setting.')
    if _DATA['odata']['dsrc'] != _DSRC_LMS_ODENS_DEF: 
        warnings.warn('Note that _LMS_ODENS_DEF does not match database setting')
    

#==============================================================================
#  Functions of age and/or field size
#==============================================================================

def _d_ocular(age = 32, var_od_lens = 0, docul0 = None):
    """
    Calculate the optical density of the ocular media for a given age.

    Args:
        :age:
            | 32, float, optional
            | Age in years.
        :var_od_lens:
            | 0, optional
            | Variation of optical density of lens.
        :docul0: 
            | None, optional
            | Uncorrected ocular media density function 
            | None defaults to the one stored in _DATA

    Returns:
        :docul:
            | ndarray with the calculated optical density of the ocular media; row 0 are wavelenghts.
    """
    if docul0 is None:
        docul = _DATA['odata']['docul'].copy() 
    else:
        docul = docul0
    
    if var_od_lens < -100:
        var_od_lens = -100
        
    # age corrected lens/ocular media density: 
    if (age <= 60):
        corrected_lomd = docul[1:2,:] * (1 + 0.02*(age-32)) + docul[2:3,:]
    else:
        corrected_lomd = docul[1:2,:] * (1.56 + 0.0667*(age-60)) + docul[2:3,:]
    corrected_lomd = corrected_lomd * (1 + var_od_lens/100) # varied overall optical density of lens
    return np.array((docul[:1,:],corrected_lomd)) 

def _d_mac_max(fieldsize = 10, var_od = 0):
    """
    Calculate the maximum optical density of the macular pigment for a given field size.
    
    Args:
        :fieldsize:
            | 10, float, optional
            | Field size in degrees.
        :var_od:
            | 0, optional
            | Variation of optical density of macula.

    Returns:
        :d_mac_max:
            | float
            | The calculated maximum optical density of the macular pigment.
    """
    if var_od < -100:
        var_od = -100
    return my_round((0.485*np.exp(-fieldsize/6.132)) * (1 + var_od/100), 3)

def _d_mac(fieldsize = 10, var_od = 0, rmd0 = None):
    """
    Calculate the optical density of the macular pigment for a given field size.
    
    Args:
        :fieldsize:
            | 10, float, optional
            | Field size in degrees.
        :var_od:
            | 0, optional
            | Variation of optical density of macula.
        :rmd0: 
            | None, optional
            | Uncorrected relative density of macula function
            | None defaults to the one stored in _DATA
            
    Returns:
        :rmd:
            | ndarray with the calculated relative optical density of the macular pigment; row 0 are wavelenghts.
    """
    if rmd0 is None:
        rmd = _DATA['odata']['rmd'].copy()
    else:
        rmd = rmd0.copy()
    rmd[1:,:] = rmd[1:,:] * _d_mac_max(fieldsize = fieldsize, var_od = var_od)
    return rmd

def _d_LM_max(fieldsize = 10, var_od = 0):
    """
    Calculate the maximum optical density of the L- and M-cone photopigments for a given field size.
    
    Args:
        :field_size:
            | 10, float, optional
            | Field size in degrees.
        :var_od:
            | 0, optional
            | Variation of optical density.
    
    Returns:
        :d_LM_max:
            | float
            | The calculated maximum optical density of the L- and M-cone photopigments
    """
    if var_od < -100:
        var_od = -100
    return my_round((0.38 + 0.54*np.exp(-fieldsize/1.333)) * (1 + var_od/100), 3)


def _d_S_max(fieldsize = 10, var_od = 0):
    """
    Calculate the maximum optical density of the S-cone photopigment for a given field size.
    
    Args:
        :fieldsize:
            | 10, float, optional
            | Field size in degrees.
        :var_od:
            | 0, optional
            | Variation of optical density.

    Returns:
        :d_S_max:
            | float
            | The calculated maximum optical density of the S-cone visual pigment
    """
    if var_od < -100:
        var_od = -100
    return my_round((0.30 + 0.45*np.exp(-fieldsize/1.333)) * (1 + var_od/100), 3)

def _LMS_absorptance(fieldsize = 10, var_shft_LMS = [0,0,0], var_od_LMS = [0, 0, 0], LMSa0 = None):
    """
    Calculate the quantal absorptance of the L, M and S cones for a given field size.
    
    Args:
        :fieldsize: 
            | 10, float, optional
            | Field size in degrees.
        :var_shft_LMS:
            | [0, 0, 0] optional
            | Variation (shift) of LMS peak absorptance.
        :var_od_LMS:
            | [0, 0, 0] optional
            | Variation of LMS optical densities.
        :LMSa0: 
            | None, optional
            | Uncorrected LMS absorptance functions
            | None defaults to the ones stored in _DATA

    Returns:
        alpha_lms: 
            | ndarray with the calculated quantal absorptances of the L, M and S cones; row 0 are wavelenghts.
    """
    
    if LMSa0 is None:
        LMSa = _DATA['odata']['LMSa'].copy()
    else:
        LMSa = LMSa0.copy()
  
    wls = LMSa[:1,:].copy() # wavelengths
    LMSa = LMSa[1:,:] # get rid of wavelengths

    # Peak Wavelength Shift:
    wl_shifted = np.empty(LMSa.shape)
    wl_shifted[0] = wls + var_shft_LMS[0] 
    wl_shifted[1] = wls + var_shft_LMS[1] 
    wl_shifted[2] = wls + var_shft_LMS[2] 
    
    LMSa_shft = np.empty(LMSa.shape)
    kind = 3 # ->'cubic'
    if var_shft_LMS[0] == 0:
        LMSa_shft[0] = LMSa[0]
    else:
        LMSa_shft[0] = sp.interpolate.InterpolatedUnivariateSpline(wl_shifted[0],LMSa[0], k = kind, ext = "extrapolate")(wls)
    if var_shft_LMS[1] == 0:
        LMSa_shft[1] = LMSa[1]
    else:
        LMSa_shft[1] = sp.interpolate.InterpolatedUnivariateSpline(wl_shifted[1],LMSa[1], k = kind, ext = "extrapolate")(wls)
    
    if var_shft_LMS[2] == 0:
        LMSa_shft[2] = LMSa[2]
    else:
        LMSa[2,np.isinf(LMSa[2,:])] = np.nan
        non_nan_indices = np.logical_not(np.isnan(LMSa[2]))
        LMSa_shft[2] = sp.interpolate.InterpolatedUnivariateSpline(wl_shifted[2][non_nan_indices],LMSa[2][non_nan_indices], k = kind, ext = "extrapolate")(wls)

        # Detect poor interpolation (sign switch due to instability):
        ssw = np.hstack((0,np.sign(np.diff(LMSa_shft[2,:])))) 
        cond = ((ssw >= 0) & (wls > 560))
        if cond.any():
            wl_min = wls[np.where(cond)].min()
            LMSa_shft[2,np.where((wls >= wl_min))] = np.nan

    # corrected LMS (no age correction):
    pkOd_L = _d_LM_max(fieldsize, var_od_LMS[0])  # varied peak optical density of L-cone
    pkOd_M = _d_LM_max(fieldsize, var_od_LMS[1])  # varied peak optical density of M-cone
    pkOd_S = _d_S_max(fieldsize, var_od_LMS[2])   # varied peak optical density of S-cone
    
    alpha_lms = 1. * LMSa_shft
    alpha_lms[0] = 1 - 10**(-pkOd_L*(10**LMSa_shft[0]))
    alpha_lms[1] = 1 - 10**(-pkOd_M*(10**LMSa_shft[1]))
    alpha_lms[2] = 1 - 10**(-pkOd_S*(10**LMSa_shft[2]))
#    alpha_lms[np.isnan(alpha_lms)] = 0
    # this fix is required because the above math fails for alpha_lms[2,:]==0
    #alpha_lms[2,np.where(wls >= _WL_CRIT)] = 0 

    return np.vstack((wls,alpha_lms))

def _LMS_quantal(fieldsize = 10, age = 32, var_od_lens = 0, var_od_mac = 0, 
                 var_shft_LMS = [0,0,0], var_od_LMS = [0, 0, 0], 
                 norm_type = 'max', out = 'LMSq', odata0 = None):
    """
    Calculate the quantal based LMS cone fundamentals for a given field size and age.
    
    Args:
        :fieldsize:
            | 10, float, optional
            | Field size in degrees.
        :age:
            | 32, float, optional
            | Age in years.
        :var_od_lens:
            | 0, optional
            | Variation of optical density of lens.
        :var_od_mac:
            | 0, optional
            | Variation of optical density of macula.
        :var_shft_LMS:
            | [0, 0, 0] optional
            | Variation (shift) of LMS peak absorptance.
        :var_od_LMS:
            | [0, 0, 0] optional
            | Variation of LMS optical densities.
        :norm_type:
            | 'max', optional
            | - 'max': normalize LMSq functions to max = 1
            | - 'area': normalize to area
            | - 'power': normalize to power
        :odata0: 
            | None, optional
            | Dict with uncorrected ocular media and macula density functions and LMS absorptance functions
            | None defaults to the ones stored in _DATA

    Returns:
        :LMSq: 
            | ndarray with the calculated quantum_based LMS cone fundamentals; first row are wavelengths.
    """
    if odata0 is None:
        odata = _DATA['odata']
    else:
        odata = odata0
        
    # field size corrected macular density:
    rmd = _d_mac(fieldsize = fieldsize, var_od = var_od_mac, rmd0 = odata['rmd']) 
    
    # age corrected lens/ocular media density: 
    docul = _d_ocular(age = age, var_od_lens = var_od_lens, docul0 = odata['docul'])
    
    # corrected LMS (no age correction):
    alpha_lms = _LMS_absorptance(fieldsize = fieldsize, var_shft_LMS = var_shft_LMS, var_od_LMS = var_od_LMS, LMSa0 = odata['LMSa'])
    
    # Corrected to Corneal Incidence:
    LMSq = alpha_lms.copy()
    LMSq[1:,:] = alpha_lms[1:,:] * (10**(-rmd[1:,:] - docul[1:,:]))*np.ones(alpha_lms[1:,:].shape)
    
    
    if norm_type == 'max':
        LMSq[1:,:] = LMSq[1:,:] / np.nanmax(LMSq[1:,:],axis = 1, keepdims = True)

    if out == 'LMSq':
        return LMSq
    elif out == 'LMSq,rmd,docul,alpha_lms':
        return LMSq,rmd,docul,alpha_lms
    else:
        return eval(out)

def _LMS_energy(fieldsize = 10, age = 32, var_od_lens = 0, var_od_mac = 0, 
                var_shft_LMS = [0,0,0], var_od_LMS = [0, 0, 0], 
                norm_type = 'max', out = 'LMSe', base = False, odata0 = None):
    """
    Calculate the energy based LMS cone fundamentals for a given field size and age.
    
    Args:
        :fieldsize:
            | 10, float, optional
            | Field size in degrees.
        :age:
            | 32, float, optional
            | Age in years.
        :var_od_lens:
            | 0, optional
            | Variation of optical density of lens.
        :var_od_mac:
            | 0, optional
            | Variation of optical density of macula.
        :var_shft_LMS:
            | [0, 0, 0] optional
            | Variation (shift) of LMS peak absorptance.
        :var_od_LMS:
            | [0, 0, 0] optional
            | Variation of LMS optical densities.
        :norm_type:
            | 'max', optional
            | - 'max': normalize LMSq functions to max = 1
            | - 'area': normalize to area
            | - 'power': normalize to power
        :base:
            | False, boolean, optional
            | The returned energy-based LMS cone fundamentals given to the
            | precision of 9 sign. figs. if 'True', and to the precision of
            | 6 sign. figs. if 'False'.
        :odata0: 
            | None, optional
            | Dict with uncorrected ocular media and macula density functions and LMS absorptance functions
            | None defaults to the ones stored in _DATA

    Returns:
        :LMSe: 
            | ndarray with the calculated quantum_based LMS cone fundamentals; first row are wavelengths.
    """
    LMSq, rmd, docul, alpha_lms  = _LMS_quantal(fieldsize = fieldsize, age = age,
                                                var_od_lens = var_od_lens, var_od_mac = var_od_mac,
                                                var_shft_LMS = var_shft_LMS, var_od_LMS = var_od_LMS, 
                                                norm_type = 'max', out = 'LMSq,rmd,docul,alpha_lms',
                                                odata0 = odata0)
    wls = LMSq[:1,:]
    LMSe = LMSq.copy()
    LMSe[1:,:] = LMSe[1:,:]*wls
    
    # Set NaN values to zero:
    LMSe[np.isnan(LMSe)] = 0
    
    # Get max values before normalization:
    LMSe_o_max = np.nanmax(LMSe[1:,:], axis = 1, keepdims = True)
    if norm_type == 'max':
        LMSe[1:,:] = LMSe[1:,:] / LMSe_o_max
    elif norm_type == 'area':
        LMSe[1:,:] = LMSe[1:,:] / np.nansum(LMSe[1:,:],axis = 1, keepdims = True)
    elif norm_type == 'power':
        LMSe[1:,:] = LMSe[1:,:] / spd_to_power(LMSe, ptype='ru')
    
    if base:
        LMSe = sign_figs(LMSe, 9)
    else:
        LMSe = sign_figs(LMSe, 6)

    if out == 'LMSe':
        return LMSe
    elif out == 'LMSe,LMSq,alpha_lms,LMSe_o_max,rmd,docul':
        return LMSe,LMSq,alpha_lms,LMSe_o_max,rmd,docul
    elif out == 'LMSe,LMSq,alpha_lms,LMSe_o_max':
        return LMSe, LMSq, alpha_lms, LMSe_o_max
    elif out == 'LMSe,LMSe_o_max':
        return LMSe, LMSe_o_max
    elif out == 'LMSe,LMSq':
        return LMSe, LMSq
    elif out == 'LMSe,LMSq,alpha_lms':
        return LMSe, LMSq, alpha_lms
    elif out == 'LMSe,rmd,docul,alpha_lms':
        return LMSe,rmd,docul,alpha_lms
    elif out == 'LMSe,LMSq,rmd,docul,alpha_lms':
        return LMSe,LMSq,rmd,docul,alpha_lms
    else:
        return eval(out)


def _relative_L_cone_weight_Vl_quantal(fieldsize = 10, age = 32, strategy_2 = True,
                                      LMSa = None, LMSq = None, 
                                      var_od_lens = 0, var_od_mac = 0, 
                                      var_shft_LMS = [0,0,0], var_od_LMS = [0, 0, 0],
                                      out = 'kLq', odata0 = None):
    """
    Compute the weighting factor of the quantal L-cone fundamental in the
    synthesis of the cone-fundamental-based quantal V(λ) function (normalized to max=1).
    
    Args:
        :fieldsize: 
            | 10, float, optional
            | Field size in degrees.
        :age:
            | 32, float, optional
            | Age in years.
        :strategy_2: 
            | True, bool, optional
            | Use strategy 2 in github.com/ifarup/ciefunctions issue #121 for 
            | computing the weighting factor. If false, strategy 3 is applied.
        :LMSa:
            | None, optional
            | Pre-calculated LMSa (if None: will be calculated)
        :LMSq:
            | None, optional
            | Pre-calculated LMSq (if None: will be calculated)
        :var_od_lens:
            | 0, optional
            | Variation of optical density of lens.
        :var_od_mac:
            | 0, optional
            | Variation of optical density of macula.
        :var_shft_LMS:
            | [0, 0, 0] optional
            | Variation (shift) of LMS peak absorptance.
        :var_od_LMS:
            | [0, 0, 0] optional
            | Variation of LMS optical densities.
        :odata0: 
            | None, optional
            | Dict with uncorrected ocular media and macula density functions and LMS absorptance functions
            | None defaults to the ones stored in _DATA

    Returns:
        kLq: 
            | float
            | The computed weighting factor of the quantal L cone fundamental 
            | in the synthesis of the quantal V(λ) function , i.e.
            | Vq(λ) = kLq lq_bar((λ) + mq_bar(λ)).
            
    Strategies:
        1. Continue using constant 1.89 for all situations
        2. Assume 1.89 for all field sizes, 32 years, and scale for other ages (most probably)
        3. Assume 1.89 for 32 years, 2 degrees, and scale from there
        4. Find general solution for l_bar_max / m_bar_max as function of age and field size
        
    """
    if strategy_2:
        field_size = 2.
    else:
        field_size = fieldsize

    if odata0 is None:
        odata = _DATA['odata']
    else:
        odata = odata0
  
    # avoid recalculation if unnecessary:  
    if (field_size != fieldsize) | (LMSa is None) | (LMSq is None):

        _, LMSq_fs_age, LMSa_fs, _ = _LMS_energy(fieldsize = field_size, age = age,
                                                      var_od_lens = var_od_lens, var_od_mac = var_od_mac,
                                                      var_shft_LMS = var_shft_LMS, var_od_LMS = var_od_LMS, 
                                                      norm_type = 'max', out = 'LMSe,LMSq,alpha_lms,LMSe_o_max',
                                                      base = True, odata0 = odata) # note: base only applies to LMSe !
    else:
        LMSa_fs = LMSa
        LMSq_fs_age = LMSq

    LMSa_2 = _LMS_absorptance(fieldsize = 2.0, LMSa0 = odata['LMSa'])
    LMSq_2_32 = _LMS_quantal(fieldsize = 2.0, age = 32, odata0 = odata)
    
    const_fs_age = (LMSa_fs[1, 0] * LMSq_fs_age[2, 0] / (LMSa_fs[2, 0] * LMSq_fs_age[1, 0]))
    const_2_32 = (LMSa_2[1, 0] * LMSq_2_32[2, 0] / (LMSa_2[2, 0] * LMSq_2_32[1, 0]))
    kLq_rel = 1.89 * const_fs_age / const_2_32
    
    if out == 'kLq':
        return kLq_rel
    else:
        return eval(out)


def _Vl_energy_and_LM_weights(fieldsize = 10, age = 32, strategy_2 = True,
                             LMSa = None, LMSq = None, LMSe = None, LMSe_o_max = None,
                             var_od_lens = 0, var_od_mac = 0, 
                             var_shft_LMS = [0,0,0], var_od_LMS = [0, 0, 0],
                             odata0 = None):
    """
    Compute the energy-based V(λ) function (starting from energy-based LMS).
    Return both V(λ) and the the corresponding L and M cone weights used
    in the synthesis.
    
    Args:
        :fieldsize: 
            | 10, float, optional
            | Field size in degrees.
        :age:
            | 32, float, optional
            | Age in years.
        :strategy_2: 
            | True, bool, optional
            | Use strategy 2 in github.com/ifarup/ciefunctions issue #121 for 
            | computing the weighting factor. If false, strategy 3 is applied.
        :LMSa:
            | None, optional
            | Pre-calculated LMSa (if None: will be calculated)
        :LMSq:
            | None, optional
            | Pre-calculated LMSq (if None: will be calculated)
        :LMSe:
            | None, optional
            | Pre-calculated LMSe (if None: will be calculated)
        :LMSe_o_max:
            | None, optional
            | Max of original (prior to normalization) LMSe.
            | Pre-calculated value (if None: will be calculated).
        :var_od_lens:
            | 0, optional
            | Variation of optical density of lens.
        :var_od_mac:
            | 0, optional
            | Variation of optical density of macula.
        :var_shft_LMS:
            | [0, 0, 0] optional
            | Variation (shift) of LMS peak absorptance.
        :var_od_LMS:
            | [0, 0, 0] optional
            | Variation of LMS optical densities.
        :odata0: 
            | None, optional
            | Dict with uncorrected ocular media and macula density functions and LMS absorptance functions
            | None defaults to the ones stored in _DATA

    Returns:
        :Vl:
            | ndarray
            | The energy-based V(λ) function; wavelengths in first row.
        :a21,a22:
            | float
            | The computed weighting factors of, respectively, the L and the
            | M cone fundamental in the synthesis of energy-based V(λ) function,
            | i.e. V(λ) = a21*l_bar(λ) + a22*m_bar(λ)
    """

    kLq_rel = _relative_L_cone_weight_Vl_quantal(fieldsize = fieldsize, age = age, strategy_2 = strategy_2,
                                                 LMSa = LMSa, LMSq = LMSq,
                                                 var_od_lens = var_od_lens, var_od_mac = var_od_mac, 
                                                 var_shft_LMS = var_shft_LMS, var_od_LMS = var_od_LMS,
                                                 out = 'kLq', odata0 = odata0)
    (Lo_max, Mo_max) = LMSe_o_max[:2]
    Vo_max = (kLq_rel * Lo_max * LMSe[1,:] + Mo_max * LMSe[2,:]).max()
    a21 = my_round(kLq_rel * Lo_max / Vo_max, 8)
    a22 = my_round(Mo_max / Vo_max, 8)
    V = sign_figs(a21 * LMSe[1,:] + a22 * LMSe[2,:], 7)
    Vl = np.array(([LMSe[0,:], V]))
    return Vl, (a21, a22)

def _xyz_interpolated_reference_system(fieldsize, XYZ31_std, XYZ64_std):
    """
    Compute the spectral chromaticity coordinates of the reference system
    by interpolation between correspoding spectral chromaticity coordinates
    of the CIE 1931 XYZ systems and the CIE 1964 XYZ systems.
    
    Args:
        :fieldsize:
            | float
            | The field size in degrees.
        :XYZ31_std:
            | ndarray
            | The CIE 1931 XYZ colour-matching functions (2°), given at 1 nm
            | steps from 360 nm to 830 nm; wavelengths in first row.
        :XYZ64_std:
            | ndarray
            | The CIE 1964 XYZ colour-matching functions (10°), given at 1 nm
            | steps from 360 nm to 830 nm; wavelengths in first row.

    Returns:
        :chromaticity:
            | ndarray
            | The computed interpolated spectral chromaticity coordinates of the
            | CIE standard XYZ systems; wavelenghts in first row.
    """
    # Compute the xyz spectral chromaticity coordinates of the CIE standards 
    xyz31 = np.vstack((XYZ31_std[:1,:],(xyz_to_Yxy((XYZ31_std[1:,:]).T)[:,1:]).T))
    xyz64 = np.vstack((XYZ64_std[:1,:],(xyz_to_Yxy(XYZ64_std[1:,:].T)[:,1:]).T))
    
    # Determine the wavelength parameters of the knots in the CIE 1931 and 
    # CIE 1964 xy diagrams that serve as guide-points for the interpolation 
    # (morphing) between the spectral CIE 1931 chromaticities and the spectral
    # CIE 1964 chromaticities. 
    [wl31, x31, y31] = (xyz31)[:3]  
    [wl64, x64, y64] = (xyz64)[:3] 
    wl31_knots = np.array([360, wl31[np.argmin(x31)], wl31[np.argmax(y31)], 700, 830])
    wl64_knots = np.array([360, wl64[np.argmin(x64)], wl64[np.argmax(y64)], 700, 830])
    
    # Determine the wavelength parameters of the knots (guide-points) in the
    # reference diagram (for the field size specified)
    a = (fieldsize - 2)/8.
    wl_knots = np.array([360.,
                       (1 - a) * wl31[np.argmin(x31)] + a * wl64[np.argmin(x64)],
                       (1 - a) * wl31[np.argmax(y31)] + a * wl64[np.argmax(y64)],
                       700.,
                       830.])
    # wl values
    wl31_interp = sp.interpolate.InterpolatedUnivariateSpline(wl_knots, wl31_knots, k = 1)(wl31)
    wl64_interp = sp.interpolate.InterpolatedUnivariateSpline(wl_knots, wl64_knots, k = 1)(wl64)

    # x values
    x31_interp = sp.interpolate.InterpolatedUnivariateSpline(wl31, x31, k = 3)(wl31_interp)
    x64_interp = sp.interpolate.InterpolatedUnivariateSpline(wl64, x64, k = 3)(wl64_interp)
    x_values = (1-a) * x31_interp + a * x64_interp
    
    # y values
    y31_interp = sp.interpolate.InterpolatedUnivariateSpline(wl31, y31, k = 3)(wl31_interp)
    y64_interp = sp.interpolate.InterpolatedUnivariateSpline(wl64, y64, k = 3)(wl64_interp)
    y_values = (1-a) * y31_interp + a * y64_interp
    
    # z values
    z_values = 1 - x_values - y_values
    return np.array([wl31, x_values, y_values, z_values])

# =============================================================================
# Minimisation function
# =============================================================================

    
def _square_sum(a13, a21, a22, a33, 
                xyz_ref_trunk, x_ref_min,
                L_wl_sum, M_wl_sum, S_wl_sum, V_wl_sum,
                L_wl_ref_min, M_wl_ref_min, S_wl_ref_min, V_wl_ref_min,
                LMS_390_830, wl_390_830, wl_ref_min, full_results = False):
    """
    Function to be optimised for determination of element a13 in the (non-renormalized) 
    transformation matrix of the linear transformation LMS --> XYZ.
    
    Args:
        :a13: 
            | ndarray
            | 1x1 array with parameter to optimise.
        :a21, a22, a33: 
            | float
            | Parameters in matrix for LMS to XYZ conversion.
        :L_wl_sum, M_wl_sum, S_wl_sum, V_wl_sum: 
            | Sum of L, M, S, V (i.e. L_spline(wl_main).sum())
        :L_wl_ref_min, M_wl_ref_min, S_wl_ref_min, V_wl_ref_min: 
            | value of L, M, S, V at wl_ref_min (i.e. L_spline(wl_ref_min))
        :wl_390_830:
            | wavelengths from 390 - 830 nm.
        :LMS_390_830:
            | LMS evaluated for wl_390_830 nm.
        :xyz_ref_trunk: 
            | ndarray
            | Truncated reference xyz chromaticity coordinates at 1 nm steps (wavelengths in first row).
        :x_ref_min:
            | min of x coordinate in xyz_ref_trunk
        :wl_ref_min: 
            | float
            | λ value that gives a minimum for the x-coordinate in the
            | corresponding reference diagram, i.e. x(wl_ref_min) = x_ref_min.
        :full_results: 
            | bool
            | Return all results or just the computed error.

    Returns:
        :err: 
            | float
            | Computed error.
        :trans_mat:
            | ndarray
            | Transformation matrix.
        :wl_test_min:
            | float
            | argmin(x(wl)).
        :ok:
            | bool
            | Hit the correct minimum wavelength.
    """
    
    # Transformation coefficients (a11 and a12 computed by Mathematica)
    a11 = (((a13 * (1 - x_ref_min) *
             (M_wl_ref_min * S_wl_sum -
              S_wl_ref_min * M_wl_sum)) +
            (x_ref_min *
             (a21 * L_wl_ref_min + a22 * M_wl_ref_min +
              a33 * S_wl_ref_min) * M_wl_sum) -
            ((1 - x_ref_min) * M_wl_ref_min * V_wl_sum)) /
           ((1 - x_ref_min) *
            (L_wl_ref_min * M_wl_sum -
             M_wl_ref_min * L_wl_sum)))
    a12 = (((a13 * (1 - x_ref_min) *
             (L_wl_ref_min * S_wl_sum -
              S_wl_ref_min * L_wl_sum)) +
            (x_ref_min *
             (a21 * L_wl_ref_min + a22 * M_wl_ref_min +
              a33 * S_wl_ref_min) * L_wl_sum) -
            ((1 - x_ref_min) * L_wl_ref_min * V_wl_sum)) /
           ((1 - x_ref_min) *
            (M_wl_ref_min * L_wl_sum -
             L_wl_ref_min * M_wl_sum)))
    a11 = my_round(a11[0], 8)
    a12 = my_round(a12[0], 8)
    a13 = my_round(a13[0], 8)
    trans_mat = np.array([[a11, a12, a13], [a21, a22, 0.], [0., 0., a33]], dtype=np.float)
    XYZ = sign_figs(np.dot(trans_mat, LMS_390_830), 7)
#    sumXYZ = X + Y + Z
#    xyz = np.array([X / sumXYZ, Y / sumXYZ, Z / sumXYZ])

    xyz = XYZ/XYZ.sum(axis=0,keepdims=True)
    err = ((xyz - xyz_ref_trunk)**2).sum()
    wl_test_min = wl_390_830[xyz[0, :].argmin()]
    ok = (wl_test_min == wl_ref_min)
    
    if not ok:
        err = err + np.inf
    
    if full_results:
        return (err, trans_mat, wl_test_min, ok)
    else:
        return err
    
def _compute_LMS(wls, L_spline, M_spline, S_spline, base=False):
    """
    Compute the LMS cone fundamentals for given wavelengths, as linear values 
    to respective specified precisions.
    
    Args:
        :wls:
            | ndarray
            | The wavelengths for which the LMS cone fundamentals are to be calculated.
        :L_spline, M_spline, S_spline:
            | Spline-interpolation functions for the LMS cone fundamentals (on a linear scale).
        :base:
            | boolean
            | The returned energy-based LMS values are given to the precision of
            | 9 sign. figs. / 8 decimal points if 'True', and to the precision of
            | 6 sign. figs. / 5 decimal points if 'False'.

    Returns:
        :LMS:
            | ndarray
            | The computed LMS cone fundamentals; wavelengths in first row.
    """

    if base:
        LMS_sf = 9
    else:
        LMS_sf = 6
        
    # Compute linear values
    LMS = chop(np.vstack((wls,
                         sign_figs(np.array([L_spline(wls), 
                                             M_spline(wls), 
                                             S_spline(wls)]), LMS_sf))))
    return LMS
    
def _compute_XYZ(L_spline, M_spline, S_spline, V_spline,
                LMS_spec, LMS_all, LM_weights, xyz_reference):
    """
    Compute the CIE cone-fundamental-based XYZ tristimulus functions.
    
    Args:
        :L_spline, M_spline, S_spline:
            | Spline-interpolation functions for the LMS cone fundamentals (on a linear scale).
        :V_spline:
            | Spline-interpolation functions for the cone-fundamental-based V(λ)-function (on a linear scale).
        :LMS_spec: 
            | ndarray
            | Table of LMS values at specified wavelengths, given to base-value
            | precision (i.e. 9 sign. figs); wavelengths in first column.
        :LMS_all: 
            | ndarray
            | Table of LMS values at 0.1 nm steps from 390 nm to 830 nm,
            | given to base-value precision (i.e. 9 sign. figs.); wavelengths in first row.
        :LM_weights:
            | ndarray
            | The weighting factors kL and kM in the synthesis of the
            | cone-fundamental-based V(λ)-function, i.e.
            | V(λ) = kL * l_bar(λ) + kM * m_bar(λ).
        :xyz_reference:
            | ndarray
            | The spectral chromaticity coordinates of the reference system
            | (obtained by shape-morphing (interpolation) between the CIE 1931
            | standard and the CIE 1964 standard).

    Returns:
        :trans_mat:
            | ndarray
            | The non-renormalized transformation matrix of the linear transformation LMS --> XYZ.
        :XYZ_spec:
            | ndarray
            | The non-renormalized CIE cone-fundamental-based XYZ spectral
            | tristimulus values for the tabulated wavelengths, given to
            | standard/specified precision; wavelengths in first row.
        :trans_mat_N:
            | ndarray
            | The renormalized transformation matrix of the linear transformation LMS --> XYZ.
        :XYZ_spec_N:
            | ndarray
            | The renormalized CIE cone-fundamental-based XYZ spectral
            | tristimulus values for the tabulated wavelengths, given to
            | standard/specified precision; wavelengths in first row.
    """
    # '_all'  : values given at 0.1 nm steps from 390 nm to 830 nm
    # '_main' : values given at 1 nm steps from 390 nm to 830 nm
    # '_spec' : values given at specified wavelengths

    xyz_ref = xyz_reference
    (a21, a22) = LM_weights
    
    d = 1/getwld(LMS_all[0,:])
    if np.isclose(d,d.mean()).all():
        d = d.mean()
    if isinstance(d,np.float):
        if d < 1:
            d = 1
        LMS_main = LMS_all[:,::int(d)]
    else:
        warnings.warn('Obtaining LMS_main by interpolation')
        LMS_main = cie_interp(LMS_all, np.arange(390, 831), kind = 'cubic')
    (wl_main, L_main, M_main, S_main) = LMS_main
    V_main = sign_figs(a21 * L_main + a22 * M_main, 7)
    a33 = my_round(V_main.sum() / S_main.sum(), 8)
    
    # Compute optimised non-renormalised transformation matrix
    wl_x_min_ref = 502
    wl_390_830 = np.arange(390, 831,1)
    
    ## Pre-compute some stuff that doesn't change over each iteration (for speed):
    LMS_390_830 = np.array([L_spline(wl_390_830),
                            M_spline(wl_390_830),
                            S_spline(wl_390_830)])
    L_wl_sum = L_spline(wl_main).sum()
    M_wl_sum = M_spline(wl_main).sum()
    S_wl_sum = S_spline(wl_main).sum()
    V_wl_sum = V_spline(wl_main).sum()
        
    xyz_ref_trunk = xyz_ref[1:,30:] # Stripping reference values in accordance with CIE2006 tables
    x_ref_min = xyz_ref_trunk[0, :].min()

    ok = False
    while not ok:
        L_wl_ref_min = L_spline(wl_x_min_ref)
        M_wl_ref_min = M_spline(wl_x_min_ref)
        S_wl_ref_min = S_spline(wl_x_min_ref)
        V_wl_ref_min = V_spline(wl_x_min_ref)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a13 = sp.optimize.fmin(_square_sum, 0.39, 
                                   (a21, a22, a33,
                                   xyz_ref_trunk, x_ref_min,
                                   L_wl_sum, M_wl_sum, S_wl_sum, V_wl_sum,
                                   L_wl_ref_min, M_wl_ref_min, S_wl_ref_min, V_wl_ref_min,
                                   LMS_390_830, wl_390_830, wl_x_min_ref, False),
                                   xtol = 1e-10, disp = False) 
        trans_mat, wl_x_min_ref, ok = (_square_sum(a13, a21, a22, a33,  
                                       xyz_ref_trunk, x_ref_min,
                                       L_wl_sum, M_wl_sum, S_wl_sum, V_wl_sum,
                                       L_wl_ref_min, M_wl_ref_min, S_wl_ref_min, V_wl_ref_min,
                                       LMS_390_830, wl_390_830, wl_x_min_ref, True)[1:])        

    # Compute renormalized transformation matrix
    wl_spec = LMS_spec[0,:]
    (X_exact_spec, Y_exact_spec, Z_exact_spec) = np.dot(trans_mat, LMS_spec[1:,:])
    if ((wl_spec[0] == 390. and wl_spec[-1] == 830.) and (my_round(wl_spec[1] - wl_spec[0], 1) == 1.0)):
        trans_mat_N = trans_mat
    else:
        (X_exact_sum, Y_exact_sum, Z_exact_sum) = (np.sum(X_exact_spec), np.sum(Y_exact_spec), np.sum(Z_exact_spec))
        trans_mat_N = my_round(trans_mat * ([Y_exact_sum / X_exact_sum], [1], [Y_exact_sum / Z_exact_sum]), 8) 
    
    # Compute spectral tristimulus values (for table)
    ### non-renormalized:
    XYZ_spec = np.vstack((wl_spec,sign_figs(np.dot(trans_mat, LMS_spec[1:,:]),7)))

    ### renormalized:
    XYZ_spec_N = np.vstack((wl_spec,sign_figs(np.dot(trans_mat_N, LMS_spec[1:,:]),7)))
    return (trans_mat, XYZ_spec, trans_mat_N, XYZ_spec_N)


def compute_cmfs(fieldsize = 10, age = 32, wl = None,
                 var_od_lens = 0, var_od_macula = 0, 
                 var_shft_LMS = [0,0,0], var_od_LMS = [0, 0, 0], 
                 norm_type = None, out = 'lms', base = False, 
                 strategy_2 = True, odata0 = None,
                 lms_to_xyz_method = None, allow_negative_values = False,
                 normalize_lms_to_xyz_matrix = False):
    """
    Generate Individual Observer CMFs (cone fundamentals) 
    based on CIE2006 cone fundamentals and published literature 
    on observer variability in color matching and in physiological parameters.
        
    Args:
        :age: 
            | 32 or float or int, optional
            | Observer age
        :fieldsize:
            | 10, optional
            | Field size of stimulus in degrees (between 2° and 10°).
        :wl: 
            | None, optional
            | Interpolation/extraplation of :LMS: output to specified wavelengths.
            | None: output original _WL
        :var_od_lens:
            | 0, optional
            | Variation of optical density of lens.
        :var_od_macula:
            | 0, optional
            | Variation of optical density of macula.
        :var_shft_LMS:
            | [0, 0, 0] optional
            | Variation (shift) of LMS peak absorptance.
        :var_od_LMS:
            | [0, 0, 0] optional
            | Variation of LMS optical densities.
        :norm_type:
            | None, optional
            | - 'max': normalize LMSq functions to max = 1
            | - 'area': normalize to area
            | - 'power': normalize to power
        :out: 
            | 'lms' or 'xyz', optional
            | Determines output.
        :base:
            | False, boolean, optional
            | The returned energy-based LMS cone fundamentals given to the
            | precision of 9 sign. figs. if 'True', and to the precision of
            | 6 sign. figs. if 'False'.
        :strategy_2: 
            | True, bool, optional
            | Use strategy 2 in github.com/ifarup/ciefunctions issue #121 for 
            | computing the weighting factor. If false, strategy 3 is applied.
        :odata0: 
            | None, optional
            | Dict with uncorrected ocular media and macula density functions and LMS absorptance functions
            | None defaults to the ones stored in _DATA
        :lms_to_xyz_method:
            | None, optional
            | Method to use to determine lms-to-xyz conversion matrix (options: 'asano', 'cietc197')
        :allow_negative_values:
            | False, optional
            | Cone fundamentals or color matching functions should not have negative values.
            |     If False: X[X<0] = 0.
        :normalize_lms_to_xyz_matrix:
            | False, optional
            | Normalize that EEW is always at [100,100,100] in XYZ and LMS system.
            
    Returns:
        :returns: 
            | - 'LMS' [or 'XYZ']: ndarray with individual observer equal area-normalized 
            |           cone fundamentals. Wavelength have been added.
            |    
            | [- 'M': lms to xyz conversion matrix
            |  -  'trans_lens': ndarray with lens transmission 
            |      (no interpolation)
            |  - 'trans_macula': ndarray with macula transmission 
            |      (no interpolation)
            |  - 'sens_photopig' : ndarray with photopigment sens. 
            |      (no interpolation)]
            
    References:
         1. `Asano Y, Fairchild MD, and Blondé L, (2016), 
         Individual Colorimetric Observer Model. 
         PLoS One 11, 1–19. 
         <http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0145671>`_
        
         2. `Asano Y, Fairchild MD, Blondé L, and Morvan P (2016). 
         Color matching experiment for highlighting interobserver variability. 
         Color Res. Appl. 41, 530–539. 
         <https://onlinelibrary.wiley.com/doi/abs/10.1002/col.21975>`_
         
         3. `CIE, TC1-36, (2006). 
         Fundamental Chromaticity Diagram with Physiological Axes - Part I 
         (Vienna: CIE). 
         <http://www.cie.co.at/publications/fundamental-chromaticity-diagram-physiological-axes-part-1>`_ 
         
         4. `Asano's Individual Colorimetric Observer Model 
         <https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php>`_
         
         5. `CIE TC1-97 Python code for cone fundamentals and XYZ cmf calculations 
         (by Ivar Farup and Jan Henrik Wold, (c) 2012-2017) 
         <http://github.com/ifarup/ciefunctions>`_
    """
    # TC1-97 ciefunctions rounds fieldsize:
    fieldsize_tmp = np.round(fieldsize,1)
    if (fieldsize_tmp == 2) | (fieldsize_tmp == 10):
        fieldsize = fieldsize_tmp
    
    if odata0 is None:
        odata = _DATA['odata']
    else:
        odata = odata0
    
    if wl is None:
        wl = odata['wls']
    else:
        wl = getwlr(wl3 = wl)
    
    if lms_to_xyz_method is None:
        lms_to_xyz_method = _LMS_TO_XYZ_METHOD
    
    out_list = out.split(',')
    
    # =======================================================================
    # Create initial data arrays
    # =======================================================================
    # '_base' : 9 sign. figs.
    # '_std'  : standard number of sign. figs./decimal places
    # '_all'  : values given at 0.1 nm steps from 390 nm to 830 nm
    # '_main' : values given at 1 nm steps from 390 nm to 830 nm
    # '_spec' : values given at specified wavelengths

    # wavelength arrays:
    wl_all = odata['wls']#my_round(np.arange(390., 830. + .01, .1), 1)
    wl_spec = wl

    # LMS-base values (9 sign.figs.) at 0.1 nm steps from 390 nm to 830 nm;
    # wavelengths in first row.
    LMS_base_all, LMSq_, LMSa_, LMSe_o_max, rmd, docul = _LMS_energy(fieldsize = fieldsize, age = age,
                                                                      var_od_lens = var_od_lens, var_od_mac = var_od_macula,
                                                                      var_shft_LMS = var_shft_LMS, var_od_LMS = var_od_LMS, 
                                                                      norm_type = 'max', out = 'LMSe,LMSq,alpha_lms,LMSe_o_max,rmd,docul',
                                                                      base = True, odata0 = odata) # note: base only applies to LMSe !

    # Do sompe checks to save on calculation time (don't calculate anything not needed.):
    wl_equal_to_all = np.array_equal(my_round(wl_spec,1), my_round(LMS_base_all[0,:],1))
    if (not wl_equal_to_all) | ((('xyz' in out_list) | ('XYZ' in out_list) | ('M' in out_list)) & (lms_to_xyz_method == 'cietc197')):

        # =======================================================================
        # Create LMS spline functions
        # =======================================================================
        # base:
        (wl_all, L_base_all, M_base_all, S_base_all) = LMS_base_all
        L_base_spline = sp.interpolate.InterpolatedUnivariateSpline(wl_all, L_base_all)
        M_base_spline = sp.interpolate.InterpolatedUnivariateSpline(wl_all, M_base_all)
        S_base_spline = sp.interpolate.InterpolatedUnivariateSpline(wl_all, S_base_all)
            
        # =======================================================================
        # Compute the LMS-base cone fundamentals 
        # =======================================================================
        if (not wl_equal_to_all):
            # - LMS-base values (9 sign. figs) for specified wavelengths;
            #   wavelengths in first row.
            LMS_base_spec = chop(_compute_LMS(wl_spec, L_base_spline, M_base_spline, S_base_spline, base = True))
        else:
            LMS_base_spec = chop(LMS_base_all)
        LMS = LMS_base_spec
    else:
        LMS = chop(LMS_base_all)
  
    # =========================================================================
    # Compute the cone-fundamental-based XYZ tristimulus functions
    # =========================================================================
    if ('xyz' in out_list) | ('XYZ' in out_list) | ('M' in out_list): 
        
        if lms_to_xyz_method == 'asano':
            XYZ, M = lmsb_to_xyzb(LMS, fieldsize, out = 'xyz,M', allow_negative_values = allow_negative_values)
                    
        elif lms_to_xyz_method == 'cietc197':
        
            # Vλ and weighting factors of the L and M cone fundamentals:
        
            # - Cone-fundamental-based V(λ) values (7 sign. figs.) at 0.1 nm steps
            #   from 390 nm to 830 nm; wavelengths in first column
            # - Weights of L and M cone fundamentals in V(λ) synthesis
            # Re-use already calculated LMSe, LMSq, ...
            (V_std_all, LM_weights) = _Vl_energy_and_LM_weights(fieldsize = fieldsize, age = age,
                                                                  strategy_2 = strategy_2,
                                                                  LMSa = LMSa_, LMSq = LMSq_, 
                                                                  LMSe = LMS_base_all, LMSe_o_max = LMSe_o_max,
                                                                  odata0 = odata0)

            # Create spline function for Vlambda:
            wl_all, V_std_all = V_std_all
            V_std_spline = sp.interpolate.InterpolatedUnivariateSpline(wl_all, V_std_all)
            
            #  Determine reference diagram
            xyz_reference = _xyz_interpolated_reference_system(fieldsize, _CMF['1931_2']['bar'].copy(), _CMF['1964_10']['bar'].copy())

            # - Non-renormalised tranformation matrix (8 decimal placed)
            # - Non-renormalised CIE cone-fundamental-based XYZ tristimulus
            #   values (7 sign. figs) for specified wavelengths; wavelengths in first row.
            # - Ditto renormalized
            (trans_mat_std, XYZ_std_spec, trans_mat_std_N, XYZ_std_spec_N) = _compute_XYZ(L_base_spline, 
                                                                                         M_base_spline,
                                                                                         S_base_spline, 
                                                                                         V_std_spline,
                                                                                         LMS_base_spec, 
                                                                                         LMS_base_all,
                                                                                         LM_weights, 
                                                                                         xyz_reference)
            
#            M = trans_mat_std, # unnormalized xyzbar! (when specified wavelenght range = [390,830,1] then XYZ_spec == XYZ_spec_N!!)
#            XYZ = XYZ_std_spec
            M = trans_mat_std_N # re-normalized xyzbar! (for specified wavelenght range)
            XYZ = XYZ_std_spec_N

    # Output extra:
    if 'trans_lens' in out_list:
        trans_lens = docul.copy()
        trans_lens[1:,:] = 10**(-docul[1:,:]) 
    if 'trans_macula' in out_list:
        trans_macula = rmd.copy()
        trans_macula[1:,:] = 10**(-rmd[1:,:]) 
    if 'sens_photopig' in out_list:
        sens_photopig = LMSa_.copy()
        sens_photopig[1:,:] = LMSa_[1:,:] * LMSa_[:1,:] 
    
    # Change normalization of M to 
    # ensure that EEW is always at [100,100,100] in XYZ system:
    if ('M' in out_list) & (normalize_lms_to_xyz_matrix == True):
        Mi = np.linalg.inv(M) # M: lms->xyz; Mi: xyz->lms
        Min = math.normalize_3x3_matrix(Mi, xyz0 = np.array([[1,1,1]])) # normalize Mi matrix
        M = np.linalg.inv(Min) # calculate new lms->xyz normalized matrix
        LMS[1:,:] = np.dot(Min,XYZ[1:,:]) # calculate lmsbar such that they match M!
        
        
    if (('xyz' in out.lower().split(',')) & ('lms' in out.lower().split(','))):
        # Change normalization of LMS, XYZ:
        if norm_type is not None:
            LMS = spd_normalize(LMS, norm_type = norm_type)
            XYZ = spd_normalize(XYZ, norm_type = norm_type)
 
    else:
        if ('xyz' in out.lower().split(',')):
            LMS = XYZ
            out = out.replace('xyz','LMS').replace('XYZ','LMS')
        if ('lms' in out.lower().split(',')):
            out = out.replace('lms','LMS')

        # Change normalization of LMS:
        if norm_type is not None:
            LMS = spd_normalize(LMS, norm_type = norm_type)
            
    if base == False:
        if ('lms' in out.lower().split(',')):
            LMS = sign_figs(LMS, 6) # only LMS. XYZ is output at 7 sign. digit level in tc197 Python code

    if (out == 'LMS') | (out == 'lms'):
        return LMS
    elif (out == 'XYZ') | (out == 'xyz'):
        return XYZ
    elif (out == 'LMS,M') | (out == 'lms,M'):
        return LMS, M
    elif (out == 'XYZ,M') | (out == 'xyz,M'):
        return XYZ, M
    elif (out == 'LMS,XYZ,M') | (out == 'lms,xyz,M'):
        return LMS, XYZ, M
    elif out == 'M':
        return M
    elif (out == 'LMS,trans_lens,trans_macula,sens_photopig'):
        return LMS,trans_lens, trans_macula, sens_photopig
    elif (out == 'LMS,trans_lens,trans_macula,sens_photopig,LMSa'):
        return LMS, trans_lens, trans_macula, sens_photopig, odata['LMSa'].copy()
    else:
        return eval(out)

def cie2006cmfsEx(age = 32,fieldsize = 10, wl = None,\
                  var_od_lens = 0, var_od_macula = 0, \
                  var_od_L = 0, var_od_M = 0, var_od_S = 0,\
                  var_shft_L = 0, var_shft_M = 0, var_shft_S = 0,\
                  norm_type = None, out = 'lms', base = False, 
                  strategy_2 = True, odata0 = None,
                  lms_to_xyz_method = None, allow_negative_values = False,
                  normalize_lms_to_xyz_matrix = False):
    """
    Generate Individual Observer CMFs (cone fundamentals) 
    based on CIE2006 cone fundamentals and published literature 
    on observer variability in color matching and in physiological parameters.
        
    Args:
        :age: 
            | 32 or float or int, optional
            | Observer age
        :fieldsize:
            | 10, optional
            | Field size of stimulus in degrees (between 2° and 10°).
        :wl: 
            | None, optional
            | Interpolation/extraplation of :LMS: output to specified wavelengths.
            | None: output original _WL 
        :var_od_lens:
            | 0, optional
            | Std Dev. in peak optical density [%] of lens.
        :var_od_macula:
            | 0, optional
            | Std Dev. in peak optical density [%] of macula.
        :var_od_L:
            | 0, optional
            | Std Dev. in peak optical density [%] of L-cone.
        :var_od_M:
            | 0, optional
            | Std Dev. in peak optical density [%] of M-cone.
        :var_od_S:
            | 0, optional
            | Std Dev. in peak optical density [%] of S-cone.
        :var_shft_L:
            | 0, optional
            | Std Dev. in peak wavelength shift [nm] of L-cone. 
        :var_shft_L:
            | 0, optional
            | Std Dev. in peak wavelength shift [nm] of M-cone.  
        :var_shft_S:
            | 0, optional
            | Std Dev. in peak wavelength shift [nm] of S-cone. 
        :norm_type:
            | None, optional
            | - 'max': normalize LMSq functions to max = 1
            | - 'area': normalize to area
            | - 'power': normalize to power
        :out: 
            | 'lms' or 'xyz', optional
            | Determines output.
        :base:
            | False, boolean, optional
            | The returned energy-based LMS cone fundamentals given to the
            | precision of 9 sign. figs. if 'True', and to the precision of
            | 6 sign. figs. if 'False'.
        :strategy_2: 
            | True, bool, optional
            | Use strategy 2 in github.com/ifarup/ciefunctions issue #121 for 
            | computing the weighting factor. If false, strategy 3 is applied.
        :odata0: 
            | None, optional
            | Dict with uncorrected ocular media and macula density functions and LMS absorptance functions
            | None defaults to the ones stored in _DATA
        :lms_to_xyz_method:
            | None, optional
            | Method to use to determine lms-to-xyz conversion matrix (options: 'asano', 'cietc197')
        :allow_negative_values:
            | False, optional
            | Cone fundamentals or color matching functions should not have negative values.
            |     If False: X[X<0] = 0.
        :normalize_lms_to_xyz_matrix:
            | False, optional
            | Normalize that EEW is always at [100,100,100] in XYZ and LMS system.

            
    Returns:
        :returns: 
            | - 'LMS' [or 'XYZ']: ndarray with individual observer equal area-normalized 
            |           cone fundamentals. Wavelength have been added.
            |   
            | [- 'M': lms to xyz conversion matrix
            |  -  'trans_lens': ndarray with lens transmission 
            |      (no interpolation)
            |  - 'trans_macula': ndarray with macula transmission 
            |      (no interpolation)
            |  - 'sens_photopig' : ndarray with photopigment sens. 
            |      (no interpolation)]
            
    References:
         1. `Asano Y, Fairchild MD, and Blondé L, (2016), 
         Individual Colorimetric Observer Model. 
         PLoS One 11, 1–19. 
         <http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0145671>`_
        
         2. `Asano Y, Fairchild MD, Blondé L, and Morvan P (2016). 
         Color matching experiment for highlighting interobserver variability. 
         Color Res. Appl. 41, 530–539. 
         <https://onlinelibrary.wiley.com/doi/abs/10.1002/col.21975>`_
         
         3. `CIE TC1-36, (2006), 
         Fundamental Chromaticity Diagram with Physiological Axes - Part I 
         (Vienna: CIE). 
         <http://www.cie.co.at/publications/fundamental-chromaticity-diagram-physiological-axes-part-1>`_ 
         
         4. `Asano's Individual Colorimetric Observer Model 
         <https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php>`_
         
         5. `CIE TC1-97 Python code for cone fundamentals and XYZ cmf calculations 
         (by Ivar Farup and Jan Henrik Wold, (c) 2012-2017) 
         <http://github.com/ifarup/ciefunctions>`_
    """
    return compute_cmfs(fieldsize = fieldsize, age = age, wl = wl,
                        var_od_lens = var_od_lens, var_od_macula = var_od_macula,
                        var_shft_LMS = [var_shft_L, var_shft_M, var_shft_S],
                        var_od_LMS = [var_od_L, var_od_M, var_od_S], 
                        norm_type = norm_type, out = out,
                        base = base, strategy_2 = strategy_2, odata0 = odata0,
                        lms_to_xyz_method = lms_to_xyz_method, 
                        allow_negative_values = allow_negative_values,
                        normalize_lms_to_xyz_matrix = normalize_lms_to_xyz_matrix)


def getMonteCarloParam(n_obs = 1, stdDevAllParam = _DATA['stdev'].copy()):
    """
    Get dict with normally-distributed physiological factors 
    for a population of observers.
    
    Args:
        :n_obs: 
            | 1, optional
            | Number of individual observers in population.
        :stdDevAllParam:
            | _DATA['stdev'], optional
            | Dict with parameters for:
            |     ['od_lens', 'od_macula', 
            |      'od_L', 'od_M', 'od_S', 
            |      'shft_L', 'shft_M', 'shft_S']
    
    Returns:
        :returns: 
            | dict with n_obs randomly drawn parameters.
    """

    varParam = {}
    keys = [x for x in stdDevAllParam.keys() if x != 'dsrc']
    for k in list(keys):
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
    t_num = _DATA['USCensus2010population'] 
    
    list_AgeCensus = t_num[0] 
    freq_AgeCensus = np.round(t_num[1]/1000) # Reduce # of populations to manageable number, this doesn't change probability
    
    # Remove age < 10 and 70 < age:
    freq_AgeCensus[:10] = 0
    freq_AgeCensus[71:] = 0
      
    list_Age = [] 
    for k in range(len(list_AgeCensus)):
        list_Age = np.hstack((list_Age, np.repeat(list_AgeCensus[k],freq_AgeCensus[k]))) 

    return list_Age    

def genMonteCarloObs(n_obs = 1, fieldsize = 10, list_Age = [32], wl = None, 
                     norm_type = None, out = 'lms', base = False, 
                     strategy_2 = True, odata0 = None,
                     lms_to_xyz_method = None, allow_negative_values = False):
    """
    Monte-Carlo generation of individual observer cone fundamentals.
    
    Args: 
        :n_obs: 
            | 1, optional
            | Number of observer CMFs to generate.
        :list_Age:
            | list of observer ages or str, optional
            | Defaults to 32 (cfr. CIE2006 CMFs)
            | If 'us_census': use US population census of 2010 
            | to generate list_Age.
        :fieldsize: 
            | fieldsize in degrees (between 2° and 10°), optional
            | Defaults to 10°.
        :wl: 
            | None, optional
            | Interpolation/extraplation of :LMS: output to specified wavelengths.
            | None: output original _WL 
        :norm_type:
            | None, optional
            | - 'max': normalize LMSq functions to max = 1
            | - 'area': normalize to area
            | - 'power': normalize to power
        :out: 
            | 'lms' or 'xyz', optional
            | Determines output.
        :base:
            | False, boolean, optional
            | The returned energy-based LMS cone fundamentals given to the
            | precision of 9 sign. figs. if 'True', and to the precision of
            | 6 sign. figs. if 'False'.
        :strategy_2: 
            | True, bool, optional
            | Use strategy 2 in github.com/ifarup/ciefunctions issue #121 for 
            | computing the weighting factor. If false, strategy 3 is applied.
        :odata0: 
            | None, optional
            | Dict with uncorrected ocular media and macula density functions and LMS absorptance functions
            | None defaults to the ones stored in _DATA
        :lms_to_xyz_method:
            | None, optional
            | Method to use to determine lms-to-xyz conversion matrix (options: 'asano', 'cietc197')
        :allow_negative_values:
            | False, optional
            | Cone fundamentals or color matching functions should not have negative values.
            |     If False: X[X<0] = 0.
    
    Returns:
        :returns: 
            | LMS [,var_age, vAll] 
            |   - LMS: ndarray with population LMS functions.
            |   - var_age: ndarray with population observer ages.
            |   - vAll: dict with population physiological factors (see .keys()) 
            
    References:
         1. `Asano Y., Fairchild M.D., and Blondé L., (2016), 
         Individual Colorimetric Observer Model. 
         PLoS One 11, 1–19. 
         <http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0145671>`_
         
         2. `Asano Y, Fairchild MD, Blondé L, and Morvan P (2016). 
         Color matching experiment for highlighting interobserver variability. 
         Color Res. Appl. 41, 530–539. 
         <https://onlinelibrary.wiley.com/doi/abs/10.1002/col.21975>`_
         
         3. `CIE TC1-36, (2006), 
         Fundamental Chromaticity Diagram with Physiological Axes - Part I. 
         (Vienna: CIE). 
         <http://www.cie.co.at/publications/fundamental-chromaticity-diagram-physiological-axes-part-1>`_ 
         
         4. `Asano's Individual Colorimetric Observer Model 
         <https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php>`_
    """
    # Get Normally-distributed Physiological Factors:
    vAll = getMonteCarloParam(n_obs = n_obs) 
     
    if isinstance(list_Age,str): 
        if list_Age == 'us_census':
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

    if odata0 is None:
        odata = _DATA['odata']
    else:
        odata = odata0
    
    # Set requested wavelength range:
    if wl is None:
        wl = odata['wls']
    else:
        wl = getwlr(wl3 = wl)
     
    if 'lms' in out.lower().split(','):
        out_ = 'lms'
    elif 'xyz' in out.lower().split(','):
        out_ = 'xyz'
    else:
        raise Exception("Must request either 'lms' or 'xyz' in :out:.")
        
    LMS_All = np.zeros((3+1, wl.shape[0],n_obs)); LMS_All.fill(np.nan)
    if 'M' in out.split(','):
        out_ = out_+',M'
        M_All = np.zeros((3, 3,n_obs)); M_All.fill(np.nan)
    for k in range(n_obs):
        returned = cie2006cmfsEx(age = var_age[k], fieldsize = fieldsize, wl = wl,\
                                var_od_lens = vAll['od_lens'][k], var_od_macula = vAll['od_macula'][k], \
                                var_od_L = vAll['od_L'][k], var_od_M = vAll['od_M'][k], var_od_S = vAll['od_S'][k],\
                                var_shft_L = vAll['shft_L'][k], var_shft_M = vAll['shft_M'][k], var_shft_S = vAll['shft_S'][k],\
                                out = out_+',trans_lens,trans_macula,sens_photopig',
                                norm_type = norm_type,  base = base, \
                                strategy_2 = strategy_2, odata0 = odata,\
                                lms_to_xyz_method = lms_to_xyz_method, allow_negative_values = allow_negative_values)
        if 'M' not in out.split(','):
            t_LMS, t_trans_lens, t_trans_macula, t_sens_photopig = returned
        else:
            t_LMS, t_M, t_trans_lens, t_trans_macula, t_sens_photopig = returned
            M_All[:,:,k] = t_M
        LMS_All[:,:,k] = t_LMS
        
    if n_obs == 1:
        LMS_All = np.squeeze(LMS_All, axis = 2)
        if 'M' in out.split(','): 
            M_All = np.squeeze(M_All, axis = 2)
    
    if ('xyz' in out.lower().split(',')):
        out = out.replace('xyz','LMS').replace('XYZ','LMS')
    if ('lms' in out.lower().split(',')):
        out = out.replace('lms','LMS')

    if (out == 'LMS'):
        return LMS_All
    elif (out == 'LMS,M'):
        return LMS_All,M_All
    elif (out == 'LMS,var_age,vAll'):
        return LMS_All, var_age, vAll 
    elif (out == 'LMS,M,var_age,vAll'):
        return LMS_All, M_All, var_age, vAll 
    else:
        return eval(out)

        
def getCatObs(n_cat = 10, fieldsize = 2,  wl = None, 
             norm_type = None, out = 'lms', base = False, 
             strategy_2 = True, odata0 = None,
             lms_to_xyz_method = None, allow_negative_values = False):
    """
    Generate cone fundamentals for categorical observers.
    
    Args: 
        :n_cat: 
            | 10, optional
            | Number of observer CMFs to generate.
        :fieldsize:
            | fieldsize in degrees (between 2° and 10°), optional
            | Defaults to 10°.
        :out: 
            | 'LMS' or str, optional
            | Determines output.
        :wl: 
            | None, optional
            | Interpolation/extraplation of :LMS: output to specified wavelengths.
            |  None: output original _WL 
        :norm_type:
            | None, optional
            | - 'max': normalize LMSq functions to max = 1
            | - 'area': normalize to area
            | - 'power': normalize to power
        :out: 
            | 'lms' or 'xyz', optional
            | Determines output.
        :base:
            | False, boolean, optional
            | The returned energy-based LMS cone fundamentals given to the
            | precision of 9 sign. figs. if 'True', and to the precision of
            | 6 sign. figs. if 'False'.
        :strategy_2: 
            | True, bool, optional
            | Use strategy 2 in github.com/ifarup/ciefunctions issue #121 for 
            | computing the weighting factor. If false, strategy 3 is applied.
        :odata0: 
            | None, optional
            | Dict with uncorrected ocular media and macula density functions and LMS absorptance functions
            | None defaults to the ones stored in _DATA
        :lms_to_xyz_method:
            | None, optional
            | Method to use to determine lms-to-xyz conversion matrix (options: 'asano', 'cietc197')
        :allow_negative_values:
            | False, optional
            | Cone fundamentals or color matching functions should not have negative values.
            |     If False: X[X<0] = 0.
    
    Returns:
        :returns:
            | LMS [,var_age, vAll] 
            |   - LMS: ndarray with population LMS functions.
            |   - var_age: ndarray with population observer ages.
            |   - vAll: dict with population physiological factors (see .keys()) 
    
    Notes:
        1. Categorical observers are observer functions that would represent 
        color-normal populations. They are finite and discrete as opposed to 
        observer functions generated from the individual colorimetric observer 
        model. Thus, they would offer more convenient and practical approaches
        for the personalized color imaging workflow and color matching analyses.
        Categorical observers were derived in two steps. 
        At the first step, 10000 observer functions were generated from the 
        individual colorimetric observer model using Monte Carlo simulation. 
        At the second step, the cluster analysis, a modified k-medoids 
        algorithm, was applied to the 10000 observers minimizing the squared 
        Euclidean distance in cone fundamentals space, and categorical 
        observers were derived iteratively. Since the proposed categorical 
        observers are defined by their physiological parameters and ages, their
        CMFs can be derived for any target field size.
        2. Categorical observers were ordered by the importance; 
        the first categorical observer vas the average observer equivalent to 
        CIEPO06 with 38 year-old for a given field size, followed by the second
        most important categorical observer, the third, and so on.
        
        3. see: https://www.rit.edu/cos/colorscience/re_AsanoObserverFunctions.php
    """
    # Use Iteratively Derived Cat.Obs.:
    var_age = _DATA['CatObsPfctr']['age'].copy()
    vAll = _DATA['CatObsPfctr'].copy()
    vAll.pop('age')

    if odata0 is None:
        odata = _DATA['odata']
    else:
        odata = odata0
    
    # Set requested wavelength range:
    if wl is None:
        wl = odata['wls']
    else:
        wl = getwlr(wl3 = wl)

    if 'lms' in out.lower().split(','):
        out_ = 'lms'
    elif 'xyz' in out.lower().split(','):
        out_ = 'xyz'
    else:
        raise Exception("Must request either 'lms' or 'xyz' in :out:.")

    LMS_All = np.zeros((3+1,wl.shape[0],n_cat)); LMS_All.fill(np.nan)
    if 'M' in out.split(','):
        out_ = out_+',M'
        M_All = np.zeros((3, 3,n_cat)); M_All.fill(np.nan)
    for k in range(n_cat):
        returned = cie2006cmfsEx(age = var_age[k],fieldsize = fieldsize, wl = wl,\
                              var_od_lens = vAll['od_lens'][k],\
                              var_od_macula = vAll['od_macula'][k],\
                              var_od_L = vAll['od_L'][k],\
                              var_od_M = vAll['od_M'][k],\
                              var_od_S = vAll['od_S'][k],\
                              var_shft_L = vAll['shft_L'][k],\
                              var_shft_M = vAll['shft_M'][k],\
                              var_shft_S = vAll['shft_S'][k],\
                              out = out_,\
                              norm_type = norm_type,  base = base, \
                              strategy_2 = strategy_2, odata0 = odata,\
                              lms_to_xyz_method = lms_to_xyz_method, \
                              allow_negative_values = allow_negative_values)
        if 'M' in out.split(','):
            t_LMS, t_M = returned
        else:
            t_LMS = returned
        LMS_All[:,:,k] = t_LMS 
    
    LMS_All[np.where(LMS_All < 0)] = 0
    
    if n_cat == 1:
        LMS_All = np.squeeze(LMS_All, axis = 2)

    if ('xyz' in out.lower().split(',')):
        out = out.replace('xyz','LMS').replace('XYZ','LMS')
    if ('lms' in out.lower().split(',')):
        out = out.replace('lms','LMS')
        
    if (out == 'LMS'):
        return LMS_All
    elif (out == 'LMS,M'):
        return LMS_All,M_All
    elif (out == 'LMS,var_age,vAll'):
        return LMS_All,var_age,vAll 
    elif (out == 'LMS,M,var_age,vAll'):
        return LMS_All,M_All,var_age,vAll 
    else:
        return eval(out)
def get_lms_to_xyz_matrix(fieldsize = 10):
    """
    Get the lms to xyz conversion matrix for specific fieldsize using Asano's method 
    (i.e. use as a weighted combination of the 2° and 10° matrices).
    
    Args:
        :fieldsize: 
            | fieldsize in degrees (between 2° and 10°), optional
            | Defaults to 10°.
            
    Returns:
        :M: 
            | ndarray with conversion matrix.
    
    Note: 
        For intermediate field sizes (2°<fieldsize<10°) the conversion matrix 
        is calculated by linear interpolation between 
        the _DATA['M']['2d'] and _DATA['M']['10d']matrices.
    """
    if fieldsize < 2:
        fieldsize = 2
    elif fieldsize > 10:
        fieldsize = 10 
    a = (10-fieldsize)/8     
    M = _DATA['M']['2d']*a + _DATA['M']['10d']*(1-a)
    return M

def lmsb_to_xyzb(lms, fieldsize = 10, out = 'xyz', allow_negative_values = False):
    """
    Convert from LMS cone fundamentals to XYZ color matching functions using Asano's method
    (use conversion matrix determined as a determined a weighted combination of the 2° and 10° matrices).
    
    Args:
        :lms: 
            | ndarray with lms cone fundamentals, optional
        :fieldsize: 
            | fieldsize in degrees, optional
            | Defaults to 10°.
        :out: 
            | 'xyz' or str, optional
            | Determines output.
        :allow_negative_values:
            | False, optional
            | XYZ color matching functions should not have negative values.
            |     If False: xyz[xyz<0] = 0.
        :method:
            | None, optional
            | None defaults to _LMS_TO_XYZ_METHOD
            | Options: 'asano' (see note below), 'cietc197'
            
    Returns:
        :returns:
            | LMS 
            |   - LMS: ndarray with population XYZ color matching functions.    
    
    Note: 
        If method == 'asano': For intermediate field sizes (2°<fieldsize<10°) 
        the conversion matrix is calculated by linear interpolation between 
        the _DATA['M']['2d'] and _DATA['M']['10d']matrices.
    """
    wl = lms[None,0] #store wavelengths
    M = get_lms_to_xyz_matrix(fieldsize = fieldsize)
    if lms.ndim > 2:
        xyz = np.vstack((wl,math.dot23(M,lms[1:,...], keepdims = False)))
    else:
        xyz = np.vstack((wl,np.dot(M,lms[1:,...])))
    if allow_negative_values == False:
        xyz[np.where(xyz < 0)] = 0
    if out.lower() == 'xyz':
        return xyz
    elif out.lower() == 'xyz,m':
        return xyz, M
    else:
        return eval(out)

def add_to_cmf_dict(bar = None, cieobs = 'indv', K = 683, M = np.eye(3)):
    """
    Add set of cmfs to _CMF dict.
    
    Args:
        :bar: 
            | None, optional
            | Set of CMFs. None: initializes to empty ndarray.
        :cieobs:
            | 'indv' or str, optional
            | Name of CMF set.
        :K: 
            | 683 (lm/W), optional
            | Conversion factor from radiometric to photometric quantity.
        :M: 
            | np.eye, optional
            | Matrix for lms to xyz conversion.
    """
    if bar is None:
        wl3 = getwlr(_WL3)
        bar = np.vstack((wl3,np.empty((3,wl3.shape[0]))))
    _CMF['types'].append(cieobs)
    _CMF[cieobs] = {'bar' : bar}
    _CMF[cieobs]['K'] = K
    _CMF[cieobs]['M'] = M
    #return _CMF
   
def plot_cmfs(cmf,axh = None, **kwargs):
    """
    Plot cmf set.
    """
    if axh is None:
        fig = plt.figure()
        axh = fig.add_subplot(111)
    axh.plot(cmf[0],cmf[1], color ='r', **kwargs)
    axh.plot(cmf[0],cmf[2], color ='g', **kwargs)
    axh.plot(cmf[0],cmf[3], color ='b', **kwargs)
    axh.set_xlabel("Wavelenghts (nm)")
    return axh

if __name__ == '__main__':
    import luxpy as lx
    init(use_my_round=True,use_sign_figs=True,use_chop=True,dsrc_lms_odens='cietc197',lms_to_xyz_method='cietc197')
    xyz2b,M2 = compute_cmfs(fieldsize=2.1,age=32,out='xyz,M',lms_to_xyz_method='cietc197',norm_type=None)
    print(lx.spd_to_xyz(lx._CIE_E,relative=False,cieobs=xyz2b,K=1))
    print(np.dot(M2,np.array([[200,200,200]]).T).T)
    ax = plot_cmfs(xyz2b)
    
if __name__ == '__main__':
    
    data = load_database(wl=_WL)
    _DATA = data.copy()
    
    lms,M=compute_cmfs(out='xyz,M',odata0=data['odata'], var_shft_LMS=[15,0,0],norm_type = 'area',lms_to_xyz_method='cietc197')
    
    
    outcmf = 'xyz'
    
    out = outcmf + ',trans_lens,trans_macula,sens_photopig,LMSa'
    LMS, trans_lens, trans_macula, sens_photopig, LMSa = cie2006cmfsEx(out = out, norm_type = 'area',var_shft_L=15,lms_to_xyz_method='asano')
    
    plt.figure()
    plt.plot(LMS[0],LMS[1], color ='r', linestyle='-')
    plt.plot(LMS[0],LMS[2], color ='g', linestyle='-')
    plt.plot(LMS[0],LMS[3], color ='b', linestyle='-')
    plt.title('cie2006cmfsEx(...)')
    
#    plt.figure()
    plt.plot(lms[0],lms[1], color ='r', linestyle='--')
    plt.plot(lms[0],lms[2], color ='g', linestyle='--')
    plt.plot(lms[0],lms[3], color ='b', linestyle='--')
    plt.show()

if __name__ == 'x__main__':    
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
    cct2,duv2 = xyz_to_cct_ohno2014(xyz2, cieobs = '1931_2', out = 'cct,duv')
    cct1,duv1 = xyz_to_cct_ohno2014(xyz1, cieobs = 'CatObs1', out = 'cct,duv')
    print('cct,duv using 1931_2: {:1.0f} K, {:1.4f}'.format(cct2[0,0],duv2[0,0]))
    print('cct,duv using CatObs1: {:1.0f} K, {:1.4f}'.format(cct1[0,0],duv1[0,0]))