# -*- coding: utf-8 -*-
"""
###############################################################################
# Module with scaling functions from DE to CRI-scale.
###############################################################################

# linear_scale():  Linear color rendering index scale from CIE13.3-1974/1995:   
                    Rfi,a = 100 - c1*DEi,a. (c1 = 4.6)

# log_scale(): Log-based color rendering index scale from Davis & Ohno (2009):  
                    Rfi,a = 10 * ln(exp((100 - c1*DEi,a)/10) + 1)

# psy_scale():  Psychometric based color rendering index scale from CRI2012 (Smet et al. 2013, LRT):  
                    Rfi,a = 100 * (2 / (exp(c1*abs(DEi,a)**(c2) + 1))) ** c3

#------------------------------------------------------------------------------                    
Created on Sun Apr 15 11:07:29 2018

@author: kevin.smet
"""
from .. import np
__all__ = ['linear_scale', 'log_scale', 'psy_scale']

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


