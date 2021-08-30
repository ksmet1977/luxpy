# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:48:21 2021

@author: u0032318
"""
import numpy as np
from scipy.interpolate import interp1d 

# Coefficients used to generate extra points for boundaries interpolation:
_SPRAGUE_COEFFICIENTS = np.array([
                                 [884, -1960, 3033, -2648, 1080, -180],
                                 [508, -540, 488, -367, 144, -24],
                                 [-24, 144, -367, 488, -540, 508],
                                 [-180, 1080, -2648, 3033, -1960, 884],
                                 ]).T / 209.0

def interp1d_sprague5(x, y, xn, extrap = None):
    """ 
    Perform a 1-dimensional 5th order Sprague interpolation.
    
    Args:
        :x:
            | ndarray with n-dimensional coordinates.
        :y: 
            | ndarray with values at coordinates in x.
        :xn:
            | ndarrat of new coordinates.
        :extrap:
            | (np.nan, np.nan) or string, optional
            | If tuple: fill with values in tuple (<x[0],>x[-1])
            | If string:  ('zeros','linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous','next')
            |           for more info on the other options see: scipy.interpolate.interp1d?
    Returns:
        :yn:
            | ndarray with values at new coordinates in xn.
            
    
    """
    # Do extrapolation:
    if ((xn<x[0]) | (xn>x[-1])).any(): # extrapolation needed !
        if isinstance(extrap,tuple):
            if extrap[0] == extrap[1]: 
                yne = np.ones((y.shape[0],len(xn)))*extrap[0]
            else:
                yne = np.zeros((y.shape[0],len(xn)))
                yne[:,(xn<x[0])] = extrap[0]
                yne[:,(xn>x[-1])] = extrap[1]
        elif isinstance(extrap,str):
            yne = interp1d(x, y, kind = extrap, bounds_error = False, fill_value = 'extrapolate')(xn)
        else:
            raise Exception('Invalid option for extrap argument. Only tuple and string allowed.')
        xn_x = xn[(xn>=x[0]) & (xn<=x[-1])]
    else:
        xn_x = xn
        yne = None
     
    # Check equal x-spacing:
    dx = np.diff(x)
    if np.all(dx == dx[0]):
        dx = dx[0] 
    else:
        raise Exception('Elements in x are not equally spaced!')
        
    # Extrapolate x, y with required additional elements for Sprague to work:
    xe = np.hstack((x[0] - 2*dx, x[0] - dx, x, x[-1] + dx, x[-1] + 2*dx))
    
    y = np.atleast_2d(y)
    ye1 = (y[:, :6] @ _SPRAGUE_COEFFICIENTS[:,0])[:,None]
    ye2 = (y[:, :6] @ _SPRAGUE_COEFFICIENTS[:,1])[:,None]
    ye3 = (y[:,-6:] @ _SPRAGUE_COEFFICIENTS[:,2])[:,None]
    ye4 = (y[:,-6:] @ _SPRAGUE_COEFFICIENTS[:,3])[:,None]
    ye = np.hstack((ye1,ye2,y,ye3,ye4)).T
    
    
    # Evaluate at xn_x (no extrapolation!!):
    i = np.searchsorted(xe, xn_x) - 1
    X = np.atleast_2d((xn_x - xe[i]) / (xe[i + 1] - xe[i])).T

    a0 = ye[i]
    a1 = ((2 * ye[i - 2] - 16 * ye[i - 1] + 16 * ye[i + 1] - 2 * ye[i + 2]) / 24)  
    a2 = ((-ye[i - 2] + 16 * ye[i - 1] - 30 * ye[i] + 16 * ye[i + 1] - ye[i + 2]) / 24) 
    a3 = ((-9 * ye[i - 2] + 39 * ye[i - 1] - 70 * ye[i] + 66 * ye[i + 1] - 33 * ye[i + 2] + 7 * ye[i + 3]) / 24)
    a4 = ((13 * ye[i - 2] - 64 * ye[i - 1] + 126 * ye[i] - 124 * ye[i + 1] + 61 * ye[i + 2] - 12 * ye[i + 3]) / 24)
    a5 = ((-5 * ye[i - 2] + 25 * ye[i - 1] - 50 * ye[i] + 50 * ye[i + 1] - 25 * ye[i + 2] + 5 * ye[i + 3]) / 24)

    yn = (a0 + a1*X + a2*X**2 + a3*X**3 + a4*X**4 + a5*X**5).T
    
    if yne is None:
        return yn
    else:
        yne[:,(xn>=x[0]) & (xn<=x[-1])] = yn
        return yne
            
if __name__ == '__main__':
    import luxpy as lx
    import matplotlib.pyplot as plt 
    
    spd1 = lx._CIE_D65.copy()
    spd2 = lx._CIE_A.copy()
    gt12 = lx.np.vstack((spd1,spd2[1:]))
    spd1 = spd1[:,20:-50:5]
    spd2 = spd2[:,20:-50:5]
    # spd1 = spd1[:,::5]
    # spd2 = spd2[:,::5]
    x = spd1[0]
    y1 = spd1[1]
    y2 = spd2[1:]
    y12 = lx.np.vstack((y1,y2))
    
    xn = gt12[0]
    yn = interp1d_sprague5(x, y12, xn, extrap = (0,0))
    
    plt.plot(x,y12.T,'b-')
    plt.plot(xn,yn.T,'r--')
