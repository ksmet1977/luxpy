# CIE224-2017 interpolation:

#------luxpy import-------------------------------------------------------
import os 
if "code\\py" not in os.getcwd(): os.chdir("./code/py/")
root = "../../" if "code\\py" in os.getcwd() else  "./"

import luxpy as lx
print('luxpy version:', lx.__version__,'\n')

from luxpy import getwld
from luxpy.math import _interpolate_with_nans, _extrap_y

#-----other imports-------------------------------------------------------
import numpy as np
import luxpy as lx
import matplotlib.pyplot as plt 
import pandas as pd

# def _interp1_sprague_cie227_2017(x,y,xn,
#                                 extrap = 'linear', 
#                                 force_scipy_interpolator = False,
#                                 scipy_interpolator = 'InterpolatedUnivariateSpline',
#                                 delete_nans = True,
#                                 choose_most_efficient_interpolator = False):
#     """ 
#     Perform a 1-dimensional Sprague interpolation as defined in (CIE-224-2017).
#     (Private: for use with interp1_sprague_cie224_2017 function)
#     """
#     # Use sprague interpolation as defined in CIE227-2017


#     # Do extrapolation:
#     xn_x, yne = _extrap_y(x, y, xn, extrap = extrap, 
#                         force_scipy_interpolator = force_scipy_interpolator,
#                         scipy_interpolator = scipy_interpolator,
#                         delete_nans = delete_nans,
#                         choose_most_efficient_interpolator = choose_most_efficient_interpolator)


#     # Extrapolate 2 extra values beyond x-boundary:
#     dx = getwld(x)
#     if not (isinstance(dx, float) | isinstance(dx,int)):
#         raise Exception('Sprague interpolation method only defined for equally spaced x!')
#     xe = np.hstack((x[0]-2*dx, x[0]-dx, x, x[-1]+dx, x[-1]+2*dx, x[-1]+3*dx))
#     ye = np.hstack((y[:,:1], y[:,:1], y, y[:,-1:], y[:,-1:],y[:,-1:]))

#     # Sprague coefficients for spacing of 5 to 1:
#     Smn = np.array([[0.0000,  0.0000, 1.0000, 0.0000,  0.0000, 0.0000],
#                     [0.0128, -0.0976, 0.9344, 0.1744, -0.0256, 0.0016],
#                     [0.0144, -0.1136, 0.7264, 0.4384, -0.0736, 0.0080],
#                     [0.0080, -0.0736, 0.4384, 0.7264, -0.1136, 0.0144],
#                     [0.0016, -0.0256, 0.1744, 0.9344, -0.0976, 0.0128]
#                     ])

#     # Find indices of Smn for bulk processing:
#     i = np.searchsorted(xe, xn)
#     I = np.vstack((i-2,i-1,i,i+1,i+2,i+3))
#     Iu = np.unique((I),axis=1)

#     # process xe (as check):
#     xe_ = (Smn @xe[Iu]).T
#     xe__ = np.reshape(xe_,(5*x.shape[0],))
#     c = (xe__>=xn[0]) & (xe__<=xn[-1])
#     xe__ = xe__[c]

#     # Process ye:
#     ye = ye.T
#     ye_ = np.transpose((Smn @ np.transpose(ye[Iu],(2,0,1))),(0,2,1))
#     ye__ = np.reshape(ye_,(ye_.shape[0],5*x.shape[0]))
#     ye__ = ye__[:,c]
#     yn = ye__

#     if yne is None:
#         return yn
#     else:
#         yne[:,(xn>=x[0]) & (xn<=x[-1])] = yn
#         return yne


# def interp1_sprague_cie227_2017(X, Y, Xnew, extrap = 'linear', 
#                      force_scipy_interpolator = False,
#                      scipy_interpolator = 'InterpolatedUnivariateSpline',
#                      delete_nans = True,
#                      choose_most_efficient_interpolator = False, verbosity = 0):
#     """ 
#     Perform a 1-dimensional Sprague interpolation according to CIE-224-2017.
    
#     Args:
#         :X:
#             | ndarray with n-dimensional coordinates.
#         :Y: 
#             | ndarray with values at coordinates in X.
#         :Xnew:
#             | ndarray of new coordinates.
#         :extrap:
#             | (np.nan, np.nan) or string, optional
#             | If tuple: fill with values in tuple (<X[0],>X[-1])
#             | If string:  ('linear', 'quadratic', 'cubic', 'zeros', 'const')
#         :force_scipy_interpolator:
#             | False, optional
#             | If False: numpy.interp function is used for linear interpolation when no or linear extrapolation is used/required (fast!). 
#         :scipy_interpolator:
#             | 'InterpolatedUnivariateSpline', optional
#             | options: 'InterpolatedUnivariateSpline', 'interp1d'
#         :delete_nans:
#             | True, optional
#             | If NaNs are present, remove them and (and try to) interpolate without them.

#     Returns:
#         :Yn:
#             | ndarray with values at new coordinates in Xnew.
#     """
#     if verbosity > 0: print('Interpolation: using luxpy interp1_sprague_cie227_2017')
#     fintp = lambda X,Y,Xnew: _interp1_sprague_cie227_2017(X, Y, Xnew, extrap = extrap, 
#                                                         force_scipy_interpolator = force_scipy_interpolator,
#                                                         scipy_interpolator = scipy_interpolator, 
#                                                         delete_nans = delete_nans,
#                                                         choose_most_efficient_interpolator = choose_most_efficient_interpolator) 
#     return _interpolate_with_nans(fintp, X, Y, Xnew, 
#                                 delete_nans = delete_nans, 
#                                 nan_indices = None)  


if __name__ == '__main__':

    # Get some data to interpolate:
    rfl5 = lx._CRI_RFL['cie-224-2017']['99']['5nm']
    rfl1 = lx._CRI_RFL['cie-224-2017']['99']['1nm']
    rfl1_ies = lx._CRI_RFL['ies-tm30']['99']['1nm']

    # limit to 380-780 nm range (as required by CIE224-2017 Rf calculation):
    rfl5 = rfl5[:,(rfl5[0]>=380) & (rfl5[0]<=780)] # 5 nm values from CIE 5 nm Rf excel-calculator
    rfl1 = rfl1[:,(rfl1[0]>=380) & (rfl1[0]<=780)] # 1 nm values from CIE 1 nm Rf excel-calculator
    rfl1_ies = rfl1_ies[:,(rfl1_ies[0]>=380) & (rfl1_ies[0]<=780)] # 1 nm values from CIE 1 nm Rf excel-calculator

    # Split rfls in wavelength part and rfl factors:
    wl, y = rfl5[0], rfl5[1:]

    # Generate new wavelength range:
    wln = lx.getwlr([380,780,1])

    # Perform Sprague interpolation:
    yn = lx.math.interp1_sprague_cie224_2017(wl,y,wln)
    yn5 = lx.math.interp1_sprague5(wl,y,wln)

    # Perform linear and cubic interpolation for comparison:
    ync = lx.cie_interp(rfl5,wln,kind = 'cubic')[1:]
    ynl = lx.cie_interp(rfl5,wln,kind = 'linear')[1:]

    # Use 1nm values from CIE 1nm Rf excel-calculator:
    y1 = rfl1[1:]
    y1_ies = rfl1_ies[1:]

    plt.plot(wln,yn.T,label=f'Sprague (CIE224) interpolated')
    plt.plot(wln,yn5.T,linestyle = '--', label=f'Sprague5 interpolated')
    plt.plot(wl,y.T,linestyle = ':', color = 'r', label=f'original')
    #plt.legend()

    print('sprague 5 vs CIE224-2017 sprague: np.abs(yn5-yn).max():',np.abs(yn5-yn).max())
    print('linear vs CIE224-2017 sprague: np.abs(ynl-yn).max():',np.abs(ynl-yn).max())
    print('cubic vs CIE224-2017 sprague: np.abs(ync-yn).max():',np.abs(ync-yn).max())
    print('1nm vs CIE224-2017 sprague: np.abs(rfl1[1:]-yn).max():',np.abs(y1-yn).max())

    print('sprague 5 vs y1: np.abs(yn5-yn).max():',np.abs(yn5-y1).max())
    print('linear vs y1: np.abs(ynl-yn).max():',np.abs(ynl-y1).max())
    print('cubic vs y1: np.abs(ync-yn).max():',np.abs(ync-y1).max())
    print('y1 vs CIE224-2017 sprague: np.abs(y1-yn).max():',np.abs(y1-yn).max())
    print('y1_ies vs CIE224-2017 sprague: np.abs(y1_ies-yn).max():',np.abs(y1_ies-yn).max())
    print('y1_ies vs y1: np.abs(y1_ies-y1).max():',np.abs(y1_ies-y1).max())

    print('\n!!! Replaced 1 nm data for CIE224 from excel with that of IES TM30-24 as the latter is only 1e-16 different from sprague CIE224 interpolation')
    