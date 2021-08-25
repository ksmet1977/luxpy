# -*- coding: utf-8 -*-
"""
Test extrapolation accuracy 

Created on Wed Aug 25 16:52:39 2021

@author: u0032318
"""

import scipy
from scipy.interpolate import interp1d
import luxpy as lx
import numpy as np 
import matplotlib.pyplot as plt 


  
def interp_ext(x,y,xn,kind = 'cubic', extrap_log = True):

    #interpolation part:
    interp = lambda x,y,xn: interp1d(x,y, kind = kind, bounds_error = False, fill_value = 0.0)(xn)
    
    # extrapolation part:
    if extrap_log:
        extrap = lambda x,y,xn: np.exp(interp1d(x,np.log(y + 1*1e-16), kind = 'quadratic', bounds_error = False, fill_value = 'extrapolate')(xn))
    else:
        extrap = lambda x,y,xn: interp1d(x,y, kind = 'quadratic', bounds_error = False, fill_value = 'extrapolate')(xn)

    yn = np.atleast_2d(interp(x,y,xn))
    yn_ext = np.atleast_2d(extrap(x,y,xn))
    yn_ext[:,(xn >= x[0]) & (xn <= x[-1])] = 0
    return yn + yn_ext

def interp_ext2(x,y,xn,kind = 'cubic', extrap_log = True):

    #interpolation part:
    interp = lambda x,y,xn: interp1d(x,y, kind = kind, bounds_error = False, fill_value = 0.0)(xn)
    
    # extrapolation part:
    if extrap_log:
        extrap = lambda x,y,xn: np.exp(interp1d(x,np.log(y + 1*1e-16), kind = 'linear', bounds_error = False, fill_value = 'extrapolate')(xn))
    else:
        extrap = lambda x,y,xn: interp1d(x,y, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')(xn)

    yn = np.atleast_2d(interp(x,y,xn))
    yn_ext = np.atleast_2d(extrap(x,y,xn))
    yn_ext[:,(xn >= x[0]) & (xn <= x[-1])] = 0
    return yn + yn_ext
    

# wlr0 = [380,780,1]
# wlr0 = lx.getwlr(wlr0)
# wlr1 = [400,700,5]
# wlr1 = lx.getwlr(wlr1)
# S0 = np.exp(-((wlr0 - 530)/120)**2)
# S1 = np.exp(-((wlr1 - 530)/120)**2)
# kind = 'quadratic'
# # S1i = np.exp(interp1d(wlr1,np.log(S1 + lx.utils._EPS), kind = kind, bounds_error = False, fill_value = 'extrapolate')(wlr0))
# interp = lambda x,y,xn: interp1d(x,y, kind = kind, bounds_error = False, fill_value = 'extrapolate')(xn)
# interp_log = lambda x,y,xn: np.exp(interp1d(x,np.log(y + 1*1e-16), kind = 'quadratic', bounds_error = False, fill_value = 'extrapolate')(xn))
# S1i = interp(wlr1,S1,wlr0)
# S1i_log = interp_log(wlr1,S1,wlr0)
# S1i_ext = interp_ext(wlr1,S1,wlr0)
# S1e = extrap(wlr1,S1,wlr0)
# # plt.plot(wlr0,S1i.T,'r.-') 
# # # plt.plot(wlr0,S1i_log.T,'g.-') 
# # plt.plot(wlr0,S1i_ext.T,'g.-') 
# # plt.plot(wlr1,S1.T,'b.-') 

spds0 = lx._IESTM3018['S']['data'].copy()
spds0 = spds0[:,::1]
spds1 = spds0[:,(spds0[0]>=400) & (spds0[0]<=700)]
# spds1 = spds1[:,::5]
# spds1i = interp(spds1[0],spds1[1:],spds0[0])
# spds1i_log = interp_log(spds1[0],spds1[1:],spds0[0])
kind = 'cubic'
spds1i_ext = np.vstack((spds0[0],interp_ext(spds1[0],spds1[1:],spds0[0], extrap_log = False, kind = kind)))
spds1i_ext_log = np.vstack((spds0[0],interp_ext(spds1[0],spds1[1:],spds0[0], extrap_log = True, kind = kind)))
spds1i_ext2 = np.vstack((spds0[0],interp_ext2(spds1[0],spds1[1:],spds0[0], extrap_log = False, kind = kind)))
spds1i_ext2_log = np.vstack((spds0[0],interp_ext2(spds1[0],spds1[1:],spds0[0], extrap_log = True, kind = kind)))

Yuv0 = lx.xyz_to_Yuv(lx.spd_to_xyz(spds0))
Yuv1_ext = lx.xyz_to_Yuv(lx.spd_to_xyz(spds1i_ext))
Yuv1_ext_log = lx.xyz_to_Yuv(lx.spd_to_xyz(spds1i_ext_log))
Yuv1_ext2 = lx.xyz_to_Yuv(lx.spd_to_xyz(spds1i_ext2))
Yuv1_ext2_log = lx.xyz_to_Yuv(lx.spd_to_xyz(spds1i_ext2_log))


DEuv = lambda Yuv0, Yuv1: np.sqrt(((Yuv1 - Yuv0)[...,1:]**2).sum(axis=-1))
print('mean:',np.nanmean(DEuv(Yuv0,Yuv1_ext)), np.nanmean(DEuv(Yuv0,Yuv1_ext_log)), np.nanmean(DEuv(Yuv0,Yuv1_ext2)), np.nanmean(DEuv(Yuv0,Yuv1_ext2_log)))
print('median:',np.nanmedian(DEuv(Yuv0,Yuv1_ext)), np.nanmedian(DEuv(Yuv0,Yuv1_ext_log)), np.nanmedian(DEuv(Yuv0,Yuv1_ext2)), np.nanmedian(DEuv(Yuv0,Yuv1_ext2_log))) 
print('max:',np.nanmax(DEuv(Yuv0,Yuv1_ext)), np.nanmax(DEuv(Yuv0,Yuv1_ext_log)), np.nanmax(DEuv(Yuv0,Yuv1_ext2)), np.nanmax(DEuv(Yuv0,Yuv1_ext2_log)))


# plt.plot(spds0[0],spds1i[1:].T,'r.-')
# plt.plot(spds0[0],spds1i_log[1:].T,'g.-')  
# plt.plot(spds0[0],spds1i_ext[1:].T,'g.-')
# plt.plot(spds1[0],spds1[1:].T,'b.-') 
