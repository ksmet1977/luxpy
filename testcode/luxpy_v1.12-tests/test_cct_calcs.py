# -*- coding: utf-8 -*-
#------luxpy import-------------------------------------------------------
import os 
if "code\\py" not in os.getcwd(): os.chdir("./code/py/")
root = "../../" if "code\\py" in os.getcwd() else  "./"

import luxpy as lx
print('luxpy version:', lx.__version__,'\n')

#------other imports------------------------------------------------------
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt

from luxpy import xyz_to_cct_bruteforce, generate_lut_bf

#-------------------------------------------------------------------------

    



# test difference between vlbar and ybar from CMF:
spds = np.round(lx._IESTM3018['S']['data'].copy(),15)
spd = spds[[0,1]]

cmf2,K2 = lx._CMF["1931_2"]["bar"].copy(),lx._CMF["1931_2"]["K"]
cmf2_ = lx.xyzbar(cieobs="1931_2")
print('cmf2 == cmf2_ ?:', (((cmf2-cmf2_)==0).all()))

cctduv_calculator = np.array([6425.14268404576,0.00719612617474981])

force_tolerance = False

xyz1 = lx.spd_to_xyz(spd,cieobs='1931_2',relative=False)
cctduv1 = lx.xyz_to_cct(xyz1,mode='ohno2014',cieobs="1931_2",force_tolerance=force_tolerance,out='[cct,duv]')
xyz2 = lx.spd_to_xyz(spd,cieobs=cmf2,K=K2,relative=False)
cctduv2 = lx.xyz_to_cct(xyz2,mode='ohno2014',cieobs=cmf2,force_tolerance=force_tolerance,out='[cct,duv]')
print('xyz1 == xyz2 ?:', ((xyz1-xyz2)==0).all())
print('cctduv1 == cctduv2 ?:', ((cctduv1-cctduv2)==0).all())
print('CCT,Duv:', cctduv1[0,0], cctduv1[0,1])

lut3a = lx._CCT_LUT['ohno2014']['luts']['Yuv60']['1931_2'][((1000.0,41000,0.25,'%'),)]
lut3b = [lx.utils.getdata('./tmp_data/ohno2014_cctlut_tm30_calculator2.0.csv'),{'f_corr': 0.99991, 'ignore_f_corr_is_None': False}]
lut3c = lut3b[0]
cctduv3a = lx.xyz_to_cct(xyz2,mode='ohno2014',cieobs=cmf2,force_tolerance=force_tolerance,out='[cct,duv]',lut = lut3a)
cctduv3b = lx.xyz_to_cct(xyz2,mode='ohno2014',cieobs=cmf2,force_tolerance=force_tolerance,out='[cct,duv]',lut = lut3b)
cctduv3c = lx.xyz_to_cct(xyz2,mode='ohno2014',cieobs=cmf2,force_tolerance=force_tolerance,out='[cct,duv]',lut = lut3c)
print('cctduv3a == cctduv3b ?:', ((cctduv3a-cctduv3b)==0).all())
print('cctduv3a - cctduv3b = ', ((cctduv3a-cctduv3b)))
print('cctduv3b == cctduv3c ?:', ((cctduv3b-cctduv3c)==0).all())
print('cctduv3b - cctduv3c = ', ((cctduv3b-cctduv3c)))

print('cctduv1 - cctduv_calculator = ', ((cctduv1-cctduv_calculator)))
print('cctduv2 - cctduv_calculator = ', ((cctduv2-cctduv_calculator)))
print('cctduv3a - cctduv_calculator = ', ((cctduv3a-cctduv_calculator)))
print('cctduv3b - cctduv_calculator = ', ((cctduv3b-cctduv_calculator)))
print('cctduv3c - cctduv_calculator = ', ((cctduv3c-cctduv_calculator)))

lut4 = generate_lut_bf(start = 1000, end = 20000, interval = 0.25, unit = '%', wl = [360,830,5], cmfs = "1931_2")
#lut4 = [lut4,{'f_corr': 0.99991, 'ignore_f_corr_is_None': False}]
cctduv4 = lx.xyz_to_cct(xyz2,mode='ohno2014',cieobs="1931_2",force_tolerance=force_tolerance,out='[cct,duv]',lut = lut4)
print('\ncctduv4 - cctduv_calculator = ', ((cctduv4-cctduv_calculator)))
print('lut4 - lut3b = ', np.max(lut4[:-1]-lut3b[0],0))
print('\nA wavelength range from 360-830 with a 5 nm interval,')
print('gives orders better agreement between LUT from the ')
print('calculator and the one generated!')


cctduv = xyz_to_cct_bruteforce(xyz1, wl = None, cmfs = "1931_2", n_max = 100, down_sampling_factor = 2,
                                start = 1000, end = 20000, interval = 0.25, unit = '%')

cctduv_r = lx.xyz_to_cct(xyz1, mode = 'robertson1968', cieobs = "1931_2", wl = None, out = '[cct,duv]')
print('cctduv - cctduv_r = ', ((cctduv-cctduv_r)))