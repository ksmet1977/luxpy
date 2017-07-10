# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:42:53 2017

@author: kevin.smet
"""

import numpy as np
import luxpy as lx

cieobs = '1931_2'

xyzD65 = lx.spd_to_xyz(lx._cie_illuminants['D65'],cieobs = cieobs)
xyzE = np.array([100,100,100])
xyz = np.vstack((xyzD65,xyzE))


mode = 'lut'

cct,duv = lx.cct.xyz_to_cct(xyz,cieobs = cieobs, out = 'cct,duv',mode = mode)
print('CCT:')
print(cct)
print('DUV:')
print(duv)

xyz2 = lx.cct.cct_to_xyz(cct,duv=duv,cieobs = cieobs, mode = mode)

print('XYZ2')
print(xyz2)