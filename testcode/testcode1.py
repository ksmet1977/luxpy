# -*- coding: utf-8 -*-
"""

Created on Sat Sep 23 20:06:06 2017

@author: kevin.smet
"""

import luxpy as lx
import numpy as np


spd = np.vstack((lx._CIE_ILLUMINANTS['D65'],lx._CIE_ILLUMINANTS['A'][1],lx._CIE_ILLUMINANTS['F4'][1],lx._CIE_ILLUMINANTS['F5'][1]))
spd = lx._IESTM30["S"]["data"].copy()
HL17 = lx._CRI_RFL["cri2012"]["HL17"].copy()
rfl = HL17
rfl = lx._IESTM30["R"]["99"]["5nm"].copy()
xyz1 = lx.spd_to_xyz(spd,rfl=rfl,cieobs = "1931_2",relative=True)
xyz2 = lx.spd_to_xyz(spd)
D65=lx._CIE_ILLUMINANTS['D65'].copy()
E=lx._CIE_ILLUMINANTS['E'].copy()

spd1 = np.vstack((lx.getwlr([400,700,1]),np.zeros((1,301))))
spd1[1,155] = 1
spd2 = spd1.copy()
spd2[1,195] =1