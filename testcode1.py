# -*- coding: utf-8 -*-
"""

Created on Sat Sep 23 20:06:06 2017

@author: kevin.smet
"""

import luxpy as lx
import numpy as np


spd = np.vstack((lx._CIE_ILLUMINANTS['D65'],lx._CIE_ILLUMINANTS['A'][1],lx._CIE_ILLUMINANTS['F4'][1],lx._CIE_ILLUMINANTS['F5'][1]))
spd = lx._IESTM30["S"]["data"]
HL17 = lx._CRI_RFL["cri2012"]["HL17"]
rfl = HL17
rfl = lx._IESTM30["R"]["99"]["5nm"]
xyz1 = lx.spd_to_xyz(spd,rfl=rfl,cieobs = "1931_2",relative=True)
xyz2 = lx.spd_to_xyz(spd)
D65=lx._CIE_ILLUMINANTS['D65']
E=lx._CIE_ILLUMINANTS['E']
