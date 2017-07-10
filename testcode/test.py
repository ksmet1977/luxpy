# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 09:55:16 2017

@author: kevin.smet
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 17:49:29 2017

@author: kevin.smet
"""

import os
import pandas as pd
import numpy as np
import luxpy as lx

datapd = pd.read_csv(os.getcwd()+'/luxpy/data/cmfs/ciexyz_1931_2.dat',names=None,index_col=0,header = None,sep = ',')

datapd = pd.read_csv(os.getcwd()+'/luxpy/data/spds/D65.dat',names=None,index_col=0,header = None,sep = ',')

D65 = lx.getdata(os.getcwd()+'/luxpy/data/spds/D65.dat',index='wl')
D65_2 = lx.ajoin((D65,D65[1,None]),0)
D65_n = lx.ajoin((D65,np.repeat(D65[1,None],5000-1,axis=0)),0)

_R_dir = lx._pckg_dir + lx._sep + 'data'+ lx._sep + 'rfls' + lx._sep #folder with rfl data

_cri2012 = {'HL17' : lx.getdata(_R_dir + 'CRI2012_HL17.dat',kind='np',index = 'wl')}
_iesrf = {'AD99' : lx.getdata(_R_dir + 'IESTM30_R99.dat',kind='np',index = 'wl')}
HL17 = _cri2012['HL17']
AD99 = _iesrf['AD99']

