# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:42:08 2020

@author: u0032318
"""
from luxpy.utils import np2d

import luxpy as lx
import numpy as np

cieobs = '1964_10'

# Set number of primaries and target chromaticity:
nprim = 4
target = np.array([[200,1/3,1/3]]) 

# define function that calculates several objectives at the same time (for speed):
def spd_to_cris(spd):
    Rf,Rg = lx.cri.spd_to_cri(spd, cri_type='ies-tm30',out='Rf,Rg')
    return np.vstack((Rf, Rg))     

from luxpy.toolboxes import spdbuild as spb
minimizer = spb.Minimizer(method='nelder-mead')
prim_constructor = spb.PrimConstructor(pdefs={'fwhm':[15],'peakwl_bnds':[400,700],'fwhm_bnds':[5,300]})
so1 = spb.SpectralOptimizer(target = np2d([100,1/3,1/3]), 
                            tar_type = 'Yxy', 
                            cspace_bwtf = {},
                            nprim = nprim, 
                            wlr = [360,830,1], 
                            cieobs = cieobs, 
                            out = 'spds,primss,Ms,results',
                            optimizer_type = '3mixer', 
                            triangle_strengths_bnds = None,
                            prim_constructor = prim_constructor, 
                            prims = None,
                            obj_fcn = spb.ObjFcns(f=[(spd_to_cris,'Rf','Rg'),lx.cri.spd_to_ciera], 
                                                  ft = [(80,90),85],
                                                  decimals = [(2,1),2],
                                                  fw = [(1,1),1]),
                            verbosity = 0)
# start optimization:
spd,M = so1.start(out = 'spds,Ms')
        
Rf, Rg = spd_to_cris(spd)
Ra = lx.cri.spd_to_ciera(spd)
print('obj_fcn1:',Rf)
print('obj_fcn2:',Rg)
print('obj_fcn3:',Ra)