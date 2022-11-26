# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 08:44:17 2022

@author: u0032318
"""

import luxpy as lx
import copy

spd = lx._CIE_D65 
cri_type = copy.deepcopy(lx.cri._CRI_DEFAULTS['iesrf-tm30-18'])

Rf0 = lx.cri.spd_to_cri(spd)
print('-0- Rf = {:1.2f}\n'.format(Rf0[0,0]))

Rf1 = lx.cri.spd_to_cri(spd, cri_type = cri_type)
print('-1- Rf = {:1.2f}\n'.format(Rf1[0,0]))

Rf2 = lx.cri.spd_to_cri(spd, cri_type = cri_type, ref_type = 'BB')
print('-2- Rf = {:1.2f}\n'.format(Rf2[0,0]))

cri_type['ref_type'] = 'BB'
Rf3 = lx.cri.spd_to_cri(spd, cri_type = cri_type)
print('-3- Rf = {:1.2f}\n'.format(Rf3[0,0]))


