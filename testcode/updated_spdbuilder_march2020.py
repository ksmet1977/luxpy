# -*- coding: utf-8 -*-
"""
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

import luxpy as lx
import numpy as np
from luxpy import cri
from luxpy.toolboxes.spdbuild import spd_optimizer
from luxpy import (plt, SPD, spd_to_xyz, xyz_to_cct)

def spd_to_tm30(spd):
    out = 'Rf,Rg,cct,duv,Rfi,jabt,jabr,Rfhi,Rcshi,Rhshi,cri_type'
    Rf,Rg,cct,duv,Rfi,jabt,jabr,Rfhi,Rcshi,Rhshi,cri_type = lx.cri.spd_to_cri(spd, cri_type = 'iesrf', out = out)
    #return Rf,Rg,cct,duv,Rfi,jabt,jabr,Rfhi,Rcshi,Rhshi
    return Rf[0,0],Rg[0,0]

if __name__ == '__main__':
    
    run_example_1 = False
    run_example_2 = True
    
    cieobs = '1931_2'
    target = 4000 # 4000 K target cct
    tar_type = 'cct'
    peakwl = [450,530,560,610]
    fwhm = [30,35,30,15] 
       
    if run_example_1 == True:
        obj_fcn1 = cri.spd_to_iesrf
        obj_fcn2 = cri.spd_to_iesrg
        obj_fcn = [obj_fcn1, obj_fcn2]
        obj_tar_vals = [90,110]
        obj_fcn_weights = [1,1]
        decimals = [5,5]
        
        N_components = 4 #if not None, spd model parameters (peakwl, fwhm, ...) are optimized
        component_spds = None; #component_spds= {}; # if empty dict, then generate using initialize_spd_model_pars and overwrite with function args: peakwl and fwhm. N_components must match length of either peakwl or fwhm
        allow_nongaussianbased_mono_spds = False
        S3, _ = spd_optimizer(target, tar_type = tar_type, cspace_bwtf = {'cieobs' : cieobs, 'mode' : 'search'},\
                              optimizer_type = '3mixer', N_components = N_components,component_spds = component_spds,\
                              allow_nongaussianbased_mono_spds = allow_nongaussianbased_mono_spds,\
                              peakwl = peakwl, fwhm = fwhm, obj_fcn = obj_fcn, obj_tar_vals = obj_tar_vals,\
                              obj_fcn_weights = obj_fcn_weights, decimals = decimals,\
                              use_piecewise_fcn=False, verbosity = 1)
                              #bw_order=[-2],bw_order_min=-2,bw_order_max=-1.5) # to test use of pure lorentzian mono spds.
        
        # Check output agrees with target:
        xyz = spd_to_xyz(S3, relative = False, cieobs = cieobs)
        cct = xyz_to_cct(xyz, cieobs = cieobs, mode = 'lut')
        Rf = obj_fcn1(S3)
        Rg = obj_fcn2(S3)
        print('\nS3: Optimization results:')
        print("S3: Optim / target cct: {:1.1f} K / {:1.1f} K".format(cct[0,0], target))
        print("S3: Optim / target Rf: {:1.3f} / {:1.3f}".format(Rf[0,0], obj_tar_vals[0]))
        print("S3: Optim / target Rg: {:1.3f} / {:1.3f}".format(Rg[0,0], obj_tar_vals[1]))
        
        #plot spd:
        plt.figure()
        SPD(S3).plot()
        
    if run_example_2 == True:
        # Specify as list of tuples
        obj_fcn = [(spd_to_tm30, 'Rf','Rg')]
        obj_tar_vals = [(90,110)]
        obj_fcn_weights = [(1,1)]
        decimals = [(5,5)]
        
        N_components = 4 #if not None, spd model parameters (peakwl, fwhm, ...) are optimized
        component_spds = None; #component_spds= {}; # if empty dict, then generate using initialize_spd_model_pars and overwrite with function args: peakwl and fwhm. N_components must match length of either peakwl or fwhm
        allow_nongaussianbased_mono_spds = False
        S3, _ = spd_optimizer(target, tar_type = tar_type, cspace_bwtf = {'cieobs' : cieobs, 'mode' : 'search'},\
                              optimizer_type = '3mixer', N_components = N_components,component_spds = component_spds,\
                              allow_nongaussianbased_mono_spds = allow_nongaussianbased_mono_spds,\
                              peakwl = peakwl, fwhm = fwhm, obj_fcn = obj_fcn, obj_tar_vals = obj_tar_vals,\
                              obj_fcn_weights = obj_fcn_weights, decimals = decimals,\
                              use_piecewise_fcn=False, verbosity = 1)
                              #bw_order=[-2],bw_order_min=-2,bw_order_max=-1.5) # to test use of pure lorentzian mono spds.
        
        # Check output agrees with target:
        xyz = spd_to_xyz(S3, relative = False, cieobs = cieobs)
        cct = xyz_to_cct(xyz, cieobs = cieobs, mode = 'lut')
        Rf, Rg = obj_fcn[0][0](S3) 
        print('\nS3: Optimization results:')
        print("S3: Optim / target cct: {:1.1f} K / {:1.1f} K".format(cct[0,0], target))
        print("S3: Optim / target Rf: {:1.3f} / {:1.3f}".format(Rf, obj_tar_vals[0][0]))
        print("S3: Optim / target Rg: {:1.3f} / {:1.3f}".format(Rg, obj_tar_vals[0][1]))
        
        #plot spd:
        plt.figure()
        SPD(S3).plot()
        
        
        spd_optimizer(target, tar_type = tar_type, cspace_bwtf = {'cieobs' : cieobs, 'mode' : 'search'},\
                              optimizer_type = '3mixer', N_components = N_components,component_spds = component_spds,\
                              allow_nongaussianbased_mono_spds = allow_nongaussianbased_mono_spds,\
                              peakwl = peakwl, fwhm = fwhm, obj_fcn = obj_fcn, obj_tar_vals = obj_tar_vals,\
                              obj_fcn_weights = obj_fcn_weights, decimals = decimals,\
                              use_piecewise_fcn=False, verbosity = 1)




