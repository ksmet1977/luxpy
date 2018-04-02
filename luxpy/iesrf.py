# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 10:55:02 2018

@author: kevin.smet
"""

import luxpy as lx
import numpy as np
import matplotlib.pyplot as plt
import colorsys

plot_iestm30_output = True

SPDs = lx._IESTM30['S']['data']

SPD = SPDs[:2]

Nspds = SPD.shape[0] - 1

cri_iestm30_defaults = lx.cri._CRI_DEFAULTS['iesrf']
rg_pars_iesttm30 =cri_iestm30_defaults['rg_pars']

# Calculate metrics:
out = 'Rf,Rg,cct,duv,Rfi,jabt,jabr, Rfhi,Rcshi,Rhshi'
spd_to_iestm30 = lambda x: lx.cri.spd_to_cri(x, cri_type = cri_iestm30_defaults, out = out)
Rf,Rg,cct,duv,Rfi,jabt,jabr,Rfhi,Rcshi,Rhshi = spd_to_iestm30(SPD)

#______________________________________________________________________________
# Plot IES TM30 output:
if plot_iestm30_output == True:
    
    # Get normalized and sliced data for plotting:
    nhbins, normalize_gamut, normalized_chroma_ref, start_hue = [rg_pars_iesttm30[x] for x in sorted(rg_pars_iesttm30.keys())]
    normalize_gamut = True #(for plotting)
    normalized_chroma_ref = 100; # np.sqrt((jabr[...,1]**2 + jabr[...,2]**2)).mean(axis = 0).mean()
    
    jabt_binned, jabr_binned, jabc_binned = lx.cri.gamut_slicer(jabt,jabr, out = 'jabt,jabr,jabc', nhbins = nhbins, start_hue = start_hue, normalize_gamut = normalize_gamut, normalized_chroma_ref = normalized_chroma_ref, close_gamut = True)


    # Create data dict with CRI info:
    data = {'SPD' : SPD, 'cct' : cct, 'duv' : duv, 'bjabt' : jabt_binned, 'bjabr' : jabr_binned,\
           'Rf' : Rf, 'Rg' : Rg, 'Rfi': Rfi, 'Rfhi' : Rfhi, 'Rcshi' : Rcshi, 'Rhshi' : Rhshi}
 


    # plot ColorVectorGraphic:
    axtype ='polar'
    #rect = [0.1, 0.1, 0.9, 0.9]
    lx.cri.plot_cri_graphics(SPD, hbins = nhbins, axtype = axtype, fig = None, plot_center_lines = False, plot_edge_lines = True, scalef = normalized_chroma_ref*1.2, force_CVG_layout = True)
    #lx.cri.plot_cri_graphics(data, hbins = nhbins, axtype = axtype, fig = ax_CVG, plot_center_lines = False, plot_edge_lines = True, scalef = normalized_chroma_ref*1.2, force_CVG_layout = True)

               
    plt.show()