# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 23:54:02 2020

4.2 SpectralOptimizer: "free" peak wls and "fixed" fwhm + "user-defined minimizer and primary constructor"
The fwhm of all primaries is kept fixed at 15 nm by specifying these default values in that dictionary in addition to the upper- and lower-bounds. 

Follow KS Example on the website
Change to my format like #### Main Code ####

packages
    luxpy 1.5.2     matplotlib 3.1.0    numpy 1.16.4    pandas 0.24.2 

@author: u0125470
"""
import luxpy as lx # package for color science calculations 
import matplotlib.pyplot as plt # package for plotting
import numpy as np # fundamental package for scientific computing 
import luxpy.toolboxes.spdbuild as spb
###########################################
########### Test codes #################
###########################################

###########################################
############ define functions #############
###########################################

# define function that calculates several objectives at the same time (for speed):
def spd_to_cris(spd):
    """
    Call function that calculates ref.illuminant and jabt & jabr only once to obtain Rf & Rg &...
    Return Rf, Rg, flatten (more visualize than Eccentricity) and theta of the fitted ellipse from 16 gamut-normalized hue-angle-bin jab of test luminaire
    """
    Rf,Rg,jabt_binned_norm,jabr_binned_norm \
            = lx.cri.spd_to_cri(spd, 
                                cri_type = 'ies-tm30', 
                                cieobs = {'xyz':'1964_10','cct':'1931_2'},
                                rg_pars = {'nhbins': 16, 'start_hue': 0, 
                                           'normalize_gamut': True, 
                                           'normalized_chroma_ref': 100.0},
                                out = 'Rf,Rg,jabt_binned,jabr_binned',)

    # get ab coordinates to fit ellipse and plot color vector graph
    # by checking shape of jab, from jabt to ab-plane, should use ab_t = jabt_binned_norm[:,:,1:] instead of ab_t = jabt_binned_norm[:,0,1:]
    # spb.Minimizer(method='nelder-mead'): shape of jabt_binned_norm: (17, 1, 3) 
    # spb.Minimizer(method=user_minim) (lx.math.particleswarm.particleswarm): shape of jabt_binned_norm: (17,2,3),(17,5,3), or ... 
    # ab_r = jabr_binned_norm[:,:,1:]  # ab (17*2 array) of normalized reference luminaire
    ab_t = jabt_binned_norm[:,:,1:]  # ab (17*2 array) of normalized test luminaire
#    print('\njabt_binned_norm',jabt_binned_norm)
#    print('\nshape of jabt_binned_norm:', jabt_binned_norm.shape)
#    print('\nab_t',ab_t)
#    print('\nshape of ab_t', ab_t.shape)    
       
    # lx.math.fit_ellipse(ab_t) only accept 2-dimension (xy)
    # thus, add this part for for multi-spectrum (several spd - such as particleswom minimizer).  refer to lx.cri.jab_to_rg
    Rmax = np.zeros((1,ab_t.shape[1]))
    Rmin = np.zeros((1,ab_t.shape[1]))
    xc = np.zeros((1,ab_t.shape[1]))
    yc = np.zeros((1,ab_t.shape[1]))
    theta_rad = np.zeros((1,ab_t.shape[1]))
    
    for ii in range(ab_t.shape[1]):       
        # ellipse properties for testtest luminaire
        #ellipse_t = lx.math.fit_ellipse(ab_t) # Rmax,Rmin, xc,yc, theta of test spd
        Rmax[:,ii],Rmin[:,ii], xc[:,ii],yc[:,ii], theta_rad[:,ii] = lx.math.fit_ellipse(ab_t[:,ii,:])
        '''
        # test fit.ellispe to see two starting points (17 points, so p[0]=p[-1])matters or not
        xy = np.array([[2.5,0],
                       [1,1],
                       [0,1],
                       [-2.5,0],
                       [-1,-1.0],
                       [0,-1.0]])
        TBD = lx.math.fit_ellipse(xy)
        # array([2.56116729, 0.99624567, 0.        , 0.        , 0.09411075])
        xy = np.array([[2.5,0],
                       [0,1],
                       [-2.5,0],
                       [0,-1],
                       [1,1],
                       [-1,-1.0],
                       [2.5,0]])
        TBD = lx.math.fit_ellipse(xy)
        # array([ 2.56116729e+00,  9.96245666e-01,  1.20624478e-18, -0.00000000e+00,        9.41107527e-02])
        xy = np.array([[2.5,0],
                       [0,1],
                       [-2.5,0],
                       [0,-1],
                       [1,1],
                       [-1,-1.0],
                       [2.5,0],
                       [2.5,0],
                       [2.5,0],
                       [2.5,0]])
        TBD = lx.math.fit_ellipse(xy)
        # array([ 2.56116729e+00,  9.96245666e-01, -2.18852144e-17,  9.64469605e-34,        9.41107527e-02])
        # seems doesn't matters
        '''

    theta_deg = 180 * theta_rad / np.pi
    
    # Flattening of ellipse
    flat = 1 - Rmin/Rmax

#    # Eccentricity of ellipse
#    ecc = np.sqrt(1- Rmin**2/Rmax**2 )
#    print('\ncheck type\n type Rf:', type(Rf),'type Rg:',type(Rg),' type flat:', type(flat),'type theta:',type(theta_deg))
#    print('\ncheck dim\n Rf shape:', Rf.shape,'Rg shape:',Rg.shape,' flat shape:', flat.shape,'theta shape:',theta_deg.shape)    

    return np.vstack((Rf, Rg, flat, theta_deg)) #scalar outputs for each objective value


# Create a minimization function with the specified interface:
def user_minim(fitnessfcn, npars, args, bounds, verbosity = 1,**opts):
    results = lx.math.particleswarm.particleswarm(fitnessfcn, npars, args = args, 
                                                 bounds = bounds, 
                                                 iters = 100, n_particles = 10, ftol = -np.inf,
                                                 options = {'c1': 0.5, 'c2': 0.3, 'w':0.9},
                                                 verbosity = verbosity)
    # Note that there is already a key 'x_final' in results
    return results


###########################################
############## define values ##############
###########################################

# Set CIE CMF set, number of primaries n and target chromaticity:
cieobs = '1964_10'
nprim = 4
target = np.array([[200,1/3,1/3]]) 

obj_fcn = [(spd_to_cris,'Rf','Rg','flat', 'theta_deg')]
obj_tar_vals = [(90,110,0,0)]
obj_weight_vals = [(1,1,1,0)]

###########################################
######## Main Code: target loop ###########
###########################################

so2 = spb.SpectralOptimizer(target = target, tar_type = 'Yxy', cspace_bwtf = {},
                        nprim = nprim, wlr = [360,830,1], cieobs = cieobs, 
                        out = 'spds,primss,Ms,results',
                        optimizer_type = '3mixer', triangle_strengths_bnds = None,
                        prim_constructor = spb.PrimConstructor(pdefs = {'fwhm': [15],
                                                                        'peakwl_bnds':[400,700]}) , 
                        obj_fcn = spb.ObjFcns(f=obj_fcn, ft = obj_tar_vals),
                        minimizer = spb.Minimizer(method=user_minim),
                        verbosity = 0)

# start optimization:
S,prim,M = so2.start(out = 'spds,primss,Ms')
props = lx.detect_peakwl(prim,n = 1,verbosity = 1)
print('prim0_fwhm',props[0]['fwhms'])
print('prim1_fwhm',props[1]['fwhms'])
print('prim2_fwhm',props[2]['fwhms'])
print('prim3_fwhm',props[3]['fwhms'])

# Check output agrees with target:
xyz = lx.spd_to_xyz(S, relative = False, cieobs = cieobs)
Yxy = lx.xyz_to_Yxy(xyz)
cct,duv = lx.xyz_to_cct(xyz, cieobs = cieobs, mode = 'search',out = 'cct,duv')
Rf, Rg, flat, theta_deg = spd_to_cris(S)
print('\nResults (optim,target):')
print("Yxy: ([{:1.0f},{:1.2f},{:1.2f}],[{:1.0f},{:1.2f},{:1.2f}])".format(Yxy[0,0],Yxy[0,1],Yxy[0,2],target[0,0],target[0,1],target[0,2]))
print("Rf: ({:1.2f},{:1.2f})".format(Rf[0], obj_tar_vals[0][0]))
print("Rg: ({:1.2f}, {:1.2f})".format(Rg[0], obj_tar_vals[0][1]))
print("cct(K), duv: ({:1.1f},{:1.4f})".format(cct[0,0], duv[0,0]))
print('\nFlux ratios of component spectra:', M)

#plot spd:
plt.figure()
lx.SPD(S).plot()
plt.figure()
lx.SPD(prim).plot()
