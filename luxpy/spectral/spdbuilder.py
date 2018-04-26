# -*- coding: utf-8 -*-
"""
###############################################################################
# Module for building SPDs
###############################################################################

# gaussian_spd(): Generate Gaussian spectrum.

# mono_led_spd(): Generate monochromatic LED spectrum based on Ohno (Opt. Eng. 2005).

# phosphor_led_spd(): Generate phosphor LED spectrum with up to 2 phosphors based on Smet (Opt. Expr. 2011).

# spd_builder(): Build spectrum based on Gaussians, monochromatic and/or phophor LED spectra.

# color3mixer(): Calculate fluxes required to obtain a target chromaticity 
                    when (additively) mixing 3 light sources.
                    
# get_w_summed_spd(): Calculate weighted sum of spds.
 
# fitnessfcn(): Fitness function that calculates closeness of solution x to 
                target values for specified objective functions.
         
# component_triangle_optimizer(): Optimizes the weights (fluxes) of a set of component spectra using a triangle method.

# spd_optimizer(): Generate a spectrum with specified white point and optimized for certain objective functions 
                    from a set of component spectra or component spectrum model parameters.
                    
#------------------------------------------------------------------------------
Created on Wed Apr 25 09:07:04 2018

@author: kevin.smet
"""
from luxpy import np, warnings, minimize, math, _WL3, _CIEOBS,  np2d, getwlr, SPD, plt, spd_to_xyz, xyz_to_Yxy, colortf, xyz_to_cct
from luxpy.cri import spd_to_iesrf, spd_to_iesrg
import itertools


#------------------------------------------------------------------------------
def gaussian_spd(peakwl = 530, fwhm = 20, wl = _WL3, with_wl = True):
    """
    Generate Gaussian spectrum.
    
    Args:
        :peakw: int or float or list or numpy.ndarray, optional
            Peak wavelength
        :fwhm: int or float or list or numpy.ndarray, optional
            Full-Width-Half-Maximum of gaussian.
        :wl: _WL3, optional 
            Wavelength range.
        :with_wl: True, optional
            True outputs a numpy.ndarray with first row wavelengths.
    
    Returns:
        :returns: numpy.ndarray with spectra.        
    """
    wl = np.atleast_2d(getwlr(wl)).T # create wavelength range
    spd = np.exp(-0.5*((wl-np.atleast_2d(peakwl))/np.atleast_2d(fwhm))**2)
    if with_wl == True:
        spd = np.vstack((wl, spd))
    return spd.T

#------------------------------------------------------------------------------
def mono_led_spd(peakwl = 530, fwhm = 20, wl = _WL3, with_wl = True, strength_shoulder = 2):
    """
    Generate monochromatic LED spectrum based on Ohno (Opt. Eng. 2005).
    
    mono_led_spd = (gaussian() + strength_shoulder*gaussian()**5)/(1+strength_shoulder)
    
    Args:
        :peakw: int or float or list or numpy.ndarray, optional
            Peak wavelength
        :fwhm: int or float or list or numpy.ndarray, optional
            Full-Width-Half-Maximum of gaussian used to simulate led.
        :wl: _WL3, optional 
            Wavelength range.
        :with_wl: True, optional
            True outputs a numpy.ndarray with first row wavelengths.
        :strength_shoulder: 2, optional
            Determines the strength of the spectrum shoulders of the monochromatic led.
    
    Returns:
        :returns: numpy.ndarray with spectra.   
    
    Reference:
        Ohno Y (2005). Spectral design considerations for white LED color rendering. Opt. Eng. 44, 111302.
    """
    g = gaussian_spd(peakwl = peakwl, fwhm = fwhm, wl = wl, with_wl = False)
    spd = (g + np.atleast_2d(strength_shoulder)*g**5)/(1+np.atleast_2d(strength_shoulder))
    if with_wl == True:
        spd = np.vstack((getwlr(wl), spd))
    return spd

#------------------------------------------------------------------------------
def phophor_led_spd(peakwl = 450, fwhm = 20, wl = _WL3, with_wl = True, strength_shoulder = 2,\
                    strength_ph = 0, peakwl_ph1 = 530, fwhm_ph1 = 80, strength_ph1 = 1,\
                    peakwl_ph2 = 560, fwhm_ph2 = 80, strength_ph2 = None,\
                    use_piecewise_fcn = True,\
                    verbosity = 0, out = 'spd'):
    """
    Generate phosphor LED spectrum with up to 2 phosphors based on Smet (Opt. Expr. 2011).
    
    Model:
        1) If strength_ph2 is not None:
              phosphor_spd = (strength_ph1*mono_led_spd(peakwl_ph1, ..., strength_shoulder = 1) 
                           + strength_ph2)*mono_led_spd(peakwl_ph2, ..., strength_shoulder = 1)) 
                            / (strength_ph1 + strength_ph2)
          else:
              phosphor_spd = (strength_ph1*mono_led_spd(peakwl_ph1, ..., strength_shoulder = 1) 
                           + (1-strength_ph1)*mono_led_spd(peakwl_ph2, ..., strength_shoulder = 1)) 

        2) S = (mono_led_spd() + strength_ph*(phosphor_spd/phosphor_spd.max()))/(1 + strength_ph)
        
        3) piecewise_fcn = S for wl < peakwl and 1 for wl >= peakwl
        
        4) phosphor_led_spd = S*piecewise_fcn 
            
    Args:
        :peakw: int or float or list or numpy.ndarray, optional
            Peak wavelengths of the monochromatic led.
        :fwhm: int or float or list or numpy.ndarray, optional
            Full-Width-Half-Maximum of gaussian.
        :wl: _WL3, optional 
            Wavelength range.
        :with_wl: True, optional
            True outputs a numpy.ndarray with first row wavelengths.
        :strength_shoulder: 2, optional
            Determines the strength of the spectrum shoulders of the monochromatic led.
        :strength_ph: 0, optional
            Total contribution of phosphors in mixture.
        :peakwl_ph1: int or float or list or numpy.ndarray, optional
            Peak wavelength of the first phosphor.
        :fwhm_ph1: int or float or list or numpy.ndarray, optional
            Full-Width-Half-Maximum of gaussian used to simulate first phosphor.
        :strength_ph1: 1, optional
            Strength of first phosphor in phosphor mixture. 
            If :strength_ph2: is None: value should be in the [0,1] range.
        :peakwl_ph2: int or float or list or numpy.ndarray, optional
            Peak wavelength of the second phosphor.
        :fwhm_ph2: int or float or list or numpy.ndarray, optional
            Full-Width-Half-Maximum of gaussian used to simulate second phosphor.
        :strength_ph2: None, optional
            Strength of second phosphor in phosphor mixture. 
            If None: strength is calculated as (1-:strength_ph1:)
                :target: np2d([100,1/3,1/3]), optional
            Numpy.ndarray with Yxy chromaticity of target.
        :verbosity: 0, optional
            If > 0: plots spectrum components (mono_led, ph1, ph2, ...)
        :out: 'spd', optional
            Specifies output.
            
    Returns:
        :returns: spd, component_spds
            numpy.ndarrays with spectra (and component spds used to build the final spectra) 
        
        
    References:
        1. Ohno Y (2005). Spectral design considerations for white LED color rendering. Opt. Eng. 44, 111302.

        2. Smet K, Ryckaert WR, Pointer MR, Deconinck G, and Hanselaer P (2011). 
            Optimal colour quality of LED clusters based on memory colours. 
            Opt. Express 19, 6903–6912.
    """
        
    
    mono_led = mono_led_spd(peakwl = peakwl, fwhm = fwhm, wl = wl, with_wl = False, strength_shoulder = strength_shoulder)
    wl = getwlr(wl)
    if strength_ph is not None:
        strength_ph = np.atleast_2d(strength_ph)
        if ((strength_ph > 0).any()): # Use phophor type led for obtaining target:
            ph1 = mono_led_spd(peakwl = peakwl_ph1, fwhm = fwhm_ph1, wl = wl, with_wl = False, strength_shoulder = 1)
            ph2 = mono_led_spd(peakwl = peakwl_ph2, fwhm = fwhm_ph2, wl = wl, with_wl = False, strength_shoulder = 1)
            component_spds = np.dstack((mono_led,ph1,ph2))
           
            if ('spd' in out.split(',')):
                strength_ph1 = np.atleast_2d(strength_ph1)
                strength_ph2 = np.atleast_2d(strength_ph2)
                if ((ph1 is not None) & (ph2 is not None)):
                    if (strength_ph2[0,0] is not None):
                        phosphors = (strength_ph1*ph1.T + strength_ph2*ph2.T).T/(strength_ph1 + strength_ph1)
                    else:
                        phosphors = (strength_ph1*ph1.T + (1-strength_ph1)*ph2.T).T
                    strength_ph = np.atleast_1d(strength_ph)
                    phosphors = phosphors/phosphors.max(axis = 1, keepdims = True)
                    spd = mono_led + (strength_ph*phosphors.T).T
                else:
                    phosphors = None
                    spd = mono_led.copy()
                
        else: # Only monochromatic leds:
            ph1 = None
            ph2 = None
            phosphors = None
            spd = mono_led.copy()
            component_spds = mono_led[:,:,None].T.copy()
    
    else: # Only monochromatic leds:
        ph1 = None
        ph2 = None
        phosphors = None
        spd = mono_led.copy()
        component_spds = mono_led[:,:,None].T.copy()
        
    
    if (use_piecewise_fcn == True):
        peakwl = np.atleast_1d(peakwl)
        if ('component_spds' in out.split(',')):
            fp = component_spds.copy()
            for i in range(fp.shape[-1]):
                fp[:,np.where(wl >= peakwl[i]),i] = 1
                component_spds[...,i] = component_spds[...,i]*fp[...,i] # multiplication with piecewise function f'
        if ('spd' in out.split(',')):
            fp = mono_led.copy()
            for i in range(fp.shape[0]):
                fp[i,np.where(wl >= peakwl[i])] = 1
                spd[i] = spd[i]*fp[i] # multiplication with piecewise function f'
    
    # Normalize to max = 1:
    spd = spd/spd.max(axis = 1, keepdims = True)
    component_spds = component_spds/component_spds.max(axis=1,keepdims=True)

    if verbosity > 0:
        mono_led_str = 'Mono_led_1'
        ph1_str = 'Phosphor_1'
        ph2_str = 'Phosphor_2'
        for i in range(spd.shape[0]):
            plt.figure()
            if ph1 is not None:
                plt.plot(wl,mono_led[i].T,'b--', label = mono_led_str)
                plt.plot(wl,ph1[i].T,'g:', label = ph1_str)
                plt.plot(wl,ph2[i].T,'y:', label = ph2_str)
                if phosphors is not None:
                    plt.plot(wl,phosphors[i].T,'r--', label = 'Ph1,2 combined')
            plt.plot(wl,spd[i].T,'k-', label = 'Output spd')
            plt.xlabel('Wavelengths (nm)')
            plt.ylabel('Normalized spectral intensity (max = 1)')
            plt.legend()
            plt.show()

    if (with_wl == True):
        spd = np.vstack((wl, spd))
        component_spds = np.vstack((wl, component_spds))

    if out == 'spd':
        return spd
    elif out == 'component_spds':
        return component_spds
    elif out == 'spd,component_spds':
        return spd, component_spds


#------------------------------------------------------------------------------
def spd_builder(flux = None, peakwl = 450, fwhm = 20, wl = _WL3, with_wl = True, strength_shoulder = 2,\
                    strength_ph = 0, peakwl_ph1 = 530, fwhm_ph1 = 80, strength_ph1 = 1,\
                    peakwl_ph2 = 560, fwhm_ph2 = 80, strength_ph2 = None,\
                    target = None, tar_type = 'Yuv', cspace_bwtf = {}, cieobs = _CIEOBS,\
                    use_piecewise_fcn = True, verbosity = 1, out = 'spd'):
    """
    Build spectrum based on Gaussians, monochromatic and/or phophor LED spectra.
           
    Args:
        :flux: None, optional
            Fluxes of each of the component spectra.
            None outputs the individual component spectra.
        :peakw: int or float or list or numpy.ndarray, optional
            Peak wavelengths of the monochromatic leds.
        :fwhm: int or float or list or numpy.ndarray, optional
            Full-Width-Half-Maximum of gaussians.
        :wl: _WL3, optional 
            Wavelength range.
        :with_wl: True, optional
            True outputs a numpy.ndarray with first row wavelengths.
        :strength_shoulder: 2, optional
            Determines the strength of the spectrum shoulders of the monochromatic leds.
        :strength_ph: 0, optional
            Total contribution of phosphors in mixtures.
        :peakwl_ph1: int or float or list or numpy.ndarray, optional
            Peak wavelength of the first phosphors.
        :fwhm_ph1: int or float or list or numpy.ndarray, optional
            Full-Width-Half-Maximum of gaussian used to simulate first phosphors.
        :strength_ph1: 1, optional
            Strength of first phosphor in phosphor mixtures. 
            If :strength_ph2: is None: value should be in the [0,1] range.
        :peakwl_ph2: int or float or list or numpy.ndarray, optional
            Peak wavelength of the second phosphors.
        :fwhm_ph2: int or float or list or numpy.ndarray, optional
            Full-Width-Half-Maximum of gaussian used to simulate second phosphors.
        :strength_ph2: None, optional
            Strength of second phosphor in phosphor mixtures. 
            If None: strength is calculated as (1-:strength_ph1:)
        
        :target: None, optional
            Numpy.ndarray with Yxy chromaticity of target.
            If None: don't override phosphor strengths, else calculate strength to obtain :target:
        :tar_type:  'Yxy' or str, optional
            Specifies the input type in :target: (e.g. 'Yxy' or 'cct')
        :cieobs: _CIEOBS, optional
            CIE CMF set used to calculate chromaticity values.
        :cspace_bwtf: {}, optional
            Backward (..._to_xyz) transform parameters (see colortf()) to go from :tar_type: to 'Yxy'.

        :verbosity: 0, optional
            If > 0: plots spectrum components (mono_led, ph1, ph2, ...)
        :out: 'spd', optional
            Specifies output.
            
    Returns:
        :returns: numpy.ndarray with spectra.  
    
    Note:
        1. Target-optimization is only for phophor_leds with three components 
            (blue pump, ph1 and ph2) spanning a sufficiently large gamut. 
        
    Reference:
        1. Ohno Y (2005). Spectral design considerations for white LED color rendering. Opt. Eng. 44, 111302.

        2. Smet K, Ryckaert WR, Pointer MR, Deconinck G, and Hanselaer P (2011). 
            Optimal colour quality of LED clusters based on memory colours. 
            Opt. Express 19, 6903–6912.
    """
    
    spd, component_spds = phophor_led_spd(peakwl = peakwl, fwhm = fwhm, wl = wl, with_wl = False, strength_shoulder = strength_shoulder,\
                                           strength_ph = strength_ph, peakwl_ph1 = peakwl_ph1, fwhm_ph1 = fwhm_ph1, strength_ph1 = strength_ph1,\
                                           peakwl_ph2 = peakwl_ph2, fwhm_ph2 = fwhm_ph2, strength_ph2 = strength_ph2,\
                                           use_piecewise_fcn = use_piecewise_fcn, verbosity = 0, out = 'spd,component_spds')
    

    wl = getwlr(wl)
    if target is not None: 
        # use component_spectra to build spds with target chromaticity
        # (ignores strength_ph values).
        N = np.int(component_spds.shape[0]) # rgb components are grouped 

        if component_spds.shape[-1] < 3:
            raise Exception('spd_builder(): Not enough component spectra for color3mixer(). Min. is 3')
        
        component_spds_2d = np.vstack((wl,component_spds[...,0],component_spds[...,1],component_spds[...,2]))


        # Calculate xyz of components:
        xyzi = spd_to_xyz(component_spds_2d, relative = False, cieobs = cieobs)

        # Calculate Yxy:
        Yxyt = colortf(target, tf = tar_type+'>Yxy', bwtf = cspace_bwtf)
        Yxyi = xyz_to_Yxy(xyzi) #input for color3mixer is Yxy

#        if verbosity > 0:
#            plt.figure()
#            plt.plot(Yxyt[0,1],Yxyt[0,2],'ko')
#            plt.plot(Yxyi[:N,1],Yxyi[:N,2],'bo')
#            plt.plot(Yxyi[N:2*N,1],Yxyi[N:2*N,2],'go')
#            plt.plot(Yxyi[2*N:3*N,1],Yxyi[2*N:3*N,2],'ro')
        
        # Calculate fluxes for obtaining target:
        M3 = color3mixer(Yxyt,Yxyi[:N,:],Yxyi[N:2*N,:],Yxyi[2*N:3*N,:])
        

        # Calculate spectrum:
        spd = math.dot23(M3,component_spds.T)
        spd = np.atleast_2d([spd[i,:,i] for i in range(N)])
        spd = spd/spd.max(axis = 1, keepdims = True)
        
        # Mark out_of_gamut solution with NaN's:
        is_out_of_gamut =  np.where(((M3<0).sum(axis=1))>0)[0]
        spd[is_out_of_gamut,:] = np.nan
        M3[is_out_of_gamut,:] = np.nan
        if verbosity > 0:
            if is_out_of_gamut.sum()>0:
                warnings.warn("spd_builder(): At least one solution is out of gamut. Check for NaN's in spd.")

    else:
        component_spds = component_spds.T
        
            

    if verbosity > 0:
        for i in range(spd.shape[0]):
            plt.figure()
            plt.plot(wl,component_spds[i,:,0],'b--', label = 'Component 1')
            if (strength_ph is not None) & (strength_ph is not 0):
                plt.plot(wl,component_spds[i,:,1],'g:', label = 'Component 2')
                plt.plot(wl,component_spds[i,:,2],'y:', label = 'Component 3')
            plt.plot(wl,spd[i],'k-', label = 'Output spd')
            plt.xlabel('Wavelengths (nm)')
            plt.ylabel('Normalized spectral intensity (max = 1)')
            plt.legend()
            plt.show()
    
    if (flux is not None):
        flux = np.atleast_2d(flux)
        if (flux.shape[1] == spd.shape[0]):
            spd_is_not_nan = np.where(np.isnan(spd[:,0])==False)[0] #keep only not nan spds
            spd = np.dot(flux[:,spd_is_not_nan],spd[spd_is_not_nan,:])
    spd = spd/spd.max(axis=1,keepdims= True)
    
    if with_wl == True:
        spd = np.vstack((wl, spd))
    return spd




   
#------------------------------------------------------------------------------
def color3mixer(Yxyt,Yxy1,Yxy2,Yxy3):
    """
    Calculate fluxes required to obtain a target chromaticity 
    when (additively) mixing 3 light sources.
    
    Args:
        :Yxyt: numpy.ndarray with target Yxy chromaticities.
        :Yxy1: numpy.ndarray with Yxy chromaticities of light sources 1.
        :Yxy2: numpy.ndarray with Yxy chromaticities of light sources 2.
        :Yxy3: numpy.ndarray with Yxy chromaticities of light sources 3.
        
    Returns:
        :M: numpy.ndarray with fluxes.
        
    Note:
        Yxyt, Yxy1, ... can contain multiple rows, each refering to single mixture.
    """
    Y1 = Yxy1[...,0]
    x1 = Yxy1[...,1]
    y1 = Yxy1[...,2]
    Y2 = Yxy2[...,0]
    x2 = Yxy2[...,1]
    y2 = Yxy2[...,2]
    Y3 = Yxy3[...,0]
    x3 = Yxy3[...,1]
    y3 = Yxy3[...,2]
    Yt = Yxyt[...,0]
    xt = Yxyt[...,1]
    yt = Yxyt[...,2]
    m1 = y1*((xt-x3)*y2-(yt-y3)*x2+x3*yt-xt*y3)/(yt*((x3-x2)*y1+(x2-x1)*y3+(x1-x3)*y2))
    m2 = -y2*((xt-x3)*y1-(yt-y3)*x1+x3*yt-xt*y3)/(yt*((x3-x2)*y1+(x2-x1)*y3+(x1-x3)*y2))
    m3 = y3*((x2-x1)*yt-(y2-y1)*xt+x1*y2-x2*y1)/(yt*((x2-x1)*y3-(y2-y1)*x3+x1*y2-x2*y1))
    M = Yt*np.vstack((m1/Y1,m2/Y2,m3/Y3))
    return M.T


def colormixer(Yxyt = None, Yxyi = None, ratios = None, source_order = None):
    """
    Calculate fluxes required to obtain a target chromaticity 
    when (additively) mixing N light sources.
    
    Args:
        :Yxyt: numpy.ndarray with target Yxy chromaticities.
        :Yxyi: numpy.ndarray with Yxy chromaticities of light sources i.
        :ratios: numpy.ndarray with light source ratio specifications.  
        
    Returns:
        :M: numpy.ndarray with fluxes.
        
    Note:
        Yxyt, Yxyi, ... can contain multiple rows, each refering to single mixture.
    """
    np.set_printoptions(precision=3)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    if Yxyt is None:
        Yxyt = np.atleast_2d([100,1/3,1/3])
    if Yxyi is None:
        n = 9
        Yxyi = np.hstack((np.ones((n,1))*100,np.random.rand(n,2)))
    if ratios is None:
        ratios = np.random.rand(n-3)
        print(ratios.shape)
    if source_order is None:
        source_order = np.arange(n)
    
    n = Yxyi.shape[0]
    if n > 3:
        
               
        Mlut_s = np.arange(n)[:,None]
        Mlut_Yxy = Yxyi
        Mlut_su = np.diagflat(np.arange(n))
        Mlut_m = np.diagflat(2*np.ones((1,n))-1)
        
        Mlut = np.hstack((np.arange(n)[:,None],Yxyi,\
                          np.diagflat(np.arange(n)),\
                          np.diagflat(2*np.ones((1,n))-1)))
        
        for k in range(n-3):
            print('k:')
            print(k)
            
            # Create current state:
            if k == 0:
                Yxyi_k = Yxyi.copy()
                ratios_k = ratios.copy()
                source_order_k = source_order.copy()
                firstrun = True
                
            # Combine last two sources of current state:
            ratioAB = ratios_k[k]
            pA = np.int(source_order_k[2*k])
            pB = np.int(source_order_k[2*k+1])
            print(pA)
            print(pB)
            
            YxyA = Yxyi_k[pA,:]
            YxyB = Yxyi_k[pB,:]
    
            YA = YxyA[0]
            xA = YxyA[1]
            yA = YxyA[2]
            
            YB = YxyB[0]
            xB = YxyB[1]
            yB = YxyB[2]
            
            XA = xA * YA / yA
            XB = xB * YB / yB
            ZA = (1 - xA - yA) * YA / yA
            ZB = (1 - xB - yB) * YB / yB

            XM = (ratioAB * XA + (1 - ratioAB) * XB)
            ZM = (ratioAB * ZA + (1 - ratioAB) * ZB)
            YM = (ratioAB * YA + (1 - ratioAB) * YB)
            xM = XM / (XM + YM + ZM)
            yM = YM / (XM + YM + ZM)
            YxyM = np.hstack((YM, xM, yM))
        
            #calculate the contributions of source 1 and source 2 needed to get the M of the temporary source
            MA = ratioAB * YA / YM
            MB = (1 - ratioAB) * YB / YM
            
            # Store in Mlut:
            sources_used_k = np.nan*np.ones((Mlut_s.shape[0]))
            sources_used_k[np.hstack((pA,pB))] = np.hstack((pA,pB))
            M_k = np.zeros((Mlut_s.shape[0]))
            M_k[np.hstack((pA,pB))] = np.hstack((MA,MB))
            
            sol_k = np.hstack((Mlut_s.shape[-1] + 1,YxyM,sources_used_k, M_k))
            Mlut = np.vstack((Mlut, sol_k))
            
            
            #Create new state if k == int(n/2):
            nk = np.int(source_order_k.shape[0]/2)
            print('nk')
            print(nk)
            N = Mlut[-1,0]+1
            if k == nk-1:
                p = np.int((N-nk))
                source_order_k = np.hstack((Mlut[p:,0],source_order_k))
                print(source_order_k)
                Yxyi_k = np.vstack((Mlut[p:,1:4],Yxyi_k))

#                if nk%2 > 0:
#                    if firstrun == True:
#                        p = 0
#                    else:
#                        p = -1
#                        firstrun = False
#                    source_order_k = np.hstack((source_order_k,source_order_k[-1]+1))
#                    Yxyi_k = np.vstack((Yxyi_k,Yxyi_k[p]))
#                    Mlut = np.vstack((Mlut,Mlut[p,:]))
            print(Mlut)
        kk = Mlut[n:,4:4+n]
        print('kk')
        print(kk)
        unused_sources = np.where((1*np.isnan(kk)).sum(axis=0)==kk.shape[0])[0]
        print(unused_sources)
        Yxyi_final = Mlut[np.hstack((unused_sources,n+k-1,n+k)),1:4]
        print(Yxyi_final)
        
        M3 = color3mixer(Yxyt,Yxyi_final[0,:],Yxyi_final[1,:],Yxyi_final[2,:])
        print('M3')
        print(M3)
        M3[M3<0] = np.nan
        print(M3)
        print('Mlut_M')
        print(Mlut[np.hstack((0,n+k-1,n+k)),(4+n):])
        M = np.dot(M3,Mlut[np.hstack((0,n+k-1,n+k)),(4+n):])
        print(M)
        
    else:
        M = color3mixer(Yxyt,Yxyi[0,:],Yxyi[1,:],Yxyi[2,:])
                
    return M

colormixer()

#------------------------------------------------------------------------------
def get_w_summed_spd(w,spds):
    """
    Calculate weighted sum of spds.
    
    Args:
        :w: numpy.ndarray with weigths (e.g. fluxes)
        :spds: numpy.ndarray with component spds.
        
    Returns:
        :returns: numpy.ndarray with weighted sum.
    """
    return np.vstack((spds[:1],np.dot(np.abs(w),spds[1:])))


#------------------------------------------------------------------------------
def fitnessfcn(x, spd_constructor, spd_constructor_pars = None, F_rss = True, decimals = [3], obj_fcn = [None], obj_fcn_pars = [{}], obj_fcn_weights = [1], obj_tar_vals = [0], verbosity = 0, out = 'F'):
    """
    Fitness function that calculates closeness of solution x to target values 
    for specified objective functions.
    
    Args:
        :x: numpy.ndarray with parameter values
        :spd_constructor: function handle to a function that constructs the spd from parameter values in :x:.
        :spd_constructor_pars: None, optional,
            Parameters required by :spd_constructor:
        :F_rss: True, optional
             Take Root-Sum-of-Squares of 'closeness' values between target and objective function values.
        :decimals: 3, optional
            Rounding decimals of objective function values.
        :obj_fcn: [None] or list of function handles to objective functions, optional
        :obj_fcn_weights: [1] or list of weigths for each objective function, optional.
        :obj_fcn_pars: [None] or list of parameter dicts for each objective functions, optional
        :obj_tar_vals: [0] or list of target values for each objective functions, optional
        :verbosity: 0, optional
            If > 0: print intermediate results.
        :out: 'F', optional
            Determines output.
            
    Returns:
        :F: float or numpy.ndarray with fitness value for current solution :x:.
    """
    
    # Keep track of solutions tried:
    global optcounter 
    optcounter = optcounter + 1
    
    # Number of objective functions:
    N = len(obj_fcn)
    
    # Get current spdi:
    spdi = spd_constructor(x,spd_constructor_pars)
    
    # Make decimals and obj_fcn_weights same size as N:
    decimals =  decimals*np.ones((N))
    obj_fcn_weights =  obj_fcn_weights*np.ones((N))
    obj_fcn_pars = np.asarray(obj_fcn_pars*N)
    obj_tar_vals = np.asarray(obj_tar_vals)
    
    # Calculate all objective functions and closeness to target values
    # store squared weighted differences for speed:
    F = np.nan*np.ones((N))
    output_str = 'c{:1.0f}: F = {:1.' + '{:1.0f}'.format(decimals.max()) + 'f}' + ' : '
    obj_vals = F.copy()
    for i in range(N):
        if obj_fcn[i] is not None:
            obj_vals[i] = obj_fcn[i](spdi, **obj_fcn_pars[i])
            
            if obj_tar_vals[i] > 0:
                f_normalize = obj_tar_vals[i]
            else:
                f_normalize = 1
                
            F[i] = (obj_fcn_weights[i]*(np.abs((np.round(obj_vals[i],np.int(decimals[i])) - obj_tar_vals[i])/f_normalize)**2))
            
            if (verbosity > 0):
                output_str = output_str + r' obj_#{:1.0f}'.format(i+1) + ' = {:1.' + '{:1.0f}'.format(np.int(decimals[i])) + 'f},'
        else:
            obj_vals[i] = np.nan
            F[i] = np.nan
  
    # Take Root-Sum-of-Squares of delta((val - tar)**2):
    if F_rss == True:
        F = np.sqrt(np.nansum(F))

    
    # Print intermediate results:
    if (verbosity > 0):
        print(output_str.format(*np.hstack((optcounter, F, obj_vals))))
    
    if out == 'F':
        return F
    elif out == 'obj_vals':
        return obj_vals
    elif out == 'F,obj_vals':
        return F, obj_vals
    elif out == 'spdi,obj_vals':
        return spdi, obj_vals
    else:
        eval(out)
        
                

#------------------------------------------------------------------------------
def component_triangle_optimizer(component_spds, Yxyi = None, Yxy_target = np2d([100,1/3,1/3]), cieobs = _CIEOBS,\
                                 obj_fcn = [None], obj_fcn_pars = [{}], obj_fcn_weights = [1],\
                                 obj_tar_vals = [0], decimals = [5], \
                                 minimize_method = 'nelder-mead', minimize_opts = None, F_rss = True,\
                                 verbosity = 0):
    """
    Optimizes the weights (fluxes) of a set of component spectra using a triangle method.
    
    The triangle method creates for all possible combinations of 3 primary component spectra 
    a spectrum that results in the target chromaticity using color3mixer() 
    and then optimizes the weights of each of the latter spectra such that 
    adding them (additive mixing) results in obj_vals as close as possible to 
    the target values.
    
    Args:
        :component_spds: numpy.ndarray of component spectra.
        :Yxyi:  None or numpy.ndarray, optional
            Yxy chromaticities of all component spectra.
            If None: they are calculated from :component_spds:
        :Yxy_target: np2d([100,1/3,1/3]), optional
            Numpy.ndarray with Yxy chromaticity of target.
        :cieobs: _CIEOBS, optional
            CIE CMF set used to calculate chromaticity values if not provided in :Yxyi:.
        :F_rss: True, optional
             Take Root-Sum-of-Squares of 'closeness' values between target and objective function values.
        :decimals: 5, optional
            Rounding decimals of objective function values.
        :obj_fcn: [None] or list of function handles to objective functions, optional
        :obj_fcn_weights: [1] or list of weigths for each objective function, optional.
        :obj_fcn_pars: [None] or list of parameter dicts for each objective functions, optional
        :obj_tar_vals: [0] or list of target values for each objective functions, optional
        :minimize_method: 'nelder-mead', optional
            Optimization method used by minimize function.
        :minimize_opts: None, optional
             Dict with minimization options. 
             None defaults to: {'xtol': 1e-5, 'disp': True, 'maxiter' : 1000*Nc, 'maxfev' : 1000*Nc,'fatol': 0.01}
        :verbosity: 0, optional
            If > 0: print intermediate results.
            
    Returns:
        :returns: M, spd_opt, obj_vals
            - 'M': numpy.ndarray with fluxes for each component spectrum.
            - 'spd_opt': optimized spectrum.
            - 'obj_vals': values of the objective functions for the optimized spectrum.
    """
    if Yxyi is None: #if not provided: calculate.
        xyzi = spd_to_xyz(spds, relative = False, cieobs = cieobs)
        Yxyi = xyz_to_Yxy(xyzi)

    # Generate all possible 3-channel combinations (component triangles):
    N = component_spds.shape[0]-1
    combos = np.array(list(itertools.combinations(range(N), 3)))   

    # calculate fluxes to obtain target Yxyt:
    M3 = color3mixer(Yxy_target,Yxyi[combos[:,0],:],Yxyi[combos[:,1],:],Yxyi[combos[:,2],:])
        
    # Get rid of out-of-gamut solutions:
    is_not_out_of_gamut =  (((M3>=0).sum(axis=1))==3)
    M3 = M3[is_not_out_of_gamut,:]
    combos = combos[is_not_out_of_gamut,:]
    Nc = combos.shape[0]
    
    # Calculate 3-channel SPDs from individual channels:
    spds_rgb = np.empty((Nc,component_spds.shape[-1]))
    for i in range(Nc):
        spds_rgb[i] = np.dot(M3[i,:],component_spds[combos[i,:]+1])
    spds_rgb = np.vstack((component_spds[:1],spds_rgb))
    
    if Nc > 1:
        # Setup optimization based on weigthed linear combination of 3-channel SPDs:
        global optcounter
        optcounter = 1
        spd_constructor = get_w_summed_spd # define constructor function
        spd_constructor_pars = spds_rgb  # define constructor  parameters
        
        def fit_fcn(x, out, obj_fcn, obj_fcn_pars, obj_fcn_weights, obj_tar_vals, F_rss, decimals, verbosity):
            F = fitnessfcn(x, spd_constructor, spd_constructor_pars = spd_constructor_pars,\
                      F_rss = F_rss, decimals = decimals,\
                      obj_fcn = obj_fcn, obj_fcn_pars = obj_fcn_pars, obj_fcn_weights = obj_fcn_weights,\
                      obj_tar_vals = obj_tar_vals, verbosity = verbosity, out = out)
            return F
    
        # Find good starting point x0 to speed up optimization (play with gamma for best results):
        gamma = 0.01
        X0 = np.diagflat(np.ones((1,combos.shape[0])))
        Fs = np.ones((combos.shape[0],1))
        for i in range(combos.shape[0]):
            Fs[i] = fit_fcn(X0[i], 'F', obj_fcn, obj_fcn_pars, obj_fcn_weights, obj_tar_vals, F_rss, decimals, verbosity)
        x0 = X0[Fs.argmin(),:] + gamma*np.ones((1,combos.shape[0]))
            
        # Perform optimzation:
        if minimize_opts is None:
            minimize_opts = {'xtol': 1e-5, 'disp': True, 'maxiter' : 1000*Nc, 'maxfev' : 1000*Nc,'fatol': 0.01}
        input_par = ('F', obj_fcn, obj_fcn_pars, obj_fcn_weights, obj_tar_vals, F_rss, decimals, verbosity)
        res = minimize(fit_fcn, x0, args = input_par, method = minimize_method, options = minimize_opts)
        x_final = np.abs(res['x'])
    else:
        x_final = M3
        

    # Calulate fluxes of all components from M3 and x_final:
    M_final = x_final[:,None]*M3
    M = np.empty((N))
    for i in range(N):
        M[i] = M_final[np.where(combos == i)].sum()
    
    # Calculate optimized SPD and get obj_vals:
    spd_opt, obj_vals = fit_fcn(x_final, 'spdi,obj_vals', obj_fcn, obj_fcn_pars, obj_fcn_weights, obj_tar_vals, F_rss, decimals, verbosity)
    return M, spd_opt, obj_vals

#------------------------------------------------------------------------------
def spd_optimizer(target, tar_type = 'Yxy', cieobs = _CIEOBS,\
                  optimizer_type = '3mixer', cspace = 'Yuv', cspace_bwtf = {}, cspace_fwtf = {},\
                  component_spds = None, N_components = None,\
                  obj_fcn = [None], obj_fcn_pars = [{}], obj_fcn_weights = [1],\
                  obj_tar_vals = [0], decimals = [5], \
                  minimize_method = 'nelder-mead', minimize_opts = None, F_rss = True,\
                  peakwl = [450,530,600], fwhm = [20,30,10], wl = _WL3, with_wl = True, strength_shoulder = 2,\
                  strength_ph = 0, peakwl_ph1 = 530, fwhm_ph1 = 80, strength_ph1 = 1,\
                  peakwl_ph2 = 560, fwhm_ph2 = 80, strength_ph2 = None,\
                  verbosity = 0):
    """
    Generate a spectrum with specified white point and optimized for certain objective functions 
    from a set of component spectra or component spectrum model parameters.
    
    Args:
        :target: np2d([100,1/3,1/3]), optional
            Numpy.ndarray with Yxy chromaticity of target.
        :tar_type:  'Yxy' or str, optional
            Specifies the input type in :target: (e.g. 'Yxy' or 'cct')
        :cieobs: _CIEOBS, optional
            CIE CMF set used to calculate chromaticity values if not provided in :Yxyi:.
        :optimizer_type: '3mixer',  optional
            Specifies type of chromaticity optimization ('3mixer' or 'mixer' or 'search')
        :cspace: 'Yuv', optional
            Color space for 'search'-type optimization. 
        :cspace_bwtf: {}, optional
            Backward (..._to_xyz) transform parameters (see colortf()) to go from :tar_type: to 'Yxy'.
        :cspace_fwtf = {}, optional
            Forward (xyz_to_...) transform parameters (see colortf()) to go from xyz to :cspace:.
        :component_spds: numpy.ndarray of component spectra.
            If None: they are built from input args.
        :F_rss: True, optional
             Take Root-Sum-of-Squares of 'closeness' values between target and objective function values.
        :decimals: 5, optional
            Rounding decimals of objective function values.
        :obj_fcn: [None] or list of function handles to objective functions, optional
        :obj_fcn_weights: [1] or list of weigths for each objective function, optional.
        :obj_fcn_pars: [None] or list of parameter dicts for each objective functions, optional
        :obj_tar_vals: [0] or list of target values for each objective functions, optional
        :minimize_method: 'nelder-mead', optional
            Optimization method used by minimize function.
        :minimize_opts: None, optional
             Dict with minimization options. 
             None defaults to: {'xtol': 1e-5, 'disp': True, 'maxiter' : 1000*Nc, 'maxfev' : 1000*Nc,'fatol': 0.01}
        :verbosity: 0, optional
            If > 0: print intermediate results.
         
         :peakwl:, :fwhm:, ... : see ?spd_builder for more info.   
            
    Returns:
        :returns: spds, M
            - 'spds': optimized spectrum.
            - 'M': numpy.ndarray with fluxes for each component spectrum.

    """
    # Get component spd:
    if component_spds is None:
        if N_components is None: # Generate component spds from input args:
            spds = spd_builder(flux = None, peakwl = peakwl, fwhm = fwhm,\
                               strength_ph = strength_ph,\
                               peakwl_ph1 = peakwl_ph1, fwhm_ph1 = fwhm_ph1, strength_ph1 = strength_ph1,\
                               peakwl_ph2 = peakwl_ph2, fwhm_ph2 = fwhm_ph2, strength_ph2 = strength_ph2,\
                               verbosity = 0)
            N_components = spds.shape[0]
        else:
            spds = None # optimize spd model parameters, such as peakwl, fwhm, ...
            if optimizer_type == '3mixer':
                raise Exception("spd_optimizer(): optimizer_type = '3mixer' not supported for component parameter optimization. Use 'search' or 'mixer' instead.")
                
    else:
        spds = component_spds 
        N_components = spds.shape[0]
    
    # Check if there are at least 3 spds:
    if spds is not None:
        if (spds.shape[0]-1 < 3):
            raise Exception('spd_optimizer(): At least 3 component spds are required.')
                
        # Calculate xyz of components:
        xyzi = spd_to_xyz(spds, relative = False, cieobs = cieobs)
    else:
        if N_components < 3:
            raise Exception('spd_optimizer(): At least 3 component spds are required.')
    
    # Optimize spectrum:
    if optimizer_type == '3mixer': # Optimize fluxes for predefined set of component spectra
        
        # Calculate Yxy:
        Yxyt = colortf(target, tf = tar_type+'>Yxy', bwtf = cspace_bwtf)
        Yxyi = xyz_to_Yxy(xyzi) #input for color3mixer is Yxy
        
        if xyzi.shape[0] == 3: # Only 1 solution
            M = color3mixer(Yxyt,Yxyi[0:1,:],Yxyi[1:2,:],Yxyi[2:3,:])
            if (M<0).any():
                warnings.warn('spd_optimizer(): target outside of gamut')
        else:
            # Use triangle optimization:
            M, spd_opt, obj_vals = component_triangle_optimizer(spds, Yxyi = Yxyi, Yxy_target = Yxyt, cieobs = cieobs,\
                                                                      obj_fcn = obj_fcn, obj_fcn_pars = obj_fcn_pars, obj_fcn_weights = obj_fcn_weights,\
                                                                      obj_tar_vals = obj_tar_vals, decimals = decimals, \
                                                                      minimize_method = minimize_method, F_rss = F_rss,\
                                                                      minimize_opts = minimize_opts,\
                                                                      verbosity = verbosity)
            
    elif optimizer_type == 'mixer': # Optimize fluxes and component model parameters 
        
        # Calculate Yxy:
        Yxyt = colortf(target, tf = tar_type+'>Yxy', bwtf = cspace_bwtf)
                
        # Use Nmixer for optimization:
        
        
        raise Exception("spd_optimizer(): optimizer_type = 'mixer' not yet implemented. Use '3mixer'. ")

        
    elif optimizer_type == 'search': # Optimize fluxes and component model parameters (chromaticity is part of obj_fcn list)
        raise Exception("spd_optimizer(): optimizer_type = 'search' not yet implemented. Use '3mixer'. ")

    # Calculate combined spd from components and their fluxes:
    spds = (np.atleast_2d(M)*spds[1:].T).T.sum(axis = 0)
    
    if with_wl == True:
        spds = np.vstack((getwlr(wl), spds))
    return spds, M       



#------------------------------------------------------------------------------
if __name__ == '__main__':
    
    plt.close('all')
    cieobs = '1931_2'
    
#    #--------------------------------------------------------------------------
##    print('1: spd_builder():')
#    # Set up two basis LED spectra:
#    target = 3500
#    flux = [1,2,3]
#    peakwl = [450,530,600] # peak wavelengths of monochromatic leds
#    fwhm = [20,30,20] # fwhm of monochromatic leds
#    
#    strength_ph = None#[0.1,0.6,0.3] # one monochromatic and one phosphor led
#    
#    # Parameters for phosphor 1:
#    peakwl_ph1 = [530,550,550] 
#    fwhm_ph1 = [80,80,80]
#    strength_ph1 = [0.9,0.5,0.8]
#    
#    # Parameters for phosphor 1:
#    peakwl_ph2 = [590,600,600]
#    fwhm_ph2 = [90,90,90]
#    strength_ph2 = None 
#    
#    # Build spd from parameters settings defined above:
#    S = spd_builder(flux = flux, peakwl = peakwl, fwhm = fwhm,\
#                    strength_ph = strength_ph,\
#                    peakwl_ph1 = peakwl_ph1, fwhm_ph1 = fwhm_ph1, strength_ph1 = strength_ph1,\
#                    peakwl_ph2 = peakwl_ph2, fwhm_ph2 = fwhm_ph2, strength_ph2 = strength_ph2,\
#                    target = target, tar_type = 'cct', cieobs = cieobs,\
#                    verbosity = 1)
#    
#    # Check output agrees with target:
#    if target is not None:
#        xyz = spd_to_xyz(S, relative = False, cieobs = cieobs)
#        cct = xyz_to_cct(xyz, cieobs = cieobs, mode = 'lut')
#        print("S: Phosphor model / target cct: {:1.1f} K / {:1.1f} K\n\n".format(cct[0,0], target))
#
#        
#    #plot final combined spd:
#    plt.figure()
#    SPD(S).plot(color = 'm')
    
#    #--------------------------------------------------------------------------
#    # Set up three basis LED spectra:
#    flux = None
#    peakwl = [450,530,610] # peak wavelengths of monochromatic leds
#    fwhm = [30,35,15] # fwhm of monochromatic leds
#    
#    S2 = spd_builder(flux = flux,peakwl = peakwl, fwhm = fwhm,\
#                    strength_ph = 0, verbosity = 1)
#    
#    #plot component spds:
#    plt.figure()
#    SPD(S2).plot()
    
#    #--------------------------------------------------------------------------
##    print('2: spd_optimizer():')
#    target = 4000 # 4000 K target cct
#    tar_type = 'cct'
#    peakwl = [450,530,560,610]
#    fwhm = [30,35,30,15] 
#    obj_fcn1 = spd_to_iesrf
#    obj_fcn2 = spd_to_iesrg
#    obj_fcn = [obj_fcn1, obj_fcn2]
#    obj_tar_vals = [90,110]
#    obj_fcn_weights = [1,1]
#    decimals = [5,5]
#    N_components = None #if not None, spd model parameters (peakwl, fwhm, ...) are optimized
#    S3, _ = spd_optimizer(target, tar_type = tar_type, cspace_bwtf = {'cieobs' : cieobs, 'mode' : 'search'},\
#                          optimizer_type = '3mixer', N_components = N_components,\
#                          peakwl = peakwl, fwhm = fwhm, obj_fcn = obj_fcn, obj_tar_vals = obj_tar_vals,\
#                          obj_fcn_weights = obj_fcn_weights, decimals = decimals,\
#                          verbosity = 0)
#    
#    # Check output agrees with target:
#    xyz = spd_to_xyz(S3, relative = False, cieobs = cieobs)
#    cct = xyz_to_cct(xyz, cieobs = cieobs, mode = 'lut')
#    Rf = obj_fcn1(S3)
#    Rg = obj_fcn2(S3)
#    print('\nS3: Optimization results:')
#    print("S3: Optim / target cct: {:1.1f} K / {:1.1f} K".format(cct[0,0], target))
#    print("S3: Optim / target Rf: {:1.3f} / {:1.3f}".format(Rf[0,0], obj_tar_vals[0]))
#    print("S3: Optim / target Rg: {:1.3f} / {:1.3f}".format(Rg[0,0], obj_tar_vals[1]))
#    
#    #plot spd:
#    plt.figure()
#    SPD(S3).plot()