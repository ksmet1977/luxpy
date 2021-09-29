# -*- coding: utf-8 -*-
"""
Module for CAM "front-end" cmf adaptation
=========================================

 :translate_cmfI_to_cmfS(): | Using smooth RGB primaries, translate input data (spectral or tristimlus)
                            | for an indivual observer to the expected tristimulus values for a standard observer. 

 :get_conversion_matrix():  | Using smooth RGB primaries, get the 'translator' matrix to convert 
                            | tristimulus values calculated using an individual observer's 
                            | color matching functions (cmfs) to those calculated using the cmfs of 
                            | a standard observer.
                            
 :get_rgb_smooth_prims(): Get smooth R, G, B primaries with specified wavelength range
 
 :_R,_G,_B: precalculated smooth primaries with [360,830,1] wavelength range.
 

Created on Wed Apr 28 11:37:17 2021

@author: ksmet1977 [at] gmail.com
"""
import numpy as np
from luxpy import xyz_to_Yuv,_RFL, _WL3, _CMF, getwlr, cie_interp, spd_to_xyz, _CIE_D65, _CIE_E, spd_normalize

__all__ = ['translate', 'get_rgb_smooth_prims','_R','_G','_B','get_conversion_matrix', 'translate_cmfI_to_cmfS']

def get_rgb_smooth_prims(wl = _WL3):
    """ Get smooth R, G, B primaries with specified wavelength range"""
    wl = getwlr(wl)
    x = (wl - 535)/80
    
    f = lambda cc, a,b,c: a + b*x[cc] + c*x[cc]**2
    
    R, G, B = np.zeros((2,wl.shape[0])), np.zeros((2,wl.shape[0])), np.zeros((2,wl.shape[0]))
    R[0], G[0], B[0] = wl, wl, wl
    
    B[1,x < -1.5] = 1
    B[1,(-1.5 <= x) & (x < -0.5)] = f(((-1.5 <= x) & (x < -0.5)), -0.125,-1.5,-0.5)
    B[1,(-0.5 <= x) & (x <  0.5)] = f(((-0.5 <= x) & (x <  0.5)), 0.125,-0.5, 0.5)
    
    G[1,(-1.5 <= x) & (x < -0.5)] = f(((-1.5 <= x) & (x < -0.5)), 1.125, 1.5, 0.5)
    G[1,(-0.5 <= x) & (x <  0.5)] = f(((-0.5 <= x) & (x <  0.5)), 0.750, 0.0,-1.0)
    G[1,( 0.5 <= x) & (x <  1.5)] = f((( 0.5 <= x) & (x <  1.5)), 1.125,-1.5, 0.5)
    
    R[1,(-0.5 <= x) & (x <  0.5)] = f(((-0.5 <= x) & (x <  0.5)), 0.125, 0.5, 0.5)
    R[1,( 0.5 <= x) & (x <  1.5)] = f((( 0.5 <= x) & (x <  1.5)), -0.125, 1.5,-0.5)
    R[1,x >= 1.5] = 1
          
    return R,G,B

_R, _G, _B = get_rgb_smooth_prims(wl = [360,830,1])

def _get_rgb_tristim_matrix(cmf, R = None, G = None, B = None, wl = None):
    
    # deal with wavelengths:
    if wl is None:
        wl = cmf[0] # set to the ones of the cmf set
    else:
        cmf = cie_interp(cmf, wl_new = wl, kind = 'cmf') # interp cmfset to desired wl
    
    # Get smooth primaries (use input or generate from scratch):
    if (R is None) | (G is None) | (B is None):
        R_, G_, B_ = get_rgb_smooth_prims(wl = wl) # generate smooth primaries
    R = cie_interp(R, wl, kind = 'spd') if (R is not None) else R_ # set to input R and interp to desired wl, else use generated one
    G = cie_interp(G, wl, kind = 'spd') if (G is not None) else G_ # set to input G and interp to desired wl, else use generated one
    B = cie_interp(B, wl, kind = 'spd') if (B is not None) else B_ # set to input B and interp to desired wl, else use generated one
    
    # get tristimulus values of primaries:
    xyz_R = spd_to_xyz(R, relative = False, K = 1, cieobs = cmf)
    xyz_G = spd_to_xyz(G, relative = False, K = 1, cieobs = cmf)
    xyz_B = spd_to_xyz(B, relative = False, K = 1, cieobs = cmf)
    
    # create matrix:
    return np.vstack((xyz_B, xyz_G, xyz_R)), R, G, B, wl
    
def get_conversion_matrix(cmfI, cmfS, R = None, G = None, B = None, wl = None, output = 'Mc'):
    """ 
    Using smooth RGB primaries, get the 'translator' matrix to convert 
    tristimulus values calculated using an individual observer's 
    color matching functions (cmfs) to those calculated using the cmfs of 
    a standard observer.
    
    Args:
        :cmfI:
            | ndarray with cmfs of the individual observer
        :cmfI:
            | ndarray with cmfs of the standard observer
        :R,G,B:
            | None, optional
            | ndarray with spectrum of Red, Green and Blue smooth primaries.
            | If None, use the ones from Smet, Webster & Whitehead (2021).
        :wl:
            | None, optional
            | Before calculating the matrices convert the cmfs and RGBs to 
            | these wavelengths.
            | If None: use the wavelengths of each cmf set.
        :output:
            | 'Mc', optional
            | string specifying the requested output
            | E.g. 'Mc,R,G,B' would output the conversion matrix and the primary spectra.
            
    Returns:
        :Mc:
            | ndarray with conversion matrix
            
        :MI,MS:
            | optional output (specify using :output: argument, e.g. output = 'Mc, MI, MS')
            | ndarrays (matrices) with the tristimulus values (rows) 
            | of the smooth R,G,B primaries calculated using the cmfI and cmfS
            | color matching functions.
        :R,G,B:
            | optional output (specify using :output: argument, e.g. output = 'Mc,R,G,B')
            | ndarrays with the spectra of the smooth R,G,B primaries.
            
    References:
        1. Smet, KAG, Webster, M, and Whitehead, L, (2021), 
        Using Smooth Metamers to Estimate the Color Perceptions of 
        Diverse Color-Normal Observers.
    """
    
    # get tristimulus matrix for individual observer:
    MI, RI, GI, BI, _ = _get_rgb_tristim_matrix(cmfI, R = R, G = G, B = B, wl = wl)
    
    # get tristimulus matrix for standard observer:
    MS, RS, GS, BS, _ = _get_rgb_tristim_matrix(cmfS, R = RI, G = GI, B = BI, wl = wl)
    R,G,B = RS, GS, BS
    
    # calculate conversion matrix:
    Mc = np.linalg.inv(MI) @ MS
    
    if output == 'Mc':
        return Mc
    elif output == 'Mc,MS,MI,R,G,B':
        return Mc,MS,MI,R,G,B
    else:
        return eval(output)
    
def translate_cmfI_to_cmfS(dataI = None, cmfI = '1931_2', cmfS = '1931_2', data_type = 'spd', output = 'xyzS',
                           R = None, G = None, B = None, wl = None, **kwargs):
    """ 
    Using smooth RGB primaries, translate input data (spectral or tristimlus)
    for an indivual observer to the expected tristimulus values for a standard observer. 
    
    Args:
        :dataI:
            | None, optional
            | ndarray with spectral data or tristimulus values for an individual observer 
            | If None: return None (or other requested outputs as specified in :output:)
        :cmfI:
            | ndarray with cmfs of the individual observer
        :cmfI:
            | ndarray with cmfs of the standard observer
        :data_type:
            | 'spd', optional
            | If 'spd': data in :dataI: is spectral, else it's pre-calculated tristimulus values for the individual observer
        :R,G,B:
            | None, optional
            | ndarray with spectrum of Red, Green and Blue smooth primaries.
            | If None, use the ones from Smet, Webster & Whitehead (2021).
        :wl:
            | None, optional
            | Before calculating the matrices convert the cmfs and RGBs to 
            | these wavelengths.
            | If None: use the wavelengths of each cmf set.
        :output:
            | 'xyzS', optional
            | string specifying the requested output
            |    E.g. 'xyzS,xyzI,Mc' would output the tristimulus values corresponding
            |    to the input in :dataI: for the individual observer ('xyzI') and
            |    the 'translated' values for the standard observer ('xyzS') obtained
            |    using the conversion matrix 'Mc'.
        :kwargs:
            | keyword arguments to luxpy.spd_to_xyz() used when converting 
            | spectral input data to tristimulus values for the individual observer. 
            
    Returns:
        :xyzS:
            | default output (as specified by output = 'xyzS'):
            |   - ndarray with translated tristimulus values for standard observer
          
        :Mc:
            | optional output (specify using :output: argument, e.g. output = 'xyzS,Mc')
            | ndarray with 'translation' matrix to go from cmfI data to cmfS data.
        :xyzI:
            | optional output (specify using :output: argument, e.g. output = 'xyzS,xyzI')
            | ndarray tristimulus values for the individual observer 
        :MI,MS:
            | optional output (specify using :output: argument, e.g. output = 'xyzS, MI, MS')
            | ndarrays (matrices) with the tristimulus values (rows) 
            | of the smooth R,G,B primaries calculated using the cmfI and cmfS
            | color matching functions.
        :R,G,B:
            | optional output (specify using :output: argument, e.g. output = 'xyzS,R,G,B')
            | ndarrays with the spectra of the smooth R,G,B primaries.
            
    References:
        1. Smet, KAG, Webster, M, and Whitehead, L, (2021), 
        Using Smooth Metamers to Estimate the Color Perceptions of 
        Diverse Color-Normal Observers.
    """
    # Deal with wavelengths:
    if wl is None:
        if (data_type == 'spd') & (dataI is not None):
            wl = dataI[0] # use wavelengths of stimuli if spectral input!
            
    # Get ndarrays with cmfs if input is a string:
    if isinstance(cmfI, str):
        kwargs['K'] = kwargs.get('K',_CMF[cmfI]['K'])
        if kwargs['K'] is None: kwargs['K'] = _CMF[cmfI]['K']
        cmfI = _CMF[cmfI]['bar'].copy()
    if isinstance(cmfS, str):
        cmfS = _CMF[cmfS]['bar'].copy()
        
    # Get conversion matrix to translate between cmfI and cmfS:     
    Mc,MS,MI,R,G,B = get_conversion_matrix(cmfI, cmfS, R = R, G = G, B = B, wl = wl, output = 'Mc,MS,MI,R,G,B')
    wl = R[0] # get wavelengths actually used to calculate matrices
    
    if dataI is not None:
 
        # convert spdI to tristimulus values using cmfI:
        if data_type == 'spd':
            xyzI = spd_to_xyz(dataI, cieobs = cmfI, **kwargs)
        else:
            xyzI = dataI
        
        # apply translation:
        if len(xyzI.shape) == 3:
            xyzS = np.einsum('ij,klj->kli', Mc.T, xyzI)
        else:
            xyzS = np.einsum('ij,lj->li', Mc.T, xyzI)
    
    
    else:
        output = 'Mc'
        xyzS = None
        xyzI = None
        

    if output == 'xyzS':
        return xyzS 
    elif output == 'xyzS,xyzI':
        return xyzS,xyzI
    elif output == 'xyzS,xyzI,Mc':
        return xyzS,xyzI,Mc 
    else:
        return eval(output)        
    
    
translate = translate_cmfI_to_cmfS
    

            

if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    from scipy import stats
    
    R,G,B = get_rgb_smooth_prims(wl = [360,830,1])
    
    plt.figure()
    plt.plot(R[0], R[1], 'r')
    plt.plot(G[0], G[1], 'g')
    plt.plot(B[0], B[1], 'b')
    plt.xlabel('Wavelengths (nm)')
    plt.ylabel('Intensity (a.u.)')
    
    
    # get some stimulus data:
    relative = False
    spd = _CIE_D65.copy()
    spd = cie_interp(spd,getwlr([380,780,1]))
    spd = spd_normalize(spd,norm_type='pu', norm_f = 100, cieobs = '1964_10')
    xyzI_10 = spd_to_xyz(spd, relative = relative, cieobs = '1964_10')
    print('Test translator with spectral and xyz input:')
    print("    xyzI_10: ", xyzI_10)
    xyzS_2 = spd_to_xyz(spd, relative = relative, cieobs = '1931_2')
    print("    xyzS_2: ", 100*xyzS_2/xyzS_2[...,1])
    
    # translate 1964_10 to 1931_2:
    xyzS_2_est_spd, xyzI_10_spd, Mc = translate_cmfI_to_cmfS(spd, cmfI = '1964_10', cmfS = '1931_2', data_type = 'spd', output = 'xyzS,xyzI,Mc', relative = relative)
    print('\n    Conversion matrix, Mc: \n', Mc)
    print("\n    xyzS_2_est_spd: ", 100*xyzS_2_est_spd/xyzS_2_est_spd[...,1])
    xyzS_2_est_xyz = translate_cmfI_to_cmfS(xyzI_10, cmfI = '1964_10', cmfS = '1931_2', data_type = 'xyz', output = 'xyzS', relative = relative)
    print("    xyzS_2_est_xyz: ", 100*xyzS_2_est_xyz/xyzS_2_est_xyz[...,1])
    
    # test None input:
    xyzS_2_est_None, xyzI_10_None, Mc = translate_cmfI_to_cmfS(None, cmfI = '1964_10', cmfS = '1931_2', data_type = 'spd', output = 'xyzS,xyzI,Mc', relative = relative)
    print('\nTest translator with None input:')
    print('\n    Conversion matrix, Mc: \n', Mc)
    print("\n    xyzS_2_est_None: ",xyzS_2_est_None)
    print("    xyzI_10_spd_None: ", xyzI_10_None)
    
    # test with rfl input:
    rfl = _RFL['cri']['ies-tm30']['4880']['5nm'].copy()
    # rfl = _RFL['munsell']['R'].copy()
    xyzS_2_rfl = spd_to_xyz(spd, relative = True, cieobs = '1931_2', rfl = rfl)
    YuvS_2_rfl = xyz_to_Yuv(xyzS_2_rfl)
    xyzS_2_rfl_est, xyzI_10_rfl, Mc = translate_cmfI_to_cmfS(spd, cmfI = '1964_10', cmfS = '1931_2', data_type = 'spd', output = 'xyzS,xyzI,Mc', relative = True, rfl = rfl)
    YuvS_2_rfl_est = xyz_to_Yuv(xyzS_2_rfl_est)
    YuvI_10_rfl = xyz_to_Yuv(xyzI_10_rfl)
    fig, axs = plt.subplots(1,2)
    axs[0].plot(YuvS_2_rfl[...,1],YuvS_2_rfl[...,2],'r.', label = 'S: standard observer (1931 2°)')
    axs[0].plot(YuvS_2_rfl_est[...,1],YuvS_2_rfl_est[...,2],'g.', label = 'S_est: estimated standard observer (1931 2°)')
    axs[0].plot(YuvI_10_rfl[...,1],YuvI_10_rfl[...,2],'b.', label = 'I: individual observer (1964 10°)')
    axs[0].legend()
    axs[0].set_xlabel("u'")
    axs[0].set_ylabel("v'")
    
    DEuv_i_est = ((YuvS_2_rfl - YuvS_2_rfl_est)[...,1:]**2).sum(axis=-1)**0.5
    DEuv_i_I = ((YuvS_2_rfl - YuvI_10_rfl)[...,1:]**2).sum(axis=-1)**0.5
    axs[1].hist(DEuv_i_I, label = 'DEuv(S,I)', bins = 20)
    axs[1].hist(DEuv_i_est, label = 'DEuv(S,S_est)', bins = 20)
    axs[1].legend()
    axs[1].set_xlabel("DEu'v'")
    axs[1].set_ylabel("Frequency")
    print('DEuv(S, I) median = {:1.4f} / iqr = {:1.4f}'.format(np.median(DEuv_i_I), stats.iqr(DEuv_i_I)))
    print('DEuv(S, S_est) median = {:1.4f} / iqr = {:1.4f}'.format(np.median(DEuv_i_est), stats.iqr(DEuv_i_est)))

    
    
    
    
    
    
