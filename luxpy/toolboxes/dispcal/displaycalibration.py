# -*- coding: utf-8 -*-
"""
Module for display calibration
==============================
 :_PATH_DATA: path to package data folder   

 :_RGB:  set of RGB values that work quite well for display characterization
   
 :_XYZ: example set of measured XYZ values corresponding to the RGB values in _RGB
 
 :find_index_in_rgb(): Find the index/indices of a specific r,g,b combination k in the ndarray rgb.
     
 :find_pure_rgb(): Find the indices of all pure r,g,b (single channel on) in the ndarray rgb.
 
 :correct_for_black: Correct xyz for black level (flare) 
 
 :TR_ggo(),TRi_ggo(): Forward (rgblin-to-xyz) and inverse (xyz-to-rgblin) GGO Tone Response models.
 
 :TR_gog(),TRi_gog(): Forward (rgblin-to-xyz) and inverse (xyz-to-rgblin) GOG Tone Response models.
 
 :TR_gogo(),TRi_gogo(): Forward (rgblin-to-xyz) and inverse (xyz-to-rgblin) GOGO Tone Response models.
 
 :TR_sigmoid(),TRi_sigmoid(): Forward (rgblin-to-xyz) and inverse (xyz-to-rgblin) SIGMOID Tone Response models.
 
 :estimate_tr(): Estimate Tone Response curves.
 
 :optimize_3x3_transfer_matrix(): Optimize the 3x3 rgb-to-xyz transfer matrix.
     
 :get_3x3_transfer_matrix_from_max_rgb(): Get the rgb-to-xyz transfer matrix from the maximum R,G,B single channel outputs
    
 :calibrate(): Calculate TR parameters/lut and conversion matrices
   
 :calibration_performance(): Check calibration performance (cfr. individual and average color differences for each stimulus). 

 :rgb_to_xyz(): Convert input rgb to xyz
    
 :xyz_to_rgb(): Convert input xyz to rgb
     
 :DisplayCalibration(): Calculate TR parameters/lut and conversion matrices and store in object.
       
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import numpy as np

from luxpy import (math, _CMF, cie_interp, colortf, _CSPACE_AXES)
from luxpy.utils import _PKG_PATH, _SEP, getdata

__all__ = ['_PATH_DATA', '_parse_rgbxyz_input', 'find_index_in_rgb',
           '_plot_target_vs_predicted_lab','_plot_DEs_vs_digital_values',
           'calibrate', 'calibration_performance', 
           'rgb_to_xyz', 'xyz_to_rgb', 'DisplayCalibration','_RGB', '_XYZ',
           'TR_ggo','TRi_ggo','TR_gog','TRi_gog','TR_gogo','TRi_gogo',
           'TR_sigmoid','TRi_sigmoid', 'correct_for_black',
           '_rgb_linearizer','_rgb_delinearizer', 'estimate_tr',
           'optimize_3x3_transfer_matrix','get_3x3_transfer_matrix_from_max_rgb'
           ]

_PATH = _PKG_PATH + _SEP + 'toolboxes' + _SEP + 'dispcal' + _SEP 
_PATH_DATA = _PATH + _SEP + 'data' + _SEP

_RGB = getdata(_PATH_DATA + 'RGBcal.csv',sep = ',', header=None) # read default rgb calibration settings
_XYZ = getdata(_PATH_DATA + 'XYZcal.csv',sep=',', header=None) # read some example measured xyz data at _RGB settings

#------------------------------------------------------------------------------
# Start function definitions:
#------------------------------------------------------------------------------
def find_index_in_rgb(rgb, k = [255,255,255], as_bool = False):
    """ 
    Find the index/indices of a specific r,g,b combination k in the ndarray rgb. 
    (return a boolean array indicating the positions if as_bool == True)
    """ 
    tmp = (rgb[:,0] == k[0]) & (rgb[:,1] == k[1]) & (rgb[:,2] == k[2])
    if not as_bool: tmp = np.where(tmp)[0]
    return tmp
  
def find_pure_rgb(rgb, as_bool = True):
    """ 
    Find the indices of all pure r,g,b (single channel on) in the ndarray rgb.
    (return a boolean array indicating the positions if as_bool == True)
    """
    tmp = [(rgb[:,1]==0) & (rgb[:,2]==0), 
           (rgb[:,0]==0) & (rgb[:,2]==0), 
           (rgb[:,0]==0) & (rgb[:,1]==0)] 
    if not as_bool: tmp = [np.where(x)[0] for x in tmp]
    return tmp

    
def _clamp0(x): 
    """Clamp x to 0 to avoid negative values."""
    x[x<0] = 0 
    return x    

# -- Tone response function ---------------------------------------------------
def TR_ggo(x,*p):
    """ 
    Forward GGO tone response model (x = rgb; p = [gain,offset,gamma]).
    
    Notes:
        1. GGO model: y = gain*x**gamma + offset
    """
    tmp = p[1] + p[0]*x**p[2]
    return tmp

def TRi_ggo(x,*p):
    """ 
    Inverse GGO tone response model (x = xyz; p = [gain,offset,gamma]).
    
    Notes:
        1. GGO model: y = gain*x**gamma + offset
    """
    tmp = ((x.T-p[1])/p[0]).astype(complex)
    # tmp[tmp<0] = 0
    tmp = np.abs(tmp**(1/p[2]))
    return tmp

def TR_gog(x,*p):
    """ 
    Forward GOG tone response model (x = rgb; p = [gain,offset,gamma]).
    
    Notes:
        1. GOG model: y = (gain*x + offset)**gamma
    """
    tmp = (p[1] + p[0]*x).astype(complex)
    return np.abs(tmp**p[2])

def TRi_gog(x,*p):
    """ 
    Inverse GOG tone response model (x = xyz; p = [gain,offset,gamma]).
    
    Notes:
        1. GOG model: y = (gain*x + offset)**gamma
    """
    tmp = (((np.abs(x.T.astype(complex)**(1/p[2])))-p[1])/p[0])
    return tmp

def TR_gogo(x,*p):
    """ 
    Forward GOGO tone response model (x = rgb; p = [gain,offset,gamma,offset_]).
    
    Notes:
        1. GOGO model: y = (gain*x + offset)**gamma + offset_
    """
    tmp = (p[1] + p[0]*x).astype(complex) 
    # tmp[tmp<0] = 0
    return np.abs(tmp**p[2]) + p[3]

def TRi_gogo(x,*p):
    """ 
    Inverse GOGO tone response model (x = xyz; p = [gain,offset,gamma,offset_]).
    
    Notes:
        1. GOGO model: y = (gain*x + offset)**gamma + offset_
    """
    tmp = (x.T - p[3]).astype(complex)
    # tmp[tmp<0] = 0
    tmp = (((np.abs(tmp**(1/p[2])))-p[1])/p[0])
    return tmp

def TR_sigmoid(x,*p):
    """ 
    Forward SIGMOID tone response model (x = rgb; p = [gain, offset, gamma, m, a, q]).
    
    Notes:
        1. SIGMOID model: y = offset + gain * [1 / (1 + q*exp(-a/gamma*(x - m)))]**(gamma)]
    """
    gain, offset, gamma, m, a, q = p 
    return offset + gain* (1/(1 + q*np.exp(-a/gamma*(x - m))))**gamma

def TRi_sigmoid(x,*p):
    """ 
    Inverse SIGMOID tone response model (x = xyz; p = [gain, offset, gamma, m, a, q]).
    
    Notes:
        1. SIGMOID model: y = offset + gain * [1 / (1 + q*exp(-a/gamma*(x - m)))]**(gamma)]
    """
    gain, offset, gamma, m, a, q = p
    return (-1/(a/gamma))*np.log((1/q)*((gain/(x-offset))**(1/gamma) - 1)) + m

def TR_sigmoid2(x,*p):
    """ 
    Forward SIGMOID tone response model (x = rgb; p = [y_min, y_max, m, a, q, v]).
    
    Notes:
        1. SIGMOID model: y = y_min + (y_max - y_min)/ (1 + q*exp(-a*v*(x - m)))**(1/v)
    """
    xmin, xmax, m, a, q, v = p 
    return xmin + np.abs(xmax - xmin)/(1 + np.abs(q)*np.exp(-np.abs(a*v)*(x - m)))**np.abs(1/v)

def TRi_sigmoid2(x,*p):
    """ 
    Inverse SIGMOID tone response model (x = xyz; p = [y_min, y_max, m, a, q, v]).
    
    Notes:
        1. SIGMOID model: y = y_min + (y_max - y_min)/ (1 + q*exp(-a*v*(x - m)))**(1/v)
    """
    xmin, xmax, m, a , q, v = p
    return (-1/np.abs(a*v))*np.log((1/np.abs(q))*((np.abs(xmax-xmin)/(x-xmin))**np.abs(v) - 1)) + m


def TR_sigmoid3(x,*p):
    """ 
    Forward SIGMOID tone response model (x = rgb; p = [gain, offset, gamma, m, a]).
    
    Notes:
        1. SIGMOID model: y = offset + gain * [1 / (1 + exp(-a/gamma*(x - m)))]**(gamma)]
    """
    gain, offset, gamma, m, a = p 
    return offset + gain* (1/(1 + np.exp(-a/1*(x - m))))**gamma

def TRi_sigmoid3(x,*p):
    """ 
    Inverse SIGMOID tone response model (x = xyz; p = [gain, offset, gamma, m, a]).
    
    Notes:
        1. SIGMOID model: y = offset + gain * [1 / (1 + exp(-a/gamma*(x - m)))]**(gamma)]
    """
    gain, offset, gamma, m, a = p
    return (-1/(a/1))*np.log(((gain/(x-offset))**(1/gamma) - 1)) + m


def _rgb_linearizer(rgb, tr, tr_type = 'lut', nbit = 8):
    """ Linearize rgb using tr tone response function represented by a GGO, GOG, GOGO, LUT or PLI (cfr. piecewise linear interpolator) model"""
    max_dac = 2**nbit - 1
    if tr_type == 'ggo':
        return _clamp0(np.array([TR_ggo(rgb[:,i]/max_dac,*tr[i]) for i in range(3)]).T) # linearize all rgb values and clamp to 0
    elif tr_type == 'gog':
        return _clamp0(np.array([TR_gog(rgb[:,i]/max_dac,*tr[i]) for i in range(3)]).T) # linearize all rgb values and clamp to 0
    elif tr_type == 'gogo':
        return _clamp0(np.array([TR_gogo(rgb[:,i]/max_dac,*tr[i]) for i in range(3)]).T) # linearize all rgb values and clamp to 0
    elif tr_type == 'lut':
        return _clamp0(np.array([tr[np.asarray(rgb[:,i],dtype= np.int32),i] for i in range(3)]).T) # linearize all rgb values and clamp to 0
    elif tr_type == 'pli':
        return _clamp0(np.array([tr['fw'][i](rgb[:,i]/max_dac) for i in range(3)]).T) # linearize all rgb values and clamp to 0
    elif tr_type == 'sigmoid':
        return _clamp0(np.array([TR_sigmoid(rgb[:,i]/max_dac,*tr[i]) for i in range(3)]).T) # linearize all rgb values and clamp to 0
    elif tr_type == 'sigmoid2':
        return _clamp0(np.array([TR_sigmoid2(rgb[:,i]/max_dac,*tr[i]) for i in range(3)]).T) # linearize all rgb values and clamp to 0
    elif tr_type == 'sigmoid3':
        return _clamp0(np.array([TR_sigmoid3(rgb[:,i]/max_dac,*tr[i]) for i in range(3)]).T) # linearize all rgb values and clamp to 0

def _rgb_delinearizer(rgblin, tr, tr_type = 'lut', nbit = 8):
    """ De-linearize linear rgblin using tr tone response function represented by GGO, GOG, GOGO, LUT or PLI (cfr. piecewise linear interpolator) model"""
    max_dac = 2**nbit - 1
    if tr_type == 'ggo':
        return np.array([TRi_ggo(rgblin[:,i],*tr[i])*max_dac for i in range(3)]).T
    elif tr_type == 'gog':
        return np.array([TRi_gog(rgblin[:,i],*tr[i])*max_dac for i in range(3)]).T
    elif tr_type == 'gogo':
        return np.array([TRi_gogo(rgblin[:,i],*tr[i])*max_dac for i in range(3)]).T
    elif tr_type == 'lut':
        maxv = (tr.shape[0] - 1)
        bins = np.vstack((tr-np.diff(tr,axis=0,prepend=0)/2,tr[-1,:]+0.01)) # create bins
        bins_mono_increasing = np.array([np.all(x[1:] >= x[:-1]) for x in bins.T])
        if not bins_mono_increasing.all():
            raise Exception('Bins not monotonically increasing -> lut cannot be inverted !')
        idxs = np.array([(np.digitize(rgblin[:,i],bins[:,i]) - 1)  for i in range(3)]).T # find bin indices
        idxs[idxs>maxv] = maxv 
        rgb = np.arange(tr.shape[0])[idxs]
        return rgb
    elif tr_type == 'pli':
        return np.array([tr['bw'][i](rgblin[:,i])*max_dac for i in range(3)]).T
    elif tr_type == 'sigmoid':
        return np.array([TRi_sigmoid(rgblin[:,i],*tr[i])*max_dac for i in range(3)]).T
    elif tr_type == 'sigmoid2':
        return np.array([TRi_sigmoid2(rgblin[:,i],*tr[i])*max_dac for i in range(3)]).T
    elif tr_type == 'sigmoid3':
        return np.array([TRi_sigmoid3(rgblin[:,i],*tr[i])*max_dac for i in range(3)]).T



def correct_for_black(xyz, rgb, xyz_black = None):
    """ Correct xyz for black level (flare) """
    if xyz_black is None: 
        p_blacks = (rgb[:,0]==0) & (rgb[:,1]==0) & (rgb[:,2]==0)
        xyz_black = xyz[p_blacks,:].mean(axis=0,keepdims=True)
    
    # Calculate flare corrected xyz:
    xyz_fc = xyz - xyz_black
    xyz_fc[xyz_fc<0] = 0 
    return xyz_fc, xyz_black
    
def estimate_tr(rgb, xyz, black_correct = True, xyz_black = None,
                tr_L_type = 'lms', tr_type = 'lut', tr_par_lower_bounds = (0,-0.1,0,-0.1), 
                cieobs = '1931_2', nbit = 8, tr_ensure_increasing_lut_at_low_rgb = 0.2, 
                tr_force_increasing_lut_at_high_rgb = True, verbosity = 1,
                tr_rms_break_threshold = 0.01, tr_smooth_window_factor = None): 
    """
    Estimate tone response functions.
    
    Args:
        :rgb:
            | ndarray [Nx3] of RGB values 
            | rgcal must contain at least the following type of settings:
            | - pure R,G,B: e.g. for pure R: (R != 0) & (G==0) & (B == 0)
            | - white(s): R = G = B = 2**nbit-1
            | - black(s): R = G = B = 0
        :xyz:
            | ndarray [Nx3] of measured XYZ values for the RGB settings in rgb.
        :black_correct:
            | True, optional
            | If True: correct xyz for black -> xyz - xyz_black
        :xyz_black:
            | None or ndarray, optional
            | If None: determine xyz_black from input data (must contain rgb = [0,0,0]!)
        :tr_L_type:
            | 'lms', optional
            | Type of response to use in the derivation of the Tone-Response curves.
            | options:
            |  - 'lms': use cone fundamental responses: L vs R, M vs G and S vs B 
            |           (reduces noise and generally leads to more accurate characterization) 
            |  - 'Y': use the luminance signal: Y vs R, Y vs G, Y vs B
        :tr_type:
            | 'lut', optional
            | options:
            |  - 'lut': Derive/specify Tone-Response as a look-up-table
            |  - 'ggo': Derive/specify Tone-Response as a gain-gamma-offset function: y = gain*x**gamma + offset
            |  - 'gog': Derive/specify Tone-Response as a gain-offset-gamma function:  y = (gain*x + offset)**gamma
            |  - 'gogo': Derive/specify Tone-Response as a gain-offset-gamma-offset function: y = (gain*x + offset)**gamma + offset
            |  - 'sigmoid': Derive/specify Tone-Response as a sigmoid function: y = offset + gain * [1 / (1 + q*exp(-(a/gamma)*(x - m)))]**(gamma)
            |  - 'pli': Derive/specify Tone-Response as a piecewise linear interpolation function
        :tr_par_lower_bounds:
            | (0,-0.1,0,-0.1), optional
            | Lower bounds used when optimizing the parameters of the GGO, GOG, GOGO tone
            | response functions. Try different set of fit fails. 
            | Tip for GOG & GOGO: try changing -0.1 to 0 (0 is not default,
            |          because in most cases this leads to a less goog fit)
        :cieobs:
            | '1931_2', optional
            | CIE CMF set used to determine the XYZ tristimulus values
            | (needed when tr_L_type == 'lms': determines the conversion matrix to
            | convert xyz to lms values)
        :nbit:
            | 8, optional
            | RGB values in nbit format (e.g. 8, 16, ...)
        :tr_ensure_increasing_lut_at_low_rgb:
            | 0.2 or float (max = 1.0) or None, optional
            | Ensure an increasing lut by setting all values below the RGB with the maximum
            | zero-crossing of np.diff(lut) and RGB/RGB.max() values of :tr_ensure_increasing_lut_at_low_rgb:
            | (values of 0.2 are a good rule of thumb value)
            | Non-strictly increasing lut values can be caused at low RGB values due
            | to noise and low measurement signal. 
            | If None: don't force lut, but keep as is.
        :tr_force_increasing_lut_at_high_rgb:
            | True, optional
            | If True: ensure the tone response curves in the lut are monotonically increasing.
            |          by finding the first 1.0 value and setting all values after that also to 1.0.
        :verbosity:
            | 1, optional
            | > 0: print and plot optimization results
        :tr_rms_break_threshold:
            | 0.01, optional
            | Threshold for breaking a loop that tries different bounds 
            | for the gain in the TR optimization for the GGO, GOG, GOGO models.
            | (for some input the curve_fit fails, but succeeds on using different bounds)
        :tr_smooth_window_factor:
            | None, optional
            | Determines window size for smoothing of data using scipy's savgol_filter prior to determining the TR curves.
            | window_size = x.shape[0]//tr_smooth_window_factor 
            | If None: don't apply any smoothing
            
    Returns:
        :tr:
            | Tone Response function parameters or lut or piecewise linear interpolation functions (forward and backward)
        :xyz_black:
            | ndarray with XYZ tristimulus values of black
        :p_pure:
            | ndarray with positions in xyz and rgb that contain data corresponding to the black level (rgb = [0,0,0]).
    """
    max_dac = 2**nbit - 1

    # for smoothing data prior to fitting TR:    
    if (tr_smooth_window_factor is not None): 
        make_odd = lambda x: x+np.abs(x%2-1)
        window_length = lambda x: make_odd(x.shape[0]//tr_smooth_window_factor)
        poly_orders = np.array([0,1,2,3])
        poly_order = lambda x: poly_orders[poly_orders < window_length(x)][-1]
    
        from scipy.signal import savgol_filter # lazy import
        
        fsm = (lambda x: savgol_filter(x, window_length(x), poly_order(x))) 
    else:
        fsm = (lambda x: x)

    # correct for black
    if black_correct:
        xyz, xyz_black = correct_for_black(xyz, rgb, xyz_black = xyz_black)
    else:
        xyz_black = np.array([[0,0,0]])
    
    # get positions of pure r, g, b values:
    p_pure = [(rgb[:,1]==0) & (rgb[:,2]==0), 
              (rgb[:,0]==0) & (rgb[:,2]==0), 
              (rgb[:,0]==0) & (rgb[:,1]==0)] 
    
    # set type of L-response to use: Y for R,G,B or L,M,S for R,G,B:
    if tr_L_type == 'Y':
        L = np.array([xyz[:,1] for i in range(3)]).T
    elif tr_L_type == 'lms':
        lms = (math.normalize_3x3_matrix(_CMF[cieobs]['M'].copy()) @ xyz.T).T
        L = np.array([lms[:,i] for i in range(3)]).T
        
        
    # Get rgb linearizer parameters for GGO, GOG, GOGO models or lut or PLI and apply to all rgb's:
    if (tr_type == 'ggo') | (tr_type == 'gog') | (tr_type == 'gogo') | ('sigmoid' in tr_type): 
    
        from scipy.optimize import curve_fit # lazy import 
        
        rgb_device = np.array([np.clip(rgb[p_pure[i],i] + 1e-300,0,max_dac) for i in range(3)])
        rgb_lin = np.array([np.clip(L[p_pure[i],i]/L[p_pure[i],i].max(),0,1) for i in range(3)])
        
        
        # Try different bounds as sometimes, depending on the input,
        # the curve_fit fails for some bounds (keep best and break loop
        #  when a solution with a rms < tr_rms_break_threshold):
        ks = [10 - i for i in range(12)]
        trs = []
        rmss = np.ones((len(ks),))*10
        for k in ks: 
            tr_par_lower_bounds_cp = [i for i in tr_par_lower_bounds]
            tr_par_lower_bounds_cp[1] = -0.1*k

            if tr_type == 'ggo':
                TR = TR_ggo
                bounds = (tr_par_lower_bounds_cp[:3],(np.inf,np.inf,np.inf))
                p0 = [1,0,1]
            elif tr_type == 'gog':
                TR = TR_gog
                bounds = (tr_par_lower_bounds_cp[:3],(np.inf,np.inf,np.inf))
                p0 = [1,0,1]
            elif tr_type == 'gogo':
                TR = TR_gogo
                bounds = (tr_par_lower_bounds_cp,(np.inf,np.inf,np.inf))
                p0 = [1,0,1,0]
            elif tr_type == 'sigmoid': # gain, offset, gamma, m, a, q
                TR = TR_sigmoid
                bounds = ((*tr_par_lower_bounds_cp[:3],0,0,0),(30,30,30,1,np.inf, np.inf))
                p0 = np.array([1,0,2,0.7,10,1])# + 2*(np.random.rand(6)-0.5)*0.005*k
            elif tr_type == 'sigmoid2':
                TR = TR_sigmoid2
                bounds = ((-1,0,0,0,0.1,0),(1,np.inf,1,np.inf,1,np.inf))
                p0 = np.array([0,1,0.5,10,1,1])# + 2*(np.random.rand(6)-0.5)*0.005*k
            elif tr_type == 'sigmoid3': # gain, offset, gamma, m, a, q
                TR = TR_sigmoid3
                bounds = ((*tr_par_lower_bounds_cp[:3],0,0,0),(30,30,30,1,np.inf))
                p0 = np.array([1,0,2,0.7,10])# + 2*(np.random.rand(6)-0.5)*0.005*k
            try: 
                tr = np.array([curve_fit(TR, rgb_device[i]/max_dac, fsm(rgb_lin[i]), p0 = p0, bounds = bounds)[0] for i in range(3)]) # calculate parameters of each TR
            except: 
                tr = np.array([curve_fit(TR, rgb_device[i]/max_dac, fsm(rgb_lin[i]), p0 = p0, bounds = (0,np.inf))[0] for i in range(3)]) 
            
            rgb_lin_est = _rgb_linearizer(rgb_device.T.copy(), tr, tr_type = tr_type, nbit = nbit)
            rms = np.array([((rgb_lin_est.T[i] - rgb_lin[i])**2).mean() for i in range(3)]).sum()**0.5
            trs.append(tr)
            rmss[10 - k] = rms
            if (rms < tr_rms_break_threshold) | (tr_type == 'sigmoid2'): break
            #print(bounds, rms)
            
        p = rmss.argmin() 
        tr = np.array(trs)[p] #use tr with best results

    if (tr_type == 'ggo') | (tr_type == 'gog') | (tr_type == 'gogo') | ('sigmoid' in tr_type):
        pass
    elif tr_type == 'lut':
    
        from scipy import interpolate # lazy import

        dac = np.arange(2**nbit)
        idxs = [rgb[p_pure[i],i].argsort() for i in range(3)] # make sure we get monotonically increasing values for interpolation
        lut = np.array([interpolate.PchipInterpolator(rgb[p_pure[i],i][idxs[i]],fsm(L[p_pure[i],i][idxs[i]]/L[p_pure[i],i][idxs[i]].max()))(dac) for i in range(3)]).T # use this one to avoid potential overshoot with cubic spline interpolation (but slightly worse performance)
        lut[lut<0] = 0
        
        lut = lut/lut.max(axis=0,keepdims=True) # ensure lut has max 1! not 0.99999999999
        #lut[np.isclose(lut,1,atol = 1e-10,rtol = 1e-10)] = 1
          
        # ensure monotonically increasing lut values for low signal:
        if tr_ensure_increasing_lut_at_low_rgb is not None:
            #tr_ensure_increasing_lut_at_low_rgb = 0.2 # anything below that has a zero-crossing for diff(lut) will be set to zero
            for i in range(3):
                p0 = np.where((np.diff(lut[dac/dac.max() < tr_ensure_increasing_lut_at_low_rgb,i])<=0))[0]
                if p0.any():
                    p0 = range(0,p0[-1]+1)
                    lut[p0,i] = 0
        tr = lut
        
        # ensure monotonically increasing lut values for high signal:
        if tr_force_increasing_lut_at_high_rgb:
            for i in range(3): 
                tr[np.where(tr[:,i] == 1)[0][0]:,i] = 1
                
    elif tr_type == 'pli':
        from scipy import interpolate # lazy import
        pli = {
                'fw' : np.array([interpolate.interp1d(rgb[p_pure[i],i]/max_dac,fsm(L[p_pure[i],i]/L[p_pure[i],i].max()), fill_value = (0,1)) for i in range(3)]).T,
                'bw' : np.array([interpolate.interp1d(L[p_pure[i],i]/L[p_pure[i],i].max(), fsm(rgb[p_pure[i],i]/max_dac), fill_value = (0,1)) for i in range(3)]).T
                }
        tr = pli
    else:
        raise Exception('tr_type: {} is not defined !'.format(tr_type))
        
    # plot:
    if verbosity > 0:
        colors = 'rgb'
        linestyles = ['-','--',':']
        rgball = np.repeat(np.arange(2**nbit)[:,None],3,axis=1)
        Lall = _rgb_linearizer(rgball, tr, tr_type = tr_type, nbit = nbit)
        
        import matplotlib.pyplot as plt # lazy import
        
        plt.figure()
        for i in range(3):
            plt.plot(rgb[p_pure[i],i],L[p_pure[i],i]/L[p_pure[i],i].max(),colors[i]+'o')
            plt.plot(rgball[:,i],Lall[:,i],colors[i]+linestyles[i],label=colors[i])
        plt.xlabel('Display RGB')
        plt.ylabel('Linear RGB')
        plt.legend()
        plt.title('Tone response curves')
        
    return tr, xyz_black, p_pure



#------------------------------------------------------------------------------
def optimize_3x3_transfer_matrix(xyz, rgb, black_correct = True, xyz_black = None, rgblin = None,
                                 nbit = 8, cspace = 'lab', avg = lambda x: ((x**2).mean()**0.5),
                                 tr = None, tr_type = None, verbosity = 0):
    """ 
    Optimize the 3x3 rgb-to-xyz transfer matrix
    
    Args:
        :xyz:
            | ndarray with measured XYZ tristimulus values (not correct for the black-level)
        :rgb:
            | device RGB values.
        :black_correct:
            | True, optional
            | If True: correct xyz for black -> xyz - xyz_black
        :xyz_black:
            | None or ndarray, optional
            | If None: determine xyz_black from input data (must contain rgb = [0,0,0]!)
        :nbit:
            | 8, optional
            | RGB values in nbit format (e.g. 8, 16, ...)
        :cspace:
            | color space or chromaticity diagram to calculate color differences in
            | when optimizing the xyz_to_rgb and rgb_to_xyz conversion matrices.
        :avg:
            | lambda x: ((x**2).mean()**0.5), optional
            | Function used to average the color differences of the individual RGB settings
            | in the optimization of the xyz_to_rgb and rgb_to_xyz conversion matrices.
        :tr:
            | None, optional
            | Tone Response function parameters or lut or piecewise linear interpolation functions (forward and backward)
            | If None -> :rgblin: must be provided !
        :tr_type:
            | 'lut', optional
            | options:
            |  - 'lut': Derive/specify Tone-Response as a look-up-table
            |  - 'ggo': Derive/specify Tone-Response as a gain-gamma-offset function: y = gain*x**gamma + offset
            |  - 'gog': Derive/specify Tone-Response as a gain-offset-gamma function:  y = (gain*x + offset)**gamma
            |  - 'gogo': Derive/specify Tone-Response as a gain-offset-gamma-offset function: y = (gain*x + offset)**gamma + offset
            |  - 'sigmoid': Derive/specify Tone-Response as a sigmoid function: y = offset + gain * [1 / (1 + q*exp(-(a/gamma)*(x - m)))]**(gamma)
            |  - 'pli': Derive/specify Tone-Response as a piecewise linear interpolation function
        :verbosity:
            | 1, optional
            | > 0: print and plot optimization results
            
    Returns:
        :M:
            | linear rgb-to-xyz conversion matrix            
    """ 

    # correct for black
    if black_correct:
        if xyz_black is None: 
            p_blacks = (rgb[:,0]==0) & (rgb[:,1]==0) & (rgb[:,2]==0)
            xyz_black = xyz[p_blacks,:].mean(axis=0,keepdims=True)
    else:
        xyz_black = np.array([[0,0,0]])
    
    if (rgblin is None):
        # linearize all rgb values and clamp to 0
        rgblin = _rgb_linearizer(rgb, tr, tr_type = tr_type, nbit = nbit) 
    
    # get rgblin to xyz matrix:
    M = np.linalg.lstsq(rgblin, xyz - xyz_black, rcond=None)[0].T 
        
    # get better approximation for conversion matrices:
    p_grays = (rgb[:,0] == rgb[:,1]) & (rgb[:,0] == rgb[:,2])
    p_whites = (rgb[:,0] == (2**nbit-1)) & (rgb[:,1] == (2**nbit-1)) & (rgb[:,2] == (2**nbit-1))
    xyzw = xyz[p_whites,:].mean(axis=0) # get xyzw for input into xyz_to_lab() or colortf()

    def optfcn(x, rgblin, xyz, xyz_black, cspace, p_grays, p_whites, out, verbosity):
        M = x.reshape((3,3))
        xyzest = (M @ rgblin.T).T + xyz_black
        xyzest[xyzest<0] = 0
        lab, labest = colortf(xyz, tf = cspace, xyzw = xyzw), colortf(xyzest, tf = cspace, xyzw = xyzw) # calculate lab coord. of cal. and est.
        DEs = ((lab - labest)**2).sum(axis=1)**0.5
        DEg = DEs[p_grays]
        DEw = DEs[p_whites]
        F = (avg(DEs)**2 + avg(DEg)**2 + avg(DEw**2))**0.5
        if verbosity > 1:
            print('\nPerformance of TR + rgb-to-xyz conversion matrix M:')
            print('all: DE(jab): avg = {:1.4f}, std = {:1.4f}'.format(avg(DEs),np.std(DEs)))
            print('grays: DE(jab): avg = {:1.4f}, std = {:1.4f}'.format(avg(DEg),np.std(DEg)))
            print('whites(s) DE(jab): avg = {:1.4f}, std = {:1.4f}'.format(avg(DEw),np.std(DEw)))
        if out == 'F':
            return F
        else:
            return eval(out)
    x0 = M.ravel()
    res = math.minimizebnd(optfcn, x0, args = (rgblin, xyz, xyz_black, cspace, p_grays, p_whites,'F',0), use_bnd=False)
    xf = res['x_final']
    M = optfcn(xf, rgblin, xyz, xyz_black, cspace, p_grays, p_whites,'M',verbosity)
    return M

def get_3x3_transfer_matrix_from_max_rgb(xyz, rgb, black_correct = True, xyz_black = None):
    """ 
    Get the rgb-to-xyz transfer matrix from the maximum R,G,B single channel outputs 
    
    Args: 
        :xyz:
            | ndarray with measured XYZ tristimulus values (not correct for the black-level)
        :rgb:
            | device RGB values.
        :black_correct:
            | True, optional
            | If True: correct xyz for black -> xyz - xyz_black
        :xyz_black:
            | None or ndarray, optional
            | If None: determine xyz_black from input data (must contain rgb = [0,0,0]!)
    
    Returns:
        :M:
            | linear rgb-to-xyz conversion matrix  
    """
    if black_correct:
        xyz_fc, _ = correct_for_black(xyz, rgb, xyz_black = xyz_black)
    else:
        xyz_fc = xyz 
    
    p_pure_max =  [(rgb[:,0]==rgb[:,0].max()) & (rgb[:,1]==0) & (rgb[:,2]==0), 
                   (rgb[:,0]==0) & (rgb[:,1]==rgb[:,1].max()) & (rgb[:,2]==0), 
                   (rgb[:,0]==0) & (rgb[:,1]==0) & (rgb[:,2]==rgb[:,2].max())]
    
    M = np.vstack((xyz_fc[p_pure_max[0]],
                   xyz_fc[p_pure_max[1]],
                   xyz_fc[p_pure_max[2]])).T
    return M
    
#------------------------------------------------------------------------------
def _parse_rgbxyz_input(rgb, xyz = None, sep = ',', header=None):
    """ Parse the rgb and xyz inputs """
    # process rgb, xyz inputs:
    if isinstance(rgb, str):
        rgb = getdata(rgb,sep=sep, header=header) # read rgb data
    if isinstance(xyz, str):
        xyz = getdata(xyz,sep=sep, header=header) # read measured xyz data 
    if xyz is None:
        rgb, xyz = rgb[...,:3], rgb[...,3:6]
    return rgb, xyz

#------------------------------------------------------------------------------
def calibrate(rgbcal, xyzcal, black_correct = True, 
              tr_L_type = 'lms', tr_type = 'lut', 
              tr_par_lower_bounds = (0,-0.1,0,-0.1),
              cieobs = '1931_2', nbit = 8, cspace = 'lab', 
              avg = lambda x: ((x**2).mean()**0.5), 
              tr_ensure_increasing_lut_at_low_rgb = 0.2, 
              tr_force_increasing_lut_at_high_rgb = True,
              tr_rms_break_threshold = 0.01,
              tr_smooth_window_factor = None,
              verbosity = 1, sep = ',',header = None, optimize_M = True): 
    """
    Calculate TR parameters/lut and conversion matrices.
    
    Args:
        :rgbcal:
            | ndarray [Nx3] or string with filename of RGB values 
            | rgcal must contain at least the following type of settings:
            | - pure R,G,B: e.g. for pure R: (R != 0) & (G==0) & (B == 0)
            | - white(s): R = G = B = 2**nbit-1
            | - gray(s): R = G = B
            | - black(s): R = G = B = 0
            | - binary colors: cyan (G = B, R = 0), yellow (G = R, B = 0), magenta (R = B, G = 0)
        :xyzcal:
            | ndarray [Nx3] or string with filename of measured XYZ values for 
            | the RGB settings in rgbcal.
        :black_correct:
            | True, optional
            | If True: correct xyz for black -> xyz - xyz_black
        :tr_L_type:
            | 'lms', optional
            | Type of response to use in the derivation of the Tone-Response curves.
            | options:
            |  - 'lms': use cone fundamental responses: L vs R, M vs G and S vs B 
            |           (reduces noise and generally leads to more accurate characterization) 
            |  - 'Y': use the luminance signal: Y vs R, Y vs G, Y vs B
        :tr_type:
            | 'lut', optional
            | options:
            |  - 'lut': Derive/specify Tone-Response as a look-up-table
            |  - 'ggo': Derive/specify Tone-Response as a gain-gamma-offset function: y = gain*x**gamma + offset
            |  - 'gog': Derive/specify Tone-Response as a gain-offset-gamma function:  y = (gain*x + offset)**gamma
            |  - 'gogo': Derive/specify Tone-Response as a gain-offset-gamma-offset function: y = (gain*x + offset)**gamma + offset
            |  - 'sigmoid': Derive/specify Tone-Response as a sigmoid function: y = offset + gain * [1 / (1 + q*exp(-(a/gamma)*(x - m)))]**(gamma)
            |  - 'pli': Derive/specify Tone-Response as a piecewise linear interpolation function
        :tr_par_lower_bounds:
            | (0,-0.1,0,-0.1), optional
            | Lower bounds used when optimizing the parameters of the GGO, GOG, GOGO tone
            | response functions. Try different set of fit fails. 
            | Tip for GOG & GOGO: try changing -0.1 to 0 (0 is not default,
            |          because in most cases this leads to a less goog fit)
        :cieobs:
            | '1931_2', optional
            | CIE CMF set used to determine the XYZ tristimulus values
            | (needed when tr_L_type == 'lms': determines the conversion matrix to
            | convert xyz to lms values)
        :nbit:
            | 8, optional
            | RGB values in nbit format (e.g. 8, 16, ...)
        :cspace:
            | color space or chromaticity diagram to calculate color differences in
            | when optimizing the xyz_to_rgb and rgb_to_xyz conversion matrices.
        :avg:
            | lambda x: ((x**2).mean()**0.5), optional
            | Function used to average the color differences of the individual RGB settings
            | in the optimization of the xyz_to_rgb and rgb_to_xyz conversion matrices.
        :tr_ensure_increasing_lut_at_low_rgb:
            | 0.2 or float (max = 1.0) or None, optional
            | Ensure an increasing lut by setting all values below the RGB with the maximum
            | zero-crossing of np.diff(lut) and RGB/RGB.max() values of :tr_ensure_increasing_lut_at_low_rgb:
            | (values of 0.2 are a good rule of thumb value)
            | Non-strictly increasing lut values can be caused at low RGB values due
            | to noise and low measurement signal. 
            | If None: don't force lut, but keep as is.
        :tr_force_increasing_lut_at_high_rgb:
            | True, optional
            | If True: ensure the tone response curves in the lut are monotonically increasing.
            |          by finding the first 1.0 value and setting all values after that also to 1.0.
        :tr_rms_break_threshold:
            | 0.01, optional
            | Threshold for breaking a loop that tries different bounds 
            | for the gain in the TR optimization for the GGO, GOG, GOGO models.
            | (for some input the curve_fit fails, but succeeds on using different bounds)
        :tr_smooth_window_factor:
            | None, optional
            | Determines window size for smoothing of data using scipy's savgol_filter prior to determining the TR curves.
            | window_size = x.shape[0]//tr_smooth_window_factor 
            | If None: don't apply any smoothing
        :verbosity:
            | 1, optional
            | > 0: print and plot optimization results
        :sep:
            | ',', optional
            | separator in files with rgbcal and xyzcal data
        :header:
            | None, optional
            | header specifier for files with rgbcal and xyzcal data 
            | (see pandas.read_csv)
        :optimize_M:
            | True, optional
            | If True: optimize transfer matrix M
            | Else: use column matrix of tristimulus values of R,G,B channels at max.
            
    Returns:
        :M:
            | linear rgb to xyz conversion matrix
        :N:
            | xyz to linear rgb conversion matrix
        :tr:
            | Tone Response function parameters or lut or piecewise linear interpolation functions (forward and backward)
        :xyz_black:
            | ndarray with XYZ tristimulus values of black
        :xyz_white:
            | ndarray with tristimlus values of white
    """
    
    # process rgb, xyzcal inputs:
    rgbcal, xyzcal = _parse_rgbxyz_input(rgbcal, xyz = xyzcal, sep = sep, header=header)
    
    # Estimate tone response curves:
    tr, xyz_black, _ = estimate_tr(rgbcal, xyzcal, black_correct = black_correct,
                                   tr_L_type = tr_L_type, tr_type = tr_type, 
                                   tr_par_lower_bounds = tr_par_lower_bounds,
                                   cieobs = cieobs, nbit = nbit, 
                                   tr_ensure_increasing_lut_at_low_rgb = tr_ensure_increasing_lut_at_low_rgb, 
                                   tr_force_increasing_lut_at_high_rgb = tr_force_increasing_lut_at_high_rgb,
                                   tr_rms_break_threshold = tr_rms_break_threshold,
                                   tr_smooth_window_factor = tr_smooth_window_factor,
                                   verbosity = verbosity)
    
    # get xyz_white
    p_whites = (rgbcal[:,0] == (2**nbit-1)) & (rgbcal[:,1] == (2**nbit-1)) & (rgbcal[:,2] == (2**nbit-1))
    xyz_white = xyzcal[p_whites,:].mean(axis=0,keepdims=True) # get xyzw for input into xyz_to_lab() or colortf()

    if optimize_M: 
        M = optimize_3x3_transfer_matrix(xyzcal, rgbcal, nbit = nbit,
                                         black_correct = black_correct, 
                                         xyz_black = xyz_black, 
                                         tr = tr, tr_type = tr_type,
                                         cspace = cspace,
                                         verbosity = verbosity)

    else:
        M = get_3x3_transfer_matrix_from_max_rgb(xyzcal, rgbcal, black_correct = black_correct, xyz_black = xyz_black)
 
    # Calculate xyz-to-rgb transfer matrix from M:                   
    N = np.linalg.inv(M)
    
    return M, N, tr, xyz_black, xyz_white

def rgb_to_xyz(rgb, M, tr, xyz_black, tr_type = 'lut', nbit = 8): 
    """
    Convert input rgb to xyz.
    
    Args:
        :rgb:
            | ndarray [Nx3] with RGB values 
        :M:
            | linear rgb to xyz conversion matrix
        :tr:
            | Tone Response function represented by GGO, GOG, GOGO, LUT or PLI (piecewise linear function) models
        :xyz_black:
            | ndarray with XYZ tristimulus values of black
        :tr_type:
            | 'lut', optional
            | Type of Tone Response in tr input argument
            | options:
            |  - 'lut': Derive/specify Tone-Response as a look-up-table
            |  - 'ggo': Derive/specify Tone-Response as a gain-gamma-offset function: y = gain*x**gamma + offset
            |  - 'gog': Derive/specify Tone-Response as a gain-offset-gamma function:  y = (gain*x + offset)**gamma
            |  - 'gogo': Derive/specify Tone-Response as a gain-offset-gamma-offset function: y = (gain*x + offset)**gamma + offset
            |  - 'sigmoid': Derive/specify Tone-Response as a sigmoid function: y = offset + gain * [1 / (1 + q*exp(-(a/gamma)*(x - m)))]**(gamma)
            |  - 'pli': Derive/specify Tone-Response as a piecewise linear interpolation function
        :nbit:
            | 8, optional
            | RGB values in nbit format (e.g. 8, 16, ...)        
    
    Returns:
        :xyz:
            | ndarray [Nx3] of XYZ tristimulus values
    """
    return np.dot(M, _rgb_linearizer(rgb, tr, tr_type = tr_type, nbit = nbit).T).T + xyz_black

def xyz_to_rgb(xyz,N,tr, xyz_black, tr_type = 'lut', nbit = 8): 
    """
    Convert xyz to input rgb. 
    
    Args:
        :xyz:
            | ndarray [Nx3] with XYZ tristimulus values 
        :N:
            | xyz to linear rgb conversion matrix
        :tr:
            | Tone Response function represented by GGO, GOG, GOGO, LUT or PLI (piecewise linear function) models
        :xyz_black:
            | ndarray with XYZ tristimulus values of black
        :tr_type:
            | 'lut', optional
            | Type of Tone Response in tr input argument
            | options:
            |  - 'lut': Derive/specify Tone-Response as a look-up-table
            |  - 'ggo': Derive/specify Tone-Response as a gain-gamma-offset function: y = gain*x**gamma + offset
            |  - 'gog': Derive/specify Tone-Response as a gain-offset-gamma function:  y = (gain*x + offset)**gamma
            |  - 'gogo': Derive/specify Tone-Response as a gain-offset-gamma-offset function: y = (gain*x + offset)**gamma + offset
            |  - 'sigmoid': Derive/specify Tone-Response as a sigmoid function: y = offset + gain * [1 / (1 + q*exp(-(a/gamma)*(x - m)))]**(gamma)
            |  - 'pli': Derive/specify Tone-Response as a piecewise linear interpolation function
        :nbit:
            | 8, optional
            | RGB values in nbit format (e.g. 8, 16, ...)      
    Returns:
        :rgb:
            | ndarray [Nx3] of display RGB values
    """
    rgblin = _clamp0(np.dot(N,(xyz - xyz_black).T).T) # calculate rgblin and clamp to zero (on separate line for speed)
    rgblin[rgblin>1] = 1 # clamp to max = 1
    return np.round(_rgb_delinearizer(rgblin,tr, tr_type = tr_type, nbit = nbit)) # delinearize rgblin

def _plot_target_vs_predicted_lab(labtarget, labpredicted, cspace = 'lab', verbosity = 1):
    """ Make a plot of target vs predicted color coordinates """
    if verbosity > 0:
        xylabels = _CSPACE_AXES[cspace]
        laball = np.vstack((labtarget,labpredicted))
        ml,ma,mb = laball.min(axis=0)
        Ml,Ma,Mb = laball.max(axis=0)
        fml = 0.95*ml
        fMl = 1.05*Ml
        fma = 1.05*ma if ma < 0 else 0.95*ma
        fMa = 0.95*Ma if Ma < 0 else 1.05*Ma
        fmb = 1.05*mb if mb < 0 else 0.95*mb
        fMb = 0.95*Mb if Mb < 0 else 1.05*Mb
        
        import matplotlib.pyplot as plt # lazy import
        
        fig,(ax0,ax1,ax2) = plt.subplots(nrows=1,ncols=3, figsize = (15,4))
        ax0.plot(labtarget[...,1],labtarget[...,2],'bo',label = 'target')
        ax0.plot(labpredicted[...,1],labpredicted[...,2],'ro',label = 'predicted')
        ax0.axis([fma,fMa,fmb,fMb])
        ax1.plot(labtarget[...,1],labtarget[...,0],'bo',label = 'target')
        ax1.plot(labpredicted[...,1],labpredicted[...,0],'ro',label = 'predicted')
        ax1.axis([fma,fMa,fml,fMl])
        ax2.plot(labtarget[...,2],labtarget[...,0],'bo',label = 'target')
        ax2.plot(labpredicted[...,2],labpredicted[...,0],'ro',label = 'predicted')
        ax2.axis([fmb,fMb,fml,fMl])
        ax0.set_xlabel(xylabels[1])
        ax0.set_ylabel(xylabels[2])
        ax1.set_xlabel(xylabels[1])
        ax1.set_ylabel(xylabels[0])
        ax2.set_xlabel(xylabels[2])
        ax2.set_ylabel(xylabels[0])
        ax2.legend(loc='upper left')

def _plot_DEs_vs_digital_values(DEslab, DEsl, DEsab, rgbcal, avg = lambda x: ((x**2).mean()**0.5), nbit = 8, verbosity = 1):
    """ Make a plot of the lab, l and ab color differences for the different calibration stimulus types. """
    if verbosity > 0:
        p_pure = [(rgbcal[:,1]==0) & (rgbcal[:,2]==0), 
              (rgbcal[:,0]==0) & (rgbcal[:,2]==0), 
              (rgbcal[:,0]==0) & (rgbcal[:,1]==0)] 
        p_grays = (rgbcal[:,0] == rgbcal[:,1]) & (rgbcal[:,0] == rgbcal[:,2])
        p_whites = (rgbcal[:,0] == (2**nbit-1)) & (rgbcal[:,1] == (2**nbit-1)) & (rgbcal[:,2] == (2**nbit-1))
        p_cyans = (rgbcal[:,0]==0) & (rgbcal[:,1]!=0) & (rgbcal[:,2]!=0)
        p_yellows = (rgbcal[:,0]!=0) & (rgbcal[:,1]!=0) & (rgbcal[:,2]==0)
        p_magentas = (rgbcal[:,0]!=0) & (rgbcal[:,1]==0) & (rgbcal[:,2]==0)
        
        import matplotlib.pyplot as plt # lazy import
        
        fig,(ax0,ax1,ax2) = plt.subplots(nrows=1,ncols=3, figsize = (15,4))
        rgb_colors='rgb'
        rgb_labels=['red','green','blue']
        marker ='o'
        markersize = 10
        if p_whites.any():
            ax0.plot(rgbcal[p_whites,0], DEslab[p_whites],'ks',markersize = markersize, label='white')
            ax1.plot(rgbcal[p_whites,0], DEsl[p_whites],'ks',markersize = markersize,label='white')
            ax2.plot(rgbcal[p_whites,0], DEsab[p_whites],'ks',markersize = markersize,label='white')
        if p_grays.any():
            ax0.plot(rgbcal[p_grays,0], DEslab[p_grays], color = 'gray', marker = marker,linestyle='none',label='gray')
            ax1.plot(rgbcal[p_grays,0], DEsl[p_grays], color = 'gray', marker = marker,linestyle='none',label='gray')
            ax2.plot(rgbcal[p_grays,0], DEsab[p_grays], color = 'gray', marker = marker,linestyle='none',label='gray')
        for i in range(3):
            if p_pure[i].any():
                ax0.plot(rgbcal[p_pure[i],i], DEslab[p_pure[i]],rgb_colors[i]+marker,label=rgb_labels[i])
                ax1.plot(rgbcal[p_pure[i],i], DEsl[p_pure[i]],rgb_colors[i]+marker,label=rgb_labels[i])
                ax2.plot(rgbcal[p_pure[i],i], DEsab[p_pure[i]],rgb_colors[i]+marker,label=rgb_labels[i])
        if p_cyans.any():
            ax0.plot(rgbcal[p_cyans,1], DEslab[p_cyans],'c'+marker,label='cyan')
            ax1.plot(rgbcal[p_cyans,1], DEsl[p_cyans],'c'+marker,label='cyan')
            ax2.plot(rgbcal[p_cyans,1], DEsab[p_cyans],'c'+marker,label='cyan')
        if p_yellows.any():
            ax0.plot(rgbcal[p_yellows,0], DEslab[p_yellows],'y'+marker,label='yellow')
            ax1.plot(rgbcal[p_yellows,0], DEsl[p_yellows],'y'+marker,label='yellow')
            ax2.plot(rgbcal[p_yellows,0], DEsab[p_yellows],'y'+marker,label='yellow')
        if p_magentas.any():
            ax0.plot(rgbcal[p_magentas,0], DEslab[p_magentas],'m'+marker,label='magenta')
            ax1.plot(rgbcal[p_magentas,0], DEsl[p_magentas],'m'+marker,label='magenta')
            ax2.plot(rgbcal[p_magentas,0], DEsab[p_magentas],'m'+marker,label='magenta')
        ax0.plot(np.array([0,(2**nbit-1)*1.05]),np.hstack((avg(DEslab),avg(DEslab))),color = 'r',linewidth=2,linestyle='--')
        ax0.set_xlabel('digital values')
        ax0.set_ylabel('Color difference DElab')
        ax0.axis([0,(2**nbit-1)*1.05,0,max(DEslab)*1.1])
        ax0.set_title('DElab')
        ax1.plot(np.array([0,(2**nbit-1)*1.05]),np.hstack((avg(DEsl),avg(DEsl))),color = 'r',linewidth=2,linestyle='--')
        ax1.set_xlabel('digital values')
        ax1.set_ylabel('Color difference DEl')
        ax1.axis([0,(2**nbit-1)*1.05,0,max(DEslab)*1.1])
        ax1.set_title('DEl')
        ax2.plot(np.array([0,(2**nbit-1)*1.05]),np.hstack((avg(DEsab),avg(DEsab))),color = 'r',linewidth=2,linestyle='--')
        ax2.set_xlabel('digital values')
        ax2.set_ylabel('Color difference DEab')
        ax2.set_title('DEab')
        ax2.axis([0,(2**nbit-1)*1.05,0,max(DEslab)*1.1])
        ax2.legend(loc='upper left')

def calibration_performance(rgb, xyztarget, M, N, tr, xyz_black, xyz_white, tr_type = 'lut', 
                            cspace='lab', avg = lambda x: ((x**2).mean()**0.5), 
                            rgb_is_xyz = False, is_verification_data = False,
                            nbit = 8, verbosity = 1, sep = ',', header = None):
    """
    Check calibration performance. Calculate DE for each stimulus. 
    
    Args:
        :rgb:
            | ndarray [Nx3] or string with filename of RGB values 
            | (or xyz values if argument rgb_to_xyz == True!)
        :xyztarget:
            | ndarray [Nx3] or string with filename of target XYZ values corresponding 
            | to the RGB settings (or the measured XYZ values, if argument rgb_to_xyz == True).
        :M:
            | linear rgb to xyz conversion matrix
        :N:
            | xyz to linear rgb conversion matrix
        :tr:
            | Tone Response function represented by GGO, GOG, GOGO, LUT or PLI (piecewise linear function) models
        :xyz_black:
            | ndarray with XYZ tristimulus values of black
        :xyz_white:
            | ndarray with tristimlus values of white
        :tr_type:
            | 'lut', optional
            | Type of Tone Response in tr input argument
            | options:
            |  - 'lut': Derive/specify Tone-Response as a look-up-table
            |  - 'ggo': Derive/specify Tone-Response as a gain-gamma-offset function: y = gain*x**gamma + offset
            |  - 'gog': Derive/specify Tone-Response as a gain-offset-gamma function:  y = (gain*x + offset)**gamma
            |  - 'gogo': Derive/specify Tone-Response as a gain-offset-gamma-offset function: y = (gain*x + offset)**gamma + offset
            |  - 'sigmoid': Derive/specify Tone-Response as a sigmoid function: y = offset + gain * [1 / (1 + q*exp(-(a/gamma)*(x - m)))]**(gamma)
            |  - 'pli': Derive/specify Tone-Response as a piecewise linear interpolation function
        :cspace:
            | color space or chromaticity diagram to calculate color differences in.
        :avg:
            | lambda x: ((x**2).mean()**0.5), optional
            | Function used to average the color differences of the individual RGB settings
            | in the optimization of the xyz_to_rgb and rgb_to_xyz conversion matrices.
        :rgb_is_xyz:
            | False, optional
            | If True: the data in argument rgb are actually measured XYZ tristimulus values
            |           and are directly compared to the target xyz.
        :is_verification_data:
            | False, optional
            | If False: the data is assumed to be corresponding to RGB value settings used 
            |           in the calibration (i.e. containing whites, blacks, grays, pure and binary mixtures)
            | If True: no assumptions on content of rgb, so use this settings when
            |          checking the performance for a set of measured and target xyz data
            |          different than the ones used in the actual calibration measurements. 
        :nbit:
            | 8, optional
            | RGB values in nbit format (e.g. 8, 16, ...)
        :verbosity:
            | 1, optional
            | > 0: print and plot optimization results
        :sep:
            | ',', optional
            | separator in files with rgbcal and xyzcal data
        :header:
            | None, optional
            | header specifier for files with rgbcal and xyzcal data 
            | (see pandas.read_csv)
            
    Returns:
        :M:
            | linear rgb to xyz conversion matrix
        :N:
            | xyz to linear rgb conversion matrix
        :tr:
            | Tone Response function parameters or lut or piecewise linear interpolation functions (forward and backward)
        :xyz_black:
            | ndarray with XYZ tristimulus values of black
        :xyz_white:
            | ndarray with tristimlus values of white

    """ 
    # process rgb, xyzcal inputs:
    rgb, xyz = _parse_rgbxyz_input(rgb, xyz = xyztarget, sep = sep, header=header)
    
    if rgb_is_xyz == False: # estimate xyz, otherwise assume rgb already contains xyzs 
        xyzest = rgb_to_xyz(rgb,M,tr, xyz_black, tr_type = tr_type, nbit = nbit) # convert rgb of all samples to xyz
    else:
        xyzest = rgb
    lab, labest = colortf(xyz,tf=cspace,xyzw=xyz_white), colortf(xyzest,tf=cspace,xyzw=xyz_white) # calculate lab coord. of cal. and est.
    DElabi,DEli, DEabi = ((lab-labest)**2).sum(axis=1)**0.5, ((lab[:,:1]-labest[:,:1])**2).sum(axis=1)**0.5, ((lab[:,1:]-labest[:,1:])**2).sum(axis=1)**0.5 # calculate DE of all samples
    if verbosity > 0:
        print("\nCalibration performance (all colors): \n    DE(l*a*b*): avg = {:1.2f}, std = {:1.2f}".format(avg(DElabi),DElabi.std())) # print mean DEl*a*b*
        print("    DE(l*)    : avg = {:1.2f}, std = {:1.2f}".format(avg(DEli),DEli.std())) # print mean DEl*
        print("    DE(a*b*)  : avg = {:1.2f}, std = {:1.2f}".format(avg(DEabi),DEabi.std())) # print mean DEa*b*
    if is_verification_data == False:
        _plot_DEs_vs_digital_values(DElabi, DEli, DEabi, rgb, nbit = nbit, avg = avg, verbosity = verbosity)
    _plot_target_vs_predicted_lab(lab, labest, cspace = cspace, verbosity = verbosity)
    return DElabi,DEli, DEabi

# Create class:
class DisplayCalibration():
    """
    Class for display_calibration.
    
    Args:
        :rgbcal:
            | ndarray [Nx3] or string with filename of RGB values 
            | rgcal must contain at least the following type of settings:
            | - pure R,G,B: e.g. for pure R: (R != 0) & (G==0) & (B == 0)
            | - white(s): R = G = B = 2**nbit-1
            | - gray(s): R = G = B
            | - black(s): R = G = B = 0
            | - binary colors: cyan (G = B, R = 0), yellow (G = R, B = 0), magenta (R = B, G = 0)
        :xyzcal:
            | None, optional
            | ndarray [Nx3] or string with filename of measured XYZ values for 
            | the RGB settings in rgbcal.
            | if None: rgbcal is [Nx6] ndarray containing rgb (columns 0-2) and xyz data (columns 3-5)
        :tr_L_type:
            | 'lms', optional
            | Type of response to use in the derivation of the Tone-Response curves.
            | options:
            |  - 'lms': use cone fundamental responses: L vs R, M vs G and S vs B 
            |           (reduces noise and generally leads to more accurate characterization) 
            |  - 'Y': use the luminance signal: Y vs R, Y vs G, Y vs B
        :tr_type:
            | 'lut', optional
            | options:
            |  - 'lut': Derive/specify Tone-Response as a look-up-table
            |  - 'ggo': Derive/specify Tone-Response as a gain-gamma-offset function: y = gain*x**gamma + offset
            |  - 'gog': Derive/specify Tone-Response as a gain-offset-gamma function:  y = (gain*x + offset)**gamma
            |  - 'gogo': Derive/specify Tone-Response as a gain-offset-gamma-offset function: y = (gain*x + offset)**gamma + offset
            |  - 'sigmoid': Derive/specify Tone-Response as a sigmoid function: y = offset + gain * [1 / (1 + q*exp(-(a/gamma)*(x - m)))]**(gamma)
            |  - 'pli': Derive/specify Tone-Response as a piecewise linear interpolation function
        :cieobs:
            | '1931_2', optional
            | CIE CMF set used to determine the XYZ tristimulus values
            | (needed when tr_L_type == 'lms': determines the conversion matrix to
            | convert xyz to lms values)
        :nbit:
            | 8, optional
            | RGB values in nbit format (e.g. 8, 16, ...)
        :cspace:
            | color space or chromaticity diagram to calculate color differences in
            | when optimizing the xyz_to_rgb and rgb_to_xyz conversion matrices.
        :avg:
            | lambda x: ((x**2).mean()**0.5), optional
            | Function used to average the color differences of the individual RGB settings
            | in the optimization of the xyz_to_rgb and rgb_to_xyz conversion matrices.
        :tr_ensure_increasing_lut_at_low_rgb:
            | 0.2 or float (max = 1.0) or None, optional
            | Ensure an increasing lut by setting all values below the RGB with the maximum
            | zero-crossing of np.diff(lut) and RGB/RGB.max() values of :tr_ensure_increasing_lut_at_low_rgb:
            | (values of 0.2 are a good rule of thumb value)
            | Non-strictly increasing lut values can be caused at low RGB values due
            | to noise and low measurement signal. 
            | If None: don't force lut, but keep as is.
        :tr_force_increasing_lut_at_high_rgb:
            | True, optional
            | If True: ensure the tone response curves in the lut are monotonically increasing.
            |          by finding the first 1.0 value and setting all values after that also to 1.0.
        :tr_rms_break_threshold:
            | 0.01, optional
            | Threshold for breaking a loop that tries different bounds 
            | for the gain in the TR optimization for the GGO, GOG, GOGO models.
            | (for some input the curve_fit fails, but succeeds on using different bounds)
        :tr_smooth_window_factor:
            | None, optional
            | Determines window size for smoothing of data using scipy's savgol_filter prior to determining the TR curves.
            | window_size = x.shape[0]//tr_smooth_window_factor 
            | If None: don't apply any smoothing
        :verbosity:
            | 1, optional
            | > 0: print and plot optimization results
        :sep:
            | ',', optional
            | separator in files with rgbcal and xyzcal data
        :header:
            | None, optional
            | header specifier for files with rgbcal and xyzcal data 
            | (see pandas.read_csv)
        :optimize_M:
            | True, optional
            | If True: optimize transfer matrix M
            | Else: use column matrix of tristimulus values of R,G,B channels at max.


    Return:
        :calobject:
            | attributes are: 
            |  - M: linear rgb to xyz conversion matrix
            |  - N: xyz to linear rgb conversion matrix
            |  - TR: Tone Response function parameters for GGO, GOG, GOGO models or lut or piecewise linear interpolation functions (forward and backward)
            |  - xyz_black: ndarray with XYZ tristimulus values of black
            |  - xyz_white: ndarray with tristimlus values of white
            | as well as: 
            |  - rgbcal, xyzcal, cieobs, avg, tr_type, nbit, cspace, verbosity
            |  - performance: dictionary with various color differences set to np.nan
            |  -    (run calobject.performance() to fill it with actual values)
    """
    def __init__(self,
                 rgbcal, 
                 xyzcal = None, 
                 tr_L_type = 'lms', 
                 cieobs = '1931_2', 
                 tr_type = 'lut', 
                 nbit = 8, 
                 cspace = 'lab', 
                 avg = lambda x: ((x**2).mean()**0.5), 
                 tr_ensure_increasing_lut_at_low_rgb = 0.2,
                 tr_force_increasing_lut_at_high_rgb = True,
                 tr_rms_break_threshold = 0.01,
                 tr_smooth_window_factor = None,
                 verbosity = 1,
                 sep = ',',
                 header = None,
                 optimize_M = True):
        # process rgb, xyzcal inputs:
        rgbcal, xyzcal = _parse_rgbxyz_input(rgbcal, xyz = xyzcal, sep = sep, header=header)

        # get calibration parameters
        M, N, tr, xyz_black, xyz_white = calibrate(rgbcal, xyzcal = xyzcal, tr_L_type = tr_L_type, 
                                                   cieobs = cieobs, tr_type = tr_type, nbit = nbit,
                                                   avg = avg, cspace = cspace,
                                                   tr_ensure_increasing_lut_at_low_rgb = tr_ensure_increasing_lut_at_low_rgb, 
                                                   tr_force_increasing_lut_at_high_rgb = tr_force_increasing_lut_at_high_rgb,
                                                   tr_rms_break_threshold = tr_rms_break_threshold,
                                                   tr_smooth_window_factor = tr_smooth_window_factor,
                                                   verbosity = verbosity,
                                                   sep = sep, header = header,
                                                   optimize_M = optimize_M) 
        self.M = M
        self.optimize_M = optimize_M
        self.N = N
        self.TR = tr
        self.xyz_black = xyz_black
        self.xyz_white = xyz_white
        
        self.rgbcal = rgbcal
        self.xyzcal = xyzcal
        self.cieobs = cieobs
        self.tr_type = tr_type
        self.nbit = nbit
        self.cspace = cspace
        self.avg = avg
        self.tr_L_type = tr_L_type
        self.tr_ensure_increasing_lut_at_low_rgb = tr_ensure_increasing_lut_at_low_rgb 
        self.tr_force_increasing_lut_at_high_rgb = tr_force_increasing_lut_at_high_rgb
        self.tr_rms_break_threshold = tr_rms_break_threshold
        self.tr_smooth_window_factor = tr_smooth_window_factor

        self.verbosity = verbosity
        self.performance = {'DElab_mean':np.nan, 'DElab_std':np.nan,
                            'DEli_mean':np.nan,'DEl_std':np.nan,
                            'DEab_mean':np.nan,'DEab_std':np.nan}
        #self.performance = self.check_performance(self, verbosity = self.verbosity)
        
    def check_performance(self, rgb = None, xyz = None, verbosity = None, 
                          sep =',', header = None, 
                          rgb_is_xyz = False, is_verification_data = True):
        """
        Check calibration performance (if rgbcal is None: use calibration data).
        
        Args:
            :rgb:
                | None, optional
                | ndarray [Nx3] or string with filename of RGB values 
                | (or xyz values if argument rgb_to_xyz == True!)
                | If None: use self.rgbcal
            :xyz:
                | None, optional
                | ndarray [Nx3] or string with filename of target XYZ values corresponding 
                | to the RGB settings (or the measured XYZ values, if argument rgb_to_xyz == True).
                | If None: use self.xyzcal
            :verbosity:
                | None, optional
                | if None: use self.verbosity
                | if > 0: print and plot optimization results
            :sep:
                | ',', optional
                | separator in files with rgb and xyz data
            :header:
                | None, optional
                | header specifier for files with rgb and xyz data 
                | (see pandas.read_csv)
            :rgb_is_xyz:
                | False, optional
                | If True: the data in argument rgb are actually measured XYZ tristimulus values
                |           and are directly compared to the target xyz.
            :is_verification_data:
                | False, optional
                | If False: the data is assumed to be corresponding to RGB value settings used 
                |           in the calibration (i.e. containing whites, blacks, grays, pure and binary mixtures)
                |           Performance results are stored in self.performance.
                | If True: no assumptions on content of rgb, so use this settings when
                |          checking the performance for a set of measured and target xyz data
                |          different than the ones used in the actual calibration measurements. 
        
        Return:
            :performance: 
                | dictionary with various color differences.
        """
        if verbosity is None:
            verbosity = self.verbosity 
        if rgb is None:
            rgb = self.rgbcal
            xyz = self.xyzcal
            is_verification_data = False
            rgb_is_xyz = False
        DElabi,DEli, DEabi = calibration_performance(rgb, xyz, self.M, self.N, 
                                                     self.TR, self.xyz_black, self.xyz_white, 
                                                     cspace = self.cspace, tr_type = self.tr_type, 
                                                     avg = self.avg, nbit = self.nbit, 
                                                     verbosity = verbosity, 
                                                     sep = sep, header = header,
                                                     rgb_is_xyz = rgb_is_xyz,
                                                     is_verification_data = is_verification_data) # calculate calibration performance in cspace='lab'
        performance = {'DElab_mean':DElabi.mean(), 'DElab_std':DElabi.std(),
                       'DEli_mean':DEli.mean(),'DEl_std':DEli.std(),
                       'DEab_mean':DEabi.mean(),'DEab_std':DEabi.std()}
        if is_verification_data == False:
            self.performance = performance
        return performance
    
    def to_xyz(self, rgb):
        """ Convert display rgb to xyz. """
        return rgb_to_xyz(rgb, self.M, self.TR, self.xyz_black, tr_type = self.tr_type, nbit = self.nbit)

    def to_rgb(self, xyz):
        """ Convert xyz to display rgb. """
        return xyz_to_rgb(xyz, self.N, self.TR, self.xyz_black, tr_type = self.tr_type, nbit = self.nbit)


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt 
    import pandas as pd
    
    plt.close('all')
    
    # data = np.load('train_test3.npy',allow_pickle=True)[()]
    # (xyz_tr, rgb_tr) = data['train']
    # (xyz_t, rgb_t) = data['test']
    # tr_par_lower_bounds = (0,-0.1,0,-0.1) 
    # tr_type = 'gog'
    # tr, xyz_black, p_pure = estimate_tr(np.clip(rgb_tr,0,255).astype(dtype = np.int32), xyz_tr,
    #             black_correct = True, xyz_black = None,
    #             tr_L_type = 'Y', tr_type = tr_type, 
    #             tr_par_lower_bounds = tr_par_lower_bounds,
    #             cieobs = '1931_2', nbit = 8, 
    #             tr_ensure_increasing_lut_at_low_rgb = 0.2, 
    #             tr_force_increasing_lut_at_high_rgb = True,
    #             verbosity = 1)
    # raise Exception('')
    # M, N, tr, xyz_black, xyz_white=calibrate(np.clip(rgb_tr,0,255).astype(dtype = np.int32), xyz_tr, black_correct = True, 
    #               tr_L_type = 'Y', tr_type = tr_type, 
    #               tr_par_lower_bounds = tr_par_lower_bounds,
    #               cieobs = '1931_2', nbit = 8, cspace = 'lab', avg = lambda x: ((x**2).mean()**0.5), 
    #               tr_ensure_increasing_lut_at_low_rgb = 0.2, tr_force_increasing_lut_at_high_rgb = True,
    #               verbosity = 1, sep = ',',header = None, optimize_M = True)
    # DElabi,DEli, DEabi = calibration_performance(np.clip(rgb_tr,0,255).astype(dtype = np.int32), xyz_tr, M, N, tr, xyz_black, xyz_white, 
    #                                               cspace='lab', tr_type = 'gogo', avg = lambda x: ((x**2).mean()**0.5), 
    #                                               verbosity = 1, is_verification_data = False) # calculate calibration performance in cspace='lab'
    # print(DElabi.mean(),DEabi.mean())    
    # raise Exception('')
    
    #--------------------------------------------------------------------------
    # Set up calibration parameters:
    #--------------------------------------------------------------------------
    cieobs = '1931_2' # CMF set corresponding to XYZ measurements
    tr_type = 'ggo' # or 'ggo', 'gog', 'gogo' ('gog': gain-offset-gamma approach, 'lut': look-up-table, 'pli' : piecewise linear interpolator function)
    tr_smooth_window_factor = None # smooth data prior to calculating TR by using a savgol_filter with window size data.shape[0]//tr_smooth_window_factor
    tr_L_type = 'Y' # or 'Y' ('Y' : use RGB vs luminance for Tone-Response curve, 'lms', use R vs L, G vs M, B vs S)
    avg = np.mean # function to average DEs in matrix optimization in calibrate()
    cspace = 'lab' # colorspace in which color differences are calculated
    
    #--------------------------------------------------------------------------
    # read calibration data:
    #--------------------------------------------------------------------------
    xyzcal = pd.read_csv(_PATH_DATA + 'XYZcal.csv',sep=',', header=None).values # read measured xyz data 
    rgbcal = pd.read_csv(_PATH_DATA + 'RGBcal.csv',sep=',', header=None).values # read rgb data

   
    #--------------------------------------------------------------------------
    # Apply functions as an example:
    #--------------------------------------------------------------------------
    print('\nFunctional example:')
    
    # Get calibration gog-parameters/lut and conversion matrices M, N as well as xyz-black:
    M, N, tr, xyz_black, xyz_white = calibrate(rgbcal, xyzcal, tr_L_type = tr_L_type, tr_type = tr_type, tr_smooth_window_factor = tr_smooth_window_factor, avg = avg, cspace = cspace) # get calibration parameters
    if (tr_type == 'ggo') | (tr_type == 'gog') | (tr_type == 'gogo'):
        print("Calibration parameters :\nTR(gamma,offset,gain[,offset])=\n", np.round(tr,5),'\nM=\n',np.round(M,5),'\nN=\n',np.round(N,5))
   
    # Check calibration performance:
    DElabi,DEli, DEabi = calibration_performance(rgbcal, xyzcal, M, N, tr, xyz_black, xyz_white, 
                                                 cspace=cspace, tr_type = tr_type, avg = avg, 
                                                 verbosity = 1, is_verification_data = False) # calculate calibration performance in cspace='lab'
    
    raise Exception('')
    # define a test xyz for converion to rgb:
    xyz_test = np.array([[100.0,100.0,100.0]])*0.5 
    print('\nTest calibration for user defined xyz:\n    xyz_test_est:', xyz_test) # print chosen test xyz 
    
    # for a test xyz calculate the estimated rgb:
    rgb_test_est = xyz_to_rgb(xyz_test, N, tr, xyz_black, tr_type = tr_type) 
    print('    rgb_test_est:', rgb_test_est) # print estimated rgb 
    
    # calculate xyz again from estimated rgb (round-trip check: xyz->rgb->xyz):
    xyz_test_est = rgb_to_xyz(rgb_test_est, M, tr, xyz_black, tr_type = tr_type)
    print('    xyz_test_est:', np.round(xyz_test_est,1)) # print estimated xyz
    
    
    # Verify calibration with measured xyz and target xyz:
    # For this example predict some data to use as measured verification data, 
    # and use xyzcal as target data:
    # (set is_verification_data = True; note that only target vs predicted plots
    # will be made, as there are no pures, whites, grays, ... available)
    xyz_verification = rgb_to_xyz(rgbcal, M, tr, xyz_black, tr_type = tr_type)
    xyz_target = xyzcal
    DElabi,DEli, DEabi = calibration_performance(xyz_verification, xyz_target, M, N, tr, xyz_black, xyz_white, 
                                                 cspace=cspace, tr_type = tr_type, avg = avg, verbosity = 1, 
                                                 rgb_is_xyz = True, is_verification_data = True) # calculate calibration performance in cspace='lab'

    
    

    #--------------------------------------------------------------------------
    # Apply class as an example: (advantage: no need to pass M, N, TR, etc. around)
    #--------------------------------------------------------------------------
    print('\nClass  DisplayCalibration example:')
    
    # Create instance of  DisplayCalibration:
    cal1 =  DisplayCalibration(rgbcal, xyzcal = xyzcal, tr_L_type = tr_L_type, cieobs = cieobs, 
                               tr_type = tr_type, tr_smooth_window_factor = tr_smooth_window_factor,
                               avg = avg, cspace = cspace,
                               verbosity = 0)
    # or directly from files:
    cal1 =  DisplayCalibration(_PATH_DATA + 'RGBcal.csv', xyzcal = _PATH_DATA + 'XYZcal.csv', tr_L_type = tr_L_type, cieobs = cieobs, 
                               tr_type = tr_type, avg = avg, cspace = cspace,
                               verbosity = 1, sep =',')
    
    # Check calibration performance for calibration data itself:
    cal1.check_performance()
        
    # define a test xyz for converion to rgb:
    xyz_test = np.array([[100.0,100.0,100.0]])*0.5 
    print('\nTest calibration for user defined xyz:\n    xyz_test_est:', xyz_test) # print chosen test xyz 
 
    # for a test xyz calculate the estimated rgb:
    rgb_test_est = cal1.to_xyz(rgb_test_est)
    print('    rgb_test_est:', rgb_test_est) # print estimated rgb 
    
    # calculate xyz again from estimated rgb (round-trip check: xyz->rgb->xyz):
    xyz_test_est = cal1.to_xyz(rgb_test_est)
    print('    xyz_test_est:', np.round(xyz_test_est,1)) # print estimated xyz
    
    # Verify calibration with measured xyz and target xyz:
    # For this example predict some data to use as measured verification data, 
    # and use xyzcal as target data:
    # (set is_verification_data = True; note that only target vs predicted plots
    # will be made, as there are no pures, whites, grays, ... available)
    xyz_verification = cal1.to_xyz(rgbcal)
    xyz_target = xyzcal
    cal1.check_performance(xyz_verification, xyz_target, 
                           rgb_is_xyz = True, is_verification_data = True)

    

