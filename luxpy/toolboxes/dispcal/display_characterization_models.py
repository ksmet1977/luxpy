# -*- coding: utf-8 -*-
"""
Module for display calibration models
=====================================

 :get_DE_stats(): Get statistics ['mean','median','rms','75p','90p','95p','max']. 
 
 :ramp_data_to_cube_data(): Create a RGB and XYZ cube from the single channel ramps in the training data.
 
 :ColCharModel: Super class for colorimetric display characterization models 
 
 :GGO_GOG_GOGO_PLI: Class for characterization models that combine a 3x3 transfer matrix and a GGO, GOG, GOGO, SIGMOID, PLI and 1-D LUT Tone response curve  
                     |  - Tone Response curve models:
                     |    * GGO: gain-gamma-offset model: y = gain*x**gamma + offset
                     |    * GOG: gain-offset-gamma model: y = (gain*x + offset)**gamma
                     |    * GOG: gain-offset-gamma-offset model: y = (gain*x + offset)**gamma + offset
                     |    * SIGMOID: sigmoid (S-shaped) model: y = offset + gain* [1 / (1 + q*exp(-a/gamma*(x - m)))]**(gamma)
                     |    * PLI: Piece-wise Linear Interpolation
                     |    * LUT: 1-D Look-Up-Tables for the TR
                     |  - RGB-to-XYZ / XYZ-to-RGB transfer matrices:
                     |     * M fixed: derived from tristimulus values of maximum single channel output
                     |     * M optimized: by minimizing the RMSE between measured and predicted XYZ values
                     
 :ML: Super class for characterization models that are based on sklearn's machine learning algorithms (MLPRegression, POlynomialRegression)
 
 :MLPR: Class for Multi-Layer Perceptron Regressor based model.
     
 :POR: Class for POlynomial Regression based model.
 
 :LUTNNLI: Class for LUT-Nearest-Neighbour-distance-weighted-Linear-Interpolation based models.
     
 :LUTQHLI: Class for LUT-QHul-Linear-Interpolation based models (cfr. scipt.interpolate.LinearNDInterpolator)
     
     
Created on Sun Nov  6 14:32:07 2022

@author: u0032318
"""
import copy
import itertools
import numpy as np

from luxpy import colortf
from luxpy.utils import getdata
from luxpy.toolboxes.dispcal import displaycalibration as dc

np.seterr(divide = 'raise', invalid = 'raise')

__all__ = ['get_DE_stats', 'ramp_data_to_cube_data', 'ColCharModel', 
           'GGO_GOG_GOGO_PLI','ML','MLPR','POR','LUTNNLI', 'LUTQHLI']

#--- helper functions ---------------------------------------------------------
def get_DE_stats(DEi, axis = None):
    """ 
    Get statistics ['mean','median','rms','75p','90p','95p','max'] 
    from the DEi ndarray along given axis and returns a dict with those values.
    """
    DE = {} 
    DEi = np.atleast_1d(DEi)
    DE['mean'] = np.mean(DEi, axis = axis) 
    DE['median'] = np.median(DEi, axis = axis) 
    DE['rms'] = np.mean(DEi**2, axis = axis)**0.5 
    DE['75p'] = np.percentile(DEi,75, axis = axis)
    DE['90p'] = np.percentile(DEi,90, axis = axis)
    DE['95p'] = np.percentile(DEi,95, axis = axis)
    DE['max'] = np.max(DEi, axis = axis) 
    return DE


#--- models -------------------------------------------------------------------
class ColCharModel:
    def __init__(self, training_data = None, single_channel_ramp_only_data = False, 
                 cspace = 'lab', nbit = 8, xyzw = None, 
                 xyzb = None,  black_correct = True, 
                 linearize_rgb = False, tr = None, tr_type = None,  
                 tr_L_type = 'Y', tr_par_lower_bounds = (0,-0.1,0,-0.1), 
                 tr_ensure_increasing_lut_at_low_rgb = 0.2,
                 tr_force_increasing_lut_at_high_rgb = True,
                 tr_rms_break_threshold = 0.01,
                 tr_smooth_window_factor = None,
                 M = None, optimize_M = True, N = None, 
                 cieobs = '1931_2', avg = lambda x: (x**2).mean()**0.5):
        """
        Super class for the colorimetric characterization models.
        
        Args:
            :training_data:
                | None
                | (xyz,rgb) pairs with xyz and rgb ndarrays.
            :single_channel_ramp_only_data:
                | False,
                | If True: select from the training pair data the single channel ramps
                |  and construct a full new training cube. The remainder of the original
                | training cube's points are ignored.
            :cspace:
                | 'lab', optional
                | Color space to train some models (-> optimized transfer matrix for GGO,GOG,GOGO,SIGMOID) in,
                |  and to calculate the DE errors in when testing.
            :nbit:
                | 8, optional
                | RGB values in nbit format (e.g. 8, 16, ...)
            :xyzw:
                | None, optional
                | White point xyz corresponding to RGB = [2**nbit-1,2**nbit-1,2**nbit-1]
                | Used when converting to a cspace that requires a white point.
                | If None: use the one from the training set.
            :xyzb:
                | None, optional
                | Black point xyz corresponding to RGB = [0,0,0]
                | Used when doing a black-correction.
                | If None: use the one from the training set.
            :black_correct:
                | True, optional
                | If True: apply black-correction.
            :linearize_rgb:
                | False, optional
                | Apply a linearization (using tone response curves) to the device RGB before running the actual model.
            :tr:
                | None, optional
                | User supplied Tone Response curves (estimated using displcal.estimate_tr)
                | If None: the TR will be estimated using the model specified in :tr_type:.
            :tr_type:
                | None, optional
                | If None: default to 'pli'
                | options:
                |  - 'lut': Derive/specify Tone-Response as a look-up-table
                |  - 'ggo': Derive/specify Tone-Response as a gain-gamma-offset function: y = gain*x**gamma + offset
                |  - 'gog': Derive/specify Tone-Response as a gain-offset-gamma function:  y = (gain*x + offset)**gamma
                |  - 'gogo': Derive/specify Tone-Response as a gain-offset-gamma-offset function: y = (gain*x + offset)**gamma + offset
                |  - 'sigmoid': Derive/specify Tone-Response as a sigmoid function: y = offset + gain* [1 / (1 + q*exp(-a/gamma*(x - m)))]**(gamma)
                |  - 'pli': Derive/specify Tone-Response as a piecewise linear interpolation function
            :tr_L_type:
                | 'lms', optional
                | Type of response to use in the derivation of the Tone-Response curves.
                | options:
                |  - 'lms': use cone fundamental responses: L vs R, M vs G and S vs B 
                |           (reduces noise and generally leads to more accurate characterization) 
                |  - 'Y': use the luminance signal: Y vs R, Y vs G, Y vs B
            :tr_par_lower_bounds:
                | (0,-0.1,0,-0.1), optional
                | Lower bounds used when optimizing the parameters of the GGO, GOG, GOGO tone
                | response functions. Try different set of fit fails. 
                | Tip for GOG & GOGO: try changing -0.1 to 0 (0 is not default,
                |          because in most cases this leads to a less goog fit)
            :tr_ensure_increasing_lut_at_low_rgb:
                | 0.2 or float (max = 1.0) or None, optional
                | Used in dispcal.estimate_tr()
                | Ensure an increasing lut by setting all values below the RGB with the maximum
                | zero-crossing of np.diff(lut) and RGB/RGB.max() values of :tr_ensure_increasing_lut_at_low_rgb:
                | (values of 0.2 are a good rule of thumb value)
                | Non-strictly increasing lut values can be caused at low RGB values due
                | to noise and low measurement signal. 
                | If None: don't force lut, but keep as is.
            :tr_force_increasing_lut_at_high_rgb:
                | True, optional
                | Used in dispcal.estimate_tr()
                | If True: ensure the tone response curves in the lut are monotonically increasing.
                |          by finding the first 1.0 value and setting all values after that also to 1.0.
            :tr_rms_break_threshold:
                | 0.01, optional
                | Used in dispcal.estimate_tr()
                | Threshold for breaking a loop that tries different bounds 
                | for the gain in the TR optimization for the GGO, GOG, GOGO models.
                | (for some input the curve_fit fails, but succeeds on using different bounds)
            :tr_smooth_window_factor:
                | None, optional
                | Determines window size for smoothing of data using scipy's svagol_filter prior to determining the TR curves.
                | window_size = x.shape[0]//tr_smooth_window_factor 
                | If None: don't apply any smoothing
             :M:
                 | None, optional
                 | RGB-to-XYZ transfer matrix. (only used by some models)
                 | If None: it is derived automatically by the model.
             :optimize_M:
                 | True, optional
                 | If True: optimize the transfer matrix, else: use the XYZ tristimulus values of the single channel max outputs.
             :N:
                 | None, optional
                 | XYZ-to-RGB transfer matrix. (only used by some models)
                 | If None: it is derived automatically by the model.
             :cieobs:
                 | '1931_2', optional
                 | CIE CMF set used to determine the XYZ tristimulus values in dispcal.estimate_tr()
                 |   when linearizing the RGB values for tr_L_type == 'lms' (-> determines the conversion matrix to
                 |   convert xyz to lms values)
             :avg:
                 | lambda x: ((x**2).mean()**0.5), optional
                 | Used in dispcal.optimize_3x3_transfer_matrix()
                 | Function used to average the color differences of the individual RGB settings
                 | in the optimization of the xyz_to_rgb and rgb_to_xyz conversion matrices.
            
        """

              
        self.training_data = training_data
        self.single_channel_ramp_only_data = single_channel_ramp_only_data
        self.nbit = nbit
        if self.training_data is not None:
            # construct full training data cube from single channel ramp data:
            if self.single_channel_ramp_only_data:
                self.training_data = ramp_data_to_cube_data(self.training_data, nbit = self.nbit)
            
            self.rgb_train = np.clip(np.round(self.training_data[1]), 0, 2**self.nbit - 1).astype(int) 
            self.xyz_train = self.training_data[0] 
        else:
            self.rgb_train = None
            self.xyz_train = None
            
        self.cspace = cspace
        self.xyzw = xyzw
        
        self.xyzb = xyzb
        self.black_correct = black_correct
        
        self.linearize_rgb = linearize_rgb
        self.tr = tr 
        self.tr_type = tr_type if tr_type is not None else 'pli'
        self.tr_par_lower_bounds = tr_par_lower_bounds
        self.tr_L_type = tr_L_type
        self.tr_ensure_increasing_lut_at_low_rgb = tr_ensure_increasing_lut_at_low_rgb 
        self.tr_force_increasing_lut_at_high_rgb = tr_force_increasing_lut_at_high_rgb
        self.tr_rms_break_threshold = tr_rms_break_threshold
        self.tr_smooth_window_factor = tr_smooth_window_factor
        
        self.M = M
        self.optimize_M = optimize_M
        self.N = N
        self.avg = avg
        self.cieobs = cieobs 

        
        self.models = {'fw' : None, 'bw' : None} # fw: rgb-to-xyz, bw: xyz-to-rgb
        
    def train(self, training_data = None, single_channel_ramp_only_data = False):
        pass 
    
    def to_rgb(self, xyz):
        pass
    
    def to_xyz(self, rgb):
        pass
    
    def test(self, test_data, xyz_test_measured = None,  virtual_display = None, cspace = 'lab'):
        self.xyz_test = test_data[0]
        if test_data[1] is not None: 
            self.rgb_test = np.clip(test_data[1], 0, 2**self.nbit - 1).astype(int)
        else:
            self.rgb_test = None
        
        # estimate rgb_test data for xyz_test using color characterization model:
        self.rgb_test_estimate = np.clip(np.round(self.to_rgb(self.xyz_test)),0,2**self.nbit-1).astype(int)
        
        # simulate measured xyz using virtual display model:
        if (xyz_test_measured is None) & (virtual_display is not None):
            self.xyz_test_measured = virtual_display.to_xyz(self.rgb_test_estimate)
        elif (xyz_test_measured is not None):
            self.xyz_test_measured = xyz_test_measured
        else:
            raise Exception('When no xyz_test_measured is provided, the virtual_diplay_pars must be !')
            
        # convert to cspace:
        xyzw = self.xyzw 
        if (xyzw is None) & (self.cspace[0] == 'l'): raise Exception("Model doesn't initiate xyzw.") # force xyzw to exist and not be None if cspace == 'lab' or 'luv'!
        self.lab_test = colortf(self.xyz_test, tf = 'xyz>'+cspace, xyzw = xyzw)
        self.lab_test_measured = colortf(self.xyz_test_measured, tf = 'xyz>'+cspace, xyzw = xyzw)
        
        # Calculate color differences:
        if self.cspace[0] == 'l': 
            self.DEi = ((self.lab_test - self.lab_test_measured)**2).sum(-1)**0.5
        elif self.cspace[0] == 'Y':
            self.DYi = ((self.lab_test - self.lab_test_measured)[:,:1]**2).sum(-1)**0.5
            self.DExyi = ((self.lab_test - self.lab_test_measured)[:,1:]**2).sum(-1)**0.5
            self.DEi = self.DYi/100*0.6 + self.DExyi # DY: 1/100 to from Y=[0-100] range to [0-1] and further divide by 3 to go to approximately [0-0.6] (should be further verified)
        
        # calculate mean, median, 95-percentile, max:
        self.DE = get_DE_stats(self.DEi)

#------------------------------------------------------------------------------ 
class GGO_GOG_GOGO_PLI(ColCharModel):
    
    def __init__(self, training_data = None, single_channel_ramp_only_data = False, cspace = 'lab', nbit = 8, 
                 xyzw = None, xyzb = None, black_correct = True, 
                 tr = None, tr_type = None, tr_L_type = 'Y',
                 tr_par_lower_bounds = (0,-0.1,0,-0.1),  
                 M = None, optimize_M = True, N = None, 
                 cieobs = '1931_2', avg = lambda x: (x**2).mean()**0.5,
                 tr_ensure_increasing_lut_at_low_rgb = 0.2,
                 tr_force_increasing_lut_at_high_rgb = True,
                 tr_rms_break_threshold = 0.01,
                 tr_smooth_window_factor = None):
        """ 
        class for characterization models that combine a 3x3 transfer matrix 
        and a GGO, GOG, GOGO, SIGMOID, PLI and 1-D LUT Tone response curve  
        
        |  - Tone Response curve models:
        |    * GGO: gain-gamma-offset model: y = gain*x**gamma + offset
        |    * GOG: gain-offset-gamma model: y = (gain*x + offset)**gamma
        |    * GOG: gain-offset-gamma-offset model: y = (gain*x + offset)**gamma + offset
        |    * SIGMOID: sigmoid (S-shaped) model: y = offset + gain* [1 / (1 + q*exp(-a/gamma*(x - m)))]**(gamma)
        |    * PLI: Piece-wise Linear Interpolation
        |    * LUT: 1-D Look-Up-Tables for the TR
        |  - RGB-to-XYZ / XYZ-to-RGB transfer matrices:
        |     * M fixed: derived from tristimulus values of maximum single channel output
        |     * M optimized: by minimizing the RMSE between measured and predicted XYZ values
        
        Args:
            :tr_type:
                | None -> needs to be set by user as it specifies the type of model.
                | If kept at None it will default to the 'pli' model (= default of super class).
        
        For info on additional arguments: do "print(ColCharModel.__init__.__doc__)"
        """
        
        super().__init__(training_data = training_data, 
                         single_channel_ramp_only_data = single_channel_ramp_only_data,
                         cspace = cspace, nbit = nbit,
                         xyzw = xyzw, xyzb = xyzb,  black_correct = black_correct, 
                         linearize_rgb = True, tr_par_lower_bounds = tr_par_lower_bounds,
                         tr = tr, tr_type = tr_type, tr_L_type = tr_L_type, 
                         M = M, optimize_M = optimize_M, N = N, cieobs = cieobs, avg = avg,
                         tr_ensure_increasing_lut_at_low_rgb = tr_ensure_increasing_lut_at_low_rgb,
                         tr_force_increasing_lut_at_high_rgb = tr_force_increasing_lut_at_high_rgb,
                         tr_rms_break_threshold = tr_rms_break_threshold,
                         tr_smooth_window_factor = tr_smooth_window_factor
                         )
        
        self.mode = ['bw','fw'] # both modes are automatically available               
        if self.training_data is not None: 
            self.train()
        
        
    def train(self, training_data = None, single_channel_ramp_only_data = None, EPS = 1e-300):
        if training_data is not None: self.training_data = training_data 
        if single_channel_ramp_only_data is not None: self.single_channel_ramp_only_data = single_channel_ramp_only_data
        if self.training_data is not None:
            
            # construct full training data cube from single channel ramp data:
            if self.single_channel_ramp_only_data:
                self.training_data = ramp_data_to_cube_data(self.training_data, nbit = self.nbit)
            
            self.rgb_train = np.clip(np.round(self.training_data[1]), 0, 2**self.nbit - 1).astype(int) 
            self.xyz_train = self.training_data[0] 
            
            M_opt, N_opt, tr, xyzb, xyzw = dc.calibrate(self.rgb_train, self.xyz_train, 
                                                            black_correct = self.black_correct, 
                                                            tr_type = self.tr_type, tr_L_type = self.tr_L_type, 
                                                            tr_par_lower_bounds = self.tr_par_lower_bounds, 
                                                            avg = self.avg, 
                                                            cspace = self.cspace,
                                                            optimize_M = self.optimize_M,
                                                            cieobs = self.cieobs,
                                                            nbit = self.nbit,
                                                            tr_ensure_increasing_lut_at_low_rgb = self.tr_ensure_increasing_lut_at_low_rgb,
                                                            tr_force_increasing_lut_at_high_rgb = self.tr_force_increasing_lut_at_high_rgb,
                                                            tr_rms_break_threshold = self.tr_rms_break_threshold,
                                                            tr_smooth_window_factor = self.tr_smooth_window_factor,
                                                            verbosity = 0)
        else:
            raise Exception('To train the model training_data = (XYZ_train, RGB_train) must be provided!')
        
        if self.xyzw is None: self.xyzw = xyzw 
        if self.xyzb is None: self.xyzb = xyzb
        if self.M is None: self.M = M_opt
        if self.N is None: self.N = N_opt
        if self.tr is None: self.tr = tr
    
    def to_rgb(self, xyz):
        return np.clip(np.round(dc.xyz_to_rgb(xyz, self.N, self.tr, self.xyzb, tr_type = self.tr_type, nbit = self.nbit)), 0, 2**self.nbit - 1).astype(int)
    
    def to_xyz(self, rgb):
        rgb = np.clip(rgb, 0, 2**self.nbit - 1).astype(int)
        return dc.rgb_to_xyz(rgb, self.M, self.tr, self.xyzb, tr_type = self.tr_type, nbit = self.nbit)
    
#------------------------------------------------------------------------------
class ML(ColCharModel):
    def __init__(self, training_data = None, single_channel_ramp_only_data = False, cspace = 'lab', nbit = 8,
                 xyzw = None, xyzb = None,  black_correct = False, 
                 linearize_rgb = False, tr_par_lower_bounds = (0,-0.1,0,-0.1), 
                 tr_L_type = 'Y', tr_type = 'pli', cieobs = '1931_2',
                 tr_ensure_increasing_lut_at_low_rgb = 0.2, 
                 tr_force_increasing_lut_at_high_rgb = True,
                 tr_rms_break_threshold = 0.01,
                 tr_smooth_window_factor = None,
                 mode = ['bw']):
        
        """ 
        Super class for Machine Learning (sklearn) based methods.
        
        Args:
            :mode:
                | ['bw'], optional
                | Model(s) to train: 'bw' -> XYZ-to-RGB; 'fw' -> RGB-to-XYZ.
        
        For info on additional arguments: do "print(ColCharModel.__init__.__doc__)"
        
        """
        
        super().__init__(training_data = training_data, 
                         single_channel_ramp_only_data = single_channel_ramp_only_data,
                         cspace = cspace, nbit = nbit,
                         xyzw = xyzw, xyzb = xyzb,  black_correct = black_correct, 
                         linearize_rgb = linearize_rgb, tr_par_lower_bounds = tr_par_lower_bounds,
                         tr = None, tr_type = tr_type, tr_L_type = tr_L_type, 
                         M = None, optimize_M = False, N = None, cieobs = cieobs, avg = None,
                         tr_ensure_increasing_lut_at_low_rgb = tr_ensure_increasing_lut_at_low_rgb,
                         tr_force_increasing_lut_at_high_rgb = tr_force_increasing_lut_at_high_rgb,
                         tr_rms_break_threshold = tr_rms_break_threshold,
                         tr_smooth_window_factor = tr_smooth_window_factor
                         )

        self.mode = mode
        self.models = {'fw' : None, 
                       'bw' : None} # fw: rgb-to-xyz, bw: xyz-to-rgb
        self.to_rgb_kwargs = {}
        self.to_xyz_kwargs = {}
    
    def train(self, training_data = None, single_channel_ramp_only_data = None, mode = None):
        if training_data is not None: self.training_data = training_data 
        if single_channel_ramp_only_data is not None: self.single_channel_ramp_only_data = single_channel_ramp_only_data
        if mode is not None: self.mode = list(set(self.mode + mode)) # add mode to available modes
        if self.training_data is not None:
            
            # construct full training data cube from single channel ramp data:
            if self.single_channel_ramp_only_data:
                self.training_data = ramp_data_to_cube_data(self.training_data, nbit = self.nbit)
            
            self.rgb_train = np.clip(np.round(self.training_data[1]),0,2**self.nbit-1).astype(int) 
            self.xyz_train = self.training_data[0] 
        
        # get measured white point from training data:
        self.xyzw = self.rgb_train[dc.find_index_in_rgb(self.rgb_train, k = [2**self.nbit-1]*3)].mean(axis = 0, keepdims = True)
        
        # black_correct xyz:
        if self.black_correct:
            self.xyz_train_blackcorrected, self.xyzb = dc.correct_for_black(self.xyz_train, self.rgb_train, xyz_black = self.xyzb)
        else:
            self.xyz_train_blackcorrected, self.xyzb = self.xyz_train, np.array([[0,0,0]])
        
        # linearize rgb (or not):
        if self.linearize_rgb:
            # Estimate tone response curves:
            self.tr, _, _ = dc.estimate_tr(self.rgb_train, self.xyz_train_blackcorrected, 
                                           black_correct = False, # already done above
                                           xyz_black = None,
                                           tr_L_type = self.tr_L_type, tr_type = self.tr_type, 
                                           tr_par_lower_bounds = self.tr_par_lower_bounds,
                                           cieobs = self.cieobs, nbit = self.nbit, 
                                           tr_ensure_increasing_lut_at_low_rgb = self.tr_ensure_increasing_lut_at_low_rgb, 
                                           tr_force_increasing_lut_at_high_rgb = self.tr_force_increasing_lut_at_high_rgb,
                                           tr_smooth_window_factor = self.tr_smooth_window_factor,
                                           verbosity = 0)
            # determine linear rgb:
            self.rgb_train_lin = dc._rgb_linearizer(self.rgb_train, self.tr, tr_type = self.tr_type, nbit = self.nbit)
        
        else:
            self.rgb_train_lin = self.rgb_train
        
        # train aall requested modes:
        for mode in self.mode: 
            if mode == 'bw': 
                self.models['bw'] = self.pipe_bw.fit(self.xyz_train_blackcorrected, self.rgb_train_lin)
            elif mode == 'fw': 
                self.models['fw'] = self.pipe_fw.fit(self.rgb_train_lin, self.xyz_train_blackcorrected)
    
    def predict(self, x, mode, **kwargs):
        return self.models[mode].predict(x,**kwargs)
    
    def to_rgb(self, xyz):
        if self.models['bw'] is not None:
            if self.black_correct: xyz = xyz - self.xyzb 
            rgb = self.predict(xyz, 'bw', **self.to_rgb_kwargs)
            if self.linearize_rgb: 
                rgb = np.clip(rgb,0,1)
                rgb = dc._rgb_delinearizer(rgb, self.tr, tr_type = self.tr_type, nbit = 8)
            rgb = np.clip(np.round(rgb),0,2**self.nbit-1).astype(int)
            return rgb
        else:
            raise Exception('Backward xyz-to-rgb model not trained.')
        
    def to_xyz(self, rgb):
        if self.models['fw'] is not None: 
            rgb = np.clip(rgb,0,2**self.nbit-1).astype(int)
            if self.linearize_rgb: rgb = dc._rgb_linearizer(rgb, self.tr, tr_type = self.tr_type, nbit = 8)
            xyz = self.predict(rgb, 'fw', **self.to_xyz_kwargs)
            if self.black_correct: xyz = xyz + self.xyzb 
            xyz[xyz<0] = 0
            return xyz
        else:
            raise Exception('Forward rgb-to-xyz model not trained.')
    
    
#------------------------------------------------------------------------------   
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPRegressor 
from sklearn.pipeline import make_pipeline  
class MLPR(ML):
    def __init__(self, training_data = None, single_channel_ramp_only_data = False, cspace = 'lab', nbit = 8,
                 xyzw = None, xyzb = None,  black_correct = False, 
                 linearize_rgb = False, tr_par_lower_bounds = (0,-0.1,0,-0.1), 
                 tr_L_type = 'Y', tr_type = 'pli', cieobs = '1931_2',
                 tr_ensure_increasing_lut_at_low_rgb = 0.2, 
                 tr_force_increasing_lut_at_high_rgb = True, 
                 tr_rms_break_threshold = 0.01,
                 tr_smooth_window_factor = None,
                 mode = ['bw'],
                 use_StandardScaler = True, 
                 number_of_hidden_layers = 500, activation = 'relu',
                 max_iter = 100000, tol = 1e-4, learning_rate = 'adaptive',
                 **kwargs):
        
        """ 
        Class for Multi-Layer Perceptron Regressor based model.
        
        Args:
            :use_StandardScaler:
                | True, optional
                | If True: apply sklearn's StandardScaler() 
                |   to "standardize features by removing the mean and scaling to unit variance".
            :number_of_hidden_layers:
                | 500, optional
                | Number of hidden layers in a fully connected Neural Net.
            :activation:
                | 'relu', optional
                | Activation function for the hidden layers in the neural network.
            :max_iter:
                | 100000, optional
                | maximum number of iterations
            :tol:
                | 1e-4, optional
                | Tolerance for the optimization.
            :learning_rate:
                | 'adaptive', optional
                | Learning rate schedule for weight updates.
            :kwargs:
                | other keyword arguments in sklearns MLPRegressor.  
            :mode:
                | ['bw'], optional
                | Model(s) to train: 'bw' -> XYZ-to-RGB; 'fw' -> RGB-to-XYZ.
        
        For info on MLPRegressor arguments, see its __doc__
        
        For info on other additional arguments: 
            - do "print(ColCharModel.__init__.__doc__)"
        
        """
        
        super().__init__(training_data = training_data, 
                         single_channel_ramp_only_data = single_channel_ramp_only_data,
                         cspace = cspace, nbit = nbit,
                         xyzw = xyzw, xyzb = xyzb,  black_correct = black_correct, 
                         linearize_rgb = linearize_rgb, tr_par_lower_bounds = tr_par_lower_bounds,
                         tr_type = tr_type, tr_L_type = tr_L_type, cieobs = cieobs, 
                         tr_ensure_increasing_lut_at_low_rgb = tr_ensure_increasing_lut_at_low_rgb,
                         tr_force_increasing_lut_at_high_rgb = tr_force_increasing_lut_at_high_rgb,
                         tr_rms_break_threshold = tr_rms_break_threshold,
                         tr_smooth_window_factor = tr_smooth_window_factor,
                         mode = mode)
        
        self.use_StandardScaler = use_StandardScaler
        self.number_of_hidden_layers = number_of_hidden_layers
        self.activation = activation
        self.max_iter = max_iter 
        self.tol = tol
        self.learning_rate = learning_rate
        self.kwargs = kwargs
        
        mlpregressor = MLPRegressor(hidden_layer_sizes = [number_of_hidden_layers], 
                                    activation = activation, max_iter = max_iter, 
                                    tol = tol, learning_rate = learning_rate, **kwargs)
        
        pipeline = (StandardScaler(), mlpregressor) if self.use_StandardScaler else (mlpregressor,) 
        
        self.pipe_fw = make_pipeline(*pipeline)
        self.pipe_bw = make_pipeline(*pipeline)
        
        self.models = {'fw' : None, 
                       'bw' : None} # fw: rgb-to-xyz, bw: xyz-to-rgb
        
        if self.training_data is not None: 
            self.train() 
            
        self.to_rgb_kwargs = {}
        self.to_xyz_kwargs = {}
        
#------------------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression  
class POR(ML):
    def __init__(self, training_data = None, single_channel_ramp_only_data = False, cspace = 'lab', nbit = 8,
                 xyzw = None, xyzb = None,  black_correct = True, 
                 linearize_rgb = True, tr_par_lower_bounds = (0,-0.1,0,-0.1), 
                 tr_L_type = 'Y', tr_type = 'pli', cieobs = '1931_2',
                 tr_ensure_increasing_lut_at_low_rgb = 0.2, 
                 tr_force_increasing_lut_at_high_rgb = True,
                 tr_rms_break_threshold = 0.01,
                 tr_smooth_window_factor = None,
                 mode = ['bw'],
                 polyfeat_degree = 5, polyfeat_include_bias = True, polyfeat_interaction_only = False,
                 linreg_fit_intercept = False, linreg_positive = False
                 ):
        """ 
        Class for POlynomial Regression based model.
        
        Args:
            :polyfeat_degree:
                | 5, optional
                | Maximum degree of all polynomial feature combinations
                | If tuple: (min_degree, max_degree). 
                | See sklearn's PolynomialFeatures.__doc__
            :polyfeat_include_bias:
                | True, optional
                | If True: then include a bias column 
                |    (i.e. a column of ones; cfr. intercept term in a linear model).
                | See sklearn's PolynomialFeatures.__doc__
            :polyfeat_interaction_only:
                | False, optional
                | If True: only interaction features are produced:
                |    - included: `x[0]`, `x[1]`, `x[0] * x[1]`, etc.
                |    - excluded: `x[0] ** 2`, `x[0] ** 2 * x[1]`, etc.
                | See sklearn's PolynomialFeatures.__doc__
            :linreg_fit_intercept:
                | False, optional
                | If True: include an intercept in the linear regression.
                | See sklearn's LinearRegression.__doc__
            :linreg_positive:
                | False, optional
                | If True: forces the coefficients to be positive.
                | See sklearn's LinearRegression.__doc__
            :mode:
                | ['bw'], optional
                | Model(s) to train: 'bw' -> XYZ-to-RGB; 'fw' -> RGB-to-XYZ.
                
        For info on other additional arguments: 
            - do "print(ColCharModel.__init__.__doc__)"
        
        """
        
        super().__init__(training_data = training_data, 
                         single_channel_ramp_only_data = single_channel_ramp_only_data,
                         cspace = cspace, nbit = nbit,
                         xyzw = xyzw, xyzb = xyzb,  black_correct = black_correct, 
                         linearize_rgb = linearize_rgb, tr_par_lower_bounds = tr_par_lower_bounds,
                         tr_type = tr_type, tr_L_type = tr_L_type, cieobs = cieobs, 
                         tr_ensure_increasing_lut_at_low_rgb = tr_ensure_increasing_lut_at_low_rgb,
                         tr_force_increasing_lut_at_high_rgb = tr_force_increasing_lut_at_high_rgb,
                         tr_rms_break_threshold = tr_rms_break_threshold,
                         tr_smooth_window_factor = tr_smooth_window_factor,
                         mode = mode)
        
        self.polyfeat_degree = polyfeat_degree
        self.polyfeat_include_bias = polyfeat_include_bias
        self.polyfeat_interaction_only = polyfeat_interaction_only
        self.linreg_fit_intercept = linreg_fit_intercept
        self.linreg_positive = linreg_positive
            
        self.pipe_fw = Pipeline([('poly', PolynomialFeatures(degree = self.polyfeat_degree, 
                                                             include_bias = self.polyfeat_include_bias,
                                                             interaction_only = self.polyfeat_interaction_only)), 
                                 ('linear', LinearRegression(fit_intercept = self.linreg_fit_intercept,
                                                             positive = self.linreg_positive))
                                 ]
                                )
        self.pipe_bw = copy.deepcopy(self.pipe_fw)                         
        self.models = {'fw' : None, 
                       'bw' : None} # fw: rgb-to-xyz, bw: xyz-to-rgb
        if self.training_data is not None: 
            self.train()
        self.to_rgb_kwargs = {}
        self.to_xyz_kwargs = {}

#------------------------------------------------------------------------------
class LUT_Pipe:
    """ 
    Pipe class for LUT based models to also have fit() method 
    and one that outputs an instance with a predict() method.
    This way the ML class can be used as super class.
    """
    def __init__(self, fcn, init_kwargs = {}):
        self.fcn = fcn
        self.init_kwargs = init_kwargs
    def fit(self, x, y = None):
        if 'interpnd' in self.fcn.__module__:
            tmp = self.fcn(x, y,**self.init_kwargs)
            tmp.predict = tmp.__call__
        elif 'ckdtree' in self.fcn.__module__:
            tmp = self.fcn(x, **self.init_kwargs)
        return tmp
        
def ramp_data_to_cube_data(training_data, black_correct = True, nbit = 8):
    """ 
    Create a RGB and XYZ cube from the single channel ramps in the training data.
    
    Args:
        :training_data:
            | tuple (xyz_train, rgb_train) of ndarrays
        :black_correct:
            | True, optional
            | If True: apply black correction before creating the cubes
            | If False: the black level will be added 3 times as the XYZ of the R, G, B channels are summed)
    """
    if training_data is not None:
        rgb_train = np.clip(np.round(training_data[1]),0,2**nbit-1).astype(int) 
        xyz_train = training_data[0] 
        
        p_black = dc.find_index_in_rgb(rgb_train, k = [0,0,0])
        xyz_black = xyz_train[p_black,:].mean(axis=0,keepdims=True)
        rgb_black = rgb_train[p_black,:].mean(axis=0,keepdims=True)
        if black_correct:  xyz_train = xyz_train - xyz_black 
            
        #get rid of multiple blacks in array:
        c = np.setxor1d(p_black,np.arange(rgb_train.shape[0]).astype(int))
        rgb_train = np.vstack((rgb_black,rgb_train[c]))
        xyz_train = np.vstack((xyz_black,xyz_train[c]))
        
        # get positions of pure r, g, b values:
        p_pure = dc.find_pure_rgb(rgb_train)
        idxs = [rgb_train[p_pure[i],i].argsort() for i in range(3)]
        
        # Get rgb and xyz of red, green and blue ordered channel ramps:
        xyz_train = np.array([xyz_train[p_pure[i],:][idxs[i]] for i in range(3)])
        rgb_train = np.array([rgb_train[p_pure[i],:][idxs[i]] for i in range(3)])
        
        # create all possible combo's:
        cube_idx = np.array(list(itertools.product(*([np.arange(rgb_train.shape[1])]*3))))
        
        # create additive light mixtures of R,G,B channels:
        rgb_train_cube = rgb_train[0,cube_idx[:,0],:] + rgb_train[1,cube_idx[:,1],:] + rgb_train[2,cube_idx[:,2],:]
        xyz_train_cube = xyz_train[0,cube_idx[:,0],:] + xyz_train[1,cube_idx[:,1],:] + xyz_train[2,cube_idx[:,2],:]
        
        rgb_train_cube = np.clip(np.round(rgb_train_cube),0,2**nbit-1).astype(int)
                
        # add xyz_black:
        if black_correct:  xyz_train_cube = xyz_train_cube + xyz_black 

        return (xyz_train_cube, rgb_train_cube)
        
    else:
        return training_data

from scipy import interpolate
class LUTQHLI(ML): 
    # Remarks:
    # 1) when training data cube is obtained from all combinations of channel ramps -> analogous to PLVC model (X = linear combination of X(dr),X(dg),X(db), same for Y,Z ... ); if not it is more general
    # 2) LinearNDInterpolator performs linear barycentric interpolation on triangles from triangulation
    def __init__(self, training_data = None, single_channel_ramp_only_data =  False, cspace = 'lab', nbit = 8,
                 xyzw = None, xyzb = None,  black_correct = True, 
                 linearize_rgb = True, tr_par_lower_bounds = (0,-0.1,0,-0.1), 
                 tr_L_type = 'Y', tr_type = 'pli', cieobs = '1931_2',
                 tr_ensure_increasing_lut_at_low_rgb = 0.2, 
                 tr_force_increasing_lut_at_high_rgb = True,
                 tr_rms_break_threshold = 0.01,
                 tr_smooth_window_factor = None,
                 rescale = False,
                 mode = ['bw']):
        
        """ 
        Class for LUT-Linear-INTERPolation based models.
        
        Args:
            :rescale:
                | False, optional
                | Rescale points to unit cube before performing interpolation.
                | see scipy.interpolate.LinearNDInterpolator.__doc__
            :mode:
                | ['bw'], optional
                | Model(s) to train: 'bw' -> XYZ-to-RGB; 'fw' -> RGB-to-XYZ.
                
        For info on other additional arguments: 
            - do "print(ColCharModel.__init__.__doc__)"
            
        Notes:
            | The interpolant is constructed by triangulating the input data
            | with Qhull, and on each triangle performing linear
            | barycentric interpolation.
        """
                
        super().__init__(training_data = training_data, 
                         single_channel_ramp_only_data = single_channel_ramp_only_data,
                         cspace = cspace, nbit = nbit,
                         xyzw = xyzw, xyzb = xyzb,  black_correct = black_correct, 
                         linearize_rgb = linearize_rgb, tr_par_lower_bounds = tr_par_lower_bounds,
                         tr_type = tr_type, tr_L_type = tr_L_type, cieobs = cieobs, 
                         tr_ensure_increasing_lut_at_low_rgb = tr_ensure_increasing_lut_at_low_rgb,
                         tr_force_increasing_lut_at_high_rgb = tr_force_increasing_lut_at_high_rgb,
                         tr_rms_break_threshold = tr_rms_break_threshold,
                         tr_smooth_window_factor = tr_smooth_window_factor,
                         mode = mode)
        self.rescale = rescale
        self.pipe_fw = LUT_Pipe(interpolate.LinearNDInterpolator, {'rescale':self.rescale})
        
        self.pipe_bw = copy.deepcopy(self.pipe_fw)                         
        self.models = {'fw' : None, 
                       'bw' : None} # fw: rgb-to-xyz, bw: xyz-to-rgb
        if self.training_data is not None: 
            self.train()
        self.to_rgb_kwargs = {}
        self.to_xyz_kwargs = {}

            
#------------------------------------------------------------------------------
from scipy.spatial import cKDTree 

class LUTNNLI(ML):
    def __init__(self, training_data = None, single_channel_ramp_only_data = False, cspace = 'lab', nbit = 8, 
                 xyzw = None, xyzb = None,  black_correct = True, 
                 linearize_rgb = True, tr_par_lower_bounds = (0,-0.1,0,-0.1), 
                 tr_L_type = 'Y', tr_type = 'pli', cieobs = '1931_2',
                 tr_ensure_increasing_lut_at_low_rgb = 0.2, 
                 tr_force_increasing_lut_at_high_rgb = True,
                 tr_rms_break_threshold = 0.01,
                 tr_smooth_window_factor = None,
                 mode = ['bw'],
                 number_of_nearest_neighbours = 4,
                 **kwargs):
        
        """ 
        Class for LUT-Nearest-Neighbour-distance-weighted-Linear-Interpolation based models.
        
        Args:
            :number_of_nearest_neighbours:
                | 4, optional
                | Number of nearest neighbours in a LUT to use to do a
                | distance weighted linear interpolation.
            :kwargs:
                | optional keyword arguments for cKDTree initialization.
            :mode:
                | ['bw'], optional
                | Model(s) to train: 'bw' -> XYZ-to-RGB; 'fw' -> RGB-to-XYZ.
                
        For info on other additional arguments: 
            - do "print(ColCharModel.__init__.__doc__)"
            
        Note:
            * Training_data must be provided at initialization !
            
        """
        
        super().__init__(training_data = training_data, 
                         single_channel_ramp_only_data = single_channel_ramp_only_data,
                         cspace = cspace, nbit = nbit,
                         xyzw = xyzw, xyzb = xyzb,  black_correct = black_correct, 
                         linearize_rgb = linearize_rgb, tr_par_lower_bounds = tr_par_lower_bounds,
                         tr_type = tr_type, tr_L_type = tr_L_type, cieobs = cieobs, 
                         tr_ensure_increasing_lut_at_low_rgb = tr_ensure_increasing_lut_at_low_rgb,
                         tr_force_increasing_lut_at_high_rgb = tr_force_increasing_lut_at_high_rgb,
                         tr_rms_break_threshold = tr_rms_break_threshold,
                         tr_smooth_window_factor = tr_smooth_window_factor,
                         mode = mode)
        
        self.number_of_nearest_neighbours = number_of_nearest_neighbours
        self.kwargs = kwargs
        self.pipe_fw = LUT_Pipe(cKDTree, self.kwargs)
        
        self.pipe_bw = copy.deepcopy(self.pipe_fw)                         
        self.models = {'fw' : None, 
                       'bw' : None} # fw: rgb-to-xyz, bw: xyz-to-rgb
        if self.training_data is not None: 
            self.train()
            self.to_rgb_kwargs = {'ckdtree' : self.models['bw'], 'x_train' : self.xyz_train_blackcorrected, 'y_train' : self.rgb_train_lin}
            self.to_xyz_kwargs = {'ckdtree' : self.models['fw'], 'x_train' : self.rgb_train_lin, 'y_train': self.xyz_train_blackcorrected}
        else:
            raise Exception('LUTNNLI: training_data must be provided at initialization')
            
    def predict(self, x, mode, ckdtree=None, x_train=None, y_train=None):
        d, inds = ckdtree.query(x, k = self.number_of_nearest_neighbours)
        
        inds[inds == x_train.shape[0]] = x_train.shape[0]-1
        d[inds == x_train.shape[0]] = np.nan
        if d.ndim == 1:
            d, inds = np.atleast_2d(d).T, np.atleast_2d(inds).T
        d += 1e-100 # avoid div by zero
        w = (1.0 / d**2)[:,:,None] # inverse distance weigthing
        y = np.sum(w * y_train[inds,:], axis=1) / np.sum(w, axis=1)
        return y
    
if __name__ == '__main__':
    
    # -------------------------------------------------------------------------
    # Generate some RGB, XYZ training pairs for the model 
    #--------------------------------------------------------------------------
    from rgbtraining_xyztest_set_generation import generate_training_data
    # generate some training RGB with axes spacing of 10, inner cube spacing 30:
    rgb_tr = generate_training_data(inc = [30, 10], verbosity = 2)
    print('rgb_tr.shape: ', rgb_tr.shape)
    
     # simulate some XYZ measurements of the training data using a virtual display:
    from virtualdisplay import VirtualDisplay
    vd = VirtualDisplay(model = 'virtualdisplay_kwak2000_SII', channel_dependence = True)
    xyz_tr = vd.to_xyz(rgb_tr) # 'measure xyz for each of the displayed rgb_tr
    print('simulated xyz_tr.shape: ', xyz_tr.shape)
        
    
    # -------------------------------------------------------------------------
    # Generate some XYZ test data (all within the display gamut) for the model 
    #  (make sure black and white are present in xyzrgb_hull!)
    #--------------------------------------------------------------------------
    from rgbtraining_xyztest_set_generation import generate_test_data
    xyz_t = generate_test_data(dlab = [10,10,10], xyzrgb_hull = (xyz_tr,rgb_tr), 
                               verbosity = 2, fig = None)
    print('xyz_t.shape: ', xyz_t.shape)
    
    # -------------------------------------------------------------------------
    # Initiate and train model(s): 
    #--------------------------------------------------------------------------
    # Gain-Gamma-Offset model for Tone Response curves, with RGB-to-XYZ transfer matrix M obtained from XYZ of max. individual channel outputs: 
    print('Training ggo_Mf ...')
    ggo_Mf = GGO_GOG_GOGO_PLI(training_data = (xyz_tr, rgb_tr), tr_type = 'ggo', optimize_M = False)
    
    # Gain-Gamma-Offset model for Tone Response curves, with RGB-to-XYZ transfer matrix M optimized: 
    print('Training ggo_Mo ...')
    ggo_Mo = GGO_GOG_GOGO_PLI(training_data = (xyz_tr, rgb_tr), tr_type = 'ggo', optimize_M = True)
    
    # Gain-offset-Gamma model for Tone Response curves, with RGB-to-XYZ transfer matrix M obtained from XYZ of max. individual channel outputs: 
    print('Training gog_Mf ...')
    gog_Mf = GGO_GOG_GOGO_PLI(training_data = (xyz_tr, rgb_tr), tr_type = 'gog', optimize_M = False)
    
    # Gain-Offset-Gamma model for Tone Response curves, with RGB-to-XYZ transfer matrix M optimized: 
    print('Training gog_Mo ...')
    gog_Mo = GGO_GOG_GOGO_PLI(training_data = (xyz_tr, rgb_tr), tr_type = 'gog', optimize_M = True)

    # Gain-offset-Gamma model for Tone Response curves, with RGB-to-XYZ transfer matrix M obtained from XYZ of max. individual channel outputs: 
    print('Training gogo_Mf ...')
    gogo_Mf = GGO_GOG_GOGO_PLI(training_data = (xyz_tr, rgb_tr), tr_type = 'gogo', optimize_M = False)
    
    # Gain-Offset-Gamma model for Tone Response curves, with RGB-to-XYZ transfer matrix M optimized: 
    print('Training gogo_Mo ...')
    gogo_Mo = GGO_GOG_GOGO_PLI(training_data = (xyz_tr, rgb_tr), tr_type = 'gogo', optimize_M = True)

    # Piecewise-Linear-Interpolator model for Tone Response curves, with RGB-to-XYZ transfer matrix M obtained from XYZ of max. individual channel outputs: 
    print('Training pli_Mf ...')
    pli_Mf = GGO_GOG_GOGO_PLI(training_data = (xyz_tr, rgb_tr), tr_type = 'pli', optimize_M = False)
    
    # Gain-Offset-Gamma model for Tone Response curves, with RGB-to-XYZ transfer matrix M optimized: 
    print('Training pli_Mo ...')
    pli_Mo = GGO_GOG_GOGO_PLI(training_data = (xyz_tr, rgb_tr), tr_type = 'pli', optimize_M = True)

    # Piecewise Linear Regression model of order 6, without/with 'pli' linearization and blacklevel correction:
    print('Training por6[_bl] ...')
    por6    = POR(training_data = (xyz_tr, rgb_tr), polyfeat_degree = 6, linearize_rgb = False, black_correct = False)
    por6_bl = POR(training_data = (xyz_tr, rgb_tr), polyfeat_degree = 6, tr_type = 'pli', linearize_rgb = True, black_correct = True)
    
    # Look-Up-Table with barycentric linear interpolation using QHull, without/with 'pli' linearization and blacklevel correction:
    print('Training lutqhli[_bl] ...')
    lutqhli    = LUTQHLI(training_data = (xyz_tr, rgb_tr), tr_type = 'pli', linearize_rgb = False, black_correct = False)
    lutqhli_bl = LUTQHLI(training_data = (xyz_tr, rgb_tr), tr_type = 'pli', linearize_rgb = True, black_correct = True)
    
    # Multi-Layer Perceptron Regression model, without/with 'pli' linearization and blacklevel correction::
    print('Training mlpr[_bl] ...')
    mlpr800    = MLPR(training_data = (xyz_tr, rgb_tr), number_of_hidden_layers = 800, tr_type = 'pli', linearize_rgb = False, black_correct = False)
    mlpr800_bl = MLPR(training_data = (xyz_tr, rgb_tr), number_of_hidden_layers = 800, tr_type = 'pli', linearize_rgb = True, black_correct = True)
        
    # -------------------------------------------------------------------------
    # Test model(s): 
    #--------------------------------------------------------------------------
    def test_model(xyz_t, model, model_name, vd):
        rgb_t_predicted = model.to_rgb(xyz_t) # get model prediction of RGB values that would generate the requested xyz_t
        xyz_t_meas = vd.to_xyz(rgb_t_predicted) # 'measure' the predicted RGB using a virtual display (replace these simulated xyz with real measured ones for a real display)
        model.test((xyz_t, rgb_t_predicted), xyz_test_measured = xyz_t_meas)
        print('\n{:s}: DE summary dict = '.format(model_name), model.DE)
        
    models_to_test = {'ggo_Mf':ggo_Mf, 'ggo_Mo':ggo_Mo,
                      'gog_Mf':gog_Mf, 'gog_Mo':gog_Mo,
                      'gogo_Mf':gogo_Mf, 'gogo_Mo':gogo_Mo,
                      'por6':por6, 'por6_bl':por6_bl, 
                      'lutqhli':lutqhli, 'lutqhli_bl':lutqhli_bl,
                      'mlpr800':mlpr800,'mlpr800_bl':mlpr800_bl}
    
    for model_name, model in models_to_test.items():
        test_model(xyz_t, model, model_name, vd)
    