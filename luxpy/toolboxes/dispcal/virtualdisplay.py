# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:10:13 2022

@author: u0032318
"""
import copy
import numpy as np 

from luxpy import colortf
from luxpy.utils import put_args_in_db


from luxpy.toolboxes.dispcal.displaycalibration import TR_ggo, TRi_ggo, TR_gog, TRi_gog, TR_gogo, TRi_gogo, TR_sigmoid, TRi_sigmoid

__all__ = ['virtualdisplay', '_VIRTUALDISPLAY_PARS',
           'virtualdisplay_kwak2000','_VIRTUALDISPLAY_KWAK2000_PARS',
           'VirtualDisplay']

#==============================================================================
# General model for GGO, GOG, GOGO, SIGMOID with RGB-to-XYZ transfer matrix M
#==============================================================================
_VIRTUALDISPLAY_PARS = {'nbit' : 8,
                       'tr_type' : 'ggo', 
                       'gain' : [1.0, 1.0, 1.0],
                       'offset' : [0.0, 0.0, 0.0],
                       'gamma' : [2.3, 2.4, 2.2],
                       'offset_gogo' : [0.0, 0.0, 0.0],
                       'sigmoid_m' : [0.7,0.7,0.7],
                       'sigmoid_q' : [1,1,1],
                       'sigmoid_a' : [10,10,10],
                       
                       # srgb matrix for D65
                       'M' : np.array([[0.4124564,  0.3575761,  0.1804375],
                                       [0.2126729,  0.7151522,  0.0721750],
                                       [0.0193339,  0.1191920,  0.9503041]]),
                       
                       'cspace_noise' : 'lab', 
                       'sigma_noise' : None, 
                       'seed' : None,
                       'channel_dependence' : False}


def virtualdisplay(x, forward = True, nbit = 8, tr_type = 'ggo',
                   M = None, gamma = None, gain = None, offset = None, offset_gogo = None,
                   sigmoid_m = None, sigmoid_a = None, sigmoid_q = None, 
                   cspace_noise = 'lab', sigma_noise = None, seed = None, **kwargs):
    """ 
    Simulate a virtual display using a GGO, GOG, GOGO or SIGMOID model. 
    
    Args:
        :x:
            | ndarray with RGB (forward mode) or XYZ (backward mode) values.
        :forward:
            | True, optional
            | If True: convert x input (=rgb) to xyz
            | If False: convert x input (=xyz) to rgb
        :nbit:
            | 8, optional
            | RGB values in nbit format (e.g. 8, 16, ...)
        :tr_type:
            | 'ggo', optional
            | Type of Tone Response curve.
            | options: 'ggo','gog','gogo','sigmoid'
        :gain:
            | None, optional
            | Gain in virtual display model. (None uses defaults of model)
        :offset:
            | None, optional
            | Offset in virtual display model. (None uses defaults of model)
        :gamma:
            | None, optional
            | Gammma in virtual display model. (None uses defaults of model)
        :offset_gogo:
            | None, optional 
            | Second offset in GOGO model. (None uses defaults of model)
        :sigmoid_m,sigmoid_a,sigmoid_q:
            | None, optional
            | Additional parameters for the SIGMOID model.
            | - sigmoid_m (>0): controls position of half-max (exact location depends on other a and q !)
            | - sigmoid_a (>0): growth rate (steepness) of sigmoid 
            | - sigmoid_q (>0): controls position of half-max (exact location depends on other a and m !)
        :M:
            | None, optional
            | RGB-to-XYZ transfer matrix in virtual display model. (None uses defaults of model)
        :cspace_noise:
            | 'lab', optional
            | Color space to add Gaussian multivariate noise in.
        :sigma_noise:
            | None, optional
            | Sigma of multivariate random noise that is added in cspace (same sigma for all axes).
            | If None: no noise is added.
        :seed:
            | None, optional
            | Seed for setting the state of numpy's random number generator.
            
    Returns:
        :xyz or rgb:
            | ndarray with XYZ values (forward mode) or RGB (backward mode) values.
    """
    # Make a copy of the defaults and replace with not-None kwargs:
    args = locals().copy()
    pars = copy.deepcopy(_VIRTUALDISPLAY_PARS)
    pars = put_args_in_db(pars, args)

    # extract variables from dict:
    (M, channel_dependence, cspace_noise, gain, gamma, nbit, offset, offset_gogo,
     seed, sigma_noise, sigmoid_a, sigmoid_m, sigmoid_q, tr_type) = [pars[x] for x in sorted(list(pars.keys()))]
    
    if tr_type is None: tr_type = 'ggo'
    gain = np.array(gain)*np.ones((3,))
    offset = np.array(offset)*np.ones((3,))
    gamma = np.array(gamma)*np.ones((3,))
    offset_gogo = np.array(offset_gogo)*np.ones((3,))
    sigmoid_m = np.array(sigmoid_m)*np.ones((3,))
    sigmoid_q = np.array(sigmoid_q)*np.ones((3,))
    sigmoid_a = np.array(sigmoid_a)*np.ones((3,))
        
    # select tone response curve model:
    if tr_type == 'ggo':
        TR,TRi = TR_ggo, TRi_ggo
        p = [gain,offset,gamma]
    elif tr_type == 'gog':
        TR,TRi = TR_gog, TRi_gog
        p = [gain,offset,gamma]
    elif tr_type == 'gogo':
        TR,TRi = TR_gogo, TRi_gogo
        p = [gain,offset,gamma,offset_gogo]
    elif tr_type == 'sigmoid':
        TR,TRi = TR_sigmoid, TRi_sigmoid
        p = [gain, offset, gamma, sigmoid_m, sigmoid_a, sigmoid_q]
    else:
        raise Exception('Unknown tr_type: {:s}'.format(tr_type))    
    p = np.array(p).T

    rgb_lin_white = np.array([TR(np.array([1]),*p[i]) for i in range(3)]).T # linear rgb for white, cfr. rgb = [255,255,255]/255

    if forward: # rgb to xyz
        x = np.clip(x,0,2**nbit-1).astype(dtype = np.int32)
        rgb_lin = np.array([TR(x.T[i]/(2**nbit-1),*p[i]) for i in range(3)]).T
        rgb_lin = rgb_lin/rgb_lin_white
    
        xyz = 100*(M @ rgb_lin.T).T

        if sigma_noise is not None:
            if cspace_noise is None: cspace_noise = 'lab' # default 
            
            xyzw = 100*(M @ (rgb_lin_white/rgb_lin_white).T).T
            lab = colortf(xyz, tf = 'xyz>' + cspace_noise, xyzw = xyzw)
            
            # add noise:
            np.random.seed(seed)
            noise = np.random.multivariate_normal([0,0,0], np.diag([1,1,1])*sigma_noise**2, size = lab.shape[0])
            
            lab += noise
            lab[:,0] = np.clip(lab[:,0],0,100) # no negative L* values, no L* values > 100 
            xyz = colortf(lab, tf = cspace_noise + '>xyz', xyzw = xyzw)
            
            xyz[xyz<0] = 0 
            
        return xyz
    
    else: # xyz to rgb
        N = np.linalg.inv(M)
        rgb_lin = (N @ (x/100).T).T * rgb_lin_white 
        rgb = np.array([TRi(rgb_lin.T[i],*p[i])  for i in range(3)]).T * (2**nbit-1)
        rgb = np.clip(np.round(rgb),0,2**nbit-1).astype(dtype = np.int32)
        return rgb

#==============================================================================
# Virtual display based on data and model presented in literature (Kwak et al. 2001)
#==============================================================================

# Kwak 2000 :
_VIRTUALDISPLAY_KWAK2000_PARS = {'xyzb' : np.array([[0.38,0.47,0.55]]),
                                'xyzw' : np.array([[114.6,137.5,134.1]]),
                                'xyz_rgb_max' : np.array([[33.45,18.10,0.66],
                                                          [57.47,112.0,5.47],
                                                          [23.99,8.15,130.1]]),
    
                                # Channel interdependency matrix:
                                'T' : np.array([[-0.0023,1.0033,-0.0011,0.0032,0.0004,-0.0019,-0.0043,0.0030],
                                              [-0.0008,0.0008,1.0011,0.0005,-0.0012,-0.0007,-0.0006,0.0009],
                                              [0.0000,0.0002,0.002,1.000,-0.0021,-0.0002,-0.0002,0.0023]]),
                    
                                # S matrix: 3 x 3 matrix which defines the dominant linear relationship 
                                # between monitor luminance levels and output CIE tristimulus values:
                                'S' : np.array([[0.241,0.417,0.172],[0.129,0.814,0.056],[0.001,0.036,0.945]]),
                    
                                # S-shapeII model parameters (for TR):
                                'S_shapeII_A' : np.array([[3.394,-0.030,0.016],[0,2.550,-0.007],[0,0.002,2.203]]),
                                'S_shapeII_abC' : np.array([[3.308,3.157,3.118],[10.783,7.166,7.956],[2.394,1.551,1.204]]),
                                
                                'channel_dependence' : True,
                                'normalize_Ywhite_to_100' : True,
                                
                                'cspace_noise' : 'lab', 
                                'sigma_noise' : None, 
                                'seed' : None,
                                }
    
# function definitions:
_get_D_1x8 = lambda rgb: np.vstack((np.ones((rgb.shape[0],)),
                                     rgb.T,
                                     rgb.T[[0,1]].prod(0),
                                     rgb.T[[1,2]].prod(0),
                                     rgb.T[[0,2]].prod(0),
                                     rgb.T.prod(0)))

# Tone response curves: convert rgb to linear rgb
_f_kwak2000 = lambda x, abC: x**abC[0] / (x**abC[1] + abC[2]) # S-shaped function
_fp_kwak2000 = lambda x, abC: ((abC[0] - abC[1])*x**(abC[0]+abC[1]-1) + abC[0]*abC[2]*x**(abC[0]-1))  / (x**abC[1] + abC[2])**2  # first deriv. of S-Shaped function
   
def _TR_SII_kwak2000(rgb,  A, abC, nbit = 8):
    dac = rgb/(2**nbit - 1)
    f_RGB = _f_kwak2000(dac,abC)
    fp_RGB = _fp_kwak2000(dac,abC)
    R = A[0,0]*f_RGB[:,0]  + A[0,1]*fp_RGB[:,1] + A[0,2]*fp_RGB[:,2]
    G = A[1,0]*fp_RGB[:,0] + A[1,1]*f_RGB[:,1]  + A[1,2]*fp_RGB[:,2]
    B = A[2,0]*fp_RGB[:,0] + A[2,1]*fp_RGB[:,1] + A[2,2]*f_RGB[:,2]
    RGB = np.vstack((R,G,B)).T
    return RGB

def virtualdisplay_kwak2000(rgb, channel_dependence = True, forward = True, nbit = 8, 
                            cspace_noise = 'lab', sigma_noise = None, seed = None, 
                            normalize_Ywhite_to_100 = True, verbosity = 0,**kwargs):
    """
    Virtual display based on the data and model published in Kwak et al. (2000)
    
    Args:
        :rgb:
            | ndarray with device RGB.
        :channel_dependence:
            | True, optional
            | If True: assume channel dependence.
        :forward:
            | True, optional
            | If True: convert rgb to xyz
            | If False: error is raised as the Shape-II TR model is not analytically invertible.
        :nbit:
            | 8, optional
            | RGB values in nbit format (e.g. 8, 16, ...)
        :cspace_noise:
            | 'lab', optional
            | Color space to add Gaussian multivariate noise in.
        :sigma_noise:
            | None, optional
            | Sigma of multivariate random noise that is added in cspace (same sigma for all axes).
            | If None: no noise is added.
        :seed:
            | None, optional
            | Seed for setting the state of numpy's random number generator.
        :verbosity:
            | 0, optional
            | Level of output.
            
    Returns:
        :xyz:
            | ndarray with XYZ tristimulus values.
    
    Reference:
        1. Y. Kwak & L. MacDonald. (2000). Characterisation of a desktop LCD projector, Displays 21, 179-194
    """

    kwak2000_pars = _VIRTUALDISPLAY_KWAK2000_PARS
    # Normalize and linearize device rgb:
    if forward:
        rgb_lin = _TR_SII_kwak2000(rgb, kwak2000_pars['S_shapeII_A'], kwak2000_pars['S_shapeII_abC'], nbit = nbit)
        
        if verbosity > 0: 
            import matplotlib.pyplot as plt # lazy import
            plt.plot(rgb[:,0],rgb_lin[:,0],'r')
            plt.plot(rgb[:,1],rgb_lin[:,1],'g--')
            plt.plot(rgb[:,2],rgb_lin[:,2],'b:')
    
        # Convert linear rgb to xyz:
        if (channel_dependence) & (kwak2000_pars['T'] is not None):
            rgb_lin = (kwak2000_pars['T'] @ _get_D_1x8(rgb_lin)).T
        xyz = (kwak2000_pars['S'] @ rgb_lin.T).T * kwak2000_pars['xyz_rgb_max'][:,1:2].sum() + kwak2000_pars['xyzb']

        if normalize_Ywhite_to_100 | (sigma_noise is not None):
            
            rgb_lin_white = _TR_SII_kwak2000(np.array([[2**nbit-1,2**nbit-1,2**nbit-1]]), kwak2000_pars['S_shapeII_A'], kwak2000_pars['S_shapeII_abC'], nbit = nbit)
            if (channel_dependence) & (kwak2000_pars['T'] is not None):
                rgb_lin_white = (kwak2000_pars['T'] @ _get_D_1x8(rgb_lin_white)).T
            xyzw = (kwak2000_pars['S'] @ rgb_lin_white.T).T * kwak2000_pars['xyz_rgb_max'][:,1:2].sum() + kwak2000_pars['xyzb']

            if normalize_Ywhite_to_100:
                xyz = 100*xyz/xyzw[0,1]
                xyzw = 100*xyzw/xyzw[0,1]
    
        # add noise:

        if sigma_noise is not None:
            
            if cspace_noise is None: cspace_noise = 'lab' # default 
            
            lab = colortf(xyz, tf = 'xyz>' + cspace_noise, fwtf = {'xyzw' : xyzw})
            
            # add noise:
            np.random.seed(seed)
            noise = np.random.multivariate_normal([0,0,0], np.diag([1,1,1])*sigma_noise**2, size = lab.shape[0])

            lab += noise
            lab[:,0] = np.clip(lab[:,0],0,100) # no negative L* values, no L* values > 100 
            xyz = colortf(lab, tf = cspace_noise + '>xyz', bwtf = {'xyzw' : xyzw})
            xyz[xyz<0] = 0 
            
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.plot(lab[:,1],lab[:,2],lab[:,0],'x')
            # ax.set_xlabel('a*')
            # ax.set_ylabel('b*')
            # ax.set_zlabel('L*')
            # return xyz
    
    
        xyz[xyz < 0] = 0
        
        return xyz
    else:
        raise Exception('Inverse model not implemented: S-shape II is not analytically invertible !')

#==============================================================================
# VirtualDisplay class definition
#==============================================================================
class VirtualDisplay:
    def __init__(self, model = 'kwak2000_SII', seed = -1, nbit = None, 
                 channel_dependence = None, **model_pars):
        if isinstance(model, str):
            if (model == 'virtualdisplay_kwak2000_SII') | (model == 'kwak2000_SII'):
                model = virtualdisplay_kwak2000
                self.model_name = 'kwak2000_SII'
            elif (model == 'virtualdisplay') | (model == 'GGO_GOG_GOGO'):
                model = virtualdisplay 
                self.model_name = 'GGO_GOG_GOGO'
            else:
                raise Exception('Virtual display model {:s} not implemented.')
        else:
            self.model_name = 'function'
        
        self.model = model
        self.model_pars = copy.deepcopy(model_pars)
        self.seed = seed 
        self.nbit = nbit
        self.channel_dependence = channel_dependence 
    
        # replace in model_pars:
        if (self.seed is not None):
            if self.seed >= 0: 
                self.model_pars['seed'] = self.seed
            else:
                self.seed = self.model_pars['seed'] if 'seed' in self.model_pars else 0  
        else:
            self.model_pars['seed'] = self.seed
        if nbit is not None: 
            self.model_pars['nbit'] = self.nbit
        else:
            self.nbit = self.model_pars['nbit'] if 'nbit' in self.model_pars else 8   
        if self.channel_dependence is not None:
            self.model_pars['channel_dependence'] = self.channel_dependence
        else: 
            self.channel_dependence = self.model_pars['channel_dependence'] if 'channel_dependence' in self.model_pars else False 
        

    def to_rgb(self, xyz,**kwargs):
        #print('VD: to_rgb->model_pars',self.model_pars)
        model_pars = copy.deepcopy(self.model_pars)
        model_pars.update(kwargs)
        # print('VD: to_rgb->model_pars:', model_pars)
        return self.model(xyz, forward = False, **model_pars)
    
    def to_xyz(self, rgb,**kwargs):
        # print('VD1: to_xyz->model_pars:', self.model_pars)
        model_pars = copy.deepcopy(self.model_pars)
        model_pars.update(kwargs)
        # print('VD2: to_xyz->kwargs:', kwargs)
        # print('VD3: to_xyz->model_pars:', model_pars)
        if ('force_no_noise' in kwargs) and kwargs['force_no_noise']: 
            model_pars['sigma_noise'] = None
        # if 'sigma_noise' not in model_pars: model_pars['sigma'] = None 
        # print('VD4: to_xyz->model_pars:', model_pars)
        return self.model(rgb, forward = True, **model_pars)
        
        

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt 
    
    # # === Test virtual diplay functions =======================================
    # rgb = np.array([[255,255,255],[200,150,60]])
    # xyz = virtualdisplay(rgb, forward = True, gain = 2, gamma = 2.4, tr_type = 'ggo', seed = 0)
    # print('\nxyz', xyz)
    # rgb2 = virtualdisplay(xyz, forward = False, gain = 2, gamma = 2.4, tr_type = 'ggo', seed = 0)
    # print('\nrgb2',rgb2)
    # xyz2 = virtualdisplay(rgb2, forward = True, gain = 2, gamma = 2.4, tr_type = 'ggo',sigma_noise=0.0358, seed = 0)
    # print('\nxyz', xyz2) 
    # lab, lab2 = colortf(xyz,tf='lab',xyzw=xyz[:1]), colortf(xyz2,tf='lab',xyzw=xyz2[:1])
    # print('\nDE: ',((lab-lab2)**2).sum(-1)**0.5)
    
    # # Kwak 2001 S-shapeII model:
    # 
    # rgb_ramps = np.array([np.arange(256)]*3).T
    # xyz = virtualdisplay_kwak2000(rgb_ramps, channel_dependence = True, 
    #                               normalize_Ywhite_to_100 = True,nbit = 8, verbosity = 1)
    
    # # === Test VirtualDisplay class ===========================================
    vd1 = VirtualDisplay(model = 'virtualdisplay', gain = 2, gamma = 2.4, tr_type = 'ggo', seed = 0, nbit = 8, channel_dependence = False)
    # xyz = vd1.to_xyz(rgb)
    # print('\nxyz', xyz)
    # rgb2 = vd1.to_rgb(xyz)
    # print('\nrgb2',rgb2)
    # xyz2 = vd1.to_xyz(rgb, sigma_noise = 0.0358, seed = 0)
    # print('\nxyz', xyz2) 
    # lab, lab2 = colortf(xyz,tf='lab',xyzw=xyz[:1]), colortf(xyz2,tf='lab',xyzw=xyz2[:1])
    # print('\nDE: ',((lab-lab2)**2).sum(-1)**0.5)

    # vd1 = VirtualDisplay(model = 'virtualdisplay_kwak2000_SII', gain = 2, gamma = 2.4, tr_type = 'ggo', seed = 0, nbit = 8, channel_dependence = False)
    # xyz = vd1.to_xyz(rgb)
    # print('\nxyz', xyz)
    # # rgb2 = vd1.to_rgb(xyz)
    # # print('\nrgb2',rgb2)
    # xyz2 = vd1.to_xyz(rgb, sigma_noise = 0.0358, seed = 0)
    # print('\nxyz', xyz2) 
    # lab, lab2 = colortf(xyz,tf='lab',xyzw=xyz[:1]), colortf(xyz2,tf='lab',xyzw=xyz2[:1])
    # print('\nDE: ',((lab-lab2)**2).sum(-1)**0.5)
    
    # Plot Virtual Display Gamut:
    import itertools
    include_max, include_min = True, True
    inc, inc_offset,nbit = 10, 0, 8
    dv = np.arange(inc_offset, 2**nbit - 1, inc)
    if (dv.max() < (2**nbit-1)) & (include_max): dv = np.hstack((dv,[2**nbit-1])) # ensure max values are present
    if (dv.min() > 0) & (include_min): dv = np.hstack((0, dv)) # ensure 0 values are present
    rgb =  np.array(list(itertools.product(*[dv]*3)))

    dv = np.arange(0,256,10)
    rgb = np.array(list(itertools.product(*[dv]*3)))
    vd2 = VirtualDisplay(model = 'virtualdisplay_kwak2000_SII', seed = 1, nbit = 8, channel_dependence = True)
    vd3 = VirtualDisplay(model = 'virtualdisplay_kwak2000_SII', seed = 1, nbit = 8, channel_dependence = False)
    xyz2 = vd2.to_xyz(rgb,sigma_noise = 0.0358)
    xyz3 = vd3.to_xyz(rgb,sigma_noise = 0.0358)
    xyzw2 = vd2.to_xyz(np.array([[255,255,255]]),sigma_noise = 0.0358)
    xyzw3 = vd3.to_xyz(np.array([[255,255,255]]),sigma_noise = 0.0358)
    lab2 = colortf(xyz2,tf = 'lab',xyzw = xyzw2)
    lab3 = colortf(xyz3,tf = 'lab',xyzw = xyzw3)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(lab2[:,1],lab2[:,2],lab2[:,0],'o')
    ax.plot(lab3[:,1],lab3[:,2],lab3[:,0],'.')
    ax.set_xlabel('a*')
    ax.set_ylabel('b*')
    ax.set_zlabel('L*')
    import luxpy as lx
    lx.math.in_hull(xyzw2,xyz2)
    lx.math.in_hull(xyzw3,xyz3)
    # virtualdisplay_kwak2000(rgb,verbosity=1, normalize_Ywhite_to_100=False, channel_dependence=False)
    # print(virtualdisplay_kwak2000(rgb[-1:],verbosity=0,channel_dependence=True,normalize_Ywhite_to_100=False))
    
    
