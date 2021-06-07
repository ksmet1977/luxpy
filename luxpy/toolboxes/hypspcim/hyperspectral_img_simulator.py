# -*- coding: utf-8 -*-
"""
Module for hyper spectral image simulation
==========================================

 :_HYPSPCIM_PATH: path to module

 :_HYPSPCIM_DEFAULT_IMAGE: path + filename to default image
 
 :_CSF_NIKON_D700: Nikon D700 camera sensitivity functions
 
 :_ROUNDING: rounding of input to xyz_to_rfl() search algorithm for improved speed

 :xyz_to_rfl(): approximate spectral reflectance of xyz based on k nearest 
                neighbour interpolation of samples from a standard reflectance 
                set.

 :render_image(): Render image under specified light source spd.

 :get_superresolution_hsi(): Get a HighResolution HyperSpectral Image (super-resolution HSI) based on a LowResolution HSI and a HighResolution Color Image.

 :hsi_to_rgb(): Convert HyperSpectral Image to rgb
 
 :rfl_to_rgb(): Convert spectral reflectance functions (illuminated by spd) to Camera Sensitivity Functions.
     
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import (cat, colortf, _CIEOBS, _CIE_ILLUMINANTS, _CRI_RFL, _CIE_D65,_CIE_E,
                   spd_to_xyz, plot_color_data, math, cie_interp, getwlr, xyz_to_srgb)
from luxpy.utils import np, plt, sp, _PKG_PATH, _SEP, _EPS 

import warnings
from imageio import imsave

__all__ =['_HYPSPCIM_PATH','_HYPSPCIM_DEFAULT_IMAGE','render_image','xyz_to_rfl',
          'get_superresolution_hsi','hsi_to_rgb','rfl_to_rgb','_CSF_NIKON_D700']             

_HYPSPCIM_PATH = _PKG_PATH + _SEP + 'hypspcim' + _SEP
_HYPSPCIM_DEFAULT_IMAGE = _PKG_PATH + _SEP + 'toolboxes' + _SEP + 'hypspcim' +  _SEP + 'data' + _SEP + 'testimage1.jpg'


_ROUNDING = 6 # to speed up xyz_to_rfl search algorithm, increase if kernel dies!!!

# Nikon D700 camera sensitivity functions:
_CSF_NIKON_D700 = np.vstack((np.arange(400,710,10),
                             np.array([[0.005, 0.007, 0.012, 0.015, 0.023, 0.025, 0.030, 0.026, 0.024, 0.019, 0.010, 0.004, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,  0.000,  0.000,  0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000], 
                                       [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.002, 0.003, 0.005, 0.007, 0.012, 0.013, 0.015, 0.016, 0.017, 0.020, 0.013, 0.011, 0.009, 0.005,  0.001,  0.001,  0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003],
                                       [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.003, 0.010, 0.012,  0.013,  0.022,  0.020, 0.020, 0.018, 0.017, 0.016, 0.016, 0.014, 0.014, 0.013]])[::-1]))


def xyz_to_rfl(xyz, CSF = None, rfl = None, out = 'rfl_est', \
                 refspd = None, D = None, cieobs = _CIEOBS, \
                 cspace = 'xyz', cspace_tf = {},\
                 interp_type = 'nd', k_neighbours = 4, verbosity = 0,
                 csf_based_rgb_rounding = _ROUNDING):
    """
    Approximate spectral reflectance of xyz values based on nd-dimensional linear interpolation 
    or k nearest neighbour interpolation of samples from a standard reflectance set.
    
    Args:
        :xyz: 
            | ndarray with xyz values of target points.
        :CSF:
            | None, optional
            | RGB camera response functions.
            | If None: input :xyz: contains raw rgb (float) values. Override :cspace:
            | argument and perform estimation directly in raw rgb space!!!
        :rfl: 
            | ndarray, optional
            | Reflectance set for color coordinate to rfl mapping.
        :out: 
            | 'rfl_est' or str, optional
        :refspd: 
            | None, optional
            | Refer ence spectrum for color coordinate to rfl mapping.
            | None defaults to D65.
        :cieobs:
            | _CIEOBS, optional
            | CMF set used for calculation of xyz from spectral data.
        :cspace:
            | 'xyz',  optional
            | Color space for color coordinate to rfl mapping.
            | Tip: Use linear space (e.g. 'xyz', 'Yuv',...) for (interp_type == 'nd'),
            |      and perceptually uniform space (e.g. 'ipt') for (interp_type == 'nearest')
        :cspace_tf:
            | {}, optional
            | Dict with parameters for xyz_to_cspace and cspace_to_xyz transform.
        :interp_type:
            | 'nd', optional
            | Options:
            | - 'nd': perform n-dimensional linear interpolation using Delaunay triangulation.
            | - 'nearest': perform nearest neighbour interpolation. 
        :k_neighbours:
            | 4 or int, optional
            | Number of nearest neighbours for reflectance spectrum interpolation.
            | Neighbours are found using scipy.spatial.cKDTree
        :verbosity:
            | 0, optional
            | If > 0: make a plot of the color coordinates of original and 
            | rendered image pixels.
        :csf_based_rgb_rounding:
            | _ROUNDING, optional
            | Int representing the number of decimals to round the RGB values (obtained from not-None CSF input) to before applying the search algorithm.
            | Smaller values increase the search speed, but could cause fatal error that causes python kernel to die. If this happens increase the rounding int value.

    Returns:
        :returns: 
            | :rfl_est:
            | ndarrays with estimated reflectance spectra.
    """

    # get rfl set:
    if rfl is None: # use IESTM30['4880'] set 
        rfl = _CRI_RFL['ies-tm30']['4880']['5nm']
    
    wlr = rfl[0]
    
    # get Ref spd:
    if refspd is None:
        refspd = _CIE_ILLUMINANTS['D65'].copy()
    refspd = cie_interp(refspd, wlr, kind = 'linear') # force spd to same wavelength range as rfl
        
    # Calculate rgb values of standard rfl set under refspd:
    if CSF is None:
        # Calculate lab coordinates:
        xyz_rr, xyz_wr = spd_to_xyz(refspd, relative = True, rfl = rfl, cieobs = cieobs, out = 2)
        cspace_tf_copy = cspace_tf.copy()
        cspace_tf_copy['xyzw'] = xyz_wr # put correct white point in param. dict
        lab_rr = colortf(xyz_rr, tf = cspace, fwtf = cspace_tf_copy, bwtf = cspace_tf_copy)[:,0,:]
    else:
        # Calculate rgb coordinates from camera sensitivity functions
        rgb_rr = rfl_to_rgb(rfl, spd = refspd, CSF = CSF, wl = None)   
        lab_rr = rgb_rr
        xyz = xyz
        lab_rr = np.round(lab_rr,csf_based_rgb_rounding) # speed up search
        
    # Convert xyz to lab-type values under refspd:
    if CSF is None:
        lab = colortf(xyz, tf = cspace, fwtf = cspace_tf_copy, bwtf = cspace_tf_copy)
    else:
        lab = xyz # xyz contained rgb values !!!
        rgb = xyz
        lab = np.round(lab,csf_based_rgb_rounding) # speed up search
    
    
    
    if interp_type == 'nearest':
        # Find rfl (cfr. lab_rr) from rfl set that results in 'near' metameric 
        # color coordinates for each value in lab_ur (i.e. smallest DE):
        # Construct cKDTree:
        tree = sp.spatial.cKDTree(lab_rr, copy_data = True)
        
        # Interpolate rfls using k nearest neightbours and inverse distance weigthing:
        d, inds = tree.query(lab, k = k_neighbours )
        if k_neighbours  > 1:
            d += _EPS
            w = (1.0 / d**2)[:,:,None] # inverse distance weigthing
            rfl_est = np.sum(w * rfl[inds+1,:], axis=1) / np.sum(w, axis=1)
        else:
            rfl_est = rfl[inds+1,:].copy()
    elif interp_type == 'nd':

        rfl_est = math.ndinterp1_scipy(lab_rr, rfl[1:], lab)
            
        _isnan = np.isnan(rfl_est[:,0]) 

        if (_isnan.any()): #do nearest neigbour method for those that fail using Delaunay (i.e. ndinterp1_scipy)

            # Find rfl (cfr. lab_rr) from rfl set that results in 'near' metameric 
            # color coordinates for each value in lab_ur (i.e. smallest DE):
            # Construct cKDTree:
            tree = sp.spatial.cKDTree(lab_rr, copy_data = True)

            # Interpolate rfls using k nearest neightbours and inverse distance weigthing:
            d, inds = tree.query(lab[_isnan,...], k = k_neighbours )

            if k_neighbours  > 1:
                d += _EPS
                w = (1.0 / d**2)[:,:,None] # inverse distance weigthing
                rfl_est_isnan = np.sum(w * rfl[inds+1,:], axis=1) / np.sum(w, axis=1)
            else:
                rfl_est_isnan = rfl[inds+1,:].copy()
            rfl_est[_isnan, :] = rfl_est_isnan

    else:
        raise Exception('xyz_to_rfl(): unsupported interp_type!')
    
    rfl_est[rfl_est<0] = 0 #can occur for points outside convexhull of standard rfl set.

    rfl_est = np.vstack((rfl[0],rfl_est))
        
    if ((verbosity > 0) | ('xyz_est' in out.split(',')) | ('lab_est' in out.split(',')) | ('DEi_ab' in out.split(',')) | ('DEa_ab' in out.split(','))) & (CSF is None):
        xyz_est, _ = spd_to_xyz(refspd, rfl = rfl_est, relative = True, cieobs = cieobs, out = 2)
        cspace_tf_copy = cspace_tf.copy()
        cspace_tf_copy['xyzw'] = xyz_wr # put correct white point in param. dict
        lab_est = colortf(xyz_est, tf = cspace, fwtf = cspace_tf_copy)[:,0,:]
        DEi_ab = np.sqrt(((lab_est[:,1:3]-lab[:,1:3])**2).sum(axis=1))
        DEa_ab = DEi_ab.mean()
    elif ((verbosity > 0) | ('xyz_est' in out.split(',')) | ('rgb_est' in out.split(',')) | ('DEi_rgb' in out.split(',')) | ('DEa_rgb' in out.split(','))) & (CSF is not None):
        rgb_est = rfl_to_rgb(rfl_est[1:], spd = refspd, CSF = CSF, wl = wlr) 
        xyz_est = rgb_est
        DEi_rgb = np.sqrt(((rgb_est - rgb)**2).sum(axis=1))
        DEa_rgb = DEi_rgb.mean()

        
    if verbosity > 0:
        if CSF is None:
            ax = plot_color_data(lab[...,1], lab[...,2], z = lab[...,0], \
                            show = False, cieobs = cieobs, cspace = cspace, \
                            formatstr = 'ro', label = 'Original')
            plot_color_data(lab_est[...,1], lab_est[...,2], z = lab_est[...,0], \
                            show = True, axh = ax, cieobs = cieobs, cspace = cspace, \
                            formatstr = 'bd', label = 'Rendered')
        else:
            n = 100 #min(rfl.shape[0]-1,rfl_est.shape[0]-1)
            s = np.random.permutation(rfl.shape[0]-1)[:min(n,rfl.shape[0]-1)]
            st = np.random.permutation(rfl_est.shape[0]-1)[:min(n,rfl_est.shape[0]-1)]
            fig = plt.figure()
            ax = np.zeros((3,),dtype=np.object)
            ax[0] = fig.add_subplot(131)
            ax[1] = fig.add_subplot(132)
            ax[2] = fig.add_subplot(133,projection='3d')
            ax[0].plot(rfl[0],rfl[1:][s].T, linestyle = '-')
            ax[0].set_title('Original RFL set (random selection of all)')
            ax[0].set_ylim([0,1])
            ax[1].plot(rfl_est[0],rfl_est[1:][st].T, linestyle = '--')
            ax[0].set_title('Estimated RFL set (random selection of targets)')
            ax[1].set_ylim([0,1])
            ax[2].plot(rgb[st,0],rgb[st,1],rgb[st,2],'ro', label = 'Original')
            ax[2].plot(rgb_est[st,0],rgb_est[st,1],rgb_est[st,2],'bd', label = 'Rendered')
            ax[2].legend()
    if out == 'rfl_est':
        return rfl_est
    elif out == 'rfl_est,xyz_est':
        return rfl_est, xyz_est
    else:
        return eval(out)


def render_image(img = None, spd = None, rfl = None, out = 'img_hyp', \
                 refspd = None, D = None, cieobs = _CIEOBS, \
                 cspace = 'xyz', cspace_tf = {}, CSF = None,\
                 interp_type = 'nd', k_neighbours = 4, show = True,
                 verbosity = 0, show_ref_img = True,\
                 stack_test_ref = 12,\
                 write_to_file = None,\
                 csf_based_rgb_rounding = _ROUNDING):
    """
    Render image under specified light source spd.
    
    Args:
        :img: 
            | None or str or ndarray with float (max = 1) rgb image.
            | None load a default image.
        :spd: 
            | ndarray, optional
            | Light source spectrum for rendering
            | If None: use CIE illuminant F4
        :rfl: 
            | ndarray, optional
            | Reflectance set for color coordinate to rfl mapping.
        :out: 
            | 'img_hyp' or str, optional
            |  (other option: 'img_ren': rendered image under :spd:)
        :refspd:
            | None, optional
            | Reference spectrum for color coordinate to rfl mapping.
            | None defaults to D65 (srgb has a D65 white point)
        :D: 
            | None, optional
            | Degree of (von Kries) adaptation from spd to refspd. 
        :cieobs:
            | _CIEOBS, optional
            | CMF set for calculation of xyz from spectral data.
        :cspace:
            | 'xyz',  optional
            | Color space for color coordinate to rfl mapping.
            | Tip: Use linear space (e.g. 'xyz', 'Yuv',...) for (interp_type == 'nd'),
            |      and perceptually uniform space (e.g. 'ipt') for (interp_type == 'nearest')
        :cspace_tf:
            | {}, optional
            | Dict with parameters for xyz_to_cspace and cspace_to_xyz transform.
        :CSF:
            | None, optional
            | RGB camera response functions.
            | If None: input :xyz: contains raw rgb values. Override :cspace:
            | argument and perform estimation directly in raw rgb space!!!
        :interp_type:
            | 'nd', optional
            | Options:
            | - 'nd': perform n-dimensional linear interpolation using Delaunay triangulation.
            | - 'nearest': perform nearest neighbour interpolation. 
        :k_neighbours:
            | 4 or int, optional
            | Number of nearest neighbours for reflectance spectrum interpolation.
            | Neighbours are found using scipy.spatial.cKDTree
        :show: 
            | True, optional
            |  Show images.
        :verbosity:
            | 0, optional
            | If > 0: make a plot of the color coordinates of original and 
              rendered image pixels.
        :show_ref_img:
            | True, optional
            | True: shows rendered image under reference spd. False: shows
            |  original image.
        :write_to_file:
            | None, optional
            | None: do nothing, else: write to filename(+path) in :write_to_file:
        :stack_test_ref: 
            | 12, optional
            |   - 12: left (test), right (ref) format for show and imwrite
            |   - 21: top (test), bottom (ref)
            |   - 1: only show/write test
            |   - 2: only show/write ref
            |   - 0: show both, write test
        :csf_based_rgb_rounding:
            | _ROUNDING, optional
            | Int representing the number of decimals to round the RGB values (obtained from not-None CSF input) to before applying the search algorithm.
            | Smaller values increase the search speed, but could cause fatal error that causes python kernel to die. If this happens increase the rounding int value.


    Returns:
        :returns: 
            | img_hyp, img_ren, 
            | ndarrays with float hyperspectral image and rendered images 
    """
    
    # Get image:
    #imread = lambda x: plt.imread(x) #matplotlib.pyplot
   
    if img is not None:
        if isinstance(img,str):
            img = plt.imread(img).copy() # use matplotlib.pyplot's imread
    else:
        img = plt.imread(_HYPSPCIM_DEFAULT_IMAGE).copy()
    
    if img.dtype == np.uint8: 
        img = img/255
    elif img.dtype == np.uint16:
        img = img/(2**16-1)
    elif (img.dtype == np.float64) | (img.dtype == np.float32):
        pass
    else:
        raise Exception('img input must be None, string or ndarray of (max = 1) float32 or float64 !')
    if img.max() > 1.0: raise Exception('img input must be None, string or ndarray of (max = 1) float32 or float64 !')
    
    
    # Convert to 2D format:
    rgb = img.reshape(img.shape[0]*img.shape[1],3) # *1.0: make float
    rgb[rgb==0] = _EPS # avoid division by zero for pure blacks.

    
    # Get unique rgb values and positions:
    rgb_u, rgb_indices = np.unique(rgb, return_inverse=True, axis = 0)

    
    # get rfl set:
    if rfl is None: # use IESTM30['4880'] set 
        rfl = _CRI_RFL['ies-tm30']['4880']['5nm']
    wlr = rfl[0] # spectral reflectance set determines wavelength range for estimation (xyz_to_rfl())
        
    # get Ref spd:
    if refspd is None:
        refspd = _CIE_ILLUMINANTS['D65'].copy()
    refspd = cie_interp(refspd, wlr, kind = 'linear') # force spd to same wavelength range as rfl


    # Convert rgb_u to xyz and lab-type values under assumed refspd:
    if CSF is None:
        xyz_wr = spd_to_xyz(refspd, cieobs = cieobs, relative = True)
        xyz_ur = colortf(rgb_u*255, tf = 'srgb>xyz')
    else:
        xyz_ur = rgb_u # for input in xyz_to_rfl (when CSF is not None: this functions assumes input is indeed rgb !!!)
    
    # Estimate rfl's for xyz_ur:
    rfl_est, xyzri = xyz_to_rfl(xyz_ur, rfl = rfl, out = 'rfl_est,xyz_est', \
                 refspd = refspd, D = D, cieobs = cieobs, \
                 cspace = cspace, cspace_tf = cspace_tf, CSF = CSF,\
                 interp_type = interp_type, k_neighbours = k_neighbours, 
                 verbosity = verbosity,
                 csf_based_rgb_rounding = csf_based_rgb_rounding)
    

    # Get default test spd if none supplied:
    if spd is None:
        spd = _CIE_ILLUMINANTS['F4']
        
    if CSF is None:
        # calculate xyz values under test spd:
        xyzti, xyztw = spd_to_xyz(spd, rfl = rfl_est, cieobs = cieobs, out = 2)
    
        # Chromatic adaptation from test spd to refspd:
        if D is not None:
            xyzti = cat.apply(xyzti, xyzw1 = xyztw, xyzw2 = xyz_wr, D = D)
    
        # Convert xyzti under test spd to srgb:
        rgbti = colortf(xyzti, tf = 'srgb')/255
    else:
        # Calculate rgb coordinates from camera sensitivity functions under spd:
        rgbti = rfl_to_rgb(rfl_est, spd = spd, CSF = CSF, wl = None) 
        
         # Chromatic adaptation from test spd to refspd:
        if D is not None:
            white = np.ones_like(spd)
            white[0] = spd[0]
            rgbwr = rfl_to_rgb(white, spd = refspd, CSF = CSF, wl = None)
            rgbwt = rfl_to_rgb(white, spd = spd, CSF = CSF, wl = None)
            rgbti = cat.apply_vonkries2(rgbti,rgbwt,rgbwr,xyzw0=np.array([[1.0,1.0,1.0]]), in_='rgb',out_= 'rgb',D=1)
        
    
    # Reconstruct original locations for rendered image rgbs:
    img_ren = rgbti[rgb_indices]
    img_ren.shape = img.shape # reshape back to 3D size of original
    img_ren = img_ren
    
    # For output:
    if show_ref_img == True:
        rgb_ref = colortf(xyzri, tf = 'srgb')/255 if (CSF is None) else xyzri # if CSF not None: xyzri contains rgbri !!!
        img_ref = rgb_ref[rgb_indices]
        img_ref.shape = img.shape # reshape back to 3D size of original
        img_str = 'Rendered (under ref. spd)'
        img = img_ref
    else:
        img_str = 'Original'
        img = img
       
    
    if (stack_test_ref > 0) | show == True:
        if stack_test_ref == 21:
            img_original_rendered = np.vstack((img_ren,np.ones((4,img.shape[1],3)),img))
            img_original_rendered_str = 'Rendered (under test spd)\n ' + img_str 
        elif stack_test_ref == 12:
            img_original_rendered = np.hstack((img_ren,np.ones((img.shape[0],4,3)),img))
            img_original_rendered_str = 'Rendered (under test spd) | ' + img_str 
        elif stack_test_ref == 1:
            img_original_rendered = img_ren
            img_original_rendered_str = 'Rendered (under test spd)' 
        elif stack_test_ref == 2:
            img_original_rendered = img
            img_original_rendered_str = img_str
        elif stack_test_ref == 0:
            img_original_rendered = img_ren
            img_original_rendered_str =  'Rendered (under test spd)' 
            
    if write_to_file is not None:
        # Convert from RGB to BGR formatand write:
        #print('Writing rendering results to image file: {}'.format(write_to_file))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imsave(write_to_file, img_original_rendered)
            
    if show == True:
        # show images using pyplot.show():
        plt.figure()
        
        plt.imshow(img_original_rendered)
        plt.title(img_original_rendered_str)
        plt.gca().get_xaxis().set_ticklabels([])
        plt.gca().get_yaxis().set_ticklabels([])
        
        if stack_test_ref == 0:
            plt.figure()
            plt.imshow(img)
            plt.title(img_str)
            plt.axis('off')
      
    if 'img_hyp' in out.split(','):
        # Create hyper_spectral image:
        rfl_image_2D = rfl_est[rgb_indices+1,:] # create array with all rfls required for each pixel
        img_hyp = rfl_image_2D.reshape(img.shape[0],img.shape[1],rfl_image_2D.shape[1])


    # Setup output:
    if out == 'img_hyp':
        return img_hyp
    elif out == 'img_ren':
        return img_ren
    else:
        return eval(out)

def rfl_to_rgb(rfl, spd = None, CSF = None, wl = None, normalize_to_white = True):
    """ 
    Convert spectral reflectance functions (illuminated by spd) to Camera Sensitivity Functions.
    
    Args:
        :rfl:
            | ndarray with spectral reflectance functions (1st row is wavelengths if wl is None).
        :spd:
            | None, optional
            | ndarray with illumination spectrum
        :CSF:
            | None, optional
            | ndarray with camera sensitivity functions 
            | If None: use Nikon D700
        :normalize_to_white:
            | True, optional
            | If True: white-balance output rgb to a perfect white diffuser.
    
    Returns:
        :rgb:
            | ndarray with rgb values for each spectral reflectance functions
    """
    rfl_cp = rfl.copy()
    if (wl is None): 
        wl = rfl_cp[0] 
        rfl_cp = rfl_cp[1:]
    wlr = getwlr(wl)
    if spd is not None:
        spd = cie_interp(spd,wlr,kind='linear')[1:]
    else:
        spd = np.ones_like(wlr)
    if CSF is None: CSF = _CSF_NIKON_D700
    CSF = cie_interp(CSF,wlr,kind='linear')
    CSF[1:] = CSF[1:]*spd
    rgb = rfl_cp @ CSF[1:].T 
    if normalize_to_white:
        white = np.ones_like(spd)
        white = white/white.sum()*spd.sum()
        rgbw = white @ CSF[1:].T  
        rgb = rgb/rgbw.max(axis = 0,keepdims=True) 
    
    return rgb
    
    
def hsi_to_rgb(hsi, spd = None, cieobs = _CIEOBS, srgb = False, 
               linear_rgb = False, CSF = None, normalize_to_white = True, 
               wl = [380,780,1]):
    """ 
    Convert HyperSpectral Image to rgb.
    
    Args:
        :hsi:
            | ndarray with hyperspectral image [M,N,L]
        :spd:
            | None, optional
            | ndarray with illumination spectrum
        :cieobs:
            | _CIEOBS, optional
            | CMF set to convert spectral data to xyz tristimulus values.
        :srgb:
            | False, optional
            | If False: Use xyz_to_srgb(spd_to_xyz(...)) to convert to srgb values
            | If True: use camera sensitivity functions.
        :linear_rgb:
            | False, optional
            | If False: use gamma = 2.4 in xyz_to_srgb, if False: use gamma = 1 and set :use_linear_part: to False.
        :CSF:
            | None, optional
            | ndarray with camera sensitivity functions 
            | If None: use Nikon D700
        :normalize_to_white:
            | True, optional
            | If True & CSF is not None: white-balance output rgb to a perfect white diffuser.
        :wl:
            | [380,780,1], optional
            | Wavelength range and spacing or ndarray with wavelengths of HSI image.
    
    Returns:
        :rgb:
            | ndarray with rgb image [M,N,3]
    """
    if spd is None:
        spd = _CIE_E.copy()
    wlr = getwlr(wl)
    spd = cie_interp(spd,wl,kind='linear')
    
    hsi_2d = np.reshape(hsi,(hsi.shape[0]*hsi.shape[1],hsi.shape[2]))
    if srgb:
        xyz = spd_to_xyz(spd, cieobs = cieobs, relative = True, rfl = np.vstack((wlr,hsi_2d)))
        gamma = 1 if linear_rgb else 2.4
        rgb = xyz_to_srgb(xyz, gamma = gamma, use_linear_part = not linear_rgb)/255
    else:
        if CSF is None: CSF = _CSF_NIKON_D700
        rgb = rfl_to_rgb(hsi_2d, spd = spd, CSF = CSF, wl = wl, normalize_to_white = normalize_to_white)        
    return np.reshape(rgb,(hsi.shape[0],hsi.shape[1],3))
       
def get_superresolution_hsi(lrhsi, hrci, CSF, wl = [380,780,1], csf_based_rgb_rounding = _ROUNDING,
                            interp_type = 'nd', k_neighbours = 4, verbosity = 0):
    """ 
    Get a HighResolution HyperSpectral Image (super-resolution HSI) based on a LowResolution HSI and a HighResolution Color Image.
    
    Args:
        :lrhsi:
            | ndarray with float (max = 1) LowResolution HSI [m,m,L].
        :hrci:
            | ndarray with float (max = 1) HighResolution HSI [M,N,3].
        :CSF:
            | None, optional
            | ndarray with camera sensitivity functions 
            | If None: use Nikon D700
        :wl:
            | [380,780,1], optional
            | Wavelength range and spacing or ndarray with wavelengths of HSI image.
        :interp_type:
            | 'nd', optional
            | Options:
            | - 'nd': perform n-dimensional linear interpolation using Delaunay triangulation.
            | - 'nearest': perform nearest neighbour interpolation. 
        :k_neighbours:
            | 4 or int, optional
            | Number of nearest neighbours for reflectance spectrum interpolation.
            | Neighbours are found using scipy.spatial.cKDTree
        :verbosity:
            | 0, optional
            | Verbosity level for sub-call to render_image().
            | If > 0: make a plot of the color coordinates of original and 
            | rendered image pixels.
        :csf_based_rgb_rounding:
            | _ROUNDING, optional
            | Int representing the number of decimals to round the RGB values (obtained from not-None CSF input) to before applying the search algorithm.
            | Smaller values increase the search speed, but could cause fatal error that causes python kernel to die. If this happens increase the rounding int value.

    Returns:
        :hrhsi:
            | ndarray with HighResolution HSI [M,N,L].
        
    Procedure:
        | Call render_image(hrci, rfl = lrhsi_2, CSF = ...) to estimate a hyperspectral image
        | from the high-resolution color image hrci with the reflectance spectra 
        | in the low-resolution hyper-spectral image as database for the estimation.
        | Estimation is done in raw RGB space with the lrhsi converted using the
        | camera sensitivity functions in CSF.
    """
    wlr = getwlr(wl)
    eew = np.vstack((wlr,np.ones_like(wlr)))
    lrhsi_2d = np.vstack((wlr,np.reshape(lrhsi,(lrhsi.shape[0]*lrhsi.shape[1],lrhsi.shape[2])))) # create 2D rfl database
    if CSF is None: CSF = _CSF_NIKON_D700
    hrhsi = render_image(hrci, spd = eew,
                         refspd = eew, rfl = lrhsi_2d, D = None,
                         interp_type = interp_type, k_neighbours = k_neighbours,
                         verbosity = verbosity, show = bool(verbosity),
                         CSF = CSF, csf_based_rgb_rounding = csf_based_rgb_rounding) # render HR-hsi from HR-ci using LR-HSI rfls as database        
    return hrhsi

if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    # Example / test code for HSI simulation and rendering:
    #--------------------------------------------------------------------------
    # plt.close('all')
    # from luxpy.toolboxes import spdbuild as spb
    # S = spb.spd_builder(peakwl = [460,525,590],fwhm=[20,40,20],target=4000, tar_type = 'cct') 
    # img = _HYPSPCIM_DEFAULT_IMAGE
    # img_hyp,img_ren = render_image(img = img, 
    #                                 cspace = 'Yuv',interp_type='nd',
    #                                 spd = S, D=1, 
    #                                 show_ref_img = True,
    #                                 stack_test_ref = 21,
    #                                 out='img_hyp,img_ren',
    #                                 write_to_file = 'test.jpg') 
    # raise Exception('')
    
    #--------------------------------------------------------------------------
    # Example / test code for super resolution:
    #--------------------------------------------------------------------------
    import time
    import luxpy as lx
    import matplotlib.pyplot as plt
    from skimage import transform
    import imageio
    from skimage.transform import rescale,resize
    
    np.random.seed(1)    
    
    # Set some default parameters:
    #----------------------------
    load_hsi = False # If True: load hrci and hrhsi from npy-file.
    file = './data/mysticlamb_center.jpg'

    cieobs = '1931_2' # CIE CMF set
    linear_rgb = 1 # only used when srgb in hsi_to_rgb == True !!!
    verbosity = 0
    
    # Create HR-rgb image and HR-HSI for code testing: 
    #---------------------------------------------------
    # get an image:
    im = imageio.imread(file)/255
    
    # rescale to n x dimensions of typical hyperspectral camera:
    n = 2 # downscale factor
    w, h = 512, 512
    cr,cc = np.array(im.shape[:2])//2
    crop = lambda im,cr,cc,h,w:im[(cr-h//2):(cr+h//2),(cc-w//2):(cc+w//2),:].copy()
    im = crop(im,cr,cc,h*n,w*n)
    print('New image shape:',im.shape)
    
    # simulate HR hyperspectral image:
    hrhsi = render_image(im,show=False)
    wlr = getwlr([380,780,1]) #  = wavelength range of default TM30 rfl set
    wlr = wlr[20:-80:10] # wavelength range from 400nm-700nm every 10 nm
    hrhsi = hrhsi[...,20:-80:10] # wavelength range from 400nm-700nm every 10 nm
    print('Simulated HR-HSI shape:',hrhsi.shape)
    # np.save(file[:-4]+'.npy',{'hrhsi':hrhsi,'im':im, 'wlr':wlr})
    
    # Illumination spectrum of HSI:    
    eew = np.vstack((wlr,np.ones_like(wlr))) 
        
    # Create fig and axes for plots:
    if verbosity > 0: fig, axs = plt.subplots(1,3)
    
    # convert HR hsi to HR rgb image:
    hrci = hsi_to_rgb(hrhsi, spd = eew, cieobs = cieobs, wl = wlr, linear_rgb = linear_rgb)
    if verbosity > 0:  axs[0].imshow(hrci)
    
    # create LR hsi image for testing:
    dl = n 
    lrhsi = hrhsi[::dl,::dl,:]
    print('Simulated LR-HSI shape:',lrhsi.shape)
    
    # convert LR hsi to LR rgb image:
    lrci = hsi_to_rgb(lrhsi, spd = eew, cieobs = cieobs, wl = wlr,linear_rgb = linear_rgb)
    if verbosity > 0:  axs[1].imshow(lrci)
    
    # # Perform rgb guided super-resolution:
    #hrci = lrci # for testing of estimation code
    tic = time.time()
    hrhsi_est = get_superresolution_hsi(lrhsi, hrci, CSF = _CSF_NIKON_D700, wl = wlr)
    print('Elapsed time (s): {:1.4f}'.format(time.time() - tic))
    hrci_est = hsi_to_rgb(hrhsi_est, spd = eew, cieobs = cieobs, wl = wlr, linear_rgb = linear_rgb)

    if verbosity > 0:  axs[2].imshow(hrci_est)
    
    #--------------------------------------------------------------------------
    # Plot some rfl to visually evaluate estimation accuracy:
    
    hsi_rmse = np.linalg.norm(hrhsi-hrhsi_est)/np.array(hrhsi.shape[:2]).prod()**0.5
    print('RMSE(ground-truth,estimate): {:1.4f}'.format(hsi_rmse))
    
    fig, axs = plt.subplots(1,4, figsize=(22,5))
    
    axs[0].imshow(transform.rescale(lrci,dl,order=0,multichannel=True),aspect='auto')
    axs[0].set_title('Color image of LR-HSI\n(HR-to-LR scale factor = {:1.2f})'.format(1/dl))
    axs[0].axis('off')
    axs[1].imshow(hrci_est,aspect='auto')
    axs[1].set_title('Color image of estimated HR-HSI')
    axs[1].axis('off')
    
    px_rmse = ((hrhsi_est-hrhsi)**2).sum(axis=-1)**0.5 # rmse per pixel
    axs[2].set_title('RMSE(ground-truth, estimated) HR-HSI\nRMSE = {:1.4f}, max = {:1.4f}'.format((px_rmse**2).mean()**0.5,px_rmse.max()))
    im = axs[2].imshow(px_rmse, cmap = 'jet',aspect='auto') # rmse per pixel
    cbar = axs[2].figure.colorbar(im, ax=axs[2])
    cbar.ax.set_ylabel('RMSE', rotation=-90, va="bottom")
    
    
    psorted = np.unravel_index(np.argsort(px_rmse, axis=None), px_rmse.shape) # index of pixels sorted by px_rmse
    np.random.seed(1)
    pxs = np.random.permutation(min(hrhsi.shape[:2]))[:12].reshape(2,3,2)
    iis = np.hstack((pxs[...,0].ravel(),psorted[0][-3:]))
    jjs = np.hstack((pxs[...,1].ravel(),psorted[1][-3:]))
    colors = np.array(['m','b','c','g','y','r','k','lightgrey','grey'])
    for t in range(len(iis)):
        ii,jj = iis[t],jjs[t]
        axs[1].plot(jj,ii,color = 'none', marker = 'o', mec = colors[t])
        axs[2].plot(jj,ii,color = 'none', marker = 'o', mec = colors[t])
        axs[3].plot(wlr,hrhsi[ii,jj,:],color = colors[t], linestyle ='-',label='ground-truth (r{:1.0f},c{:1.0f})'.format(ii,jj))
        axs[3].plot(wlr,hrhsi_est[ii,jj,:],color = colors[t], linestyle = '--',label='estimate (r{:1.0f},c{:1.0f})'.format(ii,jj))
    axs[3].legend(bbox_to_anchor=(1.05, 1))   
    axs[3].set_xlabel('Wavelengths (nm)')
    axs[3].set_ylabel('Spectral Reflectance')
    plt.subplots_adjust(right=0.8)
    
    
    
    
    
    
    