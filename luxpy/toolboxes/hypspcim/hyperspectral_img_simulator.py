# -*- coding: utf-8 -*-
"""
Module for hyper spectral image simulation
==========================================

 :_HYPSPCIM_PATH: path to module

 :_HYPSPCIM_DEFAULT_IMAGE: path + filename to default image

 :xyz_to_rfl(): approximate spectral reflectance of xyz based on k nearest 
                neighbour interpolation of samples from a standard reflectance 
                set.

 :render_image(): Render image under specified light source spd.


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import (np, plt, cv2, cKDTree, cat, colortf, _PKG_PATH, _SEP, _CIEOBS, 
                   _CIE_ILLUMINANTS, _CRI_RFL, _EPS, spd_to_xyz,plot_color_data)
from luxpy.toolboxes.spdbuild import spdbuilder as spb
   
__all__ =['_HYPSPCIM_PATH','_HYPSPCIM_DEFAULT_IMAGE','render_image']             

_HYPSPCIM_PATH = _PKG_PATH + _SEP + 'hypspcim' + _SEP
_HYPSPCIM_DEFAULT_IMAGE = _PKG_PATH + _SEP + 'toolboxes' + _SEP + 'hypspcim' +  _SEP + 'data' + _SEP + 'testimage1.jpg'


def xyz_to_rfl(xyz, rfl = None, out = 'rfl_est', \
                 refspd = None, D = None, cieobs = _CIEOBS, \
                 cspace = 'ipt', cspace_tf = {},\
                 k_neighbours = 4, verbosity = 0):
    """
    Approximate spectral reflectance of xyz based on k nearest neighbour 
    interpolation of samples from a standard reflectance set.
    
    Args:
        :xyz: 
            | ndarray with tristimulus values of target points.
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
            | 'ipt',  optional
            | Color space for color coordinate to rfl mapping.
        :cspace_tf:
            | {}, optional
            | Dict with parameters for xyz_to_... and ..._to_xyz transform.
        :k_neighbours:
            | 4 or int, optional
            | Number of nearest neighbours for reflectance spectrum interpolation.
            | Neighbours are found using scipy.cKDTree
        :verbosity:
            | 0, optional
            | If > 0: make a plot of the color coordinates of original and 
              rendered image pixels.

    Returns:
        :returns: 
            | :rfl_est:
            | ndarrays with estimated reflectance spectra.
    """

    # get rfl set:
    if rfl is None: # use IESTM30['4880'] set 
        rfl = _CRI_RFL['ies-tm30']['4880']['5nm']
        
    # get Ref spd:
    if refspd is None:
        refspd = _CIE_ILLUMINANTS['D65'].copy()

        
    # Calculate lab-type coordinates of standard rfl set under refspd:
    xyz_rr, xyz_wr = spd_to_xyz(refspd, relative = True, rfl = rfl, cieobs = cieobs, out = 2)
    cspace_tf_copy = cspace_tf.copy()
    cspace_tf_copy['xyzw'] = xyz_wr # put correct white point in param. dict
    lab_rr = colortf(xyz_rr, tf = cspace, fwtf = cspace_tf_copy, bwtf = cspace_tf_copy)[:,0,:]
    
    # Convert xyz to lab-type values under refspd:
    lab = colortf(xyz, tf = cspace, fwtf = cspace_tf_copy, bwtf = cspace_tf_copy)
    
    # Find rfl (cfr. lab_rr) from rfl set that results in 'near' metameric 
    # color coordinates for each value in lab_ur (i.e. smallest DE):
    # Construct cKDTree:
    tree = cKDTree(lab_rr, copy_data = True)
    
    # Interpolate rfls using k nearest neightbours and inverse distance weigthing:
    d, inds = tree.query(lab, k = k_neighbours )
    if k_neighbours  > 1:
        w = (1.0 / d**2)[:,:,None] # inverse distance weigthing
        rfl_est = np.sum(w * rfl[inds+1,:], axis=1) / np.sum(w, axis=1)
    else:
        rfl_est = rfl[inds+1,:].copy()
    rfl_est = np.vstack((rfl[0],rfl_est))
        
    if (verbosity > 0) | ('xyz_est' in out.split(',')) | ('lab_est' in out.split(',')) | ('DEi_ab' in out.split(',')) | ('DEa_ab' in out.split(',')):
        xyz_est, _ = spd_to_xyz(refspd, rfl = rfl_est, relative = True, cieobs = cieobs, out = 2)
        cspace_tf_copy = cspace_tf.copy()
        cspace_tf_copy['xyzw'] = xyz_wr # put correct white point in param. dict
        lab_est = colortf(xyz_est, tf = cspace, fwtf = cspace_tf_copy)[:,0,:]
        DEi_ab = np.sqrt(((lab_est[:,1:3]-lab[:,1:3])**2).sum(axis=1))
        DEa_ab = DEi_ab.mean()

    if verbosity > 0:
        ax = plot_color_data(lab[...,1], lab[...,2], z = lab[...,0], \
                        show = False, cieobs = cieobs, cspace = cspace, \
                        formatstr = 'ro', label = 'Original')
        plot_color_data(lab_est[...,1], lab_est[...,2], z = lab_est[...,0], \
                        show = True, axh = ax, cieobs = cieobs, cspace = cspace, \
                        formatstr = 'bd', label = 'Rendered')

    if out == 'rfl_est':
        return rfl_est
    elif out == 'rfl_est,xyz_est':
        return rfl_est, xyz_est
    else:
        return eval(out)

def render_image(img = None, spd = None, rfl = None, out = 'img_hyp', \
                 refspd = None, D = None, cieobs = _CIEOBS, \
                 cspace = 'ipt', cspace_tf = {},\
                 k_neighbours = 4, show = (True,True),
                 verbosity = 0, show_ref_img = True,\
                 stack_test_ref = 12,\
                 write_to_file = None,
                 use_plt_show = False):
    """
    Render image under specified light source spd.
    
    Args:
        :img: 
            | None or str or ndarray with uint8 rgb image.
            | None load a default image.
        :spd: 
            | ndarray, optional
            | Light source spectrum for rendering
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
            | 'ipt',  optional
            | Color space for color coordinate to rfl mapping.
        :cspace_tf:
            | {}, optional
            | Dict with parameters for xyz_to_cspace and cspace_to_xyz transform.
        :k_neighbours:
            | 4 or int, optional
            | Number of nearest neighbours for reflectance spectrum interpolation.
            | Neighbours are found using scipy.cKDTree
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
              original image.
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
        :use_plt_show:
            | False, optional
            |  - True: Use matplotlib.pyplot.imshow 
            |  - False: use open-cv imshow() 
    
    Returns:
        :returns: 
            | img_hyp, img_ren, 
            | ndarrays with hyperspectral image and rendered images 
    """
    
    # Get image:
    if img is not None:
        if isinstance(img,str):
            img = cv2.imread(img,1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert from BGR of opencv to RGB format
    else:
        img = cv2.imread(_HYPSPCIM_DEFAULT_IMAGE,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert from BGR of opencv to RGB format
    
    
    # Convert to 2D format:
    rgb = img.reshape(img.shape[0]*img.shape[1],3)*1.0 # *1.0: make float
    rgb[rgb==0] = _EPS # avoid division by zero for pure blacks.

    
    # Get unique rgb values and positions:
    rgb_u, rgb_indices = np.unique(rgb, return_inverse=True, axis = 0)

        
    # get Ref spd:
    if refspd is None:
        refspd = _CIE_ILLUMINANTS['D65'].copy()

    # Convert rgb_u to xyz and lab-type values under assumed refspd:
    xyz_wr = spd_to_xyz(refspd, cieobs = cieobs, relative = True)
    xyz_ur = colortf(rgb_u, tf = 'srgb>xyz')
    
    # Estimate rfl's for xyz_ur:
    rfl_est, xyzri = xyz_to_rfl(xyz_ur, rfl = rfl, out = 'rfl_est,xyz_est', \
                 refspd = refspd, D = D, cieobs = cieobs, \
                 cspace = cspace, cspace_tf = cspace_tf,\
                 k_neighbours = k_neighbours, verbosity = verbosity)
    
    
    
    # Get default test spd if none supplied:
    if spd is None:
        spd = _CIE_ILLUMINANTS['F4']
        
    # calculate xyz values under test spd:
    xyzti, xyztw = spd_to_xyz(spd, rfl = rfl_est, cieobs = cieobs, out = 2)
    
    # Chromatic adaptation from test spd to refspd:
    if D is not None:
        xyzti = cat.apply(xyzti, xyzw1 = xyztw, xyzw2 = xyz_wr, D = D)
    
    # Convert xyzti under test spd to srgb:
    rgbti = colortf(xyzti, tf = 'srgb')/255
    
    # Reconstruct original locations for rendered image rgbs:
    img_ren = rgbti[rgb_indices]
    img_ren.shape = img.shape # reshape back to 3D size of original
     
    
    # For output:
    if show_ref_img == True:
        rgb_ref = colortf(xyzri, tf = 'srgb')/255
        img_ref = rgb_ref[rgb_indices]
        img_ref.shape = img.shape # reshape back to 3D size of original
        img_str = 'Rendered (under ref. spd)'
        img = img_ref
    else:
        img_str = 'Original'
        img = img/255
    
    if (stack_test_ref > 0) | show == True:
        if stack_test_ref == 21:
            img_original_rendered = np.vstack((img_ren,np.ones((4,img.shape[1],3)),img))
            img_original_rendered_str = 'Rendered (under test spd) | ' + img_str 
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
        cv2.imwrite(write_to_file, cv2.cvtColor((255*img_original_rendered).astype(np.float32), cv2.COLOR_RGB2BGR))
        
    if show == True:
        if use_plt_show == False:
            #show rendered image using cv2:    
            cv2.imshow(img_original_rendered_str ,cv2.cvtColor(img_original_rendered.astype(np.float32), cv2.COLOR_RGB2BGR))
        
            if stack_test_ref == 0: # show both in sep. figures
                cv2.imshow(img_str ,cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR))
        else:
            # show images using pyplot.show():
            plt.figure()
            plt.imshow(img_original_rendered)
            plt.title(img_original_rendered_str)
            plt.gca().get_xaxis().set_ticklabels([])
            plt.gca().get_yaxis().set_ticklabels([])
            
            if stack_test_ref == 0:
                plt.figure()
                plt.imshow(img_str)
                plt.title(img_str)
                plt.gca().get_xaxis().set_ticklabels([])
                plt.gca().get_yaxis().set_ticklabels([])
      
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
        
    

if __name__ == '__main__':
    plt.close('all')
    S = spb.spd_builder(peakwl = [460,525,590],fwhm=[20,40,20],target=4000, tar_type = 'cct') 
    img = _HYPSPCIM_DEFAULT_IMAGE
    img_hyp,img_ren = render_image(img = img, cspace = 'ipt',spd = S, 
                                 D=1,
                                 show = True, show_ref_img = True,
                                 use_plt_show = True, stack_test_ref = 21,
                                 out='img_hyp,img_ren',
                                 write_to_file = None)    
    
        


