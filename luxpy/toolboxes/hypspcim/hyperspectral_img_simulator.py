# -*- coding: utf-8 -*-
"""
###############################################################################
# Module for hyper spectral image simulation
###############################################################################
# _HYPSPCIM_PATH: path to module

# _HYPSPCIM_DEFAULT_IMAGE: path + filename to default image

# render_image(): Render image under specified light source spd.

#------------------------------------------------------------------------------
Created on Sun Apr 29 16:30:05 2018

@author: kevin.smet
"""

from luxpy import (np, plt, cv2, cKDTree, cat, colortf, _PKG_PATH, _SEP, _CIEOBS, 
                   _CIE_ILLUMINANTS, _CRI_RFL, _EPS, spd_to_xyz,plot_color_data)
from luxpy.toolboxes.spdbuild import spdbuilder as spb
   
__all__ =['_HYPSPCIM_PATH','_HYPSPCIM_DEFAULT_IMAGE','render_image']             

_HYPSPCIM_PATH = _PKG_PATH + _SEP + 'hypspcim' + _SEP
_HYPSPCIM_DEFAULT_IMAGE = _PKG_PATH + _SEP + 'toolboxes' + _SEP + 'hypspcim' +  _SEP + 'data' + _SEP + 'default_image_.jpg'

def render_image(img = None, spd = None, rfl = None, out = 'ren_img', \
                 refspd = None, D = None, cieobs = _CIEOBS, \
                 cspace = 'ipt', cspace_tf = {},\
                 k_neighbours = 4, show = (True,True),
                 verbosity = 0, show_ref_img = True):
    """
    Render image under specified light source spd.
    
    Args:
        :img: None or str or ndarray with uint8 rgb image.
            None load a default image.
        :spd: ndarray, optional
            Light source spectrum for rendering
        :rfl: ndarray, optional
            Reflectance set for color coordinate to rfl mapping.
        :out: 'hypim' or str, optional
        :refspd: None, optional
            Reference spectrum for color coordinate to rfl mapping.
            None defaults to D65 (srgb has a D65 white point)
        :D: None, optional
            Degree of (von Kries) adaptation from spd to refspd. 
        :cieobs: _CIEOBS, optional
            CMF set for calculation of xyz from spectral data.
        :cspace: 'ipt',  optional
            Color space for color coordinate to rfl mapping.
        :cspace_tf: {}, optional
            Dict with parameters for xyz_to_... and ..._to_xyz transform.
        :k_neighbours: 4 or int, optional
            Number of nearest neighbours for reflectance spectrum interpolation.
            Neighbours are found using scipy.cKDTree
        :show: (True,True), optional
             Tuple with boolean determing whether to show 
             original and rendered images or not. If not a tuple, :show:
             determines both.
        :verbosity: 0, optional
            If > 0: make a plot of the color coordinates of original and 
            rendered image pixels.
            
        
    Returns:
        :returns: hyp_img, ren_img, 
            ndarrays with hyperspectral image and rendered images 
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

    
    # get rfl set:
    if rfl is None: # use IESTM30['4880'] set 
        rfl = _CRI_RFL['ies-tm30']['4880']['5nm']
        
    # get Ref spd:
    if refspd is None:
        refspd = _CIE_ILLUMINANTS['D65'].copy()

        
    # Calculate lab-type coordinates of rfl set under refspd:
    xyz_rr, xyz_wr = spd_to_xyz(refspd, relative = True, rfl = rfl, cieobs = cieobs, out = 2)
    cspace_tf_copy = cspace_tf.copy()
    cspace_tf_copy['xyzw'] = xyz_wr # put correct white point in param. dict
    lab_rr = colortf(xyz_rr, tf = cspace, fwtf = cspace_tf_copy, bwtf = cspace_tf_copy)[:,0,:]
    
    # Convert srgb to lab-type values under refspd:
    lab_ur = colortf(rgb_u, tf = 'srgb>'+ cspace, fwtf = cspace_tf_copy, bwtf = cspace_tf_copy)
    rgb_ur = colortf(lab_ur, tf = cspace + '>srgb', bwtf = cspace_tf_copy)/255
    
    # Find rfl (cfr. lab_rr) from rfl set that results in 'near' metameric 
    # color coordinates for each value in lab_ur (i.e. smallest DE):
    # Construct cKDTree:
    tree = cKDTree(lab_rr, copy_data = True)
    
    # Interpolate rfls using k nearest neightbours and inverse distance weigthing:
    d, inds = tree.query(lab_ur, k = k_neighbours )
    if k_neighbours  > 1:
        w = (1.0 / d**2)[:,:,None] # inverse distance weigthing
        rfl_idw = np.sum(w * rfl[inds+1,:], axis=1) / np.sum(w, axis=1)
    else:
        rfl_idw = rfl[inds+1,:].copy()
    rfl_idw = np.vstack((rfl[0],rfl_idw))
    
    # Calculate xyz values under refspd, and check DE:
    if spd is None:
        spd = _CIE_ILLUMINANTS['F4']
    
    xyz_rr_idw, _ = spd_to_xyz(refspd, rfl = rfl_idw, relative = True, cieobs = cieobs, out = 2)
    cspace_tf_copy = cspace_tf.copy()
    cspace_tf_copy['xyzw'] = xyz_wr # put correct white point in param. dict
    lab_rr_idw = colortf(xyz_rr_idw, tf = cspace, fwtf = cspace_tf_copy)[:,0,:]


    if verbosity > 0:
        ax = plot_color_data(lab_ur[...,1], lab_ur[...,2], z = lab_ur[...,0], \
                        show = False, cieobs = cieobs, cspace = cspace, \
                        formatstr = 'ro', label = 'Original')
        plot_color_data(lab_rr_idw[...,1], lab_rr_idw[...,2], z = lab_rr_idw[...,0], \
                        show = True, axh = ax, cieobs = cieobs, cspace = cspace, \
                        formatstr = 'bd', label = 'Rendered')

#    fig = plt.figure()
#    ax = plt.axes(projection='3d')
#    plt.plot(lab_ur[...,1],lab_ur[...,2],lab_ur[...,0],'ro')
#    plt.plot(lab_rr_idw[...,1],lab_rr_idw[...,2],lab_rr_idw[...,0],'bd')
#    DEi = np.sqrt(((lab_rr_idw[:,1:3]-lab_ur[:,1:3])**2).sum(axis=1))
#    DEa = DEi.mean()
    
    # calculate xyz values under spd:
    xyz_rs, xyz_ws = spd_to_xyz(spd, rfl = rfl_idw, cieobs = cieobs, out = 2)
    
    # Chromatic adaptation to refspd:
    if D is not None:
        xyz_rs = cat.apply(xyz_rs,xyzw1 = xyz_ws,xyzw2 = xyz_wr, D = D)
    
    
    
    # Convert to srgb:
    cspace_tf_copy['xyzw'] = xyz_ws
    rgb_rs = colortf(xyz_rs, tf = 'srgb', fwtf = {})/255
    
    # Reconstruct original locations for rendered image rgbs:
    ren_img = rgb_rs[rgb_indices]
    ren_img.shape = img.shape # reshape back to 3D size of original
     
    if show_ref_img == True:
        cspace_tf_copy['xyzw'] = xyz_wr
        rgb_rr = colortf(xyz_rr_idw, tf = 'srgb', fwtf = {})/255
        ren_ref_img = rgb_rr[rgb_indices]
        ren_ref_img.shape = img.shape # reshape back to 3D size of original
        img_str = 'Rendered (under ref. spd)'
        img = ren_ref_img
    else:
        img_str = 'Original'
        img = img/255
    
    if isinstance(show,tuple):
        if show[0] == True:
            # Convert from BGR to RGB float32 image and show original image:
            cv2.imshow(img_str, cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR))
        if show[1] == True:
            # Convert from BGR to RGB float32 image and show rendered image:    
            cv2.imshow('Rendered (under test spd)',cv2.cvtColor(ren_img.astype(np.float32), cv2.COLOR_RGB2BGR))
    else:
        if show == True:
            oriren_img = np.vstack((img,np.ones((4,img.shape[1],3)),ren_img))
            # Convert from BGR to RGB float32 image and show rendered image:    
            cv2.imshow( img_str + ' | rendered (under test spd)',cv2.cvtColor(oriren_img.astype(np.float32), cv2.COLOR_RGB2BGR))

    
    if 'hyp_img' in out.split(','):
        # Create hyper_spectral image:
        rfl_image_2D = rfl_idw[rgb_indices+1,:] # create array with all rfls required for each pixel
        hyp_img = rfl_image_2D.reshape(img.shape[0]*img.shape[1],rfl_image_2D.shape[1])
    

    # Setup output:
    if out == 'hyp_img':
        return himg
    elif out == 'ren_img':
        return ren_img
    else:
        return eval(out)
        
    

if __name__ == '__main__':
    plt.close('all')
    S = _CIE_ILLUMINANTS['F4']
    S = spb.spd_builder(peakwl = [450,525,590],fwhm=[20,40,10],target=4500, tar_type = 'cct') 
    img = _HYPSPCIM_DEFAULT_IMAGE
    render_image(img = img, cspace = 'ipt',spd = S, D=1,show = True, show_ref_img = True)    
    
        


