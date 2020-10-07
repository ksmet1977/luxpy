# -*- coding: utf-8 -*-
"""
Multivariate polynomial model
=============================

 :get_poly_model(): Get multivariate polynomial model parameters.
     
 :apply_poly_model_at_x(): Applies multivariate polynomial model at cartesian reference coordinates.
 
 :generate_rect_grid(): Generate a rectangular grid of Cartesian coordinates.
 
 :generate_circ_grid(): Generate a circular grid of Cartesian coordinates.

 :generate_vector_field(): Generates a field of vectors using the multivariate polynomial model.
     
 :plot_vector_field(): Makes a plot of a vector field.
     
 :MuPolyModel(): Multivariate Polynomial Model Class


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import colorsys
from luxpy.utils import np, plt, Axes3D
from . import (cart2pol, positive_arctan)

_EPS = np.finfo('float').eps # used in model to avoid division by zero ! 

__all__ = ['get_poly_model','apply_poly_model_at_x',
           'generate_rect_grid','generate_circ_grid','generate_vector_field',
           'plot_vector_field','MuPolyModel']

#------------------------------------------------------------------------------
# Multivariate Poly_Model functions:
#------------------------------------------------------------------------------  

def poly_model(xyz, p = None, k = 0):
    """
    Polynomial (2nd order) model.
    
    Args:
        :xyz:
            | ndarray with input values
        :p:
            | None, optional
            | model parameters (see notes)
            | If None, return model matrix:
            |   3D: np.array([1, x, y, x**2, y**2, xy])
            |   2D: np.array([1, x, y, z, x**2, y**2, z**2, xy, xz, yz])
            | constant is omitted depending on value of k (0: all, 1: omit)
        :k:
            | 0, optional
            | Omit constant (1) or not (0)
            | if p is not None: k is automatically determined from size of p
    
    Notes:
        1. Model types:
            2D-data:
            | poly_model (n = 5):         p[0]*x + p[1]*y + p[2]*(x**2) + p[3]*(y**2) + p[4]*x*y 
            | poly_model (n = 6):  p[0] + p[1]*x + p[2]*y + p[3]*(x**2) + p[4]*(y**2) + p[5]*x*y 
            3D-data:
            | poly_model (n = 9):          p[0]*x + p[1]*y + p[2]*y + p[3]*(x**2) + p[4]*(y**2) + p[5]*(x**2) + p[6]*x*y + p[7]*x*z + p[8]*y*z 
            | poly_model (n = 10):  p[0] + p[1]*x + p[2]*y + p[3]*y + p[4]*(x**2) + p[5]*(y**2) + p[5]*(x**2) + p[7]*x*y + p[8]*x*z + p[9]*y*z 
    """
    m = np.hstack((np.ones((xyz.shape[0],1)), xyz, xyz**2, xyz[:,[0,1]].prod(axis=1,keepdims=True)))
    if xyz.shape[1] == 3:
        m = np.hstack((m, xyz[:,[0,2]].prod(axis=1,keepdims=True),xyz[:,[1,2]].prod(axis=1,keepdims=True)))
    if p is None:
        return m.T[k:,:]
    else:
        n = xyz.shape[1]*4 - 2 
        k = n - p.shape[1]
        xyz_pred = np.dot(p, m.T[k:,]).T
        return xyz_pred


def get_poly_model(xyzt, xyzr, npar = 10, get_stats = True, dict_out = True, diff_model = True, polar_coord = False):
    """
    Get multivariate polynomial model parameters.
    
    Args:
        :xyzt: 
            | ndarray with target coordinates (to be predicted starting from reference xyzr).
        :xyzr: 
            | ndarray with reference coordinates (to be transformed to predictions of xyzt).
        :npar:
            | Specifies the number of parameters of the polynomial model in xy-coordinates 
            | (npar = 9 or 10 for 3D data, 5 or 6 for 2D data;
            |  9 and 5 omit the constant term; see notes below)
        :get_stats:
            | True, optional
            | Calculate model statistics: dxyz_pred, dxyz_res, dxyz_res_std, dRToverR_pred, dRToverR_res,  dRToverR_res_std
            | If False: fill with None's 
            | See :returns: for more info on statistics.
        :dict_out:
            | True, optional
            | Get function output as dict instead of tuple.
        :polar_coord:
            | False, optional
            | If True: also calculate dR/R (R=radial distance), dT (T=theta) (only when get_stat == True !! )
              
    Returns:   
        :returns: 
            | Dict or tuple with:
            |  - 'poly_model' : function handle to model
            |  - 'p' : ndarray with model parameters
            |  - 'M' : optimization matrix
            |  - 'dxyz pred' : ndarray with dxy model predictions from xr, yr, zr.
            |  - 'dxyz res' : ndarray with residuals between 'dx,dy,dz' of samples and 
            |                'dx,dy,dz' predicted by the model.
            |  - 'dxyz res std' : ndarray with std of 'dxyz res'
            |  - 'RTred' : ndarray with Radial distance and Theta angles of reference.
            |  - 'dR/R,dT pred' : ndarray with predictions for dR/R = (Rt - Rr)/Rr and dT = ht - hr
            |  - 'dR/R,dT res' : ndarray with residuals between 'dR/R,dT' 
            |                      of samples and 'dR/R,dT' predicted by the model.
            |  - 'dR/R,dT res std' : ndarray with std of 'dR/R,dT res: 

    Notes: 
        1. Model types:
            2D-data:
            | poly_model (n = 5):         p[0]*x + p[1]*y + p[2]*(x**2) + p[3]*(y**2) + p[4]*x*y 
            | poly_model (n = 6):  p[0] + p[1]*x + p[2]*y + p[3]*(x**2) + p[4]*(y**2) + p[5]*x*y 
            3D-data:
            | poly_model (n = 9):          p[0]*x + p[1]*y + p[2]*y + p[3]*(x**2) + p[4]*(y**2) + p[5]*(x**2) + p[6]*x*y + p[7]*x*z + p[8]*y*z 
            | poly_model (n = 10):  p[0] + p[1]*x + p[2]*y + p[3]*y + p[4]*(x**2) + p[5]*(y**2) + p[5]*(x**2) + p[7]*x*y + p[8]*x*z + p[9]*y*z 
        
        2. Calculation of dRoverR and dT:
            | dRoverR = (np.cos(Tr)*dx + np.sin(Tr)*dy)/Rr
            | dToverC = (np.cos(Tr)*dy - np.sin(Tr)*dx)/Rr   
    """
    # A. Calculate dyx:
    if diff_model == True:
        dxyz = xyzt - xyzr
    else:
        dxyz = xyzt.copy()
    
    # B Calculate model matrix:
    m = poly_model(xyzr, p = None, k = xyzt.shape[1]*4 - 2 - npar) # get "vandermonde"-type matrix of [1, x, y, x**2, y**2, xy] or [1,x,y,z,x**2,y**2,z**2,xy,xz,yz]
    M = np.dot(m,m.T)  
   
    # Get inverse matrix:
    M = np.linalg.inv(M)
    
    # C. Get model parameters:
    pmodel = np.dot(M, np.dot(m,dxyz)).T

    if get_stats == True:
        # D.1 Calculate predicted dxy (with resp. to ref.):
        dxyz_pred = poly_model(xyzr, p = pmodel)
    
        # D.2 Calculate residuals for dxyz:
        dxyz_res = dxyz - dxyz_pred
        dxyz_res_std = dxyz_res.std(axis = 0, keepdims = True)

        # E.1 Calculate Tref, Rref:
        Tref, Rref = np.arctan2(xyzr[:,1], xyzr[:,0]), (xyzr**2).sum(axis = 1)**0.5

        # E.2 Calculate res and std of dR/R, dH/C:
        # TO BE EXTENDED TO SPHERICAL COORDINATES, April 4, 2020!!
        if xyzr.shape[1] == 2:
            Mrot = np.array([[np.cos(Tref), np.sin(Tref)],
                              [- np.sin(Tref), np.cos(Tref)]])
        else:
            Mrot = np.array([[np.cos(Tref), np.sin(Tref), np.zeros_like(Tref)],
                              [- np.sin(Tref), np.cos(Tref),  np.zeros_like(Tref)],
                              [ np.zeros_like(Tref),  np.zeros_like(Tref),  np.ones_like(Tref)]])
        dRToverR = (np.einsum('ijk,jk->ki',Mrot,dxyz.T)/Rref[:,None])
        dRToverR_pred = (np.einsum('ijk,jk->ki',Mrot,dxyz_pred.T)/Rref[:,None])
        dRToverR_res = dRToverR - dRToverR_pred
        dRToverR_res_std = dRToverR_res.std(axis = 0, keepdims=True)        
        RTref = np.vstack((Rref,Tref)).T
    else:
        dxyz_pred, dxyz_res, dxyz_res_std, RTref, dRToverR_pred, dRToverR_res, dRToverR_res_std = [None]*7 

    if dict_out == True:
        return {'poly_model' : poly_model, 
                'p' : pmodel, 
                'npar':npar,
                'M' : M,
                'dxyz pred': dxyz_pred, 
                'dxyz res': dxyz_res, 
                'dxyz res std': dxyz_res_std,
                'RTref' : np.hstack((Rref,Tref)),
                'dR/R,dT pred': dRToverR_pred, 
                'dR/R,dT res': dRToverR_res, 
                'dR/R,dT res std': dRToverR_res_std}
    else:
        return (poly_model, pmodel, M,
                dxyz_pred, dxyz_res, dxyz_res_std, 
                RTref, dRToverR_pred, dRToverR_res, dRToverR_res_std)


def apply_poly_model_at_x(xyzr, pmodel, polar_coord = False, diff_model = True, out = 'xyzt,(RTt,RTr)'):
    """
    Applies multivariate polynomial model at cartesian reference coordinates.
    
    Args:
        :xyr: 
            | ndarray with reference xyz-coordinates
        :pmodel:
            | ndarray with model parameters.
        :polar_coord:
            | False, optional
            | If True: also calculate R(adial distance) and T(heta angle) for xt and yt.
        
    Returns:
        :returns:
            | xyt,(RTt,RTr)
            | 
            | xyt: ndarrays with predicted xy-coordinates, 
            | RTt: radial distance and angle predicted by the model (xyzt)
            | RTr: radial distance and angle for reference  (xyzr)
    """  
    # A Set 2nd order color multipliers (shift parameters for xyz: i.e. pxyz):
    includes_cte = pmodel.shape[1] == (xyzr.shape[1]*4-2)
    pxyz = pmodel.copy()
    if diff_model == True:
        pxyz[0,[0 + includes_cte*1]] = 1 + pxyz[0,[0 + includes_cte*1]]
        pxyz[1,[1 + includes_cte*1]] = 1 + pxyz[1,[1 + includes_cte*1]]
        if pxyz.shape[0]==3: # also make it work for 2D xy input
            pxyz[2,[2 + includes_cte*1]] = 1 + pxyz[2,[2 + includes_cte*1]]
    # B Apply model to reference coordinates using 2nd order multipliers:
    xyzt = poly_model(xyzr, p = pxyz)
    
    if polar_coord == True:
        # C Calculate R and T for xy:
        RTr = (xyzr**2).sum(axis=1,keepdims=True)**0.5, np.arctan2(xyzr[:,1:2], xyzr[:,0:1]+_EPS) #_eps avoid zero-division
        RTt = (xyzt**2).sum(axis=1,keepdims=True)**0.5, np.arctan2(xyzr[:,1:2], xyzr[:,0:1]+_EPS) # test radial distance and theta angle
    else:
        RTr, RTt = None, None
    if out == 'xyzt,(RTt,RTr)':
        return xyzt, (RTt,RTr)
    elif out == 'xyzt':
        return xyzt
    else:
        return eval(out)


#------------------------------------------------------------------------------
# Grid and vector-shift field generators:
#------------------------------------------------------------------------------

def generate_rect_grid(xyz_ranges = np.array([[-100,100,10],[-100,100,10],[0,100,10]]),
                       x_sampling = None, y_sampling = None, z_sampling = None,
                       limit_grid_radius = 0):
    """
    Generate a rectangular grid of cart. coordinates.
    
    Args:
        :xy_ranges:
            | None or ndarray, optional
            | Specifies the uniform pixelization of the 3D space.
            |  (ndarray.shape = (3,3), with  first axis: x,y,z, and second 
            |  axis: min, max, delta)
        :x_sampling:
            | None, optional
            | User defined ndarray with sequence of values. If not None: overrides xyz_ranges
        :y_sampling:
            | None, optional
            | User defined ndarray with sequence of . If not None: overrides xyz_ranges
        :z_sampling:
            | None, optional
            | User defined ndarray with sequence of . If not None: overrides xyz_ranges
        :limit_grid_radius:
            | 0, optional
            | A value of zeros keeps grid as specified  by xr,yr,zr.
            | A value > 0 only keeps (x,y) coordinates within :limit_grid_radius:
            
    Returns:
        :returns: 
            | single ndarray with x_grid,y_grid ; the grid points along each dimension
            |  or
            | seperate ndarrays for each dimension specified.
    """
    # generate grid from xy_ranges array input, otherwise use x_sampling, y_sampling, z_sampling input:
    if (x_sampling is None) | (y_sampling is None) | (z_sampling is None):
        x_sampling = np.arange(xyz_ranges[0][0],xyz_ranges[0][1],xyz_ranges[0][2])
        y_sampling = np.arange(xyz_ranges[1][0],xyz_ranges[1][1],xyz_ranges[1][2])
        if xyz_ranges.shape[0] == 3: # also make it work for 2D xy:
            z_sampling = np.arange(xyz_ranges[2][0],xyz_ranges[2][1],xyz_ranges[2][2])

    # Generate grid from x_sampling, y_sampling, z_sampling:
    if z_sampling is not None:
        X,Y,Z = np.meshgrid(x_sampling,y_sampling,z_sampling)
        grid = np.concatenate((X[...,None],Y[...,None],Z[...,None]),axis=3)
    else: # also make it work for 2D xy:
        X,Y = np.meshgrid(x_sampling,y_sampling)
        grid = np.dstack((X,Y))
    grid = np.reshape(grid,(np.array(grid.shape[:-1]).prod(),grid.ndim-1))

    if limit_grid_radius > 0:# limit radius of grid:
        Rr = (grid**2).sum(axis=1,keepdims=True)**0.5
        grid = grid[:,Rr<=limit_grid_radius]
    
    return grid
    

def generate_circ_grid(RTZ_ranges = np.array([[0,100,10],[0,360,5],[0,100,10]]),
                       R_sampling = None, T_sampling = None, Z_sampling = None, 
                       limit_grid_radius = 0):
    """
    Generate a circular sampled grid of cart. coordinates.
    
    Args:
        :RTZ_ranges:
            | None or ndarray, optional
            | Specifies the uniform pixelization of the 3D space.
            |  (ndarray.shape = (3,3), with  first axis: R(adial),T(heta),Z and second 
            |  axis: min, max, delta)
        :R_sampling:
            | None, optional
            | User defined ndarray with sequence of values. If not None: overrides RTZ_ranges
        :T_sampling:
            | None, optional
            | User defined ndarray with sequence of . If not None: overrides RTZ_ranges
        :Z_sampling:
            | None, optional
            | User defined ndarray with sequence of . If not None: overrides RTZ_ranges
        :limit_grid_radius:
            | 0, optional
            | A value > 0 only keeps (x,y) coordinates within :limit_grid_radius:
            
    Returns:
        :returns: 
            | single ndarray with x_grid,y_grid,z_grid ; the grid points along each dimension
    """
    # generate grid from RT_ranges array input, otherwise use R_sampling, T_sampling, input:
    if (R_sampling is None) | (R_sampling is None) | (Z_sampling is None):
        R_sampling = np.arange(RTZ_ranges[0][0],RTZ_ranges[0][1],RTZ_ranges[0][2])
        T_sampling = np.arange(RTZ_ranges[1][0],RTZ_ranges[1][1],RTZ_ranges[1][2])
        if RTZ_ranges.shape[0] == 3:
            Z_sampling = np.arange(RTZ_ranges[2][0],RTZ_ranges[2][1],RTZ_ranges[2][2])
    x_grid, y_grid = plotcircle(radii = R_sampling, angles = T_sampling, out = 'x,y')
    xy_grid = np.hstack((x_grid[:,None], y_grid[:,None]))
    if Z_sampling is not None:
        xy_grid = np.repeat(xy_grid, Z_sampling.shape[0], axis = 0)
        z_grid = np.tile(Z_sampling,x_grid.shape[0])[:,None]
    
        # create output:
        return np.hstack((xy_grid, z_grid))
    else: 
        return xy_grid


def generate_vector_field(pmodel = None, xyzr = None, xyzt = None, diff_model = True,
                          circle_field = False, make_grid = True, limit_grid_radius = 0,
                          xyz_ranges = np.array([[-100,100,10],[-100,100,10],[0,100,10]]),
                          x_sampling = None, y_sampling = None, z_sampling = None, 
                          RTZ_ranges = np.array([[0,100,10],[0,360,5],[0,100,10]]),
                          R_sampling = None, T_sampling = None, Z_sampling = None, 
                          color = 'k', axh = None, title = None, 
                          axtype = 'polar', use_plt_quiver = True,
                          nTbins = 32, Tbins_start_angle = 0):
    """
    Generates a field of vectors multivariate polynomial model (vectors start at ref. points).
    
    | Has the option to plot vector field.
    
    Args:
        :pmodel:
            | ndarray with model parameters.
            | If None: apply poly_model, else just generate vector-shift field between xr,yr,zr and xt,yt,zt (must be supplied)
        :xyzt:
            | None, optional
            | if not None and poly_model not None: generate vector field between xyzr and xyzt 
        :xyzr:
            | None, optional
            | if make_grid is False: use this to generate vector-shift field
        :circle_field:
            | False, optional
            | Generate a circular grid based vector field instead of a rectangular one.
        :make_grid:
            | True, optional
            | True: generate a 2d-grid from :x_sampling:, :y_sampling: or :R_sampling:, :T_sampling:.
        :limit_grid_radius:
            | 0, optional
            |   A value > 0 only keeps (x,y) coordinates within :limit_grid_radius:
        :xyz_ranges:
            | None or ndarray, optional
            | Specifies the pixelization of the 3D space.
            |  (ndarray.shape = (3,3), with  first axis: x,y,z and second 
            |  axis: min, max, delta)
        :x_sampling:
            | None, optional
            | User defined ndarray with sampling points along x. If not None: overrides xyz_ranges
        :y_sampling:
            | None, optional
            | User defined ndarray with sampling points along y. If not None: overrides xyz_ranges
        :z_sampling:
            | None, optional
            | User defined ndarray with sampling points along z. If not None: overrides xyz_ranges
        :RTZ_ranges:
            | None or ndarray, optional
            | Specifies the uniform pixelization of the 2D space.
            |  (ndarray.shape = (2,3), with  first axis: R,T, and second 
            |  axis: min, max, delta)
        :R_sampling:
            | None, optional
            | User defined ndarray with sequence of values. If not None: overrides RTZ_ranges
        :T_sampling:
            | None, optional
            | User defined ndarray with sequence of . If not None: overrides RTZ_ranges
        :Z_sampling:
            | None, optional
            | User defined ndarray with sequence of . If not None: overrides RTZ_ranges
        :color:
            | None, optional
            | For plotting the vector field.
            | If :color: == False, no plot will be generated.
            | If None: plot shifts in colors related to the hue of the ref coordinates.
        :axh:
            | None, optional
            | axes handle, if None: generate new figure with axes.
        :title:
            | None, optional
            | If not None but string: set axes title.
        :axtype:
            | 'polar' or 'cart', optional
            | Make polar or Cartesian plot.
        :use_plt_quiver:
            | False, optionaluse pyplot's quiver function to plot the vector shifts.
        :nTbins:
            | Number of theta bins to divide the vector field space with each plotted in
            | different theta (theta-discritizaion of color plot)
        :Tbins_start_angle:
            | Start angle of first theta bin in degrees.
    
    Returns:
        :returns: 
            | (xyt),(RTt, RTr)
    """
    RTt, RTr = None, None
    # Generate circle reference field:
    if ((make_grid == True) & (circle_field == True)):
        xyzr = generate_circ_grid(RTZ_ranges = RTZ_ranges,
                                  R_sampling = R_sampling, T_sampling = T_sampling, Z_sampling = Z_sampling,
                                  limit_grid_radius = limit_grid_radius)

    # Generate rectangular reference field:
    elif ((make_grid == True) & (circle_field != True)):
        xyzr = generate_rect_grid(xyz_ranges = xyz_ranges, 
                                 x_sampling = x_sampling, y_sampling = y_sampling, z_sampling = z_sampling, 
                                 limit_grid_radius = limit_grid_radius)
    
    # Use this to plot vectors between:
    elif (xyzr is not None):
        RTr = np.hstack(((xyzr**2).sum(axis=1,keepdims=True)**0.5, np.arctan2(xyzr[:,1:2], xyzr[:,0:1]+_EPS))) #_eps avoid zero-division
   
    # OPtional target coordinates for vector-shifts (only used if poly_model is not None:)    
    if (pmodel is None) & (xyzt is not None) & (make_grid == False):
        RTt = np.hstack(((xyzt**2).sum(axis=1,keepdims=True)**0.5, np.arctan2(xyzt[:,1:2], xyzt[:,0:1]+_EPS)))  # test radial distance and theta angle
        
    if (pmodel is not None):   
        # Apply model at ref. coordinates:
        xyzt, (RTt, RTr) = apply_poly_model_at_x(xyzr, pmodel, diff_model = diff_model, polar_coord = True)

    # plot vector field:
    if (color is not False):
        if (xyzt is not None):
            if xyzr.shape != xyzt.shape:
                raise Exception("xyzr.shape != xyzt.shape: make sure you're not generating a reference grid, while xyzt is not None!")
            plot_vector_field((xyzt, RTt),(xyzr, RTr),
                              color = color, axh = axh, title = title, axtype = axtype,
                              nTbins = nTbins, Tbins_start_angle = Tbins_start_angle, 
                              use_plt_quiver = use_plt_quiver)
        else:
            raise Exception("xyzt is None: cannot plot vector field!")

    return xyzt, (RTt,RTr)


def plot_vector_field(xyzRTt, xyzRTr, color = 'k', axh = None, title = None, axtype = 'polar',
                      nTbins = 32, Tbins_start_angle = 0, use_plt_quiver = True):
    """
    Makes a plot of a vector field (if color is not False). 
    
    | For more info on input parameters, see generate_vector_field?
    
    Returns:
        None if color == False else axes handle
    """
    # Plot vectorfield:
    if color is not False: 
        # unpack vector field data input:
        xyzr, RTr = xyzRTr
        xyzt, RTt = xyzRTt
        if xyzr.shape[1] == 3:
            xr, yr, zr = xyzr[:,0], xyzr[:,1], xyzr[:,2]
            xt, yt, zt = xyzt[:,0], xyzt[:,1], xyzt[:,2]
        else: # make it also work for 2D data
            xr, yr, zr = xyzr[:,0], xyzr[:,1], None 
            xt, yt, zt = xyzt[:,0], xyzt[:,1], None 
            
        
        if (axh == None):
            fig, newfig = plt.figure(), True
        else:
            newfig = False
        
        # Calculate thetas for all grid points to derive plot color:
        thetas = positive_arctan(xr,yr, htype = 'rad')
        
        rect = [0.1, 0.1, 0.8, 0.8] # setting the axis limits in [left, bottom, width, height]
        if (axtype == 'polar') & (zr is None): # only polar plots for 2D data!!
            if newfig == True: axh = fig.add_axes(rect, polar=True, frameon=False) # polar axis
            
            # Get polar coordinates and store in (override)  xr, yr and (xt, yt):
            xr, yr = cart2pol(xr,y=yr, htype = 'rad') # = Tr, Rr
            xt, yt = cart2pol(xt,y=yt, htype = 'rad') # = Tt, Rt
        else:
            if zr is None:
                if newfig == True: axh = fig.add_axes(rect)  # 2D artesian axis
            else:
                if newfig == True: axh = fig.add_axes(rect, projection ='3d')  # 3D cartesian axis
                
                  
        if color is None: # when user doesn't specify color, determine it from theta
            Tbins, hsv_hues = get_theta_bin_edges_and_cmap(nTbins = nTbins, Tbins_start_angle = Tbins_start_angle)
            for i in range(nTbins):
                c = np.abs(np.array(colorsys.hsv_to_rgb(hsv_hues[i], 0.84, 0.9)))
                xri = xr[(thetas>=Tbins[i])&(thetas<Tbins[i+1])]
                yri = yr[(thetas>=Tbins[i])&(thetas<Tbins[i+1])]
                xti = xt[(thetas>=Tbins[i])&(thetas<Tbins[i+1])]
                yti = yt[(thetas>=Tbins[i])&(thetas<Tbins[i+1])]
                if zr is not None:
                    zri = zr[(thetas>=Tbins[i])&(thetas<Tbins[i+1])]
                    zti = zt[(thetas>=Tbins[i])&(thetas<Tbins[i+1])]
                    if use_plt_quiver:
                        axh.quiver(xri, yri, zri, xti-xri, yti-yri, zti-zri, edgecolor = c,
                                   facecolor = c, arrow_length_ratio=0.2,linewidth = 0.01)
                                   #angles='uv',scale_units='y',scale = 0.4,width=0.005) # these keywords don't seem to be supported by Axes3D.quiver !!!
                    else:
                        for j in range(xri.shape[0]):
                            axh.plot(np.array([xri[j]]),np.array([yri[j]]), zs=np.array([zri[j]]), color = c, marker='.',linestyle='none') # plot a dot at the reference position
                            axh.plot(np.hstack((xri[j],xti[j])), np.hstack((yri[j],yti[j])), zs = np.hstack((zri[j],zti[j])), color = c, marker=None, linestyle='-')

                else:
                    if use_plt_quiver:
                        axh.quiver(xri, yri, xti-xri, yti-yri, edgecolor = c,
                                   facecolor = c, headlength=3, angles='uv', 
                                   scale_units='y', scale = 0.4,width=0.005,linewidth = 0.01)
                    else:
                        for j in range(xri.shape[0]):
                            axh.plot(xri[j],yri[j],color = c, marker='.',linestyle='none') # plot a dot at the reference position
                            axh.plot(np.vstack((xri[j],xti[j])),np.vstack((yri[j],yti[j])),
                                     color = c, marker=None, linestyle='-')
        else: 
            if zr is not None:
                if use_plt_quiver:
                    axh.quiver(xr, yr, zr, xt-xr, yt-yr, zt - zr, angles='uv', scale_units='y', headlength=1, scale = 0.4,color = color)
                else:
                    for j in range(xr.shape[0]):
                        axh.plot(xr[j],yr[j],zr[j],color = color, marker='.',) # plot a dot at the reference position
                        axh.plot(np.vstack((xr[j],xt[j])),np.vstack((yr[j],yt[j])),np.vstack((zr[j],zt[j])),
                                 color = color, marker=None, linestyle='-')        
            else:
                if use_plt_quiver:
                    axh.quiver(xr, yr, xt-xr, yt-yr, angles='uv', scale_units='y', headlength=1, scale = 0.4,color = color)
                else:
                    for j in range(xr.shape[0]):
                        axh.plot(xr[j],yr[j],color = color, marker='.',) # plot a dot at the reference position
                        axh.plot(np.vstack((xr[j],xt[j])),np.vstack((yr[j],yt[j])),
                                 color = color, marker=None, linestyle='-')        
        axh.set_xlabel("x")
        axh.set_ylabel("y")
        if zr is not None:
            axh.set_zlabel("z")
        if title is not None:
            axh.set_title(title)
        return axh
    return None
        

def shiftvectors(xyzt,xyzr, average = True,):
    """
    Calculate xyz-shift vectors.
    
    Args:
        :xyt: 
            | ndarray with target xyz coordinates 
        :xyr:
            | ndarray with reference ab coordinates 
        :average:
            | True, optional
            | If True, take mean of difference vectors along axis = 0.
            
    Returns:
        :returns:
            | ndarray of (mean) shift vector(s). 
    """    
    v =  xyzt - xyzr
    if average == True:
        v = v.mean(axis=0)
    return v   


#------------------------------------------------------------------------------
# Plotting related functions:
#------------------------------------------------------------------------------


def get_theta_bin_edges_and_cmap(nTbins = 16, Tbins_start_angle = 0):
    """
    Get theta bin edges (in rad) and a list of rgb values (color map) for plotting each theta bin.
    """
    dTbins = 360/(nTbins) # theta bin width
    Tbincenters = np.sort(np.arange(Tbins_start_angle + dTbins/2, 360, dTbins))
    dL = np.arange(Tbins_start_angle,360-dTbins+1,dTbins)
    
    # Setup color for plotting theta bins:
    hsv_hues = Tbincenters - 30
    hsv_hues = hsv_hues/hsv_hues.max()
    
    return np.hstack((dL,360))*np.pi/180, hsv_hues

def plotcircle(center = np.array([[0.,0.]]),radii = np.arange(0,60,10), 
               angles = np.arange(0,350,10),color = 'k',linestyle = '--', 
               out = None, axh = None):
    """
    Plot one or more concentric circles.
    
    Args:
        :center: 
            | np.array([[0.,0.]]) or ndarray with center coordinates, optional
        :radii:
            | np.arange(0,60,10) or ndarray with radii of circle(s), optional
        :angles:
            | np.arange(0,350,10) or ndarray with angles (°), optional
        :color:
            | 'k', optional
            | Color for plotting.
        :linestyle:
            | '--', optional
            | Linestyle of circles.
        :out: 
            | None, optional
            | If None: plot circles, return (x,y) otherwise.
    """
    xs = np.array([0])
    ys = xs.copy()
    if ((out != 'x,y') & (axh is None)):
        fig, axh = plt.subplots(rows=1,ncols=1)
    for ri in radii:
        x = center[:,0] + ri*np.cos(angles*np.pi/180)
        y = center[:,1] + ri*np.sin(angles*np.pi/180)
        xs = np.hstack((xs,x))
        ys = np.hstack((ys,y))
        if (out != 'x,y'):
            axh[0].plot(x,y,color = color, linestyle = linestyle)
    if out == 'x,y':
        return xs,ys
    

#------------------------------------------------------------------------------
# Multivariate Poly_Model Class:
#------------------------------------------------------------------------------

class MuPolyModel:
    """
    Multivariate Polynomial Model Class
    """
    def __init__(self, xyzt = None, xyzr = None, pmodel = None, npar = 10, get_stats = True, diff_model = True, polar_coord = False):
        """
        Initialize / get the poly_model.
        
        Args:
            :xyzt: 
                | None, optional
                | ndarray with target coordinates (to be predicted starting from reference xyzr).
                | If xyzt & xytr & pmodel is None: empty initialization.
                | If xyzt & xytr is None and pmodel is not None: initialized using pmodel parameters only.
            :xyzr: 
                | None, optional
                | ndarray with reference coordinates (to be transformed to predictions of xyzt).
                | If xyzt & xytr & pmodel is None: empty initialization.
                | If xyzt & xytr is None and pmodel is not None: initialized using pmodel parameters only.
            :pmodel:
                | None, optional
                | If not None (and xyzt & xyzr is None): initialize using pre-computed model parameters in pmodel.
            :npar:
                | Specifies the number of parameters of the polynomial model in xy-coordinates 
                | (npar = 9 or 10 for 3D data, 5 or 6 for 2D data;
                |  9 and 5 omit the constant term; see notes below)
            :get_stats:
                | True, optional
                | Calculate model statistics: dxyz_pred, dxyz_res, dxyz_res_std, dRToverR_pred, dRToverR_res,  dRToverR_res_std
                | If False: fill with None's 
                | See :returns: for more info on statistics.
            :dict_out:
                | True, optional
                | Get function output as dict instead of tuple.
            :polar_coord:
                | False, optional
                | If True: also calculate dR/R (R=radial distance), dT (T=theta) (only when get_stat == True !! )
                  
        Returns:   
            :returns: 
                | Dict or tuple with:
                |  - 'poly_model' : function handle to model
                |  - 'p' : ndarray with model parameters
                |  - 'M' : optimization matrix
                |  - 'dxyz pred' : ndarray with dxy model predictions from xr, yr, zr.
                |  - 'dxyz res' : ndarray with residuals between 'dx,dy,dz' of samples and 
                |                'dx,dy,dz' predicted by the model.
                |  - 'dxyz res std' : ndarray with std of 'dxyz res'
                |  - 'RTred' : ndarray with Radial distance and Theta angles of reference.
                |  - 'dR/R,dT pred' : ndarray with predictions for dR/R = (Rt - Rr)/Rr and dT = ht - hr
                |  - 'dR/R,dT res' : ndarray with residuals between 'dR/R,dT' 
                |                      of samples and 'dR/R,dT' predicted by the model.
                |  - 'dR/R,dT res std' : ndarray with std of 'dR/R,dT res: 
    
        Notes: 
            1. Model types:
                2D-data:
                | poly_model (n = 5):         p[0]*x + p[1]*y + p[2]*(x**2) + p[3]*(y**2) + p[4]*x*y 
                | poly_model (n = 6):  p[0] + p[1]*x + p[2]*y + p[3]*(x**2) + p[4]*(y**2) + p[5]*x*y 
                3D-data:
                | poly_model (n = 9):          p[0]*x + p[1]*y + p[2]*y + p[3]*(x**2) + p[4]*(y**2) + p[5]*(x**2) + p[6]*x*y + p[7]*x*z + p[8]*y*z 
                | poly_model (n = 10):  p[0] + p[1]*x + p[2]*y + p[3]*y + p[4]*(x**2) + p[5]*(y**2) + p[5]*(x**2) + p[7]*x*y + p[8]*x*z + p[9]*y*z 
            
            2. Calculation of dRoverR and dT:
                | dRoverR = (np.cos(Tr)*dx + np.sin(Tr)*dy)/Rr
                | dToverC = (np.cos(Tr)*dy - np.sin(Tr)*dx)/Rr   

        """
        self.model_type = None
        self.model = None
        self.p = None
        self.data = None
        self.stats = None
        self.polar_coord = polar_coord
        self.diff_model = diff_model
        if ((xyzt is None) & (xyzr is None)) & (pmodel is None):
             self.initialized = False
        else:
            if (xyzt is not None) & (xyzr is not None):
                n_par = xyzt.shape[1]*4 - 2 
                if (npar != n_par) & (npar != n_par - 1):
                    raise Exception('npar input argument not consistent with shape of xyz input. \n2D data: npar = 5 or 6, 3D data: npar = 9 or 10')
                (poly_model_, pmodel, M,
                dxyz_pred, dxyz_res, dxyz_res_std, 
                RTref, dRToverR_pred, dRToverR_res, dRToverR_res_std) = get_poly_model(xyzt, xyzr, npar = npar, 
                                                                                 get_stats = get_stats, 
                                                                                 polar_coord = polar_coord, 
                                                                                 dict_out=False,
                                                                                 diff_model = diff_model)
                self.initialized = True
                self.model = poly_model_
                self.p = np.atleast_2d(pmodel)
                self.npar = npar
                self.M = M
                self.data = {'xyzt':xyzt, 'xyzr':xyzr}
                self.stats = {'dxyz pred': dxyz_pred, 
                              'dxyz res': dxyz_res, 
                              'dxyz res std': dxyz_res_std,
                              'RTref':RTref,
                              'dR/R,dT pred': dRToverR_pred, 
                              'dR/R,dT res': dRToverR_res, 
                              'dR/R,dT res std': dRToverR_res_std}
            elif (pmodel is not None):
                self.initialized = True
                self.model = poly_model
                self.p = np.atleast_2d(pmodel)
                self.npar = len(pmodel)
                self.data = None
                self.stats = None
                
    def apply(self, xyzr = None, polar_coord = False, diff_model = None, out = 'xyzt,(RTt,RTr)'):
        """
        Apply the poly_model at coordinates in xyzr. 
        See apply_poly_model_at_x? for more info on input arguments.
        
        Returns:
            | (xyzt),(RTt,RTr)
        """
        if (xyzr is None):
            xyzr = self.data['xyzr']
        if polar_coord is None:
            polar_coord = self.polar_coord
        if diff_model is None:
            diff_model = self.diff_model
        return apply_poly_model_at_x(xyzr, self.p, polar_coord = polar_coord, diff_model = diff_model, out = out)

    def generate_vector_field(self, xyzr = None, xyzt = None, diff_model = True,
                          circle_field = False, make_grid = True, limit_grid_radius = 0,
                          xyz_ranges = np.array([[-100,100,10],[-100,100,10],[0,100,10]]),
                          x_sampling = None, y_sampling = None, z_sampling = None, 
                          RTZ_ranges = np.array([[0,100,10],[0,360,5],[0,100,10]]),
                          R_sampling = None, T_sampling = None, Z_sampling = None,
                          color = None, axh = None, title = None, axtype = 'polar', use_plt_quiver = True,
                          nTbins = 32, Tbins_start_angle = 0):
        """
        Generate a vector field. 
        For more info see generate_vector_field?
        
        Returns:
            | xyzt,(RTt,RTr)
        """
        return generate_vector_field(pmodel = self.p, 
                                     xyzr = xyzr, xyzt = xyzt, diff_model = diff_model,
                                     circle_field = circle_field, make_grid = make_grid, limit_grid_radius = limit_grid_radius,
                                     xyz_ranges = xyz_ranges,
                                     x_sampling = x_sampling, y_sampling = y_sampling, z_sampling = z_sampling, 
                                     RTZ_ranges = RTZ_ranges,
                                     R_sampling = R_sampling, T_sampling = T_sampling, Z_sampling = Z_sampling, 
                                     color = color, axh = axh, title = title, axtype = axtype, 
                                     use_plt_quiver = use_plt_quiver,
                                     nTbins = nTbins, Tbins_start_angle = Tbins_start_angle)        

    
if __name__ == '__main__':
    
    import luxpy as lx
    import pandas as pd
    
    run_example_1 = True
    run_example_2 = False
    run_example_0 = True
    
    diff_model = False
    #--------------------------------------------------------------------------
    # EXAMPLE 0: test
    #--------------------------------------------------------------------------
    if run_example_0:
        pass
    
    #--------------------------------------------------------------------------
    # EXAMPLE 1: as shift model
    #--------------------------------------------------------------------------
    if run_example_1:
        # Generate_test_data:
        F4 = lx._CIE_ILLUMINANTS['F4'].copy() 
        M = lx._MUNSELL.copy()
        rflM = M['R']
        rflM = lx.cie_interp(rflM,F4[0],kind='rfl')
        xyz31, xyzw31 = lx.spd_to_xyz(F4, cieobs = '1931_2', relative = True, rfl = rflM, out = 2)
        xyz06, xyzw06 = lx.spd_to_xyz(F4, cieobs = '2006_2', relative = True, rfl = rflM, out = 2)
        
        # For 2D modeling:
        ab31 = lx.xyz_to_lab(xyz31, xyzw = xyzw31)[:,0,1:]
        ab06 = lx.xyz_to_lab(xyz06, xyzw = xyzw06)[:,0,1:]
        
        # For 3D modeling:
        abL31 = lx.xyz_to_lab(xyz31, xyzw = xyzw31)[:,0,[1,2,0]]
        abL06 = lx.xyz_to_lab(xyz06, xyzw = xyzw06)[:,0,[1,2,0]]
    
        
        # Get model that characterizes shift between 1931 and 2006 2° CMFs 
        # based on relative sample shifts under the two sets:
        pm = MuPolyModel(ab31, ab06, npar = 6, diff_model = diff_model) # 2D fit
        pmL = MuPolyModel(abL31, abL06, npar = 10, diff_model = diff_model) # 3D fit
        
        # Create a grid of new data points 
        # and plot shifts in hue angle related colors (by seting Color = None):
        pm.generate_vector_field(circle_field = True, diff_model = diff_model,
                                 xyz_ranges = np.array([[-100,100,10],[-100,100,10]]),
                                 RTZ_ranges = np.array([[0,100,10],[0,360,10]]),
                                 make_grid = True, limit_grid_radius = 0,
                                 color = None, axh = None, axtype = 'cart', use_plt_quiver = True,
                                 nTbins = 32, Tbins_start_angle = 0, title = 'Test MuPolyModel_2D')
        
        
        #Including 3e dimension z:
        pmL.generate_vector_field(circle_field = True, diff_model = diff_model,
                                 xyz_ranges = np.array([[-100,100,10],[-100,100,10],[0,100,10]]),
                                 RTZ_ranges = np.array([[0,100,10],[0,360,10],[0,100,20]]),
                                 make_grid = True, limit_grid_radius = 0,
                                 color = None, axh = None, axtype = 'cart', use_plt_quiver = False,
                                 nTbins = 32, Tbins_start_angle = 0, title = 'Test MuPolyModel_3D')
        
        # or, generate some rect. xyz-grid:
        xyzr_ = generate_rect_grid(xyz_ranges = np.array([[-100,100,10],[-100,100,10],[0,100,10]]), 
                                   limit_grid_radius = 0)
        # Apply shift to grid:
        xyzt_ = apply_poly_model_at_x(xyzr_, pmL.p,polar_coord=True, diff_model = diff_model)[0]
        fig = plt.figure()
        axh = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection ='3d')
        axh.plot(xyzr_[:,0],xyzr_[:,1],xyzr_[:,2],'ro')
        axh.plot(xyzt_[:,0],xyzt_[:,1],xyzt_[:,2],'b.');
    
        # Apply shift to data used to obtain model parameters:
        xyzt = apply_poly_model_at_x(pmL.data['xyzr'], pmL.p, polar_coord=True, diff_model = diff_model)[0]
        fig2 = plt.figure()
        axh2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8], projection ='3d')
        axh2.plot(pmL.data['xyzr'][:,0],pmL.data['xyzr'][:,1],pmL.data['xyzr'][:,2],'ro');
        axh2.plot(xyzt[:,0],xyzt[:,1],xyzt[:,2],'b.');
        axh2.plot(abL31[:,0],abL31[:,1],abL31[:,2],'g.'); 
    
    
    #--------------------------------------------------------------------------
    # EXAMPLE 2: as transformation (example with 2D data, but should work for 3D)
    #--------------------------------------------------------------------------
    if run_example_2:
        ab_test = pd.read_csv('./data/bipolymodeltests/ab_test.dat',header=None,sep='\t').values # final ab SWW CAM signal
        ab_ref = pd.read_csv('./data/bipolymodeltests/ab_ref.dat',header=None,sep='\t').values   # final CIECAM02
        dLMS = pd.read_csv('./data/bipolymodeltests/dLMS.dat',header=None,sep='\t').values  # L-M, (L+M)/2 signal of SWW from which final ab signal should be predicted
     
        # Generate forward and reverse models:
        # pm2 = LMS -> ab:
        pm2 = MuPolyModel(ab_ref, dLMS, npar = 6, diff_model = diff_model) 
        # pm2i = ab -> LMS:
        pm2i = MuPolyModel(dLMS, ab_ref, npar = 6, diff_model = diff_model) 
        
        # appply pm2 to dLMS data to go to ab-space, then apply pm2i to return to dLMS:
        abt2 = apply_poly_model_at_x(dLMS, pm2.p, polar_coord=True, diff_model = diff_model)[0]
        abt2i = apply_poly_model_at_x(abt2, pm2i.p, polar_coord=True, diff_model = diff_model)[0]
        plt.figure()
        plt.plot(dLMS[:,0],dLMS[:,1],'ro')
        plt.plot(abt2[:,0],abt2[:,1] ,'b.');
        plt.plot(abt2i[:,0],abt2i[:,1],'g+')
        
        # Apply generated models in inverse order to go from ab -> LMS -> ab
        # Generate large grid in ab-space:
        abr_2 = generate_rect_grid(xyz_ranges = np.array([[-30,30,5],[-30,30,5]]), x_sampling = None, y_sampling = None, 
                                   limit_grid_radius = 0)
        # appply pm2i to ab-space to go to dLMS space, then apply pm2 to return to ab-space:
        abt_2i = apply_poly_model_at_x(abr_2, pm2i.p, polar_coord=True, diff_model = diff_model)[0]
        abt_2 = apply_poly_model_at_x(abt_2i, pm2.p, polar_coord=True, diff_model = diff_model)[0]
        plt.figure()
        plt.plot(abr_2[:,0], abr_2[:,1],'ro')
        plt.plot(abt_2i[:,0],abt_2i[:,1],'b.')
        plt.plot(abt_2[:,0], abt_2[:,1] ,'g+');
        
     