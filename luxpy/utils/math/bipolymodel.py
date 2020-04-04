# -*- coding: utf-8 -*-
"""
Bivariate polynomial model
==========================

 :get_poly_model(): Get bivariate polynomial model parameters.
     
 :apply_poly_model_at_x(): Applies bivariate polynomial model at cartesian reference coordinates.
 
 :generate_rect_grid(): Generate a rectangular grid of 2D Cartesian coordinates.
 
 :generate_circ_grid(): Generate a circular grid of 2D Cartesian coordinates.

 :generate_vector_field(): Generates a field of vectors using the bivariate polynomial model.
     
 :plot_vector_field(): Makes a plot of a vector field.
     
 :BiPolyModel(): Bivariate Polynomial Model Class


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import numpy as np
import matplotlib.pyplot as plt
import colorsys

#from luxpy.utils.math import (cart2pol, positive_arctan)
from . import (cart2pol, positive_arctan)

_EPS = np.finfo('float').eps # used in model to avoid division by zero ! 

__all__ = ['get_poly_model','apply_poly_model_at_x',
           'generate_rect_grid','generate_circ_grid','generate_vector_field',
           'plot_vector_field','BiPolyModel']

#------------------------------------------------------------------------------
# Bivariate Poly_Model functions:
#------------------------------------------------------------------------------  
def _poly_model_constructor(model_type):
    if not isinstance(model_type,str): # if not a string model_type contains parameters
        model_type = 'M{:1.0f}'.format(model_type.shape[1])
    if model_type == 'M5':
        poly_model = lambda x,y,p: p[0]*x + p[1]*y + p[2]*(x**2) + p[3]*x*y + p[4]*(y**2)
    elif model_type == 'M6':
        poly_model = lambda x,y,p: p[0] + p[1]*x + p[2]*y + p[3]*(x**2) + p[4]*x*y + p[5]*(y**2)
    return poly_model


def get_poly_model(xyt, xyr, model_type = 'M6', get_stats = True, dict_out = True, polar_coord = False):
    """
    Get bivariate polynomial model parameters.
    
    Args:
        :xyt: 
            | ndarray with target coordinates (to be predicted starting from reference xyr).
        :xyr: 
            | ndarray with reference coordinates (to be transformed to predictions of xyt).
        :model_type:
            | 'M6' or 'M5', optional
            | Specifies degree 5 or degree 6 polynomial model in xy-coordinates.
              (see notes below)
        :get_stats:
            | True, optional
            | Calculate model statistics: dxy_pred, dxy_res, dxy_res_std, dRToverR_pred, dRToverR_res,  dRToverR_res_std
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
            |  - 'dxy pred' : ndarray with dab model predictions from ar, br.
            |  - 'dxy res' : ndarray with residuals between 'dx,dy' of samples and 
            |                'dx,dy' predicted by the model.
            |  - 'dxy res std' : ndarray with std of 'dxy res'
            |  - 'dR/R,dT pred' : ndarray with predictions for dR/R = (Rt - Rr)/Rr and dT = ht - hr
            |  - 'dR/R,dT res' : ndarray with residuals between 'dR/R,dT' 
            |                      of samples and 'dR/R,dT' predicted by the model.
            |  - 'dR/R,dT res std' : ndarray with std of 'dR/R,dT res: 

    Notes: 
        1. Model types:
            | poly5_model = lambda x,y,p:         p[0]*x + p[1]*y + p[2]*(x**2) + p[3]*x*y + p[4]*(y**2)
            | poly6_model = lambda x,y,p:  p[0] + p[1]*x + p[2]*y + p[3]*(x**2) + p[4]*x*y + p[5]*(x**2)
        
        2. Calculation of dRoverR and dT:
            | dRoverR = (np.cos(Tr)*dx + np.sin(Tr)*dy)/Rr
            | dToverC = (np.cos(Tr)*dy - np.sin(Tr)*dx)/Rr   
    """
    xt, yt = xyt[...,0], xyt[...,1]
    xr, yr = xyr[...,0], xyr[...,1]
    
    # A. Calculate dx, dy:
    dx, dy = xt - xr, yt - yr
    
    # B.1 Calculate model matrix:
    # 5-parameter model:
    M = np.array([[np.sum(xr*xr), np.sum(xr*yr), np.sum(xr*xr**2),np.sum(xr*xr*yr),np.sum(xr*yr**2)],
            [np.sum(yr*xr), np.sum(yr*yr), np.sum(yr*xr**2),np.sum(yr*xr*yr),np.sum(yr*yr**2)],
            [np.sum((xr**2)*xr), np.sum((xr**2)*yr), np.sum((xr**2)*xr**2),np.sum((xr**2)*xr*yr),np.sum((xr**2)*yr**2)],
            [np.sum(xr*yr*xr), np.sum(xr*yr*yr), np.sum(xr*yr*xr**2),np.sum(xr*yr*xr*yr),np.sum(xr*yr*yr**2)],
            [np.sum((yr**2)*xr), np.sum((yr**2)*yr), np.sum((yr**2)*xr**2),np.sum((yr**2)*xr*yr),np.sum((yr**2)*yr**2)]])
    
    #6-parameters model:
    if model_type == 'M6': 
        M = np.vstack((np.array([[np.sum(1.0*xr),np.sum(1.0*yr), np.sum(1.0*xr**2),np.sum(1.0*xr*yr),np.sum(1.0*yr**2)]]), M)) # add row 0 of M6
        M = np.hstack((np.array([[xr.size,np.sum(xr*1.0),np.sum(yr*1.0),np.sum((xr**2)*1.0),np.sum(xr*yr*1.0),np.sum((yr**2)*1.0)]]).T,M)) # add col 0 of M6
        
    # Get inverse matrix:
    M = np.linalg.inv(M)
    
    # B.2 Define model function using constructor function:
    poly_model = _poly_model_constructor(model_type)

    # C.1 Data x,y analysis output:
    if model_type == 'M5':
        dx_model_parameters = np.dot(M, np.array([np.sum(dx*xr), np.sum(dx*yr), np.sum(dx*xr**2),np.sum(dx*xr*yr),np.sum(dx*yr**2)]))
        dy_model_parameters = np.dot(M, np.array([np.sum(dy*xr), np.sum(dy*yr), np.sum(dy*xr**2),np.sum(dy*xr*yr),np.sum(dy*yr**2)]))
    elif model_type == 'M6':
        dx_model_parameters = np.dot(M, np.array([np.sum(dx*1.0),np.sum(dx*xr), np.sum(dx*yr), np.sum(dx*xr**2),np.sum(dx*xr*yr),np.sum(dx*yr**2)]))
        dy_model_parameters = np.dot(M, np.array([np.sum(dy*1.0),np.sum(dy*xr), np.sum(dy*yr), np.sum(dy*xr**2),np.sum(dy*xr*yr),np.sum(dy*yr**2)]))
    pmodel = np.vstack((dx_model_parameters,dy_model_parameters))

    if get_stats == True:
        # D.1 Calculate predicted dx, dy (with resp. to ref.):
        dx_pred, dy_pred = poly_model(xr,yr,pmodel[0]), poly_model(xr,yr,pmodel[1])
        dxy_pred = np.hstack((dx_pred,dy_pred))
    
        # D.2 Calculate residuals for dx & dy:
        dx_res, dy_res = dx - dx_pred, dy - dy_pred
        dxy_res = np.hstack((dx_res, dy_res))
        dxy_res_std = np.vstack((np.std(dx_res,axis=0),np.std(dy_res,axis=0)))

        # E.1 Calculate href, Cref:
        href, Cref = np.arctan2(yr,xr), (xr**2 + yr**2)**0.5

        # E.2 Calculate res and std of dR/R, dH/C:
        dRoverR, dToverR  = (np.cos(href)*dx + np.sin(href)*dy)/Cref, (np.cos(href)*dy - np.sin(href)*dx)/Cref
        dRoverR_pred, dToverR_pred = (np.cos(href)*dx_pred + np.sin(href)*dy_pred)/Cref, (np.cos(href)*dy_pred - np.sin(href)*dx_pred)/Cref
        dRoverR_res, dToverR_res = dRoverR - dRoverR_pred, dToverR - dToverR_pred
        dRToverR_pred = np.vstack((dRoverR_pred, dToverR_pred))
        dRToverR_res_std = np.vstack((np.std(dRoverR_res,axis = 0),np.std(dToverR_res,axis = 0)))
        dRToverR_res = np.hstack((href,dRoverR_res,dToverR_res))
    else:
        dxy_pred, dxy_res, dxy_res_std, dRToverR_pred, dRToverR_res, dRToverR_res_std = None, None, None, None, None, None 

    if dict_out == True:
        return {'poly_model' : poly_model, 
                'p' : pmodel, 
                'M' : M,
                'dxy pred': dxy_pred, 
                'dxy res': dxy_res, 
                'dxy res std': dxy_res_std,
                'dR/R,dT pred': dRToverR_pred, 
                'dR/R,dT res': dRToverR_res, 
                'dR/R,dT res std': dRToverR_res_std}
    else:
        return (poly_model, pmodel, M,
                dxy_pred, dxy_res, dxy_res_std, 
                dRToverR_pred, dRToverR_res, dRToverR_res_std)


def apply_poly_model_at_x(pmodel, xr, yr, poly_model = None,
                          polar_coord = False):
    """
    Applies bivariate polynomial model at cartesian reference coordinates.
    
    Args:
        :pmodel:
            | ndarray with model parameters.
        :xr: 
            | ndarray with reference x-coordinates
        :yr:
            | ndarray with reference y-coordinates
        :poly_model: 
            | function handle to model (if None: construct new lambda fcn.)
        :polar_coord:
            | False, optional
            | If True: also calculate R(adial distance) and T(heta angle) for xt and yt.
        
    Returns:
        :returns:
            | (xt,yt),((Rt,Tt),(Rr,Tr))
            | 
            | (xt,yt) ndarrays with predicted xy-coordinates, 
            | (Rt,Tt) radial distance and angle predicted by the model (xyt)
            | (Rr,Tr) radial distance and angle for reference  (xyr)
    """
    # Create lambda function for model:
    if poly_model is None:
        poly_model = _poly_model_constructor(pmodel)
   
    # A Set 2nd order color multipliers (shift parameters for x and y: px & py):
    px, py = pmodel[0].copy(), pmodel[1].copy()
    
    isM6 = px.shape[0] == 6
    px[0 + isM6*1] = 1 + px[0 + isM6*1]
    py[1 + isM6*1] = 1 + py[1 + isM6*1]
    
    # B Apply model to reference coordinates using 2nd order multipliers:
    xt,yt = poly_model(xr,yr,px), poly_model(xr,yr,py) 
    
    if polar_coord == True:
        # C Calculate R and T:
        Rr, Tr = np.sqrt(xr**2 + yr**2), np.arctan(yr/(xr+_EPS)) #_eps avoid zero-division
        Rt, Tt = np.sqrt(xt**2 + yt**2), np.arctan(yt/(xt+_EPS)) # test radial distance and theta angle
    else:
        Rr, Tr, Rt, Tt = None, None, None, None
    return (xt,yt),((Rt,Tt),(Rr,Tr))


#------------------------------------------------------------------------------
# Grid and vector-shift field generators:
#------------------------------------------------------------------------------

def generate_rect_grid(xy_ranges = np.array([[-100,100,10],[-100,100,10]]),
                       x_sampling = None, y_sampling = None, 
                       limit_grid_radius = 0, out = 'grid'):
    """
    Generate a rectangular grid of 2D cart. coordinates.
    
    Args:
        :xy_ranges:
            | None or ndarray, optional
            | Specifies the uniform pixelization of the 2D space.
            |  (ndarray.shape = (2,3), with  first axis: x,y, and second 
            |  axis: min, max, delta)
        :x_sampling:
            | None, optional
            | User defined ndarray with sequence of values. If not None: overrides xy_ranges
        :y_sampling:
            | None, optional
            | User defined ndarray with sequence of . If not None: overrides xy_ranges
        :limit_grid_radius:
            | 0, optional
            | A value of zeros keeps grid as specified  by xr,yr.
            | A value > 0 only keeps (x,y) coordinates within :limit_grid_radius:
        :out:
            | 'grid' or 'vectors', optional
            |   - 'grid': outputs a single 2d numpy.nd-vector with the grid coordinates
            |   - 'vector': outputs each dimension seperately.
            
    Returns:
        :returns: 
            | single ndarray with x_grid,y_grid ; the grid points along each dimension
            |  or
            | seperate ndarrays for each dimension specified.
    """
    # generate grid from xy_ranges array input, otherwise use x_sampling, y_sampling, input:
    if (x_sampling is None) | (y_sampling is None):
        x_sampling = np.arange(xy_ranges[0][0],xy_ranges[0][1],xy_ranges[0][2])
        y_sampling = np.arange(xy_ranges[1][0],xy_ranges[1][1],xy_ranges[1][2])
   
    # Generate grid from x_sampling, y_sampling:
    X,Y = np.meshgrid(x_sampling,y_sampling)
    grid = np.dstack((X,Y))
    grid = np.reshape(grid,(np.array(grid.shape[:-1]).prod(),grid.ndim-1))
    x_grid = grid[:,0:1]
    y_grid = grid[:,1:2]

    if limit_grid_radius > 0:# limit radius of grid:
        Rr = (x_grid**2+y_grid**2)**0.5
        x_grid = x_grid[Rr<=limit_grid_radius,None]
        y_grid = y_grid[Rr<=limit_grid_radius,None]
    
    # create output:
    if out == 'grid':
        return np.hstack((x_grid,y_grid))
    else:
        return x_grid, y_grid
    

def generate_circ_grid(RT_ranges = np.array([[0,100,10],[0,360,5]]),
                       R_sampling = None, T_sampling = None, 
                       limit_grid_radius = 0, out = 'grid'):
    """
    Generate a circular sampled grid of 2D cart. coordinates.
    
    Args:
        :RT_ranges:
            | None or ndarray, optional
            | Specifies the uniform pixelization of the 2D space.
            |  (ndarray.shape = (2,3), with  first axis: R,T, and second 
            |  axis: min, max, delta)
        :R_sampling:
            | None, optional
            | User defined ndarray with sequence of values. If not None: overrides RT_ranges
        :T_sampling:
            | None, optional
            | User defined ndarray with sequence of . If not None: overrides RT_ranges
        :limit_grid_radius:
            | 0, optional
            | A value > 0 only keeps (x,y) coordinates within :limit_grid_radius:
        :out:
            | 'grid' or 'vectors', optional
            |   - 'grid': outputs a single 2d numpy.nd-vector with the grid coordinates
            |   - 'vector': outputs each dimension seperately.
            
    Returns:
        :returns: 
            | single ndarray with x_grid,y_grid ; the grid points along each dimension
            |  or
            | seperate ndarrays for each dimension specified.
    """
    # generate grid from RT_ranges array input, otherwise use R_sampling, T_sampling, input:
    if (R_sampling is None) | (R_sampling is None):
        R_sampling = np.arange(RT_ranges[0][0],RT_ranges[0][1],RT_ranges[0][2])
        T_sampling = np.arange(RT_ranges[1][0],RT_ranges[1][1],RT_ranges[1][2])
    x_grid, y_grid = plotcircle(radii = R_sampling, angles = T_sampling, out = 'x,y')
    
    # create output:
    if out == 'grid':
        return np.hstack((x_grid[:,None],y_grid[:,None]))
    else:
        return x_grid[:,None], y_grid[:,None]


def generate_vector_field(poly_model = None, pmodel = None, xr = None, yr = None, xt = None, yt = None,
                          circle_field = False, make_grid = True, limit_grid_radius = 0,
                          xy_ranges = np.array([[-100,100,10],[-100,100,10]]),
                          x_sampling = None, y_sampling = None, 
                          RT_ranges = np.array([[0,100,10],[0,360,5]]),
                          R_sampling = None, T_sampling = None, 
                          color = 'k', axh = None, title = None, 
                          axtype = 'polar', use_plt_quiver = True,
                          nTbins = 32, Tbins_start_angle = 0):
    """
    Generates a field of vectors bivariate polynomial model (vectors start at ref. points).
    
    | Has the option to plot vector field.
    
    Args:
        :poly_model: 
            | function handle to model
            | If None: apply model, else just generate vector-shift field between xr,yr and xt,yt (must be supplied)
        :pmodel:
            | ndarray with model parameters.
        :xt, yt:
            | None, optional
            | if not None and poly_model not None: generate vector field between xr,yr and xt,yt 
        :xr, yr:
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
        :xy_ranges:
            | None or ndarray, optional
            | Specifies the pixelization of the 2D space.
            |  (ndarray.shape = (2,3), with  first axis: a,b, and second 
            |  axis: min, max, delta)
        :x_sampling:
            | None, optional
            | User defined ndarray with sampling points along x. If not None: overrides xy_ranges
        :y_sampling:
            | None, optional
            | User defined ndarray with sampling points along y. If not None: overrides xy_ranges
        :RT_ranges:
            | None or ndarray, optional
            | Specifies the uniform pixelization of the 2D space.
            |  (ndarray.shape = (2,3), with  first axis: R,T, and second 
            |  axis: min, max, delta)
        :R_sampling:
            | None, optional
            | User defined ndarray with sequence of values. If not None: overrides RT_ranges
        :T_sampling:
            | None, optional
            | User defined ndarray with sequence of . If not None: overrides RT_ranges
        :color:
            | None, optional
            | For plotting the vector field.
            | If :color: == 0, no plot will be generated.
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
            | (xt,yt),((Rt,Tt),(Rr,Tr))
    """
    # Generate circle reference field:
    if ((make_grid == True) & (circle_field == True)):
        xr, yr = generate_circ_grid(RT_ranges = RT_ranges,
                                    R_sampling = R_sampling, T_sampling = T_sampling, 
                                    limit_grid_radius = limit_grid_radius, out = 'x_grid, y_grid')

    # Generate rectangular reference field:
    elif ((make_grid == True) & (circle_field != True)):
        xr, yr = generate_rect_grid(xy_ranges = xy_ranges, x_sampling = x_sampling, y_sampling = y_sampling, 
                                    out = 'x_grid, y_grid', limit_grid_radius = limit_grid_radius)
    
    # Use this to plot vectors between:
    elif (xr is not None) & (yr is not None):
        Rr, Tr = np.sqrt(xr**2 + yr**2), np.arctan(yr/(xr+_EPS)) #_eps avoid zero-division
        
    # OPtional target coordinates for vector-shifts (only used if poly_model is not None:)    
    if (poly_model is None) & ((xt is not None) & (yt is not None)):
        Rt, Tt = np.sqrt(xt**2 + yt**2), np.arctan(yt/(xt+_EPS)) # test radial distance and theta angle
        
    if (poly_model is not None):   
        # Apply model at ref. coordinates:
        (xt,yt),((Rt,Tt),(Rr,Tr)) = apply_poly_model_at_x(pmodel, xr, yr, poly_model, polar_coord = True)

    # plot vector field:
    if color != 0:
        plot_vector_field((xt, yt, Rt, Tt),(xr, yr, Rr, Tr),
                          color = color, axh = axh, title = title, axtype = axtype,
                          nTbins = nTbins, Tbins_start_angle = Tbins_start_angle, 
                          use_plt_quiver = use_plt_quiver)
    return (xt,yt),((Rt,Tt),(Rr,Tr))


def plot_vector_field(xyRTt, xyRTr, color = 'k', axh = None, title = None, axtype = 'polar',
                      nTbins = 32, Tbins_start_angle = 0, use_plt_quiver = True):
    """
    Makes a plot of a vector field (if color != 0). 
    For more info on input parameters, see generate_vector_field?
    
    Returns:
        None if color == 0 else axes handle
    """
    # Plot vectorfield:
    if color is not 0: 
        
        # unpack vector field data input:
        xr, yr, Rr, Tr = xyRTr
        xt, yt, Rt, Tt = xyRTt
        
        if (axh == None):
            fig, newfig = plt.figure(), True
        else:
            newfig = False
        
        # Calculate thetas for all grid points to derive plot color:
        thetas = positive_arctan(xr,yr, htype = 'rad')
        
        rect = [0.1, 0.1, 0.8, 0.8] # setting the axis limits in [left, bottom, width, height]
        if axtype == 'polar':
            if newfig == True: axh = fig.add_axes(rect, polar=True, frameon=False) # polar axis
            
            # Get polar coordinates and store in (override)  xr, yr and (xt, yt):
            xr, yr = cart2pol(xr,y=yr, htype = 'rad') # = Tr, Rr
            xt, yt = cart2pol(xt,y=yt, htype = 'rad') # = Tt, Rt
        else: 
            if newfig == True: axh = fig.add_axes(rect)  # cartesian axis
                  
        if color is None: # when user doesn't specify color, determine it from theta
            Tbins, hsv_hues = get_theta_bin_edges_and_cmap(nTbins = nTbins, Tbins_start_angle = Tbins_start_angle)
            for i in range(nTbins):
                c = np.abs(np.array(colorsys.hsv_to_rgb(hsv_hues[i], 0.84, 0.9)))
                xri = xr[(thetas>=Tbins[i])&(thetas<Tbins[i+1])]
                yri = yr[(thetas>=Tbins[i])&(thetas<Tbins[i+1])]
                xti = xt[(thetas>=Tbins[i])&(thetas<Tbins[i+1])]
                yti = yt[(thetas>=Tbins[i])&(thetas<Tbins[i+1])]
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
            if use_plt_quiver:
                axh.quiver(xr, yr, xt-xr, yt-yr, angles='uv', scale_units='y', headlength=1, scale = 0.4,color = color)
            else:
                for j in range(xr.shape[0]):
                    axh.plot(xr[j],yr[j],color = color, marker='.',) # plot a dot at the reference position
                    axh.plot(np.vstack((xr[j],xt[j])),np.vstack((yr[j],yt[j])),
                             color = color, marker=None, linestyle='-')        
        axh.set_xlabel("x")
        axh.set_ylabel("y")
        if title is not None:
            axh.set_title(title)
        return axh
    return None
        

def shiftvectors(xyt,xyr, average = True,):
    """
    Calculate xy-shift vectors.
    
    Args:
        :xyt: 
            | ndarray with target xy coordinates 
        :xyr:
            | ndarray with reference ab coordinates 
        :average:
            | True, optional
            | If True, take mean of difference vectors along axis = 0.
            
    Returns:
        :returns:
            | ndarray of (mean) shift vector(s). 
    """    
    v =  xyt - xyr
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

def plotcircle(center = np.array([0.,0.]),\
               radii = np.arange(0,60,10), \
               angles = np.arange(0,350,10),\
               color = 'k',linestyle = '--', out = None):
    """
    Plot one or more concentric circles.
    
    Args:
        :center: 
            | np.array([0.,0.]) or ndarray with center coordinates, optional
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
    for ri in radii:
        x = ri*np.cos(angles*np.pi/180)
        y = ri*np.sin(angles*np.pi/180)
        xs = np.hstack((xs,x))
        ys = np.hstack((ys,y))
        if out != 'x,y':
            plt.plot(x,y,color = color, linestyle = linestyle)
    if out == 'x,y':
        return xs,ys
    

#------------------------------------------------------------------------------
# Bivariate Poly_Model Class:
#------------------------------------------------------------------------------

class BiPolyModel:
    """
    Bivariate Polynomial Model Class
    """
    def __init__(self, xyt = None, xyr = None, pmodel = None, model_type = 'M6', get_stats = True, polar_coord = False):
        """
        Initialize / get the poly_model.
        See get_poly_model? for info on input.
        """
        self.model_type = None
        self.model = None
        self.p = None
        self.data = None
        self.stats = None
        self.polar_coord = polar_coord
        if ((xyt is None) & (xyr is None)) & (pmodel is None):
             self.initialized = False
        else:
            if (xyt is not None) & (xyr is not None):
                (poly_model, pmodel, M,
                dxy_pred, dxy_res, dxy_res_std, 
                dRToverR_pred, dRToverR_res, dRToverR_res_std) = get_poly_model(xyt, xyr, model_type = model_type, 
                                                                                 get_stats = get_stats, 
                                                                                 polar_coord = polar_coord, 
                                                                                 dict_out=False)
                self.initialized = True
                self.model_type = model_type
                self.model = poly_model
                self.p = pmodel
                self.M = M
                self.data = {'xyt':xyt, 'xyr':xyr}
                self.stats = {'dxy pred': dxy_pred, 
                              'dxy res': dxy_res, 
                              'dxy res std': dxy_res_std,
                              'dR/R,dT pred': dRToverR_pred, 
                              'dR/R,dT res': dRToverR_res, 
                              'dR/R,dT res std': dRToverR_res_std}
            elif (pmodel is not None):
                self.initialized = True
                self.model_type = model_type
                self.model = _poly_model_constructor(pmodel)
                self.p = pmodel
                self.data = None
                self.stats = None
                
    def apply(self, xr = None, yr = None, pmodel = None, polar_coord = False):
        """
        Apply the poly_model at coordinates in xr, yr. 
        See apply_poly_model_at_x? for more info on input arguments.
        
        Returns:
            | (xt,yt),((Rt,Tt),(Rr,Tr))
        """
        if pmodel is not None:
            poly_model = _poly_model_constructor(pmodel)
        else:
            pmodel = self.p
            poly_model = self.model
        if (xr is None) & (yr is None):
            xr, yr = self.data['xyr'][...,0], self.data['xyr'][...,1]
        if polar_coord is None:
            polar_coord = self.polar_coord
        return apply_poly_model_at_x(pmodel, xr, yr, poly_model = poly_model, polar_coord = polar_coord)

    def generate_vector_field(self, poly_model = None, pmodel = None, xr = None, yr = None, xt = None, yt = None,
                          circle_field = False, make_grid = True, limit_grid_radius = 0,
                          xy_ranges = np.array([[-100,100,10],[-100,100,10]]),
                          x_sampling = None, y_sampling = None, 
                          RT_ranges = np.array([[0,100,10],[0,360,5]]),
                          R_sampling = None, T_sampling = None, 
                          color = None, axh = None, title = None, axtype = 'polar', use_plt_quiver = True,
                          nTbins = 32, Tbins_start_angle = 0):
        """
        Generate a vector field. 
        For more info see generate_vector_field?
        
        Returns:
            | (xt,yt),((Rt,Tt),(Rr,Tr))
        """
        return generate_vector_field(poly_model = self.model, pmodel = self.p, 
                                     xr = xr, yr = yr, xt = xt, yt = yt,
                                     circle_field = circle_field, make_grid = make_grid, limit_grid_radius = limit_grid_radius,
                                     xy_ranges = xy_ranges,
                                     x_sampling = x_sampling, y_sampling = y_sampling, 
                                     RT_ranges = RT_ranges,
                                     R_sampling = R_sampling, T_sampling = T_sampling, 
                                     color = color, axh = axh, title = title, axtype = axtype, 
                                     use_plt_quiver = use_plt_quiver,
                                     nTbins = nTbins, Tbins_start_angle = Tbins_start_angle)        

    
if __name__ == '__main__':
    
    import luxpy as lx
    import pandas as pd
    
    #--------------------------------------------------------------------------
    # EXAMPLE 1: as shift model
    #--------------------------------------------------------------------------
    
    # Generate_test_data:
    F4 = lx._CIE_ILLUMINANTS['F4'].copy() 
    M = lx._MUNSELL.copy()
    rflM = M['R']
    rflM = lx.cie_interp(rflM,F4[0],kind='rfl')
    xyz31, xyzw31 = lx.spd_to_xyz(F4, cieobs = '1931_2', relative = True, rfl = rflM, out = 2)
    xyz06, xyzw06 = lx.spd_to_xyz(F4, cieobs = '2006_2', relative = True, rfl = rflM, out = 2)
    ab31 = lx.xyz_to_lab(xyz31, xyzw = xyzw31)[:,0,1:]
    ab06 = lx.xyz_to_lab(xyz06, xyzw = xyzw06)[:,0,1:]
    
    # Get model that characterizes shift between 1931 and 2006 2° CMFs 
    # based on relative sample shifts under the two sets:
    pm = BiPolyModel(ab31, ab06, model_type = 'M6') 
    
    # Create a grid of new data points 
    # and plot shifts in hue angle related colors (by seting Color = None):
    pm.generate_vector_field(circle_field = True,
                             xy_ranges = np.array([[-100,100,10],[-100,100,10]]),
                             RT_ranges = np.array([[0,100,10],[0,360,5]]),
                             make_grid = True, limit_grid_radius = 0,
                             color = None, axh = None, axtype = 'cart', use_plt_quiver = True,
                             nTbins = 32, Tbins_start_angle = 0, title = 'Test BiPolyModel')
    
    # or, generate some rect. xy-grid:
    xr_, yr_ = generate_rect_grid(xy_ranges = np.array([[-100,100,10],[-100,100,10]]), 
                                    out = 'x_grid, y_grid', limit_grid_radius = 0)
    # Apply shift to grid:
    xt_, yt_ = apply_poly_model_at_x(pm.p,xr_,yr_,pm.model,polar_coord=True)[0]
    plt.figure()
    plt.plot(xr_,yr_,'ro')
    plt.plot(xt_,yt_,'b.');

    # Apply shift to data used to obtain model parameters:
    xt,yt = apply_poly_model_at_x(pm.p,pm.data['xyr'][:,0],pm.data['xyr'][:,1],pm.model,polar_coord=True)[0]
    plt.figure()
    plt.plot(pm.data['xyr'][:,0],pm.data['xyr'][:,1],'ro');
    plt.plot(xt,yt,'b.');
    plt.plot(ab31[:,0],ab31[:,1],'g.'); 
    
    
    #--------------------------------------------------------------------------
    # EXAMPLE 2: as transformation
    #--------------------------------------------------------------------------

    ab_test = pd.read_csv('ab_test.dat',header=None,sep='\t').values
    ab_ref = pd.read_csv('ab_ref.dat',header=None,sep='\t').values   
    dLMS = pd.read_csv('dLMS.dat',header=None,sep='\t').values  
 
    # Generate forward and reverse models:
    # pm2 = LMS -> ab:
    pm2 = BiPolyModel(ab_ref, dLMS, model_type = 'M6') 
    # pm2i = ab -> LMS:
    pm2i = BiPolyModel(dLMS, ab_ref, model_type = 'M6') 
    
    # appply pm2 to dLMS data to go to ab-space, then apply pm2i to return to dLMS:
    at2,bt2 = apply_poly_model_at_x(pm2.p,dLMS[:,0],dLMS[:,1],polar_coord=True)[0]
    at2i,bt2i = apply_poly_model_at_x(pm2i.p,at2,bt2,pm2i.model,polar_coord=True)[0]
    plt.figure()
    plt.plot(dLMS[:,0],dLMS[:,1],'ro')
    plt.plot(at2,bt2 ,'b.');
    plt.plot(at2i,bt2i,'g+')
    
    # Apply generated models in inverse order to go from ab -> LMS -> ab
    # Generate large grid in ab-space:
    ar_2, br_2 = generate_rect_grid(xy_ranges = np.array([[-30,30,5],[-30,30,5]]), x_sampling = None, y_sampling = None, 
                                    out = 'x_grid,y_grid', limit_grid_radius = 0)
    # appply pm2i to ab-space to go to dLMS space, then apply pm2 to return to ab-space:
    at_2i,bt_2i = apply_poly_model_at_x(pm2i.p,ar_2, br_2,pm2i.model,polar_coord=True)[0]
    at_2,bt_2 = apply_poly_model_at_x(pm2.p,at_2i,bt_2i,pm2.model,polar_coord=True)[0]
    plt.figure()
    plt.plot(ar_2, br_2,'ro')
    plt.plot(at_2i,bt_2i,'b.')
    plt.plot(at_2, bt_2 ,'g+');
    
 