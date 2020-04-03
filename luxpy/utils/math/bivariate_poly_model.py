# -*- coding: utf-8 -*-
"""
Bivariate polynomial model
==========================

 :get_poly_model(): Get bivariate polynomial model parameters.
     
 :apply_poly_model_at_x(): Applies bivariate polynomial model at cartesian reference coordinates.
 
 :generate_ab_grid(): Generate a 2D grid of coordinates.

 :generate_vector_field(): Generates a field of vectors bivariate polynomial model.
     
 :plot_vector_field(): Makes a plot of a vector field.
     
 :BiPolyModel(): Bivariate Polynomial Model Class


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

import numpy as np
import matplotlib.pyplot as plt
import colorsys

from . import (cart2pol, positive_arctan)

_EPS = np.finfo('float').eps # used in model to avoid division by zero ! 

__all__ = ['get_poly_model','apply_poly_model_at_x',
           'generate_ab_grid','generate_vector_field',
           'plot_vector_field','BiPolyModel']

#------------------------------------------------------------------------------
# Bivariate Poly_Model functions:
#------------------------------------------------------------------------------  
def _poly_model_constructor(model_type):
    if not isinstance(model_type,str): # if not a string model_type contains parameters
        model_type = 'M{:1.0f}'.format(model_type.shape[0])
    if model_type == 'M5':
        poly_model = lambda a,b,p: p[0]*a + p[1]*b + p[2]*(a**2) + p[3]*a*b + p[4]*(b**2)
    elif model_type == 'M6':
        poly_model = lambda a,b,p: p[0] + p[1]*a + p[2]*b + p[3]*(a**2) + p[4]*a*b + p[5]*(b**2)
    return poly_model


def get_poly_model(abt, abr, model_type = 'M6', get_stats = True, dict_out = True, polar_coord = False):
    """
    Get bivariate polynomial model parameters.
    
    Args:
        :abt: 
            | ndarray with test coordinates.
        :abr: 
            | ndarray with target coordinates.
        :model_type:
            | 'M6' or 'M5', optional
            | Specifies degree 5 or degree 6 polynomial model in ab-coordinates.
              (see notes below)
        :get_stats:
            | True, optional
            | Calculate model statistics: dab_pred, dab_res, dab_res_std, dCHoverC_pred, dCHoverC_res,  dCHoverC_res_std
            | If False: fill with None's 
            | See :returns: for more info on statistics.
        :dict_out:
            | True, optional
            | Get function output as dict instead of tuple.
        :polar_coord:
            | False, optional
            | If True: also calculate dC/C, dH (only when get_stat == True !! )
              
    Returns:   
        :returns: 
            | Dict or tuple with:
            |  - 'poly_model' : function handle to model
            |  - 'p' : ndarray with model parameters
            |  - 'dab pred' : ndarray with dab model predictions from ar, br.
            |  - 'dab res' : ndarray with residuals between 'da,db' of samples and 
            |                'da,db' predicted by the model.
            |  - 'dab res std' : ndarray with std of 'dab res'
            |  - 'dC/C,dH pred' : ndarray with predictions for dC/C = (Ct - Cr)/Cr and dH = ht - hr
            |  - 'dC/C,dH res' : ndarray with residuals between 'dC/C,dH' 
            |                      of samples and 'dC/C,dH' predicted by the model.
            |  - 'dC/C,dH res std' : ndarray with std of 'dC/C,dH res: 

    Notes: 
        1. Model types:
            | poly5_model = lambda a,b,p:         p[0]*a + p[1]*b + p[2]*(a**2) + p[3]*a*b + p[4]*(b**2)
            | poly6_model = lambda a,b,p:  p[0] + p[1]*a + p[2]*b + p[3]*(a**2) + p[4]*a*b + p[5]*(b**2)
        
        2. Calculation of dCoverC and dH:
            | dCoverC = (np.cos(hr)*da + np.sin(hr)*db)/Cr
            | dHoverC = (np.cos(hr)*db - np.sin(hr)*da)/Cr   
    """
    at, bt = abt[...,0], abt[...,1]
    ar, br = abr[...,0], abr[...,1]
    
    # A. Calculate da, db:
    da, db = at - ar, bt - br
    
    # B.1 Calculate model matrix:
    # 5-parameter model:
    M = np.array([[np.sum(ar*ar), np.sum(ar*br), np.sum(ar*ar**2),np.sum(ar*ar*br),np.sum(ar*br**2)],
            [np.sum(br*ar), np.sum(br*br), np.sum(br*ar**2),np.sum(br*ar*br),np.sum(br*br**2)],
            [np.sum((ar**2)*ar), np.sum((ar**2)*br), np.sum((ar**2)*ar**2),np.sum((ar**2)*ar*br),np.sum((ar**2)*br**2)],
            [np.sum(ar*br*ar), np.sum(ar*br*br), np.sum(ar*br*ar**2),np.sum(ar*br*ar*br),np.sum(ar*br*br**2)],
            [np.sum((br**2)*ar), np.sum((br**2)*br), np.sum((br**2)*ar**2),np.sum((br**2)*ar*br),np.sum((br**2)*br**2)]])
    
    #6-parameters model:
    if model_type == 'M6': 
        M = np.vstack((np.array([[np.sum(1.0*ar),np.sum(1.0*br), np.sum(1.0*ar**2),np.sum(1.0*ar*br),np.sum(1.0*br**2)]]), M)) # add row 0 of M6
        M = np.hstack((np.array([[ar.size,np.sum(ar*1.0),np.sum(br*1.0),np.sum((ar**2)*1.0),np.sum(ar*br*1.0),np.sum((br**2)*1.0)]]).T,M)) # add col 0 of M6
        
    # Get inverse matrix:
    M = np.linalg.inv(M)
    
    # B.2 Define model function using constructor function:
    poly_model = _poly_model_constructor(model_type)

    # C.1 Data a,b analysis output:
    if model_type == 'M5':
        da_model_parameters = np.dot(M, np.array([np.sum(da*ar), np.sum(da*br), np.sum(da*ar**2),np.sum(da*ar*br),np.sum(da*br**2)]))
        db_model_parameters = np.dot(M, np.array([np.sum(db*ar), np.sum(db*br), np.sum(db*ar**2),np.sum(db*ar*br),np.sum(db*br**2)]))
    elif model_type == 'M6':
        da_model_parameters = np.dot(M, np.array([np.sum(da*1.0),np.sum(da*ar), np.sum(da*br), np.sum(da*ar**2),np.sum(da*ar*br),np.sum(da*br**2)]))
        db_model_parameters = np.dot(M, np.array([np.sum(db*1.0),np.sum(db*ar), np.sum(db*br), np.sum(db*ar**2),np.sum(db*ar*br),np.sum(db*br**2)]))
    pmodel = np.vstack((da_model_parameters,db_model_parameters))

    if get_stats == True:
        # D.1 Calculate predicted da, db (with resp. to ref.):
        da_pred, db_pred = poly_model(ar,br,pmodel[0]), poly_model(ar,br,pmodel[1])
        dab_pred = np.hstack((da_pred,db_pred))
    
        # D.2 Calculate residuals for da & db:
        da_res, db_res = da - da_pred, db - db_pred
        dab_res = np.hstack((da_res, db_res))
        dab_res_std = np.vstack((np.std(da_res,axis=0),np.std(db_res,axis=0)))

        # E.1 Calculate href, Cref:
        href, Cref = np.arctan2(br,ar), (ar**2 + br**2)**0.5

        # E.2 Calculate res and std of dC/C, dH/C:
        dCoverC, dHoverC  = (np.cos(href)*da + np.sin(href)*db)/Cref, (np.cos(href)*db - np.sin(href)*da)/Cref
        dCoverC_pred, dHoverC_pred = (np.cos(href)*da_pred + np.sin(href)*db_pred)/Cref, (np.cos(href)*db_pred - np.sin(href)*da_pred)/Cref
        dCoverC_res, dHoverC_res = dCoverC - dCoverC_pred, dHoverC - dHoverC_pred
        dCHoverC_pred = np.vstack((dCoverC_pred, dHoverC_pred))
        dCHoverC_res_std = np.vstack((np.std(dCoverC_res,axis = 0),np.std(dHoverC_res,axis = 0)))
        dCHoverC_res = np.hstack((href,dCoverC_res,dHoverC_res))
    else:
        dab_pred, dab_res, dab_res_std, dCHoverC_pred, dCHoverC_res, dCHoverC_res_std = None, None, None, None, None, None 

    if dict_out == True:
        return {'poly_model' : poly_model, 
                'p' : pmodel, 
                'dab pred': dab_pred, 
                'dab res': dab_res, 
                'dab res std': dab_res_std,
                'dC/C,dH pred': dCHoverC_pred, 
                'dC/C,dH res': dCHoverC_res, 
                'dC/C,dH res std': dCHoverC_res_std}
    else:
        return (poly_model, pmodel,
                dab_pred, dab_res, dab_res_std, 
                dCHoverC_pred, dCHoverC_res, dCHoverC_res_std)


def apply_poly_model_at_x(pmodel, axr, bxr, poly_model = None,
                          polar_coord = False):
    """
    Applies bivariate polynomial model at cartesian reference coordinates.
    
    Args:
        :pmodel:
            | ndarray with model parameters.
        :axr: 
            | ndarray with reference a-coordinates
        :bxr:
            | ndarray with reference b-coordinates
        :poly_model: 
            | function handle to model (if None: construct new lambda fcn.)
        :polar_coord:
            | False, optional
            | If True: also calculate C and h for axt and bxt.
        
    Returns:
        :returns:
            | (axt,bxt),((Cxt,hxt),(Cxr,hxr))
            | 
            | (axt,bxt) ndarrays with predicted ab-coordinates, 
            | (Cxt,hxt) radial distance and angle predicted by the model (xt)
            | (Cxr,hxr) radial distance and angle for reference  (xr)
    """
    # Create lambda function for model:
    if poly_model is None:
        poly_model = _poly_model_constructor(pmodel)
   
    # A Set 2nd order color multipliers (shiftd parameters for a and b: pa & pb):
    pa, pb = pmodel[0].copy(), pmodel[1].copy()
    
    isM6 = pa.shape[0] == 6
    pa[0 + isM6*1] = 1 + pa[0 + isM6*1]
    pb[1 + isM6*1] = 1 + pb[1 + isM6*1]
    
    # B Apply model to reference hues using 2nd order multipliers:
    axt,bxt = poly_model(axr,bxr,pa), poly_model(axr,bxr,pb) 
    
    if polar_coord == True:
        # C Calculate hxr and Cxr:
        Cxr, hxr = np.sqrt(axr**2 + bxr**2), np.arctan(bxr/(axr+_EPS)) #_eps avoid zero-division
        Cxt, hxt = np.sqrt(axt**2+bxt**2), np.arctan(bxt/(axt+_EPS)) #test chroma and hue
    else:
        Cxr, hxr, Cxt, hxt = None, None, None, None
    return (axt,bxt),((Cxt,hxt),(Cxr,hxr))


#------------------------------------------------------------------------------
# Grid and vector-shift field generators:
#------------------------------------------------------------------------------

def generate_ab_grid(ab_ranges = np.array([[-100,100,10],[-100,100,10]]),
                     ax = None, bx = None, limit_grid_radius = 0, out = 'grid'):
    """
    Generate a 2D grid of coordinates.
    
    Args:
        :ab_ranges:
            | None or ndarray, optional
            | Specifies the pixelization of the 2D space.
            |  (ndarray.shape = (2,3), with  first axis: a,b, and second 
            |  axis: min, max, delta)
        :ax:
            | None, optional
            | User defined ndarray. If not None: overrides ab_ranges
        :bx:
            | None, optional
            | User defined ndarray. If not None: overrides ab_ranges
        :limit_grid_radius:
            | 0, optional
            | A value of zeros keeps grid as specified  by axr,bxr.
            | A value > 0 only keeps (a,b) coordinates within :limit_grid_radius:
        :out:
            | 'grid' or 'vectors', optional
            |   - 'grid': outputs a single 2d numpy.nd-vector with the grid coordinates
            |   - 'vector': outputs each dimension seperately.
            
    Returns:
        :returns: 
            | single ndarray with ax,bx [,jx] 
            |  or
            | seperate ndarrays for each dimension specified.
    """
    # generate grid from ab_ranges array input, otherwise use ax, bx, input:
    if (ax is None) | (bx is None):
        ax = np.arange(ab_ranges[0][0],ab_ranges[0][1],ab_ranges[0][2])
        bx = np.arange(ab_ranges[1][0],ab_ranges[1][1],ab_ranges[1][2])
   
    # Generate grid from ax, bx:
    Ax,Bx = np.meshgrid(ax,bx)
    grid = np.dstack((Ax,Bx))
    grid = np.reshape(grid,(np.array(grid.shape[:-1]).prod(),grid.ndim-1))
    ax = grid[:,0:1]
    bx = grid[:,1:2]

    if limit_grid_radius > 0:# limit radius of grid:
        Cr = (ax**2+bx**2)**0.5
        ax = ax[Cr<=limit_grid_radius,None]
        bx = bx[Cr<=limit_grid_radius,None]
    
    # create output:
    if out == 'grid':
        return np.hstack((ax,bx))
    else:
        return ax, bx

def generate_vector_field(poly_model, pmodel, ab_ranges = np.array([[-100,100,10],[-100,100,10]]),
                          axr = None, bxr = None, make_grid = True, limit_grid_radius = 0,
                          circle_field = False, circle_field_radius = 100, 
                          circle_field_radial_step = 10, circle_field_angle_step = 5,
                          color = 'k', axh = None, title = None, axtype = 'polar', use_plt_quiver = True,
                          nhbins = 32, hbins_start_angle = 0):
    """
    Generates a field of vectors bivariate polynomial model.
    
    | Has the option to plot vector field.
    
    Args:
        :poly_model: 
            | function handle to model
        :pmodel:
            | ndarray with model parameters.
        :ab_ranges:
            | None or ndarray, optional
            | Specifies the pixelization of the 2D space.
            |  (ndarray.shape = (2,3), with  first axis: a,b, and second 
            |  axis: min, max, delta)
        :ax:
            | None, optional
            | User defined ndarray. If not None: overrides ab_ranges
        :bx:
            | None, optional
            | User defined ndarray. If not None: overrides ab_ranges
        :make_grid:
            | True, optional
            | True: generate a 2d-grid from :axr:, :bxr:.
        :limit_grid_radius:
            | 0, optional
            |   A value of zeros keeps grid as specified  by axr,bxr.
            |   A value > 0 only keeps (a,b) coordinates within :limit_grid_radius:
        :circle_field:
            | False, optional
            | Generate a polar grid based vector field instead of cartesian grid based one.
        :circle_field_radius:
            | 100, optional
            | Max. radius of circle field.
        :circle_field_radius_step:
            | 10, optional
            | Radial resolution of circle field.
        :circle_field_angle_step:
            | 5, optional
            | Angle resolution of circle field.
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
        :nhbins:
            | Number of angle bins to divide the vector field space with each plotted in
            | different hue (hue-discritizaion of color plot)
        :hbins_start_angle:
            | Start angle of first hue bin in degrees.
    
    Returns:
        :returns: 
            | (axt,bxt),((Cxt,hxt),(Cxr,hxr))
    """
    if circle_field == True:
        # Generate circle field:
        x, y = plotcircle(radii = np.arange(0,circle_field_radius,circle_field_radial_step), 
                          angles = np.arange(0,360 - circle_field_angle_step,circle_field_angle_step), out = 'x,y')
        axr, bxr = x[:,None], y[:,None] # replace input axr, bxr with those generated for a circulare grid

        # Generate grid from axr, bxr:
    if (make_grid == True) & (circle_field != True):
        axr, bxr = generate_ab_grid(ab_ranges = ab_ranges, ax = axr, bx = bxr, 
                                    out = 'ax,bx', limit_grid_radius = limit_grid_radius)

    # Apply model at ref. coordinates:
    (axt,bxt),((Cxt,hxt),(Cxr,hxr)) = apply_poly_model_at_x(pmodel, axr, bxr, poly_model, polar_coord = True)
    
    # plot vector field:
    if color != 0:
        plot_vector_field((axr, bxr, Cxr, hxr), (axt, bxt, Cxt, hxt),
                          color = color, axh = axh, title = title, axtype = axtype,
                          nhbins = nhbins, hbins_start_angle = hbins_start_angle, use_plt_quiver = use_plt_quiver)
    return (axt,bxt),((Cxt,hxt),(Cxr,hxr))


def plot_vector_field(abChxr, abChxt, color = 'k', axh = None, title = None, axtype = 'polar',
                      nhbins = 32, hbins_start_angle = 0, use_plt_quiver = True):
    """
    Makes a plot of a vector field (if color != 0). 
    For more info on input parameters, see generate_vector_field?
    
    Returns:
        None if color == 0 else axes handle
    """
    # Plot vectorfield:
    if color is not 0: 
        
        # unpack vector field data input:
        axr, bxr, Cxr, hxr = abChxr
        axt, bxt, Cxt, hxt = abChxt
        
        if (axh == None):
            fig, newfig = plt.figure(), True
        else:
            newfig = False
        
        # Calculate hues for all grid points to derive plot color:
        hues = positive_arctan(axr,bxr, htype = 'rad')
        
        rect = [0.1, 0.1, 0.8, 0.8] # setting the axis limits in [left, bottom, width, height]
        if axtype == 'polar':
            if newfig == True: axh = fig.add_axes(rect, polar=True, frameon=False) # polar axis
            
            # Get polar coordinates and store in (override)  axr, bxr
            axr, bxr = cart2pol(axr,y=bxr, htype = 'rad') # = abr_theta, abr_r
            axt, bxt = cart2pol(axt,y=bxt, htype = 'rad') # = abt_theta, abt_r
        else: 
            if newfig == True: axh = fig.add_axes(rect)  # cartesian axis
                  
        if color is None: # when user doesn't specify color, determine it from hue
            hbins, hsv_hues = get_hue_bin_edges_and_cmap(nhbins = nhbins, hbins_start_angle = hbins_start_angle)
            for i in range(nhbins):
                c = np.abs(np.array(colorsys.hsv_to_rgb(hsv_hues[i], 0.84, 0.9)))
                axri = axr[(hues>=hbins[i])&(hues<hbins[i+1])]
                bxri = bxr[(hues>=hbins[i])&(hues<hbins[i+1])]
                axti = axt[(hues>=hbins[i])&(hues<hbins[i+1])]
                bxti = bxt[(hues>=hbins[i])&(hues<hbins[i+1])]
                if use_plt_quiver:
                    axh.quiver(axri, bxri, axti-axri, bxti-bxri, edgecolor = c,
                               facecolor = c, headlength=3, angles='uv', 
                               scale_units='y', scale = 0.4,width=0.005,linewidth = 0.01)
                else:
                    for j in range(axri.shape[0]):
                        axh.plot(axri[j],bxri[j],color = c, marker='.',linestyle='none') # plot a dot at the reference position
                        axh.plot(np.vstack((axri[j],axti[j])),np.vstack((bxri[j],bxti[j])),
                                 color = c, marker=None, linestyle='-')
        else: 
            if use_plt_quiver:
                axh.quiver(axr, bxr, axt-axr, bxt-bxr, angles='uv', scale_units='y', headlength=1, scale = 0.4,color = color)
            else:
                for j in range(axr.shape[0]):
                    axh.plot(axr[j],bxr[j],color = color, marker='.',) # plot a dot at the reference position
                    axh.plot(np.vstack((axr[j],axt[j])),np.vstack((bxr[j],bxt[j])),
                             color = color, marker=None, linestyle='-')        
        axh.set_xlabel("a")
        axh.set_ylabel("b")
        if title is not None:
            axh.set_title(title)
        return axh
    return None
        

def shiftvectors(abt,abr, average = True,):
    """
    Calculate ab-shift vectors.
    
    Args:
        :abt: 
            | ndarray with test ab coordinates 
        :abr:
            | ndarray with reference ab coordinates 
        :average:
            | True, optional
            | If True, take mean of difference vectors along axis = 0.
            
    Returns:
        :returns:
            | ndarray of (mean) shift vector(s). 
    """    
    v =  abt - abr
    if average == True:
        v = v.mean(axis=0)
    return v   


#------------------------------------------------------------------------------
# Plotting related functions:
#------------------------------------------------------------------------------


def get_hue_bin_edges_and_cmap(nhbins = 16, hbins_start_angle = 0):
    """
    Get hue bin edges (in rad) and a list of rgb values (color map) for plotting each hue bin.
    """
    dhbins = 360/(nhbins) # hue bin width
    hbincenters = np.sort(np.arange(hbins_start_angle + dhbins/2, 360, dhbins))
    dL = np.arange(hbins_start_angle,360-dhbins+1,dhbins)
    
    # Setup color for plotting hue bins:
    hsv_hues = hbincenters - 30
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
    def __init__(self, abt = None, abr = None, pmodel = None, model_type = 'M6', get_stats = True, polar_coord = False):
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
        if ((abt is None) & (abr is None)) & (pmodel is None):
             self.initialized = False
        else:
            if (abt is not None) & (abr is not None):
                (poly_model, pmodel, 
                dab_pred, dab_res, dab_res_std, 
                dCHoverC_pred, dCHoverC_res, dCHoverC_res_std) = get_poly_model(abt, abr, model_type = model_type, 
                                                                                 get_stats = get_stats, 
                                                                                 polar_coord = polar_coord, 
                                                                                 dict_out=False)
                self.initialized = True
                self.model_type = model_type
                self.model = poly_model
                self.p = pmodel
                self.data = {'abt':abt, 'abr':abr}
                self.stats = {'dab pred': dab_pred, 
                              'dab res': dab_res, 
                              'dab res std': dab_res_std,
                              'dC/C,dH pred': dCHoverC_pred, 
                              'dC/C,dH res': dCHoverC_res, 
                              'dC/C,dH res std': dCHoverC_res_std}
            elif (pmodel is not None):
                self.initialized = True
                self.model_type = model_type
                self.model = _poly_model_constructor(pmodel)
                self.p = pmodel
                self.data = None
                self.stats = None
                
    def apply(self, axr = None, bxr = None, pmodel = None, polar_coord = False):
        """
        Apply the poly_model at coordinates in axr, bxr. 
        See apply_poly_model_at_x? for more info on input arguments.
        
        Returns:
            | (axt,bxt),((Cxt,hxt),(Cxr,hxr))
        """
        if pmodel is not None:
            poly_model = _poly_model_constructor(pmodel)
        else:
            poly_model = self.poly_model
        if (axr is None) & (bxr is None):
            axr, bxr = self.data['abr'][...,0], self.data['abr'][...,1]
        if polar_coord is None:
            polar_coord = self.polar_coord
        return apply_poly_model_at_x(pmodel, axr, bxr, poly_model = poly_model, polar_coord = polar_coord)

    def generate_vector_field(self, ab_ranges = np.array([[-100,100,10],[-100,100,10]]),
                          axr = None, bxr = None, make_grid = True, limit_grid_radius = 0,
                          circle_field = False, circle_field_radius = 100, 
                          circle_field_radial_step = 10, circle_field_angle_step = 5,
                          color = None, axh = None, title = None, axtype = 'polar', use_plt_quiver = True,
                          nhbins = 32, hbins_start_angle = 0):
        """
        Generate a vector field. 
        For more info see generate_vector_field?
        
        Returns:
            | (axt,bxt),((Cxt,hxt),(Cxr,hxr))
        """
        return generate_vector_field(self.model, self.p, ab_ranges = ab_ranges, axr = axr, bxr = bxr, 
                                      make_grid = make_grid, limit_grid_radius = limit_grid_radius,
                                      circle_field = circle_field, circle_field_radius = circle_field_radius, 
                                      circle_field_radial_step = circle_field_radial_step, circle_field_angle_step = circle_field_angle_step,
                                      color = color, axh = axh, title = title, axtype = axtype, use_plt_quiver = use_plt_quiver,
                                      nhbins = nhbins, hbins_start_angle = hbins_start_angle)        

    
if __name__ == '__main__':
    
    import luxpy as lx
    
    # Generate_test_data:
    F4 = lx._CIE_ILLUMINANTS['F4'].copy() 
    M = lx._MUNSELL.copy()
    rflM = M['R']
    rflM = lx.cie_interp(rflM,F4[0],kind='rfl')
    xyz31, xyzw31 = lx.spd_to_xyz(F4, cieobs = '1931_2', relative = True, rfl = rflM, out = 2)
    xyz06, xyzw06 = lx.spd_to_xyz(F4, cieobs = '2006_2', relative = True, rfl = rflM, out = 2)
    ab31 = lx.xyz_to_lab(xyz31, xyzw = xyzw31)[...,1:]
    ab06 = lx.xyz_to_lab(xyz06, xyzw = xyzw06)[...,1:]
    
    # Get model that characterizes shift between 1931 and 2006 2° CMFs 
    # based on relative sample shifts under the two sets:
    pm = BiPolyModel(ab31, ab06, model_type = 'M6') 
    
    # Create a grid of new data points 
    # and plot shifts in hue angle related colors (by seting Color = None):
    pm.generate_vector_field(ab_ranges = np.array([[-100,100,10],[-100,100,10]]),
                          make_grid = True, limit_grid_radius = 0,
                          circle_field = True, circle_field_radius = 100, 
                          circle_field_radial_step = 10, circle_field_angle_step = 5,
                          color = None, axh = None, axtype = 'cart', use_plt_quiver = True,
                          nhbins = 32, hbins_start_angle = 0, title = 'Test BiPolyModel')
    
    
            
                
                
                
             
         

