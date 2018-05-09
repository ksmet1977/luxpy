# -*- coding: utf-8 -*-
########################################################################
# <LUXPY: a Python package for lighting and color science.>
# Copyright (C) <2017>  <Kevin A.G. Smet> (ksmet1977 at gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#########################################################################
"""
Module with functions related to color rendering Vector Field model
===================================================================

 :_VF_CRI_DEFAULT: default cri_type parameters for VF model

 :_VF_CSPACE: default dict with color space parameters.

 :_VF_MAXR: maximum C to use in calculations and plotting of vector fields

 :_VF_DELTAR:  grid spacing, pixel size

 :_VF_MODEL_TYPE: type of polynomial model for base color shifts

 :_DETERMINE_HUE_ANGLES: Bool, determines whether to calculate hue_angles 
                         for 5 or 6 'informative' model parameters

 :_VF_PCOLORSHIFT: Default dict with hue_angle parameters for VF model

 :_VF_SIG:  0.3,  Determines smoothness of the transition between 
            hue-bin-boundaries (no hard cutoff at boundary).
 
 :get_poly_model(): Setup base color shift model (delta_a, delta_b), 
                    determine model parameters and accuracy.

 :apply_poly_model_at_x(): Applies base color shift model 
                           at cartesian coordinates axr, bxr.

 :generate_vector_field(): Generates a field of vectors 
                           using the base color shift model.

 :VF_colorshift_model(): Applies full vector field model calculations 
                         to spectral data.

 :generate_grid(): Generate a grid of color coordinates.

 :calculate_shiftvectors(): Calculate color shift vectors.

 :plot_shift_data(): Plots vector or circle fields.

 :plotcircle(): Plot one or more concentric circles.

 :initialize_VF_hue_angles(): Initialize the hue angles that will be used to 
                              'summarize' the VF model fitting parameters.


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""


from luxpy import np, plt, math, _CIE_ILLUMINANTS, _MUNSELL,_EPS
from ..utils.helpers import spd_to_cri
from ..utils.init_cri_defaults_database import _CRI_DEFAULTS
from ..utils.graphics import plot_hue_bins

#from munsell import *

__all__ = ['_VF_CRI_DEFAULT','_VF_CSPACE','_VF_CSPACE_EXAMPLE','_VF_CIEOBS','_VF_MAXR','_VF_DELTAR','_VF_MODEL_TYPE','_VF_SIG','_VF_PCOLORSHIFT']
__all__ += ['get_poly_model','apply_poly_model_at_x','generate_vector_field','VF_colorshift_model','initialize_VF_hue_angles']
__all__ += ['generate_grid','calculate_shiftvectors','plot_shift_data','plotcircle']

# Default color space for Vector Field model:
_VF_CRI_DEFAULT = 'iesrf'
_VF_CSPACE = _CRI_DEFAULTS[_VF_CRI_DEFAULT]['cspace'].copy()
_VF_CSPACE_EXAMPLE = {'type' : 'jab_cam02ucs','xyzw':None, 'mcat':'cat02', 'Yw':100.0, 'conditions' :{'La':100.0,'surround':'avg','D':1.0,'Yb':20.0,'Dtype':None},'yellowbluepurplecorrect' : None}
_VF_CIEOBS = _CRI_DEFAULTS[_VF_CRI_DEFAULT]['cieobs']['xyz']

_VF_MAXR = 40 #maximum C to use in calculations and plotting of vector fields
_VF_DELTAR = 5 # grid spacing, pixel size
_VF_MODEL_TYPE = 'M6' # default polynomial model (degree 6)
_DETERMINE_HUE_ANGLES = True  
_VF_SIG = 0.3 #  Determines smoothness of the transition between hue-bin-boundaries (no hard cutoff at boundary).
_VF_PCOLORSHIFT = None



#------------------------------------------------------------------------------
# Define function to get poly_model:
def get_poly_model(jabt, jabr, modeltype = _VF_MODEL_TYPE):
    """
    Setup base color shift model (delta_a, delta_b), 
    determine model parameters and accuracy.
    
    | Calculates a base color shift (delta) from the ref. chromaticity ar, br.
    
    Args:
        :jabt: 
            | ndarray with jab color coordinates under the test SPD.
        :jabr: 
            | ndarray with jab color coordinates under the reference SPD.
        :modeltype:
            | _VF_MODEL_TYPE or 'M6' or 'M5', optional
            | Specifies degree 5 or degree 6 polynomial model in ab-coordinates.
              (see notes below)
            
    Returns:
        :returns: 
            | (poly_model, 
            |       pmodel, 
            |       dab_model, 
            |        dab_res, 
            |        dCHoverC_res, 
            |        dab_std, 
            |        dCHoverC_std)
            |
            | :poly_model: function handle to model
            | :pmodel: ndarray with model parameters
            | :dab_model: ndarray with ab model predictions from ar, br.
            | :dab_res: ndarray with residuals between 'da,db' of samples and 
            |            'da,db' predicted by the model.
            | :dCHoverC_res: ndarray with residuals between 'dCoverC,dH' 
            |                 of samples and 'dCoverC,dH' predicted by the model.
            |     Note: dCoverC = (Ct - Cr)/Cr and dH = ht - hr 
            |         (predicted from model, see notes below)
            | :dab_std: ndarray with std of :dab_res:
            | :dCHoverC_std: ndarray with std of :dCHoverC_res: 

    Notes: 
        1. Model types:
            | poly5_model = lambda a,b,p:         p[0]*a + p[1]*b + p[2]*(a**2) + p[3]*a*b + p[4]*(b**2)
            | poly6_model = lambda a,b,p:  p[0] + p[1]*a + p[2]*b + p[3]*(a**2) + p[4]*a*b + p[5]*(b**2)
        
        2. Calculation of dCoverC and dH:
            | dCoverC = (np.cos(hr)*da + np.sin(hr)*db)/Cr
            | dHoverC = (np.cos(hr)*db - np.sin(hr)*da)/Cr    
    """
    at = jabt[...,1]
    bt = jabt[...,2]
    ar = jabr[...,1]
    br = jabr[...,2]
    
    # A. Calculate da, db:
    da = at - ar
    db = bt - br
    
    # B.1 Calculate model matrix:
    # 5-parameter model:
    M5 = np.array([[np.sum(ar*ar), np.sum(ar*br), np.sum(ar*ar**2),np.sum(ar*ar*br),np.sum(ar*br**2)],
            [np.sum(br*ar), np.sum(br*br), np.sum(br*ar**2),np.sum(br*ar*br),np.sum(br*br**2)],
            [np.sum((ar**2)*ar), np.sum((ar**2)*br), np.sum((ar**2)*ar**2),np.sum((ar**2)*ar*br),np.sum((ar**2)*br**2)],
            [np.sum(ar*br*ar), np.sum(ar*br*br), np.sum(ar*br*ar**2),np.sum(ar*br*ar*br),np.sum(ar*br*br**2)],
            [np.sum((br**2)*ar), np.sum((br**2)*br), np.sum((br**2)*ar**2),np.sum((br**2)*ar*br),np.sum((br**2)*br**2)]])
    #6-parameters model
    M6 = np.array([[ar.size,np.sum(1.0*ar), np.sum(1.0*br), np.sum(1.0*ar**2),np.sum(1.0*ar*br),np.sum(1.0*br**2)],
            [np.sum(ar*1.0),np.sum(ar*ar), np.sum(ar*br), np.sum(ar*ar**2),np.sum(ar*ar*br),np.sum(ar*br**2)],
            [np.sum(br*1.0),np.sum(br*ar), np.sum(br*br), np.sum(br*ar**2),np.sum(br*ar*br),np.sum(br*br**2)],
            [np.sum((ar**2)*1.0),np.sum((ar**2)*ar), np.sum((ar**2)*br), np.sum((ar**2)*ar**2),np.sum((ar**2)*ar*br),np.sum((ar**2)*br**2)],
            [np.sum(ar*br*1.0),np.sum(ar*br*ar), np.sum(ar*br*br), np.sum(ar*br*ar**2),np.sum(ar*br*ar*br),np.sum(ar*br*br**2)],
            [np.sum((br**2)*1.0),np.sum((br**2)*ar), np.sum((br**2)*br), np.sum((br**2)*ar**2),np.sum((br**2)*ar*br),np.sum((br**2)*br**2)]])
    
    # B.2 Define model function:
    poly5_model = lambda a,b,p: p[0]*a + p[1]*b + p[2]*(a**2) + p[3]*a*b + p[4]*(b**2)
    poly6_model = lambda a,b,p: p[0] + p[1]*a + p[2]*b + p[3]*(a**2) + p[4]*a*b + p[5]*(b**2)
    
    if modeltype == 'M5':
        M = M5
        poly_model = poly5_model
    else:
        M = M6
        poly_model = poly6_model

    M = np.linalg.inv(M)


    # C.1 Data a,b analysis output:
    if modeltype == 'M5':
        da_model_parameters = np.dot(M, np.array([np.sum(da*ar), np.sum(da*br), np.sum(da*ar**2),np.sum(da*ar*br),np.sum(da*br**2)]))
        db_model_parameters = np.dot(M, np.array([np.sum(db*ar), np.sum(db*br), np.sum(db*ar**2),np.sum(db*ar*br),np.sum(db*br**2)]))
    else:
        da_model_parameters = np.dot(M, np.array([np.sum(da*1.0),np.sum(da*ar), np.sum(da*br), np.sum(da*ar**2),np.sum(da*ar*br),np.sum(da*br**2)]))
        db_model_parameters = np.dot(M, np.array([np.sum(db*1.0),np.sum(db*ar), np.sum(db*br), np.sum(db*ar**2),np.sum(db*ar*br),np.sum(db*br**2)]))
    pmodel = np.vstack((da_model_parameters,db_model_parameters))

    # D.1 Calculate model da, db:
    da_model = poly_model(ar,br,pmodel[0])
    db_model = poly_model(ar,br,pmodel[1])
    dab_model = np.hstack((da_model,db_model))

    # D.2 Calculate residuals for da & db:
    da_res = da - da_model
    db_res = db - db_model
    dab_res = np.hstack((da_res,db_res))
    dab_std = np.vstack((np.std(da_res,axis=0),np.std(db_res,axis=0)))

    # E Calculate href, Cref:
    href = np.arctan2(br,ar)
    Cref = (ar**2 + br**2)**0.5
    
    # F Calculate dC/C, dH/C for data and model and calculate residuals:
    dCoverC = (np.cos(href)*da + np.sin(href)*db)/Cref
    dHoverC = (np.cos(href)*db - np.sin(href)*da)/Cref
    dCoverC_model = (np.cos(href)*da_model + np.sin(href)*db_model)/Cref
    dHoverC_model = (np.cos(href)*db_model - np.sin(href)*da_model)/Cref
    dCoverC_res = dCoverC - dCoverC_model
    dHoverC_res = dHoverC - dHoverC_model
    dCHoverC_std = np.vstack((np.std(dCoverC_res,axis = 0),np.std(dHoverC_res,axis = 0)))
    
    dCHoverC_res = np.hstack((href,dCoverC_res,dHoverC_res))

    return poly_model, pmodel, dab_model, dab_res, dCHoverC_res, dab_std, dCHoverC_std


def apply_poly_model_at_x(poly_model, pmodel,axr,bxr):
    """
    Applies base color shift model at cartesian coordinates axr, bxr.
    
    Args:
        :poly_model: 
            | function handle to model
        :pmodel:
            | ndarray with model parameters.
        :axr: 
            | ndarray with a-coordinates under the reference conditions
        :bxr:
            | ndarray with b-coordinates under the reference conditions
        
    Returns:
        :returns:
            | (axt,bxt,Cxt,hxt,
            |  axr,bxr,Cxr,hxr)
            | 
            | ndarrays with ab-coordinates, chroma and hue 
              predicted by the model (xt), under the reference (xr).
    """

    # Calculate hxr and Cxr:
    Cxr = np.sqrt(axr**2 + bxr**2)
    hxr = np.arctan(bxr/(axr+_EPS)) #_eps avoid zero-division
   
    # B Set 2nd order color multipliers (shiftd parameters for a and b: pa & pb):
    pa = pmodel[0].copy()
    pb = pmodel[1].copy()
    isM6 = pa.shape[0] == 6
    pa[0 + isM6*1] = 1 + pa[0 + isM6*1]
    pb[1 + isM6*1] = 1 + pb[1 + isM6*1]
    
    # C Apply model to reference hues using 2nd order multipliers:
    axt = poly_model(axr,bxr,pa)
    bxt = poly_model(axr,bxr,pb)
    Cxt = np.sqrt(axt**2+bxt**2) #test chroma
    hxt = np.arctan(bxt/(axt+_EPS))

    return axt,bxt,Cxt,hxt,axr,bxr,Cxr,hxr


def apply_poly_model_at_hue_x(poly_model, pmodel, dCHoverC_res, \
                              hx = None, Cxr = 40, sig = _VF_SIG):
    """
    Applies base color shift model at (hue,chroma) coordinates
    
    Args:
        :poly_model: 
            | function handle to model
        :pmodel:
            | ndarray with model parameters.
        :dCHoverC_res:
            | ndarray with residuals between 'dCoverC,dH' of samples 
            | and 'dCoverC,dH' predicted by the model.
            | Note: dCoverC = (Ct - Cr)/Cr and dH = ht - hr 
            |      (predicted from model, see notes luxpy.cri.get_poly_model())
        :hx:
            | None or ndarray, optional
            | None defaults to np.arange(np.pi/10.0,2*np.pi,2*np.pi/10.0)
        :Cxr:
            | 40, optional
        :sig: 
            | _VF_SIG or float, optional
            | Determines smooth transition between hue-bin-boundaries (no hard 
              cutoff at hue bin boundary).
        
    Returns:
        :returns: 
            | ndarrays with dCoverC_x, dCoverC_x_sig, dH_x, dH_x_sig
            | Note '_sig' denotes the uncertainty: 
            |     e.g.  dH_x_sig is the uncertainty of dH at input (hue/chroma).
    """
     
    if hx is None:
        dh = 2*np.pi/10.0;
        hx = np.arange(dh/2,2*np.pi,dh) #hue angles at which to apply model, i.e. calculate 'average' measures
        
    # A calculate reference coordinates:
    axr = Cxr*np.cos(hx)
    bxr = Cxr*np.sin(hx)
    
    # B apply model at reference coordinates to obtain test coordinates:
    axt,bxt,Cxt,hxt,axr,bxr,Cxr,hxr = apply_poly_model_at_x(poly_model, pmodel,axr,bxr)
    
    # C Calculate dC/C, dH for test and ref at fixed hues:
    dCoverC_x = (Cxt-Cxr)/(np.hstack((Cxr+Cxt)).max())
    dH_x = (180/np.pi)*(hxt-hxr)
#    dCoverC_x = np.round(dCoverC_x,decimals = 2)
#    dH_x = np.round(dH_x,decimals = 0)

    # D calculate 'average' noise measures using sig-value:
    href = dCHoverC_res[:,0:1]
    dCoverC_res = dCHoverC_res[:,1:2]
    dHoverC_res = dCHoverC_res[:,2:3]
    dHsigi = np.exp((np.dstack((np.abs(hx-href),np.abs((hx-href-2*np.pi)),np.abs(hx-href-2*np.pi))).min(axis=2)**2)/(-2)/sig)
    dH_x_sig = (180/np.pi)*(np.sqrt((dHsigi*(dHoverC_res**2)).sum(axis=0,keepdims=True)/dHsigi.sum(axis=0,keepdims=True)))
    dH_x_sig_avg = np.sqrt(np.sum(dH_x_sig**2,axis=1)/hx.shape[0])
    dCoverC_x_sig = (np.sqrt((dHsigi*(dCoverC_res**2)).sum(axis=0,keepdims=True)/dHsigi.sum(axis=0,keepdims=True)))
    dCoverC_x_sig_avg = np.sqrt(np.sum(dCoverC_x_sig**2,axis=1)/hx.shape[0])

    return dCoverC_x, dCoverC_x_sig, dH_x, dH_x_sig


def generate_vector_field(poly_model, pmodel, \
                          axr = np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR), \
                          bxr = np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR), \
                          make_grid = True, limit_grid_radius = 0,color = 'k'):
    """
    Generates a field of vectors using the base color shift model.
    
    | Has the option to plot vector field.
    
    Args:
        :poly_model: 
            | function handle to model
        :pmodel:
            | ndarray with model parameters.
        :axr: 
            | np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR), optional
            | Ndarray specifying the a-coordinates at which to apply the model.
        :bxr:
            | np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR), optional
            | Ndarray specifying the b-coordinates at which to apply the model.
        :make_grid:
            | True, optional
            | True: generate a 2d-grid from :axr:, :bxr:.
        :limit_grid_radius:
            | 0, optional
            |   A value of zeros keeps grid as specified  by axr,bxr.
            |   A value > 0 only keeps (a,b) coordinates within :limit_grid_radius:
        :color:
            | 'k', optional
            | For plotting the vector field.
            | If :color: == 0, no plot will be generated.
    
    Returns:
        :returns: 
            | If :color: == 0: ndarray of axt,bxt,axr,bxr
            | Else: handle to axes used for plotting.
    """
    
    
    # Generate grid from axr, bxr:
    if make_grid == True:
        axr, bxr = generate_grid(ax = axr, bx = bxr, out = 'ax,bx', limit_grid_radius = limit_grid_radius)

    # Apply model at ref. coordinates:
    axt,bxt,Cxt,hxt,axr,bxr,Cxr,hxr = apply_poly_model_at_x(poly_model, pmodel,axr,bxr)
    
    # Plot vectorfield:
    if color is not 0: 
        #plt.plot(axr, bxr,'ro',markersize=2)
        plt.quiver(axr, bxr, axt-axr, bxt-bxr, headlength=1,color = color)
        plt.xlabel("a'")
        plt.ylabel("b'")
        return plt.gca()#plt.show(plot1)
    else:
        return axt,bxt,axr,bxr


def VF_colorshift_model(S, cri_type = _VF_CRI_DEFAULT, model_type = _VF_MODEL_TYPE, \
                        cspace = _VF_CSPACE, sampleset = None, pool = False, \
                        pcolorshift = {'href': np.arange(np.pi/10,2*np.pi,2*np.pi/10),'Cref' : _VF_MAXR, 'sig' : _VF_SIG}, \
                        vfcolor = 'k',verbosity = 0):
    """
    Applies full vector field model calculations to spectral data.
    
    Args:
        :S: 
            | nump.ndarray with spectral data.
        :cri_type:
            | _VF_CRI_DEFAULT or str or dict, optional
            | Specifies type of color fidelity model to use. 
            | Controls choice of ref. ill., sample set, averaging, scaling, etc.
            | See luxpy.cri.spd_to_cri for more info.
        :modeltype:
            | _VF_MODEL_TYPE or 'M6' or 'M5', optional
            | Specifies degree 5 or degree 6 polynomial model in ab-coordinates.
        :cspace:
            | _VF_CSPACE or dict, optional
            | Specifies color space. See _VF_CSPACE_EXAMPLE for example structure.
        :sampleset:
            | None or str or ndarray, optional
            | Sampleset to be used when calculating vector field model.
        :pool: 
            | False, optional
            | If :S: contains multiple spectra, True pools all jab data before 
              modeling the vector field, while False models a different field 
              for each spectrum.
        :pcolorshift: 
            | default dict (see below) or user defined dict, optional
            | Dict containing the specification input 
              for apply_poly_model_at_hue_x().
            | Default dict = {'href': np.arange(np.pi/10,2*np.pi,2*np.pi/10),
            |                 'Cref' : _VF_MAXR, 
            |                 'sig' : _VF_SIG, 
            |                 'labels' : '#'} 
            | The polynomial models of degree 5 and 6 can be fully specified or 
              summarized by the model parameters themselved OR by calculating the
              dCoverC and dH at resp. 5 and 6 hues.
        :vfcolor:
            | 'k', optional
            | For plotting the vector fields.
        :verbosity: 
            | 0, optional
            | Report warnings or not.
            
    Returns:
        :returns: 
            | list[dict] (each list element refers to a different test SPD)
            | with the following keys:
            |   - 'Source': dict with ndarrays of the S, cct and duv of source spd.
            |   - 'metrics': dict with ndarrays for:
            |         * Rf (color fidelity: base + metameric shift)
            |         * Rt (metameric uncertainty index) 
            |         * Rfi (specific color fidelity indices)
            |         * Rti (specific metameric uncertainty indices)
            |         * cri_type (str with cri_type)
            |   - 'Jab': dict with with ndarrays for Jabt, Jabr, DEi
            |   - 'dC/C_dH_x_sig' : 
            |           np.vstack((dCoverC_x,dCoverC_x_sig,dH_x,dH_x_sig)).T
            |           See get_poly_model() for more info.
            |   - 'fielddata': dict with dicts containing data on the calculated 
            |      vector-field and circle-fields: 
            |        * 'vectorfield' : {'axt': vfaxt, 'bxt' : vfbxt, 
            |                           'axr' : vfaxr, 'bxr' : vfbxr},
            |        * 'circlefield' : {'axt': cfaxt, 'bxt' : cfbxt, 
            |                           'axr' : cfaxr, 'bxr' : cfbxr}},
            |   - 'modeldata' : dict with model info:
            |                {'pmodel': pmodel, 
            |                'pcolorshift' : pcolorshift, 
            |                  'dab_model' : dab_model, 
            |                  'dab_res' : dab_res,
            |                  'dab_std' : dab_std,
            |                  'modeltype' : modeltype, 
            |                  'fmodel' : poly_model,
            |                  'Jabtm' : Jabtm, 
            |                  'Jabrm' : Jabrm, 
            |                  'DEim' : DEim},
            |   - 'vshifts' :dict with various vector shifts:
            |        * 'Jabshiftvector_r_to_t' : ndarray with difference vectors
            |                                    between jabt and jabr.
            |        * 'vshift_ab_s' : vshift_ab_s: ab-shift vectors of samples 
            |        * 'vshift_ab_s_vf' : vshift_ab_s_vf: ab-shift vectors of 
            |                             VF model predictions of samples.
            |        * 'vshift_ab_vf' : vshift_ab_vf: ab-shift vectors of VF 
            |                            model predictions of vector field grid.
    """
    
    if type(cri_type) == str:
        cri_type_str = cri_type
    else:
        cri_type_str = None
    
    # Calculate Rf, Rfi and Jabr, Jabt:
    Rf, Rfi, Jabt, Jabr,cct,duv,cri_type  = spd_to_cri(S, cri_type= cri_type,out='Rf,Rfi,jabt,jabr,cct,duv,cri_type', sampleset=sampleset)
    
    # In case of multiple source SPDs, pool:
    if (len(Jabr.shape) == 3) & (Jabr.shape[1]>1) & (pool == True):
        Nsamples = Jabr.shape[0]
        Jabr = np.transpose(Jabr,(1,0,2)) # set lamps on first dimension
        Jabt = np.transpose(Jabt,(1,0,2))
        Jabr = Jabr.reshape(Jabr.shape[0]*Jabr.shape[1],3) # put all lamp data one after the other
        Jabt = Jabt.reshape(Jabt.shape[0]*Jabt.shape[1],3)
        Jabt = Jabt[:,None,:] # add dim = 1
        Jabr = Jabr[:,None,:]
    

    out = [{} for _ in range(Jabr.shape[1])] #initialize empty list of dicts
    if pool == False:
        N = Jabr.shape[1]
    else:
        N = 1
    for i in range(N):
        
        Jabr_i = Jabr[:,i,:].copy()
        Jabr_i = Jabr_i[:,None,:]
        Jabt_i = Jabt[:,i,:].copy()
        Jabt_i = Jabt_i[:,None,:]

        DEi = np.sqrt((Jabr_i[...,0] - Jabt_i[...,0])**2 + (Jabr_i[...,1] - Jabt_i[...,1])**2 + (Jabr_i[...,2] - Jabt_i[...,2])**2)

        # Determine polynomial model:
        poly_model, pmodel, dab_model, dab_res, dCHoverC_res, dab_std, dCHoverC_std = get_poly_model(Jabt_i, Jabr_i, modeltype = _VF_MODEL_TYPE)
        
        # Apply model at fixed hues:
        href = pcolorshift['href']
        Cref = pcolorshift['Cref']
        sig = pcolorshift['sig']
        dCoverC_x, dCoverC_x_sig, dH_x, dH_x_sig = apply_poly_model_at_hue_x(poly_model, pmodel, dCHoverC_res, hx = href, Cxr = Cref, sig = sig)
        
        # Calculate deshifted a,b values on original samples:
        Jt = Jabt_i[...,0].copy()
        at = Jabt_i[...,1].copy()
        bt = Jabt_i[...,2].copy()
        Jr = Jabr_i[...,0].copy()
        ar = Jabr_i[...,1].copy()
        br = Jabr_i[...,2].copy()
        ar = ar + dab_model[:,0:1] # deshift reference to model prediction
        br = br + dab_model[:,1:2] # deshift reference to model prediction
        
        Jabtm = np.hstack((Jt,at,bt))
        Jabrm = np.hstack((Jr,ar,br))
        
        # calculate color differences between test and deshifted ref:
#        DEim = np.sqrt((Jr - Jt)**2 + (at - ar)**2 + (bt - br)**2)
        DEim = np.sqrt(0*(Jr - Jt)**2 + (at - ar)**2 + (bt - br)**2) # J is not used

        # Apply scaling function to convert DEim to Rti:
        scale_factor = cri_type['scale']['cfactor']
        scale_fcn = cri_type['scale']['fcn']
        avg = cri_type['avg']  
        Rfi_deshifted = scale_fcn(DEim,scale_factor)
        Rf_deshifted = scale_fcn(avg(DEim,axis = 0),scale_factor)
        
        rms = lambda x: np.sqrt(np.sum(x**2,axis=0)/x.shape[0])
        Rf_deshifted_rms = scale_fcn(rms(DEim),scale_factor)
    
        # Generate vector field:
        vfaxt,vfbxt,vfaxr,vfbxr = generate_vector_field(poly_model, pmodel,axr = np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR), bxr = np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR), limit_grid_radius = _VF_MAXR,color = 0)
        vfaxt,vfbxt,vfaxr,vfbxr = generate_vector_field(poly_model, pmodel,axr = np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR), bxr = np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR), limit_grid_radius = _VF_MAXR,color = 0)

        # Calculate ab-shift vectors of samples and VF model predictions:
        vshift_ab_s = calculate_shiftvectors(Jabt_i, Jabr_i, average = False, vtype = 'ab')[:,0,0:3]
        vshift_ab_s_vf = calculate_shiftvectors(Jabtm,Jabrm, average = False, vtype = 'ab')

        # Calculate ab-shift vectors using vector field model:
        Jabt_vf = np.hstack((np.zeros((vfaxt.shape[0],1)), vfaxt, vfbxt))   
        Jabr_vf = np.hstack((np.zeros((vfaxr.shape[0],1)), vfaxr, vfbxr))   
        vshift_ab_vf = calculate_shiftvectors(Jabt_vf,Jabr_vf, average = False, vtype = 'ab')

        # Generate circle field:
        x,y = plotcircle(radii = np.arange(0,_VF_MAXR+_VF_DELTAR,10), angles = np.arange(0,359,1), out = 'x,y')
        cfaxt,cfbxt,cfaxr,cfbxr = generate_vector_field(poly_model, pmodel,make_grid = False,axr = x[:,None], bxr = y[:,None], limit_grid_radius = _VF_MAXR,color = 0)

        out[i] = {'Source' : {'S' : S, 'cct' : cct[i] , 'duv': duv[i]},
               'metrics' : {'Rf':Rf[:,i], 'Rt': Rf_deshifted, 'Rt_rms' : Rf_deshifted_rms, 'Rfi':Rfi[:,i], 'Rti': Rfi_deshifted, 'cri_type' : cri_type_str},
               'Jab' : {'Jabt' : Jabt_i, 'Jabr' : Jabr_i, 'DEi' : DEi},
               'dC/C_dH_x_sig' : np.vstack((dCoverC_x,dCoverC_x_sig,dH_x,dH_x_sig)).T,
               'fielddata': {'vectorfield' : {'axt': vfaxt, 'bxt' : vfbxt, 'axr' : vfaxr, 'bxr' : vfbxr},
                             'circlefield' : {'axt': cfaxt, 'bxt' : cfbxt, 'axr' : cfaxr, 'bxr' : cfbxr}},
               'modeldata' : {'pmodel': pmodel, 'pcolorshift' : pcolorshift, 
                              'dab_model' : dab_model, 'dab_res' : dab_res,'dab_std' : dab_std,
                              'model_type' : model_type, 'fmodel' : poly_model,
                              'Jabtm' : Jabtm, 'Jabrm' : Jabrm, 'DEim' : DEim},
               'vshifts' : {'Jabshiftvector_r_to_t' : np.hstack((Jt-Jr,at-ar,bt-br)),
                            'vshift_ab_s' : vshift_ab_s,
                            'vshift_ab_s_vf' : vshift_ab_s_vf,
                            'vshift_ab_vf' : vshift_ab_vf}}
     
    return out



def generate_grid(jab_ranges = None, out = 'grid', \
                  ax = np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR),\
                  bx = np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR), \
                  jx = None, limit_grid_radius = 0):
    """
    Generate a grid of color coordinates.
    
    Args:
        :out:
            | 'grid' or 'vectors', optional
            |   - 'grid': outputs a single 2d numpy.nd-vector with the grid coordinates
            |   - 'vector': outputs each dimension seperately.
        :jab_ranges:
            | None or ndarray, optional
            | Specifies the pixelization of color space.
              (ndarray.shape = (3,3), with  first axis: J,a,b, and second 
              axis: min, max, delta)
        :ax:
            | default ndarray or user defined ndarray, optional
            | default = np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR) 
        :bx:
            | default ndarray or user defined ndarray, optional
            | default = np.arange(-_VF_MAXR,_VF_MAXR+_VF_DELTAR,_VF_DELTAR) 
        :jx:
            | None, optional
            | Note that not-None :jab_ranges: override :ax:, :bx: and :jx input.
        :limit_grid_radius:
            | 0, optional
            | A value of zeros keeps grid as specified  by axr,bxr.
            | A value > 0 only keeps (a,b) coordinates within :limit_grid_radius:
            
    Returns:
        :returns: 
            | single ndarray with ax,bx [,jx] 
            |  or
            | seperate ndarrays for each dimension specified.
    """
    # generate grid from jab_ranges array input, otherwise use ax, bx, jx input:
    if jab_ranges is not None:
        if jab_ranges.shape[0] == 3:
            jx = np.arange(jab_ranges[0][0],jab_ranges[0][1],jab_ranges[0][2])
            ax = np.arange(jab_ranges[1][0],jab_ranges[1][1],jab_ranges[1][2])
            bx = np.arange(jab_ranges[2][0],jab_ranges[2][1],jab_ranges[2][2])
        else:
            jx = None
            ax = np.arange(jab_ranges[0][0],jab_ranges[0][1],jab_ranges[0][2])
            bx = np.arange(jab_ranges[1][0],jab_ranges[1][1],jab_ranges[1][2])
   
    # Generate grid from (jx), ax, bx:
    Ax,Bx = np.meshgrid(ax,bx)
    grid = np.dstack((Ax,Bx))
    grid = np.reshape(grid,(np.array(grid.shape[:-1]).prod(),grid.ndim-1))
    if jx is not None:
        for i,v in enumerate(jx):
            gridi = np.hstack((np.ones((grid.shape[0],1))*v,grid))
            if i == 0:
                gridwithJ = gridi
            else:
                gridwithJ = np.vstack((gridwithJ,gridi))
        grid = gridwithJ
    
    if jx is None:
        ax = grid[:,0:1]
        bx = grid[:,1:2]
    else:
        jx = grid[:,0:1]
        ax = grid[:,1:2]
        bx = grid[:,2:3] 
    
    if limit_grid_radius > 0:# limit radius of grid:
        Cr = (ax**2+bx**2)**0.5
        ax = ax[Cr<=limit_grid_radius,None]
        bx = bx[Cr<=limit_grid_radius,None]
        if jx is not None:
            jx = jx[Cr<=limit_grid_radius,None]
    
    # create output:
    if out == 'grid':
        if jx is None:
            return np.hstack((ax,bx))
        else:
            return np.hstack((jx,ax,bx))
    else:
        if jx is None:
            return ax, bx
        else:
            return jx, ax, bx


def calculate_shiftvectors(jabt,jabr, average = True, vtype = 'ab'):
    """
    Calculate color shift vectors.
    
    Args:
        :jabt: 
            | ndarray with jab coordinates under the test SPD
        :jabr:
            | ndarray with jab coordinates under the reference SPD
        :average:
            | True, optional
            | If True, take mean of difference vectors along axis = 0.
        :vtype:
            | 'ab' or 'jab', optional
            | Reduce output ndarray to only a,b coordinates of shift vector(s).
            
    Returns:
        :returns:
            | ndarray of (mean) shift vector(s).
            
    """    
    v =  jabt - jabr
    if average == True:
        v = v.mean(axis=0)
    if vtype == 'ab':
        v = v[...,1:3]
    return v   
    

#------------------------------------------------------------------------------
def plot_shift_data(data, fieldtype = 'vectorfield', scalef = _VF_MAXR, color = 'k', \
                    axtype = 'polar', ax = None, \
                    hbins = 10,  start_hue = 0.0, bin_labels = '#', plot_center_lines = True,  \
                    plot_axis_labels = False, plot_edge_lines = False, plot_bin_colors = True, \
                    force_CVG_layout = True):
     
    """
    Plots vector or circle fields generated by VFcolorshiftmodel() 
    or PXcolorshiftmodel().
     
    Args:
        :data: 
            | dict generated by VFcolorshiftmodel() or PXcolorshiftmodel()
            | Must contain 'fielddata'- key, which is a dict with possible keys:
            |     - key: 'vectorfield': ndarray with vector field data
            |     - key: 'circlefield': ndarray with circle field data
        :color: 
            | 'k', optional
            | Color for plotting the vector-fields.
        :axtype:
            | 'polar' or 'cart', optional
            | Make polar or Cartesian plot.
        :ax: 
            | None or 'new' or 'same', optional
            |   - None or 'new' creates new plot
            |   - 'same': continue plot on same axes.
            |   - axes handle: plot on specified axes.
        :hbins:
            | 16 or ndarray with sorted hue bin centers (°), optional
        :start_hue:
            | _VF_MAXR, optional
            | Scale factor for graphic.
        :plot_axis_labels:
            | False, optional
            | Turns axis ticks on/off (True/False).
        :bin_labels:
            | None or list[str] or '#', optional
            | Plots labels at the bin center hues.
            |   - None: don't plot.
            |   - list[str]: list with str for each bin. 
            |                (len(:bin_labels:) = :nhbins:)
            |   - '#': plots number.
        :plot_edge_lines:
            | True or False, optional
            | Plot grey bin edge lines with '--'.
        :plot_center_lines:
            | False or True, optional
            | Plot colored lines at 'center' of hue bin.
        :plot_bin_colors:
            | True, optional
            | Colorize hue-bins.
        :force_CVG_layout: 
            | False or True, optional
            | True: Force plot of basis of CVG.
    
    Returns:
        :returns:
            | figCVG, hax, cmap
        
            |   :figCVG: handle to CVG figure
            |   :hax: handle to CVG axes
            |   :cmap: list with rgb colors for hue bins 
            |          (for use in other plotting fcns)
   
    """
       
    # Plot basis of CVG:
    figCVG, hax, cmap = plot_hue_bins(hbins = hbins, axtype = axtype, ax = ax, plot_center_lines = plot_center_lines, plot_edge_lines = plot_edge_lines, plot_bin_colors = plot_bin_colors, scalef = scalef, force_CVG_layout = force_CVG_layout, bin_labels = bin_labels)
    
    # plot vector field:
    if data is not None:
        if fieldtype is not None:
            vf = data['fielddata'][fieldtype]
            if axtype == 'polar':
                if fieldtype == 'vectorfield':
                    vfrtheta = math.positive_arctan(vf['axr'], vf['bxr'],htype = 'rad')
                    vfrr = np.sqrt(vf['axr']**2 + vf['bxr']**2)
                    hax.quiver(vfrtheta, vfrr, vf['axt'] - vf['axr'], vf['bxt'] - vf['bxr'],  headlength=3,color = color,angles='uv', scale_units='y', scale = 2,linewidth = 0.5)
                else:
                    vfttheta = math.positive_arctan(vf['axt'], vf['bxt'],htype = 'rad')
                    vfrtheta = math.positive_arctan(vf['axr'], vf['bxr'],htype = 'rad')
                    vftr = np.sqrt(vf['axt']**2 + vf['bxt']**2)
                    dh = (math.angle_v1v2(np.hstack((vf['axt'],vf['bxt'])),np.hstack((vf['axr'],vf['bxr'])),htype='deg')[:,None]) #hue shift
                    dh = dh/np.nanmax(dh)
                    plt.set_cmap('jet')
                    hax.scatter(vfttheta, vftr, s = 100*dh, c = dh, linestyle = 'None', marker = 'o',norm = None)
                hax.set_ylim([0, 1.1*scalef])     
            else:
                if fieldtype == 'vectorfield':
                    hax.quiver(vf['axr'], vf['bxr'], vf['axt'] - vf['axr'], vf['bxt'] - vf['bxr'],  headlength=1,color = color,angles='uv', scale_units='xy', scale = 1,linewidth = 0.5)
                else:
                    hax.plot(vf['axr'], vf['bxr'], color = color, marker = '.',linestyle = 'None')
    
    return figCVG, hax, cmap

#------------------------------------------------------------------------------
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




##############################################################################

def initialize_VF_hue_angles(hx = None, Cxr = _VF_MAXR, \
                             cri_type = _VF_CRI_DEFAULT, \
                             modeltype = _VF_MODEL_TYPE,\
                             determine_hue_angles = _DETERMINE_HUE_ANGLES):
    """
    Initialize the hue angles that will be used to 'summarize' 
    the VF model fitting parameters.
    
    Args:       
        :hx: 
            | None or ndarray, optional
            | None defaults to Munsell H5 hues.
        :Cxr: 
            | _VF_MAXR, optional
        :cri_type: 
            | _VF_CRI_DEFAULT or str or dict, optional,
            | Cri_type parameters for cri and VF model.
        :modeltype:
            | _VF_MODEL_TYPE or 'M5' or 'M6', optional
            | Determines the type of polynomial model.
        :determine_hue_angles:
            | _DETERMINE_HUE_ANGLES or True or False, optional
            | True: determines the 10 primary / secondary Munsell hues ('5..').
            | Note that for 'M6', an additional 
            
    Returns:
        :pcolorshift: 
            | {'href': href,
            |           'Cref' : _VF_MAXR, 
            |           'sig' : _VF_SIG, 
            |           'labels' : list[str]}
    """
    
    ###########################################
    # Get Munsell H5 hues:
    ###########################################

    rflM = _MUNSELL['R']
    hn = _MUNSELL['H'] # all Munsell hues
    rH5 = np.where([_MUNSELL['H'][:,0][x][0]=='5' for x in range(_MUNSELL['H'][:,0].shape[0])])[0] #all Munsell H5 hues
    hns5 = np.unique(_MUNSELL['H'][rH5]) 
    #------------------------------------------------------------------------------
    # Determine Munsell hue angles in cam02ucs:
    pool = False  
    IllC = _CIE_ILLUMINANTS['C'] # for determining Munsell hue angles in cam02ucs
    outM = VF_colorshift_model(IllC, cri_type = cri_type, sampleset = rflM, vfcolor = 'g',pool = pool)
    #------------------------------------------------------------------------------
    if (determine_hue_angles == True) | (hx is None):
        # find samples at major Munsell hue angles:
        all_h5_Munsell_cam02ucs = np.ones(hns5.shape)
        Jabt_IllC = outM[0]['Jab']['Jabt']
        for i,v in enumerate(hns5):
            hm = np.where(hn == v)[0]
            all_h5_Munsell_cam02ucs[i] = math.positive_arctan([Jabt_IllC[hm,0,1].mean()],[Jabt_IllC[hm,0,2].mean()],htype = 'rad')[0]
        hx = all_h5_Munsell_cam02ucs
        

    #------------------------------------------------------------------------------
    # Setp color shift parameters:
    pcolorshift = {'href': hx,'Cref' : Cxr, 'sig' : _VF_SIG, 'labels' : hns5}
    return pcolorshift

_VF_PCOLORSHIFT = initialize_VF_hue_angles(determine_hue_angles = _DETERMINE_HUE_ANGLES, modeltype = _VF_MODEL_TYPE)
