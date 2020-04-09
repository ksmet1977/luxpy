# -*- coding: utf-8 -*-
"""
Module for building and optimizing SPDs (2)
===========================================

This module differs from spdbuild.py in the spdoptimizer function,
that can use several different minimization algorithms, as well as a user defined
method. It is also written such that the user can easily write his own
primary constructor function. In contrast to spdbuild.py, it only supports the
'3mixer' algorithms for calculating the mixing contributions of the primaries.

Functions
---------
 :gaussian_prim_constructor: constructs a gaussian based primary set.
 
 :_setup_wlr: Setup the wavelength range for use in prim_constructor.
 
 :_extract_prim_optimization_parameters: Extact the primary parameters from the optimization vector x and the prim_constructor_parameter_defs dict.

 :_start_optimization_tri: Start optimization of _fitnessfcn for n primaries using the specified minimize_method. (see notes in docstring on specifications for the  user-defined minimization fcn) 
   
 :spd_optimizer2(): Generate a spectrum with specified white point and optimized
                   for certain objective functions from a set of component 
                   spectra or component spectrum model parameters.
                
Notes
-----
 1. See examples below (in '__main__') for use.                

References
----------
    1. `Ohno Y (2005). 
    Spectral design considerations for white LED color rendering. 
    Opt. Eng. 44, 111302. 
    <https://ws680.nist.gov/publication/get_pdf.cfm?pub_id=841839>`_

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from luxpy import (sp,np, plt, warnings, math, _WL3, _CIEOBS, _EPS, np2d, 
                   vec_to_dict, getwlr, SPD, spd_to_power,
                   spd_to_xyz, xyz_to_Yxy, colortf, xyz_to_cct)
from luxpy import cri 
import itertools

__all__ = ['spd_optimizer2','gaussian_prim_constructor','gaussian_prim_parameter_types',
           '_color3mixer','_setup_wlr','_extract_prim_optimization_parameters',
           '_start_optimization_tri']


#------------------------------------------------------------------------------
def _color3mixer(Yxyt,Yxy1,Yxy2,Yxy3):
    """
    Calculate fluxes required to obtain a target chromaticity 
    when (additively) mixing 3 light sources.
    
    Args:
        :Yxyt: 
            | ndarray with target Yxy chromaticities.
        :Yxy1: 
            | ndarray with Yxy chromaticities of light sources 1.
        :Yxy2:
            | ndarray with Yxy chromaticities of light sources 2.
        :Yxy3:
            | ndarray with Yxy chromaticities of light sources 3.
        
    Returns:
        :M: 
            | ndarray with fluxes.
        
    Note:
        Yxyt, Yxy1, ... can contain multiple rows, referring to single mixture.
    """
    Y1, x1, y1 = Yxy1[...,0], Yxy1[...,1], Yxy1[...,2]
    Y2, x2, y2 = Yxy2[...,0], Yxy2[...,1], Yxy2[...,2]
    Y3, x3, y3 = Yxy3[...,0], Yxy3[...,1], Yxy3[...,2]
    Yt, xt, yt = Yxyt[...,0], Yxyt[...,1], Yxyt[...,2]
    m1 = y1*((xt-x3)*y2-(yt-y3)*x2+x3*yt-xt*y3)/(yt*((x3-x2)*y1+(x2-x1)*y3+(x1-x3)*y2))
    m2 = -y2*((xt-x3)*y1-(yt-y3)*x1+x3*yt-xt*y3)/(yt*((x3-x2)*y1+(x2-x1)*y3+(x1-x3)*y2))
    m3 = y3*((x2-x1)*yt-(y2-y1)*xt+x1*y2-x2*y1)/(yt*((x2-x1)*y3-(y2-y1)*x3+x1*y2-x2*y1))
    M = Yt*np.vstack((m1/Y1,m2/Y2,m3/Y3))
    return M.T

           
#------------------------------------------------------------------------------
# New triangle-based faster spectral optimizer:
#------------------------------------------------------------------------------    
     
def _triangle_mixer(Yxy_target, Yxyi, triangle_strengths):
    """
    Calculates the fluxes of each of the primaries to realize the target 
    chromaticity Yxy_target given the triangle_strengths.
    """
    # Generate all possible 3-channel combinations (component triangles):
    N = Yxyi.shape[0]
    combos = np.array(list(itertools.combinations(range(N), 3))) 
   
    # calculate fluxes to obtain target Yxyt:
    M3 = _color3mixer(Yxy_target,Yxyi[combos[:,0],:],Yxyi[combos[:,1],:],Yxyi[combos[:,2],:])

    # Get rid of out-of-gamut solutions:
    is_out_of_gamut =  (((M3<0).sum(axis=1))>0)
    n_in_gamut = M3.shape[0] - is_out_of_gamut.sum()
    M3[is_out_of_gamut,:] = 0
    Nc = combos.shape[0]
        
    M3[is_out_of_gamut,:] = np.nan
    if Nc > 1:
        if n_in_gamut > 0:
            # Calulate fluxes of all components from M3 and x_final:            
            M_final = triangle_strengths*M3
            M = np.empty((N))
            for i in range(N):
                M[i] = np.nansum(M_final[np.where(combos == i)])
            M /= n_in_gamut
        else:
            M = np.zeros((1,N))
    else:
        M = M3
        
    return M

#------------------------------------------------------------------------------
def _setup_wlr(wlr):
    """
    Setup the wavelength range for use in prim_constructor.
    """
    if len(wlr) == 3:
        wlr = getwlr(wlr)
    if wlr.ndim == 1:
        wlr = wlr[None,:]
    return wlr

def _extract_prim_optimization_parameters(x, nprims, 
                                          prim_constructor_parameter_types, 
                                          prim_constructor_parameter_defs):
    """
    Extact the primary parameters from the optimization vector x and the prim_constructor_parameter_defs dict.
    """
    types = prim_constructor_parameter_types
    pars = {}
    ct = 0
    for pt in types:
        if pt not in prim_constructor_parameter_defs: # extract value from x (to be optimized as not in _defs dict!)
           pars[pt] = x[:,(ct*nprims):(ct*nprims) + nprims] 
           ct+=1
        else:
           pars[pt] = prim_constructor_parameter_defs[pt]
    return pars
         
            

#------------------------------------------------------------------------------
# Example code for a primiary constructor function:
gaussian_prim_parameter_types = ['peakwl', 'fwhm']

def gaussian_prim_constructor(x, nprims, wlr, 
                              prim_constructor_parameter_types, 
                              **prim_constructor_parameter_defs):
    """
    Construct a set of n gaussian primaries with wavelengths wlr using the input in x and in kwargs.
    
    Args:
        :x:
            | ndarray (M x n) with optimization parameters.
        :nprim:
            | number of primaries
        :wlr:
            | wavelength range for which to construct a spectrum
        :prim_constructor:
            | function that constructs the primaries from the optimization parameters
            | Should have the form: 
            |   prim_constructor(x, n, wl, prim_constructor_parameter_types, **prim_constructor_parameter_defs)
        :prim_constructor_parameter_types:
            | gaussian_prim_parameter_types ['peakwl', 'fwhm'], optional
            | List with strings of the parameters used by prim_constructor() to
            | calculate the primary spd. All parameters listed and that do not
            | have default values (one for each prim!!!) in prim_constructor_parameters_defs 
            | will be optimized.
        :prim_constructor_parameters_defs:
            | Dict with constructor parameters required by prim_constructor and/or 
            | default values for parameters that are not being optimized.
            | For example: {'fwhm':  30} will keep fwhm fixed and not optimize it.
            
    Returns:
        :spd:
            | ndarray with spectrum of nprim primaries (1st row = wavelengths)
    """
    # Extract the primary parameters from x and prim_constructor_parameter_defs:
    pars = _extract_prim_optimization_parameters(x, nprims, prim_constructor_parameter_types,
                                                 prim_constructor_parameter_defs)
    # setup wavelengths:
    wlr = _setup_wlr(wlr)
    
    # Collect parameters from pars dict:
    return np.vstack((wlr,np.exp(-((pars['peakwl']-wlr.T)/pars['fwhm'])**2).T))  

     
#------------------------------------------------------------------------------
def _spd_constructor_tri(x, Yxy_target, n, wlr = [360,830,1], cieobs=_CIEOBS,
                       prims = None, prim_constructor = gaussian_prim_constructor,
                       prim_constructor_parameter_types = gaussian_prim_parameter_types,
                       prim_constructor_parameter_defs = {}):
    """
    Construct a mixture spectrum composed of n primaries using the 3mixer algorithm.
    
    Args:
        :x:
            | optimization parameters, first n!/(n-3)!*3! are the strengths of
            | the triangles in the '3mixer' algorithm.
        :Yxy_target:
            | Target chromaticity in Y,x,y coordinates.
        :n:
            | number of primaries
        :wlr:
            | [360,830,1],optional
            | wavelength range for which to construct a spectrum
        :cieobs:
            | _CIEOBS, optional
            | CIE CMFs to calculate chromaticity.
        :prims:
            | None, optional
            | If not None: use these pre-defined primary spectra
            | else construct primaries using the prim_constructor function
        :prim_constructor:
            | function that constructs the primaries from the optimization parameters
            | Should have the form: 
            |   prim_constructor(x, n, wl, prim_constructor_parameter_types, **prim_constructor_parameter_defs)
        :prim_constructor_parameter_types:
            | gaussian_prim_parameter_types ['peakwl', 'fwhm'], optional
            | List with strings of the parameters used by prim_constructor() to
            | calculate the primary spd. All parameters listed and that do not
            | have default values (one for each prim!!!) in prim_constructor_parameters_defs 
            | will be optimized.
        :prim_constructor_parameters_defs:
            | {}, optional
            | Dict with constructor parameters required by prim_constructor and/or 
            | default values for parameters that are not being optimized.
            | For example: {'fwhm':  30} will keep fwhm fixed and not optimize it.
            
    Returns:
        :spd, prims, M:
            | - spd: spectrum resulting from x
            | - spds: primary spds
            | - M: fluxes of all primaries
            
    Notes:
        1. '3mixer' - optimization algorithm: The triangle/trio method creates 
        for all possible combinations of 3 primary component spectra a spectrum
        that results in the target chromaticity using color3mixer() and then 
        optimizes the weights of each of the latter spectra such that adding 
        them (additive mixing) results in obj_vals as close as possible to 
        the target values.

    """
    if x.ndim == 1: x = np.atleast_2d(x)

    # get primary spectra:
    if prims is None:
        n_triangles = int(sp.special.factorial(n)/(sp.special.factorial(n-3)*sp.special.factorial(3)))
        # get triangle_strengths and remove them from x, remaining x are used to construct primaries:
        triangle_strengths = x[:,:n_triangles].T
        
        prims = prim_constructor(x[:,n_triangles:], n, wlr, 
                                 prim_constructor_parameter_types,
                                 **prim_constructor_parameter_defs)
    else:
        triangle_strengths = x.T
        wlr = prims[:1,:]
    
    # get primary chrom. coords.:
    Yxyi = colortf(prims,tf='spd>Yxy',bwtf={'cieobs':cieobs,'relative':False})

    # Get fluxes of each primary:
    M = _triangle_mixer(Yxy_target, Yxyi, triangle_strengths)
    
    if M.sum() > 0:
        # Scale M to have target Y:
        M = M*(Yxy_target[:,0]/(Yxyi[:,0]*M).sum())

    # Calculate optimized SPD:
    spd = np.vstack((prims[0],np.dot(M,prims[1:])))

    # When all out-of-gamut: set spd to NaN's:
    if M.sum() == 0:
        spd[1:,:] = np.nan
    
    return spd, prims, M


def _fitnessfcn(x, Yxy_target = np.array([[100,1/3,1/3]]), n = 3, wlr = [360,830,1], 
                cieobs=_CIEOBS, prims = None, prim_constructor = gaussian_prim_constructor,
                prim_constructor_parameter_types = gaussian_prim_parameter_types,
                prim_constructor_parameter_defs = {},
                out = 'F', F_rss = True, decimals = [5],
                obj_fcn = None, obj_fcn_pars = None, obj_fcn_weights = None, 
                obj_tar_vals = None, optimizer_type = '3mixer', verbosity = 1):
    """
    Fitness function that calculates closeness of solution x to target values 
    for specified objective functions. See docstring of spd_optimizer for info
    on the input parameters. If F_rss is True than the Root-Sum-of-Squares of F
    will be used as output during the optimization, else an F for each objective
    functions is output, allowing multi-objective optimizer to work towards a
    pareto optimal boundary.
    """

    x = np.atleast_2d(x)
    # setup parameters for use in loop(s):
    maxF = 1e308
    obj_vals = []
    F = []
    eps = 1e-16 #avoid division by zero
    if (obj_fcn is not None) & (len(decimals) == 1):
        decimals = decimals*len(obj_fcn)
    for i in range(x.shape[0]):
        
        # get spectrum for xi-parameters:
        xi = x[i:i+1,:]

        if optimizer_type == '3mixer':
            spdi, primsi, Mi = _spd_constructor_tri(xi, Yxy_target, n, wlr, cieobs = cieobs, 
                                                    prims = prims, prim_constructor = prim_constructor,
                                                    prim_constructor_parameter_types = prim_constructor_parameter_types,
                                                    prim_constructor_parameter_defs = prim_constructor_parameter_defs)
        else:
            raise Exception("Only the '3mixer' optimizer type has been implemented so far (April 10, 2020)")

        # store output for all xi when not optimizing
        if out != 'F':
            if i == 0:
                spds = spdi
                Ms = Mi
                primss = primsi
            else:
                spds = np.vstack((spds,spdi[1,:]))
                Ms = np.vstack((Ms,Mi))
                primss = np.dstack((primss,primsi))
                
        # create output string:
        Yxy_est = colortf(spdi,tf='spd>Yxy',bwtf={'cieobs':cieobs,'relative':False})
        if verbosity > 0:
            output_str = 'spdi = {:1.0f}/{:1.0f}, chrom. = E({:1.1f},{:1.4f},{:1.4f})/T({:1.1f},{:1.4f},{:1.4f}), '.format(i+1,x.shape[0],Yxy_est[0,0],Yxy_est[0,1],Yxy_est[0,2],Yxy_target[0,0],Yxy_target[0,1],Yxy_target[0,2])    

        # calculate all objective functions:
        if obj_fcn is not None:
            F_i = []
            obj_vals_i = []
            for j in range(len(obj_fcn)):
                if not np.isnan(spdi).any():
                    
                    # Set normalization factor for F-calculation:
                    obj_tar_vals_j = np.array(obj_tar_vals[j])
                    if (obj_tar_vals_j > 0).any():
                        f_normalize = obj_tar_vals_j
                        f_normalize[f_normalize==0] = 1
                    else:
                        f_normalize = 1

                    # Calculate objective function j:
                    if isinstance(obj_fcn[j],tuple): # one function for each objective:
                        obj_vals_ij = list(obj_fcn[j][0](spdi, **obj_fcn_pars[j]))
                    else: # one function for multiple objectives for increased speed:
                        obj_vals_ij = [np.squeeze(obj_fcn[j](spdi, **obj_fcn_pars[j]))]  
                     
                    # Store results in array:
                    obj_vals_i = obj_vals_i + obj_vals_ij
                    F_ij = list(obj_fcn_weights[j]*(((np.round(np.array(obj_vals_ij),int(decimals[j])) - obj_tar_vals_j + eps)**2)/((f_normalize + eps)**2))**0.5)
                    F_i = F_i + F_ij
                    
                else:
                    # Take care of output when spd construction failed:
                    if isinstance(obj_fcn[j],tuple):
                        nn = len(obj_fcn[j])-1
                        F_i = F_i + [maxF]*nn
                        obj_vals_ij = [np.nan]*nn
                    else:
                        F_i = F_i + [maxF]
                        obj_vals_ij = np.nan    
                    
                # Create output string:
                if (verbosity > 0) & (not np.isnan(spdi).any()):
                    if isinstance(obj_fcn[j],tuple):
                        output_str_sub = '('
                        for jj in range(len(obj_fcn[j])-1):
                            output_str_sub = output_str_sub + obj_fcn[j][jj+1] + ' = {:1.2f}, '
                        output_str_sub = output_str_sub[:-2] + ')'
                        output_str_sub = output_str_sub.format(*np.squeeze(obj_vals_ij))    
                        output_str = output_str + r'Fobj_#{:1.0f}'.format(j+1) + ' = {:1.2f} ' + output_str_sub + ', '
                        output_str = output_str.format(np.nansum(np.array(F_ij)**2)**0.5)
                    else:
                        fmt = 'E{:1.2f}(T{:1.2f}), '*len(obj_vals_ij)
                        fmt_values = []
                        if (not isinstance(obj_tar_vals[j],list)): obj_tar_vals_j = [obj_tar_vals[j]] 
                        for k in range(len(obj_vals_ij)):
                            fmt_values = fmt_values + [obj_vals_ij[k]] + [obj_tar_vals_j[k]]
                        output_str_ij = fmt.format(*fmt_values)
                        output_str = output_str + 'obj{:1.0f} = '.format(j+1) + output_str_ij 
        
        # Set output F, obj_vals_i when no objective functions were supplied:
        else:
            F_i = ((Yxy_est - Yxy_target)**2).sum(axis=1)**0.5 # use distance to guide out-of-gamut solutions to in-gamut ones
            if np.isnan(F_i).any():
                F_i = np.ones_like(F_i)*maxF
            obj_vals_i = np.nan
        
        # Print output_str:
        if verbosity > 0:
            print(output_str[:-2])
        
        # Add F_i and obj_vals_i to list for output
        F.append(F_i)
        obj_vals.append(obj_vals_i)
        
    
    # Take Root-Sum-of-Squares of delta((val - tar)**2):
    F = np.atleast_2d(F)
    if (F_rss == True) & (obj_fcn is not None):
         F = (np.nansum(F**2,axis = 1,keepdims = True)**0.5)[:,0]

    # return requested output:
    if out == 'F':
        return F
    elif out == 'spds,primss,Ms':
        return spds, primss, Ms
    else: 
        return eval(out)   

def _parse_bnds(bnds,n, min_ = -1e100, max_ = 1e100):
    """
    Setup the lower- and upper-bounds for n primary mixtures.
    """
    if bnds is None:
        lb = min_*np.ones((1,n))
        ub = max_*np.ones((1,n))
    else:
        if bnds[0] is None:
            lb = min_*np.ones((1,n))
        if bnds[1] is None:
            ub = max_*np.ones((1,n))
        lb = bnds[0]*np.ones((1,n)) if (isinstance(bnds[0],int) | isinstance(bnds[0],float)) else bnds[0]
        ub = bnds[1]*np.ones((1,n)) if (isinstance(bnds[1],int) | isinstance(bnds[1],float)) else bnds[1]
    return np.vstack((lb,ub))

def _get_minimize_options_and_Frss(minimize_method, minimize_opts = {}, n = None):
    """
    Set default options if not provided, as well as F_rss (True output Root-Sum-Squares of Fi in _fitnessfcn)
    """
    if (minimize_method == 'particleswarm') | (minimize_method == 'ps') | (minimize_method == 'nelder-mead'):
        F_rss = True
    elif (minimize_method == 'demo'):
        F_rss = False # must be output per objective function!!
    else:
        if 'F_rss' in minimize_opts:
            F_rss = minimize_opts['F_rss']
        else:
            F_rss = None

    if (minimize_opts == {}):
        if (minimize_method == 'particleswarm') | (minimize_method == 'ps'):
            minimize_opts = {'iters': 100, 'n_particles': 10, 'ftol': -np.inf,
                             'ps_opts' : {'c1': 0.5, 'c2': 0.3, 'w':0.9}}
            print(minimize_opts)
        elif (minimize_method == 'demo'):
            minimize_opts = math.DEMO.init_options(display = True)
        elif (minimize_method == 'nelder-mead'):
            if n is None: n = 10
            minimize_opts = {'xtol': 1e-5, 'disp': True, 'maxiter' : 1000*n, 'maxfev' : 1000*n,'fatol': 0.01}
        else:
            if not isinstance(minimize_method, str):
                minimize_method ={'type':'user-defined, specified as part of opt. function definition'}
                print('User already set the optimization options when defining the optimization function!')
            else:
                raise Exception ('Unsupported minimization method.')      
    return minimize_opts, F_rss      

def _start_optimization_tri(_fitnessfcn, n, fargs_dict, bnds, par_opt_types,
                            minimize_method, minimize_opts, Frss = None, x0 = None,
                            verbosity = 1, out = 'results'):
    """
    Start optimization of _fitnessfcn for n primaries using the specified minimize_method.
    
    Notes on minimize_method:
        1. Implemented: 'particleswarm', 'demo', 'nelder-mead'
        2. if not isinstance(minimize_method, str): 
        then it should contain an optimizer funtion with the following interface: 
            results = minimize_method(fitnessfcn, Nparameters, args = {}, 
                                      bounds = (lb, ub), verbosity = 1)
            With 'results' a dictionary containing various variables related to the
            optimization. It MUST contain a key 'x_final' containing the final optimized parameters.
            bnds must be [lowerbounds, upperbounds] with x-bounds ndarrays with values for each parameter.
            args is an argument with a dictionary containing the values for the fitnessfcn. Frss specifies
            whether the output of the fitnessfcn should be the Root-Sum-of-Squares (True) of 
            all weighted objective function values or not (False). Individual function values are
            required by true multi-objective optimizers.
    """
    
    n_triangle_strengths = int(sp.special.factorial(n)/(sp.special.factorial(n-3)*sp.special.factorial(3)))
    N = n_triangle_strengths + len(par_opt_types)*n # number of optimization parameters
    fargs_list = [v for k,v in fargs_dict.items()] 
    
    # Particle swarm optimization:
    if (minimize_method == 'particleswarm') | (minimize_method == 'ps'):
        results = math.particleswarm(_fitnessfcn, N, args = fargs_dict, bounds = (bnds[0],bnds[1]), 
                                             iters = minimize_opts['iters'],
                                             n_particles = minimize_opts['n_particles'],
                                             ftol = minimize_opts['ftol'],
                                             options = minimize_opts['ps_opts'],
                                             verbosity = verbosity)
   
    # Differential Evolutionary Multi-Objective Optimization:
    elif (minimize_method == 'demo'):
        if (bnds[0] is not None) & (bnds[1] is not None):
            xrange = np.hstack((bnds[0][:,None],bnds[1][:,None])).T
        else:
            raise Exception("Must set bnds for the 'demo' minimizer")
        fopt, xopt = math.DEMO.demo_opt(_fitnessfcn, N, args = fargs_list, xrange = xrange, options = minimize_opts)
        results = {'x_final': xopt,'F': fopt}
    
    # Local Simplex optimization using Nelder-Mead:
    elif (minimize_method == 'nelder-mead'):
        if x0 is None:
            x0 = np.array([np.random.uniform(bnds[0,i], bnds[1,i],1) for i in range(bnds.shape[1])]).T # generate random start value within bounds
        else:
            x0_triangle_strengths = np.ones((1,n_triangle_strengths))#np.array([np.random.uniform(bnds[0,i+2*n], bnds[1,i+2*n],1) for i in range(n_triangle_strengths)]).T
            x0 = np.hstack((x0_triangle_strengths, np.atleast_2d(x0)))
        results = math.minimizebnd(_fitnessfcn, x0, args = tuple(fargs_list), method = minimize_method, use_bnd = True, bounds = bnds, options = minimize_opts)
        x_final = np.abs(results['x'])
        results['x_final'] = x_final
    
    # Run user defined optimization algorithm:
    elif not isinstance(minimize_method, str):
        fargs_dict['F_rss'] = Frss
        results = minimize_method(_fitnessfcn, N, args = fargs_dict, 
                                  bounds = bnds, verbosity = verbosity,
                                  **minimize_opts)
    
    else:
        raise Exception ('Unsupported minimization method.')
    
    if out == 'results':
        return results
    else:
        return eval(out)


def _get_default_prim_parameters(nprims, parameter_types = ['peakwl', 'fwhm'], **kwargs):
    """
    Get dict with default primary parameters, dict with parameter bounds and 
    a list with parameters to be optimized.
    """
    keys = list(kwargs.keys())
    parameter_to_be_optimized = []
    parameter_defaults = {}
    parameter_bnds = {}
    for pt in parameter_types:
        # set up default parameter values (when not optimized):
        if pt not in keys:
            parameter_to_be_optimized.append(pt)
        else:
            pdefs = kwargs.pop(pt)
            if (isinstance(pdefs,int) | isinstance(pdefs,float)): pdefs = [pdefs]*nprims
            parameter_defaults[pt] = pdefs
        # Create bnds for parameters to be optimized:
        if pt not in keys:
            if pt+'_bnds' not in keys:
                parameter_bnds[pt+'_bnds'] = None
            else:
                parameter_bnds[pt+'_bnds'] = kwargs.pop(pt+'_bnds')
            parameter_bnds[pt+'_bnds'] = _parse_bnds(parameter_bnds[pt+'_bnds'], nprims) # parse temporary bnds to final ones
    parameter_defaults.update(kwargs) # add remaining parameters to dict with defaults for unpacking in prim_constructor function
    return parameter_defaults, parameter_bnds, parameter_to_be_optimized


def spd_optimizer2(target = np2d([100,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                  n = 4, wlr = [360,830,1], prims = None,
                  cieobs = _CIEOBS, out = 'spds,primss,Ms,results',
                  optimizer_type = '3mixer',
                  prim_constructor = gaussian_prim_constructor,
                  prim_constructor_parameter_types = ['peakwl', 'fwhm'], 
                  prim_constructor_parameter_defs = {},
                  decimals = [5], obj_fcn = None, obj_fcn_pars = [{}], 
                  obj_fcn_weights = None, obj_tar_vals = None,
                  triangle_strengths_bnds = None,
                  minimize_method = None, minimize_opts = {},
                  x0 = None, verbosity = 1):
    """
    Generate a spectrum with specified white point and optimized for certain 
    objective functions from a set of component spectra or component spectrum 
    model parameters.
    
    Args:
        :target: 
            | np2d([100,1/3,1/3]), optional
            | ndarray with Yxy chromaticity of target.
        :tar_type:
            | 'Yxy' or str, optional
            | Specifies the input type in :target: (e.g. 'Yxy' or 'cct')
        :cspace_bwtf:
            | {}, optional
            | Backward (cspace_to_xyz) transform parameters 
            | (see colortf()) to go from :tar_type: to 'Yxy').
        :n:
            | 4, optional
            | Number of primaries in light mixture.
        :wl: 
            | [360,830,1], optional
            | Wavelengths used in optimization when :prims: is not an
              ndarray with spectral data.
        :cieobs:
            | _CIEOBS, optional
            | CIE CMF set used to calculate chromaticity values, if not provided 
              in :Yxyi:.
        :optimizer_type:
            | '3mixer',  optional
            | Specifies type of chromaticity optimization 
            | For help on '3mixer' algorithm, see notes below.
        :prims:
            | ndarray of predefined primary spectra.
            | If None: they are built from optimization parameters using the 
            | function in :prim_constructor:
        :prim_constructor:
            | function that constructs the primaries from the optimization parameters
            | Should have the form: 
            |   prim_constructor(x, n, wl, prim_constructor_parameter_types, **prim_constructor_parameter_defs)
        :prim_constructor_parameter_types:
            | gaussian_prim_parameter_types ['peakwl', 'fwhm'], optional
            | List with strings of the parameters used by prim_constructor() to
            | calculate the primary spd. All parameters listed and that do not
            | have default values (one for each prim!!!) in prim_constructor_parameters_defs 
            | will be optimized.
        :prim_constructor_parameters_defs:
            | {}, optional
            | Dict with constructor parameters required by prim_constructor and/or 
            | default values for parameters that are not being optimized.
            | For example: {'fwhm':  30} will keep fwhm fixed and not optimize it.
        :decimals:
            | [5], optional
            | Rounding decimals of objective function values.
        :obj_fcn: 
            | [None] or list, optional
            | Function handles to objective function.
        :obj_fcn_weights:
            | [1] or list, optional.
            | Weigths for each obj. fcn
        :obj_fcn_pars:
            | [None] or list, optional
            | Parameter dicts for each obj. fcn.
        :obj_tar_vals:
            | [0] or list, optional
            | Target values for each objective function.
        :minimize_method:
            | 'nelder-mead', optional
            | Optimization method used by minimize function.
            | options: 
            |   - 'nelder-mead': Nelder-Mead simplex local optimization 
            |                    using the luxpy.math.minimizebnd wrapper
            |                    with method set to 'Nelder-Mead'.
            |   - 'particleswarm': Pseudo-global optimizer using particle swarms
            |                      (using wrapper luxpy.math.particleswarm)
            |   - 'demo' :  Differential Evolutionary Multiobjective Optimizatizer
            |               (using math.DEMO.demo_opt)
            |   - A user-defined minimization function (see _start_optimization_tri? for 
            |       info on the requirements of this function)
        :minimize_opts:
            | None, optional
            | Dict with minimization options. 
            | None defaults to the options depending on choice of minimize_method
            |  - 'Nelder-Mead'   : {'xtol': 1e-5, 'disp': True, 'maxiter': 1000*Nc,
            |                       'maxfev' : 1000*Nc,'fatol': 0.01}
            |  - 'particleswarm' : {'iters': 100, 'n_particles': 10, 'ftol': -np.inf,
            |                       'ps_opts' : {'c1': 0.5, 'c2': 0.3, 'w':0.9}}
            |  - 'demo' :          {'F': 0.5, 'CR': 0.3, 'kmax': 300, 'mu': 100, 'display': True}
            |  - dict with options for user-defined minimization method.
        :triangle_strength_bnds:
            | (None,None)
            | Specifies lower- and upper-bounds for the strengths of each of the primary
            | combinations that will be made during the optimization using '3mixer'.
        :x0:
            | None, optional
            | If None: a random starting value will be generated for the Nelder-Mead
            | minimization algorithm, else the user defined starting value will be used.
            | Note that it should only contain a value for each peakwl and/or fwhm that
            | is set to be optimized. The triangle_strengths are added automatically.
        :verbosity:
            | 0, optional
            | If > 0: print intermediate results.
        :out:
            | 'spds,primss,Ms,results', optional
            | Determines output of function (see :returns:).
            
    Returns:
        :returns: 
            | spds, primss,Ms,results
            | - 'spds': optimized spectrum (or spectra: for particleswarm and demo minimization methods)
            | - 'primss': primary spectra of each optimized spectrum
            | - 'Ms' : ndarrays with fluxes of each primary
            | - 'results': dict with optimization results

    Notes on the optimization algorithms:
         
        1. '3mixer': The triangle/trio method creates for all possible 
        combinations of 3 primary component spectra a spectrum that results in 
        the target chromaticity using color3mixer() and then optimizes the 
        weights of each of the latter spectra such that adding them 
        (additive mixing) results in obj_vals as close as possible to the 
        target values.
       
        2. '2mixer':
        APRIL 2020, NOT YET IMPLEMENTED!!
        Pairs (odd,even) of components are selected and combined using 
        'pair_strength'. This process is continued until only 3 (combined)
        intermediate sources remain. Color3mixer is then used to calculate 
        the fluxes for the remaining 3 sources, after which the fluxes of 
        all components are back-calculated.
    """
    
    if minimize_method is None:
        minimize_method = 'Nelder-Mead'
    
    # get wavelengths:
    wlr = getwlr(wlr)
    
    # convert whatever Yxy_target to actual Yxy values:
    Yxy_target = colortf(target, tf = tar_type+'>Yxy', bwtf = cspace_bwtf)
    if 'cieobs' in cspace_bwtf.keys():
        cieobs = cspace_bwtf['cieobs']
    

    # set up default prim parameters and bounds (p_d: default pars, p_b: bounds, p_o: to be optimized):
    par_def,par_bnds,par_opt_types = _get_default_prim_parameters(n, parameter_types = prim_constructor_parameter_types,
                                                                  **prim_constructor_parameter_defs)

    # setup triangle bounds and attach the other bounds at the end:
    n_triangle_strengths = int(sp.special.factorial(n)/(sp.special.factorial(n-3)*sp.special.factorial(3)))
    triangle_strengths_bnds = _parse_bnds(triangle_strengths_bnds, n_triangle_strengths, min_ = 0, max_ = 1)
    bnds = triangle_strengths_bnds
    for k,v in par_bnds.items(): bnds = np.hstack((bnds, v))

    # set default options if not provided:
    minimize_opts, F_rss = _get_minimize_options_and_Frss(minimize_method, n = bnds[0].size,  
                                                          minimize_opts = minimize_opts)
    
    # Create inputs for fit_fcn:    
    fargs_dict = {'Yxy_target':Yxy_target, 'n':n, 'wlr':wlr, 'cieobs':cieobs,
                  'prims':prims, 'prim_constructor':prim_constructor,
                  'prim_constructor_parameter_types':prim_constructor_parameter_types,
                  'prim_constructor_parameter_defs':prim_constructor_parameter_defs,
                  'out':'F', 'F_rss':F_rss, 'decimals':decimals, 'obj_fcn':obj_fcn,
                  'obj_fcn_pars':obj_fcn_pars,'obj_fcn_weights':obj_fcn_weights,
                  'obj_tar_vals':obj_tar_vals,'optimizer_type':optimizer_type,
                  'verbosity':verbosity}   
    
    # Perform optimzation:
    results = _start_optimization_tri(_fitnessfcn, n, fargs_dict, bnds, par_opt_types,
                                        minimize_method, minimize_opts, Frss = F_rss, 
                                        x0 = x0, verbosity = verbosity, out = 'results')
    
    x_final = results['x_final']
    fargs_dict['out'] = 'spds,primss,Ms'
    spds,primss,Ms = _fitnessfcn(x_final, **fargs_dict)
    
    if out == 'spds,primss,Ms,x_final,results':
        return spds, primss, Ms, x_final, results
    else:
        return eval(out)
        
      
 #------------------------------------------------------------------------------
if __name__ == '__main__':    

    run_example_1 = False # use pre-defined minimization methods

    run_example_2 = True # use user-defined  minimization method   

    import luxpy as lx
    cieobs = _CIEOBS
    
    # Set number of primaries and target chromaticity:
    n = 4
    target = np.array([[200,1/3,1/3]]) 
    
    # define function that calculates several objectives at the same time (for speed):
    def spd_to_cris(spd):
        Rf,Rg = lx.cri.spd_to_cri(spd, cri_type='ies-tm30',out='Rf,Rg')
        return Rf[0,0], Rg[0,0]   
    
    if run_example_1 == True:

        # start optimization:
        spd, prims, M = spd_optimizer2(target, tar_type = 'Yxy', cspace_bwtf = {},
                                      n = n, wlr = [360,830,1], prims = None,
                                      cieobs = _CIEOBS, out = 'spds,primss,Ms', 
                                      prim_constructor = gaussian_prim_constructor,
                                      prim_constructor_parameter_types = ['peakwl', 'fwhm'], 
                                      prim_constructor_parameter_defs = {'peakwl_bnds':[400,700],
                                                                         'fwhm_bnds':[5,100]},
                                      obj_fcn = [(spd_to_cris,'Rf','Rg')], 
                                      obj_fcn_pars = [{}], 
                                      obj_fcn_weights = [(1,1)], obj_tar_vals = [(90,110)],
                                      triangle_strengths_bnds = None,
                                      minimize_method = 'nelder-mead',
                                      minimize_opts = {},
                                      verbosity = 0)
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)


    #------------------------------------------------------------------------------
    # Example using a user-defined minimization method and a user-defined primary constructor:

    if run_example_2 == True:
                
        
        def user_prim_constructor(x, nprims, wlr, 
                              prim_constructor_parameter_types = ['peakwl','fwhm'], 
                              **prim_constructor_parameter_defs):
            """
            User defined prim constructor: lorenztian 2e order profile.
            """
            # Extract the primary parameters from x and prim_constructor_parameter_defs:
            pars = _extract_prim_optimization_parameters(x, nprims, prim_constructor_parameter_types,
                                                         prim_constructor_parameter_defs)
            # setup wavelengths:
            wlr = _setup_wlr(wlr)
            
            # Collect parameters from pars dict:
            n = 2*(2**0.5-1)**0.5
            spd = ((1 + (n*(pars['peakwl']-wlr.T)/pars['fwhm'])**2)**(-2)).T
            return np.vstack((wlr, spd))
        
        
        # Create a minimization function with the specified interface:
        def user_minim(fitnessfcn, Nparameters, args, bounds, verbosity = 1,
                       **minimize_opts):
            results = math.particleswarm(fitnessfcn, Nparameters, args = args, 
                                         bounds = bounds, 
                                         iters = 100, n_particles = 10, ftol = -np.inf,
                                         options = {'c1': 0.5, 'c2': 0.3, 'w':0.9},
                                         verbosity = verbosity)
            # Note that there is already a key 'x_final' in results
            return results
        
        
        # start optimization:
        spd, prims, M = spd_optimizer2(target, tar_type = 'Yxy', cspace_bwtf = {},
                                      n = n, wlr = [360,830,1], prims = None,
                                      cieobs = _CIEOBS, out = 'spds,primss,Ms', 
                                      prim_constructor = user_prim_constructor,
                                      prim_constructor_parameter_types = ['peakwl', 'fwhm'], 
                                      prim_constructor_parameter_defs = {'peakwl_bnds':[400,700],
                                                                         'fwhm_bnds':[5,100]},
                                      obj_fcn = [(spd_to_cris,'Rf','Rg')], 
                                      obj_fcn_pars = [{}], 
                                      obj_fcn_weights = [(1,1)], obj_tar_vals = [(90,110)],
                                      triangle_strengths_bnds = None,
                                      minimize_method = user_minim,
                                      minimize_opts = {'F_rss':True},
                                      verbosity = 1)
        
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)
    

