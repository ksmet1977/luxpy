# # -*- coding: utf-8 -*-
"""
Module for building and optimizing SPDs (2)
===========================================

This module implements a class based spectral optimizer. It differs from 
the spdoptimizer function in spdbuild.py, in that it can use several 
different minimization algorithms, as well as a user defined method. 
It is also written such that the user can easily write his own
primary constructor function. It supports the '3mixer' algorithm 
(but no '2mixer') and a 'no-mixer' algorithm (chromaticity as part of the list
of objectives) for calculating the mixing contributions of the primaries.

Functions
---------
 :gaussian_prim_constructor(): constructs a gaussian based primary set.
 
 :_setup_wlr(): Initialize the wavelength range for use with PrimConstructor.
 
 :_extract_prim_optimization_parameters(): Extract the primary parameters from the optimization vector x and the pdefs dict for use with PrimConstructor.

 :_stack_wlr_spd():  Stack the wavelength range 'on top' of the spd values for use with PrimConstructor.
 
 :PrimConstructor: class for primary (spectral) construction
     
 :Minimizer: class for minimization of fitness of each of the objective functions
 
 :ObjFcns: class to specify one or more objective functions for minimization
 
 :SpectralOptimizer: class for spectral optimization (initialization and run)
 
 :spd_optimizer2(): Generate a spectrum with specified white point and optimized
                   for certain objective functions from a set of component 
                   spectra or component spectrum model parameters 
                   (functional wrapper around SpectralOptimizer class).

                
Notes
-----
 1. See examples below (in '__main__') for use.
 
 2. Minimizer built-in options 'particleswarm' and 'nsga_ii' require
 pyswarms and pymoo packages to be installed. To minimize the dependency list 
 of luxpy on 'specialized' packages, these are not automatically installed
 along with luxpy. However, an attempt will be made to pip install them
 on first import (so please be patient when running these options for the first
 time). If the pip install fails, try a manual install using either pip or conda.
 


.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
import itertools
import numpy as np

from luxpy import (math, _CIEOBS, getwlr, colortf, xyz_to_cct)
from luxpy.utils import np2d



__all__ = ['PrimConstructor','Minimizer','ObjFcns','SpectralOptimizer',
           '_extract_prim_optimization_parameters', '_stack_wlr_spd','_setup_wlr',
           'spd_optimizer2', 'gaussian_prim_constructor', 'gaussian_prim_parameter_types',
           '_triangle_mixer', '_color3mixer']

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
    
    if Yxy1.ndim == 2:
        M = Yt*np.vstack((m1/Y1,m2/Y2,m3/Y3)).T
    else:
        M = Yt*np.dstack((m1/Y1,m2/Y2,m3/Y3))
    return M

def _triangle_mixer(Yxy_target, Yxyi, triangle_strengths):
    """
    Calculates the fluxes of each of the primaries to realize the target chromaticity Yxy_target given the triangle_strengths.
    """
    n = triangle_strengths.shape[0]
    N = Yxyi.shape[1]
    
    # Generate all possible 3-channel combinations (component triangles):
    combos = np.array(list(itertools.combinations(range(N), 3))) 
    Nc = combos.shape[0]
    
    # calculate fluxes to obtain target Yxyt:
    M3 = _color3mixer(Yxy_target,Yxyi[:,combos[:,0],:],Yxyi[:,combos[:,1],:],Yxyi[:,combos[:,2],:])
    
    # Get rid of out-of-gamut solutions:
    is_out_of_gamut =  (((M3<0).sum(axis=-1))>0)
    n_in_gamut = Nc - is_out_of_gamut.sum(axis=-1)
    n_in_gamut[n_in_gamut == 0] = 1.0 # avoid div by zero

    M3[is_out_of_gamut] = np.nan
    triangle_strengths[is_out_of_gamut] = np.nan
    if Nc > 1:
        M = np.zeros((n,N))
        
        # Calulate fluxes of all components from M3 and x_final:
        triangle_strengths = triangle_strengths/np.nansum(triangle_strengths,axis=-1,keepdims=True) # normalize to sum to 1
        M_final = triangle_strengths[...,None]*M3
        for i in range(N):
            M[:,i] = np.nansum(np.nansum(M_final*(combos == i)[None,...],axis=1),axis=-1)#/n_in_gamut
    else:
        M = M3[:,0,:]
    M[M.sum(axis=-1)==0] = np.nan
    return M

#------------------------------------------------------------------------------
def _stack_wlr_spd(wlr,spd):
    """
    Stack the wavelength range on top of the spd values for use with PrimConstructor.
    """
    spd = np.moveaxis(np.dstack((np.repeat(wlr,spd.shape[1],axis=1),spd)),0,-1)
    if spd.shape[0] == 1:
        return spd.squeeze(axis=0)
    else:
        return spd

def _setup_wlr(wlr):
    """
    Setup the wavelength range for use with PrimConstructor.
    """
    if len(wlr) == 3:
        wlr = getwlr(wlr)
    if wlr.ndim == 1:
        wlr = wlr[None,None,:]
    elif wlr.ndim == 2:
        wlr = wlr[None,:]
    return wlr.T

def _extract_prim_optimization_parameters(x, nprims, 
                                          prim_constructor_parameter_types, 
                                          prim_constructor_parameter_defs):
    """
    Extract the primary parameters from the optimization vector x and the prim_constructor_parameter_defs dict, for use with PrimConstructor..
    """
    types = prim_constructor_parameter_types
    pars = {}
    ct = 0
    for pt in types:
        if pt not in prim_constructor_parameter_defs: # extract value from x (to be optimized as not in _defs dict!)
            pars[pt] = np.array(x[:,(ct*nprims):(ct*nprims) + nprims])
            ct+=1
        else:
            pars[pt] = np.array(prim_constructor_parameter_defs[pt])
    return pars
         
            
def _get_default_prim_parameters(nprims, parameter_types = ['peakwl', 'fwhm'], **kwargs):
    """
    Get dict with default primary parameters, dict with parameter bounds and a list with parameters to be optimized.
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


#------------------------------------------------------------------------------
# Example code for a primiary constructor function:
gaussian_prim_parameter_types = ['peakwl', 'fwhm']

def gaussian_prim_constructor(x, nprims, wlr, ptypes, **pdefs):
    """
    Construct a set of nprim gaussian primaries with wavelengths wlr using the input in x and in kwargs.
    
    Args:
        :x:
            | ndarray (M x nprim) with optimization parameters.
        :nprim:
            | number of primaries
        :wlr:
            | wavelength range for which to construct a spectrum
        :prim_constructor:
            | function that constructs the primaries from the optimization parameters
            | Should have the form: 
            |   prim_constructor(x, n, wl, ptypes, pdefs)
        :ptypes:
            | gaussian_prim_parameter_types ['peakwl', 'fwhm'], optional
            | List with strings of the parameters used by PrimConstructor()) to
            | calculate the primary spd. All parameters listed and that do not
            | have default values (one for each prim!!!) in pdefs will be optimized.
        :pdefs:
            | Dict with constructor parameters required by PrimConstructor and/or 
            | default values for parameters that are not being optimized.
            | For example: {'fwhm':  [30]} will keep fwhm fixed and not optimize it.
            
    Returns:
        :spd:
            | ndarray with spectrum of nprim primaries (1st row = wavelengths)
            
            
    Example on how to create constructor:
        | ```def gaussian_prim_constructor(x, nprims, wlr, ptypes, **pdefs):```
        | ``` ```
        | ```    # Extract the primary parameters from x and pdefs:```
        | ```    pars = _extract_prim_optimization_parameters(x, nprims, ptypes, pdefs)```
        | ``` ```
        | ```    # Setup wavelengths:```
        | ```    wlr = _setup_wlr(wlr)```
        | ``` ```
        | ```    # Conversion factor for FWHM to sigma of Gaussian:```
        | ```    fwhm_to_sig = 1/(2*(2*np.log(2))**0.5) ```
        | ``` ```
        | ```    # Create spectral profile function: ```
        | ```    spd = np.exp(-0.5*((pars['peakwl']-wlr)/(pars['fwhm']*fwhm_to_sig))**2)```
        | ``` ```
        | ```    # Stack wlr and spd together: ```
        | ```    return _stack_wlr_spd(wlr,spd)``` 
        
    """
    
    # Extract the primary parameters from x and prim_constructor_parameter_defs:
    pars = _extract_prim_optimization_parameters(x, nprims, ptypes, pdefs)
    
    # setup wavelengths:
    wlr = _setup_wlr(wlr)
    
    # Collect parameters from pars dict:
    fwhm_to_sig = 1/(2*(2*np.log(2))**0.5) # conversion factor for FWHM to sigma of Gaussian
    return _stack_wlr_spd(wlr,np.exp(-0.5*((pars['peakwl']-wlr)/(pars['fwhm']*fwhm_to_sig))**2))  


class PrimConstructor():
    def __init__(self,f = gaussian_prim_constructor,
                  ptypes = ['peakwl', 'fwhm'], 
                  pdefs = {}):
        """
        Setup instance with a constructor function f for the primaries in the light mixture.
        
        Args:
            :f: 
                | gaussian_prim_constructor, optional
                | Constructor function handle.
                | see below for example on definition.
            :ptypes:
                | ['peakwl', 'fwhm'], optional
                | List of variables names in the constructor function.
            :pdefs:
                | {}, optional
                | Dictionary with default values for the constructor variables.
                | If a variable is in this dictionary it will not be optimized.
                | variable optimization bounds can also be specified as name+'_bnds', 
                | eg. {'peakwl_bnds':[400,700]} sets 400 and 700 as the lower and
                | upper bounds of the 'peakwl' variable.
            
        
            Example on how to create constructor:
                | ```def gaussian_prim_constructor(x, nprims, wlr, ptypes, **pdefs):```
                | ``` ```
                | ```    # Extract the primary parameters from x and pdefs:```
                | ```    pars = _extract_prim_optimization_parameters(x, nprims, ptypes, pdefs)```
                | ``` ```
                | ```    # Setup wavelengths:```
                | ```    wlr = _setup_wlr(wlr)```
                | ``` ```
                | ```    # Conversion factor for FWHM to sigma of Gaussian:```
                | ```    fwhm_to_sig = 1/(2*(2*np.log(2))**0.5) ```
                | ``` ```
                | ```    # Create spectral profile function: ```
                | ```    spd = np.exp(-0.5*((pars['peakwl']-wlr)/(pars['fwhm']*fwhm_to_sig))**2)```
                | ``` ```
                | ```    # Stack wlr and spd together: ```
                | ```    return _stack_wlr_spd(wlr,spd)``` 
        
        """
        self.f = f
        self.ptypes = ptypes
        self.pdefs = pdefs 
    
    def get_spd(self, nprim = None, wlr = [360,830,1]):
        """
        Get ndarray with spds for prims.
        
        Args:
            :nprim:
                | None, optional
                | If not None: generate nprim random prims (based fixed pars and bounds in pdefs) 
                | else: values for all pars should be defined in pdefs! 
                |       (nprims is determined by number of elements in pdefs[ptypes[0]])
        """
        pdefs_present = np.array([i for i,x in enumerate(self.ptypes) if x in self.pdefs])
        if (len(pdefs_present) == len(self.ptypes)): # everything needed is already in pdefs!!!
            return self.f([],nprim,wlr,self.ptypes,**self.pdefs)
        elif (nprim is not None) | (len(pdefs_present)>0):
            if (len(pdefs_present)>0): nprim = len(self.pdefs[self.ptypes[pdefs_present[0]]])
            # random x only for free bnds:
            fixed_pars_defs,free_pars_bnds,free_pars = _get_default_prim_parameters(nprim, self.ptypes,**self.pdefs) 
            bnds = np.array([[0,1]]).T
            for k,v in free_pars_bnds.items(): 
                if v is not None: # in case of self.prim not None!!
                    bnds = np.hstack((bnds, v))
            bnds = bnds[:,1:]
            x = np.array([np.random.uniform(bnds[0,i], bnds[1,i],1) for i in range(bnds.shape[1])]).T # generate random start value within bounds
            return self.f(x,nprim,wlr,self.ptypes,**self.pdefs)
        else:
            raise Exception('nprim = None in prim_constructor.')
        
        
       
class ObjFcns():
    def __init__(self,f = None, fp = [{}], fw = [1], ft = [0], ft_tol = [0],
                 f_requires_solution_info = [False], decimals = [5]):
        """
        Setup instance with objective functions, their input parameters, their respective weights and target values.
        
        Args:
            :f: 
                | None or list, optional
                | Function handles to objective function.
            :fp:
                | [{}] or list, optional
                | Parameter dicts for each obj. fcn.
            :fw:
                | [1] or list, optional.
                | Weigths for each obj. fcn
            :ft:
                | [0] or list, optional
                | Target values for each objective function.
            :ft_tol:
                | [0] or list, optional
                | Tolerance on target value.
                | If abs(f_j(x)-ft_j) < ft_tol_j:
                |    then objective value F_j (see notes below) 
                |    will be set to zero.
            :f_requires_solution_info:
                | [False] or list, optional
                | Set to True if the user-defined objective function requires
                | more info on the solution. 
                | If True the objective function should contain a keyword argument 'solution_info'.
                | In solution_info dict the user will find the following keys:
                |   - 'xs' : the current optimization values x for the spds being evaluated.
                |   - 'primss' : the primary spds corresponding to the current optimization values x.
                |   - 'Ms' : the channel fluxes corresponding to the current optimization values x.
                |   - 'Yxys' : the Yxy chromaticity coordinates corresponding to the current optimization values x.
            :decimals:
                | [5], optional
                | Rounding decimals of objective function values.
                
        Notes:
            1. The objective value F_j for each objective function f_j is calculated as: 
            |      F_j = (fw_j*abs((f_j(x)-ft_j)/ft_j)) 
            2. If ft_j==0 then ft_j in the denominator is set to 1 to avoid division by zero.
        """
        self.f = f
        self.fp = self._equalize_sizes(fp)
        self.fw = self._equalize_sizes(fw)
        self.ft = self._equalize_sizes(ft)
        self.ft_tol = self._equalize_sizes(ft_tol)
        self.decimals = self._equalize_sizes(decimals)
        self.f_requires_solution_info = self._equalize_sizes(f_requires_solution_info)
        self._get_normalization_factors()
        
        # get number of objectives:
        self.nobjs = 0
        if self.f is not None:
            for j in range(len(self.f)):
                if isinstance(self.f[j],tuple):
                    self.nobjs += (len(self.f[j]) - 1)
                else:
                    self.nobjs += 1
                
    
    def _equalize_sizes(self, x):    
        """
        Equalize structure of x to that of self.f for ease of looping of the objective functions in the fitness function
        """
        xs = []
        if (self.f is not None) & (len(x) == 1): 
            for f in self.f:
                if isinstance(f, tuple) & (not isinstance(x[0],tuple)) & (not isinstance(x[0],dict)):
                    xi = tuple(x*(len(f)-1))
                else:
                    xi = x[0]
                xs.append(xi)
        else:
            xs = x
        return xs
        
    def _calculate_fj(self, spdi, j = 0, solution_info = {}):
        """
        Calculate objective function j for input spd.
        """
        # Calculate objective function j:
        f_requires_solution_info = self.f_requires_solution_info[j][0] if isinstance(self.f_requires_solution_info[j],tuple) else self.f_requires_solution_info[j]
        
        if not f_requires_solution_info: 
            if isinstance(self.f[j],tuple): # one function for each objective:
                return self.f[j][0](spdi, **self.fp[j])
            else: # one function for multiple objectives for increased speed:
                return self.f[j](spdi, **self.fp[j])
        else:

            idx = solution_info['notnan_spds'] # to index notnan_spds
            # print(idx.shape,solution_info['xs'].shape,solution_info['Ms'].shape,solution_info['primss'].shape, solution_info['Yxys'].shape )
            solution_info_indexed = {'xs' : solution_info['xs'][idx],
                                     'Ms' : solution_info['Ms'][idx],
                                     'primss' : solution_info['primss'][idx],
                                     'Yxys' : solution_info['Yxys'][idx]
                                     }
            if isinstance(self.f[j],tuple): # one function for each objective:
                return self.f[j][0](spdi, solution_info = solution_info_indexed, **self.fp[j])
            else: # one function for multiple objectives for increased speed:
                return self.f[j](spdi, solution_info = solution_info_indexed, **self.fp[j])
        
    def _get_normalization_factors(self):
        """
          Set normalization factor for F-calculation
          """
        if self.f is not None:
            self.f_normalize = []
            for j in range(len(self.f)):
                self.ft[j] = np.array(self.ft[j])
                if (self.ft[j] != 0).any():
                    f_normalize = self.ft[j].copy()
                    f_normalize[f_normalize==0] = 1
                else:
                    f_normalize = 1
                self.f_normalize.append(f_normalize)
                
    def _get_fj_output_str(self, j, obj_vals_ij, F_ij = np.nan, verbosity = 1):
        """ get output string for objective function fj """
        output_str = ''

        if verbosity > 0:
            if isinstance(self.f[j],tuple):
                output_str_sub = '('
                for jj in range(len(self.f[j])-1): output_str_sub = output_str_sub + self.f[j][jj+1] + ' = {:1.' + '{:1.0f}'.format(self.decimals[j][jj]) + 'f}, '
                output_str_sub = output_str_sub[:-2] + ')'
                output_str_sub = output_str_sub.format(*np.squeeze(obj_vals_ij))    
                output_str = output_str + r'Fobj_#{:1.0f}'.format(j+1) + ' = {:1.3f} ' + output_str_sub + ', '
                output_str = output_str.format(np.nansum(np.array(F_ij)**2)**0.5)
            else:
                fmt = 'E{:1.2f}(T{:1.2f}), '*len(obj_vals_ij)
                fmt_values = []
                targetvals = [self.ft[j]] if not (isinstance(self.ft[j],list)) else self.ft[j]
                for k in range(len(obj_vals_ij)): fmt_values = fmt_values + [obj_vals_ij[k]] + [targetvals[k]]
                output_str = output_str + r'Fobj_#{:1.0f} = '.format(j+1) + fmt.format(*fmt_values)
        
        return output_str
            
        
        
class Minimizer():
    def __init__(self, method = 'Nelder-Mead', opts = {}, x0 = None, pareto = False, display = True):
        """
        Initialize minimization method.
        
        Args:
            :method:
                | 'Nelder-Mead', optional
                | Optimization method used by minimize function.
                | options: 
                |   - 'Nelder-Mead': Nelder-Mead simplex local optimization 
                |                    using the luxpy.math.minimizebnd wrapper
                |                    with method set to 'Nelder-Mead'.
                |   - 'demo' :  Differential Evolutionary Multiobjective Optimizatizer
                |               (using math.DEMO.demo_opt)
                |   - 'particleswarm': Pseudo-global optimizer using particle swarms
                |                      (from pyswarm wrapper module luxpy.math.pyswarms_particleswarm)
                |   - 'nsga_ii': Pareto multiobjective optimizer using the NSGA-II genetic algorithm
                |                      (from pymoo wrapper module luxpy.math.pymoo_nsga_ii)
                |   - A user-defined minimization function (see minimizer.apply? for 
                |       info on the requirements of this function)
            :opts:
                | None, optional
                | Dict with minimization options. 
                | None defaults to the options depending on choice of method
                |  - 'Nelder-Mead'   : {'xatol': 1e-5, 'disp': True, 'maxiter': 1000*Nc,
                |                       'maxfev' : 1000*Nc,'fatol': 0.01}
                |  - 'demo' :          {'F': 0.5, 'CR': 0.3, 'kmax': 300, 'mu': 100, 'display': True}
                |  - 'particleswarm' : {'iters': 100, 'n_particles': 10, 'ftol': -np.inf,
                |                       'ps_opts' : {'c1': 0.5, 'c2': 0.3, 'w':0.9}}
                |  - 'nsga_ii' : {'n_gen' : 40, 'n_pop' : 400, 'n_offsprings' : None,
                |                 'termination' : ('n_gen' , 40), 'seed' : 1,
                |                 'ga_opts' : {'sampling'  : ("real_random",{}),
                |                              'crossover' : ("real_sbx", {'prob' : 0.9, 'eta' : 15}),
                |                              'mutation'  : ("real_pm",  {'eta' : 20})}}
                |  - dict with options for user-defined minimization method.
            :pareto:
                | False, optional
                | Specifies whether the output of the fitnessfcn should be the Root-Sum-of-Squares 
                | of all weighted objective function values or not. Individual function values are
                | required by true multi-objective optimizers (i.e. pareto == True).
            :x0:
                | None, optional
                | Lets the user specify an optional starting value required by 
                | some minimizers (eg. 'Nelder-Mead'). It should contain only 
                | values for the free parameters in the primary constructor.
            :display:
                | True, optional
                | Turn native display options of minimizers on (True) or off (False).

        Notes on the user-defined minimizers:
            
            1. Must be initialzed using class Minimizer!
            2. If not isinstance(minimizer.method, str): 
                | then it should contain an minimization funtion with the following interface: 
                |     results = minimizer.method(fitnessfcn, npars, args = {}, bounds = (lb, ub), verbosity = 1)
                | With 'results' a dictionary containing various variables related to the optimization. 
                |  - It MUST contain a key 'x_final' containing the final optimized parameters.
                |  - bnds must be [lowerbounds, upperbounds] with x-bounds ndarrays with values for each parameter.
                |  - args is an argument with a dictionary containing the input arguments to the fitnessfcn.         
            3. Minimizer built-in options 'particleswarm' and 'nsga_ii' require
            pyswarms and pymoo packages to be installed. To minimize the dependency list 
            of luxpy on 'specialized' packages, these are not automatically installed
            along with luxpy. However, an attempt will be made to pip install them
            on first import (so please be patient when running these options for the first
            time). If the pip install fails, try a manual install using either pip or conda.

        """

        self.method = method
        self.opts = opts
        self.x0 = x0
        self.pareto = pareto
        self.display = display
        
        # Setup default optmization options
        self._set_defopts_and_pareto(pareto = pareto, x0 = self.x0, display = display)
        
    def _set_defopts_and_pareto(self, pareto = None, x0 = None, display = None):
        """
        Set default options if not provided, as well as pareto (False: output Root-Sum-Squares of Fi in _fitnessfcn).
        
        """
        if display is None: display = self.display
        self.display = display
        if (self.method == 'particleswarm') | (self.method == 'ps') | (self.method == 'Nelder-Mead'):
            self.pareto = False
        elif (self.method == 'demo') | (self.method == 'nsga_ii'):
            self.pareto = True # must be output per objective function!!
        else:
            if 'pareto' in self.opts:
                self.pareto = self.opts['pareto']
    
        # create dictionary with defaults options
        if (self.method == 'Nelder-Mead'):
            npar = 10 if x0 is None else x0[0].size
            tmp_opts = {'xatol': 1e-5, 'disp': display, 'maxiter' : 1000*npar, 'maxfev' : 1000*npar,'fatol': 0.01}

        elif (self.method == 'demo'):
            tmp_opts = math.DEMO.init_options(display = display)
            
        elif (self.method == 'particleswarm') | (self.method == 'ps'):
            tmp_opts = {'iters': 100, 'n_particles': 10, 'ftol': -np.inf,
                              'ps_opts' : {'c1': 0.5, 'c2': 0.3, 'w':0.9}}
            
        elif (self.method == 'nsga_ii'):
            tmp_opts = {'n_gen' : 40, 'n_pop' : 400, 'n_offsprings' : None,
                          'termination' : ('n_gen' , 40), 'seed' : 1,
                          'ga_opts' : {'sampling' : ("real_random",{}),
                                      'crossover': ("real_sbx", {'prob': 0.9, 'eta' : 15}),
                                      'mutation' : ("real_pm",  {'eta' : 20})}}
        
        else:
            if not isinstance(self.method, str):
                tmp_opts = {'type':'user-defined, specified as part of opt. function definition'}
                print('User-Defined minimizer: user should (have) set the optimization options when defining minimizer!')
            else:
                raise Exception ('Unsupported minimization method.')   

        # Update defaults with user entries:
        tmp_opts.update(self.opts)
        
        # overwrite existing self.opts with new entries
        self.opts = tmp_opts                      
                    
    def apply(self, fitness_fcn, npars, fitness_args_dict, bounds, verbosity = 1):
        """
        Run minimizer on fitness function with specified fitness_args_dict input arguments and bounds.
        """
        fitness_args_list = [v for k,v in fitness_args_dict.items()] 
        self.opts['display'] = np.array(verbosity).astype(bool)
         
        # Local Simplex optimization using Nelder-Mead:
        if (self.method == 'Nelder-Mead'):
            if self.x0 is None:
                x0 = np.array([np.random.uniform(bounds[0,i], bounds[1,i],1) for i in range(bounds.shape[1])]).T # generate random start value within bounds
            else:
                x0_triangle_strengths = np.ones((1,npars - len(self.x0)))#np.array([np.random.uniform(bnds[0,i+2*n], bnds[1,i+2*n],1) for i in range(n_triangle_strengths)]).T
                x0 = np.hstack((x0_triangle_strengths, np.atleast_2d(self.x0)))
            self.x0_with_triangle_strengths = x0
            self.opts['disp'] = self.opts.pop('display')
            results = math.minimizebnd(fitness_fcn, x0, args = tuple(fitness_args_list), method = self.method, use_bnd = True, bounds = bounds, options = self.opts)

        # Differential Evolutionary Multi-Objective Optimization:
        elif (self.method == 'demo'):
            if (bounds[0] is not None) & (bounds[1] is not None): 
                xrange = np.hstack((bounds[0][:,None],bounds[1][:,None])).T
            else:
                raise Exception("Minimizer: Must set bnds for the 'demo' minimizer")
            fopt, xopt = math.DEMO.demo_opt(fitness_fcn, npars, args = fitness_args_list, xrange = xrange, options = self.opts)
            results = {'x_final': xopt,'F': fopt}
        
        # Particle swarm optimization:
        elif (self.method == 'particleswarm') | (self.method == 'ps'):
            
            # import required minimizer function:
            try:
                from luxpy.math.pyswarms_particleswarm import particleswarm # lazy import
            except:
                raise Exception("Could not import particleswarm(), try a manual install of the 'pyswarms' package")
            
            # run minimizer:
            results = particleswarm(fitness_fcn, npars, args = fitness_args_dict, bounds = (bounds[0],bounds[1]), 
                                    iters = self.opts['iters'], n_particles = self.opts['n_particles'],
                                    ftol = self.opts['ftol'], options = self.opts['ps_opts'], verbosity = verbosity)
       
        # NSGA-II optimization:
        elif (self.method == 'nsga_ii'):
            
            # import required minimizer function:
            try:
                from luxpy.math.pymoo_nsga_ii import nsga_ii # lazy import
            except:
                raise Exception("Could not import nsga_ii(), try a manual install of the 'pymoo' package")
            
            # run minimizer:
            results = nsga_ii(fitness_fcn, npars, None, args = fitness_args_dict, bounds = (bounds[0],bounds[1]), 
                              verbosity = verbosity, pm_seed = self.opts['seed'],
                              pm_n_gen = self.opts['n_gen'], pm_n_pop = self.opts['n_pop'], 
                              pm_n_offsprings = self.opts['n_offsprings'],
                              pm_options = self.opts['ga_opts'],
                              pm_termination = self.opts['termination'])    

                
        # Run user defined optimization algorithm:
        elif not isinstance(self.method, str):
            results = self.method(fitness_fcn, 
                                  npars, 
                                  args = fitness_args_dict, 
                                  bounds = bounds, 
                                  verbosity = verbosity, 
                                  **self.opts)
        else:
            raise Exception ('Unsupported minimization method.')
        
        return results
        
class SpectralOptimizer():
    
    def __init__(self,target = np2d([100,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                  nprim = 4, wlr = [360,830,1], cieobs = _CIEOBS, 
                  out = 'spds,primss,Ms,results',
                  optimizer_type = '3mixer', triangle_strengths_bnds = None,
                  prim_constructor = PrimConstructor(), prims = None,
                  obj_fcn = ObjFcns(),
                  minimizer = Minimizer(method='Nelder-Mead'),
                  verbosity = 1):
        
        """
        | Initialize instance of SpectralOptimizer to generate a spectrum with 
        | specified white point and optimized for certain objective functions 
        | from a set of primary spectra or primary spectrum model parameters.
        
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
            :nprim:
                | 4, optional
                | Number of primaries in light mixture.
            :wl: 
                | [360,830,1], optional
                | Wavelengths used in optimization when :prims: is not an
                |  ndarray with spectral data.
            :cieobs:
                | _CIEOBS, optional
                | CIE CMF set used to calculate chromaticity values, if not provided 
                |  in :Yxyi:.
            :optimizer_type:
                | '3mixer',  optional
                | Specifies type of chromaticity optimization 
                | options: '3mixer', 'no-mixer'
                | For a short description of '3mixer' and 'no-mixer algorithms,
                | see notes below.
            :triangle_strengths_bnds:
                | None, optional
                | Bounds for the strengths of the triangle contributions ('3mixer')
                | or individual primary contributions ('no-mixer').
                | If None: bounds are set between [0,1].
            :prims:
                | ndarray of predefined primary spectra.
                | If None: they are built from optimization parameters using the 
                | function in :prim_constructor:
            :prim_constructor:
                | PrimConstructor(), optional
                | Instance of class PrimConstructor that has an attribute with a
                | function that constructs the primaries from a set of parameters.
                | PrimConstructor.f() should have the form: 
                |   prim_constructor(x, n, wl, ptypes, **pdefs)
                | see PrimConstructor.__docstring__ for more info.
            :obj_fcn:
                | ObjFcns(), optional
                | Instance of class ObjFcns that holds objective functions, their 
                | input arguments, target values, and relative weighting factors 
                | (for pseudo multi-objective optimization).
                | Notes: 
                |       1. The objective value F_j for each objective function f_j is calculated as: 
                |            F_j = (fw_j*abs((f_j(x)-ft_j)/ft_j)) 
                |       2. If ft_j==0 then ft_j in the denominator is set to 1 to avoid division by zero.
            :minimizer:
                | Minimizer(method='Nelder-Mead'), optional
                | Instance of the Minimizer class.
                | See Minimizer.__docstring__ for more info.
            :verbosity:
                | 0, optional
                | If > 0: print intermediate results 
                | (> 1 gives even more output for some options).
            :out:
                | 'spds,primss,Ms,results', optional
                | Determines output of function (see :returns:).
                
        Returns:
            :returns: 
                | spds, primss,Ms,results
                | - 'spds': optimized spectrum (or spectra: for demo, particleswarm and nsga_ii minimization methods)
                | - 'primss': primary spectra of each optimized spectrum
                | - 'Ms' : ndarrays with fluxes of each primary
                | - 'results': dict with optimization results
                |
                | Also see attribute 'optim_results' of class instance for info
                | on spds, prims, Ms, Yxy_estimate, obj_fcn.f function values and x_final.
    
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
            
            3. 'no-mixer':
            Spectrum is created as weighted sum of primaries. Any desired target 
            chromaticity should be specified as part of the objective functions.
        """
        self.target = target
        self.tar_type = tar_type
        self.cspace_bwtf = cspace_bwtf
        self.nprim = nprim
        self.wlr = getwlr(wlr)
        self._update_target(target, tar_type, cspace_bwtf = cspace_bwtf)
        self.cieobs = cieobs
        self.out = out
        self.optimizer_type = optimizer_type
        self.verbosity = verbosity
        
        # Setup primaries using either a PrimConstructor object or an ndarray:
        if not isinstance(prim_constructor,PrimConstructor):
            if prim_constructor is not None:
                self.prim_constructor = PrimConstructor(prim_constructor)
                print("prim_constructor argument not an instance of class PrimConstructor! Initializing as instance with defaults: pars = ['peakwl', 'fwhm'], opts = {}.")
            else:
                self.prim_constructor = PrimConstructor(None)
        else:
            self.prim_constructor = prim_constructor
        prim_constructor_pdefs = self.prim_constructor.pdefs
            
        self._update_nprim_prims(nprim = nprim, prims = prims)
                
        self.obj_fcn = obj_fcn
        
        if not isinstance(minimizer,Minimizer):
            self.minimizer = Minimizer(method=minimizer)
            print("minimizer argument not an instance of class Minimizer! Initializing as instance with defaults: opts = {}, x0 = None, display = True, pareto = False.")
        else:
            self.minimizer = minimizer
        self.optim_results = {'spd':None, 'prims': None, 'M':None, 'Yxy_est':None,'obj_fv': None,'x_final':None}

        # a. update fixed prim constructor pars and setup bound for free parameters
        # b. update triangle_strengths_bnds
        # c. construct a self.bnds attribute with bounds on triangle_strengths and all free parameters for an n-primary mixture.
        self._update_bnds(nprim = self.nprim, triangle_strengths_bnds = triangle_strengths_bnds, **prim_constructor_pdefs)
#        self.update(nprim = self.nprim, cieobs = self.cieobs, target = self.target, tar_type = self.tar_type, cspace_bwtf = self.cspace_bwtf,
#               triangle_strengths_bnds = triangle_strengths_bnds, **self.prim_constructor.pdefs)

    def _update_nprim_prims(self, nprim = None, prims = None):
        """
        Update prims (and nprim).
        """
        self.prims = prims
        self.nprim = nprim
        if prims is not None:
            if nprim is None: nprim = prims.shape[0]-1
            if isinstance(nprim, np.ndarray):
                nprim = list(nprim)
            if isinstance(nprim, list):
                prims = prims[[0] + nprim,:] # select specific prims in list
            nprim = prims.shape[0]-1 # set nprim
            self.prims = prims
            self.nprim = nprim
            self.wlr = prims[:1,:]
        if self.optimizer_type == '3mixer':
            if self.nprim < 3:
                raise Exception("nprim-error: number of primaries for optimizer_type == '3mixer' should be minimum 3!")
        
        

    def _update_target(self, target = None, tar_type = None, cspace_bwtf = None):
        """
        Update target chromaticity.
        """
        if target is None: target = self.target
        if tar_type is None: tar_type = self.tar_type
        if cspace_bwtf is None: cspace_bwtf = self.cspace_bwtf
        if target is not None:
            self.Yxy_target = colortf(target, tf = tar_type+'>Yxy', cspace_bwtf = cspace_bwtf)
        else:
            self.Yxy_target = None
        self.target = target
        self.tar_type = tar_type
        self.cspace_bwtf = cspace_bwtf
        if 'cieobs' in cspace_bwtf.keys():
            self.cieobs = cspace_bwtf['cieobs']
    
    def _update_prim_pars_bnds(self, nprim = None, **kwargs):
        """
        Get and set fixed and free parameters, as well as bnds on latter for an nprim primary mixture.
        """
        if nprim is not None: self.nprim = nprim
        fixed_pars_defs,free_pars_bnds,free_pars = _get_default_prim_parameters(self.nprim, self.prim_constructor.ptypes,**kwargs)
        if self.prims is None:
            self.prim_constructor.pdefs = fixed_pars_defs # update prim_constructor with defaults for fixed parameters 
            self.free_pars = free_pars
            self.free_pars_bnds = free_pars_bnds
        else:
            # in case of self.prim not None: then there are no bounds on 
            # those parameters (only triangle_strengths are free)!!
            for i, pt in enumerate(self.prim_constructor.ptypes):
                self.prim_constructor.pdefs[pt] = 'fixed_primary_set'
                if i == 0:
                    self.free_pars_bnds = {pt+'_bnds': None}
                else:
                    self.free_pars_bnds[pt+'_bnds'] = None
            self.free_pars = []
                
    def _get_n_triangle_strengths(self):
        """ Get number of triangle strengths"""
        from scipy.special import factorial # lazy import
        n_triangle_strengths = int(factorial(self.nprim)/(factorial(self.nprim-3)*factorial(3)))
        return n_triangle_strengths
            
    def _update_triangle_strengths_bnds(self, nprim = None, triangle_strengths_bnds = None):
        """
        Update bounds of triangle_strengths for for an nprim primary mixture.
        """
        if nprim is not None: self.nprim = nprim
        if self.optimizer_type == '3mixer':
            self.n_triangle_strengths = self._get_n_triangle_strengths()
            self.triangle_strengths_bnds = _parse_bnds(triangle_strengths_bnds, self.n_triangle_strengths, min_ = 0, max_ = 1)
            
        elif self.optimizer_type == 'no-mixer': # use triangle_strengths to store info on primary strengths in case of 'no-mixer'
            self.n_triangle_strengths = self.nprim
            self.triangle_strengths_bnds = _parse_bnds(triangle_strengths_bnds, self.n_triangle_strengths, min_ = 0, max_ = 1)
    
    def _update_bnds(self, nprim = None, triangle_strengths_bnds = None, **prim_kwargs): 
        """
        Update all bounds (triangle_strengths and those of free parameters of primary constructor) for an nprim primary mixture..
        """
        if nprim is not None: self.nprim = nprim
        self._update_prim_pars_bnds(nprim = self.nprim, **prim_kwargs)
        self._update_triangle_strengths_bnds(nprim = self.nprim, triangle_strengths_bnds = triangle_strengths_bnds)
        self.bnds = self.triangle_strengths_bnds
        for k,v in self.free_pars_bnds.items(): 
            if v is not None: # in case of self.prim not None!!
                self.bnds = np.hstack((self.bnds, v))
        self.npars = int(self.n_triangle_strengths + len(self.free_pars)*self.nprim)
        
    def update(self, nprim = None, prims = None, cieobs = None, target = None, tar_type = None, cspace_bwtf = None,
                triangle_strengths_bnds = None, **prim_kwargs):
        """
        Updates all that is needed when one of the input arguments is changed.
        """
        if cieobs is not None: self.cieobs = cieobs
        self._update_target(target = target, tar_type = tar_type, cspace_bwtf = cspace_bwtf)
        self._update_nprim_prims(nprim = nprim, prims = prims)
        self._update_bnds(nprim = nprim, triangle_strengths_bnds = triangle_strengths_bnds, **prim_kwargs)
        

    def _spd_constructor_tri(self, x):
        """
        Construct a mixture spectrum composed of n primaries using the 3mixer algorithm.
        
        Args:
            :x:
                | optimization parameters, first n!/(n-3)!*3! are the strengths of
                | the triangles in the '3mixer' algorithm.
                           
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
        if self.prims is None:
            # get triangle_strengths and remove them from x, remaining x are used to construct primaries:
            triangle_strengths = x[:,:self.n_triangle_strengths]
            
            prims = self.prim_constructor.f(x[:,self.n_triangle_strengths:], 
                                            self.nprim, 
                                            self.wlr,
                                            self.prim_constructor.ptypes,
                                            **self.prim_constructor.pdefs)
            if prims.ndim == 2:
                prims = prims[None,...] # ensure 3D-shape!! 
        else:
            triangle_strengths = x
            prims = self.prims.copy()
            if prims.ndim == 2:
                prims = prims[None,...]
            prims = np.repeat(prims,x.shape[0],axis=0)
           
        # reshape prims for colortf:
        wlr_ = prims[0,:1,:]
        prims_ = np.vstack((wlr_,prims[:,1:,:].reshape(prims.shape[0]*(prims.shape[1]-1),prims.shape[2])))

        # get primary chrom. coords.:
        Yxyi = colortf(prims_,tf='spd>Yxy',bwtf={'cieobs':self.cieobs,'relative':False})
        
        # reshape (N,nprims,3):
        Yxyi = Yxyi.reshape(prims.shape[0],prims.shape[1]-1,3)

        # Get fluxes of each primary:
        M = _triangle_mixer(self.Yxy_target, Yxyi, triangle_strengths)
        
        # Scale M to have target Y:
        isnan = np.isnan(M.sum(axis=-1))
        notnan = np.logical_not(isnan)

        if self.Yxy_target is not None:
            pass #if notnan.any(): M[notnan,:] = M[notnan,:]*(self.Yxy_target[...,0]/(Yxyi[notnan,:,0]*M[notnan,:]).sum(axis=-1,keepdims=True))
        else:
            M[notnan,:] = M[notnan,:]/M[notnan,:].max()
            
        # Calculate optimized SPD:
        spd = np.vstack((wlr_,np.einsum('ij,ijk->ik',M,prims[:,1:,:])))
       
        # When out-of-gamut: set spd to NaN's:
        spd[1:,:][isnan,:] = np.nan
        return spd, prims, M
    
    def _spd_constructor_nomixer(self, x):
        """
        Construct a mixture spectrum composed of n primaries using no mixer algorithm (just simple weighted sum of primaries).
        
        Args:
            :x:
                | optimization parameters, first n are the strengths of individual primaries.
                           
        Returns:
            :spd, prims, M:
                | - spd: spectrum resulting from x
                | - spds: primary spds
                | - M: fluxes of all primaries
                
        Notes:
            1. 'no-mixer' - simple weighted sum of primaries.
        """
        if x.ndim == 1: x = np.atleast_2d(x)

        # get primary spectra:
        if self.prims is None:
            # get prim_strengths and remove them from x, remaining x are used to construct primaries:
            prim_strengths = x[:,:self.nprim]
            
            prims = self.prim_constructor.f(x[:,self.nprim:], 
                                            self.nprim, 
                                            self.wlr,
                                            self.prim_constructor.ptypes,
                                            **self.prim_constructor.pdefs)
            if prims.ndim == 2:
                prims = prims[None,...] # ensure 3D-shape!! 
        else:
            prim_strengths = x
            prims = self.prims.copy()
            if prims.ndim == 2:
                prims = prims[None,...]
            prims = np.repeat(prims,x.shape[0],axis=0)

        # reshape prims for colortf:
        wlr_ = prims[0,:1,:]
        prims_ = np.vstack((wlr_,prims[:,1:,:].reshape(prims.shape[0]*(prims.shape[1]-1),prims.shape[2])))
        
        # get primary chrom. coords.:
        Yxyi = colortf(prims_,tf='spd>Yxy',bwtf={'cieobs':self.cieobs,'relative':False})

        # reshape (N,nprims,3):
        Yxyi = Yxyi.reshape(prims.shape[0],prims.shape[1]-1,3)
        
        # Get fluxes of each primary:
        M = prim_strengths
        
        # Scale M to have target Y:
        isnan = np.isnan(M.sum(axis=-1))
        notnan = np.logical_not(isnan)
        if self.Yxy_target is not None:
            if notnan.any():
                M[notnan,:] = M[notnan,:]*(self.Yxy_target[...,0]/(Yxyi[notnan,:,0]*M[notnan,:]).sum(axis=-1,keepdims=True))
        else:
            M[notnan,:] = M[notnan,:]/M[notnan,:].max()
            
        # Calculate optimized SPD:
        spd = np.vstack((wlr_,np.einsum('ij,ijk->ik',M,prims[:,1:,:])))

        # When out-of-gamut: set spd to NaN's:
        spd[1:,:][isnan,:] = np.nan
        return spd, prims, M

    def _fitness_fcn(self, x, out = 'F'):
        """
        Fitness function that calculates closeness of solution x to target values for specified objective functions. 
        """
        x = np.atleast_2d(x)

        # setup parameters for use in loop(s):
        maxF = 1e00
        F = []
        eps = 1e-16 #avoid division by zero

        if self.obj_fcn is not None:
            if (out != 'F') | (self.verbosity > 1):
                obj_fcn_vals = self.obj_fcn._equalize_sizes([np.nan])
    
        # # Loop over all xi to get all spectra:
        # for i in range(x.shape[0]):
            
        #     # get spectrum for xi-parameters:
        #     xi = x[i:i+1,:]
    
        #     if self.optimizer_type == '3mixer':
        #         spdi, primsi, Mi = self._spd_constructor_tri(xi)
        #     elif self.optimizer_type == 'no-mixer':
        #         spdi, primsi, Mi = self._spd_constructor_nomixer(xi)
        #     else:
        #         raise Exception("Only the '3mixer' and 'nomixer' optimizer type has been implemented so far (September 17, 2020)")
            
        #     if i == 0:
        #         spds = spdi
        #     else:
        #         spds = np.vstack((spds,spdi[1,:]))
            
        #     # store output for all xi when not optimizing
        #     if out != 'F':
        #         if i == 0:
        #             Ms, primss= Mi, primsi
        #         else:
        #             Ms = np.vstack((Ms,Mi))
        #             primss = np.dstack((primss,primsi))
        
        # get all spectra:
        if self.optimizer_type == '3mixer':
            spds, primss, Ms = self._spd_constructor_tri(x)
        elif self.optimizer_type == 'no-mixer':
            spds, primss, Ms = self._spd_constructor_nomixer(x)
        else:
            raise Exception("Only the '3mixer' and 'nomixer' optimizer type has been implemented so far (September 17, 2020)")

             
        # calculate for all spds at once:
        Yxy_ests = colortf(spds,tf='spd>Yxy',bwtf={'cieobs':self.cieobs,'relative':False})

        # calculate all objective functions on mass for all spectra:
        isnan_spds = np.isnan(spds[1:,:].sum(axis=1))
        if self.obj_fcn.f is not None:
            
            notnan_spds = np.logical_not(isnan_spds)
            if (notnan_spds.sum()>0):
                spds_tmp = np.vstack((spds[:1,:], spds[1:,:][notnan_spds,:])) # only calculate spds that are not nan
                
                for j in range(len(self.obj_fcn.f)):
                    
                    # Calculate objective function j:
                    obj_vals_j = self.obj_fcn._calculate_fj(spds_tmp, j = j, solution_info = {'xs' : x, 'primss' : primss, 'Ms': Ms, 'Yxys' : Yxy_ests, 'notnan_spds' : notnan_spds}).T 
                    
                    # Round objective values:
                    decimals = self.obj_fcn.decimals[j]
                    if not isinstance(decimals,tuple): decimals = (decimals,)
                    obj_vals_j = np.array([np.round(obj_vals_j[:,ii],int(decimals[ii])) for ii in range(len(decimals))]).T

                    
                    # Store F-results in array: 
                    delta = np.abs(obj_vals_j - self.obj_fcn.ft[j] + eps)
                    F_j = self.obj_fcn.fw[j]*delta/np.abs(self.obj_fcn.f_normalize[j] + eps)
                    
                    # If within tolerance on target values, set F_j to zero:
                    F_j[delta <= self.obj_fcn.ft_tol[j]] = 0.0
                    
                    if j == 0:
                        F_tmp = F_j
                    else:
                        F_tmp = np.hstack((F_tmp, F_j))
                        
                    if (out != 'F') | (self.verbosity > 1): 
                        # inflate to full size (including nan's):
                        obs_vals_j_tmp = np.ones((spds.shape[0]-1,obj_vals_j.shape[1]))
                        obs_vals_j_tmp[notnan_spds,:] =  obj_vals_j
                        
                        # store in array:
                        obj_fcn_vals[j] = obs_vals_j_tmp
    
                F = np.ones((spds.shape[0]-1,self.obj_fcn.nobjs))
                F[notnan_spds,:] = F_tmp
            else:
                F = np.ones((spds.shape[0]-1,self.obj_fcn.nobjs))*maxF
                
        # Set output F, obj_vals_i when no objective functions were supplied:
        else:
            F = ((Yxy_ests - self.Yxy_target)**2).sum(axis=1,keepdims=True)**0.5 # use distance to guide out-of-gamut solutions to in-gamut ones
            F[isnan_spds,:] = maxF
        
        # Print output:
        if self.verbosity > 1:
            for i in range(x.shape[0]):
                if self.Yxy_target is not None:
                    output_str = 'spdi = {:1.0f}/{:1.0f}, chrom. = E({:1.1f},{:1.4f},{:1.4f})/T({:1.1f},{:1.4f},{:1.4f}), '.format(i+1,x.shape[0],Yxy_ests[i,0],Yxy_ests[i,1],Yxy_ests[i,2],self.Yxy_target[0,0],self.Yxy_target[0,1],self.Yxy_target[0,2])    
                else:
                    output_str = 'spdi = {:1.0f}/{:1.0f}, chrom. = E({:1.1f},{:1.4f},{:1.4f})/T(not specified), '.format(i+1,x.shape[0],Yxy_ests[i,0],Yxy_ests[i,1],Yxy_ests[i,2])    

                if self.obj_fcn.f is not None:
                    # create output_str for spdi and print:
                    for j in range(len(self.obj_fcn.f)):
                        if isinstance(obj_fcn_vals[j],tuple):
                            obj_fcn_vals_ij = obj_fcn_vals[j]
                        else:
                            obj_fcn_vals_ij = obj_fcn_vals[j][i]
                        output_str = output_str + self.obj_fcn._get_fj_output_str(j, obj_fcn_vals_ij, F_ij =  F[i,j], verbosity = 1)
                    print(output_str,'\n')
                    
        # Take Root-Sum-of-Squares of delta((val - tar)**2):
        if (self.minimizer.pareto == False) & (self.obj_fcn.f is not None):
              F = (np.nansum(F**2,axis = 1,keepdims = True)**0.5)[:,0]

        if (self.verbosity > 0) & (self.verbosity <=1):
            print('F:', F)
        
        # store function values and spds, primss, M in attribute optim_results:     
        if ((self.obj_fcn.f is not None) & (out != 'F')): 
            self.optim_results['obj_fv'] = obj_fcn_vals
            self.optim_results['Yxy_est'] = Yxy_ests

        if (out != 'F'):
            self.optim_results['spd'] = spds
            self.optim_results['prims'] = primss
            self.optim_results['M'] = Ms
            self.optim_results['x_final'] = x
            
        # return requested output:
        if out == 'F':
            return F
        elif out == 'spds,primss,Ms':
            return spds, primss, Ms
        else: 
            return eval(out)   
        
        
    def start(self, verbosity = None, out = None):
        """
        Start optimization of _fitnessfcn for n primaries using the initialized minimizer and the selected optimizer_type.
        
        Returns variables specified in :out:
        """
        if verbosity is None: verbosity = self.verbosity
        optim_results = self.minimizer.apply(self._fitness_fcn, self.npars, {'out':'F'}, 
                                        self.bnds, verbosity)
    
        x_final = optim_results['x_final']
        spds,primss,Ms = self._fitness_fcn(x_final, out = 'spds,primss,Ms')
        
        if out is None:
            out = self.out
        if out == 'spds,primss,Ms,x_final,results':
            return spds, primss, Ms, x_final, optim_results
        else:
            return eval(out)

       
def spd_optimizer2(target = np2d([100,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                  n = 4, wlr = [360,830,1], prims = None,
                  cieobs = _CIEOBS, out = 'spds,primss,Ms,results',
                  optimizer_type = '3mixer',
                  prim_constructor = gaussian_prim_constructor,
                  prim_constructor_parameter_types = ['peakwl', 'fwhm'], 
                  prim_constructor_parameter_defs = {},
                  obj_fcn = None, 
                  obj_fcn_pars = [{}], 
                  obj_fcn_weights = [1], 
                  obj_tar_vals = [0], 
                  obj_tar_tols = [0],
                  decimals = [5], 
                  triangle_strengths_bnds = None,
                  minimize_method = 'Nelder-Mead', minimize_opts = {},
                  x0 = None, pareto = False, display = False,
                  verbosity = 1):
    """
    | Generate a spectrum with specified white point and optimized for certain objective 
    | functions from a set of primary spectra or primary spectrum model parameters.
    
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
            | in :Yxyi:.
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
            |   ``prim_constructor(x, n, wl, prim_constructor_parameter_types, **prim_constructor_parameter_defs)``
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
        :obj_fcn: 
            | [None] or list, optional
            | Function handles to objective function.
        :obj_fcn_weights:
            | [1] or list, optional.
            | Weigths for each obj. fcn
        :obj_fcn_pars:
            | [{}] or list, optional
            | Parameter dicts for each obj. fcn.
        :obj_tar_vals:
            | [0] or list, optional
            | Target values for each objective function.
        :obj_tar_tols:
            | [0] or list, optional
            | Tolerance of objective function values with target values.
        :decimals:
            | [5], optional
            | Rounding decimals of objective function values.
        :minimize_method:
            | 'Nelder-Mead', optional
            | Optimization method used by minimize function.
            | options: 
            |   - 'Nelder-Mead': Nelder-Mead simplex local optimization 
            |                    using the luxpy.math.minimizebnd wrapper
            |                    with method set to 'Nelder-Mead'.
            |   - 'demo' :  Differential Evolutionary Multiobjective Optimizatizer
            |               (using math.DEMO.demo_opt)
            |   - 'particleswarm': Pseudo-global optimizer using particle swarms
            |                      (from pyswarm wrapper module luxpy.math.pyswarms_particleswarm)
            |   - 'nsga_ii': Pareto multiobjective optimizer using the NSGA-II genetic algorithm
            |                      (from pymoo wrapper module luxpy.math.pymoo_nsga_ii)
            |   - A user-defined minimization function (see _start_optimization_tri? for 
            |       info on the requirements of this function)
        :minimize_opts:
            | None, optional
            | Dict with minimization options. 
            | None defaults to the options depending on choice of minimize_method
            |  - 'Nelder-Mead'   : {'xatol': 1e-5, 'disp': True, 'maxiter': 1000*Nc,
            |                       'maxfev' : 1000*Nc,'fatol': 0.01}
            |  - 'demo' :          {'F': 0.5, 'CR': 0.3, 'kmax': 300, 'mu': 100, 'display': True}
            |  - 'particleswarm' : {'iters': 100, 'n_particles': 10, 'ftol': -np.inf,
            |                       'ps_opts' : {'c1': 0.5, 'c2': 0.3, 'w':0.9}}
            |  - 'nsga_ii' : {'n_gen' : 40, 'n_pop' : 400, 'n_offsprings' : None,
            |                 'termination' : ('n_gen' , 40), 'seed' : 1,
            |                 'ga_opts' : {'sampling'  : ("real_random",{}),
            |                              'crossover' : ("real_sbx", {'prob' : 0.9, 'eta' : 15}),
            |                              'mutation'  : ("real_pm",  {'eta' : 20})}}
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
        :pareto:
            | False, optional
            | Specifies whether the output of the fitnessfcn should be the Root-Sum-of-Squares 
            | of all weighted objective function values or not. Individual function values are
            | required by true multi-objective optimizers (i.e. pareto == True).
        :display:
            | True, optional
            | Turn native display options of minimizers on (True) or off (False).

        :verbosity:
            | 0, optional
            | If > 0: print intermediate results.
        :out:
            | 'spds,primss,Ms,results', optional
            | Determines output of function (see :returns:).
            
    Returns:
        :returns: 
            | spds, primss,Ms,results
            | - 'spds': optimized spectrum (or spectra: for demo, particleswarm and nsga_ii minimization methods)
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
       
        2. '2mixer': APRIL 2020, NOT YET IMPLEMENTED!!
        Pairs (odd,even) of components are selected and combined using 
        'pair_strength'. This process is continued until only 3 (combined)
        intermediate sources remain. Color3mixer is then used to calculate 
        the fluxes for the remaining 3 sources, after which the fluxes of 
        all components are back-calculated.
    """
    
    so = SpectralOptimizer(target = target, tar_type = tar_type, cspace_bwtf = cspace_bwtf,
                            nprim = n, wlr = wlr, cieobs = cieobs, 
                            out = 'spds,primss,Ms,results',
                            optimizer_type = optimizer_type,
                            triangle_strengths_bnds = triangle_strengths_bnds,
                            prim_constructor = PrimConstructor(f = prim_constructor, 
                                                               ptypes = prim_constructor_parameter_types,
                                                               pdefs = prim_constructor_parameter_defs), 
                            prims = prims,
                            obj_fcn = ObjFcns(f = obj_fcn,
                                              fp = obj_fcn_pars,
                                              fw = obj_fcn_weights,
                                              ft = obj_tar_vals,
                                              ft_tol = obj_tar_tols,
                                              decimals = decimals),
                            minimizer = Minimizer(method = minimize_method,
                                                  opts = minimize_opts,
                                                  x0 = x0,
                                                  pareto = pareto,
                                                  display = display),
                            verbosity = verbosity)
    # start optimization:
    return so.start(out = out)

      
  #------------------------------------------------------------------------------
if __name__ == '__main__':  
    
    import matplotlib.pyplot as plt # lazy import
    
    run_example_class_1 = False # # class based example with pre-defined minimization methods
    
    run_example_class_2 = False # # class based example with pre-defined minimization methods and primary set

    run_example_class_3 = False # # class based example with user-defined  minimization method   

    run_example_class_4 = False # # class based example with pre-defined primaries and demo minimization

    run_example_class_4b = False # # class based example with pre-defined primaries and demo minimization (using slow approach of two separate functions)

    run_example_class_5 = False # # class based example with pre-defined primaries and demo minimization with obj_fcn using 'solution_info' to steer the optimization to solutions with the max number of channels 'on'
    
    run_example_class_5b = True # # class based example with pre-defined primaries and demo minimization with obj_fcn using 'solution_info' to steer the optimization to solutions with the max number of channels 'on'
    
    
    run_example_fcn_1 = False # function based example: use pre-defined minimization methods (spd_optimize2())

    run_example_fcn_2 = False # function based example: use user-defined  minimization method (spd_optimizer2()) 
    
    
    #--------------------------------------------------------------------------
    import luxpy as lx
    cieobs = '1964_10'
    
    
    # define function that calculates several objectives at the same time (for speed):
    def spd_to_cris(spd):
        Rf,Rg = lx.cri.spd_to_cri(spd, cri_type='ies-tm30',out='Rf,Rg')
        return np.vstack((Rf, Rg))   
    
    def spd_to_cct(spd):
        xyz = lx.spd_to_xyz(spd,cieobs=cieobs)
        cct, duv = xyz_to_cct(xyz,cieobs=cieobs,out='cct,duv')[0]
        cct = np.abs(cct) # out-of-lut ccts are encoded as negative
        return cct, duv
    
    #--------------------------------------------------------------------------
    if run_example_class_1 == True:
        
        so1 = SpectralOptimizer(target = np2d([100,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                              nprim = 4, wlr = [360,830,1], cieobs = cieobs, 
                              out = 'spds,primss,Ms,results',
                              optimizer_type = '3mixer', triangle_strengths_bnds = None,
                              prim_constructor = PrimConstructor(pdefs={'fwhm':[15],
                                                                        'peakwl_bnds':[400,700],
                                                                        'fwhm_bnds':[5,300]}), 
                              prims = None,
                              obj_fcn = ObjFcns(f=[(spd_to_cris,'Rf','Rg')], ft = [(90,110)], ft_tol = [(5,5)]),
                              minimizer = Minimizer(method='Nelder-Mead'),
                              verbosity = 0)
        # start optimization:
        spd,M = so1.start(out = 'spds,Ms')
        
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)
        
    #--------------------------------------------------------------------------
    if run_example_class_2 == True:
        
        # create set of 4 primaries with fixed fwhm at 15 nm:
        prims = PrimConstructor(pdefs={'peakwl':[450,520,580,630],'fwhm':[15],
                                        'peakwl_bnds':[400,700],
                                        'fwhm_bnds':[5,300]}).get_spd()
                    
        # create set of 4 primaries with fixed peakwl and fwhm bounds set to [5,300]:
        prims2 = PrimConstructor(pdefs={'peakwl':[450,520,580,630],
                                        'fwhm_bnds':[5,300]}).get_spd()
                    
        # create set of 4 primaries with free peakwl and fwhm bounds set to [400,700] and [5,300]:
        prims3 = PrimConstructor(pdefs={'peakwl_bnds':[400,700],
                                        'fwhm_bnds':[5,300]}).get_spd(nprim=4)
        
        so2 = SpectralOptimizer(target = np2d([100,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                              wlr = [360,830,1], cieobs = cieobs, 
                              out = 'spds,primss,Ms,results',
                              optimizer_type = '3mixer', triangle_strengths_bnds = None,
                              prim_constructor = None, 
                              prims = prims,
                              obj_fcn = ObjFcns(f=[(spd_to_cris,'Rf','Rg')], ft = [(90,110)]),
                              minimizer = Minimizer(method='Nelder-Mead'),
                              verbosity = 2)
#        # start optimization:
        spd,M = so2.start(out = 'spds,Ms')
        
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)

    
    #--------------------------------------------------------------------------
    if run_example_class_3 == True:
        
        
        def user_prim_constructor4(x, nprims, wlr, 
                              ptypes = ['peakwl','spectral_width'], 
                              **pdefs):
            """
            User defined prim constructor: lorenztian 2e order profile.
            """
            # Extract the primary parameters from x and prim_constructor_parameter_defs:
            pars = _extract_prim_optimization_parameters(x, nprims, ptypes, pdefs)
            # setup wavelengths:
            wlr = _setup_wlr(wlr)
            
            # Collect parameters from pars dict:
            n = 2*(2**0.5-1)**0.5
            spd = ((1 + (n*(pars['peakwl']-wlr)/pars['spectral_width'])**2)**(-2))
            
            # stack wavelengths and spd:
            return _stack_wlr_spd(wlr, spd)
        
        
        # Create a minimization function with the specified interface:
        from luxpy.math.pyswarms_particleswarm import particleswarm 
        def user_minim_ps(fitnessfcn, npars, args, bounds, verbosity = 1,**opts):
            results = particleswarm(fitnessfcn, npars, args = args, 
                                          bounds = bounds, 
                                          iters = 100, n_particles = 10, ftol = -np.inf,
                                          options = {'c1': 0.5, 'c2': 0.3, 'w':0.9},
                                          verbosity = verbosity)
            # Note that there is already a key 'x_final' in results
            return results
        
        from luxpy.math.pymoo_nsga_ii import nsga_ii 
        def user_minim_ga(fitnessfcn, npars, args, bounds, verbosity = 1,**opts):
            results = nsga_ii(fitnessfcn, npars, args = args, 
                              bounds = bounds, n_objectives = -1,
                              n_gen = 40, n_pop = 100, n_offspring = None,
                              verbosity = verbosity)
            # Note that there is already a key 'x_final' in results
            return results
        
        minimizer_nm = Minimizer(method='Nelder-Mead',pareto = False)
        minimizer_ps = Minimizer(method=user_minim_ps,pareto = False)
        minimizer_ga = Minimizer(method=user_minim_ga,pareto = True)
        
        so3 = SpectralOptimizer(target = np2d([100,1/3, 1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                              nprim = 4, wlr = [360,830,1], cieobs = cieobs, 
                              out = 'spds,primss,Ms,results',
                              optimizer_type = '3mixer', triangle_strengths_bnds = None,
                              prim_constructor = PrimConstructor(f = user_prim_constructor4, 
                                                                  ptypes=['peakwl','spectral_width'],
                                                                  pdefs = {'peakwl_bnds':[400,700],
                                                                          'spectral_width_bnds':[5,300]}), 
                              prims = None,
                              obj_fcn = ObjFcns(f=[(spd_to_cris,'Rf','Rg')], ft = [(90,110)]),
                              minimizer = minimizer_ps,
                              verbosity = 0)
        # start optimization:
        spd,M = so3.start(out = 'spds,Ms')
        
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)
        
        
    #--------------------------------------------------------------------------
    if run_example_class_4 == True:
        
        # create set of 4 primaries with fixed peakwl and fwhm bounds set to [5,300]:
        prims = PrimConstructor(pdefs={'peakwl':[450,520,580,630],'fwhm':[15],
                                        'peakwl_bnds':[400,700],
                                        'fwhm_bnds':[5,300]}).get_spd()
        
        # create set of 4 primaries with fixed peakwl and fwhm bounds set to [5,300]:
        prims2 = PrimConstructor(pdefs={'peakwl':[450,520,580,630],
                                        'fwhm_bnds':[5,300]}).get_spd()

        so4 = SpectralOptimizer(target = np2d([50,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                              wlr = [360,830,1], cieobs = cieobs, 
                              out = 'spds,primss,Ms,results',
                              optimizer_type = '3mixer', triangle_strengths_bnds = None,
                              prim_constructor = None, 
                              prims = prims2,
                              obj_fcn = ObjFcns(f=[(spd_to_cris,'Rf','Rg')], 
                                                ft = [(90,110)]),
                              minimizer = Minimizer(method='ps',
                                                    opts={'iters':50}),
                              verbosity = 2)
        # start optimization:
        spd,M = so4.start(out = 'spds,Ms')
        
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)
        
        #--------------------------------------------------------------------------
    if run_example_class_4b == True:

        # create set of 4 primaries with fixed peakwl and fwhm bounds set to [5,300]:
        prims = PrimConstructor(pdefs={'peakwl':[450,520,580,630],'fwhm':[15],
                                        'peakwl_bnds':[400,700],
                                        'fwhm_bnds':[5,300]}).get_spd()
        
        # create set of 4 primaries with fixed peakwl and fwhm bounds set to [5,300]:
        prims2 = PrimConstructor(pdefs={'peakwl':[450,520,580,630],
                                        'fwhm_bnds':[5,300]}).get_spd()

        so4 = SpectralOptimizer(target = np2d([50,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                              wlr = [360,830,1], cieobs = cieobs, 
                              out = 'spds,primss,Ms,results',
                              optimizer_type = '3mixer', triangle_strengths_bnds = None,
                              prim_constructor = None, 
                              prims = prims2,
                              obj_fcn = ObjFcns(f=[lx.cri.spd_to_iesrf,lx.cri.spd_to_iesrg], 
                                                ft = [90,110]),
                              minimizer = Minimizer(method='ps',
                                                    opts={'iters':50}),
                              verbosity = 2)
        # start optimization:
        spd,M = so4.start(out = 'spds,Ms')
        
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)
        
        #--------------------------------------------------------------------------
    if run_example_class_5 == True:
        
        # define user obj functions that asks for solutions info to use
        # when determining function output. For example, to give very 'bad'
        # Rf,Rg values (forcing the optimization away from this type of solutions)
        # when the number of channels with a relative weight smaller than 10%
        # is larger than 1 (in other words, try and force te search towards
        # solutins that have all channels sufficiently 'on'):
        def spd_to_cris_with_solution_info(spd, solution_info = {}):
            Ms = solution_info['Ms']
            Ms = Ms/Ms.max(axis=-1,keepdims=True) # normalize to max
            good_solutions = (Ms >= 0.1).sum(axis=-1) == Ms.shape[-1]
            out = np.ones((2,spd.shape[0] - 1))*(good_solutions.sum()/good_solutions.shape)*30 # higher number of on-channels is better (just setting zeros no matter how many bad channels makes it more difficult to optimize, same as setting a really really low value like -1000)
            if good_solutions.sum()>0:
                spds_good = np.vstack((spd[:1],spd[1:][good_solutions])) # only good solutions need calculating
                Rf,Rg = lx.cri.spd_to_cri(spds_good, cri_type='ies-tm30',out='Rf,Rg')
                RfRg = np.vstack((Rf, Rg))
                out[:,good_solutions] = RfRg 
            return out
              
        # create set of 4 primaries with fixed peakwl and fwhm bounds set to [5,300]:
        prims = PrimConstructor(pdefs={'peakwl':[450,470,500,520,560,580,630],
                                        'fwhm_bnds':[5,300]}).get_spd()

        so4 = SpectralOptimizer(target = np2d([50,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                              wlr = [360,830,1], cieobs = cieobs, 
                              out = 'spds,primss,Ms,results',
                              optimizer_type = '3mixer', triangle_strengths_bnds = None,
                              prim_constructor = None, 
                              prims = prims,
                              obj_fcn = ObjFcns(f=[(spd_to_cris_with_solution_info,'Rf','Rg')], 
                                                ft = [(90,110)],
                                                f_requires_solution_info=[True]),
                              minimizer = Minimizer(method='ps',
                                                    opts={'iters':50}),
                              verbosity = 2)
        # start optimization:
        spd,M = so4.start(out = 'spds,Ms')
        
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)
        
        # check graphically:
        plt.figure()
        plt.plot(spd[0],spd[1],'k', label = 'optimized spd (Rf={:1.0f},Rg={:1.0f})'.format(Rf[0],Rg[0]))
        cmap = lx.get_cmap(M.shape[-1],'jet')
        for i,Mi in enumerate(M[0]):
            plt.plot(prims[0],Mi*prims[i+1], color = cmap[i], linestyle = '--', label = 'channel {:1.0f} (rel. flux = {:1.3f})'.format(i,Mi/M[0].max()))
        plt.legend()
        
        
        
    if run_example_class_5b == True:
    
        # define user obj functions that asks for solutions info to use
        # when determining function output. For example, to give very 'bad'
        # Rf,Rg values (forcing the optimization away from this type of solutions)
        # when the number of channels with a relative weight smaller than 10%
        # is larger than 1 (in other words, try and force te search towards
        # solutins that have all channels sufficiently 'on'):
        def spd_to_iesrf_with_solution_info(spd, solution_info = {}):
            Ms = solution_info['Ms']
            Ms = Ms/Ms.max(axis=-1,keepdims=True) # normalize to max
            good_solutions = (Ms >= 0.1).sum(axis=-1) == Ms.shape[-1]
            out = np.ones((1,spd.shape[0] - 1))*(good_solutions.sum()/good_solutions.shape)*30 # higher number of on-channels is better (just setting zeros no matter how many bad channels makes it more difficult to optimize, same as setting a really really low value like -1000)
            if good_solutions.sum()>0:
                spds_good = np.vstack((spd[:1],spd[1:][good_solutions])) # only good solutions need calculating
                Rf = lx.cri.spd_to_iesrf(spds_good)
                out[:,good_solutions] = Rf 
            return out
        
        def spd_to_iesrg_with_solution_info(spd, solution_info = {}):
            Ms = solution_info['Ms']
            Ms = Ms/Ms.max(axis=-1,keepdims=True) # normalize to max
            good_solutions = (Ms >= 0.1).sum(axis=-1) == Ms.shape[-1]
            out = np.ones((1,spd.shape[0] - 1))*(good_solutions.sum()/good_solutions.shape)*30 # higher number of on-channels is better (just setting zeros no matter how many bad channels makes it more difficult to optimize, same as setting a really really low value like -1000)
            if good_solutions.sum()>0:
                spds_good = np.vstack((spd[:1],spd[1:][good_solutions])) # only good solutions need calculating
                Rf = lx.cri.spd_to_iesrg(spds_good)
                out[:,good_solutions] = Rf 
            return out
              
        # create set of 4 primaries with fixed peakwl and fwhm bounds set to [5,300]:
        prims = PrimConstructor(pdefs={'peakwl':[450,470,500,520,560,580,630],
                                        'fwhm_bnds':[5,300]}).get_spd()
    
        so4 = SpectralOptimizer(target = np2d([50,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                              wlr = [360,830,1], cieobs = cieobs, 
                              out = 'spds,primss,Ms,results',
                              optimizer_type = '3mixer', triangle_strengths_bnds = None,
                              prim_constructor = None, 
                              prims = prims,
                              obj_fcn = ObjFcns(f=[spd_to_iesrf_with_solution_info,
                                                   spd_to_iesrf_with_solution_info], 
                                                ft = [90,110],
                                                f_requires_solution_info=[True,True]),
                              minimizer = Minimizer(method='ps',
                                                    opts={'iters':50}),
                              verbosity = 2)
        # start optimization:
        spd,M = so4.start(out = 'spds,Ms')
        
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)
        
        # check graphically:
        plt.figure()
        plt.plot(spd[0],spd[1],'k', label = 'optimized spd (Rf={:1.0f},Rg={:1.0f})'.format(Rf[0],Rg[0]))
        cmap = lx.get_cmap(M.shape[-1],'jet')
        for i,Mi in enumerate(M[0]):
            plt.plot(prims[0],Mi*prims[i+1], color = cmap[i], linestyle = '--', label = 'channel {:1.0f} (rel. flux = {:1.3f})'.format(i,Mi/M[0].max()))
        plt.legend()

    #--------------------------------------------------------------------------
    if run_example_fcn_1 == True:

        # start optimization:
        spd, prims, M = spd_optimizer2(np2d([100,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                                      n = 4, wlr = [360,830,1], prims = None,
                                      cieobs = cieobs, out = 'spds,primss,Ms', 
                                      prim_constructor = gaussian_prim_constructor,
                                      prim_constructor_parameter_types = ['peakwl', 'fwhm'], 
                                      prim_constructor_parameter_defs = {'peakwl_bnds':[400,700],
                                                                         'fwhm_bnds':[5,300]},
                                      obj_fcn = [(spd_to_cris,'Rf','Rg')], 
                                      obj_fcn_pars = [{}], 
                                      obj_fcn_weights = [(1,1)], 
                                      obj_tar_vals = [(90,110)],
                                      triangle_strengths_bnds = None,
                                      minimize_method = 'Nelder-Mead',
                                      minimize_opts = {},
                                      verbosity = 2)
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)


    #-------------------------------------------------------------------------- 
    if run_example_fcn_2 == True:
                
        def user_prim_constructor2(x, nprims, wlr, 
                                   ptypes = ['peakwl','spectral_width'], 
                                   **pdefs):
            """
            User defined prim constructor: lorenztian 2e order profile.
            """
            # Extract the primary parameters from x and prim_constructor_parameter_defs:
            pars = _extract_prim_optimization_parameters(x, nprims, ptypes, pdefs)
            
            # setup wavelengths:
            wlr = _setup_wlr(wlr)
            
            # Collect parameters from pars dict:
            n = 2*(2**0.5-1)**0.5 # to ensure correct fwhm
            spd = ((1 + (n*(pars['peakwl']-wlr)/pars['spectral_width'])**2)**(-2))
            return _stack_wlr_spd(wlr, spd)
        
        
        # Create a minimization function with the specified interface:
        def user_minim2(fitnessfcn, Nparameters, args, bounds, verbosity = 1,
                       **minimize_opts):
            results = particleswarm(fitnessfcn, Nparameters, args = args, 
                                         bounds = bounds, 
                                         iters = 100, n_particles = 10, ftol = -np.inf,
                                         options = {'c1': 0.5, 'c2': 0.3, 'w':0.9},
                                         verbosity = verbosity)
            # Note that there is already a key 'x_final' in results
            return results
        
        
        # start optimization:
        spd, prims, M = spd_optimizer2(np2d([100,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                                      n = 4, wlr = [360,830,1], prims = None,
                                      cieobs = cieobs, out = 'spds,primss,Ms', 
                                      prim_constructor = user_prim_constructor2,
                                      prim_constructor_parameter_types = ['peakwl', 'spectral_width'], 
                                      prim_constructor_parameter_defs = {'peakwl_bnds':[400,700],
                                                                         'spectral_width_bnds':[5,300]},
                                      obj_fcn = [(spd_to_cris,'Rf','Rg')], 
                                      obj_fcn_pars = [{}], 
                                      obj_fcn_weights = [(1,1)], obj_tar_vals = [(90,110)],
                                      triangle_strengths_bnds = None,
                                      minimize_method = user_minim2,
                                      minimize_opts = {'pareto':False},
                                      verbosity = 2)
        
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)
  

