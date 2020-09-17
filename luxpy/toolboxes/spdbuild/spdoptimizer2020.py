# # -*- coding: utf-8 -*-
# """
# .. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
# """
import warnings
from luxpy import (math, _WL3, _CIEOBS, getwlr, SPD, spd_to_xyz, 
                    xyz_to_Yxy, colortf, xyz_to_cct)
from luxpy.utils import sp,np, plt, _EPS, np2d
from luxpy import cri 
from luxpy.math.particleswarm import particleswarm
from .spdbuilder2020 import (_get_default_prim_parameters, _parse_bnds, 
                              gaussian_prim_constructor, gaussian_prim_parameter_types,
                              _extract_prim_optimization_parameters, _setup_wlr, _triangle_mixer)
__all__ = ['PrimConstructor','Minimizer','ObjFcns','SpectralOptimizer']

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
            
        Example on how to create a constructor:
        
        | def gaussian_prim_constructor(x, nprims, wlr, ptypes, **pdefs):
        |    
        |    # Extract the primary parameters from x and pdefs:
        |    pars = _extract_prim_optimization_parameters(x, nprims, ptypes, pdefs)
        |    
        |    # setup wavelengths:
        |    wlr = _setup_wlr(wlr)
        |
        |    # Collect parameters from pars dict:
        |    fwhm_to_sig = 1/(2*(2*np.log(2))**0.5) # conversion factor for FWHM to sigma of Gaussian
        |    return np.vstack((wlr,np.exp(-((pars['peakwl']-wlr.T)/(pars['fwhm']*fwhm_to_sig))**2).T))     
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
    def __init__(self,f = None, fp = [{}], fw = [1], ft = [0], ft_tol = [0],  decimals = [5]):
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
        
    def _calculate_fj(self, spdi, j = 0):
        """
        Calculate objective function j for input spd.
        """
        # Calculate objective function j:
        if isinstance(self.f[j],tuple): # one function for each objective:
            return self.f[j][0](spdi, **self.fp[j])
        else: # one function for multiple objectives for increased speed:
            return self.f[j](spdi, **self.fp[j])
        
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
                |   - 'particleswarm': Pseudo-global optimizer using particle swarms
                |                      (using wrapper luxpy.math.particleswarm)
                |   - 'demo' :  Differential Evolutionary Multiobjective Optimizatizer
                |               (using math.DEMO.demo_opt)
                |   - A user-defined minimization function (see minimizer.apply? for 
                |       info on the requirements of this function)
            :opts:
                | None, optional
                | Dict with minimization options. 
                | None defaults to the options depending on choice of method
                |  - 'Nelder-Mead'   : {'xtol': 1e-5, 'disp': True, 'maxiter': 1000*Nc,
                |                       'maxfev' : 1000*Nc,'fatol': 0.01}
                |  - 'particleswarm' : {'iters': 100, 'n_particles': 10, 'ftol': -np.inf,
                |                       'ps_opts' : {'c1': 0.5, 'c2': 0.3, 'w':0.9}}
                |  - 'demo' :          {'F': 0.5, 'CR': 0.3, 'kmax': 300, 'mu': 100, 'display': True}
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
        elif (self.method == 'demo'):
            self.pareto = True # must be output per objective function!!
        else:
            if 'pareto' in self.opts:
                self.pareto = self.opts['pareto']
    
        if (self.opts == {}):
            if (self.method == 'particleswarm') | (self.method == 'ps'):
                self.opts = {'iters': 100, 'n_particles': 10, 'ftol': -np.inf,
                                  'ps_opts' : {'c1': 0.5, 'c2': 0.3, 'w':0.9}}
            elif (self.method == 'demo'):
                self.opts = math.DEMO.init_options(display = display)
            elif (self.method == 'Nelder-Mead'):
                npar = 10 if x0 is None else x0[0].size
                self.opts = {'xtol': 1e-5, 'disp': display, 'maxiter' : 1000*npar, 'maxfev' : 1000*npar,'fatol': 0.01}
            else:
                if not isinstance(self.method, str):
                    self.opts = {'type':'user-defined, specified as part of opt. function definition'}
                    print('User-Defined minimizer: user should (have) set the optimization options when defining minimizer!')
                else:
                    raise Exception ('Unsupported minimization method.')   
                    
                    
    def apply(self, fitness_fcn, npars, fitness_args_dict, bounds, verbosity = 1):
        """
        Run minimizer on fitness function with specified fitness_args_dict input arguments and bounds.
        """
        fitness_args_list = [v for k,v in fitness_args_dict.items()] 
        self.opts['display'] = np.array(verbosity).astype(bool)
                
        # Particle swarm optimization:
        if (self.method == 'particleswarm') | (self.method == 'ps'):
            results = particleswarm(fitness_fcn, npars, args = fitness_args_dict, bounds = (bounds[0],bounds[1]), 
                                          iters = self.opts['iters'], n_particles = self.opts['n_particles'],
                                          ftol = self.opts['ftol'], options = self.opts['ps_opts'], verbosity = verbosity)
       
        # Differential Evolutionary Multi-Objective Optimization:
        elif (self.method == 'demo'):
            if (bounds[0] is not None) & (bounds[1] is not None): 
                xrange = np.hstack((bounds[0][:,None],bounds[1][:,None])).T
            else:
                raise Exception("Minimizer: Must set bnds for the 'demo' minimizer")
            fopt, xopt = math.DEMO.demo_opt(fitness_fcn, npars, args = fitness_args_list, xrange = xrange, options = self.opts)
            print(fopt, ' xopt: ', xopt)
            results = {'x_final': xopt,'F': fopt}
        
        # Local Simplex optimization using Nelder-Mead:
        elif (self.method == 'Nelder-Mead'):
            if self.x0 is None:
                x0 = np.array([np.random.uniform(bounds[0,i], bounds[1,i],1) for i in range(bounds.shape[1])]).T # generate random start value within bounds
            else:
                x0_triangle_strengths = np.ones((1,npars - len(self.x0)))#np.array([np.random.uniform(bnds[0,i+2*n], bnds[1,i+2*n],1) for i in range(n_triangle_strengths)]).T
                x0 = np.hstack((x0_triangle_strengths, np.atleast_2d(self.x0)))
            self.x0_with_triangle_strengths = x0
            self.opts['disp'] = self.opts.pop('display')
            results = math.minimizebnd(fitness_fcn, x0, args = tuple(fitness_args_list), method = self.method, use_bnd = True, bounds = bounds, options = self.opts)
        
        # Run user defined optimization algorithm:
        elif not isinstance(self.method, str):
            results = self.method(fitness_fcn, npars, args = fitness_args_dict, bounds = bounds, 
                                  verbosity = verbosity, **self.opts)
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
                | For help on '3mixer' algorithm, see notes below.
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
                | - 'spds': optimized spectrum (or spectra: for particleswarm and demo minimization methods)
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
                
            
    def _update_triangle_strengths_bnds(self, nprim = None, triangle_strengths_bnds = None):
        """
        Update bounds of triangle_strengths for for an nprim primary mixture.
        """
        if nprim is not None: self.nprim = nprim
        if self.optimizer_type == '3mixer':
            self.n_triangle_strengths = int(sp.special.factorial(self.nprim)/(sp.special.factorial(self.nprim-3)*sp.special.factorial(3)))
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
            triangle_strengths = x[:,:self.n_triangle_strengths].T
            
            prims = self.prim_constructor.f(x[:,self.n_triangle_strengths:], 
                                            self.nprim, 
                                            self.wlr,
                                            self.prim_constructor.ptypes,
                                            **self.prim_constructor.pdefs)
        else:
            triangle_strengths = x.T
            prims = self.prims
            
        # get primary chrom. coords.:
        Yxyi = colortf(prims,tf='spd>Yxy',bwtf={'cieobs':self.cieobs,'relative':False})

        # Get fluxes of each primary:
        M = _triangle_mixer(self.Yxy_target, Yxyi, triangle_strengths)

        if M.sum() > 0:
            # Scale M to have target Y:
            M = M*(self.Yxy_target[:,0]/(Yxyi[:,0]*M).sum())

        # Calculate optimized SPD:
        spd = np.vstack((prims[0],np.dot(M,prims[1:])))
    
        # When all out-of-gamut: set spd to NaN's:
        if M.sum() == 0:
            spd[1:,:] = np.nan
        
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
            prim_strengths = x[:,:self.nprim].T
            
            prims = self.prim_constructor.f(x[:,self.nprim:], 
                                            self.nprim, 
                                            self.wlr,
                                            self.prim_constructor.ptypes,
                                            **self.prim_constructor.pdefs)
        else:
            prim_strengths = x.T
            prims = self.prims
            
        # get primary chrom. coords.:
        Yxyi = colortf(prims,tf='spd>Yxy',bwtf={'cieobs':self.cieobs,'relative':False})

        # Get fluxes of each primary:
        M = prim_strengths.T

        if M.sum() > 0:
            # Scale M to have target Y:
            #M = M*(self.Yxy_target[:,0]/(Yxyi[:,0]*M).sum())
            M = M/M.max() # no target available!

        # Calculate optimized SPD:
        spd = np.vstack((prims[0],np.dot(M,prims[1:])))
    
        # When all out-of-gamut: set spd to NaN's:
        if M.sum() == 0:
            spd[1:,:] = np.nan
        
        return spd, prims, M

    def _fitness_fcn(self, x, out = 'F'):
        """
        Fitness function that calculates closeness of solution x to target values for specified objective functions. 
        """
        x = np.atleast_2d(x)

        # setup parameters for use in loop(s):
        maxF = 1e308
        F = []
        eps = 1e-16 #avoid division by zero
        if self.obj_fcn is not None:
            if (out != 'F') | (self.verbosity > 1):
                obj_fcn_vals = self.obj_fcn._equalize_sizes([np.nan])
    
        # Loop over all xi to get all spectra:
        for i in range(x.shape[0]):
            
            # get spectrum for xi-parameters:
            xi = x[i:i+1,:]
    
            if self.optimizer_type == '3mixer':
                spdi, primsi, Mi = self._spd_constructor_tri(xi)
            elif self.optimizer_type == 'no-mixer':
                spdi, primsi, Mi = self._spd_constructor_nomixer(xi)
            else:
                raise Exception("Only the '3mixer' and 'nomixer' optimizer type has been implemented so far (September 17, 2020)")
            
            if i == 0:
                spds = spdi
            else:
                spds = np.vstack((spds,spdi[1,:]))
            
            # store output for all xi when not optimizing
            if out != 'F':
                if i == 0:
                    Ms, primss= Mi, primsi
                else:
                    Ms = np.vstack((Ms,Mi))
                    primss = np.dstack((primss,primsi))
             
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
                    obj_vals_j = self.obj_fcn._calculate_fj(spds_tmp, j = j).T 
                    
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

       
      
  #------------------------------------------------------------------------------
if __name__ == '__main__':  
    
    run_example_1 = False # # class based example with pre-defined minimization methods
    
    run_example_2 = False # # class based example with pre-defined minimization methods and primary set

    run_example_3 = False # # class based example with user-defined  minimization method   

    run_example_4 = True # # class based example with pre-defined primaries and demo minimization

    import luxpy as lx
    cieobs = '1964_10'
    
    # Set number of primaries and target chromaticity:
    nprim = 4
    target = np.array([[200,1/3,1/3]]) 
    
    # define function that calculates several objectives at the same time (for speed):
    def spd_to_cris(spd):
        Rf,Rg = lx.cri.spd_to_cri(spd, cri_type='ies-tm30',out='Rf,Rg')
        return np.vstack((Rf, Rg))     
    
    if run_example_1 == True:
        
        
        so1 = SpectralOptimizer(target = np2d([100,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                              nprim = nprim, wlr = [360,830,1], cieobs = cieobs, 
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
        
        
    if run_example_2 == True:
        
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

    
    
    if run_example_3 == True:
        
        
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
            spd = ((1 + (n*(pars['peakwl']-wlr.T)/pars['spectral_width'])**2)**(-2)).T
            return np.vstack((wlr, spd))
        
        
        # Create a minimization function with the specified interface:
        def user_minim4(fitnessfcn, npars, args, bounds, verbosity = 1,**opts):
            results = particleswarm(fitnessfcn, npars, args = args, 
                                          bounds = bounds, 
                                          iters = 100, n_particles = 10, ftol = -np.inf,
                                          options = {'c1': 0.5, 'c2': 0.3, 'w':0.9},
                                          verbosity = verbosity)
            # Note that there is already a key 'x_final' in results
            return results
        
        
        
        so3 = SpectralOptimizer(target = np2d([100,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                              nprim = 4, wlr = [360,830,1], cieobs = cieobs, 
                              out = 'spds,primss,Ms,results',
                              optimizer_type = '3mixer', triangle_strengths_bnds = None,
                              prim_constructor = PrimConstructor(f = user_prim_constructor4, 
                                                                  ptypes=['peakwl','spectral_width'],
                                                                  pdefs = {'peakwl_bnds':[400,700],
                                                                          'spectral_width_bnds':[5,300]}), 
                              prims = None,
                              obj_fcn = ObjFcns(f=[(spd_to_cris,'Rf','Rg')], ft = [(90,110)]),
                              minimizer = Minimizer(method=user_minim4),
                              verbosity = 1)
        # start optimization:
        spd,M = so3.start(out = 'spds,Ms')
        
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)
        
    if run_example_4 == True:
        
        # create set of 4 primaries with fixed peakwl and fwhm bounds set to [5,300]:
        prims = PrimConstructor(pdefs={'peakwl':[450,520,580,630],'fwhm':[15],
                                        'peakwl_bnds':[400,700],
                                        'fwhm_bnds':[5,300]}).get_spd()
        
        # create set of 4 primaries with fixed peakwl and fwhm bounds set to [5,300]:
        prims2 = PrimConstructor(pdefs={'peakwl':[450,520,580,630],
                                        'fwhm_bnds':[5,300]}).get_spd()

        so4 = SpectralOptimizer(target = np2d([100,1/3,1/3]), tar_type = 'Yxy', cspace_bwtf = {},
                              wlr = [360,830,1], cieobs = cieobs, 
                              out = 'spds,primss,Ms,results',
                              optimizer_type = '3mixer', triangle_strengths_bnds = None,
                              prim_constructor = None, 
                              prims = prims2,
                              obj_fcn = ObjFcns(f=[(spd_to_cris,'Rf','Rg')], ft = [(90,110)]),
                              minimizer = Minimizer(method='Nelder-Mead'),
                              verbosity = 2)
        # start optimization:
        spd,M = so4.start(out = 'spds,Ms')
        
        Rf, Rg = spd_to_cris(spd)
        print('obj_fcn1:',Rf)
        print('obj_fcn2:',Rg)