# -*- coding: utf-8 -*-
"""
    Wrapper around pyswarms.
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

__all__ = ['particleswarm']

# Import modules
import subprocess
from luxpy.utils import np, plt, is_importable


# import pyswarms (and if necessary install it):
success = is_importable('pyswarms', try_pip_install = True)
if success:
    import pyswarms as ps
    from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

__all__ = ['particleswarm']

def particleswarm(objfcn, dimensions, args = {}, use_bnds = True, bounds = (None,None), 
                  iters = 100, n_particles = 10, ftol = -np.inf,
                  options = {'c1': 0.5, 'c2': 0.3, 'w':0.9},
                  verbosity = 1,
                  **kwargs):
    """
    Global minimization function using particle swarms (wrapper around pyswarms.single.GlobalBestPSO)
    
    Args:
        :objfcn:
            | objective function
            | Should output a vector with cost values for each of the particles.
        :dimensions: 
            | Number of parameter values for the objective function. 
        :args:
            | Dict with objfcn input parameters (except the to be optimized x)
        :use_bnd:
            | True, optional
            | False: omits bounds and defaults to regular minimize function.
        :bounds:
            | (lower, upper), optional
            | Tuple of lists or dicts (x0_keys is None) of lower and upper bounds 
              for each of the parameters values.
        :iters:
            | 100, optional
            | Number of swarm iterations
        :n_particles:
            | 10, optional
            | Number of particles in swarm
        :ftol:
            | -np.inf, optional
            | Relative error in objective_func(best_pos) acceptable for
            | convergence. Default is :code:`-np.inf`
        options:
            | {'c1': 0.5, 'c2': 0.3, 'w':0.9}, optional
            | dict with keys {'c1', 'c2', 'w'}
            | A dictionary containing the parameters for the specific
            | optimization technique.
            |  - 'c1' : float, cognitive parameter
            |  - 'c2' : float, social parameter
            |  - 'w' : float, inertia parameter
        :verbosity:
            | 1, optional
            | If > 0: plot the cost history (see pyswarms's plot_cost_history function)
        :kwargs: 
            | allows input for other type of arguments for GlobalBestPSO
         
    Note:
        For more info on other input arguments, see 'ps.single.GlobalBestPSO?'
         
    Returns:
        :res: 
            | dict with output of minimization:
            | keys():
            |   - 'x_final': final solution x
            |   - 'cost': final function value of obj_fcn()
            |   - and some of the input arguments characterizing the 
            |       minimization, such as n_particles, bounds, ftol, options, optimizer.

    Reference:
        1. pyswarms documentation: https://pyswarms.readthedocs.io/
    """
         
    if (bounds[0] is None) & (bounds[1] is None):
        use_bnds = False
    if use_bnds == True:
        kwargs['bounds'] = bounds
        
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, 
                                        dimensions=dimensions, 
                                        options=options,
                                        ftol = ftol,
                                        **kwargs)

    cost, pos = optimizer.optimize(objfcn, iters=1000, **args)
    
    if verbosity > 0:
        # Plot cost history:
        plot_cost_history(cost_history=optimizer.cost_history)
    
    # create output dictionary:
    res = {'x_final': pos, 'cost' : cost,
           'iters': iters, 'n_particles' : n_particles,
           'bounds': bounds, 'ftol':ftol,
           'options': options, 'optimizer' : optimizer}
    
    return res

def rosenbrock_with_args(x, a, b, c=0):
    f = (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0] ** 2) ** 2 + c
    return f

if __name__ == '__main__':
    
    from pyswarms.utils.functions import single_obj as fx
    
    #--------------------------------------------------------------------------
    # 1: Rastrigin example:
    objfcn = fx.rastrigin
    fargs = {}
    dimensions = 2

    max_bound = 5.12 * np.ones(2)
    min_bound = - max_bound

    res1 = particleswarm(objfcn, dimensions, args = fargs, use_bnds = True, bounds = (min_bound, max_bound), 
                                            iters = 100, n_particles = 10, ftol = -np.inf,
                                            options = {'c1': 0.5, 'c2': 0.3, 'w':0.9},
                                            verbosity = 1)
    
    #--------------------------------------------------------------------------
    # 2: Rosenbrock example:
    objfcn = rosenbrock_with_args
    fargs = {"a": 1.0, "b": 100.0, 'c':0}
    dimensions = 2
    res2 = particleswarm(objfcn, dimensions, args = fargs, use_bnds = False, bounds = (None,None), 
                                            iters = 100, n_particles = 10, ftol = -np.inf,
                                            options = {'c1': 0.5, 'c2': 0.3, 'w':0.9},
                                            verbosity = 1)
    
  
    
                 
    