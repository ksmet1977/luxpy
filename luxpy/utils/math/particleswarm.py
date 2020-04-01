# -*- coding: utf-8 -*-
"""
    Wrapper around pyswarms.
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

__all__ = ['particleswarm']

# Import modules
import numpy as np
import matplotlib as plt
import subprocess
import sys

# Import PySwarms
try:
    import pyswarms as ps
except ImportError:
    try:
        subprocess.call([sys.executable, "-m", "pip", "install", 'pyswarms'])
    except:
        raise Exception("Tried importing 'pyswarms', then tried installing it. Please install it manually: pip install pyswarms")  
finally:
    import pyswarms as ps
    from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

def particleswarm(objfcn, dimensions, args = {}, use_bnds = True, bounds = (None,None), 
                  iters = 100, n_particles = 10, ftol = -np.inf,
                  options = {'c1': 0.5, 'c2': 0.3, 'w':0.9},
                  verbosity = 1,
                  **kwargs):
    
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
    
  
    
                 
    