# -*- coding: utf-8 -*-
"""
Wrapper for pymoo's NSGA-II class based optimizer.
==================================================

 :nsga_ii(): pareto multi-objective optimization using NSGA-II genetic algorithm.

Notes:
------

 * An import will try and install the pymoo package using pip install. 

    
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

# Import modules
import copy

from luxpy.utils import is_importable 


# import pymoo (and if necessary install it):
success = is_importable('pymoo', try_pip_install = True)
if success:
    import pymoo as pm
    
    from pymoo.model.problem import Problem
    from pymoo.algorithms.nsga2 import NSGA2
    from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
    from pymoo.model.problem import ConstraintsAsPenaltyProblem
    
__all__ = ['nsga_ii']


class PymooProblem(Problem):
    """
    Problem class for pymoo's NSGA2 algorithm.
    """
    def __init__(self, 
                 n_var, 
                 n_obj, 
                 fitnessfcn, 
                 fitnessfcn_kwargs_dict = {'obj_fcn' : []}, 
                 bounds = (None,None)):

        super().__init__(n_var = n_var, n_obj = n_obj, xl = bounds[0], xu = bounds[1])

        self.n_var = n_var
        self.n_obj = n_obj
        self.fitnessfcn = fitnessfcn
        self.fitnessfcn_kwargs_dict = fitnessfcn_kwargs_dict
       
        
    def get_obj_vals(self, x, out = None):
        fitnessfcn_kwargs_dict_ = copy.deepcopy(self.fitnessfcn_kwargs_dict)
        if out is not None: # update out upon request
            fitnessfcn_kwargs_dict_['out'] = out
        return  self.fitnessfcn(x, **fitnessfcn_kwargs_dict_)


    def _evaluate(self, x, out, *args, **kwargs):
        F = self.get_obj_vals(x, out = 'F')
        out["F"] = F



def nsga_ii(fitnessfcn, n_variables, n_objectives, args = {}, use_bnds = True, bounds = (None,None), 
            verbosity = 1,
            pm_n_gen = 40, pm_n_pop = 400, pm_n_offsprings = None,
            pm_options = {'sampling'    : ("real_random",{}),
                          'crossover'  : ("real_sbx", {'prob' : 0.9, 'eta' : 15}),
                          'mutation'    : ("real_pm", {'eta' : 20})},
            pm_termination = ('n_gen' , 40),
            pm_eliminate_duplicates = True,
            pm_return_least_infeasible = False,
            pm_save_history = False,
            pm_seed = 1,
            pm_algorithm = None,
            **pm_kwargs):
    """
    Pareto multi-objective minimization function using NSGA-II (NSGA2) from pymoo.
    
    Args:
        :fitnessfcn:
            | fitness function
            | Should output a N x n_obj matrix with fitness values for each of the N individuals.
        :n_variables: 
            | Number of parameter values for the objective function. 
        :n_objectives: 
            | Number of objectives optimized by fitness function. 
        :args:
            | {}
            | Dict with fitnessfcn input parameters (except the to be optimized x)
        :use_bnd:
            | True, optional
            | False: omits bounds and defaults to regular minimize function.
        :bounds:
            | (lower, upper), optional
            | Tuple of lists or dicts (x0_keys is None) of lower and upper bounds 
              for each of the parameters values.
        :verbosity:
            | 1, optional
            | Print some intermediate outputs
        :pm_n_gen:
            | 40, optional
            | Number of generations
        :pm_n_pop:
            | 400, optional
            | Population size
        :pm_n_offsprings:
            | None, optional
            | Number of offspring to keep after mating.
            | None defaults to n_pop.
        :pm_options:
            | dict, optional
            | A dictionary containing the parameters for the specific
            | optimization algorithm.
            |  - 'sampling' : ("real_random",{}),
            |  - 'crossover': ("real_sbx", ({'prob' : 0.9, 'eta' : 15}),
            |  - 'mutation' :  ("real_pm", {'eta' : 20}),
        :pm_termination:
             | ('n_gen', 40), optional
        :pm_eliminate_duplicates:
            | True, optional
        :pm_return_least_infeasible
            | False, optional
        :pm_save_history
            | False, optional
        :pm_seed:
            | 1, optional
        :pm_kwargs:
            | Additional pymoo keyword arguments for algorithm.setup()
  
    Note:
        For more info on the pymoo_specific input arguments, see pymoo documentation
         
    Returns:
        :res: 
            | dict with output of minimization:
            | keys():
            |   - 'x_final': final solution x
            |   - 'F': final function value of obj_fcn()
            |   - and some of the input arguments characterizing the 
            |       minimization, such as n_pop, n_gen, bounds, options, etc.

    """
       
    # Set up algorithm:
    if pm_algorithm is None:
        algorithm_ = NSGA2(pop_size = pm_n_pop,
                           n_offsprings = pm_n_offsprings,
                           sampling = get_sampling(pm_options['sampling'][0],
                                                    **pm_options['sampling'][1]),
                           crossover = get_crossover(pm_options['crossover'][0],
                                                      **pm_options['crossover'][1]),
                           mutation = get_mutation(pm_options['mutation'][0],
                                                      **pm_options['mutation'][1]),
                           eliminate_duplicates = pm_eliminate_duplicates)
    else:
        algorithm_ = pm_algorithm
        
    # Setup termination:
    termination_ = get_termination(pm_termination[0], pm_termination[1])
    
    # initialize problem:
    problem_ = PymooProblem(n_variables,
                            n_objectives,
                            fitnessfcn, 
                            fitnessfcn_kwargs_dict = args, 
                            bounds = bounds)
        
    # perform a copy of the algorithm to ensure reproducibility:
    obj = copy.deepcopy(algorithm_)
    
    # let the algorithm know what problem we are intending to solve and provide other attributes
    obj.setup(problem_, 
              termination = termination_, 
              seed = pm_seed,
              return_least_infeasible = pm_return_least_infeasible,
              save_history = pm_save_history,
              **pm_kwargs)

    # until the termination criterion has not been met
    while obj.has_next():
    
        # perform an iteration of the algorithm
        obj.next()
    
        # access the algorithm to print some intermediate outputs
        if verbosity > 0:
            print(f"gen: {obj.n_gen} n_nds: {len(obj.opt)} constr: {obj.opt.get('CV').min()} ideal: {obj.opt.get('F').min(axis=0)}")

    
    # finally obtain the result object
    res = obj.result()
    
    # create output dictionary: 
    result = {'x_final' : res.X,
              'F' : res.F,
              'bounds' : bounds, 
              'options' : pm_options, 
              'res': res, 
              'obj': obj}
  
    return result

                 
    