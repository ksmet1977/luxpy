# -*- coding: utf-8 -*-
"""
###############################################################################
# Module with optimizers
###############################################################################

# minimizebnd(): scipy.minimize() that allows contrained parameters onn otherwise unconstrained methods. 

Created on Fri Apr 27 12:36:03 2018

@author: kevin.smet

"""

from luxpy import np, minimize

def minimzebnd(fun, x0, args=(), method = 'nelder-mead', use_bnd = True, bounds = (None,None) , options = None, **kwargs):
    
    
    if use_bnd == False:
        return minimize(fun, x0, options = options, **kwargs)
    else:
        
        LB, UB = bounds
        
        #size checks
        xsize = x0.shape
        x0 = x0.flatten()
        n = x0.shape[0]
        
        if LB is None:
          LB = -np.inf*np.ones(n)
        else:
          LB = LB.flatten()
    
        if UB is None:
          UB = np.inf*np.ones(n)
        else:
          UB = UB.flatten()
        
        if (n!=LB.shape[0]) | (n!=UB.shape[0]):
          raise Exception('minimizebnd(): x0 is incompatible in size with either LB or UB.')
    
        
        #set default options if necessary
        if options is None:
          options = {}
        
        # stuff into a struct to pass around
        params = {}
        params['args'] = args
        params['LB'] = LB
        params['UB']= UB
        params['fun'] = fun
        params['n'] = n
        params['OutputFcn'] = None
        
    #    % 0 --> unconstrained variable
    #    % 1 --> lower bound only
    #    % 2 --> upper bound only
    #    % 3 --> dual finite bounds
    #    % 4 --> fixed variable
        params['BoundClass'] = np.zeros(n,1)
    
        for i in np.arange(n):
          k = np.isfinite(LB[i]) + 2*np.isfinite(UB[i])
          params['BoundClass'][i] = k
          if (k==3) & (LB[i]==UB[i]):
              params['BoundClass'][i] = 4
    
        # transform starting values into their unconstrained
        # surrogates. Check for infeasible starting guesses.
        x0u = x0
        k = 0
        for i in np.range(n):
            
            if params['BoundClass'][i] == 1:
                # lower bound only
                if x0[i] <= LB[i]:
                    # infeasible starting value. Use bound.
                    x0u[k] = 0
                else:
                    x0u[k] = np.sqrt(x0[i] - LB[i])
              
            elif params['BoundClass'][i] == 2:
                # upper bound only
                if x0[i] >= UB[i]:
                    # infeasible starting value. use bound.
                    x0u[k] = 0
                else:
                    x0u[k] = sqrt(UB[i] - x0[i])
            
            elif params['BoundClass'][i] == 2:
              # lower and upper bounds
              if x0[i] <= LB[i]:
                    # infeasible starting value
                    x0u[k] = -np.pi/2
              elif x0[i] >= UB[i]:
                    # infeasible starting value
                    x0u[k] = np.pi/2
              else:
                x0u[k] = 2*(x0[i] - LB[i])/(UB[i]-LB[i]) - 1
                # shift by 2*pi to avoid problems at zero in fminsearch
                #otherwise, the initial simplex is vanishingly small
                x0u[k] = 2*np.pi+np.asin(np.hstack((-1,np.hstack((1,x0u[k]).min()))).max())
                
            elif params['BoundClass'][i] == 0:
              # unconstrained variable. x0u(i) is set.
              x0u[k] = x0[i]
              
            if not (params['BoundClass'][i] == 4):
              # increment k
              k += k
            else:
              # fixed variable. drop it before fminsearch sees it.
              # k is not incremented for this variable.
              pass
    
        # if any of the unknowns were fixed, then we need to shorten x0u now.
        if k <= n:
            x0u = x0u[:k]
    
        # were all the variables fixed?
        if x0u.shape[0] == 0: 
            # All variables were fixed. quit immediately, setting the
            # appropriate parameters, then return.
              
            # undo the variable transformations into the original space
            x = xtransform(x0u,params)
              
            # final reshape
            x = reshape(x,xsize)
              
            # stuff fval with the final value
            fval = params['fun'](x, **params['args'])
              
            # minimize was not called
            output['success'] = False
              
            output['x'] = x
            output['iterations'] = 0
            output['funcount'] = 1
            output['algorithm'] = method
            output['message'] = 'All variables were held fixed by the applied bounds';
            output['status'] = 0
              
            # return with no call at all to fminsearch
            return output
      
    # Check for an outputfcn. If there is any, then substitute my
    # own wrapper function.
    # Use a nested function as the OutputFcn wrapper
    def outfun_wrapper(x,**kwargs):
        # we need to transform x first
        xtrans = xtransform(x,params)
        
        # then call the user supplied OutputFcn
        stop = params['OutputFcn'](xtrans, **kwargs)
        
        return stop
        
    if 'OutputFcn' in options:
        if options['OutputFcn'] is not None:
            params.OutputFcn = options.OutputFcn
            options['OutputFcn']  = outfun_wrapper

    
    # now we can call minimize, but with our own
    # intra-objective function.
    res = minimize(intrafun, x0u, args = params, method = method, options = options)
    #[xu,fval,exitflag,output] = fminsearch(@intrafun,x0u,options,params);
    
    # get function value:
    fval = intrafun(res['x'],params)
    
    # undo the variable transformations into the original space
    x = xtransform(res['x'], params)
    
    # final reshape
    x = x.reshape(x, xsize)
    
    res['fval'] = fval
    res['x'] = x #overwrite x in res to unconstrained format
    
    return res



# ======================================
# ========= begin subfunctions =========
# ======================================
def intrafun(x,params):
    
    # transform variables, then call original function
    
    # transform
    xtrans = xtransform(x, params)
    
    # and call fun
    fval = params['fun'](xtrans, params['args'])
    
    return fval

# ======================================
def xtransform(x,params):
    # converts unconstrained variables into their original domains
    
    xtrans = np.zeros((params['n']))
    
    # k allows some variables to be fixed, thus dropped from the optimization.
    k=1
    for i in np.arange(params['n']):
        if params['BoundClass'][i] == 1:
              # lower bound only
              xtrans[i] = params.LB[i] + x[k]**2

        elif params['BoundClass'][i] == 2:
              # upper bound only
              xtrans[i] = params.UB[i] - x[k]**2
     
        elif params['BoundClass'][i] == 3:
              # lower and upper bounds
              xtrans[i]  = (np.sin(x[k] )+1)/2;
              xtrans[i]  = xtrans[i] * (params['UB'][i]  - params['LB'][i] ) + params['LB'][i] 
              
              # just in case of any floating point problems
              xtrans[i] = np.hstack((params['LB'][i],np.hstack((params['UB'][i],xtrans[i])).min())).max()
          
        elif params['BoundClass'][i] == 4:
              # fixed variable, bounds are equal, set it at either bound
              xtrans[i] = params['LB'][i]
        
        elif params['BoundClass'][i] == 0:
              # unconstrained variable.
              xtrans[i] = x[k]
          
        if params['BoundClass'][i] != 4:
            k += 1

    return xtrans