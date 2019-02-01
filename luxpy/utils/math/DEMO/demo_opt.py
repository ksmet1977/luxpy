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
Module for demo_opt
==================================================================

 :demo_opt(): | Multi-objective optimization using the DEMO.
              | This function uses the Differential Evolution for Multi-objective 
              | Optimization (a.k.a. DEMO) to solve a multi-objective problem. The 
              | result is a set of nondominated points that are (hopefully) very close
              | to the true Pareto front.

 :fobjeval(): | Evaluates the objective function.

 :mutation(): | Performs mutation in the individuals.
 
 :recombination(): | Performs recombination in the individuals.
 
 :repair(): | Truncates the population to be in the feasible region.
 
 :selection(): | Selects the next population.
 
 :init_options(): | Initialize options dict.
 
 :ndset(): | Finds the nondominated set of a set of objective points.
 
 :crowdingdistance(): Computes the crowding distance of a nondominated front.

 :dtlz2():  | DTLZ2 problem: This function represents a hyper-sphere.
            | Using k = 10, the number of dimensions must be n = (M - 1) + k.
            | The Pareto optimal solutions are obtained when the last k variables of x
            | are equal to 0.5.
            
 :dtlz_range(): | Returns the decision range of a DTLZ function
                 | The range is simply [0,1] for all variables. What varies is the number 
                 | of decision variables in each problem. The equation for that is
                 | n = (M-1) + k
                 | wherein k = 5 for DTLZ1, 10 for DTLZ2-6, and 20 for DTLZ7.

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""

from luxpy import np, plt, Axes3D, put_args_in_db, getdata

if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

__all__ = ['demo_opt', 'fobjeval','mutation','recombination','repair','selection','init_options','ndset','crowdingdistance','dtlz2','dtlz_range']

def demo_opt(f, args = (), xrange = None, options = {}):
    """
    DEMO_OPT: Multi-objective optimization using the DEMO
    This function uses the Differential Evolution for Multi-objective 
    Optimization (a.k.a. DEMO) to solve a multi-objective problem. The 
    result is a set of nondominated points that are (hopefully) very close
    to the true Pareto front.

    Args:
      :f: 
          | handle to objective function.
          | The output must be, for each point, a column vector of size m x 1, 
          | with m > 1 the number of objectives.
      :args: (), optional
          | Input arguments required for f.
      :xrange: None, optional
          | ndarray with lower and upperbounds.
          | If n is the dimension, it will be a n x 2 matrix, such that the 
          | first column contains the lower bounds, 
          | and the second, the upper bounds.
          | None defaults to no bounds ( [-Inf, Inf] ndarray).
      :options: 
          | None, optional
          | dict with internal parameters of the algorithm.
          | None initializes default values.
          | keys:
          | - 'F': the scale factor to be used in the mutation (default: 0.5);
          | - 'CR': the crossover factor used in the recombination (def.: 0.3);
          | - 'mu': the population size (number of individuals) (def.: 100);
          | - 'kmax': maximum number of iterations (def.: 300);
          | - 'display': 'on' to display the population while the algorithm is
          |         being executed, and 'off' to not (default: 'off');
          | If any of the parameters is not set, the default ones are used
          | instead.

    Returns: fopt, xopt
          :fopt: the m x mu_opt ndarray with the mu_opt best objectives
          :xopt: the n x mu_opt ndarray with the mu_opt best individuals
    """

    # Initialize the parameters of the algorithm with values contained in dict:
    options = init_options(options = options)

    
    # Initial considerations
    n = xrange.shape[0] #dimension of the problem
    P = {'x' :  np.random.rand(n, options['mu'])} #initial decision variables
    P['f'] = fobjeval(f, P['x'], args, xrange) #evaluates the initial population
    m = P['f'].shape[0] #number of objectives
    k = 0 #iterations counter


    # Beginning of the main loop
    Pfirst = P.copy()
    axh = None
    while (k <= options['kmax']):
       # Plot the current population (if desired):
       if options['display'] == True:
           if (k == 0) & (m < 4):
               fig = plt.gcf()
               fig.show()
               fig.canvas.draw()
           if m == 2:
#              fig = plt.figure()
              axh = plt.axes()
              plt.plot(P['f'][0], P['f'][1], 'o');
              plt.title('Objective values during the execution')
              plt.xlabel('f_1')
              plt.ylabel('f_2')
              fig.canvas.draw()
              del axh.lines[0]
           elif m == 3:
#              fig = plt.figure()
              axh = plt.axes(projection='3d')
#              print(P['f'])
              axh.plot3D(P['f'][0], P['f'][1], P['f'][2], 'o');
              plt.title('Objective values during the execution')
              axh.set_xlabel('f_1')
              axh.set_ylabel('f_2')
              axh.set_zlabel('f_3')
              fig.canvas.draw()
              plt.pause(0.01)
              del axh.lines[0]
 
     
       # Perform the variation operation (mutation and recombination):
       O = {'x': mutation(P['x'], options)} #mutation
       O['x'] = recombination(P['x'].copy(), O['x'], options) #recombination
       O['x'] = repair(O['x']) #assure the offspring do not cross the search limits
       O['f'] = fobjeval(f, O['x'], args, xrange) #compute objective functions
       
       # Selection and updates
       P = selection(P, O, options)

       print('Iteration #{:1.0f} of {:1.0f}'.format(k, options['kmax']))
       k += 1


    # Return the final population:
    # First, unnormalize it
    Xmin = xrange[:,0][:,None]# * np.ones(options['mu']) #replicate lower bound
    Xmax = xrange[:,1][:,None]# * np.ones(options['mu']) #replicate upper limit
    Xun = (Xmax - Xmin)*P['x'] + Xmin
    
    # Then, return the nondominated front:
    ispar = ndset(P['f'])
    fopt = P['f'][:,ispar]
    xopt = Xun[:,ispar]
    
    return fopt, xopt

#=========================== Sub-functions ================================#
def fobjeval(f, x, args, xrange):
    """
    Evaluates the objective function.
    Since the population is normalized, this function unnormalizes it and
    computes the objective values.

    Args:
       :f: 
           | handle to objective function.
           | The output must be, for each point, a column vector of size m x 1, 
           | with m > 1 the number of objectives.
       :x: 
           | a n x mu ndarray with mu individuals (points) and n variables 
           | (dimension size)
       :args: 
          | Input arguments required for f.
       :options:
           | the dict with the parameters of the algorithm.

    Returns:
       :phi: 
           | a m x mu ndarray with the m objective values of the mu
           | individuals
    """
    
    mu = x.shape[1] #number of points
    # Unnormalizes the population:
    Xmin = xrange[:,0][:,None]# * np.ones(options['mu']) #replicate lower bound
    Xmax = xrange[:,1][:,None]# * np.ones(options['mu']) #replicate upper limit
    Xun = (Xmax - Xmin)*x + Xmin
    
    if bool(())==False:
        phi = f(Xun)
    else:
        phi = f(Xun, **args)

    return phi
#--------------------------------------------------------------------------#
def mutation(Xp, options):
    """
    Performs mutation in the individuals.
    The mutation is one of the operators responsible for random changes in
    the individuals. Each parent x will have a new individual, called trial
    vector u, after the mutation.
    To do that, pick up two random individuals from the population, x2 and
    x3, and creates a difference vector v = x2 - x3. Then, chooses another
    point, called base vector, xb, and creates the trial vector by

       u = xb + F*v = xb + F*(x2 - x3)

    wherein F is an internal parameter, called scale factor.

    Args:
       :Xp: 
           | a n x mu ndarray with mu "parents" and of dimension n
       :options: 
           | the dict with the internal parameters

    Returns:
       :Xo: 
           | a n x mu ndarray with the mu mutated individuals (of dimension n)
    """
    # Creates a mu x mu matrix of 1:n elements on each row
    A = np.arange(options['mu']).repeat(options['mu']).reshape(options['mu'],options['mu']).T   

    # Now, one removes the diagonal of A, because it contains indexes that repeat 
    # the current i-th individual
    A = np.reshape(A[(np.eye(A.shape[0]))==False],(options['mu'],options['mu']-1))

    # Now, creates a matrix that permutes the elements of A randomly
    J = np.argsort(np.random.rand(*A.shape), axis = 1)
#    J = getdata('J.txt')-1

    Ilin = J*options['mu'] + np.arange(options['mu'])[:,None]
    A = A.T.flatten()[Ilin].reshape(A.shape)
    
    # Chooses three random points (for each row)
    xbase = Xp[:, A[:,0]] #base vectors
    v = Xp[:, A[:,1]] - Xp[:, A[:,2]] #difference vector
    
    # Performs the mutation
    Xo = xbase + options['F']*v

    return Xo
#--------------------------------------------------------------------------#
def recombination(Xp, Xm, options):
    """
    Performs recombination in the individuals.
    The recombination combines the information of the parents and the
    mutated individuals (also called "trial vectors") to create the
    offspring. Assuming x represents the i-th parent, and u the i-th trial
    vector (obtained from the mutation), the offspring xo will have the
    following j-th coordinate: xo_j = u_j if rand_j <= CR, x_j otherwise
    wherein rand_j is a number drawn from a uniform distribution from 0 to
    1, and CR is called the crossover factor. To prevent mere copies, at
    least one coordinate is guaranteed to belong to the trial vector.

   Args:
      :Xp: 
          | a n x mu ndarray with the mu parents
      :Xm: 
          | a n x mu ndarray with the mu mutated points
      :options: 
          | the dict with the internal parameters

   Returns:
      Xo: 
          | a n x mu ndarray with the recombinated points (offspring)
   """
    # Draws random numbers and checks whether they are smaller or
    # greater than CR:
    n = Xp.shape[0] #dimension of the problem
    aux = np.random.rand(n, options['mu']) <= options['CR']
    
    # Now assures at least one coordinate will be changed, that is,
    # there is at least one 'true' in each column
    auxs = aux.sum(axis=0) == 0 #gets the columns with no trues
    indc = np.where(auxs)[0] #get the number of the columns
    indr = np.random.randint(0, n, auxs.sum()) #define random indexes of rows
    
    #ind = np.ravel_multi_index((indr,indc),(n, options['mu'])) #converts to lin. indexes
    aux[indr,indc] = True
    
    # Finally, creates the offspring
    Xo = Xp
    Xo[aux] = Xm[aux]
    return Xo

#--------------------------------------------------------------------------#
def repair(Xo):
    """
    Truncates the population to be in the feasible region.
    """
    # This is easy, because the population must be inside the interval [0, 1]
    Xo = np.clip(Xo,0,1) #corrects lower and upper limit
    return Xo

#--------------------------------------------------------------------------#
def selection(P, O, options):
    """
    Selects the next population.
    Each parent is compared to its offspring. If the parent dominates its 
    child, then it goes to the next population. If the offspring dominates 
    the parent, that new member is added. However, if they are incomparable
    (there is no mutual domination), them both are sent to the next 
    population. After that, the new set of individuals must be truncated to 
    mu, wherein mu is the original number of points.
    This is accomplished by the use of "non-dominated sorting", that is,
    ranks the individual in fronts of non-domination, and within each
    front, measures them by using crowding distance. With regard to these
    two metrics, the best individuals are kept in the new population.

   Args:
      :P: 
          | a dict with the parents (x and f)
      :O: 
          | a dict with the offspring
      :options: 
          | the dict with the algorithm's parameters

   Returns:
      :Pnew: 
          | the new population (a dict with x and f)
   """
   
    # ------ First part: checks dominance between parents and offspring
    # Verifies whether parent dominates offspring:
    aux1 = (P['f'] <= O['f']).all(axis = 0)
    aux2 = (P['f'] < O['f']).any(axis = 0)
    auxp = np.logical_and(aux1, aux2) #P dominates O
    
    # Now, where offspring dominates parent:
    aux1 = (P['f'] >= O['f']).all(axis = 0)
    aux2 = (P['f'] > O['f']).any(axis = 0)
    auxo = np.logical_and(aux1, aux2) #O dominates P
    auxpo = np.logical_and(~auxp, ~auxo); #P and O are incomparable

    # New population (where P dominates O, O dominates P and where they are 
    # incomparable)
    R = {'f' : np.hstack((P['f'][:,auxp].copy(), O['f'][:,auxo].copy(), P['f'][:,auxpo].copy(), O['f'][:,auxpo].copy()))}
    R['x'] = np.hstack((P['x'][:,auxp].copy(), O['x'][:,auxo].copy(), P['x'][:,auxpo].copy(), O['x'][:,auxpo].copy()))
    
    # ------- Second part: non-dominated sorting
    Pnew = {'x' : np.atleast_2d([])} 
    Pnew['f'] = np.atleast_2d([]) #prepares the new population
    while True:
       ispar = ndset(R['f']) #gets the non-dominated front

       # If the number of points in this front plus the current size of the new
       # population is smaller than mu, then include everything and keep going.
       # If it is greater, then stops and go to the truncation step:
       if ((Pnew['f'].shape[1] + ispar.sum()) < options['mu']):

          Pnew['f'] = np.hstack((Pnew['f'], R['f'][:,ispar].copy())) if (Pnew['f'].size) else R['f'][:,ispar].copy()
          Pnew['x'] = np.hstack((Pnew['x'], R['x'][:,ispar].copy())) if (Pnew['x'].size) else R['x'][:,ispar].copy()
          R['f'] = np.delete(R['f'],ispar, axis = 1) #R['f'][:,ispar] = []; #removes this front
          R['x'] = np.delete(R['x'],ispar, axis = 1) #R['x'][:,ispar] = []; #removes this front
       else:
          # Gets the points of this front and goes to the truncation part
          Frem = R['f'][:,ispar].copy()
          Xrem = R['x'][:,ispar].copy()
          break #don't forget this to stop this infinite loop
    
    # ------- Third part: truncates using crowding distance
    # If the remaining front has the exact number of points to fill the original
    # size, then just include them. If it has too many, remove some according to
    # the crowding distance (notice it cannot have too few!)
    aux = (Pnew['f'].shape[1] + Frem.shape[1]) - options['mu'] #remaining points to fill
    if aux == 0:
       Pnew['x'] = np.hstack((Pnew['x'], Xrem.copy())) 
       Pnew['f'] = np.hstack((Pnew['f'], Frem.copy()))
    elif aux > 0:
       for ii in range(aux):
          cdist = crowdingdistance(Frem)
          imin = cdist.argmin()#gets the point with smaller crowding distance
          Frem = np.delete(Frem, imin, axis = 1) # Frem(:,imin) = []; #and remove it
          Xrem = np.delete(Xrem, imin, axis = 1)  # Xrem(:,imin) = [];
       Pnew['x'] =  np.hstack((Pnew['x'], Xrem.copy())) if Pnew['x'].size else Xrem.copy()
       Pnew['f'] =  np.hstack((Pnew['f'], Frem.copy())) if Pnew['f'].size else Frem.copy()
    else: #if there are too few points... well, we're doomed!
       raise Exception('Run to the hills! This is not supposed to happen!')

    return Pnew
#--------------------------------------------------------------------------#
def init_options(options = {}, F = None, CR = None, kmax = None, mu = None, display = None):
    """
    Initialize options dict.
    If input arg is None, the default value is used. 
    
    Args:
        :options: {}, optional
         | Dict with options
         | {} initializes dict to default values.
        :F: scale factor, optional
        :CR: crossover factor, optional
        :kmax: maximum number of iterations, optional
        :mu: population size, optional
        :display: show or not the population during execution, optional
        
    Returns:
        :options: dict with options.
    """
    args = locals().copy()
    if bool(options)==False:
        options = {'F': 0.5, 'CR' : 0.3, 'kmax' : 300, 'mu' : 100, 'display' : False}
    return put_args_in_db(options, args)

#--------------------------------------------------------------------------#
def ndset(F):
    """
    Finds the nondominated set of a set of objective points.

    Args:
      F: 
          | a m x mu ndarray with mu points and m objectives

   Returns:
      :ispar: 
          | a mu-length vector with true in the nondominated points
    """
    mu = F.shape[1] #number of points

    # The idea is to compare each point with the other ones
    f1 = np.transpose(F[...,None], axes = [0, 2, 1]) #puts in the 3D direction
    f1 = np.repeat(f1,mu,axis=1)
    f2 = np.repeat(F[...,None],mu,axis=2)

    # Now, for the ii-th slice, the ii-th individual is compared with all of the
    # others at once. Then, the usual operations of domination are checked
    # Checks where f1 dominates f2
    aux1 = (f1 <= f2).all(axis = 0, keepdims = True)
    aux2 = (f1 < f2).any(axis = 0, keepdims = True)

    auxf1 = np.logical_and(aux1, aux2)
    # Checks where f1 is dominated by f2
    aux1 = (f1 >= f2).all(axis = 0, keepdims = True)
    aux2 = (f1 > f2).any(axis = 0, keepdims = True)
    auxf2 = np.logical_and(aux1, aux2)
    
    # dom will be a 3D matrix (1 x mu x mu) such that, for the ii-th slice, it
    # will contain +1 if fii dominates the current point, -1 if it is dominated 
    # by it, and 0 if they are incomparable
    dom = np.zeros((1, mu, mu), dtype = int)

    dom[auxf1] = 1
    dom[auxf2] = -1
    
    # Finally, the slices with no -1 are nondominated
    ispar = (dom != -1).all(axis = 1)
    ispar = ispar.flatten()
    return ispar

#--------------------------------------------------------------------------#
def crowdingdistance(F):
    """
    Computes the crowding distance of a nondominated front.
    The crowding distance gives a measure of how close the individuals are
    with regard to its neighbors. The higher this value, the greater the
    spacing. This is used to promote better diversity in the population.

    Args:
       F: 
           | an m x mu ndarray with mu individuals and m objectives

    Returns:
       cdist: 
           | a m-length column vector
    """
    m, mu = F.shape #gets the size of F
    
    if mu == 2:
       cdist = np.vstack((np.inf, np.inf))
       return cdist

    
    #[Fs, Is] = sort(F,2); #sorts the objectives by individuals
    Is = F.argsort(axis = 1)
    Fs = np.sort(F,axis=1)
    
    # Creates the numerator
    C = Fs[:,2:] - Fs[:,:-2]
    C = np.hstack((np.inf*np.ones((m,1)), C, np.inf*np.ones((m,1)))) #complements with inf in the extremes
    
    # Indexing to permute the C matrix in the right ordering
    Aux = np.arange(m).repeat(mu).reshape(m,mu)   
    ind = np.ravel_multi_index((Aux.flatten(),Is.flatten()),(m, mu)) #converts to lin. indexes # ind = sub2ind([m, mu], Aux(:), Is(:));
    C2 = C.flatten().copy()
    C2[ind] = C2.flatten()
    C = C2.reshape((m, mu))

    # Constructs the denominator
    den = np.repeat((Fs[:,-1] - Fs[:,0])[:,None], mu, axis = 1)
    
    # Calculates the crowding distance
    cdist = (C/den).sum(axis=0)
    cdist = cdist.flatten() #assures a column vector
    return cdist


# FOR EXAMPLE: demo_opt()
def dtlz2(x, M):
    """
    DTLZ2 multi-objective function
    This function represents a hyper-sphere.
    Using k = 10, the number of dimensions must be n = (M - 1) + k.
    The Pareto optimal solutions are obtained when the last k variables of x
    are equal to 0.5.
    
    Args:
        :x: 
            | a n x mu ndarray with mu points and n dimensions
        :M: 
            | a scalar with the number of objectives
    
       Returns:
          f: 
            | a m x mu ndarray with mu points and their m objectives computed at
            | the input
    """
    k = 10
    
    # Error check: the number of dimensions must be M-1+k
    n = (M-1) + k; #this is the default
    if x.shape[0] != n:
       raise Exception('Using k = 10, it is required that the number of dimensions be n = (M - 1) + k = {:1.0f} in this case.'.format(n))
    
    xm = x[(n-k):,:].copy() #xm contains the last k variables
    g = ((xm - 0.5)**2).sum(axis = 0)
    
    # Computes the functions:
    f = np.empty((M,x.shape[1]))
    f[0,:] = (1 + g)*np.prod(np.cos(np.pi/2*x[:(M-1),:]), axis = 0)
    for ii in range(1,M-1):
        f[ii,:] = (1 + g) * np.prod(np.cos(np.pi/2*x[:(M-ii-1),:]), axis = 0) * np.sin(np.pi/2*x[M-ii-1,:])
    f[M-1,:] = (1 + g) * np.sin(np.pi/2*x[0,:])
    return f

def dtlz_range(fname, M):
    """
    Returns the decision range of a DTLZ function
    The range is simply [0,1] for all variables. What varies is the number 
    of decision variables in each problem. The equation for that is
    n = (M-1) + k
    wherein k = 5 for DTLZ1, 10 for DTLZ2-6, and 20 for DTLZ7.
    
    Args:
        :fname: 
            | a string with the name of the function ('dtlz1', 'dtlz2' etc.)
        :M: 
            | a scalar with the number of objectives
    
       Returns:
          :lim: 
              | a n x 2 matrix wherein the first column is the lower limit 
               (0), and the second column, the upper limit of search (1)
    """
     #Checks if the string has or not the prefix 'dtlz', or if the number later
     #is greater than 7:
    fname = fname.lower()
    if (len(fname) < 5) or (fname[:4] != 'dtlz') or (float(fname[4]) > 7) :
       raise Exception('Sorry, the function {:s} is not implemented.'.format(fname))


    # If the name is o.k., defines the value of k
    if fname ==  'dtlz1':
       k = 5
    elif fname == 'dtlz7':
       k = 20
    else: #any other function
       k = 10;

    
    n = (M-1) + k #number of decision variables
    
    lim = np.hstack((np.zeros((n,1)), np.ones((n,1))))
    return lim


if __name__ == '__main__':
    # EXAMPLE USE for DTLZ2 problem:
    k = 10
    opts = init_options(display = True)
    f = lambda x: dtlz2(x,k)
    xrange = dtlz_range('dtlz2',k)
    
    fopt, xopt = demo_opt(f, xrange = xrange, options = opts)
    
    mu = xopt.shape[1]
    xlast = 0.5*np.ones((k, mu))
    d = ((xopt[(-1-k):-1,:] - xlast)**2).sum(axis=0)
    print('min(d): {:1.3f}'.format(d.min()))
    print('mean(d): {:1.3f}'.format(d.mean()))
    print('max(d): {:1.3f}'.format(d.max()))
    