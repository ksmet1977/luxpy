# -*- coding: utf-8 -*-
"""
   Code for paired permanova.
   
   The code below is based on and uses the skbio package. 
   (Copyright (c) 2013--, scikit-bio development team.)
   
   
.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from luxpy.utils import is_importable

import itertools
from functools import partial
import numpy as np

# import scipy # has become lazy import
# from scipy.stats import f_oneway # has become lazy import
# from scipy.spatial.distance import cdist # has become lazy import

# Try importing, if necessary pip-installing, hdmedian and skbio packages that
# have not been 'required' by luxpy to reduce dependencies for rarely used modules.
success = is_importable('hdmedians', try_pip_install = True)
if success:
    import hdmedians as hd
success = is_importable('skbio', try_pip_install = True)
if success:
    import skbio
    from skbio.stats.ordination import pcoa
    from skbio.stats.distance._base import _preprocess_input

__all__ = ['run_permanova_permdisp', 'permanova', 'permdisp']

def _compute_s_W_S(sample_size, num_groups, tri_idxs, distances, group_sizes, grouping, subjects, paired):
    """Compute PERMANOVA Within & Subjects Sum-of-Squares."""
    
    # Create a matrix where objects in the same group are marked with the group
    # index (e.g. 0, 1, 2, etc.). objects that are not in the same group are
    # marked with -1. If paired == True: Do similar for test subjects:
    grouping_matrix = -1 * np.ones((sample_size, sample_size), dtype= np.int32)
    for group_idx in range(num_groups):
        within_indices = _index_combinations(np.where(grouping == group_idx)[0])
        grouping_matrix[within_indices] = group_idx 

    # Extract upper triangle (in same order as distances were extracted
    # from full distance matrix).
    grouping_tri = grouping_matrix[tri_idxs]

    # Calculate s_WG for each group, accounting for different group sizes.
    s_WG = 0
    for i in range(num_groups):
        s_WG += (distances[grouping_tri == i] ** 2).sum() / group_sizes[i]
        
    # for pseudo-F2: Calculate s_WG_V for each group, accounting for different group sizes.
    s_WG_V = 0
    for i in range(num_groups):
        s_WG_V += (1-group_sizes[i]/sample_size)*((1/(group_sizes[i]*(group_sizes[i] - 1)))*distances[grouping_tri == i] ** 2).sum()

    if paired == True:
        num_subjects = sample_size//num_groups
        subjects_matrix = -1 * np.ones((sample_size, sample_size), dtype= np.int32)
        for subject_idx in range(num_subjects):
            subject_indices = _index_combinations(np.where(subjects == subject_idx)[0])
            subjects_matrix[subject_indices] = subject_idx  

        # Extract upper triangle (in same order as distances were extracted
        # from full distance matrix).
        subjects_tri = subjects_matrix[tri_idxs]
    
        # Calculate s_WS for each subject, accounting for number of groups.
        s_WS = 0
        for i in range(num_subjects):
            s_WS += (distances[subjects_tri == i] ** 2).sum() / num_groups
            
    else:
        s_WS = 0

    return s_WG, s_WS, s_WG_V
    

def _compute_f_stat(sample_size, num_groups, tri_idxs, distances, group_sizes,
                    s_T, grouping, subjects, paired):
    """Compute PERMANOVA Pseudo-F."""
    s_WG, s_WS, s_WG_V = _compute_s_W_S(sample_size, num_groups, tri_idxs, distances, group_sizes, grouping, subjects, paired)
    
    # for pseudo-F1:
    s_BG = s_T - s_WG # = s_Effect
    dfBG = (num_groups - 1)
    
    if (paired == True):
        s_BS = s_T - s_WS
        s_Error = s_WS - s_BG 
        dfErr = (num_groups - 1)*(len(np.unique(subjects)) - 1)
        if np.isclose(s_Error,0,atol=1e-9):
            s_Error = np.abs(s_Error)
        if (s_Error < 0):
            print('WARNING: s_Error = {:1.4f} < 0!'.format(s_Error))
            print('         s_BG = {:1.4f}, s_WG = {:1.4f}, s_BS = {:1.4f}, s_WS = {:1.4f}.'.format(s_BG, s_WG, s_BS, s_WS))
            print('         Setting s_Error to s_WGB (s_S -> 0) (cfr. paired = False)!')
            s_Error = s_WG
            s_BS = np.nan
            dfErr = (sample_size - num_groups)

    else:
        s_Error = s_WG # for pseudo-F1
        s_Error2 = s_WG_V # for pseudo-F2
        s_BS = np.nan
        dfErr = (sample_size - num_groups)
        
    # test statistic, pseudo-F1:
    stat_ = (s_BG / dfBG) / (s_Error / dfErr)    
    
    if paired == True:
        # test statistic, pseudo-F2 (equals pseudo-F1 for equal sample sizes!):
        stat = stat_
    else:
        # test statistic, pseudo-F2:
        stat = (s_BG) / (s_Error2) 
    
    # effect sizes:
    p_eta2 = s_BG/(s_BG + s_Error)
    omega2 = (s_BG - dfBG*(s_Error / dfErr))/(s_T - (s_Error / dfErr))
    R2 = 1.0 - 1 / (1 + stat * (dfBG / dfErr))   
    #print('t:',sample_size, num_groups, (sample_size - num_groups - 1))
    R2adj = 1.0 - ((1-R2)*(sample_size - 1)/(sample_size - num_groups - 1))
    effect_sizes = {'p_eta2': p_eta2, 'omega2':omega2, 'R2': R2, 'R2adj':R2adj}
 
#    print('s_BG = {:1.2f}, s_WG = {:1.2f}, s_BS = {:1.2f}, s_WS = {:1.2f}, s_Err = {:1.2f} -- > s_T = {:1.2f}(Sum={:1.2f}:{:1.2f}).'.format(s_BG, s_WG, s_BS, s_WS, s_Error, s_T, s_BG + s_WG, s_BS + s_WS))
    
    
    if s_Error < 0:
        print('WARNING: s_Error = {:1.4f} <= 0!'.format(s_Error))
        print('         s_BG = {:1.4f}, s_WG = {:1.4f}, s_BS = {:1.4f}, s_WS = {:1.4f}.'.format(s_BG, s_WG, s_BS, s_WS))
        print('         Setting F to NaN.')
        stat = np.nan
    return stat, effect_sizes

    
def _permutate_grouping(grouping, subjects, paired = False):
    """ permutate grouping and subjects indexing arrays"""
    if paired == False:
        perm_idx = np.arange(grouping.shape[0],dtype= np.int32)
        perm_idx = np.random.permutation(perm_idx)
        perm_grouping = grouping[perm_idx]
        perm_subjects = subjects[perm_idx]
    else:
        groups = np.unique(grouping)
        o = 10**((grouping.reshape(len(groups),len(grouping)//len(groups)))+1)
        if subjects is not None:
            s = subjects.reshape(len(groups),len(grouping)//len(groups))
        for i in range(o.shape[-1]):
            perm_idx = np.arange(o.shape[0],dtype= np.int32)
            perm_idx = np.random.permutation(perm_idx)
            o[:,i] = o[perm_idx,i]
            s[:,i] = s[perm_idx,i]
        o = np.log10(o)-1
        perm_grouping = o.flatten()
        if subjects is not None:
            perm_subjects = s.flatten()
    return perm_grouping, perm_subjects       

def _index_combinations(indices):
    """ 
    Get index combinations
    
    Modified from http://stackoverflow.com/a/11144716
    """
    return np.tile(indices, len(indices)), np.repeat(indices, len(indices))        

def _run_monte_carlo_stats(test_stat_function, grouping, subjects, permutations, paired):
    """Run stat test and compute significance with Monte Carlo permutations."""
    if permutations < 0:
        raise ValueError(
            "Number of permutations must be greater than or equal to zero.")

    stat, effect_sizes = test_stat_function(grouping, subjects, paired)
    
    p_value = np.nan
    if permutations > 0:
        perm_stats = np.empty(permutations, dtype=np.float64)

        for i in range(permutations):
            perm_grouping, perm_subjects = _permutate_grouping(grouping, subjects, paired = paired)    
            perm_stats[i], _ = test_stat_function(perm_grouping, perm_subjects, paired)
        stat, effect_sizes = test_stat_function(grouping, subjects, paired)    
        p_value = ((perm_stats >= stat).sum() + 1) / (permutations + 1)

    return stat, p_value, effect_sizes

def _create_subjects_index_arr(subjects = None, grouping = None):
    """ Create subjects indexing array"""
    if subjects is None:
        if grouping is None:
            raise Exception('Grouping must be supplied!')
        groups = np.unique(grouping)
        for i,group in enumerate(groups):
            if i == 0:
                subjects = np.arange(((grouping==group)*1).sum())
            else:
                subjects = np.hstack((subjects,np.arange(((grouping==group)*1).sum())))  
    return subjects

def permanova(distance_matrix, grouping, column=None, permutations=999, paired = False, subjects = None):
    """Test for significant differences between groups using PERMANOVA.

    | Permutational Multivariate Analysis of Variance (PERMANOVA) is a
    | non-parametric method that tests whether two or more groups of objects
    | (e.g., samples) are significantly different based on a categorical factor.
    | It is conceptually similar to ANOVA except that it operates on a distance
    | matrix, which allows for multivariate analysis. PERMANOVA computes a
    | pseudo-F2 statistic.
    |
    | Statistical significance is assessed via a permutation test. The assignment
    | of objects to groups (`grouping`) is randomly permuted a number of times
    | (controlled via `permutations`). A pseudo-F2 statistic is computed for each
    | permutation and the p-value is the proportion of permuted pseudo-F2
    | statisics that are equal to or greater than the original (unpermuted)
    | pseudo-F2 statistic.

    Args:
    :distance_matrix : 
        | DistanceMatrix
        | Distance matrix containing distances between objects (e.g., distances
        | between samples of microbial communities).
    :grouping : 
        | 1-D array_like or pandas.DataFrame
        | Vector indicating the assignment of objects to groups. For example,
        | these could be strings or integers denoting which group an object
        | belongs to. If `grouping` is 1-D ``array_like``, it must be the same
        | length and in the same order as the objects in `distance_matrix`. If
        | `grouping` is a ``DataFrame``, the column specified by `column` will be
        | used as the grouping vector. The ``DataFrame`` must be indexed by the
        | IDs in `distance_matrix` (i.e., the row labels must be distance matrix
        | IDs), but the order of IDs between `distance_matrix` and the
        | ``DataFrame`` need not be the same. All IDs in the distance matrix must
        | be present in the ``DataFrame``. Extra IDs in the ``DataFrame`` are
        | allowed (they are ignored in the calculations).
    :column: 
        | str, optional
        |Column name to use as the grouping vector if `grouping` is a
        |``DataFrame``. Must be provided if `grouping` is a ``DataFrame``.
        |Cannot be provided if `grouping` is 1-D ``array_like``.
    :permutations: 
        | int, optional
        | Number of permutations to use when assessing statistical
        | significance. Must be greater than or equal to zero. If zero,
        | statistical significance calculations will be skipped and the p-value
        | will be ``np.nan``.
    :paired:
        | bool, optional
        | If True: limit the type of permutations, such that permutations happen 
        | only over groups, not over samples. SSwithin is then estimated and 
        | subtracted from the SS between when estimating F_pseudo.
    :subjects:
        | 1-D array_like with indices for subjects for use in paired permanova.
        | If None: array (0...ni) will be generated for each group (same size!).

    Returns:
        | pandas.Series
        | Results of the statistical test, including ``test statistic`` and
        | ``p-value``.

    Notes:
    | See [1]_ for the original method reference, as well as ``vegan::adonis``,
    | available in R's vegan package [2]_.
    |
    | The p-value will be ``np.nan`` if `permutations` is zero.
    |
    | Is based on and uses the skbio package (install manually: pip install skbio).
    |
    | Based on code for permanova and permdisp, but extended for repeated measures or paired data.
    | 
    | Uses pseudo-F2 (instead of more biased pseudo-F1 in original code)


    References:
    .. [1] Anderson, Marti J. "A new method for non-parametric multivariate
       analysis of variance." Austral Ecology 26.1 (2001): 32-46.

    .. [2] http://cran.r-project.org/web/packages/vegan/index.html
    
    .. [3] M. J. Anderson, “Permutational Multivariate Analysis of Variance (PERMANOVA),” 
        Wiley StatsRef: Statistics Reference Online. pp. 1–15, 15-Nov-2017.


    Examples:
        | See :mod:`skbio.stats.distance.anosim` for usage examples (both functions
        | provide similar interfaces).

    """
    sample_size, num_groups, grouping, tri_idxs, distances = _preprocess_input(distance_matrix, grouping, column)

    # Create subjects indexing array:
    subjects = _create_subjects_index_arr(subjects = subjects, grouping = grouping)    
  
    # Calculate number of objects in each group.
    group_sizes = np.bincount(grouping)
    s_T = (distances ** 2).sum() / sample_size

    test_stat_function = partial(_compute_f_stat, sample_size, num_groups, tri_idxs, distances, group_sizes, s_T)
    
    stat, p_value, effect_sizes = _run_monte_carlo_stats(test_stat_function, grouping, subjects, permutations, paired)

    results = _build_results('PERMANOVA', paired, 'pseudo-F2', sample_size, num_groups, stat, p_value, effect_sizes, permutations)
    
    #stats_dict = {'method': 'PERMANOVA', 'paired': paired, 'statistic name':'pseudo-F','sample_size':sample_size,'num_groups':num_groups,'statistic value':stat,'p-value':p_value, 'effect_sizes':effect_sizes, 'permutations':permutations,'result string': results} 
    
    return results

def permdisp(distance_matrix, grouping, column=None, test='centroid',
             permutations=999, paired = False, subjects = None):
    """
    Test for Homogeneity of Multivariate Groups Disperisons using Martin Anderson's PERMDISP2 procedure.

    | PERMDISP is a multivariate analogue of Levene's test for homogeneity of
    | multivariate variances. Distances are handled by reducing the
    | original distances to principal coordinates. PERMDISP calculates an
    | F-statistic to assess whether the dispersions between groups is significant


    Args:
        :distance_matrix:
            | DistanceMatrix
            | Distance matrix containing distances between objects (e.g., distances
            | between samples of microbial communities).
        :grouping:
            | 1-D array_like or pandas.DataFrame
            | Vector indicating the assignment of objects to groups. For example,
            | these could be strings or integers denoting which group an object
            | belongs to. If `grouping` is 1-D ``array_like``, it must be the same
            | length and in the same order as the objects in `distance_matrix`. If
            | `grouping` is a ``DataFrame``, the column specified by `column` will be
            | used as the grouping vector. The ``DataFrame`` must be indexed by the
            | IDs in `distance_matrix` (i.e., the row labels must be distance matrix
               | IDs), but the order of IDs between `distance_matrix` and the
               | ``DataFrame`` need not be the same. All IDs in the distance matrix must
               | be present in the ``DataFrame``. Extra IDs in the ``DataFrame`` are
               | allowed (they are ignored in the calculations).
          :column: 
               | str, optional
               | Column name to use as the grouping vector if `grouping` is a
               | ``DataFrame``. Must be provided if `grouping` is a ``DataFrame``.
               | Cannot be provided if `grouping` is 1-D ``array_like``.
          :test:
               | {'centroid', 'median'}
               | determines whether the analysis is done using centroid or spatial median.
          :permutations: 
               | int, optional
               | Number of permutations to use when assessing statistical
               | significance. Must be greater than or equal to zero. If zero,
               | statistical significance calculations will be skipped and the p-value
               | will be ``np.nan``.
          :paired: 
               | bool, optional
               | If True: limit the type of permutations, such that permutations happen 
               | only over groups, not over samples. 
               | [Sep 27, 2019: CORRECT IMPLEMENTATION? Only operates on allowed permutations.]
          :subjects: 
               | 1-D array_like with indices for subjects for use in paired permanova.
               | If None: array (0...ni) will be generated for each group (same size!).

     Returns:
          | pandas.Series
        | Results of the statistical test, including ``test statistic`` and
        | ``p-value``.

    Raises:
          :TypeError:
               | If, when using the spatial median test, the pcoa ordination is not of
               | type np.float32 or np.float64, the spatial median function will fail
               | and the centroid test should be used instead
          :ValueError:
               | If the test is not centroid or median.
          :TypeError:
               | If the distance matrix is not an instance of a
               | ``skbio.DistanceMatrix``.
          :ValueError:
               | If there is only one group
          :ValueError:
               | If a list and a column name are both provided
          :ValueError:
               | If a list is provided for `grouping` and it's length does not match
               | the number of ids in distance_matrix
          :ValueError:
               | If all of the values in the grouping vector are unique
          :KeyError:
               | If there are ids in grouping that are not in distance_matrix

    See Also:
          permanova

    Notes:
          | The significance of the results from this function will be the same as the
          | results found in vegan's betadisper, however due to floating point
          | variability the F-statistic results may vary slightly.
          |
          | See [1]_ for the original method reference, as well as
          | ``vegan::betadisper``, available in R's vegan package [2]_.

    References:
    .. [1] Anderson, Marti J. "Distance-Based Tests for Homogeneity of
        Multivariate Dispersions." Biometrics 62 (2006):245-253

    .. [2] http://cran.r-project.org/web/packages/vegan/index.html
    
    .. [3] M. J. Anderson, “Permutational Multivariate Analysis of Variance (PERMANOVA),” 
        Wiley StatsRef: Statistics Reference Online. pp. 1–15, 15-Nov-2017.

    Examples:
          Load a 6x6 distance matrix and grouping vector denoting 2 groups of
          objects:

          >>> from skbio import DistanceMatrix
          >>> dm = DistanceMatrix([[0,    0.5,  0.75, 1, 0.66, 0.33],
          ...                       [0.5,  0,    0.25, 0.33, 0.77, 0.61],
          ...                       [0.75, 0.25, 0,    0.1, 0.44, 0.55],
          ...                       [1,    0.33, 0.1,  0, 0.75, 0.88],
          ...                       [0.66, 0.77, 0.44, 0.75, 0, 0.77],
          ...                       [0.33, 0.61, 0.55, 0.88, 0.77, 0]],
          ...                       ['s1', 's2', 's3', 's4', 's5', 's6'])
          >>> grouping = ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']

          Run PERMDISP using 99 permutations to caluculate the p-value:

          >>> from permanova_paired import permdisp
          >>> import numpy as np
          >>> #make output deterministic, should not be included during normal use
          >>> np.random.seed(0)
          >>> permdisp(dm, grouping, permutations=99)
          method name               PERMDISP
          test statistic name        F-value
          sample size                      6
          number of groups                 2
          test statistic             1.03296
          p-value                       0.35
          number of permutations          99
          Name: PERMDISP results, dtype: object

          The return value is a ``pandas.Series`` object containing the results of
          the statistical test.

          To suppress calculation of the p-value and only obtain the F statistic,
          specify zero permutations:

          >>> permdisp(dm, grouping, permutations=0)
          method name               PERMDISP
          test statistic name        F-value
          sample size                      6
          number of groups                 2
          test statistic             1.03296
          p-value                        NaN
          number of permutations           0
          Name: PERMDISP results, dtype: object

          PERMDISP computes variances based on two types of tests, using either
          centroids or spatial medians, also commonly referred to as a geometric
          median. The spatial median is thought to yield a more robust test
          statistic, and this test is used by default. Spatial medians are computed
          using an iterative algorithm to find the optimally minimum point from all
          other points in a group while centroids are computed using a deterministic
          formula. As such the two different tests yeild slightly different F
          statistics.

          >>> np.random.seed(0)
          >>> permdisp(dm, grouping, test='centroid', permutations=6)
          method name               PERMDISP
          test statistic name        F-value
          sample size                      6
          number of groups                 2
          test statistic             3.67082
          p-value                   0.428571
          number of permutations           6
          Name: PERMDISP results, dtype: object

          You can also provide a ``pandas.DataFrame`` and a column denoting the
          grouping instead of a grouping vector. The following DataFrame's
          Grouping column specifies the same grouping as the vector we used in the
          previous examples.:

          >>> import pandas as pd
          >>> df = pd.DataFrame.from_dict(
          ...      {'Grouping': {'s1': 'G1', 's2': 'G1', 's3': 'G1', 's4': 'G2',
          ...                    's5': 'G2', 's6': 'G2'}})
          >>> permdisp(dm, df, 'Grouping', permutations=6, test='centroid')
          method name               PERMDISP
          test statistic name        F-value
          sample size                      6
          number of groups                 2
          test statistic             3.67082
          p-value                   0.428571
          number of permutations           6
          Name: PERMDISP results, dtype: object

          Note that when providing a ``DataFrame``, the ordering of rows and/or
          columns does not affect the grouping vector that is extracted. The
          ``DataFrame`` must be indexed by the distance matrix IDs (i.e., the row
          labels must be distance matrix IDs).

          If IDs (rows) are present in the ``DataFrame`` but not in the distance
          matrix, they are ignored. The previous example's ``s7`` ID illustrates this
          behavior: note that even though the ``DataFrame`` had 7 objects, only 6
          were used in the test (see the "Sample size" row in the results above to
          confirm this). Thus, the ``DataFrame`` can be a superset of the distance
          matrix IDs. Note that the reverse is not true: IDs in the distance matrix
          *must* be present in the ``DataFrame`` or an error will be raised.

          PERMDISP should be used to determine whether the dispersions between the
          groups in your distance matrix are significantly separated.
          A non-significant test result indicates that group dispersions are similar
          to each other. PERMANOVA or ANOSIM should then be used in conjunction to
          determine whether clustering within groups is significant.
    """
    if test not in ['centroid', 'median']:
        raise ValueError('Test must be centroid or median')

    ordination = pcoa(distance_matrix)
    samples = ordination.samples

    sample_size, num_groups, grouping, tri_idxs, distances = _preprocess_input(distance_matrix, grouping, column)
    
    # Create subjects indexing array:
    subjects = _create_subjects_index_arr(subjects = subjects, grouping = grouping)    
   
    test_stat_function = partial(_compute_groups, samples, test)

    stat, p_value, effect_sizes = _run_monte_carlo_stats(test_stat_function, grouping, subjects, permutations, paired)

    results = _build_results('PERMDISP', paired, 'F-value', sample_size, num_groups, stat, p_value, effect_sizes, permutations)
       
    return results
    

def _compute_groups(samples, test_type, grouping, subjects, paired, *args):
    """ calculate test statistic for PERMDISP"""
    groups = []

    samples['grouping'] = grouping
    if test_type == 'centroid':
        centroids = samples.groupby('grouping').aggregate('mean')
    elif test_type == 'median':
        centroids = samples.groupby('grouping').aggregate(_config_med)

    from scipy.spatial.distance import cdist # lazy import
    for label, df in samples.groupby('grouping'):
        groups.append(cdist(df.values[:, :-1], [centroids.loc[label].values], metric='euclidean'))

    from scipy.stats import f_oneway # lazy import
    stat, _ = f_oneway(*groups)
    stat = stat[0]
    
    # effect sizes:
    num_groups = len(np.unique(grouping))
    sample_size = len(grouping)
    if paired == True:
        dfErr = (num_groups - 1)*(len(np.unique(subjects)) - 1)
    else:
        dfErr = sample_size - num_groups
    R2 = 1.0 - 1 / (1 + stat * num_groups / (dfErr - 1))  
    R2adj = 1 - ((1-R2)*(sample_size - 1)/(sample_size - num_groups - 1))
    effect_sizes = {'p_eta2':np.nan,'omega2':np.nan, 'R2':R2, 'R2adj':R2adj} # not yet determined
    
    return stat, effect_sizes

def _config_med(x):
    """
    slice the vector up to the last value to exclude grouping column and transpose the vector to be compatible with hd.geomedian
    """
    X = x.values[:, :-1]
    return np.array(hd.geomedian(X.T))
 
def _build_results(method_name, paired, test_stat_name, sample_size, num_groups,
                   stat, p_value, 
                   effect_sizes, 
                   permutations):
    """Return ``pandas.Series`` containing results of statistical test."""
    import pandas # lazy import
    return pandas.Series(
        data=[method_name, paired, test_stat_name, sample_size, num_groups, 
              stat, p_value, 
              effect_sizes['p_eta2'], effect_sizes['omega2'], effect_sizes['R2'],effect_sizes['R2adj'], 
              permutations],
        index=['method name', 'paired', 'test statistic name', 
               'sample size', 'number of groups', 
               'test statistic', 'p-value', 
               'p_eta2', 'omega2','R2','R2adj',
               'number of permutations'],
        name='%s results' % method_name)    

def _get_distance_matrix_grouping(*X, metric = 'euclidean', Dscale = 1):
    """ Get distance matrix (skbio format) and grouping indexing array from raw data"""
    # Create long format data array and grouping indices:
    ni = np.empty((len(X),), dtype= np.int32)
    for i,Xi in enumerate(X):
        ni[i] = Xi.shape[0]
        if i == 0:
            XY = Xi
        else:
            XY = np.vstack((XY,Xi))
    grouping = [[i]*n for i,n in enumerate(ni)]
    grouping = list(itertools.chain(*grouping))
    
    # Calculate pairwise distances:
    import scipy # lazy import
    D = scipy.spatial.distance.pdist(XY, metric=metric)*Dscale
    Dsq = scipy.spatial.distance.squareform(D)

    # Get skbio distance matrix:
    Dm = skbio.stats.distance.DistanceMatrix(Dsq)
    
    return Dm, grouping
    
def run_permanova_permdisp(*X, metric = 'euclidean', paired = True, 
                           permutations = 999, verbosity = 1, 
                           run_permanova = True, run_permdisp = True,  
                           Dscale = 1.0, permdisp_test = 'centroid'):
    """
    | Run permutation based analysis of variance "permanova" (cfr. diff. in mean)
    | and/or analysis of dispersion "permdisp" (cfr. differences in spread of data)
    | on data array *X.
    
    Args:
        :*X: 
            | ndarrays with raw data (rows: subjects, columns: conditions).
        :metric:
            | 'euclidean', optional
            | Used by scipy.spatial.distance.pdist() to convert data to 
            | pairwise distances for use in permanova() and permdisp().
        :paired:
            | True, optional
            | True: paired or dependent data (repeated measures, permute conditions only)
            | False: independent data (permute conditions and observers)
        :permutations:
            | 999, optional
            | Number of permutations (minimum p-value = 1/(permutations+1))
        :verbosity:
            | 1, optional
            | 1: print output.
        :run_permanova:
            | True, optional
        :run_permdisp:
            | True, optional
        :Dscale:
            | 1.0, optional
            | Scale factor for computed distances.
        :permdisp_test : 
            | 'centroid', optional
            | Options: 'centroid', 'median'
            | Determines whether the permdisp analysis is done using centroid or spatial median.
            
    Returns:
        :(stats_permanova, stats_permdisp):
            | pandas.Series with calculated statistics and other info on the tests.
    
    Notes:
        * Is based on and uses the skbio package (install manually: pip install skbio).
        * Based on code for permanova and permdisp, but extended for repeated measures or paired data.
        * Uses pseudo-F2 (instead of more biased pseudo-F1 in original code)
    
    References:
        1. M. J. Anderson, “Permutational Multivariate Analysis of Variance (PERMANOVA),” 
        Wiley StatsRef: Statistics Reference Online. pp. 1–15, 15-Nov-2017.
        2. `Scikit-bio <http://scikit-bio.org/>`_
        
    
    """
    # Create main distance matrix and grouping indices:
    Dm, grouping = _get_distance_matrix_grouping(*X, metric = metric, Dscale = Dscale)
    
    # Perform permdisp & permanova:
    effect_sizes_empty = {'p_eta2':np.nan,'omega2':np.nan, 'R2':np.nan, 'R2adj':np.nan}
    if run_permdisp == True:
        stats_pdisp = permdisp(Dm, grouping, column = None, permutations = permutations, paired = paired, test = permdisp_test)
    else:
        stats_pdisp = _build_results('PERMDISP', paired, 'F-value', np.nan, np.nan, np.nan, np.nan, effect_sizes_empty, np.nan)
        
    if run_permanova == True:
        stats_pman = permanova(Dm, grouping, column = None, permutations = permutations, paired = paired)
    else:
        stats_pman = _build_results('PERMANOVA', paired, 'pseudo-F', np.nan, np.nan, np.nan, np.nan, effect_sizes_empty, np.nan)
        
    if verbosity == 1:
        if run_permanova == True:
            print('\nPERMANOVA (paired == {:b}):'.format(paired))
            print(stats_pman)        
        if run_permdisp == True:
            print('\nPERMDISP (paired == {:b}):'.format(paired))
            print(stats_pdisp)
    
    return stats_pman, stats_pdisp 
    
    
if __name__ == '__main__': 

    # Significant diff.:
    scores = np.array([[33,28,26,34],
                       [22,19,20,23],
                       [24,20,22,25],
                       [28,24,20,24],
                       [26,28,24,30]])
    
    # Non-significant diff.:
    scores2 = np.array([[16,28,24,14],
                       [17,18,16,20],
                       [22,20,18,20],
                       [32,34,31,33],
                       [36,38,34,35]])
    
    Xs = [scores[:,i:i+1] for i in range(scores.shape[1])]

    out = run_permanova_permdisp(*Xs, metric = 'euclidean', paired = True, 
                                 permutations = 999, verbosity = 1, 
                                 run_permdisp = False, run_permanova = True,
                                 permdisp_test = 'centroid');
#    out = run_permanova_permdisp(*Xs, metric = 'euclidean', paired = False,
#                                 permutations = 999,  verbosity = 1, 
#                                 run_permdisp = False, run_permanova = True,
#                                 permdisp_test = 'centroid')
