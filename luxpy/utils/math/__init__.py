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
Module with useful math functions
=================================

 :normalize_3x3_matrix(): Normalize 3x3 matrix M to xyz0 -- > [1,1,1]

 :line_intersect(): | Line intersections of series of two line segments a and b. 
                    | https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

 :positive_arctan(): Calculates the positive angle (0°-360° or 0 - 2*pi rad.) 
                     from x and y.

 :dot23(): Dot product of a 2-d ndarray 
           with a (N x K x L) 3-d ndarray using einsum().

 :check_symmetric(): Checks if A is symmetric.

 :check_posdef(): Checks positive definiteness of a matrix via Cholesky.

 :symmM_to_posdefM(): | Converts a symmetric matrix to a positive definite one. 
                      | Two methods are supported:
                      |    * 'make': A Python/Numpy port of Muhammad Asim Mubeen's
                      |              matlab function Spd_Mat.m 
                      |       (https://nl.mathworks.com/matlabcentral/fileexchange/45873-positive-definite-matrix)
                      |    * 'nearest': A Python/Numpy port of John D'Errico's 
                      |                'nearestSPD' MATLAB code. 
                      |        (https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite)

 :bvgpdf(): Evaluate bivariate Gaussian probability density function (BVGPDF) 
            at (x,y) with center mu and inverse covariance matric, sigmainv.

 :mahalanobis2(): Evaluate the squared mahalanobis distance with center mu and 
                  shape and orientation determined by sigmainv. 

 :rms(): Calculates root-mean-square along axis.

 :geomean(): Calculates geometric mean along axis.

 :polyarea(): | Calculates area of polygon. 
              | (First coordinate should also be last)

 :erf(), erfinv(): erf-function and its inverse, direct import from scipy.special

 :cart2pol(): Converts Cartesian to polar coordinates.

 :pol2cart(): Converts polar to Cartesian coordinates.
 
 :cart2spher(): Converts Cartesian to spherical coordinates.
 
 :spher2cart(): Converts spherical to Cartesian coordinates.

 :magnitude_v():  Calculates magnitude of vector.

 :angle_v1v2():  Calculates angle between two vectors.

 :histogram(): | Histogram function that can take as bins either the center
               | (cfr. matlab hist) or bin-edges.
               
 :v_to_cik(): Calculate 2x2 '(covariance matrix)^-1' elements cik from v-format ellipse descriptor.

 :cik_to_v(): Calculate v-format ellipse descriptor from 2x2 'covariance matrix'^-1 cik.

 :minimizebnd(): scipy.minimize() that allows contrained parameters on 
                 unconstrained methods(port of Matlab's fminsearchbnd). 
                 Starting, lower and upper bounds values can also be provided 
                 as a dict.
                 
 :DEMO: Module for Differential Evolutionary Multi-objective Optimization  (DEMO).
 
 :particleswarm(): Global minimization function using particle swarms (wrapper around pyswarms.single.GlobalBestPSO)
 
 :vec3: Module for spherical vector coordinates.
 
 :fmod(): Floating point modulus, e.g.: fmod(theta, np.pi * 2) would keep an angle in [0, 2pi]b

 :fit_ellipse(): Fit an ellipse to supplied data points.
 
 :fit_cov_ellipse(): Fit an covariance ellipse to supplied data points.
 
 :interp1(): Perform a 1-dimensional linear interpolation (wrapper around scipy.interpolate.InterpolatedUnivariateSpline).
 
 :ndinterp1(): Perform n-dimensional interpolation using Delaunay triangulation.
 
 :ndinterp1_scipy(): Perform n-dimensional interpolation using Delaunay triangulation (wrapper around scipy.interpolate.LinearNDInterpolator)
 
 :box_m(): Performs a Box M test on covariance matrices.
 
 :pitman_morgan(): Pitman-Morgan Test for the difference between correlated variances with paired samples.
    
 :bipolymod: Module for Bivariate Polynomial Model Optimization             
===============================================================================
"""
from .basics import *
__all__ = basics.__all__

from .minimizebnd import minimizebnd
__all__ += ['minimizebnd']

from .DEMO import DEMO as DEMO
__all__ += ['DEMO']

from .vec3 import vec3 as vec3
__all__ += ['vec3']

from .particleswarm import particleswarm
__all__ += ['particleswarm']

from . import bivariate_poly_model as bipolymod
__all__ += ['bipolymod']




