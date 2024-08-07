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
Module for building and optimizing SPDs
=======================================

 spdbuilder.py
 
Functions
---------

 :gaussian_spd(): Generate Gaussian spectrum.

 :butterworth_spd(): Generate Butterworth based spectrum.
 
 :lorentzian2_spd(): Generate 2nd order Lorentzian based spectrum.
 
 :roundedtriangle_spd(): Generate a rounded triangle based spectrum.

 :mono_led_spd(): Generate monochromatic LED spectrum based on a Gaussian 
                  or butterworth profile or according to Ohno (Opt. Eng. 2005).

 :spd_builder(): Build spectrum based on Gaussians, monochromatic 
                 and/or phophor LED spectra.

 :color3mixer(): Calculate fluxes required to obtain a target chromaticity 
                 when (additively) mixing 3 light sources.

 :colormixer(): Calculate fluxes required to obtain a target chromaticity 
                when (additively) mixing N light sources.
                
 :colormixer_pinv(): Additive color mixer of N primaries using using Moore-Penrose pseudo-inverse matrix.

 :spd_builder(): Build spectrum based on Gaussians, monochromatic 
                 and/or phophor LED-type spectra.
                   
 :get_w_summed_spd(): Calculate weighted sum of spds.
 
 :fitnessfcn(): Fitness function that calculates closeness of solution x to 
                target values for specified objective functions.
         
 :spd_constructor_2(): Construct spd from spectral model parameters 
                       using pairs of intermediate sources.
                
 :spd_constructor_3(): Construct spd from spectral model parameters 
                       using trio's of intermediate sources.
     
 :spd_optimizer_2_3(): Optimizes the weights (fluxes) of a set of component 
                       spectra by combining pairs (2) or trio's (3) of 
                       components to intermediate sources until only 3 remain.
                       Color3mixer can then be called to calculate required 
                       fluxes to obtain target chromaticity and fluxes are 
                       then back-calculated.                                   
                        
 :get_optim_pars_dict(): Setup dict with optimization parameters.
                        
 :initialize_spd_model_pars(): Initialize spd_model_pars (for spd_constructor)
                               based on type of component_data.

 :initialize_spd_optim_pars(): Initialize spd_optim_pars (x0, lb, ub for use
                               with math.minimizebnd) based on type 
                               of component_data.
                
 :spd_optimizer(): Generate a spectrum with specified white point and optimized
                   for certain objective functions from a set of component 
                   spectra or component spectrum model parameters.
                    
                    
                    
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
 1. See examples below (in spdoptimizer2020.'__main__') for use.                
                   

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from .spdbuilder import *
#__all__ = spdbuilder.__all__


from .spdoptimizer2020 import *
#__all__ += spdoptimizer2020.__all__

