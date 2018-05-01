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
###############################################################################
# Module for building and optimizing SPDs.
###############################################################################

# gaussian_spd(): Generate Gaussian spectrum.

# butterworth_spd(): Generate Butterworth based spectrum.

# mono_led_spd(): Generate monochromatic LED spectrum based on a Gaussian 
                or butterworth profile or according to Ohno (Opt. Eng. 2005).

# phosphor_led_spd(): Generate phosphor LED spectrum with up to 2 phosphors 
                      based on Smet (Opt. Expr. 2011).

# spd_builder(): Build spectrum based on Gaussians, monochromatic 
                 and/or phophor LED spectra.

# color3mixer(): Calculate fluxes required to obtain a target chromaticity 
                    when (additively) mixing 3 light sources.

# colormixer(): Calculate fluxes required to obtain a target chromaticity 
                    when (additively) mixing N light sources.

# spd_builder(): Build spectrum based on Gaussians, monochromatic 
                 and/or phophor LED-type spectra.
                   
# get_w_summed_spd(): Calculate weighted sum of spds.
 
# fitnessfcn(): Fitness function that calculates closeness of solution x to 
                target values for specified objective functions.
         
# spd_constructor_2(): Construct spd from spectral model parameters 
                        using pairs of intermediate sources.
                
# spd_constructor_3(): Construct spd from spectral model parameters 
                        using trio's of intermediate sources.
     
# spd_optimizer_2_3(): Optimizes the weights (fluxes) of a set of component 
                        spectra by combining pairs (2) or trio's (3) of 
                        components to intermediate sources until only 3 remain.
                        Color3mixer can then be called to calculate required 
                        fluxes to obtain target chromaticity and fluxes are 
                        then back-calculated.                                   
                        
# get_spd_pars_optim_dict(): Setup dict with optimization parameters.
                        
# initialize_spd_model_pars(): Initialize spd_model_pars (for spd_constructor)
                                based on type of component_data.

# initialize_spd_optim_pars(): Initialize spd_optim_pars (x0, lb, ub for use
                                with math.minimizebnd) based on type 
                                of component_data.
                
# spd_optimizer(): Generate a spectrum with specified white point and optimized
                    for certain objective functions from a set of component 
                    spectra or component spectrum model parameters.
                    
#------------------------------------------------------------------------------
Created on Wed Apr 25 09:07:04 2018

@author: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
from .spdbuilder import *
