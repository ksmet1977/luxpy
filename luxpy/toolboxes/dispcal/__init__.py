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
Module for display characterization
===================================

 :_PATH_DATA: path to package data folder   

 :_RGB:  set of RGB values that work quite well for display characterization
   
 :_XYZ: example set of measured XYZ values corresponding to the RGB values in _RGB
 
 :find_index_in_rgb(): Find the index/indices of a specific r,g,b combination k in the ndarray rgb.
     
 :find_pure_rgb(): Find the indices of all pure r,g,b (single channel on) in the ndarray rgb.
 
 :correct_for_black: Correct xyz for black level (flare) 
 
 :TR_ggo(),TRi_ggo(): Forward (rgblin-to-xyz) and inverse (xyz-to-rgblin) GGO Tone Response models.
 
 :TR_gog(),TRi_gog(): Forward (rgblin-to-xyz) and inverse (xyz-to-rgblin) GOG Tone Response models.
 
 :TR_gogo(),TRi_gogo(): Forward (rgblin-to-xyz) and inverse (xyz-to-rgblin) GOGO Tone Response models.
 
 :TR_sigmoid(),TRi_sigmoid(): Forward (rgblin-to-xyz) and inverse (xyz-to-rgblin) SIGMOID Tone Response models.
 
 :estimate_tr(): Estimate Tone Response curves.
 
 :optimize_3x3_transfer_matrix(): Optimize the 3x3 rgb-to-xyz transfer matrix.
     
 :get_3x3_transfer_matrix_from_max_rgb(): Get the rgb-to-xyz transfer matrix from the maximum R,G,B single channel outputs
    
 :calibrate(): Calculate TR parameters/lut and conversion matrices
   
 :calibration_performance(): Check calibration performance (cfr. individual and average color differences for each stimulus). 

 :rgb_to_xyz(): Convert input rgb to xyz
    
 :xyz_to_rgb(): Convert input xyz to rgb
     
 :DisplayCalibration(): Calculate TR parameters/lut and conversion matrices and store in object.
 
 :generate_training_data(): Generate RGB training pairs by creating a cube of RGB values. 

 :generate_test_data(): Generate XYZ test values by creating a cube of CIELAB L*a*b* values, then converting these to XYZ values. 

 :plot_rgb_xyz_lab_of_set(): Make 3d-plots of the RGB, XYZ and L*a*b* cubes of the data in rgb_xyz_lab. 

 :split_ramps_from_cube(): Split a cube data set in pure RGB (ramps) and non-pure (remainder of cube). 

 :is_random_sampling_of_pure_rgbs(): Return boolean indicating if the RGB cube axes (=single channel ramps) are sampled (different increment) independently from the remainder of the cube.

 :ramp_data_to_cube_data(): Create a RGB and XYZ cube from the single channel ramps in the training data.
  
 :GGO_GOG_GOGO_PLI: Class for characterization models that combine a 3x3 transfer matrix and a GGO, GOG, GOGO, SIGMOID, PLI and 1-D LUT Tone response curve  
                     |  - Tone Response curve models:
                     |    * GGO: gain-gamma-offset model: y = gain*x**gamma + offset
                     |    * GOG: gain-offset-gamma model: y = (gain*x + offset)**gamma
                     |    * GOG: gain-offset-gamma-offset model: y = (gain*x + offset)**gamma + offset
                     |    * SIGMOID: sigmoid (S-shaped) model: y = offset + gain* [1 / (1 + q*exp(-a/gamma*(x - m)))]**(gamma)
                     |    * PLI: Piece-wise Linear Interpolation
                     |    * LUT: 1-D Look-Up-Tables for the TR
                     |  - RGB-to-XYZ / XYZ-to-RGB transfer matrices:
                     |     * M fixed: derived from tristimulus values of maximum single channel output
                     |     * M optimized: by minimizing the RMSE between measured and predicted XYZ values
                     
 :MLPR: Class for Multi-Layer Perceptron Regressor based model.
     
 :POR: Class for POlynomial Regression based model.
 
 :LUTNNLI: Class for LUT-Nearest-Neighbour-distance-weighted-Linear-Interpolation based models.
     
 :LUTQHLI: Class for LUT-QHul-Linear-Interpolation based models (cfr. scipt.interpolate.LinearNDInterpolator)

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""
#from .displaycalibration import *
#__all__ = displaycalibration.__all__

from luxpy.toolboxes.dispcal.displaycalibration import (_PATH_DATA, _parse_rgbxyz_input, find_index_in_rgb,
           _plot_target_vs_predicted_lab,_plot_DEs_vs_digital_values,
           calibrate, calibration_performance, 
           rgb_to_xyz, xyz_to_rgb, DisplayCalibration,_RGB, _XYZ,
           TR_ggo,TRi_ggo,TR_gog,TRi_gog,TR_gogo,TRi_gogo,
           TR_sigmoid,TRi_sigmoid, correct_for_black,
           _rgb_linearizer,_rgb_delinearizer, estimate_tr,
           optimize_3x3_transfer_matrix,get_3x3_transfer_matrix_from_max_rgb)
__all__ = ['_PATH_DATA', '_parse_rgbxyz_input', 'find_index_in_rgb',
           '_plot_target_vs_predicted_lab','_plot_DEs_vs_digital_values',
           'calibrate', 'calibration_performance', 
           'rgb_to_xyz', 'xyz_to_rgb', 'DisplayCalibration','_RGB', '_XYZ',
           'TR_ggo','TRi_ggo','TR_gog','TRi_gog','TR_gogo','TRi_gogo',
           'TR_sigmoid','TRi_sigmoid', 'correct_for_black',
           '_rgb_linearizer','_rgb_delinearizer', 'estimate_tr',
           'optimize_3x3_transfer_matrix','get_3x3_transfer_matrix_from_max_rgb'
           ]


from .rgbtraining_xyztest_set_generation import (generate_training_data, generate_test_data, split_ramps_from_cube, is_random_sampling_of_pure_rgbs, plot_rgb_xyz_lab_of_set)
__all__ += ['generate_training_data','generate_test_data','split_ramps_from_cube', 'is_random_sampling_of_pure_rgbs', 'plot_rgb_xyz_lab_of_set']

from .display_characterization_models import (ramp_data_to_cube_data, GGO_GOG_GOGO_PLI, MLPR, POR, LUTNNLI, LUTQHLI)
__all__ += ['ramp_data_to_cube_data', 'GGO_GOG_GOGO_PLI', 'MLPR', 'POR', 'LUTNNLI', 'LUTQHLI']

from .virtualdisplay import (VirtualDisplay, _VIRTUALDISPLAY_PARS,_VIRTUALDISPLAY_KWAK2000_PARS)
__all__ += ['VirtualDisplay','_VIRTUALDISPLAY_PARS','_VIRTUALDISPLAY_KWAK2000_PARS']