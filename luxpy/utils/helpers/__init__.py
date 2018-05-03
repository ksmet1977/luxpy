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
Module with helper functions 
============================


 :np2d(): Make a tuple, list or array at least a 2D numpy array.

 :np2dT(): Make a tuple, list or array at least a 2D numpy array and tranpose.

 :np3d(): Make a tuple, list or array at least a 3D numpy array.

 :np3dT(): Make a tuple, list or array at least a 3D numpy array 
           and tranpose (swap) first two axes.

 :normalize_3x3_matrix(): Normalize 3x3 matrix M to xyz0 -- > [1,1,1]

 :put_args_in_db():  | Takes the args with not-None input values of a function 
                       and overwrites the values of the corresponding keys 
                       in dict db.
                     | See put_args_in_db? for more info.

 :vec_to_dict(): Convert dict to vec and vice versa.

  getdata(): Get data from csv-file or convert between pandas dataframe
             and numpy 2d-array.

 :dictkv(): Easy input of of keys and values into dict 
            (both should be iterable lists).

 :OD(): Provides a nice way to create OrderedDict "literals".

 :meshblock(): | Create a meshed block.
               | (Similar to meshgrid, but axis = 0 is retained) 
               | To enable fast blockwise calculation.

 :aplit(): Split ndarray data on (default = last) axis.

 :ajoin(): Join tuple of ndarray data on (default = last) axis.

 :broadcast_shape(): | Broadcasts shapes of data to a target_shape. 
                     | Useful for block/vector calculations when numpy fails 
                       to broadcast correctly.

 :todim(): Expand x to dimensions that are broadcast-compatable 
           with shape of another array.

===============================================================================
"""
from .helpers import *
__all__ = helpers.__all__