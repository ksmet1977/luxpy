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
Module for IES TM30 color rendition output
==========================================

iestm30/graphics.py
-------------------
                       
 :spd_to_ies_tm30_metrics(): Calculates IES TM30 metrics from spectral data
 
 :plot_cri_graphics(): Plots graphical information on color rendition 
                       properties based on spectral data input or dict with 
                       pre-calculated measures.
                       
 :_tm30_process_spd(): Calculate all required parameters for plotting from spd using cri.spd_to_cri()

 :plot_tm30_cvg(): Plot TM30 Color Vector Graphic (CVG).
 
 :plot_tm30_Rfi(): Plot Sample Color Fidelity values (Rfi).
 
 :plot_tm30_Rxhj(): Plot Local Chroma Shifts (Rcshj), Local Hue Shifts (Rhshj) and Local Color Fidelity values (Rfhj).

 :plot_tm30_Rcshj(): Plot Local Chroma Shifts (Rcshj).

 :plot_tm30_Rhshj(): Plot Local Hue Shifts (Rhshj).

 :plot_tm30_Rfhj(): Plot Local Color Fidelity values (Rfhj).

 :plot_tm30_spd(): Plot test SPD and reference illuminant, both normalized to the same luminous power.

 :plot_tm30_report(): Plot a figure with an ANSI/IES-TM30 color rendition report.
 
 
 :plot_cri_graphics(): Plots graphical information on color rendition 
                       properties based on spectral data input or dict with 
                       pre-calculated measures (cusom design). 
                       Includes Metameric uncertainty index Rt and vector-fields
                       of color rendition shifts.


iestm30/metrics.py
------------------

 :spd_to_ies_tm30_metrics(): Calculates IES TM30 metrics from spectral data + Metameric Uncertainty + Vector Fields

 :tm30_metrics_to_annexE_recommendations(): Get ANSI/IES-TM30 Annex E recommendation for all three design intents ['Preference', 'Vividness', 'Fidelity']

iestm30/metrics_fast.py
-----------------------

 :_cri_ref(): Calculate multiple reference illuminant spectra based on ccts for color rendering index calculations.

 :_xyz_to_jab_cam02ucs(): Calculate CAM02-UCS J'a'b' coordinates from xyz tristimulus values of sample and white point.

 :spd_tom_tm30(): Calculate tm30 measures from spd.
 
 * for increased speed in spectral optmizations.

.. codeauthor:: Kevin A.G. Smet (ksmet1977 at gmail.com)
"""