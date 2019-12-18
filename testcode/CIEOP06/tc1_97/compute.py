#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute: Calculate the CIE functions provided by CIE TC 1-97.

Copyright (C) 2012-2017 Ivar Farup and Jan Henrik Wold

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy.optimize
import scipy.interpolate
import warnings
from scipy.spatial import Delaunay
from tc1_97.utils import resource_path

# The following coding conventions have been applied:
#
# * All functions have a docstring of minimum one line describing the
#   overall purpose of the function as well as the names and types
#   of the parameters and return values.
#
# * In some places, variables have been reused, typically in order to build up
#   arrays sequentially. The typical example is found in, e.g.,
#   absorptance_from_LMS10q, where the absorptance array is first initialized
#   as the absorbance, then edited in place. This is in order to achieve
#   a shorter and also more efficient code with less memory allocation, and to
#   avoid namespace pollution in the case of comput_tabulated. Unfortunately,
#   it reduces the readability of the code somewhat. Therefore, all such
#   occurences are marked with comments in the code.


# =============================================================================
# General functions
# =============================================================================


def my_round(x, n=0):
    """
    Round array x to n decimal points using round half away from zero.

    This function is needed because the rounding specified in the CIE
    recommendation is different from the standard rounding scheme in python
    (which is following the IEEE recommendation).

    Parameters
    ----------
    x : ndarray
        Array to be rounded
    n : int
        Number of decimal points

    Returns
    -------
    y : ndarray
        Rounded array
    """
    s = np.sign(x)
    return s*np.floor(np.absolute(x)*10**n + 0.5)/10**n


def sign_figs(x, n=0):
    """
    Round x to n significant figures (not decimal points).

    This function is needed because the rounding specified in the CIE
    recommendation is different from the standard rounding scheme in python
    (which is following the IEEE recommendation). Uses my_round (above).

    Parameters
    ----------
    x : int, float or ndarray
        Number or array to be rounded.

    Returns
    -------
    t : float or ndarray
        Rounded number or array.
    """
    if type(x) == float or type(x) == int:
        if x == 0.:
            return 0
        else:
            exponent = np.ceil(np.log10(x))
            return 10**exponent * my_round(x / 10**exponent, n)
    exponent = x.copy()
    exponent[x == 0] = 0
    exponent[x != 0] = np.ceil(np.log10(abs(x[x != 0])))
    return 10**exponent * my_round(x / 10**exponent, n)


def chop(arr, epsilon=1e-14):
    """
    Chop values smaller than epsilon in absolute value to zero.

    Similar to Mathematica function.

    Parameters
    ----------
    arr : float or ndarray
        Array or number to be chopped.
    epsilon : float
        Minimum number.

    Returns
    -------
    chopped : float or ndarray
        Chopped numbers.
    """
    if isinstance(arr, float) or isinstance(arr, int):
        chopped = arr
        if np.abs(chopped) < epsilon:
            chopped = 0
        return chopped
    chopped = arr.copy()                    # initialise to arr values
    chopped[np.abs(chopped) < epsilon] = 0  # set too low values to zero
    return chopped


def read_csv_file(filename, pad=-np.inf):
    """
    Read a CSV file and return pylab array.

    Parameters
    ----------
    filename : string
        Name of the CSV file to read
    pad : float
        Value to pad for missing values.

    Returns
    -------
    csv_array : ndarray
        The content of the CSV file plus padding.
    """
    f = open(resource_path(filename))
    data = f.readlines()
    for i in range(len(data)):
        data[i] = data[i].split(',')
        for j in range(len(data[i])):
            if data[i][j].strip() == '':
                data[i][j] = pad
            else:
                data[i][j] = float(data[i][j])
    return np.array(data)


#=============================================================================  
#  Function/class determining/constituting the database
#=============================================================================

def docul_fine(ocular_sum_32, docul2):
    """
    Calculate the two parts of the expression for the optical density of the 
    ocular media as function of age.

    Parameters
    ----------
    ocular_sum_32 : ndarray
        Sum of two ocular functions
    docul2 : ndarray

    Returns
    -------
    docul1_fine : ndarray
        The computedd values for part 1, docul1, tabulated with high 
        resolution
    docul2_fine : ndarray,
        The computedd values for part 2, docul2, tabulated with high 
        resolution
    """
    docul2_pad = np.zeros((75, 2))             # initialize
    docul2_pad[:, 0] = np.arange(460, 835, 5)  # fill
    docul2_pad[:, 1] = 0                       # fill
    docul2 = np.concatenate((docul2, docul2_pad))
    spl = scipy.interpolate.InterpolatedUnivariateSpline(docul2[:, 0],
                                                         docul2[:, 1])
    docul2_fine = ocular_sum_32.copy()
    docul2_fine[:, 1] = spl(ocular_sum_32[:, 0])
    docul1_fine = ocular_sum_32.copy()
    docul1_fine[:, 1] = ocular_sum_32[:, 1] - docul2_fine[:, 1]
    return (docul1_fine, docul2_fine)


class VisualData:
    """
    Class containing all visual data input to the computations.
    
    """
    absorbance = read_csv_file('data/absorbances0_1nm.csv')[:, [0, 2, 3, 4]]
    macula_2 = read_csv_file('data/absorbances0_1nm.csv')[:, [0, 6]]
    macula_rel = macula_2 / .35  # since macula at 2° has a
                                 # maximum of 0.35 at 460 (at 5nm step)
    docul2 = read_csv_file('data/docul2.csv')
    ocular_sum_32 = read_csv_file(
        'data/absorbances0_1nm.csv')[:, [0, 5]]  # 32 years only!
    (docul1_fine, docul2_fine) = docul_fine(ocular_sum_32, docul2)
    LMS10_log_quantal = read_csv_file('data/ss10q_fine_8dp.csv')
    LMS10_lin_energy_9_signfig = read_csv_file('data/linss10e_fine_8dp.csv', 0)
    LMS10_lin_energy_6_signfig = read_csv_file('data/linss10e_fine.csv', 0)
    LMS2_log_quantal = read_csv_file('data/ss2_10q_fine_8dp.csv')
    LMS2_lin_energy_9_signfig = read_csv_file('data/linss2_10e_fine_8dp.csv',0)
    LMS2_lin_energy_6_signfig = read_csv_file('data/linss2_10e_fine.csv', 0)
    VλLM_10_lin_energy = read_csv_file('data/linCIE2015v10e_fine_8dp.csv')
    VλLM_2_lin_energy = read_csv_file('data/linCIE2015v2e_fine_8dp.csv')
    VλLM_10_log_quantal = read_csv_file('data/logCIE2015v10q_fine_8dp.csv')
    VλLM_2_log_quantal = read_csv_file('data/logCIE2015v2q_fine_8dp.csv')
    XYZ31 = read_csv_file('data/ciexyz31_1.csv')
    XYZ64 = read_csv_file('data/ciexyz64_1.csv')

    
#=============================================================================  
#  Basic colorimetrically related functions
#=============================================================================

def chrom_coords_µ(tristimulus_µ):
    """
    Compute list of chromaticity coordinates of spectral/purple-line
    stimuli from corresponding tristimulus values as parameterized by
    wavelength/complementary wavelength.

    Parameters
    ----------
    tristimulus_µ : ndarray
        The tristimulus values for a given set of spectral/purple-line
        stimuli; wavelength/complementary wavelength in first column.

    Return
    ------
    cc_µ : ndarray
        The computed spectral/purple-line chromaticity coordinates;
        wavelength/complementary wavelength in first column.
    """
    # µ denotes wavelength/complementary wavelength

    (µ, A_µ, B_µ, C_µ) = tristimulus_µ.T
    sumABC_µ = A_µ + B_µ + C_µ
    cc_µ = np.array([µ, A_µ / sumABC_µ, B_µ / sumABC_µ, C_µ / sumABC_µ]).T
    return cc_µ


def chrom_coords_E(tristimulus_λ):
    """
    Compute the chromaticity coordinates of Illuminant E from
    given set of spectral tristimulus values.

    Parameters
    ----------
    tristimulus_λ : ndarray
        The tristimulus values for the given set of spectral
        stimuli; wavelength in first column.

    Return
    ------
    cc_E : ndarray
        The computed chromaticity coordinates of Illuminant E.
    """
    (A_λ, B_λ, C_λ) = (tristimulus_λ.T)[1:4]
    (A_E, B_E, C_E) = [np.sum(A_λ), np.sum(B_λ), np.sum(C_λ)]
    sumABC_E = A_E + B_E + C_E
    cc_E = np.array([A_E / sumABC_E, B_E / sumABC_E, C_E / sumABC_E]).T
    return cc_E


def linear_transformation_λ(trans_mat, tristimulus_λ):
    """
    Transformation of a set of spectral tristimulus values by linear
    transformation,

    Parameters
    ----------
    trans_mat : ndarray
        The transformation matrix of the linear transformation.
    tristimulus_λ : ndarray
        The initial spectral tristimulus values; wavelengths in first
        column.

    Return
    ------
    ABC : ndarray
        The linearly transformed spectral tristimulus values;
        wavelengths in first column.
    """
    (λ, AA_λ, BB_λ, CC_λ) = tristimulus_λ.T
    (A_λ, B_λ, C_λ) = np.dot(trans_mat, np.array([AA_λ, BB_λ, CC_λ]))          
    ABC_λ = np.array([λ, A_λ, B_λ, C_λ]).T   
    return ABC_λ  
#==============================================================================
#  Functions of age and/or field size
#==============================================================================

def d_ocular(age):
    """
    Compute the optical density of the ocular media for given age.

    Computes a weighted average of docul1 and docul2.

    Parameters
    ----------
    age : float
        Age in years.

    Returns
    -------
    docul : ndarray
        The computed optical density of the ocular media; wavelength in first
        column.
    """
    docul = VisualData.docul2_fine.copy()  # initialise for in-place editing
    if age < 60:
        docul[:, 1] = ((1 + 0.02*(age - 32)) * VisualData.docul1_fine[:, 1] +
                       VisualData.docul2_fine[:, 1])
    else:
        docul[:, 1] = ((1.56 + 0.0667*(age - 60)) *
                       VisualData.docul1_fine[:, 1] +
                       VisualData.docul2_fine[:, 1])
    return docul


def d_mac_max(field_size):
    """
    Compute the maximum optical density of the macular pigment for given field
    size.

    Parameters
    ----------
    field_size : float
        Field size in degrees.

    Returns
    -------
    d_mac_max : float
        The computed maximum optical density of the macular pigment.
    """
    return my_round(0.485*np.exp(-field_size/6.132), 3)


def d_LM_max(field_size):
    """
    Compute the maximum optical density of the L- and M-cone photopigments for
    given field size.

    Parameters
    ----------
    field_size : float
        Field size in degrees.

    Returns
    -------
    d_LM_max : float
        The computed maximum optical density of the L- and M-cone
        photopigments
    """
    return my_round(0.38 + 0.54*np.exp(-field_size/1.333), 3)


def d_S_max(field_size):
    """
    Compute the maximum optical density of the S-cone photopigment for given
    field size.

    Parameters
    ----------
    field_size : float
        Field size in degrees.

    Returns
    -------
    d_S_max : float
        The computed maximum optical density of the S-cone visual pigment
    """
    return my_round(0.30 + 0.45*np.exp(-field_size/1.333), 3)


def absorptance(field_size):
    """
    Compute the quantal absorptance of the L, M and S cones for given field
    size.

    Parameters
    ----------
    field_size : float
        Field size in degrees.

    Returns
    -------
    absorpt : ndarray
        The computed quantal absorptances of the L, M and S cones; wavelengths
        in first column.
    """
    absorpt = VisualData.absorbance.copy()  # initialize for in-place editing
    absorpt[:, 1] = 1 - 10**(-d_LM_max(field_size) *
                             10**(VisualData.absorbance[:, 1]))  # L
    absorpt[:, 2] = 1 - 10**(-d_LM_max(field_size) *
                             10**(VisualData.absorbance[:, 2]))  # M
    absorpt[:, 3] = 1 - 10**(-d_S_max(field_size) *
                             10**(VisualData.absorbance[:, 3]))  # S
    return absorpt


def LMS_quantal(field_size, age):
    """
    Compute the quantum_based LMS cone fundamentals for given field size and
    age.

    Parameters
    ----------
    field_size : float
        Field size in degrees.
    age : float
        Age in years.

    Returns
    -------
    LMSq : ndarray
        The computed quantum_based LMS cone fundamentals; wavelengths in first
        column.
    """
    abt = absorptance(field_size)
    docul = d_ocular(age)
    LMSq = abt.copy()           # initialise for in-place editing
    for i in range(1, 4):
        LMSq[:, i] = (abt[:, i] * 10**(-d_mac_max(field_size) * VisualData.macula_rel[:, 1] - docul[:, 1]))
        LMSq[:, i] = LMSq[:, i] / (LMSq[:, i].max())
    return LMSq


def LMS_energy(field_size, age, base=False):
    """
    Compute the energy-based LMS cone fundamentals for given field size and
    age, with either 9 (base) or 6 (standard) number of significant figures.

    Parameters
    ----------
    field_size : float
        Field size in degrees.
    age : float
        Age in years.
    base : boolean
        The returned energy-based LMS cone fundamentals given to the
        precision of 9 sign. figs. if 'True', and to the precision of
        6 sign. figs. if 'False'.

    Returns
    -------
    LMS : ndarray
        The computed energy-based LMS cone fundamentals; wavelengths in first
        column.
    Lo_max, Mo_max, So_max : float
        The computed maximum values of the energy-based LMS cone fundamentals
        before renormalization.
    """
    if age == 32 and np.round(field_size, 1) == 2:
        if base:
            LMS = VisualData.LMS2_lin_energy_9_signfig.copy()
        else:                   # if standard
            LMS = VisualData.LMS2_lin_energy_6_signfig.copy()
        (Lo_max, Mo_max, So_max) = (1, 1, 1)
    elif age == 32 and np.round(field_size, 1) == 10:
        if base:
            LMS = VisualData.LMS10_lin_energy_9_signfig.copy()
        else:                   # if standard
            LMS = VisualData.LMS10_lin_energy_6_signfig.copy()
        (Lo_max, Mo_max, So_max) = (1, 1, 1)
    else:
        (λ, Lq, Mq, Sq) = LMS_quantal(field_size, age).T
        (Lo, Mo, So) = (Lq * λ, Mq * λ, Sq * λ)
        (Lo_max, Mo_max, So_max) = (Lo.max(), Mo.max(), So.max())
        if base:
            (L, M, S) = sign_figs(
                    np.array([Lo / Lo_max, Mo / Mo_max, So / So_max]), 9)
        else:                   # if standard
            (L, M, S) = sign_figs(
                    np.array([Lo / Lo_max, Mo / Mo_max, So / So_max]), 6)
        LMS = np.array([λ, L, M, S]).T
    return (LMS, (Lo_max, Mo_max, So_max))


def relative_L_cone_weight_Vλ_quantal(field_size, age, strategy_2=True):
    """
    Compute the weighting factor of the quantal L-cone fundamental in the
    synthesis of the cone-fundamental-based quantal V(λ) function.

    Parameters
    ----------
    field_size : float
        Field size in degrees.
    age : float
        Age in years.
    strategy_2 : bool
        Use strategy 2 in github issue #121 for computing the weighting factor.
        If false, strategy 3 is applied.

    Returns
    -------
    kLq, : float
        The computed weighting factor of the quantal L cone fundamental in
        the synthesis of the quantal V(λ) function , i.e.
        Vq(λ) = kLq lq_bar((λ) + mq_bar(λ)).
    """
    if strategy_2:
        field_size = 2.
    abt_fs = absorptance(field_size)
    abt_2 = absorptance(2.)
    LMSq_fs_age = LMS_quantal(field_size, age)
    LMSq_2_32 = LMS_quantal(2, 32)
    const_fs_age = (abt_fs[0, 1] * LMSq_fs_age[0, 2] /
                    (abt_fs[0, 2] * LMSq_fs_age[0, 1]))
    const_2_32 = (abt_2[0, 1] * LMSq_2_32[0, 2] /
                  (abt_2[0, 2] * LMSq_2_32[0, 1]))
    kLq_rel = 1.89 * const_fs_age / const_2_32
    return kLq_rel


def Vλ_energy_and_LM_weights(field_size, age):
    """
    Compute the energy-based V(λ) function (starting from energy-based LMS).
    Return both V(λ) and the the corresponding L and M cone weights used
    in the synthesis.

    Parameters
    ----------
    field_size : float
        Field size in degrees.
    age : float
        Age in years.

    Returns
    -------
    Vλ : ndarray
        The energy-based V(λ) function; wavelengths in first column.
    a21, a22 : float
        The computed weighting factors of, respectively, the L and the
        M cone fundamental in the synthesis of energy-based V(λ) function,
        i.e. V(λ) = a21*l_bar(λ) + a22*m_bar(λ)
    """
    if age == 32 and np.round(field_size, 1) == 2:
        Vλ = VisualData.VλLM_2_lin_energy.copy()
        (a21, a22) = (0.68990272, 0.34832189)
    elif age == 32 and np.round(field_size, 1) == 10:
        Vλ = VisualData.VλLM_10_lin_energy.copy()
        (a21, a22) = (0.69283932, 0.34967567)
    else:
        kLq_rel = relative_L_cone_weight_Vλ_quantal(field_size, age)
        (LMS, LMSo_max) = LMS_energy(field_size, age, base=True)
        (λ, L, M) = (LMS.T)[:3]
        (Lo_max, Mo_max) = LMSo_max[:2]
        Vo = kLq_rel * Lo_max * L + Mo_max * M
        Vo_max = Vo.max()
        a21 = my_round(kLq_rel * Lo_max / Vo_max, 8)
        a22 = my_round(Mo_max / Vo_max, 8)
        V = sign_figs(a21 * L + a22 * M, 7)
        Vλ = np.array([λ, V]).T
    return (Vλ, (a21, a22))


def xyz_interpolated_reference_system(field_size, XYZ31_std, XYZ64_std):
    """
    Compute the spectral chromaticity coordinates of the reference system
    by interpolation between correspoding spectral chromaticity coordinates
    of the CIE 1931 XYZ systems and the CIE 1964 XYZ systems.

    Parameters
    ----------
    field_size : float
        The field size in degrees.
    XYZ31_std : ndarray
        The CIE 1931 XYZ colour-matching functions (2°), given at 1 nm
        steps from 360 nm to 830 nm; wavelengths in first column.
    XYZ64_std : ndarray
        The CIE 1964 XYZ colour-matching functions (10°), given at 1 nm
        steps from 360 nm to 830 nm; wavelengths in first column._λ

    Returns
    -------
    chromaticity : ndarray
        The computed interpolated spectral chromaticity coordinates of the
        CIE standard XYZ systems; wavelenghts in first column.
    """
    # Compute the xyz spectral chromaticity coordinates of the CIE standards 
    xyz31 = chrom_coords_µ(XYZ31_std)
    xyz64 = chrom_coords_µ(XYZ64_std)
    # Determine the wavelength parameters of the knots in the CIE 1931 and 
    # CIE 1964 xy diagrams that serve as guide-points for the interpolation 
    # (morphing) between the spectral CIE 1931 chromaticities and the spectral
    # CIE 1964 chromaticities. 
    [λ31, x31, y31] = (xyz31.T)[:3]  
    [λ64, x64, y64] = (xyz64.T)[:3] 
    λ31_knots = np.array([360, 
                          λ31[np.argmin(x31)],
                          λ31[np.argmax(y31)],
                          700,
                          830])
    λ64_knots = np.array([360,
                          λ64[np.argmin(x64)],
                          λ64[np.argmax(y64)],
                          700,
                          830])
    # Determine the wavelength parameters of the knots (guide-points) in the
    # reference diagram (for the field size specified)
    α = (field_size - 2)/8.
    λ_knots = np.array([360.,
                       (1 - α) * λ31[np.argmin(x31)] + α * λ64[np.argmin(x64)],
                       (1 - α) * λ31[np.argmax(y31)] + α * λ64[np.argmax(y64)],
                       700.,
                       830.])
    # λ values
    λ31_func = scipy.interpolate.interp1d(λ_knots, λ31_knots, kind='linear')
    λ64_func = scipy.interpolate.interp1d(λ_knots, λ64_knots, kind='linear')
    λ31_interp = λ31_func(λ31)
    λ64_interp = λ64_func(λ64)
    # x values
    x31_func = scipy.interpolate.interp1d(λ31, x31, kind='cubic')
    x64_func = scipy.interpolate.interp1d(λ64, x64, kind='cubic')
    x31_interp = x31_func(λ31_interp)
    x64_interp = x64_func(λ64_interp)
    x_values = (1-α) * x31_interp + α * x64_interp
    # y values
    y31_func = scipy.interpolate.interp1d(λ31, y31, kind='cubic')
    y64_func = scipy.interpolate.interp1d(λ64, y64, kind='cubic')
    y31_interp = y31_func(λ31_interp)
    y64_interp = y64_func(λ64_interp)
    y_values = (1-α) * y31_interp + α * y64_interp
    # z values
    z_values = 1 - x_values - y_values
    return np.array([λ31, x_values, y_values, z_values]).T


# =============================================================================
# Minimisation function
# =============================================================================

def square_sum(a13, a21, a22, a33, L_spline, M_spline, S_spline, V_spline,
               λ, λ_ref_min, xyz_ref, full_results=False):
    """
    Function to be optimised for determination of element a13 in
    the (non-renormalized) transformation matrix of the linear
    transformation LMS --> XYZ.

    Parameters
    ----------
    a13 : ndarray
        1x1 array with parameter to optimise.
    a21, a22, a33 : float
        Parameters in matrix for LMS to XYZ conversion.
    L_spline, M_spline, S_spline, V_spline: InterPolatedUnivariateSpline
        LMS and V(λ).
    λ : ndarray
        λ values according to chosen step size.
    λ_ref_min : float
        λ value that gives a minimum for the x-coordinate in the
        corresponding reference diagram, i.e. x(λ_ref_min) = x_ref_min.
    xyz_ref : ndarray
        Reference xyz chromaticity coordinates at 1 nm steps.
    full_results : bool
        Return all results or just the computed error.

    Returns
    -------
    err : float
        Computed error.
    trans_mat : ndarray
        Transformation matrix.
    λ_test_min : float
        argmin(x(λ)).
    ok : bool
        Hit the correct minimum wavelength.
    """
    # Stripping reference values in accordance with CIE2006 tables
    xyz_ref_trunk = xyz_ref[30:, 1:].T
    x_ref_min = xyz_ref_trunk[0, :].min()

    # Transformation coefficients (a11 and a12 computed by Mathematica)
    a11 = (((a13 * (1 - x_ref_min) *
             (M_spline(λ_ref_min) * S_spline(λ).sum() -
              S_spline(λ_ref_min) * M_spline(λ).sum())) +
            (x_ref_min *
             (a21 * L_spline(λ_ref_min) + a22 * M_spline(λ_ref_min) +
              a33 * S_spline(λ_ref_min)) * M_spline(λ).sum()) -
            ((1 - x_ref_min) * M_spline(λ_ref_min) * V_spline(λ).sum())) /
           ((1 - x_ref_min) *
            (L_spline(λ_ref_min) * M_spline(λ).sum() -
             M_spline(λ_ref_min) * L_spline(λ).sum())))
    a12 = (((a13 * (1 - x_ref_min) *
             (L_spline(λ_ref_min) * S_spline(λ).sum() -
              S_spline(λ_ref_min) * L_spline(λ).sum())) +
            (x_ref_min *
             (a21 * L_spline(λ_ref_min) + a22 * M_spline(λ_ref_min) +
              a33 * S_spline(λ_ref_min)) * L_spline(λ).sum()) -
            ((1 - x_ref_min) * L_spline(λ_ref_min) * V_spline(λ).sum())) /
           ((1 - x_ref_min) *
            (M_spline(λ_ref_min) * L_spline(λ).sum() -
             L_spline(λ_ref_min) * M_spline(λ).sum())))
    a11 = my_round(a11[0], 8)
    a12 = my_round(a12[0], 8)
    a13 = my_round(a13[0], 8)
    trans_mat = np.array([[a11, a12, a13], [a21, a22, 0], [0, 0, a33]])
    LMS = np.array([L_spline(np.arange(390, 831)),
                    M_spline(np.arange(390, 831)),
                    S_spline(np.arange(390, 831))])
    (X, Y, Z) = sign_figs(np.dot(trans_mat, LMS), 7)
    sumXYZ = X + Y + Z
    xyz = np.array([X / sumXYZ, Y / sumXYZ, Z / sumXYZ])
    err = ((xyz - xyz_ref_trunk)**2).sum()
    λ_test_min = np.arange(390, 831)[xyz[0, :].argmin()]
    ok = (λ_test_min == λ_ref_min)
    if not ok:
        err = err + np.inf
    if full_results:
        return (err, trans_mat, λ_test_min, ok)
    else:
        return err


# =============================================================================
# Specific functions concerning purple-line stimuli
# =============================================================================

def tangent_points_purple_line(chrom_coords_λ, MacLeod_Boynton=False,
                               tristimulus_λ=None,):
    """
    Compute the the chromaticity coordinates and, optionally, also the
    tristimulus values of the stimuli represented at the purple line's
    point of tangency with the spectrum locus.

    Parameters
    ----------
    chrom_coord_λ : ndarray
        The spectral chromaticty coordinates at 0.1 nm steps within
        the specified wavelength range; wavelengths in first column.
    tristimulus_λ : ndarray
        The spectral tristimulus values at 0.1 nm steps within the
        specified wavelength range; wavelengths in first column.
    MacLeod_Boynton : boolean
        If 'True', the parameter chrom_coord_λ is an array of spectral
        MacLeod_Boynton chromaticity coordinates.

    Returns
    -------
    cc_tg_purple : ndarray
        The computed chromaticity coordinates of the stimuli represented
        at the purple line's point of tangency with the spectrum locus;
        corresponding wavelengths in first column.
    tristim_tg_purple : ndarray
        The computed tristimulus values of the stimuli represented
        at the purple line's point of tangency with the spectrum locus;
        corresponding wavelengths in first column.
    """
    cc = chrom_coords_λ
    if MacLeod_Boynton:
        delaunay = Delaunay(cc[:, 1:4:2])
    else:
        delaunay = Delaunay(cc[:, 1:3])
    ind = np.argmax(np.abs(
        delaunay.convex_hull[:, 0] - delaunay.convex_hull[:, 1]))
    cc_tg_purple = np.zeros((2, 3))  # initialise for in-place editing
    if MacLeod_Boynton:
        cc_tg_purple[0, 0] = cc[delaunay.convex_hull[ind, 0], 0]
        cc_tg_purple[0, 1] = cc[delaunay.convex_hull[ind, 0], 1]
        cc_tg_purple[0, 2] = cc[delaunay.convex_hull[ind, 0], 3]
        cc_tg_purple[1, 0] = cc[delaunay.convex_hull[ind, 1], 0]
        cc_tg_purple[1, 1] = cc[delaunay.convex_hull[ind, 1], 1]
        cc_tg_purple[1, 2] = cc[delaunay.convex_hull[ind, 1], 3]
    else:
        cc_tg_purple[0, :3] = cc[delaunay.convex_hull[ind, 0], :3]
        cc_tg_purple[1, :3] = cc[delaunay.convex_hull[ind, 1], :3]
    if tristimulus_λ is None:
        return cc_tg_purple
    else:
        ts = tristimulus_λ
        ts_tg_purple = np.zeros((2, 4))  # initialise for in-place editing
        ts_tg_purple[0, :4] = ts[delaunay.convex_hull[ind, 0], :4]
        ts_tg_purple[1, :3] = ts[delaunay.convex_hull[ind, 1], :3]
        return (cc_tg_purple, ts_tg_purple)


def XYZ_purples(xyz_λ, xyz_E, XYZ_tg_purple_line):
    """
    Compute the cone-fundamental-based tristimulus XYZ values of purple-line
    stimuli as parameterized by complementary wavelengths.

    Parameters
    ----------
    xyz_λ : ndarray
        The cone-fundamental-based xyz spectral chromaticty coordinates at
        the specified wavelengths; wavelengths in first column.
    xyz_E : ndarray
        The cone-fundamental-based xyz chromaticity coordinates of the
        cardinal white.
    XYZ_tg_purple_line : ndarray
        The cone-fundamental-based XYZ tristimulus values of the stimuli
        represented at the purple-line termini; wavelengths in first column.

    Return
    -------
    XYZ_λc : ndarray
        The computed cone-fundamental-based XYZ tristimulus values of stimuli
        represented on the purple line, parameterized by complementary
        wavelength; complementary wavelengths in first column.
    """
    (x_E, y_E) = xyz_E[:2]
    (X_B, Y_B, Z_B) = XYZ_tg_purple_line[0, 1:]  # short-wavelength terminus
    (X_R, Y_R, Z_R) = XYZ_tg_purple_line[1, 1:]  # long-wavelength terminus
    XYZ_λc = []
    inside = False
    for i in range(len(xyz_λ[:, 0])):
        λc = my_round(xyz_λ[i, 0], 1)
        if (λc > my_round(XYZ_tg_purple_line[0, 0], 1) and
                λc < my_round(XYZ_tg_purple_line[1, 0], 1)):
            (x, y) = xyz_λ[i, 1:3]
            # Parameter for the convex linear combination of the tristimulus
            # values of the stimuli represented at the purple-line termini
            # (determined by Mathematica):
            α = (1 /
                 (1 - ((y - y_E) * X_B - (x - x_E) * Y_B +
                       (x * y_E - y * x_E) * (X_B + Y_B + Z_B)) /
                  ((y - y_E) * X_R - (x - x_E) * Y_R +
                   (x * y_E - y * x_E) * (X_R + Y_R + Z_R))))
            if α >= 0 and α <= 1:
                inside = True
                X = α * X_B + (1 - α) * X_R
                Y = α * Y_B + (1 - α) * Y_R
                Z = α * Z_B + (1 - α) * Z_R
                XYZ_λc.append([λc, X, Y, Z])
            elif inside:
                break
    return np.array(XYZ_λc)


# =============================================================================
# Functions for calculation of the items listed in the GUI drop-down menu
# =============================================================================

#    The functions are given to different precicions, at different 
#    wavelength steps, within different wavelength domains. 
#    For the variable names, the following nomenclature is used:
#        
#    '_base' LMS    : 9 sign. figs.
#            logLMS : 8 decimal places      
#    '_std'  LMS    : 6 sign. figs. 
#            logLMS : 5 decimal places
#            lms_mb : 6 decimal places  (mb: MacLeod‒Boynton)
#            lms_mw : 6 decimal places  (mw: Maxwellian)
#            XYZ    : 7 sign. figs. 
#            xyz    : 5 decimal places
#            Vλ     : 7 sign. figs              
#    '_all'         : values given at 0.1 nm steps from 390 nm to 830 nm
#    '_main'        : values given at 1 nm steps from 390 nm to 830 nm
#    '_spec'        : values given at specified wavelengths
#    '_plot'        : values given at 0.1 nm steps within specified domain


def compute_LMS(λ, L_spline, M_spline, S_spline, base=False):
    """
    Compute the LMS cone fundamentals for given wavelengths, both as
    linear and logarithmic values to respective specified precisions.

    Parameters
    ----------
    λ : ndarray
        The wavelengths for which the LMS cone fundamentals are to be
        calculated.
    L_spline, M_spline, S_spline :
        Spline-interpolation functions for the LMS cone fundamentals
        (on a linear scale).
    base : boolean
        The returned energy-based LMS values are given to the precision of
        9 sign. figs. / 8 decimal points if 'True', and to the precision of
        6 sign. figs. / 5 decimal points if 'False'.

    Returns
    -------
    LMS : ndarray
        The computed LMS cone fundamentals;
        wavelengths in first column.
    logLMS : ndarray
        The computed Briggsian logarithms (of the LMS cone fundamentals;
        wavelengths in first column.
    """

    if base:
        LMS_sf = 9
        logLMS_dp = 8
    else:
        LMS_sf = 6
        logLMS_dp = 5
    # Compute linear values
    (L, M, S) = np.array([sign_figs(L_spline(λ), LMS_sf),
                          sign_figs(M_spline(λ), LMS_sf),
                          sign_figs(S_spline(λ), LMS_sf)])
    LMS = chop(np.array([λ, L, M, S]).T) 

    # Compute logarithmic values
    logLMS = LMS.copy()  # initialize for in-line editing
    logLMS[:, 1:][logLMS[:, 1:] == 0] = -np.inf
    logLMS[:, 1:][logLMS[:, 1:] > 0] = my_round(
        np.log10(logLMS[:, 1:][logLMS[:, 1:] > 0]), logLMS_dp)
    return (LMS, logLMS)


def compute_MacLeod_Boynton_diagram(LMS_spec, LMS_plot, LMS_all,
                                    Vλ_all, Vλ_spec, LM_weights):
    """
    Compute the MacLeod‒Boynton chromaticity cooordinates for the spectral
    stimuli, Illuminant E and the stimuli represented at the purple line's
    point of tangency with the spectrum locus.

    Parameters
    ----------
    LMS_spec : ndarray
        Table of LMS values at specified wavelengths, given to base-value
        precision (i.e. 9 sign. figs.); wavelengths in first column.
    LMS_plot : ndarray
        Table of LMS values at 0.1 nm steps within the specified wavelength
        domain, given to base-value precision (i.e. 9 sign. figs);
        wavelengths in first column.
    LMS_all : ndarray
        Table of LMS values at 0.1 nm steps from 390 nm to 830 nm, given to
        base-value precision (i.e. 9 sign. figs.); wavelengths in first column.
    Vλ_all : ndarray
        Table of Vλ values at 0.1 nm steps from 390 nm to 830 nm, given to
        base-value precision (i.e. 9 sign. figs.) for age 32 AND field size
        2° OR 10°, and to standard-value precision (i.e. 7 sign. figs.) in
        other cases; wavelengths in first column.
    Vλ_spec : ndarray
        Table of Vλ values at specified wavelengths, given to base-value
        precision (i.e. 9 sign. figs.) for age 32 AND field size 2° OR 10°,
        and to standard-value precision (i.e. 7 sign. figs.) in other
        cases; wavelengths in first column.
    LM_weights : ndarray
        The weighting factors kL and kM in the synthesis of the cone-
        fundamental-based V(λ)-function, i.e. V(λ) = kL*l_bar(λ) + kM*m_bar(λ).

    Returns
    -------
    κL, κM, κS : ndarray
        the normalization coefficients (scaling factors) in the equations
        l_mw = κL * L / (κL * L_spec + κM * M_spec)
        m_mw = κM * M / (κL * L_spec + κM * M_spec)
        s_mw = κS * S / (κL * L_spec + κM * M_spec)
        defining the MacLeod‒Boynton lms chromaticity coordinates in terms
        of the (unity-peak-normalized) LMS cone fundamentals.
    lms_mb_spec : ndarray
        The spectral MacLeod‒Boynton lms chromaticity coordinates for the
        tabulated wavelengths, given to standard precision; wavelengths in
        first column (for table).
    lms_mb_E : ndarray
        The MacLeod‒Boynton lms chromaticity coordinates of Illuminant E,
        given to standard precision (for description).
    lms_mb_tg_purple : ndarray
        The MacLeod‒Boynton lms chromaticity coordinates at the purple line's
        points of tangency with the spectrum locus, given to standard
        precision (for description).
    lms_mb_plot : ndarray
        The spectral MacLeod‒Boynton lms chromaticity coordinates at 0.1 nm
        steps within the specified wavelength domain; wavelengths in first
        column (for plot).
    lms_mb_E_plot : ndarray
        The MacLeod‒Boynton lms chromaticity coordinates of illuminant E
        (for plot).
    lms_mb_tg_purple_plot : ndarray
        The MacLeod‒Boynton lms chromaticity coordinates at the purple line's
        points of tangency with the spectrum locus (for plot).

    """
    # '_mb'   : MacLeod‒Boynton
    # '_all'  : values given at 0.1 nm steps from 390 nm to 830 nm
    # '_spec' : values given at specified wavelengths
    # '_plot' : values given at 0.1 nm steps within specified domain
    
    (λ_spec, L_spec, M_spec, S_spec) = LMS_spec.T
    (λ_plot, L_plot, M_plot, S_plot) = LMS_plot.T 
    S_all = (LMS_all.T)[3]
    V_all = (Vλ_all.T)[1]
    V_spec = (Vλ_spec.T)[1]
    (κL, κM) = LM_weights           # k: kappa (greek letter)
    κS = 1 / np.max(S_all / V_all)
    # Compute spectral chromomaticity coordinates (for table)
    lms_mb_spec = np.array([λ_spec,
                            κL * L_spec / V_spec,
                            κM * M_spec / V_spec,
                            κS * S_spec / V_spec]).T
    lms_mb_spec[:,1:] = my_round(lms_mb_spec[:,1:], 6)
    # Compute plot points for spectrum locus
    V_plot = sign_figs(κL * L_plot + κM * M_plot, 7)
    lms_mb_plot = np.array([λ_plot,
                            κL * L_plot / V_plot,
                            κM * M_plot / V_plot,
                            κS * S_plot / V_plot]).T 
    # Compute white point (for description and plot)
    [L_mb_E, M_mb_E, S_mb_E] = [κL * np.sum(L_spec),
                                κM * np.sum(M_spec),
                                κS * np.sum(S_spec)]
    V_E = sign_figs(np.array(L_mb_E + M_mb_E), 7)
    lms_mb_E_plot = np.array([L_mb_E / V_E,
                              M_mb_E / V_E,
                              S_mb_E / V_E])
    lms_mb_E = my_round(lms_mb_E_plot, 6)
    # Compute purple-line tangent points (for description and plot)
    lms_mb_tg_purple_plot = tangent_points_purple_line(
            lms_mb_plot, MacLeod_Boynton=True)
    lms_mb_tg_purple = lms_mb_tg_purple_plot.copy()
    lms_mb_tg_purple[:, 0] = my_round(lms_mb_tg_purple[:, 0], 1)
    lms_mb_tg_purple[:, 1:] = my_round(lms_mb_tg_purple[:, 1:], 6)
    return (np.array([κL, κM, κS]),
            lms_mb_spec, lms_mb_E, lms_mb_tg_purple,
            lms_mb_plot, lms_mb_E_plot, lms_mb_tg_purple_plot)


def compute_Maxwellian_diagram(LMS_spec, LMS_plot):
    """
    Compute the Maxwellian chromaticity cooordinates for the spectral
    stimuli, Illuminant E and the stimuli represented at the purple
    line's points of tangency with the spectrum locus.

    Parameters
    ----------
    LMS_spec : ndarray
        Table of LMS values at specified wavelengths, given to base-value
        precision (i.e. 9 sign. figs.); wavelengths in first column.
    LMS_plot : ndarray
        Table of LMS values at 0.1 nm steps within the specified wavelength
        domain, given to base-value precision (i.e. 9 sign. figs.);
        wavelengths in first column.

    Returns
    -------
    kL, kM, kS : ndarray
        The normalization coefficients (scaling factors) kL, kM and kS in
        the equations
        l_mw = kL*L / (kL*L + kM*M + kS*S)
        m_mw = kM*M / (kL*L + kM*M + kS*S)
        s_mw = kS*S / (kL*L + kM*M + kS*S)
        defining the Maxwellian lms chromaticity coordinates in terms
        of the (unity-peak-normalized) LMS cone fundamentals.
    lms_mw_spec : ndarray
        The spectral Maxwellian lms chromaticity coordinates for the
        tabulated wavelengths, given to standard/specified precision;
        wavelengths in first column (for table)
    lms_mw_E : ndarray
        The Maxwellian lms chromaticity coordinates of Illuminant E,
        given to standard/specified precision (for description).
    lms_mb_tg_purple : ndarray
        The Maxwellian lms chromaticity coordinates at the purple
        line's points of tangency with the spectrun locus, given to
        standard precision (for description).
    lms_mb_plot : ndarray
        The spectral Maxwellian lms chromaticity coordinates at 0.1 nm
        steps within the specified wavelength domain; wavelengths in
        first column (for plot).
    lms_mb_E_plot : ndarray
        The Maxwellian lms chromaticity coordinates of Illuminant E
        (for plot).
    lms_mb_tg_purple_plot : ndarray
        The Maxwellian lms chromaticity coordinates at the purple
        line's points of tangency with the spectrum locus (for plot).
    """
    # '_mw'   : Maxwellian
    # '_spec' : values given at specified wavelengths
    # '_plot' : values given at 0.1 nm steps within specified domain
    
    (λ_spec, L_spec, M_spec, S_spec) = LMS_spec.T
    (λ_plot, L_plot, M_plot, S_plot) = LMS_plot.T
    # Compute spectral chromaticity coordinates (for table)
    (kL, kM, kS) = (1./np.sum(L_spec), 1./np.sum(M_spec), 1./np.sum(S_spec))
    LMS_spec_N = np.array([λ_spec, kL * L_spec, kM * M_spec, kS * S_spec]).T
    lms_mw_spec = chrom_coords_µ(LMS_spec_N)
    lms_mw_spec[:,1:] = my_round(lms_mw_spec[:,1:], 6)
    # Compute plot points for spectrum locus
    (cL, cM, cS) = (1./np.sum(L_plot), 1./np.sum(M_plot), 1./np.sum(S_plot))
    LMS_plot_N = np.array([λ_plot, cL * L_plot, cM * M_plot, cS * S_plot]).T
    lms_mw_plot = chrom_coords_µ(LMS_plot_N)
    # Compute chromaticity coordinates of Ill. E (for description and plot)
    lms_mw_E_plot = chrom_coords_E(LMS_spec_N)
    lms_mw_E = my_round(lms_mw_E_plot, 6) 
    # Compute purple-line tangent points (for description and plot) 
    lms_mw_tg_purple_plot = tangent_points_purple_line(lms_mw_plot)
    lms_mw_tg_purple = lms_mw_tg_purple_plot.copy()
    lms_mw_tg_purple[:, 0] = my_round(lms_mw_tg_purple[:, 0], 1)
    lms_mw_tg_purple[:, 1:] = my_round(lms_mw_tg_purple[:, 1:], 6)
    return (np.array([kL, kM, kS]),
            lms_mw_spec, lms_mw_E, lms_mw_tg_purple,
            lms_mw_plot, lms_mw_E_plot, lms_mw_tg_purple_plot)


def compute_XYZ(L_spline, M_spline, S_spline, V_spline,
                LMS_spec, LMS_plot, LMS_all,
                LM_weights, xyz_reference):
    """
    Compute the CIE cone-fundamental-based XYZ tristimulus functions.

    Parameters
    ----------
    L_spline, M_spline, S_spline :
        Spline-interpolation functions for the LMS cone fundamentals
        (on a linear scale).
    V_spline :
        Spline-interpolation functions for the cone-fundamental-based
        V(λ)-function (on a linear scale).
    LMS_spec : ndarray
        Table of LMS values at specified wavelengths, given to base-value
        precision (i.e. 9 sign. figs); wavelengths in first column.
    LMS_plot : ndarray
        Table of LMS values at 0.1 nm steps within the specified wavelength
        domain, given to base-value precision (i.e. 9 sign. figs.);
        wavelengths in first column.
    LMS_all : ndarray
        Table of LMS values at 0.1 nm steps from 390 nm to 830 nm,
        given to base-value precision (i.e. 9 sign. figs.); wavelengths
        in first column.
    LM_weights : ndarray
        The weighting factors kL and kM in the synthesis of the
        cone-fundamental-based V(λ)-function, i.e.
        V(λ) = kL * l_bar(λ) + kM * m_bar(λ).
    xyz_reference : ndarray
        The spectral chromaticity coordinates of the reference system
        (obtained by shape-morphing (interpolation) between the CIE 1931
        standard and the CIE 1964 standard).

    Returns
    -------
    trans_mat : ndarray
        The non-renormalized transformation matrix of the linear
        transformation LMS --> XYZ.
    XYZ_spec : ndarray
        The non-renormalized CIE cone-fundamental-based XYZ spectral
        tristimulus values for the tabulated wavelengths, given to
        standard/specified precision; wavelengths in first column
        (for table).
    XYZ_plot : ndarray
        The non-renormalized CIE cone-fundamental-based XYZ spectral
        tristimulus values at 0.1 nm steps within the specified
        wavelength domain, given to standard/specified precision;
        wavelengths in first column (for table).
    trans_mat_N : ndarray
        The renormalized transformation matrix of the linear
        transformation LMS --> XYZ.
    XYZ_spec_N : ndarray
        The renormalized CIE cone-fundamental-based XYZ spectral
        tristimulus values for the tabulated wavelengths, given to
        standard/specified precision; wavelengths in first column
        (for table).
    XYZ_plot_N : ndarray
        The renormalized CIE cone-fundamental-based XYZ spectral
        tristimulus values at 0.1 nm steps within the specified
        wavelength domain, given to standard/specified precision;
        wavelengths in first column (for table).
    """
    # '_all'  : values given at 0.1 nm steps from 390 nm to 830 nm
    # '_main' : values given at 1 nm steps from 390 nm to 830 nm
    # '_spec' : values given at specified wavelengths
    # '_plot' : values given at 0.1 nm steps within specified domain

    xyz_ref = xyz_reference
    (a21, a22) = LM_weights
    LMS_main = LMS_all[::10]
    (λ_main, L_main, M_main, S_main) = LMS_main.T
    V_main = sign_figs(a21 * L_main + a22 * M_main, 7)
    a33 = my_round(V_main.sum() / S_main.sum(), 8)
    
    # Compute optimised non-renormalised transformation matrix
    λ_x_min_ref = 502
    ok = False
    while not ok:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a13 = scipy.optimize.fmin(
                square_sum, 0.39, (a21, a22, a33,
                                   L_spline, M_spline, S_spline,
                                   V_spline,
                                   λ_main, λ_x_min_ref,
                                   xyz_ref, False),
                xtol=10**(-(10)), disp=False)  # exp: -(mat_dp + 2) = -10 
                
        trans_mat, λ_x_min_ref, ok = (
            square_sum(a13, a21, a22, a33, 
                       L_spline, M_spline, S_spline, 
                       V_spline, 
                       λ_main, λ_x_min_ref, 
                       xyz_ref, True)[1:]) 

    # Compute renormalized transformation matrix
    (λ_spec,
     X_exact_spec,
     Y_exact_spec,
     Z_exact_spec) = linear_transformation_λ(trans_mat, LMS_spec).T
    if ((λ_spec[0] == 390. and λ_spec[-1] == 830.) and
        (my_round(λ_spec[1] - λ_spec[0], 1) ==
         1.0)):
        trans_mat_N = trans_mat
    else:
        (X_exact_sum, Y_exact_sum, Z_exact_sum) = (np.sum(X_exact_spec),
                                                   np.sum(Y_exact_spec),
                                                   np.sum(Z_exact_spec))
        trans_mat_N = my_round(trans_mat * ([Y_exact_sum / X_exact_sum], 
                                            [1], 
                                            [Y_exact_sum / Z_exact_sum]), 8) 
    
    
    # Compute spectral tristimulus values (for table)
    ### non-renormalized:
    XYZ_spec = linear_transformation_λ(trans_mat, LMS_spec)  
    XYZ_spec[:, 1:] = sign_figs(XYZ_spec[:, 1:], 7)
    ### renormalized:
    XYZ_spec_N = linear_transformation_λ(trans_mat_N, LMS_spec)   
    XYZ_spec_N[:, 1:] = sign_figs(XYZ_spec_N[:, 1:], 7)
    # Compute plot points for tristimulus functions (for plot)
    ### non-renormalized:
    XYZ_plot = linear_transformation_λ(trans_mat, LMS_plot)   
    XYZ_plot[:, 1:] = sign_figs(XYZ_plot[:, 1:], 7)
    ### renormalized:
    XYZ_plot_N = linear_transformation_λ(trans_mat_N, LMS_plot)   
    XYZ_plot_N[:, 1:] = sign_figs(XYZ_plot_N[:, 1:], 7) 
    return (trans_mat, XYZ_spec, XYZ_plot, 
            trans_mat_N, XYZ_spec_N, XYZ_plot_N)


def compute_xy_diagram(XYZ_spec, XYZ_plot, XYZ_spec_N, XYZ_plot_N):
    """
    Compute the CIE cone-fundamental-based xyz chromaticity cooordinates for
    the spectral stimuli, Illuminant E and the stimuli represented at the
    purple line's points of tangency with the spectrum locus.

    Parameters
    ----------
    XYZ_spec : ndarray
        The non-renormalized CIE cone-fundamental-based XYZ spectral
        tristimulus values for the specified wavelengths, given to the
        precision of 7 sign. figs.; wavelengths in first column.
    XYZ_plot : ndarray
        The non-renormalized CIE cone-fundamental-based XYZ spectral
        tristimulus values at 0.1 nm steps within the specified wavelength
        domain, given to the precision of 7 sign. figs.; wavelengths in first
        column.
    XYZ_spec_N : ndarray
        The renormalized CIE cone-fundamental-based XYZ spectral tristimulus
        values for the tabulated wavelengths, given to given the precision of
        7 sign. figs.; wavelengths in first column.
    XYZ_plot_N : ndarray
        The renormalized CIE cone-fundamental-based XYZ spectral tristimulus
        values at 0.1 nm steps within the specified wavelength domain, given
        to the precision of 7 sign. figs.; wavelengths in first column.

    Returns
    -------
    xyz_spec : ndarray
        The non-renormalized CIE cone-fundamental-based xyz spectral
        chromaticity coordinates for the specified wavelengths, given to the
        precision of 5 decimal places; wavelengths in first column (for table).
    xyz_E : ndarray
        The non-renormalized CIE cone-fundamental-based xyz chromaticity
        coordinates of Illuminant E, given to the precision of 5 decimal
        places (for description).
    xyz_tg_purple : ndarray
        The non-renormalized CIE cone-fundamental-based xyz chromaticity
        coordinates at the purple line's points of tangency with the spectrum
        locus, given to the precision of 5 decimal places; wavelengths in
        first column (for description).
    XYZ_tg_purple : ndarray
        The non-renormalized CIE cone-fundamental-based XYZ tristimulus values
        at the purple line's points of tangency with the spectrun locus, given
        to the precision of 7 sign. figs.; wavelengths in first column (for
        further calculations).
    xyz_plot : ndarray
        The non-renormalized CIE cone-fundamental-based xyz spectral
        chromaticity coordinates at 0.1 nm steps within the specified
        wavelength domain; wavelengths in first column (for plot).
    xyz_E_plot : ndarray
        The non-renormalized CIE cone-fundamental-based xyz chromaticity
        coordinates of Illuminant E (for plot).
    xyz_tg_purple_plot : ndarray
        The non-renormalized CIE cone-fundamental-based xyz chromaticity
        coordinates at the purple line's points of tangency with the spectrum
        locus; wavelengths in first column (for plot).
    XYZ_tg_purple_plot : ndarray
        The non-renormalized CIE cone-fundamental-based XYZ tristimulus values
        at the purple line's points of tangency with the spectrun locus (non-
        rounded values); wavelengths in first column (for further calculation).
    xyz_spec_N : ndarray
        The renormalized CIE cone-fundamental-based xyz spectral chromaticity
        coordinates for the specified wavelengths, given to the precision of
        5 decimal places; wavelengths in first column (for table).
    xyz_E_N : ndarray
        The renormalized CIE cone-fundamental-based xyz chromaticity
        coordinates of Illuminant E, given to the precision 5 decimal places
        (for description).
    xyz_tg_purple_N : ndarray
        The renormalized CIE cone-fundamental-based xyz chromaticity
        coordinates at the purple line's points of tangency with the spectrum
        locus, given to the precision 5 decimal places; wavelengths in first
        column (for description).
    XYZ_tg_purple_N : ndarray
        The renormalized CIE cone-fundamental-based XYZ tristimulus values at
        the purple line's points of tangency with the spectrun locus, given to
        to the precision of 7 sign. figs.; wavelengths in first column (for
        further calculation).
    xyz_plot_N : ndarray
        The renormalized CIE cone-fundamental-based xyz spectral chromaticity
        coordinates at 0.1 nm steps within the specified wavelength domain;
        wavelengths in first column (for plot).
    xyz_E_plot_N : ndarray
        The renormalized CIE cone-fundamental-based xyz chromaticity
        coordinates of Illuminant E (for plot).
    xyz_tg_purple_plot_N : ndarray
        The renormalized CIE cone-fundamental-based xyz chromaticity
        coordinates at the purple line's points of tangency with the spectrum
        locus; wavelengths in first column (for plot).
    XYZ_tg_purple_plot_N : ndarray
        The renormalized CIE cone-fundamental-based XYZ tristimulus values at
        the purple line's points of tangency with the spectrun locus (non-
        rounded values); wavelengths in first column (for further calculation).
    """
    # '_spec' : values given at specified wavelengths
    # '_plot' : values given at 0.1 nm steps within specified domain

    # Compute spectral chromaticity coordinates (for table)
    # non-renormalized:
    xyz_spec = chrom_coords_µ(XYZ_spec)
    xyz_spec[:, 1:] = my_round(xyz_spec[:, 1:], 5)
    ### renormalized:
    xyz_spec_N = chrom_coords_µ(XYZ_spec_N)
    xyz_spec_N[:, 1:] = my_round(xyz_spec_N[:, 1:], 5)
    # Compute plot points for spectrum locus (for plot)   
    ### non-renormalized:
    xyz_plot = chrom_coords_µ(XYZ_plot) 
    ### renormalized     
    xyz_plot_N = chrom_coords_µ(XYZ_plot_N)
    # Compute chromaticity coordinates of Ill. E (for description and plot)
    # non-renormalized:
    xyz_E_plot = chrom_coords_E(XYZ_spec)
    xyz_E = my_round(xyz_E_plot, 5)
    ### renormalized:
    xyz_E_plot_N = chrom_coords_E(XYZ_spec_N)
    xyz_E_N = my_round(xyz_E_plot_N, 5)
    # Compute purple-line tangent points (for description and plot) 
    ### non-renormalized:     
    (xyz_tg_purple_plot,
     XYZ_tg_purple_plot) = tangent_points_purple_line(
             xyz_plot, False, XYZ_plot)
    xyz_tg_purple = xyz_tg_purple_plot.copy()
    xyz_tg_purple[:, 0] = my_round(xyz_tg_purple[:, 0], 1)
    xyz_tg_purple[:, 1:] = my_round(xyz_tg_purple[:, 1:], 5)
    XYZ_tg_purple = XYZ_tg_purple_plot.copy()                # tg XYZ-space
    XYZ_tg_purple[:, 0] = my_round(XYZ_tg_purple[:, 0], 1)   # tg XYZ-space
    XYZ_tg_purple[:, 1:] = my_round(XYZ_tg_purple[:, 1:], 7) # tg XYZ-space
    ### renormalized:
    (xyz_tg_purple_plot_N,
     XYZ_tg_purple_plot_N) = tangent_points_purple_line(
             xyz_plot_N, False, XYZ_plot_N)
    xyz_tg_purple_N = xyz_tg_purple_plot_N.copy()
    xyz_tg_purple_N[:, 0] = my_round(xyz_tg_purple_N[:, 0], 1)
    xyz_tg_purple_N[:, 1:] = my_round(xyz_tg_purple_N[:, 1:], 5)
    XYZ_tg_purple_N = XYZ_tg_purple_plot_N.copy()                # tg XYZ-space
    XYZ_tg_purple_N[:, 0] = my_round(XYZ_tg_purple_N[:, 0], 1)   # tg XYZ-space
    XYZ_tg_purple_N[:, 1:] = my_round(XYZ_tg_purple_N[:, 1:], 7) # tg XYZ-space
    return (xyz_spec, xyz_E, xyz_tg_purple, XYZ_tg_purple, 
            xyz_plot, xyz_E_plot, xyz_tg_purple_plot, XYZ_tg_purple_plot,
            xyz_spec_N, xyz_E_N, xyz_tg_purple_N, XYZ_tg_purple_N,
            xyz_plot_N, xyz_E_plot_N, xyz_tg_purple_plot_N,
            XYZ_tg_purple_plot_N)


def compute_XYZ_purples(xyz_spec, xyz_E, XYZ_tg_purple,
                        xyz_plot, xyz_E_plot, XYZ_tg_purple_plot,
                        xyz_spec_N, xyz_E_N, XYZ_tg_purple_N,
                        xyz_plot_N, xyz_E_plot_N, XYZ_tg_purple_plot_N):
    """
    Compute the XYZ cone-fundamental-based tristimulus functions of purple-
    line stimuli as parameterized by complementary wavelengths.

    Parameters
    ----------
    xyz_spec : ndarray
        The non-renormalized xyz cone-fundamental-based spectral chromaticity
        coordinates at specified wavelengths; wavelengths in first column.
    xyz_E : ndarray
        The non-renormalized xyz cone-fundamental-based chromaticity
        coordinates of Illuminant E.
    XYZ_tg_purple : ndarray
        The non-renormalized XYZ cone-fundamental-based tristimulus values
        of the stimuli represented at the purple-line termini; wavelengths in
        first column.
    xyz_plot : ndarray
        The non-renormalized xyz cone-fundamental-based spectral chromaticity
        coordinates at 0.1 nm steps within the specified wavelength domain;
        wavelengths in first column.
    xyz_E_plot : ndarray
        The non-renormalized xyz cone-fundamental-based chromaticity
        coordinates (non-rounded) of Illuminant E.
    XYZ_tg_purple_plot : ndarray
        The non-renormalized XYZ cone-fundamental-based tristimulus values
        (non-rounded) of the stimuli represented at the purple-line termini;
        wavelengths in first column.
    xyz_N : ndarray
        The renormalized xyz cone-fundamental-based spectral chromaticity
        coordinates at the specified wavelengths;wavelengths in first column.
    xyz_E_N : ndarray
        The renormalized xyz cone-fundamental-based chromaticity coordinates
        of Illuminant E.
    XYZ_tg_purple_N : ndarray
        The renormalized XYZ cone-fundamental-based tristimulus values of the
        stimuli represented at the purple-line termini; wavelengths in first
        column.
    xyz_plot_N : ndarray
        The renormalized xyz cone-fundamental-based spectral chromaticity
        coordinates at 0.1 nm steps within the specified wavelength domain;
        wavelengths in first column.
    xyz_E_plot_N : ndarray
        The renormalized xyz cone-fundamental-based chromaticity coordinates
        (non-rounded) of Illuminant E.
    XYZ_tg_purple_plot_N : ndarray
        The renormalized XYZ cone-fundamental-based tristimulus values (non-
        rounded) of the stimuli represented at the purple-line termini;
        corresponding wavelengths in first column.

    Returns
    -------
    XYZ_purples : ndarray
        The computed non-renormalized XYZ cone-fundamental-based tristimulus
        values of stimuli represented on the purple line, parameterized by
        complementary wavelength; complementary wavelengths in first column.
    XYZ_purples_N : ndarray
        The computed renormalized XYZ cone-fundamental-based tristimulus
        values of stimuli represented on the purple line, parameterized by
        complementary wavelength; complementary wavelengths in first column.
    """
    # '_spec' : values as parameterized by complementary wavelength,
    #           for specified wavelength steps
    # '_plot' : values as parameterized by complementary wavelength,
    #           for 0.1 nm wavelength steps

    # Compute tristimulus functions for purple-line stimuli (for table)
    # non-renormalized:
    XYZ_purples_spec = XYZ_purples(xyz_spec, xyz_E, XYZ_tg_purple)
    # renormalized:
    XYZ_purples_spec_N = XYZ_purples(xyz_spec_N, xyz_E_N, XYZ_tg_purple_N)
    # Compute plot points for tristimulus functions for purple line-stimuli
    ### non-renormalized:
    XYZ_purples_plot = XYZ_purples(
            xyz_plot, xyz_E_plot, XYZ_tg_purple_plot)
    # renormalized:
    XYZ_purples_plot_N = XYZ_purples(
            xyz_plot_N, xyz_E_plot_N, XYZ_tg_purple_plot_N)
    return (XYZ_purples_spec, XYZ_purples_plot,
            XYZ_purples_spec_N, XYZ_purples_plot_N)


def compute_xyz_purples(XYZ_purples_spec, XYZ_purples_plot,
                        XYZ_purples_spec_N, XYZ_purples_plot_N):
    """
    Compute the xyz cone-fundamental-based tristimulus values of
    purple-line stimuli as parameterized by complementary wavelength.

    Parameters
    ----------
    XYZ_purples_spec : ndarray
        The computed non-renormalized XYZ cone-fundamental-based
        tristimulus values of stimuli represented on the purple line,
        parameterized by complementary wavelength, at specified intervals;
        complementary wavelengths in first column.
    XYZ_purples_spec_N : ndarray
        The computed renormalized XYZ cone-fundamental-based
        tristimulus values of stimuli represented on the purple line,
        parameterized by complementary wavelength, at specified intervals;
        complementary wavelengths in first column.
    XYZ_purples_plot : ndarray
        The computed non-renormalized XYZ cone-fundamental-based
        tristimulus values of stimuli represented on the purple line,
        parameterized by complementary wavelengths, at 0.1 nm steps;
        complementary wavelength in first column.
    XYZ_purples_plot_N : ndarray
        The computed renormalized XYZ cone-fundamental-based
        tristimulus values of stimuli represented on the purple line,
        parameterized by complementary wavelengths, at 0.1 nm steps;
        complementary wavelength in first column.

    Returns
    -------
    xyz_purples_spec : ndarray
        The computed non-renormalized xyz cone-fundamental-based chromaticity
        coordinates of stimuli represented on the purple line, given to the
        precision of 5 decimal places, parameterized by complementary
        wavelength, at specified intervals; complementary wavelengths in first
        column.
    xyz_purples_spec_N : ndarray
        The computed renormalized xyz cone-fundamental-based chromaticity
        coordinates of stimuli represented on the purple line, given to the
        precision of 5 decimal places, parameterized by complementary
        wavelength, at specified intervals; complementary wavelengths in first
        column.
    xyz_purples_plot : ndarray
        The computed non-renormalized xyz cone-fundamental-based chromaticity
        coordinates of stimuli represented on the purple line, parameterized
        by complementary wavelengths, at 0.1 nm steps; complementary
        wavelength in first column.
    xyz_purples_plot_N : ndarray
        The computed renormalized xyz cone-fundamental-based chromaticity
        coordinates of stimuli represented on the purple line, parameterized
        by complementary wavelengths, at 0.1 nm steps; complementary
        wavelength in first column.
    """
    # '_spec' : values as parameterized by complementary wavelength,
    #          at specified wavelength intervals
    # '_plot' : values as parameterized by complementary wavelength,
    #          at 0.1 nm wavelength intervals

    # Compute chromaticity coordinates of purple-line stimuli (for table)
    # non-renormalized:
    xyz_purples_spec = chrom_coords_µ(XYZ_purples_spec)
    xyz_purples_spec[:, 1:] = my_round(xyz_purples_spec[:, 1:], 5)
    # renormalized:
    xyz_purples_spec_N = chrom_coords_µ(XYZ_purples_spec_N)
    xyz_purples_spec_N[:, 1:] = my_round(xyz_purples_spec_N[:, 1:], 5)
    # Compute plot points for purple-line
    ### non-renormalized:
    xyz_purples_plot = chrom_coords_µ(XYZ_purples_plot)
    ### renormalized:
    xyz_purples_plot_N = chrom_coords_µ(XYZ_purples_plot_N) 
    return (xyz_purples_spec, xyz_purples_plot, 
            xyz_purples_spec_N, xyz_purples_plot_N)
           
           
def compute_CIE_standard_XYZ(XYZ31_standard, XYZ64_standard):
    """
    Pass the CIE standard colour-matching functions as given in database.

    Parameters
    ----------         
    XYZ31_standard : ndarray
        The CIE 1931 XYZ colour-matching functions (2°), given to the 
        precision of 7 sign. figs., at 1 nm steps from 360 nm to 830 nm;
        wavelengths in first column.
    XYZ64_standard : ndarray
        The CIE 1964 XYZ colour-matching functions (10°), given to the 
        precision of 7 sign. figs., at 1 nm steps from 360 nm to 830 nm;
        wavelengths in first column.   
   
    Returns     
    -------       
    XYZ31_standard : ndarray
        The CIE 1931 XYZ colour-matching functions (2°) given to the 
        precision of 7 sign. figs., at 1 nm steps from 360 nm to 830 nm;
        wavelengths in first column.
    XYZ64_standard : ndarray
        The CIE 1964 XYZ colour-matching functions (10°) given to the 
        precision of 7 sign. figs., at 1 nm steps from 360 nm to 830 nm;
        wavelengths in first column.
    XYZ31_plot : ndarray
        The CIE 1931 XYZ colour-matching functions (2°) given to the
        precision of 7 sign. figs., at 1 nm steps from 360 nm to 830 nm;
        wavelengths in first column.
    XYZ64_plot : ndarray
        The CIE 1964 XYZ colour-matching functions (10°) given to the
        precision of 7 sign. figs., at 1 nm steps from 360 nm to 830 nm;
        wavelengths in first column.     
    """ 
    # '_main' : values given at 1 nm steps from 360 nm to 830 nm.    NB!
    # '_plot' : values given at 0.1 nm steps from 360 nm to 830 nm.  NB! 
    
    (λ_main, X31_main, Y31_main, Z31_main) = XYZ31_standard.T
    (X64_main, Y64_main, Z64_main) = (XYZ64_standard.T)[1:]
    # Create spline functions
    X31_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_main, X31_main)
    Y31_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_main, Y31_main)
    Z31_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_main, Z31_main)
    X64_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_main, X64_main)
    Y64_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_main, Y64_main)
    Z64_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_main, Z64_main)
    λ_plot = my_round(np.arange(360., 830. + .1, .1), 1)
    # Compute plot points for colour-matching functions
    XYZ31_plot = np.array([λ_plot,
                           X31_spline(λ_plot),
                           Y31_spline(λ_plot),
                           Z31_spline(λ_plot)]).T
    XYZ64_plot = np.array([λ_plot,
                           X64_spline(λ_plot),
                           Y64_spline(λ_plot),
                           Z64_spline(λ_plot)]).T
    return (XYZ31_standard, XYZ31_plot,
            XYZ64_standard, XYZ64_plot)


def compute_CIE_std_xy_diagram(XYZ31_standard, XYZ31_plot,
                               XYZ64_standard, XYZ64_plot):
    """
    Compute the CIE 1931 and CIE 1964 chromaticity cooordinates for the
    spectral stimuli, Illuminant E and the stimuli represented at the purple
    line's points of tangency with the spectrum locus.

    Parameters
    ----------    
    XYZ31_standard : ndarray
        The CIE 1931 XYZ colour-matching functions (2°) given to the 
        precision of 7 sign. figs., at 1 nm steps from 360 nm to 830 nm;
        wavelengths in first column.
    XYZ31_plot : ndarray
        The non-rounded interpolated CIE 1931 XYZ colour-matching functions
        (2°) given at 0.1 nm steps from 360 nm to 830 nm;
        wavelengths in first column.  
    XYZ64_standard : ndarray
        The CIE 1964 XYZ colour-matching functions (10°) given to the 
        precision of 7 sign. figs., at 1 nm steps from 360 nm to 830 nm;
        wavelengths in first column.
    XYZ64_plot : ndarray
        The non-rounded interpolated CIE 1964 XYZ colour-matching functions
        (10°) given at 0.1 nm step from 360 nm to 830 nm;
        wavelengths in first column.
   
    Returns     
    -------       
    xyz31_main : ndarray
        The CIE 1931 xyz spectral chromaticity coordinates, given to the 
        precision of 5 decimal places, at 1 nm step from 360 nm to 
        830 nm; wavelengths in first column (for table).
    xyz31_E : ndarray
        The CIE 1931 xyz chromaticity coordinates of Illuminant E, given to
        the precision of 5 decimal places (for description).
    xyz31_tg_purple : ndarray
        The CIE 1931 xyz chromaticity coordinates at the purple line's points
        of tangency with the spectrum locus, given to the precision of 5
        decimal places; wavelengths in first column (for description).
    xyz31_plot : ndarray
        The CIE 1931 xyz spectral chromaticity coordinates, at 0.1 nm
        steps from 360 nm to 830 nm; wavelengths in first column (for
        plot).
    xyz31_tg_purple_plot : ndarray
        The CIE 1931 xyz chromaticity coordinates at the purple line's points
        of tangency with the spectrum locus; wavelengths in first column (for
        plot).  
    xyz64_main : ndarray
        The CIE 1964 xyz spectral chromaticity coordinates, given to the 
        precision of 5 decimal places, at 1 nm steps from 360 nm to 
        830 nm; wavelengths in first column (for table).
    xyz64_E : ndarray
        The CIE 1964 xyz chromaticity coordinates of Illuminant E, given to
        the precision of 5 decimal places (for description).
    xyz64_tg_purple : ndarray
        The CIE 1964 xyz chromaticity coordinates at the purple line's points
        of tangency with the spectrum locus, given to the precision of 5
        decimal places; wavelengths in first column (for description).
    xyz64_plot : ndarray
        The CIE 1964 xyz spectral chromaticity coordinates, at 0.1 nm
        steps from 360 nm to 830 nm; wavelengths in irst column (for
        plot).
    xyz64_tg_purple_plot : ndarray
        The CIE 1964 xyz chromaticity coordinates at the purple line's points
        of tangency with the spectrum locus; wavelengths in first column (for
        plot).
    """
    # '_main' : values given at 1 nm steps from 360 nm to 830 nm.    NB!
    # '_plot'   values given at 0.1 nm steps from 360 nm to 830 nm.  NB! 
    
    # Spectral chromaticity coordinates (for table and plot) 
    xyz31_main = chrom_coords_µ(XYZ31_standard)     
    xyz31_main[:,1:] = my_round(xyz31_main[:,1:], 5)
    xyz64_main = chrom_coords_µ(XYZ64_standard)
    xyz64_main[:,1:] = my_round(xyz64_main[:,1:], 5)
    xyz31_plot = chrom_coords_µ(XYZ31_plot)
    xyz64_plot = chrom_coords_µ(XYZ64_plot)
    # Chromaticity coordinates of Ill. E (for description and plot)
    xyz31_E = np.array([0.33331, 0.33329, 0.33340])  # cf. CIE 1931 standard
    xyz64_E = np.array([0.33330, 0.33333, 0.33337])  # cf. CIE 1964 standard
    # Compute purple-line tangent points (for description and plot)
    xyz31_tg_purple_plot = tangent_points_purple_line(xyz31_plot)
    xyz31_tg_purple = xyz31_tg_purple_plot.copy()
    xyz31_tg_purple[:, 1:] = my_round(xyz31_tg_purple_plot[:, 1:], 5)
    xyz64_tg_purple_plot = tangent_points_purple_line(xyz64_plot)
    xyz64_tg_purple = xyz64_tg_purple_plot.copy()
    xyz64_tg_purple[:, 1:] = my_round(xyz64_tg_purple_plot[:, 1:], 5)  
    return (xyz31_main, xyz31_E, xyz31_tg_purple, 
            xyz31_plot, xyz31_tg_purple_plot, 
            xyz64_main, xyz64_E, xyz64_tg_purple, 
            xyz64_plot, xyz64_tg_purple_plot)       


# =============================================================================
# Main function for derivation and tabulation of visual data
# (for tables, plots and descriptions)
# =============================================================================

def compute_tabulated(field_size, age, λ_min=390, λ_max=830, λ_step=1):
    """
    Compute tabulated quantities for given field size and age, at specified
    wavlength steps, within specified wavelength domain.

    Parameters
    ----------
    field_size : float
        Field size in degrees.
    age : float
        Age in years.
    λ_min : float
        Lower limit of wavelength domain.
    λ_max : float
        Upper limit of wavelength domain.
    λ_step : float
        steps of tabulated results in nm.

    Returns
    -------
    results : dict
        All results: LMS, logLMS, LMS_base, logLMS_base,
        norm_coeffs_lms_mb, lms_mb, lms_mb_white, lms_mb_tg_purple,
        norm_coeffs_lms_mw, lms_mw, lms_mw_white, lms_mw_tg_purple,
        trans_mat, XYZ, trans_mat_N, XYZ_N, xyz, xyz_white,
        xyz_tg_purple, XYZ_tg_purple, xyz_N, xyz_white_N,
        xyz_tg_purple_N, XYZ_tg_purple_N, XYZ_purples, XYZ_purples_N,
        xyz_purples, xyz_purples_N, XYZ31, XYZ64, xyz31, xyz31_white,
        xyz31_tg_purple, xyz64, xyz64_white, xyz64_tg_purple,
        field_size, age, λ_min, λ_max, λ_step, λ_purple_min,
        λ_purple_max, λ_purple_min_N, λ_purple_max_N
    plots : dict
        Versions for plotting: LMS_base, logLMS_base, lms_mb,
        lms_mb_white, lms_mb_tg_purple, lms_mw, lms_mw_white,
        lms_mw_tg_purple, XYZ, XYZ_N, xyz, xyz_white,
        xyz_tg_purple, XYZ_tg_purple, xyz_N, xyz_white_N,
        xyz_tg_purple_N, XYZ_tg_purple_N, XYZ_purples,
        XYZ_purples_N, xyz_purples, xyz_purples_N, XYZ31, XYZ64,
        xyz31, xyz31_tg_purple, xyz64, xyz64_tg_purple, field_size,
        age, λ_min, λ_max, λ_step, λ_purple_min, λ_purple_max,
        λ_purple_min_N, λ_purple_max_N
    """

    # =======================================================================
    # Initialise result and plot directories for stacking of computed values
    # =======================================================================

    results = dict()
    plots = dict()

    # =======================================================================
    # Create initial data arrays
    # =======================================================================

    # '_base' : 9 sign. figs.
    # '_std'  : standard number of sign. figs./decimal places
    # '_all'  : values given at 0.1 nm steps from 390 nm to 830 nm
    # '_main' : values given at 1 nm steps from 390 nm to 830 nm
    # '_spec' : values given at specified wavelengths
    # '_plot' : values given at 0.1 nm steps within specified domain

    # wavelength arrays:

    λ_all = my_round(np.arange(390., 830. + .01, .1), 1)
    λ_spec = np.arange(λ_min, λ_max + .01, λ_step)
    λ_max = λ_spec[-1]
    λ_plot = my_round(np.arange(λ_min, λ_max + .01, .1), 1)

    # LMS arrays:

    # LMS-base values (9 sign.figs.) at 0.1 nm steps from 390 nm to 830 nm;
    # wavelengths in first column
    LMS_base_all = LMS_energy(field_size, age, base=True)[0]
    # LMS values (6 sign.figs.) at 0.1 nm steps from 390 nm to 830 nm;
    # wavelengths in first column
    LMS_std_all = LMS_energy(field_size, age)[0]

    # Vλ and weighting factors of the L and M cone fundamentals:

    # - Cone-fundamental-based V(λ) values (7 sign. figs.) at 0.1 nm steps
    #   from 390 nm to 830 nm; wavelengths in first column
    # - Weights of L and M cone fundamentals in V(λ) synthesis
    (Vλ_std_all, LM_weights) = Vλ_energy_and_LM_weights(field_size, age)

    # =======================================================================
    # Create spline functions
    # =======================================================================

    # base:
    (λ_all, L_base_all, M_base_all, S_base_all) = LMS_base_all.T
    L_base_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_all, L_base_all)
    M_base_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_all, M_base_all)
    S_base_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_all, S_base_all)
    # std:
    (λ_all, L_std_all, M_std_all, S_std_all) = LMS_std_all.T
    (λ_all, V_std_all) = Vλ_std_all.T
    L_std_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_all, L_std_all)
    M_std_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_all, M_std_all)
    S_std_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_all, S_std_all)
    V_std_spline = scipy.interpolate.InterpolatedUnivariateSpline(
        λ_all, V_std_all)

    # =======================================================================
    # Compute the cone-fundamental-based spectral luminous efficiency
    # functions (cone-fundamental-based V(λ)-function)
    # =======================================================================

    # - Cone-fundamental-based V(λ) values (7 sign. figs.) for specified
    #   wavelengths; wavelengths in first column.
    Vλ_std_spec = np.array([λ_spec, V_std_spline(λ_spec)]).T

    # =======================================================================
    # Compute the LMS cone fundamentals (linear and logarithmic values)
    #=======================================================================
    
    # - LMS values (7 number of sign. figs.) for specified wavelengths; 
    #   wavelengths in first column 
    # - Briggsian logarithm of LMS values (5 decimal places)
    #   for specified wavelengths; wavelengths in first column
    (LMS_std_spec,
     logLMS_std_spec) = compute_LMS(
         λ_spec, L_std_spline, M_std_spline, S_std_spline)
    (LMS_std_plot,
     logLMS_std_plot) = compute_LMS(
         λ_plot, L_std_spline, M_std_spline, S_std_spline)

    results['LMS'] = chop(LMS_std_spec)
    results['logLMS'] = chop(logLMS_std_spec)
    plots['LMS'] = chop(LMS_std_plot)
    plots['logLMS'] = chop(logLMS_std_plot)

    # =======================================================================
    # Compute the LMS-base cone fundamentals (linear and logarithmic values)
    # =======================================================================

    # - LMS-base values (9 sign. figs) for specified wavelengths;
    #   wavelengths in first column
    # - Briggsian logarithm of LMS-base values (8 decimal places)
    #   for specified wavelengths; wavelengths in first column
    # - Versions for plotting
    (LMS_base_spec,
     logLMS_base_spec) = compute_LMS(
         λ_spec, L_base_spline, M_base_spline, S_base_spline, base=True)
    (LMS_base_plot,
     logLMS_base_plot) = compute_LMS(
         λ_plot, L_base_spline, M_base_spline, S_base_spline, base=True)

    results['LMS_base'] = chop(LMS_base_spec)
    results['logLMS_base'] = chop(logLMS_base_spec)
    plots['LMS_base'] = chop(LMS_base_plot)
    plots['logLMS_base'] = chop(logLMS_base_plot)

    # =======================================================================
    # Compute the MacLeod‒Boynton ls chromaticity diagram
    # =======================================================================

    # 'mb' denotes MacLeod‒Boynton

    # - normalization coefficients (scaling factor) for calculation of
    #   the MacLeod‒Boynton lms coordinates
    # - MacLeod‒Boynton lms values (6 decimal places) for specified
    #   wavelengths; wavelengths in first column
    # - MacLeod‒Boynton lms values (6 decimal places) for Illuminant E
    # - MacLeod‒Boynton lms values (6 decimal places) for the purple line's
    #   points of tangency with the spectrum locus
    # - Respective versions for plotting
    (norm_coeffs_lms_mb,
    lms_mb_std_spec,
    lms_mb_std_E,
    lms_mb_std_tg_purple,
    lms_mb_plot,
    lms_mb_E_plot,
    lms_mb_tg_purple_plot) = compute_MacLeod_Boynton_diagram(
            results['LMS_base'], plots['LMS_base'], LMS_base_all,
            Vλ_std_all, Vλ_std_spec, LM_weights)  
    results['norm_coeffs_lms_mb'] = chop(norm_coeffs_lms_mb)
    results['lms_mb'] = chop(lms_mb_std_spec)
    results['lms_mb_white'] = lms_mb_std_E
    results['lms_mb_tg_purple'] = chop(lms_mb_std_tg_purple)
    plots['lms_mb'] = chop(lms_mb_plot)
    plots['lms_mb_white'] = lms_mb_E_plot
    plots['lms_mb_tg_purple'] = chop(lms_mb_tg_purple_plot)

    # =======================================================================
    # Compute the Maxwellian lm chromaticity diagram
    # =======================================================================

    # 'mw' denotes Maxwellian

    # - normalization coefficients (scaling factors) for calculation of the
    #   Maxwellian lms coordinates
    # - Maxwellian lms values (6 decimal places) for specified wavelengths;
    #   wavelengths in first column
    # - Maxwellian lms values (6 decimal places) for Illuminant E
    # - Maxwellian lms values (6 decimal places) for the purple line's
    #   points of tangency with the spectrum locus
    # - Respective versions for plotting
    (norm_coeffs_lms_mw,
     lms_mw_std_spec,
     lms_mw_std_E,
     lms_mw_std_tg_purple,
     lms_mw_plot,
     lms_mw_E_plot,
     lms_mw_tg_purple_plot) = compute_Maxwellian_diagram(
         results['LMS_base'], plots['LMS_base'])

    results['norm_coeffs_lms_mw'] = chop(norm_coeffs_lms_mw)
    results['lms_mw'] = chop(lms_mw_std_spec)
    results['lms_mw_white'] = lms_mw_std_E
    results['lms_mw_tg_purple'] = chop(lms_mw_std_tg_purple)
    plots['lms_mw'] = chop(lms_mw_plot)
    plots['lms_mw_white'] = lms_mw_E_plot
    plots['lms_mw_tg_purple'] = chop(lms_mw_tg_purple_plot)

    # =======================================================================
    # Compute the cone-fundamental-based XYZ tristimulus functions
    # =======================================================================

    #  Determine reference diagram
    xyz_reference = xyz_interpolated_reference_system(
            field_size, VisualData.XYZ31.copy(), VisualData.XYZ64.copy())

    # - Non-renormalised tranformation matrix (8 decimal placed)
    # - Non-renormalised CIE cone-fundamental-based XYZ tristimulus
    #   values (7 sign. figs) for specified wavelengths; wavelengths
    #   in first column
    # - version for plotting
    # - Ditto renormalized
    (trans_mat_std,
     XYZ_std_spec,
     XYZ_plot,
     trans_mat_std_N,
     XYZ_std_spec_N,
     XYZ_plot_N) = compute_XYZ(
         L_base_spline, M_base_spline, S_base_spline, V_std_spline,
         results['LMS_base'], plots['LMS_base'], LMS_base_all,
         LM_weights, xyz_reference)
    results['trans_mat'] = chop(trans_mat_std)
    results['XYZ'] = chop(XYZ_std_spec)
    results['trans_mat_N'] = chop(trans_mat_std_N)
    results['XYZ_N'] = chop(XYZ_std_spec_N)
    plots['XYZ'] = chop(XYZ_plot)
    plots['XYZ_N'] = chop(XYZ_plot_N)

    # =======================================================================
    # Compute the cone-fundamental-based xy chromaticity diagram
    # =======================================================================

    # - Non-renormalised xyz chromaticity coordinates (5 decimal places)
    #   for specified wavelengths;
    #   wavelengths in first column
    # - Non-renormalised xyz chromaticity coordinates (5 decimal places)
    #   for Illuminant E;
    # - Non-renormalised xyz chromaticity coordinates (5 decimal places)
    #   for the purple line's points of tangency with the spectrum locus;
    #   wavelengths in first column
    # - Non-renormalised XYZ tristimulus values (7 sign. figs.) for
    #   the purple line's points of tangency with the spectrum locus;
    #   wavelengths in first column
    # - Respective versions for plotting
    # - Ditto renormalised
    (xyz_std_spec,
     xyz_std_E,
     xyz_std_tg_purple,
     XYZ_std_tg_purple,
     xyz_plot,
     xyz_E_plot,
     xyz_tg_purple_plot,
     XYZ_tg_purple_plot,
     xyz_std_spec_N,
     xyz_std_E_N,
     xyz_std_tg_purple_N,
     XYZ_std_tg_purple_N,
     xyz_plot_N,
     xyz_E_plot_N,
     xyz_tg_purple_plot_N,
     XYZ_tg_purple_plot_N) = compute_xy_diagram(
         results['XYZ'], plots['XYZ'], results['XYZ_N'], plots['XYZ_N'])

    results['xyz'] = chop(xyz_std_spec)
    results['xyz_white'] = xyz_std_E
    results['xyz_tg_purple'] = chop(xyz_std_tg_purple)
    results['XYZ_tg_purple'] = chop(XYZ_std_tg_purple)
    results['xyz_N'] = chop(xyz_std_spec_N)
    results['xyz_white_N'] = xyz_std_E_N
    results['xyz_tg_purple_N'] = chop(xyz_std_tg_purple_N)
    results['XYZ_tg_purple_N'] = chop(XYZ_std_tg_purple_N)
    plots['xyz'] = chop(xyz_plot)
    plots['xyz_white'] = xyz_E_plot
    plots['xyz_tg_purple'] = chop(xyz_tg_purple_plot)
    plots['XYZ_tg_purple'] = chop(XYZ_tg_purple_plot)
    plots['xyz_N'] = chop(xyz_plot_N)
    plots['xyz_white_N'] = xyz_E_plot_N
    plots['xyz_tg_purple_N'] = chop(xyz_tg_purple_plot_N)
    plots['XYZ_tg_purple_N'] = chop(XYZ_tg_purple_plot_N)

    # =======================================================================
    # Compute the cone-fundamental-based XYZ tristimulus functions for
    # purple-line stimuli, as parameterized by complementary wavelength
    # =======================================================================

    # - non-renormalized cone-fundamental-based XYZ tristimulus values
    #   (7 sign. figs.) of stimuli represented on the purple line,
    #   parameterized by complementary wavelength;
    #   complementary wavelengths in first column.
    # - version for plotting
    # - Ditto renormalized
    (XYZ_purples_std_spec,
     XYZ_purples_plot,
     XYZ_purples_std_spec_N,
     XYZ_purples_plot_N) = compute_XYZ_purples(
         results['xyz'], results['xyz_white'],
         results['XYZ_tg_purple'],
         plots['xyz'], plots['xyz_white'], plots['XYZ_tg_purple'],
         results['xyz_N'], results['xyz_white_N'],
         results['XYZ_tg_purple_N'],
         plots['xyz_N'], plots['xyz_white_N'], plots['XYZ_tg_purple_N'])

    results['XYZ_purples'] = chop(XYZ_purples_std_spec)
    results['XYZ_purples_N'] = chop(XYZ_purples_std_spec_N)
    plots['XYZ_purples'] = chop(XYZ_purples_plot)
    plots['XYZ_purples_N'] = chop(XYZ_purples_plot_N)

    # =======================================================================
    # Compute cone-fundamental-based xyz chromaticity coordinates for
    # purple-line stimuli, as parameterized by complementary wavelength
    # =======================================================================

    # - non-renormalized cone-fundamental-based xyz chromaticity
    #   coordinates (5 decimal places) of stimuli represented on
    #   the purple line, parameterized by complementary wavelength;
    #   complementary wavelengths in first column.
    # - version for plotting
    # - Ditto renormalized
    (xyz_purples_std_spec,
     xyz_purples_plot,
     xyz_purples_std_spec_N,
     xyz_purples_plot_N) = compute_xyz_purples(
         results['XYZ_purples'], plots['XYZ_purples'],
         results['XYZ_purples_N'], plots['XYZ_purples_N'])

    results['xyz_purples'] = chop(xyz_purples_std_spec)
    results['xyz_purples_N'] = chop(xyz_purples_std_spec_N)
    plots['xyz_purples'] = chop(xyz_purples_plot)
    plots['xyz_purples_N'] = chop(xyz_purples_plot_N)

    # =======================================================================
    # Compute the CIE standard XYZ colour-matching functions
    # =======================================================================

    # NB!
    # '_main  here means values given at 1 nm steps from 360 nm to 830 nm.
    # '_plot' here means values given at 0.1 nm steps from 360 nm to 830 nm.

    # - CIE 1931 standard XYZ spectral tristimulus values (7 sign.figs.);
    #   wavelengths in first column
    # - CIE 1964 standard XYZ spectral tristimulus values (7 sign.figs.);
    #   wavelengths in first column
    (XYZ31_std_main,
     XYZ31_plot,
     XYZ64_std_main,
     XYZ64_plot) = compute_CIE_standard_XYZ(
         VisualData.XYZ31.copy(), VisualData.XYZ64.copy())

    results['XYZ31'] = chop(XYZ31_std_main)
    results['XYZ64'] = chop(XYZ64_std_main)
    plots['XYZ31'] = chop(XYZ31_plot)
    plots['XYZ64'] = chop(XYZ64_plot)

    # =======================================================================
    # Compute the CIE standard xy diagrams
    # =======================================================================

    # - CIE 1931 standard xyz spectral chromaticity
    #   coordinates (5 decimal places);
    #   wavelengths in first column
    # - CIE 1931 standard chromaticity coordinates
    #   (5 decimal places) of Illuminant E
    # - CIE 1931 standard chromaticity coordinates (5 decimal places)
    #   for the purple line's points of tangency with the spectrum locus;
    #   wavelengths in first column
    # - CIE 1931 standard xyz spectral chromaticity
    #   coordinates (5 decimal places);
    #   wavelengths in first column
    # - CIE 1931 standard chromaticity coordinates
    #   (5 decimal places) of Illuminant E
    # - CIE 1931 standard chromaticity coordinates (5 decimal places)
    #   for the purple line's points of tangency with the spectrum locus;
    #   wavelengths in first column
    # - Versions for plotting

    (xyz31_std,
     xyz31_E,
     xyz31_tg_purple,
     xyz31_plot,
     xyz31_tg_purple_plot,
     xyz64_std,
     xyz64_E,
     xyz64_tg_purple,
     xyz64_plot,
     xyz64_tg_purple_plot) = compute_CIE_std_xy_diagram(
         results['XYZ31'], results['XYZ31'],
         plots['XYZ31'], plots['XYZ64'])

    results['xyz31'] = chop(xyz31_std)
    results['xyz31_white'] = xyz31_E
    results['xyz31_tg_purple'] = chop(xyz31_tg_purple)
    results['xyz64'] = chop(xyz64_std)
    results['xyz64_white'] = xyz64_E
    results['xyz64_tg_purple'] = chop(xyz64_tg_purple)
    plots['xyz31'] = chop(xyz31_plot)
    plots['xyz31_tg_purple'] = chop(xyz31_tg_purple_plot)
    plots['xyz64'] = chop(xyz64_plot)
    plots['xyz64_tg_purple'] = chop(xyz64_tg_purple_plot) 
    
    
    #=======================================================================
    # Stack all parameters for results and plots (values from spinboxes,
    # and computed values for purples) in respective directories
    # =======================================================================

    # Assign parameter values for plots
    if np.round(field_size, 5) == np.round(field_size):
        plots['field_size'] = '%.0f' % field_size
    else:
        plots['field_size'] = '%.1f' % field_size
    plots['age'] = age
    if np.round(λ_step, 5) == np.round(λ_step) and \
       np.round(λ_min, 5) == np.round(λ_min) and \
       np.round(λ_max, 5) == np.round(λ_max):
        plots['λ_min'] = '%.0f' % λ_min
        plots['λ_max'] = '%.0f' % λ_max
        plots['λ_step'] = '%.0f' % λ_step
        plots['λ_purple_min'] = '%.0f' % plots['XYZ_purples'][0, 0]
        plots['λ_purple_max'] = '%.0f' % plots['XYZ_purples'][-1, 0]
        plots['λ_purple_min_N'] = '%.0f' % plots['XYZ_purples_N'][0, 0]
        plots['λ_purple_max_N'] = '%.0f' % plots['XYZ_purples_N'][-1, 0]

    else:
        plots['λ_min'] = '%.1f' % λ_min
        plots['λ_max'] = '%.1f' % λ_max
        plots['λ_step'] = '%.1f' % λ_step
        plots['λ_purple_min'] = '%.1f' % plots['XYZ_purples'][0, 0]
        plots['λ_purple_max'] = '%.1f' % plots['XYZ_purples'][-1, 0]
        plots['λ_purple_min_N'] = '%.1f' % plots['XYZ_purples_N'][0, 0]
        plots['λ_purple_max_N'] = '%.1f' % plots['XYZ_purples_N'][-1, 0]

    # Format parameter-string representations (for description)
    if np.round(field_size, 5) == np.round(field_size):
        results['field_size'] = '%.0f' % field_size
    else:
        results['field_size'] = '%.1f' % field_size
    results['age'] = age
    if (np.round(λ_step, 5) == np.round(λ_step) and
            np.round(λ_min, 5) == np.round(λ_min) and
            np.round(λ_max, 5) == np.round(λ_max)):
        results['λ_min'] = '%.0f' % λ_min
        results['λ_max'] = '%.0f' % λ_max
        results['λ_step'] = '%.0f' % λ_step
        results['λ_purple_min'] = '%.0f' % results['XYZ_purples'][0, 0]
        results['λ_purple_max'] = '%.0f' % results['XYZ_purples'][-1, 0]
        results['λ_purple_min_N'] = '%.0f' % results['XYZ_purples_N'][0, 0]
        results['λ_purple_max_N'] = ('%.0f' % results['XYZ_purples_N'][-1, 0])
    else:
        results['λ_min'] = '%.1f' % λ_min
        results['λ_max'] = '%.1f' % λ_max
        results['λ_step'] = '%.1f' % λ_step
        results['λ_purple_min'] = '%.1f' % results['XYZ_purples'][0, 0]
        results['λ_purple_max'] = '%.1f' % results['XYZ_purples'][-1, 0]
        results['λ_purple_min_N'] = '%.1f' % results['XYZ_purples_N'][0, 0]
        results['λ_purple_max_N'] = ('%.1f' % results['XYZ_purples_N'][-1, 0])

    # =======================================================================
    # Return all results (for tables, plots and descriptions)
    # =======================================================================

    return (results, plots)


# ==============================================================================
# For testing purposes only
# ==============================================================================

if __name__ == '__main__':
    res, plots = compute_tabulated(2.5, 37, 390, 830, 1)
    print((res['XYZ_N'] != res['XYZ']).sum())
