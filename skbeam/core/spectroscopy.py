# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################
"""
This module is for spectroscopy specific tools (spectrum fitting etc).
"""
from __future__ import absolute_import, division, print_function

import logging

import numpy as np
from scipy.integrate import simpson as simps
from six.moves import zip

from .fitting import fit_quad_to_peak

logger = logging.getLogger(__name__)


def align_and_scale(energy_list, counts_list, pk_find_fun=None):
    """

    Parameters
    ----------
    energy_list : iterable of ndarrays
        list of ndarrays with the energy of each element

    counts_list : iterable of ndarrays
        list of ndarrays of counts/element

    pk_find_fun : function or None
       A function which takes two ndarrays and returns parameters
       about the largest peak.  If None, defaults to `find_largest_peak`.
       For this demo, the output is (center, height, width), but this sould
       be pinned down better.

    Returns
    -------
    out_e : list of ndarray
       The aligned/scaled energy arrays

    out_c : list of ndarray
       The count arrays (should be the same as the input)
    """
    if pk_find_fun is None:
        pk_find_fun = find_largest_peak

    base_sigma = None
    out_e, out_c = [], []
    for e, c in zip(energy_list, counts_list):
        E0, max_val, sigma = pk_find_fun(e, c)
        if base_sigma is None:
            base_sigma = sigma
        out_e.append((e - E0) * base_sigma / sigma)
        out_c.append(c)

    return out_e, out_c


def find_largest_peak(x, y, window=None):
    """
    Finds and estimates the location, width, and height of
    the largest peak. Assumes the top of the peak can be
    approximated as a Gaussian.  Finds the peak properties
    using least-squares fitting of a parabola to the log of
    the counts.

    The region around the peak can be approximated by
    :math:`Y = Y0 * exp(- (X - X0)**2 / (2 * sigma **2))`

    Parameters
    ----------
    x : ndarray
       The independent variable

    y : ndarary
      Dependent variable sampled at positions X

    window : int, optional
       The size of the window around the maximum to use
       for the fitting


    Returns
    -------
    x0 : float
        The location of the peak

    y0 : float
        The magnitude of the peak

    sigma : float
        Width of the peak
    """

    # make sure they are _really_ arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # get the bin with the largest number of counts
    j = np.argmax(y)
    if window is not None:
        roi = slice(np.max(j - window, 0), j + window + 1)
    else:
        roi = slice(0, -1)

    (w, x0, y0), r2 = fit_quad_to_peak(x[roi], np.log(y[roi]))
    return x0, np.exp(y0), 1 / np.sqrt(-2 * w)


def integrate_ROI_spectrum(bin_edges, counts, x_min, x_max):
    """Integrate region(s) of histogram.

    If `x_min` and `x_max` are arrays/lists they must be equal in
    length. The values contained in the 'x_value_array' must be
    monotonic (up or down).  The returned value is the sum of all the
    regions and a single scalar value is returned.  Each region is
    computed independently, if regions overlap the overlapped area will
    be included multiple times in the final sum.

    `bin_edges` is an array of the left edges and the final right
    edges of the bins.  `counts` is the value in each of those bins.

    The bins who's centers fall with in the integration limits are
    included in the sum.

    Parameters
    ----------
    bin_edges : array
        Independent variable, any unit.

        Must be one longer in length than counts

    counts : array
        Dependent variable, any units

    x_min : float or array
        The lower edge of the integration region(s).

    x_max : float or array
        The upper edge of the integration region(s).

    Returns
    -------
    float
        The totals integrated value in same units as `counts`

    """
    bin_edges = np.asarray(bin_edges)
    return integrate_ROI(bin_edges[:-1] + np.diff(bin_edges), counts, x_min, x_max)


def _formatter_array_regions(x, centers, window=1, tab_count=0):
    """Returns a formatted string of sub-sections of an array

    Each value in center generates a section of the string like:

       {tab_count*\t}c : [x[c - n] ... x[c] ... x[c + n + 1]]


    Parameters
    ----------
    x : array
        The array to be looked into

    centers : iterable
        The locations to print out around

    window : int, optional
        how many values on either side of center to include

        defaults to 1

    tab_count : int, optional
       The number of tabs to pre-fix lines with

       default is 0

    Returns
    -------
    str
      The formatted string
    """
    xl = len(x)
    x = np.asarray(x)
    header = "\t" * tab_count + "center\tarray values\n" + "\t" * tab_count + "------\t------------\n"
    return header + "\n".join(
        [
            "\t" * tab_count
            + "{c}: \t {vals}".format(c=c, vals=x[np.max([0, c - window]) : np.min([xl, c + window + 1])])
            for c in centers
        ]
    )


def integrate_ROI(x, y, x_min, x_max):
    """Integrate region(s) of input data.

    If `x_min` and `x_max` are arrays/lists they must be equal in
    length. The values contained in the 'x' must be monotonic (up or
    down).  The returned value is the sum of all the regions and a
    single scalar value is returned.  Each region is computed
    independently, if regions overlap the overlapped area will be
    included multiple times in the final sum.

    This function assumes that `y` is a function of
    `x` sampled at `x`.

    Parameters
    ----------
    x : array
        Independent variable, any unit

    y : array
        Dependent variable, any units

    x_min : float or array
        The lower edge of the integration region(s)
        in units of x.

    x_max : float or array
        The upper edge of the integration region(s)
        in units of x.

    Returns
    -------
    float
        The totals integrated value in same units as `y`
    """
    # make sure x (x-values) and y (y-values) are arrays
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError(
            "Inputs (x and y) must be the same "
            "size. x.shape = {0} and y.shape = "
            "{1}".format(x.shape, y.shape)
        )

    # use np.sign() to obtain array which has evaluated sign changes in all
    # diff in input x_value array. Checks and tests are then run on the
    # evaluated sign change array.
    eval_x_arr_sign = np.sign(np.diff(x))

    # check to make sure no outliers exist which violate the monotonically
    # increasing requirement, and if exceptions exist, then error points to the
    # location within the source array where the exception occurs.
    if not np.all(eval_x_arr_sign == eval_x_arr_sign[0]):
        error_locations = np.where(eval_x_arr_sign != eval_x_arr_sign[0])[0]
        raise ValueError(
            "Independent variable must be monotonically "
            "increasing. Erroneous values found at x-value "
            "array index locations:\n" + _formatter_array_regions(x, error_locations)
        )

    # check whether the sign of all diff measures are negative in the
    # x. If so, then the input array for both x_values and
    # count are reversed so that they are positive, and monotonically increase
    # in value
    if eval_x_arr_sign[0] == -1:
        x = x[::-1]
        y = y[::-1]
        logging.debug(
            "Input values for 'x' were found to be "
            "monotonically decreasing. The 'x' and "
            "'y' arrays have been reversed prior to "
            "integration."
        )

    # up-cast to 1d and make sure it is flat
    x_min = np.atleast_1d(x_min).ravel()
    x_max = np.atleast_1d(x_max).ravel()

    # verify that the number of minimum and maximum boundary values are equal
    if len(x_min) != len(x_max):
        raise ValueError("integration bounds must have same lengths")

    # verify that the specified minimum values are actually less than the
    # sister maximum value, and raise error if any minimum value is actually
    # greater than the sister maximum value.
    if np.any(x_min >= x_max):
        raise ValueError("All lower integration bounds must be less than " "upper integration bounds.")

    # check to make sure that all specified minimum and maximum values are
    # actually contained within the extents of the independent variable array
    if np.any(x_min < x[0]):
        error_locations = np.where(x_min < x[0])[0]
        raise ValueError(
            "Specified lower integration boundary values are "
            "outside the spectrum range. All minimum integration "
            "boundaries must be greater than, or equal to the "
            "lowest value in spectrum range. The erroneous x_min_"
            "array indices are:\n" + _formatter_array_regions(x_min, error_locations, window=0)
        )

    if np.any(x_max > x[-1]):
        error_locations = np.where(x_max > x[-1])[0]
        raise ValueError(
            "Specified upper integration boundary values "
            "are outside the spectrum range. All maximum "
            "integration boundary values must be less "
            "than, or equal to the highest value in the spectrum "
            "range. The erroneous x_max array indices are: "
            "\n" + _formatter_array_regions(x_max, error_locations, window=0)
        )

    # find the bottom index of each integration bound
    bottom_indx = x.searchsorted(x_min)
    # find the top index of each integration bound
    # NOTE: +1 required for correct slicing for integration function
    top_indx = x.searchsorted(x_max) + 1

    # set up temporary variables
    accum = 0
    # integrate each region
    for bot, top in zip(bottom_indx, top_indx):
        accum += simps(y[bot:top], x=x[bot:top])

    return accum
