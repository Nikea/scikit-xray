# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 08/19/2014                                                #
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
from __future__ import (absolute_import, division,
                        unicode_literals, print_function)
import six
import numpy as np
from numpy.testing import assert_array_almost_equal

import nsls2.calibration as calibration
import nsls2.calibration as core
from nose.tools import assert_true


def _draw_gaussian_rings(shape, calibrated_center, r_list, r_width):
    R = core.pixel_to_radius(shape, calibrated_center)
    I = np.zeros_like(R)

    for r in r_list:
        tmp = 100 * np.exp(-((R - r)/r_width)**2)
        I += tmp

    return I


def test_refine_center():
    center = np.array((500, 550))
    I = _draw_gaussian_rings((1000, 1001), center,
                             [50, 75, 100, 250, 500], 5)

    out = calibration.refine_center(I, center+1, (1, 1),
                                    phi_steps=20, nx=300, min_x=10,
                                    max_x=300, window_size=5,
                                    thresh=0, max_peaks=4)

    assert np.all(np.abs(center - out) < .1)


def test_blind_d():
    gaus = lambda x, center, height, width: (
                          height * np.exp(-((x-center) / width)**2))
    name = 'Si'
    wavelength = .18
    window_size = 5
    threshold = .1
    cal = calibration.calibration_standards[name]

    tan2theta = np.tan(cal.convert_2theta(wavelength))

    D = 200
    expected_r = D * tan2theta

    bin_centers = np.linspace(0, 50, 2000)
    I = np.zeros_like(bin_centers)
    for r in expected_r:
        I += gaus(bin_centers, r, 100, .2)
    d, dstd = calibration.estimate_d_blind(name, wavelength, bin_centers,
                                     I, window_size, len(expected_r),
                                     threshold)
    assert np.abs(d - D) < 1e-6


def test_full_auto_calibration():
    name = 'Si'
    wavelength = .18
    pixel_size = (.1, .1)
    cal = calibration.calibration_standards[name]
    center = (501.25, 515.75)
    tan2theta = np.tan(cal.convert_2theta(wavelength))

    D = 200
    expected_r = D * tan2theta / pixel_size[0]

    I = _draw_gaussian_rings((1000, 1010), center, expected_r, 2)

    res = calibration.powder_auto_calibrate(I, name, wavelength, pixel_size)
    d, d_std, m_center, center_error, tilt, tilt_error = res
    print(center, m_center)

    # assert within error
    assert np.abs(d - D) < d_std
    # 1/10 pixel accuracy
    assert_array_almost_equal(center, m_center, decimal=1)


def test_tilt_roundtrip():

    phi1 = .4
    phi2 = .3
    r0 = 5

    theta = np.linspace(0, 2*np.pi, 2500, endpoint=False)
    row = r0 * np.sin(theta)
    col = r0 * np.cos(theta)

    a, b = calibration.tilt_coords(phi1, phi2, row, col)

    r, c = calibration.untilt_coords(phi1, phi2, a, b)

    assert_array_almost_equal(row, r)
    assert_array_almost_equal(col, c)


def _tilt_test_helper(tilts):
    """
    tilt -> coef -> tilt
    """
    coefs = calibration.tilt_angles_to_coefs(*tilts)
    new_tilts = calibration.coefs_to_params(*coefs)
    assert_array_almost_equal(tilts, new_tilts)


def _phi2_tests(tilts):
    r0, a1, a2 = calibration.tilt_angles_to_coefs(*tilts)
    phi1 = calibration.coefs_to_phi1(a1, a2)
    phi2_cos = calibration.coefs_to_phi2_cos(a1, r0, phi1)
    phi2_sin = calibration.coefs_to_phi2_sin(a2, r0, phi1)

    assert_true(np.abs(phi2_sin - phi2_cos) < 1e-7)


def test_tilts_round_trip():
    test_tilts = tuple((r, phi1, phi2)
                  for r in (10, 300, 1000)
                  for phi1 in np.linspace(0, np.pi/2, 10)
                  for phi2 in np.linspace(0.0001, np.pi/4, 10))
    for tilt in test_tilts:
        yield _tilt_test_helper, tilt


def test_phi2():
    test_tilts = tuple((r, phi1, phi2)
                  for r in (10, 300, 1000)
                  for phi1 in np.linspace(0.1, np.pi/2, 10)
                  for phi2 in np.linspace(0.0001, np.pi/4, 10))
    for tilt in test_tilts:
        yield _phi2_tests, tilt
