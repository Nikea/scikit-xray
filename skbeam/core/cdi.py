# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 03/27/2015                                                #
#                                                                      #
# Original code from Xiaojing Huang (xjhuang@bnl.gov) and Li Li        #
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
from __future__ import absolute_import, division, print_function

import logging
import time

import numpy as np
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


def _dist(dims):
    """
    Create array with pixel value equals to the distance from array center.

    Parameters
    ----------
    dims : list or tuple
        shape of array to create

    Returns
    -------
    arr : np.ndarray
        ND array whose pixels are equal to the distance from the center
        of the array of shape `dims`
    """
    dist_sum = []

    dist_sum = np.zeros(dims, dtype=np.float64)
    for idx, d in enumerate(dims):
        shape = np.ones(len(dims), dtype=np.int64)
        shape[idx] = d
        vec = (np.arange(d) - d // 2) ** 2
        vec = np.broadcast_to(np.reshape(vec, shape=shape), dims)
        dist_sum += vec

    return np.sqrt(dist_sum)


def gauss(dims, sigma):
    """
    Generate Gaussian function in 2D or 3D.

    Parameters
    ----------
    dims : list or tuple
        shape of the data
    sigma : float
        standard deviation of gaussian function

    Returns
    -------
    arr : array
        ND gaussian
    """
    x = _dist(dims)
    y = np.exp(-((x / sigma) ** 2) / 2)
    return y / np.sum(y)


def pi_modulus(recon_pattern, diffracted_pattern, offset_v=1e-12):
    """
    Transfer sample from real space to q space.
    Use constraint based on diffraction pattern from experiments.

    Parameters
    ----------
    recon_pattern : array
        reconstructed pattern in real space
    diffracted_pattern : array
        diffraction pattern from experiments
    offset_v : float, optional
        add small value to avoid the case of dividing something by zero

    Returns
    -------
    array :
        updated pattern in real space
    """
    diff_tmp = np.fft.fftn(recon_pattern) / np.sqrt(np.size(recon_pattern))
    index = diffracted_pattern > 0
    diff_tmp[index] = diffracted_pattern[index] * diff_tmp[index] / (np.abs(diff_tmp[index]) + offset_v)
    return np.fft.ifftn(diff_tmp) * np.sqrt(np.size(diffracted_pattern))


def find_support(sample_obj, sw_sigma, sw_threshold):
    """
    Update sample area based on thresholds.

    Parameters
    ----------
    sample_obj : array
        sample for reconstruction
    sw_sigma : float
        sigma for gaussian in shrinkwrap method
    sw_threshold : float
        threshold used in shrinkwrap method

    Returns
    -------
    array :
        index of sample support
    """
    sample_obj = np.abs(sample_obj)
    conv_fun = gaussian_filter(sample_obj, sw_sigma)
    conv_max = np.max(conv_fun)
    return conv_fun >= (sw_threshold * conv_max)


def cal_diff_error(sample_obj, diffracted_pattern):
    """
    Calculate the error in q space.

    Parameters
    ----------
    sample_obj : array
        sample data
    diffracted_pattern : array
        diffraction pattern from experiments

    Returns
    -------
    float :
        relative error in q space
    """
    new_diff = np.abs(np.fft.fftn(sample_obj)) / np.sqrt(np.size(sample_obj))
    return np.linalg.norm(new_diff - diffracted_pattern) / np.linalg.norm(diffracted_pattern)


def generate_random_phase_field(diffracted_pattern):
    """
    Initiate random phase.

    Parameters
    ----------
    diffracted_pattern : array
        diffraction pattern from experiments

    Returns
    -------
    sample_obj : array
        sample information with phase
    """
    pha_tmp = np.random.uniform(0, 2 * np.pi, diffracted_pattern.shape)
    sample_obj = np.fft.ifftn(diffracted_pattern * np.exp(1j * pha_tmp)) * np.sqrt(np.size(diffracted_pattern))
    return sample_obj


def generate_box_support(sup_radius, shape_v):
    """
    Generate support area as a box for either 2D or 3D cases.

    Parameters
    ----------
    sup_radius : float
        radius of support
    shape_v : list
        shape of diffraction pattern, which can be either 2D or 3D case.

    Returns
    -------
    sup : array
        support with a box area
    """
    slc_list = [slice(s // 2 - sup_radius, s // 2 + sup_radius) for s in shape_v]
    sup = np.zeros(shape_v)
    sup[tuple(slc_list)] = 1
    return sup


def generate_disk_support(sup_radius, shape_v):
    """
    Generate support area as a disk for either 2D or 3D cases.

    Parameters
    ----------
    sup_radius : float
        radius of support
    shape_v : list
        shape of diffraction pattern, which can be either 2D or 3D case.

    Returns
    -------
    sup : array
        support with a disk area
    """
    sup = np.zeros(shape_v)
    dummy = _dist(shape_v)
    sup[dummy < sup_radius] = 1
    return sup


def cdi_recon(
    diffracted_pattern,
    sample_obj,
    sup,
    beta=1.15,
    start_avg=0.8,
    pi_modulus_flag="Complex",
    sw_flag=True,
    sw_sigma=0.5,
    sw_threshold=0.1,
    sw_start=0.2,
    sw_end=0.8,
    sw_step=10,
    n_iterations=1000,
    cb_function=None,
    cb_step=10,
):
    """
    Run reconstruction with difference map algorithm.

    Parameters
    ----------
    diffracted_pattern : array
        diffraction pattern from experiments
    sample_obj : array
        initial sample with phase, complex number
    sup : array
        initial support
    beta : float, optional
        feedback parameter for difference map algorithm.
        default is 1.15.
    start_avg : float, optional
        define the point to start doing average.
        default is 0.8.
    pi_modulus_flag : {'complex', 'real'}, optional
        'complex' or 'real', defining the way to perform pi_modulus
        calculation.
        default is 'Complex'.
    sw_flag : Bool, optional
        flag to use shrinkwrap algorithm or not.
        default is True.
    sw_sigma : float, optional
        gaussian width used in sw algorithm.
        default is 0.5.
    sw_threshold : float, optional
        shreshold cut in sw algorithm.
        default is 0.1.
    sw_start : float, optional
        at which point to start to do shrinkwrap.
        defualt is 0.2
    sw_end : float, optional
        at which point to stop shrinkwrap.
        defualt is 0.8
    sw_step : float, optional
        the frequency to perform sw algorithm.
        defualt is 10
    n_iterations : int, optional
        number of iterations to run.
        default is 1000.
    cb_function : function, optional
        This is a callback function that expects to receive these
        four objects: sample_obj, obj_error, diff_error, sup_error.
        Sample_obj is a 2D array. And obj_error, diff_error, and sup_error
        are 1D array.
    cb_step : int, optional
        define plotting frequency, i.e., if plot_step = 10, plot results
        after every 10 iterations.

    Returns
    -------
    obj_avg : array
        reconstructed sample object
    error_dict : dict
        Error information for all iterations. The dict keys include
        obj_error, diff_error and sup_error. Obj_error is a list of
        the relative error of sample object. Diff_error is calculated as
        the difference between new diffraction pattern and the original
        diffraction pattern. And sup_error stores the size of the
        sample support.

    References
    ----------

    .. [1] V. Elser, "Phase retrieval by iterated projections",
        J. Opt. Soc. Am. A, vol. 20, No. 1, 2003
    """

    diffracted_pattern = np.array(diffracted_pattern)  # diffraction data
    diffracted_pattern = np.fft.fftshift(diffracted_pattern)

    pi_modulus_flag = pi_modulus_flag.lower()
    real_operation = False
    if pi_modulus_flag == "real":
        real_operation = True
    elif pi_modulus_flag == "complex":
        real_operation = False
    else:
        raise ValueError('py_modulus_flag must be one of ("complex","real") not ' "{!r}".format(pi_modulus_flag))

    gamma_1 = -1 / beta
    gamma_2 = 1 / beta

    # get support index
    outside_sup_index = sup != 1

    error_dict = {}
    obj_error = np.zeros(n_iterations)
    diff_error = np.zeros(n_iterations)
    sup_error = np.zeros(n_iterations)

    sup_old = np.zeros_like(diffracted_pattern)
    obj_avg = np.zeros_like(diffracted_pattern).astype(complex)
    avg_i = 0

    time_start = time.time()
    for n in range(n_iterations):
        obj_old = np.array(sample_obj)

        obj_a = pi_modulus(sample_obj, diffracted_pattern)
        if real_operation:
            obj_a = np.abs(obj_a)

        obj_a = (1 + gamma_2) * obj_a - gamma_2 * sample_obj
        obj_a[outside_sup_index] = 0  # define support

        obj_b = np.array(sample_obj)
        obj_b[outside_sup_index] = 0  # define support
        obj_b = (1 + gamma_1) * obj_b - gamma_1 * sample_obj

        obj_b = pi_modulus(obj_b, diffracted_pattern)
        if real_operation:
            obj_b = np.abs(obj_b)

        sample_obj += beta * (obj_a - obj_b)

        # calculate errors
        obj_error[n] = np.linalg.norm(sample_obj - obj_old) / np.linalg.norm(obj_old)
        diff_error[n] = cal_diff_error(sample_obj, diffracted_pattern)

        if sw_flag:
            if (n >= (sw_start * n_iterations)) and (n <= (sw_end * n_iterations)):
                if np.mod(n, sw_step) == 0:
                    logger.info("Refine support with shrinkwrap")
                    sup_index = find_support(sample_obj, sw_sigma, sw_threshold)
                    sup = np.zeros_like(diffracted_pattern)
                    sup[sup_index] = 1
                    outside_sup_index = sup != 1
                    sup_error[n] = np.sum(sup_old)
                    sup_old = np.array(sup)

        if cb_function and n_iterations % cb_step == 0:
            cb_function(sample_obj, obj_error, diff_error, sup_error)

        if n > start_avg * n_iterations:
            obj_avg += sample_obj
            avg_i += 1

        logger.info("%d object_chi= %f, diff_chi=%f" % (n, obj_error[n], diff_error[n]))

    obj_avg = obj_avg / avg_i
    time_end = time.time()

    logger.info("%d iterations takes %f sec" % (n_iterations, time_end - time_start))

    error_dict["obj_error"] = obj_error
    error_dict["diff_error"] = diff_error
    error_dict["sup_error"] = sup_error

    return obj_avg, error_dict
