# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Developed at the NSLS-II, Brookhaven National Laboratory             #
# Developed by Sameera K. Abeykoon, May 2015                           #
#                                                                      #                                                                    #
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
    This module will provide analysis codes for
    X-ray Speckle Visibility Spectroscopy (XSVS)
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
from six.moves import zip
from six import string_types

import skxray.correlation as corr
import skxray.roi as roi
import speckle_visibility as spe_vis

import logging
logger = logging.getLogger(__name__)


def xsvs(sample_dict, label_array, timebin_num=2, max_cts=20, number_of_img=50):
    """
    Parameters
    ----------
    sample_dict :

    label_array :

    timebin_num : int, optional

    max_cts : int, optional

    Returns
    -------

    Note
    ----
    These implementation is based on following references
    References: text [1]_, text [2]_

    .. [1] L. Li, P. Kwasniewski, D. Oris, L Wiegart, L. Cristofolini,
       C. Carona and A. Fluerasu , "Photon statistics and speckle visibility
       spectroscopy with partially coherent x-rays" J. Synchrotron Rad.,
       vol 21, p 1288-1295, 2014.

    .. [2] R. Bandyopadhyay, A. S. Gittings, S. S. Suh, P.K. Dixon and
       D.J. Durian "Speckle-visibilty Spectroscopy: A tool to study
       time-varying dynamics" Rev. Sci. Instrum. vol 76, p  093110, 2005.

    """

    # max_cts = max_counts(sample_dict, label_array)  ????

    # number of image sets
    number_img_sets = len(sample_dict)

    # number of ROI's
    num_roi = np.max(label_array)

    # create integration times
    time_bin = spe_vis.time_bining(timebin_num, number_of_img)

    # number of items in the time bin
    num_times = len(time_bin)

    labels, indices = corr.extract_label_indices(label_array)

    speckle_cts_all = np.zeros([num_times, num_roi], dtype=np.float64)
    speckle_cts_pow_all = np.zeros([num_times, num_roi], dtype=np.float64)

    for i, samples in dict(sample_dict).iteritems():
        # Ring buffer, a buffer with periodic boundary conditions.
        # Images must be keep for up to maximum delay in buf.
        buf = np.zeros([num_times, timebin_num], dtype=np.float64)  #// matrix of buffers

        # to track processing each level
        track_level = np.zeros(num_times)

        # to increment buffer
        cur = np.ones(num_times * timebin_num)

        # to track how many images processed in each level
        img_per_level = np.zeros((num_times), dtype=np.int64)

        speckle_cts = np.zeros([num_times, num_roi],
                                        dtype=object)
        speckle_cts_pow = np.zeros([num_times, num_roi],
                                  dtype=object)
        for n, img in enumerate(samples.operands[0]):
            cur[0] = 1 + cur[0]%timebin_num   #  ?? check (1 + cur[0])%timebin_num
            # read each frame
            # Put the image into the ring buffer.
            buf[0, cur[0] - 1 ] = (np.ravel(img))[indices]

            _process(num_roi, 0, cur[0] - 1, buf, img_per_level, labels, max_cts,
            speckle_cts, speckle_cts_pow, i)

            # check whether the number of levels is one, otherwise
            # continue processing the next level
            processing = num_times > 1
            level = 1

            while processing:
                if not track_level[level]:
                    track_level[level] = 1
                    processing = 0
                else:
                    prev =  1 + (cur[level - 1] - 2)%timebin_num
                    cur[level] =  1 + cur[level]%timebin_num

                    buf[level, cur[level]-1] = (buf[level-1,
                                               prev-1] + buf[level-1, cur[level - 1] - 1])/2
                    track_level[level] = 0

                    _process(num_roi, level, cur[level]-1, buf, img_per_level,
                             labels, max_cts, speckle_cts, speckle_cts_pow, i)
                    level += 1
                    # Checking whether there is next level for processing
                    processing = level < num_times


            speckle_cts_all += (speckle_cts -
                                         speckle_cts_all)/(i + 1)
            speckle_cts_pow_all += (speckle_cts_pow -
                                   speckle_cts_pow_all)/(i + 1)
            std_dev = np.power((speckle_cts -
                                       np.power(speckle_cts_pow, 2)), .5)

    return speckle_cts_all, speckle_cts_pow_all, std_dev


def _process(num_roi, level, bufno, buf, img_per_level, labels, max_cts,
             speckle_cts, speckle_cts_pow, i):
    img_per_level[level] += 1

    for j in xrange(num_roi):
        roi_data = buf[level, bufno][labels[j+1] ]

        spe_hist, bin_edges = np.histogram(roi_data, bins=max_cts, normed=True)

        speckle_cts[level, j] += (spe_hist -
                              speckle_cts[level, j] )/(img_per_level[level])
        speckle_cts_pow[level, j] += (np.power(spe_hist, 2) -
                                         speckle_cts_pow[level, j])/(img_per_level[level])

    return None # modifies arguments in place!