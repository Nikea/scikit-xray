from __future__ import division

"""
Histogram

General purpose histogram classes.
"""
cimport cython

import numpy as np

cimport numpy as cnp

import logging

from ..utils import bin_edges_to_centers

logger = logging.getLogger(__name__)

DEF MAX_DIMENSIONS = 10

ctypedef fused coordnumtype:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.float32_t
    cnp.float64_t

ctypedef fused wnumtype:
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.float32_t
    cnp.float64_t

cdef void* _getarrayptr(cnp.ndarray a):
    return <void*> a.data


class Histogram:

    _always_use_fillnd = False      # FIXME remove this

    def __init__(self, binlowhigh, *args):
        """

        Parameters
        ----------
        binlowhigh : iterable
            nbin, low, high = binlowhigh
            nbin is the number of bins
            low is the left most edge
            high is the right most edge
        args : iterable
            Extra instances of binlowhigh that correspond to extra dimensions
            in the Histogram

        Notes
        -----
        The right most bin is half open
        """
        if 1 + len(args) > MAX_DIMENSIONS:
            emsg = "Cannot create histogram of more than {} dimensions."
            raise ValueError(emsg.format(MAX_DIMENSIONS))
        logger.debug('binlowhigh = {}'.format(binlowhigh))
        logger.debug('args = {}'.format(args))
        nbins = []
        lows = []
        highs = []
        for bin, low, high in [binlowhigh] + list(args):
            nbins.append(bin)
            lows.append(low)
            highs.append(high)

        logger.debug("nbins = {}".format(nbins))

        # numerical type for internal floating point arrays
        fpdtp = np.dtype(float)
        # create the numpy array to hold the results
        self._values = np.zeros(nbins, dtype=fpdtp)
        self.ndims = len(nbins)
        binsizes = [(high - low) / nbin for high, low, nbin
                    in zip(highs, lows, nbins)]
        logger.debug("nbins = {}".format(nbins))
        # store everything in a numpy array
        self._nbins = np.array(nbins, dtype=np.dtype('i')).reshape(-1)
        self._lows = np.array(lows, dtype=fpdtp).reshape(-1)
        self._highs = np.array(highs, dtype=fpdtp).reshape(-1)
        self._binsizes = np.array(binsizes, dtype=fpdtp).reshape(-1)


    def reset(self):
        """Fill the histogram array with 0
        """
        self._values.fill(0)


    def fill(self, *coords, weights=1):
        """

        Parameters
        ----------
        coords : iterable of values.  Values can be np.ndarrays, integers,
            floats, or list/tuple of int/float. The length of coords is
            equivalent to the dimensionality of the histogram.  The data
            types of the coords must be the same.
        weights: int/float/np.ndarray, optional.  Defaults to 1.
            The amount each histogram bin (determined by coords) is
            to be incremented.

        Returns
        -------

        """
        # check our arguments
        if len(coords) != self.ndims:
            emsg = "Incorrect number of arguments.  Received {} expected {}."
            raise ValueError(emsg.format(len(coords), self.ndims))

        weights = np.asarray(weights).reshape(-1)

        # for user-friendliness, translate int/float/list/tuple to arrays
        if type(coords[0]) is int or type(coords[0]) is float:
            coords = tuple(np.array([c],dtype=float) for c in coords)
        elif type(coords[0]) is tuple or type(coords[0]) is list:
            coords = tuple(np.array(c,dtype=float) for c in coords)

        if type(weights) is list or type (weights) is tuple:
            weights = tuple(np.array(w,dtype=float) for w in weights)

        nexpected = len(coords[0])
        for x in coords:
            if len(x) != nexpected:
                emsg = "Coordinate arrays must have the same length."
                raise ValueError(emsg)
        if len(weights) != 1 and len(weights) != nexpected:
            emsg = ("Weights must be scalar or have the same length "
                    "as coordinates.")
            raise ValueError(emsg)
        if self._always_use_fillnd:
            self._fillnd(coords, weights)
            return
        if len(coords) == 1:
            # compute a 1D histogram
            self._fill1d(coords[0], weights)
        elif len(coords) == 2:
            # compute a 2D histogram!
            self._fill2d(coords[0], coords[1], weights)
        else:
            # do the generalized ND histogram
            self._fillnd(coords, weights)
        return


    def _fill1d(self, cnp.ndarray[coordnumtype, ndim=1] xval,
                cnp.ndarray[wnumtype, ndim=1] weight):
        cdef cnp.ndarray[cnp.float_t, ndim=1] data = self.values
        cdef cnp.float_t low = self._lows[0]
        cdef cnp.float_t high = self._highs[0]
        cdef cnp.float_t binsize = self._binsizes[0]
        cdef int i
        cdef int wstride = 0 if weight.size == 1 else 1
        cdef int xlen = len(xval)
        for i in range(xlen):
            xidx = find_indices(xval[i], low, high, binsize)
            if xidx == -1:
                continue
            data[xidx] += weight[wstride * i]
        return


    def _fill2d(self, cnp.ndarray[coordnumtype, ndim=1] xval,
                cnp.ndarray[coordnumtype, ndim=1] yval,
                cnp.ndarray[wnumtype, ndim=1] weight):
        cdef cnp.float_t [:,:] data = self.values
        cdef cnp.float_t [:] low = self._lows
        cdef cnp.float_t [:] high = self._highs
        cdef cnp.float_t [:] binsize = self._binsizes
        cdef int i
        cdef int xlen = len(xval)
        cdef int ylen = len(yval)
        cdef int wstride = 0 if weight.size == 1 else 1
        for i in range(xlen):
            xidx = find_indices(xval[i], low[0], high[0], binsize[0])
            if xidx == -1:
                continue
            yidx = find_indices(yval[i], low[1], high[1], binsize[1])
            if yidx == -1:
                continue
            data[xidx][yidx] += weight[wstride * i]
        return


    def _fillnd(self, coords, cnp.ndarray[wnumtype, ndim=1] weight):
        # allocate pointer arrays per each supported numerical types
        cdef cnp.int64_t* aint_ptr[MAX_DIMENSIONS]
        cdef int aint_count = 0
        cdef cnp.float64_t* afloat_ptr[MAX_DIMENSIONS + 1]
        cdef int afloat_count = 0
        # determine order of coordinate arrays according to their
        # numerical data type
        numtypes = [np.dtype(int), np.dtype(float)]
        numtypeindex = {tp : i for i, tp in enumerate(numtypes)}
        ctypeindices = [numtypeindex.get(x.dtype, 999) for x in coords]
        coordsorder = np.argsort(ctypeindices, kind='mergesort')
        istrides = np.asarray(self._values.strides)[coordsorder]
        istrides //= self._values.itemsize
        cdef int dataindexstrides[MAX_DIMENSIONS]
        cdef int i
        for i in range(self.ndims):
            dataindexstrides[i] = istrides[i]
        mylows = self._lows[coordsorder]
        myhighs = self._highs[coordsorder]
        mybinsizes = self._binsizes[coordsorder]
        cdef cnp.float_t [:] low = mylows
        cdef cnp.float_t [:] high = myhighs
        cdef cnp.float_t [:] binsize = mybinsizes
        cdef cnp.float_t* data = <cnp.float_t*> _getarrayptr(self._values)
        # distribute coordinates in each dimension according to their
        # numerical type.  follow the same order as in numtypes.
        for x in coords:
            if x.dtype in (np.int64,):
                aint_ptr[aint_count] = <cnp.int64_t*> _getarrayptr(x)
                aint_count += 1
            elif x.dtype == np.float64:
                afloat_ptr[afloat_count] = <cnp.float64_t*> _getarrayptr(x)
                afloat_count += 1
            else:
                emsg = "Numpy arrays of type {} are not supported."
                raise TypeError(emsg.format(x.dtype))
        cdef int j, k
        cdef int wstride = 0 if weight.size == 1 else 1
        cdef int xlen = len(coords[0])
        cdef int xidx, widx, didx
        for i in range(xlen):
            didx = 0
            for k in range(aint_count):
                j = k
                xidx = find_indices(aint_ptr[k][i],
                                    low[j], high[j], binsize[j])
                if xidx == -1:
                    didx = -1
                    break
                didx += dataindexstrides[j] * xidx
            if didx == -1:
                continue
            for k in range(afloat_count):
                j = k + aint_count
                xidx = find_indices(afloat_ptr[k][i],
                                    low[j], high[j], binsize[j])
                if xidx == -1:
                    didx = -1
                    break
                didx += dataindexstrides[j] * xidx
            if didx == -1:
                continue
            widx = wstride * i
            data[didx] += weight[widx]
        return


    @property
    def values(self):
        return self._values

    @property
    def edges(self):
        return [np.linspace(low, high, nbin+1) for nbin, low, high
                in zip(self._nbins, self._lows, self._highs)]

    @property
    def centers(self):
        return [bin_edges_to_centers(edge) for edge in self.edges]


cdef long find_indices(coordnumtype pos, double low, double high, double binsize):
    if not (low <= pos < high):
        return -1
    return int((pos - low) / binsize)


cdef void fillonecy(coordnumtype xval, wnumtype weight,
                    cnp.float_t* pdata,
                    double low, double high, double binsize):
    iidx = find_indices(xval, low, high, binsize)
    if iidx == -1:
        return
    pdata[iidx] += weight
    return


#TODO function interface
#TODO generator interface
#TODO docs!
#TODO examples
#TODO Can we support ND histogram for mixed coordinate types?
