"""
This module contains fileIO operations and file conversion for the image
processing tool kit in the NSLS-II data analysis software package.
The functions included in this module focus on reading and writing
netCDF files. This is the file format used by Mark Rivers for
x-ray computed microtomography data collected at Argonne National Laboratory,
Sector 13BMD, GSECars.
"""

from __future__ import absolute_import, division, print_function

import os


def load_netCDF(file_name):
    """
    This function loads the specified netCDF file format data set (e.g.*.volume
    APS-Sector 13 GSECARS extension) file into a numpy array for further
    analysis.

    Required Dependencies:

    netcdf4 : Python/numpy interface to the netCDF ver. 4 library
        Package name: netcdf4-python
        Install from: https://github.com/Unidata/netcdf4-python

    numpy

    Cython -- optional

    HDF5 C library version 1.8.8 or higher
        Install from:  ftp://ftp.hdfgroup.org/HDF5/current/src
        Be sure to build with '--enable-hl --enable-shared'.

    netCDF-4 C library
        Install from:
        ftp://ftp.unidata.ucar.edu/pub/netcdf. Version 4.1.1 or higher
        Be sure to build with '--enable-netcdf-4 --enable-shared', and set
        CPPFLAGS="-I $HDF5_DIR/include" and LDFLAGS="-L $HDF5_DIR/lib", where
        $HDF5_DIR is the directory where HDF5 was installed.
        If you want OPeNDAP support, add '--enable-dap'.
        If you want HDF4 SD support, add '--enable-hdf4' and add the location
        of the HDF4 headers and library to CPPFLAGS and LDFLAGS.


    Parameters
    ----------
    file_name: string
        Complete path to the file to be loaded into memory


    Returns
    -------
    md_dict: dict
        Dictionary containing all metadata contained in the netCDF file.
        This metadata contains data collection, and experiment information
        as well as values and variables pertinent to the image data.

    data: ndarray
        ndarray containing the image data contained in the netCDF file.
        The image data is scaled using the scale factor defined in the
        netCDF metadata, if a scale factor was recorded during data
        acquisition or reconstruction. If a scale factor is not present,
        then a default value of 1.0 is used.
    """
    from netCDF4 import Dataset

    with Dataset(os.path.normpath(file_name), "r") as src_file:
        data = src_file.variables["VOLUME"]
        md_dict = src_file.__dict__
        # Check for voxel intensity scale factor and apply if value is present
        data /= data.scale_factor if data.scale_factor != 1.0 else 1.0

    # Accounts for specific case where z_pixel_size doesn't get assigned
    # even though dimensions are actuall isotropic. This occurs when
    # reconstruction is completed using tomo_recon on data collected at
    # APS-13BMD.
    if md_dict["x_pixel_size"] == md_dict["y_pixel_size"] and md_dict["z_pixel_size"] == 0.0 and data.shape[0] > 1:
        md_dict["voxel_size"] = {"value": md_dict["x_pixel_size"], "type": float, "units": ""}
    return md_dict, data
