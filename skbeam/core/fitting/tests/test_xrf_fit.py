from __future__ import absolute_import, division, print_function

import copy
import logging

import numpy as np
import pytest
import six
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_equal, assert_raises

from skbeam.core.fitting.base.parameter_data import e_calibration, get_para
from skbeam.core.fitting.xrf_model import (
    _STRATEGY_REGISTRY,
    ModelSpectrum,
    ParamController,
    _set_parameter_hint,
    calculate_area,
    compute_escape_peak,
    construct_linear_model,
    define_range,
    extract_strategy,
    fit_pixel_multiprocess_nnls,
    linear_spectrum_fitting,
    register_strategy,
    sum_area,
    trim,
    update_parameter_dict,
)

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO, filemode="w")


def synthetic_spectrum():
    param = get_para()
    x = np.arange(2000)
    pileup_peak = ["Si_Ka1-Si_Ka1", "Si_Ka1-Ce_La1"]
    user_peak = ["user_peak1"]
    elemental_lines = ["Ar_K", "Fe_K", "Ce_L", "Pt_M"] + pileup_peak + user_peak
    elist, matv, area_v = construct_linear_model(x, param, elemental_lines, default_area=1e5)
    # In case that y0 might be zero at certain points.
    return np.sum(matv, 1)


def test_param_controller_fail():
    param = get_para()
    PC = ParamController(param, [])
    assert_raises(ValueError, PC._add_area_param, "Ar")


def test_parameter_controller():
    param = get_para()
    pileup_peak = ["Si_Ka1-Si_Ka1", "Si_Ka1-Ce_La1"]
    elemental_lines = ["Ar_K", "Fe_K", "Ce_L", "Pt_M"] + pileup_peak
    PC = ParamController(param, elemental_lines)
    set_opt = dict(pos="hi", width="lohi", area="hi", ratio="lo")
    PC.update_element_prop(["Fe_K", "Ce_L", pileup_peak[0]], **set_opt)
    PC.set_strategy("linear")

    # check boundary value
    for k, v in six.iteritems(PC.params):
        if "Fe" in k:
            if "ratio" in k:
                assert_equal(str(v["bound_type"]), set_opt["ratio"])
            if "center" in k:
                assert_equal(str(v["bound_type"]), set_opt["pos"])
            elif "area" in k:
                assert_equal(str(v["bound_type"]), set_opt["area"])
            elif "sigma" in k:
                assert_equal(str(v["bound_type"]), set_opt["width"])
        elif ("pileup_" + pileup_peak[0].replace("-", "_")) in k:
            if "ratio" in k:
                assert_equal(str(v["bound_type"]), set_opt["ratio"])
            if "center" in k:
                assert_equal(str(v["bound_type"]), set_opt["pos"])
            elif "area" in k:
                assert_equal(str(v["bound_type"]), set_opt["area"])
            elif "sigma" in k:
                assert_equal(str(v["bound_type"]), set_opt["width"])


def test_fit():
    param = get_para()
    pileup_peak = ["Si_Ka1-Si_Ka1", "Si_Ka1-Ce_La1"]
    user_peak = ["user_peak1"]
    elemental_lines = ["Ar_K", "Fe_K", "Ce_L", "Pt_M"] + pileup_peak + user_peak
    x0 = np.arange(2000)
    y0 = synthetic_spectrum()
    default_area = 1e5
    x, y = trim(x0, y0, 100, 1300)
    MS = ModelSpectrum(param, elemental_lines)
    MS.assemble_models()

    result = MS.model_fit(x, y, weights=1 / np.sqrt(y + 1), max_nfev=200)

    # check area of each element
    for k, v in six.iteritems(result.values):
        if "area" in k:
            # error smaller than 1e-6
            assert abs(v - default_area) / default_area < 1e-6

    # multiple peak sumed, so value should be larger than one peak area 1e5
    sum_Fe = sum_area("Fe_K", result)
    assert sum_Fe > default_area

    sum_Ce = sum_area("Ce_L", result)
    assert sum_Ce > default_area

    sum_Pt = sum_area("Pt_M", result)
    assert sum_Pt > default_area

    # create full list of parameters
    PC = ParamController(param, elemental_lines)
    new_params = PC.params
    # update values
    update_parameter_dict(new_params, result)
    for k, v in six.iteritems(new_params):
        if "area" in k:
            assert_equal(v["value"], result.values[k])


def test_define_range():
    y0 = synthetic_spectrum()
    low = 0
    high = 2000
    x, y = define_range(y0, low, high, 0, 1)
    assert_equal(y0, y)

    low = 1
    high = 1000
    x, y = define_range(y0, low, high, 0, 1)
    assert len(y) == 1000


def test_extract_strategy():
    param = get_para()
    d = extract_strategy(param, "bound_type")
    assert len(d) == len(param) - 1  # 'non_fitting_values' is not included


def test_register():
    new_strategy = e_calibration
    register_strategy("e_calibration", new_strategy, overwrite=False)
    assert_equal(len(_STRATEGY_REGISTRY), 5)

    new_strategy = copy.deepcopy(e_calibration)
    new_strategy["coherent_sct_amplitude"] = "fixed"
    register_strategy("new_strategy", new_strategy)
    assert_equal(len(_STRATEGY_REGISTRY), 6)


def test_register_error():
    with pytest.raises(RuntimeError):
        new_strategy = copy.deepcopy(e_calibration)
        new_strategy["coherent_sct_amplitude"] = "fixed"
        register_strategy("e_calibration", new_strategy, overwrite=False)


def test_pre_fit():
    # No pre-defined elements. Use all possible elements activated at
    # given energy
    y0 = synthetic_spectrum()
    x0 = np.arange(len(y0))
    # the following items should appear
    item_list = ["Ar_K", "Fe_K", "compton", "elastic"]

    param = get_para()

    # fit without weights
    x, y_total, area_v = linear_spectrum_fitting(x0, y0, param, weights=None)
    for v in item_list:
        assert v in y_total

    sum1 = np.sum([v for v in y_total.values()], axis=0)
    # r squares as a measurement
    r1 = 1 - np.sum((sum1 - y0) ** 2) / np.sum((y0 - np.mean(y0)) ** 2)
    assert r1 > 0.85

    # fit with weights
    w = 1 / np.sqrt(y0 + 1)
    x, y_total, area_v = linear_spectrum_fitting(x0, y0, param, weights=w)
    for v in item_list:
        assert v in y_total
    sum2 = np.sum([v for v in y_total.values()], axis=0)
    # r squares as a measurement
    r2 = 1 - np.sum((sum2 - y0) ** 2) / np.sum((y0 - np.mean(y0)) ** 2)
    assert r2 > 0.85


def test_escape_peak():
    y0 = synthetic_spectrum()
    ratio = 0.01
    param = get_para()
    xnew, ynew = compute_escape_peak(y0, ratio, param)
    # ratio should be the same
    assert_array_almost_equal(np.sum(ynew) / np.sum(y0), ratio, decimal=3)


def test_set_param_hint():
    param = get_para()
    elemental_lines = ["Ar_K", "Fe_K", "Ce_L", "Pt_M"]
    bound_options = ["none", "lohi", "fixed", "lo", "hi"]

    MS = ModelSpectrum(param, elemental_lines)
    MS.assemble_models()

    # get compton model
    compton = MS.mod.components[0]

    for v in bound_options:
        input_param = {"bound_type": v, "max": 13.0, "min": 9.0, "value": 11.0}
        _set_parameter_hint("coherent_sct_energy", input_param, compton)
        p = compton.make_params()
        if v == "fixed":
            assert_equal(p["coherent_sct_energy"].vary, False)
        else:
            assert_equal(p["coherent_sct_energy"].vary, True)


def test_set_param():
    with pytest.raises(ValueError):
        param = get_para()
        elemental_lines = ["Ar_K", "Fe_K", "Ce_L", "Pt_M"]

        MS = ModelSpectrum(param, elemental_lines)
        MS.assemble_models()

        # get compton model
        compton = MS.mod.components[0]

        input_param = {"bound_type": "other", "max": 13.0, "min": 9.0, "value": 11.0}
        _set_parameter_hint("coherent_sct_energy", input_param, compton)


def test_pixel_fit_multiprocess():
    param = get_para()
    y0 = synthetic_spectrum()
    x = np.arange(len(y0))
    pileup_peak = ["Si_Ka1-Si_Ka1", "Si_Ka1-Ce_La1"]
    user_peak = ["user_peak1"]
    elemental_lines = ["Ar_K", "Fe_K", "Ce_L", "Pt_M"] + pileup_peak + user_peak

    default_area = 1e5
    elist, matv, area_v = construct_linear_model(x, param, elemental_lines, default_area=default_area)
    exp_data = np.zeros([2, 1, len(y0)])
    for i in range(exp_data.shape[0]):
        exp_data[i, 0, :] = y0
    results = fit_pixel_multiprocess_nnls(exp_data, matv, param, use_snip=True)
    # output area of dict
    result_map = calculate_area(elist, matv, results, param, first_peak_area=True)

    # compare input list and output elemental list
    assert_array_equal(elist, elemental_lines + ["compton", "elastic"])

    # Total len includes all the elemental list, compton, elastic and
    # two more items, which are summed area of background and r-squared
    total_len = len(elist) + 2
    assert_array_equal(results.shape, [2, 1, total_len])

    # same exp data should output same results
    assert_array_equal(results[0, :, :], results[1, :, :])

    for k, v in six.iteritems(result_map):
        assert_equal(v[0, 0], v[1, 0])
        if k in ["snip_bkg", "r_squared"]:
            # bkg is not a fitting parameter, and r_squared is just a
            # statistical output.
            # Only compare the fitting parameters, such as area of each peak.
            continue
        # compare with default value 1e5, and get difference < 1%
        assert abs(v[0, 0] * 0.01 - default_area) / default_area < 1e-2
