# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# @author: Li Li (lili@bnl.gov)                                        #
# created on 09/10/2014                                                #
#                                                                      #
# Original code:                                                       #
# @author: Mirna Lerotic, 2nd Look Consulting                          #
#         http://www.2ndlookconsulting.com/                            #
# Copyright (c) 2013, Stefan Vogt, Argonne National Laboratory         #
# All rights reserved.                                                 #
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

import copy
import logging
import multiprocessing
from collections import OrderedDict

import numpy as np
import six
from lmfit import Model
from scipy.optimize import nnls

from ..constants import XrfElement as Element
from ..fitting.lineshapes import gaussian
from ..fitting.models import ComptonModel, ElasticModel, _gen_class_docs
from .background import snip_method
from .base import parameter_data as sfb_pd

logger = logging.getLogger(__name__)

# fmt: off
# emission line energy between (0, 30) keV
K_LINE = ['Li_K', 'Be_K', 'B_K', 'C_K', 'N_K', 'O_K', 'F_K', 'Ne_K',
          'Na_K', 'Mg_K', 'Al_K', 'Si_K', 'P_K', 'S_K', 'Cl_K', 'Ar_K', 'K_K',
          'Ca_K', 'Sc_K', 'Ti_K', 'V_K', 'Cr_K', 'Mn_K', 'Fe_K', 'Co_K',
          'Ni_K', 'Cu_K', 'Zn_K', 'Ga_K', 'Ge_K', 'As_K', 'Se_K', 'Br_K',
          'Kr_K', 'Rb_K', 'Sr_K', 'Y_K', 'Zr_K', 'Nb_K', 'Mo_K', 'Tc_K',
          'Ru_K', 'Rh_K', 'Pd_K', 'Ag_K', 'Cd_K', 'In_K', 'Sn_K', 'Sb_K',
          'Te_K', 'I_K']

L_LINE = ['Ga_L', 'Ge_L', 'As_L', 'Se_L', 'Br_L', 'Kr_L', 'Rb_L', 'Sr_L',
          'Y_L', 'Zr_L', 'Nb_L', 'Mo_L', 'Tc_L', 'Ru_L', 'Rh_L', 'Pd_L',
          'Ag_L', 'Cd_L', 'In_L', 'Sn_L', 'Sb_L', 'Te_L', 'I_L', 'Xe_L',
          'Cs_L', 'Ba_L', 'La_L', 'Ce_L', 'Pr_L', 'Nd_L', 'Pm_L', 'Sm_L',
          'Eu_L', 'Gd_L', 'Tb_L', 'Dy_L', 'Ho_L', 'Er_L', 'Tm_L', 'Yb_L',
          'Lu_L', 'Hf_L', 'Ta_L', 'W_L', 'Re_L', 'Os_L', 'Ir_L', 'Pt_L',
          'Au_L', 'Hg_L', 'Tl_L', 'Pb_L', 'Bi_L', 'Po_L', 'At_L', 'Rn_L',
          'Fr_L', 'Ac_L', 'Th_L', 'Pa_L', 'U_L', 'Np_L', 'Pu_L', 'Am_L']

M_LINE = ['Hf_M', 'Ta_M', 'W_M', 'Re_M', 'Os_M', 'Ir_M', 'Pt_M', 'Au_M',
          'Hg_M', 'Tl_M', 'Pb_M', 'Bi_M', 'Sm_M', 'Eu_M', 'Gd_M', 'Tb_M',
          'Dy_M', 'Ho_M', 'Er_M', 'Tm_M', 'Yb_M', 'Lu_M', 'Th_M', 'Pa_M',
          'U_M']

K_TRANSITIONS = ['ka1', 'ka2', 'ka3', 'kb1', 'kb2', 'kb3', 'kb4', 'kb5']
L_TRANSITIONS = ['la1', 'la2', 'lb1', 'lb2', 'lb3', 'lb4', 'lb5',
                 'lg1', 'lg2', 'lg3', 'lg4', 'll', 'ln']
M_TRANSITIONS = ['ma1', 'ma2', 'mb', 'mg']

TRANSITIONS_LOOKUP = {'K': K_TRANSITIONS, 'L': L_TRANSITIONS,
                      'M': M_TRANSITIONS}
# fmt: on


def element_peak_xrf(
    x,
    area,
    center,
    delta_center,
    delta_sigma,
    ratio,
    ratio_adjust,
    fwhm_offset,
    fwhm_fanoprime,
    e_offset,
    e_linear,
    e_quadratic,
    epsilon=2.96,
):
    """
    This is a function to construct xrf element peak, which is based on
    gauss profile, but more specific requirements need to be
    considered. For instance, the standard deviation is replaced by
    global fitting parameters, and energy calibration on x is taken into
    account.

    Parameters
    ----------
    x : array
        independent variable, channel number instead of energy
    area : float
        area of gaussian function
    center : float
        center position
    delta_center : float
        adjustment to center position
    delta_sigma : float
        adjustment to standard deviation
    ratio : float
        branching ratio
    ratio_adjust : float
        value used to adjust peak height
    fwhm_offset : float
        global fitting parameter for peak width
    fwhm_fanoprime : float
        global fitting parameter for peak width
    e_offset : float
        offset of energy calibration
    e_linear : float
        linear coefficient in energy calibration
    e_quadratic : float
        quadratic coefficient in energy calibration

    Returns
    -------
    array:
        gaussian peak profile
    """

    def get_sigma(center):
        temp_val = 2 * np.sqrt(2 * np.log(2))
        return np.sqrt((fwhm_offset / temp_val) ** 2 + center * epsilon * fwhm_fanoprime)

    x = e_offset + x * e_linear + x**2 * e_quadratic

    return (
        gaussian(x, area, center + delta_center, delta_sigma + get_sigma(center + delta_center))
        * ratio
        * ratio_adjust
    )


class ElementModel(Model):
    __doc__ = _gen_class_docs(element_peak_xrf)

    def __init__(self, *args, **kwargs):
        super(ElementModel, self).__init__(element_peak_xrf, *args, **kwargs)
        self.set_param_hint("epsilon", value=2.96, vary=False)


def _set_parameter_hint(param_name, input_dict, input_model):
    """
    Set parameter hint information to lmfit model from input dict.

    .. warning :: This function mutates the input values.

    Parameters
    ----------
    param_name : str
        one of the fitting parameter name
    input_dict : dict
        all the initial values and constraints for given parameters
    input_model : object
        model object used in lmfit
    """
    value = input_dict["value"]
    if input_dict["bound_type"] == "none":
        input_model.set_param_hint(name=param_name, value=value, vary=True)
    elif input_dict["bound_type"] == "fixed":
        input_model.set_param_hint(name=param_name, value=value, vary=False)
    elif input_dict["bound_type"] == "lohi":
        input_model.set_param_hint(
            name=param_name, value=value, vary=True, min=input_dict["min"], max=input_dict["max"]
        )
    elif input_dict["bound_type"] == "lo":
        input_model.set_param_hint(name=param_name, value=value, vary=True, min=input_dict["min"])
    elif input_dict["bound_type"] == "hi":
        input_model.set_param_hint(name=param_name, value=value, vary=True, max=input_dict["max"])
    else:
        raise ValueError("could not set values for {0}".format(param_name))
    logger.debug(
        " %s bound type: %s, value: %f, range: [%f, %f]",
        param_name,
        input_dict["bound_type"],
        value,
        input_dict["min"],
        input_dict["max"],
    )


def _copy_model_param_hints(target, source, params):
    """
    Copy parameters (set hints) from Parameters object (source) to a model (target)
    ( Sets the attribute 'expr' in the model hint to to None -
      apply only to incomplete models used for model evaluation)

    .. warning

       This updates ``target`` in-place

    Parameters
    ----------
    target : lmfit.Model
        The model to be updated
    source : lmfit.Model
        The model to copy from

    params : list
       The names of the parameters to copy


    Note: compatibility with lmfit 0.9.13
    The issue: scikit-beam is mostly using lmfit models to generate
    spectra of individual components based on parameters of spectral lines.
    Fitting is performed mostly using 'nnls' function from SciPy package.
    Setting 'expr' attribute of a parameter tied to some other parameter that does
    not belong to the model (not in Parameters object) will cause the program
    to crash when 'expr' is evaluated by asteval. Parameters object is created
    by calling Model.calling Model.make_params() function, which creates
    new Parameters object and sets initial values of parameters based on supplied hints.
    Once the set of hints is applied to the parameters, each 'eval' string
    is evaluated by asteval (this was not the case in lmfit 0.8.3). If the expression
    'expr' refers to a parameter that does not belong to the set, the program crashes.

    Example: a model describing spectrum for the line 'Ar_ka1_fwhm_offset' must be
    initialized with the value of a global parameter 'fwhm_offset' and then tied to
    it during fitting ('fwhm_offset' determines fwhm offset for each line).
    The parameter named 'fwhm_offset' does not exist in the set of parameters that
    describe a single line, so if we set 'expr="fwhm_offset"', then the program
    will crash during Model.make_params() call. On the other hand, if we are
    creating complete model for fitting that contain parameters of multiple lines as well as
    global parameters, Model.make_params() call will succeed.

    Solution:
      -- Incomplete single-line models (used for setting up linear model): set 'expr=None'
         in the hints before calling Model.make_params(), i.e. use this function.
      -- Complete models (used for fitting): set 'expr=label', where label is the name
         of a global parameter, after all parameters are assembled, i.e. use the function
         _copy_model_param_hints_EXPR (see below). Since typical approach in scikit-beam is
         to assemble complex models out of multitude of incomplete single-line models and
         _copy_model_param_hints will be called for initialization of each of the single-line
         models.
    """

    for label in params:
        target.set_param_hint(label, value=source[label].value, expr=None)  # Originally was expr=label


def _copy_model_param_hints_EXPR(target, source, *, param_hints_elements=None, param_hints_elastic=None):
    """
    Copy parameters (set hints) from Parameters object (source) to a model (target)
    ( Sets the attribute 'expr' in the model hint to the copied parameter name,
      use only on complete models that can be used for model fitting)

    .. warning

       This updates ``target`` in-place

    Parameters
    ----------
    target : lmfit.Model
        The model to be updated
    source : lmfit.Parameters
        The model to copy from
    params_hints_elements : list
       The names of the parameters to copy for element models
    params_hints_ellastic : list
       The names of the parameters to copy for elastic scattering model


    Note: function is part of the fix for compatibility with lmfit 0.9.13.
    Dmitri G. (08/19/2019)
    """

    # Only parameters that describe spectral lines and contain substring defined by 'label'
    #    should be tied to global parameter defined by 'label'. The global parameter should
    #    not be tied to itself (label != hint_name). Also the parameter should not contain
    #    prefix "compton", since compton lines have their own set of parameters with matching
    #    names, but not tied to the parameters of spectral lines.

    if param_hints_elements is None:
        param_hints_elements = []
    if param_hints_elastic is None:
        param_hints_elastic = []

    for hint_name, hint_values in target.param_hints.items():
        # Compton scattering component has its independent set of parameters
        if hint_name.startswith("compton"):
            continue
        # The sets of hints for ellastic and element models are different
        if hint_name.startswith("elastic"):
            params = param_hints_elastic
        else:
            params = param_hints_elements
        # Now go through the selected list of labels
        for label in params:
            value = source[label].value
            if hint_name.endswith(label) and (label != hint_name) and not hint_name.startswith("compton"):
                hint_values[label] = value
                hint_values["expr"] = label


def update_parameter_dict(param, fit_results):
    """
    Update fitting parameters dictionary according to given fitting results,
    usually obtained from previous run.

    .. warning :: This function mutates param.

    Parameters
    ----------
    param : dict
        saving all the fitting values and their bounds
    fit_results : object
        ModelFit object from lmfit
    """
    elastic_list = ["coherent_sct_amplitude", "coherent_sct_energy"]
    for k, v in six.iteritems(param):
        if k in elastic_list:
            k_temp = "elastic_" + k
        else:
            k_temp = k.replace("-", "_")  # pileup peak, i.e., 'Cl_K-Cl_K'
        if k_temp in fit_results.values:
            param[k]["value"] = float(fit_results.values[k_temp])
        elif k == "non_fitting_values":
            logger.debug("Ignore non fitting values.")
        else:
            logger.warning("values not updated: {}".format(k))


_STRATEGY_REGISTRY = {
    "linear": sfb_pd.linear,
    "adjust_element": sfb_pd.adjust_element,
    "e_calibration": sfb_pd.e_calibration,
    "fit_with_tail": sfb_pd.fit_with_tail,
    "free_more": sfb_pd.free_more,
}


def register_strategy(key, strategy, overwrite=True):
    """
    Register new strategy.

    Parameters
    ----------
    key : str
        strategy name
    strategy : dict
        bound for every parameter
    """
    if (not overwrite) and (key in _STRATEGY_REGISTRY):
        if _STRATEGY_REGISTRY[key] is strategy:
            return
        raise RuntimeError("You are trying to overwrite an " "existing strategy: {}".format(key))
    _STRATEGY_REGISTRY[key] = strategy


def set_parameter_bound(param, bound_option, extra_config=None):
    """
    Update the default value of bounds.

    .. warning :: This function mutates the input values.

    Parameters
    ----------
    param : dict
        saving all the fitting values and their bounds
    bound_option : str
        define bound type
    extra_config : dict
        strategy-specific configuration
    """
    strat_dict = _STRATEGY_REGISTRY[bound_option]
    if extra_config is None:
        extra_config = dict()
    for k, v in six.iteritems(param):
        if k == "non_fitting_values":
            continue
        try:
            v["bound_type"] = strat_dict[k]
        except KeyError:
            v["bound_type"] = extra_config.get(k, "fixed")


# This dict is used to update the current parameter dict to dynamically change
# the input data and do the fitting. The user can adjust parameters such as
# position, width, area or branching ratio.
PARAM_DEFAULTS = {
    "area": {"bound_type": "none", "max": 1000000000.0, "min": 0.0, "value": 1000.0},
    "pos": {"bound_type": "fixed", "max": 0.005, "min": -0.005, "value": 0.0},
    "ratio": {"bound_type": "fixed", "max": 5.0, "min": 0.1, "value": 1.0},
    "width": {"bound_type": "fixed", "max": 0.02, "min": -0.02, "value": 0.0},
}


class ParamController(object):
    """
    Update element peak information in parameter dictionary.
    This is an important step in dynamical fitting.
    """

    def __init__(self, params, elemental_lines):
        """
        Parameters
        ----------
        params : dict
            saving all the fitting values and their bounds
        elemental_lines : list
            e.g., ['Na_K', Mg_K', 'Pt_M'] refers to the
            K lines of Sodium, the K lines of Magnesium, and the M
            lines of Platinum
        """

        self._original_params = copy.deepcopy(params)
        self.params = copy.deepcopy(params)
        self.element_list = list(elemental_lines)  # to copy it
        self.element_linenames = get_activated_lines(
            self.params["coherent_sct_energy"]["value"], self.element_list
        )
        self._element_strategy = dict()
        self._initialize_params()

    def _initialize_params(self):
        """
        Add all element information, such as pos, width, ratio into
        parameter dict.
        """
        for element in self.element_list:
            for kind in ["pos", "width", "area", "ratio"]:
                self.add_param(kind, element)

    def set_strategy(self, strategy_name):
        """
        Update the default value of bounds.

        Parameters
        ----------
        strategy_name : str
            fitting strategy that this Model will use
        """
        set_parameter_bound(self.params, strategy_name, self._element_strategy)

    def update_element_prop(self, element_list, **kwargs):
        """
        Update element properties, such as pos, width, area and ratio.

        Parameters
        ----------
        element_list : list
            define which element to update
        kwargs : dict
            define what kind of property to change

        Returns
        -------
        dict : updated value
        """
        for element in element_list:
            for kind, constraint in six.iteritems(kwargs):
                self.add_param(kind, element, constraint)

    def add_param(self, kind, element, constraint=None):
        """
        Create a Parameter controlling peak position, width,
        branching ratio, or area.

        Parameters
        ----------
        kind : {'pos', 'width', 'ratio', 'area'}
        element : str
            element name
        constraint : {'lo', 'hi', 'lohi', 'fixed', 'none'}, optional
            default "bound strategy" (fitting constraints)

        .. note:: Element that can be add include elemental peak, pileup peak and user peak.
        User peak is a fake peak, but can be used as a quick replacement for pileup peaks,
        escape peaks or other unknown peaks due to detector defects or scattering process.
        """
        if element not in self.element_list:
            self.element_list.append(element)
            self.element_linenames.extend(
                get_activated_lines(self.params["coherent_sct_energy"]["value"], [element])
            )
        if kind == "area":
            return self._add_area_param(element, constraint)

        PARAM_SUFFIXES = {"pos": "delta_center", "width": "delta_sigma", "ratio": "ratio_adjust"}
        param_suffix = PARAM_SUFFIXES[kind]

        if len(element) <= 4:
            element, line = element.split("_")
            transitions = TRANSITIONS_LOOKUP[line]

            # Mg_L -> Mg_la1, which xraylib wants
            linenames = ["{0}_{1}".format(element, t) for t in transitions]

            for linename in linenames:
                # check if the line is activated
                if linename not in self.element_linenames:
                    continue
                param_name = "_".join((str(linename), param_suffix))  # as in lmfit Model
                new_pos = PARAM_DEFAULTS[kind].copy()
                if constraint is not None:
                    self._element_strategy[param_name] = constraint
                self.params.update({param_name: new_pos})
        elif "user" in element.lower():  # for user peak
            param_name = "_".join((element, param_suffix))
            new_pos = PARAM_DEFAULTS[kind].copy()
            if constraint is not None:
                self._element_strategy[param_name] = constraint
            # update parameter in place
            self.params.update({param_name: new_pos})
        else:  # for pileup peak
            linename = "pileup_" + element.replace("-", "_")
            param_name = linename + param_suffix  # as in lmfit Model
            new_pos = PARAM_DEFAULTS[kind].copy()
            if constraint is not None:
                self._element_strategy[param_name] = constraint
            # update parameter in place
            self.params.update({param_name: new_pos})

    def _add_area_param(self, element, constraint=None):
        """
        Special case for adding an area Parameter because
        we only ever fit the first peak area.

        Helper function called in self.add_param
        """
        if element in K_LINE:
            element = element.split("_")[0]
            param_name = str(element) + "_ka1_area"
        elif element in L_LINE:
            element = element.split("_")[0]
            param_name = str(element) + "_la1_area"
        elif element in M_LINE:
            element = element.split("_")[0]
            param_name = str(element) + "_ma1_area"
        elif "user" in element.lower():  # user peak
            param_name = str(element) + "_area"
        elif "-" in element:  # pileup peaks
            param_name = "pileup_" + element.replace("-", "_")
            param_name += "_area"
        else:
            raise ValueError("{} is not a well formed element string".format(element))

        new_area = PARAM_DEFAULTS["area"].copy()
        if constraint is not None:
            self._element_strategy[param_name] = constraint

        # update parameter in place
        self.params.update({param_name: new_area})


def sum_area(elemental_line, result_val):
    """
    Return the total area for given element.

    Parameters
    ----------
    elemental_line : str
        name of a given element line, such as Na_K
    result_val : obj
        result obj from lmfit to save all the fitting results

    Returns
    -------
    sumv : float
        the total area
    """
    element, line = elemental_line.split("_")
    transitions = TRANSITIONS_LOOKUP[line]

    sumv = 0

    for line_n in transitions:
        partial_name = "_".join((element, line_n))
        full_name = "_".join((partial_name, "area"))
        if full_name in result_val.values:
            tmp = 1
            for post_fix in ["area", "ratio", "ratio_adjust"]:
                tmp *= result_val.values["_".join((partial_name, post_fix))]
            sumv += tmp
    return sumv


class ModelSpectrum(object):
    """
    Construct Fluorescence spectrum which includes elastic peak,
    compton and element peaks.
    """

    def __init__(self, params, elemental_lines):
        """
        Parameters
        ----------
        params : dict
            saving all the fitting values and their bounds
        elemental_lines : list
            e.g., ['Na_K', Mg_K', 'Pt_M'] refers to the
            K lines of Sodium, the K lines of Magnesium, and the M
            lines of Platinum
        """

        # Names of global(common) model parameters used in full assembled element models
        self._global_param_hints_elements_to_copy = [
            "e_offset",
            "e_linear",
            "e_quadratic",
            "fwhm_offset",
            "fwhm_fanoprime",
        ]
        # ... for ellastic scattering model
        self._global_param_hints_elastic_to_copy = [
            "e_offset",
            "e_linear",
            "e_quadratic",
            "fwhm_offset",
            "fwhm_fanoprime",
            "coherent_sct_energy",
        ]

        self.params = copy.deepcopy(params)
        self.elemental_lines = list(elemental_lines)  # to copy
        self.incident_energy = self.params["coherent_sct_energy"]["value"]
        self.epsilon = self.params["non_fitting_values"]["epsilon"]
        self.setup_compton_model()
        self.setup_elastic_model()

    def setup_compton_model(self):
        """
        setup parameters related to Compton model
        """
        compton = ComptonModel()

        compton_list = [
            "coherent_sct_energy",
            "compton_amplitude",
            "compton_angle",
            "fwhm_offset",
            "fwhm_fanoprime",
            "e_offset",
            "e_linear",
            "e_quadratic",
            "compton_gamma",
            "compton_f_tail",
            "compton_f_step",
            "compton_fwhm_corr",
            "compton_hi_gamma",
            "compton_hi_f_tail",
        ]

        logger.debug("Started setting up parameters for compton model")
        for name in compton_list:
            if name in self.params.keys():
                _set_parameter_hint(name, self.params[name], compton)
        logger.debug(" Finished setting up parameters for compton model.")
        compton.set_param_hint("epsilon", value=self.epsilon, vary=False)

        self.compton_param = compton.make_params()
        self.compton = compton

    def setup_elastic_model(self):
        """
        setup parameters related to Elastic model
        """

        # The list is changed later in the function, so create a copy
        param_hints_elastic = self._global_param_hints_elastic_to_copy.copy()

        elastic = ElasticModel(prefix="elastic_")

        logger.debug("Started setting up parameters for elastic model")

        # set constraints for the global parameters from the Compton model
        _copy_model_param_hints(elastic, self.compton_param, param_hints_elastic)

        # with_amplitude, parameters can be updated from self.param dict
        param_hints_elastic.append("coherent_sct_amplitude")
        for item in param_hints_elastic:
            if item in self.params.keys():
                _set_parameter_hint(item, self.params[item], elastic)

        elastic.set_param_hint("epsilon", value=self.epsilon, vary=False)
        logger.debug(" Finished setting up parameters for elastic model.")
        self.elastic = elastic

    def setup_element_model(self, elemental_line, default_area=1e5):
        """
        Construct element model. Elemental line that can be add include elemental
        peak, pileup peak and user peak. User peak is a fake peak, but can be used
        as a quick replacement for pileup peaks, escape peaks or other unknown peaks
        due to detector defects or scattering process.

        Parameters
        ----------
        elemental_line : str
            elemental line, such as 'Fe_K'
            pileup peak, such as 'Si_Ka1-Si_Ka1'
            user peak, such as 'user_peak1', 'user_peak2'
        default_area : float, optional
            value for the initial area of a given element
            default is 1e5, found to be a good value
        """

        incident_energy = self.incident_energy
        parameter = self.params

        all_element_mod = None
        # global parameters

        param_hints_to_copy = self._global_param_hints_elements_to_copy

        if elemental_line in K_LINE:
            element = elemental_line.split("_")[0]
            e = Element(element)
            if e.cs(incident_energy)["ka1"] == 0:
                logger.debug("%s Ka emission line is not activated " "at this energy %f", element, incident_energy)
                return

            logger.debug(" --- Started building %s peak. ---", element)

            for num, item in enumerate(e.emission_line.all):
                line_name = item[0]
                val = item[1]

                if "k" not in line_name:
                    continue

                if e.cs(incident_energy)[line_name] == 0:
                    continue

                element_mod = ElementModel(prefix=str(element) + "_" + str(line_name) + "_")

                # copy the fixed parameters from the Compton model
                _copy_model_param_hints(element_mod, self.compton_param, param_hints_to_copy)

                element_mod.set_param_hint("epsilon", value=self.epsilon, vary=False)

                area_name = str(element) + "_" + str(line_name) + "_area"
                if area_name in parameter:
                    default_area = parameter[area_name]["value"]

                if line_name == "ka1":
                    element_mod.set_param_hint("area", value=default_area, vary=True, min=0)
                    element_mod.set_param_hint("delta_center", value=0, vary=False)
                    element_mod.set_param_hint("delta_sigma", value=0, vary=False)
                elif line_name == "ka2":
                    element_mod.set_param_hint(
                        "area", value=default_area, vary=True, expr=str(element) + "_ka1_" + "area"
                    )
                    element_mod.set_param_hint(
                        "delta_sigma", value=0, vary=False, expr=str(element) + "_ka1_" + "delta_sigma"
                    )
                    element_mod.set_param_hint(
                        "delta_center", value=0, vary=False, expr=str(element) + "_ka1_" + "delta_center"
                    )
                else:
                    element_mod.set_param_hint(
                        "area", value=default_area, vary=True, expr=str(element) + "_ka1_" + "area"
                    )
                    element_mod.set_param_hint("delta_center", value=0, vary=False)
                    element_mod.set_param_hint("delta_sigma", value=0, vary=False)

                # area needs to be adjusted
                if area_name in parameter:
                    _set_parameter_hint(area_name, parameter[area_name], element_mod)

                element_mod.set_param_hint("center", value=val, vary=False)
                ratio_v = e.cs(incident_energy)[line_name] / e.cs(incident_energy)["ka1"]
                element_mod.set_param_hint("ratio", value=ratio_v, vary=False)
                element_mod.set_param_hint("ratio_adjust", value=1, vary=False)
                logger.debug(
                    " {0} {1} peak is at energy {2} with"
                    " branching ratio {3}.".format(element, line_name, val, ratio_v)
                )

                # position needs to be adjusted
                pos_name = element + "_" + str(line_name) + "_delta_center"
                if pos_name in parameter:
                    _set_parameter_hint("delta_center", parameter[pos_name], element_mod)

                # width needs to be adjusted
                width_name = element + "_" + str(line_name) + "_delta_sigma"
                if width_name in parameter:
                    _set_parameter_hint("delta_sigma", parameter[width_name], element_mod)

                # branching ratio needs to be adjusted
                ratio_name = element + "_" + str(line_name) + "_ratio_adjust"
                if ratio_name in parameter:
                    _set_parameter_hint("ratio_adjust", parameter[ratio_name], element_mod)

                if all_element_mod:
                    all_element_mod += element_mod
                else:
                    all_element_mod = element_mod
            logger.debug("Finished building element peak for %s", element)

        elif elemental_line in L_LINE:
            element = elemental_line.split("_")[0]
            e = Element(element)
            if e.cs(incident_energy)["la1"] == 0:
                logger.debug(
                    "{0} La1 emission line is not activated " "at this energy {1}".format(element, incident_energy)
                )
                return

            for num, item in enumerate(e.emission_line.all):
                line_name = item[0]
                val = item[1]

                if "l" not in line_name:
                    continue

                if e.cs(incident_energy)[line_name] == 0:
                    continue

                element_mod = ElementModel(prefix=str(element) + "_" + str(line_name) + "_")

                # copy the fixed parameters from the Compton model
                _copy_model_param_hints(element_mod, self.compton_param, param_hints_to_copy)

                element_mod.set_param_hint("epsilon", value=self.epsilon, vary=False)

                area_name = str(element) + "_" + str(line_name) + "_area"
                if area_name in parameter:
                    default_area = parameter[area_name]["value"]

                if line_name == "la1":
                    element_mod.set_param_hint("area", value=default_area, vary=True)
                else:
                    element_mod.set_param_hint(
                        "area", value=default_area, vary=True, expr=str(element) + "_la1_" + "area"
                    )

                # area needs to be adjusted
                if area_name in parameter:
                    _set_parameter_hint(area_name, parameter[area_name], element_mod)

                element_mod.set_param_hint("center", value=val, vary=False)
                element_mod.set_param_hint("sigma", value=1, vary=False)
                element_mod.set_param_hint(
                    "ratio", value=e.cs(incident_energy)[line_name] / e.cs(incident_energy)["la1"], vary=False
                )

                element_mod.set_param_hint("delta_center", value=0, vary=False)
                element_mod.set_param_hint("delta_sigma", value=0, vary=False)
                element_mod.set_param_hint("ratio_adjust", value=1, vary=False)

                # position needs to be adjusted
                pos_name = element + "_" + str(line_name) + "_delta_center"
                if pos_name in parameter:
                    _set_parameter_hint("delta_center", parameter[pos_name], element_mod)

                # width needs to be adjusted
                width_name = element + "_" + str(line_name) + "_delta_sigma"
                if width_name in parameter:
                    _set_parameter_hint("delta_sigma", parameter[width_name], element_mod)

                # branching ratio needs to be adjusted
                ratio_name = element + "_" + str(line_name) + "_ratio_adjust"
                if ratio_name in parameter:
                    _set_parameter_hint("ratio_adjust", parameter[ratio_name], element_mod)
                if all_element_mod:
                    all_element_mod += element_mod
                else:
                    all_element_mod = element_mod

        elif elemental_line in M_LINE:
            element = elemental_line.split("_")[0]
            e = Element(element)
            if e.cs(incident_energy)["ma1"] == 0:
                logger.debug(
                    "{0} ma1 emission line is not activated " "at this energy {1}".format(element, incident_energy)
                )
                return

            for num, item in enumerate(e.emission_line.all):
                line_name = item[0]
                val = item[1]

                if "m" not in line_name:
                    continue

                if e.cs(incident_energy)[line_name] == 0:
                    continue

                element_mod = ElementModel(prefix=str(element) + "_" + str(line_name) + "_")

                # copy the fixed parameters from the Compton model
                _copy_model_param_hints(element_mod, self.compton_param, param_hints_to_copy)

                element_mod.set_param_hint("epsilon", value=self.epsilon, vary=False)

                area_name = str(element) + "_" + str(line_name) + "_area"
                if area_name in parameter:
                    default_area = parameter[area_name]["value"]

                if line_name == "ma1":
                    element_mod.set_param_hint("area", value=default_area, vary=True)
                else:
                    element_mod.set_param_hint(
                        "area", value=default_area, vary=True, expr=str(element) + "_ma1_" + "area"
                    )

                # area needs to be adjusted
                if area_name in parameter:
                    _set_parameter_hint(area_name, parameter[area_name], element_mod)

                element_mod.set_param_hint("center", value=val, vary=False)
                element_mod.set_param_hint("sigma", value=1, vary=False)
                element_mod.set_param_hint(
                    "ratio", value=e.cs(incident_energy)[line_name] / e.cs(incident_energy)["ma1"], vary=False
                )

                element_mod.set_param_hint("delta_center", value=0, vary=False)
                element_mod.set_param_hint("delta_sigma", value=0, vary=False)
                element_mod.set_param_hint("ratio_adjust", value=1, vary=False)

                if area_name in parameter:
                    _set_parameter_hint(area_name, parameter[area_name], element_mod)

                # position needs to be adjusted
                pos_name = element + "_" + str(line_name) + "_delta_center"
                if pos_name in parameter:
                    _set_parameter_hint("delta_center", parameter[pos_name], element_mod)

                # width needs to be adjusted
                width_name = element + "_" + str(line_name) + "_delta_sigma"
                if width_name in parameter:
                    _set_parameter_hint("delta_sigma", parameter[width_name], element_mod)

                # branching ratio needs to be adjusted
                ratio_name = element + "_" + str(line_name) + "_ratio_adjust"
                if ratio_name in parameter:
                    _set_parameter_hint("ratio_adjust", parameter[ratio_name], element_mod)

                if all_element_mod:
                    all_element_mod += element_mod
                else:
                    all_element_mod = element_mod

        elif "user" in elemental_line.lower():  # user peak
            logger.debug("Started setting up user peak: {}".format(elemental_line))

            e_cen = 5.0  # user peak is set 5 keV every time, this value is not important

            pre_name = elemental_line + "_"
            element_mod = ElementModel(prefix=pre_name)

            # copy the fixed parameters from the Compton model
            _copy_model_param_hints(element_mod, self.compton_param, param_hints_to_copy)

            element_mod.set_param_hint("epsilon", value=self.epsilon, vary=False)

            area_name = pre_name + "area"
            if area_name in parameter:
                default_area = parameter[area_name]["value"]

            element_mod.set_param_hint("area", value=default_area, vary=True, min=0)
            element_mod.set_param_hint("delta_center", value=0, vary=False)
            element_mod.set_param_hint("delta_sigma", value=0, vary=False)
            element_mod.set_param_hint("center", value=e_cen, vary=False)
            element_mod.set_param_hint("ratio", value=1.0, vary=False)
            element_mod.set_param_hint("ratio_adjust", value=1, vary=False)

            # area needs to be adjusted
            if area_name in parameter:
                _set_parameter_hint(area_name, parameter[area_name], element_mod)

            # position needs to be adjusted
            pos_name = pre_name + "delta_center"
            if pos_name in parameter:
                _set_parameter_hint("delta_center", parameter[pos_name], element_mod)

            # width needs to be adjusted
            width_name = pre_name + "delta_sigma"
            if width_name in parameter:
                _set_parameter_hint("delta_sigma", parameter[width_name], element_mod)

            # branching ratio needs to be adjusted
            ratio_name = pre_name + "ratio_adjust"
            if ratio_name in parameter:
                _set_parameter_hint("ratio_adjust", parameter[ratio_name], element_mod)

            all_element_mod = element_mod

        else:
            logger.debug("Started setting up pileup peaks for {}".format(elemental_line))

            element_line1, element_line2 = elemental_line.split("-")

            e1_cen = get_line_energy(element_line1)
            e2_cen = get_line_energy(element_line2)

            # no '-' allowed in prefix name in lmfit
            pre_name = "pileup_" + elemental_line.replace("-", "_") + "_"
            element_mod = ElementModel(prefix=pre_name)

            # copy the fixed parameters from the Compton model
            _copy_model_param_hints(element_mod, self.compton_param, param_hints_to_copy)

            element_mod.set_param_hint("epsilon", value=self.epsilon, vary=False)

            area_name = pre_name + "area"
            if area_name in self.params:
                default_area = self.params[area_name]["value"]

            element_mod.set_param_hint("area", value=default_area, vary=True, min=0)
            element_mod.set_param_hint("delta_center", value=0, vary=False)
            element_mod.set_param_hint("delta_sigma", value=0, vary=False)

            # area needs to be adjusted
            if area_name in self.params:
                _set_parameter_hint(area_name, self.params[area_name], element_mod)

            element_mod.set_param_hint("center", value=e1_cen + e2_cen, vary=False)
            element_mod.set_param_hint("ratio", value=1.0, vary=False)
            element_mod.set_param_hint("ratio_adjust", value=1, vary=False)

            # position needs to be adjusted
            pos_name = pre_name + "delta_center"
            if pos_name in self.params:
                _set_parameter_hint("delta_center", self.params[pos_name], element_mod)

            # width needs to be adjusted
            width_name = pre_name + "delta_sigma"
            if width_name in self.params:
                _set_parameter_hint("delta_sigma", self.params[width_name], element_mod)

            # branching ratio needs to be adjusted
            ratio_name = pre_name + "ratio_adjust"
            if ratio_name in self.params:
                _set_parameter_hint("ratio_adjust", self.params[ratio_name], element_mod)

            all_element_mod = element_mod
        return all_element_mod

    def assemble_models(self):
        """
        Put all models together to form a spectrum.
        """
        self.mod = self.compton + self.elastic

        for element in self.elemental_lines:
            self.mod += self.setup_element_model(element)

        # This is just a copy of the parameter list. Multiple occurrances.
        param_hints_elements = self._global_param_hints_elements_to_copy
        param_hints_elastic = self._global_param_hints_elastic_to_copy

        _copy_model_param_hints_EXPR(
            self.mod,
            self.compton_param,
            param_hints_elements=param_hints_elements,
            param_hints_elastic=param_hints_elastic,
        )

    def model_fit(self, channel_number, spectrum, weights=None, method="leastsq", **kwargs):
        """
        Parameters
        ----------
        channel_number : array
            independent variable
        spectrum : array
            intensity
        weights : array, optional
            weight for fitting
        method : str
            default as leastsq
        kwargs : dict
            fitting criteria, such as max number of iteration

        Returns
        -------
        result object from lmfit
        """

        pars = self.mod.make_params()
        result = self.mod.fit(spectrum, pars, x=channel_number, weights=weights, method=method, **kwargs)

        return result


def get_line_energy(elemental_line):
    """Return the energy of the emission line.

    Parameters
    ----------
    elemental_line : str
        For instance, Ca_K will return energy of the peak for Ca_ka1 line,
        Ca_Kb will return energy for Ca_Kb1 line and Ca_Kb2 line will return
        energy for Ca_Kb2 line. The letters in the element name (e.g. Ca) should
        be capitalized properly. Letters in the line name may be arbitrarily
        capitalized, e.g. Ca_kb1 is equivalent to Ca_Kb1 or Ca_KB1.
        The function is primarily used for finding center of a pileup peak,
        so typical format for the emission line is Ca_Ka1.

    Returns
    -------
    float :
        energy of emission line
    """
    name, line = elemental_line.split("_")
    line = line.lower()

    # Complete the emission line name
    if len(line) == 1:  # elemental line format 'Ca_K'
        line += "a1"
    elif len(line) == 2:  # elemental line format 'Ca_Ka'
        line += "1"

    e = Element(name)
    e_cen = e.emission_line[line]
    return e_cen


def trim(x, y, low, high):
    """
    Mask two arrays applying bounds to the first array.

    Parameters
    ----------
    x : array
        independent variable
    y : array
        dependent variable
    low : float
        low bound
    high : float
        high bound

    Returns
    -------
    array :
        x with new range
    array :
        y with new range
    """
    mask = (x >= low) & (x <= high)
    return x[mask], y[mask]


def define_range(data, low, high, a0, a1):
    """
    Cut x range based on offset and linear term of a linear function.
    a0 and a1 should be defined in param_dict.

    Parameters
    ----------
    data : 1D array
        raw spectrum
    low : float
        low bound in KeV
    high : float
        high bound in KeV
    a0 : float
        offset term of energy calibration
    a1 : float
        linear term of energy calibration

    Returns
    -------
    x : array
        trimmed channel number
    y : array
        trimmed spectrum according to x
    """
    x = np.arange(data.size)
    low_new = int(np.around((low - a0) / a1))
    high_new = int(np.around((high - a0) / a1))
    x0, y0 = trim(x, data, low_new, high_new)
    return x0, y0


def extract_strategy(param, name):
    """
    Extract given strategy from param dict.

    Parameters
    ----------
    param : dict
        saving all parameters for data fitting
    name : str
        strategy name

    Returns
    -------
    dict :
        with given strategy as value
    """
    param_new = copy.deepcopy(param)
    return {k: v[name] for k, v in six.iteritems(param_new) if k != "non_fitting_values"}


def compute_escape_peak(spectrum, ratio, params, escape_e=1.73998):
    """
    Calculate the escape peak for given detector.

    Parameters
    ----------
    spectrum : array
        original, uncorrected spectrum
    ratio : float
        ratio of shadow to full spectrum
    param : dict
        fitting parameters
    escape_e : float
        Units: keV
        By default, 1.73998 (Ka1 line of Si)

    Returns
    -------
    array:
        x after shift, and adjusted y

    """
    x = np.arange(len(spectrum))

    x = params["e_offset"]["value"] + params["e_linear"]["value"] * x + params["e_quadratic"]["value"] * x**2

    result = x - escape_e, spectrum * ratio
    return result


def construct_linear_model(channel_number, params, elemental_lines, default_area=100):
    """
    Create spectrum with parameters given from params.

    Parameters
    ----------
    channel_number : array
        N.B. This is the raw independent variable, not energy.
    params : dict
        fitting parameters
    elemental_lines : list
            e.g., ['Na_K', Mg_K', 'Pt_M'] refers to the
            K lines of Sodium, the K lines of Magnesium, and the M
            lines of Platinum
    default_area : float
        value for the initial area of a given element

    Returns
    -------
    selected_elements : list
        selected elements for given energy
    matv : array
        matrix for linear fitting
    element_area : dict
        area of the given elements
    """
    MS = ModelSpectrum(params, elemental_lines)

    selected_elements = []
    matv = []
    element_area = {}

    for elemental_line in elemental_lines:
        e_model = MS.setup_element_model(elemental_line, default_area=default_area)
        if e_model:
            p = e_model.make_params()
            for k, v in six.iteritems(p):
                if "area" in k:
                    element_area.update({elemental_line: v.value})

            y_temp = e_model.eval(x=channel_number, params=p)
            matv.append(y_temp)
            selected_elements.append(elemental_line)

    p = MS.compton.make_params()
    y_temp = MS.compton.eval(x=channel_number, params=p)
    matv.append(y_temp)
    element_area.update({"compton": p["compton_amplitude"].value})
    selected_elements.append("compton")

    p = MS.elastic.make_params()
    y_temp = MS.elastic.eval(x=channel_number, params=p)
    matv.append(y_temp)
    element_area.update({"elastic": p["elastic_coherent_sct_amplitude"].value})
    selected_elements.append("elastic")

    matv = np.array(matv)
    matv = matv.transpose()
    return selected_elements, matv, element_area


def nnls_fit(spectrum, expected_matrix, weights=None):
    """
    Non-negative least squares fitting.

    Parameters
    ----------
    spectrum : array
        spectrum of experiment data
    expected_matrix : array
        2D matrix of activated element spectrum
    weights : array, optional
        for weighted nnls fitting. Setting weights as None means fitting
        without weights.

    Returns
    -------
    results : array
        weights of different element
    residue : float
        error

    .. note:: nnls is chosen as amplitude of each element should not be negative.
    """
    if weights is not None:
        expected_matrix = np.transpose(np.multiply(np.transpose(expected_matrix), np.sqrt(weights)))
        spectrum = np.multiply(spectrum, np.sqrt(weights))
    return nnls(expected_matrix, spectrum)


def linear_spectrum_fitting(x, y, params, elemental_lines=None, weights=None):
    """
    Fit a spectrum to a linear model.

    This is a convenience function that wraps up construct_linear_model
    and nnls_fit.

    Parameters
    ----------
    x : array
        channel array
    y : array
        spectrum intensity
    param : dict
        fitting parameters
    elemental_lines : list, optional
            e.g., ['Na_K', Mg_K', 'Pt_M'] refers to the
            K lines of Sodium, the K lines of Magnesium, and the M
            lines of Platinum. If elemental_lines is set as None,
            all the possible lines activated at given energy will be used.
    weights : array, optional
        for weighted nnls fitting. Setting weights as None means fitting
        without weights.

    Returns
    -------
    x_energy : array
        x axis with unit in energy
    result_dict : dict
        Fitting results
    area_dict : dict
        the area of the first main peak, such as Ka1, of a given element
    """
    if elemental_lines is None:
        elemental_lines = K_LINE + L_LINE + M_LINE

    # Need to use deepcopy here to avoid unexpected change on parameter dict
    fitting_parameters = copy.deepcopy(params)

    total_list, matv, element_area = construct_linear_model(x, params, elemental_lines)

    # get background
    bg = snip_method(
        y,
        fitting_parameters["e_offset"]["value"],
        fitting_parameters["e_linear"]["value"],
        fitting_parameters["e_quadratic"]["value"],
        width=fitting_parameters["non_fitting_values"]["background_width"],
    )
    y = y - bg

    out, res = nnls_fit(y, matv, weights=weights)

    total_y = out * matv
    total_y = np.transpose(total_y)

    x_energy = (
        params["e_offset"]["value"] + params["e_linear"]["value"] * x + params["e_quadratic"]["value"] * x**2
    )

    area_dict = OrderedDict()
    result_dict = OrderedDict()

    for name, y, _out in zip(total_list, total_y, out):
        if np.sum(y) == 0:
            continue
        result_dict[name] = y
        area_dict[name] = _out * element_area[name]

    result_dict["background"] = bg
    area_dict["background"] = np.sum(bg)
    return x_energy, result_dict, area_dict


def get_activated_lines(incident_energy, elemental_lines):
    """
    Parameters
    ----------
    incident_energy : float
        beam energy
    element_lines : list
        e.g., ['Na_K', 'Mg_K', 'Pt_M']

    Returns
    -------
    list
        all activated lines for given elements
    """
    lines = []
    for v in elemental_lines:
        activated_line = _get_activated_line(incident_energy, v)
        if activated_line:
            lines.extend(activated_line)
    return lines


def _get_activated_line(incident_energy, elemental_line):
    """
    Collect all the activated lines for given element.

    Parameters
    ----------
    incident_energy : float
        beam energy
    elemental_line : str
        elemental line name

    Returns
    -------
    list
        all possible line names for given element
    """
    line_list = []
    if elemental_line in K_LINE:
        element = elemental_line.split("_")[0]
        e = Element(element)
        if e.cs(incident_energy)["ka1"] == 0:
            return
        for num, item in enumerate(e.emission_line.all[:4]):
            line_name = item[0]
            if e.cs(incident_energy)[line_name] == 0:
                continue
            line_list.append(str(element) + "_" + str(line_name))
        return line_list

    elif elemental_line in L_LINE:
        element = elemental_line.split("_")[0]
        e = Element(element)
        if e.cs(incident_energy)["la1"] == 0:
            return
        for num, item in enumerate(e.emission_line.all[4:-4]):
            line_name = item[0]
            if e.cs(incident_energy)[line_name] == 0:
                continue
            line_list.append(str(element) + "_" + str(line_name))
        return line_list

    elif elemental_line in M_LINE:
        element = elemental_line.split("_")[0]
        e = Element(element)
        if e.cs(incident_energy)["ma1"] == 0:
            return
        for num, item in enumerate(e.emission_line.all[-4:]):
            line_name = item[0]
            if e.cs(incident_energy)[line_name] == 0:
                continue
            line_list.append(str(element) + "_" + str(line_name))
        return line_list


def fit_per_line_nnls(data, matv, param, use_snip):
    """Fit experiment data for a given row using nnls algorithm.

    Parameters
    ----------
    data : array
        selected one row of experiment spectrum
    matv : array
        matrix for regression analysis
    param : dict
        fitting parameters
    use_snip : bool
        use snip algorithm to remove background or not

    Returns
    -------
    array :
        fitting values for all the elements at a given row. Background is
        calculated as a summed value. Also residual is included.
    """
    out = []
    bg_sum = 0
    for y in data:
        if use_snip:
            bg = snip_method(
                y,
                param["e_offset"]["value"],
                param["e_linear"]["value"],
                param["e_quadratic"]["value"],
                width=param["non_fitting_values"]["background_width"],
            )
            bg_sum = np.sum(bg)
            result, res = nnls_fit(y - bg, matv, weights=None)
        else:
            result, res = nnls_fit(y - bg, matv, weights=None)

        sst = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - res / sst
        result = list(result) + [bg_sum, r2]
        out.append(result)
    return np.array(out)


def _log_and_fit(row_num, *args):
    """Wrapper around fit_per_line_nnls to log the row num that is being used

    Parameters
    ----------
    row_num : int
        The row number that is being computed
    args
        The arguments to `fit_per_line_nnls`
    """
    logger.info("Row number at {}".format(row_num))
    return fit_per_line_nnls(*args)


def fit_pixel_multiprocess_nnls(exp_data, matv, param, use_snip=False):
    """
    Multiprocess fit of experiment data.

    Parameters
    ----------
    exp_data : array
        3D data of experiment spectrum,
        with x,y positions as the first 2-dim, and energy as the third one.
    matv : array
        matrix for regression analysis
    param : dict
        fitting parameters
    use_snip : bool, optional
        use snip algorithm to remove background or not

    Returns
    -------
    array
        Fitting values for all the elements
    """
    num_processors_to_use = multiprocessing.cpu_count()

    logger.info("cpu count: {}".format(num_processors_to_use))
    pool = multiprocessing.Pool(num_processors_to_use)

    result_pool = [
        pool.apply_async(_log_and_fit, (n, data, matv, param, use_snip)) for n, data in enumerate(exp_data)
    ]

    results = [r.get() for r in result_pool]

    pool.terminate()
    pool.join()

    return np.array(results)


def calculate_area(e_select, matv, results, param, first_peak_area=False):
    """
    Parameters
    ----------
    e_select : list
        elements
    matv : 2D array
        matrix constains elemental profile as columns
    results : 3D array
        x, y positions, and each element's weight on third dim
    param : dict
        parameters of fitting
    first_peak_area : Bool, optional
        get overal peak area or only the first peak area, such as Ar_Ka1

    Returns
    -------
    dict :
        dict of each 2D elemental distribution
    """
    total_list = e_select + ["snip_bkg"] + ["r_squared"]
    mat_sum = np.sum(matv, axis=0)

    result_map = dict()
    for i in range(len(e_select)):
        if first_peak_area is not True:
            result_map.update({total_list[i]: results[:, :, i] * mat_sum[i]})
        else:
            if total_list[i] not in K_LINE + L_LINE + M_LINE:
                ratio_v = 1
            else:
                ratio_v = get_relative_cs_ratio(total_list[i], param["coherent_sct_energy"]["value"])
            result_map.update({total_list[i]: results[:, :, i] * mat_sum[i] * ratio_v})

    # add background and res
    result_map.update({total_list[-2]: results[:, :, -2]})
    result_map.update({total_list[-1]: results[:, :, -1]})

    return result_map


def get_relative_cs_ratio(elemental_line, energy):
    """
    At given energy, multiple elemental lines may be activated. This function is
    used to calculate the ratio of the first line's cross section (cs) to
    the summed cross section from all the lines.

    Parameters
    ----------
    elemental_line : str
        e.g., 'Mg_K', refers to the K lines of Magnesium
    energy : float
        incident energy in keV

    Returns
    -------
    float :
        calculated ratio
    """
    name, line = elemental_line.split("_")
    e = Element(name)
    transition_lines = TRANSITIONS_LOOKUP[line.upper()]

    sum_v = 0
    for v in transition_lines:
        sum_v += e.cs(energy)[v]
    return e.cs(energy)[transition_lines[0]] / sum_v
