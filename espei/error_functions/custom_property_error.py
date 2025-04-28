"""
Calculate error due to equilibrium thermochemical properties.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import tinydb
from tinydb import where
from scipy.stats import norm
from pycalphad import Database, Model, ReferenceState, variables as v
from pycalphad.core.utils import instantiate_models, filter_phases, extract_parameters, unpack_species, unpack_condition
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad import Workspace, as_property

from espei.error_functions.residual_base import ResidualFunction, residual_function_registry
from espei.phase_models import PhaseModelSpecification
from espei.shadow_functions import update_phase_record_parameters
from espei.typing import SymbolName
from espei.utils import PickleableTinyDB, database_symbols_to_fit

_log = logging.getLogger(__name__)


PropData = NamedTuple('EqPropData', (('dbf', Database),
                                       ('species', Sequence[v.Species]),
                                       ('phases', Sequence[str]),
                                       ('potential_conds', Dict[v.StateVariable, float]),
                                       ('composition_conds', Sequence[Dict[v.X, float]]),
                                       ('models', Dict[str, Model]),
                                       ('params_keys', Dict[str, float]),
                                       ('phase_record_factory', PhaseRecordFactory),
                                       ('output', str),
                                       ('samples', np.ndarray),
                                       ('weight', np.ndarray),
                                       ('reference', str),
                                       ('property_kwargs', {})
                                       ))


def build_propdata(data: tinydb.database.Document,
                     dbf: Database,
                     model: Optional[Dict[str, Type[Model]]] = None,
                     parameters: Optional[Dict[str, float]] = None,
                     data_weight_dict: Optional[Dict[str, float]] = None,
                     property_std: float = 1,
                     ) -> PropData:
    """
    Build EqPropData for the calculations corresponding to a single dataset.

    Parameters
    ----------
    data : tinydb.database.Document
        Document corresponding to a single ESPEI dataset.
    dbf : Database
        Database that should be used to construct the `Model` and `PhaseRecord` objects.
    model : Optional[Dict[str, Type[Model]]]
        Dictionary phase names to pycalphad Model classes.
    parameters : Optional[Dict[str, float]]
        Mapping of parameter symbols to values.
    data_weight_dict : Optional[Dict[str, float]]
        Mapping of a data type (e.g. `HM` or `SM`) to a weight.

    Returns
    -------
    EqPropData
    """
    parameters = parameters if parameters is not None else {}
    data_weight_dict = data_weight_dict if data_weight_dict is not None else {}

    params_keys, _ = extract_parameters(parameters)

    data_comps = list(set(data['components']).union({'VA'}))
    species = sorted(unpack_species(dbf, data_comps), key=str)
    data_phases = filter_phases(dbf, species, candidate_phases=data['phases'])
    models = instantiate_models(dbf, species, data_phases, model=model, parameters=parameters)
    output = data['output']
    samples = np.array(data['values']).flatten()
    reference = data.get('reference', '')

    data['conditions'].setdefault('N', 1.0)  # Add default for N. Nothing else is supported in pycalphad anyway.
    pot_conds = OrderedDict([(getattr(v, key), unpack_condition(data['conditions'][key])) for key in sorted(data['conditions'].keys()) if not key.startswith('X_')])
    comp_conds = OrderedDict([(v.X(key[2:]), unpack_condition(data['conditions'][key])) for key in sorted(data['conditions'].keys()) if key.startswith('X_')])

    phase_record_factory = PhaseRecordFactory(dbf, species, {**pot_conds, **comp_conds}, models, parameters=parameters)

    # Now we need to unravel the composition conditions
    # (from Dict[v.X, Sequence[float]] to Sequence[Dict[v.X, float]]), since the
    # composition conditions are only broadcast against the potentials, not
    # each other. Each individual composition needs to be computed
    # independently, since broadcasting over composition cannot be turned off
    # in pycalphad.
    rav_comp_conds = [OrderedDict(zip(comp_conds.keys(), pt_comps)) for pt_comps in zip(*comp_conds.values())]

    # Build weights, should be the same size as the values
    total_num_calculations = len(rav_comp_conds)*np.prod([len(vals) for vals in pot_conds.values()])
    dataset_weights = np.array(data.get('weight', 1.0)) * np.ones(total_num_calculations)
    weights = (property_std/data_weight_dict.get(output, 1.0)/dataset_weights).flatten()

    property_kwargs = data.get('property_kwargs', {})

    return PropData(dbf, species, data_phases, pot_conds, 
                      rav_comp_conds, models, params_keys, phase_record_factory, 
                      output, samples, weights, reference, property_kwargs)


def get_custom_property_data(dbf: Database, comps: Sequence[str],
                                        phases: Sequence[str],
                                        datasets: PickleableTinyDB,
                                        model: Optional[Dict[str, Model]] = None,
                                        parameters: Optional[Dict[str, float]] = None,
                                        data_weight_dict: Optional[Dict[str, float]] = None,
                                        property_std: float = 1,
                                        ) -> Sequence[PropData]:
    """
    Get all the EqPropData for each matching equilibrium thermochemical dataset in the datasets

    Parameters
    ----------
    dbf : Database
        Database with parameters to fit
    comps : Sequence[str]
        List of pure element components used to find matching datasets.
    phases : Sequence[str]
        List of phases used to search for matching datasets.
    datasets : PickleableTinyDB
        Datasets that contain single phase data
    model : Optional[Dict[str, Type[Model]]]
        Dictionary phase names to pycalphad Model classes.
    parameters : Optional[Dict[str, float]]
        Mapping of parameter symbols to values.
    data_weight_dict : Optional[Dict[str, float]]
        Mapping of a data type (e.g. `HM` or `SM`) to a weight.

    Notes
    -----
    Found datasets will be subsets of the components and phases. Equilibrium
    thermochemical data is assumed to be any data that does not have the
    `solver` key, and does not have an output of `ZPF` or `ACR` (which
    correspond to different data types than can be calculated here.)

    Returns
    -------
    Sequence[EqPropData]
    """

    desired_data = datasets.search(
        # data that isn't ZPF or non-equilibrium thermochemical
        (where('output') != 'ZPF') & (~where('solver').exists()) &
        (where('output').test(lambda x: 'ACR' not in x)) &  # activity data not supported yet
        (where('output').test(lambda x: 'DIFF' not in x)) & (where('output').test(lambda x: 'TRACER' not in x)) & # ignore diffusivity
        (where('components').test(lambda x: set(x).issubset(comps))) &
        (where('phases').test(lambda x: set(x).issubset(set(phases))))
    )

    eq_thermochemical_data = []  # 1:1 correspondence with each dataset
    for data in desired_data:
        eq_thermochemical_data.append(build_propdata(data, dbf, model=model, parameters=parameters, data_weight_dict=data_weight_dict, property_std=property_std))
    return eq_thermochemical_data


def calc_prop_differences(propdata: PropData,
                          parameters: np.ndarray,
                          computable_property,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate differences between the expected and calculated values for a property

    Parameters
    ----------
    eqpropdata : EqPropData
        Data corresponding to equilibrium calculations for a single datasets.
    parameters : np.ndarray
        Array of parameters to fit. Must be sorted in the same symbol sorted
        order used to create the PhaseRecords.
    approximate_equilibrium : Optional[bool]
        Whether or not to use an approximate version of equilibrium that does
        not refine the solution and uses ``starting_point`` instead.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Pair of
        * differences between the calculated property and expected property
        * weights for this dataset

    """
    dbf = propdata.dbf
    species = propdata.species
    phases = propdata.phases
    pot_conds = propdata.potential_conds
    models = propdata.models
    phase_record_factory = propdata.phase_record_factory
    update_phase_record_parameters(phase_record_factory, parameters)
    params_dict = OrderedDict(zip(map(str, propdata.params_keys), parameters))
    output = as_property(propdata.output)
    weights = np.array(propdata.weight, dtype=np.float64)
    samples = np.array(propdata.samples, dtype=np.float64)
    wks = Workspace(database=dbf, components=species, phases=phases, models=models, phase_record_factory=phase_record_factory, parameters=params_dict)

    calculated_data = []
    for comp_conds in propdata.composition_conds:
        cond_dict = OrderedDict(**pot_conds, **comp_conds)
        wks.conditions = cond_dict
        wks.parameters = params_dict  # these reset models and phase_record_factory through depends_on -> lose Model.shift_reference_state, etc.
        wks.models = models
        wks.phase_record_factory = phase_record_factory
        prop = computable_property(wks, **propdata.property_kwargs)
        vals = wks.get(prop)
        calculated_data.extend(np.atleast_1d(vals).tolist())

    calculated_data = np.array(calculated_data, dtype=np.float64)

    assert calculated_data.shape == samples.shape, f"Calculated data shape {calculated_data.shape} does not match samples shape {samples.shape}"
    assert calculated_data.shape == weights.shape, f"Calculated data shape {calculated_data.shape} does not match weights shape {weights.shape}"
    differences = calculated_data - samples
    _log.debug('Output: %s differences: %s, weights: %s, reference: %s', output, differences, weights, propdata.reference)
    return differences, weights


def calculate_custom_property_probability(prop_data: Sequence[PropData],
                                          parameters: np.ndarray,
                                          computable_property,
                                          ) -> float:
    """
    Calculate the total equilibrium thermochemical probability for all EqPropData

    Parameters
    ----------
    eq_thermochemical_data : Sequence[EqPropData]
        List of equilibrium thermochemical data corresponding to the datasets.
    parameters : np.ndarray
        Values of parameters for this iteration to be updated in PhaseRecords.
    approximate_equilibrium : Optional[bool], optional

    eq_thermochemical_data : Sequence[EqPropData]

    Returns
    -------
    float
        Sum of log-probability for all thermochemical data.

    """
    if len(prop_data) == 0:
        return 0.0

    differences = []
    weights = []
    for propdata in prop_data:
        diffs, wts = calc_prop_differences(propdata, parameters, computable_property)
        if np.any(np.isinf(diffs) | np.isnan(diffs)):
            # NaN or infinity are assumed calculation failures. If we are
            # calculating log-probability, just bail out and return -infinity.
            return -np.inf
        differences.append(diffs)
        weights.append(wts)

    differences = np.concatenate(differences, axis=0)
    weights = np.concatenate(weights, axis=0)
    probs = norm(loc=0.0, scale=weights).logpdf(differences)
    return np.sum(probs)


class CustomPropertyResidual(ResidualFunction):
    prop = None
    name = 'None'
    property_std = 1

    def __init__(
        self,
        database: Database,
        datasets: PickleableTinyDB,
        phase_models: Union[PhaseModelSpecification, None],
        symbols_to_fit: Optional[List[SymbolName]] = None,
        weight: Optional[Dict[str, float]] = None,
        ):
        super().__init__(database, datasets, phase_models, symbols_to_fit, weight)

        if weight is not None:
            self.weight = weight
        else:
            self.weight = {}

        if phase_models is not None:
            comps = sorted(phase_models.components)
            model_dict = phase_models.get_model_dict()
        else:
            comps = sorted(database.elements)
            model_dict = dict()
        phases = sorted(filter_phases(database, unpack_species(database, comps), database.phases.keys()))
        if symbols_to_fit is None:
            symbols_to_fit = database_symbols_to_fit(database)
        # okay if parameters are initialized to zero, we only need the symbol names
        parameters = dict(zip(symbols_to_fit, [0]*len(symbols_to_fit)))
        self.property_data = get_custom_property_data(database, comps, phases, datasets, model_dict, parameters, data_weight_dict=self.weight, property_std=self.property_std)

    def get_residuals(self, parameters: npt.ArrayLike) -> Tuple[List[float], List[float]]:
        residuals = []
        weights = []
        for data in self.property_data:
            dataset_residuals, dataset_weights = calc_prop_differences(data, parameters, self.prop)
            residuals.extend(dataset_residuals.tolist())
            weights.extend(dataset_weights.tolist())
        return residuals, weights

    def get_likelihood(self, parameters) -> float:
        likelihood = calculate_custom_property_probability(self.property_data, parameters, self.prop)
        return likelihood
