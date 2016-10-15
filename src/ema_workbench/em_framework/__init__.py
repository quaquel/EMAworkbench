from __future__ import (absolute_import, unicode_literals, division, 
                        print_function)

__all__ = ["ema_parallel", "model_ensemble", "parameters"
           "model", "outcomes", "samplers", "uncertainties", 
           'RealUncertainty', "IntegerUncertainty", "CategoricalUncertainty",  
           "Model", 'FileModel', "ModelEnsemble",
           "Outcome", "ScalarOutcome", "TimeSeriesOutcome",
           "RealParameter", "IntegerParameter", "CategoricalParameter",
           "Scenario", "Policy", "Experiment", "Constant", "create_parameters",
           "parameters_to_csv", "Category"
           ]

from .outcomes import ScalarOutcome, TimeSeriesOutcome, Outcome
from .uncertainties import (ParameterUncertainty, CategoricalUncertainty)
from .model import Model, FileModel
from .ensemble import (ModelEnsemble)
from .parameters import (RealParameter, IntegerParameter, CategoricalParameter,
                     Scenario, Policy, Constant, Experiment, create_parameters,
                     parameters_to_csv, Category)
from .samplers import (MonteCarloSampler, FullFactorialSampler, LHSSampler, 
                       PartialFactorialSampler, LHS, MC, FF, PFF, SAMPLERS)



def perform_experiments(models, cases, policies=[Policy('None')], 
                        sampling=LHS, parallel=False, reporting_interval=None,
                        uncertainty_union=False, outcome_union=False):
    '''convenience function for running experiments
    
    Parameters
    ----------
    models : list or model instance
    cases : int or recarray
    policies : list, optional
    sampling : {LHS, MC, FF, PFF}
    parallel : bool, optional
    reporting_interval : int, optional
                         defaults to 1/10 of the number of cases
    uncertainty_union : bool, optional
    outcome_union : bool, optional
    
    Returns
    -------
    tuple
    
    
    '''
    
    import numbers
    if reporting_interval is None:
        if not (isinstance(cases, numbers.Integral)):
            n = cases.shape[0]
        else:
            n = cases
        
        reporting_interval = max(1, int(round(n / 10))) 

    ensemble = ModelEnsemble(sampler=SAMPLERS[sampling])
    ensemble.parallel = parallel
    ensemble.model_structures = models
    ensemble.policies = policies
    
    return ensemble.perform_experiments(cases, 
                                        reporting_interval=reporting_interval, 
                                        uncertainty_union=uncertainty_union, 
                                        outcome_union=outcome_union)
    
