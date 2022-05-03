"""connector for Simio, dependent on python for .net (pythonnet). The
connector assumes Simio is installed in C:/Program Files (x86)/Simio

"""
import os
import sys
import numpy as np

import clr  # @UnresolvedImport

from ema_workbench.em_framework import FileModel, SingleReplication
from ema_workbench.util import CaseError, EMAError
from ema_workbench.util.ema_logging import get_module_logger, method_logger

# TODO:: do some auto discovery here analogue to netlogo?
simio_path = "C:/Program Files (x86)/Simio"
if os.path.exists(simio_path):
    sys.path.append(simio_path)
else:
    raise EMAError("Simio not found")

clr.AddReference("SimioDLL")
clr.AddReference("SimioAPI")
import SimioAPI  # @UnresolvedImport

# Created on 27 June 2019
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
__all__ = ["SimioModel"]
_logger = get_module_logger(__name__)


class SimioModel(FileModel, SingleReplication):
    """Connector for Simio models

     Parameters
    ----------
    name : str
           name of the modelInterface. The name should contain only
           alpha-numerical characters.
    working_directory : str
                        working_directory for the model.
    model_file  : str
                 the name of the model file
    main_model : str
    n_replications : int, optional

    Attributes
    ----------
    name : str
    wd : str
    model_file : str
    main_model_name : str
    output : dict
    n_replications : int

    Notes
    -----
    responses are stored for each replication. It is up to the user
    to specify on the Python side what descriptive statistics need to be saved
    given the numpy array with replication specific responses

    """

    @method_logger(__name__)
    def __init__(
        self, name, wd=None, model_file=None, main_model=None, n_replications=10
    ):
        """interface to the model

        Parameters
        ----------
        name : str
               name of the modelInterface. The name should contain only
               alpha-numerical characters.
        wd : str
             working_directory for the model.
        model_file  : str
                     the name of the model file
        main_model : str
        n_replications : int, optional

        Raises
        ------
        EMAError
            if name contains non alpha-numerical characters
        ValueError
            if model_file cannot be found

        """
        super().__init__(name, wd=wd, model_file=model_file)
        assert main_model != None
        self.main_model_name = main_model
        self.output = {}
        self.n_replications = n_replications
        self.scenarios = None
        self.control_map = None
        self.experiment = None
        self.project = None
        self.model = None
        self.case = None

    @method_logger(__name__)
    def model_init(self, policy):
        super().model_init(policy)
        _logger.debug("initializing model")

        # get project
        path_to_file = os.path.join(self.working_directory, self.model_file)
        self.project = SimioAPI.ISimioProject(
            SimioAPI.SimioProjectFactory.LoadProject(path_to_file)
        )
        self.policy = policy

        # get model
        models = SimioAPI.IModels(self.project.get_Models())
        model = models.get_Item(self.main_model_name)

        if not model:
            raise EMAError(
                f"""main model with name {self.main_model_name} '
                            'not found"""
            )

        self.model = SimioAPI.IModel(model)

        # set up new EMA specific experiment on model
        _logger.debug("setting up EMA experiment")
        self.experiment = SimioAPI.IExperiment(
            model.Experiments.Create("ema experiment")
        )
        SimioAPI.IExperimentResponses(self.experiment.Responses).Clear()

        # use all available responses as template for experiment responses
        responses = get_responses(model)

        for outcome in self.outcomes:
            for name in outcome.variable_name:
                name = outcome.name
                try:
                    value = responses[name]
                except KeyError:
                    raise EMAError(f"response with name '{name}' not found")

                response = SimioAPI.IExperimentResponse(
                    self.experiment.Responses.Create(name)
                )
                response.set_Expression(value.Expression)
                response.set_Objective(value.Objective)

        # remove any scenarios on experiment
        self.scenarios = SimioAPI.IScenarios(self.experiment.Scenarios)
        self.scenarios.Clear()

        # make control map
        controls = SimioAPI.IExperimentControls(self.experiment.get_Controls())
        self.control_map = {}

        for i in range(controls.Count):
            control = controls.get_Item(i)

            self.control_map[control.Name] = control

        _logger.debug("model initialized successfully")

    @method_logger(__name__)
    def run_experiment(self, experiment):
        self.case = experiment
        _logger.debug("Setup SIMIO scenario")

        scenario = self.scenarios.Create()
        scenario.ReplicationsRequired = self.n_replications
        _logger.debug(f"nr. of scenarios is {self.scenarios.Count}")

        for key, value in experiment.items():
            try:
                control = self.control_map[key]
            except KeyError:
                raise EMAError(
                    """uncertainty not specified as '
                                  'control in simio model"""
                )
            else:
                ret = scenario.SetControlValue(control, str(value))

                if ret:
                    _logger.debug(f"{key} set successfully")
                else:
                    raise CaseError(f"failed to set {key}", self.case)

        _logger.debug("SIMIO scenario setup completed")

        self.experiment.ScenarioEnded += self.scenario_ended
        self.experiment.RunCompleted += self.run_completed

        _logger.debug("preparing to run model")
        self.experiment.Run()
        _logger.debug("run completed")
        return self.output

    @method_logger(__name__)
    def reset_model(self):
        """
        Method for resetting the model to its initial state. The default
        implementation only sets the outputs to an empty dict.

        """
        super().reset_model()

        self.scenarios.Clear()
        self.output = {}

    @method_logger(__name__)
    def scenario_ended(self, sender, scenario_ended_event):
        """scenario ended event handler"""

        #         ema_logging.debug('scenario ended called!')

        # This event handler will be called when all replications for a
        # given scenario have completed.  At this point the statistics
        # produced by this scenario should be available.
        experiment = SimioAPI.IExperiment(sender)
        scenario = SimioAPI.IScenario(scenario_ended_event.Scenario)

        _logger.debug(
            f"""scenario {scenario.Name} for experiment '
                       '{experiment.Name} completed"""
        )
        responses = experiment.Scenarios.get_Responses()

        # http://stackoverflow.com/questions/16484167/python-net-framework-reference-argument-double

        # results = scenario_ended_event.get_Results()
        # data = []
        # for result in results:
        #     data.append(SimioAPI.IScenarioResult(result))
        # results = data

        for response in responses:
            _logger.debug(f"{response}")
            response_value = 0.0

            replication_scores = []
            for replication in range(1, self.n_replications + 1):
                try:
                    success, value = scenario.GetResponseValueForReplication(
                        response, replication, response_value
                    )
                except TypeError:
                    _logger.warning(
                        f"""type error when trying to get a '
                                             'response for {response.Name}"""
                    )
                    raise

                if success:
                    replication_scores.append(value)
                else:
                    error = CaseError("error in simio replication", self.case)
                    _logger.exception(str(error))
                    raise error

            self.output[response.Name] = np.asarray(replication_scores)

    @method_logger(__name__)
    def run_completed(self, sender, run_completed_event):
        """run completed event handler"""

        _logger.debug("run completed")

        # This event handler is the last one to be called during the run.
        # When running async, this is the correct place to shut things down.
        experiment = SimioAPI.IExperiment(sender)
        # Un-wire from the run events when we're done.
        experiment.ScenarioEnded -= self.scenario_ended
        experiment.RunCompleted -= self.run_completed


def get_responses(model):
    """Helper function for getting responses

    this function gathers all responses defined on all experiments available
    on the model.

    Parameters
    ----------
    model : SimioAPI.IModel instance

    """

    response_map = {}

    experiments = SimioAPI.IExperiments(model.Experiments)
    for i in range(experiments.Count):
        experiment = SimioAPI.IExperiment(experiments.get_Item(i))
        responses = SimioAPI.IExperimentResponses(experiment.Responses)
        for j in range(responses.Count):
            response = SimioAPI.IExperimentResponse(responses.get_Item(j))

            response_map[response.Name] = response

    return response_map
