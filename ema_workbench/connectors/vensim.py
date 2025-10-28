"""Convenience functions and classes to be used in combination with Vensim.

Most importantly, it specifies a generic ModelStructureInterface class
for controlling vensim models. In addition, this module contains frequently
used functions with error checking. For more fine grained control, the
:mod:`vensim_dll_wrapper` can also be used directly.

"""

import os

import numpy as np

from ..em_framework import FileModel, TimeSeriesOutcome
from ..em_framework.model import SingleReplication
from ..em_framework.points import Experiment, Sample
from ..em_framework.util import Variable
from ..util import EMAWarning, ExperimentError, get_module_logger
from ..util.ema_logging import method_logger
from . import vensim_dll_wrapper
from .vensim_dll_wrapper import VensimError, VensimWarning, command, get_val

# Created on 25 mei 2011
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = [
    "VensimModel",
    "be_quiet",
    "get_data",
    "load_model",
    "read_cin_file",
    "run_simulation",
    "set_value",
]
_logger = get_module_logger(__name__)


def be_quiet():
    """Turn off the work in progress dialog of Vensim.

    Defaults to 2, suppressing all windows, for more fine-grained control, use
    :mod:`vensim_dll_wrapper` directly .
    """
    vensim_dll_wrapper.be_quiet(2)


def load_model(file_name):
    """Load the model.

    Parameters
    ----------
    file_name : str
                file name of model, relative to working directory

    Raises
    ------
    VensimError if the model cannot be loaded.

    .. note: only works for .vpm files

    """
    _logger.debug("executing COMMAND: SIMULATE>SPECIAL>LOADMODEL|" + file_name)
    try:
        command("SPECIAL>LOADMODEL|" + str(file_name))
    except VensimWarning as w:
        _logger.warning(str(w))
        raise VensimError("vensim file not found") from w


def read_cin_file(file_name):
    """Read a .cin file.

    Parameters
    ----------
    file_name : str
                file name of cin file, relative to working directory

    Raises
    ------
    VensimWarning if the cin file cannot be read.

    """
    _logger.debug("executing COMMAND: SIMULATE>READCIN|" + file_name)
    try:
        command(r"SIMULATE>READCIN|" + str(file_name))
    except VensimWarning as w:
        _logger.warning(str(w))
        raise w


def set_value(variable, value):
    """Set the value of a variable to value.

    current implementation only works for lookups and normal values. In case
    of a list, a lookup is assumed, else a normal value is assumed.
    See the DSS reference supplement, p. 58 for details.

    Parameters
    ----------
    variable : str
               name of the variable to set.
    value : int, float, or list
            the value for the variable. **note**: the value can be either a
            list, or an float/integer. If it is a list, it is assumed the
            variable is a lookup.
    """
    variable = str(variable)

    if isinstance(value, list):
        value = [str(entry) for entry in value]
        command("SIMULATE>SETVAL|" + variable + "(" + str(value)[1:-1] + ")")
    else:
        try:
            command("SIMULATE>SETVAL|" + variable + "=" + str(value))
        except VensimWarning:
            _logger.warning("variable: '" + variable + "' not found")


def run_simulation(file_name):
    """Rn a model and store the results of the run in the specified .vdf file.

    The specified output file will be overwritten by default

    Parameters
    ----------
    file_name : str
                the file name of the output file relative to the working
                directory

    Raises
    ------
    VensimError if running the model failed in some way.

    """
    file_name = str(file_name)

    try:
        _logger.debug(" executing COMMAND: SIMULATE>RUNNAME|" + file_name + "|O")
        command("SIMULATE>RUNNAME|" + file_name + "|O")
        _logger.debug("MENU>RUN|o")
        command("MENU>RUN|o")
    except VensimWarning as w:
        _logger.warning(str(w))
        raise VensimError(str(w)) from w


def get_data(filename, varname, step=1):
    """Retrieve data from simulation runs or imported data sets.

    Parameters
    ----------
    filename : str
               the name of the .vdf file that contains the data
    varname : str
              the name of the variable to retrieve data on
    step : int (optional)
           steps used in slicing. Defaults to 1, meaning the full recorded time
           series is returned.

    Returns
    -------
    numpy array with the values for varname over the simulation

    """
    vval = []
    try:
        vval, _ = vensim_dll_wrapper.get_data(filename, str(varname))
    except VensimWarning as w:
        _logger.warning(str(w))

    return vval


class VensimModel(SingleReplication, FileModel):
    """Base class for controlling Vensim models.

    This class will handle starting Vensim, loading a model, setting parameters
    on the model, running the model, and retrieving the results. T

    .. note:: This class relies on the Vensim DLL, thus a complete installation
              of Vensim DSS is needed.

    """

    @property
    def result_file(self):
        """Return path to results file."""
        return os.path.join(self.working_directory, self._result_file)

    def __init__(
        self,
        name: str,
        wd: str | None = None,
        model_file: str | None = None,
        replace_underscores: bool = True,
    ):
        """Interface to the model.

        Parameters
        ----------
        name : name of the model, should be a valid python identifier
        wd : working directory for the model.
        model_file  : The name of the vensim file to be loaded
        replace_underscores : whether to replace underscores in the name of uncertainties, levers, constants,
                              and outcomes with spaces

        Raises
        ------
        EMAError
            if name is not a valid python idenfier
        ValueError
            if model_file cannot be found or is not a vpm file

        .. note:: Anything that is relative to `self.working_directory`
          should be specified in `model_init` and not
          in `src`. Otherwise, the code will not work when running
          it in parallel. The reason for this is that the working
          directory is being updated by parallelEMA to the worker's
          separate working directory prior to calling `model_init`.

        """
        if vensim_dll_wrapper.vensim_64 is not None:
            if not model_file.endswith(".vpmx") and not model_file.endswith(".vpm"):
                raise ValueError("model file should be a .vpm or .vpmx file")
            self._result_file = "Current.vdfx"
        else:
            if not model_file.endswith(".vpm"):
                raise ValueError("model file should be a vpm file")
            self._result_file = "Current.vdf"

        super().__init__(name, wd=wd, model_file=model_file)
        self.outcomes.extend(TimeSeriesOutcome("TIME", variable_name="Time"))

        #: attribute that can be set when one wants to load a cin file
        self.cin_file = None

        self.run_length = None

        if replace_underscores:
            self._first_call = True

        # default name of the results file (default: 'Current.vdfx'
        # for 64 bit, and Current.vdf for 32 bit)

        _logger.debug("vensim interface init completed")

    def model_init(self, policy: Sample):
        """Init of the model.

        Parameters
        ----------
        policy : policy to be run.
        """
        super().model_init(policy)

        if self._first_call:
            self._first_call = False

            def handle_underscores(variables: list[Variable]):
                for variable in variables:
                    if (len(variable.variable_name) == 1) and (
                        variable.variable_name[0] == variable.name
                    ):
                        variable.variable_name = variable.name.replace("_", " ")

            handle_underscores(self.uncertainties)
            handle_underscores(self.outcomes)
            handle_underscores(self.constants)
            handle_underscores(self.levers)

        fn = os.path.join(self.working_directory, self.model_file)
        load_model(fn)  # load the model

        _logger.debug("model initialized successfully")

        be_quiet()  # minimize the screens that are shown

        try:
            initial_time = get_val("INITIAL TIME")
            final_time = get_val("FINAL TIME")
            time_step = get_val("TIME STEP")
            save_per = get_val("SAVEPER")

            if save_per > 0:
                time_step = save_per

            self.run_length = int((final_time - initial_time) / time_step + 1)
        except VensimWarning as w:
            raise EMAWarning(str(VensimWarning)) from w

    @method_logger(__name__)
    def run_experiment(self, experiment: Experiment):
        """Run the experiment.

        The provided implementation assumes that the keys (i.e., the parameter names) in the
        experiment match the variable names in the Vensim model.

        if you want to use cin_files, set the cin_file, or cin_files in
        the extension of this method to `self.cin_file`.

        Parameters
        ----------
        experiment : the experiment to run

        .. note:: setting parameters should always be done via run_model.
                  The model is reset to its initial values automatically after
                  each run.

        """
        if self.cin_file:
            cin_file = os.path.join(self.working_directory, self.cin_file)

            try:
                read_cin_file(cin_file)
            except VensimWarning as w:
                _logger.debug(str(w))
            else:
                _logger.debug("cin file read successfully")

        for key, value in experiment.items():
            set_value(key, value)
        _logger.debug("model parameters set successfully")

        _logger.debug("run simulation, results stored in " + self.result_file)
        try:
            run_simulation(self.result_file)
        except VensimError:
            raise

        # TODO:: move to separate function/method?
        def check_data(result):
            error = False
            if result.shape[0] != self.run_length:
                data = np.empty(self.run_length)
                data[:] = np.NAN
                data[0 : result.shape[0]] = result
                result = data
                error = True
            return result, error

        results = {}
        error = False
        result_filename = os.path.join(self.working_directory, self.result_file)
        for variable in self.output_variables:
            _logger.debug(f"getting data for {variable}")

            res = get_data(result_filename, variable)
            result, er = check_data(np.asarray(res))
            error = error or er

            results[variable] = result

        _logger.debug("setting results to output")
        if error:
            raise ExperimentError("run not completed", experiment)

        return results


def create_model_for_debugging(path_to_existing_model, path_to_new_model, error):
    """Create a vensim mdl file parameterized according to the experiment.

    To be able to debug the Vensim model, a few steps are needed:

    1.  The experiment that gave a bug, needs to be saved in a text  file. The entire
        experiment description should be on a single line.
    2.  Reform and clean your model ( In the Vensim menu: Model, Reform and
        Clean). Choose

         * Equation Order: Alphabetical by group (not really necessary)
         * Equation Format: Terse

    3.  Save your model as text (File, Save as..., Save as Type: Text Format
        Models
    4.  Call this function.
    5.  If the print in the end is not set([]), but set([array]), the array
        gives the values that where not found and changed
    5.  Run your new model (for example 'new text.mdl')
    6.  Vensim tells you about your critical mistake


    Parameters
    ----------
    path_to_existing_model : str
                             path to the original mdl file
    path_to_new_model : str
                        path for the new mdl file
    error : str
            the case error, only containing the parameterization

    """
    # we assume the case specification was copied from the logger
    experiment = error.split(",")
    variables = {}

    # -1 because policy entry needs to be removed
    for entry in experiment[0:-1]:
        variable, value = entry.split(":")

        # Delete the spaces and other rubish on the sides of the variable name
        variable = variable.strip()
        variable = variable.lstrip("'")
        variable = variable.rstrip("'")
        value = value.strip()

        # vensim model is in bytes, so we go from unicode to bytes
        variables[variable.encode("utf8")] = value.encode("utf8")

    # This generates a new (text-formatted) model
    with open(path_to_new_model, "wb") as new_model:
        skip_next_line = False

        for line in open(path_to_existing_model, "rb"):  # noqa: SIM115
            if skip_next_line:
                skip_next_line = False
                lin_to_write = b"\n"
            elif line.find(b"=") != -1:
                variable = line.split(b"=")[0]
                variable = variable.strip()

                try:
                    value = variables.pop(variable)
                except KeyError:
                    pass
                else:
                    lin_to_write = variable + b" = " + value
                    skip_next_line = True

            new_model.write(lin_to_write)

    _logger.info("parameters not set:")
    _logger.info(set(variables.keys()))
