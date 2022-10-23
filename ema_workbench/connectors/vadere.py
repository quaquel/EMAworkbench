import os
import pandas as pd
from functools import reduce
import operator
from ast import literal_eval
import json
from subprocess import PIPE, run
import shutil
from ema_workbench.em_framework.model import Replicator, SingleReplication
from ..em_framework.model import FileModel
from ..util.ema_logging import method_logger
from ..util import EMAError

__all__ = [
    "change_vadere_scenario",
    "update_vadere_scenario",
    "VadereModel",
    "SingleReplicationVadereModel",
]


def change_vadere_scenario(model_file, variable, value):
    """
    Change variable in vadere .scenario file structure. Note that a vadere scenario takes the format of a nested directory.
    This function enables to modify any variable in the .scenario file, given the exact level of nesting.

    Parameters
    ----------
    model_file : dict
                loaded Vadere .scenario file, use json.load to load the file as dict
    variable : tuple
                the level of the nested variable that needs to be updates. This should be
                provided as tuple. So for example, dict['x']['y'][5]['z'] should be provided as
                ('x', 'y', 5, 'z').
    value : float
            new value for the variable.

    """
    index = literal_eval(variable)
    reduce(operator.getitem, index[:-1], model_file)[index[-1]] = value


def update_vadere_scenario(model_file, experiment, output_file):
    """
    Load a vadere .scenario file, change it depending on the passed experiment, and save it again as .scenario file.

    Parameters
    ----------
    model_file : str
                path to the vadere .scenario file
    experiment : dict
                EMA experiment object
    output_file : str
                desired path to save the modified vadere .scenario file

    """
    with open(model_file) as file:
        v_model = json.load(file)

    for key, value in experiment.items():
        change_vadere_scenario(v_model, key, value)

    with open(output_file, "w") as file:
        json.dump(v_model, file)


class BaseVadereModel(FileModel):
    """Base class for interfacing with Vadere models.

    Attributes
    ----------
    model_file : str
                a relative path from the working directory
                to the model scenario file
    working_directory : str
    name : str

    """

    def __init__(self, name, vadere_jar, processor_files, wd, model_file):
        """
        init of class

        Parameters
        ----------
        wd   : str
                working directory for the model.
        name : str
                name of the modelInterface. The name should contain only
                alpha-numerical characters.
        vadere_jar : str
                    a relative path from the working directory
                    to the Vadere console jar file
        processor_files : list
                    list of output file names stored by Vadere, depending
                    on set processors. A .csv file is assumed for timeseries output,
                    and a .txt for a scaler output.

        Raises
        ------
        EMAError if name contains non alpha-numerical characters

        Note
        ----
        Anything that is relative to `self.working_directory`should be
        specified in `model_init` and not in `src`. Otherwise, the code
        will not work when running it in parallel. The reason for this is that
        the working directory is being updated by parallelEMA to the worker's
        separate working directory prior to calling `model_init`.

        """
        super().__init__(name, wd=wd, model_file=model_file)

        self.vadere_jar = vadere_jar
        self.processor_files = processor_files

    @method_logger(__name__)
    def model_init(self, policy):
        """
        Method called to initialize the model.

        Parameters
        ----------
        policy : dict
                 policy to be run.


        """
        super().model_init(policy)

    @method_logger(__name__)
    def run_experiment(self, experiment):
        """
        Method for running an instantiated model structure.

        Parameters
        ----------
        experiment : dict like

        Raises
        ------
        EMAError if the Vadere run returns no results

        """
        # change the .vadere scenario model file depending on the passed
        # experiment, and save to new "EMA.scenario" file
        update_vadere_scenario(
            os.path.join(self.working_directory, self.model_file),
            experiment,
            os.path.join(self.working_directory, "EMA.scenario"),
        )

        # make the temp dir for output, if one already exists (due to interrupted runs)
        # remove and create a new, empty, one
        try:
            shutil.rmtree(os.path.join(self.working_directory, "temp"))
        except OSError:
            pass
        try:
            os.mkdir(os.path.join(self.working_directory, "temp"))
        except OSError:
            pass

        # set up the run command for Vadere
        self.vadere = [
            "java",
            "-jar",
            os.path.join(self.working_directory, self.vadere_jar),
            "--loglevel",
            "OFF",
            "scenario-run",
            "-o",
            os.path.abspath(os.path.join(self.working_directory, "temp")),
            "-f",
            os.path.join(self.working_directory, "EMA.scenario"),
        ]

        # run the experiment
        process = run(self.vadere, stdin=PIPE, capture_output=True)

        # results are stored inside a temp dir
        # get path to nested result dir
        output_dir = ""
        for root, dirs, files in os.walk(os.path.join(self.working_directory, "temp")):
            # should only be one subdir
            # if for any reason multiple subdirs exist, only one will be
            # selected
            for subdir in dirs:
                output_dir = os.path.join(root, subdir)
        if not output_dir:
            raise EMAError(
                "Vadere model run resulted in no output files. Please check model. \n Vadere run error error: {}".format(
                    process.stderr
                )
            )
        # load results
        # .csv is assumed to be timeseries, .txt scaler
        # other file types are ignored
        timeseries_res = {}
        scalar_res = []
        for file in self.processor_files:
            if file.endswith(".csv"):
                timeseries_res[file] = pd.read_csv(os.path.join(output_dir, file), sep=" ")
            if file.endswith(".txt"):
                scalar_res.append(os.path.join(output_dir, file))

        # format data to EMA structure
        res = {}
        # handle timeseries
        if timeseries_res:
            if len(timeseries_res) > 1:
                timeseries_total = pd.concat(
                    [timeseries_res[outcome] for outcome in timeseries_res]
                )
            else:
                timeseries_total = timeseries_res[next(iter(timeseries_res))]
            # format according to EMA preference
            res = {col: series.values for col, series in timeseries_total.items()}

        # handle scalar
        if scalar_res:
            for file in scalar_res:
                s = pd.read_csv(file, sep=" ")
                for column, data in s.items():
                    res[column] = data.item()

        # remove temporal experiment output
        try:
            shutil.rmtree(os.path.join(self.working_directory, "temp"))
        except OSError:
            pass
        return res

    def cleanup(self):
        """
        This model is called after finishing all the experiments, but
        just prior to returning the results. This method gives a hook for
        doing any cleanup, such as closing applications.

        In case of running in parallel, this method is called during
        the cleanup of the pool, just prior to removing the temporary
        directories.

        """
        # cleanup moved to run_experiment, so temp dir is removed before new
        # experiment starts
        pass


class VadereModel(Replicator, BaseVadereModel):
    pass


class SingleReplicationVadereModel(SingleReplication, BaseVadereModel):
    pass
