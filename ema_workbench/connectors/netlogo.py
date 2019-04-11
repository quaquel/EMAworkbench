'''
This module specifies a generic Model class for controlling
NetLogo models.

ScalarOutcomes and ArrayOutcomes are assumed to be reporters on the
NetLogo model that return a scalar and list respectively. These will be
queried after having run the model for the specified ticks

TimeSeries outcomes write data to a .txt file for each tick. This can be used
in combination with an agent-set or reporter. However, it introduces
substantial overhead.


'''
from ema_workbench.em_framework.model import Replicator, SingleReplication
from ema_workbench.util.ema_logging import get_module_logger
from ema_workbench.em_framework.outcomes import TimeSeriesOutcome
from pyNetLogo.core import NetLogoException

try:
    import jpype
except ImportError:
    jpype = None
import os

import numpy as np

import pyNetLogo

from ..em_framework.model import FileModel
from ..util.ema_logging import method_logger

# Created on 15 mrt. 2013
#
# .. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['NetLogoModel']
_logger = get_module_logger(__name__)


class BaseNetLogoModel(FileModel):
    '''Base class for interfacing with netlogo models. This class
    extends :class:`em_framework.ModelStructureInterface`.

    Attributes
    ----------
    model_file : str
                 a relative path from the working directory to the model
    run_length : int
                 number of ticks
    command_format : str
                     default format for set operations in logo
    working_directory : str
    name : str

    '''

    @property
    def ts_output_variables(self):
        if self._ts_output_variables is None:
            timeseries = [o for o in self.outcomes if
                          isinstance(o, TimeSeriesOutcome)]

            self._ts_output_variables = [var for o in timeseries for var in
                                         o.variable_name]

        return self._ts_output_variables

    command_format = "set {0} {1}"

    def __init__(self, name, wd=None, model_file=None, netlogo_home=None,
                 netlogo_version=None, jvm_home=None, gui=False,
                 jvmargs=[]):
        """
        init of class

        Parameters
        ----------
        wd   : str
               working directory for the model.
        name : str
               name of the modelInterface. The name should contain only
               alpha-numerical characters.
        netlogo_home : str, optional
               Path to the NetLogo installation directory (required on Linux)
        netlogo_version : {'6','5'}, optional
               Used to choose command syntax for link methods (required on Linux)
        jvm_home : str, optional
               Java home directory for Jpype
        gui : bool, optional
               If true, displays the NetLogo GUI (not supported on Mac)
        jvmargs : list, optional

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
        super(BaseNetLogoModel, self).__init__(name, wd=wd,
                                               model_file=model_file)

        self.run_length = None
        self.netlogo_home = netlogo_home
        self.netlogo_version = netlogo_version
        self.jvm_home = jvm_home
        self.gui = gui
        self._ts_output_variables = None
        self.jvmargs = jvmargs

    @method_logger(__name__)
    def model_init(self, policy):
        '''
        Method called to initialize the model.

        Parameters
        ----------
        policy : dict
                 policy to be run.


        '''
        super(BaseNetLogoModel, self).model_init(policy)
        if not hasattr(self, 'netlogo'):
            _logger.debug("trying to start NetLogo")
            self.netlogo = pyNetLogo.NetLogoLink(
                netlogo_home=self.netlogo_home,
                netlogo_version=self.netlogo_version,
                jvm_home=self.jvm_home,
                gui=self.gui, jvmargs=self.jvmargs)
            _logger.debug("netlogo started")
        path = os.path.join(self.working_directory, self.model_file)
        self.netlogo.load_model(path)
        _logger.debug("model opened")

    @method_logger(__name__)
    def run_experiment(self, experiment):
        """
        Method for running an instantiated model structure.

        Parameters
        ----------
        experiment : dict like


        Raises
        ------
        jpype.JavaException if there is any exception thrown by the netlogo
        model


        """
        # TODO:: point for speedup. Send all set commands in 1 go
        for key, value in experiment.items():
            try:
                self.netlogo.command(self.command_format.format(key, value))
            except jpype.JavaException as e:
                _logger.warning(
                    'variable {} throws exception: {}'.format(
                        key, str(e)))

        _logger.debug("model parameters set successfully")

        # finish setup and invoke run
        self.netlogo.command("setup")

        # TODO:: it is possible to take advantage of of fact
        # that not all outcomes are time series
        # In that case, we need not embed the get command in the go
        # routine, but can do them at the end
        commands = []
        fns = {}
        for variable in self.ts_output_variables:
            fn = r'{0}{3}{1}{2}'.format(self.working_directory,
                                        variable,
                                        ".txt",
                                        os.sep)
            fns[variable] = fn
            fn = '"{}"'.format(fn)
            fn = fn.replace(os.sep, '/')

            if self.netlogo.report('is-agentset? {}'.format(variable)):
                # if name is name of an agentset, we
                # assume that we should count the total number of agents
                nc = r'file-open {0} file-write count {1}'.format(fn,
                                                                  variable,
                                                                  )
            else:
                # it is not an agentset, so assume that it is
                # a reporter / global variable
                nc = r'file-open {0} file-write {1}'.format(fn,
                                                            variable)
            commands.append(nc)

        c_start = "repeat {} [".format(self.run_length)
        c_close = "go ]"
        c_middle = " ".join(commands)
        command = " ".join((c_start, c_middle, c_close))
        _logger.debug(command)
        self.netlogo.command(command)

        # after the last go, we have not done a write for the outcomes
        # so we do that now
        self.netlogo.command(c_middle)

        # we also need to save the non time series outcomes
        self.netlogo.command("file-close-all")

        results = self._handle_outcomes(fns)

        # handle non time series outcomes
        non_ts_vars = set(self.output_variables) - \
            set(self.ts_output_variables)
        for variable in set(non_ts_vars):
            try:
                data = self.netlogo.report(variable)
            except NetLogoException:
                _logger.exception("{} not a reporter".format(variable))
            else:
                results[variable] = data

        return results

    def retrieve_output(self):
        """
        Method for retrieving output after a model run.

        Returns
        -------
        dict with the results of a model run.

        """
        return self.output

    def cleanup(self):
        '''
        This model is called after finishing all the experiments, but
        just prior to returning the results. This method gives a hook for
        doing any cleanup, such as closing applications.

        In case of running in parallel, this method is called during
        the cleanup of the pool, just prior to removing the temporary
        directories.

        '''
        try:
            self.netlogo.kill_workspace()
        except AttributeError:
            pass

        jpype.shutdownJVM()

    def _handle_outcomes(self, fns):
        '''helper function for parsing outcomes'''

        results = {}
        for key, value in fns.items():
            with open(value) as fh:
                result = fh.readline()
                result = result.strip()
                result = result.split()
                result = [float(entry) for entry in result]
                results[key] = np.asarray(result)
            os.remove(value)
        return results


class NetLogoModel(Replicator, BaseNetLogoModel):
    pass


class SingleReplicationNetLogoModel(SingleReplication, BaseNetLogoModel):
    pass
