'''
pysd connector
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import os

import pysd

from ..em_framework.model import AbstractModel
from ema_workbench.util.ema_logging import method_logger
from ema_workbench.em_framework.model import SingleReplication

__all__ = ['PysdModel']


class BasePysdModel(AbstractModel):
    """ Pysd model wrapper

    Parameters
    ----------
    mdl_file: string
        file name of vensim model (e.g. `Teacup.mdl`)
    uncertainties_dict: dictionary
        convenience parameter for constructing normal uncertainties, in the
        form {'Param 1': (lowerbound, upperbound), 'Param 2':(0,1)}
    outcomes_list: list of model variable names
        gets passed to 'return_columns', so can be model variable names
        or their pysafe translations
    working_directory
    name
    """

    @property
    def mdl_file(self):
        return self._mdl_file

    @mdl_file.setter
    def mdl_file(self, mdl_file):
        if not mdl_file.endswith('.mdl'):
            raise ValueError('model file needs to be a vensim .mdl file')
        if not os.path.isfile(mdl_file):
            raise ValueError('mdl_file not found')

        # Todo: replace when pysd adds an attribute for the .py filename
        self.py_model_name = mdl_file.replace('.mdl', '.py')
        self._mdl_file = mdl_file

    def __init__(self, name=None, mdl_file=None):
        self.mdl_file = mdl_file
        if name is None:
            name = pysd.utils.make_python_identifier(mdl_file)[0]
            name = name.replace('_', '')
        super(BasePysdModel, self).__init__(name)

    @method_logger(__name__)
    def model_init(self, policy, **kwargs):
        AbstractModel.model_init(self, policy, **kwargs)

        # TODO:: should be updated only if mdl file has been changed
        # or if not initialized
        self.model = pysd.read_vensim(self.mdl_file)

    @method_logger(__name__)
    def run_experiment(self, experiment):
        res = self.model.run(params=experiment,
                             return_columns=self.output_variables)

        # EMA wants output formatted properly
        output = {col: series.values for col, series in res.iteritems()}
        return output

    def reset_model(self):
        """
        Method for reseting the model to its initial state. The default
        implementation only sets the outputs to an empty dict.

        """
        super(BasePysdModel, self).reset_model()
        if self.model is not None:
            self.model.initialize()


class PysdModel(SingleReplication, BasePysdModel):
    pass
