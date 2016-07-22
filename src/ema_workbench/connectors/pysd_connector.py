'''
pysd connector
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)

import pysd
import os

from ..em_framework.model import FileModel
from ..em_framework.outcomes import TimeSeriesOutcome
from ema_workbench.em_framework.model import AbstractModel


class PySDConnector(AbstractModel):

    def __init__(self, name=None, mdl_file=None):
        """

        Parameters
        ----------
        mdl_file: string
            file name of vensim model (e.g. `Teacup.mdl`)
        uncertainties_dict: dictionary
            convenience parameter for constructing normal uncertainties, in the form
            {'Param 1': (lowerbound, upperbound), 'Param 2':(0,1)}
        outcomes_list: list of model variable names
            gets passed to 'return_columns', so can be model variable names
            or their pysafe translations
        working_directory
        name
        """
        if not mdl_file.endswith('.mdl'):
            raise ValueError('model file needs to be a vensim .mdl file')
        if name is None:
            name = pysd.utils.make_python_identifier(mdl_file)[0].replace('_','')
        
        super(PySDConnector, self).__init__(name)
#         self.outcomes = (TimeSeriesOutcome('TIME'))

        self.mdl_file = mdl_file
        self.model = pysd.read_vensim(os.path.abspath(self.mdl_file))
        # Todo: replace when pysd adds an attribute for the .py filename
        self.py_model_name = mdl_file.replace('.mdl', '.py')


    def run_model(self, kwargs):
        res = self.model.run(params=kwargs,
                             return_columns=[o.name for o in self.outcomes])
        # EMA wants output formatted properly
        self.output = {col: series.as_matrix() for col, series in res.iteritems()}

    def reset_model(self):
        """
        Method for reseting the model to its initial state. The default
        implementation only sets the outputs to an empty dict.

        """
        self.model = pysd.load(self.py_model_name)
