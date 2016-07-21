from ..em_framework import (ModelStructureInterface, Outcome,
                            ParameterUncertainty)
import pysd


class PySDInterface(ModelStructureInterface):

    def __init__(self, mdl_file, uncertainties_dict=None, outcomes_list=None,
                 working_directory=None, name=None):
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

        if name is None:
            name = pysd.utils.make_python_identifier(mdl_file)[0].replace('_','')

        if uncertainties_dict is not None:
            self.uncertainties = [ParameterUncertainty(val, key) for
                                  key, val in uncertainties_dict.iteritems()]
        else:
            self.uncertainties = []

        if outcomes_list is not None:
            self.outcomes = [Outcome(key, time=True) for
                             key in outcomes_list]
        else:
            self.outcomes = []

        self.outcomes.append(Outcome('TIME', time=True))

        self.model = pysd.read_vensim(mdl_file)
        # Todo: replace when pysd adds an attribute for the .py filename
        self.py_model_name = mdl_file.replace('.mdl', '.py')
        super(PySDInterface, self).__init__(working_directory, name)

    def model_init(self, policy, kwargs):
        # Todo: need to see what the arguments to this function should do
        pass

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
