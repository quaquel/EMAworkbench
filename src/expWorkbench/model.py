'''
Created on 23 dec. 2010

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import division
import types
import copy
import abc
import os

import samplers
import util

from EMAparallel import CalculatorPool
from expWorkbench.EMAlogging import info, warning, exception, debug
from expWorkbench.EMAexceptions import CaseError, EMAError, EMAParallelError

__all__ = ['ModelStructureInterface',
           'SimpleModelEnsemble']

#==============================================================================
# abstract Model class 
#==============================================================================
class ModelStructureInterface(object):
    '''
    :class:`ModelStructureInterface` is one of the the two main classes used 
    for performing EMA. This class should be extended to provide an interface 
    to the actual model. 
    
    '''
    
    __metaclass__ = abc.ABCMeta
    
    #: list of uncertainty instances
    uncertainties = []
    
    #: list of outcome instances
    outcomes = []
    
    #: name of the model interface
    name = None 
    
    #: results, this should be a dict with the names of the outcomes as key
    output = {}

    workingDirectory = None
    
    def __init__(self, workingDirectory, name):
        """
        interface to the model
        
        :param workingDirectory: workingDirectory for the model. 
        :param name: name of the modelInterface. The name should contain only
                     alphanumerical characters. 
        """
        self.name=None
        self.workingDirectory=None
        
        super(ModelStructureInterface, self).__init__()
        if workingDirectory:
            self.set_working_directory(workingDirectory)
    
        if not name.isalnum():
            raise EMAError("name of model should only contain alpha numerical\
                            characters")
        
        self.name = name
    
    @abc.abstractmethod
    def model_init(self, policy, kwargs):
        '''
        Method called to initialize the model.
        
        :param policy: policy to be run.
        :param kwargs: keyword arguments to be used by model_intit. This
                       gives users to the ability to pass any additional 
                       arguments. 
        
        .. note:: This method should always be implemented. Although in simple
                  cases, a simple pass can suffice. 
        '''
    
    @abc.abstractmethod
    def run_model(self, case):
        """
        Method for running an instantiated model structure. 
        
        This method should always be implemented.
        
        :param case: keyword arguments for running the model. The case is a 
                     dict with the names of the uncertainties as key, and
                     the values to which to set these uncertainties. 
        
        .. note:: This method should always be implemented.
        
        """

    def retrieve_output(self):
        """
        Method for retrieving output after a model run.
        
        :return: the results of a model run. 
        """
        return self.output
    
    def reset_model(self):
        """
        Method for reseting the model to its initial state. The default
        implementation only sets the outputs to an empty dict. 

        """
        self.output = {}
        
    
    def cleanup(self):
        '''
        This model is called after finishing all the experiments, but 
        just prior to returning the results. This method gives a hook for
        doing any cleanup, such as closing applications. 
        
        In case of running in parallel, this method is called during 
        the cleanup of the pool, just prior to removing the temporary 
        directories. 
        
        '''
        pass

    def get_model_uncertainties(self):
        """
        Method for retrieving model structure uncertainties.
        
        :return: list of the uncertainties of the model interface.
        """
        return self.uncertainties    
    
    def set_working_directory(self, wd):
        '''
        Method for setting the working directory of the model interface. This
        method is used in case of running models in parallel. In this case,
        each worker process will have its own working directory, to avoid 
        having to share files across processes. This requires the need to
        update the working directory to the new working directory. 
        
        :param wd: The new working directory.
        
        '''
        
        wd = os.path.abspath(wd)
        debug('setting working directory to '+ wd)
        
        self.workingDirectory = wd
        

#==============================================================================
# model ensemble class 
#==============================================================================
class SimpleModelEnsemble(object):
    '''
    One of the two main classes for performing EMA. The ensemble class is 
    responsible for running experiments on one or more model structures across
    one or more policies, and returning the results. 
    
    The sampling is delegated to a sampler instance.
    The storing or results is delegated to a callback instance
    
    the class has an attribute 'parallel' that specifies whether the 
    experiments are to be run in parallel or not. By default, 'parallel' is 
    False.
    
    .. rubric:: an illustration of use
    
    >>> model = UserSpecifiedModelInterface(r'working directory', 'name')
    >>> ensemble = SimpleModelEnsemble()
    >>> ensemble.set_model_structure(model)
    >>> ensemble.parallel = True #parallel processing is turned on
    >>> results = ensemble.perform_experiments(1000) #perform 1000 experiments
    
    In this example, a 1000 experiments will be carried out in parallel on 
    the user specified model interface. The uncertainties are retrieved from 
    model.uncertainties and the outcomes are assumed to be specified in
    model.outcomes.
    
    '''
    
    #: In case of parallel computing, the number of 
    #: processes to be spawned. Default is None, meaning
    #: that the number of processes will be equal to the
    #: number of available cores.
    processes=None
    
    #: boolean for turning parallel on (default is False)
    parallel = False
    
    def __init__(self, sampler=samplers.LHSSampler()):
        """
        Class responsible for running experiments on diverse model 
        structures and storing the results.

        :param sampler: the sampler to be used for generating experiments. 
                        By default, the sampling technique is 
                        :class:`~samplers.LHSSampler`.  
        """
        super(SimpleModelEnsemble, self).__init__()
        self.output = {}
        self._policies = []
        self._modelStructures = []
        self.sampler = sampler
        

    def add_policy(self, policy):
        """
        Add a policy. 
        
        :param policy: policy to be added, policy should be a dict with at  least 
                       a name.
        
        """
        self._policies.append(policy)
        
    def add_policies(self, policies):
        """
        Add policies, policies should be a collection of policies.
        
        :param policies: policies to be added, every policy should be a 
                         dict with at  least a name.
        
        """
        [self._policies.append(policy) for policy in policies]
  
    def set_model_structure(self, modelStructure):
        '''
        Set the model structure. This function wraps the model structure
        in a tuple, limiting the number of model structures to 1.
        
        :param modelStructure: a :class:`~model.ModelStructureInterface` instance.
        
        '''
        
        self._modelStructures = tuple([modelStructure])
                     
    def add_model_structure(self, ms):
        '''
        Add a model structure to the list of model structures.
        
        :param ms: a :class:`~model.ModelStructureInterface` instance.
        
        '''
        
        self._modelStructures.append(ms)   
    
    def add_model_structures(self, mss):
        '''
        add a collection of model structures to the list of model structures.
        
        :param mss: a collection of :class:`~model.ModelStructureInterface` 
                    instances
        
        '''
        
        [self._modelStructures.append(ms) for ms in mss]  
    
    def _generate_cases(self, nrOfCases):
        '''
        number of cases specifies the number of cases to generate in case
        of Monte Carlo and Latin Hypercube sampling.
        
        In case of full factorial sampling it specifies the resolution on
        non categorical uncertainties.
        
        In case of multiple model structures, the uncertainties over
        which to explore is the intersection of the sets of uncertainties of
        the model interface instances.
        
        :param nrOfCases: In case of Latin Hypercube sampling and Monte Carlo 
                          sampling, nrOfCases specifies the number of cases to
                          generate. In case of Full Factorial sampling,
                          nrOfCases specifies the resolution to use for sampling
                          continuous uncertainties.
        
        '''
        
        #get the intersection of the uncertainties of the different models
        if len(self._modelStructures)  >1:
            uncertainties = [msi.uncertainties for msi in self._modelStructures]
            uncertainties = set(uncertainties[0]).intersection(*uncertainties[:1])
            info("intersection contains %s uncertainties" %len(uncertainties))
        else:
            uncertainties = set(self._modelStructures[0].uncertainties)
         
        info("generating cases")
        
        designs = self.sampler.generate_design(uncertainties, nrOfCases)
        information = designs[1]
        designs = designs[0]
        cases = []
        for design in designs:
            case = {}
            for i, name in enumerate(information):
                case[name] = design[i]
            cases.append(case)
        
        info(str(len(cases)) + " cases generated")
        
        return cases, uncertainties
        
    def perform_experiments(self, 
                           cases,
                           callback = util.DefaultCallback,
                           kwargs = None):
        """
        Method responsible for running the experiments on a structure. In case 
        of multiple model structures, the outcomes are set to the intersection 
        of the sets of outcomes of the various models.         
        
        :param cases: In case of Latin Hypercube sampling and Monte Carlo 
                      sampling, cases specifies the number of cases to
                      generate. In case of Full Factorial sampling,
                      cases specifies the resolution to use for sampling
                      continuous uncertainties. Alternatively, one can supply
                      a list of dicts, where each dicts contains a case.
                      That is, an uncertainty name as key, and its value. 
        :param callback: Class that will be called after finishing a 
                         single experiment,
        :param kwargs: generic keyword arguments to pass to the model_init
        :returns: a `structured numpy array <http://docs.scipy.org/doc/numpy/user/basics.rec.html>`_ 
                  containing the experiments, and a dict with the names of the 
                  outcomes as keys and an numpy array as value.
                
        .. rubric:: suggested use
        
        In general, analysis scripts require both the structured array of the 
        experiments and the dictionary of arrays containing the results. The 
        recommended use is the following::
        
        >>> results = ensemble.perform_experiments(10000) #recommended use
        >>> experiments, output = ensemble.perform_experiments(10000) #will work fine
        
        The latter option will work fine, but most analysis scripts require 
        to wrap it up into a tuple again::
        
        >>> data = (experiments, output)
        
        Another reason for the recommended use is that you can save this tuple
        directly::
        
        >>> import expWorkbench.util as util
        >>> util.save_results(results, file)
          
        
        
        """
        if type(cases) ==  types.IntType:
            cases, uncertainties = self._generate_cases(cases)
        if type(cases) == types.ListType:
            
            #get the intersection of uncertainties
            if len(self._modelStructures)  >1:
                uncertainties = [msi.uncertainties for msi in self._modelStructures]
                uncertainties = set(uncertainties[0]).intersection(*uncertainties[:1])
                info("intersection contains %s uncertainties" %len(uncertainties))
            else:
                uncertainties = self._modelStructures[0].uncertainties
            
            #filter out those how ore in the cases keys
            uncertaintyNames = cases[0].keys()
            uncertainties = [uncertianty for uncertianty in uncertainties if uncertianty.name in uncertaintyNames]
        
        if not self._policies:
            self._policies.append({"name": "None"})

        nrOfExperiments =len(cases)*len(self._policies)*len(self._modelStructures) 
        info(str(nrOfExperiments) + 
             " experiment will be executed")

        
        #set outcomes to the intersect of outcomes across models
        outcomes = [msi.outcomes for msi in self._modelStructures]
        outcomes = set(outcomes[0]).intersection(*outcomes[:1])
        for msi in self._modelStructures:
            msi.outcomes = list(outcomes)
                
        #initialize the callback object
        callback = callback(uncertainties, outcomes, nrOfExperiments).callback
                
        if self.parallel:
            info("starting to perform experiments in parallel")
            pool = CalculatorPool(self._modelStructures, 
                                  processes=self.processes,
                                  callback=callback, 
                                  kwargs=kwargs)
            results = pool.runExperiments(cases, self._policies)
            
            for entry in results:
                try:
                    result = entry.get()
                except EMAParallelError as e:
                    exception(e)
                except Exception as e:
                    raise
            results = results[-1].get()
            del pool
        else:
            info("starting to perform experiments sequentially")

            def cleanup(modelInterfaces):
                for msi in modelInterfaces:
                    msi.cleanup()
                    del msi

            for policy in self._policies:
                for msi in self._modelStructures:
                    try:
                        msi.model_init(policy, kwargs)
                    except (EMAError, NotImplementedError) as inst:
                        exception(inst)
                        cleanup(self._modelStructures)
                        raise
    
                    for case in cases:
                        caseToRun = copy.deepcopy(case)
                        try:
                            msi.run_model(caseToRun)
                        except CaseError as e:
                            warning(str(e))
                        result = msi.retrieve_output()
                        msi.reset_model()
                        results = callback(
                                           caseToRun, policy, msi.name, 
                                           result
                                           )
            cleanup(self._modelStructures)
        info("experiments finished")
        
        return results