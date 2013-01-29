'''
Created on 18 jan. 2013

@author: localadmin
'''
import numpy as np

from sandbox.workbench_core_revision.model_ensemble import ModelEnsemble, UNION, INTERSECTION,\
                                                   experiment_generator 
from sandbox.workbench_core_revision.samplers import LHSSampler

from expWorkbench import ModelStructureInterface, EMAlogging

from sandbox.workbench_core_revision.uncertainties import ParameterUncertainty, CategoricalUncertainty

from expWorkbench.outcomes import Outcome



class Dummy_interface(ModelStructureInterface):
    
    def model_init(self, policy, kwargs):
        pass
    
    def run_model(self, case):
        for outcome in self.outcomes:
            self.output[outcome.name] = np.random.rand(10,)
         

def test_generate_samples():
    # everything shared
    model_a = Dummy_interface(None, "A")
    model_b = Dummy_interface(None, "B")
    model_c = Dummy_interface(None, "C")
    
    # let's add some uncertainties to this
    shared_abc_1 = ParameterUncertainty((0,1), "shared abc 1")
    shared_abc_2 = ParameterUncertainty((0,1), "shared abc 2")
    shared_ab_1 = ParameterUncertainty((0,1), "shared ab 1")
    shared_bc_1 = ParameterUncertainty((0,1), "shared bc 1")
    a_1 = ParameterUncertainty((0,1), "a 1")
    b_1 = ParameterUncertainty((0,1), "b 1")
    model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
    model_b.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, shared_bc_1, b_1]
    model_c.uncertainties = [shared_abc_1, shared_abc_2, shared_bc_1]
    
    print '------------ UNION ------------'
    ensemble = ModelEnsemble()
    ensemble.add_model_structures([model_a, model_b, model_c])
    sampled_unc = ensemble._generate_samples(10, UNION )
    for entry in sampled_unc.keys(): 
        print entry 
    
    
    print '------------ INTERSECTION ------------'
    sampled_unc = ensemble._generate_samples(10, INTERSECTION )
     
    for entry in sampled_unc.keys(): 
        print entry 


def test_determine_intersecting_uncertainties():
    
#    # let's make some interfaces
#    model_a = Dummy_interface(None, "A")
#    model_b = Dummy_interface(None, "B")
#    
#    # let's add some uncertainties to this
#    shared_ab_1 = ParameterUncertainty((0,1), "shared ab 1")
#    shared_ab_2 = ParameterUncertainty((0,10), "shared ab 1")
#    model_a.uncertainties = [shared_ab_1, shared_ab_2]
#    model_b.uncertainties = [shared_ab_1, shared_ab_2]
#    
#    ensemble = ModelEnsemble()
#    ensemble.add_model_structures([model_a, model_b])
    
    # what are all the test cases?
    # test for error in case uncertainty by same name but different 
    # in other respects

    
    # everything shared
    model_a = Dummy_interface(None, "A")
    model_b = Dummy_interface(None, "B")
    model_c = Dummy_interface(None, "C")
    
    # let's add some uncertainties to this
    shared_abc_1 = ParameterUncertainty((0,1), "shared abc 1")
    shared_abc_2 = ParameterUncertainty((0,1), "shared abc 2")
    shared_ab_1 = ParameterUncertainty((0,1), "shared ab 1")
    shared_bc_1 = ParameterUncertainty((0,1), "shared bc 1")
    a_1 = ParameterUncertainty((0,1), "a 1")
    b_1 = ParameterUncertainty((0,1), "b 1")
    model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
    model_b.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, shared_bc_1, b_1]
    model_c.uncertainties = [shared_abc_1, shared_abc_2, shared_bc_1]
    
    ensemble = ModelEnsemble()
    ensemble.add_model_structures([model_a, model_b, model_c])
    overview, unique_unc = ensemble.determine_uncertainties()
    for key, value in overview.iteritems():
        print [msi for msi in key], [un.name for un in value]
    
    for key, value in unique_unc.iteritems():
        print key, value
    
    '''
    het zou nog simpeler kunnen. als ik gewoon de dict heb met uncertainty
    name en de onzekerheid en ik weet dat er geen fouten zitten qua naam,
    dan kan ik voor elke een lhs genereren en dan in de run experiments
    gewoon uit deze dict de lhs's pakken die horen bij de onzekerheid in de
    msi.
    
    '''
    
    # some shared between all, some between a and b, some between b and c
    # some between a and c
    
    # some shared, some unique
    
    # nothing shared 
    
    ensemble = ModelEnsemble()
    

def test_perform_experiments():
#    # let's make some interfaces
#    model_a = Dummy_interface(None, "A")
#    model_b = Dummy_interface(None, "B")
#    
#    # let's add some uncertainties to this
#    shared_ab_1 = ParameterUncertainty((0,1), "shared ab 1")
#    shared_ab_2 = ParameterUncertainty((0,10), "shared ab 1")
#    model_a.uncertainties = [shared_ab_1, shared_ab_2]
#    model_b.uncertainties = [shared_ab_1, shared_ab_2]
#    
#    ensemble = ModelEnsemble()
#    ensemble.add_model_structures([model_a, model_b])
    
    # what are all the test cases?
    # test for error in case uncertainty by same name but different 
    # in other respects

    
    # everything shared
    model_a = Dummy_interface(None, "A")
    model_b = Dummy_interface(None, "B")
    model_c = Dummy_interface(None, "C")
    
    # let's add some uncertainties to this
    shared_abc_1 = ParameterUncertainty((0,1), "shared abc 1")
    shared_abc_2 = ParameterUncertainty((0,1), "shared abc 2")
    shared_ab_1 = ParameterUncertainty((0,1), "shared ab 1")
    shared_bc_1 = ParameterUncertainty((0,1), "shared bc 1")
    a_1 = ParameterUncertainty((0,1), "a 1")
    b_1 = ParameterUncertainty((0,1), "b 1")
    model_a.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, a_1]
    model_b.uncertainties = [shared_abc_1, shared_abc_2, shared_ab_1, shared_bc_1, b_1]
    model_c.uncertainties = [shared_abc_1, shared_abc_2, shared_bc_1]
    
    #let's add an outcome to this
    outcome_shared = Outcome("test", time=True)
    model_a.outcomes = [outcome_shared]
    model_b.outcomes = [outcome_shared]
    model_c.outcomes = [outcome_shared]
    
    ensemble = ModelEnsemble()
    ensemble.parallel=True
    ensemble.add_model_structures([model_a, model_b, model_c])
    
    EMAlogging.info('------------- union of uncertainties -------------')
    
    ensemble.perform_experiments(10, which_uncertainties=UNION, reporting_interval=1 )
    
    EMAlogging.info('------------- intersection of uncertainties -------------')
    ensemble.perform_experiments(10, which_uncertainties=INTERSECTION, reporting_interval=1)

def test_experiment_generator():
    sampler = LHSSampler()
    
    shared_abc_1 = ParameterUncertainty((0,1), "shared ab 1")
    shared_abc_2 = ParameterUncertainty((0,1), "shared ab 2")
    unique_a = ParameterUncertainty((0,1), "unique a ")
    unique_b = ParameterUncertainty((0,1), "unique b ")
    uncertainties = [shared_abc_1, shared_abc_2, unique_a, unique_b]
    sampled_unc = sampler.generate_samples(uncertainties, 10)
    
    # everything shared
    model_a = Dummy_interface(None, "A")
    model_b = Dummy_interface(None, "B")
    
    model_a.uncertainties = [shared_abc_1, shared_abc_2, unique_a]
    model_b.uncertainties = [shared_abc_1, shared_abc_2, unique_b]
    model_structures = [model_a, model_b]
    
    policies = [{'name':'policy 1'},
                {'name':'policy 2'},
                {'name':'policy 3'},]
    
    gen = experiment_generator(sampled_unc, model_structures, policies, sampler)
    
    experiments = []
    for entry in gen:
        print entry
        experiments.append(gen)
    print len(experiments)

if __name__ == "__main__":
    EMAlogging.log_to_stderr(EMAlogging.INFO)
#    test_determine_intersecting_uncertainties()
#    test_generate_samples()
    test_perform_experiments()
#    test_experiment_generator()