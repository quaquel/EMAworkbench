'''
Created on Jul 17, 2014

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''

import os
import unittest

from em_framework import Outcome, ParameterUncertainty, ModelEnsemble
from em_framework.uncertainties import CategoricalUncertainty 

from connectors.vensim import VensimModelStructureInterface, load_model
from connectors.vensim import LookupUncertainty

from util import ema_logging

__test__ = True

if os.name != 'nt':
    __test__ = False

class VensimExampleModel(VensimModelStructureInterface):
    '''
    example of the most simple case of doing EMA on
    a Vensim model.
    
    '''
    #note that this reference to the model should be relative
    #this relative path will be combined with the workingDirectory
    model_file = r'\model.vpm'

    #specify outcomes
    outcomes = [Outcome('a', time=True)]

    #specify your uncertainties
    uncertainties = [ParameterUncertainty((0, 2.5), "x11"),
                     ParameterUncertainty((-2.5, 2.5), "x12")]





class LookupTestModel(VensimModelStructureInterface): 
    def __init__(self, working_directory, name):
        
        self.model_file = r'\lookup_model.vpm'
        super(LookupTestModel, self).__init__(working_directory, name)

        # vensim.load_model(self.modelFile)
        self.outcomes = [Outcome('flow1', time=True)]

 
        '''
        each lookup uncertainty defined and added to the uncertainties list must be deleted immediately. it is not possible to do that in the constructor of lookups.
        or i can delete it later before generating the cases.
        '''
        self.uncertainties.append(LookupUncertainty('hearne2', [(0, 0.5), (-0.5, 0), (0, 0.75), (0.75, 1.5), (0.8, 1.2), (0.8, 1.2)], "TF", self, 0, 2))
        #self.uncertainties.pop()
        self.uncertainties.append(LookupUncertainty('approximation', [(0, 4), (1, 5), (1, 5), (0, 2), (0, 2)], "TF2", self, 0, 10))
        #self.uncertainties.pop()
        #self.uncertainties.append(ParameterUncertainty((0.02, 0.08), "rate1"))
        #self.uncertainties.append(ParameterUncertainty((0.02, 0.08), "rate2"))
        self.uncertainties.append(LookupUncertainty('categories', [[(0.0, 0.05), (0.25, 0.15), (0.5, 0.4), (0.75, 1), (1, 1.25)], 
                                                     [(0.0, 0.1), (0.25, 0.25), (0.5, 0.75), (1, 1.25)],
                                                     [(0.0, 0.0), (0.1, 0.2), (0.3, 0.6), (0.6, 0.9), (1, 1.25)]], "TF3", self, 0, 2))
        #self.uncertainties.pop()   
        self._delete_lookup_uncertainties()                  

class VensimTest(unittest.TestCase):
    
    def test_be_quiet(self):
        pass
    
    def test_load_model(self):
        pass
    
    def read_cin_file(self):
        pass
    
    def test_set_value(self):
        pass
    
    def test_run_simulation(self):
        
        model_file = r'../models/model.vpm'
        load_model(model_file)
    
    def test_get_data(self):
        pass
    
class VensimMSITest(unittest.TestCase):
    
    def test_vensim_model(self):
        
        #instantiate a model
        wd = r'../models'
        model = VensimExampleModel(wd, "simpleModel")
        
        #instantiate an ensemble
        ensemble = ModelEnsemble()
        
        #set the model on the ensemble
        ensemble.model_structure = model
        
        nr_runs = 10
        experiments, outcomes = ensemble.perform_experiments(nr_runs)
        
        self.assertEqual(experiments.shape[0], nr_runs)
        self.assertIn('TIME', outcomes.keys())
        self.assertIn(model.outcomes[0].name, outcomes.keys())
    

class LookupUncertaintyTest(unittest.TestCase):
    def test_added_uncertainties(self):
        '''
        the lookup uncertainty replaces itself with a set of other 
        uncertainties. Here we test whether this works correctly for
        each of the options provided by the lookup uncertainty
        
        
        '''
        if os.name != 'nt':
            return


        # categories
        msi = VensimModelStructureInterface('', 'test')
 
        lookup_type = 'categories'
        name = 'test'
        values = [[(0.0, 0.05), (0.25, 0.15), (0.5, 0.4), (0.75, 1), (1, 1.25)], 
                  [(0.0, 0.1), (0.25, 0.25), (0.5, 0.75), (1, 1.25)],
                  [(0.0, 0.0), (0.1, 0.2), (0.3, 0.6), (0.6, 0.9), (1, 1.25)]]
        LookupUncertainty(lookup_type, values, name, msi)
         
        self.assertEqual(len(msi.uncertainties), 1)
        self.assertTrue(isinstance(msi.uncertainties[0], 
                                   CategoricalUncertainty))


        # hearne1
        msi = VensimModelStructureInterface('', 'test')
        msi.uncertainties = []
 
        lookup_type = 'hearne1'
        name = 'test'
        values = [(0,1),(0,1),(0,1),(0,1)]
        LookupUncertainty(lookup_type, values, name, msi)
         
        self.assertEqual(len(msi.uncertainties), 4)
        for unc in msi.uncertainties:
            self.assertTrue(isinstance(unc, 
                                   ParameterUncertainty))


        # hearne2
        msi = VensimModelStructureInterface('', 'test')
        msi.uncertainties = []
        
        lookup_type = 'hearne2'
        name = 'test'
        values = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
        LookupUncertainty(lookup_type, values, name, msi)
         
        self.assertEqual(len(msi.uncertainties), 6)
        for unc in msi.uncertainties:
            self.assertTrue(isinstance(unc, 
                                   ParameterUncertainty))


        # approximation
        msi = VensimModelStructureInterface('', 'test')
        msi.uncertainties = []
        
        lookup_type = 'approximation'
        name = 'test'
        values = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
        LookupUncertainty(lookup_type, values, name, msi)
         
        self.assertEqual(len(msi.uncertainties), 5)
        for unc in msi.uncertainties:
            self.assertTrue(isinstance(unc, 
                                   ParameterUncertainty))


    def test_running_lookup_uncertainties(self):
        '''
        This is the more comprehensive test, given that the lookup
        uncertainty replaces itself with a bunch of other uncertainties, check
        whether we can successfully run a set of experiments and get results
        back. We assert that the uncertainties are correctly replaced by
        analyzing the experiments array. 
        
        '''
        if os.name != 'nt':
            return
        
        model = LookupTestModel( r'../models/', 'lookupTestModel')
        
        #model.step = 4 #reduce data to be stored
        ensemble = ModelEnsemble()
        ensemble.model_structure = model
        
        ensemble.perform_experiments(10)

if __name__ == '__main__':
    if os.name == 'nt':
        ema_logging.log_to_stderr(ema_logging.INFO)
        unittest.main()
