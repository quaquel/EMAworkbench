'''
Created on Jul 17, 2014

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import os
import unittest


from ema_workbench.em_framework import (TimeSeriesOutcome, RealParameter, 
                                    CategoricalParameter, perform_experiments) 

from ema_workbench.connectors.vensim import (VensimModel, 
                                             load_model, LookupUncertainty)

from ema_workbench.util import ema_logging


__test__ = True

if os.name != 'nt':
    __test__ = False

class VensimExampleModel(VensimModel):
    '''
    example of the most simple case of doing EMA on
    a Vensim model.
    
    '''
    #note that this reference to the model should be relative
    #this relative path will be combined with the workingDirectory
    model_file = r'\model.vpm'

    #specify outcomes
    outcomes = [TimeSeriesOutcome('a')]

    #specify your uncertainties
    uncertainties = [RealParameter("x11", 0, 2.5),
                     RealParameter("x12", -2.5, 2.5)]

class LookupTestModel(VensimModel): 
    def __init__(self, working_directory, name):
        
        self.model_file = r'\lookup_model.vpm'
        super(LookupTestModel, self).__init__(working_directory, name)

        # vensim.load_model(self.modelFile)
        self.outcomes = [TimeSeriesOutcome('flow1')]

 
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
    
    def test_read_cin_file(self):
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
        
        nr_runs = 10
        experiments, outcomes = perform_experiments(model, nr_runs)
        
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
        msi = VensimModel('', 'test')
 
        lookup_type = 'categories'
        name = 'test'
        values = [[(0.0, 0.05), (0.25, 0.15), (0.5, 0.4), (0.75, 1), (1, 1.25)], 
                  [(0.0, 0.1), (0.25, 0.25), (0.5, 0.75), (1, 1.25)],
                  [(0.0, 0.0), (0.1, 0.2), (0.3, 0.6), (0.6, 0.9), (1, 1.25)]]
        LookupUncertainty(lookup_type, values, name, msi)
         
        self.assertEqual(len(msi.uncertainties), 1)
        self.assertTrue(isinstance(msi.uncertainties[0], 
                                   CategoricalParameter))


        # hearne1
        msi = VensimModel('', 'test')
        msi.uncertainties = []
 
        lookup_type = 'hearne1'
        name = 'test'
        values = [(0,1),(0,1),(0,1),(0,1)]
        LookupUncertainty(lookup_type, values, name, msi)
         
        self.assertEqual(len(msi.uncertainties), 4)
        for unc in msi.uncertainties:
            self.assertTrue(isinstance(unc, 
                                   RealParameter))


        # hearne2
        msi = VensimModel('', 'test')
        msi.uncertainties = []
        
        lookup_type = 'hearne2'
        name = 'test'
        values = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
        LookupUncertainty(lookup_type, values, name, msi)
         
        self.assertEqual(len(msi.uncertainties), 6)
        for unc in msi.uncertainties:
            self.assertTrue(isinstance(unc, 
                                   RealParameter))


        # approximation
        msi = VensimModel('', 'test')
        msi.uncertainties = []
        
        lookup_type = 'approximation'
        name = 'test'
        values = [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]
        LookupUncertainty(lookup_type, values, name, msi)
         
        self.assertEqual(len(msi.uncertainties), 5)
        for unc in msi.uncertainties:
            self.assertTrue(isinstance(unc, RealParameter))


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
        perform_experiments(model, 10)

if __name__ == '__main__':
    if os.name == 'nt':
        ema_logging.log_to_stderr(ema_logging.INFO)
        unittest.main()
