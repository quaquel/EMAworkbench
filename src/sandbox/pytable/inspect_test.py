'''
Created on Apr 26, 2012

@author: localadmin
'''
SVN_ID = '$Id: inspect_test.py 820 2012-05-03 06:08:16Z jhkwakkel $'

import inspect
import re

from expWorkbench import AbstractCallback


def format_svn_id(SVN_ID):
    svnidrep = r'^\$Id: (?P<filename>.+) (?P<revision>\d+) (?P<date>\d{4}-\d{2}-\d{1,2}) (?P<time>\d{2}:\d{2}:\d{2})Z (?P<user>\w+) \$$'
    mo = re.match(svnidrep, SVN_ID)
    svn_id = '%s - revision %s (%s)' % mo.group('filename', 'revision', 'date')
    return svn_id

def test_with_inspect(obj):
    stack = inspect.stack()[1][0]
    print stack.f_globals['SVN_ID']
    print stack.f_globals['__file__']
    print stack.f_globals['__doc__']
    
    a = inspect.getmodule(obj)
    print a.__doc__
    print a.__file__

    
    print "blaat"
    
class InspectCallback(AbstractCallback):
    '''
    Base class from which different call back classes can be derived.
    Callback is responsible for storing the results of the runs.
    
    '''
    
    i = 0
    reporting_interval = 100
    results = []
    
    def __init__(self, 
                 uncertainties, 
                 outcomes,
                 nrOfExperiments,
                 reporting_interval=100):
        '''
        
        :param uncertainties: list of :class:`~uncertianties.AbstractUncertainty` 
                              children
        :param outcomes: list of :class:`~outcomes.Outcome` instances
        :param nrOfExperiments: the total number of runs
        
        '''
        self.reporting_interval = reporting_interval
        
        stack = inspect.stack()
        first_entry = stack[1][0]
        ensemble = first_entry.f_locals['self']
        sampler = ensemble.sampler
        modelInterfaces = ensemble._modelStructures
        
        objs = list(modelInterfaces)
        objs.append(sampler)
        objs.append(ensemble)
        for obj in objs:
            mod = inspect.getmodule(obj)
            print format_svn_id(mod.SVN_ID)
            
    
