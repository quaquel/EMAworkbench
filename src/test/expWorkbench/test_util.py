'''
Created on 22 nov. 2012

@author: localadmin
'''
from expWorkbench import util, ema_logging
import numpy as np
import os

def test_save_results():
    ema_logging.log_to_stderr(ema_logging.DEBUG)
    data  = util.load_results('./data/1000 flu cases no policy.cPickle', zip=False)
    file_name = "test.bz2"
    util.save_results(data, file_name)
    os.remove(file_name)
    ema_logging.debug("removing "+file_name)
    

def test_load_results():

    data  = np.random.rand(1000,1000)
    file_name = "test.bz2"
    util.save_results(data, file_name)
    
    ema_logging.log_to_stderr(ema_logging.DEBUG)
    util.load_results(file_name)
    os.remove(file_name)
    ema_logging.debug("removing "+file_name)


#test_load_results()
test_save_results()