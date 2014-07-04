'''
Created on 22 nov. 2012

@author: localadmin
'''
from expWorkbench import util, ema_logging
import numpy as np
import os
from test.util import load_flu_data
from expWorkbench.util import save_results, load_results



def test_save_results():
    results = load_flu_data()
 
    save_results(results, r'test.tar.gz')
    os.remove('test.tar.gz')

    

def test_load_results():
    results = load_flu_data()
 
    save_results(results, r'test.tar.gz')
    load_results(r'test.tar.gz')
    os.remove('test.tar.gz')
    

if __name__ == '__main__':
    
   
#     test_save_results()
    test_load_results()