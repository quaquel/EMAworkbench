'''
Created on Mar 22, 2014

@author: jhkwakkel
'''
from __future__ import (absolute_import, print_function, division)

import os

from ema_workbench.util import load_results


def load_flu_data():
    path = os.path.dirname(__file__)
    fn = './data/1000 flu cases no policy.tar.gz'
    fn = os.path.join(path, fn)

    experiments, outcomes = load_results(fn)
    return experiments, outcomes


def load_scarcity_data():
    path = os.path.dirname(__file__)
    fn = './data/1000 runs scarcity.tar.gz'
    fn = os.path.join(path, fn)

    experiments, outcomes = load_results(fn)
    return experiments, outcomes


def load_eng_trans_data():
    path = os.path.dirname(__file__)
    fn = './data/eng_trans.tar.gz'
    fn = os.path.join(path, fn)

    experiments, outcomes = load_results(fn)
    return experiments, outcomes
  
if __name__ == '__main__':
    load_flu_data()
    load_scarcity_data()
    load_eng_trans_data()
    