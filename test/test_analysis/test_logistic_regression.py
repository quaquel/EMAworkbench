'''
Created on 16 Mar 2019

@author: jhkwakkel
'''
from __future__ import (absolute_import, print_function, division)
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ema_workbench.analysis import logistic_regression 
from test import utilities


def flu_classify(data):
    #get the output for deceased population
    result = data['deceased population region 1'][:, -1]
    return result > 1000000

class LogitTestCase(unittest.TestCase):
    pass