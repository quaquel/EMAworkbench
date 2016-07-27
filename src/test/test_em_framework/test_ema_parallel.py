'''
Created on Jul 28, 2015

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
'''
from __future__ import (division, unicode_literals, print_function, 
                        absolute_import)

import mock
import unittest

import ipyparallel

from ema_workbench.em_framework.model import Model
from ema_workbench.em_framework.ema_parallel import IpyparallelPool

class TestIPyparallelPool(unittest.TestCase):

    def test(self):
        
        model = Model('test', mock.Mock())
        client = mock.Mock(spec=ipyparallel.Client)
        
#         IpyparallelPool([model], client)


if __name__ == "__main__":
    unittest.main()