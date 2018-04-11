'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                                        division)

import unittest

from ema_workbench.em_framework import util

class TestNamedObject(unittest.TestCase):
    pass

class TestNamedObjectMap(unittest.TestCase):
    pass

class TestNamedDict(unittest.TestCase):
    
    def test_namedict(self):
        name = 'test'
        kwargs = {'a':1, 'b':2}

        nd = util.NamedDict(name, **kwargs)
        
        self.assertEqual(nd.name, name, 'name not equal')
        
        for key, value in nd.items():
            self.assertEqual(kwargs[key], value, 'kwargs not set on inner dict correctly')
        
        kwargs = {'a':1, 'b':2}

        nd = util.NamedDict(**kwargs)
        
        self.assertEqual(nd.name, repr(kwargs), 'name not equal')
        for key, value in nd.items():
            self.assertEqual(kwargs[key], value, 'kwargs not set on inner dict correctly')
        
        # test len
        self.assertEqual(2, len(nd), 'length not correct')

        # test in        
        for entry in kwargs.keys():
            self.assertIn(entry, nd, '{} not in NamedDict'.format(entry))
        
        # test addition
        nd['c'] = 3
        self.assertIn('c', nd, 'additional item not added')
        self.assertEqual(3, nd['c'])
        
        # test removal
        del nd['c']
        self.assertNotIn('c', nd, 'item not removed succesfully')
        
    
class TestCombine(unittest.TestCase):
    
    def test_combine(self):
        pass
    
if __name__ == '__main__':
    unittest.main()