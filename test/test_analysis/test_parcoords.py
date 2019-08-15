'''
Created on 15 Aug 2019

@author: jhkwakkel
'''
import matplotlib.pyplot as plt
import pandas as pd

import unittest

from ema_workbench.analysis.parcoords import (ParallelAxes, get_limits)

class TestParcoords(unittest.TestCase):
    
    def test_parallelaxis(self):
        x = pd.DataFrame([[0.1, 0, set(('a','b'))],
                          [1.0, 9, set(('a','b'))]], 
                          columns=['a', 'b', 'c'])
        
        axes = ParallelAxes(x)
        
        self.assertEqual(2, len(axes.axes))
        
    
    def test_invert_axis(self):
        x = pd.DataFrame([[0.1, 0, set(('a','b'))],
                          [1.0, 9, set(('a','b'))]], 
                          columns=['a', 'b', 'c'])
        
        axes = ParallelAxes(x)
        
        axes.invert_axis('a')        
        self.assertEqual(axes.flipped_axes, {'a'},)
        
        axes.invert_axis('a')
        self.assertEqual(axes.flipped_axes, set(),)


        axes.invert_axis('c')        
        self.assertEqual(axes.flipped_axes, {'c'},)
        
        axes.invert_axis('c')
        self.assertEqual(axes.flipped_axes, set(),)


        axes.invert_axis(['a', 'b'])
        self.assertEqual(axes.flipped_axes, {'a', 'b'},)        
    
    def test_plot(self):
        x = pd.DataFrame([[0.1, 0, 'a'],
                          [0.2, 1, 'b'],
                          [0.3, 2, 'a'],
                          [0.4, 3, 'b'],
                          [0.5, 4, 'a'],
                          [0.6, 5, 'a'],
                          [0.7, 6, 'b'],
                          [0.8, 7, 'a'],
                          [0.9, 8, 'b'],
                          [1.0, 9, 'a']], 
                          columns=['a', 'b', 'c'])
        
        limits = get_limits(x)
        
        axes = ParallelAxes(limits)
        
        for _, row in x.iterrows():
            axes.plot(row.to_frame().T)
        plt.draw()
        
        axes = ParallelAxes(limits)
        axes.invert_axis('a')
        for _, row in x.iterrows():
            axes.plot(row.to_frame().T)
        plt.draw()


        axes = ParallelAxes(limits)
        axes.invert_axis('c')
        for _, row in x.iterrows():
            axes.plot(row.to_frame().T)
        plt.draw()
        
        axes = ParallelAxes(limits)
        for _, row in x.iterrows():
            axes.plot(row.to_frame().T)
        axes.invert_axis('c')
        plt.draw()

        axes = ParallelAxes(limits)
        for i, row in x.iterrows():
            axes.plot(row.to_frame().T, label=str(i))
        axes.legend()
        plt.draw()
    
    def test_get_limits(self):
        # heterogenous without NAN
        x = pd.DataFrame([[0.1, 0, 'a'],
                          [0.2, 1, 'b'],
                          [0.3, 2, 'a'],
                          [0.4, 3, 'b'],
                          [0.5, 4, 'a'],
                          [0.6, 5, 'a'],
                          [0.7, 6, 'b'],
                          [0.8, 7, 'a'],
                          [0.9, 8, 'b'],
                          [1.0, 9, 'a']], 
                          columns=['a', 'b', 'c'])
        
        limits = get_limits(x)
        
        self.assertEqual(limits['a'][0], 0.1)
        self.assertEqual(limits['a'][1], 1.0)
        self.assertEqual(limits['b'][0], 0)
        self.assertEqual(limits['b'][1], 9)
        self.assertEqual(limits['c'][0], set(('a', 'b')))
        self.assertEqual(limits['c'][1], set(('a', 'b')))


        
        
if __name__ == '__main__':
    unittest.main()