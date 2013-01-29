'''
Created on 21 jan. 2013

@author: localadmin
'''

from uncertainties import ParameterUncertainty

a = ParameterUncertainty((0,1), "blaat")

print a
print a.__class__
print a.__class__.__name__