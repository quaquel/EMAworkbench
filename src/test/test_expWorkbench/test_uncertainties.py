'''
Created on 18 jan. 2013

@author: localadmin
'''
from expWorkbench import ParameterUncertainty, CategoricalUncertainty

def test_uncertainty_identity():
    # what are the test cases
    
    # uncertainties are the same
    # let's add some uncertainties to this
    shared_ab_1 = ParameterUncertainty((0,10), "shared ab 1")
    shared_ab_2 = ParameterUncertainty((0,10), "shared ab 1")

    print shared_ab_1 == shared_ab_2
    print shared_ab_2 == shared_ab_1
    
    # uncertainties are not the same
    shared_ab_1 = ParameterUncertainty((0,10), "shared ab 1")
    shared_ab_2 = ParameterUncertainty((0,10), "shared ab 1", integer=True)

    print shared_ab_1 == shared_ab_2
    print shared_ab_2 == shared_ab_1
    
    # uncertainties are of different classes
    # what should happen then?
    # in principle it should return false, but what if the classes are
    # different but the __dicts__ are the same? This would be lousy coding, 
    # but should it concern us here?
    shared_ab_1 = ParameterUncertainty((0,10), "shared ab 1")
    shared_ab_2 = CategoricalUncertainty([x for x in range(11)], "shared ab 1")
    
    print shared_ab_1 == shared_ab_2
    print shared_ab_2 == shared_ab_1

if __name__ == "__main__":
    test_uncertainty_identity()
    
