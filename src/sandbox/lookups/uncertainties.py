'''

Created on 16 aug. 2011

This module contains various classes that can be used for specifying different
types of uncertainties.

.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

'''
from __future__ import division
from math import exp, pow
import decimal


from sets import ImmutableSet
from expWorkbench.ema_exceptions import EMAError
from connectors import vensimDLLwrapper 

__all__ = ['AbstractUncertainty',
           'ParameterUncertainty',
           'CategoricalUncertainty'
           'LookupUncertainty'
           ]

INTEGER = 'integer'
UNIFORM = 'uniform'

#==============================================================================
# uncertainty classes
#==============================================================================
class AbstractUncertainty(object):
    '''
    :class:`AbstractUncertainty` provides a template for specifying different
    types of uncertainties.
    '''
    
    #: the values that specify the uncertainty
    values = None
    
    #: the type of integer
    type = None
    
    #: the name of the uncertainty
    name = None
    
#    #: the datatype of the uncertainty
#    dtype = None    
    
    #: a string denoting the type of distribution to be used in sampling
    dist = None
    
    def __init__(self, values, name):
        '''
        
        :param values: the values for specifying the uncertainty from which to 
                       sample
        :param name: name of the uncertainty
        
        '''
        
        super(AbstractUncertainty, self).__init__()
        self.values = values
        self.name = name
    
    def get_values(self):
        ''' get values'''
        return self.values

    def identity(self):
        '''
        helper method that returns the elements that define an uncertainty. 
        By default these are the name, the lower value of the range and the 
        upper value of the range.
         
        '''
        
        return (self.name, self.values[0], self.values[1])

class ParameterUncertainty(AbstractUncertainty ):
    """
    :class:`ParameterUncertainty` is used for specifying parametric 
    uncertainties. An uncertainty is parametric if the range is continuous from
    the lower bound to the upper bound.
    
    Parametric uncertainties are either floats or integers. 
        
    """
    
    #: optional attribute for specifying default value for uncertainty
    default = None
    
    def __init__(self, values, name, integer=False, default = None):
        '''
        
        :param values: the values for specifying the uncertainty from which to 
                       sample. Values should be a tuple with the lower and
                       upper bound for the uncertainty. These bounds are
                       inclusive. 
        :param name: name of the uncertainty
        :param integer: boolean, if True, the parametric uncertainty is 
                        an integer
        :param default: optional argument for providing a default value
        
        '''
       
        super(ParameterUncertainty, self).__init__(values, name)
        if default: 
            self.default = default
        else: 
            self.default = abs(self.values[0]-self.values[1])
        
        self.type = "parameter"
        
        if len(values) != 2:
            raise EMAError("length of values for %s incorrect " % name)

        # self.dist should be a string. This string should be a key     
        # in the distributions dictionary 
        if integer:
            self.dist = "integer"
            self.params = (values[0], values[1]+1)
            self.default = int(round(self.default))
        else:
            self.dist = "uniform"
            #params for initializing self.dist
            self.params = (self.get_values()[0], 
                           self.get_values()[1]-self.get_values()[0])
        
    def get_default_value(self):
        ''' return default value'''
        
        return self.default

class CategoricalUncertainty(ParameterUncertainty):
    """
    :class:`CategoricalUncertainty` can can be used for sampling over 
    categorical variables. The categories can be of any type, including 
    Strings, Integers, Floats, Tuples, or any Object. As values the categories 
    are specified in a collection. 
    
    Underneath, this is treated as a integer parametric uncertainty. That is,
    an integer parametric uncertainty is used with each integer corresponding
    to a particular category.  This class  called by the sampler to transform 
    the integer back to the appropriate category.
    """
    
    #: the categories of the uncertainty
    categories = None
    
    def __init__(self, values, name, default = None):
        '''
        
        :param values: the values for specifying the uncertainty from which to 
                       sample. Values should be a collection.
        :param name: name of the uncertainty
        :param default: optional argument for providing a default value
        
        '''
        self.categories = values
        values = (0, len(values)-1)
        if default != None:
            default = self.invert(default)
        
        self.default = default
        super(CategoricalUncertainty, self).__init__(values, 
                                                     name, 
                                                     integer=True,
                                                     default=default)
        self.integer = True
                
    def transform(self, param):
        '''transform an integer to a category '''
        return self.categories[param]
    
    def invert(self, name):
        ''' transform a category to an integer'''
        return self.categories.index(name)

    def identity(self):
        categories = ImmutableSet(self.categories)
        return (self.name, categories)
    
class LookupUncertainty(AbstractUncertainty):
    # TODO this class should be in vensim.py and not in generic
    
    HEARNE1 = 'hearne1'
    HEARNE2 = 'hearne2'
    APPROX =  'approximation'
    CAT = 'categories'

    error_message = "unknown transform_type for lookup uncertainty {}"
    msi = None
    y_min = None
    y_max = None
    x_min = None
    x_max = None 
    x = []
    y = []
     
    def __init__(self, values, name, transform_type, msi, ymin, ymax):
        
        
        '''
        
        :param values: the values for specifying the uncertainty from which to 
                       sample.
           If 'transform_type' is "categories", a set of alternative lookup 
               functions to  be entered as tuples of x,y points.
               Example definition: 
               LookupUncertainty([[(0.0, 0.05), (0.25, 0.15), (0.5, 0.4), (0.75, 1), (1, 1.25)], 
                                 [(0.0, 0.1), (0.25, 0.25), (0.5, 0.75), (1, 1.25)],
                                 [(0.0, 0.0), (0.1, 0.2), (0.3, 0.6), (0.6, 0.9), (1, 1.25)]], "TF3", 'categories', self )
           if 'transform_type' is "hearne1", a list of ranges for each parameter 
               Single-extreme piecewise functions
               m: maximum deviation from l of the distortion function
               p: the point that this occurs
               l: lower end point
               u: upper end point
           If 'transform_type' is "hearne2", a list of ranges for each 
               parameter. Double extreme piecewise linear functions with 
               variable endpoints are used to distort the lookup functions. 
               These functions are defined by 6 parameters, being m1, m2, p1, 
               p2, l and u; and the uncertainty ranges for these 6 parameters 
               should  be given as the values of this lookup uncertainty if 
               Hearne's method is chosen. The meaning of these parameters is 
               simply:
               m1: maximum deviation (peak if positive, bottom if negative) of 
                the distortion function from l in the first segment
               p1: where this peak occurs in the x axis
               m2: maximum deviation of the distortion function from l or u in 
                   the second segment
               p2: where the second peak/bottom occurs
               l : lower end point, namely the y value for x_min
               u : upper end point, namely the y value for x_max
               Example definition:
               LookupUncertainty([(-1, 2), (-1, 1), (0, 1), (0, 1), (0, 0.5), (0.5, 1.5)], "TF2", 'hearne', self, 0, 2)
            If 'transform_type' is "approximation", an analytical function 
                approximation (a logistic function) will be used, instead of a 
                lookup. This function also has 6 parameters whose ranges should 
                be given:
                A: the lower asymptote
                K: the upper asymptote
                B: the growth rate
                Q: depends on the value y(0)
                M: the time of maximum growth if Q=v
                Example definition:
                TODO
        :param name: name of the uncertainty
        :param transform_type: the method to be used for alternative generation. 
                     'categories', 'hearne' or 'approximation'
        :param msi: model structure interface, to be used for adding new 
                    parameter uncertainties
        :param min: min value the lookup function can take
        :param max: max value the lookup function can take
        
        '''
        super(LookupUncertainty, self).__init__(values, name)
        self.type = transform_type
        self.y_min = ymin
        self.y_max = ymax
        self.error_message = self.error_mesage.format(self.name)
        self.transform_functions = {self.HEARNE1: self._hearne1,
                                    self.HEARNE2: self._hearne2,
                                    self.APPROX: self._approx,
                                    self.CAT: self._cat}
        
        if self.type == "categories":
            msi.uncertainties.append(CategoricalUncertainty(range(len(values)), 
                                                            "c-"+self.name))
            msi.lookup_uncertainties.append(self)  
        elif self.type == "hearne1":
            msi.uncertainties.append(ParameterUncertainty(values[0], 
                                                          "m-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[1], 
                                                          "p-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[2], 
                                                          "l-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[3], 
                                                          "u-"+self.name))
            msi.lookup_uncertainties.append(self)  
        elif self.type == "hearne2":
            msi.uncertainties.append(ParameterUncertainty(values[0], 
                                                          "m1-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[1], 
                                                          "m2-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[2], 
                                                          "p1-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[3], 
                                                          "p2-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[4], 
                                                          "l-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[5], 
                                                          "u-"+self.name))
            msi.lookup_uncertainties.append(self) 
        elif self.type == "approximation":
            msi.uncertainties.append(ParameterUncertainty(values[0], 
                                                          "A-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[1], 
                                                          "K-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[2], 
                                                          "B-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[3], 
                                                          "Q-"+self.name))
            msi.uncertainties.append(ParameterUncertainty(values[4], 
                                                          "M-"+self.name))
            msi.lookup_uncertainties.append(self) 
        else: raise EMAError(self.error_message)
        
    
    def _get_initial_lookup(self, name):
        '''
        Helper function to retrieve the lookup function as defined in the
        vensim model. This lookup is transformed using a distortion function.
        
        :param name: name of variable in vensim model that contains the lookup
        
        '''
        
        a = vensimDLLwrapper.get_varattrib(name, 3)[0]
        elements = a.split('],', 1)
        b = elements[1][0:-1]
              
        list1 = []
        list2 = []
        number = []
        for c in b:
            if (c != '(') and (c != ')'):
                list1.append(c) 
         
        list1.append(',')      
        for c in list1:
            if c != ',':
                number.append(c)
            else:
                list2.append(float(''.join(number)))
                number[:] = []
        x = []
        y = []
        xT = True
        for i in list2:
            if xT:
                x.append(i)
                xT = False
            else:
                y.append(i)
                xT = True
        return (x, y)
    
    def _gen_log(self, t, A, K, B, Q, M):
        '''
        
        helper function implements a logistic function
        :param t:
        :param A:
        :param K:
        :param B:
        :param Q:
        :param M:
        
        '''
        decimal.getcontext().prec = 3
        ex = exp(-B) 
        res = A+((K-A)/(1+Q*pow(ex,t)/pow(ex,M)))
        return res
        
    def transform(self, case):
        if not self.x:
            # first time transform is called
            self.x, self.y = self._get_initial_lookup(self.name)
            self.x_min = min(self.x)
            self.x_max = max(self.x)
        try: 
            return self.transform_functions[self.type]
        except KeyError:
            raise EMAError(self.error_message)

    def identity(self):
        '''
        helper method that returns the elements that define an uncertainty. 
        By default these are the name, the lower value of the range and the 
        upper value of the range.
         
        '''
        # TODO this identity function is tricky. Identity is dependend on
        # the exact transform type
        
        
        return (self.name, self.values[0], self.values[1])    
    
    def _hearne1(self, case):
        m = case['m-'+self.name]
        p = case['p-'+self.name]
        l = case['l-'+self.name]
        u = case['u-'+self.name]

        for char in ['m-', 'p-', 'l-', 'u-']:
            case.pop(char+self.name)
        
        df = []
        for i in self.x:
            if i < p:
                df.append(l + ((m/(p-self.x_min))*i))
            else:
                df.append(l + m - ((m+l-u)*(i-p)/(self.x_min-p)))
        new_lookup = []
        for i in range(len(self.x)):
            new_lookup.append((self.x[i], max(min(df[i]*
                                        self.y[i], self.y_max), self.y_min)))
        return new_lookup
    
    def _hearne2(self, case):
        m1 = case['m1-'+self.name]
        m2 = case['m2-'+self.name]
        p1 = case['p1-'+self.name]
        p2 = case['p2-'+self.name]
        l = case['l-'+self.name]
        u = case['u-'+self.name]

        for char in ['m1-', 'm2-', 'p1-', 'p2-', 'l-', 'u-']:
            case.pop(char+self.name)
        
        df = [] # distortion function
        for i in self.x:
            if i < p1:
                df.append(l + ((m1/(p1-self.x_min))*i))
            else:
                if i < p2:
                    df.append(l + m1 - ((m1-m2+l-u)*(i-p1)/(p2-p1)))
                else:
                    df.append(u + m2 - (m2*(i-p2)/(self.x_max-p2)))
        new_lookup = []
        for i in range(len(self.x)):
            new_lookup.append((self.x[i], max(min(df[i]*self.y[i], self.y_max), 
                                              self.y_min)))
        return new_lookup
    
    def _approx(self, case):
        A = case['A-'+self.name]
        K = case['K-'+self.name]
        B = case['B-'+self.name]
        Q = case['Q-'+self.name]
        M = case['M-'+self.name]
        
        for char in ['A-', 'K-', 'B-', 'Q-', 'M-']:
            case.pop(char+self.name)
        
        new_lookup = []
        if self.x_max > 10:
            for i in range(int(self.x_min), int(self.x_max+1)):
                new_lookup.append((i, format(max(min(
                        self._gen_log(i, A, K, B, Q, M), 
                        self.y_max), self.y_min), '.3f'))) 
        else:
            for i in range(int(self.x_min*10), 
                           int(self.x_max*10+1), 
                           int(self.x_max)):
                new_lookup.append((i/10, 
                                   format(max(min(
                                                  self._gen_log(i/10, 
                                                                A, K, B, Q, M), 
                                                  self.y_max), 
                                              self.y_min), 
                                          '.3f')))
        return new_lookup
    
    def _cat(self, case):
        return self.values[case['c-'+self.name]] 
    
    

#==============================================================================
# test functions
#==============================================================================
#def test_uncertainties():
#    import EMAlogging
#    EMAlogging.log_to_stderr(EMAlogging.INFO)
#    params = [
##              CategoricalUncertainty(('1', '5',  '10'), 
##                                        "blaat", 
##                                        default = '5'),
##              ParameterUncertainty((0, 1), "blaat2"),
#              ParameterUncertainty((0, 10), "blaat3"),
#              ParameterUncertainty((0, 5), "blaat4", integer=True)
#              ]
#
#    sampler = FullFactorialSampler()
#    a = sampler.generateDesign(params, 10)
#    a = [combo for combo in a[0]]
#    for entry in a:
#        print entry
#    
#    print len(a)
   
#=============================================================================
# running the module stand alone
#==============================================================================
#if __name__=='__main__':
#    test_uncertainties()