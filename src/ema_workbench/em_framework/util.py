'''


'''
from __future__ import (unicode_literals, print_function, absolute_import,
                        division)


from collections import OrderedDict, MutableMapping

try:
    # we assume python 2
    from collections import MutableMapping
except ImportError:
    # so it must by python 3
    from collections.abc import MutableMapping

import copy
import six

from ..util import EMAError

# from .parameters import Parameter

# Created on Jul 16, 2016
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

__all__ = ['NamedObject']

class NamedObject(object):

    def __init__(self, name):
        self.name = name
        
class NamedObjectMap(object):
     
    def __init__(self, type):
        super(NamedObjectMap, self).__init__()
        self.type = type
        self._data = OrderedDict()
         
        if not issubclass(type, NamedObject):
            raise TypeError("type must be a NamedObject")
         
    def clear(self):
        self._data = OrderedDict()
         
    def __len__(self):
        return len(self._data)
 
    def __getitem__(self, key):
        if isinstance(key, six.integer_types):
            for i, (_, v) in enumerate(six.iteritems(self._data)):
                if i == key:
                    return v
            raise KeyError(key)
        else:
            return self._data[key]
     
    def __setitem__(self, key, value):
        if not isinstance(value, self.type):
            raise TypeError("can only add " + self.type.__name__ + " objects")
         
        if isinstance(key, six.integer_types):
            self._data = OrderedDict([(value.name, value) if i==key else (k, v) for i, (k, v) in enumerate(six.iteritems(self._data))])
        else: 
            if value.name != key:
                raise ValueError("key does not match name of " + self.type.__name__)
             
            self._data[key] = value
         
    def __delitem__(self, key):
        del self._data[key]
         
    def __iter__(self):
        return iter(self._data.values())
     
    def __contains__(self, item):
        return item in self._data
     
    def extend(self, value):
        if hasattr(value, "__iter__"):
            for item in value:
                if not isinstance(item, self.type):
                    raise TypeError("can only add " + self.type.__name__ + " objects")
                 
            for item in value:
                self._data[item.name] = item
        elif isinstance(value, NamedObject):
            self._data[value.name] = value
        else:
            raise TypeError("can only add " + str(type) + " objects")
             
    def __add__(self, value):
        self.extend(value)
        return self
         
    def __iadd__(self, value):
        self.extend(value)
        return self
     
    def keys(self):
        return self._data.keys()
    

class NamedDict(NamedObject, MutableMapping):
    
    def __init__(self, name=None, **kwargs):
        self._dict = dict(**kwargs)
        
        if name is None:
            name = repr(self._dict)
        
        super(NamedDict, self).__init__(name)
        
        
    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value
    
    def __delitem__(self, key):
        del self._dict[key]
        
    def __iter__(self):
        return iter(self._dict)
    
    def __len__(self):
        return len(self._dict)

    
def combine(*args):
    '''combine scenario and policy into a single experiment dict
    
    Parameters
    ----------
    args : two or more dicts that need to be combined
    
    
    Returns
    -------
    a single unified dict containing the entries from all dicts
    
    Raises
    ------
    EMAError 
        if a keyword argument exists in more than one dict
    '''
    experiment = copy.copy(args[0])
    for entry in args[1::]:
        overlap = set(experiment.keys()).intersection(set(entry.keys()))
        if overlap:
            raise EMAError(('parameters exist in two dicts' + overlap))
        experiment.update(entry)
            

    return experiment
    