'''
Created on 13 jan. 2013

@author: localadmin
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

x = (np.random.rand(5000)*2)-1
y = (np.random.rand(5000)*2)-1


logical = x**2 + y**2
logical = logical < 0.5

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
    
ax1.scatter(x[logical],y[logical], c='r')
ax1.scatter(x[logical==False],y[logical==False], c='b')

x = ((x+1)/2)
y = ((y+1)/2)

x_k = x**2 * y**2
y_k = x**2 * y**2
x = x_k
y = y_k

ax2.scatter(x[logical],y[logical], c='r')
ax2.scatter(x[logical==False],y[logical==False], c='b')


plt.show()