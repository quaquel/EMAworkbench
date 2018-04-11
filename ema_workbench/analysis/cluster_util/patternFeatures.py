'''
Created on Sep 27, 2011

@author: gyucel
'''
from __future__ import (print_function, unicode_literals)
import numpy as np
from ema_workbench.util import utilities

def mean(ds):
    return np.mean(ds)

def variance(ds):
    return np.var(ds)

def stdDev(ds):
    return np.std(ds)

def linearFit(ds):
    return nthDegreeFit(range(len(ds)),ds,1)

def quadraticFit(ds):
    return nthDegreeFit(range(len(ds)),ds,2)

def nthDegreeFit(x,ds,degree):
    return np.polyfit(x, ds, degree)


def exponentialFit(ds):
    return 0

def autoCovariance(ds,lag):
    N = len(ds)
    k = lag
    m = mean(ds)
    sum = 0
    for i in range(N-k):
        sum += (ds[i]-m)*(ds[i+k]-m)
    return sum/N

def autoCorrelation(ds, lag):
    k = np.abs(lag)
    return autoCovariance(ds,k)/autoCovariance(ds,0)

def crossCorrelation(ds1, ds2, lag):
    std1 = stdDev(ds1)
    std2 = stdDev(ds2)
    m1 = mean(ds1)
    m2 = mean(ds2)
    K = 0
    for i in range(len(ds1)-lag):
        K+= (ds1[i+lag]-m1)*(ds2[i]-m2)
    crossCorr = (K/len(ds1))/(std1*std2)
    return crossCorr   

def varAutoCorrelation(ds, lag):
    N = len(ds)
    k = lag
    sum = 0
    for j in range(N-2):
        i = j+1
        a = autoCorrelation(ds,k-i)+autoCorrelation(ds,k+i)-2*autoCorrelation(ds,k)*autoCorrelation(ds,i)
        aSqr = a*a
        sum += (N-i)*aSqr
    var = sum/(N*(N+2))
    return var

def periodDominance(ds):
    Y = np.fft.rfft(ds)
    n = len(Y)
    powerSpect = np.abs(Y)**2
    timeStep = 1 
    freq = np.fft.fftfreq(n, d=timeStep)
    print(len(freq), len(powerSpect))
    for i in range(len(freq)/2+1):
        print(freq[i], 1/freq[i], powerSpect[i])


if __name__ == '__main__':
    
    cases, results = utilities.load_results('PatternSet_Periodic.cpickle')
    dataSeries = results.get('outcome')
    ds1 = dataSeries[25]
    ds2 = dataSeries[26]
    
    print(linearFit(ds1))
    print(quadraticFit(ds1))
    print(mean(ds1), variance(ds1), stdDev(ds1))
    print(autoCovariance(ds1,0))
    for k in range(31):
        print(k,autoCorrelation(ds1,k))
    
    for k in range(31):
        print(k, crossCorrelation(ds1,ds2,k))
    
    periodDominance(ds1)    
#    data = np.zeros(5,dtype=int)
#    data[0] = 2
#    data[1] = 4
#    data[2] = 6
#    data[3] = 8
#    data[4] = 10
#    print mean(data), varianqce(data), stdDev(data)
#    print autoCovariance(data,2)
#    print autoCorrelation(data,0),autoCorrelation(data,2)
#    linearFit(data)
#    r = nthDegreeFit(data,0)
#    print r