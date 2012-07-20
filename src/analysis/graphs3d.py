'''
.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

This module provides functionality for generating various 3d graphs, using 
mayavi.


'''
from __future__ import division
import numpy as np
import scipy.stats.kde as kde 

import mayavi.mlab as mlab

from expWorkbench.EMAlogging import log_to_stderr, INFO, info
from expWorkbench import EMAError
import expWorkbench
from expWorkbench.util import load_results
from analysis.graphs import __discretesize

__all__ = ['envelopes3d', 'lines3d', 'scatter3d', 'envelopes3d_group_by']

def lines3d(results,
            outcomes,
            policy=None):
    '''
    Function for making a 3d lines plot. This function will plot
    the 2 supplied outcomes against each other over time. It will try
    to use `TIME` if provided. 
    
    :param results: The return from :meth:`perform_experiments`.
    :param outcome: The name of outcome of interest for which you want 
                    to make the 3d lines.
    :param policy: Optional argument, if provided, only for that particular 
                   policy the lines are visualized. It is *recommended* that 
                   this argument is provided if there are indeed multiple 
                   policies. 
    
    Code based on `mayavi example <http://github.enthought.com/mayavi/mayavi/auto/example_plotting_many_lines.html#example-plotting-many-lines>`_
    
    '''
    if len(outcomes)!=2:
        raise EMAError("you provided %s outcomes, while you should supply 2" % (len(outcomes)))
    
    #prepare the data
    experiments, results = results

    #get the results for the specific outcome of interest
    results1 = results.get(outcomes[0])
    results2 = results.get(outcomes[1])

    #restrict to policy if provided
    if policy:
        logical = experiments['policy'] == policy
        results1 = results1[logical]    
        results2 = results2[logical] 

    #get time axis
    try:
        time = results.pop('TIME')
    except:
        time =  np.arange(0, results.values()[0].shape[1])
        time = np.tile(time, (results1.shape[0],1))

    a = results1
    b = time
    c = results2
    
    # The number of points per line
    N = a.shape[1]

    mlab.figure(1, size=(400, 400), bgcolor=(1, 1, 1))
    mlab.clf()
    
    # We create a list of positions and connections, each describing a line.
    # We will collapse them in one array before plotting.
    x = list()
    y = list()
    z = list()
    #s = list()
    connections = list()
    
    # The index of the current point in the total amount of points
    index = 0
    
    # Create each line one after the other in a loop
    for i in range(a.shape[0]):
        x.append(a[i, :])
        y.append(b[i, :])
        z.append(c[i, :])
        # This is the tricky part: in a line, each point is connected
        # to the one following it. We have to express this with the indices
        # of the final set of points once all lines have been combined
        # together, this is why we need to keep track of the total number of
        # points already created (index)
        connections.append(np.vstack(
                           [np.arange(index,   index + N - 1.5),
                            np.arange(index+1, index + N - .5)]
                                ).T)
        index += N
    
    # Now collapse all positions, scalars and connections in big arrays
    x = np.hstack(x)
    y = np.hstack(y)
    z = np.hstack(z)
    #s = np.hstack(s)
    connections = np.vstack(connections)
    
    # Create the points
    src = mlab.pipeline.scalar_scatter(x, y, z)
    
    # Connect them
    src.mlab_source.dataset.lines = connections
    
    # The stripper filter cleans up connected lines
    lines = mlab.pipeline.stripper(src) #@UndefinedVariable
    
    # Finally, display the set of lines
    extent = [0,1,0,1,0,1]
    mlab.pipeline.surface(lines, extent=extent, line_width=1, opacity=.4)
    
    # And choose a nice view
    mlab.axes(xlabel=outcomes[0],
              ylabel = 'time',
              zlabel=outcomes[1],
              )
    mlab.show()

def scatter3d(results,
             outcomes,
             policy=None):
    '''
    Function for making a 3d scatter plots. This function will plot
    the 3 supplied outcomes against each other over time. If the data
    is a time series, end states will be used.
    
    :param results: The return from :meth:`perform_experiments`.
    :param outcomes: The names of the 3 outcomes of interest that are to be 
                     plotted.
    :param policy: Optional argument, if provided, only for that particular 
                   policy the scatter plot is generated. It is *recommended* 
                   that this argument is provided if there are indeed multiple 
                   policies. 
    
    '''
    if len(outcomes)!=3:
        raise EMAError("you provided %s outcomes, while you should supply 3" % (len(outcomes)))
    
    #prepare the data
    experiments, results = results
    
    #get the results for the specific outcome of interest
    results1 = results.get(outcomes[0])
    results2 = results.get(outcomes[1])
    results3 = results.get(outcomes[2])
    results = [results1, results2, results3]
    
    #restrict to policy if provided
    if policy:
        logical = experiments['policy'] == policy
        results = [entry[logical] for entry in results]
    
    #convert time series
    temp = []
    for entry in results:
        if len(entry.shape) > 1:
            if (entry.shape[1]>1):
                entry = entry[:,-1]
        temp.append(entry)
    results = temp
        
    #visualize results
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    
    extent = [0,1,0,1,0,1]
    s = mlab.points3d(results[0],results[1], results[2],
                      extent=extent, mode='point')
    mlab.axes(xlabel=outcomes[0],
              ylabel=outcomes[1],
              zlabel = outcomes[2],
              ) 
    mlab.show() 


def envelopes3d(results, 
                outcome, 
                policy = None, 
                logSpace=False,
                ymin = None,
                ymax = None):
    '''
    
    Function for making 3d envelopes. In contrast to the envelopes in 
    :mod:`graphs`, this version shows the density for every time step, instead 
    of only for the end state. Note that this function makes an envelope for 
    only 1 outcome. Moreover, it cannot plot the 3d envelopes for different 
    policies into the same figure Thus, for comparing different policies, 
    make separate 3d envelopes for each policy and put them side by side for 
    comparison.
    
    :param results: The return from :meth:`perform_experiments`.
    :param outcome: The name of outcome of interest for which you want 
                    to make the 3d envelopes.
    :param policy: Optional argument, if provided, only for that particular 
                   policy the envelope is calculated. It is *recommended* that 
                   this argument is provided if there are indeed multiple 
                   policies. 
    :param logSpace: boolean, if true, the log of the input data is used
    :param ymin: if provided, lower bound for the KDE, if not, 
                 ymin = np.min(results.get(outcome))
    :param ymax: if provided, lower bound for the KDE, if not, 
                 ymax = np.max(results.get(outcome))
    
    the following code snippet 
    
    >>> import expWorkbench.util as util
    >>> data = util.load_results(r'100 flu cases.cPickle')
    >>> outcome = 'deceased population region 1'
    >>> policy = 'adaptive policy'
    >>> envelopes3d(data, outcome=outcome, policy=policy)
    
    generates the following mayavi scene. It can be further edited using the 
    mayavi pipeline
    
    .. figure:: ../ystatic/envelopes3d.png
    
    
    '''
    def f(x, y, results):
        """
        function that performs the kde for each timestep
        """
        
        x1 = x[:,0]
        y1 = y[0,:]
        results = np.asarray(results)
        
        z = []
        for i in range(len(list(x1))):
            data = results[:, i]
            try:
                z1 = kde.gaussian_kde(data)
                z1 = z1.evaluate(y1)
            except:
                z1 = np.zeros(shape=y1.shape)
            z.append(z1)
        z = np.asarray(z)
        z = np.log(z+1)
    
        return z
    
    #prepare the data
    experiments, results = results

    #get time axis
    try:
        time = results.pop('TIME')[0, :]
    except:
        time =  np.arange(0, results.values()[0].shape[1])
    
    #get the results for the specific outcome of interest
    results = results.get(outcome)
    
    #log results
    if logSpace:
        minimum = np.min(results)
        if minimum < 0:
            a = results+ (-1*minimum)+1
            results = np.log(a)
        elif minimum == 0:
            results = np.log(results+1)
        else:
            results = np.log(results)
    
    #restrict to policy if provided
    if policy:
        results = results[experiments['policy'] == policy]
    
    #generate the grid
    if ymin == None:
        ymin = np.min(results)
    if ymax == None:
        ymax = np.max(results)



    length = min(100, results.shape[1])
    y = np.arange(ymin, ymax, (ymax-ymin)/length)
    X, Y = np.meshgrid(time, y)

    #calculate the kde for the grid
    Z = f(X.T,Y.T, results= results)
    
    #visualize results
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    
    extent = [0,10,0,10, 0,5]
    s = mlab.mesh(X,Y, Z.T, extent=extent,  opacity=1)
    mlab.axes(xlabel = 'time',
         ylabel = outcome) 
    mlab.show()

def envelopes3d_group_by(results, 
                         outcome,
                         groupBy = 'policy', 
                         discretesize = None,
                         logSpace=False,
                         ymin = None,
                         ymax = None):
    '''
    
    Function for making 3d envelopes. In contrast to the envelopes in 
    :mod:`graphs`, this version shows the density for every time step, instead 
    of only for the end state. Note that this function makes an envelope for 
    only 1 outcome. This envelopes will group the results based on the
    specified uncertainty. The user can supply a discretesize function
    to control the grouping in case of parameterUncertainties. This function
    will make a separate envelope for each group.
    
    :param results: The return from :meth:`run experiments`.
    :param outcome: Specify the name of outcome of interest for which you want to make 
                    the 3d envelopes.
    :param groupBy: The uncertainty to group by. (default=policy)
    :param discretesize: a discretesize function to control the grouping in case of parameterUncertainties
    :param logSpace: Boolean, if true, the log of the input data is used
    :param ymin: If provided, lower bound for the KDE, if not, ymin = np.min(results.get(outcome))
    :param ymax: If provided, lower bound for the KDE, if not, ymax = np.max(results.get(outcome))
    
    '''
    def f(x, y, results):
        """
        function that performs the kde for each timestep
        """
        
        x1 = x[:,0]
        y1 = y[0,:]
        results = np.asarray(results)
        
        z = []
        for i in range(len(list(x1))):
            data = results[:, i]
            try:
                z1 = kde.gaussian_kde(data)
                z1 = z1.evaluate(y1)
            except:
                z1 = np.zeros(shape=y1.shape)
            z.append(z1)
        z = np.asarray(z)
        z = np.log(z+1)
    
        return z
    
    #prepare the data
    experiments, results = results

    #get time axis
    try:
        time = results.pop('TIME')[0, :]
    except:
        time =  np.arange(0, results.values()[0].shape[1])
    
    
    def make_logical(cases, criterion, interval=False):
        if interval:
            
            return (cases[groupBy] >= criterion[0]) & (cases[groupBy] < criterion[1]) 
        else:
            return cases[groupBy]== criterion
    
    
    #get the results for the specific outcome of interest
    results = results.get(outcome)
    
    #log results
    if logSpace:
        results = np.log(results+1)
   
    #generate the grid
    if ymin == None:
        ymin = np.min(results)
        info("ymin: %s" % ymin)
    if ymax == None:
        ymax = np.max(results)
        info("ymax: %s" % ymax)

    length = min(100, results.shape[1])
    y = np.arange(ymin, ymax, (ymax-ymin)/length)
    X, Y = np.meshgrid(time, y)

    z = []

    #do the preparation for grouping by
    interval=False
    if (experiments[groupBy].dtype == np.float32) |\
       (experiments[groupBy].dtype == np.float64) |\
       ((experiments[groupBy].dtype == np.int) & (len(set(experiments[groupBy])) > 5)):
        interval=True
        if discretesize:
            categories = discretesize(experiments[groupBy])
        else:
            categories = __discretesize(experiments[groupBy])
    else:
        categories = set(experiments[groupBy])
        
    
    for category in categories:
        if interval:
            info("calculating kde for (%s, %s)" % (category))
        else:
            info("calculating kde for %s" % (category))
        logical = make_logical(experiments, category, interval)
        
        Z = f(X.T,Y.T, results=results[logical])
        z.append(Z)

    #calculate the kde for the grid
    #visualize results
    fig = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    
    fig.scene.disable_render = True
    for i, category in enumerate(categories):        
        if interval:
            info("plotting (%s, %s)" % (category))
        else:
            info("plotting %s" % (category))
        
        Z = z[i]
        extent = (-14+i*10,-6+i*10, 0,10, 0,5)
        s = mlab.mesh(X,Y, Z.T, extent=extent)
        mlab.outline(s, color=(.7, .7, .7), extent=extent)
        if i==0:
            mlab.axes(s,
                      extent=extent,
                      xlabel = '',
                      ylabel = '',
                      zlabel = 'density',
                      x_axis_visibility=False,
                      y_axis_visibility=False, 
                      z_axis_visibility=False) 
        
        category_name = repr(category)
            
        mlab.text(-16+10*i, i+10, category_name, z=-2, width=0.14)
    fig.scene.disable_render = False
    mlab.title(outcome, line_width=0.5)
    mlab.show()

#==============================================================================
# Test methods
#==============================================================================
def test_envelopes3d():
    results = expWorkbench.load_results(r'1000 flu cases.cPickle')
    exp, res = results
    
    logical = exp['policy'] == 'adaptive policy'
    new_exp = exp[logical][0:100]
    new_res = {}
    for key, value in res.items():
        new_res[key] = value[logical][0:100, :]
    

    envelopes3d((new_exp, new_res), 'infected fraction R1', logSpace=True)

def test_envelopes3d_group_by():
    results = expWorkbench.load_results(r'1000 flu cases.cPickle')

    envelopes3d_group_by(results, 
                         outcome='infected fraction R1', 
                         groupBy="normal interregional contact rate",
                         logSpace=True)



def test_lines3d():
    results = expWorkbench.load_results(r'eng_trans_100.cPickle')
    lines3d(results, outcomes=['installed capacity T1',
                               'installed capacity T2'])

def test_scatter3d():
    #load the data
    experiments, results = load_results(r'1000 flu cases.cPickle')
    
    #transform the results to the required format
    newResults = {}
    
    #get time and remove it from the dict
    time = results.pop('TIME')
    
    for key, value in results.items():
        if key == 'deceased population region 1':
            newResults[key] = value[:,-1] #we want the end value
        else:
            # we want the maximum value of the peak
            newResults['max peak'] = np.max(value, axis=1) 
            
            # we want the time at which the maximum occurred
            # the code here is a bit obscure, I don't know why the transpose 
            # of value is needed. This however does produce the appropriate results
            logicalIndex = value.T==np.max(value, axis=1)
            newResults['time of max'] = time[logicalIndex.T]
    results = (experiments, newResults)
    scatter3d(results, outcomes=newResults.keys())


if __name__ == '__main__':
    log_to_stderr(level= INFO)
#    test_envelopes3d()
    test_envelopes3d_group_by()
#    test_lines3d()
#    test_scatter3d()