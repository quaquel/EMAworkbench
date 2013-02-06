r'''
.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

.. highlight:: python
   :linenothreshold: 5

.. rubric:: example of use

We provide here an extended example of how :mod:`prim` can be used. As 
a starting point we use the cPickle file generated and saved in 
the tutorial on the Flu model. We use prim to find whether there are 
one or more subspaces of the uncertainty space that result
in a high number of deaths for the 'no policy' runs.

To this end, we need to make our own :func:`classify`. This function should 
extract from the results, those related to the deceased population and 
classify them into two distinct classes:

.. math::
 
      f(x)=\begin{cases} 
               1, &\text{if $x > 1000000$;}\\
               0, &\text{otherwise.}
            \end{cases}

Here, :math:`x` is the endstate of 'deceased population region 1'.

A second thing that needs to be done is to extract from the saved results only
those results belonging to 'no policy'. To this end, we can use logical 
indexing. That is, we can use `boolean arrays <http://www.scipy.org/Tentative_NumPy_Tutorial#head-0dffc419afa7d77d51062d40d2d84143db8216c2>`_ 
for indexing. In our case, we can get the logical index in a straightforward 
way. ::
    
    logicalIndex = experiments['policy'] == 'no policy'

We can now use this index to modify the loaded results to only include 
the experiments and results we want. The modified results can than be
used as input for prim. 

Together, this results in the following script: 
    
.. literalinclude:: ../../../../src/examples/primFluExample.py
   :linenos:

which generates the following figures.

.. figure:: ../../ystatic/boxes_individually.png
   :align:  center

.. figure:: ../../ystatic/boxes_together.png
   :align:  center


'''
from __future__ import division
import numpy as np
from types import StringType
from sys import stdout
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

from expWorkbench.ema_exceptions import EMAError
import primCode.primDataTypeAware as recursivePrim
from analysis.scenario_discovery import calculate_sd_metrics
from expWorkbench.ema_logging import log_to_stderr, INFO, info, debug
from examples.flu_vensim_example import FluModel
import expWorkbench


__all__ = ['perform_prim', 'write_prim_to_stdout', 'show_boxes_individually',
           'show_boxes_together']

COLOR_LIST = ['g',
              'r',
              'm',
              'c',
              'y',
              'k']

def orig_obj_func(old_y, new_y):
    r'''
        
    The original objective function used by PRIM. In the default implementation
    in peeling and pasting, PRIM selects the box that maximizes the average 
    value of new_y. This function is called for each candidate box to calculate
    its average value:
    

    .. math::
        
        obj = \text{ave} [y_{i}\mid x_{i}\in{B-b}]

    
    where :math:`B-b` is the set of candidate new boxes. 
    
    :param old_y: the y's belonging to the old box
    :param new_y: the y's belonging to the new box
    
    '''
    return np.mean(new_y)


def def_obj_func(y_old, y_new):
    r'''
    the default objective function used by prim, instead of the original
    objective function, this function can cope with continous, integer, and
    categorical uncertainties.      
    
    .. math::
        
        obj = \frac
             {\text{ave} [y_{i}\mid x_{i}\in{B-b}] - \text{ave} [y\mid x\in{B}]}
             {|n(y_{i})-n(y)|}
    
    where :math:`B-b` is the set of candidate new boxes, :math:`B` the old box 
    and :math:`y` are the y values belonging to the old box. :math:`n(y_{i})` 
    and :math:`n(y)` are the cardinality of :math:`y_{i}` and :math:`y` 
    respectively. So, this objective function looks for the difference between
    the mean of the old box and the new box, divided by the change in the 
    number of data points in the box. This objective function offsets a problem 
    in case of categorical data where the normal objective function often 
    results in boxes mainly based on the categorical data.  
    
    :param old_y: the y's belonging to the old box
    :param new_y: the y's belonging to the new box
    
    '''
    
    mean_old = np.mean(y_old)
    mean_new = np.mean(y_new)
    obj = 0
    if mean_old != mean_new:
        if y_old.shape >= y_new.shape:
            obj = (mean_new-mean_old)/(y_old.shape[0]-y_new.shape[0])
        else:
            obj = (mean_new-mean_old)/(y_new.shape[0]-y_old.shape[0])
    return obj


def perform_prim(results, 
                 classify, 
                 peel_alpha = 0.05, 
                 paste_alpha = 0.05,
                 mass_min = 0.05, 
                 threshold = None, 
                 pasting=True, 
                 threshold_type=1,
                 obj_func=def_obj_func):
    r'''
    
    perform Patient Rule Induction Method (PRIM). This function performs 
    the PRIM algorithm on the data. It uses a Python implementation of PRIM
    inspired by the `R <http://www.oga-lab.net/RGM2/func.php?rd_id=prim:prim-package>`_ 
    algorithm. Compared to the R version, the Python version is data type aware. 
    That is, real valued, ordinal, and categorical data are treated 
    differently. Moreover, the pasting phase of PRIM in the R algorithm is
    not consistent with the literature. The Python version is. 
    
    the PRIM algorithm tries to find subspaces of the input space that share
    some characteristic in the output space. The characteristic that the 
    current implementation looks at is the mean of the results. Thus, the 
    output space is 1-D, and the characteristic is assumed to be continuous.
    
    As a work around, to deal with classes, the user can supply a classify 
    function. This function should return a binary classification 
    (i.e. 1 or 0). Then, the mean of the box is indicative of the concentration 
    of cases of class 1. That is, if the specified threshold is say 0.8 and the 
    threshold_type is 1, PRIM looks for subspaces of the input space that 
    contains at least 80\% cases of class 1.   
    
    :param results: the return from :meth:`perform_experiments`.
    :param classify: either a string denoting the outcome of interest to use
                     or a function. In case of a string and time series data, 
                     the end state is used.
    :param peel_alpha: parameter controlling the peeling stage (default = 0.05). 
    :param paste_alpha: parameter controlling the pasting stage (default = 0.05).
    :param mass_min: minimum mass of a box (default = 0.05). 
    :param threshold: the threshold of the output space that boxes should meet. 
    :param pasting: perform pasting stage (default=True) 
    :param threshold_type: If 1, the boxes should go above the threshold, if -1
                           the boxes should go below the threshold, if 0, the 
                           algorithm looks for both +1 and -1.
    :param obj_func: The objective function to use. Default is 
                     :func:`def_obj_func`
    :return: a list of PRIM objects.
    
    for each box, the scenario discovery metrics *coverage* and *density* 
    are also calculated:
    
    
    .. math::
 
        coverage=\frac
                    {{\displaystyle\sum_{y_{i}\in{B}}y_{i}{'}}}
                    {{\displaystyle\sum_{y_{i}\in{B^I}}y_{i}{'}}}
    
    
    where :math:`y_{i}{'}=1` if :math:`x_{i}\in{B}` and :math:`y_{i}{'}=0`
    otherwise.
    
    
    .. math::
 
        density=\frac
                    {{\displaystyle\sum_{y_{i}\in{B}}y_{i}{'}}}
                    {{\displaystyle\left|{y_{i}}\right|\in{B}}}
    
    where :math:`y_{i}{'}=1` if :math:`x_{i}\in{B}` and :math:`y_{i}{'}=0`
    otherwise, and :math:`{\displaystyle\left|{y_{i}}\right|\in{B}}` is the
    cardinality of :math:`y_{i}`.
    
    
    Density is the ratio of the cases of interest in a box to the 
    total number of cases in that box. *density* is identical to the mean
    in case of a binary classification.  For more detail on these metrics see 
    `Bryant and Lempert (2010) <http://www.sciencedirect.com/science/article/pii/S004016250900105X>`_
    
    .. rubric:: references to relevant papers 
        
    * `original PRIM paper <http://www.springerlink.com/content/x3gpv05t34620878/>`_
    * `paper on ordinal data and PRIM <http://www.sciencedirect.com/science/article/pii/S095741740700231X>`_
        
    **ema application** 
        
    * `Lempert et al. (2006) <http://mansci.journal.informs.org/content/52/4/514>`_
    * `Groves and Lempert (2007) <http://www.sciencedirect.com/science/article/pii/S0959378006000896#ref_bib19>`_
    * `Bryant and Lempert (2010) <http://www.sciencedirect.com/science/article/pii/S004016250900105X>`_
    
    '''
    experiments, results = results
    
    #make y
    if type(classify) == StringType:
        results = results.get(classify)
        if len(results.shape) == 2:
            y = results[:, -1]
        else:
            y = results
            
        count = np.zeros(y.shape)
        count[y*threshold_type > threshold*threshold_type] = 1
        cases_of_interest = np.sum(count)
        info("number of cases of interest is %d" % (np.sum(count)))
    elif callable(classify):
        y = classify(results)
        cases_of_interest = np.sum(y)
        info("number of cases of interest is %d" % (np.sum(y)))
    else:
        raise EMAError("incorrect specification of classify, this should be a function or a string")
   

    x = experiments
    
    #perform prim
    boxes = recursivePrim.perform_prim(x, y, box_init=None, peel_alpha=peel_alpha, 
                                            paste_alpha=paste_alpha, mass_min=mass_min, 
                                            threshold=threshold, pasting=pasting, 
                                            threshold_type=threshold_type,obj_func=obj_func,
                                            cases_of_interest=cases_of_interest)
    
    #calculate scenario discovery metrics and add these to boxes
    boxes = calculate_sd_metrics(boxes, y, threshold, threshold_type)
    
    #return prim
    return boxes



def __filter(boxes, uncertainties=[]):
    dump_box=boxes[-1]
    boxes=boxes[0:-1]
    
    uv=uncertainties
    #iterate over uncertainties
    names = []

    if uncertainties:
        uv=uncertainties
    else:
        uv = [entry[0] for entry in dump_box.dtype.descr]

    for name in uv:
        
        #determine whether to show
        for box in boxes:
            minimum = box[name][0]
            maximum = box[name][1]
            value = box.dtype.fields.get(name)[0]
            if value == 'object':
                a = dump_box[name][0]
                
                if len(a) != len(minimum):
                    ans = False
                else:
                    ans = np.all(np.equal(a, minimum))
                if not ans:
                    names.append(name)
                    break
            elif (minimum > dump_box[name][0]) or\
                 (maximum < dump_box[name][1]):
                names.append(name)
                break
    a = set(uv) -set(names)
    a = list(a)
    a.sort()
    string_list = ", ".join(a)
    
    info(string_list + " are not not visualized because they are not restricted")
    
    uv = names
    return uv

def write_prim_to_stdout(boxes, uv=[], screen=True):
    '''
    Summary function for printing the results of prim to stdout (typically
    the console). This function first prints an overview of the boxes,
    specifying their mass and mean. Mass specifies the fraction of cases in 
    the box, mean specifies the average of the cases. 
    
    :param boxes: the prim boxes as returned by :func:`perform_prim`
    :param uv: the uncertainties to show in the plot. Defaults to an empty list,
               meaning all the uncertainties will be shown. If the list is not
               empty only the uncertainties specified in uv will be plotted. 
    :param screen: boolean, if True, the uncertainties for which all the boxes
                   equal the size of the dump box are not visualized 
                   (default=True)

    
    if one wants to print these results to a file, the easiest way is to
    redirect stdout:: 

        sys.stdout = open(file.txt, 'w')
    '''
    prims = boxes
    boxes = [element.box for element in boxes]
    
    if screen:
        uv=__filter(boxes,uv)
    else:
        uv = [entry[0] for entry in boxes[0].dtype.descr]
    
    time.sleep(1)
  
    print '           \tmass\tmean\tcoverage\tdensity'
    for i, entry in enumerate(prims[0:-1]):
        print ' box %s:\t%s\t%s\t%s\t%s' %(i+1, 
                                                entry.box_mass, 
                                                entry.y_mean,
                                                entry.coverage,
                                                entry.density)
    print 'rest box    :\t%s\t%s\t%s\t%s' %(prims[-1].box_mass, 
                                            prims[-1].y_mean,
                                            prims[-1].coverage,
                                            prims[-1].density)
    
    print "box limits"
    stdout.write("  \t  ")
    
    uncertainties=uv
    
    for uncertainty in uncertainties:
        stdout.write(uncertainty+"\t")
    stdout.write("\n")        
    for i, box in enumerate(boxes):
        print "box %s:" % str(i+1)
        
        stdout.write("min:\t") 
     
        for name in uncertainties: 
            element = box[name][0]
            stdout.write(str(element)+'\t')
        
        stdout.write("\nmax:\t")
        for name in uncertainties:
            element = box[name][1] 
            stdout.write(str(element)+'\t')
        stdout.write("\n")

def show_boxes_individually(boxes, results, uv=[], screen=True):
    '''
    
    This functions visually shows the size of a list of prim boxes. Each
    box is a single plot.  The dump box is not shown. The size of the
    boxes is normalized, where 0 is the lowest sampled value for each
    uncertainty and 1 is the highest sampled value for each uncertainty. This
    is visualized using a light grey background.
    
    :param boxes: the list of prim objects as returned by :func:`perform_prim`.
    :param results: the results as returned by :meth:`perform_experiments`
    :param uv: the uncertainties to show in the plot. Defaults to an empty list,
               meaning all the uncertainties will be shown. If the list is not
               empty only the uncertainties specified in uv will be plotted. 
    :param screen: boolean, if True, the uncertainties for which all the boxes
                   equal the size of the dump box are not visualized 
                   (default=True)
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ instance
    
    '''
    experiments, results = results
    boxes = [element.box for element in boxes]
    
    #determine minima and maxima
    boxes = __normalize(boxes, experiments)
    
    uncertainties = sort_uncertainities(experiments, boxes[0], boxes[-1])

    #iterate over uncertainties
    if screen:
        uv = __filter(boxes, uncertainties)
    else:
        uv = uncertainties
 
    boxes = boxes[0:-1]
    
    #plot results    
    figure = plt.figure()
    grid = mpl.gridspec.GridSpec(1, len(boxes))                             
    
    for j, box in enumerate(boxes):
        ax = figure.add_subplot(grid[0,j])
        rect = mpl.patches.Rectangle((-0.5, 0), len(uv)+0.5, 1, facecolor="#C0C0C0",
                                 alpha=0.25, edgecolor="#C0C0C0")
        ax.add_patch(rect)
        ax.set_xlim(xmin= -0.5, xmax=len(uv)-0.5)
        ax.set_ylim(ymin=-0.2, ymax=1.2)
    
        ax.xaxis.set_ticks([x for x in range(len(uv))])
        xtickNames = plt.setp(ax, xticklabels = uv)
        plt.setp(xtickNames, rotation=90, fontsize=8)

        ax.set_ylabel('normalized uncertainty bandwidth', 
                       rotation=90, 
                       fontsize=8)        
        ytickNames = ax.get_yticklabels()
        plt.setp(ytickNames, rotation=90, fontsize=8)
        for i, name in enumerate(uv): 
            value = box.dtype.fields.get(name)[0]
            if value == 'object':
                y = box[name][0]
                x = [i+0.1*(j+1) for entry in range(len(y))]
                ax.scatter(x,y,edgecolor='b',
                           facecolor='b')
            else:
                ax.plot([i+0.1*(j+1), i+0.1*(j+1)], box[name][:], 
                        c='b')
    
    return figure

def sort_uncertainities(experiments, box, dump_box):
    uncertainties = []
    
    for entry in experiments.dtype.descr:
        uncertainties.append(entry[0])

    size_restricted_dimensions = []
    for uncertainty in uncertainties:
        value = box.dtype.fields.get(uncertainty)[0]
        if value == 'object':
            tot_nr_categories = len(dump_box[uncertainty][0])
            length = len(box[uncertainty][0])/tot_nr_categories
            
        else:
            interval = box[uncertainty]
            length = interval[1]-interval[0]

        size_restricted_dimensions.append((length, uncertainty))
     
    sorted_uncertainties = sorted(size_restricted_dimensions)
    sorted_uncertainties = [entry[1] for entry in sorted_uncertainties]
    return sorted_uncertainties


def show_boxes_together(boxes, results, uv=[], screen=True):
    '''
    
    This functions visually shows the size of a list of prim boxes. 
    Each box has its own color. The dump box is not shown. The size of the
    boxes is normalized, where 0 is the lowest sampled value for each
    uncertainty and 1 is the highest sampled value for each uncertainty. his
    is visualized using a light grey background.
    
    :param boxes: the list of prim objects as returned by :func:`perform_prim`.
    :param results: the results as returnd by :meth:`perform_experiments`
    :param uv: the uncertainties to show in the plot. Defaults to an empty list,
               meaning all the uncertainties will be shown. If the list is not
               empty only the uncertainties specified in uv will be plotted. 
    :param filter: boolean, if True, the uncertainties for which all the boxes
                   equal the size of the dump box are not visualized 
                   (default=True)
    :rtype: a `figure <http://matplotlib.sourceforge.net/api/figure_api.html>`_ instance
    
    '''
    experiments, results = results
    boxes = [element.box for element in boxes]
    
    #determine minima and maxima
    boxes = __normalize(boxes, experiments)
    
    dump_box = boxes[-1]
    boxes = boxes[0:-1]
    
    uncertainties = sort_uncertainities(experiments, boxes[0], dump_box)
    
#    uncertainties = []
#    for entry in experiments.dtype.descr:
#        uncertainties.append(entry[0])

    if not uv:
        uv = uncertainties
    
    #plot results    
    figure = plt.figure()
    ax = figure.add_subplot(111)
    
    #iterate over uncertainties
    
    names = []
    i = -1
    for name in uncertainties:
        if name in uv:
            show_uncertainty = True
            if screen:
                show_uncertainty=False
                #determine whether to show
                for box in boxes:
                    minimum = box[name][0]
                    maximum = box[name][1]
                    value = box.dtype.fields.get(name)[0]
                    if value == 'object':
                        debug("filtering name")
                        debug(dump_box[name][0])
                        debug(minimum)
                        a = dump_box[name][0]
                        if len(minimum) == len(a):
                            ans = np.all(np.equal(a, minimum))
                        else:
                            ans = False
                        if not ans:
                            show_uncertainty = True
                            break
                    elif (minimum > dump_box[name][0]) or\
                         (maximum < dump_box[name][1]):
                        show_uncertainty = True
                        break
            if show_uncertainty:    
                i+=1
                names.append(name)
                
                #iterate over boxes
                for j, box in enumerate(boxes):
                    if value == 'object':
                        y = box[name][0]
                        x = [i+0.1*(j+1) for entry in range(len(y))]
                        ax.scatter(x,y,edgecolor=COLOR_LIST[j%len(COLOR_LIST)],
                                   facecolor=COLOR_LIST[j%len(COLOR_LIST)])
                    else:
                        ax.plot([i+0.1*(j+1), i+0.1*(j+1)], box[name][:], 
                                c=COLOR_LIST[j%len(COLOR_LIST)])

    rect = mpl.patches.Rectangle((-0.5, 0), i+1.5, 1, facecolor="#C0C0C0",
                                 alpha=0.25, edgecolor="#C0C0C0")
    ax.add_patch(rect)
    ax.set_xlim(xmin= -0.5, xmax=len(names)-0.5)
    ax.set_ylim(ymin=-0.2, ymax=1.2)
    
    ax.xaxis.set_ticks([x for x in range(len(names))])
    xtickNames = plt.setp(ax, xticklabels = names)
    plt.setp(xtickNames, rotation=90, fontsize=12)
    
    ytickNames = ax.get_yticklabels()
    plt.setp(ytickNames, rotation=90, fontsize=12)
    
    ax.set_ylabel('normalized uncertainty bandwidth', 
               rotation=90, 
               fontsize=12)     
    return figure

def __normalize(boxes, experiments):
    minima = []
    maxima = []
    for entry in experiments.dtype.descr:
        name = entry[0]
        value = experiments.dtype.fields.get(entry[0])[0]
        if value != 'object':
            minima.append(np.min(experiments[name]))
            maxima.append(np.max(experiments[name]))
        else:
            minima.append(0)
            maxima.append(len( set(experiments[name]) ) )
    minima = np.asarray(minima)
    maxima = np.asarray(maxima)
    
    #normalize experiments
    a = 1/(maxima-minima)
    b = -1*minima/(maxima-minima)
   
    dtypes = []
    cats = {}
    for i, entry in enumerate(boxes[0].dtype.descr):
        name = entry[0]
        value = experiments.dtype.fields.get(name)[0]
        dtype = float
        if value == 'object':
            dtype = object
            cats[name] = list(set(experiments[name]))
        dtypes.append((name,dtype))
   
    temp_boxes = []
    
    for box in boxes:
        temp_box = np.zeros((2, ), dtypes)

        for i, entry in enumerate(temp_box.dtype.descr):
            name = entry[0]
            value = temp_box.dtype.fields.get(name)[0]
            if value == 'object':
                c_b = box[name][0]
                values = np.asarray([cats[name].index(c) for c in c_b])
                if a[i] != 1:
                    a_i = 1/(maxima[i]-1)
                    values = a_i*values
                else:
                    values = [1/2]
                temp_box[name][0] = values
                temp_box[name][1] =  values
            else:
                temp_box[name][0] = a[i]*box[name][0] + b[i]
                temp_box[name][1] = a[i]*box[name][1] + b[i]
        temp_boxes.append(temp_box)
    boxes= temp_boxes
    return boxes

if __name__ == '__main__':
    log_to_stderr(level= INFO)
    
    model = FluModel(r'..\..\models\flu', "fluCase")
    results = expWorkbench.util.load_results(r'1000 flu cases.cPickle')
    boxes = perform_prim(results, 
                         classify=model.outcomes[1].name, 
                         threshold_type=1,
                         threshold=0.8)
    write_prim_to_stdout(boxes)
    show_boxes_individually(boxes, results)
    plt.show()  
    