'''
Created in June 2012

@author: Bas M.J. Keijser
         Jan H. Kwakkel & Erik Pruyt (model class function)
'''
from __future__ import division
import numpy as np
from math import exp

import expWorkbench.EMAlogging as EMAlogging
from expWorkbench import ModelEnsemble
from expWorkbench import vensim
from expWorkbench import vensimDLLwrapper as venDLL

from expWorkbench import ParameterUncertainty, CategoricalUncertainty
from expWorkbench.vensim import VensimModelStructureInterface
from expWorkbench.outcomes import Outcome
from expWorkbench import util, save_results, load_results

#from analysis import clusterer
from analysis.graphs import lines
#from analysis.interactive_graphs import make_interactive_plot
import matplotlib.pyplot as plt

import ABP
import distance_gonenc as dg

class ScarcityModel(VensimModelStructureInterface):
    modelFile = r'\Metals EMA.vpm'
        
    outcomes = [Outcome('relative market price', time=True),
                Outcome('supply demand ratio', time=True),
                Outcome('real annual demand', time=True),
                Outcome('produced of intrinsically demanded', time=True),
                Outcome('supply', time=True),
                Outcome('Installed Recycling Capacity', time=True),
                Outcome('Installed Extraction Capacity', time=True)]
                
    uncertainties = [ParameterUncertainty((0, 0.5), "price elasticity of demand"),
                     ParameterUncertainty((0.6, 1.2), "fraction of maximum extraction capacity used"),
                     ParameterUncertainty((1,4), "initial average recycling cost"),
                     ParameterUncertainty((0, 15000),"exogenously planned extraction capacity"),
                     ParameterUncertainty((0.1, 0.5),"absolute recycling loss fraction"),
                     ParameterUncertainty((0, 0.4),"normal profit margin"),
                     ParameterUncertainty((100000, 120000),"initial annual supply"),
                     ParameterUncertainty((1500000, 2500000),"initial in goods"),
                     ParameterUncertainty((1,  10),"average construction time extraction capacity"),
                     ParameterUncertainty((20,  40),"average lifetime extraction capacity"),
                     ParameterUncertainty((20, 40),"average lifetime recycling capacity"),
                     ParameterUncertainty((5000,  20000),"initial extraction capacity under construction"),
                     ParameterUncertainty((5000, 20000),"initial recycling capacity under construction"),
                     ParameterUncertainty((5000, 20000),"initial recycling infrastructure"),
                          
                     #order of delay
                     CategoricalUncertainty((1,4,10, 1000), "order in goods delay", default = 4),
                     CategoricalUncertainty((1,4,10), "order recycling capacity delay", default = 4),
                     CategoricalUncertainty((1,4,10), "order extraction capacity delay", default = 4),
                          
                     #uncertainties associated with lookups
                     ParameterUncertainty((20, 50),"lookup shortage loc"),
                     ParameterUncertainty((1, 5),"lookup shortage speed"),
                  
                     ParameterUncertainty((0.1, 0.5),"lookup price substitute speed"),
                     ParameterUncertainty((3, 7),"lookup price substitute begin"),
                     ParameterUncertainty((15, 25),"lookup price substitute end"),
                          
                     ParameterUncertainty((0.01, 0.2),"lookup returns to scale speed"),
                     ParameterUncertainty((0.3, 0.7),"lookup returns to scale scale"),
                  
                     ParameterUncertainty((0.01, 0.2),"lookup approximated learning speed"),
                     ParameterUncertainty((0.3, 0.6),"lookup approximated learning scale"),
                     ParameterUncertainty((30, 60),"lookup approximated learning start")]


    def returnsToScale(self, x, speed, scale):
    
        return (x*1000, scale*1/(1+exp(-1* speed * (x-50))))
     
    def approxLearning(self, x, speed, scale, start):
        x = x-start
        loc = 1 - scale
        a = (x*10000, scale*1/(1+exp(speed * x))+loc)
        return a
        

    def f(self,x, speed, loc):
        return (x/10, loc*1/(1+exp(speed * x)))

    def priceSubstite(self, x, speed, begin, end):
        scale = 2 * end
        start = begin - scale/2
        
        return (x+2000, scale*1/(1+exp(-1* speed * x)) +start)

    def run_model(self, kwargs):
        """Method for running an instantiated model structure """
        
        loc = kwargs.pop("lookup shortage loc")
        speed = kwargs.pop("lookup shortage speed")
        kwargs['shortage price effect lookup'] =  [self.f(x/10, speed, loc) for x in range(0,100)]
        
        speed = kwargs.pop("lookup price substitute speed")
        begin = kwargs.pop("lookup price substitute begin")
        end = kwargs.pop("lookup price substitute end")
        kwargs['relative price substitute lookup'] = [self.priceSubstite(x, speed, begin, end) for x in range(0,100, 10)]
                
        scale = kwargs.pop("lookup returns to scale speed")
        speed = kwargs.pop("lookup returns to scale scale")
        kwargs['returns to scale lookup'] = [self.returnsToScale(x, speed, scale) for x in range(0, 101, 10)]
        
        scale = kwargs.pop("lookup approximated learning speed")
        speed = kwargs.pop("lookup approximated learning scale")
        start = kwargs.pop("lookup approximated learning start")
        kwargs['approximated learning effect lookup'] = [self.approxLearning(x, speed, scale, start) for x in range(0, 101, 10)]    
        
        super(ScarcityModel, self).run_model(kwargs)
      
        
def centroids(clusters):
    
    centrs = []
    for entry in clusters:
        target = entry.sample
        centroids.append(target)
        
    return centrs


def experiment_settings(results,runs):
#    Peel results and experiments
    experiments, outcomes = results
#    Only those experiments in which we are interested
    uncertain_values = experiments[runs]

#    Get out the names of the uncertain parameters
    new_outcomes = {}
    for key, value in outcomes.items():
        new_outcomes[key] = value[runs]
        results = experiments, new_outcomes
          
    uncertain_names = []
    for i in range(0,len(experiments[0])):
        name = experiments.dtype.descr[i][0]
        uncertain_names.append(name)
        
    return uncertain_values, uncertain_names


def behaviour_interest(results,VOI):
    outcomes = results[1]
    
    new_outcomes = {}
    for key, value in outcomes.items():
        new_outcomes[key] = value[runs]

    behaviour_int = new_outcomes['relative market price']
    
    return behaviour_int
    
    
def intervals(behaviour_int,filter):
    # If filter is true: filter slope and curvature.
    
    intervals = []
    for elem in behaviour_int:
        feature = ABP.ABP(elem,filter)
        interval = ABP.ABP_intervals(feature)
        intervals.append(interval)
        
    return intervals
        
        
def intervals_interest(intervals):   
    inds = []
    for interval in intervals:
        
        lens = []
        for elem in interval:
            length = elem[1]-elem[0]
            lens.append(length)
            
        ind = lens.index(max(lens))
        inds.append(ind)
            
    interest_intervals = []
    for i,ind in enumerate(inds):
        interest_intervals.append(intervals[i][ind])
        
    return interest_intervals
            
              
def run_interval(model,loop_index,interval,VOI,edges,ind_cons,
                 double_list,uncertain_names,uncertain_values):
    
    # Load the model.
    vensim.load_model(model)
    
    # We don't want any screens.
    vensim.be_quiet()
    
    # We run the model in game mode.
    step = vensim.get_val(r'TIME STEP')
    start_interval = str(interval[0]*step)
    venDLL.command('GAME>GAMEINTERVAL|'+start_interval)
    
    # Initiate the model to be run in game mode.
    venDLL.command("MENU>GAME")
       
    while True:
        if vensim.get_val(r'TIME')==2000:

            # Initiate the experiment of interest.
            # In other words set the uncertainties to the same value as in
            # those experiments.
            for i,value in enumerate(uncertain_values):
                name = uncertain_names[i]
                value = uncertain_values[i]
                vensim.set_value(name,value)
                
        print vensim.get_val(r'TIME')
 
        try:
            # Run the model for the length specified in game on-interval.
            venDLL.command('GAME>GAMEON')
            
            step = vensim.get_val(r'TIME STEP')
            if vensim.get_val(r'TIME')==(2000+step*interval[0]):
            
                if loop_index != 0:
                    # If loop elimination method is based on unique edge.
                    if loop_index-1 < ind_cons:
                        constant_value = vensim.get_val(edges[int(loop_index-1)][0])
                        vensim.set_value('value loop '+str(loop_index),constant_value)
                        vensim.set_value('switch loop '+str(loop_index),0)
        
                    # Else it is based on unique consecutive edges.
                    else:
                        constant_value = vensim.get_val(edges[int(loop_index-1)][0])
                        print constant_value
            
                        # Name of constant value used does not fit loop index minus 'start of cons'-index.
                        if loop_index-ind_cons in double_list:
                            vensim.set_value('value cons loop '+str(loop_index-ind_cons-1),constant_value)
                            vensim.set_value('switch cons loop '+str(loop_index-ind_cons-1),0)
                        else:
                            vensim.set_value('value cons loop '+str(loop_index-ind_cons),constant_value)
                            vensim.set_value('switch cons loop '+str(loop_index-ind_cons),0)
            
        except venDLL.VensimWarning:
            # The game on command will continue to the end of the simulation and
            # than raise a warning.
            print "The end of simulation."
            break         
    
    venDLL.finish_simulation()
    interval_series = vensim.get_data('Base.vdf',VOI)
    interval_series = interval_series[interval[0]:interval[1]]
    
    return interval_series


def dominance_distance(base,interval_series):
    
    distances = []
    
    # For every series with a loop switched off,
    # the distance to the base series is constructed.
    for series in interval_series:
        comp = np.array([base,
                         series])
        out = dg.distance_gonenc(comp)
        distance = out[0]
        distances.append(distance)
    
    dominant = distances.index(max(distances))
    
    return distances,dominant


def plot_intervals(behaviour,intervals):
    
    print(len(behaviour))
    x = range(0,len(behaviour))
    plt.plot(x,behaviour_int[0])
    plt.xlim(0,801)
    color = ['grey','lightgrey']
    
    for i,int in enumerate(intervals):
        no = np.mod(i,len(color))
        begin = int[0]
        end = int[1]
        plt.axvspan(begin,end, facecolor=color[no], alpha=0.5)
    
    plt.show()
    

if __name__ == "__main__":
    
#    CONSTRUCTING THE ENSEMBLE AND SAVING THE RESULTS
    EMAlogging.log_to_stderr(EMAlogging.DEBUG)
#    
#    model = ScarcityModel(r'D:\tbm-g367\workspace\EMA workbench\src\sandbox\sils',"scarcity")
#    
#    ensemble = ModelEnsemble()
#    ensemble.set_model_structure(model)
##    ensemble.parallel = True
#    results = ensemble.perform_experiments(1000)
#    save_results(results, r'base.cPickle')
    
    results = load_results(r'base.cPickle')
    
#    PLOTS FOR ENSEMBLE
#    fig = make_interactive_plot(results, outcomes=['relative market price'], type='lines')
#    fig = lines(results, outcomes = ['relative market price'], density=True, hist=False)
#    plt.show()
    
#    CONSTRUCTING THE CLUSTERS
#    dRow, clusters, z = clusterer.cluster(results, 
#                              outcome='relative market price',
#                              distance='gonenc',
#                              cMethod='maxclust', 
#                              cValue=20, 
#                              plotDendrogram=True,
#                              plotClusters=True,
#                              groupPlot=True)
#    save_results(clusters,r'clusters20.cPickle')

#    LOAD THE CLUSTERS
#    clusters = load_results(r'clusters.cPickle')

#    centrs = centroids(clusters)

#    GETTING OUT THOSE BEHAVIOURS AND EXPERIMENT SETTINGS
#    Indices of a number of examples, these will be looked at.
    runs = [526,781,911,988,10,780,740,943,573,991]
    
    uncertain_values, uncertain_names = experiment_settings(results,runs)
    behaviour_int = behaviour_interest(results,'relative market price')
    
    ints = intervals(behaviour_int,False)
#    intsFilt = intervals(behaviour_int,True)
    interest_intervals = intervals_interest(ints)
#    interest_intervalsFilt = intervals_interest(intsFilt)
    
#    plot_intervals(behaviour_int[0],ints[0])
#    plot_intervals(behaviour_int[0],intsFilt[0])

    unique_edges = [['In Goods', 'lost'],
                    ['loss unprofitable extraction capacity', 'decommissioning extraction capacity'],
                       ['production', 'In Goods'],
                    ['production', 'lost'],
                    ['production', 'Supply'],
                    ['Real Annual Demand', 'substitution losses'],
                    ['Real Annual Demand', 'price elasticity of demand losses'],
                    ['Real Annual Demand', 'desired extraction capacity'],
                    ['Real Annual Demand', 'economic demand growth'],
                    ['average recycling cost', 'relative market price'],
                    ['recycling fraction', 'lost'],
                    ['commissioning recycling capacity', 'Recycling Capacity Under Construction'],
                    ['maximum amount recyclable', 'recycling fraction'],
                    ['profitability recycling', 'planned recycling capacity'],
                    ['relative market price', 'price elasticity of demand losses'],
                    ['constrained desired recycling capacity', 'gap between desired and constrained recycling capacity'],
                    ['profitability extraction', 'planned extraction capacity'],
                    ['commissioning extraction capacity', 'Extraction Capacity Under Construction'],
                    ['desired recycling', 'gap between desired and constrained recycling capacity'],
                    ['Installed Recycling Capacity', 'decommissioning recycling capacity'],
                    ['Installed Recycling Capacity', 'loss unprofitable recycling capacity'],
                    ['average extraction costs', 'profitability extraction'],
                    ['average extraction costs', 'relative attractiveness recycling']]
    
    unique_cons_edges = [['recycling', 'recycling'],
                           ['recycling', 'supply demand ratio'],
                           ['decommissioning recycling capacity', 'recycling fraction'],
                           ['returns to scale', 'relative attractiveness recycling'],
                           ['shortage price effect', 'relative price last year'],
                           ['shortage price effect', 'profitability extraction'],
                           ['loss unprofitable extraction capacity', 'loss unprofitable extraction capacity'],
                           ['production', 'recycling fraction'],
                           ['production', 'constrained desired recycling capacity'],
                           ['production', 'new cumulatively recycled'],
                           ['production', 'effective fraction recycled of supplied'],
                           ['loss unprofitable recycling capacity', 'recycling fraction'],
                           ['average recycling cost', 'loss unprofitable recycling capacity'],
                           ['recycling fraction', 'new cumulatively recycled'],
                           ['substitution losses', 'supply demand ratio'],
                           ['Installed Extraction Capacity', 'Extraction Capacity Under Construction'],
                           ['Installed Extraction Capacity', 'commissioning extraction capacity'],
                           ['Installed Recycling Capacity', 'Recycling Capacity Under Construction'],
                           ['Installed Recycling Capacity', 'commissioning recycling capacity'],
                           ['average extraction costs', 'profitability extraction']]
    
    # THIS HAS TO DO WITH THE MODEL FORMULATION OF THE SWITCHES/VALUES
    double_list = [6,9,11,17,19]
    
    indCons = len(unique_edges)
    for elem in unique_cons_edges:
        unique_edges.append(elem)
    
    beh_no = 0 # Varies between 0 and 9, index style.
    interval = interest_intervals[beh_no]
    
    interval_series = []
    
    for loop_index in range(30,31):
#    for loop_index in range(1,len(unique_edges)+1):
    
        if loop_index-indCons > 0:
            model = ScarcityModel(r'D:\tbm-g367\workspace\EMA workbench\src\sandbox\sils\Models\Consecutive',"scarcity")
            model_location = r'D:\tbm-g367\workspace\EMA workbench\src\sandbox\sils\Models\Consecutive\Metals EMA.vpm'
        elif loop_index == 0:
            model = ScarcityModel(r'D:\tbm-g367\workspace\EMA workbench\src\sandbox\sils\Models\Base',"scarcity")
            model_location = r'D:\tbm-g367\workspace\EMA workbench\src\sandbox\sils\Models\Base\Metals EMA.vpm'
        else:
            model = ScarcityModel(r'D:\tbm-g367\workspace\EMA workbench\src\sandbox\sils\Models\Switches',"scarcity")
            model_location = r'D:\tbm-g367\workspace\EMA workbench\src\sandbox\sils\Models\Switches\Metals EMA.vpm'

        ensemble = ModelEnsemble()
        ensemble.set_model_structure(model)
    
        serie = run_interval(model_location,loop_index,
                              interval,'relative market price',
                              unique_edges,indCons,double_list,
                              uncertain_names,uncertain_values[beh_no])
        interval_series.append(serie)
    
    print(interval_series)
    base = behaviour_int[beh_no][interval[0]:interval[1]]
    print len(base),len(interval_series[0])
    distances, dominant = dominance_distance(behaviour_int[beh_no],interval_series)
    print distances, dominant
    
#    x = range(0,len(interval_series))
#    rmp = behaviour_int[beh_no]
#    rmp = rmp[interval[0]:interval[1]]
#    print len(rmp)
#    print len(interval_series)
##    plt.plot(x,interval_series)
#    plt.plot(x,interval_series,x,rmp)
#    plt.show()