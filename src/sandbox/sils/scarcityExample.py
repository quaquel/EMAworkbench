'''
Created in June 2012

@author: Bas M.J. Keijser
         Jan H. Kwakkel & Erik Pruyt (model class function)
'''
from __future__ import division
import numpy as np
from math import exp
import copy

import expWorkbench.ema_logging as ema_logging
#from expWorkbench import ModelEnsemble
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


def experiment_settings(results,runs, VOI):
#    Peel results and experiments
    experiments, outcomes = results

    new_outcomes = {}
    new_outcomes[VOI] = outcomes[VOI][runs]
    new_experiments = experiments[runs]
    results = new_experiments, new_outcomes
    
    return results

def experiments_to_cases(experiments):
    '''
    
    This function transform a structured experiments array into a list
    of case dicts. This can then for example be used as an argument for 
    running :meth:`~model.SimpleModelEnsemble.perform_experiments`.
    
    :param experiments: a structured array containing experiments
    :return: a list of case dicts.
    
    '''
    #get the names of the uncertainties
    uncertainties = [entry[0] for entry in experiments.dtype.descr]
    
    #remove policy and model, leaving only the case related uncertainties
    try:
        uncertainties.pop(uncertainties.index('policy'))
        uncertainties.pop(uncertainties.index('model'))
    except:
        pass
    
    #make list of of tuples of tuples
    cases = []
    for i in range(experiments.shape[0]):
        a = experiments[i]
        case = []
        for uncertainty in uncertainties:
            entry = (uncertainty, a[uncertainty])
            case.append(entry)
        cases.append(tuple(case))
  
    #cast back to list of dicts
    tempcases = []
    for case in cases:
        tempCase = {}
        for entry in case:
            tempCase[entry[0]] = entry[1]
        tempcases.append(tempCase)
    cases = tempcases
    
    return cases
    
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

def set_lookups(case):
    kwargs = case

    def returnsToScale(x, speed, scale):
        return (x*1000, scale*1/(1+exp(-1* speed * (x-50))))
     
    def approxLearning(x, speed, scale, start):
        x = x-start
        loc = 1 - scale
        a = (x*10000, scale*1/(1+exp(speed * x))+loc)
        return a

    def f(x, speed, loc):
        return (x/10, loc*1/(1+exp(speed * x)))

    def priceSubstite(x, speed, begin, end):
        scale = 2 * end
        start = begin - scale/2
        return (x+2000, scale*1/(1+exp(-1* speed * x)) +start)
    
    loc = kwargs.pop("lookup shortage loc")
    speed = kwargs.pop("lookup shortage speed")
    lookup = [f(x/10, speed, loc) for x in range(0,100)]
    vensim.set_value('shortage price effect lookup', lookup )
    
#    print venDLL.get_varattrib('shortage price effect lookup', attribute=3)
#    print lookup
    
    speed = kwargs.pop("lookup price substitute speed")
    begin = kwargs.pop("lookup price substitute begin")
    end = kwargs.pop("lookup price substitute end")
    vensim.set_value('relative price substitute lookup', [priceSubstite(x, speed, begin, end) for x in range(0,100, 10)])
        
    scale = kwargs.pop("lookup returns to scale speed")
    speed = kwargs.pop("lookup returns to scale scale")
    vensim.set_value('returns to scale lookup', [returnsToScale(x, speed, scale) for x in range(0, 101, 10)])
    
    scale = kwargs.pop("lookup approximated learning speed")
    speed = kwargs.pop("lookup approximated learning scale")
    start = kwargs.pop("lookup approximated learning start")
    vensim.set_value('approximated learning effect lookup', [approxLearning(x, speed, scale, start) for x in range(0, 101, 10)])
            
              
def run_interval(model,loop_index,interval,VOI,edges,ind_cons,
                 double_list,case):
    
    # Load the model.
    vensim.load_model(model)
    
    case = copy.deepcopy(case)
    set_lookups(case)
    
    for key,value in case.items():
        vensim.set_value(key,repr(value))
#        print key, repr(value), vensim.get_val(key), value-vensim.get_val(key)

    # We run the model in game mode.
    step = vensim.get_val(r'TIME STEP')
    start_interval = interval[0]*step
    end_interval = interval[1]*step
    venDLL.command('GAME>GAMEINTERVAL|'+str(start_interval))

    # Initiate the model to be run in game mode.
    venDLL.command("MENU>GAME")
    if start_interval > 0:
        venDLL.command('GAME>GAMEON')

    loop_on = 1
    loop_off = 0

    loop_turned_off = False
    while True:

        # Initiate the experiment of interest.
        # In other words set the uncertainties to the same value as in
        # those experiments.
        time = vensim.get_val(r'TIME')
        ema_logging.debug(time)
        
        if time ==(2000+step*interval[0]) and not loop_turned_off:
            loop_turned_off = True
            
            if loop_index != 0:
                
                # If loop elimination method is based on unique edge.
                if loop_index-1 < ind_cons:
                    constant_value = vensim.get_val(edges[int(loop_index-1)][0])
                    
                    if loop_off==1:
                        constant_value = 0
                    
                    vensim.set_value('value loop '+str(loop_index),
                                     constant_value)
                    vensim.set_value('switch loop '+str(loop_index),
                                     loop_off)
        
                # Else it is based on unique consecutive edges.
                else:
                    constant_value = vensim.get_val(edges[int(loop_index-1)][0])
                    
                    if loop_off==1:
                        constant_value = 0
                    
                    # Name of constant value used does not fit loop index minus 'start of cons'-index.
                    if loop_index-ind_cons in double_list:
                        vensim.set_value('value cons loop '+str(loop_index-ind_cons-1),
                                         constant_value)
                        vensim.set_value('switch cons loop '+str(loop_index-ind_cons-1),
                                         loop_off)
                    else:
                        vensim.set_value('value cons loop '+str(loop_index-ind_cons),
                                         constant_value)
                        vensim.set_value('switch cons loop '+str(loop_index-ind_cons),
                                         loop_off)
                        
            venDLL.command('GAME>GAMEINTERVAL|'+str(end_interval-start_interval))
            
        elif time ==(2000+step*interval[1]) and loop_turned_off:
            loop_turned_off = False
            if loop_index != 0:
                # If loop elimination method is based on unique edge.
                if loop_index-1 < ind_cons:
                    constant_value = 0
                    vensim.set_value('value loop '+str(loop_index),
                                     constant_value)
                    vensim.set_value('switch loop '+str(loop_index),
                                     loop_on)
        
                # Else it is based on unique consecutive edges.
                else:
                    constant_value = 0
                    # Name of constant value used does not fit loop index minus 'start of cons'-index.
                    if loop_index-ind_cons in double_list:
                        vensim.set_value('value cons loop '+str(loop_index-ind_cons-1),
                                         constant_value)
                        vensim.set_value('switch cons loop '+str(loop_index-ind_cons-1),
                                         loop_on)
                    else:
                        vensim.set_value('value cons loop '+str(loop_index-ind_cons),
                                         constant_value)
                        vensim.set_value('switch cons loop '+str(loop_index-ind_cons),
                                         loop_on)
            
            finalT = vensim.get_val('FINAL TIME')
            currentT = vensim.get_val('TIME')
            venDLL.command('GAME>GAMEINTERVAL|'+str(finalT - currentT))
        
        else:
            break
        
        finalT = vensim.get_val('FINAL TIME')
        currentT = vensim.get_val('TIME')
        if finalT != currentT:
            venDLL.command('GAME>GAMEON')
    
    venDLL.command('GAME>ENDGAME')
    interval_series = vensim.get_data('Base.vdf',VOI)

    
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

import os

def perform_loop_knockout():    
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
    
#    CONSTRUCTING THE ENSEMBLE AND SAVING THE RESULTS
    ema_logging.log_to_stderr(ema_logging.INFO)
    results = load_results(r'base.cPickle')

#    GETTING OUT THOSE BEHAVIOURS AND EXPERIMENT SETTINGS
#    Indices of a number of examples, these will be looked at.
    runs = [526,781,911,988,10,780,740,943,573,991]
    VOI = 'relative market price'
    
    results_of_interest = experiment_settings(results,runs,VOI)
    cases_of_interest = experiments_to_cases(results_of_interest[0])
    behaviour_int = results_of_interest[1][VOI]
    
#    CONSTRUCTING INTERVALS OF ATOMIC BEHAVIOUR PATTERNS
    ints = intervals(behaviour_int,False)

#    GETTING OUT ONLY THOSE OF MAXIMUM LENGTH PER BEHAVIOUR
    max_intervals = intervals_interest(ints)
    
#    THIS HAS TO DO WITH THE MODEL FORMULATION OF THE SWITCHES/VALUES
    double_list = [6,9,11,17,19]
    
    indCons = len(unique_edges)
#    for elem in unique_cons_edges:
#        unique_edges.append(elem)
    
    current = os.getcwd()

    for beh_no in range(0,10):
#        beh_no = 0 # Varies between 0 and 9, index style.
        interval = max_intervals[beh_no]
    
        rmp = behaviour_int[beh_no]
    #    rmp = rmp[interval[0]:interval[1]]
        x = range(0,len(rmp))
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        vensim.be_quiet()
    #    for loop_index in range(7,8):
        for loop_index in range(1,len(unique_edges)+1):
    
            if loop_index-indCons > 0:
                model_location = current + r'\Models\Consecutive\Metals EMA.vpm'
            elif loop_index == 0:
                model_location = current + r'\Models\Base\Metals EMA.vpm'
            else:
                model_location = current + r'\Models\Switches\Metals EMA.vpm'
        
            serie = run_interval(model_location,loop_index,
                                  interval,'relative market price',
                                  unique_edges,indCons,double_list,
                                  cases_of_interest[beh_no])
            
            if serie.shape != rmp.shape:
                ema_logging.info('Loop %s created a floating point error' % (loop_index))
                ema_logging.info('Caused by trying to switch %s' % (unique_edges[loop_index-1]))
                
            if serie.shape == rmp.shape:
                ax.plot(x,serie,'b')
                
    #        data = np.zeros(rmp.shape[0])
    #        data[0:serie.shape[0]] = serie
    #        ax.plot(x,data,'b')
      
        ax.plot(x,rmp,'r')
        ax.axvspan(interval[0]-1,interval[1], facecolor='lightgrey', alpha=0.5)
        f_name = 'switched unique edges only'+str(beh_no)
        plt.savefig(f_name)
#        plt.show()

if __name__ == "__main__":
    perform_loop_knockout()