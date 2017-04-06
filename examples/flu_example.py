'''
Created on 20 dec. 2010

This file illustrated the use of the workbench for a model 
specified in Python itself. The example is based on `Pruyt & Hamarat <http://www.systemdynamics.org/conferences/2010/proceed/papers/P1253.pdf>`_.
For comparison, run both this model and the flu_vensim_no_policy_example.py and 
compare the results. 


.. codeauthor:: jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>
                chamarat <c.hamarat  (at) tudelft (dot) nl>

'''
from __future__ import (absolute_import, print_function, division, 
                        unicode_literals)

import numpy as np
from numpy import sin, min
from scipy import exp
import matplotlib.pyplot as plt

from ema_workbench import (Model, RealParameter, TimeSeriesOutcome, 
                           perform_experiments, ema_logging)
                         
from ema_workbench.analysis.plotting import lines
from ema_workbench.analysis.plotting_util import KDE                         
from ema_workbench.em_framework.evaluators import MultiprocessingEvaluator


#==============================================================================
#
#    the model itself
#
#==============================================================================

FINAL_TIME = 48
INITIAL_TIME = 0
TIME_STEP = 0.0078125

switch_regions = 1.0
switch_immunity = 1.0
switch_deaths = 1.0
switch_immunity_cap = 1.0

def LookupFunctionX(variable,start,end,step,skew,growth,v=0.5):
    return start + ((end-start)/((1+skew*exp(-growth*(variable-step)))**(1/v)))

def flu_model(x11=0,x12=0,x21=0,x22=0,x31=0,x32=0,x41=0,x51=0,x52=0,x61=0,
                x62=0,x81=0,x82=0,x91=0,x92=0,x101=0,x102=0): 
    #Assigning initial values
    additional_seasonal_immune_population_fraction_R1 = float(x11)
    additional_seasonal_immune_population_fraction_R2 = float(x12)
    
    fatality_rate_region_1 = float(x21)
    fatality_rate_region_2 = float(x22)
    
    initial_immune_fraction_of_the_population_of_region_1 = float(x31)
    initial_immune_fraction_of_the_population_of_region_2 = float(x32)
    
    normal_interregional_contact_rate = float(x41) 
    interregional_contact_rate  = switch_regions*normal_interregional_contact_rate
    
    permanent_immune_population_fraction_R1 = float(x51)
    permanent_immune_population_fraction_R2 = float(x52)
    
    recovery_time_region_1 = float(x61) 
    recovery_time_region_2 = float(x62) 
    
    susceptible_to_immune_population_delay_time_region_1 = 1
    susceptible_to_immune_population_delay_time_region_2 = 1
    
    root_contact_rate_region_1 = float(x81)
    root_contact_rate_region_2 = float(x82)
        
    infection_rate_region_1 = float(x91)
    infection_rate_region_2 = float(x92) 
    
    normal_contact_rate_region_1 = float(x101) 
    normal_contact_rate_region_2 = float(x102) 
    
    ######
    susceptible_to_immune_population_flow_region_1 = 0.0
    susceptible_to_immune_population_flow_region_2 = 0.0
    ######
    
    initial_value_population_region_1 = 6.0*10**8
    initial_value_population_region_2 = 3.0*10**9
    
    initial_value_infected_population_region_1 = 10.0
    initial_value_infected_population_region_2 = 10.0
    
    initial_value_immune_population_region_1 = switch_immunity * initial_immune_fraction_of_the_population_of_region_1 * initial_value_population_region_1
    initial_value_immune_population_region_2 = switch_immunity * initial_immune_fraction_of_the_population_of_region_2 * initial_value_population_region_2
        
    initial_value_susceptible_population_region_1 = initial_value_population_region_1 - initial_value_immune_population_region_1
    initial_value_susceptible_population_region_2 = initial_value_population_region_2 - initial_value_immune_population_region_2
    
    recovered_population_region_1 = 0.0
    recovered_population_region_2 = 0.0
    
    infected_population_region_1 = initial_value_infected_population_region_1
    infected_population_region_2 = initial_value_infected_population_region_2 
    
    susceptible_population_region_1 = initial_value_susceptible_population_region_1 
    susceptible_population_region_2 = initial_value_susceptible_population_region_2
    
    immune_population_region_1 = initial_value_immune_population_region_1
    immune_population_region_2 = initial_value_immune_population_region_2
    
    deceased_population_region_1 = [0.0]
    deceased_population_region_2 = [0.0]
    runTime = [INITIAL_TIME]

    # --End of Initialization--
    
     
    Max_infected = 0

    for time in range(int(INITIAL_TIME/TIME_STEP), int(FINAL_TIME/TIME_STEP)):
        runTime.append(runTime[-1]+TIME_STEP)
        total_population_region_1 = infected_population_region_1 + recovered_population_region_1 + susceptible_population_region_1 + immune_population_region_1
        total_population_region_2 = infected_population_region_2 + recovered_population_region_2 + susceptible_population_region_2 + immune_population_region_2

        infected_population_region_1 = max(0, infected_population_region_1)
        infected_population_region_2 = max(0, infected_population_region_2)

        infected_fraction_region_1 = infected_population_region_1/total_population_region_1
        infected_fraction_region_2 = infected_population_region_2/total_population_region_2
        
        impact_infected_population_on_contact_rate_region_1 = 1-(infected_fraction_region_1**(1/root_contact_rate_region_1))
        impact_infected_population_on_contact_rate_region_2 = 1-(infected_fraction_region_2**(1/root_contact_rate_region_2))
    
#        if ((time*TIME_STEP) >= 4) and ((time*TIME_STEP)<=10):
#            normal_contact_rate_region_1 = float(x101)*(1 - 0.5)
#        else:normal_contact_rate_region_1 = float(x101) 
        
        normal_contact_rate_region_1 = float(x101)*(1 - LookupFunctionX(infected_fraction_region_1, 0, 1, 0.15, 0.75, 15))  
                
        contact_rate_region_1 = normal_contact_rate_region_1*impact_infected_population_on_contact_rate_region_1
        contact_rate_region_2 = normal_contact_rate_region_2*impact_infected_population_on_contact_rate_region_2
        
        recoveries_region_1 =(1-(fatality_rate_region_1*switch_deaths))*infected_population_region_1/recovery_time_region_1
        recoveries_region_2 =(1-(fatality_rate_region_2*switch_deaths))*infected_population_region_2/recovery_time_region_2
        
        flu_deaths_region_1 = fatality_rate_region_1*switch_deaths*infected_population_region_1/recovery_time_region_1
        flu_deaths_region_2 = fatality_rate_region_2*switch_deaths*infected_population_region_2/recovery_time_region_2
        
        infections_region_1 = (susceptible_population_region_1*contact_rate_region_1*infection_rate_region_1*infected_fraction_region_1)+ (susceptible_population_region_1*interregional_contact_rate*infection_rate_region_1*infected_fraction_region_2)
        infections_region_2 = (susceptible_population_region_2*contact_rate_region_2*infection_rate_region_2*infected_fraction_region_2)+ (susceptible_population_region_2*interregional_contact_rate*infection_rate_region_2*infected_fraction_region_1)
            
        infected_population_region_1_NEXT = infected_population_region_1 + (TIME_STEP*(infections_region_1 - flu_deaths_region_1 - recoveries_region_1))
        infected_population_region_2_NEXT = infected_population_region_2 + (TIME_STEP*(infections_region_2 - flu_deaths_region_2 - recoveries_region_2))
    
        if infected_population_region_1_NEXT < 0 or infected_population_region_2_NEXT < 0:
            pass
    
        recovered_population_region_1_NEXT = recovered_population_region_1 + (TIME_STEP*recoveries_region_1)
        recovered_population_region_2_NEXT = recovered_population_region_2 + (TIME_STEP*recoveries_region_2)
        
        if fatality_rate_region_1 >= 0.025:
            qw = 1.0
        elif fatality_rate_region_1 >= 0.01:
            qw = 0.8
        elif fatality_rate_region_1 >= 0.001:
            qw = 0.6
        elif fatality_rate_region_1 >= 0.0001:
            qw = 0.4
        else: qw = 0.2

        if (time*TIME_STEP) <= 10:
            normal_immune_population_fraction_region_1 = (additional_seasonal_immune_population_fraction_R1/2)*sin(4.5+(time*TIME_STEP/2))  + (((2*permanent_immune_population_fraction_R1) + additional_seasonal_immune_population_fraction_R1)/2) 
        else: normal_immune_population_fraction_region_1 = max((float(qw),(additional_seasonal_immune_population_fraction_R1/2)*sin(4.5+(time*TIME_STEP/2))  + (((2*permanent_immune_population_fraction_R1) + additional_seasonal_immune_population_fraction_R1)/2)))

        normal_immune_population_fraction_region_2 = switch_immunity_cap*min((sin((time*TIME_STEP/2)+1.5)*additional_seasonal_immune_population_fraction_R2/2)+(((2*permanent_immune_population_fraction_R2)+additional_seasonal_immune_population_fraction_R2)/2),(permanent_immune_population_fraction_R1+additional_seasonal_immune_population_fraction_R1))+((1-switch_immunity_cap)*((sin((time*TIME_STEP/2)+1.5)*additional_seasonal_immune_population_fraction_R2/2)+(((2*permanent_immune_population_fraction_R2)+additional_seasonal_immune_population_fraction_R2)/2))) 
    
        normal_immune_population_region_1 = normal_immune_population_fraction_region_1*total_population_region_1
        normal_immune_population_region_2 = normal_immune_population_fraction_region_2*total_population_region_2
                
        if switch_immunity == 1:
            susminreg1_1 = ((normal_immune_population_region_1-immune_population_region_1)/susceptible_to_immune_population_delay_time_region_1)
            susminreg1_2 = (susceptible_population_region_1/susceptible_to_immune_population_delay_time_region_1)
            susmaxreg1 = -(immune_population_region_1/susceptible_to_immune_population_delay_time_region_1)
            if (susmaxreg1 >= susminreg1_1) or (susmaxreg1 >= susminreg1_2):
                susceptible_to_immune_population_flow_region_1 = susmaxreg1
            elif (susminreg1_1 < susminreg1_2) and (susminreg1_1 > susmaxreg1):
                susceptible_to_immune_population_flow_region_1 = susminreg1_1
            elif (susminreg1_2 < susminreg1_1) and (susminreg1_2 > susmaxreg1):
                susceptible_to_immune_population_flow_region_1 = susminreg1_2
        else: susceptible_to_immune_population_flow_region_1 = 0
        
        if switch_immunity == 1:
            susminreg2_1 = ((normal_immune_population_region_2-immune_population_region_2)/susceptible_to_immune_population_delay_time_region_2)
            susminreg2_2 = (susceptible_population_region_2/susceptible_to_immune_population_delay_time_region_2)
            susmaxreg2 = -(immune_population_region_2/susceptible_to_immune_population_delay_time_region_2) 
            if (susmaxreg2 >= susminreg2_1) or (susmaxreg2 >= susminreg2_2):
                susceptible_to_immune_population_flow_region_2 = susmaxreg2
            elif (susminreg2_1 < susminreg2_2) and (susminreg2_1 > susmaxreg2):
                susceptible_to_immune_population_flow_region_2 = susminreg2_1
            elif (susminreg2_2 < susminreg2_1) and (susminreg2_2 > susmaxreg2):
                susceptible_to_immune_population_flow_region_2 = susminreg2_2
        else: susceptible_to_immune_population_flow_region_2 = 0
        
        susceptible_population_region_1_NEXT = susceptible_population_region_1 - (TIME_STEP*(infections_region_1 + susceptible_to_immune_population_flow_region_1))
        susceptible_population_region_2_NEXT = susceptible_population_region_2 - (TIME_STEP*(infections_region_2 + susceptible_to_immune_population_flow_region_2))
        
        immune_population_region_1_NEXT = immune_population_region_1 + (TIME_STEP*susceptible_to_immune_population_flow_region_1)
        immune_population_region_2_NEXT = immune_population_region_2 + (TIME_STEP*susceptible_to_immune_population_flow_region_2)
        
        deceased_population_region_1_NEXT = deceased_population_region_1[-1] + (TIME_STEP*flu_deaths_region_1)
        deceased_population_region_2_NEXT = deceased_population_region_2[-1] + (TIME_STEP*flu_deaths_region_2)
        
        #Updating integral values  
        if Max_infected < (infected_population_region_1_NEXT/(infected_population_region_1_NEXT+recovered_population_region_1_NEXT+susceptible_population_region_1_NEXT+immune_population_region_1_NEXT)):
            Max_infected = (infected_population_region_1_NEXT/(infected_population_region_1_NEXT+recovered_population_region_1_NEXT+susceptible_population_region_1_NEXT+immune_population_region_1_NEXT))
        
        recovered_population_region_1 = recovered_population_region_1_NEXT
        recovered_population_region_2 = recovered_population_region_2_NEXT
        
        infected_population_region_1 = infected_population_region_1_NEXT
        infected_population_region_2 = infected_population_region_2_NEXT
        
        susceptible_population_region_1 = susceptible_population_region_1_NEXT
        susceptible_population_region_2 = susceptible_population_region_2_NEXT
    
        immune_population_region_1 = immune_population_region_1_NEXT
        immune_population_region_2 = immune_population_region_2_NEXT
    
        deceased_population_region_1.append(deceased_population_region_1_NEXT)
        deceased_population_region_2.append(deceased_population_region_2_NEXT)
        
        #End of main code
    
    
    return {"TIME":runTime,
            "deceased_population_region_1":deceased_population_region_1} 


if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    model = Model('mexicanFlu', function=flu_model)
    model.uncertainties = [RealParameter('x11', 0, 0.5),
                           RealParameter('x12', 0, 0.5),
                           RealParameter('x21', 0.0001, 0.1),
                           RealParameter('x22', 0.0001, 0.1),
                           RealParameter('x31', 0, 0.5),
                           RealParameter('x32', 0, 0.5),
                           RealParameter('x41', 0, 0.9),
                           RealParameter('x51', 0, 0.5),
                           RealParameter('x52', 0, 0.5),
                           RealParameter('x61', 0, 0.8),
                           RealParameter('x62', 0, 0.8),
                           RealParameter('x81', 1, 10),
                           RealParameter('x82', 1, 10),
                           RealParameter('x91', 0, 0.1),
                           RealParameter('x92', 0, 0.1),
                           RealParameter('x101', 0, 200),
                           RealParameter('x102', 0, 200)] 
    
    model.outcomes = [TimeSeriesOutcome("TIME"),
                      TimeSeriesOutcome("deceased_population_region_1")]
    
    nr_experiments = 500
    
    results = perform_experiments(model, 10)
    
#     with MultiprocessingPoolEvaluator(model) as evaluator:
#         results = perform_experiments(model, nr_experiments,
#                                       evaluator=evaluator)
 
    lines(results, outcomes_to_show="deceased_population_region_1", 
          show_envelope=True, density=KDE, titles=None, 
          experiments_to_show=np.arange(0, nr_experiments, 10)
          )
    plt.show()  