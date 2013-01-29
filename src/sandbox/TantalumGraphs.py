'''
Created on 15 mar. 2012

For use with the EMA of the tantalum model from the cooperation with HCSS
"Tantalum"
W.L. Auping, EMA group
Delft

In the __main__ routine, an interface is given to control all analyses.

@author: wlauping
'''
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import numpy as np
import sys
import functools
import time
import gc
import math
import multiprocessing as mp

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from expWorkbench import util, EMAlogging

#from analysis.graphs import envelopes, lines
from analysis.plotting import envelopes, lines
#from sandbox.plotting_refractor.plotting import lines as new_lines 
from analysis.graphs3d import envelopes3d
from analysis.interactive_graphs import make_interactive_plot
from analysis.prim import perform_prim, show_boxes_together, write_prim_to_stdout
from analysis.pairs_plotting import pairs_scatter, pairs_density
from matplotlib.pyplot import pie
#from sandbox.optimizationJan.model import ModelEnsemble

EMAlogging.log_to_stderr(level=EMAlogging.INFO)

print "loading results..."
file_name = r'..\..\cPickle\WILLEM\Tantalum\Tantalum1000policy_analysis.cPickle'
results = util.load_results(file_name)
print "results loaded."

outcomes = [
            'Global tantalum consumption',
            'Recycling Input Rate',
            'Part of original demand substituted',
#            'Real tantalum price',
            'Relative profits industrial tantalum mining High ore grade',
            'Relative profits industrial tantalum mining Medium ore grade',
            'Relative profits industrial tantalum mining Low ore grade',
            'Marginal tantalum costs at demand',
            'Relative part artisanal mining',
            'Relative part industrial mining',
            'Relative part coproduction tantalum',
           ]

#policies = [ 
#            {'name': 'EconGrowth'},
#            {'name': 'NoEconGrowth'},
#            ]
#policies = [
#            {'name': 'NoPolicies'},                                       # 0
#            {'name': 'NoIlligalMining'},                                  # 1
#            {'name': 'StrategicReserves'},                                # 2
#            {'name': 'RecyclingCollection'},                              # 3
#            {'name': 'RecyclingEfficiency'},                              # 4
#            {'name': 'TransparentMarket'},                                # 5
#            {'name': 'SubstitutionPromotion'},                            # 6
#            {'name': 'SubstitutionThreshold'},                            # 7
#            {'name': 'RecyclingPolicies'},                                # 8
#            {'name': 'SubstitutionPolicies'},                             # 9
#            {'name': 'StrategicRecycling'},                               # 10
#            {'name': 'EUproposal'},                                       # 11
#            {'name': 'EUplusTransparency'},                               # 12
#            {'name': 'EUminusSubstitution'},                              # 13
#            {'name': 'AllPolicies'},                                      # 14
#            ]

policies = [
            {'name': 'NoPolicies'},                                       # 0
#            {'name': 'NoIlligalMining'},                                  # 1
#            {'name': 'StrategicReserves'},                                # 2
#            {'name': 'TransparentMarket'},                                # 3
#            {'name': 'RecyclingCollection'},                              # 4
#            {'name': 'RecyclingEfficiency'},                              # 5
#            {'name': 'RecyclingPolicies'},                                # 6
#            {'name': 'SubstitutionPromotion'},                            # 7
#            {'name': 'SubstitutionThreshold'},                            # 8
#            {'name': 'SubstitutionPolicies'},                             # 9
            {'name': 'EUproposal'},                                       # 10
            {'name': 'EUplusTransparency'},                               # 11
            {'name': 'EUminusSubstitution'},                              # 12
            {'name': 'StrategicRecycling'},                               # 13
            {'name': 'AllPolicies'},                                      # 14
            ]

ylabels = {}
ylabels['Global tantalum consumption'] = 'Tantalum consumption $(lb/Year)$'
ylabels['Recycling Input Rate'] = r'Recycling Input Rate'
ylabels['Part of original demand substituted'] = r'Part demand substituted'
ylabels['Real tantalum price'] = 'Real tantalum price $(\$/lb)$'
ylabels['Relative profits industrial tantalum mining High ore grade'] = 'Relative profits high ore grade industrial mining'
ylabels['Relative profits industrial tantalum mining Medium ore grade'] = 'Relative profits medium ore grade industrial mining'
ylabels['Relative profits industrial tantalum mining Low ore grade'] = 'Relative profits low ore grade industrial mining'
ylabels['Marginal tantalum costs at demand'] = 'Marginal costs $(\$/lb)$'
ylabels['Relative part artisanal mining'] = 'Relative part artisanal mining'
ylabels['Relative part industrial mining'] = 'Relative part industrial mining'
ylabels['Relative part coproduction tantalum'] = 'Relative part coproduction tantalum'

short_outcomes = {}
short_outcomes['Global tantalum consumption'] = 'consumption'
short_outcomes['Recycling Input Rate'] = 'recycling'
short_outcomes['Part of original demand substituted'] = 'substitution'
short_outcomes['Real tantalum price'] = 'price'
short_outcomes['Relative profits industrial tantalum mining High ore grade'] = 'profits_high_ore_grade'
short_outcomes['Relative profits industrial tantalum mining Medium ore grade'] = 'profits_medium_ore_grade'
short_outcomes['Relative profits industrial tantalum mining Low ore grade'] = 'profits_low_ore_grade'
short_outcomes['Marginal tantalum costs at demand'] = 'costs'
short_outcomes['Relative part artisanal mining'] = 'artisanal_mining'
short_outcomes['Relative part industrial mining'] = 'industrial_mining'
short_outcomes['Relative part coproduction tantalum'] = 'coproduction'

direction_of_change = {}
direction_of_change['Global tantalum consumption'] = 1
direction_of_change['Recycling Input Rate'] = 1
direction_of_change['Part of original demand substituted'] = 1
direction_of_change['Real tantalum price'] = -1
direction_of_change['Relative profits industrial tantalum mining High ore grade'] = 1
direction_of_change['Relative profits industrial tantalum mining Medium ore grade'] = 1
direction_of_change['Relative profits industrial tantalum mining Low ore grade'] = 1
direction_of_change['Marginal tantalum costs at demand'] = -1
direction_of_change['Relative part artisanal mining'] = -1
direction_of_change['Relative part industrial mining'] = 1
direction_of_change['Relative part coproduction tantalum'] = -1

relative_outcomes = {}
relative_outcomes['Global tantalum consumption'] = False
relative_outcomes['Recycling Input Rate'] = True
relative_outcomes['Part of original demand substituted'] = True
relative_outcomes['Real tantalum price'] = False
relative_outcomes['Relative profits industrial tantalum mining High ore grade'] = True
relative_outcomes['Relative profits industrial tantalum mining Medium ore grade'] = True
relative_outcomes['Relative profits industrial tantalum mining Low ore grade'] = True
relative_outcomes['Marginal tantalum costs at demand'] = False
relative_outcomes['Relative part artisanal mining'] = True
relative_outcomes['Relative part industrial mining'] = True
relative_outcomes['Relative part coproduction tantalum'] = True

policy_strings = {}
policy_strings['NoPolicies'] = 'No policies'
policy_strings['NoIlligalMining'] = 'No illegal mining'
policy_strings['StrategicReserves'] = 'Strategic reserves'
policy_strings['TransparentMarket'] = 'Transparent market'
policy_strings['RecyclingCollection'] = 'Recycling collection'
policy_strings['RecyclingEfficiency'] = 'Recycling efficiency'
policy_strings['RecyclingPolicies'] = 'Recycling policies'
policy_strings['SubstitutionPromotion'] = 'Substitution promotion'
policy_strings['SubstitutionThreshold'] = 'Substitution threshold'
policy_strings['SubstitutionPolicies'] = 'Substitution policies'
policy_strings['EUproposal'] = 'EU proposal'
policy_strings['EUplusTransparency'] = 'EU plus transparency'
policy_strings['EUminusSubstitution'] = 'EU minus substitution'
policy_strings['StrategicRecycling'] = 'Strategic recycling'
policy_strings['AllPolicies'] = 'All policies'

hcss_colors_html = {}
hcss_colors_html['negative'] = '#C44700' # '#FF9400'
hcss_colors_html['mostly negative'] = '#E29265' # '#FFC473'
hcss_colors_html['neutral'] = '#BABBBC'
hcss_colors_html['mostly positive'] = '#7BC3FF'
hcss_colors_html['positive'] = '#0F94FF'
hcss_colors_html['blue I'] = '#0040B2'
hcss_colors_html['blue II'] = '#0F94FF'
hcss_colors_html[r'S&C gray'] = '#D0F800'

def automated_figure(results, outcome, grouping_specifiers, figure_type, ylabels, kde='kde', legend=True, save=True):
    policies_string, n, legend = nr_policies(grouping_specifiers)
    exp1, outcomes1 = results
    del outcomes1
    del n
    policies1 = set(exp1['policy'])
    nr_runs = len(exp1)
    nr_pols = len(policies1)
    nr_runs = int(nr_runs/nr_pols)
#    clear_mem()
    if figure_type == 'Line':
        lines(results,
              outcomes_to_show=[outcome], 
              group_by='policy', 
              grouping_specifiers=grouping_specifiers, 
              ylabels=ylabels, 
              density=kde, 
              titles=None,
              legend=legend, 
              )
#        save_figure(figure_type=figure_type, 
#                    outcome=outcome,
#                    policies_string = policies_string,
#                    nr_runs = nr_runs, 
#                    figure_save=save)
    elif figure_type == 'Env':
        envelopes(results, 
                  outcomes_to_show=[outcome], 
                  group_by='policy', 
                  grouping_specifiers=grouping_specifiers, 
                  ylabels=ylabels, 
                  fill=True, 
                  density=kde, 
                  titles=None, 
                  legend=legend, 
#                  categorieslabels, 
                  )
    save_figure(figure_type=figure_type, 
                outcome=outcome,
                policies_string = policies_string,
                nr_runs = nr_runs, 
                figure_save=save)
    print 'Figure generated and saved of type '+str(figure_type)+' for outcome '+str(outcome)

def HTML_color_to_RGB(colorstring):
    """ convert #RRGGBB to an (R, G, B) tuple 
    from http://code.activestate.com/recipes/266466-html-colors-tofrom-rgb-tuples/
    """
    colorstring = colorstring.strip()
    if colorstring[0] == '#': colorstring = colorstring[1:]
    if len(colorstring) != 6:
        raise ValueError, "input #%s is not in #RRGGBB format" % colorstring
    r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
    r, g, b = [int(n, 16) for n in (r, g, b)]
    return (r, g, b)

def automated_figures(results, outcomes_to_show, grouping_specifiers, figure_types, ylabels, kde='kde', legend=True, save=True):
    for outcome in outcomes_to_show:
        for figure_type in figure_types:
            print 'The outcome is '+str(outcome)+', the figure type is '+str(figure_type)
            automated_figure(results, outcome, grouping_specifiers, figure_type, ylabels, kde=kde, legend=legend, save=save)
            
def classify(outcomes):
    outcome = outcomes['Relative part artisanal mining']
    classes = np.zeros(outcome.shape[0])
    classes[np.max(outcome, axis=1)>0.25] = 1
    return classes

def classify2(data):
    # data
    result = data['Relative part artisanal mining']
#    maxPerRun = np.max(result, axis=1)
    classes = np.zeros(result.shape[0])
#    classes[(maxPerRun < 0.7)] = 1
    classes[np.max(result, axis=1)<1] = 1
    return classes

def classify_rms(data, indicator="", cutoff_value=1):
    result = data[indicator]
    classes =  np.zeros(result.shape[0])
    classes[(result > cutoff_value)] = 1
    return classes

def clear_mem(fig=None, axes=None, hard=False):
    if fig:
        plt.close(fig)
        fig=None
        del fig
    if axes:
        axes={}
        del axes
    plt.clf()
    gc.collect()
    plt.hold(False)
    if hard:
        print 'waiting for 5 seconds...'
        time.sleep(5)
    
def save_figure(figure_type, outcome, policies_string, nr_runs=1000, figure_save=True, dpi=300):
    if figure_save:
        print 'Saving the figure...'
        plt.savefig(r'..\..\cPickle\WILLEM\Tantalum\Figures\Ta'+figure_type+str(nr_runs)+' '+outcome+' '+policies_string+'.png', dpi=300)
        print 'Figure saved.'
    else:
        print 'Figure is not saved.'
        
def effect_policies(results, outcome, ref_pol, new_pol, desirable_direction=-1, classification=False):
    '''
    This function assesses the effects of policies, by looking at the number of cases
    in which the policy is positive or negative. This results in 5 categories:
        1 - Always negative effect
        2 - Mostly negative effect
        3 - Neutral in effect
        4 - Mostly positive effect
        5 - Always positive effect
        
    :param results: list of policies to be visualized
    :param ref_pol: the name of the reference policy.
    :param new_pol: the name of the (new) policy of interest
    :param desirable_direction: the direction of a desirable policy effect. Default is
                                -1, i.e. a lower value is desired.
    
    '''
    policy_effect_array=[0,0,0,0,0]
    # In this part, the difference between each reference run and the policy effect is calculated
    difference_pols = policy_difference(results, 
                                        outcome=outcome, 
                                        ref_pol=ref_pol, 
                                        new_pol=new_pol,
                                        classification=classification
                                        )
    # determine whether there are any negative values for the run

    if desirable_direction==-1:
        difference_pols = -1 * difference_pols

    a = np.zeros(difference_pols.shape)
    a[difference_pols>0]=1
    a = np.sum(a, axis=1)
    a[a>1] = 1
    b = np.zeros(difference_pols.shape)
    b[difference_pols<0]=1
    b = np.sum(b, axis=1)
    b[b>1] = 1
    sum_of_difference = np.sum(difference_pols, axis=1)

    for i in range(sum_of_difference.shape[0]):
        score = sum_of_difference[i]
        if score == 0:
            policy_effect_array[2] += 1
        elif score < 0:
            if a[i]==1:
                policy_effect_array[1] += 1
            else:
                policy_effect_array[0] += 1
        elif score > 0:
            if b[i]==1:
                policy_effect_array[3] += 1
            else:
                policy_effect_array[4] += 1
    return policy_effect_array

def figure_to_hcss_style(fig):
    '''
    Changes a figure to something like the HCSS house style
    
    '''
    
    # TODO: complete function
    font_family_headings = 'calibri'
    font_family_text = 'georgia'
    color = (0,0.49,0.769)
    list_color = list(color)
    inversed_color = [0, 0, 0]
    for i in range(len(color)):
        inversed_color[i] = 1 - list_color[i]
    inversed_color = tuple(inversed_color)
#    complementary_color = inverse_color(color)
#    params = {'legend.fontsize': 20,
#              'legend.family': 'georgia'
#              }
#    plt.rcParams.update(params)
    for ax in fig.get_axes():
        ax.legend.set_name('calibri')
        ax.title.set_family('calibri')
        ax.title.set_color(color)
    #    ax.xlabel.set_family('calibri')
    #    ax.ylabel.set_family('calibri')
    return fig

def hcss_colors(name):
    color_html = hcss_colors_html[name]
    color_rgb = HTML_color_to_RGB(color_html)
    return color_rgb

def get_subset(results, group_by, grouping_specifier):
    '''
    note, gaat mis als grouping specifier een interval is, dit moet nog
    elegant opgelost worden, zie rewrite plotter
    
    example of use::
    
    onderstaand neemt aan dat "policy" een correcte identifer is
    voor experiments en dat policy1 de juiste grouping specifier 
    is voor deze column in de experiments array.
    
    >> results = load_results( )
    >> group1 = get_subset(results, "policy", "policy1")
    >> group2 = get_subset(results, "policy", "policy2") 
    
    
    het is misschien beter om 2 van deze functies te maken, een met een
    nog fijnere control via een callable die de logical retuneerd
    dan kan er ook gegroepeerd worden op basis van bijv. uitkomsten
    
    '''
    experiments, outcomes = results
    logical = experiments[group_by]==grouping_specifier
    new_experiments = experiments[logical]
    
    new_outcomes= {}
    for key, value in outcomes.items():
        new_outcomes[key] = value[logical]
        
    new_results = new_experiments, new_outcomes
    return new_results



def inverse_color(color):
    list_color = list(color)
    inversed_color = [0, 0, 0]
    for i in range(len(color)):
        inversed_color[i] = 1 - list_color[i]
    inversed_color = tuple(inversed_color)
    return inversed_color

def nr_policies(policies_of_choice):
    n=0
    policies_string = ""
    legend = False
    for element in policies_of_choice:
        element = upper_case(element)
        if n > 0:
            policies_string = policies_string +'_'+ element
            legend = True
        else:
            policies_string = element
            legend = False
        n += 1
    return policies_string, n, legend

def string_to_upper_case(string):
    '''
    Convert a string to uppercase
    '''
    new_string = ''
    for x in string:
        x = x.upper()
        new_string = new_string+x
    string = new_string
    return string

def policy_difference(results, outcome, ref_pol, new_pol, classification=False, relative=False):
    ref_pol_set = get_subset(results, 'policy', ref_pol)
    new_pol_set = get_subset(results, 'policy', new_pol)
    groups = [new_pol_set, ref_pol_set]
    
    sorted_groups = sort_groups(groups, "Administration time")
    sorted_new_pol, sorted_ref_pol = sorted_groups
    
    if classification:
        exp1, out1 = sorted_ref_pol
        exp2, out2 = sorted_new_pol
        logical = classify(out1)==1
        temp_exp1 = exp1[logical]
        temp_exp2 = exp2[logical]
        temp_out1 = {}
        temp_out2 = {}
        for key, value in out1.iteritems():
            temp_out1[key]=value[logical]
        for key, value in out2.iteritems():
            temp_out2[key]=value[logical]
        sorted_ref_pol = temp_exp1, temp_out1
        sorted_new_pol = temp_exp2, temp_out2
        
    experiments_new_pol, outcomes_new_pol = sorted_new_pol
    experiments_ref_pol, outcomes_ref_pol = sorted_ref_pol
    
    difference_pols = outcomes_new_pol[outcome] - outcomes_ref_pol[outcome]
    if not relative_outcomes[outcome]:
        if relative:
            difference_pols = difference_pols/outcomes_ref_pol[outcome]
    return difference_pols

def policy_influence(results, outcome, ref_pol, policies, classification=False, desirable_direction=1, relative=True):
    '''
    This function looks for any outcome for the effect of all policies asked what the sum of squares is over each run.
    This allows camparing the influences the policies have on the outcome indicator.

    :param results: list of policies to be visualized.
    :param outcome: the outcome for which the influence is calcaluted.
    :param policies: a list of policies of which the influence is calculated.
    :param classification: only calculate the influence for classified runs, i.e. runs with undesirable effects
    :param desirable direction: the direction of desirable change considering an outcome, i.e. for price a lower price
                                (direction = -1) might be desirable.
    :param relative: the sum of squares is calculated relative to the original value and averaged over the number of data
                     points.
    '''
    policies = [policy.get('name') for policy in policies]
    exp1, out1 = results
    nr_runs = len(exp1)
    nr_pols = len(policies1)
    nr_runs = int(nr_runs/nr_pols)
    policies_without_ref = list(policies)     # creates a copy of the policies list
    policies_without_ref[0:1] = []            # delete the reference policy from the list
    policy_influence_dict={}
    for i in range(len(policies_without_ref)):
        new_pol = policies_without_ref[i]
        policy_name = policy_strings[new_pol]
        difference_pols = policy_difference(results, 
                                            outcome=outcome, 
                                            ref_pol=ref_pol, 
                                            new_pol=new_pol,
                                            classification=classification,
                                            relative=relative
                                            )
        if desirable_direction==-1:
            difference_pols = -1 * difference_pols
        a = np.zeros(difference_pols.shape)
        a[difference_pols>0]=1
        a = np.sum(a, axis=1)
        a[a>1] = 1
        b = np.zeros(difference_pols.shape)
        b[difference_pols<0]=1
        b = np.sum(b, axis=1)
        b[b>1] = 1
        SQR_difference = difference_pols**2
        sum_of_SQR = np.sum(SQR_difference)
        if relative:
            sum_of_SQR = sum_of_SQR/(len(SQR_difference)*nr_runs/100)
        if relative_outcomes[outcome]:
            sum_of_SQR = sum_of_SQR*100
        policy_influence_dict[policy_name] = sum_of_SQR
    return policy_influence_dict

def policy_influence_figure(results, outcome, ref_pol, policies, classification=False, relative=True, log=True, sorted_values=True, save=True):
    color = (0,0.49,0.769)
    policy_influence_dict = policy_influence(results, 
                                             outcome=outcome, 
                                             ref_pol='NoPolicies', 
                                             policies=policies, 
                                             classification=False,
                                             relative=relative)
    length_dict = len(policy_influence_dict)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.4
    positions = np.array(range(1,length_dict+1))
    if relative:
        xlabel = 'Relative policy influences'
        ylabel = 'Average relative change (%)'
    else:
        xlabel = 'Policy influences'
        ylabel = 'Average absolute change'
    if sorted_values:
        bar_values = []
        bar_keys = []
        for key, value in sorted(policy_influence_dict.iteritems(), key=lambda (k,v): (v,k)):
            bar_values.append(value)
            bar_keys.append(key)
        sorted_string = '_sorted'
    else:
        bar_values = policy_influence_dict.values()
        bar_keys = policy_influence_dict.keys()
        sorted_string = ''
    ax.bar(positions-width/2, bar_values, align='edge', color=color, width=width, log=log)
    ax.set_xticks(range(length_dict+2))
    xtick_labels = bar_keys
    xtick_labels[:0] = ' '
    ax.set_xticklabels(xtick_labels,
                       family='calibri', 
                       size='12', rotation=90)
    ax.set_ylabel(ylabel, family='calibri', size='14')
    ax.set_xlabel(xlabel, family='calibri', size='14')
    ax.set_title(outcome, color=color, family='calibri', size = '20')
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_family('calibri')
        tick.label.set_fontsize(12)
    fig.tight_layout()
    if log:
        log_string = '_log'
    else:
        log_string = ''
    if save:
        short_outcome = short_outcomes[outcome]
        plt.savefig(r'..\..\cPickle\WILLEM\Tantalum\Figures\TaPolInfluence_'+short_outcome+log_string+sorted_string+'.png', 
                    dpi=300)
    return fig

def policy_influence_per_case(results, outcome, ref_pol, new_pol, classification):
    pass

def policy_list_without_ref(policies, ref_pol):
    policies_without_ref = list(policies)     # creates a copy of the policies list
    policies_without_ref[0:1] = []            # delete the reference policy from the list
    return policies_without_ref

def regret_analysis(results, 
                    outcome, 
                    ref_pol, 
                    new_pol, 
                    figure_type='bar', 
                    desirable_direction=-1, 
                    relative_values=True, 
                    annotate=True):
#    color = (0,0.49,0.769)
    color = hcss_colors_html['blue I']
#    color2 = (0,0.25098,0.682353)
    color2 = hcss_colors_html['blue II']
#    complementary_color = inverse_color(color)
    complementary_color = (0.769, 0.278, 0)
    policy_name = policy_strings[new_pol]
    policy_effect_array = effect_policies(results, 
                                          outcome=outcome, 
                                          ref_pol='NoPolicies', 
                                          new_pol=new_pol,
                                          desirable_direction=desirable_direction,
                                          classification=False
                                          )
    # This selects only the results which displayed undesirable behaviour.
   
#    new_results_set = get_subset(results, 'policy', 'NoPolicies')
    policy_effect_array_interest = effect_policies(results, 
                                                   outcome=outcome, 
                                                   ref_pol='NoPolicies', 
                                                   new_pol=new_pol,
                                                   desirable_direction=desirable_direction,
                                                   classification=True
                                                   )

#    return policy_effect_array_vls, policy_effect_array_interest_vls
    # TODO: split the function here in the first, analytic part and a second plotting part
    # TODO:   the plotting part will have an option for either a bar chart (relative or not) 
    # TODO:   and a pie chart (always relative)
    policy_effect_array_vls = list(policy_effect_array)
    policy_effect_array_interest_vls = list(policy_effect_array_interest)

    labels = ['negative', 'mostly negative', 'neutral', 'mostly positive', 'positive']
    
    number_cases = sum(policy_effect_array)
    number_cases_interest = sum(policy_effect_array_interest)
    if figure_type=='bar':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        width = 0.4
        positions = np.array([1,2,3,4,5])
        xlabel = 'Policy effects'
        if relative_values:
            annotate = True
            ylabel = 'Relative proportion cases'
            for i in range(len(policy_effect_array)):
                policy_effect_array[i] = policy_effect_array[i]/number_cases
                policy_effect_array_interest[i] = policy_effect_array_interest[i]/number_cases_interest
        else:
            annotate = True
            ylabel = 'Number of cases'
        p1 = ax.bar(positions-width, policy_effect_array, align='edge', color=color, width=width)
        p2 = ax.bar(positions, policy_effect_array_interest, align='edge', color=complementary_color, width=width)
        ax.set_xticks([0,1,2,3,4,5,6])
        x_tick_labels = ['']
        x_tick_labels.append(labels)
        x_tick_labels.append('')
        ax.set_xticklabels(x_tick_labels, family='calibri', size='12')
        ax.set_ylabel(ylabel, family='calibri', size='14')
        ax.set_xlabel(xlabel, family='calibri', size='14')
        ax.set_title(outcome+': '+policy_name, color=color, family='calibri', size = '20')
        ax.legend( (p1[0], p2[0]), ('All scenarios', 'Undesirable scenarios'), loc=0, prop={'family':'calibri', 'size':12})
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_family('calibri')
            tick.label.set_fontsize(12)
        distance_annotation = max(policy_effect_array)/100
        if annotate:
            for i in range(len(policy_effect_array)):
                ax.annotate('('+str(policy_effect_array_vls[i])+')',
                            xy=(i+1-width/2,policy_effect_array[i]+distance_annotation),
                            horizontalalignment='center', family='calibri')
                ax.annotate('('+str(policy_effect_array_interest_vls[i])+')',
                            xy=(i+1+width/2,policy_effect_array_interest[i]+distance_annotation),
                            horizontalalignment='center', family='calibri')
    elif figure_type=='pie':
        figure_size = 8
#        rcParams['font.size'] = 12.0
        fig = plt.figure(figsize=(figure_size,figure_size))
        ax = fig.add_subplot(111)
#        colors=[(0.9,0,0), (0.9,0.4,0), (0.9,0.9,0), (0.5,0.9,0.2), (0,0.7,0)]
#        colors=[(1,0,0), (1,0.5,0), (1,1,0), (0.6,1,0.3), (0,0.8,0)]
        colors=[hcss_colors_html['negative'],
                hcss_colors_html['mostly negative'],
                hcss_colors_html['neutral'],
                hcss_colors_html['mostly positive'],
                hcss_colors_html['positive']
                ]
        relative_size = math.sqrt(number_cases_interest/number_cases)
        pctdistance=1-(1-relative_size)/2
#        inset_colors=[(1,0.1,0.1), (1,0.6,0.1), (1,1,0.1), (0.7,1,0.4), (0,0.9,0)]
        plt.pie(policy_effect_array, 
                explode=None, 
                labels=labels, 
                colors=colors, 
                autopct='%1.1f%%', 
                pctdistance=pctdistance,
                radius=1)
        ax.legend() #labels, family='calibri', fontsize=12
        pie_title = outcome+':\n'+policy_name
        pie_title = string_to_upper_case(pie_title)
        ax.set_title(pie_title, color=hcss_colors_html['blue II'], family='calibri', size='20')
        legend = ax.get_legend()
        legend.get_frame().set_alpha(0.5)
        legend_text = legend.get_texts()
        for entry in legend_text:
            entry.set_family('calibri')
            entry.set_fontsize(14)
            entry.set_weight('light')
        for entry in ax.texts:
            if entry.get_text().find("%") != -1:
                entry.set_color('white')
                entry.set_family('calibri')
                entry.set_fontsize(14)
                entry.set_weight('medium')
            else:
                entry.set_text("")
        ax.annotate('Undesirable scenarios',
                    xy=(relative_size/math.sqrt(2),relative_size/math.sqrt(2)),
                    xytext=(relative_size*0.7, relative_size),
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3",
                                    color=hcss_colors_html['blue I']),
#                    horizontalalignment='center', 
                    family='calibri',
                    color=hcss_colors_html['blue I'],
                    size=14)
        ax.annotate('All scenarios',
                    xy=(-1/math.sqrt(2),1/math.sqrt(2)),
                    xytext=(-0.9, 0.9),
                    arrowprops=dict(arrowstyle="-",
                                    connectionstyle="arc3",
                                    color=hcss_colors_html['blue I']),
#                    horizontalalignment='center', 
                    family='calibri',
                    color=hcss_colors_html['blue I'],
                    size=14)
        fig.tight_layout()
        # Now create a small pie chart inside the big pie chart
        # The heigth and width are relative to the relative size of the number of cases of interest
        height = figure_size*relative_size
        width = figure_size*relative_size

        inset = inset_axes(ax, width, height, loc=10,
                                )
        inset = plt.pie(policy_effect_array_interest, 
                        explode=None, 
                        colors=colors, 
                        autopct='%1.1f%%', 
                        radius=1,
                        shadow=False)
        for text in inset[2]:
            if text.get_text().find("%") != -1:
                text.set_color('white')
                text.set_family('calibri')
                text.set_fontsize(14)
                text.set_weight('medium')
        print 'plot saved'
    else:
        print 'No figuretype defined. Try either bar or pie'
    
    return fig

def regret_analysis_first_part(results, outcome, ref_pol, new_pol, desirable_direction=-1, relative_values=True, annotate=True):
    policy_effect_array = effect_policies(results, 
                                          outcome=outcome, 
                                          ref_pol='NoPolicies', 
                                          new_pol=new_pol,
                                          desirable_direction=desirable_direction,
                                          classification=False
                                          )
    # This selects only the results which displayed undesirable behaviour.
    policy_effect_array_interest = effect_policies(results, 
                                                   outcome=outcome, 
                                                   ref_pol='NoPolicies', 
                                                   new_pol=policy,
                                                   desirable_direction=desirable_direction,
                                                   classification=True
                                                   )

    return policy_effect_array, policy_effect_array_interest

def regret_analysis_second_part(policy_effect_array, policy_effect_array_interest):
    color = (0,0.49,0.769)
#    complementary_color = inverse_color(color)
    complementary_color = (0.769, 0.278, 0)
    policy_name = policy_strings[policy]
    policy_effect_array_vls = list(policy_effect_array)
    policy_effect_array_interest_vls = list(policy_effect_array_interest)
    figure = plt.figure()
    ax = figure.add_subplot(111)
    width = 0.4
    positions = np.array([1,2,3,4,5])
    xlabel = 'Policy effects'
    if relative_values:
        annotate = True
        ylabel = 'Relative proportion cases'
        number_cases = sum(policy_effect_array)
        number_cases_interest = sum(policy_effect_array_interest)
        for i in range(len(policy_effect_array)):
            policy_effect_array[i] = policy_effect_array[i]/number_cases
            policy_effect_array_interest[i] = policy_effect_array_interest[i]/number_cases_interest
    else:
        annotate = True
        ylabel = 'Number of cases'
    p1 = ax.bar(positions-width, policy_effect_array, align='edge', color=color, width=width)
    p2 = ax.bar(positions, policy_effect_array_interest, align='edge', color=complementary_color, width=width)
    ax.set_xticks([0,1,2,3,4,5,6])
    ax.set_xticklabels(["", 
                        "negative", 
                        "mostly negative", 
                        "neutral", 
                        "mostly positive", 
                        "positive", 
                        ""], family='calibri', size='12')
    ax.set_ylabel(ylabel, family='calibri', size='14')
    ax.set_xlabel(xlabel, family='calibri', size='14')
    ax.set_title(outcome+': '+policy_name, color=color, family='calibri', size = '20')
    ax.legend( (p1[0], p2[0]), ('All scenarios', 'Undesirable scenarios'), loc=0, prop={'family':'calibri', 'size':12})
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_family('calibri')
        tick.label.set_fontsize(12)
    distance_annotation = max(policy_effect_array)/100
    if annotate:
        for i in range(len(policy_effect_array)):
            ax.annotate('('+str(policy_effect_array_vls[i])+')',
                        xy=(i+1-width/2,policy_effect_array[i]+distance_annotation),
                        horizontalalignment='center', family='calibri')
            ax.annotate('('+str(policy_effect_array_interest_vls[i])+')',
                        xy=(i+1+width/2,policy_effect_array_interest[i]+distance_annotation),
                        horizontalalignment='center', family='calibri')
    return figure
    
def sort_groups(groups, sort_by):
    """
    
    :params groups: iterable of groups, as returned by get_subset
    :params sort_by: the name of the column in the experiments array to sort by.
    
    ..note:: to guarantee unique sorting, the sort by should be a continuous variable. 
             Moreover, works only in case of LHS sampling
    
    """
    
    indices = np.argsort(groups[0][0][sort_by])  
    
    temp_groups = []
    for entry in groups:
        experiments, results = entry
        experiments = experiments[indices]
        
        for key, value in results.items():
            results[key] = value[indices]
        temp_groups.append((experiments, results))
    return temp_groups

def select_policies(policies, ref_pol=0, max_pol=6):
    '''
    This function is used to split the list of specified policies to analyze
    in visualizable amounts.
    
    :param policies: list of policies to be visualized
    :param ref_pol: the index of the reference policy in the list.
                    Default = 0
    :param max_pol: the maximum of policies to visualize in one graph. Should be a value
                    over 1. Default = 6
    
    '''
    # Only get the names out of the policies
    policies = [policy.get('name') for policy in policies]
    # Strip the policies of the reference run
    policies_without_ref = list(policies)                   # creates a copy of the policies list
    policies_without_ref[ref_pol:ref_pol+1] = []            # delete the reference policy from the list
    # Determine size of policy array
    nr_sets_float = len(policies_without_ref)/(max_pol-1)   # start calculating the number of sets necessary
    nr_sets = len(policies_without_ref)//(max_pol-1)
    if (nr_sets_float > nr_sets):
        nr_sets += 1                                        # nr_sets is now the number of sets necessary
    nr_per_set_float = len(policies_without_ref)/nr_sets    # start calculating the number of policies per set
    nr_per_set = len(policies_without_ref)//nr_sets
    if (nr_per_set_float > nr_per_set):
        nr_per_set += 1 
    index_help = 0
    start_indicator = 0
    end_indicator = 0
    policy_sets = [['']*nr_per_set]*nr_sets # np.zeros([nr_sets,nr_per_set+1]) # The policy array is set up.
    for i in range(nr_sets):
        start_indicator = index_help
        end_indicator = index_help+nr_per_set
        index_list = [policies[ref_pol]]
        for element in policies_without_ref[start_indicator:end_indicator]:
            index_list.append(element)
        policy_sets[[i][0]] = index_list
        index_help += nr_per_set
    return policy_sets

def upper_case(string):
    '''
    This def takes the upper case characters from a string and returns them. Probably, there is a built in function
    which does exactly the same. 
    '''
    string_uppercase = ''
    for letter in string:
        if letter.isupper():
            if len(string_uppercase) > 0:
                string_uppercase += letter
            else:
                string_uppercase = letter
    return string_uppercase
    
def worker(results, outcome, grouping_specifiers, figure_types, ylabels, kde, legend, figure_save):
    print 'worker started'
    for figure_type in figure_types:
        print 'The outcome is '+str(outcome)+', the figure type is '+str(figure_type)
        automated_figure(results, outcome, grouping_specifiers, figure_type, ylabels, kde=kde, legend=legend, save=figure_save)

def y_limits(outcome_of_interest):
    log_space = False
    if outcome_of_interest == 'Global tantalum consumption':
        ymin = 0.0
        ymax = 25000000.0 # 25000000.0 (global consumption)
    elif (outcome_of_interest == 'Relative profits industrial tantalum mining High ore grade' or
          outcome_of_interest == 'Relative profits industrial tantalum mining Medium ore grade' or
          outcome_of_interest == 'Relative profits industrial tantalum mining Low ore grade'):
        ymin = -1.0
        ymax = 1.0
    elif (outcome_of_interest == 'Real tantalum price' or
          outcome_of_interest == 'Marginal tantalum costs at demand'):
        ymin = 0.0
        ymax = 20000.0
        log_space = True
    else:
        ymin = 0.0
        ymax = 1.0
    return ymin, ymax, log_space

if __name__ =='__main__':
    EMAlogging.log_to_stderr(EMAlogging.INFO)                # DEFAULT_LEVEL of INFO
#    mp.log_to_stderr(EMAlogging.DEBUG)
    ## Interface --------------------------------------------------------------------
    analysis = True            # When analysis is False, no analysis is performed, but tests are possible
    automate = False
    kde='kde'
    max_pol=4                   # the maximum number of policies per figure. Should be at least 2
    figure_type = 'Regret'      # Line Env Interactive 3D Pairs Regret Influence none
    outcome_of_interest = 'Global tantalum consumption'
    prim = False
    prim_normal = True
    figure_save = True
    plot_show = False
    # Only for Regret:
    annotate = True
    # Only for influence:
    sorted_values = True
    ## End of interface -------------------------------------------------------------
    exp1, outcomes1 = results
    policies1 = set(exp1['policy'])
    nr_runs = len(exp1)
    nr_pols = len(policies1)
    nr_runs = int(nr_runs/nr_pols)
    if not figure_type=='Regret' or not figure_type=='Influence':
        policy_set = select_policies(policies, max_pol=max_pol)
    ymin, ymax, log_space = y_limits(outcome_of_interest)
    if analysis:
        if figure_type=='Regret':
            pie = True
            policies = [policy.get('name') for policy in policies]
            # Strip the policies of the reference run
            policies_without_ref = list(policies)     # creates a copy of the policies list
            policies_without_ref[0:1] = []            # delete the reference policy from the list
            outcome=outcome_of_interest
            desirable_direction = direction_of_change[outcome]
            relativity = [True, False]
            if pie==True:
                figure='pie'
                relative_values = True
                figure_addition='Pie'
                relativity = [True]
            else:
                figure = 'bar'
                figure_addition = 'Bar'
            for relative_values in relativity:
                for policy in policies_without_ref:
                    fig = regret_analysis(results, 
                                          outcome=outcome, 
                                          ref_pol='NoPolicies', 
                                          new_pol=policy, 
                                          figure_type=figure,
                                          desirable_direction=desirable_direction, 
                                          relative_values=relative_values, 
                                          annotate=annotate
                                          )
                    if figure_save:
                        if relative_values:
                            rel_part = '_Perc'
                        else:
                            rel_part = ''
                        short_outcome = short_outcomes[outcome]
                    plt.savefig(r'..\..\cPickle\WILLEM\Tantalum\Figures\TaPolEffect'+figure_addition+'_'+short_outcome+'_'+policy+rel_part+'.png', 
                                dpi=600)
        elif figure_type=='Influence':
            print 'Graph type: ', figure_type
            for outcome in outcomes:
                print outcome
                figure = policy_influence_figure(results, 
                                                 outcome=outcome, 
                                                 ref_pol='NoPolicies', 
                                                 policies=policies, 
                                                 classification=False, 
                                                 relative=True, 
                                                 log=False, 
                                                 sorted_values=True, 
                                                 save=figure_save)
        else:                      
            for i in range(len(policy_set)):
                policies_of_choice = policy_set[i]
                policiesString = ""
                ### Figure generation process
                if automate:
            #        mp.log_to_stderr(EMAlogging.DEBUG)
                    print 'Automatic generation of figures...'
                    jobs = []
                    figure_save = True
                    figure_types = ['Line', 'Env'] # 'Line',
                    policies_string, n, legend = nr_policies(policies_of_choice)
                    grouping_specifiers=policies_of_choice
                    save = figure_save
                        
                    for outcome in outcomes:
                        print 'Automated process started, Outcome is now: '+str(outcome)
                        print 'Initiating a process...'
                        proc=mp.Process(target=worker, args=(results, outcome,grouping_specifiers, figure_types, ylabels, kde, legend, figure_save))
                        jobs.append(proc)
            #            proc.daemon=True
                        print 'process initiated'
                        proc.start()
                        print 'process started'
                        proc.join()
                        print 'process stopped'
                    print 'Finished'
                else:
                    print 'The outcome of interest is '+str(outcome_of_interest)+', ymin is: '+str(ymin)+', ymax is: '+str(ymax)
                    ### Analysis
                    ##   Comparing effects of policies
                    if prim:
                        if prim_normal:
                            boxes = perform_prim(results,classify=classify, mass_min=0.05, threshold=0.8)
                            show_boxes_together(boxes, results)
                            write_prim_to_stdout(boxes)
                        else:
                            EconGrowth_set = get_subset(results, 'policy', 'EconGrowth')
                            NoEconGrowth_set = get_subset(results, 'policy', 'NoEconGrowth')
                            groups = [EconGrowth_set, NoEconGrowth_set]
                            
                            sorted_groups = sort_groups(groups, "Administration time")
                            sorted_EG, sorted_noEG = sorted_groups
                            
                            experiments_EG, outcomes_EG = sorted_EG
                            experiments_noEG, outcomes_noEG = sorted_noEG
                            
                            Difference_EG_noEG = outcomes_EG['Relative part artisanal mining'] - outcomes_noEG['Relative part artisanal mining']
                            SQR_Diff = Difference_EG_noEG**2
                            SumOfSquares = np.sum(SQR_Diff, axis=1)
                            print SumOfSquares
                            print 'Median is '+str(np.median(SumOfSquares))
                            print 'Average is '+str(np.average(SumOfSquares))
                            log_SumOfSquares = np.log(SumOfSquares)
                            fig1 = plt.hist(log_SumOfSquares, log=False)
                            plt.savefig(r'..\..\cPickle\WILLEM\Tantalum\Figures\TaAnalysis MSQRT EconGrowth ArtisanalMining', dpi=300)
                            results_RMS = experiments_EG, {"EconGrowth":SumOfSquares}
                            
                            classify_rms = functools.partial(classify_rms, indicator="EconGrowth", cutoff_value=np.median(SumOfSquares))
                            boxes = perform_prim(results_RMS, classify=classify_rms, mass_min=0.05, threshold=0.8)
                            show_boxes_together(boxes, results_RMS)
                            write_prim_to_stdout(boxes)
                    ### Figures
                    policies_string, n, legend = nr_policies(policies_of_choice)
                    if figure_type == 'Line':
                        fig1 = lines(results,
                                     outcomes_to_show=[outcome_of_interest], #outcomes_to_show 
                                     group_by='policy',  #group_by
                                     grouping_specifiers=policies_of_choice, 
                                     ylabels=ylabels, 
                                     density='kde', 
                                     titles=None,
                                     legend=legend, 
                                     )
                        save_figure(figure_type=figure_type, 
                                    outcome=outcome_of_interest,
                                    policies_string=policies_string,
                                    nr_runs = nr_runs, 
                                    figure_save=figure_save)
            #        elif figure_type == 'new_line':
            #            new_lines(results, 
            #                      outcomes_to_show=[outcome_of_interest], 
            #    #                  group_by='policy', 
            #    #                  grouping_specifiers, 
            #                      density='kde', 
            #                      results_to_show=np.random.randint(0,results[0].shape[0], (100,)),
            #                      )
                    elif figure_type == 'Env':
                        print "Legend is: "+str(legend)
                        fig1 = envelopes(results, 
                                         outcomes_to_show=[outcome_of_interest], 
                                         group_by='policy', 
                                         grouping_specifiers=policies_of_choice, 
                                         ylabels=ylabels, 
                                         fill=True, 
                                         density='kde', 
                                         titles=None, 
                                         legend=legend, 
                #                         categorieslabels, 
                                         )
                        save_figure(figure_type=figure_type, 
                                    outcome=outcome_of_interest,
                                    policies_string=policies_string,
                                    nr_runs = nr_runs, 
                                    figure_save=figure_save)
                    elif figure_type == 'Interactive':
                        exp1, outcomes1 = results
                        if nr_runs < 1001:
                            make_interactive_plot(results,
                                                  outcomes=[outcome_of_interest]
                                                  )
                        else:
                            print 'Too many data points: '+str(len(exp1))
                    elif figure_type == '3D':
                        if n == 1 or n == 0:
                            fig1 = envelopes3d(results, 
                                               outcome=outcome_of_interest,
                                               policy = policies_of_choice,
                                               logSpace = log_space,
                #                               ymin = ymin,
                #                               ymax = ymax
                                               )
                        else:
                            print 'Too many policies defined, max is 1. Nr of policies defined is ', n
                    elif figure_type == 'Pairs':
                        print policies_of_choice
                        outcomes_to_show = outcomes
                        fig1 = pairs_scatter(results=results,
                                             outcomes_to_show=outcomes, 
                                             group_by='policy', 
                                             grouping_specifiers=policies_of_choice, 
                                             ylabels=ylabels, 
                                             legend=legend, 
        #                                     results_to_show, 
                                             point_in_time=-1
                                             )
                        save_figure(figure_type=figure_type,
                                    outcome='Multiplot',
                                    policies_string=policies_string,
                                    nr_runs = nr_runs,
                                    figure_save=figure_save)
                    else:
                        print 'No figure defined'
    else:
        # This part of the script can be used for performing tests on (new) scripts
        
        # TODO: make a pie chart of the bar chart for positive, negative and neutral policy effects.
#        plt.pie(x, explode, labels, colors, autopct, pctdistance, shadow, labeldistance, hold)
        desirable_direction = direction_of_change[outcome_of_interest]
        relative_values = True
        fig = regret_analysis(results, 
                              outcome=outcome_of_interest, 
                              ref_pol='NoPolicies', 
                              new_pol='EUproposal', 
                              figure_type='pie',
                              desirable_direction=desirable_direction, 
                              relative_values=relative_values, 
                              annotate=annotate
                              )
    if plot_show:
        plt.show()
    print 'Finished'