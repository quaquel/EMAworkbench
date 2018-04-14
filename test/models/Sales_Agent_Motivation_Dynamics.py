"""
Python model "Sales_Agent_Motivation_Dynamics.py"
Translated using PySD version 0.8.3
"""
from __future__ import division
import numpy as np
from pysd import utils
import xarray as xr

from pysd.py_backend.functions import cache
from pysd.py_backend import functions

_subscript_dict = {}

_namespace = {
    'TIME': 'time',
    'Time': 'time',
    'Still Employed': 'still_employed',
    'Motivation Threshold': 'motivation_threshold',
    'Accumulating Income': 'accumulating_income',
    'Accumulating Sales': 'accumulating_sales',
    'Accumulating Tenure': 'accumulating_tenure',
    'Total Cumulative Income': 'total_cumulative_income',
    'Total Cumulative Sales': 'total_cumulative_sales',
    'Tenure': 'tenure',
    'Fraction of Effort for Sales': 'fraction_of_effort_for_sales',
    'Total Effort Available': 'total_effort_available',
    'Sales Effort Available': 'sales_effort_available',
    'Effort': 'effort',
    'Effort Required to Make a Sale': 'effort_required_to_make_a_sale',
    'Impact of Motivation on Effort': 'impact_of_motivation_on_effort',
    'Income': 'income',
    'Months of Expenses per Sale': 'months_of_expenses_per_sale',
    'Motivation': 'motivation',
    'Motivation Adjustment': 'motivation_adjustment',
    'Motivation Adjustment Time': 'motivation_adjustment_time',
    'Sales': 'sales',
    'Startup Subsidy': 'startup_subsidy',
    'Startup Subsidy Length': 'startup_subsidy_length',
    'Success Rate': 'success_rate',
    'FINAL TIME': 'final_time',
    'INITIAL TIME': 'initial_time',
    'SAVEPER': 'saveper',
    'TIME STEP': 'time_step'
}

__pysd_version__ = "0.8.3"


@cache('step')
def still_employed():
    """
    Still Employed

    Dmnl

    component


    """
    return functions.if_then_else(motivation() > motivation_threshold(), 1, 0)


@cache('run')
def motivation_threshold():
    """
    Motivation Threshold



    constant


    """
    return 0.1


@cache('step')
def accumulating_income():
    """
    Accumulating Income

    Month/Month

    component


    """
    return income()


@cache('step')
def accumulating_sales():
    """
    Accumulating Sales

    Persons/Month

    component


    """
    return sales()


@cache('step')
def accumulating_tenure():
    """
    Accumulating Tenure

    Months/Month

    component


    """
    return still_employed()


@cache('step')
def total_cumulative_income():
    """
    Total Cumulative Income

    Month

    component

    Express income in units of 'months of expenses'
    """
    return integ_total_cumulative_income()


@cache('step')
def total_cumulative_sales():
    """
    Total Cumulative Sales

    Persons

    component


    """
    return integ_total_cumulative_sales()


@cache('step')
def tenure():
    """
    Tenure

    Month

    component


    """
    return integ_tenure()


@cache('run')
def fraction_of_effort_for_sales():
    """
    Fraction of Effort for Sales

    Dmnl

    constant


    """
    return 0.25


@cache('run')
def total_effort_available():
    """
    Total Effort Available

    Hours/Month

    constant


    """
    return 200


@cache('step')
def sales_effort_available():
    """
    Sales Effort Available

    Hours/Month

    component


    """
    return functions.if_then_else(still_employed() > 0,
                                  total_effort_available() * fraction_of_effort_for_sales(), 0)


@cache('step')
def effort():
    """
    Effort

    Hours/Month

    component


    """
    return sales_effort_available() * impact_of_motivation_on_effort(motivation())


@cache('run')
def effort_required_to_make_a_sale():
    """
    Effort Required to Make a Sale

    Hours/Person

    constant


    """
    return 4


def impact_of_motivation_on_effort(x):
    """
    Impact of Motivation on Effort

    Dmnl

    lookup


    """
    return functions.lookup(
        x, [0, 0.285132, 0.448065, 0.570265, 0.733198, 0.95723, 1.4664, 3.19756, 4.03259],
        [0, 0.0616114, 0.232228, 0.492891, 0.772512, 0.862559, 0.914692, 0.952607, 0.957346])


@cache('step')
def income():
    """
    Income

    Dmnl

    component

    Technically in units of months of expenses earned per month
    """
    return months_of_expenses_per_sale() * sales() + functions.if_then_else(
        time() < startup_subsidy_length(), startup_subsidy(), 0)


@cache('run')
def months_of_expenses_per_sale():
    """
    Months of Expenses per Sale

    Month/Person

    constant


    """
    return 12 / 50


@cache('step')
def motivation():
    """
    Motivation

    Dmnl

    component


    """
    return integ_motivation()


@cache('step')
def motivation_adjustment():
    """
    Motivation Adjustment

    1/Month

    component


    """
    return (income() - motivation()) / motivation_adjustment_time()


@cache('run')
def motivation_adjustment_time():
    """
    Motivation Adjustment Time

    Month

    constant


    """
    return 3


@cache('step')
def sales():
    """
    Sales

    Persons/Month

    component


    """
    return effort() / effort_required_to_make_a_sale() * success_rate()


@cache('run')
def startup_subsidy():
    """
    Startup Subsidy

    Dmnl

    constant

    Months of expenses per month
    """
    return 0.5


@cache('run')
def startup_subsidy_length():
    """
    Startup Subsidy Length

    Month

    constant


    """
    return 6


@cache('run')
def success_rate():
    """
    Success Rate

    Dmnl

    constant


    """
    return 0.2


@cache('run')
def final_time():
    """
    FINAL TIME

    Month

    constant

    The final time for the simulation.
    """
    return 200


@cache('run')
def initial_time():
    """
    INITIAL TIME

    Month

    constant

    The initial time for the simulation.
    """
    return 0


@cache('step')
def saveper():
    """
    SAVEPER

    Month [0,?]

    component

    The frequency with which output is stored.
    """
    return time_step()


@cache('run')
def time_step():
    """
    TIME STEP

    Month [0,?]

    constant

    The time step for the simulation.
    """
    return 0.0625


integ_total_cumulative_income = functions.Integ(lambda: accumulating_income(), lambda: 0)

integ_total_cumulative_sales = functions.Integ(lambda: accumulating_sales(), lambda: 0)

integ_tenure = functions.Integ(lambda: accumulating_tenure(), lambda: 0)

integ_motivation = functions.Integ(lambda: motivation_adjustment(), lambda: 1)
