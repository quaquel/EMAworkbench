
"""
Python model /Domain/tudelft.net/Users/jhkwakkel/EMAworkbench/src/test/test_connectors/../models/Sales_Agent_Motivation_Dynamics.py
Translated using PySD version 0.6.3
"""
from __future__ import division
import numpy as np
from pysd import utils
import xarray as xr

from pysd.functions import cache
from pysd import functions

_subscript_dict = {}

_namespace = {
    'Months of Expenses per Sale': 'months_of_expenses_per_sale',
    'Effort Required to Make a Sale': 'effort_required_to_make_a_sale',
    'Accumulating Income': 'accumulating_income',
    'TIME STEP': 'time_step',
    'Total Cumulative Income': 'total_cumulative_income',
    'Sales': 'sales',
    'SAVEPER': 'saveper',
    'Fraction of Effort for Sales': 'fraction_of_effort_for_sales',
    'Income': 'income',
    'INITIAL TIME': 'initial_time',
    'Impact of Motivation on Effort': 'impact_of_motivation_on_effort',
    'Total Effort Available': 'total_effort_available',
    'Sales Effort Available': 'sales_effort_available',
    'FINAL TIME': 'final_time',
    'Effort': 'effort',
    'Startup Subsidy Length': 'startup_subsidy_length',
    'Motivation Adjustment': 'motivation_adjustment',
    'Tenure': 'tenure',
    'Success Rate': 'success_rate',
    'Motivation': 'motivation',
    'Total Cumulative Sales': 'total_cumulative_sales',
    'Startup Subsidy': 'startup_subsidy',
    'Accumulating Tenure': 'accumulating_tenure',
    'Motivation Threshold': 'motivation_threshold',
    'Still Employed': 'still_employed',
    'Motivation Adjustment Time': 'motivation_adjustment_time',
    'Accumulating Sales': 'accumulating_sales'}


def _init_tenure():
    """
    Implicit
    --------
    (_init_tenure)
    See docs for tenure
    Provides initial conditions for tenure function
    """
    return 0


@cache('step')
def motivation():
    """
    Motivation
    ----------
    (motivation)
    Dmnl

    """
    return _state['motivation']


@cache('run')
def final_time():
    """
    FINAL TIME
    ----------
    (final_time)
    Month
    The final time for the simulation.
    """
    return 200


@cache('step')
def total_cumulative_sales():
    """
    Total Cumulative Sales
    ----------------------
    (total_cumulative_sales)
    Persons

    """
    return _state['total_cumulative_sales']


@cache('run')
def startup_subsidy_length():
    """
    Startup Subsidy Length
    ----------------------
    (startup_subsidy_length)
    Month

    """
    return 6


@cache('run')
def startup_subsidy():
    """
    Startup Subsidy
    ---------------
    (startup_subsidy)
    Dmnl
    Months of expenses per month
    """
    return 0.5


@cache('run')
def motivation_adjustment_time():
    """
    Motivation Adjustment Time
    --------------------------
    (motivation_adjustment_time)
    Month

    """
    return 3


@cache('run')
def time_step():
    """
    TIME STEP
    ---------
    (time_step)
    Month [0,?]
    The time step for the simulation.
    """
    return 0.0625


@cache('step')
def motivation_adjustment():
    """
    Motivation Adjustment
    ---------------------
    (motivation_adjustment)
    1/Month

    """
    return (income() - motivation()) / motivation_adjustment_time()


@cache('run')
def fraction_of_effort_for_sales():
    """
    Fraction of Effort for Sales
    ----------------------------
    (fraction_of_effort_for_sales)
    Dmnl

    """
    return 0.25


@cache('step')
def accumulating_income():
    """
    Accumulating Income
    -------------------
    (accumulating_income)
    Month/Month

    """
    return income()


def _init_total_cumulative_income():
    """
    Implicit
    --------
    (_init_total_cumulative_income)
    See docs for total_cumulative_income
    Provides initial conditions for total_cumulative_income function
    """
    return 0


@cache('run')
def effort_required_to_make_a_sale():
    """
    Effort Required to Make a Sale
    ------------------------------
    (effort_required_to_make_a_sale)
    Hours/Person

    """
    return 4


@cache('run')
def success_rate():
    """
    Success Rate
    ------------
    (success_rate)
    Dmnl

    """
    return 0.2


@cache('step')
def sales_effort_available():
    """
    Sales Effort Available
    ----------------------
    (sales_effort_available)
    Hours/Month

    """
    return functions.if_then_else(
        still_employed() > 0,
        total_effort_available() *
        fraction_of_effort_for_sales(),
        0)


@cache('step')
def income():
    """
    Income
    ------
    (income)
    Dmnl
    Technically in units of months of expenses earned per month
    """
    return months_of_expenses_per_sale() * sales() + functions.if_then_else(time() <
                                                                            startup_subsidy_length(), startup_subsidy(), 0)


@cache('run')
def motivation_threshold():
    """
    Motivation Threshold
    --------------------
    (motivation_threshold)


    """
    return 0.1


def impact_of_motivation_on_effort(x):
    """
    Impact of Motivation on Effort
    ------------------------------
    (impact_of_motivation_on_effort)
    Dmnl

    """
    return functions.lookup(x, [0, 0.285132, 0.448065, 0.570265, 0.733198, 0.95723, 1.4664, 3.19756, 4.03259], [
                            0, 0.0616114, 0.232228, 0.492891, 0.772512, 0.862559, 0.914692, 0.952607, 0.957346])


def _init_motivation():
    """
    Implicit
    --------
    (_init_motivation)
    See docs for motivation
    Provides initial conditions for motivation function
    """
    return 1


@cache('step')
def time():
    """
    Time
    ----
    (time)
    None
    The time of the model
    """
    return _t


@cache('run')
def months_of_expenses_per_sale():
    """
    Months of Expenses per Sale
    ---------------------------
    (months_of_expenses_per_sale)
    Month/Person

    """
    return 12 / 50


@cache('step')
def sales():
    """
    Sales
    -----
    (sales)
    Persons/Month

    """
    return effort() / effort_required_to_make_a_sale() * success_rate()


@cache('step')
def accumulating_tenure():
    """
    Accumulating Tenure
    -------------------
    (accumulating_tenure)
    Months/Month

    """
    return still_employed()


@cache('step')
def _dtenure_dt():
    """
    Implicit
    --------
    (_dtenure_dt)
    See docs for tenure
    Provides derivative for tenure function
    """
    return accumulating_tenure()


@cache('step')
def still_employed():
    """
    Still Employed
    --------------
    (still_employed)
    Dmnl

    """
    return functions.if_then_else(motivation() > motivation_threshold(), 1, 0)


@cache('step')
def _dmotivation_dt():
    """
    Implicit
    --------
    (_dmotivation_dt)
    See docs for motivation
    Provides derivative for motivation function
    """
    return motivation_adjustment()


@cache('step')
def effort():
    """
    Effort
    ------
    (effort)
    Hours/Month

    """
    return sales_effort_available() * impact_of_motivation_on_effort(motivation())


@cache('run')
def total_effort_available():
    """
    Total Effort Available
    ----------------------
    (total_effort_available)
    Hours/Month

    """
    return 200


def _init_total_cumulative_sales():
    """
    Implicit
    --------
    (_init_total_cumulative_sales)
    See docs for total_cumulative_sales
    Provides initial conditions for total_cumulative_sales function
    """
    return 0


@cache('step')
def _dtotal_cumulative_sales_dt():
    """
    Implicit
    --------
    (_dtotal_cumulative_sales_dt)
    See docs for total_cumulative_sales
    Provides derivative for total_cumulative_sales function
    """
    return accumulating_sales()


@cache('step')
def _dtotal_cumulative_income_dt():
    """
    Implicit
    --------
    (_dtotal_cumulative_income_dt)
    See docs for total_cumulative_income
    Provides derivative for total_cumulative_income function
    """
    return accumulating_income()


@cache('step')
def total_cumulative_income():
    """
    Total Cumulative Income
    -----------------------
    (total_cumulative_income)
    Month
    Express income in units of 'months of expenses'
    """
    return _state['total_cumulative_income']


@cache('step')
def saveper():
    """
    SAVEPER
    -------
    (saveper)
    Month [0,?]
    The frequency with which output is stored.
    """
    return time_step()


@cache('step')
def tenure():
    """
    Tenure
    ------
    (tenure)
    Month

    """
    return _state['tenure']


@cache('step')
def accumulating_sales():
    """
    Accumulating Sales
    ------------------
    (accumulating_sales)
    Persons/Month

    """
    return sales()


@cache('run')
def initial_time():
    """
    INITIAL TIME
    ------------
    (initial_time)
    Month
    The initial time for the simulation.
    """
    return 0


def time():
    return _t
functions.time = time
functions.initial_time = initial_time
