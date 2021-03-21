"""
Python model "Sales_Agent_Motivation_Dynamics.py"
Translated using PySD version 0.10.0
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

__pysd_version__ = "0.10.0"

__data = {'scope': None, 'time': lambda: 0}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def time():
    return __data['time']()


@cache('step')
def still_employed():
    """
    Real Name: b'Still Employed'
    Original Eqn: b'IF THEN ELSE(Motivation>Motivation Threshold, 1 , 0 )'
    Units: b'Dmnl'
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.if_then_else(motivation() > motivation_threshold(), 1, 0)


@cache('run')
def motivation_threshold():
    """
    Real Name: b'Motivation Threshold'
    Original Eqn: b'0.1'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 0.1


@cache('step')
def accumulating_income():
    """
    Real Name: b'Accumulating Income'
    Original Eqn: b'Income'
    Units: b'Month/Month'
    Limits: (None, None)
    Type: component

    b''
    """
    return income()


@cache('step')
def accumulating_sales():
    """
    Real Name: b'Accumulating Sales'
    Original Eqn: b'Sales'
    Units: b'Persons/Month'
    Limits: (None, None)
    Type: component

    b''
    """
    return sales()


@cache('step')
def accumulating_tenure():
    """
    Real Name: b'Accumulating Tenure'
    Original Eqn: b'Still Employed'
    Units: b'Months/Month'
    Limits: (None, None)
    Type: component

    b''
    """
    return still_employed()


@cache('step')
def total_cumulative_income():
    """
    Real Name: b'Total Cumulative Income'
    Original Eqn: b'INTEG ( Accumulating Income, 0)'
    Units: b'Month'
    Limits: (None, None)
    Type: component

    b"Express income in units of 'months of expenses'"
    """
    return _integ_total_cumulative_income()


@cache('step')
def total_cumulative_sales():
    """
    Real Name: b'Total Cumulative Sales'
    Original Eqn: b'INTEG ( Accumulating Sales, 0)'
    Units: b'Persons'
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_total_cumulative_sales()


@cache('step')
def tenure():
    """
    Real Name: b'Tenure'
    Original Eqn: b'INTEG ( Accumulating Tenure, 0)'
    Units: b'Month'
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_tenure()


@cache('run')
def fraction_of_effort_for_sales():
    """
    Real Name: b'Fraction of Effort for Sales'
    Original Eqn: b'0.25'
    Units: b'Dmnl'
    Limits: (None, None)
    Type: constant

    b''
    """
    return 0.25


@cache('run')
def total_effort_available():
    """
    Real Name: b'Total Effort Available'
    Original Eqn: b'200'
    Units: b'Hours/Month'
    Limits: (None, None)
    Type: constant

    b''
    """
    return 200


@cache('step')
def sales_effort_available():
    """
    Real Name: b'Sales Effort Available'
    Original Eqn: b'IF THEN ELSE(Still Employed > 0, Total Effort Available * Fraction of Effort for Sales\\\\ , 0 )'
    Units: b'Hours/Month'
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.if_then_else(still_employed() > 0,
                                  total_effort_available() * fraction_of_effort_for_sales(), 0)


@cache('step')
def effort():
    """
    Real Name: b'Effort'
    Original Eqn: b'Sales Effort Available * Impact of Motivation on Effort(Motivation)'
    Units: b'Hours/Month'
    Limits: (None, None)
    Type: component

    b''
    """
    return sales_effort_available() * impact_of_motivation_on_effort(motivation())


@cache('run')
def effort_required_to_make_a_sale():
    """
    Real Name: b'Effort Required to Make a Sale'
    Original Eqn: b'4'
    Units: b'Hours/Person'
    Limits: (None, None)
    Type: constant

    b''
    """
    return 4


def impact_of_motivation_on_effort(x):
    """
    Real Name: b'Impact of Motivation on Effort'
    Original Eqn: b'( [(0,0)-(10,1)],(0,0),(0.285132,0.0616114),(0.448065,0.232228),(0.570265,0.492891),(0.733198\\\\ ,0.772512),(0.95723,0.862559),(1.4664,0.914692),(3.19756,0.952607),(4.03259,0.957346\\\\ ))'
    Units: b'Dmnl'
    Limits: (None, None)
    Type: lookup

    b''
    """
    return functions.lookup(
        x, [0, 0.285132, 0.448065, 0.570265, 0.733198, 0.95723, 1.4664, 3.19756, 4.03259],
        [0, 0.0616114, 0.232228, 0.492891, 0.772512, 0.862559, 0.914692, 0.952607, 0.957346])


@cache('step')
def income():
    """
    Real Name: b'Income'
    Original Eqn: b'Months of Expenses per Sale * Sales + IF THEN ELSE(Time < Startup Subsidy Length, Startup Subsidy\\\\ , 0 )'
    Units: b'Dmnl'
    Limits: (None, None)
    Type: component

    b'Technically in units of months of expenses earned per month'
    """
    return months_of_expenses_per_sale() * sales() + functions.if_then_else(
        time() < startup_subsidy_length(), startup_subsidy(), 0)


@cache('run')
def months_of_expenses_per_sale():
    """
    Real Name: b'Months of Expenses per Sale'
    Original Eqn: b'12/50'
    Units: b'Month/Person'
    Limits: (None, None)
    Type: constant

    b''
    """
    return 12 / 50


@cache('step')
def motivation():
    """
    Real Name: b'Motivation'
    Original Eqn: b'INTEG ( Motivation Adjustment, 1)'
    Units: b'Dmnl'
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_motivation()


@cache('step')
def motivation_adjustment():
    """
    Real Name: b'Motivation Adjustment'
    Original Eqn: b'(Income - Motivation) / Motivation Adjustment Time'
    Units: b'1/Month'
    Limits: (None, None)
    Type: component

    b''
    """
    return (income() - motivation()) / motivation_adjustment_time()


@cache('run')
def motivation_adjustment_time():
    """
    Real Name: b'Motivation Adjustment Time'
    Original Eqn: b'3'
    Units: b'Month'
    Limits: (None, None)
    Type: constant

    b''
    """
    return 3


@cache('step')
def sales():
    """
    Real Name: b'Sales'
    Original Eqn: b'Effort / Effort Required to Make a Sale * Success Rate'
    Units: b'Persons/Month'
    Limits: (None, None)
    Type: component

    b''
    """
    return effort() / effort_required_to_make_a_sale() * success_rate()


@cache('run')
def startup_subsidy():
    """
    Real Name: b'Startup Subsidy'
    Original Eqn: b'0.5'
    Units: b'Dmnl'
    Limits: (None, None)
    Type: constant

    b'Months of expenses per month'
    """
    return 0.5


@cache('run')
def startup_subsidy_length():
    """
    Real Name: b'Startup Subsidy Length'
    Original Eqn: b'6'
    Units: b'Month'
    Limits: (None, None)
    Type: constant

    b''
    """
    return 6


@cache('run')
def success_rate():
    """
    Real Name: b'Success Rate'
    Original Eqn: b'0.2'
    Units: b'Dmnl'
    Limits: (None, None)
    Type: constant

    b''
    """
    return 0.2


@cache('run')
def final_time():
    """
    Real Name: b'FINAL TIME'
    Original Eqn: b'200'
    Units: b'Month'
    Limits: (None, None)
    Type: constant

    b'The final time for the simulation.'
    """
    return 200


@cache('run')
def initial_time():
    """
    Real Name: b'INITIAL TIME'
    Original Eqn: b'0'
    Units: b'Month'
    Limits: (None, None)
    Type: constant

    b'The initial time for the simulation.'
    """
    return 0


@cache('step')
def saveper():
    """
    Real Name: b'SAVEPER'
    Original Eqn: b'TIME STEP'
    Units: b'Month'
    Limits: (0.0, None)
    Type: component

    b'The frequency with which output is stored.'
    """
    return time_step()


@cache('run')
def time_step():
    """
    Real Name: b'TIME STEP'
    Original Eqn: b'0.0625'
    Units: b'Month'
    Limits: (0.0, None)
    Type: constant

    b'The time step for the simulation.'
    """
    return 0.0625


_integ_total_cumulative_income = functions.Integ(lambda: accumulating_income(), lambda: 0)

_integ_total_cumulative_sales = functions.Integ(lambda: accumulating_sales(), lambda: 0)

_integ_tenure = functions.Integ(lambda: accumulating_tenure(), lambda: 0)

_integ_motivation = functions.Integ(lambda: motivation_adjustment(), lambda: 1)
