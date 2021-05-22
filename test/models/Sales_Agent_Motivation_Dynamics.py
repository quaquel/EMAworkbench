"""
Python model "Sales_Agent_Motivation_Dynamics.py"
Translated using PySD version 1.3.0
"""
from os import path

from pysd.py_backend.functions import lookup, Integ, if_then_else
from pysd import cache

_subscript_dict = {}

_namespace = {
    "TIME": "time",
    "Time": "time",
    "Still Employed": "still_employed",
    "Motivation Threshold": "motivation_threshold",
    "Accumulating Income": "accumulating_income",
    "Accumulating Sales": "accumulating_sales",
    "Accumulating Tenure": "accumulating_tenure",
    "Total Cumulative Income": "total_cumulative_income",
    "Total Cumulative Sales": "total_cumulative_sales",
    "Tenure": "tenure",
    "Fraction of Effort for Sales": "fraction_of_effort_for_sales",
    "Total Effort Available": "total_effort_available",
    "Sales Effort Available": "sales_effort_available",
    "Effort": "effort",
    "Effort Required to Make a Sale": "effort_required_to_make_a_sale",
    "Impact of Motivation on Effort": "impact_of_motivation_on_effort",
    "Income": "income",
    "Months of Expenses per Sale": "months_of_expenses_per_sale",
    "Motivation": "motivation",
    "Motivation Adjustment": "motivation_adjustment",
    "Motivation Adjustment Time": "motivation_adjustment_time",
    "Sales": "sales",
    "Startup Subsidy": "startup_subsidy",
    "Startup Subsidy Length": "startup_subsidy_length",
    "Success Rate": "success_rate",
    "FINAL TIME": "final_time",
    "INITIAL TIME": "initial_time",
    "SAVEPER": "saveper",
    "TIME STEP": "time_step",
}

__pysd_version__ = "1.3.0"

__data = {"scope": None, "time": lambda: 0}

_root = path.dirname(__file__)


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def time():
    return __data["time"]()


@cache.step
def still_employed():
    """
    Real Name: Still Employed
    Original Eqn: IF THEN ELSE(Motivation>Motivation Threshold, 1 , 0 )
    Units: Dmnl
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return if_then_else(motivation() > motivation_threshold(), lambda: 1, lambda: 0)


@cache.run
def motivation_threshold():
    """
    Real Name: Motivation Threshold
    Original Eqn: 0.1
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 0.1


@cache.step
def accumulating_income():
    """
    Real Name: Accumulating Income
    Original Eqn: Income
    Units: Month/Month
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return income()


@cache.step
def accumulating_sales():
    """
    Real Name: Accumulating Sales
    Original Eqn: Sales
    Units: Persons/Month
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return sales()


@cache.step
def accumulating_tenure():
    """
    Real Name: Accumulating Tenure
    Original Eqn: Still Employed
    Units: Months/Month
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return still_employed()


@cache.step
def total_cumulative_income():
    """
    Real Name: Total Cumulative Income
    Original Eqn: INTEG ( Accumulating Income, 0)
    Units: Month
    Limits: (None, None)
    Type: component
    Subs: None

    Express income in units of 'months of expenses'
    """
    return _integ_total_cumulative_income()


@cache.step
def total_cumulative_sales():
    """
    Real Name: Total Cumulative Sales
    Original Eqn: INTEG ( Accumulating Sales, 0)
    Units: Persons
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_total_cumulative_sales()


@cache.step
def tenure():
    """
    Real Name: Tenure
    Original Eqn: INTEG ( Accumulating Tenure, 0)
    Units: Month
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_tenure()


@cache.run
def fraction_of_effort_for_sales():
    """
    Real Name: Fraction of Effort for Sales
    Original Eqn: 0.25
    Units: Dmnl
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 0.25


@cache.run
def total_effort_available():
    """
    Real Name: Total Effort Available
    Original Eqn: 200
    Units: Hours/Month
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 200


@cache.step
def sales_effort_available():
    """
    Real Name: Sales Effort Available
    Original Eqn: IF THEN ELSE(Still Employed > 0, Total Effort Available * Fraction of Effort for Sales, 0 )
    Units: Hours/Month
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return if_then_else(
        still_employed() > 0,
        lambda: total_effort_available() * fraction_of_effort_for_sales(),
        lambda: 0,
    )


@cache.step
def effort():
    """
    Real Name: Effort
    Original Eqn: Sales Effort Available * Impact of Motivation on Effort(Motivation)
    Units: Hours/Month
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return sales_effort_available() * impact_of_motivation_on_effort(motivation())


@cache.run
def effort_required_to_make_a_sale():
    """
    Real Name: Effort Required to Make a Sale
    Original Eqn: 4
    Units: Hours/Person
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 4


def impact_of_motivation_on_effort(x):
    """
    Real Name: Impact of Motivation on Effort
    Original Eqn: ( [(0,0)-(10,1)],(0,0),(0.285132,0.0616114),(0.448065,0.232228),(0.570265,0.492891),(0.733198,0.772512),(0.95723,0.862559),(1.4664,0.914692),(3.19756,0.952607),(4.03259,0.957346))
    Units: Dmnl
    Limits: (None, None)
    Type: lookup
    Subs: None


    """
    return lookup(
        x,
        [0, 0.285132, 0.448065, 0.570265, 0.733198, 0.95723, 1.4664, 3.19756, 4.03259],
        [
            0,
            0.0616114,
            0.232228,
            0.492891,
            0.772512,
            0.862559,
            0.914692,
            0.952607,
            0.957346,
        ],
    )


@cache.step
def income():
    """
    Real Name: Income
    Original Eqn: Months of Expenses per Sale * Sales + IF THEN ELSE(Time < Startup Subsidy Length, Startup Subsidy, 0 )
    Units: Dmnl
    Limits: (None, None)
    Type: component
    Subs: None

    Technically in units of months of expenses earned per month
    """
    return months_of_expenses_per_sale() * sales() + if_then_else(
        time() < startup_subsidy_length(), lambda: startup_subsidy(), lambda: 0
    )


@cache.run
def months_of_expenses_per_sale():
    """
    Real Name: Months of Expenses per Sale
    Original Eqn: 12/50
    Units: Month/Person
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 12 / 50


@cache.step
def motivation():
    """
    Real Name: Motivation
    Original Eqn: INTEG ( Motivation Adjustment, 1)
    Units: Dmnl
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_motivation()


@cache.step
def motivation_adjustment():
    """
    Real Name: Motivation Adjustment
    Original Eqn: (Income - Motivation) / Motivation Adjustment Time
    Units: 1/Month
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return (income() - motivation()) / motivation_adjustment_time()


@cache.run
def motivation_adjustment_time():
    """
    Real Name: Motivation Adjustment Time
    Original Eqn: 3
    Units: Month
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 3


@cache.step
def sales():
    """
    Real Name: Sales
    Original Eqn: Effort / Effort Required to Make a Sale * Success Rate
    Units: Persons/Month
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return effort() / effort_required_to_make_a_sale() * success_rate()


@cache.run
def startup_subsidy():
    """
    Real Name: Startup Subsidy
    Original Eqn: 0.5
    Units: Dmnl
    Limits: (None, None)
    Type: constant
    Subs: None

    Months of expenses per month
    """
    return 0.5


@cache.run
def startup_subsidy_length():
    """
    Real Name: Startup Subsidy Length
    Original Eqn: 6
    Units: Month
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 6


@cache.run
def success_rate():
    """
    Real Name: Success Rate
    Original Eqn: 0.2
    Units: Dmnl
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 0.2


@cache.run
def final_time():
    """
    Real Name: FINAL TIME
    Original Eqn: 200
    Units: Month
    Limits: (None, None)
    Type: constant
    Subs: None

    The final time for the simulation.
    """
    return 200


@cache.run
def initial_time():
    """
    Real Name: INITIAL TIME
    Original Eqn: 0
    Units: Month
    Limits: (None, None)
    Type: constant
    Subs: None

    The initial time for the simulation.
    """
    return 0


@cache.step
def saveper():
    """
    Real Name: SAVEPER
    Original Eqn: TIME STEP
    Units: Month
    Limits: (0.0, None)
    Type: component
    Subs: None

    The frequency with which output is stored.
    """
    return time_step()


@cache.run
def time_step():
    """
    Real Name: TIME STEP
    Original Eqn: 0.0625
    Units: Month
    Limits: (0.0, None)
    Type: constant
    Subs: None

    The time step for the simulation.
    """
    return 0.0625


_integ_total_cumulative_income = Integ(lambda: accumulating_income(), lambda: 0)


_integ_total_cumulative_sales = Integ(lambda: accumulating_sales(), lambda: 0)


_integ_tenure = Integ(lambda: accumulating_tenure(), lambda: 0)


_integ_motivation = Integ(lambda: motivation_adjustment(), lambda: 1)
