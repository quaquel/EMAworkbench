"""
Python model "Sales_Agent_Market_Building_Dynamics.py"
Translated using PySD version 1.3.0
"""
from os import path
import numpy as np

from pysd.py_backend.functions import Integ, if_then_else
from pysd import cache

_subscript_dict = {}

_namespace = {
    "TIME": "time",
    "Time": "time",
    "Accumulating Income": "accumulating_income",
    "Accumulating Sales": "accumulating_sales",
    "Accumulating Tenure": "accumulating_tenure",
    "Total Cumulative Sales": "total_cumulative_sales",
    "Tenure": "tenure",
    "Total Cumulative Income": "total_cumulative_income",
    "Tier 2 Referrals": "tier_2_referrals",
    "Effort Remaining after Servicing Existing Clients": "effort_remaining_after_servicing_existing_clients",
    "Down Referral Fraction": "down_referral_fraction",
    "Effort Devoted to Tier 2 Leads": "effort_devoted_to_tier_2_leads",
    "Tier 2 Lead Aquisition": "tier_2_lead_aquisition",
    "Tier 1 Referrals from Tier 2": "tier_1_referrals_from_tier_2",
    "Effort Remaining after Servicing Tier 2 Leads": "effort_remaining_after_servicing_tier_2_leads",
    "Income": "income",
    "Qualification Rate": "qualification_rate",
    "Tier 1 Lead Aquisition": "tier_1_lead_aquisition",
    "Success Rate": "success_rate",
    "Tier 1 Sales": "tier_1_sales",
    "Tier 2 Sales": "tier_2_sales",
    "Still Employed": "still_employed",
    "Effort Devoted to Tier 1 Clients": "effort_devoted_to_tier_1_clients",
    "Tier 1 Income": "tier_1_income",
    "Effort Devoted to Tier 2 Clients": "effort_devoted_to_tier_2_clients",
    "Fraction of Effort for Sales": "fraction_of_effort_for_sales",
    "Expenses": "expenses",
    "Sales Effort Available": "sales_effort_available",
    "Initial Buffer": "initial_buffer",
    "Startup Subsidy Length": "startup_subsidy_length",
    "Total Effort Available": "total_effort_available",
    "Months of Buffer": "months_of_buffer",
    "Months of Expenses per Tier 1 Sale": "months_of_expenses_per_tier_1_sale",
    "Months of Expenses per Tier 2 Sale": "months_of_expenses_per_tier_2_sale",
    "Tier 2 Income": "tier_2_income",
    "Startup Subsidy": "startup_subsidy",
    "Time per Client Meeting": "time_per_client_meeting",
    "Client Lifetime": "client_lifetime",
    "Effort Devoted to Tier 1 Leads": "effort_devoted_to_tier_1_leads",
    "Frequency of Meetings": "frequency_of_meetings",
    "Lead Shelf Life": "lead_shelf_life",
    "Referrals from Tier 1 Clients": "referrals_from_tier_1_clients",
    "Referrals from Tier 2 Clients": "referrals_from_tier_2_clients",
    "Referrals per meeting": "referrals_per_meeting",
    "Tier 1 Client Turnover": "tier_1_client_turnover",
    "Up Referral Fraction": "up_referral_fraction",
    "Tier 1 Leads Going Stale": "tier_1_leads_going_stale",
    "Tier 1 Referrals": "tier_1_referrals",
    "Tier 2 Client Turnover": "tier_2_client_turnover",
    "Tier 2 Clients": "tier_2_clients",
    "Tier 2 Leads": "tier_2_leads",
    "Tier 2 Leads Going Stale": "tier_2_leads_going_stale",
    "Tier 2 Referrals from Tier 1": "tier_2_referrals_from_tier_1",
    "Effort Required to Make a Sale": "effort_required_to_make_a_sale",
    "Minimum Time to Make a Sale": "minimum_time_to_make_a_sale",
    "Tier 1 Leads": "tier_1_leads",
    "Tier 1 Clients": "tier_1_clients",
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
def accumulating_income():
    """
    Real Name: Accumulating Income
    Original Eqn: Income
    Units: Months/Month
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return income()


@cache.step
def accumulating_sales():
    """
    Real Name: Accumulating Sales
    Original Eqn: Tier 1 Sales + Tier 2 Sales
    Units: Persons/Month
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return tier_1_sales() + tier_2_sales()


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
    Units: Months
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_tenure()


@cache.step
def total_cumulative_income():
    """
    Real Name: Total Cumulative Income
    Original Eqn: INTEG ( Accumulating Income, 0)
    Units: Months
    Limits: (None, None)
    Type: component
    Subs: None


    """
    return _integ_total_cumulative_income()


@cache.step
def tier_2_referrals():
    """
    Real Name: Tier 2 Referrals
    Original Eqn: Referrals from Tier 2 Clients * (1-Down Referral Fraction)
    Units: Referrals/Month
    Limits: (None, None)
    Type: component
    Subs: None

    This is the number of Tier 2 leads that are aquired through referrals from
        any tier client.
    """
    return referrals_from_tier_2_clients() * (1 - down_referral_fraction())


@cache.step
def effort_remaining_after_servicing_existing_clients():
    """
    Real Name: Effort Remaining after Servicing Existing Clients
    Original Eqn: MAX(Sales Effort Available - (Effort Devoted to Tier 1 Clients + Effort Devoted to Tier 2 Clients), 0)
    Units: Hours/Month
    Limits: (None, None)
    Type: component
    Subs: None

    How much effort remains after higher priority sales and maintenance
        activities are complete?
    """
    return np.maximum(
        sales_effort_available()
        - (effort_devoted_to_tier_1_clients() + effort_devoted_to_tier_2_clients()),
        0,
    )


@cache.run
def down_referral_fraction():
    """
    Real Name: Down Referral Fraction
    Original Eqn: 0.2
    Units:
    Limits: (None, None)
    Type: constant
    Subs: None


    """
    return 0.2


@cache.step
def effort_devoted_to_tier_2_leads():
    """
    Real Name: Effort Devoted to Tier 2 Leads
    Original Eqn: MIN(Effort Remaining after Servicing Existing Clients, Effort Required to Make a Sale* Tier 2 Leads / Minimum Time to Make a Sale )
    Units: Hours/Month
    Limits: (None, None)
    Type: component
    Subs: None

    This is the amount of time the agent spends with a tier 2 lead in a given
        year, working to make a sale.
    """
    return np.minimum(
        effort_remaining_after_servicing_existing_clients(),
        effort_required_to_make_a_sale()
        * tier_2_leads()
        / minimum_time_to_make_a_sale(),
    )


@cache.step
def tier_2_lead_aquisition():
    """
    Real Name: Tier 2 Lead Aquisition
    Original Eqn: Qualification Rate * (Tier 2 Referrals + Tier 2 Referrals from Tier 1)
    Units: Persons/Month
    Limits: (None, None)
    Type: component
    Subs: None

    How many new tier 2 leads does an agent net?
    """
    return qualification_rate() * (tier_2_referrals() + tier_2_referrals_from_tier_1())


@cache.step
def tier_1_referrals_from_tier_2():
    """
    Real Name: Tier 1 Referrals from Tier 2
    Original Eqn: Referrals from Tier 2 Clients * Down Referral Fraction
    Units: Referrals/Month
    Limits: (None, None)
    Type: component
    Subs: None

    This is the number of Tier 1 leads that are aquired through referrals from
        tier 2.
    """
    return referrals_from_tier_2_clients() * down_referral_fraction()


@cache.step
def effort_remaining_after_servicing_tier_2_leads():
    """
    Real Name: Effort Remaining after Servicing Tier 2 Leads
    Original Eqn: MAX(Effort Remaining after Servicing Existing Clients - Effort Devoted to Tier 2 Leads, 0)
    Units: Hours/Month
    Limits: (None, None)
    Type: component
    Subs: None

    How much effort remains after higher priority sales and maintenance
        activities are complete?
    """
    return np.maximum(
        effort_remaining_after_servicing_existing_clients()
        - effort_devoted_to_tier_2_leads(),
        0,
    )


@cache.step
def income():
    """
    Real Name: Income
    Original Eqn: Tier 1 Income + Tier 2 Income + IF THEN ELSE(Time < Startup Subsidy Length, Startup Subsidy, 0 )
    Units: Months/Month
    Limits: (None, None)
    Type: component
    Subs: None

    The total income from commissions on sales to all tiers.
    """
    return (
        tier_1_income()
        + tier_2_income()
        + if_then_else(
            time() < startup_subsidy_length(), lambda: startup_subsidy(), lambda: 0
        )
    )


@cache.run
def qualification_rate():
    """
    Real Name: Qualification Rate
    Original Eqn: 1
    Units: Persons/Referral
    Limits: (None, None)
    Type: constant
    Subs: None

    What is the likelihood that a lead will be worth pursuing? Some leads
        might not be worth your effort. According to interviewees, leads that are
        properly solicited and introduced are almost always worth following up
        with.
    """
    return 1


@cache.step
def tier_1_lead_aquisition():
    """
    Real Name: Tier 1 Lead Aquisition
    Original Eqn: Qualification Rate * (Tier 1 Referrals + Tier 1 Referrals from Tier 2)
    Units: Persons/Month
    Limits: (None, None)
    Type: component
    Subs: None

    How many new tier 1 leads does an agent net?
    """
    return qualification_rate() * (tier_1_referrals() + tier_1_referrals_from_tier_2())


@cache.run
def success_rate():
    """
    Real Name: Success Rate
    Original Eqn: 0.2
    Units: Dmnl
    Limits: (None, None)
    Type: constant
    Subs: None

    What is the likelihood that a given lead will become a client, if the
        agent devotes the appropriate amount of attention to them?
    """
    return 0.2


@cache.step
def tier_1_sales():
    """
    Real Name: Tier 1 Sales
    Original Eqn: Success Rate*MIN(Effort Devoted to Tier 1 Leads / Effort Required to Make a Sale, Tier 1 Leads/Minimum Time to Make a Sale)
    Units: Persons/Month
    Limits: (None, None)
    Type: component
    Subs: None

    The rate at which Tier 1 leads become clients. This is limited either by
        the effort of the agent, or the natural calendar time required to make a
        sale.
    """
    return success_rate() * np.minimum(
        effort_devoted_to_tier_1_leads() / effort_required_to_make_a_sale(),
        tier_1_leads() / minimum_time_to_make_a_sale(),
    )


@cache.step
def tier_2_sales():
    """
    Real Name: Tier 2 Sales
    Original Eqn: Success Rate*MIN(Effort Devoted to Tier 2 Leads / Effort Required to Make a Sale, Tier 2 Leads/Minimum Time to Make a Sale)
    Units: Persons/Month
    Limits: (None, None)
    Type: component
    Subs: None

    The rate at which Tier 2 leads become clients. This is limited either by
        the effort of the agent, or the natural calendar time required to make a
        sale.
    """
    return success_rate() * np.minimum(
        effort_devoted_to_tier_2_leads() / effort_required_to_make_a_sale(),
        tier_2_leads() / minimum_time_to_make_a_sale(),
    )


@cache.step
def still_employed():
    """
    Real Name: Still Employed
    Original Eqn: IF THEN ELSE(Months of Buffer < 0 , 0 , 1 )
    Units: Dmnl
    Limits: (None, None)
    Type: component
    Subs: None

    Flag for whether the agent is still with the firm. Goes to zero when the
        buffer becomes negative.
    """
    return if_then_else(months_of_buffer() < 0, lambda: 0, lambda: 1)


@cache.step
def effort_devoted_to_tier_1_clients():
    """
    Real Name: Effort Devoted to Tier 1 Clients
    Original Eqn: Tier 1 Clients * Time per Client Meeting * Frequency of Meetings
    Units: Hours/Month
    Limits: (None, None)
    Type: component
    Subs: None

    How much time does the agent devote to meetings for maintenance and
        soliciting referrals from Tier 1 Clients.
    """
    return tier_1_clients() * time_per_client_meeting() * frequency_of_meetings()


@cache.step
def tier_1_income():
    """
    Real Name: Tier 1 Income
    Original Eqn: Tier 1 Sales * Months of Expenses per Tier 1 Sale
    Units: Months/Month
    Limits: (None, None)
    Type: component
    Subs: None

    This is the amount of money an agent makes from all commissions on Tier 1
        Sales
    """
    return tier_1_sales() * months_of_expenses_per_tier_1_sale()


@cache.step
def effort_devoted_to_tier_2_clients():
    """
    Real Name: Effort Devoted to Tier 2 Clients
    Original Eqn: Tier 2 Clients * Time per Client Meeting * Frequency of Meetings
    Units: Hours/Month
    Limits: (None, None)
    Type: component
    Subs: None

    How much time does the agent devote to meetings for maintenance and
        soliciting referrals from Tier 2 Clients.
    """
    return tier_2_clients() * time_per_client_meeting() * frequency_of_meetings()


@cache.run
def fraction_of_effort_for_sales():
    """
    Real Name: Fraction of Effort for Sales
    Original Eqn: 0.25
    Units: Dmnl
    Limits: (None, None)
    Type: constant
    Subs: None

    Of all the effort devoted to work, what fraction is actually spent doing
        sales and maintenance activities? This includes time spent with existing
        clients soliciting referrals.
    """
    return 0.25


@cache.run
def expenses():
    """
    Real Name: Expenses
    Original Eqn: 1
    Units: Months/Month
    Limits: (None, None)
    Type: constant
    Subs: None

    How many months of expenses are expended per month. This is a bit of a
        tautology, but its the right way to account for the agents income and
        spending while preserving their privacy.
    """
    return 1


@cache.step
def sales_effort_available():
    """
    Real Name: Sales Effort Available
    Original Eqn: Fraction of Effort for Sales * Total Effort Available * Still Employed
    Units: Hours/Month
    Limits: (None, None)
    Type: component
    Subs: None

    How much total time per month can an agent actually spend in sales or
        maintenance meetings?
    """
    return fraction_of_effort_for_sales() * total_effort_available() * still_employed()


@cache.run
def initial_buffer():
    """
    Real Name: Initial Buffer
    Original Eqn: 6
    Units: Months
    Limits: (None, None)
    Type: constant
    Subs: None

    How long can the agent afford to go with zero income? This could be months
        of expenses in the bank, or months of 'rent equivalent' they are able to
        borrow from family, etc.
    """
    return 6


@cache.run
def startup_subsidy_length():
    """
    Real Name: Startup Subsidy Length
    Original Eqn: 3
    Units: Months
    Limits: (None, None)
    Type: constant
    Subs: None

    How long does a sales agent recieve a subsidy for, before it is cut off?
    """
    return 3


@cache.run
def total_effort_available():
    """
    Real Name: Total Effort Available
    Original Eqn: 200
    Units: Hours/Month
    Limits: (None, None)
    Type: constant
    Subs: None

    This is the total number of hours the agent is willing to work in a month.
    """
    return 200


@cache.step
def months_of_buffer():
    """
    Real Name: Months of Buffer
    Original Eqn: INTEG ( Income-Expenses, Initial Buffer)
    Units: Months
    Limits: (None, None)
    Type: component
    Subs: None

    This is the stock at any given time of the money in the bank, or remaining
        familial goodwill, etc.
    """
    return _integ_months_of_buffer()


@cache.run
def months_of_expenses_per_tier_1_sale():
    """
    Real Name: Months of Expenses per Tier 1 Sale
    Original Eqn: 12/300
    Units: Months/Person
    Limits: (None, None)
    Type: constant
    Subs: None

    Income from commission for a sale to a tier 1 lead. Measured in units of
        months of expenses, to preserve agents privacy.
    """
    return 12 / 300


@cache.run
def months_of_expenses_per_tier_2_sale():
    """
    Real Name: Months of Expenses per Tier 2 Sale
    Original Eqn: 12/30
    Units: Months/Person
    Limits: (None, None)
    Type: constant
    Subs: None

    Income from commission for a sale to a tier 2 lead. Measured in units of
        months of expenses, to preserve agents privacy.
    """
    return 12 / 30


@cache.step
def tier_2_income():
    """
    Real Name: Tier 2 Income
    Original Eqn: Months of Expenses per Tier 2 Sale * Tier 2 Sales
    Units: Months/Month
    Limits: (None, None)
    Type: component
    Subs: None

    This is the amount of money an agent makes from all commissions on Tier 2
        Sales
    """
    return months_of_expenses_per_tier_2_sale() * tier_2_sales()


@cache.run
def startup_subsidy():
    """
    Real Name: Startup Subsidy
    Original Eqn: 0.75
    Units: Months/Month
    Limits: (0.0, 1.0, 0.1)
    Type: constant
    Subs: None

    How much does an agent recieve each month from his sales manager to help
        defer his expenses, in units of months of expenses?
    """
    return 0.75


@cache.run
def time_per_client_meeting():
    """
    Real Name: Time per Client Meeting
    Original Eqn: 1
    Units: Hours/Meeting
    Limits: (None, None)
    Type: constant
    Subs: None

    This is the number of hours an agent spends with a client, maintaining the
        relationship/accounts, and soliciting referrals, in one sitting.
    """
    return 1


@cache.run
def client_lifetime():
    """
    Real Name: Client Lifetime
    Original Eqn: 120
    Units: Months
    Limits: (None, None)
    Type: constant
    Subs: None

    How long, on average, does a client remain with an agent?
    """
    return 120


@cache.step
def effort_devoted_to_tier_1_leads():
    """
    Real Name: Effort Devoted to Tier 1 Leads
    Original Eqn: Effort Remaining after Servicing Tier 2 Leads
    Units: Hours/Month
    Limits: (None, None)
    Type: component
    Subs: None

    This is the amount of time the agent spends with a tier 1 lead in a given
        year, working to make a sale.
    """
    return effort_remaining_after_servicing_tier_2_leads()


@cache.run
def frequency_of_meetings():
    """
    Real Name: Frequency of Meetings
    Original Eqn: 1/12
    Units: Meetings/Month/Person
    Limits: (None, None)
    Type: constant
    Subs: None

    How many maintenance meetings does the agent have with each client in a
        month?
    """
    return 1 / 12


@cache.run
def lead_shelf_life():
    """
    Real Name: Lead Shelf Life
    Original Eqn: 3
    Units: Months
    Limits: (None, None)
    Type: constant
    Subs: None

    After a certain amount of time, leads go stale. It gets awkward to keep
        interacting with them, and you're better off moving on. How long is that?
    """
    return 3


@cache.step
def referrals_from_tier_1_clients():
    """
    Real Name: Referrals from Tier 1 Clients
    Original Eqn: Tier 1 Clients * Frequency of Meetings * Referrals per meeting
    Units: Referrals/Month
    Limits: (None, None)
    Type: component
    Subs: None

    The number of referrals coming in from maintenance meetings with tier 1
        clients.
    """
    return tier_1_clients() * frequency_of_meetings() * referrals_per_meeting()


@cache.step
def referrals_from_tier_2_clients():
    """
    Real Name: Referrals from Tier 2 Clients
    Original Eqn: Tier 2 Clients * Referrals per meeting * Frequency of Meetings
    Units: Referrals/Month
    Limits: (None, None)
    Type: component
    Subs: None

    The number of referrals coming in from maintenance meetings with tier 2
        clients.
    """
    return tier_2_clients() * referrals_per_meeting() * frequency_of_meetings()


@cache.run
def referrals_per_meeting():
    """
    Real Name: Referrals per meeting
    Original Eqn: 2
    Units: Referrals/Meeting
    Limits: (None, None)
    Type: constant
    Subs: None

    How many referrals can an agent comfortably gather from his clients in a
        given maintenance meeting?
    """
    return 2


@cache.step
def tier_1_client_turnover():
    """
    Real Name: Tier 1 Client Turnover
    Original Eqn: Tier 1 Clients/Client Lifetime
    Units: Persons/Month
    Limits: (None, None)
    Type: component
    Subs: None

    This is the flow of tier 1 clients leaving the practice.
    """
    return tier_1_clients() / client_lifetime()


@cache.run
def up_referral_fraction():
    """
    Real Name: Up Referral Fraction
    Original Eqn: 0.15
    Units: Dmnl
    Limits: (None, None)
    Type: constant
    Subs: None

    The likelihood that a referral from a tier 1 or tier 2 client will be to a
        lead of the tier above them.
    """
    return 0.15


@cache.step
def tier_1_leads_going_stale():
    """
    Real Name: Tier 1 Leads Going Stale
    Original Eqn: Tier 1 Leads/Lead Shelf Life
    Units: Persons/Month
    Limits: (None, None)
    Type: component
    Subs: None

    These are tier 1 leads that grow old before they are sold, and are unable
        to be followed up on.
    """
    return tier_1_leads() / lead_shelf_life()


@cache.step
def tier_1_referrals():
    """
    Real Name: Tier 1 Referrals
    Original Eqn: Referrals from Tier 1 Clients * (1-Up Referral Fraction)
    Units: Referrals/Month
    Limits: (None, None)
    Type: component
    Subs: None

    This is the number of Tier 1 leads that are aquired through referrals from
        any tier client.
    """
    return referrals_from_tier_1_clients() * (1 - up_referral_fraction())


@cache.step
def tier_2_client_turnover():
    """
    Real Name: Tier 2 Client Turnover
    Original Eqn: Tier 2 Clients/Client Lifetime
    Units: Persons/Month
    Limits: (None, None)
    Type: component
    Subs: None

    This is the flow of Tier 2 clients leaving the practice.
    """
    return tier_2_clients() / client_lifetime()


@cache.step
def tier_2_clients():
    """
    Real Name: Tier 2 Clients
    Original Eqn: INTEG ( Tier 2 Sales-Tier 2 Client Turnover, 0)
    Units: Persons
    Limits: (None, None)
    Type: component
    Subs: None

    These are active clients who provide a regular level of return to the
        company.
    """
    return _integ_tier_2_clients()


@cache.step
def tier_2_leads():
    """
    Real Name: Tier 2 Leads
    Original Eqn: INTEG ( Tier 2 Lead Aquisition+Tier 2 Sales-Tier 2 Leads Going Stale, 0)
    Units: Persons
    Limits: (None, None)
    Type: component
    Subs: None

    These are individuals who have been identified as targets and are somewhere in the
        sales process, before a sale has been made.         They may or may not have been contacted by the agent yet. If they can be
        converted to clients, they will have a regular level of return for the
        company.
    """
    return _integ_tier_2_leads()


@cache.step
def tier_2_leads_going_stale():
    """
    Real Name: Tier 2 Leads Going Stale
    Original Eqn: Tier 2 Leads/Lead Shelf Life
    Units: Persons/Month
    Limits: (None, None)
    Type: component
    Subs: None

    These are tier 2 leads that grow old before they are sold, and are unable
        to be followed up on.
    """
    return tier_2_leads() / lead_shelf_life()


@cache.step
def tier_2_referrals_from_tier_1():
    """
    Real Name: Tier 2 Referrals from Tier 1
    Original Eqn: Referrals from Tier 1 Clients * Up Referral Fraction
    Units: Referrals/Month
    Limits: (None, None)
    Type: component
    Subs: None

    This is the number of Tier 2 leads that are aquired through referrals from
        tier 1.
    """
    return referrals_from_tier_1_clients() * up_referral_fraction()


@cache.run
def effort_required_to_make_a_sale():
    """
    Real Name: Effort Required to Make a Sale
    Original Eqn: 4
    Units: Hours/Person
    Limits: (0.0, 50.0)
    Type: constant
    Subs: None

    This is the amount of time the agent must spend (on average) with a lead
        (high or low value, for now) to make a sale.
    """
    return 4


@cache.run
def minimum_time_to_make_a_sale():
    """
    Real Name: Minimum Time to Make a Sale
    Original Eqn: 1
    Units: Months
    Limits: (None, None)
    Type: constant
    Subs: None

    What is the absolute minimum calendar time it would take to make a sale to
        a person, even if you had all the hours in the day to devote to them?
    """
    return 1


@cache.step
def tier_1_leads():
    """
    Real Name: Tier 1 Leads
    Original Eqn: INTEG ( Tier 1 Lead Aquisition+Tier 1 Sales-Tier 1 Leads Going Stale, 100)
    Units: Persons
    Limits: (None, None)
    Type: component
    Subs: None

    These are individuals who have been identified as targets and are somewhere in the
        sales process, before a sale has been made.         They may or may not have been contacted by the agent yet. If they can be converted
        to clients, they will have a regular level of return for the company.                We initialize to 100 because agents begin their sales careers with a list
        of 200 friends and family, about 50% of whom they might contact.
    """
    return _integ_tier_1_leads()


@cache.step
def tier_1_clients():
    """
    Real Name: Tier 1 Clients
    Original Eqn: INTEG ( Tier 1 Sales-Tier 1 Client Turnover, 0)
    Units: Persons
    Limits: (None, None)
    Type: component
    Subs: None

    These are active clients who provide a regular level of return to the
        company.
    """
    return _integ_tier_1_clients()


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


_integ_total_cumulative_sales = Integ(lambda: accumulating_sales(), lambda: 0)


_integ_tenure = Integ(lambda: accumulating_tenure(), lambda: 0)


_integ_total_cumulative_income = Integ(lambda: accumulating_income(), lambda: 0)


_integ_months_of_buffer = Integ(lambda: income() - expenses(), lambda: initial_buffer())


_integ_tier_2_clients = Integ(
    lambda: tier_2_sales() - tier_2_client_turnover(), lambda: 0
)


_integ_tier_2_leads = Integ(
    lambda: tier_2_lead_aquisition() + tier_2_sales() - tier_2_leads_going_stale(),
    lambda: 0,
)


_integ_tier_1_leads = Integ(
    lambda: tier_1_lead_aquisition() + tier_1_sales() - tier_1_leads_going_stale(),
    lambda: 100,
)


_integ_tier_1_clients = Integ(
    lambda: tier_1_sales() - tier_1_client_turnover(), lambda: 0
)
