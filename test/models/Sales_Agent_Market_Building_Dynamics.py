"""
Python model /Domain/tudelft.net/Users/jhkwakkel/EMAworkbench/test/test_connectors/../models/Sales_Agent_Market_Building_Dynamics.py
Translated using PySD version 0.7.8
"""
from __future__ import division
import numpy as np
from pysd import utils
import xarray as xr

from pysd.functions import cache
from pysd import functions

_subscript_dict = {}

_namespace = {
    'TIME':
    'time',
    'Time':
    'time',
    'Accumulating Income':
    'accumulating_income',
    'Accumulating Sales':
    'accumulating_sales',
    'Accumulating Tenure':
    'accumulating_tenure',
    'Total Cumulative Sales':
    'total_cumulative_sales',
    'Tenure':
    'tenure',
    'Total Cumulative Income':
    'total_cumulative_income',
    'Tier 2 Referrals':
    'tier_2_referrals',
    'Effort Remaining after Servicing Existing Clients':
    'effort_remaining_after_servicing_existing_clients',
    'Down Referral Fraction':
    'down_referral_fraction',
    'Effort Devoted to Tier 2 Leads':
    'effort_devoted_to_tier_2_leads',
    'Tier 2 Lead Aquisition':
    'tier_2_lead_aquisition',
    'Tier 1 Referrals from Tier 2':
    'tier_1_referrals_from_tier_2',
    'Effort Remaining after Servicing Tier 2 Leads':
    'effort_remaining_after_servicing_tier_2_leads',
    'Income':
    'income',
    'Qualification Rate':
    'qualification_rate',
    'Tier 1 Lead Aquisition':
    'tier_1_lead_aquisition',
    'Success Rate':
    'success_rate',
    'Tier 1 Sales':
    'tier_1_sales',
    'Tier 2 Sales':
    'tier_2_sales',
    'Still Employed':
    'still_employed',
    'Effort Devoted to Tier 1 Clients':
    'effort_devoted_to_tier_1_clients',
    'Tier 1 Income':
    'tier_1_income',
    'Effort Devoted to Tier 2 Clients':
    'effort_devoted_to_tier_2_clients',
    'Fraction of Effort for Sales':
    'fraction_of_effort_for_sales',
    'Expenses':
    'expenses',
    'Sales Effort Available':
    'sales_effort_available',
    'Initial Buffer':
    'initial_buffer',
    'Startup Subsidy Length':
    'startup_subsidy_length',
    'Total Effort Available':
    'total_effort_available',
    'Months of Buffer':
    'months_of_buffer',
    'Months of Expenses per Tier 1 Sale':
    'months_of_expenses_per_tier_1_sale',
    'Months of Expenses per Tier 2 Sale':
    'months_of_expenses_per_tier_2_sale',
    'Tier 2 Income':
    'tier_2_income',
    'Startup Subsidy':
    'startup_subsidy',
    'Time per Client Meeting':
    'time_per_client_meeting',
    'Client Lifetime':
    'client_lifetime',
    'Effort Devoted to Tier 1 Leads':
    'effort_devoted_to_tier_1_leads',
    'Frequency of Meetings':
    'frequency_of_meetings',
    'Lead Shelf Life':
    'lead_shelf_life',
    'Referrals from Tier 1 Clients':
    'referrals_from_tier_1_clients',
    'Referrals from Tier 2 Clients':
    'referrals_from_tier_2_clients',
    'Referrals per meeting':
    'referrals_per_meeting',
    'Tier 1 Client Turnover':
    'tier_1_client_turnover',
    'Up Referral Fraction':
    'up_referral_fraction',
    'Tier 1 Leads Going Stale':
    'tier_1_leads_going_stale',
    'Tier 1 Referrals':
    'tier_1_referrals',
    'Tier 2 Client Turnover':
    'tier_2_client_turnover',
    'Tier 2 Clients':
    'tier_2_clients',
    'Tier 2 Leads':
    'tier_2_leads',
    'Tier 2 Leads Going Stale':
    'tier_2_leads_going_stale',
    'Tier 2 Referrals from Tier 1':
    'tier_2_referrals_from_tier_1',
    'Effort Required to Make a Sale':
    'effort_required_to_make_a_sale',
    'Minimum Time to Make a Sale':
    'minimum_time_to_make_a_sale',
    'Tier 1 Leads':
    'tier_1_leads',
    'Tier 1 Clients':
    'tier_1_clients',
    'FINAL TIME':
    'final_time',
    'INITIAL TIME':
    'initial_time',
    'SAVEPER':
    'saveper',
    'TIME STEP':
    'time_step'}


@cache('step')
def accumulating_income():
    """
    Accumulating Income

    Months/Month


    """
    return income()


@cache('step')
def accumulating_sales():
    """
    Accumulating Sales

    Persons/Month


    """
    return tier_1_sales() + tier_2_sales()


@cache('step')
def accumulating_tenure():
    """
    Accumulating Tenure

    Months/Month


    """
    return still_employed()


@cache('step')
def total_cumulative_sales():
    """
    Total Cumulative Sales

    Persons


    """
    return integ_total_cumulative_sales()


@cache('step')
def tenure():
    """
    Tenure

    Months


    """
    return integ_tenure()


@cache('step')
def total_cumulative_income():
    """
    Total Cumulative Income

    Months


    """
    return integ_total_cumulative_income()


@cache('step')
def tier_2_referrals():
    """
    Tier 2 Referrals

    Referrals/Month

    This is the number of Tier 2 leads that are aquired through referrals from 
        any tier client.
    """
    return referrals_from_tier_2_clients() * (1 - down_referral_fraction())


@cache('step')
def effort_remaining_after_servicing_existing_clients():
    """
    Effort Remaining after Servicing Existing Clients

    Hours/Month

    How much effort remains after higher priority sales and maintenance 
        activities are complete?
    """
    return np.maximum(sales_effort_available() -
                      (effort_devoted_to_tier_1_clients() + effort_devoted_to_tier_2_clients()), 0)


@cache('run')
def down_referral_fraction():
    """
    Down Referral Fraction




    """
    return 0.2


@cache('step')
def effort_devoted_to_tier_2_leads():
    """
    Effort Devoted to Tier 2 Leads

    Hours/Month

    This is the amount of time the agent spends with a tier 2 lead in a given 
        year, working to make a sale.
    """
    return np.minimum(
        effort_remaining_after_servicing_existing_clients(),
        effort_required_to_make_a_sale() * tier_2_leads() / minimum_time_to_make_a_sale())


@cache('step')
def tier_2_lead_aquisition():
    """
    Tier 2 Lead Aquisition

    Persons/Month

    How many new tier 2 leads does an agent net?
    """
    return qualification_rate() * (tier_2_referrals() + tier_2_referrals_from_tier_1())


@cache('step')
def tier_1_referrals_from_tier_2():
    """
    Tier 1 Referrals from Tier 2

    Referrals/Month

    This is the number of Tier 1 leads that are aquired through referrals from 
        tier 2.
    """
    return referrals_from_tier_2_clients() * down_referral_fraction()


@cache('step')
def effort_remaining_after_servicing_tier_2_leads():
    """
    Effort Remaining after Servicing Tier 2 Leads

    Hours/Month

    How much effort remains after higher priority sales and maintenance 
        activities are complete?
    """
    return np.maximum(
        effort_remaining_after_servicing_existing_clients() - effort_devoted_to_tier_2_leads(), 0)


@cache('step')
def income():
    """
    Income

    Months/Month

    The total income from commissions on sales to all tiers.
    """
    return tier_1_income() + tier_2_income() + functions.if_then_else(
        time() < startup_subsidy_length(), startup_subsidy(), 0)


@cache('run')
def qualification_rate():
    """
    Qualification Rate

    Persons/Referral

    What is the likelihood that a lead will be worth pursuing? Some leads 
        might not be worth your effort. According to interviewees, leads that are 
        properly solicited and introduced are almost always worth following up 
        with.
    """
    return 1


@cache('step')
def tier_1_lead_aquisition():
    """
    Tier 1 Lead Aquisition

    Persons/Month

    How many new tier 1 leads does an agent net?
    """
    return qualification_rate() * (tier_1_referrals() + tier_1_referrals_from_tier_2())


@cache('run')
def success_rate():
    """
    Success Rate

    Dmnl

    What is the likelihood that a given lead will become a client, if the 
        agent devotes the appropriate amount of attention to them?
    """
    return 0.2


@cache('step')
def tier_1_sales():
    """
    Tier 1 Sales

    Persons/Month

    The rate at which Tier 1 leads become clients. This is limited either by 
        the effort of the agent, or the natural calendar time required to make a 
        sale.
    """
    return success_rate() * np.minimum(effort_devoted_to_tier_1_leads(
    ) / effort_required_to_make_a_sale(), tier_1_leads() / minimum_time_to_make_a_sale())


@cache('step')
def tier_2_sales():
    """
    Tier 2 Sales

    Persons/Month

    The rate at which Tier 2 leads become clients. This is limited either by 
        the effort of the agent, or the natural calendar time required to make a 
        sale.
    """
    return success_rate() * np.minimum(effort_devoted_to_tier_2_leads(
    ) / effort_required_to_make_a_sale(), tier_2_leads() / minimum_time_to_make_a_sale())


@cache('step')
def still_employed():
    """
    Still Employed

    Dmnl

    Flag for whether the agent is still with the firm. Goes to zero when the 
        buffer becomes negative.
    """
    return functions.if_then_else(months_of_buffer() < 0, 0, 1)


@cache('step')
def effort_devoted_to_tier_1_clients():
    """
    Effort Devoted to Tier 1 Clients

    Hours/Month

    How much time does the agent devote to meetings for maintenance and 
        soliciting referrals from Tier 1 Clients.
    """
    return tier_1_clients() * time_per_client_meeting() * frequency_of_meetings()


@cache('step')
def tier_1_income():
    """
    Tier 1 Income

    Months/Month

    This is the amount of money an agent makes from all commissions on Tier 1 
        Sales
    """
    return tier_1_sales() * months_of_expenses_per_tier_1_sale()


@cache('step')
def effort_devoted_to_tier_2_clients():
    """
    Effort Devoted to Tier 2 Clients

    Hours/Month

    How much time does the agent devote to meetings for maintenance and 
        soliciting referrals from Tier 2 Clients.
    """
    return tier_2_clients() * time_per_client_meeting() * frequency_of_meetings()


@cache('run')
def fraction_of_effort_for_sales():
    """
    Fraction of Effort for Sales

    Dmnl

    Of all the effort devoted to work, what fraction is actually spent doing 
        sales and maintenance activities? This includes time spent with existing 
        clients soliciting referrals.
    """
    return 0.25


@cache('run')
def expenses():
    """
    Expenses

    Months/Month

    How many months of expenses are expended per month. This is a bit of a 
        tautology, but its the right way to account for the agents income and 
        spending while preserving their privacy.
    """
    return 1


@cache('step')
def sales_effort_available():
    """
    Sales Effort Available

    Hours/Month

    How much total time per month can an agent actually spend in sales or 
        maintenance meetings?
    """
    return fraction_of_effort_for_sales() * total_effort_available() * still_employed()


@cache('run')
def initial_buffer():
    """
    Initial Buffer

    Months

    How long can the agent afford to go with zero income? This could be months 
        of expenses in the bank, or months of 'rent equivalent' they are able to 
        borrow from family, etc.
    """
    return 6


@cache('run')
def startup_subsidy_length():
    """
    Startup Subsidy Length

    Months

    How long does a sales agent recieve a subsidy for, before it is cut off?
    """
    return 3


@cache('run')
def total_effort_available():
    """
    Total Effort Available

    Hours/Month

    This is the total number of hours the agent is willing to work in a month.
    """
    return 200


@cache('step')
def months_of_buffer():
    """
    Months of Buffer

    Months

    This is the stock at any given time of the money in the bank, or remaining 
        familial goodwill, etc.
    """
    return integ_months_of_buffer()


@cache('run')
def months_of_expenses_per_tier_1_sale():
    """
    Months of Expenses per Tier 1 Sale

    Months/Person

    Income from commission for a sale to a tier 1 lead. Measured in units of 
        months of expenses, to preserve agents privacy.
    """
    return 12 / 300


@cache('run')
def months_of_expenses_per_tier_2_sale():
    """
    Months of Expenses per Tier 2 Sale

    Months/Person

    Income from commission for a sale to a tier 2 lead. Measured in units of 
        months of expenses, to preserve agents privacy.
    """
    return 12 / 30


@cache('step')
def tier_2_income():
    """
    Tier 2 Income

    Months/Month

    This is the amount of money an agent makes from all commissions on Tier 2 
        Sales
    """
    return months_of_expenses_per_tier_2_sale() * tier_2_sales()


@cache('run')
def startup_subsidy():
    """
    Startup Subsidy

    Months/Month [0,1,0.1]

    How much does an agent recieve each month from his sales manager to help 
        defer his expenses, in units of months of expenses?
    """
    return 0.75


@cache('run')
def time_per_client_meeting():
    """
    Time per Client Meeting

    Hours/Meeting

    This is the number of hours an agent spends with a client, maintaining the 
        relationship/accounts, and soliciting referrals, in one sitting.
    """
    return 1


@cache('run')
def client_lifetime():
    """
    Client Lifetime

    Months

    How long, on average, does a client remain with an agent?
    """
    return 120


@cache('step')
def effort_devoted_to_tier_1_leads():
    """
    Effort Devoted to Tier 1 Leads

    Hours/Month

    This is the amount of time the agent spends with a tier 1 lead in a given 
        year, working to make a sale.
    """
    return effort_remaining_after_servicing_tier_2_leads()


@cache('run')
def frequency_of_meetings():
    """
    Frequency of Meetings

    Meetings/Month/Person

    How many maintenance meetings does the agent have with each client in a 
        month?
    """
    return 1 / 12


@cache('run')
def lead_shelf_life():
    """
    Lead Shelf Life

    Months

    After a certain amount of time, leads go stale. It gets awkward to keep 
        interacting with them, and you're better off moving on. How long is that?
    """
    return 3


@cache('step')
def referrals_from_tier_1_clients():
    """
    Referrals from Tier 1 Clients

    Referrals/Month

    The number of referrals coming in from maintenance meetings with tier 1 
        clients.
    """
    return tier_1_clients() * frequency_of_meetings() * referrals_per_meeting()


@cache('step')
def referrals_from_tier_2_clients():
    """
    Referrals from Tier 2 Clients

    Referrals/Month

    The number of referrals coming in from maintenance meetings with tier 2 
        clients.
    """
    return tier_2_clients() * referrals_per_meeting() * frequency_of_meetings()


@cache('run')
def referrals_per_meeting():
    """
    Referrals per meeting

    Referrals/Meeting

    How many referrals can an agent comfortably gather from his clients in a 
        given maintenance meeting?
    """
    return 2


@cache('step')
def tier_1_client_turnover():
    """
    Tier 1 Client Turnover

    Persons/Month

    This is the flow of tier 1 clients leaving the practice.
    """
    return tier_1_clients() / client_lifetime()


@cache('run')
def up_referral_fraction():
    """
    Up Referral Fraction

    Dmnl

    The likelihood that a referral from a tier 1 or tier 2 client will be to a 
        lead of the tier above them.
    """
    return 0.15


@cache('step')
def tier_1_leads_going_stale():
    """
    Tier 1 Leads Going Stale

    Persons/Month

    These are tier 1 leads that grow old before they are sold, and are unable 
        to be followed up on.
    """
    return tier_1_leads() / lead_shelf_life()


@cache('step')
def tier_1_referrals():
    """
    Tier 1 Referrals

    Referrals/Month

    This is the number of Tier 1 leads that are aquired through referrals from 
        any tier client.
    """
    return referrals_from_tier_1_clients() * (1 - up_referral_fraction())


@cache('step')
def tier_2_client_turnover():
    """
    Tier 2 Client Turnover

    Persons/Month

    This is the flow of Tier 2 clients leaving the practice.
    """
    return tier_2_clients() / client_lifetime()


@cache('step')
def tier_2_clients():
    """
    Tier 2 Clients

    Persons

    These are active clients who provide a regular level of return to the 
        company.
    """
    return integ_tier_2_clients()


@cache('step')
def tier_2_leads():
    """
    Tier 2 Leads

    Persons

    These are individuals who have been identified as targets and are somewhere in the 
        sales process, before a sale has been made.         They may or may not have been contacted by the agent yet. If they can be 
        converted to clients, they will have a regular level of return for the 
        company.
    """
    return integ_tier_2_leads()


@cache('step')
def tier_2_leads_going_stale():
    """
    Tier 2 Leads Going Stale

    Persons/Month

    These are tier 2 leads that grow old before they are sold, and are unable 
        to be followed up on.
    """
    return tier_2_leads() / lead_shelf_life()


@cache('step')
def tier_2_referrals_from_tier_1():
    """
    Tier 2 Referrals from Tier 1

    Referrals/Month

    This is the number of Tier 2 leads that are aquired through referrals from 
        tier 1.
    """
    return referrals_from_tier_1_clients() * up_referral_fraction()


@cache('run')
def effort_required_to_make_a_sale():
    """
    Effort Required to Make a Sale

    Hours/Person [0,50]

    This is the amount of time the agent must spend (on average) with a lead 
        (high or low value, for now) to make a sale.
    """
    return 4


@cache('run')
def minimum_time_to_make_a_sale():
    """
    Minimum Time to Make a Sale

    Months

    What is the absolute minimum calendar time it would take to make a sale to 
        a person, even if you had all the hours in the day to devote to them?
    """
    return 1


@cache('step')
def tier_1_leads():
    """
    Tier 1 Leads

    Persons

    These are individuals who have been identified as targets and are somewhere in the 
        sales process, before a sale has been made.         They may or may not have been contacted by the agent yet. If they can be converted 
        to clients, they will have a regular level of return for the company.                We initialize to 100 because agents begin their sales careers with a list 
        of 200 friends and family, about 50% of whom they might contact.
    """
    return integ_tier_1_leads()


@cache('step')
def tier_1_clients():
    """
    Tier 1 Clients

    Persons

    These are active clients who provide a regular level of return to the 
        company.
    """
    return integ_tier_1_clients()


@cache('run')
def final_time():
    """
    FINAL TIME

    Month

    The final time for the simulation.
    """
    return 200


@cache('run')
def initial_time():
    """
    INITIAL TIME

    Month

    The initial time for the simulation.
    """
    return 0


@cache('step')
def saveper():
    """
    SAVEPER

    Month [0,?]

    The frequency with which output is stored.
    """
    return time_step()


@cache('run')
def time_step():
    """
    TIME STEP

    Month [0,?]

    The time step for the simulation.
    """
    return 0.0625


integ_total_cumulative_sales = functions.Integ(lambda: accumulating_sales(), lambda: 0)

integ_tenure = functions.Integ(lambda: accumulating_tenure(), lambda: 0)

integ_total_cumulative_income = functions.Integ(lambda: accumulating_income(), lambda: 0)

integ_months_of_buffer = functions.Integ(lambda: income() - expenses(), lambda: initial_buffer())

integ_tier_2_clients = functions.Integ(lambda: tier_2_sales() - tier_2_client_turnover(),
                                       lambda: 0)

integ_tier_2_leads = functions.Integ(
    lambda: tier_2_lead_aquisition() + tier_2_sales() - tier_2_leads_going_stale(), lambda: 0)

integ_tier_1_leads = functions.Integ(
    lambda: tier_1_lead_aquisition() + tier_1_sales() - tier_1_leads_going_stale(), lambda: 100)

integ_tier_1_clients = functions.Integ(lambda: tier_1_sales() - tier_1_client_turnover(),
                                       lambda: 0)
