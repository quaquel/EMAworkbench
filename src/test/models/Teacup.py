
"""
Python model /Users/jhkwakkel/EMAworkbench/src/test/test_connectors/../models/Teacup.py
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
    'INITIAL TIME': 'initial_time',
    'Characteristic Time': 'characteristic_time',
    'Heat Loss to Room': 'heat_loss_to_room',
    'SAVEPER': 'saveper',
    'FINAL TIME': 'final_time',
    'Room Temperature': 'room_temperature',
    'Teacup Temperature': 'teacup_temperature',
    'TIME STEP': 'time_step'}


@cache('step')
def teacup_temperature():
    """
    Teacup Temperature
    ------------------
    (teacup_temperature)
    Degrees

    """
    return _state['teacup_temperature']


def _init_teacup_temperature():
    """
    Implicit
    --------
    (_init_teacup_temperature)
    See docs for teacup_temperature
    Provides initial conditions for teacup_temperature function
    """
    return 180


@cache('run')
def final_time():
    """
    FINAL TIME
    ----------
    (final_time)
    Minute
    The final time for the simulation.
    """
    return 30


@cache('run')
def room_temperature():
    """
    Room Temperature
    ----------------
    (room_temperature)


    """
    return 70


@cache('step')
def heat_loss_to_room():
    """
    Heat Loss to Room
    -----------------
    (heat_loss_to_room)
    Degrees/Minute
    This is the rate at which heat flows from the cup into the room. We can
                ignore it at this point.
    """
    return (teacup_temperature() - room_temperature()) / characteristic_time()


@cache('run')
def characteristic_time():
    """
    Characteristic Time
    -------------------
    (characteristic_time)
    Minutes

    """
    return 10


@cache('step')
def saveper():
    """
    SAVEPER
    -------
    (saveper)
    Minute [0,?]
    The frequency with which output is stored.
    """
    return time_step()


@cache('run')
def initial_time():
    """
    INITIAL TIME
    ------------
    (initial_time)
    Minute
    The initial time for the simulation.
    """
    return 0


@cache('run')
def time_step():
    """
    TIME STEP
    ---------
    (time_step)
    Minute [0,?]
    The time step for the simulation.
    """
    return 0.125


@cache('step')
def _dteacup_temperature_dt():
    """
    Implicit
    --------
    (_dteacup_temperature_dt)
    See docs for teacup_temperature
    Provides derivative for teacup_temperature function
    """
    return -heat_loss_to_room()


def time():
    return _t
functions.time = time
functions.initial_time = initial_time
