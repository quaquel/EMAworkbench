"""
Python model /Domain/tudelft.net/Users/jhkwakkel/EMAworkbench/test/test_connectors/../models/Teacup.py
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
    'TIME': 'time',
    'Time': 'time',
    'Characteristic Time': 'characteristic_time',
    'Heat Loss to Room': 'heat_loss_to_room',
    'Room Temperature': 'room_temperature',
    'Teacup Temperature': 'teacup_temperature',
    'FINAL TIME': 'final_time',
    'INITIAL TIME': 'initial_time',
    'SAVEPER': 'saveper',
    'TIME STEP': 'time_step'}


@cache('run')
def characteristic_time():
    """
    Characteristic Time

    Minutes


    """
    return 10


@cache('step')
def heat_loss_to_room():
    """
    Heat Loss to Room

    Degrees/Minute

    This is the rate at which heat flows from the cup into the room. We can 
        ignore it at this point.
    """
    return (teacup_temperature() - room_temperature()) / characteristic_time()


@cache('run')
def room_temperature():
    """
    Room Temperature




    """
    return 70


@cache('step')
def teacup_temperature():
    """
    Teacup Temperature

    Degrees


    """
    return integ_teacup_temperature()


@cache('run')
def final_time():
    """
    FINAL TIME

    Minute

    The final time for the simulation.
    """
    return 30


@cache('run')
def initial_time():
    """
    INITIAL TIME

    Minute

    The initial time for the simulation.
    """
    return 0


@cache('step')
def saveper():
    """
    SAVEPER

    Minute [0,?]

    The frequency with which output is stored.
    """
    return time_step()


@cache('run')
def time_step():
    """
    TIME STEP

    Minute [0,?]

    The time step for the simulation.
    """
    return 0.125


integ_teacup_temperature = functions.Integ(lambda: -heat_loss_to_room(), lambda: 180)
