"""
Python model "Teacup.py"
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
    'Characteristic Time': 'characteristic_time',
    'Heat Loss to Room': 'heat_loss_to_room',
    'Room Temperature': 'room_temperature',
    'Teacup Temperature': 'teacup_temperature',
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


@cache('run')
def characteristic_time():
    """
    Real Name: b'Characteristic Time'
    Original Eqn: b'10'
    Units: b'Minutes'
    Limits: (None, None)
    Type: constant

    b''
    """
    return 10


@cache('step')
def heat_loss_to_room():
    """
    Real Name: b'Heat Loss to Room'
    Original Eqn: b'(Teacup Temperature - Room Temperature) / Characteristic Time'
    Units: b'Degrees/Minute'
    Limits: (None, None)
    Type: component

    b'This is the rate at which heat flows from the cup into the room. We can \\n    \\t\\tignore it at this point.'
    """
    return (teacup_temperature() - room_temperature()) / characteristic_time()


@cache('run')
def room_temperature():
    """
    Real Name: b'Room Temperature'
    Original Eqn: b'70'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 70


@cache('step')
def teacup_temperature():
    """
    Real Name: b'Teacup Temperature'
    Original Eqn: b'INTEG ( -Heat Loss to Room, 180)'
    Units: b'Degrees'
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_teacup_temperature()


@cache('run')
def final_time():
    """
    Real Name: b'FINAL TIME'
    Original Eqn: b'30'
    Units: b'Minute'
    Limits: (None, None)
    Type: constant

    b'The final time for the simulation.'
    """
    return 30


@cache('run')
def initial_time():
    """
    Real Name: b'INITIAL TIME'
    Original Eqn: b'0'
    Units: b'Minute'
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
    Units: b'Minute'
    Limits: (0.0, None)
    Type: component

    b'The frequency with which output is stored.'
    """
    return time_step()


@cache('run')
def time_step():
    """
    Real Name: b'TIME STEP'
    Original Eqn: b'0.125'
    Units: b'Minute'
    Limits: (0.0, None)
    Type: constant

    b'The time step for the simulation.'
    """
    return 0.125


_integ_teacup_temperature = functions.Integ(lambda: -heat_loss_to_room(), lambda: 180)
