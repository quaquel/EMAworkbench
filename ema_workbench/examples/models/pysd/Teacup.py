"""
Python model 'Teacup.py'
Translated using PySD
"""

from pathlib import Path
import numpy as np

from pysd.py_backend.statefuls import Integ
from pysd import Component

__pysd_version__ = "3.12.0"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 30,
    "time_step": lambda: 0.125,
    "saveper": lambda: time_step(),
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    """
    Current time of the model.
    """
    return __data["time"]()


@component.add(name="FINAL TIME", units="Minute", comp_type="Constant", comp_subtype="Normal")
def final_time():
    """
    The final time for the simulation.
    """
    return __data["time"].final_time()


@component.add(name="INITIAL TIME", units="Minute", comp_type="Constant", comp_subtype="Normal")
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="SAVEPER",
    units="Minute",
    limits=(0.0, np.nan),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time_step": 1},
)
def saveper():
    """
    The frequency with which output is stored.
    """
    return __data["time"].saveper()


@component.add(
    name="TIME STEP",
    units="Minute",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def time_step():
    """
    The time step for the simulation.
    """
    return __data["time"].time_step()


#######################################################################
#                           MODEL VARIABLES                           #
#######################################################################


@component.add(
    name="Characteristic Time",
    units="Minutes",
    comp_type="Constant",
    comp_subtype="Normal",
)
def characteristic_time():
    return 10


@component.add(
    name="Heat Loss to Room",
    units="Degrees/Minute",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "teacup_temperature": 1,
        "room_temperature": 1,
        "characteristic_time": 1,
    },
)
def heat_loss_to_room():
    """
    This is the rate at which heat flows from the cup into the room. We can ignore it at this point.
    """
    return (teacup_temperature() - room_temperature()) / characteristic_time()


@component.add(name="Room Temperature", comp_type="Constant", comp_subtype="Normal")
def room_temperature():
    return 70


@component.add(
    name="Teacup Temperature",
    units="Degrees",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_teacup_temperature": 1},
    other_deps={"_integ_teacup_temperature": {"initial": {}, "step": {"heat_loss_to_room": 1}}},
)
def teacup_temperature():
    return _integ_teacup_temperature()


_integ_teacup_temperature = Integ(
    lambda: -heat_loss_to_room(), lambda: 180, "_integ_teacup_temperature"
)
