"""Example of using the workbench together with mesa."""

# Import EMA Workbench modules
# Necessary packages for the model
import math
import sys
from enum import Enum

import mesa
import networkx as nx
import numpy as np

from ema_workbench import (
    ArrayOutcome,
    Constant,
    IntegerParameter,
    RealParameter,
    ReplicatorModel,
    ema_logging,
    perform_experiments,
)

# MESA demo model "Virus on a Network", from https://github.com/projectmesa/mesa-examples/blob/d16736778fdb500a3e5e05e082b27db78673b562/examples/virus_on_network/virus_on_network/model.py


class State(Enum):
    """Possible states of an agent."""

    SUSCEPTIBLE = 0
    INFECTED = 1
    RESISTANT = 2


def number_state(model, state):
    return sum(1 for a in model.grid.agents if a.state is state)


def number_infected(model):
    return number_state(model, State.INFECTED)


def number_susceptible(model):
    return number_state(model, State.SUSCEPTIBLE)


def number_resistant(model):
    return number_state(model, State.RESISTANT)


class VirusOnNetwork(mesa.Model):
    """A virus model with some number of agents."""

    def __init__(
        self,
        num_nodes=10,
        avg_node_degree=3,
        initial_outbreak_size=1,
        virus_spread_chance=0.4,
        virus_check_frequency=0.4,
        recovery_chance=0.3,
        gain_resistance_chance=0.5,
        rng=None
    ):
        super().__init__(rng=rng)
        self.num_nodes = num_nodes
        prob = avg_node_degree / self.num_nodes
        graph = nx.erdos_renyi_graph(n=self.num_nodes, p=prob)

        self.grid = mesa.discrete_space.Network(graph, capacity=1, random=self.random)

        self.initial_outbreak_size = (
            initial_outbreak_size if initial_outbreak_size <= num_nodes else num_nodes
        )

        self.datacollector = mesa.DataCollector(
            {
                "Infected": number_infected,
                "Susceptible": number_susceptible,
                "Resistant": number_resistant,
            }
        )

        # Create agents
        VirusAgent.create_agents(
            self,
            num_nodes,
            State.SUSCEPTIBLE,
            virus_spread_chance,
            virus_check_frequency,
            recovery_chance,
            gain_resistance_chance,
            list(self.grid.all_cells),
        )

        # Infect some nodes
        infected_nodes = mesa.discrete_space.CellCollection(
            self.random.sample(list(self.grid.all_cells), self.initial_outbreak_size),
            random=self.random,
        )
        for a in infected_nodes.agents:
            a.state = State.INFECTED

        self.running = True
        self.datacollector.collect(self)

    def resistant_susceptible_ratio(self):
        try:
            return number_state(self, State.RESISTANT) / number_state(
                self, State.SUSCEPTIBLE
            )
        except ZeroDivisionError:
            return math.inf

    def step(self):
        # collect data
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

    def run_model(self, n):
        for _ in range(n):
            self.step()


class VirusAgent(mesa.discrete_space.FixedAgent):
    def __init__(
        self,
        model,
        initial_state,
        virus_spread_chance,
        virus_check_frequency,
        recovery_chance,
        gain_resistance_chance,
        cell
    ):
        super().__init__( model)

        self.state = initial_state

        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance
        self.cell = cell

    def try_infect_neighbors(self):
        for agent in self.cell.neighborhood.agents:
            if (agent.state is State.SUSCEPTIBLE) and (
                self.random.random() < self.virus_spread_chance
            ):
                agent.state = State.INFECTED

    def try_gain_resistance(self):
        if self.random.random() < self.gain_resistance_chance:
            self.state = State.RESISTANT

    def try_remove_infection(self):
        # Try to remove
        if self.random.random() < self.recovery_chance:
            # Success
            self.state = State.SUSCEPTIBLE
            self.try_gain_resistance()
        else:
            # Failed
            self.state = State.INFECTED

    def check_situation(self):
        if (self.random.random() < self.virus_check_frequency) and (
            self.state is State.INFECTED
        ):
            self.try_remove_infection()

    def step(self):
        if self.state is State.INFECTED:
            self.try_infect_neighbors()
        self.check_situation()


# Setting up the model as a function
def model_virus_on_network(
    num_nodes=1,
    avg_node_degree=1,
    initial_outbreak_size=1,
    virus_spread_chance=1,
    virus_check_frequency=1,
    recovery_chance=1,
    gain_resistance_chance=1,
    steps=10,
    rng=None,
):
    """Run the model once."""
    # Initialising the model
    virus_on_network = VirusOnNetwork(
        num_nodes=num_nodes,
        avg_node_degree=avg_node_degree,
        initial_outbreak_size=initial_outbreak_size,
        virus_spread_chance=virus_spread_chance,
        virus_check_frequency=virus_check_frequency,
        recovery_chance=recovery_chance,
        gain_resistance_chance=gain_resistance_chance,
        rng=rng
    )

    # Run the model steps times
    virus_on_network.run_model(steps)

    # Get model outcomes
    outcomes = virus_on_network.datacollector.get_model_vars_dataframe()

    # Return model outcomes
    return {
        "Infected": outcomes["Infected"].tolist(),
        "Susceptible": outcomes["Susceptible"].tolist(),
        "Resistant": outcomes["Resistant"].tolist(),
    }


if __name__ == "__main__":
    # Initialize logger to keep track of experiments run
    ema_logging.log_to_stderr(ema_logging.INFO)

    # Instantiate and pass the model
    model = ReplicatorModel("VirusOnNetwork", function=model_virus_on_network)

    # Define model parameters and their ranges to be sampled
    model.uncertainties = [
        IntegerParameter("num_nodes", 10, 100),
        IntegerParameter("avg_node_degree", 2, 8),
        RealParameter("virus_spread_chance", 0.1, 1),
        RealParameter("virus_check_frequency", 0.1, 1),
        RealParameter("recovery_chance", 0.1, 1),
        RealParameter("gain_resistance_chance", 0.1, 1),
    ]

    # Define model parameters that will remain constant
    model.constants = [Constant("initial_outbreak_size", 1), Constant("steps", 30)]

    # Define model outcomes
    model.outcomes = [
        ArrayOutcome("Infected"),
        ArrayOutcome("Susceptible"),
        ArrayOutcome("Resistant"),
    ]

    # Define the number of replications and the seed for each of then
    n_replications = 10
    model.replications = [{"rng": i} for i in np.random.default_rng().integers(sys.maxsize, size=n_replications)]

    # Run experiments with the aforementioned parameters and outputs
    experiments, outcomes  = perform_experiments(models=model, scenarios=20)

