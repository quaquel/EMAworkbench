"""

This example is a proof of principle for how MESA models can be
controlled using the ema_workbench.

"""

# Import EMA Workbench modules
from ema_workbench import (
    ReplicatorModel,
    RealParameter,
    BooleanParameter,
    IntegerParameter,
    Constant,
    ArrayOutcome,
    perform_experiments,
    save_results,
    ema_logging,
)

# Necessary packages for the model
import math
from enum import Enum
import mesa
import networkx as nx

# MESA demo model "Virus on a Network", from https://github.com/projectmesa/mesa-examples/blob/d16736778fdb500a3e5e05e082b27db78673b562/examples/virus_on_network/virus_on_network/model.py


class State(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RESISTANT = 2


def number_state(model, state):
    return sum(1 for a in model.grid.get_all_cell_contents() if a.state is state)


def number_infected(model):
    return number_state(model, State.INFECTED)


def number_susceptible(model):
    return number_state(model, State.SUSCEPTIBLE)


def number_resistant(model):
    return number_state(model, State.RESISTANT)


class VirusOnNetwork(mesa.Model):
    """A virus model with some number of agents"""

    def __init__(
        self,
        num_nodes=10,
        avg_node_degree=3,
        initial_outbreak_size=1,
        virus_spread_chance=0.4,
        virus_check_frequency=0.4,
        recovery_chance=0.3,
        gain_resistance_chance=0.5,
    ):
        self.num_nodes = num_nodes
        prob = avg_node_degree / self.num_nodes
        self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=prob)
        self.grid = mesa.space.NetworkGrid(self.G)
        self.schedule = mesa.time.RandomActivation(self)
        self.initial_outbreak_size = (
            initial_outbreak_size if initial_outbreak_size <= num_nodes else num_nodes
        )
        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance

        self.datacollector = mesa.DataCollector(
            {
                "Infected": number_infected,
                "Susceptible": number_susceptible,
                "Resistant": number_resistant,
            }
        )

        # Create agents
        for i, node in enumerate(self.G.nodes()):
            a = VirusAgent(
                i,
                self,
                State.SUSCEPTIBLE,
                self.virus_spread_chance,
                self.virus_check_frequency,
                self.recovery_chance,
                self.gain_resistance_chance,
            )
            self.schedule.add(a)
            # Add the agent to the node
            self.grid.place_agent(a, node)

        # Infect some nodes
        infected_nodes = self.random.sample(list(self.G), self.initial_outbreak_size)
        for a in self.grid.get_cell_list_contents(infected_nodes):
            a.state = State.INFECTED

        self.running = True
        self.datacollector.collect(self)

    def resistant_susceptible_ratio(self):
        try:
            return number_state(self, State.RESISTANT) / number_state(self, State.SUSCEPTIBLE)
        except ZeroDivisionError:
            return math.inf

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in range(n):
            self.step()


class VirusAgent(mesa.Agent):
    def __init__(
        self,
        unique_id,
        model,
        initial_state,
        virus_spread_chance,
        virus_check_frequency,
        recovery_chance,
        gain_resistance_chance,
    ):
        super().__init__(unique_id, model)

        self.state = initial_state

        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance

    def try_to_infect_neighbors(self):
        neighbors_nodes = self.model.grid.get_neighborhood(self.pos, include_center=False)
        susceptible_neighbors = [
            agent
            for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)
            if agent.state is State.SUSCEPTIBLE
        ]
        for a in susceptible_neighbors:
            if self.random.random() < self.virus_spread_chance:
                a.state = State.INFECTED

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

    def try_check_situation(self):
        if (self.random.random() < self.virus_check_frequency) and (self.state is State.INFECTED):
            self.try_remove_infection()

    def step(self):
        if self.state is State.INFECTED:
            self.try_to_infect_neighbors()
        self.try_check_situation()


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
):
    # Initialising the model
    virus_on_network = VirusOnNetwork(
        num_nodes=num_nodes,
        avg_node_degree=avg_node_degree,
        initial_outbreak_size=initial_outbreak_size,
        virus_spread_chance=virus_spread_chance,
        virus_check_frequency=virus_check_frequency,
        recovery_chance=recovery_chance,
        gain_resistance_chance=gain_resistance_chance,
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

    # Define the number of replications
    model.replications = 5

    # Run experiments with the aforementioned parameters and outputs
    results = perform_experiments(models=model, scenarios=20)

    # Get the results
    experiments, outcomes = results
