from Parent import Parent
from MarkovModel import *

class MarkovParent (Parent):
    def __init__(self, num_cells, dox, aba, parameters, true_values):
        # Parent.parameters is a list of the parameters used as input
        self.dox, self.aba = dox, aba
        self.parameters = parameters
        # Parent.evaluator is what the Parent outputs for the fitness function.
        self.simulation_states, self.transition_times = RunAllTransitions(num_cells, dox, aba, parameters)
        self.red_percent, self.yellow_percent = FinalTransitionRate(self.simulation_states, self.transition_times, T=96., dt=1.)
        self.evaluator = [self.red_percent[-1], self.yellow_percent[-1]]
        self.true_values = true_values

    def GetParameters(self):
        return self.parameters

    def GetTrueValues(self):
        return self.true_values

    def GetSimulation(self):
        return [self.simulation_states, self.transition_times]

    def GetFinalValues(self):
        return self.evaluator

    def GetFitness(self):
        return sum((i - j) ** 2 for i, j in zip(self.evaluator, self.true_values)) ** (1 / 2)
