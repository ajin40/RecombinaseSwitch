class Parent:
    def __init__(self, parameters, true_values):
        # Parent.parameters is a list of the parameters used as input
        self.parameters = parameters

        # Parent.evaluator is what the Parent outputs for the fitness function.
        self.evaluator = sum(parameters) / len(parameters)
        self.true_values = true_values

    def GetParameters(self):
        return self.parameters

    def GetTrueValues(self):
        return self.true_values

    def GetFitness(self):
        return ((sum(self.evaluator - self.true_values)) ** 2) ** (1 / 2)
