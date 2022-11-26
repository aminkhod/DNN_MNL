import numpy as np
class Beta:

    BetaList = []
    BetaName = []
    # global BetaList

    def __init__(self, name, initial_value=np.random.normal(0, 1), constraint=0):
        self.betaName = name
        if initial_value == None:
            initial_value = np.random.normal(0, 1)
        self.initial_value = initial_value
        self.constraint = constraint

        if name not in Beta.BetaName:
            Beta.BetaList.append(self)
            Beta.BetaName.append(name)

        else:
            Beta.BetaList[Beta.BetaName.index(name)].betaName = name
            Beta.BetaList[Beta.BetaName.index(name)].initial_value = initial_value
            Beta.BetaList[Beta.BetaName.index(name)].constraint = constraint




    def get_name(self):
        return self.betaName

    def get_constraint(self):
        return self.constraint

    def get_initialValue(self):
        return self.initialValue
