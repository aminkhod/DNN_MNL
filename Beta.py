import numpy as np
class Beta:

    BetaList = []
    # global BetaList

    def __init__(self, name, initial_value=np.random.normal(0, 1), constraint=0):

        self.error = False
        if name not in Beta.BetaList:
            Beta.BetaList.append(name)
        else:
            self.error = True
            raise Exception('The Beta name is taken before. Please s elect unique name for your Beta.')
            self.__del__()
        self.name = name
        self.initialValue = initial_value
        self.constraint = constraint



    def get_name(self):
        return self.name

    def get_constraint(self):
        return self.constraint

    def get_initialValue(self):
        return self.initialValue

    def __del__(self):
        if self.error:
            Beta.BetaList = list(set(Beta.BetaList))
        else:
            Beta.BetaList.remove(self.name)
