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

            # self.error = True
            # raise Exception('The Beta name is taken before. Please select unique name for your Beta.')
            # self.__del__()
        # print('from class', id(self))




    def get_name(self):
        return self.betaName

    def get_constraint(self):
        return self.constraint

    def get_initialValue(self):
        return self.initialValue

    # def __del__(self):
    #     if self.betaName in Beta.BetaName:
    #         # Beta.BetaList.remove([Beta.BetaName.index(self.betaName)])
    #         # Beta.BetaName.remove(self.betaName)
    #         Beta.BetaList = list(set(Beta.BetaList))
    #         Beta.BetaName = list(set(Beta.BetaName))


        # if self.error:
        #     Beta.BetaList = list(set(Beta.BetaList))
        #     Beta.BetaName = list(set(Beta.BetaName))
        #
        # else:
        #     Beta.BetaName.remove(self.name)
