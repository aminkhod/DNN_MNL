from Beta import Beta
import pandas as pd


class Formula:
    formulaList = []
    formulaPair = []
    dataFrame = pd.DataFrame()

    # nameList = []
    def __init__(self, *args):
        for arg in args:
            # print(type(arg))
            if isinstance(arg, Beta):
                continue
            else:
                try:
                    arg[0].betaName

                    arg[1].name
                except:
                    # self.__del__()
                    raise Exception('Please, write formula in a proper structure, which must contain pairs like:'
                                    ' (weight, input_variable) or single weight like (weight)')

        # self.name = name
        self.args = args
        # if name not in Formula.nameList:
        Formula.formulaPair.append(self)
        # Formula.nameList.append(self.name)
        # else:
        #     Formula.formulaPair[Formula.nameList.index(name)][1].args = args

        # for f in Formula.formulaPair:
        #     print(f[0], f[1].args)
        # for f in Formula.nameList:
        #     print(f)

        # Formula.formulaList.sort()
        # for f in Formula.formulaList:
        # print(f.className)

        # else:
        #     self.error = True
        #     raise Exception('The Formula name is taken before. Please s elect unique name for your Formula.')
        #     self.__del__()
        # self.name = name
        # self.initialValue = initial_value
        # self.constraint = constraint

    def createFormulaDataset():
        ## Creating a local database based on formula variables.
        z = list(Formula.dataFrame.columns)
        for form in Formula.formulaList:
            for arg in form.args:
                # print(type(arg))
                if not isinstance(arg, Beta):
                    z.append(arg[1].betaName)
                    Formula.dataFrame = pd.concat([Formula.dataFrame, arg[1]], axis=1)
                    Formula.dataFrame.columns = z

    def get_args(self):
        text = ''
        # print(self.args[0].name)
        for arg in self.args:
            # print(type(arg))
            if isinstance(arg, Beta):
                text += '(' + arg.betaName + ')'
                text += ' + '
            else:
                text += '(' + arg[0].betaName
                try:
                    text += ' * ' + arg[1].name + ')'
                except:
                    text += ' * ' + str(arg[1]) + ')'
                text += ' + '

        text = text[:-3]

        return text
    def __del__(self):
        print('Formula creation was unsuccessful or Formula has been deleted!')
