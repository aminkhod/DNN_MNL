from Beta import Beta
import pandas as pd


class Formula:
    formulaList = []
    formulaPair = []
    dataFrame = pd.DataFrame()


    def __init__(self, args, errorWeight=1):
        self.errorWeight = errorWeight
        for arg in args:
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

    def createFormulaDataset():
        Formula.dataFrame = pd.DataFrame()
        ## Creating a local database based on formula variables.
        z = list(Formula.dataFrame.columns)
        for form in Formula.formulaList:
            for arg in form.args:
                # print(type(arg))
                if not isinstance(arg, Beta):
                    z.append(arg[1].name)
                    Formula.dataFrame = pd.concat([Formula.dataFrame, arg[1]], axis=1)
                    Formula.dataFrame.columns = z

    def get_args(self):
        text = ''
        for arg in self.args:
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
