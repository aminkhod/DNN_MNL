from Beta import Beta
import pandas as pd

class Formula:
    formulaList = []
    dataFrame = pd.DataFrame()
    def __init__(self, *args):

        # self.error = False
        self.args = args
        # if name not in Formula.FormulaList:
        Formula.formulaList.append(self)

        z = list(Formula.dataFrame.columns)
        for arg in self.args:
            # print(type(arg))
            if not isinstance(arg, Beta):
                z.append(arg[1].name)
                Formula.dataFrame = pd.concat([Formula.dataFrame, arg[1]], axis=1)
                Formula.dataFrame.columns = z

        # else:
        #     self.error = True
        #     raise Exception('The Formula name is taken before. Please s elect unique name for your Formula.')
        #     self.__del__()
        # self.name = name
        # self.initialValue = initial_value
        # self.constraint = constraint

    def get_args(self):
        text = ''
        # print(self.args[0].name)
        for arg in self.args:
            # print(type(arg))
            if isinstance(arg, Beta):
                text += '(' + arg.name + ')'
                text += ' + '
            else:
                text += '(' + arg[0].name
                try:
                    text += ' * ' + arg[1].name + ')'
                except:
                    text += ' * ' + str(arg[1]) + ')'
                text += ' + '

        text = text[:-3]

        return text

