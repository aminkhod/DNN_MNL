class Formula:
    formulaList = []

    def __init__(self, *args):

        # self.error = False
        self.args = args
        # if name not in Formula.FormulaList:
        Formula.formulaList.append(self)
        # else:
        #     self.error = True
        #     raise Exception('The Formula name is taken before. Please s elect unique name for your Formula.')
        #     self.__del__()
        # self.name = name
        # self.initialValue = initial_value
        # self.constraint = constraint

    def get_args(self):
        text = ''
        for arg in self.args:
            # print(arg[0])
            text += '(' + arg[0].name
            try:
                text += ' * ' + arg[1].name + ')'
            except:
                text += ' * ' + str(arg[1]) + ')'
            text += ' + '
        text = text[:-3]

        return text

