import matplotlib.pyplot as plt

class line_plot :

    def __init__(self, data, name, l_rate, activation_function):
        self.data = data
        self.name = name
        self.learning_rate = str(l_rate)
        self.activation_function = activation_function
        return

    def showing(self):
        plt.plot(list(range(0, 10002, 100)), self.data, marker='p')
        plt.title("{}\n(l_rate : {}, act_func : {})".format(self.name, self.learning_rate, self.activation_function))
        plt.xlabel('Episodes')
        plt.ylabel('Success Rate')
        plt.show()
