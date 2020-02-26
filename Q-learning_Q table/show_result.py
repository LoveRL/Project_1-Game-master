import matplotlib.pyplot as plt

class line_plot :

    def __init__(self, data, name):
        self.data = data
        self.name = name
        return

    def showing(self):
        plt.plot(list(range(0, 10002, 100)), self.data, marker='p')
        plt.title(self.name)
        plt.xlabel('Episodes')
        plt.ylabel('Success Rate')
        plt.show()
