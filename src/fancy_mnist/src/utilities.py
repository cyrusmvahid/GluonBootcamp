import random
import os
from gluoncv.utils import TrainingHistory
from mxnet import metric
from mxnet import nd

class visual_utilities():
    @staticmethod
    def random_hex_colours(count):
        colours = []
        r = lambda: random.randint(0, 255)
        for i in range(count):
            colours.append('#%02X%02X%02X' % (r(), r(), r()))
        return colours


import os


class TradingHistoryList():
    def __init__(self, metric_list, plot_location=os.getcwd() + "/"):
        self.metric_list = metric_list
        self._trainings_histories = self._create_train_history()
        self._plot_location = plot_location
        self._save_path = self._create_save_path()

    @property
    def plot_location(self):
        return self._plot_location

    @property
    def plot_names(self):
        return self._save_path

    @property
    def trainings_histories(self):
        return self._trainings_histories

    def _create_train_history(self):
        ems = list(self.metric_list)
        ths = []
        for i in ems:
            ths.append(TrainingHistory([i, i+"_val"]))
        return tuple(ths)

    def _create_save_path(self):
        ems = list(self.metric_list)
        plot_names = []
        for m in ems:
            plot_names.append(self._plot_location + m + "-plot.png")
        return plot_names

    def update(self, values):
        for i in range(len(self.trainings_histories)):
            self.trainings_histories[i].update(values[i])

    def plot(self, colors, ):
        if colors == None:
            colors = visual_utilities.random_hex_colours(2)
        ths = list(self._trainings_histories)
        min_val, max_val = 0, 0
        for th in ths:
            min_val = nd.array(th.history[th.labels[0]]).min().asscalar()
            min_val = min(min_val, nd.array(th.history[th.labels[1]]).min().asscalar()) // 2
            max_val = nd.array(th.history[th.labels[0]]).max().asscalar()
            max_val = max(max_val, nd.array(th.history[th.labels[1]]).max().asscalar()) * 2
            th.plot(labels=th.labels, y_lim=(min_val, max_val), colors=colors, save_path=self._save_path)


if __name__ == '__main__':
    tshs = TradingHistoryList(metric_list=[metric.RMSE().name, metric.Accuracy().name])

    updates = [[0,0], [1,1]]
    tshs.update(updates)
    for th in tshs.trainings_histories:
        print(th.history)
