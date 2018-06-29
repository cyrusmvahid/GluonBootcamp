import random
import os
from gluoncv.utils import TrainingHistory
from mxnet import metric

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
    def __init__(self, metric_list=(metric.RMSE(), metric.Accuracy()), plot_location=os.getcwd() + "/"):
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
        composite_eval_metrics = []
        th = []
        for i in range(len(ems)):
            titles = []
            composite_eval_metrics.append(metric.CompositeEvalMetric())
            titles.append(ems[i].get()[0])
            titles.append(ems[i].get()[0] + "_val")
            th.append(TrainingHistory(titles))
        return tuple(th)

    def _create_save_path(self):
        ems = list(self.metric_list)
        plot_names = []
        for m in ems:
            plot_names.append(self._plot_location + m.get()[0] + "-plot.png")
        return plot_names

    def update(self, train, val):
        for th in self._trainings_histories:
            th.update([train] + [val])

    def plot(self):
        ths = list(self._trainings_histories)
        for th in ths:
            print(th.labels)
            th.plot(th.labels)


tshs = TradingHistoryList()
print(tshs.trainings_histories)
res1 = [.1, .2, .3, .4, .5]

for i in res1:
    tshs.update(i, i * 2)

tshs.plot()
for h in tshs.trainings_histories:
    print(h.history)