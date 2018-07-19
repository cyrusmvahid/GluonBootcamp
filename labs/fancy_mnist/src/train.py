import mxnet as mx
from mxnet import gluon, autograd
from utilities import visual_utilities as vu
from utilities import TradingHistoryList
from network import Network
from data_iter import DataIterBuilder

class Train():
    def __init__(self, network, train_iter, test_iter, eval_metrics_list=(mx.metric.RMSE(), mx.metric.Accuracy()), initializer=mx.init.Xavier(magnitude=2.43), trainer=None, ctx_list=(mx.cpu()), batch_size=64):
        self._eval_metrics_list = eval_metrics_list
        self.ctx_list = list(ctx_list)
        self.network = network
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.batch_size = batch_size

        self._eval_metrics = self._create_eval_metrics()
        self._optimizer = 'adam'
        self._learning_rate = 0.001
        self._training_history_list = self._create_training_history_list()

        self.network.network.collect_params().initialize(initializer, ctx=self.ctx_list, force_reinit=True)

        if trainer == None:
            self.trainer = gluon.Trainer(self.network.network.collect_params(), self._optimizer, {'learning_rate': self._learning_rate})
        else:
            self.trainer = trainer


    @property
    def train_history(self):
        return self._train_history


    @property
    def optimizer(self):
        return self._optimizer

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def eval_metrics(self):
        return self._eval_metrics

    @property
    def training_history_list(self):
        return self._training_history_list

    def _create_eval_metrics(self):
        eval_metrics = mx.metric.CompositeEvalMetric()
        for cm in list(self._eval_metrics_list):
            eval_metrics.add(cm)
        return eval_metrics

    def _create_training_history_list(self):
        metric_list = self._eval_metrics.get()[0]
        thl = TradingHistoryList(metric_list=metric_list)
        return thl


    def eval_accuracy(self, mode='test'):
        metrics_list = self._eval_metrics_list
        data_iter = self.test_iter
        if mode != 'test':
            data_iter = self.train_iter
        acc = mx.metric.CompositeEvalMetric()
        for cm in metrics_list:
            acc.add(cm)
        for i, (data, label) in enumerate(data_iter):
            X = gluon.utils.split_and_load(data, self.ctx_list)
            Y = gluon.utils.split_and_load(label, self.ctx_list)
            y_hat = [self.network.network(x) for x in X]
            acc.update(preds=y_hat, labels=Y)
        return acc


    def train(self, num_epocs=10):
        log_interval = 100

        for e in range(num_epocs):
            self.eval_metrics.reset()
            for i, (data, label) in enumerate(self.train_iter):
                X = gluon.utils.split_and_load(data, self.ctx_list)
                Y = gluon.utils.split_and_load(label, self.ctx_list)
                with autograd.record():
                    yhats = [self.network.network(x) for x in X]
                    losses = [self.network.loss_fn(yhat, y) for yhat, y in zip(yhats, Y)]
                autograd.backward(losses)
                self.trainer.step(self.batch_size)
                self.eval_metrics.update(preds=yhats, labels=Y)
                if log_interval and not (i + 1) % log_interval:
                    test_metrics = self.eval_accuracy()
                    print("\nEPOCH:{}; TRAIN_DATA: BATCH:{}; Metrics:{}".format(e, i, self.eval_metrics.get_name_value()))
                    print("EPOCH:{}; TEST_DATA: BATCH:{}; Metrics:{}".format(e, i, test_metrics.get_name_value()))

            update_list = []
            for j in range(len(self.eval_metrics.get()[0])):
                train_val = self.eval_metrics.get()[1][j]
                test_val = test_metrics.get()[1][j]
                update_list.append([train_val, test_val])
            self.training_history_list.update(values=update_list)


if __name__ == "__main__":
    data_iter = DataIterBuilder()
    train_iter, test_iter = data_iter.get_data_iter(mode='mnist')
    network = Network()
    a = network.network.collect_params()
    ctx_list = [mx.gpu(i) for i in range(4)]
    t = Train(ctx_list=ctx_list, train_iter=train_iter, test_iter=test_iter, network=network)
    print("EVAL METRICS: {}".format(t.eval_metrics))
    for th in t.training_history_list.trainings_histories:
        print(th)
    t.train(num_epocs=10)
    for th in t.training_history_list.trainings_histories:
        print(th.history)
    t.training_history_list.plot()

'''
    def plot_results(self, colors=None, save_path='~/plot.png'):
        if colors == None:
            colors = vu.random_hex_colours(len(self.train_history.labels))
        self.train_history.plot(self.train_history.labels, colors=colors, save_path=save_path)
        
        
data_iter = DataIterBuilder()
train_iter, test_iter = data_iter.get_data_iter(mode='mnist')
network = Network()
a = network.network.collect_params()
ctx_list = (mx.gpu(i) for i in range(4))
t = Train(ctx_list=ctx_list, train_iter=train_iter, test_iter=test_iter, network=network)
t.train(num_epocs=10)
t.plot_results()
print(t.train_history.history)

print(t.train_history.labels)

#t.hello()
'''



