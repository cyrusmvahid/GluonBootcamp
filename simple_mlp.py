import mxnet as mx
from mxnet import gluon, nd, autograd, metric
import numpy as np
from gluoncv.utils import TrainingHistory



# creating data iterators for mnist
class DataIterBuilder():
    def __init__(self, transform=None, batch_size=64):

        self.batch_size = batch_size
        if transform != None:
            self.transform = transform
        else:
            def transform(data, label):
                return nd.transpose(data.astype(np.float32)/255, (2,0,1)), label.astype(np.float32)
                #return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255, label.astype(np.float32)
            self.transform = transform

    def get_data_iter(self, mode="fashion"):
        if mode == "fashion":
            dataset = gluon.data.vision.FashionMNIST
        elif mode == "mnist":
            dataset = gluon.data.vision.MNIST
        else:
            raise ValueError("valid modes are 'fashion' and mnist. {} is not a vlaid mode".format(mode))
        train = gluon.data.DataLoader(dataset(train=True, transform=self.transform), shuffle=True, batch_size=self.batch_size)
        test = gluon.data.DataLoader(dataset(train=False, transform=self.transform), shuffle=False, batch_size=self.batch_size)
        return train, test



class Network():
    def __init__(self, mode='mlp'):
        if mode == 'mlp':
            def mlp(self):
                net = gluon.nn.HybridSequential()
                with net.name_scope():
                    net.add(gluon.nn.Dense(units=128, activation='relu'))
                    net.add(gluon.nn.Dense(units=64, activation='relu'))
                    net.add(gluon.nn.Dense(10))
                return net
            self.network = mlp(self)
        elif mode == 'lenet':
            def lenet(self):
                net = gluon.nn.HybridSequential()
                with net.name_scope():
                    net.add(gluon.nn.Conv2D(channels=20, filter=5, activation='relu'))
                    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
                    net.add(gluon.nn.Conv2D(channels=50, filter=3, activation='relu'))
                    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
                    net.add(gluon.nn.Dense(128, activation='relu'))
                    net.add(gluon.nn.Dense(54,activation='relu'))
                    net.add(gluon.nn.Dense(2))
                return net
            self.network = lenet(self)
        else:
            raise ValueError("valid modes are 'mlp' and '")

    def loss_fn(self):
        return gluon.loss.SoftmaxCrossEntropyLoss()

class Train():
    def __init__(self, network, train_iter, test_iter, initializer=mx.init.Xavier(magnitude=2.43), trainer=None, eval_metrics_list=(mx.metric.Accuracy(), mx.metric.RMSE()), ctx_list=(mx.cpu()), batch_size=64):
        self.eval_metrics_list = list(eval_metrics_list)
        self.ctx_list = list(ctx_list)
        self.network = network
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.batch_size = batch_size

        self._optimizer = 'adam'
        self._learning_rate = 0.001

        self.network.network.collect_params().initialize(initializer, ctx=self.ctx_list, force_reinit=True)

        if trainer == None:
            self.trainer = gluon.Trainer(self.network.network.collect_params(), self._optimizer, {'learning_rate': self._learning_rate})
        else:
            self.trainer = trainer

        self._train_history = self.__create_train_history()


    @property
    def optimizer(self):
        return self._optimizer

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def train_history(self):
        return self._train_history

    def __create_train_history(self):
        acc = mx.metric.CompositeEvalMetric()
        for cm in self.eval_metrics_list:
            acc.add(cm)
        train_hist = list(acc.get()[0])
        num_elements = len(train_hist)
        for i in range(num_elements):
            train_hist.append(train_hist[i]+"-validation")
        return TrainingHistory(train_hist)

    def eval_accuracy(self, data_iter, metrics_list=None):
        if metrics_list == None:
            metrics_list = self.eval_metrics_list
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
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        log_interval = 50
        acc = mx.metric.CompositeEvalMetric()
        for cm in self.eval_metrics_list:
            acc.add(cm)
        for e in range(num_epocs):
            acc.reset()
            for i, (data, label) in enumerate(self.train_iter):
                X = gluon.utils.split_and_load(data, self.ctx_list)
                Y = gluon.utils.split_and_load(label, self.ctx_list)
                with autograd.record():
                    yhats = [self.network.network(x) for x in X]
                    losses = [loss_fn(yhat, y) for yhat, y in zip(yhats, Y)]
                autograd.backward(losses)
                self.trainer.step(self.batch_size)
                acc.update(preds=yhats, labels=Y)
                if log_interval and not (i + 1) % log_interval:
                    metrics = acc.get()
                    print("EPOCH:{}; BATCH:{}; Metrics:{}".format(e, i, acc.get_name_value()))

                #print(self.eval_accuracy())







data_iter = DataIterBuilder()
train_iter, test_iter = data_iter.get_data_iter(mode='mnist')
network = Network()
a = network.network.collect_params()
ctx_list = (mx.gpu(i) for i in range(4))
t = Train(ctx_list=ctx_list, train_iter=train_iter, test_iter=test_iter, network=network)
t.train()

#t.hello()
