from mxnet import gluon

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
            self._network = mlp(self)
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
            self._network = lenet(self)
        else:
            raise ValueError("valid modes are 'mlp' and '")

        self._loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    @property
    def network(self):
        return self._network

    @property
    def loss_fn(self):
        return self._loss_fn

if __name__ == '__main__':
    test_net = Network()
    print(test_net.network.collect_params())