from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import MNIST
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon import nn, Trainer
from mxnet import initializer, init, cpu, gpu, autograd
from mxnet.metric import RMSE, Accuracy, CompositeEvalMetric
from numpy import float32
from mxnet.gluon.utils import split_and_load
from gluoncv.utils import TrainingHistory
NUM_GPU=4
ctx =[gpu(i) for i in range(NUM_GPU)]
bs = 64

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128, activation='relu'))
    net.add(nn.Dense(64, activation='relu'))
    net.add(nn.Dense(10))
loss_fn = SoftmaxCrossEntropyLoss()
net.collect_params().initialize(init.Xavier(magnitude=2.4), ctx=ctx)
trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001, 'momentum': .9})
def transform(data, label):
    return data.astype(float32)/255, label.astype(float32)
train_data = DataLoader(MNIST(train=True, transform=transform), batch_size=bs, shuffle=True)

train_hist = TrainingHistory(['rmse', 'accuracy'])
epochs = 1
ml = CompositeEvalMetric()
mc = [RMSE(), Accuracy()]
for c in mc:
    ml.add(c)
for e in range(epochs):
    ml.reset()
    for i, (data, label) in enumerate(train_data):
        #data = data.as_in_context(ctx)
        #label = label.as_in_context(ctx)
        data = split_and_load(data, ctx)
        label = split_and_load(label, ctx)
        with autograd.record():
            #output = net(data)
            output = [net(X) for X in data]
            #loss = loss_fn(output, label)
            loss = [loss_fn(yhat, Y) for yhat, Y in zip(output, label)]
        autograd.backward(loss)
        trainer.step(bs)
        ml.update(labels=label, preds=output)
        train_hist.update(ml.get()[1])
    print("epoch: {}; metrics: {}".format(e, ml.get_name_value()))
    a = ml.get()[0]
train_hist.plot(labels=ml.get()[0], colors=['red', 'green'], save_path='~/plot.png')


print(train_hist)