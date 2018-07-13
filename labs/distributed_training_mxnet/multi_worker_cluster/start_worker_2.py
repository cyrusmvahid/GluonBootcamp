import os

os.environ.update({
  "DMLC_ROLE": "worker",
  "DMLC_PS_ROOT_URI": "127.0.0.1",
  "DMLC_PS_ROOT_PORT": "9000",
  "DMLC_NUM_SERVER": "1",
  "DMLC_NUM_WORKER": "2",
  "PS_VERBOSE": "2"
})

import mxnet as mx
import logging
import numpy as np
import time
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.DEBUG)

# lineral equation
def f(x):
  # a = 5
  # b = 2
  return 5 * x + 2

# Data
X = np.arange(100, step=0.001)
Y = f(X)

# Split data for taining and evaluation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
kv_store = mx.kv.create('dist_async')

batch_size = 1024
train_iter = mx.io.NDArrayIter(X_train, 
                               Y_train, 
                               batch_size, 
                               shuffle=True,
                               label_name='lin_reg_label')
eval_iter = mx.io.NDArrayIter(X_test, 
                              Y_test, 
                              batch_size, 
                              shuffle=False)

X = mx.sym.Variable('data')
Y = mx.symbol.Variable('lin_reg_label')
fully_connected_layer  = mx.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)
lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

model = mx.mod.Module(
    symbol = lro ,
    data_names=['data'],
    label_names = ['lin_reg_label']# network structure
)
time1 = time.time()

model.fit(train_iter, eval_iter,
            optimizer_params={
                'learning_rate':0.000000002},
            num_epoch=20,
            eval_metric='mae',
            batch_end_callback
                 = mx.callback.Speedometer(batch_size, 20),
            kvstore=kv_store)

time2 = time.time()
print('training took %0.3f ms' % ((time2 - time1) * 1000.0))
