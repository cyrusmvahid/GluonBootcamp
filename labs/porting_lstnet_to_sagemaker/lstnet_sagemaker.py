import mxnet as mx
from mxnet import nd, gluon, autograd, kv
import numpy as np
from mxnet.gluon import nn, rnn
import os
from lstnet import LSTNet
from timeseriesdataset import TimeSeriesData, TimeSeriesDataset
import re
import time

def get_first_file_path_in_dir(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            return os.path.join(input_dir,filename)
    return None

def load_file(file_path,hyperparameters):
    print('Loading file {}'.format(file_path))
    if not file_path:
        print('Could not load data file: ' + file_path)
        return None
    ts_data = TimeSeriesData(file_path,
        window=hyperparameters['window'],
        horizon=hyperparameters['horizon'],
        train_ratio=1.0)
    return ts_data

def train(
    hyperparameters,
    input_data_config,
    channel_input_dirs,
    output_data_dir,
    model_dir,
    num_gpus,
    num_cpus,
    hosts,
    current_host,
    **kwargs):

    """
    [Required]

    Runs Apache MXNet training. Amazon SageMaker calls this function with information
    about the training environment. When called, if this function returns an
    object, that object is passed to a save function.  The save function
    can be used to serialize the model to the Amazon SageMaker training job model
    directory.

    The **kwargs parameter can be used to absorb any Amazon SageMaker parameters that
    your training job doesn't need to use. For example, if your training job
    doesn't need to know anything about the training environment, your function
    signature can be as simple as train(**kwargs).

    Amazon SageMaker invokes your train function with the following python kwargs:

    Args:
        - hyperparameters: The Amazon SageMaker Hyperparameters dictionary. A dict
            of string to string.
        - input_data_config: The Amazon SageMaker input channel configuration for
            this job.
        - channel_input_dirs: A dict of string-to-string maps from the
            Amazon SageMaker algorithm input channel name to the directory containing
            files for that input channel. Note, if the Amazon SageMaker training job
            is run in PIPE mode, this dictionary will be empty.
        - output_data_dir:
            The Amazon SageMaker output data directory. After the function returns, data written to this
            directory is made available in the Amazon SageMaker training job
            output location.
        - model_dir: The Amazon SageMaker model directory. After the function returns, data written to this
            directory is made available to the Amazon SageMaker training job
            model location.
        - num_gpus: The number of GPU devices available on the host this script
            is being executed on.
        - num_cpus: The number of CPU devices available on the host this script
            is being executed on.
        - hosts: A list of hostnames in the Amazon SageMaker training job cluster.
        - current_host: This host's name. It will exist in the hosts list.
        - kwargs: Other keyword args.

    Returns:
        - (object): Optional. An Apache MXNet model to be passed to the model
            save function. If you do not return anything (or return None),
            the save function is not called.
    """

    train_file_path = get_first_file_path_in_dir(channel_input_dirs['train'])
    print('Train file path {}'.format(train_file_path))
    test_file_path = get_first_file_path_in_dir(channel_input_dirs['test'])
    print('Test file path {}'.format(test_file_path))
    ts_data_train = load_file(train_file_path,hyperparameters)
    ts_data_test = load_file(test_file_path,hyperparameters)

    ctx = [mx.cpu(i) for i in range(num_cpus)]
    if num_gpus > 0:
        ctx = ctx = [mx.gpu(i) for i in range(num_gpus)]
    print('Running on {}'.format(ctx))
    print('Hosts {}'.format(hosts))
    print('Current Host {}'.format(current_host))

    net = LSTNet(
        num_series=ts_data_train.num_series,
        conv_hid=hyperparameters['conv_hid'],
        gru_hid=hyperparameters['gru_hid'],
        skip_gru_hid=hyperparameters['skip_gru_hid'],
        skip=hyperparameters['skip'],
        ar_window=hyperparameters['ar_window'])

    net.initialize(init=mx.init.Xavier(factor_type="in", magnitude=2.34), ctx=ctx)

    kvstore = 'local'
    if len(hosts) == 1:
        kvstore = 'device' if num_gpus > 0 else 'local'
    else:
        kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'
    print('kvstore {}'.format(kvstore))
    store = kv.create(kvstore)
    trainer = gluon.Trainer(net.collect_params(),
        kvstore=store,
        optimizer='adam',
        optimizer_params={'learning_rate': hyperparameters['learning_rate'], 'clip_gradient': hyperparameters['clip_gradient']})

    batch_size = hyperparameters['batch_size']
    train_data_loader = gluon.data.DataLoader(
        ts_data_train.train, batch_size=batch_size, shuffle=True, num_workers=16, last_batch='discard')
    test_data_loader = gluon.data.DataLoader(
        ts_data_test.train, batch_size=batch_size, shuffle=True, num_workers=16, last_batch='discard')

    epochs = hyperparameters['epochs']
    print("Training Start")
    metric = mx.metric.RMSE()
    tic = time.time()
    for e in range(epochs):
        metric.reset()
        epoch_start_time = time.time()
        for data, label in train_data_loader:
            batch_forward_backward(data,label,ctx,net,trainer,batch_size,metric)
        name, value = metric.get()
        print("Epoch {}: {} {} time {:.4f} s".format(e, name, value, time.time()-epoch_start_time))

    # Calculate the test RMSE when training has finished
    validate(train_data_loader,metric,ctx,net)

    print("Total training time: {}".format(time.time()-tic))

    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    net.save_params(os.path.join(output_data_dir,'lstnet_params.params'))
    print("Training End")
    return

def validate(data_loader,metric,ctx,net):
    metric.reset()
    for data, label in data_loader:
        data = data.as_in_context(ctx[0])
        label = label.as_in_context(ctx[0])
        metric.update(label,net(data))
    name, value = metric.get()
    print('Final {} {}'.format(name,value))
    return name,value

def batch_forward_backward(data, label, ctx, net, trainer, batch_size, metric):
    l1 = gluon.loss.L1Loss()
    data = data.as_in_context(ctx[0])
    label = label.as_in_context(ctx[0])
    with autograd.record():
        z = net(data)
        loss = l1(z,label)
    autograd.backward(loss)
    trainer.step(batch_size)
    metric.update(label,z)
