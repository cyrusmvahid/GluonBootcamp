import mxnet as mx
from mxnet import nd, autograd, gluon
import os

def get_first_file_path_in_dir(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            return os.path.join(input_dir,filename)
        
def get_file_path(input_dir,current_host,hosts):
    file_path = None
    if len(hosts) <= 1:
        file_path = get_first_file_path_in_dir(input_dir)
    else:
        numbers_in_host_name = re.findall('[0-9]+', current_host)
        index = int(numbers_in_host_name[0]) - 1
        file_path = '{}/{}'.format(input_dir,index)
    return file_path

def train(hyperparameters,channel_input_dirs,num_gpus,hosts,current_host,**kwargs):

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
    
    train_file_path = get_file_path(channel_input_dirs['train'],current_host,hosts)
    print('Train file path {}'.format(train_file_path))
    train_dict = mx.nd.load(train_file_path)
    X = train_dict['X']
    y = train_dict['y']
    
    num_examples = len(X)
    print('Number of examples {}'.format(num_examples))
    
    batch_size = hyperparameters['batch_size']
    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=True)
    
    ctx = mx.gpu(0)
    
    net = gluon.nn.Dense(1)
    net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=ctx)
    square_loss = gluon.loss.L2Loss()

    kvstore = 'local'
    if num_gpus > 0:
        if len(hosts) > 1:
            kvstore = 'dist_device_sync'
        else:
            kvstore = 'device'
    print('kvstore {}'.format(kvstore))
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.0001}, kvstore=kvstore)
    
    epochs = hyperparameters['epochs']
    loss_sequence = []
    num_batches = num_examples / batch_size

    for e in range(epochs):
        cumulative_loss = 0
        # inner loop
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.mean(loss).asscalar()
        print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))
        loss_sequence.append(cumulative_loss)