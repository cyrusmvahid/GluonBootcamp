import unittest
import lstnet_sagemaker
import os
import mxnet as mx
from mxnet import gluon

class TestLstnet_sagemaker(unittest.TestCase):

    training_data_dir = '../data'
    output_dir = 'output'

    # def test_batch_forward_backward(self):
    #
    #     batch_size = 30
    #     data = mx.ndarray.random_uniform(shape=(batch_size,40,10))
    #     label = mx.ndarray.random_uniform(shape=(batch_size,10))
    #
    #     ctx = [mx.gpu(0),mx.gpu(1)]
    #     net = lstnet_sagemaker.LSTNet(num_series=10,conv_hid=20,gru_hid=30,skip_gru_hid=5,skip=2,ar_window=4)
    #     net.initialize(init=mx.init.Constant(0.5), ctx=ctx)
    #     trainer = gluon.Trainer(net.collect_params(),
    #         optimizer='adam',
    #         optimizer_params={'learning_rate': 0.01, 'clip_gradient': 10.0})
    #     loss_gpu = lstnet_sagemaker.batch_forward_backward(data, label, ctx, net, trainer, batch_size)
    #     self.assertIsNotNone(loss_gpu)
    #
    #     ctx = [mx.gpu(0)]
    #     net = lstnet_sagemaker.LSTNet(num_series=10,conv_hid=20,gru_hid=30,skip_gru_hid=5,skip=2,ar_window=4)
    #     net.initialize(init=mx.init.Constant(0.5), ctx=ctx)
    #     trainer = gluon.Trainer(net.collect_params(),
    #         optimizer='adam',
    #         optimizer_params={'learning_rate': 0.01, 'clip_gradient': 10.0})
    #     loss_cpu = lstnet_sagemaker.batch_forward_backward(data, label, ctx, net, trainer, batch_size)
    #     self.assertIsNotNone(loss_cpu)
    #     self.assertEqual(loss_gpu,loss_cpu)

    def test_train(self):
        hyperparameters = {
            'conv_hid' : 10,
            'gru_hid' : 10,
            'skip_gru_hid' : 2,
            'skip' : 5,
            'ar_window' : 6,
            'window' : 24*7,
            'horizon' : 24,
            'learning_rate' : 0.01,
            'clip_gradient' : 10.,
            'batch_size' : 128,
            'epochs' : 1
        }
        channel_input_dirs = {'train':self.training_data_dir,'test':self.training_data_dir}
        model = lstnet_sagemaker.train(hyperparameters,
            None,
            channel_input_dirs,
            self.output_dir,
            None,
            0,
            1,
            ['alg-1'],
            'alg-1')
        print(model)

if __name__ == '__main__':
    unittest.main()
