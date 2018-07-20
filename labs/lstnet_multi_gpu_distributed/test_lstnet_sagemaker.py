import unittest
import lstnet_sagemaker
import os
import mxnet as mx
from mxnet import gluon

class TestLstnet_sagemaker(unittest.TestCase):

    training_data_dir = '../data'
    output_dir = 'output'

    def test_get_first_file_path_in_dir(self):
        filename = lstnet_sagemaker.get_first_file_path_in_dir(self.training_data_dir)
        self.assertEqual(filename,os.path.join(self.training_data_dir,'electricity.txt'))

    def test_get_file_path_single_host(self):
        path = lstnet_sagemaker.get_file_path(self.training_data_dir,'algo-1',['algo-1'])
        self.assertEqual(path,os.path.join(self.training_data_dir,'electricity.txt'))

    def test_get_file_path_multi_host(self):
        path = lstnet_sagemaker.get_file_path(self.training_data_dir,'algo-2',['algo-1','algo-2'])
        self.assertEqual(path,os.path.join(self.training_data_dir,'train_1.csv'))

if __name__ == '__main__':
    unittest.main()
