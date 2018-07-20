import unittest
from lstnet import LSTNet
from mxnet import nd, gluon

class TestLSTNet(unittest.TestCase):

    FILE_PATH = '../data/electricity.txt'

    def test_TimeSeriesDataset(self):
        net = LSTNet(num_series=321, conv_hid=100, gru_hid=100, skip_gru_hid=5, skip=24, ar_window=24)
        x = nd.random.uniform(shape=(128, 1000, 321))
        net.initialize()
        y = net(x)
        assert y.shape == (128, 321)
        nd.waitall()

if __name__ == '__main__':
    unittest.main()
