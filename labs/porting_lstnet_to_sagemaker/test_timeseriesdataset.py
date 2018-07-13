import unittest
from timeseriesdataset import TimeSeriesDataset, TimeSeriesData
import numpy as np

class TestTimeSeriesDataset(unittest.TestCase):

    FILE_PATH = '../data/electricity.txt'

    def test_TimeSeriesDataset(self):
        dataset = TimeSeriesDataset(np.arange(0, 100).reshape(-1, 1).repeat(10, axis=1), window=10, horizon=5)
        assert len(dataset) == 85
        for i in range(len(dataset)):
            d, l = dataset[i]
            self.assertTrue(np.array_equal(d, np.arange(i, i + 10).reshape(-1, 1).repeat(10, axis=1)))
            self.assertTrue(np.array_equal(l, np.array([i + 14]).repeat(10, axis=0)))

    def test_TimeSeriesData(self):
        data = TimeSeriesData(self.FILE_PATH,window=100,horizon=1,train_ratio=0.9)
        self.assertEqual(len(data.train),23572)
        self.assertEqual(len(data.val),2530)

if __name__ == '__main__':
    unittest.main()
