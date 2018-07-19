from mxnet import gluon, nd
import numpy as np

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

if __name__ == '__main__':
    test = DataIterBuilder()