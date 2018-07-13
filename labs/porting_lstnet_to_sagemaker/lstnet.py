import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn, rnn

class LSTNet(gluon.Block):
    """
    LSTNet auto-regressive block
    """
    def __init__(self, num_series, conv_hid, gru_hid, skip_gru_hid, skip, ar_window):
        super(LSTNet, self).__init__()
        kernel_size = 6
        dropout_rate = 0.2
        self.skip = skip
        self.ar_window = ar_window
        with self.name_scope():
            self.conv = nn.Conv1D(conv_hid, kernel_size=kernel_size, layout='NCW', activation='relu')
            self.dropout = nn.Dropout(dropout_rate)
            self.gru = rnn.GRU(gru_hid, layout='TNC')
            self.skip_gru = rnn.GRU(skip_gru_hid, layout='TNC')
            self.fc = nn.Dense(num_series)
            self.ar_fc = nn.Dense(1)

    def forward(self, x):
        """
        :param nd.NDArray x: input data in NTC layout (N: batch-size, T: sequence len, C: channels)
        :return: output of LSTNet in NC layout
        :rtype nd.NDArray
        """
        # Convolution
        c = self.conv(x.transpose((0, 2, 1)))  # Transpose NTC to to NCT (a.k.a NCW) before convolution
        c = self.dropout(c)

        # GRU
        r = self.gru(c.transpose((2, 0, 1)))  # Transpose NCT to TNC before GRU
        r = r[-1]  # Only keep the last output
        r = self.dropout(r)  # Now in NC layout

        # Skip GRU
        # Slice off multiples of skip from convolution output
        skip_c = c[:, :, -(c.shape[2] // self.skip) * self.skip:]
        skip_c = skip_c.reshape((c.shape[0], c.shape[1], -1, self.skip))  # Reshape to NCT x skip
        skip_c = skip_c.transpose((2, 0, 3, 1))  # Transpose to T x N x skip x C
        skip_c = skip_c.reshape((skip_c.shape[0], -1, skip_c.shape[3]))  # Reshape to Tx (Nxskip) x C
        s = self.skip_gru(skip_c)
        s = s[-1]  # Only keep the last output (now in (Nxskip) x C layout)
        s = s.reshape((x.shape[0], -1))  # Now in N x (skipxC) layout

        # FC layer
        fc = self.fc(nd.concat(r, s))  # NC layout

        # Autoregressive highway
        ar_x = x[:, -self.ar_window:, :]  # NTC layout
        ar_x = ar_x.transpose((0, 2, 1))  # NCT layout
        ar_x = ar_x.reshape((-1, ar_x.shape[2]))  # (NC) x T layout
        ar = self.ar_fc(ar_x)
        ar = ar.reshape((x.shape[0], -1))  # NC layout

        # Add autoregressive and fc outputs
        res = fc + ar
        return res
