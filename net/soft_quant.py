import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        delta = torch.max(x) - torch.min(x)
        x = (x / delta + 0.5)
        return x.round() * 2 - 1

    @staticmethod
    def backward(ctx, g):
        return g


class DSQFunc(nn.Module):
    """
    Convert to the int8
    Note:
    There is only one learnable parameter, that is, alpha.
    """

    def __init__(self, num_bit=8, alpha=0.2, input_max=0.5, input_min=-0.5):
        """
        Args:
            num_bit: The number of bits eg: int8 -> 8
            alpha: The parameter which determines the precision of quantization
            input_max, input_min: Range of input data
        """
        super(DSQFunc, self).__init__()

        bit_range = 2 ** num_bit - 1
        self.uW = torch.tensor(2 ** (num_bit - 1) - 1).float()
        self.lW = torch.tensor(-1 * (2 ** (num_bit - 1))).float()
        self.register_buffer('running_uw', torch.tensor([self.uW.data]))
        self.register_buffer('running_lw', torch.tensor([self.lW.data]))

        self.input_max = input_max
        self.input_min = input_min
        # The length of each interval
        self.delta = (self.input_max - self.input_min) / (bit_range)

        # learnable parameter
        self.alphaW = nn.Parameter(data=torch.tensor(alpha).float())

    def clipping(self, x, upper, lower):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)
        return x

    def phi_function(self, x, mi, alpha, delta):
        # alpha should less than 2 or log will be None
        # alpha = alpha.clamp(None, 2)
        device = x.device
        alpha = alpha.float()
        alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).to(device), alpha)
        s = 1 / (1 - alpha)
        k = (2 / alpha - 1).log() * (1 / delta)
        x = (((x - mi) * k).tanh()) * s
        return x

    def sgn(self, x):
        x = RoundWithGradient.apply(x)
        return x

    def dequantize(self, x, lower_bound, delta, interval):
        # save mem
        x = ((x + 1) / 2 + interval) * delta + lower_bound
        # y = (interval - 128).clone().detach().type(torch.int8)
        y = (x - lower_bound) / delta 
        return x, torch.round(y).type(torch.uint8)

    def recover(self, q_uint8):
        return q_uint8 * self.delta + self.input_min
    
    def to_uint8(self, q_v):
        y = (q_v - self.input_min) / self.delta
        return torch.round(y).type(torch.uint8)

    def forward(self, x):
        cur_running_lw = self.running_lw
        cur_running_uw = self.running_uw
        Qvalue = self.clipping(x, cur_running_uw, cur_running_lw)
        interval_idx = ((Qvalue - self.input_min) / self.delta).trunc()
        mi = (interval_idx + 0.5) * self.delta + self.input_min
        Qvalue = self.phi_function(Qvalue, mi, self.alphaW, self.delta)
        Qvalue = self.sgn(Qvalue)
        DQvalue, Qvalue = self.dequantize(Qvalue, self.input_min, self.delta, interval_idx)
        return DQvalue, Qvalue


class SoftQuant(nn.Module):

    def __init__(self, encoder_dims, decoder_dims) -> None:
        super(SoftQuant, self).__init__()
        self.encoder = MLP(encoder_dims, do_bn=True)
        self.sigmoid = nn.Sigmoid()
        self.quantizate = DSQFunc()
        self.decoder = MLP(decoder_dims)

    def encode(self, feats):
        q_feats = self.encoder(feats)
        qv_feats = self.sigmoid(q_feats) - 0.5
        q_feats, quanti_r_feats = self.quantizate(qv_feats)
        return q_feats, quanti_r_feats, qv_feats

    def decode(self, q_feats):
        feats = self.decoder(q_feats)
        return feats


if __name__ == "__main__":
    dsq = DSQFunc()
    input = torch.tensor([-0.2, 0.2, 0.3])
    output = dsq(input)
    print(output)