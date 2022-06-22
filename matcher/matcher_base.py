import torch.nn as nn

class BaseMatcher(nn.Module):

    def __init__(self):
        super(BaseMatcher, self).__init__()

    def encode_pt_feats(self, input):
        pass
