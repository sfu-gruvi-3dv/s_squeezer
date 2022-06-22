import torch.nn as nn


class BaseMatcher(nn.Module):

    def extract_local_feats(self, input):
        pass

    def match(self, f_a, f_b):
        pass