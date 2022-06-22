import torch.cuda
import torch.nn as nn
from matcher import superglue_matcher


class PointFeatEncoder(nn.Module):

    def __init__(self, device='cpu'):
        super(PointFeatEncoder, self).__init__()

        model = superglue_matcher.get_feature_matcher(device=device)
        self.encoder = model.net.kenc

    def forward(self, desc: torch.Tensor, normalized_kpts: torch.Tensor, scores: torch.Tensor):
        ori_dev = desc.device
        cur_dev = torch.cuda.current_device()

        desc = desc.to(cur_dev)
        normalized_kpts = normalized_kpts.to(cur_dev)
        scores = scores.to(cur_dev)

        return (desc + self.encoder(normalized_kpts, scores)).to(ori_dev)
