import numpy as np
import torch
import torch.nn as nn
from hloc import match_features, matchers
from hloc.utils.base_model import dynamic_load
from hloc.utils.tools import map_tensor

from SuperGluePretrainedNetwork.models.superglue import *
from matcher.matcher_base import BaseMatcher

def get_feature_matcher(conf='superglue', device=None):
    conf = match_features.confs[conf]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)
    return model

def find_matches(feats0, feats1, model, min_match_score=None):
    device = next(model.parameters()).device
    data = {}
    for k in feats1.keys():
        data[k+'0'] = feats0[k].__array__()
    for k in feats1.keys():
        data[k+'1'] = feats1[k].__array__()
    data = {k: torch.from_numpy(v)[None].float().to(device)
            for k, v in data.items()}

    # some matchers might expect an image but only use its size
    data['image0'] = torch.empty((1, 1,)+tuple(feats0['image_size'])[::-1])
    data['image1'] = torch.empty((1, 1,)+tuple(feats1['image_size'])[::-1])

    pred = model(data)
    matches = pred['matches0'].cpu().numpy().__array__()
    valid = matches > -1
    if min_match_score:
        scores = hfile[pair]['matching_scores0'].__array__()
        valid = valid & (scores > min_match_score)
    matches = np.stack([np.where(valid)[-1], matches[valid]], -1)

    return matches


class SuperGlueMatcher(BaseMatcher):

    def __init__(self, device=0):
        super(SuperGlueMatcher, self).__init__()
        self.model = get_feature_matcher(device=device)
        self.net = self.model.net
        self.pt_encoder = self.net.kenc

    def encode_pt_feats(self, input):
        desc, normalized_kpts, scores = input

        ori_dev = desc.device
        cur_dev = torch.cuda.current_device()

        desc = desc.to(cur_dev)
        normalized_kpts = normalized_kpts.to(cur_dev)
        scores = scores.to(cur_dev)

        return (desc + self.pt_encoder(normalized_kpts, scores)).to(ori_dev)

    def get_score(self, desc0, desc1, optimal_transport=True):
        cur_dev = torch.cuda.current_device()

        desc0 = desc0.to(cur_dev)
        desc1 = desc1.to(cur_dev)

        # Multi-layer Transformer network.
        desc0, desc1 = self.net.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.net.final_proj(desc0), self.net.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.net.config['descriptor_dim']**.5

        # Run the optimal transport.
        if optimal_transport:
            scores = log_optimal_transport(
                scores, self.net.bin_score,
                iters=self.net.config['sinkhorn_iterations'])

        return scores

    def get_matches(self, scores, optimal_transport=True):
        # Run the optimal transport.
        if optimal_transport:
            scores = log_optimal_transport(scores, self.net.bin_score, iters=self.net.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.net.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'scores': scores,
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }

    def forward(self, query, reference):
        cur_dev = torch.cuda.current_device()

        desc0 = query['desc'].to(cur_dev)
        desc1 = reference['desc'].to(cur_dev)

        # Multi-layer Transformer network.
        desc0, desc1 = self.net.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.net.final_proj(desc0), self.net.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.net.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.net.bin_score,
            iters=self.net.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.net.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'scores': scores,
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }
