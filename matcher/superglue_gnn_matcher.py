from core_io.print_msg import notice_msg
import numpy as np
import torch
import torch.nn as nn

from matcher.matcher_base import BaseMatcher
from matcher.superglue_base import *
from matcher.superglue_matcher import SuperGlueMatcher
from matcher.superglue_base import log_optimal_transport_safe

class SuperGlueGNNMatcher(BaseMatcher):

    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, init_with_weights: SuperGlueMatcher):
        super(SuperGlueGNNMatcher, self).__init__()
        self.config = self.default_config
        self.gnn = AttentionalGNN(self.config['descriptor_dim'], self.config['GNN_layers'])
        self.final_proj = nn.Conv1d(self.config['descriptor_dim'], self.config['descriptor_dim'], kernel_size=1, bias=True)
        bin_score = torch.nn.Parameter(torch.tensor(1.))

        if init_with_weights is not None:
            # copy the weight
            self.gnn.load_state_dict(init_with_weights.model.net.gnn.state_dict())
            self.final_proj.load_state_dict(init_with_weights.model.net.final_proj.state_dict())
            bin_score = torch.nn.Parameter(torch.tensor(1.) * init_with_weights.model.net.bin_score)
            notice_msg('inited with external weights', obj=self)

        self.register_parameter('bin_score', bin_score)

    def get_score(self, desc0, desc1, optimal_transport=True):

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim'] ** .5

        # Run the optimal transport.
        if optimal_transport:
            scores = log_optimal_transport_safe(
                scores, self.bin_score,
                iters=self.config['sinkhorn_iterations'])

        return scores

    def get_matches(self, scores, optimal_transport=True):
        # Run the optimal transport.
        if optimal_transport:
            scores = log_optimal_transport_safe(scores, self.bin_score, iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
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