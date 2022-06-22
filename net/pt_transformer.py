import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from point_transformer_ops.point_transformer_modules import PointTransformerBlock, TransitionUp, TransitionDown
import point_transformer_ops.point_transformer_utils as pt_utils
from typing import Tuple


class SqueezerMixedTransformer(nn.Module):

    def __init__(self, dim=[6,32,64,128], k=8, out_kernel_dim=64, sampling_ratio=0.25, fix_encoder=False, fix_d_decoder=False):
        super(SqueezerMixedTransformer, self).__init__()

        self.fix_encoder = fix_encoder
        self.fix_d_decoder = fix_d_decoder

        # encoder
        self.Encoder = nn.ModuleList()
        for i in range(len(dim)-1):
            if i == 0:
                self.Encoder.append(nn.Linear(dim[i], dim[i+1], bias=False))
            else:
                self.Encoder.append(TransitionDown(dim[i], dim[i+1], k, sampling_ratio, fast=True))
            self.Encoder.append(PointTransformerBlock(dim[i+1], k))

        self.Decoder = nn.ModuleList()
        self.kernel_decoder = nn.ModuleList()

        # decoder
        for i in range(len(dim)-1, 0, -1):
            if i == len(dim) - 1:
                self.Decoder.append(nn.Linear(dim[i], dim[i], bias=False))
                self.kernel_decoder.append(nn.Linear(dim[i], dim[i], bias=False))
            else:
                self.Decoder.append(TransitionUp(dim[i+1], dim[i]))
                self.kernel_decoder.append(TransitionUp(dim[i+1], dim[i]))

            self.Decoder.append(PointTransformerBlock(dim[i], k))
            self.kernel_decoder.append(PointTransformerBlock(dim[i], k))

        # output fc
        self.fc_layer = nn.Sequential(
            nn.Conv1d(dim[1], dim[1], kernel_size=1, bias=False),
            nn.BatchNorm1d(dim[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(dim[1], 1, kernel_size=1),
        )

        # output fc 2: kernel decoder
        self.kernel_fc_layer = nn.Sequential(
            nn.Conv1d(dim[1], dim[1], kernel_size=1, bias=False),
            nn.BatchNorm1d(dim[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(dim[1], out_kernel_dim, kernel_size=1),
        )

    def forward(self, pt_pos, pt_feats):

        B, N, C = pt_feats.shape
        l_xyz, l_features = [pt_pos], [pt_feats]

        # encoding
        for i in range(int(len(self.Encoder) / 2)):
            if i == 0:
                li_features, li_xyz = self.Encoder[2 * i](l_features[i]), l_xyz[i]              # linear layer
            else:
                li_features, li_xyz = self.Encoder[2 * i](l_features[i], l_xyz[i])              # transition down

            li_features = self.Encoder[2 * i + 1](li_features, li_xyz)
            l_features.append(li_features)
            l_xyz.append(li_xyz)
            del li_features, li_xyz

        if self.fix_encoder:
            l_features = [l.detach() for l in l_features]
            l_xyz = [l.detach() for l in l_xyz]

        # decoding dist_score
        o_feats = [None for _ in l_features]
        D_n = int(len(self.Decoder)/2)
        for i in range(D_n):
            if i == 0:
                o_feats[D_n-i] = self.Decoder[2*i](l_features[D_n-i])
                o_feats[D_n-i] = self.Decoder[2*i+1](o_feats[D_n-i], l_xyz[D_n-i])
            else:
                o_feats[D_n-i], l_xyz[D_n-i] = self.Decoder[2*i](o_feats[D_n-i+1], l_xyz[D_n-i+1], l_features[D_n-i], l_xyz[D_n-i])
                o_feats[D_n-i] = self.Decoder[2*i+1](o_feats[D_n-i], l_xyz[D_n-i])

        if self.fix_d_decoder:
            o_feats = [o.detach() if isinstance(o, torch.Tensor) else o for o in o_feats ]

        # decoding kernel feats
        ko_feats = [None for _ in l_features]
        for i in range(D_n):
            if i == 0:
                ko_feats[D_n-i] = self.kernel_decoder[2*i](l_features[D_n-i])
                ko_feats[D_n-i] = self.kernel_decoder[2*i+1](ko_feats[D_n-i], l_xyz[D_n-i])
            else:
                ko_feats[D_n-i], l_xyz[D_n-i] = self.kernel_decoder[2*i](ko_feats[D_n-i+1], l_xyz[D_n-i+1], l_features[D_n-i], l_xyz[D_n-i])
                ko_feats[D_n-i] = self.kernel_decoder[2*i+1](ko_feats[D_n-i], l_xyz[D_n-i])

        # clean
        del o_feats[0], o_feats[1:]
        del ko_feats[0], ko_feats[1:]
        del l_xyz

        dist_score = self.fc_layer(o_feats[0].transpose(1,2).contiguous())
        k_feats = self.kernel_fc_layer(ko_feats[0].transpose(1,2).contiguous())
        k_feats = rearrange(k_feats, 'b c n -> b n c')
        k_feats = F.normalize(k_feats, dim=-1)
        return dist_score.view(B, N), k_feats.view(B, N, -1)


class BasePointTransformer(nn.Module):

    def __init__(self, dim=[6,32,64,128], output_dim=1, k=8, sampling_ratio=0.25):
        super(BasePointTransformer, self).__init__()

        # encoder
        self.Encoder = nn.ModuleList()
        for i in range(len(dim)-1):
            if i == 0:
                self.Encoder.append(nn.Linear(dim[i], dim[i+1], bias=False))
            else:
                self.Encoder.append(TransitionDown(dim[i], dim[i+1], k, sampling_ratio, fast=True))
            self.Encoder.append(PointTransformerBlock(dim[i+1], k))
        self.Decoder = nn.ModuleList()

        # decoder
        for i in range(len(dim)-1, 0, -1):
            if i == len(dim) - 1:
                self.Decoder.append(nn.Linear(dim[i], dim[i], bias=False))
            else:
                self.Decoder.append(TransitionUp(dim[i+1], dim[i]))

            self.Decoder.append(PointTransformerBlock(dim[i], k))

        # output fc
        self.fc_layer = nn.Sequential(
            nn.Conv1d(dim[1], dim[1], kernel_size=1, bias=False),
            nn.BatchNorm1d(dim[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(dim[1], output_dim, kernel_size=1),
        )

    def forward(self, pc: Tuple):

        pt_pos, pt_feats = pc               # input dim: (B, N, 3), (B, N, C)
        # pt_feats = rearrange(pt_feats, 'b n c -> b c n')
        # B, C, N = pt_feats.shape

        l_xyz, l_features = [pt_pos], [pt_feats]

        # encoding
        for i in range(int(len(self.Encoder) / 2)):
            if i == 0:

                # linear layer
                li_features = self.Encoder[2 * i](l_features[i])
                li_xyz = l_xyz[i]
            else:
                # transition down
                li_features, li_xyz = self.Encoder[2 * i](l_features[i], l_xyz[i])

            li_features = self.Encoder[2 * i + 1](li_features, li_xyz)
            # print(li_features.max(), self.Encoder[2 * i + 1])

            l_features.append(li_features)
            l_xyz.append(li_xyz)
            del li_features, li_xyz

        # decoding
        D_n = int(len(self.Decoder)/2)
        for i in range(D_n):
            if i == 0:
                l_features[D_n-i] = self.Decoder[2*i](l_features[D_n-i])
                l_features[D_n-i] = self.Decoder[2*i+1](l_features[D_n-i], l_xyz[D_n-i])
            else:
                l_features[D_n-i], l_xyz[D_n-i] = self.Decoder[2*i](l_features[D_n-i+1], l_xyz[D_n-i+1], l_features[D_n-i], l_xyz[D_n-i])
                l_features[D_n-i] = self.Decoder[2*i+1](l_features[D_n-i], l_xyz[D_n-i])

        del l_features[0], l_features[1:], l_xyz
        out = self.fc_layer(l_features[0].transpose(1,2).contiguous())
        return out



if __name__ == '__main__':

    pt_pos = torch.rand((1, 401, 3)).to(1)
    pt_feats = torch.rand((1, 401, 256)).to(1)

    input_channel = 256
    # pt_transformer = BasePointTransformer(dim=[input_channel, 64, 128, 128]).to(1)
    # out = pt_transformer.forward((pt_pos, pt_feats))

    s_transformer = SqueezerMixedTransformer(dim=[input_channel, 64, 128, 128], k=4, out_kernel_dim=64).to(1)
    dist_score, kernel_feats = s_transformer.forward(pt_pos, pt_feats)
    print(dist_score.shape)


