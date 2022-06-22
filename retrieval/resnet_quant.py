import torch, dgl, pickle, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from core_dl.torch_vision_ext import UnNormalize
from core_io.print_msg import *

from retrieval.rmac_resnet import ResNet_RMAC

# from net.apg.nets.backbones.resnet import *
# from net.apg.nets import create_model
# from net.apg.utils import common
# from net.apg.nets.layers.pooling import GeneralizedMeanPoolingP

from net.soft_quant import *
from core_io.meta_io import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

unorm = UnNormalize()


def l2_normalize(x, axis=-1):
    x = F.normalize(x, p=2, dim=axis)
    return x


class ResQuantGem(nn.Module):
    
    def __init__(self, pretrained_path, fix_backbone=True):
        super(ResQuantGem, self).__init__()
        
        # params
        self.fix_backbone = fix_backbone

        # load pretrained resnet backbone
        backbone_ckpt = common.load_checkpoint(pretrained_path)
        net = create_model(pretrained="", **backbone_ckpt['model_options'])
        net.load_state_dict(backbone_ckpt['state_dict'])
        net.preprocess = backbone_ckpt.get('preprocess', net.preprocess)
        self.net = net
        
        # soft quant
        self.encoder = MLP([2048, 1024], do_bn=True)
        self.sigmoid = nn.Sigmoid()
        self.quantizate = DSQFunc()
        self.decoder = nn.Conv1d(1024, 1024, 1)

    def forward(self, img):
        N, C, H, W = img.shape
        
        if self.fix_backbone:
            with torch.no_grad():
                self.net = self.net.eval()
                x = ResNet_RMAC.forward(self.net, img)
        else:
            x = ResNet_RMAC.forward(self.net, img)
            
        # encode into quant feats
        y = self.encoder(x.view(N, 2048, 1))
        y = self.sigmoid(y) - 0.5
        y_quat, _ = self.quantizate(y.view(N, 1024))
        # y_rec_f = self.decoder(y_quat.view(N, 1024, 1))
        y_rec_f = l2_normalize(y_quat.view(N, 1024), axis=-1)
        
        return y_rec_f, None
    
    def encode_uint8(self, img):
        N, C, H, W = img.shape
        
        if self.fix_backbone:
            with torch.no_grad():
                x = ResNet_RMAC.forward(self.net, img)
        else:
            x = ResNet_RMAC.forward(self.net, img)
            
        # encode into quant feats
        y = self.encoder(x.view(N, 2048, 1))
        y = self.sigmoid(y) - 0.5
        _, y_quant_uint8 = self.quantizate(y.view(N, 1024))
        return y_quant_uint8
    
    def decode_uint8(self, x_uint8):
        assert x_uint8.ndim == 2
        N = x_uint8.shape[0]
        y = self.quantizate.recover(x_uint8)
        y_rec_f = self.decoder(y.view(N, 1024, 1))        
        y_rec_f = l2_normalize(y, axis=-1)
        return y_rec_f
    

if __name__ == '__main__':

    resnet = ResQuantGem('/mnt/hosthome/Projects/pretrained_model/Resnet101-AP-GeM-LM18.pt')

    # simple test
    dev_id = 1

    # simple test
    N, C, H, W = 5, 3, 512, 512
    imgs = torch.rand((N, C, H, W)).to(dev_id)

    resnet = resnet.to(dev_id)
    out = resnet(imgs)
    print(out)
    