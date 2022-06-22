# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from core_io.print_msg import *


class TensorboardLogger:
    """ Logger that write the information to Tensorboard.
    """

    keys = ['Time', 'Event', 'Iteration', 'net']

    def __init__(self, log_file_dir=None, purge_step=None):
        self.log_file_dir = log_file_dir
        self.writer = SummaryWriter(log_dir=log_file_dir, purge_step=purge_step)
        self.enable_param_histogram = True
        self.cur_iteration = -1

    def __del__(self):
        self.writer.close()

    def add_keys(self, keys):
        if isinstance(keys, list):
            for key in keys:
                if key not in self.keys:
                    self.keys.append(key)
        elif keys not in self.keys:
            self.keys.append(keys)

    def attach_layer_params(self, net):
        self.enable_param_histogram = True
        for name, param in net.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.cur_iteration)

    def log(self, log_dict: dict):
        self.cur_iteration = log_dict['Iteration']
        for key, log_value in log_dict.items():

            # Write the Scalar to tensorboard
            if key.startswith('Loss') or key.startswith('Accuracy') or key.startswith('Scalar'):
                self.writer.add_scalar(key, float(log_value), self.cur_iteration)

            # Write the Scalars to tensorboard
            if key.startswith('Scalars'):
                # The log_dict[key] should be another dict, e.g.
                self.writer.add_scalars(key, log_value, self.cur_iteration)

            if key.startswith('Histogram'):
                if isinstance(log_value, torch.Tensor):
                    self.writer.add_histogram(key, log_value.cpu().detach().numpy(), self.cur_iteration)
                elif isinstance(log_value, np.ndarray):
                    self.writer.add_histogram(key, log_value, self.cur_iteration)

            # update related statistic info of network structure (and parameters)
            if isinstance(log_value, torch.nn.Module) and self.enable_param_histogram:
                # Update the parameter histogram
                for name, param in log_value.named_parameters():
                    self.writer.add_histogram('param_' + name, param.clone().cpu().data.numpy(), self.cur_iteration)
                    self.writer.add_histogram('grad_' + name, param.grad.clone().cpu().data.numpy(), self.cur_iteration)

            # Write the Image to tensorboard
            if key.startswith('Image'):
                # The log_dict[key] should be a list of image tensors
                img_grid = log_value
                if len(log_dict[key]) > 1:
                    img_grid = make_grid(img_grid, nrow=1)
                else:
                    img_grid = img_grid[0]
                self.writer.add_image(key, img_grid, self.cur_iteration)

    def flush(self):
        pass

    def report(self):
        msg('Log dir: %s' % self.log_file_dir, obj=self)
