# -*- coding: utf-8 -*-

from einops.einops import rearrange
import torch
import cv2
import numpy as np


def colormap(tensor: torch.tensor, cmap='jet', clip_range=None, scale_each=True, chw_order=True):
    """
        Create colormap for each single channel input map.

    Args:
        tensor (torch.Tensor): input single-channel image, dim (N, H, W) or (N, 1, H, W)
        cmap (str): the type of color map
        clip_range (list): the minimal or maximal clamp on input tensor, list: [min, max]
        scale_each (bool): normalize the input based on each image instead of the whole batch
        chw_order (bool): the output type of tensor, either CHW or HWC

    Returns:
        colormap tensor, dim (N, 3, H, W) if 'chw_order' is True or (N, H, W, 3)

    """
    if cmap == 'gray':
        cmap_tag = cv2.COLORMAP_BONE
    elif cmap == 'hsv':
        cmap_tag = cv2.COLORMAP_HSV
    elif cmap == 'hot':
        cmap_tag = cv2.COLORMAP_HOT
    elif cmap == 'cool':
        cmap_tag = cv2.COLORMAP_COOL
    else:
        cmap_tag = cv2.COLORMAP_JET

    if tensor.dim() == 2:
        # single image
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    elif tensor.dim() == 4:
        if tensor.size(1) == 1:
            tensor = tensor.view(tensor.size(0), tensor.size(2), tensor.size(3))
        else:
            raise Exception("The input image should has one channel.")
    elif tensor.dim() > 4:
        raise Exception("The input image should has dim of (N, H, W) or (N, 1, H, W).")

    # normalize
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if clip_range is not None:
        assert isinstance(clip_range, tuple), \
            "range has to be a tuple (min, max) if specified. min and max are numbers"

    def norm_ip(img, min_, max_):
        img.clamp_(min=min_, max=max_)
        img.add_(-min_).div_(max_ - min_ + 1e-5)

    def norm_range(tensor_, range_):
        if range_ is not None:
            norm_ip(tensor_, range_[0], range_[1])
        else:
            norm_ip(tensor_, float(tensor_.min()), float(tensor_.max()))

    if scale_each is True:
        # loop over mini-batch dimension
        for t in tensor:
            norm_range(t, clip_range)
    else:
        norm_range(tensor, clip_range)

    # apply color map
    N, H, W = tensor.shape
    color_tensors = []
    for n in range(N):
        sample = tensor[n, ...].detach().cpu().numpy()
        colormap_sample = cv2.applyColorMap((sample * 255).astype(np.uint8), cmap_tag)
        colormap_sample = cv2.cvtColor(colormap_sample, cv2.COLOR_BGR2RGB)
        color_tensors.append(torch.from_numpy(colormap_sample).cpu())
    color_tensors = torch.stack(color_tensors, dim=0).float() / 255.0

    return color_tensors.permute(0, 3, 1, 2) if chw_order else color_tensors


def heatmap_blend(img: torch.tensor,
                  heatmap: torch.tensor,
                  heatmap_blend_alpha=0.5,
                  heatmap_clip_range=None,
                  cmap='jet', chw_order=True) -> torch.Tensor:
    """
        Blend the colormap onto original image

    Args:
        img (torch.Tensor): original image in RGB, dim (N, 3, H, W)
        heatmap (torch.Tensor): input heatmap, dim (N, H, W) or (N, 1, H, W)
        heatmap_blend_alpha (float): blend factor, setting with 0 to make the output identical to original img.
        heatmap_clip_range (list): the minimal or maximal clamp on input tensor, list: [min, max]
        cmap (str): colormap type to blend
        chw_order (bool): the output type of tensor, either CHW or HWC

    Returns:
        blended heatmap image, dim (N, 3, H, W)

    """
    if heatmap.dim() == 4:
        if heatmap.size(1) == 1:
            heatmap = heatmap.view(heatmap.size(0), heatmap.size(2), heatmap.size(3))
        else:
            raise Exception("The heatmap should be (N, 1, H, W) or (N, H, W)")
    N, C3, H, W = img.shape

    assert heatmap_blend_alpha < 1.0
    assert H == heatmap.size(1)
    assert W == heatmap.size(2)
    assert N == heatmap.size(0)
    assert C3 == 3                      # input image has three channel RGB

    color_map = colormap(heatmap, cmap=cmap, clip_range=heatmap_clip_range, chw_order=chw_order).to(img.device)
    output_heat_map = img.clone()*(1.0 - heatmap_blend_alpha) + color_map * heatmap_blend_alpha
    return output_heat_map


class UnNormalize(object):
    """
        Inverse normalized image (tensor), widely used in visualization.

    Example:
       >>> tensor = torch.randn((3, 32, 32))                              # (C, H, W)
       >>> unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
       >>> unorm(tensor)
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
            Inverse the normalized tensor.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) that has been normalized.
        Returns:
            Tensor: Normalized image, dim (C, H, W)
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def tensor_to_vis(t: torch.Tensor, unormalize=UnNormalize()):
    """
        Convert the tensor to image array
    Args:
        t (tensor): input tensor, dim: (N, C, H, W) or (N, L, C, H, W)
        unormalize (callable): callable function to unormalize input tensor

    Returns:
        imgs (np.ndarray) with dim (H, W, C).

    """
    if t.dim() == 5:
        t = rearrange(t, 'N L C H W -> (N L) C H W')
    elif t.dim() == 3:
        t = t.unsqueeze(0)

    t_list = [t[i, :, :, :].clone() for i in range(t.shape[0])]
    if unormalize:
        t_list = [unormalize(t) for t in t_list]
    return [t.permute(1, 2, 0).cpu().numpy() for t in t_list]
