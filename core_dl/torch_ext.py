import torch
import numpy as np
from typing import Iterable, List, Tuple, Union

Scalar = Union[int, float]
Tensor = torch.Tensor
Vector = Union[Scalar, List[Scalar], Tuple[Scalar, ...], Tensor]  # anything working with `torch.as_tensor`
Shape = Union[List[int], Tuple[int, ...], torch.Size]
Device = torch.device


def batch_sel_3d(x, dim, index):
    """ select features given index (batch version)
    """
    assert x.ndim == 3 and index.ndim == 2
    assert x.shape[0] == index.shape[0]
    assert index.dtype == torch.long
    B, K = index.shape

    if dim == 1:
        sel_index_ = index.unsqueeze(2).expand(B, K, x.size(2))
    elif dim == 2:
        sel_index_ = index.unsqueeze(1).expand(B, x.size(1), K)
    else:
        raise Exception('only dim==1 and dim==2 is allowed')
       
    return torch.gather(x, dim=dim, index=sel_index_)


def index_of_elements(a, elements):

    a_ = a.cpu().numpy() if isinstance(a, torch.Tensor) else a
    elements_ = elements.cpu().numpy() if isinstance(elements, torch.Tensor) else elements

    sorted_a = np.sort(a_)
    sorted_a_inds = np.argsort(a_)
    idx = np.searchsorted(sorted_a, elements_)

    invalid_mask = (idx >= sorted_a.shape[0])
    idx[invalid_mask] = 0 

    out = sorted_a_inds[idx]
 
    mask = np.logical_or(invalid_mask, np.logical_not(np.isin(elements_, a_)))
    out[mask] = -1

    if isinstance(a, torch.Tensor):
        out = torch.from_numpy(out)

    return out



def multi_index_select(A: Tensor, idx: Tensor) -> Tensor:
    r"""Select the element given the index, this function supports multi-dimension, unlike the
        original `torch.index_select`, which only support indexing 1-D vector.

        Args:
            A: input tensor with arbitary dimension.
            idx: the element index, should be (N, n_dim_of_A)

        Returns:
            selected element, (N, ).

    """
    assert idx.ndim == 2
    assert idx.shape[1] == A.ndim
    assert idx.dtype == torch.long

    # convert the multi dim index to 1d
    idx_1d = ravel_multi_index(idx, A.shape)
    sel = torch.index_select(A.view(-1), dim=0, index=idx_1d)
    return sel


def ravel_multi_index(coords: Tensor, shape: Shape) -> Tensor:
    r"""Converts a tensor of coordinate vectors into a tensor of flat elements.

    This is a `torch` implementation of `numpy.ravel_multi_index`.

    Args:
        coords: A tensor of coordinate vectors, (*, D).
        shape: The source shape.

    Returns:
        The raveled elements, (*,).
    """

    coef = coords.new_tensor(shape[1:] + (1,))
    coef = coef.flipud().cumprod(0).flipud()

    if coords.is_cuda and not coords.is_floating_point():
        return (coords * coef).sum(dim=-1)
    else:
        return coords @ coef


def unravel_index(elements: Tensor, shape: Shape) -> Tensor:
    r"""Converts a tensor of flat elements into a tensor of coordinate vectors.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        elements: A tensor of flat elements, (*,).
        shape: The target shape.

    Returns:
        The unraveled coordinates, (*, D).
    """

    coords = []

    for dim in reversed(shape):
        coords.append(elements % dim)
        elements = elements // dim

    coords = torch.stack(coords[::-1], dim=-1)

    return coords
