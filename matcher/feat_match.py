import torch
import numpy as np

def _find_nn(sim, ratio_thresh, distance_thresh):
    sim_nn, ind_nn = sim.topk(2 if ratio_thresh else 1, dim=-1, largest=True)
    dist_nn = 2 * (1 - sim_nn)
    mask = torch.ones(ind_nn.shape[:-1], dtype=torch.bool, device=sim.device)
    if ratio_thresh:
        mask = mask & (dist_nn[..., 0] <= (ratio_thresh**2)*dist_nn[..., 1])
    if distance_thresh:
        mask = mask & (dist_nn[..., 0] <= distance_thresh**2)
    matches = torch.where(mask, ind_nn[..., 0], ind_nn.new_tensor(-1))
    scores = torch.where(mask, (sim_nn[..., 0]+1)/2, sim_nn.new_tensor(0))
    return matches, scores

def _mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    ok = (m0 > -1) & (inds0 == loop)
    m0_new = torch.where(ok, m0, m0.new_tensor(-1))
    return m0_new

def _mutual_nn_matcher(descriptors1, descriptors2):
    # Mutual nearest neighbors (NN) matcher for L2 normalized descriptors.
    device = descriptors1.device
    sim = torch.matmul(descriptors1, descriptors2.t())
    nn_sim, nn12 = torch.max(sim, dim=1)
    nn_dist = torch.sqrt(2 - 2 * nn_sim)
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t(), nn_dist[mask]

def _ratio_matcher(descriptors1, descriptors2, ratio=0.8):
    # Lowe's ratio matcher for L2 normalized descriptors.
    device = descriptors1.device
    sim = torch.matmul(descriptors1, descriptors2.t())
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    ids1 = torch.arange(0, sim.shape[0], device=device)
    matches = torch.stack([ids1, nns[:, 0]])
    ratios = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    mask = (ratios <= ratio)
    matches = matches[:, mask]
    return matches.t(), nns_dist[mask, 0]


def _ratio_mutual_nn_matcher(descriptors1, descriptors2, ratio=0.8):
    # Lowe's ratio matcher + mutual NN for L2 normalized descriptors.
    device = descriptors1.device
    sim = torch.matmul(descriptors1, descriptors2.t())
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nn12 = nns[:, 0]
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    matches = torch.stack([ids1, nns[:, 0]])
    ratios = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    mask = torch.min(ids1 == nn21[nn12], ratios <= ratio)
    matches = matches[:, mask]
    return matches.t(), nns_dist[mask, 0]

def _similarity_matcher(descriptors1, descriptors2, threshold=0.9):
    # Similarity threshold matcher for L2 normalized descriptors.
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()
    nn_sim, nn12 = torch.max(sim, dim=1)
    nn_dist = torch.sqrt(2 - 2 * nn_sim)
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (nn_sim >= threshold)
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t(), nn_dist[mask]

def match(desc1, desc2, device=None, params={'method': 'mutual'}):
    """

    Args:
        desc1: feature at frame 1, dim: (N, C)
        desc2: feature at frame 2, dim: (M, C)
        device: device to be used for computing feature matching
        params: the configuration of matching, can be:
                {'method': 'mutual'},
                {'method': 'ratio', 'ratio':0.8}
                {'method': 'mutual_nn_ratio', 'ratio':0.8}
                {'method': 'similarity', 'threshold':0.7}
    Returns:

    """
    return_ndarray = isinstance(desc1, np.ndarray)

    if isinstance(desc1, np.ndarray):
        desc1 = torch.from_numpy(desc1)
    if isinstance(desc2, np.ndarray):
        desc2 = torch.from_numpy(desc2)

    assert desc1.shape[1] == desc2.shape[1]
    if device is not None:
        desc1 = desc1.to(device)
        desc2 = desc2.to(device)

    if params['method'] == 'similarity':
        thres = 0.8 if 'threshold' not in params else params['threshold']
        matches, scores = _similarity_matcher(desc1, desc2, threshold=thres)
    elif params['method'] == 'mutual_nn_ratio':
        ratio = 0.8 if 'ratio' not in params else params['ratio']
        matches, scores = _ratio_mutual_nn_matcher(desc1, desc2, ratio=ratio)
    elif params['method'] == 'ratio':
        ratio = 0.8 if 'ratio' not in params else params['ratio']
        matches, scores = _ratio_mutual_nn_matcher(desc1, desc2, ratio=ratio)
    elif params['method'] == 'mutual':
        matches, scores = _mutual_nn_matcher(desc1, desc2)

    if return_ndarray:
        matches = matches.cpu().numpy()
        scores = scores.cpu().numpy()

    return matches, scores

