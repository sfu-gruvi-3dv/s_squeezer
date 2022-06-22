import torch

from core_io.meta_io import from_meta, path_from_meta
from core_dl.torch_vision_ext import *
from torchvision.utils import make_grid
from core_dl.torch_ext import *
from net.scene_sq import corres_pos_from_pairs
from exp.scene_sq_utils import get_corres_ref_2d_indices
from einops import asnumpy
from visualizer.corres_plot import *
from PIL import Image
import io


def plot_to_ndarray(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im


def get_corres_ref_2d(sample, res, args: dict, use_gt_matches: bool = False):
    pass

def get_corres_ref_2d(sample, res, args: dict, use_gt_matches: bool = False):
    q_imgs, q_info, ref_imgs, ref_info = sample
    match_key = 'gt' if use_gt_matches else 'matches'

    num_r_frames = len(ref_info['img_names'])

    # parameters
    out_file_path = path_from_meta(args, 'out_file_path', check_exist=False, raise_exception=False)
    sel_q_ids = from_meta(args, 'sel_q_ids', default=[q for q in res.keys() if isinstance(q, int)])
    max_ref_frames = from_meta(args, 'max_ref_frames', default=num_r_frames)
    num_corres = from_meta(args, 'num_corres')
    lw = from_meta(args, 'line_width', default=0.5)
    pr = from_meta(args, 'pt_radius', default=2)
    alpha = from_meta(args, 'line_alpha', default=0.5)

    # gathering q2r corresopndences in 2d to 2d
    ref_frame_ids = np.arange(num_r_frames)
    corres_dict = dict()
    for q in sel_q_ids:
        if q not in res:
            continue

        corres_matches = res[q][match_key].cpu()
        corres_dict[q] = list()

        for r in ref_frame_ids:
            r_3d_obs = ref_info['pt2d_obs3d'][r][0].cpu()
            r_2d_pos = ref_info['pt2d_pos'][r][0].cpu()
            sel_matches = get_corres_ref_2d_indices(corres_matches, r_3d_obs)
            corres_dict[q].append(sel_matches)
    return corres_dict


def query_2_ref_frames(ref_info: dict, q2r_matches: dict, args: dict):
    assert isinstance(q2r_matches, dict)

    num_r_frames = len(ref_info['img_names'])

    # parameters
    sel_q_ids = from_meta(args, 'sel_q_ids', default=[q for q in q2r_matches.keys() if isinstance(q, int)])
    sel_ref_ids = from_meta(args, 'sel_ref_ids', default=np.arange(num_r_frames))

    # gathering q2r corresopndences in 2d to 2d
    corres_dict = dict()
    for q in sel_q_ids:

        corres_matches = q2r_matches[q]
        if isinstance(corres_matches, np.ndarray):
            corres_matches = torch.from_numpy(corres_matches)

        corres_dict[q] = list()

        for r in sel_ref_ids:
            r_3d_obs = ref_info['pt2d_obs3d'][r]
            # r_2d_pos = ref_info['pt2d_pos'][r]
            if isinstance(r_3d_obs, np.ndarray):
                r_3d_obs = torch.from_numpy(r_3d_obs)

            sel_matches = get_corres_ref_2d_indices(corres_matches, r_3d_obs)
            corres_dict[q].append(sel_matches)
    return corres_dict

def plot_q2r(sample, res, args: dict, to_grid=True, show=False):
    q_imgs, q_info, ref_imgs, ref_info = sample
    q_vis_imgs = tensor_to_vis(q_imgs)
    r_vis_imgs = tensor_to_vis(ref_imgs)

    num_r_frames = len(ref_info['img_names'])

    # parameters
    out_file_path = path_from_meta(args, 'out_file_path', check_exist=False, raise_exception=False)
    sel_q_ids = from_meta(args, 'sel_q_ids', default=[q for q in res.keys() if isinstance(q, int)])
    max_ref_frames = from_meta(args, 'max_ref_frames', default=num_r_frames)
    num_corres = from_meta(args, 'num_corres')
    lw = from_meta(args, 'line_width', default=0.5)
    pr = from_meta(args, 'pt_radius', default=2)
    alpha = from_meta(args, 'line_alpha', default=0.5)

    # gathering q2r corresopndences in 2d to 2d
    ref_frame_ids = np.arange(num_r_frames)
    corres_dict = dict()
    for q in sel_q_ids:
        if q not in res:
            continue

        corres_matches = res[q]['matches'].cpu()
        corres_dict[q] = list()

        for r in ref_frame_ids:
            r_3d_obs = ref_info['pt2d_obs3d'][r][0].cpu()
            r_2d_pos = ref_info['pt2d_pos'][r][0].cpu()
            sel_matches = get_corres_ref_2d_indices(corres_matches, r_3d_obs)
            corres_dict[q].append(sel_matches)

    # plot images
    imgs = []
    for q in sel_q_ids:
        q_2d_pos = q_info['pt2d_pos'][q].cpu()
        q_pos_scale_f = q_vis_imgs[q].shape[0] / q_info['dims'][q][0].item()

        q2r_num_matches = [m.shape[0] for m in corres_dict[q]]
        q2r_ref_idx = np.argsort(q2r_num_matches)[::-1]

        if max_ref_frames is not None:
            q2r_ref_idx = q2r_ref_idx[:max_ref_frames]

        for ref_idx in q2r_ref_idx:

            r_2d_pos = ref_info['pt2d_pos'][ref_idx].cpu()
            r_pos_scale_f = r_vis_imgs[ref_idx].shape[0] / ref_info['dims'][ref_idx][0].item()
            sel_matches = corres_dict[q][ref_idx]

            q_pos_m, ref_pos_m = corres_pos_from_pairs(q_2d_pos.view(-1, 2), r_2d_pos.view(-1, 2), sel_matches)

            rand_idx = np.arange(q_pos_m.shape[0])
            np.random.shuffle(rand_idx)
            if num_corres is not None:
                rand_idx = rand_idx[: num_corres]

            plt.clf()
            plot_images([q_vis_imgs[q], r_vis_imgs[ref_idx]])
            plot_matches(q_pos_m[rand_idx] * q_pos_scale_f, ref_pos_m[rand_idx] * r_pos_scale_f,
                         a=alpha, lw=lw, ps=pr)
            fig = plot_to_ndarray(plt.gcf())[:, :, :3]
            imgs.append(fig)
            if show:
                plt.show()
            else:
                plt.close()

    imgs = torch.cat([torch.from_numpy(imgs[i].copy()).permute(2, 0, 1).unsqueeze(0)
                      for i in range(len(imgs))], dim=0)

    if to_grid or out_file_path:
        imgs = make_grid(imgs, nrow=1)

        if out_file_path:
            plt.imsave(out_file_path, asnumpy(imgs.permute(1, 2, 0)))

    return imgs


def attach_score_to_ref_pts(ref_info, ref_pt_scores):
    if isinstance(ref_pt_scores, torch.Tensor):
        confidence = asnumpy(ref_pt_scores).ravel()
    else:
        confidence = ref_pt_scores.ravel()
    num_r_frames = len(ref_info['img_names'])

    kpt_pos = []
    kpt_confs = []

    for r in range(num_r_frames):
        r_3d_obs = asnumpy(ref_info['pt2d_obs3d'][r][0])
        r_2d_pos = asnumpy(ref_info['pt2d_pos'][r][0].cpu())

        r_2d_confid = confidence[r_3d_obs]
        kpt_pos.append(r_2d_pos)
        kpt_confs.append(r_2d_confid)
    
    return kpt_pos, kpt_confs
    

def plot_sel_ref_pts(ref_imgs, ref_info, ref_pt_scores, args: dict, return_pts=False, show=False):
    if ref_imgs is not None:
        r_vis_imgs = tensor_to_vis(ref_imgs)
    num_r_frames = len(ref_info['img_names'])

    # params
    clip_range = from_meta(args, 'range', default=(-1, -1))
    cmap_str = from_meta(args, 'cmap', 'Blues')
    cmap = plt.get_cmap(cmap_str)
    pr = from_meta(args, 'pt_radius', default=5)
    max_ref_frames = from_meta(args, 'max_ref_frames', default=num_r_frames)

    ref_pt_scores = asnumpy(ref_pt_scores)
    r_pt_pos, r_pt_scores = attach_score_to_ref_pts(ref_info, ref_pt_scores)
    r_pt_score_n = [(r - r.min()) / (r.max() - r.min()) for r in r_pt_scores]
    r_pt_color = [[cmap(s) for s in sn] for sn in r_pt_score_n]
    r_pt_text = [['id:%d, s:%.2f' % (i, s) for i, s in enumerate(scores)] for scores in r_pt_scores]
    
    (min_, max_) = clip_range
    if min_ == -1 and max_ == -1:
        min_ = ref_pt_scores.min()
        max_ = ref_pt_scores.max()

    sel_idx_list = [[i for i, s in enumerate(pt_s) if min_ <= s <= max_] for pt_s in r_pt_scores]
    sel_r_pt_color, sel_r_pt_pos, sel_r_pt_text = [], [], []
    for r in range(num_r_frames):
        ori_h, ori_w = ref_info['dims'][r]
        ori_h, ori_w = ori_h.item(), ori_w.item()
        
        h, w = ref_imgs[r].shape[:2] if ref_imgs is not None else (ori_h, ori_w)
        
        sel_idx = sel_idx_list[r]
        if len(sel_idx) > 0:
            sel_r_color = [r_pt_color[r][i] for i in sel_idx]
            sel_r_text = [r_pt_text[r][i] for i in sel_idx]
            sel_r_pos2d = r_pt_pos[r][np.asarray(sel_idx), :]
            
            sel_r_pos2d[:, 0] *= (w / ori_w)
            sel_r_pos2d[:, 1] *= (h / ori_h)
        else:
            sel_r_color = []
            sel_r_pos2d = []
            sel_r_text = []
        sel_r_pt_color.append(sel_r_color)
        sel_r_pt_pos.append(sel_r_pos2d)
        sel_r_pt_text.append(sel_r_text)
        
    if return_pts:
        return sel_r_pt_pos, sel_r_pt_color, sel_r_pt_text
    
    # plot
    plt.clf()
    plot_images(r_vis_imgs[:max_ref_frames])
    plot_keypoints(kpts=sel_r_pt_pos[:max_ref_frames], colors=sel_r_pt_color[:max_ref_frames], ps=pr)
    img = plot_to_ndarray(plt.gcf())[:, :, :3]
    if show:
        plt.show()
    else:
        plt.close()

    return torch.from_numpy(img.copy()).permute(2, 0, 1)
