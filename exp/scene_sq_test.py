from comet_ml import Experiment
from pytorch_lightning import callbacks, loggers, trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl

import torch, argparse, sys
torch.manual_seed(1000)

from core_dl.train_params import TrainParameters
from exp.scene_sq_box import SceneSQBox
from exp.scene_sq_visualizer import get_corres_ref_2d
from net.scene_sq import corres_pos_from_pairs
from net.qp_layer import QPLayer
from dataset.data_module import RankedDataModule, CachedDataModule
from core_dl.get_host_name import get_host_name
from core_io.print_msg import *
from core_dl.lightning_logger import LightningLogger, PeriodicCheckpoint
from core_dl.lightning_trainer import create_pl_trainer
from core_3dv.quaternion import quaternion2rot
from core_math.transfom import rotation_from_matrix

from pathlib import Path
import numpy as np
from tqdm import tqdm

import pycolmap

# sys.path.append('/mnt/Exp_fast/tools/Hierarchical-Localization')
# sys.path.append('/mnt/Exp_fast/tools/Hierarchical-Localization/third_party')


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Test squeezer network')
    parser.add_argument('--exp',
                        type=str,
                        default='exp_squeezer_init',
                        help='the experiment will be performed')
    parser.add_argument('--name_tag',
                        type=str,
                        help='Optional, set the experiment name tag explictly')
    parser.add_argument('--dataset',
                        type=str,
                        default='robotcar_within10m',
                        help='dataset to be used')
    parser.add_argument('--hostname',
                        type=str,
                        default=get_host_name(),
                        help='the machine that will be used to train the stg1 network')
    parser.add_argument('--debug',
                        action='store_true')
    parser.add_argument('--log_dir',
                        type=str,
                        help='path to log directory, this will overwrite the experiment configuration.')
    parser.add_argument('--temp_dir',
                        type=str,
                        help='path to temporary output directory, used to dump intermediate results.')
    parser.add_argument('--ckpt_path',
                        type=str,
                        required=True,
                        help='load from checkpoint path, will overwrite params.')
    parser.add_argument('--epoch',
                        type=int,
                        default=20,
                        help='number of epoches.')
    parser.add_argument('--qp_max_input_keypoints',
                        type=int,
                        default=800,
                        help='qp_max_input_keypoints')
    parser.add_argument('--qp_compression_ratio',
                        type=float,
                        default=0.25,
                        help='qp_compression_ratio')
    parser.add_argument('--qp_distinctiveness_weight',
                        type=float,
                        default=100.0,
                        help='qp_distinctiveness_weight')
    parser.add_argument('--devices', help='specific device id', type=str)

    return parser


def overwrite_params(params: TrainParameters, args) -> TrainParameters:
    """ specific unique config for current training parameters
    """
    params.DEBUG = True if params.DEBUG is True or args.debug is True else False
    if params.DEBUG is True:
        params.LOG_DIR = None
    if args.hostname != params.HOSTNAME:
        params.HOSTNAME = args.hostname
    if args.name_tag is not None:
        params.NAME_TAG = args.name_tag
    if args.temp_dir is not None and os.path.exists(args.temp_dir):
        params.DEBUG_OUTPUT_DIR = args.temp_dir
    if args.ckpt_path is not None and os.path.exists(args.ckpt_path):
        params.CKPT_DICT['stg1_net'] = args.ckpt_path
    if args.epoch is not None:
        params.MAX_EPOCHS = args.epoch
    if args.log_dir is not None and params.DEBUG is False:
        params.LOG_DIR = args.log_dir
    if args.devices is not None:
        params.DEV_IDS = [int(item) for item in args.devices.split(',')]

    params.AUX_CFG_DICT['qp_max_input_keypoints'] = args.qp_max_input_keypoints
    params.AUX_CFG_DICT['qp_compression_ratio'] = args.qp_compression_ratio
    params.AUX_CFG_DICT['qp_distinctiveness_weight'] = args.qp_distinctiveness_weight

    return params


@torch.no_grad()
def main(train_box, qp_layer, data_model) -> None:
    loader = data_model \
                .val_dataloader()
                # .train_dataloader()

    # QP solution gt the threshold is considered selected
    soln_threshold = 1. \
        / (qp_layer.qp_max_input_keypoints * qp_layer.qp_compression_ratio)
    # small number subtracted to avoid rounding error invalidating all points
    soln_threshold -= 1e-6

    rot_errs = []
    trans_errs = []
    failure_count = 0

    for sample in tqdm(loader):
        q_imgs, _, fq_info_, r_imgs, r_info = sample
        loss, res_dict, debug_dict =  train_box.forward(sample)
        if loss is None:
            continue
        sel_q_idx = debug_dict['q_idx']
        q_imgs = torch.cat([q_imgs[:, i:i+1, :, :, :] for i in sel_q_idx], dim=1)
        q_info = train_box.split_info(fq_info_, sel_q_idx)
        corres_q2r_frames = get_corres_ref_2d(
            [q_imgs, q_info, r_imgs, r_info], res=res_dict, args={}
        )

        ori_score = res_dict['log_var'].ravel()
        scores = torch.exp(-ori_score)
        ref_xyz = r_info['pt3d'].to(scores.device)

        qp_soln = qp_layer(scores, ref_xyz)
        selection_mask = (qp_soln > soln_threshold)

        for q_idx, matches in corres_q2r_frames.items():
            q_pos_2d = q_info['pt2d_pos'][q_idx].view(-1, 2)
            r_pos_2d = r_info['pt2d_pos'][0].view(-1, 2)
            r_obs3d = r_info['pt2d_obs3d'][0].view(-1)

            q_pos_sel, ref_pos_sel \
                = corres_pos_from_pairs(q_pos_2d, r_pos_2d, matches[0])

            ref_3d_sel = r_info['pt3d'][0][r_obs3d][matches[0][:, 1]]
            selection_mask_2d = selection_mask[r_obs3d][matches[0][:, 1]]

            q_pos_sel = q_pos_sel[selection_mask_2d]
            ref_pos_sel = ref_pos_sel[selection_mask_2d]
            ref_3d_sel = ref_3d_sel[selection_mask_2d]

            R_gt = q_info['Tcws'][q_idx][:, :3, :3]
            t_gt = q_info['Tcws'][q_idx][:, :3, -1]

            pose_estimate = pose_estimation(q_pos_sel, ref_3d_sel, q_info, q_idx)

            if not pose_estimate['success'] \
                    or 'qvec' not in pose_estimate \
                    or 'tvec' not in pose_estimate:
                failure_count += 1
                continue

            R_est = quaternion2rot(
                torch.from_numpy(pose_estimate['qvec']).unsqueeze(0)
            ).type(R_gt.dtype)
            t_est = torch.from_numpy(pose_estimate['tvec']) \
                        .unsqueeze(0).type(t_gt.dtype)

            R_err = torch.bmm(R_est.transpose(-1, -2), R_gt)
            t_err = torch.norm(t_gt - t_est)
            r_err, _, _ = rotation_from_matrix(R_err[0].cpu().numpy())

            rot_errs.append(r_err)
            trans_errs.append(t_err.item())

    rot_errs = np.rad2deg(np.abs(rot_errs))

    print('mean rot error', np.mean(rot_errs))
    print('mean trans error', np.mean(trans_errs))

    print('median rot error', np.median(rot_errs))
    print('median trans error', np.median(trans_errs))

    print('failures', failure_count)


def pose_estimation(q_pos_2d, ref_3d, q_info, q_idx):
    K = q_info['K'][q_idx]
    cfg = {
        'model': 'PINHOLE',
        'width': int(q_info['dims'][q_idx][1].item()),
        'height': int(q_info['dims'][q_idx][0].item()),
        'params': [
            K[0, 0, 0].item(), K[0, 1, 1].item(), K[0, 0, 2].item(), K[0, 1, 2].item()
        ]
    }
    ret = pycolmap.absolute_pose_estimation(
        q_pos_2d.cpu().numpy() + 0.5,  # COLMAP coordinates
        ref_3d.cpu().numpy(),
        cfg, 12.0
    )
    ret['cfg'] = cfg
    return ret


if __name__ == '__main__':
    args, _ = get_parser().parse_known_args()
    notice_msg('Run on machine: %s with experiment: %s, DEBUG=%s' % (args.hostname, args.exp, args.debug))

    """ Train Parameters -----------------------------------------------------------------------------------------------
    """
    params = TrainParameters(Path('exp_config', '%s.json' % (args.exp)))
    params = overwrite_params(params, args)
    if params.VERBOSE_MODE:
        params.report()

    """ Network --------------------------------------------------------------------------------------------------------
    """
    train_box = SceneSQBox.load_from_checkpoint(
        str(params.CKPT_DICT['stg1_net']), params=params
    )
    train_box.eval()
    train_box.on_train_start()

    qp_layer = QPLayer(params)
    qp_layer.eval()

    """ Dataset --------------------------------------------------------------------------------------------------------
    """
    data_model = RankedDataModule(json_file_path=Path('dataset', 'config', 'data.%s.json' % args.hostname),
                                  train_params=params,
                                  dataset_name=args.dataset)

    # data_model = CachedDataModule(data_module=data_model, max_items=10)
    # data_model.dump_to_disk(Path(params.DEBUG_OUTPUT_DIR, 'cached_data.bin'))
    # exit(0)

    # data_model = CachedDataModule.load_from_disk(Path(params.DEBUG_OUTPUT_DIR, 'cached_data.bin'))

    """ Running --------------------------------------------------------------------------------------------------------
    """
    main(train_box, qp_layer, data_model)
