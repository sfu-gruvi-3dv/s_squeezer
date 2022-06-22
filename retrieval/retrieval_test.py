import torch

from retrieval.rmac_resnet import resnet101_rmac

from core_dl.train_params import TrainParameters
from core_dl.get_host_name import get_host_name
from dataset.data_module import RankedDataModule, CachedDataModule

import argparse, sys
from pathlib import Path
from tqdm import tqdm
import pickle


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train squeezer network')
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
                        help='load from checkpoint path, will overwrite params.')
    parser.add_argument('--epoch',
                        type=int,
                        default=20,
                        help='number of epoches.')
    parser.add_argument('--stage',
                        type=str,
                        help='specific training stages, could be "only_d, d_K, dual_matcher"')
    parser.add_argument('--valid_steps',
                        type=int,
                        help='the number of steps for calling the validation.')
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
    if args.temp_dir is not None and Path(args.temp_dir).exists():
        params.DEBUG_OUTPUT_DIR = args.temp_dir
    if args.ckpt_path is not None and Path(args.ckpt_path).exists():
        params.CKPT_DICT['instance'] = args.ckpt_path
    if args.epoch is not None:
        params.MAX_EPOCHS = args.epoch
    if args.log_dir is not None and params.DEBUG is False:
        params.LOG_DIR = args.log_dir
    if args.devices is not None:
        params.DEV_IDS = [int(item) for item in args.devices.split(',')]
    if args.stage is not None:
        params.AUX_CFG_DICT['stage'] = args.stage
    if args.valid_steps is not None:
        params.VALID_STEPS = args.valid_steps

    return params


if __name__ == '__main__':
    rmac_model = resnet101_rmac()
    state_dict = torch.load('/mnt/Exp/logs/Resnet-101-AP-GeM/Resnet-101-AP-GeM-state-dict.pt')

    new_dict = dict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        new_dict[k] = v

    rmac_model.load_state_dict(new_dict)

    args, _ = get_parser().parse_known_args()

    """ Train Parameters -----------------------------------------------------------------------------------------------
    """
    params = TrainParameters(Path('exp_config', '%s.json' % (args.exp)))
    params = overwrite_params(params, args)
    if params.VERBOSE_MODE:
        params.report()


    dbg_cache_dataset_path = Path(params.DEBUG_OUTPUT_DIR, 'cached_data.bin')

    if not dbg_cache_dataset_path.exists():
        data_model = RankedDataModule(json_file_path=Path('dataset', 'config', 'data.%s.json' % args.hostname),
                                      train_params=params,
                                      dataset_name=args.dataset)
        data_model = CachedDataModule(data_module=data_model, max_items=10)
        data_model.dump_to_disk(dbg_cache_dataset_path)

    data_model = CachedDataModule.load_from_disk(dbg_cache_dataset_path)

    loader = data_model.train_dataloader()

    for sample in tqdm(loader):
        q_imgs = sample[0]
        desc = rmac_model(q_imgs[0])
        print('img:', q_imgs.shape, 'desc:', desc.shape)
