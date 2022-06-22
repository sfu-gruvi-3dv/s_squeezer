from comet_ml import Experiment
from pytorch_lightning import callbacks, loggers, trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch, argparse, sys
torch.manual_seed(1000)
from core_dl.train_params import TrainParameters
from exp.scene_sq_box_naive_aggre import SceneSQBox
from dataset.data_module import RankedDataModule, CachedDataModule
from core_dl.get_host_name import get_host_name
from core_io.print_msg import *
import pytorch_lightning as pl
from core_dl.lightning_logger import LightningLogger, PeriodicCheckpoint
from core_dl.lightning_trainer import create_pl_trainer
from pathlib import Path


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

    return params


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
    train_box = SceneSQBox(params)

    """ Logger ---------------------------------------------------------------------------------------------------------
    """
    logger = LightningLogger.create(params=params, log_types='tensorboard|comet')

    """ Dataset --------------------------------------------------------------------------------------------------------
    """
    # if not params.DEBUG:
    # data_model = RankedDataModule(json_file_path=Path('dataset', 'config', 'data.%s.json' % args.hostname),
    #                               train_params=params,
    #                               dataset_name=args.dataset)
    
    #     # data_model = CachedDataModule(data_module=data_model, max_items=10)
    #     # data_model.dump_to_disk(Path(params.DEBUG_OUTPUT_DIR, 'cached_data.bin'))
    #     # exit(0)
    
    data_model = CachedDataModule.load_from_disk(Path(params.DEBUG_OUTPUT_DIR, 'cached_data.bin'))

    """ Running --------------------------------------------------------------------------------------------------------
    """
    trainer = create_pl_trainer(params=params, logger=logger)
    trainer.fit(train_box, data_model)
