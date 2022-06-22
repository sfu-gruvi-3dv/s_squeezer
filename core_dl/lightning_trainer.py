from logging import DEBUG
import pytorch_lightning as pl
from core_io.print_msg import title_msg
from core_dl.train_params import TrainParameters
from core_dl.lightning_logger import LightningLogger, PeriodicCheckpoint


def create_pl_trainer(params: TrainParameters, 
                      logger: LightningLogger or None, 
                      gpus=[0], 
                      additional_callbacks=None, 
                      disable_valid=False):
    """ Setup pytorch lightning trainer
    """
    if params.VERBOSE_MODE:
        title_msg('Training' if params.DEBUG is False else 'Training (Debug)')

    trainer_callbacks = []
    if isinstance(additional_callbacks, list):
        trainer_callbacks += additional_callbacks

    # period checkpoint call backs
    if logger is not None:
        trainer_callbacks += [PeriodicCheckpoint(every=params.CHECKPOINT_STEPS,
                                                 dirpath=logger.exp_dir / 'ckpt', rm_previous=True)]

    # create trainer
    trainer = pl.Trainer(gpus=gpus,
                         default_root_dir=logger.exp_dir if logger else params.DEBUG_OUTPUT_DIR,
                         callbacks=trainer_callbacks,
                         weights_save_path=None,
                         flush_logs_every_n_steps=25,
                         val_check_interval=params.VALID_STEPS if disable_valid is False else 1.0,
                         limit_val_batches=params.MAX_VALID_BATCHES_NUM if disable_valid is False else 0.0,
                         log_every_n_steps=params.LOG_STEPS,
                         max_epochs=params.MAX_EPOCHS,
                         weights_summary='top' if params.VERBOSE_MODE else None,
                         progress_bar_refresh_rate=None if params.TQDM_PROGRESS else None,
                         logger=logger.get_loggers() if logger else None)
    
    return trainer
