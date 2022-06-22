# -*- coding: utf-8 -*-

import atexit
import datetime
import json
from logging import log
from os import remove
import signal
import socket
import sys
import warnings
import numpy as np
from numpy.lib.arraysetops import isin

import torch
from torch.autograd import Variable

from core_dl.train_params import TrainParameters
from core_io.print_msg import *
from core_io.meta_io import from_meta
from pathlib import Path
from typing import List
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.base import Callback
from einops import asnumpy
import comet_ml
from torch.utils.tensorboard import SummaryWriter


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, every: int, rm_previous=True, **kwargs):
        kwargs['every_n_val_epochs'] = from_meta(kwargs, 'every_n_val_epochs', default=None)
        super().__init__(**kwargs)
        self.every = every
        self.remove_previous = rm_previous
        self.pre_ckpt_path = None

    def __save__(self, trainer: pl.Trainer, pl_module: pl.LightningModule, epoch_end_flag=False, stop_flag=False):
        ckpt_dir = self.dirpath if self.dirpath is not None else trainer.default_root_dir
        assert ckpt_dir is not None
        cur_step = pl_module.global_step
        ckpt_path = Path(ckpt_dir) / (f"iter_{cur_step}.ckpt" if not stop_flag else f"iter_{cur_step}_stopped.ckpt")
        if epoch_end_flag:
            ckpt_path = Path(ckpt_dir) / (f"epoch_{pl_module.current_epoch}.ckpt")

        trainer.save_checkpoint(ckpt_path, weights_only=self.save_weights_only)

        LightningLogger.add_text(trainer.logger.experiment,
                                 text='ckpt saved: %s' % str(ckpt_path),
                                 tag='ckpt', step=pl_module.global_step)
        if self.verbose:
            notice_msg('checkpoint saved to %s [STEP: %d | EPOCH: %d]' %
                       (ckpt_path, pl_module.global_step, pl_module.current_epoch), obj=self)

        return ckpt_path

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs):
        if pl_module.global_step % self.every == 0 and pl_module.global_step > 0:
            ckpt_path = self.__save__(trainer, pl_module)
            if self.remove_previous and self.pre_ckpt_path is not None:
                self.pre_ckpt_path.unlink(missing_ok=True)
            self.pre_ckpt_path = ckpt_path

    def on_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        self.__save__(trainer, pl_module, epoch_end_flag=True)

    def on_keyboard_interrupt(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.__save__(trainer, pl_module, stop_flag=True)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.__save__(trainer, pl_module, stop_flag=True)


class LightningLogger:
    """
        Logger for recording training status.

    Use prefix to classify the term:
        - The Scalars:'Loss' and 'Accuracy' 'Scalar'
        - The Scalars: 'Scalars' used for visualize multiple records in single figure, use 'dict'
          for value (Only in tensorboard)
        - The net instance: 'net' used for parameter histogram (Only in tensorboard)
        - The Image visualization: 'Image' used for visualize bitmap (Only in tensorboard)

    Examples:

        >>> logger = Logger("./runs", "csv|txt|tensorboard")
        >>> logger.add_keys(['Loss/class_layer', 'Accuracy/class_layer'])
        >>> logger.log({'Loss/class_layer': 0.04, 'Accuracy/class_layer': 0.4, 'Iteration': 4})

    """

    """ Logger instance """
    loggers = {}

    def __init__(self, params: TrainParameters, log_types='csv|tensorboard'):
        """

        Args:
            base_dir (str): The base directory stores the log file
            log_types (str):  the log file types including 'csv', 'txt', 'tensorboard'
            tag (str): Additional tag of the log.
            description (str): the description of current experiment.
            hostname (str): the running machine hostname.

        """

        self.params = params

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        self.hostname = self.params.HOSTNAME

        self.log_dir = Path(self.params.LOG_DIR)
        self.exp_name = current_time + '_' + self.hostname + '_' + self.params.NAME_TAG
        self.exp_dir = Path(self.log_dir, self.exp_name)

        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)
        if not self.exp_dir.exists():
            self.exp_dir.mkdir()

        # check if metadata exists
        self.meta_file_path = Path(self.exp_dir, 'meta.json')
        if self.meta_file_path.exists():
            with open(self.meta_file_path) as json_data:
                self.meta_dict = json.load(json_data)
                json_data.close()
                self.meta_dict['description'] += \
                    '#[' + current_time + '_' + self.hostname + ']:\n' + self.params.DESCRIPTION + '\n'
                self.meta_dict['comment'] += \
                    '#[' + current_time + '_' + self.hostname + ']:\n' + self.params.NAME_TAG + '\n'
        else:
            # new meta data
            self.meta_dict = dict()
            self.meta_dict['history'] = \
                '#[' + current_time + '_' + self.hostname + ']:\n' + 'start initial training' + '\n'
            self.meta_dict['description'] = \
                '#[' + current_time + '_' + self.hostname + ']:\n' + self.params.DESCRIPTION + '\n'
            self.meta_dict['comment'] = '#[' + current_time + '_' + self.hostname + ']:\n' + self.params.NAME_TAG + '\n'
            self.meta_dict['lastest_step'] = 0

        self.log_types = log_types
        log_types_token = log_types.split('|')
        for log_type in log_types_token:
            log_type = log_type.strip()
            logger = self.logger_factory(log_type)
            if logger is not None:
                self.loggers[log_type] = logger

        if self.params.VERBOSE_MODE:
            self.report()

        # save parameters
        self.params.save(self.exp_dir / 'params.json')
        self.save_meta_info()

    @staticmethod
    def create(params: TrainParameters, log_types='tensorboard'):
        """ Create instance of LightningLogger
        """
        if params.VERBOSE_MODE:
            title_msg('Logger at %s' % params.HOSTNAME)

        if params.LOG_DIR is None:
            if params.VERBOSE_MODE:
                warn_msg('No logger will be used.', obj='LightningLogger')
            return None
        else:
            return LightningLogger(params, log_types)

    def logger_factory(self, logger_name):
        if logger_name == "csv":
            return pl_loggers.CSVLogger(save_dir=self.exp_dir, name=self.params.NAME_TAG)
        elif logger_name == "tensorboard":
            return pl_loggers.TensorBoardLogger(save_dir=self.exp_dir, name=self.params.NAME_TAG, default_hp_metric=False)
        elif logger_name == "comet":
            comet_api_key = from_meta(self.params.AUX_CFG_DICT, 'comet_api')
            comet_project_key = from_meta(self.params.AUX_CFG_DICT, 'comet_project')

            if comet_api_key is None:
                notice_msg('The comet logger require online access with valid api_key', self)
            comet_logger = pl_loggers.CometLogger(api_key=comet_api_key,
                                                  save_dir=self.exp_dir,
                                                  project_name=comet_project_key,
                                                  experiment_name=self.params.NAME_TAG)
            comet_logger.experiment.log_parameters(self.params.extract_dict())
            return comet_logger
        else:
            return None

    def get_loggers(self) -> List:
        return [v for k, v in self.loggers.items()]

    def print_meta_info(self):
        if self.meta_dict is not None:
            for meta_key in self.meta_dict.keys():
                print('--- ' + meta_key + ' -----\n' + str(self.meta_dict[meta_key]))

    def save_meta_info(self, add_log_dict=None):
        # get the current iteration
        self.meta_dict['lastest_step'] = 0
        if add_log_dict is not None:
            current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
            for add_key in add_log_dict.keys():
                if add_key in self.meta_dict:
                    if add_key == 'history':
                        self.meta_dict[add_key] += '#[' + current_time + '_' + self.hostname \
                                                   + ']:\n' + str(add_log_dict[add_key]) + '\n'
                    else:
                        self.meta_dict[add_key] += add_log_dict[add_key]
                else:
                    self.meta_dict[add_key] = add_log_dict[add_key]

        with open(self.meta_file_path, "w") as json_file:
            json.dump(self.meta_dict, json_file, indent=2)

    def report(self):
        msg('log_dir: %s' % self.log_dir, self)
        msg('exp_name: %s' % self.exp_name, self)

        for _, logger in self.loggers.items():
            if isinstance(logger, pl_loggers.CometLogger):
                msg('api_key: %s, project_name: %s' % (str(logger.api_key), str(logger._project_name)),
                    obj='CometLogger')
            elif isinstance(logger, pl_loggers.TensorBoardLogger):
                msg('saved_dir: %s' % (str(logger.save_dir)),
                    obj='TensorBoardLogger')
            elif isinstance(logger, pl_loggers.CSVLogger):
                msg('saved_dir: %s' % (str(logger.save_dir)),
                    obj='CSVLogger')

    @staticmethod
    def add_text(loggers: list, text: str, tag=None, step=0):
        if loggers is None:
            return

        if not isinstance(loggers, list):
            loggers = [loggers]

        for logger in loggers:
            if isinstance(logger, comet_ml.Experiment):
                logger.log_text(text, step)
            elif isinstance(logger, SummaryWriter):
                tag = 'untitled' if tag is None else tag
                logger.add_text(tag=tag, text_string=text, global_step=step)

    @staticmethod
    def add_sources_code(loggers: list, code_file_paths: list):
        if loggers is None:
            return

        if not isinstance(loggers, list):
            loggers = [loggers]

        if not isinstance(code_file_paths, list):
            code_file_paths = [code_file_paths]

        for logger in loggers:
            if isinstance(logger, comet_ml.Experiment):
                for file_path in code_file_paths:
                    if Path(file_path).exists():
                        logger.log_code(file_path)
            elif isinstance(logger, SummaryWriter):
                code_dir = Path(logger.log_dir) / 'net_def'
                if not code_dir.exists():
                    code_dir.mkdir(parents=True)

                for file_path in code_file_paths:
                    shutil.copy(file_path, code_dir / Path(file_path).name)

    @staticmethod
    def add_system_info(loggers: list, system_info: dict):
        if loggers is None:
            return

        if not isinstance(loggers, list):
            loggers = [loggers]

        for logger in loggers:
            if isinstance(logger, comet_ml.Experiment):
                for k, v in system_info.items():
                    logger.log_system_info(k, v)

    @staticmethod
    def add_image(loggers: list, image: torch.Tensor or np.ndarray, name=None, step=0, ignore_comet=True):
        if loggers is None:
            return

        if not isinstance(loggers, list):
            loggers = [loggers]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        assert image.ndim == 3
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)

        for logger in loggers:
            if isinstance(logger, comet_ml.Experiment) and ignore_comet is False:
                logger.log_image(image, name, step=step)
            elif isinstance(logger, SummaryWriter):
                name = 'untitled' if name is None else name
                logger.add_image(name, image, global_step=step, dataformats='HWC')

    @staticmethod
    def add_hist(loggers: list, vec: torch.Tensor or np.ndarray, name=None, step=0):
        if loggers is None:
            return

        if not isinstance(loggers, list):
            loggers = [loggers]

        if isinstance(vec, np.ndarray):
            vec = torch.from_numpy(vec)

        vec = vec.view(-1).detach().cpu()
        for logger in loggers:
            if isinstance(logger, comet_ml.Experiment):
                logger.log_histogram_3d(asnumpy(vec), name=name, step=step)
            elif isinstance(logger, SummaryWriter):
                name = 'untitled' if name is None else name
                logger.add_histogram(name, vec, global_step=step, max_bins=100)

    @staticmethod
    def add_mul_hist(loggers: list, vecs: list, names: list, step=0):
        assert len(vecs) == len(names)

        if loggers is None:
            return

        if not isinstance(loggers, list):
            loggers = [loggers]

        for vec, name in zip(vecs, names):

            if isinstance(vec, np.ndarray):
                vec = torch.from_numpy(vec)

            vec = vec.view(-1).detach().cpu()
            for logger in loggers:
                if isinstance(logger, comet_ml.Experiment):
                    logger.log_histogram_3d(asnumpy(vec), name=name, step=step)
                elif isinstance(logger, SummaryWriter):
                    logger.add_histogram(name, vec, global_step=step, max_bins=100)
