# -*- coding: utf-8 -*-
import datetime
import inspect
from pathlib import Path
from typing import List

import torch
from pytorch_lightning import LightningDataModule, loggers
from torch.utils.data import dataloader
from tqdm.autonotebook import tqdm

import core_dl.module_util as dl_util
from core_dl.expr_ctx import ExprCtx
from core_dl.logger import Logger
from core_dl.lightning_logger import LightningLogger
from core_io.print_msg import *
from core_dl.train_params import TrainParameters
import pytorch_lightning as pl


def is_overridden_func(func):
    obj = func.__self__
    print_m = getattr(super(type(obj), obj), func.__name__)
    return func.__func__ != print_m.__func__


class BaseLightningModule(pl.LightningModule):
    """ The extended LightningModule version with useful tools.
    """

    """ verbose mode """
    verbose_mode = False

    """ Initialization -------------------------------------------------------------------------------------------------
    """
    def __init__(self, train_params: TrainParameters, auto_optimize=True, auto_assign_devices=True):
        super().__init__()
        self.args = train_params

        self.dev_ids = self.args.DEV_IDS
        self.verbose_mode = self.args.VERBOSE_MODE
        self.ckpt_path_dict = self.args.CKPT_DICT
        self.automatic_optimization = auto_optimize
        self.auto_assign_devices = auto_assign_devices

        if self.verbose_mode:
            title_msg('Network: %s' % self.args.NAME_TAG)

        # set experiment singleton ctx
        if self.args.DEBUG_OUTPUT_DIR is not None:
            ExprCtx().set_tmp_dir(self.args.DEBUG_OUTPUT_DIR)

        self._set_network(self.args.AUX_CFG_DICT)
        self.dev2models = self._instance_devices()
        self.model2dev = None
        if self.auto_assign_devices is False and self.dev2models is None:
            e = err_msg("Set the device of each model by overloading the function `_instance_devices()`",
                        obj=self, return_only=True)
            raise Exception(e)
        elif self.dev2models is not None:
            self.model2dev = dict()
            for device, models in self.dev2models.items():
                for m in models:
                    self.model2dev[m] = device

        # report the training init
        if self.verbose_mode:
            self.report()

    def __move_to_devices__(self):
        """ Move model to device
        """
        if self.auto_assign_devices is False:
            for device, models in self.dev2models.items():
                for m in models:
                    m.to(device)
                    if self.args.VERBOSE_MODE:
                        notice_msg('Move %s to device: %s' % (type(m), str(device)))

    def load_from(self, ckpt_paths: dict):
        pass

    def device_of(self, obj):
        return self.model2dev[obj]

    def on_train_start(self) -> None:
        super(BaseLightningModule, self).on_train_start()
        self.__move_to_devices__()

        if self.logger is not None and self.logger.experiment is not None:

            # log parameters
            LightningLogger.add_text(self.logger.experiment, str(self.args), tag='params', step=0)

            # log the source codes
            scripts_paths = []
            for i in self._instance_scripts():
                if isinstance(i, str):
                    scripts_paths.append(str(Path(i).absolute()))
                elif hasattr(i, '__class__'):
                    scripts_paths.append(inspect.getfile(i.__class__))

            LightningLogger.add_sources_code(self.logger.experiment, scripts_paths)

    def on_test_start(self):
        super(BaseLightningModule, self).on_test_start()
        self.__move_to_devices__()

    def on_validation_start(self):
        super(BaseLightningModule, self).on_validation_start()
        self.__move_to_devices__()

    def on_predict_model_eval(self):
        super(BaseLightningModule, self).on_predict_model_eval()
        self.__move_to_devices__()

    def on_test_model_eval(self):
        super(BaseLightningModule, self).on_test_model_eval()
        self.__move_to_devices__()

    def on_validation_model_eval(self):
        super(BaseLightningModule, self).on_validation_model_eval()
        self.__move_to_devices__()

    def eval(self):
        _ = super(BaseLightningModule, self).eval()
        notice_msg('Setting Lightning Model to eval model.', obj=self)
        self.__move_to_devices__()

    def _instance_scripts(self) -> List: 
        """ set the instance to be saved (its scripts)
        """
        return []

    def _instance_devices(self) -> dict or None:
        """ set instance devices when self.auto_assign_devices = False
        """
        return None

    def on_visualize(self, ignore_first=False):
        if ignore_first and self.global_step == 0:
            return False
        else:
            return self.global_step % self.args.VIS_STEPS == 0

    """ Set network ----------------------------------------------------------------------------------------------------
    """
    def _set_network(self, args: dict or None):
        """
            [Override function]
            Set the network instance.
        """
        self.loaded_network = False

    """ Verbose --------------------------------------------------------------------------------------------------------
    """
    def report(self):
        """ Report the training details. parameters, optimizer, network etc.
        """
        print('Network information, N/A')
        if not self.auto_assign_devices:
            notice_msg('Disabled: automatic assigning device')
        if not self.automatic_optimization:
            notice_msg("Disabled: pl's automatic optimizer")