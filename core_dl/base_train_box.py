# -*- coding: utf-8 -*-
import datetime
import inspect

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import dataloader
from tqdm.autonotebook import tqdm

import core_dl.module_util as dl_util
from core_dl.expr_ctx import ExprCtx
from core_dl.logger import Logger
from core_io.print_msg import *
from core_dl.train_params import TrainParameters


def is_overridden_func(func):
    obj = func.__self__
    print_m = getattr(super(type(obj), obj), func.__name__)
    return func.__func__ != print_m.__func__


class BaseTrainBox:
    """ The train box that defines the training routines.
    """

    """ verbose mode """
    verbose_mode = False

    """ cuda device id """
    dev_ids = [0]

    """ training optimizer """
    optimizer = None

    """ network model instance """
    model = None

    """ training logger """
    logger = None

    """ loss function """
    criterion = None

    """ training parameters """
    train_params = TrainParameters()

    """ Initialization -------------------------------------------------------------------------------------------------
    """
    def __init__(self, train_params: TrainParameters, load_optimizer=False):

        self.verbose_mode = train_params.VERBOSE_MODE
        self.train_params = train_params
        self.load_optimizer = load_optimizer
        self.ckpt_path_dict = self.train_params.CKPT_DICT
        comment_msg = train_params.NAME_TAG

        if self.verbose_mode:
            notice_msg('Run on machine: %s' % train_params.HOSTNAME, self)

        # set workspace for temp dataset, checkpoints etc.
        self.workspace_dir = train_params.LOG_DIR
        if self.workspace_dir is not None and not os.path.exists(self.workspace_dir):
            os.mkdir(self.workspace_dir)

        # set experiment singleton ctx
        if train_params.DEBUG_OUTPUT_DIR is not None:
            ExprCtx().set_tmp_dir(train_params.DEBUG_OUTPUT_DIR)

        # set the device to run training process
        self._set_dev_id(train_params.DEV_IDS)

        # load Checkpoints if needed
        if isinstance(self.ckpt_path_dict, str) and not file_not_exists(self.ckpt_path_dict, self,
                                                                        raise_exception=False):
            self.model_state_dict = dl_util.load_checkpoints(self.ckpt_path_dict)['net_instance']
        elif isinstance(self.ckpt_path_dict, dict):
            self.model_state_dict = dict()

            # load state dicts
            for ckpt_key in self.ckpt_path_dict:
                if file_not_exists(self.ckpt_path_dict[ckpt_key], self, raise_exception=False):
                    self.model_state_dict[ckpt_key] = torch.load(self.ckpt_path_dict[ckpt_key],
                                                                 map_location=torch.device('cpu'))

        # set dataset or model
        self.data_model = None
        self.train_loader = None
        self.valid_loader = None

        # set network
        self._set_network()
        self.load_statues_dict = dict()
        if self.model_state_dict is not None:
            self._load_network_from_ckpt(self.model_state_dict, self.load_statues_dict)

        # set the loss function
        self._set_loss_func()

        # set the optimizer
        self._set_optimizer()
        self.loaded_optimizer = False
        if self.load_optimizer is True:
            if self.optimizer is not None and \
                    self.model_state_dict is not None and \
                    'optimizer' in self.model_state_dict.keys():
                self.optimizer.load_state_dict(self.model_state_dict['optimizer'])
                self.loaded_optimizer = True

        # set the logger
        self._set_logger(self.workspace_dir, comment_msg)
        if self.verbose_mode and self.logger is not None:
            self.logger.meta_dict['dev_id'] = train_params.DEV_IDS
            self.logger.meta_dict['start_learning_rate'] = train_params.START_LR

            logger = self.logger.get_tensorboard_writer()
            if logger is not None:
                name_str = 'NAME_TAG: %s\nDESCRIPTION: %s' % (self.train_params.NAME_TAG, self.train_params.DESCRIPTION)
                logger.add_text(tag='Description', 
                                text_string=name_str.replace("\n", "  \n"), global_step=0)

                dir_str = 'LOG_DIR: %s\nDEBUG_DUMP_DIR: %s' % (self.logger.log_base_dir, 
                                                               str(self.train_params.DEBUG_OUTPUT_DIR))
                logger.add_text(tag='Directory', 
                                text_string=dir_str.replace("\n", "  \n"), global_step=0)

                logger.add_text(tag='Parameters', 
                                text_string=str(self.train_params).replace("\n", "  \n"), global_step=0)

        # save net definition
        self.model_def_dir = None
        if self.logger is not None:
            self.model_def_dir = os.path.join(self.logger.log_base_dir, 'net_def')
            if not os.path.exists(self.model_def_dir):
                os.mkdir(self.model_def_dir)
            # self.model.save_net_def(self.model_def_dir)
            self.train_params.save(os.path.join(self.model_def_dir, 'train_param.json'))
            self._save_net_def(self.model_def_dir)

        # report the training init
        self.report()

        self.train_start_time = -1

        # state variables
        self.is_training = False

    @staticmethod
    def save_instances_definition_to_dir(instances: list, dir_: str):
        """
            Save the network definition (with other required files) to the disk.

        Args:
            instances (list of object): the list of network instance.
            dir_ (str): the directory to save the network definition scripts.
        """
        for instance in instances:
            file_path = inspect.getfile(instance.__class__)
            shutil.copy(file_path, dir_)

    def _save_net_def(self, model_def_dir: str):
        """
            [Override function]
            Manually Save the network definition (with other required files) to the disk.

        Args:
            model_def_dir (str): the directory that stores the network definition.

        """
        pass

    def _set_dev_id(self, ids: list):
        """
            Set GPU devices by provide available gpu indices.

        Args:
            ids (list): the indices of GPU that will be used.

        """
        self.dev_ids = ids

    def _set_logger(self, workspace_dir: str, comment_msg: str):
        """
            Set the logger that records the training status.

        Args:
            workspace_dir (str): the directory that stores logger records.
            comment_msg (str): the description string.

        """

        if workspace_dir is not None:
            log_base_dir = workspace_dir

            # setup the logger if dir has provided and add default keys
            if isinstance(self.train_params.CKPT_DICT, str) and self.train_params.LOG_CONTINUE_STEP > 0:
                self.logger = Logger(base_dir=self.train_params.LOG_CONTINUE_DIR,
                                     log_types='tensorboard',
                                     tag=comment_msg,
                                     hostname=self.train_params.HOSTNAME,
                                     description=self.train_params.DESCRIPTION,
                                     ckpt_path=self.train_params.CKPT_DICT,
                                     continue_from_step=self.train_params.LOG_CONTINUE_STEP,
                                     create_exp_folder=self.train_params.LOG_CREATE_EXP_FOLDER)
            else:
                self.logger = Logger(base_dir=log_base_dir,
                                     hostname=self.train_params.HOSTNAME,
                                     log_types='tensorboard',
                                     create_exp_folder=self.train_params.LOG_CREATE_EXP_FOLDER,
                                     tag=comment_msg)
            self.logger.add_keys(['Epoch', 'Iteration', 'net', 'Event'])

            # prepare save model dir
            self.model_checkpoint_dir = os.path.join(self.logger.log_base_dir, 'checkpoints')
            if not os.path.exists(self.model_checkpoint_dir):
                os.mkdir(self.model_checkpoint_dir)
        else:
            self.logger = None

    def _set_network(self):
        """
            [Override function]
            Set the network instance.
        """
        self.loaded_network = False

    def _load_network_from_ckpt(self, checkpoint_dict: dict, load_statues_dict: dict) -> bool:
        """
            [Override function]
            Load network parameters from checkpoint, the load status are stored in load_statues_dict.

        Args:
            checkpoint_dict (dict): the checkpoint dict.
            load_statues_dict (dict): the status to be stored.

        Returns:
             boolean indicate whether successfully loaded or not.

        """
        pass

    def _set_optimizer(self):
        """
            [Override function]
            Set the optimizer.
        """
        pass

    def _set_loss_func(self):
        """
            [Override function]
            Set the loss function.
        """
        pass

    def _add_log_keys(self, keys: list):
        """
            Add pre-defined log keys

        Args:
            keys (list): the list of pre-defined keys

        """
        if self.logger is not None:
            self.logger.add_keys(keys)

    """ Training Routines ----------------------------------------------------------------------------------------------
    """

    def _prepare_train(self):
        """
            [Override function]
            The operation before the training, used for preparing the training status.  (e.g. net.train())
        """

        pass

    def _prepare_eval(self):
        """
            [Override function]
            The operation before the evaluation, used for preparing the evaluation status. (e.g. net.eval())
        """
        pass

    def _optimizer_update(self):
        """
            [Override function]
            The operation update the optimizer, used for change the learning rate etc.
        """
        pass

    def _train_feed(self,
                    train_sample: list,
                    cur_train_epoch: int,
                    cur_train_itr: int,
                    logger=None,
                    eval_flag=False) -> dict:
        """
            [Override function]
            Train the model by feeding one sample.

        Args:
            train_sample (list): single samples that will be feed in network for training.
            cur_train_epoch (int): the current training epoch.
            cur_train_itr (int): the current training iteration.
            logger (object): the logger instance that manually add information to it.
            eval_flag (bool): evaluation flag, used for manually set net.eval().

        Returns:
             recorded dict for logger.

        """

        if self.is_training is False:
            self._prepare_train()
            self.is_training = True
        return dict()

    def _valid_loop(self, valid_loader: dataloader, cur_train_epoch, cur_train_itr) -> dict:
        """
            [Override function]
            Validate the training process by providing multiple validating samples.

        Args:
            valid_loader (dataloader): the dataloader that iterate the validation set.
            cur_train_epoch (int): the current training epoch.
            cur_train_itr (int): the current training iteration.

        Returns:
            recorded dict for logger.

        """

        self.is_training = False
        self._prepare_eval()

        return dict()

    def _prepare_train_loop(self):
        """
            [Override function]
            The operation before each training iteration.
        """
        pass

    def _start_of_train_epoch(self, epoch, itr):
        """
            [Override function]
            The operation before each training epoch.
        """
        pass

    def _end_of_train_epoch(self, epoch, itr):
        """
            [Override function]
            The operation after each training epoch.
        """
        pass

    def train_loop(self, data_model: LightningDataModule):
        """
            The training loop that fit the dataset.

        Args:
            data_model (LightningDataModule): the data model contains the training samples.

        """

        # prepare the dataloader if the input parameter is instance of dataset
        self.data_model = data_model
        self.train_loader = data_model.train_dataloader()
        self.valid_loader = data_model.val_dataloader()

        # prepare the training process
        self._prepare_train_loop()

        epoch, itr = 0, 0
        self.train_start_time = datetime.datetime.now()
        title_msg('Running')

        try:
            for epoch in range(0, self.train_params.MAX_EPOCHS):

                train_loader, valid_loader = self.train_loader, self.valid_loader
                progress = tqdm(total=len(train_loader), ncols=100, leave=False) \
                    if self.train_params.TQDM_PROGRESS else None

                self._start_of_train_epoch(epoch, itr)

                for train_batch_idx, train_sample in enumerate(train_loader):

                    itr += 1
                    if self.train_params.TQDM_PROGRESS:
                        progress.update(1)
                        progress.set_description(
                            '[Train] epoch = %d, lr=%f' % (epoch, dl_util.get_learning_rate(self.optimizer))
                        )

                    self.optimizer.zero_grad()

                    # update optimizer
                    self._optimizer_update()

                    # forward and backward
                    log_dict = self._train_feed(train_sample, epoch, itr)
                    if log_dict:
                        log_dict['Iteration'] = itr + 1
                        log_dict['Epoch'] = epoch
                        log_dict['Event'] = 'Training'

                    # optimize the parameters
                    self.optimizer.step()

                    # log the training information
                    if self.logger is not None and self.check_log_step(itr):
                        self.logger.log(log_dict)

                    # save the training checkpoints every 'checkpoint_steps'
                    if self.check_checkpoint_step(itr):
                        self.save_checkpoint(epoch, itr)

                    # do validation
                    if self.check_valid_step(itr) and valid_loader is not None:

                        if self.train_params.TQDM_PROGRESS is not None:
                            progress.set_description('[Valid]')

                        with torch.no_grad():
                            valid_log_dict = self._valid_loop(valid_loader, epoch, itr)
                            torch.cuda.empty_cache()

                        # log the validation
                        if valid_log_dict and self.logger is not None:
                            valid_log_dict['Iteration'] = itr + 1
                            valid_log_dict['Epoch'] = epoch
                            valid_log_dict['Event'] = 'Validating'
                            self.logger.log(valid_log_dict)

                self._end_of_train_epoch(epoch, itr)

                # save the checkpoint
                self.save_checkpoint(epoch, itr)

                if self.train_params.TQDM_PROGRESS:
                    progress.close()

        except Exception as e:
            import traceback
            print(traceback.format_exc())

            print('[Exception]: ' + str(e))
            self.save_checkpoint(epoch, itr)

    def test_loop(self, data_model: LightningDataModule, max_test_itr=-1):
        """
            The test loop that evaluate the model.

        Args:
            data_model (LightningDataModule): the data model contains the testing samples.
            max_test_itr (int): the maximum number of test iterations.

        """

        valid_loader = data_model.val_dataloader()

        # prepare the training process
        self._prepare_train_loop()

        self.train_start_time = datetime.datetime.now()
        title_msg('Running')

        try:

            # forward and backward
            log_dict = self._test_loop(valid_loader, max_test_itr)

            self._test_loop_post_processing(log_dict)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            err_msg('[Exception]: ' + str(e), self)

    def _test_loop(self, test_loader: dataloader, max_test_itr: int) -> dict:
        """
            [Override function]
            The test operations that evaluate multiple test samples.

        Args:
            test_loader (dataloader): the dataset that contains test samples.
            max_test_itr (int): maximum test iterations.

        Returns:
            dictionary that contains statistic information.

        """
        return dict()

    def _test_loop_post_processing(self, log_dict):
        """
            [Override function]
            The operations that used to process the results of test samples (stored in `log_dict`).

        Args:
            log_dict (dict): the results from tested samples.

        """
        pass

    def check_log_step(self, itr):
        """
            Check if perform the log operation at current iteration.

        Args:
            itr (int): current iteration

        Returns:
            boolean that marks the log operation.

        """
        return (itr + 1) % self.train_params.LOG_STEPS == 0

    def check_checkpoint_step(self, itr):
        """
            Check if storing the checkpoint at current iteration.

        Args:
            itr (int): current iteration

        Returns:
            boolean that marks storing the checkpoint at current iteration.

        """
        return self.train_params.CHECKPOINT_STEPS > 0 and (itr + 1) % self.train_params.CHECKPOINT_STEPS == 0

    def check_valid_step(self, itr):
        """
            Check if starting validation loop at current iteration.

        Args:
            itr (int): current iteration

        Returns:
             boolean that marks validating operation at current iteration.

        """
        return self.train_params.VALID_STEPS > 0 and (itr + 1) % self.train_params.VALID_STEPS == 0

    def check_visualization_step(self, itr):
        """
            Check if perform the visualization at current iteration.

        Args:
            itr (int): current iteration

        Returns:
            boolean that marks the visualization operation.

        """
        return self.train_params.VIS_STEPS > 0 and (itr + 1) % self.train_params.VIS_STEPS == 0

    def _save_checkpoint_dict(self, checkpoint_dict: dict):
        """ Define the items or instances that save to checkpoint_dict with its key.
        """
        pass

    def save_checkpoint(self, epoch=None, itr=None, path=None):
        """
            Save the checkpoint to disk.

        Args:
            epoch (int): the current epoch during training.
            itr (int): the current iteration during training.
            path (str): the file path of checkpoint.

        """
        if path is not None:
            checkpoint_dict = {}
            self._save_checkpoint_dict(checkpoint_dict)
            dl_util.save_checkpoint(checkpoint_dict, filename=path, is_best=True)
            notice_msg('[Checkpoints] Save checkpoint to ' + path, self)

        if self.logger is not None:
            checkpoint_file_name = os.path.join(self.model_checkpoint_dir, 'iter_%06d.pth.tar' % (itr + 1))
            if self.logger is not None:
                self.logger.log({
                    'Iteration': itr + 1,
                    'Epoch': epoch,
                    'Event': "Check point saved to %s" % checkpoint_file_name
                })

            checkpoint_dict = {
                'log_dir': self.logger.log_base_dir,
                'iteration': itr + 1,
                'epoch': epoch + 1,
                'optimizer': self.optimizer.state_dict(),
            }

            self._save_checkpoint_dict(checkpoint_dict)
            dl_util.save_checkpoint(checkpoint_dict, filename=checkpoint_file_name, is_best=False)

            if self.verbose_mode:
                notice_msg('[Checkpoints] Save checkpoint to ' + checkpoint_file_name, self)

            tf_logger = self.logger.get_tensorboard_writer()
            if tf_logger is not None:
                tf_logger.add_text(tag='Checkpoints', 
                                   text_string='Save checkpoint to ' + checkpoint_file_name, global_step=(itr + 1))

            self.logger.save_meta_info(add_log_dict={'history': 'Save checkpoint to %s' % checkpoint_file_name})

    """ Verbose --------------------------------------------------------------------------------------------------------
    """

    def _report_network(self):
        """
            [Override function]
            Print network structure information
        """
        print('Network information, N/A')

    def _report_optimizer(self):
        """ Print the optimizer information
        """
        if self.optimizer is not None:
            msg("Learning Rate: %e" % (dl_util.get_learning_rate(self.optimizer)), obj=self.optimizer)

    def report(self):
        """ Report the training details. parameters, optimizer, network etc.
        """

        # optimizer ----------------------------------------------------------------------------------------------------
        title_msg('Training Parameters Overview')
        self.train_params.report()

        # optimizer ----------------------------------------------------------------------------------------------------
        title_msg('Optimizer Overview')
        self._report_optimizer()
        if self.loaded_optimizer:
            notice_msg('Loaded optimizer', obj=self, emphasize=False)

        # network ------------------------------------------------------------------------------------------------------
        title_msg('Network Overview')
        self._report_network()
        if len(self.load_statues_dict) > 0:
            for key in self.load_statues_dict:
                ckpt_path = self.train_params.CKPT_DICT[key]
                status = self.load_statues_dict[key]
                if status is True:
                    notice_msg('Loaded module %s from: %s' % (key, ckpt_path), obj=self)
                else:
                    warn_msg('No module %s load from: %s' % (key, ckpt_path), obj=self)
        elif isinstance(self.train_params.CKPT_DICT, str):
            for key in self.load_statues_dict:
                status = self.load_statues_dict[key]
                if status is True:
                    notice_msg('Loaded module %s from: %s' % (key, self.train_params.CKPT_DICT), obj=self)
                else:
                    warn_msg('No module %s load from: %s' % (key, self.train_params.CKPT_DICT), obj=self)
        else:
            warn_msg('Will not load main checkpoint from disk', self)

        # logger -------------------------------------------------------------------------------------------------------
        title_msg('Logger Overview')
        if self.logger is not None:
            self.logger.report_logger()
        elif self.logger is None:
            warn_msg('will not write any log to disk.', self)

