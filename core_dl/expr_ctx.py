# -*- coding: utf-8 -*-

from core_io.print_msg import *


class Singleton(type):
    """ Base singleton class
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ExprCtx(metaclass=Singleton):
    """ Singleton instance for storing intermediate variables, mainly used in debugging.

    Examples:
        >>> ExprCtx().set_ckpt_dir('')
        >>> gpu_dev_id = ExprCtx().gpu_workspace_dev

    """

    """ Temporary directory """
    tmp_dir = None

    """ Directory that stores the checkpoints """
    ckpt_dir = None

    """ Directory that stores th log """
    log_dir = None

    """ The GPU devices that used in the script """
    gpu_workspace_dev = None

    """ Additional attribute meta stores the intermediate status """
    attr_dict = dict()

    """ Enable or disable the ExprCtx """
    disable = False

    def set_tmp_dir(self, dir_path):
        msg('Set temporary output dir: %s' % dir_path, self)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.tmp_dir = dir_path

    def set_ckpt_dir(self, ckpt_dir):
        msg('Set checkpoint dir: %s' % ckpt_dir, self)        
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        self.ckpt_dir = ckpt_dir

    def set_log_dir(self, log_dir):
        msg('Set log dir: %s' % log_dir, self)                
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir

    def set_workspace_gpu_dev(self, dev):
        msg('Set workspace gpu: %s' % str(dev), self)                        
        self.gpu_workspace_dev = dev
