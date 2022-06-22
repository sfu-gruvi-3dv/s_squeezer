import json
import socket

from core_dl.get_host_name import get_host_name
from core_io.print_msg import *


class TrainParameters:
    """ Store the parameters that used in training """

    """ Hostname, machine that runs the training/validation process """
    HOSTNAME = ''

    """ Debug Flag """
    DEBUG = False

    """ Debug Output Dir """
    DEBUG_OUTPUT_DIR = ''

    """ Device ID """
    DEV_IDS = [2]

    """ Verbose mode (Debug) """
    VERBOSE_MODE = False

    """ Max epochs when training """
    MAX_EPOCHS = 4

    """ Mini-Batch Size """
    LOADER_BATCH_SIZE = 6               # for training batch size

    LOADER_VALID_BATCH_SIZE = 1    # for validation batch size

    """ Pytorch Dataloader  """
    LOADER_NUM_THREADS = 4              # threads

    LOADER_SHUFFLE = True               # data loader shuffle enabled

    LOADER_PIN_MEM = True               # should not be changed this parameter

    """ Learning Rate """
    START_LR = 1e-4

    """ Log Dir """
    LOG_DIR = ''

    """ Log sub directory for specific experiment name """
    LOG_CREATE_EXP_FOLDER = True        # create the sub log directory and named with experiment name + comments
    
    """ Logging Steps """
    LOG_STEPS = 5               # per training-iterations

    """ Validation Steps """
    VALID_STEPS = 200           # per training-iterations

    """ Visualization Steps """
    VIS_STEPS = 100             # per training-iterations

    """ Validation Maximum Batch Number """
    MAX_VALID_BATCHES_NUM = 50

    """ Checkpoint Steps (iteration) """
    CHECKPOINT_STEPS = 5000      # per training-iterations

    """ Continue Step (iteration) """
    LOG_CONTINUE_STEP = 0            # continue step, used for logger

    """ Continue from (Dir) """
    LOG_CONTINUE_DIR = ''           # continue dir

    TQDM_PROGRESS = True

    """ Description """
    NAME_TAG = ''                   # short tag to describe training process

    DESCRIPTION = ''                # train description, used for logging changes
    
    """ CHECKPOINT PATHS """
    CKPT_DICT = dict()
    
    """ Additional Configuration Params """
    AUX_CFG_DICT = dict()

    def __init__(self, from_json_file=None):
        if from_json_file is not None:
            
            if not os.path.exists(from_json_file):
                raise Exception(
                    err_msg('[TrainParameters] File %s is not exists.' % from_json_file, obj=self, return_only=True))
            else:
                msg('Load Train parameters from %s' % from_json_file, self)
            
            with open(from_json_file) as json_data:
                params = json.loads(json_data.read())
                host_name = get_host_name()
                assert host_name is not None
                params = params[host_name]
                json_data.close()

                # Extract parameters
                self.HOSTNAME = str(params['hostname']) if 'hostname' in params else socket.gethostname()
                self.DEBUG = bool(params['debug']) if 'debug' in params else False
                self.DEBUG_OUTPUT_DIR = str(params['debug_output_dir']) if 'debug_output_dir' in params else None
                self.DEV_IDS = params['dev_id'] if 'dev_id' in params else [0]
                self.MAX_EPOCHS = int(params['max_epochs']) if 'max_epochs' in params else 10
                self.LOADER_BATCH_SIZE = int(params['loader_batch_size']) if 'loader_batch_size' in params else 1
                self.LOADER_VALID_BATCH_SIZE = int(params['loader_valid_batch_size'])
                self.LOADER_NUM_THREADS = int(params['loader_num_threads'])
                self.LOADER_SHUFFLE = bool(params['loader_shuffle'])
                self.START_LR = float(params['start_learning_rate'])
                self.VERBOSE_MODE = bool(params['verbose'])
                self.VALID_STEPS = int(params["valid_per_batches"])
                self.MAX_VALID_BATCHES_NUM = int(params["valid_max_batch_num"])
                self.CHECKPOINT_STEPS = int(params['checkpoint_per_iterations'])
                self.VIS_STEPS = int(params['visualize_per_iterations'])
                self.LOG_DIR = str(params['log_dir'])                
                self.LOG_CONTINUE_DIR = str(params['log_continue_dir'])
                self.LOG_CONTINUE_STEP = int(params['log_continue_step'])
                self.LOG_CREATE_EXP_FOLDER = str(params['log_create_exp_folder'])
                self.DESCRIPTION = str(params['description']) if 'description' in params else 'no_description'
                self.NAME_TAG = str(params['name_tag']) if 'name_tag' in params else 'no_name_tag'
                self.CKPT_DICT = params['ckpt_path_dict'] if 'ckpt_path_dict' in params else dict()
                self.AUX_CFG_DICT = params['additional_cfg'] if 'additional_cfg' in params else dict()

    def extract_dict(self):
        params = dict()
        params['description'] = self.DESCRIPTION
        params['name_tag'] = self.NAME_TAG
        params['hostname'] = self.HOSTNAME
        params['verbose'] = self.VERBOSE_MODE        
        params['debug'] = self.DEBUG
        params['debug_output_dir'] = self.DEBUG_OUTPUT_DIR
        params['dev_id'] = self.DEV_IDS
        params['max_epochs'] = self.MAX_EPOCHS
        params['loader_batch_size'] = self.LOADER_BATCH_SIZE
        params['loader_valid_batch_size'] = self.LOADER_VALID_BATCH_SIZE
        params['loader_shuffle'] = self.LOADER_SHUFFLE
        params['loader_num_threads'] = self.LOADER_NUM_THREADS
        params['start_learning_rate'] = self.START_LR
        params['valid_per_batches'] = self.VALID_STEPS
        params['valid_max_batch_num'] = self.MAX_VALID_BATCHES_NUM
        params['checkpoint_per_iterations'] = self.CHECKPOINT_STEPS
        params['visualize_per_iterations'] = self.VIS_STEPS
        params['log_dir'] = self.LOG_DIR
        params['log_continue_dir'] = self.LOG_CONTINUE_DIR
        params['log_continue_step'] = self.LOG_CONTINUE_STEP        
        params['ckpt_path_dict'] = self.CKPT_DICT
        params['additional_cfg'] = self.AUX_CFG_DICT
        params['log_create_exp_folder'] = self.LOG_CREATE_EXP_FOLDER
        return params

    def save(self, json_file_path):
        """ Save the parameters to json file.
        """
        params = self.extract_dict()
        with open(json_file_path, 'w') as out_json_file:
            json.dump(params, out_json_file, indent=2)

    def report(self):
        """ Print the parameters
        """
        title_msg('Parameters')
        print(str(self))

    def __str__(self) -> str:
        """ The description of train parameters
        """
        def print_dict(dict_items: dict, prefix=None):
            strs = ""
            if prefix is None:
                prefix = ""
            for key, value in dict_items.items():
                if isinstance(value, dict):
                    str_ = prefix + '%s:\n' % str(key)
                    str_ += print_dict(value, prefix=prefix + "--" if prefix is not None else "--")
                else:
                    str_ = prefix + '%s: %s' % (str(key), str(value)) + '\n'
                strs += str_
            return strs
        
        params = self.extract_dict()
        out_str = print_dict(params, prefix="")
        return out_str
        
    def to_json(self):
        params = self.extract_dict()
        return json.dumps({self.HOSTNAME: params}, indent=4)
