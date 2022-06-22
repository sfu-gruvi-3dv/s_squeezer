# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import shutil
from torch.autograd import Variable
from collections import OrderedDict
from collections import deque
from core_io.print_msg import warn_msg


def get_learning_rate(optimizer: torch.optim.Optimizer):
    """
        Get the learning rate from pytorch's optimizer
    Args:
        optimizer ():

    Returns:
        learning rate from optimizer

    """
    lr = None
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr


def save_checkpoint(state: dict, is_best: bool, filename='checkpoint.pth.tar'):
    """
        Save checkpoint to disk

    Args:
        state (dict): the network instance's state in dictionary (serialization)
        is_best (bool): mark if the current model is the best or not.
        filename (str): file name

    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoints(file_path: str):
    """
        Load checkpoint from disk

    Args:
        file_path (str): the file contains checkpoint.

    Returns:
        checkpoint instance in dictionary

    """
    return torch.load(file_path, map_location=torch.device('cpu'))


def freeze_bn_layer(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()


def load_state_dict_by_key(model: torch.nn.Module, state_dict: dict, verbose=True) -> bool:
    """
        Load the checkpoint and matched by key, used to load network with different modules but part module share
        with the same key.

    Args:
        model (torch.nn.Module): network instance
        state_dict (dict): the dictionary contains network weights.
        verbose (bool): print information

    Returns:
        the status of loading the weights.

    """
    cur_model_state = model.state_dict()
    for k in cur_model_state.keys():
        if k not in state_dict and verbose:
            warn_msg('Missing Module: %s' % k, obj='Load State')
            
    input_state = {k: v for k, v in state_dict.items() if
                   k in cur_model_state and v.size() == cur_model_state[k].size()}
    cur_model_state.update(input_state)
    model.load_state_dict(cur_model_state)

    return True


def assign_layer_tags(module: torch.nn.Module):
    """
        Assign an unique tag for each layer.

    Args:
        module (torch.nn.Module): network instance

    """

    q = deque()

    # Add first submodules to queue
    for (name, module) in module.modules():
        setattr(module, 'tag', name)
        q.append(module)

    while len(q) != 0:
        front_module = q.popleft()
        module_tag = getattr(front_module, 'tag')
        for (name, submodule) in front_module.modules().items():
            setattr(submodule, 'tag', module_tag + '.' + name)
            q.append(submodule)


def summary_layers(model: torch.nn.Module, input_size: list):
    """
        Summarize the network layers.

    Args:
        model (torch.nn.Module): network instance
        input_size (list): the dim of input tensor that used for tracing the layers.

    """

    def register_hook(module):
        def hook(module_, input_, output):
            class_name = str(module_.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)
            if hasattr(module_, 'tag'):
                module_tag = getattr(module_, 'tag')
            else:
                module_tag = ''

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['type'] = class_name
            summary[m_key]['idx'] = module_idx
            summary[m_key]['tag'] = module_tag
            summary[m_key]['input_shape'] = list(input_[0].size())
            summary[m_key]['input_shape'][0] = -1
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module_, 'weight'):
                params += torch.prod(torch.LongTensor(list(module_.weight.size())))
                summary[m_key]['trainable'] = module_.weight.requires_grad
            if hasattr(module_, 'bias') and hasattr(module_.bias, 'size'):
                params += torch.prod(torch.LongTensor(list(module_.bias.size())))
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and
                not isinstance(module, nn.ModuleList) and
                not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to feed the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1, *in_size)).type(torch.FloatTensor) for in_size in input_size]
    else:
        x = Variable(torch.rand(1, *input_size)).type(torch.FloatTensor)

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # print information
    print('-------------------------------------------------------------------------------------------------')
    line_new = '{:>20} {:>20} {:>10} {:>25} {:>15}'.format('Type', 'Tag', 'Index', 'Output Shape', 'Param #')
    print(line_new)
    print('=================================================================================================')
    total_params = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = '{:>20} {:>20} {:>10} {:>25} {:>15}'.format(summary[layer]['type'],
                                                               summary[layer]['tag'],
                                                               str(summary[layer]['idx']),
                                                               str(summary[layer]['output_shape']),
                                                               summary[layer]['nb_params'])
        total_params += summary[layer]['nb_params']
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] is True:
                trainable_params += summary[layer]['nb_params']
        print(line_new)
    print('=================================================================================================')
    print('Total params: ' + str(total_params))
    print('Trainable params: ' + str(trainable_params))
    print('Non-trainable params: ' + str(total_params - trainable_params))
    print('-------------------------------------------------------------------------------------------------')
    # return summary
