import json
import os
import copy
from core_io.print_msg import *
from pathlib import Path
from deprecated.sphinx import deprecated, versionadded, versionchanged

""" Utilities for loading items in meta dictionary ---------------------------------------------------------------------
"""


def write_args_to_json(args: dict, json_file_path: str):
    """
        Write the args to json file

    Args:
        args (dict):
        json_file_path (str): output json file path

    """
    with open(json_file_path, 'w') as f:
        json.dump(args, f, indent=2)


def load_args_from_json(json_file_path, verbose=True):
    """
        Load args from json file
    Args:
        json_file_path (str): input json file path
        verbose (bool): report the arg items

    Returns:
        dictionary of args

    """
    with open(json_file_path, 'r') as f:
        args = json.load(f)
    if verbose:
        for k, v in args.items():
            print('%s: %s' % (k, str(v)))
    return args


def from_meta(meta: dict, key: str, default=None) -> object:
    """
        Load item from meta dictionary by key, if the key is not in meta dictionary, use the default instead.
    Args:
        meta (dict): meta dictionary
        key (str): the key string
        default (object): if the key is not in meta dictionary, the default will be returned

    Returns:
        the item in meta dictionary that corresponds to the key

    """
    if meta is None:
        return default
    return meta[key] if key in meta else default


@deprecated(version='1.0', reason="This function will soon be replaced by `path_from_meta_`")
def path_from_meta(meta: dict, key: str, default_path=None, check_exist=True, raise_exception=True):
    """
        Load the path from the meta dictionary with the provided key, if the file is not exist,
        one can raise the exception.

    Args:
        meta (dict): meta dictionary
        key (str): the key string
        default_path (object): if the key is not in meta dictionary, the default_path will be used
        check_exist (bool): check if file exist
        raise_exception (bool): raise the exception if file not exist

    Returns:
        the file path

    """
    path_ = meta[key] if key in meta else default_path
    if check_exist is True:
        if path_ is None or not os.path.exists(path_):
            warn_msg('Path for %s not exist: %s' % (key, str(path_)))
            if raise_exception:
                raise Exception('Path for %s not exist: %s' % (key, str(path_)))
            path_ = None
    return path_

@versionadded(version='1.0', reason="This function will soon be used")
def path_from_meta_(meta: dict, key: str, default_path=None, check_exist=True, raise_exception=False, verbose=False):
    """
        Load the path from the meta dictionary with the provided key, if the file is not exist,
        one can raise the exception.

    Args:
        meta (dict): meta dictionary
        key (str): the key string
        default_path (object): if the key is not in meta dictionary, the default_path will be used
        check_exist (bool): check if file exist
        raise_exception (bool): raise the exception if file not exist
        verbose (bool): print information

    Returns:
        the file path.

    """
    path_ = meta[key] if key in meta else default_path
    if path_ is None:
        path_ = 'null'
    path_ = Path(path_)

    if check_exist is True and not path_.exists():
        if verbose:
            warn_msg('Path for %s not exist: %s' % (key, str(path_)))
        if raise_exception:
            raise Exception('Path for %s not exist: %s' % (key, str(path_)))
        return Path('null')
    return path_


def inv_dict(dict_: dict) -> dict:
    """ Inverse the dictionary items:
    """
    return {v: k for k, v in dict_.items()}

def merge_dict(dict_list: list) -> dict:
    """ Merge two dictionary items
    """
    if len(dict_list) == 0:
        return dict()

    merged = copy.deepcopy(dict_list[0])
    for dict_b in dict_list[1:]:
        for k, v in dict_b.items():
            if k in merged:
                notice_msg('Key: %s is existing in `dict_a`, and will be replaced' % (str(k)))
            merged[k] = copy.deepcopy(v)
    return merged

def copy_attr(from_: object, to_: object):
    """
        Copy the attribute froms `from_` object to `to_` object.

    Args:
        from_: object that provides source attributes
        to_: target object

    """
    for attr in dir(from_):
        instance = getattr(from_, attr)
        if not hasattr(instance, '__call__') and not attr.startswith('__'):
            setattr(to_, attr, getattr(from_, attr))
