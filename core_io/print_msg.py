# -*- coding: utf-8 -*-

import os
import shutil

from colorama import Fore, Back, Style


def title_msg(msg_str: str, emphasize=False):
    """
        Print the message in title format.
    """
    terminal_size = shutil.get_terminal_size(fallback=(120, 50))
    columns = terminal_size[0]
    msg_text = '[ %s ]' % msg_str
    
    if emphasize:
        print(Fore.WHITE + Back.BLUE + Style.BRIGHT + msg_text.center(columns, '*') + Style.RESET_ALL)
    else:
        print(Fore.BLUE + msg_text.center(columns, '*') + Style.RESET_ALL)


def subtitle_msg(msg_str: str):
    """ Print the message in sub-title format.
    """
    terminal_size = shutil.get_terminal_size(fallback=(120, 50))
    columns = terminal_size[0]
    msg_text = '{ %s }' % msg_str

    print(Fore.BLUE + msg_text.center(columns, '-') + Style.RESET_ALL)


def warn_msg(msg_str: str, obj=None, emphasize=False, return_only=False):
    """
        Print the message in warning format.

    Args:
        msg_str (str): the message
        obj (object): the object that report the message
        emphasize (bool): emphasize the message if needed
        return_only (bool): just return the message and not to print to console

    """
    if obj is not None and not isinstance(obj, str):
        obj = obj.__class__.__name__
    msg_text = '[%s] WARNING: %s' % (obj, msg_str) if obj is not None else 'WARNING: %s' % msg_str
    
    if return_only:
        return msg_text
    
    if emphasize:
        print(Fore.BLACK + Back.YELLOW + Style.BRIGHT + msg_text + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + msg_text + Fore.RESET)


def msg(msg_str: str, obj=None, return_only=False):
    """
        Print the message.

    Args:
        msg_str (str): the message
        obj (object): the object that report the message
        return_only (bool): just return the message and not to print to console

    """
    if obj is not None and not isinstance(obj, str):
        obj = obj.__class__.__name__
    msg_text = '[%s] %s' % (obj, msg_str) if obj is not None else '%s' % msg_str
    if return_only:
        return msg_text
        
    print(msg_text)
    return msg_text


def err_msg(msg_str: str, obj=None, emphasize=True, return_only=False):
    """
        Print the message in error format.

    Args:
        msg_str (str): the message
        obj (object): the object that report the message.
        emphasize (bool): emphasize the message if needed.
        return_only (bool): just return the message and not to print to console.

    Returns:
        message that contains the object that report the message.

    """
    if obj is not None and not isinstance(obj, str):
        obj = obj.__class__.__name__
    msg_text = '[%s] ERROR: %s' % (obj, msg_str) if obj is not None else 'ERROR: %s' % msg_str
    if return_only:
        return msg_text
    
    if emphasize:
        print(Fore.BLACK + Back.RED + Style.BRIGHT + msg_text + Style.RESET_ALL)
    else:
        print(Fore.RED + msg_text + Fore.RESET)


def file_not_exists(file_path: str, obj=None, raise_exception=True):
    """
        Check if file exists.

    Args:
        file_path (str): the message
        obj (object): the object that report the message.
        raise_exception (bool): raise the exception if file is not exists.

    Returns:
        boolean of the file exists status.

    """

    if not os.path.exists(file_path):
        if raise_exception is False:
            warn_msg('File not exists: %s' % file_path, obj, return_only=False)
        else:
            err_text = err_msg('File not exists: %s' % file_path, obj, return_only=True)
            if raise_exception:
                raise Exception(err_text)
        return False
    else:
        return True


def notice_msg(msg_str: str, obj=None, emphasize=False, return_only=False):
    """
        Print the message in notice format.

    Args:
        msg_str (str): the message
        obj (object): the object that report the message.
        emphasize (bool): emphasize the message if needed.
        return_only (bool): just return the message and not to print to console.

    Returns:
        message that contains the object that report the message.

    """
    if obj is not None and not isinstance(obj, str):
        obj = obj.__class__.__name__
    msg_text = '[%s] NOTE: %s' % (obj, msg_str) if obj is not None else 'NOTE: %s' % msg_str
    if return_only:
        return msg_text
    
    if emphasize:
        print(Fore.BLACK + Back.GREEN + Style.BRIGHT + msg_text + Style.RESET_ALL)
    else:
        print(Fore.GREEN + msg_text + Fore.RESET)


def dim_msg(msg_str: str, obj=None, return_only=False):
    """
        Print the message in dimmed format.

    Args:
        msg_str (str): the message
        obj (object): the object that report the message.
        return_only (bool): just return the message and not to print to console.

    Returns:
        message that contains the object that report the message.

    """
    if obj is not None and not isinstance(obj, str):
        obj = obj.__class__.__name__
    msg_text = '[%s] %s' % (obj, msg_str) if obj is not None else '%s' % msg_str
    if return_only:
        return msg_text
        
    if obj is not None:
        print(Style.DIM + msg_text + Style.RESET_ALL)
    else:
        print(Style.DIM + msg_text + Style.RESET_ALL)
