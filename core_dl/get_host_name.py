# -*- coding: utf-8 -*-

import os
import socket
import core_io.print_msg as msg

def get_host_name() -> str or None:
    """
        Get the current running matching hostname.

    Returns:
        hostname of current machine.

    """
    name = socket.gethostname()
    parent_host_name = os.getenv("PARENT_HOST")

    if parent_host_name is not None:
        msg.msg('RAW HOSTNAME: %s' % parent_host_name, 'SOCKET')

    if name.startswith('docker'):
        return 'docker'
    elif name.startswith('cs-guv-gpu02'):
        return 'docker'
    elif name.startswith('cs-gruvi-04'):
        return 'docker'
    else:
        msg.err_msg('No entry matches hostname: %s' % (name))

    return None
