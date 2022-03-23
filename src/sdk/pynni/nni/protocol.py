# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import logging
import threading
from enum import Enum


class CommandType(Enum):
    # in
    Initialize = b'IN'
    RequestTrialJobs = b'GE'
    ReportMetricData = b'ME'
    UpdateSearchSpace = b'SS'
    ImportData = b'FD'
    AddCustomizedTrialJob = b'AD'
    TrialEnd = b'EN'
    Terminate = b'TE'
    Ping = b'PI'

    # out
    Initialized = b'ID'
    NewTrialJob = b'TR'
    SendTrialJobParameter = b'SP'
    NoMoreTrialJobs = b'NO'
    KillTrialJob = b'KI'

_lock = threading.Lock()
try:
    _in_file = open(0, 'rb')
    _out_file = open(1, 'wb')
    logging.getLogger(__name__).debug("success open fd")
except OSError:
    _msg = 'IPC pipeline not exists, maybe you are importing tuner/assessor from trial code?'
    logging.getLogger(__name__).debug(_msg)


def send(command, data):
    """Send command to Training Service.
    command: CommandType object.
    data: string payload.
    """
    global _lock
    try:
        _lock.acquire()
        data = data.encode('utf8')
        assert len(data) < 1000000, 'Command too long'
        msg = b'%b%06d%b' % (command.value, len(data), data)
        logging.getLogger(__name__).debug('Sending command, data: [%s]', msg)
        _out_file.write(msg)
        _out_file.flush()
    finally:
        _lock.release()


def receive():
    """Receive a command from Training Service.
    Returns a tuple of command (CommandType) and payload (str)
    """
    logging.getLogger(__name__).debug("block reading")
    #a = sys.stdin.read()
    #logging.getLogger(__name__).debug("read {}".format(a))
    header = _in_file.read(8)
    logging.getLogger(__name__).debug('Received command, header: [%s]', header)
    if header is None or len(header) < 8:
        # Pipe EOF encountered
        logging.getLogger(__name__).debug('Pipe EOF encountered')
        return None, None
    length = int(header[2:])
    data = _in_file.read(length)
    command = CommandType(header[:2])
    data = data.decode('utf8')
    logging.getLogger(__name__).debug('Received command, data: [%s]', data)
    return command, data
