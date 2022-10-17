import argparse
from concurrent.futures import thread
import os
import json
from signal import signal
from time import sleep
from django import dispatch
from .launcher_utils import validate_all_content
import pkg_resources
from colorama import init
from .common_utils import print_error, get_yml_content
from .config_utils import Config, Experiments
import logging
import random
import string
import base64
import signal
import subprocess
import threading
import requests
import signal
import sys

init(autoreset=True)
logger = logging.getLogger("aiperfctrl")
log_format = "INFO: %(asctime)s %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)

if os.environ.get('COVERAGE_PROCESS_START'):
    import coverage
    coverage.process_startup()

"""
/**
 * Generate command line to start automl algorithm(s),
 * either start advisor or start a process which runs tuner and assessor
 *
 * @param expParams: experiment startup parameters
 *
 */
function getMsgDispatcherCommand(expParams: ExperimentParams): string {
    const clonedParams = Object.assign({}, expParams);
    delete clonedParams.searchSpace;
    return `${getCmdPy()} -m nni --exp_params ${Buffer.from(JSON.stringify(clonedParams)).toString('base64')}`;
}

"""
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

def read_response(fio):
    header = fio.read(8)
    logger.info('Received response: [%s]', header)
    data = ""
    try:
        length = int(header[2:])
    except:
        logging.info("ERROR!")
        logging.info(fio.read(30))
    data = fio.read(length)
    command = CommandType(header[:2])
    data = data.decode('utf8')
    logging.getLogger(__name__).debug('Received command, data: [%s...]', data[:40])
    return command, data

dispatch_pid = -1

def term_sig_handler(signum, frame):
    logger.info("AIPerf controller exiting")
    if(dispatch_pid!=-1):
        os.kill(dispatch_pid, signal.SIGKILL)
    sleep(1)
    logger.info("AIPerf controller exit!")
    sys.exit()

def start_dispatcher(experiment_config):
    global dispatch_pid
    msg_dispatcher_command = "python3 -m nni --exp_params {}".format(
        base64.b64encode(json.dumps(experiment_config).encode()).decode()
    )
    # print(msg_dispatcher_command)
    msg_dispatcher_command = tuple(msg_dispatcher_command.split())

    process = subprocess.Popen(
        msg_dispatcher_command, 
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True
    )
    dispatch_pid = process.pid
    logger.info("dispatcher pid : {}".format(process.pid))
    child_stdin = process.stdin
    child_stderr = process.stderr
    
    child_stdin.write(b'IN000000')
    child_stdin.flush()
    command, data = read_response(child_stderr)

    sleep(2)

    return process, child_stdin, child_stderr

class TrialInfo:
    
    def __init__(
        self, 
        exp_id,
        trial_id,
        trial_seq_id,
        sys_dir,
        output_dir,
        trial_concurrency,
        cmd
    ):
        self.exp_id = exp_id
        self.trial_id = trial_id
        self.trial_seq_id = trial_seq_id
        self.sys_dir = sys_dir
        self.output_dir = output_dir
        self.trial_concurrency = trial_concurrency
        self.cmd = cmd
        self.status = "running"
        return

    def to_submit_data(self):
        data = {
            "cmd":self.cmd,
            "env":{}
        }
        data["env"]["NNI_PLATFORM"] = "local"
        data["env"]["NNI_EXP_ID"] = "{}".format(self.exp_id)
        data["env"]["NNI_SYS_DIR"] = "{}".format(self.sys_dir)
        data["env"]["NNI_TRIAL_JOB_ID"] = "{}".format(self.trial_id)
        data["env"]["NNI_OUTPUT_DIR"] = "{}".format(self.output_dir)
        data["env"]["NNI_TRIAL_SEQ_ID"] = "{}".format(self.trial_seq_id)
        data["env"]["MULTI_PHASE"] = "false"
        data["env"]["TRIAL_CONCURRENCY"] = "{}".format(self.trial_concurrency)
        return data

TRIALS_LIST = []

def launch_experiment(args, experiment_config, mode, config_file_name, experiment_id=None):
    '''follow steps to start rest server and start experiment'''
    headers = {'Content-Type': 'application/json;charset=UTF-8'}
    SUBMIT_URL="{}/api/trial/create".format(args.server)
    HEARTBEAT_URL = "{}/api/trial/heartbeat".format(args.server)
    STOP_URL = "{}/api/trial/stop".format(args.server)
    CLEAR_URL = "{}/api/trial/clear".format(args.server)
    signal.signal(signal.SIGTERM, term_sig_handler)
    signal.signal(signal.SIGINT, term_sig_handler)
    global dispatch_pid
    logger.info("launch_experiment")
    # 0. warm up
    try :
        os.makedirs(os.environ["AIPERF_WORKDIR"]+"/nni/experiments")
    except Exception as e:
        pass
    nni_config = Config(config_file_name)
    experiment_config = nni_config.get_config("experimentConfig")
    print(
        json.dumps(
            experiment_config,
            indent=4
        )
    )

    NNI_EXP_ID = "".join(random.sample(string.ascii_letters + string.digits, 8))
    NNI_EXP_DIR = os.environ["AIPERF_WORKDIR"]+"/nni/experiments/"+NNI_EXP_ID
    NNI_EXP_LOG_DIR = NNI_EXP_DIR + "/log"
    NNI_TRIALS_DIR = NNI_EXP_DIR + "/trials"
    NNI_CKPT_DIR = NNI_EXP_DIR + "/checkpoint"
    os.mkdir(NNI_EXP_DIR)
    os.mkdir(NNI_EXP_LOG_DIR)
    os.mkdir(NNI_TRIALS_DIR)
    os.mkdir(NNI_CKPT_DIR)

    NNI_WORK_DIR = os.environ["AIPERF_WORKDIR"]+"/mountdir/nni/experiments/"+NNI_EXP_ID
    os.makedirs(NNI_WORK_DIR)

    logger.info("NNI_EXP_ID: {}".format(NNI_EXP_ID))

    # 1. START DISPATCHER
    process, child_stdin, child_stderr = start_dispatcher(experiment_config)

    # 2. 开一个trial poll 然后开始跑
    next_trial_seq_id = 0
    trial_concurrency = experiment_config["trialConcurrency"]
    # 创建必要的文件
    with open("{}/trial_concurrency.txt".format(os.environ["AIPERF_WORKDIR"]), "w") as f:
        f.write(str(trial_concurrency))
    gen_cmd = (
        "GE" + 
        "0"*(6-len(str(len(str(trial_concurrency))))) +
        str(len(str(trial_concurrency))) +
        str(trial_concurrency)
    )
    requests.get(STOP_URL, headers=headers)
    requests.get(CLEAR_URL, headers=headers)

    child_stdin.write(gen_cmd.encode())
    child_stdin.flush()
    for i in range(trial_concurrency):
        NNI_TRIAL_SEQ_ID = str(next_trial_seq_id)
        NNI_TRIAL_ID = "".join(random.sample(string.ascii_letters + string.digits, 5))
        NNI_SYS_DIR = NNI_TRIALS_DIR + "/" + NNI_TRIAL_ID
        NNI_OUTPUT_DIR = NNI_TRIALS_DIR + "/" + NNI_TRIAL_ID
        os.mkdir(NNI_SYS_DIR)
        NNI_PARAM_PATH = NNI_SYS_DIR + "/parameter.cfg"
        command, data = read_response(child_stderr)
        f = open(NNI_PARAM_PATH, "w")
        f.write(data)
        f.close()
        t = TrialInfo(
            NNI_EXP_ID,
            NNI_TRIAL_ID,
            NNI_TRIAL_SEQ_ID,
            NNI_SYS_DIR,
            NNI_OUTPUT_DIR,
            trial_concurrency,
            experiment_config["trial"]["command"].replace("\\", "", 999999)
        )
        TRIALS_LIST.append(t)
        next_trial_seq_id += 1
        headers = {'Content-Type': 'application/json;charset=UTF-8'}
        res = requests.post(
            SUBMIT_URL,
            headers = headers,
            json=t.to_submit_data()
        )
        requests.get(HEARTBEAT_URL, headers=headers)

    while True:
        logging.info("Current max trial id: {}".format(next_trial_seq_id-1))

        for t in TRIALS_LIST:
            if t.status == "finish":
                continue
            QUERY_URL="{}/api/trial/query?".format(args.server)+t.trial_id
            
            data={"trial":t.trial_id}
            res = requests.post(
                QUERY_URL,
                headers = headers,
                json=data
            )
            if res.json()["status"]=="finish":
                logger.info("{} finish".format(t.trial_id))
                t.status = "finish"
                # New trial
                NNI_TRIAL_SEQ_ID = str(next_trial_seq_id)
                NNI_TRIAL_ID = "".join(random.sample(string.ascii_letters + string.digits, 5))
                NNI_SYS_DIR = NNI_TRIALS_DIR + "/" + NNI_TRIAL_ID
                NNI_OUTPUT_DIR = NNI_TRIALS_DIR + "/" + NNI_TRIAL_ID
                os.mkdir(NNI_SYS_DIR)
                NNI_PARAM_PATH = NNI_SYS_DIR + "/parameter.cfg"
                gen_cmd = 'GE0000011'
                child_stdin.write(gen_cmd.encode())
                child_stdin.flush()
                command, data = read_response(child_stderr)
                f = open(NNI_PARAM_PATH, "w")
                f.write(data)
                f.close()
                t = TrialInfo(
                    NNI_EXP_ID,
                    NNI_TRIAL_ID,
                    NNI_TRIAL_SEQ_ID,
                    NNI_SYS_DIR,
                    NNI_OUTPUT_DIR,
                    trial_concurrency,
                    experiment_config["trial"]["command"].replace("\\", "", 999999)
                )
                TRIALS_LIST.append(t)
                headers = {'Content-Type': 'application/json;charset=UTF-8'}
                res = requests.post(
                    SUBMIT_URL,
                    headers = headers,
                    json=t.to_submit_data()
                )
                requests.get(HEARTBEAT_URL, headers=headers)

        if next_trial_seq_id >= experiment_config["maxTrialNum"]:
            break
        sleep(30)

    # stop all
    child_stdin.write(b'TE000000')
    child_stdin.flush()
    sleep(3)
    os.kill(process.pid, signal.SIGKILL)
    return

def create_experiment(args):
    '''start a new experiment'''
    config_file_name = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    logger.info("create_experiment")
    config_path = os.path.abspath(args.config)
    logger.info("config path : {}".format(config_path))
    if not os.path.exists(config_path):
        print_error('Please set correct config path!')
        exit(1)
    experiment_config = get_yml_content(config_path)
    validate_all_content(experiment_config, config_path)
    # print(json.dumps(experiment_config, indent=4))

    nni_config = Config(config_file_name)
    nni_config.set_config('experimentConfig', experiment_config)

    try:
        launch_experiment(args, experiment_config, 'new', config_file_name)
    except Exception as exception:
        print_error(exception)
        exit(1) 


def clean_environment(args):
    '''clean environment'''
    headers = {'Content-Type': 'application/json;charset=UTF-8'}
    STOP_URL = "{}/api/trial/stop".format(args.server)
    CLEAR_URL = "{}/api/trial/clear".format(args.server)

    logger.info("clean environment for all nodes by aiperf_ctrl")
    requests.get(STOP_URL, headers=headers)
    requests.get(CLEAR_URL, headers=headers)
    return


def aiperf_info(*args):
    logger.info("aiperf_info")
    if args[0].version:
        try:
            print(pkg_resources.get_distribution('aiperf').version)
        except pkg_resources.ResolutionError:
            print_error('Get version failed, please use `pip3 list | grep aiperf` to check aiperf version!')
    else:
        print('please run "aiperf {positional argument} --help" to see aiperfctl guidance')

def parse_args():
    '''Definite the arguments users need to follow and input'''
    logger.info("aiperf start")
    parser = argparse.ArgumentParser(prog='aiperfctl', description='use aiperfctl command to control aiperf experiments')
    parser.add_argument('--version', '-v', action='store_true')
    parser.set_defaults(func=aiperf_info)

    # create subparsers for args with sub values
    subparsers = parser.add_subparsers()

    # parse start command
    parser_start = subparsers.add_parser('create', help='create a new experiment')
    parser_start.add_argument('--config', '-c', required=True, dest='config', help='the path of yaml config file')
    parser_start.add_argument('--server', '-s', dest='server', help='control server, http://0.0.0.0:9987', default="http://{}:{}".format(os.environ['AIPERF_MASTER_IP'], os.environ['AIPERF_MASTER_PORT']))
    parser_start.add_argument('--debug', '-d', action='store_true', help=' set debug mode')
    parser_start.set_defaults(func=create_experiment)

    parser_clean = subparsers.add_parser('clean', help='use aiperf_ctrl to clean environment for all nodes')
    parser_clean.add_argument('--server', '-s', dest='server', help='control server, http://0.0.0.0:9987', default="http://{}:{}".format(os.environ['AIPERF_MASTER_IP'], os.environ['AIPERF_MASTER_PORT']))
    parser_clean.set_defaults(func=clean_environment)

    args = parser.parse_args()
    args.func(args)
    
if __name__ == '__main__':
    parse_args()
