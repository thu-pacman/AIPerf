import argparse
import time
import logging
import moxing as mox
import os
import json 
from mindspore import context as mds_context
N = 8



INSTANCE_SAMPLE={
    "devices": [
        {
            "device_id": "3",
            "device_ip": "127.0.0.1"
        }
    ],
    "rank_id": "0",
    "server_id": "127.0.0.1"
}
def get_args():
    """ get args from command line
    """
    parser = argparse.ArgumentParser("modelarts")
    parser.add_argument("--data_url", type=str)
    parser.add_argument("--data", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    if(int(os.environ["RANK_ID"])%8==0):
        timeStr = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        args = get_args()
        os.system("pip3 install colorama ruamel.yaml torch schema --no-index -f {}".format(args.data_url))
        os.system("pip3 install -e /home/ma-user/modelarts/user-job-dir/code/AIPerf/src/sdk/pynni/  --no-index -f {}".format(args.data_url))
        os.system("pip3 install -e /home/ma-user/modelarts/user-job-dir/code/AIPerf/src/aiperf_manager/  --no-index -f {}".format(args.data_url))
        for k in os.environ:
            print(k, " -----> ", os.environ[k])
        
        RANK_TABLE_PATH = os.environ["RANK_TABLE_FILE"]
        f = open(RANK_TABLE_PATH, "r")
        RANK_TABLE_RAW = f.read()
        print("RANK_TABLE_RAW", RANK_TABLE_RAW)
        RANK_TABLE_DATA = json.loads( RANK_TABLE_RAW )
        N = len(RANK_TABLE_DATA["server_list"][0]["device"])
        os.system("ps -ef | grep python")
        HCCL_SAMPLE={
            "board_id": "0x0020",
            "chip_info": "910",
            "deploy_mode": "lab",
            "group_count": "1",
            "group_list": [
                {
                    "device_num": str(N),
                    "server_num": "1",
                    "group_name": "",
                    "instance_count": str(N),
                    "instance_list": []
                }
            ],
            "para_plane_nic_location": "device",
            "para_plane_nic_name": ["eth0"],
            "para_plane_nic_num": "1",
            "status": "completed"
        }
        SERVER_COUNT = int(RANK_TABLE_DATA["server_count"])
        os.system("echo {} > ~/trial_concurrency.txt".format(SERVER_COUNT))
        os.system("mkdir -p /dev/shm/nni")
        os.system("mkdir -p /dev/shm/mountdir")
        os.system("ln -s /dev/shm/nni /home/ma-user/nni")
        os.system("ln -s /dev/shm/mountdir /home/ma-user/mountdir")
        os.system("mkdir -p /home/ma-user/nni/experiments")
        os.system("mkdir -p /home/ma-user/mountdir/experiments")
        os.system("mkdir -p /home/ma-user/aiperflog")
        SERVER_IDx = 0
        START_RANK_ID = int(os.environ["RANK_ID"])
        for i in range(SERVER_COUNT):
            for j in range(N):
                if int(RANK_TABLE_DATA["server_list"][i]["device"][j]["rank_id"]) == START_RANK_ID:
                    SERVER_IDx = i
                    break

        for i in range(N):
            dev = RANK_TABLE_DATA["server_list"][SERVER_IDx]["device"][i]
            HCCL_SAMPLE["group_list"][0]["instance_list"].append(
                {
                    "devices": [
                        {
                            "device_id": dev["device_id"],
                            "device_ip": dev["device_ip"]
                        }
                    ],
                    "rank_id": str(int(dev["rank_id"])-START_RANK_ID),
                    "server_id": RANK_TABLE_DATA["server_list"][SERVER_IDx]["server_id"]
                }
            )
        print("AIPerf HCCL:", HCCL_SAMPLE)
        os.system("ifconfig")
        f = open("/home/ma-user/modelarts/user-job-dir/code/AIPerf/hccl.json","w")
        f.write(json.dumps(HCCL_SAMPLE))
        f.close()
    else:
        mds_context.reset_auto_parallel_context()
        exit(0)
        args = get_args()
        RANK_TABLE_PATH = os.environ["RANK_TABLE_FILE"]
        f = open(RANK_TABLE_PATH, "r")
        RANK_TABLE_RAW = f.read()
        print("RANK_TABLE_RAW", RANK_TABLE_RAW)
        RANK_TABLE_DATA = json.loads( RANK_TABLE_RAW )
        N = len(RANK_TABLE_DATA["server_list"][0]["device"])
        time.sleep(20)
    os.system("npu-smi info")
    mds_context.reset_auto_parallel_context()
    os.system("npu-smi info")
    print("cd /home/ma-user/modelarts/user-job-dir/code/AIPerf/examples/trials/network_morphism/imagenet/ && unset RANK_TABLE_FILE && " + 
              " MINDSPORE_HCCL_CONFIG_PATH=/home/ma-user/modelarts/user-job-dir/code/AIPerf/hccl.json " +
              #" ASCEND_SLOG_PRINT_TO_STDOUT=1 " + 
              "SIZE_LIMIT=28  NPU_NUM={} python3 multithread_demo.py  --batch_size 256  --epoch 15  --train_data_dir {}train/  --val_data_dir {}val/".format(N, args.data,args.data))
    #os.system("cd /home/ma-user/modelarts/user-job-dir/code/AIPerf/examples/trials/network_morphism/imagenet/ && unset RANK_TABLE_FILE && " + 
    #          " MINDSPORE_HCCL_CONFIG_PATH=/home/ma-user/modelarts/user-job-dir/code/AIPerf/hccl.json " +
    #          #" ASCEND_SLOG_PRINT_TO_STDOUT=1 " + 
    #          "SIZE_LIMIT=28 NPU_NUM={} python3 multithread_demo.py  --batch_size 256  --epoch 15  --train_data_dir {}train/  --val_data_dir {}val/".format(N, args.data,args.data))
    last_cmd_id = 1
    # MA_CURRENT_IP  ----->  172.16.0.56
    MA_CURRENT_IP = os.environ["MA_CURRENT_IP"]
    fip = mox.file.File("obs://debug/aiperf/runtime/{}/{}".format(os.getenv("TASKID","NONE"),os.environ["MA_CURRENT_IP"]), "w")
    fip.write(MA_CURRENT_IP)
    fip.close()
    os.environ["LD_PRELOAD"]="/home/ma-user/miniconda3/envs/MindSpore-1.3.0-aarch64/lib/libgomp.so.1"
    while(True):
        timeStr = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        print("{}:[{}]Idling...".format(timeStr, MA_CURRENT_IP))
        datas = []
        
        try:
            f = mox.file.File("obs://debug/aiperf/runtime/cmd.json", "r")
            datas = json.loads(f.read())
            f.close()
        except:
            pass
        for data in datas:
            if(data["target"]!=os.environ["MA_CURRENT_IP"]):
                continue
            if(data["id"] > last_cmd_id):
                last_cmd_id = data["id"]
                
                for idx in range(len(data["cmds"])):
                    cmd = data["cmds"][idx]
                    print("receive {} : {}".format(data["id"], cmd["cmd"]))
                    try:
                        if(cmd["type"]=="bash"):
                            os.system(cmd["cmd"])
                        else :
                            eval(cmd["cmd"])
                    except:
                        print("error!:{}".format(cmd["cmd"]))
                    time.sleep(1)
            else :
                print("CURRENT[{}] receive OLD {} ".format(last_cmd_id, data["id"]))
        
        time.sleep(20)