import argparse
import time
import logging
import moxing as mox
import os
import json 

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
    if(int(os.environ["RANK_ID"])==0):
        timeStr = time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        args = get_args()
        os.system("pip3 install torch --no-index -f {}".format(args.data_url))
        os.system("pip3 install -e /home/ma-user/modelarts/user-job-dir/code/AIPerf/src/sdk/pynni/  --no-index -f {}".format(args.data_url))
        for k in os.environ:
            print(k, " -----> ", os.environ[k])
        
        RANK_TABLE_PATH = os.environ["RANK_TABLE_FILE"]
        f = open(RANK_TABLE_PATH, "r")
        RANK_TABLE_RAW = f.read()
        print("RANK_TABLE_RAW", RANK_TABLE_RAW)
        RANK_TABLE_DATA = json.loads( RANK_TABLE_RAW )
        N = len(RANK_TABLE_DATA["server_list"][0]["device"])
        os.system("ps -ef | grep python")
    else:
        args = get_args()
        RANK_TABLE_PATH = os.environ["RANK_TABLE_FILE"]
        f = open(RANK_TABLE_PATH, "r")
        RANK_TABLE_RAW = f.read()
        print("RANK_TABLE_RAW", RANK_TABLE_RAW)
        RANK_TABLE_DATA = json.loads( RANK_TABLE_RAW )
        N = len(RANK_TABLE_DATA["server_list"][0]["device"])
        time.sleep(60)

    # os.unsetenv("RANK_TABLE_FILE")
    os.system("cd /home/ma-user/modelarts/user-job-dir/code/AIPerf/examples/trials/network_morphism/imagenet/ &&" + 
              # " MINDSPORE_HCCL_CONFIG_PATH=/home/ma-user/modelarts/user-job-dir/code/AIPerf/hccl.json " +
              # " ASCEND_SLOG_PRINT_TO_STDOUT=1 " + 
              "NPU_NUM={} python3 demo.py  --batch_size 256  --epoch 150  --train_data_dir {}train/  --val_data_dir {}val/".format(N, args.data,args.data))





'''
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
    for i in range(N):
        dev = RANK_TABLE_DATA["server_list"][0]["device"][i]
        HCCL_SAMPLE["group_list"][0]["instance_list"].append(
            {
                "devices": [
                    {
                        "device_id": dev["device_id"],
                        "device_ip": dev["device_ip"]
                    }
                ],
                "rank_id": dev["rank_id"],
                "server_id": RANK_TABLE_DATA["server_list"][0]["server_id"]
            }
        )
    print("AIPerf HCCL:", HCCL_SAMPLE)
    os.system("ifconfig")
    f = open("/home/ma-user/modelarts/user-job-dir/code/AIPerf/hccl.json","w")
    f.write(json.dumps(HCCL_SAMPLE))
    f.close()
'''