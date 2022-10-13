#!/GPUFS/thu_wgchen_2/.conda/envs/aiperf_py36/bin/python3
import os
import sys
import requests
import time

TARGET_OS_ENV = [
    "NNI_PLATFORM",
    "NNI_EXP_ID",
    "NNI_SYS_DIR",
    "NNI_TRIAL_JOB_ID",
    "NNI_OUTPUT_DIR",
    "NNI_TRIAL_SEQ_ID",
    "MULTI_PHASE",
    "TRIAL_CONCURRENCY",
    "CUDA_VISIBLE_DEVICES",
]

AIPERF_MASTER_IP = os.environ['AIPERF_MASTER_IP']
AIPERF_MASTER_PORT = os.environ['AIPERF_MASTER_PORT']
URL="http://{}:{}/api/trial/create".format(AIPERF_MASTER_IP, AIPERF_MASTER_PORT)

if __name__ == "__main__":
    print("reveive submitter!!!")

    data = {
        "cmd":"",
        "env":{}
    }
    
    NNI_PLATFORM = os.environ.get('NNI_PLATFORM')
    for k in os.environ:
        if k in TARGET_OS_ENV:
            print(k,":",os.environ.get(k))
            data["env"][k] = os.environ.get(k)
    # print("NNI_PLATFORM:{}".format(NNI_PLATFORM))
    print()
    #print(sys.argv)
    cmd = " ".join(sys.argv[1:])
    print("CMD:")
    print(cmd)
    data["cmd"] = cmd

    headers = {'Content-Type': 'application/json;charset=UTF-8'}
    res = requests.post(
        URL,
        headers = headers,
        json=data
    )
    print(res.json()["success"])

    while(True):
        URL="http://{}:{}/api/trial/query?{}".format(AIPERF_MASTER_IP, AIPERF_MASTER_PORT, os.environ.get("NNI_TRIAL_JOB_ID"))
        headers = {'Content-Type': 'application/json;charset=UTF-8'}
        data={"trial":os.environ.get("NNI_TRIAL_JOB_ID")}
        res = requests.post(
            URL,
            headers = headers,
            json=data
        )
        if res.json()["status"]=="finish":
            print("DONE!")
            break
        else :
            print(res.json())
            print("waiting...")
            time.sleep(60)
