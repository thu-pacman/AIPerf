from django.http import JsonResponse
import json
import os

# 总工作目录
AIPERF_WORKDIR = os.environ['AIPERF_WORKDIR']
# 计算节点工作目录
AIPERF_SLAVE_WORKDIR = os.environ['AIPERF_SLAVE_WORKDIR']
# master节点ip与port
AIPERF_MASTER_IP = os.environ['AIPERF_MASTER_IP']
AIPERF_MASTER_PORT = os.environ['AIPERF_MASTER_PORT']

with open("server_env_init.sh", "r") as f:
    ENV_CMD = [f.read()]

ENV_CMD += [
    "export AIPERF_WORKDIR={}".format(AIPERF_WORKDIR),
    "export AIPERF_SLAVE_WORKDIR={}".format(AIPERF_SLAVE_WORKDIR),
    "export AIPERF_MASTER_IP={}".format(AIPERF_MASTER_IP),
    "export AIPERF_MASTER_PORT={}".format(AIPERF_MASTER_PORT),
]

# ssh 用户用户名
USER = os.environ["USER"]
SSH_USERNAME = USER


SERVER_LIST = []

SERVER_LIST=json.loads(open("servers.json","r").read())

TRIAL_LIST = []

TRIAL_LIST=json.loads(open("trial.json","r").read())

ENV_CMD = "\n".join(ENV_CMD) + "\n"


def create_trial(request):
    global SERVER_LIST, TRIAL_LIST
    if request.method != "POST":
        res = {
            "success":False
        }
        return JsonResponse(res)

    trialConfig = json.loads(request.body.decode("utf-8"))
    trialConfig["status"] = "waiting"
    FOUND = False
    for t in TRIAL_LIST:
        if t["env"]["NNI_TRIAL_JOB_ID"]==trialConfig["env"]["NNI_TRIAL_JOB_ID"]:
            FOUND = True
    if not FOUND:
        TRIAL_LIST.append(trialConfig)
    f = open("servers.json","w")
    f.write(json.dumps(SERVER_LIST, indent=4))
    f.close()
    
    f = open("trial.json","w")
    f.write(json.dumps(TRIAL_LIST, indent=4))
    f.close()
    res = {
        "success":True
    }
    return JsonResponse(res)

def clear(request):
    global SERVER_LIST, TRIAL_LIST
    for s in SERVER_LIST:
        s["status"]="waiting"
        s["tag"]=""
    TRIAL_LIST = []
    f = open("servers.json","w")
    f.write(json.dumps(SERVER_LIST, indent=4))
    f.close()
    
    f = open("trial.json","w")
    f.write(json.dumps(TRIAL_LIST, indent=4))
    f.close()
    
    return JsonResponse({"success":True})

def heartbeat(request):
    global SERVER_LIST, TRIAL_LIST
    found = False
    for s in SERVER_LIST:
        for t in TRIAL_LIST:
            if (s["status"]=="waiting") and (t["status"]=="waiting"):
                print("run trial {} on {}".format(
                    t["env"]["NNI_TRIAL_JOB_ID"],
                    s["ip"]
                ))
                s["status"] = "running"
                s["tag"] = t["env"]["NNI_TRIAL_JOB_ID"]
                t["status"] = "running"
                sshExec(s,t)
                found=True
            if found:
                break
        if found:
            break
            
    f = open("servers.json","w")
    f.write(json.dumps(SERVER_LIST, indent=4))
    f.close()
    
    f = open("trial.json","w")
    f.write(json.dumps(TRIAL_LIST, indent=4))
    f.close()
    return JsonResponse({"success":True})

def query_trial(request):
    global SERVER_LIST, TRIAL_LIST
    if request.method != "POST":
        res = {
            "success":False
        }
        return JsonResponse(res)
    reportData = json.loads(request.body.decode("utf-8"))
    res={
        "status":"finish"
    } 
    for t in TRIAL_LIST:
        if (t["env"]["NNI_TRIAL_JOB_ID"]==reportData["trial"]):
            res["status"]=t["status"]
    # print(res)
    return JsonResponse(res)

def finish_trial(request):
    global SERVER_LIST, TRIAL_LIST
    if request.method != "POST":
        res = {
            "success":False
        }
        return JsonResponse(res)
    reportData = json.loads(request.body.decode("utf-8"))
    
    for server in SERVER_LIST:
        if (server["status"]=="running") and (server["tag"]==reportData["trial"]):
            server["status"]="waiting"
            server["tag"]=""

    for t in TRIAL_LIST:
        if (t["status"]=="running") and (t["env"]["NNI_TRIAL_JOB_ID"]==reportData["trial"]):
            t["status"]="finish"

    res = {
        "success":True
    }
    f = open("servers.json","w")
    f.write(json.dumps(SERVER_LIST, indent=4))
    f.close()
    
    f = open("trial.json","w")
    f.write(json.dumps(TRIAL_LIST, indent=4))
    f.close()
    return JsonResponse(res)

def stop_all(request):
    global SERVER_LIST, TRIAL_LIST
    sshKill()
    res = {
        "success":True
    }
    return JsonResponse(res)

def sshKill():
    global SERVER_LIST, TRIAL_LIST
    
    for server in SERVER_LIST:
        os.system("ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {}@{} cp {}/AIPerf/aiperf_ctrl/kill.sh {}".format(SSH_USERNAME, server["ip"], AIPERF_WORKDIR, AIPERF_SLAVE_WORKDIR))
        os.system("ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {}@{} 'cd {}; bash kill.sh &'".format(SSH_USERNAME, server["ip"], AIPERF_SLAVE_WORKDIR))
    return
    

def sshExec(server, trial):
    global SERVER_LIST, TRIAL_LIST
    bashCmd = (
        "#!/bin/bash -l\n"+
        ENV_CMD
    )

    bashCmd += "cd '{}/AIPerf/examples/trials/network_morphism/imagenet/.'\n".format(AIPERF_WORKDIR)
    for k in trial["env"]:
        v = trial["env"][k]
        if k=="CUDA_VISIBLE_DEVICES":
            v = server["CUDA_VISIBLE_DEVICES"]
        bashCmd += "export {}='{}'\n".format(
            k,
            v
        )
    bashCmd += "{}\n".format(trial["cmd"])
    
    for k in trial["env"]:
        v = trial["env"][k]
        bashCmd += "unset {}\n".format(
            k
        )
    
    bashCmd += (
        "/usr/bin/curl --location --request POST 'http://{}:{}/api/trial/finish' ".format(AIPERF_MASTER_IP, AIPERF_MASTER_PORT) +
        "--header 'Content-Type: application/json' --data '{\"trial\":\""
        +trial["env"]["NNI_TRIAL_JOB_ID"] +"\"}'"
    )
    f=open("{}/AIPerf/aiperf_ctrl/tmp.sh".format(AIPERF_WORKDIR),"w")
    f.write(bashCmd)
    f.close()
    # TODO: set env before this cmd or remove it ?
    #os.system("ssh -o ConnectTimeout=10 {}@{} 'cd {}/AIPerf/examples/trials/network_morphism/imagenet/; python3 resource_monitor.py --id {} &'".format(SSH_USERNAME, server["ip"], AIPERF_WORKDIR, trial["env"]["NNI_EXP_ID"]) )
    os.system("ssh -o ConnectTimeout=10 {}@{} 'mkdir -p {}/aiperflog/{} ; cp {}/AIPerf/aiperf_ctrl/tmp.sh {}/aiperflog/{}'".format(
        SSH_USERNAME, 
        server["ip"], 
        AIPERF_SLAVE_WORKDIR,
        trial["env"]["NNI_TRIAL_JOB_ID"],
        AIPERF_WORKDIR,
        AIPERF_SLAVE_WORKDIR,
        trial["env"]["NNI_TRIAL_JOB_ID"]
    ))
    os.system("ssh -o ConnectTimeout=10 {}@{} 'cd {}/aiperflog/{}; source tmp.sh >stdout.log 2>stderr.log &'".format(SSH_USERNAME,server["ip"], AIPERF_SLAVE_WORKDIR, trial["env"]["NNI_TRIAL_JOB_ID"]))
    return
