from django.http import JsonResponse
import json
import os
import moxing as mox

SERVER_LIST = []

SERVER_LIST=json.loads(open("servers.json","r").read())

TRIAL_LIST = []

TRIAL_LIST=json.loads(open("trial.json","r").read())

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
    if True:
        cmds = [
            {
                "type":"bash",
                "cmd":"ps -aux | grep imagenet_train | grep -v grep | grep -v manage | awk '{print $2}' | xargs kill"
            },
            {
                "type":"bash",
                "cmd":"ps -aux | grep tmp.sh | grep -v grep | grep -v manage | awk '{print $2}' | xargs kill"
            }, #resource_monitor
            {
                "type":"bash",
                "cmd":"ps -aux | grep resource_monitor | grep -v grep | grep -v manage | awk '{print $2}' | xargs kill"
            }
        ]
        f = mox.file.File("obs://aiperf/aiperf/runtime/cmd.json", "r")
        datas = json.loads(f.read())
        f.close()
        for idx in range(len(datas)):
            datas[idx]["id"] = datas[idx]["id"] + 1
            datas[idx]["cmds"] = cmds
        f = mox.file.File("obs://aiperf/aiperf/runtime/cmd.json", "w")
        f.write(json.dumps(datas))
        f.close()
    return
    

def sshExec(server, trial):
    print("sshExec")
    global SERVER_LIST, TRIAL_LIST
    HOST_IP = os.environ["MA_CURRENT_IP"]
    bashCmd = (
        "#!/usr/bin/bash -l\n " +
        "export no_proxy={}\n".format(HOST_IP) + 
        "export NO_PROXY={}\n".format(HOST_IP) +
        "export MINDSPORE_HCCL_CONFIG_PATH=/home/ma-user/modelarts/user-job-dir/code/AIPerf/hccl.json\n"+
        "export NPU_NUM=8\n"+
        "export LD_PRELOAD=/home/ma-user/miniconda3/envs/MindSpore-1.3.0-aarch64/lib/libgomp.so.1\n"+
        "unset RANK_TABLE\nunset RANK_TABLE_FILE\n"
    )

    bashCmd += "cd '/home/ma-user/modelarts/user-job-dir/code/AIPerf/examples/trials/network_morphism/imagenet/.'\n"
    for k in trial["env"]:
        v = trial["env"][k]
        bashCmd += "export {}='{}'\n".format(
            k,
            v
        )
    bashCmd += (
        "no_proxy={} NO_PROXY={} HTTP_PROXY='' HTTPS_PROXY='' ".format(HOST_IP,HOST_IP) +
        "LD_PRELOAD=/home/ma-user/miniconda3/envs/MindSpore-1.3.0-aarch64/lib/libgomp.so.1 "+
        "{}\n".format(trial["cmd"])
    )
    
    for k in trial["env"]:
        v = trial["env"][k]
        bashCmd += "unset {}\n".format(
            k
        )
    
    bashCmd += (
        "no_proxy={} NO_PROXY={} HTTP_PROXY='' HTTPS_PROXY='' /usr/bin/curl --noproxy {} --location --request POST 'http://{}:9987/api/trial/finish' ".format(HOST_IP, HOST_IP, HOST_IP, HOST_IP) +
        "--header 'Content-Type: application/json' --data '{\"trial\":\""
        +trial["env"]["NNI_TRIAL_JOB_ID"] +"\"}'"
    )
    
    mox_file_name = "obs://aiperf/aiperf/runtime/tmp_{}.sh".format(trial["env"]["NNI_TRIAL_JOB_ID"])
    f = mox.file.File(mox_file_name, "w")
    f.write(bashCmd)
    f.close()
    cmds = [
        {
            "type":"bash",
            "cmd":"cd /home/ma-user/modelarts/user-job-dir/code/AIPerf/examples/trials/network_morphism/imagenet/; python3 resource_monitor.py --id {} &".format(trial["env"]["NNI_EXP_ID"])
        },
        {
            "type":"bash",
            "cmd":"mkdir /home/ma-user/aiperflog/{} ".format(trial["env"]["NNI_TRIAL_JOB_ID"])
        },
        {
            "type":"eval",
            "cmd":"mox.file.copy(\"{}\", \"/home/ma-user/aiperflog/{}/tmp.sh\")".format(mox_file_name, trial["env"]["NNI_TRIAL_JOB_ID"])
        },
        {
            "type":"bash",
            "cmd":"source /home/ma-user/aiperflog/{}/tmp.sh >/home/ma-user/aiperflog/{}/stdout.log 2>/home/ma-user/aiperflog/{}/stderr.log &".format(trial["env"]["NNI_TRIAL_JOB_ID"],trial["env"]["NNI_TRIAL_JOB_ID"],trial["env"]["NNI_TRIAL_JOB_ID"])
        }
    ]
    f = mox.file.File("obs://aiperf/aiperf/runtime/cmd.json", "r")
    datas = json.loads(f.read())
    f.close()
    for idx in range(len(datas)):
        if(datas[idx]["target"]!=server["ip"]):
            continue
        datas[idx]["id"] = datas[idx]["id"] + 1
        datas[idx]["cmds"] = cmds
    f = mox.file.File("obs://aiperf/aiperf/runtime/cmd.json", "w")
    f.write(json.dumps(datas))
    f.close()
    return