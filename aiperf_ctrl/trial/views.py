from django.http import JsonResponse
import json
import os

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
    for server in SERVER_LIST:
        #os.system("sshpass -p 123123 ssh -p 222 -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@{} cp /mnt/zoltan/public/dataset/imagenet/aiperf_ctrl/kill.sh /root".format(server["ip"]))
        #os.system("sshpass -p 123123 ssh -p 222 -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@{} 'cd /root; bash kill.sh &'".format(server["ip"]))
        os.system("ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 wxp@{} cp /mnt/zoltan/public/dataset/imagenet/AIPerf/aiperf_ctrl/kill.sh /home/wxp".format(server["ip"]))
        os.system("ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 wxp@{} 'cd /home/wxp; bash kill.sh &'".format(server["ip"]))
    return
    

def sshExec(server, trial):
    global SERVER_LIST, TRIAL_LIST
    bashCmd = (
        "#!/bin/bash -l\nsource /usr/local/Modules/init/bash\n"+ 
        "module load cuda-10.0/cuda cuda-10.0/cudnn-7.4.2\n"+
        "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/wxp\n"
    )

    bashCmd += "cd '/mnt/zoltan/public/dataset/imagenet/AIPerf/examples/trials/network_morphism/imagenet/.'\n"
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
        "/usr/bin/curl --location --request POST 'http://172.23.33.30:9987/api/trial/finish' " +
        "--header 'Content-Type: application/json' --data '{\"trial\":\""
        +trial["env"]["NNI_TRIAL_JOB_ID"] +"\"}'"
    )
    f=open("tmp.sh","w")
    f.write(bashCmd)
    f.close()
    #os.system("sshpass -p 123123 ssh -p 222 -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@{} 'cd /imagenet/AIPerf/examples/trials/network_morphism/imagenet/; python3 resource_monitor.py --id {} &'".format(server["ip"], trial["env"]["NNI_EXP_ID"]) )
    #os.system("sshpass -p 123123 ssh -p 222 -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@{} cp /imagenet/aiperf_ctrl/tmp.sh /root".format(server["ip"]))
    #os.system("sshpass -p 123123 ssh -p 222 -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@{} 'cd /root; bash tmp.sh >stdout.log 2>stderr.log &'".format(server["ip"]))
    os.system("ssh -o ConnectTimeout=10 wxp@{} 'cd /mnt/zoltan/public/dataset/imagenet/AIPerf/examples/trials/network_morphism/imagenet/; python3 resource_monitor.py --id {} &'".format(server["ip"], trial["env"]["NNI_EXP_ID"]) )
    os.system("ssh -o ConnectTimeout=10 wxp@{} 'mkdir /home/wxp/aiperflog/{} ; cp /mnt/zoltan/public/dataset/imagenet/AIPerf/aiperf_ctrl/tmp.sh /home/wxp/aiperflog/{}'".format(
        server["ip"], 
        trial["env"]["NNI_TRIAL_JOB_ID"],
        trial["env"]["NNI_TRIAL_JOB_ID"]
    ))
    os.system("ssh -o ConnectTimeout=10 wxp@{} 'cd /home/wxp/aiperflog/{}; source tmp.sh >stdout.log 2>stderr.log &'".format(server["ip"], trial["env"]["NNI_TRIAL_JOB_ID"]))
    return