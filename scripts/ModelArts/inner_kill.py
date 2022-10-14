import os
import json

AIPERF_OBS_WORKDIR = os.environ['AIPERF_OBS_WORKDIR']
AIPERF_MASTER_IP = os.environ['AIPERF_MASTER_IP']
AIPERF_MASTER_PORT = os.environ['AIPERF_MASTER_PORT']
ips_raw = os.listdir("./ips/NONE")
print("Got ip: {}".format(len(ips_raw)))
ips_raw.sort()
cmds = []
servers = []
for ip in ips_raw:
    cmds.append({
        "id": 211,
        "cmds": [
            {
                "type":"bash",
                "cmd":"ps -aux | grep imagenet_train | grep -v grep | awk '{print $2}' | xargs kill"
            },
            {
                "type":"bash",
                "cmd":"ps -aux | grep tmp.sh | grep -v grep | awk '{print $2}' | xargs kill"
            },
            {
                "type":"bash",
                "cmd":"ps -aux | grep resource_monitor | grep -v grep | awk '{print $2}' | xargs kill"
            },
            {
                "type":"bash",
                "cmd":"ps -aux | grep manage | grep -v grep | awk '{print $2}' | xargs kill"
            },
            {
                "type":"bash",
                "cmd":"ps -aux | grep aiperf | grep -v grep | awk '{print $2}' | xargs kill"
            },
            {
                "type":"bash",
                "cmd":"ps -aux | grep nni | grep -v grep | awk '{print $2}' | xargs kill"
            },
            {
                "type":"bash",
                "cmd":"ps -aux | grep python"
            }
        ],
        "target": ip
    })

f = open("cmd.json","w")
f.write(json.dumps(cmds, indent=4))
f.close()
