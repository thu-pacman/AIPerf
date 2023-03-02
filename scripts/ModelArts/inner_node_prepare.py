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
        "id": 1,
        "cmds": [
            {
                "type": "eval",
                "cmd": "exec(\"os.environ['AIPERF_OBS_WORKDIR'] = '{}'\")".format(AIPERF_OBS_WORKDIR)
            },
            {
                "type": "eval",
                "cmd": "exec(\"os.environ['AIPERF_MASTER_IP'] = '{}'\")".format(AIPERF_MASTER_IP)
            },
            {
                "type": "eval",
                "cmd": "exec(\"os.environ['AIPERF_MASTER_PORT'] = '{}'\")".format(AIPERF_MASTER_PORT)
            },
            {
                "type": "bash",
                "cmd": "env | grep AIPERF"
            },
            {
                "type": "eval",
                "cmd": "mox.file.copy_parallel('obs://{}/code/AIPerf','/home/ma-user/modelarts/user-job-dir/code/AIPerf')".format(AIPERF_OBS_WORKDIR)
            },
            {
                "type": "bash",
                "cmd": "python3 -m pip install -r /home/ma-user/modelarts/user-job-dir/code/AIPerf/requirements_master.txt"
            },
            {
                "type": "bash",
                "cmd": "python3 -m pip install -r /home/ma-user/modelarts/user-job-dir/code/AIPerf/requirements_slave.txt"
            },
            {
                "type": "bash",
                "cmd": "for i in {1..3}; do python3 -m pip install -e /home/ma-user/modelarts/user-job-dir/code/AIPerf/src/sdk/pynni; done"
            },
            {
                "type": "bash",
                "cmd": "for i in {1..3}; do python3 -m pip install -e /home/ma-user/modelarts/user-job-dir/code/AIPerf/src/aiperf_manager; done"
            },
            {
                "type": "bash",
                "cmd": "python3 -m pip install torch"
            }
        ],
        "target": ip
    })

for ip in ips_raw:
    servers.append({
        "ip": ip,
        "tag": "",
        "status": "waiting"
    })

f = open("cmd.json","w")
f.write(json.dumps(cmds, indent=4))
f.close()

f = open("servers.json","w")
f.write(json.dumps(servers, indent=4))
f.close()
