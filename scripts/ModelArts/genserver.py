import os
import json

ips_raw = os.listdir("./ips/NONE")
print("Got ip: {}".format(len(ips_raw)))
ips_raw.sort()
cmds = []
servers = []
for ip in ips_raw:
    cmds.append({
        "id": 120,
        "cmds": [
            {
                "type": "eval",
                "cmd": "mox.file.copy_parallel('obs://aiperf/aiperf/code/AIPerf','/home/ma-user/modelarts/user-job-dir/code/AIPerf')"
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