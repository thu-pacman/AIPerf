import os
import json

AIPERF_OBS_WORKDIR = os.environ['AIPERF_OBS_WORKDIR']
AIPERF_MASTER_IP = os.environ['AIPERF_MASTER_IP']
AIPERF_MASTER_PORT = os.environ['AIPERF_MASTER_PORT']

cmds = []
cmds.append({
    "id": 2,
    "cmds": [
        {
            "type": "eval",
            "cmd": "mox.file.copy('obs://{}/code/AIPerf/aiperf_ctrl/servers.json','/home/ma-user/modelarts/user-job-dir/code/AIPerf/aiperf_ctrl/servers.json')".format(AIPERF_OBS_WORKDIR)
        },
        {
            "type": "bash",
            "cmd": "cat /home/ma-user/modelarts/user-job-dir/code/AIPerf/aiperf_ctrl/servers.json"
        },
        {
            "type": "bash",
            "cmd": "cd /home/ma-user/modelarts/user-job-dir/code/AIPerf/aiperf_ctrl/ && python3 manage.py runserver {}:{} > /tmp/aiperf_ctrl.log 2>&1 &".format(AIPERF_MASTER_IP, AIPERF_MASTER_PORT)
        },
        {
            "type": "bash",
            "cmd": "cat /tmp/aiperf_ctrl.log"
        }
    ],
    "target": AIPERF_MASTER_IP
})
f = open("cmd.json","w")
f.write(json.dumps(cmds, indent=4))
f.close()
