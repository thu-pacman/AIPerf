import os
import json

AIPERF_OBS_WORKDIR = os.environ['AIPERF_OBS_WORKDIR']
AIPERF_MASTER_IP = os.environ['AIPERF_MASTER_IP']
AIPERF_MASTER_PORT = os.environ['AIPERF_MASTER_PORT']

cmds = []
cmds.append({
    "id": 3,
    "cmds": [
        {
            "type": "bash",
            "cmd": "cd /home/ma-user/modelarts/user-job-dir/code/AIPerf/examples/trials/network_morphism/imagenet/ && no_proxy={} NO_PROXY={} HTTP_PROXY=''  HTTPS_PROXY='' LD_PRELOAD=/home/ma-user/anaconda3/envs/MindSpore/lib/libgomp.so TF_CPP_MIN_LOG_LEVEL=3 OBSHOME=obs://{}/runlog aiperf create -c config.yml --server http://{}:{} > /tmp/aiperf.log 2>&1 &".format(AIPERF_MASTER_IP, AIPERF_MASTER_IP, AIPERF_OBS_WORKDIR, AIPERF_MASTER_IP, AIPERF_MASTER_PORT)
        },
    ],
    "target": AIPERF_MASTER_IP
})
f = open("cmd.json","w")
f.write(json.dumps(cmds, indent=4))
f.close()
