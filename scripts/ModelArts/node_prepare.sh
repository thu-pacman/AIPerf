#!/bin/bash
set -x
source obs_env.sh
source setenv.sh

python3 inner_node_prepare.py

mv servers.json ../../aiperf_ctrl/
mv cmd.json ../../examples/trials/network_morphism/imagenet/
# cmd.json  上传
./cmd_upload.sh
# servers.json 上传
./server_upload.sh
