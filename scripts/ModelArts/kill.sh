#!/bin/bash
set -x
source obs_env.sh
source setenv.sh

python3 inner_kill.py

mv cmd.json ../../examples/trials/network_morphism/imagenet/
# cmd.json  上传
./cmd_upload.sh
