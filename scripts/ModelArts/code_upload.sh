#!/bin/bash
set -x
source obs_env.sh


rm ../../examples/trials/network_morphism/imagenet/cmd.json
rm ../../aiperf_ctrl/servers.json
./obsutil cp -r -f ../../../AIPerf obs://$AIPERF_OBS_WORKDIR/code/
