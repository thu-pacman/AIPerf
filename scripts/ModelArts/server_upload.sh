#!/bin/bash
set -x
source obs_env.sh
./obsutil cp -f ../../aiperf_ctrl/servers.json obs://${AIPERF_OBS_WORKDIR}/code/AIPerf/aiperf_ctrl/servers.json

