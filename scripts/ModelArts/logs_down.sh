#!/bin/bash
set -x
source obs_env.sh

mkdir -p cloud_log
./obsutil cp -r -f  obs://${AIPERF_OBS_WORKDIR}/runlog/nni cloud_log/
./obsutil cp -r -f  obs://${AIPERF_OBS_WORKDIR}/runlog/mountdir cloud_log/


