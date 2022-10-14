#!/bin/bash
set -x
source obs_env.sh
./obsutil mkdir obs://$AIPERF_OBS_WORKDIR/runlog/nni/experiments
./obsutil mkdir obs://$AIPERF_OBS_WORKDIR/runlog/mountdir/nni/experiments
./obsutil mkdir obs://$AIPERF_OBS_WORKDIR/runlog/mountdir/device_info
./obsutil mkdir obs://$AIPERF_OBS_WORKDIR/runtime/
./obsutil mkdir obs://$AIPERF_OBS_WORKDIR/task_log
