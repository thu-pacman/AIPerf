#!/bin/bash
set -x
source obs_env.sh

rm -r ./ips
./obsutil cp -f -r obs://${AIPERF_OBS_WORKDIR}/runtime/NONE/ ./ips/
