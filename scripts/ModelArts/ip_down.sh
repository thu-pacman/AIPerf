#!/bin/bash
set -x
source obs_env.sh

rm -r ./ips
./obsutil cp -f -r obs://${AIPERF_OBS_WORKDIR}/runtime/NONE/ ./ips/

# display
tree ./ips
ip_num=$(ls -lR ips/ | grep "^-" | wc -l)
echo "IP num = $ip_num"
