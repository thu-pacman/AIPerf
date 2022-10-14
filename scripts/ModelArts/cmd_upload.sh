#!/bin/bash
set -x
source obs_env.sh

./obsutil cp -f ../../examples/trials/network_morphism/imagenet/cmd.json obs://$AIPERF_OBS_WORKDIR/runtime/cmd.json
