#!/bin/bash
set -x
source obs_env.sh

./obsutil config -i=$ak -k=$sk -e=$endpoint
./obsutil ls -s
