#!/bin/bash
set -x
source obs_env.sh

DATASET=/mnt/zoltan/public/dataset/rawdata-mini/


./obsutil cp -r -f $DATASET/train obs://$AIPERF_OBS_WORKDIR/rawdata-mini
./obsutil cp -r -f $DATASET/val obs://$AIPERF_OBS_WORKDIR/rawdata-mini