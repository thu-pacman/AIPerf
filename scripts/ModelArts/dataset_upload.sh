#!/bin/bash
set -x
source obs_env.sh

DATASET=/mnt/zoltan/public/dataset/rawdata/


./obsutil cp -r -f $DATASET/train obs://$AIPERF_OBS_WORKDIR/rawdata
./obsutil cp -r -f $DATASET/val obs://$AIPERF_OBS_WORKDIR/rawdata
