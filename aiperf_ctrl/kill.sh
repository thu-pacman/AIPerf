#!/bin/bash
set -x
ps -aux | grep ${USER} | grep train | grep -v grep | awk '{print $2}' | xargs kill
ps -aux | grep ${USER} | grep tmp.sh | grep -v grep | awk '{print $2}' | xargs kill
