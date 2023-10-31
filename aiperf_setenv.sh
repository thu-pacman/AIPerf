#!/bin/bash

# Set env for AIPerf

export AIPERF_WORKDIR=/share/aiperf_workspace
export AIPERF_SLAVE_WORKDIR=/root/
export AIPERF_MASTER_IP=10.0.1.100
export AIPERF_MASTER_PORT=9987

echo "AIPERF_WORKDIR:       ${AIPERF_WORKDIR}"
echo "AIPERF_SLAVE_WORKDIR: ${AIPERF_SLAVE_WORKDIR}"
echo "AIPERF_MASTER_IP:     ${AIPERF_MASTER_IP}"
echo "AIPERF_MASTER_PORT:   ${AIPERF_MASTER_PORT}"