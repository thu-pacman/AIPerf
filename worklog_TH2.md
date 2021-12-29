module load  anaconda3/5.3.1  CUDA/10.1.2 nccl/2.6.4-1-cuda-10.1 cudnn/7.6.4-CUDA10.1

BIN_FOLDER=/GPUFS/thu_wgchen_2/.local/bin NNI_PKG_FOLDER=/GPUFS/thu_wgchen_2/.local/nni make dev-install

source activate

conda activate aiperf_py36 

conda deactivate


python3 -m pip install  --no-index --find-links=~/download/ -r requirements.txt

ssh -o StrictHostKeyChecking=no -i /GPUFS/thu_wgchen_2/.ssh/thu_wgchen_2.id  -o ConnectTimeout=10 thu_wgchen_2@89.72.32.12 'module load  anaconda3/5.3.1  CUDA/10.1.2 nccl/2.6.4-1-cuda-10.1 cudnn/7.6.4-CUDA10.1 && source activate && conda activate aiperf_py36 && cd /GPUFS/thu_wgchen_2/aiperf/AIPerf-wxp/examples/trials/network_morphism/imagenet/; python3 resource_monitor.py --id abcde &'

from .utils import init_dispatcher_logger

import logging

init_dispatcher_logger()

_logger = logging.getLogger(__name__)

单卡的实验
/GPUFS/thu_wgchen_2/mountdir/nni/experiments/KlylJ3bE/results
