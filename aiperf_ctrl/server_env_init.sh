# 计算节点环境加载 根据计算节点实际环境编写
. /share/aiperf_workspace/spack/share/spack/setup-env.sh
spack config add modules:prefix_inspections:lib64:[LD_LIBRARY_PATH]
spack config add modules:prefix_inspections:lib:[LD_LIBRARY_PATH]

spack load python
spack load cuda@11.2
spack load cudnn@8.1
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$(spack find --paths cuda@11.2 | awk 'NR>1 {print $2}')
