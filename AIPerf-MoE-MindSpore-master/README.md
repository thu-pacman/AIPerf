AIPerf-MoE on MindSpore
===

AIPerf-MoE is a benchmark of AIPerf for systems to train large AI models.
It aims at measuring both computation and communication power of a system.

## Benchmarking guide

All command lines in this document is executed in the root directory of this repository.

### Install prerequisites

This benchmark is partly based on Megatron-LM's codebase.
MindSpore replaces PyTorch as the NN training framework, so that it runs on Ascend.

### Prepare the data

AIPerf-MoE is using `enwik8` dataset for training.
Run the following command to download a pre-precessed dataset and place it in `data` directory.

```bash
mkdir -p data && curl https://pacman.cs.tsinghua.edu.cn/~laekov/moebench-data.tgz | tar -xz -C data
```

### Configuration

A configuration file is required by AIPerf-MoE, namely `config.yaml`.
An example configuation is shown in `config.default.yaml`.
Modification of any line in this configuation file is allowed for better performance.
Note that validation performance is not counted into final performance.
Therefore, to increase reported performance as high as possible, `eval_interval` should be set large enough in a final run.

Other parts of this benchmark is not allowed to be modified unless necessary.
Please report if a submission involves any modification other than `config.yaml`.

### Start testing

AIPerf-MoE is launched by `scripts/run.sh` for each process.

If distributed testing is being enabled, remenber to prepare the HCCL configuration file, and `pretrain.py` is supposed to be modified accordingly.

### Scoring

Evaluation is performed after training `train_iters` iterations.
**A valid run requires** the validation loss to be no greater than `3`.

MACs in all attention and MLP layers in forward and backward are counted as FLOP, regardless of their precision.
FP16, FP32 and any other FP format are regarded as the same, as long as the run is valid.
Overall FLOP per second since training begins is the only metric for ranking.
