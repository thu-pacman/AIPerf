# AIPerf 环境依赖

## 集群情况

可以通过ssh访问所有计算节点，计算节点与主节点可以通过socket通信，支持HTTP请求(交换训练过程信息)

## 共享文件系统NFS

所有节点均可以访问读写的目录，需要存放数据集， 建议预留至少*300G*空间

数据文件下载地址：
```
# 训练集
https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar?username=reburnw&Accesskey=80989ceb2977eeaf29b98705672144b2b93c31ca
# 验证集
https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar?username=reburnw&Accesskey=80989ceb2977eeaf29b98705672144b2b93c31ca
```

## Python 与 深度学习计算框架

**本部分依赖在年后新版本AIPerf中产生变化**

目前仅支持tensorflow2.2, 建议配置为：
```
TensorFlow2.2.0
CUDA10.1
python3.5.2 or python3.6
```

具体的依赖文件见：
```
https://github.com/thu-pacman/AIPerf/blob/wxp/requirements.txt
```

### **后续情况**

预计年后会提供基于ONNX的中间表示，由用户将tfModel或者onnxModel转换为自定义模型进行训练，要求训练时能够提供log记录，包括:
1. 每个Epoch的开始时间与结束时间
2. 每个Epoch结束后在训练集和验证集上的指标

## Nodejs

**本部分依赖在年后新版本AIPerf中不再需求**
```
Nodejs >= 10
对应的yarn包管理
```
需要安装的依赖见：
```
https://github.com/thu-pacman/AIPerf/blob/wxp/src/nni_manager/yarn.lock
```

### **后续情况**

预计年后不再需要Nodejs支持