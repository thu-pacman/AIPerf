![](logo.JPG)

![](logo_THU.jpg) ![](logo_PCL.jpg) 

**<font size=4>开发单位：清华大学(THU)，鹏城实验室(PCL) </font>**

**<font size=4>特别感谢国防科技大学窦勇老师及其团队的宝贵意见和支持</font>**




# <span id="head1">AIPerf Benchmark v1.0</span>

## <span id="head2"> Benchmark结构设计</span>

**关于AIPerf设计理念，技术细节，以及测试结果，请参考论文：https://arxiv.org/abs/2008.07141** 

AIPerf Benchmark基于微软NNI开源框架，以自动化机器学习（AutoML）为负载，使用network morphism进行网络结构搜索和TPE进行超参搜索。



# 部署 AIPerf

## 1. 环境NFS准备

**AIPerf代码**等文件必须放置在共享存储下，能被所有计算节点访问到

具体如何配置NFS请自行查阅资料或者联系集群管理员

## 2. 数据集准备

**数据集下载**

 *Imagenet官方地址：http://www.image-net.org/index* 

官方提供四种数据集：  Flowers、CIFAR-10、MNIST、ImageNet-2012  前三个数据集数据量小，直接调用相关脚本自动会完成下载、转换（TFRecord格式）的过程，在 /userhome/AIPerf/scripts/build_data目录下执行以下脚本：

```javascript
cd  /userhome/AIPerf/scripts/build_data
./download_imagenet.sh
```

原始的ImageNet-2012下载到当前的imagenet目录并包含以下两个文件:

- ILSVRC2012_img_val.tar
- ILSVRC2012_img_train.tar


**TFReord制作**

训练集和验证集需要按照1000个子目录下包含图片的格式，处理步骤：

1. 将train 和 val 的数据按照文件夹分类
2. 指定参数运行build_imagenet_data.py

**可以按照以下步骤执行**:  假设数据存放在/userhome/AIPerf/scripts/build_data/imagenet目录下，TFRecord文件的输出目录是/userhome/AIPerf/scripts/build_data/ILSVRC2012/output

```shell
# 做验证集
cd  /userhome/AIPerf/scripts/build_data
mkdir -p ILSVRC2012/raw-data/imagenet-data/validation/  
tar -xvf imagenet/ILSVRC2012_img_val.tar -C ILSVRC2012/raw-data/imagenet-data/validation/
python preprocess_imagenet_validation_data.py ILSVRC2012/raw-data/imagenet-data/validation/ imagenet_2012_validation_synset_labels.txt

# 做训练集
mkdir -p ILSVRC2012/raw-data/imagenet-data/train/
tar -xvf imagenet/ILSVRC2012_img_train.tar -C ILSVRC2012/raw-data/imagenet-data/train/ && cd ILSVRC2012/raw-data/imagenet-data/train
find . -name "*.tar" | while read NAE ; do mkdir -p "${NAE%.tar}"; tar -xvf "${NAE}" -C "${NAE%.tar}"; rm -f "${NAE}"; done
cd -

# 请注意 如果您使用的框架非TensorFlow 则不需要下面制作TFRecord的步骤！
# 执行转换
mkdir -p ILSVRC2012/output
python build_imagenet_data.py --train_directory=ILSVRC2012/raw-data/imagenet-data/train --validation_directory=ILSVRC2012/raw-data/imagenet-data/validation --output_directory=ILSVRC2012/output --imagenet_metadata_file=imagenet_metadata.txt --labels_file=imagenet_lsvrc_2015_synsets.txt
```

对TensorFlow 上面步骤执行完后，路径ILSVRC2012/output包含128个validation开头的验证集文件和1024个train开头的训练集文件。需要分别将验证集和数据集移动到slave节点的物理机上

```shell
mkdir -p /root/datasets/imagenet/train
mkdir -p /root/datasets/imagenet/val
mv ILSVRC2012/output/train-* /root/datasets/imagenet/train
mv ILSVRC2012/output/validation-* /root/datasets/imagenet/val
```

对其他模型，分别将解压后验证集和数据集移动到slave节点的物理机上，这一部分可以根据框架需求自行调整

```shell
# 对 PyTorch
mkdir -p /root/datasets/imagenet/
mv ILSVRC2012/train /root/datasets/imagenet
mv ILSVRC2012/val /root/datasets/imagenet
```

**为了使读数据不成为训练瓶颈，有条件的集群最好能够将数据放在每台机器的本地存储（而不是共享存储）并在训练时从本地存储读数据**

## 3. 项目安装

#### 重要的环境变量声明

```bash
# 假设共享存储目录为/share/
export AIPERF_WORKDIR=/share/aiperf_workspace
export AIPERF_SLAVE_WORKDIR=/root/
export AIPERF_MASTER_IP=10.0.1.100
export AIPERF_MASTER_PORT=9987
```

* AIPERF_WORKDIR

  AIPerf工作目录，**必须在共享存储上**，能被所有节点访问到

  AIPerf代码，部分log文件等将会被放到这个目录

* AIPERF_SLAVE_WORKDIR

  计算节点工作目录，不需要在共享存储上（也可以放在共享存储）

  部分计算节点log文件将会被生成到这个目录下

* AIPERF_MASTER_IP

  AIPerf调度服务IP，即控制节点IP

* AIPERF_MASTER_PORT

  AIPerf调度服务端口

#### 在共享存储上创建AIPerf工作目录

```bash
mkdir -p $AIPERF_WORKDIR
```

#### 下载AIPerf到AIPerf工作目录

```bash
git clone https://github.com/thu-pacman/AIPerf.git $AIPERF_WORKDIR/AIPerf
```

* AIPerf目录下有```aiperf_setenv.sh```文件，可以通过这个来统一设置环境变量

  ```bash
  # 设置aiperf_setenv.sh里的环境变量
  # 使用环境变量
  source $AIPERF_WORKDIR/AIPerf/aiperf_setenv.sh
  ```

#### 安装项目依赖

##### 安装CUDA与cuDNN

结合集群情况自行安装在**计算节点**

* 请确保CUDA版本，cuDNN版本和计算框架的一致性，对于tensorflow计算框架，可参考[官方文档](https://www.tensorflow.org/install/source_windows#gpu)

##### 配置python依赖

对于控制节点安装依赖

```bash
python3 -m pip install -r $AIPERF_WORKDIR/AIPerf/requirements_master.txt
```

对于**所有计算节点**安装依赖

```bash
python3 -m pip install -r $AIPERF_WORKDIR/AIPerf/requirements_slave.txt
# 根据CUDA版本，cuDNN版本选择计算框架版本，例如对于CUDA 11.2，cuDNN 8.1，可选择tensorflow 2.5.0
python3 -m pip install tensorflow==2.5.0
```

* 请确保CUDA版本，cuDNN版本和计算框架的一致性，对于tensorflow计算框架，可参考[官方文档](https://www.tensorflow.org/install/source_windows#gpu)

#### 安装AIPerf

编译安装

```bash
# 安装AutoML组件
cd $AIPERF_WORKDIR/AIPerf/src/sdk/pynni/
python3 -m pip install -e .
# 安装aiperf控制组件
cd $AIPERF_WORKDIR/AIPerf/src/aiperf_manager/
python3 -m pip install -e .
```

检查aiperf安装

执行

```bash
aiperf --help
```

正常打印帮助信息表示安装成功

#### 下载模型权重

将权重文件下载到AIPerf工作目录中

```bash
wget -P $AIPERF_WORKDIR https://github.com/AI-HPC-Research-Team/Weight/releases/download/AIPerf1.0/resnet50_weights_tf_dim_ordering_tf_kernels.h5
```

# 4. 启动测试

## 启动调度服务

### 配置计算节点

进入 ```${AIPERF_WORKDIR}/AIPerf/aiperf_ctrl```配置 servers.json ，每张计算卡的描述包括`ip`和`CUDA_VISIBLE_DEVICES`两部分，应保证servers.json的list的长度恰好等于等待测试的计算节点总数

例如，若该集群有2个节点，每个节点有4张卡，则编写如下

```json
[
    {
        "ip": "172.23.33.33",
        "tag": "",
        "status": "waiting",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3"
    },
    {
        "ip": "172.23.33.34",
        "tag": "",
        "status": "waiting",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3"
    }   
]
```

### 配置计算节点初始化环境

请修改```${AIPERF_WORKDIR}/AIPerf/aiperf_ctrl/server_env_init.sh```以适配计算节点的环境加载

例如，若计算节点需加载spack环境，并用spack load加载cuda/cudnn环境，则编写如下

```bash
# 计算节点环境加载 根据计算节点实际环境编写
. /share/aiperf_workspace/spack/share/spack/setup-env.sh
spack config add modules:prefix_inspections:lib64:[LD_LIBRARY_PATH]
spack config add modules:prefix_inspections:lib:[LD_LIBRARY_PATH]

spack load python
spack load cuda@11.2
spack load cudnn@8.1
```

### 调度服务启动

```bash
cd $AIPERF_WORKDIR/AIPerf/aiperf_ctrl
python3 manage.py runserver ${AIPERF_MASTER_IP}:${AIPERF_MASTER_PORT}
```

保持该服务一直运行

## 启动AIPerf测试

**启动实验**

为了使结果有效，测试满足的基本条件是：
1. 测试运行时间应不少于1小时；
2. 测试的计算精度不低于FP-16；
3. 测试完成时所取得的最高正确率应大于70%；

### 初始化配置

*(以下操作均在控制节点进行)*
根据需求修改```${AIPERF_WORKDIR}/AIPerf/examples/trials/network_morphism/imagenet/config.yml```配置

|      |         可选参数         |                说明                 |       默认值        |
| ---- | :----------------------: | :---------------------------------: | :-----------------: |
| 1    |     trialConcurrency     |          同时运行的trial数          |          1          |
| 2    |     maxExecDuration      |       设置测试时间(单位 ：h)        |         12          |
| 3    |   CUDA_VISIBLE_DEVICES   |      指定测试程序可用的gpu索引      |   0,1,2,3,4,5,6,7   |
| 4    | srun：--cpus-per-task=30 |     参数为slurm可用cpu核数减 1      |         30          |
| 5    |         --slave          | **跟 trialConcurrency参数保持一致** |          1          |
| 6    |           --ip           |    master节点ip，直接使用默认值     | ${AIPERF_MASTER_IP} |
| 7    |       --batch_size       |             batch size              |         448         |
| 8    |         --epochs         |           正常训练epoch数           |         60          |
| 9    |       --initial_lr       |             初始学习率              |        1e-1         |
| 10   |        --final_lr        |             最终学习率              |          0          |
| 11   |     --train_data_dir     |           训练数据集路径            |        None         |
| 12   |      --val_data_dir      |           验证数据集路径            |        None         |
| 13   |        --warmup_1        |      warm up机制第一轮epoch数       |         15          |
| 14   |        --warmup_2        |      warm up机制第二轮epoch数       |         30          |
| 15   |        --warmup_3        |      warm up机制第三轮epoch数       |         45          |
| 16   |   --num_parallel_calls   |        tfrecord数据加载加速         |         48          |

可参照如下配置：

```
authorName: default
experimentName: example_imagenet-network-morphism-test
trialConcurrency: 2		# 1
maxExecDuration: 12h	# 2
maxTrialNum: 30000
trainingServicePlatform: local
useAnnotation: false
tuner:
 \#choice: TPE, Random, Anneal, Evolution, BatchTuner, NetworkMorphism
 \#SMAC (SMAC should be installed through nnictl)
 builtinTunerName: NetworkMorphism
 classArgs:
  optimize_mode: maximize
  task: cv
  input_width: 224
  input_channel: 3
  n_output_node: 1000
  
trial:
 command: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  \                                  # 3
       python3 imagenet_train.py \
       --slave 2 \								  # 5
       --ip ${AIPERF_MASTER_IP} \							  # 6
       --batch_size 448 \						  # 7
       --epoch 60 \						          # 8
       --initial_lr 1e-1 \						  # 9
       --final_lr 0 \						  # 10
       --train_data_dir /root/datasets/imagenet/train/ \  # 11
       --val_data_dir /root/datasets/imagenet/val/ # 12

 codeDir: .
 gpuNum: 0
```

* 注意配置文件中slave参数需要和trialConcurrency参数一致

### 运行benchmark

在```${AIPERF_WORKDIR}/AIPerf/examples/trials/network_morphism/imagenet/```目录下执行以下命令运行用例

注：若使用pyTorch，请在``` ${AIPERF_WORKDIR}/AIPerf/examples/trials/network_morphism/imagenetTorch/```下执行

```
aiperf create -c config.yml
```

#### **查看运行过程**

当测试运行过程中，运行以下程序会在终端打印experiment的Error、Score、Regulated Score等信息

```
python3 $AIPERF_WORKDIR/AIPerf/scripts/reports/report.py --id  experiment_ID  
```

同时会产生实验报告存放在experiment_ID的对应路径```${AIPERF_WORKDIR}/mountdir/nni/experiments/experiment_ID/results```目录下

实验成功时报告为 Report_Succeed.html

实验失败时报告为 Report_Failed.html

实验失败会报告失败原因，请查阅AI Benchmark测试规范分析失败原因

#### <span id="head13"> 停止实验</span>

停止expriments,  退出前台的aiperf进程即可


**保存日志&结果数据**

运行以下程序可将测试产生的日志以及数据统一保存到```${AIPERF_WORKDIR}/mountdir/nni/experiments/experiment_ID/results/logs```中，便于实验分析

```
python3 $AIPERF_WORKDIR/AIPerf/scripts/reports/report.py --id  experiment_ID  --logs True
```

由于实验数据在复制过程中会导致额外的网络、内存、cpu等资源开销，建议在实验停止/结束后再执行日志保存操作。

# 5. 测试参数设置及推荐环境配置

#### <span id="head15"> 可变设置</span>

1. slave计算节点的GPU卡数：默认将单个物理服务器作为一个slave节点，并使用其所有GPU；
2. 深度学习框架：默认使用keras+tensorflow；
3. 数据集加载方式：默认将数据预处理成TFRecord格式，以加快数据加载的效率；
4. 数据集存储方式：默认采用网络共享存储；
5. 超参设置：默认初始batch size=448，默认初始学习率=0.1，默认最终学习率=0，默认正常训练epochs=60，默认从第四轮trial开始，每个trial搜索1次，默认超参为kernel size和batch size。

#### <span id="head16"> 推荐环境配置</span>

- 环境：Ubuntu16.04

- 软件：TensorFlow2.2.0，CUDA10.2，python3.6
- Container：36个物理CPU核，512GB内存，8张GPU

***NOTE: 推荐基于Intel Xeon Skylake Platinum8268 and NVIDIA Tesla NVLink v100配置***

# 6. 二次开发与迁移

请参考[二次开发参考文档](./Migration.md)

#  7. Benchmark报告反馈

若测试中遇到问题，请联系zhaijidong@tsinghua.edu.cn，并附上`${AIPERF_WORKDIR}/mountdir/nni/experiments/experiment_ID/results/`中的html版报告。

## <span id="head18"> 许可</span>

基于 MIT license
