![](logo.JPG)

![](logo_THU.jpg) ![](logo_PCL.jpg) 

**<font size=4>开发单位：清华大学(THU)，鹏城实验室(PCL) </font>**

**<font size=4>特别感谢国防科技大学窦勇老师及其团队的宝贵意见和支持</font>**




# <span id="head1">AIPerf Benchmark v1.0</span>

## <span id="head2"> Benchmark结构设计</span>

**关于AIPerf设计理念，技术细节，以及测试结果，请参考论文：https://arxiv.org/abs/2008.07141** 

AIPerf Benchmark基于微软NNI开源框架，以自动化机器学习（AutoML）为负载，使用network morphism进行网络结构搜索和TPE进行超参搜索。



# 部署 AIPerf

# 1. 环境NFS准备

配置共享文件系统需要在物理机环境中进行，若集群环境中已有共享文件系统则跳过配置共享文件系统的步骤,若无共享文件系统，则需配置共享文件系统。

*安装NFS服务端*

将NFS服务端部署在master节点

```
apt install nfs-kernel-server -y
```

*配置共享目录*

创建共享目录/userhome，后面的所有数据共享将会在/userhome进行

```
mkdir /userhome
```

*修改权限*

```
chmod -R 777 /userhome
```

*打开NFS配置文件，配置NFS*

```
vim /etc/exports
```

添加以下内容

```
/userhome   *(rw,sync,insecure,no_root_squash)
```

*重启NFS服务*

```
service nfs-kernel-server restart
```

*安装NFS客户端*

所有slave节点安装NFS客户端

```
apt install nfs-common -y
```

slave节点创建本地挂载点

```
mkdir /userhome
```

slave节点将NFS服务器的共享目录挂载到本地挂载点/userhome

```
mount NFS-server-ip:/userhome /userhome
```

*检查NFS服务*

在任意节点执行

```
touch /userhome/test
```

如其他节点能在/userhome下看见 test 文件则运行正常。

# 2. 数据集准备

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


# 3. 安装项目依赖

**配置python运行环境**

请保证python版本，CUDA版本，计算框架版本的一致性，在这里，我们以`python3.6`, `CUDA10.1`, `tensorflow 2.2.0` 为例

*安装python3.6*

```
apt install --install-recommends python3 python3-dev python3-pip -y
```

*升级pip*

```
pip3 install --upgrade pip
```

**安装AIPerf**

*下载源代码到共享目录/userhome*

```shell
git clone https://github.com/AI-HPC-Research-Team/AIPerf.git /userhome/AIPerf
```

*安装python环境库*

```
cd /userhome/AIPerf
pip3 install -r requirements.txt --timeout 3000
```

*编译安装*

```shell
# 安装AutoML组件
cd /userhome/AIPerf/src/sdk/pynni/
pip3 install -e .
# 安装aiperf控制组件
cd /userhome/AIPerf/src/aiperf_manager/
pip3 install -e .
```

*检查AIPerf安装*

执行

```
aiperf --help
```

出现：
```
usage: aiperf [-h] [--version] {create} ...

use aiperfctl command to control aiperf experiments

positional arguments:
  {create}
    create       create a new experiment

optional arguments:
  -h, --help     show this help message and exit
  --version, -v

```

表示安装成功

**目录调整**

*创建必要的目录*

mountdir 存放实验过程数据，nni存放实验过程日志

```shell
mkdir /userhome/mountdir
mkdir /userhome/nni
```

将共享目录下的相关目录链接到用户home目录下

```shell
ln -s /userhome/mountdir /root/mountdir
ln -s /userhome/nni /root/nni
```

*必要的路径及数据配置*

 将权重文件复制到共享目录/userhome中

```shell
wget -P /userhome https://github.com/AI-HPC-Research-Team/Weight/releases/download/AIPerf1.0/resnet50_weights_tf_dim_ordering_tf_kernels.h5
```

在共享目录下配置工作节点的总数

```shell
echo 16 > /userhome/trial_concurrency.txt
```

# 4. 启动测试

**启动调度服务**

进入 aiperf_ctrl服务，配置 servers.json ，每张计算卡的描述包括`ip`和`CUDA_VISIBLE_DEVICES`两部分，应保证servers.json的list的长度恰好等于等待测试的计算卡总数，也等于之前填写的trial_concurrency.txt内的数字

slave节点和调度服务通过http协议进行运行时信息交互，请在全文搜索四处`255.255.255.255`，并替换为本机的IP地址

例如，本机的ip地址为127.0.0.1，请在aiperf_ctrl下执行

```python3
python3 manage.py runserver 127.0.0.1:9987
```

请调整`aiperf_ctrl/trial/views.py`下的相关语句, 包括：

1. `sshKill` `sshExec` 中使用ssh下发命令的方式，使用密钥或者证书
2. `sshKill` `sshExec` 中相关的代码路径 TODO: 改成自动监测
3. `sshExec` 中加载环境的代码`module load ...`按需选择保留与修改

并保持该服务一直运行

**启动实验**

为了使结果有效，测试满足的基本条件是：
1. 测试运行时间应不少于1小时；
2. 测试的计算精度不低于FP-16；
3. 测试完成时所取得的最高正确率应大于70%；

#### <span id="head11"> 初始化配置</span>

*(以下操作均在master节点进行)*
根据需求修改/userhome/AIPerf/examples/trials/network_morphism/imagenet/config.yml配置

|      |         可选参数         |              说明               |     默认值      |
| ---- | :----------------------: | :-----------------------------: | :-------------: |
| 1    |     trialConcurrency     |        同时运行的trial数        |        1        |
| 2    |     maxExecDuration      |     设置测试时间(单位 ：h)      |       12        |
| 3    |   CUDA_VISIBLE_DEVICES   |    指定测试程序可用的gpu索引    | 0,1,2,3,4,5,6,7 |
| 4    | srun：--cpus-per-task=30 |   参数为slurm可用cpu核数减 1    |       30        |
| 5    |         --slave          | 跟 trialConcurrency参数保持一致 |        1        |
| 6    |           --ip           |          master节点ip           |    127.0.0.1    |
| 7    |       --batch_size       |           batch size            |       448       |
| 8    |         --epochs         |         正常训练epoch数         |       60        |
| 9    |       --initial_lr       |           初始学习率            |      1e-1       |
| 10   |        --final_lr        |           最终学习率            |        0        |
| 11   |     --train_data_dir     |         训练数据集路径          |      None       |
| 12   |      --val_data_dir      |         验证数据集路径          |      None       |
| 13   |        --warmup_1        |    warm up机制第一轮epoch数     |       15        |
| 14   |        --warmup_2        |    warm up机制第二轮epoch数     |       30        |
| 15   |        --warmup_3        |    warm up机制第三轮epoch数     |       45        |
| 16   |   --num_parallel_calls   |      tfrecord数据加载加速       |       48        |

可参照如下配置：

```
authorName: default
experimentName: example_imagenet-network-morphism-test
trialConcurrency: 1		# 1
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
       /GPUFS/thu_wgchen_2/aiperf/AIPerf-wxp/submitter.py \ # 请修改为正确的submitter 地址
       python3 imagenet_train.py \
       --slave 1 \								  # 5
       --ip 127.0.0.1 \							  # 6
       --batch_size 448 \						  # 7
       --epoch 60 \						          # 8
       --initial_lr 1e-1 \						  # 9
       --final_lr 0 \						  # 10
       --train_data_dir /root/datasets/imagenet/train/ \  # 11
       --val_data_dir /root/datasets/imagenet/val/ # 12

 codeDir: .
 gpuNum: 0
```

#### <span id="head12"> 运行benchmark</span>

在/userhome/AIPerf/examples/trials/network_morphism/imagenet/目录下执行以下命令运行用例

注：若使用pyTorch，请在 /userhome/AIPerf/examples/trials/network_morphism/imagenetTorch/下执行

```
aiperf create -c config.yml
```

**查看运行过程**

当测试运行过程中，运行以下程序会在终端打印experiment的Error、Score、Regulated Score等信息

```
python3 /userhome/AIPerf/scripts/reports/report.py --id  experiment_ID  
```

同时会产生实验报告存放在experiment_ID的对应路径/userhome/mountdir/nni/experiments/experiment_ID/results目录下

实验成功时报告为 Report_Succeed.html

实验失败时报告为 Report_Failed.html

实验失败会报告失败原因，请查阅AI Benchmark测试规范分析失败原因

#### <span id="head13"> 停止实验</span>

停止expriments,  退出前台的aiperf进程即可


**保存日志&结果数据**

运行以下程序可将测试产生的日志以及数据统一保存到/userhome/mountdir/nni/experiments/experiment_ID/results/logs中，便于实验分析

```
python3 /userhome/AIPerf/scripts/reports/report.py --id  experiment_ID  --logs True
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

若测试中遇到问题，请联系zhaijidong@tsinghua.edu.cn，并附上`/userhome/mountdir/nni/experiments/experiment_ID/results/`中的html版报告。

## <span id="head18"> 许可</span>

基于 MIT license
