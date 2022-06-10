#!/bin/bash
DATA_PATH=imagenet
if [ ! -d $DATA_PATH ];then
    mkdir -p $DATA_PATH
fi
cd $DATA_PATH
while true
do
    echo "开始下载训练集"
    aria2c -x 16  "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar?username=reburnw&Accesskey=80989ceb2977eeaf29b98705672144b2b93c31ca"
    if [ $? -ne 0 ];then
        echo -e "\033[31m\n网络错误，正在重试，支持断点续传\n\033[0m"
        continue
    fi
    echo "训练集下载成功！"
    echo “开始下载验证集”
    aria2c -x 16  "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar?username=reburnw&Accesskey=80989ceb2977eeaf29b98705672144b2b93c31ca"
    if [ $? -ne 0 ];then
        echo -e "\033[31m\n网络错误，正在重试，支持断点续传\n\033[0m"
        continue
    fi
    echo "验证集下载成功！"
    break
done
