# PSENet-Paddle

基于Paddle框架的PSENet复现

本项目基于paddlepaddle框架复现PSENet，并参加百度第三届论文复现赛，将在2021年5月15日比赛完后提供AIStudio链接～敬请期待

[AIStudio链接](https://aistudio.baidu.com/aistudio/projectdetail/1899550?shared=1)

参考项目：

[whai362-PSENet](https://github.com/whai362/PSENet)

# 环境配置

本项目利用`AIstudio`平台，采用paddlepaddle: 2.0.2-gpu Version，除此之外你需要通过`pip install mmcv editdistance Polygon3 pyclipper`或者`pip install -r requirement.txt`来安装依赖包

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/f9d36c2370f04d96bc0883683af5ec93bfd13c06ea3d4f049723bd6be935a8f7" width="400"/></center>

# 数据集
本项目已搭载PSENet比赛指定数据集，你可以[在此](https://aistudio.baidu.com/aistudio/datasetdetail/86292)找到搭载的数据集，包含`ICDAR2015 Task4`以及`Total-Text`

# 工程目录

**注意到你需要将`submitPSENet`重命名为`PSENet`**

```
/home/aistudio/PSENet
|───data(解压的data.zip)
└───config
└───models
└───dataset
└───eval
└───utils
└───compile.sh
└───__init__.py
└───test.py
└───train.py
└───requirement.txt
└───logo.gif
```

# 项目配置**

**注意：由于aistudio的docker环境并不适配本项目的编译，所以你需要在本地计算机编译完成后上传编译文件，在本地计算机我才用如下配置，你可以使用`gcc --version`和`g++ --version`查看配置**

| AIStudio|Local PC|
| --------     | -------- | 
|gcc (Ubuntu 7.5.0-3ubuntu1~16.04) 7.5.0<br>Copyright (C) 2017 Free Software Foundation, Inc.<br>This is free software; see the source for copying conditions.  There is NO<br>warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.|gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0<br>Copyright (C) 2017 Free Software Foundation, Inc.<br>This is free software; see the source for copying conditions.  There is NO<br>warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.| 
|g++ (Ubuntu 5.4.0-6ubuntu1~16.04.12) 5.4.0 20160609<br>Copyright (C) 2015 Free Software Foundation, Inc.<br>This is free software; see the source for copying conditions.  There is NO<br>warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.|g++ (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0<br>Copyright (C) 2017 Free Software Foundation, Inc.<br>This is free software; see the source for copying conditions.  There is NO<br>warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.| 

可以发现AIStudio的`g++`版本不适配，**注意：你需要相同的架构，系统以及python版本，(Ubuntu)linux-x86_64&python3.7**

```
`./compile.sh` or `bash compile.sh` if come out bash: ./compile.sh: Permission denied
```
或者直接进入指定目录，手动编译
```
cd /home/aistudio/PSENet/models/post_processing/pse
python setup.py build_ext --inplace
```
编译完成后你会在`/home/aistudio/PSENet/models/post_processing/pse`得到`build/temp.linux-x86_64-3.7/pse.o`文件和`pse.cpython-37m-x86_64-linux-gnu.so`文件

***注意：本项目已经全部配置完成，这一步无需操作***

# 训练

**需要注意的是，在paddlepaddle-2.0.2中并不支持字典数据读取，因此我在`/home/aistudio/PSENet/utils/data_loader.py`利用迭代器重写了`DataLoader`这拉慢了数据读取的速度，会导致训练速度略慢，例如在使用`psenet_r50_ic15_1024_finetune.py`训练一个epoch需要512.4秒，另外`paddlepaddle2.0.2`暂不支持`Identity`方法，因此我在`/home/aistudio/PSENet/models/utils/fuse_conv_bn.py`通过继承`Paddle.nn.Layer`写了`Identity`类**

```
cd /home/aistudio/PSENet/
python train.py ${CONFIG_FILE}
```
例如：
```
cd /home/aistudio/PSENet/
python train.py config/psenet/psenet_r50_ic15_736.py
```
训练开启时，会生成一个类似`/home/aistudio/PSENet/checkpoints/psenet_r50_ic15_1024_finetune`的文件夹，里面将保存权重和优化器参数

# 测试
```
cd /home/aistudio/PSENet/
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```
例如：
```
cd /home/aistudio/PSENet/
python test.py config/psenet/psenet_r50_ic15_736.py PSENet/PretrainedModel/checkpoint_ic15_736.pdparams
```

# 评估
**你需要注意的是：测试和评估是递进的，通过测试生成文件后，进行评估**
## [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4)
```shell script
cd /home/aistudio/PSENet/eval
`./eval_ic15.sh` or `bash ./eval_ic15.sh`
```
你会得到如下类似信息：
```
Calculated!{"precision": 0.8620689655172413, "recall": 0.7944150216658642, "hmean": 0.826860435980957, "AP": 0}
```
以下是`paddlepaddle`预训练模型测试指标
| Method | Backbone | Fine-tuning | Scale | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | N | Shorter Side: 736 | psenet_r50_ic15_736.py| 83.6 | 74.0 | 78.5 | checkpoint_ic15_736 |
| PSENet | ResNet50 | N | Shorter Side: 1024 | psenet_r50_ic15_1024.py| 84.4 | 76.3 | 80.2 | checkpoint_ic15_1024 |
| PSENet | ResNet50 | Y | Shorter Side: 736 | psenet_r50_ic15_736_finetune.py| 85.3 | 76.8 | 80.9 |checkpoint_ic15_736_finetune |
| PSENet | ResNet50 | Y | Shorter Side: 1024 | psenet_r50_ic15_1024_finetune.py| 86.2 | 79.4 | 82.7 | checkpoint_ic15_1024_finetune |

## [Total-Text](https://github.com/cs-chan/Total-Text-Dataset)
Text detection
```shell script
cd /home/aistudio/PSENet/eval
./eval_tt.sh or `bash ./eval_tt.sh`
```
你会得到如下类似信息：
```
Precision:_0.8727937336814604_______/Recall:_0.7786751361161512/Hmean:_0.8230524859472805

pb
```

以下是`paddlepaddle`预训练模型测试指标
| Method | Backbone | Fine-tuning | Config | Precision (%) | Recall (%) | F-measure (%) | Model |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PSENet | ResNet50 | N | psenet_r50_tt.py | 87.3 | 77.9 | 82.3 |checkpoint_tt |
| PSENet | ResNet50 | Y | psenet_r50_tt_finetune.py | 89.3 | 79.6 | 84.2 | checkpoint_tt_finetune|

## 速度测试
```shell script
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --report_speed
```
例如：
```
cd /home/aistudio/PSENet/
python test.py config/psenet/psenet_r50_ic15_736.py PSENet/PretrainedModel/checkpoint_ic15_736.pdparams --report_speed
```
你会得到如下类似信息
```
Testing 283/3000
backbone_time: 0.0152
neck_time: 0.0029
det_head_time: 0.0005
det_pse_time: 0.0660
FPS: 11.8
Testing 284/3000
backbone_time: 0.0152
neck_time: 0.0029
det_head_time: 0.0005
det_pse_time: 0.0660
FPS: 11.8
Testing 285/3000
backbone_time: 0.0152
neck_time: 0.0029
det_head_time: 0.0005
det_pse_time: 0.0660
FPS: 11.8
Testing 286/3000
backbone_time: 0.0152
neck_time: 0.0029
det_head_time: 0.0005
det_pse_time: 0.0660
FPS: 11.8
```

# Citation
```
@inproceedings{wang2019shape,
  title={Shape robust text detection with progressive scale expansion network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```
