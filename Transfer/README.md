# 阿里天池·雪浪

## 初赛赛题简介：训练集1316(normal) + 706(abnormal), 2分类

## 项目说明

### 运行环境

- python 3.6.5

- tensorflow-gpu 1.9.0

- keras 2.2.0

- Pillow 5.1.0

- Numpy 1.14.3

- Pandas 0.23.0

### 数据预处理与数据构造

- 任意路径：将3个part的训练数据完全下载，合并，放置于train/下，得到48个子文件夹，2728个文件(jpg+xml)

- 任意路径：下载test数据，放置于test/下，无类别，test下直接包含662个jpg文件

- 在项目<b>Transfer/</b>下执行以下命令：

```
mkdir dataset
mkdir dataset/train
mkdir dataset/val
mkdir dataset/train/normal
mkdir dataset/train/abnormal
mkdir dataset/val/normal
mkdir dataset/val/abnormal
mkdir dataset/test/0/
```

以下自行修改/your/abspath/to/train/为你的实际放置数据的路径
```
ln -s /your/abspath/to/train/正常/* /your/abspath/to/project/Transfer/dataset/train/normal/
ln -s /your/abspath/to/train/*/* /your/abspath/to/project/Transfer/dataset/train/abnormal/
ln -s /your/abspath/to/test/* /your/abspath/to/project/Transfer/dataset/test/0/
```

切换至train/abnormal/路径下，删除在normal中出现过的文件名（留下的是其余47个异常类别的文件）
```
cd dataset/train/abnormal/
ls ../normal/ | tr '\n' '\0' |  xargs -0 rm
```

执行下列命令，即可自动完成train/val的划分
核心代码：train_test_split@Utils line 46，可按需修改划分比例，但要考虑val对test的代表能力
```
cd /your/abspath/to/project/Transfer/

python3
from Utils import *
datapath = 'dataset/'
moveImg(datapath)
```

处理完成后整个目录结构如下：
```
dataset/
├── test
│   └── 0
│       ├── J01_2018.06.13\ 13_22_11.jpg -> /home/professorsfx/Xuelang/test/J01_2018.06.13\ 13_22_11.jpg
│       ├── ………………………………
│       └── J01_2018.06.28\ 15_52_02.jpg -> /home/professorsfx/Xuelang/test/J01_2018.06.28\ 15_52_02.jpg
├── train
│   ├── abnormal
│   │   ├── J01_2018.06.13\ 13_17_04.jpg -> /home/professorsfx/Xuelang/train/\346\223\246\346\264\236/J01_2018.06.13\ 13_17_04.jpg
│       ├── ………………………………
│   │   └── J01_2018.06.28\ 15_48_53.jpg -> /home/professorsfx/Xuelang/train/\346\223\246\346\264\236/J01_2018.06.28\ 15_48_53.jpg
│   └── normal
│       ├── J01_2018.06.13\ 13_23_08.jpg -> /home/professorsfx/Xuelang/train/\346\255\243\345\270\270/J01_2018.06.13\ 13_23_08.jpg
│       ├── ………………………………
│       └── J01_2018.06.28\ 15_50_57.jpg -> /home/professorsfx/Xuelang/train/\346\255\243\345\270\270/J01_2018.06.28\ 15_50_57.jpg
├── train_split.txt
├── val
│   ├── abnormal
│   │   ├── J01_2018.06.13\ 13_43_44.jpg -> /home/professorsfx/Xuelang/train/\350\267\263\350\212\261/J01_2018.06.13\ 13_43_44.jpg
│       ├── ………………………………
│   │   └── J01_2018.06.28\ 15_02_16.jpg -> /home/professorsfx/Xuelang/train/\346\223\246\346\264\236/J01_2018.06.28\ 15_02_16.jpg
│   └── normal
│       ├── J01_2018.06.13\ 13_24_39.jpg -> /home/professorsfx/Xuelang/train/\346\255\243\345\270\270/J01_2018.06.13\ 13_24_39.jpg
│       ├── ………………………………
│       └── J01_2018.06.28\ 15_20_01.jpg -> /home/professorsfx/Xuelang/train/\346\255\243\345\270\270/J01_2018.06.28\ 15_20_01.jpg
└── val_split.txt

8 directories, 2686 files
```

### 模型介绍

- 以迁移学习为主，结合图像特性修改模型结构，使用特征工程等竞赛技巧进行优化。

- 可使用DenseNet201，InceptionResNetV2，ResNet50，Xception，InceptionV3，NasNetMobile等网络，均来自于keras.application。

- 上述模型均可使用imagenet预训练权值为基础进行迁移学习

### 代码介绍及Usage

Note：虽然我们提供了训练权值及相关训练、测试代码，但由于我们对训练数据做了补充，且我们划分的val源自一次随机划分。可能导致在原始数据下重新训练和预测的精度与我们的线上成绩有所出入。

#### 训练

- Usage: ```python3 Train.py --channel=3 --modelType=densenet --loss=cc```

- 参数说明：channel：3,4或5，默认为3.指定使用RGB还是RGB+Gray，或者是RGB+Gray+Gabor，详见Model.py

- 参数说明：modelType从["densenet", "InceptionResNetV2", "Resnet", "xception", "inception", "nas"]中选择

- 参数说明：loss从["cc", "ccs", "focal"]中选择。cc=传统交叉熵。ccs=交叉熵+smooth，相当于交叉熵增加正则项，源自inceptionv3的trick，代码位于Losses.py. focal loss源自Retinanet

- 训练说明：必须先在3通道下完成Imagenet到竞赛数据的迁移（训练至earlystop或val_acc=1），代码会自动保存best_checkpoint；再在其它通道选项下训练至earlystop。

- 注意：必须先在3通道下完成迁移训练，其余通道选项只会加载当前路径下的模型权重，而不会加载imagenet。如果直接开始训练4、5通道，相当于从随机初始化训练CNN，将无法收敛！

- 训练说明：我们使用了三个callback来确保模型达到最佳拟合状态而没有过拟合：EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

#### 预测

- Usage ```python3 Test.py --channel=4 --modelType=densenet```

- 参数说明：与train部分参数意义相同，但注意与train的训练参数保持一致

- 预测说明：正确设定test数据路径后，在当前目录下生成一个result.csv