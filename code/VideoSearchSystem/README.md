# **代码：算法部分**

## 1 作用
此部分代码包括关键帧选取、关键帧保存、模型、向量数据库四个部分，
主要对应论文的第三章“算法设计与实现”。 

为训练（train）、视频文本检索系统（Graduation_Project）提供数据准备。

## 2 文件（夹）大致内容
1. get_args.py： 所有文件路径、超参数定义；
2. Tools.py： 一些零碎的对外小接口，功能请参见其下各函数的说明；
3. VideoSearchSystem.py： 包装该模块为“视频检索模块”，为项目整体提供search接口；
4. requirements.txt、README.py
### 2.1 模型部分
Model/：
1. /clip/： CLIP预训练模型发布的代码，除添加注释外未改动： 
   1. clip.py是CLIP结构定义；
   2. model.py是CLIP加载与下载等；
   3. simple_tokenizer.py是完整的tokenizer；
2. CLIP4Clip.py是使用meanP的CLIP4Clip的结构定义；
3. AIM.py是Adapter结构与插入方法的定义；
4. My_Model.py是模型的封装，进行CLIP4Clip和AIM的初始化、加载、冻结等。
### 2.2 关键帧选取部分
preprocess/frame_extract/：
1. /keyFrameExtractor_diff.py： 最大帧间差选取方案，提供keyFrameExtractor_diff接口；
2. /keyFrameExtractor_Uniform.py： 平均选取方案，提供keyFrameExtractor_Uniform接口；
### 2.3 关键帧保存部分
1. preprocess/frame_extract/frame_extract_tool.py： 关键帧保存，提供extract_all_videos接口。
2. preprocess/dataloader_keyFrame.py： 使用关键帧保存之后的DataLoader；
3. 为DataLoader生成训练时用到的train.csv文件；
### 2.4 向量数据库部分
TensorDataBase/向量数据库：
1. TensorDataBase.py： 向量数据库的定义，提供load、store接口；
2. /construct.py： 包装store接口，提供向量数据库的构造函数；
3. /construct_dataloader.py： 包装DataLoader，为construct.py中的函数提供加速。

## 3 运行所需文件（res文件下）
1. 数据集的原始视频：MSRVTT文件夹，关键帧抽取、系统的视频播放等过程需要；
2. 数据集的视频描述、训练集与测试集划分：msrvtt_data文件夹，train.csv文件生成、向量数据库初始化等过程需要；
3. 生成：key_frames_diff，存储最大帧间差选取方案抽取出的关键帧；
4. 生成：key_frames_uniform，存储平均差选取方案抽取出的关键帧；
5. 生成：outputs，存储训练过程的输出log和checkpoint；
6. 生成：TensorDataBase，存储向量数据库；
7. 生成：thumbnails，存储系统的检索结果展示页面的视频封面。