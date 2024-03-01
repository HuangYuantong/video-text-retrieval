# **代码：视频文本检索系统**

### 作用
此部分代码为完整、可运行的“视频文本检索系统”，
主要对应论文的第四章“系统设计与实现”。

### 运行方法
以cmd为例：
1. 首先cd进入到manage.py所在路径级
2. 然后确保python安装库满足requirements.txt，
确保res文件夹与项目位于同一级，且res文件夹下运行所需文件齐全
3. 然后运行命令`python manage.py runserver`即可启动视频文本检索系统

### 文件（夹）大致内容
1. static/： CSS样式文件、一些icon的图片；
2. VideoSearchSystem/： 对应论文“4.2系统架构”图中的视频检索模块，
模型的文本编码器、向量数据库的load接口将被启用；
3. website/，对应论文“4.2系统架构”图中的Web模块、文字匹配模块： 
   1. /template/website/： HTML模板，Web前端的逻辑实现，对应Django的视图层；
   2. /models.py： 文本数据库的定义，对应Django的模型层；
   3. /db_build.py： 文本数据库的构造接口，通过“网址/db_construct”访问；
   4. /urls.py： Django的url分发机制；
   5. /views.py： Web后端的逻辑实现，对应Django的模板层；
4. Graduation_Project/setting.py： Django项目配置文件，如静态资源映射、接口白名单等等；
5. db.sqlite3：文本数据库的SqLite数据库实体；
6. requirements.txt、README.py

### 运行所需文件（res文件下）
1. 保存的模型参数：CLIP4Clip.pt或AIM.pt；
2. 原始视频：MSR-VTT文件夹下，存有视频库中所有原始视频，用于视频播放；
3. 向量数据库：TensorDataBase/MSRVTT_test.pkl，存储所有视频特征；
4. 视频封面：thumbnails文件夹，存有视频对应的视频封面，用于检索结果展示界面；
5. 若要进行文本数据库初始化（db.sqlite3文件丢失或不符合项目时），则额外需要msrvtt_data文件夹。