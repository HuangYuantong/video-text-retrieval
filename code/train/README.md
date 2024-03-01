# **代码：训练**

### 作用
此部分代码进行模型训练

### 运行方法
1. 运行以下命令即可启动训练：

`python -m torch.distributed.launch --nproc_per_node=1 main_task_retrieval.py`

2. 所有超参定义、初始值在get_args.py中；
3. 训练时的具体设置在main_task_retrieval.py的主函数中进行修改。