# PyTorch Learning 🚀

这是一个用于学习 PyTorch 深度学习框架的实战仓库。
课程基于 Bilibili 刘二大人《PyTorch深度学习实践》完结版视频课程。

这份代码仓库包含了从最基础的线性回归到复杂的 CNN、RNN 序列模型的全套学习代码、个人笔记与项目实战，所有代码均保证可跑通，并详细标注了踩坑点与各种维度的计算思路。

## 📁 核心项目结构

| 讲次 | 文件名 | 核心知识点 |
| :---: | :--- | :--- |
| **02** | [`02_线性模型.ipynb`](./02_线性模型.ipynb) | 基础线性回归公式与前向传播 |
| **03** | [`03_梯度下降算法.ipynb`](./03_梯度下降算法.ipynb) | 随机梯度下降(SGD)原理实现计算 |
| **04** | [`04_反向传播.ipynb`](./04_反向传播.ipynb) | PyTorch `backward()` 计算图与自动链式求导机制 |
| **05** | [`05_用PyTorch实现线性回归.ipynb`](./05_用PyTorch实现线性回归.ipynb) | `nn.Module` 和最基础的模型套路搭建 |
| **06** | [`06_逻辑斯蒂回归.ipynb`](./06_逻辑斯蒂回归.ipynb) | Sigmoid 激活与 `BCELoss` 的分类运用 |
| **07** | [`07_处理多维特征的输入.ipynb`](./07_处理多维特征的输入.ipynb) | 多维张量处理与糖尿病数据集实战预测 |
| **08** | [`08_加载数据集.ipynb`](./08_加载数据集.ipynb) | **[核心]** 自定义 `Dataset`、重写魔法函数与 `DataLoader` mini-batch 打包 |
| **09** | [`09_多分类问题.ipynb`](./09_多分类问题.ipynb) | `CrossEntropyLoss` 机制理解及其内部整合的 Softmax |
| **10** | [`10_卷积神经网络_基础.ipynb`](./10_卷积神经网络_基础.ipynb) | `nn.Conv2d` 和 `nn.MaxPool2d`，通道数与特征图维度计算法 |
| **11** | [`11_卷积神经网络_高级.ipynb`](./11_卷积神经网络_高级.ipynb) | 1x1 卷积运用、Inception 融合模块及 ResNet 惨差网络思想 |
| **12** | [`12_循环神经网络_基础.ipynb`](./12_循环神经网络_基础.ipynb) | 获取序列数据的输入形状，RNN / GRU 基础应用 |
| **13** | [`13_循环神经网络_高级.ipynb`](./13_循环神经网络_高级.ipynb) | **[硬核]** 双向 GRU、时间序列动态填充处理 (`pack_padded_sequence`) |

### 📖 附加文档
- [`知识点总结与复习.md`](./知识点总结与复习.md)：记录了学习过程中踩坑的重难点梳理，对于复习各层维度、填坑 (CPU/GPU) 以及参数计算极具价值。

## 🛠️ 使用方法与环境

本项目代码在 Python 3.10 上测试通过。

推荐使用 Conda 建立独立的运行环境：
```bash
conda create -n pytorch-gpu python=3.10
conda activate pytorch-gpu
# 根据个人显卡 CUDA 版本到 PyTorch 官网获取安装命令
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

打开 Jupyter Notebook 直接运行即可：
```bash
jupyter notebook
```
> **注意**：如果不使用 GPU，可以将所有 `USE_GPU = True` 改为 `False` 强行切换到 CPU 跑小批量数据。

## 💡 声明

这不仅是一份用于学习的副本，也是给未来复习自己的礼物。希望对其他同行学习 PyTorch 亦有参考价值！
