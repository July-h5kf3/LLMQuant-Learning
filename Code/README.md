<center>
    <h1>代码实践</h1>
</center>

本部分文件夹为选题代码实践文件夹，包括了一些论文复现以及学期中的Pytorch代码复现以及Jittor框架代码复现内容

### 论文复现

其中包括两部分，其中一部分参考[仓库](https://github.com/Ranking666/Base-quantization)实现，采用同样的网络基线，基于MNIST数据集，时使用VGG网络作为模型进行各种量化方法的集成。另一部分则是一些较大模型的量化方法的复现，以及学期任务。



### Base-Quantization

目前采用VGG网络作为模型进行各种量化方法的集成。



目前支持更换Per_tensor,Per_channal量化，支持基于Minmax,adaround的PTQ量化方法，支持基于Minmax的QAT方法。



后续将会有更多的模型作为量化对象，更多的量化方法被实现~



10.11:框架搭建

10.12:实现AdaRound



|      Models      |  VGGS  |
| :--------------: | :----: |
|     **FP32**     | 99.43% |
| **PTQ_adaround** | 99.41% |

### 学期作业

To Be Continue~