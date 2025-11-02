<center>
    <h1>Hessian矩阵系列串讲</h1>
</center>


### 定义

假设有一实值函数$f(x_1,x_2,\dots,x_n)$，若$f$的所有二阶偏导数都存在且在定义域里连续，那么我们定义函数$f$的Hessian矩阵为如下一个$n\times n$的方阵:
$$
\mathbf{H} = \left[
\begin{matrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1\partial x_2} & \dots&\frac{\partial^2 f}{\partial x_1 \partial x_n}\\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} &
\dots& \frac{\partial^2 f}{\partial x_2 \partial x_n}\\
\vdots & \vdots & \ddots & \vdots\\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \dots & \frac{\partial^2 f}{\partial x_n^2}
\end{matrix}
\right]
$$


或使用下标标记表示为:
$$
\mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

### 一般性质

**对称性**:

对于机器学习中遇到的大多数函数(特别是那些具有连续二阶偏导数的函数),混合偏导数的求导顺序无关紧要。这被称为克莱罗定理或施瓦茨定理,它说明了混合偏导数的相等性:
$$
\frac{\partial^2 f}{\partial x_i\partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}
$$
这意味着Hessian矩阵是对称的，即$\mathbf{H} = \mathbf{H}^\top$

**正定性与局部最优值的相关性**

通过Hessian矩阵的正定性，我们可以判断该点是局部最小值，局部最大值还是鞍点

- 正定:若Hessian矩阵$H(x^*)$是正定的(意味着$\forall v \neq 0,v^\top\mathbf{H}v>0$)，则函数在$x^*$处有局部最小值。函数在该点周围向所有方向向上弯曲，类似于碗的底部
- 负定:若Hessian矩阵$H(x^{*})$是负定的(意味着$\forall v \neq 0,v^\top\mathbf{H}v<0$),则函数在$x^*$处有局部最大值。函数在该点周围向所有方向向下弯曲，类似于圆顶的顶部
- 不定:若Hessian矩阵$H(x^{*})$是不定的,则函数在$x^*$处有鞍点。函数在某些方向向上弯曲，在另一些方向向下弯曲，就像马鞍

### Hessian在神经网络中的计算

Hessian矩阵在神经网络计算的许多方面有着重要作用，包括:

- ⼀些⽤来训练神经⽹络的⾮线性最优化算法是基于误差曲⾯的⼆阶性质的，这些性质

  由Hessian矩阵控制(比如牛顿法和拟牛顿法)

- 对于训练数据的微⼩改变，Hessian矩阵构成了快速重新训练前馈⽹络的算法的基础

- Hessian矩阵的逆矩阵⽤来鉴别神经⽹络中最不重要的权值，这是⽹络“剪枝”算法的⼀部分(LeCun的OBD)

对于Hessian矩阵的众多应用而言，一个重要的需要考虑的问题是计算效率。在神经网络中有$W$个参数(包括权值和偏置),那么Hessian矩阵的大小就是$W\times W$,那么计算Hessian矩阵的计算量为$O(W^2)$。这在具有大量参数的神经网络中是难以接受的，因此我们需要进行一些高效的近似。

**对角近似**

这个方法最早由Yan LeCun在OBD剪枝方法中提出

我们只保留Hessian矩阵的对角线元素$H_{i,i}$，把非对角线元素置为0.(这个方法在很大程度可以可以方便求逆)

对于神经网络中的第j个神经元的输入$a_j$对应权重为$w_{ji}$:
$$
\frac{\partial^2 E_n}{\partial w_{ji}^2} = \frac{\partial^2 E_n}{\partial a_j^2}z_i^2
$$
其中$z_i$是上一层神经元的输出。

而$\frac{\partial^2 E_n}{\partial a_j^2}$可以通过链式法则递归计算(类似于反向传播):
$$
\frac{\partial^2 E_n}{\partial a_j^2} = \underbrace{\frac{\partial}{\partial a_j}[h'(a_j)\sum_{k}w_{kj}\frac{\partial E_n}{\partial a_k}]}_{链式法则}=h'(a_j)^2 \sum_{k,k'}w_{kj}w_{k'j}\frac{\partial^2 E_n}{\partial w_k\partial w_{k'}} + h''(a_j)\sum_k w_{kj}\frac{\partial E_n}{\partial a_n}
$$
忽略二阶导中的非对角线项$k\neq k'$:
$$
\frac{\partial^2 E_n}{\partial a_j^2}\approx \underbrace{h'(a_j)^2\sum_k w_{kj}^2 \frac{\partial^2 E_n}{\partial a_k^2}}_{链式法则} + \underbrace{h''(a_j)\sum_{k}w_{kj}\frac{\partial E_n}{\partial a_k}}_{链式法则}
$$
从而一次反向传播便可以计算出来，时间复杂度为$O(W)$



但是如你所见，在这个方法下，Hessian矩阵完全退化为了一个diag，这在神经网络中是不合理的，因为非线性层的引入会让Hessian矩阵的交叉项不为0，更重要的是这种误差会随着网络层数的堆砌不断放大。

因此针对这个问题在对角近似上近年来不断有相关文章发表，下面对我阅读过的一些进行简单介绍:

1. HesScale:在LeCun的方法的骨架上，针对常见网络的最后一层的softmax+CE结构可以通过一个简单的公式求出diag(H)的精确值:
   $$
   \frac{\partial^2 E_n}{\partial^2 a_j^2} = p - p\circ p
   $$
   其中p为$\text{softmax}(a_j)$,$\circ$表示向量的逐元素乘积,为方便书写我们将$\{a_1,\dots,a_n\}$写作z

   证明如下:

   ​	预测$\mathbf{p} = \text{softmax}(z)$,即$p_i = \frac{\exp(z_i)}{\sum_{j = 1}^k \exp(z_j)}$,设目标分布为$t\in \Delta^{K-1},\sum_{i}t_i = 1$，对于单样本而言，其损失可以写作:
   $$
   \mathcal{L}(z) = -\sum_{j=1}^Kt_i\log p_i = -t^\top z + \log \sum_{j=1}^K \exp(z_j)
   $$
   求一阶导我们有:
   $$
   \nabla_z \mathcal{L} = -t + \nabla_z \log \sum_{j=1}^K \exp{z_j} = -t + \mathbf{p}
   $$
   求二阶导相当于$\mathbf{p}$对z求导。我们知道$\text{softmax}$的Jocobian为
   $$
   \frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j),J_{\text{softmax}}(z) = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top 
   $$
   其中$\delta_{i,j}$为Kronecker函数，写成矩阵形式，则只有对角线元素为1，其余为0.

   因此我们有:
   $$
   \nabla^2_z\mathcal{L} = \text{diag}(\mathbf{p}) - \mathbf{pp}^\top
   $$
   于是我们可以得到:
   $$
   \frac{\partial^2 L}{\partial^2 a_j} = p_j(1-p_j)
   $$

2. AdaHessian:

   具体而言，AdaHessian在进行Hessian矩阵的对角近似的时候利用了Hutchinson估计，它通常也被用来对矩阵的迹进行估计:

   ​	对任意矩阵$\mathbf{A}\in\mathbb{R}^{d\times d}$,若随机向量$z = (z_1,\dots,z_d)^\top$满足:
   $$
   \mathbb{E}[z_i] = 0\\
   \mathbb{E}[z_iz_j] = \delta_{ij}
   $$
   ​	那么有:
   $$
   \mathbb{E}[z\odot (Az)] = \text{diag}(A)
   $$
   ​	下面进行证明:

   ​		记估计量$d = z\odot (Az)$的第i个分量:
   $$
   d_i = z_i(Az)_i = z_i\sum_{j=1}^d A_{ij}z_j = \sum_{j = 1}^d A_{ij}z_iz_j\\
   \mathbb{E}[d_i] = \sum_{j=1}^dA_{ij}\mathbb{E}[z_iz_j] = \sum_{j=1}^d A_{ij}\delta_{ij} = A_{ii}
   $$
   ​		由此我们证明了$\mathbb{E}[z\odot(Az)] = \text{diag}(A)$

   在神经网络中，相较于直接计算完整的Hessian矩阵以及其OBD方式的对角近似，计算其矩阵向量积(HVP,Hessian Vector product)是并不困难的，它只需要一次反向传播：
   $$
   Hz = \frac{\partial (g^\top z)}{\partial \theta} = \frac{\partial g^\top}{\partial \theta}z + g^\top \frac{\partial z}{\partial \theta} = \frac{\partial g^\top}{\partial \theta}z
   $$
   由此，我们可以通过多次在满足Rademacher分布的向量取样计算$z\odot (Az)$的期望就能得到$\text{diag}(A)$的**无偏估计**，在实际的应用中，只进行一次取样就能得到较为不错的结果。

**外积近似**

在神经网络应用于回归问题时，通常采用下面形式的平方和误差
$$
E = \frac{1}{2}\sum_{n=1}^N(y_n-t_n)^2
$$
​	那么Hessian矩阵可以写成如下形式:
$$
H = \nabla\nabla E = \nabla(\sum_{n=1}^N (y_n-t_n)\nabla y_n) = \sum_{n = 1}^N \nabla y_n(\nabla y_n)^\top + \sum_{n=1}^N(y_n-t)\nabla\nabla y_n
$$
​	在网络已经训练好的情况下，输出$y_n$与$t_n$接近，因此第二项可以忽略，由此我们得到了Hessian矩阵的外积近似
$$
H\approx \sum_{n=1}^N b_n b_n^\top
$$
​	其中，$b_n = \nabla y_n = \nabla a_n$(输出单元的激活函数就是恒等函数)。这种方法中的Hessian矩阵可以跟随反向传播算法在$O(W)$个步骤内高效地求出误差函数地一阶导数。再通过简单地乘法就可以在$O(W^2)$步骤内求出矩阵元素。

**Hessian矩阵逆的计算**

使用外积近似，我们可以提出一个计算Hessian矩阵的逆的高效办法,首先我们有:
$$
\mathbf{H}_N = \sum_{i=1}^n b_n b_n^\top
$$
其中，$b_n = \nabla w a_n$时数据点n产生的输出单元激活对梯度的贡献。我们现在推导一个建立Hessian矩阵的顺序步骤，每次处理一个数据点。假设我们已经使用了前L个数据点得到了Hessian矩阵的逆。通过将第L+1个数据点的贡献单独写出来，我们有:
$$
\mathbf{H}_{L+1} = \mathbf{H}_L + b_{L+1}b_{L+1}^\top
$$
为了计算Hessian矩阵的逆，我们考虑下面这个矩阵的恒等式:
$$
(M + vv^\top)^{-1} = M^{-1} - \frac{(M^{-1}v)(v^\top M^{-1})}{1+v^\top M^{-1}v}
$$
若我们令$H_L = M$,且$b_{L+1} = v$,我们有:
$$
H_{L+1}^{-1} = H_{L}^{-1} - \frac{H_L^{-1}b_{L+1}b_{L+1}^\top H_{L}^{-1}}{1 + b_{L+1}^\top H_L^{-1}b_{L+1}}
$$
通过这种方式，数据点可以依次使用，直到L+1=N,整个数据集处理完毕。于是，这个结果表示一个计算Hessian矩阵的逆的算法。这个算法只需对数据集扫描一次。最开始的矩阵$\mathbf{H_0}$被选为$\alpha I$,其中$\alpha$是一个较小的量，从而算法实际找的是$\mathbf{H} + \alpha I$的逆。结果对于$\alpha$的精确值不敏感。

​	**Hessian矩阵的逆计算的K-FAC分解方法**：

​	在特定场景下，Hessian矩阵的逆有一个利用Kronecker分解的近似方法，它利用了在loss为log-like形式下$\mathbb{E}[H] = F$的特点，其中$F$为Fisher信息矩阵:
$$
F = \mathbb{E}[\nabla \log p(x)\nabla \log p(x)^\top]
$$
其中上式中做了如下形式的简写:
$$
\nabla \log{p(x)} = \nabla_\theta \log{(p|\theta)}\\
\mathbb{E}[p(x)] = E_{x\sim p(x|\theta)}
$$
首先我们有:
$$
\mathbb{E}[\nabla \log p(x)] = \int (\nabla \log p(x))p(x) dx\\
=\int \frac{\nabla \log p(x)}{p(x)}p(x) dx\\
=\int \nabla \log p(x)dx\\
=\nabla 1 = 0
$$
那么当loss为log-like形式下时:
$$
L = -\log p(x)
$$
我们求它的Hessian:
$$
\nabla^2 L = -\nabla^2 \log p(x)\\
=-\nabla \frac{\nabla p(x)}{p(x)}\\
=\frac{\nabla p(x)}{p(x)}^\top \frac{\nabla P(x)}{p(x)} - \frac{P(x)^2}{p(x)}\\
=\nabla \log p(x)^\top \nabla \log p(x) - \frac{\nabla^2 P(x)}{p(x)}
$$
对该式求期望，我们有:
$$
\mathbb{E}[H]=\mathbb{E}[\nabla \log p(x)^\top \nabla \log p(x)] - \mathbb{E}[\frac{\nabla^2 p(x)}{p(x)}] = F - \int \frac{\nabla^2 p(x)}{p(x)} p(x) dx\\
=F - \int \nabla^2 p(x)dx\\
=F - \nabla^2 \int p(x)\\
=F
$$
由于$\theta$本质为所有层$\mathbf{W}$的拼接，我们有:
$$
d\theta = \nabla_{\theta}L(x)\\
F = \mathbb{E}[\nabla L(x)\nabla L(x)^\top] = \mathbb{E}[d\theta d\theta^\top]\\
\theta = [\text{vec}(W_0)^\top,\text{vec}(W_1)^\top,\dots,\text{vec}(W_n)^\top]^\top
$$
带入展开得到:
$$
F_{ij} = \mathbb{E}[\text{vec}(dW_i)\text{vec}(dW_j)^\top]
$$
令$a_i,g_i$分别为第i层的前向输入和反向传播梯度，由反向传播算法有:
$$
d W_i = g_ia_i^\top
$$
带入有:
$$
\text{vec}(dW_i) = \text{vec}(g_ia_i^\top) = a_i\otimes g_i
$$
这里可能不太直观，这是因为平常对kronecker积($\otimes$)的接触较少，具体而言，我们这样定义Kronecker积:
$$
A\otimes B = \left[
\begin{matrix}
a_{11}B &\dots &a_{1n}B\\
\vdots & \ddots& \vdots\\
a_{m1}B & \dots & a_{mn}B
\end{matrix}
\right]
$$
在这里，由于我们将$dW_i$即$\nabla W_i$第i层参数的梯度进行了向量化，原本的$\nabla W_i$ 是如下形式:
$$
\nabla W_i = g_i a_{i}^\top = \left[\begin{matrix}g_1a_1 & \dots &g_1a_n\\ \vdots & \ddots & \vdots\\ g_ma_1 &\dots&g_ma_n\end{matrix} \right]
$$
按列堆叠(vec)就有:
$$
\text{vec}(dW_i) = \left[
\begin{matrix}
g_1a_1\\
\vdots\\
g_1a_n\\
\vdots\\
g_ma_n
\end{matrix}
\right] = a_i\otimes g_i
$$
那么我们可以将Fisher矩阵写作如下形式:
$$
F_{ij} = \mathbb{E}[\text{vec}(dW_i)\text{vec}(dW_j)]\\
=\mathbb{E}[(a_i\otimes g_i)(a_j\otimes g_j)^\top]\\
=\mathbb{E}[(a_ia_j^\top)\otimes (g_ig_j^\top)]\\
\approx \mathbb{E}[a_ia_j^\top]\otimes\mathbb{E}[g_ig_j^\top]
$$
这个近似相当于我们忽略了$\text{Cov}_\otimes(a_ia_j^\top,g_ig_j^\top)$,这实际上是合理的，尤其是在网络较深的情况下。这里给出一个[链接](https://truenobility303.github.io/KFAC/)提供一个比较详细的说明。	

虽然直接求$F_{ij}$的复杂度仍然不变，但是利用Kronecker积的性质，我们在求逆时可以得到较大的性能提升:
$$
(A\otimes B)^{-1} = A^{-1}\otimes G^{-1}
$$
并且在实际运算中，并不是整个网络的Fisher进行计算，而是按层做块对角运算:
$$
\mathbf{F} = \text{blockdiag}(F_1,F_2,\dots),F_l\approx A_l\otimes G_l
$$
这样每层求逆的时候只需要求两个小矩阵的逆。

**一些统计量的计算**：

​	除了直接对Hessian矩阵的近似计算，我们有时候仅仅需要对Hessian矩阵的统计量进行计算，例如矩阵的迹，最大特征值等。

​	在计算这些统计量时，我们需要一个重要的算子:Hv,即Hessian矩阵与任意向量的乘积，这个乘积我们在上面已经证明了通过一次简单的反向传播算法可以得到。(PyHessian是一个Python库，它实现了这个算子的高效计算)

- 顶部若干特征值

  利用幂迭代法 + HVP可以高效地求解特征值。

  简单来说幂迭代法的流程很简单:

  给定矩阵A以及初始非零向量$x_0$,

  迭代:
  $$
  y_{k+1} = Ax_k\\
  x_{k+1} = \frac{y_{k+1}}{||y_{k+1}||}\\
  \mu_{k+1} = \frac{x_{k+1}Ax_{k+1}^\top}{x_{k+1}x_{k+1}^\top}
  $$
  具体的收敛性证明见[教材](https://link.zhihu.com/?target=https%3A//ergodic.ugr.es/cphys/lecciones/fortran/power_method.pdf)

- 矩阵的迹

  在上面近似计算中，我们曾用Hutchinson估计对Hessian矩阵进行了对角近似，事实上，我们还可以进行迹的估计。

  简单来说，方法一样，从Rademacher分布中取一随机向量$v$,然后有恒等式:
  $$
  \text{Tr}(H) = \text{Tr}(HI) = \text{Tr}(H\mathbb{E}[vv^\top]) = \mathbb{E}[\text{Tr}(Hvv^\top)] = \mathbb{E}[v^\top H v]
  $$

  ### Hessian矩阵在神经网络下的特殊结构

 	本小节内容主要基于[Towards Quantifying the Hessian Structure of Neural Networks](https://arxiv.org/pdf/2505.02809),B站上有作者的讲解视频[FAI\] 港中深 张雨舜 | 浅谈神经网络Hessian矩阵的特殊结构](https://www.bilibili.com/video/BV1To3TzmEX3/?spm_id_from=333.1387.homepage.video_card.click&vd_source=76e54ba50c020fb612c90d28c211c638)

​	这篇文章主要说明了在神经网络中，Hessian矩阵往往具有近块对角结构，这表明曲率信息(二阶信息)主要在层内耦合，层与层的二阶交互很弱。这间接的说明了Layer-wise的量化/剪枝的合理性。

​	此外，文章还进一步从理论和实验上阐释了造成这个现象的原因：

1. 静态力量:即使在随机初始化阶段，即在训练开始之前，神经网络的Hessian矩阵就已经呈现出近块对角结构。这种结构的形成与网络的架构设计有关，因此被称为“静态力量”。具体来说，对于线性模型和单隐藏层网络，无论是使用均方误差（MSE）损失还是交叉熵（CE）损失，Hessian矩阵的对角块和非对角块在随机初始化时就已经表现出明显的差异。
2. 动态力量:在训练过程中，Hessian矩阵的结构会进一步发生变化。特别是在使用CE损失时，训练过程会逐渐消除初始时存在于跨层Hessian分量（Hwv）中的“块循环”（block-circulant）模式，而对角块和非对角块的近块对角结构则保持稳定。这种由训练过程引起的结构变化被称为“动态力量”。

另外一点，文章证明了类别数C是影响Hessian矩阵结构的主要因素。(在实验中，隐藏层Hessian的非对角块与对角块的比值以$\frac{1}{\sqrt{C}}$的速度衰减，输出层Hessian的衰减速率为$\frac{1}{C}$).这个结果对于大模型而言是友好的，因为在神经网络中C的大小往往是$1e3\sim 1e4$级别的，这说明大模型的Hessian是具有强对角块的结构的！这一点实际上在传统的方法上人们意识or无意识的用到了(对角近似)，但是更丰富层面以及基于这个发现的在计算的可行性和性能的Trade-off做的工作是比较少的，在优化器那边做的比较多。

此外还有不少文献揭露了Hessian矩阵具有低秩特征谱，即只有少数的特征值显著大，其余大多接近0.



