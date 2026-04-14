## Motivation

最近在完成CS336的Assignment1，从头搭建一个大模型，其中实现使用的是主流的旋转位置编码(RoPE)。虽然对照指导书完成了相关实验，但是对位置编码的选择以及原理存在一定的知识缺漏，因此写本博客以加强理解。

## Note

我们在下面的数学推到中使用**列向量推导**

## 为什么Transformer需要位置编码?

设输入序列长度为L，模型的维度为d，那么我们输入的序列第i个token的表示为$x_i \in \mathbb{R}^{d}$,将整个序列拼接起来我们有:
$$
X = [x_1,x_2,\dots,x_L]\in \mathbb{R}^{d\times L}
$$
那么Transformer Block中的MHA(由于最终结果是每一个Head通过相同的过程得到的不同结果进行concat，因此我们这里只考虑一个Head)在计算Attention Score时首先通过线性映射得到:
$$
Q = W_Q X \quad K = W_KX\quad V = W_V X
$$
其中:
$$
W_Q,W_K,W_V \in \mathbb{R}^{d\times d}
$$
因此:
$$
Q,K,V \in \mathbb{R}^{d\times L}
$$


接下来会计算第i个query和第j个key的相似度:
$$
s_{ij} = \frac{q_i^\top k_j}{\sqrt{d}}
$$
写成矩阵形式就是:
$$
S = \frac{Q^\top K}{\sqrt{d}}\in \mathbb{R}^{L\times L}
$$
然后进行Softmax操作(沿key维度):
$$
A = \text{softmax}(S)\in \mathbb{R}^{L\times L}
$$
最后进行加权求和:
$$
\text{Attention(Q,K,V)} = VA^\top \in \mathbb{R}^{d\times L}
$$
令$P \in \mathbb{R}^{L\times L}$为置换矩阵，表示对序列顺序的重排。我们有:
$$
X' = XP
$$
那么我们重新计算Attention:
$$
Q' = W_Q X' = W_QXP = QP \quad K' = W_KX'=KP \quad V' = VP
$$

$$
S' = \frac{Q'^\top K'}{\sqrt{d}} = \frac{P^\top Q^\top K P}{\sqrt{d}}
$$

$$
A' = \text{softmax}(S') = P^\top AP
$$

$$
\text{Attention}(Q',K',V') = VPA'^\top = VPP^\top A^\top P = VA^\top P = \text{Attention}(Q,K,V)P
$$



上式表明，在不引入任何位置信息的情况下，Attention计算具有置换同变性。但是我们需要注意到，P是一个**置换矩阵**，它只是索引的重定向，我们计算出来的Attention的数值没有发生任何变化。如果我们把Attention输出结果看作带有语义信息的Embedding向量，那么它的语义将不会携带位置语义的信息。

举一个具体的例子:

我们输入的文本序列是`["猫","吃","鱼"]`，对应的输入序列为`[x_1,x_2,x_3]`,而对应的Attention输出为`[y1,y2,y3]`,而我们进行交换得到:`["鱼","吃","猫"]`，对应的输入序列为`[x_3,x_2,x_1]`,根据我们上面的证明,Attention的输出将是`[y3,y2,y1]`，在数值上没有任何变换，只是简单的位置交换，也就是说Attention的输出没有提取出在语义结构(如主谓宾等)的信息。

## 位置编码为什么可以解决这个问题？

我们现在需要解决的问题是打破这种数值不变性，也就是交换了输入位置后，我们希望最终的输出为`[y3',y2',y1']`.我们将Attention视为一个输出为集合的函数f,那么上述不变性用数学语言表述就是:
$$
f(\dots,x_m,\dots,x_n,\dots) = f(\dots,x_n,\dots,x_m,\dots)
$$
因此，我们要做的事情就是打破这种不变性，比如在每个位置都加上一个不同的编码向量:
$$
\hat f(\dots,x_m,\dots,x_n,\dots) = f(\dots,x_m+p_m,\dots,x_n+p_n,\dots)
$$
一般来说，只要每个位置的编码向量不同，那么这种全对称性就被打破了，即可以用f~代替f来处理有序的输入。

我们写作矩阵的形式:

我们将位置编码矩阵定义为$E = [e_1,e_2,\dots,e_L]\in \mathbb{R}^{d\times L}$。每一个列向量$e_i$仅与位置索引i有关。

融入后的输入矩阵为:
$$
X_{pos} = X + E
$$
此时我们重新推导Q,K,V的生成过程(以Q为例):
$$
Q = W_Q(X + E) = W_Q X + W_Q E = Q_X + Q_E
$$
那么同理:
$$
K = K_X + K_E,V = V_X + V_E
$$
我们可以将$\{K,Q,V\}_{E}$视为位置信息的“特征表达”

接下来我们计算$S = \frac{Q^\top K}{\sqrt{d}}$,我们具体来看其中一个点积项$s_{ij}$:
$$
s_{ij} = \frac{1}{\sqrt{d}}(q_{x,i}+q_{e,i})^\top(k_{x,j}+k_{e,j})\\
=\frac{1}{\sqrt{d}}(q^\top_{x,i}k_{x,j}+q^\top_{x,i}k_{e,j}+q^\top_{e,i}k_{x,j}+q^\top_{e,i}k_{e,j})
$$
那么此时就多出了位置-内容，位置-位置的信息。

我们再按照同样的推理,令$X' = XP$,那么此时:
$$
Q' = W_Q(X'+E) = W_Q(XP+E) = Q_XP+Q_E
$$
同理:
$$
K' = K_XP+K_E \quad V' = V_XP+V_E
$$
那么:
$$
S' = \frac{Q'^\top K'}{\sqrt{d}} = \frac{(Q_XP+Q_E)^\top(K_XP+K_E)}{\sqrt{d}}\\
=\frac{1}{\sqrt{d}}(P^\top Q_X^\top K_XP + P^\top Q_X^\top K_E + Q_E^\top K_XP+Q_E^\top K_E)
$$
显而易见,$S' \neq P^\top SP$,且其中的$Q_E^\top K_E$项没有被P作用，也就意味着原本的置换对称性也就被打破了，因此我们可以用$\hat f $来代替f来处理有序的输入。

也就是说，位置编码的引入可以解决我们说的Attention置换数值不变性的问题。

## 怎样的位置编码是好的？

我们现在想要进一步分析位置编码的性质，从而设计更好的位置编码。我们将$\hat f$展开至二阶项(为了简化考虑，写作矩阵形式):
$$
\hat f(X) =f(X+E) \approx f(X) + \nabla f(X)\cdot E+\frac{1}{2}E^\top H_f(X)E
$$
那么我们来看与位置编码有关的项:

- 一阶项$\nabla f(X)\cdot E = \sum_{i=1}^L e^\top_i \frac{\partial f}{\partial x_i}$:这项依赖于单一位置，所以是绝对位置信息
- 二阶项$\frac{1}{2}E^\top H_{f(X)}E = \sum_{i=1}^L\sum_{j=1}^L e_i^\top \frac{\partial^2 f}{\partial{x_i}\partial x_j}e_j$:这项与包含了任意两个位置的交互，所以是相对位置信息

对于一阶项，只要位置编码在每个位置都是独特的就能表述绝对位置的信息。而对于二阶信息，对于一个理想的位置编码而言，应该让这个二阶项满足某种平移不变性，即对于任意位移k，位置i与位置i+k的交互模式应该是稳定的。

我们先从最简单的情况入手，假设$\mathbf{H} = I$为单位矩阵，那么此时$E^\top E$是两个位置编码的内积，我们希望在这个简单的例子中该项表达的是相对位置信息，即存在某个函数g使得:
$$
<p_m,p_n> = g(m-n)
$$
这里的$p_m,p_n$为d维向量，这里我们从最简单的$d = 2$入手。我们称上式是一个位置编码为一个合理位置编码的条件式。

对于2维向量，我们借助复数来推导，视向量[x,y]为复数$x+yi$,那么我们有:
$$
<p_m,p_n>=a_xb_x+a_yb_y = \text{Re}(p_m\hat p_n)
$$
其中$\hat w$为$w$的共轭复数。

为了满足上式，我们可以假设存在复数$q_{m-n}$,使得:
$$
p_m\hat p_n = q_{m-n}
$$
这样两边取实部就得到条件式。为了解这个方程，我们可以使用复数的指数形式，假设$p_m = r_me^{i\phi_m},\hat p_n = r_ne^{-i\phi_n},q_{m-n} = R_{m-n}e^{i\Phi_{m-n}}$,那么有:
$$
r_mr_ne^{i(\phi_m-\phi_n)}=R_{m-n}e^{i\Phi_{m-n}}
$$
于是我们得到等式:
$$
\left\{
\begin{array}{l}
r_m r_n = R_{m-n} \\
\phi_m - \phi_n = \Phi_{m-n}
\end{array}
\right.
$$
对于第一个方程，带入m = n，可以得到$r_m^2 = R_0$,即$r_m$为一个常数，为了简单，我们令其为1；

对于第二个方程，显然等差数列满足上述性质(令m = 0有$\Phi_m = \phi_m$)，设公差为$\theta$，则通项为$\phi_m = \Phi_m = m\theta$,由此我们得到二维情况下的位置编码的解:
$$
p_m = e^{im\theta} \to p_m = \left(\begin{align}\cos m\theta\newline \sin m\theta\end{align}\right)
$$
由于内积满足线性叠加性，我们可以由二维的情况直接扩展到更高偶数维的情况:
$$
p_m = \left(\begin{matrix}e^{im\theta_0}\\e^{im\theta_1}\\\vdots\\e^{im\theta_{d/2-1}}\end{matrix}\right) \to p_m = \left(\begin{matrix}
\cos m\theta_0 \\
\sin m\theta_0 \\
\cos m\theta_1 \\
\sin m\theta_1 \\
\vdots\\
\cos m\theta_{d/2-1} \\
\sin m\theta_{d/2-1} \\
\end{matrix}\right)
$$
这样我们就求出了满足条件式的一组解，显然解不唯一。

此外，一个好的位置编码应该满足远程衰减的性质，即随着$|m-n|$的增大,$<p_m,p_n>$有趋于0的趋势。

那么有:
$$
<p_m,p_n> = \text{Re}[e^{i(m-n)\theta_0} + e^{i(m-n)\theta_1}+\dots+e^{i(m-n)\theta_{d/2-1}}]\\
=\sum_{j=0}^{d/2-1}\cos (k\theta_j) \quad k = m-n
$$
由于在LM中d通常为768，是一个较大值，因此我们可以将离散的索引j映射到连续变量$t\in[0,1]$上。设$\theta_j$是某个光滑单调函数$f(t)$生成的，即$\theta_j = f(2j/d)$。利用Euler-Maclaurin的一阶近似，我们可以将求和转化为积分:
$$
<p_m,p_n> \approx \frac{d}{2}\int_0^1 \cos(k\cdot f(t))dt
$$
那么现在的问题就转化为了，寻找一个函数$f(t)$,使得上述震荡积分在k很大时具有较好的衰减性质(足够快)。

根据黎曼-勒贝格引理，只要频率分布函数f(t)满足光滑且严格单调的条件，当$k\to \inf$时，被积函数的高频震荡将导致正负面积互相抵消，使得积分值必然趋近于0.

在Transformer2017的论文中的Sinusoidal位置编码选择的是$\theta_t = 10000^{-t}$.

由此，我们便推导出了Sinusoidal位置编码的形式:
$$
\left\{
\begin{array}
p_{k,2i} = \sin(k/10000^{2i/d})\\
p_{k,2i+1} = \cos(k / 10000^{2i/d})
\end{array}
\right.
$$
我们只能说明它的合理性，但是无法说明它的最优性，因为它并不一定最优[Lol]

事实上，一个可行的方案是将位置编码中的$\theta_i$设为可学习的参数，其初始值为$\theta_i = 10000^{-2i/d}$.



在上述推导中，都是基于H = I这个简单情况，对于一般的H，使用上述Sinusoidal位置编码，还能具备我们理想的性质吗？

事实上，有研究表明$^{[3]}$ 在网络规模足够大的情况下，其Hessian矩阵将呈现出块对角的形式，因此我们考虑H是一个对角阵的情况，此时:
$$
p_m^\top H p_n = \sum_{i=1}^{d/2}H_{2i,2i}\cos m\theta_i \cos n\theta_i+H_{2i+1,2i+1}\sin m\theta_i\sin n\theta_i
$$
由和差化积有:
$$
\sum_{i=1}^{d/2}\frac{1}{2}(H_{2i,2i}+H_{2i+1,2i+1})\cos(m-n)\theta_i + \frac{1}{2}(H_{2i,2i}-H_{2i+1,2i+1})\cos(m+n)\theta_i
$$
可以看到其中是包含了相对位置项(m-n)的，只是会出现m+n项。

因此我们可以认为Sinusoidal位置编码是一个有效的位置编码。

## 为什么RoPE$^{[4]}$比Sinusoidal位置编码更好？

在先前分析为什么需要位置编码的过程中，我们知道位置编码的作用是在计算Attention Score时，让QK点积中包含文本的结构语义信息，具体而言是下式中的后三项包含位置信息:
$$
S' = \frac{Q'^\top K'}{\sqrt{d}} = \frac{(Q_XP+Q_E)^\top(K_XP+K_E)}{\sqrt{d}}\\=\frac{1}{\sqrt{d}}(P^\top Q_X^\top K_XP + P^\top Q_X^\top K_E + Q_E^\top K_XP+Q_E^\top K_E)
$$
我们可以发现，除了最后一项是相对位置信息外，另外两项则是绝对位置信息与文本语义信息的耦合，这实际上是一种冗余，因为绝对位置信息的作用相较于相对位置信息而言一方面具有误导性，例如在短文本中结尾出现在绝对位置10中，而10在长文本中可能仅仅是文本开头；另一方面模型还会额外去学习已经得到的相对位置信息造成资源的浪费.

为了解决这个问题，工业界进行了许多尝试$^{[5]}$,但是这些尝试普遍会带来大量的额外计算存储开销。由于绝对位置编码具有实现简单，计算速度快的特点，并且在Sinusoidal位置编码中我们也能看到通过绝对位置编码在一定程度上是可以得到相对位置信息的,如果可以通过绝对位置编码的方式实现相对位置编码，那么就是“集各家之所长”，“鱼和熊掌兼得”了。

为了实现这个目标，我们假设通过下述运算来给q,k添加绝对位置信息:
$$
\hat q_m = f(q,m)\quad \hat k_n = f(k,n)
$$
而我们希望得到如下恒等关系:
$$
<f(q,m),f(k,n)> = g(q,k,m-n)
$$
为了求解方便，我们令:
$$
f(q,0) = q \quad f(k,0) = k
$$
求解思路与之前推导Sinusoidal类似，我们先考虑二维的情况，然后借助复数来求解。
$$
\text{Re}[f(q,m)\hat f(k,n)] = g(q,k,m-n)
$$
设:
$$
\begin{array}
f(q,m) &=& R_f(q,m)e^{i\theta_f(q,m)}\\
\hat f(k,n) &=& R_f(k,n)e^{-i\theta_f(k,n)}\\
g(q,k,m-n) &=& R_g(q,k,m-n)e^{i\theta_g(q,k,m-n)}
\end{array}
$$
那么带入方程求解得到:
$$
\left\{ 
\begin{array}
R_f(q,m)R_f(k,n) = R_g(q,k,m-n)\\
\theta_f(q,m) - \theta_f(k,n) = \theta_g(q,k,m-n)
\end{array}
\right .
$$
对于第一个方程，令m = n有:
$$
R_f(q,m)R_f(k,n) = R_g(q,k,0) = R_f(q,0)R_f(k,0)=||q||||k||
$$
那么我们可以直接令$R_f(q,m) = ||q||$,即它不依赖于m。

对于第二个方程，同样地令m=n有:
$$
\theta_f(q,m)-\theta_f(k,n)=\theta_g(q,k,0)=\theta_f(q,0)-\theta_f(k,0) = \theta_q-\theta_k
$$
这里的$\theta_q,\theta_k$是q,k本身的辐角。

由此我们得到:
$$
\theta_f(q,m) - \theta_q = \theta_f(k,n) - \theta_k
$$
因此$\theta_f(q,m)-\theta_q$应该是一个只与m有关的而与q无关的函数，因为我们希望恒等式对任意的q,k都成立，所以右侧必须与q,k无关，因此记为常数，记为$\theta$，该式记为$\phi(m)$,即
$$
\theta_f(q,m) = \theta_q + \phi(m)
$$
令n = m - 1 有:
$$
\phi(m) - \phi(m-1) = \theta_g(q,k,1)+\theta_k-\theta_q
$$
故$\{\phi(m)\}$为等差数列，公差为$\theta$,得到$\phi(m) = m\theta$

由此，我们得到了二维情况下用复数表示的RoPE:
$$
f(q,m) = R_f(q,m)e^{i\theta_f(q,m)} = ||q||e^{i(\theta_q+m\theta)} = \vec{q}e^{im\theta}
$$
根据复数乘法的集合意义，该变换实际上对应着向量的旋转，我们可以将其写为矩阵形式:
$$
f(q,m) = \left(
\begin{matrix}
\cos m\theta \quad -\sin m\theta\\
\sin m \theta \quad \cos m\theta
\end{matrix}
\right)
\left(
\begin{matrix}
q_0\\
q_1
\end{matrix}
\right)
$$
由内积的线性叠加性，我们可以得到任意偶数维度的RoPE:
$$
\left(
\begin{matrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \dots & 0 & 0 & \\
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \dots & 0 & 0 & \\
0	&	0 &	\cos m\theta_1	&	-\sin m\theta_1	& \dots & 0& 0\\
\vdots & \vdots & \vdots & \vdots	& \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \dots &\cos m\theta_{d/2-1} & -\sin m \theta_{d/2-1}\\
0 & 0 & 0 & 0 & \dots & \sin m\theta_{d/2-1} & \cos m \theta_{d/2-1}
\end{matrix}
\right)
\left(
\begin{matrix}
q_0\\q_1\\q_2\\q_3\\ \vdots\\q_{d-2}\\q_{d-1}
\end{matrix}
\right)
$$
我们称左侧的矩阵为旋转矩阵记为$R_m$.也就是说给位置为m的向量q乘上矩阵$R_m$,位置为n的向量k乘上矩阵$R_n$,用变换后的Q，K序列做Attention，那么Attention就自动包含相对位置信息了，因为恒等式成立:
$$
(R_mq)^\top(R_nk) = q^\top R_m^\top R_nk = q^\top R_{n-m}k
$$
值得指出的是，Rm是一个正交矩阵，它不会改变向量的模长，因此通常来说它不会改变原模型的稳定性。

此外，在具体实现时，我们并不会拿这个大矩阵去乘以向量，而是采用逐位相乘的方式:
$$
\left(
\begin{matrix}
q_0\\q_1\\q_2\\q_3\\ \vdots\\q_{d-2}\\q_{d-1}
\end{matrix}
\right)\otimes
\left(
\begin{matrix}
\cos m\theta_0\\\cos m\theta_0 \\\cos m\theta_1 \\ \cos m\theta_1\\ \vdots \\ \cos m\theta_{d/2-1} \\ \cos m\theta_{d/2-1}
\end{matrix}
\right)+
\left(
\begin{matrix}
-q_1\\q_0\\-q_3\\q_2\\ \vdots\\-q_{d-1}\\q_{d-2}
\end{matrix}
\right)\otimes
\left(
\begin{matrix}
\sin m\theta_0\\\sin m\theta_0\\\sin m\theta_1\\\sin m\theta_1\\ \vdots\\\sin m\theta_{d/2-1}\\\sin m\theta_{d/2-1}
\end{matrix}
\right)
$$
对于$\theta$的选择，作者选择了与Sinusoidal位置编码一样的$\theta_i = 10000^{-2i/d}$,从而带来远程衰减性。

至此，我们完成了对RoPE的推导，并通过推导成功说明了为何RoPE相较于Sinusoidal位置编码更好，因为它通过绝对位置的注入方式，实现了相对位置的注入，而不带来其它冗余。

最后，下面是我在CS336中实现的一个RoPE:

```
from einops import rearrange, einsum
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta,d_k,max_seq_len,device):
        """
        d_k: int, 维度大小，必须为偶数
        theta: float, RoPE中的\Theta值
        max_seq_len: int, 最大序列长度
        device: torch.device, 设备
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        #一共有d / 2个频率
        half_dk = d_k // 2
        k = torch.arange(0,half_dk,device=device).float()
        inv_freq = 1.0 / (self.theta ** (2.0 * k / d_k))

        positions = torch.arange(0,max_seq_len,device = device).float()

        angles = einsum(positions,inv_freq,"max_seq_len,half_dk->max_seq_len half_dk")
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos",cos,persistent = False)
        self.register_buffer("sin",sin,persistent = False)

    def forward(self,x,token_positions):
        """
        inputs:
            x: ...,seq_len,d_k
            token_positions:...,seq_len
        returns:
            x_rotated: ...,seq_len,d_k
        """
        cos = self.cos[token_positions]  # ...,seq_len,half_dk
        sin = self.sin[token_positions]  # ...,seq_len,half_dk

        x_even = x[...,0::2]
        x_odd = x[...,1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[...,0::2] = x_rot_even
        out[...,1::2] = x_rot_odd
        return out

```



## 参考:

1. [Transformer升级之路：1、Sinusoidal位置编码追根溯源 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/8231)
2. [Transformer升级之路：2、博采众长的旋转式位置编码 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/8265)
3. [[2505.02809\] Towards Quantifying the Hessian Structure of Neural Networks](https://arxiv.org/abs/2505.02809)
4. [RoFormer: Enhanced Transformer with Rotary Position Embedding | Cool Papers - Immersive Paper Discovery](https://papers.cool/arxiv/2104.09864)
5. [让研究人员绞尽脑汁的Transformer位置编码 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/8130)