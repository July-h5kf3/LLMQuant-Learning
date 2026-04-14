## Motivation

CS336课程中在介绍实现FlashAttention2的时候介绍了Triton，我发现这是一个比CUDA更加方便的GPU编程语言。

另一方面许多大厂高性能计算岗位包括但不限于算子开发都逐渐趋向于使用Triton，因此目前看来学习Triton是一件从投资未来的角度来看性价比很高的事情

## Introduction

Triton 是 OpenAI 开发的一种面向深度学习与高性能计算的编程语言及编译器，旨在简化 GPU 高性能算子的开发过程，并帮助开发者更容易地写出高效代码。

Triton的[官方文档](https://triton-lang.cn/main/programming-guide/chapter-1/introduction.html)指出,其项目动机来自于基于分块算法的编程范式可以促进构建用于神经网络的高性能计算内核。与传统 CUDA 编程更强调线程级的执行方式不同，Triton 提供了更高层次的块级抽象，开发者可以直接围绕数据块来组织计算，而不必过多关注底层线程调度细节。

我们可以通过矩阵乘法的例子来更加直观的理解。

CUDA编程模型:

```
#pragma parallel
for(int m = 0; m < M; m++)
#pragma parallel
for(int n = 0; n < N; n++){
  float acc = 0;
  for(int k = 0; k < K; k++)
    acc += A[m, k] * B[k, n];

  C[m, n] = acc;
}
```

Triton编程模型:

```
#pragma parallel
for(int m = 0; m < M; m += MB)
#pragma parallel
for(int n = 0; n < N; n += NB){
  float acc[MB, NB] = 0;
  for(int k = 0; k < K; k += KB)
    acc +=  A[m:m+MB, k:k+KB]
          @ B[k:k+KB, n:n+NB];
  C[m:m+MB, n:n+NB] = acc;
}
```

## GPU 

在具体介绍Triton之前，我们先简单了解一下GPU相关的知识，以便减少后续专业术语带来的知识bias。

以下内容来自于本人于南开大学计算机体系结构期末突击的笔记。

相较于CPU而言，GPU的设计思路可以概括为三点:

- **简化流水线，增加核数**：具体而言，对流水线进行瘦身，去掉乱序执行，分支预测等复杂逻辑，节省的空间用来增加大量的计算核心，从而能够同时处理大量数据
- **多个计算单元共用一条指令**：GPU 擅长处理“很多数据做同一种计算”的场景，例如对向量中的每个元素做相同变换。为此，它通常让一组线程(这样的一组线程称为Warp)按照相同的程序逻辑并行执行，只是每个线程处理的数据不同。这种执行方式通常称为 SIMT。
- **驻留大量线程**：GPU 会同时准备很多线程。当一部分线程因为访存而需要等待时，硬件可以立即切换去执行另一部分线程，从而减少计算单元空闲的时间。

我们接下来以数组加法来介绍GPU编程模型中的几个基本概念，这些概念有助于理解后续的并行执行，分块计算和访存优化。

假设我们要计算数组加法:

```
for i in range(1024):
	C[i] = A[i] + B[i]
```

那么一个GPU Kernel会以大量线程的形式启动。所有线程执行相同的Kernel代码，但每个线程会根据自己的编号处理不同位置的数据。在本例中，每个线程可以分别负责不同的i

- Thread:即线程，这是程序员视角下最基本的并行单位。每个线程执行同一份kernel代码，但处理的数据不同
- Block:多个Thread会组成一个Block。其意义在于**协作**，即同一个block内的线程可以通过共享内存交换数据，也可以使用同步原语协调执行不同 block 之间通常不能直接同步或共享局部数据。
- Grid:一次Kernel启动产生的所有线程构成一个grid。grid只是一次kernel启动时的逻辑组织方式，本身并不对应某个具体的硬件结构
- Warp:在硬件执行层面，线程通常不会以单个 thread 为单位独立调度，而是会进一步组成 warp。以 NVIDIA GPU 为例，一个 warp 通常包含 32 个线程。warp 内的线程会执行相同的指令流程，但处理各自的数据。这意味着，虽然我们在编程时使用的是 thread 和 block 这些抽象概念，但在硬件上，block 内的线程最终仍会被拆分为多个 warp 来执行。
- SM:流处理器，GPU上的block会被分配到SM上执行。一个SM可以同时驻留多个warp，甚至是多个block。当某个warp因访存等高延迟操作暂时停顿时，硬件调度器可以快速切换到其他就绪的 warp 继续执行，从而隐藏延迟、提升整体吞吐。

## Weighted Sum

我们接下来通过“加权求和”的例子来具体了解Triton的知识以及它是如何与Pytorch进行协同工作。

#### Forward Pass

“加权求和”的前向传播过程可以通过如下形式进行描述:

​	给定一个输入矩阵`x`,我们将它的每个元素乘以一个按列加权的向量`w`,然后对每一行求和，最终得到矩阵`x`与向量`w`的加权和。

用python代码描述则是:

```
def weighted_sum(x,weight):
	return (weight*x).sum(axis=-1)
```

而我们想把这个过程用Triton进行并行化，那么首先需要分析可并行性，我们可以发现矩阵与向量的乘积在最后求和得到结果之前各个元素的运算都是相互独立的，因此每个元素的计算都是可并行的。

因此在编写Triton Kernel时，我们会让每个program instance(Triton中一组执行同一段程序的线程块)负责`x`中某一块行的加权和，并把对应的标量输出写入输出张量。

例如，假设`x`的形状是`[ROWS,D]`,`weight`的形状是`[D]`，我们会让每个program instance负责`x`中若干行的加权和的计算，得到每个块行的加权和结果，最后将结果均写入输出张量。

与直接把tensor作为参数不同，我们会在kernel中传入:

- 指向张量首元素的指针
- 每个tensor的stride，其含义是告诉我们如何沿着各个维度移动

我们可以利用这些 stride，结合 program ID，将工作分配给不同实例（例如，第 `i` 个实例处理 `x` 的第 `i` 个行块），从而加载当前实例所负责的 `x` 的一块数据。

在本例中，Triton的Forward Pass和PyTorch的主要区别在于:Triton需要显式地进行指针运算以及load/store操作。

我们来看具体的代码:

```
import triton
import triton.language as tl
@triton.jit 
def weighted_sum_fwd(
	x_ptr,weight_ptr,			#输入指针
	output_ptr,					#输出指针
	x_stride_row,x_stride_dim,	#stride告诉我们在tensor的每个轴上移动一个元素需要跳多远
	weight_stride_dim,			#通常是1
	output_stride_dim,			#通常是1
	ROWS,D,
	ROWS_TILE_SIZE,D_TILE_SIZE,	#tile形状必须在编译期已知
):
	#每个instance负责计算x中一个行块的加权和
	#`tl.program_id`用于查看当前运行的是哪个线程块
	row_tile_idx = tl.program_id(0)
	
	#block pointer 允许我们从一个N维内存区域中选取数据块
	#并在这个区域中移动所选的数据块
	#在使用block pointer时必须知道:
	#- tensor 第一个元素的指针
	#- tensor 的整体形状
	#- 每个维度的stride
	#- 起始block的N维坐标
	#- 每次load/store的block形状
	#- 内存中维度从主到次的顺序
	x_block_ptr = tl.make_block_ptr(
		x_ptr,
		shape=(ROWS,D),
		strides=(x_stride_row,x_stride_dim),
		offsets=(row_tile_idx * ROW_TILE_SIZE,0),
		block_shape=(ROW_TILE_SIZE,D_TILE_SIZE),
		order(1,0),
	)
	weight_block_ptr=tl.make_block_ptr(
		weight_ptr,
		shape=(D,),
		strides=(weight_stride_dim,),
		offsets=(0,),
		block_shape=(D_TILE_SIZE,),
		order(0,),
	)
	output_block_ptr=tl.make_block_ptr(
		output_ptr,
		shape=(ROWS,),
		strides=(output_stride_row,),
		offsets=(row_tile_idx*ROWS_TILE_SIZE,),
		block_shape=(ROWS_TILE_SIZE,),
		order(0,),
	)
	#初始化一个用于写入的buffer
	output = tl.zeros((ROWS_TILE_SIZE,),dtype=tl.float32)
	for i in range(tl.cdiv(D,D_TILE,SIZE)):
		#加载当前block pointer指向的数据块
		#因为ROW_TILE_SIZE可能不整除ROWS，
		#D_TILE_SIZE 也可能不能整除 D，
		#所以两个维度都需要进行边界检查
		row = tl.load(x_block_ptr,boundary_check=(0,1),padding_option="zero")
		weight = tl.load(weight_block_ptr,boundary_check=(0,),padding_option="zero")
		#计算当前行块的加权和
		output += tl.sum(row * weight[None,:],axis=1)
		#将指针移动到下一个tile,这里参数时(行偏移，列偏移)
		x_block_ptr = x_block_ptr.advance(0,D_TILE_SIZE)	#在最后一个维度上前进D_TILE_SIZE
		weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
	# 将输出写入 output block pointer（每行一个标量）
    # 因为 ROWS_TILE_SIZE 可能不能整除 ROWS，所以需要边界检查
	tl.store(output_block_ptr,output,boundary_check(0,))
		
		
		
```

接下来我们将这个实现的Kernel封装到一个Pytorch `AutoGrad`函数中，使其能够与PyTorch生态协同工作(即接收 Tensor 输入、输出 Tensor，并在 backward pass 中和 autograd 引擎协作):

```
class WeightedSumFunc(torch.autograd.Function):
	@staticmethod
	def forward(ctx,x,weight):#这里ctx指上下文
		#将x和weight缓存起来，以便在backward中使用
		#backward时我们只会接收到输出张量的梯度
		# 需要利用这些缓存来计算 x 和 weight 的梯度
		D,output_dims = x.shape[-1],x.shape[:-1]
		input_shape = x.shape
		x = rearrange(x,"... d -> (...) d")
		ctx.save_for_backward(x,weight)
		
		assert len(weight.shape) == 1 and weight.shape[0] == D, "维度不匹配"
        assert x.is_cuda and weight.is_cuda, "期望输入是 CUDA tensor"
        assert x.is_contiguous(), "我们的指针运算默认 x 是连续的"
        
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) # 16 #大致让dim维度循环16次
        ctx.ROWS_TILE_SIZE = 16						# 每个线程一次处理16个batch元素
        ctx.input_shape = input_shape
        # 需要初始化一个空的结果 tensor
        # 注意：这里的元素未必初始化为 0
        y = torch.empty(output_dims,device = x.device)
        
        # 以 1D launch grid 启动 kernel
        n_rows = y.numel()
        weighted_sum_fwd[(cdiv(n_rows,ctx.ROWS_TILE_SIZE),)](	#和CUDA的<<<>>>有点像，作用是定义grid大小
        	x,weight,
        	y,
        	x.stride(0),x.stride(1),
        	weight.stride(0),
        	y.stride(0),
        	ROWS = n_rows,D=D,
        	ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(input_shape[:-1])
```

#### Backward Pass

由于我们定义了自己的kernel，因此也需要自己编写backward函数。

在forward pass中，我们拿到了层的输入，并计算其输出。而在backward pass中我们会拿到目标函数对该层输出的梯度，然后需要计算目标函数对每个输入的梯度。

在这个例子中，我们的操作有两个输入:

- 矩阵$x \in \mathbb{R}^{n\times h}$
- 权重向量$w \in \mathbb{R}^h$

记我们的操作为$f(x,w)$，其输出属于$\mathbb{R}^n$

若给定损失函数L对该层输出的梯度$\nabla_{f(x,w)}L$,则根据多元链式法则，可以得到关于x和w的梯度:
$$
(\nabla_xL)_{ij}=\sum_{k=1}^n\frac{\partial f(x,w)_k}{\partial x_{ij}}(\nabla_{f(x,w)}L)_k = w_j \cdot (\nabla_{f(x,w)}L)_i\\
(\nabla_w L)_{j} = \sum_{i=1}^n\frac{\partial f(x,w)_i}{\partial w_j}(\nabla_{f(x,w)}L)_i=\sum_{i=1}^nx_{ij}\cdot(\nabla_{f(x,w)}L)_i
$$
这里给出一个很简单的backward计算公式。

为了得到关于x的backward结果，我们根据上式计算w和$\nabla f(x,w)$的外积。而为了得到关于w的backward结果(即$(\nabla_w L)_j$),我们必须把输入与对应输出的梯度逐行相乘并求和。

按照同样的思路，我们来实现这个kernel:

```
@triton.jit
def weighted_sum_backward(
	x_ptr,weight_ptr,
	grad_output_ptr,
	grad_x_ptr,partial_grad_weight_ptr,
	stride_xr,stride_xd,
	stride_wd,
	stride_gr,
	stride_gxr,stride_gxd,
	stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE, D_TILE_SIZE,
):
	row_tile_idx = tl.program_id(0)
	n_row_tiles = tl.num_programs(0)
	#输入
	grad_output_block_ptr = tl.make_block_ptr(
		grad_output_ptr,
		shape=(NUM_ROWS,),strides=(stride_gr,),
		offsets=(row_tile_idx*ROW_TILE_SIZE,),
		block_shape=(ROWS_TILE_SIZE,),
		order=(0,),
	)
	x_block_ptr = tl.make_block_ptr(
		x_ptr,
		shape=(NUM_ROWS,D,),strides=(stride_xr,stride_xd),
		offsets=(row_tile_idx * ROW_TILE_SIZE,0),
		block_shape=(ROWS_TILE_SIZE,D_TILE_SIZE),
		order=(1,0),
	)
	weight_block_ptr = tl.make_block_ptr(
		weight_ptr,
		shape=(D,),strides=(stride_wd,),
		offsets=(0,),block_shape=(D_TILE_SIZE,),
		order=(0,),
	)
	grad_x_block_ptr = tl.make_block_ptr(
		grad_x_ptr,
		shape=(NUM_ROWS,D,),strides=(stride_gxr,stride_gxd),
		offsets=(row_tile_idx*ROW_TILE_SIZE,D_TILE_SIZE),
		block_shape=(ROW_TILE_SIZE,D_TILE_SIZE),
		order=(1,0),
	)
	partial_grad_weight_block_ptr = tl.make_block_ptr(
		partial_grad_weight_ptr,
		shape=(n_row_tiles,D,),strides=(stride_gwb,stride_gwd),
		offsets=(row_tile_idx,0),
		block_shape=(1,D_TILE_SIZE),
		order=(1,0),
	)
	for i in range(tl.cdiv(D,D_TILE_SIZE)):
		grad_output = tl.load(
			grad_output_block_ptr,
			boundary_check=(0,),
			padding_option="zero",
		)
		weight = tl.load(
			weight_block_ptr,
			boundary_check=(0,),
			padding_option="zero",
		)
		grad_x_row = grad_output[:,None] * weight[None,:]
		tl.store(grad_x_block_ptr,grad_x_row,boundary_chekc=(0,1))
		
		row = tl.load(
			x_block_ptr,
			boundary_check=(0,1),
			padding_option="zero",
		)
		grad_weight_row = tl.sum(row * grad_output[:,None],axis=0,keep_dims=True)
		tl.store(partial_grad_weight_block_ptr,grad_weight_row,boundary_check=(1,))
		
		x_block_ptr = x_block_ptr.advance((0,D_TILE_SIZE))
		weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
		partial_grad_weight_block_ptr = parital_grad_weight_block_ptr.advance((0,D_TILE_SIZE))
		grad_x_block_ptr = grad_x_block_ptr.advance((0,D_TILE_SIZE))
```

每个 kernel instance 只负责处理 `x` 的一个行块，但我们现在需要沿着 `x` 的行方向求和。
 因此，在 backward kernel 内，我们**不直接完成这个总和**，而是假设 `partial_grad_weight_ptr` 是一个形状为 `n_row_tiles × H` 的矩阵，其中第一维表示来自每个行块的局部归约结果。

也就是说：

- kernel 内部：只在当前行块内完成局部归约
- kernel 外部：再用 `torch.sum` 把所有行块的结果加起来，得到最终的 $\nabla_w$

这样一来，`autograd.Function` 的最后一部分就相对简单了：

```
class WeightSumFunc(torch.autograd.Function):
	@staticmethod
	def forward(ctx,x,weight):
		#这里省略，前面定义过了
	@staticmethod
	def backward(ctx,grad_out):
		x,weight = ctx.saved_tensors
		ROWS_TILE_SIZE,D_TILE_SIZE = ctx.ROWS_TILE_SIZE,ctx.D_TILE_SIZE
		n_rows,D = x.shape()
		
		#我们的策略是：每个线程块先写入一个partial buffer，然后对这个buffer做归约得到最终梯度
		parital_grad_weight = torch.empty(
			(cdiv(n_rows,ROWS_TILE_SIZE),D),
			device=x.device,
			dtype=x.dtype
		)
		grad_x = torch.empty_like(x)
		weighted_sum_backward[(cdiv(n_rows,ROWS_TILE_SIZE),)](
			x,weight,
			grad_out,
			grad_x,partial_grad_weight,
			x.stride(0),x.stride(1),
			weight.stride(0),
			grad_out.stride(0),
			grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
		)
		grad_weight = partial_grad_weight.sum(axis=0)
		return grad_x,grad_weight
```

最后我们就可以得到一个函数，它的使用方法和`torch.nn.functional`里的函数很相似:

```
f_weightedsum = WeightedSumFunc.apply
```

我们可以对两个PyTorch tensor `x`和`w`调用这个函数，就会得到类似如下的结果:

```
tensor([ 90.8563, -93.6815, -80.8884, ..., 103.4840, -21.4634, -24.0192],
       device='cuda:0', grad_fn=<WeightedSumFuncBackward>)
```

请注意输出 tensor 上附带的 `grad_fn` —— 这表明 PyTorch 已经知道：当这个 tensor 出现在计算图中并需要做 backward pass 时，应该调用什么函数。

至此，我们就完成了这个 **weighted sum 操作的 Triton 实现**。

对于其它算子，我们可以按照同样的思路进行。实际上重点还是在于我们应该分析出哪些是可以并行的，program instance该如何划分。

简单来说，在设计Triton算子的时候我们需要考虑三个问题:

- 输出的自然分块是什么？例如在Weighted Sum中Y[0],Y[1]相互独立按行产生，那么Program Instance可以负责生成一个`ROW_TILE_SIZE`大小的行块的结果。
- 一个块的输出需要哪些输入？
- 这些输入能不能以连续，重复利用的方式被加载？

这三个问题虽然简单，但是在Triton编程中很重要，在实际问题中也很难立即给出标准答案，需要我们不断试错积累经验。



建议参考仓库(Assignment2-systems/Triton/WeightedSum.py)中的代码，不要直接复制粘贴，不然会出现一些奇奇怪怪的错误，这里的代码是参考了CS336 Assignment 2中的代码.



## Triton Puzzles (Lite)

为了更深一步的学习Triton,我在网上查找到了一份通过做题来学习Triton的repo。

原始的repo是[srush/Triton-Puzzles: Puzzles for learning Triton](https://github.com/srush/Triton-Puzzles)

但是有一个更好的实现版本是:[SiriusNEO/Triton-Puzzles-Lite: Puzzles for learning Triton, play it with minimal environment configuration!](https://github.com/SiriusNEO/Triton-Puzzles-Lite)

因此我们还是以这个仓库为题面来进行解答。

另外需要强调的一点是，正如Assignment2实验手册中提到的那样，目前大多教程没有使用更新的，更方便的block pointer抽象。因此在此基础上，在本blog中，我将以block pointer抽象来完成这些任务。



之所以说block pointer更加方便，是因为在Triton中，我们的Program Instance的操作对象是划分后的Tile，直观来看就是一个矩形区域。在使用block pointer之前，我们需要得到这个处理对象需要通过Program ID以及SIZE用公式去计算。例如我们想知道处理一个`x[ROWS,D]`的tile，我们需要写:

```
row_offsets = row_tile_idx * ROWS_TILE_SIZE + tl.arrange(0,ROWS_TILE_SIZE)
col_offsets = tl.arrange(0,D_TILE_SIZE)
#计算出当前块负责哪些行和哪些列
x_ptrs = x_ptr + row_offsets[:,None] * x_stride_row + col_offsets[None,:] * x_stride_dim
#然后把他们拼接为二维地址
mask = (row_offsets[:, None] < ROWS) & (col_offsets[None, :] < D)
#然后再load
x = tl.load(x_ptrs, mask=mask, other=0.0)

```

而使用block pointer的话，我们就可以写:

```
x_block_ptr = tl.make_block_ptr(...)
x = tl.load(x_block,boundary_check=(0,1),padding="zero")
```

这样做就省去了大量的地址计算，一方面更加便捷，另一方面代码的可读性也大大提升了。接下来我们看看具体的问题以及如何解决吧？

### Puzzles 1/2 Constant Add

问题描述:
	
对向量中的每个元素都加上常数10.使用一个程序ID轴。

问题解答：

我们按照上面Weighted Sum中`make_block_ptr`的方式定义好`x_block_ptr`以及`z_block_ptr`，然后`tl.load`出x，进行运算得到z最后进行`tl.store`即可。

由于N0不一定能被B0整除，因此需要进行boundary check

```
@triton.jit
def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(row_tile_idx * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(row_tile_idx * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    x = tl.load(x_block_ptr,boundary_check=(0,),padding_option='zero')
    z = x + 10.0
    tl.store(z_block_ptr, z, boundary_check=(0,))
    # Finish me!
    return
```

### Puzzles 3/4 Outer Vector Add

问题描述:

把两个向量按广播方式相加，得到一个二维结果矩阵`z`。
$$
z_{j,i} = x_i+y_j \quad i=1,\dots,B_0,j=1,\dots,B_1
$$

问题解答:

这个可以利用Python中的broadcast机制:`x[None,:]+y[:,None]`。在计算时会先将x的形状变为`(1,N0)`,y变为`(N1,1)`然后相加的时候触发广播机制，两者的 shape 均广播到 `(N1, N0)`。

在Triton中也有同样的机制，我们可以利用这个得到最终的结果。

考虑到N0,N1,B0,B1的差异，我们的`grid`相应地也要变成二维的，第一个维度是走`x`的tile，第二个维度走`y`的tile

```
@triton.jit
def add_vec_block_kernel(
    x_ptr, y_ptr, z_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr
):
    x_tile_idx = tl.program_id(0)
    y_tile_idx = tl.program_id(1)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(x_tile_idx * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(N1,),
        strides=(1,),
        offsets=(y_tile_idx * B1,),
        block_shape=(B1,),
        order=(0,)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N1, N0),
        strides=(N0, 1),
        offsets=(y_tile_idx * B1, x_tile_idx * B0),
        block_shape=(B1, B0),
        order=(0, 1)
    )
    x = tl.load(x_block_ptr,boundary_check=(0,),padding_option='zero')
    y = tl.load(y_block_ptr,boundary_check=(0,),padding_option='zero')
    z = x[None, :] + y[:, None]
    tl.store(z_block_ptr, z, boundary_check=(0,1))
    # Finish me!
    return
```

### Puzzles 5 Fused Outer Multiplication

问题描述:

将行向量`x`与列向量`y`做外积，然后对结果矩阵的每个元素应用ReLU.
$$
z_{j,i} = \max(0,x_i\cdot y_j),\quad i = 1,\dots,N_0,j=1,\dots,N_1
$$

问题解答：

这一问实际上是与Puzzle 3/4类似的，只有符号从加法变成了乘法。对于ReLU操作直接将z与0取max就行:`z = tl.maximum(0,z)`

### Puzzles 6 Fused Outer Multiplication - Backwards

问题描述：

对如下函数操作进行反向传播:矩阵`x`与向量`y`按行相乘，再经过ReLU。

$$
f(x,y) = \text{relu}(x_{j,i}\times y_j),\quad i = 1 \dots N_0,j = 1 \dots N_1\\
dx_{j,i} = f'_x(x,y)_{j,i}\times dz_{j,i} = 1(x_{j,i}\cdot y_j >0)\cdot y_j \cdot dz_{j,i}
$$
其中$1(\cdots)$表示指示函数，括号内条件满足时为1，否则为0

问题解答：

按照公式实现即可，没有较为复杂的逻辑，因此不贴代码了。

### Puzzles 7  Long Sum

问题描述:

给定一个二维张量`x`，对其每一行进行求和。

问题解答:

这是一个二维的张量，我们在分块计算的时候虽然直观的想法是每行进行并行计算，但是考虑到每行有T个元素	，而我们设计的块的大小是$B_0\times B_1$，而B1远小于B0,因此我们需要沿着第二维分块累加，这也是为什么题目提醒我们需要使用for循环。

当我们使用block pointer时，我们可以直接用`tl.advance(pointer,(stride_dim1,...))`的方式移动指针。

```
@triton.jit
def sum_kernel(x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr):
    # Finish me!
    tile_id = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0, T),
        strides=(T, 1),
        offsets=(tile_id * B0, 0),
        block_shape=(B0, B1),
        order=(1, 0)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(tile_id * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    z = tl.zeros((B0,), dtype=tl.float32)
    for i in range(0, T, B1):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
        z += tl.sum(x,1)
        x_block_ptr = tl.advance(x_block_ptr,(0,B1))
    tl.store(z_block_ptr, z, boundary_check=(0,))
    return
```

### Puzzles 8 Long Softmax

问题描述:

对一批logits做Softmax。需要保证数值稳定，即减去最大值后求Softmax
$$
z_{i,j} = \text{softmax}(x_{i,1},\dots,x_{i,T}) \quad i = 1,\dots,N_0
$$

另外需要注意在Triton中建议不要直接用`exp`而是用`exp2`.$\text{exp}(x) = 2^{\log_2(e)x}$

问题解答:

我们先来做没有优化的三个for的版本。实际上我们可以将任务做一个拆分。因为我们是分行块处理的，但是tile的大小`[B0,B1]`其中$B1 < T$，因此我们需要for循环来得到:

1. 每行的max
2. 每行的sum
3. 每行的结果

因此需要3个for循环。因为后续我们将通过Online Softmax算法来实现更加高效实用的版本，因此这里就不贴出具体的代码了，思路和Puzzle 7类似。

Online Softmax的思路其实比较简单，就是第一个for和第二个for可以合并!

考虑两个tile，第一个tile的max我们记为$m_1$,那么此时局部求和的结果为:
$$
sum = \sum_{i = 0}^{B_0-1} \exp(x_i - m_1)
$$
第二个tile的max记为$m_2(m_2 > m_1)$,我们记$a = m_1 - m_2$,那么此时我们可以更新局部求和结果为:
$$
sum' = sum \times \exp(a) + \sum_{B_0}^{2B_0-1}\exp(x_i - m_2)
$$
由此我们可以通过一个for循环，便得到了求最终结果需要的max以及sum。

```
@triton.jit
def softmax_kernel(
    x_ptr, z_ptr, N0, N1, T, B0: tl.constexpr, B1: tl.constexpr
):
    """3 loops ver."""
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0, T),
        strides=(T, 1),
        offsets=(block_id_i * B0, 0),
        block_shape=(B0, B1),
        order=(1, 0)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N0, T),
        strides=(T, 1),
        offsets=(block_id_i * B0, 0),
        block_shape=(B0, B1),
        order=(1, 0)
    )
    z = tl.zeros((B0, B1), dtype=tl.float32)
    x_max = tl.full((B0,), float("-inf"), dtype=tl.float32)
    sum = tl.zeros((B0,), dtype=tl.float32)
    x_block_ptr1 = x_block_ptr
    for i in range(0,T,B1):
        x = tl.load(x_block_ptr1, boundary_check=(0, 1), padding_option='zero')
        new_x_max = tl.maximum(x_max, tl.max(x,1))
        if i == 0:
            x_max = new_x_max
            sum = tl.sum(tl.exp2(log2_e * (x - x_max[:, None])),1)
        else:
            scale = tl.exp2(log2_e * (x_max - new_x_max))
            sum = sum * scale + tl.sum(tl.exp2(log2_e * (x - new_x_max[:, None])),1)
            x_max = new_x_max
        x_block_ptr1 = tl.advance(x_block_ptr1,(0,B1))
    for i in range(0,T,B1):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')
        x = x - x_max[:, None]
        x_exp = tl.exp2(log2_e * x)
        z = x_exp / sum[:, None]
        tl.store(z_block_ptr, z, boundary_check=(0, 1))
        x_block_ptr = tl.advance(x_block_ptr,(0,B1))
        z_block_ptr = tl.advance(z_block_ptr,(0,B1))
        # Finish me!
    return
```

### Puzzles 9 Simple FlashAttention

问题描述:

不使用 program。大小 B0 表示在总共 N0 个 q 中，本次要处理的一批。序列长度是 T。每次处理 B1 < T 个元素（k、v） ，其中 B1 是某个块大小。
$$
z_i = \sum_{j=1}^T \text{softmax}(q_ik_1,\dots,q_ik_T)_j v_j
$$
这个问题可以通过类似于Online Softmax的思路来解决。

问题解答:(为便于理解，推荐读者用纸笔画一下)

考虑q的一个大小为`[B0=2]`的tile中元素为`[q_0,q_1]`,k和v的一个大小为`[B1=2]`的tile中的元素为`[k_1,k_2]`,`[v_1,v_2]`，那么对应的注意力矩阵元素为一个大小为`[B0,B1]`的二维矩阵，其中元素为`[s_11,s_12,s_21,s_22]`。

仅考虑结果中的第一个元素$z_1$,此时其累加和为:$z_1 = \text{softmax}(s_{11})v_1+\text{softmax}(s_{12})v_2 = \frac{\exp(s_{11}-m_1)}{l_1}v_1+\frac{\exp(s_12-m_1)}{l_1}v_2$,其中$m_1$为此时的局部最大值，$l_1$为此时的累计指数和.

计算完毕后，指针沿着axis=1的维度进行移动(步长为B1)，此时k和v的一个大小为`[B1=2]`的tile中的元素为`[k_3,k_4]`,`[v_3,v_4]`，那么对应的注意力矩阵元素为一个大小为`[B0,B1]`的二维矩阵，其中元素为`[s_13,s_14,s_23,s_24]`。

同样考虑结果中的第一个元素$z_1$,此时最大值更新为$m_2$,指数累计和应更新为$l_2 = l_1\times(\text{scale}=\exp(m_1-m_2))+\sum_{j=3}^4 \exp(s_{1j}-m_2)$.

那么我们分两部分考虑此时的结果$z_1$,一部分是新的注意力权重$s_{13},s_{14}$与$v_3,v_4$的加权和，这部分直接和之前一样相加就行，而另一部分则为之前计算出来的贡献，这部分由于m和l的更新，也要对应更新。

更新可以从分子和分母考虑。分子上和Online Softmax类似，需要乘以$\exp(m_1-m_2)$,而分母则需要进行替换为$l_2$,因此最终我们可以更新局部加权和$z_1$为:
$$
z_1' = z_1\times scale\times \frac{l_1}{l_2} + \sum_{j=3}^4\frac{\exp(s_{1j}-m_2)}{l_2}v_j
$$
由此不断迭代，便能得到最终结果。

```
@triton.jit
def flashatt_kernel(
    q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr, B1: tl.constexpr
):
    block_id_i = tl.program_id(0)
    log2_e = 1.44269504
    myexp = lambda x: tl.exp2(log2_e * x)
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(block_id_i * B0,),
        block_shape=(B0,),
        order=(0,)
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(T,),
        strides=(1,),
        offsets=(0,),
        block_shape=(B1,),
        order=(0,)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(T,),
        strides=(1,),
        offsets=(0,),
        block_shape=(B1,),
        order=(0,)
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N0,),
        strides=(1,),
        offsets=(block_id_i * B0,),
        block_shape=(B0,),
        order=(0,)
    )

    qk_max = tl.full((B0,), float("-inf"), dtype=tl.float32)
    qk_sum = tl.zeros((B0,), dtype=tl.float32)
    o = tl.zeros((B0,), dtype=tl.float32)
    for i in range(0, T, B1):
        q = tl.load(q_block_ptr, boundary_check=(0,), padding_option='zero')
        k = tl.load(k_block_ptr, boundary_check=(0,), padding_option='zero')
        v = tl.load(v_block_ptr, boundary_check=(0,), padding_option='zero')
        qk = q[:, None] * k[None, :]
        new_qk_max = tl.maximum(qk_max,tl.max(qk, 1))
        if i == 0:
            qk_max = new_qk_max
            qk_exp = myexp(qk - qk_max[:,None])
            qk_sum = tl.sum(qk_exp, 1)
            o = tl.sum(qk_exp * v[None, :] / qk_sum[:,None], 1)
        else:
            qk_exp = myexp(qk - new_qk_max[:,None])
            scale = myexp(qk_max - new_qk_max)
            scale1 = scale * qk_sum
            qk_sum = qk_sum * scale + tl.sum(qk_exp, 1)
            o = o * scale1 / qk_sum + tl.sum(qk_exp * v[None, :] / qk_sum[:,None], 1)
            qk_max = new_qk_max
        k_block_ptr = tl.advance(k_block_ptr,(B1,))
        v_block_ptr = tl.advance(v_block_ptr,(B1,))
    tl.store(z_block_ptr, o, boundary_check=(0,))
    # Finish me!
    return
```

### Puzzles 10 Two Dimensional Convolution

问题描述:

实现一个带批处理的二维卷积。使用一个program id 轴。块大小为`B0`表示在`N0`中一次处理多少个batch。图像`x`的大小为`H x W`，并且只有一个通道；卷积核`k`的大小为`kH x kW`
$$
z_{i, j, l} = \sum_{oj, ol}^{j+oj\le H, l+ol\le W} k_{oj,ol} \times x_{i,j + oj, l + ol} 
    \text{ for } i = 1\ldots N_0 \text{ for } j = 1\ldots H \text{ for } l = 1\ldots W
$$


问题解决：(为了方便理解，最好对照题面的图来理解)

我们考虑如何分块，题目中要求的是按照批维度进行分块拆分并行。那么块的第一维就是B0,注意到卷积运算每次只会用到图像矩阵中KW*KH大小的子矩阵，因此块最合适的大小就是`[B0,KH,KW]`。这个块与卷积核进行卷积运算，得到一个标量，因此结果的块的大小就是`[B0,1,1]`，那么这样就能写出block pointer的定义。

接下来模拟二维卷积运算就行了，根据相关知识，我们知道卷积运算就是卷积核在图像矩阵上进行平移(这里默认步长为1).那么我们用两个for循环来模拟这个平移计算的流程就行，然后在内层for进行一次后沿着dim=1平移一格，在外层for进行一次后沿dim=0平移一格。

```
@triton.jit
def conv2d_kernel(
    x_ptr, k_ptr, z_ptr, N0, H, W, KH: tl.constexpr, KW: tl.constexpr, B0: tl.constexpr
):
    block_id_i = tl.program_id(0)
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(KH, KW),
        strides=(KW, 1),
        offsets=(0, 0),
        block_shape=(KH, KW),
        order=(1, 0),
    )
    k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")

    x_row_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N0, H, W),
        strides=(H * W, W, 1),
        offsets=(block_id_i * B0, 0, 0),
        block_shape=(B0, KH, KW),
        order=(2, 1, 0),
    )
    z_row_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N0, H, W),
        strides=(H * W, W, 1),
        offsets=(block_id_i * B0, 0, 0),
        block_shape=(B0, 1, 1),
        order=(2, 1, 0),
    )

    for _ in range(H):
        x_block_ptr = x_row_block_ptr
        z_block_ptr = z_row_block_ptr
        for _ in range(W):
            x = tl.load(x_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")
            z = (x * k[None, :, :]).sum(1).sum(1)
            tl.store(z_block_ptr, z[:, None, None], boundary_check=(0, 1, 2))
            x_block_ptr = tl.advance(x_block_ptr, (0, 0, 1))
            z_block_ptr = tl.advance(z_block_ptr, (0, 0, 1))
        x_row_block_ptr = tl.advance(x_row_block_ptr, (0, 1, 0))
        z_row_block_ptr = tl.advance(z_row_block_ptr, (0, 1, 0))

    return
```

### Puzzles 11 Matrix Multiplication

问题描述：

使用三条program id轴。块大小`B2`表示在`N2`中要处理的Batch数量。块大小`B0`表示在`N0`中要处理的行数，块大小`B1`表示在`N1`中要处理的列数。

中间维度的大小为`MID`
$$
z_{i, j, k} = \sum_{l} x_{i,j, l} \times y_{i, l, k}\quad 
\text{for } i = 1\ldots N_2,\ j = 1\ldots N_0,\ k = 1\ldots N_1
$$

Hint:可以使用`tl.dot`，它可以计算一个更小规模的矩阵乘法

问题解答:

我们知道两个矩阵的形状分别为`shape(x) = [N2,N0,MID],shape(y) = [N2,MID,N1]`.我们分块的话，显然需要在MID维度上进行拆分，那么两个矩阵对应的块大小为:`[B2,B0,B_MID],[B2,B_MID,B1]`,然后将这些小矩阵的乘积进行累加，便能得到最终的结果。

```
@triton.jit
def dot_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    N0,
    N1,
    N2,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B_MID: tl.constexpr,
):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)
    block_id_i = tl.program_id(2)
    # Finish me!
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(N2,N0,MID),
        strides=(N0 * MID,MID,1),
        offsets=(block_id_i * B2, block_id_j * B0, 0),
        block_shape=(B2, B0, B_MID),
        order=(0, 1, 2),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(N2,MID,N1),
        strides=(MID * N1,N1,1),
        offsets=(block_id_i * B2, 0, block_id_k * B1),
        block_shape=(B2, B_MID, B1),
        order=(0, 1, 2),
    )
    z_block_ptr = tl.make_block_ptr(
        base=z_ptr,
        shape=(N2,N0,N1),
        strides=(N0 * N1,N1,1),
        offsets=(block_id_i * B2, block_id_j * B0, block_id_k * B1),
        block_shape=(B2, B0, B1),
        order=(0, 1, 2),
    )
    z = tl.zeros((B2, B0, B1), dtype=tl.float32)
    for i in range(0, MID, B_MID):
        x = tl.load(x_block_ptr, boundary_check=(0,1,2),padding_option='zero')
        y = tl.load(y_block_ptr, boundary_check=(0,1,2),padding_option='zero')
        z += tl.dot(x,y)
        x_block_ptr = tl.advance(x_block_ptr,(0,0,B_MID))
        y_block_ptr = tl.advance(y_block_ptr,(0,B_MID,0))
    tl.store(z_block_ptr, z, boundary_check=(0,1,2))
    return

```

### Puzzles 12 Quantized Matrix Mult

问题描述:

我们将对一个矩阵乘法进行量化压缩，具体而言是将权重矩阵以更低精度存储，并额外配合一个偏移项和缩放项。

在本问题中，我们的`weight`将以4bit存储。一个32位整数中可以存放`FPINT`个这样的值。此外，对每个连续`group`个权重，我们还会额外存储:

- 1个`scale`浮点值
- 1个`shift`的4bit值

这些`scale`和`shift`是按`weight`的列来存储的。而`activation`则单独以普通浮点数格式存储。
$$
z_{j, k} = \sum_{l} sc_{j, \frac{l}{g}} (w_{j, l} - sh_{j, \frac{l}{g}}) \times y_{l, k}
\quad \text{for } j = 1\ldots N_0,\ k = 1\ldots N_1
$$
其中`g`表示分组大小。

问题解答:

本题的主要难点，不在于计算，而是张量的形状处理。我们先来梳理以下完整的操作流程。

首先我们还是按照正常的矩阵乘法在MID维度上进行拆分，然后由于权重是以`INT8`格式存储的，我们还需要将其提取出来(利用位运算即可)。然后每个GROUP我们都会存储一个浮点型的`scale`和`INT4`类型的`shift`.

为了方便后续理解，我们看看各个分块的大小(仅看输入):

- 激活值,这个最好理解，它是以浮点型存储的，然后因为是在MID维度做了拆分，因此块大小为`[B_MID,B_1]`,需要注意访问的顺序是列优先的，因此`order = (0,1)`
- 权重矩阵，这个需要考虑到`INT32`存储`INT8`带来的形状差异，我们记一个`INT32`可以存储`FPINT`个`INT4`,那么权重矩阵实际的大小为`[N0,MID//FPINT]`,那么对应的块大小就是`[B0,B_MID//FPINT]`
- scale,它是浮点型存储的，并且是一个GROUP(在MID维度分组)一个,那么其形状就是`[N0,GROUP]`,对应块大小为`[B0,GROUP]`
- offset,它与Scale类似，是一个GROUP一个，但是它是以`INT4`存储的，因此也需要考虑`FPINT`的事情，它的实际大小就是:`[B0,GROUP//FPINT]`

分完块之后就需要把用`int4`的块提取为`int32`格式的，方便后续计算。

然后需要考虑形状的问题了，我们提取出来的`INT32`格式的Weight的张量形状为`[B0,B_MID // FPINT,FPINT]`,但是我们的`scale`的形状为`[B0,GROUP]`，我们无法将其直接与Weight进行运算得到dequant的权重，因此我们需要将其reshape为`[B0,GROUP,B_MID//GROUP]`.`offsets`也同理，需要从`[B0, GROUP // FPINT, FPINT]`处理为`[B0,GROUP]`。之后借助广播进行dequant后，再进行矩阵乘法。

```
@triton.jit
def quant_dot_kernel(
    scale_ptr,
    offset_ptr,
    weight_ptr,
    activation_ptr,
    z_ptr,
    N0,
    N1,
    MID,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B_MID: tl.constexpr,
):
    block_id_j = tl.program_id(0)
    block_id_k = tl.program_id(1)

    def extract(x):
        over = tl.arange(0,8) * 4
        mask = 2**4 - 1
        return (x[:,:, None] >> over) & mask

    activation_block_ptr = tl.make_block_ptr(
        base = activation_ptr,
        shape = (MID,N1),
        strides = (N1,1),
        offsets = (0, block_id_k * B1),
        block_shape = (B_MID, B1),
        order = (0,1)
    )
    weight_block_ptr = tl.make_block_ptr(
        base = weight_ptr,
        shape = (N0,MID // FPINT),
        strides = (MID // FPINT,1),
        offsets = (block_id_j * B0, 0),
        block_shape = (B0, B_MID // FPINT),
        order = (1,0)
    )
    scale_block_ptr = tl.make_block_ptr(
        base = scale_ptr,
        shape = (N0, GROUP),
        strides = (GROUP,1),
        offsets = (block_id_j * B0, 0),
        block_shape = (B0, GROUP),
        order = (1,0)
    )
    offset_block_ptr = tl.make_block_ptr(
        base = offset_ptr,
        shape = (N0,GROUP // FPINT),
        strides = (GROUP // FPINT,1),
        offsets = (block_id_j * B0, 0),
        block_shape = (B0, GROUP // FPINT),
        order = (1,0)
    )
    z_block_ptr = tl.make_block_ptr(
        base = z_ptr,
        shape = (N0,N1),
        strides = (N1,1),
        offsets = (block_id_j * B0, block_id_k * B1),
        block_shape = (B0, B1),
        order = (1,0)
    )
    z = tl.zeros((B0, B1), dtype=tl.float32)
    for i in range(0, MID, B_MID):
        scale_fp32 = tl.load(scale_block_ptr, boundary_check=(0,1), padding_option='zero')#[B0,GROUP]
        
        offset_int32 = extract(tl.load(offset_block_ptr, boundary_check=(0,1), padding_option='zero'))#[B0, GROUP // FPINT, FPINT]
        offset_int32 = offset_int32.reshape(B0, GROUP)#[B0, GROUP]
        
        weight_int32 = extract(tl.load(weight_block_ptr, boundary_check=(0,1), padding_option='zero'))#[B0, B_MID // FPINT, FPINT]
        weight_int32 = weight_int32.reshape(B0, B_MID)#[B0, B_MID]
        weight_int32 = weight_int32.reshape(B0,GROUP,B_MID // GROUP)#[B0, GROUP, B_MID // GROUP]

        activation_fp32 = tl.load(activation_block_ptr, boundary_check=(0,1), padding_option='zero')
        
        weight_fp32 = scale_fp32[:,:,None] * (weight_int32 - offset_int32[:,:,None])
        weight_fp32 = weight_fp32.reshape(B0,B_MID)

        z += tl.dot(weight_fp32, activation_fp32)
        scale_block_ptr = tl.advance(scale_block_ptr,(0, GROUP))
        offset_block_ptr = tl.advance(offset_block_ptr,(0, GROUP // FPINT))
        weight_block_ptr = tl.advance(weight_block_ptr,(0, B_MID // FPINT))
        activation_block_ptr = tl.advance(activation_block_ptr,(B_MID, 0))
    tl.store(z_block_ptr, z, boundary_check=(0,1))

    # Finish me!
    return
```

## Reference

1. [namoe的解答](https://zhuanlan.zhihu.com/p/20539246076)
2. [Lite作者的解答](https://zhuanlan.zhihu.com/p/5964285807)
3. [先进编译实验室的讲解](https://www.bilibili.com/video/BV193fFYkE7P/)
4. CS336 Assignment2-systems的实验指导手册
5. [CSE559M Note](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)