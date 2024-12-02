# Hidden Markov models (HMMs)

## Introduction

**Hidden Markov model (隐马尔可夫模型) 是一种 SSM（状态空间模型）**，他的隐状态是离散的。它假设由一个隐藏的马尔科夫链随机生成不可观测的隐状态序列，再由各个隐状态随机生成一个观测态从而得到可观测序列，如下图：

![HMM](figures\HMM.png)

HMM可以由初始概率，状态转移概率，观测概率确定。引入以下定义。（这里首先假设观测概率服从 *Categorical Distribution* （离散））

设 $\mathcal Q $ 是所有可能的状态的结集合，$\mathcal V$ 是所有可能的观测的集合：
$$
\mathcal Q = \{q_1, q_2, \dots, q_N\} \\
\mathcal V = \{v_1, v_2, \dots, v_M\}
$$
其中，$N, M$ 分别表示状态数和观测数。隐状态序列和观测序列可分别表示为：
$$
\mathbf I = (i_1, i_2, \dots, i_T),\quad i_t = q_j \in \mathcal Q \\
\mathbf O = (o_1, o_2, \dots, o_T), \quad o_t = v_k \in \mathcal V
$$
由于状态是离散的，可以用矩阵来表示状态的转移。状态转移概率矩阵可以表示为：
$$
\mathbf A = [a_{ij}]_{N \times N} \\
a_{ij} = p(i_{t+1} = q_j | i_t = q_i)
$$
观测概率矩阵可以表示为：
$$
\mathbf B = [b_j(k)]_{N \times M} \\
b_j(k) = p(o_t = v_k | i_t = q_j)
$$
同时，我们需要初始概率分布：
$$
\boldsymbol \pi = (\pi_i) \\
\pi_i = p(i_1 = q_j)
$$
所以，HMM可以完全由这三种参数决定：
$$
\boldsymbol \lambda = (\boldsymbol \pi, \mathbf A, \mathbf B)
$$


HMM有两个重要假设：

1. 齐次马尔可夫假设（无后向性）
    即未来与过去无关，未来只依赖于当前状态：
    $$
    p(i_{t+1} | i_t, i_{t-1}, \dots, i_1, o_t. \dots, o_1) = p(i_{t+1} | i_t)
    $$

2. 观测独立假设
    $$
    p(o_t | i_t, i_{t-1}, \dots, i_1, o_{t-1}, \dots, o_1) = p(o_t | i_t)
    $$
    



HMM涉及三个问题：

1. Evaluation：
    - 求 $p(\mathbf O | \boldsymbol \lambda)$ ，即给定模型 $\boldsymbol \lambda$ 和观测序列 $\mathbf O$​ ，计算来自该模型的似然。
    - 涉及算法：forward-backward 算法
2. Learning：
    - 参数学习，$\boldsymbol \lambda$ 求解。非监督：$\hat{\boldsymbol \lambda} = \arg \max_{\boldsymbol \lambda} p(\mathbf O | \boldsymbol \lambda)$ ；监督：$\hat{\boldsymbol \lambda} = \arg \max_{\boldsymbol \lambda} p(\mathbf I, \mathbf O | \boldsymbol \lambda)$
    - 涉及算法：EM / Baum Welch 算法
3. Decoding：
    - 求隐状态序列，$\hat{\mathbf I} = \arg \max_{\mathbf I} p(\mathbf I | \mathbf O, \boldsymbol \lambda)$​
    - 涉及算法：Viterbi 算法
    - 预测问题：$p(i_{t+1} | o_{1:t})$ ，即已知当前观测下，预测下一状态，即 viterbi 算法
    - 滤波问题：$p(i_t | o_{1:t})$ ，即求 $t$ 时刻的隐状态





## Evaluation

当模型的 $\boldsymbol \lambda$ 给定（即模型的结构和参数给定），计算观测序列的似然度 $p(\mathbf O | \boldsymbol \lambda)$ 。
$$
p(\mathbf O | \boldsymbol \lambda) = \sum_{\mathbf I} p(\mathbf I, \mathbf O | \boldsymbol \lambda) = \sum_{\mathbf I} p(\mathbf O | \mathbf I, \boldsymbol \lambda)\ p(\mathbf I | \boldsymbol \lambda)
$$

$$
\begin{aligned}
	p(\mathbf I | \boldsymbol \lambda) 
	&= p(i_1, i_2, \dots, i_T | \boldsymbol \lambda) \\
	&= p(i_T | i_{1:T-1}, \boldsymbol \lambda)\ p(i_{1:T-1} | \boldsymbol \lambda) \\
	&= p(i_T | i_{T-1})\ p(i_{T-1} | i_{T-2}) \dots p(i_2 | i_1)\ p(i_1) \\
	&= p(i_1) \prod_{t=2}^{T} p(i_{t} | i_{t-1})
\end{aligned}
$$

$$
\begin{aligned}
	p(\mathbf O | \mathbf I, \boldsymbol \lambda) 
	&= p(o_1, \dots, o_T | \mathbf I, \boldsymbol \lambda) \\
	&= \prod_{t=1}^{T} p(o_t | i_t)
\end{aligned}
$$

$$
\begin{aligned}
	\therefore \quad p(\mathbf O | \boldsymbol \lambda) 
	&= \sum_{\mathbf I} p(i_1)\prod_{t=2}^{T} p(i_{t} | i_{t-1}) \prod_{t=1}^{T} p(o_t | i_t) \\
	&= \overbrace{\sum_{i_1} \sum_{i_2} \dots \sum_{i_T}}^{\color{blue}O(N^T)}\ p(i_1)\prod_{t=2}^{T} p(i_{t} | i_{t-1}) \prod_{t=1}^{T} p(o_t | i_t)
\end{aligned}
$$

由上面的公式可以看到，连续的求和带来 $O(N^T)$ 的时间复杂度，解决方案可以采用动态规划的方式，将需要重复计算的结果缓存起来。简单分析一下：当对于 $i_2, \dots, i_T$ 进行求和计算时，会对前面的 $p(i_1) p(o_1 | i_1)$ 重复计算多次，如果在每次计算是都利用前面已经计算好的结果时，这将大大减少计算量。于是引入前向后向算法。



### Forward algorithm

![HMM_alpha](figures\HMM_alpha.png)

首先，对前向概率做定义：$\alpha_t({\color{blue}i}) = p(o_{1:t}, i_t = q_{\color{blue}i} | \boldsymbol \lambda)$ ，于是 $\alpha_T(i) = p(o_{1:T}, i_T = q_i | \boldsymbol \lambda)$ 。
$$
P(\mathbf O | \boldsymbol \lambda) = \sum_{i=1}^N p(\mathbf O, i_T = q_i | \boldsymbol \lambda) = {\color{red}\sum_{i=1}^N \alpha_T(i)}
$$
我们希望找到一个表达 $\alpha_t(i)$ 和 $\alpha_{t+1}(i)$ 之间关系的迭代式：
$$
\begin{aligned}
	\alpha_{t+1}(i) 
	&= p(o_{1:t+1}, i_{t+1} = q_j | \boldsymbol \lambda) \\
	&= \sum_{i=1}^N p(o_{1:t+1}, i_{t+1}=q_j, {\color{blue}i_{t}=q_i} | \boldsymbol \lambda) \\
	&= \sum_{i=1}^N p(o_{t+1} | o_{1:t}, i_{t+1}=q_j,  {i_{t}=q_i} , \boldsymbol \lambda)\, p(o_{1:t}, i_t=q_i, i_{t+1}=q_j | \boldsymbol \lambda) \\
	&= \sum_{i=1}^N p(o_{t+1} | i_{t+1}=q_j, \boldsymbol \lambda)\, p(i_{t+1}=q_j | o_{1:t}, i_t=q_i, \boldsymbol \lambda)\, p(o_{1:t}, i_t=q_i | \boldsymbol \lambda) \\
	&= \sum_{i=1}^N p(o_{t+1} | i_{t+1}=q_j, \boldsymbol \lambda)\, p(i_{t+1}=q_j | i_t=q_i, \boldsymbol \lambda)\, p(o_{1:t}, i_t=q_i | \boldsymbol \lambda) \\
	&= \sum_{i=1}^N b_j(o_{t+1})\, a_{ij}\, {\color{red}\alpha_t(i)}
\end{aligned}
$$

$$
\therefore \quad {\color{red}\alpha_{t+1}(j)} = \left[\sum_{i=1}^N {\color{red}\alpha_t(i)}\, a_{ij}\right] b_j(o_{t+1})
$$

可以发现使用动态规划后，复杂度大大降低了，只有 $O(N^2T)$。

> **Forward algorithm**
>
> 1. 计算初始值：
>     $$
>     \alpha_1(i) = \pi_i b_i(o_1),\quad i=1:N
>     $$
>
> 2. 递推，对于 $t=2, \dots, T-1$ ，计算：
>     $$
>     \alpha_{t+1}(j) = \left[\sum_{i=1}^N {\alpha_t(i)}\, a_{ij}\right] b_j(o_{t+1}),\quad j=1:N
>     $$
>
> 3. 最后
>     $$
>     p(\mathbf O | \boldsymbol \lambda) = \sum_{i=1}^N \alpha_T(i)
>     $$





### Backward algorithm

由于条件独立性假设，后面的概率计算其实与前面相互独立，所以我们也可以从反方向使用动态规划

![HMM_beta](figures\HMM_beta.png)

记 $\beta_t(i) = p(o_{t+1:T} | i_t=q_i, \boldsymbol \lambda)$ ，于是 $\beta_1(i) = p(o_{2:T} | i_1=q_i, \boldsymbol \lambda)$ 。
$$
\begin{aligned}
	p(\mathbf O | \boldsymbol \lambda)
	&= p(o_{1:T} | \boldsymbol \lambda) \\
	&= \sum_{i=1}^N p(o_{1:T}, {\color{blue}i_1=q_i} | \boldsymbol \lambda) \\
	&= \sum_{i=1}^N p(o_{1:T} | {\color{blue}i_1=q_i}, \boldsymbol \lambda)\, \pi_i \\
	&= \sum_{i=1}^N p(o_1 | o_{2:T}, {i_1=q_i}, \boldsymbol \lambda)\, p(o_{2:T} | i_1=q_i, \boldsymbol \lambda)\, \pi_i \\
	&= \sum_{i=1}^N \pi_i b_i(o_1) {\color{red}\beta_1(i) }
\end{aligned}
$$
我们希望找到一个表达 $\beta_t+1(i)$ 和 $\beta_{t}(i)$ 之间关系的迭代式：
$$
\begin{aligned}
	\beta_t(i) &= p(o_{t+1:T} | i_t=q_i, \boldsymbol \lambda) \\
	&= \sum_{j=1}^N p(o_{t+1:T}, i_{t+1}=q_j | i_t=q_i, \boldsymbol \lambda) \\
	&= \sum_{j=1}^N \underbrace{p(o_{t+1:T} | i_{t+1}=q_j, i_t=q_i, \boldsymbol \lambda)}_{\color{orange}\text{(1) head to tail model}}\, p(i_{t+1}=q_j | i_t=q_i, \boldsymbol \lambda) \\
	&= \sum_{j=1}^N p(o_{t+1:T} | i_{t+1}=q_j, \boldsymbol \lambda)\, a_{ij} \\
	&= \sum_{j=1}^N p(o_{t+1} | o_{t+2:T}, i_{t+1}=q_j, \boldsymbol \lambda)\, p(o_{t+2:T} | i_{t+1}=q_j, \boldsymbol \lambda)\, a_{ij} \\
	&= \sum_{j=1}^N a_{ij}\, b_j(o_{t+1})\, {\color{red}\beta_{t+1}(j)}
\end{aligned}
$$

> **(1) head to tail model**
>
> <img src="figures\head2tail.png" alt="head2tail" style="zoom: 35%;" align='left' />
>
> 当 $b$ 被观测时，$a, c$ 条件独立，$p(c | a, b) = p(c | b)$ 。
>
> 证明：$p(o_{t+1:T} | i_{t+1}=q_j, i_t=q_i, \boldsymbol \lambda) = p(o_{t+1:T} | i_{t+1}=q_j, \boldsymbol \lambda)$
> $$
> \begin{aligned}
> 	p(o_{t+1:T} | i_{t+1}, i_t) &= \frac{p(o_{t+1:T}, i_t, i_{t+1})}{p(i_t, i_{t+1})} \\
> 	&= \frac{p(o_{t+1:T}, i_t | i_{t+1})\, \cancel{p(i_{t+1})}}{p(i_t | i_{t+1})\, \cancel{p(i_{t+1})}} \\
> 	&= \frac{p(o_{t+1:T} | i_{t+1})\, p(i_t | i_{t+1})}{p(i_t | i_{t+1})\, } \\
> 	&= p(o_{t+1:T} | i_{t+1})
> \end{aligned}

> $$


$$
\therefore \quad {\color{red}\beta_t(i)} = \sum_{j=1}^N a_{ij}\, b_j(o_{t+1})\, {\color{red}\beta_{t+1}(j)}
$$

> **Backward algorithm**
>
> 1. 计算初始值：
>     $$
>         
>     $$
>     
>
> 2. 递推，对于 $T-1, \dots, 1$​​ ，计算：
>     $$
>         
>     $$
>     
>
> 3. 最后：
>     $$
>         
>     $$
>     











## Learning







## Examples

假设我们在赌场中，观察一系列掷出骰子的点数 $y_t \in \{1, 2, \dots, 6\}$​。作为一名眼尖的统计学家，我们注意到点数的分布并不是我们对公平骰子的期望：似乎存在偶尔出现的“连续”，其中 6 似乎比其他值出现得更频繁。我们想估计底层状态，即骰子是公平的还是不公平的，这样我们就能对未来做出预测。

为了形式化，设 $z_t \in \{1, 2\}$ 表示时刻 $t$ 的未知隐藏状态（公平或欺骗），令 $y_t \in \{1, \dots, 6\}$ 表示观察到的结果（掷骰子）。设 $A_{jk} = p(z_t=k | z_{t-1}=j)$ 为状态转移矩阵。大多数时候，赌场使用一个公平的骰子，$z=1$ ，但偶尔也会在短时间内切换到一个欺骗的骰子，$z=2 $，状态转换如图所示。设 $B_{kl} = p(y_t=l | z_t=k)$ 为观测矩阵。如果 $z=1$ ，则观测分布是均匀分类分布，如果 $z=2$ ，则观测分布向 6 倾斜，即
$$
p(y_t | z_t = 1) = \mathrm{Cat}(y_t | [1/6, \dots, 1/6]) \\
p(y_t | z_t = 2) = \mathrm{Cat}(y_t | [1/10, 1/10, \dots, 5/10])
$$
<img src="figures\example_1.png" alt="example_1" style="zoom: 40%;" />

如果我们从这个模型中抽样，我们可能会得到如下数据

- hidden: [ 0 0 1 1 1 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 ]
- observation: [ 3 2 5 3 3 2 0 4 5 0 2 5 5 5 5 5 3 4 5 1 ]











## Extension

观测模型 $p(\boldsymbol y_t | z_t = j)$​ 可以采用多种形式，具体取决于数据类型。对于上述离散观测，可以使用
$$
p(y_t=k | z_t=j) = b_{jk}
$$
如果每个时间步长有 $D$ 个离散观测值，可以使用以下形式的因子模型：
$$
p(\boldsymbol y_t | z_t=j) = \prod_{d=1}^D \mathrm{Cat}(y_{td} | )
$$






### Gaussian likelihood

#### Gaussian HMM

如果 $\boldsymbol y_t$​ 是连续的，通常使用高斯观测模型：
$$
p(\boldsymbol y_t | z_t = j, \boldsymbol \theta) = \mathcal N(\boldsymbol y_t | \boldsymbol \mu_j, \boldsymbol \Sigma_j)
$$
其中发射参数 $\boldsymbol \theta = \{(\boldsymbol \mu_j, \boldsymbol \Sigma_j)\}_{j=1}^N$ 包括 $N$ 个离散状态的均值和协方差。

举一个简单的例子，假设我们有一个具有 5 个隐藏状态的 HMM，每个隐藏状态都会生成一个 2d 高斯分布。可以将这些高斯分布表示为 2d 椭圆，如下图所示。



#### GMM-HMM

我们还可以使用更灵活的观测模型。例如，如果我们使用 $M$-分量 GMM，那么有：
$$
p(\boldsymbol y_t | z_t = j, \boldsymbol \theta) = \sum_{k=1}^M w_{jk}\, \mathcal N(\boldsymbol y_t | \boldsymbol \mu_{jk}, \boldsymbol \Sigma_{jk})
$$
这被称为 **GMM-HMM**。





