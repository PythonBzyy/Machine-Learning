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

