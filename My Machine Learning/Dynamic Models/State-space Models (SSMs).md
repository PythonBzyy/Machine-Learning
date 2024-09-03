# State-space Models (SSMs)

## Introduction

状态空间模型 (SSM) 是一种部分观测的马尔可夫模型，其中的隐状态 $\boldsymbol z_t$ 根据马尔可夫过程随时间演变，并且每个隐藏状态在每个时间步生成一些观测值 $\boldsymbol y_t$ 。

<img src="C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240503143354730.png" alt="image-20240503143354730" style="zoom:67%;" />

我们可以将它们视为具有条件独立性的潜变量序列模型，如上图所示。相应的联合分布具有如下形式：
$$
\begin{aligned}
	p(\boldsymbol y_{1:T}, \boldsymbol z_{1:T} | \boldsymbol u_{1:T}) = \left[p(\boldsymbol z_1 | \boldsymbol u_1) \prod_{t=2}^T p(\boldsymbol z_t | \boldsymbol z_{t-1}, \boldsymbol u_t) \right] \left[\prod_{t=1}^T p(\boldsymbol y_t | \boldsymbol z_t, \boldsymbol u_t)\right]
\end{aligned}
$$
其中，$\boldsymbol z_t$ 是时间 $t$ 的隐变量，$\boldsymbol y_t$ 是观测值（输出），$\boldsymbol u_t$ 是可选输入。$p(\boldsymbol z_t | \boldsymbol z_{t-1}, \boldsymbol u_t)$ 称为动态模型 (dynamics model) 或转移模型 (transition model) ，$p(\boldsymbol y_t | \boldsymbol z_t, \boldsymbol u_t)$ 称为观测模型 (observation model) 或测量模型 (measurement model) ，$p(\boldsymbol z_1 | \boldsymbol u_1)$ 是先验或初始状态分布。



### Bayesian Filtering Equations

贝叶斯滤波器是一种算法，用于在给定上一步的先验信念 $p(\boldsymbol z_{t-1} | \boldsymbol y_{1:t-1})$ ，新的观测值 $\boldsymbol y_t$ 和模型时，递归计算信念状态 (belief state) $p(\boldsymbol z_t | \boldsymbol y_{1:t})$ 。这可以使用**顺序贝叶斯更新 (sequential Bayerian updating)** 来完成。并且每个时间步需要恒定的计算量 (与 $t$ 无关) 。对于动态模型，可以简化为一下所描述的 **predict-update** 周期。

- **Prediction step:** 预测步就是 **Chapman-Kolmogorov 方程**
    $$
    \begin{aligned}
    	\underbrace{p(\boldsymbol z_t | \boldsymbol y_{1:t-1})}_{\text{prior at $t$}} = \int p(\boldsymbol z_t | \boldsymbol z_{t-1})\, \underbrace{p(\boldsymbol z_{t-1} | \boldsymbol y_{1:t-1})}_{\text{posterior at $t-1$}}\, \mathrm{d}\boldsymbol z_{t-1}
    \end{aligned}
    $$
    预测步骤计算潜在状态的一步预测分布，将前一个时间步的后验更新为当前步骤的先验

- **Update step:** 更新步就是贝叶斯规则
    $$
    \begin{aligned}
    	p(\boldsymbol z_t | \boldsymbol y_{1:t}) = \frac{1}{Z_t} p(\boldsymbol y_t | \boldsymbol z_t)\, p(\boldsymbol z_t | \boldsymbol y_{1:t-1})
    \end{aligned}
    $$
    其中 $Z_t$ 是归一化常数，$Z_t = \displaystyle\int p(\boldsymbol y_t | \boldsymbol z_t)\, p(\boldsymbol z_t | \boldsymbol y_{1:t-1})\, \mathrm{d} \boldsymbol z_t = p(\boldsymbol y_t | \boldsymbol y_{1:t-1})$ 。

    可以使用归一化常数来计算序列的对数似然：
    $$
    \begin{aligned}
    	\log p(\boldsymbol y_{1:T}) = \sum_{t=1}^T \log p(\boldsymbol y_t | \boldsymbol y_{1:t-1}) = \sum_{t=1}^T \log Z_t
    \end{aligned}
    $$
    其中定义 $p(\boldsymbol y_1 | \boldsymbol y_0) = p(\boldsymbol y_1)$ ，这对于计算参数的MLE很有用。



