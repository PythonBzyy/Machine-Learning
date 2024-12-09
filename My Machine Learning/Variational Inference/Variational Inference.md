# Variational Inference

## Introduction

参数估计，从频率派角度是优化问题，从贝叶斯角度是**积分问题**。参数后验为：
$$
p(\boldsymbol \theta | \boldsymbol x) = \frac{p(\boldsymbol x | \boldsymbol \theta)\, p(\boldsymbol \theta)}{p(\boldsymbol x)}
$$
令 $\boldsymbol x_*$ 为新样本，求 $p(\boldsymbol x_* | \boldsymbol x)$
$$
\begin{aligned}
	p(\boldsymbol x_* | \boldsymbol x)
	&= \int_{\boldsymbol \theta} p(\boldsymbol x_*, \boldsymbol \theta | \boldsymbol x)\,\mathrm{d}\boldsymbol \theta \\
	&= \int_{\boldsymbol \theta} p(\boldsymbol x_* | \boldsymbol\theta, \boldsymbol x)\, \underbrace{p(\boldsymbol \theta | \boldsymbol x)}_{\text{posterior}}\,\mathrm{d}\boldsymbol \theta \\
	&= \mathbb E_{p(\boldsymbol \theta | \boldsymbol x)} [p(\boldsymbol x_* | \boldsymbol \theta)]
\end{aligned}
$$
所以在贝叶斯框架中，关键是求后验分布 $p(\boldsymbol \theta | \boldsymbol x)$​ ，然后求积分。求后验的过程也是 Inference 所关注的。

**Inference：**

- 精确推断

- 近似推断，参数空间无法精确求解

    - 确定性近似：VI
    - 随机近似：MCMC，MH，Gibbs

    

---



## Object

已知 $\boldsymbol x$ 为观测数据，$\boldsymbol z$ 为隐变量，$\boldsymbol \theta$ 为固定参数。
假设先验为 $p_{\theta}(\boldsymbol z)$ ，似然为 $p_{\theta}(\boldsymbol x | \boldsymbol z)$ ，于是非归一化的联合发分布为 $p_{\theta}(\boldsymbol x, \boldsymbol z) = p_{\theta}(\boldsymbol x | \boldsymbol z)\, p_{\theta}(\boldsymbol z)$ ，后验为 $p_{\theta}(\boldsymbol z | \boldsymbol x) = p_{\theta}(\boldsymbol z, \boldsymbol x) / p_{\theta}(\boldsymbol x)$ 。假设无法计算归一化常数 $p_{\theta}(\boldsymbol x) = \int_{\boldsymbol z}p_{\theta}(\boldsymbol x, \boldsymbol z)\,\mathrm{d}\boldsymbol z$ ，那么也无法计算归一化后验。于是，需要求后验的近似值 $q(\boldsymbol z)$ ：
$$
q = \arg\min_{q \in \mathcal Q} \mathbb{KL}(q(\boldsymbol z) \parallel p_{\theta}(\boldsymbol z | \boldsymbol x))
$$
由于要最小化函数（即分布 $q$），因此这称为**变分方法**。

实践中选择一个参数族 $\mathcal Q$ ：使用 $\boldsymbol\psi$ 来作为变分参数，并计算最优变分参数：
$$
\begin{aligned}
	\boldsymbol\psi &= \arg\min_{\boldsymbol\psi} \mathbb{KL}[q_{\psi}(\boldsymbol z) \parallel p_{\theta}(\boldsymbol z | \boldsymbol x)] \\
    &= \arg\min_{\boldsymbol\psi} \mathbb E_{q_{\boldsymbol\psi}(\boldsymbol z)}\left[\log q_{\boldsymbol \psi}(\boldsymbol z) - \log\frac{p_{\boldsymbol \theta}(\boldsymbol x | \boldsymbol z)\, p_{\boldsymbol \theta}(\boldsymbol z)}{p_{\boldsymbol \theta}(\boldsymbol x)} \right] \\
    &= \arg\min_{\boldsymbol\psi} \underbrace{\mathbb E_{q_{\boldsymbol\psi}(\boldsymbol z)}\left[\log q_{\boldsymbol \psi}(\boldsymbol z) - \log p_{\boldsymbol \theta}(\boldsymbol x | \boldsymbol z) +\log p_{\boldsymbol \theta}(\boldsymbol z) \right]}_{\mathcal L(\boldsymbol \theta, \boldsymbol \psi | \boldsymbol x)} + \log {p_{\boldsymbol \theta}(\boldsymbol x)}
\end{aligned}
$$
最后一项通常难以计算，但是与 $\boldsymbol \psi$ 无关，于是
$$
\mathcal L(\boldsymbol \theta, \boldsymbol \psi | \boldsymbol x) = \underbrace{\mathbb E_{q_{\boldsymbol \psi}(\boldsymbol z)} \left[\log q_{\boldsymbol \psi}(\boldsymbol z) - \log p_{\boldsymbol \theta}(\boldsymbol x, \boldsymbol z) \right]}_{\color{blue}\mathrm{-ELBO}}
$$
最小化这个目标即最小化 KL 散度，使近似分布 $q_{\boldsymbol \psi}(\boldsymbol z)$ 接近真实后验 $p_{\boldsymbol \theta}(\boldsymbol z | \boldsymbol x)$​ 。



---



接下来给出这个目标的几种解释。

### View 1

在 EM 算法中，
$$
\begin{aligned}
	\log p(\boldsymbol x) &= \log \int_{\boldsymbol z} p(\boldsymbol x, \boldsymbol z) \\
	&= \log \int_{\boldsymbol z} q(\boldsymbol z) \frac{p( \boldsymbol x, \boldsymbol z)}{q(\boldsymbol z)} \\
	&= \log \mathbb E_{q(\boldsymbol z)}\left[\frac{p(\boldsymbol x, \boldsymbol z)}{q(\boldsymbol z)}\right] \\
	&\geq \mathbb E_{q(\boldsymbol z)}\left[\log \frac{p(\boldsymbol x, \boldsymbol z)}{q(\boldsymbol z)} \right] \qquad \text{by Jensen’s inequality} \\
	&= \underbrace{\mathbb E_{q(\boldsymbol z)}\left[\log p(\boldsymbol x, \boldsymbol z)\right] + \mathbb H \left[q(\boldsymbol z)\right]}_{\color{blue}\text{ELBO}}
\end{aligned}
$$
并且
$$
\begin{aligned}
	\log p(\boldsymbol x) &= \text{ELBO} + \mathbb{KL}(q(\boldsymbol z) \parallel p(\boldsymbol z | \boldsymbol x)) \\
	&= \underbrace{\mathcal L(q)}_{\text{variation}} + \underbrace{\mathbb{KL}(q(\boldsymbol z) \parallel p(\boldsymbol z | \boldsymbol x))}_{\geq 0}
\end{aligned}
$$
式中 $\log p(\boldsymbol x)$ 与 $q$ 无关，并且当 $\boldsymbol x$ 给定时，左侧为一常数。我们需要找到一个 $q(\boldsymbol z)$ 近似 $p(\boldsymbol z | \boldsymbol x)$ ，此时 $\mathbb{KL}$ 最小，同时 $\mathcal L(q)$ 最大。于是，我们的最终目标为：
$$
\begin{aligned}
	q^*(\boldsymbol z) &= \arg\max_{q(\boldsymbol z)} \mathcal L(q) \rightarrow \text{ELBO} \\
	&= \arg\max_{q(\boldsymbol z)} \int_{\boldsymbol z} q(\boldsymbol z) \log \frac{p(\boldsymbol x, \boldsymbol z)}{q(\boldsymbol z)}\,\mathrm{d}\boldsymbol z 
\end{aligned}
$$
下图描述了 $\log p(\boldsymbol x)$ ，$\mathcal L$ 和 $\mathbb{KL}$ 之间的关系。

<img src="figures\VI_ELBO.png" alt="VI_ELBO" style="zoom:30%;" />





### View 2: physics

==TODO==





### View 3: statistics

==TODO==







## 







## Coordinate ascent VI

坐标上升VI（Coordinate ascent VI），基于平均场近似（mean field approximation）。

