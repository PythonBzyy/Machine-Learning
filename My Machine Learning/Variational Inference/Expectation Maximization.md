# Expectation Maximization



## Introduction

**期望最大算法（Expectation Maximization，EM）**旨在计算具有缺失数据或隐藏变量的概率模型的 MLE 或 MAP 参数估计值。它是 MM 算法的一个特例。 EM 的基本思想是在 E 步骤（期望步骤）期间交替估计隐藏变量（或缺失值），然后在 M 步骤（最大化步骤）期间使用完全观察到的数据（$\{\boldsymbol y, \boldsymbol z | \boldsymbol \theta\}$）计算 MLE。

> ==TODO==
> 边界优化或 MM 算法的算法。 在最小化的背景下，MM 代表最大化化-最小化。在最大化的背景下，MM 代表最小化-最大化。



为什么要使用EM，遇到什么问题（==TODO==）



描述一个通用模型，$\boldsymbol y_i$ 为观测数据（observed），$\boldsymbol z_i$ 为隐藏数据（latent）。直接写出**参数 $\boldsymbol \theta$ 的更新（迭代）公式**：
$$
\begin{aligned}
	\boldsymbol \theta^{t+1} &= \arg \max_{\boldsymbol \theta} \int_{\boldsymbol z} \log \left[p(\boldsymbol y, \boldsymbol z | \boldsymbol \theta)\right]\, \underbrace{p(\boldsymbol z | \boldsymbol y, \boldsymbol \theta^t)}_{q(\boldsymbol z_i | \boldsymbol y_i, \boldsymbol \theta^t) = q_i^t}\, \mathrm{d} \boldsymbol z \\
	&= \arg \max_{\boldsymbol \theta} \sum_{i} \mathbb E_{q_i^t} [\log p(\boldsymbol y_i, \boldsymbol z_i | \boldsymbol \theta)]
\end{aligned}
$$

> 隐变量 $\boldsymbol z$ 的加入，不改变观测数据 $\boldsymbol y$ 的边缘分布 $p(\boldsymbol y)$，同时可以有效解决问题。

---



## Lower bound

EM 的目标是**最大化**观测数据的对数似然（log-likelihood）：
$$
\ell(\boldsymbol \theta) = \sum_{i=1}^N \log p(\boldsymbol y_i | \boldsymbol \theta) = \sum_{i=1}^N \log \sum_{\boldsymbol z_i} p(\boldsymbol y_i, \boldsymbol z_i | \boldsymbol \theta)
$$
这很难优化，$\log$ 无法放到 $\Sigma$ 中。

于是 EM 解决这个问题的方法如下：首先考虑每个隐变量 $\boldsymbol z_i$ 上的一组任意分布 $q_i(\boldsymbol z_i)$ 。
$$
\ell(\boldsymbol \theta) = \sum_{i=1}^N \log \left[\sum_{\boldsymbol z_i} q_i(\boldsymbol z_i) \frac{p(\boldsymbol y_i, \boldsymbol z_i | \boldsymbol \theta)}{q_i(\boldsymbol z_i)} \right]
$$
利用 Jensen 不等式，将 $\log$ （凹函数）推入期望内，将得到 log-likelihood 的下界（ELBO）：
$$
\begin{aligned}
	\ell(\boldsymbol \theta) 
	&\geq \sum_{i} \sum_{\boldsymbol z_i} q_i(\boldsymbol z_i) \log \frac{p(\boldsymbol y_i, \boldsymbol z_i | \boldsymbol \theta)}{q_i(\boldsymbol z_i)} \\
	&= \sum_{i} \underbrace{\big[\mathbb E_{q_i(\boldsymbol z_i)} \left[\log p(\boldsymbol y_i, \boldsymbol z_i | \boldsymbol \theta) \right] + \mathbb H[q_i(\boldsymbol z_i)]\big]}_{\color{blue}\text{ELBO}} \\
	&= \sum_{i} \mathcal L(\boldsymbol \theta, q_i | \boldsymbol y_i) \\
	&\triangleq \mathcal {\boldsymbol L}(\boldsymbol \theta, q_{1:N} | \mathcal D) \longrightarrow {\color{blue}\text{ELBO}}
\end{aligned}
$$
其中 $\mathbb H(q)$ 是概率分布 $q$ 的熵，$\mathcal L(\boldsymbol \theta, q_{1:N} | \mathcal D)$ 称为**证据下限（evidence lower bound，ELBO）**，因为它是对数边际似然 $\log p(\boldsymbol y_{1:N} | \boldsymbol \theta)$ 的下限，也称为证据。*优化这个界限也是 **[变分推断](./Variational Inference.md)** 的基础。



---



## E step

ELBO是 $N$ 项的和，每一项都有如下形式：
$$
\begin{aligned}
	\mathcal L(\boldsymbol \theta, q_i | \boldsymbol y_i) 
	&= \sum_{\boldsymbol z_i} q_i(\boldsymbol z_i) \log \frac{p(\boldsymbol y_i, \boldsymbol z_i | \boldsymbol \theta)}{q_i(\boldsymbol z_i)} \\
	&= \sum_{\boldsymbol z_i} q_i(\boldsymbol z_i) \log \frac{p(\boldsymbol z_i | \boldsymbol y_i, \boldsymbol \theta)\, p(\boldsymbol y_i | \boldsymbol \theta)}{q_i(\boldsymbol z_i)} \\
	&= \sum_{\boldsymbol z_i} q_i(\boldsymbol z_i) \log \frac{p(\boldsymbol z_i | \boldsymbol y_i, \boldsymbol \theta)}{q_i(\boldsymbol z_i)} + \sum_{\boldsymbol z_i} q_i(\boldsymbol z_i) \log p(\boldsymbol y_i | \boldsymbol \theta) \\
	&= - \mathbb{KL}\left[q_i(\boldsymbol z_i) \parallel p(\boldsymbol z_i | \boldsymbol y_i, \boldsymbol \theta)\right] + \log p(\boldsymbol y_i | \boldsymbol \theta)
\end{aligned}
$$

$$
\therefore \quad \log p(\boldsymbol y_i | \boldsymbol \theta) = \mathcal L(\boldsymbol \theta, q_i | \boldsymbol y_i) + \mathbb{KL}\left[q_i(\boldsymbol z_i) \parallel p(\boldsymbol z_i | \boldsymbol y_i, \boldsymbol \theta)\right] \\
\log p(\boldsymbol y | \boldsymbol \theta) = \text{ELBO} + \mathbb{KL}\left[q(\boldsymbol z) \parallel p(\boldsymbol z | \boldsymbol y, \boldsymbol \theta)\right]
$$

其中 $\mathbb{KL}(q \parallel p) \geq 0$ 且 $\mathbb{KL}(q \parallel p) = 0$ 当且仅当 $q = p$ 。因此可以最大化 ELBO，令 $\color{red}q^*_i = p(\boldsymbol z_i | \boldsymbol y_i, \boldsymbol \theta)$ ，从而确保 ELBO 是一个严格的下界：
$$
\mathcal{\boldsymbol L}(\boldsymbol \theta, q_{1:N}^* | \mathcal D) = \sum_{i}\log p(\boldsymbol y_i | \boldsymbol \theta) = \ell(\boldsymbol \theta)
$$


与边界优化关联，定义：
$$
Q({\color{blue}\boldsymbol \theta}, {\color{green}\boldsymbol \theta^t}) = \mathcal {\boldsymbol L}({\color{blue}\boldsymbol \theta},\{p(\boldsymbol z_i | \boldsymbol y_i, {\color{green}\boldsymbol \theta^t}) | i=1:N\})
$$
其中 $\boldsymbol \theta$ 是未知的，$\boldsymbol \theta^t$ 是已知的，
$$
\therefore \quad Q(\boldsymbol \theta, \boldsymbol \theta^t) \leq \ell(\boldsymbol \theta), \quad Q(\boldsymbol \theta^t, \boldsymbol \theta^t) = \ell(\boldsymbol \theta^t)
$$
如果无法计算 $p(\boldsymbol z_i | \boldsymbol y_i, \boldsymbol \theta^t)$ ，则可以使用*近似分布* $q(\boldsymbol z_i | \boldsymbol y_i, \boldsymbol \theta^t)$ ，这将会产生对数似然的非紧密下界，这种广义版的EM称为**变分EM（Variational EM）**。

---



## M step

在 M-step 中，需要最大化 $\mathcal {\boldsymbol L}(\boldsymbol \theta, q_{1:N}^t)$ ，其中 $q_i^t$ 是 E-step 在 $t$ 迭代计算得到的分布。熵是常数，可以丢弃。
$$
\begin{aligned}
	\ell^t(\boldsymbol \theta)
	&= \sum_{i} \mathbb E_{q_i^t(\boldsymbol z_i)} [\log p(\boldsymbol y_i, \boldsymbol z_i | \boldsymbol \theta)] \\
	&\longrightarrow \text{expected complete data log likelihood}
\end{aligned}
$$
这称为完整数据对数似然期望（expected complete data log likelihood）。

在 M-step 中，最大化完整数据对数似然期望，得到
$$
\boldsymbol \theta^{t+1} = \arg \max_{\boldsymbol \theta} \sum_{i}\mathbb E_{q_i^t(\boldsymbol z_i)} [\log p(\boldsymbol y_i, \boldsymbol z_i | \boldsymbol \theta)]
$$