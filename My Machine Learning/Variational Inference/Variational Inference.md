# Variational Inference

[TOC]

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

物理学观点：最小化**变分自由能（variational free energy，VFE）**。

定义 $\mathcal E_{\boldsymbol \theta}(\boldsymbol z) = -\log p_{\boldsymbol\theta}(\boldsymbol x, \boldsymbol z)$ 为能量，则可以重写 $\mathcal L(\boldsymbol\theta, \boldsymbol\psi | \boldsymbol x)$
$$
\begin{aligned}
	\mathcal L(\boldsymbol\theta, \boldsymbol\psi | \boldsymbol x) 
	&= \mathbb E_{q_{\boldsymbol\psi}(\boldsymbol z)}[\mathcal E_{\boldsymbol \theta}(\boldsymbol z)] - \mathbb H[q_{\boldsymbol\psi}] \\
	&= \text{expected energy − entropy}
\end{aligned}
$$
在物理学中，这称为变分自由能（variational free energy，VFE），是自由能（free energy，FE，$-\log p_{\boldsymbol\theta}(\boldsymbol x)$）的上限，因为
$$
\mathbb{KL}[q_{\boldsymbol\psi}(\boldsymbol z) \parallel p_{\boldsymbol\theta}(\boldsymbol z | \boldsymbol x)] = \mathcal L(\boldsymbol\theta, \boldsymbol\psi | \boldsymbol x) + \log p_{\boldsymbol \theta}(\boldsymbol x) \geq 0 \\
\therefore\quad \underbrace{\mathcal L(\boldsymbol\theta, \boldsymbol\psi | \boldsymbol x)}_{\text{VFE}} \geq \underbrace{- \log p_{\boldsymbol \theta}(\boldsymbol x)}_{\text{FE}}
$$
VI 相当于最小化 VFE，如果达到 $-\log p_{\boldsymbol\theta}(\boldsymbol x)$ 的最小值，那么 $\mathbb{KL}=0$ ，则近似后验是精确的。



### View 3: statistics

==TODO==







## Relationship with EM

*回顾 EM 算法推导，[link](./Expectation Maximization.md)。

> **NOTE**
> 在一些推导中将参数和隐变量合并到一个变量 $\boldsymbol z$ 中，其实参数也可以看作隐变量，只是狭义上理解的隐变量会随着观测数据的规模递增，而参数不会。在这里会拆分两者，其中  $\boldsymbol z$ 表示隐变量（狭义），$\boldsymbol \theta$ 表示参数。

上述 VI 框架可以表示如下：
$$
\log p(\boldsymbol x | \boldsymbol \theta) = \mathcal L(q, \boldsymbol \theta) + \mathbb{KL}(q \parallel p)
$$
其中
$$
\begin{aligned}
	\mathcal L(q, \boldsymbol \theta) &= \int_{\boldsymbol z} q(\boldsymbol z) \log \frac{p(\boldsymbol x, \boldsymbol z | \boldsymbol \theta)}{q(\boldsymbol z)}\,\mathrm{d}\boldsymbol z \\
	\mathbb{KL}(q \parallel p) &= -\int_{\boldsymbol z} q(\boldsymbol z) \log \frac{p(\boldsymbol z | \boldsymbol x, \boldsymbol \theta)}{q(\boldsymbol z)}\,\mathrm{d}\boldsymbol z
\end{aligned}
$$
EM 中的目标是最大化 $\log p(\boldsymbol x | \boldsymbol \theta)$ ，使 $\log p(\boldsymbol x | \boldsymbol \theta^{t+1}) \geq \log p(\boldsymbol x | \boldsymbol \theta^{t})$ 。可以先固定参数 $\boldsymbol\theta$ ，找到一个最优的近似分布 $q^*(\boldsymbol z)$ 使得 $\mathcal L(q^*, \boldsymbol \theta)$ 达到最大，即 $\mathbb{KL}$ 最小，$q^*(\boldsymbol z)=p(\boldsymbol z | \boldsymbol x, \boldsymbol \theta)$ 。之后固定 $q^*(\boldsymbol z)$ ，寻找最优 $\boldsymbol\theta$ 使得 $\mathcal L(q^*, \boldsymbol \theta)$ 最大。以此迭代。

- **E-step：**
    目的是更新 $q(\boldsymbol z)$ ，于是令 $q(\boldsymbol z) = p(\boldsymbol z | \boldsymbol x, \boldsymbol\theta^{t})$ ，使 $\mathbb{KL}=0$ 。
- **M-step：**
    目的是更新模型参数 $\boldsymbol\theta$ 使 $\mathcal L(\boldsymbol q, \boldsymbol \theta^{t+1}) \geq \mathcal L(\boldsymbol q, \boldsymbol \theta^{t})$，进而使 $\log p(\boldsymbol x | \boldsymbol \theta^{t+1}) \geq \log p(\boldsymbol x | \boldsymbol \theta^{t})$ 。（这里最大化 ELBO 的最优解和最大化 $\mathcal Q$ 函数相同，两者相差常数 $\mathbb{H}$ 。）更新后 $\mathbb{KL}$ 中的 $q(\boldsymbol z)$ 还固定为 $p(\boldsymbol z | \boldsymbol x, \boldsymbol \theta^t)$ ，而 $p(\boldsymbol z | \boldsymbol x, \boldsymbol \theta)$ 已经更新为 $p(\boldsymbol z | \boldsymbol x, \boldsymbol \theta^{t+1})$ ，因此 $\mathbb{KL}(q \parallel p) \geq 0$ 。

通过 $T$ 次迭代使得对数边际似然函数 $\log p(\boldsymbol x, \boldsymbol \theta)$ 不断增大直至收敛，从而得到最终模型的参数 $\boldsymbol \theta^T$ 。上述过程可以描述为：

<img src="figures\EM_steps.png" alt="EM_steps" style="zoom:80%;" />



那么 VI 和 EM 的差别：

- VI 求一个 $q(\boldsymbol z)$ 使得 $q(\boldsymbol z) \rightarrow p(\boldsymbol z | \boldsymbol x)$ ，但是不会影响 $\log p(\boldsymbol x)$ 。这可以是在 E-step 中发生，但是不会涉及 M-step 。
- EM 求一个最优参数 $\boldsymbol \theta^*$ 使得 $\log p(\boldsymbol x | \boldsymbol \theta^*)$ 尽可能大。



---





## Learning

[**Cooridinate Ascent VI, CAVI**](./Cooridinate Ascent VI.md)

[**Gradient-based VI, GVI**](./Gradient-based VI.md)



















