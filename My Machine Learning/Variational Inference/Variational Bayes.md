## Variational Bayes

[TOC]



### Introduction

在 VI 中，我们一直关注于潜在变量 $\boldsymbol z_i$ ，并假设模型参数 $\boldsymbol\theta$ 已知（固定），现在将参数视为潜在变量，因此目的是推断参数本身，即推断近似参数后验。
$$
p(\boldsymbol\theta | \mathcal D) \propto p(\mathcal D | \boldsymbol\theta)\, p(\boldsymbol\theta)
$$
将 VI 应用在此类问题称为**变分贝叶斯（Variational Bayes，VB）**。

假设除了共享的全局参数之外没有其他潜在变量，因此模型具有以下形式
$$
p(\boldsymbol{\theta},\mathcal{D})=p(\boldsymbol{\theta})\prod_{i=1}^Np(\mathcal{D}_i|\boldsymbol{\theta})
$$
条件独立性图示为：

<img src="figures\VB.png" alt="VB" style="zoom:30%;" />

通过最大化 ELBO 来拟合变分后验，
$$
\mathcal L(\boldsymbol\psi_{\boldsymbol\theta} | \boldsymbol x) = \mathbb{E}_{q(\boldsymbol\theta | \boldsymbol\psi_{\boldsymbol\theta})} [\log p(\boldsymbol x, \boldsymbol\theta)] + \mathbb{H}[q(\boldsymbol\theta | \boldsymbol\psi_{\boldsymbol\theta})] 
$$
假设变分后验参数的因子分解（基于平均场）为
$$
q(\boldsymbol\theta | \boldsymbol\psi_{\boldsymbol\theta}) = \prod_{j} q(\boldsymbol\theta_j | \boldsymbol\psi_{\boldsymbol\theta_j})
$$
于是可以使用 CAVI 来更新每一个 $\boldsymbol\theta_j$ 。





### Example: VB for a univariate Gaussian

[Code & Notebook]()

#### Inference

我们的目的是估计一维高斯函数的参数后验 $p(\mu, \tau | \mathcal D)$ ，其中 $\tau = \sigma^{-2}$ 是精度。假设有 $N$ 个观测数据 $\mathcal D = \mathbf X = \{x_1, \dots, x_N\}$ ，于是似然为
$$
\begin{aligned}
	{\color{blue}\text{linklihood}} \longrightarrow p(\mathcal D | \boldsymbol\theta) &= \prod_{i=1}^N \mathcal N(x_i | \mu, \tau^{-1}) \\
	&= \prod_{i=1}^N (\frac{\tau}{2 \pi})^{1/2} \exp\bigg\{-\frac{\tau}{2} (x_i - \mu)^2 \bigg\} \\
     &= (\frac{\tau}{2 \pi})^{N / 2} \exp\bigg\{- \frac{\tau}{2} \sum_{i=1}^N (x_i - \mu)^2\bigg\} 
\end{aligned}
$$
引入他的共轭先验
$$
{\color{green}\text{prior}} \longrightarrow p(\mu, \tau) = \mathcal N(\mu | \mu_0, (\lambda_0 \tau)^{-1})\, \mathrm{Ga}(\tau | a_0, b_0) \\
\begin{cases}
	p(\mu | \tau) = N(\mu | \mu_0, (\lambda_0 \tau)^{-1}) &\propto \exp\big\{\frac{-\lambda_0 \tau}{2} (\mu - \mu_0)^2 \big\} \\
	p(\tau) = \mathrm{Ga}(\tau | a_0, b_0) & \propto \tau^{a_0 -1} \exp\big\{-b_0 \tau \big\}
\end{cases}
$$
其中 $\mu$ 和 $\tau$ 不独立。由于共轭性，可以精确推导出该模型的后验 $p(\mu, \tau | \mathcal D)$
$$
\begin{aligned}
    {\color{red}\text{posterior}} \longrightarrow p(\mu, \tau | \mathcal D) &= \frac{p(\mathcal D | \mu, \tau) \, p(\mu | \tau) \, p(\tau)}{p(\mathcal D)} \\
    & \propto p(\mathcal D | \mu, \tau) \, p(\mu | \tau) \, p(\tau) \\
    &= \underbrace{\mathcal N(\mu_N, (\lambda_N \tau)^{-1})\,\text{Ga} (\tau | a_N, b_N)}_{\mu_N, \lambda_N, a_N, b_N}
\end{aligned}
$$
可以看到，一共有4个参数：$\mu_N, \lambda_N, a_N, b_N$ 。注意，这里是有后验的解析解的：
$$
\begin{aligned}
    \mu_N &= \frac{\lambda_0 \mu_0 + N \bar{x}}{\lambda_0 + N} \\
    \lambda_N &= \lambda_0 + N \\
    a_N &= a_0 + N / 2 \\
    b_N &= b_0 + \frac{1}{2} \sum_{i=1}^N (x_i - \bar{x})^2 + \frac{\lambda_0 N (\bar{x} - \mu_0)^2}{2 (\lambda_0 + N)}
\end{aligned}
$$

> **参数精确解推导**
> $$
> \begin{aligned}
>     p(\mu, \tau | \mathbf X) &\propto p(\mathbf X | \mu, \tau) \, p(\mu | \tau) \, p(\tau) \\
>     &\propto (\frac{\tau}{2 \pi})^{N/2} \exp\bigg\{-\frac{\tau}{2} \sum_{i=1}^N (x_i - \mu)^2 \bigg\} \, (\frac{\lambda_0 \tau}{2 \pi})^{1/2} \exp\big\{-\frac{\lambda_0 \tau}{2} (\mu - \mu_0)^2 \big\} \, \tau^{a_0 - 1} \exp\{ -b_0 \tau\} \\
>     &\propto \tau^{(N+1)/2 + a_0 -1} \exp\{-b_0 \tau \} \, \exp\bigg\{-\frac{\tau}{2} \sum_{i=1}^N (x_i - \mu)^2 - \frac{\lambda_0 \tau}{2} (\mu - \mu_0)^2 \bigg\} \\
>     &\propto \tau^{(N+1)/2 + a_0 -1} \exp\{-b_0 \tau \} \, \exp\bigg\{-\frac{\tau}{2} \sum_{i=1}^N x_i^2 \bigg\} \, \exp\bigg\{-\frac{\lambda_0 \mu_0^2 \tau}{2} \bigg\} \, \exp\bigg\{-\frac{(\lambda_0 + N) \tau}{2} \big(\mu - \frac{\sum_{i=1}^N x_i + \lambda_0 \mu_0}{\lambda_0 + N} \big)^2 \bigg\} \\
>     &\propto \tau^{(N+1)/2 + a_0 -1} \exp\bigg\{- \big(b_0 + \frac{1}{2} \sum_{i=1}^N x_i^2 + \frac{\lambda_0 \mu_0^2}{2} \big) \tau \bigg\} \, \exp\bigg\{- \frac{(\lambda_0 + N) \tau}{2} \big(\mu - \frac{\sum_{i=1}^N x_i + \lambda_0 \mu_0}{\lambda_0 + N} \big)^2 \bigg\}
> \end{aligned}
> $$
> 于是可以发现，前半部分 $\tau^{(N+1)/2 + a_0 -1} \exp\left\{- \big(b_0 + \frac{1}{2} \sum_{i=1}^N x_i^2 + \frac{\lambda_0 \mu_0^2}{2} \big) \tau \right\}$ 是 Gamma 分布的形式，后半部分 $\exp\left\{- \frac{(\lambda_0 + N) \tau}{2} \big(\mu - \frac{\sum_{i=1}^N x_i + \lambda_0 \mu_0}{\lambda_0 + N} \big)^2 \right\}$ 是 Gaussian 分布的形式，所以后验分布的解析式可以直接写得：
> $$
> p(\mu, \tau | \mathbf X) = \mathcal N \bigg(\mu | \frac{\sum_{i=1}^N x_i + \lambda_0 \mu_0}{\lambda_0 + N}, \big[(\lambda_0 + N) \tau \big]^{-1} \bigg) \cdot \text{Ga} \bigg(\tau | a_0 + \frac{N}{2}, b_0 + \frac{1}{2} \sum_{i=1}^N x_i^2 + \frac{\lambda_0 \mu_0^2}{2} \bigg)
> $$
> 对应上述精确解。



那么假如不知道这个后验分布的解，比如一些非常复杂的后验分布无法求解，我们则可以通过 VI（基于平均场）来近似这一后验。我们设该近似分布为：
$$
q(\mu, \tau) = q_{\mu}(\mu) \, q_{\tau}(\tau)
$$
目的是让 $q(\mu, \tau) \rightarrow p(\mu, \tau | \mathcal D)$。于是我们得到最优解 $q_{\mu}^{*}(\mu)$ 满足：
$$
\begin{aligned}
    \log q_{\mu}^{*}(\mu) &= \mathbb E_{q_{\tau}(\tau)} [\log p(\mu, \tau, \mathbf X)] \\
    &= \mathbb E_{q_{\tau}(\tau)} [\log p(\mathbf X | \mu, \tau) + \log p(\mu | \tau)] + \text{const} \\
    &= \int_{\tau} q_{\tau}(\tau) \left[\frac{N}{2} \log (\tau) - \frac{\tau}{2} \sum_{i=1}^N (x_i - \mu)^2 - \frac{\lambda_0 \tau}{2} (\mu - \mu_0)^2 \right] + \text{const} \quad (将与 \mu 无关的项用 \text{const} 表示)\\
    &= -\frac{\mathbb E_{q_{\tau}(\tau)} [\tau]}{2} \left[\sum_{i=1}^N (x_i - \mu)^2 + \lambda_0 (\mu - \mu_0)^2 \right] + \text{const} \\
    &= - \frac{\mathbb E_{q_{\tau}(\tau)} [\tau] (N + \lambda_0)}{2} \left(\mu - \frac{N \bar{x} + \lambda_0 \mu_0}{N + \lambda_0} \right)^2 + \text{const}
\end{aligned}
$$
所以，$q_{\mu}^*(\mu) = \mathcal N \big(\frac{N\bar{x} + \lambda_0 \mu_0}{N + \lambda_0}, \mathbb E_{q_{\tau}} [\tau] (N + \lambda_0) \big)$。
同理，
$$
\log q_{\tau}^*(\tau) = \big(\underbrace{\frac{N}{2} + a_0}_{a_N} - 1 \big) \log(\tau) - \tau \big(\underbrace{b_0 + \frac{1}{2} \mathbb E_{q_{\mu}} [\sum_{i=1}^N (x_i - \mu)^2 + \lambda_0 (\mu - \mu_0)^2]}_{b_N} \big) + \text{const}
$$
所以，$q_{\tau}^*(\tau) = \mathrm{Ga} (a_N, b_N)$ 。



**可以得出结论**：

(1) 无须指定 $q_{\mu}(\mu)$ 和 $q_{\tau}(\tau)$ 的函数形式，因为它们可以从似然函数和共轭先验自动推导出来；

(2) 虽然我们假设了 $q_{\mu}(\mu)$ 和 $q_{\tau}(\tau)$ 相互独立，但求解结果表明它们是相互耦合的，即 $q_{\mu}(\mu)$ 依赖于 $q_{\tau}(\tau)$，而反过来 $q_{\tau}(\tau)$ 依赖于 $q_{\mu}(\mu)$。

(3) $\mu_N$ 和 $a_N$ 是固定常数，只有 $\lambda_N$ 和 $b_N$ 需要迭代更新。





#### Iterative optimization and Computing the expectation

根据上面的推断结果，进行一定顺序下的迭代优化求解：
$$
\mathbb E [\tau] \longrightarrow q_{\mu}(\mu): \mathbb E[\mu], \\
\mathbb E[\mu^2] \longrightarrow q_{\tau}(\tau): a_N, \\
b_N \longrightarrow \mathbb E[\tau] \longrightarrow \dots
$$
所以接下来的问题就是如何设置初始值 $\mathbb E[\tau]$，由于两个分布相互耦合，那么初始值一定会满足某些约束（为了简化计算，我们不妨令参数 $a_0 = b_0 = \mu_0 = \tau_0 = 0$ （即无信息先验）），为了实现更新，我们必须指定如何计算各种期望，接下来推导一下。
由于 $q_{\mu}(\mu) = \mathcal N(\mu | \mu_N, \lambda_N^{-1})$​，我们得到：
$$
\begin{aligned}
    \mathbb E_{q(\mu)}[\mu] &= \mu_N = \bar{x} \\
    \mathbb E_{q(\mu)}[\mu^2] &= \frac{1}{\lambda_N} + \mu_N^2 = \frac{1}{N \mathbb E_{q(\tau)}[\tau]} + \bar{x}^2 \quad (\frac{a_N}{b_N} = \mathbb E_{q(\tau)}[\tau])
\end{aligned}
$$
由于 $q(\tau) = \text{Ga}(\tau | a_N, b_N)$​，我们得到：
$$
\begin{aligned}
    \mathbb E_{q(\tau)}[\tau] &= \frac{a_N}{b_N} \\
    \frac{1}{\mathbb E_{q(\tau)}[\tau]} &= \frac{b_N}{a_N} = \frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2
\end{aligned}
$$
所以：$\mathbb E_{q(\tau)}[\tau] = \frac{N}{\sum_{i=1}^N (x_i - \bar{x})^2}$。我们由此确定了 $\mathbb E_{q(\tau)}[\tau]$​ 的初值。



现在我们可以给出**更新方程的显式形式**。

对于 $q(\mu)$​ 我们有：
$$
\begin{aligned}
    \mu_N &= \frac{\lambda_0 \mu_0 + N \bar{x}}{\lambda_0 + N} \\
    \lambda_N &= (\lambda_0 + N) \frac{a_N}{b_N}
\end{aligned}
$$
对于 $q(\tau)$​ 我们有：
$$
\begin{aligned}
    a_N &= a_0 + \frac{N + 1}{2} \\
    b_N &= b_0 + \frac{1}{2} \lambda_0 \big(\mathbb E [\mu^2] + \mu_0^2 - 2 \mathbb E [\mu] \mu_0 \big) + \frac{1}{2} \sum_{i=1}^N \big(x_i^2 + \mathbb E [\mu^2] - 2 \mathbb E [\mu] x_i \big)
\end{aligned}
$$
接下来就可以进行迭代优化。可以看到 $\mu_N$ 和 $a_N$ 实际上是固定常数，只有 $\lambda_N$ 和 $b_N$ 需要迭代更新。事实上可以通过解析方式求解 $\lambda_N$ 和 $b_N$ 的不动点，但为了说明迭代更新方案，在这里不这样做。

