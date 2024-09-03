# GPs with non-Gaussian likelihoods

2024-04-04

------

使用高斯似然进行回归的GP，后验也会是一个GP，并且所有计算都可以**解析**地进行。然而，如果似然是非高斯的，就无法再精确地计算后验概率。

[TOC]

---

## GP Classification

### Binary classification

对于二元分类，GP预测背后的思想非常简单，将GP至于潜在函数 $f(x)$ 上，然后通过逻辑函数挤压得到 $\boldsymbol y$ 的先验 $\pi(\boldsymbol{x}) \triangleq p(y = +1 | \boldsymbol{x}) = \sigma(f(\boldsymbol{x}))$。$\pi$ 是 $f(\boldsymbol{x})$ 的确定性函数，$f(\boldsymbol{x})$ 是随机变量，所以 $\pi$ 也是随机变量。

潜在函数 $\boldsymbol f$ 扮演了干扰函数的角色：我们不观察干扰函数 $\boldsymbol f$ 本身的值 ，只观察输入 $\mathbf X$ 和类别标签 $\boldsymbol y$ 。我们对 $\boldsymbol f$ 的值不感兴趣，而是对 $\pi$ 感兴趣，特别是对于测试样本 $\pi(\boldsymbol x_*)$ 。$\boldsymbol f$ 的目的仅仅是方便表述模型，在后面的部分中计算的目标是将 $\boldsymbol f$ 积分掉。

**推断分为两个步骤：**

1. 首先计算对应于测试的潜在变量 $\boldsymbol f_*$ 的分布：
    $$
    p(\boldsymbol f_* | \boldsymbol y, \mathbf X, \boldsymbol x_*) = \int p(\boldsymbol f_* | \boldsymbol f_X, \mathbf X, \boldsymbol x_*)\, \underbrace{p(\boldsymbol f_X | \boldsymbol y, \mathbf X)}_{\color{royalblue}\text{non-Gaussain posterior}}\, \mathrm{d}\boldsymbol f_X
    $$
    其中，$p(\boldsymbol f_X| \boldsymbol y, \mathbf X) = \underbrace{p(\boldsymbol y | \boldsymbol f_X)}_{\color{royalblue}\text{non-Gaussian likelihood}}\, \underbrace{p(\boldsymbol f_X | \mathbf X)}_\text{prior}\,  / \, \underbrace{p(\boldsymbol y | \mathbf X)}_\text{marginal log likelihood}$ 是潜在变量 $\boldsymbol f_X$​ 的后验。

2. 然后使用这个分布对潜在变量 $\boldsymbol f_*$ 进行概率预测：
    $$
    \begin{aligned}
    	\pi_* &\triangleq p(y_* = +1 | \boldsymbol x_*, \mathbf X, \boldsymbol y) \\
    	&= \int \underbrace{\sigma(f_*)}_{\color{blue}\text{prior of $y_*$}}\, \underbrace{p(f_* | \boldsymbol y, \mathbf X, \boldsymbol x_*)}_{\color{blue}\text{posterior of $f_*$}}\, \mathrm{d}f_*
    \end{aligned}
    $$

在GPR（高斯似然）下，预测的计算是直接的，相关积分是高斯的，可以解析计算。但上式中的 **非高斯似然和后验** 使得积分难以计算。



近似推断的最简单的方法是使用**拉普拉斯近似（Laplace Approximation）**。对于对数联合分布的梯度和Hessian矩阵如下：
$$
\begin{aligned}
	\nabla \mathcal L &= \nabla \log p(\boldsymbol y | \boldsymbol f_X) - \mathbf K_{XX}^{-1} \boldsymbol f_X \\
	\nabla^2 \mathcal L &= \underbrace{\nabla^2 \log p(\boldsymbol y | \boldsymbol f_X)}_{-\mathbf \Lambda \text{(diagonal matrix)}} - \mathbf K_{XX}^{-1} = - \mathbf \Lambda - \mathbf K_{XX}^{-1}
\end{aligned}
$$
收敛后，后验的拉普拉斯近似采用以下形式：
$$
p(\boldsymbol f_X | \mathcal D) \approx q(\boldsymbol f_X) = \mathcal N(\underbrace{\hat{\boldsymbol f}}_{\text{MAP estimate}}, (\mathbf K_{XX}^{-1} + \mathbf \Lambda)^{-1})
$$
关于这里拉普拉斯近似方法的详细推导，参考[Appendix Ⅰ](# Appendix Ⅰ: Laplace Approximation for the Binary GP Classifier)。



为了提高准确性，我们可以使用变分推断，其中我们假设 $q(\boldsymbol f_X) = \mathcal N(\boldsymbol f_X | \boldsymbol m, \mathbf S)$ 。然后我们使用（随机）梯度下降来优化 $\boldsymbol m, \mathbf S$ ，而不是假设 $\mathbf S$ 是该模式下的Hessian矩阵。

一旦有了高斯后验 $q(\boldsymbol f_X | \mathcal D)$ ，我们就可以使用标准GP预测来计算 $q(f_* | \boldsymbol x_*, \mathcal D)$ 。最后使用以来方法来近似二元标签的后验预测分布：
$$
\pi_* = p(y_* = +1 | \boldsymbol x_*, \mathcal D) = \int \underbrace{p(y_*=+1 | f_*)}_{\sigma(f_*)}\, q(f_* | \boldsymbol x_*, \mathcal D)\, \mathrm{d}f_*
$$
这个一维积分可以使用 ==**概率近似（probit approximation）**== 来计算。可以得到 $\pi_* \approx \sigma(\kappa(v)\, \mathbb E[f_*])$ ，其中 $v = \mathbb V[f_*], \kappa^{2}(v) = (1+\pi v / 8)^{-1}$ 。





### Multiclass classification

==TODO==





---

## Variational Gaussian Process

> Version 1
>
> 2024-04-04

VGP可以针对非高斯似然的情况，利用VI来近似非高斯后验。

### Likelihood

随机变量 $y(x) \rightarrow y$ ，把观测值 $y$ 建模为 $Y$ 中随机变量的样本，每个 $y_i$ 对应随机变量 $Y_i$ 的一个样本， $y$ 的长度为 $N$， $N$ 为训练点的个数。
$$
\underbrace{y \longrightarrow 观测随机变量}_{\color{blue} \text{observational random variable}} \quad\quad \underbrace{f, f_* \longrightarrow 潜在随机变量}_{\color{blue} \text{latent random variable}}
$$
似然（likelihood） $p(y | f, f_*)$ 描述了在给定GP先验的潜在变量 $f, f_*$ 下，观测到 $y$ 的概率。
$$
p(y | f, f_*) = p(y | f)
$$
因为 $y_i$ 是一个二值变量，所以建模为一个Bernoulli随机变量的一个样本：
$$
p(y_i | f_i) = g_i^{y_i} (1 - g_i)^{1-y_i}
$$
$g$ 是伯努利分布的 **唯一参数** ，$g_i = g(f_i)$， $y$ 是一个伯努利随机变量的向量。
$$
g_i = g(f_i) = \frac{e^{f_i}}{1 + e^{f_i}}
$$
所以：
$$
\begin{align}
p(y_i | f_i) = \left(\frac{e^{f_i}}{1 + e^{f_i}}\right)^{y_i} \left(1 - \frac{e^{f_i}}{1 + e^{f_i}}\right)^{1-y_i} \tag{\color{red}likelihood} 

\end{align}
$$




### Posterior

$$
\begin{aligned}
	p(f_*, f | y) &= p(f_* | f, y)\, p(f | y) \\
	&= p(f_* | f)\, \color{red}{p(f | y)}
\end{aligned}
$$

我们需要近似这个后验分布 $p(f | y)$，我们想要变分分布 $q(f; \mu, \Sigma)$ 来接近后验 $p(f |y; \ell , \sigma^2)$。
$$
\begin{aligned}
	\mathrm{KL}\big[q(f; \mu, \Sigma) \parallel p(f | y; \ell, \sigma^2)\big] &= \mathbb E_{q(f)} \left[log \frac{q(f; \mu, \Sigma)}{p(f | y; \ell, \sigma^2)}\right]\ \\
	&= \int \log \frac{q(f; \mu, \Sigma)}{p(f | y; \ell, \sigma^2)}\, q(f; \mu, \Sigma)\, \mathrm{d} f
\end{aligned}
$$
我们的目标就是找到KL散度的最小值：
$$
\arg \min_{\color{red}{\mu, \Sigma, \ell, \sigma^2}} \mathrm{KL} \big[q(f; {\color{red}\mu, \Sigma}) \parallel p(f | y; {\color{red}\ell, \sigma^2})\big]
$$
可与将这个优化问题联系到ELBO，下面给出ELBO的导出：
$$
\begin{aligned}
	\mathrm{KL} \big[q(f; \mu, \Sigma) \parallel p(f | y; \ell, \sigma^2)\big] &= \mathbb E_{q(f)} \left[\log \frac{q(f)}{p(f | y)}\right] \\
	&= \mathbb E
\end{aligned}
$$

























## Appendix Ⅰ: Laplace Approximation for the Binary GP Classifier
**References:** GP for ML 3.4

拉普拉斯近似利用 **高斯近似分布** $q(\boldsymbol f_X | \boldsymbol y, \mathbf X)$ 来近似后验 $p(\boldsymbol f_X | \boldsymbol y, \mathbf X)$ 。对 $\log p(\boldsymbol f | \boldsymbol y, \mathbf X)$ 围绕后验最大值进行二阶泰勒展开，得到高斯近似：
$$
q(\boldsymbol f_X | \boldsymbol y, \mathbf X) = \mathcal N(\boldsymbol f_X | \hat{\boldsymbol f}_X, \mathbf A^{-1}) \propto \exp\bigg\{-\frac{1}{2}(\boldsymbol f_X - \hat{\boldsymbol f}_X)^{\top} \mathbf A (\boldsymbol f_X - \hat{\boldsymbol f}_X) \bigg\}
$$
其中 $\hat{\boldsymbol f}_X = \arg \max_{\boldsymbol f} p(\boldsymbol f_X | \boldsymbol y , \mathbf X)$ ，$\mathbf A = -\nabla \nabla \log p(\boldsymbol f_X | \boldsymbol y, \mathbf X) |_{\boldsymbol f_X = \hat{\boldsymbol f}_X}$ 是该点的负对数后验的Hessian矩阵。$\hat{\boldsymbol f}_X$ 指的就是 $p(\boldsymbol f_X | \boldsymbol y, \mathbf X)$ 最大时所对应的 $\boldsymbol f_X$ ，即MAP估计 。

### Posterior

这一节描述如何找到 $\hat{\boldsymbol f}_X$ 和 $\mathbf A$ 。

已知贝叶斯后验 $p(\boldsymbol f_X | \boldsymbol y, \mathbf X) = p(\boldsymbol y | \boldsymbol f_X)\, p(\boldsymbol f_X | \mathbf X) / p(\boldsymbol y | \mathbf X)$ ，因为 $p(\boldsymbol y | \mathbf X)$ 与 $\boldsymbol f_X$ 无关，因此我们在最大化 w.r.t. $\boldsymbol f_X$ 时只需要考虑非标准化的后验。于是对数非归一化后验 $\Psi(\boldsymbol f)$ 的表达式为：<font color='red'>（简单化，这里用 $\boldsymbol f$ 来表示 $\boldsymbol f_X$， 后续有时间再统一修改）</font>
$$
\begin{aligned}
	\Psi(\boldsymbol f) &\triangleq \log p(\boldsymbol y | \boldsymbol f) + \log p(\boldsymbol f | \mathbf X) \\
	&= \log p(\boldsymbol y | \boldsymbol f) -\frac{1}{2} \boldsymbol f^{\top} \mathbf K^{-1} \boldsymbol f -\frac{1}{2} \log |\mathbf K| - \frac{N}{2} \log 2\pi
\end{aligned}
$$
上述微分方程 w.r.t. $\boldsymbol f$ ，于是得到：
$$
\begin{aligned}
	\nabla\Psi(\boldsymbol f) &= \nabla \log p(\boldsymbol y | \boldsymbol f) - \mathbf K^{-1} \boldsymbol f \\
	\nabla^2 \Psi(\boldsymbol f) &= \underbrace{\nabla^2 \log p(\boldsymbol y | \boldsymbol f)}_{- \mathbf W (\text{diagonal matrix})} - \mathbf K^{-1} = - \mathbf W - \mathbf K^{-1}
\end{aligned}
$$
其中 $\mathbf W \triangleq - \nabla^2 \log p(\boldsymbol y | \boldsymbol f)$ 是**对角矩阵**，源于似然的分解，$y_i$ 只依赖于 $f_i$ 而不是 $f_{j \neq i}$​ 。

> **NOTE**
>
> 如果似然 $p(\boldsymbol y | \boldsymbol f)$ 是对数凹的，则 $\mathbf W$ 的对角线元素是非负的，并且等式中的Hessian矩阵是负定的，因此 $\Psi(\boldsymbol f)$ 是凹的并且拥有唯一的最大值。

似然度有许多可能的函数形式，它给出了作为潜在变量 $\boldsymbol f$ 的函数的目标类别概率。两种常用的似然函数是逻辑斯谛函数（logistic）和累积高斯函数（cumulative Gaussian）。

<img src="C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240421125936477.png" alt="image-20240421125936477" style="zoom: 67%;" />

这些似然函数及其一阶和二阶导数的对数似然表达式潜在变量在下表中给出：

<img src="C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240421130210572.png" alt="image-20240421130210572" style="zoom:67%;" />

其中我们定义了 $\pi_i = p(y_i = 1 | f_i)$ 和 $\boldsymbol t = (\boldsymbol y + \boldsymbol 1) / 2$ 。在 $\Psi(\boldsymbol f)$ 的最大处，我们有：
$$
\nabla \Psi = \boldsymbol 0 \Longrightarrow \hat{\boldsymbol f} = \mathbf K \nabla \log p(\boldsymbol y | \hat{\boldsymbol f})
$$
是一个 $\hat{\boldsymbol f}$ 的自洽方程，但是由于 $\nabla \log p(\boldsymbol y | \hat{\boldsymbol f})$ 是 $\hat{\boldsymbol f}$ 的非线性函数，因此无法直接求解上述方程。为了找到 $\Psi$ 的最大值，可以使用牛顿法迭代：
$$
\begin{aligned}
	\boldsymbol f^{\mathrm{new}} = \boldsymbol f - (\nabla^2\Psi)^{-1} \nabla \Psi &= \boldsymbol f + (\mathbf K^{-1} + \mathbf W)^{-1} (\nabla \log p(\boldsymbol y | \boldsymbol f) - \mathbf K^{-1} \boldsymbol f) \\
	&= (\mathbf K^{-1} + \mathbf W)^{-1} (\mathbf W \boldsymbol f + \nabla \log p(\boldsymbol y | \boldsymbol f))
\end{aligned}
$$
==直观理解 TODO==



找到了最大后验 $\hat{\boldsymbol f}$ 后，我们现在可以将后验的拉普拉斯近似指定为高斯，其均值与协方差矩阵由 $\Psi$ 的负逆Hessian矩阵给出：
$$
q(\boldsymbol f | \mathbf X, \boldsymbol y) = \mathcal N(\hat{\boldsymbol f}, (\mathbf K^{-1} + \mathbf W)^{-1})
$$
拉普拉斯近似的一个问题是它本质上是不受控制的，因为 Hessian 矩阵（在 $\hat{\boldsymbol f}$ 处估计）可能对后验的真实形状给出较差的近似。该峰可能比 Hessian 矩阵指示的更宽或更窄，或者它可能是斜峰，而拉普拉斯近似假设它具有椭圆形轮廓。













### Implementation

避免数值不稳定的计算，同时最大限度地减少计算量；两者可以同时实现。事实证明，几个所需的项可以用对称正定矩阵来表示：
$$
\mathbf B = \mathbf I + \mathbf W^{\frac{1}{2}} \mathbf K \mathbf W^{\frac{1}{2}}
$$
由于 $\mathbf W$ 是对角线，因此其计算成本仅为 $O(N^2)$ 。 $\mathbf B$ 矩阵的特征值下界为 1，上界为 $1 + N \max_{ij}(\mathbf K_{ij}) / 4$ ，因此对于许多协方差函数 $\mathbf B$ 保证是良条件的，并且它是可以在数值上安全地计算其 Cholesky 分解 $\mathbf L \mathbf L^{\top} = \mathbf B$ ，这在计算涉及 $\mathbf B^{-1}$ 和 $|\mathbf B|$ 的项时很有用。
$$
(\mathbf K^{-1} + \mathbf W)^{-1} = \mathbf K - \mathbf K \mathbf W^{\frac{1}{2}} \mathbf B^{-1} \mathbf W^{\frac{1}{2}} \mathbf K
$$

> **NOTE**  矩阵求逆
> $$
> (\mathbf Z + \mathbf U \mathbf W \mathbf V^{\top})^{-1} = \mathbf Z^{-1} - \mathbf Z^{-1}\mathbf U(\mathbf W^{-1} + \mathbf V^{\top} \mathbf Z^{-1} \mathbf U)^{-1} \mathbf V^{\top} \mathbf Z^{-1}
> $$







### Marginal likelihood 

计算边际似然 $p(\boldsymbol y | \mathbf X)$ 的拉普拉斯近似也很有用。（对于具有高斯噪声的回归情况，可以再次通过分析计算边际似然）
$$
p(\boldsymbol y | \mathbf X) = \int p(\boldsymbol y | \boldsymbol f)\, p(\boldsymbol f | \mathbf X)\, \mathrm{d} \boldsymbol f = \int \exp\{\Psi(\boldsymbol f)\}\, \mathrm{d}\boldsymbol f
$$
使用 $\Psi(\boldsymbol f)$ 在 $\hat{\boldsymbol f}$ 的局部泰勒展开，我们的得到 $\Psi(\boldsymbol f) \approx \Psi(\hat{\boldsymbol f}) - \frac{1}{2} (\boldsymbol f - \hat{\boldsymbol f})^{\top} \mathbf A (\boldsymbol f - \hat{\boldsymbol f})$ ，从而得到边际似然的近似值 $q(\boldsymbol y | \mathbf X)$ ：
$$
p(\boldsymbol y | \mathbf X) \approx q(\boldsymbol y | \mathbf X) = \exp\{\Psi(\hat{\boldsymbol f})\} \int \exp\{-\frac{1}{2}(\boldsymbol f - \hat{\boldsymbol f})^{\top} \mathbf A (\boldsymbol f - \hat{\boldsymbol f})\}\, \mathrm{d}\boldsymbol f
$$
这个高斯积分可以通过解析计算得到对数边际似然的近似值:
$$
\log q(\boldsymbol y | \mathbf X, \boldsymbol \theta) = -\frac{1}{2}\hat{\boldsymbol f}^{\top} \mathbf K^{-1} \hat{\boldsymbol f} + \log p(\boldsymbol y | \hat{\boldsymbol f}) - \frac{1}{2} \log |\mathbf B|
$$
其中， $\mathbf B = \mathbf I + \mathbf W^{\frac{1}{2}} \mathbf K \mathbf W^{\frac{1}{2}}$ ，$\boldsymbol \theta$ 是协方差函数的超参数向量。
