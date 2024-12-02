# Sparse GPs

Begin: 2024-04-04	End: 2024-

GP推断，预测和参数学习中都需要矩阵求逆，即 $\mathbf K^{-1}$，最佳方法是计算 $N \times N$ Gram矩阵的Cholesky分解。不幸的是，这需要 $O(N^3)$ 的时间。因此有必要针对这个问题提出改进。

> **为什么会出现 $\mathbf{K}^{-1}$​ ?**
>
> $\mathbf{K}^{-1}$ 出现在预测分布 $p(f_* | f)$ 和参数学习目标 $\log p(y)$ 中，因为我们使用了以下的GP先验分解：
> $$
> p(f_*, f) = p(f_* | f)\, p(f)
> $$
>
> $$
> p(f) = \frac{1}{(2\pi)^{N/2} |\mathbf{K}|^{1/2}}\, \exp\{-\frac{1}{2} y^\top \textcolor{red}{\mathbf{K}^{-1}} y \}
> $$
>
> 









[TOC]





## Nyström Approximation

==TODO==





## Inducing Point Methods

> 2024-04-04
>
> Refer: Book 2 18.5.3

基于归纳点（**inducing points**）的近似方法，也称为伪输入（**pseudo inputs**），就像我们可以条件化的训练数据的总结，而不是条件化所有数据。

已知 $\mathbf X$ 是我们观测到的输入数据，







------

## Sparse Variational GP (SVGP)

GP 推理的变分方法，称为 **稀疏变分 GP** 或  **SVGP 近似** ，也称为 **变分自由能 **或 **VFE 方法** 。

首先将训练数据 $\mathcal X$ 划分成三个子集：训练集 $\mathbf X$ ，诱导点 $\mathbf Z$ ，和其他（可以看成是测试集） $\mathbf X_*$ ，假设这些集合是不相交的。令 $\boldsymbol f_X, \boldsymbol f_Z, \boldsymbol f_*$ 表示这些点上相应的未知函数值，令 $\boldsymbol f = [\boldsymbol f_X, \boldsymbol f_Z, \boldsymbol f_*]$ 为所有未知数（这里我们使用固定长度的向量 $\boldsymbol f$，但结果可推广到高斯过程）。假设函数是从GP中采样的，$p(\boldsymbol f) = \mathcal N (m(\mathcal X), \mathcal K(\mathcal X, \mathcal X))$​​ 。



------



### Prior

用多元高斯分布建模 $\boldsymbol f_X$ 和 $\boldsymbol f_Z$ 的关系，称之为稀疏 GP 先验，他们的联合分布为：
$$
\begin{aligned}
	p(\boldsymbol f_X, \boldsymbol f_Z) = \mathcal N \left(
	\left.\begin{bmatrix}
        \boldsymbol f_X \\
        \boldsymbol f_Z
	\end{bmatrix} \right |
	\begin{bmatrix}
		\boldsymbol 0 \\
		\boldsymbol 0
	\end{bmatrix}, 
	\begin{bmatrix}
		\mathbf K_{XX} & \mathbf K_{XZ} \\
		\mathbf K_{ZX} & \mathbf K_{ZZ}
	\end{bmatrix}
	\right).
\end{aligned}
$$
如果对联合分布进行因式分解：
$$
\begin{aligned}
	p(\boldsymbol f_X, \boldsymbol f_Z) = p(\boldsymbol f_X | \boldsymbol f_Z)\, p(\boldsymbol f_Z) ,
\end{aligned}
$$
我们可以应用高斯条件规则来推导<font color=red>边际先验</font> $p(\boldsymbol f_Z)$ 和<font color=red>条件先验</font> $p(\boldsymbol f_X | \boldsymbol f_Z)$ 。

> **TIP**
>
> 对比GPR的先验，两者具有相同的结构：
> $$
> \begin{aligned}
> 	p(\boldsymbol f_*, \boldsymbol f_X) = \mathcal N \left(
> 	\left.\begin{bmatrix}
>         \boldsymbol f_* \\
>         \boldsymbol f_X
> 	\end{bmatrix} \right |
> 	\begin{bmatrix}
> 		\boldsymbol 0 \\
> 		\boldsymbol 0
> 	\end{bmatrix}, 
> 	\begin{bmatrix}
> 		\mathbf K_{**} & \mathbf K_{*X} \\
> 		\mathbf K_{*X}^{\top} & \mathbf K_{XX}
> 	\end{bmatrix}
> 	\right)
> \end{aligned}
> $$
> SVGP使用 $p(\boldsymbol f_X | \boldsymbol f_Z)$ 从 $\boldsymbol f_Z$ 的信息中解释 $\boldsymbol f_X$ ，协方差矩阵 $\mathbf K_{XZ}$ 定义了 $\boldsymbol f_X$ 和 $\boldsymbol f_Z$ 的相关性；GPR使用 $p(\boldsymbol f_* | \boldsymbol f_X)$ 从 $\boldsymbol f_X$ 的信息中解释 $\boldsymbol f_*$ ，协方差矩阵 $\mathbf K_{X*}$ 定义了 $\boldsymbol f_X$ 和 $\boldsymbol f_*$​ 的相关性。
>
> SVGP先验和GPR先验都使用**多元高斯条件规则**，解释一个随机变量向量从另一个随机变量向量的机制。



---



### 诱导变量 $\boldsymbol f_Z$ 背后的直觉

使用诱导变量 $\mathbf Z$ 来总结训练数据 $\mathbf X$ 的含义就是 **用简短的形式表达关于某事的最重要的事实** 。

- 通过条件概率密度函数 $p(\boldsymbol f_X | \boldsymbol f_Z)$ ，用 $\boldsymbol f_Z$ 来表示 $\boldsymbol f_X$ 。

- 要求诱导变量的数量 $N_Z$ 小于(通常要小得多)训练数据点的数量 $N$​ 。这就是诱导变量总结训练数据的原因。这也是为什么我们称SVGP模型为稀疏的原因——我们想用关键诱导位置的少量诱导变量来解释训练位置的大量随机变量。
- 诱导变量或位置的数量 $N_Z$ 不是模型参数。在确定了$N_Z$ 的值之后，我们将有一个长度为 $N_Z$ 的向量 $\mathbf Z$​ ，表示这些诱导变量的位置。我们不知道这些位置在哪里，它们是模型参数，我们将使用 **参数学习** 来找到这些诱导位置的具体值，以及其他模型参数。



---



### Marginal prior over inducing variables

诱导变量的边际先验由下式给出
$$
\begin{aligned}
	p(\boldsymbol f_Z) = \mathcal N(\boldsymbol f_Z | \boldsymbol 0, \mathbf K_{ZZ}).
\end{aligned}
$$

> **NOTE**
>
> **高斯过程符号**
>
> 可以在输入为诱导点 $\boldsymbol z$ 时表示诱导变量 $\boldsymbol f_Z(\boldsymbol z)$ 或 $\boldsymbol u(\boldsymbol z)$ 
> $$
> \begin{aligned}
> 	p(\boldsymbol u(\boldsymbol z)) = \mathcal{GP} (\boldsymbol 0, \mathcal K_{\boldsymbol \theta}(\boldsymbol z, \boldsymbol z')).
> \end{aligned}
> $$



### Conditional prior

对诱导变量的联合先验分布进行条件化，利用多元高斯条件规则
$$
\begin{aligned}
	p(\boldsymbol f_X | \boldsymbol f_Z) = \mathcal N \big(\boldsymbol f_X | \mathbf K_{XZ} \mathbf K_{ZZ}^{-1} (\boldsymbol f_Z - \boldsymbol 0), \mathbf K_{XX} - \mathbf K_{XZ} \mathbf K_{ZZ}^{-1} \mathbf K_{XZ}^{\top}\big)
\end{aligned}
$$

> **NOTE**
>
> **高斯过程符号**
>
> 在输入为 $\mathbf X$ ，给定 $\boldsymbol f_Z$ 时，用条件概率 $p(\boldsymbol f(\mathbf X) | \boldsymbol f_Z)$ 表示函数值 $\boldsymbol f(\mathbf X)$ 的分布：
> $$
> 
> $$
> 



----



### Variational distribution


关于 **诱导点的假设**：$p(\boldsymbol f_*, \boldsymbol f_X, \boldsymbol f_Z) \approx p(\boldsymbol f_* | \boldsymbol f_Z)\, p(\boldsymbol f_X | \boldsymbol f_Z)\, p(\boldsymbol f_Z)$ 来近似GP先验。选择诱导点 $\boldsymbol f_Z$ 使观测数据的似然最大化，然后在这个近似的模型中执行精确推理。相比之下，在SVGP中，将保持模型不变，但**使用变分推断来近似后验** $p(\boldsymbol f | \boldsymbol y)$ 。$\color{red}\boldsymbol f = [\boldsymbol f_X, \boldsymbol f_Z, \boldsymbol f_*]$ 。

> **NOTE**
>
> 在VFE的视角中，诱导点 $\mathbf Z$ 和诱导变量 $\boldsymbol f_Z$ （通常用 $\boldsymbol u$ 表示）是变分参数，而不是模型参数，这避免了 **过拟合的风险**。可以证明，随着诱导点数量的增加，后验的质量不断提高，最终恢复精确的推理。相比之下，在经典的诱导点方法中，增加诱导点数量并不总是会带来更好的性能。



VFE方法尝试找到近似后验 $q(\boldsymbol f)$ 来最小化 $D_{\mathbb {KL}} (q(\boldsymbol f) \parallel p(\boldsymbol f | \boldsymbol y))$ ，==关键假设是 $q(\boldsymbol f) = p(\boldsymbol f_*, \boldsymbol f_X | \boldsymbol f_Z)\, q(\boldsymbol f_Z)$ （${\color{RoyalBlue}q(\boldsymbol f_X, \boldsymbol f_Z) = p(\boldsymbol f_X | \boldsymbol f_Z)\, q(\boldsymbol f_Z)}$）==。其中 $p(\boldsymbol f_*, \boldsymbol f_X | \boldsymbol f_Z)$ 是用GP先验精确计算的，并且 $q(\boldsymbol f_Z)$ 通过最小化 $\mathcal K(q) = D_{\mathbb{KL}} (q(\boldsymbol f) \parallel p(\boldsymbol f | \boldsymbol y))$ 来学习。可以证明 $\color{orangered}D_{\mathbb{KL}} (q(\boldsymbol f) \parallel p(\boldsymbol f | y)) = D_{\mathbb{KL}} (q(\boldsymbol f_X, \boldsymbol f_Z) \parallel p(\boldsymbol f_X, \boldsymbol f_Z | \boldsymbol y))$​ 。

$q(\boldsymbol f) = p(\boldsymbol f_*, \boldsymbol f_X | \boldsymbol f_Z)\, q(\boldsymbol f_Z)$ ：直观上 $q(\boldsymbol f_Z)$ 充当瓶颈，吸收 $\boldsymbol y$ 的所有观测值的信息，然后通过 $\boldsymbol f_X$ 或 $\boldsymbol f_*$ 对 $\boldsymbol f_Z$​ 的依赖，而不是他们彼此之间的依赖来进行后验预测。



我们指定一个联合变分分布 $q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z)$ 的因子分解为：
$$
\begin{aligned}
	q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z) \triangleq p(\boldsymbol f_X | \boldsymbol f_Z)\, q_{\boldsymbol \psi}(\boldsymbol f_Z).
\end{aligned}
$$
其中 $\boldsymbol \psi$ 是变分参数。我们指定变分分布是一个多元高斯分布：
$$
\begin{aligned}
	q_{\boldsymbol \psi}(\boldsymbol f_Z) \triangleq \mathcal N(\boldsymbol f_Z | \boldsymbol m, \mathbf S),
\end{aligned}
$$
那么变分参数为 $\boldsymbol \psi = \{\boldsymbol m, \mathbf S\}$ 。$\boldsymbol m$ 是均值向量，长度为 $N_Z$ ，$\mathbf S$ 是协方差矩阵，大小为 $N_Z \times N_Z$ 。



---



### Varational Inference

我们通过变分分布 $q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z)$ 来近似精确后验 $p(\boldsymbol f_X, \boldsymbol f_Z | \boldsymbol y)$ ，为此，我们最小化 $q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z)$ 和 $p(\boldsymbol f_X, \boldsymbol f_Z | \boldsymbol y)$ 之间的KL散度。
$$
\begin{aligned}
	D_{\mathbb{KL}}(q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z) \parallel p(\boldsymbol f_X, \boldsymbol f_Z | \boldsymbol y)) &= \mathbb E_{q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z)}\left[\log \frac{q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z)}{p(\boldsymbol f_X, \boldsymbol f_Z | \boldsymbol y)} \right] \\
	&= \int q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z)\, \log \frac{q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z)}{p(\boldsymbol f_X, \boldsymbol f_Z | \boldsymbol y)}\, \mathrm{d}\boldsymbol f_X\, \mathrm{d}\boldsymbol f_Z \\
	&= \int q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z)\, \log \frac{q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z)\, p(\boldsymbol y)}{p(\boldsymbol f_X, \boldsymbol f_Z, \boldsymbol y)}\, \mathrm{d}\boldsymbol f_X\, \mathrm{d}\boldsymbol f_Z \\
	&= \log p(\boldsymbol y) + \mathbb E_{q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z)}\left[\log \frac{q_{\boldsymbol \psi}(\boldsymbol f_X, \boldsymbol f_Z)}{p(\boldsymbol f_X, \boldsymbol f_Z, \boldsymbol y)} \right] \\
	&= \log p(\boldsymbol y) - \mathrm{ELBO}(\boldsymbol \psi, \mathbf Z)
\end{aligned}
$$




推导出损失的形式用来计算后验 $q(\boldsymbol f_Z)$ ：
$$
\begin{aligned}
	\mathcal K(q) &= D_{\mathbb{KL}} (q(\boldsymbol f_*, \boldsymbol f_X, \boldsymbol f_Z) \parallel p(\boldsymbol f_*, \boldsymbol f_X, \boldsymbol f_Z | \boldsymbol y)) \\
	&= \int q(\boldsymbol f_*, \boldsymbol f_X, \boldsymbol f_Z)\, \log \frac{q(\boldsymbol f_*, \boldsymbol f_X, \boldsymbol f_Z)}{p(\boldsymbol f_*, \boldsymbol f_X, \boldsymbol f_Z)}\, \mathrm{d}\boldsymbol f_*\, \mathrm{d}\boldsymbol f_X\, \mathrm{d}\boldsymbol f_Z \\
	&= \int p(\boldsymbol f_*, \boldsymbol f_X | \boldsymbol f_Z)\, q(\boldsymbol f_Z)\, \log \frac{{\color{gray}p(\boldsymbol f_* | \boldsymbol f_X, \boldsymbol f_Z)\, p(\boldsymbol f_X | \boldsymbol f_Z)}\,  q(\boldsymbol f_Z)\, p(\boldsymbol y)}{{\color{gray}p(\boldsymbol f_* | \boldsymbol f_X, \boldsymbol f_Z)\, p(\boldsymbol f_X | \boldsymbol f_Z)}\, p(\boldsymbol f_Z)\, p(\boldsymbol y | \boldsymbol f_X)} \, \mathrm{d}\boldsymbol f_*\, \mathrm{d}\boldsymbol f_X\, \mathrm{d}\boldsymbol f_Z \\
	&= \int p(\boldsymbol f_*, \boldsymbol f_X | \boldsymbol f_Z)\, q(\boldsymbol f_Z)\, \log \frac{q(\boldsymbol f_Z)\, p(\boldsymbol y)}{p(\boldsymbol f_Z)\, p(\boldsymbol y | \boldsymbol f_X)} \, \mathrm{d}\boldsymbol f_* \mathrm{d}\boldsymbol f_X \mathrm{d}\boldsymbol f_Z \\
	&= \int q(\boldsymbol f_Z)\, \log \frac{q(\boldsymbol f_Z)}{p(\boldsymbol f_Z)}\, \mathrm{d} \boldsymbol f_Z - \int p(\boldsymbol f_X | f_Z) \, q(\boldsymbol f_Z)\, \log p(\boldsymbol y | \boldsymbol f_X)\, \mathrm{d} \boldsymbol f_X\, \mathrm{d} \boldsymbol f_Z + C \\
    &= \underbrace{\color{blue}D_{\mathbb{KL}} (q(\boldsymbol f_Z) \parallel p(\boldsymbol f_Z)) - \mathbb E_{q(\boldsymbol f_X)} \left[\log p(\boldsymbol y | \boldsymbol f_X)\right]}_{- \mathrm{ELBO}} + \underbrace{\color{red}C}_{C = \log p(\boldsymbol y)}
\end{aligned}
$$
也可以将目标转化为**最大化证据下界ELBO**：
$$
\begin{aligned}
	\log p(\boldsymbol y) &= \mathcal K(q) + \mathbb E_{q(\boldsymbol f_X)} [\log p(\boldsymbol y | \boldsymbol f_X)] - D_{\mathbb{KL}}(q(\boldsymbol f_Z) \parallel p(\boldsymbol f_Z)) \\
	&\geq \underbrace{\mathbb E_{q(\boldsymbol f_X)} [\log p(\boldsymbol y | \boldsymbol f_X)]}_{\color{royalblue}\text{likelihood term}} - \underbrace{D_{\mathbb{KL}}(q(\boldsymbol f_Z) \parallel p(\boldsymbol f_Z))}_{\color{royalblue}\text{KL term}} \\
	&\triangleq \mathcal L(q) \longrightarrow \color{blue}\mathrm{ELBO}
\end{aligned}
$$

所以写出ELBO的表达式：
$$
{\color{blue}\mathrm{ELBO}} = \mathcal L(q) = \underbrace{\mathbb E_{q(\boldsymbol f_X)} [\log p(\boldsymbol y | \boldsymbol f_X)]}_{\color{royalblue}\text{likelihood term}} - \underbrace{D_{\mathbb{KL}}(q(\boldsymbol f_Z) \parallel p(\boldsymbol f_Z))}_{\color{royalblue}\text{KL term}}
$$

> **Illustrate**
>
> 上面公式的详细推导：（参考其他）
> $$
> \begin{aligned}
> 	\log p(\boldsymbol y) &= \log \iint p(\boldsymbol y | \boldsymbol f_X, \boldsymbol f_Z)\, p(\boldsymbol f_X, \boldsymbol f_Z)\, \mathrm{d}\boldsymbol f_X\, \mathrm{d}\boldsymbol f_Z \\
> 	&= \log \iint p(\boldsymbol y | \boldsymbol f_X)\, p(\boldsymbol f_X, \boldsymbol f_Z)\, \mathrm{d}\boldsymbol f_X\, \mathrm{d}\boldsymbol f_Z \\
> 	&= \log \iint p(\boldsymbol y | \boldsymbol f_X)\, \frac{p(\boldsymbol f_X, \boldsymbol f_Z)}{\color{red}q(\boldsymbol f_X, \boldsymbol f_Z)}\, {\color{red}q(\boldsymbol f_X, \boldsymbol f_Z)}\, \mathrm{d}\boldsymbol f_X\, \mathrm{d}\boldsymbol f_Z \\
> 	&= \log \mathbb E_{q(\boldsymbol f_X, \boldsymbol f_Z)} \left[p(\boldsymbol y | \boldsymbol f_X)\, \frac{p(\boldsymbol f_X, \boldsymbol f_Z)}{q(\boldsymbol f_X, \boldsymbol f_Z)} \right] \\
> 	&\geq E_{q(\boldsymbol f_X, \boldsymbol f_Z)} \left[\log (p(\boldsymbol y | \boldsymbol f_X)\, \frac{p(\boldsymbol f_X, \boldsymbol f_Z)}{q(\boldsymbol f_X, \boldsymbol f_Z)}) \right] \\
> 	&= \iint q(\boldsymbol f_X, \boldsymbol f_Z)\, \log p(\boldsymbol y | \boldsymbol f_X)\, \mathrm{d}\boldsymbol f_X\, \mathrm{d}\boldsymbol f_Z - \iint q(\boldsymbol f_X, \boldsymbol f_Z)\, \log \frac{q(\boldsymbol f_X, \boldsymbol f_Z)}{p(\boldsymbol f_X, \boldsymbol f_Z)}\, \mathrm{d}\boldsymbol f_X\, \mathrm{d}\boldsymbol f_Z \\
> 	&= \int q(\boldsymbol f_X)\, \log p(\boldsymbol y | \boldsymbol f_X)\, \mathrm{d}\boldsymbol f_X - \int q(\boldsymbol f_Z)\, \log \frac{q(\boldsymbol f_Z)}{p(\boldsymbol f_Z)}\, \mathrm{d}\boldsymbol f_Z \\
> 	&= \underbrace{\mathbb E_{q(\boldsymbol f_X)} [\log p(\boldsymbol y | \boldsymbol f_X)]}_{\color{royalblue}\text{likelihood term}} - \underbrace{D_{\mathbb{KL}}(q(\boldsymbol f_Z) \parallel p(\boldsymbol f_Z))}_{\color{royalblue}\text{KL term}}
> \end{aligned}
> $$
> 



#### The KL term

${\color{blue}\textbf{KL term}}: D_{\mathbb{KL}} (q(\boldsymbol f_Z) \parallel p(\boldsymbol f_Z))$

现在假设选择 **高斯后验** 近似，$q(\boldsymbol f_Z) = \mathcal N(\boldsymbol f_Z | \boldsymbol m, \mathbf S)$ 。由于 $p(\boldsymbol f_Z) = \mathcal N (\boldsymbol f_Z | 0, \mathcal K(\mathbf Z, \mathbf Z))$ ，我们可以使用高斯之间的KL散度公式的封闭形式计算KL项。

> **KL divergence between two Gaussians**
> $$
> \begin{aligned}
> 	&D_{\mathbb{KL}} \big(\mathcal N(\boldsymbol x | \boldsymbol \mu_1, \boldsymbol \Sigma_1) \parallel \mathcal N(\boldsymbol x | \boldsymbol \mu_2, \boldsymbol \Sigma_2)\big) \\
> 	 &= \frac{1}{2}\left[\tr(\boldsymbol \Sigma_2^{-1} \boldsymbol \Sigma_1) + (\boldsymbol \mu_2 - \boldsymbol \mu_1)^\top \boldsymbol \Sigma_2^{-1} (\boldsymbol \mu_2 - \boldsymbol \mu_1) - D + \log \frac{\det(\boldsymbol \Sigma_2)}{\det(\boldsymbol \Sigma_1)} \right]
> \end{aligned}
> $$
> 在标量情况下，变为：
> $$
> D_{\mathbb{KL}} \big(\mathcal N(x | \mu_1, \sigma_1) \parallel \mathcal N(x | \mu_2, \sigma_2)\big) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2 \sigma_2^2} - \frac{1}{2}
> $$
> 

所以 **KL项的解析表达式** 为：
$$
{\color{blue}\textbf{KL term}}: D_{\mathbb{KL}}(q(\boldsymbol f_Z) \parallel p(\boldsymbol f_Z)) = \frac{1}{2}\left[\tr(\mathbf K_{ZZ}^{-1} \mathbf S) + (\boldsymbol 0 - \boldsymbol m)^\top \mathbf K_{ZZ}^{-1} (\boldsymbol 0 - \boldsymbol m) - N_Z + \log \frac{\det(\mathbf K_{ZZ})}{\det(\mathbf S)} \right]
$$

关于参数，$\boldsymbol m, \mathbf S$ 来自 $q(\boldsymbol f_Z)$ 。此外，可以看到这个KL项完全没有提到训练数据 $(\mathbf X, \boldsymbol y)$ 。





#### The likelihood term

${\color{blue}\textbf{likelihood term}}: \mathbb E_{q(\boldsymbol f_X)} [\log p(\boldsymbol y | \boldsymbol f_X)]$ ，这是 *expected log-likelihood (ELL)* 。

要计算对数似然的期望，首先要计算训练点处的潜在函数值 $\boldsymbol f_X$ 的后验：
$$
q(\boldsymbol f_X | \boldsymbol m, \mathbf S) = \int p(\boldsymbol f_X | \boldsymbol f_Z, \mathbf X, \mathbf Z)\, q(\boldsymbol f_Z | \boldsymbol m, \mathbf S)\, \mathrm{d}\boldsymbol f_Z = \mathcal N (\boldsymbol f_X | {\color{orangered}\tilde{\boldsymbol \mu}}, {\color{royalblue}\tilde{\boldsymbol \Sigma}})
$$
其中：
$$
\begin{aligned}
	{\color{orangered}\tilde{\mu}_i} &= m(\boldsymbol x_i) + \boldsymbol {\color{green}\alpha(\boldsymbol x_i)}^\top (\boldsymbol m - m(\mathbf Z)) \\
	
	{\color{royalblue}\tilde{\Sigma}_{ij}} &= \mathcal K(\boldsymbol x_i, \boldsymbol x_j) - \boldsymbol {\color{green}\alpha(\boldsymbol x_i)}^\top (\mathcal K(\mathbf Z, \mathbf Z) - \mathbf S) \boldsymbol {\color{green}\alpha(\boldsymbol x_j)} \\
	
	{\color{green}\alpha(\boldsymbol x_i)} &= \mathcal K(\mathbf Z, \mathbf Z)^{-1} \mathcal K(\mathbf Z, \boldsymbol x_i)
\end{aligned}
$$

> 不能简单地从 $q(\boldsymbol f_X, \boldsymbol f_Z)$ 中读取出 $q(\boldsymbol f_X)$ ，因为我们没有将 $q(\boldsymbol f_X, \boldsymbol f_Z)$​ 定义为多元高斯分布。
>
> 这一步的详细推导：（待验证）
> $$
> \begin{aligned}
> 	q(\boldsymbol f_X) &= 
> \end{aligned}
> $$

因此单点的边际为 $q(f_i) = \mathcal N(f_i | {\color{orangered}\tilde{\mu}_i}, {\color{royalblue}\tilde{\Sigma}_{ii}})$ ，可以用它来计算对数似然期望：
$$
{\color{blue}\textbf{likelihood term}}: \mathbb E_{q(\boldsymbol f_X)} [\log p(\boldsymbol y | \boldsymbol f_X)] = \sum_{i=1}^{N} \mathbb E_{q(f_i)} [\log p(y_i | f_i)]
$$
接下来讨论如何计算这个期望。



---

#### Gaussian likelihood

如果有一个**高斯观测模型（高斯似然）**，我们可以计算**封闭式的对数似然期望**。特别地，如果我们假设 $m(\boldsymbol x) = \boldsymbol 0$ ，可以有：
$$
\mathbb E_{q(f_i)} [\log \mathcal N(y_i | f_i, \beta^{-1})] = \log \mathcal N (y_i | \boldsymbol k_i^\top \mathbf K_{ZZ}^{-1} \boldsymbol m, \beta^{-1}) - \frac{1}{2}\beta \tilde{k}_{ii} - \frac{1}{2}\tr(\mathbf S \mathbf \Lambda_i)
$$
其中，$\tilde{k}_{ii} = k_{ii} - \boldsymbol k_i^\top \mathbf K_{ZZ}^{-1} \boldsymbol k_i$ ，$\boldsymbol k_i$ 是 $\mathbf K_{ZX}$ 的第 $i$ 列，$\mathbf \Lambda_i = \beta \mathbf K_{ZZ}^{-1} \boldsymbol k_i \boldsymbol k_i^\top \mathbf K_{ZZ}^{-1}$ 。所以整体的ELBO形式为：
$$
\begin{aligned}
	\mathcal L(q) ({\color{blue}\mathrm{ELBO}}) &= \log \mathcal N(\boldsymbol y | \mathbf K_{XZ} \mathbf K_{ZZ}^{-1} \boldsymbol m, \beta^{-1} \mathbf I_N) - \frac{1}{2}\beta\tr(\mathbf K_{XZ} \mathbf K_{ZZ}^{-1} \mathbf S \mathbf K_{ZZ}^{-1} \mathbf K_{ZX}) \\
	&- \frac{1}{2} \beta \tr(\mathbf K_{XX} - \mathbf Q_{XX}) - D_{\mathbb{KL}}(q(\boldsymbol f_Z) \parallel p(\boldsymbol f_Z))
\end{aligned}
$$
其中，$\mathbf Q_{XX} = \mathbf K_{ZX}^\top \mathbf K_{ZZ}^{-1} \mathbf K_{ZX}$ 。

> **IMPORTANT**
>
> 为了计算ELBO的梯度，我们利用以下结果：
> $$
> \begin{aligned}
> 	\frac{\partial}{\partial \mu} \mathbb E_{\mathcal N(x | \mu, \sigma^2)} [h(x)] &= \mathbb E_{\mathcal N(x | \mu, \sigma^2)} \left[\frac{\partial}{\partial x} h(x)\right] \\
> 	\frac{\partial}{\partial \sigma^2} \mathbb E_{\mathcal N(x | \mu, \sigma^2)} [h(x)] &= \frac{1}{2} \mathbb E_{\mathcal N(x | \mu, \sigma^2)} \left[\frac{\partial^2}{\partial x^2} h(x)\right]
> \end{aligned}
> $$

然后我们用 $\log p(y_i | f_i)$ 替换 $h(x)$ ，可以得到：
$$
\begin{aligned}
	\nabla_{\boldsymbol m} \mathcal L(q) &= \beta \mathbf K_{ZZ}^{-1} \mathbf K_{ZX} \boldsymbol y - \mathbf \Lambda\boldsymbol m \\
	\nabla_{\mathbf S} \mathcal L(q) &= \frac{1}{2}\mathbf S^{-1} - \frac{1}{2} \mathbf \Lambda
\end{aligned}
$$
令梯度为0，可以解得参数最优解：
$$
\begin{aligned}
	{\color{red}\mathbf S} &= \mathbf \Lambda^{-1} \\
	{\color{red}\mathbf \Lambda} &= \beta \mathbf K_{ZZ}^{-1} \mathbf K_{ZX} \mathbf K_{XZ} \mathbf K_{ZZ}^{-1} + \mathbf K_{ZZ}^{-1} \\
	{\color{red}\boldsymbol m} &= \beta \mathbf \Lambda^{-1} \mathbf K_{ZZ}^{-1} \mathbf K_{ZX} \boldsymbol y
\end{aligned}
$$
这也可以称为 **稀疏高斯过程回归 （sparse GP regression， SGPR）**。





---

#### Non-Gaussian likelihood

在非高斯似然的情况下，可以通过定义 $h(f_i) = \log p(y_i | f_i)$ 然后使用 MCMC 方法来近似 ELL 的梯度。对于二元分类器，可以使用下表中的结果来计算内部的 $\frac{\partial}{\partial f_i} h(f_i)$ 和 $\frac{\partial^2}{\partial f_i^2} h(f_i)$ 项。因为 $q(\boldsymbol f_X)$ 是高斯的，或者可以使用数值积分的技术，如**高斯正交 (Gaussian quadrature)**。



![image-20240430214718061](C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240430214718061.png)



#### Minibatch SVI

小批量的随机变分推断方法，目标函数改变成：
$$
\begin{aligned}
	\mathcal L(q)({\color{blue}\mathrm{ELBO}}) = \left[\frac{N}{B} \sum_{b=1}^B \frac{1}{|\mathcal B_b|} \sum_{i \in \mathcal B_b} \mathbb E_{q(f_i)} \big[\log p(y_i | f_i)\big] \right] - D_{\mathbb{KL}}\big(q(\boldsymbol f_Z) \parallel p(\boldsymbol f_Z)\big)
\end{aligned}
$$
其中 $\mathcal B_b$ 是第 $b$ 个 batch，$B$ 是 batch 的数量。由于 GP 模型（具有高斯似然）属于指数族，因此我们可以有效地计算上式关于 $q(\boldsymbol f_Z)$ 的标准参数的 **自然梯度 (natural gradient)** ，这比遵循标准梯度收敛得快得多。

> 关于参数：
>
> $\boldsymbol m, \mathbf S$ 来自 $q(\boldsymbol f_Z)$ ，是**变分参数**；
>
> 其他是**模型参数**



### Parameter Learning

从上一节中可知，ELBO的解析表达式是一个以所有模型参数作为参数的函数，我们使用梯度下降法进行参数学习。

最大化目标函数 $\mathcal L(q)({\color{blue}\mathrm{ELBO}})$ ，ELBO在一个公式中结合了两个优化：

- 找出核参数 $\ell$ 和 $\sigma^2$ ，噪声方差 $\eta^2$ (高斯似然) 和诱导位置 $\mathbf Z$ ，使真实后验能够很好地解释训练数据。
- 求变分参数 $\boldsymbol m$ 和 $\mathbf S$ ，使变分分布更接近真实精确后验 $q(\boldsymbol f_X, \boldsymbol f_Z) \rightarrow p(\boldsymbol f_X, \boldsymbol f_Z | \boldsymbol y)$ 。





### Making Predictions

version-1 ==Uncorrected==

贝叶斯模型使用后验进行预测，在给定测试点 $\mathbf X_*$ ，可以推导出预测分布 $p(\boldsymbol f_* | \boldsymbol y)$ 。思考一下SVGP是如何进行预测的，可以归结为测试随机变量 $\boldsymbol f_*$ 与诱导随机变量 $\boldsymbol f_Z$ 的关系。已知完整的SVGP先验为联合分布：
$$
\begin{aligned}
	p(\boldsymbol f_*, \boldsymbol f_X, \boldsymbol f_Z) = \mathcal N \left(
	\left.\begin{bmatrix}
		\boldsymbol f_* \\
		\boldsymbol f_X \\
		\boldsymbol f_Z
	\end{bmatrix} \right |
	\begin{bmatrix}
		\boldsymbol 0 \\
		\boldsymbol 0 \\
		\boldsymbol 0
	\end{bmatrix}, 
	\begin{bmatrix}
		\mathbf K_{**} & \mathbf K_{*X} & \mathbf K_{*Z} \\
		\mathbf K_{*X}^{\top} & \mathbf K_{XX} & \mathbf K_{XZ} \\
		\mathbf K_{*Z}^{\top} & \mathbf K_{XZ}^{\top} & \mathbf K_{ZZ}
	\end{bmatrix}
	\right)
\end{aligned}
$$
SVGP使用相同的核函数 $\mathcal K$ 将每对随机变量关联在一起，参数学习为核参数 $\boldsymbol \theta$ 找到唯一值，这意味着模型将使用相同的相关结构（多元高斯条件规则）来：

- 从诱导点 $\mathbf Z$ 的诱导随机变量 $\boldsymbol f_Z$ 解释或总结训练点 $\mathbf X$ 的训练随机变量 $\boldsymbol f_X$ ；
- 从诱导点 $\mathbf Z$ 的诱导随机变量 $\boldsymbol f_Z$ 预测测试点 $\mathbf X_*$ 的测试随机变量 $\boldsymbol f_*$ ；

我们假设测试数据与训练数据来自相同的生成过程，通过梯度下降找到的最优参数值的核函数能够使用诱导变量 (在高的边际似然 $p(\boldsymbol y)$ 意义上，或者等效地，在高的 ELBO 意义上) 来总结训练数据，那么具有相同参数设置的核函数应该允许我们在测试位置对 $\boldsymbol f_*$ 做出合理的预测。

t测。

**推导预测分布 $p(\boldsymbol f_* | \boldsymbol y)$：**
$$
\begin{aligned}
	p(\boldsymbol f_* | \boldsymbol y) &= \iint p(\boldsymbol f_*, \boldsymbol f_X, \boldsymbol f_Z | \boldsymbol y)\, \mathrm{d}\boldsymbol f_X\, \mathrm{d} \boldsymbol f_Z \\
	&= \iint p(\boldsymbol f_*, \boldsymbol f_X | \boldsymbol f_Z)\, q(\boldsymbol f_Z)\, \mathrm{d}\boldsymbol f_X\, \mathrm{d}\boldsymbol f_Z \\
	&= \int \left(\int p(\boldsymbol f_*, \boldsymbol f_X | \boldsymbol f_Z)\, \mathrm{d}\boldsymbol f_X\right) q(\boldsymbol f_Z)\, \mathrm{d}\boldsymbol f_Z \\
	&= {\color{orangered}\int p(\boldsymbol f_* | \boldsymbol f_Z)\, q(\boldsymbol f_Z)\, \mathrm{d}\boldsymbol f_Z}
\end{aligned}
$$
从上式可以看出，预测分布只依赖于诱导变量 $\boldsymbol f_Z$ ，而不依赖于训练位置的随机变量 $\boldsymbol f_X$ ，这意味着所有来自训练数据的信息都被吸收到分布 $q(\boldsymbol f_Z | \boldsymbol m, \mathbf S)$ ，以及通过梯度下降法得到的其他模型参数 $\boldsymbol \theta$ 中。经过参数学习后，模型不再需要训练数据，表明诱导变量真实地总结了训练数据。这与高斯过程回归模型不同，高斯过程回归模型需要训练数据进行预测。

所以预测分布 $\color{red}p(\boldsymbol f_* | \boldsymbol y) = \int p(\boldsymbol f_* | \boldsymbol f_Z)\, q(\boldsymbol f_Z)\, \mathrm{d}\boldsymbol f_Z$ 。已知 $q(\boldsymbol f_Z) = \mathcal N(\boldsymbol f_Z | \boldsymbol m, \mathbf S)$ ，对稀疏先验 $p(\boldsymbol f_*, \boldsymbol f_Z)$ 应用多元高斯条件规则推导出 $p(\boldsymbol f_* | \boldsymbol f_Z)$ ：
$$
\begin{aligned}
	p(\boldsymbol f_*, \boldsymbol f_Z) &= \mathcal N\left(
    \left.\begin{bmatrix}
    	\boldsymbol f_* \\
    	\boldsymbol f_Z
    \end{bmatrix} \right |
    \begin{bmatrix}
    	\boldsymbol 0 \\
    	\boldsymbol 0
    \end{bmatrix},
    \begin{bmatrix}
    	\mathbf K_{**} & \mathbf K_{*Z} \\
    	\mathbf K_{*Z}^{\top} & \mathbf K_{ZZ}
    \end{bmatrix}
    \right) \\
    p(\boldsymbol f_* | \boldsymbol f_Z) &= \mathcal N(\boldsymbol f_* | \boldsymbol \mu_{*|Z}, \boldsymbol \Sigma_{*|Z}) \\
    &= \mathcal N(\boldsymbol f_* | \underbrace{\boldsymbol 0 + \mathbf K_{*Z} \mathbf K_{ZZ}^{-1}(\boldsymbol f_Z - \boldsymbol 0)}_{\boldsymbol \mu_{*|Z}}, \ 
    \underbrace{\mathbf K_{**} - \mathbf K_{*Z} \mathbf K_{ZZ}^{-1} \mathbf K_{*Z}^{\top}}_{\boldsymbol \Sigma_{*|Z}})
\end{aligned}
$$

> **IMPORTANT**
>
> **条件高斯函数标准规则**（再来亿遍，不厌其烦 (๑＞ڡ＜)☆）
>
> 假设 $\boldsymbol x = (\boldsymbol x_1, \boldsymbol x_2)$ 为联合高斯分布：
> $$
> \begin{aligned}
> 	\boldsymbol \mu = 
> 	\begin{bmatrix}
> 		\boldsymbol \mu_1 \\
> 		\boldsymbol \mu_2
> 	\end{bmatrix} \quad \boldsymbol \Sigma = 
> 	\begin{bmatrix}
> 		\boldsymbol \Sigma_{11} & \boldsymbol \Sigma_{12} \\
> 		\boldsymbol \Sigma_{21} & \boldsymbol \Sigma_{22}
> 	\end{bmatrix} \quad \boldsymbol \Lambda = \boldsymbol \Sigma^{-1} = 
> 	\begin{bmatrix}
> 		\boldsymbol \Lambda_{11} & \boldsymbol \Lambda_{12} \\
> 		\boldsymbol \Lambda_{21} & \boldsymbol \Lambda_{22}
> 	\end{bmatrix}
> \end{aligned}
> $$
> 边缘分布：
> $$
> \begin{aligned}
> 	p(\boldsymbol x_1) &= \int \mathcal N(\boldsymbol x | \boldsymbol \mu, \boldsymbol \Sigma)\, \mathrm{d}\boldsymbol x_2 \triangleq \mathcal N(\boldsymbol x_1 | \boldsymbol \mu_1, \boldsymbol \Sigma_1) \\
> 	p(\boldsymbol x_2) &= \int \mathcal N(\boldsymbol x | \boldsymbol \mu, \boldsymbol \Sigma)\, \mathrm{d}\boldsymbol x_1 \triangleq \mathcal N(\boldsymbol x_2 | \boldsymbol \mu_2, \boldsymbol \Sigma_2)
> \end{aligned}
> $$
> 条件分布：
> $$
> \begin{aligned}
> 	p(\boldsymbol x_1 | \boldsymbol x_2) &= \mathcal N(\boldsymbol x_1 | {\color{royalblue}\boldsymbol \mu_{1|2}}, {\color{red}\boldsymbol \Sigma_{1|2}}) = 
> 	\mathcal N(\boldsymbol x_1 | \underbrace{\color{royalblue}\boldsymbol \mu_1 + \boldsymbol \Sigma_{12} \boldsymbol \Sigma_{22}^{-1}(\boldsymbol x_2 - \boldsymbol \mu_2)}_{\color{royalblue}\boldsymbol \mu_{1|2}},\  
> 	\underbrace{\color{red}\boldsymbol \Sigma_{11} - \boldsymbol \Sigma_{12} \boldsymbol \Sigma_{22}^{-1} \boldsymbol \Sigma_{21}}_{\color{red}\boldsymbol \Sigma_{1|2}}) \\
> 	p(\boldsymbol x_2 | \boldsymbol x_1) &= \mathcal N(\boldsymbol x_2 | {\color{royalblue}\boldsymbol \mu_{2|1}}, {\color{red}\boldsymbol \Sigma_{2|1}}) = 
> 	\mathcal N(\boldsymbol x_2 | \underbrace{\color{royalblue}\boldsymbol \mu_2 + \boldsymbol \Sigma_{21} \boldsymbol \Sigma_{11}^{-1}(\boldsymbol x_1 - \boldsymbol \mu_1)}_{\color{royalblue}\boldsymbol \mu_{2|1}},\  
> 	\underbrace{\color{red}\boldsymbol \Sigma_{22} - \boldsymbol \Sigma_{21} \boldsymbol \Sigma_{11}^{-1} \boldsymbol \Sigma_{12}}_{\color{red}\boldsymbol \Sigma_{2|1}})
> \end{aligned}
> $$





---



## Reparameterization of $\boldsymbol f_Z$

已知对于稀疏先验的原始定义和诱导变量的变分分布为：
$$
\begin{aligned}
	p(\boldsymbol f_Z) &= \mathcal N(\boldsymbol f_Z | \boldsymbol 0, \mathbf K_{ZZ}) \quad &\text{sparse prior}\\
	q(\boldsymbol f_Z) &= \mathcal N(\boldsymbol f_Z | \boldsymbol m, \mathbf S) \quad &\text{variational distribution}
\end{aligned}
$$
先验分布和变分分布完全脱离了模型的观点，优化器可以改变模型参数 $\boldsymbol \theta$ 而独立于变分参数 $\boldsymbol m, \mathbf S$​ ，这种自由度为优化器提供了不同的任务。



**重参数化过程：**

引入一个新的随机变量 $\boldsymbol v$ (和 $\boldsymbol f_Z$ 有相同长度)，于是重新定义 $\boldsymbol f_Z$ 为 $\boldsymbol v$ 的表达式：
$$
\begin{aligned}
	\boldsymbol f_Z = \mathbf L \boldsymbol v
\end{aligned}
$$
其中，$\mathbf L$ 是由 $\mathbf K_{ZZ}$ 的 **Cholesky 分解** 得到的 **下三角矩阵** ，即 $\mathbf L \mathbf L^{\top} = \mathbf K_{ZZ}$ 。

1. 在稀疏先验中，令 $\boldsymbol v$ 来自标准的多元高斯分布 $\boldsymbol v \sim \mathcal N(\boldsymbol 0, \boldsymbol 1)$ 。

$$
\begin{aligned}
	p(\boldsymbol f_Z) &= \mathcal N(\boldsymbol f_Z | \mathbf L \cdot \boldsymbol 0,\ \mathbf L \cdot \boldsymbol 1 \cdot\mathbf L^{\top}) \\
	&= \mathcal N(\boldsymbol f_Z | \boldsymbol 0,\ \mathbf K_{ZZ})
\end{aligned}
$$

2. 对于变分分布，设 $\boldsymbol v \sim \mathcal N(\boldsymbol \mu_{v}, \boldsymbol \Sigma_{v})$ 。由于 $\boldsymbol f_Z = \mathbf L \boldsymbol v$，于是：
    $$
    \begin{aligned}
    	q(\boldsymbol f_Z) = \mathcal N(\boldsymbol f_Z | \mathbf L \boldsymbol \mu_{v},\ \mathbf L \boldsymbol \Sigma_{v} \mathbf L^{\top})
    \end{aligned}
    $$
    新的变分分布与原来的变分分布 $q(\boldsymbol f_Z) = \mathcal N(\boldsymbol f_Z | \boldsymbol m, \mathbf S)$​ 的区别是多元高斯分布的**参数化方式不同**：

    - $q(\boldsymbol f_Z) = \mathcal N(\boldsymbol f_Z | \boldsymbol m, \mathbf S)$ 由 $\boldsymbol m, \mathbf S$ 参数化；
    - $q(\boldsymbol f_Z) = \mathcal N(\boldsymbol f_Z | \mathbf L \boldsymbol \mu_{v},\ \mathbf L \boldsymbol \Sigma_{v} \mathbf L^{\top})$ 由 $\mathbf L$ (最终由 $\boldsymbol \theta$ ) 和 $\boldsymbol \mu_{v}, \boldsymbol \Sigma_{v}$ 参数化。



**重参数化的动机**

首先是显而易见的一点，将 $\boldsymbol f_Z$ 定义为 $\mathbf L \boldsymbol v$ ，将模型参数 $\boldsymbol \theta$ 和变分参数 $\boldsymbol \mu_{v}, \boldsymbol \Sigma_{v}$ 连接在一起形成新的变分分布 $q(\boldsymbol f_Z) = \mathcal N(\boldsymbol f_Z | \mathbf L \boldsymbol \mu_{v}, \mathbf L \boldsymbol \Sigma_{v} \mathbf L^{\top})$ 。在 [参数优化](###Parameter Learning) 中有提到 ELBO 将两个优化目标结合在一个函数中，而新的参数化将这两个目标连接在一起——每次优化器更新内核参数 $\boldsymbol \theta$ 时也会改变 $\mathbf L$ ，从而引起变分分布的变化。

> **NOTE**
>
> $\boldsymbol f_Z = \mathbf L \boldsymbol v$ 参数化引入的先验和变分分布的联合运动只发生在一个方向上。移动先验 (通过改变核参数 $\boldsymbol \theta$ 的值)，然后变分分布移动。然而，如果移动变分分布 (通过改变 $\boldsymbol \mu_{v}$ 和 $\boldsymbol \Sigma_{v}$ 的值)，先验不会移动。



### ELBO

#### The KL term

${\color{blue}\textbf{KL term}}: D_{\mathbb{KL}} (q(\boldsymbol f_Z) \parallel p(\boldsymbol f_Z))$​

1. 

$$
\begin{aligned}
	q(\boldsymbol f_Z) &= \mathcal N(\boldsymbol f_Z | \boldsymbol 0,\ \mathbf K_{ZZ}) \\
	&= \mathcal N(\mathbf L \boldsymbol v | \mathbf L \boldsymbol \mu_{v},\ \mathbf L \boldsymbol \Sigma_{v} \mathbf L^{\top}) \\
	&= \frac{1}{(2\pi)^{N_Z / 2} [\det(\mathbf L \boldsymbol \Sigma_{v} \mathbf L^{\top})]^{1/2}} \exp \left\{-\frac{1}{2}(\mathbf L \boldsymbol v - \mathbf L \boldsymbol \mu_{v})^{\top} (\mathbf L \boldsymbol \Sigma_{v} \mathbf L^{\top})^{-1} (\mathbf L \boldsymbol v - \mathbf L \boldsymbol \mu_{v}) \right\}
\end{aligned}
$$

化简其中的部分：
$$
\begin{aligned}
	\det(\mathbf L \boldsymbol \Sigma_{v} \mathbf L^{\top}) &= \det(\mathbf L)^2 \det(\boldsymbol \Sigma_{v}) \\
	(\mathbf L \boldsymbol v - \mathbf L \boldsymbol \mu_{v})^{\top} (\mathbf L \boldsymbol \Sigma_{v} \mathbf L^{\top})^{-1} (\mathbf L \boldsymbol v - \mathbf L \boldsymbol \mu_{v}) &= (\boldsymbol v - \boldsymbol \mu_{v})^{\top} \boldsymbol \Sigma_{v}^{-1} (\boldsymbol v - \boldsymbol \mu_{v})
\end{aligned}
$$
最后化简为：
$$
\begin{aligned}
	q(\boldsymbol f_Z) &= \frac{1}{\det(\mathbf L) (2\pi)^{N_Z / 2} [\det(\boldsymbol \Sigma_{v})]^{1/2}} \exp \left\{-\frac{1}{2} (\boldsymbol v - \boldsymbol \mu_{v})^{\top} \boldsymbol \Sigma_{v}^{-1} (\boldsymbol v - \boldsymbol \mu_{v}) \right\} \\
    &= {\color{royalblue}\frac{1}{\det(\mathbf L)} \mathcal N(\boldsymbol v | \boldsymbol \mu_{v}, \boldsymbol \Sigma_{v})}
\end{aligned}
$$
可以看到 $q(\boldsymbol f_Z)$ 最后表示为 $\mathcal N(\boldsymbol v | \boldsymbol \mu_{v}, \boldsymbol \Sigma_{v})$ 按照 $\frac{1}{\det(\mathbf L)}$​ 的倍数缩放。



2. 

$$
\begin{aligned}
	p(\boldsymbol f_Z) &= \mathcal N(\boldsymbol f_Z | \boldsymbol 0, \mathbf K_{ZZ}) \\
	&= \mathcal N(\mathbf L \boldsymbol v | \mathbf L \cdot \boldsymbol 0, \mathbf L \cdot \boldsymbol 1 \cdot \mathbf L^{\top}) \\
	&= {\color{red}\frac{1}{\det(\mathbf L)} \mathcal N(\boldsymbol v | \boldsymbol 0, \boldsymbol 1)}
\end{aligned}
$$

于是，完整的KL项可以表示为：
$$
\begin{aligned}
	D_{\mathbb{KL}} (q(\boldsymbol f_Z) \parallel p(\boldsymbol f_Z)) &= \int q(\boldsymbol f_Z) \log \frac{q(\boldsymbol f_Z)}{p(\boldsymbol f_Z)}\, \mathrm{d}\boldsymbol f_Z \\
	&= \int {\color{royalblue}\frac{1}{\det(\mathbf L)} \mathcal N(\boldsymbol v | \boldsymbol \mu_{v}, \boldsymbol \Sigma_{v})} \log \left[\frac{{\color{royalblue}\frac{1}{\det(\mathbf L)} \mathcal N(\boldsymbol v | \boldsymbol \mu_{v}, \boldsymbol \Sigma_{v})}}{{\color{red}\frac{1}{\det(\mathbf L)} \mathcal N(\boldsymbol v | \boldsymbol 0, \boldsymbol 1)}}\right]\, \mathrm{d}\boldsymbol f_Z \\
	&= \int {\color{royalblue}\frac{1}{\det(\mathbf L)} \mathcal N(\boldsymbol v | \boldsymbol \mu_{v}, \boldsymbol \Sigma_{v})} \log \left[\frac{{\color{royalblue}\mathcal N(\boldsymbol v | \boldsymbol \mu_{v}, \boldsymbol \Sigma_{v})}}{{\color{red}\mathcal N(\boldsymbol v | \boldsymbol 0, \boldsymbol 1)}} \right]\, \mathrm{d}({\color{green}\mathbf L \boldsymbol v}) \\
	&= \int {\color{green}\frac{1}{\det(\mathbf L)}} {\color{royalblue}\mathcal N(\boldsymbol v | \boldsymbol \mu_{v}, \boldsymbol \Sigma_{v})} \log \left[\frac{{\color{royalblue}\mathcal N(\boldsymbol v | \boldsymbol \mu_{v}, \boldsymbol \Sigma_{v})}}{{\color{red}\mathcal N(\boldsymbol v | \boldsymbol 0, \boldsymbol 1)}} \right] {\color{green}\det(\mathbf L)}\, \mathrm{d}\boldsymbol v \\
	&= \int {\color{royalblue}\mathcal N(\boldsymbol v | \boldsymbol \mu_{v}, \boldsymbol \Sigma_{v})} \log \left[\frac{{\color{royalblue}\mathcal N(\boldsymbol v | \boldsymbol \mu_{v}, \boldsymbol \Sigma_{v})}}{{\color{red}\mathcal N(\boldsymbol v | \boldsymbol 0, \boldsymbol 1)}} \right]\, \mathrm{d}\boldsymbol v \\
	&= D_{\mathbb{KL}} \big({\color{royalblue}\mathcal N(\boldsymbol v | \boldsymbol \mu_{v}, \boldsymbol \Sigma_{v})} \parallel {\color{red}\mathcal N(\boldsymbol v | \boldsymbol 0, \boldsymbol 1)}\big)
\end{aligned}
$$

> **IMPORTANT**
> $$
> \begin{aligned}
> 	D_{\mathbb{KL}} \big(\mathcal N(\mathbf A \boldsymbol \mu_0, \mathbf A \boldsymbol \Sigma_0 \mathbf A^{\top}) \parallel \mathcal N(\mathbf A \boldsymbol \mu_1, \mathbf A \boldsymbol \Sigma_1 \mathbf A^{\top}) \big) = 
> 	D_{\mathbb{KL}} \big( \mathcal N(\boldsymbol \mu_0, \boldsymbol \Sigma_0) \parallel \mathcal N(\boldsymbol \mu_1, \boldsymbol \Sigma_1 )\big)
> \end{aligned}
> $$
> 在上式中我们设置 $\boldsymbol \mu_0 = \boldsymbol \mu_{v}, \boldsymbol \Sigma_0 = \boldsymbol \Sigma_{v}, \boldsymbol \mu_1 = \boldsymbol 0, \boldsymbol \Sigma_1 = \boldsymbol 1, \mathbf A = \mathbf L$ 。



#### The likelihood term (ELL)

${\color{blue}\textbf{likelihood term}}: \mathbb E_{q(\boldsymbol f_X)} [\log p(\boldsymbol y | \boldsymbol f_X)] = \int q(\boldsymbol f_X) \log p(\boldsymbol y | \boldsymbol f_X)\, \mathrm{d}\boldsymbol f_X$




$$

$$




## Optimize the Inducing Locations

已知优化器改变核参数来使模型能够更好地解释数据，这里将注意力放在这个新的模型参数上：诱导点 $\mathbf Z$​ 。为什么优化器将更多的诱导位置分配给训练数据变化更快的区域，而将更少的诱导位置分配给训练数据更平滑的区域。

优化器的工作是最大化 ELBO ，它是边际似然 $p(\boldsymbol y)$ 的替代，衡量诱导变量 $\boldsymbol f_Z$ 对于训练随机变量 $\boldsymbol f_X$ 的解释程度。如前所述，模型将[多元高斯条件规则应用在 SVGP 上](###Conditional prior)，先推断出条件概率 $p(\boldsymbol f_X | \boldsymbol f_Z)$ 来使用 $\boldsymbol f_Z$ 解释 $\boldsymbol f_X$ ：
$$
\begin{aligned}
	p(\boldsymbol f_X | \boldsymbol f_Z) = \mathcal N \big(\boldsymbol f_X | \mathbf K_{XZ} \mathbf K_{ZZ}^{-1} (\boldsymbol f_Z - \boldsymbol 0), \mathbf K_{XX} - \mathbf K_{XZ} \mathbf K_{ZZ}^{-1} \mathbf K_{XZ}^{\top}\big)
\end{aligned}
$$
优化器为核参数 $\boldsymbol \theta$ 选择一个单一值，代表所有训练数据点之间的折衷，表示所有训练数据的平均水平，两个随机变量需要有多接近才能有足够大的相关性。在这个单一的参数值下，模型解释训练数据变化更快的区域的唯一方法，是在这些区域中有更多的诱导变量。这是因为只有附近的诱导变量才能更有效地参与加权和公式来解释那些快速变化的训练数据点。即只有与训练数据点足够接近的诱导变量才具有足够大的权重 (由上式中的 $\mathbf K_{XZ}$ 分量定义) 来解释该训练数据点。这就是为什么优化器在训练数据变化更快的区域分配更多的诱导点。

---



## Reference

[1] [Scalable Variational Gaussian Process Classification](./reference/Scalable Variational Gaussian Process Classification.pdf)

[2] [稀疏高斯过程及其变分推断. 西山晴雪的知识笔记]([🔥 稀疏高斯过程及其变分推断 | 西山晴雪的知识笔记 (xishansnow.github.io)](https://xishansnow.github.io/posts/1aa0965f.html))

[3] [A Tutorial on Sparse Gaussian Processes and Variational Inference](./reference/A Tutorial on Sparse Gaussian Processes and Variational Inference.pdf)

[4] [Scalable Variational Gaussian Process Classification](./reference/Scalable Variational Gaussian Process Classification.pdf)
