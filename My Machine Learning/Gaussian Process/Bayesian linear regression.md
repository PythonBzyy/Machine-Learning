# Bayesian Linear Regression



> **Look at the conclusion first**
>
> Inference：求参数 $\boldsymbol w$ 的后验 $p(\boldsymbol w | \mathcal D)$ :
> $$
> p(\boldsymbol w | \mathcal D) = \mathcal N(\boldsymbol w | \hat{\boldsymbol w}, \hat{\boldsymbol \Sigma}) \\
> 
> \hat{\boldsymbol w} = \left(\sigma^2 \mathbf I_N \boldsymbol y^{\top} \mathbf X + \breve{\boldsymbol \Sigma}^{-1} \breve{\boldsymbol w} \right) \hat{\boldsymbol \Sigma} \\
> 
> \hat{\boldsymbol \Sigma} = \left(\sigma^2 \mathbf I_N \mathbf X^{\top} \mathbf X + \breve{\boldsymbol{\Sigma}}^{-1} \right)^{-1}
> $$
>
> Prediction：给定测试样本 $\boldsymbol x_*$ ，求 $y_*$ :
> $$
> p(y_* | \boldsymbol x_*, \mathcal D, \sigma^2) = \mathcal N(y | \hat{\boldsymbol w}^{\top} \boldsymbol x_*, \sigma^2 + \boldsymbol x_*^{\top} \hat{\boldsymbol \Sigma} \boldsymbol x_*)
> $$





已知数据集 $\mathcal D = \{(\boldsymbol x_i, y_i) \}_{i=1}^N$ ，其中 $\boldsymbol x_i \in \mathbb R^d, y_i \in \mathbb R$​ 。
线性回归的模型假设：
$$
\begin{aligned}
	f(\boldsymbol x) &= \boldsymbol w^{\top} \boldsymbol x \\
	y &= f(\boldsymbol x) + \varepsilon = \boldsymbol w^{\top} \boldsymbol x + \varepsilon
\end{aligned}
$$
其中，$\boldsymbol x, y, \varepsilon$ 都是随机变量，$\varepsilon \sim \mathcal N(0, \sigma^2)$​ 。

![1](figures\bayesian linear regression\1.jpg)

这里的方法都是频率派的方法（$\boldsymbol w$ is an unknown constant），即是一个关于 $\boldsymbol w$ 优化问题，点估计。
而在贝叶斯方法中，这个问题不再是点估计问题（$\boldsymbol w$ is a random variable），我们需要估计 $\boldsymbol w$ 的后验 $p(\boldsymbol w | \mathcal D)$ 。



贝叶斯方法包含两个部分，Inference 和 Prediction ：

- Inference：posterior($\boldsymbol w$)，即参数 $\boldsymbol w$ 服从某一分布，而不是未知的常量
- Prediction：$\boldsymbol x_* \rightarrow y_*$



## Inference

我们首先引入高斯先验：
$$
p(\boldsymbol w) = \mathcal N(\boldsymbol w | \breve{\boldsymbol w}, \breve{\boldsymbol \Sigma})
$$
后验：
$$
\begin{aligned}
	p(\boldsymbol w | \mathcal D) &= p(\boldsymbol w | \mathbf X, \boldsymbol y) \\
	&= \frac{p(\boldsymbol w, \boldsymbol y | \mathbf X)}{p(\boldsymbol y | \mathbf X)} \\
	&= \frac{p(\boldsymbol y | \mathbf X, \boldsymbol w)\ p(\boldsymbol w | \mathbf X)}{\int p(\boldsymbol y | \boldsymbol w, \mathbf X)\ p(\boldsymbol w)\ \mathrm{d}\boldsymbol w}
\end{aligned}
$$
分母与参数 $\boldsymbol w$ 无关，$p(\boldsymbol w | \mathbf X) = p(\boldsymbol w)$ ，于是
$$
\begin{aligned}
	p(\boldsymbol w | \mathbf X, \boldsymbol y) &\propto \underbrace{p(\boldsymbol y | \mathbf X, \boldsymbol w)}_{\color{blue}\text{likelihood}} \ \underbrace{p(\boldsymbol w)}_{\color{blue}\text{prior}} \\
	&\propto \mathcal N(\boldsymbol y | \mathbf X \boldsymbol w, \sigma^2 \mathbf I_N) \ \mathcal N(\boldsymbol w | \breve{\boldsymbol w}, \breve{\boldsymbol \Sigma}) \\
	&= \mathcal N(\boldsymbol w | \hat{\boldsymbol w}, \hat{\boldsymbol \Sigma})

\end{aligned}
$$
最后的目的就是求解 $\hat{\boldsymbol w}, \hat{\boldsymbol \Sigma}$​ 。高斯分布取高斯先验的共轭分布依然是高斯分布。高斯分布是**自共轭的**。



### Likelihood

似然函数的求解过程：
$$
\begin{aligned}
	p(\boldsymbol y | \mathbf X, \boldsymbol w) &= \prod_{i=1}^N p(y_i | \boldsymbol x_i, \boldsymbol w) \\
	&= \prod_{i=1}^N \mathcal N(y_i | \boldsymbol w^{\top} \boldsymbol x_i, \sigma^2) \\
	&= \prod_{i=1}^{N} \frac{1}{(2\pi)^{1/2} \sigma}\ \exp\bigg\{-\frac{1}{2\sigma^2}(y_i - \boldsymbol w^{\top} \boldsymbol x_i)^2 \bigg\} \\
	&= \frac{1}{(2\pi)^{N/2} \sigma^N}\ \exp \bigg\{-\frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \boldsymbol w^{\top} \boldsymbol x_i)^2 \bigg\} \\
	&= \frac{1}{(2\pi)^{N/2} \sigma^N}\ \exp \bigg\{-\frac{1}{2\sigma^2} (\boldsymbol y - \mathbf X \boldsymbol w)^{\top}(\boldsymbol y - \mathbf X \boldsymbol w)\bigg\} \\
	&= \frac{1}{(2\pi)^{N/2} \sigma^N}\ \exp \bigg\{-\frac{1}{2}(\boldsymbol y - \mathbf X \boldsymbol w)^{\top} \sigma^{-2}\mathbf I_N (\boldsymbol y - \mathbf X \boldsymbol w) \bigg\} \\
	&= \color{red}{\mathcal N(\boldsymbol y | \mathbf X \boldsymbol w, \sigma^2 \mathbf I_N)}
\end{aligned}
$$


### Conclusion

$$
\begin{aligned}
	p(\boldsymbol w | \mathbf X, \boldsymbol y) &= \mathcal N(\boldsymbol w | \hat{\boldsymbol w}, \hat{\boldsymbol \Sigma}) \\
	&\propto \mathcal N(\boldsymbol y | \mathbf X \boldsymbol w, \sigma^2 \mathbf I_N) \ \mathcal N(\boldsymbol w | \breve{\boldsymbol w}, \breve{\boldsymbol \Sigma}) \\
	&\propto \exp\bigg\{-\frac{1}{2}(\boldsymbol y - \mathbf X \boldsymbol w)^{\top} \sigma^{-2}\mathbf I_N (\boldsymbol y - \mathbf X \boldsymbol w) \bigg\} \times \\ &\qquad \exp \bigg\{-\frac{1}{2} (\boldsymbol w - \breve{\boldsymbol w})^{\top} \breve{\boldsymbol \Sigma}^{-1} (\boldsymbol w - \breve{\boldsymbol w}) \bigg\} \\
	&= \exp\bigg\{-\frac{1}{2} \sigma^2 \mathbf{I}_N (\boldsymbol y - \mathbf X \boldsymbol w)^{\top} (\boldsymbol y - \mathbf X \boldsymbol w) - \frac{1}{2} \breve{\boldsymbol \Sigma}^{-1} (\boldsymbol w - \breve{\boldsymbol w})^{\top} (\boldsymbol w - \breve{\boldsymbol w}) \bigg\} \\
	&= \exp\bigg\{-\frac{1}{2}\sigma^2 \mathbf I_N \left(\boldsymbol y^{\top}\boldsymbol y - 2\boldsymbol y^{\top} \mathbf X \boldsymbol w + \boldsymbol w^{\top} \mathbf X^{\top} \mathbf X \boldsymbol w \right) - \frac{1}{2}\breve{\boldsymbol \Sigma}^{-1} \left(\boldsymbol w^{\top} \boldsymbol w - 2 \breve{\boldsymbol w}^{\top} \boldsymbol w + \breve{\boldsymbol w}^{\top} \breve{\boldsymbol w} \right) \bigg\} \\
	&= \exp \bigg\{-\frac{1}{2}{\color{blue}\boldsymbol w^{\top} \left(\sigma^2 \mathbf I_N \mathbf X^{\top} \mathbf X + \breve{\boldsymbol{\Sigma}}^{-1} \right) \boldsymbol w} + \frac{1}{2} {\color{blue}\left(2 \sigma^2 \mathbf I_N \boldsymbol y^{\top} \mathbf X + 2\breve{\boldsymbol \Sigma}^{-1} \breve{\boldsymbol w} \right)\boldsymbol w} + \text{constant} \bigg\}
\end{aligned}
$$

#### 使用配方法求解 $\mathcal N(\boldsymbol w | \hat{\boldsymbol w}, \hat{\boldsymbol \Sigma})$

> 对于分布 $p(\boldsymbol x) \sim \mathcal N(\boldsymbol \mu, \boldsymbol \Sigma)$ ，他的指数部分的标准写法为：
> $$
> \exp\left\{-\frac{1}{2}(\boldsymbol x - \boldsymbol \mu)^{\top} \boldsymbol \Sigma^{-1}(\boldsymbol x - \boldsymbol \mu) \right\} \\
> = \exp \left\{-\frac{1}{2}(\underbrace{\boldsymbol x^{\top}\boldsymbol \Sigma^{-1}\boldsymbol x}_{\color{green}\text{二次项}} \underbrace{- 2\boldsymbol \mu^{\top}\boldsymbol \Sigma^{-1}\boldsymbol x}_{\color{green}\text{一次项}} + \mathrm{constant})\right\}
> $$



- 二次项
    $$
    \boldsymbol x^{\top} \boldsymbol \Sigma^{-1} \boldsymbol x \longleftrightarrow \boldsymbol w^{\top} \hat{\boldsymbol \Sigma}^{-1} \boldsymbol w = 
    {\color{blue}\boldsymbol w^{\top} \left(\sigma^2 \mathbf I_N \mathbf X^{\top} \mathbf X + \breve{\boldsymbol{\Sigma}}^{-1} \right) \boldsymbol w} \\
    {\color{red}\therefore \quad \hat{\boldsymbol \Sigma} = \left(\sigma^2 \mathbf I_N \mathbf X^{\top} \mathbf X + \breve{\boldsymbol{\Sigma}}^{-1} \right)^{-1}}
    $$

    > 精度矩阵 $\mathbf A = \hat{\boldsymbol \Sigma}^{-1}$ ，对称，$\mathbf A^{\top} = \mathbf A$ 。
    > $$
    > \mathbf A = \sigma^2 \mathbf I_N \mathbf X^{\top} \mathbf X + \breve{\boldsymbol{\Sigma}}
    > $$

    
    

- 一次项
    $$
    \boldsymbol \mu^{\top}\boldsymbol \Sigma^{-1}\boldsymbol x \longleftrightarrow \hat{\boldsymbol w}^{\top} \hat{\boldsymbol \Sigma}^{-1} \boldsymbol w
    = {\color{blue}\left(\sigma^2 \mathbf I_N \boldsymbol y^{\top} \mathbf X + \breve{\boldsymbol \Sigma}^{-1} \breve{\boldsymbol w} \right)\boldsymbol w} \\
    
    {\color{red}\therefore \quad \hat{\boldsymbol w} = \left(\sigma^2 \mathbf I_N \boldsymbol y^{\top} \mathbf X + \breve{\boldsymbol \Sigma}^{-1} \breve{\boldsymbol w} \right) \hat{\boldsymbol \Sigma}}
    $$
    



#### 使用高斯分布的贝叶斯规则*

todo













## Prediction

给定新的样本数据（测试样本） $\boldsymbol x_*$ ，求 $y_*$ 。在 Inference 部分，已获得对于模型参数的不确定性 $p(\boldsymbol w | \mathcal D) \sim \mathcal N(\boldsymbol w | \hat{\boldsymbol w}, \hat{\boldsymbol \Sigma})$ ，则可以推断出：
$$
\begin{aligned}
	f(\boldsymbol x_*) &= \boldsymbol x_*^{\top} \boldsymbol w \\
 	&\sim \mathcal N(\boldsymbol x_*^{\top} \hat{\boldsymbol w}, \boldsymbol x_*^{\top} \hat{\boldsymbol \Sigma} \boldsymbol x_*)
\end{aligned}
$$

$$
\begin{aligned}
	p(y_* | \boldsymbol x_*, \mathcal D, \sigma^2) 
	&= \int \mathcal N(y_* | \boldsymbol x_*^{\top} \hat{\boldsymbol w}, \sigma^2)\ \mathcal N(\boldsymbol w | \hat{\boldsymbol w}, \hat{\boldsymbol \Sigma})\,\mathrm{d}\boldsymbol w \\
	&= {\color{red}\mathcal N(y | \hat{\boldsymbol w}^{\top} \boldsymbol x_*, \sigma^2 + \boldsymbol x_*^{\top} \hat{\boldsymbol \Sigma} \boldsymbol x_*)}
\end{aligned}
$$

可以证明测试点 $\boldsymbol x_*$ 处的后验预测分布也是高斯分布。

> todo