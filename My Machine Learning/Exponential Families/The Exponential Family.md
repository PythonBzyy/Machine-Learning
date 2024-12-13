## The Exponential Family

指数族（The Exponential Family）是一类分布，包括高斯分布，二项分布，泊松分布，Beta分布，Dirichlet分布，Gamma分布等一系列分布。

<img src="figures\exponential families.png" alt="exponential families" style="zoom:80%;" />

指数族分布有以下特点：

- **最大熵**：指数族是唯一的一组分布，它具有最大的熵（因此产生最少的假设集合）。对于经验分布利用最大熵原理推导出的分布就是指数族分布
- 指数族是 **GLMs（Generalized linear models，广义线性模型）** 的核心
- 指数族是**变分推断**的核心



### Definition

指数族分布可以写成统一的形式，
$$
\begin{aligned}
	p(\boldsymbol{x}|\boldsymbol{\eta})
	&\triangleq\frac{1}{Z(\boldsymbol{\eta})}h(\boldsymbol{x})\exp[\boldsymbol{\eta}^{\top}\mathcal{T}(\boldsymbol{x})] \\
	&=h(\boldsymbol{x})\exp[\boldsymbol{\eta}^{\top}\mathcal{T}(\boldsymbol{x})-A(\boldsymbol{\eta})]
\end{aligned}
$$

- $\boldsymbol \eta \in \mathbb R^{K}$ ：自然参数 / 规范参数（向量）
- $h(\boldsymbol x)$ ：缩放常数 / 基本测度（base measure），通常为1
- $Z(\boldsymbol \eta)$​ ：配分函数（partition function）的归一化常数
- $A(\boldsymbol \eta) = \log Z(\boldsymbol \eta)$​ ：对数配分函数（log partition function）（规范化因子），凸函数
- $\mathcal T(\boldsymbol x) \in \mathbb R^K$ ：充分统计量，包含样本集合的所有信息